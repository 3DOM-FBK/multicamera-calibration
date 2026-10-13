"""
CCT (Concentric Circular coded Target) detector.

Target geometry (in units of the inner white circle semi-axis R):
  0 .. R     White filled centre
  R .. 2R    Black guard ring
  2R .. 3R   Code ring (14 sectors, white=1, black=0)

Pipeline:
  1. Otsu + adaptive binarisation  ->  contour extraction
  2. Ellipse fit on circular-enough contours; contours of the same blob from
     different binarisations are kept together as one candidate group
  3. Per group hypothesis: threshold-independent dot scale from the radial
     intensity edge, then affine rectification to a canonical circle
  4. Strict ring validation on the binarised rectified patch
  5. Angular profile sampling on the grayscale rectified patch
  6. Decode: codebook correlation when the valid codes are known, otherwise
     phase search with canonical code via minimum cyclic rotation; several
     hypotheses of one group must agree on the ID
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Collection, Sequence

import cv2
import numpy as np

from .refinement import (
    CenterEstimate,
    prepare_refinement_image,
    refine_projected_center,
)

# Stamped into detection caches so stale detections can be recognised.
DETECTOR_VERSION = "cct-detect-2"

Ellipse = tuple[tuple[float, float], tuple[float, float], float]

# ---------------------------------------------------------------------------
# Code utilities
# ---------------------------------------------------------------------------

def _rotate_left(v: int, k: int, n: int) -> int:
    mask = (1 << n) - 1
    return ((v << k) & mask) | ((v & mask) >> (n - k))


def canonical_code(value: int, n_bits: int) -> int:
    best = value
    for k in range(1, n_bits):
        best = min(best, _rotate_left(value, k, n_bits))
    return best


def cyclic_hamming_distance(a: int, b: int, n_bits: int) -> int:
    """Hamming distance between two ring codes, minimised over rotations."""
    return min((a ^ _rotate_left(b, k, n_bits)).bit_count() for k in range(n_bits))


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

@dataclass
class Detection:
    target_id: int
    center: tuple[float, float]
    ellipse: tuple[tuple[float, float], tuple[float, float], float]
    confidence: float
    code_bits: list[int]
    raw_code: int
    pattern: str
    # ``center`` is the best available estimate of the projected physical
    # target centre.  ``ellipse_center`` is retained because the ellipse fit
    # is only a provisional image measurement under perspective/distortion.
    ellipse_center: tuple[float, float] | None = None
    center_correction_px: tuple[float, float] | None = None
    center_method: str = "ellipse"
    center_covariance_px2: tuple[float, float, float, float] | None = None
    # Number of agreeing hypotheses from the candidate group, and the decoder
    # that produced the ID ("generic" phase search or "codebook" correlation).
    votes: int = 1
    decoder: str = "generic"
    # Number of binarisation contours in the candidate group.
    support: int = 1


@dataclass
class DetectionDiagnostics:
    """Stage counts and rejection locations from the production detector."""

    counts: dict[str, int]
    rejected: list[dict[str, object]]


@dataclass
class CandidateGroup:
    """Contours of one blob found by different binarisations.

    ``seed`` is the most circular member, which is the single candidate the
    historical detector kept.  The other members are retained because the
    most circular contour of a small blurred dot is effectively a random pick
    among thresholds, and its scale can be biased by more than the ring
    validation tolerates.
    """

    seed: Ellipse
    members: list[tuple[float, Ellipse]] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Detector
# ---------------------------------------------------------------------------

class CCTDetector:
    def __init__(self, n_bits: int = 14,
                 valid_id_range: tuple[int, int] | None = None,
                 min_target_radius: float = 5.0,
                 max_target_radius: float | None = 150.0,
                 min_target_minor_radius: float = 3.0,
                 codebook: Collection[int] | None = None,
                 max_hypotheses: int = 3,
                 codebook_min_correlation: float = 0.90,
                 codebook_min_margin: float = 0.12,
                 codebook_low_information_min_correlation: float = 0.95,
                 codebook_low_information_min_margin: float = 0.20,
                 codebook_single_vote_min_correlation: float = 0.92):
        self.n_bits = n_bits
        self.valid_id_range = valid_id_range  # (min_id, max_id) inclusive, or None
        self.min_target_radius = float(min_target_radius)
        # Keep the conservative historical default.  ``None`` is treated as
        # the same fixed 150 px cap; an image-relative cap is deliberately not
        # used because it makes arbitrary large scene contours eligible.
        self.max_target_radius = 150.0 if max_target_radius is None else float(max_target_radius)
        # Historically 4 px.  Oblique views foreshorten the centre dot along
        # one axis while the code ring stays decodable; false positives are
        # controlled by the ring validation and decoding gates, not by this.
        self.min_target_minor_radius = float(min_target_minor_radius)
        self.max_hypotheses = max(1, int(max_hypotheses))
        # Codebook acceptance.  A codebook ID that the generic decoder reads
        # identically from the same profile is accepted as before.  Otherwise
        # the correlation evidence must clear a bar that is higher for
        # low-information codes (<= 4 transitions): blocky profiles from card
        # glyphs, sector blobs and target edges correlate well with them.
        # Thresholds were set against projected reference targets: they keep
        # the misidentification rate at the generic decoder's level while
        # recovering most targets the generic decoder rejects.  An ID that
        # only one hypothesis supports needs stronger evidence still.
        self.codebook_min_correlation = float(codebook_min_correlation)
        self.codebook_min_margin = float(codebook_min_margin)
        self.codebook_low_information_min_correlation = float(codebook_low_information_min_correlation)
        self.codebook_low_information_min_margin = float(codebook_low_information_min_margin)
        self.codebook_single_vote_min_correlation = float(codebook_single_vote_min_correlation)
        self._last_decode_corroborated = False
        self.codebook: tuple[int, ...] | None = None
        self._codebook_fft_conj: np.ndarray | None = None
        if codebook is not None:
            self.set_codebook(codebook)

    @property
    def _n_angular_samples(self) -> int:
        return max(360, 60 * self.n_bits)

    def set_codebook(self, codebook: Collection[int] | None) -> None:
        """Enable correlation decoding against the known valid codes.

        Codes are reduced to their canonical rotation because rotations of one
        code are physically the same target.  Codes without at least one white
        and one black sector carry no angular signal and are ignored.
        """
        if codebook is None:
            self.codebook = None
            self._codebook_fft_conj = None
            return
        limit = 1 << self.n_bits
        canonical = sorted({
            canonical_code(int(code), self.n_bits)
            for code in codebook
            if 0 < int(code) < limit - 1
        })
        if not canonical:
            raise ValueError("codebook contains no usable codes")
        n_ang = self._n_angular_samples
        step = n_ang / self.n_bits
        sector = np.minimum((np.arange(n_ang) / step).astype(int), self.n_bits - 1)
        templates = np.empty((len(canonical), n_ang), dtype=np.float64)
        for row, code in enumerate(canonical):
            bits = np.array([(code >> k) & 1 for k in range(self.n_bits)], dtype=np.float64)
            template = bits[sector]
            template -= template.mean()
            templates[row] = template / np.linalg.norm(template)
        self.codebook = tuple(canonical)
        self._codebook_fft_conj = np.conj(np.fft.rfft(templates, axis=1))

    # ---- 1. candidate ellipses ------------------------------------------

    @staticmethod
    def _binarizations(gray: np.ndarray):
        """Yield (label, binary image) for every threshold the detector uses."""
        blurred = cv2.GaussianBlur(gray, (5, 5), 1.2)

        # Otsu with broader offset sweep
        otsu_val, _ = cv2.threshold(blurred, 0, 255,
                                     cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        for off in (-30, -20, -10, 0, 10, 20, 30):
            tv = int(np.clip(otsu_val + off, 30, 230))
            _, bw = cv2.threshold(blurred, tv, 255, cv2.THRESH_BINARY)
            yield f"otsu{off:+d}", bw

        # Adaptive
        for bs in (31, 61, 91):
            for C in (5, 10, 20):
                bw = cv2.adaptiveThreshold(
                    blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                    cv2.THRESH_BINARY, bs, -C,
                )
                yield f"adaptive{bs}/{C}", bw

    def _gate_contour(self, cnt, min_area: float, max_area: float, min_circ: float):
        """Return (rejection reason or None, circularity, ellipse or None)."""
        area = cv2.contourArea(cnt)
        if area < min_area or area > max_area:
            return "area", 0.0, None
        perim = cv2.arcLength(cnt, True)
        if perim < 1:
            return "perimeter", 0.0, None
        circ = 4.0 * math.pi * area / (perim * perim)
        # This is deliberately a hard gate: arbitrary scene contours
        # must not reach the decoder merely because they can be fit by
        # an ellipse.
        if circ < min_circ:
            return "circularity", circ, None
        if len(cnt) < 10:
            return "contour_points", circ, None
        ell = cv2.fitEllipse(cnt)
        (cx, cy), (aw, ah), ang = ell
        major = max(aw, ah) / 2.0
        minor = min(aw, ah) / 2.0
        if (major < self.min_target_radius or minor < self.min_target_minor_radius
                or major > self.max_target_radius):
            return "size", circ, ell
        if minor / (major + 1e-9) < 0.30:
            return "aspect", circ, ell
        return None, circ, ell

    def _find_candidate_groups(self, gray: np.ndarray, min_circ: float = 0.60) -> list[CandidateGroup]:
        """
        Find ellipse candidates from multiple binarisations, grouped by blob.
        """
        H, W = gray.shape[:2]
        min_area = max(30, int(H * W * 1e-6))
        max_area = int(H * W * 0.008)

        scored: list[tuple[float, tuple]] = []
        rejection_counts = {
            "area": 0, "perimeter": 0, "circularity": 0,
            "contour_points": 0, "size": 0, "aspect": 0,
        }
        for _, bw in self._binarizations(gray):
            contours, _ = cv2.findContours(
                bw, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE
            )
            for cnt in contours:
                reason, circ, ell = self._gate_contour(cnt, min_area, max_area, min_circ)
                if reason is not None:
                    rejection_counts[reason] += 1
                    continue
                scored.append((circ, ell))

        # Group.  Seeds are taken most circular first, as the historical
        # dedup kept candidates.  A contour closer than max(4, 0.4 r) to a
        # seed of compatible scale is the same blob at another threshold and
        # joins that group.  A concentric contour of clearly different scale
        # (e.g. the bright core of a large dot under a strong adaptive
        # threshold, or the dot inside its black square) starts its own
        # group: the historical dedup discarded it, which lost the real dot
        # whenever the other contour happened to be more circular.  Duplicate
        # decodes are still resolved by the detection-level spatial dedup.
        scored.sort(key=lambda x: x[0], reverse=True)
        groups: list[CandidateGroup] = []
        centers = np.empty((len(scored), 2), dtype=np.float64)
        seed_major = np.empty(len(scored), dtype=np.float64)
        for circ, ell in scored:
            c = np.array([ell[0][0], ell[0][1]])
            major = max(ell[1])
            r = major / 2.0
            if groups:
                distances = np.linalg.norm(centers[:len(groups)] - c, axis=1)
                close = np.flatnonzero(distances < max(4.0, r * 0.4))
                if close.size:
                    ratios = major / np.maximum(seed_major[close], 1e-9)
                    compatible = close[(ratios >= 0.55) & (ratios <= 1.8)]
                    if compatible.size:
                        nearest = int(compatible[np.argmin(distances[compatible])])
                        groups[nearest].members.append((circ, ell))
                        continue
            centers[len(groups)] = c
            seed_major[len(groups)] = major
            groups.append(CandidateGroup(seed=ell, members=[(circ, ell)]))
        self._last_candidate_rejections = rejection_counts
        return groups

    def _find_candidates(self, gray: np.ndarray, min_circ: float = 0.60):
        """
        Find ellipse candidates from multiple binarisations.

        Returns one ellipse per blob (the group seed), as historically.
        """
        return [group.seed for group in self._find_candidate_groups(gray, min_circ)]

    def contour_gate_reasons(
        self,
        gray: np.ndarray,
        points: np.ndarray,
        radius_px: np.ndarray | float,
        min_circ: float = 0.60,
    ) -> list[str]:
        """Developer diagnostic: furthest candidate gate reached near points.

        For each point, every contour whose centroid lies within ``radius_px``
        is passed through the production gates.  The result is ``"no_contour"``,
        the most advanced rejection reason, or ``"candidate"`` when at least
        one contour passed every gate.
        """
        points = np.asarray(points, dtype=np.float64).reshape(-1, 2)
        radii = np.broadcast_to(np.asarray(radius_px, dtype=np.float64), (len(points),))
        rank = {"no_contour": 0, "area": 1, "perimeter": 1, "contour_points": 1,
                "circularity": 2, "aspect": 3, "size": 3, "candidate": 4}
        best = ["no_contour"] * len(points)
        if not len(points):
            return best
        H, W = gray.shape[:2]
        min_area = max(30, int(H * W * 1e-6))
        max_area = int(H * W * 0.008)
        for _, bw in self._binarizations(gray):
            contours, _ = cv2.findContours(bw, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
            for cnt in contours:
                centroid = cnt.reshape(-1, 2).astype(np.float64).mean(axis=0)
                near = np.flatnonzero(np.linalg.norm(points - centroid, axis=1) <= radii)
                if not near.size:
                    continue
                reason, _, _ = self._gate_contour(cnt, min_area, max_area, min_circ)
                reason = reason or "candidate"
                for index in near:
                    if rank[reason] > rank[best[index]]:
                        best[index] = reason
        return best

    # ---- 1b. threshold-independent dot scale -----------------------------

    @staticmethod
    def _measure_dot_scale(gray: np.ndarray, ell) -> float | None:
        """Scale of the white-dot edge relative to a hypothesis ellipse.

        The angular mean of radial intensity profiles is taken along the
        ellipse's own affine rays, and the strongest white-to-black step is
        located with sub-sample precision.  A blurred step's gradient maximum
        sits at mid-contrast, so this does not depend on which binarisation
        produced the hypothesis.  Returns None when no clear edge is found.
        """
        (cx, cy), (aw, ah), ang = ell
        if not np.all(np.isfinite([cx, cy, aw, ah, ang])) or min(aw, ah) <= 0:
            return None
        theta = np.deg2rad(ang)
        angles = np.linspace(0.0, 2.0 * np.pi, 120, endpoint=False)
        local = np.column_stack([0.5 * aw * np.cos(angles), 0.5 * ah * np.sin(angles)])
        rotation = np.array([[np.cos(theta), -np.sin(theta)],
                             [np.sin(theta), np.cos(theta)]])
        directions = local @ rotation.T
        scales = np.linspace(0.5, 1.6, 89)
        x = cx + directions[:, 0, None] * scales
        y = cy + directions[:, 1, None] * scales
        height, width = gray.shape[:2]
        rows = np.all((x >= 0) & (x <= width - 1) & (y >= 0) & (y <= height - 1), axis=1)
        if np.mean(rows) < 0.75:
            return None
        source = gray if gray.dtype == np.float32 else gray.astype(np.float32)
        values = cv2.remap(source, x.astype(np.float32), y.astype(np.float32),
                           cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        profile = values[rows].mean(axis=0).astype(np.float64)
        gradient = np.gradient(profile, scales)
        window = np.flatnonzero((scales >= 0.6) & (scales <= 1.5))
        index = int(window[np.argmin(gradient[window])])
        if index <= window[0] or index >= window[-1]:
            return None
        span = 8  # +-0.1 of the hypothesis radius
        contrast = profile[max(0, index - span)] - profile[min(len(profile) - 1, index + span)]
        if not np.isfinite(contrast) or contrast < 15.0:
            return None
        lo, mid, hi = gradient[index - 1], gradient[index], gradient[index + 1]
        denominator = lo - 2.0 * mid + hi
        offset = 0.5 * (lo - hi) / denominator if abs(denominator) > 1e-12 else 0.0
        scale = float(scales[index] + np.clip(offset, -0.5, 0.5) * (scales[1] - scales[0]))
        return scale if 0.6 <= scale <= 1.5 else None

    @staticmethod
    def _normalized_radius(ell, point) -> float:
        """Distance of ``point`` from the ellipse centre in units of the ellipse."""
        (cx, cy), (aw, ah), ang = ell
        theta = np.deg2rad(ang)
        delta = np.asarray(point, dtype=np.float64) - np.array([cx, cy])
        local = np.array([[np.cos(theta), np.sin(theta)],
                          [-np.sin(theta), np.cos(theta)]]) @ delta
        return float(np.hypot(local[0] / (0.5 * aw), local[1] / (0.5 * ah)))

    def _verify_ring_geometry(self, prepared_gray: np.ndarray, ell) -> bool:
        """Check that measured ring edges sit at 2R and 3R of the dot edge.

        The dot ellipse is fitted to sub-pixel ring-1 edges (the same sampler
        the centre refinement uses), then every black-to-white edge on the
        code ring's inner boundary and every outer code-ring edge must lie at
        the expected normalised radius.  Only the white sectors provide such
        edges, so this works for any code length, including single arcs.
        """
        from .refinement import _fit_edge_ellipse, _ring_edges

        fit = _fit_edge_ellipse(_ring_edges(prepared_gray, ell, 1.0))
        if fit is None:
            return False
        dot = fit[0]
        for ring, tolerance in ((2.0, 0.25), (3.0, 0.35)):
            points = _ring_edges(prepared_gray, dot, ring)
            if len(points) < 12:
                return False
            rho = np.array([self._normalized_radius(dot, point) for point in points])
            median = float(np.median(rho))
            spread = float(np.median(np.abs(rho - median)))
            if abs(median - ring) > tolerance or spread > 0.12 * ring:
                return False
        return True

    def _float_image(self, gray: np.ndarray) -> np.ndarray:
        """float32 copy of the current image, converted once per image."""
        cached = getattr(self, "_float_cache", None)
        if cached is None or cached[0] is not gray:
            self._float_cache = (gray, gray.astype(np.float32))
        return self._float_cache[1]

    def _group_hypotheses(self, group: CandidateGroup) -> list:
        """Members ordered from the median scale outwards, capped."""
        members = [ell for _, ell in group.members]
        areas = np.array([max(ell[1][0] * ell[1][1], 1e-9) for ell in members])
        median_area = float(np.median(areas))
        order = sorted(range(len(members)),
                       key=lambda i: (abs(math.log(areas[i] / median_area)), i))
        return [members[i] for i in order[:self.max_hypotheses]]

    # ---- 2. affine rectification ---------------------------------------

    @staticmethod
    def _rectify_patch(
        gray: np.ndarray,
        ell,
        out_size: int = 200,
        return_mask: bool = False,
    ):
        """
        Warp the neighbourhood of *ell* so the ellipse becomes a circle
        centred in a (out_size x out_size) patch.  Returns the warped
        grayscale patch or None if the source region is out of bounds.
        """
        (cx, cy), (aw, ah), ang = ell
        H, W = gray.shape[:2]

        # The 3x outer ellipse
        major3 = max(aw, ah) * 1.6  # a bit more than 3x radius
        half = int(math.ceil(major3))

        # Source ROI — pad image if near border rather than rejecting
        r_min = int(round(cy - half))
        r_max = int(round(cy + half))
        c_min = int(round(cx - half))
        c_max = int(round(cx + half))
        pad = 0
        original_h, original_w = H, W
        if r_min < 0 or c_min < 0 or r_max > H or c_max > W:
            pad = half + 2
            gray = cv2.copyMakeBorder(gray, pad, pad, pad, pad,
                                      cv2.BORDER_REFLECT_101)
            cx += pad
            cy += pad
            H, W = gray.shape[:2]
            r_min = int(round(cy - half))
            r_max = int(round(cy + half))
            c_min = int(round(cx - half))
            c_max = int(round(cx + half))

        # Build affine: map 3x-ellipse bounding box corners to square
        theta = math.radians(ang)
        cos_t, sin_t = math.cos(theta), math.sin(theta)
        a3 = aw / 2.0 * 3.0
        b3 = ah / 2.0 * 3.0

        # 3 source points on the 3x-ellipse (0deg, 90deg, and center)
        # Point at 0 deg on ellipse
        src_pts = np.float32([
            [cx + cos_t * a3,           cy + sin_t * a3],
            [cx - sin_t * b3,           cy + cos_t * b3],
            [cx,                         cy],
        ])
        # Corresponding destination: circle of radius out_size/2
        r_out = out_size / 2.0
        dst_pts = np.float32([
            [out_size / 2.0 + r_out,  out_size / 2.0],
            [out_size / 2.0,          out_size / 2.0 + r_out],
            [out_size / 2.0,          out_size / 2.0],
        ])
        M = cv2.getAffineTransform(src_pts, dst_pts)
        # Warp only the source footprint of the patch (plus an interpolation
        # margin) instead of the whole image: the result is the same, but no
        # full-size arrays are created per candidate on large images.
        inverse = cv2.invertAffineTransform(M)
        corners = np.array([[0.0, 0.0, 1.0], [out_size, 0.0, 1.0],
                            [0.0, out_size, 1.0], [out_size, out_size, 1.0]])
        footprint = corners @ inverse.T
        x0 = min(max(int(np.floor(footprint[:, 0].min())) - 2, 0), W - 1)
        y0 = min(max(int(np.floor(footprint[:, 1].min())) - 2, 0), H - 1)
        x1 = max(min(int(np.ceil(footprint[:, 0].max())) + 3, W), x0 + 1)
        y1 = max(min(int(np.ceil(footprint[:, 1].max())) + 3, H), y0 + 1)
        M_crop = M.copy()
        M_crop[:, 2] += M[:, :2] @ np.array([x0, y0], dtype=np.float64)
        warped = cv2.warpAffine(gray[y0:y1, x0:x1], M_crop, (out_size, out_size),
                                flags=cv2.INTER_LINEAR,
                                borderMode=cv2.BORDER_CONSTANT,
                                borderValue=128)
        if not return_mask:
            return warped
        # Valid source pixels are those of the original (unpadded) image.
        source_valid = np.zeros((y1 - y0, x1 - x0), dtype=np.uint8)
        source_valid[max(pad - y0, 0):max(pad + original_h - y0, 0),
                     max(pad - x0, 0):max(pad + original_w - x0, 0)] = 1
        valid = cv2.warpAffine(
            source_valid, M_crop, (out_size, out_size),
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        ) > 0
        return warped, valid

    # ---- 3. strict ring validation on binarised rectified patch ---------

    @staticmethod
    def _validate_rectified(
        patch_gray: np.ndarray,
        valid_mask: np.ndarray | None = None,
        sample_n: int = 36,
    ):
        """
        On the rectified 200x200 patch, binarise and check:
          - centre ring (0.5 r1) is ALL white
          - guard ring  (1.5 r1) is ALL black
          - code ring   (2.5 r1) has at least 2 white AND 2 black samples
        where r1 = patch_size / 6  (= radius of inner white circle).
        
        Returns True if the pattern matches a valid CCT.
        """
        sz = patch_gray.shape[0]
        X0 = Y0 = sz / 2.0
        r1 = sz / 6.0   # inner white circle radius

        # Binarise the rectified patch
        _, bw = cv2.threshold(patch_gray, 0, 255,
                              cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        n_white_center = n_black_guard = n_white_code = 0
        n_center = n_guard = n_code = 0

        for j in range(sample_n):
            angle = 2.0 * math.pi * j / sample_n
            cos_a = math.cos(angle)
            sin_a = math.sin(angle)

            # Centre sample at 0.5 r1
            x = int(round(X0 + 0.5 * r1 * cos_a))
            y = int(round(Y0 + 0.5 * r1 * sin_a))
            if 0 <= x < sz and 0 <= y < sz and (valid_mask is None or valid_mask[y, x]):
                if bw[y, x] > 0:
                    n_white_center += 1
                n_center += 1

            # Guard ring at 1.5 r1
            x = int(round(X0 + 1.5 * r1 * cos_a))
            y = int(round(Y0 + 1.5 * r1 * sin_a))
            if 0 <= x < sz and 0 <= y < sz and (valid_mask is None or valid_mask[y, x]):
                if bw[y, x] == 0:
                    n_black_guard += 1
                n_guard += 1

            # Code ring at 2.5 r1
            x = int(round(X0 + 2.5 * r1 * cos_a))
            y = int(round(Y0 + 2.5 * r1 * sin_a))
            if 0 <= x < sz and 0 <= y < sz and (valid_mask is None or valid_mask[y, x]):
                if bw[y, x] > 0:
                    n_white_code += 1
                n_code += 1

        # A padded/border patch must never pass by normalising its evidence to
        # only the visible samples.  The historical detector required every
        # angular sample to be present; retain that invariant when a validity
        # mask is available.
        if min(n_center, n_guard, n_code) < sample_n:
            return False

        # Allow tolerance for perspective residuals and uneven illumination
        if n_white_center < sample_n * 0.70:
            return False
        if n_black_guard < sample_n * 0.70:
            return False
        n_black_code = sample_n - n_white_code
        if n_white_code < 2 or n_black_code < 2:
            return False

        return True

    @staticmethod
    def _validate_radial_consistency(
        patch_gray: np.ndarray,
        valid_mask: np.ndarray | None = None,
        sample_n: int = 48,
    ) -> bool:
        """Require the three rings to persist over radial samples.

        The original single-radius checks are retained as the first gate.  A
        second, independent check rejects a crescent, text glyph, or other
        arbitrary ellipse that happens to match at one radius: the white
        centre and black guard must remain stable over neighbouring radii, and
        each code sector must keep the same binary value through the code
        annulus.
        """
        sz = patch_gray.shape[0]
        center = sz / 2.0
        r1 = sz / 6.0
        _, bw = cv2.threshold(
            patch_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )

        def ring_samples(radius: float) -> np.ndarray | None:
            values: list[int] = []
            for j in range(sample_n):
                angle = 2.0 * math.pi * j / sample_n
                x = int(round(center + radius * r1 * math.cos(angle)))
                y = int(round(center + radius * r1 * math.sin(angle)))
                if not (0 <= x < sz and 0 <= y < sz):
                    return None
                if valid_mask is not None and not bool(valid_mask[y, x]):
                    return None
                values.append(int(bw[y, x] > 0))
            return np.asarray(values, dtype=np.int8)

        centre_profiles = [ring_samples(radius) for radius in (0.30, 0.50, 0.70)]
        guard_profiles = [ring_samples(radius) for radius in (1.25, 1.50, 1.75)]
        code_profiles = [ring_samples(radius) for radius in (2.25, 2.50, 2.75)]
        if any(profile is None for profile in (*centre_profiles, *guard_profiles, *code_profiles)):
            return False

        centre = np.stack([profile for profile in centre_profiles if profile is not None])
        guard = np.stack([profile for profile in guard_profiles if profile is not None])
        code = np.stack([profile for profile in code_profiles if profile is not None])
        if any(float(np.mean(profile)) < 0.70 for profile in centre):
            return False
        if any(float(np.mean(profile == 0)) < 0.70 for profile in guard):
            return False

        # Radial agreement is evaluated per angle, so a valid sector may be
        # white or black but cannot change value merely with radius.
        code_vote = np.mean(code, axis=0) >= 0.5
        agreement = np.mean(code == code_vote[None, :])
        if float(agreement) < 0.80:
            return False
        if int(np.count_nonzero(code_vote)) < 2 or int(np.count_nonzero(~code_vote)) < 2:
            return False
        return True

    # ---- 4. decode angular profile from rectified grayscale patch -------

    def _angular_profile(self, patch_gray: np.ndarray, valid_mask: np.ndarray | None = None):
        """Smoothed angular intensity profile of the code annulus, or None."""
        sz = patch_gray.shape[0]
        X0 = Y0 = sz / 2.0
        r1 = sz / 6.0

        n_ang = self._n_angular_samples  # 840 for 14 bits
        # The decoder intentionally samples the established code annulus,
        # including its outer edge; the strict ring checks below and the
        # residual threshold are what reject scene patterns that only happen
        # to look circular.
        inner_r = 2.4 * r1
        outer_r = 3.2 * r1
        radii = np.linspace(inner_r, outer_r, 7)
        # Use the established integer samples and never interpolate missing
        # rays.  Interpolation can fabricate a plausible code from a clipped
        # border or unrelated scene contour.
        angles = 2.0 * np.pi * np.arange(n_ang) / n_ang
        xs = np.round(X0 + radii[None, :] * np.cos(angles)[:, None]).astype(np.int64)
        ys = np.round(Y0 + radii[None, :] * np.sin(angles)[:, None]).astype(np.int64)
        inside = (xs >= 0) & (xs < sz) & (ys >= 0) & (ys < sz)
        xc = np.clip(xs, 0, sz - 1)
        yc = np.clip(ys, 0, sz - 1)
        if valid_mask is not None:
            inside &= valid_mask[yc, xc].astype(bool)
        counts = inside.sum(axis=1)
        if np.any(counts == 0):
            return None
        values = patch_gray[yc, xc].astype(np.float64)
        profile = np.where(inside, values, 0.0).sum(axis=1) / counts

        # Smooth
        ks = max(3, n_ang // 80)
        if ks % 2 == 0:
            ks += 1
        pad = ks // 2
        padded = np.concatenate([profile[-pad:], profile, profile[:pad]])
        kernel = np.ones(ks) / ks
        return np.convolve(padded, kernel, mode='valid')[:n_ang]

    def _decode_patch(self, patch_gray: np.ndarray, valid_mask: np.ndarray | None = None):
        """
        Sample the code ring on the rectified grayscale patch and decode.
        Returns (canon, raw, bits_list, confidence, pattern_str) or None.

        With a codebook the confidence is the normalised correlation with the
        best code; otherwise it is the historical mean bit margin.
        """
        self._last_decode_corroborated = False
        smooth = self._angular_profile(patch_gray, valid_mask)
        if smooth is None:
            return None
        if self._codebook_fft_conj is not None:
            return self._decode_codebook(smooth)
        return self._decode_generic(smooth)

    def _code_transitions(self, code: int) -> int:
        bits = [(code >> k) & 1 for k in range(self.n_bits)]
        return sum(bits[k] != bits[(k + 1) % self.n_bits] for k in range(self.n_bits))

    def _decode_generic(self, smooth: np.ndarray):
        """Phase-search decode without a codebook (historical behaviour)."""
        n_ang = smooth.size
        step = n_ang / self.n_bits
        p_lo = np.percentile(smooth, 15)
        p_hi = np.percentile(smooth, 85)
        if p_hi - p_lo < 8:
            return None
        thresh = (p_lo + p_hi) / 2.0

        n_phases = max(1, int(round(step)))
        phases = np.arange(n_phases)[:, None]
        sectors = np.arange(self.n_bits)[None, :]
        lo = np.round(phases + sectors * step).astype(np.int64)
        hi = np.round(phases + (sectors + 1) * step).astype(np.int64)
        lengths = hi - lo
        cumulative = np.concatenate([[0.0], np.cumsum(np.concatenate([smooth, smooth]))])
        means = (cumulative[hi] - cumulative[lo]) / np.maximum(lengths, 1)

        bits = (means >= thresh).astype(np.int64)
        transitions = np.sum(bits != np.roll(bits, 1, axis=1), axis=1)
        ones = bits.sum(axis=1)
        usable = (
            np.all(lengths > 0, axis=1)
            & (transitions >= 2) & (transitions <= 13)
            & (ones >= 2) & (ones <= self.n_bits - 2)
        )
        if not np.any(usable):
            return None

        # Two-level reconstruction per phase: sample j belongs to the sector
        # whose [lo, hi) range contains it (ranges tile one full turn).
        offsets = (np.arange(n_ang)[None, :] - lo[:, :1]) % n_ang
        boundaries = lo - lo[:, :1]
        sector_of = np.sum(offsets[:, :, None] >= boundaries[:, None, :], axis=2) - 1
        levels = np.take_along_axis(bits, sector_of, axis=1)
        recon = np.where(levels > 0, p_hi, p_lo)
        mse = np.mean((smooth[None, :] - recon) ** 2, axis=1)
        margin = np.mean(np.abs(means - thresh), axis=1)
        score = mse - 0.15 * margin

        best_by_id: dict[int, float] = {}
        best_phase: int | None = None
        for phase in np.flatnonzero(usable):
            raw = sum((1 << k) for k, b in enumerate(bits[phase]) if b)
            canon = canonical_code(raw, self.n_bits)
            prior = best_by_id.get(canon)
            if prior is None or score[phase] < prior:
                best_by_id[canon] = float(score[phase])
            if best_phase is None or score[phase] < score[best_phase]:
                best_phase = int(phase)

        best_bits = bits[best_phase]
        raw = sum((1 << k) for k, b in enumerate(best_bits) if b)
        canon = canonical_code(raw, self.n_bits)
        range_sq = (p_hi - p_lo) ** 2
        if range_sq > 0 and mse[best_phase] / range_sq > 0.030:
            return None

        # Reject block codes (e.g. 0000001111111) — real CCT codes have >= 4
        # transitions between sectors; simple all-ones blocks are noise FPs.
        # Without a codebook this deliberately also excludes genuine
        # single-arc codes; codebook decoding does not need this heuristic.
        if int(transitions[best_phase]) < 4:
            return None

        # A close competing ID is an ambiguity, not a confident detection.
        ranked_scores = sorted(best_by_id.values())
        if len(ranked_scores) > 1 and ranked_scores[1] - ranked_scores[0] < 0.006:
            return None
        pattern = ''.join(str(int(b)) for b in best_bits)
        return canon, raw, [int(b) for b in best_bits], float(margin[best_phase]), pattern

    def _decode_codebook(self, smooth: np.ndarray):
        """Decode by circular correlation with every known code.

        All 14 rotations and every sub-sector phase are covered by the circular
        shift, so no bit thresholding is involved.  The best code is accepted
        when the generic decoder reads the same code from this profile
        (``_last_decode_corroborated``), or when it reaches the correlation and
        margin over the best *different* code required for its information
        content (see the constructor).
        """
        n_ang = smooth.size
        p_lo = np.percentile(smooth, 15)
        p_hi = np.percentile(smooth, 85)
        self._last_codebook_scores = None
        if p_hi - p_lo < 8:
            return None
        centred = smooth - smooth.mean()
        norm = float(np.linalg.norm(centred))
        if norm <= 1e-9:
            return None
        spectrum = np.fft.rfft(centred / norm)
        correlation = np.fft.irfft(spectrum[None, :] * self._codebook_fft_conj, n=n_ang, axis=1)
        best_shift = np.argmax(correlation, axis=1)
        best_value = correlation[np.arange(len(best_shift)), best_shift]
        order = np.argsort(best_value)[::-1]
        first = int(order[0])
        top = float(best_value[first])
        second = float(best_value[order[1]]) if len(order) > 1 else -1.0
        self._last_codebook_scores = (top, second)
        code = self.codebook[first]
        generic = self._decode_generic(smooth)
        corroborated = generic is not None and int(generic[0]) == int(code)
        if not corroborated:
            if self._code_transitions(code) <= 4:
                min_correlation = self.codebook_low_information_min_correlation
                min_margin = self.codebook_low_information_min_margin
            else:
                min_correlation = self.codebook_min_correlation
                min_margin = self.codebook_min_margin
            if top < min_correlation or top - second < min_margin:
                return None
        self._last_decode_corroborated = corroborated

        shift = int(best_shift[first])
        # Profile sample j matches template sample j - shift; read the
        # observed sector bits at sector centres.
        step = n_ang / self.n_bits
        centres = (np.arange(self.n_bits) + 0.5) * step
        template_sector = np.floor(((centres - shift) % n_ang) / step).astype(int) % self.n_bits
        bits = [int((code >> int(k)) & 1) for k in template_sector]
        raw = sum((1 << k) for k, b in enumerate(bits) if b)
        pattern = ''.join(str(b) for b in bits)
        return code, raw, bits, top, pattern

    def _evaluate_group(self, gray: np.ndarray, measure_gray: np.ndarray,
                        group: CandidateGroup, counts: dict[str, int]):
        """Validate and decode hypotheses of one group until enough agree.

        Returns ((ellipse, decode_result, votes), None) on success or
        (None, failure_stage) otherwise.  Groups of three or more contours
        need two agreeing hypotheses; smaller groups keep the historical
        single-decode acceptance.
        """
        hypotheses = self._group_hypotheses(group)
        required = 2 if len(group.members) >= 3 else 1
        required = min(required, len(hypotheses))
        votes: dict[int, int] = {}
        best: dict[int, tuple] = {}
        corroborated: dict[int, bool] = {}
        failure = "validate"
        for ell in hypotheses:
            counts["hypotheses"] += 1
            scale = self._measure_dot_scale(self._float_image(measure_gray), ell)
            if scale is not None:
                counts["scale_measured"] += 1
                (cx, cy), (aw, ah), angle = ell
                ell = ((float(cx), float(cy)), (float(aw * scale), float(ah * scale)), float(angle))
            if max(ell[1]) / 2.0 > self.max_target_radius:
                failure = "size"
                continue

            # Rectify
            patch, valid_mask = self._rectify_patch(gray, ell, out_size=200, return_mask=True)
            if patch is None:
                continue

            # Validate ring structure on binarised rectified patch
            if not self._validate_rectified(patch, valid_mask):
                failure = "validate"
                continue
            if not self._validate_radial_consistency(patch, valid_mask):
                failure = "radial_validate"
                continue

            # Decode
            result = self._decode_patch(patch, valid_mask)
            if result is None:
                failure = "decode"
                continue
            canon = int(result[0])
            votes[canon] = votes.get(canon, 0) + 1
            corroborated[canon] = corroborated.get(canon, False) or self._last_decode_corroborated
            if canon not in best or result[3] > best[canon][1][3]:
                best[canon] = (ell, result)
            leader = max(votes, key=votes.get)
            others = max((v for k, v in votes.items() if k != leader), default=0)
            if votes[leader] >= required and votes[leader] > others:
                break

        if not votes:
            return None, failure
        leader = max(votes, key=votes.get)
        others = max((v for k, v in votes.items() if k != leader), default=0)
        if votes[leader] < required or votes[leader] <= others:
            return None, "vote"
        ell, result = best[leader]
        if self._codebook_fft_conj is not None and not corroborated.get(leader, False):
            if votes[leader] < 2 and result[3] < self.codebook_single_vote_min_correlation:
                return None, "vote"
            # A single-arc code (two transitions) carries little angular
            # information: any bright/dark edge correlates with it.  Such IDs
            # additionally need the ring geometry to be confirmed.
            if self._code_transitions(leader) < 4:
                if not self._verify_ring_geometry(prepare_refinement_image(measure_gray), ell):
                    return None, "ring_verification"
        return (ell, result, votes[leader]), None

    # ---- 5. main pipeline -----------------------------------------------

    def detect_with_diagnostics(
        self, image_bgr: np.ndarray,
        intrinsics: np.ndarray | None = None,
    ) -> tuple[list[Detection], DetectionDiagnostics]:
        raw_gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        gray = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(raw_gray)
        groups = self._find_candidate_groups(gray)
        groups.sort(key=lambda g: max(g.seed[1]), reverse=True)

        counts = {
            "candidates": len(groups),
            "candidate_contours": int(sum(len(group.members) for group in groups)),
            "hypotheses": 0, "scale_measured": 0,
            "validated": 0, "decoded": 0,
            "refined_centers": 0, "reject_validate": 0, "reject_decode": 0,
            "reject_vote": 0, "reject_size": 0, "reject_ring_verification": 0,
            "spatial_duplicates": 0, "id_duplicates": 0,
        }
        counts.update({f"candidate_reject_{key}": int(value) for key, value in getattr(self, "_last_candidate_rejections", {}).items()})
        decoder_name = "codebook" if self._codebook_fft_conj is not None else "generic"
        rejected: list[dict[str, object]] = []
        detections: list[Detection] = []
        used: list[np.ndarray] = []

        for group in groups:
            outcome, failure = self._evaluate_group(gray, raw_gray, group, counts)
            if outcome is None:
                seed_x, seed_y = group.seed[0]
                if failure in ("validate", "radial_validate"):
                    counts["reject_validate"] += 1
                else:
                    counts[f"reject_{failure}"] += 1
                rejected.append({
                    "x": float(seed_x), "y": float(seed_y),
                    "r": float(max(group.seed[1]) / 2.0),
                    "stage": failure, "contours": len(group.members),
                })
                continue
            counts["validated"] += 1
            counts["decoded"] += 1
            ell, result, votes = outcome
            (cx, cy), (aw, ah), ang = ell
            r = max(aw, ah) / 2.0

            canon, raw, bits, margin, pat = result

            # Keep acceptance and deduplication tied to the conservative
            # ellipse measurement.  Calibrated conic refinement is applied
            # only after the candidate has survived every detector gate.
            center = np.array([cx, cy], dtype=float)
            candidate = Detection(
                target_id=canon, center=(float(center[0]), float(center[1])), ellipse=ell,
                confidence=margin, code_bits=bits, raw_code=raw,
                pattern=pat,
                ellipse_center=(float(cx), float(cy)),
                center_correction_px=(0.0, 0.0),
                center_method="ellipse",
                votes=int(votes),
                decoder=decoder_name,
                support=len(group.members),
            )

            # Dedup: full target outer edge ≈ 3× inner-circle radius;
            # use 4× to also catch candidates from outer rings of the same
            # target while not eating adjacent distinct targets.
            c_arr = np.array([cx, cy])
            for i, uc in enumerate(used):
                det_r = max(detections[i].ellipse[1]) / 2.0
                # Keep the original broad spatial suppression.  A single
                # printed/painted shape often yields several contour scales;
                # retaining them is a direct path to different hallucinated
                # IDs surviving the later ID-level deduplication.
                dup_r = max(20.0, max(r, det_r) * 4.0)
                if np.linalg.norm(c_arr - uc) < dup_r:
                    # A candidate centred on the other one's code ring is a
                    # code sector of that target, whatever its decode score.
                    new_on_ring = 1.6 <= self._normalized_radius(detections[i].ellipse, c_arr) <= 3.4
                    old_on_ring = 1.6 <= self._normalized_radius(ell, uc) <= 3.4
                    # Otherwise prefer the better-corroborated candidate: a
                    # real dot is found by most binarisations, while small
                    # printed card glyphs can decode with a high score from
                    # only a few.  The decode score is the final tie-break.
                    if new_on_ring != old_on_ring:
                        replace = old_on_ring
                    else:
                        previous = detections[i]
                        replace = (int(votes), len(group.members), margin) > (
                            previous.votes, previous.support, previous.confidence)
                    if replace:
                        rejected.append({
                            "x": float(detections[i].center[0]), "y": float(detections[i].center[1]),
                            "r": float(det_r), "stage": "spatial_duplicate",
                            "target_id": int(detections[i].target_id),
                        })
                        detections[i] = candidate
                        used[i] = c_arr
                    else:
                        rejected.append({
                            "x": float(cx), "y": float(cy), "r": float(r),
                            "stage": "spatial_duplicate", "target_id": int(canon),
                        })
                    counts["spatial_duplicates"] += 1
                    break
            else:
                detections.append(candidate)
                used.append(c_arr)

        detections.sort(key=lambda d: (d.center[1], d.center[0]))

        # Final ID-level dedup: if the same target_id appears more than once,
        # keep only the highest-confidence instance.
        seen: dict[int, int] = {}  # target_id -> index in best_detections
        best_detections: list[Detection] = []
        for det in detections:
            if det.target_id in seen:
                idx = seen[det.target_id]
                loser = det
                if det.confidence > best_detections[idx].confidence:
                    loser = best_detections[idx]
                    best_detections[idx] = det
                rejected.append({
                    "x": float(loser.center[0]), "y": float(loser.center[1]),
                    "r": float(max(loser.ellipse[1]) / 2.0),
                    "stage": "id_duplicate", "target_id": int(loser.target_id),
                })
                counts["id_duplicates"] += 1
            else:
                seen[det.target_id] = len(best_detections)
                best_detections.append(det)
        best_detections.sort(key=lambda d: (d.center[1], d.center[0]))

        # Filter by valid ID range if specified
        if self.valid_id_range is not None:
            lo, hi = self.valid_id_range
            best_detections = [d for d in best_detections if lo <= d.target_id <= hi]

        # Refine only finalized detections.  This makes the calibrated center
        # estimate an accuracy improvement, never a source of extra accepted
        # candidates or altered spatial/ID deduplication decisions.
        refined_detections: list[Detection] = []
        refinement_gray = (
            prepare_refinement_image(raw_gray) if intrinsics is not None else raw_gray
        )
        for det in best_detections:
            ellipse_center = det.ellipse_center or (
                float(det.ellipse[0][0]), float(det.ellipse[0][1])
            )
            estimate: CenterEstimate = refine_projected_center(
                refinement_gray, det.ellipse, intrinsics,
                image_is_prepared=intrinsics is not None,
            )
            if estimate.valid:
                counts["refined_centers"] += 1
                center = estimate.projected_center_px
                correction = (
                    float(center[0] - ellipse_center[0]),
                    float(center[1] - ellipse_center[1]),
                )
                method = estimate.method
                covariance = (
                    tuple(float(v) for v in estimate.conditional_covariance_px2.reshape(-1))
                    if estimate.conditional_covariance_px2 is not None else None
                )
            else:
                center = np.asarray(ellipse_center, dtype=float)
                correction = det.center_correction_px or (0.0, 0.0)
                method = "ellipse"
                covariance = None
            refined_detections.append(Detection(
                target_id=det.target_id,
                center=(float(center[0]), float(center[1])),
                ellipse=det.ellipse,
                confidence=det.confidence,
                code_bits=det.code_bits,
                raw_code=det.raw_code,
                pattern=det.pattern,
                ellipse_center=(float(ellipse_center[0]), float(ellipse_center[1])),
                center_correction_px=correction,
                center_method=method,
                center_covariance_px2=covariance,
                votes=det.votes,
                decoder=det.decoder,
                support=det.support,
            ))
        best_detections = sorted(
            refined_detections,
            key=lambda d: (d.center[1], d.center[0]),
        )

        return best_detections, DetectionDiagnostics(counts=counts, rejected=rejected)

    def detect(self, image_bgr: np.ndarray, intrinsics: np.ndarray | None = None) -> list[Detection]:
        detections, _ = self.detect_with_diagnostics(image_bgr, intrinsics=intrinsics)
        return detections

    # ---- 6. annotation ---------------------------------------------------

    def annotate(self, image_bgr: np.ndarray,
                 detections: Sequence[Detection]) -> np.ndarray:
        out = image_bgr.copy()
        for det in detections:
            cx, cy = int(round(det.center[0])), int(round(det.center[1]))
            cv2.ellipse(out, det.ellipse, (0, 255, 255), 2)
            cv2.circle(out, (cx, cy), 3, (0, 255, 0), -1)
            label = str(det.target_id)
            cv2.putText(out, label, (cx + 8, cy - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)
        return out
