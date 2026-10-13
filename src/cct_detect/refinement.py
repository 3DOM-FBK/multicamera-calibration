"""Physical-centre refinement conditional on a calibrated distortion model.

Edges are measured in the original image and undistorted before fitting the
concentric conic pencil.  Its isolated generalized eigenvector is the projected
physical centre; that point is distorted back into the original image.  Local
covariance remains unavailable until the edge-noise model has been validated.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from scipy.linalg import eig

Ellipse = tuple[tuple[float, float], tuple[float, float], float]


@dataclass(frozen=True)
class CenterEstimate:
    ellipse_center_px: np.ndarray
    projected_center_px: np.ndarray
    conditional_covariance_px2: np.ndarray | None
    method: str
    valid: bool
    reason: str
    edge_rms_px: float


def _ellipse_conic(ellipse: Ellipse) -> np.ndarray:
    (cx, cy), (width, height), angle = ellipse
    axes = np.asarray([width, height], dtype=float) / 2.0
    if not np.all(np.isfinite(axes)) or np.min(axes) <= 0:
        raise ValueError("invalid ellipse axes")
    theta = np.deg2rad(angle)
    rotation = np.array([[np.cos(theta), -np.sin(theta)],
                         [np.sin(theta), np.cos(theta)]])
    metric = rotation @ np.diag(1.0 / axes**2) @ rotation.T
    center = np.asarray([cx, cy], dtype=float)
    conic = np.zeros((3, 3))
    conic[:2, :2] = metric
    conic[:2, 2] = -metric @ center
    conic[2, :2] = conic[:2, 2]
    conic[2, 2] = center @ metric @ center - 1.0
    return conic


def _center_from_conics(inner: np.ndarray, outer: np.ndarray,
                        radius_ratio: float) -> tuple[np.ndarray, float] | None:
    """Recover the isolated eigenvector, invariant to arbitrary conic scale.

    C_r = H^-T diag(1,1,-r^2) H^-1. The isolated generalized eigenvector
    of (C_inner,C_outer) is H[:,2]. Use centred/scaled input coordinates.
    """
    try:
        values, vectors = eig(inner, outer)
    except (ValueError, np.linalg.LinAlgError):
        return None
    if not np.all(np.isfinite(values)) or np.max(np.abs(values.imag)) > 1e-7 * max(np.max(np.abs(values)), 1.0):
        return None
    values = values.real
    pairs = [(0, 1, 2), (0, 2, 1), (1, 2, 0)]
    i, j, k = min(pairs, key=lambda p: abs(values[p[0]] - values[p[1]]))
    repeated = 0.5 * (values[i] + values[j])
    gap = abs(repeated - values[k])
    if gap <= 1e-10 * max(np.max(np.abs(values)), 1.0) or abs(repeated) < 1e-12:
        return None
    mismatch = abs(values[i] - values[j]) / gap
    expected = radius_ratio**2
    if mismatch > 0.08 or abs(values[k] / repeated - expected) > 0.12 * expected:
        return None
    point = vectors[:, k]
    if np.max(np.abs(point.imag)) > 1e-7 or abs(point[2]) < 1e-10:
        return None
    point = point.real
    return point[:2] / point[2], float(mismatch)


def _ring_edges(gray: np.ndarray, ellipse: Ellipse, ring: float) -> np.ndarray:
    """Sample radial edges, including discontinuous outer-ring arcs."""
    (cx, cy), (aw, ah), angle = ellipse
    theta = np.deg2rad(angle)
    angles = np.linspace(0, 2 * np.pi, 180, endpoint=False)
    local = np.column_stack([0.5 * aw * np.cos(angles), 0.5 * ah * np.sin(angles)])
    rotation = np.array([[np.cos(theta), -np.sin(theta)],
                         [np.sin(theta), np.cos(theta)]])
    directions = local @ rotation.T
    margin = 0.20 if ring == 1.0 else 0.40
    scales = np.linspace(ring - margin, ring + margin, 65)
    x = cx + directions[:, 0, None] * scales
    y = cy + directions[:, 1, None] * scales
    height, width = gray.shape
    valid = (x >= 1) & (x <= width - 2) & (y >= 1) & (y <= height - 2)
    values = cv2.remap(gray.astype(np.float32), x.astype(np.float32), y.astype(np.float32),
                       cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    gradient = np.gradient(values, scales, axis=1)
    # R: white -> black. 2R: black -> white code sectors.
    score = -gradient if ring == 1.0 else gradient if ring == 2.0 else np.abs(gradient)
    peaks = np.argmax(score[:, 2:-2], axis=1) + 2
    rows = np.arange(len(angles))
    valid_rows = np.all(valid, axis=1) & (np.ptp(values, axis=1) >= 15.0)
    valid_rows &= np.isfinite(score[rows, peaks]) & (score[rows, peaks] > 10.0)
    valid_rows &= (peaks > 2) & (peaks < len(scales) - 3)
    lo, mid, hi = score[rows, peaks - 1], score[rows, peaks], score[rows, peaks + 1]
    denominator = lo - 2.0 * mid + hi
    offset = np.divide(0.5 * (lo - hi), denominator,
                       out=np.zeros_like(mid), where=np.isfinite(denominator) & (np.abs(denominator) > 1e-9))
    scale = scales[peaks] + np.clip(offset, -0.5, 0.5) * (scales[1] - scales[0])
    points = np.asarray([cx, cy]) + directions * scale[:, None]
    return points[valid_rows]


def _fit_edge_ellipse(points: np.ndarray) -> tuple[Ellipse, float] | None:
    if len(points) < 30:
        return None
    original_count = len(points)
    for iteration in range(5):
        try:
            ellipse = cv2.fitEllipse(points.astype(np.float32).reshape(-1, 1, 2))
            conic = _ellipse_conic(ellipse)
        except (cv2.error, ValueError):
            return None
        homogeneous = np.column_stack([points, np.ones(len(points))])
        gradient = 2.0 * (homogeneous @ conic)[:, :2]
        residual = np.einsum("ni,ij,nj->n", homogeneous, conic, homogeneous)
        residual /= np.maximum(np.linalg.norm(gradient, axis=1), 1e-12)
        cutoff = max(0.35, 3.0 * 1.4826 * np.median(np.abs(residual - np.median(residual))))
        keep = np.abs(residual) <= cutoff
        if np.count_nonzero(keep) < max(30, int(0.55 * original_count)):
            return None
        if np.all(keep) or iteration == 4:
            break
        points = points[keep]
    center = np.asarray(ellipse[0])
    delta = points - center
    theta = np.deg2rad(ellipse[2])
    rotation = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    local = delta @ rotation / (np.asarray(ellipse[1]) / 2.0)
    bins = np.floor((np.arctan2(local[:, 1], local[:, 0]) + np.pi) * 24 / (2 * np.pi)).astype(int) % 24
    if len(np.unique(bins)) < 12:
        return None
    rms = float(np.sqrt(np.mean(residual**2)))
    if not np.isfinite(rms) or rms > 0.8:
        return None
    return ellipse, rms


def prepare_refinement_image(gray: np.ndarray) -> np.ndarray:
    """Prepare one grayscale image for any number of centre refinements."""
    if gray.ndim != 2:
        raise ValueError("centre refinement requires a grayscale image")
    return cv2.GaussianBlur(gray, (3, 3), 0.6)


def refine_projected_center(gray: np.ndarray, ellipse: Ellipse,
                            intrinsics: np.ndarray | None = None,
                            *, image_is_prepared: bool = False) -> CenterEstimate:
    """Refine a physical centre conditional on an OPENCV-8 calibration."""
    initial = np.asarray(ellipse[0], dtype=float)

    def unavailable(reason: str) -> CenterEstimate:
        return CenterEstimate(initial, initial.copy(), None, "ellipse", False, reason, float("nan"))

    if intrinsics is None:
        return unavailable("calibrated distortion model required for physical-centre refinement")
    intrinsics = np.asarray(intrinsics, dtype=float)
    if intrinsics.shape != (8,) or not np.all(np.isfinite(intrinsics)) or min(intrinsics[:2]) <= 0:
        return unavailable("invalid OPENCV-8 calibration")
    matrix = np.array([[intrinsics[0], 0, intrinsics[2]], [0, intrinsics[1], intrinsics[3]], [0, 0, 1.]])
    if not image_is_prepared:
        gray = prepare_refinement_image(gray)
    fitted = []
    for ring in (1.0, 2.0, 3.0):
        points = _ring_edges(gray, ellipse, ring)
        if len(points) < 30:
            continue
        measured = points
        points = cv2.undistortPoints(measured.reshape(-1, 1, 2), matrix, intrinsics[4:8], P=matrix).reshape(-1, 2)
        if not np.all(np.isfinite(points)):
            continue
        rays = np.column_stack([(points[:, 0] - intrinsics[2]) / intrinsics[0],
                                 (points[:, 1] - intrinsics[3]) / intrinsics[1],
                                 np.ones(len(points))])
        roundtrip = cv2.projectPoints(rays, np.zeros(3), np.zeros(3), matrix, intrinsics[4:8])[0].reshape(-1, 2)
        if np.max(np.linalg.norm(roundtrip - measured, axis=1)) > 0.05:
            continue  # Finite output alone does not establish successful undistortion.
        fit = _fit_edge_ellipse(points)
        if fit is not None:
            fitted.append((ring, fit[0], fit[1]))
    if len(fitted) < 2:
        return unavailable("fewer than two supported concentric boundaries")
    origin = cv2.undistortPoints(initial.reshape(1, 1, 2), matrix, intrinsics[4:8], P=matrix)[0, 0]
    radius = max(ellipse[1]) / 2.0
    transform = np.array([[radius, 0, origin[0]], [0, radius, origin[1]], [0, 0, 1.]])
    estimates = []
    for index, (ri, ei, rmsi) in enumerate(fitted):
        for rj, ej, rmsj in fitted[index + 1:]:
            ci = transform.T @ _ellipse_conic(ei) @ transform
            cj = transform.T @ _ellipse_conic(ej) @ transform
            estimate = _center_from_conics(ci, cj, ri / rj)
            if estimate is None:
                continue
            center, mismatch = estimate
            center = origin + radius * center
            if np.linalg.norm(center - origin) <= 0.35 * radius:
                estimates.append((rmsi + rmsj + mismatch, center))
    if not estimates:
        return unavailable("concentric pencil is unsupported or ill-conditioned")
    estimates.sort(key=lambda item: item[0])
    center = estimates[0][1]
    if any(np.linalg.norm(other - center) > max(0.5, 0.03 * radius) for _, other in estimates[1:]):
        return unavailable("independent boundary pairs disagree on projected centre")
    normalized = np.array([(center[0] - intrinsics[2]) / intrinsics[0],
                            (center[1] - intrinsics[3]) / intrinsics[1], 1.])
    projected = cv2.projectPoints(normalized.reshape(1, 3), np.zeros(3), np.zeros(3), matrix, intrinsics[4:8])[0].reshape(2)
    if not np.all(np.isfinite(projected)):
        return unavailable("nonfinite distorted centre projection")
    return CenterEstimate(initial, projected, None, "calibrated-concentric-conic", True,
                          "conditional on intrinsics; localization covariance unavailable",
                          float(np.mean([item[2] for item in fitted])))
