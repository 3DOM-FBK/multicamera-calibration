"""Alternative CCT detector with anchor-aware oriented decoding."""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

from cct_detect.detector import CCTDetector as BaseCCTDetector
from cct_detect.detector import canonical_code
from cct_detect.detector import Detection


@dataclass
class _DecodeResult:
    target_id: int
    raw_code: int
    code_bits: list[int]
    confidence: float
    pattern: str
    score: float
    anchor_score: float
    second_best_gap: float


@dataclass(frozen=True)
class _CodeCandidate:
    canonical_id: int
    oriented_raw: int
    oriented_bits: tuple[int, ...]
    rotation: int


@dataclass
class DetectionDiagnostics:
    stage_points: dict[str, list[dict[str, Any]]]
    counts: dict[str, int]


def _bits_from_int(value: int, n_bits: int) -> np.ndarray:
    return np.array([(value >> bit_index) & 1 for bit_index in range(n_bits)], dtype=np.float64)


def _int_from_bits(bits: np.ndarray) -> int:
    return sum((1 << bit_index) for bit_index, bit in enumerate(bits.tolist()) if int(bit))


class CCTDetector(BaseCCTDetector):
    """Alternative detector that preserves oriented bit order.

    The candidate finding, rectification, and annotation stages are reused from
    the baseline detector. The decode stage is replaced by a codebook-aware,
    anchor-aware phase search that keeps absolute bit orientation.
    """

    def __init__(
        self,
        n_bits: int = 14,
        valid_ids: set[int] | None = None,
        max_id_hamming_distance: int = 1,
        anchor_bits: tuple[int, int] = (0, 7),
        require_anchor_bits: bool = True,
        min_bit_purity: float = 0.75,
        mean_bit_purity: float = 0.88,
        bit_core_margin_fraction: float = 0.20,
    ):
        valid_id_range = None
        if valid_ids:
            valid_id_range = (min(valid_ids), max(valid_ids))
        super().__init__(n_bits=n_bits, valid_id_range=valid_id_range)
        self.valid_ids = valid_ids
        self.max_id_hamming_distance = max_id_hamming_distance
        self.anchor_bits = tuple(anchor_bits)
        self.require_anchor_bits = require_anchor_bits
        self.min_bit_purity = float(min_bit_purity)
        self.mean_bit_purity = float(mean_bit_purity)
        self.bit_core_margin_fraction = float(bit_core_margin_fraction)
        self.mixed_bit_low = 0.30
        self.mixed_bit_high = 0.70
        self.max_mixed_bits = 2
        self.fraction_error_threshold = 0.30
        self.max_fraction_error_bits = 2
        self.min_anchor_score = 0.70
        self.strip_variants: tuple[tuple[float, float, int], ...] = (
            (2.25, 2.95, 11),
            (2.20, 3.15, 11),
            (2.50, 3.50, 11),
        )
        self._candidate_codes = self._prepare_candidate_codes(valid_ids)

    def _prepare_candidate_codes(self, valid_ids: set[int] | None) -> list[_CodeCandidate] | None:
        if valid_ids is None:
            return None
        candidates: list[_CodeCandidate] = []
        seen_oriented_bits: set[tuple[int, ...]] = set()
        for canonical_id in sorted(valid_ids):
            base_bits = _bits_from_int(canonical_id, self.n_bits).astype(np.int32)
            for rotation in range(self.n_bits):
                oriented_bits = np.roll(base_bits, rotation)
                if self.require_anchor_bits and not all(int(oriented_bits[idx]) == 1 for idx in self.anchor_bits):
                    continue
                transitions = int(np.sum(oriented_bits != np.roll(oriented_bits, 1)))
                if transitions < 4:
                    continue
                key = tuple(int(bit) for bit in oriented_bits.tolist())
                if key in seen_oriented_bits:
                    continue
                seen_oriented_bits.add(key)
                candidates.append(
                    _CodeCandidate(
                        canonical_id=canonical_id,
                        oriented_raw=_int_from_bits(oriented_bits),
                        oriented_bits=key,
                        rotation=rotation,
                    )
                )
        return candidates or None

    def _snap_to_valid_id(self, raw_id: int) -> tuple[int | None, float]:
        if self.valid_ids is None:
            return raw_id, 0.0
        if raw_id in self.valid_ids:
            return raw_id, 0.0

        best_id: int | None = None
        best_distance: int | None = None
        ambiguous = False
        for candidate_id in self.valid_ids:
            distance = (raw_id ^ candidate_id).bit_count()
            if best_distance is None or distance < best_distance:
                best_id = candidate_id
                best_distance = distance
                ambiguous = False
            elif distance == best_distance:
                ambiguous = True
        if best_id is None or best_distance is None or ambiguous or best_distance > self.max_id_hamming_distance:
            return None, float("inf")
        return best_id, float(best_distance)

    def _sample_polar_strip(
        self,
        patch_gray: np.ndarray,
        inner_scale: float = 2.25,
        outer_scale: float = 2.95,
        n_radii: int = 11,
    ) -> np.ndarray | None:
        sz = patch_gray.shape[0]
        center = sz / 2.0
        r1 = sz / 6.0
        inner_r = inner_scale * r1
        outer_r = outer_scale * r1
        radii = np.linspace(inner_r, outer_r, n_radii)
        n_ang = max(360, 80 * self.n_bits)
        strip = np.full((len(radii), n_ang), np.nan, dtype=np.float64)

        for angle_index in range(n_ang):
            angle = 2.0 * np.pi * angle_index / n_ang
            cos_a = np.cos(angle)
            sin_a = np.sin(angle)
            for radius_index, radius in enumerate(radii):
                x = int(round(center + radius * cos_a))
                y = int(round(center + radius * sin_a))
                if 0 <= x < sz and 0 <= y < sz:
                    strip[radius_index, angle_index] = float(patch_gray[y, x])
        if np.all(np.isnan(strip)):
            return None
        return strip

    def _sector_means_for_phase(self, strip: np.ndarray, phase: int) -> np.ndarray | None:
        n_ang = strip.shape[1]
        step = n_ang / self.n_bits
        profile = np.nanmedian(strip, axis=0)
        if not np.all(np.isfinite(profile)):
            return None

        kernel_size = max(3, int(round(step * 0.18)))
        if kernel_size % 2 == 0:
            kernel_size += 1
        pad = kernel_size // 2
        kernel = np.ones(kernel_size, dtype=np.float64) / kernel_size
        padded = np.concatenate([profile[-pad:], profile, profile[:pad]])
        smooth = np.convolve(padded, kernel, mode="valid")[:n_ang]

        means: list[float] = []
        for sector_index in range(self.n_bits):
            lo = int(round(phase + sector_index * step))
            hi = int(round(phase + (sector_index + 1) * step))
            idx = np.arange(lo, hi) % n_ang
            if len(idx) == 0:
                return None
            sector_pixels = strip[:, idx].reshape(-1)
            sector_pixels = sector_pixels[np.isfinite(sector_pixels)]
            if sector_pixels.size == 0:
                return None
            means.append(float(np.median(sector_pixels)))
        return np.array(means, dtype=np.float64)

    def _normalize_sector_means(self, means: np.ndarray) -> tuple[np.ndarray, float, float, float]:
        lo = float(np.percentile(means, 15))
        hi = float(np.percentile(means, 85))
        spread = hi - lo
        if spread < 6.0:
            return np.zeros_like(means), spread, lo, hi
        normalized = np.clip((means - lo) / (spread + 1e-9), 0.0, 1.0)
        return normalized, spread, lo, hi

    def _sector_stats_for_phase(
        self,
        strip: np.ndarray,
        phase: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float] | None:
        means = self._sector_means_for_phase(strip, phase)
        if means is None:
            return None
        sector_white, spread, lo, hi = self._normalize_sector_means(means)
        if spread < 6.0:
            return None

        threshold = 0.5 * (lo + hi)
        n_ang = strip.shape[1]
        step = n_ang / self.n_bits
        white_fractions: list[float] = []
        bit_purities: list[float] = []
        bit_fragmentation: list[float] = []
        for sector_index in range(self.n_bits):
            core_lo = int(round(phase + (sector_index + self.bit_core_margin_fraction) * step))
            core_hi = int(round(phase + (sector_index + 1.0 - self.bit_core_margin_fraction) * step))
            if core_hi <= core_lo:
                core_lo = int(round(phase + sector_index * step))
                core_hi = int(round(phase + (sector_index + 1) * step))
            idx = np.arange(core_lo, core_hi) % n_ang
            if len(idx) == 0:
                return None
            sector_band = strip[:, idx]
            sector_pixels = sector_band.reshape(-1)
            sector_pixels = sector_pixels[np.isfinite(sector_pixels)]
            if sector_pixels.size == 0:
                return None

            column_white = np.nanmean(sector_band >= threshold, axis=0)
            column_white = column_white[np.isfinite(column_white)]
            if column_white.size == 0:
                return None

            white_fraction = float(np.mean(column_white))
            white_fractions.append(white_fraction)
            bit_purities.append(float(np.mean(np.maximum(column_white, 1.0 - column_white))))

            if column_white.size <= 1:
                bit_fragmentation.append(0.0)
            else:
                angular_binary = column_white >= 0.5
                fragmentation = float(np.mean(angular_binary[1:] != angular_binary[:-1]))
                bit_fragmentation.append(fragmentation)

        white_fraction_array = np.array(white_fractions, dtype=np.float64)
        purity = np.array(bit_purities, dtype=np.float64)
        fragmentation = np.array(bit_fragmentation, dtype=np.float64)
        return sector_white, white_fraction_array, purity, fragmentation, spread

    def _score_oriented_code(self, sector_white: np.ndarray, bits: np.ndarray) -> float:
        return float(np.mean((sector_white - bits) ** 2))

    @staticmethod
    def _sample_annulus_profile(
        bw: np.ndarray,
        radii: np.ndarray,
        sample_n: int,
        reducer: str = "mean",
    ) -> np.ndarray | None:
        sz = bw.shape[0]
        center = sz / 2.0
        profile: list[float] = []
        for sample_index in range(sample_n):
            angle = 2.0 * np.pi * sample_index / sample_n
            cos_a = float(np.cos(angle))
            sin_a = float(np.sin(angle))
            samples: list[float] = []
            for radius in radii:
                x = int(round(center + float(radius) * cos_a))
                y = int(round(center + float(radius) * sin_a))
                if 0 <= x < sz and 0 <= y < sz:
                    samples.append(1.0 if bw[y, x] > 0 else 0.0)
            if not samples:
                return None
            values = np.array(samples, dtype=np.float64)
            if reducer == "upper_half_mean":
                split_index = values.size // 2
                values = np.sort(values)[split_index:]
            profile.append(float(np.mean(values)))
        return np.array(profile, dtype=np.float64)

    def _validate_rectified(self, patch_gray: np.ndarray, sample_n: int = 48) -> bool:
        sz = patch_gray.shape[0]
        r1 = sz / 6.0
        _, bw = cv2.threshold(patch_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        center_profile = self._sample_annulus_profile(
            bw,
            np.linspace(0.12 * r1, 0.82 * r1, 7),
            sample_n,
        )
        code_profile = self._sample_annulus_profile(
            bw,
            np.linspace(2.20 * r1, 3.15 * r1, 11),
            sample_n,
            reducer="upper_half_mean",
        )
        if center_profile is None or code_profile is None:
            return False

        best_guard_stats: tuple[float, float, float] | None = None
        for inner_scale, outer_scale, n_radii in (
            (1.20, 1.50, 7),
            (1.25, 1.60, 7),
            (1.30, 1.60, 7),
            (1.35, 1.65, 7),
        ):
            guard_profile = self._sample_annulus_profile(
                bw,
                np.linspace(inner_scale * r1, outer_scale * r1, n_radii),
                sample_n,
            )
            if guard_profile is None:
                continue
            guard_black_ratio = float(np.mean(1.0 - guard_profile))
            guard_mixed_ratio = float(np.mean((guard_profile > 0.10) & (guard_profile < 0.90)))
            guard_bright_ratio = float(np.mean(guard_profile > 0.22))
            current_stats = (guard_black_ratio, guard_mixed_ratio, guard_bright_ratio)
            if best_guard_stats is None:
                best_guard_stats = current_stats
                continue
            if current_stats[0] > best_guard_stats[0]:
                best_guard_stats = current_stats

        if best_guard_stats is None:
            return False

        center_white_ratio = float(np.mean(center_profile))
        center_mixed_ratio = float(np.mean((center_profile > 0.10) & (center_profile < 0.90)))
        center_dark_ratio = float(np.mean(center_profile < 0.78))
        guard_black_ratio, guard_mixed_ratio, guard_bright_ratio = best_guard_stats
        white_code_angles = int(np.count_nonzero(code_profile >= 0.30))
        black_code_angles = int(np.count_nonzero(code_profile <= 0.15))
        code_contrast = float(np.max(code_profile) - np.min(code_profile))

        if center_white_ratio < 0.90:
            return False
        if center_mixed_ratio > 0.12:
            return False
        if center_dark_ratio > 0.10:
            return False
        if guard_black_ratio < 0.72:
            return False
        if guard_mixed_ratio > 0.25:
            return False
        if guard_bright_ratio > 0.35:
            return False
        if white_code_angles < 2 or black_code_angles < 2:
            return False
        if code_contrast < 0.18:
            return False
        return True

    def _decode_strip(self, strip: np.ndarray) -> _DecodeResult | None:
        if self._candidate_codes is not None:
            result = self._decode_with_codebook(strip)
            if result is not None:
                return result
        return self._decode_without_codebook(strip)

    def _collect_strip_variant_results(self, patch_gray: np.ndarray) -> list[_DecodeResult]:
        results: list[_DecodeResult] = []
        for inner_scale, outer_scale, n_radii in self.strip_variants:
            strip = self._sample_polar_strip(
                patch_gray,
                inner_scale=inner_scale,
                outer_scale=outer_scale,
                n_radii=n_radii,
            )
            if strip is None:
                continue
            result = self._decode_strip(strip)
            if result is not None:
                results.append(result)
        return results

    def _decode_with_baseline_patch(self, patch_gray: np.ndarray) -> _DecodeResult | None:
        baseline_patch = patch_gray
        if patch_gray.shape[0] != 200 or patch_gray.shape[1] != 200:
            baseline_patch = cv2.resize(patch_gray, (200, 200), interpolation=cv2.INTER_LINEAR)

        result = super()._decode_patch(baseline_patch)
        if result is None:
            return None

        target_id, raw_code, code_bits, confidence, pattern = result
        if self.valid_ids is not None and int(target_id) not in self.valid_ids:
            snapped_id, snap_distance = self._snap_to_valid_id(int(target_id))
            if snapped_id is None:
                return None
            target_id = snapped_id
            confidence = max(0.0, float(confidence) - 0.05 * float(snap_distance))

        return _DecodeResult(
            target_id=int(target_id),
            raw_code=int(raw_code),
            code_bits=[int(bit) for bit in code_bits],
            confidence=float(confidence),
            pattern=str(pattern),
            score=float(-confidence),
            anchor_score=0.0,
            second_best_gap=1.0,
        )

    def _filter_explicit_valid_ids(self, detections: list[Detection]) -> list[Detection]:
        if self.valid_ids is None:
            return detections
        return [det for det in detections if det.target_id in self.valid_ids]

    def _baseline_detect(self, image_bgr: np.ndarray) -> list[Detection]:
        valid_id_range = None
        if self.valid_ids:
            valid_id_range = (min(self.valid_ids), max(self.valid_ids))
        detector = BaseCCTDetector(n_bits=self.n_bits, valid_id_range=valid_id_range)
        detections = detector.detect(image_bgr)
        return self._filter_explicit_valid_ids(detections)

    @staticmethod
    def _merge_detection_sets(
        primary: list[Detection],
        supplemental: list[Detection],
    ) -> tuple[list[Detection], list[Detection]]:
        merged = list(primary)
        added: list[Detection] = []
        for det in supplemental:
            center = np.array(det.center, dtype=np.float64)
            duplicate = False
            for existing in merged:
                existing_center = np.array(existing.center, dtype=np.float64)
                det_radius = max(det.ellipse[1]) / 2.0
                existing_radius = max(existing.ellipse[1]) / 2.0
                merge_radius = max(35.0, min(det_radius, existing_radius) * 2.0)
                if np.linalg.norm(center - existing_center) < merge_radius:
                    duplicate = True
                    break
            if duplicate:
                continue
            merged.append(det)
            added.append(det)
        return merged, added

    @staticmethod
    def _dedup_by_id(detections: list[Detection]) -> list[Detection]:
        best_by_id: dict[int, Detection] = {}
        for det in detections:
            current = best_by_id.get(det.target_id)
            if current is None or det.confidence > current.confidence:
                best_by_id[det.target_id] = det
        return sorted(best_by_id.values(), key=lambda det: (det.center[1], det.center[0]))

    def _phase_gate_failures(
        self,
        white_fractions: np.ndarray,
        bit_purity: np.ndarray,
        bit_fragmentation: np.ndarray,
        anchor_score: float,
    ) -> tuple[list[str], int]:
        failures: list[str] = []
        if np.min(bit_purity) < self.min_bit_purity:
            failures.append("min_bit_purity")
        if float(np.mean(bit_purity)) < self.mean_bit_purity:
            failures.append("mean_bit_purity")
        if float(np.max(bit_fragmentation)) > 0.22:
            failures.append("max_fragmentation")
        if float(np.mean(bit_fragmentation)) > 0.10:
            failures.append("mean_fragmentation")
        mixed_bits = int(np.count_nonzero((white_fractions > self.mixed_bit_low) & (white_fractions < self.mixed_bit_high)))
        if mixed_bits > self.max_mixed_bits:
            failures.append("mixed_bits")
        if anchor_score < self.min_anchor_score:
            failures.append("anchor_score")
        return failures, mixed_bits

    @staticmethod
    def _make_debug_point(
        cx: float,
        cy: float,
        *,
        label: str | None = None,
        confidence: float | None = None,
    ) -> dict[str, Any]:
        point: dict[str, Any] = {"x": float(cx), "y": float(cy)}
        if label:
            point["label"] = label
        if confidence is not None:
            point["confidence"] = float(confidence)
        return point

    def _decode_with_codebook(self, strip: np.ndarray) -> _DecodeResult | None:
        n_ang = strip.shape[1]
        step = n_ang / self.n_bits
        n_phases = max(1, int(round(step)))

        candidate_codes = self._candidate_codes
        if not candidate_codes:
            return None

        best_by_canonical_id: dict[int, tuple[float, _CodeCandidate, float, float]] = {}

        for phase in range(n_phases):
            sector_stats = self._sector_stats_for_phase(strip, phase)
            if sector_stats is None:
                continue
            sector_white, white_fractions, bit_purity, bit_fragmentation, spread = sector_stats
            anchor_score = float(np.mean(white_fractions[list(self.anchor_bits)]))
            gate_failures, _ = self._phase_gate_failures(
                white_fractions,
                bit_purity,
                bit_fragmentation,
                anchor_score,
            )
            if gate_failures:
                continue

            for candidate in candidate_codes:
                bits = np.array(candidate.oriented_bits, dtype=np.float64)
                anchor_penalty = 0.0
                if self.require_anchor_bits:
                    anchor_penalty = float(np.mean((1.0 - white_fractions[list(self.anchor_bits)]) ** 2))
                fraction_error = np.where(bits > 0.5, 1.0 - white_fractions, white_fractions)
                if int(np.count_nonzero(fraction_error > self.fraction_error_threshold)) > self.max_fraction_error_bits:
                    continue
                fit_score = self._score_oriented_code(sector_white, bits)
                score = (
                    fit_score
                    + 0.75 * float(np.mean(fraction_error))
                    + 0.25 * float(np.mean(bit_fragmentation))
                    + 0.35 * anchor_penalty
                    - 0.04 * anchor_score
                )
                current = best_by_canonical_id.get(candidate.canonical_id)
                if current is None or score < current[0]:
                    best_by_canonical_id[candidate.canonical_id] = (score, candidate, anchor_score, spread)

        if not best_by_canonical_id:
            return None

        ranked = sorted(best_by_canonical_id.values(), key=lambda item: item[0])
        score, candidate, anchor_score, spread = ranked[0]
        second_score = ranked[1][0] if len(ranked) > 1 else float("inf")
        bits = np.array(candidate.oriented_bits, dtype=np.int32)
        if second_score < float("inf") and (second_score - score) < 0.012:
            return None
        confidence = float(max(0.0, second_score - score) if second_score < float("inf") else (0.15 - score))
        return _DecodeResult(
            target_id=int(candidate.canonical_id),
            raw_code=int(candidate.oriented_raw),
            code_bits=[int(b) for b in bits.tolist()],
            confidence=confidence,
            pattern="".join(str(int(b)) for b in bits.tolist()),
            score=float(score),
            anchor_score=float(anchor_score),
            second_best_gap=float(second_score - score) if second_score < float("inf") else 1.0,
        )

    def _decode_without_codebook(self, strip: np.ndarray) -> _DecodeResult | None:
        n_ang = strip.shape[1]
        step = n_ang / self.n_bits
        n_phases = max(1, int(round(step)))
        best: tuple[float, int, np.ndarray, float] | None = None
        second_score = float("inf")

        for phase in range(n_phases):
            sector_stats = self._sector_stats_for_phase(strip, phase)
            if sector_stats is None:
                continue
            sector_white, white_fractions, bit_purity, bit_fragmentation, spread = sector_stats
            anchor_score = float(np.mean(white_fractions[list(self.anchor_bits)]))
            gate_failures, _ = self._phase_gate_failures(
                white_fractions,
                bit_purity,
                bit_fragmentation,
                anchor_score,
            )
            if gate_failures:
                continue
            bits = (sector_white >= 0.5).astype(np.int32)
            transitions = int(np.sum(bits != np.roll(bits, 1)))
            if transitions < 4:
                continue
            fraction_error = np.where(bits > 0.5, 1.0 - white_fractions, white_fractions)
            if int(np.count_nonzero(fraction_error > self.fraction_error_threshold)) > self.max_fraction_error_bits:
                continue
            score = float(
                np.mean(np.minimum(sector_white, 1.0 - sector_white))
                + 0.75 * np.mean(fraction_error)
                + 0.25 * np.mean(bit_fragmentation)
                - 0.15 * anchor_score
            )
            raw_id = sum((1 << index) for index, bit in enumerate(bits.tolist()) if bit)
            if score < second_score:
                if best is None or score < best[0]:
                    if best is not None:
                        second_score = best[0]
                    best = (score, raw_id, bits, anchor_score)
                else:
                    second_score = score

        if best is None:
            return None

        score, raw_id, bits, anchor_score = best
        canonical_id = canonical_code(int(raw_id), self.n_bits)
        snapped_id, snap_distance = self._snap_to_valid_id(canonical_id)
        if snapped_id is None:
            return None
        if second_score < float("inf") and (second_score - score) < 0.012:
            return None
        confidence = float(max(0.0, second_score - score) if second_score < float("inf") else (0.15 - score))
        return _DecodeResult(
            target_id=int(snapped_id),
            raw_code=int(raw_id),
            code_bits=[int(b) for b in bits.tolist()],
            confidence=confidence,
            pattern="".join(str(int(b)) for b in bits.tolist()),
            score=float(score + (0.02 * snap_distance if np.isfinite(snap_distance) else 0.0)),
            anchor_score=float(anchor_score),
            second_best_gap=float(second_score - score) if second_score < float("inf") else 1.0,
        )

    def _decode_patch(self, patch_gray: np.ndarray) -> _DecodeResult | None:
        strip_results = self._collect_strip_variant_results(patch_gray)
        baseline_result = self._decode_with_baseline_patch(patch_gray)

        if not strip_results:
            return baseline_result

        strip_votes = Counter(result.target_id for result in strip_results)
        top_target_id, top_votes = strip_votes.most_common(1)[0]
        best_strip_result = max(
            (result for result in strip_results if result.target_id == top_target_id),
            key=lambda result: (result.confidence, result.second_best_gap, result.anchor_score),
        )

        if baseline_result is not None and baseline_result.target_id == top_target_id:
            if baseline_result.confidence >= best_strip_result.confidence:
                return baseline_result
            return best_strip_result

        if top_votes >= 2:
            return best_strip_result

        if baseline_result is not None:
            return baseline_result

        return None

    def detect_with_diagnostics(
        self,
        image_bgr: np.ndarray,
    ) -> tuple[list[Detection], DetectionDiagnostics]:
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)
        candidates = self._find_candidates(gray)
        candidates.sort(key=lambda ellipse: max(ellipse[1]), reverse=True)

        stage_points: dict[str, list[dict[str, Any]]] = {
            "candidates": [],
            "reject_rectify": [],
            "validated": [],
            "reject_validate": [],
            "decoded": [],
            "reject_decode": [],
            "spatial_dedup": [],
            "baseline": [],
            "hybrid_added": [],
            "final": [],
        }

        detections: list[Detection] = []
        used_centers: list[np.ndarray] = []
        for ell in candidates:
            (cx, cy), (aw, ah), _ = ell
            radius = max(aw, ah) / 2.0
            stage_points["candidates"].append(self._make_debug_point(cx, cy))
            patch = self._rectify_patch(gray, ell, out_size=220)
            if patch is None:
                stage_points["reject_rectify"].append(self._make_debug_point(cx, cy))
                continue
            if not self._validate_rectified(patch):
                stage_points["reject_validate"].append(self._make_debug_point(cx, cy))
                continue
            stage_points["validated"].append(self._make_debug_point(cx, cy))
            result = self._decode_patch(patch)
            if result is None:
                stage_points["reject_decode"].append(self._make_debug_point(cx, cy))
                continue
            stage_points["decoded"].append(
                self._make_debug_point(
                    cx,
                    cy,
                    label=f"{result.target_id}",
                    confidence=result.confidence,
                )
            )

            center = np.array([cx, cy], dtype=np.float64)
            duplicate = False
            for idx, used_center in enumerate(used_centers):
                det_radius = max(detections[idx].ellipse[1]) / 2.0
                dedup_radius = max(20.0, max(radius, det_radius) * 4.0)
                if np.linalg.norm(center - used_center) < dedup_radius:
                    if result.confidence > detections[idx].confidence:
                        detections[idx] = Detection(
                            target_id=result.target_id,
                            center=(float(cx), float(cy)),
                            ellipse=ell,
                            confidence=float(result.confidence),
                            code_bits=result.code_bits,
                            raw_code=result.raw_code,
                            pattern=result.pattern,
                        )
                        used_centers[idx] = center
                    duplicate = True
                    break
            if duplicate:
                continue

            detections.append(
                Detection(
                    target_id=result.target_id,
                    center=(float(cx), float(cy)),
                    ellipse=ell,
                    confidence=float(result.confidence),
                    code_bits=result.code_bits,
                    raw_code=result.raw_code,
                    pattern=result.pattern,
                )
            )
            used_centers.append(center)

        for det in detections:
            stage_points["spatial_dedup"].append(
                self._make_debug_point(
                    det.center[0],
                    det.center[1],
                    label=str(det.target_id),
                    confidence=det.confidence,
                )
            )

        detections.sort(key=lambda det: (det.center[1], det.center[0]))
        detections = self._filter_explicit_valid_ids(detections)
        detections = self._dedup_by_id(detections)

        baseline_detections = self._baseline_detect(image_bgr)
        for det in baseline_detections:
            stage_points["baseline"].append(
                self._make_debug_point(
                    det.center[0],
                    det.center[1],
                    label=str(det.target_id),
                    confidence=det.confidence,
                )
            )

        detections, hybrid_added = self._merge_detection_sets(baseline_detections, detections)
        detections = self._dedup_by_id(detections)

        for det in hybrid_added:
            stage_points["hybrid_added"].append(
                self._make_debug_point(
                    det.center[0],
                    det.center[1],
                    label=str(det.target_id),
                    confidence=det.confidence,
                )
            )

        for det in detections:
            stage_points["final"].append(
                self._make_debug_point(
                    det.center[0],
                    det.center[1],
                    label=str(det.target_id),
                    confidence=det.confidence,
                )
            )

        counts = {stage_name: len(points) for stage_name, points in stage_points.items()}
        diagnostics = DetectionDiagnostics(stage_points=stage_points, counts=counts)
        return detections, diagnostics

    def detect(self, image_bgr: np.ndarray) -> list[Detection]:
        detections, _ = self.detect_with_diagnostics(image_bgr)
        return detections
