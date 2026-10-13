from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Dict, List, Sequence, Set

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pyceres
try:
    import rerun as rr
except ModuleNotFoundError:
    rr = None

from cct_detect.detector import (
    DETECTOR_VERSION,
    CCTDetector as _CCTDetector,
    cyclic_hamming_distance,
)


@dataclass
class ImageDetections:
    image_path: Path
    image_index: int
    width: int
    height: int
    detections: Dict[int, np.ndarray]
    # Immutable measurement/provenance used by calibrated centre refinement.
    # Legacy four-column caches leave these empty and are upgraded lazily.
    raw_detections: Dict[int, np.ndarray] = field(default_factory=dict)
    refinement_ellipses: Dict[int, tuple] = field(default_factory=dict)
    refinement_status: Dict[int, Dict[str, Any]] = field(default_factory=dict)
    raw_cache_verified: bool = False


@dataclass
class CalibrationState:
    intrinsics: np.ndarray
    poses: Dict[int, np.ndarray]
    points: Dict[int, np.ndarray]
    seed_image_indices: List[int]
    inlier_track_ids: List[int]


@dataclass
class CameraCalibrationResult:
    camera_name: str
    image_size: tuple[int, int]
    used_images: int
    registered_images: int
    tracks: int
    observations: int
    summary: str
    intrinsics: np.ndarray
    mean_reprojection_error: float
    rms_reprojection_error: float
    mean_object_space_error: float
    rms_object_space_error: float
    iterations: int
    initial_cost: float
    final_cost: float
    report_path: Path
    convergence_plot_path: Path
    rerun_recording_path: Path
    scene_data_path: Path
    targets_3d_path: Path
    cameras_path: Path
    colmap_dir_path: Path
    detections_dir_path: Path
    target_detections_path: Path


@dataclass
class SolverDiagnostics:
    summary: str
    iterations: int
    initial_cost: float
    final_cost: float
    cost_history: List[float]
    adjustment: Any | None = None


@dataclass
class DetectionOnlyResult:
    camera_name: str
    total_images: int
    used_images: int
    detections_dir_path: Path
    target_detections_path: Path


def parse_valid_ids_file(path: Path) -> Set[int]:
    tokens = path.read_text(encoding="utf-8").replace(",", " ").split()
    valid_ids = {int(token) for token in tokens}
    if not valid_ids:
        raise ValueError(f"No valid IDs found in {path}")
    return valid_ids


def resolve_valid_ids(valid_ids_file: Path | None, valid_id_max: int | None) -> Set[int] | None:
    if valid_ids_file is not None:
        return parse_valid_ids_file(valid_ids_file)
    if valid_id_max is not None:
        return set(range(valid_id_max + 1))
    return None


def load_targets3d(path: Path) -> Dict[int, np.ndarray]:
    rows = [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows:
        raise ValueError(f"No 3D target rows found in {path}")

    header = rows[0].split("\t")
    if {"Label actual", "X", "Y", "Z"}.issubset(header):
        targets: Dict[int, np.ndarray] = {}
        reader = csv.DictReader(rows, delimiter="\t")
        for row in reader:
            actual = row.get("Label actual")
            x_value = row.get("X")
            y_value = row.get("Y")
            z_value = row.get("Z")
            if not actual or x_value is None or y_value is None or z_value is None:
                continue
            target_id = int(actual)
            targets[target_id] = np.array(
                [float(x_value), float(y_value), float(z_value)],
                dtype=np.float64,
            )
        if not targets:
            raise ValueError(f"No Metashape-style targets parsed from {path}")
        return targets

    targets = {}
    for line in rows:
        if line.lstrip().startswith("#"):
            continue
        parts = line.replace(",", " ").split()
        if len(parts) < 4:
            continue
        target_id = int(parts[0])
        targets[target_id] = np.array([float(parts[1]), float(parts[2]), float(parts[3])], dtype=np.float64)
    if not targets:
        raise ValueError(f"No simple target rows parsed from {path}")
    return targets


def normalize_target_id(
    raw_id: int,
    valid_ids: Set[int] | None,
    max_hamming_distance: int,
    n_bits: int = 14,
) -> tuple[int | None, bool]:
    if valid_ids is None:
        return raw_id, False
    if raw_id in valid_ids:
        return raw_id, False
    if max_hamming_distance <= 0:
        return None, False

    # Decoded IDs are canonical (minimum-rotation) ring codes, so the code
    # distance must be minimised over rotations; a plain XOR of two canonical
    # forms is not the physical bit-error count.
    best_id: int | None = None
    best_distance: int | None = None
    is_ambiguous = False
    for candidate in valid_ids:
        distance = cyclic_hamming_distance(int(raw_id), int(candidate), n_bits)
        if best_distance is None or distance < best_distance:
            best_id = candidate
            best_distance = distance
            is_ambiguous = False
        elif distance == best_distance:
            is_ambiguous = True

    if best_id is None or best_distance is None or is_ambiguous or best_distance > max_hamming_distance:
        return None, False
    return best_id, True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Calibrate a single camera from coded-target detections. Calibration runs the "
                    "combined pipeline (run_combined.py) as its one-camera case, so it produces the same "
                    "adjustment diagnostics and PDF report; any of its options (e.g. --checkpoints, "
                    "--observation-sigma-px, --evaluate-detections, --no-pdf-report) may be added."
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("data"),
        help="Root directory that contains the photos_* folders.",
    )
    parser.add_argument(
        "--image-root",
        type=Path,
        default=None,
        help="Optional directory containing image files directly for a single-camera dataset.",
    )
    parser.add_argument(
        "--camera",
        action="append",
        dest="cameras",
        help="Camera name to calibrate, for example camera_0. Repeat to run multiple cameras.",
    )
    parser.add_argument(
        "--min-detections",
        type=int,
        default=10,
        help="Keep only images with more than this many unique detections.",
    )
    parser.add_argument(
        "--min-shared",
        type=int,
        default=8,
        help="Minimum number of shared tracks required to seed or register a view.",
    )
    parser.add_argument(
        "--max-reprojection-error",
        type=float,
        default=12.0,
        help="Drop initialized observations above this pixel error before bundle adjustment.",
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=1000,
        help="Safety cap on pyceres iterations. Early termination is driven by loss convergence.",
    )
    parser.add_argument(
        "--huber-delta",
        type=float,
        default=3.0,
        help="Huber loss delta in pixels.",
    )
    parser.add_argument(
        "--loss-relative-tolerance",
        type=float,
        default=1e-5,
        help="Relative loss-change threshold for convergence-based stopping.",
    )
    parser.add_argument(
        "--loss-patience",
        type=int,
        default=5,
        help="Terminate after this many consecutive successful iterations below the relative loss threshold.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("calibration_output"),
        help="Directory for convergence plots, pose visualizations, and JSON summaries.",
    )
    parser.add_argument(
        "--show-3d",
        action="store_true",
        help="Open a Rerun viewer with oriented camera poses and target-center points after calibration.",
    )
    parser.add_argument(
        "--valid-ids-file",
        type=Path,
        default=None,
        help="Optional text file with whitespace- or comma-separated valid target IDs.",
    )
    parser.add_argument(
        "--valid-id-max",
        type=int,
        default=None,
        help="Optional maximum valid target ID. If set, valid IDs are assumed to be 0..N.",
    )
    parser.add_argument(
        "--targets3d",
        type=Path,
        default=None,
        help="Optional file with known 3D target coordinates. Supports reference.txt Metashape export format.",
    )
    parser.add_argument(
        "--max-id-hamming-distance",
        type=int,
        default=0,
        help="If valid IDs are provided, snap decoded IDs to the nearest valid ID only when the Hamming distance is at most this value.",
    )
    parser.add_argument(
        "--detect-only",
        action="store_true",
        help="Run detection only, skip pose initialization and bundle adjustment, and export detections.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=0,
        help="Parallel image workers for detection, centre refinement and detection evaluation "
             "(0 = automatic: up to 8, limited by the CPU count; 1 = sequential).",
    )
    parser.add_argument(
        "--force-detections",
        action="store_true",
        help="Re-run CCT detection even if cached target_detections.txt exists.",
    )
    return parser.parse_known_args()


def iter_camera_images(data_root: Path, camera_name: str, image_root: Path | None = None) -> List[Path]:
    if image_root is not None:
        return sorted(
            path for path in image_root.iterdir()
            if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
        )

    image_paths: List[Path] = []
    for folder in sorted(data_root.glob("photos_*")):
        if not folder.is_dir():
            continue
        image_paths.extend(sorted(folder.glob(f"{camera_name}_*.jpg")))
    return image_paths


def merge_duplicate_detections(points: Sequence[np.ndarray], merge_radius: float) -> np.ndarray | None:
    if len(points) == 1:
        return points[0]
    stacked = np.vstack(points)
    center = np.mean(stacked, axis=0)
    if np.max(np.linalg.norm(stacked - center, axis=1)) <= merge_radius:
        return center
    return None


def save_annotated_detection_image(output_path: Path, image: np.ndarray) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), image):
        raise RuntimeError(f"Failed to save annotated detection image to {output_path}")


REFINEMENT_CACHE_SCHEMA = 1
REFINEMENT_CACHE_NAME = "target_refinement_cache.json"


def _image_freshness(path: Path) -> Dict[str, int] | None:
    try:
        stat = path.stat()
    except OSError:
        return None
    return {"size": int(stat.st_size), "mtime_ns": int(stat.st_mtime_ns)}


def load_refinement_cache(
    images: Sequence[ImageDetections],
    camera_output_dir: Path,
) -> int:
    """Attach versioned raw-centre/ellipse evidence to cached observations."""
    # Always establish the legacy fallback, including when a companion cache
    # exists but is unreadable or from a newer schema.
    for image in images:
        image.raw_detections = {
            int(target_id): np.asarray(point, dtype=np.float64).copy()
            for target_id, point in image.detections.items()
        }
    path = camera_output_dir / REFINEMENT_CACHE_NAME
    if not path.exists():
        return 0
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return 0
    if payload.get("schema_version") != REFINEMENT_CACHE_SCHEMA:
        return 0
    cached_version = payload.get("detector_version")
    if cached_version != DETECTOR_VERSION:
        print(
            f"  warning: cached detections in {camera_output_dir} were produced by detector "
            f"version {cached_version or 'unknown (pre-versioning)'}; the current detector is "
            f"{DETECTOR_VERSION}. Use --force-detections to redetect.",
            flush=True,
        )
    records = payload.get("images", {})
    attached = 0
    for image in images:
        cached = records.get(image.image_path.name)
        if not isinstance(cached, dict):
            continue
        if cached.get("freshness") != _image_freshness(image.image_path):
            continue
        if [image.width, image.height] != cached.get("image_size"):
            continue
        valid_raw_ids: set[int] = set()
        for target_text, item in cached.get("observations", {}).items():
            try:
                target_id = int(target_text)
                if target_id not in image.detections:
                    continue
                raw = np.asarray(item["raw_center"], dtype=np.float64)
                ellipse_values = item.get("ellipse")
                if raw.shape != (2,) or not np.all(np.isfinite(raw)):
                    continue
                image.raw_detections[target_id] = raw
                valid_raw_ids.add(target_id)
                if ellipse_values is not None and len(ellipse_values) == 5:
                    cx, cy, width, height, angle = map(float, ellipse_values)
                    if np.all(np.isfinite([cx, cy, width, height, angle])) and min(width, height) > 0:
                        image.refinement_ellipses[target_id] = (
                            (cx, cy), (width, height), angle,
                        )
                status = item.get("refinement")
                if isinstance(status, dict):
                    image.refinement_status[target_id] = dict(status)
                attached += int(target_id in image.refinement_ellipses)
            except (KeyError, TypeError, ValueError):
                continue
        image.raw_cache_verified = set(image.detections).issubset(valid_raw_ids)
    return attached


def save_refinement_cache(
    images: Sequence[ImageDetections],
    camera_output_dir: Path,
    *,
    merge_existing: bool = False,
) -> Path:
    """Atomically persist immutable measurements and reusable ellipse evidence."""
    path = camera_output_dir / REFINEMENT_CACHE_NAME
    previous_images: Dict[str, Any] = {}
    # A fresh write follows a detection run with the current detector.  A
    # merge only updates refinement evidence of already cached detections, so
    # it keeps the version of the detector that produced them.
    detector_version: str | None = None if merge_existing else DETECTOR_VERSION
    if merge_existing and path.exists():
        try:
            previous = json.loads(path.read_text(encoding="utf-8"))
            if previous.get("schema_version") == REFINEMENT_CACHE_SCHEMA:
                previous_images = dict(previous.get("images", {}))
                detector_version = previous.get("detector_version")
        except (OSError, ValueError, TypeError):
            previous_images = {}
    for image in images:
        observations: Dict[str, Any] = {}
        for target_id, active in sorted(image.detections.items()):
            raw = image.raw_detections.get(target_id, active)
            ellipse = image.refinement_ellipses.get(target_id)
            ellipse_values = None
            if ellipse is not None:
                ellipse_values = [
                    float(ellipse[0][0]), float(ellipse[0][1]),
                    float(ellipse[1][0]), float(ellipse[1][1]), float(ellipse[2]),
                ]
            observations[str(int(target_id))] = {
                "raw_center": [float(raw[0]), float(raw[1])],
                "active_center": [float(active[0]), float(active[1])],
                "ellipse": ellipse_values,
                "refinement": dict(image.refinement_status.get(target_id, {})),
            }
        image_record = {
            "image_size": [int(image.width), int(image.height)],
            "freshness": _image_freshness(image.image_path),
            "observations": observations,
        }
        previous_record = previous_images.get(image.image_path.name)
        if (
            merge_existing and isinstance(previous_record, dict)
            and previous_record.get("image_size") == image_record["image_size"]
            and previous_record.get("freshness") == image_record["freshness"]
        ):
            merged_observations = dict(previous_record.get("observations", {}))
            merged_observations.update(observations)
            image_record["observations"] = merged_observations
        previous_images[image.image_path.name] = image_record
    payload = {
        "schema_version": REFINEMENT_CACHE_SCHEMA,
        "centre_estimator": "calibrated-concentric-conic-v1",
        "detector_version": detector_version,
        "images": previous_images,
    }
    camera_output_dir.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")
    temporary.replace(path)
    return path


def save_target_detections(
    images: Sequence[ImageDetections],
    camera_name: str,
    output_dir: Path,
) -> Path:
    camera_output_dir = output_dir / camera_name
    camera_output_dir.mkdir(parents=True, exist_ok=True)

    output_path = camera_output_dir / "target_detections.txt"
    lines = ["image_name\ttarget_id\tx_image\ty_image"]
    for image in images:
        for target_id, point in sorted(image.detections.items()):
            lines.append(
                f"{image.image_path.name}\t{int(target_id)}\t{float(point[0]):.6f}\t{float(point[1]):.6f}"
            )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    save_refinement_cache(images, camera_output_dir)
    return output_path


def load_cached_detections(
    cache_path: Path,
    image_root: Path,
) -> List[ImageDetections] | None:
    if not cache_path.exists():
        return None

    try:
        lines = cache_path.read_text(encoding="utf-8").splitlines()
    except Exception:
        return None

    if not lines or not lines[0].startswith("image_name"):
        return None

    from collections import OrderedDict

    grouped: Dict[str, Dict[int, np.ndarray]] = OrderedDict()
    for line in lines[1:]:
        line = line.strip()
        if not line:
            continue
        parts = line.split("\t")
        if len(parts) < 4:
            continue
        image_name, target_id, x_coord, y_coord = parts[0], parts[1], parts[2], parts[3]
        grouped.setdefault(image_name, {})[int(target_id)] = np.array(
            [float(x_coord), float(y_coord)],
            dtype=np.float64,
        )

    result: List[ImageDetections] = []
    width, height = 0, 0
    for image_index, (image_name, detections) in enumerate(grouped.items()):
        image_path = image_root / image_name
        if width == 0 and image_path.exists():
            image = cv2.imread(str(image_path))
            if image is not None:
                height, width = image.shape[:2]
        result.append(
            ImageDetections(
                image_path=image_path,
                image_index=image_index,
                width=width,
                height=height,
                detections=detections,
            )
        )

    evidence_count = load_refinement_cache(result, cache_path.parent)
    print(
        f"loaded {len(result)} images from cache: {cache_path} "
        f"({evidence_count} detections ready for centre refinement)",
        flush=True,
    )
    return result


def filter_images_for_bundle(
    images: Sequence[ImageDetections],
    min_detections: int,
) -> List[ImageDetections]:
    return [image for image in images if len(image.detections) > min_detections]


def build_combined_detection_lines(
    images: Sequence[ImageDetections],
    camera_name: str,
) -> List[str]:
    lines: List[str] = []
    for image in images:
        for target_id, point in sorted(image.detections.items()):
            lines.append(
                f"{camera_name}\t{image.image_path.name}\t{int(target_id)}\t{float(point[0]):.6f}\t{float(point[1]):.6f}"
            )
    return lines


def save_combined_target_detections(
    detections_by_camera: Sequence[tuple[str, Sequence[ImageDetections]]],
    output_dir: Path,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "target_detections.txt"
    lines = ["camera_name\timage_name\ttarget_id\tx_image\ty_image"]
    for camera_name, images in detections_by_camera:
        lines.extend(build_combined_detection_lines(images, camera_name))
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output_path


_WORKER_DETECTORS: Dict[Any, Any] = {}


def _detector_for(codebook: Set[int] | None):
    """One detector per process and codebook (codebook templates are built once)."""
    key = frozenset(codebook) if codebook is not None else None
    if key not in _WORKER_DETECTORS:
        _WORKER_DETECTORS[key] = _CCTDetector(n_bits=14, codebook=codebook)
    return _WORKER_DETECTORS[key]


def _init_detection_worker() -> None:
    # Parallelism comes from the process pool; avoid oversubscribing cores.
    cv2.setNumThreads(1)


def _detect_image(
    image_index: int,
    image_path: Path,
    detections_output_dir: Path | None,
    valid_ids: Set[int] | None,
    max_id_hamming_distance: int,
    initial_intrinsics: np.ndarray | None,
    codebook: Set[int] | None,
):
    """Detect, identify and package the targets of one image (process-safe).

    Returns None for an unreadable image, otherwise
    (width, height, ImageDetections or None, diagnostics entry or None).
    """
    image = cv2.imread(str(image_path))
    if image is None:
        return None
    height, width = image.shape[:2]
    detector = _detector_for(codebook)
    kept = None
    production_diagnostics: list[dict[str, Any]] = []
    if hasattr(detector, "detect_with_diagnostics"):
        detections, detector_diag = detector.detect_with_diagnostics(image, intrinsics=initial_intrinsics)
    else:
        detections = detector.detect(image)
        detector_diag = None

    grouped: Dict[int, List[np.ndarray]] = {}
    grouped_records: Dict[int, List[Any]] = {}
    accepted_for_annotation = []
    rejected_ids = 0
    corrected_ids = 0
    for det in detections:
        raw_id = det.target_id
        target_id, corrected = normalize_target_id(raw_id, valid_ids, max_id_hamming_distance)
        if target_id is None:
            rejected_ids += 1
            continue
        corrected_ids += int(corrected)
        # Do not draw raw decoder IDs that are rejected by the configured
        # target codebook/range.  The production overlay must show the
        # same observations that can enter calibration; when a permitted
        # Hamming correction is used, display the corrected ID.
        accepted = replace(det, target_id=int(target_id))
        accepted_for_annotation.append(accepted)
        grouped_records.setdefault(target_id, []).append(accepted)
        grouped.setdefault(target_id, []).append(
            np.array([det.center[0], det.center[1]], dtype=np.float64)
        )

    merge_radius = max(width, height) * 0.01
    unique_detections: Dict[int, np.ndarray] = {}
    for target_id, points in grouped.items():
        merged = merge_duplicate_detections(points, merge_radius)
        if merged is not None:
            unique_detections[target_id] = merged

    if detector_diag is not None:
        production_diagnostics.append({
            "image": image_path.name,
            "counts": dict(detector_diag.counts),
            "valid_id_rejections": int(rejected_ids),
            "hamming_corrections": int(corrected_ids),
            "duplicate_id_groups": int(max(0, len(grouped) - len(unique_detections))),
            "rejected": list(getattr(detector_diag, "rejected", [])),
        })

    if detections_output_dir is not None:
        annotated = detector.annotate(image, accepted_for_annotation)
        save_annotated_detection_image(detections_output_dir / image_path.name, annotated)

    if unique_detections:
        raw_detections: Dict[int, np.ndarray] = {}
        refinement_ellipses: Dict[int, tuple] = {}
        refinement_status: Dict[int, Dict[str, Any]] = {}
        for target_id in unique_detections:
            candidates = grouped_records.get(target_id, [])
            if len(candidates) != 1:
                continue
            detection = candidates[0]
            ellipse_center = detection.ellipse_center or detection.ellipse[0]
            raw_detections[target_id] = np.asarray(ellipse_center, dtype=np.float64)
            refinement_ellipses[target_id] = detection.ellipse
            refinement_status[target_id] = {
                "method": detection.center_method,
                "intrinsics": (
                    [float(value) for value in initial_intrinsics]
                    if initial_intrinsics is not None and detection.center_method != "ellipse"
                    else None
                ),
                "reason": "verified during production detection",
            }
        kept = (
            ImageDetections(
                image_path=image_path,
                image_index=image_index,
                width=width,
                height=height,
                detections=unique_detections,
                raw_detections=raw_detections,
                refinement_ellipses=refinement_ellipses,
                refinement_status=refinement_status,
                raw_cache_verified=True,
            )
        )

    return width, height, kept, (production_diagnostics[0] if production_diagnostics else None)


def _load_detections_cct_detect(
    image_paths: Sequence[Path],
    detections_output_dir: Path | None,
    valid_ids: Set[int] | None,
    max_id_hamming_distance: int,
    initial_intrinsics: np.ndarray | None = None,
    codebook: Set[int] | None = None,
    workers: int = 1,
) -> List[ImageDetections]:
    """Detection path using the modern cct_detect detector with affine rectification.

    ``codebook`` (the valid target IDs, known when reference coordinates are
    supplied) switches the decoder to correlation against those codes;
    without it the generic decoder is used.  Images are independent, so with
    ``workers`` > 1 they are processed in parallel worker processes; results
    are identical and kept in input order.
    """
    from concurrent.futures import ProcessPoolExecutor

    arguments = [
        (index, path, detections_output_dir, valid_ids, max_id_hamming_distance, initial_intrinsics, codebook)
        for index, path in enumerate(image_paths)
    ]
    workers = max(1, min(int(workers), len(arguments)))
    pool = None
    if workers > 1:
        print(f"detecting with {workers} worker processes", flush=True)
        pool = ProcessPoolExecutor(max_workers=workers, initializer=_init_detection_worker)
        results = pool.map(_detect_image, *zip(*arguments))
    else:
        results = (_detect_image(*item) for item in arguments)

    kept: List[ImageDetections] = []
    production_diagnostics: list[dict[str, Any]] = []
    expected_size: tuple[int, int] | None = None
    try:
        for (image_index, image_path, *_), result in zip(arguments, results):
            if result is not None:
                width, height, image_detections, diagnostics_entry = result
                if expected_size is None:
                    expected_size = (width, height)
                elif expected_size != (width, height):
                    raise ValueError(
                        f"Image size mismatch for {image_path}: {(width, height)} != {expected_size}"
                    )
                if diagnostics_entry is not None:
                    production_diagnostics.append(diagnostics_entry)
                if image_detections is not None:
                    kept.append(image_detections)
            if (image_index + 1) % 20 == 0 or image_index + 1 == len(image_paths):
                print(
                    f"processed {image_index + 1}/{len(image_paths)} images, kept {len(kept)}",
                    flush=True,
                )
    finally:
        if pool is not None:
            pool.shutdown(cancel_futures=True)

    if detections_output_dir is not None and production_diagnostics:
        detections_output_dir.mkdir(parents=True, exist_ok=True)
        (detections_output_dir / "detection_diagnostics.json").write_text(
            json.dumps(production_diagnostics, indent=2), encoding="utf-8",
        )
    return kept


def load_image_detections(
    image_paths: Sequence[Path],
    detections_output_dir: Path | None = None,
    valid_ids: Set[int] | None = None,
    max_id_hamming_distance: int = 0,
    initial_intrinsics: np.ndarray | None = None,
    codebook: Set[int] | None = None,
    workers: int = 1,
) -> List[ImageDetections]:
    return _load_detections_cct_detect(
        image_paths,
        detections_output_dir,
        valid_ids,
        max_id_hamming_distance,
        initial_intrinsics,
        codebook=codebook,
        workers=workers,
    )


def build_track_index(images: Sequence[ImageDetections]) -> Dict[int, Dict[int, np.ndarray]]:
    tracks: Dict[int, Dict[int, np.ndarray]] = {}
    for image in images:
        for target_id, point in image.detections.items():
            tracks.setdefault(target_id, {})[image.image_index] = point
    return tracks


def intrinsics_matrix(intrinsics: np.ndarray) -> np.ndarray:
    return np.array(
        [
            [intrinsics[0], 0.0, intrinsics[2]],
            [0.0, intrinsics[1], intrinsics[3]],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def distortion_vector(intrinsics: np.ndarray) -> np.ndarray:
    return np.array([intrinsics[4], intrinsics[5], intrinsics[6], intrinsics[7]], dtype=np.float64)


def rotation_matrix_from_pose(pose: np.ndarray) -> np.ndarray:
    rotation, _ = cv2.Rodrigues(pose[:3])
    return rotation


def project_point(intrinsics: np.ndarray, pose: np.ndarray, point: np.ndarray) -> np.ndarray:
    rotation = rotation_matrix_from_pose(pose)
    point_camera = rotation @ point + pose[3:]
    z_coord = point_camera[2]
    if z_coord <= 1e-8:
        return np.array([1e6, 1e6], dtype=np.float64)

    x_coord = point_camera[0] / z_coord
    y_coord = point_camera[1] / z_coord
    radius_sq = x_coord * x_coord + y_coord * y_coord
    radial = 1.0 + intrinsics[4] * radius_sq + intrinsics[5] * radius_sq * radius_sq
    x_tangential = 2.0 * intrinsics[6] * x_coord * y_coord + intrinsics[7] * (radius_sq + 2.0 * x_coord * x_coord)
    y_tangential = intrinsics[6] * (radius_sq + 2.0 * y_coord * y_coord) + 2.0 * intrinsics[7] * x_coord * y_coord
    x_distorted = x_coord * radial + x_tangential
    y_distorted = y_coord * radial + y_tangential
    return np.array(
        [
            intrinsics[0] * x_distorted + intrinsics[2],
            intrinsics[1] * y_distorted + intrinsics[3],
        ],
        dtype=np.float64,
    )


def pose_to_projection(intrinsics: np.ndarray, pose: np.ndarray) -> np.ndarray:
    rotation = rotation_matrix_from_pose(pose)
    translation = pose[3:].reshape(3, 1)
    return intrinsics_matrix(intrinsics) @ np.hstack([rotation, translation])


def triangulate_two_views(
    intrinsics: np.ndarray,
    pose_a: np.ndarray,
    pose_b: np.ndarray,
    point_a: np.ndarray,
    point_b: np.ndarray,
) -> np.ndarray | None:
    projection_a = pose_to_projection(intrinsics, pose_a)
    projection_b = pose_to_projection(intrinsics, pose_b)
    homogeneous = cv2.triangulatePoints(
        projection_a,
        projection_b,
        point_a.reshape(2, 1),
        point_b.reshape(2, 1),
    )
    if abs(homogeneous[3, 0]) < 1e-12:
        return None
    point = (homogeneous[:3, 0] / homogeneous[3, 0]).astype(np.float64)
    if not np.all(np.isfinite(point)):
        return None
    for pose in (pose_a, pose_b):
        if (rotation_matrix_from_pose(pose) @ point + pose[3:])[2] <= 1e-6:
            return None
    return point


def choose_seed_state(images: Sequence[ImageDetections], min_shared: int) -> CalibrationState:
    if not images:
        raise RuntimeError("No images passed the detection count filter.")

    width = images[0].width
    height = images[0].height
    intrinsics = np.array(
        [
            float(max(width, height)),
            float(max(width, height)),
            width / 2.0,
            height / 2.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ],
        dtype=np.float64,
    )
    camera_matrix = intrinsics_matrix(intrinsics)

    best_state: CalibrationState | None = None
    best_inlier_count = 0
    for first_index in range(len(images)):
        first_ids = set(images[first_index].detections)
        for second_index in range(first_index + 1, len(images)):
            shared_ids = sorted(first_ids & set(images[second_index].detections))
            if len(shared_ids) < min_shared:
                continue

            first_points = np.array([images[first_index].detections[target_id] for target_id in shared_ids], dtype=np.float64)
            second_points = np.array([images[second_index].detections[target_id] for target_id in shared_ids], dtype=np.float64)

            essential, essential_mask = cv2.findEssentialMat(
                first_points,
                second_points,
                camera_matrix,
                method=cv2.RANSAC,
                prob=0.999,
                threshold=1.5,
            )
            if essential is None or essential_mask is None:
                continue

            recovered, rotation, translation, pose_mask = cv2.recoverPose(
                essential,
                first_points,
                second_points,
                camera_matrix,
            )
            if recovered < min_shared or pose_mask is None:
                continue

            inlier_ids = [shared_ids[index] for index in range(len(shared_ids)) if pose_mask[index, 0] != 0]
            if len(inlier_ids) < min_shared or len(inlier_ids) <= best_inlier_count:
                continue

            pose_a = np.zeros(6, dtype=np.float64)
            pose_b = np.zeros(6, dtype=np.float64)
            pose_b[:3, ...] = cv2.Rodrigues(rotation)[0].reshape(3)
            pose_b[3:] = translation.reshape(3)

            points: Dict[int, np.ndarray] = {}
            for target_id in inlier_ids:
                point = triangulate_two_views(
                    intrinsics,
                    pose_a,
                    pose_b,
                    images[first_index].detections[target_id],
                    images[second_index].detections[target_id],
                )
                if point is not None:
                    points[target_id] = point

            if len(points) < min_shared:
                continue

            best_inlier_count = len(points)
            best_state = CalibrationState(
                intrinsics=intrinsics.copy(),
                poses={
                    images[first_index].image_index: pose_a,
                    images[second_index].image_index: pose_b,
                },
                points=points,
                seed_image_indices=[images[first_index].image_index, images[second_index].image_index],
                inlier_track_ids=sorted(points),
            )

    if best_state is None:
        raise RuntimeError("Failed to initialize a seed image pair with enough shared detections.")
    return best_state


def _screened_focal_initialization(
    object_points_list: List[np.ndarray],
    image_points_list: List[np.ndarray],
    width: int,
    height: int,
) -> tuple[float, List[np.ndarray], List[np.ndarray]]:
    """Starting focal length for calibrateCamera, plus outlier-screened points.

    A log-spaced focal scan scores each candidate by the median over frames of
    the per-frame median PnP reprojection error (pinhole, principal point at
    the image centre).  Medians make the score insensitive to the few
    misidentified targets.  At the chosen focal length, correspondences far
    beyond their frame's typical error are dropped; the threshold is lenient
    because unmodelled lens distortion alone can reach tens of pixels at the
    edge of a wide-angle image.  Frames left with fewer than six points keep
    their original correspondences.
    """
    size = float(max(width, height))

    def frame_errors(focal: float, object_points: np.ndarray, image_points: np.ndarray):
        matrix = np.array([[focal, 0.0, width / 2.0], [0.0, focal, height / 2.0], [0.0, 0.0, 1.0]])
        objects = object_points.reshape(-1, 3).astype(np.float64)
        pixels = image_points.reshape(-1, 2).astype(np.float64)
        try:
            ok, rvec, tvec, _ = cv2.solvePnPRansac(
                objects, pixels, matrix, None, flags=cv2.SOLVEPNP_EPNP,
                reprojectionError=max(8.0, 0.01 * size), iterationsCount=100,
            )
            if not ok:
                return None
            ok, rvec, tvec = cv2.solvePnP(objects, pixels, matrix, None, rvec, tvec,
                                          useExtrinsicGuess=True, flags=cv2.SOLVEPNP_ITERATIVE)
        except cv2.error:
            return None
        if not ok:
            return None
        projected = cv2.projectPoints(objects, rvec, tvec, matrix, None)[0].reshape(-1, 2)
        errors = np.linalg.norm(projected - pixels, axis=1)
        return errors if np.all(np.isfinite(errors)) else None

    best_focal, best_score = size, float("inf")
    for focal in size * np.geomspace(0.15, 2.0, 25):
        medians = []
        for object_points, image_points in zip(object_points_list, image_points_list):
            errors = frame_errors(float(focal), object_points, image_points)
            medians.append(float(np.median(errors)) if errors is not None else float("inf"))
        score = float(np.median(medians))
        if score < best_score:
            best_focal, best_score = float(focal), score
    if not np.isfinite(best_score):
        return size, object_points_list, image_points_list

    screened_objects: List[np.ndarray] = []
    screened_images: List[np.ndarray] = []
    for object_points, image_points in zip(object_points_list, image_points_list):
        errors = frame_errors(best_focal, object_points, image_points)
        if errors is not None:
            keep = errors <= max(25.0, 4.0 * float(np.median(errors)))
            if np.count_nonzero(keep) >= 6:
                object_points, image_points = object_points[keep], image_points[keep]
        screened_objects.append(object_points)
        screened_images.append(image_points)
    return best_focal, screened_objects, screened_images


def _robust_intrinsic_adjustment(
    object_points_list: List[np.ndarray],
    image_points_list: List[np.ndarray],
    width: int,
    height: int,
    initial_focal: float,
) -> tuple[float, np.ndarray, List[np.ndarray | None]]:
    """Huber-robust fit of OPENCV-8 intrinsics and view poses to known targets.

    Views are initialised by PnP at ``initial_focal``.  A pinhole stage
    (distortion held at zero) precedes the full fit so that the distortion
    terms start from a sensible focal length and principal point.  Bounds
    match the bundle adjustment.  Returns the per-point RMS reprojection
    error (pixels), the intrinsics and one pose per input view (None when the
    view could not be posed).
    """
    from scipy.optimize import least_squares
    from scipy.sparse import lil_matrix

    size = float(max(width, height))
    focal = float(np.clip(initial_focal, 0.21 * size, 2.9 * size))
    objects = [np.asarray(o, dtype=np.float64).reshape(-1, 3) for o in object_points_list]
    pixels = [np.asarray(p, dtype=np.float64).reshape(-1, 2) for p in image_points_list]
    matrix = np.array([[focal, 0.0, width / 2.0], [0.0, focal, height / 2.0], [0.0, 0.0, 1.0]])

    initial_poses: List[np.ndarray | None] = []
    for obj, pix in zip(objects, pixels):
        pose = None
        try:
            ok, rvec, tvec, _ = cv2.solvePnPRansac(
                obj, pix, matrix, None, flags=cv2.SOLVEPNP_EPNP,
                reprojectionError=max(8.0, 0.01 * size), iterationsCount=100,
            )
            if ok:
                ok, rvec, tvec = cv2.solvePnP(obj, pix, matrix, None, rvec, tvec,
                                              useExtrinsicGuess=True, flags=cv2.SOLVEPNP_ITERATIVE)
            if ok:
                pose = np.concatenate([rvec.reshape(3), tvec.reshape(3)]).astype(np.float64)
        except cv2.error:
            pose = None
        initial_poses.append(pose)
    active = [index for index, pose in enumerate(initial_poses) if pose is not None]
    intrinsics = np.array([focal, focal, width / 2.0, height / 2.0, 0.0, 0.0, 0.0, 0.0])
    if len(active) < 2:
        return float("inf"), intrinsics, [None] * len(objects)

    lower_all = np.array([0.2 * size, 0.2 * size, 0.25 * width, 0.25 * height, -1.0, -1.0, -1.0, -1.0])
    upper_all = np.array([3.0 * size, 3.0 * size, 0.75 * width, 0.75 * height, 1.0, 1.0, 1.0, 1.0])
    poses = np.array([initial_poses[index] for index in active])
    row_counts = [2 * len(objects[index]) for index in active]

    def evaluate(intr: np.ndarray, pose_block: np.ndarray) -> np.ndarray:
        camera = np.array([[intr[0], 0.0, intr[2]], [0.0, intr[1], intr[3]], [0.0, 0.0, 1.0]])
        out = []
        for slot, index in enumerate(active):
            projected = cv2.projectPoints(objects[index], pose_block[slot, :3], pose_block[slot, 3:],
                                          camera, intr[4:8])[0].reshape(-1, 2)
            out.append((projected - pixels[index]).reshape(-1))
        return np.concatenate(out)

    for free in (np.arange(4), np.arange(8)):
        def unpack(x: np.ndarray):
            intr = intrinsics.copy()
            intr[free] = x[:free.size]
            return intr, x[free.size:].reshape(-1, 6)

        x0 = np.concatenate([intrinsics[free], poses.reshape(-1)])
        sparsity = lil_matrix((sum(row_counts), x0.size), dtype=int)
        row = 0
        for slot, count in enumerate(row_counts):
            sparsity[row:row + count, :free.size] = 1
            start = free.size + 6 * slot
            sparsity[row:row + count, start:start + 6] = 1
            row += count
        lower = np.concatenate([lower_all[free], np.full(poses.size, -np.inf)])
        upper = np.concatenate([upper_all[free], np.full(poses.size, np.inf)])
        inset = 1e-9 * np.where(np.isfinite(lower), np.abs(lower) + 1.0, 0.0)
        x0 = np.clip(x0, lower + inset, upper - inset)
        solution = least_squares(
            lambda x: evaluate(*unpack(x)), x0, jac_sparsity=sparsity, bounds=(lower, upper),
            loss="huber", f_scale=3.0, x_scale="jac", max_nfev=200, method="trf",
        )
        intrinsics, poses = unpack(solution.x)
        intrinsics = intrinsics.copy()
        poses = poses.copy()

    residual = evaluate(intrinsics, poses).reshape(-1, 2)
    rms = float(np.sqrt(np.mean(np.sum(residual ** 2, axis=1))))
    solved: List[np.ndarray | None] = [None] * len(objects)
    for slot, index in enumerate(active):
        solved[index] = poses[slot].copy()
    return rms, intrinsics, solved


def initialize_known_target_state(
    images: Sequence[ImageDetections],
    known_targets3d: Dict[int, np.ndarray],
    min_shared: int,
) -> tuple[CalibrationState, Set[int]]:
    if not images:
        raise RuntimeError("No images passed the detection count filter.")

    width = images[0].width
    height = images[0].height
    object_points_list: List[np.ndarray] = []
    image_points_list: List[np.ndarray] = []
    image_indices: List[int] = []
    observed_target_ids: Set[int] = set()

    for image in images:
        shared_ids = sorted(set(image.detections) & set(known_targets3d))
        if len(shared_ids) < min_shared:
            continue
        object_points_list.append(
            np.array([known_targets3d[target_id] for target_id in shared_ids], dtype=np.float32).reshape(-1, 1, 3)
        )
        image_points_list.append(
            np.array([image.detections[target_id] for target_id in shared_ids], dtype=np.float32).reshape(-1, 1, 2)
        )
        image_indices.append(image.image_index)
        observed_target_ids.update(shared_ids)

    if len(object_points_list) < 2:
        raise RuntimeError(
            "Not enough images with known 3D target correspondences to initialize single-camera calibration."
        )

    # cv2.calibrateCamera was used here.  It initialises every view itself
    # from the intrinsic guess, and on this wide-angle data it converged to
    # different wrong minima (RMS tens of pixels, even negative focal lengths)
    # for nearly identical inputs.  The robust adjustment below starts from
    # a data-driven focal length and per-view PnP poses instead.
    initial_focal, object_points_list, image_points_list = _screened_focal_initialization(
        object_points_list, image_points_list, width, height,
    )
    calibration_rms, intrinsics, solved_poses = _robust_intrinsic_adjustment(
        object_points_list, image_points_list, width, height, initial_focal,
    )
    poses = {
        image_index: pose
        for image_index, pose in zip(image_indices, solved_poses, strict=False)
        if pose is not None
    }
    if len(poses) < 2:
        raise RuntimeError("Known-target initialisation could not pose at least two images.")
    points = {target_id: known_targets3d[target_id].copy() for target_id in sorted(observed_target_ids)}
    print(
        f"initialized from known 3D targets using {len(image_indices)} images and {len(points)} fixed targets "
        f"(robust initialisation RMS {calibration_rms:.4f} px)",
        flush=True,
    )
    return CalibrationState(
        intrinsics=intrinsics,
        poses=poses,
        points=points,
        seed_image_indices=[],
        inlier_track_ids=sorted(points),
    ), set(points)


def register_remaining_images(
    images: Sequence[ImageDetections],
    state: CalibrationState,
    min_shared: int,
) -> None:
    images_by_index = {image.image_index: image for image in images}
    while True:
        progress_made = False
        for image in images:
            if image.image_index in state.poses:
                continue

            shared_ids = sorted(set(image.detections) & set(state.points))
            if len(shared_ids) < min_shared:
                continue

            object_points = np.array([state.points[target_id] for target_id in shared_ids], dtype=np.float64)
            image_points = np.array([image.detections[target_id] for target_id in shared_ids], dtype=np.float64)
            solved, rotation_vec, translation_vec, inliers = cv2.solvePnPRansac(
                object_points,
                image_points,
                intrinsics_matrix(state.intrinsics),
                distortion_vector(state.intrinsics),
                flags=cv2.SOLVEPNP_ITERATIVE,
                reprojectionError=8.0,
                iterationsCount=200,
            )
            if not solved or inliers is None or len(inliers) < min_shared:
                continue

            inlier_indices = inliers.reshape(-1)
            inlier_object_points = object_points[inlier_indices]
            inlier_image_points = image_points[inlier_indices]
            refined, rotation_vec, translation_vec = cv2.solvePnP(
                inlier_object_points,
                inlier_image_points,
                intrinsics_matrix(state.intrinsics),
                distortion_vector(state.intrinsics),
                rvec=rotation_vec,
                tvec=translation_vec,
                useExtrinsicGuess=True,
                flags=cv2.SOLVEPNP_ITERATIVE,
            )
            if not refined:
                continue

            pose = np.concatenate([rotation_vec.reshape(3), translation_vec.reshape(3)]).astype(np.float64)
            state.poses[image.image_index] = pose
            progress_made = True

            for target_id in image.detections:
                if target_id in state.points:
                    continue
                candidate_pairs: List[tuple[float, int]] = []
                for other_image_index in state.poses:
                    if other_image_index == image.image_index:
                        continue
                    other_image = images_by_index[other_image_index]
                    if target_id not in other_image.detections:
                        continue
                    baseline = float(
                        np.linalg.norm(state.poses[other_image_index][3:] - pose[3:])
                    )
                    candidate_pairs.append((baseline, other_image_index))

                for _, other_image_index in sorted(candidate_pairs, reverse=True):
                    other_image = images_by_index[other_image_index]
                    point = triangulate_two_views(
                        state.intrinsics,
                        state.poses[other_image_index],
                        pose,
                        other_image.detections[target_id],
                        image.detections[target_id],
                    )
                    if point is not None:
                        state.points[target_id] = point
                        break

        if not progress_made:
            break


def collect_observations(
    images: Sequence[ImageDetections],
    state: CalibrationState,
    max_reprojection_error: float,
) -> List[tuple[int, int, np.ndarray]]:
    observations: List[tuple[int, int, np.ndarray]] = []
    for image in images:
        pose = state.poses.get(image.image_index)
        if pose is None:
            continue
        for target_id, point_2d in image.detections.items():
            point_3d = state.points.get(target_id)
            if point_3d is None:
                continue
            residual = project_point(state.intrinsics, pose, point_3d) - point_2d
            if not np.all(np.isfinite(residual)):
                continue
            if np.linalg.norm(residual) <= max_reprojection_error:
                observations.append((image.image_index, target_id, point_2d))
    return observations


def compute_reprojection_errors(
    state: CalibrationState,
    observations: Sequence[tuple[int, int, np.ndarray]],
) -> np.ndarray:
    errors: List[float] = []
    for image_index, target_id, point_2d in observations:
        pose = state.poses.get(image_index)
        point_3d = state.points.get(target_id)
        if pose is None or point_3d is None:
            continue
        residual = project_point(state.intrinsics, pose, point_3d) - point_2d
        if np.all(np.isfinite(residual)):
            errors.append(float(np.linalg.norm(residual)))
    return np.asarray(errors, dtype=np.float64)


def compute_object_space_residual(
    state: CalibrationState,
    observation: tuple[int, int, np.ndarray],
) -> float | None:
    image_index, target_id, point_2d = observation
    pose = state.poses.get(image_index)
    point_3d = state.points.get(target_id)
    if pose is None or point_3d is None:
        return None

    rotation = rotation_matrix_from_pose(pose)
    camera_center = camera_center_from_pose(pose)
    undistorted = cv2.undistortPoints(
        np.asarray(point_2d, dtype=np.float64).reshape(1, 1, 2),
        intrinsics_matrix(state.intrinsics),
        distortion_vector(state.intrinsics),
    )
    if undistorted is None:
        return None

    ray_camera = np.array(
        [undistorted[0, 0, 0], undistorted[0, 0, 1], 1.0],
        dtype=np.float64,
    )
    ray_camera_norm = np.linalg.norm(ray_camera)
    if not np.isfinite(ray_camera_norm) or ray_camera_norm <= 1e-12:
        return None
    ray_camera /= ray_camera_norm

    ray_world = rotation.T @ ray_camera
    ray_world_norm = np.linalg.norm(ray_world)
    if not np.isfinite(ray_world_norm) or ray_world_norm <= 1e-12:
        return None
    ray_world /= ray_world_norm

    offset = point_3d - camera_center
    depth_along_ray = float(np.dot(offset, ray_world))
    if not np.isfinite(depth_along_ray) or depth_along_ray <= 1e-12:
        return None

    residual = float(np.linalg.norm(offset - depth_along_ray * ray_world))
    if not np.isfinite(residual):
        return None
    return residual


def compute_object_space_errors(
    state: CalibrationState,
    observations: Sequence[tuple[int, int, np.ndarray]],
) -> np.ndarray:
    errors: List[float] = []
    for observation in observations:
        residual = compute_object_space_residual(state, observation)
        if residual is not None:
            errors.append(residual)
    return np.asarray(errors, dtype=np.float64)


class LossConvergenceCallback(pyceres.IterationCallback):
    def __init__(self, relative_tolerance: float, patience: int):
        super().__init__()
        self.relative_tolerance = relative_tolerance
        self.patience = patience
        self.cost_history: List[float] = []
        self._stable_steps = 0

    def __call__(self, summary) -> pyceres.CallbackReturnType:
        cost = float(summary.cost)
        self.cost_history.append(cost)

        if len(self.cost_history) == 1:
            return pyceres.CallbackReturnType.SOLVER_CONTINUE

        if not getattr(summary, "step_is_successful", True):
            self._stable_steps = 0
            return pyceres.CallbackReturnType.SOLVER_CONTINUE

        previous_cost = self.cost_history[-2]
        relative_change = abs(previous_cost - cost) / max(abs(previous_cost), 1.0)
        if relative_change <= self.relative_tolerance:
            self._stable_steps += 1
        else:
            self._stable_steps = 0

        if self._stable_steps >= self.patience:
            return pyceres.CallbackReturnType.SOLVER_TERMINATE_SUCCESSFULLY
        return pyceres.CallbackReturnType.SOLVER_CONTINUE


def camera_center_from_pose(pose: np.ndarray) -> np.ndarray:
    rotation = rotation_matrix_from_pose(pose)
    translation = pose[3:]
    return (-rotation.T @ translation).astype(np.float64)


def camera_axes_from_pose(pose: np.ndarray) -> np.ndarray:
    rotation = rotation_matrix_from_pose(pose)
    return rotation.T.astype(np.float64)


def build_alignment_basis(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    centroid = np.mean(points, axis=0)
    centered = points - centroid
    if len(points) < 3:
        return centroid, np.eye(3, dtype=np.float64)

    _, _, basis_t = np.linalg.svd(centered, full_matrices=False)
    basis = basis_t.T
    if np.linalg.det(basis) < 0:
        basis[:, 2] *= -1.0
    return centroid, basis


def align_scene(state: CalibrationState) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    ordered_pose_indices = np.array(sorted(state.poses), dtype=np.int32)
    camera_centers = np.array([camera_center_from_pose(state.poses[index]) for index in ordered_pose_indices], dtype=np.float64)
    camera_axes = np.array([camera_axes_from_pose(state.poses[index]) for index in ordered_pose_indices], dtype=np.float64)

    track_ids = np.array(sorted(state.points), dtype=np.int32)
    target_points = np.array([state.points[target_id] for target_id in track_ids], dtype=np.float64) if len(track_ids) else np.empty((0, 3), dtype=np.float64)

    combined = np.vstack([camera_centers, target_points]) if len(target_points) else camera_centers
    centroid, basis = build_alignment_basis(combined)

    aligned_centers = (camera_centers - centroid) @ basis
    aligned_points = (target_points - centroid) @ basis if len(target_points) else target_points
    aligned_axes = np.empty_like(camera_axes)
    for camera_index in range(len(camera_axes)):
        aligned_axes[camera_index] = basis.T @ camera_axes[camera_index]

    return ordered_pose_indices, aligned_centers, aligned_axes, track_ids, aligned_points


def save_scene_data(
    state: CalibrationState,
    camera_name: str,
    output_dir: Path,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{camera_name}_scene.npz"
    pose_indices, camera_centers, camera_axes, track_ids, target_points = align_scene(state)
    np.savez_compressed(
        output_path,
        pose_indices=pose_indices,
        camera_centers=camera_centers,
        camera_axes=camera_axes,
        track_ids=track_ids,
        target_points=target_points,
    )
    return output_path


def save_text_exports(
    state: CalibrationState,
    camera_name: str,
    output_dir: Path,
) -> tuple[Path, Path]:
    camera_output_dir = output_dir / camera_name
    camera_output_dir.mkdir(parents=True, exist_ok=True)

    targets_3d_path = camera_output_dir / "targets_3d.txt"
    cameras_path = camera_output_dir / "cameras.txt"

    pose_indices, camera_centers, camera_axes, track_ids, target_points = align_scene(state)

    target_lines = [
        f"{int(track_id)} {point[0]:.9f} {point[1]:.9f} {point[2]:.9f}"
        for track_id, point in zip(track_ids, target_points, strict=False)
    ]
    targets_3d_path.write_text("\n".join(target_lines) + ("\n" if target_lines else ""), encoding="utf-8")

    camera_lines = []
    for pose_index, center, axes in zip(pose_indices, camera_centers, camera_axes, strict=False):
        quaternion = rotation_matrix_to_quaternion(axes)
        camera_lines.append(
            f"{int(pose_index)} {center[0]:.9f} {center[1]:.9f} {center[2]:.9f} {quaternion[0]:.9f} {quaternion[1]:.9f} {quaternion[2]:.9f} {quaternion[3]:.9f}"
        )
    cameras_path.write_text("\n".join(camera_lines) + ("\n" if camera_lines else ""), encoding="utf-8")

    return targets_3d_path, cameras_path


def save_colmap_output(
    state: CalibrationState,
    bundle_images: Sequence[ImageDetections],
    camera_name: str,
    output_dir: Path,
) -> Path:
    """Save calibration results in COLMAP text format.

    Produces cameras.txt, images.txt, and points3D.txt inside
    ``output_dir / camera_name / colmap``.
    """
    colmap_dir = output_dir / camera_name / "colmap"
    colmap_dir.mkdir(parents=True, exist_ok=True)

    images_by_index = {img.image_index: img for img in bundle_images}
    image_width = bundle_images[0].width if bundle_images else 0
    image_height = bundle_images[0].height if bundle_images else 0

    pose_indices, camera_centers, camera_axes, track_ids, target_points = align_scene(state)

    # --- cameras.txt (single shared camera, OPENCV model) ---
    intr = state.intrinsics  # [fx, fy, cx, cy, k1, k2, p1, p2]
    cameras_lines = [
        "# Camera list with one line of data per camera:",
        "#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[fx,fy,cx,cy,k1,k2,p1,p2]",
        f"1 OPENCV {image_width} {image_height} "
        f"{intr[0]:.9f} {intr[1]:.9f} {intr[2]:.9f} {intr[3]:.9f} "
        f"{intr[4]:.9f} {intr[5]:.9f} {intr[6]:.9f} {intr[7]:.9f}",
    ]
    (colmap_dir / "cameras.txt").write_text("\n".join(cameras_lines) + "\n", encoding="utf-8")

    # Build a mapping: track_id → COLMAP point3D_id (1-based)
    track_id_to_colmap_id = {int(tid): idx + 1 for idx, tid in enumerate(track_ids)}

    # --- images.txt ---
    # COLMAP stores world-to-camera transforms (R, t) such that X_cam = R * X_world + t
    # Our poses are already in that convention (rvec, tvec).
    # But align_scene applies a rotation/translation, so we need to recompute
    # the aligned poses from aligned_centers and aligned_axes.
    images_lines = [
        "# Image list with two lines of data per image:",
        "#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME",
        "#   POINTS2D[] as (X, Y, POINT3D_ID)",
    ]
    for colmap_image_id, (pose_idx, center, axes) in enumerate(
        zip(pose_indices, camera_centers, camera_axes, strict=False), start=1
    ):
        # axes is the camera rotation matrix (world-to-camera is axes.T)
        R_w2c = axes.T  # axes = R.T, so axes.T = R (world-to-camera rotation)
        t_w2c = -R_w2c @ center  # t = -R * C
        quat = rotation_matrix_to_quaternion(R_w2c)  # [x, y, z, w]
        qw, qx, qy, qz = float(quat[3]), float(quat[0]), float(quat[1]), float(quat[2])

        image_name = images_by_index[int(pose_idx)].image_path.name if int(pose_idx) in images_by_index else f"image_{int(pose_idx):04d}.jpg"

        images_lines.append(
            f"{colmap_image_id} {qw:.9f} {qx:.9f} {qy:.9f} {qz:.9f} "
            f"{t_w2c[0]:.9f} {t_w2c[1]:.9f} {t_w2c[2]:.9f} 1 {image_name}"
        )

        # 2D points line: list detected targets that correspond to 3D points
        points2d_parts = []
        img_det = images_by_index.get(int(pose_idx))
        if img_det is not None:
            for target_id, pt2d in sorted(img_det.detections.items()):
                colmap_pt_id = track_id_to_colmap_id.get(target_id, -1)
                points2d_parts.append(f"{pt2d[0]:.4f} {pt2d[1]:.4f} {colmap_pt_id}")
        images_lines.append(" ".join(points2d_parts) if points2d_parts else "")

    (colmap_dir / "images.txt").write_text("\n".join(images_lines) + "\n", encoding="utf-8")

    # --- points3D.txt ---
    # Build track info (which images observe each point)
    points3d_lines = [
        "# 3D point list with one line of data per point:",
        "#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)",
    ]
    # Build per-point observation track
    point_tracks: Dict[int, List[tuple[int, int]]] = {int(tid): [] for tid in track_ids}
    colmap_image_id_map = {int(pi): ci for ci, pi in enumerate(pose_indices, start=1)}
    for pose_idx_int, img_det in images_by_index.items():
        if pose_idx_int not in state.poses:
            continue
        colmap_img_id = colmap_image_id_map.get(pose_idx_int)
        if colmap_img_id is None:
            continue
        sorted_dets = sorted(img_det.detections.items())
        for pt2d_idx, (target_id, _) in enumerate(sorted_dets):
            if target_id in point_tracks:
                point_tracks[target_id].append((colmap_img_id, pt2d_idx))

    for idx, (tid, pt3d) in enumerate(zip(track_ids, target_points, strict=False)):
        colmap_pt_id = idx + 1
        track_entries = point_tracks.get(int(tid), [])
        track_str = " ".join(f"{img_id} {pt_idx}" for img_id, pt_idx in track_entries)
        points3d_lines.append(
            f"{colmap_pt_id} {pt3d[0]:.9f} {pt3d[1]:.9f} {pt3d[2]:.9f} 200 200 200 0.0 {track_str}"
        )

    (colmap_dir / "points3D.txt").write_text("\n".join(points3d_lines) + "\n", encoding="utf-8")

    return colmap_dir


def save_report(
    state: CalibrationState,
    camera_name: str,
    image_size: tuple[int, int],
    observations: Sequence[tuple[int, int, np.ndarray]],
    diagnostics: SolverDiagnostics,
    output_dir: Path,
) -> Path:
    reprojection_errors = compute_reprojection_errors(state, observations)
    object_space_errors = compute_object_space_errors(state, observations)
    track_ids = {target_id for _, target_id, _ in observations}

    lines: List[str] = []
    lines.append("=" * 60)
    lines.append("MONO CCT CALIBRATION REPORT")
    lines.append("=" * 60)
    lines.append("")
    lines.append(f"Camera:            {camera_name}")
    lines.append(f"Resolution:        {image_size[0]} x {image_size[1]}")
    lines.append(f"Used images:       {len(state.poses)}")
    lines.append(f"Registered images: {len(state.poses)}")
    lines.append(f"3D targets:        {len(track_ids)}")
    lines.append(f"Observations:      {len(observations)}")
    lines.append("")

    lines.append("--- Reprojection errors ---")
    if reprojection_errors.size:
        lines.append(f"  Mean: {float(np.mean(reprojection_errors)):.4f} px")
        lines.append(f"  RMS:  {float(np.sqrt(np.mean(np.square(reprojection_errors)))):.4f} px")
        lines.append(f"  Max:  {float(np.max(reprojection_errors)):.4f} px")
        lines.append(f"  Median: {float(np.median(reprojection_errors)):.4f} px")
    lines.append("")

    lines.append("--- Object-space errors ---")
    if object_space_errors.size:
        lines.append(f"  Mean: {float(np.mean(object_space_errors)):.6f} m")
        lines.append(f"  RMS:  {float(np.sqrt(np.mean(np.square(object_space_errors)))):.6f} m")
        lines.append(f"  Max:  {float(np.max(object_space_errors)):.6f} m")
        lines.append(f"  Median: {float(np.median(object_space_errors)):.6f} m")
    lines.append("")

    lines.append("--- Solver ---")
    lines.append(f"  Iterations:   {diagnostics.iterations}")
    lines.append(f"  Initial cost: {diagnostics.initial_cost:.6f}")
    lines.append(f"  Final cost:   {diagnostics.final_cost:.6f}")
    lines.append(f"  {diagnostics.summary}")
    lines.append("")

    intrinsics = state.intrinsics
    lines.append("--- Intrinsics (OPENCV) ---")
    lines.append(f"  fx = {intrinsics[0]:.6f}")
    lines.append(f"  fy = {intrinsics[1]:.6f}")
    lines.append(f"  cx = {intrinsics[2]:.6f}")
    lines.append(f"  cy = {intrinsics[3]:.6f}")
    lines.append(f"  k1 = {intrinsics[4]:.6f}")
    lines.append(f"  k2 = {intrinsics[5]:.6f}")
    lines.append(f"  p1 = {intrinsics[6]:.6f}")
    lines.append(f"  p2 = {intrinsics[7]:.6f}")

    report_path = output_dir / f"{camera_name}_report.txt"
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report_path


def rotation_matrix_to_quaternion(rotation: np.ndarray) -> np.ndarray:
    trace = float(np.trace(rotation))
    if trace > 0.0:
        scale = np.sqrt(trace + 1.0) * 2.0
        w_value = 0.25 * scale
        x_value = (rotation[2, 1] - rotation[1, 2]) / scale
        y_value = (rotation[0, 2] - rotation[2, 0]) / scale
        z_value = (rotation[1, 0] - rotation[0, 1]) / scale
    elif rotation[0, 0] > rotation[1, 1] and rotation[0, 0] > rotation[2, 2]:
        scale = np.sqrt(1.0 + rotation[0, 0] - rotation[1, 1] - rotation[2, 2]) * 2.0
        w_value = (rotation[2, 1] - rotation[1, 2]) / scale
        x_value = 0.25 * scale
        y_value = (rotation[0, 1] + rotation[1, 0]) / scale
        z_value = (rotation[0, 2] + rotation[2, 0]) / scale
    elif rotation[1, 1] > rotation[2, 2]:
        scale = np.sqrt(1.0 + rotation[1, 1] - rotation[0, 0] - rotation[2, 2]) * 2.0
        w_value = (rotation[0, 2] - rotation[2, 0]) / scale
        x_value = (rotation[0, 1] + rotation[1, 0]) / scale
        y_value = 0.25 * scale
        z_value = (rotation[1, 2] + rotation[2, 1]) / scale
    else:
        scale = np.sqrt(1.0 + rotation[2, 2] - rotation[0, 0] - rotation[1, 1]) * 2.0
        w_value = (rotation[1, 0] - rotation[0, 1]) / scale
        x_value = (rotation[0, 2] + rotation[2, 0]) / scale
        y_value = (rotation[1, 2] + rotation[2, 1]) / scale
        z_value = 0.25 * scale
    quaternion = np.array([x_value, y_value, z_value, w_value], dtype=np.float64)
    quaternion /= max(np.linalg.norm(quaternion), 1e-12)
    return quaternion


def write_rerun_visualization(
    state: CalibrationState,
    camera_name: str,
    image_size: tuple[int, int],
    output_dir: Path,
    spawn_viewer: bool,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    recording_path = output_dir / f"{camera_name}.rrd"

    if rr is None:
        recording_path.write_text("rerun is not installed in this environment\n", encoding="utf-8")
        return recording_path

    pose_indices, camera_centers, camera_axes, track_ids, target_points = align_scene(state)
    recording = rr.init(f"calibration_{camera_name}", spawn=False)
    if spawn_viewer:
        rr.spawn(recording=recording)

    rr.log("/", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True, recording=recording)

    if len(target_points):
        rr.log(
            f"world/{camera_name}/targets",
            rr.Points3D(
                target_points,
                radii=0.02,
                colors=[180, 180, 180],
                labels=[str(int(track_id)) for track_id in track_ids],
            ),
            recording=recording,
        )

    image_width, image_height = image_size
    for pose_index, center, axes in zip(pose_indices, camera_centers, camera_axes, strict=False):
        camera_path = f"world/{camera_name}/cameras/{int(pose_index):04d}"
        rr.log(
            camera_path,
            rr.Transform3D(
                translation=center,
                quaternion=rr.Quaternion(xyzw=rotation_matrix_to_quaternion(axes)),
            ),
            rr.TransformAxes3D(axis_length=0.15),
            recording=recording,
        )
        rr.log(
            camera_path,
            rr.Pinhole(
                image_from_camera=intrinsics_matrix(state.intrinsics),
                resolution=[image_width, image_height],
                camera_xyz=rr.ViewCoordinates.RDF,
                image_plane_distance=0.3,
                line_width=0.004,
                color=[255, 128, 0],
            ),
            recording=recording,
        )

    rr.save(recording_path, recording=recording)
    return recording_path


def save_convergence_plot(
    camera_name: str,
    cost_history: Sequence[float],
    output_dir: Path,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{camera_name}_convergence.png"

    fig, axis = plt.subplots(figsize=(10, 6))
    iterations = np.arange(len(cost_history), dtype=np.int32)
    axis.plot(iterations, cost_history, color="#1f77b4", linewidth=2)
    axis.set_title(f"Loss convergence for {camera_name}")
    axis.set_xlabel("Iteration")
    axis.set_ylabel("Cost")
    axis.grid(True, linestyle="--", alpha=0.4)
    if len(cost_history) > 1 and min(cost_history) > 0:
        axis.set_yscale("log")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


class ReprojectionCost(pyceres.CostFunction):
    def __init__(self, observation: np.ndarray):
        super().__init__()
        self.observation = observation.astype(np.float64)
        self.set_num_residuals(2)
        self.set_parameter_block_sizes([8, 6, 3])

    def Evaluate(self, parameters, residuals, jacobians):
        parameter_blocks = [np.array(parameters[index], dtype=np.float64, copy=True) for index in range(3)]
        prediction = project_point(parameter_blocks[0], parameter_blocks[1], parameter_blocks[2])
        residual_vector = prediction - self.observation
        residuals[0] = float(residual_vector[0])
        residuals[1] = float(residual_vector[1])

        if jacobians is not None:
            for block_index, block in enumerate(parameter_blocks):
                if jacobians[block_index] is None:
                    continue
                jacobian = np.zeros((2, block.shape[0]), dtype=np.float64)
                for column in range(block.shape[0]):
                    step = 1e-6 * max(1.0, abs(block[column]))
                    plus_blocks = [value.copy() for value in parameter_blocks]
                    minus_blocks = [value.copy() for value in parameter_blocks]
                    plus_blocks[block_index][column] += step
                    minus_blocks[block_index][column] -= step
                    residual_plus = project_point(plus_blocks[0], plus_blocks[1], plus_blocks[2]) - self.observation
                    residual_minus = project_point(minus_blocks[0], minus_blocks[1], minus_blocks[2]) - self.observation
                    jacobian[:, column] = (residual_plus - residual_minus) / (2.0 * step)

                for row in range(2):
                    for column in range(block.shape[0]):
                        jacobians[block_index][row * block.shape[0] + column] = jacobian[row, column]
        return True


def _filter_inlier_observations(
    state: CalibrationState,
    observations: Sequence[tuple[int, int, np.ndarray]],
    max_error: float,
) -> List[tuple[int, int, np.ndarray]]:
    """Keep only observations whose reprojection error is within *max_error* pixels."""
    inliers: List[tuple[int, int, np.ndarray]] = []
    for image_index, target_id, point_2d in observations:
        pose = state.poses.get(image_index)
        point_3d = state.points.get(target_id)
        if pose is None or point_3d is None:
            continue
        residual = project_point(state.intrinsics, pose, point_3d) - point_2d
        if np.all(np.isfinite(residual)) and np.linalg.norm(residual) <= max_error:
            inliers.append((image_index, target_id, point_2d))
    return inliers


def solve_bundle_adjustment(
    images: Sequence[ImageDetections],
    state: CalibrationState,
    observations: Sequence[tuple[int, int, np.ndarray]],
    max_iterations: int,
    huber_delta: float,
    loss_relative_tolerance: float,
    loss_patience: int,
    robust_loss: str = "huber",
    cauchy_scale: float = 3.0,
    fixed_point_ids: Set[int] | None = None,
) -> SolverDiagnostics:
    problem = pyceres.Problem()
    if robust_loss == "cauchy":
        loss = pyceres.CauchyLoss(cauchy_scale)
    else:
        loss = pyceres.HuberLoss(huber_delta)
    manifolds: List[pyceres.Manifold] = []
    used_points: Set[int] = set()

    for image_index, target_id, point_2d in observations:
        cost = ReprojectionCost(point_2d)
        problem.add_residual_block(
            cost,
            loss,
            [state.intrinsics, state.poses[image_index], state.points[target_id]],
        )
        used_points.add(target_id)

    if fixed_point_ids:
        for target_id in fixed_point_ids:
            if target_id in used_points:
                problem.set_parameter_block_constant(state.points[target_id])
    else:
        if state.seed_image_indices:
            problem.set_parameter_block_constant(state.poses[state.seed_image_indices[0]])
        if len(state.seed_image_indices) > 1:
            second_seed_pose = state.poses[state.seed_image_indices[1]]
            translation_fixed_manifold = pyceres.SubsetManifold(6, [3, 4, 5])
            manifolds.append(translation_fixed_manifold)
            problem.set_manifold(second_seed_pose, translation_fixed_manifold)

    width = images[0].width
    height = images[0].height
    max_dimension = float(max(width, height))
    problem.set_parameter_lower_bound(state.intrinsics, 0, 0.2 * max_dimension)
    problem.set_parameter_lower_bound(state.intrinsics, 1, 0.2 * max_dimension)
    problem.set_parameter_upper_bound(state.intrinsics, 0, 3.0 * max_dimension)
    problem.set_parameter_upper_bound(state.intrinsics, 1, 3.0 * max_dimension)
    problem.set_parameter_lower_bound(state.intrinsics, 2, 0.25 * width)
    problem.set_parameter_upper_bound(state.intrinsics, 2, 0.75 * width)
    problem.set_parameter_lower_bound(state.intrinsics, 3, 0.25 * height)
    problem.set_parameter_upper_bound(state.intrinsics, 3, 0.75 * height)
    for distortion_index in range(4, 8):
        problem.set_parameter_lower_bound(state.intrinsics, distortion_index, -1.0)
        problem.set_parameter_upper_bound(state.intrinsics, distortion_index, 1.0)

    options = pyceres.SolverOptions()
    options.linear_solver_type = pyceres.LinearSolverType.DENSE_SCHUR
    options.max_num_iterations = max_iterations
    options.minimizer_progress_to_stdout = True
    options.num_threads = 1
    options.update_state_every_iteration = True
    options.function_tolerance = loss_relative_tolerance

    callback = LossConvergenceCallback(loss_relative_tolerance, loss_patience)
    options.callbacks = [callback]

    summary = pyceres.SolverSummary()
    pyceres.solve(options, problem, summary)
    cost_history = callback.cost_history or [float(summary.initial_cost), float(summary.final_cost)]
    return SolverDiagnostics(
        summary=summary.BriefReport(),
        iterations=int(summary.num_successful_steps + summary.num_unsuccessful_steps + 1),
        initial_cost=float(summary.initial_cost),
        final_cost=float(summary.final_cost),
        cost_history=[float(cost) for cost in cost_history],
    )


def calibrate_camera(
    camera_name: str,
    data_root: Path,
    image_root: Path | None,
    min_detections: int,
    min_shared: int,
    max_reprojection_error: float,
    max_iterations: int,
    huber_delta: float,
    loss_relative_tolerance: float,
    loss_patience: int,
    output_dir: Path,
    show_3d: bool,
    known_targets3d: Dict[int, np.ndarray] | None,
    valid_ids: Set[int] | None,
    max_id_hamming_distance: int,
    force_detections: bool,
) -> CameraCalibrationResult:
    image_paths = iter_camera_images(data_root, camera_name, image_root)
    if not image_paths:
        raise RuntimeError(f"No images found for {camera_name} under {data_root}")

    print(f"collecting detections for {camera_name} from {len(image_paths)} images", flush=True)
    detections_dir_path = output_dir / camera_name / "annotated_detections"
    cache_path = output_dir / camera_name / "target_detections.txt"
    images = None if force_detections else load_cached_detections(cache_path, image_paths[0].parent)
    if images is None:
        images = load_image_detections(
            image_paths,
            detections_output_dir=detections_dir_path,
            valid_ids=valid_ids,
            max_id_hamming_distance=max_id_hamming_distance,
            codebook=valid_ids if known_targets3d is not None else None,
        )
        target_detections_path = save_target_detections(images, camera_name, output_dir)
    else:
        target_detections_path = cache_path
    bundle_images = filter_images_for_bundle(images, min_detections)
    if len(bundle_images) < 2:
        raise RuntimeError(f"Not enough images with more than {min_detections} detections for {camera_name}")

    fixed_point_ids: Set[int] | None = None
    if known_targets3d is not None:
        state, fixed_point_ids = initialize_known_target_state(bundle_images, known_targets3d, min_shared)
    else:
        state = choose_seed_state(bundle_images, min_shared)
    register_remaining_images(bundle_images, state, min_shared)

    observations = collect_observations(bundle_images, state, max_reprojection_error)
    if len(observations) < min_shared * 2:
        raise RuntimeError(f"Not enough geometrically consistent observations for {camera_name}")

    registered_before = len(state.poses)
    diagnostics = solve_bundle_adjustment(
        bundle_images,
        state,
        observations,
        max_iterations,
        huber_delta,
        loss_relative_tolerance,
        loss_patience,
        fixed_point_ids=fixed_point_ids,
    )

    # Second pass: try to register images that failed on noisy initial geometry.
    # After BA the 3D points and intrinsics are refined, so PnP may now succeed.
    register_remaining_images(bundle_images, state, min_shared)
    observations = collect_observations(bundle_images, state, max_reprojection_error)
    registered_after = len(state.poses)
    if registered_after > registered_before:
        print(
            f"second pass registered {registered_after - registered_before} additional images "
            f"({registered_after} total)",
            flush=True,
        )
        diagnostics = solve_bundle_adjustment(
            bundle_images,
            state,
            observations,
            max_iterations,
            huber_delta,
            loss_relative_tolerance,
            loss_patience,
            fixed_point_ids=fixed_point_ids,
        )
    # --- Robust refinement pass ---
    # Re-collect all possible observations (generous threshold) and run BA with
    # Cauchy loss to downweight outlier detections / triangulations.
    all_observations = collect_observations(bundle_images, state, max_reprojection_error * 3.0)
    if len(all_observations) >= len(observations):
        print(
            f"robust refinement: {len(all_observations)} observations "
            f"(was {len(observations)})",
            flush=True,
        )
        diagnostics = solve_bundle_adjustment(
            bundle_images,
            state,
            all_observations,
            max_iterations,
            huber_delta,
            loss_relative_tolerance,
            loss_patience,
            robust_loss="cauchy",
            cauchy_scale=3.0,
            fixed_point_ids=fixed_point_ids,
        )
        # Discard observations that remain outliers after robust optimisation
        observations = _filter_inlier_observations(state, all_observations, max_reprojection_error)
        print(
            f"robust refinement kept {len(observations)}/{len(all_observations)} inlier observations",
            flush=True,
        )
        # Final clean BA on inliers only (standard Huber)
        diagnostics = solve_bundle_adjustment(
            bundle_images,
            state,
            observations,
            max_iterations,
            huber_delta,
            loss_relative_tolerance,
            loss_patience,
            fixed_point_ids=fixed_point_ids,
        )

    reprojection_errors = compute_reprojection_errors(state, observations)
    if reprojection_errors.size == 0:
        raise RuntimeError(f"No reprojection errors available after optimization for {camera_name}")
    object_space_errors = compute_object_space_errors(state, observations)

    report_path = save_report(state, camera_name, (bundle_images[0].width, bundle_images[0].height), observations, diagnostics, output_dir)
    convergence_plot_path = save_convergence_plot(camera_name, diagnostics.cost_history, output_dir)
    scene_data_path = save_scene_data(state, camera_name, output_dir)
    rerun_recording_path = write_rerun_visualization(state, camera_name, (bundle_images[0].width, bundle_images[0].height), output_dir, show_3d)
    targets_3d_path, cameras_path = save_text_exports(state, camera_name, output_dir)
    colmap_dir_path = save_colmap_output(state, bundle_images, camera_name, output_dir)
    track_ids = {target_id for _, target_id, _ in observations}
    return CameraCalibrationResult(
        camera_name=camera_name,
        image_size=(bundle_images[0].width, bundle_images[0].height),
        used_images=len(bundle_images),
        registered_images=len(state.poses),
        tracks=len(track_ids),
        observations=len(observations),
        summary=diagnostics.summary,
        intrinsics=state.intrinsics.copy(),
        mean_reprojection_error=float(np.mean(reprojection_errors)),
        rms_reprojection_error=float(np.sqrt(np.mean(np.square(reprojection_errors)))),
        mean_object_space_error=float(np.mean(object_space_errors)) if object_space_errors.size else 0.0,
        rms_object_space_error=float(np.sqrt(np.mean(np.square(object_space_errors)))) if object_space_errors.size else 0.0,
        iterations=diagnostics.iterations,
        initial_cost=diagnostics.initial_cost,
        final_cost=diagnostics.final_cost,
        report_path=report_path,
        convergence_plot_path=convergence_plot_path,
        rerun_recording_path=rerun_recording_path,
        scene_data_path=scene_data_path,
        targets_3d_path=targets_3d_path,
        cameras_path=cameras_path,
        colmap_dir_path=colmap_dir_path,
        detections_dir_path=detections_dir_path,
        target_detections_path=target_detections_path,
    )


def detect_camera(
    camera_name: str,
    data_root: Path,
    image_root: Path | None,
    min_detections: int,
    output_dir: Path,
    valid_ids: Set[int] | None,
    max_id_hamming_distance: int,
    force_detections: bool,
    codebook: Set[int] | None = None,
    workers: int = 1,
) -> tuple[DetectionOnlyResult, List[ImageDetections]]:
    image_paths = iter_camera_images(data_root, camera_name, image_root)
    if not image_paths:
        raise RuntimeError(f"No images found for {camera_name} under {data_root}")

    print(f"collecting detections for {camera_name} from {len(image_paths)} images", flush=True)
    detections_dir_path = output_dir / camera_name / "annotated_detections"
    cache_path = output_dir / camera_name / "target_detections.txt"
    images = None if force_detections else load_cached_detections(cache_path, image_paths[0].parent)
    if images is None:
        images = load_image_detections(
            image_paths,
            detections_output_dir=detections_dir_path,
            valid_ids=valid_ids,
            max_id_hamming_distance=max_id_hamming_distance,
            codebook=codebook,
            workers=workers,
        )
        target_detections_path = save_target_detections(images, camera_name, output_dir)
    else:
        target_detections_path = cache_path
    return DetectionOnlyResult(
        camera_name=camera_name,
        total_images=len(image_paths),
        used_images=len(images),
        detections_dir_path=detections_dir_path,
        target_detections_path=target_detections_path,
    ), images


def format_result(result: CameraCalibrationResult) -> dict:
    intrinsics = result.intrinsics
    return {
        "camera": result.camera_name,
        "image_size": {"width": result.image_size[0], "height": result.image_size[1]},
        "used_images": result.used_images,
        "registered_images": result.registered_images,
        "tracks": result.tracks,
        "observations": result.observations,
        "iterations": result.iterations,
        "summary": result.summary,
        "initial_cost": result.initial_cost,
        "final_cost": result.final_cost,
        "avg_reprojection_error_px": result.mean_reprojection_error,
        "rms_reprojection_error_px": result.rms_reprojection_error,
        "avg_object_space_error_m": result.mean_object_space_error,
        "rms_object_space_error_m": result.rms_object_space_error,
        "report": str(result.report_path),
        "convergence_plot": str(result.convergence_plot_path),
        "rerun_recording": str(result.rerun_recording_path),
        "scene_data": str(result.scene_data_path),
        "targets_3d": str(result.targets_3d_path),
        "cameras_tum": str(result.cameras_path),
        "colmap_dir": str(result.colmap_dir_path),
        "annotated_detections": str(result.detections_dir_path),
        "target_detections": str(result.target_detections_path),
        "camera_parameters": {
            "fx": float(intrinsics[0]),
            "fy": float(intrinsics[1]),
            "cx": float(intrinsics[2]),
            "cy": float(intrinsics[3]),
            "k1": float(intrinsics[4]),
            "k2": float(intrinsics[5]),
            "p1": float(intrinsics[6]),
            "p2": float(intrinsics[7]),
        },
    }


def format_detection_result(result: DetectionOnlyResult) -> dict:
    return {
        "camera": result.camera_name,
        "total_images": result.total_images,
        "used_images": result.used_images,
        "annotated_detections": str(result.detections_dir_path),
        "target_detections": str(result.target_detections_path),
    }


def calibrate_with_combined_pipeline(args: argparse.Namespace, extra_args: Sequence[str]) -> int:
    """Single-camera calibration as the one-camera case of the combined pipeline."""
    from cct_calibration.run_combined import main as combined_main

    if args.image_root is None:
        print("ERROR: calibration needs --image-root pointing to a folder with the images of one camera; "
              "the legacy --data-root photos_* layout is only supported with --detect-only.", flush=True)
        return 1
    if args.cameras and args.cameras != [args.image_root.resolve().name]:
        print(f"note: the camera is named after its image folder ({args.image_root.resolve().name}).", flush=True)
    if args.show_3d:
        print("note: --show-3d is not available in the combined pipeline; inspect <output-dir>/colmap instead.",
              flush=True)
    argv = [
        "--image-root", str(args.image_root), "--output-dir", str(args.output_dir),
        # This entry point kept images with *more than* N detections; the combined one uses >= N.
        "--min-detections", str(args.min_detections + 1),
        "--min-shared", str(args.min_shared),
        "--max-reprojection-error", str(args.max_reprojection_error),
        "--max-iterations", str(args.max_iterations),
        "--huber-delta", str(args.huber_delta),
        "--loss-tolerance", str(args.loss_relative_tolerance),
        "--loss-patience", str(args.loss_patience),
        "--max-id-hamming-distance", str(args.max_id_hamming_distance),
        "--workers", str(args.workers),
    ]
    if args.targets3d is not None:
        # As before, known targets initialise the camera directly (no SfM).
        argv += ["--targets3d", str(args.targets3d), "--skip-sfm"]
    if args.valid_ids_file is not None:
        argv += ["--valid-ids-file", str(args.valid_ids_file)]
    if args.valid_id_max is not None:
        argv += ["--valid-id-max", str(args.valid_id_max)]
    if args.force_detections:
        argv.append("--force-detections")
    return combined_main([*argv, *[a for a in extra_args if a != "--skip-sfm" or args.targets3d is None]])


def main() -> int:
    args, extra_args = parse_args()
    if not args.detect_only:
        return calibrate_with_combined_pipeline(args, extra_args)
    if extra_args:
        print(f"ERROR: unrecognized arguments for --detect-only: {' '.join(extra_args)}", flush=True)
        return 2
    known_targets3d = load_targets3d(args.targets3d) if args.targets3d is not None else None
    valid_ids = resolve_valid_ids(args.valid_ids_file, args.valid_id_max)
    if valid_ids is None and known_targets3d is not None:
        valid_ids = set(known_targets3d.keys())

    image_root = args.image_root.resolve() if args.image_root is not None else None
    if image_root is not None:
        if args.cameras and len(args.cameras) != 1:
            raise RuntimeError("--image-root supports exactly one camera name")
        cameras = args.cameras or [image_root.name]
    else:
        cameras = args.cameras or ["camera_0", "camera_1"]

    if args.detect_only:
        detection_results: List[dict] = []
        detections_by_camera: List[tuple[str, List[ImageDetections]]] = []
        for camera_name in cameras:
            result, images = detect_camera(
                camera_name=camera_name,
                data_root=args.data_root,
                image_root=image_root,
                min_detections=args.min_detections,
                output_dir=args.output_dir,
                valid_ids=valid_ids,
                max_id_hamming_distance=args.max_id_hamming_distance,
                force_detections=args.force_detections,
                codebook=valid_ids if known_targets3d is not None else None,
                workers=args.workers if args.workers > 0 else min(8, os.cpu_count() or 1),
            )
            result_dict = format_detection_result(result)
            detection_results.append(result_dict)
            detections_by_camera.append((camera_name, images))
            print(json.dumps(result_dict, indent=2), flush=True)

        combined_detections_path = save_combined_target_detections(detections_by_camera, args.output_dir)
        combined_result = {
            "detect_only": True,
            "cameras": detection_results,
            "combined_target_detections": str(combined_detections_path),
        }
        summary_path = args.output_dir / "detection_summary.json"
        args.output_dir.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(combined_result, indent=2), encoding="utf-8")
        print(json.dumps(combined_result, indent=2), flush=True)
        return 0

    results: List[dict] = []
    for camera_name in cameras:
        result = calibrate_camera(
            camera_name=camera_name,
            data_root=args.data_root,
            image_root=image_root,
            min_detections=args.min_detections,
            min_shared=args.min_shared,
            max_reprojection_error=args.max_reprojection_error,
            max_iterations=args.max_iterations,
            huber_delta=args.huber_delta,
            loss_relative_tolerance=args.loss_relative_tolerance,
            loss_patience=args.loss_patience,
            output_dir=args.output_dir,
            show_3d=args.show_3d,
            known_targets3d=known_targets3d,
            valid_ids=valid_ids,
            max_id_hamming_distance=args.max_id_hamming_distance,
            force_detections=args.force_detections,
        )
        result_dict = format_result(result)
        results.append(result_dict)
        output_dir = args.output_dir
        output_dir.mkdir(parents=True, exist_ok=True)
        summary_path = output_dir / f"{camera_name}_summary.json"
        summary_path.write_text(json.dumps(result_dict, indent=2), encoding="utf-8")
        print(json.dumps(result_dict, indent=2), flush=True)

    if len(results) > 1:
        print(json.dumps({"results": results}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
