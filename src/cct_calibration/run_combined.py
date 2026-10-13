"""Combined multi-camera CCT calibration pipeline.

Pipeline:
  1. Run SfM via multicamera-calibration to obtain initial camera poses.
  2. Run CCT target detection on all images.
  3. Initialize a multi-camera CalibrationState from SfM poses + CCT detections.
  4. Joint bundle adjustment of all cameras with per-camera intrinsics and
     a fixed relative-pose constraint within each rig frame.
"""

from __future__ import annotations

import argparse
import copy
import csv
import gc
import json
import math
import os
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Sequence, Set, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pyceres
import pycolmap
import yaml
from cct_detect.detector import CCTDetector

from cct_calibration.run import (
    CalibrationState,
    ImageDetections,
    LossConvergenceCallback,
    ReprojectionCost,
    SolverDiagnostics,
    _load_detections_cct_detect,
    camera_center_from_pose,
    collect_observations,
    compute_reprojection_errors,
    distortion_vector,
    filter_images_for_bundle,
    initialize_known_target_state,
    intrinsics_matrix,
    load_refinement_cache,
    normalize_target_id,
    project_point,
    rotation_matrix_from_pose,
    rotation_matrix_to_quaternion,
    save_convergence_plot,
    save_refinement_cache,
    save_target_detections,
    triangulate_two_views,
    _filter_inlier_observations,
)
from cct_detect.refinement import prepare_refinement_image, refine_projected_center
from cct_calibration.run import _detector_for, _init_detection_worker

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def list_images(directory: Path) -> List[Path]:
    """Image files directly inside ``directory`` (case-insensitive suffixes)."""
    if not directory.is_dir():
        return []
    return sorted(p for p in directory.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES)


def parse_known_baseline_argument(raw: object) -> tuple[float | None, np.ndarray | None]:
    """Normalize the CLI baseline input.

    Accepts either a single scalar or three XYZ values. Scalar values are
    treated as a magnitude for the existing scale-based path, while three
    values are applied directly as the relative-pose translation components.
    """
    if raw is None:
        return None, None

    if isinstance(raw, str):
        raw = raw.replace(",", " ").split()

    if isinstance(raw, np.ndarray):
        values = [float(v) for v in raw.reshape(-1)]
    elif isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)):
        values = [float(v) for v in raw]
    else:
        values = [float(raw)]

    if len(values) == 1:
        if not np.isfinite(values[0]) or values[0] <= 0:
            raise ValueError("--known-baseline scalar must be finite and positive")
        return float(values[0]), None
    if len(values) == 3:
        vector = np.asarray(values, dtype=np.float64)
        if not np.all(np.isfinite(vector)) or float(np.linalg.norm(vector)) <= 1e-12:
            raise ValueError("--known-baseline vector must be finite and non-zero")
        return None, vector
    raise ValueError("--known-baseline must be a single scalar or three XYZ values")


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class MultiCameraState:
    """State for joint multi-camera calibration.

    Each camera has its own intrinsics vector (8-element OPENCV model).
    ``rig_poses`` stores the 6-DOF pose (Rodrigues + translation) of camera-0
    for each rig frame.  The pose of camera *k* in rig frame *f* is computed as:

        pose_cam_k = compose_poses(rig_poses[f], relative_poses[k])

    ``relative_poses[0]`` is always the identity (cam-0 is the reference).
    """
    camera_intrinsics: Dict[str, np.ndarray]          # cam_name -> [fx,fy,cx,cy,k1,k2,p1,p2]
    rig_poses: Dict[str, np.ndarray]                   # frame_id (timestamp) -> 6-vector (cam-0 pose)
    relative_poses: Dict[str, np.ndarray]              # cam_name -> 6-vector relative to cam-0
    points: Dict[int, np.ndarray]                      # target_id -> 3D point
    camera_names: List[str]                            # ordered camera names


@dataclass
class MultiCameraResult:
    camera_results: Dict[str, dict]   # per-camera intrinsics + errors
    total_observations: int
    total_tracks: int
    total_registered_frames: int
    mean_reprojection_error: float
    rms_reprojection_error: float
    summary: str
    iterations: int
    initial_cost: float
    final_cost: float


@dataclass
class MetricInlierFilterStats:
    total_observations: int
    kept_observations: int
    excluded_observations: int
    invalid_observations: int
    median_residual: float
    mad: float
    threshold: float
    mad_scale: float
    spread_source: str = "mad"


@dataclass
class AdjustmentDiagnostics:
    """Linearised quality information for the final bundle adjustment.

    Covariances are marginal covariances: frame poses and free object points
    remain in the normal equations as nuisance parameters.  Fixed parameters
    have NaN covariance entries and are identified by ``parameter_status``.
    """

    parameter_names: List[str]
    parameter_initial: np.ndarray
    parameter_final: np.ndarray
    parameter_status: List[str]
    covariance: np.ndarray
    correlation: np.ndarray
    sigma0_px: float
    dof: int
    variance_factor: float
    observation_sigma_px: float | None
    chi_square_statistic: float | None
    chi_square_p_value: float | None
    chi_square_consistent_95: bool | None
    condition_number: float
    leverage_uv: np.ndarray
    local_redundancy_uv: np.ndarray
    standardized_residuals_uv: np.ndarray
    covariance_method: str
    warnings: List[str]
    jacobian_rows: int = 0
    jacobian_columns: int = 0
    numerical_rank: int | None = None
    rank_tolerance: float | None = None
    rank_verified: bool = False
    nullity: int | None = None
    singular_values: np.ndarray | None = None
    weakest_singular_values: np.ndarray | None = None
    exact_left_vectors: np.ndarray | None = None


# ---------------------------------------------------------------------------
# Pose composition utilities
# ---------------------------------------------------------------------------

def compose_poses(pose_a: np.ndarray, pose_b: np.ndarray) -> np.ndarray:
    """Compose two world-to-camera poses:  result = B ∘ A.

    If A maps world→cam0 and B maps cam0→camK, the result maps world→camK.
    """
    R_a = rotation_matrix_from_pose(pose_a)
    t_a = pose_a[3:]
    R_b = rotation_matrix_from_pose(pose_b)
    t_b = pose_b[3:]
    R = R_b @ R_a
    t = R_b @ t_a + t_b
    rvec, _ = cv2.Rodrigues(R)
    return np.concatenate([rvec.reshape(3), t]).astype(np.float64)


def compose_pose_from_parts(
    rig_pose: np.ndarray,
    rel_rvec: np.ndarray,
    rel_tvec: np.ndarray,
) -> np.ndarray:
    """Compose a rig pose with an explicit relative rotation/translation pair."""
    R_a = rotation_matrix_from_pose(rig_pose)
    t_a = rig_pose[3:]
    rel_rvec = np.asarray(rel_rvec, dtype=np.float64).reshape(3)
    rel_tvec = np.asarray(rel_tvec, dtype=np.float64).reshape(3)
    R_b, _ = cv2.Rodrigues(rel_rvec)
    R = R_b @ R_a
    t = R_b @ t_a + rel_tvec
    rvec, _ = cv2.Rodrigues(R)
    return np.concatenate([rvec.reshape(3), t]).astype(np.float64)


def identity_pose() -> np.ndarray:
    return np.zeros(6, dtype=np.float64)


def remove_relative_pose(abs_pose: np.ndarray, rel_pose: np.ndarray) -> np.ndarray:
    """Recover the rig world-to-cam0 pose from an absolute camera pose.

    Given abs_pose = compose_poses(rig_pose, rel_pose), solve for rig_pose.
    """
    R_abs = rotation_matrix_from_pose(abs_pose)
    t_abs = abs_pose[3:]
    R_rel = rotation_matrix_from_pose(rel_pose)
    t_rel = rel_pose[3:]
    R_rig = R_rel.T @ R_abs
    t_rig = R_rel.T @ (t_abs - t_rel)
    rvec, _ = cv2.Rodrigues(R_rig)
    return np.concatenate([rvec.reshape(3), t_rig]).astype(np.float64)


# ---------------------------------------------------------------------------
# Step 1 — SfM via pycolmap
# ---------------------------------------------------------------------------

def run_sfm(
    image_root: Path,
    camera_names: List[str],
    sfm_output_dir: Path,
    initial_focal: float | None = None,
    known_baseline: float | None = None,
    force: bool = False,
) -> Tuple[Dict[str, np.ndarray], Dict[str, Dict[str, np.ndarray]]]:
    """Run pycolmap SfM and return per-camera intrinsics and per-image poses.

    Returns
    -------
    intrinsics_by_camera : dict  cam_name -> 8-vector [fx,fy,cx,cy,k1,k2,p1,p2]
    poses_by_camera : dict  cam_name -> { image_stem: 6-vector pose }
    """
    db_path = sfm_output_dir / "sfm.db"
    colmap_image_root = sfm_output_dir / "images"
    recon_output = sfm_output_dir / "reconstruction"

    # Determine image dimensions from first image
    first_cam_dir = image_root / camera_names[0]
    first_img_path = list_images(first_cam_dir)[0]
    first_img = cv2.imread(str(first_img_path))
    img_h, img_w = first_img.shape[:2]

    # Camera ID mapping (deterministic)
    camera_id_map: Dict[str, int] = {cn: idx + 1 for idx, cn in enumerate(camera_names)}

    # If no initial_focal provided, use a common heuristic: max(w,h) * 0.5
    if initial_focal is None:
        initial_focal = max(img_w, img_h) * 0.5

    # Check for cached reconstruction
    images_txt = recon_output / "images.txt"
    cameras_txt = recon_output / "cameras.txt"
    use_cache = (not force) and images_txt.exists() and cameras_txt.exists()

    if use_cache:
        print("  reloading existing SfM reconstruction...", flush=True)
        reconstruction = pycolmap.Reconstruction()
        reconstruction.read_text(str(recon_output))
    else:
        # Copy images into COLMAP directory structure
        rig_dir = colmap_image_root / "rig1"
        for cam_name in camera_names:
            dst = rig_dir / cam_name
            dst.mkdir(parents=True, exist_ok=True)
            src_dir = image_root / cam_name
            for img in list_images(src_dir):
                target = dst / img.name
                if not target.exists():
                    shutil.copy2(str(img), str(target))

        # Clean stale DB
        if db_path.exists():
            try:
                os.remove(str(db_path))
            except PermissionError:
                import gc; gc.collect()
                os.remove(str(db_path))

        db = pycolmap.Database.open(str(db_path))

        # Create cameras
        for cam_name in camera_names:
            cam_config = {
                "model": "OPENCV",
                "width": img_w,
                "height": img_h,
                "params": [initial_focal, initial_focal, img_w / 2.0, img_h / 2.0,
                            0.0, 0.0, 0.0, 0.0],
            }
            camera = pycolmap.Camera(cam_config)
            db.write_camera(camera)

        # Create rig (cam0 is reference)
        colmap_rig = pycolmap.Rig({"rig_id": 1})
        ref_sensor = pycolmap.sensor_t({
            "type": pycolmap.SensorType.CAMERA,
            "id": camera_id_map[camera_names[0]],
        })
        colmap_rig.add_ref_sensor(ref_sensor)

        baseline_value, baseline_vector = parse_known_baseline_argument(known_baseline)
        for index, cam_name in enumerate(camera_names[1:], start=1):
            sensor = pycolmap.sensor_t({
                "type": pycolmap.SensorType.CAMERA,
                "id": camera_id_map[cam_name],
            })
            if baseline_vector is not None and index == 1:
                translation = [baseline_vector[0], baseline_vector[1], baseline_vector[2]]
            else:
                # A supplied baseline is defined only for cam0-cam1.  Other
                # sensors retain an ordinary initialization and are not
                # silently assigned the same translation.
                baseline_guess = baseline_value if (baseline_value is not None and index == 1) else 0.3
                translation = [-baseline_guess, 0.0, 0.0]
            transform = pycolmap.Rigid3d(
                rotation=pycolmap.Rotation3d([0.0, 0.0, 0.0, 1.0]),
                translation=translation,
            )
            colmap_rig.add_sensor(sensor, transform)
        db.write_rig(colmap_rig)

        # Create images and frames
        frames_dir = image_root / camera_names[0]
        frame_stems = sorted([f.stem for f in list_images(frames_dir)])
        file_by_stem = {
            cam_name: {f.stem: f.name for f in list_images(image_root / cam_name)}
            for cam_name in camera_names
        }
        image_id = 1
        frame_id = 1
        for stem in frame_stems:
            colmap_frame = pycolmap.Frame({"frame_id": frame_id, "rig_id": 1})
            frame_id += 1
            for cam_name in camera_names:
                if stem not in file_by_stem[cam_name]:
                    continue
                img_name = f"rig1/{cam_name}/{file_by_stem[cam_name][stem]}"
                image = pycolmap.Image(
                    name=img_name,
                    points2D=np.empty((0, 2), dtype=np.float64),
                    camera_id=camera_id_map[cam_name],
                    image_id=image_id,
                )
                db.write_image(image, use_image_id=False)
                colmap_frame.add_data_id(image.data_id)
                image_id += 1
            db.write_frame(colmap_frame)

        try:
            db.close()
        except AttributeError:
            pass
        del db

        # Clean old reconstruction
        if recon_output.exists():
            shutil.rmtree(str(recon_output))

        # Use bounded-memory extraction settings.  Full-resolution extraction
        # with the default 8192 SIFT features per image can leave incremental
        # mapping with several gigabytes of feature/match allocations on a
        # large rig sequence.
        extraction_opts = pycolmap.FeatureExtractionOptions()
        extraction_opts.max_image_size = 3000
        extraction_opts.num_threads = 1
        extraction_opts.sift.max_num_features = 2048
        extraction_opts.sift.first_octave = 0

        pycolmap.extract_features(
            db_path, colmap_image_root,
            extraction_options=extraction_opts,
        )

        # Sequential matching is much faster than exhaustive for ordered rig data
        seq_opts = pycolmap.SequentialPairingOptions()
        seq_opts.overlap = 10
        seq_opts.quadratic_overlap = False
        seq_opts.loop_detection = False
        seq_opts.num_threads = 1
        pycolmap.match_sequential(db_path, pairing_options=seq_opts)

        # Keep one model and one worker: disconnected-model bookkeeping and
        # per-thread BA workspaces are a common source of ``bad allocation``
        # failures for long image sequences.  The calibration BA below still
        # performs the full high-precision solve after this initialization.
        mapping_opts = pycolmap.IncrementalPipelineOptions()
        mapping_opts.multiple_models = False
        mapping_opts.max_num_models = 1
        mapping_opts.extract_colors = False
        mapping_opts.num_threads = 1
        mapping_opts.mapper.num_threads = 1
        mapping_opts.mapper.ba_local_num_images = 4
        mapping_opts.init_num_trials = 80
        mapping_opts.ba_use_gpu = False
        try:
            maps = pycolmap.incremental_mapping(
                db_path, colmap_image_root, recon_output, options=mapping_opts,
            )
        except MemoryError as exc:
            # Release Python-side handles before surfacing a useful recovery
            # instruction.  The caller can rerun with --skip-sfm and known
            # targets when the image graph itself cannot fit in memory.  Do
            # not leave a partial reconstruction that a later non-forced run
            # might mistake for a valid cache.
            gc.collect()
            if recon_output.exists():
                shutil.rmtree(str(recon_output), ignore_errors=True)
            raise RuntimeError(
                "SfM incremental mapping ran out of memory even with bounded "
                "single-thread settings; reduce the image set/resolution or "
                "use --skip-sfm with --targets3d."
            ) from exc

        if not maps:
            raise RuntimeError("SfM reconstruction failed — no maps produced.")

        reconstruction = maps[0]
        reconstruction.write_text(recon_output)

    # Extract camera intrinsics (use SfM values directly as initialisation)
    intrinsics_by_camera: Dict[str, np.ndarray] = {}
    for cam_name in camera_names:
        cam_id = camera_id_map[cam_name]
        colmap_cam = reconstruction.cameras[cam_id]
        params = np.array(colmap_cam.params, dtype=np.float64)
        intrinsics_by_camera[cam_name] = params

    # Extract per-image poses
    poses_by_camera: Dict[str, Dict[str, np.ndarray]] = {cn: {} for cn in camera_names}
    for img_id, colmap_img in reconstruction.images.items():
        name = colmap_img.name  # e.g. "rig1/cam0/1769095475851869445.jpg"
        parts = Path(name).parts  # ('rig1', 'cam0', 'xxx.jpg')
        if len(parts) < 3:
            continue
        cam_name = parts[1]
        stem = Path(parts[2]).stem

        # COLMAP stores world-to-camera as quaternion [qw, qx, qy, qz] + translation
        rigid = colmap_img.cam_from_world()
        R = rigid.rotation.matrix()
        t = np.array(rigid.translation, dtype=np.float64)
        rvec, _ = cv2.Rodrigues(R)
        pose = np.concatenate([rvec.reshape(3), t]).astype(np.float64)
        poses_by_camera[cam_name][stem] = pose

    print(f"SfM: reconstructed {sum(len(v) for v in poses_by_camera.values())} images "
          f"across {len(camera_names)} cameras", flush=True)
    return intrinsics_by_camera, poses_by_camera


# ---------------------------------------------------------------------------
# Step 2 — CCT detection
# ---------------------------------------------------------------------------

def _load_cached_detections(
    cache_path: Path,
    image_root_cam: Path,
) -> List[ImageDetections] | None:
    """Try to load cached detections from a target_detections.txt file.

    Returns None if the cache file doesn't exist or is unreadable.
    """
    if not cache_path.exists():
        return None

    try:
        lines = cache_path.read_text(encoding="utf-8").splitlines()
    except Exception:
        return None

    if not lines or not lines[0].startswith("image_name"):
        return None

    # Parse into per-image groups
    from collections import OrderedDict
    grouped: Dict[str, Dict[int, np.ndarray]] = OrderedDict()
    for line in lines[1:]:
        line = line.strip()
        if not line:
            continue
        parts = line.split("\t")
        if len(parts) < 4:
            continue
        img_name, tid_str, x_str, y_str = parts[0], parts[1], parts[2], parts[3]
        grouped.setdefault(img_name, {})[int(tid_str)] = np.array(
            [float(x_str), float(y_str)], dtype=np.float64
        )

    # Build ImageDetections, reading dimensions from the first image
    result: List[ImageDetections] = []
    width, height = 0, 0
    for idx, (img_name, dets) in enumerate(grouped.items()):
        img_path = image_root_cam / img_name
        if width == 0 and img_path.exists():
            img = cv2.imread(str(img_path))
            if img is not None:
                height, width = img.shape[:2]
        result.append(ImageDetections(
            image_path=img_path,
            image_index=idx,
            width=width,
            height=height,
            detections=dets,
        ))

    evidence_count = load_refinement_cache(result, cache_path.parent)
    print(
        f"  loaded {len(result)} images from cache: {cache_path} "
        f"({evidence_count} detections ready for centre refinement)",
        flush=True,
    )
    return result


def load_targets3d(path: Path) -> Dict[int, np.ndarray]:
    """Load known 3D target positions from a TSV file.

    Supported formats:

    1. Simple TSV/whitespace with optional header::

        target_id  x  y  z

    2. The Metashape export used in this workspace::

        Label Metashape  Label physical  Label actual  X  Y  Z  ...
    """
    points: Dict[int, np.ndarray] = {}
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines()
    if not lines:
        return points

    header = lines[0].strip().split("\t")
    if {"Label actual", "X", "Y", "Z"}.issubset(header):
        reader = csv.DictReader(lines, delimiter="\t")
        for row in reader:
            actual = (row.get("Label actual") or "").strip()
            if not actual:
                continue
            tid = int(actual)
            points[tid] = np.array(
                [float(row["X"]), float(row["Y"]), float(row["Z"])],
                dtype=np.float64,
            )
        print(f"loaded {len(points)} known 3D target positions from {path}", flush=True)
        return points

    for line in lines:
        line = line.strip()
        if not line or line.startswith("#") or line.startswith("target_id"):
            continue
        parts = line.split()
        if len(parts) < 4:
            continue
        tid = int(parts[0])
        points[tid] = np.array([float(parts[1]), float(parts[2]), float(parts[3])], dtype=np.float64)
    print(f"loaded {len(points)} known 3D target positions from {path}", flush=True)
    return points


def detect_all_cameras(
    image_root: Path,
    camera_names: List[str],
    output_dir: Path,
    valid_ids: Set[int] | None = None,
    max_id_hamming_distance: int = 0,
    min_detections: int = 5,
    force: bool = False,
    initial_intrinsics: Dict[str, np.ndarray] | None = None,
    codebook: Set[int] | None = None,
    workers: int = 1,
) -> Dict[str, List[ImageDetections]]:
    """Run CCT detection on all cameras.

    Returns dict: cam_name -> list of ImageDetections.
    Image indices are globally unique across cameras.
    ``codebook`` enables correlation decoding against the known valid IDs.
    """
    all_detections: Dict[str, List[ImageDetections]] = {}
    global_index = 0

    for cam_name in camera_names:
        cam_dir = image_root / cam_name
        image_paths = list_images(cam_dir)
        if not image_paths:
            print(f"  warning: no images found for {cam_name}", flush=True)
            continue

        # Try loading from cache first
        cache_path = output_dir / cam_name / "target_detections.txt"
        cached = None if force else _load_cached_detections(cache_path, cam_dir)

        if cached is not None:
            raw = cached
        else:
            print(f"detecting CCT targets in {cam_name} ({len(image_paths)} images)...", flush=True)
            detections_dir = output_dir / cam_name / "annotated_detections"
            raw = _load_detections_cct_detect(
                image_paths,
                detections_dir,
                valid_ids,
                max_id_hamming_distance,
                initial_intrinsics=initial_intrinsics.get(cam_name) if initial_intrinsics else None,
                codebook=codebook,
                workers=workers,
            )
            # Save detections to cache
            save_target_detections(raw, cam_name, output_dir)

        # Reassign globally unique image indices and keep only images with enough detections
        cam_detections: List[ImageDetections] = []
        for det in raw:
            if len(det.detections) >= min_detections:
                cam_detections.append(ImageDetections(
                    image_path=det.image_path,
                    image_index=global_index,
                    width=det.width,
                    height=det.height,
                    detections=det.detections,
                    raw_detections={
                        target_id: point.copy()
                        for target_id, point in det.raw_detections.items()
                    },
                    refinement_ellipses=dict(det.refinement_ellipses),
                    refinement_status={
                        target_id: dict(status)
                        for target_id, status in det.refinement_status.items()
                    },
                    raw_cache_verified=det.raw_cache_verified,
                ))
            global_index += 1

        all_detections[cam_name] = cam_detections
        print(f"  {cam_name}: {len(cam_detections)} images with >= {min_detections} detections", flush=True)

    return all_detections


def _recover_verified_ellipse(
    image: np.ndarray,
    expected_id: int,
    seed: np.ndarray,
    detector: CCTDetector,
    valid_ids: Set[int] | None,
    max_id_hamming_distance: int,
) -> tuple | None:
    """Re-detect one known target in bounded ROIs around its observation."""
    height, width = image.shape[:2]
    for half_size in (64, 128, 256, 384):
        center_x, center_y = float(seed[0]), float(seed[1])
        left = max(0, int(np.floor(center_x - half_size)))
        top = max(0, int(np.floor(center_y - half_size)))
        right = min(width, int(np.ceil(center_x + half_size)))
        bottom = min(height, int(np.ceil(center_y + half_size)))
        if right - left < 32 or bottom - top < 32:
            continue
        candidates = detector.detect(image[top:bottom, left:right])
        associated: list[tuple] = []
        for candidate in candidates:
            normalized_id, _ = normalize_target_id(
                int(candidate.target_id), valid_ids, max_id_hamming_distance,
            )
            if normalized_id != expected_id:
                continue
            (local_x, local_y), axes, angle = candidate.ellipse
            global_center = np.array([local_x + left, local_y + top], dtype=np.float64)
            radius = max(axes) / 2.0
            if np.linalg.norm(global_center - seed) > max(2.0, 0.4 * radius):
                continue
            # The strict verifier needs the complete three-ring target.
            # Do not cache evidence whose outer ring touched the ROI edge.
            margin = 3.2 * radius + 2.0
            if min(local_x, local_y, right - left - local_x, bottom - top - local_y) < margin:
                continue
            associated.append((
                (float(global_center[0]), float(global_center[1])),
                (float(axes[0]), float(axes[1])),
                float(angle),
            ))
        if len(associated) == 1:
            return associated[0]
        if len(associated) > 1:
            return None
    return None


def _cached_model_is_current(
    images: Sequence[ImageDetections], current: np.ndarray,
) -> tuple[bool, float, float]:
    old_models: list[np.ndarray] = []
    sample_points: list[np.ndarray] = []
    for image_record in images:
        for target_id, point in image_record.detections.items():
            status = image_record.refinement_status.get(target_id, {})
            # Local ellipse recovery depends on the immutable image and
            # observation, not the camera model. Do not repeat a bounded
            # search that already exhausted all permitted ROIs.
            if (
                target_id not in image_record.refinement_ellipses
                and not status.get("recovery_exhausted", False)
            ):
                return False, float("inf"), float("inf")
            values = status.get("intrinsics")
            if values is None:
                return False, float("inf"), float("inf")
            model = np.asarray(values, dtype=np.float64)
            if model.shape != (8,) or not np.all(np.isfinite(model)):
                return False, float("inf"), float("inf")
            old_models.append(model)
            sample_points.append(np.asarray(point, dtype=np.float64))
    if not old_models or not sample_points:
        return False, float("inf"), float("inf")
    old = old_models[0]
    if any(not np.allclose(model, old, rtol=0.0, atol=1e-12) for model in old_models[1:]):
        return False, float("inf"), float("inf")
    points = np.asarray(sample_points, dtype=np.float64).reshape(-1, 1, 2)
    old_rays = cv2.undistortPoints(
        points, intrinsics_matrix(old), distortion_vector(old),
    ).reshape(-1, 2)
    new_rays = cv2.undistortPoints(
        points, intrinsics_matrix(current), distortion_vector(current),
    ).reshape(-1, 2)
    pixel_scale = np.array([current[0], current[1]], dtype=np.float64)
    shifts = np.linalg.norm((new_rays - old_rays) * pixel_scale, axis=1)
    shifts = shifts[np.isfinite(shifts)]
    if shifts.size == 0:
        return False, float("inf"), float("inf")
    p95 = float(np.percentile(shifts, 95))
    maximum = float(np.max(shifts))
    return p95 <= 0.02 and maximum <= 0.05, p95, maximum

def _refine_image_centres(
    image_record: ImageDetections,
    intrinsics: np.ndarray,
    valid_ids: Set[int] | None,
    max_id_hamming_distance: int,
    codebook: Set[int] | None,
):
    """Calibrated centre refinement of one image (process-safe).

    Returns the updated record, the (changed, accepted, unavailable,
    recovered) counters and the accepted displacements.
    """
    # Same decoder configuration as production detection, so a locally
    # recovered ellipse is verified under the same ID rules.
    detector = _detector_for(codebook)
    count = accepted = unavailable = recovered = 0
    displacements: list[float] = []
    image = cv2.imread(str(image_record.image_path))
    if image is None:
        unavailable += len(image_record.detections)
        return image_record, (count, accepted, unavailable, recovered), displacements
    raw_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    prepared_gray: np.ndarray | None = None
    for target_id in list(image_record.detections):
        previous = image_record.detections[target_id]
        ellipse = image_record.refinement_ellipses.get(target_id)
        if ellipse is None:
            ellipse = _recover_verified_ellipse(
                image, int(target_id), np.asarray(previous, dtype=np.float64), detector,
                valid_ids, max_id_hamming_distance,
            )
            if ellipse is not None:
                image_record.refinement_ellipses[target_id] = ellipse
                image_record.raw_detections[target_id] = np.asarray(
                    ellipse[0], dtype=np.float64,
                )
                recovered += 1
        if ellipse is None:
            image_record.refinement_status[target_id] = {
                "method": "unchanged",
                "intrinsics": [float(value) for value in intrinsics],
                "reason": "verified ellipse unavailable after bounded local recovery",
                "recovery_exhausted": True,
            }
            unavailable += 1
            continue
        image_record.raw_detections.setdefault(
            target_id, np.asarray(ellipse[0], dtype=np.float64),
        )
        if prepared_gray is None:
            prepared_gray = prepare_refinement_image(raw_gray)
        estimate = refine_projected_center(
            prepared_gray, ellipse, intrinsics, image_is_prepared=True,
        )
        radius = max(ellipse[1]) / 2.0
        measured = np.asarray(estimate.projected_center_px, dtype=np.float64)
        shift_from_ellipse = float(np.linalg.norm(measured - np.asarray(ellipse[0])))
        if (
            not estimate.valid
            or not np.all(np.isfinite(measured))
            or shift_from_ellipse > max(2.0, 0.5 * radius)
        ):
            image_record.refinement_status[target_id] = {
                "method": "unchanged",
                "intrinsics": [float(value) for value in intrinsics],
                "reason": estimate.reason if not estimate.valid else "centre displacement gate",
                "edge_rms_px": (
                    float(estimate.edge_rms_px)
                    if np.isfinite(estimate.edge_rms_px) else None
                ),
            }
            unavailable += 1
            continue
        displacement = float(np.linalg.norm(measured - previous))
        displacements.append(displacement)
        accepted += 1
        if displacement > 1e-9:
            image_record.detections[target_id] = measured
            count += 1
        image_record.refinement_status[target_id] = {
            "method": estimate.method,
            "intrinsics": [float(value) for value in intrinsics],
            "reason": estimate.reason,
            "edge_rms_px": (
                float(estimate.edge_rms_px)
                if np.isfinite(estimate.edge_rms_px) else None
            ),
        }
    return image_record, (count, accepted, unavailable, recovered), displacements


def refine_detected_centers_after_calibration(
    detections_by_camera: Dict[str, List[ImageDetections]],
    camera_intrinsics: Dict[str, np.ndarray],
    valid_ids: Set[int] | None = None,
    max_id_hamming_distance: int = 0,
    codebook: Set[int] | None = None,
    workers: int = 1,
) -> dict[str, int]:
    """Refine existing verified observations without full-frame redetection.

    Cached verified ellipses go directly to the calibrated conic estimator.
    Legacy ID/x/y caches recover missing ellipses inside bounded local ROIs,
    using the unchanged strict ring/code detector and requiring the same ID.
    Refinement can never add, remove, or relabel an observation.  Images are
    independent, so with ``workers`` > 1 they are refined in parallel worker
    processes; results are applied in input order and are identical.
    """
    from concurrent.futures import ProcessPoolExecutor


    updated: dict[str, int] = {}
    workers = max(1, int(workers))
    pool = None  # created on first use; cameras with current caches need none
    all_images = sum(len(images) for images in detections_by_camera.values())
    all_observations = sum(
        len(image.detections)
        for images in detections_by_camera.values()
        for image in images
    )
    print(
        "  calibrated centre refinement: refining existing detections; "
        "missing ellipse geometry is recovered from bounded image regions; "
        f"images={all_images}, observations={all_observations}",
        flush=True,
    )
    try:
        for cam_name, images in detections_by_camera.items():
            intrinsics = camera_intrinsics.get(cam_name)
            if intrinsics is None:
                continue
            cache_current, model_p95, model_max = _cached_model_is_current(images, intrinsics)
            if cache_current:
                updated[cam_name] = 0
                print(
                    f"    {cam_name}: cached refined coordinates reused "
                    f"(camera-model change p95={model_p95:.4f}px, max={model_max:.4f}px)",
                    flush=True,
                )
                continue
            count = 0
            accepted = 0
            unavailable = 0
            recovered = 0
            displacements: list[float] = []
            started = time.perf_counter()
            last_progress = started
            evidence_before = sum(
                len(set(image.detections) & set(image.refinement_ellipses)) for image in images
            )
            print(
                f"    {cam_name}: {len(images)} images, "
                f"cached detections={evidence_before}/{sum(len(image.detections) for image in images)}",
                flush=True,
            )
            arguments = [(record, intrinsics, valid_ids, max_id_hamming_distance, codebook) for record in images]
            if workers > 1 and pool is None:
                pool = ProcessPoolExecutor(max_workers=workers, initializer=_init_detection_worker)
            if pool is not None:
                results = pool.map(_refine_image_centres, *zip(*arguments))
            else:
                results = (_refine_image_centres(*item) for item in arguments)
            for image_number, (image_record, result) in enumerate(zip(images, results), start=1):
                refined, counters, image_displacements = result
                if refined is not image_record:
                    # Update the caller's record in place (other containers may share it).
                    for name in ("detections", "raw_detections", "refinement_ellipses", "refinement_status"):
                        mapping = getattr(image_record, name)
                        mapping.clear()
                        mapping.update(getattr(refined, name))
                count += counters[0]
                accepted += counters[1]
                unavailable += counters[2]
                recovered += counters[3]
                displacements.extend(image_displacements)
                now = time.perf_counter()
                if image_number == len(images) or image_number % 20 == 0 or now - last_progress >= 10.0:
                    elapsed = max(now - started, 1e-9)
                    rate = image_number / elapsed
                    eta = (len(images) - image_number) / max(rate, 1e-9)
                    print(
                        f"      {cam_name}: {image_number}/{len(images)} images; "
                        f"accepted={accepted}, recovered={recovered}, unavailable={unavailable}; "
                        f"{rate:.2f} images/s, ETA={eta / 60.0:.1f} min",
                        flush=True,
                    )
                    last_progress = now
            updated[cam_name] = count
            elapsed = time.perf_counter() - started
            if displacements:
                displacement_values = np.asarray(displacements, dtype=np.float64)
                displacement_text = (
                    f"median={np.median(displacement_values):.4f}px, "
                    f"p95={np.percentile(displacement_values, 95):.4f}px, "
                    f"max={np.max(displacement_values):.4f}px"
                )
            else:
                displacement_text = "no accepted coordinate updates"
            print(
                f"    {cam_name}: finished in {elapsed:.1f}s; changed={count}, "
                f"accepted={accepted}, locally recovered detections={recovered}, unavailable={unavailable}; "
                + displacement_text,
                flush=True,
            )
    finally:
        if pool is not None:
            pool.shutdown()
    return updated


def _persist_refined_detection_cache(
    detections_by_camera: Dict[str, List[ImageDetections]],
    output_dir: Path,
) -> None:
    """Merge refined coordinates into the cache without dropping raw frames.

    ``detections_by_camera`` contains only frames admitted by the current
    minimum-detection filter, whereas the original cache may also contain
    rejected frames.  Rewriting it from the filtered list would make changing
    that threshold on a later run irreversible, so update matching rows in
    place and preserve every other cached row.
    """
    for cam_name, images in detections_by_camera.items():
        cache_path = output_dir / cam_name / "target_detections.txt"
        updates = {
            (image.image_path.name, int(target_id)): np.asarray(point, dtype=np.float64)
            for image in images
            for target_id, point in image.detections.items()
        }
        if not updates:
            continue
        existing = cache_path.read_text(encoding="utf-8").splitlines() if cache_path.exists() else []
        lines = ["image_name\ttarget_id\tx_image\ty_image"]
        seen: set[tuple[str, int]] = set()
        for line in existing:
            stripped = line.strip()
            if not stripped or stripped.startswith("image_name"):
                continue
            fields = stripped.split()
            if len(fields) != 4:
                continue
            try:
                key = (fields[0], int(fields[1]))
            except ValueError:
                continue
            point = updates.get(key)
            if point is None:
                lines.append(stripped)
            else:
                lines.append(f"{key[0]}\t{key[1]}\t{float(point[0]):.6f}\t{float(point[1]):.6f}")
                seen.add(key)
        # A cache produced by an older filtered run may not contain every
        # currently retained key; append those coordinates deterministically.
        for key in sorted(set(updates) - seen):
            point = updates[key]
            lines.append(f"{key[0]}\t{key[1]}\t{float(point[0]):.6f}\t{float(point[1]):.6f}")
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        save_refinement_cache(images, cache_path.parent, merge_existing=True)


# ---------------------------------------------------------------------------
# Step 3 — Initialize multi-camera state from SfM poses
# ---------------------------------------------------------------------------

def _compute_relative_pose(
    poses_cam0: Dict[str, np.ndarray],
    poses_camk: Dict[str, np.ndarray],
) -> np.ndarray:
    """Estimate relative pose cam0→camK from overlapping frames.

    Returns the median relative transformation as a 6-vector.
    """
    shared = sorted(set(poses_cam0) & set(poses_camk))
    if not shared:
        raise RuntimeError("No shared frames between cameras for relative pose computation.")

    relative_translations: List[np.ndarray] = []
    relative_rotations: List[np.ndarray] = []
    for stem in shared:
        R0 = rotation_matrix_from_pose(poses_cam0[stem])
        t0 = poses_cam0[stem][3:]
        Rk = rotation_matrix_from_pose(poses_camk[stem])
        tk = poses_camk[stem][3:]
        # Relative: R_rel = Rk @ R0^T,  t_rel = tk - R_rel @ t0
        R_rel = Rk @ R0.T
        t_rel = tk - R_rel @ t0
        relative_rotations.append(R_rel)
        relative_translations.append(t_rel)

    # Take the median translation
    median_t = np.median(np.array(relative_translations), axis=0)

    # For rotation, take the one closest to the median Rodrigues vector
    rvecs = [cv2.Rodrigues(R)[0].reshape(3) for R in relative_rotations]
    median_rvec = np.median(np.array(rvecs), axis=0)
    best_idx = int(np.argmin([np.linalg.norm(rv - median_rvec) for rv in rvecs]))
    best_rvec = rvecs[best_idx]

    return np.concatenate([best_rvec, median_t]).astype(np.float64)


def initialize_multi_camera_state(
    sfm_intrinsics: Dict[str, np.ndarray],
    sfm_poses: Dict[str, Dict[str, np.ndarray]],
    detections_by_camera: Dict[str, List[ImageDetections]],
    camera_names: List[str],
    max_reprojection_error: float = 12.0,
    init_reproj_tolerance: float = 50.0,
    known_targets3d: Dict[int, np.ndarray] | None = None,
    known_baseline: float | None = None,
    reuse_known_world_poses: bool = False,
) -> MultiCameraState:
    """Build a MultiCameraState from SfM outputs and CCT detections.

    For each rig frame, we take cam-0's SfM pose as the rig pose.
    Camera intrinsics come from SfM as initial values.
    3D target positions are triangulated from multi-view CCT observations,
    or taken from ``known_targets3d`` when available.

    If ``reuse_known_world_poses`` is true, the input poses are fixed-K PnP
    solutions in the known target coordinate system and are reused directly;
    only timestamps missing from the reference camera are recovered from other
    solved cameras.
    """
    ref_cam = camera_names[0]

    # Compute relative poses
    relative_poses: Dict[str, np.ndarray] = {ref_cam: identity_pose()}
    baseline_value, baseline_vector = parse_known_baseline_argument(known_baseline)
    for cam_name in camera_names[1:]:
        relative_poses[cam_name] = _compute_relative_pose(
            sfm_poses[ref_cam], sfm_poses[cam_name],
        )

    # SfM is up to scale.  For a free SfM reconstruction, rescale every
    # relative translation once using only the cam0-cam1 baseline.  Known
    # metric target coordinates are already in a physical unit and must never
    # be rescaled.  In that branch an optional baseline is an explicit metric
    # constraint on cam0-cam1 only; it must not distort the other camera
    # translations.
    scale_factor = 1.0
    if baseline_value is not None or baseline_vector is not None:
        if len(camera_names) < 2:
            raise ValueError("A baseline constraint requires at least two cameras")
        estimated = relative_poses[camera_names[1]]
        norm = float(np.linalg.norm(estimated[3:]))
        if norm <= 1e-9:
            raise ValueError("Cannot establish scene scale from a zero cam0-cam1 baseline")
        requested = baseline_value if baseline_value is not None else float(np.linalg.norm(baseline_vector))
        if reuse_known_world_poses:
            if baseline_vector is not None:
                relative_poses[camera_names[1]][3:] = baseline_vector
            else:
                relative_poses[camera_names[1]][3:] *= requested / norm
            print(
                "known-world poses retained in metric coordinates; baseline "
                f"applied only to {ref_cam}->{camera_names[1]}", flush=True,
            )
        else:
            scale_factor = requested / norm
            for cam_name in camera_names[1:]:
                relative_poses[cam_name][3:] *= scale_factor

    if baseline_vector is not None and not reuse_known_world_poses:
        relative_poses[camera_names[1]][3:] = baseline_vector

    for cam_name in camera_names[1:]:
        print(f"relative pose {ref_cam}->{cam_name}: "
              f"t=[{relative_poses[cam_name][3]:.4f}, {relative_poses[cam_name][4]:.4f}, {relative_poses[cam_name][5]:.4f}]"
              f" (|t|={float(np.linalg.norm(relative_poses[cam_name][3:])):.4f} m)",
              flush=True)

    # Build rig poses from cam-0's poses, keyed by image stem.  SfM poses are
    # up to scale; known-world bootstrap poses already use the metric control
    # frame and must not be rescaled by a baseline option.
    rig_poses: Dict[str, np.ndarray] = {}
    for stem, pose in sfm_poses[ref_cam].items():
        p = pose.copy()
        if not reuse_known_world_poses and abs(scale_factor - 1.0) > 1e-12:
            p[3:] *= scale_factor
        rig_poses[stem] = p
    if reuse_known_world_poses:
        # Preserve timestamps for which the reference camera failed PnP but a
        # secondary camera solved successfully.  Relative poses are already in
        # the known target frame, so this is a direct frame conversion.
        for cam_name in camera_names[1:]:
            for stem, absolute_pose in sfm_poses[cam_name].items():
                if stem not in rig_poses:
                    rig_poses[stem] = remove_relative_pose(
                        absolute_pose, relative_poses[cam_name],
                    )

    # Build image_index → (cam_name, stem) mapping
    index_to_cam_stem: Dict[int, Tuple[str, str]] = {}
    for cam_name, dets in detections_by_camera.items():
        for det in dets:
            stem = det.image_path.stem
            index_to_cam_stem[det.image_index] = (cam_name, stem)

    # Collect all target observations with their absolute poses
    # observation = (target_id, cam_name, stem, 2D point)
    target_observations: Dict[int, List[Tuple[str, str, np.ndarray]]] = {}
    for cam_name, dets in detections_by_camera.items():
        cam_intrinsics = sfm_intrinsics[cam_name]
        for det in dets:
            stem = det.image_path.stem
            if stem not in rig_poses:
                continue
            for target_id, pt2d in det.detections.items():
                target_observations.setdefault(target_id, []).append(
                    (cam_name, stem, pt2d)
                )

    # Use known 3D points where available, triangulate the rest
    points: Dict[int, np.ndarray] = {}
    fixed_point_ids: Set[int] = set()
    if known_targets3d:
        for tid, pt3d in known_targets3d.items():
            if tid in target_observations:
                points[tid] = pt3d.copy()
                fixed_point_ids.add(tid)
        print(f"  {len(fixed_point_ids)} targets matched from known targets3D file", flush=True)

    if fixed_point_ids and not reuse_known_world_poses:
        detections_by_cam_stem: Dict[str, Dict[str, ImageDetections]] = {
            cam_name: {det.image_path.stem: det for det in dets}
            for cam_name, dets in detections_by_camera.items()
        }
        aligned_frames = 0
        for stem in list(rig_poses.keys()):
            best_pose: np.ndarray | None = None
            best_score: tuple[int, float] | None = None
            for cam_name in camera_names:
                det = detections_by_cam_stem.get(cam_name, {}).get(stem)
                if det is None:
                    continue

                shared_ids = sorted(set(det.detections) & fixed_point_ids)
                if len(shared_ids) < 6:
                    continue

                object_points = np.array([points[tid] for tid in shared_ids], dtype=np.float64)
                image_points = np.array([det.detections[tid] for tid in shared_ids], dtype=np.float64)
                solved, rotation_vec, translation_vec, inliers = cv2.solvePnPRansac(
                    object_points,
                    image_points,
                    intrinsics_matrix(sfm_intrinsics[cam_name]),
                    distortion_vector(sfm_intrinsics[cam_name]),
                    flags=cv2.SOLVEPNP_ITERATIVE,
                    reprojectionError=8.0,
                    iterationsCount=200,
                )
                if not solved or inliers is None or len(inliers) < 6:
                    continue

                inlier_indices = inliers.reshape(-1)
                refined, rotation_vec, translation_vec = cv2.solvePnP(
                    object_points[inlier_indices],
                    image_points[inlier_indices],
                    intrinsics_matrix(sfm_intrinsics[cam_name]),
                    distortion_vector(sfm_intrinsics[cam_name]),
                    rvec=rotation_vec,
                    tvec=translation_vec,
                    useExtrinsicGuess=True,
                    flags=cv2.SOLVEPNP_ITERATIVE,
                )
                if not refined:
                    continue

                abs_pose = np.concatenate(
                    [rotation_vec.reshape(3), translation_vec.reshape(3)]
                ).astype(np.float64)
                candidate_rig_pose = abs_pose if cam_name == ref_cam else remove_relative_pose(
                    abs_pose,
                    relative_poses[cam_name],
                )

                projected_errors: List[float] = []
                for target_id in shared_ids:
                    pose_for_camera = abs_pose if cam_name != ref_cam else candidate_rig_pose
                    projected = project_point(sfm_intrinsics[cam_name], pose_for_camera, points[target_id])
                    projected_errors.append(float(np.linalg.norm(projected - det.detections[target_id])))
                median_error = float(np.median(projected_errors)) if projected_errors else float("inf")
                score = (len(inlier_indices), -median_error)
                if best_score is None or score > best_score:
                    best_pose = candidate_rig_pose
                    best_score = score

            if best_pose is not None:
                rig_poses[stem] = best_pose
                aligned_frames += 1

        print(f"  re-estimated {aligned_frames} rig poses from known 3D targets", flush=True)
    elif fixed_point_ids:
        print(
            "  reused fixed-intrinsic known-world poses from bootstrap "
            "(missing-frame PnP is not repeated)",
            flush=True,
        )

    reject_reasons: Dict[str, int] = {"too_few": 0, "homogeneous": 0, "non_finite": 0,
                                      "depth": 0, "reprojection": 0}
    all_median_errors: List[float] = []  # for diagnostics
    for target_id, obs_list in target_observations.items():
        if target_id in points:  # already known
            continue
        if len(obs_list) < 2:
            reject_reasons["too_few"] += 1
            continue

        # Build projection matrices for DLT triangulation
        proj_matrices: List[np.ndarray] = []
        pts_2d: List[np.ndarray] = []
        for cam_name, stem, pt2d in obs_list:
            rig_pose = rig_poses[stem]
            abs_pose = compose_poses(rig_pose, relative_poses[cam_name])
            R = rotation_matrix_from_pose(abs_pose)
            t = abs_pose[3:].reshape(3, 1)
            K = intrinsics_matrix(sfm_intrinsics[cam_name])
            P = K @ np.hstack([R, t])
            proj_matrices.append(P)
            # DLT uses the ideal pinhole projection K[R|t]; measured centres
            # remain in the original distorted image for subsequent BA.
            undistorted = cv2.undistortPoints(
                np.asarray(pt2d, dtype=np.float64).reshape(1, 1, 2),
                K, distortion_vector(sfm_intrinsics[cam_name]), P=K,
            )[0, 0]
            pts_2d.append(undistorted)

        if not np.all(np.isfinite(pts_2d)):
            reject_reasons["non_finite"] += 1
            continue
        # Multi-view DLT triangulation (uses all views)
        A = np.zeros((2 * len(proj_matrices), 4), dtype=np.float64)
        for i, (P, pt) in enumerate(zip(proj_matrices, pts_2d)):
            A[2 * i] = pt[0] * P[2] - P[0]
            A[2 * i + 1] = pt[1] * P[2] - P[1]
        _, _, Vt = np.linalg.svd(A)
        pt4d = Vt[-1]

        if abs(pt4d[3]) < 1e-12:
            reject_reasons["homogeneous"] += 1
            continue
        pt3d = (pt4d[:3] / pt4d[3]).astype(np.float64)
        if not np.all(np.isfinite(pt3d)):
            reject_reasons["non_finite"] += 1
            continue

        # Depth check + reprojection check.
        # SfM poses are approximate, so use a generous tolerance here;
        # bundle adjustment will tighten things up later.
        n_depth_ok = 0
        reproj_errors: List[float] = []
        for cam_name, stem, pt2d in obs_list:
            rig_pose = rig_poses[stem]
            abs_pose = compose_poses(rig_pose, relative_poses[cam_name])
            R = rotation_matrix_from_pose(abs_pose)
            depth = (R @ pt3d + abs_pose[3:])[2]
            if depth > 1e-6:
                n_depth_ok += 1
            projected = project_point(sfm_intrinsics[cam_name], abs_pose, pt3d)
            err = float(np.linalg.norm(projected - pt2d))
            reproj_errors.append(err)

        # Reject only if majority of views have negative depth
        if n_depth_ok < max(1, len(obs_list) // 2):
            reject_reasons["depth"] += 1
            continue
        # Reject if median reprojection error is beyond the generous init tolerance
        median_err = float(np.median(reproj_errors))
        all_median_errors.append(median_err)
        if median_err > init_reproj_tolerance:
            reject_reasons["reprojection"] += 1
            continue

        points[target_id] = pt3d

    print(f"initialized {len(points)} 3D target points from {len(target_observations)} target tracks "
          f"({len(fixed_point_ids)} known, {len(points) - len(fixed_point_ids)} triangulated)", flush=True)
    print(f"  rejection reasons: {reject_reasons}", flush=True)
    if all_median_errors:
        me = np.array(all_median_errors)
        print(f"  median reprojection errors across targets: "
              f"min={me.min():.1f} median={np.median(me):.1f} "
              f"p95={np.percentile(me,95):.1f} max={me.max():.1f} px", flush=True)
    print(f"initialized {len(rig_poses)} rig frames", flush=True)

    state = MultiCameraState(
        camera_intrinsics={cn: sfm_intrinsics[cn].copy() for cn in camera_names},
        rig_poses=rig_poses,
        relative_poses=relative_poses,
        points=points,
        camera_names=camera_names,
    )
    state._fixed_point_ids = fixed_point_ids  # type: ignore[attr-defined]
    return state


def _known_target_correspondences(
    image: ImageDetections,
    known_targets3d: Dict[int, np.ndarray],
    min_shared: int,
) -> tuple[list[int], np.ndarray, np.ndarray] | None:
    """Return finite, non-collinear known-target correspondences for a frame."""
    ids = sorted(set(image.detections) & set(known_targets3d))
    required = max(6, int(min_shared))
    if len(ids) < required:
        return None

    object_points = np.ascontiguousarray(
        [known_targets3d[target_id] for target_id in ids], dtype=np.float64
    )
    image_points = np.ascontiguousarray(
        [image.detections[target_id] for target_id in ids], dtype=np.float64
    )
    if object_points.ndim != 2 or object_points.shape[1] != 3:
        return None
    if image_points.ndim != 2 or image_points.shape[1] != 2:
        return None
    if not np.isfinite(object_points).all() or not np.isfinite(image_points).all():
        return None

    # Collinear controls do not constrain a camera pose reliably.  Planar
    # control fields are valid: only the second image/object spread singular
    # value is checked here, not the third (plane-normal) singular value.
    if not _known_target_spread_is_valid(
        object_points, image_points, image.width, image.height,
    ):
        return None
    return ids, object_points, image_points


def _known_target_spread_is_valid(
    object_points: np.ndarray,
    image_points: np.ndarray,
    width: int | float,
    height: int | float,
) -> bool:
    """Check that surviving controls retain 2-D image and object spread."""
    if (
        len(object_points) < 2
        or len(image_points) < 2
        or not np.isfinite(width)
        or not np.isfinite(height)
        or float(width) <= 0.0
        or float(height) <= 0.0
    ):
        return False
    uv_norm = image_points / np.array([width, height], dtype=np.float64)
    _, image_singular, _ = np.linalg.svd(
        uv_norm - np.mean(uv_norm, axis=0), full_matrices=False,
    )
    _, object_singular, _ = np.linalg.svd(
        object_points - np.mean(object_points, axis=0), full_matrices=False,
    )
    if (len(image_singular) < 2 or image_singular[1] /
            max(image_singular[0], 1e-12) < 0.02):
        return False
    if (len(object_singular) < 2 or object_singular[1] /
            max(object_singular[0], 1e-12) < 0.01):
        return False
    return True


def _frame_descriptor(
    image: ImageDetections,
    correspondences: tuple[list[int], np.ndarray, np.ndarray],
    ordinal: int,
    total: int,
) -> np.ndarray:
    """Cheap geometry descriptor used before intrinsics are known."""
    _, _, image_points = correspondences
    normalized = image_points / np.array([image.width, image.height], dtype=np.float64)
    centroid = np.mean(normalized, axis=0)
    centered = normalized - centroid
    covariance = centered.T @ centered / max(len(normalized) - 1, 1)
    eigenvalues = np.linalg.eigvalsh(covariance)[::-1]
    hull = cv2.convexHull(normalized.astype(np.float32).reshape(-1, 1, 2))
    hull_area = float(cv2.contourArea(hull)) if len(hull) >= 3 else 0.0
    log_scale = float(np.log(max(np.sqrt(hull_area), 1e-6)))

    occupancy = np.zeros((6, 8), dtype=np.float64)
    cells_x = np.clip((normalized[:, 0] * occupancy.shape[1]).astype(int), 0, occupancy.shape[1] - 1)
    cells_y = np.clip((normalized[:, 1] * occupancy.shape[0]).astype(int), 0, occupancy.shape[0] - 1)
    occupancy[cells_y, cells_x] = 1.0
    time_fraction = float(ordinal / max(total - 1, 1))
    return np.concatenate([
        centroid,
        [log_scale, float(eigenvalues[0]), float(eigenvalues[1]), time_fraction],
        occupancy.reshape(-1),
    ])


def _select_bootstrap_frames(
    images: Sequence[ImageDetections],
    known_targets3d: Dict[int, np.ndarray],
    min_shared: int,
    max_calibration_frames: int = 24,
    max_candidate_frames: int = 96,
) -> tuple[list[ImageDetections], list[ImageDetections], list[ImageDetections]]:
    """Select a bounded, deterministic, coverage-diverse calibration subset."""
    records: list[tuple[ImageDetections, tuple[list[int], np.ndarray, np.ndarray], int]] = []
    for ordinal, image in enumerate(images):
        correspondences = _known_target_correspondences(image, known_targets3d, min_shared)
        if correspondences is not None:
            records.append((image, correspondences, ordinal))
    if not records:
        raise RuntimeError("No frames contain enough finite, non-collinear known 3D targets for bootstrap.")

    all_records = list(records)
    if len(records) < 8:
        raise RuntimeError(
            f"Only {len(records)} frames contain enough known-target geometry; "
            "at least eight are required for a bounded intrinsic bootstrap."
        )
    if len(records) > max_candidate_frames:
        # Keep a deterministic temporal scaffold, then fill it with the most
        # supported frames so short high-quality views are not lost.
        scaffold_count = max(1, max_candidate_frames // 2)
        scaffold = np.linspace(0, len(records) - 1, scaffold_count, dtype=int)
        selected_positions = set(int(index) for index in scaffold.tolist())
        support_order = sorted(
            range(len(records)),
            key=lambda index: len(records[index][1][0]),
            reverse=True,
        )
        for index in support_order:
            if len(selected_positions) >= max_candidate_frames:
                break
            selected_positions.add(index)
        records = [records[index] for index in sorted(selected_positions)]

    descriptors = np.asarray([
        _frame_descriptor(image, corr, ordinal, len(images))
        for image, corr, ordinal in records
    ], dtype=np.float64)
    scale = np.std(descriptors, axis=0)
    descriptors = (descriptors - np.mean(descriptors, axis=0)) / np.maximum(scale, 1e-9)

    target_count = min(max_calibration_frames, len(records))
    if target_count <= 0:
        raise RuntimeError("No eligible bootstrap frames remain after geometry checks.")

    support = np.asarray([len(corr[0]) for _, corr, _ in records], dtype=np.float64)
    hull_score = descriptors[:, 2]
    first = int(np.argmax(support + 0.1 * hull_score))
    chosen = [first]
    chosen_set = {first}
    while len(chosen) < target_count:
        best_index = None
        best_score = -float("inf")
        occupancy_counts = np.sum(
            np.asarray([
                _frame_descriptor(image, corr, ordinal, len(images))[-48:]
                for image, corr, ordinal in [records[index] for index in chosen]
            ], dtype=np.float64), axis=0,
        ) if chosen else np.zeros(48, dtype=np.float64)
        for index in range(len(records)):
            if index in chosen_set:
                continue
            # Keep pose/scale/time geometry separate from the binary
            # occupancy signature.  A raw 54-D Euclidean distance lets the
            # 48 occupancy bits dominate the much more useful geometric terms.
            geometry = descriptors[index, :6]
            occupancy_descriptor = descriptors[index, 6:]
            novelty = min(
                0.7 * float(np.linalg.norm(geometry - descriptors[other, :6]) / np.sqrt(6.0))
                + 0.3 * float(np.linalg.norm(
                    occupancy_descriptor - descriptors[other, 6:],
                ) / np.sqrt(48.0))
                for other in chosen
            )
            occupancy = _frame_descriptor(
                records[index][0], records[index][1], records[index][2], len(images)
            )[-48:]
            coverage_gain = float(np.sum(occupancy * np.maximum(0.0, 3.0 - occupancy_counts)) / 144.0)
            frame_support = min(float(len(records[index][1][0])) / 30.0, 1.0)
            score = 0.55 * novelty + 0.35 * coverage_gain + 0.10 * frame_support
            if score > best_score:
                best_score = score
                best_index = index
        if best_index is None:
            break
        chosen.append(best_index)
        chosen_set.add(best_index)

    chosen.sort()
    calibration = [records[index][0] for index in chosen]
    return calibration, [record[0] for record in records], [record[0] for record in all_records]


def _solve_known_target_pose(
    image: ImageDetections,
    known_targets3d: Dict[int, np.ndarray],
    intrinsics: np.ndarray,
    min_shared: int,
) -> tuple[np.ndarray, int, float] | None:
    """Estimate one world-to-camera pose with fixed bootstrap K and distortion."""
    correspondences = _known_target_correspondences(image, known_targets3d, min_shared)
    if correspondences is None:
        return None
    _, object_points, image_points = correspondences
    camera_matrix = intrinsics_matrix(intrinsics)
    distortion = distortion_vector(intrinsics)
    required_inliers = max(6, int(min_shared))

    try:
        solved, rotation_vec, translation_vec, inliers = cv2.solvePnPRansac(
            object_points,
            image_points,
            camera_matrix,
            distortion,
            flags=cv2.SOLVEPNP_EPNP,
            reprojectionError=8.0,
            confidence=0.999,
            iterationsCount=100,
        )
    except cv2.error:
        solved, rotation_vec, translation_vec, inliers = False, None, None, None

    if not solved or inliers is None or len(inliers) < required_inliers:
        try:
            solved, rotation_vec, translation_vec = cv2.solvePnP(
                object_points,
                image_points,
                camera_matrix,
                distortion,
                flags=cv2.SOLVEPNP_ITERATIVE,
            )
            inliers = np.arange(len(object_points), dtype=np.int32).reshape(-1, 1)
        except cv2.error:
            return None
    if not solved or rotation_vec is None or translation_vec is None:
        return None

    inlier_indices = np.asarray(inliers, dtype=np.int32).reshape(-1)
    try:
        if hasattr(cv2, "solvePnPRefineLM"):
            rotation_vec, translation_vec = cv2.solvePnPRefineLM(
                object_points[inlier_indices], image_points[inlier_indices],
                camera_matrix, distortion, rotation_vec, translation_vec,
                criteria=(cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 20, 1e-7),
            )
        else:
            _, rotation_vec, translation_vec = cv2.solvePnP(
                object_points[inlier_indices], image_points[inlier_indices],
                camera_matrix, distortion, rvec=rotation_vec, tvec=translation_vec,
                useExtrinsicGuess=True, flags=cv2.SOLVEPNP_ITERATIVE,
            )
    except cv2.error:
        return None

    def evaluate_pose(candidate_rvec: np.ndarray, candidate_tvec: np.ndarray):
        pose_value = np.concatenate(
            [candidate_rvec.reshape(3), candidate_tvec.reshape(3)]
        ).astype(np.float64)
        rotation_value = rotation_matrix_from_pose(pose_value)
        errors_value = np.asarray([
            float(np.linalg.norm(project_point(intrinsics, pose_value, point) - measured))
            for point, measured in zip(object_points, image_points)
        ], dtype=np.float64)
        depths_value = np.asarray([
            float((rotation_value @ point + pose_value[3:])[2])
            for point in object_points
        ], dtype=np.float64)
        supported_value = (
            np.isfinite(errors_value)
            & (errors_value <= 8.0)
            & np.isfinite(depths_value)
            & (depths_value > 1e-8)
        )
        return pose_value, errors_value, supported_value

    pose, errors, supported = evaluate_pose(rotation_vec, translation_vec)
    if int(np.count_nonzero(supported)) < required_inliers or float(np.mean(supported)) < 0.60:
        return None

    # Refine once on the support verified from actual reprojection/depth
    # residuals.  Never trust the inlier array returned by a fallback solver as
    # proof that every correspondence is an inlier.
    supported_indices = np.flatnonzero(supported).astype(np.int32)
    try:
        if hasattr(cv2, "solvePnPRefineLM"):
            rotation_vec, translation_vec = cv2.solvePnPRefineLM(
                object_points[supported_indices], image_points[supported_indices],
                camera_matrix, distortion, rotation_vec, translation_vec,
                criteria=(cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 20, 1e-7),
            )
        else:
            _, rotation_vec, translation_vec = cv2.solvePnP(
                object_points[supported_indices], image_points[supported_indices],
                camera_matrix, distortion, rvec=rotation_vec, tvec=translation_vec,
                useExtrinsicGuess=True, flags=cv2.SOLVEPNP_ITERATIVE,
            )
    except cv2.error:
        return None
    pose, errors, supported = evaluate_pose(rotation_vec, translation_vec)
    support_count = int(np.count_nonzero(supported))
    if support_count < required_inliers or support_count / len(object_points) < 0.60:
        return None
    supported_indices = np.flatnonzero(supported).astype(np.int32)
    if not _known_target_spread_is_valid(
        object_points[supported_indices], image_points[supported_indices],
        image.width, image.height,
    ):
        return None
    median_error = float(np.median(errors[supported])) if support_count else float("inf")
    if not np.isfinite(median_error):
        return None
    return pose, support_count, median_error


def _bootstrap_subset_variants(
    calibration_frames: Sequence[ImageDetections],
    candidate_frames: Sequence[ImageDetections],
    max_calibration_frames: int,
) -> list[list[ImageDetections]]:
    """Return a small deterministic set of bounded calibration alternatives."""
    eligible_count = len(candidate_frames)
    # Reserve at least eight frames for validation whenever the dataset is
    # larger than one bounded calibration subset.  This avoids the 25--31
    # frame corner case where a 24-frame fit leaves too few held-out frames.
    target_count = (
        min(max_calibration_frames, eligible_count - 8)
        if eligible_count > max_calibration_frames
        else min(max_calibration_frames, eligible_count)
    )
    if target_count == 0:
        return []
    variants: list[list[ImageDetections]] = []
    seen: set[tuple[str, ...]] = set()

    def add(frames: Sequence[ImageDetections]) -> None:
        if len(frames) < min(8, target_count):
            return
        ordered = list(frames[:target_count])
        key = tuple(image.image_path.stem for image in ordered)
        if key not in seen:
            seen.add(key)
            variants.append(ordered)

    add(calibration_frames)
    add(list(candidate_frames)[-target_count:])
    even_indices = np.linspace(0, len(candidate_frames) - 1, target_count, dtype=int)
    add([candidate_frames[int(index)] for index in even_indices.tolist()])
    add(list(candidate_frames)[:target_count])
    return variants


def bootstrap_from_known_targets(
    detections_by_camera: Dict[str, List[ImageDetections]],
    camera_names: List[str],
    known_targets3d: Dict[int, np.ndarray],
    min_shared: int,
) -> Tuple[Dict[str, np.ndarray], Dict[str, Dict[str, np.ndarray]]]:
    """Estimate per-camera intrinsics and absolute poses without SfM.

    This uses the known 3D targets directly for each camera, then converts the
    per-image poses to the same structures returned by ``run_sfm``.
    """
    intrinsics_by_camera: Dict[str, np.ndarray] = {}
    poses_by_camera: Dict[str, Dict[str, np.ndarray]] = {}

    for cam_name in camera_names:
        cam_detections = detections_by_camera.get(cam_name, [])
        calibration_frames, candidate_frames, all_eligible_frames = _select_bootstrap_frames(
            cam_detections,
            known_targets3d,
            min_shared,
            max_calibration_frames=24,
            max_candidate_frames=96,
        )
        print(
            f"  {cam_name}: evaluating bounded calibration subsets "
            f"({len(candidate_frames)}-frame candidate pool, {len(all_eligible_frames)} eligible total; "
            "all retained frames will receive fixed-K PnP)",
            flush=True,
        )

        # A single arbitrary subset can be numerically weak even when its
        # image occupancy looks good.  Evaluate a few deterministic alternatives
        # on held-out known-target frames, but never fall back to an unbounded
        # all-frame calibrateCamera call.
        best_state: CalibrationState | None = None
        best_score: tuple[int, float, float] | None = None
        subsets = _bootstrap_subset_variants(calibration_frames, all_eligible_frames, 24)
        for subset_index, subset in enumerate(subsets, start=1):
            try:
                state_candidate, _ = initialize_known_target_state(
                    subset, known_targets3d, min_shared,
                )
            except (cv2.error, ValueError) as exc:
                print(
                    f"    {cam_name}: subset {subset_index}/{len(subsets)} "
                    f"calibration failed ({exc}); trying the next bounded subset",
                    flush=True,
                )
                continue
            # Reject numerically nonsensical calibrations before they can win
            # the held-out comparison (e.g. a focal length that escaped to a
            # degenerate local minimum).
            width = float(subset[0].width)
            height = float(subset[0].height)
            candidate_intrinsics = np.asarray(state_candidate.intrinsics, dtype=np.float64)
            if (
                candidate_intrinsics.shape[0] < 4
                or not np.isfinite(candidate_intrinsics).all()
                or candidate_intrinsics[0] <= 0.0
                or candidate_intrinsics[1] <= 0.0
                or candidate_intrinsics[0] > 10.0 * max(width, height)
                or candidate_intrinsics[1] > 10.0 * max(width, height)
                or candidate_intrinsics[2] < -width
                or candidate_intrinsics[2] > 2.0 * width
                or candidate_intrinsics[3] < -height
                or candidate_intrinsics[3] > 2.0 * height
            ):
                print(
                    f"    {cam_name}: subset {subset_index}/{len(subsets)} "
                    "produced implausible intrinsics; trying the next bounded subset",
                    flush=True,
                )
                continue
            subset_stems = {image.image_path.stem for image in subset}
            validation_frames = [
                image for image in all_eligible_frames
                if image.image_path.stem not in subset_stems
            ]
            validation_kind = "held-out"
            if not validation_frames:
                validation_frames = list(subset)
                validation_kind = "training-only"
            if len(validation_frames) > 24:
                validation_indices = np.linspace(
                    0, len(validation_frames) - 1, 24, dtype=int,
                )
                validation_frames = [validation_frames[int(index)] for index in validation_indices]
            validation_errors: list[float] = []
            validation_solved = 0
            for image in validation_frames:
                solved = _solve_known_target_pose(
                    image, known_targets3d, state_candidate.intrinsics, min_shared,
                )
                if solved is not None:
                    validation_solved += 1
                    validation_errors.append(float(solved[2]))

            train_errors: list[float] = []
            for image in subset:
                pose = state_candidate.poses.get(image.image_index)
                if pose is None:
                    continue
                correspondences = _known_target_correspondences(
                    image, known_targets3d, min_shared,
                )
                if correspondences is None:
                    continue
                _, object_points, image_points = correspondences
                train_errors.extend(
                    float(np.linalg.norm(
                        project_point(state_candidate.intrinsics, pose, point) - measured,
                    ))
                    for point, measured in zip(object_points, image_points)
                )
            validation_median = float(np.median(validation_errors)) if validation_errors else float("inf")
            train_median = float(np.median(train_errors)) if train_errors else float("inf")
            score = (validation_solved, -validation_median, -train_median)
            print(
                f"    {cam_name}: subset {subset_index}/{len(subsets)} "
                f"frames={len(subset)}, {validation_kind} PnP="
                f"{validation_solved}/{len(validation_frames)}, "
                f"{validation_kind} median={validation_median:.2f} px",
                flush=True,
            )
            if best_score is None or score > best_score:
                best_score = score
                best_state = state_candidate

        # Require both an absolute validation floor and a strong success
        # fraction.  Tiny datasets may only support training-only validation,
        # which is reported as such above and remains weaker than true holdout
        # validation, but it may not pass when most of its own poses fail.
        eligible_count = len(all_eligible_frames)
        training_count = (
            min(24, eligible_count - 8)
            if eligible_count > 24
            else min(24, eligible_count)
        )
        validation_count = (
            eligible_count - training_count
            if eligible_count > training_count
            else training_count
        )
        validation_count = min(24, validation_count)
        required_validation = max(8, math.ceil(0.8 * validation_count))
        if best_state is None or best_score is None or best_score[0] < required_validation:
            raise RuntimeError(f"{cam_name}: bounded known-target bootstrap produced no usable subset.")
        state = best_state
        intrinsics_by_camera[cam_name] = state.intrinsics.copy()
        poses: Dict[str, np.ndarray] = {}
        failures = 0
        for frame_number, image in enumerate(cam_detections, start=1):
            solved = _solve_known_target_pose(
                image, known_targets3d, state.intrinsics, min_shared
            )
            if solved is None:
                failures += 1
                continue
            pose, inlier_count, median_error = solved
            poses[image.image_path.stem] = pose
            if frame_number % 25 == 0 or frame_number == len(cam_detections):
                print(
                    f"    {cam_name}: fixed-K PnP {frame_number}/{len(cam_detections)} "
                    f"frames, solved={len(poses)}, failed={failures}",
                    flush=True,
                )
        if not poses:
            raise RuntimeError(
                f"{cam_name}: fixed-intrinsic PnP failed for every retained frame "
                "after known-target bootstrap."
            )
        poses_by_camera[cam_name] = poses
        print(
            f"  {cam_name}: bootstrapped {len(poses_by_camera[cam_name])} poses from known 3D targets, "
            f"fx={state.intrinsics[0]:.1f} fy={state.intrinsics[1]:.1f} "
            f"cx={state.intrinsics[2]:.1f} cy={state.intrinsics[3]:.1f}",
            flush=True,
        )

    return intrinsics_by_camera, poses_by_camera


# ---------------------------------------------------------------------------
# Step 4 — Multi-camera bundle adjustment
# ---------------------------------------------------------------------------

class MultiCamReprojectionCost(pyceres.CostFunction):
    """Reprojection residual for a single observation.

    Parameter blocks: [intrinsics(8), rig_pose(6), relative_rvec(3), relative_tvec(3), point(3)]
    The absolute camera pose is: compose(rig_pose, relative_rvec, relative_tvec).
    """

    def __init__(self, observation: np.ndarray):
        super().__init__()
        self.observation = observation.astype(np.float64)
        self.set_num_residuals(2)
        self.set_parameter_block_sizes([8, 6, 3, 3, 3])

    def Evaluate(self, parameters, residuals, jacobians):
        intrinsics = np.array(parameters[0], dtype=np.float64, copy=True)
        rig_pose = np.array(parameters[1], dtype=np.float64, copy=True)
        rel_rvec = np.array(parameters[2], dtype=np.float64, copy=True)
        rel_tvec = np.array(parameters[3], dtype=np.float64, copy=True)
        point = np.array(parameters[4], dtype=np.float64, copy=True)

        abs_pose = compose_pose_from_parts(rig_pose, rel_rvec, rel_tvec)
        prediction = project_point(intrinsics, abs_pose, point)
        residual_vec = prediction - self.observation
        residuals[0] = float(residual_vec[0])
        residuals[1] = float(residual_vec[1])

        if jacobians is not None:
            param_blocks = [intrinsics, rig_pose, rel_rvec, rel_tvec, point]
            for block_idx, block in enumerate(param_blocks):
                if jacobians[block_idx] is None:
                    continue
                jac = np.zeros((2, block.shape[0]), dtype=np.float64)
                for col in range(block.shape[0]):
                    step = 1e-6 * max(1.0, abs(block[col]))
                    plus_blocks = [b.copy() for b in param_blocks]
                    minus_blocks = [b.copy() for b in param_blocks]
                    plus_blocks[block_idx][col] += step
                    minus_blocks[block_idx][col] -= step

                    abs_plus = compose_pose_from_parts(plus_blocks[1], plus_blocks[2], plus_blocks[3])
                    abs_minus = compose_pose_from_parts(minus_blocks[1], minus_blocks[2], minus_blocks[3])
                    res_plus = project_point(plus_blocks[0], abs_plus, plus_blocks[4]) - self.observation
                    res_minus = project_point(minus_blocks[0], abs_minus, minus_blocks[4]) - self.observation
                    jac[:, col] = (res_plus - res_minus) / (2.0 * step)

                for row in range(2):
                    for col in range(block.shape[0]):
                        jacobians[block_idx][row * block.shape[0] + col] = jac[row, col]
        return True


def multi_cam_collect_observations(
    detections_by_camera: Dict[str, List[ImageDetections]],
    state: MultiCameraState,
    max_reprojection_error: float | None,
    unobservable: List[Tuple[str, str, int, np.ndarray]] | None = None,
) -> List[Tuple[str, str, int, np.ndarray]]:
    """Collect valid observations across all cameras.

    Returns list of (cam_name, frame_stem, target_id, 2D_point).  A detection
    whose target lies behind the camera, or outside its field of view (10%
    beyond the undistorted image corners), cannot be that target: it is a
    false identification and is skipped (and appended to ``unobservable``).
    """
    observations: List[Tuple[str, str, int, np.ndarray]] = []
    for cam_name, dets in detections_by_camera.items():
        intrinsics = state.camera_intrinsics[cam_name]
        rel_pose = state.relative_poses[cam_name]
        field_radius: float | None = None
        for det in dets:
            stem = det.image_path.stem
            rig_pose = state.rig_poses.get(stem)
            if rig_pose is None:
                continue
            abs_pose = compose_poses(rig_pose, rel_pose)
            rotation = rotation_matrix_from_pose(abs_pose)
            if field_radius is None and det.width > 0 and det.height > 0:
                # Pinhole radius of the image corners; a radius test rather than
                # a pixel test, because the polynomial distortion model can fold
                # far off-axis directions back into the image.
                corners = np.array([[0.0, 0.0], [det.width - 1.0, 0.0], [0.0, det.height - 1.0],
                                    [det.width - 1.0, det.height - 1.0]]).reshape(-1, 1, 2)
                undistorted = cv2.undistortPoints(corners, intrinsics_matrix(intrinsics),
                                                  distortion_vector(intrinsics)).reshape(-1, 2)
                field_radius = 1.1 * float(np.max(np.linalg.norm(undistorted, axis=1)))
            for target_id, pt2d in det.detections.items():
                pt3d = state.points.get(target_id)
                if pt3d is None:
                    continue
                camera_point = rotation @ pt3d + abs_pose[3:]
                if camera_point[2] <= 1e-9 or (
                    field_radius is not None
                    and np.hypot(camera_point[0], camera_point[1]) > field_radius * camera_point[2]
                ):
                    if unobservable is not None:
                        unobservable.append((cam_name, stem, target_id, pt2d))
                    continue
                residual = project_point(intrinsics, abs_pose, pt3d) - pt2d
                if not np.all(np.isfinite(residual)):
                    continue
                if max_reprojection_error is None or np.linalg.norm(residual) <= max_reprojection_error:
                    observations.append((cam_name, stem, target_id, pt2d))
    return observations


def multi_cam_compute_object_space_residual(
    state: MultiCameraState,
    observation: Tuple[str, str, int, np.ndarray],
) -> float | None:
    """Return the point-to-ray distance in metres for one observation."""
    cam_name, stem, target_id, pt2d = observation
    intrinsics = state.camera_intrinsics[cam_name]
    rig_pose = state.rig_poses.get(stem)
    pt3d = state.points.get(target_id)
    if rig_pose is None or pt3d is None:
        return None

    abs_pose = compose_poses(rig_pose, state.relative_poses[cam_name])
    rotation = rotation_matrix_from_pose(abs_pose)
    camera_center = camera_center_from_pose(abs_pose)

    undistorted = cv2.undistortPoints(
        np.asarray(pt2d, dtype=np.float64).reshape(1, 1, 2),
        intrinsics_matrix(intrinsics),
        distortion_vector(intrinsics),
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

    offset = pt3d - camera_center
    depth_along_ray = float(np.dot(offset, ray_world))
    if not np.isfinite(depth_along_ray) or depth_along_ray <= 1e-12:
        return None

    residual = float(np.linalg.norm(offset - depth_along_ray * ray_world))
    if not np.isfinite(residual):
        return None
    return residual


def multi_cam_filter_inlier_observations(
    state: MultiCameraState,
    observations: Sequence[Tuple[str, str, int, np.ndarray]],
    mad_scale: float,
) -> Tuple[List[Tuple[str, str, int, np.ndarray]], MetricInlierFilterStats]:
    """Keep observations within a MAD threshold on object-space residuals."""
    residual_pairs: List[Tuple[Tuple[str, str, int, np.ndarray], float]] = []
    invalid_count = 0
    for observation in observations:
        residual = multi_cam_compute_object_space_residual(state, observation)
        if residual is None:
            invalid_count += 1
            continue
        residual_pairs.append((observation, residual))

    if not residual_pairs:
        return [], MetricInlierFilterStats(
            total_observations=len(observations),
            kept_observations=0,
            excluded_observations=len(observations),
            invalid_observations=invalid_count,
            median_residual=0.0,
            mad=0.0,
            threshold=0.0,
            mad_scale=mad_scale,
        )

    residuals = np.array([residual for _, residual in residual_pairs], dtype=np.float64)
    median = float(np.median(residuals))
    deviations = np.abs(residuals - median)
    mad = float(np.median(deviations))
    spread_source = "mad"
    if mad <= 1e-12:
        std_fallback = float(np.std(residuals))
        if std_fallback > 0.0:
            mad = std_fallback
            spread_source = "std"

    threshold = median + mad_scale * mad
    inliers = [observation for observation, residual in residual_pairs if residual <= threshold]
    excluded_count = len(observations) - len(inliers)
    return inliers, MetricInlierFilterStats(
        total_observations=len(observations),
        kept_observations=len(inliers),
        excluded_observations=excluded_count,
        invalid_observations=invalid_count,
        median_residual=median,
        mad=mad,
        threshold=threshold,
        mad_scale=mad_scale,
        spread_source=spread_source,
    )


def multi_cam_compute_reprojection_errors(
    state: MultiCameraState,
    observations: Sequence[Tuple[str, str, int, np.ndarray]],
) -> np.ndarray:
    errors: List[float] = []
    for cam_name, stem, target_id, pt2d in observations:
        intrinsics = state.camera_intrinsics[cam_name]
        rig_pose = state.rig_poses.get(stem)
        pt3d = state.points.get(target_id)
        if rig_pose is None or pt3d is None:
            continue
        abs_pose = compose_poses(rig_pose, state.relative_poses[cam_name])
        residual = project_point(intrinsics, abs_pose, pt3d) - pt2d
        if np.all(np.isfinite(residual)):
            errors.append(float(np.linalg.norm(residual)))
    return np.array(errors, dtype=np.float64)


def multi_cam_compute_object_space_errors(
    state: MultiCameraState,
    observations: Sequence[Tuple[str, str, int, np.ndarray]],
) -> np.ndarray:
    errors: List[float] = []
    for observation in observations:
        residual = multi_cam_compute_object_space_residual(state, observation)
        if residual is not None:
            errors.append(residual)
    return np.array(errors, dtype=np.float64)


def _linearised_adjustment_diagnostics(
    problem: pyceres.Problem,
    state: MultiCameraState,
    observations: Sequence[Tuple[str, str, int, np.ndarray]],
    used_rig_poses: Set[str],
    used_points: Set[int],
    fixed_point_ids: Set[int],
    initial_camera_intrinsics: Dict[str, np.ndarray] | None,
    initial_relative_poses: Dict[str, np.ndarray] | None,
    observation_sigma_px: float | None,
) -> AdjustmentDiagnostics:
    """Extract marginal covariance and residual reliability from Ceres.

    The covariance is evaluated on the un-robustified final least-squares
    problem.  Ceres marginalises all unrequested frame/point blocks, so the
    reported camera covariance includes their uncertainty.
    """
    from scipy import sparse
    from scipy.sparse.linalg import lsmr
    from scipy.stats import chi2

    warnings: List[str] = []
    intr_names = ["fx", "fy", "cx", "cy", "k1", "k2", "p1", "p2"]
    pose_names = ["rx", "ry", "rz", "tx", "ty", "tz"]

    # Keep blocks exactly as they were passed to Ceres: array identity matters.
    global_specs: List[Tuple[List[str], np.ndarray, np.ndarray, List[str]]] = []

    def _status_for(block: np.ndarray, index: int, base: str) -> str:
        if base.startswith("fixed"):
            return base
        if problem.has_manifold(block):
            base = f"{base} (manifold-constrained)"
        try:
            lower = float(problem.get_parameter_lower_bound(block, index))
            upper = float(problem.get_parameter_upper_bound(block, index))
            value = float(block[index])
            tolerance = 1e-7 * max(1.0, abs(value), abs(lower) if np.isfinite(lower) else 0.0, abs(upper) if np.isfinite(upper) else 0.0)
            if np.isfinite(lower) and abs(value - lower) <= tolerance:
                base += " [active lower bound]"
            elif np.isfinite(upper) and abs(value - upper) <= tolerance:
                base += " [active upper bound]"
        except (TypeError, ValueError, RuntimeError):
            pass
        return base
    for cam_name in state.camera_names:
        block = state.camera_intrinsics[cam_name]
        status = "fixed" if problem.is_parameter_block_constant(block) else "estimated"
        initial = (initial_camera_intrinsics or {}).get(cam_name, block).copy()
        global_specs.append((
            [f"{cam_name}.{name}" for name in intr_names], block, initial,
            [_status_for(block, index, status) for index in range(block.size)],
        ))

    ref_cam = state.camera_names[0]
    for cam_name in state.camera_names:
        rp = state.relative_poses[cam_name]
        initial_rp = (initial_relative_poses or {}).get(cam_name, rp).copy()
        for suffix, block, initial, names in (
            ("rotation", rp[:3], initial_rp[:3], pose_names[:3]),
            ("translation", rp[3:], initial_rp[3:], pose_names[3:]),
        ):
            if cam_name == ref_cam:
                label = "fixed (rig reference)"
            elif problem.is_parameter_block_constant(block):
                label = "fixed/constrained"
            elif problem.has_manifold(block):
                label = "estimated (manifold-constrained)"
            else:
                label = "estimated"
            global_specs.append((
                [f"{cam_name}.{name}" for name in names], block, initial,
                [_status_for(block, index, label) for index in range(block.size)],
            ))

    names = [name for spec in global_specs for name in spec[0]]
    initial = np.concatenate([spec[2] for spec in global_specs]).astype(np.float64)
    final = np.concatenate([spec[1] for spec in global_specs]).astype(np.float64)
    statuses = [status for spec in global_specs for status in spec[3]]
    n_global = len(names)
    covariance = np.full((n_global, n_global), np.nan, dtype=np.float64)

    # Active parameter blocks in an explicit order for the adjustment Jacobian.
    active_blocks: List[np.ndarray] = []
    for cam_name in state.camera_names:
        block = state.camera_intrinsics[cam_name]
        if problem.has_parameter_block(block) and not problem.is_parameter_block_constant(block):
            active_blocks.append(block)
    for stem in sorted(used_rig_poses):
        block = state.rig_poses[stem]
        if not problem.is_parameter_block_constant(block):
            active_blocks.append(block)
    for cam_name in state.camera_names:
        rp = state.relative_poses[cam_name]
        for block in (rp[:3], rp[3:]):
            if problem.has_parameter_block(block) and not problem.is_parameter_block_constant(block):
                active_blocks.append(block)
    for target_id in sorted(used_points):
        block = state.points[target_id]
        if target_id not in fixed_point_ids and not problem.is_parameter_block_constant(block):
            active_blocks.append(block)

    eval_options = pyceres.EvaluateOptions()
    eval_options.apply_loss_function = False
    eval_options.set_parameter_blocks(active_blocks)
    jac_crs = problem.evaluate_jacobian(eval_options)
    row_index, col_index, values = jac_crs.to_tuple()
    jacobian = sparse.coo_matrix(
        (values, (row_index, col_index)),
        shape=(jac_crs.num_rows, jac_crs.num_cols),
        dtype=np.float64,
    ).tocsr()
    residual = np.asarray(problem.evaluate_residuals(), dtype=np.float64)
    # The evaluated Jacobian is in Ceres tangent coordinates (important for
    # manifold-constrained blocks).  Estimate numerical rank on column-scaled
    # columns.  Dense SVD is exact for small reports; larger reports use a
    # conservative sparse small-spectrum estimate and are explicitly marked
    # unverified rather than claiming m-n determinability.
    column_norms = np.sqrt(np.asarray(jacobian.power(2).sum(axis=0)).reshape(-1))
    column_scale = np.divide(
        1.0, column_norms, out=np.ones_like(column_norms), where=column_norms > 1e-12,
    )
    scaled_jacobian = jacobian @ sparse.diags(column_scale)
    rank_verified = False
    rank_tolerance: float | None = None
    singular_values: np.ndarray | None = None
    weakest_singular_values: np.ndarray | None = None
    numerical_rank: int | None = None
    exact_left_vectors: np.ndarray | None = None
    if scaled_jacobian.shape[1] == 0:
        numerical_rank = 0
        rank_verified = True
        rank_tolerance = 0.0
    elif scaled_jacobian.shape[1] <= 600:
        dense_jacobian = scaled_jacobian.toarray()
        exact_left_vectors, singular_values, _ = np.linalg.svd(dense_jacobian, full_matrices=False)
        rank_tolerance = float(max(scaled_jacobian.shape) * np.finfo(float).eps * max(float(singular_values[0]), 1.0))
        numerical_rank = int(np.count_nonzero(singular_values > rank_tolerance))
        weakest_singular_values = singular_values[-min(12, len(singular_values)):].copy()
        rank_verified = True
    else:
        from scipy.sparse.linalg import svds
        try:
            k = min(24, scaled_jacobian.shape[1] - 1, scaled_jacobian.shape[0] - 1)
            small = np.sort(np.abs(svds(scaled_jacobian, k=k, which="SM", return_singular_vectors=False, maxiter=4000))) if k > 0 else np.empty(0)
            large = np.abs(svds(scaled_jacobian, k=1, which="LM", return_singular_vectors=False, maxiter=4000))
            rank_tolerance = float(max(scaled_jacobian.shape) * np.finfo(float).eps * max(float(large[-1]), 1.0))
            nullity_estimate = int(np.count_nonzero(small <= rank_tolerance))
            numerical_rank = int(min(scaled_jacobian.shape[0], scaled_jacobian.shape[1] - nullity_estimate))
            weakest_singular_values = small
            warnings.append(
                "Numerical rank is estimated from sparse extreme singular values; full rank is unverified for this problem size."
            )
        except Exception as exc:
            warnings.append(f"Numerical rank could not be estimated ({exc}); DOF is unavailable.")
    if numerical_rank is None:
        dof = 0
    else:
        dof = int(jacobian.shape[0] - numerical_rank)
    if dof <= 0:
        warnings.append("Non-positive degrees of freedom after rank analysis; variance-factor statistics are unavailable.")
    rss = float(residual @ residual)
    variance_factor = float(rss / dof) if dof > 0 else float("nan")
    sigma0 = float(np.sqrt(max(variance_factor, 0.0))) if np.isfinite(variance_factor) else float("nan")

    chi_stat: float | None = None
    chi_p: float | None = None
    chi_consistent: bool | None = None
    if observation_sigma_px is not None and observation_sigma_px > 0 and dof > 0:
        chi_stat = rss / float(observation_sigma_px ** 2)
        lower = float(chi2.ppf(0.025, dof))
        upper = float(chi2.ppf(0.975, dof))
        chi_consistent = bool(lower <= chi_stat <= upper)
        chi_p = float(2.0 * min(chi2.cdf(chi_stat, dof), chi2.sf(chi_stat, dof)))
    elif observation_sigma_px is None:
        warnings.append(
            "No a-priori image-coordinate sigma was supplied; the residual variance ratio is unavailable."
        )
        if not rank_verified:
            warnings.append(
                "Residual variance diagnostics use degrees of freedom from the sparse numerical-rank estimate."
            )
    elif numerical_rank is None:
        warnings.append(
            "Residual variance diagnostics are unavailable because numerical rank could not be estimated."
        )

    # Marginal covariance of global calibration blocks.
    variable_specs: List[Tuple[int, np.ndarray]] = []
    offset = 0
    for spec in global_specs:
        block = spec[1]
        if problem.has_parameter_block(block) and not problem.is_parameter_block_constant(block):
            variable_specs.append((offset, block))
        offset += block.size

    covariance_method = "Ceres SPARSE_QR marginal covariance"
    rank_deficient = numerical_rank is not None and numerical_rank < jacobian.shape[1]
    if rank_deficient:
        covariance_method = "unavailable (rank-deficient tangent Jacobian)"
        warnings.append(
            "Marginal covariance was suppressed because the tangent Jacobian has unresolved null directions."
        )
    elif variable_specs:
        pairs = [(a[1], b[1]) for i, a in enumerate(variable_specs) for b in variable_specs[i:]]
        options = pyceres.CovarianceOptions()
        options.algorithm_type = pyceres.CovarianceAlgorithmType.SPARSE_QR
        options.apply_loss_function = False
        options.num_threads = -1
        cov_solver = pyceres.Covariance(options)
        ok = bool(cov_solver.compute(pairs, problem))
        if not ok:
            warnings.append("Ceres SPARSE_QR covariance failed; retrying with dense SVD.")
            covariance_method = "Ceres DENSE_SVD marginal covariance"
            options.algorithm_type = pyceres.CovarianceAlgorithmType.DENSE_SVD
            cov_solver = pyceres.Covariance(options)
            ok = bool(cov_solver.compute(pairs, problem))
        if ok:
            for i, (off_i, block_i) in enumerate(variable_specs):
                ni = block_i.size
                for off_j, block_j in variable_specs[i:]:
                    nj = block_j.size
                    cov_block = np.asarray(
                        cov_solver.get_covariance_block(block_i, block_j),
                        dtype=np.float64,
                    ).reshape(ni, nj)
                    cov_block *= variance_factor
                    covariance[off_i:off_i + ni, off_j:off_j + nj] = cov_block
                    covariance[off_j:off_j + nj, off_i:off_i + ni] = cov_block.T
        else:
            covariance_method = "unavailable"
            warnings.append("Marginal covariance could not be computed; parameter uncertainties are unavailable.")

    correlation = np.full_like(covariance, np.nan)
    estimated = np.flatnonzero(np.isfinite(np.diag(covariance)) & (np.diag(covariance) >= 0))
    if estimated.size:
        std = np.sqrt(np.maximum(np.diag(covariance)[estimated], 0.0))
        denom = np.outer(std, std)
        sub = covariance[np.ix_(estimated, estimated)]
        corr_sub = np.divide(sub, denom, out=np.zeros_like(sub), where=denom > 0)
        np.fill_diagonal(corr_sub, 1.0)
        correlation[np.ix_(estimated, estimated)] = corr_sub
        try:
            condition_number = float(np.linalg.cond(corr_sub))
        except np.linalg.LinAlgError:
            condition_number = float("inf")
        warnings.append(
            "Correlation condition number describes parameter coupling only; it is not an absolute observability certificate."
        )
    else:
        condition_number = float("nan")

    # Approximate the diagonal of the hat matrix with deterministic Hutchinson
    # probes. This accounts for all active nuisance parameters without forming
    # the full inverse normal matrix.
    n_residuals = jacobian.shape[0]
    leverage = np.full(n_residuals, np.nan, dtype=np.float64)
    leverage_reliable = False
    if exact_left_vectors is not None and numerical_rank is not None:
        leverage = np.sum(exact_left_vectors[:, :numerical_rank] ** 2, axis=1)
        leverage_reliable = bool(np.all(np.isfinite(leverage)))
    elif n_residuals and jacobian.shape[1]:
        rng = np.random.default_rng(20260826)
        estimate = np.zeros(n_residuals, dtype=np.float64)
        n_probes = 24
        probe_failures = 0
        for _ in range(n_probes):
            z = rng.choice(np.array([-1.0, 1.0]), size=n_residuals)
            solve_result = lsmr(
                scaled_jacobian, z, atol=1e-8, btol=1e-8,
                maxiter=max(200, min(2500, 3 * jacobian.shape[1])),
            )
            solution, istop, normr = solve_result[0], int(solve_result[1]), float(solve_result[3])
            if istop not in (1, 2) or not np.isfinite(normr):
                probe_failures += 1
                continue
            estimate += z * np.asarray(scaled_jacobian @ solution).reshape(-1)
        if probe_failures == 0:
            leverage = estimate / n_probes
            leverage_reliable = bool(np.all(np.isfinite(leverage)))
        else:
            warnings.append(f"Local redundancy was suppressed: {probe_failures}/{n_probes} projection probes did not converge.")
    if leverage_reliable:
        leverage = np.minimum(np.maximum(leverage, 0.0), 1.0)
        if abs(float(np.sum(leverage)) - float(numerical_rank or 0)) > max(1.0, 0.05 * max(float(numerical_rank or 0), 1.0)):
            leverage_reliable = False
            warnings.append("Local redundancy was suppressed: estimated hat-matrix trace disagreed with numerical rank.")
    if not leverage_reliable:
        leverage[:] = np.nan
    redundancy = 1.0 - leverage
    scale = sigma0 * np.sqrt(np.maximum(redundancy, 0.0))
    standardized = np.divide(
        residual, scale,
        out=np.full_like(residual, np.nan),
        where=np.isfinite(scale) & (scale > 0),
    )
    if residual.size != 2 * len(observations):
        warnings.append(
            "Residual ordering did not match the observation list; local reliability values were not attached."
        )
        leverage_uv = np.zeros((0, 2), dtype=np.float64)
        redundancy_uv = np.zeros((0, 2), dtype=np.float64)
        standardized_uv = np.zeros((0, 2), dtype=np.float64)
    else:
        leverage_uv = leverage.reshape(-1, 2)
        redundancy_uv = redundancy.reshape(-1, 2)
        standardized_uv = standardized.reshape(-1, 2)

    return AdjustmentDiagnostics(
        parameter_names=names,
        parameter_initial=initial,
        parameter_final=final,
        parameter_status=statuses,
        covariance=covariance,
        correlation=correlation,
        sigma0_px=sigma0,
        dof=dof,
        variance_factor=variance_factor,
        observation_sigma_px=observation_sigma_px,
        chi_square_statistic=chi_stat,
        chi_square_p_value=chi_p,
        chi_square_consistent_95=chi_consistent,
        condition_number=condition_number,
        leverage_uv=leverage_uv,
        local_redundancy_uv=redundancy_uv,
        standardized_residuals_uv=standardized_uv,
        covariance_method=covariance_method,
        warnings=warnings,
        jacobian_rows=int(jacobian.shape[0]),
        jacobian_columns=int(jacobian.shape[1]),
        numerical_rank=numerical_rank,
        rank_tolerance=rank_tolerance,
        rank_verified=rank_verified,
        nullity=(int(jacobian.shape[1] - numerical_rank) if numerical_rank is not None else None),
        singular_values=singular_values,
        weakest_singular_values=weakest_singular_values,
    )


def solve_multi_cam_bundle_adjustment(
    state: MultiCameraState,
    observations: Sequence[Tuple[str, str, int, np.ndarray]],
    max_iterations: int = 500,
    huber_delta: float = 3.0,
    loss_relative_tolerance: float = 1e-5,
    loss_patience: int = 5,
    robust_loss: str = "huber",
    cauchy_scale: float = 3.0,
    fix_relative_poses: bool = True,
    fix_relative_pose_translations: bool = False,
    fix_intrinsics: bool = False,
    fixed_point_ids: Set[int] | None = None,
    known_baseline: float | None = None,
    collect_adjustment_diagnostics: bool = False,
    initial_camera_intrinsics: Dict[str, np.ndarray] | None = None,
    initial_relative_poses: Dict[str, np.ndarray] | None = None,
    observation_sigma_px: float | None = None,
) -> SolverDiagnostics:
    """Joint bundle adjustment over all cameras.

    Parameter blocks per observation:
      - camera intrinsics (8)  — shared per camera
      - rig pose (6)          — shared per frame
      - relative pose (6)     — fixed per non-reference camera
      - 3D point (3)

    The first rig frame is held constant (gauge freedom).
    Relative poses are held constant by default to enforce the rig constraint.
    A supplied scalar baseline adds an exact cam0-cam1 translation-norm
    manifold; a supplied XYZ vector fixes only that cam0-cam1 translation.
    """
    problem = pyceres.Problem()
    baseline_value, baseline_vector = parse_known_baseline_argument(known_baseline)
    constrained_baseline_camera = state.camera_names[1] if len(state.camera_names) > 1 else None
    baseline_manifold = None
    if baseline_vector is not None and constrained_baseline_camera is not None:
        state.relative_poses[constrained_baseline_camera][3:] = baseline_vector
    if robust_loss == "cauchy":
        loss = pyceres.CauchyLoss(cauchy_scale)
    elif robust_loss == "none":
        loss = pyceres.TrivialLoss()
    else:
        loss = pyceres.HuberLoss(huber_delta)

    used_rig_poses: set = set()
    used_points: set = set()

    for cam_name, stem, target_id, pt2d in observations:
        cost = MultiCamReprojectionCost(pt2d)
        rel_pose = state.relative_poses[cam_name]
        problem.add_residual_block(
            cost,
            loss,
            [
                state.camera_intrinsics[cam_name],
                state.rig_poses[stem],
                rel_pose[:3],
                rel_pose[3:],
                state.points[target_id],
            ],
        )
        used_rig_poses.add(stem)
        used_points.add(target_id)

    # Fix the first rig frame (gauge freedom) — unless known 3D
    # target points already anchor the coordinate frame.
    if not fixed_point_ids:
        first_frame = sorted(used_rig_poses)[0]
        problem.set_parameter_block_constant(state.rig_poses[first_frame])

    # Keep cam0 as the rig reference to avoid redundancy with rig_poses.
    ref_cam = state.camera_names[0]
    problem.set_parameter_block_constant(state.relative_poses[ref_cam][:3])
    problem.set_parameter_block_constant(state.relative_poses[ref_cam][3:])

    # Optionally fix the remaining relative poses (rig constraint).  A known
    # vector fixes cam0-cam1 translation components.  A scalar constrains only
    # its norm and leaves the two direction degrees of freedom adjustable.
    if fix_relative_poses:
        for cam_name in state.camera_names[1:]:
            problem.set_parameter_block_constant(state.relative_poses[cam_name][:3])
            problem.set_parameter_block_constant(state.relative_poses[cam_name][3:])
    elif baseline_vector is not None and constrained_baseline_camera is not None:
        problem.set_parameter_block_constant(state.relative_poses[constrained_baseline_camera][3:])
    elif fix_relative_pose_translations:
        for cam_name in state.camera_names[1:]:
            problem.set_parameter_block_constant(state.relative_poses[cam_name][3:])

    if (
        baseline_value is not None
        and constrained_baseline_camera is not None
        and not fix_relative_poses
    ):
        translation = state.relative_poses[constrained_baseline_camera][3:]
        norm = float(np.linalg.norm(translation))
        if norm <= 1e-12:
            raise ValueError("cam0-cam1 translation is zero; cannot apply a scalar baseline constraint")
        translation[:] *= float(baseline_value) / norm
        # Keep this object alive through solve/evaluation; Ceres stores the
        # manifold pointer but not ownership in all Python bindings.
        baseline_manifold = pyceres.SphereManifold(3)
        problem.set_manifold(translation, baseline_manifold)

    # A single camera without control points or baseline keeps a free scale
    # after the first pose is fixed (similarity about its centre).  Remove it
    # with one constraint: hold the coordinate of the object point that is
    # farthest from that centre along one axis.
    if not fixed_point_ids and baseline_value is None and baseline_vector is None             and len(state.camera_names) == 1 and used_points:
        centre = camera_center_from_pose(state.rig_poses[first_frame])
        offsets = {tid: state.points[tid] - centre for tid in used_points}
        gauge_id = max(offsets, key=lambda tid: float(np.max(np.abs(offsets[tid]))))
        axis = int(np.argmax(np.abs(offsets[gauge_id])))
        scale_gauge_manifold = pyceres.SubsetManifold(3, [axis])
        problem.set_manifold(state.points[gauge_id], scale_gauge_manifold)

    # Fix known 3D points (metric ground truth)
    if fixed_point_ids:
        for tid in fixed_point_ids:
            if tid in used_points:
                problem.set_parameter_block_constant(state.points[tid])

    # Intrinsics bounds (for each camera)
    for cam_name in state.camera_names:
        intr = state.camera_intrinsics[cam_name]
        # We need at least one observation for this camera in the problem
        if not any(cn == cam_name for cn, _, _, _ in observations):
            continue

        if fix_intrinsics:
            problem.set_parameter_block_constant(intr)
            continue

        # Determine image size from first observation of this camera
        # Use a reasonable max_dimension from existing intrinsics
        max_dim = max(intr[0], intr[1]) * 2.0  # rough estimate
        cx_est = intr[2]
        cy_est = intr[3]
        w_est = cx_est * 2.0
        h_est = cy_est * 2.0

        problem.set_parameter_lower_bound(intr, 0, 0.2 * max_dim)
        problem.set_parameter_lower_bound(intr, 1, 0.2 * max_dim)
        problem.set_parameter_upper_bound(intr, 0, 3.0 * max_dim)
        problem.set_parameter_upper_bound(intr, 1, 3.0 * max_dim)
        problem.set_parameter_lower_bound(intr, 2, 0.25 * w_est)
        problem.set_parameter_upper_bound(intr, 2, 0.75 * w_est)
        problem.set_parameter_lower_bound(intr, 3, 0.25 * h_est)
        problem.set_parameter_upper_bound(intr, 3, 0.75 * h_est)
        for di in range(4, 8):
            problem.set_parameter_lower_bound(intr, di, -1.0)
            problem.set_parameter_upper_bound(intr, di, 1.0)

    options = pyceres.SolverOptions()
    options.linear_solver_type = pyceres.LinearSolverType.DENSE_SCHUR
    options.max_num_iterations = max_iterations
    options.minimizer_progress_to_stdout = True
    options.num_threads = -1
    options.update_state_every_iteration = True
    options.function_tolerance = loss_relative_tolerance

    callback = LossConvergenceCallback(loss_relative_tolerance, loss_patience)
    options.callbacks = [callback]

    summary = pyceres.SolverSummary()
    pyceres.solve(options, problem, summary)
    cost_history = callback.cost_history or [float(summary.initial_cost), float(summary.final_cost)]
    diagnostics = SolverDiagnostics(
        summary=summary.BriefReport(),
        iterations=int(summary.num_successful_steps + summary.num_unsuccessful_steps),
        initial_cost=float(summary.initial_cost),
        final_cost=float(summary.final_cost),
        cost_history=[float(c) for c in cost_history],
    )
    if collect_adjustment_diagnostics:
        diagnostics.adjustment = _linearised_adjustment_diagnostics(
            problem=problem,
            state=state,
            observations=observations,
            used_rig_poses=set(used_rig_poses),
            used_points=set(used_points),
            fixed_point_ids=set(fixed_point_ids or set()),
            initial_camera_intrinsics=initial_camera_intrinsics,
            initial_relative_poses=initial_relative_poses,
            observation_sigma_px=observation_sigma_px,
        )
    return diagnostics


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def save_multi_cam_colmap_output(
    state: MultiCameraState,
    detections_by_camera: Dict[str, List[ImageDetections]],
    output_dir: Path,
) -> Path:
    """Save joint calibration results in COLMAP text format."""
    colmap_dir = output_dir / "colmap"
    colmap_dir.mkdir(parents=True, exist_ok=True)

    # cameras.txt — one camera per sensor
    cam_lines = [
        "# Camera list with one line of data per camera:",
        "#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[fx,fy,cx,cy,k1,k2,p1,p2]",
    ]
    cam_id_map: Dict[str, int] = {}
    for idx, cam_name in enumerate(state.camera_names):
        cam_id = idx + 1
        cam_id_map[cam_name] = cam_id
        intr = state.camera_intrinsics[cam_name]
        # Get image size from detections
        w, h = 0, 0
        if detections_by_camera.get(cam_name):
            w = detections_by_camera[cam_name][0].width
            h = detections_by_camera[cam_name][0].height
        cam_lines.append(
            f"{cam_id} OPENCV {w} {h} "
            f"{intr[0]:.9f} {intr[1]:.9f} {intr[2]:.9f} {intr[3]:.9f} "
            f"{intr[4]:.9f} {intr[5]:.9f} {intr[6]:.9f} {intr[7]:.9f}"
        )
    (colmap_dir / "cameras.txt").write_text("\n".join(cam_lines) + "\n", encoding="utf-8")

    # images.txt — all registered images
    img_lines = [
        "# Image list with two lines of data per image:",
        "#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME",
        "#   POINTS2D[] as (X, Y, POINT3D_ID)",
    ]
    # Build point3D ID mapping
    track_ids = sorted(state.points.keys())
    track_to_colmap_id = {tid: i + 1 for i, tid in enumerate(track_ids)}

    colmap_img_id = 1
    for stem in sorted(state.rig_poses.keys()):
        for cam_name in state.camera_names:
            abs_pose = compose_poses(state.rig_poses[stem], state.relative_poses[cam_name])
            R = rotation_matrix_from_pose(abs_pose)
            t = abs_pose[3:]
            quat = rotation_matrix_to_quaternion(R)
            qw, qx, qy, qz = float(quat[3]), float(quat[0]), float(quat[1]), float(quat[2])

            img_name = next(
                (f"{cam_name}/{det.image_path.name}" for det in detections_by_camera.get(cam_name, [])
                 if det.image_path.stem == stem),
                f"{cam_name}/{stem}.jpg",
            )
            img_lines.append(
                f"{colmap_img_id} {qw:.9f} {qx:.9f} {qy:.9f} {qz:.9f} "
                f"{t[0]:.9f} {t[1]:.9f} {t[2]:.9f} {cam_id_map[cam_name]} {img_name}"
            )

            # 2D points line
            pts2d_parts: List[str] = []
            # Find detections for this camera+stem
            for det in detections_by_camera.get(cam_name, []):
                if det.image_path.stem == stem:
                    for target_id, pt2d in sorted(det.detections.items()):
                        colmap_pt_id = track_to_colmap_id.get(target_id, -1)
                        pts2d_parts.append(f"{pt2d[0]:.4f} {pt2d[1]:.4f} {colmap_pt_id}")
                    break
            img_lines.append(" ".join(pts2d_parts) if pts2d_parts else "")
            colmap_img_id += 1

    (colmap_dir / "images.txt").write_text("\n".join(img_lines) + "\n", encoding="utf-8")

    # points3D.txt
    pts_lines = [
        "# 3D point list with one line of data per point:",
        "#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)",
    ]
    for tid in track_ids:
        pt = state.points[tid]
        colmap_pt_id = track_to_colmap_id[tid]
        pts_lines.append(f"{colmap_pt_id} {pt[0]:.9f} {pt[1]:.9f} {pt[2]:.9f} 200 200 200 0.0")
    (colmap_dir / "points3D.txt").write_text("\n".join(pts_lines) + "\n", encoding="utf-8")

    return colmap_dir


def save_report(
    state: MultiCameraState,
    observations: Sequence[Tuple[str, str, int, np.ndarray]],
    detections_by_camera: Dict[str, List[ImageDetections]],
    diagnostics: SolverDiagnostics,
    output_dir: Path,
) -> Path:
    """Write a human-readable calibration report to report.txt."""
    reproj_errors = multi_cam_compute_reprojection_errors(state, observations)
    object_errors = multi_cam_compute_object_space_errors(state, observations)
    track_ids = {tid for _, _, tid, _ in observations}
    frame_stems = {stem for _, stem, _, _ in observations}

    lines: List[str] = []
    lines.append("=" * 60)
    lines.append("MULTI-CAMERA CCT CALIBRATION REPORT")
    lines.append("=" * 60)
    lines.append("")
    lines.append(f"Cameras:           {', '.join(state.camera_names)}")
    lines.append(f"Registered frames: {len(frame_stems)}")
    lines.append(f"3D targets:        {len(track_ids)}")
    lines.append(f"Observations:      {len(observations)}")
    lines.append("")

    lines.append("--- Reprojection errors ---")
    if reproj_errors.size:
        lines.append(f"  Mean: {float(np.mean(reproj_errors)):.4f} px")
        lines.append(f"  RMS:  {float(np.sqrt(np.mean(np.square(reproj_errors)))):.4f} px")
        lines.append(f"  Max:  {float(np.max(reproj_errors)):.4f} px")
        lines.append(f"  Median: {float(np.median(reproj_errors)):.4f} px")
    lines.append("")

    lines.append("--- Object-space errors ---")
    if object_errors.size:
        lines.append(f"  Mean: {float(np.mean(object_errors)):.6f} m")
        lines.append(f"  RMS:  {float(np.sqrt(np.mean(np.square(object_errors)))):.6f} m")
    lines.append("")

    lines.append("--- Solver ---")
    lines.append(f"  Iterations:   {diagnostics.iterations}")
    lines.append(f"  Initial cost: {diagnostics.initial_cost:.6f}")
    lines.append(f"  Final cost:   {diagnostics.final_cost:.6f}")
    lines.append(f"  {diagnostics.summary}")
    lines.append("")

    for cam_name in state.camera_names:
        intr = state.camera_intrinsics[cam_name]
        cam_obs = [(cn, s, t, p) for cn, s, t, p in observations if cn == cam_name]
        cam_errors = multi_cam_compute_reprojection_errors(state, cam_obs)
        cam_object_errors = multi_cam_compute_object_space_errors(state, cam_obs)

        # Image resolution from detections
        w, h = 0, 0
        if detections_by_camera.get(cam_name):
            w = detections_by_camera[cam_name][0].width
            h = detections_by_camera[cam_name][0].height

        lines.append(f"--- {cam_name} ---")
        lines.append(f"  Resolution: {w} x {h}")
        lines.append(f"  Observations: {len(cam_obs)}")
        if cam_errors.size:
            lines.append(f"  Mean error: {float(np.mean(cam_errors)):.4f} px")
            lines.append(f"  RMS error:  {float(np.sqrt(np.mean(np.square(cam_errors)))):.4f} px")
        if cam_object_errors.size:
            lines.append(f"  Mean object-space error: {float(np.mean(cam_object_errors)):.6f} m")
            lines.append(f"  RMS object-space error:  {float(np.sqrt(np.mean(np.square(cam_object_errors)))):.6f} m")
        lines.append(f"  Intrinsics (OPENCV):")
        lines.append(f"    fx = {intr[0]:.6f}")
        lines.append(f"    fy = {intr[1]:.6f}")
        lines.append(f"    cx = {intr[2]:.6f}")
        lines.append(f"    cy = {intr[3]:.6f}")
        lines.append(f"    k1 = {intr[4]:.6f}")
        lines.append(f"    k2 = {intr[5]:.6f}")
        lines.append(f"    p1 = {intr[6]:.6f}")
        lines.append(f"    p2 = {intr[7]:.6f}")

        rp = state.relative_poses[cam_name]
        R_rel = rotation_matrix_from_pose(rp)
        t_rel = rp[3:]
        lines.append(f"  Relative pose (w.r.t. {state.camera_names[0]}):")
        lines.append(f"    rvec = [{rp[0]:.6f}, {rp[1]:.6f}, {rp[2]:.6f}]")
        lines.append(f"    tvec = [{t_rel[0]:.6f}, {t_rel[1]:.6f}, {t_rel[2]:.6f}]")
        baseline = float(np.linalg.norm(t_rel))
        lines.append(f"    baseline = {baseline:.6f} m")
        lines.append("")

    report_path = output_dir / "report.txt"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path


def save_kalibr_camchain(
    state: MultiCameraState,
    detections_by_camera: Dict[str, List[ImageDetections]],
    output_dir: Path,
) -> Path:
    """Save camera chain in Kalibr camchain.yaml format.

    Kalibr format uses a 4x4 T_cn_cnm1 matrix expressing the transformation
    from camera n-1 to camera n.  Camera 0 has no such field.
    Distortion model: radtan [k1, k2, p1, p2].
    """
    camchain: dict = {}

    for idx, cam_name in enumerate(state.camera_names):
        intr = state.camera_intrinsics[cam_name]
        rp = state.relative_poses[cam_name]

        # Image resolution
        w, h = 0, 0
        if detections_by_camera.get(cam_name):
            w = detections_by_camera[cam_name][0].width
            h = detections_by_camera[cam_name][0].height

        cam_key = f"cam{idx}"
        cam_entry: dict = {
            "camera_model": "pinhole",
            "intrinsics": [float(intr[0]), float(intr[1]),
                           float(intr[2]), float(intr[3])],
            "distortion_model": "radtan",
            "distortion_coeffs": [float(intr[4]), float(intr[5]),
                                  float(intr[6]), float(intr[7])],
            "resolution": [int(w), int(h)],
            "rostopic": f"/{cam_name}/image_raw",
        }

        # T_cn_cnm1: 4x4 transformation from camera (n-1) to camera n
        if idx > 0:
            R = rotation_matrix_from_pose(rp)
            t = rp[3:]
            T = np.eye(4, dtype=np.float64)
            T[:3, :3] = R
            T[:3, 3] = t
            cam_entry["T_cn_cnm1"] = T.tolist()

        camchain[cam_key] = cam_entry

    yaml_path = output_dir / "camchain.yaml"
    yaml_path.write_text(yaml.dump(camchain, default_flow_style=None, sort_keys=False),
                         encoding="utf-8")
    return yaml_path


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def partition_independent_checkpoints(
    detections_by_camera: Dict[str, List[ImageDetections]],
    known_targets3d: Dict[int, np.ndarray],
    ratio: float,
    seed: int,
    min_detections: int,
) -> tuple[Dict[str, List[ImageDetections]], Set[int], dict]:
    """Remove selected target IDs from every calibration input container."""
    observed_ids = {
        int(target_id)
        for images in detections_by_camera.values()
        for image in images
        for target_id in image.detections
    }
    eligible = sorted(
        target_id for target_id, point in known_targets3d.items()
        if target_id in observed_ids
        and np.asarray(point).shape == (3,)
        and np.all(np.isfinite(point))
    )
    count = int(math.ceil(ratio * len(eligible)))
    if not eligible or count < 1:
        raise ValueError("no observed finite known targets are eligible for checkpoints")
    if count >= len(eligible):
        raise ValueError(
            f"checkpoint split would withhold all {len(eligible)} eligible targets; reduce the ratio"
        )
    rng = np.random.Generator(np.random.PCG64(seed))
    checkpoint_ids = {int(value) for value in rng.choice(eligible, count, replace=False)}

    checkpoint_images: Dict[str, List[ImageDetections]] = {}
    for camera, images in detections_by_camera.items():
        retained: List[ImageDetections] = []
        checks: List[ImageDetections] = []
        for source in images:
            check = copy.deepcopy(source)
            check.detections = {
                tid: point for tid, point in check.detections.items() if tid in checkpoint_ids
            }
            check.raw_detections = {
                tid: point for tid, point in check.raw_detections.items() if tid in checkpoint_ids
            }
            check.refinement_ellipses = {
                tid: value for tid, value in check.refinement_ellipses.items() if tid in checkpoint_ids
            }
            check.refinement_status = {
                tid: value for tid, value in check.refinement_status.items() if tid in checkpoint_ids
            }
            if check.detections:
                checks.append(check)

            for mapping_name in (
                "detections", "raw_detections", "refinement_ellipses", "refinement_status",
            ):
                mapping = getattr(source, mapping_name)
                for target_id in checkpoint_ids:
                    mapping.pop(target_id, None)
            if len(source.detections) >= min_detections:
                retained.append(source)
        detections_by_camera[camera] = retained
        checkpoint_images[camera] = checks

    manifest = {
        "enabled": True,
        "requested_ratio": float(ratio),
        "realized_ratio": float(count / len(eligible)),
        "seed": int(seed),
        "generator": "numpy.PCG64",
        "detection_cache_reused": False,
        "sfm_uses_cct_target_ids_or_reference_coordinates": False,
        "eligible_target_ids": eligible,
        "checkpoint_target_ids": sorted(checkpoint_ids),
        "calibration_target_ids": sorted(set(eligible) - checkpoint_ids),
    }
    return checkpoint_images, checkpoint_ids, manifest


def evaluate_independent_checkpoints(
    state: MultiCameraState,
    checkpoint_images: Dict[str, List[ImageDetections]],
    checkpoint_ids: Set[int],
    known_targets3d: Dict[int, np.ndarray],
    calibrated_frame_stems: Set[str],
) -> dict:
    """Evaluate withheld targets with the final calibration held fixed."""
    by_target: Dict[int, list[dict]] = {target_id: [] for target_id in checkpoint_ids}
    reprojection: list[dict] = []
    for camera, images in checkpoint_images.items():
        intrinsics = state.camera_intrinsics[camera]
        matrix = intrinsics_matrix(intrinsics)
        distortion = distortion_vector(intrinsics)
        for image in images:
            stem = image.image_path.stem
            if stem not in calibrated_frame_stems:
                continue
            rig_pose = state.rig_poses.get(stem)
            if rig_pose is None:
                continue
            pose = compose_poses(rig_pose, state.relative_poses[camera])
            rotation = rotation_matrix_from_pose(pose)
            center = -rotation.T @ pose[3:]
            for target_id, measured in image.detections.items():
                if target_id not in checkpoint_ids:
                    continue
                normalized = cv2.undistortPoints(
                    np.asarray(measured, dtype=np.float64).reshape(1, 1, 2), matrix, distortion,
                )[0, 0]
                direction = rotation.T @ np.array([normalized[0], normalized[1], 1.0])
                direction /= np.linalg.norm(direction)
                by_target[target_id].append({
                    "camera": camera, "frame": stem, "center": center, "direction": direction,
                    "pose": pose, "measured": np.asarray(measured, dtype=np.float64),
                })
                predicted = project_point(intrinsics, pose, known_targets3d[target_id])
                delta = predicted - measured
                reference_delta = known_targets3d[target_id] - center
                object_error = float(np.linalg.norm(
                    reference_delta - direction * float(reference_delta @ direction)
                ))
                reprojection.append({
                    "target_id": int(target_id), "camera": camera, "frame": stem,
                    "du_px": float(delta[0]), "dv_px": float(delta[1]),
                    "error_px": float(np.linalg.norm(delta)),
                    "object_error_m": object_error,
                })

    targets: list[dict] = []
    for target_id in sorted(checkpoint_ids):
        rays = by_target[target_id]
        result = {"target_id": int(target_id), "n_observations": len(rays)}
        if len(rays) < 2:
            result.update(status="failed", reason="fewer than two posed observations")
            targets.append(result)
            continue
        centers = np.asarray([item["center"] for item in rays])
        if np.max(np.linalg.norm(centers[:, None] - centers[None, :], axis=2)) <= 1e-9:
            result.update(status="failed", reason="observations share one optical center")
            targets.append(result)
            continue
        directions = np.asarray([item["direction"] for item in rays])
        cosines = np.clip(directions @ directions.T, -1.0, 1.0)
        max_angle = float(np.degrees(np.max(np.arccos(cosines))))
        identity = np.eye(3)
        normal = sum(identity - np.outer(direction, direction) for direction in directions)
        rhs = sum(
            (identity - np.outer(direction, direction)) @ center
            for direction, center in zip(directions, centers)
        )
        singular = np.linalg.svd(normal, compute_uv=False)
        condition = float(singular[0] / singular[-1]) if singular[-1] > 0 else float("inf")
        if np.linalg.matrix_rank(normal) < 3 or not np.isfinite(condition) or condition > 1e12:
            result.update(status="failed", reason="ill-conditioned ray intersection",
                          intersection_angle_deg=max_angle, condition_number=condition)
            targets.append(result)
            continue
        reconstructed = np.linalg.solve(normal, rhs)
        positive = all(
            float((rotation_matrix_from_pose(item["pose"]) @ reconstructed + item["pose"][3:])[2]) > 0
            for item in rays
        )
        if not positive:
            result.update(status="failed", reason="non-positive reconstructed depth",
                          intersection_angle_deg=max_angle, condition_number=condition)
            targets.append(result)
            continue
        error = reconstructed - known_targets3d[target_id]
        result.update(
            status="ok", reconstructed_xyz_m=reconstructed.tolist(),
            reference_xyz_m=known_targets3d[target_id].tolist(), error_xyz_m=error.tolist(),
            error_3d_m=float(np.linalg.norm(error)), intersection_angle_deg=max_angle,
            condition_number=condition, weak_geometry=bool(max_angle < 1.0),
        )
        targets.append(result)

    successful = [item for item in targets if item["status"] == "ok"]
    errors = np.asarray([item["error_xyz_m"] for item in successful], dtype=np.float64)
    magnitudes = np.asarray([item["error_3d_m"] for item in successful], dtype=np.float64)
    uv = np.asarray([[item["du_px"], item["dv_px"]] for item in reprojection], dtype=np.float64)
    summary = {
        "selected_targets": len(checkpoint_ids),
        "observed_in_posed_frames": sum(bool(by_target[target_id]) for target_id in checkpoint_ids),
        "xyz_reconstructed": len(successful),
        "xyz_failed": len(checkpoint_ids) - len(successful),
        "weak_geometry": sum(bool(item.get("weak_geometry")) for item in successful),
    }
    if errors.size:
        summary.update(
            mean_error_xyz_m=np.mean(errors, axis=0).tolist(),
            std_error_xyz_m=(np.std(errors, axis=0, ddof=1) if len(errors) > 1 else np.zeros(3)).tolist(),
            rmse_xyz_m=np.sqrt(np.mean(errors ** 2, axis=0)).tolist(),
            rmse_3d_m=float(np.sqrt(np.mean(magnitudes ** 2))),
            median_3d_m=float(np.median(magnitudes)), p95_3d_m=float(np.percentile(magnitudes, 95)),
            max_3d_m=float(np.max(magnitudes)),
        )
    if uv.size:
        norms = np.linalg.norm(uv, axis=1)
        object_errors = np.asarray(
            [item["object_error_m"] for item in reprojection], dtype=np.float64,
        )
        summary["withheld_reprojection"] = {
            "n": len(uv), "mean_du_px": float(np.mean(uv[:, 0])),
            "mean_dv_px": float(np.mean(uv[:, 1])),
            "rms_u_px": float(np.sqrt(np.mean(uv[:, 0] ** 2))),
            "rms_v_px": float(np.sqrt(np.mean(uv[:, 1] ** 2))),
            "rms_2d_px": float(np.sqrt(np.mean(norms ** 2))),
            "p95_2d_px": float(np.percentile(norms, 95)),
            "object_space_rms_m": float(np.sqrt(np.mean(object_errors ** 2))),
            "object_space_median_m": float(np.median(object_errors)),
            "object_space_p95_m": float(np.percentile(object_errors, 95)),
        }
    return {"summary": summary, "targets": targets, "reprojection_observations": reprojection}

def parse_combined_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Combined CCT calibration of one camera or a multi-camera rig: initialisation -> "
                    "CCT detection -> joint BA. A folder that directly contains images is calibrated "
                    "as a single camera."
    )
    parser.add_argument(
        "--image-root", type=Path, required=True,
        help="Root directory containing cam0/, cam1/, ... subdirectories with images.",
    )
    parser.add_argument(
        "--camera", action="append", dest="cameras",
        help="Camera subdirectory name (e.g. cam0). Repeat for each camera. "
             "If omitted, auto-detected from image-root.",
    )
    parser.add_argument(
        "--known-baseline", nargs="+", type=float, default=None,
        help="Known baseline between cam0 and cam1. Provide either one scalar magnitude or three XYZ components.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("combined_output"))
    parser.add_argument("--skip-sfm", action="store_true",
                        help="Skip the pycolmap SfM stage and bootstrap poses from known 3D targets instead.")
    parser.add_argument("--force-sfm", action="store_true",
                        help="Re-run SfM even if a cached reconstruction exists.")
    parser.add_argument("--force-detections", action="store_true",
                        help="Re-run CCT detection even if cached detections exist.")
    parser.add_argument("--targets3d", type=Path, default=None,
                        help="TSV file with known 3D target positions: target_id x y z")
    parser.add_argument(
        "--checkpoints", nargs="?", type=float, const=0.2, default=0.0,
        metavar="RATIO",
        help="Withhold a reproducible fraction of known target IDs for independent checks. "
             "Using the flag without RATIO selects 0.2; zero disables it.",
    )
    parser.add_argument(
        "--checkpoint-seed", type=int, default=42,
        help="Integer seed for reproducible checkpoint target selection (default: 42).",
    )
    parser.add_argument(
        "--observation-sigma-px", type=float, default=None,
        help="A-priori 1-sigma precision of each measured image coordinate, in pixels. "
             "Enables standardized residuals and the observed-to-assumed residual variance ratio.",
    )
    parser.add_argument("--min-detections", type=int, default=5)
    parser.add_argument("--min-shared", type=int, default=6)
    parser.add_argument("--max-reprojection-error", type=float, default=12.0)
    parser.add_argument("--max-iterations", type=int, default=500)
    parser.add_argument("--huber-delta", type=float, default=3.0)
    parser.add_argument("--loss-tolerance", type=float, default=1e-5)
    parser.add_argument("--loss-patience", type=int, default=5)
    parser.add_argument(
        "--outlier-mad-scale", type=float, default=3.0,
        help="Exclude observations after robust BA when object-space residual exceeds median + scale * MAD.",
    )
    parser.add_argument("--valid-ids-file", type=Path, default=None)
    parser.add_argument("--valid-id-max", type=int, default=None)
    parser.add_argument("--max-id-hamming-distance", type=int, default=0)
    report_options = parser.add_mutually_exclusive_group()
    report_options.add_argument(
        "--no-pdf-report", action="store_true",
        help="Skip generation of the rich PDF calibration report.",
    )
    report_options.add_argument(
        "--save-report-plots", action="store_true",
        help="Also save every PDF report plot as a 600 dpi PNG in output-dir/plots/.",
    )
    parser.add_argument(
        "--workers", type=int, default=0,
        help="Parallel image workers for detection, centre refinement and detection evaluation "
             "(0 = automatic: up to 8, limited by the CPU count; 1 = sequential). Each worker holds one "
             "image in memory.",
    )
    parser.add_argument(
        "--evaluate-detections", action="store_true", default=False,
        help="Developer diagnostic (requires --targets3d): project every reference target with "
             "the final calibration, report detection recall and suspect detections, and "
             "attribute each miss to the detector stage that rejected it. Writes "
             "detection_evaluation.json. Re-runs detection on images with misses.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_combined_args(argv)
    image_root: Path = args.image_root.resolve()
    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    baseline_value, baseline_vector = parse_known_baseline_argument(args.known_baseline)
    args.known_baseline = baseline_vector if baseline_vector is not None else baseline_value
    workers = args.workers if args.workers > 0 else min(8, os.cpu_count() or 1)

    known_targets3d = load_targets3d(args.targets3d) if args.targets3d else None
    if not np.isfinite(args.checkpoints) or args.checkpoints < 0.0 or args.checkpoints >= 1.0:
        print("ERROR: --checkpoints ratio must be finite and in [0, 1)", flush=True)
        return 1
    if args.checkpoints > 0.0 and known_targets3d is None:
        print("ERROR: --checkpoints requires --targets3d", flush=True)
        return 1
    if args.skip_sfm and known_targets3d is None:
        print("ERROR: --skip-sfm currently requires --targets3d", flush=True)
        return 1
    if args.evaluate_detections and known_targets3d is None:
        print("ERROR: --evaluate-detections requires --targets3d", flush=True)
        return 1

    # Auto-detect cameras.  A folder that directly contains images is one
    # camera: the single-camera case is a rig with only the reference camera.
    if list_images(image_root) and not args.cameras:
        camera_names = [image_root.name]
        image_root = image_root.parent
    elif args.cameras:
        camera_names = args.cameras
    else:
        camera_names = sorted(d.name for d in image_root.iterdir() if d.is_dir() and list_images(d))
    if not camera_names:
        print(f"ERROR: no images or camera folders with images found in {image_root}", flush=True)
        return 1
    print(f"cameras: {camera_names}", flush=True)
    if known_targets3d is None:
        print(
            "WARNING: no --targets3d supplied. Calibration from coded targets with known coordinates is "
            "strongly recommended; without them, initialisation relies on natural-feature SfM and the "
            "targets become free object points"
            + (", and for a single camera the scale is arbitrary (only the interior orientation is "
               "meaningful)." if len(camera_names) == 1 else "."),
            flush=True,
        )

    # Resolve valid IDs
    valid_ids: Set[int] | None = None
    if args.valid_ids_file:
        from cct_calibration.run import parse_valid_ids_file
        valid_ids = parse_valid_ids_file(args.valid_ids_file)
    elif args.valid_id_max is not None:
        valid_ids = set(range(args.valid_id_max + 1))
    elif known_targets3d is not None:
        valid_ids = set(known_targets3d.keys())
    # With reference coordinates the valid codes are known, so decoding can
    # correlate against them; otherwise the generic decoder is used.
    codebook: Set[int] | None = valid_ids if known_targets3d is not None else None
    if codebook is not None:
        print(f"codebook decoding enabled with {len(codebook)} valid target IDs", flush=True)

    sfm_intrinsics: Dict[str, np.ndarray]
    sfm_poses: Dict[str, Dict[str, np.ndarray]]

    # ── Step 1: SfM ──────────────────────────────────────────────
    print("=" * 60, flush=True)
    print("STEP 1: Structure-from-Motion (pycolmap)", flush=True)
    print("=" * 60, flush=True)
    if args.skip_sfm:
        print("  skipping SfM; known 3D targets will be used to bootstrap poses after detection", flush=True)
        sfm_intrinsics = {}
        sfm_poses = {}
    else:
        sfm_dir = output_dir / "sfm"
        sfm_dir.mkdir(parents=True, exist_ok=True)

        sfm_intrinsics, sfm_poses = run_sfm(
            image_root, camera_names, sfm_dir,
            known_baseline=args.known_baseline,
            force=args.force_sfm,
        )
        for cam_name in camera_names:
            intr = sfm_intrinsics[cam_name]
            n_poses = len(sfm_poses[cam_name])
            print(f"  {cam_name}: {n_poses} poses, "
                  f"fx={intr[0]:.1f} fy={intr[1]:.1f} cx={intr[2]:.1f} cy={intr[3]:.1f}",
                  flush=True)

    # ── Step 2: CCT Detection ────────────────────────────────────
    print("=" * 60, flush=True)
    print("STEP 2: CCT target detection", flush=True)
    print("=" * 60, flush=True)

    detections_by_camera = detect_all_cameras(
        image_root, camera_names, output_dir,
        valid_ids=valid_ids,
        max_id_hamming_distance=args.max_id_hamming_distance,
        min_detections=args.min_detections,
        force=args.force_detections,
        initial_intrinsics=sfm_intrinsics if sfm_intrinsics else None,
        codebook=codebook,
        workers=workers,
    )

    if args.checkpoints > 0.0:
        raw_cache_complete = all(
            image.raw_cache_verified
            for images in detections_by_camera.values()
            for image in images
        )
        if not raw_cache_complete:
            print(
                "  checkpoint split requires original detector coordinates; "
                "cache is legacy/incomplete, running CCT detection once to upgrade it",
                flush=True,
            )
            detections_by_camera = detect_all_cameras(
                image_root, camera_names, output_dir,
                valid_ids=valid_ids,
                max_id_hamming_distance=args.max_id_hamming_distance,
                min_detections=args.min_detections,
                force=True,
                initial_intrinsics=sfm_intrinsics if sfm_intrinsics else None,
                codebook=codebook,
                workers=workers,
            )
        else:
            print(
                "  checkpoint split: reusing cached detections and restoring original "
                "detector coordinates before calibration",
                flush=True,
            )
        # Cached active coordinates may have been refined by a previous
        # calibration that used a different target partition. Start from the
        # immutable detector coordinates, but retain the cached ellipses so
        # calibrated centre refinement stays inexpensive.
        for images in detections_by_camera.values():
            for image in images:
                image.detections = {
                    target_id: image.raw_detections[target_id].copy()
                    for target_id in image.detections
                }
                image.refinement_status.clear()

    checkpoint_images: Dict[str, List[ImageDetections]] = {}
    checkpoint_ids: Set[int] = set()
    checkpoint_manifest: dict | None = None
    calibration_targets3d = known_targets3d
    if args.checkpoints > 0.0:
        try:
            checkpoint_images, checkpoint_ids, checkpoint_manifest = partition_independent_checkpoints(
                detections_by_camera, known_targets3d, args.checkpoints,
                args.checkpoint_seed, args.min_detections,
            )
        except ValueError as exc:
            print(f"ERROR: invalid checkpoint split: {exc}", flush=True)
            return 1
        calibration_targets3d = {
            target_id: point for target_id, point in known_targets3d.items()
            if target_id not in checkpoint_ids
        }
        (output_dir / "checkpoint_split.json").write_text(
            json.dumps(checkpoint_manifest, indent=2), encoding="utf-8",
        )
        print(
            f"independent checkpoints: {len(checkpoint_ids)}/"
            f"{len(checkpoint_manifest['eligible_target_ids'])} eligible targets; "
            f"seed={args.checkpoint_seed}; IDs={sorted(checkpoint_ids)}",
            flush=True,
        )

    if args.skip_sfm:
        print("=" * 60, flush=True)
        print("STEP 3A: Bootstrap From Known 3D Targets", flush=True)
        print("=" * 60, flush=True)
        sfm_intrinsics, sfm_poses = bootstrap_from_known_targets(
            detections_by_camera,
            camera_names,
            calibration_targets3d,
            args.min_shared,
        )

    # ── Step 3: Initialize multi-camera state ────────────────────
    print("=" * 60, flush=True)
    print("STEP 3: Initialize multi-camera state", flush=True)
    print("=" * 60, flush=True)

    state = initialize_multi_camera_state(
        sfm_intrinsics, sfm_poses, detections_by_camera, camera_names,
        max_reprojection_error=args.max_reprojection_error,
        known_targets3d=calibration_targets3d,
        known_baseline=args.known_baseline,
        reuse_known_world_poses=args.skip_sfm,
    )
    initial_camera_intrinsics = {
        name: values.copy() for name, values in state.camera_intrinsics.items()
    }
    initial_relative_poses = {
        name: values.copy() for name, values in state.relative_poses.items()
    }
    fixed_point_ids: Set[int] = getattr(state, '_fixed_point_ids', set())
    if checkpoint_ids:
        initialization_ids = set(state.points) | set(fixed_point_ids)
        if checkpoint_ids & initialization_ids:
            raise RuntimeError("checkpoint leakage detected in initialized target state")
    use_known_baseline = args.known_baseline is not None
    optimize_relative_poses = bool(fixed_point_ids) or use_known_baseline
    if optimize_relative_poses:
        if use_known_baseline and not fixed_point_ids:
            print("  known baseline constraint enabled: optimizing non-reference relative poses", flush=True)
        else:
            print("  metric target constraints enabled: optimizing non-reference relative poses", flush=True)
    else:
        print("  no metric target constraints: keeping non-reference relative poses fixed", flush=True)

    # ── Step 4: Joint bundle adjustment ──────────────────────────
    print("=" * 60, flush=True)
    print("STEP 4: Joint multi-camera bundle adjustment", flush=True)
    print("=" * 60, flush=True)

    unobservable: List[Tuple[str, str, int, np.ndarray]] = []
    observations = multi_cam_collect_observations(
        detections_by_camera, state, None, unobservable=unobservable,
    )
    print(f"initial observations (ungated): {len(observations)}", flush=True)
    if unobservable:
        print(
            f"  excluded {len(unobservable)} false identification(s): target behind the camera or outside "
            "its field of view: " + ", ".join(f"{cam}/{stem}:{tid}" for cam, stem, tid, _ in unobservable),
            flush=True,
        )
    if len(observations) < 10:
        print("ERROR: not enough observations for bundle adjustment", flush=True)
        return 1

    # First BA (plain LS)
    diagnostics = solve_multi_cam_bundle_adjustment(
        state, observations,
        max_iterations=args.max_iterations,
        huber_delta=args.huber_delta,
        loss_relative_tolerance=args.loss_tolerance,
        loss_patience=args.loss_patience,
        robust_loss="none",
        fix_relative_poses=not optimize_relative_poses,
        fix_relative_pose_translations=baseline_vector is not None,
        fixed_point_ids=fixed_point_ids,
        known_baseline=args.known_baseline,
    )
    print(f"  first BA (plain LS): {diagnostics.summary}", flush=True)
    bundle_history = [("Initial least-squares", diagnostics)]

    center_intrinsics_history = [{name: values.tolist() for name, values in state.camera_intrinsics.items()}]
    center_updates = refine_detected_centers_after_calibration(
        detections_by_camera, state.camera_intrinsics,
        valid_ids=valid_ids,
        max_id_hamming_distance=args.max_id_hamming_distance,
        codebook=codebook,
        workers=workers,
    )
    center_update_history = [dict(center_updates)]
    # Checkpoint newly recovered ellipse evidence immediately. If a later BA
    # or report step is interrupted, the next run still avoids that migration
    # work and resumes from the exact verified observations.
    for cam_name, images in detections_by_camera.items():
        save_refinement_cache(images, output_dir / cam_name, merge_existing=True)
    print(
        "  calibrated centre remeasurement: "
        + ", ".join(f"{name}={count}" for name, count in center_updates.items()),
        flush=True,
    )
    observations = multi_cam_collect_observations(
        detections_by_camera, state, None,
    )
    if checkpoint_ids & {int(target_id) for _, _, target_id, _ in observations}:
        raise RuntimeError("checkpoint leakage detected in bundle observations")

    wide_observations = multi_cam_collect_observations(
        detections_by_camera, state, args.max_reprojection_error * 3.0,
    )
    print(
        f"  robust BA candidates: {len(wide_observations)} observations "
        f"(reprojection gate {args.max_reprojection_error * 3.0:.3f} px)",
        flush=True,
    )
    if len(wide_observations) < 10:
        print("ERROR: not enough observations for robust refinement", flush=True)
        return 1

    diagnostics = solve_multi_cam_bundle_adjustment(
        state, wide_observations,
        max_iterations=args.max_iterations,
        huber_delta=args.huber_delta,
        loss_relative_tolerance=args.loss_tolerance,
        loss_patience=args.loss_patience,
        robust_loss="cauchy",
        cauchy_scale=3.0,
        fix_relative_poses=not optimize_relative_poses,
        fix_relative_pose_translations=baseline_vector is not None,
        fix_intrinsics=True,
        fixed_point_ids=fixed_point_ids,
        known_baseline=args.known_baseline,
    )
    print(f"  robust BA (Cauchy, intrinsics fixed): {diagnostics.summary}", flush=True)
    bundle_history.append(("Robust Cauchy (intrinsics fixed)", diagnostics))

    prefilter_reprojection_by_camera = {
        cam_name: multi_cam_compute_reprojection_errors(
            state, [item for item in wide_observations if item[0] == cam_name],
        )
        for cam_name in camera_names
    }
    observations, filter_stats = multi_cam_filter_inlier_observations(
        state, wide_observations, args.outlier_mad_scale,
    )
    print(
        f"  metric MAD filter: threshold={filter_stats.threshold:.6f} m "
        f"(median={filter_stats.median_residual:.6f} m + "
        f"{filter_stats.mad_scale:.2f} * {filter_stats.spread_source.upper()}={filter_stats.mad:.6f} m); "
        f"excluded={filter_stats.excluded_observations}/{filter_stats.total_observations} observations",
        flush=True,
    )
    if filter_stats.invalid_observations:
        print(
            f"    invalid metric residuals excluded: {filter_stats.invalid_observations}",
            flush=True,
        )
    # A frame left with fewer observations than the initialization requires
    # has an undetermined (or weakly determined) pose; drop it entirely.
    frame_support: Dict[str, int] = {}
    for _, stem, _, _ in observations:
        frame_support[stem] = frame_support.get(stem, 0) + 1
    dropped_frames = sorted(stem for stem, n in frame_support.items() if n < args.min_shared)
    if dropped_frames:
        observations = [item for item in observations if item[1] not in set(dropped_frames)]
        print(
            f"  dropped {len(dropped_frames)} frame(s) with fewer than {args.min_shared} retained "
            f"observations: {', '.join(dropped_frames)}",
            flush=True,
        )
    if len(observations) < 10:
        print("ERROR: not enough inlier observations after robust filtering", flush=True)
        return 1

    diagnostics = solve_multi_cam_bundle_adjustment(
        state, observations,
        max_iterations=args.max_iterations,
        huber_delta=args.huber_delta,
        loss_relative_tolerance=args.loss_tolerance,
        loss_patience=args.loss_patience,
        robust_loss="none",
        fix_relative_poses=not optimize_relative_poses,
        fix_relative_pose_translations=baseline_vector is not None,
        fixed_point_ids=fixed_point_ids,
        known_baseline=args.known_baseline,
        collect_adjustment_diagnostics=True,
        initial_camera_intrinsics=initial_camera_intrinsics,
        initial_relative_poses=initial_relative_poses,
        observation_sigma_px=args.observation_sigma_px,
    )
    print(f"  final BA (plain LS): {diagnostics.summary}", flush=True)
    bundle_history.append(("Post-filter least-squares", diagnostics))

    # Refresh centres once more only when the preceding BA changed the camera
    # model enough to matter in image space. This is a bounded two-pass
    # refinement, not a verified fixed-point convergence.
    # Preserve the robust-filter
    # selection by key (camera, frame, target), then perform one final BA on
    # the refreshed coordinates.  A second pass is intentionally the endpoint
    # so the report can state exactly how many conditional remeasurements ran.
    center_intrinsics_history.append({name: values.tolist() for name, values in state.camera_intrinsics.items()})
    final_center_updates = refine_detected_centers_after_calibration(
        detections_by_camera, state.camera_intrinsics,
        valid_ids=valid_ids,
        max_id_hamming_distance=args.max_id_hamming_distance,
        codebook=codebook,
        workers=workers,
    )
    center_update_history.append(dict(final_center_updates))
    if sum(final_center_updates.values()) > 0:
        retained_keys = {(cam, stem, int(target_id)) for cam, stem, target_id, _ in observations}
        refreshed = multi_cam_collect_observations(detections_by_camera, state, None)
        observations = [
            item for item in refreshed
            if (item[0], item[1], int(item[2])) in retained_keys
        ]
        diagnostics = solve_multi_cam_bundle_adjustment(
            state, observations,
            max_iterations=args.max_iterations,
            huber_delta=args.huber_delta,
            loss_relative_tolerance=args.loss_tolerance,
            loss_patience=args.loss_patience,
            robust_loss="none",
            fix_relative_poses=not optimize_relative_poses,
            fix_relative_pose_translations=baseline_vector is not None,
            fixed_point_ids=fixed_point_ids,
            known_baseline=args.known_baseline,
            collect_adjustment_diagnostics=True,
            initial_camera_intrinsics=initial_camera_intrinsics,
            initial_relative_poses=initial_relative_poses,
            observation_sigma_px=args.observation_sigma_px,
        )
        print(f"  final BA after calibrated centre pass: {diagnostics.summary}", flush=True)
        bundle_history.append(("After centre refinement", diagnostics))

    # Persist the exact image measurements used by the final adjustment.  A
    # subsequent cached run must not silently fall back to the provisional
    # ellipse centres that preceded calibrated remeasurement, but frames
    # rejected by the current threshold remain in the cache.
    _persist_refined_detection_cache(detections_by_camera, output_dir)

    checkpoint_quality: dict | None = None
    if checkpoint_ids:
        print("  evaluating withheld checkpoints with the final calibration frozen", flush=True)
        calibrated_frame_stems = {stem for _, stem, _, _ in observations}
        posed_checkpoint_images = {
            camera: [image for image in images if image.image_path.stem in calibrated_frame_stems]
            for camera, images in checkpoint_images.items()
        }
        refine_detected_centers_after_calibration(
            posed_checkpoint_images, state.camera_intrinsics,
            valid_ids=valid_ids,
            max_id_hamming_distance=args.max_id_hamming_distance,
            codebook=codebook,
            workers=workers,
        )
        for camera, images in posed_checkpoint_images.items():
            save_refinement_cache(images, output_dir / camera, merge_existing=True)
        checkpoint_quality = evaluate_independent_checkpoints(
            state, posed_checkpoint_images, checkpoint_ids, known_targets3d,
            calibrated_frame_stems,
        )
        checkpoint_quality["split"] = checkpoint_manifest
        (output_dir / "checkpoint_quality.json").write_text(
            json.dumps(checkpoint_quality, indent=2), encoding="utf-8",
        )
        summary = checkpoint_quality["summary"]
        print(
            f"  checkpoint XYZ: {summary['xyz_reconstructed']}/{summary['selected_targets']} "
            f"reconstructed; 3D RMSE="
            f"{summary.get('rmse_3d_m', float('nan')) * 1000.0:.3f} mm",
            flush=True,
        )

    if args.evaluate_detections:
        print("  evaluating detections against projected reference targets", flush=True)
        from cct_calibration.detection_evaluation import evaluate_detections
        evaluate_detections(
            state, detections_by_camera, checkpoint_images, known_targets3d,
            codebook, output_dir / "detection_evaluation.json", workers=workers,
            frame_stems={stem for _, stem, _, _ in observations},
        )

    (output_dir / "center_remeasurement.json").write_text(
        json.dumps({
            "passes": [
                {"pass": index + 1, "changed_by_camera": values,
                 "intrinsics_used": center_intrinsics_history[index]}
                for index, values in enumerate(center_update_history)
            ],
            "method": "calibrated-concentric-conic",
            "conditional_on_current_intrinsics": True,
            "intrinsics_source": "preceding bundle adjustment, as recorded for each pass",
            "convergence_verified": False,
            "center_localization_covariance_available": False,
            "final_retained_observations": [
                {"camera": camera, "frame": frame, "target_id": int(target_id),
                 "u_px": float(point[0]), "v_px": float(point[1])}
                for camera, frame, target_id, point in observations
            ],
        }, indent=2), encoding="utf-8",
    )

    # ── Results ──────────────────────────────────────────────────
    print("=" * 60, flush=True)
    print("RESULTS", flush=True)
    print("=" * 60, flush=True)

    reproj_errors = multi_cam_compute_reprojection_errors(state, observations)
    track_ids = {tid for _, _, tid, _ in observations}
    frame_stems = {stem for _, stem, _, _ in observations}

    overall = {
        "total_observations": len(observations),
        "total_tracks": len(track_ids),
        "total_frames": len(frame_stems),
        "mean_reprojection_error": float(np.mean(reproj_errors)) if reproj_errors.size else 0.0,
        "rms_reprojection_error": float(np.sqrt(np.mean(np.square(reproj_errors)))) if reproj_errors.size else 0.0,
        "iterations": diagnostics.iterations,
        "initial_cost": diagnostics.initial_cost,
        "final_cost": diagnostics.final_cost,
    }

    # Per-camera results
    per_camera: Dict[str, dict] = {}
    for cam_name in camera_names:
        cam_obs = [(cn, s, t, p) for cn, s, t, p in observations if cn == cam_name]
        cam_errors = multi_cam_compute_reprojection_errors(
            state, cam_obs,
        )
        intr = state.camera_intrinsics[cam_name]
        per_camera[cam_name] = {
            "observations": len(cam_obs),
            "mean_reprojection_error": float(np.mean(cam_errors)) if cam_errors.size else 0.0,
            "rms_reprojection_error": float(np.sqrt(np.mean(np.square(cam_errors)))) if cam_errors.size else 0.0,
            "intrinsics": {
                "fx": float(intr[0]), "fy": float(intr[1]),
                "cx": float(intr[2]), "cy": float(intr[3]),
                "k1": float(intr[4]), "k2": float(intr[5]),
                "p1": float(intr[6]), "p2": float(intr[7]),
            },
        }
    overall["cameras"] = per_camera

    # Relative poses
    rel_poses_out: Dict[str, dict] = {}
    for cam_name in camera_names:
        rp = state.relative_poses[cam_name]
        rel_poses_out[cam_name] = {
            "rvec": rp[:3].tolist(),
            "tvec": rp[3:].tolist(),
        }
    overall["relative_poses"] = rel_poses_out
    overall["frames_dropped_after_filtering"] = dropped_frames
    overall["false_identifications_out_of_view"] = [
        {"camera": cam, "frame": stem, "target_id": int(tid)} for cam, stem, tid, _ in unobservable
    ]
    if checkpoint_quality is not None:
        overall["independent_checkpoint_quality"] = checkpoint_quality

    if diagnostics.adjustment is not None:
        adjustment = diagnostics.adjustment
        diagonal = np.diag(adjustment.covariance)
        def _json_number(value: float) -> float | None:
            return float(value) if np.isfinite(value) else None

        def _json_matrix(matrix: np.ndarray) -> list[list[float | None]]:
            return [
                [_json_number(float(value)) for value in row]
                for row in np.asarray(matrix)
            ]

        overall["adjustment_quality"] = {
            "dof": adjustment.dof,
            "jacobian_rows": adjustment.jacobian_rows,
            "jacobian_columns": adjustment.jacobian_columns,
            "numerical_rank": adjustment.numerical_rank,
            "rank_tolerance": _json_number(adjustment.rank_tolerance or float("nan")),
            "rank_verified": adjustment.rank_verified,
            "nullity": adjustment.nullity,
            "sigma0_px": _json_number(adjustment.sigma0_px),
            "variance_factor_px2": _json_number(adjustment.variance_factor),
            "observation_sigma_px": _json_number(adjustment.observation_sigma_px) if adjustment.observation_sigma_px is not None else None,
            "residual_variance_ratio_observed_to_assumed": (
                _json_number(adjustment.variance_factor / adjustment.observation_sigma_px ** 2)
                if adjustment.observation_sigma_px is not None and adjustment.observation_sigma_px > 0
                else None
            ),
            "residual_variance_interpretation": (
                ("below_assumed_conservative_sigma"
                 if adjustment.variance_factor / adjustment.observation_sigma_px ** 2 <= 1.0
                 else "above_assumed_investigate")
                if adjustment.observation_sigma_px is not None and adjustment.observation_sigma_px > 0
                else "unavailable_without_apriori_sigma"
            ),
            "correlation_condition_number": _json_number(adjustment.condition_number),
            "covariance_method": adjustment.covariance_method,
            "parameters": [
                {
                    "name": name,
                    "initial": float(adjustment.parameter_initial[index]),
                    "final": float(adjustment.parameter_final[index]),
                    "stddev": (
                        float(np.sqrt(diagonal[index]))
                        if np.isfinite(diagonal[index]) and diagonal[index] >= 0 else None
                    ),
                    "status": adjustment.parameter_status[index],
                }
                for index, name in enumerate(adjustment.parameter_names)
            ],
            "covariance": _json_matrix(adjustment.covariance),
            "correlation": _json_matrix(adjustment.correlation),
            "weakest_singular_values": (
                [_json_number(float(v)) for v in adjustment.weakest_singular_values]
                if adjustment.weakest_singular_values is not None else None
            ),
            "warnings": adjustment.warnings,
        }

    summary_path = output_dir / "combined_summary.json"
    summary_path.write_text(json.dumps(overall, indent=2), encoding="utf-8")
    print(json.dumps(overall, indent=2), flush=True)

    # Save convergence plot
    save_convergence_plot("combined", diagnostics.cost_history, output_dir)

    # Save COLMAP output
    colmap_dir = save_multi_cam_colmap_output(state, detections_by_camera, output_dir)
    print(f"COLMAP output: {colmap_dir}", flush=True)

    # Save human-readable report
    report_path = save_report(state, observations, detections_by_camera, diagnostics, output_dir)
    print(f"Report: {report_path}", flush=True)

    # Save Kalibr camchain
    kalibr_path = save_kalibr_camchain(state, detections_by_camera, output_dir)
    print(f"Kalibr camchain: {kalibr_path}", flush=True)

    # ── Rich PDF report ──────────────────────────────────────────
    if not args.no_pdf_report:
        try:
            from cct_calibration.reporting import (
                build_multi_camera_report_data,
                generate_pdf_report,
            )

            meta = {"Image root": str(image_root)}
            if args.targets3d is not None:
                meta["Known 3-D targets"] = str(args.targets3d)
            if baseline_value is not None:
                meta["Known baseline"] = f"{baseline_value} m"
            elif baseline_vector is not None:
                meta["Known baseline vector"] = "[" + ", ".join(f"{float(v):.6g}" for v in baseline_vector) + "] m"
            if args.observation_sigma_px is not None:
                meta["A-priori image sigma"] = f"{args.observation_sigma_px:.4g} px per coordinate"
            if checkpoint_manifest is not None:
                meta["Independent checkpoints"] = (
                    f"{len(checkpoint_ids)} targets; requested ratio {args.checkpoints:.4g}; "
                    f"seed {args.checkpoint_seed}"
                )
            meta["Calibrated centre remeasurement"] = "; ".join(
                f"pass {index + 1}: " + ", ".join(f"{name}={count}" for name, count in values.items())
                for index, values in enumerate(center_update_history)
            )
            meta["Centre refinement method"] = (
                "Concentric conics conditional on preceding-BA intrinsics; "
                "up to two passes, with the second pass reusing cached coordinates when "
                "the camera-model change is below 0.02 px at p95 and 0.05 px maximum. "
                "Counts are changed coordinates."
            )
            if unobservable:
                meta["False identifications (target out of view)"] = f"{len(unobservable)} excluded"
            if dropped_frames:
                meta["Frames dropped after filtering"] = (
                    f"{len(dropped_frames)} with fewer than {args.min_shared} retained observations: "
                    + ", ".join(dropped_frames)
                )
            meta["Centre uncertainty"] = (
                "Localisation covariance unavailable; adjustment uncertainty is conditional "
                "on the final frozen image measurements."
            )

            def observation_key(item):
                cam_name, stem, target_id, point = item
                return cam_name, stem, int(target_id)

            kept_keys = {observation_key(item) for item in observations}
            rejected_observations = [
                item for item in wide_observations
                if observation_key(item) not in kept_keys
            ]

            report_data = build_multi_camera_report_data(
                state, observations, detections_by_camera, diagnostics,
                filter_stats=filter_stats,
                known_targets3d_ids=fixed_point_ids,
                known_baseline=baseline_value,
                rejected_observations=rejected_observations,
                prefilter_reprojection_by_camera=prefilter_reprojection_by_camera,
                checkpoint_quality=checkpoint_quality,
                bundle_history=bundle_history,
                meta=meta,
            )
            pdf_path = generate_pdf_report(
                report_data, output_dir / "calibration_report.pdf",
                plots_dir=output_dir / "plots" if args.save_report_plots else None,
            )
            print(f"PDF report: {pdf_path}", flush=True)
            if args.save_report_plots:
                print(f"Report plots (600 dpi PNG): {output_dir / 'plots'}", flush=True)
        except Exception as exc:  # reporting must never break the pipeline
            print(f"WARNING: PDF report generation failed: {exc}", flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
