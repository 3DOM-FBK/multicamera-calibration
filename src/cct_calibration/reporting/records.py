"""Per-observation record extraction for calibration reporting.

Converts the optimiser-level state (intrinsics, poses, 3D points, observation
tuples) into flat per-observation records that carry everything the statistics
and plotting layers need: measured/predicted image coordinates, pixel and
object-space residuals, depth, normalised coordinates, etc.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Sequence, Set, Tuple

import numpy as np

from cct_calibration.run import (
    ImageDetections,
    SolverDiagnostics,
    camera_center_from_pose,
    project_point,
    rotation_matrix_from_pose,
)
from cct_calibration.run_combined import (
    MetricInlierFilterStats,
    MultiCameraState,
    compose_poses,
    multi_cam_compute_object_space_residual,
)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class ObservationRecord:
    """Everything worth knowing about a single 2D-3D observation."""
    camera: str
    frame: str
    target_id: int
    u_meas: float
    v_meas: float
    u_pred: float
    v_pred: float
    dx: float                 # predicted - measured (x)
    dy: float                 # predicted - measured (y)
    residual_radial: float    # signed component away from principal point
    residual_tangential: float
    err_px: float             # euclidean pixel residual
    err_obj_m: float | None   # point-to-ray object-space residual (metres)
    depth_m: float | None     # z of the 3D point in the camera frame
    xn: float = 0.0           # normalised coords (distorted pinhole plane)
    yn: float = 0.0
    r_norm: float = 0.0       # distance from principal point (normalised)
    standardized_dx: float | None = None
    standardized_dy: float | None = None
    leverage_u: float | None = None
    leverage_v: float | None = None
    local_redundancy_u: float | None = None
    local_redundancy_v: float | None = None


@dataclass
class CameraInfo:
    name: str
    width: int
    height: int
    intrinsics: np.ndarray    # [fx, fy, cx, cy, k1, k2, p1, p2]
    n_images_detected: int = 0   # images with >= min_detections (input to BA)
    n_images_total: int = 0      # all images found on disk


@dataclass
class RigInfo:
    """Camera trajectory and scene geometry; relative orientations are trivial for one camera."""
    relative_poses: Dict[str, np.ndarray]            # cam -> 6-vector (w.r.t. reference)
    rig_poses: Dict[str, np.ndarray]                 # retained frame -> reference-camera pose
    frame_stems: List[str]
    baselines_m: Dict[str, float]                    # cam -> |t_rel|
    rel_rotation_deg: Dict[str, float]               # cam -> angle of R_rel
    camera_centers: Dict[str, np.ndarray]            # cam -> (N, 3) world-frame centres
    camera_forwards: Dict[str, np.ndarray]           # cam -> (N, 3) world-frame viewing dirs
    rig_centers: np.ndarray                          # (N, 3) reference-camera centres
    points_array: np.ndarray                         # (M, 3) optimised target positions
    fixed_point_ids: Set[int]                        # targets anchored to ground truth
    known_baseline_input: float | None               # user-supplied baseline magnitude


@dataclass
class ReportData:
    """Self-contained payload consumed by the PDF builder."""
    mode: str                                        # "multi-camera" | "single-camera"
    generated_at: datetime
    cameras: List[CameraInfo]
    records: List[ObservationRecord]
    rejected_records: List[ObservationRecord]
    prefilter_reprojection_by_camera: Dict[str, np.ndarray]
    points3d: Dict[int, np.ndarray]
    solver: SolverDiagnostics | None
    filter_stats: MetricInlierFilterStats | None
    rig: RigInfo | None
    adjustment: Any | None = None
    checkpoint_quality: Dict[str, Any] | None = None
    bundle_history: List[Tuple[str, Any]] = field(default_factory=list)
    meta: Dict[str, str] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------

def _normalised_coords(intr: np.ndarray, u: float, v: float) -> Tuple[float, float, float]:
    xn = (u - intr[2]) / intr[0]
    yn = (v - intr[3]) / intr[1]
    return xn, yn, float(np.hypot(xn, yn))


def records_from_multi_camera(
    state: MultiCameraState,
    observations: Sequence[Tuple[str, str, int, np.ndarray]],
) -> List[ObservationRecord]:
    """Build one :class:`ObservationRecord` per observation of a solved rig."""
    records: List[ObservationRecord] = []
    for cam_name, stem, target_id, pt2d in observations:
        intr = state.camera_intrinsics[cam_name]
        rig_pose = state.rig_poses.get(stem)
        pt3d = state.points.get(target_id)
        if rig_pose is None or pt3d is None:
            continue
        abs_pose = compose_poses(rig_pose, state.relative_poses[cam_name])
        pred = project_point(intr, abs_pose, pt3d)
        dx = float(pred[0] - pt2d[0])
        dy = float(pred[1] - pt2d[1])
        radial = np.array([pt2d[0] - intr[2], pt2d[1] - intr[3]], dtype=np.float64)
        radial_norm = float(np.linalg.norm(radial))
        if radial_norm > 1e-12:
            radial /= radial_norm
            residual_radial = float(dx * radial[0] + dy * radial[1])
            residual_tangential = float(-dx * radial[1] + dy * radial[0])
        else:
            residual_radial = 0.0
            residual_tangential = 0.0

        rotation = rotation_matrix_from_pose(abs_pose)
        point_cam = rotation @ pt3d + abs_pose[3:]
        depth = float(point_cam[2]) if point_cam[2] > 1e-9 else None

        obs_tuple = (cam_name, stem, target_id, pt2d)
        err_obj = multi_cam_compute_object_space_residual(state, obs_tuple)

        xn, yn, r_norm = _normalised_coords(intr, float(pt2d[0]), float(pt2d[1]))
        records.append(ObservationRecord(
            camera=cam_name,
            frame=stem,
            target_id=int(target_id),
            u_meas=float(pt2d[0]),
            v_meas=float(pt2d[1]),
            u_pred=float(pred[0]),
            v_pred=float(pred[1]),
            dx=dx,
            dy=dy,
            residual_radial=residual_radial,
            residual_tangential=residual_tangential,
            err_px=float(np.hypot(dx, dy)),
            err_obj_m=err_obj,
            depth_m=depth,
            xn=xn,
            yn=yn,
            r_norm=r_norm,
        ))
    return records


def build_multi_camera_report_data(
    state: MultiCameraState,
    observations: Sequence[Tuple[str, str, int, np.ndarray]],
    detections_by_camera: Dict[str, List[ImageDetections]],
    diagnostics: SolverDiagnostics | None,
    filter_stats: MetricInlierFilterStats | None = None,
    known_targets3d_ids: Set[int] | None = None,
    known_baseline: float | None = None,
    rejected_observations: Sequence[Tuple[str, str, int, np.ndarray]] | None = None,
    prefilter_reprojection_by_camera: Dict[str, np.ndarray] | None = None,
    checkpoint_quality: Dict[str, Any] | None = None,
    bundle_history: Sequence[Tuple[str, Any]] | None = None,
    meta: Dict[str, str] | None = None,
) -> ReportData:
    """Assemble the full reporting payload from a solved multi-camera state."""
    cameras: List[CameraInfo] = []
    for cam_name in state.camera_names:
        dets = detections_by_camera.get(cam_name, [])
        w = dets[0].width if dets else 0
        h = dets[0].height if dets else 0
        cameras.append(CameraInfo(
            name=cam_name,
            width=int(w),
            height=int(h),
            intrinsics=state.camera_intrinsics[cam_name].copy(),
            n_images_detected=len(dets),
        ))

    records = records_from_multi_camera(state, observations)
    rejected_records = records_from_multi_camera(state, rejected_observations or [])

    adjustment = getattr(diagnostics, "adjustment", None) if diagnostics is not None else None
    if adjustment is not None and adjustment.standardized_residuals_uv.shape == (len(records), 2):
        for index, record in enumerate(records):
            values = (
                adjustment.standardized_residuals_uv[index, 0],
                adjustment.standardized_residuals_uv[index, 1],
                adjustment.leverage_uv[index, 0],
                adjustment.leverage_uv[index, 1],
                adjustment.local_redundancy_uv[index, 0],
                adjustment.local_redundancy_uv[index, 1],
            )
            record.standardized_dx = float(values[0]) if np.isfinite(values[0]) else None
            record.standardized_dy = float(values[1]) if np.isfinite(values[1]) else None
            record.leverage_u = float(values[2]) if np.isfinite(values[2]) else None
            record.leverage_v = float(values[3]) if np.isfinite(values[3]) else None
            record.local_redundancy_u = float(values[4]) if np.isfinite(values[4]) else None
            record.local_redundancy_v = float(values[5]) if np.isfinite(values[5]) else None

    # Scene geometry -------------------------------------------------------
    ref_cam = state.camera_names[0]
    # Plot/report only frames that survived the final observation selection.
    stems = sorted({record.frame for record in records if record.frame in state.rig_poses})
    rig_centers = np.array(
        [camera_center_from_pose(state.rig_poses[s]) for s in stems],
        dtype=np.float64,
    ) if stems else np.zeros((0, 3))

    centers_by_cam: Dict[str, np.ndarray] = {}
    forwards_by_cam: Dict[str, np.ndarray] = {}
    for cam_name in state.camera_names:
        pts = []
        fwd = []
        for s in stems:
            abs_pose = compose_poses(state.rig_poses[s], state.relative_poses[cam_name])
            pts.append(camera_center_from_pose(abs_pose))
            fwd.append(rotation_matrix_from_pose(abs_pose).T @ np.array([0.0, 0.0, 1.0]))
        centers_by_cam[cam_name] = np.array(pts, dtype=np.float64) if pts else np.zeros((0, 3))
        forwards_by_cam[cam_name] = np.array(fwd, dtype=np.float64) if fwd else np.zeros((0, 3))

    baselines: Dict[str, float] = {}
    angles_deg: Dict[str, float] = {}
    for cam_name in state.camera_names:
        rp = state.relative_poses[cam_name]
        baselines[cam_name] = float(np.linalg.norm(rp[3:]))
        angles_deg[cam_name] = float(np.degrees(np.linalg.norm(rp[:3])))

    if state.points:
        points_array = np.array([state.points[t] for t in sorted(state.points)], dtype=np.float64)
    else:
        points_array = np.zeros((0, 3))

    rig = RigInfo(
        relative_poses={cn: state.relative_poses[cn].copy() for cn in state.camera_names},
        rig_poses={stem: state.rig_poses[stem].copy() for stem in stems},
        frame_stems=list(stems),
        baselines_m=baselines,
        rel_rotation_deg=angles_deg,
        camera_centers=centers_by_cam,
        camera_forwards=forwards_by_cam,
        rig_centers=rig_centers,
        points_array=points_array,
        fixed_point_ids=set(known_targets3d_ids or set()),
        known_baseline_input=float(known_baseline) if known_baseline is not None else None,
    )

    return ReportData(
        mode="multi-camera" if len(state.camera_names) > 1 else "single-camera",
        generated_at=datetime.now(),
        cameras=cameras,
        records=records,
        rejected_records=rejected_records,
        prefilter_reprojection_by_camera={
            name: np.asarray(values, dtype=np.float64).copy()
            for name, values in (prefilter_reprojection_by_camera or {}).items()
        },
        points3d={tid: p.copy() for tid, p in state.points.items()},
        solver=diagnostics,
        filter_stats=filter_stats,
        rig=rig,
        adjustment=adjustment,
        checkpoint_quality=checkpoint_quality,
        bundle_history=list(bundle_history or []),
        meta=dict(meta or {}),
    )
