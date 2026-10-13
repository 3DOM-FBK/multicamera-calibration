"""Statistical analysis layer for calibration reports.

Computes error statistics, image-plane coverage and observation richness
metrics. The layer is deliberately factual: it produces numbers, not
interpretations.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import cv2
import numpy as np

from cct_calibration.reporting.records import ObservationRecord, ReportData


# ---------------------------------------------------------------------------
# Basic error statistics
# ---------------------------------------------------------------------------

@dataclass
class ErrorStats:
    n: int
    mean: float
    rms: float
    std: float
    median: float
    p95: float
    max: float

    @classmethod
    def from_values(
        cls, values: Sequence[float], *, percentile_absolute: bool = False,
    ) -> "ErrorStats":
        v = np.asarray(values, dtype=np.float64)
        if v.size == 0:
            return cls(0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        return cls(
            n=int(v.size),
            mean=float(np.mean(v)),
            rms=float(np.sqrt(np.mean(np.square(v)))),
            std=float(np.std(v)),
            median=float(np.median(v)),
            p95=float(np.percentile(np.abs(v) if percentile_absolute else v, 95)),
            max=float(np.max(v)),
        )


@dataclass
class SignedResidualStats:
    u: ErrorStats
    v: ErrorStats
    radial: ErrorStats
    tangential: ErrorStats


def signed_residual_stats(records: Sequence[ObservationRecord]) -> SignedResidualStats:
    return SignedResidualStats(
        u=ErrorStats.from_values([record.dx for record in records], percentile_absolute=True),
        v=ErrorStats.from_values([record.dy for record in records], percentile_absolute=True),
        radial=ErrorStats.from_values(
            [record.residual_radial for record in records], percentile_absolute=True,
        ),
        tangential=ErrorStats.from_values(
            [record.residual_tangential for record in records], percentile_absolute=True,
        ),
    )


# ---------------------------------------------------------------------------
# Image-plane coverage
# ---------------------------------------------------------------------------

@dataclass
class CoverageStats:
    grid_rows: int
    grid_cols: int
    occupancy: np.ndarray                 # (rows, cols) bool
    counts: np.ndarray                    # (rows, cols) observation count
    occupied_cells: int
    total_cells: int
    occupancy_fraction: float
    quadrant_occupancy: Dict[str, float]  # TL/TR/BL/BR -> fraction
    empty_regions: List[str]              # human-readable descriptions
    margin_px: Dict[str, float]           # min distance to left/right/top/bottom
    r_norm_max: float
    r_norm_p95: float
    min_cell_observations: int
    sufficient_cells: int
    sufficient_fraction: float


def _quadrant_of(row: int, col: int, rows: int, cols: int) -> str:
    vr = 0 if row < rows / 2 else 1
    hc = 0 if col < cols / 2 else 1
    return {(0, 0): "top-left", (0, 1): "top-right",
            (1, 0): "bottom-left", (1, 1): "bottom-right"}[(vr, hc)]


def compute_coverage(
    records: Sequence[ObservationRecord],
    width: int,
    height: int,
    grid_cols: int = 8,
    grid_rows: int = 5,
    min_cell_observations: int = 20,
) -> CoverageStats:
    """Grid occupancy of the FoV plus margins and normalised-radius reach."""
    aspect = width / max(height, 1)
    grid_rows = max(2, int(round(grid_cols / aspect)))
    occ = np.zeros((grid_rows, grid_cols), dtype=bool)
    counts = np.zeros((grid_rows, grid_cols), dtype=np.int64)

    us = np.array([r.u_meas for r in records], dtype=np.float64)
    vs = np.array([r.v_meas for r in records], dtype=np.float64)

    if us.size and width > 0 and height > 0:
        cols = np.clip((us / width * grid_cols).astype(int), 0, grid_cols - 1)
        rows = np.clip((vs / height * grid_rows).astype(int), 0, grid_rows - 1)
        occ[rows, cols] = True
        np.add.at(counts, (rows, cols), 1)

    occupied = int(occ.sum())
    total = int(occ.size)
    sufficient = int(np.count_nonzero(counts >= min_cell_observations))

    quad_counts: Dict[str, List[bool]] = {"top-left": [], "top-right": [],
                                          "bottom-left": [], "bottom-right": []}
    for row in range(grid_rows):
        for col in range(grid_cols):
            quad_counts[_quadrant_of(row, col, grid_rows, grid_cols)].append(bool(occ[row, col]))
    quadrant_occ = {k: float(np.mean(v)) if v else 0.0 for k, v in quad_counts.items()}

    empty_regions: List[str] = []
    for row in range(grid_rows):
        for col in range(grid_cols):
            if not occ[row, col]:
                empty_regions.append(_quadrant_of(row, col, grid_rows, grid_cols))
    # compress to unique quadrants that have any empty cell, ordered
    seen: List[str] = []
    for q in empty_regions:
        if q not in seen:
            seen.append(q)

    if us.size and width > 0:
        margins = {
            "left": float(us.min()),
            "right": float(width - us.max()),
            "top": float(vs.min()),
            "bottom": float(height - vs.max()),
        }
    else:
        margins = {"left": 0.0, "right": 0.0, "top": 0.0, "bottom": 0.0}

    rnorms = np.array([r.r_norm for r in records], dtype=np.float64)
    return CoverageStats(
        grid_rows=grid_rows,
        grid_cols=grid_cols,
        occupancy=occ,
        counts=counts,
        occupied_cells=occupied,
        total_cells=total,
        occupancy_fraction=float(occupied) / max(total, 1),
        quadrant_occupancy=quadrant_occ,
        empty_regions=seen,
        margin_px=margins,
        r_norm_max=float(rnorms.max()) if rnorms.size else 0.0,
        r_norm_p95=float(np.percentile(rnorms, 95)) if rnorms.size else 0.0,
        min_cell_observations=int(min_cell_observations),
        sufficient_cells=sufficient,
        sufficient_fraction=float(sufficient) / max(total, 1),
    )


# ---------------------------------------------------------------------------
# Observation richness
# ---------------------------------------------------------------------------

@dataclass
class RichnessStats:
    n_observations: int
    n_targets: int
    n_frames: int
    obs_per_target: np.ndarray
    targets_per_frame: np.ndarray
    obs_per_frame: np.ndarray
    unknowns: int
    dof: int
    redundancy_ratio: float

    @property
    def median_obs_per_target(self) -> float:
        return float(np.median(self.obs_per_target)) if self.obs_per_target.size else 0.0

    @property
    def median_targets_per_frame(self) -> float:
        return float(np.median(self.targets_per_frame)) if self.targets_per_frame.size else 0.0


def compute_richness(records: Sequence[ObservationRecord], n_cameras: int,
                     n_fixed_points: int = 0) -> RichnessStats:
    obs_per_target_list: Dict[int, int] = {}
    targets_per_frame: Dict[str, set] = {}
    obs_per_frame: Dict[str, int] = {}
    for rec in records:
        obs_per_target_list[rec.target_id] = obs_per_target_list.get(rec.target_id, 0) + 1
        targets_per_frame.setdefault(rec.frame, set()).add(rec.target_id)
        obs_per_frame[rec.frame] = obs_per_frame.get(rec.frame, 0) + 1

    opt = np.array(sorted(obs_per_target_list.values()), dtype=np.float64)
    tpf = np.array([len(v) for v in targets_per_frame.values()], dtype=np.float64)
    opf = np.array(list(obs_per_frame.values()), dtype=np.float64)

    n_obs = len(records)
    n_targets = int(opt.size)
    n_frames = len(targets_per_frame)
    # The nominal count follows the parameterization used by the joint
    # adjustment.  With free object coordinates and no metric anchors the
    # first rig pose is fixed to remove the six-parameter world gauge; with
    # fixed 3-D targets the target frame already supplies that datum and all
    # rig poses remain estimable.
    gauge_columns = 0 if n_fixed_points else min(6, 6 * n_frames)
    unknowns = 8 * n_cameras + 6 * n_frames + 3 * max(n_targets - n_fixed_points, 0) \
        + 6 * max(n_cameras - 1, 0) - gauge_columns
    dof = 2 * n_obs - unknowns
    redundancy = dof / (2 * n_obs) if n_obs else 0.0

    return RichnessStats(
        n_observations=n_obs,
        n_targets=n_targets,
        n_frames=n_frames,
        obs_per_target=opt,
        targets_per_frame=tpf,
        obs_per_frame=opf,
        unknowns=int(unknowns),
        dof=int(dof),
        redundancy_ratio=float(redundancy),
    )


# ---------------------------------------------------------------------------
# Spatial residual structure
# ---------------------------------------------------------------------------

def radial_error_trend(records: Sequence[ObservationRecord], n_bins: int = 8) -> Tuple[np.ndarray, np.ndarray]:
    """Binned mean error vs normalised radius. Returns (bin_centers, bin_means)."""
    r = np.array([rec.r_norm for rec in records])
    e = np.array([rec.err_px for rec in records])
    if r.size == 0:
        return np.array([]), np.array([])
    r_max = max(r.max(), 1e-6)
    edges = np.linspace(0, r_max * 1.0001, n_bins + 1)
    centers, means = [], []
    for i in range(n_bins):
        m = (r >= edges[i]) & (r < edges[i + 1])
        if m.any():
            centers.append(float(0.5 * (edges[i] + edges[i + 1])))
            means.append(float(e[m].mean()))
    return np.array(centers), np.array(means)


# ---------------------------------------------------------------------------
# Aggregated per-camera report statistics
# ---------------------------------------------------------------------------

@dataclass
class CameraReportStats:
    camera: str
    errors: ErrorStats
    errors_obj_mm: ErrorStats
    coverage: CoverageStats
    n_depth_valid: int
    depth_median_m: float


@dataclass
class EntityReliability:
    name: str
    n_observations: int
    rms_px: float
    mean_local_redundancy: float | None


def weakest_frames_and_targets(
    records: Sequence[ObservationRecord],
    limit: int = 10,
) -> Tuple[List[EntityReliability], List[EntityReliability]]:
    """Rank frames/targets using high residual and low local redundancy."""
    def aggregate(key):
        grouped: Dict[object, List[ObservationRecord]] = {}
        for record in records:
            grouped.setdefault(key(record), []).append(record)
        result: List[EntityReliability] = []
        for name, group in grouped.items():
            errors = np.array([record.err_px for record in group], dtype=np.float64)
            redundancies = [
                0.5 * (record.local_redundancy_u + record.local_redundancy_v)
                for record in group
                if record.local_redundancy_u is not None and record.local_redundancy_v is not None
            ]
            result.append(EntityReliability(
                name=str(name),
                n_observations=len(group),
                rms_px=float(np.sqrt(np.mean(errors ** 2))),
                mean_local_redundancy=float(np.mean(redundancies)) if redundancies else None,
            ))
        result.sort(key=lambda item: (
            item.mean_local_redundancy if item.mean_local_redundancy is not None else 1.0,
            -item.rms_px,
            item.n_observations,
        ))
        return result[:limit]

    return aggregate(lambda record: record.frame), aggregate(lambda record: record.target_id)


@dataclass
class RigPairStats:
    camera_a: str
    camera_b: str
    shared_frames: int
    shared_targets: int
    common_observations: int
    sampson_px: np.ndarray
    vertical_disparity_px: np.ndarray
    intersection_angle_deg: np.ndarray
    baseline_depth_ratio: np.ndarray
    internal_xyz_errors_m: np.ndarray
    rectified_disparity_axis: str = "unavailable"


def _relative_transform_between(data: ReportData, camera_a: str, camera_b: str) -> Tuple[np.ndarray, np.ndarray]:
    assert data.rig is not None
    pose_a = data.rig.relative_poses[camera_a]
    pose_b = data.rig.relative_poses[camera_b]
    rotation_a = cv2.Rodrigues(pose_a[:3])[0]
    rotation_b = cv2.Rodrigues(pose_b[:3])[0]
    rotation = rotation_b @ rotation_a.T
    translation = pose_b[3:] - rotation @ pose_a[3:]
    return rotation, translation


def _triangulate_pair_normalized(
    point_a: np.ndarray,
    point_b: np.ndarray,
    rotation_ab: np.ndarray,
    translation_ab: np.ndarray,
) -> np.ndarray | None:
    projection_a = np.hstack([np.eye(3), np.zeros((3, 1))])
    projection_b = np.hstack([rotation_ab, translation_ab.reshape(3, 1)])
    homogeneous = cv2.triangulatePoints(
        projection_a, projection_b,
        point_a.reshape(2, 1), point_b.reshape(2, 1),
    )
    if abs(float(homogeneous[3, 0])) < 1e-12:
        return None
    point = homogeneous[:3, 0] / homogeneous[3, 0]
    return point if np.all(np.isfinite(point)) else None


def compute_rig_pair_stats(data: ReportData) -> List[RigPairStats]:
    """Internal stereo diagnostics for every camera pair.

    ``internal_xyz_errors_m`` compares stereo triangulation with the adjusted
    target coordinates. It is deliberately not labelled independent accuracy.
    """
    if data.rig is None or len(data.cameras) < 2:
        return []
    camera_by_name = {camera.name: camera for camera in data.cameras}
    grouped = {(r.camera, r.frame, r.target_id): r for r in data.records}
    results: List[RigPairStats] = []
    for ia, camera_a in enumerate(data.cameras):
        for camera_b in data.cameras[ia + 1:]:
            pairs = []
            for key, record_a in grouped.items():
                if key[0] != camera_a.name:
                    continue
                record_b = grouped.get((camera_b.name, key[1], key[2]))
                if record_b is not None:
                    pairs.append((record_a, record_b))
            rotation, translation = _relative_transform_between(data, camera_a.name, camera_b.name)
            essential = np.array([
                [0.0, -translation[2], translation[1]],
                [translation[2], 0.0, -translation[0]],
                [-translation[1], translation[0], 0.0],
            ]) @ rotation
            matrix_a = np.array([
                [camera_a.intrinsics[0], 0.0, camera_a.intrinsics[2]],
                [0.0, camera_a.intrinsics[1], camera_a.intrinsics[3]],
                [0.0, 0.0, 1.0],
            ])
            matrix_b = np.array([
                [camera_b.intrinsics[0], 0.0, camera_b.intrinsics[2]],
                [0.0, camera_b.intrinsics[1], camera_b.intrinsics[3]],
                [0.0, 0.0, 1.0],
            ])
            distortion_a = camera_a.intrinsics[4:8]
            distortion_b = camera_b.intrinsics[4:8]
            rectification = None
            if (camera_a.width, camera_a.height) == (camera_b.width, camera_b.height):
                try:
                    rectification = cv2.stereoRectify(
                        matrix_a, distortion_a, matrix_b, distortion_b,
                        (camera_a.width, camera_a.height), rotation, translation,
                        flags=cv2.CALIB_ZERO_DISPARITY, alpha=0,
                    )
                except cv2.error:
                    rectification = None

            # Fundamental matrix for undistorted *pixel* coordinates.  This
            # avoids the former normalized-coordinate/average-focal shortcut.
            fundamental = np.linalg.inv(matrix_b).T @ essential @ np.linalg.inv(matrix_a)
            sampson, vertical, angles, ratios, xyz_errors = [], [], [], [], []
            disparity_axis = "unavailable"
            disparity_component = None
            if rectification is not None:
                projection_b_rect = rectification[3]
                horizontal_epipolar = abs(float(projection_b_rect[0, 3])) >= abs(float(projection_b_rect[1, 3]))
                disparity_component = 1 if horizontal_epipolar else 0
                disparity_axis = "vertical" if horizontal_epipolar else "horizontal"
            baseline = float(np.linalg.norm(translation))
            for record_a, record_b in pairs:
                uv_a = np.array([record_a.u_meas, record_a.v_meas], dtype=np.float64)
                uv_b = np.array([record_b.u_meas, record_b.v_meas], dtype=np.float64)
                norm_a = cv2.undistortPoints(uv_a.reshape(1, 1, 2), matrix_a, distortion_a)[0, 0]
                norm_b = cv2.undistortPoints(uv_b.reshape(1, 1, 2), matrix_b, distortion_b)[0, 0]
                undistorted_a_px = cv2.undistortPoints(
                    uv_a.reshape(1, 1, 2), matrix_a, distortion_a, P=matrix_a,
                )[0, 0]
                undistorted_b_px = cv2.undistortPoints(
                    uv_b.reshape(1, 1, 2), matrix_b, distortion_b, P=matrix_b,
                )[0, 0]
                x_a = np.array([undistorted_a_px[0], undistorted_a_px[1], 1.0])
                x_b = np.array([undistorted_b_px[0], undistorted_b_px[1], 1.0])
                ex = fundamental @ x_a
                etx = fundamental.T @ x_b
                numerator = float(x_b @ ex)
                denominator = ex[0] ** 2 + ex[1] ** 2 + etx[0] ** 2 + etx[1] ** 2
                if denominator > 1e-18:
                    sampson.append(abs(numerator) / np.sqrt(denominator))

                if rectification is not None:
                    r1, r2, p1, p2 = rectification[:4]
                    rect_a = cv2.undistortPoints(uv_a.reshape(1, 1, 2), matrix_a, distortion_a, R=r1, P=p1)[0, 0]
                    rect_b = cv2.undistortPoints(uv_b.reshape(1, 1, 2), matrix_b, distortion_b, R=r2, P=p2)[0, 0]
                    if disparity_component is not None:
                        vertical.append(float(rect_b[disparity_component] - rect_a[disparity_component]))

                ray_a = np.array([norm_a[0], norm_a[1], 1.0])
                ray_b_a = rotation.T @ np.array([norm_b[0], norm_b[1], 1.0])
                ray_a /= np.linalg.norm(ray_a)
                ray_b_a /= np.linalg.norm(ray_b_a)
                cosine = float(np.clip(ray_a @ ray_b_a, -1.0, 1.0))
                angles.append(float(np.degrees(np.arccos(cosine))))
                depths = [value for value in (record_a.depth_m, record_b.depth_m) if value is not None]
                if depths and np.mean(depths) > 0:
                    ratios.append(baseline / float(np.mean(depths)))

                triangulated_a = _triangulate_pair_normalized(norm_a, norm_b, rotation, translation)
                target_world = data.points3d.get(record_a.target_id)
                rig_pose = data.rig.rig_poses.get(record_a.frame)
                if triangulated_a is not None and target_world is not None and rig_pose is not None:
                    # Bring the adjusted target to camera-A coordinates.
                    rel_a = data.rig.relative_poses[camera_a.name]
                    rotation_rig = cv2.Rodrigues(rig_pose[:3])[0]
                    rotation_a = cv2.Rodrigues(rel_a[:3])[0] @ rotation_rig
                    translation_a = cv2.Rodrigues(rel_a[:3])[0] @ rig_pose[3:] + rel_a[3:]
                    target_a = rotation_a @ target_world + translation_a
                    xyz_errors.append(triangulated_a - target_a)

            results.append(RigPairStats(
                camera_a=camera_a.name,
                camera_b=camera_b.name,
                shared_frames=len({a.frame for a, _ in pairs}),
                shared_targets=len({a.target_id for a, _ in pairs}),
                common_observations=len(pairs),
                sampson_px=np.asarray(sampson, dtype=np.float64),
                vertical_disparity_px=np.asarray(vertical, dtype=np.float64),
                intersection_angle_deg=np.asarray(angles, dtype=np.float64),
                baseline_depth_ratio=np.asarray(ratios, dtype=np.float64),
                internal_xyz_errors_m=np.asarray(xyz_errors, dtype=np.float64).reshape(-1, 3),
                rectified_disparity_axis=disparity_axis,
            ))
    return results


def per_camera_stats(data: ReportData) -> List[CameraReportStats]:
    out: List[CameraReportStats] = []
    for cam in data.cameras:
        recs = [r for r in data.records if r.camera == cam.name]
        errs = ErrorStats.from_values([r.err_px for r in recs])
        objs = ErrorStats.from_values([r.err_obj_m * 1000.0 for r in recs
                                       if r.err_obj_m is not None])
        depths = [r.depth_m for r in recs if r.depth_m is not None]
        out.append(CameraReportStats(
            camera=cam.name,
            errors=errs,
            errors_obj_mm=objs,
            coverage=compute_coverage(recs, cam.width, cam.height),
            n_depth_valid=len(depths),
            depth_median_m=float(np.median(depths)) if depths else 0.0,
        ))
    return out
