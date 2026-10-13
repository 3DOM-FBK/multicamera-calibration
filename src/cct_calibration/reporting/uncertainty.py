"""Propagation of calibration covariance into operational pixel/3-D units."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from cct_calibration.run import project_point
from cct_calibration.reporting.records import CameraInfo, ReportData


INTRINSIC_NAMES = ("fx", "fy", "cx", "cy", "k1", "k2", "p1", "p2")
POSE_NAMES = ("rx", "ry", "rz", "tx", "ty", "tz")


def _covariance_submatrix(adjustment, indices: list[int] | np.ndarray) -> np.ndarray | None:
    """Return an honest covariance block, preserving unavailable estimates."""
    idx = np.asarray(indices, dtype=int)
    covariance = np.asarray(adjustment.covariance[np.ix_(idx, idx)], dtype=np.float64).copy()
    statuses = [adjustment.parameter_status[int(i)] for i in idx]
    estimated = [not str(status).startswith("fixed") for status in statuses]
    for row, is_estimated in enumerate(estimated):
        if not is_estimated:
            covariance[row, :] = 0.0
            covariance[:, row] = 0.0
    if any(estimated) and not np.all(np.isfinite(covariance[np.ix_(estimated, estimated)])):
        return None
    if not np.all(np.isfinite(covariance)):
        return None
    return covariance


def _matrix(intrinsics: np.ndarray) -> np.ndarray:
    return np.array([
        [intrinsics[0], 0.0, intrinsics[2]],
        [0.0, intrinsics[1], intrinsics[3]],
        [0.0, 0.0, 1.0],
    ])


def _forward_intrinsics(intrinsics: np.ndarray, normalized_point: np.ndarray) -> np.ndarray:
    x, y = normalized_point
    radius_sq = x * x + y * y
    radial = 1.0 + intrinsics[4] * radius_sq + intrinsics[5] * radius_sq ** 2
    xt = 2.0 * intrinsics[6] * x * y + intrinsics[7] * (radius_sq + 2.0 * x * x)
    yt = intrinsics[6] * (radius_sq + 2.0 * y * y) + 2.0 * intrinsics[7] * x * y
    return np.array([
        intrinsics[0] * (x * radial + xt) + intrinsics[2],
        intrinsics[1] * (y * radial + yt) + intrinsics[3],
    ])


def _step(value: float) -> float:
    return 1e-6 * max(1.0, abs(float(value)))


def projection_uncertainty_grid(
    data: ReportData,
    camera: CameraInfo,
    columns: int = 25,
    rows: int = 16,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """1-sigma camera-local projection uncertainty over the sensor.

    The normalized ray is held fixed, so this visualizes the intrinsic-model
    contribution without mixing it with an arbitrary world/rig pose.
    """
    adjustment = data.adjustment
    if adjustment is None or adjustment.covariance.size == 0:
        return None
    name_to_index = {name: i for i, name in enumerate(adjustment.parameter_names)}
    indices = [name_to_index.get(f"{camera.name}.{name}") for name in INTRINSIC_NAMES]
    if any(index is None for index in indices):
        return None
    indices = np.asarray(indices, dtype=int)
    covariance = _covariance_submatrix(adjustment, indices)
    if covariance is None:
        return None

    xs = np.linspace(0.0, camera.width - 1.0, columns)
    ys = np.linspace(0.0, camera.height - 1.0, rows)
    grid = np.zeros((rows, columns), dtype=np.float64)
    camera_matrix = _matrix(camera.intrinsics)
    for row, v in enumerate(ys):
        for col, u in enumerate(xs):
            normalized = cv2.undistortPoints(
                np.array([u, v], dtype=np.float64).reshape(1, 1, 2),
                camera_matrix, camera.intrinsics[4:8],
            )[0, 0]
            jacobian = np.zeros((2, 8), dtype=np.float64)
            for parameter in range(8):
                step = _step(camera.intrinsics[parameter])
                plus = camera.intrinsics.copy()
                minus = camera.intrinsics.copy()
                plus[parameter] += step
                minus[parameter] -= step
                jacobian[:, parameter] = (
                    _forward_intrinsics(plus, normalized)
                    - _forward_intrinsics(minus, normalized)
                ) / (2.0 * step)
            output_covariance = jacobian @ covariance @ jacobian.T
            grid[row, col] = np.sqrt(max(float(np.trace(output_covariance)), 0.0))
    return xs, ys, grid


def _parameter_vector(data: ReportData) -> tuple[np.ndarray, dict[str, int]]:
    adjustment = data.adjustment
    return adjustment.parameter_final.copy(), {
        name: index for index, name in enumerate(adjustment.parameter_names)
    }


def _intrinsics_from(vector: np.ndarray, lookup: dict[str, int], camera: str) -> np.ndarray:
    return np.array([vector[lookup[f"{camera}.{name}"]] for name in INTRINSIC_NAMES])


def _pose_from(vector: np.ndarray, lookup: dict[str, int], camera: str) -> np.ndarray:
    return np.array([vector[lookup[f"{camera}.{name}"]] for name in POSE_NAMES])


def _projected_in_sensor(intrinsics: np.ndarray, pose: np.ndarray, point: np.ndarray, width: int, height: int) -> bool:
    rotation = cv2.Rodrigues(pose[:3])[0]
    camera_point = rotation @ point + pose[3:]
    if not np.all(np.isfinite(camera_point)) or float(camera_point[2]) <= 0:
        return False
    projected = project_point(intrinsics, pose, point)
    return bool(
        np.all(np.isfinite(projected))
        and -0.5 <= float(projected[0]) < width - 0.5
        and -0.5 <= float(projected[1]) < height - 0.5
    )


def projection_uncertainty_vs_distance(
    data: ReportData,
    camera: CameraInfo,
    distances: np.ndarray,
) -> np.ndarray | None:
    """Propagate joint intrinsic/extrinsic covariance for a point on cam0's axis."""
    if data.adjustment is None or data.rig is None:
        return None
    vector, lookup = _parameter_vector(data)
    covariance = np.asarray(data.adjustment.covariance, dtype=np.float64)
    relevant_names = [f"{camera.name}.{name}" for name in INTRINSIC_NAMES]
    if camera.name != data.cameras[0].name:
        relevant_names += [f"{camera.name}.{name}" for name in POSE_NAMES]
    relevant = [lookup[name] for name in relevant_names if name in lookup]
    if not relevant:
        return None
    local_covariance = _covariance_submatrix(data.adjustment, relevant)
    if local_covariance is None:
        return None

    output = np.zeros_like(distances, dtype=np.float64)
    for distance_index, distance in enumerate(distances):
        point = np.array([0.0, 0.0, float(distance)])

        def evaluate(values: np.ndarray) -> np.ndarray:
            intrinsics = _intrinsics_from(values, lookup, camera.name)
            pose = _pose_from(values, lookup, camera.name)
            return project_point(intrinsics, pose, point)

        if not _projected_in_sensor(
            _intrinsics_from(vector, lookup, camera.name),
            _pose_from(vector, lookup, camera.name), point,
            camera.width, camera.height,
        ):
            output[distance_index] = np.nan
            continue

        jacobian = np.zeros((2, len(relevant)), dtype=np.float64)
        for column, parameter in enumerate(relevant):
            step = _step(vector[parameter])
            plus, minus = vector.copy(), vector.copy()
            plus[parameter] += step
            minus[parameter] -= step
            jacobian[:, column] = (evaluate(plus) - evaluate(minus)) / (2.0 * step)
        projected = jacobian @ local_covariance @ jacobian.T
        output[distance_index] = np.sqrt(max(float(np.trace(projected)), 0.0))
    return output


def _triangulate(
    intrinsics_a: np.ndarray,
    intrinsics_b: np.ndarray,
    pose_b: np.ndarray,
    point_a_px: np.ndarray,
    point_b_px: np.ndarray,
) -> np.ndarray | None:
    normalized_a = cv2.undistortPoints(
        point_a_px.reshape(1, 1, 2), _matrix(intrinsics_a), intrinsics_a[4:8],
    )[0, 0]
    normalized_b = cv2.undistortPoints(
        point_b_px.reshape(1, 1, 2), _matrix(intrinsics_b), intrinsics_b[4:8],
    )[0, 0]
    rotation_b = cv2.Rodrigues(pose_b[:3])[0]
    projection_a = np.hstack([np.eye(3), np.zeros((3, 1))])
    projection_b = np.hstack([rotation_b, pose_b[3:].reshape(3, 1)])
    homogeneous = cv2.triangulatePoints(
        projection_a, projection_b,
        normalized_a.reshape(2, 1), normalized_b.reshape(2, 1),
    )
    if abs(float(homogeneous[3, 0])) < 1e-12:
        return None
    point = homogeneous[:3, 0] / homogeneous[3, 0]
    return point if np.all(np.isfinite(point)) else None


@dataclass
class TriangulationUncertaintyCurve:
    camera_a: str
    camera_b: str
    distances_m: np.ndarray
    sigma_x_m: np.ndarray
    sigma_y_m: np.ndarray
    sigma_z_m: np.ndarray


def triangulation_uncertainty_vs_distance(
    data: ReportData,
    camera_b: CameraInfo,
    distances: np.ndarray,
) -> TriangulationUncertaintyCurve | None:
    """Linearized stereo coordinate uncertainty on the reference optical axis."""
    if data.adjustment is None or data.rig is None:
        return None
    camera_a = data.cameras[0]
    vector, lookup = _parameter_vector(data)
    covariance = np.asarray(data.adjustment.covariance, dtype=np.float64)
    relevant_names = (
        [f"{camera_a.name}.{name}" for name in INTRINSIC_NAMES]
        + [f"{camera_b.name}.{name}" for name in INTRINSIC_NAMES]
        + [f"{camera_b.name}.{name}" for name in POSE_NAMES]
    )
    relevant = [lookup[name] for name in relevant_names if name in lookup]
    observation_sigma = data.adjustment.observation_sigma_px
    if observation_sigma is None or not np.isfinite(observation_sigma) or observation_sigma <= 0:
        observation_sigma = data.adjustment.sigma0_px
    if not np.isfinite(observation_sigma) or observation_sigma <= 0:
        return None
    observation_covariance = np.eye(4) * float(observation_sigma ** 2)
    parameter_covariance = _covariance_submatrix(data.adjustment, relevant)
    if parameter_covariance is None:
        return None

    sx, sy, sz = [], [], []
    for distance in distances:
        true_point = np.array([0.0, 0.0, float(distance)])
        intrinsics_a = _intrinsics_from(vector, lookup, camera_a.name)
        intrinsics_b = _intrinsics_from(vector, lookup, camera_b.name)
        pose_b = _pose_from(vector, lookup, camera_b.name)
        point_a = project_point(intrinsics_a, np.zeros(6), true_point)
        point_b = project_point(intrinsics_b, pose_b, true_point)
        if not (
            _projected_in_sensor(intrinsics_a, np.zeros(6), true_point, camera_a.width, camera_a.height)
            and _projected_in_sensor(intrinsics_b, pose_b, true_point, camera_b.width, camera_b.height)
        ):
            sx.append(np.nan)
            sy.append(np.nan)
            sz.append(np.nan)
            continue

        jacobian_observation = np.zeros((3, 4), dtype=np.float64)
        measured = np.concatenate([point_a, point_b])
        for column in range(4):
            step = 1e-4
            plus, minus = measured.copy(), measured.copy()
            plus[column] += step
            minus[column] -= step
            xp = _triangulate(intrinsics_a, intrinsics_b, pose_b, plus[:2], plus[2:])
            xm = _triangulate(intrinsics_a, intrinsics_b, pose_b, minus[:2], minus[2:])
            if xp is None or xm is None:
                return None
            jacobian_observation[:, column] = (xp - xm) / (2.0 * step)

        jacobian_parameters = np.zeros((3, len(relevant)), dtype=np.float64)
        for column, parameter in enumerate(relevant):
            step = _step(vector[parameter])
            plus, minus = vector.copy(), vector.copy()
            plus[parameter] += step
            minus[parameter] -= step
            xp = _triangulate(
                _intrinsics_from(plus, lookup, camera_a.name),
                _intrinsics_from(plus, lookup, camera_b.name),
                _pose_from(plus, lookup, camera_b.name), point_a, point_b,
            )
            xm = _triangulate(
                _intrinsics_from(minus, lookup, camera_a.name),
                _intrinsics_from(minus, lookup, camera_b.name),
                _pose_from(minus, lookup, camera_b.name), point_a, point_b,
            )
            if xp is None or xm is None:
                return None
            jacobian_parameters[:, column] = (xp - xm) / (2.0 * step)

        output_covariance = (
            jacobian_observation @ observation_covariance @ jacobian_observation.T
            + jacobian_parameters @ parameter_covariance @ jacobian_parameters.T
        )
        standard = np.sqrt(np.maximum(np.diag(output_covariance), 0.0))
        sx.append(standard[0])
        sy.append(standard[1])
        sz.append(standard[2])

    return TriangulationUncertaintyCurve(
        camera_a=camera_a.name,
        camera_b=camera_b.name,
        distances_m=np.asarray(distances),
        sigma_x_m=np.asarray(sx),
        sigma_y_m=np.asarray(sy),
        sigma_z_m=np.asarray(sz),
    )
