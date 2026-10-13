"""Developer diagnostic: detection recall and suspect detections.

After calibration every reference target is projected into every posed image.
A target is *expected* when its whole code ring projects inside the image, it
lies in front of the camera and, when a local surface normal can be estimated
from neighbouring reference targets, it is not seen too obliquely.  Expected
targets without a matching detection are misses; each miss is attributed to
the detector stage that rejected the nearest candidate.  Detections far from
the projection of their ID are reported as suspects (a misidentification, or
a poor pose for that frame).

The projections come from the calibration being evaluated, so this is an
internal consistency tool for detector development, not independent ground
truth.  Occlusions are not modelled.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Set

import cv2
import numpy as np

from concurrent.futures import ProcessPoolExecutor

from cct_calibration.run import (
    _detector_for,
    _init_detection_worker,
    ImageDetections,
    project_point,
    rotation_matrix_from_pose,
)


def _target_normals(points: Dict[int, np.ndarray], neighbours: int = 6) -> Dict[int, np.ndarray]:
    """Unoriented local plane normals; omitted where neighbours are not planar."""
    ids = sorted(points)
    if len(ids) < neighbours + 1:
        return {}
    xyz = np.array([points[target_id] for target_id in ids], dtype=np.float64)
    normals: Dict[int, np.ndarray] = {}
    for index, target_id in enumerate(ids):
        distances = np.linalg.norm(xyz - xyz[index], axis=1)
        local = xyz[np.argsort(distances)[:neighbours + 1]]
        _, singular, vt = np.linalg.svd(local - local.mean(axis=0), full_matrices=False)
        if singular[1] <= 1e-12 or singular[2] / singular[1] > 0.1:
            continue  # e.g. a target at the junction of two walls
        normals[target_id] = vt[2]
    return normals


def _attribute_image_misses(path: Path, queries: list, codebook: Set[int] | None) -> list:
    """Detector stage for each missed target of one image (process-safe).

    ``queries`` holds (projected_uv, dot_radius_px, target_id) per miss; the
    result holds (stage, decoded_ids or None) in the same order.
    """
    image = cv2.imread(str(path))
    if image is None:
        return [("image_unreadable", None)] * len(queries)
    detector = _detector_for(codebook)
    detections, diagnostics = detector.detect_with_diagnostics(image)
    outcomes: list = [None] * len(queries)
    unresolved = []
    for index, (projected_uv, dot_radius_px, target_id) in enumerate(queries):
        uv = np.asarray(projected_uv)
        radius = max(6.0, dot_radius_px)
        nearby = [d for d in detections if np.linalg.norm(np.asarray(d.center) - uv) <= max(10.0, 2.0 * radius)]
        if any(d.target_id == target_id for d in nearby):
            outcomes[index] = ("detected_then_dropped_by_pipeline", None)
            continue
        if nearby:
            outcomes[index] = ("decoded_as_other_id", sorted({int(d.target_id) for d in nearby}))
            continue
        records = [r for r in diagnostics.rejected
                   if np.hypot(r["x"] - uv[0], r["y"] - uv[1]) <= radius]
        if records:
            closest = min(records, key=lambda r: np.hypot(r["x"] - uv[0], r["y"] - uv[1]))
            outcomes[index] = (str(closest["stage"]), None)
        else:
            unresolved.append(index)
    if unresolved:
        gray = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(
            cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))
        reasons = detector.contour_gate_reasons(
            gray, np.array([queries[index][0] for index in unresolved]),
            np.array([max(6.0, queries[index][1]) for index in unresolved]),
        )
        for index, reason in zip(unresolved, reasons):
            outcomes[index] = (f"candidate_gate:{reason}", None)
    return outcomes


def evaluate_detections(
    state,
    detections_by_camera: Dict[str, List[ImageDetections]],
    checkpoint_images: Dict[str, List[ImageDetections]],
    known_targets3d: Dict[int, np.ndarray],
    codebook: Set[int] | None,
    output_path: Path,
    incidence_limit_deg: float = 75.0,
    workers: int = 1,
    frame_stems: Set[str] | None = None,
) -> dict:
    # Deferred import: run_combined imports this module lazily.
    from cct_calibration.run_combined import compose_poses

    # Merge calibration and withheld-checkpoint detections per image.
    images: Dict[tuple[str, str], dict] = {}
    for source in (detections_by_camera, checkpoint_images or {}):
        for camera, records in source.items():
            for record in records:
                key = (camera, record.image_path.stem)
                entry = images.setdefault(key, {
                    "path": record.image_path, "width": record.width,
                    "height": record.height, "detections": {}, "ellipses": {},
                })
                entry["detections"].update(record.detections)
                entry["ellipses"].update(record.refinement_ellipses)

    poses: Dict[tuple[str, str], np.ndarray] = {}
    for camera, stem in images:
        if frame_stems is not None and stem not in frame_stems:
            continue  # not part of the final solution (e.g. dropped frame)
        rig_pose = state.rig_poses.get(stem)
        if rig_pose is not None and camera in state.relative_poses:
            poses[(camera, stem)] = compose_poses(rig_pose, state.relative_poses[camera])

    def camera_frame(camera: str, pose: np.ndarray, point: np.ndarray):
        rotation = rotation_matrix_from_pose(pose)
        local = rotation @ point + pose[3:]
        center = -rotation.T @ pose[3:]
        return local, center

    # Physical dot radius per target from detected ellipse sizes.
    radius_samples: Dict[int, list[float]] = defaultdict(list)
    centres_seen: Dict[int, list[np.ndarray]] = defaultdict(list)
    all_centres: list[np.ndarray] = []
    for (camera, stem), entry in images.items():
        pose = poses.get((camera, stem))
        if pose is None:
            continue
        intrinsics = state.camera_intrinsics[camera]
        focal = 0.5 * (intrinsics[0] + intrinsics[1])
        all_centres.append(-rotation_matrix_from_pose(pose).T @ pose[3:])
        for target_id, ellipse in entry["ellipses"].items():
            if target_id not in known_targets3d or target_id not in entry["detections"]:
                continue
            local, center = camera_frame(camera, pose, known_targets3d[target_id])
            if local[2] > 0:
                radius_samples[target_id].append(0.5 * max(ellipse[1]) * float(local[2]) / focal)
                centres_seen[target_id].append(center)
    per_target_radius = {t: float(np.median(v)) for t, v in radius_samples.items() if v}
    default_radius = float(np.median(list(per_target_radius.values()))) if per_target_radius else None
    if default_radius is None:
        result = {"status": "unavailable", "reason": "no detected target sizes to scale projections"}
        output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
        return result

    normals = _target_normals(known_targets3d)
    mean_centre_all = np.mean(all_centres, axis=0) if all_centres else np.zeros(3)
    for target_id, normal in list(normals.items()):
        reference = np.mean(centres_seen[target_id], axis=0) if centres_seen.get(target_id) else mean_centre_all
        if float(normal @ (reference - known_targets3d[target_id])) < 0:
            normals[target_id] = -normal

    misses: list[dict] = []
    suspects: list[dict] = []
    unreferenced = 0
    summary_by_camera: Dict[str, Counter] = defaultdict(Counter)
    per_image_recall: Dict[str, list[float]] = defaultdict(list)
    for (camera, stem), entry in sorted(images.items()):
        pose = poses.get((camera, stem))
        if pose is None:
            continue
        intrinsics = state.camera_intrinsics[camera]
        focal = 0.5 * (intrinsics[0] + intrinsics[1])
        width, height = entry["width"], entry["height"]
        projections: Dict[int, dict] = {}
        for target_id, point in known_targets3d.items():
            local, center = camera_frame(camera, pose, point)
            depth = float(local[2])
            if depth <= 1e-6:
                continue
            uv = project_point(intrinsics, pose, point)
            dot_radius = focal * per_target_radius.get(target_id, default_radius) / depth
            incidence = None
            if target_id in normals:
                ray = center - point
                cosine = float(normals[target_id] @ ray / max(np.linalg.norm(ray), 1e-12))
                incidence = float(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))))
            outer = 3.0 * dot_radius + 2.0
            inside = outer <= uv[0] <= width - 1 - outer and outer <= uv[1] <= height - 1 - outer
            facing = incidence is None or incidence <= incidence_limit_deg
            projections[target_id] = {
                "uv": uv, "depth": depth, "dot_radius_px": dot_radius,
                "incidence_deg": incidence, "expected": bool(inside and facing),
                "in_image": bool(0 <= uv[0] < width and 0 <= uv[1] < height),
            }

        matched: set[int] = set()
        for target_id, measured in entry["detections"].items():
            projection = projections.get(target_id)
            if target_id not in known_targets3d:
                unreferenced += 1
                continue
            if projection is None or not projection["in_image"]:
                suspects.append({"camera": camera, "image": entry["path"].name, "target_id": int(target_id),
                                 "detected_uv": [float(v) for v in measured], "reason": "id projects outside the image"})
                continue
            error = float(np.linalg.norm(np.asarray(measured) - projection["uv"]))
            if error <= max(10.0, 2.0 * projection["dot_radius_px"]):
                matched.add(target_id)
            else:
                suspects.append({"camera": camera, "image": entry["path"].name, "target_id": int(target_id),
                                 "detected_uv": [float(v) for v in measured],
                                 "projected_uv": [float(v) for v in projection["uv"]],
                                 "distance_px": error, "reason": "far from the projection of its id"})

        expected = {t for t, p in projections.items() if p["expected"]}
        summary_by_camera[camera]["expected"] += len(expected)
        summary_by_camera[camera]["matched_expected"] += len(expected & matched)
        summary_by_camera[camera]["matched_not_expected"] += len(matched - expected)
        if expected:
            per_image_recall[camera].append(len(expected & matched) / len(expected))
        for target_id in sorted(expected - matched):
            projection = projections[target_id]
            misses.append({
                "camera": camera, "image": entry["path"].name, "target_id": int(target_id),
                "projected_uv": [float(v) for v in projection["uv"]],
                "depth_m": projection["depth"], "dot_radius_px": projection["dot_radius_px"],
                "incidence_deg": projection["incidence_deg"], "stage": None,
            })

    # Attribute misses by re-running the detector on the affected images.
    by_image: Dict[tuple[str, str], list[dict]] = defaultdict(list)
    for miss in misses:
        by_image[(miss["camera"], miss["image"])].append(miss)
    print(f"  detection evaluation: attributing {len(misses)} misses in {len(by_image)} images", flush=True)
    jobs = []
    for (camera, image_name), items in sorted(by_image.items()):
        path = next(e["path"] for (c, _), e in images.items() if c == camera and e["path"].name == image_name)
        jobs.append((path, [(item["projected_uv"], item["dot_radius_px"], item["target_id"]) for item in items]))
    workers = max(1, min(int(workers), len(jobs)))
    pool = ProcessPoolExecutor(max_workers=workers, initializer=_init_detection_worker) if workers > 1 else None
    try:
        if pool is not None:
            results = pool.map(_attribute_image_misses, [job[0] for job in jobs],
                               [job[1] for job in jobs], [codebook] * len(jobs))
        else:
            results = (_attribute_image_misses(path, queries, codebook) for path, queries in jobs)
        for number, (((camera, image_name), items), outcomes) in enumerate(
                zip(sorted(by_image.items()), results), start=1):
            for item, (stage, decoded_ids) in zip(items, outcomes):
                item["stage"] = stage
                if decoded_ids is not None:
                    item["decoded_ids"] = decoded_ids
            if number % 25 == 0 or number == len(by_image):
                print(f"    attributed {number}/{len(by_image)} images", flush=True)
    finally:
        if pool is not None:
            pool.shutdown()

    stages_by_camera: Dict[str, Counter] = defaultdict(Counter)
    for miss in misses:
        stages_by_camera[miss["camera"]][miss["stage"]] += 1
    cameras = {}
    for camera, counts in sorted(summary_by_camera.items()):
        recalls = np.asarray(per_image_recall[camera], dtype=np.float64)
        cameras[camera] = {
            "expected": int(counts["expected"]),
            "matched_expected": int(counts["matched_expected"]),
            "recall": float(counts["matched_expected"] / counts["expected"]) if counts["expected"] else None,
            "matched_not_expected": int(counts["matched_not_expected"]),
            "per_image_recall_min": float(recalls.min()) if recalls.size else None,
            "per_image_recall_median": float(np.median(recalls)) if recalls.size else None,
            "misses_by_stage": dict(stages_by_camera[camera].most_common()),
            "suspects": sum(1 for s in suspects if s["camera"] == camera),
        }
    result = {
        "note": ("Internal consistency diagnostic: projections use the calibration being "
                 "evaluated; occlusions are not modelled."),
        "decoder": "codebook" if codebook is not None else "generic",
        "incidence_limit_deg": incidence_limit_deg,
        "targets_with_normal": len(normals),
        "default_dot_radius_m": default_radius,
        "cameras": cameras,
        "unreferenced_detections": unreferenced,
        "misses": misses,
        "suspects": suspects,
    }
    output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    for camera, values in cameras.items():
        recall = values["recall"]
        print(
            f"  {camera}: recall {values['matched_expected']}/{values['expected']}"
            f" = {recall:.3f}" if recall is not None else f"  {camera}: no expected targets",
            flush=True,
        )
        print(f"    misses by stage: {values['misses_by_stage']}; suspects={values['suspects']}", flush=True)
    return result
