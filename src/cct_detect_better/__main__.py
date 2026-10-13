"""CLI for the alternative CCT detector."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from .detector import CCTDetector, Detection, DetectionDiagnostics


def parse_valid_ids_file(path: Path) -> set[int]:
    tokens = path.read_text(encoding="utf-8").replace(",", " ").split()
    valid_ids = {int(token) for token in tokens}
    if not valid_ids:
        raise ValueError(f"No valid IDs found in {path}")
    return valid_ids


def save_detections(image_name: str, detections: list[Detection], text_path: Path) -> None:
    lines = ["image_name\ttarget_id\tx_image\ty_image"]
    for det in detections:
        lines.append(
            f"{image_name}\t{det.target_id}\t{det.center[0]:.3f}\t{det.center[1]:.3f}"
        )
    text_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def save_stage_overlay(
    image_bgr: np.ndarray,
    stage_name: str,
    points: list[dict],
    output_path: Path,
) -> None:
    colors = {
        "candidates": (0, 255, 255),
        "reject_rectify": (0, 128, 255),
        "validated": (0, 255, 0),
        "reject_validate": (0, 0, 255),
        "decoded": (255, 255, 0),
        "reject_decode": (255, 0, 255),
        "spatial_dedup": (255, 128, 0),
        "final": (0, 255, 0),
    }
    color = colors.get(stage_name, (255, 255, 255))
    overlay = image_bgr.copy()
    for point in points:
        cx = int(round(float(point["x"])))
        cy = int(round(float(point["y"])))
        cv2.circle(overlay, (cx, cy), 6, color, 2)
        label = str(point.get("label", ""))
        if label:
            cv2.putText(
                overlay,
                label,
                (cx + 8, cy - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                color,
                1,
                cv2.LINE_AA,
            )
    title = f"{stage_name}: {len(points)}"
    cv2.putText(
        overlay,
        title,
        (24, 36),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        color,
        2,
        cv2.LINE_AA,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), overlay)


def save_debug_overlays(
    image_bgr: np.ndarray,
    output_dir: Path,
    stem: str,
    diagnostics: DetectionDiagnostics,
) -> dict[str, str]:
    overlay_paths: dict[str, str] = {}
    for stage_name, points in diagnostics.stage_points.items():
        overlay_path = output_dir / f"{stem}_better_{stage_name}.jpg"
        save_stage_overlay(image_bgr, stage_name, points, overlay_path)
        overlay_paths[stage_name] = str(overlay_path)

    summary_path = output_dir / f"{stem}_better_debug.json"
    summary_path.write_text(
        json.dumps(
            {
                "counts": diagnostics.counts,
                "stages": diagnostics.stage_points,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    overlay_paths["summary"] = str(summary_path)
    return overlay_paths


def process_image(
    image_path: Path,
    output_dir: Path,
    detector: CCTDetector,
    debug_overlays: bool = False,
) -> dict:
    image_bgr = cv2.imread(str(image_path))
    if image_bgr is None:
        raise RuntimeError(f"Cannot read {image_path}")

    detections, diagnostics = detector.detect_with_diagnostics(image_bgr)

    output_dir.mkdir(parents=True, exist_ok=True)
    stem = image_path.stem

    annotated = detector.annotate(image_bgr, detections)
    ann_path = output_dir / f"{stem}_better_det.jpg"
    cv2.imwrite(str(ann_path), annotated)

    txt_path = output_dir / f"{stem}_better_det.txt"
    save_detections(image_path.name, detections, txt_path)

    debug_files: dict[str, str] = {}
    if debug_overlays:
        debug_files = save_debug_overlays(image_bgr, output_dir, stem, diagnostics)

    return {
        "image": str(image_path),
        "detections": len(detections),
        "annotated": str(ann_path),
        "text": str(txt_path),
        "target_ids": [d.target_id for d in detections],
        "stage_counts": diagnostics.counts,
        "debug_files": debug_files,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Alternative CCT target detector")
    parser.add_argument("--image", type=Path, required=True, help="Image to process")
    parser.add_argument(
        "--output-dir", type=Path, default=Path("cct_output_better"), help="Output directory"
    )
    parser.add_argument("--valid-id-min", type=int, default=None,
                        help="Minimum valid target ID (inclusive).")
    parser.add_argument("--valid-id-max", type=int, default=None,
                        help="Maximum valid target ID (inclusive).")
    parser.add_argument("--valid-ids-file", type=Path, default=None,
                        help="Optional text file with whitespace- or comma-separated valid target IDs.")
    parser.add_argument("--max-id-hamming-distance", type=int, default=1,
                        help="If a valid codebook is provided, snap to nearest valid ID up to this Hamming distance.")
    parser.add_argument("--debug-overlays", action="store_true",
                        help="Save stage-by-stage diagnostic overlays and JSON summaries.")
    args = parser.parse_args()

    valid_ids: set[int] | None = None
    if args.valid_ids_file is not None:
        valid_ids = parse_valid_ids_file(args.valid_ids_file)
    elif args.valid_id_min is not None or args.valid_id_max is not None:
        lo = args.valid_id_min if args.valid_id_min is not None else 0
        hi = args.valid_id_max if args.valid_id_max is not None else (1 << 14) - 1
        valid_ids = set(range(lo, hi + 1))

    detector = CCTDetector(
        valid_ids=valid_ids,
        max_id_hamming_distance=args.max_id_hamming_distance,
    )
    result = process_image(args.image, args.output_dir, detector, debug_overlays=args.debug_overlays)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
