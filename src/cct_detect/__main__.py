"""CLI for CCT target detection."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2

from .detector import CCTDetector, Detection


def save_detections(
    image_name: str,
    detections: list[Detection],
    text_path: Path,
) -> None:
    lines = ["image_name\ttarget_id\tx_image\ty_image"]
    for det in detections:
        lines.append(
            f"{image_name}\t{det.target_id}\t{det.center[0]:.3f}\t{det.center[1]:.3f}"
        )
    text_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def process_image(
    image_path: Path, output_dir: Path, detector: CCTDetector
) -> dict:
    image_bgr = cv2.imread(str(image_path))
    if image_bgr is None:
        raise RuntimeError(f"Cannot read {image_path}")

    detections = detector.detect(image_bgr)

    output_dir.mkdir(parents=True, exist_ok=True)
    stem = image_path.stem

    # Save annotated image
    annotated = detector.annotate(image_bgr, detections)
    ann_path = output_dir / f"{stem}_det.jpg"
    cv2.imwrite(str(ann_path), annotated)

    # Save text detections
    txt_path = output_dir / f"{stem}_det.txt"
    save_detections(image_path.name, detections, txt_path)

    return {
        "image": str(image_path),
        "detections": len(detections),
        "annotated": str(ann_path),
        "text": str(txt_path),
        "target_ids": [d.target_id for d in detections],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="CCT target detector")
    parser.add_argument("--image", type=Path, required=True, help="Image to process")
    parser.add_argument(
        "--output-dir", type=Path, default=Path("cct_output"), help="Output directory"
    )
    parser.add_argument("--valid-id-min", type=int, default=None,
                        help="Minimum valid target ID (inclusive).")
    parser.add_argument("--valid-id-max", type=int, default=None,
                        help="Maximum valid target ID (inclusive).")
    args = parser.parse_args()

    valid_id_range = None
    if args.valid_id_min is not None or args.valid_id_max is not None:
        lo = args.valid_id_min if args.valid_id_min is not None else 0
        hi = args.valid_id_max if args.valid_id_max is not None else (1 << 14) - 1
        valid_id_range = (lo, hi)

    detector = CCTDetector(valid_id_range=valid_id_range)
    result = process_image(args.image, args.output_dir, detector)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
