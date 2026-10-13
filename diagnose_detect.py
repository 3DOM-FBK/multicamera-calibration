"""Trace the exact production CCT detector pipeline for one or more images."""

from __future__ import annotations

import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).parent / "src"))
from cct_detect.detector import CCTDetector


def diagnose(image_path: Path) -> None:
    detector = CCTDetector(n_bits=14)
    image = cv2.imread(str(image_path))
    if image is None:
        raise RuntimeError(f"Could not read {image_path}")
    detections, diagnostics = detector.detect_with_diagnostics(image)
    height, width = image.shape[:2]
    print(f"\n{'=' * 72}")
    print(f"Image: {image_path.name} ({width}x{height})")
    print(f"{'=' * 72}")
    for key, value in sorted(diagnostics.counts.items()):
        print(f"  {key:32s}: {value}")
    print(f"  final IDs: {', '.join(str(det.target_id) for det in detections) or '(none)'}")
    methods = {}
    corrections = []
    for det in detections:
        methods[det.center_method] = methods.get(det.center_method, 0) + 1
        if det.center_correction_px is not None:
            corrections.append(det.center_correction_px)
    print(f"  center methods: {methods}")
    if corrections:
        magnitudes = [float((x * x + y * y) ** 0.5) for x, y in corrections]
        median = sorted(magnitudes)[len(magnitudes) // 2]
        print(f"  center correction magnitude: median={median:.3f} px, max={max(magnitudes):.3f} px")
    if diagnostics.rejected:
        print("  rejected candidate examples:")
        for item in diagnostics.rejected[:12]:
            print(f"    {item}")


if __name__ == "__main__":
    base = Path(__file__).parent.parent / "sample"
    for image_path in sorted(base.glob("*.JPG")):
        if "_det" not in image_path.stem:
            diagnose(image_path)
