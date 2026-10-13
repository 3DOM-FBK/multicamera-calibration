from __future__ import annotations

from pathlib import Path

import cv2

from cct_detect_better.detector import CCTDetector


def main() -> None:
    image_path = Path("..") / "sample" / "1.JPG"
    image = cv2.imread(str(image_path))
    detector = CCTDetector(valid_ids=set(range(130, 421)))
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    candidates = detector._find_candidates(gray)
    candidates.sort(key=lambda ellipse: max(ellipse[1]), reverse=True)

    counts = {"candidates": len(candidates), "rectified": 0, "validated": 0, "decoded": 0}
    anchor_scores: list[float] = []
    spreads: list[float] = []
    for ellipse in candidates[:120]:
        patch = detector._rectify_patch(gray, ellipse, out_size=220)
        if patch is None:
            continue
        counts["rectified"] += 1
        if not detector._validate_rectified(patch):
            continue
        counts["validated"] += 1
        strip = detector._sample_polar_strip(patch)
        if strip is None:
            continue
        n_ang = strip.shape[1]
        step = n_ang / detector.n_bits
        for phase in range(max(1, int(round(step)))):
            means = detector._sector_means_for_phase(strip, phase)
            if means is None:
                continue
            _, spread = detector._normalize_sector_means(means)
            sector_white, _ = detector._normalize_sector_means(means)
            anchor_scores.append(float(sector_white[list(detector.anchor_bits)].mean()))
            spreads.append(float(spread))
        result = detector._decode_patch(patch)
        if result is not None:
            counts["decoded"] += 1
            print("decoded", result.target_id, result.confidence)

    print(counts)
    if anchor_scores:
        anchor_scores.sort()
        spreads.sort()
        print("anchor", anchor_scores[:5], anchor_scores[-5:])
        print("spread", spreads[:5], spreads[-5:])


if __name__ == "__main__":
    main()
