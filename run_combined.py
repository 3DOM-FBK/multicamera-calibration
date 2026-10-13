"""Combined multi-camera CCT calibration pipeline.

Usage:
    python run_combined.py --image-root images [OPTIONS]

Examples:
    # Calibrate 2 cameras from images/ folder
    python run_combined.py --image-root images --output-dir combined_output

    # With known baseline for scale
    python run_combined.py --image-root images --known-baseline 0.263

Run with --help for full option list.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from cct_calibration.run_combined import main

if __name__ == "__main__":
    raise SystemExit(main())
