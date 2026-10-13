"""Main entry point for CCT camera calibration.

Usage:
    python run_cct_calibration.py [OPTIONS]

Examples:
    # Full calibration (both cameras by default)
    python run_cct_calibration.py --data-root data --output-dir calibration_output

    # Detection only
    python run_cct_calibration.py --detect-only --data-root data --output-dir calibration_output

    # Single camera
    python run_cct_calibration.py --camera camera_0 --data-root data

Run with --help for full option list.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from cct_calibration.run import main

if __name__ == "__main__":
    raise SystemExit(main())
