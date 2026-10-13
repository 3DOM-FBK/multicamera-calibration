"""Rich calibration reporting: statistics, plots and PDF generation.

Public API:
    build_multi_camera_report_data()  -- extract per-observation records from a
                                         solved ``MultiCameraState``.
    generate_pdf_report()             -- render a polished PDF report.

The module is deliberately independent from the optimisation code: it only
consumes plain data structures (numpy arrays, dicts, dataclasses) so it can be
reused by the single-camera pipeline as well.
"""

from __future__ import annotations

from cct_calibration.reporting.records import (
    CameraInfo,
    ObservationRecord,
    ReportData,
    RigInfo,
    build_multi_camera_report_data,
)
from cct_calibration.reporting.pdf import generate_pdf_report

__all__ = [
    "CameraInfo",
    "ObservationRecord",
    "ReportData",
    "RigInfo",
    "build_multi_camera_report_data",
    "generate_pdf_report",
]
