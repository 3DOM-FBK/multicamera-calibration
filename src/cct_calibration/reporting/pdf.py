"""PDF report assembly for calibration results (reportlab platypus).

Produces a clean, factual A4 document aimed at photogrammetry experts:

* Title page with key figures and run information
* Summary of residuals, coverage and solution redundancy (facts only)
* Signed/standardized residual analysis and outlier diagnostics
* Marginal parameter covariance, correlation and projection uncertainty
* Per-camera spatial residual and density diagnostics
* Network-strength and local-reliability analysis
* Pairwise rig geometry and internal stereo diagnostics

The layout is intentionally minimal: no decorative bands, grades or scores.
"""

from __future__ import annotations

import io
import re
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.lib.utils import ImageReader
from reportlab.platypus import (
    BaseDocTemplate,
    Frame,
    HRFlowable,
    Image,
    KeepTogether,
    NextPageTemplate,
    PageBreak,
    PageTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
)

from cct_calibration.reporting.plots import (
    fig_binned_signed_residuals,
    fig_camera_comparison,
    fig_camera_target_connectivity,
    fig_checkpoint_accuracy,
    fig_convergence,
    fig_coverage_map,
    fig_distortion_profile,
    fig_error_distribution,
    fig_frame_target_residuals,
    fig_outlier_diagnostics,
    fig_parameter_correlation,
    fig_pose_and_viewing_coverage,
    fig_projection_uncertainty,
    fig_radial_error,
    fig_residual_component_heatmaps,
    fig_rig_pair_quality,
    fig_rig_diagram,
    fig_scene_geometry,
    fig_spatial_heatmap,
    fig_standardized_residuals,
    fig_target_errors,
    fig_track_richness,
    fig_uncertainty_vs_distance,
    figure_to_bytes,
)
from cct_calibration.reporting.records import ReportData
from cct_calibration.reporting.statistics import (
    ErrorStats,
    compute_rig_pair_stats,
    compute_richness,
    per_camera_stats,
    signed_residual_stats,
    weakest_frames_and_targets,
)

# ---------------------------------------------------------------------------
# Palette & styles
# ---------------------------------------------------------------------------

INK = colors.HexColor("#1A1A1A")
MUTED = colors.HexColor("#6B7280")
HAIRLINE = colors.HexColor("#C9CED6")
THEAD = colors.HexColor("#EEF0F3")
ROW_ALT = colors.HexColor("#F8F9FA")

PAGE_W, PAGE_H = A4
MARGIN_L = MARGIN_R = 17 * mm
MARGIN_T = 18 * mm
MARGIN_B = 16 * mm
CONTENT_W = PAGE_W - MARGIN_L - MARGIN_R

_DOC_META: Dict[str, str] = {"title": "Calibration Report", "cameras": "", "date": ""}


def _styles() -> Dict[str, ParagraphStyle]:
    ss = getSampleStyleSheet()
    return {
        "kicker": ParagraphStyle("kicker", parent=ss["Normal"], fontSize=9, leading=12,
                                 textColor=MUTED),
        "doc_title": ParagraphStyle("doc_title", parent=ss["Title"], fontSize=24,
                                    leading=29, textColor=INK, alignment=TA_LEFT),
        "doc_subtitle": ParagraphStyle("doc_subtitle", parent=ss["Normal"], fontSize=11,
                                       leading=15, textColor=MUTED),
        "h1": ParagraphStyle("h1", parent=ss["Heading1"], fontSize=13.5, leading=17,
                             textColor=INK, spaceBefore=0, spaceAfter=2),
        "h2": ParagraphStyle("h2", parent=ss["Heading2"], fontSize=10.5, leading=14,
                             textColor=INK, spaceBefore=8, spaceAfter=3),
        "body": ParagraphStyle("body", parent=ss["BodyText"], fontSize=9, leading=13),
        "small": ParagraphStyle("small", parent=ss["BodyText"], fontSize=7.5, leading=10.5,
                                textColor=MUTED),
        "kpi_value": ParagraphStyle("kpi_value", parent=ss["BodyText"], fontSize=15,
                                    leading=18, textColor=INK, alignment=TA_CENTER,
                                    fontName="Helvetica-Bold"),
        "kpi_label": ParagraphStyle("kpi_label", parent=ss["BodyText"], fontSize=7,
                                    leading=9, textColor=MUTED, alignment=TA_CENTER),
        "caption": ParagraphStyle("caption", parent=ss["BodyText"], fontSize=7.5,
                                  leading=10, textColor=MUTED, alignment=TA_CENTER),
        "cell": ParagraphStyle("cell", parent=ss["BodyText"], fontSize=8, leading=10.5),
        "cell_head": ParagraphStyle("cell_head", parent=ss["BodyText"], fontSize=8,
                                    leading=10.5, textColor=INK,
                                    fontName="Helvetica-Bold"),
    }


# ---------------------------------------------------------------------------
# Page decoration callbacks (minimal)
# ---------------------------------------------------------------------------

def _later_page_decor(canvas, doc) -> None:
    canvas.saveState()
    y_top = PAGE_H - 12 * mm
    canvas.setStrokeColor(HAIRLINE)
    canvas.setLineWidth(0.5)
    canvas.line(MARGIN_L, y_top, PAGE_W - MARGIN_R, y_top)
    canvas.setFillColor(MUTED)
    canvas.setFont("Helvetica", 7.5)
    canvas.drawString(MARGIN_L, y_top + 3.2 * mm, _DOC_META["title"])
    canvas.drawRightString(PAGE_W - MARGIN_R, y_top + 3.2 * mm, _DOC_META["cameras"])

    canvas.line(MARGIN_L, 11 * mm, PAGE_W - MARGIN_R, 11 * mm)
    canvas.drawString(MARGIN_L, 7.4 * mm, f"Generated {_DOC_META['date']}")
    canvas.drawRightString(PAGE_W - MARGIN_R, 7.4 * mm, f"Page {doc.page}")
    canvas.restoreState()


def _title_page_decor(canvas, doc) -> None:
    """No decoration on the title page."""
    return None


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

def _section(number: str, title: str, styles: Dict[str, ParagraphStyle]) -> List:
    head = Paragraph(f"{number}&nbsp;&nbsp;·&nbsp;&nbsp;{title}", styles["h1"])
    rule = HRFlowable(width="100%", thickness=0.7, color=INK, spaceBefore=1, spaceAfter=6)
    return [head, rule]


def _figure_image(fig, width: float = CONTENT_W, *, export_path: Optional[Path] = None) -> Image:
    """Convert a matplotlib figure to a centred, width-scaled image flowable."""
    if export_path is not None:
        # Render from the original figure, not the lower-resolution PDF image.
        fig.savefig(export_path, format="png", dpi=600, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
    fig_bytes = figure_to_bytes(fig)
    buf = io.BytesIO(fig_bytes)
    reader = ImageReader(buf)
    iw, ih = reader.getSize()
    height = width * ih / iw
    buf.seek(0)
    img = Image(buf, width=width, height=height)
    img.hAlign = "CENTER"
    return img


def _data_table(header: List[str], rows: List[List[str]],
                col_widths: Optional[List[float]] = None,
                styles: Dict[str, ParagraphStyle] = None) -> Table:
    head = [Paragraph(h, styles["cell_head"]) for h in header]
    body = [[Paragraph(str(c), styles["cell"]) for c in row] for row in rows]
    t = Table([head] + body, colWidths=col_widths, repeatRows=1)
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), THEAD),
        ("LINEBELOW", (0, 0), (-1, 0), 0.7, INK),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, ROW_ALT]),
        ("LINEBELOW", (0, 1), (-1, -1), 0.4, HAIRLINE),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 2.5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 2.5),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
    ]))
    return t


def _kpi_cards(items: List[tuple], styles: Dict[str, ParagraphStyle]) -> Table:
    n = len(items)
    gap = 3 * mm
    cw = (CONTENT_W - gap * (n - 1)) / n
    cells, widths = [], []
    for i, (value, label) in enumerate(items):
        inner = Table(
            [[Paragraph(value, styles["kpi_value"])],
             [Paragraph(label.upper(), styles["kpi_label"])]],
            colWidths=[cw],
        )
        inner.setStyle(TableStyle([
            ("BOX", (0, 0), (-1, -1), 0.6, HAIRLINE),
            ("TOPPADDING", (0, 0), (-1, 0), 7),
            ("BOTTOMPADDING", (0, 1), (-1, 1), 7),
            ("TOPPADDING", (0, 1), (-1, 1), 1),
            ("BOTTOMPADDING", (0, 0), (-1, 0), 1),
        ]))
        cells.append(inner)
        widths.append(cw)
        if i < n - 1:
            cells.append("")
            widths.append(gap)
    outer = Table([cells], colWidths=widths)
    outer.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 0),
        ("RIGHTPADDING", (0, 0), (-1, -1), 0),
        ("TOPPADDING", (0, 0), (-1, -1), 0),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 0),
    ]))
    return outer


def _parameter_rows(data: ReportData, prefix: str) -> List[List[str]]:
    adjustment = data.adjustment
    if adjustment is None:
        return []
    rows: List[List[str]] = []
    diagonal = np.diag(adjustment.covariance)
    for index, name in enumerate(adjustment.parameter_names):
        if not name.startswith(prefix + "."):
            continue
        finite = np.isfinite(diagonal[index]) and diagonal[index] >= 0
        standard = float(np.sqrt(diagonal[index])) if finite else None
        ci95 = 1.96 * standard if standard is not None else None
        component = name.split(".", 1)[1]
        if component in {"fx", "fy", "cx", "cy"}:
            unit = "px"
        elif component in {"rx", "ry", "rz"}:
            unit = "rad"
        elif component in {"tx", "ty", "tz"}:
            unit = "m"
        else:
            unit = "—"
        rows.append([
            component, unit,
            f"{adjustment.parameter_initial[index]:.8g}",
            f"{adjustment.parameter_final[index]:.8g}",
            f"{standard:.3g}" if standard is not None else "—",
            f"±{ci95:.3g}" if ci95 is not None else "—",
            adjustment.parameter_status[index],
        ])
    return rows


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def generate_pdf_report(
    data: ReportData, out_path: Path, *, plots_dir: Optional[Path] = None,
) -> Path:
    """Build the PDF, optionally exporting every embedded figure at 600 dpi."""
    if plots_dir is not None:
        plots_dir = Path(plots_dir)
        plots_dir.mkdir(parents=True, exist_ok=True)
    plot_count = 0

    def _img(fig, width: float = CONTENT_W) -> Image:
        nonlocal plot_count
        export_path = None
        if plots_dir is not None:
            plot_count += 1
            title = fig._suptitle.get_text() if fig._suptitle is not None else ""
            if not title:
                title = next((axis.get_title() for axis in fig.axes if axis.get_title()), "plot")
            slug = re.sub(r"[^a-z0-9]+", "_", title.lower()).strip("_")[:100] or "plot"
            export_path = plots_dir / f"{plot_count:02d}_{slug}.png"
        return _figure_image(fig, width, export_path=export_path)

    S = _styles()
    overall_err = ErrorStats.from_values([r.err_px for r in data.records])
    obj_vals = [r.err_obj_m * 1000.0 for r in data.records if r.err_obj_m is not None]
    overall_obj = ErrorStats.from_values(obj_vals)
    checkpoint_summary = (
        data.checkpoint_quality.get("summary", {}) if data.checkpoint_quality else {}
    )
    checkpoint_observations = (
        data.checkpoint_quality.get("reprojection_observations", [])
        if data.checkpoint_quality else []
    )
    checkpoint_reproj = ErrorStats.from_values(
        [item["error_px"] for item in checkpoint_observations]
    )
    checkpoint_obj = ErrorStats.from_values(
        [item["object_error_m"] * 1000.0 for item in checkpoint_observations]
    )
    cam_stats = per_camera_stats(data)
    signed = signed_residual_stats(data.records)
    n_fixed = len(data.rig.fixed_point_ids) if data.rig else 0
    rich = compute_richness(data.records, len(data.cameras), n_fixed)
    weak_frames, weak_targets = weakest_frames_and_targets(data.records)
    rig_pairs = compute_rig_pair_stats(data)

    _DOC_META.update({
        "title": f"{'Multi-camera' if data.mode == 'multi-camera' else 'Single-camera'} "
                 f"CCT calibration report",
        "cameras": " · ".join(c.name for c in data.cameras),
        "date": data.generated_at.strftime("%Y-%m-%d %H:%M"),
    })

    doc = BaseDocTemplate(
        str(out_path), pagesize=A4,
        leftMargin=MARGIN_L, rightMargin=MARGIN_R,
        topMargin=MARGIN_T, bottomMargin=MARGIN_B,
        title=_DOC_META["title"], author="CCT Calibration pipeline",
    )
    frame_title = Frame(MARGIN_L, MARGIN_B, CONTENT_W, PAGE_H - MARGIN_T - MARGIN_B, id="ft")
    frame_later = Frame(MARGIN_L, MARGIN_B, CONTENT_W,
                        PAGE_H - MARGIN_T - 16 * mm - MARGIN_B, id="fl")
    doc.addPageTemplates([
        PageTemplate(id="Title", frames=[frame_title], onPage=_title_page_decor),
        PageTemplate(id="Later", frames=[frame_later], onPage=_later_page_decor),
    ])

    story: List = []

    # ── Title page ────────────────────────────────────────────────────────
    story.append(NextPageTemplate("Later"))
    story.append(Spacer(1, 22 * mm))
    story.append(Paragraph("CCT CALIBRATION · "
                           + ("MULTI-CAMERA RIG" if data.mode == "multi-camera"
                              else "SINGLE CAMERA"), S["kicker"]))
    story.append(Spacer(1, 2 * mm))
    story.append(Paragraph("Calibration report", S["doc_title"]))
    story.append(Spacer(1, 2 * mm))
    story.append(Paragraph(
        f"{len(data.cameras)} camera(s): <b>"
        f"{', '.join(c.name for c in data.cameras)}</b>", S["doc_subtitle"]))
    if data.checkpoint_quality:
        split = data.checkpoint_quality.get("split", {})
        total_targets = len(split.get("eligible_target_ids", []))
        checkpoint_targets = int(checkpoint_summary.get("selected_targets", 0))
        calibration_targets = max(0, total_targets - checkpoint_targets)
        story.append(Paragraph(
            f"Known target split: <b>{total_targets} total</b> · "
            f"<b>{calibration_targets}</b> used for calibration · "
            f"<b>{checkpoint_targets}</b> withheld as checkpoints",
            S["doc_subtitle"],
        ))
    story.append(Spacer(1, 14 * mm))

    rms_txt = f"{overall_err.rms:.2f} px" if overall_err.n else "n/a"
    med_txt = f"{overall_err.median:.2f} px" if overall_err.n else "n/a"
    obj_txt = f"{overall_obj.median:.2f} mm" if overall_obj.n else "n/a"
    story.append(_kpi_cards([
        (f"{rich.n_observations:,}", "observations"),
        ((f"{rich.n_targets} + {checkpoint_summary.get('selected_targets', 0)}"
          if data.checkpoint_quality else f"{rich.n_targets}"),
         "calibration + checkpoint targets" if data.checkpoint_quality else "targets"),
        (f"{rich.n_frames}", "frames"),
        (rms_txt, "rms reproj."),
        (med_txt, "median reproj."),
        (obj_txt, "median ray err."),
    ], S))
    story.append(Spacer(1, 14 * mm))

    meta_rows = [["Generated", data.generated_at.strftime("%Y-%m-%d %H:%M:%S")]]
    for key, value in data.meta.items():
        meta_rows.append([key, value])
    story.append(_data_table(["Run information", ""], meta_rows,
                             col_widths=[45 * mm, CONTENT_W - 45 * mm], styles=S))

    story.append(PageBreak())

    # ── 1 Summary ─────────────────────────────────────────────────────────
    story += _section("1", "Summary", S)

    err_header = ["Residuals", "N", "Mean", "RMS", "Median", "P95", "Max"]
    rows = []
    if overall_err.n:
        rows.append(["Calibration reprojection (px)", f"{overall_err.n}",
                     f"{overall_err.mean:.2f}", f"{overall_err.rms:.2f}",
                     f"{overall_err.median:.2f}", f"{overall_err.p95:.2f}",
                     f"{overall_err.max:.2f}"])
    if overall_obj.n:
        rows.append(["Calibration point-to-ray (mm)", f"{overall_obj.n}",
                     f"{overall_obj.mean:.2f}", f"{overall_obj.rms:.2f}",
                     f"{overall_obj.median:.2f}", f"{overall_obj.p95:.2f}",
                     f"{overall_obj.max:.2f}"])
    if checkpoint_reproj.n:
        rows.append(["Checkpoint reprojection (px)", f"{checkpoint_reproj.n}",
                     f"{checkpoint_reproj.mean:.2f}", f"{checkpoint_reproj.rms:.2f}",
                     f"{checkpoint_reproj.median:.2f}", f"{checkpoint_reproj.p95:.2f}",
                     f"{checkpoint_reproj.max:.2f}"])
    if checkpoint_obj.n:
        rows.append(["Checkpoint point-to-ray (mm)", f"{checkpoint_obj.n}",
                     f"{checkpoint_obj.mean:.2f}", f"{checkpoint_obj.rms:.2f}",
                     f"{checkpoint_obj.median:.2f}", f"{checkpoint_obj.p95:.2f}",
                     f"{checkpoint_obj.max:.2f}"])
    story.append(_data_table(err_header, rows, styles=S))
    story.append(Spacer(1, 4 * mm))

    signed_rows = []
    for label, component in (
        ("Horizontal, Δu", signed.u), ("Vertical, Δv", signed.v),
        ("Radial", signed.radial), ("Tangential", signed.tangential),
    ):
        signed_rows.append([
            label, f"{component.mean:.4f}", f"{component.std:.4f}",
            f"{component.rms:.4f}", f"{component.p95:.4f}",
        ])
    story.append(_data_table(
        ["Directional residual (px)", "Mean bias", "Std. dev.", "RMS", "P95 absolute"],
        signed_rows, styles=S,
    ))
    story.append(Paragraph(
        "Residual = predicted image position minus measured position. Positive Δu points right "
        "and positive Δv points down. Positive radial residual points outward from the calibrated "
        "principal point; tangential is perpendicular to that direction. P95 absolute is the "
        "95th percentile of component magnitude, irrespective of sign.", S["caption"],
    ))
    story.append(Spacer(1, 4 * mm))

    cam_header = ["Camera", "Resolution", "Obs", "Mean", "RMS", "Median", "P95",
                  "Max (px)", "Ray med. (mm)", "FoV occupancy"]
    cam_rows = []
    for cs in cam_stats:
        cam_info = next(c for c in data.cameras if c.name == cs.camera)
        ray_med = f"{cs.errors_obj_mm.median:.2f}" if cs.errors_obj_mm.n else "—"
        cam_rows.append([
            cs.camera, f"{cam_info.width}×{cam_info.height}", f"{cs.errors.n:,}",
            f"{cs.errors.mean:.2f}", f"{cs.errors.rms:.2f}", f"{cs.errors.median:.2f}",
            f"{cs.errors.p95:.2f}", f"{cs.errors.max:.2f}", ray_med,
            f"{100 * cs.coverage.occupancy_fraction:.0f}%",
        ])
    story.append(_data_table(cam_header, cam_rows, styles=S))
    story.append(Spacer(1, 6 * mm))

    reported_dof = data.adjustment.dof if data.adjustment is not None else rich.dof
    reported_redundancy = reported_dof / max(2 * rich.n_observations, 1) if reported_dof > 0 else float("nan")
    red_rows = [
        ["Registered frames / targets / observations",
         f"{rich.n_frames} / {rich.n_targets} / {rich.n_observations:,}"],
        ["Observations per target (median, min–max)",
         f"{rich.median_obs_per_target:.1f}  ({int(rich.obs_per_target.min())}–"
         f"{int(rich.obs_per_target.max())})" if rich.obs_per_target.size else "—"],
        ["Targets per frame (median, min–max)",
         f"{rich.median_targets_per_frame:.1f}  ({int(rich.targets_per_frame.min())}–"
         f"{int(rich.targets_per_frame.max())})" if rich.targets_per_frame.size else "—"],
        ["Nominal parameter columns", f"{rich.unknowns:,}"],
        ["Degrees of freedom (rank-aware)", f"{reported_dof:,}" if reported_dof > 0 else "unavailable"],
        ["Global redundancy ratio", f"{reported_redundancy:.3f}" if np.isfinite(reported_redundancy) else "unavailable"],
    ]
    if data.filter_stats is not None and data.filter_stats.total_observations:
        fs = data.filter_stats
        red_rows.append(["Outlier filter (object-space MAD)",
                         f"{fs.excluded_observations}/{fs.total_observations} removed · "
                         f"threshold {fs.threshold * 1000:.2f} mm "
                         f"(median {fs.median_residual * 1000:.2f} + "
                         f"{fs.mad_scale:.1f}·{fs.spread_source.upper()} "
                         f"{fs.mad * 1000:.2f})"])
    if data.rig is not None and data.rig.fixed_point_ids:
        red_rows.append(["Ground-truth anchored targets",
                         f"{len(data.rig.fixed_point_ids)} of {rich.n_targets}"])
    if data.adjustment is not None:
        adj = data.adjustment
        red_rows += [
            ["A-posteriori sigma0", f"{adj.sigma0_px:.4f} px per coordinate"],
            ["Camera-parameter correlation condition number",
             f"{adj.condition_number:.3g}" if np.isfinite(adj.condition_number) else "unavailable"],
            ["Jacobian rank / active tangent columns",
             f"{adj.numerical_rank} / {adj.jacobian_columns} ({'verified' if adj.rank_verified else 'estimated'})"
             if adj.numerical_rank is not None else "unavailable"],
        ]
        if adj.observation_sigma_px is not None:
            variance_ratio = adj.variance_factor / adj.observation_sigma_px ** 2
            interpretation = (
                "below the assumed dispersion; the a-priori sigma is conservative"
                if variance_ratio <= 1.0 else
                "above the assumed dispersion; inspect measurement quality and model adequacy"
            )
            red_rows.append([
                "Residual variance ratio (observed / assumed)",
                f"{variance_ratio:.4f} — {interpretation}",
            ])
        else:
            red_rows.append(["Residual variance ratio", "not available: a-priori sigma not supplied"])
    story.append(_data_table(["Solution redundancy", ""], red_rows,
                             col_widths=[80 * mm, CONTENT_W - 80 * mm], styles=S))
    story.append(Paragraph(
        "Nominal parameter columns are a structural count; active tangent-column "
        "rank and degrees of freedom from the linearised adjustment take precedence "
        "when available.", S["caption"],
    ))
    story.append(PageBreak())

    # ── 2 Residual analysis ───────────────────────────────────────────────
    story += _section("2", "Residual and statistical analysis", S)
    story.append(Spacer(1, 2 * mm))
    story.append(_img(fig_error_distribution(data)))
    story.append(Spacer(1, 4 * mm))
    if len(data.cameras) > 1:
        story.append(_img(fig_camera_comparison(data)))
        story.append(Spacer(1, 4 * mm))
    conv = fig_convergence(data)
    if conv is not None:
        story.append(_img(conv))
        story.append(Paragraph(
            "Each line is a separate bundle adjustment and restarts at iteration zero. Absolute "
            "costs are not directly comparable when the observation set or loss changes; the "
            "robust Cauchy stage uses a different objective from the least-squares stages.",
            S["caption"],
        ))
    story.append(PageBreak())

    standardized_fig = fig_standardized_residuals(data)
    if standardized_fig is not None:
        story.append(_img(standardized_fig))
        story.append(Spacer(1, 4 * mm))
    story.append(_img(fig_frame_target_residuals(data)))
    outlier_fig = fig_outlier_diagnostics(data)
    if outlier_fig is not None:
        story.append(Spacer(1, 4 * mm))
        story.append(_img(outlier_fig))
    story.append(PageBreak())

    # ── 3 Parameter uncertainty and observability ────────────────────────
    story += _section("3", "Parameter uncertainty and observability", S)
    if data.adjustment is None:
        story.append(Paragraph(
            "The final adjustment did not provide a covariance solution; parameter uncertainty "
            "and observability are unavailable.", S["body"],
        ))
    else:
        adj = data.adjustment
        uncertainty_rows = [
            ["Covariance method", adj.covariance_method],
            ["Degrees of freedom", f"{adj.dof:,}"],
            ["A-posteriori variance factor", f"{adj.variance_factor:.6g} px²"],
            ["A-posteriori sigma0", f"{adj.sigma0_px:.6g} px per coordinate"],
            ["Correlation-matrix condition number",
             f"{adj.condition_number:.6g}" if np.isfinite(adj.condition_number) else "unavailable"],
            ["Rank / nullity", f"{adj.numerical_rank} / {adj.nullity}" if adj.numerical_rank is not None and adj.nullity is not None else "unavailable"],
        ]
        if adj.observation_sigma_px is not None:
            variance_ratio = adj.variance_factor / adj.observation_sigma_px ** 2
            interpretation = (
                "observed dispersion is below the assumption (conservative a-priori sigma)"
                if variance_ratio <= 1.0 else
                "observed dispersion exceeds the assumption; examine data and model"
            )
            uncertainty_rows += [
                ["A-priori image-coordinate sigma", f"{adj.observation_sigma_px:.6g} px"],
                ["Residual variance ratio (observed / assumed)",
                 f"{variance_ratio:.6g} — {interpretation}"],
            ]
        else:
            uncertainty_rows.append([
                "Residual variance ratio", "not available: no a-priori coordinate sigma",
            ])
        story.append(_data_table(["Adjustment diagnostic", "Value"], uncertainty_rows,
                                 col_widths=[65 * mm, CONTENT_W - 65 * mm], styles=S))
        story.append(Paragraph(
            "The variance ratio compares the a-posteriori residual variance with the variance "
            "implied by the supplied image-coordinate sigma. Values above 1 indicate more "
            "dispersion than assumed and merit investigation. Values below 1 indicate that the "
            "assumed sigma is conservative; they are not reported as a calibration failure. "
            "Residual filtering and correlation mean this ratio is a diagnostic, not a formal "
            "pass/fail test.", S["caption"],
        ))
        correlation_fig = fig_parameter_correlation(data)
        if correlation_fig is not None:
            story.append(PageBreak())
            story.append(_img(correlation_fig))
        uncertainty_distance_fig = fig_uncertainty_vs_distance(data)
        if uncertainty_distance_fig is not None:
            story.append(PageBreak())
            story.append(_img(uncertainty_distance_fig))
            story.append(Paragraph(
                "The curves combine linearized calibration covariance with the supplied image-"
                "coordinate sigma (or sigma0 when none was supplied). The projection curve is "
                "evaluated for a nominal point on the reference-camera optical axis. The reference "
                "camera pose is fixed, so its pixel location is nearly invariant with distance and "
                "its curve can be flat. A secondary camera sees that point with a baseline offset; "
                "translation perturbations then contribute an angular term that decreases roughly "
                "as 1/Z. This is the expected geometry of the linearized model, not a failure of "
                "the calibration.", S["caption"],
            ))
    story.append(PageBreak())

    # ── Per-camera sections ───────────────────────────────────────────────
    sec_idx = 4
    for cam, cstats in zip(data.cameras, cam_stats):
        story += _section(str(sec_idx), f"Camera {cam.name}", S)

        intr = cam.intrinsics
        estimate_rows = _parameter_rows(data, cam.name)
        if estimate_rows:
            story.append(_data_table(
                ["Parameter", "Unit", "Initial", "Final", "Std. dev.", "95% CI", "Status"],
                estimate_rows, styles=S,
            ))
            story.append(Paragraph(
                "Uncertainties are linearized marginal estimates; frame poses and free object "
                "points are retained as nuisance parameters in the covariance solution.", S["caption"],
            ))
            story.append(Spacer(1, 4 * mm))
        param_rows = [
            ["Resolution", f"{cam.width} × {cam.height} px"],
            ["Observations", f"{cstats.errors.n:,}"],
            ["Mean / RMS error", f"{cstats.errors.mean:.2f} / {cstats.errors.rms:.2f} px"],
            ["Median / P95 error", f"{cstats.errors.median:.2f} / {cstats.errors.p95:.2f} px"],
        ]
        camera_signed = signed_residual_stats([r for r in data.records if r.camera == cam.name])
        param_rows += [
            ["Signed u bias / std", f"{camera_signed.u.mean:.4f} / {camera_signed.u.std:.4f} px"],
            ["Signed v bias / std", f"{camera_signed.v.mean:.4f} / {camera_signed.v.std:.4f} px"],
        ]
        if cstats.errors_obj_mm.n:
            param_rows.append(["Ray residual (median)",
                               f"{cstats.errors_obj_mm.median:.2f} mm"])
        if data.rig is not None and cam.name != data.cameras[0].name:
            rp = data.rig.relative_poses[cam.name]
            param_rows += [
                ["Baseline to reference", f"{data.rig.baselines_m[cam.name]:.4f} m"],
                ["Relative rotation", f"{data.rig.rel_rotation_deg[cam.name]:.3f}°"],
                ["Relative translation",
                 f"[{rp[3]:.4f}, {rp[4]:.4f}, {rp[5]:.4f}] m"],
            ]

        left_col = _data_table(["Parameter", "Value"], param_rows,
                               col_widths=[42 * mm, 62 * mm], styles=S)
        right_col = _coverage_mini_table(cstats, S)
        two_col = Table([[left_col, right_col]],
                        colWidths=[CONTENT_W * 0.56, CONTENT_W * 0.44])
        two_col.setStyle(TableStyle([
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("LEFTPADDING", (0, 0), (-1, -1), 0),
            ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ]))
        story.append(two_col)
        story.append(Spacer(1, 4 * mm))
        story.append(_img(fig_error_distribution(data, cam.name)))
        story.append(PageBreak())

        story.append(_img(fig_binned_signed_residuals(data, cam)))
        story.append(Spacer(1, 3 * mm))
        story.append(_img(fig_radial_error(data, cam)))
        story.append(PageBreak())

        story.append(_img(fig_residual_component_heatmaps(data, cam)))
        story.append(PageBreak())

        story.append(_img(fig_spatial_heatmap(data, cam)))
        story.append(Spacer(1, 3 * mm))
        story.append(_img(fig_distortion_profile(cam)))
        story.append(PageBreak())

        cov = cstats.coverage
        story.append(_img(fig_coverage_map(data, cam, cov)))
        story.append(Spacer(1, 2 * mm))
        story.append(Paragraph(
            f"Occupied cells: {cov.occupied_cells}/{cov.total_cells}; cells meeting the "
            f"minimum density ({cov.min_cell_observations} observations): "
            f"{cov.sufficient_cells}/{cov.total_cells}. Slash hatching means empty; dotted "
            "hatching means occupied but below the stated density threshold.",
            S["caption"]))
        projection_fig = fig_projection_uncertainty(data, cam)
        if projection_fig is not None:
            story.append(Spacer(1, 3 * mm))
            story.append(_img(projection_fig))
        story.append(PageBreak())
        sec_idx += 1

    # ── Network strength ──────────────────────────────────────────────────
    if data.checkpoint_quality:
        quality = data.checkpoint_quality
        summary = quality.get("summary", {})
        split = quality.get("split", {})
        story += _section(str(sec_idx), "Independent checkpoint accuracy", S)
        rows = [
            ["Requested / realised ratio", f"{split.get('requested_ratio', 0):.3f} / {split.get('realized_ratio', 0):.3f}"],
            ["Random seed / generator", f"{split.get('seed')} / {split.get('generator', 'unknown')}"],
            ["Selected target IDs", ", ".join(map(str, split.get('checkpoint_target_ids', [])))],
            ["Selected / posed / reconstructed", f"{summary.get('selected_targets', 0)} / {summary.get('observed_in_posed_frames', 0)} / {summary.get('xyz_reconstructed', 0)}"],
            ["Failed / weak geometry", f"{summary.get('xyz_failed', 0)} / {summary.get('weak_geometry', 0)}"],
        ]
        if "rmse_3d_m" in summary:
            bias = np.asarray(summary["mean_error_xyz_m"]) * 1000.0
            rmse = np.asarray(summary["rmse_xyz_m"]) * 1000.0
            rows += [
                ["Signed XYZ bias (mm)", "/".join(f"{value:.3f}" for value in bias)],
                ["XYZ RMSE (mm)", "/".join(f"{value:.3f}" for value in rmse)],
                ["3D RMSE / median / P95 / max (mm)",
                 f"{summary['rmse_3d_m']*1000:.3f} / {summary['median_3d_m']*1000:.3f} / "
                 f"{summary['p95_3d_m']*1000:.3f} / {summary['max_3d_m']*1000:.3f}"],
            ]
        reprojection = summary.get("withheld_reprojection")
        if reprojection:
            rows.append(["Checkpoint reprojection: N / bias u,v / RMS / P95",
                         f"{reprojection['n']} / {reprojection['mean_du_px']:.3f}, "
                         f"{reprojection['mean_dv_px']:.3f} px / {reprojection['rms_2d_px']:.3f} / "
                         f"{reprojection['p95_2d_px']:.3f} px"])
            rows.append(["Checkpoint point-to-ray: RMS / median / P95",
                         f"{reprojection['object_space_rms_m']*1000:.3f} / "
                         f"{reprojection['object_space_median_m']*1000:.3f} / "
                         f"{reprojection['object_space_p95_m']*1000:.3f} mm"])
        story.append(_data_table(["Checkpoint diagnostic", "Value"], rows,
                                 col_widths=[70 * mm, CONTENT_W - 70 * mm], styles=S))
        checkpoint_fig = fig_checkpoint_accuracy(data)
        if checkpoint_fig is not None:
            story.append(Spacer(1, 4 * mm))
            story.append(_img(checkpoint_fig))
        failures = [item for item in quality.get("targets", []) if item.get("status") != "ok"]
        if failures:
            story.append(Paragraph(
                "Non-reconstructable checkpoints: " + "; ".join(
                    f"{item['target_id']} ({item.get('reason', 'unknown')})" for item in failures
                ), S["small"],
            ))
        story.append(Paragraph(
            "These target IDs were withheld from target-based initialization, bundle adjustment "
            "and calibration outlier filtering. Cameras and poses were frozen before equal-weight "
            "multi-ray reconstruction; no checkpoint alignment was fitted. Formal checkpoint "
            "p-values are not reported because calibration errors are shared and reference-"
            "coordinate uncertainty is not supplied. The adjustment residual-variance ratio "
            "is reported separately when an a-priori image-coordinate sigma is provided.", S["caption"],
        ))
        story.append(PageBreak())
        sec_idx += 1

    story += _section(str(sec_idx), "Network strength and observation geometry", S)
    story.append(Spacer(1, 2 * mm))
    story.append(_img(fig_track_richness(rich)))
    story.append(Spacer(1, 4 * mm))
    story.append(_img(fig_target_errors(data)))
    story.append(PageBreak())

    pose_fig = fig_pose_and_viewing_coverage(data)
    if pose_fig is not None:
        story.append(_img(pose_fig))
        story.append(Paragraph(
            "Target-surface incidence angles are not available because the input target file "
            "contains point coordinates but no target normals. The field-angle plot is not a "
            "substitute for surface incidence.", S["caption"],
        ))
        story.append(PageBreak())
    story.append(_img(fig_camera_target_connectivity(data)))
    story.append(Spacer(1, 4 * mm))

    def reliability_rows(items):
        return [[
            item.name, f"{item.n_observations}", f"{item.rms_px:.3f}",
            f"{item.mean_local_redundancy:.3f}" if item.mean_local_redundancy is not None else "—",
        ] for item in items]

    weak_table_header = ["ID", "Observations", "RMS (px)", "Mean local redundancy"]
    weak_tables = Table([[
        _data_table(["Weak frame"] + weak_table_header[1:], reliability_rows(weak_frames), styles=S),
        _data_table(["Weak target"] + weak_table_header[1:], reliability_rows(weak_targets), styles=S),
    ]], colWidths=[CONTENT_W / 2, CONTENT_W / 2])
    weak_tables.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 2),
        ("RIGHTPADDING", (0, 0), (-1, -1), 2),
    ]))
    story.append(weak_tables)
    story.append(Paragraph(
        "Local redundancy is 1 - diag(H), estimated from the full bundle Jacobian, including "
        "frame and object-point nuisance parameters. Missing values indicate an unreliable "
        "projection estimate.", S["caption"],
    ))
    story.append(PageBreak())

    # Scene geometry applies to any trajectory, including a single camera.
    if data.rig is not None:
        scene_fig = fig_scene_geometry(data)
        if scene_fig is not None:
            story.append(_img(scene_fig))
            story.append(PageBreak())

    # ── Rig geometry (multi-camera only) ──────────────────────────────────
    if data.mode == "multi-camera" and data.rig is not None:
        story += _section(str(sec_idx + 1), "Rig geometry", S)
        story.append(Spacer(1, 2 * mm))
        rig_fig = fig_rig_diagram(data)
        if rig_fig is not None:
            story.append(_img(rig_fig, CONTENT_W * 0.68))
            story.append(Spacer(1, 4 * mm))
        rel_rows = []
        for cam in data.cameras:
            rp = data.rig.relative_poses[cam.name]
            rel_rows.append([
                cam.name,
                "reference" if cam == data.cameras[0] else f"{data.rig.baselines_m[cam.name]:.4f} m",
                f"{data.rig.rel_rotation_deg[cam.name]:.3f}°",
                f"[{rp[3]:.4f}, {rp[4]:.4f}, {rp[5]:.4f}]",
            ])
        story.append(_data_table(
            ["Camera", "Baseline", "Rel. rotation", "Translation [x, y, z] (m)"],
            rel_rows, styles=S))
        story.append(Spacer(1, 4 * mm))
        for cam in data.cameras[1:]:
            pose_rows = [
                row for row in _parameter_rows(data, cam.name)
                if row[0] in {"rx", "ry", "rz", "tx", "ty", "tz"}
            ]
            if pose_rows:
                story.append(Paragraph(f"{cam.name} rig-extrinsic uncertainty", S["h2"]))
                story.append(_data_table(
                    ["Component", "Unit", "Initial", "Final", "Std. dev.", "95% CI", "Status"],
                    pose_rows, styles=S,
                ))
                story.append(Spacer(1, 3 * mm))

        if rig_pairs:
            pair_rows = []
            for pair in rig_pairs:
                def median_or_dash(values, scale=1.0):
                    return f"{np.median(values) * scale:.4g}" if values.size else "—"

                xyz_rms = (
                    np.sqrt(np.mean(pair.internal_xyz_errors_m ** 2, axis=0)) * 1000.0
                    if pair.internal_xyz_errors_m.size else None
                )
                pair_rows.append([
                    f"{pair.camera_a}–{pair.camera_b}",
                    f"{pair.shared_frames}/{pair.shared_targets}/{pair.common_observations}",
                    median_or_dash(pair.sampson_px),
                    median_or_dash(pair.vertical_disparity_px),
                    median_or_dash(pair.intersection_angle_deg),
                    median_or_dash(pair.baseline_depth_ratio),
                    (f"{xyz_rms[0]:.3g}/{xyz_rms[1]:.3g}/{xyz_rms[2]:.3g}" if xyz_rms is not None else "—"),
                ])
            story.append(Paragraph("Pairwise rig diagnostics", S["h2"]))
            story.append(_data_table(
                ["Pair", "Frames/targets/common", "Sampson med. (px)",
                 "Rectified perp. disparity med. (px)", "Angle med. (deg)", "B/Z med.",
                 "Internal XYZ RMS (mm)"], pair_rows, styles=S,
            ))
            story.append(Paragraph(
                "Sampson distance is computed in undistorted pixel coordinates. Rectified disparity "
                "is the signed component perpendicular to the epipolar lines. Internal XYZ "
                "discrepancies compare stereo triangulation with the same adjusted targets used by "
                "the bundle and are network-consistency diagnostics.",
                S["caption"],
            ))
        story.append(PageBreak())
        for pair in rig_pairs:
            story.append(_img(fig_rig_pair_quality(pair)))
            story.append(Paragraph(
                f"Shared support for {pair.camera_a}–{pair.camera_b}: "
                f"{pair.shared_frames} frames, {pair.shared_targets} targets, "
                f"{pair.common_observations} paired observations.", S["caption"],
            ))
            story.append(PageBreak())
        sec_idx += 2

    doc.build(story)
    return out_path


def _coverage_mini_table(cstats, S: Dict[str, ParagraphStyle]) -> Table:
    cov = cstats.coverage
    rows = [
        ["FoV occupancy", f"{100 * cov.occupancy_fraction:.0f}% "
                          f"({cov.occupied_cells}/{cov.total_cells} cells)"],
        [f"Cells with >= {cov.min_cell_observations} obs",
         f"{cov.sufficient_cells}/{cov.total_cells} ({100 * cov.sufficient_fraction:.0f}%)"],
        ["Radius reach (max)", f"{cov.r_norm_max:.2f}"],
        ["Radius reach (P95)", f"{cov.r_norm_p95:.2f}"],
        ["Border margins (px)",
         f"L {cov.margin_px['left']:.0f} · R {cov.margin_px['right']:.0f} · "
         f"T {cov.margin_px['top']:.0f} · B {cov.margin_px['bottom']:.0f}"],
        ["Depth median", f"{cstats.depth_median_m:.2f} m"],
    ]
    return _data_table(["Coverage", "Value"], rows,
                       col_widths=[32 * mm, 40 * mm], styles=S)
