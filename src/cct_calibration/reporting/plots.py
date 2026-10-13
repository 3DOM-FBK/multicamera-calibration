"""Figure builders for calibration reports (matplotlib + seaborn).

Every public function returns a ``matplotlib.figure.Figure``; the PDF layer
converts them to high-resolution PNG bytes. All figures share a consistent
visual style so the final document looks cohesive.
"""

from __future__ import annotations

import io
from typing import Dict, List, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import cv2
import numpy as np
import seaborn as sns
from matplotlib.lines import Line2D
from matplotlib.colors import LogNorm
from matplotlib.patches import Polygon, Rectangle
from matplotlib.ticker import FuncFormatter

from cct_calibration.reporting.records import CameraInfo, ObservationRecord, ReportData
from cct_calibration.reporting.statistics import (
    CoverageStats,
    RigPairStats,
    RichnessStats,
    radial_error_trend,
)
from cct_calibration.reporting.uncertainty import (
    projection_uncertainty_grid,
    projection_uncertainty_vs_distance,
    triangulation_uncertainty_vs_distance,
)

# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------

CAMERA_COLORS: Dict[str, str] = {}
_PALETTE = ["#2A9D8F", "#E76F51", "#457B9D", "#E9C46A", "#9B5DE5", "#00BBF9"]


def _camera_color(name: str) -> str:
    if name not in CAMERA_COLORS:
        CAMERA_COLORS[name] = _PALETTE[len(CAMERA_COLORS) % len(_PALETTE)]
    return CAMERA_COLORS[name]


def apply_style() -> None:
    sns.set_theme(style="whitegrid", context="notebook")
    plt.rcParams.update({
        "figure.dpi": 110,
        "savefig.dpi": 150,
        "font.size": 10,
        "axes.titleweight": "bold",
        "axes.titlesize": 11.5,
        "axes.labelsize": 10,
        "axes.edgecolor": "#444444",
        "axes.linewidth": 0.8,
        "legend.frameon": True,
        "legend.framealpha": 0.9,
        "legend.fontsize": 9,
    })


apply_style()

ACCENT = "#1F3B57"
GOOD = "#2A9D8F"
WARN = "#E76F51"


def figure_to_bytes(fig: plt.Figure, dpi: int = 160) -> bytes:
    """Render a figure to PNG bytes and release its resources."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    return buf.getvalue()


def _records_for(data: ReportData, cam_name: str | None) -> List[ObservationRecord]:
    if cam_name is None:
        return data.records
    return [r for r in data.records if r.camera == cam_name]


def _subsample(values: Sequence[np.ndarray], max_points: int = 20000) -> List[np.ndarray]:
    total = sum(v.shape[0] for v in values)
    if total <= max_points:
        return list(values)
    keep = max_points / total
    out = []
    for v in values:
        n = max(1, int(v.shape[0] * keep))
        idx = np.random.default_rng(0).choice(v.shape[0], size=n, replace=False)
        out.append(v[idx])
    return out


def _camera_triangle(ax, x: float, y: float, dx: float, dy: float, size: float,
                     color: str, alpha: float = 1.0, zorder: int = 5) -> None:
    """Draw a camera as an oriented triangle.

    The apex (vertex) sits exactly on the camera centre; the triangle height
    runs along the viewing direction ``(dx, dy)`` towards the opposite side.
    """
    direction = np.array([dx, dy], dtype=np.float64)
    norm = float(np.linalg.norm(direction))
    if norm < 1e-12:
        direction = np.array([0.0, 1.0])
    else:
        direction = direction / norm
    perp = np.array([-direction[1], direction[0]])
    apex = np.array([x, y], dtype=np.float64)
    base_centre = apex + size * direction
    half_width = 0.62 * size
    vertices = [apex,
                base_centre + half_width * perp,
                base_centre - half_width * perp]
    ax.add_patch(Polygon(vertices, closed=True, facecolor=color,
                         edgecolor="#222222", linewidth=0.6, alpha=alpha,
                         zorder=zorder))


# ---------------------------------------------------------------------------
# Error distribution figures
# ---------------------------------------------------------------------------

def fig_error_distribution(data: ReportData, cam_name: str | None = None) -> plt.Figure:
    recs = _records_for(data, cam_name)
    errs = np.array([r.err_px for r in recs])
    title = f"Reprojection error distribution — {cam_name}" if cam_name \
        else "Reprojection error distribution — all cameras"

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.0, 3.4), constrained_layout=True)
    sns.histplot(errs, bins=60, kde=True, ax=ax1, color=_camera_color(cam_name or "all"),
                 edgecolor="none", alpha=0.75)
    rms = float(np.sqrt(np.mean(errs ** 2))) if errs.size else 0.0
    med = float(np.median(errs)) if errs.size else 0.0
    ax1.axvline(med, color=ACCENT, ls="--", lw=1.2, label=f"median {med:.2f} px")
    ax1.axvline(rms, color=WARN, ls="-", lw=1.2, label=f"RMS {rms:.2f} px")
    ax1.set_xlabel("reprojection error (px)")
    ax1.set_ylabel("observations")
    ax1.set_title("Histogram + KDE")
    ax1.legend()

    sns.ecdfplot(errs, ax=ax2, color=_camera_color(cam_name or "all"), lw=1.8)
    if errs.size:
        p95 = np.percentile(errs, 95)
        ax2.axvline(p95, color=WARN, ls=":", lw=1.2)
        ax2.annotate(f"p95 = {p95:.2f} px", xy=(p95, 0.95), xytext=(p95, 0.55),
                     arrowprops=dict(arrowstyle="->", color="#666666"), fontsize=9)
    ax2.set_xlabel("reprojection error (px)")
    ax2.set_ylabel("cumulative fraction")
    ax2.set_title("Empirical CDF")
    fig.suptitle(title, fontsize=12.5, fontweight="bold")
    return fig


def fig_camera_comparison(data: ReportData) -> plt.Figure:
    names = [c.name for c in data.cameras]
    err_groups = [np.array([r.err_px for r in data.records if r.camera == n]) for n in names]
    obj_groups = [np.array([r.err_obj_m * 1000.0 for r in data.records
                            if r.camera == n and r.err_obj_m is not None]) for n in names]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.0, 3.6), constrained_layout=True)
    sns.violinplot(data=err_groups, ax=ax1, palette=[_camera_color(n) for n in names],
                   cut=0, inner="quartile", linewidth=0.8)
    ax1.set_xticks(range(len(names)), names)
    ax1.set_ylabel("reprojection error (px)")
    ax1.set_title("Pixel residuals per camera")

    sns.violinplot(data=obj_groups, ax=ax2, palette=[_camera_color(n) for n in names],
                   cut=0, inner="quartile", linewidth=0.8)
    ax2.set_xticks(range(len(names)), names)
    ax2.set_ylabel("object-space error (mm)")
    ax2.set_title("Object-space (ray) residuals per camera")
    fig.suptitle("Cross-camera comparison", fontsize=12.5, fontweight="bold")
    return fig


# ---------------------------------------------------------------------------
# Spatial structure figures
# ---------------------------------------------------------------------------

def fig_residual_quiver(data: ReportData, cam: CameraInfo) -> plt.Figure:
    recs = _records_for(data, cam.name)
    us = np.array([r.u_meas for r in recs])
    vs = np.array([r.v_meas for r in recs])
    dx = np.array([r.dx for r in recs])
    dy = np.array([r.dy for r in recs])
    err = np.array([r.err_px for r in recs])

    fig, ax = plt.subplots(figsize=(8.6, 5.4), constrained_layout=True)
    scale = 14.0
    q = ax.quiver(us, vs, dx * scale, dy * scale, err, cmap="viridis",
                  angles="xy", scale_units="xy", scale=1.0,
                  width=0.0028, headwidth=3.2, clim=(0, max(np.percentile(err, 98), 1e-6)))
    cbar = fig.colorbar(q, ax=ax, shrink=0.85, pad=0.02)
    cbar.set_label("residual magnitude (px)")
    ax.add_patch(Rectangle((0, 0), cam.width, cam.height, fill=False,
                           ec="#333333", lw=1.4))
    ax.set_xlim(-cam.width * 0.03, cam.width * 1.03)
    ax.set_ylim(cam.height * 1.03, -cam.height * 0.03)
    ax.set_aspect("equal")
    ax.set_xlabel("u (px)")
    ax.set_ylabel("v (px)")
    ax.set_title(f"{cam.name}: residual vector field (arrows ×{scale:.0f}, predicted − measured)")
    return fig


def fig_spatial_heatmap(data: ReportData, cam: CameraInfo, grid_cols: int = 8) -> plt.Figure:
    recs = _records_for(data, cam.name)
    aspect = cam.width / max(cam.height, 1)
    rows = max(2, int(round(grid_cols / aspect)))
    err_grid = np.full((rows, grid_cols), np.nan)
    cnt_grid = np.zeros((rows, grid_cols), dtype=int)

    for r in recs:
        col = int(np.clip(r.u_meas / cam.width * grid_cols, 0, grid_cols - 1))
        row = int(np.clip(r.v_meas / cam.height * rows, 0, rows - 1))
        cnt_grid[row, col] += 1
        e = 0.0 if np.isnan(err_grid[row, col]) else err_grid[row, col]
        # incremental mean
        err_grid[row, col] = e + (r.err_px - e) / cnt_grid[row, col]

    fig, ax = plt.subplots(figsize=(8.6, 5.0), constrained_layout=True)
    cmap = sns.color_palette("magma", as_cmap=True)
    cmap.set_bad("#E8E8E8")
    im = ax.imshow(err_grid, cmap=cmap, aspect="auto",
                   extent=[0, cam.width, cam.height, 0], interpolation="nearest")
    vmax = np.nanmax(err_grid) if np.isfinite(np.nanmax(err_grid)) else 1.0
    im.set_clim(0, vmax)
    for row in range(rows):
        for col in range(grid_cols):
            x0, x1 = col * cam.width / grid_cols, (col + 1) * cam.width / grid_cols
            y0, y1 = row * cam.height / rows, (row + 1) * cam.height / rows
            if cnt_grid[row, col]:
                txt = f"{err_grid[row, col]:.1f}\n({cnt_grid[row, col]})"
                color = "white" if err_grid[row, col] > 0.55 * vmax else "#222222"
                ax.text(0.5 * (x0 + x1), 0.5 * (y0 + y1), txt, ha="center",
                        va="center", fontsize=7.5, color=color)
            else:
                ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fc="#E8E8E8",
                                       ec="#BBBBBB", lw=0.5, hatch="///"))
                ax.text(0.5 * (x0 + x1), 0.5 * (y0 + y1), "no data", ha="center",
                        va="center", fontsize=7, color="#888888")
    cbar = fig.colorbar(im, ax=ax, shrink=0.85, pad=0.02)
    cbar.set_label("mean reprojection error (px)")
    ax.set_xlabel("u (px)")
    ax.set_ylabel("v (px)")
    ax.set_title(f"{cam.name}: mean error per image region ({grid_cols}×{rows} grid)")
    return fig


def fig_radial_error(data: ReportData, cam: CameraInfo) -> plt.Figure:
    recs = _records_for(data, cam.name)
    r = np.array([rec.r_norm for rec in recs])
    e = np.array([rec.err_px for rec in recs])

    fig, ax = plt.subplots(figsize=(8.6, 4.2), constrained_layout=True)
    (rs,) = _subsample([r]) if r.size else [np.array([])]
    (es,) = _subsample([e]) if e.size else [np.array([])]
    ax.scatter(rs, es, s=4, alpha=0.12, color=_camera_color(cam.name), edgecolors="none")

    centers, means = radial_error_trend(recs)
    if centers.size:
        # std band per bin
        edges = np.linspace(0, max(r.max(), 1e-6) * 1.0001, len(centers) + 1)
        stds = []
        for i in range(len(centers)):
            m = (r >= edges[i]) & (r < edges[i + 1])
            stds.append(float(e[m].std()) if m.any() else 0.0)
        stds = np.array(stds)
        ax.plot(centers, means, "-o", color=ACCENT, lw=2.0, ms=4, label="binned mean")
        ax.fill_between(centers, means - stds, means + stds, color=ACCENT, alpha=0.15,
                        label="±1σ")
        if r.size:
            p95 = float(np.percentile(r, 95))
            ax.axvline(p95, color=WARN, ls=":", lw=1.2)
            ax.annotate(f"p95 radius = {p95:.2f}", xy=(p95, ax.get_ylim()[1] * 0.92),
                        fontsize=9, color=WARN, ha="right")
    ax.set_xlabel("normalised image radius $r=\\sqrt{x_n^2+y_n^2}$")
    ax.set_ylabel("reprojection error (px)")
    ax.set_title(f"{cam.name}: error growth with field angle")
    ax.legend(loc="upper left")
    return fig


def fig_coverage_map(data: ReportData, cam: CameraInfo, coverage: CoverageStats) -> plt.Figure:
    recs = _records_for(data, cam.name)
    us = np.array([r.u_meas for r in recs])
    vs = np.array([r.v_meas for r in recs])

    fig, ax = plt.subplots(figsize=(8.6, 5.4), constrained_layout=True)
    if us.size > 30:
        try:
            sns.kdeplot(x=us, y=vs, fill=True, cmap="Blues", levels=12, thresh=0.03,
                        alpha=0.75, ax=ax)
        except Exception:
            pass
    ax.scatter(us, vs, s=3, alpha=0.25, color="#264653", edgecolors="none")

    # occupancy grid overlay
    g_rows, g_cols = coverage.grid_rows, coverage.grid_cols
    for row in range(g_rows):
        for col in range(g_cols):
            x0, x1 = col * cam.width / g_cols, (col + 1) * cam.width / g_cols
            y0, y1 = row * cam.height / g_rows, (row + 1) * cam.height / g_rows
            if coverage.occupancy[row, col]:
                ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False,
                                       ec="#2A9D8F", lw=0.7, alpha=0.6))
                if coverage.counts[row, col] < coverage.min_cell_observations:
                    ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fc="#E9C46A",
                                           alpha=0.16, ec="#E9C46A", lw=0.7, hatch=".."))
            else:
                ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fc="#E76F51",
                                       alpha=0.16, ec="#E76F51", lw=0.7, hatch="///"))
            ax.text(0.5 * (x0 + x1), 0.5 * (y0 + y1), str(coverage.counts[row, col]),
                    fontsize=6.5, color="#444444", ha="center", va="center")

    # principal point marker
    ax.plot(cam.intrinsics[2], cam.intrinsics[3], "+", ms=12, color="#C1121F", mew=2.0)
    ax.annotate("principal point", xy=(cam.intrinsics[2], cam.intrinsics[3]),
                xytext=(cam.intrinsics[2] + 0.05 * cam.width,
                        cam.intrinsics[3] + 0.05 * cam.height),
                fontsize=8.5, color="#C1121F")

    ax.set_xlim(0, cam.width)
    ax.set_ylim(cam.height, 0)
    ax.set_aspect("equal")
    ax.set_xlabel("u (px)")
    ax.set_ylabel("v (px)")
    ax.set_title(f"{cam.name}: observation density & FoV coverage — "
                 f"{coverage.sufficient_cells}/{coverage.total_cells} cells have "
                 f">={coverage.min_cell_observations} observations")
    return fig


def _component_grid(
    records: Sequence[ObservationRecord],
    camera: CameraInfo,
    attribute: str,
    columns: int = 8,
) -> tuple[np.ndarray, np.ndarray]:
    rows = max(2, int(round(columns / (camera.width / max(camera.height, 1)))))
    sums = np.zeros((rows, columns), dtype=np.float64)
    counts = np.zeros((rows, columns), dtype=np.int64)
    for record in records:
        column = int(np.clip(record.u_meas / camera.width * columns, 0, columns - 1))
        row = int(np.clip(record.v_meas / camera.height * rows, 0, rows - 1))
        sums[row, column] += float(getattr(record, attribute))
        counts[row, column] += 1
    grid = np.divide(sums, counts, out=np.full_like(sums, np.nan), where=counts > 0)
    return grid, counts


def fig_binned_signed_residuals(data: ReportData, camera: CameraInfo, columns: int = 12) -> plt.Figure:
    """Mean signed residual vector per image region, with sample support."""
    records = _records_for(data, camera.name)
    dx, counts = _component_grid(records, camera, "dx", columns)
    dy, _ = _component_grid(records, camera, "dy", columns)
    rows = dx.shape[0]
    x = (np.arange(columns) + 0.5) * camera.width / columns
    y = (np.arange(rows) + 0.5) * camera.height / rows
    xx, yy = np.meshgrid(x, y)
    magnitude = np.hypot(dx, dy)

    fig, ax = plt.subplots(figsize=(8.6, 5.0), constrained_layout=True)
    background = ax.imshow(
        counts, extent=[0, camera.width, camera.height, 0], aspect="auto",
        cmap="Greys", alpha=0.25, interpolation="nearest",
    )
    scale = 25.0
    quiver = ax.quiver(
        xx, yy, dx * scale, dy * scale, magnitude,
        cmap="viridis", angles="xy", scale_units="xy", scale=1.0,
        width=0.0045, headwidth=3.5,
    )
    fig.colorbar(quiver, ax=ax, shrink=0.82, pad=0.02).set_label("mean signed-vector magnitude (px)")
    for row in range(rows):
        for column in range(columns):
            if counts[row, column]:
                ax.text(xx[row, column], yy[row, column], str(counts[row, column]),
                        fontsize=6.5, color="#333333", ha="center", va="bottom")
    ax.set_xlim(0, camera.width)
    ax.set_ylim(camera.height, 0)
    ax.set_aspect("equal")
    ax.set_xlabel("u (px)")
    ax.set_ylabel("v (px)")
    ax.set_title(f"{camera.name}: binned mean signed residuals (arrows x{scale:.0f}; labels are counts)")
    return fig


def fig_residual_component_heatmaps(data: ReportData, camera: CameraInfo) -> plt.Figure:
    records = _records_for(data, camera.name)
    components = [
        ("dx", "u residual"),
        ("dy", "v residual"),
        ("residual_radial", "radial residual"),
        ("residual_tangential", "tangential residual"),
    ]
    grids = [_component_grid(records, camera, attribute)[0] for attribute, _ in components]
    finite = np.concatenate([grid[np.isfinite(grid)] for grid in grids if np.any(np.isfinite(grid))])
    limit = float(np.percentile(np.abs(finite), 98)) if finite.size else 1.0
    limit = max(limit, 1e-9)
    fig, axes = plt.subplots(2, 2, figsize=(8.6, 6.0), constrained_layout=True)
    image = None
    for axis, grid, (_, title) in zip(axes.flat, grids, components):
        image = axis.imshow(
            grid, extent=[0, camera.width, camera.height, 0], aspect="auto",
            cmap="coolwarm", vmin=-limit, vmax=limit, interpolation="nearest",
        )
        axis.set_title(title)
        axis.set_xlabel("u (px)")
        axis.set_ylabel("v (px)")
    if image is not None:
        fig.colorbar(image, ax=axes, shrink=0.8, pad=0.02).set_label("mean signed residual (px)")
    fig.suptitle(f"{camera.name}: signed residual components", fontsize=12.5, fontweight="bold")
    return fig


def fig_standardized_residuals(data: ReportData) -> plt.Figure | None:
    standardized_u = np.array([
        record.standardized_dx for record in data.records if record.standardized_dx is not None
    ])
    standardized_v = np.array([
        record.standardized_dy for record in data.records if record.standardized_dy is not None
    ])
    if standardized_u.size == 0 or standardized_v.size == 0:
        return None
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.0, 3.7), constrained_layout=True)
    bins = np.linspace(
        max(-8.0, float(min(standardized_u.min(), standardized_v.min()))),
        min(8.0, float(max(standardized_u.max(), standardized_v.max()))), 70,
    )
    ax1.hist(standardized_u, bins=bins, density=True, alpha=0.55, label="u", color="#E76F51")
    ax1.hist(standardized_v, bins=bins, density=True, alpha=0.55, label="v", color="#457B9D")
    x = np.linspace(bins[0], bins[-1], 300)
    ax1.plot(x, np.exp(-0.5 * x ** 2) / np.sqrt(2.0 * np.pi), "k--", lw=1.1, label="N(0,1)")
    ax1.set_xlabel("internally standardized residual")
    ax1.set_ylabel("density")
    ax1.set_title("Standardized coordinate residuals")
    ax1.legend()

    # Pair all four values from the same observation.  Covariance/hat-matrix
    # diagnostics can legitimately contain NaNs for a singular or fixed
    # component; independently filtering x and y creates arrays of different
    # lengths and makes Matplotlib reject the scatter call.
    paired = [
        record for record in data.records
        if record.local_redundancy_u is not None
        and record.local_redundancy_v is not None
        and record.standardized_dx is not None
        and record.standardized_dy is not None
        and np.isfinite(record.local_redundancy_u)
        and np.isfinite(record.local_redundancy_v)
        and np.isfinite(record.standardized_dx)
        and np.isfinite(record.standardized_dy)
    ]
    redundancies = np.array([
        0.5 * (record.local_redundancy_u + record.local_redundancy_v)
        for record in paired
    ])
    standardized_norm = np.array([
        np.hypot(record.standardized_dx, record.standardized_dy)
        for record in paired
    ])
    ax2.scatter(redundancies, standardized_norm, s=4, alpha=0.18, color=ACCENT, edgecolors="none")
    ax2.set_xlabel("local redundancy number (mean u/v)")
    ax2.set_ylabel("standardized residual magnitude")
    ax2.set_title("Residual reliability")
    return fig


def fig_frame_target_residuals(data: ReportData) -> plt.Figure:
    def aggregate(attribute: str, sort_by_residual: bool = True):
        grouped: Dict[object, List[float]] = {}
        for record in data.records:
            grouped.setdefault(getattr(record, attribute), []).append(record.err_px)
        names = list(grouped)
        rms = np.array([np.sqrt(np.mean(np.square(grouped[name]))) for name in names])
        counts = np.array([len(grouped[name]) for name in names])
        if sort_by_residual:
            order = np.argsort(rms)[::-1]
        else:
            order = np.argsort(np.asarray(names, dtype=np.int64))
        return [names[index] for index in order], rms[order], counts[order]

    frame_names, frame_rms, frame_counts = aggregate("frame")
    target_names, target_rms, target_counts = aggregate("target_id", sort_by_residual=False)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8.8, 6.0), constrained_layout=True)
    ax1.scatter(np.arange(frame_rms.size), frame_rms,
                s=np.clip(18.0 + 3.0 * frame_counts, 24.0, 180.0),
                alpha=0.88, color="#457B9D", edgecolors="#1D3557", linewidths=0.45)
    ax1.set_xlabel("frames ranked by RMS (worst to best)")
    ax1.set_ylabel("RMS residual (px)")
    ax1.set_title("Per-frame residuals (larger markers have more observations)")
    target_positions = np.arange(target_rms.size)
    ax2.bar(target_positions, target_rms, color="#2A9D8F", alpha=0.82,
            edgecolor="#1D3557", linewidth=0.35)
    if target_rms.size <= 30:
        ax2.set_xticks(target_positions, [str(target) for target in target_names])
    else:
        tick_step = max(1, int(np.ceil(target_rms.size / 20.0)))
        tick_positions = target_positions[::tick_step]
        ax2.set_xticks(tick_positions, [str(target_names[index]) for index in tick_positions])
    ax2.set_xlabel("target ID")
    ax2.set_ylabel("RMS residual (px)")
    ax2.set_title("Per-target residuals by target ID (bar height = RMS)")
    ax2.grid(axis="x", visible=False)
    return fig


def fig_outlier_diagnostics(data: ReportData) -> plt.Figure | None:
    if not data.rejected_records:
        return None
    fig, axes = plt.subplots(2, len(data.cameras), figsize=(9.0, 6.1), constrained_layout=True)
    axes = np.asarray(axes).reshape(2, -1)
    for camera_index, camera in enumerate(data.cameras):
        axis = axes[0, camera_index]
        kept = [record for record in data.records if record.camera == camera.name]
        rejected = [record for record in data.rejected_records if record.camera == camera.name]
        axis.scatter([r.u_meas for r in kept], [r.v_meas for r in kept], s=3,
                     alpha=0.12, color="#457B9D", label=f"kept ({len(kept)})")
        axis.scatter([r.u_meas for r in rejected], [r.v_meas for r in rejected], s=9,
                     alpha=0.6, color="#C1121F", label=f"rejected ({len(rejected)})")
        axis.set_xlim(0, camera.width)
        axis.set_ylim(camera.height, 0)
        axis.set_aspect("equal")
        axis.set_title(camera.name)
        axis.set_xlabel("u (px)")
        axis.set_ylabel("v (px)")
        axis.legend(loc="best")
        distribution_axis = axes[1, camera_index]
        prefilter = data.prefilter_reprojection_by_camera.get(camera.name, np.array([]))
        if prefilter.size:
            sns.ecdfplot(prefilter, ax=distribution_axis,
                         color="#E9C46A", label="pre-filter candidates (robust solution)")
        if kept:
            sns.ecdfplot([record.err_px for record in kept], ax=distribution_axis,
                         color="#457B9D", label="post-filter retained (final solution)")
        if rejected:
            sns.ecdfplot([record.err_px for record in rejected], ax=distribution_axis,
                         color="#C1121F", ls="--", label="rejected (evaluated with final model)")
        distribution_axis.set_xlabel("reprojection residual (px)")
        distribution_axis.set_ylabel("cumulative fraction")
        distribution_axis.set_title(f"{camera.name}: retained vs rejected")
        distribution_axis.legend(fontsize=7)
    fig.suptitle("Outlier filtering: spatial selection and post-fit distributions",
                 fontsize=12.5, fontweight="bold")
    return fig


# ---------------------------------------------------------------------------
# Richness figures
# ---------------------------------------------------------------------------

def fig_track_richness(rich: RichnessStats) -> plt.Figure:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.0, 3.4), constrained_layout=True)
    sns.histplot(rich.obs_per_target, bins=min(30, max(10, int(rich.obs_per_target.max()))),
                 ax=ax1, color=GOOD, edgecolor="none", alpha=0.8)
    med = rich.median_obs_per_target
    ax1.axvline(med, color=ACCENT, ls="--", lw=1.3, label=f"median = {med:.1f}")
    ax1.set_xlabel("observations per target")
    ax1.set_ylabel("targets")
    ax1.set_title("Track lengths")
    ax1.legend()

    sns.histplot(rich.targets_per_frame, bins=25, ax=ax2, color="#457B9D",
                 edgecolor="none", alpha=0.8)
    med2 = rich.median_targets_per_frame
    ax2.axvline(med2, color=ACCENT, ls="--", lw=1.3, label=f"median = {med2:.1f}")
    ax2.set_xlabel("distinct targets seen per frame")
    ax2.set_ylabel("frames")
    ax2.set_title("Frame richness")
    ax2.legend()
    fig.suptitle("Observation richness", fontsize=12.5, fontweight="bold")
    return fig


def fig_target_errors(data: ReportData) -> plt.Figure:
    per_target: Dict[int, List[float]] = {}
    per_target_cam: Dict[int, Dict[str, int]] = {}
    for r in data.records:
        per_target.setdefault(r.target_id, []).append(r.err_px)
        per_target_cam.setdefault(r.target_id, {})
        per_target_cam[r.target_id][r.camera] = per_target_cam[r.target_id].get(r.camera, 0) + 1

    ids = sorted(per_target)
    counts = np.array([len(per_target[t]) for t in ids], dtype=float)
    means = np.array([np.mean(per_target[t]) for t in ids])
    cams_seen = np.array([len(per_target_cam[t]) for t in ids], dtype=float)

    fig, ax = plt.subplots(figsize=(8.6, 4.4), constrained_layout=True)
    sc = ax.scatter(counts, means, s=28, c=cams_seen, cmap="viridis", alpha=0.85,
                    edgecolors="#333333", linewidths=0.4)
    cbar = fig.colorbar(sc, ax=ax, ticks=[int(cams_seen.min()), int(cams_seen.max())])
    cbar.set_label("number of cameras observing target")

    worst = np.argsort(means)[-3:]
    for i in worst:
        ax.annotate(f"id {ids[i]}", xy=(counts[i], means[i]),
                    xytext=(counts[i] + 0.3, means[i]), fontsize=8.5, color=WARN)

    ax.set_xlabel("observations per target")
    ax.set_ylabel("mean reprojection residual magnitude (px)")
    ax.set_xscale("log")
    ax.xaxis.set_major_formatter(FuncFormatter(
        lambda value, _position: (
            "" if value <= 0 else f"{value:g}" if value < 1 else f"{value:.0f}"
        )
    ))
    ax.set_title("Per-target fit residual vs. observation count")
    return fig


def fig_depth_distribution(data: ReportData) -> plt.Figure:
    names = [c.name for c in data.cameras]
    groups = [[r.depth_m for r in data.records if r.camera == n and r.depth_m is not None]
              for n in names]
    groups = [g for g in groups if g]

    fig, ax = plt.subplots(figsize=(7.2, 3.6), constrained_layout=True)
    if groups:
        sns.violinplot(data=groups, ax=ax, cut=0, inner="quartile", linewidth=0.8,
                       palette=[_camera_color(n) for n in names[:len(groups)]])
        ax.set_xticks(range(len(groups)), names[:len(groups)])
    ax.set_ylabel("target depth (m)")
    ax.set_title("Range distribution of observed targets")
    return fig


# ---------------------------------------------------------------------------
# Geometry figures
# ---------------------------------------------------------------------------

def fig_distortion_profile(cam: CameraInfo) -> plt.Figure:
    intr = cam.intrinsics
    k1, k2, p1, p2 = intr[4], intr[5], intr[6], intr[7]
    focal = 0.5 * (intr[0] + intr[1])
    corner_r = float(np.hypot(cam.width / 2.0 / intr[0], cam.height / 2.0 / intr[1]))
    r = np.linspace(0, corner_r * 1.05, 300)

    radial_factor = 1 + k1 * r ** 2 + k2 * r ** 4
    dr_px = focal * r * (radial_factor - 1.0)

    # tangential displacement evaluated along the diagonal x = y = r/sqrt(2)
    xd = yd = r / np.sqrt(2.0)
    rsq = xd ** 2 + yd ** 2
    dtx = 2 * p1 * xd * yd + p2 * (rsq + 2 * xd ** 2)
    dty = p1 * (rsq + 2 * yd ** 2) + 2 * p2 * xd * yd
    tang_px = focal * np.hypot(dtx, dty)

    fig, ax = plt.subplots(figsize=(8.2, 4.0), constrained_layout=True)
    ax.plot(r, dr_px, color=ACCENT, lw=2.2, label="radial $\\Delta r$")
    ax.plot(r, tang_px, color=WARN, lw=2.0, ls="--", label="tangential $|\\Delta t|$ (diagonal)")
    ax.axvline(corner_r, color="#888888", ls=":", lw=1.2)
    ax.annotate("image corner", xy=(corner_r, ax.get_ylim()[1] * 0.05),
                fontsize=9, color="#555555", ha="right", rotation=90)
    ax.axhline(0.0, color="#CCCCCC", lw=0.8)
    ax.set_xlabel("normalised radius from principal point")
    ax.set_ylabel("displacement (px)")
    ax.set_title(f"{cam.name}: distortion profile  ($k_1$={k1:.4f}, $k_2$={k2:.4f}, "
                 f"$p_1$={p1:.4f}, $p_2$={p2:.4f})")
    ax.legend()
    return fig


def fig_scene_geometry(data: ReportData) -> plt.Figure:
    assert data.rig is not None
    rig = data.rig
    traj = rig.rig_centers
    pts = rig.points_array

    all_pts = np.vstack([p for p in [traj, pts] if p.shape[0]]) if (traj.shape[0] or pts.shape[0]) else np.zeros((0, 3))
    centered = all_pts - all_pts.mean(axis=0)
    if centered.shape[0] >= 2:
        _, _, vt = np.linalg.svd(centered, full_matrices=False)
        basis = vt[:2]
    else:
        basis = np.eye(2, 3)

    def proj(p: np.ndarray) -> np.ndarray:
        return (p - all_pts.mean(axis=0)) @ basis.T

    def proj_dir(d: np.ndarray) -> np.ndarray:
        return d @ basis.T

    fig, ax = plt.subplots(figsize=(8.6, 5.6), constrained_layout=True)

    legend_handles: List[Line2D] = []
    if pts.shape[0]:
        pp = proj(pts)
        ax.scatter(pp[:, 0], pp[:, 1], s=14, c="#264653", alpha=0.7, zorder=3)
        legend_handles.append(Line2D([0], [0], linestyle="none", marker="o",
                                     markerfacecolor="#264653", alpha=0.7,
                                     markersize=6, label=f"targets ({pts.shape[0]})"))

    # marker scale from the projected scene extent
    if all_pts.shape[0]:
        pa = proj(all_pts)
        span = float(max(np.ptp(pa[:, 0]), np.ptp(pa[:, 1])))
    else:
        span = 1.0
    tri_size = max(span * 0.035, 1e-9)

    colors = ["#C1121F", "#2A9D8F", "#457B9D", "#E9C46A"]
    for i, cam_name in enumerate(rig.camera_centers):
        centers = rig.camera_centers[cam_name]
        forwards = rig.camera_forwards.get(cam_name)
        if centers.shape[0] == 0:
            continue
        color = colors[i % len(colors)]
        pc = proj(centers)
        if centers.shape[0] > 1:
            ax.plot(pc[:, 0], pc[:, 1], "-", color=color, lw=1.0, alpha=0.5, zorder=3)
        # oriented camera triangles, subsampled for readability
        step = max(1, centers.shape[0] // 24)
        for k in range(0, centers.shape[0], step):
            if forwards is not None and forwards.shape[0] == centers.shape[0]:
                d2 = proj_dir(forwards[k])
            else:
                d2 = np.array([0.0, 1.0])
            _camera_triangle(ax, pc[k, 0], pc[k, 1], d2[0], d2[1], tri_size,
                             color, zorder=4)
        legend_handles.append(Line2D([0], [0], linestyle="-", color=color, lw=1.0,
                                     alpha=0.6, marker="^", markerfacecolor=color,
                                     markeredgecolor="#222222", markersize=7,
                                     label=f"{cam_name} ({centers.shape[0]} frames)"))

    ax.set_aspect("equal")
    ax.set_xlabel("principal component 1 (m)")
    ax.set_ylabel("principal component 2 (m)")
    span3d = float(np.linalg.norm(all_pts.max(axis=0) - all_pts.min(axis=0))) if all_pts.shape[0] else 0.0
    ax.set_title(f"Scene geometry — top-down projection (scene span ≈ {span3d:.2f} m)")
    if legend_handles:
        ax.legend(handles=legend_handles, loc="best")
    return fig


def fig_rig_diagram(data: ReportData) -> plt.Figure | None:
    """Zoomed top view of the rig itself: reference camera + relative offsets.

    Each camera is drawn as a triangle whose apex sits on the camera centre and
    whose height points along the camera viewing direction.
    """
    assert data.rig is not None
    rig = data.rig
    others = [c for c in data.cameras if c.name != data.cameras[0].name]
    if not others:
        return None

    import cv2  # local import to avoid hard dependency at module load

    fig, ax = plt.subplots(figsize=(6.4, 4.6), constrained_layout=True)

    all_xy = np.array([[0.0, 0.0]] + [[rig.relative_poses[c.name][3],
                                       rig.relative_poses[c.name][5]] for c in others])
    span = max(np.ptp(all_xy[:, 0]), np.ptp(all_xy[:, 1]), 0.05)
    tri_size = span * 0.22

    def _label(text: str, x: float, y: float, fwd2: np.ndarray,
               color: str) -> None:
        """Place a label behind the camera (opposite its viewing direction)."""
        d = np.array([fwd2[0], fwd2[1]], dtype=np.float64)
        n = float(np.linalg.norm(d))
        d = d / n if n > 1e-12 else np.array([0.0, 1.0])
        off = -d * span * 0.30
        ax.annotate(text, xy=(x, y), xytext=(x + off[0], y + off[1]),
                    ha="center", va="center", fontsize=9, color=color)

    # Reference camera: apex at origin, viewing along its +Z (world frame)
    R0 = cv2.Rodrigues(rig.relative_poses[data.cameras[0].name][:3])[0]
    fwd0 = R0.T @ np.array([0.0, 0.0, 1.0])
    _camera_triangle(ax, 0.0, 0.0, fwd0[0], fwd0[2], tri_size, "#C1121F")
    _label(f"{data.cameras[0].name}\n(reference)", 0.0, 0.0,
           np.array([fwd0[0], fwd0[2]]), "#C1121F")

    for cam in others:
        rp = rig.relative_poses[cam.name]
        t = rp[3:]
        # baseline arrow from the reference camera to this camera
        ax.annotate("", xy=(t[0], t[2]), xytext=(0, 0),
                    arrowprops=dict(arrowstyle="-|>", color="#2A9D8F", lw=2.0))
        # oriented camera triangle: viewing direction = R_rel @ +Z in cam0 frame
        R_rel = cv2.Rodrigues(rp[:3])[0]
        fwd = R_rel.T @ np.array([0.0, 0.0, 1.0])
        _camera_triangle(ax, t[0], t[2], fwd[0], fwd[2], tri_size, "#2A9D8F")
        bl = rig.baselines_m[cam.name]
        ang = rig.rel_rotation_deg[cam.name]
        _label(f"{cam.name}\n|t|={bl:.4f} m, rot={ang:.2f}°", t[0], t[2],
               np.array([fwd[0], fwd[2]]), "#1D3557")

    pad = span * 0.45 + 0.02 + tri_size
    cx, cy = all_xy.mean(axis=0)
    ax.set_xlim(cx - span / 2 - pad, cx + span / 2 + pad)
    ax.set_ylim(cy - span / 2 - pad, cy + span / 2 + pad)
    ax.set_aspect("equal")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Z (m)")
    ax.set_title("Rig configuration (top view, world frame)")
    return fig


def fig_parameter_correlation(data: ReportData) -> plt.Figure | None:
    adjustment = data.adjustment
    if adjustment is None:
        return None
    estimated = np.flatnonzero(np.isfinite(np.diag(adjustment.correlation)))
    if estimated.size < 2:
        return None
    labels = [adjustment.parameter_names[index] for index in estimated]
    matrix = adjustment.correlation[np.ix_(estimated, estimated)]
    size = max(6.5, min(12.0, 0.32 * len(labels)))
    fig, ax = plt.subplots(figsize=(size, size * 0.85), constrained_layout=True)
    sns.heatmap(
        matrix, cmap="coolwarm", vmin=-1.0, vmax=1.0, center=0.0,
        xticklabels=labels, yticklabels=labels, square=True,
        cbar_kws={"label": "correlation coefficient"}, ax=ax,
    )
    ax.tick_params(axis="x", labelrotation=90, labelsize=7)
    ax.tick_params(axis="y", labelrotation=0, labelsize=7)
    ax.set_title("Marginal correlation of estimated camera/rig parameters")
    return fig


def fig_projection_uncertainty(data: ReportData, camera: CameraInfo) -> plt.Figure | None:
    result = projection_uncertainty_grid(data, camera)
    if result is None:
        return None
    xs, ys, grid = result
    fig, ax = plt.subplots(figsize=(8.6, 5.0), constrained_layout=True)
    positive = grid[np.isfinite(grid) & (grid > 0)]
    if positive.size == 0:
        plt.close(fig)
        return None
    vmin = max(float(np.percentile(positive, 5)), 1e-6)
    vmax = max(float(np.percentile(positive, 99)), vmin * 1.01)
    image = ax.imshow(
        grid, extent=[xs[0], xs[-1], ys[-1], ys[0]], aspect="equal",
        cmap="magma", interpolation="bilinear", norm=LogNorm(vmin=vmin, vmax=vmax),
    )
    records = _records_for(data, camera.name)
    ax.scatter([record.u_meas for record in records], [record.v_meas for record in records],
               s=2, alpha=0.12, color="#35D0BA", edgecolors="none", label="calibration support")
    fig.colorbar(image, ax=ax, shrink=0.84, pad=0.02).set_label(
        "1-sigma projection uncertainty (px, logarithmic scale)"
    )
    ax.set_xlabel("u (px)")
    ax.set_ylabel("v (px)")
    ax.set_title(f"{camera.name}: intrinsic projection-uncertainty map")
    ax.legend(loc="upper right", fontsize=7)
    return fig


def fig_uncertainty_vs_distance(data: ReportData) -> plt.Figure | None:
    depths = np.array([record.depth_m for record in data.records if record.depth_m is not None])
    # Rig-only figure: the curves are defined relative to the reference camera,
    # whose pose is held fixed, so a single camera has nothing meaningful to plot.
    if data.adjustment is None or data.rig is None or depths.size == 0 or len(data.cameras) < 2:
        return None
    near = max(float(np.percentile(depths, 2)), 1e-3)
    far = max(float(np.percentile(depths, 98)), near * 1.05)
    distances = np.linspace(near, far, 30)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.1, 3.8), constrained_layout=True)
    plotted = False
    for camera in data.cameras:
        uncertainty = projection_uncertainty_vs_distance(data, camera, distances)
        if uncertainty is not None:
            ax1.plot(distances, uncertainty, lw=1.8, label=camera.name,
                     color=_camera_color(camera.name))
            plotted = True
    ax1.set_xlabel("distance on reference-camera optical axis (m)")
    ax1.set_ylabel("1-sigma projection uncertainty (px)")
    ax1.set_title("Joint calibration projection uncertainty")
    if plotted:
        ax1.legend()

    ref = data.cameras[0]
    plotted_tri = False
    for camera in data.cameras[1:]:
        curve = triangulation_uncertainty_vs_distance(data, camera, distances)
        if curve is None:
            continue
        ax2.plot(curve.distances_m, curve.sigma_z_m * 1000.0, lw=1.8,
                 label=f"{ref.name}-{camera.name}: depth", color=_camera_color(camera.name))
        transverse = np.sqrt(curve.sigma_x_m ** 2 + curve.sigma_y_m ** 2) * 1000.0
        ax2.plot(curve.distances_m, transverse, lw=1.2, ls="--",
                 label=f"{ref.name}-{camera.name}: transverse", color=_camera_color(camera.name))
        plotted_tri = True
    ax2.set_xlabel("distance on reference-camera optical axis (m)")
    ax2.set_ylabel("1-sigma triangulation uncertainty (mm)")
    ax2.set_title("Expected stereo coordinate uncertainty")
    if plotted_tri:
        ax2.legend(fontsize=7)
    if not plotted and not plotted_tri:
        plt.close(fig)
        return None
    return fig


def fig_pose_and_viewing_coverage(data: ReportData) -> plt.Figure | None:
    if data.rig is None or not data.rig.rig_poses:
        return None
    from scipy.spatial.transform import Rotation

    euler = []
    for pose in data.rig.rig_poses.values():
        rotation = cv2.Rodrigues(pose[:3])[0]
        euler.append(Rotation.from_matrix(rotation).as_euler("xyz", degrees=True))
    euler = np.asarray(euler)
    depths = np.array([record.depth_m for record in data.records if record.depth_m is not None])
    off_axis = np.degrees(np.arctan(np.array([record.r_norm for record in data.records])))
    image_scale = np.array([
        0.5 * (next(c for c in data.cameras if c.name == record.camera).intrinsics[0]
               + next(c for c in data.cameras if c.name == record.camera).intrinsics[1]) / record.depth_m
        for record in data.records if record.depth_m is not None and record.depth_m > 0
    ])

    fig, axes = plt.subplots(2, 2, figsize=(8.8, 6.0), constrained_layout=True)
    for column, label, color in zip(range(3), ("roll", "pitch", "yaw"), _PALETTE[:3]):
        sns.histplot(euler[:, column], bins=24, element="step", fill=False,
                     ax=axes[0, 0], color=color, label=label)
    axes[0, 0].set_xlabel("reference-camera orientation (deg)")
    axes[0, 0].set_title("Pose-orientation diversity")
    axes[0, 0].legend()
    sns.histplot(depths, bins=30, ax=axes[0, 1], color="#457B9D", edgecolor="none")
    axes[0, 1].set_xlabel("target depth (m)")
    axes[0, 1].set_title("Range coverage")
    sns.histplot(image_scale, bins=30, ax=axes[1, 0], color="#2A9D8F", edgecolor="none")
    axes[1, 0].set_xlabel("approximate image scale (px/m)")
    axes[1, 0].set_title("Image-scale coverage")
    sns.histplot(off_axis, bins=30, ax=axes[1, 1], color="#E76F51", edgecolor="none")
    axes[1, 1].set_xlabel("viewing-ray off-axis angle (deg)")
    axes[1, 1].set_title("Field-angle coverage")
    fig.suptitle("Network pose and viewing geometry", fontsize=12.5, fontweight="bold")
    return fig


def fig_camera_target_connectivity(data: ReportData) -> plt.Figure:
    cameras = [camera.name for camera in data.cameras]
    targets = sorted({record.target_id for record in data.records})
    camera_index = {name: index for index, name in enumerate(cameras)}
    target_index = {target: index for index, target in enumerate(targets)}
    matrix = np.zeros((len(cameras), len(targets)), dtype=np.int64)
    frames_by_camera_target: Dict[tuple[str, int], set[str]] = {}
    for record in data.records:
        matrix[camera_index[record.camera], target_index[record.target_id]] += 1
        frames_by_camera_target.setdefault((record.camera, record.target_id), set()).add(record.frame)

    shared = np.zeros((len(cameras), len(cameras)), dtype=np.int64)
    for ia, camera_a in enumerate(cameras):
        for ib, camera_b in enumerate(cameras):
            shared[ia, ib] = sum(
                bool(frames_by_camera_target.get((camera_a, target), set())
                     & frames_by_camera_target.get((camera_b, target), set()))
                for target in targets
            )
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.2, 3.8), constrained_layout=True,
                                   gridspec_kw={"width_ratios": [2.1, 1.0]})
    sns.heatmap(matrix, cmap="Blues", ax=ax1, cbar_kws={"label": "observations"},
                yticklabels=cameras, xticklabels=False)
    ax1.set_xlabel(f"targets ({len(targets)}, ordered by ID)")
    ax1.set_ylabel("camera")
    ax1.set_title("Camera-target connectivity matrix")
    sns.heatmap(shared, annot=True, fmt="d", cmap="Greens", ax=ax2,
                xticklabels=cameras, yticklabels=cameras, cbar=False)
    ax2.set_title("Targets co-visible in >=1 frame")
    return fig


def fig_rig_pair_quality(pair: RigPairStats) -> plt.Figure:
    fig, axes = plt.subplots(2, 2, figsize=(8.8, 6.0), constrained_layout=True)
    datasets = [
        (pair.sampson_px, "Sampson distance", "px"),
        (pair.vertical_disparity_px, f"Signed rectified {pair.rectified_disparity_axis} disparity", "px"),
        (pair.intersection_angle_deg, "Stereo intersection angle", "deg"),
        (pair.baseline_depth_ratio, "Baseline-to-depth ratio", "B/Z"),
    ]
    for axis, (values, title, unit) in zip(axes.flat, datasets):
        if values.size:
            sns.histplot(values, bins=35, kde=values.size > 20, ax=axis,
                         color="#457B9D", edgecolor="none")
            axis.axvline(np.median(values), color=WARN, ls="--", lw=1.1,
                        label=f"median {np.median(values):.3g} {unit}")
            axis.legend(fontsize=7)
        else:
            axis.text(0.5, 0.5, "not available", transform=axis.transAxes,
                      ha="center", va="center", color="#777777")
        axis.set_xlabel(unit)
        axis.set_title(title)
    fig.suptitle(
        f"Rig-pair internal quality: {pair.camera_a} - {pair.camera_b}",
        fontsize=12.5, fontweight="bold",
    )
    return fig


def fig_convergence(data: ReportData) -> plt.Figure | None:
    histories = data.bundle_history
    if not histories and data.solver is not None:
        histories = [("Final bundle", data.solver)]
    histories = [(label, solver) for label, solver in histories if solver.cost_history]
    if not histories:
        return None
    colors = ("#264653", "#E76F51", "#2A9D8F", "#8A5A9E", "#E9C46A")
    linestyles = ("-", "--", "-.", ":", (0, (5, 2)))
    fig, ax = plt.subplots(figsize=(8.6, 4.2), constrained_layout=True)
    for index, (label, solver) in enumerate(histories):
        history = np.asarray(solver.cost_history, dtype=np.float64)
        ax.semilogy(
            np.arange(history.size), history,
            color=colors[index % len(colors)],
            linestyle=linestyles[index % len(linestyles)],
            marker="o", markersize=2.8, linewidth=1.5, label=label,
        )
    ax.set_xlabel("solver iteration")
    ax.set_ylabel("total cost")
    ax.set_title("Bundle-adjustment cost by solver stage")
    ax.legend(fontsize=8)
    ax.grid(True, which="both", alpha=0.2)
    return fig


def fig_checkpoint_accuracy(data: ReportData) -> plt.Figure | None:
    quality = data.checkpoint_quality
    if not quality:
        return None
    successful = [item for item in quality.get("targets", []) if item.get("status") == "ok"]
    if not successful:
        return None
    ids = np.asarray([item["target_id"] for item in successful])
    errors_mm = np.asarray([item["error_xyz_m"] for item in successful]) * 1000.0
    magnitudes_mm = np.linalg.norm(errors_mm, axis=1)
    angles = np.asarray([item.get("intersection_angle_deg", np.nan) for item in successful])
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.0), constrained_layout=True)
    positions = np.arange(len(ids), dtype=float)
    bar_width = 0.25
    for index, (label, color) in enumerate(zip("XYZ", ("#457B9D", "#E09F3E", "#8A5A9E"))):
        axes[0].bar(
            positions + (index - 1) * bar_width, errors_mm[:, index],
            width=bar_width, label=label, color=color,
        )
    axes[0].axhline(0.0, color="#333333", lw=0.8)
    axes[0].set_xlabel("checkpoint target ID")
    axes[0].set_xticks(positions, [str(value) for value in ids], rotation=45, ha="right")
    axes[0].set_ylabel("signed error (mm)")
    axes[0].set_title("Independent checkpoint component errors")
    axes[0].legend(fontsize=8)
    axes[1].scatter(angles, magnitudes_mm, c=ids, cmap="viridis", s=30)
    axes[1].set_xlabel("maximum ray-intersection angle (deg)")
    axes[1].set_ylabel("3D error (mm)")
    axes[1].set_title("Accuracy versus intersection geometry")
    return fig
