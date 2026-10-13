# CCT Calibration

This project calibrates one or more cameras from CCT target detections using a mix of detection, initialization, and bundle adjustment. The main workflows are:

- `run_combined.py` for multi-camera rigs
- `run_cct_calibration.py` for single-camera calibration

The repository is designed to work well both with a known 3D target reference and with a fully automatic SfM-based initialization path.

## Requirements

- Python 3.10 or newer
- `uv` for dependency management and execution

## Installation

Run this from the repository root:

```bash
uv sync
```

You can then invoke the scripts without activating a virtual environment:

```bash
uv run python run_cct_calibration.py --help
uv run python run_combined.py --help
```

## Data layout

### Multi-camera rig

For multi-camera calibration, place one subfolder per camera under a common root:

```text
images/
  cam0/
    frame000.jpg
    frame001.jpg
  cam1/
    frame000.jpg
    frame001.jpg
```

The image file names should match across cameras so the pipeline can associate them by frame.

### Single camera

For single-camera calibration, use a flat folder of images (the legacy `photos_*` layout is only supported with `--detect-only`).

Example flat-folder layout:

```text
reflex/
  _CAL9476.JPG
  _CAL9477.JPG
  ...
```

## Reference target file

If you have metric 3D target coordinates, pass them with `--targets3d`.

Supported input formats are:

- simple whitespace- or tab-separated text: `target_id x y z`
- Metashape-style exports with columns such as `Label actual`, `X`, `Y`, and `Z`

When `--targets3d` is provided, the pipeline can use those known coordinates to anchor the solution to a metric frame.

The reference file also defines the codebook: target IDs are then decoded by correlating the code ring with each valid code (all rotations), which recovers targets the generic decoder misses, including single-arc codes. Without `--targets3d` the generic decoder is used, unchanged (`--valid-ids-file` / `--valid-id-max` then only filter the decoded IDs).

Detection caches are stamped with the detector version; a cache from an older detector is reused with a warning, so pass `--force-detections` after a detector update.

## Quick start

### 1. Multi-camera calibration

```bash
uv run python run_combined.py \
  --image-root "images" \
  --output-dir "combined_output"
```

This runs the full pipeline:

1. detect CCT targets in each camera
2. initialize the rig with SfM
3. run bundle adjustment
4. write reports and calibration outputs

### 2. Multi-camera calibration with known target coordinates

```bash
uv run python run_combined.py \
  --image-root "images" \
  --output-dir "combined_output_reference" \
  --targets3d "reference.txt"
```

This is the preferred mode when you have a reliable reference file. It constrains the solution to the known metric target geometry and improves the accuracy of the final poses and intrinsics.

### 3. Multi-camera calibration without SfM

If SfM is unstable or fails on a difficult dataset, you can skip it entirely:

```bash
uv run python run_combined.py \
  --image-root "images" \
  --output-dir "combined_output_skip_sfm" \
  --targets3d "reference.txt" \
  --skip-sfm
```

This mode requires `--targets3d`. It bypasses `pycolmap`, uses the known 3D targets to initialize the camera poses, and then runs the remaining calibration steps.

### 4. Single-camera calibration

```bash
uv run python run_cct_calibration.py \
  --image-root "reflex" \
  --output-dir "reflex_output" \
  --targets3d "reference.txt"
```

This is the standard mono workflow for a flat image folder. A single camera is calibrated as the one-camera case of the combined pipeline: the same centre refinement, staged adjustment, covariance diagnostics, optional checkpoints and PDF report are used, and only rig-specific results (relative orientation, stereo diagnostics) are omitted. With `--targets3d`, the camera is initialized from the known targets (no SfM). Any option of `run_combined.py` (e.g. `--checkpoints`, `--observation-sigma-px`, `--evaluate-detections`, `--no-pdf-report`) can be added. `run_combined.py --image-root <folder of images>` is equivalent.

Calibrating without `--targets3d` is possible but not recommended and prints a warning: initialization then uses natural-feature SfM, targets become free object points, and the scale of a single-camera solution is arbitrary (only the interior orientation is meaningful).

### 5. Detection-only mode

```bash
uv run python run_cct_calibration.py \
  --image-root "reflex" \
  --output-dir "reflex_output" \
  --targets3d "reference.txt" \
  --detect-only
```

This runs detection only and exports the detections without performing pose initialization or bundle adjustment.

## Important flags

### Common flags

- `--image-root`: root folder with images. For multi-camera rigs this should contain camera subfolders such as `cam0/` and `cam1/`. For mono calibration it can be a folder of image files directly.
- `--output-dir`: destination folder for reports, summaries, caches, and plots.
- `--targets3d`: path to a known 3D target reference file. Use this when you want a metric calibration.
- `--checkpoints [RATIO]`: optionally withhold complete target IDs from target-based initialization and every calibration bundle for final independent accuracy checks. With no ratio it uses `0.2`; `--checkpoints 0` preserves the ordinary workflow. Checkpoint runs restore original detector coordinates from the cache so refined centers from an earlier target partition cannot leak into the experiment; only legacy or incomplete caches require a one-time redetection.
- `--checkpoint-seed`: reproducible integer seed for checkpoint selection (default `42`). The selected IDs and realised ratio are written to `checkpoint_split.json`.
- `--force-detections`: ignore existing detection cache files and rerun target detection.
- `--workers`: number of parallel image-processing workers for target detection, calibrated centre refinement and `--evaluate-detections` (default `0` = automatic, up to 8 limited by the CPU count; `1` = sequential). Results are identical; each worker holds one image in memory, so lower it for very large images on machines with little RAM.

### Multi-camera flags

- `--camera`: choose specific camera folders. Repeat it to select multiple cameras explicitly, for example `--camera cam0 --camera cam1`.
- `--known-baseline`: provide either the cam0–cam1 baseline magnitude (one value, metres) or its XYZ translation vector (three values); it initializes the SfM scale and applies the corresponding rig constraint.
- `--skip-sfm`: bypass the SfM stage and bootstrap from known 3D targets instead. Requires `--targets3d`. Intrinsics are initialized from a bounded, geometry-diverse subset (up to 24 frames, selected from at most 96 candidates); fixed-intrinsic PnP then registers the remaining retained frames before the full bundle adjustment.
- `--force-sfm`: ignore any existing SfM reconstruction cache and recompute it.
- `--min-detections`: minimum number of detections required for an image to be kept.
- `--min-shared`: minimum number of shared target observations required for initialization.
- `--max-reprojection-error`: threshold in pixels for rejecting weak initial observations.
- `--max-iterations`: maximum number of bundle-adjustment iterations.
- `--huber-delta`: Huber loss parameter used in the solver.
- `--loss-tolerance`: convergence threshold for stopping based on relative loss change.
- `--loss-patience`: number of stable iterations before early termination.
- `--outlier-mad-scale`: robust outlier threshold based on the median absolute deviation of object-space residuals.
- `--valid-ids-file`: optional file listing valid target IDs.
- `--valid-id-max`: use all target IDs from `0` to `N` as valid.
- `--max-id-hamming-distance`: allow decoded target IDs to be snapped to the nearest valid ID only within this Hamming-distance limit.

### Developer diagnostics

- `--evaluate-detections` (off by default, requires `--targets3d`): projects every reference target with the final calibration, reports per-camera detection recall and detections far from the projection of their ID, and attributes each miss to the detector stage that rejected it (`detection_evaluation.json`). It re-runs detection on images with misses, so it is slow; it is a consistency tool for detector development, not independent ground truth.

### Single-camera flags

- `--data-root`: root directory containing the legacy `photos_*` folders (`--detect-only` only).
- `--camera`: camera name for `--detect-only`; for calibration the camera is named after its image folder.
- `--detect-only`: run detection only and skip initialization and bundle adjustment.
- `--show-3d`: no longer available for calibration; inspect `<output-dir>/colmap` instead.

## Caching behavior

The pipeline reuses previous intermediate results where possible.

- Multi-camera detections are cached under each camera output folder as `target_detections.txt`.
- Multi-camera SfM is cached under the output directory in the `sfm/` folder.
- Single-camera detections are cached the same way, under `<output-dir>/<camera>/`.

Use `--force-detections` or `--force-sfm` when you want to rebuild those intermediate results from scratch.

## PDF calibration report

The multi-camera pipeline automatically generates `calibration_report.pdf` in the output directory. The report is aimed at photogrammetry experts and contains:

- **Summary** — residual statistics (mean/RMS/median/percentiles for pixel and object-space residuals), per-camera overview, and solution redundancy (unknowns, degrees of freedom, redundancy ratio, outlier-filter accounting).
- **Residual analysis** — signed u/v and radial/tangential statistics, standardized residuals, local redundancy, per-frame/per-target diagnostics, outlier selection, error histograms/CDFs, and bundle-adjustment convergence.
- **Parameter determinability** — initial/final/status tables, marginal standard deviations and 95% intervals, the full correlation heatmap, a condition indicator, and the a-posteriori variance factor. Supplying `--observation-sigma-px` also enables a chi-square consistency test.
- **Projection uncertainty** — intrinsic projection-uncertainty maps and joint calibration/triangulation uncertainty versus working distance.
- **Per-camera deep dive** — binned signed-vector fields, separate u/v/radial/tangential heatmaps, error-vs-radius trends, distortion profiles, and density-aware FoV coverage.
- **Network strength** — pose, range, image-scale and field-angle distributions; camera-target connectivity; local reliability; and explicit weak-frame/weak-target tables. Surface-incidence angles are reported as unavailable unless target normals become part of the input model.
- **Rig quality** (multi-camera only) — component-wise extrinsic uncertainty, shared support, Sampson error, rectified vertical disparity, stereo intersection angle, baseline/depth ratio, internal triangulation discrepancies, rig geometry and scene layout.

Pass `--no-pdf-report` to skip PDF generation. Use `--observation-sigma-px VALUE` to state the a-priori one-sigma precision of each image coordinate. The reporting code lives in `src/cct_calibration/reporting/`.

Pass `--save-report-plots` to also export every plot included in the PDF as a 600 dpi PNG in `<output-dir>/plots/`. Filenames are numbered in report order and include the plot title. The PDF itself keeps its usual resolution. This flag cannot be combined with `--no-pdf-report`.

## Output files

### Multi-camera outputs

Typical outputs include:

- `combined_summary.json`
- `report.txt`
- `calibration_report.pdf` — rich PDF report (see below)
- `camchain.yaml`
- `combined_convergence.png`
- `sfm/`
- `colmap/`
- per-camera detection folders and caches

### Single-camera outputs

The same files as for a multi-camera rig (`combined_summary.json`, `report.txt`, `calibration_report.pdf`, `camchain.yaml`, `colmap/`, detection folders and caches), without the rig-specific content.

The reports contain reprojection error in pixels and, when available, object-space error in meters.

## Suggested workflows

- Use `--targets3d` whenever you have a reliable 3D reference file.
- Use `--skip-sfm` when `pycolmap` is unstable or when the dataset is difficult.
- Use `--detect-only` to inspect the detections before running full calibration.
- Use `--force-detections` if you have changed the detector or the input images and want to rebuild the detections cache.

## Example commands

### Standard multi-camera run

```bash
uv run python run_combined.py \
  --image-root "images" \
  --output-dir "combined_output"
```

### Metric multi-camera run

```bash
uv run python run_combined.py \
  --image-root "images" \
  --output-dir "combined_output_reference" \
  --targets3d "reference.txt"
```

### Robust multi-camera run without SfM

```bash
uv run python run_combined.py \
  --image-root "images" \
  --output-dir "combined_output_skip_sfm" \
  --targets3d "reference.txt" \
  --skip-sfm
```

### Mono run on a flat image folder

```bash
uv run python run_cct_calibration.py \
  --image-root "reflex" \
  --output-dir "reflex_output" \
  --targets3d "reference.txt"
```

