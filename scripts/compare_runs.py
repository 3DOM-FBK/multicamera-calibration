"""Compare detections and calibration results of two run_combined output folders.

Usage:
    python scripts/compare_runs.py OLD_OUTPUT_DIR NEW_OUTPUT_DIR [--write-json PATH]

Detections are read from each camera's target_refinement_cache.json (raw
detector centres) and target_detections.txt (final, possibly refined
centres).  Calibration figures come from combined_summary.json.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def load_detections(run: Path) -> dict[str, dict[tuple[str, int], dict]]:
    out: dict[str, dict[tuple[str, int], dict]] = {}
    for camera_dir in sorted(p for p in run.iterdir() if p.is_dir() and (p / "target_detections.txt").exists()):
        rows: dict[tuple[str, int], dict] = {}
        for line in (camera_dir / "target_detections.txt").read_text(encoding="utf-8").splitlines()[1:]:
            parts = line.split()
            if len(parts) == 4:
                rows[(parts[0], int(parts[1]))] = {"final": np.array([float(parts[2]), float(parts[3])])}
        cache = camera_dir / "target_refinement_cache.json"
        if cache.exists():
            payload = json.loads(cache.read_text(encoding="utf-8"))
            for image, record in payload.get("images", {}).items():
                for target, item in record.get("observations", {}).items():
                    key = (image, int(target))
                    if key in rows:
                        rows[key]["raw"] = np.asarray(item["raw_center"], dtype=float)
                        rows[key]["method"] = (item.get("refinement") or {}).get("method")
        out[camera_dir.name] = rows
    return out


def stats(values) -> dict:
    v = np.asarray(values, dtype=float)
    if not v.size:
        return {"n": 0}
    return {"n": int(v.size), "median": float(np.median(v)), "p95": float(np.percentile(v, 95)),
            "max": float(v.max())}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("old", type=Path)
    parser.add_argument("new", type=Path)
    parser.add_argument("--write-json", type=Path, default=None)
    args = parser.parse_args()

    old, new = load_detections(args.old), load_detections(args.new)
    report: dict = {"detections": {}, "calibration": {}}
    print(f"OLD: {args.old}\nNEW: {args.new}\n")
    print("=== Detections (all cached frames) ===")
    for camera in sorted(set(old) | set(new)):
        a, b = old.get(camera, {}), new.get(camera, {})
        common = set(a) & set(b)
        gained, lost = set(b) - set(a), set(a) - set(b)
        frames_a = {k[0] for k in a}
        frames_b = {k[0] for k in b}
        per_target_gain = defaultdict(int)
        for _, target in gained:
            per_target_gain[target] += 1
        per_target_loss = defaultdict(int)
        for _, target in lost:
            per_target_loss[target] += 1
        raw_shift = [np.linalg.norm(a[k]["raw"] - b[k]["raw"]) for k in common if "raw" in a[k] and "raw" in b[k]]
        final_shift = [np.linalg.norm(a[k]["final"] - b[k]["final"]) for k in common]
        methods_a = defaultdict(int)
        methods_b = defaultdict(int)
        for k in a:
            methods_a[a[k].get("method")] += 1
        for k in b:
            methods_b[b[k].get("method")] += 1
        entry = {
            "old_detections": len(a), "new_detections": len(b), "common": len(common),
            "gained": len(gained), "lost": len(lost),
            "old_frames": len(frames_a), "new_frames": len(frames_b),
            "old_ids": len({k[1] for k in a}), "new_ids": len({k[1] for k in b}),
            "ids_only_new": sorted({k[1] for k in b} - {k[1] for k in a}),
            "ids_only_old": sorted({k[1] for k in a} - {k[1] for k in b}),
            "top_gained_ids": sorted(per_target_gain.items(), key=lambda kv: -kv[1])[:10],
            "top_lost_ids": sorted(per_target_loss.items(), key=lambda kv: -kv[1])[:10],
            "raw_centre_shift_px": stats(raw_shift),
            "final_centre_shift_px": stats(final_shift),
            "old_refinement_methods": dict(methods_a), "new_refinement_methods": dict(methods_b),
        }
        report["detections"][camera] = entry
        print(f"[{camera}] detections old={len(a)} new={len(b)} (+{len(gained)} / -{len(lost)}, "
              f"{len(common)} common); frames old={len(frames_a)} new={len(frames_b)}; "
              f"IDs old={entry['old_ids']} new={entry['new_ids']}")
        print(f"   IDs only in new: {entry['ids_only_new']}  IDs only in old: {entry['ids_only_old']}")
        print(f"   most gained IDs: {entry['top_gained_ids']}")
        print(f"   most lost IDs:   {entry['top_lost_ids']}")
        print(f"   raw-centre shift on common detections: {entry['raw_centre_shift_px']}")
        print(f"   final-centre shift on common detections: {entry['final_centre_shift_px']}")
        print(f"   refinement methods old={dict(methods_a)} new={dict(methods_b)}")

    print("\n=== Calibration ===")
    summaries = {}
    for label, run in (("old", args.old), ("new", args.new)):
        path = run / "combined_summary.json"
        summaries[label] = json.loads(path.read_text(encoding="utf-8")) if path.exists() else None
    if all(summaries.values()):
        so, sn = summaries["old"], summaries["new"]
        for key in ("total_observations", "total_tracks", "total_frames", "mean_reprojection_error",
                    "rms_reprojection_error"):
            print(f"   {key:26s} old={so.get(key)!s:>22} new={sn.get(key)!s:>22}")
        qo, qn = so.get("adjustment_quality", {}), sn.get("adjustment_quality", {})
        for key in ("sigma0_px", "dof", "correlation_condition_number"):
            print(f"   {key:26s} old={qo.get(key)!s:>22} new={qn.get(key)!s:>22}")
        po = {p["name"]: p for p in qo.get("parameters", [])}
        pn = {p["name"]: p for p in qn.get("parameters", [])}
        print("   parameter                       old            new          diff    sigma_old  sigma_new  |diff|/sigma_new")
        rows = []
        for name in po:
            if name not in pn or po[name]["stddev"] is None or pn[name]["stddev"] is None:
                continue
            diff = pn[name]["final"] - po[name]["final"]
            ratio = abs(diff) / pn[name]["stddev"] if pn[name]["stddev"] else float("nan")
            rows.append({"name": name, "old": po[name]["final"], "new": pn[name]["final"], "diff": diff,
                         "sigma_old": po[name]["stddev"], "sigma_new": pn[name]["stddev"], "ratio": ratio})
            print(f"   {name:24s} {po[name]['final']:14.6f} {pn[name]['final']:14.6f} {diff:12.6f} "
                  f"{po[name]['stddev']:10.6f} {pn[name]['stddev']:10.6f} {ratio:8.2f}")
        report["calibration"]["parameters"] = rows
        co = (so.get("independent_checkpoint_quality") or {}).get("summary", {})
        cn = (sn.get("independent_checkpoint_quality") or {}).get("summary", {})
        for key in ("selected_targets", "xyz_reconstructed", "rmse_3d_m", "median_3d_m", "p95_3d_m", "max_3d_m"):
            print(f"   checkpoint {key:15s} old={co.get(key)!s:>22} new={cn.get(key)!s:>22}")
        wo, wn = co.get("withheld_reprojection", {}), cn.get("withheld_reprojection", {})
        for key in ("n", "rms_2d_px", "p95_2d_px", "object_space_rms_m"):
            print(f"   withheld reproj {key:10s} old={wo.get(key)!s:>22} new={wn.get(key)!s:>22}")
        split_o = (so.get("independent_checkpoint_quality") or {}).get("split", {})
        split_n = (sn.get("independent_checkpoint_quality") or {}).get("split", {})
        same = split_o.get("checkpoint_target_ids") == split_n.get("checkpoint_target_ids")
        print(f"   checkpoint IDs identical: {same}")
        if not same:
            print(f"      old: {split_o.get('checkpoint_target_ids')}\n      new: {split_n.get('checkpoint_target_ids')}")
        report["calibration"]["summary_old"] = {k: so.get(k) for k in ("total_observations", "rms_reprojection_error")}
        report["calibration"]["summary_new"] = {k: sn.get(k) for k in ("total_observations", "rms_reprojection_error")}
        report["calibration"]["checkpoints_old"] = co
        report["calibration"]["checkpoints_new"] = cn
    if args.write_json:
        args.write_json.write_text(json.dumps(report, indent=2, default=float), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
