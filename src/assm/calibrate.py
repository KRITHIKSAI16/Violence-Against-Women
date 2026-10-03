"""M9 calibration: how do the thresholds behave on the real corpus? (read-only, changes no config)

Runs the M9 analysis over every clip that has cached tracks and prints:
  1. per-category rates: no-pair, flagged, FOLLOW / HOVER / CORNER / ESCALATION / ordered progression
  2. how long each state lasts in Normal clips vs the other categories (quantiles of segment length)
  3. a one-at-a-time threshold sweep: for each value of a parameter, the flag rate on Normal clips
     (false alarms) and on all other clips, and the gap between them
  4. suggested values: the largest gap among values that keep Normal's flag rate under the target

Honest note: this tunes thresholds using the Normal label (Normal = should rarely flag). It is a calibration
of a rule-based gate, not a trained classifier, but the numbers it prints are in-sample. The independent check is
the human review sheet from the report step.

Usage:  python -m src.assm.calibrate [--config CFG] [--workers N] [--target 0.15] [--no-sweep]
"""
import argparse
import csv
import json
import logging
import os
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from src.assm.gate import CORNER, ESCALATION, FOLLOW, HOVER, analyze_clip
from src.assm.track_poses import load_tracks
from src.config import load_config, parse_overrides, resolve_path

log = logging.getLogger(__name__)

SWEEP = {
    "follow_min_s": [2.0, 3.0, 4.0, 6.0, 8.0],
    "follow_max_d": [2.0, 3.0, 4.0],
    "follow_cos": [0.3, 0.5, 0.7],
    "hover_min_s": [2.0, 3.0, 5.0, 8.0],
    "hover_max_d": [1.0, 1.5, 2.0],
    "corner_min_s": [1.0, 1.5, 2.5, 4.0],
    "corner_blocked_speed": [0.2, 0.3, 0.5],
    "min_track_conf": [0.0, 0.3, 0.35, 0.4, 0.5],
}


def evaluate(items, cfg):
    """items: list of (category, tracks dict). Returns list of gate dicts (no files written)."""
    return [analyze_clip(t, cfg, f"{cat}_{k}", cat)[0] for k, (cat, t) in enumerate(items)]


def rates(gates):
    """Per-category rate dict + pooled Normal / non-Normal flag rates."""
    by = defaultdict(list)
    for g in gates:
        by[g["category"]].append(g)
    out = {}
    for cat, lst in by.items():
        n = len(lst)

        def has(s):
            return sum(1 for g in lst if s in g["states_seen"]) / n
        out[cat] = {"n": n, "no_pair": sum(g["no_interaction"] for g in lst) / n,
                    "flagged": sum(g["flag"] for g in lst) / n, "FOLLOW": has(FOLLOW), "HOVER": has(HOVER),
                    "CORNER": has(CORNER), "ESCALATION": has(ESCALATION),
                    "ordered": sum(g["ordered_progression"] for g in lst) / n}
    normal = [g for g in gates if g["category"] == "Normal"]
    other = [g for g in gates if g["category"] != "Normal"]
    out["_normal_flag"] = sum(g["flag"] for g in normal) / len(normal) if normal else float("nan")
    out["_other_flag"] = sum(g["flag"] for g in other) / len(other) if other else float("nan")
    return out


def _variant(args):
    cfg, paths = args
    items = [(cat, load_tracks(p)) for cat, p in paths]
    r = rates(evaluate(items, cfg))
    return r["_normal_flag"], r["_other_flag"]


def sweep(paths, cfg, workers, target):
    """One-at-a-time sweep over SWEEP. Returns (rows, best-per-parameter)."""
    jobs, labels = [], []
    for param, values in SWEEP.items():
        for v in values:
            jobs.append(({**cfg, param: v}, paths))
            labels.append((param, v))
    if workers > 1:
        with ProcessPoolExecutor(workers) as ex:
            res = list(ex.map(_variant, jobs))
    else:
        res = [_variant(j) for j in jobs]
    rows = [{"param": p, "value": v, "normal_flag": n, "other_flag": o, "gap": o - n}
            for (p, v), (n, o) in zip(labels, res)]
    best = {}
    for param in SWEEP:
        ok = [r for r in rows if r["param"] == param and r["normal_flag"] <= target]
        best[param] = max(ok, key=lambda r: r["gap"]) if ok else None
    return rows, best


def duration_quantiles(gates):
    d = defaultdict(lambda: defaultdict(list))
    for g in gates:
        grp = "Normal" if g["category"] == "Normal" else "Other"
        for s in g["segments"]:
            d[s["state"]][grp].append(s["dur_s"])
    return d


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    ap = argparse.ArgumentParser(description="M9 calibration report")
    ap.add_argument("--config", default=None)
    ap.add_argument("--workers", type=int, default=min(4, os.cpu_count() or 1))
    ap.add_argument("--target", type=float, default=0.15, help="max acceptable flag rate on Normal clips")
    ap.add_argument("--no-sweep", action="store_true")
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE", help="start from these gate values")
    args = ap.parse_args()

    cfg0 = load_config(args.config)
    cfg0["gate"] = {**cfg0["gate"], **parse_overrides(args.set)}
    cfg = {**cfg0["gate"], "kpt_conf": cfg0["assm"]["kpt_conf"], "corridor": cfg0["assm"]["corridor"]}
    clean = json.loads(resolve_path(cfg0["preprocess"]["clean_manifest_path"]).read_text("utf-8"))
    tdir = resolve_path(cfg0["assm"]["tracks_dir"])
    paths = [(c["category"], tdir / c["category"] / f"{c['clip_id']}.npz") for c in clean["clips"]]
    paths = [(cat, p) for cat, p in paths if p.exists()]
    print(f"{len(paths)} clips with tracks\n")

    items = [(cat, load_tracks(p)) for cat, p in paths]
    gates = evaluate(items, cfg)
    r = rates(gates)
    print("1) Behavior rates per category (share of clips)")
    print(f"{'category':<16}{'clips':>6}{'no-pair':>9}{'flagged':>9}{'FOLLOW':>8}{'HOVER':>7}{'CORNER':>8}{'ESCAL':>7}{'ordered':>9}")
    for cat in sorted(k for k in r if not k.startswith("_")):
        x = r[cat]
        print(f"{cat:<16}{x['n']:>6}{x['no_pair']:>9.0%}{x['flagged']:>9.0%}{x['FOLLOW']:>8.0%}{x['HOVER']:>7.0%}"
              f"{x['CORNER']:>8.0%}{x['ESCALATION']:>7.0%}{x['ordered']:>9.0%}")
    print(f"\nFlag rate: Normal {r['_normal_flag']:.0%}  vs all other categories {r['_other_flag']:.0%}")

    print("\n2) Segment length in seconds (median / 90th percentile), Normal vs other categories")
    for state, grp in sorted(duration_quantiles(gates).items()):
        def q(v):
            return f"{np.median(v):.1f}/{np.percentile(v, 90):.1f} (n={len(v)})" if v else "-"
        print(f"  {state:<11} Normal {q(grp.get('Normal', [])):<24} Other {q(grp.get('Other', []))}")

    if not args.no_sweep:
        print(f"\n3) One-at-a-time sweep (target: Normal flag rate <= {args.target:.0%})")
        rows, best = sweep(paths, cfg, args.workers, args.target)
        print(f"{'parameter':<22}{'value':>7}{'Normal flagged':>16}{'Other flagged':>15}{'gap':>7}")
        for row in rows:
            cur = " <- current" if abs(cfg[row["param"]] - row["value"]) < 1e-9 else ""
            print(f"{row['param']:<22}{row['value']:>7}{row['normal_flag']:>16.0%}{row['other_flag']:>15.0%}"
                  f"{row['gap']:>+7.0%}{cur}")
        print("\n4) Suggested values (largest gap with Normal flag rate under target):")
        for p, b in best.items():
            print(f"  {p:<22} " + (f"{b['value']}   (Normal {b['normal_flag']:.0%}, Other {b['other_flag']:.0%})" if b
                                    else "no value meets the target - leave as is and check the video review"))
        out = resolve_path(cfg0["gate"]["gate_dir"]) / "calibration_sweep.csv"
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        print(f"\nSweep saved to {out}")
    print("\nNote: in-sample tuning using the Normal label. Confirm with the human review sheet before trusting the numbers.")


if __name__ == "__main__":
    main()
