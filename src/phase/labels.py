"""Layer 2, step 2: temporal labels (what happens when in a clip) and how they turn into window targets.

A label row per clip: `act_start_s` (physical violence begins; blank = none visible), `buildup_start_s` (the aggressor's approach / following /
threat begins; blank = none, i.e. a direct attack or no incident), `source` = human | vlm | rule. Human rows always win over proposals.
Window target: ACT after act_start, BUILDUP between buildup_start and act_start, BACKGROUND otherwise (also for the whole of a clip without act).
"""
import csv
from pathlib import Path

import numpy as np

from src.phase.decode import ACT, BG, BUILD

FIELDS = ["clip_id", "act_start_s", "buildup_start_s", "source", "note"]
RANK = {"human": 3, "vlm": 2, "rule": 1}


def _f(x):
    x = "" if x is None else str(x).strip()
    return None if x == "" or x.lower() in ("none", "nan") else float(x)


def read_labels(path):
    """CSV -> {clip_id: {act_start_s, buildup_start_s, source, note}}; a clip listed twice keeps the higher-ranked source."""
    out = {}
    if not Path(path).exists():
        return out
    with open(path, newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            row = {"act_start_s": _f(r.get("act_start_s")), "buildup_start_s": _f(r.get("buildup_start_s")),
                   "source": (r.get("source") or "human").strip(), "note": r.get("note", "")}
            old = out.get(r["clip_id"])
            if old is None or RANK.get(row["source"], 0) >= RANK.get(old["source"], 0):
                out[r["clip_id"]] = row
    return out


def write_labels(path, labels):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        for cid in sorted(labels):
            r = labels[cid]
            w.writerow({"clip_id": cid, "act_start_s": "" if r["act_start_s"] is None else r["act_start_s"],
                        "buildup_start_s": "" if r["buildup_start_s"] is None else r["buildup_start_s"],
                        "source": r.get("source", "human"), "note": r.get("note", "")})


def merge(*label_sets):
    """Human beats vlm beats rule, whatever the argument order; equal rank keeps the first seen."""
    out = {}
    for ls in label_sets:
        for cid, r in ls.items():
            if cid not in out or RANK.get(r["source"], 0) > RANK.get(out[cid]["source"], 0):
                out[cid] = r
    return out


def window_targets(t_centers, act_start_s, buildup_start_s):
    """Per-window target (0 background, 1 buildup, 2 act) from a clip's labels."""
    tc = np.asarray(t_centers, float)
    y = np.full(len(tc), BG, int)
    if act_start_s is None:
        return y
    if buildup_start_s is not None and buildup_start_s < act_start_s:
        y[(tc >= buildup_start_s) & (tc < act_start_s)] = BUILD
    y[tc >= act_start_s] = ACT
    return y
