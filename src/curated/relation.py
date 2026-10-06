"""Curated dataset: what is the relationship between two people at each moment? Concerning AND benign interactions get a name.

A bare "no cue found" is not an explanation, and "following" must not fire for two people who simply walk together. Each frame of a pair is given
one label (first rule that applies, in this order), then runs of one label become spans with their distances:

  contact_or_reach          hands at the other's torso, or a fast reach toward it
  follows                   lagged-path following (one retraces the other's path; side-by-side walking is excluded by that test)
  approaches_from_behind    one closes in on someone who is facing away
  approaches                one moves toward the other and the gap shrinks
  walking_together          close, both walking at similar speed, gap steady       (benign)
  standing_together         close and both nearly still, or facing each other      (benign)
  moving_apart              the gap grows                                          (neutral)
  close_unclear             within 1.5 m with none of the above
  apart                     none of the above

Thresholds are heuristics (listed in THRESH), not tuned on any data set; the jury report prints them. Distances are approximate meters.
"""
import numpy as np

CONCERN = {"contact_or_reach", "follows", "approaches_from_behind", "approaches"}
BENIGN = {"walking_together", "standing_together"}
LABELS = ["contact_or_reach", "follows", "approaches_from_behind", "approaches", "walking_together", "standing_together", "moving_apart", "close_unclear", "apart"]
THRESH = {"approach_toward_ms": 0.3, "approach_closing_ms": 0.3, "reach_ms": 1.5, "together_max_m": 3.0, "walking_ms": 0.4, "speed_ratio_min": 0.6,
          "steady_closing_ms": 0.3, "still_ms": 0.5, "apart_closing_ms": -0.3, "close_unclear_m": 1.5, "min_span_s": 0.4, "merge_gap_s": 0.3}


def _nz(x):
    return np.nan_to_num(np.asarray(x, float), nan=0.0)


def frame_labels(a, thresh=THRESH):
    """Pair arrays dict -> array of label names, one per frame (strings)."""
    d = np.asarray(a["dist_m"], float)
    n = len(d)
    clo = _nz(a["closing_ms"])
    ti, tj = _nz(a["toward_i"]), _nz(a["toward_j"])
    si, sj = _nz(a["speed_i"]), _nz(a["speed_j"])
    valid = np.isfinite(d)
    near = valid & (d < thresh["together_max_m"])
    reach = np.fmax(_nz(a["reach_i_to_j_ms"]), _nz(a["reach_j_to_i_ms"]))
    out = np.full(n, "apart", dtype=object)
    conds = [
        ("contact_or_reach", np.asarray(a["contact"], bool) | (reach > thresh["reach_ms"])),
        ("follows", np.asarray(a["follow_i_j"], bool) | np.asarray(a["follow_j_i"], bool)),
        ("approaches_from_behind", np.asarray(a["approach_behind_i_j"], bool) | np.asarray(a["approach_behind_j_i"], bool)),
        ("approaches", valid & (d < 8.0) & (np.fmax(ti, tj) > thresh["approach_toward_ms"]) & (clo > thresh["approach_closing_ms"])),
        ("walking_together", near & (np.fmin(si, sj) > thresh["walking_ms"]) & (np.fmin(si, sj) / np.maximum(np.fmax(si, sj), 1e-6) > thresh["speed_ratio_min"])
         & (np.abs(clo) < thresh["steady_closing_ms"])),
        ("standing_together", near & (((si < thresh["still_ms"]) & (sj < thresh["still_ms"])) | np.asarray(a["mutual_facing"], bool))),
        ("moving_apart", valid & (clo < thresh["apart_closing_ms"])),
        ("close_unclear", valid & (d < thresh["close_unclear_m"])),
    ]
    for name, mask in reversed(conds):                      # first rule in the list wins: write the last one first
        out[mask] = name
    out[~valid] = "none"
    return out


def spans(labels, fps, dist=None, closing=None, thresh=THRESH):
    """Runs of one label -> [{label, start_s, end_s, dist_start_m, dist_end_m, mean_closing_ms}]; short runs are absorbed, tiny gaps are bridged."""
    labels = np.asarray(labels, dtype=object).copy()
    n = len(labels)
    gap = max(1, round(thresh["merge_gap_s"] * fps))
    k = 0
    while k < n:                                            # bridge a short run of another label that sits between two runs of the same label
        j = k
        while j < n and labels[j] == labels[k]:
            j += 1
        if 0 < k and j < n and j - k <= gap and labels[k - 1] == labels[j] and labels[k] != "none":
            labels[k:j] = labels[j]
        k = j
    out, k = [], 0
    min_len = max(1, round(thresh["min_span_s"] * fps))
    while k < n:
        j = k
        while j < n and labels[j] == labels[k]:
            j += 1
        if labels[k] != "none" and j - k >= min_len:
            seg = slice(k, j)
            d = None if dist is None else np.asarray(dist, float)[seg]
            c = None if closing is None else np.asarray(closing, float)[seg]
            out.append({"label": str(labels[k]), "start_s": round(k / fps, 2), "end_s": round(j / fps, 2),
                        "dist_start_m": None if d is None or not np.isfinite(d).any() else round(float(d[np.isfinite(d)][0]), 2),
                        "dist_end_m": None if d is None or not np.isfinite(d).any() else round(float(d[np.isfinite(d)][-1]), 2),
                        "mean_closing_ms": None if c is None or not np.isfinite(c).any() else round(float(np.nanmean(c)), 2)})
        k = j
    return out


def pair_spans(a, fps, thresh=THRESH):
    return spans(frame_labels(a, thresh), fps, a["dist_m"], a["closing_ms"], thresh)


def label_seconds(span_list):
    """{label: seconds} over a list of spans."""
    out = {}
    for s in span_list:
        out[s["label"]] = out.get(s["label"], 0.0) + s["end_s"] - s["start_s"]
    return {k: round(v, 2) for k, v in out.items()}
