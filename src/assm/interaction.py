"""M8b — ASSM step 2: Algorithm 1 pairwise interaction scores from cached tracks.

Reads the M8a cache (data/tracks/<Category>/<clip>.npz) and, for every pair of tracked
people (i, j) that appear in both frame t and frame t-1, computes

    d_ij   distance between the two people's pose centroids, in BODY HEIGHTS
           (pixel distance / mean bbox height of the pair), so "close" means the same thing
           for a near camera and a far camera. Floored at d_floor to keep 1/d finite.
    v_ij   closing speed in body-heights/second between the SAME two identities;
           positive = approaching. Measured over up to `lag` frames to reduce jitter.
    b_ij   path obstruction: 1 if one person stands between the other and the nearest
           frame edge (ExtrAnom has no marked-exit map, so the nearest frame edge stands in
           for the exit), else 0.
    score  w1/d + w2*v + w3*b                                 (Algorithm 1, Review Report I)

Outputs per clip (data/scores/<Category>/):
    <clip>_pairs.csv   one row per (frame, pair): d, v, b, score
    <clip>_curve.csv   one row per frame: n_people, n_pairs, max score, top pair  (the buildup curve)

Usage:  python -m src.assm.interaction [--config CFG] [--w1 W --w2 W --w3 W]
"""
import argparse
import csv
import json
import logging
from collections import defaultdict
from pathlib import Path

import numpy as np

from src.assm.track_poses import load_tracks
from src.config import load_config, resolve_path

log = logging.getLogger(__name__)

PAIR_FIELDS = ["frame", "time_s", "id_i", "id_j", "d", "v", "b", "score"]
CURVE_FIELDS = ["frame", "time_s", "n_people", "n_pairs", "max_score", "top_i", "top_j"]


def centroids(t, kpt_conf=0.5, min_kpts=3):
    """Pose centroid per row: mean of confident keypoints, else bbox centre."""
    box_c = np.stack([(t["bbox"][:, 0] + t["bbox"][:, 2]) / 2,
                      (t["bbox"][:, 1] + t["bbox"][:, 3]) / 2], axis=1)
    kp = t["kpts"]
    ok = kp[:, :, 2] > kpt_conf
    n = ok.sum(axis=1)
    sums = (kp[:, :, :2] * ok[:, :, None]).sum(axis=1)
    out = box_c.copy()
    use = n >= min_kpts
    out[use] = sums[use] / n[use][:, None]
    return out


def blocks_path(ci, cj, size, corridor):
    """True if j lies on the segment from i to the nearest frame edge (within `corridor` px)."""
    w, h = size
    x, y = ci
    gaps = {"left": x, "right": w - x, "top": y, "bottom": h - y}
    edge = min(gaps, key=gaps.get)
    exit_pt = {"left": (0.0, y), "right": (float(w), y), "top": (x, 0.0), "bottom": (x, float(h))}[edge]
    a = np.array(ci, float)
    ab = np.array(exit_pt, float) - a
    L2 = float(ab @ ab)
    if L2 < 1e-9:
        return False
    s = float((np.array(cj, float) - a) @ ab) / L2   # position of j along i -> exit
    if not 0.0 < s < 1.0:
        return False
    perp = np.linalg.norm(np.array(cj, float) - (a + s * ab))
    return bool(perp <= corridor)


def compute_pair_scores(t, cfg):
    """Algorithm 1 over a whole clip. Returns a list of row dicts (see PAIR_FIELDS)."""
    w1, w2, w3 = cfg["w1"], cfg["w2"], cfg["w3"]
    lag, d_floor, corridor_k = cfg["lag_frames"], cfg["d_floor"], cfg["corridor"]
    fps = float(t["fps"])
    size = (float(t["width"]), float(t["height"]))
    cen = centroids(t, cfg["kpt_conf"])
    bh = (t["bbox"][:, 3] - t["bbox"][:, 1]).clip(min=1.0)

    by_frame = defaultdict(dict)  # frame -> {track_id: row index}
    for k, (f, tid) in enumerate(zip(t["frame_idx"].tolist(), t["track_id"].tolist())):
        by_frame[f][tid] = k

    dist_hist = defaultdict(dict)  # (i, j) -> {frame: d in body heights}
    rows = []
    for f in sorted(by_frame):
        cur, prev = by_frame[f], by_frame.get(f - 1, {})
        for i in sorted(cur):
            for j in sorted(cur):
                if j <= i or i not in prev or j not in prev:
                    continue  # need both identities in t and t-1
                ki, kj = cur[i], cur[j]
                h_pair = (bh[ki] + bh[kj]) / 2
                d = float(np.linalg.norm(cen[ki] - cen[kj]) / h_pair)
                dist_hist[(i, j)][f] = d
                # closing speed over the longest available stretch up to `lag` frames back
                back = next((L for L in range(lag, 0, -1) if (f - L) in dist_hist[(i, j)]), None)
                if back is None:  # only t-1 raw pixels exist: use them directly
                    kpi, kpj = prev[i], prev[j]
                    d_prev = float(np.linalg.norm(cen[kpi] - cen[kpj]) / ((bh[kpi] + bh[kpj]) / 2))
                    back = 1
                else:
                    d_prev = dist_hist[(i, j)][f - back]
                v = (d_prev - d) / (back / fps)
                corridor = corridor_k * h_pair
                b = int(blocks_path(cen[ki], cen[kj], size, corridor)
                        or blocks_path(cen[kj], cen[ki], size, corridor))
                score = w1 / max(d, d_floor) + w2 * v + w3 * b
                rows.append({"frame": f, "time_s": round(f / fps, 4), "id_i": i, "id_j": j,
                             "d": round(d, 4), "v": round(v, 4), "b": b, "score": round(score, 4)})
    return rows


def clip_curve(rows, t):
    """Per-frame buildup curve: the max pair score each frame (0 where no pair exists)."""
    fps, n_frames = float(t["fps"]), int(t["n_frames"])
    people = defaultdict(int)
    for f in t["frame_idx"].tolist():
        people[f] += 1
    best = {}
    npairs = defaultdict(int)
    for r in rows:
        npairs[r["frame"]] += 1
        if r["frame"] not in best or r["score"] > best[r["frame"]]["score"]:
            best[r["frame"]] = r
    curve = []
    for f in range(n_frames):
        b = best.get(f)
        curve.append({"frame": f, "time_s": round(f / fps, 4), "n_people": people.get(f, 0),
                      "n_pairs": npairs.get(f, 0), "max_score": b["score"] if b else 0.0,
                      "top_i": b["id_i"] if b else -1, "top_j": b["id_j"] if b else -1})
    return curve


def write_csv(path, fields, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def read_csv(path, int_cols=(), float_cols=()):
    """Read one of the CSVs written above back into typed row dicts."""
    out = []
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            for c in int_cols:
                r[c] = int(r[c])
            for c in float_cols:
                r[c] = float(r[c])
            out.append(r)
    return out


def load_pairs(path):
    return read_csv(path, ("frame", "id_i", "id_j", "b"), ("time_s", "d", "v", "score"))


def load_curve(path):
    return read_csv(path, ("frame", "n_people", "n_pairs", "top_i", "top_j"), ("time_s", "max_score"))


def score_clip(tracks_path, cfg, scores_dir, category, clip_id):
    t = load_tracks(tracks_path)
    rows = compute_pair_scores(t, cfg)
    curve = clip_curve(rows, t)
    write_csv(scores_dir / category / f"{clip_id}_pairs.csv", PAIR_FIELDS, rows)
    write_csv(scores_dir / category / f"{clip_id}_curve.csv", CURVE_FIELDS, curve)
    return rows, curve


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    ap = argparse.ArgumentParser(description="M8b: Algorithm 1 interaction scores")
    ap.add_argument("--config", default=None)
    for k in ("w1", "w2", "w3"):
        ap.add_argument(f"--{k}", type=float, default=None, help=f"override {k}")
    args = ap.parse_args()

    cfg = load_config(args.config)
    a = dict(cfg["assm"])
    for k in ("w1", "w2", "w3"):
        if getattr(args, k) is not None:
            a[k] = getattr(args, k)
    clean = json.loads(resolve_path(cfg["preprocess"]["clean_manifest_path"]).read_text("utf-8"))
    tracks_dir, scores_dir = resolve_path(a["tracks_dir"]), resolve_path(a["scores_dir"])

    print(f"weights w1={a['w1']} w2={a['w2']} w3={a['w3']}  d_floor={a['d_floor']}  lag={a['lag_frames']}")
    print(f"{'clip':<20}{'frames':>7}{'pair-frames':>12}{'mean':>8}{'max':>8}{'p95':>8}")
    for c in clean["clips"]:
        p = tracks_dir / c["category"] / f"{c['clip_id']}.npz"
        if not p.exists():
            log.warning("No tracks for %s (run M8a first)", c["clip_id"])
            continue
        rows, curve = score_clip(p, a, scores_dir, c["category"], c["clip_id"])
        s = np.array([r["score"] for r in rows]) if rows else np.zeros(1)
        print(f"{c['clip_id']:<20}{len(curve):>7}{len(rows):>12}{s.mean():>8.2f}{s.max():>8.2f}"
              f"{np.percentile(s, 95):>8.2f}" + ("   (no interaction)" if not rows else ""))
    print(f"\nWrote CSVs to {scores_dir}")


if __name__ == "__main__":
    main()
