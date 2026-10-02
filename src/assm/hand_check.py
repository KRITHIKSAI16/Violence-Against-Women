"""M8 verification tool: recompute one score row from the raw cached tracks, independently.

Uses only numpy and the M8a cache (not interaction.py), so agreement with the CSV shows
the scoring code does what Algorithm 1 says. Default row = the clip's top-scoring row.

Usage:  python -m src.assm.hand_check <clip_id> [--config CFG] [--frame N]
"""
import argparse
import csv
import json

import numpy as np

from src.config import load_config, resolve_path


def main():
    ap = argparse.ArgumentParser(description="Recompute one score row by hand")
    ap.add_argument("clip_id")
    ap.add_argument("--config", default=None)
    ap.add_argument("--frame", type=int, default=None, help="check the best pair in this frame")
    args = ap.parse_args()
    cfg = load_config(args.config)
    a = cfg["assm"]
    clean = json.loads(resolve_path(cfg["preprocess"]["clean_manifest_path"]).read_text("utf-8"))
    cat = next(c["category"] for c in clean["clips"] if c["clip_id"] == args.clip_id)
    rows = list(csv.DictReader(open(resolve_path(a["scores_dir"]) / cat / f"{args.clip_id}_pairs.csv")))
    if args.frame is not None:
        rows = [r for r in rows if int(r["frame"]) == args.frame]
    if not rows:
        raise SystemExit("no pair rows for this clip/frame (nobody interacting there)")
    r = max(rows, key=lambda r: float(r["score"]))
    f, i, j = int(r["frame"]), int(r["id_i"]), int(r["id_j"])
    z = np.load(resolve_path(a["tracks_dir"]) / cat / f"{args.clip_id}.npz")
    fps = float(z["fps"])

    def person(frame, tid):
        k = np.where((z["frame_idx"] == frame) & (z["track_id"] == tid))[0][0]
        kp = z["kpts"][k]
        ok = kp[:, 2] > a["kpt_conf"]
        c = kp[ok, :2].mean(0) if ok.sum() >= 3 else z["bbox"][k].reshape(2, 2).mean(0)
        return c, float(z["bbox"][k][3] - z["bbox"][k][1])

    def dist(frame):
        (ci, hi), (cj, hj) = person(frame, i), person(frame, j)
        return float(np.linalg.norm(ci - cj) / ((hi + hj) / 2))

    d = dist(f)
    have = {(int(x["frame"]), int(x["id_i"]), int(x["id_j"])) for x in csv.DictReader(
        open(resolve_path(a["scores_dir"]) / cat / f"{args.clip_id}_pairs.csv"))}
    back = next((L for L in range(a["lag_frames"], 0, -1) if (f - L, i, j) in have), 1)
    v = (dist(f - back) - d) / (back / fps)          # positive = getting closer
    score = a["w1"] / max(d, a["d_floor"]) + a["w2"] * v + a["w3"] * int(r["b"])
    ok = lambda x, y: "OK " if abs(x - y) < 1e-2 else "MISMATCH"
    print(f"{args.clip_id}  frame {f} (t={f / fps:.2f}s)  pair id{i}-id{j}")
    print(f"  d      file={float(r['d']):8.4f}  hand={d:8.4f}  {ok(float(r['d']), d)}  (body heights apart)")
    print(f"  v      file={float(r['v']):8.4f}  hand={v:8.4f}  {ok(float(r['v']), v)}  (looked {back} frame(s) back; + = approaching)")
    print(f"  b      file={int(r['b'])}  (1 = one person blocks the other's way to the nearest frame edge)")
    print(f"  score  file={float(r['score']):8.4f}  hand={score:8.4f}  {ok(float(r['score']), score)}"
          f"  = {a['w1']}/max({d:.3f},{a['d_floor']}) + {a['w2']}*{v:.3f} + {a['w3']}*{r['b']}")


if __name__ == "__main__":
    main()
