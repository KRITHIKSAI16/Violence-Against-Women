"""M9 verification tool: recompute one frame of the key pair's behavior features from the raw tracks.

Independent of gate.py (only numpy and the M8a cache), so agreement with data/gate/<Category>/<clip>_states.csv shows
the gate computes what the docs say. Default frame = the middle of the first behavior segment of the key pair.

Usage:  python -m src.assm.hand_check_gate <clip_id> [--config CFG] [--frame N]
"""
import argparse
import csv
import json

import numpy as np

from src.assm.gate import ESCALATION, load_gate
from src.config import load_config, resolve_path


def main():
    ap = argparse.ArgumentParser(description="Recompute one frame of a clip's key-pair features by hand")
    ap.add_argument("clip_id")
    ap.add_argument("--config", default=None)
    ap.add_argument("--frame", type=int, default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)
    a, g = cfg["assm"], cfg["gate"]
    clean = json.loads(resolve_path(cfg["preprocess"]["clean_manifest_path"]).read_text("utf-8"))
    cat = next(c["category"] for c in clean["clips"] if c["clip_id"] == args.clip_id)
    gate = load_gate(resolve_path(g["gate_dir"]) / cat / f"{args.clip_id}_gate.json")
    if not gate["key_pair"]:
        raise SystemExit("This clip has no key pair (no behavior detected), nothing to recompute.")
    i, j = gate["key_pair"]
    segs = [s for s in gate["segments"] if (s["id_i"], s["id_j"]) == (i, j) and s["state"] != ESCALATION]
    frame = args.frame if args.frame is not None else (segs[0]["start_f"] + segs[0]["end_f"]) // 2
    z = np.load(resolve_path(a["tracks_dir"]) / cat / f"{args.clip_id}.npz")
    fps = float(z["fps"])
    W = max(1, round(g["smooth_s"] * fps))
    L = max(1, round(g["vel_s"] * fps))

    def centre(k):                       # mean of confident keypoints, else bbox centre (same rule as M8b)
        kp = z["kpts"][k]
        ok = kp[:, 2] > a["kpt_conf"]
        return kp[ok, :2].mean(0) if ok.sum() >= 3 else z["bbox"][k].reshape(2, 2).mean(0)

    def raw(tid, f):
        k = np.where((z["frame_idx"] == f) & (z["track_id"] == tid))[0]
        return None if len(k) == 0 else (centre(k[0]), float(z["bbox"][k[0]][3] - z["bbox"][k[0]][1]))

    def smooth(tid, f):                  # trailing mean over the last W frames present
        pts = [raw(tid, ff) for ff in range(f - W + 1, f + 1)]
        pts = [p for p in pts if p is not None]
        return None if not pts else (np.mean([p[0] for p in pts], 0), np.mean([p[1] for p in pts]))

    Si, Sj = smooth(i, frame), smooth(j, frame)
    Pi, Pj = smooth(i, frame - L), smooth(j, frame - L)
    h = (Si[1] + Sj[1]) / 2
    d = np.linalg.norm(Si[0] - Sj[0]) / h
    vi = (Si[0] - Pi[0]) / (L / fps) / h
    vj = (Sj[0] - Pj[0]) / (L / fps) / h
    si, sj = np.linalg.norm(vi), np.linalg.norm(vj)
    cos = float(vi @ vj / (si * sj)) if si >= g["move_speed"] and sj >= g["move_speed"] else float("nan")

    row = next((r for r in csv.DictReader(open(resolve_path(g["gate_dir"]) / cat / f"{args.clip_id}_states.csv")) if int(r["frame"]) == frame), None)
    if row is None:
        raise SystemExit(f"frame {frame} is not in the states file (pair not co-tracked there)")

    def ok(file_val, hand):
        if file_val == "" and np.isnan(hand):
            return "OK "
        return "OK " if file_val != "" and abs(float(file_val) - hand) < 2e-2 else "MISMATCH"
    print(f"{args.clip_id}  key pair id{i}-id{j}  frame {frame} (t={frame / fps:.2f}s)  state in file: {row['state'] or '-'}")
    print(f"  d        file={row['d']:>8}  hand={d:8.3f}  {ok(row['d'], d)}  (body heights apart)")
    print(f"  speed_i  file={row['speed_i']:>8}  hand={si:8.3f}  {ok(row['speed_i'], si)}  (body heights / s)")
    print(f"  speed_j  file={row['speed_j']:>8}  hand={sj:8.3f}  {ok(row['speed_j'], sj)}")
    print(f"  cos      file={row['cos']:>8}  hand={cos:8.3f}  {ok(row['cos'], cos)}  (heading alignment; blank = someone not moving)")
    for s in segs:
        if s["start_f"] <= frame <= s["end_f"]:
            print(f"  segment: {s['state']} {s['start_s']}-{s['end_s']}s  actor={s.get('actor')} target={s.get('target')}")


if __name__ == "__main__":
    main()
