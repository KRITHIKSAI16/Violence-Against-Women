"""M8 verification tool: draw tracks, the top pair and the buildup curve onto a clip.

Produces an .mp4 (re-encoded to h264 so it plays in browsers/Colab) for manual checking:
  - green box + id for every tracked person
  - the highest-scoring pair joined by a line (red when score is above --tau)
  - text: d (body heights), v (+ = approaching), b, score
  - bottom strip: the clip's buildup curve with a moving cursor

Usage:  python -m src.assm.render <clip_id> [--config CFG] [--tau 3.0] [--out FILE]
"""
import argparse
import json
import subprocess
from pathlib import Path

import cv2
import numpy as np

from src.assm.interaction import centroids, load_curve, load_pairs
from src.assm.track_poses import load_tracks
from src.config import load_config, resolve_path

STRIP_H = 90


def _draw_strip(img, curve_scores, cur, ymax, tau):
    h, w = img.shape[:2]
    top = h - STRIP_H
    cv2.rectangle(img, (0, top), (w, h), (30, 30, 30), -1)
    n = len(curve_scores)
    xs = np.linspace(4, w - 4, n)
    ys = h - 6 - (np.clip(curve_scores, 0, ymax) / ymax) * (STRIP_H - 14)
    pts = np.stack([xs, ys], 1).astype(np.int32)
    if tau is not None:
        ty = int(h - 6 - min(tau / ymax, 1) * (STRIP_H - 14))
        cv2.line(img, (0, ty), (w, ty), (0, 140, 255), 1)
    cv2.polylines(img, [pts], False, (255, 255, 0), 1)
    cv2.line(img, (int(xs[cur]), top), (int(xs[cur]), h), (255, 255, 255), 1)


def render(clip, video_path, tracks_path, pairs_path, curve_path, out_path, tau):
    t = load_tracks(tracks_path)
    pairs = {(r["frame"], r["id_i"], r["id_j"]): r for r in load_pairs(pairs_path)}
    curve = load_curve(curve_path)
    scores = np.array([c["max_score"] for c in curve])
    ymax = max(float(scores.max()), (tau or 0) * 1.2, 1.0)
    cen = centroids(t)
    rows_by_frame = {}
    for k, f in enumerate(t["frame_idx"].tolist()):
        rows_by_frame.setdefault(f, []).append(k)

    cap = cv2.VideoCapture(str(video_path))
    w, h = int(cap.get(3)), int(cap.get(4))
    fps = cap.get(cv2.CAP_PROP_FPS)
    tmp = Path(out_path).with_suffix(".raw.mp4")
    vw = cv2.VideoWriter(str(tmp), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h + STRIP_H))
    f = 0
    while True:
        ok, img = cap.read()
        if not ok:
            break
        canvas = np.zeros((h + STRIP_H, w, 3), np.uint8)
        canvas[:h] = img
        idx = rows_by_frame.get(f, [])
        pos = {}
        for k in idx:
            x1, y1, x2, y2 = t["bbox"][k].astype(int)
            tid = int(t["track_id"][k])
            pos[tid] = tuple(cen[k].astype(int))
            cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(canvas, f"id{tid}", (x1, max(14, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        c = curve[min(f, len(curve) - 1)]
        label = f"t={c['time_s']:.2f}s  people={c['n_people']}  "
        if c["top_i"] >= 0:
            r = pairs[(f, c["top_i"], c["top_j"])]
            hot = tau is not None and r["score"] >= tau
            col = (0, 0, 255) if hot else (0, 255, 255)
            cv2.line(canvas, pos[c["top_i"]], pos[c["top_j"]], col, 3)
            label += f"pair id{c['top_i']}-id{c['top_j']}  d={r['d']:.2f} v={r['v']:+.2f} b={r['b']} score={r['score']:.2f}"
        else:
            label += "no pair"
        cv2.rectangle(canvas, (0, 0), (w, 24), (0, 0, 0), -1)
        cv2.putText(canvas, label, (4, 17), cv2.FONT_HERSHEY_SIMPLEX, min(0.5, w / 1500), (255, 255, 255), 1)
        _draw_strip(canvas, scores, min(f, len(scores) - 1), ymax, tau)
        vw.write(canvas)
        f += 1
    cap.release()
    vw.release()
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", str(tmp), "-c:v", "libx264",
                    "-pix_fmt", "yuv420p", str(out_path)], check=True)
    tmp.unlink()
    return out_path


def main():
    ap = argparse.ArgumentParser(description="Render an annotated verification video for one clip")
    ap.add_argument("clip_id")
    ap.add_argument("--config", default=None)
    ap.add_argument("--tau", type=float, default=None, help="draw this threshold; pair line turns red above it")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)
    clean = json.loads(resolve_path(cfg["preprocess"]["clean_manifest_path"]).read_text("utf-8"))
    c = next(x for x in clean["clips"] if x["clip_id"] == args.clip_id)
    a = cfg["assm"]
    stem = f"{c['category']}/{c['clip_id']}"
    sd = resolve_path(a["scores_dir"])
    out = Path(args.out) if args.out else resolve_path(a["scores_dir"]) / "videos" / f"{c['clip_id']}.mp4"
    out.parent.mkdir(parents=True, exist_ok=True)
    render(c["clip_id"], resolve_path(c["path"]), resolve_path(a["tracks_dir"]) / f"{stem}.npz",
           sd / f"{stem}_pairs.csv", sd / f"{stem}_curve.csv", out, args.tau)
    print("Wrote", out)


if __name__ == "__main__":
    main()
