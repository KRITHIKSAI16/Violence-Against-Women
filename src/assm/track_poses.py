"""M8a — ASSM step 1: person detection + pose + ByteTrack, cached per clip.

Runs YOLOv8n-pose with Ultralytics' built-in ByteTrack on each cleaned clip (M2 output)
and saves what M8b (Algorithm 1 scoring) and Phase II need, so the slow model never has
to run again when weights/thresholds change:

  data/tracks/<Category>/<clip_id>.npz
    frame_idx (N,)   int32    0-based frame number
    track_id  (N,)   int32    persistent identity (ByteTrack id, after stitch_tracks)
    raw_track_id (N,) int32   the id exactly as ByteTrack gave it (before stitching)
    bbox      (N,4)  float32  x1,y1,x2,y2 in pixels
    conf      (N,)   float32  detection confidence
    kpts      (N,17,3) float32  COCO keypoints x, y, keypoint-confidence
    fps, width, height, n_frames   clip metadata
One row per (frame, tracked person). Frames with nobody tracked have no rows.

Usage:  python -m src.assm.track_poses [--config CFG] [--limit N] [--force] [--preview-dir DIR]
"""
import argparse
import json
import logging
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

from src.config import load_config, resolve_path

log = logging.getLogger(__name__)
N_KPTS = 17


def pack_tracks(rows, fps, width, height, n_frames):
    """Turn a list of (frame_idx, track_id, bbox, conf, kpts) rows into the npz arrays."""
    n = len(rows)
    return {
        "frame_idx": np.array([r[0] for r in rows], np.int32).reshape(n),
        "track_id": np.array([r[1] for r in rows], np.int32).reshape(n),
        "raw_track_id": np.array([r[1] for r in rows], np.int32).reshape(n),
        "bbox": np.array([r[2] for r in rows], np.float32).reshape(n, 4),
        "conf": np.array([r[3] for r in rows], np.float32).reshape(n),
        "kpts": np.array([r[4] for r in rows], np.float32).reshape(n, N_KPTS, 3),
        "fps": np.float32(fps), "width": np.int32(width),
        "height": np.int32(height), "n_frames": np.int32(n_frames),
    }


def load_tracks(path):
    """Load a cached track file into a plain dict of arrays."""
    with np.load(path) as z:
        return {k: z[k] for k in z.files}


def stitch_tracks(t, max_gap=45, max_dist=1.5, size_ratio=1.7):
    """Join broken tracklets of the same person (ByteTrack matches on box overlap only, so a
    small or fast person can get a new id after a short detection gap).

    Tracklet B is attached to an earlier chain A when A ended before B starts, the gap is at most
    `max_gap` frames, A's last position is within `max_dist` body heights of B's first position,
    and the heights agree within `size_ratio`. A and B never overlap in time, so two people seen
    together are never merged. Returns a copy of `t` with `track_id` rewritten (raw ids kept).
    """
    out = dict(t)
    ids = t["raw_track_id"]
    if len(ids) == 0:
        return out
    info = {}
    for tid in sorted(set(ids.tolist())):
        k = np.where(ids == tid)[0]
        k = k[np.argsort(t["frame_idx"][k])]
        b = t["bbox"][k]
        c = np.stack([(b[:, 0] + b[:, 2]) / 2, (b[:, 1] + b[:, 3]) / 2], 1)
        info[tid] = {"first": int(t["frame_idx"][k[0]]), "last": int(t["frame_idx"][k[-1]]),
                     "c0": c[0], "c1": c[-1], "h": float((b[:, 3] - b[:, 1]).mean())}
    chains = {}   # chain id -> {last, c1, h}
    label = {}
    for tid in sorted(info, key=lambda x: info[x]["first"]):
        a = info[tid]
        best, best_d = None, None
        for cid, ch in chains.items():
            gap = a["first"] - ch["last"]
            if not 0 < gap <= max_gap:
                continue
            ratio = max(a["h"], ch["h"]) / max(min(a["h"], ch["h"]), 1e-6)
            d = float(np.linalg.norm(a["c0"] - ch["c1"])) / ((a["h"] + ch["h"]) / 2)
            if ratio <= size_ratio and d <= max_dist and (best is None or d < best_d):
                best, best_d = cid, d
        if best is None:
            chains[tid] = {"last": a["last"], "c1": a["c1"], "h": a["h"]}
            label[tid] = tid
        else:
            chains[best].update(last=a["last"], c1=a["c1"], h=a["h"])
            label[tid] = best
    out["track_id"] = np.array([label[i] for i in ids.tolist()], np.int32)
    return out


def _device(setting):
    if setting != "auto":
        return setting
    import torch
    return 0 if torch.cuda.is_available() else "cpu"


def track_clip(video_path, cfg, fps, width, height):
    """Run pose + ByteTrack over one clip; return the packed arrays."""
    from ultralytics import YOLO

    # A fresh model per clip resets ByteTrack state, so IDs never leak between clips.
    model = YOLO(str(resolve_path(cfg["model"])))
    tracker = resolve_path(cfg["tracker"])
    tracker = str(tracker) if tracker.exists() else cfg["tracker"]  # else an Ultralytics built-in name
    rows, n_frames = [], 0
    stream = model.track(source=str(video_path), stream=True, persist=True,
                         tracker=tracker, conf=cfg["conf"], imgsz=cfg["imgsz"],
                         device=_device(cfg["device"]), verbose=False)
    for i, r in enumerate(stream):
        n_frames = i + 1
        if r.boxes is None or r.boxes.id is None or r.keypoints is None:
            continue
        ids = r.boxes.id.int().cpu().numpy()
        xyxy = r.boxes.xyxy.cpu().numpy()
        conf = r.boxes.conf.cpu().numpy()
        kp = r.keypoints.data.cpu().numpy()  # (n,17,3)
        for k in range(len(ids)):
            rows.append((i, ids[k], xyxy[k], conf[k], kp[k]))
    t = pack_tracks(rows, fps, width, height, n_frames)
    if cfg.get("stitch", True):
        t = stitch_tracks(t, cfg["stitch_max_gap"], cfg["stitch_max_dist"])
    return t


def clip_summary(t):
    """Small stats dict for printing/sanity-checking one clip's tracks."""
    per_frame = Counter(t["frame_idx"].tolist())
    n_frames = int(t["n_frames"])
    return {
        "frames": n_frames,
        "rows": int(len(t["frame_idx"])),
        "unique_ids": int(len(set(t["track_id"].tolist()))),
        "raw_ids": int(len(set(t["raw_track_id"].tolist()))) if "raw_track_id" in t else None,
        "frames_with_1plus": len(per_frame),
        "frames_with_2plus": sum(1 for v in per_frame.values() if v >= 2),
        "max_people": max(per_frame.values(), default=0),
    }


def render_preview(video_path, t, out_path):
    """Save one annotated frame (the frame with the most people) as a jpg for eyeballing."""
    per_frame = Counter(t["frame_idx"].tolist())
    if not per_frame:
        return False
    frame_no = max(per_frame, key=lambda f: (per_frame[f], -f))
    cap = cv2.VideoCapture(str(video_path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
    ok, img = cap.read()
    cap.release()
    if not ok:
        return False
    for k in np.where(t["frame_idx"] == frame_no)[0]:
        x1, y1, x2, y2 = t["bbox"][k].astype(int)
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img, f"id{t['track_id'][k]}", (x1, max(12, y1 - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        for x, y, c in t["kpts"][k]:
            if c > 0.5:
                cv2.circle(img, (int(x), int(y)), 3, (0, 0, 255), -1)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), img)
    return True


def run(clean_manifest, cfg, tracks_dir, limit=None, force=False, preview_dir=None):
    """Track every clip in the M2 manifest; return a list of per-clip summaries."""
    summaries, failed = [], []
    clips = clean_manifest["clips"][:limit] if limit else clean_manifest["clips"]
    for c in clips:
        out = tracks_dir / c["category"] / f"{c['clip_id']}.npz"
        video = resolve_path(c["path"])
        try:
            if out.exists() and not force:
                t = load_tracks(out)  # resumable: reuse earlier result
            else:
                t = track_clip(video, cfg, c["fps"], c["width"], c["height"])
                out.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(out, **t)
        except Exception as e:  # one bad clip must not stop a long Colab run
            log.warning("Skipping %s: %s", c["clip_id"], e)
            failed.append({"clip_id": c["clip_id"], "reason": str(e)})
            continue
        s = {"clip_id": c["clip_id"], "category": c["category"], **clip_summary(t)}
        summaries.append(s)
        log.info("%-20s frames=%-4d ids=%-3d (raw %-3s) 2+people=%-4d max=%d", s["clip_id"], s["frames"],
                 s["unique_ids"], s["raw_ids"], s["frames_with_2plus"], s["max_people"])
        if preview_dir:
            render_preview(video, t, Path(preview_dir) / c["category"] / f"{c['clip_id']}.jpg")
    return summaries, failed


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    ap = argparse.ArgumentParser(description="M8a: detect + pose + ByteTrack, cache tracks")
    ap.add_argument("--config", default=None)
    ap.add_argument("--limit", type=int, default=None, help="only first N clips (testing)")
    ap.add_argument("--force", action="store_true", help="recompute even if cached")
    ap.add_argument("--preview-dir", default=None, help="save one annotated jpg per clip here")
    args = ap.parse_args()

    cfg = load_config(args.config)
    clean = json.loads(resolve_path(cfg["preprocess"]["clean_manifest_path"]).read_text("utf-8"))
    tracks_dir = resolve_path(cfg["assm"]["tracks_dir"])
    summaries, failed = run(clean, cfg["assm"], tracks_dir, args.limit, args.force, args.preview_dir)

    print(f"\nTracked {len(summaries)} clips, {len(failed)} failed -> {tracks_dir}")
    by_cat = {}
    for s in summaries:
        d = by_cat.setdefault(s["category"], {"clips": 0, "frames": 0, "two_plus": 0})
        d["clips"] += 1
        d["frames"] += s["frames"]
        d["two_plus"] += s["frames_with_2plus"]
    for cat, d in sorted(by_cat.items()):
        print(f"  {cat:<16} clips={d['clips']}  frames with 2+ people: {d['two_plus']}/{d['frames']}")
    for f in failed:
        print(f"  FAILED {f['clip_id']}: {f['reason']}")


if __name__ == "__main__":
    main()
