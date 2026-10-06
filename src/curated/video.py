"""Curated dataset: the per-clip video for the jury, ending exactly where the violence starts.

Pairs found      the existing story video (people boxes, minimap in meters, cue strip) of the whole trimmed clip, then an end card
No pair          a plain video with every tracked person boxed and a caption saying why no interaction could be measured, then the same end card
The clips are already trimmed at the human start time, so no footage of the violent act is ever rendered.
"""
import subprocess
from pathlib import Path

import cv2
import numpy as np

from src.assm.track_poses import load_tracks
from src.config import resolve_path
from src.report.buildup_video import load_overrides, put_text
from src.video.shots import local_id

CARD_S = 1.2


def end_card(w, h, text, sub=""):
    img = np.zeros((h, w, 3), np.uint8)
    put_text(img, text, (max(10, w // 2 - 8 * len(text)), h // 2), 0.9, (255, 255, 255), 2)
    if sub:
        put_text(img, sub, (max(10, w // 2 - 4 * len(sub)), h // 2 + 34), 0.5, (200, 200, 200), 1)
    return img


def _reencode(raw, out):
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", str(raw), "-c:v", "libx264", "-pix_fmt", "yuv420p", str(out)], check=True)
    Path(raw).unlink(missing_ok=True)


def append_end_card(video, out, text, sub="", seconds=CARD_S):
    """Copy `video` to `out` with an end card appended (H.264, even sizes preserved)."""
    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    raw = out.with_suffix(".raw.mp4")
    vw = cv2.VideoWriter(str(raw), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    while True:
        ok, f = cap.read()
        if not ok:
            break
        vw.write(f)
    cap.release()
    card = end_card(w, h, text, sub)
    for _ in range(max(1, round(seconds * fps))):
        vw.write(card)
    vw.release()
    _reencode(raw, out)
    return out


def plain_video(video_path, tracks, caption, out_path, key_ids=()):
    """Every tracked person boxed (per-shot local ids) plus a caption bar; used when no pair was found."""
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    bar = 36
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    raw = out_path.with_suffix(".raw.mp4")
    vw = cv2.VideoWriter(str(raw), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h + bar))
    rows = {}
    for k, f in enumerate(tracks["frame_idx"].tolist()):
        rows.setdefault(f, []).append(k)
    f = 0
    while True:
        ok, img = cap.read()
        if not ok:
            break
        for k in rows.get(f, []):
            x1, y1, x2, y2 = (int(v) for v in tracks["bbox"][k])
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 220, 0), 2)
            put_text(img, f"id{local_id(int(tracks['track_id'][k]))}", (x1, max(12, y1 - 4)), 0.5, (255, 255, 255), 1)
        canvas = np.zeros((h + bar, w, 3), np.uint8)
        canvas[bar:] = img
        put_text(canvas, caption[:max(10, w // 9)], (6, 24), 0.55, (255, 255, 255), 1)
        vw.write(canvas)
        f += 1
    cap.release()
    vw.release()
    _reencode(raw, out_path)
    return out_path


def render_curated_clip(clip, cfg, result):
    """-> {video, storyboard} paths (None where not possible). Uses the existing story renderer when a pair exists."""
    from src.report.story_video import render_clip
    cur = cfg["curated"]
    cat, cid = clip["category"], clip["clip_id"]
    out_dir = Path(cur["videos_dir"])
    final = out_dir / cat / f"{cid}.mp4"
    sub = f"{clip['duration_s']:.1f} s of footage shown" if clip.get("start_s") is not None else ""
    text = f"Violence starts at {clip['start_s']:.1f} s" if clip.get("start_s") is not None else "End of the Normal clip"
    board = None
    mid = None
    if not result.get("no_pair"):
        try:
            r = render_clip(clip, cfg, load_overrides(cfg["report"]["overrides_csv"]), out_dir / "_story")
            mid, board = r["video"], r["storyboard"]
        except Exception as e:                                      # fall back to the plain video; the reason is kept in the result
            result["render_error"] = str(e)
    if mid is None:
        t = load_tracks(resolve_path(cfg["assm"]["tracks_dir"]) / cat / f"{cid}.npz") if (resolve_path(cfg["assm"]["tracks_dir"]) / cat / f"{cid}.npz").exists() else None
        cap = result["lines"][0] if result.get("lines") else ""
        if t is None:
            return {"video": None, "storyboard": None}
        mid = plain_video(resolve_path(clip["path"]), t, cap, out_dir / "_plain" / f"{cid}.mp4")
    append_end_card(mid, final, text, sub)
    return {"video": final, "storyboard": board}
