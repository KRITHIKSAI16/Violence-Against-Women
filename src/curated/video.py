"""Curated dataset: the per-clip video for the jury, ending exactly where the violence starts.

Pairs found      the PAIR VIDEO: the two people boxed (the one who moves toward the other in red), a line between them carrying the live distance in
                 meters, a top bar naming what they are doing at that moment, a distance strip at the bottom, then an end card
No pair          a plain video with every tracked person boxed and a caption saying why no interaction could be measured, then the same end card
The clips are already trimmed at the human start time, so no footage of the violent act is ever rendered.
"""
import subprocess
from pathlib import Path

import cv2
import numpy as np

from src.assm.track_poses import load_tracks
from src.config import resolve_path
from src.report.buildup_video import put_text
from src.video.shots import local_id

CARD_S = 1.2
ZONE_BGR = [(212, 215, 248), (207, 231, 251), (212, 245, 253), (226, 243, 238)]      # intimate .. public, light tints
KIND_BGR = {"concern": (43, 57, 192), "benign": (87, 139, 46), "neutral": (150, 150, 150)}


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


def plain_video(video_path, tracks, caption, out_path):
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


def span_at(spans_list, t):
    for s in spans_list:
        if s["start_s"] <= t < s["end_s"]:
            return s
    return None


def _strip(canvas, y0, strip, w, d, f, dmax):
    n = len(d)
    for zi, (lo, hi) in enumerate([(0, 0.46), (0.46, 1.2), (1.2, 3.7), (3.7, dmax)]):
        ya = int(y0 + strip - 6 - (min(hi, dmax) / dmax) * (strip - 12))
        yb = int(y0 + strip - 6 - (lo / dmax) * (strip - 12))
        canvas[ya:yb, :] = ZONE_BGR[zi]
    xs = lambda i: int(6 + (w - 12) * i / max(n - 1, 1))
    pr = None
    for i in range(n):
        if np.isfinite(d[i]):
            pt = (xs(i), int(y0 + strip - 6 - min(d[i], dmax) / dmax * (strip - 12)))
            if pr is not None and i - pr[1] <= 3:
                cv2.line(canvas, pr[0], pt, (92, 59, 31), 2)
            pr = (pt, i)
    cv2.line(canvas, (xs(min(f, n - 1)), y0), (xs(min(f, n - 1)), y0 + strip), (0, 0, 0), 2)
    put_text(canvas, f"distance between the two (0-{dmax:.0f} m)", (6, y0 + 16), 0.42, (40, 40, 40), 1)
    put_text(canvas, "violence starts", (max(6, w - 120), y0 + 16), 0.42, (0, 0, 192), 1)


def render_pair_video(video_path, tracks, result, dist_series, out_path):
    """The key pair boxed, live distance on a line between them, what they do now (top bar), distance strip at the bottom. Returns the path."""
    from src.curated.figures import span_kind
    from src.curated.interaction import PHRASE
    inter = result["interaction"]
    ids = inter["ids"]
    sp = inter["spans"]
    d = np.asarray(dist_series, float)
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    top, strip = 40, 96
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    raw = out_path.with_suffix(".raw.mp4")
    vw = cv2.VideoWriter(str(raw), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h + top + strip))
    rows = {}
    for k, (f, tid) in enumerate(zip(tracks["frame_idx"].tolist(), tracks["track_id"].tolist())):
        if tid in ids:
            rows.setdefault(f, {})[tid] = k
    last_actor = inter["last"][max(inter["last"], key=float)].get("actor")
    n = len(d)
    dmax = max(3.0, float(np.nanmax(d[np.isfinite(d)])) * 1.25) if np.isfinite(d).any() else 3.0
    f = 0
    while True:
        ok, img = cap.read()
        if not ok:
            break
        t = f / fps
        pts = {}
        cur = span_at(sp, t)
        actor = cur["actor"] if cur is not None and cur.get("actor") is not None else (last_actor if cur is None or cur["label"] in ("approaches", "follows", "approaches_from_behind") else None)
        for tid, k in rows.get(f, {}).items():
            x1, y1, x2, y2 = (int(v) for v in tracks["bbox"][k])
            col = (60, 60, 230) if tid == actor else (230, 140, 40)
            cv2.rectangle(img, (x1, y1), (x2, y2), col, 2)
            put_text(img, f"id{local_id(tid)}", (x1, max(12, y1 - 4)), 0.5, (255, 255, 255), 1)
            pts[tid] = ((x1 + x2) // 2, y2)
        if len(pts) == 2 and f < n and np.isfinite(d[f]):
            a, b = list(pts.values())
            cv2.line(img, a, b, (0, 0, 255) if d[f] < 1.2 else (0, 200, 255), 2)
            put_text(img, f"{d[f]:.1f} m", ((a[0] + b[0]) // 2, (a[1] + b[1]) // 2 - 6), 0.6, (255, 255, 255), 2)
        canvas = np.zeros((h + top + strip, w, 3), np.uint8)
        canvas[top:top + h] = img
        s = span_at(sp, t)
        if s is not None:
            if s.get("actor") is not None:
                a_, b_ = f"id{local_id(s['actor'])}", f"id{local_id(s['target'])}"
            else:
                a_, b_ = f"id{local_id(ids[0])}", f"id{local_id(ids[1])}"
            text, col = PHRASE[s["label"]].format(a=a_, b=b_), KIND_BGR[span_kind(s["label"])]
        else:
            text, col = ("no pair measurement at this moment" if f >= n or not np.isfinite(d[min(f, n - 1)]) else "the two are apart"), (90, 90, 90)
        canvas[:top] = col
        put_text(canvas, f"{t:.1f}s  {text}"[:max(20, w // 9)], (8, 26), 0.6, (255, 255, 255), 1)
        _strip(canvas, top + h, strip, w, d, f, dmax)
        vw.write(canvas)
        f += 1
    cap.release()
    vw.release()
    _reencode(raw, out_path)
    return out_path


def storyboard(video_path, result, out_path, n_frames=4, tile_w=420):
    """Key frames: the first, the start of each concerning span, and the last frame before the violence, each with its caption."""
    from src.curated.relation import CONCERN
    inter = result["interaction"]
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    nfr = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cand = [(0.0, "start of the footage")] + [(s["start_s"], s["label"].replace("_", " ")) for s in inter["spans"] if s["label"] in CONCERN and s["start_s"] > 0.2]
    pick = [cand[0]] + cand[1:][:max(0, n_frames - 2)] + [((nfr - 1) / fps, "last frame before the violence")]
    tiles = []
    for t, cap_txt in pick:
        cap.set(cv2.CAP_PROP_POS_FRAMES, min(nfr - 1, int(t * fps)))
        ok, im = cap.read()
        if not ok:
            continue
        im = cv2.resize(im, (tile_w, int(im.shape[0] * tile_w / im.shape[1])))
        bar = np.zeros((26, tile_w, 3), np.uint8)
        put_text(bar, f"{t:.1f}s {cap_txt}"[:tile_w // 8], (4, 18), 0.45, (255, 255, 255), 1)
        tiles.append(np.vstack([bar, im]))
    cap.release()
    if not tiles:
        return None
    hmax = max(x.shape[0] for x in tiles)
    tiles = [cv2.copyMakeBorder(x, 0, hmax - x.shape[0], 0, 0, cv2.BORDER_CONSTANT) for x in tiles]
    cols = 2 if len(tiles) > 2 else len(tiles)
    while len(tiles) % cols:
        tiles.append(np.zeros_like(tiles[0]))
    grid = np.vstack([np.hstack(tiles[i:i + cols]) for i in range(0, len(tiles), cols)])
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), grid)
    return Path(out_path)


def render_curated_clip(clip, cfg, result):
    """-> {video, storyboard} paths (None where not possible). The pair video when a pair exists, else the plain video; both end with the end card."""
    cur = cfg["curated"]
    cat, cid = clip["category"], clip["clip_id"]
    out_dir = Path(cur["videos_dir"])
    final = out_dir / cat / f"{cid}.mp4"
    sub = f"{clip['duration_s']:.1f} s of footage shown" if clip.get("start_s") is not None else ""
    text = f"Violence starts at {clip['start_s']:.1f} s" if clip.get("start_s") is not None else "End of the Normal clip"
    tp = resolve_path(cfg["assm"]["tracks_dir"]) / cat / f"{cid}.npz"
    if not tp.exists():
        return {"video": None, "storyboard": None}
    t = load_tracks(tp)
    src = resolve_path(clip["path"])
    board = None
    if not result.get("no_pair"):
        from src.context.features import load_context
        _, arrays = load_context(resolve_path(cfg["context"]["context_dir"]), cat, cid)
        mid = render_pair_video(src, t, result, arrays[tuple(result["interaction"]["ids"])]["dist_m"], out_dir / "_pair" / f"{cid}.mp4")
        board = storyboard(src, result, out_dir / "_board" / f"{cid}.jpg")
    else:
        mid = plain_video(src, t, result["lines"][0] if result.get("lines") else "", out_dir / "_plain" / f"{cid}.mp4")
    append_end_card(mid, final, text, sub)
    return {"video": final, "storyboard": board}
