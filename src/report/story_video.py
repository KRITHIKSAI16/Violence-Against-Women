"""Phase 1 deliverable: the STORY video of a clip's pre-violence buildup (violence not shown).

Draws the scene graph of the clip's key pair onto the footage:
  header        the episodes active right now, as sentences ("id3 follows id1, about 1.8 s behind ...")
  video         boxes with track ids (key pair highlighted), detected doors outlined (only when the layout is reliable)
  mini-map      bird's-eye view in meters (approximate): the two people with 3 s trails and Hall's proxemic rings (0.46 / 1.2 / 3.7 m) around the target
  cue strip     one row per cue family (approach, follow, look back, linger, block, act, target reaction, conversation), the weighted cue curve, a cursor
                and a red line at the cut
Footage stops before the first physical-act cue (reach, contact, running away) minus a margin, or earlier for categories with an act and nothing
detected (uniform tail trim), exactly as in `buildup_video.cut_point`; after the cut a card lists what was found. The cut is a heuristic: nobody
annotated where the act starts, so check the clips you present and use configs/clip_overrides.csv for exact times.

Outputs: <out>/videos/<Category>/<clip>_story.mp4 and <out>/storyboards/<Category>/<clip>_story.jpg
Usage:  python -m src.report.story_video <clip_id> [<clip_id> ...] [--config CFG] [--out DIR]
"""
import argparse
import json
import logging
import subprocess
from pathlib import Path

import cv2
import numpy as np

from src.assm.track_poses import load_tracks
from src.config import load_config, resolve_path
from src.context.graph import ACT_PREDS, load_story
from src.context.ground import ground_series
from src.context.narrative import episode_text, scene_line, summary_line
from src.context.scene import load_layout
from src.report.buildup_video import cut_point, load_overrides, put_text

log = logging.getLogger(__name__)

FAMILY = [("APPR", ("approaches_from_behind", "approaches_side", "approaches_frontal"), (0, 215, 255)),
          ("FOLL", ("follows",), (0, 140, 255)),
          ("LOOK", ("looks_back",), (255, 0, 200)),
          ("LING", ("hovers_near", "very_close"), (255, 170, 40)),
          ("BLOK", ("blocks_exit", "pinned_against"), (40, 40, 235)),
          ("ACT", ACT_PREDS, (0, 0, 200)),
          ("TGT", ("target_stops", "target_speeds_up"), (200, 100, 160)),
          ("CONV", ("mutual_facing",), (80, 200, 80))]
COLOR_OF = {p: c for _, preds, c in FAMILY for p in preds}
ROW, PAD, CURVE_H, HEADER_H = 9, 6, 30, 58      # all even: H.264 needs even frame sizes


def strip_height():
    return len(FAMILY) * ROW + CURVE_H + 3 * PAD


def key_episodes(story):
    return story["pairs"][story["key_pair"]]["episodes"] if story.get("key_pair") else []


def active_episodes(story, frame):
    """Episodes of the key pair active at `frame`, act cues first, then by confidence."""
    eps = [e for e in key_episodes(story) if e["start_f"] <= frame <= e["end_f"]]
    return sorted(eps, key=lambda e: (e["pred"] not in ACT_PREDS, -e["conf"]))


def story_cut(story, category, gate_escalation_s, rep, overrides=None):
    """Cut point from the story's first act cue (and the M9 gate's escalation if it exists), via the common cut rule."""
    times = [x for x in (story["escalation"]["time_s"] if story.get("escalation") else None, gate_escalation_s) if x is not None]
    g = {"clip_id": story["clip_id"], "duration_s": story["n_frames"] / story["fps"], "fps": story["fps"],
         "escalation": {"time_s": min(times)} if times else None}
    return cut_point(g, category, rep, overrides)


def _fit(text, width_px, scale):
    per = max(8, int((width_px - 10) / (17.0 * scale)))
    return text if len(text) <= per else text[:per - 3] + "..."


# ---------------------------------------------------------------- mini-map
def pair_ground(t, ctx, pair):
    """Ground-plane (X, Z) meters of both people per frame (no camera compensation: the map is approximate)."""
    g = ground_series(t, ctx, None, list(pair))
    out = {}
    for tid in pair:
        xz = np.stack([g[tid]["X"], g[tid]["Z"]], 1)
        xz[~(g[tid]["conf"] >= ctx["min_ground_conf"])] = np.nan        # cropped / sitting boxes give wrong depth: do not draw them
        out[tid] = xz
    return out


def draw_minimap(canvas, x0, y0, size, pos, pair, frame, fps, active, extent):
    """Bird's-eye inset. pos: {tid: (n,2) meters}. extent: half-width of the map in meters. Draws whoever of the pair is currently in view."""
    cv2.rectangle(canvas, (x0, y0), (x0 + size, y0 + size), (30, 30, 30), -1)
    cv2.rectangle(canvas, (x0, y0), (x0 + size, y0 + size), (110, 110, 110), 1)
    ids = list(pair)
    P = {t: pos[t][min(frame, len(pos[t]) - 1)] for t in ids}
    here = [t for t in ids if np.isfinite(P[t]).all()]
    if not here:
        put_text(canvas, "pair not in view", (x0 + 6, y0 + size // 2), 0.38, (170, 170, 170))
        return
    mid = np.mean([P[t] for t in here], axis=0)

    def to_px(p):
        return int(x0 + size / 2 + (p[0] - mid[0]) / extent * size / 2), int(y0 + size / 2 - (p[1] - mid[1]) / extent * size / 2)
    target = active[0]["target"] if active else ids[1]
    if target in here:
        c = to_px(P[target])
        for r_m, col in ((0.46, (60, 60, 160)), (1.2, (60, 110, 160)), (3.7, (70, 110, 70))):
            rp = int(r_m / extent * size / 2)
            if 2 < rp < size:
                cv2.circle(canvas, c, rp, col, 1)
    for t in here:
        trail = pos[t][max(0, frame - int(3 * fps)):frame + 1:max(1, int(fps / 5))]
        pts = [to_px(p) for p in trail if np.isfinite(p).all()]
        if len(pts) > 1:
            cv2.polylines(canvas, [np.array(pts, np.int32)], False, (150, 150, 150), 1)
        col = (0, 140, 255) if (active and t == active[0]["actor"]) else (255, 170, 40)
        cv2.circle(canvas, to_px(P[t]), 5, col, -1)
        put_text(canvas, f"id{t}", (to_px(P[t])[0] + 6, to_px(P[t])[1] - 4), 0.35, (255, 255, 255))
    if len(here) == 2:
        put_text(canvas, f"{np.linalg.norm(P[ids[0]] - P[ids[1]]):.1f} m", (x0 + 4, y0 + size - 5), 0.38, (230, 230, 230))
    else:
        put_text(canvas, "only one in view", (x0 + 4, y0 + size - 5), 0.33, (170, 170, 170))
    put_text(canvas, f"+-{extent:.0f} m", (x0 + size - 46, y0 + 12), 0.33, (160, 160, 160))


# ---------------------------------------------------------------- strip / header / card
def draw_header(canvas, w, eps, scale):
    cv2.rectangle(canvas, (0, 0), (w, HEADER_H), (25, 25, 25), -1)
    if not eps:
        put_text(canvas, "no cue active at this moment", (6, 22), scale, (170, 170, 170))
    for k, e in enumerate(eps[:3]):
        put_text(canvas, _fit(episode_text(e), w, scale), (6, 17 + 18 * k), scale, COLOR_OF.get(e["pred"], (255, 255, 255)), 1)


def draw_strip(canvas, top, w, story, frame, n_frames, cut_f):
    h = strip_height()
    cv2.rectangle(canvas, (0, top), (w, top + h), (25, 25, 25), -1)
    x0, x1 = 34, w - 6
    X = lambda f: int(x0 + (x1 - x0) * f / max(n_frames - 1, 1))
    eps = key_episodes(story)
    for r, (label, preds, col) in enumerate(FAMILY):
        y = top + PAD + r * ROW
        put_text(canvas, label, (1, y + ROW - 1), 0.28, col)
        cv2.rectangle(canvas, (x0, y + 1), (x1, y + ROW - 1), (45, 45, 45), -1)
        for e in eps:
            if e["pred"] in preds and e["start_f"] < cut_f:
                cv2.rectangle(canvas, (X(e["start_f"]), y + 1), (X(min(e["end_f"], cut_f - 1)), y + ROW - 1), col, -1)
    cy0 = top + 2 * PAD + len(FAMILY) * ROW
    cv2.rectangle(canvas, (x0, cy0), (x1, cy0 + CURVE_H), (40, 40, 40), -1)
    curve = story["pairs"][story["key_pair"]]["cue_curve"] if story.get("key_pair") else []
    if curve:
        vals = np.array([c[1] for c in curve])
        ymax = max(2.0, float(np.abs(vals).max()))
        zero = cy0 + int(CURVE_H * 0.62)
        cv2.line(canvas, (x0, zero), (x1, zero), (90, 90, 90), 1)
        pts = np.array([(X(int(c[0] * story["fps"])), int(zero - c[1] / ymax * CURVE_H * 0.55)) for c in curve], np.int32)
        cv2.polylines(canvas, [pts], False, (255, 255, 0), 1)
    put_text(canvas, "cue", (1, cy0 + 20), 0.28, (255, 255, 0))
    if cut_f < n_frames:
        cv2.line(canvas, (X(cut_f), top), (X(cut_f), top + h), (0, 0, 255), 2)
    cv2.line(canvas, (X(min(frame, n_frames - 1)), top), (X(min(frame, n_frames - 1)), top + h), (255, 255, 255), 1)


def withheld_card(w, h, story, cut, esc_text):
    img = np.full((h, w, 3), (45, 25, 25), np.uint8)
    sc = min(0.7, w / 700)
    lines = ["PHYSICAL ACT STARTS" if story.get("escalation") and cut["reason"] == "before detected escalation" else "END OF PRE-VIOLENCE FOOTAGE",
             "footage withheld", esc_text]
    for e in key_episodes(story)[:5]:
        if e["pred"] not in ACT_PREDS:
            lines.append(_fit(f"{e['start_s']:.1f}-{e['end_s']:.1f}s  {e['pred'].replace('_', ' ')}  id{e['actor']} -> id{e['target']}", w, sc * 0.8))
    y = max(30, h // 2 - 18 * len(lines) // 2)
    for k, text in enumerate(lines):
        if not text:
            continue
        s = sc * (1.2 if k == 0 else 0.9)
        size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, s, 2 if k == 0 else 1)[0]
        put_text(img, text, (max(4, (w - size[0]) // 2), y + 24 * k), s, (255, 255, 255) if k == 0 else (205, 205, 205), 2 if k == 0 else 1)
    return img


def _draw_people(img, t, rows, key_ids):
    for k in rows:
        x1, y1, x2, y2 = t["bbox"][k].astype(int)
        tid = int(t["track_id"][k])
        hot = tid in key_ids
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 120, 255) if hot else (0, 200, 0), 3 if hot else 1)
        put_text(img, f"id{tid}", (x1, max(12, y1 - 4)), 0.5, (255, 255, 255), 1)


def _draw_doors(img, layout):
    if not layout or not layout.get("reliable"):
        return
    h, w = img.shape[:2]
    for d in layout.get("doors", []):
        cv2.rectangle(img, (int(d["x1"] * w), int(d["y1"] * h)), (int(d["x2"] * w), int(d["y2"] * h)), (255, 255, 0), 1)
        put_text(img, "door", (int(d["x1"] * w), int(d["y1"] * h) - 3), 0.4, (255, 255, 0))


# ---------------------------------------------------------------- rendering
def render_story_video(video_path, t, story, layout, cut, ctx, rep, out_path):
    """Write the story video; None when there is too little pre-violence footage."""
    if not cut["show"] or story["no_pair"]:
        return None
    pair = tuple(story["pairs"][story["key_pair"]]["pair"])
    pos = pair_ground(t, ctx, pair)
    shown = slice(0, max(1, cut["cut_f"]))
    dd = np.linalg.norm(pos[pair[0]][shown] - pos[pair[1]][shown], axis=1)
    dmax = float(np.nanmax(dd)) if np.isfinite(dd).any() else 3.0
    extent = float(np.clip(1.3 * dmax, 3.0, 12.0))
    cap = cv2.VideoCapture(str(video_path))
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or story["fps"]
    n_frames, sh = story["n_frames"], strip_height()
    H = HEADER_H + h + sh
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(".raw.mp4")
    vw = cv2.VideoWriter(str(tmp), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, H))
    rows = {}
    for k, f in enumerate(t["frame_idx"].tolist()):
        rows.setdefault(f, []).append(k)
    sc = min(0.55, w / 900)
    mm = int(min(150, w * 0.32, h * 0.45))
    for f in range(cut["cut_f"]):
        ok, img = cap.read()
        if not ok:
            break
        _draw_people(img, t, rows.get(f, []), set(pair))
        _draw_doors(img, layout)
        canvas = np.zeros((H, w, 3), np.uint8)
        act = active_episodes(story, f)
        draw_header(canvas, w, act, sc)
        canvas[HEADER_H:HEADER_H + h] = img
        draw_minimap(canvas, w - mm - 4, HEADER_H + 4, mm, pos, pair, f, fps, act, extent)
        put_text(canvas, f"{f / fps:.1f}s", (6, HEADER_H + 16), 0.5, (255, 255, 255))
        draw_strip(canvas, HEADER_H + h, w, story, f, n_frames, cut["cut_f"])
        vw.write(canvas)
    cap.release()
    if cut["trimmed"]:
        esc = story.get("escalation")
        txt = f"first act cue: {esc['kind'].replace('_', ' ')} at {esc['time_s']:.1f}s" if esc else ""
        card = withheld_card(w, H, story, cut, txt)
        for _ in range(max(1, round(rep["withheld_card_s"] * fps))):
            vw.write(card)
    vw.release()
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", str(tmp), "-c:v", "libx264", "-pix_fmt", "yuv420p", str(out_path)], check=True)
    tmp.unlink()
    return out_path


def pick_story_frames(story, cut, max_frames=6):
    """[(frame, caption)] one keyframe per distinct predicate episode, before the cut."""
    last = max(0, cut["cut_f"] - 1)
    seen, out = set(), []
    for e in key_episodes(story):
        if e["start_f"] >= cut["cut_f"] or e["pred"] in seen or e["pred"] in ACT_PREDS:
            continue
        seen.add(e["pred"])
        out.append((min(last, (e["start_f"] + e["end_f"]) // 2), episode_text(e)))
    out = out[:max_frames]
    return out or [(f, "no cue detected at this moment") for f in sorted({0, last // 2, last})]


def render_story_board(video_path, t, story, cut, out_path, tile_w=440):
    if not cut["show"]:
        return None
    cap = cv2.VideoCapture(str(video_path))
    fps = story["fps"]
    rows = {}
    for k, f in enumerate(t["frame_idx"].tolist()):
        rows.setdefault(f, []).append(k)
    pair = set(story["pairs"][story["key_pair"]]["pair"]) if story.get("key_pair") else set()
    tiles = []
    for f, caption in pick_story_frames(story, cut):
        cap.set(cv2.CAP_PROP_POS_FRAMES, f)
        ok, img = cap.read()
        if not ok:
            continue
        _draw_people(img, t, rows.get(f, []), pair)
        img = cv2.resize(img, (tile_w, int(img.shape[0] * tile_w / img.shape[1])))
        bar = np.full((40, tile_w, 3), (25, 25, 25), np.uint8)
        put_text(bar, f"t={f / fps:.1f}s", (4, 13), 0.38, (255, 255, 255))
        put_text(bar, _fit(caption, tile_w, 0.36), (4, 31), 0.36, (0, 215, 255))
        tiles.append(np.vstack([bar, img]))
    cap.release()
    if not tiles:
        return None
    hmax = max(x.shape[0] for x in tiles)
    tiles = [cv2.copyMakeBorder(x, 0, hmax - x.shape[0], 0, 0, cv2.BORDER_CONSTANT, value=(25, 25, 25)) for x in tiles]
    cols = 1 if len(tiles) == 1 else 2
    while len(tiles) % cols:
        tiles.append(np.full_like(tiles[0], 25))
    grid = np.vstack([np.hstack(tiles[i:i + cols]) for i in range(0, len(tiles), cols)])
    title = np.full((74, grid.shape[1], 3), (25, 25, 25), np.uint8)
    put_text(title, f"{story['clip_id']} ({story['category']})   shown {cut['cut_s']:.1f}s of {story['n_frames'] / story['fps']:.1f}s ({cut['reason']})", (6, 18), 0.5, (255, 255, 255))
    put_text(title, _fit(scene_line(story["scene"]), grid.shape[1], 0.38), (6, 40), 0.38, (190, 190, 190))
    put_text(title, _fit(summary_line(story), grid.shape[1], 0.38), (6, 60), 0.38, (0, 215, 255))
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), np.vstack([title, grid]))
    return out_path


def render_clip(clip, cfg, overrides=None, out_dir=None):
    """Render story video + storyboard for one manifest_clean entry."""
    ctx_dir = resolve_path(cfg["context"]["context_dir"])
    cat, cid = clip["category"], clip["clip_id"]
    story = load_story(ctx_dir, cat, cid)
    if story is None:
        raise FileNotFoundError(f"no story for {cid}: run python -m src.context.story")
    t = load_tracks(resolve_path(cfg["assm"]["tracks_dir"]) / cat / f"{cid}.npz")
    _, layout = load_layout(ctx_dir, cat, cid)
    gp = resolve_path(cfg["gate"]["gate_dir"]) / cat / f"{cid}_gate.json"
    gate_esc = None
    if gp.exists():
        from src.assm.gate import load_gate
        g = load_gate(gp)
        gate_esc = g["escalation"]["time_s"] if g.get("escalation") else None
    cut = story_cut(story, cat, gate_esc, cfg["report"], overrides)
    out = Path(out_dir) if out_dir else resolve_path(cfg["report"]["report_dir"])
    video = resolve_path(clip["path"])
    v = render_story_video(video, t, story, layout, cut, cfg["context"], cfg["report"], out / "videos" / cat / f"{cid}_story.mp4")
    s = render_story_board(video, t, story, cut, out / "storyboards" / cat / f"{cid}_story.jpg")
    return {"clip_id": cid, "category": cat, "cut": cut, "video": v, "storyboard": s}


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    ap = argparse.ArgumentParser(description="Render the story video + storyboard for clips")
    ap.add_argument("clips", nargs="+")
    ap.add_argument("--config", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)
    clean = json.loads(resolve_path(cfg["preprocess"]["clean_manifest_path"]).read_text("utf-8"))
    by_id = {c["clip_id"]: c for c in clean["clips"]}
    ov = load_overrides(cfg["report"]["overrides_csv"])
    for cid in args.clips:
        r = render_clip(by_id[cid], cfg, ov, args.out)
        c = r["cut"]
        print(f"{cid}: shows {c['cut_s']}s ({c['reason']})  video={r['video']}  storyboard={r['storyboard']}")


if __name__ == "__main__":
    main()
