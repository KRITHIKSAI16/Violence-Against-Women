"""Phase 1 deliverable: the buildup video and storyboard for one clip (violence not shown).

Takes a cleaned clip (M2), its tracks (M8a) and its behavior analysis (M9) and draws the pre-violence
CONTEXT: who is approaching / following / hovering near / blocking whom, for how long. The footage stops
at the cut point (before the detected escalation) and is replaced by a "footage withheld" card.

Cut rule (config `report:`):
  1. manual override in configs/clip_overrides.csv (clip_id,cut_s)
  2. detected escalation  ->  cut = escalation time - cut_margin_s            (any category except Normal)
  3. act category (Assassination / Chain_Snatching / Kidnapping), nothing detected -> drop the last act_tail_trim_s
  4. otherwise (Stalking, Harassment, Normal) the whole clip is shown
Honest limit: nobody annotated where the act starts, so rules 2-3 are heuristics and cannot guarantee that
no violence is visible. Check showcase clips once, and use rule 1 for the ones you present.

Outputs:  <out>/<Category>/<clip>_buildup.mp4   and   <out>/<Category>/<clip>_storyboard.jpg

Usage:  python -m src.report.buildup_video <clip_id> [<clip_id> ...] [--config CFG]
"""
import argparse
import csv
import json
import logging
import subprocess
from pathlib import Path

import cv2
import numpy as np

from src.assm.gate import APPROACH, CORNER, ESCALATION, FOLLOW, HOVER, RANK, load_gate
from src.assm.track_poses import load_tracks
from src.config import load_config, resolve_path

log = logging.getLogger(__name__)

BUILDUP_STATES = [APPROACH, FOLLOW, HOVER, CORNER]
COLOR = {APPROACH: (0, 215, 255), FOLLOW: (0, 140, 255), HOVER: (255, 170, 40), CORNER: (40, 40, 235)}  # BGR
SHORT = {APPROACH: "APPR", FOLLOW: "FOLL", HOVER: "HOVR", CORNER: "CORN"}
HEADER_H, STRIP_ROW, STRIP_PAD = 46, 11, 6


# ------------------------------------------------------------------ cut point
def load_overrides(path):
    """clip_id -> cut_s from a CSV (header clip_id,cut_s; '#' lines ignored). Missing file = no overrides."""
    p = resolve_path(path) if path else None
    if not p or not p.exists():
        return {}
    out = {}
    with open(p, newline="", encoding="utf-8") as f:
        for row in csv.reader(line for line in f if not line.lstrip().startswith("#")):
            if len(row) >= 2 and row[0].strip() != "clip_id":
                out[row[0].strip()] = float(row[1])
    return out


def cut_point(gate, category, rep, overrides=None):
    """Where the footage stops. Returns {cut_s, cut_f, reason, trimmed, show} (show False = too short to render)."""
    dur, fps = gate["duration_s"], gate["fps"]
    cut, reason = dur, "whole clip"
    if overrides and gate["clip_id"] in overrides:
        cut, reason = min(dur, overrides[gate["clip_id"]]), "manual override"
    elif category != "Normal" and gate.get("escalation"):
        cut, reason = max(0.0, gate["escalation"]["time_s"] - rep["cut_margin_s"]), "before detected escalation"
    elif category in rep["act_categories"]:
        cut, reason = max(0.0, dur - rep["act_tail_trim_s"]), "uniform tail trim (act category)"
    return {"cut_s": round(cut, 3), "cut_f": int(cut * fps), "reason": reason, "trimmed": cut < dur - 1e-6,
            "show": cut >= rep["min_video_s"]}


# ------------------------------------------------------------------ captions
def phrase(seg, other):
    a, t = seg.get("actor"), seg.get("target", other)
    return {APPROACH: f"id{a} APPROACHING id{other}", FOLLOW: f"id{a} FOLLOWING id{t}",
            HOVER: f"id{a} HOVERING near id{t}", CORNER: f"id{a} BLOCKING id{t}'s way out"}[seg["state"]]


def key_segments(gate):
    """Non-escalation segments of the key pair (the ones drawn), ordered by start."""
    if not gate.get("key_pair"):
        return []
    kp = tuple(gate["key_pair"])
    return sorted((s for s in gate["segments"] if (s["id_i"], s["id_j"]) == kp and s["state"] != ESCALATION),
                  key=lambda s: s["start_f"])


def active_captions(gate, frame):
    """[(text, state)] for the key pair at `frame`, highest rank first, with time-so-far."""
    fps = gate["fps"]
    out = []
    for s in key_segments(gate):
        if s["start_f"] <= frame <= s["end_f"]:
            other = s["id_j"] if s.get("actor") == s["id_i"] else s["id_i"]
            out.append((f"{phrase(s, other)}  {(frame - s['start_f'] + 1) / fps:.1f}s", s["state"]))
    return sorted(out, key=lambda x: -RANK[x[1]])


# ------------------------------------------------------------------ drawing
def put_text(img, text, org, scale, color, thick=1):
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thick + 2, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)


def _rows_by_frame(t):
    d = {}
    for k, f in enumerate(t["frame_idx"].tolist()):
        d.setdefault(f, []).append(k)
    return d


def draw_people(img, t, rows, key_ids):
    for k in rows:
        x1, y1, x2, y2 = t["bbox"][k].astype(int)
        tid = int(t["track_id"][k])
        hot = tid in key_ids
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 120, 255) if hot else (0, 200, 0), 3 if hot else 1)
        put_text(img, f"id{tid}", (x1, max(12, y1 - 4)), 0.5, (255, 255, 255), 1)


def draw_header(canvas, caps, w, nothing_text):
    cv2.rectangle(canvas, (0, 0), (w, HEADER_H), (25, 25, 25), -1)
    sc = min(0.6, w / 800)
    if not caps:
        put_text(canvas, nothing_text, (6, 28), sc, (170, 170, 170))
    for n, (text, state) in enumerate(caps[:2]):
        put_text(canvas, text, (6, 19 + 21 * n), sc, COLOR[state], 1)


def draw_strip(canvas, gate, top, w, frame, n_frames, cut_f):
    """Phase strip: one row per behavior, bar where it is active, cursor at `frame`, red marker at the cut."""
    segs = key_segments(gate)
    h = len(BUILDUP_STATES) * STRIP_ROW + 2 * STRIP_PAD
    cv2.rectangle(canvas, (0, top), (w, top + h), (25, 25, 25), -1)
    x0, x1 = 40, w - 6
    X = lambda f: int(x0 + (x1 - x0) * f / max(n_frames - 1, 1))
    for r, st in enumerate(BUILDUP_STATES):
        y = top + STRIP_PAD + r * STRIP_ROW
        put_text(canvas, SHORT[st], (2, y + STRIP_ROW - 2), 0.3, COLOR[st])
        cv2.rectangle(canvas, (x0, y + 2), (x1, y + STRIP_ROW - 2), (50, 50, 50), -1)
        for s in segs:
            if s["state"] == st and s["start_f"] < cut_f:
                cv2.rectangle(canvas, (X(s["start_f"]), y + 2), (X(min(s["end_f"], cut_f - 1)), y + STRIP_ROW - 2), COLOR[st], -1)
    if cut_f < n_frames:
        cv2.line(canvas, (X(cut_f), top), (X(cut_f), top + h), (0, 0, 255), 2)
    cv2.line(canvas, (X(min(frame, n_frames - 1)), top), (X(min(frame, n_frames - 1)), top + h), (255, 255, 255), 1)
    return h


def strip_height():
    return len(BUILDUP_STATES) * STRIP_ROW + 2 * STRIP_PAD


def withheld_card(w, h, lines):
    img = np.full((h, w, 3), (45, 25, 25), np.uint8)
    sc = min(0.8, w / 640)
    y = h // 2 - 30 * (len(lines) // 2)
    for n, text in enumerate(lines):
        size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, sc if n else sc * 1.2, 2)[0]
        put_text(img, text, (max(4, (w - size[0]) // 2), y + n * int(34 * sc + 8)), sc if n else sc * 1.2,
                 (255, 255, 255) if n == 0 else (200, 200, 200), 2 if n == 0 else 1)
    return img


def card_lines(gate, cut, rep):
    if gate.get("escalation") and cut["reason"] == "before detected escalation":
        head = "ESCALATION POINT"
    elif cut["reason"] == "manual override":
        head = "END OF PRE-VIOLENCE FOOTAGE"
    else:
        head = "END OF PRE-VIOLENCE FOOTAGE"
    lines = [head, "footage withheld"]
    for s in key_segments(gate)[:4]:
        lines.append(f"{s['state']} {s['start_s']:.1f}-{s['end_s']:.1f}s")
    return lines


# ------------------------------------------------------------------ outputs
def _encode(tmp, out):
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", str(tmp), "-c:v", "libx264", "-pix_fmt", "yuv420p", str(out)],
                   check=True)
    Path(tmp).unlink()


def render_buildup_video(video_path, t, gate, cut, rep, out_path):
    """Write the buildup video; returns out_path, or None when there is too little pre-violence footage."""
    if not cut["show"]:
        return None
    cap = cv2.VideoCapture(str(video_path))
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or gate["fps"]
    n_frames = gate["n_frames"]
    sh = strip_height()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(".raw.mp4")
    vw = cv2.VideoWriter(str(tmp), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, HEADER_H + h + sh))
    rows = _rows_by_frame(t)
    key_ids = set(gate["key_pair"] or [])
    if gate["no_interaction"]:
        nothing = "fewer than 2 people tracked together"
    elif gate["key_pair"]:
        nothing = "no flagged behavior at this moment"
    else:
        nothing = "no sustained buildup behavior detected in this clip"
    for f in range(cut["cut_f"]):
        ok, img = cap.read()
        if not ok:
            break
        draw_people(img, t, rows.get(f, []), key_ids)
        canvas = np.zeros((HEADER_H + h + sh, w, 3), np.uint8)
        draw_header(canvas, active_captions(gate, f), w, nothing)
        canvas[HEADER_H:HEADER_H + h] = img
        put_text(canvas, f"{f / fps:.1f}s", (w - 52, HEADER_H + 16), 0.5, (255, 255, 255))
        draw_strip(canvas, gate, HEADER_H + h, w, f, n_frames, cut["cut_f"])
        vw.write(canvas)
    cap.release()
    if cut["trimmed"]:
        card = np.zeros((HEADER_H + h + sh, w, 3), np.uint8)
        card[:] = withheld_card(w, HEADER_H + h + sh, card_lines(gate, cut, rep))
        for _ in range(max(1, round(rep["withheld_card_s"] * fps))):
            vw.write(card)
    vw.release()
    _encode(tmp, out_path)
    return out_path


def pick_storyboard_frames(gate, cut):
    """[(frame, caption)] up to 6 keyframes strictly before the cut."""
    fps, last = gate["fps"], max(0, cut["cut_f"] - 1)
    picks, seen = [], set()
    for s in key_segments(gate):
        if s["start_f"] >= cut["cut_f"] or s["state"] in seen:
            continue
        seen.add(s["state"])
        f = min(last, s["start_f"] + min(int(fps), (s["end_f"] - s["start_f"]) // 2))
        picks.append((f, s))
    out = [(f, caption_for(gate, s, f)) for f, s in picks][:6]
    if not out:   # nothing detected: show the beginning, middle and end of what is shown
        out = [(f, "no sustained buildup behavior detected") for f in sorted({0, last // 2, last})]
    return out


def caption_for(gate, seg, frame):
    other = seg["id_j"] if seg.get("actor") == seg["id_i"] else seg["id_i"]
    return f"{phrase(seg, other)}  (starts {seg['start_s']:.1f}s)"


def render_storyboard(video_path, t, gate, cut, out_path, tile_w=420):
    """Captioned keyframes at the behavior transitions; returns out_path or None if nothing to show."""
    if not cut["show"]:
        return None
    cap = cv2.VideoCapture(str(video_path))
    fps = gate["fps"]
    rows = _rows_by_frame(t)
    key_ids = set(gate["key_pair"] or [])
    tiles = []
    for f, caption in pick_storyboard_frames(gate, cut):
        cap.set(cv2.CAP_PROP_POS_FRAMES, f)
        ok, img = cap.read()
        if not ok:
            continue
        draw_people(img, t, rows.get(f, []), key_ids)
        scale = tile_w / img.shape[1]
        img = cv2.resize(img, (tile_w, int(img.shape[0] * scale)))
        bar = np.full((34, tile_w, 3), (25, 25, 25), np.uint8)
        put_text(bar, f"t={f / fps:.1f}s", (4, 14), 0.4, (255, 255, 255))
        put_text(bar, caption, (4, 28), 0.4, (0, 215, 255))
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
    title = np.full((52, grid.shape[1], 3), (25, 25, 25), np.uint8)
    seq = " > ".join(gate["phase_sequence"]) or "none"
    put_text(title, f"{gate['clip_id']} ({gate['category']})   buildup: {seq}", (6, 20), 0.55, (255, 255, 255))
    put_text(title, f"shown {cut['cut_s']:.1f}s of {gate['duration_s']:.1f}s  ({cut['reason']})"
                    f"   ordered progression: {'yes' if gate['ordered_progression'] else 'no'}", (6, 42), 0.45, (190, 190, 190))
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), np.vstack([title, grid]))
    return out_path


def render_clip(clip, cfg, overrides=None, out_dir=None):
    """Render video + storyboard for one manifest_clean entry. Returns a dict describing what was written."""
    rep = cfg["report"]
    cat, cid = clip["category"], clip["clip_id"]
    gate = load_gate(resolve_path(cfg["gate"]["gate_dir"]) / cat / f"{cid}_gate.json")
    t = load_tracks(resolve_path(cfg["assm"]["tracks_dir"]) / cat / f"{cid}.npz")
    cut = cut_point(gate, cat, rep, overrides)
    out = Path(out_dir) if out_dir else resolve_path(rep["report_dir"])
    video = resolve_path(clip["path"])
    v = render_buildup_video(video, t, gate, cut, rep, out / "videos" / cat / f"{cid}_buildup.mp4")
    s = render_storyboard(video, t, gate, cut, out / "storyboards" / cat / f"{cid}_storyboard.jpg")
    return {"clip_id": cid, "category": cat, "cut": cut, "video": v, "storyboard": s}


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    ap = argparse.ArgumentParser(description="Render buildup video + storyboard for clips")
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
