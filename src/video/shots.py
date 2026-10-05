"""Layer 0: shot segmentation. Find the editing cuts so that nothing temporal is ever computed ACROSS a cut.

Many ExtrAnom clips are edited (compilations, news, replays, several camera views). Tracking ids, speeds, "following", camera motion and every
window assume one continuous scene; a cut makes a different scene look like a person teleporting. So every later stage works per shot:
the tracker restarts at a cut (ids are made unique per shot), identity stitching never joins tracks of different shots, camera motion restarts,
and pair features / windows / stories stay inside one shot.

Methods (config `shots.method`):
  transnet    TransNetV2 (neural, handles gradual transitions; BBC F1 0.967 vs PySceneDetect 0.889)   needs `pip install transnetv2-pytorch`
  scenedetect PySceneDetect AdaptiveDetector (CPU, content based)                                  needs `pip install scenedetect`
  hist        colour-histogram + pixel-difference cuts (no dependencies; hard cuts only; used in tests and as a last resort)
  auto        transnet if installed, else scenedetect, else hist
Very short shots (default < 0.4 s: flashes, end cards) are merged into their neighbour.

Output per clip: <context_dir>/<Category>/<clip>_shots.json
Usage:  python -m src.video.shots [--config CFG] [--clips A B] [--method M] [--force] [--sheet]
"""
import argparse
import json
import logging
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

from src.config import load_config, resolve_path

log = logging.getLogger(__name__)
SHOT_BASE = 100000      # track ids are made globally unique as shot_index * SHOT_BASE + local_id (see src/assm/track_poses.py)
_TRANSNET = {}


# ---------------------------------------------------------------- detectors (each returns a list of (start_frame, end_frame) inclusive)
def hist_shots(video_path, corr_thr=0.55, diff_thr=25.0):
    """Hard cuts from the drop in HSV histogram correlation AND a large mean pixel change between consecutive frames."""
    cap = cv2.VideoCapture(str(video_path))
    prev_h = prev_s = None
    cuts, k = [], 0
    while True:
        ok, img = cap.read()
        if not ok:
            break
        small = cv2.resize(img, (160, 90))
        hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
        h = cv2.normalize(cv2.calcHist([hsv], [0, 1, 2], None, [8, 8, 8], [0, 180, 0, 256, 0, 256]), None).flatten()
        if prev_h is not None:
            corr = cv2.compareHist(prev_h, h, cv2.HISTCMP_CORREL)
            diff = float(np.abs(small.astype(int) - prev_s.astype(int)).mean())
            if corr < corr_thr and diff > diff_thr:
                cuts.append(k)
        prev_h, prev_s = h, small
        k += 1
    cap.release()
    return _cuts_to_shots(cuts, k)


def _cuts_to_shots(cut_frames, n_frames):
    starts = [0] + sorted(set(int(c) for c in cut_frames if 0 < c < n_frames))
    ends = [s - 1 for s in starts[1:]] + [n_frames - 1]
    return list(zip(starts, ends))


def transnet_shots(video_path, threshold=0.5, device="auto"):
    from transnetv2_pytorch import TransNetV2
    key = device
    if key not in _TRANSNET:
        m = TransNetV2(device=device)
        m.eval()
        _TRANSNET[key] = m
    scenes = _TRANSNET[key].detect_scenes(str(video_path), threshold=threshold)
    return [(int(s["start_frame"]), int(s["end_frame"])) for s in scenes]


def scenedetect_shots(video_path, n_frames):
    from scenedetect import AdaptiveDetector, detect
    scenes = detect(str(video_path), AdaptiveDetector())
    cuts = [s[0].get_frames() for s in scenes[1:]]
    return _cuts_to_shots(cuts, n_frames)


def merge_short(shots, fps, min_shot_s):
    """Merge shots shorter than min_shot_s into the previous shot (or the next one for the first). Returns a new list."""
    min_f = max(1, int(round(min_shot_s * fps)))
    out = [list(s) for s in shots]
    changed = True
    while changed and len(out) > 1:
        changed = False
        for k, (a, b) in enumerate(out):
            if b - a + 1 < min_f:
                if k > 0:
                    out[k - 1][1] = b
                else:
                    out[1][0] = a
                del out[k]
                changed = True
                break
    return [tuple(s) for s in out]


# ---------------------------------------------------------------- public API
def detect_shots(video_path, method="auto", threshold=0.5, min_shot_s=0.4, device="auto", clip_id=""):
    """Shots of one video as a JSON-serialisable dict. Frame indices refer to the given video file (use the processed clip)."""
    cap = cv2.VideoCapture(str(video_path))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap.release()
    used = method
    if method == "auto":
        for cand in ("transnet", "scenedetect", "hist"):
            try:
                if cand == "transnet":
                    import transnetv2_pytorch  # noqa: F401
                elif cand == "scenedetect":
                    import scenedetect  # noqa: F401
                used = cand
                break
            except ImportError:
                continue
        else:
            used = "hist"
    try:
        if used == "transnet":
            raw = transnet_shots(video_path, threshold, device)
        elif used == "scenedetect":
            raw = scenedetect_shots(video_path, n)
        else:
            raw = hist_shots(video_path)
    except Exception as e:  # a detector failure must not stop a long run: fall back to hard-cut detection
        log.warning("shot detection (%s) failed for %s: %s; using histogram cuts", used, clip_id or video_path, e)
        used, raw = "hist", hist_shots(video_path)
    if raw and raw[-1][1] < n - 1:       # the detector may stop before the last frame
        raw[-1] = (raw[-1][0], n - 1)
    shots = merge_short(raw or [(0, max(0, n - 1))], fps, min_shot_s)
    return {"clip_id": clip_id, "fps": float(fps), "n_frames": n, "method": used, "threshold": threshold, "min_shot_s": min_shot_s,
            "shots": [{"id": i, "start_f": a, "end_f": b, "start_s": round(a / fps, 3), "end_s": round((b + 1) / fps, 3)} for i, (a, b) in enumerate(shots)],
            "cuts_s": [round(a / fps, 3) for a, _ in shots[1:]]}


def shot_starts(shots_json):
    return [s["start_f"] for s in shots_json["shots"]]


def shot_index(frame, starts):
    """Index of the shot a frame belongs to (starts = sorted shot start frames). Works on scalars and arrays."""
    return np.searchsorted(np.asarray(starts), frame, side="right") - 1


def save_shots(out_dir, category, clip_id, d):
    p = Path(out_dir) / category / f"{clip_id}_shots.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(d, indent=1), encoding="utf-8")
    return p


def load_shots(out_dir, category, clip_id):
    p = Path(out_dir) / category / f"{clip_id}_shots.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def local_id(track_id):
    """Per-shot track id (what a person would call 'id3')."""
    return int(track_id) % SHOT_BASE


def shot_of_id(track_id):
    return int(track_id) // SHOT_BASE


# ---------------------------------------------------------------- contact sheet (look at the cuts)
def cut_sheet(video_path, shots_json, out_path, tile_w=300):
    """For every cut: the last frame of the shot before and the first frame of the shot after, side by side with the time."""
    if not shots_json["cuts_s"]:
        return None
    cap = cv2.VideoCapture(str(video_path))
    rows = []
    for s_prev, s_next in zip(shots_json["shots"][:-1], shots_json["shots"][1:]):
        pair = []
        for f in (s_prev["end_f"], s_next["start_f"]):
            cap.set(cv2.CAP_PROP_POS_FRAMES, f)
            ok, img = cap.read()
            if not ok:
                img = np.zeros((90, 160, 3), np.uint8)
            img = cv2.resize(img, (tile_w, int(img.shape[0] * tile_w / img.shape[1])))
            pair.append(img)
        h = max(p.shape[0] for p in pair)
        pair = [cv2.copyMakeBorder(p, 0, h - p.shape[0], 0, 0, cv2.BORDER_CONSTANT) for p in pair]
        row = np.hstack(pair)
        cv2.putText(row, f"cut at {s_next['start_s']:.2f}s  (last frame of shot {s_prev['id']} | first of shot {s_next['id']})", (4, 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1, cv2.LINE_AA)
        rows.append(row)
    cap.release()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), np.vstack(rows))
    return Path(out_path)


# ---------------------------------------------------------------- CLI
def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    ap = argparse.ArgumentParser(description="Layer 0: shot segmentation of every cleaned clip")
    ap.add_argument("--config", default=None)
    ap.add_argument("--clips", nargs="*", default=None)
    ap.add_argument("--method", default=None, help="auto | transnet | scenedetect | hist (default from config)")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--sheet", action="store_true", help="write a contact sheet of the cuts for clips that have any")
    args = ap.parse_args()
    cfg = load_config(args.config)
    sc = cfg["shots"]
    method = args.method or sc["method"]
    clean = json.loads(resolve_path(cfg["preprocess"]["clean_manifest_path"]).read_text("utf-8"))
    out_dir = resolve_path(cfg["context"]["context_dir"])
    done = []
    for k, c in enumerate(clean["clips"], 1):
        if args.clips and c["clip_id"] not in args.clips:
            continue
        d = None if args.force else load_shots(out_dir, c["category"], c["clip_id"])
        video = resolve_path(c["path"])
        if d is None:
            d = detect_shots(video, method, float(sc["threshold"]), float(sc["min_shot_s"]), sc["device"], c["clip_id"])
            save_shots(out_dir, c["category"], c["clip_id"], d)
        if args.sheet and d["cuts_s"]:
            cut_sheet(video, d, out_dir / "shot_sheets" / c["category"] / f"{c['clip_id']}.jpg")
        done.append((c, d))
        if k % 50 == 0:
            print(f"  shots: {k}/{len(clean['clips'])} clips", flush=True)
    by = defaultdict(list)
    for c, d in done:
        by[c["category"]].append(d)
    print(f"\n{len(done)} clips -> {out_dir}  (method: {done[0][1]['method'] if done else method})")
    print(f"{'category':<16}{'clips':>6}{'with cuts':>11}{'cuts total':>12}{'mean shots':>12}{'median shot s':>15}")
    for cat, lst in sorted(by.items()):
        lens = [s["end_s"] - s["start_s"] for d in lst for s in d["shots"]]
        print(f"{cat:<16}{len(lst):>6}{sum(bool(d['cuts_s']) for d in lst):>11}{sum(len(d['cuts_s']) for d in lst):>12}"
              f"{np.mean([len(d['shots']) for d in lst]):>12.2f}{np.median(lens):>15.1f}")
    for c, d in done:
        if d["cuts_s"]:
            print(f"  {c['clip_id']:<22} cuts at {d['cuts_s']}")


if __name__ == "__main__":
    main()
