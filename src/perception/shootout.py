"""Layer 1: perception shootout. Which detector / pose model / tracker finds the most usable people, most reliably, for the time it costs?

There is no box ground truth for ExtrAnom, so the comparison uses measurable proxies computed from the cached tracks of each configuration,
on the same clips and the same first `max_s` seconds, run shot by shot (so cuts are handled identically for every candidate):

  pair_clips      share of clips that end up with at least one USABLE PAIR (two tracks of one shot co-existing >= 1 s, both confident)
  pair_seconds    seconds of co-tracked pair time per clip (how much interaction can be analysed)
  ids_per_person  track ids per person in view (ids after stitching / maximum people in a frame, per shot): 1.0 = no identity fragmentation
  long_track      share of person-frames that belong to tracks lasting >= 3 s
  kpt_good        share of person-rows with at least 10 of the 17 keypoints confident (> 0.5): pose quality
  small_share     share of rows whose box is shorter than 80 px (small / distant people the detector managed to find)
  ghost_share     share of tracks whose mean detection confidence is below 0.35 (likely false detections: chairs, reflections)
  sec_per_clip    wall-clock time per clip (depends on the machine: compare on the same machine)

Proxies cannot prove precision of a detection; always look at the comparison contact sheet (`--sheet`) of a few clips before choosing.

Usage:  python -m src.perception.shootout [--config CFG] [--configs NAME ...] [--clips A B] [--max-s 12] [--sheet] [--force]
"""
import argparse
import json
import logging
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

from src.assm.track_poses import load_tracks, track_clip
from src.config import load_config, resolve_path
from src.video.shots import SHOT_BASE, detect_shots, load_shots, save_shots, shot_of_id

log = logging.getLogger(__name__)
USABLE_PAIR_S = 1.0
MIN_TRACK_CONF = 0.35


# ---------------------------------------------------------------- metrics (pure functions on a track dict)
def track_stats(t):
    """Per-track dict: frames (sorted), mean det conf, rows index."""
    out = {}
    for tid in sorted(set(t["track_id"].tolist())):
        k = np.where(t["track_id"] == tid)[0]
        out[tid] = {"rows": k, "frames": np.sort(t["frame_idx"][k]), "conf": float(t["conf"][k].mean())}
    return out


def usable_pairs(t, min_s=USABLE_PAIR_S, min_conf=MIN_TRACK_CONF):
    """[(id_i, id_j, seconds)] pairs of confident tracks of ONE shot that are visible together for at least min_s seconds."""
    fps = float(t["fps"])
    st = {k: v for k, v in track_stats(t).items() if v["conf"] >= min_conf}
    ids = sorted(st)
    out = []
    for a in range(len(ids)):
        for b in range(a + 1, len(ids)):
            i, j = ids[a], ids[b]
            if shot_of_id(i) != shot_of_id(j):
                continue
            n = len(np.intersect1d(st[i]["frames"], st[j]["frames"], assume_unique=True))
            if n >= min_s * fps:
                out.append((i, j, n / fps))
    return out


def clip_metrics(t):
    """Proxy metrics of one clip's tracks."""
    fps = float(t["fps"])
    n_rows = len(t["frame_idx"])
    pairs = usable_pairs(t)
    st = track_stats(t)
    # identity fragmentation: ids per person, per shot (max simultaneous people is the number of people at least)
    per_shot = defaultdict(lambda: {"ids": set(), "max": 0})
    counts = defaultdict(lambda: defaultdict(int))
    for f, tid in zip(t["frame_idx"].tolist(), t["track_id"].tolist()):
        counts[shot_of_id(tid)][f] += 1
        per_shot[shot_of_id(tid)]["ids"].add(tid)
    ratios = []
    for sh, d in per_shot.items():
        mx = max(counts[sh].values()) if counts[sh] else 0
        if mx:
            ratios.append(len(d["ids"]) / mx)
    long_rows = sum(len(v["frames"]) for v in st.values() if len(v["frames"]) >= 3 * fps)
    kpt_ok = int(((t["kpts"][:, :, 2] > 0.5).sum(axis=1) >= 10).sum()) if n_rows else 0
    h = (t["bbox"][:, 3] - t["bbox"][:, 1]) if n_rows else np.zeros(0)
    return {
        "has_pair": int(len(pairs) > 0), "pairs": len(pairs), "pair_seconds": float(sum(p[2] for p in pairs)),
        "ids_per_person": float(np.mean(ratios)) if ratios else float("nan"),
        "long_track": long_rows / n_rows if n_rows else float("nan"),
        "kpt_good": kpt_ok / n_rows if n_rows else float("nan"),
        "small_share": float((h < 80).mean()) if n_rows else float("nan"),
        "ghost_share": float(np.mean([v["conf"] < MIN_TRACK_CONF for v in st.values()])) if st else float("nan"),
        "n_rows": n_rows, "n_tracks": len(st),
    }


def aggregate(rows):
    """rows: list of per-clip metric dicts -> one summary dict (means over clips; clips without data skipped per metric)."""
    def m(key):
        v = [r[key] for r in rows if r.get(key) == r.get(key)]
        return float(np.mean(v)) if v else float("nan")
    return {"clips": len(rows), "pair_clips": m("has_pair"), "pairs_per_clip": m("pairs"), "pair_seconds": m("pair_seconds"),
            "ids_per_person": m("ids_per_person"), "long_track": m("long_track"), "kpt_good": m("kpt_good"),
            "small_share": m("small_share"), "ghost_share": m("ghost_share"), "sec_per_clip": m("seconds")}


# ---------------------------------------------------------------- running a configuration
def run_config(conf, base, clip, shot_starts, max_frames, out_dir, force=False):
    """Track one clip with one configuration (cached). Returns (tracks, seconds)."""
    p = Path(out_dir) / conf["name"] / clip["category"] / f"{clip['clip_id']}.npz"
    meta = p.with_suffix(".json")
    if p.exists() and not force:
        return load_tracks(p), json.loads(meta.read_text())["seconds"]
    cfg = {**base, **{k: v for k, v in conf.items() if k != "name"}}
    video = resolve_path(clip["path"])
    t0 = time.time()
    t = track_clip(video, cfg, clip["fps"], clip["width"], clip["height"], shot_starts, max_frames)
    sec = time.time() - t0
    p.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(p, **t)
    meta.write_text(json.dumps({"seconds": sec, "config": conf}), encoding="utf-8")
    return t, sec


def get_shot_starts(clip, shots_dir, shots_cfg):
    d = load_shots(shots_dir, clip["category"], clip["clip_id"])
    if d is None:
        d = detect_shots(resolve_path(clip["path"]), shots_cfg["method"], float(shots_cfg["threshold"]), float(shots_cfg["min_shot_s"]), shots_cfg["device"], clip["clip_id"])
        save_shots(shots_dir, clip["category"], clip["clip_id"], d)
    return [s["start_f"] for s in d["shots"]]


def compare_sheet(clip, results, out_path, frame, tile_w=420):
    """One frame of a clip with the boxes of every configuration, side by side (to judge detection quality by eye)."""
    video = resolve_path(clip["path"])
    cap = cv2.VideoCapture(str(video))
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
    ok, img0 = cap.read()
    cap.release()
    if not ok:
        return None
    tiles = []
    for name, t in results.items():
        img = img0.copy()
        for k in np.where(t["frame_idx"] == frame)[0]:
            x1, y1, x2, y2 = t["bbox"][k].astype(int)
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 220, 0), 2)
            for x, y, c in t["kpts"][k]:
                if c > 0.5:
                    cv2.circle(img, (int(x), int(y)), 3, (0, 0, 255), -1)
        n = int((t["frame_idx"] == frame).sum())
        img = cv2.resize(img, (tile_w, int(img.shape[0] * tile_w / img.shape[1])))
        cv2.rectangle(img, (0, 0), (tile_w, 18), (0, 0, 0), -1)
        cv2.putText(img, f"{name}: {n} people", (3, 13), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)
        tiles.append(img)
    hmax = max(x.shape[0] for x in tiles)
    tiles = [cv2.copyMakeBorder(x, 0, hmax - x.shape[0], 0, 0, cv2.BORDER_CONSTANT) for x in tiles]
    cols = min(3, len(tiles))
    while len(tiles) % cols:
        tiles.append(np.zeros_like(tiles[0]))
    grid = np.vstack([np.hstack(tiles[i:i + cols]) for i in range(0, len(tiles), cols)])
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), grid)
    return Path(out_path)


def best_frame(results):
    """The frame where the configurations found the most people in total (an informative frame for the comparison sheet)."""
    tot = defaultdict(int)
    for t in results.values():
        for f in t["frame_idx"].tolist():
            tot[f] += 1
    return max(tot, key=tot.get) if tot else 0


def print_table(summary):
    cols = ["pair_clips", "pairs_per_clip", "pair_seconds", "ids_per_person", "long_track", "kpt_good", "small_share", "ghost_share", "sec_per_clip"]
    print(f"{'config':<24}" + "".join(f"{c[:11]:>13}" for c in cols))
    for name, s in summary.items():
        print(f"{name:<24}" + "".join(f"{s[c]:>13.2f}" for c in cols))
    print("\nHigher is better: pair_clips, pairs_per_clip, pair_seconds, long_track, kpt_good. Lower is better: ids_per_person (1.0 = no fragmentation), ghost_share, sec_per_clip.")
    print("small_share is neutral (more small people found is good only if they are real: check the sheet).")


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    ap = argparse.ArgumentParser(description="Layer 1: perception shootout on a set of clips")
    ap.add_argument("--config", default=None)
    ap.add_argument("--configs", nargs="*", default=None, help="names from perception.configs (default: all)")
    ap.add_argument("--clips", nargs="*", default=None)
    ap.add_argument("--sample", type=int, default=0, help="use N clips per category (evenly spaced, held-out clips excluded): for the full Drive set")
    ap.add_argument("--max-s", type=float, default=None)
    ap.add_argument("--sheet", action="store_true", help="comparison contact sheets for the first few clips")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    cfg = load_config(args.config)
    pc = cfg["perception"]
    clean = json.loads(resolve_path(cfg["preprocess"]["clean_manifest_path"]).read_text("utf-8"))
    clips = [c for c in clean["clips"] if not args.clips or c["clip_id"] in args.clips]
    if args.sample:
        hold = {l.strip() for l in resolve_path(pc["holdout_file"]).read_text("utf-8").splitlines() if l.strip() and not l.startswith("#")}
        by_cat = defaultdict(list)
        for c in sorted(clips, key=lambda c: c["clip_id"]):
            if c["clip_id"] not in hold:
                by_cat[c["category"]].append(c)
        clips = [cs[int(i)] for cs in by_cat.values() for i in np.linspace(0, len(cs) - 1, min(args.sample, len(cs)))]
    confs = [c for c in pc["configs"] if not args.configs or c["name"] in args.configs]
    max_s = args.max_s or float(pc["max_s"])
    out_dir = resolve_path(pc["shootout_dir"])
    shots_dir = resolve_path(cfg["context"]["context_dir"])
    results = {c["name"]: [] for c in confs}
    keep = defaultdict(dict)
    for n, clip in enumerate(clips, 1):
        starts = get_shot_starts(clip, shots_dir, cfg["shots"])
        max_frames = int(max_s * clip["fps"])
        for conf in confs:
            try:
                t, sec = run_config(conf, cfg["assm"], clip, starts, max_frames, out_dir, args.force)
            except Exception as e:  # a configuration that cannot run (missing weights, memory) must not stop the others
                log.warning("%s on %s failed: %s", conf["name"], clip["clip_id"], e)
                continue
            m = clip_metrics(t)
            m["seconds"] = sec
            results[conf["name"]].append(m)
            keep[clip["clip_id"]][conf["name"]] = t
        print(f"  shootout: {n}/{len(clips)} clips", flush=True)
    summary = {name: aggregate(rows) for name, rows in results.items() if rows}
    print()
    print_table(summary)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "shootout_results.json").write_text(json.dumps({"max_s": max_s, "summary": summary, "per_clip": results}, indent=1, default=float), encoding="utf-8")
    if args.sheet:
        for cid, res in list(keep.items())[:8]:
            clip = next(c for c in clips if c["clip_id"] == cid)
            compare_sheet(clip, res, out_dir / "sheets" / f"{cid}.jpg", best_frame(res))
        print(f"\nComparison sheets: {out_dir / 'sheets'}")
    print(f"Results: {out_dir / 'shootout_results.json'}")


if __name__ == "__main__":
    main()
