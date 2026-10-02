"""M1 — Data Acquisition: build a unified manifest of the ExtrAnom video corpus.

Scans <data_root>/<Category>/*.mp4, reads real metadata with OpenCV, and writes
a manifest JSON that later modules (M2 preprocessing, M3 segments, M8 ASSM)
use as their single list of clips. Unreadable files are logged and skipped.

Usage:  python -m src.data.manifest [--config CFG] [--data-root DIR]
"""
import argparse
import json
import logging
from collections import Counter
from pathlib import Path

import cv2

from src.config import REPO_ROOT, load_config, resolve_path

log = logging.getLogger(__name__)


def probe_video(path):
    """Return video metadata read via cv2. Raises ValueError if unreadable."""
    cap = cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened():
            raise ValueError("cannot open file")
        fps = cap.get(cv2.CAP_PROP_FPS)
        frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if fps <= 0 or frames <= 0 or width <= 0 or height <= 0:
            raise ValueError(f"invalid metadata (fps={fps}, frames={frames}, {width}x{height})")
        ok, _ = cap.read()
        if not ok:
            raise ValueError("first frame could not be decoded")
    finally:
        cap.release()
    return {
        "duration_s": round(frames / fps, 3),
        "fps": round(fps, 3),
        "frame_count": frames,
        "width": width,
        "height": height,
    }


def _rel_path(path):
    """Path relative to the repo root if inside it, else absolute (e.g. Drive). POSIX style."""
    path = Path(path).resolve()
    try:
        return path.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def build_manifest(data_root, source, normal_category, extensions):
    """Scan data_root and return the manifest dict (clips + skipped files)."""
    data_root = Path(data_root)
    if not data_root.is_dir():
        raise FileNotFoundError(f"data_root not found: {data_root}")
    exts = {e.lower() for e in extensions}
    clips, skipped = [], []

    for cat_dir in sorted(p for p in data_root.iterdir() if p.is_dir()):
        for f in sorted(p for p in cat_dir.iterdir() if p.is_file() and p.suffix.lower() in exts):
            try:
                meta = probe_video(f)
            except Exception as e:  # corrupt/unreadable: warn and move on
                log.warning("Skipping %s: %s", f, e)
                skipped.append({"path": _rel_path(f), "reason": str(e)})
                continue
            clips.append({
                "clip_id": f.stem,
                "category": cat_dir.name,
                "label": "normal" if cat_dir.name == normal_category else "pre_violence",
                "source": source,
                "path": _rel_path(f),
                **meta,
            })

    return {
        "source": source,
        "data_root": _rel_path(data_root),
        "num_clips": len(clips),
        "counts_per_category": dict(Counter(c["category"] for c in clips)),
        "clips": clips,
        "skipped": skipped,
    }


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    ap = argparse.ArgumentParser(description="M1: build data/manifest.json")
    ap.add_argument("--config", default=None, help="YAML config (default configs/extran.yaml)")
    ap.add_argument("--data-root", default=None, help="override data_root from config")
    ap.add_argument("--out", default=None, help="override manifest_path from config")
    args = ap.parse_args()

    cfg = load_config(args.config)
    manifest = build_manifest(
        resolve_path(args.data_root or cfg["data_root"]),
        cfg["source"], cfg["normal_category"], cfg["video_extensions"],
    )
    out = resolve_path(args.out or cfg["manifest_path"])
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"\nWrote {out}  ({manifest['num_clips']} clips, {len(manifest['skipped'])} skipped)")
    for cat, n in sorted(manifest["counts_per_category"].items()):
        print(f"  {cat:<16}{n}")
    for s in manifest["skipped"]:
        print(f"  SKIPPED {s['path']}: {s['reason']}")


if __name__ == "__main__":
    main()
