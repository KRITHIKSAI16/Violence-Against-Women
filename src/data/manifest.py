"""M1 — Data Acquisition: build a unified manifest of the ExtrAnom video corpus.

Scans <data_root>/<Category>/*.mp4, reads real metadata with OpenCV, and writes
a manifest JSON that later modules (M2 preprocessing, M3 segments, M8 ASSM)
use as their single list of clips. Unreadable files are logged and skipped.

Usage:  python -m src.data.manifest [--config CFG] [--data-root DIR]
"""
import argparse
import json
import logging
import os
import subprocess
from collections import Counter
from pathlib import Path

import cv2

from src.config import REPO_ROOT, load_config, resolve_path

log = logging.getLogger(__name__)


def _probe_cv2(path):
    """Metadata via OpenCV. Raises ValueError if the file cannot be opened/decoded."""
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
    return {"duration_s": round(frames / fps, 3), "fps": round(fps, 3),
            "frame_count": frames, "width": width, "height": height}


def _probe_ffprobe(path):
    """Metadata from the container/stream headers via ffprobe (no decoding needed).

    Used when OpenCV cannot decode a file (e.g. AV1 on Colab). The file is still a valid
    video: M2 re-encodes it to H.264 with ffmpeg, after which OpenCV can read it.
    """
    r = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_packets",
                        "-show_entries", "stream=codec_name,width,height,r_frame_rate,nb_read_packets,duration",
                        "-of", "json", str(path)], capture_output=True, text=True)
    if r.returncode != 0:
        raise ValueError(f"ffprobe failed: {r.stderr.strip()[:120]}")
    streams = json.loads(r.stdout).get("streams") or []
    if not streams:
        raise ValueError("no video stream")
    st = streams[0]
    num, _, den = st["r_frame_rate"].partition("/")
    fps = float(num) / float(den or 1)
    frames = int(st.get("nb_read_packets") or 0)
    width, height = int(st["width"]), int(st["height"])
    if fps <= 0 or frames <= 0 or width <= 0 or height <= 0:
        raise ValueError(f"invalid metadata (fps={fps}, frames={frames}, {width}x{height})")
    return {"duration_s": round(frames / fps, 3), "fps": round(fps, 3), "frame_count": frames,
            "width": width, "height": height, "codec": st.get("codec_name")}


def probe_video(path):
    """Return video metadata. OpenCV first; ffprobe fallback for codecs OpenCV cannot decode.

    Raises ValueError if neither can read the file (truly corrupt).
    """
    try:
        return _probe_cv2(path)
    except ValueError as cv_err:
        try:
            meta = _probe_ffprobe(path)
        except (ValueError, FileNotFoundError, OSError):
            raise cv_err  # report the original reason
        log.info("%s: OpenCV could not decode (%s); metadata read with ffprobe (codec=%s)",
                 path, cv_err, meta.get("codec"))
        return meta


def _rel_path(path):
    """Path relative to the repo root if inside it, else absolute (e.g. Drive). POSIX style."""
    path = Path(os.path.abspath(path))  # abspath, not resolve(): keep Drive shortcut paths as given
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
