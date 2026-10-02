"""M2 — Preprocessing: turn the raw corpus (M1 manifest) into a cleaned corpus.

For every clip in the M1 manifest: re-encode with ffmpeg at a fixed frame rate
and a fixed longest side (aspect ratio kept), optionally denoise, then re-probe
the output to confirm it is readable. Clips that fail are logged and skipped.
Writes data/processed/<Category>/<clip_id>.mp4 and data/manifest_clean.json.

Usage:  python -m src.data.preprocess [--config CFG] [--manifest M] [--force]
"""
import argparse
import json
import logging
import subprocess
from collections import Counter
from pathlib import Path

from src.config import REPO_ROOT, load_config, resolve_path
from src.data.manifest import _rel_path, probe_video

log = logging.getLogger(__name__)


def build_filter(fps, max_side, denoise):
    """ffmpeg -vf chain: optional denoise -> fps resample -> scale longest side to max_side."""
    scale = (f"scale='if(gt(iw,ih),{max_side},-2)':'if(gt(iw,ih),-2,{max_side})'"
             ":flags=bicubic")
    parts = (["hqdn3d=2:1.5:3:2.5"] if denoise else []) + [f"fps={fps}", scale]
    return ",".join(parts)


def preprocess_clip(src, dst, fps, max_side, denoise):
    """Re-encode one clip. Raises RuntimeError if ffmpeg fails or output is unreadable."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    cmd = ["ffmpeg", "-y", "-v", "error", "-i", str(src),
           "-vf", build_filter(fps, max_side, denoise),
           "-an", "-c:v", "libx264", "-preset", "fast", "-crf", "18",
           "-pix_fmt", "yuv420p", str(dst)]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        dst.unlink(missing_ok=True)
        raise RuntimeError(f"ffmpeg failed: {r.stderr.strip()[:200]}")
    try:
        return probe_video(dst)  # confirm the cleaned file really decodes
    except ValueError as e:
        dst.unlink(missing_ok=True)
        raise RuntimeError(f"output unreadable: {e}")


def build_clean_manifest(manifest, out_dir, fps, max_side, denoise, force=False):
    """Process every clip in the M1 manifest; return the cleaned manifest dict."""
    clips, skipped = [], list(manifest.get("skipped", []))  # keep M1's corrupt files on record
    for c in manifest["clips"]:
        src = resolve_path(c["path"])
        dst = out_dir / c["category"] / f"{c['clip_id']}.mp4"
        try:
            if dst.exists() and not force:
                meta = probe_video(dst)  # reuse earlier output (resumable on Colab)
            else:
                meta = preprocess_clip(src, dst, fps, max_side, denoise)
        except Exception as e:
            log.warning("Skipping %s: %s", c["clip_id"], e)
            skipped.append({"path": c["path"], "reason": str(e)})
            continue
        clips.append({**c, "raw_path": c["path"], "path": _rel_path(dst), **meta})
        log.info("%-22s %4dx%-4d %5.2ffps -> %4dx%-4d %dfps",
                 c["clip_id"], c["width"], c["height"], c["fps"],
                 meta["width"], meta["height"], round(meta["fps"]))
    return {
        "source": manifest["source"],
        "stage": "M2",
        "settings": {"fps": fps, "max_side": max_side, "denoise": denoise},
        "num_clips": len(clips),
        "counts_per_category": dict(Counter(c["category"] for c in clips)),
        "clips": clips,
        "skipped": skipped,
    }


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    ap = argparse.ArgumentParser(description="M2: preprocess the M1 corpus")
    ap.add_argument("--config", default=None)
    ap.add_argument("--manifest", default=None, help="override M1 manifest path")
    ap.add_argument("--out-dir", default=None, help="override processed_dir")
    ap.add_argument("--force", action="store_true", help="re-encode even if output exists")
    args = ap.parse_args()

    cfg = load_config(args.config)
    pp = cfg["preprocess"]
    manifest = json.loads(resolve_path(args.manifest or cfg["manifest_path"]).read_text("utf-8"))
    out_dir = resolve_path(args.out_dir or pp["processed_dir"])
    clean = build_clean_manifest(manifest, out_dir, pp["target_fps"], pp["max_side"],
                                 pp["denoise"], args.force)
    out = resolve_path(pp["clean_manifest_path"])
    out.write_text(json.dumps(clean, indent=2), encoding="utf-8")

    print(f"\nWrote {out}  ({clean['num_clips']} clips, {len(clean['skipped'])} skipped)")
    for cat, n in sorted(clean["counts_per_category"].items()):
        print(f"  {cat:<16}{n}")
    for s in clean["skipped"]:
        print(f"  SKIPPED {s['path']}: {s['reason']}")


if __name__ == "__main__":
    main()
