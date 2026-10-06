"""Curated dataset, step 3: the clips the pipeline sees are cut at the human start time T.

Each violent clip is re-encoded (same normalisation as M2: 30 fps, longest side 640) and TRIMMED to [0, T). Every later stage (shots, tracking,
camera, layout, story, video) then runs on that trimmed clip, so no cue, distance or box from the violent act itself can leak into the analysis.
Normal clips have no T: they are normalised whole and capped at `max_normal_s`. Writes curated_manifest_clean.json in the M2 schema.
"""
import json
import logging
import subprocess
from pathlib import Path

from src.data.manifest import probe_video
from src.data.preprocess import build_filter

log = logging.getLogger(__name__)


def prepare_clip(src, dst, start_s, fps, max_side, max_len_s=None):
    """Re-encode src to dst keeping only [0, start_s) (or the first max_len_s seconds when start_s is None). Returns probe metadata of dst."""
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    length = start_s if start_s is not None else max_len_s
    cmd = ["ffmpeg", "-y", "-v", "error", "-i", str(src)] + (["-t", f"{length:.3f}"] if length else []) + \
          ["-vf", build_filter(fps, max_side, False), "-an", "-c:v", "libx264", "-preset", "fast", "-crf", "18", "-pix_fmt", "yuv420p", str(dst)]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        dst.unlink(missing_ok=True)
        raise RuntimeError(f"ffmpeg failed: {r.stderr.strip()[:200]}")
    return probe_video(dst)


def build_curated_manifest(clips, cfg, max_normal_s=45.0, force=False):
    """clips: [{clip_id, category, path, start_s, status, name_in_sheet}] (violent clips with start_s, Normal with start_s None).
    Writes the clean manifest (+ cut_times.csv for the existing cut override mechanism) and returns the manifest dict."""
    pre = cfg["preprocess"]
    out_dir = Path(pre["processed_dir"])
    entries, skipped = [], []
    for c in clips:
        dst = out_dir / c["category"] / f"{c['clip_id']}.mp4"
        try:
            if dst.exists() and not force:
                meta = probe_video(dst)
            else:
                meta = prepare_clip(c["path"], dst, c["start_s"], pre["target_fps"], pre["max_side"], max_normal_s)
            raw = probe_video(c["path"])
        except Exception as e:
            log.warning("Skipping %s: %s", c["clip_id"], e)
            skipped.append({"clip_id": c["clip_id"], "reason": str(e)})
            continue
        entries.append({"clip_id": c["clip_id"], "category": c["category"], "label": "normal" if c["start_s"] is None else "pre_violence",
                        "source": "curated", "path": dst.as_posix(), "raw_path": c["path"], "start_s": c["start_s"], "status": c.get("status", "ok"),
                        "raw_duration_s": raw["duration_s"], "raw_width": raw["width"], "raw_height": raw["height"], "raw_fps": raw["fps"], **meta})
        print(f"  prepared {len(entries)}/{len(clips)}: {c['clip_id']}", flush=True)
    manifest = {"source": "curated", "stage": "M2-trimmed", "num_clips": len(entries), "clips": entries, "skipped": skipped}
    p = Path(pre["clean_manifest_path"])
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    # cut_times.csv: the trimmed clips already end at T, so the existing renderer must show them completely (cut = their own length)
    Path(cfg["report"]["overrides_csv"]).write_text("clip_id,cut_s\n" + "".join(f"{e['clip_id']},{e['duration_s']}\n" for e in entries), encoding="utf-8")
    return manifest
