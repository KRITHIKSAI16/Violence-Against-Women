"""Classification of whole, untrimmed videos (task "full"): is violence in this video? Label = the category folder (Normal -> 0, any other category -> 1), so no start
time is needed. Every video is prepared WHOLE (30 fps, 640 px, no trimming), then tracked and analysed with the same stage CLIs as colab2, described in text
(+ optional frame captions) and encoded by the frozen video models; the classifiers see the video features, the text and the geometry.

    python -m src.classfull.run --stage all --out-root <out> --data-root <folder with the category folders> [--curated-root <colab2 folder>] [--no-captions]
    stages: prepare -> track (GPU) -> context (GPU for scene/depth) -> analyze -> captions (Layer B, GPU) -> features (GPU) -> train (cross-validation, report)
--curated-root is optional: when it holds perception_choice.json the same person detector is used. --max-clips N runs a quick trial on about N clips.
"""
import argparse
import json
import logging
import time
from pathlib import Path

from src.classcommon.config import build_config
from src.classcommon.labels import collect_videos
from src.classcommon.pipeline import load_records
from src.classcommon.stages import sh, stage_features, stage_train
from src.classcommon.vlm_caption import run_captions
from src.curated.locate import find_violence_root
from src.curated.pipeline import analyze_clip
from src.curated.prepare import build_curated_manifest

STAGES = ["prepare", "track", "context", "analyze", "captions", "features", "train"]
TASK = "full"


def stage_prepare(cfg, args):
    cl = cfg["classify"]
    root = Path(args.data_root) if args.data_root else find_violence_root(args.drive_root)
    if root is None or not Path(root).is_dir():
        raise FileNotFoundError("video folder not found: pass --data-root (the folder that holds the category folders)")
    clips, skipped, notes = collect_videos(root, args.n_normal or cl["n_normal"], args.max_clips)
    n_violent = sum(c["category"] != "Normal" for c in clips)
    print(f"video folder: {root}\nclips: {n_violent} violent + {len(clips) - n_violent} Normal | names not understood (skipped): {len(skipped)}")
    for s in skipped[:20]:
        print(f"  skipped {s!r}: name does not look like Category_vNN")
    for n in notes[:20]:
        print(f"  note: {n}")
    if not clips:
        raise RuntimeError("no video found")
    build_curated_manifest(clips, cfg, max_normal_s=cl["max_len_s"])
    mp = Path(cfg["preprocess"]["clean_manifest_path"])
    man = json.loads(mp.read_text(encoding="utf-8"))
    for e in man["clips"]:                                                      # build_curated_manifest labels clips without a start time "normal": correct that
        e["label"] = "normal" if e["category"] == "Normal" else "violent"
        e["source"] = "class_full"
    mp.write_text(json.dumps(man, indent=1), encoding="utf-8")


def stage_analyze(cfg):
    clips = json.loads(Path(cfg["preprocess"]["clean_manifest_path"]).read_text(encoding="utf-8"))["clips"]
    for n, c in enumerate(clips, 1):
        r = analyze_clip(c, cfg)
        print(f"  analyzed {n}/{len(clips)}: {c['clip_id']} -> {r['interaction']['leadup']['type'] if r.get('interaction') else 'no pair'}", flush=True)


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    ap = argparse.ArgumentParser(description="Classification of whole videos (full task)")
    ap.add_argument("--stage", default="all", choices=["all"] + STAGES)
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--data-root", default=None)
    ap.add_argument("--drive-root", default="/content/drive/MyDrive")
    ap.add_argument("--curated-root", default=None)
    ap.add_argument("--n-normal", type=int, default=None)
    ap.add_argument("--max-clips", type=int, default=None)
    ap.add_argument("--no-captions", action="store_true", help="switch Layer B (frame captions by a video-language model) off")
    args = ap.parse_args()
    cfg, path = build_config(args.out_root, TASK, args.curated_root, use_captions=False if args.no_captions else None)
    print(f"config: {path} | captions (Layer B): {'on' if cfg['classify']['use_captions'] else 'off'} | detector: {cfg['assm']['model']} at {cfg['assm']['imgsz']} px")
    for s in (STAGES if args.stage == "all" else [args.stage]):
        t0 = time.time()
        print(f"\n=== stage: {s} ===", flush=True)
        if s == "prepare":
            stage_prepare(cfg, args)
        elif s == "track":
            sh("src.video.shots", "--config", str(path))
            sh("src.assm.track_poses", "--config", str(path))
        elif s == "context":
            for m in ("features", "scene", "story"):
                sh(f"src.context.{m}", "--config", str(path))
        elif s == "analyze":
            stage_analyze(cfg)
        elif s == "captions":
            if cfg["classify"]["use_captions"]:
                run_captions(load_records(cfg, TASK), cfg, TASK)
            else:
                print("captions are switched off")
        elif s == "features":
            stage_features(cfg, load_records(cfg, TASK), TASK)
        else:
            print(f"report: {stage_train(cfg, TASK)}")
        print(f"  [{time.time() - t0:.0f} s]", flush=True)


if __name__ == "__main__":
    main()
