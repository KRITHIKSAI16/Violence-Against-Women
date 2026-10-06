"""Classification of the clips cut at the violence start (task "trim"): will violence follow? Violent clips end at the human start time T, Normal clips are whole;
every clip is judged on its last `span_s` seconds (text, video windows and geometry), so clip length cannot be read off the input.

It starts from a FINISHED colab2 run (the folder passed as --curated-root: trimmed clips, tracks, stories, and the video windows its ml stage encoded) and writes only
under --out-root.

    python -m src.classtrim.run --stage all --out-root <out> --curated-root <colab2 out folder> [--no-captions]
    stages: captions (Layer B, GPU) -> features (video windows not yet encoded, GPU) -> train (documents, cross-validation, report)
"""
import argparse
import logging
import time

from src.classcommon.config import build_config
from src.classcommon.pipeline import load_records
from src.classcommon.stages import stage_features, stage_train
from src.classcommon.vlm_caption import run_captions

STAGES = ["captions", "features", "train"]
TASK = "trim"


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    ap = argparse.ArgumentParser(description="Classification of pre-violence clips (trim task)")
    ap.add_argument("--stage", default="all", choices=["all"] + STAGES)
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--curated-root", required=True, help="the finished colab2 output folder")
    ap.add_argument("--no-captions", action="store_true", help="switch Layer B (frame captions by a video-language model) off")
    args = ap.parse_args()
    cfg, path = build_config(args.out_root, TASK, args.curated_root, use_captions=False if args.no_captions else None)
    print(f"config: {path} | captions (Layer B): {'on' if cfg['classify']['use_captions'] else 'off'}")
    for s in (STAGES if args.stage == "all" else [args.stage]):
        t0 = time.time()
        print(f"\n=== stage: {s} ===", flush=True)
        if s == "captions":
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
