"""Jury showcase for the full-video classification run: slide pictures (PNG, 16:9) and a side-by-side video (MP4) of a violent vs a Normal video.

    python -m src.classfull.showcase --out-root <classfull folder> [--pairs 3] [--violent Chain_Snatching_v38] [--normal Normal_v52] [--no-video]
Writes into <out-root>/showcase/. Reads the finished stages (no GPU needed); run it after `--stage analyze` (and `--stage train` to add the model scores).
"""
import argparse
from pathlib import Path

from src.classcommon.config import load_class_config
from src.classcommon.showcase import build_showcase


def main():
    ap = argparse.ArgumentParser(description="Slide pictures and side-by-side video: violent vs Normal")
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--pairs", type=int, default=3, help="how many violent/Normal pairs to render")
    ap.add_argument("--violent", default=None, help="force this violent clip id as the first example")
    ap.add_argument("--normal", default=None, help="force this Normal clip id as the first example")
    ap.add_argument("--max-seconds", type=float, default=40.0)
    ap.add_argument("--no-video", action="store_true")
    a = ap.parse_args()
    cfg = load_class_config(a.out_root)
    for p in build_showcase(cfg, Path(a.out_root) / "showcase", a.pairs, a.violent, a.normal, a.max_seconds, not a.no_video):
        print("wrote", p)


if __name__ == "__main__":
    main()
