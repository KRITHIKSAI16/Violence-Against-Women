# Video Context Understanding: Predicting and Preventing Violence (VAW)

SSN College of Engineering capstone (team of four, supervisor Dr. Anusha Jayasimhan). Phase 1 = modules M1–M9 of Review Report I: a data pipeline plus a screening front end that **describes, for each video,
what happens between the people before any violent act** (who approaches, follows, looks back at, lingers near, blocks or reaches for whom), with times, approximate meters and scene facts, and an honest evaluation
of what such geometry can and cannot tell. Dataset: ExtrAnom (Assassination, Chain_Snatching, Harassment, Kidnapping, Normal, Stalking; 793 clips).

**Start here (in this order):** `docs/STATUS_REPORT.md` (what we did, found and decided, in plain language - share this) → `PROJECT_BRIEF.md` (scope and spec) → `STATE.md` (live status and next steps) →
`docs/ARCHITECTURE.md` (how it is built) → `docs/VERIFY_CONTEXT.md` (how to check every output). If you use Claude Code, it reads `CLAUDE.md` first.

## Pipeline
```
clips → M1 manifest → M2 clean clips → M8a tracks + pose (GPU) ─┬─ baseline: M8b score, M9 rule gate
                                                                 └─ deep context: camera, meters, pose features, following (A) → scene layout + depth (B, GPU)
                                                                      → scene graph + narrative (C) → evaluation with shortcut probes (D)
                                                                      → story video + report page + annotation benchmark
```

## Layout
```
configs/extran.yaml        laptop config (sample data)       configs/colab.yaml   Colab config (Drive paths)
src/data/                  M1 manifest, M2 preprocess        src/assm/            M8a tracks, M8b score, M9 rule gate (baseline), calibrate, checks
src/context/               deep context layer (stages A-D)   src/report/          story video, report page, benchmark (+ older baseline report)
tests/                     pytest (synthetic data, no GPU)   colab/               notebooks 01-08 + instructions
docs/                      status report, architecture, plan, verification guides
data/sample/               12 clips, LOCAL ONLY (git-ignored; get them from the team Drive)
```

## Run on your laptop
Python 3.10+ and ffmpeg on PATH.
```
git clone https://github.com/KRITHIKSAI16/Violence-Against-Women.git && cd Violence-Against-Women
pip install -r requirements.txt
python -m pytest -q                                              # 176 tests, no dataset needed
# put 2 clips per category under data/sample/<Category>/ , then:
python -m src.data.manifest && python -m src.data.preprocess
python -m src.assm.track_poses                                   # downloads yolov8n-pose.pt (~6 MB); CPU is fine for 12 clips
python -m src.context.features                                   # Stage A
python -m src.context.scene                                      # Stage B (downloads ~120 MB of model weights once)
python -m src.context.story --example Kidnapping_v28             # Stage C: prints the clip's story
python -m src.report.story_report --set showcase_top=2 showcase_random=1     # -> data/report/index.html
python -m src.report.story_video Stalking_v3                     # one clip: story video + storyboard
```
## Run on Colab (full set)
See `colab/README.md`: one-time setup (Drive shortcut, GitHub token secret), then notebooks `01` → `08` in order. The final deliverable is produced by notebook 08.

## Workflow
Code and docs travel through GitHub; video data and results travel only through Drive (generated files are git-ignored). Pull before you start, keep `pytest` green, small commits that name the module.
Results are reported honestly: compare every category with Normal, state limits, and measure detector quality with the annotated benchmark, not with the category rates.
