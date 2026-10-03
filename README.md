# Video Context Understanding: Predicting and Preventing Violence (VAW)

SSN College of Engineering capstone (team of four, supervisor Dr. Anusha Jayasimhan). Phase 1 = modules M1–M9 of Review Report I:
a data pipeline plus a screening front end that shows the **pre-violence buildup** between people (approaching, following, hovering near,
blocking someone's way out) before any violent act. Dataset: ExtrAnom (Assassination, Chain_Snatching, Harassment, Kidnapping, Normal, Stalking; 793 clips).

**Start here:** `PROJECT_BRIEF.md` (context) → `STATE.md` (status, findings, next steps) → `docs/ARCHITECTURE.md` (how it works) → `docs/VERIFY_M9.md` (how to check results).
If you use Claude Code, it reads `CLAUDE.md` first.

## Pipeline
```
clips → M1 manifest → M2 clean clips → M8a tracks + pose (GPU) → M8b interaction score
                                                \→ M9 behavior states + proposals → buildup videos + category report
```

## Layout
```
configs/extran.yaml        laptop config (sample data)        configs/colab.yaml   Colab config (Drive paths)
src/data/                  M1 manifest, M2 preprocess         src/assm/            M8a tracks, M8b scores, M9 gate, calibrate, checks
src/report/                buildup video + category report    tests/               pytest (synthetic data, no GPU)
colab/                     notebooks 01-04 + instructions     docs/                architecture, plan, verification guides
data/sample/               12 clips, LOCAL ONLY (git-ignored; get them from the team Drive)
```

## Run on your laptop
Python 3.10+ and ffmpeg on PATH.
```
git clone https://github.com/KRITHIKSAI16/Violence-Against-Women.git && cd Violence-Against-Women
pip install -r requirements.txt
python -m pytest -q                                   # no dataset needed
# put 2 clips per category under data/sample/<Category>/ , then:
python -m src.data.manifest && python -m src.data.preprocess
python -m src.assm.track_poses --preview-dir data/previews      # downloads yolov8n-pose.pt (~6 MB); CPU is fine for 12 clips
python -m src.assm.interaction && python -m src.assm.gate
python -m src.report.buildup_report                   # -> data/report/index.html (open in a browser)
python -m src.report.buildup_video Kidnapping_v28     # one clip: video + storyboard
```
## Run on Colab (full set)
See `colab/README.md`: one-time setup (Drive shortcut, GitHub token secret), then notebooks in order
`01` (M1/M2) → `02` (M8a, GPU) → `03` (M8b) → `04` (M9, calibration, report).

## Workflow
Code and docs travel through GitHub; video data travels only through Drive. Pull before you start, keep `pytest` green, small commits that name the
module (M1, M8, M9 ...). Results are reported honestly: compare with Normal, state limits, confirm with the human review sheet.
