# Video Context Understanding: Predicting and Preventing Violence (VAW)

SSN College of Engineering capstone. Phase 1 = modules M1-M9 from Review Report I: data pipeline plus the screening
front end that scores person-to-person interaction frame by frame (the pre-violence "buildup" curve).
Dataset: ExtrAnom (Assassination, Chain_Snatching, Harassment, Kidnapping, Normal, Stalking).

Read `PROJECT_BRIEF.md` (context + module spec), `STATE.md` (current status), `docs/PHASE1_PLAN.md` (plan).

## Layout
```
configs/extran.yaml   laptop config (sample data)      configs/colab.yaml   Colab config (Drive paths)
src/config.py         config loader                    src/data/manifest.py M1   src/data/preprocess.py M2
tests/                pytest tests (no GPU/dataset)    colab/               notebooks + Colab instructions
data/sample/          12 sample clips, LOCAL ONLY (git-ignored; get them from the team Drive)
```

## Run on your laptop
Needs Python 3.10+ and ffmpeg on PATH.
```
git clone https://github.com/KRITHIKSAI16/Violence-Against-Women.git && cd Violence-Against-Women
pip install -r requirements.txt
python -m pytest -q                      # no dataset needed
# put 2 clips per category under data/sample/<Category>/ , then:
python -m src.data.manifest              # M1 -> data/manifest.json
python -m src.data.preprocess            # M2 -> data/processed/, data/manifest_clean.json
```
## Run on Colab
See `colab/README.md`.

## Workflow
Code and docs travel through GitHub; video data travels only through Drive. Pull before you start, keep `pytest` green,
small commits, mention the module (M1, M8, ...) in the message.
