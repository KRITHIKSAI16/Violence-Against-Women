# Running the pipeline on Google Colab

Colab is for the full ExtrAnom set (793 clips in Drive) and the GPU steps. Notebooks here contain only cells: mount Drive, get code, install, run a script, inspect.
All logic is in `src/`, so the same code runs on a laptop and on Colab.

## One-time setup (each teammate)
1. **Drive shortcut.** In Google Drive, right-click the shared `ExtrAnom` folder -> *Add shortcut to Drive* -> *My Drive*.
   (Shared-with-me paths are unreliable in Colab.) Expected path: `/content/drive/MyDrive/ExtrAnom/<Category>/*.mp4`.
2. **GitHub token** (the repo is private). GitHub -> Settings -> Developer settings -> Personal access tokens -> fine-grained, read access to this repo (*Contents: read*).
   In Colab: left sidebar *Secrets* (key icon) -> add `GITHUB_TOKEN`, turn on *Notebook access*. Never paste the token into a cell.
3. Open a notebook: Colab -> File -> Open notebook -> GitHub tab, or upload the `.ipynb` from `colab/`.
4. Runtime: **GPU (T4) for notebooks 02 and 07**; everything else runs on CPU.

## Notebooks (run in order 01 -> 08; every step is resumable, so if the session drops, reconnect and rerun the same notebook)
| # | Notebook | Runtime | Roughly | What it does -> output in Drive `VAW_results/` |
|---|---|---|---|---|
| 01 | `01_setup_and_m1_m2` | CPU | long | tests, M1 manifest, M2 preprocess -> `manifest.json`, `processed/`, `manifest_clean.json` |
| 02 | `02_m8a_track_poses` | **GPU** | longest | YOLOv8n-pose + ByteTrack + stitching -> `tracks/<Category>/*.npz`, `previews/` |
| 03 | `03_m8b_interaction_scores` | CPU | minutes | Algorithm 1 scores (baseline) -> `scores/` |
| 04 | `04_m9_buildup_report` | CPU | minutes | M9 rule gate + first report (baseline) -> `gate/`, `report/` |
| 05 | `05_context_features` | CPU | ~40 min | Stage A: camera, meters, pose features, following, events -> `context/*_scene.json`, `*_pairs.npz` |
| 06 | `06_learn_and_evaluate` | CPU | ~15-20 min | the evaluation with shortcut probes -> `learn/results.json` |
| 07 | `07_scene_layout_and_depth` | **GPU** | ~20-40 min | Stage B: scene layout, doors, light, depth of the key pair -> `context/*_layout.*`, `*_depth.json` |
| 08 | `08_story_report_and_benchmark` | CPU | ~15 min + annotation | Stage C stories, **report page**, annotation sheets, benchmark evaluation -> `context/*_story.json`, `report/` |

Notebook 08 produces the final deliverable. What each output means and how to check it: `docs/VERIFY_CONTEXT.md`. Results so far and decisions: `docs/STATUS_REPORT.md`.

## Paths
`configs/colab.yaml` holds the Drive paths. If your folder names differ, edit that file (and push) rather than the code. Outputs go to Drive, so they survive the session and teammates can see them.
Do not use `--force` unless you changed settings and want everything recomputed.

## Changing settings
Notebook cells accept `--set key=value` for one run (e.g. `--set showcase_top=10`). To make a value permanent, edit the matching block in both `configs/extran.yaml` and `configs/colab.yaml` (blocks: `gate`, `context`, `story`, `learn`,
`report`), commit and push, then `git pull` in Colab (first cell) and rerun. Exact video cut times for clips you present: add `clip_id,cut_s` lines to `configs/clip_overrides.csv`.

## If something fails
- `data_root not found`: shortcut missing or named differently; check step 1 and `configs/colab.yaml`.
- `git clone` auth error: token missing, expired, or notebook access off.
- A cell shows nothing for a long time: many scripts print only at the end or every 50-100 clips; check the Drive output folder or wait. Stop and rerun is safe (resumable).
- Session disconnected: reconnect, rerun the first cell, then the run cell; finished clips are skipped.
- Report videos not playing in Drive's preview: download the `report` folder and open `index.html` locally.
- Model download errors in notebook 07: Colab needs internet access to Hugging Face on first use; rerun the cell.
