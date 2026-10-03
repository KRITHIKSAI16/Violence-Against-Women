# Running the pipeline on Google Colab

Colab is for the full ExtrAnom set (793 clips in Drive) and, for M8a, the GPU.
Notebooks here contain only cells: mount Drive, get code, install, run a script, inspect.
All logic is in `src/`, so the same code runs on a laptop and on Colab.

## One-time setup (each teammate)
1. **Drive shortcut.** In Google Drive, right-click the shared `ExtrAnom` folder -> *Add shortcut to Drive* -> *My Drive*.
   (Shared-with-me paths are unreliable in Colab.) Expected path: `/content/drive/MyDrive/ExtrAnom/<Category>/*.mp4`.
2. **GitHub token** (the repo is private). GitHub -> Settings -> Developer settings -> Personal access tokens ->
   fine-grained, read access to this repo (*Contents: read*). In Colab: left sidebar *Secrets* (key icon) -> add
   `GITHUB_TOKEN`, turn on *Notebook access*. Never paste the token into a cell.
3. Open a notebook: Colab -> File -> Open notebook -> GitHub tab, or upload the `.ipynb` from `colab/`.
4. Runtime: only notebook 02 (M8a) needs a GPU (Runtime -> Change runtime type -> T4). The others run on CPU.

## Notebooks (run in order 01 -> 02 -> 03 -> 04)
| Notebook | What it runs | Output (in Drive `VAW_results/`) | Typical time |
|---|---|---|---|
| `01_setup_and_m1_m2.ipynb` | tests, M1 manifest, M2 preprocess, sanity checks | `manifest.json`, `processed/`, `manifest_clean.json` | long (793 re-encodes), resumable |
| `02_m8a_track_poses.ipynb` | M8a: YOLOv8n-pose + ByteTrack + stitching on all cleaned clips (**GPU**) | `tracks/<Category>/*.npz`, `previews/` | the slowest step, resumable |
| `03_m8b_interaction_scores.ipynb` | M8b: Algorithm 1 scores, per-category summary, annotated video, hand check | `scores/<Category>/*_pairs.csv`, `*_curve.csv`, `scores/videos/` | minutes |
| `04_m9_buildup_report.ipynb` | **M9 behavior states, calibration, buildup videos, category report, review sheet** | `gate/`, `report/` (`index.html`, `categories.csv`, `clips.csv`, `review_sheet.csv`, `videos/`, `storyboards/`) | minutes to ~15 min |

Notebook 04 is the Phase 1 deliverable. What to look at and how to judge it: `docs/VERIFY_M9.md`.

## Paths
`configs/colab.yaml` holds the Drive paths. If your folder names differ, edit that file (and push) rather than the code.
Outputs go to Drive, so they survive the session and teammates can see them. M2 and M8a are resumable: re-running skips finished clips
(do not use `--force` unless you changed the settings and want everything recomputed).

## Changing thresholds
In notebook 04, section 3, set `SET = "follow_min_s=3 ..."` for a trial run. To make values permanent, edit the `gate:` block in both
`configs/extran.yaml` and `configs/colab.yaml`, commit and push, then `git pull` in Colab (first cell) and rerun gate and report.

## If something fails
- `data_root not found`: shortcut missing or named differently; check step 1 and `configs/colab.yaml`.
- `git clone` auth error: token missing, expired, or notebook access off.
- Many clips skipped in M1/M2: read the `SKIPPED` lines; they list the file and reason (AV1 clips are handled automatically).
- Session disconnected during 01/02: reconnect, rerun the first cell, then the run cell; it continues where it stopped.
- Report videos not playing in Drive's preview: download the `report` folder and open `index.html` locally.
