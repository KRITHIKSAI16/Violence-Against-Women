# Running the pipeline on Google Colab

Colab is for the full ExtrAnom set (~140-150 clips, Drive) and, from M8 on, the GPU.
Notebooks here contain only cells: mount Drive, get code, install, run a script, inspect.
All logic is in `src/`, so the same code runs on a laptop and on Colab.

## One-time setup (each teammate)
1. **Drive shortcut.** In Google Drive, right-click the shared `ExtrAnom` folder -> *Add shortcut to Drive* -> *My Drive*.
   (Shared-with-me paths are unreliable in Colab.) Expected path: `/content/drive/MyDrive/ExtrAnom/<Category>/*.mp4`.
2. **GitHub token** (the repo is private). GitHub -> Settings -> Developer settings -> Personal access tokens ->
   fine-grained, read access to this repo (*Contents: read*). In Colab: left sidebar *Secrets* (key icon) -> add
   `GITHUB_TOKEN`, turn on *Notebook access*. Never paste the token into a cell.
3. Open the notebook: Colab -> File -> Open notebook -> GitHub tab, or upload `colab/01_setup_and_m1_m2.ipynb`.
4. Runtime: M1/M2 do not need a GPU. Use a GPU runtime (Runtime -> Change runtime type) only for M8 notebooks.

## Notebooks
| Notebook | What it runs | Output (in Drive `VAW_results/`) |
|---|---|---|
| `01_setup_and_m1_m2.ipynb` | tests, M1 manifest, M2 preprocess, sanity checks | `manifest.json`, `processed/`, `manifest_clean.json` |
| `02_m8a_track_poses.ipynb` | M8a: YOLOv8n-pose + ByteTrack on all cleaned clips (GPU) | `tracks/<Category>/*.npz`, `previews/` |

More notebooks are added as modules (M8, M9, report) are built.

## Paths
`configs/colab.yaml` holds the Drive paths. If your folder names differ, edit that file (and push) rather than the code.
Outputs go to Drive, so they survive the session and teammates can see them. M2 is resumable: re-running skips clips already done.

## If something fails
- `data_root not found`: shortcut missing or named differently; check step 1 and `configs/colab.yaml`.
- `git clone` auth error: token missing, expired, or notebook access off.
- Many clips skipped: read the `SKIPPED` lines at the end of the notebook; they list the file and reason.
