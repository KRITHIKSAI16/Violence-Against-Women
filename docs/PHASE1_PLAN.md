# Phase 1 plan: pre-violence buildup per category (M1 -> M2 -> M8 -> M9)

Basis: Review Report I (Phase I = M1-M9) and PROJECT_BRIEF.md. Goal: a Phase 1 deliverable that shows the
pre-violence **buildup curve for every ExtrAnom category**, from one common pipeline, reusable by Phase II.

## Pipeline
```
raw clips -> M1 manifest -> M2 clean clips -> M8a tracks+pose -> M8b interaction scores -> M9 gate -> report
```
| Step | Where | What |
|---|---|---|
| M1 `src/data/manifest.py` | laptop/Colab | list clips, labels (Normal=normal, others=pre_violence), cv2 metadata |
| M2 `src/data/preprocess.py` | laptop/Colab | ffmpeg: 30 fps, longest side 640, aspect kept; skip corrupt |
| M8a `src/assm/track_poses.py` | Colab GPU | YOLOv8n-pose (conf 0.4) + ByteTrack (`persist=True`); cache per-frame, per-track bbox/keypoints/centroid (.npz) |
| M8b `src/assm/interaction.py` | CPU | Algorithm 1 from cached tracks: `score_ij = w1/d + w2*v + w3*b`; per-clip timeline + clip curve (max over pairs) |
| M9 `src/assm/gate.py` | CPU | flag when score >= tau for >= N s; tracking persistence, sudden motion, path blocking -> proposals (start, end, reason) |
| Report `src/report/buildup_report.py` | CPU | per-clip curves, per-category normalized-time overlay vs Normal, summary CSV, index.html |

Why M8 is split: YOLO is the slow part; weights, tau and b_ij can be retuned by rerunning only M8b/M9 (seconds).
The cached tracks are exactly what Phase II (M10) needs.

## Common handling of all six categories
Same code and config for every category. Stalking/Harassment: whole clip is buildup. Assassination/Chain_Snatching/
Kidnapping contain the violent act: one uniform `tail_trim_s` config (default 0; report shows full and trimmed once),
never per clip. Normal is the baseline in every plot. Fewer than 2 tracked people -> "no interaction", not an error.

## Algorithm 1 details we chose
- `d_ij`: centroid distance normalized by frame diagonal (clips differ in size). `v_ij`: closing speed, lightly smoothed.
- `b_ij`: ExtrAnom has no exit map, so the nearest frame edge stands in for the exit; `w3` is configurable (compare w3=0 vs 1).
- Start w1=w2=w3=1; choose tau from score distributions on the full set. Depth (Depth Anything) is Phase II.

## Phase II fit
Cached tracks feed M10 graphs; Algorithm 2 only swaps `d_ij` for depth-corrected distance on gated frames; gate proposals
give a first estimate of segment timing for horizon labels; M3-M7 can be added on the same manifest.
Limitation: 2D pixel heuristics depend on camera angle (crowded Normal scenes may score high) - what Phase II addresses.

## Testing
Laptop: `pytest` on synthetic data + smoke run on the 12 sample clips. Colab: `pytest -q` as env check, then the same
modules with `configs/colab.yaml`; each script prints a summary.

## Build order
1. M1, M2, tests, Colab folder, team docs (done). 2. M8a. 3. M8b. 4. M9. 5. Report on samples. 6. Full Colab run + tuning.
