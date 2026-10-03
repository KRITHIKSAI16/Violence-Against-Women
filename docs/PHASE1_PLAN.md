# Phase 1 plan: pre-violence buildup per category (M1 -> M2 -> M8 -> M9)

Basis: Review Report I (Phase I = M1-M9) and PROJECT_BRIEF.md. Goal: a Phase 1 deliverable that shows the
**pre-violence buildup (approaching, following, hovering, cornering) for every ExtrAnom category**, from one common pipeline,
without showing the violence, and reusable by Phase II. Status per module: `STATE.md`. How it is built: `docs/ARCHITECTURE.md`.

## Pipeline
```
raw clips -> M1 manifest -> M2 clean clips -> M8a tracks+pose (GPU) -> M8b Algorithm 1 scores
                                                      \-> M9 behavior states + proposals -> calibrate -> buildup videos + category report
```
| Step | Module | What |
|---|---|---|
| M1 | `src/data/manifest.py` | list clips, labels (Normal = normal, others = pre_violence), metadata (OpenCV, ffprobe fallback for AV1) |
| M2 | `src/data/preprocess.py` | ffmpeg: 30 fps, longest side 640, aspect kept; skip corrupt |
| M8a | `src/assm/track_poses.py` | YOLOv8n-pose + ByteTrack (low conf, long buffer) + tracklet stitching; cache `.npz` |
| M8b | `src/assm/interaction.py` | Algorithm 1: d (body heights), closing speed, b_ij (frame-edge exit), score; per-pair CSV and curve |
| M9 | `src/assm/gate.py` | behavior states APPROACH / FOLLOW / HOVER / CORNER / ESCALATION per pair, flags, temporal proposals |
| M9 tools | `calibrate.py`, `hand_check_gate.py` | threshold sweep vs Normal; independent recompute of one frame |
| Deliverable | `src/report/buildup_video.py`, `buildup_report.py` | buildup video (stops before the act), storyboard, category report, review sheet |

## Why M9 is states, not a score
The Algorithm 1 score on the full set (Colab run): mean per category Normal 1.80, Harassment 1.44, Chain 1.41, Stalking 0.94,
Kidnapping 0.74. One number cannot say what is happening and Normal scored highest. The context is the ordered sequence of
behaviors between two people, so M9 reports who approaches/follows/hovers near/blocks whom, and for how long. The three states map
to the report's M9 heuristics (tracking persistence, path blocking, sudden abnormal motion).

## Same method for every category
Same code and config for all six. Stalking/Harassment: whole clip is buildup. Assassination/Chain_Snatching/Kidnapping contain the act:
the video stops at the detected escalation minus a margin, else a uniform tail trim (never per-clip by hand, except explicit overrides for
clips you present). Normal is the baseline in every table.

## Deliverable contents
Per category: share of clips flagged, with each behavior, ordered progression, median buildup length, time to escalation, no-pair share;
Normal as false-alarm baseline; showcase (top-evidence + random clips) with video and storyboard; a by-eye review sheet.

## Honest limits
2D geometry, small/distant people missed (13-28% of clips per category have no trackable pair), false boxes, domestic scenes, hand-held
cameras, no annotation of the act's start (cut is a heuristic), in-sample threshold calibration. Normal clips differ in style from the rest.

## Build/run order
1. Colab notebooks `01` (M1/M2) -> `02` (M8a, GPU) -> `03` (M8b, optional) -> `04` (M9, calibration, report). 2. Read the calibration table, set
thresholds, rerun gate + report. 3. Fill the review sheet, report precision by eye. 4. Pick clips to present; set cut overrides; render again.

## After Phase 1
M3-M7 (segment pairing, features, EDA, baseline classifier, benchmark) on the same manifest; then Phase II (M10-M13): depth-corrected distance,
interaction graphs from the tracks, plausibility-constrained training using the state sequences, lead-time evaluation.
