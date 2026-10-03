# Phase 1 plan and what was actually built

Basis: Review Report I (Phase I = M1–M9) and PROJECT_BRIEF.md. This file records the plan, how it changed after the first full-dataset results, and the final delivered scope.
Status per part: `STATE.md`. Results and decisions: `docs/STATUS_REPORT.md`. How it is built: `docs/ARCHITECTURE.md`.

## Original plan (2026-10-02)
Goal: show the pre-violence buildup (approaching, following, hovering, cornering) for every ExtrAnom category from one common pipeline, without showing the violence, reusable by Phase II.
Pipeline: M1 manifest -> M2 clean clips -> M8a tracks + pose -> M8b Algorithm 1 score -> M9 gate -> report. Same code and config for all six categories; Normal is the baseline.

## What the first full-dataset run showed (2026-10-03)
* M8b score: Normal highest (1.80 vs Stalking 0.94). M9 rule gate: Normal flagged 27% vs 15% for the others; FOLLOW in 2 of 793 clips. 36-45% of clips have no usable pair.
* Decision: do not tune the same simple rules; use what was cached but unused (17 body keypoints), real-world geometry and scene context, and evaluate honestly.

## Revised plan, now delivered (stages; each built and tested before the next)
| Stage | What | Code | Notebook | Status |
|---|---|---|---|---|
| A | camera motion; meters (ground plane, Hall zones); body / head facing; contact and reach; lagged-path following; per-pair events | `src/context/{camera,ground,pose_features,follow,features}.py` | 05 | done, run on 793 clips |
| D | window table + cross-validated baseline by clip + style-only shortcut probe + static-camera and style-matched views | `windows.py`, `learn.py` | 06 | done, run |
| B | scene layout (SegFormer-B0: walkable, obstacle, door, vehicle, sky, vegetation), place type, doors, robust light; Depth Anything V2 Small on the key pair | `scene.py` | 07 | built, tested; to run |
| C | interaction scene graph (episodes), narrative sentences, cue curve, first physical-act cue | `graph.py`, `narrative.py`, `story.py` | 08 | built, tested; to run |
| Deliverable | story video (mini-map, cue strip, ends before the act), storyboard, report page, clip tables | `report/story_video.py`, `story_report.py` | 08 | built, tested; to run |
| Benchmark | annotation sheets, kappa, detector precision / recall vs humans, cut-point accuracy | `report/benchmark.py` | 08 | built, tested; waiting for annotations |

Baselines kept: M8b score (`src/assm/interaction.py`), M9 rule gate (`gate.py`, `calibrate.py`), older report (`buildup_report.py`, `buildup_video.py`).

## Same method for every category
Same code and config for all six. Stalking and Harassment: whole clip is available. Assassination / Chain_Snatching / Kidnapping contain the act: the video stops at the first physical-act cue minus 1 s, else a uniform 3 s tail trim,
never per-clip by hand except explicit overrides for clips you present. Normal is the comparison in every table.

## What is deliberately not done
Future-risk prediction, lead time, memory model (Phase II); InternVideo2; gender / age / identity inference; a language model in the pipeline; an LLM annotation layer.

## Phase II fit
Tracks + meters + poses = graph nodes and edge features (M10); episodes = predicate sequences for the plausibility grammar (M12); depth is already computed for the key pair (Algorithm 2); scene facts = context metadata;
benchmark annotations = first ground truth for horizon labels; `learn.py` results = the baseline Phase II must beat.

## Run order
Colab notebooks 01 -> 08 (see `colab/README.md`). Laptop: `python -m pytest -q`, then the commands in `README.md`.
