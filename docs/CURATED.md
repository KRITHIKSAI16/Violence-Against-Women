# Curated dataset flow (`src/curated/`, `colab2/`)

A separate path from the Phase 1 pipeline for the hand-picked clips with human-annotated violence start times. Nothing in `STATE.md`, `STATUS_REPORT.md` or the earlier code was changed; the new code imports the old modules and never edits them.

## Why it exists
The goal for the jury is the *understanding of the interaction before the violence*. With a human start time T per clip the cut is no longer guessed: each clip is trimmed to [0, T) before any analysis, so nothing from the act can leak in, and the output explains the lead-up in numbers and sentences.

## Data flow
`FYP_data_filter.xlsx` + videos -> `index.py` (tolerant matching on category + clip number) -> `prepare.py` (re-encode 30 fps / 640 px, trim at T) -> existing stage CLIs on the curated config (`src.video.shots`, `src.assm.track_poses`, `src.context.features|scene|story`) -> `pipeline.py` + `interaction.py` + `relation.py` (per-clip understanding) -> `figures.py`, `video.py` (pair video + end card) -> `ml_encoders.py`, `ml_head.py` (pretrained video models) -> `evaluate.py` -> `report.py`.

## Modules
| file | role |
|---|---|
| `index.py` | read Excel (every sheet) / CSV, parse times, match names to files, report unmatched rows and unlabeled videos |
| `locate.py` | find the shared `violence` folder and the Excel on Drive, print the exact fix when missing |
| `config.py` | curated config built from `configs/colab.yaml` (paths moved under the new output folder; shorter co-tracking for short footage); `choose_perception` |
| `prepare.py` | trim each clip at T, Normal clips whole (capped at 45 s), write the manifest |
| `relation.py` | relation per frame and span: contact/reach, follows, approaches from behind, approaches, walking together, standing together, moving apart |
| `interaction.py` | per-pair summary: spans with who-does-what-to-whom, last 1.5 s / 3 s facts, lead-up type, arm-raised pose cue, concern score, sentences |
| `pipeline.py` | per-clip glue and the "why no pair" reasons |
| `figures.py`, `video.py` | distance figure with proximity zones; pair video (boxes, live meters, caption bar, distance strip) and end card |
| `ml_encoders.py` | InternVideo2 CLIP / X-CLIP / V-JEPA 2 as frozen encoders; zero-shot prompt margin; fallback chain |
| `ml_head.py` | leave-one-clip-out learned head, clip scores, within-clip trend |
| `evaluate.py` | coverage, within-clip trend, lead-up types, Normal comparison (AUC with bootstrap interval, style baseline, same-resolution check, verdict) |
| `report.py` | jury page, `summary.csv` |
| `run.py` | the stages and the CLI |

## Honest points
* Heuristics: relation thresholds, the lead-up rules and the concern weights are not tuned (listed on the report page). Meters are approximate.
* Perception is the main limit on short clips: a person who is not detected gives "no interacting pair"; the notebook chooses the detector by measured pair coverage. On a 3-clip local trial yolo26x@1280 + BoT-SORT found a pair in 1 clip where yolov8n@640 found none (CPU, 294 s per clip: use the GPU).
* Normal is a different source. The Normal comparison is judged against the best single filming-style feature and against a same-resolution subset; the verdict text only claims what survives both. The within-clip trend (last 3 s vs earlier, same clip) is the check that filming style cannot explain.
* The learned head and zero-shot scores come from out-of-fold models (a clip is never scored by a model that saw it) and are shown next to the geometric score in one ablation table.
* InternVideo2: its Hub page documents no usage; the loader probes the remote code for video/text feature functions and falls back to X-CLIP when it cannot find them. X-CLIP (`microsoft/xclip-base-patch16`, 8 frames) was checked on real clips locally.
* HIG: the interaction graph uses the HIG vocabulary (situation, position, interaction, relation) but is not the pretrained HIG model (no confirmed public weights); appearance is skipped on purpose.
* Dropped from the first plan: comparing the old rule gate's escalation time with T. The clips are trimmed at T, so the gate would see no escalation by construction; it could be added by running the gate on the full clips.
