# Phase 1 status report (for the team)

Written 2026-10-03. Everything here is also in the code, `STATE.md` and `docs/ARCHITECTURE.md`; this file is the readable summary to share and discuss.
Plain language, real numbers, and what we are NOT able to claim.

---

## 1. In one paragraph

We built the Phase 1 pipeline of the project on the ExtrAnom dataset (793 clips, six categories): clean the videos, detect and track people with pose, and then **describe, for every clip,
what happens between the people before any violence** (who approaches whom, follows, looks back, lingers, blocks, reaches), with times, approximate meters and scene facts, shown as a
video that stops before the physical act. We then **tested honestly whether these behaviors tell the buildup categories apart from Normal clips. They do not**, and we found out why: the Normal
clips come from a different kind of footage, and the category label is not a per-second annotation. So the deliverable is a detailed, checkable *description* of each clip plus an honest
evaluation, not a classifier. A small human-annotated benchmark (about 60 clips, the whole team) is the remaining step that turns "looks plausible" into measured precision and recall.

---

## 2. What Phase 1 contains (report modules M1-M9) and where we are

| Module (Review Report I) | What it does | Status |
|---|---|---|
| M1 Data acquisition | list every clip with category label and metadata | done, run on all 793 clips |
| M2 Preprocessing | 30 fps, longest side 640 px, corrupt-file handling (AV1 clips handled) | done, 793/793 |
| M8 ASSM (a) detection + pose + tracking | YOLOv8n-pose + ByteTrack, identity stitching | done, run on all clips |
| M8 ASSM (b) Algorithm 1 interaction score | `1/d + closing speed + path blocking` | done, run on all clips; **does not separate categories** (see section 5) |
| M9 Selective activation | first version: rule gate with five behavior states; **now superseded** by the deep context layer below | done, kept as the baseline |
| **Deep context layer** (new, `src/context/`) | camera motion, meters, body-pose facing/contact/reach, lagged following, scene layout, depth, scene graph, narrative | built and tested; Stage A and the evaluation run on Colab; scene/depth/story/report notebooks (07, 08) ready to run |
| M3-M7 (segments, features, EDA, baseline, benchmark) | covered in a controlled form by the evaluation (`learn.py`) and the annotation benchmark | evaluation run on Colab; benchmark waiting for the team's annotations |
| Phase II (M10-M13) | prediction, lead time, memory model | not started, by design |

---

## 3. What changed during the work, and why (decisions log)

| # | Decision | Why |
|---|---|---|
| 1 | Followed the Review Report's own module list (M1-M9), no LLM annotation layer | the report and the supervisor ask for an engineered, interpretable signal; the earlier Gemini-annotation idea was out of spec |
| 2 | Split M8 into a slow cached step (tracks + pose) and fast scoring steps | thresholds and rules can be retuned in seconds without rerunning YOLO on a GPU; the cache is also Phase II's input |
| 3 | Distances in **body heights** first, then **approximate meters** from the standing-height prior (1.7 m) | "close" must mean the same for near and far cameras; meters allow Hall's proxemic zones |
| 4 | Tracking: low detection threshold + longer track buffer + stitching of broken tracks | IDs flipped on small, moving people (6 IDs for 2 people in Stalking_v3); now far more stable |
| 5 | M8b score replaced by **behavior states / scene-graph episodes** as the context | on the full set the single score was highest for Normal clips (mean 1.80 vs Stalking 0.94); one number cannot say *what* is happening |
| 6 | Speed-based features ignore people whose box is cut by the frame edge | looking at frames showed every first "running away" detection was a cropped box giving a wrong depth |
| 7 | Added scene layout (SegFormer) and depth (Depth Anything V2 Small) from Phase II early | doors, walls and "same depth plane" give real meaning to blocking and contact; this is the report's Algorithm 2 |
| 8 | Video ends **before** the first physical-act cue (reach / contact / fast flee) minus 1 s margin; uniform 3 s trim for act categories with nothing detected | you asked not to show the violence; the act start was never annotated, so this is a heuristic and clips must be checked before presenting |
| 9 | Evaluation uses cross-validation **by clip**, equal weight per clip, and a **style-only shortcut probe** | to avoid fooling ourselves; this exposed the source difference of the Normal clips |
| 10 | F1-maximising thresholds replaced by Youden/balanced accuracy | on this data F1 collapses to "call everything positive" (F1 0.77 for free) |
| 11 | Everything generated stays out of git; Drive holds the data and results | repo is shared; videos and results are large |

---

## 4. The data (real counts)

793 clips: Normal 294, Harassment 188, Chain_Snatching 176, Kidnapping 73, Stalking 39, Assassination 23 (very unbalanced; Stalking, the most relevant category, is small).
3 Kidnapping clips are AV1 and need ffmpeg to decode (handled). Many Normal clips are phone-style recordings (478x6xx, often 60 fps); the others are mostly CCTV-style or news footage.
36% of clips were filmed with a moving or hand-held camera (Assassination 57%, Kidnapping 49%, Stalking 46%, Harassment 44%, Chain 43%, **Normal 21%**).
36-45% of clips per category have no usable pair of people (people too small, too far, or only one person), so nothing can be said about them.

---

## 5. What we found (the honest part)

### 5.1 The first scores did not separate categories
* M8b interaction score (mean of per-clip means): Normal 1.80, Harassment 1.44, Chain_Snatching 1.41, Assassination 1.01, Stalking 0.94, Kidnapping 0.74.
* M9 rule gate flagged **27% of Normal clips but only 15% of the buildup categories** pooled (Stalking 21%, Assassination 22%, Kidnapping 18%, Harassment 17%, Chain 10%); FOLLOW fired in 2 of 793 clips.

### 5.2 The deep context layer: behaviors are plausible but equally common in Normal clips
Share of clips in which each behavior was detected (Stage A, all 793 clips, 39 minutes on Colab):

| behavior | Normal | all other categories pooled |
|---|---|---|
| approach from behind | 19% | 13% |
| contact (hand near body) | 28% | 29% |
| reaching | 15% | 19% |
| following (lagged path) | 1% | 1% |
| looking back | 4% | 2% |
| mutual facing (conversation, benign) | 11% | 2% |
| camera moving (not a behavior: how it was filmed) | 21% | 45% |

Detections spot-checked by eye on clips were plausible (a man grabbing a woman is detected as reach and contact; the metric scale gave 1.0-1.5 m/s median walking speed on the sample clips,
0.78 m/s over the whole set including slow and crowded scenes). The point is that ordinary footage contains the same cues.

### 5.3 Controlled evaluation (`learn.py`, notebook 06): behavior does not reproduce the category label
793 clips, 420 with a usable pair, 7779 windows. AUC, buildup categories vs Normal, cross-validated by clip (0.5 = chance):

| model | AUC all clips (clips without a pair score 0) | AUC clips with a pair | AUC static-camera clips |
|---|---|---|---|
| style only (how it was filmed: camera, brightness, **original resolution and frame rate**) | 0.591 | **0.988** | 0.567 |
| behavior features only | 0.504 | 0.693 | **0.492** |
| behavior + scene | 0.507 | 0.703 | 0.495 |
| M9 rule gate (flag) | 0.441 | | |

Per category vs Normal (behavior model): Assassination 0.50, Chain 0.49, Harassment 0.54, Kidnapping 0.46, Stalking 0.51 = chance.
Reading: **(a) source leakage** - original resolution/frame rate identify Normal almost perfectly, so the Normal class is a different source; **(b)** on static-camera clips behavior is at chance;
**(c)** the 0.69 among clips with a pair most likely comes from correlates of framing (distance to camera, pair age), not from threatening behavior; **(d)** the labels are the clip's *category*, not what
happens each second: a Chain_Snatching clip is mostly a sudden snatch with no visible buildup, a Normal clip can contain people walking up behind each other.

### 5.4 What we can and cannot claim
* **Can:** a transparent, per-clip description of the interaction between people, with times, approximate meters, roles, target reaction and scene facts, up to just before the physical act; an evaluation showing the limits of behavior
  features on this labeled dataset; a reproducible pipeline that runs on a laptop and on Colab.
* **Cannot:** that the system detects "pre-violence buildup" as a category, that a cue is a threat, or precise detector accuracy - until the annotated benchmark is done.
* **Do not report:** the in-sample category rates as accuracy; F1 numbers from the first learning run (they equal "predict everything positive").

---

## 6. What the deliverable is (and how to read it)

1. **Story report page** (`VAW_results/report/index.html`): what it shows / does not show, key findings from the evaluation, category table (compare every row with Normal), and per category a showcase of
   top-evidence and random clips: scene line, timed narrative sentences, storyboard image, and the story video.
2. **Story video** (per clip): live sentences, key pair highlighted, doors outlined when the layout is reliable, bird's-eye mini-map in meters with proxemic rings, cue strip + weighted cue curve (a heuristic, unvalidated weights),
   red line where the footage is cut, then a card instead of the violent part.
3. **Clip tables** (`clips_story.csv`, `categories_story.csv`): every number behind the page.
4. **Annotation benchmark** (`report/benchmark/`): sheets for four annotators; `evaluate` gives precision/recall per behavior against human majority, annotator agreement (kappa), and the error of the "act starts here" cue.
5. **Evaluation results** (`VAW_results/learn/results.json`): the table in 5.3.

Example of a narrative line (from the code, deterministic text templates, no language model):
"id3 approaches id1 from behind (4.2 m -> 1.6 m)", "id3 follows id1 along the same path, about 1.8 s behind (average gap 2.1 m)", "id1 looks back toward id3 while walking away",
"id3 stands between id1 and a door", "hand-to-body contact between id3 and id1 (closest 0.12 m) [confirmed at the same depth]".

---

## 7. How to run everything (Colab, in this order; each notebook is resumable)

| # | Notebook | Runtime | Time | Produces |
|---|---|---|---|---|
| 01 | `01_setup_and_m1_m2` | CPU | long (793 re-encodes) | manifest, cleaned videos |
| 02 | `02_m8a_track_poses` | **GPU** | longest | tracks + poses |
| 03 | `03_m8b_interaction_scores` | CPU | minutes | Algorithm 1 scores (baseline) |
| 04 | `04_m9_buildup_report` | CPU | minutes | rule gate + first report (baseline) |
| 05 | `05_context_features` | CPU | ~40 min | meters, pose features, following, camera, events |
| 06 | `06_learn_and_evaluate` | CPU | ~15-20 min | the evaluation in 5.3 |
| 07 | `07_scene_layout_and_depth` | **GPU** | ~20-40 min | layout, doors, light, depth for the key pair |
| 08 | `08_story_report_and_benchmark` | CPU | ~15 min + annotation | stories, report page, annotation sheets, benchmark evaluation |

One-time setup (Drive shortcut, GitHub token secret) is in `colab/README.md`. Laptop: `pip install -r requirements.txt`, `python -m pytest -q` (176 tests), commands in `README.md`.

---

## 8. What is left, in order

1. Run notebooks 07 and 08 on Colab (everything is built and tested against a simulated Drive layout; the real run is the first time on the full set).
2. **Team annotation** (the key step): about 60 clips, 15 minutes per member per 15 clips; then `benchmark evaluate`. This turns the detectors from "plausible" into measured precision and recall, and shows
   whether the cut before the violence is accurate (`act_start_s`).
3. Look at the report page together; choose 5-10 clips to present (check each video once for the cut; add `clip_id,cut_s` to `configs/clip_overrides.csv` if needed).
4. Decide with the supervisor how to frame the result: the honest framing is "context description + evaluation of what geometry can and cannot show" (section 5), and Phase II (learned model, depth graph, memory) is where
   the category-level prediction belongs.
5. Optional improvements, only after the benchmark says where the errors are: a larger pose detector (about 40% of clips have no usable pair because people are too small), calibrating the cue weights on the benchmark,
   per-behavior thresholds, a style-matched Normal subset (Normal clips that look like CCTV).

## 9. Known limits (say them when presenting)

* 2D camera views; distances are approximate meters from an assumed 60 degree field of view and 1.7 m people. Their accuracy has NOT been measured; the only check so far is that the median walking speed comes out near real walking speed. Seated, cropped and far people give wrong values and are suppressed or flagged.
* Small or distant people are missed; 36-45% of clips per category have no usable pair.
* The scene layout fails on dark or unusual scenes; it is only used when its confidence is high and the camera is still. "Low light" means a dim image, not the time of day (infrared night cameras look bright).
* Hand-held cameras make speeds unreliable (flagged in each story).
* The category label is weak and the Normal clips are a different source.
* The video cut is a heuristic; no annotation of the act start exists yet.
* No gender, age or identity inference was done, on purpose (unreliable and ethically fraught); roles come from behavior only.

## 10. Where things are

`PROJECT_BRIEF.md` (scope and spec), `STATE.md` (live status), `docs/ARCHITECTURE.md` (design and file formats), `docs/VERIFY_CONTEXT.md` (how to check every output), `docs/VERIFY_M8.md`, `docs/VERIFY_M9.md` (baseline),
`docs/PHASE1_PLAN.md`, `colab/README.md`, `CLAUDE.md` (instructions for Claude Code), `README.md`.
