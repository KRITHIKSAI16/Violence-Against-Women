# VAW Capstone — Current State

Live status. `PROJECT_BRIEF.md` = stable scope/spec. `docs/STATUS_REPORT.md` = readable summary with all results and the decisions log (share this with the team).
`docs/ARCHITECTURE.md` = how it is built and every file format. `docs/VERIFY_CONTEXT.md` = how to check every output.

Last updated: 2026-10-03.  Code: everything below is committed and pushed to `origin/main`; `python -m pytest -q` passes (176 tests).

---

## Status by module (report numbering M1–M9; see PROJECT_BRIEF.md §4)

| Part | Status | Notes |
|---|---|---|
| M1 Data acquisition | **done, run on full set** | 793 clips, 0 skipped (ffprobe fallback for 3 AV1 clips) |
| M2 Preprocessing | **done, run on full set** | 30 fps, longest side 640, aspect kept |
| M8a tracks + pose | **done, run on full set** | YOLOv8n-pose + ByteTrack (conf 0.1) + stitching |
| M8b Algorithm 1 score | **done, run on full set** | baseline; does NOT separate categories (Normal highest, 1.80) |
| M9 rule gate (5 behavior states) | **done, run on full set** | baseline; flagged Normal 27% vs 15% others; FOLLOW in 2 of 793 clips |
| Deep context Stage A (`features.py` etc.) | **done, run on full set (39 min)** | camera, meters, pose facing/contact/reach, lagged following |
| Evaluation (`learn.py`, notebook 06) | **done, run on full set** | behavior at chance on static-camera clips; style-only AUC 0.99 (source shortcut) |
| Stage B scene layout + depth (`scene.py`, nb 07) | **built, tested, run on 12 local clips and a simulated Drive; NOT yet run on Colab** | needs GPU runtime |
| Stage C scene graph + narrative (`graph.py`, `narrative.py`, `story.py`) | **built, tested, run on 12 local clips and a simulated Drive; NOT yet run on Colab** | |
| Story video + report page (`story_video.py`, `story_report.py`, nb 08) | **built, tested; NOT yet run on Colab** | |
| Annotation benchmark (`benchmark.py`) | **built, tested; waiting for the team's annotations** | the step that gives real precision / recall |
| M3–M7 | covered by `windows.py` / `learn.py` + the benchmark | no separate M3 module |
| Phase II (M10–M13) | not started, by design | |

## What to do next (in order)

1. Colab: `git pull`, run **notebook 07** (GPU, ~20-40 min) then **notebook 08** (CPU). Open `VAW_results/report/index.html`.
2. Look at the page together. For clips you will present: watch the video once, check the cut; add `clip_id,cut_s` lines to `configs/clip_overrides.csv` if the violence shows (push, `git pull`, rerun notebook 08 step 3).
3. **Team annotation** (notebook 08 steps 4-5): 4 sheets, about 60 clips; each person annotates without looking at the system's output; then `benchmark evaluate`; rerun step 6 so the page includes the benchmark table.
4. Discuss with the supervisor how to frame the result (`docs/STATUS_REPORT.md` section 5 and 8).
5. Only after the benchmark says where the errors are: consider a larger pose detector (about 40% of clips have no usable pair), calibrating `story.cue_weights`, a CCTV-style Normal subset.
6. Phase II hand-off is described in `docs/ARCHITECTURE.md` section 8.

## Key results so far (real numbers; details and tables in docs/STATUS_REPORT.md)

* Data: 793 clips — Normal 294, Harassment 188, Chain_Snatching 176, Kidnapping 73, Stalking 39, Assassination 23. 36% filmed with a moving camera (Normal only 21%). 36-45% of clips per category have no usable pair.
* M8b score (mean of per-clip means): Normal 1.80, Harassment 1.44, Chain 1.41, Assassination 1.01, Stalking 0.94, Kidnapping 0.74.
* Deep layer, share of clips with the behavior (Normal vs others pooled): approach from behind 19% vs 13%, contact 28% vs 29%, reach 15% vs 19%, follow 1% vs 1%, looking back 4% vs 2%, mutual facing 11% vs 2%.
* Evaluation (AUC, buildup categories vs Normal, by clip): style only 0.988 (clips with a pair), behavior 0.693 (pairs) / 0.492 (static camera), per-category 0.46-0.54, rule gate 0.44. F1 0.77 in the first run equals "predict all positive".
* Verified by eye on sample clips: reach/contact (a man grabs a woman), metric scale (walking 1.0-1.5 m/s), tracking IDs, fixed false "running away" (cropped boxes), fixed false night/daylight (lamp-lit night street).

## Decisions and why (short; full log in docs/STATUS_REPORT.md section 3)

Followed the report's M1–M9, no LLM annotation layer; M8 split into cached tracks + cheap scoring; body heights then meters; tracking with stitching; states/episodes instead of one score; speed features ignore cropped
people; scene layout and depth used early (Algorithm 2 for context only, no prediction); video ends before the first physical-act cue (heuristic, check each presented clip); evaluation by clip with a style-only shortcut probe
and Youden/balanced accuracy; generated data stays out of git; Claude Code may commit/push to `origin main` in the owner's sessions (no force-push, no data).

## Setup checklist

- [x] ExtrAnom sample (12 clips) local in `data/sample/<Category>/` (git-ignored); full set on Drive (shortcut in My Drive)
- [x] Repo, README, CLAUDE.md, docs, 176 tests, Colab notebooks 01–08, `configs/colab.yaml`
- [x] Colab runs done: M1, M2, M8a, M8b, M9 gate, Stage A, evaluation (notebooks 01-06)
- [ ] Colab runs to do: notebook 07 (scene + depth), notebook 08 (stories, report, benchmark sheets)
- [ ] Team annotation + `benchmark evaluate`
- [ ] Teammates added as GitHub collaborators; each needs the `GITHUB_TOKEN` Colab secret (`colab/README.md`)

## Known limits (state them when presenting)

2D views; approximate meters (assumed 60 degree field of view, accuracy not measured); small / distant people missed; false boxes; layouts fail on dark scenes and are ignored then; infrared night cameras look bright
(low light = dim image); hand-held cameras corrupt speeds; the category label is weak and the Normal clips are a different source; the video cut is a heuristic; detector precision / recall unmeasured until the benchmark.

## Open questions

* How to frame the result for the supervisor: description + evaluation of limits (recommended) vs. a category classifier (not supported by the data).
* After the benchmark: which cues are reliable enough to keep as "concern" cues, and the cue weights.
* A CCTV-style subset of Normal clips for a fairer comparison (needs a decision on how to select it without using the label).
* Whether to reintroduce UCF-Crime later as a second dataset (not needed now).

## How to keep this file useful
Update this file when a module status changes, a notebook has been run, a decision is made, or a question is resolved. Edit `PROJECT_BRIEF.md` only if scope, dataset or constraints change.
