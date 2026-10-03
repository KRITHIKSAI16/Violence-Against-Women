# VAW Capstone — Current State

Where the project actually stands. `PROJECT_BRIEF.md` = stable context (scope, dataset, module spec, constraints).
`docs/ARCHITECTURE.md` = how it is built (data flow, file formats, design decisions). `docs/PHASE1_PLAN.md` = the plan.
This file changes often: read it to know what is done, what was found, and what to do next.

Last updated: 2026-10-03.  Code state: everything below is committed and pushed to `origin/main`; `python -m pytest -q` passes.

---

## Module status (report numbering M1–M9; see PROJECT_BRIEF.md §4)

| # | Module | Status | Notes |
|---|--------|--------|-------|
| M1 | Data Acquisition | **Done, run on full Drive set** | `src/data/manifest.py`. 793 clips, 0 skipped (after AV1 fix). OpenCV metadata, ffprobe fallback for AV1 |
| M2 | Preprocessing | **Done, run on full Drive set** | `src/data/preprocess.py`. 793/793 to 30 fps, longest side 640, aspect kept. Denoise off (config flag) |
| M3–M7 | Segments, features, EDA, baseline classifier, benchmark | Not started | M3 needs the "whole clip vs trimmed" decision (see open questions) |
| M8a | Track + pose | **Done, run on full Drive set** | `src/assm/track_poses.py`: YOLOv8n-pose + ByteTrack (conf 0.1, `configs/bytetrack_vaw.yaml`) + `stitch_tracks`; cache `tracks/*.npz` |
| M8b | Algorithm 1 scores | **Done, run on full Drive set** | `src/assm/interaction.py`; tools `render.py`, `hand_check.py` (hand recompute OK on Colab, Stalking_v3) |
| M9 | Selective Activation (behavior states) | **Built and tested locally; NOT yet run on Colab** | `src/assm/gate.py`, `calibrate.py`, `hand_check_gate.py` |
| Deliverable | Buildup videos + category report | **Built and tested locally; NOT yet run on Colab** | `src/report/buildup_video.py`, `buildup_report.py`; Colab notebook `04_m9_buildup_report.ipynb` |

Committed Phase 1 deliverable = M1 → M2 → M8 → M9 (+ the buildup report). Remaining for Phase 1: run notebook 04 on Colab, calibrate, review.

## What was found on the full set (Colab, 2026-10-03)

- **793 clips, very unbalanced**: Normal 294, Harassment 188, Chain_Snatching 176, Kidnapping 73, Stalking 39, Assassination 23.
  (The brief used to say ~140-150: wrong.)
- M8b "no interaction" (fewer than 2 people ever tracked together): Assassination 3/23, Chain 35/176, Harassment 25/188, Kidnapping 11/73,
  Normal 69/294, Stalking 11/39 (13-28% per category). Detector limits: small/distant/cropped people.
- **The M8b score does not separate categories**: mean of per-clip means Normal 1.80, Harassment 1.44, Chain 1.41, Assassination 1.01,
  Stalking 0.94, Kidnapping 0.74 (mean p95: Normal 3.60, Chain 3.74, Harassment 3.63, Stalking 2.98, Kidnapping 2.73, Assassination 3.17).
  One scalar mixes closeness/approach/blocking and cannot say what is happening. This is why M9 is behavior states (decision below).
- 3 Kidnapping clips are AV1 (OpenCV on Colab cannot decode): fixed with an ffprobe fallback; M2 re-encodes them.
- Local sample (12 clips) with default thresholds, M9: Kidnapping_v28 CORNER (id4 blocks id5, 2.2-3.8 s) then burst 3.9 s; Assassination_v2 APPROACH
  3.6-5.3 s then burst 5.6 s; Normal_v6 flagged HOVER (a man walks up to a seated woman at home: a geometry false alarm); Stalking_v3 NOT flagged
  (the scooter rider stays within arm's reach only ~1.5 s; with `hover_min_s=1.5` it flags, along with more false alarms). These are
  12-clip observations, not rates: calibrate on the full set.

## Decisions and why

- **Scope**: Phase 1 = M1–M9 only (report's own split). No lead-time prediction, no LLM annotation layer (`src/annotate/` must not return).
- **M8 split** into M8a (slow, GPU, cached) and M8b (cheap): retune without rerunning YOLO; cache is Phase II's input.
- **Body-height units** for distance/speed, **nearest frame edge as stand-in exit** for b_ij (no exit map in ExtrAnom), weights w1=w2=w3=1 (not tuned).
- **Tracking**: model run at conf 0.1 so ByteTrack's low-confidence pass works, buffer 90, plus tracklet stitching (ids per clip dropped, e.g.
  Kidnapping_v28 14→5, Stalking_v3 6→4). Raw ids kept as `raw_track_id`.
- **M9 = interpretable behavior states** (APPROACH, FOLLOW, HOVER, CORNER, ESCALATION) with roles and durations, flagged when FOLLOW/HOVER/CORNER or
  APPROACH→ESCALATION; ordered-progression flag; tracks with mean detection confidence < 0.35 ignored (chair boxed as person in Normal_v6).
  Thresholds are physical defaults NOT yet calibrated on the full set.
- **Deliverable**: buildup video that stops BEFORE the detected escalation (cut = escalation − 1.0 s; uniform 3 s tail trim for act categories with no
  detection; per-clip override CSV), storyboard, category report with Normal as false-alarm baseline, showcase = top-evidence + random clips,
  human review sheet. Cut margin 1.5 s was tried and hid almost all the cornering in Kidnapping_v28; default is 1.0 s (configurable).
- **Git**: Claude Code may commit and push to `origin main` (no force-push, never commit videos/Drive data). Teammates' Claude: ask your user first.

## What to do next (in order)

1. **Colab**: `git pull`, run notebook `colab/04_m9_buildup_report.ipynb` (CPU is enough). Read the calibration tables (Normal flag rate vs others).
2. Choose thresholds from the sweep (`--set` in the notebook, then copy into `configs/extran.yaml` + `configs/colab.yaml` `gate:` and push). Rerun gate + report.
3. Fill `report/review_sheet.csv` by eye (2 reviewers if possible); report precision by eye, not the in-sample rates. Watch every clip you will present;
   add `clip_id,cut_s` lines to `configs/clip_overrides.csv` for any clip where the automatic cut shows the act.
4. Decide with the supervisor: how to describe the results (see limits), and whether Normal (phone-style footage) needs a style-matched comparison.
5. If time remains in Phase 1: M3 (segment pairs from native labels) → M4 (features) → M5 (EDA) → M6 (baseline) → M7 (benchmark).
6. Phase II (M10–M13) hand-off is described in `docs/ARCHITECTURE.md` §7.

## Setup checklist

- [x] ExtrAnom sample data local: `data/sample/<Category>/` (2 per category, git-ignored); full set on Drive (shortcut in My Drive)
- [x] Scaffold, private GitHub repo `KRITHIKSAI16/Violence-Against-Women`, README, CLAUDE.md, docs
- [x] pytest suite (`tests/`), Colab notebooks 01–04 + `colab/README.md`, `configs/colab.yaml`
- [x] Colab run: M1, M2, M8a, M8b on the full set
- [ ] Teammates added as GitHub collaborators (each needs the `GITHUB_TOKEN` Colab secret, see `colab/README.md`)
- [ ] Colab run: M9 + report (notebook 04), calibration, review sheet

## Known limits (state them when presenting)
2D pixel geometry (camera angle; hand-held cameras corrupt velocities); small/distant people missed; false boxes; domestic scenes look like stalking;
no annotation of where the act starts (cut is a heuristic, check showcase clips); thresholds tuned in-sample; Normal clips are visually different
from the other categories; Algorithm 1's weights/tau never tuned (M8b kept as the report's algorithm and for Phase II).

## Open questions
- Final M9 thresholds (after calibration). Is a hover of ~1.5-2 s meaningful (it flags Stalking_v3 but also more Normal), or should Stalking be
  described as "approach + brief close contact"?
- M3: use Assassination/Chain_Snatching/Kidnapping clips whole or trimmed (uniform rule, e.g. same cut as the video)? Not decided.
- Reintroduce UCF-Crime later as a generalization set? Not needed now.
- Supervisor: is the behavior-state view (instead of the single Algorithm 1 score) acceptable as M9's "interaction heuristics"? It implements the same
  three heuristics the report names.

## How to keep this file useful
Update this file (not `PROJECT_BRIEF.md`) when a module status changes, a checklist item is done, a decision is made, or a question is resolved.
Edit `PROJECT_BRIEF.md` only if the spec, dataset or constraints change.
