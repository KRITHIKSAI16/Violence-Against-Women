# Instructions for Claude Code in this repo

Read these fully before doing anything, in this order:
1. `docs/STATUS_REPORT.md` — what was done, found and decided (real numbers, decisions log, what is left).
2. `PROJECT_BRIEF.md` — project context, dataset (793 clips), module spec (M1–M9), working style, pitfalls.
3. `STATE.md` — live status and next steps.
4. `docs/ARCHITECTURE.md` — data flow, every file format, the deep context layer, design decisions, Phase II hand-off.
5. `docs/CURATED.md` (colab2: hand-picked clips with human start times T) and `docs/CLASSIFY.md` (colabclasstrim / colabclassfull: violent vs non-violent classifiers), then `docs/PHASE1_PLAN.md`, and the verification guides `docs/VERIFY_CONTEXT.md` (current), `VERIFY_M9.md` / `VERIFY_M8.md` (baselines).

## Commands (run from the repo root)
```
pip install -r requirements.txt          # needs Python 3.10+ and ffmpeg/ffprobe on PATH
python -m pytest -q                      # 178 tests, no dataset or GPU needed; keep it green
python -m src.data.manifest | src.data.preprocess | src.assm.track_poses        # M1, M2, M8a
python -m src.context.features | scene | story | learn                          # deep context stages A, B, C, D
python -m src.report.story_report | story_video <clip> | benchmark make/evaluate  # deliverable and benchmark
# all take --config configs/colab.yaml on Colab; several take --clips A B, --force, --set key=value
```
Newer, separate paths (they import the Phase 1 modules and never edit them): `python -m src.curated.run` (colab2, notebook `colab2/`), `python -m src.classtrim.run` and `python -m src.classfull.run` (stages `--stage ...`, `--out-root`, `--no-captions`; notebooks `colabclasstrim/`, `colabclassfull/`; settings `configs/classify.yaml`; shared code `src/classcommon/`).
Laptop sample: 12 clips in `data/sample/<Category>/` (git-ignored; ask the team). Colab runs the full set from Drive through notebooks `colab/01..08`.

## Rules
- Use the report's module names (M1, M2, M8, M9) in code, docstrings and commit messages. Build one stage at a time; show real output (look at rendered frames) before moving on.
- Code lives in `src/` as `.py` modules; Colab notebooks hold only cells (mount, pull, install, run a script, inspect).
- Every change needs tests with a known answer (synthetic worlds); keep `pytest` passing. Do not trust numbers alone: look at the frames they describe (this caught cropped-box "running", a wrong night flag, a map drawing a head-only box).
- Be honest in results: compare every category with Normal; Normal is a different source (style-only AUC 0.99), so never report category rates as accuracy; say when thresholds are tuned in-sample; label heuristics as heuristics
  (cue weights are untuned, the video cut is a heuristic, meters are approximate).
- Never commit videos, tracks, Drive data, generated outputs (`data/context`, `data/report`, ...) or secrets. Never put the GitHub token in a file or notebook.
- No future prediction (Phase II), no language model in the pipeline (one exception: the optional, switchable Layer B frame captions in `src/classcommon/vlm_caption.py`, ablated in the classify report; use `--no-captions` / `USE_CAPTIONS=False` to turn it off), no identity / gender / age inference, no `src/annotate/` LLM layer.
- Update `STATE.md` when status or decisions change and `docs/STATUS_REPORT.md` when results change; edit `PROJECT_BRIEF.md` only if scope, dataset or constraints change.

## Classification pipelines (read `docs/CLASSIFY.md` before touching them)
- trim = "will violence follow?" (colab2 clips, every clip judged on its last 3 s); full = "is violence in this video?" (whole videos, label = category folder, no start times). Outputs go to `VAW_results/classtrim/` and `VAW_results/classfull/`.
- Text documents are generated from numbers with identical wording for every clip and never contain violence/category words; keep it that way (leakage). Always report the `style` and `length` baselines next to every model; the verdict only claims behaviour when the best model beats both and holds on same-resolution clips.
- Best-model numbers are optimistic (selected on the same CV). Thresholds, PCA, imputers and fusion weights are fitted inside training folds only. Keep near-duplicate videos in one fold.
- GPU stages (captions, encoders, tracking) are not unit-tested; the logic around them is, with fake encoders / captioner (`tests/class_helpers.py`).

## Git
The repo owner authorized Claude Code to commit and push to `origin main` in the owner's sessions (small commits, no force-push). If you are working for another team member, ask your user before pushing.

## Pitfalls (more in PROJECT_BRIEF.md §8)
- Windows + git-bash: Python cannot see `/tmp`; heredocs with nested quotes break the shell, so write scripts and files with the editor tools.
- Colab sessions drop: M2, M8a, Stage A, B and C are resumable; never use `--force` casually (recomputes everything). A cell prints nothing until it finishes: check progress counters or the Drive folder.
- `src.assm.gate --set ...` overwrites gate outputs with those thresholds; rerun without `--set` to restore defaults.
- H.264 needs even frame sizes (the video layout heights are even on purpose).
- Detectors box chairs as people and miss small people; cropped / sitting boxes give wrong depth. Do not "fix" this by tuning to the 12 sample clips.
