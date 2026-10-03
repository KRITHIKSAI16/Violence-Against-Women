# Instructions for Claude Code in this repo

Read these fully before doing anything, in this order:
1. `PROJECT_BRIEF.md` — project context, dataset (793 clips), exact module spec (M1–M9), working style, known pitfalls.
2. `STATE.md` — what is done, what was found on the full dataset, decisions and why, next steps, open questions.
3. `docs/ARCHITECTURE.md` — data flow, every file format, the M9 behavior rules, design decisions and rejected alternatives, Phase II hand-off.
4. `docs/PHASE1_PLAN.md` (plan) and `docs/VERIFY_M8.md`, `docs/VERIFY_M9.md` (how to check outputs).

## Commands (run from the repo root)
```
pip install -r requirements.txt          # needs Python 3.10+ and ffmpeg/ffprobe on PATH
python -m pytest -q                      # no dataset or GPU needed; keep it green
python -m src.data.manifest | preprocess | src.assm.track_poses | interaction | gate | calibrate
python -m src.report.buildup_report      # category report; buildup_video <clip_id> for one clip
# all take --config configs/colab.yaml on Colab; gate/calibrate/report take --set key=value
```
Laptop sample: 12 clips in `data/sample/<Category>/` (git-ignored; ask the team for them). Colab runs the full set from Drive via notebooks in `colab/`.

## Rules
- Use the report's module names (M1, M2, M8, M9) in code, docstrings, commit messages. Build one module at a time and show real output before moving on.
- Code lives in `src/` as `.py` modules; Colab notebooks hold only cells (mount, pull, install, run a script, inspect).
- Every change needs tests; keep `pytest` passing. Verify on the 12 sample clips and look at rendered output, do not trust numbers alone.
- Never commit videos, tracks, Drive data or secrets (`.gitignore` covers data/). Never put the GitHub token in a file or notebook.
- Do not resurrect an LLM annotation module (`src/annotate/`). Phase 1 has no lead-time prediction (that is Phase II).
- Update `STATE.md` when module status or decisions change; edit `PROJECT_BRIEF.md` only if scope/dataset/constraints change.
- Be honest in results: compare categories with Normal, report limits, say when numbers are tuned in-sample.

## Git
The repo owner authorized Claude Code to commit and push to `origin main` in the owner's sessions (small commits, no force-push). If you are working for
another team member, ask your user before pushing.

## Pitfalls (details in PROJECT_BRIEF.md §8)
- Windows + git-bash: Python cannot see `/tmp`; avoid heredocs with nested quotes, use the editor tools to write files.
- Colab sessions drop: M2 and M8a are resumable; never use `--force` casually (recomputes everything).
- `python -m src.assm.gate --set ...` overwrites gate outputs with those thresholds; rerun without `--set` to restore the defaults.
- Detectors box chairs as people and miss small people; do not "fix" this by tuning to the 12 sample clips.
