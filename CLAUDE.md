# Instructions for Claude Code in this repo

Before doing anything, read these fully:
1. `PROJECT_BRIEF.md` - project context, dataset, exact module spec (M1-M9), working style.
2. `STATE.md` - what is done / next / open questions. Update it when module status changes.
3. `docs/PHASE1_PLAN.md` - the agreed Phase 1 plan and design.

Rules: use the report's module names (M1, M2, M8, M9) in code, docstrings and commits; build one module at a time and
show output before moving on; code lives in `src/` as `.py` modules (Colab notebooks hold only cells); keep `pytest`
passing; never commit videos or Drive data; do not resurrect an LLM annotation module (`src/annotate/`).
