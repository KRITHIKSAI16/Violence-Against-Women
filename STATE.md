# VAW Capstone — Current State

This file tracks where the project actually stands, updated as work happens.
`PROJECT_BRIEF.md` has the full stable context (scope, dataset, module
spec, constraints) and should not need to change often. This file changes
often — read it to know what's actually done vs. still TODO.

Last updated: 2026-10-02

---

## Module status (using the report's own M1–M9 numbering — see PROJECT_BRIEF.md §4)

| # | Module | Status | Notes |
|---|--------|--------|-------|
| M1 | Data Acquisition | Built, tested locally | `src/data/manifest.py` writes `data/manifest.json` (12/12 sample clips, 0 skipped; corrupt-file skip tested). Not yet run on full Drive set (Colab) |
| M2 | Preprocessing | Built, tested locally | `src/data/preprocess.py`: ffmpeg re-encode to 30 fps, longest side 640 (aspect kept), denoise off by default; writes data/processed/ + data/manifest_clean.json. 12/12 ok; corrupt-file skip tested. Not yet run on full set |
| M3 | Pre-Violence Segment Identification | Not started | No annotation needed — uses ExtrAnom folder names as native labels directly (see PROJECT_BRIEF.md §3) |
| M4 | Feature Engineering & Caching | Not started | |
| M5 | Exploratory Data Analysis | Not started | |
| M6 | Baseline Classifier | Not started | |
| M7 | Benchmark Evaluation | Not started | |
| M8 | Adaptive Suspicious Activity Screening (ASSM) | Partial | YOLOv8-Pose detection working (from earlier UCF-Crime work, needs re-pointing at ExtrAnom); ByteTrack not yet added; Algorithm 1 scoring (distance/closing-speed/path-obstruction) not yet implemented |
| M9 | Selective Activation | Not started | Depends on M8 |

**Committed Phase 1 deliverable = M1 → M2 → M8 → M9** (independent track,
doesn't require M3–M7 to be done first — see PROJECT_BRIEF.md §6).
M3–M7 are the natural next set once that track is solid.

## Repo / environment setup status

- [x] ExtrAnom sample data in place: `data/sample/<Category>/`, 2 clips each,
      6 categories (12 clips total)
- [x] Claude Code installed and pointed at the project folder (VAW)
- [~] Minimal scaffold created (configs/extran.yaml, src/config.py, requirements.txt, .gitignore); no README yet. Clips moved ExtrAtom/ -> data/sample/, 'Assasination' -> 'Assassination'
- [ ] Git repo initialized locally
- [ ] GitHub repo created (private) and remote connected
- [ ] First commit + push done
- [ ] Teammates added as collaborators on GitHub
- [ ] Colab notebook set up to `git pull` the repo and mount Drive
- [ ] ExtrAnom Drive folder added as a shortcut to "My Drive" (needed for
      reliable Colab access — unconfirmed if done yet)

## Decisions made this session (2026-10-02)

- **Corrected the module framework entirely.** Earlier session drafts used
  invented module names and an LLM-annotation-based "context understanding"
  plan. Re-read the actual submitted Review Report I: Phase 1 is already
  specified as Modules M1–M9 (Screening + Understanding stages only, no
  lead-time prediction, no annotation step). The supervisor's instruction
  to focus on "pre-violence buildup" for Phase 1 is a correction back to
  this original M1–M9 spec, not a new direction.
- M8's **Algorithm 1** (pairwise interaction score: `score_ij = w1/d_ij +
  w2·v_ij + w3·b_ij`, using YOLOv8-Pose + ByteTrack) is the concrete,
  already-specified mechanism that produces the "buildup context" —
  computed per-frame across a clip, it traces the behavioral escalation
  curve directly. Full algorithm in PROJECT_BRIEF.md §4.1.
- Dropped the LLM annotation module (`src/annotate/`) entirely — not part
  of the report's Phase 1 spec, do not resurrect it.
- Reconfirmed ExtrAnom as the primary (only, for now) Phase 1 dataset.
  Its category folder names serve directly as M3's "native labels" — no
  annotation needed to produce them. UCF-Crime work (including its Gemini
  onset annotations) is archived/reference-only.
- Reconfirmed working split: Claude Code does local scaffolding/coding on
  the laptop; Colab is batch-GPU-only, pulling from GitHub; user pushes to
  GitHub manually, not Claude Code.
- New committed Phase 1 deliverable: **M1 → M2 → M8 → M9**, chosen because
  it's a self-contained track per the report's own figure (M8–M9 run in
  parallel with M1–M7, not after).

## Immediate next steps (in order)

1. Confirm the scaffolding prompt finished correctly — check the folder
   tree Claude Code produced, confirm `data/sample/` was untouched.
2. User: `git init`, create private GitHub repo, connect remote, first
   commit + push (manual steps given earlier — not repeated here).
3. **M1** — Build `src/data/manifest.py` against the 12 local sample clips:
   scans `data/sample/<Category>/`, emits a manifest with clip_id/category/
   label (Normal = negative, all others = positive)/source/path.
4. **M2** — Build `src/data/preprocess.py`: corrupt-clip handling, frame
   rate/resolution normalization, logging of clip duration/frame count.
5. **M8** — Re-point existing YOLOv8-Pose code at ExtrAnom sample clips
   (not old UCF-Crime paths), add ByteTrack (`tracker="bytetrack.yaml",
   persist=True`), then implement Algorithm 1 exactly as specified in
   PROJECT_BRIEF.md §4.1 — output per-clip `{score_ij}` timelines.
6. **M9** — Build the selective-activation heuristic on top of M8's
   output: decide, per segment, whether it crosses into "needs deeper
   analysis" based on sustained high interaction scores.
7. Once M1/M2/M8/M9 work on the 12-clip local sample, move to Colab: add
   ExtrAnom Drive shortcut, `git pull`, run the same scripts against the
   full ~140-150 clip dataset.
8. If time remains in Phase 1: M3 (segment pairing from native labels) →
   M4 (feature caching) → M5 (EDA) → M6 (baseline classifier) → M7
   (benchmark evaluation).

## Open questions / things to confirm with supervisor or team

- Exact values for Algorithm 1's weights (`w1, w2, w3`) and trigger
  threshold (`τ_interaction`) — not fixed in the report. Plan: start with
  equal weights, compute score distributions on the 12 sample clips, tune
  from there before running at full scale.
- Need a static "marked exits" map for the `b_ij` path-obstruction term in
  Algorithm 1 — the report assumes this exists (e.g. for a monitored
  corridor/building). For ExtrAnom's varied, uncontrolled scenes, decide
  whether to: (a) skip `b_ij` for Phase 1 and use only distance + closing
  speed, (b) approximate "exit" as frame edges/off-screen direction, or
  (c) define per-clip manually for a small pilot set. Needs a decision
  before M8 is finalized — flagged for the team/supervisor.
- Whether Assassination/Chain_Snatching/Kidnapping clips should be used
  whole in M3, or trimmed (e.g. drop the last 1–2 seconds) so the later
  baseline classifier (M6) focuses on pre-violence context rather than the
  overt violent act itself. Leaning toward a simple fixed trim rule,
  applied uniformly — not decided yet.
- Whether UCF-Crime will be reintroduced later as a secondary/
  generalization dataset — not needed now, revisit after M1/M2/M8/M9 are
  done.

## How to keep this file useful

Update this file (not `PROJECT_BRIEF.md`) whenever:
- a module (M1–M9) status changes
- a setup checklist item gets done
- a new decision gets made that affects what to build next
- a new open question comes up, or an existing one gets resolved

Keep `PROJECT_BRIEF.md` stable — only edit it if the module spec, dataset,
or constraints themselves actually change again.
