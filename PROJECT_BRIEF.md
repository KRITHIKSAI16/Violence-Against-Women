# VAW Capstone — Project Brief (Full Reference)

This file is the single source of truth for project context. Claude Code
should read this at the start of any session before doing scaffolding or
pipeline work. It does not change often — day-to-day progress and current
status live in `STATE.md` instead, which gets updated as work happens.

This version replaces an earlier draft that used invented module names and
an LLM-annotation step. Both were wrong turns — corrected below using the
actual submitted Review Report I module structure.

---

## 1. Project identity

- **Title:** Video Context Understanding: Predicting and Preventing Violence
  (submitted title — "predicting" refers to the full two-phase project;
  Phase 1 itself does not predict/forecast, see §2)
- **Institution:** Sri Sivasubramaniya Nadar College of Engineering, Dept.
  of Computer Science and Engineering
- **Team:** Sowmya Anand, Srinidhi Rajagopal, G Sudharsan Manickam, Krithik
  Sai S — 4 members
- **Supervisor:** Dr. Anusha Jayasimhan, Assistant Professor
- **Document of record:** "Review Report – I" (Academic Year 2026–2027).
  This brief is derived from that report plus the supervisor's subsequent
  verbal clarification on Phase 1 scope (see §2).
- **Core framing (from the report's motivation):** Existing surveillance
  systems treat violence as a post-incident binary classification problem
  and are blind to the progressive, multi-step behavioral timeline that
  precedes a violent act. Threats against women in particular often begin
  with low-kinetic predatory behavior — persistent following, unwanted
  close proximity, coercive path obstruction, cornering — which systems
  tuned to detect blatant confrontations miss entirely.
- Written up in parallel as a research paper (Elsevier journal format).

## 2. Phase scope — READ THIS CAREFULLY

The original report's own architecture (Fig 3.1, "Methodology") already
splits the system into four stages: **Screening → Understanding → Temporal
Memory → Risk Prediction.** The report's own phase plan (§3.2–3.3) assigns
these stages to the two phases as follows:

- **Phase I (Months 1–4) = Screening + Understanding stages = Modules
  M1–M9.** No future-horizon risk scores, no Lead-Time AUC, no temporal
  memory model. This is a data + detection + interaction-scoring pipeline.
- **Phase II (Months 5–8) = Temporal Memory + Risk Prediction stages =
  Modules M10–M13.** This is where t+2s/t+5s/t+10s risk horizons, the
  STRD streaming memory module, plausibility-constrained training, and
  Lead-Time AUC all live.

**The supervisor's instruction to drop lead-time prediction for now is not
a new scope change — it is a correction back to what the report's own
Phase I (M1–M9) already specified.** Earlier working-plan drafts in this
project (an LLM/Gemini-based onset-timestamp annotation effort, framed as
part of "Phase 1") were in fact Phase-II-flavored work done early, outside
the submitted M1–M9 spec. That effort is now set aside — see §5.

**What Phase 1 (M1–M9) concretely requires:**
1. Acquire and clean a video corpus with native category labels (M1, M2)
2. Identify pre-violence segments directly from those native labels — no
   LLM annotation, no onset-timestamp localization (M3)
3. Extract and cache reusable spatio-temporal features from those segments
   (M4)
4. Exploratory analysis of the cached features (M5)
5. Train and evaluate a baseline (non-sequential) classifier distinguishing
   pre-violence from normal segments (M6, M7)
6. Independently (parallel track, not dependent on M1–M7): run lightweight
   person detection + persistent tracking + a pairwise interaction-scoring
   heuristic across every frame of every clip, and use it to gate which
   segments need deeper analysis (M8 — ASSM, M9)

**This interaction-scoring step (M8's Algorithm 1) IS the "pre-violence
buildup context" the supervisor is asking for.** It is already fully
specified with a concrete formula (see §4). Run across a clip's full
duration, the resulting score is literally the buildup curve: rising and
staying elevated for Stalking-type behavior, flat/noisy for Normal clips.
No natural-language explanation layer is needed or wanted — the engineered,
interpretable score itself is the context/explanation output for Phase 1.

**Explicitly out of scope for Phase 1** (deferred to Phase II, M10–M13):
- Lead-time / future-horizon risk prediction (t+2s, t+5s, t+10s)
- The Lead-Time AUC evaluation metric
- The STRD (Streaming Temporal Relational Dynamics) memory module
- Plausibility-constrained / counterfactual sequence training
- InternVideo2 semantic feature extraction (listed in the report's
  "Understanding" stage architecture, but not required for a working
  Phase 1 M1–M9 delivery — the baseline classifier in M6 can run on
  pose/interaction-derived features alone; InternVideo2 can be added
  later if time permits, but is not a hard Phase 1 dependency)
- LLM-based clip or window annotation (Gemini or otherwise) — no behavior
  taxonomy, no structured annotation schema, no annotation runner
- Depth-aware (Depth Anything) proximity refinement — this is introduced
  in M10 (Phase II) when building the full interaction graph; M8's
  Algorithm 1 for Phase I uses plain 2D pixel distance, which is
  sufficient for the screening-stage buildup signal

## 3. Dataset

### Primary dataset: ExtrAnom
- Video-only dataset (no existing labels beyond folder/category name, no
  timestamps, no metadata files) — this is exactly the kind of input M3
  expects ("cleaned corpus with its native violence/non-violence labels")
- Organized as one folder per category, each ~23–25 clips:
  - `Assassination`
  - `Chain_Snatching`
  - `Harassment`
  - `Kidnapping`
  - `Normal`
  - `Stalking`
- The category folder name IS the native label M3 consumes. No annotation
  step is needed to produce it.
- **Stalking** and **Harassment**: the entire clip length is the relevant
  pre-violence behavioral signal — there is no distinct "event moment"
  inside them to window around, so for M3 the whole clip can serve directly
  as the pre-violence segment.
- **Assassination / Chain_Snatching / Kidnapping**: have a more visually
  identifiable violent moment within the clip, but Phase 1 does not need
  to locate it precisely (that precision belongs to Phase II's segment
  timing work in M10). For Phase 1, using the whole clip (or a simple
  fixed trim, e.g. drop the last 1–2 seconds if the violent act is visibly
  concentrated there) is sufficient and should be decided once, consistently,
  not per-clip by hand.
- **Normal** is the negative class, used as M3's comparison windows.
- Currently: 2 sample clips per category downloaded locally to the project
  under `data/sample/<Category>/` (12 clips total) for development with
  Claude Code. The full dataset (~23–25 clips/category, ~140–150 clips
  total) lives in Google Drive under a "Shared with me" folder named
  `ExtrAnom`, and is processed via Google Colab (GPU), not on the laptop.
- To make the ExtrAnom folder reliably visible inside Colab, it should be
  added as a shortcut to "My Drive" (right-click the shared folder → "Add
  shortcut to Drive") rather than relied upon as a Shared-with-me path.

### Legacy dataset: UCF-Crime (archived, not part of active Phase 1 pipeline)
- Used in earlier work before the pivot to ExtrAnom, and referenced in the
  report's literature survey as dataset [1] (Sultani et al. 2018) — the
  report's own M1 spec actually names UCF-Crime, RWF-2000, and XD-Violence
  as the intended Phase 1 corpus. ExtrAnom is a substitution the team has
  since made (VAW-specific categories fit the project's framing better than
  general crime categories) — this substitution is reasonable and doesn't
  conflict with the M1–M9 architecture, since M1–M9 only requires "a video
  corpus with native labels," not these specific three datasets by name.
- Correct source used previously: `minhajuddinmeraj/anomalydetectiondatasetucf`
  (~49GB) — note the `odins0n` version of this dataset contains no actual
  video files and should not be used again if UCF-Crime is ever revisited.
- 58 clips were previously selected (29 violent, 29 normal) and had
  YOLOv8-Pose extraction completed and verified on all 58.
- Onset timestamps for these clips were annotated with LLM assistance
  (Gemini) and verified against clip-level ground truth. **This annotation
  work belongs to the earlier, since-corrected Phase-II-flavored effort
  (§2) and is not reused in the active M1–M9 pipeline.**
- These clips and their annotations remain in Drive as historical/reference
  material only, under the old `capstone_dataset/` folder (subfolders:
  `annotations/`, `frames/`, `pose_output/`, `raw_clips/`). Could be
  revisited later as an optional secondary/generalization dataset, not a
  Phase 1 dependency.
- Old Drive paths (reference only, not used by the new pipeline):
  - raw_root: `/content/drive/MyDrive/Colab Notebooks/capstone_dataset/raw_clips`
  - pose_output: `/content/drive/MyDrive/Colab Notebooks/capstone_dataset/pose_output`
  - progress file: `.../annotations/pose_progress.json`
- A stray duplicate folder at `/content/drive/MyDrive/capstone_dataset`
  (unrelated to the `Colab Notebooks/` one above) existed and was flagged
  for cleanup — may still need deleting, not urgent.

## 4. Phase 1 modules — M1 through M9 (as specified in Review Report I)

This is the authoritative module list. Use these exact names/numbers in
code, commit messages, and any documentation — they must match what was
submitted to the supervisor.

| # | Module | Input | Processing | Output | Depends on |
|---|--------|-------|------------|--------|------------|
| **M1** | Data Acquisition | Public dataset source(s) | Download/consolidate video corpus | Unified raw video corpus | none |
| **M2** | Preprocessing | Raw corpus | Handle missing/corrupt clips, denoise, normalize frame rate/resolution | Cleaned corpus | M1 |
| **M3** | Pre-Violence Segment Identification | Cleaned corpus with native labels | For each violence-labeled clip, extract the window of footage leading up to (or constituting) the labeled event; pair with comparable windows from normal clips | Pre-violence and normal segment pairs | M2 |
| **M4** | Feature Engineering & Caching | Pre-violence/normal segments (M3) | Extract reusable spatio-temporal features, persist them | Cached feature repository | M2, M3 |
| **M5** | Exploratory Data Analysis | Cached features | Study class distribution, temporal patterns, correlations, class imbalance | Data-quality and design insights | M4 |
| **M6** | Baseline Classifier | Cached features + EDA insights | Train a conventional (non-sequential) classifier: pre-violence vs normal | Trained baseline model | M4, M5 |
| **M7** | Benchmark Evaluation | Baseline predictions | Compute Accuracy, Precision, Recall, F1-score | Benchmark report (comparison point for Phase II) | M6 |
| **M8** | Adaptive Suspicious Activity Screening Module (ASSM) | Raw frames | Lightweight person detection + pose (YOLOv8/YOLOv8-Pose), ByteTrack for persistent identity across frames, compute pairwise interaction scores (Algorithm 1, §4.1 below) between tracked people every frame | Candidate interaction signals with persistent track IDs | none — independent track, runs on every frame regardless of M1–M7 |
| **M9** | Selective Activation | M8 interaction signals | Apply interaction heuristics (tracking persistence, sudden abnormal motion, path blocking) to decide whether a segment needs deeper analysis | Temporal proposals (validated in Phase I, routed to Phase II's M10 later) | M8 |

**Important structural note from the report's own figure (Fig 3.2):** M8–M9
form an independent track that depends only on off-the-shelf detection/pose
tooling, not on anything built in M1–M7. They can (and should) be developed
and validated in parallel with the M1–M7 data/classifier track, not after
it.

### 4.1 Algorithm 1 — ASSM interaction scoring (the core "buildup" signal)

This is the concrete, already-specified algorithm that produces the
pre-violence buildup context for Phase 1. Reproduced here exactly as it
appears in the report (Algorithm 1, "Adaptive Suspicious Activity
Screening (ASSM) Gate"):

**Requires:** frame `f_t`, tracker state `T_{t-1}` (ByteTrack tracklets +
pose history), a static map of marked exits, threshold `τ_interaction`

**Ensures:** trigger flag, updated tracker state `T_t`, ID-indexed poses
`P_t`, pairwise interaction scores `{score_ij}`

```
1.  D_t ← YOLOv8(f_t)                      # person bounding boxes + confidence
2.  if |D_t| < 2: return (false, T_{t-1}, ∅, ∅)   # need ≥2 people to interact
3.  T_t ← ByteTrack(D_t, T_{t-1})          # associates boxes to existing tracklets
                                            # (keeps identity through brief occlusion/blur)
4.  P_t ← YOLOv8-Pose(f_t, T_t)            # body keypoints per tracked identity i
5.  trigger ← false
6.  for each pair (i, j), i < j, present in both T_t and T_{t-1}:
7.      d_ij ← ||centroid(P_t[i]) − centroid(P_t[j])||_2   # current pixel distance
8.      v_ij ← (d_ij(P_{t-1}) − d_ij(P_t)) / Δt            # closing speed between the
                                                             # SAME tracked identities;
                                                             # positive = approaching
9.      b_ij ← 1 if j lies between i and the nearest marked exit, else 0  # path obstruction
10.     score_ij ← w1/d_ij + w2·v_ij + w3·b_ij
11.     if score_ij ≥ τ_interaction: trigger ← true
12. return (trigger, T_t, P_t, {score_ij})
```

**Why this is the "buildup context":** computed every frame across a
clip's full duration, `score_ij` over time is literally the behavioral
buildup curve. For a Stalking clip, expect it to rise and stay elevated
(sustained proximity + persistent closing speed). For a Normal clip,
expect it to stay low/flat/noisy. This is interpretable by construction —
no separate explanation layer is needed; the three terms (distance,
closing speed, path obstruction) ARE the explanation.

`w1, w2, w3` and `τ_interaction` are tunable weights/threshold — not yet
fixed in the report; need to be chosen empirically once scores are
computed on real sample clips (start with equal weights, e.g. w1=w2=w3=1,
and inspect score distributions before tuning).

Note: the report's Phase II (M10) later upgrades `d_ij` from raw 2D pixel
distance to a depth-corrected pseudo-3D distance using Depth Anything —
**this refinement is explicitly Phase II**, not required for Phase 1's
Algorithm 1.

## 5. What changed from the earlier working plan (for context, not action)

An earlier draft plan for this project proposed using an LLM (Gemini) to
generate structured behavioral annotations per clip (a closed vocabulary
of behaviors like "following," "cornering," scene-context tags, etc.),
framed as the "context understanding" deliverable for Phase 1. That
approach is **no longer part of the plan.** Reasons:
1. The supervisor directed that Phase 1 needs the pre-violence buildup
   construct, not an annotation/explanation layer.
2. The original Review Report I's own M1–M9 architecture never included
   an annotation module — it already specifies M8's Algorithm 1 as the
   interpretable interaction signal. The annotation idea was an
   out-of-spec addition, not a replacement for something the report asked
   for.
3. Do not resurrect an annotation module (`src/annotate/`) for Phase 1.

## 6. Committed Phase 1 deliverable (the "4 modules")

Given YOLOv8-Pose work is already underway (part of M8), the most
coherent, independently-deliverable set of 4 modules is:

**M1 → M2 → M8 → M9**

This is a complete, self-contained track — confirmed by the report's own
figure note that M8–M9 run in parallel with M1–M7 rather than after them —
taking raw ExtrAnom clips all the way through to tracked, scored
interaction signals with a working screening gate. This directly produces
the pre-violence buildup context the supervisor asked for, end to end.

M3–M7 (segment pairing, feature caching, EDA, baseline classifier,
benchmark) form the natural next set once M1/M2/M8/M9 are solid, time
permitting within Phase 1.

## 7. Working style / constraints

- Code lives in a git repo as real `.py` modules, not notebook cells. Teammates run
  it on their laptops and on Colab (`colab/` folder).
  Claude Code works on this repo locally, on the user's laptop.
- Google Colab is used ONLY for GPU-heavy batch runs against the full
  Drive-hosted dataset. Colab pulls the repo via `git pull` and runs it;
  no hand-written pipeline logic lives in Colab cells — cells are limited
  to: mount Drive, clone/pull repo, install requirements, invoke a script,
  inspect output.
- **Git:** as of 2026-10-02 the user authorized Claude Code to commit AND push to
  `origin main` itself (no force-push, never commit videos/Drive data). The repo is
  shared with 3 teammates, so keep commits small and traceable to a module.
- Prefers code delivered in clear, individually runnable pieces/cells where
  relevant, with the "why" behind each step explained, not just code
  dumped with no rationale.
- Avoids buzzwords and unnecessarily complex phrasing in written material.
- Research paper follows Elsevier formatting conventions.
- Use the report's own module names/numbers (M1, M2, M8, M9, ...) in code
  comments, file docstrings, and commit messages, not invented alternative
  names — keeps the codebase traceable to the submitted proposal.
- Each weekly progress entry in academic documentation should carry
  meaningful, non-trivial content — administrative/debugging tasks should
  not stand alone as a primary discussion item for a given week.

## 8. Known pitfalls from earlier work (avoid repeating)

- **Drive path inconsistency across Colab sessions** previously caused
  phantom progress entries and missing outputs. Any progress/checkpoint
  file must be rebuilt from actual disk state when in doubt, never trusted
  blindly as-is.
- **Gemini free-tier has a daily quota** — only relevant if LLM calls are
  ever reintroduced in Phase II. Not needed anywhere in the current M1–M9
  Phase 1 plan.
- **Ultralytics saves `.avi` with mp4v fourcc**, which is not
  browser-decodable. If inline video preview is ever needed in a notebook,
  re-encode to `.mp4` via ffmpeg first.
- **UCF-Crime dataset source matters**: `minhajuddinmeraj/anomalydetectiondatasetucf`
  has real video files; the `odins0n` version does not. (Only relevant if
  UCF-Crime is ever revisited.)
- **Shared-with-me Drive folders are unreliable paths inside Colab** —
  add a shortcut to "My Drive" first, or mounting/listing can silently
  fail or point to the wrong location.

## 9. Tooling stack

- Google Colab with GPU as the batch-processing compute environment
- Google Drive for persistent storage and team handoff of the full dataset
- Local laptop + Claude Code for all actual pipeline code development and
  testing (against the 12-clip local sample)
- GitHub (private repo) as the single source of truth for code, pulled by
  Colab at the start of each session
- YOLOv8 / YOLOv8-Pose (`yolov8n-pose.pt`, conf=0.4) for person detection
  and pose, as specified for M8
- ByteTrack (via Ultralytics' built-in `bytetrack.yaml` tracker config,
  `persist=True`) for multi-object persistent tracking, as specified for M8
- Elsevier journal format for the paper

## 10. Research paper status (as of last check)

- Abstract and Introduction drafted, matching the full two-phase framing
  (Review Report I)
- Results sections are placeholders pending Phase 1 (M1–M9) experiments
- The section-roadmap paragraph at the end of the Introduction is
  intentionally left out until the paper's final structure is settled
- Paper content does not need rescoping — the submitted report already
  correctly separates Phase 1 (M1–M9, no lead-time) from Phase 2 (M10–M13,
  lead-time). Only the team's working *implementation plan* had drifted
  from this; the written proposal was already correct.
