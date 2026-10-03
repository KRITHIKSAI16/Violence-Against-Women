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
No natural-language MODEL is needed or wanted. The context is engineered and interpretable; the sentences shown in the story are produced by
deterministic templates from the detected episodes (nothing is generated that the detectors did not find).

**Update 2026-10-03 - what Phase 1 became (read `docs/STATUS_REPORT.md`).** The simple M8b score and the first M9 rule gate did not separate the
categories from Normal (Normal was flagged MORE often: 27% vs 15%). Phase 1 therefore delivers: (1) the data pipeline M1, M2, M8a; (2) a **deep context layer**
(`src/context/`, an elaboration of M8 + M9): approximate meters, camera motion, body-pose facing / contact / reach, lagged-path following, scene layout, depth for the key pair, an interaction
scene graph and a deterministic narrative per clip; (3) a **story video** per clip that stops before the first physical-act cue; (4) a **controlled evaluation** (the report's M4-M7 in a form that
detects dataset shortcuts) and an **annotated benchmark** (about 60 clips, the team) that measures detector precision / recall. The honest result: behavior features cannot reproduce the category
labels (AUC 0.49 on static-camera clips) and the Normal clips are a different source (original resolution / frame rate identify them with AUC 0.99). So Phase 1 delivers a detailed, checkable
description of each clip and an evaluation of its limits, NOT a buildup classifier. M3 (segment pairing) was folded into the cut-point rule; M4-M7 exist as `windows.py` / `learn.py` plus the benchmark.

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
- A full depth-aware interaction graph with learned edge weights (M10). **Update 2026-10-03:** the team decided to use Depth Anything V2 Small and
  scene segmentation EARLY, in Phase 1, as a context check only (which person is nearer, whether two people are on the same depth plane, pinned
  against a wall, doors as real exits) on the single most active pair of each clip. This is the report's Algorithm 2 used for description, not for
  prediction. Everything that predicts the future stays in Phase II.

## 3. Dataset

### Primary dataset: ExtrAnom
- Video-only dataset (no existing labels beyond folder/category name, no
  timestamps, no metadata files) — this is exactly the kind of input M3
  expects ("cleaned corpus with its native violence/non-violence labels")
- Organized as one folder per category (**793 clips in total, very unbalanced**, counted on the Drive copy on 2026-10-03):
  - `Assassination` (23)
  - `Chain_Snatching` (176)
  - `Harassment` (188)
  - `Kidnapping` (73)
  - `Normal` (294)
  - `Stalking` (39)
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
  Claude Code. The full dataset (793 clips, counts above — an earlier note
  said ~140-150, that was wrong) lives in Google Drive under a "Shared with me"
  folder named `ExtrAnom`, and is processed via Google Colab (GPU), not on the
  laptop. Dataset quirks found on the first Colab run: 3 Kidnapping clips are AV1
  (OpenCV cannot decode them on Colab; M1 falls back to ffprobe and M2 re-encodes
  to H.264); many Normal clips are 60 fps phone-style footage while the other
  categories are mostly CCTV-style, so Normal differs in look as well as behavior.
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

**How M9 is implemented (decided 2026-10-03, see docs/ARCHITECTURE.md):** the single
M8b score cannot say *what* is happening (on the full set Normal clips scored highest, mean 1.80 vs
Stalking 0.94), so M9 turns the same cached tracks into interpretable behavior states per pair —
APPROACH, FOLLOW, HOVER, CORNER, ESCALATION — with who-does-what and durations. These implement the
report's three M9 heuristics (persistence, path blocking, sudden motion). A clip is flagged when FOLLOW, HOVER or CORNER
occurs, or an APPROACH ends in ESCALATION. Output = temporal proposals. **This rule gate was then run on all 793 clips and did not work**
(Normal flagged 27% vs 15% for the other categories; FOLLOW in 2 clips). It is kept as the BASELINE. The final M9 deliverable is the deep context layer
(`src/context/`, scene graph episodes + narrative, see the update in section 2 and `docs/ARCHITECTURE.md`), whose story video shows the context up to just
before the first physical-act cue, never the violence.

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

M3–M7 (segment pairing, feature caching, EDA, baseline classifier, benchmark): partly covered since 2026-10-03 — see the update in section 2: the window feature table and
the cross-validated baseline with shortcut probes (`src/context/windows.py`, `learn.py`) and the annotated benchmark (`src/report/benchmark.py`). A separate M3 segment-pairing module was not needed.

**Final Phase 1 deliverable (2026-10-03):** the story report (`VAW_results/report/index.html`) with per-clip story videos and storyboards, the category table (always compared with Normal), the
evaluation findings, and the annotation benchmark. Run order: Colab notebooks 01 -> 08 (`colab/README.md`).

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
  (If you are a teammate's Claude Code: ask your user before pushing.)
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

- **Colab sessions drop**: M8a (tracks) and M2 are resumable (finished clips are skipped on rerun); never use
  `--force` casually, it recomputes everything.
- **`gate --set key=value` overwrites the gate outputs** with those thresholds (recorded under `params` in each
  gate json). Run `python -m src.assm.gate` again with no `--set` to restore the config defaults.
- **Windows + git-bash**: `/tmp` paths are not visible to Python; write files with the editor tools, not heredocs with
  nested quotes.
- **False boxes**: detectors box chairs/scooters as people and miss small or distant people (13-28% of clips per category have no
  trackable pair). M9 ignores tracks with mean detection confidence below `gate.min_track_conf`; the rest is a known limit.

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
- Results: Phase 1 experiments exist (see `docs/STATUS_REPORT.md` section 5); the key honest findings to report are the failure of the simple score, the source shortcut in the
  Normal class (style-only AUC 0.99), behavior at chance on static-camera clips, and (after annotation) detector precision / recall from the benchmark
- The section-roadmap paragraph at the end of the Introduction is
  intentionally left out until the paper's final structure is settled
- Paper content does not need rescoping — the submitted report already
  correctly separates Phase 1 (M1–M9, no lead-time) from Phase 2 (M10–M13,
  lead-time). Only the team's working *implementation plan* had drifted
  from this; the written proposal was already correct.
