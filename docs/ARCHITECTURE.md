# Architecture: how the Phase 1 pipeline works, file by file

Read `PROJECT_BRIEF.md` first (what and why), `STATE.md` (what is done), `docs/STATUS_REPORT.md` (results and decisions in plain language), then this file (how it is built).
Module names (M1, M2, M8, M9) are the ones from Review Report I.

## 1. Data flow

```
data/sample/<Cat>/*.mp4  (laptop, 12 clips)          Drive: ExtrAnom/<Cat>/*.mp4  (Colab, 793 clips)
        |                                                    |   same code, only the config differs:
        v   M1  src/data/manifest.py                         |   configs/extran.yaml (laptop)  vs  configs/colab.yaml (Drive paths)
   manifest.json
        v   M2  src/data/preprocess.py   (ffmpeg: 30 fps, longest side 640, aspect kept)
   processed/<Cat>/<clip>.mp4 + manifest_clean.json
        v   M8a src/assm/track_poses.py  (YOLOv8n-pose + ByteTrack + tracklet stitching; GPU)         <- the expensive step, cached
   tracks/<Cat>/<clip>.npz
        |--------------------------------+---------------------------------------------+
        v  baseline (kept)               v  DEEP CONTEXT LAYER  src/context/            |
  M8b  src/assm/interaction.py     Stage A  features.py  (camera, meters, pose, following)  -> context/<Cat>/<clip>_scene.json, _pairs.npz
       -> scores/*_pairs.csv,      Stage B  scene.py     (layout + key-pair depth; GPU)     -> _layout.json/.npz, _depth.json
          _curve.csv               Stage C  story.py     (scene graph episodes + narrative) -> _story.json
  M9   src/assm/gate.py            Stage D  learn.py     (evaluation with shortcut probes)  -> learn/results.json, windows.csv, clips.csv
       -> gate/*  (rule gate)      Benchmark report/benchmark.py (annotation sheets, detector precision/recall vs humans)
        |                                v
        |                          report/story_report.py + story_video.py
        |                          -> report/index.html, categories_story.csv, clips_story.csv, videos/<Cat>/<clip>_story.mp4, storyboards/
        v  (older report: report/buildup_report.py, buildup_video.py, kept as the baseline deliverable)
```

Why M8 is split: YOLO/ByteTrack is the slow part (GPU). Once tracks are cached, thresholds and behavior rules change in seconds-to-minutes on CPU, and the cache is Phase II's input.
Why a separate deep layer: the baseline (M8b score, M9 rule gate) did not separate the categories and ignored the 17 pose keypoints; see `docs/STATUS_REPORT.md` section 5.

## 2. Modules, commands, inputs, outputs

All commands accept `--config configs/colab.yaml`. Most accept `--clips A B`, `--force`, and several accept `--set key=value ...` (this run only).

| Module | Command | Reads | Writes |
|---|---|---|---|
| M1 | `python -m src.data.manifest` | `data_root/<Category>/*` | `manifest.json` |
| M2 | `python -m src.data.preprocess` | `manifest.json` | `processed/`, `manifest_clean.json` |
| M8a | `python -m src.assm.track_poses` | `manifest_clean.json`, processed clips | `tracks/<Cat>/<clip>.npz` |
| M8b | `python -m src.assm.interaction` | tracks | `scores/<Cat>/<clip>_pairs.csv`, `_curve.csv` |
| M9 rule gate | `python -m src.assm.gate` | tracks | `gate/<Cat>/<clip>_gate.json`, `_states.csv`, `gate/proposals.json` |
| gate calibration | `python -m src.assm.calibrate` | tracks | prints; `gate/calibration_sweep.csv` |
| Stage A | `python -m src.context.features` | tracks, processed clips | `context/<Cat>/<clip>_scene.json`, `_pairs.npz` |
| Stage B | `python -m src.context.scene` | context, tracks, processed clips | `_layout.json`, `_layout.npz`, `_depth.json` |
| Stage C | `python -m src.context.story` | context (+ layout, depth) | `_story.json` |
| Stage D | `python -m src.context.learn` | context, gate (baseline) | `learn/results.json`, `windows.csv`, `clips.csv`, `clip_scores_behavior.csv` |
| report | `python -m src.report.story_report` | stories, learn results, benchmark results | `report/index.html`, `categories_story.csv`, `clips_story.csv`, videos, storyboards |
| one clip | `python -m src.report.story_video <clip_id>` | same | its story video + storyboard |
| benchmark | `python -m src.report.benchmark make / evaluate` | stories, filled sheets | `report/benchmark/*` |
| checks | `src.assm.render`, `hand_check`, `hand_check_gate` | | see `docs/VERIFY_M8.md`, `VERIFY_M9.md`, `VERIFY_CONTEXT.md` |

## 3. File formats

**manifest.json / manifest_clean.json**: `{source, data_root, num_clips, counts_per_category, clips[], skipped[]}`. Clip: `clip_id` (filename stem, unique across the corpus), `category` (folder = native label),
`label` (`normal` / `pre_violence`), `source`, `path`, `duration_s, fps, frame_count, width, height` (+ `codec` when read via ffprobe). The clean manifest adds `stage: "M2"`, `settings`, `raw_path` (original) and `path` (processed clip).

**tracks/<Cat>/<clip>.npz** (one row per tracked person per frame): `frame_idx`, `track_id` (after stitching), `raw_track_id` (as ByteTrack gave it), `bbox (N,4)` x1,y1,x2,y2 px, `conf`, `kpts (N,17,3)` COCO keypoints
(x, y, confidence), scalars `fps, width, height, n_frames`. Pixels of the processed clip.

**scores/..._pairs.csv / _curve.csv** (M8b baseline): `frame,time_s,id_i,id_j,d,v,b,score`; per frame `max_score` curve. **gate/..._gate.json / _states.csv / proposals.json** (M9 rule gate): segments of APPROACH, FOLLOW, HOVER, CORNER,
ESCALATION per pair, `proposals[]`, `flag`, `params` (thresholds used). Section 5 describes the rules.

**context/<Cat>/<clip>_scene.json** (Stage A): `fps, n_frames, duration_s, camera{moving, moving_share, drift_frac, brightness, night, sharpness, flow_ok_frac}, n_tracks, n_tracks_used, n_pairs, no_pair, max_people, mean_people,
isolated_frac, assumptions{fov_deg, person_height_m}, pairs{"i_j": events}, clip_events`. `events` = runs per behavior (`count, seconds, first_s, runs[[start,end],...]`) plus `min_dist_m`, `intimate_or_personal_s`.
**..._pairs.npz**: keys `"<i>_<j>__<feature>"`, arrays over all frames (NaN where not co-tracked): `dist_m, closing_ms (+ approaching), zone (0 intimate .. 4 far, Hall), speed_i/j (m/s), ang_i_to_j / ang_j_to_i (0 deg = faces the other),
head_ang_i/j, toward_i/j, wrist_torso_m, reach_i_to_j_ms / reach_j_to_i_ms, ground_conf, approach_behind_i_j / _j_i, looking_back_i/j, mutual_facing, contact, flee_i/j, still_i/j, follow_i_j / _j_i, lag_i_j / _j_i, dev_i_j / _j_i`.
Ground-plane frame: X right, Z away from the camera, meters; facing vectors use the same frame.

**..._layout.json / .npz** (Stage B): `fractions{walkable, obstacle, door, vehicle, sky, vegetation, furniture, other}, road_like_frac, place_type (indoor/outdoor/unknown), doors[{x1,y1,x2,y2,cx,base_y,area}] (normalised), walkable_frac,
layout_conf, light{median_luma, dark_frac, low_light}, reliable`; the npz holds a 128x128 group-label map in normalised image coordinates. **..._depth.json**: `pair, frames[{t, frame, depth_i, depth_j, depth_gap, height_gap,
nearer_by_depth, nearer_by_height, pinned_i, pinned_j}], depth_agree_frac, n_decisive` for the single most active pair (relative depth: only orderings and small gaps are used).

**..._story.json** (Stage C): `clip_id, category, fps, n_frames, scene{place_type, layout_reliable, layout_conf, doors, low_light, camera_moving, max_people, isolated_frac, n_pairs, duration_s, depth_agree_frac, assumptions},
key_pair, no_pair, pairs{"i_j": {pair, episodes[], concern_seconds, benign_seconds, cue_sum, phases, ordered_progression, cue_curve[[t, value]], min_dist_m}}, escalation{time_s, kind, pair, conf}|null,
concern_seconds, benign_seconds, preds_seen[], weights`. Episode: `{pred, actor, target, start_f, end_f, start_s, end_s, conf, detail{...}}`.

**learn/results.json**: `n_clips, n_clips_with_pair, n_windows, label_counts, models{"gb:style_only", "gb:behavior", ..., "logreg:*": {all_clips, clips_with_pair, static_camera_clips, per_category_auc_vs_normal}}, importance, overlap,
rule_gate`. **report/benchmark/benchmark_sheet_<annotator>.csv**: `clip_id, category, video, annotator`, one column per behavior (1/0), `buildup_visible_0_to_2, act_start_s, notes`; **benchmark_results.json** after `evaluate`.

## 4. The deep context layer (`src/context/`)

| Module | What | Idea / source |
|---|---|---|
| `camera.py` | per-frame camera motion (sparse optical flow + RANSAC similarity), cumulative transform, `moving`, brightness | same idea as BoT-SORT camera-motion compensation; person boxes are masked out of the corner detection |
| `ground.py` | ground plane in meters: Z = f*1.7/h_px, X = (cx - W/2)*1.7/h_px, f from an assumed field of view; Hall zones 0.46 / 1.2 / 3.7 / 7.6 m | monocular pedestrian metrology; `conf` 0.3 for cropped or sitting boxes |
| `pose_features.py` | body facing (shoulder line + nose offset), head facing (nose/eyes vs ears), wrist-to-torso contact distance (m), reach speed (m/s) | AAAI 2024 hidden-follower work (gaze + spacing); NTU RGB+D pairwise-joint features |
| `follow.py` | lagged-path following: follower(t) vs leader(t - tau), tau 0.5-6 s; must beat the present separation; the leader must walk | Li et al., ICDM 2013 |
| `features.py` | orchestrator: per-pair arrays + first events + scene facts; `rank_pairs` | |
| `scene.py` | SegFormer-B0 layout (walkable / obstacle / door / vehicle / sky / vegetation / furniture), place type, doors, robust light statistic; Depth Anything V2 Small on a few frames of the key pair: nearer person, same depth plane, pinned against an obstacle | report Algorithm 2 (DAIR) used early; layouts carry a confidence and are ignored when unreliable |
| `graph.py` | scene graph episodes (actor -> predicate -> target) with confidence and details; cue curve; phases; first physical-act cue | HIG-style interaction graph in a Phase-1 form |
| `narrative.py` | deterministic sentence templates for episodes, scene line, summary | no language model |
| `story.py` | runs graph + narrative for every clip | |
| `windows.py`, `learn.py` | window-level features (behavior / scene / style groups), grouped cross-validation, shortcut probes, importance, comparison with the rule gate | Sultani et al. style multiple-instance scoring |

Predicates: approaches_from_behind / approaches_side / approaches_frontal, follows, looks_back, hovers_near, very_close, mutual_facing (benign), blocks_exit (needs a reliable layout and a door), pinned_against (reliable layout + depth
on two consecutive samples + the other person within 1.5 m), reaches_for / contact / flees_from (ACT cues), target_stops / target_speeds_up. Act cues are depth-confirmed ("same depth") or discounted ("different depth").
The cue curve uses `story.cue_weights` (untuned defaults, benign negative); it is for visual emphasis and ranking inside the report, not a validated score.

Rules that came from looking at frames (keep them): speed-derived features (speed, flee, following) and the mini-map use only upright, uncropped people (`context.min_ground_conf`) because a box cut by the frame edge gives a wrong
depth and fake 4-9 m/s speeds; the mean-brightness night flag is unreliable (a lamp-lit night street read as daylight), so the story uses the median / dark-pixel statistic from Stage B; the F1-maximising threshold collapses to
"everything positive" when 63% of clips are positive, so evaluation uses Youden / balanced accuracy and prints the all-positive F1.

## 5. The M9 rule gate (baseline), exactly

Units: body height (bh) = mean bbox height of the two people; speeds bh/s; times s. Positions = pose centroid smoothed (`smooth_s`). A state only counts after its minimum duration; gaps up to `gap_s` are bridged.

| State | Rule |
|---|---|
| APPROACH | closing speed > `approach_closing`, d < `approach_max_d`, net drop >= `approach_min_drop`, >= `approach_min_s` |
| FOLLOW | d <= `follow_max_d`, both moving, heading cosine >= `follow_cos`, the other moves away along the line between them, >= `follow_min_s` |
| HOVER | d <= `hover_max_d`, exactly one standing still while the other moves, >= `hover_min_s` |
| CORNER | blocker between the blocked person and the nearest frame edge, d <= `corner_max_d`, blocked person slow, blocker moved in recently, >= `corner_min_s` |
| ESCALATION | relative speed burst (short window) while close |

Flagging: FOLLOW, HOVER or CORNER, or an APPROACH followed by ESCALATION. Result on the full set: Normal flagged 27% vs 15% for the other categories, FOLLOW in 2 clips; it is kept as the baseline that the evaluation compares against.

## 6. Design decisions and rejected alternatives

- **Body-height units first, meters later**: "close" should mean the same for near and far cameras; a frame-diagonal normalisation was dropped by reasoning, not by experiment. Meters (standing-height prior) came with the deep layer.
- **States/episodes, not one score**: the Algorithm 1 score mixes closeness, approach and blocking and was highest for Normal (1.80 vs 0.94 Stalking). M8b is kept as the report's Algorithm 1 and as a baseline.
- **Nearest frame edge as the exit** (baseline): ExtrAnom has no exit map. The deep layer replaces it with detected doors when the layout is reliable.
- **ByteTrack at low conf (0.1), long buffer, plus stitching**: with `conf=0.4` the low-confidence rescue pass never ran and ids flipped.
- **No LLM / annotation layer**: not in the report's scope (the report says no natural-language explanation); text comes from templates. A vision-language cross-check was considered only as an optional offline evaluation aid and is not built.
- **Video cut before the first act cue**: you asked not to show violence; without annotation of the act start the cut is a heuristic (margin, uniform trim, per-clip overrides); the benchmark measures its error.
- **Showcase = top-evidence + random clips** so the page is not cherry-picked; compare every category with Normal.
- **Evaluation by clip with a style-only probe**: Normal is a different source (resolution / frame rate identify it with AUC 0.99), so a naive classifier would "work" for the wrong reason.
- **Depth Anything and SegFormer only where they add something**: key pair, few frames; layouts used only when confident and the camera is still.
- **No identity, gender or age inference.**

## 7. Known limits (say them when presenting)
2D views; approximate meters (assumed 60 degree field of view, accuracy not measured; the only check is typical walking speed); small / distant people are missed (36-45% of clips per category have no usable pair);
false boxes; layouts fail on dark scenes; infrared night cameras look bright (low light means a dim image); hand-held cameras corrupt speeds (36% of clips); the act start is not annotated; the labels are the clip's category,
not per-second annotations; Normal clips are a different source; detector precision/recall is not measured until the benchmark is annotated.

## 8. Phase II hand-off (M10-M13)
- `tracks/*.npz` and `context/*_pairs.npz`: identities, poses, meters = the nodes and edge features of the M10 interaction graph; depth (Algorithm 2) is already computed for the key pair.
- `_story.json` episodes: ready-made predicate sequences with times, roles and confidence for the plausibility grammar (M12: cornering preceded by following); `ordered_progression` measures it. Act cues give a first estimate of segment timing for horizon labels.
- Scene facts (`place_type`, `low_light`, `isolated_frac`, camera) are the first context metadata the report asks for.
- The benchmark annotations (behaviors with `act_start_s`) are the first ground truth for horizon labels.
- `learn.py` results are the Phase I baseline that the Phase II model must beat; `windows.csv` is the feature table.

## 9. Config blocks (`configs/extran.yaml`, mirrored in `configs/colab.yaml`)
`preprocess` (M2), `assm` (M8a/M8b: model, tracker, conf, stitching, weights, paths), `gate` (rule-gate thresholds), `context` (field of view, person height, smoothing, following, contact, facing, flee, layout keyframes, depth frames),
`story` (episode thresholds and `cue_weights`), `learn` (windows, folds, top-k, max pairs per clip), `report` (cut rules, showcase sizes, `report_dir`). Colab uses absolute Drive paths; relative paths resolve against the repo root.

## 10. Tests
`python -m pytest -q` (178 tests): synthetic people with known motion for every behavior (approach from behind vs frontal, following with lag vs side by side, hovering, corner, door blocking, pinned against, contact, reach,
fleeing, cropped boxes, camera pan), camera estimation on textured synthetic video, ground-plane meters, scene logic (layout facts, doors, light, depth helpers), graph episodes and narrative text, story video frame counts
(footage up to the cut + card, nothing after), report numbers and HTML, benchmark metrics (kappa, ties, cut accuracy), the evaluation including **the shortcut probe** (identical behavior, different filming => style AUC ~1, behavior AUC ~0.5).
No dataset or GPU needed; ffmpeg is needed for video tests (skipped if missing); one SegFormer smoke test runs only if the weights are cached.
