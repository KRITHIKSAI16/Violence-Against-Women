# Architecture: how the Phase 1 pipeline works, file by file

Read `PROJECT_BRIEF.md` first (what and why), `STATE.md` (what is done), then this file (how it is built).
Module names (M1, M2, M8, M9) are the ones from Review Report I.

## 1. Data flow

```
data/sample/<Cat>/*.mp4  (laptop, 12 clips)          Drive: ExtrAnom/<Cat>/*.mp4  (Colab, 793 clips)
        |                                                    |
        v   M1  src/data/manifest.py                         |  same code, only the config differs:
   manifest.json            ------------------------------>  |  configs/extran.yaml (laptop)  vs  configs/colab.yaml (Drive paths)
        v   M2  src/data/preprocess.py   (ffmpeg: 30 fps, longest side 640, aspect kept)
   processed/<Cat>/<clip>.mp4 + manifest_clean.json
        v   M8a src/assm/track_poses.py  (YOLOv8n-pose + ByteTrack + tracklet stitching; GPU on Colab)
   tracks/<Cat>/<clip>.npz                      <-- the expensive step, cached; everything below is cheap CPU
        |-----------------------------+
        v   M8b src/assm/interaction.py          v   M9 src/assm/gate.py
   scores/<Cat>/<clip>_pairs.csv,            gate/<Cat>/<clip>_gate.json, _states.csv, gate/proposals.json
        _curve.csv  (Algorithm 1 score)             |  (behavior states, proposals)
                                                    v   src/assm/calibrate.py   (threshold sweep, prints only)
                                                    v   src/report/buildup_video.py  +  buildup_report.py
                                              report/index.html, categories.csv, clips.csv, review_sheet.csv,
                                              videos/<Cat>/<clip>_buildup.mp4, storyboards/<Cat>/<clip>_storyboard.jpg
```

Why M8 is split: YOLO/ByteTrack is the slow part (GPU). Once tracks are cached, weights, thresholds and behavior
rules can be changed and rerun in seconds-to-minutes on CPU. The cached tracks are also exactly what Phase II needs.

## 2. Modules, inputs, outputs

| Module | Command | Reads | Writes |
|---|---|---|---|
| M1 | `python -m src.data.manifest` | `data_root/<Category>/*` | `manifest.json` |
| M2 | `python -m src.data.preprocess` | `manifest.json` | `processed/`, `manifest_clean.json` |
| M8a | `python -m src.assm.track_poses` | `manifest_clean.json`, processed clips | `tracks/<Cat>/<clip>.npz` |
| M8b | `python -m src.assm.interaction` | tracks | `scores/<Cat>/<clip>_pairs.csv`, `_curve.csv` |
| M9 | `python -m src.assm.gate` | tracks | `gate/<Cat>/<clip>_gate.json`, `_states.csv`, `gate/proposals.json` |
| calibrate | `python -m src.assm.calibrate` | tracks | prints; `gate/calibration_sweep.csv` |
| report | `python -m src.report.buildup_report` | gate json, tracks, processed clips | `report/*` |
| one clip | `python -m src.report.buildup_video <clip_id>` | same | its video + storyboard |
| checks | `render`, `hand_check`, `hand_check_gate` | | see `docs/VERIFY_M8.md`, `docs/VERIFY_M9.md` |

All commands accept `--config configs/colab.yaml`. `gate`, `calibrate`, `buildup_report` accept `--set key=value ...`.

## 3. File formats

**manifest.json / manifest_clean.json**: `{source, data_root, num_clips, counts_per_category, clips[], skipped[]}`.
Clip: `clip_id` (filename stem, unique across the corpus), `category` (folder name = native label), `label`
(`normal` for Normal, `pre_violence` otherwise), `source`, `path`, `duration_s, fps, frame_count, width, height`
(+ `codec` when read via ffprobe). The clean manifest adds `stage: "M2"`, `settings`, and per clip `raw_path` (original) with
`path` pointing at the processed clip. `skipped[]` = `{path, reason}`.

**tracks/<Cat>/<clip>.npz** (one row per tracked person per frame; frames with nobody have no rows):
`frame_idx (N) int32`, `track_id (N) int32` (identity after stitching), `raw_track_id (N)` (as ByteTrack gave it),
`bbox (N,4) x1,y1,x2,y2 px`, `conf (N)`, `kpts (N,17,3)` COCO keypoints x, y, confidence, and scalars `fps, width, height, n_frames`.
Coordinates are in the processed (M2) clip's pixels.

**scores/..._pairs.csv**: `frame,time_s,id_i,id_j,d,v,b,score` one row per frame and pair present in both t and t-1
(Algorithm 1). `d` body heights, `v` closing speed (+ approaching), `b` path obstruction 0/1, `score = w1/max(d,d_floor)+w2*v+w3*b`.
**..._curve.csv**: `frame,time_s,n_people,n_pairs,max_score,top_i,top_j` (max over pairs per frame; 0 if no pair).

**gate/..._gate.json**: `clip_id, category, fps, n_frames, duration_s, n_tracks, n_tracks_dropped_low_conf, n_pairs,
no_interaction, key_pair [id,id]|null, flag, states_seen[], ordered_progression, phase_sequence[], escalation
{frame,time_s,pair}|null, proposals[], segments[], params{thresholds used}, max_people, isolated_frac`.
- `segments[]`: `{state, id_i, id_j, start_f, end_f, start_s, end_s, dur_s, actor, target|..}` where `state` is one of
  APPROACH, FOLLOW, HOVER, CORNER, ESCALATION. `actor` = the one doing it (follower / hoverer / blocker / approacher);
  `target` = the other.
- `proposals[]`: `{start_f, end_f, start_s, end_s, pair, reasons[], ends_in_escalation}`: temporal windows that need deeper
  analysis. This is M9's output for Phase II (M10); `gate/proposals.json` is all of them across the corpus.
- `escalation`: earliest burst by anyone connected to the key pair (or anyone if no key pair). The video cut uses it.
**..._states.csv**: per frame of the key pair: `frame,time_s,d,closing,speed_i,speed_j,cos,block,state`.

**report/clips.csv**, **categories.csv**, **review_sheet.csv**: see `src/report/buildup_report.py` docstring; `index.html` is static
and self-contained (videos and images are relative links, so keep the folder together).

## 4. The behavior states (M9), exactly

Units: body height (bh) = mean bbox height of the two people; speeds bh/s; times s. Positions = pose centroid (mean of keypoints
with confidence > `kpt_conf`, else box centre), smoothed with a trailing mean (`smooth_s`). Velocities over `vel_s`. Track gaps up to
`gap_fill_s` are interpolated; gaps up to `gap_s` inside a state are bridged. A state only counts after its minimum duration.

| State | Rule | Meaning |
|---|---|---|
| APPROACH | closing speed > `approach_closing`, d < `approach_max_d`, net drop >= `approach_min_drop`, >= `approach_min_s` | one moves toward the other |
| FOLLOW | d <= `follow_max_d`, both moving (>= `move_speed`), heading cosine >= `follow_cos`, the other moves away along the line between them (`follow_behind`), >= `follow_min_s` | trailing someone |
| HOVER | d <= `hover_max_d`, exactly one standing still (< `still_speed`) while the other moves, >= `hover_min_s` | lingering next to a still person |
| CORNER | blocker between the blocked person and the nearest frame edge (b_ij, corridor `corridor` bh), d <= `corner_max_d`, blocked person slow (< `corner_blocked_speed`), blocker moved in within `corner_activity_s`, >= `corner_min_s` | standing in their way out |
| ESCALATION | relative speed (short window) > max(`burst_abs`, median + `burst_mad_k` robust deviations of the pair), d <= `burst_max_d`, >= `burst_min_s` | sudden burst while close |

Flagging: FOLLOW, HOVER or CORNER, or an APPROACH followed by ESCALATION. A pure APPROACH never flags a clip.
`ordered_progression`: the phases of the key pair started in non-decreasing rank (APPROACH 1 < FOLLOW = HOVER 2 < CORNER 3 < ESCALATION 4)
and at least two ranks occurred. Tracks whose mean detection confidence is below `min_track_conf` are ignored.
Contact (box overlap) is NOT used for ESCALATION: perspective overlap makes it unreliable.

## 5. Design decisions and rejected alternatives

- **Body-height units, not pixels or frame fractions**: "close" should mean the same for near and far cameras. A frame-diagonal
  normalisation was the first idea (in the plan) and was replaced by body heights by reasoning; the two were not compared experimentally.
- **States, not one score**: the Algorithm 1 score mixes closeness, approach and blocking, so it cannot say what happens; on the full set
  Normal clips had the highest mean (1.80). Rejected: tuning the weights w1..w3 (cannot fix the missing "what"). M8b is kept as the
  report's Algorithm 1 and for Phase II.
- **Nearest frame edge as the "exit" for b_ij**: ExtrAnom has no exit map. Weight configurable (`w3`). Also reused by CORNER.
- **ByteTrack run at low conf (0.1) with its own thresholds, long buffer, plus stitching**: with `conf=0.4` the low-confidence rescue pass
  never ran and ids flipped. Stitching joins a track that ended to one that starts nearby soon after (no time overlap).
- **No LLM / annotation layer**: not in the report's M1-M9; the interpretable states are the context. Do not add `src/annotate/`.
- **Video cut before the escalation, not after**: the user does not want to show violence. Without annotations the cut is a heuristic
  (margin `cut_margin_s`, uniform `act_tail_trim_s`, per-clip overrides CSV).
- **Showcase = top-evidence + random**, so the report is not cherry-picked, and a human review sheet gives an independent precision estimate.
- **Calibration is in-sample** (uses the Normal label to pick thresholds). Stated everywhere it is shown.

## 6. Known limits (say them when presenting)
2D pixel geometry only (camera angle matters; hand-held cameras corrupt velocities); small/distant people missed; false boxes;
domestic scenes look like stalking to geometry (Normal_v6: a man walking up to a seated woman at home flags HOVER); the act's start
is not annotated; 13-28% of clips per category have no trackable pair (Colab M8b run); Normal clips differ in style from the others.

## 7. Phase II hand-off (M10-M13)
- `tracks/*.npz`: ByteTrack identities + poses = nodes of the interaction graph (M10). Depth Anything later only replaces `d` on
  gated frames; no re-tracking needed.
- `gate/proposals.json` + `segments[]`: first estimate of segment timing and per-horizon label candidates; states are ready-made
  predicates (APPROACH, FOLLOW, HOVER, CORNER) for the plausibility grammar (cornering preceded by following, M12); `ordered_progression`
  measures it.
- `isolated_frac` (share of frames with exactly 2 people) is the first piece of scene context; time of day and location type are Phase II.

## 8. Config blocks (`configs/extran.yaml`, mirrored in `configs/colab.yaml`)
`preprocess` (M2), `assm` (M8a/M8b: model, tracker, conf, stitching, weights, paths), `gate` (M9 thresholds above + `gate_dir`),
`report` (cut rules, showcase sizes, review sheet sizes, `report_dir`). Colab uses absolute Drive paths; relative paths resolve against the repo root.

## 9. Tests
`python -m pytest -q`: synthetic people with known motion (approach, follow, side-by-side, hover, corner, escalation, dropouts),
file round trips, calibration, cut rules, rendered video frame counts, report numbers. No dataset or GPU needed; ffmpeg needed for the
video/preprocess tests (skipped if missing).

## 10. Deep context layer (`src/context/`, in progress)

Motivation: the M9 rule gate (section 4) uses only 2D box-centre distance and speed. On the full set it flagged Normal clips more often than the buildup
categories (27% vs 15%) and FOLLOW almost never fired. The deep layer uses what was already cached but unused (the 17 keypoints) plus real-world geometry.
Still no prediction, no LLM, no identity / gender inference.

| Module | What | Idea / source |
|---|---|---|
| `camera.py` | per-frame camera motion (sparse optical flow + RANSAC similarity), cumulative transform to frame 0, `moving`, brightness (`night`) | same idea as BoT-SORT camera-motion compensation; person boxes are masked out of the corner detection |
| `ground.py` | ground plane (X lateral, Z depth) in meters: Z = f*1.7/h_px, X = (cx - W/2)*1.7/h_px, f from an assumed field of view (`context.fov_deg`) | monocular pedestrian metrology; Hall proxemic zones 0.46 / 1.2 / 3.7 / 7.6 m; `conf` 0.3 for cropped or sitting boxes |
| `pose_features.py` | body facing (shoulder line + nose offset), head facing (nose/eyes vs ears), wrist-to-torso contact distance (m), reach speed (m/s) | AAAI 2024 hidden-follower work (gaze + spacing); NTU RGB+D pairwise-joint features |
| `follow.py` | lagged-path following: follower(t) vs leader(t - tau), tau 0.5-6 s; must beat the present separation; the leader must really walk | Li et al., ICDM 2013, "Mining following relationships in movement data" |
| `features.py` | orchestrator: per-pair arrays (`dist_m`, `closing_ms`, `zone`, `ang_i_to_j`, `follow_i_j`, `approach_behind_*`, `looking_back_*`, `mutual_facing`, `contact`, `reach_*`, `flee_*`) + first events + scene facts | writes `context/<Cat>/<clip>_pairs.npz` (keys `<i>_<j>__<feature>`) and `_scene.json` |

Coordinates: X to the right, Z away from the camera; facing vectors use the same frame ("faces the camera" = (0,-1)). Angles are 0 deg when a person faces the other.
Rules that came from looking at frames: speed-derived features (speed, flee, following) use only upright, uncropped people (`min_ground_conf`), because a box cut by the frame
edge or a person leaning over a table gives a wrong depth and fake 4-9 m/s speeds. Reach and contact were checked by eye on two clips. The metric scale was checked
(median moving speed 1.0-1.5 m/s). Assumptions (FOV, person height) are stored in every scene json. Stages B-E (scene segmentation, Depth Anything, interaction graph + narrative,
learned window scorer with a style-confound probe, annotation benchmark) follow; see STATE.md.
