# Checking the deep context layer, the stories and the evaluation

Design and file formats: `docs/ARCHITECTURE.md` section 10. Older checks: `docs/VERIFY_M8.md` (tracks, Algorithm 1), `docs/VERIFY_M9.md` (the rule gate, now the baseline).
Rule of thumb for everything below: **numbers are only trusted after you have looked at the frames they describe.**

## 0. Fast health check
```
python -m pytest -q                 # 176 tests, synthetic worlds with known answers; needs ffmpeg for the video tests
```
Laptop end-to-end on the 12 sample clips (after M1, M2, M8a):
```
python -m src.context.features --force      # Stage A  -> data/context/<Cat>/<clip>_scene.json + _pairs.npz
python -m src.context.scene --force         # Stage B  -> _layout.json/.npz + _depth.json   (downloads ~120 MB of model weights once)
python -m src.context.story --force --example Kidnapping_v28     # Stage C -> _story.json and the narrative printed
python -m src.report.story_report --set showcase_top=2 showcase_random=1    # -> data/report/index.html
```

## 1. Camera and scene state (`camera.py`)
Open any `_scene.json`: `camera.moving`, `moving_share`, `drift_frac`, `brightness`. Check on a clip you know: a tripod CCTV clip should say `moving: false`; a hand-held phone clip `true`.
Moving cameras make every speed unreliable (the story says so). `night` from Stage A is the mean brightness and is unreliable; the story uses the robust `light.low_light` from Stage B
(median luma and share of dark pixels). Infrared night cameras look bright and will read as not low light.

## 2. Meters (`ground.py`)
Assumptions are stored in each scene json (`fov_deg` 60, `person_height_m` 1.7). Two checks:
* **Scale**: notebook 05 prints the median speed of moving people; walking is about 1.4 m/s. Sample clips gave 1.0-1.5 m/s, the whole set 0.78 (slow and crowded scenes). Far off means the field of view is wrong.
* **Depth cross-check** (Stage B): `_depth.json` -> `depth_agree_frac` = how often Depth Anything and box height agree about who is nearer, on frames where both are decisive. Stalking_v3: 1.00, Normal_v6: 0.83.
Cropped, sitting or far people give wrong depth: they have ground confidence 0.3 and are ignored for speed, following and the mini-map.

## 3. Pose features (`pose_features.py`, `features.py`)
Per pair arrays are in `_pairs.npz` (keys `<i>_<j>__<feature>`), events in `_scene.json -> pairs -> "<i>_<j>"` with `runs` (start, end seconds). To check a behavior, open the processed clip at those seconds:
* `contact` / `reach_*`: two hands-on-body clips were checked (Kidnapping_v41 at 3.2 s: a man grabs a woman; Assassination_v2 near 4.8 s). Contact needs wrist-to-torso below 0.25 m AND the two within 1.2 m on the ground plane.
* `looking_back_*`: head turned toward the other person while the body faces away; needs a visible face while the back is turned. Rare (1-4% of clips).
* `follow_*`: the follower is where the leader was 0.5-6 s ago, the leader walked at least 2 m, the follower moved, and for at least 3 s. Two people walking side by side, or one standing still, are not following.
* `approach_behind_*`: moving toward the other at more than 0.3 m/s while the other's body faces away (>120 degrees).
Wrong detections we found and fixed by looking: cropped boxes producing fake 4-9 m/s "running away"; a head-only box at the frame bottom placed 2 m from another person on the map.

## 4. Scene layout (`scene.py`)
`_layout.json`: `place_type`, `layout_conf`, `reliable`, `doors`, `fractions`, `light`. Notebook 07 section 3 draws the label map of any clip; compare it with the frame. Do not trust it when `reliable` is false
(low confidence or moving camera): dark/unusual scenes fail (a night alley came out as 'wall 68%, floor 31%'). On the 12 samples most place types were 'unknown' and no doors were found: the heuristics are
conservative on purpose. Doors only matter for `blocks_exit`; pinned-against needs a reliable layout, depth samples on two consecutive frames and the other person within 1.5 m.

## 5. The story (`graph.py`, `narrative.py`, `story.py`)
`_story.json` holds, for the two pairs with the most evidence: episodes (predicate, actor, target, start/end, confidence, details), the weighted cue curve, phases and whether they escalate in order, the first
physical-act cue (`escalation`), and scene facts. `python -m src.context.story --example <clip>` prints it as sentences. Check each sentence against the video: right people? right time? plausible meters?
Predicates: approaches_from_behind / side / frontal, follows, looks_back, hovers_near, very_close, mutual_facing (benign), blocks_exit, pinned_against, reaches_for, contact, flees_from, target_stops, target_speeds_up.
Act cues (reach, contact, flee) define where the footage is cut. Low-confidence episodes are marked and count less in the cue curve. The cue weights in `configs/*.yaml -> story.cue_weights` are untuned.

## 6. The story video and cut (`report/story_video.py`)
Watch `data/report/videos/<Cat>/<clip>_story.mp4` (or the Colab report folder). Header = what is happening now; orange boxes = the key pair; yellow outlines = doors (reliable layout only); mini-map = bird's-eye in meters
with rings at 0.46 / 1.2 / 3.7 m around the target; strip rows APPR FOLL LOOK LING BLOK ACT TGT CONV; red line = cut; dark card = withheld part.
Ask: does it stop before the violence? The rule: first act cue minus 1 s (`report.cut_margin_s`), else for Assassination / Chain_Snatching / Kidnapping a uniform 3 s tail trim. If a clip shows too much or too little, add
`clip_id,cut_s` to `configs/clip_overrides.csv` and render again. Clips with less than 1.5 s before the act are not rendered.

## 7. Evaluation (`learn.py`) and how to read it
`python -m src.context.learn --rebuild` (notebook 06). Read in this order:
1. `style_only` vs `behavior` among clips with a pair: if style is high (0.99 here), the dataset has a source shortcut; behavior results are then only meaningful on the static-camera column and the style-matched line.
2. Per-category AUC vs Normal near 0.5 means that category is not distinguishable from Normal by these features.
3. Permutation importance: which features the model leaned on (for style, usually original resolution / fps).
4. `AUC all` counts clips without a pair as 0, so it is diluted by coverage; `AUC pairs` isolates discrimination.
5. Thresholds are Youden (balanced) and chosen on the same scores: optimistic. F1 must be compared with `f1_all_positive`.
The labels are weak (category, not per-second); an AUC near 0.5 does not prove the detectors are wrong. That is what the benchmark measures.

## 8. Annotation benchmark (`report/benchmark.py`) - the measurement that matters
```
python -m src.report.benchmark make --per-category 10        # 4 sheets, ~60 clips, instructions next to them
python -m src.report.benchmark evaluate                       # after the sheets are filled
```
Each annotator watches the processed clip WITHOUT the system's output and fills, per behavior, 1 (happens before the physical violence) or 0, plus `act_start_s`. Output: per behavior the number annotated, detected, precision,
recall, F1, the average pairwise Cohen's kappa (0.4-0.6 moderate, 0.6-0.8 substantial), the false-positive and missed clip ids to look at, and for the act cue: median error in seconds (negative = cut earlier than the real act, safe),
mean absolute error, and the share detected at or before the true start. Ties between annotators are excluded from precision/recall and counted. Use these numbers, not the category rates, when you describe detector quality.

## 9. Things that look wrong but are expected
* Normal clips with cues (people walking up behind each other, standing near each other): ordinary behavior; the page compares every category with Normal and shows conversation as a benign cue.
* "no pair" for 36-45% of clips: small or distant people, or a single person.
* Layout 'unknown' and no doors: conservative heuristics; blocking by a door is rare in this data.
* Few or no frames shown for some act clips: the first act cue came early (e.g. a news clip that starts with the incident).
