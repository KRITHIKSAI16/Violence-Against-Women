# Checking M9 and the buildup deliverable

M8 checks (tracks, Algorithm 1 scores) are in `docs/VERIFY_M8.md`. This file covers the behavior states, the videos and the report.
Design and file formats: `docs/ARCHITECTURE.md`.

## What you get
| Output | Where (Colab: `VAW_results/`) | What it is |
|---|---|---|
| `gate/<Cat>/<clip>_gate.json` | gate | behavior segments, proposals, key pair, escalation, thresholds used |
| `gate/<Cat>/<clip>_states.csv` | gate | per-frame features and active state of the key pair |
| `gate/proposals.json` | gate | every temporal proposal in the corpus (hand-off to Phase II) |
| `report/index.html` | report | the category table + showcase cards (storyboard, claim, video) |
| `report/categories.csv`, `clips.csv` | report | the numbers behind the page |
| `report/review_sheet.csv` | report | clips for the by-eye check |
| `report/videos/...`, `storyboards/...` | report | the buildup videos and captioned keyframes |

## Check 1: watch a buildup video (2 minutes per clip)
Laptop: `python -m src.report.buildup_video Kidnapping_v28` -> `data/report/videos/Kidnapping/Kidnapping_v28_buildup.mp4`.
Colab: notebook 04, section 5. What you see:
- boxes with track ids; the two people the system talks about have an **orange** box, others green;
- top bar: what is happening now and for how long, e.g. `id4 BLOCKING id5's way out 0.8s` (colors: yellow APPROACH, orange FOLLOW,
  blue HOVER, red CORNER); grey text = nothing flagged at that moment;
- bottom strip: one row per behavior (APPR / FOLL / HOVR / CORN), colored where it is active, white cursor = now, **red line = the cut**;
- after the cut: a dark card "ESCALATION POINT - footage withheld" (or "END OF PRE-VIOLENCE FOOTAGE") and the list of behaviors.
Judge: do the captions match what you see? Do the ids stay on the same people? Does the video stop before the act?
Also open the `_storyboard.jpg`: 2-6 captioned keyframes at the moments each behavior starts, readable without playing the video.

## Check 2: the cut rule (is violence hidden?)
Order of rules: manual override (`configs/clip_overrides.csv`, `clip_id,cut_s`) > detected escalation minus `cut_margin_s`
(any category except Normal) > uniform `act_tail_trim_s` for Assassination / Chain_Snatching / Kidnapping > whole clip.
`clips.csv` lists `cut_s` and `cut_reason` for every clip. **This is a heuristic**: nobody annotated where the act starts.
A bigger `cut_margin_s` is safer but hides more buildup (with 1.5 s the cornering in Kidnapping_v28 was almost entirely cut; the default is 1.0).
Before presenting a clip, watch it once; if the act is visible, set an override for it. Clips with less than `min_video_s` of
pre-violence footage are not rendered (the card says so in the HTML).

## Check 3: recompute the numbers by hand
`python -m src.assm.hand_check_gate Kidnapping_v28` (Colab: `!python -m src.assm.hand_check_gate <clip> --config configs/colab.yaml`).
It recomputes distance, both speeds and the heading cosine at one frame of the key pair from the raw tracks, without using `gate.py`,
and prints OK or MISMATCH next to the saved values (`--frame N` to pick the frame). All should say OK. Sample output:
```
Kidnapping_v28  key pair id4-id5  frame 90 (t=3.00s)  state in file: CORNER
  d        file=   0.381  hand=   0.381  OK   (body heights apart)
  speed_i  file=   0.193  hand=   0.193  OK
```

## Check 4: human review sheet (the independent measurement)
`report/review_sheet.csv` lists ~20 flagged clips (spread over categories, Normal included) and ~10 unflagged non-Normal clips with the
system's claim in words. Play each video and write in `verdict`: `correct`, `partly`, `wrong`; for unflagged clips `missed` if a
buildup is visible, else `ok`. Then: precision by eye = correct / flagged; misses = missed / unflagged. Use two reviewers if you can and
note disagreements. These numbers, not the in-sample rates, are what you report.

## Check 5: calibration (are the thresholds sensible?)
`python -m src.assm.calibrate --config configs/colab.yaml --workers 2`. Look at:
- section 1: flagged share per category vs **Normal** (the false-alarm rate). If Normal is close to the others the gate does not separate;
- section 2: how long behaviors last in Normal vs others (if Normal HOVER lasts as long as others, hover_min_s is too short);
- sections 3-4: each threshold swept one at a time; pick values that keep Normal low (target 15%) with the largest gap.
It uses the Normal label, so it is in-sample tuning. After changing thresholds (`--set` or the config) rerun `gate`, then the report.
Remember `gate --set` overwrites the outputs with those thresholds; run `gate` without `--set` to restore the defaults.

## Check 6: tests
`python -m pytest -q`. Gate tests use synthetic people whose motion is known (follow, side by side, hover, stand together, corner,
blocker behind, escalation, ordered progression, dropout, low-confidence track). Report tests cover the cut rules, rendered video
frame count (footage up to the cut + card, nothing more), captions, showcase selection and the HTML.

## Reading results honestly
- Always compare each category with **Normal**.
- Expect gaps: ~13-28% of clips per category have no trackable pair (small/distant people) and are reported as such, not hidden.
- Known failure types: false boxes, domestic scenes that look like stalking (Normal_v6), hand-held cameras, no scene context.
- Say that thresholds were tuned on the same data and that the review sheet is the independent estimate.
