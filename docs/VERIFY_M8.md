# Checking M8 output: what the files mean and how to verify them

## What M8 produces (per clip)
| File | Contents |
|---|---|
| `tracks/<Cat>/<clip>.npz` (M8a) | per frame, per tracked person: track id, box, confidence, 17 keypoints |
| `scores/<Cat>/<clip>_pairs.csv` (M8b) | one row per (frame, pair of ids): `d, v, b, score` |
| `scores/<Cat>/<clip>_curve.csv` (M8b) | one row per frame: people, pairs, `max_score` (the **buildup curve**), top pair |
| `scores/videos/<clip>.mp4` | annotated verification video |

Columns: **d** = distance between the two people in *body heights* (1 = about one body height apart);
**v** = closing speed in body heights/second (+ approaching, - moving apart);
**b** = 1 if one person stands between the other and the nearest frame edge (stand-in "exit"), else 0;
**score** = `1/d + v + b` with the starting weights (w1=w2=w3=1, `d` floored at 0.25).

## Check 1: does the video agree with what you see? (2 minutes per clip)
Laptop: `python -m src.assm.render Stalking_v3 --tau 3.0` -> open `data/scores/videos/Stalking_v3.mp4`.
Colab: notebook 03, section 3. Look for:
- every real person has a green box with a stable `id` (does the id stay on the same person? ids that flip = tracking problem);
- the line joins the two people the score is about; `d` small when they are close, `v` positive while one approaches;
- the curve at the bottom rises when you see the behavior build up. Pay attention to the **false peaks**: a box on a chair or a
  box that jumps for a frame gives a `v` spike (seen in Normal_v6). That is why M9 will require a score to stay high for
  a few seconds instead of reacting to one peak.

## Check 2: are the numbers right? (hand recomputation)
`python -m src.assm.hand_check Stalking_v3` (or notebook 03, section 4; `--frame N` to pick a frame).
It recomputes `d`, `v` and `score` for one row directly from the raw keypoints without using the scoring code, and prints
`OK` or `MISMATCH` for each. All three should be `OK`. To do it fully by hand: pick a row in `*_pairs.csv`, take the two ids' boxes in that frame
from the `.npz`, average the confident keypoints for each centroid, divide the pixel distance by the mean box height -> `d`.

## Check 3: pytest (logic)
`python -m pytest -q` - 30 tests, including synthetic people that approach, recede, stand still, block a path, overlap.

## Check 4: coverage (is there anything to score?)
The M8b summary prints `(no interaction)` for clips where fewer than 2 people were tracked together. On the 12 samples
Assassination_v1 (tiny blurred person) and Stalking_v9 (small, cropped people) have none. These are detection limits of the
small YOLOv8n model, not scoring bugs; they stay in the results, flagged, and are counted in the report.

## Honest reading of results
Scores are heuristics on 2D pixels. Peaks alone do not separate categories (Normal clips also peak). What to compare is
the **sustained** level and shape over time per category, which is what M9 and the report measure.
