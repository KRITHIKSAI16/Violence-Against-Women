# colabclassfull: is violence in this video? (whole, untrimmed videos)

One notebook, `VAW_class_full.ipynb`. Label = the category folder (Normal -> non-violent, every other category -> violent), so **no start times are needed**. Nothing from colab2 or earlier work is changed: the code is `src/classcommon/` (shared) and `src/classfull/`, the settings are `configs/classify.yaml`, and everything is written to **`VAW_results/classfull/`**.

## What you do
1. Colab: **Runtime > T4 GPU**, open `VAW_class_full.ipynb`, check `DATA_ROOT` (the shared `violence` folder) and `OUT` in the settings cell, then **Run all**.
2. Try `MAX_CLIPS = 20` first for a quick run (about 10 violent + 10 Normal clips), then set it back to `None`.
3. `USE_CAPTIONS = True` (default) turns **Layer B** on; set it to `False` to switch it off.
4. Optional: `CURATED` (a finished colab2 folder) makes this run use the person detector that colab2 measured as best.

## Stages (every one is resumable)
| stage | GPU? | what |
|---|---|---|
| prepare | no | list every video named `Category_vNN`, re-encode each WHOLE (30 fps, 640 px, no trimming, no 45 s cap), write the manifest (names that do not match are printed) |
| track | yes | shots + pose tracking (the existing stage CLIs) |
| context | yes (scene/depth) | camera, distances in meters, scene layout, story (existing CLIs) |
| analyze | no | per clip: relations over time, pair facts, lead-up type (existing `analyze_clip`, no videos rendered) |
| captions | yes | Layer B: Qwen3-VL describes 8 frames spread over the clip; the answer is cleaned (identity words neutralised, category names removed) |
| features | yes | frozen X-CLIP / V-JEPA 2 / InternVideo2 windows (at most 24, evenly spread; mean and max pooled) |
| train | no | documents, nested grouped cross-validation, report |

## What the classifiers see
The whole video: the video features, the text of what happens between the two main people over the whole clip (+ captions), and the geometry numbers. Because the act itself is in the video, expect higher scores than for the trim task; the report puts the same baselines next to every model: filming `style` (Normal clips come from another source, earlier work found style alone separates them with AUC 0.99) and clip `length`. The verdict only claims behaviour when the best model beats both and holds on clips of the same resolution.

## Reading the report
`report/report.md` has accuracy, balanced accuracy, precision, recall, specificity, F1, MCC, ROC-AUC, PR-AUC, Brier, expected calibration error (95% bootstrap intervals over clips), the confusion matrix, recall per category, the misclassified clips with their text, and the verdict. The best model is picked on the same cross-validation it is reported on, so its numbers are optimistic. Near-duplicate videos are kept in the same fold (frame hash) so a re-upload cannot sit in training and test.

## If something fails
* "video folder not found": fix `DATA_ROOT`.
* A video that cannot be read is skipped with its name; "no interacting pair" clips are kept and flagged (the report shows the metrics with and without them).
* Long videos need more time in `track`; set `classify.max_len_s` in `configs/classify.yaml` to cap the prepared length (default: the whole video).
* Disconnected: run all cells again; finished clips are skipped.
