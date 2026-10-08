# colabclasstrim: will violence follow? (clips cut at the violence start)

One notebook, `VAW_class_trim.ipynb`. It starts from a **finished colab2 run** and classifies violent clips (cut at the human start time T) against Normal clips. Nothing from colab2 or earlier work is changed: the code is `src/classcommon/` (shared) and `src/classtrim/`, the settings are `configs/classify.yaml`, and everything is written to **`VAW_results/classtrim/`**.

## What you do
1. Run colab2 first (`colab2/VAW_curated_previolence.ipynb`), including its `analyze` and `ml` stages. Note the output folder (for example `.../VAW_results/curated_previolence_v2`).
2. Colab: **Runtime > T4 GPU**, open `VAW_class_trim.ipynb`, set `CURATED` (the colab2 folder) and `OUT` in the settings cell, then **Run all**.
3. `USE_CAPTIONS = True` (default) turns **Layer B** on; set it to `False` to switch it off (geometry text only).

## Stages
| stage | GPU? | what | output |
|---|---|---|---|
| captions | yes | Layer B: a video-language model (Qwen3-VL) describes 4 frames of the last 3 s of every clip; the answer is cleaned (identity words neutralised, category names and, for this task, act words removed) | `captions/` |
| features | yes, only if needed | video windows that colab2 did not already encode (colab2's own windows are reused; if everything is there no model is loaded) | `features/video/` |
| train | no | documents (geometry text + captions), feature blocks, nested grouped cross-validation, report | `text/`, `report/` |

## What the classifiers see
Every clip, violent or Normal, is judged on its **last 3 seconds** (`span_s`): the text describes only that window, the video windows lie inside it, the geometry vector is computed over it. Normal clips are kept whole, so there is no cutting; clip length cannot be read off the input, and a `length` baseline is reported next to every model. Documents are generated from the numbers with identical wording for every clip: they never contain the words violence or a category name.

Models: a logistic regression per block (video embeddings X-CLIP / V-JEPA 2 / InternVideo2; sentence embeddings of the text; TF-IDF of the text; geometry numbers) and three fusions (late average, stacking, early concatenation). Baselines: filming `style` and clip `length`.

## Reading the report
**Start with `report/index.html`**: one self-contained page (open it from Drive or download it) that explains what the system does, how well it works (all methods against the shortcut baselines, ROC / PR / calibration charts) and where it fails (missed and falsely alarmed clips with the text the models saw, error rate by category / pair found / lead-up / length / resolution, hardest clips, a filterable table of every clip). `report.md` has the same numbers as plain text.

Accuracy, balanced accuracy, precision, recall, specificity, F1, MCC, ROC-AUC, PR-AUC, Brier, expected calibration error, each with a 95% bootstrap interval over clips; confusion matrix, recall per category, the misclassified clips with the text the model saw, and the **verdict**, which only claims behaviour when the best model beats both baselines and holds on clips of the same resolution. The best model is picked on the same cross-validation it is reported on, so its numbers are optimistic; the report says so.

## If something fails
* "the colab2 output folder was not found": fix `CURATED`; "curated_manifest_clean.json not found": run colab2's prepare stage.
* "not evaluated: N violent and M non-violent clips": a class has fewer than 10 clips (`cv.min_per_class` in `configs/classify.yaml`). Add start times to `FYP_Annotations.csv`, rerun colab2, then rerun this notebook; the features of finished clips are reused.
* A clip shows "no interacting pair": colab2 found no pair of people; the text says so, the geometry vector carries a `no_pair` flag, and the report lists how many clips are flagged.
* Captions failed for a clip: it keeps the geometry text only (named in the output).
