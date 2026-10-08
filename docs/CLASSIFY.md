# Classification pipelines (`src/classcommon/`, `src/classtrim/`, `src/classfull/`, `colabclasstrim/`, `colabclassfull/`)

New code only: nothing in `STATE.md`, `STATUS_REPORT.md`, `src/curated/` or earlier modules was changed; the new code imports them.

## Two tasks
| | trim (`colabclasstrim`) | full (`colabclassfull`) |
|---|---|---|
| question | will violence follow? | is violence in this video? |
| input | colab2 clips (violent: cut at T; Normal: whole), judged on their last `span_s` = 3 s | whole videos |
| label | category: Normal 0, else 1 | same |
| start times needed | yes (only colab2's clips with a T are used) | no |

## Data flow
`records` (manifest + analysis result per clip) -> `text_build.describe` (geometry sentences) [+ `vlm_caption` (Layer B)] -> documents -> blocks: video (`feat_video`), text embedding + TF-IDF (`feat_text`), geometry (`feat_geom`), baselines style / length -> `cv` (per-block models, three fusions, nested grouped cross-validation) -> `metrics` -> `report`.

## Modules
| file | role |
|---|---|
| `config.py` | class config from `configs/colab.yaml` (through `src.curated.config`) + `configs/classify.yaml`; reuse of colab2's detector choice |
| `labels.py` | label from the category folder, video collection for the full task |
| `text_build.py` | geometry text; same wording for every clip, no label words; trim uses only the last `span_s` seconds with times relative to that window |
| `vlm_caption.py` | Layer B (optional): frame captions by a video-language model, cleaning of the answer |
| `feat_geom.py` | geometry vector, style vector, length |
| `feat_video.py` | frozen encoder windows (cached; trim reuses colab2's), pooling, zero-shot margin |
| `feat_text.py` | sentence embeddings with a fingerprinted cache |
| `edgecases.py` | duplicate groups (frame hash), too-few-clips check, clip flags, label-word scan |
| `cv.py` | models, fusion (late / early / stack), threshold choice, repeated stratified group cross-validation |
| `metrics.py` | threshold and score metrics, bootstrap intervals, per-category recall, same-resolution AUC |
| `pipeline.py`, `stages.py`, `report.py`, `report_html.py` | the stages shared by both tasks, GPU stages, the evaluation report (`report.md`, csv, json) and the single-file `index.html` (system, data, performance, failures, every clip) |
| `classtrim/run.py`, `classfull/run.py` | the CLIs (`--stage`, `--out-root`, `--no-captions`) |

## Edge cases and what is done about them
| case | handling |
|---|---|
| Normal is a different source (style alone separated it with AUC 0.99) | `style` and `length` baselines scored by the same cross-validation; verdict needs the best model to beat both and to hold on same-resolution clips |
| Normal clips are longer than trimmed clips | no cutting: every clip is judged on the same last `span_s` seconds (text, video windows, geometry); `length` baseline |
| the existing result wording differs between violent and Normal clips ("before the violence" vs "of the clip") | the text is generated from the numbers, not from the ready-made lines |
| captions name the act or the category | category sentences always removed, act sentences removed for trim (a hit means the cut is late); identity words neutralised; counts reported |
| too few clips in a class | no metric is printed; the report says how many are missing |
| near-duplicate videos | 64-bit frame hash groups, kept in one fold |
| clips with no pair, very short clips, no result | kept, flagged, `no_pair` feature, metrics shown with and without |
| class imbalance | balanced class weights, balanced-accuracy thresholds, PR-AUC, counts everywhere |
| a modality or encoder is missing | it is left out and named in the report |
| optimistic model selection | stated in the report; thresholds, PCA, imputers and fusion weights are fitted inside the training clips only |

## Honest points
* Layer B puts a language model into the pipeline, which the repository rules otherwise exclude; it is a switch, ablated in the report (blocks `text_*_geom` vs `text_*_full`), and the prompt forbids appearance, gender, age and identity (the cleaner is best effort).
* With very few clips per class the intervals are wide; the report shows them and no claim is made without the counts.
* The trim task has no time-to-T analysis and the report does not compare the two tasks automatically: compare the two `metrics.json` files.
* The GPU stages (captioning, encoders, tracking) and Drive access are not exercised by the unit tests; the logic around them is, with fake encoders and a fake captioner.

## Jury showcase, notebook `colabshowcase/VAW_showcase.ipynb` (`src/classcommon/showcase.py`, `python -m src.classfull.showcase --out-root <classfull folder>`)
Reads the finished stages (no GPU) and writes `<out-root>/showcase/`: `features_overview.png` (cue shares, closest distance, concern score, classifier AUCs with the style / length baselines, all videos), `snapshot_<k>_<violent>_vs_<Normal>.png` (16:9 slide picture: key frames with the pair boxed, distance curve, behaviour timeline, numbers, out-of-fold model score) and `compare_<k>_...mp4` (the two videos side by side with live distance strip). The example pair is chosen by a rule (most vs least reach / follow / approach evidence among videos of 8-30 s with a measured pair), not at random; override with `--violent` / `--normal`. Cues and the concern score are heuristics, meters are approximate.
