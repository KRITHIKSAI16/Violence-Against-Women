# colab2: pre-violence interaction understanding on the hand-picked clips

One notebook, `VAW_curated_previolence.ipynb`. It does not touch anything from the earlier work: it has its own code (`src/curated/`), its own config (built at run time from `configs/colab.yaml`), and writes only to a **new sub-folder** `VAW_results/curated_previolence/` on your Drive.

## What you do (once)
1. Drive: open **Shared with me > violence**, right-click the folder > **Organize > Add shortcut > My Drive**. (A shared folder is not visible to Colab until it has a shortcut.)
2. Upload `FYP_data_filter.xlsx` (sheets per category, columns `video name` and `start time` in seconds) to My Drive, or into `VAW_results`, or into the `violence` folder.
3. Colab: **Runtime > Change runtime type > T4 GPU**, then **Runtime > Run all**.

If a clip name in the Excel does not match a video, stage 2 prints it (and the videos that have no row). Names are matched on category + clip number (`Assassination_v4 (1` matches `Assassination_v4 (1).mp4`), so spelling of the sheet name or folder does not matter.

## What happens, in order
| stage | what | output |
|---|---|---|
| prepare | read the start times, match the videos, cut every clip at its start time T (nothing after T is analysed), take Normal clips as control | `curated_manifest_clean.json`, `processed/` |
| perception | compare yolov8n@640 with yolo26x@1280 + BoT-SORT on a sample, keep the better by measured pair coverage | `perception_choice.json` |
| track | shots (editing cuts), pose + tracking | `tracks/`, `context/*_shots.json` |
| context | camera motion, distances in meters, facing, following, scene layout and depth, interaction graph | `context/` |
| analyze | per clip: relations over time, last seconds before T, lead-up type, plain sentences, distance figure, the pair video ending with a "violence starts at T" card | `stories/`, `figures/`, `videos/` |
| ml | pretrained video models (InternVideo2 CLIP, X-CLIP, V-JEPA 2): zero-shot prompt scores and a leave-one-clip-out head | `ml/`, `ml_summary.json` |
| report | evaluation, the jury page, the bundle | `report/index.html`, `summary.csv`, `evaluation.md`, `curated_previolence_bundle.zip` |

Each stage skips clips that are already done: if Colab disconnects, run all again. A cell prints its summary only when it ends; the `n/N` progress lines and the Drive folder show it is working. Time is not guessed here: every stage prints its own seconds when it finishes.

## Reading the output
* `report/index.html` is the page for the jury: summary table, a card per clip (video, distance over time with the proximity zones, key frames, sentences), the comparison with Normal clips and the honest verdict, methods and limits.
* In the pair video the person moving toward the other is boxed in red, the line between them carries the live distance in meters, the top bar says what they are doing (red = concerning, green = benign, grey = neutral), and the strip at the bottom shows the distance so far.
* The lead-up type (approach from behind, following, approach, closing in, raised arm, reach, already close, walking together, no visible lead-up) is a **rule-based heuristic**; its thresholds are printed on the page and are not tuned on data. Meters are approximate (1.7 m person-height prior, assumed field of view).
* The Normal comparison is only called a difference in behavior when it beats the filming-style baseline and holds among clips of the same resolution; otherwise the page says so.
* The interaction graph follows the vocabulary of the Hierarchical Interlacement Graph (HIG, CVPR 2024): situation, position, interaction, relation. It is HIG-structured, **not** the pretrained HIG model, and appearance is skipped on purpose (no identity, gender or age).

## If something fails
* "folder 'violence' was not found": step 1 above (the shortcut).
* "start-time file was not found": step 2 above, or set `ANNOTATIONS` in the second cell.
* An encoder (for example InternVideo2, whose Hub page documents no usage) cannot be loaded: it is named in the stage 7 output and skipped; the other models still run.
* A clip shows "no interacting pair identified": a person was not detected or the two were never tracked together; the card says which.
