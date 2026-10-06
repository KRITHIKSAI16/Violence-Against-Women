"""Curated dataset, step 2: a config for this dataset built FROM configs/colab.yaml, so the original config and every earlier result stay untouched.

Every Drive path that points into `.../VAW_results/<x>` is moved to `<out_root>/<x>` (out_root = `.../VAW_results/curated_previolence` on Colab),
the curated settings are applied, and the result is written next to the outputs as `curated_config.yaml`. All existing stage CLIs
(`python -m src.assm.track_poses --config <that file>` ...) then run on this dataset only.
"""
from pathlib import Path

import yaml

from src.config import load_config

DRIVE_RESULTS = "/content/drive/MyDrive/VAW_results"

# Short pre-violence footage is the point here, so pairs and episodes may be a little shorter than on the full dataset.
CONTEXT_OVERRIDES = {"min_cotracked_s": 0.6}
STORY_OVERRIDES = {"approach_min_s": 0.6, "very_close_min_s": 1.0, "mutual_min_s": 1.5, "block_min_s": 0.6}
PERCEPTION_CANDIDATES = [
    {"name": "v8n_640_byte", "model": "models/yolov8n-pose.pt", "imgsz": 640, "tracker": "configs/bytetrack_vaw.yaml"},
    {"name": "y26x_1280_botsort", "model": "models/yolo26x-pose.pt", "imgsz": 1280, "tracker": "configs/botsort_vaw.yaml"},
]


def _move(value, out_root):
    if isinstance(value, str) and value.startswith(DRIVE_RESULTS):
        return str(Path(out_root) / value[len(DRIVE_RESULTS):].lstrip("/"))
    if isinstance(value, dict):
        return {k: _move(v, out_root) for k, v in value.items()}
    if isinstance(value, list):
        return [_move(v, out_root) for v in value]
    return value


def build_config(out_root, base_config="configs/colab.yaml", write=True):
    """-> (cfg dict, path of the written yaml or None). out_root: the dataset's own output folder (created when write=True)."""
    out_root = Path(out_root)
    cfg = _move(load_config(base_config), out_root)
    cfg["context"].update(CONTEXT_OVERRIDES)
    cfg["story"].update(STORY_OVERRIDES)
    cfg["manifest_path"] = str(out_root / "curated_manifest.json")
    cfg["preprocess"]["clean_manifest_path"] = str(out_root / "curated_manifest_clean.json")
    cfg["preprocess"]["processed_dir"] = str(out_root / "processed")
    cfg["report"]["overrides_csv"] = str(out_root / "cut_times.csv")
    cfg["perception"] = {"shootout_dir": str(out_root / "shootout"), "max_s": 12, "holdout_file": "configs/holdout_clips.txt", "configs": PERCEPTION_CANDIDATES}
    cfg["curated"] = {"out_root": str(out_root), "report_dir": str(out_root / "report"), "videos_dir": str(out_root / "videos"),
                      "figures_dir": str(out_root / "figures"), "stories_dir": str(out_root / "stories"), "ml_dir": str(out_root / "ml"),
                      "n_normal": 40, "window_s": 2.0, "step_s": 0.5, "last_seconds": [1.5, 3.0]}
    path = None
    if write:
        out_root.mkdir(parents=True, exist_ok=True)
        path = out_root / "curated_config.yaml"
        path.write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True), encoding="utf-8")
    return cfg, path


def choose_perception(summary, cfg):
    """Pick the candidate with the best measured pair coverage (then pair-seconds, then fewest ids per person) from a shootout summary
    {name: {pair_clips, pair_seconds, ids_per_person, ...}} and write it into cfg['assm']. Returns (name, reason)."""
    ok = {n: s for n, s in summary.items() if s.get("clips")}
    if not ok:
        return PERCEPTION_CANDIDATES[0]["name"], "no shootout result: kept the default (yolov8n, 640 px)"
    key = lambda n: (round(ok[n]["pair_clips"], 3), round(ok[n]["pair_seconds"], 2), -ok[n]["ids_per_person"] if ok[n]["ids_per_person"] == ok[n]["ids_per_person"] else -9)
    best = max(ok, key=key)
    conf = next(c for c in PERCEPTION_CANDIDATES if c["name"] == best)
    cfg["assm"].update({"model": conf["model"], "imgsz": conf["imgsz"], "tracker": conf["tracker"]})
    return best, f"best measured pair coverage ({ok[best]['pair_clips']:.2f} of clips with a usable pair, {ok[best]['pair_seconds']:.1f} pair-seconds per clip)"
