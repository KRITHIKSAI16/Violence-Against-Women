"""Classification pipelines, step 1: one config for a run, built FROM configs/colab.yaml (through src.curated.config) plus configs/classify.yaml.

The curated config moves every Drive path under the run's own output folder; this module adds a `classify` block (settings + the folders of this layer)
and writes the result as `class_config.yaml` there. Nothing in the existing configs is changed.
"""
import json
from pathlib import Path

import yaml

from src.config import resolve_path
from src.curated.config import PERCEPTION_CANDIDATES
from src.curated.config import build_config as build_curated_config

TASKS = ("trim", "full")


def load_classify_defaults(path="configs/classify.yaml"):
    return yaml.safe_load(resolve_path(path).read_text(encoding="utf-8"))["classify"]


def build_config(out_root, task, curated_root=None, use_captions=None, base_config="configs/colab.yaml", classify_path="configs/classify.yaml", write=True):
    """-> (cfg dict, path of the written yaml or None).

    out_root: this run's own output folder. task: "trim" (clips cut at the violence start) or "full" (whole videos).
    curated_root: the finished colab2 folder (curated_previolence...) that the trim task reads; None for the full task.
    use_captions: True/False overrides the yaml (Layer B on/off); None keeps the yaml value."""
    if task not in TASKS:
        raise ValueError(f"task must be one of {TASKS}, got {task!r}")
    cfg, _ = build_curated_config(out_root, base_config, write=False)
    cl = load_classify_defaults(classify_path)
    if use_captions is not None:
        cl["use_captions"] = bool(use_captions)
    out = Path(out_root)
    cl.update({"task": task, "out_root": str(out), "curated_root": str(curated_root) if curated_root else None,
               "text_dir": str(out / "text"), "caption_dir": str(out / "captions"), "feat_dir": str(out / "features"), "report_dir": str(out / "report")})
    cfg["classify"] = cl
    if task == "full":
        apply_perception_choice(cfg, curated_root)
    path = None
    if write:
        out.mkdir(parents=True, exist_ok=True)
        path = out / "class_config.yaml"
        path.write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True), encoding="utf-8")
    return cfg, path


def apply_perception_choice(cfg, curated_root):
    """Use the detector that the colab2 perception stage measured as best (perception_choice.json), so this run tracks people the same way.
    Returns the chosen name, or None when there is no such file (the default detector of the config is kept)."""
    if not curated_root:
        return None
    p = Path(curated_root) / "perception_choice.json"
    if not p.exists():
        return None
    name = json.loads(p.read_text(encoding="utf-8")).get("chosen")
    conf = next((c for c in PERCEPTION_CANDIDATES if c["name"] == name), None)
    if conf is None:
        return None
    cfg["assm"].update({"model": conf["model"], "imgsz": conf["imgsz"], "tracker": conf["tracker"]})
    return name


def load_class_config(out_root):
    return yaml.safe_load((Path(out_root) / "class_config.yaml").read_text(encoding="utf-8"))
