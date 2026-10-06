import json

import cv2
import numpy as np

from src.curated.config import DRIVE_RESULTS, build_config, choose_perception
from src.curated.prepare import build_curated_manifest, prepare_clip


def make_video(path, seconds=8, fps=25, size=(320, 240)):
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
    for k in range(seconds * fps):
        vw.write(np.full((size[1], size[0], 3), 8 * (k // fps), np.uint8))
    vw.release()


def test_config_moves_only_drive_paths_and_leaves_the_original_alone(tmp_path):
    cfg, path = build_config(tmp_path / "out")
    assert path.exists() and cfg["assm"]["tracks_dir"] == str(tmp_path / "out" / "tracks") and cfg["context"]["context_dir"] == str(tmp_path / "out" / "context")
    assert DRIVE_RESULTS not in json.dumps(cfg) and cfg["context"]["min_cotracked_s"] == 0.6
    from src.config import load_config
    orig = load_config("configs/colab.yaml")
    assert orig["assm"]["tracks_dir"].startswith(DRIVE_RESULTS) and orig["context"]["min_cotracked_s"] != 0.6       # the real colab config is unchanged
    assert cfg["report"]["overrides_csv"].startswith(str(tmp_path / "out"))


def test_choose_perception_prefers_pair_coverage_then_pair_seconds(tmp_path):
    cfg, _ = build_config(tmp_path / "o", write=False)
    s = {"v8n_640_byte": {"clips": 9, "pair_clips": 0.6, "pair_seconds": 5.0, "ids_per_person": 1.2},
         "y26x_1280_botsort": {"clips": 9, "pair_clips": 0.8, "pair_seconds": 4.0, "ids_per_person": 1.1}}
    name, why = choose_perception(s, cfg)
    assert name == "y26x_1280_botsort" and cfg["assm"]["imgsz"] == 1280 and "0.80" in why
    cfg2, _ = build_config(tmp_path / "o2", write=False)
    assert choose_perception({}, cfg2)[0] == "v8n_640_byte" and cfg2["assm"]["imgsz"] == 640


def test_clips_are_trimmed_at_the_start_time_and_the_manifest_is_written(tmp_path):
    make_video(tmp_path / "a.mp4")
    meta = prepare_clip(tmp_path / "a.mp4", tmp_path / "o" / "a.mp4", 3.0, 30, 160)
    assert abs(meta["duration_s"] - 3.0) < 0.2 and max(meta["width"], meta["height"]) == 160
    cap = cv2.VideoCapture(str(tmp_path / "o" / "a.mp4"))
    ok, last = True, None
    while ok:
        ok, f = cap.read()
        last = f if ok else last
    assert last.mean() < 8 * 3 + 6                      # nothing brighter than second 2.x of the source (brightness steps once a second): no frame from after T
    cfg, _ = build_config(tmp_path / "out")
    clips = [{"clip_id": "Assassination_v4", "category": "Assassination", "path": str(tmp_path / "a.mp4"), "start_s": 3.0, "status": "ok"},
             {"clip_id": "Normal_v1", "category": "Normal", "path": str(tmp_path / "a.mp4"), "start_s": None},
             {"clip_id": "Bad_v1", "category": "Stalking", "path": str(tmp_path / "missing.mp4"), "start_s": 2.0}]
    m = build_curated_manifest(clips, cfg, max_normal_s=4.0)
    assert [c["clip_id"] for c in m["clips"]] == ["Assassination_v4", "Normal_v1"] and m["skipped"][0]["clip_id"] == "Bad_v1"
    by = {c["clip_id"]: c for c in m["clips"]}
    assert abs(by["Assassination_v4"]["duration_s"] - 3.0) < 0.2 and abs(by["Normal_v1"]["duration_s"] - 4.0) < 0.2 and by["Normal_v1"]["label"] == "normal"
    assert (tmp_path / "out" / "cut_times.csv").read_text().startswith("clip_id,cut_s")
