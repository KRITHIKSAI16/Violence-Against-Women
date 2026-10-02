"""M8a tests: track cache format, summaries, empty clips, and a real-model smoke run."""
from pathlib import Path

import numpy as np
import pytest

from src.assm.track_poses import clip_summary, load_tracks, pack_tracks, run, track_clip
from src.config import REPO_ROOT, load_config

KP = np.zeros((17, 3), np.float32)


def _rows():
    # frame 0: ids 1,2 ; frame 1: id 1 only ; frame 3: ids 1,2,3
    box = np.array([0, 0, 10, 20], np.float32)
    return [(0, 1, box, 0.9, KP), (0, 2, box, 0.8, KP), (1, 1, box, 0.9, KP),
            (3, 1, box, 0.9, KP), (3, 2, box, 0.7, KP), (3, 3, box, 0.6, KP)]


def test_pack_shapes_and_dtypes():
    t = pack_tracks(_rows(), fps=30, width=640, height=360, n_frames=5)
    assert t["bbox"].shape == (6, 4) and t["kpts"].shape == (6, 17, 3)
    assert t["frame_idx"].dtype == np.int32 and t["track_id"].dtype == np.int32
    assert int(t["n_frames"]) == 5 and float(t["fps"]) == 30


def test_pack_empty_clip_has_valid_shapes():
    t = pack_tracks([], 30, 640, 360, 10)
    assert t["bbox"].shape == (0, 4) and t["kpts"].shape == (0, 17, 3)
    assert clip_summary(t)["max_people"] == 0


def test_summary_counts():
    s = clip_summary(pack_tracks(_rows(), 30, 640, 360, 5))
    assert s["unique_ids"] == 3
    assert s["frames_with_1plus"] == 3
    assert s["frames_with_2plus"] == 2
    assert s["max_people"] == 3


def test_npz_roundtrip(tmp_path):
    t = pack_tracks(_rows(), 25, 320, 180, 5)
    p = tmp_path / "a.npz"
    np.savez_compressed(p, **t)
    back = load_tracks(p)
    for k in t:
        assert np.array_equal(back[k], t[k])


MODEL = REPO_ROOT / "models" / "yolov8n-pose.pt"


@pytest.mark.skipif(not MODEL.exists(), reason="yolov8n-pose.pt not downloaded")
def test_real_model_on_blank_video_gives_empty_tracks(make_video, tmp_path):
    cfg = load_config()["assm"]
    v = make_video(tmp_path / "blank.mp4", n_frames=12)
    t = track_clip(v, cfg, 25.0, 320, 180)
    assert int(t["n_frames"]) == 12
    assert len(t["frame_idx"]) == 0  # moving white square is not a person


@pytest.mark.skipif(not MODEL.exists(), reason="yolov8n-pose.pt not downloaded")
def test_run_is_resumable_and_skips_failures(make_video, tmp_path):
    cfg = load_config()["assm"]
    v = make_video(tmp_path / "Normal" / "ok.mp4", n_frames=8)
    clean = {"clips": [
        {"clip_id": "ok", "category": "Normal", "path": str(v), "fps": 25.0, "width": 320, "height": 180},
        {"clip_id": "gone", "category": "Normal", "path": str(tmp_path / "missing.mp4"),
         "fps": 25.0, "width": 320, "height": 180}]}
    out = tmp_path / "tracks"
    s1, f1 = run(clean, cfg, out)
    assert len(s1) == 1 and len(f1) == 1 and (out / "Normal" / "ok.npz").exists()
    m = (out / "Normal" / "ok.npz").stat().st_mtime_ns
    run(clean, cfg, out)
    assert (out / "Normal" / "ok.npz").stat().st_mtime_ns == m  # not recomputed
