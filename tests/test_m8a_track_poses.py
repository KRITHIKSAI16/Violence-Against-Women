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


# ---- stitch_tracks ----
from src.assm.track_poses import stitch_tracks


def _tr(entries):
    """entries: (frame, id, cx, cy, h) -> packed tracks."""
    rows = [(f, i, np.array([cx - 15, cy - h / 2, cx + 15, cy + h / 2], np.float32), 0.9, KP)
            for f, i, cx, cy, h in entries]
    return pack_tracks(rows, 30, 640, 360, 200)


def _ids(t):
    return t["track_id"].tolist()


def test_stitch_joins_same_person_after_short_gap():
    t = _tr([(f, 1, 100, 200, 80) for f in range(0, 20)] + [(f, 2, 104, 200, 82) for f in range(30, 50)])
    s = stitch_tracks(t)
    assert len(set(_ids(s))) == 1
    assert set(s["raw_track_id"].tolist()) == {1, 2}  # raw ids preserved


def test_stitch_never_merges_people_seen_together():
    t = _tr([(f, 1, 100, 200, 80) for f in range(0, 30)] + [(f, 2, 110, 200, 80) for f in range(10, 40)])
    assert len(set(_ids(stitch_tracks(t)))) == 2


def test_stitch_respects_gap_distance_and_size_limits():
    base = [(f, 1, 100, 200, 80) for f in range(0, 20)]
    far_gap = _tr(base + [(f, 2, 100, 200, 80) for f in range(120, 140)])       # 100-frame gap
    far_pos = _tr(base + [(f, 2, 400, 200, 80) for f in range(30, 50)])         # jumped ~4 body heights
    diff_size = _tr(base + [(f, 2, 100, 200, 200) for f in range(30, 50)])      # 2.5x taller
    for t in (far_gap, far_pos, diff_size):
        assert len(set(_ids(stitch_tracks(t)))) == 2


def test_stitch_chains_multiple_fragments_and_handles_empty():
    t = _tr([(0, 1, 100, 200, 80), (1, 1, 100, 200, 80), (10, 2, 102, 200, 80), (11, 2, 102, 200, 80),
             (20, 3, 104, 200, 80), (21, 3, 104, 200, 80)])
    assert len(set(_ids(stitch_tracks(t)))) == 1
    assert stitch_tracks(pack_tracks([], 30, 640, 360, 5))["track_id"].shape == (0,)
