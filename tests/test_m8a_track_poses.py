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



# ---- shot awareness (scripted fake detector: no model needed) ----
import cv2  # noqa: E402

from src.video.shots import SHOT_BASE  # noqa: E402


class _A:
    """Mimics the tiny part of a torch tensor that track_clip uses."""
    def __init__(self, a):
        self.a = np.asarray(a)

    def int(self):
        return _A(self.a.astype(np.int64))

    def cpu(self):
        return self

    def numpy(self):
        return self.a


class _Res:
    def __init__(self, ids, xyxy, kp):
        self.boxes = type("B", (), {"id": _A(ids), "xyxy": _A(xyxy), "conf": _A([0.9] * len(ids))})() if len(ids) else type("B", (), {"id": None})()
        self.keypoints = type("K", (), {"data": _A(kp)})() if len(ids) else None


def make_fake_yolo(resets):
    class FakeYOLO:
        def __init__(self, path):
            self.predictor = object()                  # pretend a tracker already exists
            self.calls = 0

        def track(self, frame, **kw):
            if self.predictor is None:
                resets.append(self.calls)              # the model noticed it must build a fresh tracker at this frame
                self.predictor = object()
            self.calls += 1
            # one person with the tracker's local id 1 (a fresh tracker numbers people from 1 again) and one with id 2
            ids = [1, 2]
            xyxy = np.array([[10, 10, 50, 110], [100, 10, 140, 110]], np.float32)
            kp = np.zeros((2, 17, 3), np.float32)
            return [_Res(ids, xyxy, kp)]
    return FakeYOLO


def test_track_clip_restarts_at_cuts_and_makes_ids_unique_per_shot(tmp_path, make_video, monkeypatch):
    resets = []
    monkeypatch.setattr("ultralytics.YOLO", make_fake_yolo(resets))
    v = make_video(tmp_path / "v.mp4", w=160, h=120, fps=25.0, n_frames=30)
    cfg = {**load_config()["assm"], "stitch": False}
    t = track_clip(v, cfg, 25.0, 160, 120, shot_starts=[0, 10, 20])
    assert resets == [10, 20]                                           # tracker dropped exactly at the two cuts
    shots = (t["track_id"] // SHOT_BASE)
    assert sorted(set(shots.tolist())) == [0, 1, 2] and sorted(set(t["track_id"].tolist())) == [1, 2, SHOT_BASE + 1, SHOT_BASE + 2, 2 * SHOT_BASE + 1, 2 * SHOT_BASE + 2]
    assert t["shot_starts"].tolist() == [0, 10, 20] and int(t["n_frames"]) == 30
    assert (t["frame_idx"][shots == 1].min(), t["frame_idx"][shots == 1].max()) == (10, 19)


def test_no_shots_given_means_one_shot_and_old_id_range(tmp_path, make_video, monkeypatch):
    resets = []
    monkeypatch.setattr("ultralytics.YOLO", make_fake_yolo(resets))
    v = make_video(tmp_path / "v.mp4", w=160, h=120, fps=25.0, n_frames=12)
    t = track_clip(v, {**load_config()["assm"], "stitch": False}, 25.0, 160, 120)
    assert resets == [] and t["track_id"].max() == 2 and t["shot_starts"].tolist() == [0]


def test_max_frames_stops_early(tmp_path, make_video, monkeypatch):
    monkeypatch.setattr("ultralytics.YOLO", make_fake_yolo([]))
    v = make_video(tmp_path / "v.mp4", w=160, h=120, fps=25.0, n_frames=40)
    t = track_clip(v, {**load_config()["assm"], "stitch": False}, 25.0, 160, 120, max_frames=15)
    assert int(t["n_frames"]) == 15 and t["frame_idx"].max() == 14


def test_stitch_never_joins_tracks_of_different_shots():
    box = np.array([0, 0, 10, 20], np.float32)
    # same place, 5 frames apart (well inside the stitch limits) but on different sides of a cut
    rows = [(f, 7, box, 0.9, KP) for f in range(0, 20)] + [(f, SHOT_BASE + 3, box, 0.9, KP) for f in range(25, 45)]
    out = stitch_tracks(pack_tracks(rows, 30, 640, 360, 60))
    assert sorted(set(out["track_id"].tolist())) == [7, SHOT_BASE + 3]
    same_shot = [(f, 7, box, 0.9, KP) for f in range(0, 20)] + [(f, 9, box, 0.9, KP) for f in range(25, 45)]
    assert len(set(stitch_tracks(pack_tracks(same_shot, 30, 640, 360, 60))["track_id"].tolist())) == 1       # the same pair inside one shot IS joined


def test_run_detects_and_caches_shots_and_passes_them_to_the_tracker(tmp_path, make_video, monkeypatch):
    from src.video.shots import load_shots
    resets = []
    monkeypatch.setattr("ultralytics.YOLO", make_fake_yolo(resets))
    v = make_video(tmp_path / "Normal" / "x.mp4", w=160, h=120, fps=25.0, n_frames=20)
    clean = {"clips": [{"clip_id": "x", "category": "Normal", "path": str(v), "fps": 25.0, "width": 160, "height": 120}]}
    monkeypatch.setattr("src.assm.track_poses.detect_shots",
                        lambda *a, **k: {"clip_id": "x", "fps": 25.0, "n_frames": 20, "method": "fake", "threshold": 0.5, "min_shot_s": 0.4,
                                         "shots": [{"id": 0, "start_f": 0, "end_f": 9, "start_s": 0.0, "end_s": 0.4}, {"id": 1, "start_f": 10, "end_f": 19, "start_s": 0.4, "end_s": 0.8}],
                                         "cuts_s": [0.4]})
    shots_cfg = load_config()["shots"]
    cfg = {**load_config()["assm"], "stitch": False}
    s1, f1 = run(clean, cfg, tmp_path / "tracks", shots_dir=tmp_path / "ctx", shots_cfg=shots_cfg)
    assert len(s1) == 1 and f1 == [] and s1[0]["shots"] == 2 and resets == [10]
    assert load_shots(tmp_path / "ctx", "Normal", "x")["cuts_s"] == [0.4]                # cached for the next stages


def test_unreadable_video_raises(tmp_path, monkeypatch):
    monkeypatch.setattr("ultralytics.YOLO", make_fake_yolo([]))
    with pytest.raises(ValueError):
        track_clip(tmp_path / "missing.mp4", {**load_config()["assm"], "stitch": False}, 25.0, 160, 120)
