"""Tests for camera-motion estimation and ground-plane geometry (synthetic worlds with known answers)."""
import math

import cv2
import numpy as np
import pytest

from src.config import load_config
from src.context.camera import boxes_from_tracks, compensate, estimate_camera, summarize_camera
from src.context.ground import (focal_px, ground_series, pair_distance_m, speed_ms, standing_confidence,
                                zone_index, zone_name)
from src.assm.track_poses import pack_tracks

CFG = load_config()["context"]
FPS = 30.0


# ---------------------------------------------------------------- camera
def _textured(w=400, h=300, seed=0):
    rng = np.random.default_rng(seed)
    img = np.zeros((h * 2, w * 2), np.uint8)
    for _ in range(400):
        x, y, r = rng.integers(0, w * 2), rng.integers(0, h * 2), rng.integers(3, 12)
        cv2.circle(img, (int(x), int(y)), int(r), int(rng.integers(60, 255)), -1)
    return cv2.cvtColor(cv2.GaussianBlur(img, (3, 3), 0), cv2.COLOR_GRAY2BGR)


def _video(path, step_x, n=40, w=400, h=300):
    big = _textured(w, h)
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (w, h))
    for k in range(n):
        x0 = int(round(50 + step_x * k))
        vw.write(np.ascontiguousarray(big[40:40 + h, x0:x0 + w]))
    vw.release()
    return path


def test_static_camera_is_not_moving(tmp_path):
    cam = estimate_camera(_video(tmp_path / "s.mp4", 0.0), max_side=320)
    s = summarize_camera(cam, CFG)
    assert not s["moving"] and s["drift_frac"] < 0.01
    assert np.allclose(cam["T"][-1], np.eye(3), atol=1.5)


def test_panning_camera_detected_and_compensation_recovers_world_position(tmp_path):
    cam = estimate_camera(_video(tmp_path / "p.mp4", 3.0), max_side=320)      # scene shifts 3 px per frame (camera pans right)
    s = summarize_camera(cam, CFG)
    assert s["moving"] and s["moving_share"] > 0.8
    # a point fixed in the world at frame-0 position (100, 150) appears at x = 100 - 3*k in frame k; compensation must map it back
    k = 30
    seen = np.array([[100 - 3.0 * k, 150.0]])
    back = compensate(cam["T"], np.array([k]), seen)
    assert back[0, 0] == pytest.approx(100.0, abs=4.0) and back[0, 1] == pytest.approx(150.0, abs=4.0)


def test_people_boxes_do_not_pull_the_estimate(tmp_path):
    # static background; a bright moving square far from any texture must not make the camera "move"
    w, h = 400, 300
    big = _textured(w, h)
    vw = cv2.VideoWriter(str(tmp_path / "m.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (w, h))
    boxes = {}
    for k in range(40):
        fr = np.ascontiguousarray(big[40:40 + h, 50:50 + w]).copy()
        x = 20 + 6 * k
        cv2.rectangle(fr, (x, 100), (x + 40, 220), (255, 255, 255), -1)
        vw.write(fr)
        boxes[k] = [(x, 100, x + 40, 220)]
    vw.release()
    cam = estimate_camera(tmp_path / "m.mp4", boxes, 320)
    assert not summarize_camera(cam, CFG)["moving"]


def test_compensate_none_is_identity():
    xy = np.array([[10.0, 20.0], [30.0, 40.0]])
    assert np.allclose(compensate(None, np.array([0, 5]), xy), xy)


def test_brightness_marks_night(tmp_path):
    dark = tmp_path / "d.mp4"
    vw = cv2.VideoWriter(str(dark), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (160, 120))
    for _ in range(12):
        vw.write(np.full((120, 160, 3), 20, np.uint8))
    vw.release()
    assert summarize_camera(estimate_camera(dark), CFG)["night"]


# ---------------------------------------------------------------- ground plane
def _tracks(people, n=60, W=640, H=480):
    """people: {tid: fn(frame) -> (cx, y_top, box_h)}; boxes are 0.4*h wide."""
    rows = []
    for tid, fn in people.items():
        for f in range(n):
            cx, top, h = fn(f)
            rows.append((f, tid, np.array([cx - 0.2 * h, top, cx + 0.2 * h, top + h], np.float32), 0.9,
                         np.zeros((17, 3), np.float32)))
    return pack_tracks(rows, FPS, W, H, n)


def test_known_distance_in_meters():
    # two people 100 px tall (so 1.7 m): their centres 100 px apart laterally = 1.7 m apart; same depth
    t = _tracks({1: lambda f: (300, 200, 100.0), 2: lambda f: (400, 200, 100.0)})
    g = ground_series(t, CFG)
    d = pair_distance_m(g[1], g[2])
    assert d[30] == pytest.approx(1.7, rel=0.02)


def test_depth_from_height_and_focal_length():
    f = focal_px(640, CFG["fov_deg"])
    t = _tracks({1: lambda f_: (320, 100, 170.0)})            # 170 px tall -> 10 px/0.1 m -> Z = f * 1.7 / 170 = f/100
    g = ground_series(t, CFG)
    assert g[1]["Z"][30] == pytest.approx(f / 100.0, rel=0.01)
    assert g[1]["X"][30] == pytest.approx(0.0, abs=1e-6)


def test_farther_person_in_depth_changes_distance():
    near = 200.0                      # 200 px tall
    far = 100.0                       # 100 px tall = twice as far
    t = _tracks({1: lambda f: (320, 100, near), 2: lambda f: (320, 100, far)})
    g = ground_series(t, CFG)
    f = focal_px(640, CFG["fov_deg"])
    assert pair_distance_m(g[1], g[2])[30] == pytest.approx(f * 1.7 / far - f * 1.7 / near, rel=0.02)


def test_walking_speed_in_ms():
    # walks laterally 1.7 m/s: box 100 px tall -> 100 px per 1.7 m -> 100 px/s -> 3.33 px/frame
    t = _tracks({1: lambda f: (100 + 100.0 * f / FPS, 200, 100.0)})
    g = ground_series(t, CFG)
    assert speed_ms(g[1], FPS, CFG["vel_s"])[40] == pytest.approx(1.7, rel=0.05)


def test_zone_names_and_indices_follow_hall():
    assert [zone_name(d) for d in (0.3, 0.8, 2.0, 5.0, 9.0)] == ["intimate", "personal", "social", "public", "far"]
    assert zone_name(float("nan")) == "unknown"
    assert zone_index(np.array([0.3, 0.8, 2.0, 5.0, 9.0, np.nan])).tolist() == [0, 1, 2, 3, 4, -1]


def test_standing_confidence_flags_sitting_and_cropped():
    assert standing_confidence((100, 50, 140, 150), 640, 480) == 1.0
    assert standing_confidence((100, 50, 200, 120), 640, 480) == 0.3          # wide box: sitting
    assert standing_confidence((100, 300, 140, 480), 640, 480) == 0.3         # touches the bottom edge


def test_camera_compensation_changes_lateral_position_only_by_the_pan():
    t = _tracks({1: lambda f: (300 + 3.0 * f, 200, 100.0)})       # appears to move 3 px/frame
    T = np.tile(np.eye(3), (60, 1, 1))
    for k in range(60):
        T[k][0, 2] = -3.0 * k                                      # T maps frame-k pixels back to frame 0: the camera panned, the person is stationary
    still = ground_series(t, CFG, T)
    moving = ground_series(t, CFG, None)
    assert abs(still[1]["X"][50] - still[1]["X"][10]) < 0.02
    assert abs(moving[1]["X"][50] - moving[1]["X"][10]) > 1.0
    assert boxes_from_tracks(t)[0][0][3] == pytest.approx(300.0)
