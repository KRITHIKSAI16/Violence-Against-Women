"""Tests for pose-based features (facing, head, contact, reach) and lagged-path following, on synthetic people."""
import numpy as np
import pytest

from src.config import load_config
from src.context.follow import follow_scores, rolling_mean
from src.context.pose_features import (angle_deg, body_facing, contact_features, head_facing, runs_of, track_pose)
from src.assm.track_poses import pack_tracks

CFG = load_config()["context"]
FPS = 30.0
H = 100.0


def skeleton(cx, top, h, facing="camera", wrists=None):
    """17x3 COCO keypoints for a person of box height h. facing: camera | away | right | left (image side)."""
    kp = np.zeros((17, 3))
    sw = 0.23 * h
    mid = cx
    if facing == "camera":
        lx, rx = mid + sw / 2, mid - sw / 2          # their left shoulder appears on the image right
    elif facing == "away":
        lx, rx = mid - sw / 2, mid + sw / 2
    else:
        lx, rx = mid + 0.02 * h, mid - 0.02 * h      # sideways: shoulders overlap
    kp[5] = [lx, top + 0.2 * h, 0.9]
    kp[6] = [rx, top + 0.2 * h, 0.9]
    kp[11] = [cx + 0.08 * h, top + 0.5 * h, 0.9]
    kp[12] = [cx - 0.08 * h, top + 0.5 * h, 0.9]
    if facing == "camera":
        kp[0] = [cx, top + 0.07 * h, 0.9]
        kp[1] = [cx + 0.02 * h, top + 0.06 * h, 0.9]
        kp[2] = [cx - 0.02 * h, top + 0.06 * h, 0.9]
        kp[3] = [cx + 0.05 * h, top + 0.07 * h, 0.9]
        kp[4] = [cx - 0.05 * h, top + 0.07 * h, 0.9]
    elif facing == "away":
        kp[3] = [cx - 0.05 * h, top + 0.07 * h, 0.9]
        kp[4] = [cx + 0.05 * h, top + 0.07 * h, 0.9]
        kp[0:3, 2] = 0.05                              # face not visible
    else:
        sgn = 1.0 if facing == "right" else -1.0
        kp[0] = [cx + sgn * 0.06 * h, top + 0.07 * h, 0.9]
        kp[3 if facing == "left" else 4] = [cx - sgn * 0.01 * h, top + 0.07 * h, 0.9]
    kp[9] = kp[10] = [cx, top + 0.45 * h, 0.9] if wrists is None else [*wrists[0], 0.9]
    if wrists is not None:
        kp[10] = [*wrists[1], 0.9]
    return kp


def bat(kp_list, hb=H):
    return np.stack(kp_list), np.full(len(kp_list), hb)


# ---------------------------------------------------------------- facing
@pytest.mark.parametrize("facing,expect", [("camera", (0.0, -1.0)), ("away", (0.0, 1.0)), ("right", (1.0, 0.0)), ("left", (-1.0, 0.0))])
def test_body_facing_direction(facing, expect):
    kp, hb = bat([skeleton(300, 100, H, facing)])
    v, c = body_facing(kp, hb)
    assert v[0] == pytest.approx(expect, abs=0.2) and c[0] > 0.4         # sideways bodies keep a little shoulder separation


def test_body_facing_unknown_without_shoulders():
    kp, hb = bat([skeleton(300, 100, H)])
    kp[0, 5:7, 2] = 0.0
    v, c = body_facing(kp, hb)
    assert np.isnan(v[0]).all() and c[0] == 0.0


def test_head_front_back_and_side():
    for facing, z in (("camera", -1), ("away", +1)):
        kp, hb = bat([skeleton(300, 100, H, facing)])
        v, c = head_facing(kp, hb)
        assert np.sign(v[0, 1]) == z and c[0] > 0.5
    kp, hb = bat([skeleton(300, 100, H, "right")])
    v, _ = head_facing(kp, hb)
    assert v[0, 0] > 0.5


def test_head_turned_back_while_body_faces_away_is_detected():
    kp = skeleton(300, 100, H, "away")
    kp[0] = [300, 107, 0.9]; kp[1] = [302, 106, 0.9]; kp[2] = [298, 106, 0.9]       # face visible although the back is turned
    kps, hb = bat([kp])
    b, _ = body_facing(kps, hb)
    h, _ = head_facing(kps, hb)
    assert b[0, 1] > 0.5 and h[0, 1] < -0.5          # body away (+Z), head toward the camera (-Z)


def test_angle_deg():
    assert angle_deg(np.array([[0.0, 1.0]]), np.array([[0.0, 1.0]]))[0] == pytest.approx(0.0, abs=1e-6)
    assert angle_deg(np.array([[0.0, 1.0]]), np.array([[0.0, -1.0]]))[0] == pytest.approx(180.0, abs=1e-6)
    assert angle_deg(np.array([[1.0, 0.0]]), np.array([[0.0, 1.0]]))[0] == pytest.approx(90.0, abs=1e-6)
    assert np.isnan(angle_deg(np.array([[np.nan, np.nan]]), np.array([[0.0, 1.0]]))[0])


def test_runs_of():
    assert runs_of([0, 1, 1, 1, 0, 1, 1], 3) == [(1, 3)]
    assert runs_of([1, 1], 3) == []


# ---------------------------------------------------------------- contact / reach
def _two_people(wrist_i_fn, n=30):
    rows = []
    for f in range(n):
        for tid, cx, wr in ((1, 300.0, wrist_i_fn(f)), (2, 330.0, None)):
            kp = skeleton(cx, 100, H, "camera", wrists=wr)
            rows.append((f, tid, np.array([cx - 20, 100, cx + 20, 200], np.float32), 0.9, kp.astype(np.float32)))
    return pack_tracks(rows, FPS, 640, 480, n)


def test_contact_when_wrist_reaches_other_torso():
    t = _two_people(lambda f: ((332.0, 125.0), (300.0, 145.0)))          # i's left wrist on j's shoulder line
    pi, pj = track_pose(t, 1, 30), track_pose(t, 2, 30)
    c = contact_features(pi, pj, 1.7, 3, FPS)
    assert c["wrist_torso_m"][10] < CFG["contact_wrist_m"]           # wrist on the other person's shoulder = contact
    t2 = _two_people(lambda f: ((250.0, 145.0), (250.0, 145.0)))         # hands down at own side, far from j
    c2 = contact_features(track_pose(t2, 1, 30), track_pose(t2, 2, 30), 1.7, 3, FPS)
    assert c2["wrist_torso_m"][10] > CFG["contact_wrist_m"]            # standing 0.5 m apart with hands down is not contact


def test_reach_speed_toward_other_person():
    # i's wrist sweeps from own body toward j's chest at 2 body widths per ~0.3 s
    def wr(f):
        x = 300.0 + 3.0 * f
        return (x, 125.0), (300.0, 145.0)
    t = _two_people(wr)
    c = contact_features(track_pose(t, 1, 30), track_pose(t, 2, 30), 1.7, 3, FPS)
    assert np.nanmax(c["reach_i_to_j_ms"]) > 1.0
    assert np.nanmax(c["reach_j_to_i_ms"]) < 0.5


# ---------------------------------------------------------------- following
def _walk(x0, z0, vx, vz, n, noise=0.0, seed=0):
    t = np.arange(n) / FPS
    rng = np.random.default_rng(seed)
    return np.c_[x0 + vx * t, z0 + vz * t] + rng.normal(0, noise, (n, 2))


def test_rolling_mean_handles_nan():
    x = np.array([1.0, 1.0, np.nan, 1.0, 1.0, 1.0])
    assert rolling_mean(x, 3)[4] == pytest.approx(1.0)
    assert np.isnan(rolling_mean(np.array([np.nan] * 6), 3)).all()


def test_follower_retracing_path_with_lag_is_detected_with_right_lag():
    n = 300
    leader = _walk(0, 10, 1.4, 0, n)
    lag = int(2 * FPS)
    follower = np.vstack([np.full((lag, 2), np.nan), leader[:-lag]]) + np.array([0.0, 0.3])   # same path, 2 s later, 30 cm off
    follower[:lag] = leader[0] - np.array([2.8, 0.0])
    r = follow_scores(follower, leader, FPS, CFG)
    assert r["follow"][200]
    assert r["lag_s"][200] == pytest.approx(2.0, abs=0.5)
    assert not follow_scores(leader, follower, FPS, CFG)["follow"][200]              # the roles are not symmetric


def test_walking_side_by_side_is_not_following():
    n = 300
    a = _walk(0, 10, 1.4, 0, n)
    b = _walk(0, 11.0, 1.4, 0, n)
    assert not follow_scores(a, b, FPS, CFG)["follow"][100:].any()
    assert not follow_scores(b, a, FPS, CFG)["follow"][100:].any()


def test_person_standing_near_a_stationary_person_is_not_following():
    n = 300
    a = _walk(1.0, 10, 0.0, 0.0, n)
    b = _walk(0.0, 10, 0.0, 0.0, n)
    assert not follow_scores(a, b, FPS, CFG)["follow"].any()


def test_two_people_walking_apart_is_not_following():
    n = 300
    a = _walk(0, 10, -1.4, 0, n)
    b = _walk(0, 10, 1.4, 0, n)
    assert not follow_scores(a, b, FPS, CFG)["follow"].any()


def test_follow_survives_position_noise():
    n = 300
    leader = _walk(0, 10, 1.4, 0, n, noise=0.08, seed=1)
    base = _walk(0, 10, 1.4, 0, n)
    lag = int(1.5 * FPS)
    follower = np.vstack([np.full((lag, 2), np.nan), base[:-lag]]) + np.random.default_rng(2).normal(0, 0.08, (n, 2))
    follower[:lag] = base[0] - np.array([2.1, 0.0])
    assert follow_scores(follower, leader, FPS, CFG)["follow"][150:].mean() > 0.7


def test_clip_shorter_than_the_largest_lag_does_not_crash():
    a = _walk(0, 10, 1.4, 0, 100)          # 3.3 s, the lag range goes to 6 s
    b = _walk(2, 10, 1.4, 0, 100)
    r = follow_scores(a, b, FPS, CFG)
    assert len(r["follow"]) == 100
