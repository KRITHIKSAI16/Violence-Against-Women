"""Tests for the Stage A orchestrator: end-to-end behaviors on synthetic worlds with known answers."""
import numpy as np
import pytest

from src.assm.track_poses import pack_tracks
from src.config import load_config
from src.context.features import analyze_context, clip_events, load_context, save_context
from tests.test_context_pose_follow import skeleton

CTX = load_config()["context"]
FPS, W, H = 30.0, 640, 480


def world(people, n=300, facing=None, h=100.0):
    """people: {tid: fn(t_s) -> (cx, top)}; box height h px unless the fn returns 3 values (cx, top, h)."""
    rows = []
    for tid, fn in people.items():
        for f in range(n):
            r = fn(f / FPS)
            cx, top = r[0], r[1]
            hh = r[2] if len(r) > 2 else h
            fc = (facing or {}).get(tid, "camera")
            kp = skeleton(cx, top, hh, fc).astype(np.float32)
            rows.append((f, tid, np.array([cx - 0.2 * hh, top, cx + 0.2 * hh, top + hh], np.float32), 0.9, kp))
    return pack_tracks(rows, FPS, W, H, n)


def has(scene, group):
    return group in scene["clip_events"]


def test_follower_walking_behind_leader_is_found_with_lag():
    # both walk right at 0.85 m/s (50 px/s at 100 px per 1.7 m) and stay inside the frame; the follower retraces the leader's path 2 s later
    leader = lambda t: (160 + 50.0 * t, 150)
    follower = lambda t: (160 + 50.0 * (t - 2.0), 150)
    sc, _ = analyze_context(world({1: follower, 2: leader}, n=300), CTX)
    assert has(sc, "follow")
    ev = sc["pairs"]["1_2"]
    assert "follow_i_j" in ev and "follow_j_i" not in ev           # id1 follows id2, not the reverse
    assert any(r[1] - r[0] >= CTX["follow_min_s"] for r in ev["follow_i_j"]["runs"])


def test_walking_side_by_side_is_not_following():
    a = lambda t: (100 + 50.0 * t, 150, 100.0)
    b = lambda t: (100 + 50.0 * t, 150, 104.0)             # same path, slightly different depth (about 0.7 m apart in depth)
    sc, _ = analyze_context(world({1: a, 2: b}, n=300), CTX)
    assert not has(sc, "follow")


def test_stationary_person_nearby_is_not_followed():
    sc, _ = analyze_context(world({1: lambda t: (300, 150), 2: lambda t: (360, 150)}, n=240), CTX)
    assert not has(sc, "follow")


def test_approach_from_behind_when_target_faces_away():
    # id2 stands still facing away from the camera; id1 (nearer the camera, i.e. behind id2) walks up to 0.9 m
    # facing away means id2's back is to the camera, so the approacher must come from the camera side
    approacher = lambda t: (300 + 40.0, 150, 130.0 - min(t, 3.0) * 8.0)       # box shrinks as it moves away from the camera? keep simple below
    still = lambda t: (300, 150, 100.0)
    # build with explicit depth change: approacher starts nearer to the camera (taller box) and closes in
    approach = lambda t: (330, 120, 140.0 - 12.0 * min(t, 3.0))
    sc, arr = analyze_context(world({1: approach, 2: still}, n=150, facing={1: "away", 2: "away"}), CTX)
    a = arr[(1, 2)]
    # id2 faces away from id1 (id1 is nearer the camera, behind id2's back? id2 faces +Z away from camera, id1 is at smaller Z)
    assert np.nanmax(a["ang_j_to_i"]) > 120.0
    assert a["approach_behind_i_j"].any()


def test_no_approach_from_behind_when_target_faces_the_approacher():
    approach = lambda t: (330, 120, 140.0 - 12.0 * min(t, 3.0))
    still = lambda t: (300, 150, 100.0)
    sc, arr = analyze_context(world({1: approach, 2: still}, n=150, facing={1: "away", 2: "camera"}), CTX)
    assert not arr[(1, 2)]["approach_behind_i_j"].any()


def test_cropped_fast_box_does_not_create_flee():
    # id2's box touches the bottom frame edge and its height jumps (a person leaning in); speed must be ignored
    still = lambda t: (300, 150)
    cropped = lambda t: (330, 380 if t < 2 else 330, 100.0 if t < 2 else 150.0)          # y2 = 480 -> cropped
    sc, arr = analyze_context(world({1: still, 2: cropped}, n=150), CTX)
    assert not has(sc, "flee")


def test_running_away_is_flee():
    # id2 runs away laterally at ~4 m/s (235 px/s at 100 px per 1.7 m) from a standing person
    still = lambda t: (200, 150)
    run = lambda t: (260 + 235.0 * t, 150)
    sc, _ = analyze_context(world({1: still, 2: run}, n=75), CTX)
    assert has(sc, "flee")


def test_nothing_happens_for_two_people_standing_far_apart():
    sc, _ = analyze_context(world({1: lambda t: (100, 150), 2: lambda t: (560, 150)}, n=150), CTX)
    assert sc["clip_events"] == {}


def test_single_person_means_no_pair_and_scene_facts_present():
    sc, arr = analyze_context(world({1: lambda t: (300, 150)}, n=90), CTX)
    assert sc["no_pair"] and arr == {} and sc["camera"]["moving"] is False
    assert sc["assumptions"]["fov_deg"] == CTX["fov_deg"] and sc["isolated_frac"] == 0.0


def test_save_and_load_roundtrip(tmp_path):
    leader = lambda t: (200 + 82.0 * t, 150)
    follower = lambda t: (200 + 82.0 * (t - 2.0), 150)
    sc, arr = analyze_context(world({1: follower, 2: leader}, n=240), CTX)
    save_context(sc, arr, tmp_path, "Test", "syn")
    sc2, arr2 = load_context(tmp_path, "Test", "syn")
    assert sc2["clip_events"] == sc["clip_events"]
    assert set(arr2) == set(arr)
    assert np.array_equal(arr2[(1, 2)]["follow_i_j"], arr[(1, 2)]["follow_i_j"])
    assert np.allclose(arr2[(1, 2)]["dist_m"], arr[(1, 2)]["dist_m"], equal_nan=True)


def test_clip_events_aggregates_over_pairs():
    sc = {"pairs": {"1_2": {"follow_i_j": {"count": 1, "seconds": 3.0}}, "1_3": {"follow_j_i": {"count": 2, "seconds": 5.0}, "contact": {"count": 1, "seconds": 0.4}}}}
    assert clip_events(sc) == {"follow": {"count": 3, "seconds": 8.0}, "contact": {"count": 1, "seconds": 0.4}}
