"""Tests for Stage B logic (layout facts, door blocking, depth helpers). The models themselves are not needed: all logic is on arrays."""
import importlib.util

import numpy as np
import pytest

from src.config import load_config
from src.context.features import analyze_context, rank_pairs
from src.context.scene import (GROUP, SceneModels, blocks_door, group_at, group_lut, layout_facts, load_depth, load_layout,
                               person_depth, pinned_against, road_like_ids, save_layout)
from tests.test_context_features import world

CTX = load_config()["context"]
ID2LABEL = {0: "wall", 1: "building, edifice", 2: "sky", 3: "floor, flooring", 6: "road, route", 11: "sidewalk, pavement", 14: "door", 20: "car, auto",
            12: "person, individual", 5: "ceiling", 18: "plant, flora", 7: "bed", 99: "screen door, screen"}


def test_group_lut_uses_synonym_lists_in_class_names():
    lut = group_lut(ID2LABEL)
    assert lut[0] == GROUP["obstacle"] and lut[1] == GROUP["obstacle"] and lut[2] == GROUP["sky"] and lut[3] == GROUP["walkable"]
    assert lut[14] == GROUP["door"] and lut[99] == GROUP["door"] and lut[20] == GROUP["vehicle"] and lut[18] == GROUP["vegetation"]
    assert lut[12] == GROUP["other"] and lut[5] == GROUP["furniture"]
    assert road_like_ids(ID2LABEL) == {6, 11}


def _map(h=128, w=128):
    return np.zeros((h, w), np.uint8)


def test_layout_facts_indoor_outdoor_and_unknown():
    m = _map()
    m[:] = GROUP["walkable"]
    m[:50] = GROUP["furniture"]                         # lots of ceiling / furniture
    assert layout_facts(m, 0.9, 0.0, False)["place_type"] == "indoor"
    m2 = _map()
    m2[:30] = GROUP["sky"]
    assert layout_facts(m2, 0.9, 0.0, False)["place_type"] == "outdoor"
    assert layout_facts(_map(), 0.9, 0.0, False)["place_type"] == "unknown"
    assert layout_facts(_map(), 0.9, 0.3, False)["place_type"] == "outdoor"          # road-like pixels count as outdoor


def test_layout_reliability_needs_confidence_and_a_still_camera():
    m = _map()
    assert layout_facts(m, 0.9, 0, False)["reliable"]
    assert not layout_facts(m, 0.3, 0, False)["reliable"]            # a night alley segmented badly
    assert not layout_facts(m, 0.9, 0, True)["reliable"]             # moving camera: one merged layout is meaningless


def test_doors_are_found_as_tall_regions_and_tiny_or_wide_ones_are_ignored():
    m = _map()
    m[40:100, 60:76] = GROUP["door"]                  # tall door
    m[10:14, 5:60] = GROUP["door"]                    # wide thin strip (not a door)
    m[120:122, 0:2] = GROUP["door"]                   # speck
    f = layout_facts(m, 0.9)
    assert len(f["doors"]) == 1
    d = f["doors"][0]
    assert d["cx"] == pytest.approx(68 / 128, abs=0.02) and d["base_y"] == pytest.approx(100 / 128, abs=0.02)


def test_group_at_clamps_and_handles_outside():
    m = _map()
    m[64:, :] = GROUP["walkable"]
    assert group_at(m, 0.5, 0.9) == GROUP["walkable"] and group_at(m, 0.5, 0.1) == 0 and group_at(m, 1.5, 0.5) == 0


def test_blocks_door_geometry():
    door = (500.0, 400.0)
    me = (100.0, 400.0)
    assert blocks_door(me, (300.0, 405.0), door, 100.0)              # standing on the way to the door
    assert not blocks_door(me, (300.0, 300.0), door, 100.0)          # 100 px off the line (a whole body height)
    assert not blocks_door(me, (50.0, 400.0), door, 100.0)           # behind me
    assert not blocks_door(me, (520.0, 400.0), door, 100.0)          # past the door
    assert not blocks_door(me, me, me, 100.0)


def test_person_depth_samples_lower_torso():
    dm = np.zeros((200, 200), np.float32)
    dm[40:140, 80:120] = 0.8                       # a person-shaped nearer blob
    assert person_depth(dm, (80, 20, 120, 160)) == pytest.approx(0.8)
    assert person_depth(dm, (0, 0, 10, 10)) == pytest.approx(0.0)


def test_pinned_against_needs_obstacle_behind_at_the_same_depth():
    H = W = 200
    dm = np.full((H, W), 0.2, np.float32)
    dm[40:150, 80:120] = 0.6                        # the person
    labels = np.zeros((128, 128), np.uint8)
    labels[:, :30] = GROUP["obstacle"]                # a wall on the left of the image
    bbox = (80, 40, 120, 150)
    other_cx = 180.0                                  # the other person is on the right, so "behind me" is the LEFT side
    dm_wall_far = dm.copy()                           # wall far behind (depth 0.2 vs person 0.6): background wall, not adjacent
    assert not pinned_against(dm_wall_far, labels, bbox, other_cx)
    labels2 = np.zeros((128, 128), np.uint8)
    labels2[:, 30:55] = GROUP["obstacle"]             # wall right behind the person (x 47..86 px of 200)...
    dm_adj = dm.copy()
    dm_adj[:, 40:80] = 0.6                            # ...and at the same depth
    assert pinned_against(dm_adj, labels2, bbox, other_cx)
    assert not pinned_against(dm_adj, labels2, bbox, other_cx=20.0)       # the other person is on the wall side: the free side is behind me


def test_layout_roundtrip(tmp_path):
    m = _map()
    m[10:50, 10:20] = GROUP["door"]
    facts = layout_facts(m, 0.8)
    save_layout(tmp_path, "Normal", "c1", m, facts)
    m2, f2 = load_layout(tmp_path, "Normal", "c1")
    assert np.array_equal(m, m2) and f2["doors"][0]["area"] == facts["doors"][0]["area"]
    assert load_layout(tmp_path, "Normal", "missing") == (None, None) and load_depth(tmp_path, "Normal", "missing") is None


def test_rank_pairs_prefers_following_pairs_and_penalises_conversation():
    follower = lambda t: (160 + 50.0 * (t - 2.0), 150)
    leader = lambda t: (160 + 50.0 * t, 150)
    sc, _ = analyze_context(world({1: follower, 2: leader, 3: lambda t: (600, 150)}, n=300), CTX)
    assert rank_pairs(sc, 1) == ["1_2"]
    assert len(rank_pairs(sc, 5)) == len(sc["pairs"])


needs_models = pytest.mark.skipif(importlib.util.find_spec("transformers") is None, reason="transformers not installed")


@needs_models
def test_segformer_smoke_if_weights_are_cached():
    try:
        m = SceneModels("cpu")
        img = np.zeros((240, 320, 3), np.uint8)
        img[:, :, 1] = 120
        labels, conf, road = m.segment(img)
    except Exception as e:                      # offline machine / weights not downloaded
        pytest.skip(f"SegFormer weights unavailable: {type(e).__name__}")
    assert labels.shape == (128, 128) and 0.0 <= conf <= 1.0 and 0.0 <= road <= 1.0


def test_depth_samples_are_json_serialisable_and_agreement_counts_only_decisive_frames(tmp_path):
    import json
    from src.assm.track_poses import pack_tracks
    from src.context.scene import pair_depth_samples

    class FakeModels:
        def depth(self, img):
            d = np.zeros(img.shape[:2], np.float32)
            d[:, :200] = 0.8          # left half near, right half far
            d[:, 200:] = 0.2
            return d
    rows = []
    for f in range(40):
        for tid, x1 in ((1, 80), (2, 280)):
            rows.append((f, tid, np.array([x1, 40, x1 + 40, 190 if tid == 1 else 150], np.float32), 0.9, np.zeros((17, 3), np.float32)))
    t = pack_tracks(rows, 30.0, 400, 240, 40)
    video = tmp_path / "v.mp4"
    import cv2
    vw = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (400, 240))
    for _ in range(40):
        vw.write(np.zeros((240, 400, 3), np.uint8))
    vw.release()
    out = pair_depth_samples(t, {"fps": 30.0}, (1, 2), np.zeros((128, 128), np.uint8), video, FakeModels(), 6, 40)
    json.dumps(out)                                              # must not raise (numpy types)
    assert out["n_decisive"] == len(out["frames"]) == 6
    assert out["depth_agree_frac"] == 1.0                        # person 1: nearer by depth (0.8) AND taller box


def test_light_facts_is_not_fooled_by_a_single_bright_lamp():
    from src.context.scene import light_facts
    night = np.full((120, 160), 15, np.uint8)
    night[:, 88:] = 255                              # 45% of the frame is a glaring lamp / lit road: the MEAN (123) says 'lit', the median says 'dark'
    assert night.mean() > 70 and light_facts([night])["low_light"] is True
    day = np.full((120, 160), 140, np.uint8)
    assert light_facts([day])["low_light"] is False
    mostly_dark = np.concatenate([np.full((120, 100), 20, np.uint8), np.full((120, 60), 200, np.uint8)], axis=1)
    assert light_facts([mostly_dark])["low_light"] is True                # 62% of pixels very dark
    assert light_facts([])["low_light"] is None
    f = layout_facts(np.zeros((128, 128), np.uint8), 0.9, light=light_facts([night]))
    assert f["light"]["low_light"] is True


def test_story_prefers_layout_light_over_the_stage_a_mean_brightness_flag():
    from src.context.graph import _low_light
    scene = {"camera": {"night": False}}
    assert _low_light({"light": {"low_light": True}}, scene) is True       # lamp-lit night street: Stage A said 'not night', the robust statistic says low light
    assert _low_light(None, {"camera": {"night": True}}) is True
    assert _low_light({"light": {"low_light": None}}, scene) is False


def test_key_shot_range_picks_the_shot_of_the_key_pair_else_the_longest():
    from src.context.scene import key_shot_range
    from src.video.shots import SHOT_BASE as B
    tr = {"shot_starts": np.array([0, 40, 100], np.int32)}
    scene = {"n_frames": 300}
    assert key_shot_range(tr, scene, (B + 3, B + 5)) == (40, 99, 1)             # pair ids carry the shot index
    assert key_shot_range(tr, scene, (3, 5)) == (0, 39, 0)
    assert key_shot_range(tr, scene, None) == (100, 299, 2)                      # no pair: the longest shot
    assert key_shot_range({}, scene, (3, 5)) == (0, 299, 0)                      # old caches without shots: the whole clip
    assert key_shot_range(tr, scene, (9 * B + 1, 9 * B + 2))[2] == 2             # an out-of-range shot index falls back to the longest shot


def test_clip_layout_samples_only_inside_the_given_shot(tmp_path):
    import cv2
    from src.context.scene import clip_layout

    class SpyModels:
        def __init__(self):
            self.means = []

        def segment(self, img):
            self.means.append(int(img.mean()))
            return np.zeros((128, 128), np.uint8), 0.9, 0.0
    vw = cv2.VideoWriter(str(tmp_path / "v.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (64, 48))
    for k in range(120):
        vw.write(np.full((48, 64, 3), 40 if k < 60 else 200, np.uint8))             # shot A is dark, shot B is bright
    vw.release()
    spy = SpyModels()
    clip_layout(tmp_path / "v.mp4", (60, 119), spy, False, 5)
    assert spy.means and all(m > 150 for m in spy.means)                              # only bright (shot B) frames were segmented
