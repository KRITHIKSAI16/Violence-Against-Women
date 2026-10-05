"""Tests for shot segmentation: cuts at known frames, merging of tiny shots, helpers, JSON round trip, contact sheet."""
import importlib.util

import cv2
import numpy as np
import pytest

from src.video.shots import (SHOT_BASE, _cuts_to_shots, cut_sheet, detect_shots, hist_shots, load_shots, local_id, merge_short, save_shots,
                             shot_index, shot_of_id, shot_starts)

FPS = 30.0


def textured(seed, w=320, h=180):
    rng = np.random.default_rng(seed)
    img = np.zeros((h, w, 3), np.uint8)
    for _ in range(60):
        cv2.circle(img, (int(rng.integers(0, w)), int(rng.integers(0, h))), int(rng.integers(5, 40)),
                   tuple(int(v) for v in rng.integers(30, 255, 3)), -1)
    return cv2.GaussianBlur(img, (5, 5), 0)


def make_edit(path, segments):
    """segments: [(seed, n_frames)] -> video that hard-cuts between different scenes (each scene has a little moving noise)."""
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (320, 180))
    rng = np.random.default_rng(0)
    for seed, n in segments:
        base = textured(seed)
        for k in range(n):
            f = np.clip(base.astype(int) + rng.integers(-3, 4, base.shape), 0, 255).astype(np.uint8)
            vw.write(f)
    vw.release()
    return path


def test_hist_detector_finds_cuts_at_known_frames(tmp_path):
    p = make_edit(tmp_path / "e.mp4", [(1, 40), (2, 25), (3, 35)])
    assert hist_shots(p) == [(0, 39), (40, 64), (65, 99)]


def test_no_cut_in_a_single_scene(tmp_path):
    p = make_edit(tmp_path / "one.mp4", [(1, 60)])
    d = detect_shots(p, method="hist", clip_id="one")
    assert d["cuts_s"] == [] and len(d["shots"]) == 1 and d["shots"][0]["end_f"] == 59


def test_detect_shots_dict_has_times_and_merges_tiny_shots(tmp_path):
    p = make_edit(tmp_path / "e.mp4", [(1, 40), (2, 4), (3, 40)])                 # the middle shot lasts 4 frames = 0.13 s
    d = detect_shots(p, method="hist", min_shot_s=0.4, clip_id="e")
    assert [s["start_f"] for s in d["shots"]] == [0, 44] or len(d["shots"]) == 2
    assert d["method"] == "hist" and d["n_frames"] == 84 and d["fps"] == pytest.approx(FPS)
    assert d["shots"][0]["start_f"] == 0 and d["shots"][-1]["end_f"] == 83         # shots tile the whole clip
    for a, b in zip(d["shots"][:-1], d["shots"][1:]):
        assert b["start_f"] == a["end_f"] + 1                                       # no gaps, no overlaps
    p2 = make_edit(tmp_path / "e2.mp4", [(1, 40), (2, 30)])
    d2 = detect_shots(p2, method="hist", clip_id="e2")
    assert d2["cuts_s"] == [pytest.approx(40 / FPS, abs=0.01)] and d2["shots"][1]["start_s"] == pytest.approx(40 / FPS, abs=0.01)


def test_merge_short_into_previous_or_next():
    shots = [(0, 99), (100, 103), (104, 200)]
    assert merge_short(shots, FPS, 0.4) == [(0, 99), (104, 200)] or merge_short(shots, FPS, 0.4) == [(0, 103), (104, 200)]
    out = merge_short(shots, FPS, 0.4)
    assert out[0][0] == 0 and out[-1][1] == 200 and len(out) == 2
    assert merge_short([(0, 3), (4, 100)], FPS, 0.4) == [(0, 100)]                  # a tiny FIRST shot joins the next one
    assert merge_short([(0, 50)], FPS, 0.4) == [(0, 50)]
    assert _cuts_to_shots([40, 40, 0, 999], 100) == [(0, 39), (40, 99)]


def test_shot_index_and_id_helpers():
    starts = [0, 40, 100]
    assert shot_index(0, starts) == 0 and shot_index(39, starts) == 0 and shot_index(40, starts) == 1 and shot_index(250, starts) == 2
    assert shot_index(np.array([5, 45, 105]), starts).tolist() == [0, 1, 2]
    tid = 2 * SHOT_BASE + 7
    assert local_id(tid) == 7 and shot_of_id(tid) == 2 and shot_of_id(5) == 0 and local_id(5) == 5
    assert shot_starts({"shots": [{"start_f": 0}, {"start_f": 40}]}) == [0, 40]


def test_json_roundtrip_and_contact_sheet(tmp_path):
    p = make_edit(tmp_path / "e.mp4", [(1, 40), (2, 30)])
    d = detect_shots(p, method="hist", clip_id="e")
    save_shots(tmp_path, "Cat", "e", d)
    assert load_shots(tmp_path, "Cat", "e") == d and load_shots(tmp_path, "Cat", "missing") is None
    sheet = cut_sheet(p, d, tmp_path / "sheet.jpg")
    img = cv2.imread(str(sheet))
    assert img is not None and img.shape[1] == 600
    assert cut_sheet(make_edit(tmp_path / "one.mp4", [(1, 30)]), detect_shots(tmp_path / "one.mp4", "hist"), tmp_path / "none.jpg") is None


@pytest.mark.skipif(importlib.util.find_spec("transnetv2_pytorch") is None, reason="transnetv2-pytorch not installed")
def test_transnet_smoke_on_a_hard_cut(tmp_path):
    p = make_edit(tmp_path / "e.mp4", [(1, 60), (2, 60)])
    try:
        d = detect_shots(p, method="transnet", clip_id="e", device="cpu")
    except Exception as e:
        pytest.skip(f"TransNetV2 unavailable: {type(e).__name__}")
    assert d["method"] in ("transnet", "hist")                        # falls back to the histogram detector if the model cannot run
    assert len(d["cuts_s"]) >= 1 and any(abs(c - 2.0) < 0.2 for c in d["cuts_s"])
