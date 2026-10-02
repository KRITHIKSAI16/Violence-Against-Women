"""M8b tests: Algorithm 1 math on synthetic tracks (no GPU, no video)."""
import numpy as np
import pytest

from src.assm.interaction import (blocks_path, centroids, clip_curve, compute_pair_scores,
                                  load_curve, load_pairs, score_clip)
from src.assm.track_poses import pack_tracks
from src.config import load_config

CFG = load_config()["assm"]
W, H, FPS = 1000, 1000, 10.0
BODY = 100.0  # bbox height in px


def _row(f, tid, cx, cy):
    bbox = np.array([cx - 20, cy - BODY / 2, cx + 20, cy + BODY / 2], np.float32)
    return (f, tid, bbox, 0.9, np.zeros((17, 3), np.float32))  # no keypoints -> bbox centre


def _tracks(paths, n):
    """paths: {track_id: [(cx, cy) per frame]} -> packed tracks."""
    rows = [_row(f, tid, *p[f]) for tid, p in paths.items() for f in range(len(p))]
    return pack_tracks(rows, FPS, W, H, n)


def _line(x0, x1, y, n):
    return [(x0 + (x1 - x0) * f / (n - 1), y) for f in range(n)]


def test_centroid_prefers_keypoints_else_bbox():
    t = pack_tracks([_row(0, 1, 500, 500)], FPS, W, H, 1)
    assert np.allclose(centroids(t)[0], [500, 500])
    t["kpts"][0, :5] = [300, 400, 0.9]  # 5 confident keypoints at (300,400)
    assert np.allclose(centroids(t)[0], [300, 400])


def test_approaching_pair_has_positive_closing_speed_and_rising_score():
    n = 20
    t = _tracks({1: _line(200, 400, 500, n), 2: _line(800, 600, 500, n)}, n)
    rows = compute_pair_scores(t, CFG)
    assert len(rows) == n - 1  # first frame has no t-1
    assert all(r["v"] > 0 for r in rows)
    assert rows[-1]["d"] < rows[0]["d"]
    assert rows[-1]["score"] > rows[0]["score"]


def test_receding_pair_has_negative_closing_speed():
    n = 20
    t = _tracks({1: _line(400, 200, 500, n), 2: _line(600, 800, 500, n)}, n)
    assert all(r["v"] < 0 for r in compute_pair_scores(t, CFG))


def test_static_pair_zero_speed_and_closer_scores_higher():
    n = 6
    near = compute_pair_scores(_tracks({1: [(500, 500)] * n, 2: [(550, 500)] * n}, n), CFG)
    far = compute_pair_scores(_tracks({1: [(300, 500)] * n, 2: [(700, 500)] * n}, n), CFG)
    assert near[0]["v"] == pytest.approx(0) and far[0]["v"] == pytest.approx(0)
    assert near[0]["score"] > far[0]["score"]


def test_distance_is_in_body_heights():
    n = 3
    r = compute_pair_scores(_tracks({1: [(400, 500)] * n, 2: [(600, 500)] * n}, n), CFG)
    assert r[0]["d"] == pytest.approx(200 / BODY)


def test_pair_needs_both_ids_in_previous_frame():
    # id 2 appears only from frame 1 -> pair row starts at frame 2
    rows = [_row(0, 1, 400, 500), _row(1, 1, 400, 500), _row(1, 2, 600, 500),
            _row(2, 1, 400, 500), _row(2, 2, 600, 500)]
    r = compute_pair_scores(pack_tracks(rows, FPS, W, H, 3), CFG)
    assert [x["frame"] for x in r] == [2]


def test_single_person_gives_no_pairs():
    n = 5
    t = _tracks({1: [(500, 500)] * n}, n)
    assert compute_pair_scores(t, CFG) == []


def test_blocks_path_uses_nearest_edge():
    size = (W, H)
    # i near left edge; exit is x=0 at i's height. j between i and the edge blocks.
    assert blocks_path((100, 500), (50, 500), size, 100)
    assert not blocks_path((100, 500), (300, 500), size, 100)   # behind i, not between
    assert not blocks_path((100, 500), (50, 800), size, 100)    # beside the path, outside corridor


def test_b_term_changes_score_by_w3():
    n = 4
    t = _tracks({1: [(100, 500)] * n, 2: [(50, 500)] * n}, n)
    on = compute_pair_scores(t, {**CFG, "w3": 1.0})[0]
    off = compute_pair_scores(t, {**CFG, "w3": 0.0})[0]
    assert on["b"] == 1 and on["score"] == pytest.approx(off["score"] + 1.0)


def test_d_floor_keeps_score_finite_for_overlapping_people():
    n = 4
    r = compute_pair_scores(_tracks({1: [(500, 500)] * n, 2: [(500, 500)] * n}, n), CFG)
    assert np.isfinite(r[0]["score"]) and r[0]["score"] <= CFG["w1"] / CFG["d_floor"] + 1


def test_curve_is_max_over_pairs_and_zero_when_no_pair():
    n = 6
    t = _tracks({1: _line(200, 400, 500, n), 2: _line(800, 600, 500, n), 3: [(100, 900)] * n}, n)
    rows = compute_pair_scores(t, CFG)
    curve = clip_curve(rows, t)
    assert len(curve) == n and curve[0]["max_score"] == 0.0 and curve[0]["n_pairs"] == 0
    for c in curve[1:]:
        best = max(r["score"] for r in rows if r["frame"] == c["frame"])
        assert c["max_score"] == best and c["n_people"] == 3


def test_csv_roundtrip(tmp_path):
    n = 8
    t = _tracks({1: _line(200, 400, 500, n), 2: _line(800, 600, 500, n)}, n)
    np.savez_compressed(tmp_path / "x.npz", **t)
    rows, curve = score_clip(tmp_path / "x.npz", CFG, tmp_path / "scores", "Normal", "x")
    pr = load_pairs(tmp_path / "scores" / "Normal" / "x_pairs.csv")
    cr = load_curve(tmp_path / "scores" / "Normal" / "x_curve.csv")
    assert len(pr) == len(rows) and len(cr) == n
    assert pr[0]["score"] == pytest.approx(rows[0]["score"])
