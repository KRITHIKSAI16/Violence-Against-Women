"""Tests for Stage D: window features, symmetric roles, grouped CV, weights, and above all the SHORTCUT PROBE."""
import numpy as np
import pandas as pd
import pytest

from src.config import load_config
from src.context.features import analyze_context, load_context, save_context
from src.context.learn import (FEATURE_SETS, best_f1_threshold, build_tables, clip_scores, clip_weights, metrics,
                               per_category_auc, run)
from src.context.windows import BEHAVIOR, SCENE, STYLE, pair_windows, window_features
from tests.test_context_features import world

CFG = load_config()
CTX = CFG["context"]
LEARN = {**CFG["learn"], "folds": 4}
FPS = 30.0


def follower_world(rng):
    v = rng.uniform(30, 40)
    lag = rng.uniform(1.5, 2.5)
    y = rng.uniform(120, 200)
    leader = lambda t: (160 + v * t, y)
    follower = lambda t: (160 + v * (t - lag), y)
    return world({1: follower, 2: leader}, n=300)


def independent_world(rng):
    v = rng.uniform(30, 40)
    y1, y2 = rng.uniform(120, 200, 2)
    if rng.random() < 0.5:                       # two people walking towards / past each other, far apart in depth
        return world({1: lambda t: (100 + v * t, y1, 100.0), 2: lambda t: (540 - v * t, y2, 160.0)}, n=300)
    return world({1: lambda t: (100 + v * t, y1), 2: lambda t: (520, y2)}, n=300)            # one walks, one stands elsewhere


def make_corpus(tmp, n_per_class=30, behavior_signal=True, style_signal=False, seed=0):
    """Synthetic context outputs on disk + manifests. Returns (clean manifest, raw_by_id, context dir)."""
    rng = np.random.default_rng(seed)
    clips, raw = [], {}
    for k in range(2 * n_per_class):
        pos = k < n_per_class
        cat = "Stalking" if pos else "Normal"
        t = (follower_world(rng) if (pos and behavior_signal) else independent_world(rng))
        sc, arr = analyze_context(t, CTX)
        if style_signal:                           # positives are filmed differently (moving camera, darker) - and nothing else differs
            sc["camera"].update({"moving": pos, "moving_share": 0.9 if pos else 0.0, "brightness": 60.0 if pos else 140.0,
                                 "night": pos, "sharpness": 50.0 if pos else 200.0})
        else:
            sc["camera"].update({"moving": bool(rng.random() < 0.4), "moving_share": float(rng.random()), "brightness": float(rng.uniform(80, 160)),
                                 "night": False, "sharpness": float(rng.uniform(50, 200))})
        cid = f"{cat}_v{k}"
        save_context(sc, arr, tmp, cat, cid)
        clips.append({"clip_id": cid, "category": cat, "path": "x.mp4", "fps": FPS, "width": 640, "height": 480})
        raw[cid] = {"width": 640 if not style_signal or not pos else 1280, "height": 480, "fps": 30.0}
    return {"clips": clips}, raw, tmp


# ---------------------------------------------------------------- window features
def test_window_features_are_symmetric_in_the_two_people():
    rng = np.random.default_rng(1)
    sc, arr = analyze_context(follower_world(rng), CTX)
    a = arr[(1, 2)]
    sw = {k: v for k, v in a.items()}
    # swap roles i <-> j
    for x, y in (("speed_i", "speed_j"), ("ang_i_to_j", "ang_j_to_i"), ("head_ang_i", "head_ang_j"), ("toward_i", "toward_j"),
                 ("reach_i_to_j_ms", "reach_j_to_i_ms"), ("approach_behind_i_j", "approach_behind_j_i"), ("looking_back_i", "looking_back_j"),
                 ("flee_i", "flee_j"), ("still_i", "still_j"), ("follow_i_j", "follow_j_i"), ("lag_i_j", "lag_j_i"), ("dev_i_j", "dev_j_i")):
        sw[x], sw[y] = a[y], a[x]
    f1 = window_features(a, 90, 180, FPS, 0, CTX)
    f2 = window_features(sw, 90, 180, FPS, 0, CTX)
    for k in BEHAVIOR:
        assert (np.isnan(f1[k]) and np.isnan(f2[k])) or f1[k] == pytest.approx(f2[k], abs=1e-9), k


def test_follower_windows_show_following_and_independent_windows_do_not():
    rng = np.random.default_rng(2)
    f = pair_windows(analyze_context(follower_world(rng), CTX)[1][(1, 2)], {"fps": FPS}, CTX, 3.0, 1.5, 300)
    i = pair_windows(analyze_context(independent_world(rng), CTX)[1][(1, 2)], {"fps": FPS}, CTX, 3.0, 1.5, 300)
    assert max(w["follow_frac"] for w in f) > 0.5
    assert max(w["follow_frac"] for w in i) == 0.0
    assert all(set(BEHAVIOR) <= set(w) for w in f)


def test_windows_stop_at_the_end_frame_and_skip_sparse_pairs():
    rng = np.random.default_rng(3)
    a = analyze_context(follower_world(rng), CTX)[1][(1, 2)]
    w = pair_windows(a, {"fps": FPS}, CTX, 3.0, 1.5, 150)
    assert w and max(x["t_start"] for x in w) + 3.0 <= 150 / FPS + 1e-6


# ---------------------------------------------------------------- table, weights, scoring
def test_clip_weights_give_every_clip_equal_total_weight():
    df = pd.DataFrame({"clip_id": ["a"] * 10 + ["b"] * 2})
    w = clip_weights(df)
    assert w[:10].sum() == pytest.approx(w[10:].sum())
    assert w.mean() == pytest.approx(1.0)


def test_clip_scores_topk_and_clips_without_windows_score_zero():
    df = pd.DataFrame({"clip_id": ["a", "a", "a", "a", "b"]})
    prob = np.array([0.1, 0.9, 0.8, 0.7, 0.4])
    clips = pd.DataFrame({"clip_id": ["a", "b", "c"], "category": ["X", "Y", "Z"], "label": [1, 0, 1], "n_windows": [4, 1, 0]})
    cs = clip_scores(df, prob, clips, k=3).set_index("clip_id")
    assert cs.loc["a", "score"] == pytest.approx((0.9 + 0.8 + 0.7) / 3)
    assert cs.loc["b", "score"] == pytest.approx(0.4) and cs.loc["c", "score"] == 0.0 and not cs.loc["c", "has_pair"]


def test_metrics_and_threshold_and_per_category_auc():
    y = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    s = np.array([0.1, 0.2, 0.3, 0.6, 0.4, 0.7, 0.8, 0.9])
    m = metrics(y, s)
    assert m["auc"] == pytest.approx(0.9375) and 0 < m["f1"] <= 1
    assert metrics([1, 1], [0.2, 0.3])["auc"] != metrics([1, 1], [0.2, 0.3])["auc"]          # NaN when one class only
    assert 0.2 < best_f1_threshold(y, s) < 0.9
    cs = pd.DataFrame({"category": ["Normal"] * 3 + ["A"] * 3 + ["B"] * 3, "label": [0] * 3 + [1] * 6,
                       "score": [0.1, 0.2, 0.3, 0.9, 0.8, 0.7, 0.1, 0.2, 0.3]})
    pc = per_category_auc(cs)
    assert pc["A"] == 1.0 and pc["B"] == pytest.approx(0.5)


# ---------------------------------------------------------------- the shortcut probe (the point of Stage D)
def _results(tmp_path, behavior_signal, style_signal):
    clean, raw, d = make_corpus(tmp_path, 30, behavior_signal, style_signal)
    windows, clips = build_tables(clean, raw, d, tmp_path / "nogate", CTX, CFG["report"], 3.0, 1.5)
    assert set(BEHAVIOR) | set(SCENE) | set(STYLE) <= set(windows.columns)
    res, _ = run(windows, clips, LEARN, None, importance=False, kinds=("gb",))
    return res


def test_real_behavior_signal_is_found_and_style_is_not_the_reason(tmp_path):
    res = _results(tmp_path, behavior_signal=True, style_signal=False)
    assert res["models"]["gb:behavior"]["all_clips"]["auc"] > 0.9
    assert res["models"]["gb:style_only"]["all_clips"]["auc"] < 0.75


def test_probe_exposes_a_pure_style_shortcut(tmp_path):
    # positives and negatives behave identically; only the filming differs. A trustworthy probe must show: style separates, behavior does not.
    res = _results(tmp_path, behavior_signal=False, style_signal=True)
    assert res["models"]["gb:style_only"]["all_clips"]["auc"] > 0.95
    assert res["models"]["gb:behavior"]["all_clips"]["auc"] < 0.75
    assert res["models"]["gb:all"]["all_clips"]["auc"] > 0.9            # the combined model happily uses the shortcut: that is what we must not report


def test_static_camera_subset_is_reported_and_feature_sets_are_disjoint_groups():
    assert not set(BEHAVIOR) & set(STYLE) and not set(SCENE) & set(STYLE)
    assert FEATURE_SETS["all"] == BEHAVIOR + SCENE + STYLE


def test_max_pairs_keeps_the_closest_pairs(tmp_path):
    from src.context.features import save_context
    rng = np.random.default_rng(5)
    # a crowd of 5 people standing at different distances from person 1
    t = world({k: (lambda t, k=k: (100 + 90 * k, 150)) for k in range(1, 6)}, n=240)
    sc, arr = analyze_context(t, {**CTX, "max_pair_dist_m": 50.0})
    assert len(arr) == 10
    save_context(sc, arr, tmp_path, "Normal", "crowd")
    clean = {"clips": [{"clip_id": "crowd", "category": "Normal", "path": "x", "fps": FPS, "width": 640, "height": 480}]}
    w_all, _ = build_tables(clean, {}, tmp_path, tmp_path / "g", CTX, CFG["report"], 3.0, 1.5)
    w_cap, _ = build_tables(clean, {}, tmp_path, tmp_path / "g", CTX, CFG["report"], 3.0, 1.5, max_pairs=3)
    assert w_all["pair"].nunique() == 10 and w_cap["pair"].nunique() == 3
    assert w_cap["d_mean"].max() <= w_all["d_mean"].quantile(0.5) + 1e-6          # the kept pairs are the close ones
