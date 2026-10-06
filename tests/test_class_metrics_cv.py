"""Classification metrics, models, fusion and cross-validation: known answers and leakage checks on synthetic data."""
from unittest import mock

import numpy as np
import pytest

from src.classcommon import cv as ccv
from src.classcommon import edgecases
from src.classcommon import metrics as M

CV = {"outer_folds": 4, "repeats": 3, "inner_folds": 3, "seed": 7, "pca_components": 4, "C": 0.5, "min_per_class": 10}


def test_threshold_metrics_hand_computed():
    y = [1, 1, 1, 0, 0, 0, 0, 0]
    pred = [1, 1, 0, 1, 0, 0, 0, 0]                                           # tp 2, fn 1, fp 1, tn 4
    m = M.threshold_metrics(y, pred)
    assert (m["tp"], m["fn"], m["fp"], m["tn"]) == (2, 1, 1, 4)
    assert m["accuracy"] == pytest.approx(0.75) and m["recall"] == pytest.approx(2 / 3) and m["precision"] == pytest.approx(2 / 3)
    assert m["specificity"] == pytest.approx(0.8) and m["balanced_accuracy"] == pytest.approx((2 / 3 + 0.8) / 2)
    assert m["f1"] == pytest.approx(2 / 3) and m["mcc"] == pytest.approx(7 / 15)


def test_degenerate_threshold_metrics():
    m = M.threshold_metrics([1, 1, 0, 0], [0, 0, 0, 0])                       # nothing predicted positive
    assert m["precision"] == 0.0 and m["recall"] == 0.0 and m["f1"] == 0.0 and m["mcc"] == 0.0 and m["specificity"] == 1.0
    assert M.threshold_metrics([1, 0], [1, 0])["mcc"] == 1.0 and M.threshold_metrics([1, 0], [0, 1])["mcc"] == -1.0


def test_score_metrics_hand_computed():
    y = [1, 1, 1, 0, 0, 0, 0, 0]
    p = [0.9, 0.8, 0.4, 0.6, 0.3, 0.2, 0.1, 0.05]
    s = M.score_metrics(y, p)
    assert s["auc"] == pytest.approx(14 / 15)                                  # 14 of the 15 violent/non-violent pairs are ranked correctly
    assert s["brier"] == pytest.approx(np.mean((np.array(p) - np.array(y)) ** 2))
    assert s["pr_auc"] == pytest.approx((1 / 1 + 2 / 2 + 3 / 4) / 3)           # ranks of the violent clips: 1st, 2nd and 4th -> precision 1, 1, 3/4
    assert np.isnan(M.score_metrics([1, 1], [0.2, 0.9])["auc"])
    assert M.ece([1, 1, 1, 1, 0], [0.9] * 5) == pytest.approx(0.1)             # one bin: observed 0.8, predicted 0.9
    assert M.ece([1, 0], [1.0, 0.0]) == 0.0


def test_bootstrap_interval_contains_the_estimate_and_is_reproducible():
    rng = np.random.default_rng(0)
    y = np.r_[np.ones(40), np.zeros(40)].astype(int)
    p = np.clip(y * 0.3 + rng.normal(0.35, 0.2, 80), 0, 1)
    pred = (p >= 0.5).astype(int)
    a = M.bootstrap(y, p, pred, n=200, seed=1)
    assert a == M.bootstrap(y, p, pred, n=200, seed=1)
    auc = M.score_metrics(y, p)["auc"]
    assert a["auc"][0] <= auc <= a["auc"][1] and 0 <= a["f1"][0] <= a["f1"][1] <= 1


def test_per_category_recall_and_same_group_auc():
    cats = ["A", "A", "B", "B", "Normal", "Normal"]
    y = [1, 1, 1, 1, 0, 0]
    pred = [1, 0, 1, 1, 0, 1]
    assert M.per_category_recall(cats, y, pred) == {"A": (2, 0.5), "B": (2, 1.0)}
    groups = ["r1", "r1", "r1", "r2", "r2", "r2"]                              # r1 holds only violent clips; r2 holds both classes
    n1, n0, auc = M.same_group_auc(groups, [1, 1, 1, 1, 0, 0], [0.9, 0.8, 0.7, 0.6, 0.1, 0.2])
    assert (n1, n0) == (1, 2) and auc == 1.0                                   # only group r2 holds both classes
    assert M.same_group_auc(["a", "a"], [1, 1], [0.5, 0.5]) == (0, 0, pytest.approx(float("nan"), nan_ok=True))


def test_choose_threshold_and_adaptive_pca():
    assert ccv.choose_threshold([0, 0, 1, 1], [0.1, 0.2, 0.3, 0.4]) == pytest.approx(0.25)
    assert ccv.choose_threshold([0, 1], [0.6, 0.7]) == pytest.approx(0.65)
    t = ccv.choose_threshold([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9])             # any threshold in (0.2, 0.8) is perfect: the one closest to 0.5 wins
    assert 0.2 < t < 0.8 and abs(t - 0.5) <= 0.3
    x = np.random.default_rng(0).normal(size=(3, 10))
    assert ccv.AdaptivePCA(4).fit(x).transform(np.zeros((1, 10))).shape == (1, 2)           # min(4, 3 - 1, 10)
    assert np.allclose(ccv.AdaptivePCA(16).fit(x).transform(x), x)                          # 10 columns <= 16: passed through


def _toy(n=60, seed=0):
    rng = np.random.default_rng(seed)
    y = np.array([1] * (n // 2) + [0] * (n // 2))
    sig = y[:, None] * 2.0 + rng.normal(0, 1, (n, 5))
    noise = rng.normal(0, 1, (n, 5))
    txt = np.array([("follows closely behind" if (yy and rng.random() < 0.9) or (not yy and rng.random() < 0.1) else "walks far apart") for yy in y], dtype=object)
    return ccv.Data({"sig": sig, "noise": noise, "txt": txt}, {"sig": "num", "noise": "num", "txt": "text"}, y, [f"c{i}" for i in range(n)])


def test_cross_validate_signal_noise_and_fusions():
    d = _toy()
    methods = [ccv.Method("sig", "single", ["sig"]), ccv.Method("noise", "single", ["noise"]), ccv.Method("txt", "single", ["txt"]),
               ccv.Method("late", "late", ["sig", "txt"]), ccv.Method("stack", "stack", ["sig", "txt"]), ccv.Method("early", "early", ["sig", "noise"])]
    res = ccv.cross_validate(methods, d, np.arange(len(d.y)), CV)
    auc = {k: M.score_metrics(d.y, v["proba"])["auc"] for k, v in res.items()}
    assert auc["sig"] > 0.9 and auc["txt"] > 0.7 and auc["late"] > 0.9 and auc["stack"] > 0.9 and auc["early"] > 0.85
    assert 0.25 < auc["noise"] < 0.75                                          # no signal, no leak: chance level
    for v in res.values():
        assert v["proba"].shape == (60,) and set(np.unique(v["pred"])) <= {0, 1} and len(v["auc_per_repeat"]) == 3
    acc = M.threshold_metrics(d.y, res["sig"]["pred"])["accuracy"]
    assert acc > 0.8


def test_duplicates_never_sit_in_a_training_and_a_test_fold_together():
    d = _toy(40)
    groups = np.repeat(np.arange(20), 2)                                      # clips 2k and 2k+1 are duplicates
    seen = []
    real = ccv.fold_predict

    def spy(m, data, tr, te, cv):
        seen.append((set(groups[tr]), set(groups[te])))
        return real(m, data, tr, te, cv)
    with mock.patch.object(ccv, "fold_predict", spy):
        ccv.cross_validate([ccv.Method("sig", "single", ["sig"])], d, groups, {**CV, "repeats": 2})
    assert seen and all(not (a & b) for a, b in seen)
    assert all(len(b) >= 1 for _, b in seen)


def test_inner_threshold_falls_back_when_a_class_is_tiny():
    d = _toy(20)
    tr = np.array([0] + list(range(10, 20)))                                  # one violent clip in the training part: no inner folds
    p, thr = ccv.fold_predict(ccv.Method("sig", "single", ["sig"]), d, tr, np.array([1, 2]), CV)
    assert thr == 0.5 and p.shape == (2,)
    ps = ccv.method_proba(ccv.Method("stack", "stack", ["sig", "txt"]), d, tr, np.array([1, 2]), CV)          # stacking falls back to the mean
    assert ps.shape == (2,) and np.all((0 <= ps) & (ps <= 1))


def test_cross_validate_needs_two_clips_per_class():
    d = ccv.Data({"a": np.zeros((5, 2))}, {"a": "num"}, np.array([1, 0, 0, 0, 0]), list("abcde"))
    with pytest.raises(ValueError):
        ccv.cross_validate([ccv.Method("a", "single", ["a"])], d, np.arange(5), CV)


def test_missing_values_are_imputed_inside_the_fold():
    d = _toy()
    d.blocks["sig"] = d.blocks["sig"].copy()
    d.blocks["sig"][::7] = np.nan
    res = ccv.cross_validate([ccv.Method("sig", "single", ["sig"])], d, np.arange(60), {**CV, "repeats": 1})
    assert np.isfinite(res["sig"]["proba"]).all()


def test_edgecases_dup_groups_enough_flags_leak(tmp_path, make_video):
    h = {"a": 0b0000, "b": 0b0011, "c": 0b1111_1111_1111, "d": 0b1111_0000_0000_0000, "e": None}
    g = edgecases.dup_groups(h, max_dist=2)
    assert g["a"] == g["b"] and g["b"] != g["c"] and g["d"] not in (g["a"], g["c"]) and g["e"] not in (g["a"], g["c"], g["d"])
    assert len(set(g.values())) == 4                                                          # {a, b}, {c}, {d}, {e}
    chain = edgecases.dup_groups({"a": 0b000, "b": 0b011, "c": 0b111}, max_dist=2)          # a~b, b~c: one group although a and c differ in 3 bits
    assert len(set(chain.values())) == 1
    v = make_video(tmp_path / "a.mp4", n_frames=20)
    w = tmp_path / "copy.mp4"
    w.write_bytes(v.read_bytes())
    assert edgecases.ahash(v) == edgecases.ahash(w) and edgecases.ahash(tmp_path / "missing.mp4") is None
    assert edgecases.check_enough([1] * 3 + [0] * 20, 10)[0] is False and "3 violent and 20 non-violent" in edgecases.check_enough([1] * 3 + [0] * 20, 10)[1]
    assert edgecases.check_enough([1] * 10 + [0] * 10, 10)[0] is True
    recs = [{"clip_id": "a", "duration_s": 1.0, "result": {"no_pair": False}}, {"clip_id": "b", "duration_s": 5.0, "result": {"no_pair": True}}, {"clip_id": "c", "duration_s": 5.0, "result": None}]
    assert edgecases.flag_records(recs) == {"a": ["short"], "b": ["no_pair"], "c": ["no_result"]}
    assert edgecases.leak_scan({"x": "he will attack", "y": "walks"}, r"\b(attack|fight)\b") == {"x": ["attack"]}
