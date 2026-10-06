import numpy as np

from src.curated.evaluate import auc, bootstrap_auc, evaluate_all, markdown, normal_vs_pre, sign_test_p, trend


def res(cid, cat, normal, last=None, maxw=None, windows=None, dur=8.0, size=(640, 360), pair=True, typ="approach"):
    inter = None
    if pair:
        inter = {"concern_last_s": last, "concern_max_window": maxw, "concern_windows": windows or [], "duration_s": dur, "leadup": {"type": typ}}
    return {"clip_id": cid, "category": cat, "is_normal": normal, "no_pair": not pair, "duration_s": dur, "interaction": inter, "reason": "one_person",
            "raw": {"raw_width": size[0], "raw_height": size[1], "raw_fps": 25.0}, "camera": {"moving": False, "brightness": 100.0, "sharpness": 50.0}}


def test_auc_basics_and_bootstrap():
    assert auc([3, 4], [1, 2]) == 1.0 and auc([1, 2], [3, 4]) == 0.0 and auc([1], [1]) == 0.5 and np.isnan(auc([], [1]))
    a, lo, hi = bootstrap_auc(np.arange(10) + 20.0, np.arange(10) * 1.0)
    assert a == 1.0 and lo == 1.0 and hi == 1.0
    a, lo, hi = bootstrap_auc([1, 2, 3, 4, 5, 6], [2, 3, 4, 5, 6, 7])
    assert lo < a < hi


def test_sign_test():
    assert abs(sign_test_p(10, 10) - 2 / 1024) < 1e-9 and sign_test_p(5, 10) == 1.0 and np.isnan(sign_test_p(0, 0))


def test_trend_compares_last_three_seconds_with_the_earlier_part_of_the_same_clip():
    w = [(0.0, 0.0), (0.5, 0.0), (1.0, 0.1), (3.0, 2.0), (3.5, 2.5), (4.0, 2.0)]
    rs = [res("a", "Stalking", False, windows=w, dur=7.0), res("b", "Stalking", False, windows=[(0.0, 1.0), (4.5, 0.2)], dur=7.0),
          res("c", "Stalking", False, windows=w, dur=2.0, pair=False), res("d", "Normal", True, windows=w, dur=7.0)]
    t = trend(rs)
    assert t["clips"] == 2 and t["higher_in_last"] == 1 and len(t["diffs"]) == 2


def test_normal_vs_pre_verdict_is_conservative():
    rng = np.random.default_rng(1)
    mixed = lambda i: (640, 360) if i % 2 else (1280, 720)
    rs = [res(f"v{i}", "Stalking", False, last=2.0 + rng.normal(0, .3), size=mixed(i)) for i in range(12)]
    rs += [res(f"n{i}", "Normal", True, maxw=0.0 + rng.normal(0, .3), size=mixed(i)) for i in range(12)]
    o = normal_vs_pre(rs)
    assert o["auc"] > 0.95 and o["auc_low"] > 0.5 and o["same_resolution"]["violent"] == 12 and o["style_only_auc"] < 0.7
    assert o["verdict"].startswith("The geometric concern score is higher before violence") and "holds among clips of the same resolution" in o["verdict"]
    assert "Too few" in normal_vs_pre(rs[:2] + rs[12:14])["verdict"]
    same = [res(f"v{i}", "Stalking", False, last=1.0 + rng.normal(0, 1)) for i in range(10)] + [res(f"n{i}", "Normal", True, maxw=1.0 + rng.normal(0, 1)) for i in range(10)]
    assert "does not clearly separate" in normal_vs_pre(same)["verdict"]


def test_style_only_separation_is_called_out():
    rs = [res(f"v{i}", "Stalking", False, last=2.0 + 0.01 * i, size=(640, 360)) for i in range(10)] + [res(f"n{i}", "Normal", True, maxw=0.0 + 0.01 * i, size=(1920, 1080)) for i in range(10)]
    o = normal_vs_pre(rs)
    assert o["style_only_auc"] == 1.0 and o["same_resolution"]["violent"] == 0
    assert "filming style alone separates them" in o["verdict"]


def test_markdown_and_full_evaluation_run_on_a_small_set():
    rs = [res("a", "Stalking", False, last=1.0, windows=[(0.0, 0.0), (5.0, 1.0)], dur=7.0), res("b", "Stalking", False, pair=False), res("c", "Normal", True, maxw=0.1)]
    ev = evaluate_all(rs)
    assert ev["coverage"]["Stalking"]["with_pair"] == 1 and ev["leadup_types"]["Stalking"]["approach"] == 1 and "no pair: one_person" in ev["leadup_types"]["Stalking"]
    md = markdown(ev)
    assert "## Pre-violence vs Normal" in md and "different source" in md
