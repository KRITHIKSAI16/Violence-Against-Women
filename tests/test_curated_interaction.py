import numpy as np
import pytest

from src.config import load_config
from src.context.features import analyze_context
from src.curated.interaction import concern_series, leadup_type, summarize, window_scores
from tests.test_context_features import world

CTX = load_config()["context"]
FPS = 30.0


def run(people, n=240, facing=None):
    sc, arr = analyze_context(world(people, n=n, facing=facing), CTX)
    return summarize(arr[(1, 2)], FPS, (1, 2))


def test_approach_from_behind_is_described_with_numbers_and_roles():
    approach = lambda t: (330, 120, 140.0 - 12.0 * min(t, 3.0))
    s = run({1: approach, 2: lambda t: (300, 150, 100.0)}, n=150, facing={1: "away", 2: "away"})
    assert s["leadup"]["type"] == "approach from behind"
    L = s["last"]["3"]
    assert L["actor"] == 1 and L["target"] == 2 and L["target_faces_away_deg"] > 110 and L["net_closing_m"] is not None
    txt = " ".join(s["lines"])
    assert "id1 approaches id2 from behind" in txt and "facing away" in txt and "Lead-up type" in txt


def test_following_walking_together_and_far_apart_get_different_types():
    f = run({1: lambda t: (160 + 50.0 * (t - 2.0), 150), 2: lambda t: (160 + 50.0 * t, 150)}, n=300)
    assert f["leadup"]["type"] == "following" and "id1 follows id2" in " ".join(f["lines"])
    w = run({1: lambda t: (100 + 50.0 * t, 150, 100.0), 2: lambda t: (100 + 50.0 * t, 150, 120.0)}, n=300)
    assert w["leadup"]["type"] == "walking together"
    far = run({1: lambda t: (100, 150, 100.0), 2: lambda t: (560, 150, 100.0)}, n=150)
    assert far["leadup"]["type"] == "no visible lead-up" and far["min_dist_m"] > 3


def test_concern_is_high_for_the_approach_and_low_for_walking_together():
    approach = lambda t: (330, 120, 140.0 - 12.0 * min(t, 3.0))
    sc, arr = analyze_context(world({1: approach, 2: lambda t: (300, 150, 100.0)}, n=150, facing={1: "away", 2: "away"}), CTX)
    s_app, _ = concern_series(arr[(1, 2)], FPS)
    sc, arr = analyze_context(world({1: lambda t: (100 + 50.0 * t, 150, 100.0), 2: lambda t: (100 + 50.0 * t, 150, 120.0)}, n=240), CTX)
    s_walk, _ = concern_series(arr[(1, 2)], FPS)
    peak = lambda x: max(v for _, v in window_scores(x, FPS, 1.0, 0.5))
    assert peak(s_app) > 1.0 and np.nanmean(s_walk) < 0 and peak(s_app) > peak(s_walk) + 1.5       # the approach peaks; walking together never does


def test_windows_skip_missing_data_and_leadup_rule_order():
    s = np.array([1.0] * 60 + [np.nan] * 60)
    w = window_scores(s, FPS, 2.0, 0.5)
    assert w and w[0][0] == 0.0 and all(t <= 2.0 for t, _ in w)
    L = {"seconds": 3.0, "contact_frac": 0.0, "net_closing_m": 1.2, "dist_end_m": 1.0}
    assert leadup_type({"follows": 2.0}, L, 5.0)[0] == "following"          # an earlier rule wins over a later one
    assert leadup_type({}, L, 5.0)[0] == "closing in"
    assert leadup_type({}, {**L, "net_closing_m": 0.1}, 1.0)[0] == "already close at the start"
