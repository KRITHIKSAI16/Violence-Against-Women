import numpy as np

from src.config import load_config
from src.context.features import analyze_context
from src.curated.relation import BENIGN, CONCERN, frame_labels, label_seconds, pair_spans, spans
from tests.test_context_features import world

CTX = load_config()["context"]
FPS = 30.0


def arrays(people, n=300, facing=None):
    sc, arr = analyze_context(world(people, n=n, facing=facing), CTX)
    return arr[(1, 2)]


def secs(a):
    return label_seconds(pair_spans(a, FPS))


def test_two_people_walking_side_by_side_are_walking_together_not_following():
    a = lambda t: (100 + 50.0 * t, 150, 100.0)
    b = lambda t: (100 + 50.0 * t, 150, 120.0)                 # same speed, about 1.4 m closer to the camera, not on the other's path
    s = secs(arrays({1: a, 2: b}))
    assert s.get("walking_together", 0) > 3.0 and "follows" not in s and "approaches" not in s and set(s) & BENIGN


def test_a_lagged_path_is_following():
    s = secs(arrays({1: lambda t: (160 + 50.0 * (t - 2.0), 150), 2: lambda t: (160 + 50.0 * t, 150)}))
    assert s.get("follows", 0) >= 3.0 and "follows" in CONCERN


def test_approach_from_behind_and_frontal_approach_are_different_labels():
    approach = lambda t: (330, 120, 140.0 - 12.0 * min(t, 3.0))
    still = lambda t: (300, 150, 100.0)
    behind = secs(arrays({1: approach, 2: still}, n=150, facing={1: "away", 2: "away"}))
    assert behind.get("approaches_from_behind", 0) > 0.5
    front = secs(arrays({1: approach, 2: still}, n=150, facing={1: "away", 2: "camera"}))
    assert front.get("approaches", 0) > 0.5 and "approaches_from_behind" not in front


def test_two_still_people_facing_each_other_are_standing_together():
    s = secs(arrays({1: lambda t: (300, 150, 130.0), 2: lambda t: (300, 150, 100.0)}, n=150, facing={1: "away", 2: "camera"}))
    assert s.get("standing_together", 0) > 3.0 and "approaches" not in s


def test_people_far_apart_and_still_are_just_apart():
    s = secs(arrays({1: lambda t: (100, 150, 100.0), 2: lambda t: (560, 150, 100.0)}, n=150))
    assert set(s) <= {"apart", "close_unclear"} and s.get("apart", 0) > 3.0


def test_span_cleaning_bridges_flicker_and_drops_tiny_runs():
    lab = np.array(["approaches"] * 20 + ["apart"] * 3 + ["approaches"] * 20 + ["walking_together"] * 2 + ["none"] * 5, dtype=object)
    sp = spans(lab, FPS)
    assert [s["label"] for s in sp] == ["approaches"] and sp[0]["start_s"] == 0.0 and abs(sp[0]["end_s"] - 43 / FPS) < 0.05
    d = np.linspace(5, 1, len(lab))
    assert spans(lab, FPS, d)[0]["dist_start_m"] == 5.0 and spans(lab, FPS, d)[0]["dist_end_m"] < 5.0
