"""M9 tests: behavior states on synthetic people with known motion (fps 30, body height 100 px)."""
import json

import numpy as np
import pytest

from src.assm.gate import (APPROACH, CORNER, ESCALATION, FOLLOW, HOVER, analyze_clip, load_gate,
                           load_states, runs, write_outputs)
from src.assm.track_poses import pack_tracks
from src.config import load_config

_cfg = load_config()
CFG = {**_cfg["gate"], "kpt_conf": _cfg["assm"]["kpt_conf"], "corridor": _cfg["assm"]["corridor"]}
FPS, W, H, BODY = 30.0, 1000, 1000, 100.0
KP = np.zeros((17, 3), np.float32)


def tracks(paths, secs):
    """paths: {id: fn(t_seconds) -> (x, y)}; one row per frame for each id."""
    n = int(secs * FPS)
    rows = []
    for tid, fn in paths.items():
        for f in range(n):
            x, y = fn(f / FPS)
            rows.append((f, tid, np.array([x - 20, y - BODY / 2, x + 20, y + BODY / 2], np.float32), 0.9, KP))
    return pack_tracks(rows, FPS, W, H, n)


def states(g):
    return {s["state"] for s in g["segments"]}


def run(paths, secs):
    g, rows = analyze_clip(tracks(paths, secs), CFG, "synthetic", "Test")
    return g, rows


def test_runs_helper_bridges_gaps_and_enforces_min_length():
    m = [0, 1, 1, 0, 1, 1, 0, 0, 0, 1]
    assert runs(m, 4, 1) == [(1, 5)]          # gap of 1 bridged -> length 5
    assert runs(m, 4, 0) == []                # no bridging -> all runs shorter than 4
    assert runs([1, 1, 1, 1], 4, 0) == [(0, 3)]


def test_follow_detected_when_one_trails_the_other():
    # B walks right at 0.8 body heights/s; A follows 2 body heights behind, same speed, 8 s
    g, _ = run({1: lambda t: (200 + 80 * t, 500), 2: lambda t: (400 + 80 * t, 500)}, 8)
    f = [s for s in g["segments"] if s["state"] == FOLLOW]
    assert f and f[0]["actor"] == 1 and f[0]["target"] == 2     # id1 follows id2
    assert g["flag"] and FOLLOW in g["proposals"][0]["reasons"]


def test_walking_side_by_side_is_not_following():
    g, _ = run({1: lambda t: (200 + 80 * t, 450), 2: lambda t: (200 + 80 * t, 600)}, 8)
    assert FOLLOW not in states(g) and not g["flag"]


def test_following_for_too_short_is_ignored():
    g, _ = run({1: lambda t: (200 + 80 * t, 500), 2: lambda t: (400 + 80 * t, 500)}, 2.5)
    assert FOLLOW not in states(g)


def test_walking_apart_gives_nothing():
    g, _ = run({1: lambda t: (500 - 80 * t, 500), 2: lambda t: (500 + 80 * t, 500)}, 6)
    assert not g["flag"] and not (states(g) - set())


def test_approach_alone_does_not_flag_a_clip():
    # B walks up to a stationary A: reported as APPROACH, but a pure approach never flags
    g, _ = run({1: lambda t: (800, 500), 2: lambda t: (min(100 + 200 * t, 700), 500)}, 4)
    assert APPROACH in states(g) and not g["flag"]


def test_hover_next_to_a_stationary_person():
    # A stands still; B shuffles around within ~1 body height of A for 6 s
    g, _ = run({1: lambda t: (500, 500), 2: lambda t: (580 + 20 * np.sin(2 * t), 500 + 30 * np.cos(3 * t))}, 6)
    h = [s for s in g["segments"] if s["state"] == HOVER]
    assert h and h[0]["actor"] == 2 and g["flag"]


def test_two_people_standing_together_is_not_hover():
    g, _ = run({1: lambda t: (500, 500), 2: lambda t: (580, 500)}, 8)
    assert HOVER not in states(g) and not g["flag"]


def test_corner_when_blocker_moves_between_person_and_frame_edge():
    # A (blocked) stands still near the left edge; B walks over and stops between A and the edge
    g, _ = run({1: lambda t: (150, 500), 2: lambda t: (max(500 - 220 * t, 60), 560)}, 6)
    c = [s for s in g["segments"] if s["state"] == CORNER]
    assert c and c[0]["actor"] == 2 and c[0]["target"] == 1      # id2 corners id1
    assert g["flag"]


def test_arriving_behind_a_person_is_not_a_corner():
    # B walks up and stops on the room side of A: A stands between B and the edge, A never moved
    g, _ = run({1: lambda t: (150, 500), 2: lambda t: (max(700 - 220 * t, 260), 500)}, 6)
    assert CORNER not in states(g)


def test_two_people_standing_together_near_a_wall_is_not_a_corner():
    g, _ = run({1: lambda t: (150, 500), 2: lambda t: (60, 500)}, 6)
    assert CORNER not in states(g) and not g["flag"]


def test_escalation_after_buildup_ends_the_proposal():
    # B hovers 1.3 bh from stationary A for 6 s, then bursts at A at 6 body heights/s
    def b(t):
        return (630 + 15 * np.sin(2 * t), 500) if t < 6 else (630 - 600 * (t - 6), 500)
    g, _ = run({1: lambda t: (500, 500), 2: b}, 6.6)
    assert ESCALATION in states(g)
    p = g["proposals"][0]
    assert p["ends_in_escalation"] and p["end_s"] <= 6.6 and g["escalation"]["time_s"] >= 6.0


def test_ordered_progression_approach_then_corner():
    # A stays near the left edge; B runs up, then settles between A and the edge
    def b(t):
        if t < 3:
            return (700 - 150 * t, 520)           # approach
        return (max(250 - 190 * (t - 3), 60), 520)   # slide to the edge side of A
    g, _ = run({1: lambda t: (150, 500), 2: b}, 8)
    assert g["flag"] and g["ordered_progression"]
    assert g["phase_sequence"][0] == APPROACH and CORNER in g["phase_sequence"]


def test_single_person_means_no_interaction():
    g, rows = run({1: lambda t: (500, 500)}, 5)
    assert g["no_interaction"] and g["n_pairs"] == 0 and not g["flag"] and rows == []


def test_track_dropout_is_bridged():
    # follow scenario, but the follower disappears for 0.1 s mid-way: still one FOLLOW segment
    t = tracks({1: lambda t: (200 + 80 * t, 500), 2: lambda t: (400 + 80 * t, 500)}, 8)
    keep = ~((t["track_id"] == 1) & (t["frame_idx"] >= 100) & (t["frame_idx"] < 103))
    t = {k: (v[keep] if isinstance(v, np.ndarray) and v.ndim and len(v) == len(keep) else v) for k, v in t.items()}
    g, _ = analyze_clip(t, CFG, "x", "Test")
    assert len([s for s in g["segments"] if s["state"] == FOLLOW]) == 1


def test_output_files_roundtrip(tmp_path):
    g, rows = run({1: lambda t: (200 + 80 * t, 500), 2: lambda t: (400 + 80 * t, 500)}, 8)
    write_outputs(g, rows, tmp_path, "Test", "synthetic")
    g2 = load_gate(tmp_path / "Test" / "synthetic_gate.json")
    st = load_states(tmp_path / "Test" / "synthetic_states.csv")
    assert g2["flag"] == g["flag"] and g2["proposals"] == g["proposals"]
    assert len(st) == len(rows) and any(r["state"] == FOLLOW for r in st)
    json.dumps(g2)  # serializable


# ---- calibrate ----
from src.assm.calibrate import SWEEP, evaluate, rates, sweep  # noqa: E402


def _two_scenarios():
    follow = tracks({1: lambda t: (200 + 80 * t, 500), 2: lambda t: (400 + 80 * t, 500)}, 8)
    apart = tracks({1: lambda t: (500 - 80 * t, 500), 2: lambda t: (500 + 80 * t, 500)}, 8)
    return follow, apart


def test_calibrate_rates_separate_following_from_walking_apart():
    follow, apart = _two_scenarios()
    r = rates(evaluate([("Stalking", follow), ("Stalking", follow), ("Normal", apart), ("Normal", apart)], CFG))
    assert r["Stalking"]["flagged"] == 1.0 and r["Stalking"]["FOLLOW"] == 1.0
    assert r["_normal_flag"] == 0.0 and r["_other_flag"] == 1.0


def test_sweep_runs_and_picks_a_value_under_target(tmp_path):
    follow, apart = _two_scenarios()
    paths = []
    for k, (cat, t) in enumerate([("Stalking", follow), ("Normal", apart)]):
        p = tmp_path / f"{k}.npz"
        np.savez_compressed(p, **t)
        paths.append((cat, p))
    rows, best = sweep(paths, CFG, workers=1, target=0.15)
    assert len(rows) == sum(len(v) for v in SWEEP.values())
    assert best["follow_min_s"] is not None and best["follow_min_s"]["normal_flag"] <= 0.15


def test_parse_overrides():
    from src.config import parse_overrides
    assert parse_overrides(["follow_min_s=3", "x=abc"]) == {"follow_min_s": 3.0, "x": "abc"}
    with pytest.raises(ValueError):
        parse_overrides(["nokey"])


def test_gate_records_thresholds_used():
    g, _ = run({1: lambda t: (200 + 80 * t, 500), 2: lambda t: (400 + 80 * t, 500)}, 8)
    assert g["params"]["follow_min_s"] == CFG["follow_min_s"] and "burst_abs" in g["params"]


def test_low_confidence_track_is_ignored_like_a_chair_boxed_as_a_person():
    # B hovers next to A for 6 s; with high detection confidence it is a HOVER, with low confidence the track is dropped
    def build(conf_b):
        rows = []
        for f in range(int(6 * FPS)):
            t = f / FPS
            for tid, (x, y), c in ((1, (500, 500), 0.9), (2, (580 + 20 * np.sin(2 * t), 500 + 30 * np.cos(3 * t)), conf_b)):
                rows.append((f, tid, np.array([x - 20, y - 50, x + 20, y + 50], np.float32), c, KP))
        return pack_tracks(rows, FPS, W, H, int(6 * FPS))
    real, _ = analyze_clip(build(0.8), CFG, "r", "T")
    chair, _ = analyze_clip(build(0.2), CFG, "c", "T")
    assert real["flag"] and HOVER in real["states_seen"]
    assert chair["n_tracks_dropped_low_conf"] == 1 and chair["no_interaction"] and not chair["flag"]


def test_prepare_then_finalize_equals_analyze_and_one_prepare_serves_many_thresholds():
    from src.assm.gate import finalize_clip, prepare_clip
    t = tracks({1: lambda t: (200 + 80 * t, 500), 2: lambda t: (400 + 80 * t, 500)}, 8)
    prep = prepare_clip(t, CFG)
    assert finalize_clip(prep, CFG, "x", "T")[0] == analyze_clip(t, CFG, "x", "T")[0]
    strict = finalize_clip(prep, {**CFG, "follow_min_s": 20.0}, "x", "T")[0]     # same prep, stricter rule
    assert not strict["flag"] and finalize_clip(prep, CFG, "x", "T")[0]["flag"]


def test_sweep_matches_naive_evaluation(tmp_path):
    follow = tracks({1: lambda t: (200 + 80 * t, 500), 2: lambda t: (400 + 80 * t, 500)}, 8)
    apart = tracks({1: lambda t: (500 - 80 * t, 500), 2: lambda t: (500 + 80 * t, 500)}, 8)
    paths = []
    for k, (cat, t) in enumerate([("Stalking", follow), ("Normal", apart), ("Normal", follow)]):
        p = tmp_path / f"{k}.npz"
        np.savez_compressed(p, **t)
        paths.append((cat, p))
    rows, _ = sweep(paths, CFG, workers=1, target=0.15, progress=False)
    row = next(r for r in rows if r["param"] == "follow_min_s" and r["value"] == 8.0)
    naive = rates(evaluate([("Stalking", follow), ("Normal", apart), ("Normal", follow)], {**CFG, "follow_min_s": 8.0}))
    assert row["normal_flag"] == pytest.approx(naive["_normal_flag"]) and row["other_flag"] == pytest.approx(naive["_other_flag"])
