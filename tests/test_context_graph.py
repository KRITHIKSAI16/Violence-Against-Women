"""Tests for Stage C: episodes (scene-graph predicates), cue curve, escalation, narrative text, story assembly."""
import json

import numpy as np
import pytest

from src.config import load_config
from src.context.features import analyze_context
from src.context.graph import (ACT_PREDS, _confirm, _reaction, build_story, find_escalation, geometry_for, load_story, pair_episodes, save_story,
                               summarize_pair)
from src.context.narrative import episode_text, narrate, narrative_markdown, scene_line, summary_line
from tests.test_context_features import world

CFG = load_config()
CTX, ST = CFG["context"], CFG["story"]
FPS, W, H = 30.0, 640, 480


def episodes(people, n=300, facing=None, layout=None, depth=None, ctx=CTX):
    t = world(people, n=n, facing=facing)
    sc, arr = analyze_context(t, ctx)
    ids = sorted(people)
    i, j = ids[0], ids[1]
    geo = geometry_for(t, layout, depth, sc, i, j)
    return pair_episodes(i, j, arr[(i, j)], FPS, ctx, ST, geo, False), sc, arr, t


def preds(eps):
    return {e["pred"] for e in eps}


# ---------------------------------------------------------------- predicates
def test_follow_episode_with_roles_and_lag():
    eps, *_ = episodes({1: lambda t: (160 + 50.0 * (t - 2.0), 150), 2: lambda t: (160 + 50.0 * t, 150)})
    f = [e for e in eps if e["pred"] == "follows"]
    assert f and f[0]["actor"] == 1 and f[0]["target"] == 2
    assert f[0]["detail"]["lag_s"] == pytest.approx(2.0, abs=0.6) and f[0]["end_s"] - f[0]["start_s"] >= CTX["follow_min_s"]


def test_approach_from_behind_vs_frontal():
    approach = lambda t: (330, 120, 140.0 - 12.0 * min(t, 3.0))
    still = lambda t: (300, 150, 100.0)
    behind, *_ = episodes({1: approach, 2: still}, n=150, facing={1: "away", 2: "away"})
    assert "approaches_from_behind" in preds(behind)
    front, *_ = episodes({1: approach, 2: still}, n=150, facing={1: "away", 2: "camera"})
    assert "approaches_from_behind" not in preds(front) and ("approaches_frontal" in preds(front) or "approaches_side" in preds(front))
    e = next(e for e in front if e["pred"].startswith("approaches"))
    assert e["detail"]["dist_start"] > e["detail"]["dist_end"]


def test_mutual_facing_is_a_benign_episode_with_negative_weight():
    # two still people 2.2 m apart in depth facing each other: i faces away from the camera (toward j), j faces the camera (toward i)
    eps, *_ = episodes({1: lambda t: (300, 150, 130.0), 2: lambda t: (300, 150, 100.0)}, n=150, facing={1: "away", 2: "camera"})
    assert "mutual_facing" in preds(eps)
    s = summarize_pair(eps, 150, FPS, ST["cue_weights"])
    assert s["benign_seconds"] > 3 and s["concern_seconds"] == 0 and s["cue_sum"] < 0


def test_hover_near_a_person_standing_still():
    # person 2 stands still; person 1 stays within ~1 m moving around (not facing each other)
    mover = lambda t: (300 + 50 + 15 * np.sin(3 * t), 150)       # stays 0.6-1.1 m away, always moving
    eps, *_ = episodes({1: mover, 2: lambda t: (300, 150)}, n=240, facing={1: "right", 2: "right"})
    h = [e for e in eps if e["pred"] == "hovers_near"]
    assert h and h[0]["actor"] == 1 and h[0]["target"] == 2


def test_blocks_exit_uses_the_detected_door_and_the_blockers_position():
    layout = {"doors": [{"cx": 500 / W, "base_y": 250 / H, "x1": 0.7, "x2": 0.85, "y1": 0.3, "y2": 0.5, "area": 0.02}], "reliable": True, "place_type": "indoor"}
    i_left = lambda t: (200, 150)
    j_between = lambda t: (300, 150)
    eps, *_ = episodes({1: i_left, 2: j_between}, n=150, layout=layout)
    b = [e for e in eps if e["pred"] == "blocks_exit"]
    assert b and b[0]["actor"] == 2 and b[0]["target"] == 1                # id2 stands between id1 and the door
    eps2, *_ = episodes({1: lambda t: (300, 150), 2: lambda t: (200, 150)}, n=150, layout=layout)
    b2 = [e for e in eps2 if e["pred"] == "blocks_exit"]
    assert b2 and b2[0]["actor"] == 1 and b2[0]["target"] == 2             # roles swap when the positions swap
    eps3, *_ = episodes({1: i_left, 2: j_between}, n=150, layout={**layout, "reliable": False})
    assert "blocks_exit" not in preds(eps3)                                  # an unreliable layout is ignored
    eps4, *_ = episodes({1: i_left, 2: j_between}, n=150, layout=None)
    assert "blocks_exit" not in preds(eps4)


def test_pinned_against_comes_from_consecutive_depth_samples():
    rows = [{"frame": 30 * k, "depth_gap": 0.05, "pinned_i": k in (2, 3, 4), "pinned_j": False} for k in range(8)]
    depth = {"pair": [1, 2], "frames": rows}
    good_layout = {"doors": [], "reliable": True, "place_type": "indoor"}
    eps, *_ = episodes({1: lambda t: (200, 150), 2: lambda t: (270, 150)}, n=300, depth=depth, layout=good_layout)
    p = [e for e in eps if e["pred"] == "pinned_against"]
    assert len(p) == 1 and p[0]["target"] == 1 and p[0]["actor"] == 2 and p[0]["detail"]["samples"] == 3
    one = {"pair": [1, 2], "frames": [{**r, "pinned_i": k == 2} for k, r in enumerate(rows)]}
    eps2, *_ = episodes({1: lambda t: (200, 150), 2: lambda t: (270, 150)}, n=300, depth=one, layout=good_layout)
    assert "pinned_against" not in preds(eps2)                              # a single sample is not enough
    flipped = {"pair": [2, 1], "frames": [{**r, "pinned_j": r["pinned_i"], "pinned_i": False} for r in rows]}      # same facts, the depth file uses the other pair order
    eps3, *_ = episodes({1: lambda t: (200, 150), 2: lambda t: (270, 150)}, n=300, depth=flipped, layout=good_layout)
    assert any(e["pred"] == "pinned_against" and e["target"] == 1 for e in eps3)
    eps4, *_ = episodes({1: lambda t: (200, 150), 2: lambda t: (270, 150)}, n=300, depth=depth, layout={**good_layout, "reliable": False})
    assert "pinned_against" not in preds(eps4)                              # unreliable segmentation: no obstacle claims
    eps5, *_ = episodes({1: lambda t: (100, 150), 2: lambda t: (500, 150)}, n=300, depth=depth, layout=good_layout)
    assert "pinned_against" not in preds(eps5)                              # the other person is 6.8 m away: nobody is pinning anyone


def test_act_cues_are_confirmed_or_discounted_by_depth():
    ep = {"pred": "contact", "start_f": 100, "end_f": 110, "conf": 0.8, "detail": {}}
    same = _confirm(dict(ep, detail={}), {"depth_rows": [{"frame": 105, "depth_gap": 0.05}], "fps": 30})
    diff = _confirm(dict(ep, detail={}), {"depth_rows": [{"frame": 105, "depth_gap": 0.4}], "fps": 30})
    far = _confirm(dict(ep, detail={}), {"depth_rows": [{"frame": 900, "depth_gap": 0.4}], "fps": 30})
    none = _confirm(dict(ep, detail={}), None)
    assert same["detail"]["depth"] == "same depth" and same["conf"] == 0.8
    assert diff["detail"]["depth"] == "different depth" and diff["conf"] == pytest.approx(0.4)
    assert far["detail"]["depth"] == "not checked" and none["detail"]["depth"] == "not checked"


def test_target_reaction_stop_and_speed_up():
    a = {"ground_conf": np.ones(400)}
    stops = np.r_[np.full(150, 1.0), np.full(250, 0.0)]
    ups = np.r_[np.full(150, 0.8), np.full(250, 2.5)]
    walks = np.full(400, 1.0)
    f = lambda sp: _reaction(1, 2, 150, 200, FPS, a, False, ST, sp)
    assert [e["pred"] for e in f(stops)] == ["target_stops"]
    assert [e["pred"] for e in f(ups)] == ["target_speeds_up"]
    assert f(walks) == []


# ---------------------------------------------------------------- aggregation
def test_escalation_is_the_earliest_act_cue_of_any_narrated_pair():
    mk = lambda p, s: {"pred": p, "start_s": s, "conf": 0.9}
    esc = find_escalation({"1_2": [mk("follows", 1.0), mk("contact", 9.0)], "1_3": [mk("reaches_for", 6.5)]})
    assert esc == {"time_s": 6.5, "kind": "reaches_for", "pair": "1_3", "conf": 0.9}
    assert find_escalation({"1_2": [mk("follows", 1.0)]}) is None
    assert set(ACT_PREDS) == {"reaches_for", "contact", "flees_from"}


def test_summary_phases_order_and_cue_curve():
    n = 300
    e = lambda p, s, t: {"pred": p, "start_f": s, "end_f": t, "conf": 1.0}
    s = summarize_pair([e("approaches_from_behind", 30, 80), e("follows", 90, 200), e("contact", 230, 250), e("mutual_facing", 0, 20)], n, FPS, ST["cue_weights"])
    assert s["phases"] == ["approach", "follow / linger", "physical act"] and s["ordered_progression"]
    assert len(s["cue_curve"]) == n // 15 and max(c[1] for c in s["cue_curve"]) >= 2.0
    assert s["benign_seconds"] == pytest.approx(21 / FPS, abs=0.05)
    back = summarize_pair([e("contact", 30, 40), e("approaches_from_behind", 100, 150)], n, FPS, ST["cue_weights"])
    assert not back["ordered_progression"]                    # the act before the approach is not an escalation


def test_low_confidence_episodes_count_less():
    n = 100
    full = summarize_pair([{"pred": "follows", "start_f": 0, "end_f": 99, "conf": 1.0}], n, FPS, ST["cue_weights"])["cue_sum"]
    weak = summarize_pair([{"pred": "follows", "start_f": 0, "end_f": 99, "conf": 0.2}], n, FPS, ST["cue_weights"])["cue_sum"]
    assert weak < full * 0.7


# ---------------------------------------------------------------- the story of a clip
def _story(people, n=300, facing=None, layout=None, depth=None):
    t = world(people, n=n, facing=facing)
    sc, arr = analyze_context(t, CTX)
    return build_story(t, sc, arr, layout, depth, CTX, ST, "syn", "Stalking"), t


def test_build_story_follow_scenario_end_to_end(tmp_path):
    story, _ = _story({1: lambda t: (160 + 50.0 * (t - 2.0), 150), 2: lambda t: (160 + 50.0 * t, 150)})
    assert story["key_pair"] == "1_2" and "follows" in story["preds_seen"] and story["concern_seconds"] >= 3
    json.dumps(story, default=float)
    save_story(tmp_path, "Stalking", "syn", story)
    assert load_story(tmp_path, "Stalking", "syn")["preds_seen"] == story["preds_seen"] and load_story(tmp_path, "Stalking", "nope") is None
    md = narrative_markdown(story)
    assert "id1 follows id2" in md and "Summary:" in md and "Scene:" in md


def test_build_story_without_pairs_and_with_standing_far_apart_people():
    story, _ = _story({1: lambda t: (300, 150)}, n=90)
    assert story["no_pair"] and story["key_pair"] is None and story["escalation"] is None
    assert "fewer than two people" in summary_line(story)
    quiet, _ = _story({1: lambda t: (100, 150), 2: lambda t: (560, 150)}, n=150)
    assert quiet["preds_seen"] == [] and "no approach" in summary_line(quiet)


def test_narrative_text_notes_unconfirmed_and_benign_cues():
    ep = {"pred": "contact", "actor": 1, "target": 2, "conf": 0.3, "detail": {"wrist_torso_m": 0.12, "depth": "different depth"}, "start_s": 1, "end_s": 2}
    t = episode_text(ep)
    assert "id1" in t and "id2" in t and "0.12" in t and "unconfirmed" in t and "low confidence" in t
    mf = {"pred": "mutual_facing", "actor": 1, "target": 2, "conf": 0.9, "detail": {"mean_dist": 1.4}, "start_s": 0, "end_s": 3}
    assert "benign" in episode_text(mf)


def test_scene_line_reports_uncertainty_and_assumptions():
    facts = {"place_type": "outdoor", "layout_reliable": False, "low_light": True, "camera_moving": True, "max_people": 3, "isolated_frac": 0.4, "doors": 1,
             "assumptions": {"fov_deg": 60.0, "person_height_m": 1.7}}
    s = scene_line(facts)
    for needle in ("outdoor (uncertain)", "low-light", "moving camera", "3 people", "40%", "1 door", "60 degree"):
        assert needle in s
