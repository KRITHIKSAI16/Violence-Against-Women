"""Tests for the story video / storyboard: cut from the first act cue, drawn layers, file structure."""
import shutil

import cv2
import numpy as np
import pytest

from src.config import load_config
from src.context.features import analyze_context
from src.context.graph import build_story
from src.report.story_video import (active_episodes, draw_minimap, pair_ground, pick_story_frames, render_story_board, render_story_video, story_cut,
                                    strip_height)
from tests.test_context_features import world

CFG = load_config()
CTX, ST, REP = CFG["context"], CFG["story"], dict(CFG["report"])
FPS = 30.0
pytestmark = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")


def video_for(path, n):
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (640, 480))
    for k in range(n):
        f = np.full((480, 640, 3), 60, np.uint8)
        cv2.putText(f, str(k), (20, 40), 0, 1, (255, 255, 255), 2)
        vw.write(f)
    vw.release()
    return path


def follow_story(n=300, category="Stalking"):
    t = world({1: lambda t: (160 + 50.0 * (t - 2.0), 150), 2: lambda t: (160 + 50.0 * t, 150)}, n=n)
    sc, arr = analyze_context(t, CTX)
    return t, build_story(t, sc, arr, None, None, CTX, ST, "syn", category)


def story_with_act(act_s):
    """A follow story plus a hand-made contact episode at act_s seconds."""
    t, st = follow_story()
    ep = {"pred": "contact", "actor": 1, "target": 2, "start_f": int(act_s * FPS), "end_f": int(act_s * FPS) + 10, "start_s": act_s, "end_s": act_s + 0.4,
          "conf": 0.9, "detail": {"wrist_torso_m": 0.1, "depth": "same depth"}}
    st["pairs"][st["key_pair"]]["episodes"].append(ep)
    st["escalation"] = {"time_s": act_s, "kind": "contact", "pair": st["key_pair"], "conf": 0.9}
    return t, st


def test_cut_follows_the_first_act_cue_and_uses_the_earlier_of_story_and_gate():
    _, st = story_with_act(6.0)
    c = story_cut(st, "Kidnapping", None, REP)
    assert c["cut_s"] == pytest.approx(6.0 - REP["cut_margin_s"]) and c["trimmed"] and c["reason"] == "before detected escalation"
    assert story_cut(st, "Kidnapping", 4.0, REP)["cut_s"] == pytest.approx(4.0 - REP["cut_margin_s"])      # the gate saw it earlier
    _, plain = follow_story()
    assert story_cut(plain, "Stalking", None, REP)["cut_s"] == pytest.approx(10.0) and not story_cut(plain, "Stalking", None, REP)["trimmed"]
    assert story_cut(plain, "Chain_Snatching", None, REP)["reason"].startswith("uniform tail trim")
    assert not story_cut(story_with_act(1.2)[1], "Kidnapping", None, REP)["show"]                         # too little footage before the act


def test_active_episodes_put_act_cues_first():
    _, st = story_with_act(6.0)
    f = int(6.1 * FPS)
    act = active_episodes(st, f)
    assert act and act[0]["pred"] == "contact"
    assert all(e["start_f"] <= f <= e["end_f"] for e in act)


def test_video_has_footage_before_the_cut_plus_a_card_and_nothing_after(tmp_path):
    t, st = story_with_act(6.0)
    video = video_for(tmp_path / "v.mp4", 300)
    cut = story_cut(st, "Kidnapping", None, REP)
    out = render_story_video(video, t, st, None, cut, CTX, REP, tmp_path / "o" / "s.mp4")
    cap = cv2.VideoCapture(str(out))
    assert int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) == cut["cut_f"] + round(REP["withheld_card_s"] * FPS)
    assert cap.get(cv2.CAP_PROP_FRAME_WIDTH) == 640 and cap.get(cv2.CAP_PROP_FRAME_HEIGHT) == 58 + 480 + strip_height()
    cap.set(cv2.CAP_PROP_POS_FRAMES, cut["cut_f"] + 2)
    ok, card = cap.read()
    cap.release()
    assert ok and card[5:-5, 5:-5].mean() < 80                          # the dark card, not footage (footage background is grey 60 + bright text elsewhere)


def test_whole_clip_has_no_card_and_short_or_pairless_stories_are_skipped(tmp_path):
    t, st = follow_story()
    video = video_for(tmp_path / "v.mp4", 300)
    cut = story_cut(st, "Stalking", None, REP)
    out = render_story_video(video, t, st, None, cut, CTX, REP, tmp_path / "w.mp4")
    assert int(cv2.VideoCapture(str(out)).get(cv2.CAP_PROP_FRAME_COUNT)) == 300
    t2, st2 = story_with_act(1.0)
    assert render_story_video(video, t2, st2, None, story_cut(st2, "Kidnapping", None, REP), CTX, REP, tmp_path / "x.mp4") is None
    alone = world({1: lambda t: (300, 150)}, n=90)
    sc, arr = analyze_context(alone, CTX)
    nopair = build_story(alone, sc, arr, None, None, CTX, ST, "solo", "Normal")
    assert render_story_video(video, alone, nopair, None, {"show": True, "cut_f": 90, "trimmed": False}, CTX, REP, tmp_path / "y.mp4") is None


def test_storyboard_has_one_frame_per_distinct_predicate_before_the_cut(tmp_path):
    t, st = story_with_act(6.0)
    cut = story_cut(st, "Kidnapping", None, REP)
    picks = pick_story_frames(st, cut)
    assert picks and all(f < cut["cut_f"] for f, _ in picks) and not any("contact" in c for _, c in picks)
    assert any("follows" in c for _, c in picks)
    video = video_for(tmp_path / "v.mp4", 300)
    out = render_story_board(video, t, st, cut, tmp_path / "sb.jpg")
    img = cv2.imread(str(out))
    assert img is not None and img.shape[1] >= 440
    assert render_story_board(video, t, st, {**cut, "show": False}, tmp_path / "no.jpg") is None


def test_minimap_draws_available_people_and_ignores_cropped_positions():
    t, st = follow_story()
    pos = pair_ground(t, CTX, (1, 2))
    canvas = np.zeros((300, 300, 3), np.uint8)
    draw_minimap(canvas, 10, 10, 150, pos, (1, 2), 200, FPS, [], 6.0)
    assert canvas[10:160, 10:160].sum() > 0
    # cropped person: box touches the bottom frame edge -> position must be NaN
    crop = world({1: lambda t: (300, 380, 100.0), 2: lambda t: (360, 150)}, n=60)
    p2 = pair_ground(crop, CTX, (1, 2))
    assert np.isnan(p2[1]).all() and np.isfinite(p2[2][30]).all()
    empty = np.zeros((300, 300, 3), np.uint8)
    draw_minimap(empty, 10, 10, 150, {1: np.full((60, 2), np.nan), 2: np.full((60, 2), np.nan)}, (1, 2), 10, FPS, [], 6.0)
    assert empty[10:160, 10:160].sum() > 0                               # the 'pair not in view' note is drawn
