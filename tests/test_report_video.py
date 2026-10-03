"""Tests for the buildup video / storyboard: cut rules, captions, and the rendered files."""
import shutil

import cv2
import numpy as np
import pytest

from src.assm.track_poses import pack_tracks
from src.config import load_config
from src.report.buildup_video import (active_captions, cut_point, load_overrides, pick_storyboard_frames,
                                      render_buildup_video, render_storyboard)

REP = dict(load_config()["report"])
FPS = 25.0


def gate(category="Kidnapping", esc=None, dur=10.0, segs=None, key=(1, 2), clip_id="c1"):
    return {"clip_id": clip_id, "category": category, "fps": FPS, "n_frames": int(dur * FPS), "duration_s": dur,
            "key_pair": list(key) if key else None, "no_interaction": key is None, "flag": bool(segs),
            "escalation": {"frame": int(esc * FPS), "time_s": esc, "pair": [1, 2]} if esc is not None else None,
            "phase_sequence": [s["state"] for s in (segs or [])], "ordered_progression": False,
            "segments": segs or [], "proposals": []}


def seg(state, a, b, actor=1, target=2):
    return {"state": state, "id_i": 1, "id_j": 2, "start_f": int(a * FPS), "end_f": int(b * FPS),
            "start_s": a, "end_s": b, "dur_s": b - a, "actor": actor, "target": target}


# ---------------- cut rules
def test_cut_whole_clip_for_stalking_harassment_and_normal():
    for cat in ("Stalking", "Harassment", "Normal"):
        c = cut_point(gate(cat), cat, REP)
        assert c["cut_s"] == 10.0 and not c["trimmed"] and c["show"]


def test_cut_before_detected_escalation_with_margin():
    c = cut_point(gate("Assassination", esc=6.0), "Assassination", REP)
    assert c["cut_s"] == pytest.approx(6.0 - REP["cut_margin_s"]) and c["trimmed"]
    assert c["reason"] == "before detected escalation"


def test_act_category_without_escalation_gets_uniform_tail_trim():
    c = cut_point(gate("Chain_Snatching"), "Chain_Snatching", REP)
    assert c["cut_s"] == pytest.approx(10.0 - REP["act_tail_trim_s"]) and c["trimmed"]


def test_normal_ignores_escalation():
    assert cut_point(gate("Normal", esc=3.0), "Normal", REP)["cut_s"] == 10.0


def test_stalking_with_escalation_is_also_cut():
    assert cut_point(gate("Stalking", esc=5.0), "Stalking", REP)["cut_s"] == pytest.approx(5.0 - REP["cut_margin_s"])


def test_manual_override_wins_and_too_short_is_not_shown():
    g = gate("Kidnapping", esc=6.0, clip_id="k9")
    assert cut_point(g, "Kidnapping", REP, {"k9": 2.5})["cut_s"] == 2.5
    assert cut_point(g, "Kidnapping", REP, {"k9": 2.5})["reason"] == "manual override"
    assert cut_point(gate("Kidnapping", esc=1.5), "Kidnapping", REP)["show"] is False


def test_load_overrides_skips_comments_and_header(tmp_path):
    p = tmp_path / "o.csv"
    p.write_text("clip_id,cut_s\n# comment\nA_v1,3.2\nB_v2,4\n")
    assert load_overrides(str(p)) == {"A_v1": 3.2, "B_v2": 4.0}
    assert load_overrides(str(tmp_path / "missing.csv")) == {}


# ---------------- captions
def test_captions_name_actor_target_and_time_and_rank_order():
    g = gate(segs=[seg("APPROACH", 1.0, 5.0), seg("FOLLOW", 2.0, 6.0), seg("CORNER", 3.0, 4.0, actor=2, target=1)])
    caps = active_captions(g, int(3.5 * FPS))
    assert [s for _, s in caps] == ["CORNER", "FOLLOW", "APPROACH"]          # highest rank first
    assert caps[0][0].startswith("id2 BLOCKING id1") and "id1 FOLLOWING id2" in caps[1][0]
    assert active_captions(g, int(0.5 * FPS)) == []
    assert active_captions(gate(key=None), 10) == []


def test_storyboard_frames_are_before_the_cut_and_one_per_state():
    g = gate("Stalking", segs=[seg("APPROACH", 1.0, 3.0), seg("FOLLOW", 3.0, 8.0), seg("FOLLOW", 8.5, 9.5)])
    cut = cut_point(g, "Stalking", REP)
    picks = pick_storyboard_frames(g, cut)
    assert len(picks) == 2 and all(f < cut["cut_f"] for f, _ in picks)
    assert "APPROACHING" in picks[0][1] and "FOLLOWING" in picks[1][1]
    none = pick_storyboard_frames(gate(key=None), cut_point(gate(key=None), "Stalking", REP))
    assert len(none) >= 1 and "no sustained" in none[0][1]


# ---------------- rendered files
@pytest.fixture
def synthetic(make_video, tmp_path):
    video = make_video(tmp_path / "v.mp4", w=320, h=180, fps=FPS, n_frames=75)       # 3.0 s
    rows = [(f, tid, np.array([40 + 100 * tid + f, 60, 70 + 100 * tid + f, 150], np.float32), 0.9,
             np.zeros((17, 3), np.float32)) for f in range(75) for tid in (1, 2)]
    t = pack_tracks(rows, FPS, 320, 180, 75)
    return video, t


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")
def test_video_stops_at_cut_then_shows_card_and_storyboard_written(synthetic, tmp_path):
    video, t = synthetic
    g = gate("Kidnapping", esc=2.0, dur=3.0, segs=[seg("FOLLOW", 0.2, 1.0)])
    rep = {**REP, "cut_margin_s": 0.5, "min_video_s": 0.5}
    cut = cut_point(g, "Kidnapping", rep)
    assert cut["cut_f"] == int(1.5 * FPS)
    out = render_buildup_video(video, t, g, cut, rep, tmp_path / "o" / "x_buildup.mp4")
    cap = cv2.VideoCapture(str(out))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    assert n == cut["cut_f"] + round(rep["withheld_card_s"] * FPS)       # footage up to the cut + the card, nothing more
    assert cap.get(cv2.CAP_PROP_FRAME_WIDTH) == 320 and cap.get(cv2.CAP_PROP_FRAME_HEIGHT) > 180
    cap.set(cv2.CAP_PROP_POS_FRAMES, n - 2)
    ok, last = cap.read()
    cap.release()
    assert ok and last[5:-5, 5:-5, 2].mean() < 90 and last[5:-5, 5:-5, 0].mean() < 90   # card is uniformly dark
    sb = render_storyboard(video, t, g, cut, tmp_path / "o" / "x_story.jpg")
    img = cv2.imread(str(sb))
    assert img is not None and img.shape[1] >= 400


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")
def test_whole_clip_has_no_card_and_too_short_clip_is_skipped(synthetic, tmp_path):
    video, t = synthetic
    g = gate("Stalking", dur=3.0, segs=[seg("FOLLOW", 0.2, 2.0)])
    cut = cut_point(g, "Stalking", REP)
    out = render_buildup_video(video, t, g, cut, REP, tmp_path / "w.mp4")
    assert int(cv2.VideoCapture(str(out)).get(cv2.CAP_PROP_FRAME_COUNT)) == 75           # no card appended
    short = cut_point(gate("Kidnapping", esc=1.0, dur=3.0), "Kidnapping", REP)
    assert render_buildup_video(video, t, gate("Kidnapping", esc=1.0, dur=3.0), short, REP, tmp_path / "s.mp4") is None
    assert render_storyboard(video, t, gate("Kidnapping", esc=1.0, dur=3.0), short, tmp_path / "s.jpg") is None
