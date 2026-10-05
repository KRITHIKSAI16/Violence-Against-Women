import cv2
import numpy as np

from src.label.vlm_propose import combine, parse_answer, propose_clip, read_frames, sample_times, score_against_human


def test_sample_times_respects_offset_and_cap():
    t = sample_times(10.0, 2.0, 100)
    assert t[0] == 0.0 and abs(t[1] - 0.5) < 1e-9 and len(t) == 20
    assert len(sample_times(60.0, 2.0, 40)) == 40 and sample_times(60.0, 2.0, 40)[-1] > 58
    assert sample_times(10.0, 1.0, 100, offset_s=0.5)[0] == 0.5


def test_parse_answer_handles_messy_model_text():
    assert parse_answer('Sure! {"act_start_s": 4.5, "buildup_start_s": null}') == {"act_start_s": 4.5, "buildup_start_s": None}
    assert parse_answer("```json\n{\"act_start_s\": null, \"buildup_start_s\": null}\n```") == {"act_start_s": None, "buildup_start_s": None}
    assert parse_answer("act_start_s: 3.2, buildup_start_s: 1") == {"act_start_s": 3.2, "buildup_start_s": 1.0}
    assert parse_answer("I cannot tell") is None and parse_answer("") is None
    assert parse_answer('{"act_start_s": -3, "buildup_start_s": "soon"}') == {"act_start_s": None, "buildup_start_s": None}


def test_combine_needs_both_passes_to_agree():
    a = {"act_start_s": 8.0, "buildup_start_s": 4.0}
    lab, st = combine(a, {"act_start_s": 8.8, "buildup_start_s": 4.5}, 20.0)
    assert st == "agree"
    assert lab == {"act_start_s": 8.4, "buildup_start_s": 4.25}
    assert combine(a, {"act_start_s": 12.0, "buildup_start_s": 4.0}, 20.0)[1] == "disagree"
    assert combine(a, {"act_start_s": 8.0, "buildup_start_s": None}, 20.0)[1] == "disagree"      # one sees a buildup, the other does not
    both_none = {"act_start_s": None, "buildup_start_s": None}
    assert combine(both_none, dict(both_none), 20.0) == (both_none, "agree")
    assert combine(a, None, 20.0)[1] == "invalid" and combine({"act_start_s": 99.0, "buildup_start_s": None}, a, 20.0)[1] == "invalid"


def test_a_tiny_leadup_counts_as_none():
    lab, st = combine({"act_start_s": 8.0, "buildup_start_s": 7.6}, {"act_start_s": 8.2, "buildup_start_s": 7.8}, 20.0)
    assert st == "agree" and lab["buildup_start_s"] is None


def test_scoring_against_human():
    props = {"a": {"status": "agree", "act_start_s": 5.0, "buildup_start_s": None}, "b": {"status": "agree", "act_start_s": 9.0, "buildup_start_s": None},
             "c": {"status": "disagree", "act_start_s": None, "buildup_start_s": None}}
    human = {k: {"source": "human", "act_start_s": v, "buildup_start_s": None} for k, v in {"a": 5.4, "b": 6.0, "c": 3.0}.items()}
    s = score_against_human(props, human)
    assert s["human_clips"] == 3 and s["with_agreed_answer"] == 2 and s["within_1s"] == 0.5 and s["within_2s"] == 0.5


def make_video(path, seconds=6, fps=10):
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (160, 120))
    for k in range(seconds * fps):
        vw.write(np.full((120, 160, 3), 20 * (k // fps), np.uint8))
    vw.release()


def test_read_frames_and_two_pass_run_with_a_fake_model(tmp_path):
    make_video(tmp_path / "v.mp4")
    fr = read_frames(tmp_path / "v.mp4", [0.0, 2.0, 4.0], short_side=60)
    assert len(fr) == 3 and min(fr[0][1].shape[:2]) == 60
    seen = []

    def fake(frames, prompt):
        seen.append((len(frames), prompt[:20]))
        return '{"act_start_s": 4.0, "buildup_start_s": 2.0}'
    lcfg = {"fps_a": 2, "fps_b": 1, "max_frames": 40, "short_side": 60, "agree_s": 1.5, "min_build_s": 1.0}
    r = propose_clip(tmp_path / "v.mp4", 6.0, fake, lcfg)
    assert r["status"] == "agree" and r["act_start_s"] == 4.0 and r["buildup_start_s"] == 2.0
    assert len(seen) == 2 and seen[0][1] != seen[1][1] and seen[0][0] > seen[1][0]                    # two different passes
