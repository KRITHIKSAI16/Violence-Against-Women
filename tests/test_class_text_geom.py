"""Classification text builder, geometry vector, caption cleaner, label collection, config (known answers on synthetic results)."""
import numpy as np
import pytest

from src.classcommon import config as ccfg
from src.classcommon import feat_geom, text_build, vlm_caption
from src.classcommon.labels import collect_videos, label_of, spread
from tests.class_helpers import fake_result


def test_trim_text_uses_only_the_last_window_and_relative_times():
    res = fake_result(dur=10.0)                      # follows 2-6 s, approaches 7-10 s; the last 3 s are 7-10 s
    t = text_build.describe(res, "trim", 3.0)
    assert "follows" not in t                        # the follows span (2-6 s) is outside the window
    assert "From 0.0 to 3.0 s: person A moves toward person B (3.0 m -> 1.0 m, closing 0.7 m/s)." in t
    assert "In the last 3.0 s the two go from 3.0 m to 1.0 m apart (closest 1.0 m); person A moves toward person B, who is facing away" in t
    assert "a hand reaches" not in t                 # reach 0 and no contact


def test_full_text_describes_the_whole_clip():
    t = text_build.describe(fake_result(dur=10.0), "full", 3.0)
    assert "person A follows person B" in t and "person A moves toward person B" in t
    assert "From 2.0 to 6.0 s" in t and "Over the whole clip: follows 4.0 s, approaches 3.0 s." in t and "closest the two come is 1.0 m" in t


def test_text_never_contains_label_words_and_is_same_wording_for_both_classes():
    for cat in ("Normal", "Stalking"):
        t = text_build.describe(fake_result(category=cat, reach=2.0), "trim").lower()
        assert "violen" not in t and cat.lower() not in t and "before the" not in t
    a = text_build.describe(fake_result(category="Normal"), "trim")
    b = text_build.describe(fake_result(category="Kidnapping"), "trim")
    assert a == b


def test_no_pair_text_and_reach_sentence():
    assert text_build.describe(fake_result(pair=False), "full") == ("No interacting pair of people could be measured: people were never tracked together long enough. "
                                                                  "Up to 2 people were visible.")
    assert "a hand reaches the other person's body" in text_build.describe(fake_result(reach=2.0), "trim")


def test_window_helpers():
    assert text_build.window_bounds(10.0, "trim", 3.0) == (7.0, 10.0)
    assert text_build.window_bounds(2.0, "trim", 3.0) == (0.0, 2.0)
    assert text_build.window_bounds(10.0, "full", 3.0) == (0.0, 10.0)
    sp = [{"label": "a", "start_s": 0.0, "end_s": 1.0}, {"label": "b", "start_s": 5.0, "end_s": 9.0}, {"label": "c", "start_s": 9.9, "end_s": 10.0}]
    assert text_build.window_label_seconds(sp, 7.0, 10.0) == {"b": 2.0}           # a is outside; c overlaps 0.1 s only (< min span)
    assert text_build.pick_last({"1.5": "x", "3": "y"}, 2.9) == "y" and text_build.pick_last({}, 3.0) is None


def test_geom_vector_values():
    names = feat_geom.geom_names()
    v = dict(zip(names, feat_geom.geom_vector(fake_result(dur=10.0, reach=2.0), "trim", 3.0)))
    assert v["no_pair"] == 0.0 and v["max_people"] == 2.0 and v["last_dist_end_m"] == 1.0 and v["last_reach_ms"] == 2.0
    assert v["rel_approaches_frac"] == pytest.approx(1.0) and v["rel_follows_frac"] == 0.0           # last 3 s are fully 'approaches'
    assert np.isnan(v["clip_min_dist_m"])                                                             # whole-clip minimum only for the full task
    assert v["arm_raised_s"] == 0.6 and v["leadup_3"] == 1.0 and sum(v[f"leadup_{i}"] for i in range(9)) == 1.0
    f = dict(zip(names, feat_geom.geom_vector(fake_result(dur=10.0), "full", 3.0)))
    assert f["clip_min_dist_m"] == 1.0 and f["rel_follows_frac"] == pytest.approx(0.4) and f["rel_approaches_frac"] == pytest.approx(0.3)
    # concern windows: starts 0..8, win 2 s; trim window 7-10 s keeps starts 6, 7, 8 (those that overlap it)
    assert v["concern_span_mean"] == pytest.approx(np.mean([0.5 + 0.1 * t for t in (6, 7, 8)]))


def test_geom_vector_without_a_pair_and_style_vector():
    v = dict(zip(feat_geom.geom_names(), feat_geom.geom_vector(fake_result(pair=False), "trim")))
    assert v["no_pair"] == 1.0 and np.isnan(v["last_dist_end_m"]) and v["leadup_0"] == 0.0
    assert len(feat_geom.geom_vector({}, "full")) == len(feat_geom.geom_names())
    s = feat_geom.style_vec(fake_result())
    assert s[0] == 640 and s[1] == 360 and s[2] == 30.0 and s[3] == 0.0 and s[4] == 100.0 and s[5] == 50.0
    assert feat_geom.length_vec(7.5).tolist() == [7.5]


def test_clean_caption_neutralises_identity_and_drops_label_sentences():
    raw = "A young man follows a woman. She looks back at him. The man starts to hit her. This looks like a kidnapping."
    full, n_full = vlm_caption.clean_caption(raw, "full")
    assert full == "A person follows a person. They looks back at them. The person starts to hit their." and n_full == 1       # category sentence removed, act kept
    trim, n_trim = vlm_caption.clean_caption(raw, "trim")
    assert trim == "A person follows a person. They looks back at them." and n_trim == 2                                       # act sentence removed too
    assert vlm_caption.clean_caption("", "trim") == ("", 0) and vlm_caption.clean_caption(None, "full") == ("", 0)
    assert vlm_caption.clean_caption("Two women and three men talk.", "full")[0] == "Two people and three people talk."
    assert vlm_caption.clean_caption("An elderly man waits.", "full")[0] == "An person waits."                  # best effort: the grammar of 'an' is not repaired


def test_frame_times_are_slice_midpoints():
    assert vlm_caption.frame_times(0.0, 8.0, 4) == [1.0, 3.0, 5.0, 7.0]
    assert vlm_caption.frame_times(7.0, 10.0, 1) == [8.5]


def test_caption_clip_and_cache(tmp_path, make_video):
    v = make_video(tmp_path / "c.mp4", fps=10.0, n_frames=80)                    # 8 s
    seen = {}

    def ask(frames, prompt):
        seen["n"], seen["t"], seen["p"] = len(frames), [t for t, _ in frames], prompt
        return "A man walks. He is attacked."
    out = vlm_caption.caption_clip(v, 8.0, ask, "trim", 3.0, 3)
    assert seen["n"] == 3 and seen["t"] == [5.5, 6.5, 7.5]                       # inside the last 3 s only
    assert "violence" not in seen["p"].lower() and out["text"] == "A person walks." and out["removed"] == 1
    cfg = {"classify": {"caption_dir": str(tmp_path / "cap"), "span_s": 3.0, "caption": {"model": "m", "load_4bit": False, "max_new_tokens": 5, "n_frames_trim": 2, "n_frames_full": 4, "short_side": 64}}}
    recs = [{"clip_id": "Stalking_v1", "category": "Stalking", "path": str(v), "duration_s": 8.0}]
    got = vlm_caption.run_captions(recs, cfg, "full", ask=ask)
    assert got["Stalking_v1"]["text"] == "A person walks. They is attacked." and seen["n"] == 4        # full: 4 frames, act sentence kept
    seen.clear()
    vlm_caption.run_captions(recs, cfg, "full", ask=lambda *a: pytest.fail("cached caption must not be asked again"))
    assert not seen


def test_labels_and_collect_videos(tmp_path, make_video):
    assert label_of("Normal") == 0 and label_of("Stalking") == 1
    assert spread(list(range(10)), 5) == [0, 2, 4, 6, 8] and spread([1, 2], None) == [1, 2] and spread([1, 2], 0) == []
    for name in ("Normal/Normal_v1.mp4", "Normal/Normal_v2.mp4", "Normal/Normal_v3.mp4", "Stalking/Stalking_v1.mp4", "Stalking/Stalking_v2 (1).mp4", "Stalking/random name.mp4",
                 "Other/Stalking_v1.mp4"):
        make_video(tmp_path / name, n_frames=3)
    clips, skipped, notes = collect_videos(tmp_path)
    ids = sorted(c["clip_id"] for c in clips)
    assert ids == ["Normal_v1", "Normal_v2", "Normal_v3", "Stalking_v1", "Stalking_v2"]
    assert skipped == ["random name"] and len(notes) == 1 and "Stalking_v1" in notes[0]
    assert all(c["start_s"] is None for c in clips)
    clips2, _, _ = collect_videos(tmp_path, n_normal=1)
    assert sorted(c["category"] for c in clips2) == ["Normal", "Stalking", "Stalking"]
    clips3, _, _ = collect_videos(tmp_path, limit=2)
    assert sorted(c["category"] for c in clips3) == ["Normal", "Stalking"]


def test_build_config_paths_and_switch(tmp_path):
    cfg, p = ccfg.build_config(tmp_path / "out", "full", curated_root=None, use_captions=False)
    assert p.exists() and cfg["classify"]["use_captions"] is False and cfg["classify"]["task"] == "full"
    for k in ("text_dir", "caption_dir", "feat_dir", "report_dir"):
        assert cfg["classify"][k].startswith(str(tmp_path / "out"))
    assert cfg["preprocess"]["clean_manifest_path"].startswith(str(tmp_path / "out")) and cfg["assm"]["tracks_dir"].startswith(str(tmp_path / "out"))
    assert ccfg.load_class_config(tmp_path / "out")["classify"]["span_s"] == 3.0
    cfg2, _ = ccfg.build_config(tmp_path / "o2", "trim", curated_root="/x", write=False)
    assert cfg2["classify"]["use_captions"] is True and not (tmp_path / "o2").exists()
    with pytest.raises(ValueError):
        ccfg.build_config(tmp_path, "nope")


def test_perception_choice_is_applied_only_for_the_full_task(tmp_path):
    cur = tmp_path / "cur"
    cur.mkdir()
    (cur / "perception_choice.json").write_text('{"chosen": "y26x_1280_botsort"}', encoding="utf-8")
    full, _ = ccfg.build_config(tmp_path / "a", "full", curated_root=cur, write=False)
    assert full["assm"]["imgsz"] == 1280 and full["assm"]["model"].endswith("yolo26x-pose.pt")
    assert ccfg.apply_perception_choice({"assm": {}}, tmp_path / "none") is None
    trim, _ = ccfg.build_config(tmp_path / "b", "trim", curated_root=cur, write=False)
    assert trim["assm"]["imgsz"] == 640
