"""Tests for the story report page: category numbers, showcase honesty, findings text, HTML content."""
import numpy as np
import pytest

from src.report.story_report import build_html, category_rows, clip_rows, findings_html, select_showcase


def mk(cid, cat, concern=0.0, benign=0.0, preds=(), no_pair=False, esc=None, cue_sum=None, ordered=False):
    pair = {"pair": [1, 2], "episodes": [{"pred": p, "actor": 1, "target": 2, "start_f": 0, "end_f": 30, "start_s": 0.0, "end_s": 1.0, "conf": 0.9, "detail": {"dist_start": 3.0, "dist_end": 1.0, "lag_s": 1.5, "mean_dist": 1.0}} for p in preds],
            "concern_seconds": concern, "benign_seconds": benign, "cue_sum": concern if cue_sum is None else cue_sum, "phases": ["approach"] if preds else [], "ordered_progression": ordered, "cue_curve": []}
    return {"clip_id": cid, "category": cat, "fps": 30.0, "n_frames": 300, "no_pair": no_pair, "key_pair": None if no_pair else "1_2",
            "pairs": {} if no_pair else {"1_2": pair}, "escalation": {"time_s": esc, "kind": "contact", "pair": "1_2", "conf": 0.9} if esc else None,
            "concern_seconds": concern, "benign_seconds": benign, "preds_seen": sorted(preds),
            "scene": {"place_type": "indoor", "layout_reliable": True, "layout_conf": 0.7, "doors": 0, "low_light": False, "camera_moving": False, "max_people": 2,
                      "isolated_frac": 0.8, "n_pairs": 1, "duration_s": 10.0, "depth_agree_frac": None, "assumptions": {"fov_deg": 60.0, "person_height_m": 1.7}}}


@pytest.fixture
def stories():
    s = [mk(f"S{k}", "Stalking", concern=2.0 + k, preds=("follows",), cue_sum=2.0 + k, ordered=k % 2 == 0, esc=7.0 if k == 0 else None) for k in range(6)]
    s += [mk("S_quiet", "Stalking"), mk("S_nopair", "Stalking", no_pair=True)]
    s += [mk(f"N{k}", "Normal", benign=3.0, preds=("mutual_facing",)) for k in range(4)] + [mk("N_c", "Normal", concern=1.5, preds=("approaches_from_behind",))]
    return s


def test_category_rows_numbers(stories):
    scenes = {s["clip_id"]: {"camera": {"moving": s["clip_id"] == "S0"}} for s in stories}
    rows = {r["category"]: r for r in category_rows(stories, scenes)}
    st, no = rows["Stalking"], rows["Normal"]
    assert st["clips"] == 8 and st["no_pair"] == pytest.approx(1 / 8) and st["any_concern_cue"] == pytest.approx(6 / 8)
    assert st["follows"] == pytest.approx(6 / 8) and st["ordered"] == pytest.approx(3 / 8) and st["act_cue"] == pytest.approx(1 / 8)
    assert st["camera_moving"] == pytest.approx(1 / 8) and no["camera_moving"] == 0.0
    assert no["mutual_facing"] == pytest.approx(4 / 5) and no["median_benign_s"] == pytest.approx(3.0) and no["approaches_from_behind"] == pytest.approx(1 / 5)


def test_showcase_has_top_and_random_is_deterministic_and_zero_evidence_only_random(stories):
    a = select_showcase(stories, 2, 2, seed=1)
    assert a == select_showcase(stories, 2, 2, seed=1)
    st = a["Stalking"]
    assert [c for c, w in st if w == "top"] == ["S5", "S4"] and len([1 for _, w in st if w == "random"]) == 2
    assert len({c for c, _ in st}) == len(st)
    assert all(w == "random" for c, w in st if c in ("S_quiet", "S_nopair"))
    assert [c for c, w in a["Normal"] if w == "top"] == ["N_c"]            # only clips with concern evidence rank as top


LEARN = {"models": {"gb:style_only": {"clips_with_pair": {"auc": 0.988}}, "gb:behavior": {"clips_with_pair": {"auc": 0.693}, "static_camera_clips": {"auc": 0.492}}},
         "rule_gate": {"all_clips": {"auc": 0.441}}, "overlap": {"n_pos": 5, "n_neg": 3, "enough": False}}
BENCH = {"n_clips": 40, "annotators": ["a", "b"], "behaviors": {"following": {"annotated_present": 6, "detected": 5, "precision": 0.6, "recall": 0.5, "kappa": 0.7},
                                                              "fleeing": {"annotated_present": 1, "detected": 0, "precision": float("nan"), "recall": 0.0, "kappa": float("nan")}}}


def test_findings_use_the_real_numbers_and_say_when_things_are_missing():
    h = findings_html(LEARN, BENCH)
    for needle in ("0.99", "0.69", "0.49", "0.44", "different source", "too few clips", "following", "0.60"):
        assert needle in h
    assert "fleeing" not in h                                              # fewer than 3 annotated: not reported
    none = findings_html(None, None)
    assert "not found" in none and "not been run" in none


def test_html_contains_everything_and_escapes(stories, tmp_path):
    scenes = {}
    cats = category_rows(stories, scenes)
    showcase = select_showcase(stories, 1, 1, 0)
    top = next(c for c, w in showcase["Stalking"] if w == "top")
    rendered = {top: {"video": f"videos/Stalking/{top}_story.mp4", "storyboard": f"storyboards/Stalking/{top}_story.jpg"}}
    cuts = {top: {"cut_s": 6.0, "reason": "before detected escalation"}}
    build_html(stories, cats, showcase, rendered, cuts, LEARN, None, tmp_path, len(stories))
    h = (tmp_path / "index.html").read_text(encoding="utf-8")
    for needle in ("What this shows, and what it does not", "Key findings", "Categories", "Stalking", "Normal (baseline)", f"{top}_story.mp4", f"{top}_story.jpg",
                   "shows 6.0 s", "follows id2", "no pre-violence footage to show"):
        assert needle in h, needle
    assert "<script" not in h


def test_clip_rows_have_cut_and_scene_columns(stories):
    rows = clip_rows(stories, {}, {"S0": {"cut_s": 6.0, "reason": "before detected escalation"}}, {"S0": {"video": "v.mp4", "storyboard": "s.jpg"}})
    r = next(x for x in rows if x["clip_id"] == "S0")
    assert r["cut_s"] == 6.0 and r["video"] == "v.mp4" and r["act_cue_kind"] == "contact" and r["place_type"] == "indoor" and r["ordered"] == 1
    assert next(x for x in rows if x["clip_id"] == "S_nopair")["no_pair"] == 1
