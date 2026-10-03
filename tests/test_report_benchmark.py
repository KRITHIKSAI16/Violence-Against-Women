"""Tests for the annotation benchmark: sheet creation, kappa, detector precision / recall against majority labels, cut accuracy."""
import csv

import numpy as np
import pytest

from src.report.benchmark import (BEHAVIORS, SHEET_FIELDS, cohen_kappa, detector_hits, evaluate, make_sheet, read_sheet, select_clips)


def story(preds_at, no_pair=False, esc=None):
    """preds_at: [(pred, start_s)] -> minimal story dict."""
    eps = [{"pred": p, "start_s": s, "end_s": s + 1.0} for p, s in preds_at]
    return {"no_pair": no_pair, "pairs": {"1_2": {"episodes": eps}}, "escalation": {"time_s": esc} if esc is not None else None}


def test_cohen_kappa_basic_cases():
    assert cohen_kappa([1, 0, 1, 0], [1, 0, 1, 0]) == pytest.approx(1.0)
    assert cohen_kappa([1, 1, 0, 0], [0, 0, 1, 1]) == pytest.approx(-1.0)
    assert abs(cohen_kappa([1, 0, 1, 0, 1, 0, 1, 0], [1, 1, 0, 0, 1, 1, 0, 0])) < 1e-9          # independent labels
    assert np.isnan(cohen_kappa([1, 1, 1], [1, 1, 1]))                                           # undefined when everybody always says yes


def test_detector_hits_respects_the_act_start_and_no_pair():
    st = story([("follows", 2.0), ("contact", 9.0), ("hovers_near", 12.0)])
    assert detector_hits(st, None) == {"following", "reaching_or_contact", "lingering_near"}
    assert detector_hits(st, 8.0) == {"following"}                      # only what happens before the act
    assert detector_hits(story([], no_pair=True), None) == set() and detector_hits(None, None) == set()


def test_select_clips_is_stratified_prefers_clips_with_pairs_and_is_deterministic():
    clean = {"clips": [{"clip_id": f"{c}_{k}", "category": c, "path": "x"} for c in ("A", "Normal") for k in range(30)]}
    stories = {f"A_{k}": story([], no_pair=(k >= 10)) for k in range(30)}
    stories.update({f"Normal_{k}": story([], no_pair=True) for k in range(30)})
    a = select_clips(stories, clean, 10, seed=3)
    assert a == select_clips(stories, clean, 10, seed=3) and len(a) == 20
    assert sum(c["category"] == "A" for c in a) == 10 and sum(c["category"] == "Normal" for c in a) == 10
    with_pair = [c for c in a if c["category"] == "A" and not stories[c["clip_id"]]["no_pair"]]
    assert len(with_pair) == 7                                           # 70% of 10 come from clips with a usable pair
    assert len({c["clip_id"] for c in a}) == 20


def test_make_sheet_writes_one_copy_per_annotator_with_all_columns(tmp_path):
    clips = [{"clip_id": "c1", "category": "A", "path": "p1.mp4"}, {"clip_id": "c2", "category": "B", "path": "p2.mp4"}]
    paths = make_sheet(clips, tmp_path / "s.csv", ("ann_a", "ann_b"))
    assert [p.name for p in paths] == ["s_ann_a.csv", "s_ann_b.csv"] and (tmp_path / "ANNOTATION_INSTRUCTIONS.txt").exists()
    rows = list(csv.DictReader(open(paths[1], encoding="utf-8")))
    assert list(rows[0]) == SHEET_FIELDS and rows[0]["annotator"] == "ann_b" and rows[1]["video"] == "p2.mp4" and rows[0]["following"] == ""


def sheet(rows):
    """rows: {clip: {behavior: 0/1/None, act: float|None}} -> sheet dict as read_sheet would give."""
    out = {}
    for cid, r in rows.items():
        d = {"clip_id": cid, "act_start_s": "" if r.get("act") is None else str(r["act"])}
        for b in BEHAVIORS:
            d[b] = "" if r.get(b) is None else str(r[b])
        out[cid] = d
    return out


def test_evaluate_precision_recall_kappa_and_cut_accuracy():
    clips = ["c1", "c2", "c3", "c4"]
    truth = {"c1": {"following": 1, "act": 8.0}, "c2": {"following": 1, "act": 5.0}, "c3": {"following": 0, "act": None}, "c4": {"following": 0, "act": 3.0}}
    sheets = {"a": sheet({c: {**truth[c]} for c in clips}),
              "b": sheet({**{c: {**truth[c]} for c in clips}, "c4": {"following": 1, "act": 3.2}})}              # a and b disagree on c4: a tie, left out
    stories = {"c1": story([("follows", 2.0)], esc=8.4),            # true positive; cut 0.4 s after the true start
               "c2": story([("approaches_from_behind", 1.0)], esc=4.0),   # follow missed; act detected early
               "c3": story([("follows", 4.0)]),                     # false positive (no act, nothing annotated)
               "c4": story([("follows", 4.0)], esc=3.0)}            # follow only AFTER the act start (3.0 s) so it does not count
    res = evaluate(sheets, stories)
    f = res["behaviors"]["following"]
    assert res["n_clips"] == 4 and f["tp"] == 1 and f["fp"] == 1 and f["fn"] == 1
    assert f["precision"] == pytest.approx(0.5) and f["recall"] == pytest.approx(0.5) and f["false_positive_clips"] == ["c3"] and f["missed_clips"] == ["c2"]
    assert f["ties_or_unannotated"] == 1 and f["tn"] == 0
    assert f["kappa"] == pytest.approx(cohen_kappa([1, 1, 0, 0], [1, 1, 0, 1]))
    assert np.isnan(res["behaviors"]["fleeing"]["kappa"]) or res["behaviors"]["fleeing"]["annotated_present"] == 0
    cut = res["cut"]
    assert cut["n"] == 3 and cut["act_annotated"] == 3 and cut["act_detected_of_annotated"] == 3
    # errors: c1 8.4-8.0 = +0.4, c2 4.0-5.0 = -1.0, c4 3.0-3.1 (median of 3.0 and 3.2) = -0.1
    assert cut["median_error_s"] == pytest.approx(-0.1) and cut["detected_before_or_at_act_frac"] == pytest.approx(2 / 3)


def test_evaluate_skips_unannotated_cells_and_handles_single_annotator(tmp_path):
    sheets = {"a": sheet({"c1": {"following": 1}, "c2": {"following": None}})}
    res = evaluate(sheets, {"c1": story([("follows", 1.0)]), "c2": story([])})
    f = res["behaviors"]["following"]
    assert f["tp"] == 1 and f["fn"] == 0 and f["fp"] == 0 and np.isnan(f["kappa"])           # c2 was not annotated for this behavior: ignored
    p = tmp_path / "x.csv"
    make_sheet([{"clip_id": "c1", "category": "A", "path": "p"}], p, ("z",))
    assert read_sheet(tmp_path / "x_z.csv")["c1"]["category"] == "A"
