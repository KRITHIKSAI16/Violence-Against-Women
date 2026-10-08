"""Jury showcase (src/classcommon/showcase.py): example choice, per-class cue shares, facts and model scores, on synthetic results with known answers."""
import csv
import json

import numpy as np

from src.classcommon import showcase as S
from tests.class_helpers import fake_result


def _case(cid, cat, spans=None, pair=True, dur=12.0, **kw):
    res = fake_result(cid, cat, dur, spans, pair=pair, **kw)
    return {"clip": {"clip_id": cid, "category": cat}, "result": res, "y": int(cat != "Normal")}


def _spans(label, a=2.0, b=6.0):
    return [{"label": label, "start_s": a, "end_s": b, "dist_start_m": 3.0, "dist_end_m": 1.0, "mean_closing_ms": 0.5, "actor": 1, "target": 2}]


def test_evidence_counts_concerning_seconds_and_concern_window():
    c = _case("A_v1", "Kidnapping", _spans("follows", 2, 6))
    assert abs(S.evidence(c["result"]) - (4.0 + 1.4)) < 1e-9                      # 4 s of "follows" + concern_max_window 1.4 from the fake result
    n = _case("Normal_v1", "Normal", _spans("walking_together", 2, 6))
    assert abs(S.evidence(n["result"]) - 1.4) < 1e-9                              # benign seconds do not count


def test_pick_examples_most_vs_least_evidence_and_filters():
    cases = [_case("K_v1", "Kidnapping", _spans("follows", 2, 8)), _case("K_v2", "Kidnapping", _spans("follows", 2, 4)),
             _case("K_v3", "Kidnapping", pair=False), _case("K_v4", "Kidnapping", _spans("contact_or_reach", 1, 11), dur=60.0),
             _case("Normal_v1", "Normal", _spans("walking_together", 1, 9)), _case("Normal_v2", "Normal", _spans("approaches", 1, 5))]
    (v, n), = S.pick_examples(cases, 1)
    assert v["clip"]["clip_id"] == "K_v1"                                         # K_v4 is too long, K_v3 has no pair, K_v2 has less evidence
    assert n["clip"]["clip_id"] == "Normal_v1"                                    # least evidence
    (v, n), = S.pick_examples(cases, 1, violent="K_v2", normal="Normal_v2")       # overrides win
    assert (v["clip"]["clip_id"], n["clip"]["clip_id"]) == ("K_v2", "Normal_v2")
    assert S.pick_examples([c for c in cases if c["y"] == 1], 1) == []            # no Normal video: nothing to compare


def test_cohort_cue_shares_and_counts():
    cases = [_case("K_v1", "Kidnapping", _spans("follows", 2, 6)), _case("K_v2", "Kidnapping", _spans("follows", 2, 2.3)), _case("K_v3", "Kidnapping", pair=False),
             _case("Normal_v1", "Normal", _spans("walking_together", 2, 6))]
    co = S.cohort(cases)
    assert co["violent"]["videos"] == 3 and co["violent"]["with_pair"] == 2
    assert co["violent"]["cue_share"]["follows"] == 0.5                           # 4 s counts, 0.3 s is below CUE_MIN_S
    assert co["Normal"]["cue_share"]["follows"] == 0.0 and co["Normal"]["cue_share"]["walking_together"] == 1.0
    assert co["Normal"]["min_dist"] == [1.0]


def test_facts_time_close_and_labels():
    c = _case("K_v1", "Kidnapping", _spans("contact_or_reach", 2, 5))
    d = np.array([0.5, 0.8, 2.0, 3.0, np.nan])
    f = dict(S.facts(c["result"], d, 30.0))
    assert f["time closer than 1.2 m"] == "50 %"                                  # 2 of the 4 finite frames
    assert f["reaches for / touches"] == "3.0 s" and f["follows / approaches"] == "0.0 s" and f["closest distance"] == "1.0 m"


def test_load_scores_reads_best_model_column(tmp_path):
    (tmp_path / "metrics.json").write_text(json.dumps({"best": "late"}), encoding="utf-8")
    with open(tmp_path / "predictions.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["clip_id", "category", "label", "late_p", "style_p", "flags"])
        w.writerow(["K_v1", "Kidnapping", 1, "0.9100", "0.5", ""])
    assert S.load_scores(tmp_path) == ({"K_v1": 0.91}, "late")
    assert S.load_scores(tmp_path / "missing") == ({}, None)


def test_overview_png_is_16_9(tmp_path):
    cases = [_case("K_v1", "Kidnapping", _spans("follows")), _case("Normal_v1", "Normal", _spans("walking_together"))]
    out = S.render_overview(S.cohort(cases), {"evaluated": True, "best": "late", "verdict": "x", "methods": {
        "late": {"metrics": {"auc": 0.8}, "ci": {"auc": [0.7, 0.9]}}, "style": {"metrics": {"auc": 0.95}, "ci": {"auc": [0.9, 0.99]}}}}, tmp_path / "o.png")
    import cv2
    h, w = cv2.imread(str(out)).shape[:2]
    assert (w, h) == (1920, 1080)
