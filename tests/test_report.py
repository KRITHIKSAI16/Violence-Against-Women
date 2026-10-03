"""Tests for the category report: numbers, selection, review sheet, HTML."""
import csv

import pytest

from src.config import load_config
from src.report.buildup_report import (build_html, category_rows, claim_text, clip_rows, evidence_score,
                                       select_review, select_showcase, write_csv)

REP = dict(load_config()["report"])
FPS = 30.0


def mk(cid, cat, segs=(), esc=None, ordered=False, no_pair=False, dur=10.0):
    seglist = [{"state": s, "id_i": 1, "id_j": 2, "start_f": int(a * FPS), "end_f": int(b * FPS), "start_s": a, "end_s": b,
                "dur_s": b - a, "actor": 1, "target": 2} for s, a, b in segs]
    flagged = any(s in ("FOLLOW", "HOVER", "CORNER") for s, _, _ in segs)
    props = [{"start_f": 0, "end_f": 1, "start_s": segs[0][1], "end_s": segs[-1][2], "pair": [1, 2], "reasons": [],
              "ends_in_escalation": esc is not None}] if flagged else []
    return {"clip_id": cid, "category": cat, "fps": FPS, "n_frames": int(dur * FPS), "duration_s": dur,
            "no_interaction": no_pair, "key_pair": None if (no_pair or not seglist) else [1, 2],
            "flag": flagged, "states_seen": sorted({s for s, _, _ in segs}), "ordered_progression": ordered,
            "phase_sequence": [s for s, _, _ in segs], "escalation": {"frame": 0, "time_s": esc, "pair": [1, 2]} if esc else None,
            "proposals": props, "segments": seglist, "max_people": 2, "isolated_frac": 1.0}


@pytest.fixture
def gates():
    g = []
    for k in range(6):
        g.append(mk(f"S{k}", "Stalking", [("APPROACH", 1, 3), ("FOLLOW", 3, 3 + 2 * k + 1)], ordered=k % 2 == 0))
    g.append(mk("S_none", "Stalking"))
    g.append(mk("S_nopair", "Stalking", no_pair=True))
    for k in range(5):
        g.append(mk(f"N{k}", "Normal", [("HOVER", 2, 6)] if k == 0 else []))
    g.append(mk("K1", "Kidnapping", [("CORNER", 2, 4)], esc=4.5, ordered=True))
    return g


def test_evidence_orders_clips_by_buildup(gates):
    by = {g["clip_id"]: g for g in gates}
    assert evidence_score(by["S5"]) > evidence_score(by["S0"]) > evidence_score(by["S_none"]) == 0


def test_category_rows_numbers(gates):
    rows = {r["category"]: r for r in category_rows(gates)}
    s, n = rows["Stalking"], rows["Normal"]
    assert s["clips"] == 8 and s["flagged"] == pytest.approx(6 / 8) and s["no_pair"] == pytest.approx(1 / 8)
    assert s["FOLLOW"] == pytest.approx(6 / 8) and s["ordered"] == pytest.approx(3 / 8)
    assert n["flagged"] == pytest.approx(1 / 5) and n["HOVER"] == pytest.approx(1 / 5)
    assert rows["Kidnapping"]["median_time_to_escalation_s"] == pytest.approx(2.5)   # 4.5 - proposal start 2


def test_showcase_is_deterministic_has_top_and_random_and_no_duplicates(gates):
    a = select_showcase(gates, 2, 2, seed=1)
    assert a == select_showcase(gates, 2, 2, seed=1)
    st = a["Stalking"]
    assert [c for c, w in st if w == "top"] == ["S4", "S5"]      # S4: ordered-progression bonus puts it first
    assert len([1 for _, w in st if w == "random"]) == 2
    assert len({c for c, _ in st}) == len(st)
    assert all(c != "S_none" or w == "random" for c, w in st)        # zero-evidence clips only appear as random


def test_review_sheet_balances_categories_and_includes_unflagged(gates):
    picks = select_review(gates, n_flagged=4, n_unflagged=1, seed=3)
    flagged = [g["category"] for g, grp in picks if grp == "flagged"]
    assert len(flagged) == 4 and set(flagged) == {"Stalking", "Normal", "Kidnapping"} - set() or len(set(flagged)) >= 2
    unfl = [g for g, grp in picks if grp == "unflagged"]
    assert len(unfl) == 1 and unfl[0]["category"] != "Normal" and not unfl[0]["flag"] and not unfl[0]["no_interaction"]


def test_claim_text(gates):
    by = {g["clip_id"]: g for g in gates}
    assert "FOLLOWING id2" in claim_text(by["S3"]) and "burst at 4.5s" in claim_text(by["K1"])
    assert claim_text(by["S_none"]) == "no sustained buildup behavior detected"


def test_files_written_and_html_has_categories_and_limits(gates, tmp_path):
    cats = category_rows(gates)
    showcase = select_showcase(gates, 1, 1, 0)
    top = next(c for c, w in showcase["Stalking"] if w == "top")
    rendered = {top: {"video": f"videos/{top}.mp4", "storyboard": f"storyboards/{top}.jpg"}}
    build_html(gates, cats, showcase, rendered, REP, tmp_path, ["note"])
    h = (tmp_path / "index.html").read_text(encoding="utf-8")
    for needle in ("Stalking", "Normal", "false-alarm baseline", "Limits", f"videos/{top}.mp4", f"storyboards/{top}.jpg", top):
        assert needle in h
    write_csv(tmp_path / "clips.csv", clip_rows(gates, REP, {}, rendered))
    rows = list(csv.DictReader(open(tmp_path / "clips.csv", encoding="utf-8")))
    assert len(rows) == len(gates)
    s5 = next(r for r in rows if r["clip_id"] == top)
    assert s5["flag"] == "1" and s5["video"] == f"videos/{top}.mp4" and s5["cut_reason"] == "whole clip"
    k1 = next(r for r in rows if r["clip_id"] == "K1")
    assert k1["cut_reason"] == "before detected escalation"
