import datetime

import pandas as pd
import pytest

from src.curated.index import (check_times, clip_key, find_videos, match, normal_videos, parse_time, read_annotations)


def test_clip_key_is_tolerant_about_names():
    assert clip_key("Assassination_v4 (1") == ("assassination", 4)
    assert clip_key("Assassination_v04.mp4") == ("assassination", 4)
    assert clip_key("CHAIN SNATCHING v13.MP4") == ("chain_snatching", 13)
    assert clip_key("Chain_Snatching_v2") == ("chain_snatching", 2)
    assert clip_key("harrasment_v5") == ("harassment", 5)
    assert clip_key("notes.txt") is None


def test_parse_time_formats():
    assert parse_time(3) == 3.0 and parse_time("3") == 3.0 and parse_time("3.5 s") == 3.5
    assert parse_time("0:03") == 3.0 and parse_time("1:02") == 62.0 and parse_time("1:02:03") == 3723.0
    assert parse_time(datetime.time(0, 0, 15)) == 15.0
    assert parse_time(None) is None and parse_time("") is None and parse_time("soon") is None and parse_time(float("nan")) is None


def test_read_annotations_from_csv_and_every_sheet_of_an_excel(tmp_path):
    pd.DataFrame({"video name": ["Assassination_v4 (1", "Assassination_v10"], "start time": [3, 2]}).to_csv(tmp_path / "a.csv", index=False)
    r = read_annotations(tmp_path / "a.csv")
    assert [(x["name"], x["start_s"]) for x in r] == [("Assassination_v4 (1", 3.0), ("Assassination_v10", 2.0)]
    pytest.importorskip("openpyxl")
    with pd.ExcelWriter(tmp_path / "f.xlsx") as w:
        pd.DataFrame({"video name": ["Assassination_v4"], "start time": [3]}).to_excel(w, sheet_name="Assassination", index=False)
        pd.DataFrame({"video name": ["Stalking_v9", None], "start time": ["0:05", 7]}).to_excel(w, sheet_name="Stalking", index=False)
    x = read_annotations(tmp_path / "f.xlsx")
    assert [(a["name"], a["start_s"], a["sheet"]) for a in x] == [("Assassination_v4", 3.0, "Assassination"), ("Stalking_v9", 5.0, "Stalking")]


def make_tree(tmp_path):
    for folder, names in {"ASSASSINATION": ["Assassination_v4 (1).mp4", "Assassination_v10.mp4", "Assassination_v11.mp4"],
                          "CHAIN SNATCHING": ["Chain_Snatching_v2.mp4"], "NORMAL": [f"Normal_v{i}.mp4" for i in range(1, 7)]}.items():
        (tmp_path / "violence" / folder).mkdir(parents=True)
        for n in names:
            (tmp_path / "violence" / folder / n).write_bytes(b"x")
    return find_videos(tmp_path / "violence")


def test_matching_reports_everything_and_never_drops_silently(tmp_path):
    vids = make_tree(tmp_path)
    ann = [{"name": "Assassination_v4 (1", "start_s": 3.0, "sheet": "s", "raw_time": 3}, {"name": "Assassination_v10", "start_s": 2.0, "sheet": "s", "raw_time": 2},
           {"name": "Chain_Snatching_v2", "start_s": None, "sheet": "s", "raw_time": "soon"}, {"name": "Kidnapping_v9", "start_s": 4.0, "sheet": "s", "raw_time": 4},
           {"name": "Assassination_v10", "start_s": 9.0, "sheet": "s", "raw_time": 9}, {"name": "random text", "start_s": 1.0, "sheet": "s", "raw_time": 1}]
    matched, unmatched, unlabeled, notes = match(ann, vids)
    assert [m["clip_id"] for m in matched] == ["Assassination_v4", "Assassination_v10"]
    assert matched[0]["path"].endswith("Assassination_v4 (1).mp4") and matched[1]["start_s"] == 2.0           # the first start time of a duplicate row is kept
    reasons = {u["name"]: u["reason"] for u in unmatched}
    assert "not understood" in reasons["Chain_Snatching_v2"] and "no video" in reasons["Kidnapping_v9"] and "does not look" in reasons["random text"]
    assert [v["stem"] for v in unlabeled] == ["Assassination_v11", "Chain_Snatching_v2"] and any("more than once" in n for n in notes)


def test_normal_control_clips_are_spread_and_times_are_checked(tmp_path):
    vids = make_tree(tmp_path)
    n = normal_videos(vids, limit=3)
    assert [x["clip_id"] for x in n] == ["Normal_v1", "Normal_v3", "Normal_v5"] and all(x["start_s"] is None and x["category"] == "Normal" for x in n)
    m = [{"clip_id": "a", "start_s": 3.0}, {"clip_id": "b", "start_s": 1.0}, {"clip_id": "c", "start_s": 20.0}]
    out = {x["clip_id"]: x for x in check_times(m, {"a": 10.0, "b": 10.0, "c": 12.0})}
    assert out["a"]["status"] == "ok" and out["b"]["status"] == "short" and out["c"]["status"] == "beyond_end" and out["c"]["start_s"] == 12.0
