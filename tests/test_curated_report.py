import csv

from src.curated.evaluate import evaluate_all
from src.curated.report import build_html, write_csv
from tests.test_curated_evaluate import res


def full(cid, cat, normal, **kw):
    r = res(cid, cat, normal, **kw)
    r.update({"start_s": None if normal else 5.0, "lines": [f"{cid}: id2 approaches id1 (6.0 m -> 2.0 m)", "Lead-up type (heuristic): approach - test"], "note": "", "shots": 1, "people": {}})
    if r["interaction"]:
        r["interaction"].update({"first_dist_m": 6.0, "min_dist_m": 1.9, "label_seconds": {"approaches": 2.0}, "ids": [1, 2],
                                 "last": {"3": {"dist_start_m": 5.0, "dist_end_m": 2.0, "net_closing_m": 3.0, "seconds": 3.0}}})
    return r


def test_report_page_and_csv_contain_every_clip_and_the_honest_verdict(tmp_path):
    rs = [full(f"V{i}", "Stalking", False, last=2.0 + i * 0.01, windows=[(0.0, 0.1), (5.0, 2.0)], dur=7.0) for i in range(6)]
    rs += [full("Vx", "Kidnapping", False, pair=False)]
    rs += [full(f"N{i}", "Normal", True, maxw=0.0 + i * 0.01) for i in range(6)]
    ev = evaluate_all(rs)
    (tmp_path / "videos").mkdir()
    (tmp_path / "videos" / "V0.mp4").write_bytes(b"x")
    assets = {"V0": {"video": tmp_path / "videos" / "V0.mp4", "figure": tmp_path / "f" / "V0.png", "ml": {"encoder": "xclip", "last": 0.3, "max": 0.4}}}
    p = build_html(rs, ev, assets, {"perception": "yolo26x + BoT-SORT", "encoders": "xclip"}, tmp_path / "report" / "index.html")
    h = p.read_text(encoding="utf-8")
    assert "V0" in h and "Vx" in h and "N5" in h and ev["normal_vs_pre"]["verdict"].split(",")[0][:30] in h
    assert "../videos/V0.mp4" in h and "starts at 5.0 s" in h and "not the pretrained HIG model" in h and "no identity, gender or age" in h
    assert "no pair: one person" in h and "<video" in h and "Video model (zero-shot, xclip)" in h
    write_csv(rs, tmp_path / "summary.csv")
    rows = list(csv.DictReader(open(tmp_path / "summary.csv", encoding="utf-8")))
    assert len(rows) == 13 and rows[6]["pair_found"] == "0" and rows[0]["leadup_type"] == "approach" and rows[7]["violence_start_s"] == ""
