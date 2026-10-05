import json
import re

import cv2
import numpy as np

from src.label.annotate_pack import PAGE, build_pack
from src.phase.labels import read_labels


def make_video(path, seconds=2, fps=10):
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (160, 121))      # odd height on purpose
    for k in range(seconds * fps):
        vw.write(np.full((121, 160, 3), 30, np.uint8))
    vw.release()


def test_pack_has_proxies_and_a_page_without_model_proposals(tmp_path, monkeypatch):
    import src.label.annotate_pack as ap
    monkeypatch.setattr(ap, "resolve_path", lambda p: p if str(p).startswith(str(tmp_path)) else tmp_path / p)
    make_video(tmp_path / "a.mp4")
    make_video(tmp_path / "b.mp4")
    clips = [{"clip_id": "A_v1", "path": str(tmp_path / "a.mp4")}, {"clip_id": "B_v2", "path": str(tmp_path / "b.mp4")}]
    page = build_pack(clips, tmp_path / "pack")
    assert (tmp_path / "pack" / "videos" / "A_v1.mp4").exists()
    h = page.read_text(encoding="utf-8")
    items = json.loads(re.search(r"const CLIPS = (\[.*?\]);", h, re.S).group(1))
    assert [x["id"] for x in items] == ["A_v1", "B_v2"] and items[0]["src"] == "videos/A_v1.mp4"
    assert "vlm" not in h.lower() and "proposal" not in h.lower()                           # unbiased: nothing from the model is shown
    cap = cv2.VideoCapture(str(tmp_path / "pack" / "videos" / "A_v1.mp4"))
    assert cap.isOpened() and int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) % 2 == 0


def test_csv_header_written_by_the_page_matches_the_label_reader(tmp_path):
    assert "clip_id,act_start_s,buildup_start_s,source,note" in PAGE
    (tmp_path / "labels.csv").write_text('clip_id,act_start_s,buildup_start_s,source,note\nA_v1,6.40,3.10,human,"saw ""it"""\nB_v2,,,human,\n', encoding="utf-8")
    r = read_labels(tmp_path / "labels.csv")
    assert r["A_v1"]["act_start_s"] == 6.4 and r["A_v1"]["buildup_start_s"] == 3.1 and r["A_v1"]["note"] == 'saw "it"'
    assert r["B_v2"]["act_start_s"] is None
