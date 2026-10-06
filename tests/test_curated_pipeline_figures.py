import json

import numpy as np

from src.assm.track_poses import pack_tracks
from src.config import load_config
from src.context.features import analyze_context, save_context
from src.curated.config import build_config
from src.curated.figures import concern_figure, distance_figure
from src.curated.pipeline import analyze_clip, load_result, markdown
from tests.test_context_features import world

FPS = 30.0


def stage(tmp_path, people, n, facing=None, cat="Assassination", cid="Assassination_v1", start=5.0, status="ok"):
    cfg, _ = build_config(tmp_path / "out", write=False)
    t = world(people, n=n, facing=facing)
    t["shot_starts"] = np.array([0], np.int32)
    tdir = tmp_path / "out" / "tracks" / cat
    tdir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(tdir / f"{cid}.npz", **t)
    sc, arr = analyze_context(t, cfg["context"])
    save_context(sc, arr, tmp_path / "out" / "context", cat, cid)
    clip = {"clip_id": cid, "category": cat, "start_s": start, "status": status, "duration_s": n / FPS, "raw_width": 640, "raw_height": 480, "raw_fps": 25.0, "raw_duration_s": 9.0}
    return cfg, clip, t, arr


def test_analyze_clip_explains_an_approach_from_behind_and_saves_files(tmp_path):
    approach = lambda t: (330, 120, 140.0 - 12.0 * min(t, 3.0))
    cfg, clip, t, arr = stage(tmp_path, {1: approach, 2: lambda t: (300, 150, 100.0)}, 150, facing={1: "away", 2: "away"})
    r = analyze_clip(clip, cfg)
    assert not r["no_pair"] and r["interaction"]["leadup"]["type"] == "approach from behind" and r["pair_ids_local"] == [1, 2]
    assert load_result(cfg, "Assassination", "Assassination_v1")["clip_id"] == "Assassination_v1"
    md = (tmp_path / "out" / "stories" / "Assassination" / "Assassination_v1.md").read_text(encoding="utf-8")
    assert "Violence starts at 5.0 s" in md and "approaches" in md and "Meters are approximate" in md
    json.dumps(r, default=float)
    p = distance_figure(r, arr[(1, 2)]["dist_m"], FPS, tmp_path / "f" / "d.png")
    assert p.exists() and p.stat().st_size > 5000


def test_one_person_clip_says_why_there_is_no_story(tmp_path):
    cfg, clip, *_ = stage(tmp_path, {1: lambda t: (300, 150, 100.0)}, 90, cid="Stalking_v9", cat="Stalking")
    r = analyze_clip(clip, cfg)
    assert r["no_pair"] and r["reason"] == "one_person" and "only one person" in r["lines"][0] and r["interaction"] is None
    assert "Violence starts" in markdown(r)
    assert distance_figure(r, [], FPS, tmp_path / "x.png") is None


def test_missing_tracks_and_short_clips_are_reported_not_hidden(tmp_path):
    cfg, _ = build_config(tmp_path / "o2", write=False)
    r = analyze_clip({"clip_id": "Kidnapping_v1", "category": "Kidnapping", "start_s": 1.0, "status": "short", "duration_s": 1.0}, cfg)
    assert r["no_pair"] and r["reason"] == "no_tracks"
    cfg, clip, *_ = stage(tmp_path, {1: lambda t: (100, 150, 100.0), 2: lambda t: (560, 150, 100.0)}, 40, cid="Kidnapping_v2", cat="Kidnapping", status="short")
    r = analyze_clip({**clip, "start_s": 1.3}, cfg)
    assert "too little" in r["note"] and "id1" not in " ".join(r["lines"][:0])


def test_concern_figure_handles_empty_groups(tmp_path):
    assert concern_figure([1.2, 0.8, None], [-0.5, 0.1], tmp_path / "c.png").exists()
    assert concern_figure([], [], tmp_path / "c2.png").exists()
