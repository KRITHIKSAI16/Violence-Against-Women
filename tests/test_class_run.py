"""Classification CLIs: whole-video preparation (real ffmpeg, tiny videos) and the wiring of the stages."""
import json
import shutil
import sys
from argparse import Namespace
from pathlib import Path

import pytest

from src.classcommon import config as ccfg
from src.classfull import run as full_run
from src.classtrim import run as trim_run

needs_ffmpeg = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg is not installed")


def _corpus(tmp_path, make_video):
    root = tmp_path / "violence"
    make_video(root / "Normal" / "Normal_v1.mp4", n_frames=50)                     # 2.0 s at 25 fps
    make_video(root / "Normal" / "Normal_v2.mp4", n_frames=50)
    make_video(root / "Stalking" / "Stalking_v1.mp4", w=180, h=320, fps=30.0, n_frames=60)      # 2.0 s at 30 fps
    make_video(root / "Stalking" / "unnamed clip.mp4", n_frames=5)
    return root


@needs_ffmpeg
def test_full_prepare_keeps_whole_videos_and_labels_them(tmp_path, make_video, capsys):
    root = _corpus(tmp_path, make_video)
    cfg, _ = ccfg.build_config(tmp_path / "out", "full", write=True)
    full_run.stage_prepare(cfg, Namespace(data_root=str(root), drive_root="/nonexistent", n_normal=None, max_clips=None))
    man = json.loads(Path(cfg["preprocess"]["clean_manifest_path"]).read_text(encoding="utf-8"))
    by = {c["clip_id"]: c for c in man["clips"]}
    assert sorted(by) == ["Normal_v1", "Normal_v2", "Stalking_v1"]
    assert by["Stalking_v1"]["label"] == "violent" and by["Normal_v1"]["label"] == "normal" and by["Normal_v1"]["source"] == "class_full"
    assert all(c["start_s"] is None for c in man["clips"])
    assert all(1.9 <= c["duration_s"] <= 2.1 for c in man["clips"])                # nothing trimmed
    assert by["Stalking_v1"]["fps"] == pytest.approx(30.0, abs=0.5) and max(by["Stalking_v1"]["width"], by["Stalking_v1"]["height"]) <= 640
    out = capsys.readouterr().out
    assert "clips: 1 violent + 2 Normal" in out and "skipped 'unnamed clip'" in out


@needs_ffmpeg
def test_full_prepare_length_cap_and_trial_limit(tmp_path, make_video):
    root = _corpus(tmp_path, make_video)
    cfg, _ = ccfg.build_config(tmp_path / "out", "full", write=True)
    cfg["classify"]["max_len_s"] = 1.0
    full_run.stage_prepare(cfg, Namespace(data_root=str(root), drive_root="/x", n_normal=None, max_clips=2))
    man = json.loads(Path(cfg["preprocess"]["clean_manifest_path"]).read_text(encoding="utf-8"))
    assert sorted(c["category"] for c in man["clips"]) == ["Normal", "Stalking"]   # limit 2: one violent, one Normal
    assert all(0.9 <= c["duration_s"] <= 1.1 for c in man["clips"])


def test_full_prepare_without_a_video_folder_says_so(tmp_path):
    cfg, _ = ccfg.build_config(tmp_path / "out", "full", write=False)
    with pytest.raises(FileNotFoundError, match="--data-root"):
        full_run.stage_prepare(cfg, Namespace(data_root=str(tmp_path / "missing"), drive_root=str(tmp_path / "also_missing"), n_normal=None, max_clips=None))


def test_trim_cli_stage_wiring_and_caption_switch(tmp_path, monkeypatch, capsys):
    calls = []
    monkeypatch.setattr(trim_run, "load_records", lambda cfg, task: ["rec"])
    monkeypatch.setattr(trim_run, "run_captions", lambda recs, cfg, task: calls.append(("captions", recs, task)))
    monkeypatch.setattr(trim_run, "stage_features", lambda cfg, recs, task: calls.append(("features", recs, task)))
    monkeypatch.setattr(trim_run, "stage_train", lambda cfg, task: calls.append(("train", task)) or "report.md")
    base = ["prog", "--out-root", str(tmp_path / "o"), "--curated-root", str(tmp_path / "c")]
    monkeypatch.setattr(sys, "argv", base + ["--stage", "all"])
    trim_run.main()
    assert calls == [("captions", ["rec"], "trim"), ("features", ["rec"], "trim"), ("train", "trim")]
    assert "captions (Layer B): on" in capsys.readouterr().out
    calls.clear()
    monkeypatch.setattr(sys, "argv", base + ["--stage", "all", "--no-captions"])
    trim_run.main()
    assert [c[0] for c in calls] == ["features", "train"]                           # Layer B switched off: no captioning
    out = capsys.readouterr().out
    assert "captions (Layer B): off" in out and "captions are switched off" in out
    assert ccfg.load_class_config(tmp_path / "o")["classify"]["use_captions"] is False


def test_full_cli_stage_wiring(tmp_path, monkeypatch, capsys):
    cmds = []
    monkeypatch.setattr(full_run, "sh", lambda *a: cmds.append(a))
    monkeypatch.setattr(full_run, "stage_analyze", lambda cfg: cmds.append(("analyze",)))
    monkeypatch.setattr(sys, "argv", ["prog", "--out-root", str(tmp_path / "o"), "--stage", "track"])
    full_run.main()
    cfgfile = str(tmp_path / "o" / "class_config.yaml")
    assert cmds == [("src.video.shots", "--config", cfgfile), ("src.assm.track_poses", "--config", cfgfile)]
    cmds.clear()
    monkeypatch.setattr(sys, "argv", ["prog", "--out-root", str(tmp_path / "o"), "--stage", "context"])
    full_run.main()
    assert [c[0] for c in cmds] == ["src.context.features", "src.context.scene", "src.context.story"] and all(c[2] == cfgfile for c in cmds)
    cmds.clear()
    monkeypatch.setattr(sys, "argv", ["prog", "--out-root", str(tmp_path / "o"), "--stage", "analyze"])
    full_run.main()
    assert cmds == [("analyze",)]
    assert Path(cfgfile).exists()
