"""M1 tests: manifest labels, metadata, and corrupt-file handling."""
import pytest

from src.data.manifest import build_manifest, probe_video

EXTS = [".mp4"]


def test_probe_video_reads_real_metadata(make_video, tmp_path):
    p = make_video(tmp_path / "a.mp4", w=320, h=180, fps=25.0, n_frames=50)
    meta = probe_video(p)
    assert (meta["width"], meta["height"]) == (320, 180)
    assert meta["frame_count"] == 50
    assert meta["fps"] == pytest.approx(25.0)
    assert meta["duration_s"] == pytest.approx(2.0, abs=0.05)


def test_probe_video_rejects_corrupt_file(tmp_path):
    bad = tmp_path / "bad.mp4"
    bad.write_bytes(b"garbage")
    with pytest.raises(ValueError):
        probe_video(bad)


def test_labels_and_fields(mini_corpus):
    m = build_manifest(mini_corpus, "ExtrAnom", "Normal", EXTS)
    by_id = {c["clip_id"]: c for c in m["clips"]}
    assert by_id["Normal_v1"]["label"] == "normal"
    assert by_id["Stalking_v1"]["label"] == "pre_violence"
    for c in m["clips"]:
        assert c["source"] == "ExtrAnom"
        assert {"category", "path", "duration_s", "fps", "frame_count", "width", "height"} <= c.keys()


def test_corrupt_skipped_not_fatal_and_non_video_ignored(mini_corpus):
    m = build_manifest(mini_corpus, "ExtrAnom", "Normal", EXTS)
    assert m["num_clips"] == 2
    assert [s["path"].endswith("bad.mp4") for s in m["skipped"]] == [True]  # notes.txt not counted
    assert m["counts_per_category"] == {"Normal": 1, "Stalking": 1}


def test_output_is_deterministic(mini_corpus):
    a = build_manifest(mini_corpus, "ExtrAnom", "Normal", EXTS)
    b = build_manifest(mini_corpus, "ExtrAnom", "Normal", EXTS)
    assert a == b


def test_missing_data_root_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        build_manifest(tmp_path / "nope", "ExtrAnom", "Normal", EXTS)


def test_probe_falls_back_to_ffprobe_when_opencv_cannot_decode(make_video, tmp_path, monkeypatch):
    import shutil
    if shutil.which("ffprobe") is None:
        pytest.skip("ffprobe not installed")
    import src.data.manifest as mf
    p = make_video(tmp_path / "a.mp4", w=320, h=180, fps=25.0, n_frames=50)

    def boom(_):
        raise ValueError("first frame could not be decoded")  # what an AV1 file does on Colab
    monkeypatch.setattr(mf, "_probe_cv2", boom)
    meta = mf.probe_video(p)
    assert (meta["width"], meta["height"], meta["frame_count"]) == (320, 180, 50)
    assert meta["fps"] == pytest.approx(25.0)


def test_corrupt_file_still_rejected_when_both_probes_fail(tmp_path):
    bad = tmp_path / "bad.mp4"
    bad.write_bytes(b"garbage")
    with pytest.raises(ValueError):
        probe_video(bad)
