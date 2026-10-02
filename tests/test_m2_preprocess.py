"""M2 tests: fps/size normalization, corrupt handling, resumability. Needs ffmpeg."""
import shutil

import pytest

from src.data.manifest import build_manifest
from src.data.preprocess import build_clean_manifest, build_filter

pytestmark = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")


def _clean(corpus, out_dir, **kw):
    manifest = build_manifest(corpus, "ExtrAnom", "Normal", [".mp4"])
    args = dict(fps=30, max_side=160, denoise=False)
    args.update(kw)
    return manifest, build_clean_manifest(manifest, out_dir, **args)


def test_filter_chain_contents():
    assert "hqdn3d" not in build_filter(30, 640, False)
    assert build_filter(30, 640, True).startswith("hqdn3d")
    assert "fps=30" in build_filter(30, 640, False)


def test_fps_and_longest_side_normalized(mini_corpus, tmp_path):
    _, clean = _clean(mini_corpus, tmp_path / "out")
    assert clean["num_clips"] == 2
    for c in clean["clips"]:
        assert round(c["fps"]) == 30
        assert max(c["width"], c["height"]) == 160
        assert c["width"] % 2 == 0 and c["height"] % 2 == 0  # h264 needs even sizes


def test_aspect_ratio_kept(mini_corpus, tmp_path):
    _, clean = _clean(mini_corpus, tmp_path / "out")
    by_id = {c["clip_id"]: c for c in clean["clips"]}
    landscape, portrait = by_id["Normal_v1"], by_id["Stalking_v1"]  # 320x180 and 180x320
    assert landscape["width"] > landscape["height"]
    assert landscape["width"] / landscape["height"] == pytest.approx(320 / 180, rel=0.02)
    assert portrait["height"] > portrait["width"]


def test_duration_preserved(mini_corpus, tmp_path):
    manifest, clean = _clean(mini_corpus, tmp_path / "out")
    raw = {c["clip_id"]: c["duration_s"] for c in manifest["clips"]}
    for c in clean["clips"]:
        assert c["duration_s"] == pytest.approx(raw[c["clip_id"]], abs=0.1)


def test_corrupt_clip_is_skipped_not_fatal(mini_corpus, tmp_path):
    manifest = build_manifest(mini_corpus, "ExtrAnom", "Normal", [".mp4"])
    # inject a clip that passed M1 on paper but is garbage on disk
    manifest["clips"].append({**manifest["clips"][0], "clip_id": "ghost",
                              "path": str(mini_corpus / "Stalking" / "bad.mp4")})
    clean = build_clean_manifest(manifest, tmp_path / "out", 30, 160, False)
    assert clean["num_clips"] == 2
    assert any("bad.mp4" in s["path"] for s in clean["skipped"])


def test_existing_output_is_reused(mini_corpus, tmp_path):
    out = tmp_path / "out"
    _clean(mini_corpus, out)
    f = out / "Normal" / "Normal_v1.mp4"
    mtime = f.stat().st_mtime_ns
    _clean(mini_corpus, out)
    assert f.stat().st_mtime_ns == mtime  # not re-encoded
