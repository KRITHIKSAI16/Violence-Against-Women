"""Shared test helpers: tiny synthetic videos so tests need no real dataset or GPU."""
import cv2
import numpy as np
import pytest


def write_video(path, w=320, h=180, fps=25.0, n_frames=30):
    """Write a small moving-square mp4 to `path`."""
    path.parent.mkdir(parents=True, exist_ok=True)
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    assert vw.isOpened(), "cv2.VideoWriter could not open (codec missing?)"
    for i in range(n_frames):
        frame = np.zeros((h, w, 3), np.uint8)
        x = (i * 5) % (w - 20)
        frame[40:60, x:x + 20] = 255
        vw.write(frame)
    vw.release()
    return path


@pytest.fixture
def make_video():
    return write_video


@pytest.fixture
def mini_corpus(tmp_path):
    """data_root with 1 clip in Normal and 1 in Stalking, plus one corrupt file."""
    root = tmp_path / "corpus"
    write_video(root / "Normal" / "Normal_v1.mp4")
    write_video(root / "Stalking" / "Stalking_v1.mp4", w=180, h=320, fps=30.0)
    (root / "Stalking" / "bad.mp4").write_bytes(b"not a video")
    (root / "Stalking" / "notes.txt").write_text("ignored")
    return root
