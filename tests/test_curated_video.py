import cv2
import numpy as np

from src.curated.video import append_end_card, plain_video
from tests.test_context_features import world


def make_video(path, n=45, fps=30, size=(160, 120)):
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
    for k in range(n):
        vw.write(np.full((size[1], size[0], 3), 90, np.uint8))
    vw.release()


def frames(path):
    cap = cv2.VideoCapture(str(path))
    out = []
    while True:
        ok, f = cap.read()
        if not ok:
            return out
        out.append(f)


def test_end_card_is_appended_after_all_footage(tmp_path):
    make_video(tmp_path / "v.mp4")
    out = append_end_card(tmp_path / "v.mp4", tmp_path / "o" / "v.mp4", "Violence starts at 1.5 s", "1.5 s of footage shown", seconds=1.0)
    fr = frames(out)
    assert abs(len(fr) - (45 + 30)) <= 2 and fr[10].mean() > 60 and fr[-3].mean() < 40          # footage first, dark card last
    assert fr[0].shape[0] % 2 == 0 and not (tmp_path / "o" / "v.raw.mp4").exists()


def test_plain_video_draws_boxes_and_a_caption(tmp_path):
    make_video(tmp_path / "v.mp4", n=30)
    t = world({1: lambda t: (80, 20, 70.0)}, n=30)
    out = plain_video(tmp_path / "v.mp4", t, "No interacting pair identified: only one person", tmp_path / "p.mp4")
    fr = frames(out)
    assert len(fr) == 30 and fr[0].shape[0] == 120 + 36 and (fr[5][:36] > 200).any()           # caption text in the bar
    assert (fr[5][36:, :, 1] > 180).any()                                                       # a green box
