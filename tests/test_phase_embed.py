import cv2
import numpy as np
import pandas as pd

from src.phase.embed import attach_pca, embed_clip, emb_path, load_frames, window_clips


def make_video(path, seconds=6, fps=20):
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (160, 120))
    for k in range(seconds * fps):
        vw.write(np.full((120, 160, 3), 10 + 40 * (k // fps), np.uint8))               # brightness steps once per second
    vw.release()


def test_frames_are_sampled_at_the_embed_rate_and_windows_slice_16_frames(tmp_path):
    make_video(tmp_path / "v.mp4")
    fr = load_frames(tmp_path / "v.mp4", embed_fps=8.0, short_side=60)
    assert abs(len(fr) - 48) <= 1 and min(fr.shape[1:3]) == 60 and fr.shape[1] % 2 == 0
    w = window_clips(fr, [0.0, 3.0, 5.5], 2.0)
    assert len(w) == 3 and all(x.shape[0] == 16 for x in w)
    assert w[1].mean() > w[0].mean()                                                   # a later window shows later (brighter) frames


def test_embed_clip_passes_one_clip_per_window_to_the_encoder(tmp_path):
    make_video(tmp_path / "v.mp4")
    seen = []

    def enc(clips):
        seen.append(len(clips))
        return np.array([[c.mean(), 1.0, 0.0] for c in clips])
    e = embed_clip(tmp_path / "v.mp4", [0.0, 1.0, 2.0, 3.0], 2.0, enc)
    assert e.shape == (4, 3) and seen == [4] and e[3, 0] > e[0, 0]


def test_pca_is_fitted_on_training_clips_only_and_missing_clips_stay_nan(tmp_path):
    rng = np.random.default_rng(0)
    rows, cats = [], {}
    for c in ["a", "b", "c"]:
        cats[c] = "Stalking"
        ts = np.arange(0, 5, 0.5)
        rows += [{"clip_id": c, "t_start": float(t)} for t in ts]
        if c != "c":                                                                   # clip c has no cached embedding
            p = emb_path(tmp_path, "Stalking", c)
            p.parent.mkdir(parents=True, exist_ok=True)
            base = 10.0 if c == "b" else 0.0                                           # clip b is far away from the training clip a
            np.savez_compressed(p, t_start=ts, emb=(base + rng.normal(0, 1, (len(ts), 20))).astype(np.float16))
    df = pd.DataFrame(rows)
    out = attach_pca(df, tmp_path, cats, {"a"}, k=4)
    cols = [c for c in out.columns if c.startswith("emb_")]
    assert len(cols) == 4 and out[out.clip_id == "c"][cols].isna().all().all()
    a, b = out[out.clip_id == "a"][cols].to_numpy(), out[out.clip_id == "b"][cols].to_numpy()
    assert abs(a.mean()) < 1e-6 and abs(b.mean(axis=0)).max() > 1.0                    # centred on the training clip only, so clip b is offset
