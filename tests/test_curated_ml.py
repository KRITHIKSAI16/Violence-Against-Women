import numpy as np

from src.curated.ml_encoders import ALL_PROMPTS, PROMPTS, encode_clip, load_encoders, margin
from src.curated.ml_head import (build_table, clip_scores, leave_one_clip_out, save_clip_encoding, summarize_scores, trend_from_windows, window_starts)


def test_margin_is_positive_for_concern_prompts_and_negative_for_benign():
    k, n = len(PROMPTS["concern"]), len(ALL_PROMPTS)
    a = np.zeros((1, n))
    a[0, 0] = 5.0
    b = np.zeros((1, n))
    b[0, k] = 5.0
    assert margin(a)[0] > 0.3 and margin(b)[0] < -0.3 and abs(margin(np.zeros((1, n)))[0] - (k - (n - k)) / n) < 1e-9


def test_encoder_fallback_chain_names_failures_and_keeps_the_rest():
    class Good:
        name = "good"

        def __init__(self, device="auto"):
            pass

    class Bad:
        def __init__(self, device="auto"):
            raise RuntimeError("no network")
    ok, failed = load_encoders(("a", "b"), loaders={"a": Bad, "b": Good})
    assert list(ok) == ["b"] and "no network" in failed["a"] and "RuntimeError" in failed["a"]


def fake_clips(tmp_path, n_v=8, n_n=8, seed=0):
    rng = np.random.default_rng(seed)
    clips, dur = [], {}
    for i in range(n_v + n_n):
        v = i < n_v
        cid = f"{'V' if v else 'N'}{i}"
        d = 8.0
        ts = np.array(window_starts(d))
        late = ts + 2.0 >= d - 3.0
        emb = rng.normal(0, 1, (len(ts), 12))
        if v:
            emb[late, 0] += 3.0                                   # the last seconds before the violence have a signature
        logits = rng.normal(0, 0.3, (len(ts), len(ALL_PROMPTS)))
        if v:
            logits[late, 0] += 4.0
        save_clip_encoding(tmp_path, "fake", "Stalking" if v else "Normal", cid, ts, {"emb": emb, "logits": logits})
        clips.append({"clip_id": cid, "category": "Stalking" if v else "Normal", "start_s": d if v else None})
        dur[cid] = d
    return clips, dur


def test_window_starts_and_table_round_trip(tmp_path):
    assert window_starts(1.0) == [0.0] and window_starts(4.0) == [0.0, 0.5, 1.0, 1.5, 2.0] and window_starts(2.0) == [0.0]
    clips, dur = fake_clips(tmp_path)
    t = build_table(clips, tmp_path, "fake")
    assert t["emb"].shape[1] == 12 and set(t["label"]) == {0, 1} and np.isfinite(t["margin"]).all()


def test_leave_one_clip_out_finds_the_planted_signature_and_never_trains_on_the_held_out_clip(tmp_path):
    clips, dur = fake_clips(tmp_path)
    t = build_table(clips, tmp_path, "fake")
    p = leave_one_clip_out(t)
    assert np.isfinite(p).all()
    cs = clip_scores(t, p, dur)
    s = summarize_scores(cs)
    assert s["violent_n"] == 8 and s["normal_n"] == 8
    tr = trend_from_windows(t, p, dur)
    assert tr["clips"] == 8 and tr["higher_in_last"] >= 7 and tr["median_diff"] > 0
    zs = summarize_scores(clip_scores(t, t["margin"], dur))                                  # zero-shot margin: planted concern prompt in the last seconds
    assert zs["auc"] > 0.9
    # a held-out clip must not influence its own score: shuffle one clip's labels -> its score cannot move
    t2 = {k: v.copy() for k, v in t.items()}
    c0 = t["clip"] == "V0"
    t2["label"][c0] = 0
    p2 = leave_one_clip_out(t2)
    assert np.allclose(p[c0], p2[c0], atol=1e-9)


def test_encode_clip_hands_one_window_per_start_to_the_encoder(tmp_path):
    import cv2
    vw = cv2.VideoWriter(str(tmp_path / "v.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 20, (160, 120))
    for k in range(120):
        vw.write(np.full((120, 160, 3), 10 + k % 200, np.uint8))
    vw.release()
    seen = []

    class E:
        name = "xclip"

        def __call__(self, windows):
            seen.append((len(windows), windows[0].shape[0]))
            return {"emb": np.zeros((len(windows), 3)), "logits": None}
    out = encode_clip(E(), tmp_path / "v.mp4", [0.0, 1.0, 2.0], 2.0)
    assert seen == [(3, 8)] and out["emb"].shape == (3, 3)
