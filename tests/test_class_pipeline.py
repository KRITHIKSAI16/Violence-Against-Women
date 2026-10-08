"""Classification video features, records, documents and the whole train stage on a synthetic colab2 folder (no GPU, no model download)."""
import json
from unittest import mock

import numpy as np
import pytest

from src.classcommon import config as ccfg
from src.classcommon import feat_text, feat_video, pipeline, report, stages
from src.curated.ml_head import save_clip_encoding
from tests.class_helpers import fake_records


class FakeEncoder:
    name = "xclip"

    def __init__(self, shift=0.0):
        self.shift, self.calls = shift, []

    def __call__(self, windows, prompts=None, batch=4):
        self.calls.append(len(windows))
        emb = np.array([[w.mean() / 255.0 + self.shift, w.std() / 255.0, 1.0, 0.0] for w in windows], np.float32)
        return {"emb": emb, "logits": np.tile(np.linspace(-1, 1, 11, dtype=np.float32), (len(windows), 1)) + self.shift}


def det_embed(texts, dim=32):
    """Deterministic bag-of-words embedding standing in for the sentence model."""
    out = np.zeros((len(texts), dim), np.float32)
    for i, t in enumerate(texts):
        for w in t.lower().replace(".", " ").replace(",", " ").split():
            out[i, sum(map(ord, w)) % dim] += 1.0
    n = np.linalg.norm(out, axis=1, keepdims=True)
    return out / np.maximum(n, 1e-9)


def test_window_starts_and_pooling():
    assert feat_video.span_starts(10.0, 3.0, 2.0, 0.5) == [7.0, 7.5, 8.0]
    assert feat_video.span_starts(3.1, 3.0, 2.0, 0.5) == [0.1, 0.6, 1.1]
    assert feat_video.span_starts(1.0, 3.0, 2.0, 0.5) == [0.0]
    assert feat_video.spread_starts(10.0, 2.0, 24) == [0.0, 2.0, 4.0, 6.0, 8.0]
    assert feat_video.spread_starts(100.0, 2.0, 5) == pytest.approx([0.0, 24.5, 49.0, 73.5, 98.0])
    assert feat_video.spread_starts(1.0, 2.0, 5) == [0.0]
    e = np.array([[1.0, 4.0], [3.0, 2.0]])
    assert feat_video.pool(e, "mean").tolist() == [2.0, 3.0] and feat_video.pool(e, "meanmax").tolist() == [2.0, 3.0, 3.0, 4.0]
    with pytest.raises(ValueError):
        feat_video.pool(e, "x")
    t, em, lg = feat_video.select_span([0, 1, 2, 3, 4, 5], np.arange(6)[:, None], np.zeros((6, 2)), 8.0, 3.0)
    assert t.tolist() == [5.0] and em.tolist() == [[5]]                          # only start 5 lies in the last 3 s (>= 5.0)
    t, em, _ = feat_video.select_span([0, 1], np.arange(2)[:, None], None, 20.0, 3.0)
    assert t.tolist() == [1.0]                                                    # nothing in the span: the last window is kept


def _cfg(tmp_path, task="trim", curated=None, **over):
    cfg, _ = ccfg.build_config(tmp_path / "out", task, curated_root=curated, use_captions=False, write=False)
    cl = cfg["classify"]
    cl["encoders"] = ["xclip"]
    cl["cv"].update({"outer_folds": 3, "repeats": 1, "inner_folds": 2, "min_per_class": 5})
    cl["bootstrap"] = 50
    cl.update(over)
    return cfg


def test_clip_windows_compute_cache_and_video_block(tmp_path, make_video):
    v = make_video(tmp_path / "c.mp4", fps=10.0, n_frames=100)                    # 10 s
    cfg = _cfg(tmp_path)
    rec = {"clip_id": "Stalking_v1", "category": "Stalking", "path": str(v), "duration_s": 10.0}
    enc = FakeEncoder()
    assert feat_video.missing([rec], cfg, "trim", "xclip") == [rec]
    t, e, lg = feat_video.clip_windows("xclip", enc, rec, cfg, "trim")
    assert t.tolist() == [7.0, 7.5, 8.0] and e.shape == (3, 4) and lg.shape == (3, 11) and enc.calls == [3]
    assert feat_video.missing([rec], cfg, "trim", "xclip") == []                  # cached now
    feat_video.clip_windows("xclip", None, rec, cfg, "trim")                      # no encoder needed any more
    X, z = feat_video.video_block([rec, {**rec, "clip_id": "Stalking_v2"}], cfg, "trim", "xclip")
    assert X.shape == (2, 4) and np.isfinite(X[0]).all() and np.isnan(X[1]).all() and np.isfinite(z[0]) and np.isnan(z[1])
    tf, ef, _ = feat_video.clip_windows("xclip", enc, {**rec, "clip_id": "Stalking_v3"}, cfg, "full")
    assert tf.tolist() == [0.0, 2.0, 4.0, 6.0, 8.0] and ef.shape == (5, 4)        # whole clip, 2 s windows
    Xf, _ = feat_video.video_block([{**rec, "clip_id": "Stalking_v3"}], cfg, "full", "xclip")
    assert Xf.shape == (1, 8)                                                     # mean and max pooled


def test_trim_reuses_the_windows_colab2_already_encoded(tmp_path):
    cur = tmp_path / "cur"
    ts = np.arange(0.0, 8.01, 0.5)
    save_clip_encoding(cur / "ml", "xclip", "Normal", "Normal_v1", ts, {"emb": np.arange(len(ts))[:, None] * np.ones((1, 4)), "logits": None})
    cfg = _cfg(tmp_path, curated=str(cur))
    rec = {"clip_id": "Normal_v1", "category": "Normal", "path": "does_not_exist.mp4", "duration_s": 10.0}
    t, e, lg = feat_video.clip_windows("xclip", None, rec, cfg, "trim")
    assert t.tolist() == [7.0, 7.5, 8.0] and e[:, 0].tolist() == [14, 15, 16] and lg.shape == (3, 0)
    assert feat_video.missing([rec], cfg, "trim", "xclip") == []
    assert feat_video.missing([{**rec, "clip_id": "Normal_v2"}], cfg, "trim", "xclip") != []


def test_text_embedding_cache_is_invalidated_by_a_changed_document(tmp_path):
    calls = []

    def fn(t):
        calls.append(len(t))
        return det_embed(t)
    p = tmp_path / "t.npz"
    a = feat_text.cached_embeddings(["a b", "c d"], "m", p, fn)
    b = feat_text.cached_embeddings(["a b", "c d"], "m", p, fn)
    assert calls == [2] and np.allclose(a, b)
    feat_text.cached_embeddings(["a b", "c e"], "m", p, fn)
    feat_text.cached_embeddings(["a b", "c e"], "other model", p, fn)
    assert calls == [2, 2, 2]


def _write_colab2(root, recs, with_windows=True, seed=0):
    """A fake finished colab2 folder: manifest, stories, and (optionally) X-CLIP windows whose first feature carries the label."""
    rng = np.random.default_rng(seed)
    clips = []
    for r in recs:
        clips.append({"clip_id": r["clip_id"], "category": r["category"], "path": str(root / "processed" / r["category"] / f"{r['clip_id']}.mp4"), "duration_s": 10.0,
                      "start_s": 10.0 if r["y"] else None, "raw_width": 640, "raw_height": 360, "status": "ok"})
        p = root / "stories" / r["category"] / f"{r['clip_id']}.json"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(r["result"]), encoding="utf-8")
        if with_windows:
            ts = np.arange(0.0, 8.01, 0.5)
            emb = rng.normal(0, 1, (len(ts), 6)).astype(np.float32)
            emb[:, 0] += 2.0 * r["y"]
            save_clip_encoding(root / "ml", "xclip", r["category"], r["clip_id"], ts, {"emb": emb, "logits": rng.normal(0, 1, (len(ts), 11)) + r["y"]})
    (root / "curated_manifest_clean.json").write_text(json.dumps({"clips": clips}), encoding="utf-8")


def test_load_records_and_documents(tmp_path):
    cur = tmp_path / "cur"
    recs = fake_records(3, 3)
    _write_colab2(cur, recs, with_windows=False)
    (cur / "stories" / "Normal" / "Normal_v6.json").unlink()                       # one clip without an analysis result
    cfg = _cfg(tmp_path, curated=str(cur))
    got = pipeline.load_records(cfg, "trim")
    assert [r["y"] for r in got] == [1, 1, 1, 0, 0, 0] and got[-1]["result"] is None and got[0]["result"]["clip_id"] == "Assassination_v1"
    caps = {"Assassination_v1": {"text": "A person walks."}}
    geom, full = pipeline.build_docs(got, cfg, "trim", caps)
    assert full["Assassination_v1"] == geom["Assassination_v1"] + " Scene description: A person walks." and full["Assassination_v2"] == geom["Assassination_v2"]
    assert geom["Normal_v6"].startswith("No interacting pair of people could be measured: no tracking result exists")
    with pytest.raises(ValueError):
        pipeline.load_records(ccfg.build_config(tmp_path / "x", "trim", write=False)[0], "trim")
    with pytest.raises(FileNotFoundError):
        pipeline.load_records(_cfg(tmp_path / "y", curated=str(tmp_path / "nothing")), "trim")


def test_train_stage_end_to_end_trim(tmp_path, monkeypatch):
    cur = tmp_path / "cur"
    recs = fake_records(14, 14)
    _write_colab2(cur, recs)
    cfg = _cfg(tmp_path, curated=str(cur))
    with mock.patch("src.curated.ml_encoders.load_encoders", side_effect=AssertionError("no encoder must be loaded: colab2 already encoded everything")):
        assert stages.stage_features(cfg, pipeline.load_records(cfg, "trim"), "trim") == []
    out = stages.stage_train(cfg, "trim", embed_fn=det_embed)
    assert out.name == "report.md"
    rep = out.read_text(encoding="utf-8")
    m = json.loads((out.parent / "metrics.json").read_text(encoding="utf-8"))
    assert m["evaluated"] is True and m["best"] not in ("style", "length")
    meths = m["methods"]
    assert {"video_xclip", "geom", "text_tfidf_geom", "text_emb_geom", "style", "length", "fusion_late", "fusion_stack", "fusion_early"} <= set(meths)
    assert meths["geom"]["metrics"]["auc"] > 0.95 and meths["video_xclip"]["metrics"]["auc"] > 0.9              # both carry the label by construction
    assert meths["style"]["metrics"]["auc"] == pytest.approx(0.5) and meths["length"]["metrics"]["auc"] == pytest.approx(0.5)    # constant shortcuts: chance
    assert "## Verdict" in rep and "beats the filming-style and clip-length baselines" in rep
    assert "label-name check: 0 documents contain a category name" in rep and "act-word check (trim): 0 documents" in rep
    assert all((out.parent / f).exists() for f in ("predictions.csv", "errors.csv", "roc.png", "metrics.json"))
    lines = (out.parent / "predictions.csv").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 29 and lines[0].startswith("clip_id,category,label,")
    assert (tmp_path / "out" / "text" / "Normal" / "Normal_v15.json").exists()
    page = (out.parent / "index.html").read_text(encoding="utf-8")
    for part in ("What the system does", "How well it works", "Where it fails", "Every clip", "Confusion matrix", "Is it learning behaviour", "data:image/png;base64", "Normal_v15"):
        assert part in page
    assert page.count("<img") >= 3 and "chart unavailable" not in page


def test_train_stage_reports_why_it_does_not_evaluate(tmp_path):
    cur = tmp_path / "cur"
    _write_colab2(cur, fake_records(4, 20))
    cfg = _cfg(tmp_path, curated=str(cur))
    out = stages.stage_train(cfg, "trim", embed_fn=det_embed)
    rep = out.read_text(encoding="utf-8")
    assert "not evaluated: 4 violent and 20 non-violent clips" in rep and "No accuracy, precision, recall or AUC is reported" in rep
    m = json.loads((out.parent / "metrics.json").read_text(encoding="utf-8"))
    assert m["evaluated"] is False and m["clips_per_category"] == {"Assassination": 4, "Normal": 20}
    assert not (out.parent / "predictions.csv").exists()
    page = (out.parent / "index.html").read_text(encoding="utf-8")
    assert "not evaluated: 4 violent and 20 non-violent clips" in page and "Assassination" in page


def test_captions_switch_adds_the_full_document_blocks(tmp_path):
    cur = tmp_path / "cur"
    recs = fake_records(14, 14)
    _write_colab2(cur, recs)
    cfg = _cfg(tmp_path, curated=str(cur), use_captions=True)
    cap_dir = tmp_path / "out" / "captions"
    for r in recs:
        p = cap_dir / r["category"] / f"{r['clip_id']}.json"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps({"text": "Two people stand apart." if not r["y"] else "One person moves toward the other.", "removed": 1}), encoding="utf-8")
    rep = stages.stage_train(cfg, "trim", embed_fn=det_embed).read_text(encoding="utf-8")
    m = json.loads((tmp_path / "out" / "report" / "metrics.json").read_text(encoding="utf-8"))
    assert {"text_tfidf_full", "text_emb_full"} <= set(m["methods"]) and "captions: 28 of 28 clips captioned, 28 sentences removed" in rep
    assert "Captions (Layer B): on" in rep


def test_verdict_wording():
    rows = {"geom": {"metrics": {"auc": 0.9}, "ci": {"auc": (0.8, 0.97)}}, "style": {"metrics": {"auc": 0.95}, "ci": {"auc": (0.9, 1.0)}}, "length": {"metrics": {"auc": 0.5}, "ci": {"auc": (0.4, 0.6)}}}
    assert "a filming-style or clip-length baseline does as well or better" in report.verdict(rows, "geom", (10, 10, 0.9))
    rows["style"]["metrics"]["auc"] = 0.6
    assert "beats the filming-style and clip-length baselines" in report.verdict(rows, "geom", (10, 10, 0.9))
    assert "does not clearly survive the same-resolution check" in report.verdict(rows, "geom", (3, 3, 0.9))
    rows["geom"]["ci"]["auc"] = (0.45, 0.9)
    assert "No model separates the classes clearly" in report.verdict(rows, "geom", (10, 10, 0.9))
    assert report.best_method(rows) == "geom" and report.best_method({"style": rows["style"]}) is None
