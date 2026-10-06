"""Classification pipelines: one video embedding per clip from the frozen encoders of src.curated.ml_encoders (X-CLIP, V-JEPA 2, InternVideo2).

A clip is a handful of windows (2 s each, 8 or 16 frames). Windows come from the analysed part only:
  trim  the last `span_s` seconds of the clip (the same length for violent and Normal clips)
  full  at most `max_windows` evenly spaced windows over the whole video
Window embeddings are pooled to one vector (trim: mean; full: mean and max, so a short act inside a long video is not averaged away).
Window embeddings are cached per clip. The trim task first looks for the windows that colab2 already encoded (`<curated_root>/ml/<encoder>/...`) and reuses them.
The X-CLIP prompt margin (zero-shot) is kept next to the embedding: it needs no training and is reported as a reference.
"""
from pathlib import Path

import numpy as np

from src.config import resolve_path


def span_starts(duration_s, span_s, win_s, step_s):
    """Window starts inside the last span_s seconds: the last window ends at the clip end. A clip shorter than one window gets one window at 0."""
    hi = max(0.0, float(duration_s) - float(win_s))
    lo = min(max(0.0, float(duration_s) - float(span_s)), hi)
    return [round(float(t), 3) for t in np.arange(lo, hi + 1e-6, step_s)] or [round(lo, 3)]


def spread_starts(duration_s, win_s, max_windows):
    """At most max_windows window starts, evenly spread over the whole clip (the last window ends at the clip end)."""
    hi = max(0.0, float(duration_s) - float(win_s))
    n = int(max(1, min(max_windows, np.floor(hi / max(win_s, 1e-6)) + 1)))
    return [round(float(t), 3) for t in np.linspace(0.0, hi, n)]


def pool(emb, how):
    """(n_windows, D) -> (D,) for 'mean' or (2D,) for 'meanmax'."""
    emb = np.asarray(emb, np.float32)
    if how == "mean":
        return emb.mean(axis=0)
    if how == "meanmax":
        return np.concatenate([emb.mean(axis=0), emb.max(axis=0)])
    raise ValueError(f"unknown pooling {how!r}")


def select_span(t_start, emb, logits, duration_s, span_s):
    """From stored windows keep those whose start lies in the last span_s seconds (all of them are kept for a shorter clip); never an empty selection."""
    t = np.asarray(t_start, float)
    keep = t >= float(duration_s) - float(span_s) - 1e-6
    if not keep.any():
        keep = np.zeros(len(t), bool)
        keep[-1] = True
    return t[keep], np.asarray(emb)[keep], (np.asarray(logits)[keep] if logits is not None and len(logits) else logits)


def feat_path(feat_dir, enc, category, clip_id):
    return Path(feat_dir) / "video" / enc / category / f"{clip_id}.npz"


def _save(path, t, out):
    path.parent.mkdir(parents=True, exist_ok=True)
    lg = out.get("logits")
    np.savez_compressed(path, t_start=np.asarray(t, float), emb=np.asarray(out["emb"], np.float32),
                        logits=np.asarray(lg, np.float32) if lg is not None else np.zeros((len(t), 0), np.float32))


def _load(path):
    z = np.load(path)
    return z["t_start"], z["emb"], z["logits"]


def curated_windows(curated_root, enc, rec, span_s):
    """Windows colab2 already encoded for this clip, cut to the last span_s seconds; None when there are none."""
    from src.curated.ml_head import load_clip_encoding
    if not curated_root:
        return None
    z = load_clip_encoding(Path(curated_root) / "ml", enc, rec["category"], rec["clip_id"])
    if z is None or not len(z["t_start"]):
        return None
    return select_span(z["t_start"], z["emb"], z["logits"], rec["duration_s"], span_s)


def clip_windows(enc_name, encoder, rec, cfg, task):
    """-> (t_starts, emb (n,D), logits (n,P) or empty) for one clip and one encoder, from cache, from colab2 (trim) or computed now (encoder given)."""
    cl = cfg["classify"]
    p = feat_path(cl["feat_dir"], enc_name, rec["category"], rec["clip_id"])
    if p.exists():
        return _load(p)
    if task == "trim":
        got = curated_windows(cl.get("curated_root"), enc_name, rec, float(cl["span_s"]))
        if got is not None:
            t, e, lg = got
            _save(p, t, {"emb": e, "logits": lg})
            return _load(p)
    if encoder is None:
        return None
    from src.curated.ml_encoders import encode_clip
    win = float(cl["win_s"])
    ts = span_starts(rec["duration_s"], cl["span_s"], win, cl["step_s"]) if task == "trim" else spread_starts(rec["duration_s"], win, int(cl["max_windows"]))
    out = encode_clip(encoder, resolve_path(rec["path"]), ts, win)
    _save(p, ts, out)
    return _load(p)


def missing(recs, cfg, task, enc_name):
    """The clips that have no windows for this encoder yet (neither cached nor, for trim, taken from colab2): they still need the encoder."""
    out = []
    for r in recs:
        try:
            got = clip_windows(enc_name, None, r, cfg, task)
        except Exception:
            got = None
        if got is None:
            out.append(r)
    return out


def encode_all(recs, cfg, task, enc_names, encoders=None):
    """Make sure every clip has cached windows for every encoder. encoders: {name: encoder object} (loaded with load_encoders) or None to use only caches/colab2.
    Returns the list of failures [(encoder, clip_id, error)]."""
    failures = []
    for name in enc_names:
        enc = (encoders or {}).get(name)
        for k, r in enumerate(recs, 1):
            try:
                clip_windows(name, enc, r, cfg, task)
            except Exception as e:                                    # one bad clip must not stop a Colab run
                failures.append((name, r["clip_id"], f"{type(e).__name__}: {str(e)[:120]}"))
            if k % 10 == 0 or k == len(recs):
                print(f"  {name}: {k}/{len(recs)} clips", flush=True)
    return failures


def video_block(recs, cfg, task, enc_name):
    """(matrix (n, D) float32 with NaN rows for clips without features, zero-shot score (n,) or NaN) for one encoder, from the caches only."""
    how = "mean" if task == "trim" else "meanmax"
    rows, zs = [], []
    from src.curated.ml_encoders import margin
    for r in recs:
        p = feat_path(cfg["classify"]["feat_dir"], enc_name, r["category"], r["clip_id"])
        if not p.exists():
            rows.append(None)
            zs.append(float("nan"))
            continue
        t, e, lg = _load(p)
        rows.append(pool(e, how))
        zs.append(float(np.mean(margin(lg))) if lg.shape[1] else float("nan"))
    d = next((len(x) for x in rows if x is not None), 0)
    X = np.full((len(recs), d), np.nan, np.float32)
    for i, x in enumerate(rows):
        if x is not None:
            X[i] = x
    return X, np.array(zs, float)
