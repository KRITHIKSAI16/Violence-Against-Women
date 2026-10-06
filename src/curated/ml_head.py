"""Curated dataset, ML layer: window scores from frozen video embeddings, with the checks that keep the result honest.

Per window of a clip (2 s, step 0.5 s):
  zero-shot margin   concern minus benign probability from video-text similarity (needs a text-capable encoder; no training)
  learned score      logistic regression on the PCA-reduced embedding: windows BEFORE the violence start of violent clips (label 1) vs windows of Normal clips (label 0),
                     leave-one-clip-out so every score comes from a model that never saw that clip. PCA is fitted on the training clips of each fold only.
Clip-level numbers used in the report: violent = the score of the last window before the violence, Normal = its highest window (against us).
The shortcut-proof check is inside the violent clips: is the score higher in the last 3 s than earlier in the same clip? (see `trend_from_windows`)
Normal is a different source, so the learned score is always shown next to the style-only baseline of `src.curated.evaluate`.
"""
from pathlib import Path

import numpy as np

from src.curated.evaluate import auc, bootstrap_auc, sign_test_p


def emb_path(ml_dir, enc_name, category, clip_id):
    return Path(ml_dir) / enc_name / category / f"{clip_id}.npz"


def save_clip_encoding(ml_dir, enc_name, category, clip_id, t_starts, out):
    p = emb_path(ml_dir, enc_name, category, clip_id)
    p.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(p, t_start=np.asarray(t_starts, float), emb=np.asarray(out["emb"], np.float32),
                        logits=np.asarray(out["logits"], np.float32) if out.get("logits") is not None else np.zeros((len(t_starts), 0), np.float32))


def load_clip_encoding(ml_dir, enc_name, category, clip_id):
    p = emb_path(ml_dir, enc_name, category, clip_id)
    if not p.exists():
        return None
    z = np.load(p)
    return {"t_start": z["t_start"], "emb": z["emb"], "logits": z["logits"]}


def window_starts(duration_s, win_s=2.0, step_s=0.5):
    """Window start times covering a clip of this duration (a clip shorter than one window gets one window clamped to its length)."""
    if duration_s <= win_s:
        return [0.0]
    return [round(float(t), 3) for t in np.arange(0.0, duration_s - win_s + 1e-6, step_s)]


def build_table(clips, ml_dir, enc_name, win_s=2.0):
    """clips: manifest entries. -> dict of arrays: clip (ids), t_start, label (1 violent / 0 Normal), emb (n,D), margin (n,) (NaN if no text logits)."""
    from src.curated.ml_encoders import margin
    rows = {"clip": [], "t_start": [], "label": [], "emb": [], "margin": []}
    for c in clips:
        z = load_clip_encoding(ml_dir, enc_name, c["category"], c["clip_id"])
        if z is None:
            continue
        m = margin(z["logits"]) if z["logits"].shape[1] else np.full(len(z["t_start"]), np.nan)
        for t, e, mm in zip(z["t_start"], z["emb"], m):
            rows["clip"].append(c["clip_id"])
            rows["t_start"].append(float(t))
            rows["label"].append(0 if c.get("start_s") is None else 1)
            rows["emb"].append(e)
            rows["margin"].append(float(mm))
    return {k: np.asarray(v) for k, v in rows.items()}


def leave_one_clip_out(table, k=16, C=0.5):
    """Out-of-fold P(violent) for every window. PCA and the classifier are fitted without the held-out clip. Returns (n,) probabilities."""
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LogisticRegression
    X, y, clip = table["emb"], table["label"], table["clip"]
    out = np.full(len(y), np.nan)
    for c in np.unique(clip):
        te = clip == c
        tr = ~te
        if len(np.unique(y[tr])) < 2:
            continue
        kk = int(min(k, tr.sum() - 1, X.shape[1]))
        pca = PCA(n_components=kk, random_state=7).fit(X[tr])
        w = np.where(y[tr] == 1, 0.5 / max(1, (y[tr] == 1).sum()), 0.5 / max(1, (y[tr] == 0).sum()))        # classes balanced
        clf = LogisticRegression(C=C, max_iter=500).fit(pca.transform(X[tr]), y[tr], sample_weight=w * len(w))
        out[te] = clf.predict_proba(pca.transform(X[te]))[:, 1]
    return out


def clip_scores(table, window_scores, durations, last_s=3.0, win_s=2.0):
    """Per clip: violent -> mean score of windows inside the last `last_s` s; Normal -> highest window score. -> {clip: (label, score)}"""
    out = {}
    for c in np.unique(table["clip"]):
        m = table["clip"] == c
        s, t, lab = window_scores[m], table["t_start"][m], int(table["label"][m][0])
        ok = np.isfinite(s)
        if not ok.any():
            continue
        if lab == 0:
            out[c] = (0, float(np.nanmax(s)))
        else:
            late = ok & (t + win_s >= durations[c] - last_s + 1e-6)
            out[c] = (1, float(np.nanmean(s[late])) if late.any() else float(np.nanmean(s[ok])))
    return out


def trend_from_windows(table, window_scores, durations, last_s=3.0, win_s=2.0):
    """Within violent clips: mean score of the last-seconds windows minus the earlier windows (needs both). Counts and a sign test."""
    diffs = []
    for c in np.unique(table["clip"][table["label"] == 1]):
        m = table["clip"] == c
        s, t = window_scores[m], table["t_start"][m]
        ok = np.isfinite(s)
        late = ok & (t + win_s >= durations[c] - last_s + 1e-6)
        early = ok & (t + win_s <= durations[c] - last_s + 1e-6)
        if late.any() and early.any():
            diffs.append(float(np.mean(s[late]) - np.mean(s[early])))
    d = np.asarray(diffs)
    wins = int((d > 0).sum())
    return {"clips": len(d), "higher_in_last": wins, "median_diff": float(np.median(d)) if len(d) else float("nan"), "p_sign": sign_test_p(wins, len(d))}


def summarize_scores(cs):
    """{clip: (label, score)} -> AUC with bootstrap interval over clips."""
    pos = [s for lab, s in cs.values() if lab == 1]
    neg = [s for lab, s in cs.values() if lab == 0]
    a, lo, hi = bootstrap_auc(pos, neg)
    return {"violent_n": len(pos), "normal_n": len(neg), "auc": a, "auc_low": lo, "auc_high": hi}


def ablation_table(rows):
    """rows: {name: summarize_scores output (+ optional 'trend')} -> list of printable lines."""
    lines = [f"{'score':<34}{'violent':>8}{'normal':>8}{'AUC':>7}{'95% interval':>16}"]
    for name, r in rows.items():
        lines.append(f"{name:<34}{r['violent_n']:>8}{r['normal_n']:>8}{r['auc']:>7.2f}{'   %.2f-%.2f' % (r['auc_low'], r['auc_high']):>16}")
    return lines


__all__ = ["auc", "build_table", "leave_one_clip_out", "clip_scores", "trend_from_windows", "summarize_scores", "ablation_table", "window_starts",
           "save_clip_encoding", "load_clip_encoding"]
