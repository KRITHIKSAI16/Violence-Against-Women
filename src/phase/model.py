"""Layer 2, step 5: the learned phase model. Per-window state probabilities (background / buildup / act) -> constrained decoding -> clip timeline.

Model: gradient boosting (handles missing values: a window where a feature could not be measured is normal here) on the window features of
`src.phase.dataset`. Clips count equally (a 60 s clip does not outvote a 6 s clip) and the three states are balanced, because most windows
are background. Training clips are chosen by the caller (pseudo-labeled clips, never the held-out ones); `grouped_cv` keeps all windows of a
clip in one fold so a clip is never in both training and test.
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import GroupKFold

from src.phase.decode import decode_clip
from src.phase.labels import window_targets


def add_targets(df, labels):
    """df with clip_id, t_center + labels {clip: {act_start_s, buildup_start_s}} -> df restricted to labeled clips with a 'target' column."""
    parts = []
    for cid, g in df.groupby("clip_id", sort=False):
        if cid not in labels:
            continue
        g = g.copy()
        g["target"] = window_targets(g["t_center"].to_numpy(), labels[cid]["act_start_s"], labels[cid]["buildup_start_s"])
        parts.append(g)
    return pd.concat(parts, ignore_index=True) if parts else df.iloc[0:0].assign(target=[])


def sample_weights(df):
    """Equal weight per clip, then balanced over the states present."""
    w = 1.0 / df.groupby("clip_id")["clip_id"].transform("size").to_numpy()
    for s in np.unique(df["target"]):
        m = (df["target"] == s).to_numpy()
        w[m] *= 1.0 / w[m].sum()
    return w / w.mean()


def fit(df, cols, seed=7):
    """df must have a 'target' column. Returns a model whose predict_probs gives (n,3)."""
    clf = HistGradientBoostingClassifier(max_depth=3, learning_rate=0.08, max_iter=150, min_samples_leaf=20, l2_regularization=1.0, random_state=seed)
    clf.fit(df[cols].to_numpy(float), df["target"].to_numpy(), sample_weight=sample_weights(df))
    return clf


def predict_probs(clf, df, cols):
    p = clf.predict_proba(df[cols].to_numpy(float))
    out = np.full((len(df), 3), 1e-4)
    for k, c in enumerate(clf.classes_):
        out[:, int(c)] = p[:, k]
    return out / out.sum(axis=1, keepdims=True)


def decode_table(df, probs, switch_penalty=2.0, min_build_s=1.0):
    """Decode every clip of df (rows in any order) -> {clip_id: {act_start_s, buildup_start_s, has_buildup}}.
    Several shots of one clip are decoded together in time order: the phases of a clip never go backwards across a cut."""
    out = {}
    for cid, idx in df.groupby("clip_id", sort=False).indices.items():
        g = df.iloc[idx]
        order = np.argsort(g["t_center"].to_numpy(), kind="stable")
        pr = probs[idx][order]
        _, b = decode_clip(pr, g["t_center"].to_numpy()[order], switch_penalty, min_build_s)
        out[cid] = b
    return out


def grouped_cv(df, cols, folds=5, seed=7):
    """Out-of-fold probabilities for every row of df (needs 'target'); folds are made of whole clips."""
    probs = np.zeros((len(df), 3))
    groups = df["clip_id"].to_numpy()
    n_groups = len(np.unique(groups))
    for tr, te in GroupKFold(n_splits=max(2, min(folds, n_groups))).split(df, groups=groups):
        clf = fit(df.iloc[tr], cols, seed)
        probs[te] = predict_probs(clf, df.iloc[te], cols)
    return probs
