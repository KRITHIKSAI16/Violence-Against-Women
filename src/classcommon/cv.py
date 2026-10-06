"""Classification pipelines: the models, the fusion of modalities and the nested, grouped cross-validation.

Data: named BLOCKS, one row per clip. A "num" block is a float matrix (video embedding, text embedding, geometry, a baseline); a "text" block is an array of strings (TF-IDF).
Every model is a scikit-learn pipeline fitted on training clips only: median imputation (+ missing indicators) -> standardisation -> PCA (k adapts to the
training size) -> class-balanced logistic regression. Nothing is fitted on a test clip: not the imputer, the scaler, the PCA, the threshold or the fusion weights.

Methods:  single  one block          late   mean of the single-block probabilities
          early   all numeric blocks, each reduced by its own PCA, concatenated, one logistic regression
          stack   a logistic regression over the out-of-fold probabilities of the blocks (fitted inside the training clips)
Threshold: chosen on out-of-fold probabilities INSIDE the training clips (balanced accuracy), then applied to the test clips.
Evaluation: repeated stratified GROUP k-fold (a group = a clip, or several clips that are near duplicates); per clip the probabilities are averaged over the
repeats and the decisions are a majority vote over the repeats.
"""
from dataclasses import dataclass, field

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


class AdaptivePCA(BaseEstimator, TransformerMixin):
    """PCA with at most k components, never more than the training rows - 1 or the columns (so tiny folds do not crash).
    A block that already has k or fewer columns is passed through unchanged: an unsupervised PCA on few columns and few clips can only throw signal away."""

    def __init__(self, k=16, seed=7):
        self.k = k
        self.seed = seed

    def fit(self, X, y=None):
        n, d = X.shape
        self.pca_ = None if d <= self.k else PCA(n_components=int(max(1, min(self.k, n - 1, d))), random_state=self.seed).fit(X)
        return self

    def transform(self, X):
        return X if self.pca_ is None else self.pca_.transform(X)


def _block_pre(cv):
    steps = [("imp", SimpleImputer(strategy="median", add_indicator=True)), ("sc", StandardScaler())]
    if cv.get("pca_components"):
        steps.append(("pca", AdaptivePCA(int(cv["pca_components"]), int(cv["seed"]))))
    return Pipeline(steps)


def make_estimator(kind, cv):
    lr = LogisticRegression(C=float(cv["C"]), class_weight="balanced", max_iter=2000)
    if kind == "num":
        return Pipeline([("pre", _block_pre(cv)), ("lr", lr)])
    if kind == "text":
        return Pipeline([("tf", TfidfVectorizer(ngram_range=(1, 2), min_df=1, sublinear_tf=True)), ("lr", lr)])
    raise ValueError(f"unknown block kind {kind!r}")


@dataclass
class Data:
    """blocks: {name: array (n, d) or array of n strings}; kinds: {name: 'num' | 'text'}; y: (n,) 0/1."""
    blocks: dict
    kinds: dict
    y: np.ndarray
    ids: list = field(default_factory=list)


@dataclass
class Method:
    name: str
    kind: str                # single | late | early | stack
    blocks: list


def _logit(p):
    p = np.clip(np.asarray(p, float), 1e-3, 1 - 1e-3)
    return np.log(p / (1 - p))


def _single(data, block, tr, te, cv):
    est = make_estimator(data.kinds[block], cv).fit(data.blocks[block][tr], data.y[tr])
    return est.predict_proba(data.blocks[block][te])[:, 1]


def _inner_splits(y_tr, k, seed):
    """Stratified inner folds as index arrays into the training clips; None when a class is too small."""
    y_tr = np.asarray(y_tr).astype(int)
    kk = int(min(k, np.bincount(y_tr, minlength=2).min()))
    if kk < 2:
        return None
    return list(StratifiedKFold(n_splits=kk, shuffle=True, random_state=seed).split(np.zeros(len(y_tr)), y_tr))


def method_proba(m, data, tr, te, cv):
    """P(violent) for the test clips `te`, every model fitted on the training clips `tr` only."""
    if m.kind == "single":
        return _single(data, m.blocks[0], tr, te, cv)
    if m.kind == "late":
        return np.mean([_single(data, b, tr, te, cv) for b in m.blocks], axis=0)
    if m.kind == "early":
        mats = [np.asarray(data.blocks[b], float) for b in m.blocks]
        cols, start = [], 0
        for a in mats:
            cols.append(list(range(start, start + a.shape[1])))
            start += a.shape[1]
        X = np.hstack(mats)
        pre = ColumnTransformer([(f"b{i}", _block_pre(cv), c) for i, c in enumerate(cols)])
        est = Pipeline([("pre", pre), ("lr", LogisticRegression(C=float(cv["C"]), class_weight="balanced", max_iter=2000))]).fit(X[tr], data.y[tr])
        return est.predict_proba(X[te])[:, 1]
    if m.kind == "stack":
        base_te = np.column_stack([_single(data, b, tr, te, cv) for b in m.blocks])
        splits = _inner_splits(data.y[tr], int(cv["inner_folds"]), int(cv["seed"]))
        if splits is None:
            return base_te.mean(axis=1)
        oof = np.full((len(tr), len(m.blocks)), np.nan)
        for a, b in splits:
            for j, blk in enumerate(m.blocks):
                oof[b, j] = _single(data, blk, tr[a], tr[b], cv)
        meta = LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000).fit(_logit(oof), data.y[tr])
        return meta.predict_proba(_logit(base_te))[:, 1]
    raise ValueError(f"unknown method kind {m.kind!r}")


def choose_threshold(y, p):
    """Threshold on probabilities maximising balanced accuracy (ties: the one closest to 0.5). Candidates: the midpoints between sorted scores and 0.5."""
    y, p = np.asarray(y).astype(int), np.asarray(p, float)
    s = np.unique(p)
    cand = np.unique(np.concatenate([[0.5], (s[:-1] + s[1:]) / 2, [s.min() - 1e-9, s.max() + 1e-9]]))
    best, best_key = 0.5, None
    for t in cand:
        pred = p >= t
        tpr = pred[y == 1].mean() if (y == 1).any() else 0.0
        tnr = (~pred[y == 0]).mean() if (y == 0).any() else 0.0
        key = ((tpr + tnr) / 2, -abs(t - 0.5))
        if best_key is None or key > best_key:
            best, best_key = float(t), key
    return best


def fold_predict(m, data, tr, te, cv):
    """-> (P(violent) of the test clips, threshold chosen from out-of-fold probabilities inside the training clips (0.5 when they cannot be made))."""
    p_te = method_proba(m, data, tr, te, cv)
    splits = _inner_splits(data.y[tr], int(cv["inner_folds"]), int(cv["seed"]))
    if splits is None:
        return p_te, 0.5
    oof = np.full(len(tr), np.nan)
    for a, b in splits:
        oof[b] = method_proba(m, data, tr[a], tr[b], cv)
    return p_te, choose_threshold(data.y[tr], oof)


def cross_validate(methods, data, groups, cv):
    """-> {method name: {proba (n,), pred (n,), auc_per_repeat [..]}}. Raises ValueError when a class has fewer clips than the number of folds."""
    from sklearn.metrics import roc_auc_score
    y = np.asarray(data.y).astype(int)
    n = len(y)
    k = int(min(cv["outer_folds"], (y == 1).sum(), (y == 0).sum()))
    if k < 2:
        raise ValueError("each class needs at least 2 clips to cross-validate")
    reps = int(cv["repeats"])
    P = {m.name: np.full((reps, n), np.nan) for m in methods}
    D = {m.name: np.full((reps, n), np.nan) for m in methods}
    for r in range(reps):
        splitter = StratifiedGroupKFold(n_splits=k, shuffle=True, random_state=int(cv["seed"]) + r)
        for tr, te in splitter.split(np.zeros(n), y, groups):
            if len(np.unique(y[tr])) < 2:
                continue
            for m in methods:
                p, thr = fold_predict(m, data, tr, te, cv)
                P[m.name][r, te] = p
                D[m.name][r, te] = (p >= thr).astype(float)
    out = {}
    for m in methods:
        proba = np.nanmean(P[m.name], axis=0)
        vote = (np.nanmean(D[m.name], axis=0) >= 0.5).astype(int)
        aucs = [float(roc_auc_score(y, P[m.name][r])) for r in range(reps) if not np.isnan(P[m.name][r]).any() and len(np.unique(y)) == 2]
        out[m.name] = {"proba": proba, "pred": vote, "auc_per_repeat": aucs}
    return out
