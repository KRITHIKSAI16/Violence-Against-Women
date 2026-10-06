"""Classification pipelines: every metric of the report, computed from clip-level predictions (y in {0, 1}, 1 = violent).

Threshold metrics (need a hard decision): accuracy, balanced accuracy, precision, recall (= sensitivity), specificity, F1, MCC.
Score metrics (use the probability): ROC-AUC, PR-AUC (average precision), Brier score, expected calibration error.
Intervals: 95% percentile bootstrap over CLIPS (never windows). A resample that holds a single class is skipped.
Conventions: precision is 0 when nothing is predicted positive; MCC is 0 when a margin of the confusion matrix is empty; AUC is NaN with one class.
"""
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

THRESHOLD_KEYS = ["accuracy", "balanced_accuracy", "precision", "recall", "specificity", "f1", "mcc"]
SCORE_KEYS = ["auc", "pr_auc", "brier", "ece"]


def confusion(y, pred):
    y, pred = np.asarray(y).astype(int), np.asarray(pred).astype(int)
    return {"tp": int(((y == 1) & (pred == 1)).sum()), "fp": int(((y == 0) & (pred == 1)).sum()),
            "tn": int(((y == 0) & (pred == 0)).sum()), "fn": int(((y == 1) & (pred == 0)).sum())}


def _div(a, b):
    return float(a / b) if b else 0.0


def threshold_metrics(y, pred):
    c = confusion(y, pred)
    tp, fp, tn, fn = c["tp"], c["fp"], c["tn"], c["fn"]
    prec, rec, spec = _div(tp, tp + fp), _div(tp, tp + fn), _div(tn, tn + fp)
    den = float(np.sqrt(float(tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)))
    return {"accuracy": _div(tp + tn, tp + fp + tn + fn), "balanced_accuracy": (rec + spec) / 2, "precision": prec, "recall": rec, "specificity": spec,
            "f1": _div(2 * prec * rec, prec + rec), "mcc": float((tp * tn - fp * fn) / den) if den else 0.0, **c}


def ece(y, p, bins=10):
    """Expected calibration error: bin-weighted gap between the mean predicted probability and the observed violent share."""
    y, p = np.asarray(y, float), np.asarray(p, float)
    idx = np.minimum((p * bins).astype(int), bins - 1)
    return float(sum((idx == b).mean() * abs(y[idx == b].mean() - p[idx == b].mean()) for b in range(bins) if (idx == b).any()))


def score_metrics(y, p):
    y, p = np.asarray(y).astype(int), np.asarray(p, float)
    two = len(np.unique(y)) == 2
    return {"auc": float(roc_auc_score(y, p)) if two else float("nan"), "pr_auc": float(average_precision_score(y, p)) if two else float("nan"),
            "brier": float(np.mean((p - y) ** 2)), "ece": ece(y, p)}


def all_metrics(y, p, pred):
    return {**threshold_metrics(y, pred), **score_metrics(y, p)}


def bootstrap(y, p, pred, n=1000, seed=7, keys=("auc", "pr_auc", "accuracy", "balanced_accuracy", "precision", "recall", "f1", "mcc")):
    """{metric: (low, high)}: 95% percentile interval over resamples of clips. NaN when no resample had both classes."""
    y, p, pred = np.asarray(y).astype(int), np.asarray(p, float), np.asarray(pred).astype(int)
    rng = np.random.default_rng(seed)
    vals = {k: [] for k in keys}
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        if len(np.unique(y[i])) < 2:
            continue
        m = all_metrics(y[i], p[i], pred[i])
        for k in keys:
            vals[k].append(m[k])
    return {k: ((float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))) if v else (float("nan"), float("nan"))) for k, v in vals.items()}


def per_category_recall(categories, y, pred):
    """{category: (violent clips, recall)} over the violent clips of each category."""
    cats, y, pred = np.asarray(categories), np.asarray(y).astype(int), np.asarray(pred).astype(int)
    return {c: (int(((cats == c) & (y == 1)).sum()), float(pred[(cats == c) & (y == 1)].mean())) for c in sorted(set(cats[y == 1]))}


def same_group_auc(groups, y, p):
    """AUC over only those groups (for example a raw resolution) that hold both classes: the filming-style check. -> (violent n, normal n, auc)."""
    groups, y, p = np.asarray(groups, object), np.asarray(y).astype(int), np.asarray(p, float)
    keep = np.zeros(len(y), bool)
    for g in set(groups.tolist()):
        m = groups == g
        if (y[m] == 1).any() and (y[m] == 0).any():
            keep |= m
    if not keep.any():
        return 0, 0, float("nan")
    return int((y[keep] == 1).sum()), int((y[keep] == 0).sum()), float(roc_auc_score(y[keep], p[keep]))
