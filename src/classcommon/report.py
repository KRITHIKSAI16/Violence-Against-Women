"""Classification pipelines: the evaluation report (markdown + json + csv + ROC figure).

Everything is out-of-fold at clip level. The best model is picked on the same cross-validation it is reported on, so its numbers are optimistic: the report says so.
A claim that behaviour is learned is made only when the best model beats the filming-style and the clip-length baselines and holds on clips of the same resolution.
"""
import csv
import json
from pathlib import Path

import numpy as np

from src.classcommon import metrics as M
from src.classcommon.pipeline import BASELINES

COLS = [("auc", "AUC"), ("pr_auc", "PR-AUC"), ("accuracy", "Acc"), ("balanced_accuracy", "BalAcc"), ("precision", "Prec"), ("recall", "Recall"), ("specificity", "Spec"),
        ("f1", "F1"), ("mcc", "MCC"), ("brier", "Brier"), ("ece", "ECE")]


def method_rows(y, cvres, n_boot, seed):
    rows = {}
    for name, r in cvres.items():
        rows[name] = {"metrics": M.all_metrics(y, r["proba"], r["pred"]), "ci": M.bootstrap(y, r["proba"], r["pred"], n_boot, seed), "auc_per_repeat": r["auc_per_repeat"]}
    return rows


def best_method(rows):
    """Highest AUC among the real models (the baselines are never 'best')."""
    cand = {k: v for k, v in rows.items() if k not in BASELINES and v["metrics"]["auc"] == v["metrics"]["auc"]}
    return max(cand, key=lambda k: cand[k]["metrics"]["auc"]) if cand else None


def verdict(rows, best, same_res):
    if best is None:
        return "No model could be scored."
    base = [rows[b]["metrics"]["auc"] for b in BASELINES if b in rows and rows[b]["metrics"]["auc"] == rows[b]["metrics"]["auc"]]
    top = max(base) if base else float("nan")
    auc, lo = rows[best]["metrics"]["auc"], rows[best]["ci"]["auc"][0]
    beats = (top != top) or auc > top + 0.05
    holds = same_res[0] >= 5 and same_res[1] >= 5 and same_res[2] > 0.6
    if lo > 0.5 and beats and holds:
        return f"`{best}` separates the classes (AUC {auc:.2f}), beats the filming-style and clip-length baselines (best {top:.2f}) and holds among clips of the same resolution."
    if lo > 0.5 and not beats:
        return f"`{best}` separates the classes (AUC {auc:.2f}) but a filming-style or clip-length baseline does as well or better ({top:.2f}): this cannot be read as evidence about behaviour."
    if lo > 0.5:
        return f"`{best}` separates the classes (AUC {auc:.2f}) and beats the baselines, but it does not clearly survive the same-resolution check: treat it as suggestive."
    return f"No model separates the classes clearly (best `{best}`, AUC {auc:.2f}, lower bound {lo:.2f})."


def _fmt(v):
    return "nan" if v != v else f"{v:.3f}"


def _table(rows, order):
    head = "| method | " + " | ".join(n for _, n in COLS) + " |\n|---|" + "---|" * len(COLS)
    lines = [head]
    for name in order:
        m, ci = rows[name]["metrics"], rows[name]["ci"]
        cells = []
        for k, _ in COLS:
            c = _fmt(m[k])
            if k in ci and ci[k][0] == ci[k][0]:
                c += f" ({ci[k][0]:.2f}-{ci[k][1]:.2f})"
            cells.append(c)
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _roc_png(path, y, cvres, names):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve
    fig, ax = plt.subplots(figsize=(5.5, 5))
    for n in names:
        fpr, tpr, _ = roc_curve(y, cvres[n]["proba"])
        ax.plot(fpr, tpr, label=n, linestyle="--" if n in BASELINES else "-")
    ax.plot([0, 1], [0, 1], color="grey", linewidth=0.8)
    ax.set_xlabel("false positive rate")
    ax.set_ylabel("true positive rate")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def write_not_evaluated(out_dir, task, message, recs, docs):
    """No metrics (too few clips of a class): write what exists (counts, documents) and say why."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    cats = {}
    for r in recs:
        cats[r["category"]] = cats.get(r["category"], 0) + 1
    md = [f"# Classification report ({task})", "", f"**{message}**", "", "No accuracy, precision, recall or AUC is reported: with this few clips they would be noise.", "",
          "## Clips per category", ""] + [f"- {c}: {n}" for c, n in sorted(cats.items())] + ["", "## Documents written", "",
          f"{len(docs)} text documents are in the text folder; every clip has its features cached, so a rerun after adding clips is fast."]
    (out / "report.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    (out / "metrics.json").write_text(json.dumps({"task": task, "evaluated": False, "message": message, "clips_per_category": cats}, indent=1), encoding="utf-8")
    return out / "report.md"


def write_report(out_dir, task, cfg, recs, data, cvres, zs, flags, groups_res, docs, notes, message):
    """Evaluate the cross-validated probabilities and write report.md, metrics.json, predictions.csv, errors.csv and roc.png in out_dir."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    cl = cfg["classify"]
    y = np.asarray(data.y).astype(int)
    cats = np.array([r["category"] for r in recs])
    rows = method_rows(y, cvres, int(cl["bootstrap"]), int(cl["cv"]["seed"]))
    best = best_method(rows)
    same = M.same_group_auc(groups_res, y, cvres[best]["proba"]) if best else (0, 0, float("nan"))
    order = sorted(rows, key=lambda k: -(rows[k]["metrics"]["auc"] if rows[k]["metrics"]["auc"] == rows[k]["metrics"]["auc"] else -1))
    md = [f"# Classification report ({task})", "", f"{message}. Captions (Layer B): {'on' if cl['use_captions'] else 'off'}. Every number is out-of-fold at clip level "
          f"({cl['cv']['outer_folds']}-fold grouped cross-validation x {cl['cv']['repeats']} repeats; thresholds chosen inside the training clips). "
          "Intervals are 95% bootstrap over clips.", "", "## Results", "", _table(rows, order), "",
          "The best model is picked on the same cross-validation it is reported on: its numbers are optimistic. `style` (resolution, fps, camera motion, brightness, sharpness) "
          "and `length` (clip duration) are shortcut baselines: Normal clips come from another source.", ""]
    md += ["## Verdict", "", verdict(rows, best, same), ""]
    if best:
        m = rows[best]["metrics"]
        md += [f"## Best model: {best}", "", f"Confusion matrix (rows: true, columns: predicted): non-violent [{m['tn']}, {m['fp']}], violent [{m['fn']}, {m['tp']}].", "",
               f"Same-resolution check: {same[0]} violent and {same[1]} non-violent clips in resolution groups that hold both classes, AUC {_fmt(same[2])}.", "",
               "Recall per violent category: " + ", ".join(f"{c} {v[1]:.2f} (n={v[0]})" for c, v in M.per_category_recall(cats, y, cvres[best]["pred"]).items()) + ".", ""]
        keep = np.array([not (set(flags.get(r["clip_id"], [])) & {"short", "no_pair", "no_result"}) for r in recs])
        if 0 < keep.sum() < len(keep) and len(np.unique(y[keep])) == 2:
            mk = M.all_metrics(y[keep], cvres[best]["proba"][keep], cvres[best]["pred"][keep])
            md += [f"Without the {int((~keep).sum())} flagged clips (short, no measurable pair, no result): AUC {mk['auc']:.3f}, F1 {mk['f1']:.3f}, balanced accuracy {mk['balanced_accuracy']:.3f}.", ""]
    if zs:
        md += ["## Zero-shot reference (no training)", ""] + [f"- {e} prompt margin: AUC {M.score_metrics(y, np.nan_to_num(z, nan=0.0))['auc']:.3f}" for e, z in zs.items()] + [""]
    wrong = [i for i in range(len(y)) if best and cvres[best]["pred"][i] != y[i]]
    if wrong:
        md += [f"## Misclassified clips ({len(wrong)}) with the text the models saw", ""] + [
            f"- **{recs[i]['clip_id']}** (true {'violent' if y[i] else 'non-violent'}, P(violent) {cvres[best]['proba'][i]:.2f}): {docs[recs[i]['clip_id']][:400]}" for i in wrong[:40]] + [""]
    md += ["## Notes and edge cases", ""] + [f"- {n}" for n in notes] + ["", "![ROC](roc.png)", ""]
    (out / "report.md").write_text("\n".join(md), encoding="utf-8")
    (out / "metrics.json").write_text(json.dumps({"task": task, "evaluated": True, "message": message, "best": best, "verdict": verdict(rows, best, same),
                                                   "methods": {k: {"metrics": v["metrics"], "ci": v["ci"], "auc_per_repeat": v["auc_per_repeat"]} for k, v in rows.items()}},
                                                  indent=1, default=float), encoding="utf-8")
    with open(out / "predictions.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["clip_id", "category", "label"] + [f"{n}_p" for n in order] + ["flags"])
        for i, r in enumerate(recs):
            w.writerow([r["clip_id"], r["category"], int(y[i])] + [f"{cvres[n]['proba'][i]:.4f}" for n in order] + [";".join(flags.get(r["clip_id"], []))])
    with open(out / "errors.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["clip_id", "category", "true", "predicted", "p_violent", "text"])
        for i in wrong:
            w.writerow([recs[i]["clip_id"], recs[i]["category"], int(y[i]), int(cvres[best]["pred"][i]), f"{cvres[best]['proba'][i]:.4f}", docs[recs[i]["clip_id"]]])
    _roc_png(out / "roc.png", y, cvres, [n for n in order[:5]] + [b for b in BASELINES if b in cvres and b not in order[:5]])
    return out / "report.md"
