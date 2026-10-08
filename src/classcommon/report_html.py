"""Classification pipelines: the single-file HTML report (`index.html`), written next to report.md.

One self-contained page (inline CSS, charts embedded as base64 PNG, a little inline JS for filtering) that answers three questions:
  1. what is the system doing   (task, stages, what the models see, how it is evaluated, the settings of this run)
  2. how well is it doing       (headline numbers in plain words, all methods against the shortcut baselines, ROC / PR / calibration / score spread)
  3. where is it failing        (missed and falsely alarmed clips with the text the models saw, error rate by category / flag / resolution / length / lead-up,
                                 clips that nearly every model gets wrong, a filterable table of every clip)
Everything is computed from the same out-of-fold predictions as report.md. Nothing here is fitted on test clips.
"""
import base64
import html
import io
from collections import OrderedDict
from pathlib import Path

import numpy as np

from src.classcommon import metrics as M
from src.classcommon.pipeline import BASELINES

esc = html.escape

CSS = """
:root{--bg:#fafaf8;--fg:#1d1d1b;--mut:#6b6b66;--card:#fff;--line:#e2e1db;--acc:#1f5fbf;--bad:#b3261e;--good:#1b7a3e;--warn:#9a6700;--badbg:#fdecea;--goodbg:#e7f5ec;--warnbg:#fff4d6}
@media (prefers-color-scheme:dark){:root{--bg:#161615;--fg:#ecebe6;--mut:#a3a29b;--card:#202020;--line:#34332f;--acc:#7fb0ff;--bad:#ff8b82;--good:#6fd18f;--warn:#e8b84a;--badbg:#3a1d1b;--goodbg:#173323;--warnbg:#3a2f12}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);font:15px/1.55 system-ui,-apple-system,Segoe UI,Roboto,sans-serif}
main{max-width:1080px;margin:0 auto;padding:20px 16px 60px}h1{font-size:1.7rem;margin:.2em 0}h2{font-size:1.3rem;margin:2em 0 .5em;border-bottom:1px solid var(--line);padding-bottom:.2em}
h3{font-size:1.05rem;margin:1.4em 0 .4em}small,.mut{color:var(--mut)}nav{position:sticky;top:0;background:var(--bg);padding:8px 0;border-bottom:1px solid var(--line);z-index:5;font-size:.88rem}
nav a{margin-right:12px;color:var(--acc);text-decoration:none;white-space:nowrap}
.box{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:12px 16px;margin:10px 0}.verdict{border-left:5px solid var(--acc)}
.verdict.good{border-left-color:var(--good)}.verdict.warn{border-left-color:var(--warn)}.verdict.bad{border-left-color:var(--bad)}
.kpis{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:10px;margin:10px 0}.kpi{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:10px 12px}
.kpi b{font-size:1.5rem;display:block}.kpi span{font-size:.8rem;color:var(--mut)}
table{border-collapse:collapse;width:100%;font-size:.85rem;margin:8px 0}th,td{border-bottom:1px solid var(--line);padding:5px 7px;text-align:left;vertical-align:top}
th{background:var(--card);position:sticky;top:0}td.n,th.n{text-align:right;font-variant-numeric:tabular-nums}.scroll{overflow-x:auto}
tr.best td{background:var(--goodbg);font-weight:600}tr.base td{color:var(--mut);font-style:italic}.tag{display:inline-block;font-size:.72rem;border:1px solid var(--line);border-radius:10px;padding:0 7px;margin-right:4px;color:var(--mut)}
.tag.fn{color:var(--bad);border-color:var(--bad)}.tag.fp{color:var(--warn);border-color:var(--warn)}.tag.ok{color:var(--good);border-color:var(--good)}
.bar{background:var(--line);border-radius:3px;height:10px;min-width:80px;position:relative}.bar i{display:block;height:100%;border-radius:3px;background:var(--acc)}.bar i.r{background:var(--bad)}
.cm{border-collapse:collapse;width:auto}.cm td,.cm th{text-align:center;padding:10px 18px;border:1px solid var(--line);font-size:1rem;position:static}
.cm .g{background:var(--goodbg)}.cm .b{background:var(--badbg)}.imgs{display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:12px}.imgs figure{margin:0;background:var(--card);border:1px solid var(--line);border-radius:8px;padding:8px}
.imgs img{width:100%;height:auto;display:block}figcaption{font-size:.8rem;color:var(--mut);margin-top:4px}.clip{border-bottom:1px solid var(--line);padding:7px 0}.clip .txt{font-size:.82rem;color:var(--mut)}
ol.steps{counter-reset:s;list-style:none;padding:0}ol.steps li{counter-increment:s;margin:6px 0;padding:8px 12px 8px 44px;background:var(--card);border:1px solid var(--line);border-radius:8px;position:relative}
ol.steps li:before{content:counter(s);position:absolute;left:12px;top:8px;width:22px;height:22px;border-radius:50%;background:var(--acc);color:var(--bg);text-align:center;font-weight:700;font-size:.8rem;line-height:22px}
details{margin:6px 0}summary{cursor:pointer;color:var(--acc)}input.f,select.f{font:inherit;padding:4px 8px;border:1px solid var(--line);border-radius:6px;background:var(--card);color:var(--fg);margin:2px 6px 8px 0}
code{background:var(--card);border:1px solid var(--line);padding:0 4px;border-radius:4px;font-size:.85em}
"""

JS = """
function flt(){var q=document.getElementById('q').value.toLowerCase(),c=document.getElementById('fc').value,s=document.getElementById('fs').value,rows=document.querySelectorAll('#all tbody tr'),n=0;
rows.forEach(function(r){var ok=(!q||r.textContent.toLowerCase().indexOf(q)>=0)&&(!c||r.dataset.cat===c)&&(!s||r.dataset.st===s);r.style.display=ok?'':'none';if(ok)n++});
document.getElementById('cnt').textContent=n+' of '+rows.length+' clips shown'}
"""

NAMES = {"xclip": "X-CLIP", "vjepa2": "V-JEPA 2", "internvideo2": "InternVideo2"}
METHOD_HELP = {
    "style": "SHORTCUT BASELINE: filming style only (resolution, fps, camera motion, brightness, sharpness). Says nothing about behaviour.",
    "length": "SHORTCUT BASELINE: clip duration only. Says nothing about behaviour.",
    "geom": "Numbers about the main pair of people: distances, closing speed, reaching, contact, how long each relation lasted, lead-up type.",
    "text_tfidf_geom": "Word counts of the geometry text, logistic regression.",
    "text_emb_geom": "Sentence embedding (BGE) of the geometry text, logistic regression.",
    "text_tfidf_full": "Word counts of geometry text + frame captions.",
    "text_emb_full": "Sentence embedding (BGE) of geometry text + frame captions.",
    "fusion_late": "Mean of the probabilities of the video, text and geometry models.",
    "fusion_stack": "A second logistic regression learns how to weigh the single models (fitted inside the training clips).",
    "fusion_early": "All numeric blocks (each reduced by PCA) joined into one vector, one logistic regression.",
}

GLOSSARY = [
    ("AUC (ROC-AUC)", "Chance that a random violent clip gets a higher violence score than a random non-violent one. 0.5 = coin flip, 1.0 = perfect. Does not depend on a threshold."),
    ("PR-AUC", "Like AUC but focused on the violent class: how pure the top-ranked clips are. Compare with the share of violent clips (the no-skill level)."),
    ("Accuracy", "Share of all clips classified correctly. Misleading when one class is much bigger."),
    ("Balanced accuracy", "Mean of recall and specificity: accuracy as if both classes were equally big."),
    ("Precision", "Of the clips called violent, the share that really are."),
    ("Recall (sensitivity)", "Of the violent clips, the share that were caught. 1 - recall = missed violence."),
    ("Specificity", "Of the non-violent clips, the share correctly left alone. 1 - specificity = false alarms."),
    ("F1", "Harmonic mean of precision and recall."),
    ("MCC", "Correlation between prediction and truth, -1 to 1; 0 = no better than chance; robust to class imbalance."),
    ("Brier score", "Mean squared error of the probabilities. Lower is better; always saying 50% gives 0.25."),
    ("ECE", "Expected calibration error: how far the stated probabilities are from the observed violent share. Lower is better."),
    ("Out-of-fold", "Each clip is scored by models that never saw it (nor any near-duplicate of it) in training."),
    ("95% interval", "Percentile bootstrap over clips: re-draw the clips with replacement 1000 times and take the 2.5% and 97.5% values. Wide = few clips = unstable."),
    ("Style / length baselines", "Models that see only how the clip was filmed or how long it is. If they score as high as the real models, the real models may be reading the source of the clip, not behaviour."),
    ("Flags", "short = under 1.5 s; no_pair = no two people tracked together; no_result = no analysis result at all."),
]


# ---------------------------------------------------------------- helpers
def _f(v, d=3):
    return "n/a" if v is None or v != v else f"{v:.{d}f}"


def _pct(v):
    return "n/a" if v is None or v != v else f"{100 * v:.0f}%"


def _ci(ci, k):
    c = ci.get(k)
    return "" if not c or c[0] != c[0] else f" ({c[0]:.2f}-{c[1]:.2f})"


def _fig_b64(fig):
    import matplotlib.pyplot as plt
    buf = io.BytesIO()
    fig.tight_layout()
    fig.savefig(buf, format="png", dpi=110)
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


def _img(fig, caption):
    return f"<figure><img alt='{esc(caption)}' src='data:image/png;base64,{_fig_b64(fig)}'><figcaption>{esc(caption)}</figcaption></figure>"


def _plots(y, cvres, rows, order, best):
    """-> list of <figure> html. Each chart is optional: a failure of one never loses the page."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import precision_recall_curve, roc_curve
    out = []
    shown = list(order[:5]) + [b for b in BASELINES if b in cvres and b not in order[:5]]

    def safe(fn):
        try:
            out.append(fn())
        except Exception as e:  # noqa: BLE001
            out.append(f"<figure><figcaption>chart unavailable: {esc(str(e))}</figcaption></figure>")

    def auc_bars():
        names = [n for n in order if rows[n]["metrics"]["auc"] == rows[n]["metrics"]["auc"]][::-1]
        fig, ax = plt.subplots(figsize=(6, max(2.5, 0.32 * len(names) + 1)))
        a = np.array([rows[n]["metrics"]["auc"] for n in names])
        lo = np.array([rows[n]["ci"]["auc"][0] for n in names])
        hi = np.array([rows[n]["ci"]["auc"][1] for n in names])
        ax.barh(names, a, xerr=[np.clip(a - lo, 0, None), np.clip(hi - a, 0, None)], color=["#8a8a85" if n in BASELINES else ("#1b7a3e" if n == best else "#1f5fbf") for n in names], capsize=2)
        ax.axvline(0.5, color="grey", lw=0.8, ls=":")
        ax.set_xlim(0, 1.02)
        ax.set_xlabel("AUC with 95% interval (grey = shortcut baseline, green = best)")
        return _img(fig, "Every method's AUC. A real model must clearly beat the grey bars.")

    def roc():
        fig, ax = plt.subplots(figsize=(5, 4.6))
        for n in shown:
            fpr, tpr, _ = roc_curve(y, cvres[n]["proba"])
            ax.plot(fpr, tpr, label=n, ls="--" if n in BASELINES else "-", lw=2.2 if n == best else 1.2)
        ax.plot([0, 1], [0, 1], color="grey", lw=0.8)
        ax.set_xlabel("false alarm rate (1 - specificity)")
        ax.set_ylabel("violence caught (recall)")
        ax.legend(fontsize=7)
        return _img(fig, "ROC curve: closer to the top-left corner is better; the diagonal is chance.")

    def pr():
        fig, ax = plt.subplots(figsize=(5, 4.6))
        for n in shown:
            p, r, _ = precision_recall_curve(y, cvres[n]["proba"])
            ax.plot(r, p, label=n, ls="--" if n in BASELINES else "-", lw=2.2 if n == best else 1.2)
        ax.axhline(y.mean(), color="grey", lw=0.8, ls=":")
        ax.set_xlabel("recall")
        ax.set_ylabel("precision")
        ax.set_ylim(0, 1.02)
        ax.legend(fontsize=7)
        return _img(fig, f"Precision-recall curve. Dotted line = share of violent clips ({y.mean():.2f}), the no-skill level.")

    def hist():
        p = np.asarray(cvres[best]["proba"], float)
        fig, ax = plt.subplots(figsize=(5, 4.2))
        bins = np.linspace(0, 1, 21)
        ax.hist(p[y == 0], bins=bins, alpha=0.65, label="non-violent", color="#1f5fbf")
        ax.hist(p[y == 1], bins=bins, alpha=0.65, label="violent", color="#b3261e")
        ax.set_xlabel(f"P(violent) from {best}")
        ax.set_ylabel("clips")
        ax.legend(fontsize=8)
        return _img(fig, "How the best model's scores spread. Little overlap = easy separation; clips of the wrong colour on the wrong side are the errors.")

    def calib():
        p = np.asarray(cvres[best]["proba"], float)
        idx = np.minimum((p * 10).astype(int), 9)
        xs, ys, ns = [], [], []
        for b in range(10):
            m = idx == b
            if m.any():
                xs.append(p[m].mean())
                ys.append(y[m].mean())
                ns.append(int(m.sum()))
        fig, ax = plt.subplots(figsize=(5, 4.2))
        ax.plot([0, 1], [0, 1], color="grey", lw=0.8)
        ax.plot(xs, ys, "o-", color="#1f5fbf")
        for x, v, n in zip(xs, ys, ns):
            ax.annotate(str(n), (x, v), textcoords="offset points", xytext=(4, -10), fontsize=7)
        ax.set_xlabel("predicted P(violent)")
        ax.set_ylabel("observed violent share")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        return _img(fig, "Calibration of the best model (numbers = clips per point). On the diagonal = the probabilities can be taken at face value.")

    safe(auc_bars)
    safe(roc)
    safe(pr)
    if best:
        safe(hist)
        safe(calib)
    return out


def _table(rows, order, best):
    cols = [("auc", "AUC"), ("pr_auc", "PR-AUC"), ("accuracy", "Acc"), ("balanced_accuracy", "BalAcc"), ("precision", "Prec"), ("recall", "Recall"), ("specificity", "Spec"),
            ("f1", "F1"), ("mcc", "MCC"), ("brier", "Brier"), ("ece", "ECE")]
    h = "<div class='scroll'><table><tr><th>method</th>" + "".join(f"<th class='n'>{n}</th>" for _, n in cols) + "</tr>"
    for name in order:
        m, ci = rows[name]["metrics"], rows[name]["ci"]
        cls = "best" if name == best else ("base" if name in BASELINES else "")
        tip = esc(METHOD_HELP.get(name, NAMES.get(name[6:], "") and f"Frozen {NAMES.get(name[6:])} video features, logistic regression." or ""))
        h += f"<tr class='{cls}'><td title='{tip}'>{esc(name)}</td>" + "".join(f"<td class='n'>{_f(m[k])}<small>{_ci(ci, k)}</small></td>" for k, _ in cols) + "</tr>"
    return h + "</table></div>"


def _method_notes(order):
    li = []
    for n in order:
        t = METHOD_HELP.get(n) or (f"Frozen {NAMES.get(n[6:], n[6:])} video features (windows pooled), PCA, logistic regression." if n.startswith("video_") else "")
        if t:
            li.append(f"<li><code>{esc(n)}</code> {esc(t)}</li>")
    return "<details><summary>What each method is</summary><ul>" + "".join(li) + "</ul></details>"


def _breakdown(title, groups, wrong, y, note=""):
    """Error table: groups = list of labels per clip. Rows sorted by clips."""
    d = OrderedDict()
    for i, g in enumerate(groups):
        d.setdefault(g, []).append(i)
    h = f"<h3>{esc(title)}</h3>" + (f"<small>{esc(note)}</small>" if note else "") + "<div class='scroll'><table><tr><th>group</th><th class='n'>clips</th><th class='n'>violent</th><th class='n'>missed (FN)</th><th class='n'>false alarms (FP)</th><th>error rate</th></tr>"
    for g, idx in sorted(d.items(), key=lambda kv: -len(kv[1])):
        idx = np.array(idx)
        w = wrong[idx]
        fn = int((w & (y[idx] == 1)).sum())
        fp = int((w & (y[idx] == 0)).sum())
        rate = w.mean()
        h += (f"<tr><td>{esc(str(g))}</td><td class='n'>{len(idx)}</td><td class='n'>{int((y[idx] == 1).sum())}</td><td class='n'>{fn}</td><td class='n'>{fp}</td>"
              f"<td><div class='bar'><i class='r' style='width:{100 * rate:.0f}%'></i></div><small>{_pct(rate)}</small></td></tr>")
    return h + "</table></div>"


def _clip_div(r, i, y, proba, pred, docs, kind):
    t = "fn" if kind == "FN" else "fp"
    lab = "missed violence" if kind == "FN" else "false alarm"
    return (f"<div class='clip'><span class='tag {t}'>{lab}</span><b>{esc(r['clip_id'])}</b> <small>{esc(r['category'])} &middot; P(violent) {proba[i]:.2f} &middot; {r['duration_s']:.1f} s"
            f"{' &middot; ' + esc(', '.join(r['_flags'])) if r['_flags'] else ''}</small><div class='txt'>{esc(docs[r['clip_id']][:700])}</div></div>")


def _leadup(r):
    res = r.get("result") or {}
    it = res.get("interaction")
    return (it.get("leadup") or {}).get("type", "unknown") if it else "no pair"


def _dur_bucket(d):
    for hi, lab in ((3, "under 3 s"), (6, "3-6 s"), (15, "6-15 s"), (30, "15-30 s")):
        if d < hi:
            return lab
    return "30 s or more"


# ---------------------------------------------------------------- sections
def _system_section(task, cfg, data, recs):
    cl = cfg["classify"]
    enc = [e for e in cl["encoders"] if f"video_{e}" in data.blocks]
    if task == "trim":
        q = "Will violence follow? (a clip is judged on its final seconds, before any violence happens)"
        what = (f"Each violent clip ends at the moment a human annotator marked the start of the violence; each Normal clip is a whole non-violent clip. <b>Both are judged on exactly the last "
                f"{cl['span_s']:g} seconds</b>, so clip length cannot give the answer away. The system therefore sees only the build-up, never the act itself.")
    else:
        q = "Is there violence in this video? (whole, untrimmed videos)"
        what = ("Each video is used whole (re-encoded to 30 fps, 640 px, no trimming). The label is just the category folder: <i>Normal</i> = non-violent, every other category = violent, "
                f"so no start times are needed. Up to {cl['max_windows']} evenly spaced {cl['win_s']:g} s windows are encoded per video. Because the act itself is in the video, scores are expected to be "
                "higher than for the trim task.")
    steps = ["<b>Prepare / load clips.</b> Videos are listed, labelled by folder (Normal = 0, any other category = 1) and re-encoded to a common format.",
             "<b>Track people.</b> A pose model with a multi-object tracker finds the people in every frame and follows them; shots and camera motion are measured.",
             "<b>Understand the scene.</b> Distances are converted to metres, the two people who interact most (the <i>main pair</i>) are chosen, and relations over time are labelled "
             "(follows, approaches from behind, reaches for / contact, walking together, moving apart ...). A <i>lead-up type</i> summarises the pair.",
             "<b>Describe it in words (Layer A).</b> A fixed template turns those numbers into a short text using the same words for every clip. It never mentions violence and uses "
             "&ldquo;person A / person B&rdquo; instead of identities.",
             f"<b>Caption frames (Layer B, optional).</b> A video-language model ({esc(str(cl['caption']['model']))}) describes a few frames in plain sentences; the answer is cleaned of identity and category/act words. "
             f"This run: <b>{'ON' if cl['use_captions'] else 'OFF'}</b>.",
             f"<b>Encode the video.</b> Frozen video models ({esc(', '.join(NAMES.get(e, e) for e in enc) or 'none available')}) turn windows of the video into feature vectors. They are not trained here.",
             "<b>Train and evaluate.</b> Small, regularised logistic-regression models are trained on each information source and on fusions of them, with grouped, repeated cross-validation. "
             "Everything in this page is out-of-fold."]
    cv = cl["cv"]
    sets = [("Task", task), ("Captions (Layer B)", "on" if cl["use_captions"] else "off"), ("Video encoders in use", ", ".join(enc) or "none"), ("Text model", cl["text_model"]),
            ("Window / step / max windows", f"{cl['win_s']:g} s / {cl['step_s']:g} s / {cl['max_windows']}"),
            ("Cross-validation", f"{cv['outer_folds']}-fold grouped x {cv['repeats']} repeats; inner {cv['inner_folds']}-fold for thresholds and stacking"),
            ("Model", f"standardise -> PCA(max {cv['pca_components']}) -> class-balanced logistic regression, C={cv['C']}"),
            ("Duplicate guard", f"clips with frame hashes within {cv['dup_hash_distance']} bits share a fold"), ("Bootstrap resamples", cl["bootstrap"]),
            ("Minimum clips per class", cv["min_per_class"]), ("Seed", cv["seed"])]
    if task == "trim":
        sets.insert(1, ("Judged window", f"last {cl['span_s']:g} s of each clip"))
    h = (f"<h2 id='system'>1. What the system does</h2><div class='box'><b>Question:</b> {q}<br><br>{what}</div>"
         "<h3>The pipeline</h3><ol class='steps'>" + "".join(f"<li>{s}</li>" for s in steps) + "</ol>"
         "<h3>Three kinds of information, compared fairly</h3><ul>"
         "<li><b>Video</b>: what frozen video models &ldquo;see&rdquo; (appearance and motion of the whole scene).</li>"
         "<li><b>Text</b>: the geometry sentence (+ frame captions when on): what the two main people do relative to each other.</li>"
         "<li><b>Geometry numbers</b>: the same pair facts as numbers (distance, closing speed, reaching, contact ...).</li>"
         "<li><b>Shortcut baselines</b> (<code>style</code>, <code>length</code>): how the clip was filmed and how long it is. Normal clips come from another source, so these can separate the classes "
         "without any knowledge of behaviour. They are the bar the real models must clear.</li></ul>"
         "<h3>Settings of this run</h3><div class='scroll'><table>" + "".join(f"<tr><td>{esc(str(k))}</td><td>{esc(str(v))}</td></tr>" for k, v in sets) + "</table></div>")
    return h


def _data_section(recs, y, cats, flags, groups_res):
    n1, n0 = int((y == 1).sum()), int((y == 0).sum())
    dur = np.array([r["duration_s"] for r in recs], float)
    h = (f"<h2 id='data'>2. The data</h2><div class='kpis'><div class='kpi'><b>{len(y)}</b><span>clips</span></div><div class='kpi'><b>{n1}</b><span>violent</span></div>"
         f"<div class='kpi'><b>{n0}</b><span>non-violent (Normal)</span></div><div class='kpi'><b>{_pct(n1 / len(y))}</b><span>violent share (no-skill PR-AUC)</span></div>"
         f"<div class='kpi'><b>{np.median(dur):.1f} s</b><span>median length ({dur.min():.1f}-{dur.max():.1f})</span></div>"
         f"<div class='kpi'><b>{len(set(groups_res))}</b><span>distinct resolutions</span></div></div>")
    h += "<div class='scroll'><table><tr><th>category</th><th class='n'>clips</th><th class='n'>median length (s)</th><th class='n'>no pair found</th><th class='n'>short (&lt;1.5 s)</th></tr>"
    for c in sorted(set(cats)):
        idx = [i for i, r in enumerate(recs) if r["category"] == c]
        h += (f"<tr><td>{esc(c)}{' (non-violent)' if c == 'Normal' else ''}</td><td class='n'>{len(idx)}</td><td class='n'>{np.median(dur[idx]):.1f}</td>"
              f"<td class='n'>{sum('no_pair' in flags.get(recs[i]['clip_id'], []) for i in idx)}</td><td class='n'>{sum('short' in flags.get(recs[i]['clip_id'], []) for i in idx)}</td></tr>")
    h += "</table></div>"
    res = OrderedDict()
    for g, yy in zip(groups_res, y):
        res.setdefault(g, [0, 0])[int(yy)] += 1
    h += ("<details><summary>Resolution groups (filming-style check)</summary><small>The same-resolution check only uses groups that hold both classes.</small><table>"
          "<tr><th>raw resolution</th><th class='n'>violent</th><th class='n'>non-violent</th></tr>"
          + "".join(f"<tr><td>{esc(str(g))}</td><td class='n'>{v[1]}</td><td class='n'>{v[0]}</td></tr>" for g, v in sorted(res.items(), key=lambda kv: -sum(kv[1])))
          + "</table></details>")
    return h


def _perf_section(rows, order, best, verdict_text, same, y, cvres, zs, flags, recs, figs, task):
    h = "<h2 id='perf'>3. How well it works</h2>"
    if not best:
        return h + "<div class='box verdict bad'>No model could be scored.</div>"
    m, ci = rows[best]["metrics"], rows[best]["ci"]
    base = [(b, rows[b]["metrics"]["auc"]) for b in BASELINES if b in rows]
    top = max((a for _, a in base if a == a), default=float("nan"))
    cls = "good" if (ci["auc"][0] > 0.5 and top == top and m["auc"] > top + 0.05) else ("warn" if ci["auc"][0] > 0.5 else "bad")
    h += f"<div class='box verdict {cls}'><b>Verdict.</b> {esc(verdict_text).replace('`', '')}</div>"
    nv, nn = m["tp"] + m["fn"], m["tn"] + m["fp"]
    h += (f"<h3>Best model: <code>{esc(best)}</code></h3><div class='kpis'>"
          f"<div class='kpi'><b>{_f(m['auc'], 2)}</b><span>AUC{_ci(ci, 'auc')}</span></div><div class='kpi'><b>{_f(m['balanced_accuracy'], 2)}</b><span>balanced accuracy{_ci(ci, 'balanced_accuracy')}</span></div>"
          f"<div class='kpi'><b>{_pct(m['recall'])}</b><span>of violent clips caught{_ci(ci, 'recall')}</span></div><div class='kpi'><b>{_pct(m['specificity'])}</b><span>of non-violent clips left alone</span></div>"
          f"<div class='kpi'><b>{_pct(m['precision'])}</b><span>of alarms are real{_ci(ci, 'precision')}</span></div><div class='kpi'><b>{_f(m['f1'], 2)}</b><span>F1{_ci(ci, 'f1')}</span></div>"
          f"<div class='kpi'><b>{_f(top, 2)}</b><span>best shortcut baseline AUC</span></div></div>"
          f"<div class='box'>In plain words: of {nv} violent clips the system caught <b>{m['tp']}</b> and missed <b>{m['fn']}</b>; of {nn} non-violent clips it correctly passed <b>{m['tn']}</b> and raised a "
          f"false alarm on <b>{m['fp']}</b>. These are honest out-of-fold numbers, but the best model was <i>chosen</i> on this same cross-validation, so they are slightly optimistic.</div>")
    h += ("<h3>Confusion matrix</h3><table class='cm'><tr><th></th><th>predicted non-violent</th><th>predicted violent</th></tr>"
          f"<tr><th>truly non-violent</th><td class='g'>{m['tn']}<br><small>correct</small></td><td class='b'>{m['fp']}<br><small>false alarm</small></td></tr>"
          f"<tr><th>truly violent</th><td class='b'>{m['fn']}<br><small>missed</small></td><td class='g'>{m['tp']}<br><small>caught</small></td></tr></table>")
    h += "<h3>All methods</h3><small>Sorted by AUC. Grey italic rows are the shortcut baselines; the green row is the best real model. Hover a method name for what it is. Brackets: 95% interval.</small>"
    h += _table(rows, order, best) + _method_notes(order)
    h += "<h3>Is it learning behaviour, or the filming style?</h3><ul>"
    for b in BASELINES:
        if b in rows:
            d = m["auc"] - rows[b]["metrics"]["auc"]
            h += f"<li><code>{b}</code> baseline AUC {_f(rows[b]['metrics']['auc'])}; best model is {d:+.3f} {'above' if d >= 0 else 'below'} it.</li>"
    h += (f"<li>Same-resolution check: among clips whose resolution group holds both classes ({same[0]} violent, {same[1]} non-violent), the best model's AUC is <b>{_f(same[2])}</b>. "
          "If this drops far below the overall AUC, the model leans on resolution.</li></ul>")
    keep = np.array([not (set(flags.get(r["clip_id"], [])) & {"short", "no_pair", "no_result"}) for r in recs])
    if 0 < keep.sum() < len(keep) and len(np.unique(y[keep])) == 2:
        mk = M.all_metrics(y[keep], cvres[best]["proba"][keep], cvres[best]["pred"][keep])
        h += (f"<p>Without the {int((~keep).sum())} flagged clips (short / no measurable pair / no result): AUC {_f(mk['auc'])}, F1 {_f(mk['f1'])}, balanced accuracy {_f(mk['balanced_accuracy'])} "
              f"(on {int(keep.sum())} clips).</p>")
    if zs:
        h += ("<h3>Zero-shot reference (no training at all)</h3><ul>" + "".join(f"<li>{esc(NAMES.get(e, e))} prompt margin: AUC {_f(M.score_metrics(y, np.nan_to_num(z, nan=0.0))['auc'])}</li>" for e, z in zs.items())
              + "</ul><small>A model that only compares the video with &ldquo;violent&rdquo; / &ldquo;calm&rdquo; text prompts. Trained models should beat it.</small>")
    h += "<h3>Charts</h3><div class='imgs'>" + "".join(figs) + "</div>"
    return h


def _fail_section(recs, y, cats, cvres, best, docs, flags, groups_res, rows):
    h = "<h2 id='fail'>4. Where it fails</h2>"
    if not best:
        return h + "<p>No best model, so no error analysis.</p>"
    proba, pred = np.asarray(cvres[best]["proba"], float), np.asarray(cvres[best]["pred"]).astype(int)
    wrong = pred != y
    for r in recs:
        r["_flags"] = flags.get(r["clip_id"], [])
    fn = [i for i in np.where(wrong & (y == 1))[0]]
    fp = [i for i in np.where(wrong & (y == 0))[0]]
    fn.sort(key=lambda i: proba[i])      # most confidently missed first
    fp.sort(key=lambda i: -proba[i])     # most confident false alarm first
    h += (f"<div class='box'>Best model <code>{esc(best)}</code> is wrong on <b>{int(wrong.sum())}</b> of {len(y)} clips: <b>{len(fn)}</b> violent clips missed and <b>{len(fp)}</b> false alarms. "
          "Below, errors are listed most-confident first (the model was most sure and still wrong), with the exact text the models saw.</div>")
    h += _breakdown("Error rate by category", [r["category"] for r in recs], wrong, y, "Violent categories: missed clips (FN). Normal: false alarms (FP).")
    rec = M.per_category_recall(cats, y, pred)
    if rec:
        h += "<h3>Recall per violent category</h3><table><tr><th>category</th><th class='n'>violent clips</th><th>recall</th></tr>"
        for c, (n, v) in sorted(rec.items(), key=lambda kv: kv[1][1]):
            h += f"<tr><td>{esc(c)}</td><td class='n'>{n}</td><td><div class='bar'><i style='width:{100 * v:.0f}%'></i></div><small>{_pct(v)}</small></td></tr>"
        h += "</table>"
    pair_state = ["pair measured" if "no_pair" not in f and "no_result" not in f else ("no analysis result" if "no_result" in f else "no pair found") for f in (flags.get(r["clip_id"], []) for r in recs)]
    h += _breakdown("Error rate by whether a pair of people was found", pair_state, wrong, y, "If errors pile up where no pair was tracked, the geometry/text sources had nothing to work with.")
    h += _breakdown("Error rate by lead-up type", [_leadup(r) for r in recs], wrong, y, "The lead-up type is the pair summary chosen by the rules (e.g. approach, follow, no pair).")
    h += _breakdown("Error rate by clip length", [_dur_bucket(r["duration_s"]) for r in recs], wrong, y)
    h += _breakdown("Error rate by raw resolution", groups_res, wrong, y, "A resolution where one class dominates is a place where the shortcut baselines work.")
    h += _breakdown("Error rate by flag", [", ".join(flags.get(r["clip_id"], [])) or "none" for r in recs], wrong, y)
    real = [k for k in rows if k not in BASELINES]
    nwrong = np.zeros(len(y), int)
    for k in real:
        nwrong += (np.asarray(cvres[k]["pred"]).astype(int) != y)
    hard = [i for i in np.argsort(-nwrong) if nwrong[i] >= max(2, int(0.75 * len(real)))][:25]
    if hard:
        h += (f"<h3>Hardest clips: wrong in at least 75% of the {len(real)} real models</h3><small>Persistent errors usually mean the clip is ambiguous, mislabelled, or shows something the "
              "sources cannot capture (few people, occlusion, off-screen action).</small>")
        h += "<div class='scroll'><table><tr><th>clip</th><th>category</th><th>true</th><th class='n'>models wrong</th><th class='n'>best P(violent)</th><th>flags</th></tr>"
        for i in hard:
            h += (f"<tr><td>{esc(recs[i]['clip_id'])}</td><td>{esc(recs[i]['category'])}</td><td>{'violent' if y[i] else 'non-violent'}</td><td class='n'>{nwrong[i]}/{len(real)}</td>"
                  f"<td class='n'>{proba[i]:.2f}</td><td>{esc(', '.join(recs[i]['_flags']))}</td></tr>")
        h += "</table></div>"
    else:
        h += "<p><small>No clip is wrong in most models: the errors are specific to the best model.</small></p>"
    h += f"<h3>Missed violence ({len(fn)})</h3>" + ("".join(_clip_div(recs[i], i, y, proba, pred, docs, "FN") for i in fn[:40]) or "<p>None.</p>")
    h += f"<h3>False alarms ({len(fp)})</h3>" + ("".join(_clip_div(recs[i], i, y, proba, pred, docs, "FP") for i in fp[:40]) or "<p>None.</p>")
    if len(fn) > 40 or len(fp) > 40:
        h += "<small>Lists are capped at 40 each; <code>errors.csv</code> has all of them.</small>"
    return h


def _all_clips(recs, y, cvres, best, order, docs, flags):
    h = "<h2 id='clips'>5. Every clip</h2>"
    if not best:
        return h
    proba, pred = np.asarray(cvres[best]["proba"], float), np.asarray(cvres[best]["pred"]).astype(int)
    cats = sorted({r["category"] for r in recs})
    h += ("<input class='f' id='q' placeholder='search clip or text' oninput='flt()'><select class='f' id='fc' onchange='flt()'><option value=''>all categories</option>"
          + "".join(f"<option>{esc(c)}</option>" for c in cats) + "</select><select class='f' id='fs' onchange='flt()'><option value=''>all outcomes</option><option value='caught'>caught</option>"
          "<option value='missed'>missed</option><option value='false alarm'>false alarm</option><option value='correct'>correctly passed</option></select> <small id='cnt'></small>")
    h += "<div class='scroll'><table id='all'><thead><tr><th>clip</th><th>category</th><th>true</th><th class='n'>P(violent)</th><th>outcome</th><th>flags</th><th>text the models saw</th></tr></thead><tbody>"
    for i, r in enumerate(recs):
        st = ("caught" if pred[i] == 1 else "missed") if y[i] else ("false alarm" if pred[i] == 1 else "correct")
        tg = "ok" if st in ("caught", "correct") else ("fn" if st == "missed" else "fp")
        h += (f"<tr data-cat='{esc(r['category'])}' data-st='{st}'><td>{esc(r['clip_id'])}</td><td>{esc(r['category'])}</td><td>{'violent' if y[i] else 'non-violent'}</td><td class='n'>{proba[i]:.2f}</td>"
              f"<td><span class='tag {tg}'>{st}</span></td><td>{esc(', '.join(flags.get(r['clip_id'], [])))}</td><td><small>{esc(docs[r['clip_id']][:300])}</small></td></tr>")
    return h + "</tbody></table></div>"


def _glossary():
    return ("<h2 id='gloss'>7. How to read the numbers</h2><table>" + "".join(f"<tr><td><b>{esc(k)}</b></td><td>{esc(v)}</td></tr>" for k, v in GLOSSARY) + "</table>")


def _files():
    fl = [("report.md", "the same findings as plain markdown"), ("metrics.json", "all metrics and intervals for every method"), ("predictions.csv", "per clip probability from every method"),
          ("errors.csv", "every misclassified clip with its text"), ("roc.png", "ROC figure"), ("../text/", "one JSON per clip: the exact document the models read"),
          ("../features/", "cached video / text features"), ("../class_config.yaml", "the full configuration of this run")]
    return "<h2 id='files'>8. Files of this run</h2><ul>" + "".join(f"<li><code>{esc(a)}</code> {esc(b)}</li>" for a, b in fl) + "</ul>"


def _page(task, title, body):
    nav = "".join(f"<a href='#{a}'>{b}</a>" for a, b in (("system", "System"), ("data", "Data"), ("perf", "Performance"), ("fail", "Failures"), ("clips", "All clips"), ("caveats", "Caveats"), ("gloss", "Glossary"), ("files", "Files")))
    return (f"<!doctype html><html lang='en'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'><title>{esc(title)}</title>"
            f"<style>{CSS}</style></head><body><main><h1>{esc(title)}</h1><nav>{nav}</nav>{body}</main><script>{JS}if(document.getElementById('q'))flt();</script></body></html>")


# ---------------------------------------------------------------- entry points
def write_html(out_dir, task, cfg, recs, data, cvres, rows, order, best, verdict_text, same, zs, flags, groups_res, docs, notes, message):
    """Write out_dir/index.html. Returns its path."""
    y = np.asarray(data.y).astype(int)
    cats = np.array([r["category"] for r in recs])
    title = "Violence classification report: " + ("whole videos" if task == "full" else "clips cut before the violence")
    try:
        figs = _plots(y, cvres, rows, order, best)
    except Exception as e:  # noqa: BLE001
        figs = [f"<figure><figcaption>charts unavailable: {esc(str(e))}</figcaption></figure>"]
    cl = cfg["classify"]
    intro = (f"<div class='box'><b>{esc(message)}.</b> Captions (Layer B): {'on' if cl['use_captions'] else 'off'}. This one page explains what the system does, how well it does it, and where it fails. "
             "Every number is out-of-fold at clip level.</div>")
    body = (intro + _system_section(task, cfg, data, recs) + _data_section(recs, y, cats, flags, groups_res)
            + _perf_section(rows, order, best, verdict_text, same, y, cvres, zs, flags, recs, figs, task)
            + _fail_section(recs, y, cats, cvres, best, docs, flags, groups_res, rows) + _all_clips(recs, y, cvres, best, order, docs, flags)
            + _caveats_simple(notes, task) + _glossary() + _files())
    p = Path(out_dir) / "index.html"
    p.write_text(_page(task, title, body), encoding="utf-8")
    return p


def _caveats_simple(notes, task):
    fixed = ["The best method is picked on the same cross-validation it is reported on, so its numbers are optimistic. Treat the interval, not the point value, as the result.",
             "Normal clips come from a different source than the violent categories. This is why the style and length baselines exist, and why a high AUC alone proves nothing about behaviour.",
             "With few clips per class, percentages move a lot with one clip; look at the counts and the intervals.",
             "The text the models read is generated from tracked people; if tracking misses people (small, distant, occluded) the text says so and the model has little to go on."]
    if task == "trim":
        fixed.append("Trim task: the cut is at the human-marked start of violence. If it is late, act words can leak into captions; the act-word check in the notes counts these.")
    h = "<h2 id='caveats'>6. Caveats and notes from this run</h2><h3>Checks run on this data</h3><ul>" + "".join(f"<li>{esc(n)}</li>" for n in notes) + "</ul>"
    return h + "<h3>Always keep in mind</h3><ul>" + "".join(f"<li>{esc(n)}</li>" for n in fixed) + "</ul>"


def write_not_evaluated_html(out_dir, task, message, recs, docs):
    cats = {}
    for r in recs:
        cats[r["category"]] = cats.get(r["category"], 0) + 1
    body = (f"<div class='box verdict bad'><b>{esc(message)}</b><br>No accuracy, precision, recall or AUC is reported: with this few clips they would be noise.</div>"
            "<h2>Clips per category</h2><table>" + "".join(f"<tr><td>{esc(c)}</td><td class='n'>{n}</td></tr>" for c, n in sorted(cats.items())) + "</table>"
            f"<p>{len(docs)} text documents were written to the text folder; every clip has its features cached, so a rerun after adding clips is fast.</p>")
    p = Path(out_dir) / "index.html"
    p.write_text(_page(task, "Violence classification report: not evaluated", body), encoding="utf-8")
    return p
