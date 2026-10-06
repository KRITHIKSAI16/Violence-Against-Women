"""Curated dataset: what can be measured now that every violent clip has a human start time T? And how does the pre-violence footage differ from Normal?

Measurements (all counts shown, all intervals bootstrapped over CLIPS, never windows):
  coverage        per category: share of clips where an interacting pair was found before T, and median seconds of pair data
  trend           inside each violent clip (shortcut-proof, same source and style): concern in the last 3 s before T vs the earlier part of the same clip
  leadup_types    which lead-up type was assigned, per category and for Normal
  normal_vs_pre   concern score of the last window before T (violent) vs the HIGHEST window of each Normal clip (against us), AUC with a bootstrap interval,
                  next to a style-only baseline (how the clip was filmed) and the same comparison restricted to clips of the same raw resolution
A claim that behavior separates pre-violence from Normal is only made when it beats the style baseline and survives the same-resolution subset.
"""
import numpy as np

STYLE_KEYS = ["raw_width", "raw_height", "raw_fps", "cam_moving", "brightness", "sharpness"]


def auc(pos, neg):
    """P(score of a random positive > score of a random negative), ties count half. NaN when a group is empty."""
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    if not len(pos) or not len(neg):
        return float("nan")
    gt = (pos[:, None] > neg[None, :]).mean()
    eq = (pos[:, None] == neg[None, :]).mean()
    return float(gt + 0.5 * eq)


def bootstrap_auc(pos, neg, n=1000, seed=7):
    """(auc, low, high) with a 95% percentile interval over clip resamples."""
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    base = auc(pos, neg)
    if len(pos) < 2 or len(neg) < 2:
        return base, float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    v = [auc(rng.choice(pos, len(pos)), rng.choice(neg, len(neg))) for _ in range(n)]
    return base, float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))


def sign_test_p(wins, n):
    """Two-sided exact binomial p of `wins` out of n under p=0.5."""
    from math import comb
    if n == 0:
        return float("nan")
    k = max(wins, n - wins)
    return float(min(1.0, 2 * sum(comb(n, i) for i in range(k, n + 1)) / 2 ** n))


def coverage(results):
    out = {}
    for cat in sorted({r["category"] for r in results}):
        rs = [r for r in results if r["category"] == cat]
        with_pair = [r for r in rs if not r["no_pair"]]
        out[cat] = {"clips": len(rs), "with_pair": len(with_pair), "share": len(with_pair) / len(rs),
                    "median_seconds": float(np.median([r["duration_s"] for r in with_pair])) if with_pair else float("nan")}
    return out


def trend(results, last_s=3.0, win_s=2.0):
    """Per violent clip: mean concern of windows inside the last `last_s` seconds minus the mean of earlier windows (needs both). Returns dict with counts."""
    diffs = []
    for r in results:
        if r["is_normal"] or r["no_pair"]:
            continue
        w = r["interaction"]["concern_windows"]
        dur = r["interaction"]["duration_s"]
        late = [s for t, s in w if t + win_s >= dur - last_s + 1e-6]
        early = [s for t, s in w if t + win_s <= dur - last_s + 1e-6]
        if late and early:
            diffs.append(float(np.mean(late) - np.mean(early)))
    d = np.asarray(diffs)
    wins = int((d > 0).sum())
    ties = int((d == 0).sum())
    return {"clips": len(d), "higher_in_last": wins, "equal": ties, "median_diff": float(np.median(d)) if len(d) else float("nan"),
            "p_sign": sign_test_p(wins, len(d) - ties), "diffs": diffs}


def leadup_types(results):
    out = {}
    for r in results:
        key = "Normal" if r["is_normal"] else r["category"]
        t = r["interaction"]["leadup"]["type"] if r.get("interaction") else ("no pair: " + r.get("reason", "?"))
        out.setdefault(key, {})
        out[key][t] = out[key].get(t, 0) + 1
    return out


def concern_scores(results):
    """-> (violent last-window scores, normal highest-window scores, matching result lists) for clips with a pair."""
    v = [r for r in results if not r["is_normal"] and r.get("interaction") and r["interaction"]["concern_last_s"] is not None]
    n = [r for r in results if r["is_normal"] and r.get("interaction") and r["interaction"]["concern_max_window"] is not None]
    return [r["interaction"]["concern_last_s"] for r in v], [r["interaction"]["concern_max_window"] for r in n], v, n


def style_vector(r):
    raw, cam = r.get("raw") or {}, r.get("camera") or {}
    return [raw.get("raw_width"), raw.get("raw_height"), raw.get("raw_fps"), float(bool(cam.get("moving"))), cam.get("brightness"), cam.get("sharpness")]


def style_baseline(vres, nres):
    """How well can HOW THE CLIP WAS FILMED alone tell violent from Normal clips? The best single style feature's AUC (folded to >= 0.5, so the
    direction does not matter; taking the best of several is deliberately generous to the shortcut). NaN when a group is too small or has no style data."""
    best = float("nan")
    for k in range(len(STYLE_KEYS)):
        pos = [style_vector(r)[k] for r in vres]
        neg = [style_vector(r)[k] for r in nres]
        pos = [x for x in pos if x is not None and np.isfinite(x)]
        neg = [x for x in neg if x is not None and np.isfinite(x)]
        if len(pos) < 2 or len(neg) < 2:
            continue
        a = auc(pos, neg)
        a = max(a, 1 - a)
        best = a if best != best else max(best, a)
    return best


def same_resolution_auc(vres, nres, vs, ns):
    """AUC restricted to groups (raw width x height) that contain both a violent and a Normal clip."""
    groups = {}
    for r, s in [(r, s) for r, s in zip(vres, vs)]:
        groups.setdefault((r["raw"]["raw_width"], r["raw"]["raw_height"]), ([], []))[0].append(s)
    for r, s in zip(nres, ns):
        groups.setdefault((r["raw"]["raw_width"], r["raw"]["raw_height"]), ([], []))[1].append(s)
    pos = [s for p, n in groups.values() if p and n for s in p]
    neg = [s for p, n in groups.values() if p and n for s in n]
    return {"violent": len(pos), "normal": len(neg), "auc": auc(pos, neg)}


def normal_vs_pre(results, extra_scores=None):
    vs, ns, vres, nres = concern_scores(results)
    a, lo, hi = bootstrap_auc(vs, ns)
    out = {"violent_n": len(vs), "normal_n": len(ns), "auc": a, "auc_low": lo, "auc_high": hi, "style_only_auc": style_baseline(vres, nres),
           "same_resolution": same_resolution_auc(vres, nres, vs, ns), "median_violent": float(np.median(vs)) if vs else float("nan"),
           "median_normal": float(np.median(ns)) if ns else float("nan"), "violent_scores": vs, "normal_scores": ns}
    out["verdict"] = verdict(out)
    return out


def verdict(o):
    """A plain-words statement that only claims what the numbers support."""
    if o["violent_n"] < 5 or o["normal_n"] < 5:
        return "Too few clips with a measurable pair to compare pre-violence footage with Normal."
    sr = o["same_resolution"]
    beats_style = o["style_only_auc"] != o["style_only_auc"] or o["auc"] > o["style_only_auc"] + 0.05
    holds = sr["violent"] >= 5 and sr["normal"] >= 5 and sr["auc"] > 0.6
    if o["auc_low"] > 0.5 and beats_style and holds:
        return "The geometric concern score is higher before violence than in Normal clips, beats the style-only baseline and holds among clips of the same resolution."
    if o["auc_low"] > 0.5 and not beats_style:
        return "The geometric concern score separates the groups, but filming style alone separates them as well or better: this cannot be read as evidence about behavior."
    if o["auc_low"] > 0.5:
        return "The geometric concern score is higher before violence than in Normal clips, but it does not clearly survive the same-resolution check; treat it as suggestive."
    return "The geometric concern score does not clearly separate pre-violence footage from Normal clips in this set."


def evaluate_all(results):
    return {"clips": len(results), "violent": sum(not r["is_normal"] for r in results), "normal": sum(r["is_normal"] for r in results),
            "coverage": coverage([r for r in results if not r["is_normal"]]) if any(not r["is_normal"] for r in results) else {},
            "normal_coverage": coverage([r for r in results if r["is_normal"]]).get("Normal") if any(r["is_normal"] for r in results) else None,
            "trend": trend(results), "leadup_types": leadup_types(results), "normal_vs_pre": normal_vs_pre(results)}


def markdown(ev):
    t, n = ev["trend"], ev["normal_vs_pre"]
    lines = ["# Evaluation of the pre-violence analysis", "",
             f"{ev['violent']} violent clips (cut at the human start time) and {ev['normal']} Normal control clips. Every number comes with its count; the set is small.", "",
             "## Was an interacting pair found before the violence?", "", "| category | clips | with a pair | share |", "|---|---|---|---|"]
    for c, v in ev["coverage"].items():
        lines.append(f"| {c} | {v['clips']} | {v['with_pair']} | {v['share']:.0%} |")
    lines += ["", "## Inside the same clip: last 3 s before the violence vs earlier", "",
              f"Clips long enough to compare: {t['clips']}. Concern higher in the last 3 s: {t['higher_in_last']} (equal: {t['equal']}); median difference {t['median_diff']:.2f}; sign-test p = {t['p_sign']:.3f}.",
              "This comparison cannot be explained by filming style (same clip).", "", "## Pre-violence vs Normal", "",
              f"Clips with a measurable pair: {n['violent_n']} violent, {n['normal_n']} Normal. Median concern {n['median_violent']:.2f} vs {n['median_normal']:.2f} (Normal: its highest window, i.e. against our claim).",
              f"AUC of the concern score: {n['auc']:.2f} (95% bootstrap interval {n['auc_low']:.2f}-{n['auc_high']:.2f}). Best single filming-style feature (resolution, fps, camera motion, brightness, sharpness) alone: AUC {n['style_only_auc']:.2f}.",
              f"Same raw resolution only: {n['same_resolution']['violent']} violent vs {n['same_resolution']['normal']} Normal clips, AUC {n['same_resolution']['auc']:.2f}.", "",
              f"**{n['verdict']}**", "", "## Lead-up types assigned (heuristic)", ""]
    for c, d in ev["leadup_types"].items():
        lines.append(f"- {c}: " + ", ".join(f"{k} {v}" for k, v in sorted(d.items(), key=lambda kv: -kv[1])))
    lines += ["", "Normal is a different source (earlier work: filming style alone separated it from the other categories with AUC 0.99), which is why the style baseline and the same-resolution check are shown."]
    return "\n".join(lines) + "\n"
