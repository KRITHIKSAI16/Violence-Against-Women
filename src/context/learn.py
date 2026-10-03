"""Deep context, Stage D: what in the behavior features actually separates buildup clips from Normal? (report M4-M7 done honestly)

Weakly supervised: the only label is the clip's folder (Normal = 0, any other = 1). A clip's score is the mean of its top-k window scores
(multiple-instance learning, as in Sultani et al.). Everything is cross-validated with GroupKFold-style splits BY CLIP (windows of one clip
never sit in both train and test), every clip counts equally (sample weight 1 / windows in the clip), and four question-answering models are
compared so shortcuts show up:

  style_only      how the clip was filmed (camera moving, brightness, resolution ...). If this separates the categories, the dataset has a shortcut.
  behavior        pair behavior features only (the thing we care about)
  behavior+scene  plus how crowded / isolated the scene is
  all             everything including style
  rule_gate       the M9 hand-written rule gate (flag yes/no), for reference

It also reports the same metrics on STATIC-camera clips only (removes the biggest style difference), per category against Normal, and
permutation importance of the behavior model. Clips with no usable pair get score 0 (predicted Normal) in the main numbers and are
excluded in a second set of numbers, so coverage and discrimination are not mixed up.

Usage:  python -m src.context.learn [--config CFG] [--rebuild] [--no-importance] [--set key=value ...]
"""
import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.assm.gate import load_gate
from src.config import load_config, parse_overrides, resolve_path
from src.context.features import load_context
from src.context.windows import BEHAVIOR, SCENE, STYLE, pair_windows, scene_features, style_features
from src.report.buildup_video import cut_point

log = logging.getLogger(__name__)

FEATURE_SETS = {
    "style_only": STYLE,
    "behavior": BEHAVIOR,
    "behavior+scene": BEHAVIOR + SCENE,
    "all": BEHAVIOR + SCENE + STYLE,
}


# ------------------------------------------------------------------ table building
def clip_end_frame(clip, scene, gate_dir, rep):
    """Last frame usable for this clip: the video cut point (before the act) for non-Normal categories, else the whole clip."""
    gp = Path(gate_dir) / clip["category"] / f"{clip['clip_id']}_gate.json"
    g = load_gate(gp) if gp.exists() else {"clip_id": clip["clip_id"], "duration_s": scene["duration_s"], "fps": scene["fps"], "escalation": None}
    cut = cut_point(g, clip["category"], rep)
    return cut["cut_f"] if cut["trimmed"] else scene["n_frames"], cut


def build_tables(clean, raw_by_id, ctx_dir, gate_dir, ctx, rep, win_s, step_s, max_pairs=None, progress=False):
    """-> (windows DataFrame, clips DataFrame). One clip row per clip with context results (also those without pairs)."""
    wrows, crows = [], []
    for k, c in enumerate(clean["clips"], 1):
        if progress and k % 100 == 0:
            print(f"  windows: {k}/{len(clean['clips'])} clips, {len(wrows)} windows so far", flush=True)
        sp = Path(ctx_dir) / c["category"] / f"{c['clip_id']}_scene.json"
        if not sp.exists():
            continue
        scene, arrays = load_context(ctx_dir, c["category"], c["clip_id"])
        end, cut = clip_end_frame(c, scene, gate_dir, rep)
        style = style_features(scene, raw_by_id.get(c["clip_id"], {}))
        sc = scene_features(scene)
        n_w = 0
        items = list(arrays.items())
        if max_pairs and len(items) > max_pairs:      # crowded scenes: keep the pairs that stay closest (typically the interacting ones)
            items.sort(key=lambda kv: float(np.nanmedian(kv[1]["dist_m"])) if np.isfinite(kv[1]["dist_m"]).any() else np.inf)
            items = items[:int(max_pairs)]
        for (i, j), a in items:
            for r in pair_windows(a, scene, ctx, win_s, step_s, end):
                r.update(sc)
                r.update(style)
                r.update({"clip_id": c["clip_id"], "category": c["category"], "pair": f"{i}_{j}", "label": int(c["category"] != "Normal")})
                wrows.append(r)
                n_w += 1
        crows.append({"clip_id": c["clip_id"], "category": c["category"], "label": int(c["category"] != "Normal"), "n_windows": n_w,
                      "cut_s": cut["cut_s"], "cut_reason": cut["reason"], **style})
    return pd.DataFrame(wrows), pd.DataFrame(crows)


# ------------------------------------------------------------------ models
def make_model(kind, seed=7):
    if kind == "logreg":
        return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), LogisticRegression(C=0.3, max_iter=500, class_weight="balanced"))
    return HistGradientBoostingClassifier(max_depth=3, learning_rate=0.08, max_iter=120, l2_regularization=1.0, class_weight="balanced", random_state=seed)


def _fit(model, X, y, w):
    if hasattr(model, "steps"):
        model.fit(X, y, **{model.steps[-1][0] + "__sample_weight": w})
    else:
        model.fit(X, y, sample_weight=w)
    return model


def clip_weights(df):
    """Every clip counts equally: weight 1 / number of its windows, then rescaled so the mean weight is 1."""
    w = 1.0 / df.groupby("clip_id")["clip_id"].transform("size")
    return (w / w.mean()).to_numpy()


def cross_validate(df, cols, kind="gb", folds=5, seed=7, importance=False):
    """Out-of-fold window probabilities (grouped by clip). Returns (probs array, importance DataFrame or None)."""
    X = df[cols].to_numpy(float)
    y = df["label"].to_numpy()
    w = clip_weights(df)
    # stratify clips by category so every fold holds every category
    clip_cat = df.groupby("clip_id")["category"].first()
    skf = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=seed)
    oof = np.full(len(df), np.nan)
    imps = []
    for tr, te in skf.split(X, df["category"], groups=df["clip_id"]):
        m = _fit(make_model(kind, seed), X[tr], y[tr], w[tr])
        oof[te] = m.predict_proba(X[te])[:, 1]
        if importance and len(np.unique(y[te])) == 2:
            pi = permutation_importance(m, X[te], y[te], scoring="roc_auc", n_repeats=3, random_state=seed, sample_weight=w[te])
            imps.append(pi.importances_mean)
    imp = pd.DataFrame({"feature": cols, "importance": np.mean(imps, axis=0)}).sort_values("importance", ascending=False) if imps else None
    return oof, imp


def clip_scores(df, prob, clips, k=3):
    """Mean of the top-k window probabilities per clip; clips without windows score 0."""
    s = pd.Series(prob, index=df.index).groupby(df["clip_id"]).apply(lambda v: float(np.sort(v.to_numpy())[::-1][:k].mean()))
    out = clips.set_index("clip_id").copy()
    out["score"] = s.reindex(out.index).fillna(0.0)
    out["has_pair"] = out["n_windows"] > 0
    return out.reset_index()


def best_f1_threshold(y, s):
    cand = np.unique(np.quantile(s, np.linspace(0.02, 0.98, 49)))
    f = [f1_score(y, s >= t, zero_division=0) for t in cand]
    return float(cand[int(np.argmax(f))])


def metrics(y, s, thr=None):
    """AUC plus accuracy / precision / recall / F1 at a threshold (default: the F1-maximising one; optimistic, stated as such)."""
    y, s = np.asarray(y), np.asarray(s)
    if len(np.unique(y)) < 2:
        return {"n": len(y), "auc": np.nan}
    thr = best_f1_threshold(y, s) if thr is None else thr
    p = s >= thr
    return {"n": int(len(y)), "auc": float(roc_auc_score(y, s)), "acc": float(accuracy_score(y, p)),
            "precision": float(precision_score(y, p, zero_division=0)), "recall": float(recall_score(y, p, zero_division=0)),
            "f1": float(f1_score(y, p, zero_division=0)), "thr": float(thr)}


def per_category_auc(cs):
    """AUC of each non-Normal category against Normal."""
    out = {}
    normal = cs[cs["category"] == "Normal"]
    for cat in sorted(set(cs["category"]) - {"Normal"}):
        sub = pd.concat([cs[cs["category"] == cat], normal])
        out[cat] = float(roc_auc_score(sub["label"], sub["score"])) if sub["label"].nunique() == 2 else np.nan
    return out


def gate_scores(clips, gate_dir):
    """Flag of the M9 rule gate as a 0/1 'score' (the baseline to beat)."""
    out = []
    for _, r in clips.iterrows():
        p = Path(gate_dir) / r["category"] / f"{r['clip_id']}_gate.json"
        out.append(float(load_gate(p)["flag"]) if p.exists() else np.nan)
    return np.array(out)


# ------------------------------------------------------------------ report
def run(windows, clips, cfg_learn, gate_dir=None, importance=True, kinds=("gb", "logreg")):
    """All experiments. Returns a results dict (JSON-serialisable) and the clip-score tables."""
    k, folds, seed = int(cfg_learn["topk"]), int(cfg_learn["folds"]), int(cfg_learn["seed"])
    res = {"n_clips": int(len(clips)), "n_clips_with_pair": int((clips["n_windows"] > 0).sum()), "n_windows": int(len(windows)),
           "label_counts": clips["label"].value_counts().to_dict(), "models": {}, "importance": {}}
    tables = {}
    static = set(clips.loc[clips["cam_moving"] == 0, "clip_id"])
    for kind in kinds:
        for name, cols in FEATURE_SETS.items():
            oof, imp = cross_validate(windows, cols, kind, folds, seed, importance=(importance and name == "behavior" and kind == "gb"))
            print(f"  done: {kind}:{name}", flush=True)
            cs = clip_scores(windows, oof, clips, k)
            tables[f"{kind}:{name}"] = cs
            has = cs[cs["has_pair"]]
            st = cs[cs["clip_id"].isin(static)]
            res["models"][f"{kind}:{name}"] = {
                "all_clips": metrics(cs["label"], cs["score"]),
                "clips_with_pair": metrics(has["label"], has["score"]),
                "static_camera_clips": metrics(st["label"], st["score"]),
                "per_category_auc_vs_normal": per_category_auc(cs),
            }
            if imp is not None:
                res["importance"][kind] = imp.head(12).to_dict("records")
    if gate_dir is not None:
        g = gate_scores(clips, gate_dir)
        ok = ~np.isnan(g)
        if ok.sum() > 5:
            res["rule_gate"] = {"all_clips": metrics(clips["label"][ok], g[ok], 0.5)}
    return res, tables


def print_results(res):
    print(f"\n{res['n_clips']} clips ({res['n_clips_with_pair']} with at least one usable pair), {res['n_windows']} windows. Labels: {res['label_counts']}\n")
    print("Clip-level discrimination: buildup categories (1) vs Normal (0), cross-validated by clip. AUC 0.5 = chance.")
    print(f"{'model':<22}{'AUC all':>9}{'AUC pairs':>11}{'AUC static':>12}{'F1':>7}{'prec':>7}{'recall':>8}")
    for name, m in res["models"].items():
        a, p, s = m["all_clips"], m["clips_with_pair"], m["static_camera_clips"]
        print(f"{name:<22}{a['auc']:>9.3f}{p['auc']:>11.3f}{s['auc']:>12.3f}{a.get('f1', np.nan):>7.2f}{a.get('precision', np.nan):>7.2f}{a.get('recall', np.nan):>8.2f}")
    if "rule_gate" in res:
        g = res["rule_gate"]["all_clips"]
        print(f"{'M9 rule gate (flag)':<22}{g['auc']:>9.3f}{'':>11}{'':>12}{g['f1']:>7.2f}{g['precision']:>7.2f}{g['recall']:>8.2f}")
    print("\nPer-category AUC against Normal (behavior model, gradient boosting):")
    for cat, v in res["models"].get("gb:behavior", {}).get("per_category_auc_vs_normal", {}).items():
        print(f"  {cat:<16}{v:.3f}")
    for kind, imp in res["importance"].items():
        print(f"\nPermutation importance, behavior model ({kind}); drop in window AUC when the feature is scrambled:")
        for r in imp[:10]:
            print(f"  {r['feature']:<18}{r['importance']:+.4f}")
    sty, beh = res["models"].get("gb:style_only", {}), res["models"].get("gb:behavior", {})
    if sty and beh:
        s_auc, b_auc = sty["all_clips"]["auc"], beh["all_clips"]["auc"]
        print("\nHow to read this:")
        print(f"  style_only AUC {s_auc:.3f} vs behavior AUC {b_auc:.3f}.", end=" ")
        if max(s_auc, b_auc) < 0.6:
            print("Neither how the clips were filmed nor the behavior features separate buildup clips from Normal in this data (both near chance).")
        elif s_auc >= b_auc - 0.02:
            print("The way the clips were filmed separates the classes about as well as the behavior does: treat behavior results as unproven.")
        else:
            print("Behavior carries signal beyond how the clips were filmed.")
        print("  AUC on static-camera clips is the fairer number. Labels are weak (the category, not a per-second annotation): an AUC well below 0.8 means the")
        print("  features cannot reproduce the category, not that the detectors are wrong; confirm with the annotated benchmark.")


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    ap = argparse.ArgumentParser(description="Stage D: window-level learning with shortcut probes")
    ap.add_argument("--config", default=None)
    ap.add_argument("--rebuild", action="store_true", help="rebuild the window table even if cached")
    ap.add_argument("--no-importance", action="store_true")
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE")
    args = ap.parse_args()
    cfg = load_config(args.config)
    cfg["learn"] = {**cfg["learn"], **parse_overrides(args.set)}
    L = cfg["learn"]
    out = resolve_path(L["learn_dir"])
    out.mkdir(parents=True, exist_ok=True)
    clean = json.loads(resolve_path(cfg["preprocess"]["clean_manifest_path"]).read_text("utf-8"))
    raw = json.loads(resolve_path(cfg["manifest_path"]).read_text("utf-8")) if resolve_path(cfg["manifest_path"]).exists() else {"clips": []}
    raw_by_id = {c["clip_id"]: c for c in raw["clips"]}
    wp, cp = out / "windows.csv", out / "clips.csv"
    if wp.exists() and cp.exists() and not args.rebuild:
        windows, clips = pd.read_csv(wp), pd.read_csv(cp)
        print(f"Loaded cached tables ({len(windows)} windows); use --rebuild to recompute.")
    else:
        windows, clips = build_tables(clean, raw_by_id, resolve_path(cfg["context"]["context_dir"]), resolve_path(cfg["gate"]["gate_dir"]),
                                      cfg["context"], cfg["report"], L["window_s"], L["step_s"], L.get("max_pairs_per_clip"), progress=True)
        windows.to_csv(wp, index=False)
        clips.to_csv(cp, index=False)
    if windows.empty or clips["label"].nunique() < 2:
        raise SystemExit("Not enough data: need both Normal and non-Normal clips with usable pairs. Run src.context.features first.")
    print(f"Window table ready ({len(windows)} windows). Running experiments (8 models, 5-fold) ...", flush=True)
    res, tables = run(windows, clips, L, resolve_path(cfg["gate"]["gate_dir"]), importance=not args.no_importance)
    (out / "results.json").write_text(json.dumps(res, indent=1, default=float), encoding="utf-8")
    tables["gb:behavior"].to_csv(out / "clip_scores_behavior.csv", index=False)
    print_results(res)
    print(f"\nSaved: {out / 'results.json'}, windows.csv, clips.csv, clip_scores_behavior.csv")


if __name__ == "__main__":
    main()
