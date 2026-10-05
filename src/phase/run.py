"""Layer 2, step 6: train the phase model on automatic labels, test it on human-labeled held-out clips, compare with the rule gate.

  train clips  non-held-out violent clips with an agreed VLM pseudo-label (`label.vlm_labels`)
  test clips   held-out clips with a human label (`label.human_labels`); never trained on, never used to choose settings
  reports      (1) cross-validated agreement with the PSEUDO labels (a self-consistency check, NOT accuracy)
               (2) the real score: decoded timeline vs HUMAN labels on the test clips, for the full model, feature ablations,
                   and two baselines (the rule gate's escalation time, and "the act starts at 60% of the clip")
Writes data/phase/pred/<Category>/<clip>_phase.json (per-window probabilities, decoded timeline) for the report and video.

    python -m src.phase.run [--config CFG] [--rebuild] [--ablate]
"""
import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from src.assm.gate import load_gate
from src.config import load_config, resolve_path
from src.phase.dataset import CHANGE, build_window_table, feature_columns
from src.phase.evaluate import evaluate, print_scores
from src.phase.labels import merge, read_labels
from src.phase.model import add_targets, decode_table, fit, grouped_cv, predict_probs

log = logging.getLogger(__name__)


def holdout_ids(cfg):
    p = resolve_path(cfg["perception"]["holdout_file"])
    return {l.strip() for l in p.read_text("utf-8").splitlines() if l.strip() and not l.startswith("#")}


def baseline_predictions(clips, gate_dir, durations, frac=0.6):
    """Two baselines per clip: the rule gate's escalation time (None when it found none), and a fixed fraction of the clip."""
    rule, fixed = {}, {}
    for c in clips:
        gp = Path(gate_dir) / c["category"] / f"{c['clip_id']}_gate.json"
        esc = load_gate(gp).get("escalation") if gp.exists() else None
        rule[c["clip_id"]] = {"act_start_s": esc["time_s"] if esc else None, "buildup_start_s": None}
        fixed[c["clip_id"]] = {"act_start_s": round(frac * durations[c["clip_id"]], 2), "buildup_start_s": None}
    return rule, fixed


def save_predictions(out_dir, clips, df, probs, pred):
    cat = {c["clip_id"]: c["category"] for c in clips}
    for cid, idx in df.groupby("clip_id", sort=False).indices.items():
        g = df.iloc[idx]
        d = Path(out_dir) / cat[cid]
        d.mkdir(parents=True, exist_ok=True)
        rec = {"clip_id": cid, "t_center": g["t_center"].round(3).tolist(), "probs": np.round(probs[idx], 4).tolist(),
               "shot": g["shot"].tolist(), "pair": g["pair"].tolist(), **pred[cid]}
        (d / f"{cid}_phase.json").write_text(json.dumps(rec), encoding="utf-8")


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    ap = argparse.ArgumentParser(description="Layer 2: train / evaluate the phase model")
    ap.add_argument("--config", default=None)
    ap.add_argument("--rebuild", action="store_true", help="rebuild the window table")
    ap.add_argument("--ablate", action="store_true", help="also score feature-group ablations")
    args = ap.parse_args()
    cfg = load_config(args.config)
    pc, lc = cfg["phase"], cfg["label"]
    clean = json.loads(resolve_path(cfg["preprocess"]["clean_manifest_path"]).read_text("utf-8"))["clips"]
    out = resolve_path(pc["dir"])
    out.mkdir(parents=True, exist_ok=True)
    table_path = out / "windows.pkl"
    if table_path.exists() and not args.rebuild:
        df = pd.read_pickle(table_path)
    else:
        df = build_window_table(clean, resolve_path(cfg["context"]["context_dir"]), cfg["context"], float(pc["win_s"]), float(pc["step_s"]), progress=True)
        df.to_pickle(table_path)
    log.info("window table: %d windows, %d clips", len(df), df["clip_id"].nunique() if len(df) else 0)
    hold = holdout_ids(cfg)
    vlm, human = read_labels(resolve_path(lc["vlm_labels"])), read_labels(resolve_path(lc["human_labels"]))
    labels = merge(vlm, human)
    train_ids = {c for c, l in vlm.items() if c not in hold and l["source"] == "vlm"}
    test_ids = {c for c, l in human.items() if c in hold and l["source"] == "human"}
    extra = tuple(pc.get("extra_prefixes", []))
    cols = feature_columns(df, extra)
    dur = {c["clip_id"]: c["duration_s"] for c in clean}
    log.info("train clips (agreed VLM labels, not held out): %d | test clips (human labels, held out): %d", len(train_ids), len(test_ids))
    if not train_ids or not test_ids:
        print("Need both label files: run src.label.vlm_propose (train labels) and label the validation pack (human labels), then rerun.")
        return
    tr = add_targets(df[df["clip_id"].isin(train_ids)], labels)
    te = df[df["clip_id"].isin(test_ids)]
    if not len(tr) or not len(te):
        print("No windows for the train or the test clips (clips without a usable pair have no windows).")
        return
    sw, mb = float(pc["switch_penalty"]), float(pc["min_build_s"])
    print(f"\n(1) Cross-validated agreement with the PSEUDO labels ({tr['clip_id'].nunique()} clips) - consistency check, not accuracy:")
    cv_pred = decode_table(tr, grouped_cv(tr, cols, int(pc["folds"])), sw, mb)
    print_scores(evaluate(cv_pred, {c: labels[c] for c in cv_pred}, dur), "  ")
    print(f"\n(2) Held-out test against HUMAN labels ({len(test_ids)} clips; clips without a usable pair get no prediction and count as missed):")
    truth = {c: human[c] for c in test_ids}
    clf = fit(tr, cols)
    probs = predict_probs(clf, te, cols)
    pred = decode_table(te, probs, sw, mb)
    for c in test_ids:
        pred.setdefault(c, {"act_start_s": None, "buildup_start_s": None, "has_buildup": False})
    print_scores(evaluate(pred, truth, dur), "  phase model  : ")
    test_clips = [c for c in clean if c["clip_id"] in test_ids]
    rule, fixed = baseline_predictions(test_clips, resolve_path(cfg["gate"]["gate_dir"]), dur)
    print_scores(evaluate(rule, truth, dur), "  rule gate    : ")
    print_scores(evaluate(fixed, truth, dur), "  fixed 60%    : ")
    if args.ablate:
        groups = {"without change features": [c for c in cols if c not in CHANGE],
                  "change features only": [c for c in cols if c in CHANGE] + [c for c in cols if c.startswith("scene_")]}
        for name, cc in groups.items():
            m = fit(tr, cc)
            print_scores(evaluate(decode_table(te, predict_probs(m, te, cc), sw, mb), truth, dur), f"  {name:<22}: ")
    save_predictions(out / "pred", clean, te, probs, pred)
    print("\nSmall test set: read these as counts, not percentages with error bars. The VLM labels' own accuracy is measured by `python -m src.label.vlm_propose --score`.")


if __name__ == "__main__":
    main()
