"""Annotated benchmark: measure the detectors against what people actually see (the only honest precision / recall we can get).

The category labels (Stalking, Kidnapping ...) say what a clip is about, not what happens second by second, so they cannot tell us whether
"approaches from behind" or "follows" are detected correctly. About 60 clips (10 per category, Normal included) are annotated by hand:

  make the sheet     python -m src.report.benchmark make     -> benchmark_sheet.csv  (one copy per annotator; fill the columns, 1 = yes, 0 = no)
  evaluate           python -m src.report.benchmark evaluate -> agreement between annotators (Cohen's kappa) and, per behavior, precision / recall / F1
                                                             of the detectors against the majority of annotators; accuracy of the video cut point

Annotators watch the clip's PROCESSED video, note every behavior that happens BEFORE any physical violence, and write act_start_s = the second
when the physical act (grab, hit, snatch, shot) starts, or leave it empty if there is none. Detector output is compared only up to that second.
"""
import argparse
import csv
import json
import logging
import random
from collections import defaultdict
from pathlib import Path

import numpy as np

from src.config import load_config, resolve_path
from src.context.graph import load_story

log = logging.getLogger(__name__)

BEHAVIORS = {
    "approach_from_behind": ("approaches_from_behind",),
    "following": ("follows",),
    "looking_back": ("looks_back",),
    "lingering_near": ("hovers_near", "very_close"),
    "blocking_or_cornering": ("blocks_exit", "pinned_against"),
    "reaching_or_contact": ("reaches_for", "contact"),
    "fleeing": ("flees_from",),
    "conversation_benign": ("mutual_facing",),
}
SHEET_FIELDS = (["clip_id", "category", "video", "annotator"] + list(BEHAVIORS) + ["buildup_visible_0_to_2", "act_start_s", "notes"])
INSTRUCTIONS = ("Fill one row per clip. For each behavior column write 1 if it happens at any time BEFORE the physical violence, 0 if not. "
                "buildup_visible_0_to_2: 0 = nothing leading up to the act, 1 = weak, 2 = clear. act_start_s: second when the physical act starts "
                "(empty if the clip has no violence). Do not look at the system's output while annotating.")


# ---------------------------------------------------------------- sheet
def select_clips(stories, clean, per_category, seed, share_with_pair=0.7):
    """Stratified sample: per category, about 70% clips with a usable pair and the rest random (so misses are visible too)."""
    rng = random.Random(seed)
    by = defaultdict(list)
    for c in clean["clips"]:
        by[c["category"]].append(c)
    chosen = []
    for cat, lst in sorted(by.items()):
        lst = sorted(lst, key=lambda c: c["clip_id"])
        with_pair = [c for c in lst if c["clip_id"] in stories and not stories[c["clip_id"]]["no_pair"]]
        rng.shuffle(with_pair)
        k1 = min(len(with_pair), round(per_category * share_with_pair))
        pick = with_pair[:k1]
        rest = [c for c in lst if c not in pick]
        rng.shuffle(rest)
        pick += rest[:per_category - len(pick)]
        chosen += sorted(pick, key=lambda c: c["clip_id"])
    return chosen


def make_sheet(clips, out_csv, annotators=("annotator_1",), video_rel=None):
    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    rows = [{"clip_id": c["clip_id"], "category": c["category"], "video": (video_rel or (lambda c: c["path"]))(c)} for c in clips]
    paths = []
    for a in annotators:
        p = out_csv.with_name(f"{out_csv.stem}_{a}.csv")
        with open(p, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=SHEET_FIELDS)
            w.writeheader()
            for r in rows:
                w.writerow({**r, "annotator": a})
        paths.append(p)
    (out_csv.parent / "ANNOTATION_INSTRUCTIONS.txt").write_text(INSTRUCTIONS + "\n", encoding="utf-8")
    return paths


# ---------------------------------------------------------------- evaluation
def cohen_kappa(a, b):
    """Cohen's kappa for two binary label lists (NaN when it is undefined)."""
    a, b = np.asarray(a, int), np.asarray(b, int)
    if len(a) == 0:
        return float("nan")
    po = float((a == b).mean())
    pe = float(a.mean() * b.mean() + (1 - a.mean()) * (1 - b.mean()))
    return float("nan") if pe >= 1.0 else (po - pe) / (1 - pe)


def read_sheet(path):
    out = {}
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["clip_id"]:
                out[r["clip_id"]] = r
    return out


def _flag(v):
    v = (v or "").strip()
    return None if v == "" else int(float(v) > 0)


def detector_hits(story, upto_s):
    """Behaviors the detectors found in the story's pairs before `upto_s` seconds (None = whole clip)."""
    found = set()
    if story is None or story["no_pair"]:
        return found
    for pair in story["pairs"].values():
        for e in pair["episodes"]:
            if upto_s is not None and e["start_s"] >= upto_s:
                continue
            for beh, preds in BEHAVIORS.items():
                if e["pred"] in preds:
                    found.add(beh)
    return found


def evaluate(sheets, stories):
    """sheets: {annotator: {clip_id: row}}; stories: {clip_id: story}. Returns a results dict."""
    ann = list(sheets)
    clips = sorted(set.intersection(*[set(s) for s in sheets.values()])) if ann else []
    res = {"annotators": ann, "n_clips": len(clips), "behaviors": {}, "cut": {}}
    majority, act = {}, {}
    for cid in clips:
        acts = [float(sheets[a][cid]["act_start_s"]) for a in ann if (sheets[a][cid].get("act_start_s") or "").strip() != ""]
        act[cid] = float(np.median(acts)) if acts else None
        for beh in BEHAVIORS:
            votes = [_flag(sheets[a][cid].get(beh)) for a in ann]
            votes = [v for v in votes if v is not None]
            # strict majority only: a tie means the behavior is ambiguous, so the clip is left out of precision / recall for it
            majority[(cid, beh)] = (None if not votes or sum(votes) * 2 == len(votes) else int(sum(votes) * 2 > len(votes)))
    for beh in BEHAVIORS:
        kap = []
        for x in range(len(ann)):
            for y in range(x + 1, len(ann)):
                pairs = [(_flag(sheets[ann[x]][c].get(beh)), _flag(sheets[ann[y]][c].get(beh))) for c in clips]
                pairs = [p for p in pairs if p[0] is not None and p[1] is not None]
                if pairs:
                    kap.append(cohen_kappa([p[0] for p in pairs], [p[1] for p in pairs]))
        tp = fp = fn = tn = ties = 0
        fp_examples, fn_examples = [], []
        for cid in clips:
            truth = majority[(cid, beh)]
            if truth is None:
                ties += 1
                continue
            pred = int(beh in detector_hits(stories.get(cid), act[cid]))
            tp += truth and pred
            fp += (not truth) and pred
            fn += truth and (not pred)
            tn += (not truth) and (not pred)
            if (not truth) and pred:
                fp_examples.append(cid)
            if truth and not pred:
                fn_examples.append(cid)
        prec = tp / (tp + fp) if tp + fp else float("nan")
        rec = tp / (tp + fn) if tp + fn else float("nan")
        res["behaviors"][beh] = {"ties_or_unannotated": int(ties), "annotated_present": int(tp + fn), "detected": int(tp + fp), "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
                                 "precision": prec, "recall": rec, "f1": (2 * prec * rec / (prec + rec)) if prec == prec and rec == rec and prec + rec else float("nan"),
                                 "kappa": float(np.nanmean(kap)) if kap else float("nan"), "false_positive_clips": fp_examples[:8], "missed_clips": fn_examples[:8]}
    errs, safe, n = [], 0, 0
    for cid in clips:
        st = stories.get(cid)
        if act[cid] is None or st is None or not st.get("escalation"):
            continue
        n += 1
        errs.append(st["escalation"]["time_s"] - act[cid])
        safe += st["escalation"]["time_s"] <= act[cid] + 0.3
    res["cut"] = {"n": n, "median_error_s": float(np.median(errs)) if errs else float("nan"),
                  "mean_abs_error_s": float(np.mean(np.abs(errs))) if errs else float("nan"), "detected_before_or_at_act_frac": safe / n if n else float("nan"),
                  "act_annotated": int(sum(a is not None for a in act.values())),
                  "act_detected_of_annotated": int(sum(1 for c in clips if act[c] is not None and stories.get(c) and stories[c].get("escalation")))}
    return res


def print_results(res):
    print(f"\nBenchmark: {res['n_clips']} clips, annotators: {', '.join(res['annotators'])}\n")
    print(f"{'behavior':<24}{'present':>8}{'found':>7}{'prec':>7}{'recall':>8}{'F1':>6}{'kappa':>7}")
    for beh, m in res["behaviors"].items():
        f = lambda v: "  -  " if v != v else f"{v:5.2f}"
        print(f"{beh:<24}{m['annotated_present']:>8}{m['detected']:>7}{f(m['precision']):>7}{f(m['recall']):>8}{f(m['f1']):>6}{f(m['kappa']):>7}")
    c = res["cut"]
    print(f"\nAct cue vs annotated act start: {c['act_detected_of_annotated']} of {c['act_annotated']} annotated acts were detected; for {c['n']} clips with both,"
          f" median error {c['median_error_s']:+.2f} s (negative = cut too early, safe), mean |error| {c['mean_abs_error_s']:.2f} s,"
          f" detected at or before the true start in {c['detected_before_or_at_act_frac']:.0%}.")
    print("kappa: 0.4-0.6 moderate, 0.6-0.8 substantial agreement between annotators; low kappa means the behavior itself is ambiguous.")


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    ap = argparse.ArgumentParser(description="Annotation benchmark: make sheets / evaluate detectors")
    ap.add_argument("cmd", choices=["make", "evaluate"])
    ap.add_argument("--config", default=None)
    ap.add_argument("--per-category", type=int, default=10)
    ap.add_argument("--annotators", nargs="*", default=["annotator_1", "annotator_2", "annotator_3", "annotator_4"])
    ap.add_argument("--sheets", nargs="*", default=None, help="filled sheets to evaluate (default: all benchmark_sheet_*.csv in the report dir)")
    args = ap.parse_args()
    cfg = load_config(args.config)
    ctx_dir = resolve_path(cfg["context"]["context_dir"])
    out = resolve_path(cfg["report"]["report_dir"]) / "benchmark"
    clean = json.loads(resolve_path(cfg["preprocess"]["clean_manifest_path"]).read_text("utf-8"))
    stories = {c["clip_id"]: load_story(ctx_dir, c["category"], c["clip_id"]) for c in clean["clips"]}
    stories = {k: v for k, v in stories.items() if v}
    if args.cmd == "make":
        clips = select_clips(stories, clean, args.per_category, int(cfg["report"]["seed"]))
        paths = make_sheet(clips, out / "benchmark_sheet.csv", args.annotators, lambda c: c["path"])
        print(f"{len(clips)} clips selected ({args.per_category} per category). Sheets written:")
        for p in paths:
            print("  ", p)
        print("Give one sheet to each team member (Drive: open with Google Sheets). Instructions:", out / "ANNOTATION_INSTRUCTIONS.txt")
    else:
        files = [Path(p) for p in args.sheets] if args.sheets else sorted(out.glob("benchmark_sheet_*.csv"))
        sheets = {}
        for p in files:
            rows = read_sheet(p)
            filled = [r for r in rows.values() if any((r.get(b) or "").strip() for b in BEHAVIORS)]
            if filled:
                sheets[p.stem.replace("benchmark_sheet_", "")] = {k: v for k, v in rows.items() if any((v.get(b) or "").strip() for b in BEHAVIORS)}
        if not sheets:
            raise SystemExit("No filled sheets found. Fill benchmark_sheet_<name>.csv first.")
        res = evaluate(sheets, stories)
        (out / "benchmark_results.json").write_text(json.dumps(res, indent=1, default=float), encoding="utf-8")
        print_results(res)
        print(f"\nSaved {out / 'benchmark_results.json'}")


if __name__ == "__main__":
    main()
