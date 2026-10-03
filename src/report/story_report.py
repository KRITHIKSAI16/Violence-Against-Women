"""Phase 1 deliverable page: the buildup stories of every category, with the honest evaluation next to them.

Reads the clip stories (src.context.story), the evaluation results (src.context.learn -> results.json, optional) and the annotation benchmark
(src.report.benchmark -> benchmark_results.json, optional); renders story videos + storyboards for a showcase (top-evidence AND random clips per
category, so it is not cherry-picked) and writes into the report directory:

  index.html            findings, category table, showcase cards (scene, narrative, storyboard, video)
  categories_story.csv  one row per category          clips_story.csv  one row per clip (scene facts, cues, cut point, files)

Usage:  python -m src.report.story_report [--config CFG] [--no-videos] [--all] [--workers N] [--set showcase_top=8 ...]
"""
import argparse
import csv
import html
import json
import logging
import random
from collections import defaultdict
from pathlib import Path

import numpy as np

from src.config import load_config, parse_overrides, resolve_path
from src.context.graph import ACT_PREDS, load_story
from src.context.narrative import narrate
from src.report.buildup_video import load_overrides
from src.report.story_video import render_clip, story_cut

log = logging.getLogger(__name__)
KEY_PREDS = ["approaches_from_behind", "follows", "looks_back", "hovers_near", "blocks_exit", "pinned_against", "reaches_for", "contact", "mutual_facing"]


def _med(v):
    v = [x for x in v if x is not None]
    return float(np.median(v)) if v else float("nan")


def category_rows(stories, scenes):
    by = defaultdict(list)
    for s in stories:
        by[s["category"]].append(s)
    rows = []
    for cat, lst in sorted(by.items()):
        n = len(lst)
        pr = [s for s in lst if not s["no_pair"]]
        cam = [scenes[s["clip_id"]]["camera"]["moving"] for s in lst if s["clip_id"] in scenes]
        row = {"category": cat, "clips": n, "no_pair": sum(s["no_pair"] for s in lst) / n,
               "any_concern_cue": sum(s["concern_seconds"] > 0 for s in lst) / n,
               "median_concern_s": _med([s["concern_seconds"] for s in pr]), "median_benign_s": _med([s["benign_seconds"] for s in pr]),
               "ordered": sum(s["pairs"][s["key_pair"]]["ordered_progression"] for s in pr) / n if pr else 0.0,
               "act_cue": sum(s["escalation"] is not None for s in lst) / n,
               "camera_moving": float(np.mean(cam)) if cam else float("nan"),
               "layout_reliable": sum(s["scene"]["layout_reliable"] for s in lst) / n}
        for p in KEY_PREDS:
            row[p] = sum(p in s["preds_seen"] for s in lst) / n
        rows.append(row)
    return rows


def select_showcase(stories, top, rand, seed):
    """{category: [(clip_id, 'top'|'random')]}. 'top' = most cue evidence; zero-evidence clips only enter as random."""
    rng = random.Random(seed)
    by = defaultdict(list)
    for s in stories:
        by[s["category"]].append(s)
    out = {}
    for cat, lst in sorted(by.items()):
        lst = sorted(lst, key=lambda s: s["clip_id"])
        ranked = sorted([s for s in lst if not s["no_pair"] and s["concern_seconds"] > 0], key=lambda s: (-s["pairs"][s["key_pair"]]["cue_sum"], s["clip_id"]))
        top_ids = [s["clip_id"] for s in ranked[:top]]
        rest = [s["clip_id"] for s in lst if s["clip_id"] not in top_ids]
        out[cat] = [(c, "top") for c in top_ids] + [(c, "random") for c in rng.sample(rest, min(rand, len(rest)))]
    return out


def clip_rows(stories, scenes, cuts, rendered):
    rows = []
    for s in stories:
        cut = cuts.get(s["clip_id"], {})
        r = rendered.get(s["clip_id"], {})
        sc = s["scene"]
        rows.append({"clip_id": s["clip_id"], "category": s["category"], "no_pair": int(s["no_pair"]), "place_type": sc["place_type"],
                     "layout_reliable": int(sc["layout_reliable"]), "camera_moving": int(sc["camera_moving"]), "low_light": int(sc["low_light"]),
                     "max_people": sc["max_people"], "isolated_frac": sc["isolated_frac"], "key_pair": s["key_pair"] or "",
                     "preds_seen": " ".join(s["preds_seen"]), "concern_seconds": s["concern_seconds"], "benign_seconds": s["benign_seconds"],
                     "phases": " > ".join(s["pairs"][s["key_pair"]]["phases"]) if s["key_pair"] else "",
                     "ordered": int(bool(s["key_pair"] and s["pairs"][s["key_pair"]]["ordered_progression"])),
                     "act_cue_s": s["escalation"]["time_s"] if s["escalation"] else "", "act_cue_kind": s["escalation"]["kind"] if s["escalation"] else "",
                     "cut_s": cut.get("cut_s", ""), "cut_reason": cut.get("reason", ""), "video": r.get("video", ""), "storyboard": r.get("storyboard", "")})
    return rows


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


# ---------------------------------------------------------------- findings text (from the evaluation files)
def findings_html(learn, bench):
    esc = html.escape
    out = []
    if learn:
        m = learn["models"]
        sty, beh = m["gb:style_only"], m["gb:behavior"]
        gate = learn.get("rule_gate", {}).get("all_clips", {}).get("auc")
        ov = learn.get("overlap") or {}
        out.append("<li><b>The label 'buildup category' is not predicted by behavior.</b> Cross-validated by clip, behavior features separate the buildup categories from Normal with AUC "
                   f"{beh['clips_with_pair']['auc']:.2f} among clips that have a usable pair and {beh['static_camera_clips']['auc']:.2f} on static-camera clips (0.5 = chance). "
                   f"The earlier rule gate scored {gate:.2f} (below chance: it flagged Normal clips more often).</li>" if gate is not None else
                   "<li><b>The label 'buildup category' is not predicted by behavior.</b> Behavior features separate the buildup categories from Normal with AUC "
                   f"{beh['clips_with_pair']['auc']:.2f} (clips with a pair) and {beh['static_camera_clips']['auc']:.2f} (static camera); 0.5 = chance.</li>")
        out.append("<li><b>Normal footage comes from a different source.</b> How a clip was filmed (original resolution, frame rate, camera motion, brightness) identifies Normal vs the rest with AUC "
                   f"{sty['clips_with_pair']['auc']:.2f}. Any classifier on this label can win by recognising the source, which is why behavior is judged separately.</li>")
        if ov:
            out.append("<li><b>Style-matched comparison:</b> " + (f"on {ov['n_pos']} buildup vs {ov['n_neg']} Normal clips that look alike in style, behavior AUC is {ov['behavior_auc']:.2f}." if ov.get("enough")
                       else f"too few clips look alike in style ({ov['n_pos']} buildup, {ov['n_neg']} Normal) to compare behavior fairly.") + "</li>")
    else:
        out.append("<li>Evaluation results not found yet (run notebook 06).</li>")
    if bench:
        good = [(b, v) for b, v in bench["behaviors"].items() if v["annotated_present"] >= 3 and v["precision"] == v["precision"]]
        def row(b, v):
            kappa = "-" if v["kappa"] != v["kappa"] else format(v["kappa"], ".2f")
            return (f"<tr><td>{esc(b)}</td><td>{v['annotated_present']}</td><td>{v['detected']}</td><td>{v['precision']:.2f}</td>"
                    f"<td>{v['recall']:.2f}</td><td>{kappa}</td></tr>")
        rows = "".join(row(b, v) for b, v in good)
        out.append(f"<li><b>Annotated benchmark ({bench['n_clips']} clips, {len(bench['annotators'])} annotators):</b> detector precision / recall against human labels.<table><tr><th>behavior</th><th>annotated</th><th>detected</th>"
                   f"<th>precision</th><th>recall</th><th>kappa</th></tr>{rows}</table></li>")
    else:
        out.append("<li>The annotated benchmark has not been run yet: until it is, detector precision and recall are only spot-checked by eye.</li>")
    return "<ul>" + "".join(out) + "</ul>"


CSS = """body{font-family:system-ui,Segoe UI,Arial,sans-serif;margin:0;padding:24px;background:#f6f6f4;color:#222;max-width:1500px}
h1{margin:0 0 4px}h2{margin-top:34px;border-bottom:2px solid #ddd;padding-bottom:4px}table{border-collapse:collapse;background:#fff;font-size:13px;margin:6px 0}
th,td{border:1px solid #ddd;padding:5px 9px;text-align:right}th:first-child,td:first-child{text-align:left}
.note{background:#fff8e1;border:1px solid #f0d98a;padding:12px 18px;max-width:1000px;line-height:1.5}.ok{background:#eef7ee;border-color:#b9dcb9}
.grid{display:flex;flex-wrap:wrap;gap:16px}.card{background:#fff;border:1px solid #ddd;padding:10px;width:470px;font-size:13px}
.card img{width:100%}.tag{font-size:11px;padding:1px 6px;border-radius:8px;background:#e8e8e8;margin-right:3px}.top{background:#ffe0b2}
.scene{color:#555;margin:4px 0}.lines{margin:4px 0 4px 16px;padding:0}.lines li{margin:2px 0}.act{color:#a00}.sum{color:#035;margin:4px 0}small{color:#666}video{width:100%;margin-top:6px}"""


def build_html(stories, cats, showcase, rendered, cuts, learn, bench, out_dir, n_total):
    esc = html.escape
    by_id = {s["clip_id"]: s for s in stories}
    p = [f"<!doctype html><html><head><meta charset='utf-8'><title>Pre-violence buildup: the scene story</title><style>{CSS}</style></head><body>",
         "<h1>Pre-violence buildup: what happens before the act</h1>",
         f"<small>{n_total} clips. For each clip the system describes who approaches, follows, looks back at, lingers near, blocks or reaches for whom, with times, approximate meters "
         "and scene facts, using only geometry, body pose and the scene layout. Footage stops before the first physical-act cue. No prediction, no language model.</small>",
         "<h2>What this shows, and what it does not</h2><div class='note ok'><b>It shows</b> a detailed, checkable account of the interaction between two people in each clip: "
         "the order of approach, following, looking back, lingering, blocking and reaching, the target's reaction, and the scene (indoor / outdoor, night, camera, crowd). "
         "<br><b>It does not</b> classify a clip as violent, predict anything, or claim a cue is a threat: many cues (walking up behind someone, standing close) are also ordinary behavior, "
         "which is why conversation (mutual facing) is shown as a benign cue and why every category is compared with Normal.</div>",
         "<h2>Key findings of the evaluation</h2><div class='note'>" + findings_html(learn, bench) + "</div>"]
    p.append("<h2>Categories</h2><table><tr><th>category</th><th>clips</th><th>no pair</th><th>any concern cue</th><th>median concern s</th><th>median benign s</th>"
             "<th>ordered</th><th>act cue</th><th>camera moving</th><th>layout reliable</th>" + "".join(f"<th>{esc(k.replace('_', ' '))}</th>" for k in KEY_PREDS) + "</tr>")
    pct = lambda x: "-" if x != x else f"{x:.0%}"
    num = lambda x: "-" if x != x else f"{x:.1f}"
    for r in cats:
        p.append(f"<tr><td>{esc(r['category'])}</td><td>{r['clips']}</td><td>{pct(r['no_pair'])}</td><td>{pct(r['any_concern_cue'])}</td><td>{num(r['median_concern_s'])}</td>"
                 f"<td>{num(r['median_benign_s'])}</td><td>{pct(r['ordered'])}</td><td>{pct(r['act_cue'])}</td><td>{pct(r['camera_moving'])}</td><td>{pct(r['layout_reliable'])}</td>"
                 + "".join(f"<td>{pct(r[k])}</td>" for k in KEY_PREDS) + "</tr>")
    p.append("</table><p><small>Shares are of all clips in the category. 'no pair': fewer than two people tracked together long enough. 'act cue': a reach, contact or fast flee was detected. "
             "Compare every row with Normal (the false-alarm baseline).</small></p>")
    for cat, items in showcase.items():
        p.append(f"<h2>{esc(cat)}{' (baseline)' if cat == 'Normal' else ''}</h2><div class='grid'>")
        for cid, why in items:
            s, r, cut = by_id[cid], rendered.get(cid, {}), cuts.get(cid, {})
            n = narrate(s)
            p.append("<div class='card'>")
            if r.get("storyboard"):
                p.append(f"<a href='{esc(r['storyboard'])}'><img loading='lazy' src='{esc(r['storyboard'])}'></a>")
            p.append(f"<b>{esc(cid)}</b> <span class='tag {'top' if why == 'top' else ''}'>{why}</span>"
                     f"{'<span class=tag>no pair</span>' if s['no_pair'] else ''}<div class='scene'>{esc(n['scene'])}</div>")
            if n["lines"]:
                p.append("<ul class='lines'>" + "".join(f"<li class='{'act' if l['act'] else ''}'>{l['t0']:.1f}-{l['t1']:.1f} s: {esc(l['text'])}</li>" for l in n["lines"][:8])
                         + ("<li>...</li>" if len(n["lines"]) > 8 else "") + "</ul>")
            p.append(f"<div class='sum'>{esc(n['summary'])}</div>")
            if cut:
                p.append(f"<small>shows {cut['cut_s']:.1f} s ({esc(cut['reason'])})</small>")
            p.append(f"<video controls preload='none' src='{esc(r['video'])}'></video>" if r.get("video") else "<small>no pre-violence footage to show</small>")
            p.append("</div>")
        p.append("</div>")
    p.append("</body></html>")
    (out_dir / "index.html").write_text("\n".join(p), encoding="utf-8")


def _render_job(args):
    clip, cfg, overrides, out_dir = args
    try:
        r = render_clip(clip, cfg, overrides, out_dir)
        rel = lambda x: str(Path(x).relative_to(out_dir)).replace("\\", "/") if x else ""
        return clip["clip_id"], {"video": rel(r["video"]), "storyboard": rel(r["storyboard"]), "cut": r["cut"]}
    except Exception as e:  # one bad clip must not stop the report
        log.warning("Could not render %s: %s", clip["clip_id"], e)
        return clip["clip_id"], {}


def _load_json(p):
    return json.loads(Path(p).read_text(encoding="utf-8")) if Path(p).exists() else None


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    ap = argparse.ArgumentParser(description="Build the Phase 1 story report")
    ap.add_argument("--config", default=None)
    ap.add_argument("--no-videos", action="store_true")
    ap.add_argument("--all", action="store_true", help="render every clip with a pair (slow)")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE")
    args = ap.parse_args()
    cfg = load_config(args.config)
    cfg["report"] = {**cfg["report"], **parse_overrides(args.set)}
    rep = cfg["report"]
    out_dir = resolve_path(rep["report_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    ctx_dir = resolve_path(cfg["context"]["context_dir"])
    clean = json.loads(resolve_path(cfg["preprocess"]["clean_manifest_path"]).read_text("utf-8"))
    clips = {c["clip_id"]: c for c in clean["clips"]}
    stories, scenes = [], {}
    for c in clean["clips"]:
        st = load_story(ctx_dir, c["category"], c["clip_id"])
        if st:
            stories.append(st)
            sc = _load_json(ctx_dir / c["category"] / f"{c['clip_id']}_scene.json")
            if sc:
                scenes[c["clip_id"]] = sc
    if not stories:
        raise SystemExit("No stories found. Run: python -m src.context.story")
    learn = _load_json(resolve_path(cfg["learn"]["learn_dir"]) / "results.json")
    bench = _load_json(out_dir / "benchmark" / "benchmark_results.json")
    overrides = load_overrides(rep["overrides_csv"])
    showcase = select_showcase(stories, int(rep["showcase_top"]), int(rep["showcase_random"]), int(rep["seed"]))
    want = {cid for items in showcase.values() for cid, _ in items}
    if args.all:
        want = {s["clip_id"] for s in stories if not s["no_pair"]}
    rendered, cuts = {}, {}
    if not args.no_videos:
        jobs = [(clips[cid], cfg, overrides, out_dir) for cid in sorted(want)]
        print(f"Rendering {len(jobs)} clips (story video + storyboard) ...", flush=True)
        if args.workers > 1:
            from concurrent.futures import ProcessPoolExecutor
            with ProcessPoolExecutor(args.workers) as ex:
                results = dict(ex.map(_render_job, jobs))
        else:
            results = dict(_render_job(j) for j in jobs)
        rendered = {k: v for k, v in results.items() if v}
        cuts = {k: v["cut"] for k, v in rendered.items()}
    for s in stories:     # cut points for every clip (cheap) so clips.csv is complete
        if s["clip_id"] not in cuts:
            cuts[s["clip_id"]] = story_cut(s, s["category"], None, rep, overrides)
    cats = category_rows(stories, scenes)
    write_csv(out_dir / "categories_story.csv", cats)
    write_csv(out_dir / "clips_story.csv", clip_rows(stories, scenes, cuts, rendered))
    build_html(stories, cats, showcase, rendered, cuts, learn, bench, out_dir, len(stories))
    print(f"\nReport written to {out_dir}: index.html, categories_story.csv, clips_story.csv")
    print(f"{'category':<16}{'clips':>6}{'no pair':>9}{'concern':>9}{'act cue':>9}{'cam mov':>9}")
    for r in cats:
        print(f"{r['category']:<16}{r['clips']:>6}{r['no_pair']:>9.0%}{r['any_concern_cue']:>9.0%}{r['act_cue']:>9.0%}{r['camera_moving']:>9.0%}")


if __name__ == "__main__":
    main()
