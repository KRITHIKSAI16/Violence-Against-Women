"""Phase 1 deliverable: the category report (index.html + CSVs) with buildup videos and a human review sheet.

Reads the M9 results (data/gate/<Category>/<clip>_gate.json), renders the buildup video + storyboard for a
showcase set, and writes into the report directory:

  index.html            category table, how to read it, limits, and per-category showcase cards
  categories.csv        one row per category (rates, medians), Normal = false-alarm baseline
  clips.csv             one row per clip (flag, behaviors, phase sequence, proposal window, cut point, files)
  review_sheet.csv      clips to check BY EYE (flagged = precision, unflagged = what was missed); fill `verdict`
  videos/ storyboards/  rendered files

The showcase is NOT cherry-picked: per category it takes the `showcase_top` clips with the most buildup evidence
AND `showcase_random` random clips (seeded), so weak or empty results are visible too.

Usage:  python -m src.report.buildup_report [--config CFG] [--no-videos] [--all] [--workers N]
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

from src.assm.gate import APPROACH, CORNER, ESCALATION, FOLLOW, HOVER, load_gate
from src.config import load_config, parse_overrides, resolve_path
from src.report.buildup_video import cut_point, key_segments, load_overrides, phrase, render_clip

log = logging.getLogger(__name__)


# ------------------------------------------------------------------ numbers
def evidence_score(g):
    """How much buildup evidence a clip has (used only to pick showcase clips)."""
    dur = {s: sum(x["dur_s"] for x in g["segments"] if x["state"] == s) for s in (APPROACH, FOLLOW, HOVER, CORNER)}
    return (dur[FOLLOW] + dur[HOVER] + dur[CORNER] + 0.5 * dur[APPROACH]
            + (3.0 if g["ordered_progression"] else 0.0) + (1.0 if g["escalation"] else 0.0))


def _median(v):
    return float(np.median(v)) if len(v) else float("nan")


def category_rows(gates):
    by = defaultdict(list)
    for g in gates:
        by[g["category"]].append(g)
    rows = []
    for cat, lst in sorted(by.items()):
        n = len(lst)

        def share(pred):
            return sum(1 for g in lst if pred(g)) / n
        flagged = [g for g in lst if g["flag"]]
        lens = [max(p["end_s"] - p["start_s"] for p in g["proposals"]) for g in flagged]
        t2e = [g["escalation"]["time_s"] - g["proposals"][0]["start_s"] for g in flagged
               if g["escalation"] and g["escalation"]["time_s"] >= g["proposals"][0]["start_s"]]
        rows.append({
            "category": cat, "clips": n, "no_pair": share(lambda g: g["no_interaction"]),
            "flagged": share(lambda g: g["flag"]),
            "FOLLOW": share(lambda g: FOLLOW in g["states_seen"]), "HOVER": share(lambda g: HOVER in g["states_seen"]),
            "CORNER": share(lambda g: CORNER in g["states_seen"]),
            "ESCALATION": share(lambda g: ESCALATION in g["states_seen"]),
            "ordered": share(lambda g: g["ordered_progression"]),
            "median_buildup_s": _median(lens), "median_time_to_escalation_s": _median(t2e),
            "median_clip_s": _median([g["duration_s"] for g in lst]),
        })
    return rows


def claim_text(g):
    """One line describing what the system claims happened (what a reviewer checks)."""
    segs = key_segments(g)
    if not segs:
        return "no sustained buildup behavior detected"
    parts = []
    for s in segs[:4]:
        other = s["id_j"] if s.get("actor") == s["id_i"] else s["id_i"]
        parts.append(f"{phrase(s, other)} {s['start_s']:.1f}-{s['end_s']:.1f}s")
    if g["escalation"]:
        parts.append(f"burst at {g['escalation']['time_s']:.1f}s")
    return "; ".join(parts)


# ------------------------------------------------------------------ selection
def select_showcase(gates, top, rand, seed):
    """{category: [(clip_id, 'top'|'random'), ...]} deterministic for a seed."""
    rng = random.Random(seed)
    by = defaultdict(list)
    for g in gates:
        by[g["category"]].append(g)
    out = {}
    for cat, lst in sorted(by.items()):
        lst = sorted(lst, key=lambda g: g["clip_id"])
        ranked = sorted(lst, key=lambda g: (-evidence_score(g), g["clip_id"]))
        top_ids = [g["clip_id"] for g in ranked if evidence_score(g) > 0][:top]
        rest = [g["clip_id"] for g in lst if g["clip_id"] not in top_ids]
        rnd = rng.sample(rest, min(rand, len(rest)))
        out[cat] = [(c, "top") for c in top_ids] + [(c, "random") for c in rnd]
    return out


def select_review(gates, n_flagged, n_unflagged, seed):
    """Clips for the human review sheet: flagged ones spread evenly over categories, plus unflagged non-Normal."""
    rng = random.Random(seed)
    flagged = defaultdict(list)
    for g in sorted(gates, key=lambda g: g["clip_id"]):
        if g["flag"]:
            flagged[g["category"]].append(g)
    for lst in flagged.values():
        rng.shuffle(lst)
    picks, cats = [], sorted(flagged)
    while len(picks) < n_flagged and any(flagged[c] for c in cats):
        for c in cats:
            if flagged[c] and len(picks) < n_flagged:
                picks.append((flagged[c].pop(), "flagged"))
    unflagged = [g for g in sorted(gates, key=lambda g: g["clip_id"])
                 if not g["flag"] and g["category"] != "Normal" and not g["no_interaction"]]
    rng.shuffle(unflagged)
    picks += [(g, "unflagged") for g in unflagged[:n_unflagged]]
    return picks


# ------------------------------------------------------------------ writing
def write_csv(path, rows, fields=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = fields or list(rows[0])
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def clip_rows(gates, rep, overrides, rendered):
    rows = []
    for g in gates:
        cut = cut_point(g, g["category"], rep, overrides)
        r = rendered.get(g["clip_id"], {})
        p = g["proposals"][0] if g["proposals"] else None
        rows.append({
            "clip_id": g["clip_id"], "category": g["category"], "flag": int(g["flag"]),
            "no_interaction": int(g["no_interaction"]), "behaviors": " ".join(g["states_seen"]),
            "phase_sequence": " > ".join(g["phase_sequence"]), "ordered_progression": int(g["ordered_progression"]),
            "first_proposal_start_s": p["start_s"] if p else "", "first_proposal_end_s": p["end_s"] if p else "",
            "escalation_s": g["escalation"]["time_s"] if g["escalation"] else "",
            "cut_s": cut["cut_s"], "cut_reason": cut["reason"], "isolated_frac": g["isolated_frac"],
            "claim": claim_text(g), "video": r.get("video", ""), "storyboard": r.get("storyboard", "")})
    return rows


def _pct(x):
    return "-" if x != x else f"{x:.0%}"


def _num(x):
    return "-" if x != x else f"{x:.1f}"


CSS = """body{font-family:system-ui,Segoe UI,Arial,sans-serif;margin:0;padding:24px;background:#f6f6f4;color:#222}
h1{margin:0 0 4px}h2{margin-top:36px;border-bottom:2px solid #ddd;padding-bottom:4px}
table{border-collapse:collapse;background:#fff;font-size:14px}th,td{border:1px solid #ddd;padding:6px 10px;text-align:right}
th:first-child,td:first-child{text-align:left}.note{background:#fff8e1;border:1px solid #f0d98a;padding:12px 16px;max-width:900px;
line-height:1.5}.grid{display:flex;flex-wrap:wrap;gap:16px}.card{background:#fff;border:1px solid #ddd;padding:8px;width:440px}
.card img{width:100%}.tag{font-size:12px;padding:1px 6px;border-radius:8px;background:#e8e8e8}.top{background:#ffe0b2}
.claim{font-size:13px;color:#444;margin:4px 0}video{width:100%;margin-top:6px}small{color:#666}"""


def build_html(gates, cats, showcase, rendered, rep, out_dir, notes):
    esc = html.escape
    by_id = {g["clip_id"]: g for g in gates}
    parts = [f"<!doctype html><html><head><meta charset='utf-8'><title>Pre-violence buildup report</title>"
             f"<style>{CSS}</style></head><body>",
             "<h1>Pre-violence buildup: what happens before the act</h1>",
             f"<small>{len(gates)} clips, behaviors detected from person tracks (M8a) by the M9 gate. "
             "Violent footage is not shown: each video stops before the detected escalation.</small>",
             "<h2>How to read this</h2><div class='note'>"
             "<b>APPROACH</b> distance shrinking &middot; <b>FOLLOW</b> one person trails another, close, same heading &middot; "
             "<b>HOVER</b> lingering next to someone standing still &middot; <b>CORNER</b> one person stands between another and "
             "the nearest frame edge (the stand-in for an exit) &middot; <b>ESCALATION</b> sudden relative-speed burst while close. "
             "A clip is <i>flagged</i> when FOLLOW, HOVER or CORNER occurs, or an APPROACH ends in ESCALATION. "
             "<b>Ordered</b> = the behaviors started in escalating order (approach, then follow/hover, then corner).<br><br>"
             "<b>Limits.</b> Everything comes from 2D positions of detected people: small or distant people are missed "
             "(see <i>no pair</i>), a false box (a chair, a scooter) can create a false behavior, hand-held cameras blur the motion "
             "signals, and the act's start was never annotated, so the video cut is a heuristic. "
             "<b>Normal</b> is the false-alarm baseline: compare every category with it. "
             "Rates are tuned on this same data; the review sheet is the independent check.</div>",
             "<h2>Categories</h2><table><tr><th>category</th><th>clips</th><th>no pair</th><th>flagged</th><th>FOLLOW</th>"
             "<th>HOVER</th><th>CORNER</th><th>ESCALATION</th><th>ordered</th><th>median buildup (s)</th>"
             "<th>median start-to-escalation (s)</th></tr>"]
    for r in cats:
        parts.append(f"<tr><td>{esc(r['category'])}</td><td>{r['clips']}</td><td>{_pct(r['no_pair'])}</td>"
                     f"<td>{_pct(r['flagged'])}</td><td>{_pct(r['FOLLOW'])}</td><td>{_pct(r['HOVER'])}</td>"
                     f"<td>{_pct(r['CORNER'])}</td><td>{_pct(r['ESCALATION'])}</td><td>{_pct(r['ordered'])}</td>"
                     f"<td>{_num(r['median_buildup_s'])}</td><td>{_num(r['median_time_to_escalation_s'])}</td></tr>")
    parts.append("</table>")
    for n in notes:
        parts.append(f"<p><small>{esc(n)}</small></p>")
    params = gates[0].get("params") if gates else None
    if params:
        parts.append("<details><summary><small>Thresholds used (body heights = bh, seconds = s)</small></summary><small>"
                     + ", ".join(f"{esc(k)}={v}" for k, v in sorted(params.items())) + "</small></details>")
    for cat, items in showcase.items():
        label = " (false-alarm baseline)" if cat == "Normal" else ""
        parts.append(f"<h2>{esc(cat)}{label}</h2><div class='grid'>")
        for cid, why in items:
            g, r = by_id[cid], rendered.get(cid, {})
            parts.append("<div class='card'>")
            if r.get("storyboard"):
                parts.append(f"<a href='{esc(r['storyboard'])}'><img loading='lazy' src='{esc(r['storyboard'])}'></a>")
            parts.append(f"<b>{esc(cid)}</b> <span class='tag {'top' if why == 'top' else ''}'>{why}</span> "
                         f"<span class='tag'>{'flagged' if g['flag'] else 'not flagged'}</span>"
                         f"<div class='claim'>{esc(claim_text(g))}</div>")
            if r.get("video"):
                parts.append(f"<video controls preload='none' src='{esc(r['video'])}'></video>")
            else:
                parts.append("<small>no pre-violence footage to show</small>")
            parts.append("</div>")
        parts.append("</div>")
    parts.append("</body></html>")
    (out_dir / "index.html").write_text("\n".join(parts), encoding="utf-8")


def _render_job(args):
    clip, cfg, overrides, out_dir = args
    try:
        r = render_clip(clip, cfg, overrides, out_dir)
        rel = lambda p: str(Path(p).relative_to(out_dir)).replace("\\", "/") if p else ""
        return clip["clip_id"], {"video": rel(r["video"]), "storyboard": rel(r["storyboard"])}
    except Exception as e:  # one bad clip must not stop the report
        log.warning("Could not render %s: %s", clip["clip_id"], e)
        return clip["clip_id"], {}


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    ap = argparse.ArgumentParser(description="Build the Phase 1 buildup report")
    ap.add_argument("--config", default=None)
    ap.add_argument("--no-videos", action="store_true", help="tables and HTML only")
    ap.add_argument("--all", action="store_true", help="render every clip (slow)")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE", help="override report settings")
    args = ap.parse_args()

    cfg = load_config(args.config)
    cfg["report"] = {**cfg["report"], **parse_overrides(args.set)}
    rep = cfg["report"]
    out_dir = resolve_path(rep["report_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    clean = json.loads(resolve_path(cfg["preprocess"]["clean_manifest_path"]).read_text("utf-8"))
    clips = {c["clip_id"]: c for c in clean["clips"]}
    gdir = resolve_path(cfg["gate"]["gate_dir"])
    gates = []
    for c in clean["clips"]:
        p = gdir / c["category"] / f"{c['clip_id']}_gate.json"
        if p.exists():
            gates.append(load_gate(p))
    if not gates:
        raise SystemExit("No M9 results found. Run: python -m src.assm.gate")
    overrides = load_overrides(rep["overrides_csv"])

    showcase = select_showcase(gates, int(rep["showcase_top"]), int(rep["showcase_random"]), int(rep["seed"]))
    review = select_review(gates, int(rep["review_flagged"]), int(rep["review_unflagged"]), int(rep["seed"]))
    want = {cid for items in showcase.values() for cid, _ in items} | {g["clip_id"] for g, _ in review}
    if args.all:
        want = {g["clip_id"] for g in gates}
    rendered = {}
    if not args.no_videos:
        jobs = [(clips[cid], cfg, overrides, out_dir) for cid in sorted(want)]
        print(f"Rendering {len(jobs)} clips (video + storyboard) ...")
        if args.workers > 1:
            from concurrent.futures import ProcessPoolExecutor
            with ProcessPoolExecutor(args.workers) as ex:
                rendered = dict(ex.map(_render_job, jobs))
        else:
            rendered = dict(_render_job(j) for j in jobs)

    cats = category_rows(gates)
    write_csv(out_dir / "categories.csv", cats)
    write_csv(out_dir / "clips.csv", clip_rows(gates, rep, overrides, rendered))
    by_id = {g["clip_id"]: g for g in gates}
    write_csv(out_dir / "review_sheet.csv", [
        {"clip_id": g["clip_id"], "category": g["category"], "group": grp, "system_claim": claim_text(g),
         "video": rendered.get(g["clip_id"], {}).get("video", ""),
         "verdict (correct / partly / wrong / missed)": "", "notes": ""} for g, grp in review],
        ["clip_id", "category", "group", "system_claim", "video", "verdict (correct / partly / wrong / missed)", "notes"])
    notes = []
    if args.no_videos:
        notes.append("Videos were not rendered in this run (--no-videos).")
    build_html(gates, cats, showcase, rendered, rep, out_dir, notes)

    print(f"\nReport written to {out_dir}\n  index.html, categories.csv, clips.csv, review_sheet.csv")
    print(f"\n{'category':<16}{'clips':>6}{'no pair':>9}{'flagged':>9}{'FOLLOW':>8}{'HOVER':>7}{'CORNER':>8}{'ESCAL':>7}{'ordered':>9}")
    for r in cats:
        print(f"{r['category']:<16}{r['clips']:>6}{_pct(r['no_pair']):>9}{_pct(r['flagged']):>9}{_pct(r['FOLLOW']):>8}"
              f"{_pct(r['HOVER']):>7}{_pct(r['CORNER']):>8}{_pct(r['ESCALATION']):>7}{_pct(r['ordered']):>9}")
    print(f"\nReview sheet: {len(review)} clips to check by eye -> fill the verdict column (see docs/VERIFY_M9.md)")


if __name__ == "__main__":
    main()
