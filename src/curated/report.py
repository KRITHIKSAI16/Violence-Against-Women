"""Curated dataset: the jury page. One self-contained HTML file (images and videos are linked relative to it) plus summary.csv.

Order on the page: what this shows -> headline numbers -> summary table -> per-category clip cards (video, distance figure, storyboard, plain sentences)
-> pre-violence vs Normal side by side (with the honest verdict) -> how it was done (models, heuristics) -> limits.
"""
import csv
import html
import os
from pathlib import Path

from src.curated.relation import THRESH

CSS = """
:root{--bg:#fbfbfa;--fg:#1d2227;--mut:#5d6670;--card:#fff;--line:#dfe3e8;--acc:#1f3b5c;--bad:#b83227;--ok:#2e7d4f}
@media (prefers-color-scheme:dark){:root{--bg:#14171a;--fg:#e6e9ec;--mut:#9aa4ae;--card:#1d2226;--line:#2d343a;--acc:#8fb4dc;--bad:#e8786d;--ok:#6cc08f}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);font:15px/1.5 system-ui,Segoe UI,Roboto,sans-serif}
main{max-width:1100px;margin:auto;padding:20px 16px 60px}h1{font-size:26px;margin:.2em 0}h2{font-size:20px;margin:1.8em 0 .5em;border-bottom:1px solid var(--line);padding-bottom:4px}
h3{font-size:16px;margin:.2em 0}.mut{color:var(--mut)}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:10px}
.stat{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:10px 12px}.stat b{font-size:22px;display:block}
table{border-collapse:collapse;width:100%;font-size:13.5px;background:var(--card)}th,td{border:1px solid var(--line);padding:5px 8px;text-align:left;vertical-align:top}th{background:var(--bg)}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:12px;margin:14px 0}.cols{display:grid;grid-template-columns:minmax(0,1.1fr) minmax(0,1fr);gap:14px}
@media (max-width:800px){.cols{grid-template-columns:1fr}}video,img{max-width:100%;border-radius:6px;background:#000}ul{margin:.3em 0 .3em 1.1em;padding:0}
.tag{display:inline-block;border-radius:10px;padding:1px 9px;font-size:12px;border:1px solid var(--line);margin-right:4px}.warn{color:var(--bad)}.box{background:var(--card);border:1px solid var(--line);border-left:4px solid var(--acc);padding:10px 14px;border-radius:6px;margin:10px 0}
"""


def e(x):
    return html.escape(str(x))


def rel(path, base):
    return os.path.relpath(str(path), str(base)).replace("\\", "/")


def short_line(r):
    if r.get("interaction"):
        L = r["interaction"]["last"][max(r["interaction"]["last"], key=float)]
        t = f"{r['interaction']['leadup']['type']}"
        if L["dist_start_m"] is not None:
            t += f"; {L['dist_start_m']:.1f} -> {L['dist_end_m']:.1f} m in the last {L['seconds']:.1f} s"
        return t
    return f"no pair: {r.get('reason', '?').replace('_', ' ')}"


def write_csv(results, path):
    cols = ["clip_id", "category", "violence_start_s", "footage_s", "pair_found", "leadup_type", "first_dist_m", "min_dist_m", "last_dist_start_m", "last_dist_end_m",
            "net_closing_last_m", "concern_last", "relation_seconds", "note"]
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(cols)
        for r in results:
            i = r.get("interaction")
            L = i["last"][max(i["last"], key=float)] if i else {}
            w.writerow([r["clip_id"], r["category"], "" if r.get("start_s") is None else r["start_s"], r["duration_s"], int(not r["no_pair"]),
                        i["leadup"]["type"] if i else "no pair: " + r.get("reason", ""), i["first_dist_m"] if i else "", i["min_dist_m"] if i else "",
                        L.get("dist_start_m", ""), L.get("dist_end_m", ""), L.get("net_closing_m", ""), "" if not i or i["concern_last_s"] is None else round(i["concern_last_s"], 2),
                        ";".join(f"{k}:{v}" for k, v in (i["label_seconds"].items() if i else [])), r.get("note", "")])


def card(r, assets, base):
    cid = r["clip_id"]
    a = assets.get(cid, {})
    head = f"<h3>{e(cid)} <span class='tag'>{e(r['category'])}</span>" + (f"<span class='tag'>violence starts at {r['start_s']:.1f} s</span>" if r.get("start_s") is not None else "<span class='tag'>Normal</span>")
    head += f"<span class='tag'>{e(short_line(r)[:60])}</span></h3>"
    media = ""
    if a.get("video"):
        media += f"<video controls preload='metadata' src='{rel(a['video'], base)}'></video>"
    right = ""
    if a.get("figure"):
        right += f"<img alt='distance over time' src='{rel(a['figure'], base)}'>"
    if a.get("storyboard"):
        right += f"<img alt='key frames' src='{rel(a['storyboard'], base)}'>"
    lines = "".join(f"<li>{e(l)}</li>" for l in r["lines"])
    ml = ""
    if a.get("ml"):
        ml = f"<p class='mut'>Video model (zero-shot, {e(a['ml']['encoder'])}): concern margin in the last window {a['ml']['last']:+.2f}; highest {a['ml']['max']:+.2f}.</p>"
    note = f"<p class='warn'>{e(r['note'])}</p>" if r.get("note") else ""
    return f"<div class='card'>{head}<div class='cols'><div>{media}{note}</div><div>{right}</div></div><ul>{lines}</ul>{ml}</div>"


def build_html(results, ev, assets, info, out_path):
    """results: per-clip result dicts; ev: src.curated.evaluate.evaluate_all output; assets: {clip: {video, figure, storyboard, ml}}; info: dict of run facts."""
    out_path = Path(out_path)
    base = out_path.parent
    viol = [r for r in results if not r["is_normal"]]
    norm = [r for r in results if r["is_normal"]]
    with_pair = [r for r in viol if not r["no_pair"]]
    nv = ev["normal_vs_pre"]
    stats = [("violent clips", len(viol)), ("with an interacting pair found", f"{len(with_pair)} ({len(with_pair) / max(1, len(viol)):.0%})"),
             ("Normal control clips", len(norm)), ("concern higher in the last 3 s (same clip)", f"{ev['trend']['higher_in_last']} of {ev['trend']['clips']}")]
    parts = [f"<!doctype html><html lang='en'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'><title>Pre-violence interaction understanding</title><style>{CSS}</style></head><body><main>",
             "<h1>What happens between the people before the violence</h1>",
             "<p class='mut'>Hand-picked clips, each cut exactly at the start of the violence (start times annotated by the team). For every clip: who moves toward whom, how the distance between them changes, "
             "whether one follows or approaches from behind, where hands and arms are, and what the last seconds before the violence look like. Nothing after the start time is analysed or shown.</p>",
             "<div class='grid'>" + "".join(f"<div class='stat'><b>{e(v)}</b>{e(k)}</div>" for k, v in stats) + "</div>",
             "<h2>Summary</h2><table><tr><th>clip</th><th>category</th><th>starts at</th><th>lead-up (heuristic)</th><th>distance in the last seconds</th><th>closest</th></tr>"]
    for r in viol:
        i = r.get("interaction")
        L = i["last"][max(i["last"], key=float)] if i else None
        parts.append(f"<tr><td><a href='#{e(r['clip_id'])}'>{e(r['clip_id'])}</a></td><td>{e(r['category'])}</td><td>{r['start_s']:.1f} s</td><td>{e(i['leadup']['type'] if i else 'no pair: ' + r.get('reason', '').replace('_', ' '))}</td>"
                     f"<td>{'' if not L or L['dist_start_m'] is None else '%.1f -> %.1f m' % (L['dist_start_m'], L['dist_end_m'])}</td><td>{'' if not i or i['min_dist_m'] is None else '%.1f m' % i['min_dist_m']}</td></tr>")
    parts.append("</table>")
    for cat in sorted({r["category"] for r in viol}):
        parts.append(f"<h2>{e(cat.replace('_', ' '))}</h2>")
        for r in [x for x in viol if x["category"] == cat]:
            parts.append(f"<a id='{e(r['clip_id'])}'></a>" + card(r, assets, base))
    parts.append("<h2>Pre-violence footage vs Normal clips</h2>")
    parts.append(f"<div class='box'><b>{e(nv['verdict'])}</b><br><span class='mut'>Clips with a measurable pair: {nv['violent_n']} pre-violence, {nv['normal_n']} Normal. "
                 f"Median concern {nv['median_violent']:.2f} vs {nv['median_normal']:.2f} (Normal clips are scored by their highest window, which is the hardest comparison for us). "
                 f"AUC {nv['auc']:.2f} (95% interval {nv['auc_low']:.2f}-{nv['auc_high']:.2f}); best single filming-style feature alone: {nv['style_only_auc']:.2f}; "
                 f"same raw resolution only: {nv['same_resolution']['auc']:.2f} (n = {nv['same_resolution']['violent']} vs {nv['same_resolution']['normal']}).</span></div>")
    if info.get("concern_figure"):
        parts.append(f"<img alt='concern distribution' src='{rel(info['concern_figure'], base)}'>")
    if info.get("ml_table"):
        parts.append("<h3>Video models next to the geometry</h3><pre>" + e("\n".join(info["ml_table"])) + "</pre>")
        parts.append(f"<p class='mut'>{e(info.get('ml_note', ''))}</p>")
    parts.append("<h3>Normal clips, same analysis</h3>")
    for r in norm[:8]:
        parts.append(card(r, assets, base))
    parts.append("<h2>How it was done</h2><div class='box'><ul>"
                 f"<li>Clips are normalised (30 fps, longest side 640 px) and trimmed at the start time, so no cue from the violent act can leak in.</li>"
                 f"<li>People: {e(info.get('perception', 'pose detector + tracker'))}; tracking restarts at editing cuts ({r_shots(results)} clips contain cuts).</li>"
                 "<li>Distances in meters come from a 1.7 m person-height prior and an assumed 60 degree field of view: approximate, good for trends, not for exact values.</li>"
                 "<li>Relations (following, approach from behind, approach, reach, walking together, standing together) and the lead-up type are rule-based heuristics with the thresholds below; their weights are not tuned on data.</li>"
                 f"<li>Interaction graph layers follow the vocabulary of the Hierarchical Interlacement Graph (situation, position, interaction, relation); this is {e(info.get('hig', 'HIG-structured, not the pretrained HIG model'))}. Appearance is skipped on purpose: no identity, gender or age is inferred.</li>"
                 f"<li>Video models (frozen, no text is generated): {e(info.get('encoders', 'none loaded'))}.</li></ul>"
                 "<details><summary>Thresholds</summary><pre>" + e("\n".join(f"{k} = {v}" for k, v in THRESH.items())) + "</pre></details></div>")
    parts.append("<h2>Limits</h2><div class='box'><ul>"
                 "<li>Short clips (1-3 s before the violence) allow little to be measured; the page says so per clip instead of inventing a story.</li>"
                 "<li>If a person is not detected (small, far, occluded, behind a car), no pair exists for that clip; this is a perception limit.</li>"
                 "<li>Normal clips come from a different source than the other categories. A difference between them is only read as behavior when it beats the filming-style baseline and holds at equal resolution.</li>"
                 f"<li>{len(viol)} violent clips is a small set: counts, not percentages with false precision.</li></ul></div></main></body></html>")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("".join(parts), encoding="utf-8")
    return out_path


def r_shots(results):
    return sum(1 for r in results if r.get("shots", 1) > 1)
