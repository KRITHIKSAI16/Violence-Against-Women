"""Human validation labels: an offline page where one person marks, per clip, when the violence begins and when the lead-up begins.

The point is a small, trustworthy reference (about 40 clips, roughly a minute each) that tells us how accurate the automatic labels and the
phase model are. The page deliberately shows NO model proposal, so the person is not nudged. It makes a folder (360p proxy videos + index.html);
open index.html in a browser, play, press the buttons at the right moment, and download labels.csv (same format as `src.phase.labels`).

    python -m src.label.annotate_pack [--config CFG] [--clips A B | --holdout] [--out data/label/pack]
    then, after labeling:  copy labels.csv to data/label/human_labels.csv
"""
import argparse
import html
import json
import subprocess
from pathlib import Path

from src.config import load_config, resolve_path

PAGE = """<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>VAW labeling</title><style>
body{font-family:system-ui,sans-serif;margin:0;padding:16px;background:#fafafa;color:#111;max-width:900px;margin:auto}
video{width:100%;max-height:60vh;background:#000}button{padding:10px 14px;margin:4px 4px 4px 0;font-size:15px;cursor:pointer}
.row{margin:10px 0}.v{font-weight:600}.done{color:#0a7d33}code{background:#eee;padding:2px 5px}
@media (prefers-color-scheme:dark){body{background:#161616;color:#eee}code{background:#333}}
</style></head><body>
<h2>Label when it happens</h2>
<p>Play the clip. Press <b>Violence starts now</b> at the first moment of physical violence (hit, grab by force, choke, shooting, forced drag).
Press <b>Lead-up starts now</b> when the aggressor first starts approaching, following or confronting the victim. If there is no lead-up (a sudden
attack) press <b>No lead-up</b>; if there is no violence in the clip press <b>No violence</b>. You can step with the arrow keys (&larr; &rarr; one frame at a time when paused).</p>
<div class="row"><b id="name"></b> <span id="count"></span></div>
<video id="vid" controls preload="auto"></video>
<div class="row"><button id="act">Violence starts now</button><button id="lead">Lead-up starts now</button><button id="nolead">No lead-up</button><button id="noviol">No violence</button></div>
<div class="row">Violence starts: <span class="v" id="vact">-</span> &nbsp; Lead-up starts: <span class="v" id="vlead">-</span> &nbsp; <input id="note" placeholder="note (optional)" size="30"></div>
<div class="row"><button id="prev">&laquo; Previous</button><button id="next">Next &raquo;</button><button id="save">Download labels.csv</button> <span id="saved"></span></div>
<script>
const CLIPS = __CLIPS__;
let i = 0; const L = {}; try { Object.assign(L, JSON.parse(localStorage.getItem("vaw_labels") || "{}")); } catch (e) {}
const $ = id => document.getElementById(id), vid = $("vid");
function cur() { const c = CLIPS[i]; return L[c.id] = L[c.id] || {act: null, lead: null, none: false, nolead: false, note: ""}; }
function persist() { try { localStorage.setItem("vaw_labels", JSON.stringify(L)); } catch (e) {} }
function show() {
  const c = CLIPS[i], l = cur(); vid.src = c.src; $("name").textContent = c.id; $("count").textContent = "(" + (i + 1) + " of " + CLIPS.length + ", labeled " + Object.values(L).filter(x => x.act !== null || x.none).length + ")";
  $("vact").textContent = l.none ? "no violence" : (l.act === null ? "-" : l.act.toFixed(2) + " s"); $("vlead").textContent = l.nolead ? "no lead-up" : (l.lead === null ? "-" : l.lead.toFixed(2) + " s"); $("note").value = l.note || "";
}
$("act").onclick = () => { const l = cur(); l.act = vid.currentTime; l.none = false; persist(); show(); };
$("lead").onclick = () => { const l = cur(); l.lead = vid.currentTime; l.nolead = false; persist(); show(); };
$("nolead").onclick = () => { const l = cur(); l.lead = null; l.nolead = true; persist(); show(); };
$("noviol").onclick = () => { const l = cur(); l.none = true; l.act = null; l.lead = null; persist(); show(); };
$("note").oninput = () => { cur().note = $("note").value; persist(); };
$("next").onclick = () => { if (i < CLIPS.length - 1) { i++; show(); } }; $("prev").onclick = () => { if (i > 0) { i--; show(); } };
document.addEventListener("keydown", e => { if (e.target.tagName === "INPUT") return; if (e.key === "ArrowRight") { vid.pause(); vid.currentTime += 1 / 25; } if (e.key === "ArrowLeft") { vid.pause(); vid.currentTime -= 1 / 25; } });
$("save").onclick = () => {
  const q = s => '"' + String(s || "").replace(/"/g, '""') + '"';
  let out = "clip_id,act_start_s,buildup_start_s,source,note\\n";
  for (const c of CLIPS) { const l = L[c.id]; if (!l || (l.act === null && !l.none)) continue;
    out += [c.id, l.act === null ? "" : l.act.toFixed(2), l.lead === null || l.none ? "" : l.lead.toFixed(2), "human", q(l.note)].join(",") + "\\n"; }
  const a = document.createElement("a"); a.href = URL.createObjectURL(new Blob([out], {type: "text/csv"})); a.download = "labels.csv"; a.click(); $("saved").textContent = "downloaded";
};
show();
</script></body></html>
"""


def make_proxy(src, dst, height=360):
    """Small H.264 copy for fast browsing (even sizes, no audio, keyframes often so seeking is exact enough)."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", str(src), "-vf", f"scale=-2:{height}", "-an", "-c:v", "libx264", "-pix_fmt", "yuv420p",
                    "-g", "5", "-preset", "veryfast", str(dst)], check=True)


def build_pack(clips, out_dir, height=360):
    """clips: manifest entries. Writes out_dir/videos/<id>.mp4 and out_dir/index.html; returns the page path."""
    out = Path(out_dir)
    items = []
    for c in clips:
        dst = out / "videos" / f"{c['clip_id']}.mp4"
        if not dst.exists():
            make_proxy(resolve_path(c["path"]), dst, height)
        items.append({"id": c["clip_id"], "src": f"videos/{c['clip_id']}.mp4"})
    page = out / "index.html"
    page.write_text(PAGE.replace("__CLIPS__", json.dumps(items)), encoding="utf-8")
    return page


def main():
    ap = argparse.ArgumentParser(description="Offline labeling page for the human validation set")
    ap.add_argument("--config", default=None)
    ap.add_argument("--clips", nargs="*", default=None)
    ap.add_argument("--holdout", action="store_true", help="the held-out local clips (violent ones) plus --extra more violent clips")
    ap.add_argument("--extra", type=int, default=13, help="with --holdout: extra violent clips from the rest (evenly spaced)")
    ap.add_argument("--out", default="data/label/pack")
    args = ap.parse_args()
    cfg = load_config(args.config)
    clean = json.loads(resolve_path(cfg["preprocess"]["clean_manifest_path"]).read_text("utf-8"))["clips"]
    if args.clips:
        clips = [c for c in clean if c["clip_id"] in args.clips]
    else:
        hold = {l.strip() for l in resolve_path(cfg["perception"]["holdout_file"]).read_text("utf-8").splitlines() if l.strip() and not l.startswith("#")}
        clips = [c for c in clean if c["clip_id"] in hold]                 # includes Normal clips so the person can mark "no violence"
        rest = [c for c in sorted(clean, key=lambda c: c["clip_id"]) if c["clip_id"] not in hold and c["category"] != "Normal"]
        if args.extra and rest:
            step = max(1, len(rest) // args.extra)
            clips += rest[::step][:args.extra]
    page = build_pack(clips, resolve_path(args.out))
    print(f"{len(clips)} clips -> {page}\nOpen it in a browser, label, press 'Download labels.csv', then copy it to {cfg['label']['human_labels']}")


if __name__ == "__main__":
    main()
