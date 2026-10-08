"""Jury showcase for the classification pipelines: a violent video next to a Normal one, shown through the features the pipeline measured.

    snapshot_<k>.png      16:9 picture for a slide: three key frames of each video (the pair boxed, live distance), the distance curve with the relation
                          timeline, and the measured numbers side by side (+ the out-of-fold model score when predictions.csv exists)
    compare_video_<k>.mp4 the same two videos playing side by side with the live distance strip and the behaviour caption of each
    features_overview.png 16:9 picture over ALL videos: how often each cue appears in violent vs Normal videos, closest distance, concern score, classifier AUCs

Everything is read from the outputs of the earlier stages (stories, tracks, context, processed videos); nothing is recomputed on a GPU. The examples are chosen
by a rule (clear lead-up cues vs calm behaviour), not at random: the snapshot says so. All distances are monocular estimates; the cues are heuristics.
"""
import csv
import json
import subprocess
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from src.assm.track_poses import load_tracks  # noqa: E402
from src.config import resolve_path  # noqa: E402
from src.context.features import load_context  # noqa: E402
from src.curated.figures import SPAN_COLOR, ZONES, span_kind  # noqa: E402
from src.curated.interaction import PHRASE  # noqa: E402
from src.curated.video import KIND_BGR, ZONE_BGR, span_at  # noqa: E402
from src.report.buildup_video import put_text  # noqa: E402
from src.video.shots import local_id  # noqa: E402

RED, GREEN, INK = "#c0392b", "#2e8b57", "#1f3b5c"
VIOLENT_BGR, NORMAL_BGR = (43, 57, 192), (87, 139, 46)
CUES = [("contact_or_reach", "reaches for / touches"), ("follows", "follows"), ("approaches_from_behind", "approaches from behind"), ("approaches", "moves toward"),
        ("walking_together", "walks together"), ("standing_together", "stands close together")]
CUE_MIN_S = 0.5                                                                      # a cue counts for a video when it lasts at least this long


# ---------------------------------------------------------------- cases and facts

def load_cases(cfg):
    """Every clip of the run: {clip, result, y}. y = 1 violent (any category except Normal), 0 Normal."""
    man = json.loads(Path(cfg["preprocess"]["clean_manifest_path"]).read_text(encoding="utf-8"))["clips"]
    out = []
    for c in man:
        p = Path(cfg["curated"]["stories_dir"]) / c["category"] / f"{c['clip_id']}.json"
        if p.exists():
            out.append({"clip": c, "result": json.loads(p.read_text(encoding="utf-8")), "y": int(c["category"] != "Normal")})
    return out


def has_pair(res):
    return bool(res) and not res.get("no_pair") and bool(res.get("interaction"))


def _sec(res, labels):
    ls = res["interaction"].get("label_seconds", {})
    return float(sum(ls.get(k, 0.0) for k in labels))


def evidence(res):
    """Rule used to pick the violent example: seconds of reaching / following / approaching plus the highest concern window (heuristic units)."""
    cm = res["interaction"].get("concern_max_window")
    return _sec(res, ["contact_or_reach", "follows", "approaches_from_behind", "approaches"]) + (cm if cm is not None and np.isfinite(cm) else 0.0)


def pick_examples(cases, k=1, min_s=8.0, max_s=30.0, min_pair_s=4.0, violent=None, normal=None):
    """-> [(violent case, Normal case)] k pairs. Violent: the most cue evidence among videos with a measured pair; Normal: the least. Overrides by clip id."""
    def ok(c):
        r = c["result"]
        return has_pair(r) and min_s <= r["duration_s"] <= max_s and (r["interaction"].get("pair_visible_s") or 0) >= min_pair_s
    byid = {c["clip"]["clip_id"]: c for c in cases}
    v = sorted([c for c in cases if c["y"] == 1 and ok(c)], key=lambda c: -evidence(c["result"]))
    n = sorted([c for c in cases if c["y"] == 0 and ok(c)], key=lambda c: evidence(c["result"]))
    if violent:
        v = [byid[violent]] + [c for c in v if c["clip"]["clip_id"] != violent]
    if normal:
        n = [byid[normal]] + [c for c in n if c["clip"]["clip_id"] != normal]
    return list(zip(v[:k], n[:k]))


def pair_distance(cfg, case):
    """-> (per-frame pair distance in meters, fps)."""
    c = case["clip"]
    scene, arrays = load_context(resolve_path(cfg["context"]["context_dir"]), c["category"], c["clip_id"])
    return np.asarray(arrays[tuple(case["result"]["interaction"]["ids"])]["dist_m"], float), float(scene["fps"])


def facts(res, d, fps):
    """The numbers shown under a video: [(name, text)]."""
    it = res["interaction"]
    fin = d[np.isfinite(d)]
    near = float(np.mean(fin < 1.2)) if len(fin) else float("nan")
    pe = res.get("people", {})
    cm = it.get("concern_mean")
    return [("video length", f"{res['duration_s']:.0f} s"), ("two people tracked together", f"{it.get('pair_visible_s', 0):.0f} s"),
            ("closest distance", f"{it['min_dist_m']:.1f} m" if it.get("min_dist_m") is not None else "n/a"),
            ("time closer than 1.2 m", "n/a" if near != near else f"{100 * near:.0f} %"),
            ("reaches for / touches", f"{_sec(res, ['contact_or_reach']):.1f} s"),
            ("follows / approaches", f"{_sec(res, ['follows', 'approaches_from_behind', 'approaches']):.1f} s"),
            ("walks / stands together", f"{_sec(res, ['walking_together', 'standing_together']):.1f} s"),
            ("concern score, mean", f"{cm:+.2f}" if cm is not None and np.isfinite(cm) else "n/a"),
            ("people in view (max)", f"{pe.get('max_people', '?')}")]


def load_scores(report_dir):
    """-> ({clip_id: P(violent) of the best model, out-of-fold}, best model name) or ({}, None)."""
    rd = Path(report_dir)
    try:
        best = json.loads((rd / "metrics.json").read_text(encoding="utf-8")).get("best")
        with open(rd / "predictions.csv", encoding="utf-8") as f:
            return {r["clip_id"]: float(r[f"{best}_p"]) for r in csv.DictReader(f)}, best
    except (OSError, KeyError, ValueError, TypeError):
        return {}, None


def cohort(cases):
    """Per class over the videos with a measured pair: counts, share with each cue (>= CUE_MIN_S), closest distance and concern mean per video."""
    out = {}
    for y, name in ((1, "violent"), (0, "Normal")):
        sel = [c for c in cases if c["y"] == y]
        pr = [c["result"] for c in sel if has_pair(c["result"])]
        out[name] = {"videos": len(sel), "with_pair": len(pr),
                     "cue_share": {k: (float(np.mean([_sec(r, [k]) >= CUE_MIN_S for r in pr])) if pr else float("nan")) for k, _ in CUES},
                     "min_dist": [r["interaction"]["min_dist_m"] for r in pr if r["interaction"].get("min_dist_m") is not None],
                     "concern": [r["interaction"]["concern_mean"] for r in pr if r["interaction"].get("concern_mean") is not None and np.isfinite(r["interaction"]["concern_mean"])]}
    return out


# ---------------------------------------------------------------- one annotated pair

class PairView:
    """Draws one video's key pair (the one that moves toward the other in red) with the live distance, and says what they do at that moment."""

    def __init__(self, cfg, case):
        c, self.res = case["clip"], case["result"]
        self.case = case
        self.src = resolve_path(c["path"])
        self.tracks = load_tracks(resolve_path(cfg["assm"]["tracks_dir"]) / c["category"] / f"{c['clip_id']}.npz")
        self.d, self.fps = pair_distance(cfg, case)
        it = self.res["interaction"]
        self.ids, self.spans = it["ids"], it["spans"]
        last = it.get("last") or {}
        self.last_actor = last[max(last, key=float)].get("actor") if last else None
        self.rows = {}
        for k, (f, tid) in enumerate(zip(self.tracks["frame_idx"].tolist(), self.tracks["track_id"].tolist())):
            if tid in self.ids:
                self.rows.setdefault(f, {})[tid] = k
        fin = self.d[np.isfinite(self.d)]
        self.dmax = max(3.0, float(fin.max()) * 1.25) if len(fin) else 3.0

    def annotate(self, img, f):
        """Draw boxes + distance on img (in place). -> (caption text, caption BGR colour)."""
        t = f / self.fps
        cur = span_at(self.spans, t)
        actor = cur["actor"] if cur is not None and cur.get("actor") is not None else (self.last_actor if cur is not None and cur["label"] in ("approaches", "follows", "approaches_from_behind") else None)
        pts = {}
        for tid, k in self.rows.get(f, {}).items():
            x1, y1, x2, y2 = (int(v) for v in self.tracks["bbox"][k])
            cv2.rectangle(img, (x1, y1), (x2, y2), (60, 60, 230) if tid == actor else (230, 140, 40), 2)
            put_text(img, f"id{local_id(tid)}", (x1, max(12, y1 - 4)), 0.5, (255, 255, 255), 1)
            pts[tid] = ((x1 + x2) // 2, y2)
        dv = self.d[f] if f < len(self.d) else float("nan")
        if len(pts) == 2 and np.isfinite(dv):
            a, b = list(pts.values())
            cv2.line(img, a, b, (0, 0, 255) if dv < 1.2 else (0, 200, 255), 2)
            put_text(img, f"{dv:.1f} m", ((a[0] + b[0]) // 2, (a[1] + b[1]) // 2 - 6), 0.6, (255, 255, 255), 2)
        if cur is not None:
            a_, b_ = (f"id{local_id(cur['actor'])}", f"id{local_id(cur['target'])}") if cur.get("actor") is not None else (f"id{local_id(self.ids[0])}", f"id{local_id(self.ids[1])}")
            return PHRASE[cur["label"]].format(a=a_, b=b_), KIND_BGR[span_kind(cur["label"])]
        return ("the two are apart" if np.isfinite(dv) else "no pair measurement at this moment"), (90, 90, 90)

    def key_times(self, n=3):
        """First measured moment, the strongest cue moment (middle of the first reach / follow / approach span, else the closest frame), the closest approach."""
        fin = np.where(np.isfinite(self.d), self.d, np.inf)
        closest = float(np.argmin(fin)) / self.fps if np.isfinite(fin).any() else 0.0
        cue = [s for s in self.spans if s["label"] in ("contact_or_reach", "follows", "approaches_from_behind", "approaches")]
        mid = (cue[0]["start_s"] + 0.5 * (cue[0]["end_s"] - cue[0]["start_s"])) if cue else self.res["duration_s"] / 2
        first = float(np.argmax(np.isfinite(self.d))) / self.fps if np.isfinite(self.d).any() else 0.0
        ts = [first + 0.2, mid, closest]
        return [min(max(t, 0.0), max(0.0, self.res["duration_s"] - 0.2)) for t in ts][:n]

    def frame_at(self, t):
        cap = cv2.VideoCapture(str(self.src))
        f = int(round(t * self.fps))
        cap.set(cv2.CAP_PROP_POS_FRAMES, f)
        ok, img = cap.read()
        cap.release()
        if not ok:
            return None, f
        return img, f


# ---------------------------------------------------------------- the slide picture

def _model_line(case, scores, best):
    p = scores.get(case["clip"]["clip_id"])
    return "" if p is None else f"Model score (out-of-fold, {best}): P(violent) = {p:.2f}"


def _column(fig, x0, w, case, view, label, colour, scores, best, top):
    """One column of the snapshot: header, 3 key frames, distance curve + relation timeline, numbers."""
    res, c = view.res, case["clip"]
    fig.text(x0, top, f"{label}: {c['clip_id']}", fontsize=21, fontweight="bold", color=colour, va="top")
    ml = _model_line(case, scores, best)
    if ml:
        fig.text(x0, top - 0.034, ml, fontsize=11.5, color="#444", va="top")
    gap = 0.008
    tw = (w - 2 * gap) / 3
    ytop = top - 0.075
    hfr = 0.27
    for i, t in enumerate(view.key_times()):
        ax = fig.add_axes([x0 + i * (tw + gap), ytop - hfr, tw, hfr])
        img, f = view.frame_at(t)
        ax.axis("off")
        if img is None:
            continue
        cap_txt, _ = view.annotate(img, f)
        ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        ax.set_title(f"{f / view.fps:.1f} s  {cap_txt}"[:34], fontsize=10, loc="left", color="#222", pad=3)
    # distance curve + relation timeline
    ya = ytop - hfr - 0.215
    ax = fig.add_axes([x0 + 0.03, ya + 0.065, w - 0.075, 0.115])
    bx = fig.add_axes([x0 + 0.03, ya, w - 0.075, 0.045], sharex=ax)
    d, tt = view.d, np.arange(len(view.d)) / view.fps
    top_d = max(6.0, view.dmax)
    for lo, hi, name, col in ZONES:
        ax.axhspan(lo, min(hi, top_d), color=col, lw=0)
        if lo < top_d:
            ax.text(1.004, (lo + min(hi, top_d)) / 2, name, transform=ax.get_yaxis_transform(), fontsize=7.5, color="#666", va="center")
    ax.plot(tt, d, color=INK, lw=2.2)
    ax.set_ylim(0, top_d)
    ax.set_ylabel("distance (m)", fontsize=10)
    ax.tick_params(labelsize=9, labelbottom=False)
    ax.grid(alpha=0.2)
    ax.set_title("distance between the two people", fontsize=11, loc="left")
    bx.set_ylim(0, 1)
    bx.set_yticks([])
    for sp in view.spans:
        kind = span_kind(sp["label"])
        bx.axvspan(sp["start_s"], sp["end_s"], color=SPAN_COLOR[kind], alpha=0.85 if kind != "neutral" else 0.45)
        bx.text((sp["start_s"] + sp["end_s"]) / 2, 0.5, sp["label"].replace("_", " "), ha="center", va="center", fontsize=7, color="white" if kind != "neutral" else "#333", clip_on=True)
    bx.set_xlim(0, res["duration_s"])
    bx.set_xlabel("time (s)     red = concerning behaviour, green = benign, grey = neutral", fontsize=9)
    bx.tick_params(labelsize=9)
    # numbers, two columns
    rows = facts(res, d, view.fps)
    half = (len(rows) + 1) // 2
    y0 = ya - 0.065
    for j, (name, val) in enumerate(rows):
        cx, yy = x0 + (0.0 if j < half else w / 2 + 0.01), y0 - (j % half) * 0.030
        fig.text(cx + 0.005, yy, name, fontsize=12, color="#333", va="top")
        fig.text(cx + w / 2 - 0.02, yy, val, fontsize=12, color=colour, va="top", ha="right", fontweight="bold")


def render_snapshot(cfg, vcase, ncase, out_path, scores=None, best=None):
    """16:9 picture (1920x1080): the violent video on the left, the Normal video on the right."""
    scores = scores or {}
    fig = plt.figure(figsize=(19.2, 10.8), dpi=100, facecolor="white")
    fig.text(0.02, 0.975, "What the pipeline measures: a violent video vs a Normal video", fontsize=27, fontweight="bold", color="#111", va="top")
    fig.text(0.02, 0.935, "Same pipeline on both: people tracked, distance between the pair, who moves toward whom, reach / contact, behaviour timeline.", fontsize=13.5, color="#444", va="top")
    views = [PairView(cfg, vcase), PairView(cfg, ncase)]
    _column(fig, 0.02, 0.46, vcase, views[0], "VIOLENT", RED, scores, best, 0.895)
    _column(fig, 0.52, 0.46, ncase, views[1], "NORMAL", GREEN, scores, best, 0.895)
    fig.add_artist(plt.Line2D([0.5, 0.5], [0.09, 0.89], color="#ddd", lw=1.5))
    fig.text(0.02, 0.028, "These two videos were picked by a rule (violent: most reach / follow / approach evidence; Normal: least), not at random, so they show the clearest contrast. "
             "Distances are monocular estimates; cues and the concern score are rule-based heuristics. Normal videos come from a different source than the violent ones.",
             fontsize=9.5, color="#555", va="bottom", wrap=True)
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=100, facecolor="white")
    plt.close(fig)
    return out


# ---------------------------------------------------------------- overview over all videos

def _strip(ax, groups, ylabel, title):
    rng = np.random.default_rng(0)
    for k, (name, vals, col) in enumerate(groups):
        v = np.asarray(vals, float)
        if len(v):
            ax.scatter(k + rng.uniform(-0.14, 0.14, len(v)), v, color=col, alpha=0.6, s=26)
            ax.hlines(np.median(v), k - 0.28, k + 0.28, color="black", lw=2.5)
            ax.text(k + 0.3, np.median(v), f"median {np.median(v):.2f}", fontsize=9, va="center")
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels([f"{g[0]}\n(n={len(g[1])})" for g in groups], fontsize=10)
    ax.set_xlim(-0.5, len(groups) - 0.2)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_title(title, fontsize=12, loc="left")
    ax.grid(alpha=0.2)


def render_overview(cohort_stats, metrics, out_path):
    """16:9 picture over all videos: cue shares, closest distance, concern, and (when available) the classifier AUCs with their baselines."""
    fig = plt.figure(figsize=(19.2, 10.8), dpi=100, facecolor="white")
    v, n = cohort_stats["violent"], cohort_stats["Normal"]
    fig.text(0.02, 0.975, "Violent vs Normal videos: how the measured features differ", fontsize=27, fontweight="bold", va="top")
    fig.text(0.02, 0.935, f"{v['videos']} violent and {n['videos']} Normal videos. Only videos where two people were tracked together are counted "
             f"({v['with_pair']} violent, {n['with_pair']} Normal).", fontsize=13.5, color="#444", va="top")
    ax = fig.add_axes([0.17, 0.10, 0.33, 0.75])
    y = np.arange(len(CUES))[::-1]
    ax.barh(y + 0.19, [100 * v["cue_share"][k] for k, _ in CUES], 0.36, color=RED, label=f"violent (n={v['with_pair']})")
    ax.barh(y - 0.19, [100 * n["cue_share"][k] for k, _ in CUES], 0.36, color=GREEN, label=f"Normal (n={n['with_pair']})")
    for yi, (k, _) in zip(y, CUES):
        for off, g, col in ((0.19, v, RED), (-0.19, n, GREEN)):
            s = g["cue_share"][k]
            if s == s:
                ax.text(100 * s + 1, yi + off, f"{100 * s:.0f}%", va="center", fontsize=9.5, color=col)
    ax.set_yticks(y)
    ax.set_yticklabels([lab for _, lab in CUES], fontsize=11.5)
    ax.set_xlim(0, 100)
    ax.set_xlabel(f"% of videos where the cue lasts at least {CUE_MIN_S:.1f} s", fontsize=10.5)
    ax.set_title("Behaviour cues between the key pair", fontsize=13, loc="left")
    ax.legend(fontsize=10.5, loc="lower right")
    ax.grid(axis="x", alpha=0.2)
    bx = fig.add_axes([0.58, 0.50, 0.17, 0.35])
    _strip(bx, [("violent", v["min_dist"], RED), ("Normal", n["min_dist"], GREEN)], "closest distance (m)", "Closest the pair gets")
    cx = fig.add_axes([0.81, 0.50, 0.17, 0.35])
    _strip(cx, [("violent", v["concern"], RED), ("Normal", n["concern"], GREEN)], "concern score (heuristic units)", "Mean concern score")
    tx = fig.add_axes([0.58, 0.09, 0.40, 0.32])
    tx.axis("off")
    lines = ["Can the features tell them apart?"]
    if metrics and metrics.get("evaluated"):
        ms, best = metrics["methods"], metrics.get("best")
        for name in [best, "style", "length"]:
            if name in ms:
                m, ci = ms[name]["metrics"], ms[name].get("ci", {}).get("auc")
                rng_txt = f"  (95% interval {ci[0]:.2f}-{ci[1]:.2f})" if ci and ci[0] == ci[0] else ""
                tag = "best model" if name == best else "baseline: " + ("filming style" if name == "style" else "video length")
                lines.append(f"  {tag}: AUC {m['auc']:.2f}{rng_txt}")
        lines += ["", metrics.get("verdict", "")]
    else:
        lines.append("  (run the train stage to add the classifier result)")
    tx.text(0, 1, "\n".join(lines[:1]), fontsize=15, fontweight="bold", va="top")
    import textwrap
    tx.text(0, 0.9, "\n".join(sum((textwrap.wrap(l, 62, subsequent_indent="    ") or [""] for l in lines[1:]), [])), fontsize=11.5, va="top", color="#222", linespacing=1.45)
    fig.text(0.02, 0.028, "Out-of-fold, grouped cross-validation; the best model is picked on the same folds, so its AUC is optimistic. AUC 0.5 = chance. Normal videos come from a "
             "different source (filming style alone separates them), so differences here are evidence about the data and the method, not a measure of real-world accuracy.",
             fontsize=9.5, color="#555", va="bottom", wrap=True)
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=100, facecolor="white")
    plt.close(fig)
    return out


# ---------------------------------------------------------------- the side-by-side video

PANEL_W, HEAD, TOPBAR, STRIP, FOOT, GAP = 640, 64, 40, 96, 84, 12


def _even(x):
    return int(x) // 2 * 2


def _panel_strip(canvas, y0, view, f):
    d, n, w = view.d, len(view.d), canvas.shape[1]
    for zi, (lo, hi) in enumerate([(0, 0.46), (0.46, 1.2), (1.2, 3.7), (3.7, view.dmax)]):
        ya = int(y0 + STRIP - 6 - (min(hi, view.dmax) / view.dmax) * (STRIP - 12))
        yb = int(y0 + STRIP - 6 - (lo / view.dmax) * (STRIP - 12))
        if ya < yb:
            canvas[ya:yb, :] = ZONE_BGR[zi]
    xs = lambda i: int(6 + (w - 12) * i / max(n - 1, 1))
    pr = None
    for i in range(n):
        if np.isfinite(d[i]):
            pt = (xs(i), int(y0 + STRIP - 6 - min(d[i], view.dmax) / view.dmax * (STRIP - 12)))
            if pr is not None and i - pr[1] <= 3:
                cv2.line(canvas, pr[0], pt, (92, 59, 31), 2)
            pr = (pt, i)
    cv2.line(canvas, (xs(min(f, n - 1)), y0), (xs(min(f, n - 1)), y0 + STRIP), (0, 0, 0), 2)
    cv2.putText(canvas, f"distance between the two (0-{view.dmax:.0f} m)", (6, y0 + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (40, 40, 40), 1, cv2.LINE_AA)


def _encode(raw, out):
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", str(raw), "-c:v", "libx264", "-pix_fmt", "yuv420p", str(out)], check=True)
    Path(raw).unlink(missing_ok=True)


def render_compare_video(cfg, vcase, ncase, out_path, max_seconds=40.0, scores=None, best=None):
    """The violent and the Normal video side by side (30 fps). A video that ends first holds its last frame; the clip stops at the longer one (max_seconds)."""
    scores = scores or {}
    views = [PairView(cfg, vcase), PairView(cfg, ncase)]
    caps = [cv2.VideoCapture(str(v.src)) for v in views]
    sizes = []
    for cap in caps:
        w, h = cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        sizes.append((PANEL_W, _even(h * PANEL_W / w)))
    vh = max(s[1] for s in sizes)
    W, H = 2 * PANEL_W + GAP, HEAD + TOPBAR + vh + STRIP + FOOT
    fps = min(v.fps for v in views)
    nframes = int(min(max_seconds, max(v.res["duration_s"] for v in views)) * fps)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    raw = out_path.with_suffix(".raw.mp4")
    vw = cv2.VideoWriter(str(raw), cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))
    last = [None, None]
    meta = [("VIOLENT", VIOLENT_BGR, vcase), ("NORMAL", NORMAL_BGR, ncase)]
    for f in range(nframes):
        canvas = np.full((H, W, 3), 255, np.uint8)
        canvas[HEAD + TOPBAR:HEAD + TOPBAR + vh] = 0
        canvas[HEAD + TOPBAR + vh + STRIP:] = (40, 40, 40)
        for k, (view, cap, (lab, col, case)) in enumerate(zip(views, caps, meta)):
            ok, img = cap.read()
            if ok:
                last[k] = img
            elif last[k] is None:
                continue
            img = last[k].copy()
            ff = min(f, int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) - 1) if not ok else f
            text, tcol = view.annotate(img, ff)
            img = cv2.resize(img, sizes[k])
            x0 = k * (PANEL_W + GAP)
            canvas[:HEAD, x0:x0 + PANEL_W] = col
            put_text(canvas, f"{lab}: {case['clip']['clip_id']}", (x0 + 10, 42), 0.85, (255, 255, 255), 2)
            canvas[HEAD:HEAD + TOPBAR, x0:x0 + PANEL_W] = tcol
            put_text(canvas, f"{f / fps:.1f}s  {text}"[:PANEL_W // 9], (x0 + 8, HEAD + 27), 0.6, (255, 255, 255), 1)
            oy = HEAD + TOPBAR + (vh - sizes[k][1]) // 2
            canvas[oy:oy + sizes[k][1], x0:x0 + PANEL_W] = img
            sub = canvas[:, x0:x0 + PANEL_W]
            _panel_strip(sub, HEAD + TOPBAR + vh, view, ff)
            fy = HEAD + TOPBAR + vh + STRIP + 4
            rows = dict(facts(view.res, view.d, view.fps))
            put_text(canvas, f"closest {rows['closest distance']} | reach/touch {rows['reaches for / touches']} | follow/approach {rows['follows / approaches']}", (x0 + 6, fy + 18), 0.46, (255, 255, 255), 1)
            put_text(canvas, f"closer than 1.2 m: {rows['time closer than 1.2 m']} | concern mean {rows['concern score, mean']}", (x0 + 6, fy + 38), 0.46, (255, 255, 255), 1)
            ml = _model_line(case, scores, best)
            if ml:
                put_text(canvas, ml[:PANEL_W // 8], (x0 + 6, fy + 58), 0.42, (200, 200, 200), 1)
        vw.write(canvas)
    for cap in caps:
        cap.release()
    vw.release()
    _encode(raw, out_path)
    return out_path


# ---------------------------------------------------------------- driver

def build_showcase(cfg, out_dir, k=3, violent=None, normal=None, max_seconds=40.0, video=True):
    """Write the overview and k snapshot (+ video) pairs into out_dir. -> list of written paths."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    cases = load_cases(cfg)
    if not cases:
        raise RuntimeError("no analysed clips found: run the earlier stages first")
    scores, best = load_scores(cfg["classify"]["report_dir"]) if cfg.get("classify") else ({}, None)
    metrics = None
    mp = Path(cfg["classify"]["report_dir"]) / "metrics.json" if cfg.get("classify") else None
    if mp and mp.exists():
        metrics = json.loads(mp.read_text(encoding="utf-8"))
    written = [render_overview(cohort(cases), metrics, out / "features_overview.png")]
    pairs = pick_examples(cases, k, violent=violent, normal=normal)
    if not pairs:
        print("no violent/Normal pair of videos with a measured interaction to show")
    for i, (v, n) in enumerate(pairs, 1):
        written.append(render_snapshot(cfg, v, n, out / f"snapshot_{i}_{v['clip']['clip_id']}_vs_{n['clip']['clip_id']}.png", scores, best))
        print(f"  snapshot {i}: {v['clip']['clip_id']} vs {n['clip']['clip_id']}", flush=True)
        if video:
            written.append(render_compare_video(cfg, v, n, out / f"compare_{i}_{v['clip']['clip_id']}_vs_{n['clip']['clip_id']}.mp4", max_seconds, scores, best))
            print(f"  video {i} done", flush=True)
    return written
