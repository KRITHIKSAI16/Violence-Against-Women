"""Curated dataset: the whole run in one place. Stages are resumable (finished clips are skipped) and can be run one by one or all at once.

    python -m src.curated.run --stage all --out-root /content/drive/MyDrive/VAW_results/curated_previolence
    stages: prepare (read start times, match videos, cut clips at T) -> perception (pick the detector by measured pair coverage) -> track -> context
            (camera, distances, scene, story) -> analyze (per-clip understanding, figures, videos) -> ml (video models) -> report (evaluation + jury page)

Everything is written under --out-root; nothing outside it (and nothing from earlier runs) is touched.
"""
import argparse
import json
import logging
import shutil
import subprocess
import sys
import time
import zipfile
from pathlib import Path

import numpy as np
import yaml

from src.config import REPO_ROOT, resolve_path
from src.curated import evaluate as ev_mod
from src.curated.config import build_config, choose_perception
from src.curated.figures import concern_figure, distance_figure
from src.curated.index import check_times, find_videos, match, normal_videos, read_annotations
from src.curated.locate import locate
from src.curated.pipeline import analyze_clip
from src.curated.prepare import build_curated_manifest
from src.curated.report import build_html, write_csv
from src.curated.video import render_curated_clip

log = logging.getLogger(__name__)
STAGES = ["prepare", "perception", "track", "context", "analyze", "ml", "report"]


def sh(*args):
    """Run an existing stage CLI (python -m ...) from the repo root; stops the run if it fails."""
    t0 = time.time()
    r = subprocess.run([sys.executable, "-m", *args], cwd=str(REPO_ROOT))
    if r.returncode != 0:
        raise RuntimeError(f"stage failed: python -m {' '.join(args)}")
    print(f"  [{time.time() - t0:.0f} s]", flush=True)


def cfg_file(out_root):
    return Path(out_root) / "curated_config.yaml"


def load_cfg(out_root):
    return yaml.safe_load(cfg_file(out_root).read_text(encoding="utf-8"))


def save_cfg(cfg, out_root):
    cfg_file(out_root).write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True), encoding="utf-8")


def manifest(cfg):
    return json.loads(Path(cfg["preprocess"]["clean_manifest_path"]).read_text(encoding="utf-8"))["clips"]


# ------------------------------------------------------------------ stages
def stage_prepare(args):
    from src.data.manifest import probe_video
    out_root = Path(args.out_root)
    data_root, ann_path, msgs = locate(args.drive_root, out_root.parent, args.data_root, args.annotations)
    print("\n".join(msgs))
    ann = read_annotations(ann_path)
    videos = find_videos(data_root)
    matched, unmatched, unlabeled, notes = match(ann, videos)
    print(f"\nstart times read: {len(ann)} | videos found: {len(videos)} | matched: {len(matched)} | unmatched rows: {len(unmatched)} | videos without a start time: {len(unlabeled)}")
    for u in unmatched:
        print(f"  NOT MATCHED  {u['name']!r} (sheet {u['sheet']}): {u['reason']}")
    for n in notes:
        print(f"  note: {n}")
    if unlabeled:
        print("  videos in the folders with no row in the sheet (not used): " + ", ".join(v["stem"] for v in unlabeled[:30]) + (" ..." if len(unlabeled) > 30 else ""))
    if not matched:
        raise RuntimeError("no start time could be matched to a video: check the names printed above")
    durations = {}
    for m in matched:
        try:
            durations[m["clip_id"]] = probe_video(m["path"])["duration_s"]
        except Exception as e:
            print(f"  unreadable video {m['path']}: {e}")
    matched = check_times([m for m in matched if m["clip_id"] in durations], durations)
    for m in matched:
        if m["status"] != "ok":
            print(f"  {m['clip_id']}: {m['status']} (T = {m['start_s']:.1f} s, clip {m['duration_s']:.1f} s)")
    cfg, _ = build_config(out_root)
    clips = matched + normal_videos(videos, args.n_normal or cfg["curated"]["n_normal"])
    print(f"\nclips to process: {len(matched)} violent + {len(clips) - len(matched)} Normal control")
    if args.max_clips:
        clips = clips[:args.max_clips]
    build_curated_manifest(clips, cfg)


def stage_perception(args):
    out_root = Path(args.out_root)
    cfg = load_cfg(out_root)
    if args.skip_perception:
        print("perception check skipped: keeping", cfg["assm"]["model"], cfg["assm"]["imgsz"])
        return
    sh("src.perception.shootout", "--config", str(cfg_file(out_root)), "--sample", "3", "--sheet")
    res = json.loads((Path(cfg["perception"]["shootout_dir"]) / "shootout_results.json").read_text(encoding="utf-8"))
    name, why = choose_perception(res["summary"], cfg)
    save_cfg(cfg, out_root)
    (out_root / "perception_choice.json").write_text(json.dumps({"chosen": name, "reason": why, "summary": res["summary"]}, indent=1), encoding="utf-8")
    print(f"\nperception chosen: {name} ({why})")


def stage_track(args):
    sh("src.video.shots", "--config", str(cfg_file(args.out_root)))
    sh("src.assm.track_poses", "--config", str(cfg_file(args.out_root)))


def stage_context(args):
    c = str(cfg_file(args.out_root))
    sh("src.context.features", "--config", c)
    sh("src.context.scene", "--config", c)
    sh("src.context.story", "--config", c)


def stage_analyze(args):
    from src.context.features import load_context
    out_root = Path(args.out_root)
    cfg = load_cfg(out_root)
    cur = cfg["curated"]
    results, assets = [], {}
    clips = manifest(cfg)
    for n, clip in enumerate(clips, 1):
        r = analyze_clip(clip, cfg)
        a = {}
        try:
            if r.get("interaction"):
                scene, arrays = load_context(cfg["context"]["context_dir"], clip["category"], clip["clip_id"])
                key = tuple(r["interaction"]["ids"])
                fig = distance_figure(r, arrays[key]["dist_m"], scene["fps"], Path(cur["figures_dir"]) / clip["category"] / f"{clip['clip_id']}_distance.png")
                a["figure"] = str(fig) if fig else None
            out = render_curated_clip(clip, cfg, r)
            a["video"], a["storyboard"] = (str(out["video"]) if out["video"] else None), (str(out["storyboard"]) if out["storyboard"] else None)
        except Exception as e:                                                   # one failing render must not stop the run
            r["render_error"] = str(e)
            print(f"  render failed for {clip['clip_id']}: {e}")
        results.append(r)
        assets[clip["clip_id"]] = a
        print(f"  analyzed {n}/{len(clips)}: {clip['clip_id']} -> {r['interaction']['leadup']['type'] if r.get('interaction') else 'no pair'}", flush=True)
    (out_root / "results.json").write_text(json.dumps(results, indent=1, default=float), encoding="utf-8")
    (out_root / "assets.json").write_text(json.dumps(assets, indent=1), encoding="utf-8")


def stage_ml(args):
    from src.curated import ml_head
    from src.curated.ml_encoders import encode_clip, load_encoders
    out_root = Path(args.out_root)
    cfg = load_cfg(out_root)
    cur = cfg["curated"]
    clips = manifest(cfg)
    enc, failed = load_encoders(tuple(args.encoders), device="auto")
    print("encoders loaded:", list(enc) or "none", "| not available:", failed or "none")
    win, step = float(cur["window_s"]), float(cur["step_s"])
    for name, e in enc.items():
        todo = [c for c in clips if not ml_head.emb_path(cur["ml_dir"], name, c["category"], c["clip_id"]).exists()]
        print(f"{name}: {len(todo)} clips to encode")
        for k, c in enumerate(todo, 1):
            ts = ml_head.window_starts(c["duration_s"], win, step if c.get("start_s") is not None else max(step, 1.0))
            try:
                ml_head.save_clip_encoding(cur["ml_dir"], name, c["category"], c["clip_id"], ts, encode_clip(e, resolve_path(c["path"]), ts, win))
            except Exception as ex:
                print(f"  {name} failed on {c['clip_id']}: {ex}")
            if k % 5 == 0 or k == len(todo):
                print(f"  {name}: {k}/{len(todo)} clips", flush=True)
    rows, trends, per_clip = {}, {}, {}
    dur = {c["clip_id"]: c["duration_s"] for c in clips}
    for name in list(enc) or ["xclip", "internvideo2", "vjepa2"]:
        t = ml_head.build_table(clips, cur["ml_dir"], name, win)
        if not len(t["label"]) or len(set(t["label"])) < 2:
            continue
        n_by = {lab: len(set(t["clip"][t["label"] == lab])) for lab in (0, 1)}
        if min(n_by.values()) >= 3:
            p = ml_head.leave_one_clip_out(t)
            rows[f"learned head on {name} embeddings"] = ml_head.summarize_scores(ml_head.clip_scores(t, p, dur, win_s=win))
            trends[f"learned head on {name} embeddings"] = ml_head.trend_from_windows(t, p, dur, win_s=win)
        if np.isfinite(t["margin"]).any():
            m = t["margin"]
            rows[f"zero-shot prompt margin ({name})"] = ml_head.summarize_scores(ml_head.clip_scores(t, m, dur, win_s=win))
            trends[f"zero-shot prompt margin ({name})"] = ml_head.trend_from_windows(t, m, dur, win_s=win)
            for cid in np.unique(t["clip"]):
                mm = t["clip"] == cid
                late = mm & (t["t_start"] + win >= dur[cid] - 3.0 + 1e-6)
                per_clip.setdefault(cid, {"encoder": name, "last": float(np.nanmean(m[late])) if late.any() else float(np.nanmean(m[mm])), "max": float(np.nanmax(m[mm]))})
    (out_root / "ml_summary.json").write_text(json.dumps({"loaded": list(enc), "failed": failed, "rows": rows, "trends": trends, "per_clip": per_clip}, indent=1, default=float), encoding="utf-8")
    print("\n".join(ml_head.ablation_table(rows)) if rows else "no ML scores (no encoder loaded or too few clips)")


def stage_report(args):
    out_root = Path(args.out_root)
    cfg = load_cfg(out_root)
    cur = cfg["curated"]
    results = json.loads((out_root / "results.json").read_text(encoding="utf-8"))
    assets = json.loads((out_root / "assets.json").read_text(encoding="utf-8"))
    ev = ev_mod.evaluate_all(results)
    nv = ev["normal_vs_pre"]
    (out_root / "evaluation.md").write_text(ev_mod.markdown(ev), encoding="utf-8")
    fig = concern_figure(nv["violent_scores"], nv["normal_scores"], Path(cur["figures_dir"]) / "concern_vs_normal.png")
    info = {"concern_figure": str(fig), "hig": "HIG-structured (vocabulary and layers), not the pretrained HIG model"}
    ch = out_root / "perception_choice.json"
    info["perception"] = (f"{json.loads(ch.read_text())['chosen']} ({json.loads(ch.read_text())['reason']})" if ch.exists() else f"{cfg['assm']['model']} at {cfg['assm']['imgsz']} px")
    mls = out_root / "ml_summary.json"
    if mls.exists():
        ml = json.loads(mls.read_text(encoding="utf-8"))
        from src.curated import ml_head
        rows = {"geometric concern score (rules)": {"violent_n": nv["violent_n"], "normal_n": nv["normal_n"], "auc": nv["auc"], "auc_low": nv["auc_low"], "auc_high": nv["auc_high"]}}
        rows.update(ml["rows"])
        info["ml_table"] = ml_head.ablation_table(rows) + ["", f"best single filming-style feature alone: AUC {nv['style_only_auc']:.2f}"] + \
            [f"within-clip, last 3 s vs earlier, {k}: higher in {v['higher_in_last']} of {v['clips']} clips (sign-test p = {v['p_sign']:.3f})" for k, v in ml["trends"].items()]
        info["ml_note"] = "Video-model scores are out-of-fold (a clip is never scored by a model that saw it). A score that does not beat the style-only line is not evidence about behavior."
        info["encoders"] = ", ".join(ml["loaded"]) + (f" (not available: {', '.join(ml['failed'])})" if ml["failed"] else "")
        for cid, v in ml["per_clip"].items():
            assets.setdefault(cid, {})["ml"] = v
    for a in assets.values():
        for k in ("video", "figure", "storyboard"):
            if a.get(k) and not Path(a[k]).exists():
                a[k] = None
    page = build_html(results, ev, assets, info, Path(cur["report_dir"]) / "index.html")
    write_csv(results, out_root / "summary.csv")
    zpath = out_root / "curated_previolence_bundle.zip"
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
        for sub in ("report", "videos", "figures", "stories"):
            for f in (out_root / sub).rglob("*"):
                if f.is_file() and "_pair" not in f.parts and "_plain" not in f.parts:
                    z.write(f, f.relative_to(out_root))
        for f in ("summary.csv", "evaluation.md"):
            z.write(out_root / f, f)
    print(f"\nJury page: {page}\nSummary: {out_root / 'summary.csv'}\nEvaluation: {out_root / 'evaluation.md'}\nBundle: {zpath}")
    print("\n" + ev_mod.markdown(ev))


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    ap = argparse.ArgumentParser(description="Pre-violence understanding on the hand-picked dataset")
    ap.add_argument("--stage", default="all", choices=["all"] + STAGES)
    ap.add_argument("--out-root", default="/content/drive/MyDrive/VAW_results/curated_previolence")
    ap.add_argument("--drive-root", default="/content/drive/MyDrive")
    ap.add_argument("--data-root", default=None, help="the folder with the category folders (default: find 'violence' in Drive)")
    ap.add_argument("--annotations", default=None, help="the Excel / CSV with video name and start time")
    ap.add_argument("--n-normal", type=int, default=None)
    ap.add_argument("--max-clips", type=int, default=None, help="for a quick trial run: process only the first N clips")
    ap.add_argument("--skip-perception", action="store_true")
    ap.add_argument("--encoders", nargs="*", default=["internvideo2", "xclip", "vjepa2"])
    args = ap.parse_args()
    todo = STAGES if args.stage == "all" else [args.stage]
    fn = {"prepare": stage_prepare, "perception": stage_perception, "track": stage_track, "context": stage_context, "analyze": stage_analyze, "ml": stage_ml, "report": stage_report}
    for s in todo:
        print(f"\n=== stage: {s} ===", flush=True)
        fn[s](args)


if __name__ == "__main__":
    main()
