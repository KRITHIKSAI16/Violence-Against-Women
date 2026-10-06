"""Classification pipelines: the stage functions that need a GPU, a model download or the whole evaluation. Called by src.classtrim.run and src.classfull.run."""
import subprocess
import sys
import time

from src.classcommon import edgecases, feat_video
from src.classcommon.pipeline import build_data, load_records, run_cv, stage_text
from src.classcommon.report import write_not_evaluated, write_report
from src.classcommon.vlm_caption import ACT_RE, CATEGORY_RE
from src.config import REPO_ROOT


def sh(*args):
    """Run an existing stage CLI (python -m ...) from the repo root; stops the run when it fails."""
    t0 = time.time()
    r = subprocess.run([sys.executable, "-m", *args], cwd=str(REPO_ROOT))
    if r.returncode != 0:
        raise RuntimeError(f"stage failed: python -m {' '.join(args)}")
    print(f"  [{time.time() - t0:.0f} s]", flush=True)


def stage_features(cfg, recs, task):
    """Encode the clips that have no cached video windows yet. Encoders are loaded only when some clip needs one (a trim run that finds everything in colab2 loads none)."""
    cl = cfg["classify"]
    todo = {e: feat_video.missing(recs, cfg, task, e) for e in cl["encoders"]}
    todo = {e: m for e, m in todo.items() if m}
    for e in cl["encoders"]:
        print(f"{e}: {len(recs) - len(todo.get(e, []))} clips ready, {len(todo.get(e, []))} to encode", flush=True)
    failures = []
    if todo:
        from src.curated.ml_encoders import load_encoders
        enc, failed = load_encoders(tuple(todo), device="auto")
        print("encoders loaded:", list(enc) or "none", "| not available:", failed or "none", flush=True)
        for name, clips in todo.items():
            if name in enc:
                failures += feat_video.encode_all(clips, cfg, task, [name], enc)
    for f in failures[:10]:
        print("  FAILED", f)
    return failures


def stage_train(cfg, task, embed_fn=None):
    """Documents -> feature blocks -> cross-validation -> report. Returns the path of report.md."""
    cl = cfg["classify"]
    recs = load_records(cfg, task)
    geom, full, caps = stage_text(cfg, recs, task)
    data, zs, notes = build_data(cfg, recs, task, geom, full, embed_fn)
    docs = full if "text_emb_full" in data.blocks else geom
    ok, msg, cvres = run_cv(cfg, data, recs)
    if not ok:
        print(msg)
        return write_not_evaluated(cl["report_dir"], task, msg, recs, docs)
    flags = edgecases.flag_records(recs)
    for tag in ("short", "no_pair", "no_result"):
        n = sum(tag in f for f in flags.values())
        if n:
            notes.append(f"{n} clips flagged '{tag}'")
    hit = edgecases.leak_scan(docs, CATEGORY_RE)
    notes.append(f"label-name check: {len(hit)} documents contain a category name" + (f" ({', '.join(sorted(hit)[:5])})" if hit else ""))
    if task == "trim":
        act = edgecases.leak_scan(docs, ACT_RE)
        notes.append(f"act-word check (trim): {len(act)} documents contain a word naming the violent act" + (f" ({', '.join(sorted(act)[:5])}): the cut may be late for these" if act else ""))
    if caps:
        notes.append(f"captions: {sum(1 for c in caps.values() if c)} of {len(recs)} clips captioned, {sum(c['removed'] for c in caps.values() if c)} sentences removed by the cleaner")
    groups_res = [f"{r.get('raw_width')}x{r.get('raw_height')}" for r in recs]
    return write_report(cl["report_dir"], task, cfg, recs, data, cvres, zs, flags, groups_res, docs, notes, msg)
