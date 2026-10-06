"""Classification pipelines: the stages that both tasks share (records, documents, feature blocks, training, report). The tasks differ only in
how the clips are prepared (`src.classtrim.run`, `src.classfull.run`) and in the `task` argument ("trim" | "full").

    records -> documents (geometry text, + captions when Layer B is on) -> blocks (video, text, geometry, baselines) -> cross-validation -> report
"""
import json
from pathlib import Path

import numpy as np

from src.classcommon import edgecases, feat_geom, feat_text, feat_video
from src.classcommon.cv import Data, Method, cross_validate
from src.classcommon.labels import label_of
from src.classcommon.text_build import describe
from src.classcommon.vlm_caption import load_captions
from src.config import resolve_path

STYLE_RAW = ("raw_width", "raw_height", "raw_fps", "raw_duration_s")


def load_records(cfg, task):
    """The clips of this run with their label (`y`) and their analysis result (`result`, None when the pipeline produced none)."""
    cl = cfg["classify"]
    if task == "trim":
        if not cl.get("curated_root"):
            raise ValueError("the trim task needs the finished colab2 folder: pass --curated-root")
        root = Path(cl["curated_root"])
        manifest, stories = root / "curated_manifest_clean.json", root / "stories"
    else:
        manifest, stories = Path(cfg["preprocess"]["clean_manifest_path"]), Path(cfg["curated"]["stories_dir"])
    if not manifest.exists():
        raise FileNotFoundError(f"{manifest} not found: run the earlier stages first (for the trim task: the whole colab2 notebook)")
    recs = []
    for c in json.loads(manifest.read_text(encoding="utf-8"))["clips"]:
        r = dict(c)
        r["y"] = label_of(c["category"])
        p = stories / c["category"] / f"{c['clip_id']}.json"
        r["result"] = json.loads(p.read_text(encoding="utf-8")) if p.exists() else None
        recs.append(r)
    return recs


def build_docs(recs, cfg, task, captions=None):
    """-> (geometry documents, full documents) as {clip_id: str}. The full document adds the captions; a clip without a caption keeps the geometry text."""
    span = float(cfg["classify"]["span_s"])
    geom, full = {}, {}
    for r in recs:
        g = describe(r["result"] or {"interaction": None, "reason": "no_tracks"}, task, span)
        cap = (captions or {}).get(r["clip_id"])
        geom[r["clip_id"]] = g
        full[r["clip_id"]] = g + (f" Scene description: {cap['text']}" if cap and cap.get("text") else "")
    return geom, full


def stage_text(cfg, recs, task):
    """Write one text file per clip (text_dir/<category>/<clip>.json) and return (geometry docs, full docs, captions or None)."""
    cl = cfg["classify"]
    caps = load_captions(recs, cl["caption_dir"]) if cl["use_captions"] else None
    geom, full = build_docs(recs, cfg, task, caps)
    for r in recs:
        p = Path(cl["text_dir"]) / r["category"] / f"{r['clip_id']}.json"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps({"clip_id": r["clip_id"], "category": r["category"], "label": r["y"], "geometry_text": geom[r["clip_id"]],
                                 "caption": (caps or {}).get(r["clip_id"]), "document": full[r["clip_id"]]}, indent=1), encoding="utf-8")
    return geom, full, caps


def load_groups(recs, cfg):
    """Fold groups: clips whose frame hashes are (nearly) equal share a group. Cached in features/groups.json."""
    cl = cfg["classify"]
    p = Path(cl["feat_dir"]) / "groups.json"
    ids = [r["clip_id"] for r in recs]
    if p.exists():
        g = json.loads(p.read_text(encoding="utf-8"))
        if set(g) >= set(ids):
            return np.array([g[i] for i in ids])
    hashes = {}
    for r in recs:
        try:
            hashes[r["clip_id"]] = edgecases.ahash(resolve_path(r["path"]))
        except Exception:
            hashes[r["clip_id"]] = None
    g = edgecases.dup_groups(hashes, int(cl["cv"]["dup_hash_distance"]))
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(g), encoding="utf-8")
    return np.array([g[i] for i in ids])


def build_data(cfg, recs, task, geom_docs, full_docs, embed_fn=None):
    """-> (Data, zero-shot scores {encoder: (n,)}, notes). Blocks are built from the caches only (run the features stage first)."""
    cl = cfg["classify"]
    span, win = float(cl["span_s"]), float(cl["win_s"])
    ids = [r["clip_id"] for r in recs]
    blocks, kinds, zs, notes = {}, {}, {}, []
    for enc in cl["encoders"]:
        X, z = feat_video.video_block(recs, cfg, task, enc)
        if X.shape[1] and np.isfinite(X).any():
            blocks[f"video_{enc}"], kinds[f"video_{enc}"], zs[enc] = X, "num", z
            miss = int((~np.isfinite(X).any(axis=1)).sum())
            if miss:
                notes.append(f"encoder {enc}: {miss} clips without features (imputed)")
        else:
            notes.append(f"encoder {enc}: no features available, left out")
    blocks["geom"], kinds["geom"] = np.vstack([feat_geom.geom_vector(r["result"] or {}, task, span, win) for r in recs]), "num"
    use_caps = bool(cl["use_captions"]) and any(full_docs[i] != geom_docs[i] for i in ids)
    if cl["use_captions"] and not use_caps:
        notes.append("captions are switched on but no clip has one: the full document equals the geometry text")
    variants = [("geom", geom_docs)] + ([("full", full_docs)] if use_caps else [])
    for tag, docs in variants:
        texts = [docs[i] for i in ids]
        blocks[f"text_tfidf_{tag}"], kinds[f"text_tfidf_{tag}"] = np.array(texts, dtype=object), "text"
        blocks[f"text_emb_{tag}"] = feat_text.cached_embeddings(texts, cl["text_model"], Path(cl["feat_dir"]) / f"text_{tag}.npz", embed_fn)
        kinds[f"text_emb_{tag}"] = "num"
    styles = [feat_geom.style_vec(r["result"] or {"raw": {k: r.get(k) for k in STYLE_RAW}}) for r in recs]
    blocks["style"], kinds["style"] = np.vstack(styles), "num"
    blocks["length"], kinds["length"] = np.vstack([feat_geom.length_vec(r["duration_s"]) for r in recs]), "num"
    return Data(blocks, kinds, np.array([r["y"] for r in recs], int), ids), zs, notes


BASELINES = ("style", "length")


def make_methods(data):
    """One method per block (the baselines included) and three fusions of the real modalities (not the baselines)."""
    methods = [Method(b, "single", [b]) for b in data.blocks]
    text = "text_emb_full" if "text_emb_full" in data.blocks else "text_emb_geom"
    fused = [b for b in data.blocks if b.startswith("video_")] + [text, "geom"]
    if len(fused) >= 2:
        methods += [Method("fusion_late", "late", fused), Method("fusion_stack", "stack", fused), Method("fusion_early", "early", [b for b in fused if data.kinds[b] == "num"])]
    return methods


def run_cv(cfg, data, recs):
    """-> (ok, message, results {method: {...}} or None)."""
    cv = cfg["classify"]["cv"]
    ok, msg = edgecases.check_enough(data.y, int(cv["min_per_class"]))
    if not ok:
        return False, msg, None
    return True, msg, cross_validate(make_methods(data), data, load_groups(recs, cfg), cv)
