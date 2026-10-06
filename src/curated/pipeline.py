"""Curated dataset: per-clip analysis glue. Runs AFTER the existing stage CLIs (shots, track_poses, context features, scene, story) have been run on the
trimmed clips; reads their outputs and turns them into the pre-violence understanding of one clip (result dict + files).

    result = analyze_clip(clip, cfg)         -> dict (also saved as <stories_dir>/<category>/<clip>.json and .md)
"""
import json
from pathlib import Path

import numpy as np

from src.assm.track_poses import load_tracks
from src.config import resolve_path
from src.context.features import load_context, rank_pairs
from src.context.graph import load_story
from src.curated.interaction import summarize
from src.curated.relation import THRESH
from src.video.shots import local_id, shot_of_id

NO_PAIR_REASONS = {
    "no_people": "no person was detected",
    "one_person": "only one person was tracked reliably, so there is no interaction to measure",
    "never_together": "two or more people were seen, but never tracked together long enough (or never within 12 m) to measure an interaction",
}


def pick_key_pair(scene, arrays, story=None):
    """The pair to explain: the story's key pair (most evidence), else the best-ranked pair; None when there is no pair."""
    cand = [story["key_pair"]] if story and story.get("key_pair") else []
    cand += rank_pairs(scene, 3)
    for k in cand:
        ij = tuple(int(x) for x in k.split("_"))
        if ij in arrays:
            return ij
    return None


def people_facts(t, scene):
    ids = sorted(set(t["track_id"].tolist()))
    fps = float(t["fps"])
    per = {i: round(float((t["track_id"] == i).sum()) / fps, 2) for i in ids}
    return {"tracks": len(ids), "tracks_used": scene["n_tracks_used"], "max_people": scene["max_people"], "seconds_visible": {str(local_id(i)): s for i, s in per.items()},
            "camera_moving": bool(scene["camera"].get("moving")), "night": bool(scene["camera"].get("night"))}


def no_pair_reason(facts):
    if facts["max_people"] == 0:
        return "no_people"
    if facts["tracks_used"] < 2:
        return "one_person"
    return "never_together"


def analyze_clip(clip, cfg):
    """clip: entry of the curated clean manifest. Needs tracks + context (+ story) of the trimmed clip."""
    cur = cfg["curated"]
    cat, cid = clip["category"], clip["clip_id"]
    ctx_dir = resolve_path(cfg["context"]["context_dir"])
    res = {"clip_id": cid, "category": cat, "start_s": clip.get("start_s"), "status": clip.get("status", "ok"), "duration_s": clip["duration_s"],
           "is_normal": clip.get("start_s") is None, "thresholds": THRESH}
    tp = resolve_path(cfg["assm"]["tracks_dir"]) / cat / f"{cid}.npz"
    if not tp.exists() or not (ctx_dir / cat / f"{cid}_scene.json").exists():
        res.update({"interaction": None, "no_pair": True, "reason": "no_tracks", "lines": ["The pipeline produced no tracks for this clip."]})
        return _save(res, cur)
    t = load_tracks(tp)
    scene, arrays = load_context(ctx_dir, cat, cid)
    story = load_story(ctx_dir, cat, cid)
    res["people"] = people_facts(t, scene)
    res["shots"] = int(len(t["shot_starts"])) if "shot_starts" in t else 1
    res["raw"] = {k: clip.get(k) for k in ("raw_width", "raw_height", "raw_fps", "raw_duration_s")}
    res["camera"] = scene["camera"]
    if clip.get("status") == "short":
        res["note"] = f"only {clip['duration_s']:.1f} s of footage before the violence: too little to describe a lead-up"
    key = pick_key_pair(scene, arrays, story)
    if key is None:
        r = no_pair_reason(res["people"])
        res.update({"interaction": None, "no_pair": True, "reason": r, "lines": [f"No interacting pair identified: {NO_PAIR_REASONS[r]}."]})
        return _save(res, cur)
    a = arrays[key]
    fps = scene["fps"]
    inter = summarize(a, fps, key, tuple(cur["last_seconds"]), float(cur["window_s"]), float(cur["step_s"]))
    res.update({"interaction": inter, "no_pair": False, "key_pair": f"{key[0]}_{key[1]}", "pair_ids_local": [local_id(key[0]), local_id(key[1])], "lines": inter["lines"],
                "story_cues": (story or {}).get("preds_seen", []), "shot": shot_of_id(key[0])})
    return _save(res, cur)


def _save(res, cur):
    d = Path(cur["stories_dir"]) / res["category"]
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{res['clip_id']}.json").write_text(json.dumps(res, indent=1, default=float), encoding="utf-8")
    (d / f"{res['clip_id']}.md").write_text(markdown(res), encoding="utf-8")
    return res


def markdown(res):
    head = f"# {res['clip_id']} ({res['category']})\n\n"
    head += (f"Violence starts at {res['start_s']:.1f} s; this describes the {res['duration_s']:.1f} s before it.\n\n" if res.get("start_s") is not None
             else f"Normal clip (no violence); this describes the first {res['duration_s']:.1f} s.\n\n")
    if res.get("note"):
        head += f"_{res['note']}_\n\n"
    body = "\n".join(f"- {l}" for l in res["lines"])
    p = res.get("people")
    foot = ""
    if p:
        foot = f"\n\nPeople tracked: {p['tracks']} (used {p['tracks_used']}), most at once {p['max_people']}; camera {'moving' if p['camera_moving'] else 'static'}" + (", low light" if p["night"] else "") + ".\n"
    return head + body + foot + "\nMeters are approximate (1.7 m height prior, assumed field of view); the lead-up type is a rule-based heuristic.\n"


def load_result(cfg, category, clip_id):
    p = Path(cfg["curated"]["stories_dir"]) / category / f"{clip_id}.json"
    return json.loads(p.read_text("utf-8")) if p.exists() else None
