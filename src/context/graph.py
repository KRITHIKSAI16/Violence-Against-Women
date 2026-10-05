"""Deep context, Stage C: the interaction scene graph and the buildup story of a clip (no prediction, no language model).

Nodes are people (and, from the scene layout, doors / obstacles). Edges are PREDICATES between two people over time, called EPISODES:
actor -> predicate -> target, with start / end, a confidence, and details. This is the report's interaction graph (Phase II, M10) in a
Phase-1 form built only from geometry, body pose and the scene layout. Predicates:

  approaches_from_behind / approaches_side / approaches_frontal   one person moves toward the other (closing speed, direction of the target's facing)
  follows               lagged-path following (the follower retraces the leader's path)
  looks_back            head turned toward the other person while the body faces away (the hidden-follower cue from the literature)
  hovers_near           stays within `hover_dist_m` of a person who stands still
  very_close            within arm's reach for a while
  mutual_facing         both face each other at conversational distance           (BENIGN cue)
  blocks_exit           one person stands between the other and a detected door  (needs a reliable layout)
  pinned_against        one person is against a wall / vehicle (obstacle at the same depth) with the other in front (needs depth)
  reaches_for           a hand moves fast toward the other person's torso        (ACT cue)
  contact               hand-to-body distance below `contact_wrist_m`            (ACT cue)
  flees_from            moves away fast                                          (ACT cue)
  target_stops / target_speeds_up   the target reacts to an approach / following: stops walking, or speeds up

Concern and benign cues carry weights (`story.cue_weights`). The resulting cue curve is a transparent HEURISTIC for visual emphasis and
ranking inside the report; it is NOT a validated risk score and the weights are untuned defaults (the annotated benchmark is the way to tune them).
ACT cues (reach, contact, flee) mark the start of the physical act, so the buildup video ends before the first of them.
"""
import json
import logging
from pathlib import Path

import numpy as np

from src.assm.gate import runs
from src.config import load_config, resolve_path
from src.video.shots import SHOT_BASE

log = logging.getLogger(__name__)

ACT_PREDS = ("reaches_for", "contact", "flees_from")
BENIGN_PREDS = ("mutual_facing",)
PHASE = {"approaches_from_behind": 1, "approaches_side": 1, "approaches_frontal": 1, "follows": 2, "looks_back": 2, "hovers_near": 2,
         "very_close": 2, "blocks_exit": 3, "pinned_against": 3, "reaches_for": 4, "contact": 4, "flees_from": 4}
PHASE_NAMES = {1: "approach", 2: "follow / linger", 3: "block / pin", 4: "physical act"}


def _runs(mask, fps, min_s, gap_s=0.4):
    m = np.nan_to_num(np.asarray(mask, float), nan=0.0).astype(bool)
    return runs(m, max(1, round(min_s * fps)), max(0, round(gap_s * fps)))


def _ep(pred, actor, target, s, e, fps, a, cam_moving, **detail):
    gc = float(np.nanmean(a["ground_conf"][s:e + 1])) if np.isfinite(a["ground_conf"][s:e + 1]).any() else 0.5
    conf = round(min(1.0, gc) * (0.7 if cam_moving else 1.0), 2)
    return {"pred": pred, "actor": int(actor), "target": int(target), "start_f": int(s), "end_f": int(e), "start_s": round(s / fps, 2),
            "end_s": round((e + 1) / fps, 2), "conf": conf, "detail": {k: (round(float(v), 2) if isinstance(v, (float, np.floating)) else v) for k, v in detail.items()}}


def _nm(x):
    x = np.asarray(x, float)
    return float(np.nanmean(x)) if np.isfinite(x).any() else float("nan")


def pair_episodes(i, j, a, fps, ctx, story, geo=None, cam_moving=False):
    """All episodes of one pair from its feature arrays `a` (see features.pair_arrays). geo: optional scene geometry (doors, feet, depth rows)."""
    n = len(a["dist_m"])
    d, cl, zone = a["dist_m"], a["closing_ms"], a["zone"]
    eps = []

    # --- approach (each direction) + the target's reaction
    for actor, target, toward, behind, tface in ((i, j, a["toward_i"], a["approach_behind_i_j"], a["ang_j_to_i"]),
                                                  (j, i, a["toward_j"], a["approach_behind_j_i"], a["ang_i_to_j"])):
        mask = (toward > story["approach_toward_ms"]) & (cl > story["approach_closing_ms"]) & (d < 8.0)
        for s, e in _runs(mask, fps, story["approach_min_s"]):
            fb = float(np.mean(behind[s:e + 1]))
            ff = float(np.nanmean(tface[s:e + 1] < ctx["facing_toward_deg"])) if np.isfinite(tface[s:e + 1]).any() else 0.0
            pred = "approaches_from_behind" if fb >= 0.4 else "approaches_frontal" if ff >= 0.5 else "approaches_side"
            eps.append(_ep(pred, actor, target, s, e, fps, a, cam_moving, dist_start=d[s], dist_end=d[e]))
            eps.extend(_reaction(actor, target, s, e, fps, a, cam_moving, story, a["speed_j"] if target == j else a["speed_i"]))

    # --- following (each direction) + reaction
    for actor, target, key, lagk in ((i, j, "follow_i_j", "lag_i_j"), (j, i, "follow_j_i", "lag_j_i")):
        for s, e in _runs(a[key], fps, ctx["follow_min_s"], 1.0):
            eps.append(_ep("follows", actor, target, s, e, fps, a, cam_moving, lag_s=float(np.nanmedian(a[lagk][s:e + 1])), mean_dist=_nm(d[s:e + 1])))
            eps.extend(_reaction(actor, target, s, e, fps, a, cam_moving, story, a["speed_j"] if target == j else a["speed_i"]))

    # --- looking back (the head turned toward the other person while the body faces away)
    for actor, target, key in ((i, j, "looking_back_i"), (j, i, "looking_back_j")):
        for s, e in _runs(a[key], fps, 0.3, 1.5):
            eps.append(_ep("looks_back", actor, target, s, e, fps, a, cam_moving, dist=_nm(d[s:e + 1])))

    # --- lingering near a person who stands still
    one_still = (a["still_i"] ^ a["still_j"]) & ~a["mutual_facing"]
    for s, e in _runs((d < story["hover_dist_m"]) & one_still, fps, story["hover_min_s"], 0.6):
        mover, still = (i, j) if _nm(a["speed_i"][s:e + 1]) > _nm(a["speed_j"][s:e + 1]) else (j, i)
        eps.append(_ep("hovers_near", mover, still, s, e, fps, a, cam_moving, mean_dist=_nm(d[s:e + 1])))
    for s, e in _runs(zone == 0, fps, story["very_close_min_s"], 0.4):
        eps.append(_ep("very_close", i, j, s, e, fps, a, cam_moving, mean_dist=_nm(d[s:e + 1])))

    # --- benign: facing each other at conversational distance
    for s, e in _runs(a["mutual_facing"], fps, story["mutual_min_s"], 0.6):
        eps.append(_ep("mutual_facing", i, j, s, e, fps, a, cam_moving, mean_dist=_nm(d[s:e + 1])))

    # --- scene geometry: a person between the other and a door; against an obstacle (needs layout / depth)
    if geo:
        eps.extend(_geometry_episodes(i, j, a, fps, story, geo, cam_moving))

    # --- act cues, confirmed against depth when available (image overlap at different depths is not closeness)
    for actor, target, key in ((i, j, "reach_i_to_j_ms"), (j, i, "reach_j_to_i_ms")):
        for s, e in _runs(np.nan_to_num(a[key], nan=0) > story["reach_ms"], fps, 0.06, 0.2):
            eps.append(_confirm(_ep("reaches_for", actor, target, s, e, fps, a, cam_moving, peak_ms=np.nanmax(a[key][s:e + 1])), geo))
    for s, e in _runs(a["contact"], fps, 0.2, 0.2):
        eps.append(_confirm(_ep("contact", i, j, s, e, fps, a, cam_moving, wrist_torso_m=np.nanmin(a["wrist_torso_m"][s:e + 1])), geo))
    for actor, target, key, sk in ((i, j, "flee_i", "speed_i"), (j, i, "flee_j", "speed_j")):
        for s, e in _runs(a[key], fps, 0.5, 0.3):
            eps.append(_ep("flees_from", actor, target, s, e, fps, a, cam_moving, speed_ms=np.nanmax(a[sk][s:e + 1])))
    eps.sort(key=lambda x: (x["start_f"], x["pred"]))
    return eps


def _reaction(actor, target, s, e, fps, a, cam_moving, story, tspeed):
    """Does the TARGET stop or speed up right after the approach / following starts?"""
    b0, b1 = max(0, s - round(story["reaction_before_s"] * fps)), s
    a0, a1 = s + round(0.5 * fps), min(len(tspeed), s + round(story["reaction_after_s"] * fps))
    before, after = _nm(tspeed[b0:b1 + 1]), _nm(tspeed[a0:a1 + 1])
    out = []
    if np.isfinite(before) and np.isfinite(after) and a1 - a0 >= round(0.8 * fps):
        if before >= 0.5 and after <= 0.2:
            out.append(_ep("target_stops", actor, target, a0, a1 - 1, fps, a, cam_moving, speed_before=before, speed_after=after))
        elif after >= max(1.2, 1.5 * before) and before >= 0.3:
            out.append(_ep("target_speeds_up", actor, target, a0, a1 - 1, fps, a, cam_moving, speed_before=before, speed_after=after))
    return out


def _confirm(ep, geo):
    """Mark an act cue as depth-confirmed / unconfirmed using the nearest depth sample of the pair (if any)."""
    rows = (geo or {}).get("depth_rows") or []
    if not rows:
        ep["detail"]["depth"] = "not checked"
        return ep
    mid = (ep["start_f"] + ep["end_f"]) / 2
    r = min(rows, key=lambda r: abs(r["frame"] - mid))
    if abs(r["frame"] - mid) > (geo.get("fps", 30) * 2.0):
        ep["detail"]["depth"] = "not checked"
    elif r["depth_gap"] <= 0.20:
        ep["detail"]["depth"] = "same depth"
    else:
        ep["detail"]["depth"] = "different depth"
        ep["conf"] = round(ep["conf"] * 0.5, 2)
    return ep


def _geometry_episodes(i, j, a, fps, story, geo, cam_moving):
    from src.context.scene import blocks_door   # local import: scene.py imports features, avoid a cycle at module load
    out = []
    n = len(a["dist_m"])
    doors, feet, hpx = geo.get("doors") or [], geo.get("feet") or {}, geo.get("h") or {}
    if doors and geo.get("layout_reliable") and i in feet and j in feet:
        W, H = geo["size"]
        for door in doors[:2]:
            base = (door["cx"] * W, door["base_y"] * H)
            for blocker, blocked in ((j, i), (i, j)):
                m = np.zeros(n, bool)
                for t in np.where(a["dist_m"] < story["block_dist_m"])[0]:
                    fb, fk = feet[blocker][t], feet[blocked][t]
                    if np.isfinite(fb).all() and np.isfinite(fk).all():
                        m[t] = blocks_door(fk, fb, base, hpx[blocked][t])
                for s, e in _runs(m, fps, story["block_min_s"], 0.4):
                    out.append(_ep("blocks_exit", blocker, blocked, s, e, fps, a, cam_moving, door_cx=door["cx"]))
    rows = geo.get("depth_rows") or []
    if not geo.get("layout_reliable"):
        return out                      # the 'obstacle' map is only meaningful when the segmentation is trustworthy (confident, still camera)
    for who, other, key in ((i, j, "pinned_i"), (j, i, "pinned_j")):
        # a depth sample counts only if the other person is actually close to this one at that moment
        flags = [bool(r.get(key, False)) and r["frame"] < n and np.isfinite(a["dist_m"][r["frame"]]) and a["dist_m"][r["frame"]] < story["pinned_dist_m"]
                 for r in rows]
        k = 0
        while k < len(rows):
            if flags[k]:
                m = k
                while m + 1 < len(rows) and flags[m + 1]:
                    m += 1
                if m > k:    # at least two consecutive depth samples
                    out.append(_ep("pinned_against", other, who, rows[k]["frame"], rows[m]["frame"], fps, a, cam_moving, samples=m - k + 1))
                k = m + 1
            else:
                k += 1
    return out


# ---------------------------------------------------------------- pair / clip assembly
def episode_mask(ep, n):
    m = np.zeros(n, bool)
    m[ep["start_f"]:ep["end_f"] + 1] = True
    return m


def summarize_pair(eps, n, fps, weights):
    """Seconds with any concern / benign cue, the weighted cue curve (2 per second), phases and whether they escalate in order."""
    concern = np.zeros(n, bool)
    benign = np.zeros(n, bool)
    curve = np.zeros(n)
    for ep in eps:
        w = weights.get(ep["pred"], 0.0) * min(1.0, 0.5 + ep["conf"] / 2)        # low-confidence episodes count a little less
        m = episode_mask(ep, n)
        if ep["pred"] in BENIGN_PREDS or w < 0:
            benign |= m
        elif w > 0:
            concern |= m
        curve[m] += w
    first = {}
    for ep in eps:
        ph = PHASE.get(ep["pred"])
        if ph and (ph not in first or ep["start_f"] < first[ph]):
            first[ph] = ep["start_f"]
    order = [PHASE_NAMES[p] for p, _ in sorted(first.items(), key=lambda kv: kv[1])]
    ranks = [p for p, _ in sorted(first.items(), key=lambda kv: kv[1])]
    k = max(1, round(0.5 * fps))
    smooth = np.convolve(curve, np.ones(k) / k, mode="same")
    step = max(1, round(fps / 2))
    return {"concern_seconds": round(float(concern.sum() / fps), 2), "benign_seconds": round(float(benign.sum() / fps), 2),
            "cue_sum": round(float(curve.sum() / fps), 2), "phases": order,
            "ordered_progression": bool(len(set(ranks)) >= 2 and all(x <= y for x, y in zip(ranks, ranks[1:]))),
            "cue_curve": [[round(t / fps, 2), round(float(smooth[t]), 2)] for t in range(0, n, step)]}


def find_escalation(pairs_eps):
    """Earliest ACT cue (reach / contact / flee) over the narrated pairs: {time_s, kind, pair, conf} or None."""
    best = None
    for key, eps in pairs_eps.items():
        for ep in eps:
            if ep["pred"] in ACT_PREDS and (best is None or ep["start_s"] < best["time_s"]):
                best = {"time_s": ep["start_s"], "kind": ep["pred"], "pair": key, "conf": ep["conf"]}
    return best


def geometry_for(t, layout_facts, depth, scene, i, j):
    """Pixel-space feet / heights per track and layout / depth info for graph episodes. Everything optional."""
    n = scene["n_frames"]
    feet, hpx = {}, {}
    for tid in (i, j):
        k = np.where(t["track_id"] == tid)[0]
        F = np.full((n, 2), np.nan)
        Hh = np.full(n, np.nan)
        b = t["bbox"][k]
        F[t["frame_idx"][k]] = np.c_[(b[:, 0] + b[:, 2]) / 2, b[:, 3]]
        Hh[t["frame_idx"][k]] = b[:, 3] - b[:, 1]
        feet[tid], hpx[tid] = F, Hh
    rows = None
    if depth and depth.get("pair") and sorted(depth["pair"]) == sorted([i, j]):
        rows = depth.get("frames") or []
        if depth["pair"][0] != i:       # the depth file's 'i' is the first of ITS pair; keep orientation consistent with (i, j)
            rows = [{**r, "pinned_i": r.get("pinned_j"), "pinned_j": r.get("pinned_i")} for r in rows]
    return {"doors": (layout_facts or {}).get("doors", []), "layout_reliable": bool((layout_facts or {}).get("reliable")),
            "size": (float(t["width"]), float(t["height"])), "feet": feet, "h": hpx, "depth_rows": rows, "fps": scene["fps"]}


def _low_light(layout_facts, scene):
    """Low light from the robust keyframe statistics (Stage B) if available, else from Stage A's mean-brightness flag (less reliable)."""
    light = (layout_facts or {}).get("light") or {}
    if light.get("low_light") is not None:
        return bool(light["low_light"])
    return bool(scene["camera"].get("night"))


def shot_facts(t, scene, key_pair_ids):
    """Which shot the story describes (the one holding the key pair) and how many shots the clip has."""
    n, fps = scene["n_frames"], scene["fps"]
    starts = [int(x) for x in t["shot_starts"]] if "shot_starts" in t else [0]
    ends = [x - 1 for x in starts[1:]] + [n - 1]
    shot = min(int(key_pair_ids[0]) // SHOT_BASE, len(starts) - 1) if key_pair_ids else int(np.argmax([e - s for s, e in zip(starts, ends)]))
    return {"index": shot, "n_shots": len(starts), "start_f": starts[shot], "end_f": ends[shot],
            "start_s": round(starts[shot] / fps, 2), "end_s": round((ends[shot] + 1) / fps, 2)}


def build_story(t, scene, arrays, layout_facts, depth, ctx, story, clip_id="", category=""):
    """The whole story of a clip: scene facts, episodes of the top pairs, key pair, escalation, narrative-ready summaries."""
    from src.context.features import rank_pairs
    fps, n = scene["fps"], scene["n_frames"]
    cam_moving = bool(scene["camera"].get("moving"))
    keys = [k for k in rank_pairs(scene, int(story["story_pairs"])) if tuple(int(x) for x in k.split("_")) in arrays]
    pairs, all_eps = {}, {}
    for key in keys:
        i, j = (int(x) for x in key.split("_"))
        geo = geometry_for(t, layout_facts, depth, scene, i, j)
        eps = pair_episodes(i, j, arrays[(i, j)], fps, ctx, story, geo, cam_moving)
        all_eps[key] = eps
        pairs[key] = {"pair": [i, j], "episodes": eps, **summarize_pair(eps, n, fps, story["cue_weights"]),
                      "min_dist_m": scene["pairs"][key].get("min_dist_m")}
    key_pair = max(pairs, key=lambda k: (pairs[k]["cue_sum"], pairs[k]["concern_seconds"])) if pairs else None
    if key_pair and pairs[key_pair]["concern_seconds"] == 0 and pairs[key_pair]["benign_seconds"] == 0 and not pairs[key_pair]["episodes"]:
        pass
    esc = find_escalation(all_eps)
    facts = {
        "place_type": (layout_facts or {}).get("place_type", "unknown"), "layout_reliable": bool((layout_facts or {}).get("reliable")),
        "layout_conf": (layout_facts or {}).get("layout_conf"), "doors": len((layout_facts or {}).get("doors", [])),
        "low_light": _low_light(layout_facts, scene), "camera_moving": cam_moving, "max_people": scene["max_people"],
        "isolated_frac": scene["isolated_frac"], "n_pairs": scene["n_pairs"], "duration_s": scene["duration_s"],
        "depth_agree_frac": (depth or {}).get("depth_agree_frac"), "assumptions": scene["assumptions"],
        "shot": shot_facts(t, scene, pairs[key_pair]["pair"] if key_pair else None),
    }
    return {"clip_id": clip_id, "category": category, "fps": fps, "n_frames": n, "scene": facts, "key_pair": key_pair,
            "no_pair": not pairs, "pairs": pairs, "escalation": esc,
            "concern_seconds": pairs[key_pair]["concern_seconds"] if key_pair else 0.0,
            "benign_seconds": pairs[key_pair]["benign_seconds"] if key_pair else 0.0,
            "preds_seen": sorted({e["pred"] for eps in all_eps.values() for e in eps}),
            "weights": story["cue_weights"]}


def save_story(out_dir, category, clip_id, story_dict):
    d = Path(out_dir) / category
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{clip_id}_story.json").write_text(json.dumps(story_dict, indent=1, default=float), encoding="utf-8")


def load_story(out_dir, category, clip_id):
    p = Path(out_dir) / category / f"{clip_id}_story.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None
