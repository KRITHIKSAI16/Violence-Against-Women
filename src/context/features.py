"""Deep context, Stage A orchestrator: per-pair features in meters + first behavior events, per clip.

For every pair of tracked people (co-tracked long enough, not too far apart) it computes, per frame:
  dist_m, closing_ms (+ = approaching), zone (Hall proxemics 0 intimate .. 4 far), speed_i/j (m/s),
  ang_i_to_j / ang_j_to_i (0 deg = faces the other, 180 = faces away), head_ang_i / head_ang_j,
  toward_i / toward_j (m/s each moves toward the other),
  wrist_torso_m (contact distance), reach_i_to_j_ms / reach_j_to_i_ms,
  follow_i_j / follow_j_i (i follows j; lagged path), lag_i_j / lag_j_i (s), dev_i_j / dev_j_i (m),
  and the boolean predicates approach_behind_i_j (i approaches j from behind), approach_behind_j_i, looking_back_i / looking_back_j
  (head turned toward the other while the body faces away), mutual_facing, contact, flee_i / flee_j, still_i / still_j.
A short list of events per pair (runs of these predicates) and scene facts (camera moving, night, crowd, isolation) go into the scene json.

Outputs per clip (context_dir/<Category>/):  <clip>_pairs.npz  (keys "<i>_<j>__<feature>")  and  <clip>_scene.json
Usage:  python -m src.context.features [--config CFG] [--clips A B] [--force] [--summary]
"""
import argparse
import json
import logging
from collections import defaultdict
from pathlib import Path

import numpy as np

from src.assm.track_poses import load_tracks
from src.config import load_config, parse_overrides, resolve_path
from src.context.camera import boxes_from_tracks, estimate_camera, summarize_camera
from src.context.follow import follow_scores
from src.context.ground import ground_series, pair_distance_m, speed_ms, zone_index
from src.context.pose_features import angle_deg, contact_features, runs_of, track_pose

log = logging.getLogger(__name__)


def _lag(x, L):
    out = np.full_like(x, np.nan, dtype=float)
    out[L:] = x[:-L]
    return out


def _unit(v):
    n = np.linalg.norm(v, axis=1, keepdims=True)
    with np.errstate(all="ignore"):
        return np.where(n > 1e-9, v / n, np.nan)


def _heading(g, fps, vel_s):
    """Direction of travel (n,2) in ground-plane coordinates and speed (m/s)."""
    L = max(1, round(vel_s * fps))
    d = np.stack([g["X"] - _lag(g["X"], L), g["Z"] - _lag(g["Z"], L)], axis=1)
    return d / (L / fps)


def pair_arrays(i, j, gi, gj, pi, pj, fps, ctx):
    """All per-frame feature arrays for the ordered-by-id pair (i, j)."""
    n = len(gi["X"])
    vel = ctx["vel_s"]
    L = max(1, round(vel * fps))
    dist = pair_distance_m(gi, gj)
    closing = (_lag(dist, L) - dist) / (L / fps)
    # Speeds come from box heights, so they are only trusted for upright, uncropped people (ground confidence).
    ok_i, ok_j = gi["conf"] >= ctx["min_ground_conf"], gj["conf"] >= ctx["min_ground_conf"]
    vi, vj = _heading(gi, fps, vel), _heading(gj, fps, vel)
    vi[~ok_i] = np.nan
    vj[~ok_j] = np.nan
    si, sj = np.linalg.norm(vi, axis=1), np.linalg.norm(vj, axis=1)
    dir_ij = np.stack([gj["X"] - gi["X"], gj["Z"] - gi["Z"]], axis=1)
    dir_ji = -dir_ij
    # facing: pose-based when known, otherwise direction of travel for people who are clearly walking
    face_i = np.where(np.isnan(pi["body"]), np.where((si > 0.5)[:, None], vi, np.nan), pi["body"])
    face_j = np.where(np.isnan(pj["body"]), np.where((sj > 0.5)[:, None], vj, np.nan), pj["body"])
    ang_ij, ang_ji = angle_deg(face_i, dir_ij), angle_deg(face_j, dir_ji)
    body_ang_i, body_ang_j = angle_deg(pi["body"], dir_ij), angle_deg(pj["body"], dir_ji)
    head_ang_i = np.where(pi["head_conf"] >= 0.5, angle_deg(pi["head"], dir_ij), np.nan)
    head_ang_j = np.where(pj["head_conf"] >= 0.5, angle_deg(pj["head"], dir_ji), np.nan)
    toward_i = (vi * _unit(dir_ij)).sum(axis=1)
    toward_j = (vj * _unit(dir_ji)).sum(axis=1)
    cf = contact_features(pi, pj, ctx["person_height_m"], L, fps)
    # running away is judged over a longer window: single-frame box-height jitter would otherwise look like sprinting
    vi_s, vj_s = _heading(gi, fps, ctx["flee_vel_s"]), _heading(gj, fps, ctx["flee_vel_s"])
    vi_s[~ok_i] = np.nan
    vj_s[~ok_j] = np.nan
    si_s, sj_s = np.linalg.norm(vi_s, axis=1), np.linalg.norm(vj_s, axis=1)
    Ls = max(1, round(ctx["flee_vel_s"] * fps))
    closing_s = (_lag(dist, Ls) - dist) / (Ls / fps)

    away, toward = ctx["facing_away_deg"], ctx["facing_toward_deg"]
    with np.errstate(invalid="ignore"):
        close = dist < 6.0
        out = {
            "dist_m": dist, "closing_ms": closing, "zone": zone_index(dist), "speed_i": si, "speed_j": sj,
            "ang_i_to_j": ang_ij, "ang_j_to_i": ang_ji, "head_ang_i": head_ang_i, "head_ang_j": head_ang_j,
            "toward_i": toward_i, "toward_j": toward_j,
            "wrist_torso_m": cf["wrist_torso_m"], "reach_i_to_j_ms": cf["reach_i_to_j_ms"], "reach_j_to_i_ms": cf["reach_j_to_i_ms"],
            "ground_conf": np.fmin(gi["conf"], gj["conf"]),
            "approach_behind_i_j": close & (toward_i > 0.3) & (ang_ji > away),
            "approach_behind_j_i": close & (toward_j > 0.3) & (ang_ij > away),
            "looking_back_i": (dist < 8.0) & (head_ang_i < toward) & (body_ang_i > 100.0),
            "looking_back_j": (dist < 8.0) & (head_ang_j < toward) & (body_ang_j > 100.0),
            "mutual_facing": (dist < 4.0) & (ang_ij < toward) & (ang_ji < toward),
            "contact": (cf["wrist_torso_m"] < ctx["contact_wrist_m"]) & (dist < ctx["contact_max_ground_m"]),
            "flee_i": (si_s > ctx["flee_speed_ms"]) & (closing_s < -0.5), "flee_j": (sj_s > ctx["flee_speed_ms"]) & (closing_s < -0.5),
            "still_i": si < 0.2, "still_j": sj < 0.2,
        }
    Pi = np.stack([gi["X"], gi["Z"]], axis=1)
    Pj = np.stack([gj["X"], gj["Z"]], axis=1)
    Pi[~ok_i] = np.nan
    Pj[~ok_j] = np.nan
    for name, (a, b) in (("i_j", (Pi, Pj)), ("j_i", (Pj, Pi))):
        r = follow_scores(a, b, fps, ctx)
        out[f"follow_{name}"], out[f"lag_{name}"], out[f"dev_{name}"] = r["follow"], r["lag_s"], r["dev_m"]
    for k, v in out.items():
        if v.dtype == bool:
            out[k] = np.nan_to_num(v, nan=0).astype(bool) if v.dtype != bool else v
    return out


def pair_events(a, fps, ctx):
    """First behavior events of a pair as plain numbers: counts, total seconds, first start (s)."""
    me = max(1, round(ctx["min_event_s"] * fps))
    spec = {  # name -> (boolean array, min frames)
        "follow_i_j": (a["follow_i_j"], round(ctx["follow_min_s"] * fps)),
        "follow_j_i": (a["follow_j_i"], round(ctx["follow_min_s"] * fps)),
        "approach_behind_i_j": (a["approach_behind_i_j"], me), "approach_behind_j_i": (a["approach_behind_j_i"], me),
        "looking_back_i": (a["looking_back_i"], max(1, round(0.3 * fps))), "looking_back_j": (a["looking_back_j"], max(1, round(0.3 * fps))),
        "mutual_facing": (a["mutual_facing"], round(2 * fps)), "contact": (a["contact"], max(1, round(0.2 * fps))),
        "flee_i": (a["flee_i"], me), "flee_j": (a["flee_j"], me),
        "reach_i_to_j": (np.nan_to_num(a["reach_i_to_j_ms"], nan=0) > 1.5, 2),
        "reach_j_to_i": (np.nan_to_num(a["reach_j_to_i_ms"], nan=0) > 1.5, 2),
    }
    out = {}
    for name, (mask, mn) in spec.items():
        rr = runs_of(mask, max(1, mn))
        if rr:
            out[name] = {"count": len(rr), "seconds": round(sum(e - s + 1 for s, e in rr) / fps, 2),
                         "first_s": round(rr[0][0] / fps, 2), "runs": [[round(s / fps, 2), round((e + 1) / fps, 2)] for s, e in rr[:8]]}
    zone = a["zone"]
    out["min_dist_m"] = round(float(np.nanmin(a["dist_m"])), 2) if np.isfinite(a["dist_m"]).any() else None
    out["intimate_or_personal_s"] = round(float(np.sum((zone >= 0) & (zone <= 1)) / fps), 2)
    return out


def analyze_context(t, ctx, cam=None):
    """Stage A for one clip. `cam` = estimate_camera() result (None = static camera assumed). Returns (scene dict, pair arrays dict)."""
    fps, n = float(t["fps"]), int(t["n_frames"])
    cam_sum = summarize_camera(cam, ctx) if cam is not None else {"moving": False, "night": False, "brightness": None}
    T = cam["T"] if cam is not None else None
    ids_all = sorted(set(t["track_id"].tolist()))
    mean_conf = {tid: float(t["conf"][t["track_id"] == tid].mean()) for tid in ids_all}
    ids = [tid for tid in ids_all if mean_conf[tid] >= ctx["min_track_conf"]]
    G = ground_series(t, ctx, T, ids)
    P = {tid: track_pose(t, tid, n) for tid in ids}
    min_co = max(2, round(ctx["min_cotracked_s"] * fps))
    pairs, arrays = {}, {}
    for a in range(len(ids)):
        for b in range(a + 1, len(ids)):
            i, j = ids[a], ids[b]
            co = ~np.isnan(G[i]["X"]) & ~np.isnan(G[j]["X"])
            if co.sum() < min_co:
                continue
            if np.nanmin(pair_distance_m(G[i], G[j])) > ctx["max_pair_dist_m"]:
                continue
            arr = pair_arrays(i, j, G[i], G[j], P[i], P[j], fps, ctx)
            arrays[(i, j)] = arr
            pairs[f"{i}_{j}"] = pair_events(arr, fps, ctx)
    people = np.zeros(n, int)
    for f in t["frame_idx"].tolist():
        people[f] += 1
    scene = {
        "fps": fps, "n_frames": n, "duration_s": round(n / fps, 3), "camera": cam_sum,
        "n_tracks": len(ids_all), "n_tracks_used": len(ids), "n_pairs": len(pairs), "no_pair": len(pairs) == 0,
        "max_people": int(people.max()) if n else 0,
        "mean_people": round(float(people[people > 0].mean()), 2) if (people > 0).any() else 0.0,
        "isolated_frac": round(float((people[people > 0] == 2).mean()), 3) if (people > 0).any() else 0.0,
        "assumptions": {"fov_deg": ctx["fov_deg"], "person_height_m": ctx["person_height_m"]},
        "pairs": pairs,
    }
    scene["clip_events"] = clip_events(scene)
    return scene, arrays


EVENT_GROUPS = {"follow": ("follow_i_j", "follow_j_i"), "approach_from_behind": ("approach_behind_i_j", "approach_behind_j_i"),
                "looking_back": ("looking_back_i", "looking_back_j"), "contact": ("contact",), "reach": ("reach_i_to_j", "reach_j_to_i"),
                "flee": ("flee_i", "flee_j"), "mutual_facing": ("mutual_facing",)}


def clip_events(scene):
    """Which behavior groups occur anywhere in the clip (any pair): {group: {count, seconds}}."""
    out = {}
    for g, names in EVENT_GROUPS.items():
        cnt, sec = 0, 0.0
        for ev in scene["pairs"].values():
            for nm in names:
                if nm in ev:
                    cnt += ev[nm]["count"]
                    sec += ev[nm]["seconds"]
        if cnt:
            out[g] = {"count": cnt, "seconds": round(sec, 2)}
    return out


def save_context(scene, arrays, out_dir, category, clip_id):
    d = Path(out_dir) / category
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{clip_id}_scene.json").write_text(json.dumps(scene, indent=1), encoding="utf-8")
    flat = {f"{i}_{j}__{k}": v for (i, j), arr in arrays.items() for k, v in arr.items()}
    np.savez_compressed(d / f"{clip_id}_pairs.npz", **flat)


def load_context(out_dir, category, clip_id):
    """-> (scene dict, {(i,j): {feature: array}})."""
    d = Path(out_dir) / category
    scene = json.loads((d / f"{clip_id}_scene.json").read_text(encoding="utf-8"))
    arrays = defaultdict(dict)
    with np.load(d / f"{clip_id}_pairs.npz") as z:
        for key in z.files:
            pair, feat = key.split("__", 1)
            i, j = pair.split("_")
            arrays[(int(i), int(j))][feat] = z[key]
    return scene, dict(arrays)


def _one_clip(args):
    clip, cfg, force = args
    ctx = cfg["context"]
    out_dir = resolve_path(ctx["context_dir"])
    sc = out_dir / clip["category"] / f"{clip['clip_id']}_scene.json"
    if sc.exists() and not force:
        return clip, json.loads(sc.read_text(encoding="utf-8")), "cached"
    tp = resolve_path(cfg["assm"]["tracks_dir"]) / clip["category"] / f"{clip['clip_id']}.npz"
    if not tp.exists():
        return clip, None, "no tracks"
    try:
        t = load_tracks(tp)
        cam = estimate_camera(resolve_path(clip["path"]), boxes_from_tracks(t), ctx["camera_max_side"])
        scene, arrays = analyze_context(t, ctx, cam)
        save_context(scene, arrays, out_dir, clip["category"], clip["clip_id"])
        return clip, scene, "ok"
    except Exception as e:  # one bad clip must not stop a long run
        return clip, None, f"failed: {e}"


def print_summary(rows):
    """rows: list of (category, scene). Decision-gate table: how often each behavior is found, by category."""
    by = defaultdict(list)
    for cat, sc in rows:
        by[cat].append(sc)
    cols = ["follow", "approach_from_behind", "looking_back", "contact", "reach", "flee", "mutual_facing"]
    head = f"{'category':<16}{'clips':>6}{'no-pair':>9}{'cam-mov':>9}{'night':>7}{'isolated':>9}" + "".join(f"{c[:9]:>11}" for c in cols)
    print(head)
    for cat, lst in sorted(by.items()):
        n = len(lst)
        iso = np.mean([s["isolated_frac"] for s in lst if not s["no_pair"]] or [0])
        print(f"{cat:<16}{n:>6}{sum(s['no_pair'] for s in lst)/n:>9.0%}{sum(s['camera']['moving'] for s in lst)/n:>9.0%}"
              f"{sum(bool(s['camera'].get('night')) for s in lst)/n:>7.0%}{iso:>9.0%}"
              + "".join(f"{sum(c in s['clip_events'] for s in lst)/n:>11.0%}" for c in cols))


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    ap = argparse.ArgumentParser(description="Deep context Stage A: per-pair features in meters")
    ap.add_argument("--config", default=None)
    ap.add_argument("--clips", nargs="*", default=None)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--summary", action="store_true", help="only print the table from existing results")
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE", help="override context settings for this run")
    args = ap.parse_args()
    cfg = load_config(args.config)
    cfg["context"] = {**cfg["context"], **parse_overrides(args.set)}
    clean = json.loads(resolve_path(cfg["preprocess"]["clean_manifest_path"]).read_text("utf-8"))
    clips = [c for c in clean["clips"] if not args.clips or c["clip_id"] in args.clips]
    out_dir = resolve_path(cfg["context"]["context_dir"])
    rows, failed = [], []
    if args.summary:
        for c in clips:
            p = out_dir / c["category"] / f"{c['clip_id']}_scene.json"
            if p.exists():
                rows.append((c["category"], json.loads(p.read_text(encoding="utf-8"))))
    else:
        jobs = [(c, cfg, args.force) for c in clips]
        if args.workers > 1:
            from concurrent.futures import ProcessPoolExecutor
            with ProcessPoolExecutor(args.workers) as ex:
                results = ex.map(_one_clip, jobs, chunksize=4)
                results = list(results)
        else:
            results = map(_one_clip, jobs)
        for k, (c, sc, status) in enumerate(results, 1):
            if sc is None:
                failed.append((c["clip_id"], status))
            else:
                rows.append((c["category"], sc))
            if k % 100 == 0:
                print(f"  context: {k}/{len(jobs)} clips", flush=True)
    print(f"\n{len(rows)} clips with context results -> {out_dir}  ({len(failed)} failed or without tracks)\n")
    if rows:
        print_summary(rows)
    for cid, why in failed[:10]:
        print(f"  {cid}: {why}")


if __name__ == "__main__":
    main()
