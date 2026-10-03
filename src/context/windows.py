"""Deep context, Stage D step 1: turn per-pair, per-frame features into window-level rows for learning (report M4 "feature caching").

One row = one pair of people in one window of `window_s` seconds (step `step_s`). Pair roles are arbitrary (the lower track id is "i"),
so every feature is symmetric: it looks at either direction (max / min over the two people), never at who is i or j.

Feature groups (so the learning step can switch them on and off and look for shortcuts):
  behavior : distance, proxemic zones, approach/closing, speeds, stillness, following, approach-from-behind, looking back, mutual facing,
             facing away, contact, reach, running away, how long the pair has been together
  scene    : how many people, how isolated (context the report says matters)
  style    : how the clip was FILMED - camera moving, brightness, sharpness, duration, original resolution / fps. These should carry no
             information about violence; if they separate the categories, the dataset has a shortcut and results must be read with them removed.
Label = the clip's category (weak: Normal = 0, any other = 1). For categories with an act, only windows BEFORE the cut point are used
(`src.report.buildup_video.cut_point`), so the learner is not handed the violent act itself.
"""
import numpy as np

from src.context.pose_features import runs_of

BEHAVIOR = [
    "d_mean", "d_min", "d_net_closing", "frac_intimate", "frac_personal", "frac_social", "closing_mean", "closing_max",
    "speed_max", "speed_min", "speed_ratio", "still_any", "still_both", "follow_frac", "follow_dev", "follow_lag",
    "behind_frac", "lookback_frac", "lookback_events", "mutual_frac", "away_frac", "ang_mean", "contact_frac", "reach_max",
    "flee_frac", "pair_age_s", "ground_conf",
]
SCENE = ["scene_max_people", "scene_isolated_frac", "scene_n_pairs"]
STYLE = ["cam_moving", "cam_moving_share", "brightness", "sharpness", "night", "duration_s", "raw_width", "raw_height", "raw_fps"]


def _nm(x, fn=np.nanmean):
    x = np.asarray(x, float)
    return float(fn(x)) if np.isfinite(x).any() else np.nan


def _either(a, b):
    return np.asarray(a, bool) | np.asarray(b, bool)


def window_features(a, s, e, fps, first_frame, ctx):
    """Behavior features of one pair over frames [s, e). a: pair arrays dict (from analyze_context / load_context)."""
    sl = slice(s, e)
    d = a["dist_m"][sl]
    zone = a["zone"][sl]
    valid = np.isfinite(d)
    n = max(1, e - s)
    k = max(1, round(0.2 * n))
    first, last = d[valid][:k], d[valid][-k:]
    si, sj = a["speed_i"][sl], a["speed_j"][sl]
    smax = np.fmax(si, sj)
    smin = np.fmin(si, sj)
    follow = _either(a["follow_i_j"][sl], a["follow_j_i"][sl])
    lag = np.where(a["follow_i_j"][sl], a["lag_i_j"][sl], np.where(a["follow_j_i"][sl], a["lag_j_i"][sl], np.nan))
    dev = np.fmin(a["dev_i_j"][sl], a["dev_j_i"][sl])
    lb = _either(a["looking_back_i"][sl], a["looking_back_j"][sl])
    ang = np.fmin(a["ang_i_to_j"][sl], a["ang_j_to_i"][sl])
    away = _either(a["ang_i_to_j"][sl] > ctx["facing_away_deg"], a["ang_j_to_i"][sl] > ctx["facing_away_deg"])
    reach = np.fmax(a["reach_i_to_j_ms"][sl], a["reach_j_to_i_ms"][sl])
    with np.errstate(all="ignore"):
        row = {
            "d_mean": _nm(d), "d_min": _nm(d, np.nanmin),
            "d_net_closing": float(np.nanmean(first) - np.nanmean(last)) if len(first) and len(last) else np.nan,
            "frac_intimate": float(np.mean(zone[valid] == 0)) if valid.any() else np.nan,
            "frac_personal": float(np.mean(zone[valid] == 1)) if valid.any() else np.nan,
            "frac_social": float(np.mean(zone[valid] == 2)) if valid.any() else np.nan,
            "closing_mean": _nm(a["closing_ms"][sl]), "closing_max": _nm(a["closing_ms"][sl], np.nanmax),
            "speed_max": _nm(smax), "speed_min": _nm(smin),
            "speed_ratio": _nm(smin) / _nm(smax) if np.isfinite(_nm(smax)) and _nm(smax) > 0.05 else np.nan,
            "still_any": float(np.mean(_either(a["still_i"][sl], a["still_j"][sl]))),
            "still_both": float(np.mean(a["still_i"][sl] & a["still_j"][sl])),
            "follow_frac": float(np.mean(follow)), "follow_dev": _nm(dev), "follow_lag": _nm(lag),
            "behind_frac": float(np.mean(_either(a["approach_behind_i_j"][sl], a["approach_behind_j_i"][sl]))),
            "lookback_frac": float(np.mean(lb)), "lookback_events": float(len(runs_of(lb, max(1, round(0.3 * fps))))),
            "mutual_frac": float(np.mean(a["mutual_facing"][sl])), "away_frac": float(np.mean(away)),
            "ang_mean": _nm(ang), "contact_frac": float(np.mean(a["contact"][sl])), "reach_max": _nm(reach, np.nanmax),
            "flee_frac": float(np.mean(_either(a["flee_i"][sl], a["flee_j"][sl]))),
            "pair_age_s": (s - first_frame) / fps, "ground_conf": _nm(a["ground_conf"][sl]),
        }
    return row


def pair_windows(a, scene, ctx, win_s, step_s, end_frame, min_valid=0.5):
    """All windows of one pair that end before `end_frame`; windows with too little co-tracked data are skipped."""
    fps = scene["fps"]
    W, S = max(2, round(win_s * fps)), max(1, round(step_s * fps))
    valid = np.isfinite(a["dist_m"])
    if not valid.any():
        return []
    first = int(np.argmax(valid))
    out = []
    for s in range(0, max(1, min(end_frame, len(valid)) - W + 1), S):
        e = s + W
        if valid[s:e].mean() < min_valid:
            continue
        r = window_features(a, s, e, fps, first, ctx)
        r["t_start"] = round(s / fps, 2)
        out.append(r)
    return out


def scene_features(scene):
    return {"scene_max_people": scene["max_people"], "scene_isolated_frac": scene["isolated_frac"], "scene_n_pairs": scene["n_pairs"]}


def style_features(scene, raw):
    """How the clip was filmed. `raw` = the M1 manifest entry of the ORIGINAL clip (resolution / fps before M2 normalised them)."""
    cam = scene["camera"]
    return {"cam_moving": float(cam.get("moving", False)), "cam_moving_share": cam.get("moving_share", 0.0) or 0.0,
            "brightness": cam.get("brightness") if cam.get("brightness") is not None else np.nan,
            "sharpness": cam.get("sharpness") if cam.get("sharpness") is not None else np.nan,
            "night": float(cam.get("night", False)), "duration_s": scene["duration_s"],
            "raw_width": raw.get("width", np.nan), "raw_height": raw.get("height", np.nan), "raw_fps": raw.get("fps", np.nan)}
