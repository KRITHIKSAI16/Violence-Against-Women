"""Curated dataset: the understanding of one pair of people before the violence, in numbers and plain sentences.

Input: the per-frame pair arrays of `src.context.features` (meters) of a clip that ENDS at the violence start T, so the final seconds of the arrays
are exactly the final seconds before the violence. Output (`summarize`):

  spans         relation spans (src.curated.relation) with who does what to whom, distances and closing speed
  last          the last 1.5 s and 3 s: distance start -> end, net closing, who closes in, whether the other person faces away, reach / contact
  leadup        one lead-up type with the numbers that justify it (a heuristic; the order of the rules is in LEADUP_RULES)
  concern       per-frame concern score and its windows (heuristic weights in CONCERN_W; used to compare with Normal clips)
  lines         plain-language sentences for the report

Everything is geometry from a monocular camera: meters are approximate (1.7 m height prior, assumed field of view).
"""
import warnings

import numpy as np

from src.curated.relation import BENIGN, CONCERN, THRESH, frame_labels, label_seconds, spans
from src.video.shots import local_id

CONCERN_W = {"arm_raised": 1.0, "follows": 2.0, "approaches_from_behind": 1.0, "approaches": 0.6, "contact_or_reach": 2.0, "looking_back": 1.5, "close": 0.5,
             "walking_together": -1.5, "standing_together": -1.0, "closing_credit": 0.4}
LEADUP_RULES = ["contact or reach at the end", "following", "approach from behind", "approach", "closing in", "raised arm near the other person", "already close at the start",
                "walking together", "no visible lead-up"]
PHRASE = {"contact_or_reach": "{a} reaches for / touches {b}", "follows": "{a} follows {b}", "approaches_from_behind": "{a} approaches {b} from behind",
          "approaches": "{a} moves toward {b}", "walking_together": "the two walk together", "standing_together": "the two are close and nearly still or facing each other",
          "moving_apart": "the two move apart", "close_unclear": "the two are close", "apart": "the two are apart"}


def _nz(x):
    return np.nan_to_num(np.asarray(x, float), nan=0.0)


def _mean(x):
    x = np.asarray(x, float)
    return float(np.nanmean(x)) if np.isfinite(x).any() else None


def _ends(a, sl):
    d = np.asarray(a["dist_m"], float)[sl]
    d = d[np.isfinite(d)]
    if len(d) < 2:
        return None, None
    k = max(1, len(d) // 5)
    return float(np.median(d[:k])), float(np.median(d[-k:]))


def who(a, sl, ids):
    """(actor, target) ids for the frames in sl: whoever moves toward the other faster (m/s) is the actor; None when neither does."""
    ti, tj = _mean(np.asarray(a["toward_i"], float)[sl]), _mean(np.asarray(a["toward_j"], float)[sl])
    ti, tj = (ti if ti is not None else -9), (tj if tj is not None else -9)
    if max(ti, tj) < 0.2:
        return None, None
    return (ids[0], ids[1]) if ti >= tj else (ids[1], ids[0])


def raised_arm(pose, hpx=None, margin=0.05):
    """track_pose dict -> bool per frame: a wrist is above the shoulder line (arm raised: a swing, a threat, a grab) by more than `margin` of the box height."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)                       # frames without visible shoulders / wrists are NaN by design
        sho = np.nanmin(pose["torso"][:, 0:2, 1], axis=1)
        wr = np.nanmin(pose["wrist"][:, :, 1], axis=1)
    h = pose["hpx"] if hpx is None else hpx
    with np.errstate(invalid="ignore"):
        return np.nan_to_num(wr < sho - margin * h, nan=0).astype(bool)


def concern_series(a, fps, extra=None):
    """Per-frame concern score of the pair (heuristic weights CONCERN_W) and the per-frame relation labels."""
    lab = frame_labels(a)
    s = np.zeros(len(lab))
    for name, w in CONCERN_W.items():
        if name in ("looking_back", "close", "closing_credit"):
            continue
        s += w * (lab == name)
    lb = np.asarray(a["looking_back_i"], bool) | np.asarray(a["looking_back_j"], bool)
    s += CONCERN_W["looking_back"] * lb
    zone = np.asarray(a["zone"], float)
    s += CONCERN_W["close"] * ((zone <= 1) & ~np.isin(lab, list(BENIGN)))
    s += CONCERN_W["closing_credit"] * np.clip(_nz(a["closing_ms"]), 0, 2) * (np.nan_to_num(np.asarray(a["dist_m"], float), nan=99) < 6.0)
    if extra is not None and "arm_raised" in extra:
        s += CONCERN_W["arm_raised"] * (np.asarray(extra["arm_raised"], bool)[:len(s)] & (np.nan_to_num(np.asarray(a["dist_m"], float), nan=99) < 3.0))
    s[lab == "none"] = np.nan
    return s, lab


def window_scores(series, fps, win_s, step_s):
    """[(t_start_s, mean score)] over windows of win_s; windows with too little data are skipped."""
    W, S = max(2, round(win_s * fps)), max(1, round(step_s * fps))
    out = []
    for s in range(0, max(1, len(series) - W + 1), S):
        w = series[s:s + W]
        if np.isfinite(w).mean() >= 0.5:
            out.append((round(s / fps, 2), float(np.nanmean(w))))
    return out


def last_window(a, fps, ids, seconds):
    """Facts about the last `seconds` of the clip (= right before the violence)."""
    n = len(a["dist_m"])
    sl = slice(max(0, n - round(seconds * fps)), n)
    d0, d1 = _ends(a, sl)
    actor, target = who(a, sl, ids)
    ang_t = None
    if actor is not None:
        ang_t = _mean(np.asarray(a["ang_j_to_i" if actor == ids[0] else "ang_i_to_j"], float)[sl])         # target's facing relative to the actor: 180 = faces away
    reach = float(np.nanmax(np.fmax(_nz(a["reach_i_to_j_ms"])[sl], _nz(a["reach_j_to_i_ms"])[sl]))) if sl.stop > sl.start else 0.0
    dd = np.asarray(a["dist_m"], float)[sl]
    return {"seconds": round((sl.stop - sl.start) / fps, 2), "dist_start_m": None if d0 is None else round(d0, 2), "dist_end_m": None if d1 is None else round(d1, 2),
            "net_closing_m": None if d0 is None else round(d0 - d1, 2), "mean_closing_ms": None if _mean(a["closing_ms"][sl]) is None else round(_mean(a["closing_ms"][sl]), 2),
            "min_dist_m": round(float(np.nanmin(dd)), 2) if np.isfinite(dd).any() else None, "actor": actor, "target": target,
            "target_faces_away_deg": None if ang_t is None else round(ang_t, 0), "reach_ms": round(reach, 2),
            "contact_frac": round(float(np.mean(np.asarray(a["contact"], bool)[sl])), 2) if sl.stop > sl.start else 0.0,
            "intimate_or_personal_frac": round(float(np.mean(np.asarray(a["zone"], float)[sl] <= 1)), 2) if sl.stop > sl.start else 0.0}


def leadup_type(sp_secs, last, first_dist, raised_s=0.0):
    """-> (type, reason). Rules in LEADUP_RULES order, on the last 3 s of relation seconds and the distances."""
    L = last
    if sp_secs.get("contact_or_reach", 0) >= 0.3 or (L["contact_frac"] or 0) > 0.2 or (L.get("reach_ms") or 0) >= THRESH["reach_ms"]:
        return LEADUP_RULES[0], f"hands reach the other person's body in the last {L['seconds']:.1f} s"
    if sp_secs.get("follows", 0) >= 1.5:
        return LEADUP_RULES[1], f"one person retraces the other's path for {sp_secs['follows']:.1f} s"
    if sp_secs.get("approaches_from_behind", 0) >= 0.5:
        return LEADUP_RULES[2], f"closes in on someone who is facing away for {sp_secs['approaches_from_behind']:.1f} s"
    if sp_secs.get("approaches", 0) >= 0.5:
        return LEADUP_RULES[3], f"moves toward the other person for {sp_secs['approaches']:.1f} s"
    if (L["net_closing_m"] or 0) >= 0.8:
        return LEADUP_RULES[4], f"the gap shrinks by {L['net_closing_m']:.1f} m in the last {L['seconds']:.1f} s"
    if raised_s >= 0.3 and (L["min_dist_m"] or 9) < 3.0:
        return LEADUP_RULES[5], f"an arm is raised above the shoulder for {raised_s:.1f} s while within {L['min_dist_m']:.1f} m of the other person"
    if first_dist is not None and first_dist < 1.5 and (L["dist_end_m"] or 9) < 1.5:
        return LEADUP_RULES[6], f"within {first_dist:.1f} m at the first frame and {L['dist_end_m']:.1f} m at the end"
    if sp_secs.get("walking_together", 0) + sp_secs.get("standing_together", 0) >= 1.0:
        return LEADUP_RULES[7], "the two move or stand together without closing in"
    return LEADUP_RULES[8], "no approach, following, reach or closing distance is measurable in the footage before the violence"


def summarize(a, fps, ids, last_seconds=(1.5, 3.0), win_s=2.0, step_s=0.5, extra=None):
    """Everything about one pair before the violence. a: pair arrays of the key pair; ids: (id_i, id_j) track ids.
    extra: optional {"arm_raised": {track id: bool array per frame}} from the pose keypoints."""
    raised = None
    if extra and extra.get("arm_raised"):
        raised = {"arm_raised": np.any([np.asarray(v, bool)[:len(a["dist_m"])] for v in extra["arm_raised"].values()], axis=0)}
    n = len(a["dist_m"])
    labels = frame_labels(a)
    sp = spans(labels, fps, a["dist_m"], a["closing_ms"])
    for s in sp:                                                         # who does what to whom
        k0, k1 = int(round(s["start_s"] * fps)), int(round(s["end_s"] * fps))
        sl = slice(k0, min(k1, n))
        if s["label"] == "follows":
            fi = float(np.mean(np.asarray(a["follow_i_j"], bool)[sl]))
            fj = float(np.mean(np.asarray(a["follow_j_i"], bool)[sl]))
            s["actor"], s["target"] = (ids[0], ids[1]) if fi >= fj else (ids[1], ids[0])
        elif s["label"] == "approaches_from_behind":
            bi = float(np.mean(np.asarray(a["approach_behind_i_j"], bool)[sl]))
            bj = float(np.mean(np.asarray(a["approach_behind_j_i"], bool)[sl]))
            s["actor"], s["target"] = (ids[0], ids[1]) if bi >= bj else (ids[1], ids[0])
        elif s["label"] == "approaches":
            s["actor"], s["target"] = who(a, sl, ids)
        elif s["label"] == "contact_or_reach":                           # the hand that reaches, not the one that walks: compare the two reach speeds
            ri, rj = _mean(np.asarray(a["reach_i_to_j_ms"], float)[sl]), _mean(np.asarray(a["reach_j_to_i_ms"], float)[sl])
            ri, rj = (ri if ri is not None else 0.0), (rj if rj is not None else 0.0)
            if abs(ri - rj) >= 0.3:
                s["actor"], s["target"] = (ids[0], ids[1]) if ri > rj else (ids[1], ids[0])
            else:
                s["actor"] = s["target"] = None                          # both or neither: the direction is not claimed
        else:
            s["actor"] = s["target"] = None
    secs = label_seconds(sp)
    dd = np.asarray(a["dist_m"], float)
    first = _ends(a, slice(0, n))[0]
    lasts = {f"{L:g}": last_window(a, fps, ids, L) for L in last_seconds}
    lt = lasts[f"{max(last_seconds):g}"]
    up = 0.0
    if extra and extra.get("arm_raised"):
        up = max(float(np.sum(np.asarray(v, bool)[max(0, n - round(max(last_seconds) * fps)):])) / fps for v in extra["arm_raised"].values())
    typ, why = leadup_type(secs, lt, first, up)
    series, _ = concern_series(a, fps, raised)
    out = {"ids": [int(ids[0]), int(ids[1])], "duration_s": round(n / fps, 2), "pair_visible_s": round(float(np.isfinite(dd).sum()) / fps, 2), "spans": sp, "label_seconds": secs, "first_dist_m": None if first is None else round(first, 2),
           "min_dist_m": round(float(np.nanmin(dd)), 2) if np.isfinite(dd).any() else None, "last": lasts, "leadup": {"type": typ, "reason": why},
           "concern_last_s": None, "concern_windows": window_scores(series, fps, win_s, step_s)}
    w = series[max(0, n - round(max(last_seconds) * fps)):]
    out["concern_last_s"] = _mean(w)
    out["concern_mean"] = _mean(series)
    out["concern_max_window"] = max((s for _, s in out["concern_windows"]), default=None)
    if extra and extra.get("arm_raised"):
        Lw = max(last_seconds)
        out["arm_raised_last_s"] = {str(local_id(i)): round(float(np.sum(np.asarray(v, bool)[max(0, n - round(Lw * fps)):])) / fps, 2) for i, v in extra["arm_raised"].items()}
    out["lines"] = explain_lines(out)
    return out


def _nm(i):
    return f"id{local_id(i)}"


def explain_lines(s):
    """Plain sentences: each relation span, the last seconds, and the lead-up type."""
    lines = []
    if s.get("pair_visible_s") is not None and s["pair_visible_s"] < 0.8 * s["duration_s"]:
        lines.append(f"The two are measurable together for {s['pair_visible_s']:.1f} s of the {s['duration_s']:.1f} s shown (the rest: not both detected).")
    for sp in s["spans"]:
        if sp["label"] in ("apart", "close_unclear") and sp["end_s"] - sp["start_s"] < 1.0:
            continue
        a, b = (_nm(sp["actor"]), _nm(sp["target"])) if sp.get("actor") is not None else (_nm(s["ids"][0]), _nm(s["ids"][1]))
        text = PHRASE[sp["label"]].format(a=a, b=b)
        if sp["label"] == "contact_or_reach" and sp.get("actor") is None:
            text = f"a hand reaches the other person's body or the two touch ({_nm(s['ids'][0])}, {_nm(s['ids'][1])}; direction not clear)"
        if sp["dist_start_m"] is not None and sp["dist_end_m"] is not None:
            text += f" ({sp['dist_start_m']:.1f} m -> {sp['dist_end_m']:.1f} m" + (f", closing {sp['mean_closing_ms']:.1f} m/s" if sp["mean_closing_ms"] and abs(sp["mean_closing_ms"]) >= 0.15 else "") + ")"
        lines.append(f"{sp['start_s']:.1f}-{sp['end_s']:.1f} s: {text}")
    L = s["last"][max(s["last"], key=float)]
    if L["dist_start_m"] is not None:
        t = f"Last {L['seconds']:.1f} s before the violence: {L['dist_start_m']:.1f} m -> {L['dist_end_m']:.1f} m apart (closest {L['min_dist_m']:.1f} m)"
        if L["actor"] is not None:
            t += f"; {_nm(L['actor'])} moves toward {_nm(L['target'])}"
            if L["target_faces_away_deg"] is not None and L["target_faces_away_deg"] > 110:
                t += f", who is facing away"
        if L["reach_ms"] >= THRESH["reach_ms"] or L["contact_frac"] > 0.2:
            t += "; a hand reaches the other person's body"
        lines.append(t + ".")
    for i, secs_up in sorted(s.get("arm_raised_last_s", {}).items(), key=lambda kv: -kv[1]):
        if secs_up >= 0.3:
            lines.append(f"id{i} has an arm raised above the shoulder for {secs_up:.1f} s of the last {max(float(k) for k in s['last']):.1f} s (a swing, grab or threat is possible; pose heuristic).")
            break
    lines.append(f"Lead-up type (heuristic): {s['leadup']['type']} - {s['leadup']['reason']}.")
    return lines
