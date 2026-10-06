"""Classification pipelines, step 3: the geometry text of one clip (Layer A), built from the structured result of src.curated.pipeline.analyze_clip.

Why not reuse the ready-made `lines` of that result: for Normal clips they were rewritten ("before the violence" -> "of the clip"), so the wording itself would
tell the classifier the label. Everything here is generated from the numbers, with the same words for every clip, and never mentions violence.

Two tasks:
  trim  the clip ends where the violence starts (violent) or is a whole Normal clip: both are judged on the SAME last `span_s` seconds, so clip length
        cannot be read off the text. Times are given relative to the start of that window.
  full  the whole video is described.
Pair ids are replaced by "person A" / "person B" (no identity, no track numbers).
"""
from src.curated.relation import THRESH

NO_PAIR = {"no_people": "no person was detected", "one_person": "only one person was tracked reliably", "never_together": "people were never tracked together long enough",
           "no_tracks": "no tracking result exists for this clip"}
SPAN_TEXT = {"contact_or_reach": "{a} reaches for or touches {b}", "follows": "{a} follows {b}", "approaches_from_behind": "{a} approaches {b} from behind",
             "approaches": "{a} moves toward {b}", "walking_together": "the two walk together",
             "standing_together": "the two are close and nearly still or facing each other", "moving_apart": "the two move apart", "close_unclear": "the two are close",
             "apart": "the two are apart"}
MIN_OVERLAP_S = THRESH["min_span_s"]


def window_bounds(duration_s, task, span_s):
    """(lo, hi) of the analysed window in clip seconds: the last span_s seconds for trim, the whole clip for full."""
    hi = float(duration_s)
    return (max(0.0, hi - float(span_s)) if task == "trim" else 0.0), hi


def clip_spans(spans, lo, hi):
    """Spans overlapping [lo, hi] for at least MIN_OVERLAP_S, cut to it -> [(span, start, end)] (clip seconds)."""
    out = []
    for s in spans:
        a, b = max(float(s["start_s"]), lo), min(float(s["end_s"]), hi)
        if b - a >= MIN_OVERLAP_S - 1e-9:
            out.append((s, a, b))
    return out


def window_label_seconds(spans, lo, hi):
    """{label: seconds of that relation inside [lo, hi]}."""
    out = {}
    for s, a, b in clip_spans(spans, lo, hi):
        out[s["label"]] = out.get(s["label"], 0.0) + (b - a)
    return out


def pick_last(last, span_s):
    """The entry of the result's `last` dict ({"1.5": ..., "3": ...}) whose window length is closest to span_s."""
    if not last:
        return None
    return last[min(last, key=lambda k: abs(float(k) - float(span_s)))]


def _name(ids, i):
    if i is not None and i == ids[0]:
        return "person A"
    if i is not None and i == ids[1]:
        return "person B"
    return None


def _span_sentence(s, a, b, ids, lo, inside):
    actor, target = _name(ids, s.get("actor")), _name(ids, s.get("target"))
    if s["label"] == "contact_or_reach" and actor is None:
        text = "a hand reaches the other person's body or the two touch (direction not clear)"
    else:
        text = SPAN_TEXT[s["label"]].format(a=actor or "person A", b=target or "person B")
    d0, d1 = s.get("dist_start_m"), s.get("dist_end_m")
    if inside and d0 is not None and d1 is not None:
        text += f" ({d0:.1f} m -> {d1:.1f} m"
        c = s.get("mean_closing_ms")
        text += (f", closing {c:.1f} m/s)" if c is not None and abs(c) >= 0.15 else ")")
    elif d1 is not None and abs(b - float(s["end_s"])) < 1e-9:
        text += f" (ending {d1:.1f} m apart)"
    return f"From {a - lo:.1f} to {b - lo:.1f} s: {text}."


def describe(res, task, span_s=3.0):
    """Plain-language geometry text of one clip. res: result dict of analyze_clip (or {"interaction": None, "reason": ...} when nothing exists)."""
    inter = res.get("interaction")
    people = res.get("people") or {}
    parts = []
    if not inter:
        parts.append(f"No interacting pair of people could be measured: {NO_PAIR.get(res.get('reason'), 'unknown reason')}.")
        if people.get("max_people"):
            parts.append(f"Up to {int(people['max_people'])} people were visible.")
        return " ".join(parts)
    ids = inter["ids"]
    lo, hi = window_bounds(inter["duration_s"], task, span_s)
    if people.get("max_people"):
        parts.append(f"Up to {int(people['max_people'])} people were visible.")
    if task == "full" and inter.get("pair_visible_s") is not None and inter["pair_visible_s"] < 0.8 * inter["duration_s"]:
        parts.append(f"The two main people are measurable together for {inter['pair_visible_s']:.1f} s of {inter['duration_s']:.1f} s.")
    for s, a, b in clip_spans(inter["spans"], lo, hi):
        if s["label"] in ("apart", "close_unclear") and b - a < 1.0:
            continue
        parts.append(_span_sentence(s, a, b, ids, lo, inside=(a == float(s["start_s"]) and b == float(s["end_s"]))))
    L = pick_last(inter.get("last"), span_s)
    if L and L.get("dist_start_m") is not None:
        t = f"In the last {L['seconds']:.1f} s the two go from {L['dist_start_m']:.1f} m to {L['dist_end_m']:.1f} m apart (closest {L['min_dist_m']:.1f} m)"
        if L.get("actor") is not None:
            t += f"; {_name(ids, L['actor']) or 'one person'} moves toward {_name(ids, L['target']) or 'the other'}"
            if L.get("target_faces_away_deg") is not None and L["target_faces_away_deg"] > 110:
                t += ", who is facing away"
        if (L.get("reach_ms") or 0) >= THRESH["reach_ms"] or (L.get("contact_frac") or 0) > 0.2:
            t += "; a hand reaches the other person's body"
        parts.append(t + ".")
    arm = inter.get("arm_raised_last_s") or {}
    if arm and max(arm.values()) >= 0.3:
        parts.append(f"An arm is raised above the shoulder for {max(arm.values()):.1f} s in the last seconds (pose heuristic).")
    if task == "full":
        secs = {k: v for k, v in (inter.get("label_seconds") or {}).items() if k != "apart"}
        top = sorted(secs.items(), key=lambda kv: -kv[1])[:4]
        if top:
            parts.append("Over the whole clip: " + ", ".join(f"{k.replace('_', ' ')} {v:.1f} s" for k, v in top) + ".")
        if inter.get("min_dist_m") is not None:
            parts.append(f"The closest the two come is {inter['min_dist_m']:.1f} m.")
    if inter.get("leadup"):
        parts.append(f"Rule-based pattern in the final seconds: {inter['leadup']['type']}.")
    return " ".join(parts)
