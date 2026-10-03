"""Deep context, Stage C: the buildup story as plain sentences (deterministic templates, no language model).

Turns the episodes of a clip's key pair into a timed list of sentences plus a scene line and a summary. Every sentence names the people by
their track ids ("id3"), gives times in seconds and approximate meters, and says when a cue is a benign one (people talking) or unconfirmed
(different depth, low confidence). The text is generated from the data only; nothing is added that the detectors did not find.
"""
from src.context.graph import ACT_PREDS, BENIGN_PREDS


def _who(n):
    return f"id{n}"


def episode_text(ep):
    """One sentence for one episode."""
    a, t, d, p = _who(ep["actor"]), _who(ep["target"]), ep["detail"], ep["pred"]
    m = lambda k: d.get(k)
    if p == "approaches_from_behind":
        s = f"{a} approaches {t} from behind ({m('dist_start'):.1f} m -> {m('dist_end'):.1f} m)"
    elif p == "approaches_frontal":
        s = f"{a} walks toward {t}, who is facing them ({m('dist_start'):.1f} m -> {m('dist_end'):.1f} m)"
    elif p == "approaches_side":
        s = f"{a} moves toward {t} ({m('dist_start'):.1f} m -> {m('dist_end'):.1f} m)"
    elif p == "follows":
        s = f"{a} follows {t} along the same path, about {m('lag_s'):.1f} s behind (average gap {m('mean_dist'):.1f} m)"
    elif p == "looks_back":
        s = f"{a} looks back toward {t} while walking away"
    elif p == "hovers_near":
        s = f"{a} stays within {m('mean_dist'):.1f} m of {t}, who is standing still"
    elif p == "very_close":
        s = f"{a} and {t} are within arm's reach ({m('mean_dist'):.1f} m)"
    elif p == "mutual_facing":
        s = f"{a} and {t} face each other at conversational distance ({m('mean_dist'):.1f} m) - a benign cue"
    elif p == "blocks_exit":
        s = f"{a} stands between {t} and a door"
    elif p == "pinned_against":
        s = f"{t} is against a wall or vehicle with {a} in front (possible)"
    elif p == "reaches_for":
        s = f"{a}'s hand moves fast toward {t}'s body (peak {m('peak_ms'):.1f} m/s)"
    elif p == "contact":
        s = f"hand-to-body contact between {a} and {t} (closest {m('wrist_torso_m'):.2f} m)"
    elif p == "flees_from":
        s = f"{a} moves away fast from {t} ({m('speed_ms'):.1f} m/s)"
    elif p == "target_stops":
        s = f"{t} stops walking (from {m('speed_before'):.1f} m/s to {m('speed_after'):.1f} m/s) after {a} comes closer"
    elif p == "target_speeds_up":
        s = f"{t} speeds up (from {m('speed_before'):.1f} m/s to {m('speed_after'):.1f} m/s) after {a} comes closer"
    else:
        s = f"{a} {p} {t}"
    notes = []
    if d.get("depth") == "different depth":
        notes.append("unconfirmed: the two are at different depths")
    elif d.get("depth") == "same depth":
        notes.append("confirmed at the same depth")
    if ep["conf"] < 0.5:
        notes.append("low confidence")
    return s + (f" [{'; '.join(notes)}]" if notes else "")


def scene_line(facts):
    place = facts["place_type"]
    if place != "unknown" and not facts["layout_reliable"]:
        place += " (uncertain)"
    parts = [f"{place} scene" if place != "unknown" else "scene type unknown", "low-light scene (night or dark)" if facts["low_light"] else "normally lit scene",
             "moving camera (speeds less reliable)" if facts["camera_moving"] else "fixed camera",
             f"up to {facts['max_people']} people in view", f"only two people in view {facts['isolated_frac']:.0%} of the time"]
    if facts.get("doors"):
        parts.append(f"{facts['doors']} door(s) detected")
    a = facts["assumptions"]
    return "Scene: " + ", ".join(parts) + f". Distances are approximate meters (assumed {a['fov_deg']:.0f} degree field of view, {a['person_height_m']} m people)."


def summary_line(story):
    if story["no_pair"]:
        return "Summary: fewer than two people were tracked together long enough, so no interaction could be described."
    kp = story["pairs"][story["key_pair"]]
    eps = kp["episodes"]
    if not eps:
        return f"Summary: the main pair (id{kp['pair'][0]}, id{kp['pair'][1]}) shows no approach, following, lingering or contact cues."
    concern = sorted({e["pred"].replace("_", " ") for e in eps if e["pred"] not in BENIGN_PREDS})
    benign = sorted({e["pred"].replace("_", " ") for e in eps if e["pred"] in BENIGN_PREDS})
    s = f"Summary: concern cues {kp['concern_seconds']:.1f} s ({', '.join(concern) or 'none'}); benign cues {kp['benign_seconds']:.1f} s ({', '.join(benign) or 'none'})."
    if kp["phases"]:
        s += f" Phases in order: {' -> '.join(kp['phases'])}" + (" (escalating order)." if kp["ordered_progression"] else ".")
    esc = story.get("escalation")
    if esc:
        s += f" First physical-act cue: {esc['kind'].replace('_', ' ')} at {esc['time_s']:.1f} s."
    return s


def narrate(story, max_lines=40):
    """-> {"scene": str, "lines": [{"t0","t1","pred","text"}], "summary": str} for the key pair of a clip story."""
    out = {"scene": scene_line(story["scene"]), "summary": summary_line(story), "lines": []}
    if story["no_pair"]:
        return out
    for ep in story["pairs"][story["key_pair"]]["episodes"][:max_lines]:
        out["lines"].append({"t0": ep["start_s"], "t1": ep["end_s"], "pred": ep["pred"], "act": ep["pred"] in ACT_PREDS, "text": episode_text(ep)})
    return out


def narrative_markdown(story):
    n = narrate(story)
    lines = [f"**{story['clip_id']}** ({story['category']})", "", n["scene"], ""]
    lines += [f"- {l['t0']:.1f}-{l['t1']:.1f} s: {l['text']}" for l in n["lines"]] or ["- (no episodes)"]
    return "\n".join(lines + ["", n["summary"]])
