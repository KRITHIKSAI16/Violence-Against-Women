"""Classification pipelines: the numeric geometry vector of one clip (the same pair facts as the text, as numbers) and the two shortcut baselines.

geometry  pair distances, closing speed, reach, contact, relation seconds, lead-up type, concern scores. Nothing about how the clip was filmed.
style     resolution, fps, camera motion, brightness, sharpness (src.curated.evaluate.style_vector): the shortcut Normal clips are known to carry.
length    the duration of the prepared clip.
A model that does not beat `style` and `length` has learned the source of the clip, not behaviour. Missing values are NaN (the model imputes them and
adds a missing-indicator); a clip without a measurable pair has `no_pair` = 1.
"""
import numpy as np

from src.classcommon.text_build import pick_last, window_bounds, window_label_seconds
from src.curated.evaluate import style_vector
from src.curated.interaction import LEADUP_RULES
from src.curated.relation import LABELS

LAST_FIELDS = ["dist_start_m", "dist_end_m", "net_closing_m", "mean_closing_ms", "min_dist_m", "target_faces_away_deg", "reach_ms", "contact_frac", "intimate_or_personal_frac"]
SCENE = ["no_pair", "max_people", "pair_visible_frac", "clip_min_dist_m", "concern_last_s", "concern_span_mean", "concern_span_max", "arm_raised_s"]


def geom_names():
    return (SCENE + [f"last_{f}" for f in LAST_FIELDS] + [f"rel_{k}_frac" for k in LABELS] + [f"leadup_{i}" for i in range(len(LEADUP_RULES))])


def _f(x):
    return float("nan") if x is None else float(x)


def geom_vector(res, task, span_s=3.0, win_s=2.0):
    """Result dict of analyze_clip -> float vector in the order of geom_names(). Trim: only the last span_s seconds count; full: the whole clip."""
    n = len(geom_names())
    v = dict.fromkeys(geom_names(), float("nan"))
    inter = res.get("interaction")
    v["no_pair"] = 0.0 if inter else 1.0
    people = res.get("people") or {}
    v["max_people"] = _f(people.get("max_people"))
    if inter:
        dur = float(inter["duration_s"])
        lo, hi = window_bounds(dur, task, span_s)
        v["pair_visible_frac"] = _f(inter.get("pair_visible_s")) / dur if dur > 0 and inter.get("pair_visible_s") is not None else float("nan")
        if task == "full":
            v["clip_min_dist_m"] = _f(inter.get("min_dist_m"))               # whole-clip minimum: only meaningful when the whole clip is analysed
        v["concern_last_s"] = _f(inter.get("concern_last_s"))
        w = [s for t, s in inter.get("concern_windows", []) if t + win_s > lo + 1e-9 and t < hi and s is not None]
        if w:
            v["concern_span_mean"], v["concern_span_max"] = float(np.mean(w)), float(np.max(w))
        arm = inter.get("arm_raised_last_s") or {}
        v["arm_raised_s"] = float(max(arm.values())) if arm else float("nan")
        L = pick_last(inter.get("last"), span_s) or {}
        for f in LAST_FIELDS:
            v[f"last_{f}"] = _f(L.get(f))
        secs = window_label_seconds(inter.get("spans", []), lo, hi)
        for k in LABELS:
            v[f"rel_{k}_frac"] = secs.get(k, 0.0) / max(hi - lo, 1e-6)
        typ = (inter.get("leadup") or {}).get("type")
        for i, name in enumerate(LEADUP_RULES):
            v[f"leadup_{i}"] = 1.0 if typ == name else 0.0
    else:
        for i in range(len(LEADUP_RULES)):
            v[f"leadup_{i}"] = 0.0
    return np.array([v[k] for k in geom_names()], float).reshape(n)


def style_vec(res):
    """The six filming-style numbers of the clip (NaN where unknown)."""
    return np.array([float("nan") if x is None else float(x) for x in style_vector(res)], float)


def length_vec(duration_s):
    return np.array([float(duration_s)], float)
