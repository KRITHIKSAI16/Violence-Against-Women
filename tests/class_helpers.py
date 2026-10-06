"""Synthetic analysis results and records for the classification tests (shaped like the output of src.curated.pipeline.analyze_clip)."""
import numpy as np


def last_facts(seconds, d0, d1, actor=1, target=2, reach=0.0, contact=0.0, faces_away=150.0):
    return {"seconds": seconds, "dist_start_m": d0, "dist_end_m": d1, "net_closing_m": None if d0 is None else round(d0 - d1, 2), "mean_closing_ms": 0.5,
            "min_dist_m": d1, "actor": actor, "target": target, "target_faces_away_deg": faces_away, "reach_ms": reach, "contact_frac": contact,
            "intimate_or_personal_frac": 0.5}


def fake_result(clip_id="Assassination_v1", category="Assassination", dur=10.0, spans=None, pair=True, leadup="approach", reach=0.0, arm=None, max_people=2):
    """A result with a pair: by default one 'follows' span 2-6 s (person 1 follows person 2) and one 'approaches' span 7-10 s."""
    res = {"clip_id": clip_id, "category": category, "duration_s": dur, "people": {"max_people": max_people, "tracks_used": 2},
           "raw": {"raw_width": 640, "raw_height": 360, "raw_fps": 30.0}, "camera": {"moving": False, "brightness": 100.0, "sharpness": 50.0}}
    if not pair:
        res.update({"interaction": None, "no_pair": True, "reason": "never_together"})
        return res
    spans = spans if spans is not None else [
        {"label": "follows", "start_s": 2.0, "end_s": 6.0, "dist_start_m": 4.0, "dist_end_m": 3.0, "mean_closing_ms": 0.3, "actor": 1, "target": 2},
        {"label": "approaches", "start_s": 7.0, "end_s": 10.0, "dist_start_m": 3.0, "dist_end_m": 1.0, "mean_closing_ms": 0.7, "actor": 1, "target": 2}]
    secs = {}
    for s in spans:
        secs[s["label"]] = secs.get(s["label"], 0.0) + s["end_s"] - s["start_s"]
    res["interaction"] = {"ids": [1, 2], "duration_s": dur, "pair_visible_s": dur, "spans": spans, "label_seconds": secs, "first_dist_m": 4.0, "min_dist_m": 1.0,
                          "last": {"1.5": last_facts(1.5, 1.8, 1.0, reach=reach), "3": last_facts(3.0, 3.0, 1.0, reach=reach)}, "leadup": {"type": leadup, "reason": "x"},
                          "concern_last_s": 1.2, "concern_windows": [[float(t), 0.5 + 0.1 * t] for t in range(0, int(dur) - 1)], "concern_mean": 0.8,
                          "concern_max_window": 1.4, "arm_raised_last_s": arm if arm is not None else {"1": 0.6}}
    res.update({"no_pair": False})
    return res


def fake_records(n_violent=14, n_normal=14, seed=3, signal=True):
    """Records with `result`, label y, duration; violent results end in a close approach (signal) or are drawn from the same distribution as Normal ones (no signal)."""
    rng = np.random.default_rng(seed)
    recs = []
    for i in range(n_violent + n_normal):
        viol = i < n_violent
        cat = "Assassination" if viol else "Normal"
        cid = f"{cat}_v{i + 1}"
        close = viol and signal
        spans = [{"label": "approaches" if close else "walking_together", "start_s": 6.0, "end_s": 10.0, "dist_start_m": 4.0 if close else 2.0, "dist_end_m": 1.0 if close else 2.0,
                  "mean_closing_ms": 0.8 if close else 0.0, "actor": 1 if close else None, "target": 2 if close else None}]
        res = fake_result(cid, cat, 10.0, spans, leadup="approach" if close else "walking together", reach=2.0 if close else 0.0)
        res["interaction"]["last"]["3"]["dist_end_m"] = float(1.0 + rng.normal(0, 0.1)) if close else float(2.0 + rng.normal(0, 0.1))
        res["interaction"]["concern_last_s"] = float((1.5 if close else 0.2) + rng.normal(0, 0.1))
        recs.append({"clip_id": cid, "category": cat, "y": 1 if viol else 0, "duration_s": 10.0, "path": f"{cid}.mp4", "raw_width": 640, "raw_height": 360, "result": res})
    return recs
