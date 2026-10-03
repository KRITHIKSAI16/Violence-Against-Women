"""M9 — Selective Activation: behavior states per pair + temporal proposals.

Reads the M8a track cache and describes, for every pair of tracked people, WHAT is happening
and for how long, using only interpretable geometry (no ML, no LLM). Implements the three
heuristics named in Review Report I for M9:

  tracking persistence   -> FOLLOW (one trails the other) and HOVER (loitering next to someone)
  path blocking          -> CORNER (one stands between the other and the nearest frame edge)
  sudden abnormal motion -> ESCALATION (relative-speed burst while close)
  closing speed          -> APPROACH (distance shrinking)

All distances are in BODY HEIGHTS, all speeds in body-heights/second, all durations in seconds
(see the `gate:` block of the config). A state only counts after it has lasted its minimum
duration; short gaps inside a state are bridged (hysteresis) so it does not flicker.

A clip is flagged ("needs deeper analysis") when a pair shows FOLLOW, HOVER or CORNER, or an
APPROACH that ends in ESCALATION. A pure APPROACH alone is reported but never flags a clip:
people walk towards each other all the time.

Outputs per clip (data/gate/<Category>/):
  <clip>_gate.json    segments, proposals, key pair, escalation, ordered-progression flag
  <clip>_states.csv   per-frame features + active state of the key pair (for plots/videos)

Usage:  python -m src.assm.gate [--config CFG] [--clips A B ...]
"""
import argparse
import csv
import json
import logging
from collections import defaultdict
from pathlib import Path

import numpy as np

from src.assm.interaction import blocks_path, centroids, write_csv
from src.assm.track_poses import load_tracks
from src.config import load_config, parse_overrides, resolve_path

log = logging.getLogger(__name__)

APPROACH, FOLLOW, HOVER, CORNER, ESCALATION = "APPROACH", "FOLLOW", "HOVER", "CORNER", "ESCALATION"
RANK = {APPROACH: 1, FOLLOW: 2, HOVER: 2, CORNER: 3, ESCALATION: 4}   # canonical order of buildup
FLAGGING = {FOLLOW, HOVER, CORNER}
STATE_FIELDS = ["frame", "time_s", "d", "closing", "speed_i", "speed_j", "cos", "block", "state"]


# ---------------------------------------------------------------- series building
def _fill_gaps(x, max_gap):
    """Linearly fill NaN gaps of at most `max_gap` frames between valid samples (in place)."""
    valid = np.where(~np.isnan(x[:, 0]))[0]
    for a, b in zip(valid[:-1], valid[1:]):
        gap = b - a - 1
        if 0 < gap <= max_gap:
            for c in range(x.shape[1]):
                x[a + 1:b, c] = np.interp(np.arange(a + 1, b), [a, b], [x[a, c], x[b, c]])


def _trailing_mean(x, w):
    out = np.full_like(x, np.nan)
    for t in np.where(~np.isnan(x[:, 0]))[0]:
        win = x[max(0, t - w + 1):t + 1]
        out[t] = np.nanmean(win, axis=0)
    return out


def track_series(t, cfg):
    """Per track id: smoothed centroid (n,2) in pixels and smoothed body height (n,), NaN when absent."""
    fps = float(t["fps"])
    n = int(t["n_frames"])
    cen = centroids(t, cfg["kpt_conf"])
    bh = (t["bbox"][:, 3] - t["bbox"][:, 1]).clip(min=1.0)
    gap_fill = max(1, round(cfg["gap_fill_s"] * fps))
    w = max(1, round(cfg["smooth_s"] * fps))
    out = {}
    for tid in sorted(set(t["track_id"].tolist())):
        k = np.where(t["track_id"] == tid)[0]
        P = np.full((n, 3), np.nan)
        for kk in k:
            P[t["frame_idx"][kk]] = [cen[kk][0], cen[kk][1], bh[kk]]
        _fill_gaps(P, gap_fill)
        S = _trailing_mean(P, w)
        out[tid] = (S[:, :2], S[:, 2])
    return out


def runs(mask, min_len, max_gap):
    """Maximal True runs of `mask`, bridging False gaps <= max_gap, kept if length >= min_len. Inclusive ends."""
    m = np.asarray(mask, bool).copy()
    idx = np.where(m)[0]
    for a, b in zip(idx[:-1], idx[1:]):
        if 0 < b - a - 1 <= max_gap:
            m[a:b] = True
    out, start = [], None
    for t, v in enumerate(np.append(m, False)):
        if v and start is None:
            start = t
        elif not v and start is not None:
            if t - start >= min_len:
                out.append((start, t - 1))
            start = None
    return out


def _rolling_max(x, w):
    """Max of the last `w` samples (NaN ignored); NaN where the whole window is NaN."""
    out = np.full(len(x), np.nan)
    for t in range(len(x)):
        win = x[max(0, t - w + 1):t + 1]
        if not np.all(np.isnan(win)):
            out[t] = np.nanmax(win)
    return out


def _lag(x, L):
    out = np.full_like(x, np.nan)
    out[L:] = x[:-L] if L > 0 else x
    return out


# ---------------------------------------------------------------- per-pair analysis
def pair_features(Si, hi, Sj, hj, fps, size, cfg, with_block=True):
    """All per-frame features of one pair. Arrays are (n,), NaN where the pair is not co-tracked."""
    n = len(hi)
    h = (hi + hj) / 2
    L = max(1, round(cfg["vel_s"] * fps))
    dt = L / fps
    d = np.linalg.norm(Si - Sj, axis=1) / h
    vi = (Si - _lag(Si, L)) / dt / h[:, None]
    vj = (Sj - _lag(Sj, L)) / dt / h[:, None]
    si, sj = np.linalg.norm(vi, axis=1), np.linalg.norm(vj, axis=1)
    closing = (_lag(d, L) - d) / dt                       # + = approaching
    with np.errstate(invalid="ignore", divide="ignore"):
        cos = (vi * vj).sum(1) / (si * sj)
        u_ij = (Sj - Si) / np.linalg.norm(Sj - Si, axis=1)[:, None]   # from i towards j
        behind_i = (u_ij * vj).sum(1) / sj                # j moves away from i along the line: i trails j
        behind_j = (-u_ij * vi).sum(1) / si
        toward_i = (vi * u_ij).sum(1)                     # speed of i towards j
        toward_j = (vj * -u_ij).sum(1)
    cos = np.where((si >= cfg["move_speed"]) & (sj >= cfg["move_speed"]), cos, np.nan)
    # who blocks whom (blocker stands between the blocked person and the nearest frame edge)
    corridor = cfg["corridor"] * h
    blk_i = np.zeros(n, bool)     # i is blocked by j
    blk_j = np.zeros(n, bool)
    for t in (np.where(~np.isnan(d))[0] if with_block else []):
        blk_i[t] = blocks_path(Si[t], Sj[t], size, corridor[t])
        blk_j[t] = blocks_path(Sj[t], Si[t], size, corridor[t])
    return {"d": d, "closing": closing, "si": si, "sj": sj, "cos": cos, "behind_i": behind_i,
            "behind_j": behind_j, "toward_i": toward_i, "toward_j": toward_j,
            "blk_i": blk_i, "blk_j": blk_j, "vi": vi, "vj": vj}


def _seg(state, i, j, a, b, fps, **extra):
    return {"state": state, "id_i": int(i), "id_j": int(j), "start_f": int(a), "end_f": int(b),
            "start_s": round(a / fps, 3), "end_s": round((b + 1) / fps, 3),
            "dur_s": round((b - a + 1) / fps, 3), **extra}


def pair_segments(i, j, f, fps, cfg, fast=None):
    """Behavior-state segments for one pair from its feature dict `f`."""
    g = lambda s: max(1, round(cfg[s] * fps))
    gap = g("gap_s")
    d, cl = f["d"], f["closing"]
    with np.errstate(invalid="ignore"):
        valid = ~np.isnan(d)
        segs = []

        # APPROACH: distance shrinking for a while, net drop large enough
        raw = valid & (cl > cfg["approach_closing"]) & (d < cfg["approach_max_d"])
        for a, b in runs(raw, g("approach_min_s"), gap):
            drop = float(np.nanmax(d[a:b + 1]) - d[b])
            if drop >= cfg["approach_min_drop"]:
                mover = i if np.nanmean(f["toward_i"][a:b + 1]) >= np.nanmean(f["toward_j"][a:b + 1]) else j
                segs.append(_seg(APPROACH, i, j, a, b, fps, actor=int(mover), drop_bh=round(drop, 2)))

        # FOLLOW: close, both moving, same heading, one consistently behind the other
        for who, behind, a_id, b_id in (("i", f["behind_i"], i, j), ("j", f["behind_j"], j, i)):
            raw = (valid & (d <= cfg["follow_max_d"]) & (f["cos"] >= cfg["follow_cos"])
                   & (behind >= cfg["follow_behind"]))
            for a, b in runs(raw, g("follow_min_s"), gap):
                segs.append(_seg(FOLLOW, i, j, a, b, fps, actor=int(a_id), target=int(b_id),
                                 mean_d=round(float(np.nanmean(d[a:b + 1])), 2)))

        # HOVER: right next to someone who stands still while the other one moves around/loiters
        still = cfg["still_speed"]
        one_still = ((f["si"] < still) & (f["sj"] >= cfg["move_speed"] / 2)) | \
                    ((f["sj"] < still) & (f["si"] >= cfg["move_speed"] / 2))
        raw = valid & (d <= cfg["hover_max_d"]) & one_still
        for a, b in runs(raw, g("hover_min_s"), gap):
            mover = i if np.nanmean(f["si"][a:b + 1]) > np.nanmean(f["sj"][a:b + 1]) else j
            segs.append(_seg(HOVER, i, j, a, b, fps, actor=int(mover), target=int(j if mover == i else i),
                             mean_d=round(float(np.nanmean(d[a:b + 1])), 2)))

        # CORNER: blocker between the blocked person and the frame edge, close, blocked person slow
        act_w = g("corner_activity_s")
        for blk, spd, blocked, blocker, bspd in ((f["blk_i"], f["si"], i, j, f["sj"]),
                                                 (f["blk_j"], f["sj"], j, i, f["si"])):
            # the blocker must have MOVED into position recently; two people simply standing near a wall do not count
            moved = _rolling_max(bspd, act_w) >= cfg["move_speed"]
            raw = valid & blk & moved & (d <= cfg["corner_max_d"]) & (spd < cfg["corner_blocked_speed"])
            for a, b in runs(raw, g("corner_min_s"), gap):
                segs.append(_seg(CORNER, i, j, a, b, fps, actor=int(blocker), target=int(blocked),
                                 mean_d=round(float(np.nanmean(d[a:b + 1])), 2)))

        # ESCALATION: relative-speed burst, close, far above this pair's own normal movement
        ff = fast if fast is not None else f      # short-window velocities: a burst is brief
        rel = np.linalg.norm(ff["vi"] - ff["vj"], axis=1)
        r = rel[valid & ~np.isnan(rel)]
        if len(r) >= 5:
            med = np.median(r)
            mad = np.median(np.abs(r - med)) * 1.4826
            thr = max(cfg["burst_abs"], med + cfg["burst_mad_k"] * mad)
            raw = valid & (rel > thr) & (d <= cfg["burst_max_d"])
            for a, b in runs(raw, g("burst_min_s"), 0):
                segs.append(_seg(ESCALATION, i, j, a, b, fps, peak_rel_speed=round(float(np.nanmax(rel[a:b + 1])), 2)))
    return segs


# ---------------------------------------------------------------- clip level
def _ordered(first_start):
    """True when the distinct phases that occurred started in non-decreasing canonical order (>= 2 ranks)."""
    seq = sorted(first_start.items(), key=lambda kv: kv[1])
    ranks = [RANK[s] for s, _ in seq]
    return len(set(ranks)) >= 2 and all(a <= b for a, b in zip(ranks, ranks[1:]))


def analyze_clip(t, cfg, clip_id="", category=""):
    """Run the whole M9 analysis for one clip's tracks; returns (gate dict, key-pair states rows)."""
    fps = float(t["fps"])
    n = int(t["n_frames"])
    size = (float(t["width"]), float(t["height"]))
    series = track_series(t, cfg)
    fast_series = track_series(t, {**cfg, "smooth_s": cfg["burst_smooth_s"]})
    ids = sorted(series)
    min_co = max(2, round(cfg["min_cotracked_s"] * fps))

    people = np.zeros(n, int)
    for f in t["frame_idx"].tolist():
        people[f] += 1

    all_segs, feats, pair_ids = [], {}, []
    for a in range(len(ids)):
        for b in range(a + 1, len(ids)):
            i, j = ids[a], ids[b]
            Si, hi = series[i]
            Sj, hj = series[j]
            if (~np.isnan(hi) & ~np.isnan(hj)).sum() < min_co:
                continue
            f = pair_features(Si, hi, Sj, hj, fps, size, cfg)
            fast = pair_features(*fast_series[i], *fast_series[j], fps, size,
                                 {**cfg, "vel_s": cfg["burst_vel_s"]}, with_block=False)
            segs = pair_segments(i, j, f, fps, cfg, fast)
            feats[(i, j)] = f
            pair_ids.append((i, j))
            all_segs.extend(segs)

    # evidence per pair -> key pair
    def evidence(p):
        return sum(s["dur_s"] for s in all_segs if (s["id_i"], s["id_j"]) == p and s["state"] != ESCALATION)
    key_pair = max(pair_ids, key=evidence) if pair_ids and any(evidence(p) > 0 for p in pair_ids) else None

    proposals, first_start, esc = [], {}, None
    for p in pair_ids:
        ps = [s for s in all_segs if (s["id_i"], s["id_j"]) == p]
        build = [s for s in ps if s["state"] != ESCALATION]
        escs = sorted((s for s in ps if s["state"] == ESCALATION), key=lambda s: s["start_f"])
        if not build:
            continue
        start = min(s["start_f"] for s in build)
        later_esc = next((s for s in escs if s["start_f"] >= start), None)
        flagging = any(s["state"] in FLAGGING for s in build)
        if not flagging and later_esc is None:
            continue   # pure APPROACH never flags a clip
        end_f = later_esc["start_f"] if later_esc else max(s["end_f"] for s in build)
        reasons = sorted({s["state"] for s in build} | ({ESCALATION} if later_esc else set()),
                         key=lambda s: RANK[s])
        proposals.append({"start_f": int(start), "end_f": int(end_f), "start_s": round(start / fps, 3),
                          "end_s": round((end_f + (0 if later_esc else 1)) / fps, 3), "pair": [int(p[0]), int(p[1])],
                          "reasons": reasons, "ends_in_escalation": later_esc is not None})
        if p == key_pair:
            for s in build + ([later_esc] if later_esc else []):
                if s["state"] not in first_start or s["start_f"] < first_start[s["state"]]:
                    first_start[s["state"]] = s["start_f"]
    proposals.sort(key=lambda p: p["start_f"])

    # Earliest escalation by anyone connected to the buildup pair (or by anyone, if no buildup pair): the video cut
    # uses this, because the act can involve a third person (a second attacker) and we must not show it.
    esc_segs = [s for s in all_segs if s["state"] == ESCALATION]
    if key_pair:
        esc_segs = [s for s in esc_segs if {s["id_i"], s["id_j"]} & set(key_pair)]
    if esc_segs:
        e = min(esc_segs, key=lambda s: s["start_f"])
        esc = {"frame": e["start_f"], "time_s": round(e["start_f"] / fps, 3), "pair": [e["id_i"], e["id_j"]]}
        if first_start and e["start_f"] >= min(first_start.values()):
            first_start.setdefault(ESCALATION, e["start_f"])

    rows = []
    if key_pair is not None:
        f = feats[key_pair]
        ksegs = [s for s in all_segs if (s["id_i"], s["id_j"]) == key_pair]
        label = [""] * n
        for s in sorted(ksegs, key=lambda s: RANK[s["state"]]):   # higher rank overwrites lower
            for fr in range(s["start_f"], s["end_f"] + 1):
                label[fr] = s["state"]
        for fr in range(n):
            if np.isnan(f["d"][fr]):
                continue
            rows.append({"frame": fr, "time_s": round(fr / fps, 4), "d": round(float(f["d"][fr]), 3),
                         "closing": "" if np.isnan(f["closing"][fr]) else round(float(f["closing"][fr]), 3),
                         "speed_i": "" if np.isnan(f["si"][fr]) else round(float(f["si"][fr]), 3),
                         "speed_j": "" if np.isnan(f["sj"][fr]) else round(float(f["sj"][fr]), 3),
                         "cos": "" if np.isnan(f["cos"][fr]) else round(float(f["cos"][fr]), 3),
                         "block": int(f["blk_i"][fr] or f["blk_j"][fr]), "state": label[fr]})

    states_seen = sorted({s["state"] for s in all_segs}, key=lambda s: RANK[s])
    gate = {
        "clip_id": clip_id, "category": category, "fps": fps, "n_frames": n,
        "duration_s": round(n / fps, 3), "n_tracks": len(ids), "n_pairs": len(pair_ids),
        "no_interaction": len(pair_ids) == 0,
        "key_pair": list(map(int, key_pair)) if key_pair else None,
        "flag": len(proposals) > 0,
        "states_seen": states_seen,
        "ordered_progression": _ordered(first_start) if key_pair else False,
        "phase_sequence": [s for s, _ in sorted(first_start.items(), key=lambda kv: kv[1])],
        "escalation": esc,
        "proposals": proposals,
        "segments": sorted(all_segs, key=lambda s: (s["start_f"], RANK[s["state"]])),
        "params": {k: v for k, v in cfg.items() if isinstance(v, (int, float)) and not isinstance(v, bool)},
        "max_people": int(people.max()) if n else 0,
        "isolated_frac": round(float((people[people > 0] == 2).mean()), 3) if (people > 0).any() else 0.0,
    }
    return gate, rows


def write_outputs(gate, rows, out_dir, category, clip_id):
    d = Path(out_dir) / category
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{clip_id}_gate.json").write_text(json.dumps(gate, indent=1), encoding="utf-8")
    write_csv(d / f"{clip_id}_states.csv", STATE_FIELDS, rows)


def write_proposals(gates, out_dir):
    """One file with every temporal proposal in the corpus: the M9 hand-off to Phase II (M10)."""
    allp = [{"clip_id": g["clip_id"], "category": g["category"], "fps": g["fps"], **p}
            for g in gates for p in g["proposals"]]
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    (Path(out_dir) / "proposals.json").write_text(json.dumps(allp, indent=1), encoding="utf-8")


def load_gate(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_states(path):
    with open(path, newline="", encoding="utf-8") as fh:
        out = []
        for r in csv.DictReader(fh):
            r["frame"] = int(r["frame"])
            r["time_s"] = float(r["time_s"])
            out.append(r)
        return out


def run(clean_manifest, cfg_assm, cfg_gate, only=None):
    """Analyze every clip with cached tracks; returns list of gate dicts."""
    tracks_dir = resolve_path(cfg_assm["tracks_dir"])
    out_dir = resolve_path(cfg_gate["gate_dir"])
    gates, missing = [], 0
    for c in clean_manifest["clips"]:
        if only and c["clip_id"] not in only:
            continue
        p = tracks_dir / c["category"] / f"{c['clip_id']}.npz"
        if not p.exists():
            missing += 1
            continue
        t = load_tracks(p)
        t["kpt_conf"] = None
        cfg = {**cfg_gate, "kpt_conf": cfg_assm["kpt_conf"], "corridor": cfg_assm["corridor"]}
        gate, rows = analyze_clip(t, cfg, c["clip_id"], c["category"])
        write_outputs(gate, rows, out_dir, c["category"], c["clip_id"])
        gates.append(gate)
    if missing:
        log.warning("%d clips had no tracks (run M8a first)", missing)
    if not only:   # a partial run must not overwrite the corpus-wide hand-off file
        write_proposals(gates, out_dir)
    return gates


def print_summary(gates):
    cats = defaultdict(list)
    for g in gates:
        cats[g["category"]].append(g)
    print(f"{'category':<16}{'clips':>6}{'no-pair':>9}{'flagged':>9}{'FOLLOW':>8}{'HOVER':>7}{'CORNER':>8}{'ESCAL':>7}{'ordered':>9}")
    for cat, lst in sorted(cats.items()):
        has = lambda s: sum(1 for g in lst if s in g["states_seen"])
        print(f"{cat:<16}{len(lst):>6}{sum(g['no_interaction'] for g in lst):>9}{sum(g['flag'] for g in lst):>9}"
              f"{has(FOLLOW):>8}{has(HOVER):>7}{has(CORNER):>8}{has(ESCALATION):>7}"
              f"{sum(g['ordered_progression'] for g in lst):>9}")


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    ap = argparse.ArgumentParser(description="M9: behavior states + proposals")
    ap.add_argument("--config", default=None)
    ap.add_argument("--clips", nargs="*", default=None, help="only these clip ids")
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE",
                    help="override gate thresholds for this run, e.g. --set follow_min_s=3 hover_max_d=2")
    args = ap.parse_args()
    cfg = load_config(args.config)
    over = parse_overrides(args.set)
    cfg["gate"] = {**cfg["gate"], **over}
    if over:
        print(f"NOTE: thresholds overridden for this run: {over}")
        print(f"      outputs in {cfg['gate']['gate_dir']} now use them (recorded in each *_gate.json under 'params').")
        print("      Run again without --set to restore the config defaults.\n")
    clean = json.loads(resolve_path(cfg["preprocess"]["clean_manifest_path"]).read_text("utf-8"))
    gates = run(clean, cfg["assm"], cfg["gate"], set(args.clips) if args.clips else None)
    print(f"Analyzed {len(gates)} clips -> {resolve_path(cfg['gate']['gate_dir'])}\n")
    print_summary(gates)
    print("\nPer clip:")
    for g in gates:
        phases = " > ".join(g["phase_sequence"]) or "-"
        print(f"  {g['clip_id']:<20} pairs={g['n_pairs']:<3} flag={str(g['flag']):<5} phases: {phases}")


if __name__ == "__main__":
    main()
