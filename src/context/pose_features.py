"""Deep context, step 3: what the 17 body keypoints say about WHO FACES WHOM, who reaches, who touches.

Until now the pipeline cached 17 keypoints per person and used them only for a centroid. Here they give:

  body facing    shoulder line: if the person's LEFT shoulder appears right of their RIGHT shoulder in the image they face the camera,
                 the reverse means they face away, overlapping shoulders mean sideways (nose offset gives left/right).
  head facing    nose + eyes visible = head toward camera; only ears visible = head away; used to find "looking back".
  contact        wrist-to-torso keypoint distance between two people (meters) while also close on the ground plane.
  reach          a wrist moving fast toward the other person's torso.

Facing vectors live in the ground-plane frame of `ground.py`: X right, Z away from the camera, so
"facing the camera" = (0, -1) and "facing away" = (0, +1). Everything is approximate (monocular 2D pose, no calibration) and every
output carries a confidence so weak evidence can be ignored. Hidden-follower research (AAAI 2024) found gaze / head cues add
information on top of spacing, which is why head facing is computed at all.
"""
import warnings

import numpy as np

NOSE, LEYE, REYE, LEAR, REAR, LSHO, RSHO, LELB, RELB, LWRI, RWRI, LHIP, RHIP = range(13)
SHOULDER_W = 0.23     # shoulder width as a share of box height
CONF = 0.4


def _unit(v):
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return np.where(n > 1e-9, v / np.maximum(n, 1e-9), np.nan)


def body_facing(kp, hb):
    """kp (N,17,3), hb (N,) box heights -> (vec (N,2) [fx, fz], conf (N,)). NaN vec where shoulders are not visible."""
    c = kp[:, :, 2]
    ok = (c[:, LSHO] > CONF) & (c[:, RSHO] > CONF)
    s = np.clip((kp[:, LSHO, 0] - kp[:, RSHO, 0]) / (SHOULDER_W * hb), -1.0, 1.0)
    mid = (kp[:, LSHO, 0] + kp[:, RSHO, 0]) / 2
    nose_off = kp[:, NOSE, 0] - mid
    sx = np.where((c[:, NOSE] > CONF) & (np.abs(nose_off) > 0.02 * hb), np.sign(nose_off), 0.0)
    vec = np.stack([sx * np.sqrt(1 - s ** 2), -s], axis=1)
    conf = np.where(ok, np.where((np.abs(s) > 0.35) | (sx != 0), 0.8, 0.5), 0.0)
    vec[~ok] = np.nan
    return vec, conf


def head_facing(kp, hb):
    """-> (vec (N,2), conf (N,)). Front (face visible) -> toward camera, back (only ears) -> away, else sideways/unknown."""
    c = kp[:, :, 2]
    front = (c[:, NOSE] > 0.5) & (c[:, LEYE] > 0.5) & (c[:, REYE] > 0.5)
    back = (c[:, NOSE] < 0.3) & (c[:, LEYE] < 0.3) & (c[:, REYE] < 0.3) & ((c[:, LEAR] > CONF) | (c[:, REAR] > CONF))
    side = ~front & ~back & ((c[:, NOSE] > CONF) | (c[:, LEAR] > CONF) | (c[:, REAR] > CONF))
    hs = np.where(front, 0.9, np.where(back, -0.9, 0.0))
    mid = (kp[:, LSHO, 0] + kp[:, RSHO, 0]) / 2
    off = kp[:, NOSE, 0] - mid
    hx = np.where((c[:, NOSE] > CONF) & ~front & (np.abs(off) > 0.02 * hb), np.sign(off), 0.0)
    vec = np.stack([hx * np.sqrt(1 - hs ** 2), -hs], axis=1)
    conf = np.where(front | back, 0.8, np.where(side, 0.4, 0.0))
    vec[conf == 0] = np.nan
    return vec, conf


def angle_deg(vec, direction):
    """Angle between facing vectors and directions (both (N,2)); NaN if either is undefined."""
    a, b = _unit(vec), _unit(direction)
    dot = np.clip((a * b).sum(axis=1), -1.0, 1.0)
    return np.degrees(np.arccos(dot))


def runs_of(mask, min_len):
    """Inclusive (start, end) index pairs of True runs at least min_len long."""
    out, start = [], None
    for t, v in enumerate(np.append(np.asarray(mask, bool), False)):
        if v and start is None:
            start = t
        elif not v and start is not None:
            if t - start >= min_len:
                out.append((start, t - 1))
            start = None
    return out


def track_pose(t, tid, n):
    """Per-frame pose arrays of one track over n frames (NaN where absent): body/head facing vectors and confidences,
    wrist and torso keypoints in pixels (NaN where keypoint confidence is low), box height."""
    k = np.where(t["track_id"] == tid)[0]
    fr = t["frame_idx"][k]
    kp = t["kpts"][k]
    hb = np.maximum(t["bbox"][k][:, 3] - t["bbox"][k][:, 1], 1.0)
    bv, bc = body_facing(kp, hb)
    hv, hc = head_facing(kp, hb)

    def put(a, shape, fill=np.nan):
        o = np.full((n,) + shape, fill)
        o[fr] = a
        return o

    def pts(idx):
        p = kp[:, idx, :2].copy()
        p[kp[:, idx, 2] < CONF] = np.nan
        return p

    return {"body": put(bv, (2,)), "body_conf": put(bc, (), 0.0), "head": put(hv, (2,)), "head_conf": put(hc, (), 0.0),
            "wrist": put(np.stack([pts(LWRI), pts(RWRI)], 1), (2, 2)),
            "torso": put(np.stack([pts(LSHO), pts(RSHO), pts(LHIP), pts(RHIP)], 1), (4, 2)),
            "hpx": put(hb, ())}


def _min_dist_px(a, b):
    """a (n,A,2), b (n,B,2) -> (n,) minimum pairwise distance ignoring NaN."""
    d = np.linalg.norm(a[:, :, None, :] - b[:, None, :, :], axis=-1)
    d = d.reshape(len(a), -1)
    with np.errstate(all="ignore"):
        m = np.nanmin(np.where(np.isnan(d), np.inf, d), axis=1)
    return np.where(np.isinf(m), np.nan, m)


def contact_features(pi, pj, hm, vel_frames, fps):
    """Wrist-to-torso distance (m) between the two people and the speed (m/s) of a wrist toward the other's torso.

    pi, pj: track_pose dicts. Returns {wrist_torso_m, reach_i_to_j_ms, reach_j_to_i_ms}.
    """
    h = (pi["hpx"] + pj["hpx"]) / 2
    scale = hm / h                                               # meters per pixel at this pair's depth
    d_ij = _min_dist_px(pi["wrist"], pj["torso"])                # i's wrists vs j's torso
    d_ji = _min_dist_px(pj["wrist"], pi["torso"])
    out = {"wrist_torso_m": np.fmin(d_ij, d_ji) * scale}

    def reach(pa, pb):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)         # frames where the torso keypoints are not visible give NaN, by design
            tc = np.nanmean(pb["torso"], axis=1) if np.isfinite(pb["torso"]).any() else np.full((len(h), 2), np.nan)
        best = np.full(len(h), np.nan)
        for w in (0, 1):
            wr = pa["wrist"][:, w, :]
            prev = np.vstack([np.full((vel_frames, 2), np.nan), wr[:-vel_frames]])
            v = (wr - prev) / (vel_frames / fps)                                    # px/s
            to = tc - wr
            u = to / np.maximum(np.linalg.norm(to, axis=1, keepdims=True), 1e-9)
            toward = (v * u).sum(axis=1) * scale                                    # m/s toward the other's torso
            near = np.linalg.norm(to, axis=1) * scale < 1.0
            toward = np.where(near, toward, np.nan)
            best = np.fmax(best, toward)
        return best

    with np.errstate(all="ignore"):
        out["reach_i_to_j_ms"] = reach(pi, pj)
        out["reach_j_to_i_ms"] = reach(pj, pi)
    return out
