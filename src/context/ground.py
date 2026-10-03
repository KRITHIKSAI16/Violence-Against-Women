"""Deep context, step 2: pseudo-metric ground-plane geometry (meters instead of pixels).

Monocular trick used in pedestrian metrology: a standing adult is about 1.7 m tall, so a person whose box is h pixels high
stands at depth  Z = f * 1.7 / h  and lateral position  X = (cx - W/2) * 1.7 / h  (meters; f from an assumed field of view).
The distance between two people on this (X, Z) ground plane is in meters, so Edward Hall's proxemic zones
(intimate < 0.46 m, personal < 1.2 m, social < 3.7 m, public < 7.6 m) can be used directly.

Honest limits (flagged, not hidden): needs upright, fully visible people; sitting / cropped / far people have wrong heights
(low `conf`); focal length is an assumption (`fov_deg`); camera tilt and lens distortion are ignored; a moving camera makes Z
unreliable. Depth Anything (Stage B) is the cross-check.
"""
import math

import numpy as np

from src.assm.gate import _fill_gaps, _trailing_mean
from src.context.camera import compensate

HALL_ZONES = [(0.46, "intimate"), (1.2, "personal"), (3.7, "social"), (7.6, "public")]


def focal_px(width, fov_deg):
    return (width / 2.0) / math.tan(math.radians(fov_deg) / 2.0)


def zone_name(d_m):
    """Hall proxemic zone of a distance in meters ('far' beyond 7.6 m, 'unknown' for NaN)."""
    if d_m is None or d_m != d_m:
        return "unknown"
    for lim, name in HALL_ZONES:
        if d_m < lim:
            return name
    return "far"


def zone_index(d_m):
    """0 intimate .. 3 public, 4 far; -1 unknown. Vectorised."""
    d = np.asarray(d_m, float)
    out = np.full(d.shape, 4)
    for k in range(len(HALL_ZONES) - 1, -1, -1):
        out = np.where(d < HALL_ZONES[k][0], k, out)
    return np.where(np.isnan(d), -1, out)


def standing_confidence(bbox, width, height, margin=3):
    """1.0 for a plausibly upright, uncropped person box; 0.3 otherwise (sitting, cropped by the frame edge, odd aspect)."""
    x1, y1, x2, y2 = bbox
    w, h = max(x2 - x1, 1.0), max(y2 - y1, 1.0)
    cropped = y1 <= margin or y2 >= height - margin or x1 <= margin or x2 >= width - margin
    upright = h / w >= 1.6
    return 1.0 if (upright and not cropped) else 0.3


def ground_series(t, cfg, cam_T=None, tids=None):
    """Per track id: smoothed ground-plane arrays over all frames (NaN where absent).

    Returns {tid: {"X","Z","h","cx","cy","conf"}}: X, Z meters; h box height px; cx, cy compensated image centre (frame-0 coords);
    conf standing confidence (0.3 or 1.0, smoothed).
    """
    fps = float(t["fps"])
    n = int(t["n_frames"])
    W, H = float(t["width"]), float(t["height"])
    f = focal_px(W, cfg["fov_deg"])
    hm = cfg["person_height_m"]
    gap = max(1, round(cfg["gap_fill_s"] * fps))
    win = max(1, round(cfg["smooth_s"] * fps))
    out = {}
    for tid in (sorted(set(t["track_id"].tolist())) if tids is None else tids):
        k = np.where(t["track_id"] == tid)[0]
        fr = t["frame_idx"][k]
        b = t["bbox"][k]
        cx = (b[:, 0] + b[:, 2]) / 2
        cy = (b[:, 1] + b[:, 3]) / 2
        h = np.maximum(b[:, 3] - b[:, 1], 1.0)
        cc = compensate(cam_T, fr, np.c_[cx, cy])           # pan/shake removed; no-op for static cameras
        X = (cc[:, 0] - W / 2.0) * hm / h
        Z = f * hm / h
        conf = np.array([standing_confidence(bb, W, H) for bb in b])
        P = np.full((n, 6), np.nan)
        P[fr] = np.c_[X, Z, h, cc[:, 0], cc[:, 1], conf]
        _fill_gaps(P, gap)
        S = _trailing_mean(P, win)
        out[tid] = {"X": S[:, 0], "Z": S[:, 1], "h": S[:, 2], "cx": S[:, 3], "cy": S[:, 4], "conf": S[:, 5]}
    return out


def pair_distance_m(gi, gj):
    """Ground-plane distance (m) between two ground_series entries, per frame."""
    return np.hypot(gi["X"] - gj["X"], gi["Z"] - gj["Z"])


def speed_ms(g, fps, vel_s):
    """Ground-plane speed (m/s) over a trailing window of vel_s seconds."""
    L = max(1, round(vel_s * fps))
    dx = g["X"] - np.r_[[np.nan] * L, g["X"][:-L]]
    dz = g["Z"] - np.r_[[np.nan] * L, g["Z"][:-L]]
    return np.hypot(dx, dz) / (L / fps)
