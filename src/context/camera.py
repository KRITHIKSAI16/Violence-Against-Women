"""Deep context, step 1: camera motion and scene state (how the clip was filmed).

Person velocities only mean something if the camera is still. This module estimates the global image motion between
consecutive frames (sparse Lucas-Kanade optical flow on background corners + a RANSAC similarity transform; the same idea as
BoT-SORT's camera-motion compensation), builds the cumulative transform to frame 0, and reports:

  moving        True if the camera pans / shakes / is hand-held for a large share of the clip
  T             (n,3,3) transform taking frame-t pixel coordinates to frame-0 coordinates (identity when static)
  brightness    mean luma 0-255 (night / dark scenes: the report stresses late-night context)
  sharpness     variance of the Laplacian (blur / low quality)

Person boxes are masked out of the corner detection so walking people do not pull the estimate.
All pixel quantities are in the coordinates of the processed (M2) clip.
"""
import cv2
import numpy as np


def _small(frame, scale):
    g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return cv2.resize(g, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) if scale < 1 else g


def _mask(shape, boxes, scale, pad=0.15):
    m = np.full(shape, 255, np.uint8)
    for x1, y1, x2, y2 in boxes:
        w, h = x2 - x1, y2 - y1
        a, b = int((x1 - pad * w) * scale), int((y1 - pad * h) * scale)
        c, d = int((x2 + pad * w) * scale), int((y2 + pad * h) * scale)
        m[max(b, 0):max(d, 0), max(a, 0):max(c, 0)] = 0
    return m


def estimate_camera(video_path, boxes_by_frame=None, max_side=320):
    """Per-frame frame-to-previous motion and scene stats. boxes_by_frame: {frame: [(x1,y1,x2,y2), ...]} in video pixels."""
    boxes_by_frame = boxes_by_frame or {}
    cap = cv2.VideoCapture(str(video_path))
    ok, frame = cap.read()
    if not ok:
        cap.release()
        raise ValueError(f"cannot read {video_path}")
    h0, w0 = frame.shape[:2]
    scale = min(1.0, max_side / max(h0, w0))
    prev = _small(frame, scale)
    ms = [np.eye(3)]                      # ms[t]: maps frame t-1 coords -> frame t coords (original pixels)
    good_frames = [True]
    lum, sharp = [float(prev.mean())], [float(cv2.Laplacian(prev, cv2.CV_64F).var())]
    t = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        t += 1
        cur = _small(frame, scale)
        M = np.eye(3)
        good = False
        pts = cv2.goodFeaturesToTrack(prev, maxCorners=200, qualityLevel=0.01, minDistance=7,
                                      mask=_mask(prev.shape, boxes_by_frame.get(t - 1, []), scale))
        if pts is not None and len(pts) >= 10:
            nxt, st, _ = cv2.calcOpticalFlowPyrLK(prev, cur, pts, None)
            sel = st.ravel() == 1
            if sel.sum() >= 10:
                A, inl = cv2.estimateAffinePartial2D(pts[sel], nxt[sel], method=cv2.RANSAC, ransacReprojThreshold=1.5)
                if A is not None and inl is not None and inl.sum() >= 8:
                    M = np.vstack([A, [0, 0, 1]])
                    M[:2, 2] /= scale                      # translation back to original pixels
                    good = True
        ms.append(M)
        good_frames.append(good)
        if t % 10 == 0:
            lum.append(float(cur.mean()))
            sharp.append(float(cv2.Laplacian(cur, cv2.CV_64F).var()))
        prev = cur
    cap.release()
    ms = np.array(ms)
    n = len(ms)
    T = np.empty_like(ms)
    T[0] = np.eye(3)
    for k in range(1, n):
        T[k] = T[k - 1] @ np.linalg.inv(ms[k])             # frame k -> frame k-1 -> ... -> frame 0
    return {"M": ms, "T": T, "ok": np.array(good_frames), "size": (w0, h0), "brightness": float(np.mean(lum)),
            "sharpness": float(np.median(sharp))}


def summarize_camera(cam, cfg):
    """Scalars describing the camera: moving flag, drift, jitter, quality."""
    w, h = cam["size"]
    side = float(max(w, h))
    tr = np.linalg.norm(cam["M"][1:, :2, 2], axis=1) / side if len(cam["M"]) > 1 else np.zeros(1)
    drift = float(np.linalg.norm(cam["T"][-1][:2, 2]) / side)
    share = float((tr > cfg["camera_move_frac"]).mean()) if len(tr) else 0.0
    moving = share > cfg["camera_moving_share"] or drift > 0.15
    return {"moving": bool(moving), "moving_share": round(share, 3), "drift_frac": round(drift, 3),
            "median_step_frac": round(float(np.median(tr)), 5), "brightness": round(cam["brightness"], 1),
            "night": bool(cam["brightness"] < cfg["night_brightness"]), "sharpness": round(cam["sharpness"], 1),
            "flow_ok_frac": round(float(cam["ok"].mean()), 3)}


def compensate(T, frames, xy):
    """Map points seen in `frames` (int array) to frame-0 coordinates. xy: (N,2) pixels. T may be None (static)."""
    if T is None:
        return np.asarray(xy, float)
    frames = np.minimum(np.asarray(frames, int), len(T) - 1)
    p = np.c_[np.asarray(xy, float), np.ones(len(frames))]
    out = np.einsum("nij,nj->ni", T[frames], p)
    return out[:, :2]


def boxes_from_tracks(t):
    """{frame: [(x1,y1,x2,y2)]} from an M8a track dict, for masking people out of the camera estimate."""
    d = {}
    for f, b in zip(t["frame_idx"].tolist(), t["bbox"].tolist()):
        d.setdefault(f, []).append(tuple(b))
    return d
