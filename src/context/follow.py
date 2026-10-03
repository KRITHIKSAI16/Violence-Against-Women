"""Deep context, step 4: is one person FOLLOWING another? (lagged path similarity)

Following means the follower retraces the leader's path with a time lag that is unknown and varies (Li et al., "Mining following
relationships in movement data", ICDM 2013). A heading cosine cannot see this, and neither can "they are close": two people walking
side by side are close and aligned but nobody follows. Here, for every candidate lag tau in [lag_min, lag_max]:

    dev(t, tau) = || follower(t) - leader(t - tau) ||           (ground-plane meters)

averaged over a trailing window. The follower follows if, for the best lag,
  * dev is small (follower is where the leader was `tau` seconds ago),
  * dev is clearly smaller than their present separation (so it is retracing, not just being near),
  * the leader really walked (a stationary leader cannot be followed) and so did the follower.
The best lag, the deviation and the roles come out per frame, for both directions.
"""
import numpy as np


def rolling_mean(x, w, min_valid=0.6):
    """Trailing window mean ignoring NaN; NaN where fewer than min_valid of the window is valid."""
    valid = ~np.isnan(x)
    cs = np.concatenate([[0.0], np.cumsum(np.where(valid, x, 0.0))])
    cn = np.concatenate([[0], np.cumsum(valid)])
    idx = np.arange(len(x))
    lo = np.maximum(idx + 1 - w, 0)
    n = cn[idx + 1] - cn[lo]
    with np.errstate(all="ignore"):
        m = (cs[idx + 1] - cs[lo]) / n
    return np.where(n >= min_valid * np.minimum(w, idx + 1), m, np.nan)


def _path_length(P, w):
    """Trailing path length (m) of an (n,2) trajectory over w frames."""
    step = np.linalg.norm(np.diff(P, axis=0, prepend=P[:1]), axis=1)
    step = np.where(np.isnan(step), 0.0, step)
    cs = np.cumsum(step)
    idx = np.arange(len(P))
    return cs - cs[np.maximum(idx - w, 0)]


def follow_scores(a, b, fps, cfg):
    """Does `a` follow `b`?  a, b: (n,2) ground-plane trajectories (X, Z) in meters, NaN when absent.

    Returns {"follow": bool (n,), "lag_s": (n,), "dev_m": (n,), "sep_m": (n,)}.
    """
    n = len(a)
    W = max(2, round(cfg["follow_window_s"] * fps))
    lags = np.arange(max(1, round(cfg["follow_lag_min_s"] * fps)), round(cfg["follow_lag_max_s"] * fps) + 1,
                     max(1, round(0.25 * fps)))
    sep = rolling_mean(np.linalg.norm(a - b, axis=1), W)
    best_dev = np.full(n, np.inf)
    best_lag = np.zeros(n, int)
    for L in lags:
        if L >= n:                     # clip shorter than this lag: nothing to compare
            continue
        shifted = np.vstack([np.full((L, 2), np.nan), b[:-L]])        # leader position L frames ago
        dev = rolling_mean(np.linalg.norm(a - shifted, axis=1), W)
        better = np.nan_to_num(dev, nan=np.inf) < best_dev
        best_dev = np.where(better, np.nan_to_num(dev, nan=np.inf), best_dev)
        best_lag = np.where(better, L, best_lag)
    best_dev = np.where(np.isinf(best_dev), np.nan, best_dev)
    leader_path = _path_length(b, W)
    follower_path = _path_length(a, W)
    with np.errstate(invalid="ignore"):
        follow = ((best_dev <= cfg["follow_max_path_dev_m"]) & (best_dev <= cfg["follow_rel_dev"] * sep)
                  & (leader_path >= cfg["follow_min_leader_path_m"]) & (follower_path >= 1.0)
                  & (sep <= cfg["follow_max_sep_m"]))
    return {"follow": np.nan_to_num(follow, nan=0).astype(bool), "lag_s": best_lag / fps, "dev_m": best_dev, "sep_m": sep}
