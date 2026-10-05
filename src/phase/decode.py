"""Layer 2, step 1: left-to-right constrained decoding of a clip's phase timeline.

Every window of a clip gets per-state probabilities from the model. A clip passes through the phases in one order only:

    BACKGROUND (0)  ->  BUILDUP (1)  ->  ACT (2)          BUILDUP can be skipped (a direct attack), ACT may never occur

so the decoder finds the best path that never goes backwards (Viterbi with forbidden transitions plus a switch penalty). This is the report's
plausibility "grammar" used as a decoder: it turns noisy per-window scores into clean boundaries (`buildup_start_s`, `act_start_s`) and the
"no visible buildup" outcome. It describes what the video already shows; it does not forecast.
"""
import numpy as np

BG, BUILD, ACT = 0, 1, 2
NAMES = ["background", "buildup", "act"]
ALLOWED = {(0, 0), (1, 1), (2, 2), (0, 1), (1, 2), (0, 2)}


def viterbi(logp, switch_penalty=2.0, start_any=True):
    """logp (n,3) log-probabilities -> best monotone state path (n,) of ints. start_any: a clip may begin already in BUILDUP or ACT."""
    logp = np.asarray(logp, float)
    n = len(logp)
    if n == 0:
        return np.zeros(0, int)
    neg = -1e9
    score = np.full((n, 3), neg)
    back = np.zeros((n, 3), int)
    if start_any:
        score[0] = logp[0] - np.array([0.0, switch_penalty, switch_penalty])
    else:
        score[0, 0] = logp[0, 0]
    for t in range(1, n):
        for s in range(3):
            best, arg = neg, s
            for p in range(3):
                if (p, s) not in ALLOWED:
                    continue
                v = score[t - 1, p] - (0.0 if p == s else switch_penalty)
                if v > best:
                    best, arg = v, p
            score[t, s] = best + logp[t, s]
            back[t, s] = arg
    path = np.zeros(n, int)
    path[-1] = int(np.argmax(score[-1]))
    for t in range(n - 1, 0, -1):
        path[t - 1] = back[t, path[t]]
    return path


def boundaries(path, t_centers, min_build_s=1.0):
    """Path + window-centre times (s) -> {act_start_s, buildup_start_s, has_buildup}. A boundary sits half a step before the first window of the new state."""
    path, tc = np.asarray(path), np.asarray(t_centers, float)
    if len(path) == 0:
        return {"act_start_s": None, "buildup_start_s": None, "has_buildup": False}
    half = (tc[1] - tc[0]) / 2 if len(tc) > 1 else 0.0

    def first(state):
        k = np.where(path == state)[0]
        return None if len(k) == 0 else float(max(0.0, tc[k[0]] - half))
    act = first(ACT)
    build = first(BUILD)
    end_build = act if act is not None else float(tc[-1] + half)
    has = build is not None and (end_build - build) >= min_build_s
    return {"act_start_s": act, "buildup_start_s": build if has else None, "has_buildup": bool(has)}


def decode_clip(probs, t_centers, switch_penalty=2.0, min_build_s=1.0, eps=1e-4):
    """probs (n,3) per-window state probabilities -> (path, boundaries dict)."""
    p = np.clip(np.asarray(probs, float), eps, 1.0)
    path = viterbi(np.log(p / p.sum(axis=1, keepdims=True)), switch_penalty)
    return path, boundaries(path, t_centers, min_build_s)
