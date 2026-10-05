"""Layer 2, step 3: within-clip evaluation against human labels (immune to the 'Normal is a different source' shortcut).

For each clip with a human label we compare the decoded timeline with the human one:
  act-onset error      |predicted act_start - human act_start| (clips where both exist), median and share within 1 s / 2 s; also how many
                       human acts were missed and how many acts were invented
  has-buildup accuracy among clips with a human act: does the clip have a visible buildup (human) vs does the model say so
                       (a direct attack is a valid answer)
  buildup IoU          overlap of the predicted and human buildup intervals (clips where both have one)
Small numbers: always print the clip counts next to every score.
"""
import numpy as np


def _interval(b, end):
    if b.get("buildup_start_s") is None:
        return None
    stop = b.get("act_start_s")
    return (b["buildup_start_s"], stop if stop is not None else end)


def iou(a, b):
    lo, hi = max(a[0], b[0]), min(a[1], b[1])
    inter = max(0.0, hi - lo)
    union = (a[1] - a[0]) + (b[1] - b[0]) - inter
    return inter / union if union > 0 else 0.0


def evaluate(pred, truth, durations=None):
    """pred, truth: {clip_id: {act_start_s, buildup_start_s}} -> dict of scores over the clips present in both."""
    ids = [c for c in truth if c in pred]
    errs, ious, has_t, has_p = [], [], [], []
    missed = invented = 0
    for c in ids:
        t, p = truth[c], pred[c]
        known = [x for x in (t.get("act_start_s"), p.get("act_start_s"), t.get("buildup_start_s"), p.get("buildup_start_s")) if x is not None]
        end = (durations or {}).get(c, (max(known) if known else 0.0) + 1.0)
        if t.get("act_start_s") is not None and p.get("act_start_s") is not None:
            errs.append(abs(t["act_start_s"] - p["act_start_s"]))
        elif t.get("act_start_s") is not None:
            missed += 1
        elif p.get("act_start_s") is not None:
            invented += 1
        if t.get("act_start_s") is not None:                     # buildup questions only make sense for clips with a human act
            has_t.append(t.get("buildup_start_s") is not None)
            has_p.append(p.get("buildup_start_s") is not None)
        it, ip = _interval(t, end), _interval(p, end)
        if it and ip:
            ious.append(iou(it, ip))
    errs = np.array(errs)
    has_t, has_p = np.array(has_t, bool), np.array(has_p, bool)
    return {
        "clips": len(ids), "act_pairs": len(errs), "act_missed": missed, "act_invented": invented,
        "act_err_median_s": float(np.median(errs)) if len(errs) else float("nan"),
        "act_within_1s": float((errs <= 1).mean()) if len(errs) else float("nan"),
        "act_within_2s": float((errs <= 2).mean()) if len(errs) else float("nan"),
        "buildup_clips": int(len(has_t)), "has_buildup_acc": float((has_t == has_p).mean()) if len(has_t) else float("nan"),
        "has_buildup_share_true": float(has_t.mean()) if len(has_t) else float("nan"),
        "buildup_iou_mean": float(np.mean(ious)) if ious else float("nan"), "buildup_iou_clips": len(ious),
    }


def print_scores(s, name=""):
    print(f"{name}clips {s['clips']} | act onset: median error {s['act_err_median_s']:.2f}s, within 1s {s['act_within_1s']:.2f}, within 2s {s['act_within_2s']:.2f} "
          f"(n={s['act_pairs']}, missed {s['act_missed']}, invented {s['act_invented']}) | has-buildup acc {s['has_buildup_acc']:.2f} "
          f"(n={s['buildup_clips']}, true share {s['has_buildup_share_true']:.2f}) | buildup IoU {s['buildup_iou_mean']:.2f} (n={s['buildup_iou_clips']})")
