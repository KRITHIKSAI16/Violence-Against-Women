"""Layer 2, step 4: the window table the phase model learns from.

For each clip and each SHOT the most interacting pair (`rank_pairs`) is windowed over the WHOLE clip (the act and what follows are included,
unlike the older learning table that stopped at the cut). One row = one 2 s window (step 0.5 s) of one pair, with

  behavior features   the interpretable pair features of `src.context.windows` (distance, closing, following, facing, contact, reach, flee ...)
  change features     how the window differs from the clip's own past: distance change, speed burst against the previous seconds, the closest
                      distance so far, contact / following seen so far. Phases are about CHANGE inside a clip, so these carry most of the signal.
  scene features      how many people / how isolated.
No style features (camera, resolution, fps, duration) and no absolute time: they would let the model learn how a clip was filmed, not what happens.
Optional extra columns (V-JEPA embedding `emb_*`, pose-sequence `pose_*`) are added by `src.phase.embed` / `src.phase.pose_seq` and merged on (clip_id, pair, t_start).
"""
import numpy as np
import pandas as pd

from src.context.features import rank_pairs
from src.context.windows import BEHAVIOR, SCENE, pair_windows, scene_features
from src.video.shots import shot_of_id

CHANGE = ["d_change", "d_vs_min_so_far", "speed_burst", "speed_vs_start", "contact_so_far", "follow_so_far", "closing_vs_past", "d_min_so_far"]
BASE_FEATURES = BEHAVIOR + SCENE + CHANGE


def key_pair_per_shot(scene):
    """{shot_index: "i_j"}: the highest-evidence pair of every shot that has pairs."""
    by_shot = {}
    for key in scene["pairs"]:
        by_shot.setdefault(shot_of_id(int(key.split("_")[0])), {})[key] = scene["pairs"][key]
    out = {}
    for sh, pairs in by_shot.items():
        top = rank_pairs({"pairs": pairs}, 1)
        if top:
            out[sh] = top[0]
    return out


def add_change_features(df):
    """Adds CHANGE columns, computed per (clip, pair) in time order using only the window itself and EARLIER windows of the same clip."""
    df = df.sort_values(["clip_id", "pair", "t_start"]).reset_index(drop=True)
    g = df.groupby(["clip_id", "pair"], sort=False)
    prev_d = g["d_mean"].shift(2)                                        # 1 s earlier at step 0.5 s
    df["d_change"] = df["d_mean"] - prev_d
    df["d_min_so_far"] = g["d_min"].cummin()
    df["d_vs_min_so_far"] = df["d_mean"] - df["d_min_so_far"]
    med_prev = g["speed_max"].transform(lambda s: s.shift(1).rolling(6, min_periods=2).median())
    df["speed_burst"] = df["speed_max"] - med_prev
    df["speed_vs_start"] = df["speed_max"] - g["speed_max"].transform("first")
    df["contact_so_far"] = g["contact_frac"].cummax()
    df["follow_so_far"] = g["follow_frac"].cummax()
    df["closing_vs_past"] = df["closing_mean"] - g["closing_mean"].transform(lambda s: s.shift(1).rolling(6, min_periods=2).mean())
    return df


def clip_windows(scene, arrays, ctx, win_s, step_s, clip_id="", category=""):
    """Window rows of one clip (key pair of every shot, whole clip). Empty DataFrame if the clip has no pair."""
    rows = []
    fps = scene["fps"]
    sc = scene_features(scene)
    for sh, key in key_pair_per_shot(scene).items():
        i, j = (int(x) for x in key.split("_"))
        valid = np.where(np.isfinite(arrays[(i, j)]["dist_m"]))[0]
        lo, hi = valid[0] / fps, (valid[-1] + 1) / fps               # the span where the pair exists: a window never reaches past it (e.g. over a cut)
        for r in pair_windows(arrays[(i, j)], scene, ctx, win_s, step_s, scene["n_frames"]):
            if r["t_start"] < lo - 1e-6 or r["t_start"] + win_s > hi + 1e-6:
                continue
            r.update(sc)
            r.update({"clip_id": clip_id, "category": category, "pair": key, "shot": sh, "t_center": round(r["t_start"] + win_s / 2, 3)})
            rows.append(r)
    if not rows:
        return pd.DataFrame()
    df = add_change_features(pd.DataFrame(rows))
    df["duration_s"] = scene["duration_s"]
    return df


def build_window_table(clips, ctx_dir, ctx, win_s, step_s, progress=False):
    """All clips with context results -> one DataFrame (columns: BASE_FEATURES + clip_id, category, pair, shot, t_start, t_center, duration_s)."""
    from src.context.features import load_context
    parts = []
    for k, c in enumerate(clips, 1):
        if progress and k % 100 == 0:
            print(f"  phase windows: {k}/{len(clips)} clips", flush=True)
        try:
            scene, arrays = load_context(ctx_dir, c["category"], c["clip_id"])
        except FileNotFoundError:
            continue
        df = clip_windows(scene, arrays, ctx, win_s, step_s, c["clip_id"], c["category"])
        if len(df):
            parts.append(df)
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def feature_columns(df, extra_prefixes=()):
    """BASE_FEATURES present in df plus any column starting with an extra prefix (e.g. 'emb_', 'pose_')."""
    cols = [c for c in BASE_FEATURES if c in df.columns]
    return cols + [c for c in df.columns if any(c.startswith(p) for p in extra_prefixes)]
