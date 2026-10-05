import numpy as np
import pandas as pd

from src.config import load_config
from src.context.features import analyze_context
from src.phase.dataset import BASE_FEATURES, add_change_features, clip_windows, feature_columns, key_pair_per_shot
from src.phase.evaluate import evaluate
from src.phase.labels import window_targets
from src.phase.model import add_targets, decode_table, fit, grouped_cv, predict_probs
from src.video.shots import SHOT_BASE
from tests.test_context_graph import two_shot_tracks

CTX = load_config()["context"]


def test_windows_cover_the_whole_clip_and_use_one_key_pair_per_shot():
    t = two_shot_tracks()
    sc, arr = analyze_context(t, CTX)
    kp = key_pair_per_shot(sc)
    assert kp[1] == f"{SHOT_BASE + 1}_{SHOT_BASE + 2}" and kp[0] == "1_2"                      # one key pair per shot
    df = clip_windows(sc, arr, CTX, 2.0, 0.5, "edit", "Stalking")
    cut_s = 100 / 30.0
    assert set(df["shot"]) == {0, 1}
    assert (df[df["shot"] == 0]["t_start"] + 2.0 <= cut_s + 0.05).all() and (df[df["shot"] == 1]["t_start"] >= cut_s - 0.05).all()   # no window spans the cut
    assert df["t_center"].max() > 10.0                                                          # runs to the end, no cut trimming
    assert not any(c in df.columns and c in BASE_FEATURES for c in ("cam_moving", "raw_fps", "duration_s"))   # no style features among the model inputs
    assert "t_center" not in feature_columns(df)


def test_change_features_use_only_the_past():
    n = 12
    base = {c: 0.0 for c in ["d_mean", "d_min", "speed_max", "contact_frac", "follow_frac", "closing_mean"]}
    df = pd.DataFrame([{**base, "clip_id": "c", "pair": "1_2", "t_start": 0.5 * k, "d_mean": 5.0 - 0.2 * k, "d_min": 5.0 - 0.2 * k,
                        "speed_max": 1.0 if k < 8 else 4.0, "contact_frac": 1.0 if k == 10 else 0.0} for k in range(n)])
    a = add_change_features(df)
    assert a.loc[5, "d_change"] < 0 and np.isnan(a.loc[0, "d_change"])
    assert a.loc[8, "speed_burst"] > 2.0 and abs(a.loc[5, "speed_burst"]) < 1e-9                  # burst only when speed jumps
    assert a.loc[9, "contact_so_far"] == 0.0 and a.loc[11, "contact_so_far"] == 1.0              # remembered after it happens
    b = add_change_features(df.iloc[:6])
    assert np.allclose(a.loc[:5, "d_min_so_far"], b["d_min_so_far"])                              # later windows never change earlier values


def synthetic_clips(n_clips=14, seed=0):
    """Clips where a buildup (distance shrinking, following) precedes an act (speed burst + contact) at a known time."""
    rng = np.random.default_rng(seed)
    rows, labels = [], {}
    for c in range(n_clips):
        act = float(rng.uniform(9, 13))
        build = act - float(rng.uniform(3, 5)) if c % 3 else None                                # every third clip is a direct attack
        labels[f"c{c}"] = {"act_start_s": act, "buildup_start_s": build}
        for k in range(40):
            tc = 1.0 + 0.5 * k
            in_build = build is not None and build <= tc < act
            in_act = tc >= act
            rows.append({"clip_id": f"c{c}", "pair": "1_2", "t_start": tc - 1.0, "t_center": tc, "shot": 0,
                         "d_mean": 3.0 - (1.2 if in_build else 0) - (2.0 if in_act else 0) + rng.normal(0, .3),
                         "follow_frac": (0.8 if in_build else 0.0) + rng.normal(0, .1),
                         "speed_max": (3.5 if in_act else 1.0) + rng.normal(0, .3),
                         "contact_frac": (0.7 if in_act else 0.0) + abs(rng.normal(0, .05))})
    return pd.DataFrame(rows), labels


def test_phase_model_recovers_planted_onsets_on_held_out_clips():
    df, labels = synthetic_clips()
    cols = ["d_mean", "follow_frac", "speed_max", "contact_frac"]
    df = add_targets(df, labels)
    assert set(df["target"]) == {0, 1, 2}
    train, test = df[~df["clip_id"].isin(["c1", "c2", "c4", "c5"])], df[df["clip_id"].isin(["c1", "c2", "c4", "c5"])]
    clf = fit(train, cols)
    pred = decode_table(test, predict_probs(clf, test, cols))
    s = evaluate(pred, {c: labels[c] for c in pred})
    assert s["act_pairs"] == 4 and s["act_within_1s"] == 1.0 and s["act_missed"] == 0
    # c1, c2, c4, c5: c3-type direct attacks are c0, c3 ... clip 3k has no buildup; c1,c2,c4,c5 all have one
    assert s["has_buildup_acc"] >= 0.75


def test_grouped_cv_never_mixes_a_clip_across_folds_and_a_direct_attack_has_no_buildup():
    df, labels = synthetic_clips()
    cols = ["d_mean", "follow_frac", "speed_max", "contact_frac"]
    df = add_targets(df, labels)
    probs = grouped_cv(df, cols, folds=4)
    assert probs.shape == (len(df), 3) and np.allclose(probs.sum(axis=1), 1.0)
    pred = decode_table(df, probs)
    direct = [c for c in labels if labels[c]["buildup_start_s"] is None]
    assert sum(pred[c]["has_buildup"] for c in direct) <= len(direct) // 3 + 1                    # direct attacks are mostly recognised as such
    s = evaluate(pred, labels)
    assert s["act_within_2s"] >= 0.9


def test_missing_class_in_training_still_gives_valid_probabilities():
    df, labels = synthetic_clips(6)
    labels = {c: {"act_start_s": v["act_start_s"], "buildup_start_s": None} for c, v in labels.items()}     # no buildup anywhere
    df = add_targets(df, labels)
    cols = ["d_mean", "speed_max"]
    p = predict_probs(fit(df, cols), df, cols)
    assert p.shape == (len(df), 3) and p[:, 1].max() < 0.01 and np.allclose(p.sum(axis=1), 1.0)
