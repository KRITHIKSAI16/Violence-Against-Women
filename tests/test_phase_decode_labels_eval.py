import numpy as np

from src.phase.decode import ACT, BG, BUILD, boundaries, decode_clip, viterbi
from src.phase.evaluate import evaluate, iou
from src.phase.labels import merge, read_labels, window_targets, write_labels


def centers(n, step=0.5, win=2.0):
    return np.arange(n) * step + win / 2


def noisy_probs(truth, flip_every=0, seed=0):
    p = np.full((len(truth), 3), 0.1)
    p[np.arange(len(truth)), truth] = 0.8
    rng = np.random.default_rng(seed)
    if flip_every:
        for k in range(2, len(truth), flip_every):
            p[k] = [0.1, 0.1, 0.1]
            p[k, rng.integers(0, 3)] = 0.8                            # an isolated wrong window
    return p


def test_targets_follow_the_labels():
    tc = centers(20)
    y = window_targets(tc, 8.0, 5.0)
    assert (y[tc < 5] == BG).all() and (y[(tc >= 5) & (tc < 8)] == BUILD).all() and (y[tc >= 8] == ACT).all()
    assert (window_targets(tc, None, 5.0) == BG).all()                  # no act: nothing to label
    assert BUILD not in window_targets(tc, 8.0, None)                   # direct attack: no buildup


def test_decoder_recovers_known_boundaries_through_noise():
    tc = centers(40)
    truth = window_targets(tc, 12.0, 7.0)
    path, b = decode_clip(noisy_probs(truth, flip_every=5), tc)
    assert np.all(np.diff(path) >= 0)                                     # never goes backwards
    assert abs(b["act_start_s"] - 12.0) <= 1.0 and abs(b["buildup_start_s"] - 7.0) <= 1.0 and b["has_buildup"]


def test_direct_attack_and_no_act_are_valid_outputs():
    tc = centers(40)
    _, b = decode_clip(noisy_probs(window_targets(tc, 10.0, None)), tc)
    assert b["has_buildup"] is False and abs(b["act_start_s"] - 10.0) <= 1.0
    _, b = decode_clip(noisy_probs(window_targets(tc, None, None)), tc)
    assert b["act_start_s"] is None and b["buildup_start_s"] is None


def test_a_flicker_of_buildup_is_not_a_buildup():
    tc = centers(30)
    path = np.zeros(30, int)
    path[10] = BUILD
    path[11:] = ACT
    assert boundaries(path, tc, min_build_s=1.0)["has_buildup"] is False


def test_viterbi_never_decreases_even_with_adversarial_input():
    lp = np.log(np.array([[.1, .1, .8], [.8, .1, .1], [.1, .8, .1]]))
    assert np.all(np.diff(viterbi(lp)) >= 0)


def test_label_files_round_trip_and_human_wins(tmp_path):
    a = {"c1": {"act_start_s": 5.0, "buildup_start_s": None, "source": "vlm", "note": ""},
         "c2": {"act_start_s": None, "buildup_start_s": None, "source": "vlm", "note": ""}}
    h = {"c1": {"act_start_s": 6.5, "buildup_start_s": 3.0, "source": "human", "note": "x"}}
    write_labels(tmp_path / "a.csv", a)
    back = read_labels(tmp_path / "a.csv")
    assert back["c1"]["act_start_s"] == 5.0 and back["c2"]["act_start_s"] is None
    m = merge(back, h)
    assert m["c1"]["act_start_s"] == 6.5 and m["c1"]["source"] == "human" and "c2" in m
    assert merge(h, back)["c1"]["source"] == "human"


def test_evaluation_scores_known_errors():
    truth = {"a": {"act_start_s": 10.0, "buildup_start_s": 5.0}, "b": {"act_start_s": 8.0, "buildup_start_s": None},
             "c": {"act_start_s": 6.0, "buildup_start_s": None}, "d": {"act_start_s": None, "buildup_start_s": None}}
    pred = {"a": {"act_start_s": 10.5, "buildup_start_s": 6.0}, "b": {"act_start_s": 11.0, "buildup_start_s": 5.0},
            "c": {"act_start_s": None, "buildup_start_s": None}, "d": {"act_start_s": 3.0, "buildup_start_s": None}}
    s = evaluate(pred, truth)
    assert s["act_pairs"] == 2 and s["act_missed"] == 1 and s["act_invented"] == 1
    assert s["act_within_1s"] == 0.5 and s["act_within_2s"] == 0.5 and abs(s["act_err_median_s"] - 1.75) < 1e-9
    assert s["buildup_clips"] == 3 and abs(s["has_buildup_acc"] - 2 / 3) < 1e-9        # a ok, b wrong (invented), c ok
    assert abs(iou((5, 10), (6, 10.5)) - 4 / 5.5) < 1e-9
