import numpy as np

from src.perception.shootout import aggregate, clip_metrics, usable_pairs
from src.video.shots import SHOT_BASE as B


def make(rows, fps=10.0):
    """rows: (frame, track_id, height, conf, n_good_kpts)."""
    n = len(rows)
    kp = np.zeros((n, 17, 3), np.float32)
    bb = np.zeros((n, 4), np.float32)
    for k, (f, tid, h, c, g) in enumerate(rows):
        bb[k] = [10, 10, 10 + h / 2, 10 + h]
        kp[k, :g, 2] = 0.9
    return {"frame_idx": np.array([r[0] for r in rows]), "track_id": np.array([r[1] for r in rows]),
            "bbox": bb, "conf": np.array([r[3] for r in rows], np.float32), "kpts": kp, "fps": np.float32(fps)}


def two_people(frames, ids=(1, 2), conf=0.9, h=200, g=15):
    return [(f, i, h, conf, g) for f in range(frames) for i in ids]


def test_pair_needs_one_second_together_and_same_shot():
    assert len(usable_pairs(make(two_people(30)))) == 1                        # 3 s together at 10 fps
    assert usable_pairs(make(two_people(5))) == []                             # 0.5 s only
    assert usable_pairs(make(two_people(30, ids=(1, B + 1)))) == []            # different shots never pair
    assert usable_pairs(make(two_people(30, conf=0.2))) == []                  # ghosts do not make pairs


def test_fragmentation_and_quality_metrics():
    whole = clip_metrics(make(two_people(40)))
    assert whole["ids_per_person"] == 1.0 and whole["long_track"] == 1.0 and whole["kpt_good"] == 1.0
    frag = [(f, 1 if f < 20 else 3, 200, 0.9, 15) for f in range(40)] + [(f, 2, 200, 0.9, 15) for f in range(40)]
    m = clip_metrics(make(frag))
    assert m["ids_per_person"] == 1.5                                          # 3 ids for 2 people
    small = clip_metrics(make(two_people(40, h=50, g=4)))
    assert small["small_share"] == 1.0 and small["kpt_good"] == 0.0
    assert clip_metrics(make(two_people(40, ids=(1, 2), conf=0.2)))["ghost_share"] == 1.0


def test_aggregate_skips_nan_and_means_over_clips():
    a = clip_metrics(make(two_people(40)))
    a["seconds"] = 2.0
    b = {k: float("nan") for k in a}
    b.update(has_pair=0, pairs=0, pair_seconds=0.0, seconds=4.0)
    s = aggregate([a, b])
    assert s["pair_clips"] == 0.5 and s["sec_per_clip"] == 3.0 and s["ids_per_person"] == 1.0
