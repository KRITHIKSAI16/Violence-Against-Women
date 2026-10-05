import json
import sys

import pandas as pd

from src.phase.labels import write_labels
from src.phase.run import baseline_predictions, main
from tests.test_phase_dataset_model import synthetic_clips


def test_baselines_read_the_gate_and_use_a_fixed_fraction(tmp_path):
    d = tmp_path / "Stalking"
    d.mkdir()
    (d / "c1_gate.json").write_text(json.dumps({"escalation": {"time_s": 7.5}}), encoding="utf-8")
    (d / "c2_gate.json").write_text(json.dumps({"escalation": None}), encoding="utf-8")
    clips = [{"clip_id": "c1", "category": "Stalking"}, {"clip_id": "c2", "category": "Stalking"}, {"clip_id": "c3", "category": "Stalking"}]
    rule, fixed = baseline_predictions(clips, tmp_path, {"c1": 20.0, "c2": 10.0, "c3": 5.0})
    assert rule["c1"]["act_start_s"] == 7.5 and rule["c2"]["act_start_s"] is None and rule["c3"]["act_start_s"] is None
    assert fixed["c2"]["act_start_s"] == 6.0


def test_run_trains_on_pseudo_labels_and_scores_against_human(tmp_path, monkeypatch, capsys):
    df, labels = synthetic_clips(16)
    ids = sorted(labels)
    hold = ids[:4]
    cfg = {"preprocess": {"clean_manifest_path": str(tmp_path / "clean.json")}, "phase": {"dir": str(tmp_path / "phase"), "win_s": 2.0, "step_s": 0.5, "switch_penalty": 2.0,
           "min_build_s": 1.0, "folds": 4, "extra_prefixes": []}, "label": {"vlm_labels": str(tmp_path / "vlm.csv"), "human_labels": str(tmp_path / "human.csv")},
           "perception": {"holdout_file": str(tmp_path / "hold.txt")}, "gate": {"gate_dir": str(tmp_path / "gate")}, "context": {"context_dir": str(tmp_path / "ctx")}}
    (tmp_path / "cfg.yaml").write_text(json.dumps(cfg), encoding="utf-8")                       # JSON is valid YAML
    (tmp_path / "clean.json").write_text(json.dumps({"clips": [{"clip_id": c, "category": "Stalking", "duration_s": 22.0} for c in ids]}), encoding="utf-8")
    (tmp_path / "hold.txt").write_text("# held out\n" + "\n".join(hold) + "\n", encoding="utf-8")
    (tmp_path / "phase").mkdir()
    for c in ["d_min", "closing_mean", "d_net_closing"]:
        df[c] = 0.0
    df["scene_max_people"] = 2.0
    for c in ids:                                                                                # a gate file per clip so the baseline can be read
        (tmp_path / "gate" / "Stalking").mkdir(parents=True, exist_ok=True)
        (tmp_path / "gate" / "Stalking" / f"{c}_gate.json").write_text(json.dumps({"escalation": None}), encoding="utf-8")
    from src.phase.dataset import add_change_features
    df["category"] = "Stalking"
    df["duration_s"] = 22.0
    for c in ["follow_frac", "speed_max", "contact_frac", "d_mean"]:
        assert c in df.columns
    for c in ["d_min", "closing_mean"]:
        assert c in df.columns
    add_change_features(df).to_pickle(tmp_path / "phase" / "windows.pkl")
    write_labels(tmp_path / "vlm.csv", {c: {**labels[c], "source": "vlm", "note": "agree"} for c in ids})
    write_labels(tmp_path / "human.csv", {c: {**labels[c], "source": "human", "note": ""} for c in hold})
    monkeypatch.setattr(sys, "argv", ["run", "--config", str(tmp_path / "cfg.yaml")])
    main()
    out = capsys.readouterr().out
    assert "phase model" in out and "rule gate" in out and "fixed 60%" in out and "PSEUDO" in out
    line = [l for l in out.splitlines() if "phase model" in l][0]
    assert "within 2s 1.00" in line                                                             # the planted act onsets are recovered on the held-out clips
    assert list((tmp_path / "phase" / "pred" / "Stalking").glob("*_phase.json"))
