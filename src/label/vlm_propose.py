"""Layer 3: automatic temporal labels (pseudo-labels) from a vision-language model. An OFFLINE labeling aid, never part of the deliverable output.

For every violent clip a VLM (Qwen3-VL by default) is shown time-stamped frames and asked two things: when does physical violence begin
(`act_start_s`) and when does the aggressor first start approaching / following / threatening the victim (`buildup_start_s`). It is asked
TWICE with different frame sampling and wording. A clip becomes a training label only when both answers agree within `agree_s` seconds
(about "no act" / "no buildup" they must both say none); everything else is kept as `disagree` and left out of training.

These labels are proposals. Their accuracy is unknown until a person labels a small validation set (`src/label/annotate_pack.py`) and
`src.label.vlm_propose --score` compares them (act within 1 s / 2 s). Nothing here is shown in the report.

    python -m src.label.vlm_propose [--config CFG] [--clips A B] [--model ID] [--force]
    python -m src.label.vlm_propose --score data/labels/human_labels.csv
"""
import argparse
import json
import logging
import re
from pathlib import Path

import cv2
import numpy as np

from src.config import load_config, resolve_path
from src.phase.labels import FIELDS, read_labels, write_labels
from src.video.shots import load_shots

log = logging.getLogger(__name__)

PROMPTS = [
    ("You are given frames of a surveillance or phone video, each preceded by its time in seconds. "
     "Answer about this video only. (1) At what time does physical violence FIRST begin (hitting, grabbing by force, choking, shooting, stabbing, "
     "forced dragging, snatching with force)? (2) At what earlier time does the aggressor first start approaching, following or confronting "
     "the victim in a way that leads to that violence? Use null when there is none (for example a sudden attack with no lead-up, or no violence at all). "
     'Reply with JSON only: {"act_start_s": number or null, "buildup_start_s": number or null}'),
    ("Look at the time-stamped frames. Find the moment the first violent physical contact or weapon use happens, and separately the moment "
     "a person starts to chase, follow, corner or close in on another person before that. Give seconds as numbers; write null if it never happens "
     'or if there was no lead-up. Output only this JSON: {"act_start_s": ..., "buildup_start_s": ...}'),
]


# ---------------------------------------------------------------- pure helpers
def sample_times(duration_s, fps_sample, max_frames, offset_s=0.0, start_s=0.0):
    """Frame times (s) for one pass: every 1/fps_sample seconds from start_s+offset, thinned evenly to at most max_frames."""
    step = 1.0 / fps_sample
    t = np.arange(start_s + offset_s, max(duration_s - 1e-6, start_s + offset_s + 1e-6), step)
    if len(t) > max_frames:
        t = t[np.linspace(0, len(t) - 1, max_frames).round().astype(int)]
    return [float(x) for x in t]


def parse_answer(text):
    """Model text -> {act_start_s, buildup_start_s} (numbers or None), or None if no usable JSON-like answer."""
    if not text:
        return None
    m = re.search(r"\{[^{}]*\}", text, re.S)
    if m:
        try:
            d = json.loads(m.group(0))
            return {"act_start_s": _num(d.get("act_start_s")), "buildup_start_s": _num(d.get("buildup_start_s"))}
        except (ValueError, AttributeError):
            pass
    a = re.search(r"act_start_s\W+(null|none|[\d.]+)", text, re.I)
    b = re.search(r"buildup_start_s\W+(null|none|[\d.]+)", text, re.I)
    if a and b:
        return {"act_start_s": _num(a.group(1)), "buildup_start_s": _num(b.group(1))}
    return None


def _num(x):
    if x is None:
        return None
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if v >= 0 and np.isfinite(v) else None


def _close(a, b, tol):
    if a is None or b is None:
        return a is None and b is None
    return abs(a - b) <= tol


def combine(a, b, duration_s, agree_s=1.5, min_build_s=1.0):
    """Two answers -> (label dict, status). status: 'agree' | 'disagree' | 'invalid'.
    Times outside the clip, or a buildup that does not precede the act, make an answer invalid."""
    def ok(x):
        if x is None:
            return False
        for k in ("act_start_s", "buildup_start_s"):
            if x[k] is not None and x[k] > duration_s + 0.5:
                return False
        if x["buildup_start_s"] is not None and x["act_start_s"] is not None and x["act_start_s"] - x["buildup_start_s"] < min_build_s:
            x["buildup_start_s"] = None                     # a lead-up shorter than min_build_s is the same as none
        return True
    if not ok(a) or not ok(b):
        return None, "invalid"
    if _close(a["act_start_s"], b["act_start_s"], agree_s) and _close(a["buildup_start_s"], b["buildup_start_s"], agree_s):
        def mean(p, q):
            return None if p is None else float(round((p + q) / 2, 2))
        return {"act_start_s": mean(a["act_start_s"], b["act_start_s"]), "buildup_start_s": mean(a["buildup_start_s"], b["buildup_start_s"])}, "agree"
    return None, "disagree"


def score_against_human(proposals, human, tol=(1.0, 2.0)):
    """How good are the proposals? proposals: {clip: {act_start_s, buildup_start_s, status}}, human: read_labels() output.
    Reports, over clips with a human act: share with an answer, and among those the share within tol of the human act time."""
    ids = [c for c in human if c in proposals and human[c]["source"] == "human" and human[c]["act_start_s"] is not None]
    got = [c for c in ids if proposals[c]["status"] == "agree" and proposals[c]["act_start_s"] is not None]
    errs = np.array([abs(proposals[c]["act_start_s"] - human[c]["act_start_s"]) for c in got])
    out = {"human_clips": len(ids), "with_agreed_answer": len(got)}
    for t in tol:
        out[f"within_{t:g}s"] = float((errs <= t).mean()) if len(errs) else float("nan")
    out["median_err_s"] = float(np.median(errs)) if len(errs) else float("nan")
    return out


# ---------------------------------------------------------------- frames and model
def read_frames(video_path, times, short_side=336):
    """[(t, RGB uint8 array)] at the given times, resized so the short side is `short_side`."""
    cap = cv2.VideoCapture(str(video_path))
    out = []
    for t in times:
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000.0)
        ok, img = cap.read()
        if not ok:
            continue
        h, w = img.shape[:2]
        s = short_side / min(h, w)
        if s < 1:
            img = cv2.resize(img, (int(w * s) // 2 * 2, int(h * s) // 2 * 2), interpolation=cv2.INTER_AREA)
        out.append((t, cv2.cvtColor(img, cv2.COLOR_BGR2RGB)))
    cap.release()
    return out


class QwenAsker:
    """Qwen3-VL via transformers (4-bit when bitsandbytes is available). Needs a GPU; not exercised by the unit tests."""

    def __init__(self, model_id, load_4bit=True, max_new_tokens=60):
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor
        kw = {"device_map": "auto"}
        if load_4bit:
            from transformers import BitsAndBytesConfig
            kw["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
        else:
            kw["torch_dtype"] = torch.float16
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModelForImageTextToText.from_pretrained(model_id, **kw)
        self.max_new_tokens = max_new_tokens

    def __call__(self, frames, prompt):
        from PIL import Image
        content = []
        for t, img in frames:
            content.append({"type": "text", "text": f"t={t:.1f}s"})
            content.append({"type": "image", "image": Image.fromarray(img)})
        content.append({"type": "text", "text": prompt})
        inputs = self.processor.apply_chat_template([{"role": "user", "content": content}], add_generation_prompt=True, tokenize=True,
                                                    return_dict=True, return_tensors="pt").to(self.model.device)
        out = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens, do_sample=False)
        return self.processor.batch_decode(out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0]


# ---------------------------------------------------------------- one clip, whole run
def propose_clip(video_path, duration_s, ask, lcfg, shot_windows=None):
    """Two passes (different sampling and wording). Returns {answers, status, act_start_s, buildup_start_s}."""
    passes = [(float(lcfg["fps_a"]), 0.0, PROMPTS[0]), (float(lcfg["fps_b"]), 0.5 / float(lcfg["fps_b"]), PROMPTS[1])]
    answers = []
    for fps_s, off, prompt in passes:
        times = sample_times(duration_s, fps_s, int(lcfg["max_frames"]), off)
        text = ask(read_frames(video_path, times, int(lcfg["short_side"])), prompt)
        answers.append(parse_answer(text))
    lab, status = combine(answers[0], answers[1], duration_s, float(lcfg["agree_s"]), float(lcfg["min_build_s"]))
    return {"answers": answers, "status": status, "act_start_s": (lab or {}).get("act_start_s"), "buildup_start_s": (lab or {}).get("buildup_start_s")}


def load_proposals(vlm_dir, clip_ids):
    out = {}
    for c in clip_ids:
        for p in Path(vlm_dir).glob(f"*/{c}.json"):
            out[c] = json.loads(p.read_text("utf-8"))
    return out


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    ap = argparse.ArgumentParser(description="Layer 3: VLM pseudo-labels (act start / buildup start) with a two-pass agreement filter")
    ap.add_argument("--config", default=None)
    ap.add_argument("--clips", nargs="*", default=None)
    ap.add_argument("--model", default=None)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--score", default=None, help="human labels CSV: report how often the proposals match it, then exit")
    args = ap.parse_args()
    cfg = load_config(args.config)
    lc = cfg["label"]
    vlm_dir = resolve_path(lc["vlm_dir"])
    clean = json.loads(resolve_path(cfg["preprocess"]["clean_manifest_path"]).read_text("utf-8"))
    clips = [c for c in clean["clips"] if c["category"] != "Normal" and (not args.clips or c["clip_id"] in args.clips)]
    if args.score:
        props = load_proposals(vlm_dir, [c["clip_id"] for c in clean["clips"]])
        s = score_against_human(props, read_labels(args.score))
        print(f"VLM proposals vs human: {s}")
        return
    hold = {l.strip() for l in resolve_path(cfg["perception"]["holdout_file"]).read_text("utf-8").splitlines() if l.strip() and not l.startswith("#")}
    todo = [c for c in clips if args.force or not (vlm_dir / c["category"] / f"{c['clip_id']}.json").exists()]
    log.info("%d violent clips, %d to do (held-out clips are proposed too: they are only used as a test, never trained on)", len(clips), len(todo))
    ask = QwenAsker(args.model or lc["model"], bool(lc["load_4bit"])) if todo else None
    for n, c in enumerate(todo, 1):
        try:
            r = propose_clip(resolve_path(c["path"]), c["duration_s"], ask, lc)
        except Exception as e:                                    # one bad clip must not stop a Colab run
            log.warning("%s failed: %s", c["clip_id"], e)
            continue
        r.update({"clip_id": c["clip_id"], "category": c["category"], "duration_s": c["duration_s"], "model": args.model or lc["model"]})
        p = vlm_dir / c["category"] / f"{c['clip_id']}.json"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(r, indent=1), encoding="utf-8")
        print(f"  vlm: {n}/{len(todo)} clips ({c['clip_id']}: {r['status']})", flush=True)
    props = load_proposals(vlm_dir, [c["clip_id"] for c in clips])
    labels = {cid: {"act_start_s": p["act_start_s"], "buildup_start_s": p["buildup_start_s"], "source": "vlm", "note": p["status"]}
              for cid, p in props.items() if p["status"] == "agree"}
    write_labels(resolve_path(lc["vlm_labels"]), labels)
    stat = {k: sum(1 for p in props.values() if p["status"] == k) for k in ("agree", "disagree", "invalid")}
    has_act = sum(1 for l in labels.values() if l["act_start_s"] is not None)
    has_build = sum(1 for l in labels.values() if l["buildup_start_s"] is not None)
    print(f"\nProposals: {stat}. Agreed labels: {len(labels)} (with an act {has_act}, with a buildup {has_build}) -> {resolve_path(lc['vlm_labels'])}")
    print("These are pseudo-labels: accuracy is unknown until the human validation set is scored (--score).")


if __name__ == "__main__":
    main()
