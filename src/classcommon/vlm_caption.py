"""Classification pipelines, Layer B: a video-language model says in a few plain sentences what the people in a handful of frames are doing.

Layer B is a switch (`classify.use_captions`, CLI `--no-captions`): the geometry text of Layer A (src.classcommon.text_build) works without it.
This is a language model inside the pipeline, which the repository rules otherwise exclude; it is therefore optional, reported as its own ablation, and constrained:
  * the prompt asks for movements, positions and objects only, never appearance, gender, age or identity, and never mentions violence;
  * its answer is cleaned (`clean_caption`): identity words are replaced by neutral ones, sentences that name a dataset category are dropped, and for the
    trim task sentences that name the violent act itself are dropped too (the clip is meant to END before the act, so such a word means the cut is late);
  * frames are read only from the analysed window (trim: the last `span_s` seconds of a clip that already ends at the violence start).
The model needs a GPU; the unit tests use a fake `ask`.
"""
import json
import re
from pathlib import Path

PROMPT = ("These are frames from one video clip, in time order. In two or three short factual sentences, say what the people are doing: how they move, where they are "
          "relative to each other, and what they hold or touch. Do not describe clothing, appearance, gender, age or identity. Do not guess intentions or feelings. "
          "If nothing notable happens, say so.")
CATEGORY_RE = re.compile(r"\b(assassin\w*|kidnap\w*|snatch\w*|harass\w*|stalk\w*|abduct\w*)\b", re.I)
ACT_RE = re.compile(r"\b(attack\w*|assault\w*|fight\w*|punch\w*|hit|hits|hitting|kick\w*|stab\w*|shoot\w*|shot|beat\w*|strangl\w*|chok\w*|slap\w*|violen\w*|kill\w*|murder\w*|grab\w*|struggl\w*|wrestl\w*)\b", re.I)
# identity words -> neutral ones (best effort: the prompt is the first line of defence)
_AGE_PREFIX = r"(?:young|old|elderly|middle-aged|teenage|teen)\s+"
_PEOPLE_SING = r"(?:man|woman|boy|girl|guy|lady|gentleman|male|female|child|kid|toddler|teenager|adult)"
_PEOPLE_PLUR = r"(?:men|women|boys|girls|guys|ladies|gentlemen|males|females|children|kids|teenagers|adults)"
IDENTITY = [(re.compile(rf"\b(?:{_AGE_PREFIX})?{_PEOPLE_SING}\b", re.I), "person"), (re.compile(rf"\b(?:{_AGE_PREFIX})?{_PEOPLE_PLUR}\b", re.I), "people"),
            (re.compile(r"\b(?:he|she)\b", re.I), "they"), (re.compile(r"\bhim\b", re.I), "them"), (re.compile(r"\b(?:his|her)\b", re.I), "their"),
            (re.compile(r"\bhers\b", re.I), "theirs"), (re.compile(r"\b(?:himself|herself)\b", re.I), "themselves"), (re.compile(r"\belderly\b", re.I), "")]
_SENT = re.compile(r"(?<=[.!?])\s+")


def clean_caption(text, task):
    """-> (cleaned text, number of sentences removed). Identity words are neutralised; category names are removed everywhere, act words only for task 'trim'."""
    kept, removed = [], 0
    for s in _SENT.split(str(text or "").strip()):
        if not s.strip():
            continue
        if CATEGORY_RE.search(s) or (task == "trim" and ACT_RE.search(s)):
            removed += 1
            continue
        for rx, rep in IDENTITY:
            s = rx.sub(lambda m, rep=rep: rep[:1].upper() + rep[1:] if m.group(0)[:1].isupper() else rep, s)     # keep a capital letter at the start of a sentence
        kept.append(re.sub(r"\s{2,}", " ", s).strip())
    return " ".join(kept), removed


def frame_times(lo, hi, n):
    """n times at the middle of n equal slices of [lo, hi] (never the very last instant, where a decoder often returns nothing)."""
    n = max(1, int(n))
    step = (hi - lo) / n
    return [round(lo + (k + 0.5) * step, 3) for k in range(n)]


def caption_path(caption_dir, category, clip_id):
    return Path(caption_dir) / category / f"{clip_id}.json"


def caption_clip(video_path, duration_s, ask, task, span_s, n_frames, short_side=336):
    """One clip -> {times, raw, text, removed}. ask(frames, prompt) -> str (frames = [(t, RGB uint8 array)])."""
    from src.classcommon.text_build import window_bounds
    from src.label.vlm_propose import read_frames
    lo, hi = window_bounds(duration_s, task, span_s)
    times = frame_times(lo, hi, n_frames)
    frames = read_frames(video_path, times, short_side)
    if not frames:
        raise ValueError(f"no frame could be read from {video_path}")
    raw = ask(frames, PROMPT)
    text, removed = clean_caption(raw, task)
    return {"times": [t for t, _ in frames], "raw": raw, "text": text, "removed": removed}


def run_captions(recs, cfg, task, ask=None, force=False):
    """Caption every clip that has no cached caption (resumable). `recs`: records with category, clip_id, path, duration_s. Returns {clip_id: caption dict or None}."""
    cl = cfg["classify"]
    cc = cl["caption"]
    todo = [r for r in recs if force or not caption_path(cl["caption_dir"], r["category"], r["clip_id"]).exists()]
    print(f"captions: {len(recs) - len(todo)} cached, {len(todo)} to do", flush=True)
    if todo and ask is None:
        from src.label.vlm_propose import QwenAsker
        ask = QwenAsker(cc["model"], bool(cc["load_4bit"]), int(cc["max_new_tokens"]))
    n_frames = int(cc["n_frames_trim"] if task == "trim" else cc["n_frames_full"])
    for k, r in enumerate(todo, 1):
        try:
            out = caption_clip(r["path"], r["duration_s"], ask, task, float(cl["span_s"]), n_frames, int(cc["short_side"]))
        except Exception as e:                                    # one bad clip must not stop a Colab run
            print(f"  caption failed for {r['clip_id']}: {e}", flush=True)
            continue
        out.update({"clip_id": r["clip_id"], "category": r["category"], "model": cc["model"]})
        p = caption_path(cl["caption_dir"], r["category"], r["clip_id"])
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(out, indent=1), encoding="utf-8")
        if k % 5 == 0 or k == len(todo):
            print(f"  captioned {k}/{len(todo)}", flush=True)
    return load_captions(recs, cl["caption_dir"])


def load_captions(recs, caption_dir):
    out = {}
    for r in recs:
        p = caption_path(caption_dir, r["category"], r["clip_id"])
        out[r["clip_id"]] = json.loads(p.read_text(encoding="utf-8")) if p.exists() else None
    return out
