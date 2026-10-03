"""Deep context, Stage C runner: build the buildup story (scene graph episodes + narrative) for every clip.

Needs Stage A (src.context.features). Uses Stage B (src.context.scene) when its layout / depth files exist; without them the doors / pinned
predicates are simply absent. CPU only, resumable.

Outputs per clip:  context_dir/<Category>/<clip>_story.json
Usage:  python -m src.context.story [--config CFG] [--clips A B] [--force] [--example CLIP]
"""
import argparse
import json
import logging
from collections import defaultdict

import numpy as np

from src.assm.track_poses import load_tracks
from src.config import load_config, parse_overrides, resolve_path
from src.context.features import load_context
from src.context.graph import ACT_PREDS, build_story, load_story, save_story
from src.context.narrative import narrative_markdown
from src.context.scene import load_depth, load_layout

log = logging.getLogger(__name__)
SHOW = ["approaches_from_behind", "follows", "looks_back", "hovers_near", "mutual_facing", "blocks_exit", "pinned_against", "reaches_for", "contact",
        "flees_from", "target_stops", "target_speeds_up"]


def story_for_clip(clip, cfg, out_dir):
    scene, arrays = load_context(out_dir, clip["category"], clip["clip_id"])
    t = load_tracks(resolve_path(cfg["assm"]["tracks_dir"]) / clip["category"] / f"{clip['clip_id']}.npz")
    _, layout = load_layout(out_dir, clip["category"], clip["clip_id"])
    depth = load_depth(out_dir, clip["category"], clip["clip_id"])
    return build_story(t, scene, arrays, layout, depth, cfg["context"], cfg["story"], clip["clip_id"], clip["category"])


def print_table(stories):
    by = defaultdict(list)
    for s in stories:
        by[s["category"]].append(s)
    print(f"{'category':<16}{'clips':>6}{'no-pair':>9}{'any cue':>9}{'ordered':>9}{'act cue':>9}{'med concern s':>15}{'med benign s':>14}")
    for cat, lst in sorted(by.items()):
        n = len(lst)
        pr = [s for s in lst if not s["no_pair"]]
        print(f"{cat:<16}{n:>6}{sum(s['no_pair'] for s in lst) / n:>9.0%}{sum(bool(s['preds_seen']) for s in lst) / n:>9.0%}"
              f"{sum(s['pairs'][s['key_pair']]['ordered_progression'] for s in pr) / n:>9.0%}{sum(s['escalation'] is not None for s in lst) / n:>9.0%}"
              f"{np.median([s['concern_seconds'] for s in pr] or [0]):>15.1f}{np.median([s['benign_seconds'] for s in pr] or [0]):>14.1f}")
    print(f"\n{'share of clips with predicate':<30}" + "".join(f"{c:>10}" for c in sorted(by)))
    for p in SHOW:
        print(f"{p:<30}" + "".join(f"{sum(p in s['preds_seen'] for s in by[c]) / len(by[c]):>10.0%}" for c in sorted(by)))


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    ap = argparse.ArgumentParser(description="Stage C: scene graph episodes and narrative for every clip")
    ap.add_argument("--config", default=None)
    ap.add_argument("--clips", nargs="*", default=None)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--example", default=None, help="print the narrative of this clip")
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE", help="override story settings for this run")
    args = ap.parse_args()
    cfg = load_config(args.config)
    cfg["story"] = {**cfg["story"], **parse_overrides(args.set)}
    clean = json.loads(resolve_path(cfg["preprocess"]["clean_manifest_path"]).read_text("utf-8"))
    out_dir = resolve_path(cfg["context"]["context_dir"])
    stories, failed = [], []
    for k, c in enumerate(clean["clips"], 1):
        if args.clips and c["clip_id"] not in args.clips:
            continue
        if not (out_dir / c["category"] / f"{c['clip_id']}_scene.json").exists():
            continue
        try:
            st = None if args.force else load_story(out_dir, c["category"], c["clip_id"])
            if st is None:
                st = story_for_clip(c, cfg, out_dir)
                save_story(out_dir, c["category"], c["clip_id"], st)
            stories.append(st)
        except Exception as e:  # one bad clip must not stop a long run
            failed.append((c["clip_id"], str(e)))
            log.warning("Story failed for %s: %s", c["clip_id"], e)
        if k % 100 == 0:
            print(f"  story: {k}/{len(clean['clips'])} clips", flush=True)
    print(f"\n{len(stories)} stories -> {out_dir}  ({len(failed)} failed)\n")
    if stories:
        print_table(stories)
    if args.example:
        s = next((x for x in stories if x["clip_id"] == args.example), None)
        print("\n" + (narrative_markdown(s) if s else f"no story for {args.example}"))
    for cid, why in failed[:10]:
        print(f"  FAILED {cid}: {why}")


if __name__ == "__main__":
    main()
