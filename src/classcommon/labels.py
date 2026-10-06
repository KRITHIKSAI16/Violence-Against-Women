"""Classification pipelines, step 2: which videos exist and what is their label.

Label = the category folder: Normal -> 0 (non-violent), every other category -> 1 (violent). The full task needs no start time, so every video whose
name looks like `Category_vNN` is used; names that do not are reported, never guessed.
"""
from src.curated.index import NORMAL, category_of, find_videos


def label_of(category):
    """0 for Normal, 1 for every violent category."""
    return 0 if category == NORMAL else 1


def spread(items, n):
    """n items spread evenly over the list (all of them when n is None or not smaller than the list)."""
    if n is None or n >= len(items):
        return list(items)
    if n <= 0:
        return []
    step = len(items) / n
    return [items[int(i * step)] for i in range(n)]


def collect_videos(data_root, n_normal=None, limit=None):
    """All usable videos under data_root -> (clips, skipped_names, notes).

    clips: [{clip_id, category, path, start_s (None: whole video), status, name_in_sheet}] in the shape src.curated.prepare.build_curated_manifest reads.
    n_normal caps the Normal clips (evenly spread). limit (for a quick trial) keeps about limit/2 violent and limit/2 Normal clips, evenly spread."""
    seen, skipped, notes = {}, [], []
    for v in find_videos(data_root):
        if v["key"] is None:
            skipped.append(v["stem"])
        elif v["key"] in seen:
            notes.append(f"{v['stem']}: clip {v['key']} already taken from {seen[v['key']]['stem']}, this file is ignored")
        else:
            seen[v["key"]] = v
    clips = [{"clip_id": f"{category_of(k)}_v{k[1]}", "category": category_of(k), "path": v["path"], "start_s": None, "status": "ok", "name_in_sheet": ""}
             for k, v in sorted(seen.items())]
    normal = spread([c for c in clips if c["category"] == NORMAL], n_normal)
    violent = [c for c in clips if c["category"] != NORMAL]
    if limit:
        n_v = limit // 2
        violent, normal = spread(violent, n_v), spread(normal, limit - n_v)
    return violent + normal, skipped, notes
