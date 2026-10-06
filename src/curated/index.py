"""Curated dataset, step 1: read the hand-made start times and match them to the video files.

The user's Excel (`FYP_data_filter.xlsx`, one sheet per category) or CSV has two columns, `video name` and `start time`: the second at which the
violence starts. Names there do not always equal the file names (truncated "Assassination_v4 (1", different case, "CHAIN SNATCHING" folders), so
matching works on a normalised key (category, clip number) taken from the file name itself, never from the sheet or folder name.
Nothing is dropped silently: `match()` returns the unmatched rows and the unlabeled videos for the notebook to print.
"""
import re
from pathlib import Path

VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv"}
MIN_PREVIOLENCE_S = 1.5
NORMAL = "Normal"
CATEGORY_NAMES = {"assassination": "Assassination", "chain_snatching": "Chain_Snatching", "harassment": "Harassment", "kidnapping": "Kidnapping",
                  "stalking": "Stalking", "normal": NORMAL}
_KEY = re.compile(r"^\s*([A-Za-z][A-Za-z _\-]*?)[ _\-]*v0*(\d+)", re.I)


def clip_key(name):
    """'Assassination_v4 (1).mp4' / 'CHAIN SNATCHING v13' -> ('chain_snatching', 13); None if it does not look like a clip name."""
    stem = Path(str(name)).stem if Path(str(name)).suffix.lower() in VIDEO_EXTS else str(name)
    m = _KEY.match(stem)
    if not m:
        return None
    cat = re.sub(r"[ \-]+", "_", m.group(1).strip().lower()).strip("_")
    cat = {"harrasment": "harassment", "harassement": "harassment", "harasment": "harassment", "chain_snatch": "chain_snatching"}.get(cat, cat)
    return cat, int(m.group(2))


def category_of(key):
    return CATEGORY_NAMES.get(key[0], key[0].title()) if key else None


def parse_time(x):
    """Seconds from 3, '3', '3.5', '0:03', '1:02:03', '3 s'; None when empty or not a time."""
    if x is None:
        return None
    if isinstance(x, (int, float)):
        return float(x) if x == x and x >= 0 else None
    if hasattr(x, "hour") and hasattr(x, "minute"):                      # a spreadsheet time cell
        return float(x.hour * 3600 + x.minute * 60 + x.second)
    s = str(x).strip().lower().replace("sec", "").replace("s", "").strip()
    if not s or s in ("nan", "none"):
        return None
    try:
        if ":" in s:
            parts = [float(p) for p in s.split(":")]
            sec = 0.0
            for p in parts:
                sec = sec * 60 + p
            return sec
        return float(s)
    except ValueError:
        return None


def read_annotations(path):
    """Excel (every sheet) or CSV -> [{name, start_s, sheet, raw_time}] for every row that has a name. Columns are found by their header words."""
    import pandas as pd
    p = Path(path)
    if p.suffix.lower() in (".xlsx", ".xlsm", ".xls"):
        sheets = pd.read_excel(p, sheet_name=None, header=0)
    else:
        sheets = {p.stem: pd.read_csv(p, header=0, encoding="utf-8-sig")}
    rows = []
    for sheet, df in sheets.items():
        cols = {str(c).strip().lower(): c for c in df.columns}
        name_col = next((c for k, c in cols.items() if "name" in k or "video" in k or "clip" in k or "file" in k), None)
        time_col = next((c for k, c in cols.items() if "start" in k or "time" in k or "second" in k), None)
        if name_col is None or time_col is None:
            if len(df.columns) >= 2:
                name_col, time_col = df.columns[0], df.columns[1]
            else:
                continue
        for _, r in df.iterrows():
            nm = r[name_col]
            if nm is None or (isinstance(nm, float) and nm != nm) or not str(nm).strip():
                continue
            rows.append({"name": str(nm).strip(), "start_s": parse_time(r[time_col]), "sheet": str(sheet), "raw_time": r[time_col]})
    return rows


def find_videos(root):
    """All video files under root (recursive) -> [{path, stem, key, folder}]."""
    out = []
    for f in sorted(Path(root).rglob("*")):
        if f.is_file() and f.suffix.lower() in VIDEO_EXTS:
            out.append({"path": str(f), "stem": f.stem, "key": clip_key(f.name), "folder": f.parent.name})
    return out


def match(annotations, videos):
    """-> (matched, unmatched_rows, unlabeled_videos, notes).
    matched: [{clip_id, category, path, start_s, name_in_sheet}]. One video per key: an exact normalised-stem match wins over a looser one."""
    by_key = {}
    for v in videos:
        if v["key"]:
            by_key.setdefault(v["key"], []).append(v)
    matched, unmatched, notes, used = [], [], [], set()
    for a in annotations:
        key = clip_key(a["name"])
        cands = by_key.get(key, []) if key else []
        if not key or not cands:
            unmatched.append({**a, "reason": "no video with this category and number" if key else "name does not look like Category_vNN"})
            continue
        if a["start_s"] is None:
            unmatched.append({**a, "reason": f"start time not understood: {a['raw_time']!r}"})
            continue
        norm = re.sub(r"[^a-z0-9]", "", Path(a["name"]).stem.lower())
        best = sorted(cands, key=lambda v: (re.sub(r"[^a-z0-9]", "", v["stem"].lower()) != norm, len(v["stem"])))[0]
        if len(cands) > 1:
            notes.append(f"{a['name']}: {len(cands)} files share this clip number, using {Path(best['path']).name}")
        if key in used:
            notes.append(f"{a['name']}: clip {key} is listed more than once in the sheet; the first start time is kept")
            continue
        used.add(key)
        matched.append({"clip_id": f"{category_of(key)}_v{key[1]}", "category": category_of(key), "path": best["path"], "start_s": float(a["start_s"]),
                        "name_in_sheet": a["name"]})
    unlabeled = [v for v in videos if v["key"] and v["key"] not in used and category_of(v["key"]) != NORMAL]
    return matched, unmatched, unlabeled, notes


def normal_videos(videos, limit=None):
    """Normal clips for the control group, evenly spread when limited."""
    norm = [v for v in videos if v["key"] and category_of(v["key"]) == NORMAL]
    if limit and len(norm) > limit:
        step = len(norm) / limit
        norm = [norm[int(i * step)] for i in range(limit)]
    return [{"clip_id": f"{NORMAL}_v{v['key'][1]}", "category": NORMAL, "path": v["path"], "start_s": None, "name_in_sheet": ""} for v in norm]


def check_times(matched, durations, min_s=MIN_PREVIOLENCE_S):
    """Adds `status` to every matched clip: ok | short (less than min_s of pre-violence) | beyond_end (T after the clip ends: T is clipped to the end)."""
    out = []
    for m in matched:
        d = durations.get(m["clip_id"])
        st = "ok"
        T = m["start_s"]
        if d is not None and T > d + 0.05:
            st, T = "beyond_end", d
        elif T < min_s:
            st = "short"
        out.append({**m, "start_s": T, "status": st, "duration_s": d})
    return out
