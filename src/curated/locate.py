"""Curated dataset: find the shared `violence` folder and the annotation file on the (Colab) Drive, and say exactly what to do when they are missing.

A folder that is only "Shared with me" does not appear under My Drive; one click makes it appear: right-click the folder -> Organize -> Add shortcut -> My Drive.
"""
from pathlib import Path

ANNOTATION_PATTERNS = ["FYP_data_filter*.xlsx", "FYP_data_filter*.xls", "FYP_Annotations*.xlsx", "FYP_Annotations*.csv", "*.xlsx"]
HELP_FOLDER = ("The folder 'violence' was not found in your Drive. It is shared with you, so add a shortcut: open Google Drive -> Shared with me -> right-click 'violence' -> "
               "Organize -> Add shortcut -> My Drive, then run this cell again.")
HELP_EXCEL = ("The start-time file (FYP_data_filter.xlsx) was not found. Upload it to My Drive (or into VAW_results or the violence folder), "
              "or pass its path with --annotations.")


def find_violence_root(drive_root="/content/drive/MyDrive", name="violence", max_depth=2):
    """Folder called `name` (case-insensitive) with category sub-folders, searched down to max_depth levels under drive_root. None if absent."""
    root = Path(drive_root)
    if not root.is_dir():
        return None
    level = [root]
    for _ in range(max_depth + 1):
        nxt = []
        for d in level:
            try:
                kids = sorted(p for p in d.iterdir() if p.is_dir())
            except OSError:
                continue
            for k in kids:
                if k.name.lower() == name.lower() and any(c.is_dir() for c in k.iterdir()):
                    return k
            nxt += kids
        level = nxt
    return None


def find_annotations(search_dirs, patterns=ANNOTATION_PATTERNS):
    """First file matching the patterns (in pattern order, then directory order) in any of the directories (top level only). None if absent."""
    for pat in patterns:
        for d in search_dirs:
            d = Path(d)
            if d.is_dir():
                hits = sorted(d.glob(pat))
                if hits:
                    return hits[0]
    return None


def locate(drive_root="/content/drive/MyDrive", results_dir=None, data_root=None, annotations=None):
    """-> (data_root Path, annotations Path, messages list). Raises FileNotFoundError with the help text when something is missing."""
    msgs = []
    dr = Path(data_root) if data_root else find_violence_root(drive_root)
    if dr is None or not Path(dr).is_dir():
        raise FileNotFoundError(HELP_FOLDER)
    msgs.append(f"video folder: {dr}")
    ann = Path(annotations) if annotations else find_annotations([dr, drive_root, results_dir or Path(drive_root) / "VAW_results"])
    if ann is None or not Path(ann).exists():
        raise FileNotFoundError(HELP_EXCEL)
    msgs.append(f"start times: {ann}")
    return Path(dr), Path(ann), msgs
