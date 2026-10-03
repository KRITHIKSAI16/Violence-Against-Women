"""Config loader shared by all modules (M1-M9)."""
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = REPO_ROOT / "configs" / "extran.yaml"


def load_config(path=None):
    """Load the YAML config. Relative paths are resolved against the repo root."""
    path = Path(path) if path else DEFAULT_CONFIG
    if not path.is_absolute():
        path = REPO_ROOT / path
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_path(p):
    """Resolve a config path against the repo root (absolute paths pass through)."""
    p = Path(p)
    return p if p.is_absolute() else REPO_ROOT / p


def parse_overrides(pairs):
    """Turn ["follow_min_s=3", "hover_max_d=2"] into {"follow_min_s": 3.0, "hover_max_d": 2.0}."""
    out = {}
    for item in pairs or []:
        key, _, val = item.partition("=")
        if not key or not _:
            raise ValueError(f"override must look like key=value, got {item!r}")
        try:
            out[key.strip()] = float(val)
        except ValueError:
            out[key.strip()] = val.strip()
    return out
