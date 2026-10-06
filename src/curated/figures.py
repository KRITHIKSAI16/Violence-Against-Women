"""Curated dataset: figures for the jury. Matplotlib only, light colours on a white page, no interactivity.

distance_figure   the pair's distance to each other over time with proxemic zones shaded, relation spans under it, the violence start marked
concern_figure    violent clips' last-window concern vs Normal clips' windows (distribution plot for the Normal comparison)
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from src.curated.relation import BENIGN, CONCERN  # noqa: E402

ZONES = [(0.0, 0.46, "intimate", "#f8d7d4"), (0.46, 1.2, "personal", "#fbe7cf"), (1.2, 3.7, "social", "#fdf5d4"), (3.7, 7.6, "public", "#eef3e2")]
SPAN_COLOR = {"concern": "#c0392b", "benign": "#2e8b57", "neutral": "#8a8f98"}


def span_kind(label):
    return "concern" if label in CONCERN else "benign" if label in BENIGN else "neutral"


def distance_figure(res, dist_series, fps, out_path):
    """res: clip result (needs interaction). dist_series: per-frame distance (m) of the key pair. Returns the path or None."""
    if not res.get("interaction"):
        return None
    inter = res["interaction"]
    d = np.asarray(dist_series, float)
    t = np.arange(len(d)) / fps
    fig, (ax, bx) = plt.subplots(2, 1, figsize=(8.0, 3.8), gridspec_kw={"height_ratios": [3, 1.2]}, sharex=True)
    top = max(8.0, float(np.nanmax(d[np.isfinite(d)])) * 1.1 if np.isfinite(d).any() else 8.0)
    for lo, hi, name, col in ZONES:
        ax.axhspan(lo, min(hi, top), color=col, lw=0)
        ax.text(t[-1] * 1.005 if len(t) else 0, (lo + min(hi, top)) / 2, name, va="center", fontsize=7, color="#666")
    ax.plot(t, d, color="#1f3b5c", lw=2)
    ax.set_ylim(0, top)
    ax.set_ylabel("distance between the two (m)")
    ax.grid(alpha=0.2)
    if res.get("start_s") is not None:
        ax.axvline(res["duration_s"], color="#c0392b", lw=2, ls="--")
        ax.text(res["duration_s"], top * 0.97, " violence starts", color="#c0392b", fontsize=8, ha="right", va="top")
    ax.set_title(f"{res['clip_id']}: distance before the violence" if res.get("start_s") is not None else f"{res['clip_id']}: distance (Normal clip)", fontsize=10, loc="left")
    bx.set_ylim(0, 1)
    bx.set_yticks([])
    for sp in inter["spans"]:
        kind = span_kind(sp["label"])
        bx.axvspan(sp["start_s"], sp["end_s"], color=SPAN_COLOR[kind], alpha=0.85 if kind != "neutral" else 0.45)
        bx.text((sp["start_s"] + sp["end_s"]) / 2, 0.5, sp["label"].replace("_", " "), ha="center", va="center", fontsize=6.5, color="white" if span_kind(sp["label"]) != "neutral" else "#333", clip_on=True)
    bx.set_xlabel("time (s)")
    fig.tight_layout()
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out


def concern_figure(violent, normal, out_path, title="Concern score: last seconds before the violence vs Normal clips"):
    """violent: list of per-clip scores (last window). normal: list of per-clip scores (their highest window). Strip plot with medians."""
    fig, ax = plt.subplots(figsize=(6.2, 3.4))
    rng = np.random.default_rng(0)
    for k, (name, vals, col) in enumerate([("pre-violence (last window)", violent, "#c0392b"), ("Normal (highest window)", normal, "#2e8b57")]):
        v = np.asarray([x for x in vals if x is not None and np.isfinite(x)], float)
        if len(v):
            ax.scatter(k + rng.uniform(-0.12, 0.12, len(v)), v, color=col, alpha=0.75, s=28)
            ax.hlines(np.median(v), k - 0.25, k + 0.25, color="black", lw=2)
        ax.text(k, ax.get_ylim()[0], f"n={len(v)}", ha="center", va="bottom", fontsize=8, color="#555")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["pre-violence\n(last window)", "Normal\n(highest window)"])
    ax.set_ylabel("concern score (heuristic units)")
    ax.set_title(title, fontsize=9, loc="left")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out
