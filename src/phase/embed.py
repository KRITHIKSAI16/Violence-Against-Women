"""Layer 2, step 7: frozen V-JEPA 2 video embeddings of every window (appearance and motion that pose cannot see: struggle, grabbing, weapons).

For each clip the video is decoded once at `embed_fps` (8 fps, short side 256 px) and each 2 s window of the phase table becomes a 16-frame clip
for V-JEPA 2 (`facebook/vjepa2-vitl-fpc64-256`, frozen, mean-pooled tokens -> 1024 numbers). Embeddings are cached per clip. Before training they are
reduced with a PCA fitted on TRAINING windows only (`attach_pca`), giving columns `emb_0 .. emb_{k-1}`.

Risk named up front: an embedding can encode how a clip was FILMED rather than what happens. The phase model is therefore judged only inside clips
(act onset, buildup interval), where the filming style is constant, and the ablation `without emb_` is always reported next to the full model.

    python -m src.phase.embed [--config CFG] [--clips A B] [--force]      (GPU recommended; resumable per clip)
"""
import argparse
import logging
from pathlib import Path

import cv2
import numpy as np

from src.config import load_config, resolve_path

log = logging.getLogger(__name__)
N_FRAMES = 16


def load_frames(video_path, embed_fps=8.0, short_side=256):
    """Frames at embed_fps from the start: uint8 (N,H,W,3) RGB (even sizes) and the frame times."""
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    step = fps / embed_fps
    frames, k, nxt = [], 0, 0.0
    while True:
        ok, img = cap.read()
        if not ok:
            break
        if k + 1e-9 >= nxt:
            h, w = img.shape[:2]
            s = short_side / min(h, w)
            if s < 1:
                img = cv2.resize(img, (int(w * s) // 2 * 2, int(h * s) // 2 * 2), interpolation=cv2.INTER_AREA)
            frames.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            nxt += step
        k += 1
    cap.release()
    return np.stack(frames) if frames else np.zeros((0, 2, 2, 3), np.uint8)


def window_clips(frames, t_starts, win_s, embed_fps=8.0, n=N_FRAMES):
    """[(n,H,W,3)] one per window: n frames evenly spread over [t_start, t_start + win_s). Windows beyond the video are clamped to its last frame."""
    out = []
    for t0 in t_starts:
        idx = np.clip(np.round((t0 + np.linspace(0, win_s, n, endpoint=False)) * embed_fps).astype(int), 0, len(frames) - 1)
        out.append(frames[idx])
    return out


class VJepa:
    """Frozen V-JEPA 2 encoder via transformers. Needs the weights download (about 1.3 GB for ViT-L); not exercised by the unit tests."""

    def __init__(self, model_id="facebook/vjepa2-vitl-fpc64-256", device="auto"):
        import torch
        from transformers import AutoModel, AutoVideoProcessor
        self.torch = torch
        self.device = ("cuda" if torch.cuda.is_available() else "cpu") if device == "auto" else device
        self.processor = AutoVideoProcessor.from_pretrained(model_id)
        self.model = AutoModel.from_pretrained(model_id, torch_dtype=torch.float16 if self.device == "cuda" else torch.float32).to(self.device).eval()

    def __call__(self, clips, batch=4):
        out = []
        with self.torch.no_grad():
            for b in range(0, len(clips), batch):
                videos = [self.torch.from_numpy(c).permute(0, 3, 1, 2) for c in clips[b:b + batch]]       # T,C,H,W uint8
                inputs = self.processor(videos, return_tensors="pt").to(self.device)
                hid = self.model(**inputs).last_hidden_state
                out.append(hid.float().mean(dim=1).cpu().numpy())
        return np.concatenate(out) if out else np.zeros((0, 1024), np.float32)


def embed_clip(video_path, t_starts, win_s, encoder, embed_fps=8.0):
    """-> (len(t_starts), D) embeddings of the given windows."""
    frames = load_frames(video_path, embed_fps)
    if len(frames) == 0:
        raise ValueError(f"no frames in {video_path}")
    return encoder(window_clips(frames, t_starts, win_s, embed_fps))


def emb_path(emb_dir, category, clip_id):
    return Path(emb_dir) / category / f"{clip_id}_emb.npz"


def attach_pca(df, emb_dir, categories, fit_ids, k=64):
    """Adds emb_0..emb_{k-1} to df (rows of clips without a cached embedding stay NaN). PCA is fitted on the windows of `fit_ids` only."""
    from sklearn.decomposition import PCA
    raw = np.full((len(df), 0), np.nan)
    parts, have = {}, np.zeros(len(df), bool)
    for cid, idx in df.groupby("clip_id", sort=False).indices.items():
        p = emb_path(emb_dir, categories[cid], cid)
        if not p.exists():
            continue
        z = np.load(p)
        look = {round(float(t), 3): r for t, r in zip(z["t_start"], z["emb"])}
        for i in idx:
            r = look.get(round(float(df["t_start"].iloc[i]), 3))
            if r is not None:
                parts[i] = r
                have[i] = True
    if not have.any():
        return df
    D = len(next(iter(parts.values())))
    raw = np.full((len(df), D), np.nan)
    for i, r in parts.items():
        raw[i] = r
    fit_mask = have & df["clip_id"].isin(fit_ids).to_numpy()
    kk = int(min(k, fit_mask.sum(), D))
    pca = PCA(n_components=kk, random_state=7).fit(raw[fit_mask])
    red = np.full((len(df), kk), np.nan)
    red[have] = pca.transform(raw[have])
    out = df.copy()
    for c in range(kk):
        out[f"emb_{c}"] = red[:, c]
    return out


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    import json

    import pandas as pd
    ap = argparse.ArgumentParser(description="Layer 2: V-JEPA 2 window embeddings")
    ap.add_argument("--config", default=None)
    ap.add_argument("--clips", nargs="*", default=None)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    cfg = load_config(args.config)
    pc = cfg["phase"]
    clean = {c["clip_id"]: c for c in json.loads(resolve_path(cfg["preprocess"]["clean_manifest_path"]).read_text("utf-8"))["clips"]}
    df = pd.read_pickle(resolve_path(pc["dir"]) / "windows.pkl")
    emb_dir = resolve_path(pc["dir"]) / "emb"
    ids = [c for c in df["clip_id"].unique() if not args.clips or c in args.clips]
    todo = [c for c in ids if args.force or not emb_path(emb_dir, clean[c]["category"], c).exists()]
    log.info("%d clips with windows, %d to embed", len(ids), len(todo))
    enc = VJepa(pc["embed_model"]) if todo else None
    for n, cid in enumerate(todo, 1):
        c = clean[cid]
        ts = np.sort(df.loc[df["clip_id"] == cid, "t_start"].unique())
        try:
            e = embed_clip(resolve_path(c["path"]), ts, float(pc["win_s"]), enc, float(pc["embed_fps"]))
        except Exception as ex:                       # one bad clip must not stop a Colab run
            log.warning("%s failed: %s", cid, ex)
            continue
        p = emb_path(emb_dir, c["category"], cid)
        p.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(p, t_start=ts, emb=e.astype(np.float16))
        print(f"  embed: {n}/{len(todo)} clips ({cid}, {len(ts)} windows)", flush=True)


if __name__ == "__main__":
    main()
