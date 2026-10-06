"""Curated dataset, ML layer: pretrained video models as frozen encoders. They give a second opinion on what the VIDEO looks like, next to the geometry.

Encoders (fallback chain, whichever loads is named in the report):
  internvideo2   OpenGVLab InternVideo2 CLIP (distilled, 0.4B). Loaded with trust_remote_code; its Hub page documents no usage, so the loader probes the
                 remote code for video / text feature functions and raises if it cannot find them. The 6B and the 8B chat variants are not used (too heavy / a language model).
  xclip          microsoft/xclip-base-patch16 (transformers, 8 frames, 224 px): video-text similarity
  vjepa2         facebook/vjepa2-vitl-fpc64-256 (embeddings only, no text): strong on motion; reuses src.phase.embed

Zero-shot prompt scoring: each window is compared with short text prompts (contrastive video-text similarity: nothing is generated, no language model
runs). `margin()` = probability mass on concern prompts minus benign prompts. Known limit: CLIP-style models are weak at fine interaction; the report
validates the margin against the human start times before trusting it.
"""
import logging

import numpy as np

log = logging.getLogger(__name__)

PROMPTS = {
    "concern": ["a person following another person", "a person approaching someone from behind", "a person grabbing another person", "a person threatening another person",
                "a person chasing someone", "people fighting"],
    "benign": ["two people walking together", "people talking to each other", "a person walking alone", "people standing and waiting", "a person walking past another person"],
}
ALL_PROMPTS = PROMPTS["concern"] + PROMPTS["benign"]
N_FRAMES = 8


def margin(logits, temperature=1.0):
    """logits (n, len(ALL_PROMPTS)) -> (n,) probability on concern prompts minus probability on benign prompts (softmax over all prompts)."""
    z = np.asarray(logits, float) / temperature
    z = z - z.max(axis=1, keepdims=True)
    p = np.exp(z)
    p /= p.sum(axis=1, keepdims=True)
    k = len(PROMPTS["concern"])
    return p[:, :k].sum(axis=1) - p[:, k:].sum(axis=1)


class XClip:
    name = "xclip"
    has_text = True

    def __init__(self, model_id="microsoft/xclip-base-patch16", device="auto"):
        import torch
        from transformers import AutoProcessor, XCLIPModel
        self.torch = torch
        self.device = ("cuda" if torch.cuda.is_available() else "cpu") if device == "auto" else device
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = XCLIPModel.from_pretrained(model_id).to(self.device).eval()
        self.n_frames = int(self.model.config.vision_config.num_frames)         # 8 for this checkpoint, 32 for the zero-shot one

    def __call__(self, windows, prompts=ALL_PROMPTS, batch=4):
        """windows: list of (8,H,W,3) uint8 RGB arrays -> {"emb": (n,D), "logits": (n,P)}."""
        embs, logs = [], []
        with self.torch.no_grad():
            for b in range(0, len(windows), batch):
                chunk = [list(w) for w in windows[b:b + batch]]
                inp = self.processor(text=prompts, videos=chunk, return_tensors="pt", padding=True).to(self.device)
                out = self.model(**inp)
                embs.append(out.video_embeds.float().cpu().numpy())
                logs.append(out.logits_per_video.float().cpu().numpy())
        return {"emb": np.concatenate(embs), "logits": np.concatenate(logs)}


class InternVideo2Clip:
    name = "internvideo2"
    has_text = True

    def __init__(self, model_id="OpenGVLab/InternVideo2_CLIP_S", device="auto"):
        import torch
        from transformers import AutoModel
        self.torch = torch
        self.device = ("cuda" if torch.cuda.is_available() else "cpu") if device == "auto" else device
        self.model = AutoModel.from_pretrained(model_id, trust_remote_code=True, torch_dtype=torch.float16 if self.device == "cuda" else torch.float32).to(self.device).eval()
        self.vid = next((getattr(self.model, n) for n in ("get_vid_feat", "encode_vision", "get_video_features") if hasattr(self.model, n)), None)
        self.txt = next((getattr(self.model, n) for n in ("get_txt_feat", "encode_text", "get_text_features") if hasattr(self.model, n)), None)
        if self.vid is None or self.txt is None:
            raise RuntimeError("InternVideo2 remote code exposes no video / text feature function that this loader knows")

    def __call__(self, windows, prompts=ALL_PROMPTS, batch=2):
        torch = self.torch
        mean = torch.tensor([0.485, 0.456, 0.406], device=self.device).view(1, 1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=self.device).view(1, 1, 3, 1, 1)
        import cv2
        embs = []
        with torch.no_grad():
            t = self.txt(prompts)
            t = torch.nn.functional.normalize(t.float(), dim=-1)
            for b in range(0, len(windows), batch):
                x = np.stack([np.stack([cv2.resize(f, (224, 224)) for f in w]) for w in windows[b:b + batch]])        # B,T,224,224,3
                x = torch.from_numpy(x).to(self.device).permute(0, 1, 4, 2, 3).float() / 255.0
                x = ((x - mean) / std).to(next(self.model.parameters()).dtype)
                v = torch.nn.functional.normalize(self.vid(x).float(), dim=-1)
                embs.append(v.cpu().numpy())
        emb = np.concatenate(embs)
        return {"emb": emb, "logits": 100.0 * emb @ t.cpu().numpy().T}


class VJepa2:
    name = "vjepa2"
    has_text = False

    def __init__(self, model_id="facebook/vjepa2-vitl-fpc64-256", device="auto"):
        from src.phase.embed import VJepa
        self.enc = VJepa(model_id, device)

    def __call__(self, windows, prompts=None, batch=4):
        emb = self.enc(windows, batch)
        return {"emb": emb, "logits": None}


LOADERS = {"internvideo2": InternVideo2Clip, "xclip": XClip, "vjepa2": VJepa2}


def load_encoders(names=("internvideo2", "xclip", "vjepa2"), device="auto", loaders=None):
    """Try each encoder in order. Returns ({name: encoder}, {name: error text for those that failed}); the notebook prints both."""
    loaders = loaders or LOADERS
    ok, failed = {}, {}
    for n in names:
        try:
            ok[n] = loaders[n](device=device)
            log.info("loaded encoder %s", n)
        except Exception as e:                                       # a model that cannot load (remote code, memory, no network) must not stop the run
            failed[n] = f"{type(e).__name__}: {str(e)[:200]}"
            log.warning("encoder %s not available: %s", n, failed[n])
    return ok, failed


def window_frames(video_path, t_starts, win_s, n_frames=N_FRAMES, embed_fps=8.0, short_side=256):
    """[(n_frames,H,W,3)] one per window start, frames spread evenly over the window. Reuses the V-JEPA frame loader."""
    from src.phase.embed import load_frames, window_clips
    fr = load_frames(video_path, embed_fps, short_side)
    if len(fr) == 0:
        raise ValueError(f"no frames in {video_path}")
    return window_clips(fr, t_starts, win_s, embed_fps, n_frames)


def encode_clip(encoder, video_path, t_starts, win_s, n_frames=N_FRAMES):
    n = getattr(encoder, "n_frames", 16 if encoder.name == "vjepa2" else n_frames)
    return encoder(window_frames(video_path, t_starts, win_s, n))
