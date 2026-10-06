"""Classification pipelines: text embeddings of the clip documents (a small frozen sentence-embedding model, no training, no generation).

The embeddings are cached with a fingerprint of the texts: if any document changes (captions switched on, thresholds changed) they are recomputed.
"""
import hashlib
from pathlib import Path

import numpy as np


def fingerprint(texts, model_id):
    h = hashlib.sha1(model_id.encode("utf-8"))
    for t in texts:
        h.update(b"\x00" + t.encode("utf-8"))
    return h.hexdigest()


def embed_texts(texts, model_id, embed_fn=None):
    """-> (n, D) float32, L2-normalised. embed_fn(list of str) -> array replaces the real model (tests)."""
    if embed_fn is None:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(model_id)
        embed_fn = lambda t: model.encode(t, normalize_embeddings=True, batch_size=32, show_progress_bar=False)
    e = np.asarray(embed_fn(list(texts)), np.float32)
    return e.reshape(len(texts), -1)


def cached_embeddings(texts, model_id, cache_path, embed_fn=None):
    """Embeddings of `texts`, from cache_path (.npz) when its fingerprint matches, else computed and saved."""
    p = Path(cache_path)
    fp = fingerprint(texts, model_id)
    if p.exists():
        z = np.load(p, allow_pickle=False)
        if str(z["fp"]) == fp:
            return z["emb"]
    e = embed_texts(texts, model_id, embed_fn)
    p.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(p, fp=np.array(fp), emb=e)
    return e
