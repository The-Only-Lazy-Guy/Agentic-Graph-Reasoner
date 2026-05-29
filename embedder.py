"""
Shared MiniLM embedder (raw HuggingFace transformers path).

We do not import sentence_transformers — it segfaults during init on this
env. Raw transformers + mean-pool + L2 normalize gives the same sentence
embeddings.

Both novelty_metrics.py and anchor_retrieval.py depend on this so the
model loads exactly once per process.
"""

from __future__ import annotations

import os
from typing import List, Tuple

import numpy as np

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EMBED_DIM = 384

_TOK = None
_MODEL = None


def get_embedder():
    """Lazy-load tokenizer + model; cached for the process lifetime."""
    global _TOK, _MODEL
    if _MODEL is None:
        os.environ.setdefault("HF_HOME", os.path.join(os.getcwd(), "cache"))
        from transformers import AutoTokenizer, AutoModel
        _TOK = AutoTokenizer.from_pretrained(MODEL_NAME)
        _MODEL = AutoModel.from_pretrained(MODEL_NAME)
        _MODEL.eval()
    return _TOK, _MODEL


def encode_batch(texts: List[str], *, max_length: int = 128) -> np.ndarray:
    """Mean-pooled, L2-normalized embeddings for a list of strings.

    Returns [N, EMBED_DIM] float32 numpy array. Rows are unit-norm so
    cosine similarity reduces to dot product.
    """
    import torch
    tok, model = get_embedder()
    if not texts:
        return np.zeros((0, EMBED_DIM), dtype=np.float32)
    inputs = tok(
        list(texts),
        padding=True,
        truncation=True,
        return_tensors="pt",
        max_length=max_length,
    )
    with torch.no_grad():
        outs = model(**inputs)
    mask = inputs["attention_mask"].unsqueeze(-1).float()
    pooled = (outs.last_hidden_state * mask).sum(1) / mask.sum(1).clamp(min=1)
    pooled = torch.nn.functional.normalize(pooled, p=2, dim=-1)
    return pooled.cpu().numpy().astype(np.float32)


def encode_one(text: str) -> np.ndarray:
    """Convenience for single-string encoding. Returns [EMBED_DIM]."""
    arr = encode_batch([text])
    return arr[0]
