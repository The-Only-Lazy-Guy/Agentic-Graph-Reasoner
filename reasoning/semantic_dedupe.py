"""Semantic deduplication via embedding cosine similarity.

Replaces the heuristic text-match dedupe with embedding-based detection.
Reuses the cached embeddings from anchor_retrieval (same model, same cache
dir) so there's zero additional model-load overhead.

A proposed text is a duplicate if its cosine similarity to ANY existing
node exceeds `threshold` (default 0.92). The ambiguous zone (0.80-0.92)
flags candidates for LLM judge review rather than auto-deciding.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from anchor_retrieval import get_graph_embeddings
from embedder import encode_one
from graph_core import MemoryGraph


@dataclass
class DedupeMatch:
    node_id: str
    similarity: float


class DedupeIndex:
    """Cosine similarity index over all graph nodes' text embeddings."""

    def __init__(self, node_ids: List[str], emb: np.ndarray):
        self.node_ids = list(node_ids)
        # Unit-normalize rows for dot-product = cosine similarity.
        norms = np.linalg.norm(emb, axis=1, keepdims=True)
        norms = np.where(norms < 1e-9, 1.0, norms)
        self.emb = emb / norms  # [N, 384]

    def query(
        self,
        text: str,
        threshold: float = 0.92,
    ) -> Optional[DedupeMatch]:
        """Return the best match above threshold, or None."""
        q = encode_one(text)
        q = q / (np.linalg.norm(q) + 1e-9)
        sims = self.emb @ q  # [N]
        idx = int(np.argmax(sims))
        best_sim = float(sims[idx])
        if best_sim >= threshold:
            return DedupeMatch(node_id=self.node_ids[idx], similarity=best_sim)
        return None

    def query_topk(
        self,
        text: str,
        k: int = 3,
    ) -> List[DedupeMatch]:
        """Return the top-k most similar nodes (regardless of threshold)."""
        q = encode_one(text)
        q = q / (np.linalg.norm(q) + 1e-9)
        sims = self.emb @ q
        top_indices = np.argsort(sims)[::-1][:k]
        return [
            DedupeMatch(node_id=self.node_ids[int(i)], similarity=float(sims[i]))
            for i in top_indices
        ]

    def classify(
        self,
        text: str,
        dup_threshold: float = 0.92,
        ambiguous_threshold: float = 0.80,
    ) -> tuple:
        """Classify a proposed text as duplicate / ambiguous / novel.

        Returns (status, best_match):
          - ("duplicate", DedupeMatch) — cosine >= dup_threshold
          - ("ambiguous", DedupeMatch) — ambiguous_threshold <= cosine < dup_threshold
          - ("novel", None)            — cosine < ambiguous_threshold
        """
        q = encode_one(text)
        q = q / (np.linalg.norm(q) + 1e-9)
        sims = self.emb @ q
        idx = int(np.argmax(sims))
        best_sim = float(sims[idx])
        match = DedupeMatch(node_id=self.node_ids[idx], similarity=best_sim)
        if best_sim >= dup_threshold:
            return "duplicate", match
        if best_sim >= ambiguous_threshold:
            return "ambiguous", match
        return "novel", None


def build_dedupe_index(
    graph: MemoryGraph,
    graph_basename: str = "graph",
    cache_dir: str = "cache/anchor_embeddings",
) -> DedupeIndex:
    """Build a DedupeIndex from the graph's cached embeddings."""
    node_ids, node_emb = get_graph_embeddings(
        graph, graph_basename=graph_basename, cache_dir=cache_dir,
    )
    return DedupeIndex(list(node_ids), node_emb)
