"""Worlds graph + tools for v6. IMPORTS from v5, does not reimplement.

The worlds graph is the long-term memory: nodes carry real MiniLM embeddings, typed edges, and a
node can CONTAIN a smaller graph (real 'contains'/'in' edges, kept out of the retrieval matrix so a
world cannot outrank the members it summarises). Retrieval is two-level: route to worlds in O(W),
then rank inside. That is what makes the vocabulary OPEN -- the retrieval gate can pull any node,
including ones added after training, so novelty comes from the graph rather than a fixed menu.

Measured properties carried over from v5, so v6 does not have to rediscover them:
  * worlds buy SPEED, not accuracy: ~782 candidates gives R@5 18.5% / R@20 38.0% against flat
    19.0% / 38.0%. Budget CANDIDATES, not top_w -- a fixed top_w silently scanned 0.65% of a
    9,264-node graph and handed back near-random atoms.
  * cosine ORDERING is misleading while cosine MEMBERSHIP is informative: a random draw from the
    top-5 beat rank-1. Treat retrieval as a set, not a ranking.
"""
from __future__ import annotations

import sys
from pathlib import Path

_ROOT = str(Path(__file__).resolve().parents[1])
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import numpy as np

from v5.runtime.membrane import Atom, AtomGraph          # noqa: E402
from v5.runtime.algo_grr_agent import ToolRegistry, default_registry  # noqa: E402

__all__ = ["Atom", "AtomGraph", "ToolRegistry", "default_registry",
           "build_graph", "retrieve"]


def build_graph(nodes: dict[str, str], worlds: bool = True,
                join_threshold: float = 0.55) -> AtomGraph:
    """nodes: {name: description}. Returns a worlds-enabled AtomGraph.

    Node kinds are FREE TEXT in this graph -- concept / trap / procedure / tool all coexist, which
    is what lets one store hold trivial knowledge, recorded mistakes, and executable skills without
    a schema change.
    """
    g = AtomGraph()
    if worlds:
        g.enable_worlds(join_threshold=join_threshold, materialize=True)
    for name, desc in nodes.items():
        # DESCRIPTION ONLY for the retrieval embedding. Putting the opaque name in the embedded
        # text was tried and REVERTED: MiniLM has no semantics for "op_wd8", so it encodes surface
        # form and every node shares the "op_" prefix. Measured mean pairwise cosine over 28 nodes:
        #     description only        0.3886   <- keep this
        #     "op_xxx: description"   0.5048   <- worse
        #     opaque names alone      0.6434
        # The name still has to reach the LM, but through the LM's OWN embedding table, not through
        # MiniLM -- see content_embeddings(). MiniLM retrieves, native injects.
        atom = Atom(name=name, code="", description=desc, kind="concept", provenance="v6")
        if worlds:
            g.add_or_merge_world(atom)
        else:
            g.add_or_merge(atom)
    return g


def retrieve(g: AtomGraph, query: str, k: int = 4, min_candidates: int = 256) -> list[str]:
    """Two-level retrieval, returning a SET of k names (order is not meaningful -- see module note)."""
    if getattr(g, "_worlds_on", False) and getattr(g, "worlds", None):
        return g.cosine_rank_world(query, k=k, min_candidates=min_candidates)
    return g.cosine_rank(query, k=k)


def content_embeddings(wb, names: list[str]) -> "torch.Tensor":
    """[N, d_lm] rows: each node's NAME through the LM's OWN embedding table.

    This is what gets WRITTEN into the slots. Retrieval decides WHICH node (MiniLM, on
    descriptions, which separate at 0.3886); the slot then carries that node's identifier in the
    space the LM actually reads, so there is no cross-model bridge for CE to collapse. v5's own
    note: "MiniLM stays fine for cheap cosine RETRIEVAL, which never goes through the adapter" --
    and then it routed the adapter through MiniLM anyway.
    """
    import torch
    from v5.runtime.trm_wm import native_text_embedding
    return torch.stack([native_text_embedding(wb, n) for n in names]).float()


def node_embeddings(g: AtomGraph, names: list[str]) -> np.ndarray:
    """[N, 384] MiniLM rows for the given nodes, in the order given."""
    return np.stack([g.atoms[n].emb for n in names]).astype(np.float32)
