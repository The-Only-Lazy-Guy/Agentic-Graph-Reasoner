"""Retrieval with failure-pattern boost.

Wraps anchor_retrieval's primitives to give `failure_pattern`-typed nodes
a multiplicative score boost before top-k selection. Reasoning: anti-
patterns are high-information when relevant — a single "don't do X here"
is worth several confirming facts in steering the model away from a
known failure mode. See PHASE1_PLAN.md §7.

Boost factor default: 1.4× (resolved decision). Tunable per call.

If anchor_retrieval's internals change, this module is the canary — the
import below is the documented coupling point.
"""
from __future__ import annotations

from typing import List, Optional

import numpy as np

# Documented coupling: we use private _score_nodes + _select_topk from
# anchor_retrieval to stay byte-equivalent with the production retrieval
# pipeline. If those symbols change shape, update this module too.
from anchor_retrieval import _score_nodes, _select_topk, get_graph_embeddings
from graph_core import lexical_overlap
from graph_core import MemoryGraph


def _lexical_topk_with_failure_boost(
    question: str,
    graph: MemoryGraph,
    *,
    k: int,
    failure_boost: float,
) -> List[str]:
    scored = []
    for node in graph.nodes.values():
        score = lexical_overlap(question, node.text or "")
        score += 0.05 * float(getattr(node, "importance", 0.0) or 0.0)
        if getattr(node, "node_type", "") == "failure_pattern":
            score *= failure_boost
        scored.append((score, node.id))
    scored.sort(key=lambda item: item[0], reverse=True)
    return [nid for _score, nid in scored[:k]]


def retrieve_with_failure_boost(
    question: str,
    graph: MemoryGraph,
    *,
    k: int = 12,
    failure_boost: float = 1.4,
    graph_basename: str = "graph",
    cache_dir: str = "cache/anchor_embeddings",
) -> List[str]:
    """Top-k anchor node IDs with failure_pattern nodes' scores
    multiplied by `failure_boost` before sorting.

    Returns: list of node IDs of length min(k, |graph.nodes|).

    Edge cases:
      - Empty graph: returns []
      - No failure_pattern nodes in graph: behaves identically to the
        unboosted top-k path.
      - failure_boost == 1.0: identical to retrieve_anchors_v2(strategy='topk')
    """
    if k <= 0 or not graph.nodes:
        return []

    try:
        from embedder import encode_one

        node_ids, node_emb = get_graph_embeddings(
            graph, graph_basename=graph_basename, cache_dir=cache_dir,
        )
        q_emb = encode_one(question)
        scores = _score_nodes(question, graph, node_ids, node_emb, q_emb)
    except Exception:
        return _lexical_topk_with_failure_boost(
            question,
            graph,
            k=k,
            failure_boost=failure_boost,
        )

    # Apply boost to failure_pattern nodes.
    boosted = scores.copy()
    for i, nid in enumerate(node_ids):
        node = graph.nodes.get(nid)
        if node is None:
            continue
        ntype = getattr(node, "node_type", None) or ""
        if ntype == "failure_pattern":
            boosted[i] *= failure_boost

    return _select_topk(boosted, node_ids, k)


def diagnose_boost_effect(
    question: str,
    graph: MemoryGraph,
    *,
    k: int = 12,
    failure_boost: float = 1.4,
    graph_basename: str = "graph",
    cache_dir: str = "cache/anchor_embeddings",
) -> dict:
    """Side-by-side report of which IDs are in top-k with vs without boost.

    Use this during eval to see whether the boost is actually changing
    retrieval outcomes on real questions. Doesn't return scores
    (they're sometimes negative and not directly comparable across
    runs), just set membership.
    """
    if not graph.nodes:
        return {"unboosted": [], "boosted": [], "added_by_boost": [], "removed_by_boost": []}

    try:
        from embedder import encode_one

        node_ids, node_emb = get_graph_embeddings(
            graph, graph_basename=graph_basename, cache_dir=cache_dir,
        )
        q_emb = encode_one(question)
        scores = _score_nodes(question, graph, node_ids, node_emb, q_emb)
        unboosted = _select_topk(scores, node_ids, k)

        boosted_scores = scores.copy()
        for i, nid in enumerate(node_ids):
            node = graph.nodes.get(nid)
            if node is None:
                continue
            if getattr(node, "node_type", "") == "failure_pattern":
                boosted_scores[i] *= failure_boost
        boosted = _select_topk(boosted_scores, node_ids, k)
    except Exception:
        unboosted = _lexical_topk_with_failure_boost(
            question,
            graph,
            k=k,
            failure_boost=1.0,
        )
        boosted = _lexical_topk_with_failure_boost(
            question,
            graph,
            k=k,
            failure_boost=failure_boost,
        )

    unboosted_set = set(unboosted)
    boosted_set = set(boosted)
    return {
        "unboosted": unboosted,
        "boosted": boosted,
        "added_by_boost": [nid for nid in boosted if nid not in unboosted_set],
        "removed_by_boost": [nid for nid in unboosted if nid not in boosted_set],
    }
