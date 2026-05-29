"""
Anchor retrieval v2 for Answerer-v2.

The legacy `answerer_v1.retrieve_anchors` is pure lexical Jaccard with a
0.20 * importance bonus. That bonus dominates whenever lexical scores
are weak, which pulls high-importance hubs into the anchor set
regardless of question relevance (see the BFS/DFS run that pulled
`fermat_inverse_*` into the anchors).

This module replaces that with a combined-signal scorer plus an optional
MMR selection step so anchors:
  - match the question both lexically AND semantically
  - prefer specific evidence over hubs / bridges / summaries
  - prefer concrete `_apply` nodes
  - demote misconception `_false` and hypothesis `_hyp` nodes
  - cover a diverse region of the graph instead of clustering

Strategies:
  legacy  - delegate to answerer_v1.retrieve_anchors (kept for A/B)
  topk    - score every node, return the top-k
  mmr     - score every node, then greedy MMR selection with diversity
            penalty `lambda * max_cos(pick, already_picked)`

Embeddings are cached per-graph on disk under cache/anchor_embeddings/.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np

from graph_core import MemoryGraph, Node, lexical_overlap
from embedder import encode_batch, encode_one, EMBED_DIM


# ---------------------------------------------------------------------------
# Scoring weights — tuned to demote hubs and surface specific evidence.
# Subject to A/B revision; if a tuning round changes any, update here.
# ---------------------------------------------------------------------------

WEIGHT_LEXICAL = 0.45
WEIGHT_EMBED = 0.45
WEIGHT_IMPORTANCE = 0.05

NODE_TYPE_PRIOR = {
    "concept":   0.00,
    "evidence": +0.10,
    "bridge":   +0.05,
    "summary":  -0.05,
    "hub":      -0.15,
    "example":  +0.05,
}

ID_PATTERN_PRIOR: List[Tuple[str, float]] = [
    ("_hub",      -0.15),
    ("_bridge",   -0.05),
    ("_false",    -0.10),
    ("_hyp",      -0.10),
    ("_summary",  -0.05),
    ("_apply",    +0.05),
]


# ---------------------------------------------------------------------------
# Per-graph embedding cache
# ---------------------------------------------------------------------------

def _graph_text_hash(graph: MemoryGraph) -> str:
    items = sorted(
        (nid, (node.text or "")) for nid, node in graph.nodes.items()
    )
    payload = json.dumps(items, ensure_ascii=False, sort_keys=True)
    return hashlib.md5(payload.encode("utf-8")).hexdigest()[:16]


def _cache_path(graph_basename: str, text_hash: str, cache_dir: str) -> Path:
    return Path(cache_dir) / f"{graph_basename}__{text_hash}.npz"


def get_graph_embeddings(
    graph: MemoryGraph,
    *,
    graph_basename: str = "graph",
    cache_dir: str = "cache/anchor_embeddings",
) -> Tuple[List[str], np.ndarray]:
    """Return (node_ids, embeddings[N, EMBED_DIM]) for every node in
    the graph. Cached to disk keyed by (basename, text_hash) so a graph
    edit invalidates only its own cache row."""
    text_hash = _graph_text_hash(graph)
    cpath = _cache_path(graph_basename, text_hash, cache_dir)
    if cpath.exists():
        try:
            data = np.load(cpath, allow_pickle=False)
            return list(data["ids"]), data["emb"].astype(np.float32)
        except Exception:
            pass  # corrupted; recompute

    node_ids = list(graph.nodes.keys())
    texts = [graph.nodes[nid].text or "" for nid in node_ids]
    emb = encode_batch(texts)

    cpath.parent.mkdir(parents=True, exist_ok=True)
    try:
        np.savez_compressed(cpath, ids=np.array(node_ids), emb=emb)
    except Exception:
        pass  # caching is best-effort
    return node_ids, emb


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def _node_type_prior(node) -> float:
    nt = (getattr(node, "node_type", "") or "").lower()
    return NODE_TYPE_PRIOR.get(nt, 0.0)


def _id_pattern_prior(node_id: str) -> float:
    nid = (node_id or "").lower()
    total = 0.0
    for suffix, w in ID_PATTERN_PRIOR:
        if nid.endswith(suffix) or (suffix.strip("_") in nid and suffix.startswith("_") and suffix.endswith("_") is False):
            # endswith covers the typical _false / _hub / _apply case
            total += w
    return total


def _score_nodes(
    question: str,
    graph: MemoryGraph,
    node_ids: List[str],
    node_emb: np.ndarray,
    q_emb: np.ndarray,
) -> np.ndarray:
    """Return raw scores aligned with node_ids."""
    scores = np.zeros(len(node_ids), dtype=np.float32)
    cos = node_emb @ q_emb  # [N], unit-norm * unit-norm = cos
    for i, nid in enumerate(node_ids):
        node = graph.nodes.get(nid)
        if node is None:
            continue
        lex = lexical_overlap(question, node.text or "")
        importance = float(getattr(node, "importance", 0.0))
        scores[i] = (
            WEIGHT_LEXICAL * lex
            + WEIGHT_EMBED * float(cos[i])
            + WEIGHT_IMPORTANCE * importance
            + _node_type_prior(node)
            + _id_pattern_prior(nid)
        )
    return scores


# ---------------------------------------------------------------------------
# Selection strategies
# ---------------------------------------------------------------------------

def _select_topk(scores: np.ndarray, node_ids: List[str], k: int) -> List[str]:
    order = np.argsort(-scores)
    return [node_ids[i] for i in order[:k]]


def _select_mmr(
    scores: np.ndarray,
    node_emb: np.ndarray,
    node_ids: List[str],
    k: int,
    lam: float,
) -> List[str]:
    """Greedy MMR: pick = argmax(score - lam * max_cos(pick, already_picked))."""
    picked: List[int] = []
    candidates = list(range(len(node_ids)))
    # Always seed with the top-scoring node.
    order = list(np.argsort(-scores))
    if not order:
        return []
    seed = order[0]
    picked.append(seed)
    candidates.remove(seed)
    while len(picked) < k and candidates:
        picked_emb = node_emb[picked]  # [P, D]
        cand_emb = node_emb[candidates]  # [C, D]
        sim_to_picked = cand_emb @ picked_emb.T  # [C, P]
        max_sim = sim_to_picked.max(axis=1)  # [C]
        adj_scores = scores[candidates] - lam * max_sim
        best_local = int(np.argmax(adj_scores))
        best_global = candidates[best_local]
        picked.append(best_global)
        candidates.remove(best_global)
    return [node_ids[i] for i in picked]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def retrieve_anchors_v2(
    question: str,
    graph: MemoryGraph,
    *,
    k: int = 8,
    strategy: str = "topk",
    mmr_lambda: float = 0.3,
    graph_basename: str = "graph",
    cache_dir: str = "cache/anchor_embeddings",
) -> List[str]:
    """Pick k anchor node ids for the question.

    Args:
        strategy: "legacy" | "topk" | "mmr"
                  legacy = answerer_v1.retrieve_anchors (untouched)
                  topk   = combined-signal score, pure top-k  (DEFAULT,
                           empirical winner — see
                           artifacts/anchor_strategy_compare.json)
                  mmr    = combined-signal score with diversity penalty.
                           Empirically WORSE than topk on counterfactual
                           rows where required evidence clusters tightly;
                           kept available for future tuning.
        mmr_lambda: ignored when strategy != "mmr"
    """
    if strategy == "legacy":
        from answerer_v1 import retrieve_anchors as legacy_retrieve
        return legacy_retrieve(question, graph, k=k)

    if not graph.nodes:
        return []

    node_ids, node_emb = get_graph_embeddings(
        graph, graph_basename=graph_basename, cache_dir=cache_dir,
    )

    # Exclude deprecated nodes (misconceptions flagged by post-processing)
    keep = [i for i, nid in enumerate(node_ids)
            if not graph.nodes.get(nid, Node(id="", text="")).metadata.get("deprecated")]
    if keep and len(keep) < len(node_ids):
        node_ids = [node_ids[i] for i in keep]
        node_emb = node_emb[keep]

    q_emb = encode_one(question)
    scores = _score_nodes(question, graph, node_ids, node_emb, q_emb)

    if strategy == "topk":
        return _select_topk(scores, node_ids, k)
    if strategy == "mmr":
        return _select_mmr(scores, node_emb, node_ids, k, mmr_lambda)
    raise ValueError(f"unknown strategy: {strategy!r}")


# ---------------------------------------------------------------------------
# Stage 3 overlay: role/focus-aware additive bonus on top of evidence_score.
#
# Design (locked):
#   - Pure additive overlay. Returns 0.0 with no focus_text.
#   - Does NOT change the base evidence_score weights.
#   - Three signals:
#       focus_bonus      = W_FOCUS     * focus_relevance
#       role_bonus       = W_ROLE      * role_match * focus_relevance   (gated)
#       contradict_bonus = W_CONTRADIC * contradict_pair_signal
#
#   - role_bonus is GATED by focus_relevance so a generic "mechanism"
#     node doesn't get a free boost in mechanism-seeking plan_steps unless
#     it also matches the focus topic.
#
#   - contradict_pair_signal fires only when:
#       (a) the node is endpoint of any contradict edge in the session, AND
#       (b) the OPPOSITE endpoint has focus_relevance >= threshold, AND
#       (c) focus_roles overlap with the refutation-flavored set.
#     This preserves Stage 0 safety: misconceptions still need a structural
#     reason to surface (a real contradict pair where both sides matter).
#
# Weights are module-level constants so they can be ablated in 30s.
# Defaults are CONSERVATIVE; tune up if offline diff says we need more.
# ---------------------------------------------------------------------------

STAGE3_FOCUS_W = 0.15
STAGE3_ROLE_W = 0.15
STAGE3_CONTRADICT_W = 0.10
STAGE3_CONTRADICT_RELEVANCE_THRESHOLD = 0.20
STAGE3_CONTRADICT_TRIGGER_ROLES = {"misconception", "condition"}


def stage3_overlay_score(
    session,  # SessionGraph
    node_id: str,
    *,
    focus_text: str = "",
    focus_roles=None,
    w_focus: float = None,
    w_role: float = None,
    w_contradict: float = None,
) -> float:
    """Additive Stage 3 score overlay. Returns 0.0 when focus is empty.

    Weight kwargs let the offline diff harness sweep configurations
    without touching the source constants.
    """
    if not focus_text:
        return 0.0
    node = session.nodes.get(node_id)
    if node is None:
        return 0.0

    wf = w_focus if w_focus is not None else STAGE3_FOCUS_W
    wr = w_role if w_role is not None else STAGE3_ROLE_W
    wc = w_contradict if w_contradict is not None else STAGE3_CONTRADICT_W

    # Local import avoids any circular module hazard.
    from roles import detect_node_roles, ACTIVE_STAGE3_ROLES

    # 1. focus relevance (lex-only v1; embedding cos can be layered later)
    focus_rel = float(lexical_overlap(focus_text, node.text or ""))

    # 2. role match — node and focus must share at least one ACTIVE role
    node_roles_full = detect_node_roles(node)
    node_roles = node_roles_full & ACTIVE_STAGE3_ROLES
    role_match = 1.0 if (focus_roles and (node_roles & set(focus_roles))) else 0.0

    # 3. contradict_pair signal — needs structural pair AND focus-relevant
    #    opposite endpoint AND a refutation-flavored focus role
    contradict_signal = 0.0
    if focus_roles and (set(focus_roles) & STAGE3_CONTRADICT_TRIGGER_ROLES):
        for e in session.edges:
            if e.relation != "contradict":
                continue
            if e.src == node_id:
                other = e.dst
            elif e.dst == node_id:
                other = e.src
            else:
                continue
            other_node = session.nodes.get(other)
            if other_node is None:
                continue
            other_rel = float(lexical_overlap(focus_text, other_node.text or ""))
            if other_rel >= STAGE3_CONTRADICT_RELEVANCE_THRESHOLD:
                contradict_signal = 1.0
                break

    return (
        wf * focus_rel
        + wr * role_match * focus_rel
        + wc * contradict_signal
    )


# ---------------------------------------------------------------------------
# Diagnostic: how well do retrieved anchors cover the required_evidence?
# ---------------------------------------------------------------------------

def anchor_quality(
    anchors: Sequence[str], required_evidence: Sequence[str],
) -> float:
    """Fraction of required_evidence ids that appear in the anchor set."""
    if not required_evidence:
        return 1.0
    anchor_set = set(anchors)
    hit = sum(1 for e in required_evidence if e in anchor_set)
    return hit / len(required_evidence)
