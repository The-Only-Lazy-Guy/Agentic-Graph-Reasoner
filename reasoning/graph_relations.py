"""Graph relation type registry.

All valid `Edge.relation` string values are defined here with documentation.
This file is the single source of truth for relation semantics.

The graph_core.Edge.relation field is a free string, so nothing enforces
these values at runtime — but all code that creates edges MUST use constants
from this module to stay consistent and searchable.

Usage:
    from reasoning.graph_relations import Rel
    edge = Edge(src="strat_abc", dst="fact_123", relation=Rel.LEVERAGED)

GNN edge feature mapping:
    Each relation type maps to a numeric type_id for GNN embedding lookup.
    See RELATION_TYPE_ID for the mapping. Add new relations at the END to
    avoid changing existing IDs (would invalidate trained GNN weights).
"""
from __future__ import annotations

from typing import Dict, FrozenSet


class Rel:
    """Namespace of all valid Edge.relation values."""

    # ------------------------------------------------------------------
    # Core knowledge relations (V4 — stable)
    # ------------------------------------------------------------------

    # Node A entails node B: if A is true, B follows logically.
    # Direction: A ──entails──> B
    ENTAILS = "entails"

    # Node A contradicts node B: A and B cannot both be true.
    # Direction: bidirectional in practice, but stored as directed.
    CONTRADICTS = "contradicts"

    # Node A partially overlaps with node B in meaning or scope.
    # Use when entails is too strong but the nodes are meaningfully related.
    OVERLAPS = "overlaps"

    # Node A supports node B: A provides evidence for B without entailing it.
    # Weaker than entails; used for probabilistic or analogical support.
    SUPPORTS = "supports"

    # A strategy/solved_subgoal/reasoning_chain leveraged this fact/claim.
    # Direction: strategy ──leveraged──> fact
    LEVERAGED = "leveraged"

    # Generic semantic relatedness. Use as a fallback when no specific
    # relation fits. The GNN treats this as low-signal.
    RELATED = "related"

    # Session memory → related graph node (created by session_to_graph.py).
    USED_SIGNAL = "used_signal"

    # An epistemic_state node describes the belief status of a target node.
    # Direction: epistemic_state ──epistemic_of──> target_node
    EPISTEMIC_OF = "epistemic_of"

    # ------------------------------------------------------------------
    # V5 additions — meta-reasoning control
    # ------------------------------------------------------------------

    # A claim/strategy/shortcut is invalidated by a specific condition.
    # Direction: claim ──invalidated_by──> condition_node
    #
    # Semantics: "This claim/shortcut is UNSAFE to use when the condition
    # described by the destination node is true."
    #
    # Use cases:
    #   - solved_subgoal ──invalidated_by──> condition ("negative edges present")
    #   - strategy ──invalidated_by──> context_node ("user asks about DAG, not general graph")
    #   - reasoning_chain ──invalidated_by──> exception_node ("only holds for integer weights")
    #
    # GNN role: during the Layer 20 evidence pass, if the model attends to a
    # shortcut node AND that node has an active invalidated_by edge whose
    # destination matches the current question context, the model should NOT
    # finalize — it must verify further.
    INVALIDATED_BY = "invalidated_by"

    # A strategy/procedure/answer requires a specific task-frame slot to be filled.
    # Direction: strategy ──requires_slot──> slot_node (or stored as metadata)
    #
    # Semantics: "This strategy cannot produce a complete answer unless the
    # named slot is filled. If the slot is missing, the model must ask for it
    # or derive it from the graph before finalizing."
    #
    # Use cases:
    #   - strategy_algorithm_applicability ──requires_slot──> verdict
    #   - strategy_algorithm_applicability ──requires_slot──> reason
    #   - strategy_algorithm_applicability ──requires_slot──> alternative
    #   - strategy_algorithm_applicability ──requires_slot──> caveat
    #
    # This makes slot requirements first-class graph structure instead of
    # prompt instructions. The GNN can propagate "missing slot" signals
    # directly from the graph rather than relying on system prompt parsing.
    #
    # Note: The destination node of requires_slot is typically a task_frame_item
    # or a dedicated slot_descriptor node (plain claim node with node_type="claim"
    # and metadata={"slot_name": "verdict"}).
    REQUIRES_SLOT = "requires_slot"

    # A reasoning atom/strategy/chain transfers its structure to a different domain.
    # Direction: source ──transfers_to──> application_node
    #
    # Semantics: "The reasoning structure of the source node applies analogically
    # to the problem or domain described by the destination node."
    #
    # Use cases:
    #   - reasoning_atom "monotonic invariant allows binary search"
    #       ──transfers_to──> application "parametric search"
    #   - strategy "rank via cumulative frequency"
    #       ──transfers_to──> application "Fenwick tree leaderboard design"
    #   - solved_subgoal "Dijkstra on non-negative graph"
    #       ──transfers_to──> application "A* with admissible heuristic"
    #
    # GNN role: transfers_to edges enable the GNN to propagate reasoning
    # patterns across domains. During the Layer 8 planning pass, if the model
    # attends to a strategy node, the GNN also surfaces the transfer targets
    # as candidate planning anchors for analogical reasoning.
    #
    # This is the mechanism for: "This problem has the same structure as
    # another known problem I have seen."
    TRANSFERS_TO = "transfers_to"

    # A reasoning_chain's steps are connected by chain_step edges.
    # Direction: reasoning_chain ──chain_step──> intermediate_node (ordered)
    CHAIN_STEP = "chain_step"


# ------------------------------------------------------------------
# Numeric IDs for GNN edge embedding lookup
# ------------------------------------------------------------------
# IMPORTANT: Never reorder or remove entries — this would invalidate
# trained GNN weights. Only append new entries at the end.

RELATION_TYPE_ID: Dict[str, int] = {
    Rel.ENTAILS:       0,
    Rel.CONTRADICTS:   1,
    Rel.OVERLAPS:      2,
    Rel.SUPPORTS:      3,
    Rel.LEVERAGED:     4,
    Rel.RELATED:       5,
    Rel.USED_SIGNAL:   6,
    Rel.EPISTEMIC_OF:  7,
    Rel.INVALIDATED_BY: 8,
    Rel.REQUIRES_SLOT:  9,
    Rel.TRANSFERS_TO:  10,
    Rel.CHAIN_STEP:    11,
}

# Inverse lookup: type_id → relation string
RELATION_FROM_ID: Dict[int, str] = {v: k for k, v in RELATION_TYPE_ID.items()}

# Relations that carry strong negative signal (contradiction/invalidation)
NEGATIVE_RELATIONS: FrozenSet[str] = frozenset({
    Rel.CONTRADICTS,
    Rel.INVALIDATED_BY,
})

# Relations that carry strong positive signal (entailment/support)
POSITIVE_RELATIONS: FrozenSet[str] = frozenset({
    Rel.ENTAILS,
    Rel.SUPPORTS,
    Rel.LEVERAGED,
})

# Relations used during Layer 8 GNN planning pass propagation
PLANNING_PASS_RELATIONS: FrozenSet[str] = frozenset({
    Rel.LEVERAGED,
    Rel.TRANSFERS_TO,
    Rel.CHAIN_STEP,
    Rel.REQUIRES_SLOT,
    Rel.EPISTEMIC_OF,
})

# Relations used during Layer 20 GNN evidence pass propagation
EVIDENCE_PASS_RELATIONS: FrozenSet[str] = frozenset({
    Rel.ENTAILS,
    Rel.SUPPORTS,
    Rel.CONTRADICTS,
    Rel.OVERLAPS,
    Rel.INVALIDATED_BY,
    Rel.EPISTEMIC_OF,
})


def relation_type_id(relation: str) -> int:
    """Return the numeric type_id for a relation string.

    Unknown relations get ID = len(RELATION_TYPE_ID) (treated as RELATED
    by the GNN with a generic embedding).
    """
    return RELATION_TYPE_ID.get(relation, len(RELATION_TYPE_ID))


def is_negative(relation: str) -> bool:
    return relation in NEGATIVE_RELATIONS


def is_positive(relation: str) -> bool:
    return relation in POSITIVE_RELATIONS
