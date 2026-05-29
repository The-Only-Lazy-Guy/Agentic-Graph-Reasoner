"""ActiveSubgraph: clean interface between MemoryGraph and the V5 attention adapter.

Encapsulates:
  - ordered node list + id→index mapping
  - GNN encoder inputs (text embeddings, type ids, edge index/type)
  - node_type_mask_planning  [N] bool — nodes attended at Layer 8
  - node_type_mask_evidence  [N] bool — nodes attended at Layer 20
  - invalidator_flags        [N] float — static invalidator presence (pre-loop)
  - slot_relevance           [N] float — how relevant each node is to required slots

GraphMemoryKV: structured output of RGCNEncoder.forward(), consumed by
RecurrentAttentionBlock.

See V5_ARCHITECTURE.md §6 for planning vs evidence node pool definitions.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import torch
from torch import Tensor

from v5.gnn_encoder import (
    GNN_HIDDEN_DIM,
    GraphEncoderInputs,
    NODE_TYPE_ID,
    _epistemic_status_id,
    _node_type_id,
    _relation_type_id,
    TEXT_EMBED_DIM,
)

# ── node type pools ────────────────────────────────────────────────────────────
# Layer 8 planning: strategy, failure_pattern, control_rule, reasoning_chain,
#   reasoning_atom, epistemic_state where status in {uncertain}
PLANNING_NODE_TYPES = frozenset({
    "strategy", "failure_pattern", "control_rule",
    "reasoning_chain", "reasoning_atom",
})
# epistemic_state nodes included in planning only when uncertain/open
PLANNING_EPISTEMIC_STATUSES = frozenset({"uncertain", "unknown"})

# Layer 20 evidence: fact, claim, application, solved_subgoal, procedure,
#   epistemic_state where status in {verified, supported}
EVIDENCE_NODE_TYPES = frozenset({
    "fact", "claim", "application", "solved_subgoal", "procedure",
})
EVIDENCE_EPISTEMIC_STATUSES = frozenset({"verified", "supported"})

# ── invalidator relations ─────────────────────────────────────────────────────
from reasoning.graph_relations import Rel
INVALIDATOR_RELATIONS = frozenset({Rel.INVALIDATED_BY, Rel.CONTRADICTS})


@dataclass
class GraphMemoryKV:
    """Structured output of the GNN encoder, consumed by RecurrentAttentionBlock.

    Stores raw node embeddings only. Each RecurrentAttentionBlock projects
    its own K/V via its own K_proj/V_proj, so planning and evidence blocks
    can learn different key spaces.

    All tensors share the same device.
    """
    node_embeddings: Tensor            # [N, GNN_HIDDEN_DIM]  raw R-GCN output
    node_ids: List[str]
    node_types: List[str]
    planning_mask: Tensor              # [N] bool — attend at Layer 8
    evidence_mask: Tensor              # [N] bool — attend at Layer 20
    invalidator_flags: Tensor          # [N] float — 1.0 if node has outgoing invalidator
    slot_relevance: Tensor             # [N] float — relevance to required slots (0-1)

    @property
    def num_nodes(self) -> int:
        return len(self.node_ids)

    @property
    def device(self) -> torch.device:
        return self.K.device


@dataclass
class ActiveSubgraph:
    """Pre-processed subgraph ready for GNN encoding and attention masking.

    Built by build_active_subgraph() from a MemoryGraph + pre-computed text embeddings.
    Passed into GraphAttentionInjector.prepare_session().
    """
    encoder_inputs: GraphEncoderInputs
    node_ids: List[str]
    node_types: List[str]
    planning_mask: Tensor      # [N] bool
    evidence_mask: Tensor      # [N] bool
    invalidator_flags: Tensor  # [N] float
    slot_relevance: Tensor     # [N] float
    task_frame: Optional[dict] = None


def build_active_subgraph(
    graph,
    node_ids: List[str],
    text_embeddings: Dict[str, List[float]],
    device: torch.device,
    task_frame: Optional[dict] = None,
) -> ActiveSubgraph:
    """Build an ActiveSubgraph from a MemoryGraph node subset.

    Args:
        graph: MemoryGraph instance
        node_ids: ordered list of node IDs for the subgraph (already pre-filtered)
        text_embeddings: {node_id: [768-float]} pre-computed BERT embeddings
        device: torch device
        task_frame: optional TaskFrame dict; used for slot_relevance scoring
    """
    from v5.gnn_encoder import build_encoder_inputs
    encoder_inputs = build_encoder_inputs(graph, node_ids, text_embeddings, device)

    node_types: List[str] = []
    planning_flags: List[bool] = []
    evidence_flags: List[bool] = []
    inv_flags: List[float] = []

    # Pre-build outgoing invalidator set: nodes that ARE the source of
    # an invalidated_by or contradicts edge
    invalidator_sources = {
        e.src for e in graph.edges
        if e.relation in INVALIDATOR_RELATIONS and e.src in set(node_ids)
    }

    for nid in node_ids:
        node = graph.nodes.get(nid)
        if node is None:
            node_types.append("unknown")
            planning_flags.append(False)
            evidence_flags.append(False)
            inv_flags.append(0.0)
            continue

        ntype = node.node_type
        node_types.append(ntype)

        meta = getattr(node, "metadata", {}) or {}
        status = meta.get("status", "unknown") if isinstance(meta, dict) else "unknown"

        # Planning mask: strategy/failure/control/chain/atom + uncertain epistemic
        in_planning = ntype in PLANNING_NODE_TYPES
        if ntype == "epistemic_state":
            in_planning = status in PLANNING_EPISTEMIC_STATUSES
        planning_flags.append(in_planning)

        # Evidence mask: fact/claim/application/subgoal/procedure + verified epistemic
        in_evidence = ntype in EVIDENCE_NODE_TYPES
        if ntype == "epistemic_state":
            in_evidence = status in EVIDENCE_EPISTEMIC_STATUSES
        evidence_flags.append(in_evidence)

        inv_flags.append(1.0 if nid in invalidator_sources else 0.0)

    # Slot relevance: how relevant each node is to required slot names
    required_slots = list((task_frame or {}).get("required_slots") or [])
    slot_relevance = _compute_slot_relevance(graph, node_ids, required_slots)

    return ActiveSubgraph(
        encoder_inputs=encoder_inputs,
        node_ids=node_ids,
        node_types=node_types,
        planning_mask=torch.tensor(planning_flags, dtype=torch.bool, device=device),
        evidence_mask=torch.tensor(evidence_flags, dtype=torch.bool, device=device),
        invalidator_flags=torch.tensor(inv_flags, dtype=torch.float32, device=device),
        slot_relevance=torch.tensor(slot_relevance, dtype=torch.float32, device=device),
        task_frame=task_frame,
    )


def _compute_slot_relevance(graph, node_ids: List[str], required_slots: List[str]) -> List[float]:
    """Score each node by how many required slot names appear in its text/metadata."""
    if not required_slots:
        return [0.5] * len(node_ids)

    scores = []
    slot_tokens = {s.lower() for s in required_slots}
    for nid in node_ids:
        node = graph.nodes.get(nid)
        if node is None:
            scores.append(0.0)
            continue
        text = (node.text or "").lower()
        meta_str = str(getattr(node, "metadata", "") or "").lower()
        combined = text + " " + meta_str
        hits = sum(1 for tok in slot_tokens if tok in combined)
        scores.append(min(1.0, hits / max(1, len(slot_tokens))))
    return scores
