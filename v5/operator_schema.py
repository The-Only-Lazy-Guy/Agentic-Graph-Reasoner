"""LOCKED operator schema — the single source of truth for Operator Attention.

A node is a TYPED OPERATOR, not inert text. Its `op_kind` fixes the OPERATION KIND (non-
interchangeable: add / subtract / gate / transform), so structure can't be averaged away
(the fix for generic-collapse). An edge's `edge_op` fixes the WIRING role.

VALIDATED before locking (v5/optest_invalidate.py + v5/optest_reasoning.py, local 1.5B):
  - INVALIDATE (subtract) flips a frozen LM's factual belief, edge-gated (floor PASS).
  - INVALIDATE suppresses a reasoning-trap's wrong path -> the frozen LM reasons better (7/8 PASS).
So the operators do LOGIC a plain content-blend / array / RAG cannot. Per-node CONTENT stays
generic (projector-from-text collapses) -> the operator KIND is the load-bearing value, not content.

op_kind is DERIVED from node_type via OP_OF (no graph regeneration needed); the grower may also
stamp metadata["op_kind"] for clarity. Keep this map + the coverage test as the contract.
"""
from __future__ import annotations

from typing import Dict
# NOTE: gnn_encoder (torch_geometric) is imported LAZILY in _coverage_check only, so op_kind_for /
# edge_op_for stay dependency-light and safe to import into the torch-free data grower.

# ── node operation kinds (non-interchangeable) ───────────────────────────────────
#   ASSERT     +v        contribute positively (declarative evidence / grounding content)
#   INVALIDATE -v        SUBTRACT / suppress a wrong path            (PROVEN load-bearing)
#   GATE       *g        multiplicative precondition (logical AND)   (product, not sum)
#   TRANSFORM  W*state    apply a learned operator (the "how")
#   SLOT       register   writable intermediate result (DeltaNet state); written mid-reasoning
OP_KINDS = ("ASSERT", "INVALIDATE", "GATE", "TRANSFORM", "SLOT")

# node_type -> op_kind. Code types (symbol/module) appended; SLOT is emitted by the reasoning
# loop, not the grower, so no node_type maps to it here.
OP_OF: Dict[str, str] = {
    "fact": "ASSERT", "claim": "ASSERT", "application": "ASSERT",
    "solved_subgoal": "ASSERT", "reasoning_atom": "ASSERT", "reasoning_chain": "TRANSFORM",
    "strategy": "TRANSFORM", "procedure": "TRANSFORM",
    "control_rule": "GATE", "epistemic_state": "GATE",
    "failure_pattern": "INVALIDATE",
    "symbol": "ASSERT", "module": "ASSERT",
    "unknown": "ASSERT",
}

# ── edge wiring roles ────────────────────────────────────────────────────────────
#   DATAFLOW    src result feeds dst input (computation order)
#   GATES       src (precondition) must be satisfied for dst to fire
#   INVALIDATES src suppresses dst                              (PROVEN load-bearing)
#   REFINES     typed combination / generic relation
EDGE_OPS = ("DATAFLOW", "GATES", "INVALIDATES", "REFINES")

EDGEOP_OF: Dict[str, str] = {
    "contains": "DATAFLOW", "leveraged": "DATAFLOW", "chain_step": "DATAFLOW",
    "contradicts": "INVALIDATES", "invalidated_by": "INVALIDATES",
    "gates": "GATES", "precondition": "GATES", "requires": "GATES",
    "supports": "REFINES", "entails": "REFINES", "transfers_to": "REFINES",
    "related": "REFINES", "refines": "REFINES",
}


def op_kind_for(node_type: str) -> str:
    """node_type -> operation kind (default ASSERT = safe positive grounding)."""
    return OP_OF.get(node_type, "ASSERT")


def edge_op_for(relation: str) -> str:
    """relation -> wiring role (default REFINES = generic typed combine)."""
    return EDGEOP_OF.get(relation, "REFINES")


def _coverage_check():
    """Every GNN node type must have an op_kind (can't silently diverge — like the GNN drift guard)."""
    from v5.gnn_encoder import NODE_TYPE_VOCAB
    missing = [t for t in NODE_TYPE_VOCAB if t not in OP_OF]
    bad_op = [t for t, o in OP_OF.items() if o not in OP_KINDS]
    bad_edge = [r for r, o in EDGEOP_OF.items() if o not in EDGE_OPS]
    ok = not (missing or bad_op or bad_edge)
    print("node types:", len(NODE_TYPE_VOCAB), "| mapped:", len(OP_OF))
    if missing:
        print("  MISSING op_kind for node types:", missing)
    if bad_op:
        print("  INVALID op_kind values:", bad_op)
    if bad_edge:
        print("  INVALID edge_op values:", bad_edge)
    import collections
    print("op_kind distribution:", dict(collections.Counter(OP_OF.values())))
    print("edge_op distribution:", dict(collections.Counter(EDGEOP_OF.values())))
    print("COVERAGE", "OK" if ok else "FAIL")
    return ok


if __name__ == "__main__":
    raise SystemExit(0 if _coverage_check() else 1)
