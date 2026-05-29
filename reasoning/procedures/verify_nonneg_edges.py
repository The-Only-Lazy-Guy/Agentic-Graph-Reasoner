"""Seed procedure (Phase 2A leaf): VerifyNonNegativeEdges.

Given an instance description, check whether the described graph
contains any edges with negative weight. Returns the list of violating
edges as state. No sub-procedures.

Designed to be called from a composer (e.g., VerifyShortestPath) via
the structured `CALL VerifyNonNegativeEdges WITH instance=...` command.
"""
from __future__ import annotations

from reasoning.schemas import ProcedureNode, Provenance


def build_verify_nonneg_edges() -> ProcedureNode:
    """Return the seed ProcedureNode for VerifyNonNegativeEdges."""
    return ProcedureNode(
        id="proc_verify_nonneg_edges_v1",
        name="VerifyNonNegativeEdges",
        purpose=(
            "Check whether a described graph contains any edges with "
            "negative weight."
        ),
        when_to_use=(
            "Use when the question or the parent procedure needs a yes/no "
            "answer about negative-edge presence in a specific graph instance. "
            "Typically called as a sub-procedure of a shortest-path verifier. "
            "Skip for purely conceptual questions about negative edges."
        ),
        signature={
            "inputs": [
                {"name": "instance_description", "type": "str"},
            ],
            "outputs": [
                {"name": "has_negative_edge", "type": "bool"},
                {"name": "violating_edges", "type": "list[str]"},
            ],
        },
        state_schema={
            "violating_edges": "list[str]",
            "checked_edges": "list[str]",
        },
        body=(
            "Inspect the instance description for any edge with negative weight.\n"
            "For each edge mentioned in the instance:\n"
            "  - If you can determine its weight, add it to checked_edges\n"
            "  - If the weight is negative, ALSO add the edge to violating_edges\n\n"
            "Use exactly these mutation commands:\n"
            "  ADD <edge_label> TO state.checked_edges\n"
            "  ADD <edge_label> TO state.violating_edges\n"
            "  DONE\n"
            "After DONE, give a one-sentence verdict."
        ),
        example_use={
            "session_id": "<seed>",
            "inputs": {
                "instance_description": (
                    "Directed graph with edges (a->b, weight 3), "
                    "(b->c, weight -1), (a->c, weight 5)."
                ),
            },
            "final_state": {
                "violating_edges": ["b->c"],
                "checked_edges": ["a->b", "b->c", "a->c"],
            },
            "final_output": {
                "has_negative_edge": True,
                "violating_edges": ["b->c"],
            },
        },
        provenance=Provenance(
            created_in_session_id="<seed>",
            validating_examples=["<seed>"],
            depends_on=[],
            citation_count=0,
        ),
    )
