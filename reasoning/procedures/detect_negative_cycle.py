"""Seed procedure (Phase 2A leaf): DetectNegativeCycle.

Given an instance description, check whether the described graph
contains any reachable negative cycle. Returns the cycle descriptions
if found. No sub-procedures.

Designed to be called from a composer (e.g., VerifyShortestPath) via
the structured `CALL DetectNegativeCycle WITH instance=...` command.
"""
from __future__ import annotations

from reasoning.schemas import ProcedureNode, Provenance


def build_detect_negative_cycle() -> ProcedureNode:
    """Return the seed ProcedureNode for DetectNegativeCycle."""
    return ProcedureNode(
        id="proc_detect_negative_cycle_v1",
        name="DetectNegativeCycle",
        purpose=(
            "Detect any reachable negative cycle in a described graph."
        ),
        when_to_use=(
            "Use when a graph contains at least one negative edge and the "
            "shortest-path algorithm under consideration (e.g., Bellman-Ford) "
            "is only correct in the absence of negative cycles. Typically "
            "called as a sub-procedure of a shortest-path verifier."
        ),
        signature={
            "inputs": [
                {"name": "instance_description", "type": "str"},
            ],
            "outputs": [
                {"name": "has_negative_cycle", "type": "bool"},
                {"name": "detected_cycles", "type": "list[str]"},
            ],
        },
        state_schema={
            "detected_cycles": "list[str]",
            "checked_paths": "list[str]",
        },
        body=(
            "Inspect the instance description for any cycle whose total edge "
            "weight is negative.\n\n"
            "Steps:\n"
            "  1. Identify the closed paths (cycles) implied by the edges.\n"
            "  2. For each cycle, sum the edge weights along it.\n"
            "  3. If the sum is negative, the cycle is a negative cycle.\n\n"
            "Use exactly these mutation commands:\n"
            "  ADD <cycle_description> TO state.checked_paths\n"
            "  ADD <cycle_description> TO state.detected_cycles\n"
            "  DONE\n"
            "After DONE, briefly state whether a negative cycle was found."
        ),
        example_use={
            "session_id": "<seed>",
            "inputs": {
                "instance_description": (
                    "Directed graph with edges (a->b, weight 1), "
                    "(b->c, weight -3), (c->a, weight 1). The cycle "
                    "a->b->c->a has total weight 1+(-3)+1 = -1."
                ),
            },
            "final_state": {
                "detected_cycles": ["a->b->c->a (sum -1)"],
                "checked_paths": ["a->b->c->a (sum -1)"],
            },
            "final_output": {
                "has_negative_cycle": True,
                "detected_cycles": ["a->b->c->a (sum -1)"],
            },
        },
        provenance=Provenance(
            created_in_session_id="<seed>",
            validating_examples=["<seed>"],
            depends_on=[],
            citation_count=0,
        ),
    )
