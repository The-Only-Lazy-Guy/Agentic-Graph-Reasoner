"""Seed procedure: VerifyAlgorithmPreconditions.

The first procedure shipped with the substrate. Exercised end-to-end by
Sub-phase 1.9 integration test. Designed to be applicable to the Dijkstra
and Bellman-Ford questions the front-end already produces good answers on,
so we can A/B-compare substrate vs legacy on real questions.

This procedure is meant to be loaded into a graph on first run, then
treated like any other procedure node — retrievable, citable, eventually
promotable to long-term memory.
"""
from __future__ import annotations

from reasoning.schemas import ProcedureNode, Provenance


def build_seed_procedure() -> ProcedureNode:
    """Return the seed ProcedureNode. Construct lazily so test runs and
    real loads share one definition without import-time side effects."""
    return ProcedureNode(
        id="proc_verify_algorithm_preconditions_v1",
        name="VerifyAlgorithmPreconditions",
        purpose=(
            "Check whether a named algorithm's stated preconditions hold for a "
            "given problem instance."
        ),
        when_to_use=(
            "Use when the question asks whether a specific algorithm can be "
            "applied to a given graph, dataset, or input, OR when the user is "
            "debugging a suspected algorithm misuse. Skip if the question is "
            "purely about algorithm description or comparison without a concrete "
            "instance."
        ),
        signature={
            "inputs": [
                {"name": "algorithm_name", "type": "str"},
                {"name": "instance_description", "type": "str"},
            ],
            "outputs": [
                {"name": "preconditions_satisfied", "type": "bool"},
                {"name": "violated_preconditions", "type": "list[str]"},
                {"name": "deferred_preconditions", "type": "list[str]"},
                {"name": "recommended_alternative", "type": "str | None"},
            ],
        },
        state_schema={
            "preconditions_checked": "list[str]",
            "preconditions_violated": "list[str]",
            "preconditions_deferred": "list[str]",
            "evidence_for_violations": "dict[str, str]",
        },
        body=(
            "You are verifying the preconditions of the named algorithm against "
            "a concrete problem instance.\n\n"
            "Steps:\n"
            "  1. Identify the algorithm's stated preconditions (typically 2-4).\n"
            "  2. For each precondition, decide one of: SATISFIED / VIOLATED / DEFERRED.\n"
            "     DEFERRED means the instance does not provide enough information.\n"
            "  3. For each VIOLATED precondition, cite the specific evidence from the\n"
            "     instance that proves the violation.\n"
            "  4. If any precondition is violated, suggest the recommended\n"
            "     alternative algorithm from background facts.\n\n"
            "Use exactly these mutation commands to update state:\n"
            "  ADD <precondition_name> TO state.preconditions_checked\n"
            "  ADD <precondition_name> TO state.preconditions_violated\n"
            "  ADD <precondition_name> TO state.preconditions_deferred\n"
            "  SET state.evidence_for_violations.<precondition_name> = \"<evidence text>\"\n"
            "  DONE\n"
            "After DONE, give a 1-2 sentence summary in plain prose."
        ),
        example_use={
            "session_id": "<seed>",
            "inputs": {
                "algorithm_name": "Dijkstra",
                "instance_description": (
                    "Directed graph with 5 vertices; edges include "
                    "(a->b, weight 3), (b->c, weight -1), (a->c, weight 5)."
                ),
            },
            "final_state": {
                "preconditions_checked": [
                    "all edges nonnegative",
                    "single source defined",
                ],
                "preconditions_violated": ["all edges nonnegative"],
                "preconditions_deferred": [],
                "evidence_for_violations": {
                    "all edges nonnegative": "Edge b->c has weight -1.",
                },
            },
            "final_output": {
                "preconditions_satisfied": False,
                "violated_preconditions": ["all edges nonnegative"],
                "deferred_preconditions": [],
                "recommended_alternative": "Bellman-Ford",
            },
        },
        provenance=Provenance(
            created_in_session_id="<seed>",
            validating_examples=["<seed>"],
            depends_on=[],
            citation_count=0,
        ),
    )
