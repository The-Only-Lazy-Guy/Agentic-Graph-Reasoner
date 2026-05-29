"""Seed procedure (Phase 2A composer): VerifyShortestPath.

Composes Phase-1 and Phase-2A leaves to produce a complete verdict on
whether a named shortest-path algorithm can be safely applied to a
described problem instance.

Body uses the structured CALL grammar to invoke:
  - VerifyAlgorithmPreconditions  (Phase-1: generic precondition check)
  - VerifyNonNegativeEdges        (Phase-2A: leaf for Dijkstra path)
  - DetectNegativeCycle           (Phase-2A: leaf for Bellman-Ford path)

Children's mutations are independent. The composer's own state
captures only the aggregated verdict; per-child detail lives in each
child's session_object.
"""
from __future__ import annotations

from reasoning.schemas import ProcedureNode, Provenance


def build_verify_shortest_path() -> ProcedureNode:
    """Return the seed ProcedureNode for VerifyShortestPath."""
    return ProcedureNode(
        id="proc_verify_shortest_path_v1",
        name="VerifyShortestPath",
        purpose=(
            "Verify whether a named shortest-path algorithm "
            "(Dijkstra, Bellman-Ford, etc.) can be safely applied to a "
            "given graph instance by composing sub-checks."
        ),
        when_to_use=(
            "Use when the question is 'can algorithm X solve this "
            "shortest-path problem?' or 'is the user's algorithm choice "
            "safe on this graph?'. Prefer this over the lower-level "
            "VerifyAlgorithmPreconditions when the topic is specifically "
            "shortest-path applicability."
        ),
        signature={
            "inputs": [
                {"name": "algorithm_name", "type": "str"},
                {"name": "instance_description", "type": "str"},
            ],
            "outputs": [
                {"name": "verdict", "type": "str"},
                {"name": "safe_to_apply", "type": "bool"},
                {"name": "recommended_alternative", "type": "str | None"},
            ],
        },
        state_schema={
            "verdict": "str",
            "safe_to_apply": "bool",
            "recommended_alternative": "str",
            "sub_results_summary": "list[str]",
        },
        body=(
            "You are the VerifyShortestPath composer. Your ONLY job is to:\n"
            "  (a) dispatch the appropriate sub-procedures via CALL commands, and\n"
            "  (b) write a final verdict based on what those sub-procedures will find.\n"
            "You are NOT allowed to synthesize the sub-procedures' results in your\n"
            "own state. The system runs the sub-procedures separately; their\n"
            "results live in their own session objects (which the UI renders).\n\n"
            "OUTPUT FORMAT (emit exactly in this order, one command per line):\n\n"
            "Step 1 — Emit a CALL for the generic precondition check. ALWAYS required:\n"
            "    CALL VerifyAlgorithmPreconditions WITH algorithm_name={algorithm_name} instance_description={instance_description}\n\n"
            "Step 2 — Emit ONE algorithm-specific CALL:\n"
            "  If {algorithm_name} is Dijkstra (or a variant):\n"
            "    CALL VerifyNonNegativeEdges WITH instance_description={instance_description}\n"
            "  If {algorithm_name} is Bellman-Ford (or any algorithm that allows\n"
            "  negative edges but forbids negative cycles):\n"
            "    CALL DetectNegativeCycle WITH instance_description={instance_description}\n"
            "  For any other algorithm: skip this step (no second CALL).\n\n"
            "Step 3 — Emit your verdict (reason about what the sub-procedures WILL\n"
            "find given the algorithm's preconditions; you do NOT need to wait\n"
            "for their results):\n"
            "    SET state.safe_to_apply = <true|false as a JSON boolean>\n"
            "    SET state.verdict = \"<one-line summary>\"\n"
            "    SET state.recommended_alternative = \"<alternative algorithm name, or empty string>\"\n"
            "    DONE\n\n"
            "ABSOLUTE CONSTRAINTS:\n"
            "  1. You MUST emit at least ONE `CALL` command. A response without any\n"
            "     CALL line is invalid — the sub-procedures must be dispatched.\n"
            "  2. DO NOT write to state.sub_results_summary. Leave it empty. The\n"
            "     children populate their own state; the UI shows their findings.\n"
            "  3. DO NOT free-text the sub-procedures' results in your verdict.\n"
            "     Your verdict should reason from the algorithm's known\n"
            "     preconditions, not from claimed sub-procedure outputs.\n"
            "  4. After DONE you may add one short prose sentence (≤25 words).\n"
            "     Anything longer or mentioning the sub-procedures by name in\n"
            "     prose is forbidden."
        ),
        example_use={
            "session_id": "<seed>",
            "inputs": {
                "algorithm_name": "Dijkstra",
                "instance_description": (
                    "Directed graph with edges "
                    "(a->b, weight 3), (b->c, weight -1), (a->c, weight 5)."
                ),
            },
            "expected_dispatch": [
                # The composer's body MUST emit these CALL commands. The
                # dispatcher will fire them as sub-invocations under the
                # composer's session_object.
                "CALL VerifyAlgorithmPreconditions WITH algorithm_name=Dijkstra "
                "instance_description=Directed graph with edges (a->b, weight 3), "
                "(b->c, weight -1), (a->c, weight 5).",
                "CALL VerifyNonNegativeEdges WITH instance_description=Directed "
                "graph with edges (a->b, weight 3), (b->c, weight -1), "
                "(a->c, weight 5).",
            ],
            "final_state": {
                # sub_results_summary INTENTIONALLY empty — the children
                # populate their own session_object states; this composer
                # is not allowed to synthesize their results here.
                "sub_results_summary": [],
                "verdict": "Dijkstra is unsafe on this graph because Dijkstra requires non-negative edge weights.",
                "safe_to_apply": False,
                "recommended_alternative": "Bellman-Ford",
            },
            "final_output": {
                "verdict": "Dijkstra is unsafe on this graph because Dijkstra requires non-negative edge weights.",
                "safe_to_apply": False,
                "recommended_alternative": "Bellman-Ford",
            },
        },
        provenance=Provenance(
            created_in_session_id="<seed>",
            validating_examples=["<seed>"],
            depends_on=[
                "proc_verify_algorithm_preconditions_v1",
                "proc_verify_nonneg_edges_v1",
                "proc_detect_negative_cycle_v1",
            ],
            citation_count=0,
        ),
    )
