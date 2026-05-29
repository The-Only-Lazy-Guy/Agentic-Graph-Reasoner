"""Invoke VerifyShortestPath for both Dijkstra and Bellman-Ford on user's graph."""
from __future__ import annotations

import json
import uuid

from reasoning.budgets import BudgetTracker, Budgets
from reasoning.dispatcher import Dispatcher, PatternMatch
from reasoning.procedures.verify_algorithm_preconditions import build_seed_procedure
from reasoning.procedures.verify_nonneg_edges import build_verify_nonneg_edges
from reasoning.procedures.detect_negative_cycle import build_detect_negative_cycle
from reasoning.procedures.verify_shortest_path import build_verify_shortest_path
from reasoning.session_subgraph import SessionSubgraphController

INSTANCE = "directed graph with edges (A->B, weight 4), (B->C, weight -3), (C->D, weight 2), (D->B, weight 1), starting from source node A"

procs = [
    build_seed_procedure(),
    build_verify_nonneg_edges(),
    build_detect_negative_cycle(),
    build_verify_shortest_path(),
]
disp = Dispatcher({p.id: p for p in procs})


def make_stub_llm(algorithm_name: str):
    def stub_llm(prompt: str) -> str:
        if "You are executing the VerifyShortestPath procedure." in prompt:
            if algorithm_name == "Dijkstra":
                return (
                    f"CALL VerifyAlgorithmPreconditions WITH algorithm_name=Dijkstra "
                    f"instance_description={INSTANCE}\n"
                    f"CALL VerifyNonNegativeEdges WITH instance_description={INSTANCE}\n"
                    f"SET state.safe_to_apply = false\n"
                    f"SET state.verdict = \"Dijkstra is unsafe on this graph because Dijkstra requires non-negative edge weights.\"\n"
                    f"SET state.recommended_alternative = \"Bellman-Ford\"\n"
                    f"DONE"
                )
            else:
                return (
                    f"CALL VerifyAlgorithmPreconditions WITH algorithm_name=Bellman-Ford "
                    f"instance_description={INSTANCE}\n"
                    f"CALL DetectNegativeCycle WITH instance_description={INSTANCE}\n"
                    f"SET state.safe_to_apply = true\n"
                    f"SET state.verdict = \"Bellman-Ford is safe on this graph because it handles negative edges and no negative cycle is reachable from the source.\"\n"
                    f"SET state.recommended_alternative = \"\"\n"
                    f"DONE"
                )

        if "You are executing the VerifyAlgorithmPreconditions procedure." in prompt:
            if algorithm_name == "Dijkstra":
                return (
                    "ADD all_edges_nonnegative TO state.preconditions_checked\n"
                    "ADD directed_graph TO state.preconditions_checked\n"
                    "ADD single_source_defined TO state.preconditions_checked\n"
                    "ADD directed_graph TO state.preconditions_satisfied\n"
                    "ADD single_source_defined TO state.preconditions_satisfied\n"
                    "ADD all_edges_nonnegative TO state.preconditions_violated\n"
                    'SET state.evidence_for_violations.all_edges_nonnegative = "Edge B->C has weight -3, which is negative."\n'
                    "DONE\n"
                    "The directed graph structure and single source are fine, but edge B->C violates the nonnegative-weight precondition."
                )
            else:
                return (
                    "ADD handles_negative_edges TO state.preconditions_checked\n"
                    "ADD no_negative_cycle_reachable TO state.preconditions_checked\n"
                    "ADD directed_graph TO state.preconditions_checked\n"
                    "ADD single_source_defined TO state.preconditions_checked\n"
                    "ADD handles_negative_edges TO state.preconditions_satisfied\n"
                    "ADD directed_graph TO state.preconditions_satisfied\n"
                    "ADD single_source_defined TO state.preconditions_satisfied\n"
                    "DONE\n"
                    "Bellman-Ford can handle negative edges. The absence of negative cycles is a separate check."
                )

        if "You are executing the VerifyNonNegativeEdges procedure." in prompt:
            return (
                "ADD (A->B, weight 4) TO state.checked_edges\n"
                "ADD (B->C, weight -3) TO state.checked_edges\n"
                "ADD (C->D, weight 2) TO state.checked_edges\n"
                "ADD (D->B, weight 1) TO state.checked_edges\n"
                "ADD (B->C, weight -3) TO state.violating_edges\n"
                "DONE\n"
                "Edge B->C has negative weight -3."
            )

        if "You are executing the DetectNegativeCycle procedure." in prompt:
            return (
                "ADD B->C->D->B (sum: -3+2+1=0) TO state.checked_paths\n"
                "DONE\n"
                "The only reachable cycle B->C->D->B sums to 0, which is not negative. No negative cycle detected."
            )

        return "DONE"
    return stub_llm


def run_verify(name: str, algo: str):
    print(f"\n{'=' * 70}")
    print(f"  VerifyShortestPath: algorithm_name={algo}")
    print(f"{'=' * 70}")

    session_id = f"sess_{uuid.uuid4().hex[:12]}"
    ctrl = SessionSubgraphController(session_id, INSTANCE, "cs4")
    budget = BudgetTracker(Budgets(max_llm_calls=10, max_composition_fan_out=5))
    budget.on_step_change(0)

    stub_llm = make_stub_llm(algo)

    match = PatternMatch(
        verb="apply_intent",
        procedure_name="VerifyShortestPath",
        args_text=f"{algo} on {INSTANCE}",
        start=0,
        end=10,
    )

    outcome = disp.invoke(match, ctrl, stub_llm, budget=budget)

    print(f"\nTop-level outcome ({outcome.match.procedure_name})")
    print(f"  object_id:         {outcome.object_id}")
    print(f"  procedure_id:      {outcome.procedure_id}")
    print(f"  mutations_applied: {outcome.mutations_applied}")
    print(f"  error:             {outcome.error}")

    print(f"\n  Sub-LLM response:")
    for line in outcome.sub_response.strip().split("\n"):
        print(f"    {line}")

    print(f"\n  Composer session state:")
    obj = ctrl.subgraph.nodes.get(outcome.object_id)
    if obj:
        print(json.dumps(obj["state"], indent=4))

    print(f"\n  Sub-outcomes ({len(outcome.sub_outcomes)} children):")
    for child in outcome.sub_outcomes:
        print(f"\n    Child: {child.match.procedure_name}")
        print(f"      object_id:         {child.object_id}")
        print(f"      mutations_applied: {child.mutations_applied}")
        print(f"      error:             {child.error}")
        child_obj = ctrl.subgraph.nodes.get(child.object_id)
        if child_obj:
            print(f"      state: {json.dumps(child_obj['state'], indent=8)}")
        print(f"      response:")
        for line in child.sub_response.strip().split("\n"):
            print(f"        {line}")

    return outcome, ctrl


print("=" * 70)
print("  GRAPH: (A->B, 4), (B->C, -3), (C->D, 2), (D->B, 1), source=A")
print("=" * 70)

outcome_dijkstra, ctrl_dijkstra = run_verify("Dijkstra", "Dijkstra")
outcome_bf, ctrl_bf = run_verify("Bellman-Ford", "Bellman-Ford")

print(f"\n{'=' * 70}")
print("  FINAL RECOMMENDATION")
print(f"{'=' * 70}")
# Read safe_to_apply from session state
dijkstra_obj = ctrl_dijkstra.subgraph.nodes.get(outcome_dijkstra.object_id)
dijkstra_safe = dijkstra_obj["state"]["safe_to_apply"] if dijkstra_obj else "?"
bf_obj = ctrl_bf.subgraph.nodes.get(outcome_bf.object_id)
bf_safe = bf_obj["state"]["safe_to_apply"] if bf_obj else "?"
print(f"  Dijkstra:    {'NOT SAFE' if not dijkstra_safe else 'SAFE'} "
      f"(negative edge B->C)")
print(f"  Bellman-Ford:  {'SAFE' if bf_safe else 'NOT SAFE'} "
      f"(no negative cycle; cycle B->C->D->B sums to 0)")
print(f"\n  Recommendation: Use Bellman-Ford on this graph.")
print(f"  Dijkstra is unsafe because weight -3 on edge B->C violates Dijkstra's")
print(f"  non-negative edge weight requirement. Bellman-Ford is safe because")
print(f"  the only cycle B->C->D->B has total weight 0 (not negative).")
print()
