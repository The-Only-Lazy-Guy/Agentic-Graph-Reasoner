"""Script: invoke VerifyShortestPath with stubbed LLM and print full result."""
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


# ---- parameters --------------------------------------------------------- #
ALGORITHM = "Dijkstra"
INSTANCE = "directed graph with edges (a->b, weight 3), (b->c, weight -1), (a->c, weight 5)"


# ---- build procedure pool ----------------------------------------------- #
procs = [
    build_seed_procedure(),
    build_verify_nonneg_edges(),
    build_detect_negative_cycle(),
    build_verify_shortest_path(),
]
disp = Dispatcher({p.id: p for p in procs})


# ---- scripted LLM stub -------------------------------------------------- #
def stub_llm(prompt: str) -> str:
    # Composer body for VerifyShortestPath
    if "You are executing the VerifyShortestPath procedure." in prompt:
        return (
            "CALL VerifyAlgorithmPreconditions WITH algorithm_name=Dijkstra "
            "instance_description=directed graph with edges (a->b, weight 3), "
            "(b->c, weight -1), (a->c, weight 5)\n"
            "CALL VerifyNonNegativeEdges WITH instance_description=directed graph "
            "with edges (a->b, weight 3), (b->c, weight -1), (a->c, weight 5)\n"
            "SET state.safe_to_apply = false\n"
            "SET state.verdict = \"Dijkstra is unsafe on this graph because Dijkstra requires non-negative edge weights.\"\n"
            "SET state.recommended_alternative = \"Bellman-Ford\"\n"
            "DONE"
        )

    # VerifyAlgorithmPreconditions leaf body
    if "You are executing the VerifyAlgorithmPreconditions procedure." in prompt:
        return (
            "ADD nonneg_edges TO state.preconditions_checked\n"
            "ADD directed_graph TO state.preconditions_checked\n"
            "ADD directed_graph TO state.preconditions_satisfied\n"
            "ADD nonneg_edges TO state.preconditions_violated\n"
            "DONE\n"
            "directed graph structure is compatible with Dijkstra"
        )

    # VerifyNonNegativeEdges leaf body
    if "You are executing the VerifyNonNegativeEdges procedure." in prompt:
        return (
            "ADD (a->b, weight 3) TO state.checked_edges\n"
            "ADD (b->c, weight -1) TO state.checked_edges\n"
            "ADD (a->c, weight 5) TO state.checked_edges\n"
            "ADD (b->c, weight -1) TO state.violating_edges\n"
            "DONE\n"
            "edge b->c has negative weight -1"
        )

    return "DONE"


# ---- create session and invoke ----------------------------------------- #
session_id = f"sess_{uuid.uuid4().hex[:12]}"
ctrl = SessionSubgraphController(session_id, INSTANCE, "cs4")
budget = BudgetTracker(Budgets(max_llm_calls=10, max_composition_fan_out=5))
budget.on_step_change(0)

match = PatternMatch(
    verb="apply_intent",
    procedure_name="VerifyShortestPath",
    args_text=f"{ALGORITHM} on {INSTANCE}",
    start=0,
    end=10,
)

outcome = disp.invoke(match, ctrl, stub_llm, budget=budget)

# ---- print full result ------------------------------------------------- #
print("=" * 60)
print("FULL RESULT: VerifyShortestPath invocation")
print("=" * 60)

print(f"\n--- Top-level outcome ({outcome.match.procedure_name}) ---")
print(f"  object_id:         {outcome.object_id}")
print(f"  procedure_id:      {outcome.procedure_id}")
print(f"  mutations_applied: {outcome.mutations_applied}")
print(f"  parent_object_id:  {outcome.parent_object_id}")
print(f"  error:             {outcome.error}")

print(f"\n  Sub-LLM prompt (first 300 chars):")
print(f"    {outcome.sub_prompt[:300]}...")

print(f"\n  Sub-LLM response:")
for line in outcome.sub_response.strip().split("\n"):
    print(f"    {line}")

print(f"\n--- Composer's session state ---")
obj = ctrl.subgraph.nodes.get(outcome.object_id)
if obj:
    print(json.dumps(obj["state"], indent=4))

print(f"\n--- Sub-outcomes ({len(outcome.sub_outcomes)} children) ---")
for child in outcome.sub_outcomes:
    print(f"\n  Child: {child.match.procedure_name}")
    print(f"    object_id:         {child.object_id}")
    print(f"    mutations_applied: {child.mutations_applied}")
    print(f"    parent_object_id:  {child.parent_object_id}")
    print(f"    error:             {child.error}")

    child_obj = ctrl.subgraph.nodes.get(child.object_id)
    if child_obj:
        print(f"    state: {json.dumps(child_obj['state'], indent=6)}")

    print(f"    response (summary):")
    for line in child.sub_response.strip().split("\n"):
        print(f"      {line}")

print(f"\n--- Sub_invocation_of edges ---")
from reasoning.composition import SUB_INVOCATION_OF
sub_edges = [e for e in ctrl.subgraph.edges if e.relation == SUB_INVOCATION_OF]
for e in sub_edges:
    print(f"  {e.src} -> {e.dst}  relation={e.relation}")
    if e.metadata:
        print(f"    metadata: {json.dumps(e.metadata)}")

print(f"\n--- All session objects ---")
for nid, node in ctrl.subgraph.nodes.items():
    if node.get("node_type") == "session_object":
        print(f"  {nid}: name={node['name']}, "
              f"state_keys={list(node['state'].keys())}")

print(f"\n--- Budget usage ---")
print(json.dumps(budget.summary(), indent=2))
print("\nDone.")
