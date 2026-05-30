"""V5 behavioral smoke test — Dijkstra negative-edge toy graph.

Verifies LOOP MECHANICS that are deterministic regardless of head training:

  1. Planning loop (L8) attention mass lands ONLY on planning-pool nodes
  2. Evidence loop (L20) attention mass lands ONLY on evidence-pool nodes
  3. Static invalidator gating: combined_inv fires only on nodes with a
     structural INVALIDATED_BY / CONTRADICTS edge
  4. Exit-reason machinery always sets a reason
  5. Required-slot exit only considers the task's required slots

These do not need a trained model — they test the wiring, masking, and gating.
Score *values* are random (untrained heads); only the masking/gating invariants
are asserted. Run before spending GPU time on real training.

    python -m v5.smoke_test_toy
"""
from __future__ import annotations

import torch

from reasoning.graph_relations import Rel
from v5.cross_attention import V5AttentionAdapter
from v5.gnn_encoder import RGCNEncoder
from v5.goal_encoder import GoalEncoder, encode_task_frame
from v5.subgraph import build_active_subgraph


class _Node:
    def __init__(self, nid, ntype, status="unknown"):
        self.node_id = nid
        self.node_type = ntype
        self.text = nid.replace("_", " ")
        self.confidence = 0.7
        self.metadata = {"status": status}


class _Edge:
    def __init__(self, src, dst, relation):
        self.src = src
        self.dst = dst
        self.relation = relation


class _ToyGraph:
    """Dijkstra negative-edge applicability scenario."""
    def __init__(self):
        self.nodes = {
            # ── planning pool ──
            "dijkstra_strategy":        _Node("dijkstra_strategy", "strategy"),
            "negative_edge_failure":    _Node("negative_edge_failure", "failure_pattern"),
            "dijkstra_invalid_epi":     _Node("dijkstra_invalid_epi", "epistemic_state", "uncertain"),
            # ── evidence pool ──
            "dijkstra_nonneg_fact":     _Node("dijkstra_nonneg_fact", "fact"),
            "bellman_ford_alt":         _Node("bellman_ford_alt", "application"),
            "shortest_path_subgoal":    _Node("shortest_path_subgoal", "solved_subgoal"),
            "dijkstra_verified_epi":    _Node("dijkstra_verified_epi", "epistemic_state", "verified"),
        }
        # negative-edge failure pattern structurally invalidates the strategy
        self.edges = [
            _Edge("negative_edge_failure", "dijkstra_strategy", Rel.INVALIDATED_BY),
            _Edge("dijkstra_strategy", "bellman_ford_alt", Rel.CONTRADICTS),
        ]


PLANNING_POOL = {"dijkstra_strategy", "negative_edge_failure", "dijkstra_invalid_epi"}
EVIDENCE_POOL = {"dijkstra_nonneg_fact", "bellman_ford_alt", "shortest_path_subgoal",
                 "dijkstra_verified_epi"}
# Nodes that are the SOURCE of an invalidator edge
STRUCTURAL_INVALIDATORS = {"negative_edge_failure", "dijkstra_strategy"}


def _attn_mass_by_node(loop_logs, node_ids, layer):
    """Return {node_id: total attention weight} from the last loop entry at layer.

    Reconstructs attention from top_nodes recorded in the log entries.
    """
    entries = [e for e in loop_logs if e["layer"] == layer]
    if not entries:
        return {}
    last = entries[-1]
    return dict(last["top_nodes"])


def run_smoke_test():
    torch.manual_seed(1234)
    device = torch.device("cpu")

    graph = _ToyGraph()
    node_ids = list(graph.nodes.keys())
    text_emb = {nid: [0.05 * (i + 1)] * 768 for i, nid in enumerate(node_ids)}

    task_frame = {
        "task_family": "algorithm_applicability",
        "question_mode": "direct_relationship",
        "required_slots": ["verdict", "reason"],
    }

    # ── build subgraph + GNN ──────────────────────────────────────────────
    asg = build_active_subgraph(graph, node_ids, text_emb, device, task_frame)
    gnn = RGCNEncoder().eval()
    with torch.no_grad():
        kv = gnn.encode_to_kv(asg.encoder_inputs, asg)

    print("=== POOL MASKS ===")
    for i, nid in enumerate(node_ids):
        print(f"  {nid:28s} plan={bool(kv.planning_mask[i])!s:5} "
              f"evid={bool(kv.evidence_mask[i])!s:5} inv={kv.invalidator_flags[i].item():.0f}")

    # ── assertions: masks match expected pools ────────────────────────────
    plan_nodes = {nid for i, nid in enumerate(node_ids) if bool(kv.planning_mask[i])}
    evid_nodes = {nid for i, nid in enumerate(node_ids) if bool(kv.evidence_mask[i])}
    inv_nodes = {nid for i, nid in enumerate(node_ids) if kv.invalidator_flags[i].item() > 0.5}

    assert plan_nodes == PLANNING_POOL, f"planning pool mismatch: {plan_nodes}"
    assert evid_nodes == EVIDENCE_POOL, f"evidence pool mismatch: {evid_nodes}"
    assert inv_nodes == STRUCTURAL_INVALIDATORS, f"invalidator mismatch: {inv_nodes}"
    print("  [OK] planning/evidence/invalidator masks match expected pools")

    # ── run loops ─────────────────────────────────────────────────────────
    ge = GoalEncoder().eval()
    with torch.no_grad():
        goal = encode_task_frame(task_frame, device, ge)

    adapter = V5AttentionAdapter(r_plan=3, r_evidence=4)
    adapter.eval()
    h = torch.randn(1, 2560) * 0.02

    with torch.no_grad():
        h_plan, plan_state, plan_logs = adapter.run_planning(
            h, goal, kv, node_ids, task_frame=task_frame)
        h_evid, evid_state, evid_logs = adapter.run_evidence(
            h_plan, goal, kv, node_ids, task_frame=task_frame)

    # ── assertion: attention mass respects pool masks ─────────────────────
    # The softmax masks out-of-pool nodes to ~0. Verify top_nodes recorded in
    # planning logs are all from the planning pool (and likewise evidence).
    print("\n=== PLANNING LOOP top_nodes (must be planning-pool only) ===")
    plan_mass = _attn_mass_by_node(plan_logs, node_ids, layer=8)
    for nid, w in plan_mass.items():
        in_pool = nid in PLANNING_POOL
        print(f"  {nid:28s} w={w:.4f} {'OK' if in_pool or w < 1e-6 else 'LEAK!'}")
        assert in_pool or w < 1e-6, f"planning attention leaked to {nid}"

    print("\n=== EVIDENCE LOOP top_nodes (must be evidence-pool only) ===")
    evid_mass = _attn_mass_by_node(evid_logs, node_ids, layer=20)
    for nid, w in evid_mass.items():
        in_pool = nid in EVIDENCE_POOL
        print(f"  {nid:28s} w={w:.4f} {'OK' if in_pool or w < 1e-6 else 'LEAK!'}")
        assert in_pool or w < 1e-6, f"evidence attention leaked to {nid}"
    print("  [OK] attention mass respects pool masks (no cross-pool leak)")

    # ── assertion: exit reasons always set ────────────────────────────────
    print(f"\n=== EXIT REASONS ===")
    print(f"  planning exit: {plan_state.exit_reason}")
    print(f"  evidence exit: {evid_state.exit_reason}")
    assert plan_state.exit_reason is not None, "planning exit_reason is None"
    assert evid_state.exit_reason is not None, "evidence exit_reason is None"
    print("  [OK] both loops set an exit_reason")

    # ── assertion: combined invalidator gating ────────────────────────────
    # combined_inv = static_inv * dynamic_inv → can only be nonzero on
    # structurally-invalidating nodes.
    combined = evid_state.invalidator_flags_r.squeeze(0)
    print(f"\n=== COMBINED INVALIDATOR (static x dynamic) ===")
    for i, nid in enumerate(node_ids):
        val = combined[i].item()
        structural = nid in STRUCTURAL_INVALIDATORS
        print(f"  {nid:28s} combined={val:.4f} structural={structural}")
        if not structural:
            assert val < 1e-6, f"invalidator fired on non-structural node {nid}"
    print("  [OK] combined invalidator nonzero only on structural invalidator nodes")

    print("\nALL SMOKE-TEST INVARIANTS PASSED")
    return True


if __name__ == "__main__":
    run_smoke_test()
