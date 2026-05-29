"""Synthetic Phase 3D adaptive-planning drivers.

These examples are deliberately deterministic. They model the exact plan-tree
behavior we want before any prompt-mode integration is allowed to steer it.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict

from reasoning.activation import FrameItem, GraphTaskFrame
from reasoning.adaptive_planning import (
    AdaptivePlanTree,
    PlanCheckResult,
    attach_plan_tree_to_session,
)
from reasoning.session_subgraph import SessionSubgraphController


def build_ioi_synthetic_plan_tree() -> AdaptivePlanTree:
    """Kadane fails online updates, then a segment-tree sibling succeeds."""
    tree = AdaptivePlanTree(
        "sess_phase3d_ioi_synthetic",
        "solve dynamic maximum subarray with point updates",
        root_hypothesis="Need C++17, online updates, non-empty subarray, negative values, 64-bit sums.",
        root_node_id="plan_ioi_root",
    )
    choose = tree.add_child(
        tree.state.root_node_id,
        goal="choose algorithm",
        hypothesis="Compare Kadane direct recomputation against a segment tree aggregate.",
        mode="plan",
        checkpoint_quality=0.86,
        evidence_ids=["fi_ioi_updates", "fi_ioi_segment_tree"],
        node_id="plan_ioi_choose_algorithm",
    )
    kadane = tree.add_child(
        choose,
        goal="try Kadane direct",
        hypothesis="Kadane computes maximum subarray after each update.",
        mode="execute",
        checkpoint_quality=0.35,
        node_id="plan_ioi_kadane_direct",
    )
    fail_kadane = PlanCheckResult(
        checked_node_id=kadane,
        passed=False,
        failure_scope="algorithm_choice",
        failed_requirements=["q online point updates"],
        reason="Kadane recomputation is O(n) per update and fails q up to 200000.",
        check_id="plan_check_ioi_kadane_fails_updates",
    )
    segment = tree.revise_from_failure(
        kadane,
        fail_kadane,
        new_goal="try segment tree aggregate",
        new_hypothesis="Store sum, max_prefix, max_suffix, and best/max_sub in every node.",
        mode="execute",
        checkpoint_quality=0.88,
        evidence_ids=["fi_ioi_updates", "fi_ioi_segment_tree", "fi_ioi_long_long"],
        node_id="plan_ioi_segment_tree",
    )
    merge = tree.add_child(
        segment,
        goal="derive merge rule",
        hypothesis="best is max(left.best, right.best, left.suffix + right.prefix).",
        mode="execute",
        checkpoint_quality=0.82,
        evidence_ids=["fi_ioi_segment_tree"],
        node_id="plan_ioi_merge_rule",
    )
    pass_merge = PlanCheckResult(
        checked_node_id=merge,
        passed=True,
        failure_scope="local_step",
        reason="Merge rule covers left, right, and cross-boundary subarrays.",
        check_id="plan_check_ioi_merge_passes",
    )
    tree.record_check(pass_merge)
    tree.mark_passed(merge)
    final = tree.add_child(
        segment,
        goal="final answer",
        hypothesis="Answer with segment tree C++17 solution and edge-case notes.",
        mode="finalize",
        checkpoint_quality=0.84,
        evidence_ids=["fi_ioi_segment_tree", "fi_ioi_long_long", "fi_ioi_all_negative"],
        node_id="plan_ioi_final_answer",
    )
    frame = GraphTaskFrame(
        session_id=tree.state.session_id,
        constraints=[
            FrameItem(
                "fi_ioi_segment_tree",
                "answer_requirement",
                "Use a segment tree node with sum, max_prefix, max_suffix, and max_sub/best.",
                95,
                [],
            ),
            FrameItem(
                "fi_ioi_long_long",
                "constraint",
                "Use long long/int64 for sums and segment aggregate fields.",
                95,
                [],
            ),
            FrameItem(
                "fi_ioi_all_negative",
                "pitfall",
                "For non-empty subarrays, all-negative arrays must return the maximum element, not 0.",
                95,
                [],
            ),
        ],
    )
    answer = (
        "Use a segment tree. Each node stores long long sum, prefix, suffix, "
        "and best maximum subarray. Merge with left.suffix + right.prefix. "
        "Leaves use the element value so non-empty all-negative arrays return "
        "the maximum element, not 0."
    )
    final_check = tree.try_finalize(final, frame, answer)
    if final_check.passed:
        tree.mark_passed(segment)
        tree.mark_passed(choose)
        tree.mark_passed(tree.state.root_node_id)
    return tree


def build_dijkstra_synthetic_plan_tree() -> AdaptivePlanTree:
    """Dijkstra fails a negative-edge check, then Bellman-Ford succeeds."""
    tree = AdaptivePlanTree(
        "sess_phase3d_dijkstra_synthetic",
        "decide shortest path algorithm with a negative edge",
        root_hypothesis="Need decide whether Dijkstra's nonnegative-edge precondition holds.",
        root_node_id="plan_dijkstra_root",
    )
    choose = tree.add_child(
        tree.state.root_node_id,
        goal="choose shortest-path algorithm",
        hypothesis="Use Dijkstra only if edges are nonnegative; otherwise use Bellman-Ford.",
        mode="plan",
        checkpoint_quality=0.9,
        evidence_ids=["fi_dijkstra_negative_edge"],
        node_id="plan_dijkstra_choose_algorithm",
    )
    dijkstra = tree.add_child(
        choose,
        goal="try Dijkstra",
        hypothesis="Run Dijkstra from the source.",
        mode="execute",
        checkpoint_quality=0.4,
        node_id="plan_dijkstra_try_dijkstra",
    )
    fail_dijkstra = PlanCheckResult(
        checked_node_id=dijkstra,
        passed=False,
        failure_scope="algorithm_choice",
        failed_requirements=["negative edge"],
        reason="A negative edge violates Dijkstra's nonnegative-weight precondition.",
        check_id="plan_check_dijkstra_fails_negative_edge",
    )
    bellman = tree.revise_from_failure(
        dijkstra,
        fail_dijkstra,
        new_goal="try Bellman-Ford",
        new_hypothesis="Bellman-Ford supports negative edges if no negative cycle is reachable.",
        mode="execute",
        checkpoint_quality=0.86,
        evidence_ids=["fi_dijkstra_negative_edge"],
        node_id="plan_dijkstra_bellman_ford",
    )
    final = tree.add_child(
        bellman,
        goal="final answer",
        hypothesis="Explain Dijkstra is unsafe and select Bellman-Ford.",
        mode="finalize",
        checkpoint_quality=0.84,
        evidence_ids=["fi_dijkstra_negative_edge"],
        node_id="plan_dijkstra_final_answer",
    )
    frame = GraphTaskFrame(
        session_id=tree.state.session_id,
        pitfalls=[
            FrameItem(
                "fi_dijkstra_negative_edge",
                "pitfall",
                "Verify non-negative edges before trusting Dijkstra; use Bellman-Ford when violated.",
                95,
                [],
            ),
        ],
    )
    answer = (
        "Dijkstra is unsafe with a negative edge because its non-negative edge "
        "precondition is violated. Use Bellman-Ford, and separately check for "
        "a reachable negative cycle if needed."
    )
    final_check = tree.try_finalize(final, frame, answer)
    if final_check.passed:
        tree.mark_passed(bellman)
        tree.mark_passed(choose)
        tree.mark_passed(tree.state.root_node_id)
    return tree


def persist_synthetic_plan_sessions(root: Path = Path("data/session_subgraphs")) -> Dict[str, Path]:
    """Persist the deterministic IOI and Dijkstra plan trees for inspection."""
    out: Dict[str, Path] = {}
    cases = {
        "ioi": (
            build_ioi_synthetic_plan_tree(),
            "Solve dynamic maximum subarray with online point updates.",
        ),
        "dijkstra": (
            build_dijkstra_synthetic_plan_tree(),
            "Can Dijkstra handle a graph with one negative edge?",
        ),
    }
    for name, (tree, query) in cases.items():
        session = SessionSubgraphController(tree.state.session_id, query, "merged_graph")
        attach_plan_tree_to_session(session, tree)
        out[name] = session.close(root)
    return out
