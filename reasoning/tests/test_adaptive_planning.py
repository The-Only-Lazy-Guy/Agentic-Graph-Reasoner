"""Tests for reasoning/adaptive_planning.py - Phase 3D."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from reasoning.activation import FrameItem, GraphTaskFrame
from reasoning.adaptive_planning import (
    AdaptivePlanTree,
    PlanCheckResult,
    PlanningBudgetExceeded,
    attach_plan_tree_to_session,
)
from reasoning.adaptive_planning_examples import (
    build_dijkstra_synthetic_plan_tree,
    build_ioi_synthetic_plan_tree,
    persist_synthetic_plan_sessions,
)
from reasoning.session_subgraph import SessionSubgraphController


class TestAdaptivePlanRoundTrip(unittest.TestCase):
    def test_plan_tree_round_trip(self):
        tree = AdaptivePlanTree("sess_plan", "solve dynamic max-subarray")
        choose = tree.add_child(
            tree.state.root_node_id,
            goal="choose algorithm",
            hypothesis="start by choosing between Kadane and segment tree",
            mode="plan",
            checkpoint_quality=0.82,
            evidence_ids=["fi_updates"],
        )
        leaf = tree.add_child(
            choose,
            goal="try Kadane",
            hypothesis="Kadane may compute max subarray directly",
            mode="execute",
            checkpoint_quality=0.45,
        )
        check = PlanCheckResult(
            checked_node_id=leaf,
            passed=False,
            failure_scope="algorithm_choice",
            failed_requirements=["online point updates"],
            reason="Kadane is O(n) per query.",
        )
        tree.record_check(check)

        restored = AdaptivePlanTree.from_dict(json.loads(json.dumps(tree.to_dict())))
        self.assertEqual(restored.state.session_id, "sess_plan")
        self.assertEqual(restored.nodes[choose].goal, "choose algorithm")
        self.assertEqual(restored.checks[check.check_id].failure_scope, "algorithm_choice")
        self.assertTrue(any(e.relation == "checked_by" for e in restored.edges))


class TestBacktrackingPolicy(unittest.TestCase):
    def test_local_step_failure_selects_parent(self):
        tree = AdaptivePlanTree("sess_local", "derive segment tree solution")
        choose = tree.add_child(
            tree.state.root_node_id,
            goal="choose data structure",
            hypothesis="segment tree is needed",
            mode="plan",
            checkpoint_quality=0.95,
        )
        merge = tree.add_child(
            choose,
            goal="derive merge rule",
            hypothesis="combine left/right prefix/suffix/best fields",
            mode="execute",
            checkpoint_quality=0.6,
        )
        buggy = tree.add_child(
            merge,
            goal="write merge formula",
            hypothesis="best = max(left.best, right.best)",
            mode="execute",
            checkpoint_quality=0.4,
        )
        check = PlanCheckResult(
            checked_node_id=buggy,
            passed=False,
            failure_scope="local_step",
            failed_requirements=["cross-boundary subarray"],
            reason="Merge rule missed left.suffix + right.prefix.",
        )

        selected = tree.choose_backtrack_node(buggy, check)
        self.assertEqual(selected.node_id, merge)

    def test_algorithm_failure_selects_choice_ancestor(self):
        tree = AdaptivePlanTree("sess_algo", "solve online max-subarray")
        choose = tree.add_child(
            tree.state.root_node_id,
            goal="choose algorithm",
            hypothesis="compare Kadane against segment tree",
            mode="plan",
            checkpoint_quality=0.8,
        )
        kadane = tree.add_child(
            choose,
            goal="execute Kadane branch",
            hypothesis="use Kadane after each update",
            mode="execute",
            checkpoint_quality=0.95,
        )
        detail = tree.add_child(
            kadane,
            goal="estimate complexity",
            hypothesis="O(nq) may still pass",
            mode="execute",
            checkpoint_quality=0.9,
        )
        check = PlanCheckResult(
            checked_node_id=detail,
            passed=False,
            failure_scope="algorithm_choice",
            failed_requirements=["q online updates"],
            reason="The chosen algorithm is too slow for point updates.",
        )

        selected = tree.choose_backtrack_node(detail, check)
        self.assertEqual(selected.node_id, choose)

    def test_revision_creates_sibling_and_abandons_failed_branch(self):
        tree = AdaptivePlanTree("sess_revise", "decide shortest path algorithm")
        choose = tree.add_child(
            tree.state.root_node_id,
            goal="choose shortest-path algorithm",
            hypothesis="Dijkstra if all edges nonnegative, Bellman-Ford otherwise",
            mode="plan",
            checkpoint_quality=0.85,
        )
        dijkstra = tree.add_child(
            choose,
            goal="try Dijkstra",
            hypothesis="use Dijkstra from source",
            mode="execute",
            checkpoint_quality=0.5,
        )
        check = PlanCheckResult(
            checked_node_id=dijkstra,
            passed=False,
            failure_scope="algorithm_choice",
            failed_requirements=["negative edge"],
            reason="Dijkstra precondition fails on negative edges.",
        )

        bellman = tree.revise_from_failure(
            dijkstra,
            check,
            new_goal="try Bellman-Ford",
            new_hypothesis="Bellman-Ford supports negative edges if no negative cycle",
            mode="execute",
            checkpoint_quality=0.8,
        )

        self.assertEqual(tree.nodes[dijkstra].status, "abandoned")
        self.assertEqual(tree.nodes[bellman].parent_id, choose)
        self.assertEqual(tree.state.active_node_id, bellman)
        self.assertEqual(tree.state.revision_count, 1)
        self.assertTrue(any(e.src == dijkstra and e.dst == bellman and e.relation == "plan_revision_of" for e in tree.edges))
        self.assertTrue(any(e.src == dijkstra and e.dst == choose and e.relation == "backtracked_to" for e in tree.edges))

    def test_revision_budget_caps_unbounded_search(self):
        tree = AdaptivePlanTree("sess_budget", "solve task", max_revisions=1)
        branch = tree.add_child(
            tree.state.root_node_id,
            goal="try first branch",
            hypothesis="first idea",
            mode="execute",
        )
        check = PlanCheckResult(
            checked_node_id=branch,
            passed=False,
            failure_scope="unknown",
            reason="failed",
        )
        tree.revise_from_failure(branch, check, new_goal="try second branch", new_hypothesis="second idea")
        active = tree.state.active_node_id
        check2 = PlanCheckResult(
            checked_node_id=active,
            passed=False,
            failure_scope="unknown",
            reason="failed again",
        )
        with self.assertRaises(PlanningBudgetExceeded):
            tree.revise_from_failure(active, check2, new_goal="try third branch", new_hypothesis="third idea")


class TestCoverageAndSessionProjection(unittest.TestCase):
    def test_critical_coverage_miss_blocks_finalization(self):
        tree = AdaptivePlanTree("sess_cov", "write IOI solution")
        final_node = tree.add_child(
            tree.state.root_node_id,
            goal="final answer",
            hypothesis="write segment tree solution",
            mode="finalize",
            checkpoint_quality=0.8,
        )
        frame = GraphTaskFrame(
            session_id="sess_cov",
            constraints=[
                FrameItem("fi_ll", "constraint", "Use long long for sums.", 95, []),
                FrameItem("fi_cpp", "constraint", "Return C++17 code.", 70, []),
            ],
        )
        check = tree.try_finalize(final_node, frame, "Use an int segment tree.")

        self.assertFalse(check.passed)
        self.assertFalse(tree.state.finalized)
        self.assertEqual(tree.nodes[final_node].status, "failed")
        self.assertIn("Use long long for sums.", check.failed_requirements)

    def test_covered_answer_finalizes(self):
        tree = AdaptivePlanTree("sess_pass", "write IOI solution")
        final_node = tree.add_child(
            tree.state.root_node_id,
            goal="final answer",
            hypothesis="write segment tree solution",
            mode="finalize",
        )
        frame = GraphTaskFrame(
            session_id="sess_pass",
            constraints=[FrameItem("fi_ll", "constraint", "Use long long for sums.", 95, [])],
        )
        check = tree.try_finalize(final_node, frame, "Use long long fields in the segment tree.")

        self.assertTrue(check.passed)
        self.assertTrue(tree.state.finalized)
        self.assertEqual(tree.nodes[final_node].status, "passed")

    def test_attach_plan_tree_to_session_persists_nodes_and_edges(self):
        tree = AdaptivePlanTree("sess_attach_plan", "decide Dijkstra safety")
        choose = tree.add_child(
            tree.state.root_node_id,
            goal="choose shortest-path algorithm",
            hypothesis="Dijkstra or Bellman-Ford",
            mode="plan",
        )
        dijkstra = tree.add_child(
            choose,
            goal="try Dijkstra",
            hypothesis="use Dijkstra",
            mode="execute",
        )
        check = PlanCheckResult(
            checked_node_id=dijkstra,
            passed=False,
            failure_scope="algorithm_choice",
            reason="Negative edge violates Dijkstra precondition.",
        )
        tree.revise_from_failure(
            dijkstra,
            check,
            new_goal="try Bellman-Ford",
            new_hypothesis="supports negative edges",
        )

        session = SessionSubgraphController("sess_attach_plan", "Can Dijkstra handle a negative edge?", "merged_graph")
        attach_plan_tree_to_session(session, tree)

        node_types = {node["node_type"] for node in session.subgraph.nodes.values()}
        relations = {edge.relation for edge in session.subgraph.edges}
        self.assertIn("plan_node", node_types)
        self.assertIn("plan_check", node_types)
        self.assertIn("plan_child", relations)
        self.assertIn("checked_by", relations)
        self.assertIn("backtracked_to", relations)


class TestSyntheticAdaptiveDrivers(unittest.TestCase):
    def test_ioi_driver_backtracks_from_kadane_to_segment_tree_and_finalizes(self):
        tree = build_ioi_synthetic_plan_tree()

        self.assertTrue(tree.state.finalized)
        self.assertEqual(tree.nodes["plan_ioi_kadane_direct"].status, "abandoned")
        self.assertEqual(tree.nodes["plan_ioi_choose_algorithm"].status, "passed")
        self.assertEqual(tree.nodes["plan_ioi_segment_tree"].parent_id, "plan_ioi_choose_algorithm")
        self.assertEqual(tree.nodes["plan_ioi_segment_tree"].status, "passed")
        self.assertEqual(tree.nodes["plan_ioi_final_answer"].status, "passed")
        self.assertTrue(any(
            e.src == "plan_ioi_kadane_direct"
            and e.dst == "plan_ioi_choose_algorithm"
            and e.relation == "backtracked_to"
            for e in tree.edges
        ))
        self.assertTrue(any(
            e.src == "plan_ioi_kadane_direct"
            and e.dst == "plan_ioi_segment_tree"
            and e.relation == "plan_revision_of"
            for e in tree.edges
        ))

    def test_dijkstra_driver_backtracks_from_dijkstra_to_bellman_ford_and_finalizes(self):
        tree = build_dijkstra_synthetic_plan_tree()

        self.assertTrue(tree.state.finalized)
        self.assertEqual(tree.nodes["plan_dijkstra_try_dijkstra"].status, "abandoned")
        self.assertEqual(tree.nodes["plan_dijkstra_choose_algorithm"].status, "passed")
        self.assertEqual(tree.nodes["plan_dijkstra_bellman_ford"].parent_id, "plan_dijkstra_choose_algorithm")
        self.assertEqual(tree.nodes["plan_dijkstra_bellman_ford"].status, "passed")
        self.assertEqual(tree.nodes["plan_dijkstra_final_answer"].status, "passed")
        self.assertTrue(any(
            e.src == "plan_dijkstra_try_dijkstra"
            and e.dst == "plan_dijkstra_choose_algorithm"
            and e.relation == "backtracked_to"
            for e in tree.edges
        ))
        self.assertTrue(any(
            e.src == "plan_dijkstra_try_dijkstra"
            and e.dst == "plan_dijkstra_bellman_ford"
            and e.relation == "plan_revision_of"
            for e in tree.edges
        ))

    def test_synthetic_driver_sessions_persist_replayable_subgraphs(self):
        with tempfile.TemporaryDirectory() as td:
            paths = persist_synthetic_plan_sessions(Path(td))

            self.assertEqual(set(paths), {"ioi", "dijkstra"})
            for path in paths.values():
                subgraph_path = path / "subgraph.json"
                audit_path = path / "audit_log.jsonl"
                self.assertTrue(subgraph_path.exists())
                self.assertTrue(audit_path.exists())
                data = json.loads(subgraph_path.read_text(encoding="utf-8"))
                node_types = {node["node_type"] for node in data["nodes"].values()}
                relations = {edge["relation"] for edge in data["edges"]}
                self.assertIn("plan_node", node_types)
                self.assertIn("plan_check", node_types)
                self.assertIn("backtracked_to", relations)
                self.assertIn("plan_revision_of", relations)


if __name__ == "__main__":
    unittest.main()
