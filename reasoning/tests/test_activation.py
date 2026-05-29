"""Tests for reasoning/activation.py - Phase 3C."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from graph_core import Edge, MemoryGraph, Node
from reasoning.activation import (
    DEFAULT_HEURISTICS,
    ActivationTraceLogger,
    FrameItem,
    GraphActivationTrace,
    GraphTaskFrame,
    evaluate_coverage,
    render_task_frame,
    run_graph_activation,
    attach_activation_to_session,
)
from reasoning.procedures.verify_nonneg_edges import build_verify_nonneg_edges
from reasoning.procedures.verify_shortest_path import build_verify_shortest_path
from reasoning.session_subgraph import SessionSubgraphController


def _make_graph() -> MemoryGraph:
    nodes = {
        "seg_example": Node(
            id="seg_example",
            node_type="example",
            text=(
                "Dynamic maximum subarray with point updates uses a segment tree. "
                "Each node stores sum, max_prefix, max_suffix, and max_sub."
            ),
            confidence=0.9,
            importance=0.9,
            metadata={"domain": "computer_science"},
        ),
        "dijkstra_warning": Node(
            id="dijkstra_warning",
            node_type="claim",
            text="Dijkstra is unsafe on graphs containing a negative edge.",
            confidence=0.9,
            importance=0.8,
            metadata={"domain": "computer_science"},
        ),
        "generic": Node(
            id="generic",
            node_type="fact",
            text="Algorithms should state complexity clearly.",
            confidence=0.7,
            importance=0.4,
            metadata={"domain": "computer_science"},
        ),
    }
    edges = [
        Edge(src="seg_example", dst="generic", relation="related", strength=0.5),
        Edge(src="dijkstra_warning", dst="generic", relation="related", strength=0.5),
    ]
    return MemoryGraph(nodes, edges, metadata={"domain": "computer_science"})


class TestActivationRoundTrip(unittest.TestCase):
    def test_activation_heuristics_are_named(self):
        self.assertEqual(DEFAULT_HEURISTICS.generic_constraint_min_confidence, 0.55)
        self.assertEqual(DEFAULT_HEURISTICS.provisional_missing_context_confidence, 0.65)
        self.assertEqual(DEFAULT_HEURISTICS.content_token_min_chars, 4)

    def test_activation_trace_round_trip(self):
        graph = _make_graph()
        trace = run_graph_activation(
            session_id="sess_test",
            graph_id="merged_graph",
            question=(
                "Solve dynamic maximum subarray sum under point updates, "
                "n,q <= 200000, negative values allowed, non-empty subarray, C++17."
            ),
            graph=graph,
            anchor_ids=["seg_example"],
        )
        restored = GraphActivationTrace.from_dict(json.loads(json.dumps(trace.to_dict())))
        self.assertEqual(restored.session_id, "sess_test")
        self.assertGreater(len(restored.signals), 0)
        self.assertGreater(len(restored.task_frame.all_items()), 0)

    def test_logger_append_and_read_all(self):
        graph = _make_graph()
        trace = run_graph_activation(
            session_id="sess_log",
            graph_id="merged_graph",
            question="Is Dijkstra safe with a negative edge?",
            graph=graph,
            anchor_ids=["dijkstra_warning"],
        )
        with tempfile.TemporaryDirectory() as td:
            logger = ActivationTraceLogger(Path(td))
            logger.append(trace)
            loaded = logger.read_all()
            self.assertEqual(len(loaded), 1)
            self.assertEqual(loaded[0].session_id, "sess_log")


class TestActivationFrame(unittest.TestCase):
    def test_ioi_task_frame_contains_expected_guidance(self):
        graph = _make_graph()
        trace = run_graph_activation(
            session_id="sess_ioi",
            graph_id="merged_graph",
            question=(
                "Solve dynamic maximum subarray sum under point updates. "
                "n,q <= 200000, negative values allowed, non-empty subarray, C++17."
            ),
            graph=graph,
            anchor_ids=["seg_example"],
        )
        rendered = render_task_frame(trace.task_frame).lower()
        self.assertIn("segment tree", rendered)
        self.assertIn("long long", rendered)
        self.assertIn("all-negative", rendered)
        self.assertIn("non-empty", rendered)

    def test_dijkstra_task_frame_suggests_procedure_and_pitfall(self):
        graph = _make_graph()
        trace = run_graph_activation(
            session_id="sess_dijkstra",
            graph_id="merged_graph",
            question="Can I trust Dijkstra on a graph with one negative edge?",
            graph=graph,
            anchor_ids=["dijkstra_warning"],
            procedure_pool=[build_verify_shortest_path(), build_verify_nonneg_edges()],
        )
        rendered = render_task_frame(trace.task_frame).lower()
        self.assertIn("dijkstra", rendered)
        self.assertIn("negative edge", rendered)
        self.assertIn("procedure may apply", rendered)

    def test_missing_context_creates_session_gap_and_bridge(self):
        graph = MemoryGraph(
            nodes={
                "anchor": Node(
                    id="anchor",
                    node_type="fact",
                    text="The query is about online updates.",
                    confidence=0.7,
                    importance=0.7,
                )
            },
            edges=[],
        )
        trace = run_graph_activation(
            session_id="sess_gap",
            graph_id="toy",
            question="Need process 200000 updates safely.",
            graph=graph,
            anchor_ids=["anchor"],
        )
        node_types = {n["node_type"] for n in trace.provisional_nodes}
        self.assertIn("session_gap", node_types)
        self.assertIn("session_bridge", node_types)


class TestCoverageAndSessionProjection(unittest.TestCase):
    def test_coverage_flags_missing_long_long_and_all_negative(self):
        frame = GraphTaskFrame(
            session_id="sess_cov",
            constraints=[
                FrameItem("fi_long", "constraint", "Use long long for sums.", 90, []),
                FrameItem(
                    "fi_neg",
                    "pitfall",
                    "For non-empty subarrays, all-negative arrays must return the maximum element, not 0.",
                    95,
                    [],
                ),
            ],
        )
        result = evaluate_coverage(frame, "Use a segment tree with int fields.")
        self.assertIn("fi_long", result["missed_item_ids"])
        self.assertIn("fi_neg", result["missed_item_ids"])

    def test_attach_activation_adds_session_scoped_nodes(self):
        graph = _make_graph()
        trace = run_graph_activation(
            session_id="sess_attach",
            graph_id="merged_graph",
            question=(
                "Solve dynamic maximum subarray sum under point updates, "
                "negative values allowed, non-empty subarray."
            ),
            graph=graph,
            anchor_ids=["seg_example"],
        )
        session = SessionSubgraphController("sess_attach", trace.context.question, "merged_graph")
        attach_activation_to_session(session, trace)
        node_types = {n.get("node_type") for n in session.subgraph.nodes.values()}
        self.assertIn("activation_signal", node_types)
        self.assertIn("task_frame_item", node_types)
        for node in session.subgraph.nodes.values():
            if node.get("node_type") in {"session_gap", "session_bridge"}:
                self.assertTrue(node["metadata"]["session_scoped"])


if __name__ == "__main__":
    unittest.main()
