"""Tests for session_to_graph consolidation module."""
from __future__ import annotations

import json
import tempfile
import time
from pathlib import Path
from unittest import TestCase

from graph_core import Edge, MemoryGraph, Node
from reasoning.outcome_scorer import SubstrateOutcomeRow
from reasoning.reasoning_loop import ReasoningResult, SessionSubgraph
from reasoning.session_to_graph import (
    GraphUpdate,
    SessionMemory,
    apply_graph_updates,
    build_graph_updates,
    extract_session_memory,
    save_graph_atomically,
)


def _make_graph() -> MemoryGraph:
    nodes = {
        "sig_1": Node(id="sig_1", text="Use a segment tree for range queries", node_type="claim"),
        "sig_2": Node(id="sig_2", text="Handle all-negative arrays by returning max element", node_type="claim"),
        "sig_3": Node(id="sig_3", text="Use long long for overflow safety", node_type="claim"),
        "concept_a": Node(id="concept_a", text="Segment tree stores sum, prefix, suffix, best", node_type="concept"),
    }
    edges = [
        Edge(src="sig_1", dst="concept_a", relation="supports", strength=0.6),
    ]
    return MemoryGraph(nodes, edges)


def _make_fake_result(
    answer: str = "segment tree approach",
    signal_ids: list | None = None,
    source_ids: list | None = None,
) -> ReasoningResult:
    if signal_ids is None:
        signal_ids = ["sigv2_sig_1", "sigv2_sig_2"]
    if source_ids is None:
        source_ids = ["sig_1", "sig_2"]

    debug_dump = [
        {"id": sid, "source_node_id": src_id, "text_preview": "test"}
        for sid, src_id in zip(signal_ids, source_ids)
    ]

    return ReasoningResult(
        answer=answer,
        reasoning_trace="",
        raw_outputs=[],
        session_subgraph=SessionSubgraph(session_id="test", query="?", graph_id="test"),
        session_subgraph_path=Path("."),
        audit_summary={
            "debug_signal_dump": debug_dump,
        },
        consolidation_decisions=[],
        budget_usage={},
        dispatch_outcomes=[],
        anchor_ids=[],
        iterations_completed=1,
    )


def _make_fake_outcome_row(
    correct: bool = True,
    score: float = 1.0,
    source: str = "deterministic",
) -> SubstrateOutcomeRow:
    return SubstrateOutcomeRow(
        packet_id="p1",
        delta_transaction_id="t1",
        checker_results=[],
        final_answer="segment tree approach",
        outcome_correct=correct,
        outcome_score=score,
        outcome_source=source,
        step_count=1,
        llm_calls=1,
        task_id="test_task",
        task_kind="algorithm_design",
        question="Maintain array under online updates?",
        elapsed_sec=1.0,
    )


# ── extract_session_memory ───────────────────────────────────────────────


class TestExtractSessionMemory(TestCase):
    def test_extracts_signal_ids(self):
        result = _make_fake_result()
        task = {"id": "t1", "question": "test?", "kind": "test"}
        mem = extract_session_memory(result, task)
        self.assertEqual(mem.task_id, "t1")
        self.assertEqual(mem.activated_signal_ids, ["sigv2_sig_1", "sigv2_sig_2"])
        self.assertEqual(mem.activated_signal_source_ids, ["sig_1", "sig_2"])

    def test_with_outcome_row(self):
        result = _make_fake_result()
        task = {"id": "t1", "question": "test?"}
        row = _make_fake_outcome_row(correct=True, score=1.0)
        mem = extract_session_memory(result, task, outcome_row=row)
        self.assertEqual(mem.outcome_correct, True)
        self.assertEqual(mem.outcome_score, 1.0)
        self.assertEqual(mem.outcome_source, "deterministic")

    def test_without_outcome_row(self):
        result = _make_fake_result()
        task = {"id": "t1", "question": "test?"}
        mem = extract_session_memory(result, task)
        self.assertIsNone(mem.outcome_correct)
        self.assertEqual(mem.outcome_score, 0.0)
        self.assertEqual(mem.outcome_source, "unknown")

    def test_memory_id_unique(self):
        mem1 = SessionMemory(
            task_id="t1", question="q", answer="a",
            activated_signal_ids=[], activated_signal_source_ids=[],
            outcome_correct=True, outcome_score=1.0, outcome_source="det",
            timestamp="2026-01-01T00:00:00",
        )
        mem2 = SessionMemory(
            task_id="t1", question="q", answer="a",
            activated_signal_ids=[], activated_signal_source_ids=[],
            outcome_correct=True, outcome_score=1.0, outcome_source="det",
            timestamp="2026-01-01T00:00:01",
        )
        self.assertNotEqual(mem1.memory_id, mem2.memory_id)

    def test_summary_text_truncates(self):
        mem = SessionMemory(
            task_id="t1",
            question="A" * 200,
            answer="B" * 200,
            activated_signal_ids=[], activated_signal_source_ids=[],
            outcome_correct=True, outcome_score=1.0, outcome_source="det",
            timestamp="",
        )
        text = mem.summary_text(max_question=10, max_answer=10)
        self.assertIn("A" * 10, text)
        self.assertIn("B" * 10, text)
        self.assertNotIn("A" * 11, text)
        self.assertNotIn("B" * 11, text)


# ── build_graph_updates ──────────────────────────────────────────────────


class TestBuildGraphUpdates(TestCase):
    def test_adds_memory_node_per_memory(self):
        graph = _make_graph()
        mems = [
            SessionMemory(
                task_id="t1", question="q", answer="a",
                activated_signal_ids=[], activated_signal_source_ids=[],
                outcome_correct=True, outcome_score=1.0, outcome_source="det",
                timestamp="2026-01-01T00:00:00",
            ),
            SessionMemory(
                task_id="t2", question="q", answer="a",
                activated_signal_ids=[], activated_signal_source_ids=[],
                outcome_correct=False, outcome_score=0.0, outcome_source="det",
                timestamp="2026-01-01T00:00:01",
            ),
        ]
        updates = build_graph_updates(mems, graph)
        self.assertEqual(len(updates.new_nodes), 2)

    def test_links_to_existing_signal_nodes(self):
        graph = _make_graph()
        mem = SessionMemory(
            task_id="t1", question="q", answer="a",
            activated_signal_ids=["sigv2_sig_1"],
            activated_signal_source_ids=["sig_1"],
            outcome_correct=True, outcome_score=1.0, outcome_source="det",
            timestamp="2026-01-01T00:00:00",
        )
        updates = build_graph_updates([mem], graph)
        # 1 memory node + 1 edge to sig_1
        self.assertEqual(len(updates.new_edges), 1)
        edge = updates.new_edges[0]
        self.assertEqual(edge.src, mem.memory_id)
        self.assertEqual(edge.dst, "sig_1")
        self.assertEqual(edge.relation, "used_signal")

    def test_skips_missing_source_node(self):
        graph = _make_graph()
        mem = SessionMemory(
            task_id="t1", question="q", answer="a",
            activated_signal_ids=["sigv2_nonexistent"],
            activated_signal_source_ids=["nonexistent_sig"],
            outcome_correct=True, outcome_score=1.0, outcome_source="det",
            timestamp="2026-01-01T00:00:00",
        )
        updates = build_graph_updates([mem], graph)
        signal_edges = [e for e in updates.new_edges if e.relation == "used_signal"]
        self.assertEqual(len(signal_edges), 0)

    def test_links_to_related_by_lexical_overlap(self):
        graph = _make_graph()
        mem = SessionMemory(
            task_id="t1",
            question="segment tree for range sum queries",
            answer="Use a segment tree approach",
            activated_signal_ids=[], activated_signal_source_ids=[],
            outcome_correct=True, outcome_score=1.0, outcome_source="det",
            timestamp="2026-01-01T00:00:00",
        )
        updates = build_graph_updates([mem], graph)
        related_edges = [e for e in updates.new_edges if e.relation == "related"]
        dsts = {e.dst for e in related_edges}
        self.assertIn("sig_1", dsts)  # "segment tree" overlaps
        self.assertIn("concept_a", dsts)  # "segment tree" overlaps

    def test_no_double_link(self):
        graph = _make_graph()
        mem = SessionMemory(
            task_id="t1",
            question="segment tree",
            answer="segment tree approach",
            activated_signal_ids=["sigv2_sig_1"],
            activated_signal_source_ids=["sig_1"],
            outcome_correct=True, outcome_score=1.0, outcome_source="det",
            timestamp="2026-01-01T00:00:00",
        )
        updates = build_graph_updates([mem], graph)
        all_dsts = [e.dst for e in updates.new_edges]
        self.assertEqual(all_dsts.count("sig_1"), 1)


# ── apply_graph_updates ──────────────────────────────────────────────────


class TestApplyGraphUpdates(TestCase):
    def test_does_not_duplicate_existing_node(self):
        graph = _make_graph()
        pre_count = len(graph.nodes)
        updates = GraphUpdate(
            new_nodes=[Node(id="sig_1", text="dupe", node_type="claim")],
        )
        added = apply_graph_updates(graph, updates)
        self.assertEqual(added, 0)
        self.assertEqual(len(graph.nodes), pre_count)

    def test_adds_edges(self):
        graph = _make_graph()
        mid = "mem_1"
        graph.nodes[mid] = Node(id=mid, text="memory", node_type="session_memory")
        updates = GraphUpdate(
            new_edges=[Edge(src=mid, dst="sig_1", relation="used_signal")],
        )
        apply_graph_updates(graph, updates)
        self.assertIsNotNone(graph.edge_between(mid, "sig_1"))

    def test_edge_strength_update(self):
        graph = _make_graph()
        updates = GraphUpdate(
            edge_strength_updates=[("sig_1", "concept_a", 0.1)],
        )
        apply_graph_updates(graph, updates)
        edge = graph.edge_between("sig_1", "concept_a")
        self.assertIsNotNone(edge)
        self.assertAlmostEqual(edge.strength, 0.7)  # 0.6 + 0.1

    def test_edge_strength_clamped(self):
        graph = _make_graph()
        edge = graph.edge_between("sig_1", "concept_a")
        edge.strength = 0.0
        updates = GraphUpdate(
            edge_strength_updates=[("sig_1", "concept_a", -0.1)],
        )
        apply_graph_updates(graph, updates)
        self.assertGreaterEqual(edge.strength, 0.0)

    def test_rebuild_index_after_mutations(self):
        graph = _make_graph()
        mid = "mem_new"
        graph.nodes[mid] = Node(id=mid, text="new memory", node_type="session_memory")
        updates = GraphUpdate(
            new_edges=[Edge(src=mid, dst="sig_1", relation="used_signal")],
        )
        apply_graph_updates(graph, updates)
        # After _rebuild_index, edge_between should work
        self.assertIsNotNone(graph.edge_between(mid, "sig_1"))


# ── save_graph_atomically ────────────────────────────────────────────────


class TestSaveGraphAtomically(TestCase):
    def test_writes_and_renames(self):
        graph = _make_graph()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "graph.json"
            save_graph_atomically(graph, path)
            self.assertTrue(path.exists())
            loaded = MemoryGraph.load_json(path)
            self.assertEqual(len(loaded.nodes), len(graph.nodes))
            self.assertEqual(len(loaded.edges), len(graph.edges))


# ── Integration smoke ────────────────────────────────────────────────────


class TestBatchConsolidate(TestCase):
    def test_no_results_returns_zero(self):
        from reasoning.session_to_graph import batch_consolidate

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "graph.json"
            _make_graph().save_json(path)
            count = batch_consolidate([], path)
            self.assertEqual(count, 0)

    def test_missing_graph_returns_zero(self):
        from reasoning.session_to_graph import batch_consolidate

        count = batch_consolidate([(_make_fake_result(), {"id": "t1"})], "nonexistent.json")
        self.assertEqual(count, 0)

    def test_end_to_end(self):
        from reasoning.session_to_graph import batch_consolidate

        graph = _make_graph()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "graph.json"
            graph.save_json(path)

            pre_count = len(graph.nodes)
            results = [
                (_make_fake_result(answer="segment tree for max subarray"), {"id": "t1", "question": "segment tree?"}),
            ]
            added = batch_consolidate(results, path, outcome_rows=[_make_fake_outcome_row()])
            self.assertGreater(added, 0)

            loaded = MemoryGraph.load_json(path)
            self.assertEqual(len(loaded.nodes), pre_count + added)

    def test_batch_multiple_sessions(self):
        from reasoning.session_to_graph import batch_consolidate

        graph = _make_graph()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "graph.json"
            graph.save_json(path)

            results = [
                (_make_fake_result(answer="segment tree"), {"id": "t1", "question": "segment tree?"}),
                (_make_fake_result(answer="different approach", signal_ids=["sigv2_sig_3"], source_ids=["sig_3"]),
                 {"id": "t2", "question": "other question?"}),
            ]
            rows = [_make_fake_outcome_row(), _make_fake_outcome_row(correct=False, score=0.0)]
            added = batch_consolidate(results, path, outcome_rows=rows)
            self.assertEqual(added, 2)

            loaded = MemoryGraph.load_json(path)
            memory_nodes = [n for n in loaded.nodes.values() if n.node_type == "session_memory"]
            self.assertEqual(len(memory_nodes), 2)
