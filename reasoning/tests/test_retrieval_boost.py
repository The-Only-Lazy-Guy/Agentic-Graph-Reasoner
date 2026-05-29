"""Tests for reasoning/retrieval_boost.py.

Strategy: mock anchor_retrieval._score_nodes to return known scores,
then verify the boost logic re-ranks failure_pattern nodes correctly.

We avoid loading the real embedder model in unit tests — that's
integration territory (covered in Sub-phase 1.9).
"""
from __future__ import annotations

import unittest
from unittest.mock import patch

import numpy as np

from reasoning import retrieval_boost


class _FakeNode:
    """Minimal Node-shaped object for testing. The retrieval pipeline
    reads node_type and a few attrs; we mock the rest."""
    def __init__(self, node_id, node_type, text="", importance=0.5):
        self.id = node_id
        self.node_type = node_type
        self.text = text
        self.importance = importance


class _FakeGraph:
    def __init__(self, nodes):
        self.nodes = {n.id: n for n in nodes}


def _make_mixed_graph():
    """Five fact nodes, two failure_pattern nodes. Failure patterns
    have slightly lower raw scores; the boost should pull them into top-k."""
    return _FakeGraph([
        _FakeNode("f1", "fact"),
        _FakeNode("f2", "fact"),
        _FakeNode("f3", "fact"),
        _FakeNode("f4", "fact"),
        _FakeNode("f5", "fact"),
        _FakeNode("fp1", "failure_pattern"),
        _FakeNode("fp2", "failure_pattern"),
    ])


class TestBoostLogic(unittest.TestCase):
    """Verify the multiplier is applied only to failure_pattern nodes
    and that top-k changes accordingly."""

    def _patched(self, scores_map):
        """Helper: patch the three anchor_retrieval primitives so we can
        feed in known scores and node order."""
        graph = _make_mixed_graph()
        node_ids = list(graph.nodes.keys())
        scores = np.array([scores_map[nid] for nid in node_ids], dtype=np.float32)
        embeddings = np.zeros((len(node_ids), 4), dtype=np.float32)
        return graph, node_ids, scores, embeddings

    def test_no_failure_patterns_in_topk_without_boost(self):
        # Scores set so fact nodes outrank failure_patterns by a small margin
        graph, node_ids, scores, emb = self._patched({
            "f1": 0.9, "f2": 0.8, "f3": 0.7, "f4": 0.6, "f5": 0.5,
            "fp1": 0.45, "fp2": 0.40,
        })
        with patch.object(retrieval_boost, "get_graph_embeddings", return_value=(node_ids, emb)), \
             patch.object(retrieval_boost, "_score_nodes", return_value=scores):
            # Without boost: failure_patterns should NOT be in top-3
            ids = retrieval_boost.retrieve_with_failure_boost(
                "q", graph, k=3, failure_boost=1.0,
            )
            self.assertEqual(ids, ["f1", "f2", "f3"])

    def test_boost_lifts_failure_pattern_into_topk(self):
        # Same scores; with 1.4x boost, fp1 (0.45 * 1.4 = 0.63) should
        # surpass f4 (0.6) and f5 (0.5)
        graph, node_ids, scores, emb = self._patched({
            "f1": 0.9, "f2": 0.8, "f3": 0.7, "f4": 0.6, "f5": 0.5,
            "fp1": 0.45, "fp2": 0.40,
        })
        with patch.object(retrieval_boost, "get_graph_embeddings", return_value=(node_ids, emb)), \
             patch.object(retrieval_boost, "_score_nodes", return_value=scores):
            ids = retrieval_boost.retrieve_with_failure_boost(
                "q", graph, k=4, failure_boost=1.4,
            )
            # Top 4 should now include fp1 (boosted to 0.63), bumping out f5
            self.assertEqual(set(ids[:4]), {"f1", "f2", "f3", "fp1"})
            self.assertNotIn("f5", ids)

    def test_boost_factor_one_is_identity(self):
        graph, node_ids, scores, emb = self._patched({
            "f1": 0.9, "f2": 0.8, "f3": 0.7, "f4": 0.6, "f5": 0.5,
            "fp1": 0.45, "fp2": 0.40,
        })
        with patch.object(retrieval_boost, "get_graph_embeddings", return_value=(node_ids, emb)), \
             patch.object(retrieval_boost, "_score_nodes", return_value=scores):
            ids_no_boost = retrieval_boost.retrieve_with_failure_boost(
                "q", graph, k=5, failure_boost=1.0,
            )
            ids_default = retrieval_boost.retrieve_with_failure_boost(
                "q", graph, k=5, failure_boost=1.4,
            )
            self.assertEqual(ids_no_boost, ["f1", "f2", "f3", "f4", "f5"])
            # With 1.4x, fp1 (0.63) bumps in
            self.assertIn("fp1", ids_default)

    def test_high_boost_dominates(self):
        graph, node_ids, scores, emb = self._patched({
            "f1": 0.9, "f2": 0.8, "f3": 0.7, "f4": 0.6, "f5": 0.5,
            "fp1": 0.45, "fp2": 0.40,
        })
        # 5x boost: fp1=2.25, fp2=2.0 — both above all facts
        with patch.object(retrieval_boost, "get_graph_embeddings", return_value=(node_ids, emb)), \
             patch.object(retrieval_boost, "_score_nodes", return_value=scores):
            ids = retrieval_boost.retrieve_with_failure_boost(
                "q", graph, k=3, failure_boost=5.0,
            )
            self.assertEqual(set(ids), {"fp1", "fp2", "f1"})

    def test_empty_graph_returns_empty(self):
        graph = _FakeGraph([])
        ids = retrieval_boost.retrieve_with_failure_boost("q", graph, k=5)
        self.assertEqual(ids, [])


class TestDiagnoseBoostEffect(unittest.TestCase):
    def test_diagnostic_added_and_removed(self):
        graph = _make_mixed_graph()
        node_ids = list(graph.nodes.keys())
        scores = np.array([0.9, 0.8, 0.7, 0.6, 0.5, 0.45, 0.40], dtype=np.float32)
        emb = np.zeros((len(node_ids), 4), dtype=np.float32)

        with patch.object(retrieval_boost, "get_graph_embeddings", return_value=(node_ids, emb)), \
             patch.object(retrieval_boost, "_score_nodes", return_value=scores):
            d = retrieval_boost.diagnose_boost_effect(
                "q", graph, k=4, failure_boost=1.4,
            )

        self.assertEqual(d["unboosted"], ["f1", "f2", "f3", "f4"])
        self.assertIn("fp1", d["boosted"])
        self.assertIn("fp1", d["added_by_boost"])
        # f4 (0.6) is bumped out by fp1 (0.45*1.4=0.63)
        self.assertIn("f4", d["removed_by_boost"])


if __name__ == "__main__":
    unittest.main()
