from __future__ import annotations

import unittest

from graph_core import Edge, MemoryGraph, Node
from reasoning.micro_controller import (
    MicroAction,
    _active_invalidators,
    run_micro_epistemic_controller,
)


def _graph_with_invalidator() -> MemoryGraph:
    return MemoryGraph(
        nodes={
            "claim_dijkstra_negative_edge_invalid": Node(
                id="claim_dijkstra_negative_edge_invalid",
                node_type="claim",
                confidence=0.95,
                text="Dijkstra is invalid when any negative edge weight exists.",
            ),
            "cond_dag_longest_path": Node(
                id="cond_dag_longest_path",
                node_type="claim",
                confidence=0.9,
                text="Question is about longest path in a directed acyclic graph, "
                     "not general Dijkstra shortest path.",
            ),
            "fact_negate_weights_dag_longest": Node(
                id="fact_negate_weights_dag_longest",
                node_type="fact",
                confidence=0.9,
                text="For longest path in a DAG, negate edge weights and run DAG shortest path.",
            ),
        },
        edges=[
            Edge(
                src="claim_dijkstra_negative_edge_invalid",
                dst="cond_dag_longest_path",
                relation="invalidated_by",
            ),
        ],
    )


class TestActiveInvalidators(unittest.TestCase):
    def test_invalidator_fires_when_question_overlaps_condition(self):
        graph = _graph_with_invalidator()
        question = (
            "Can I use Dijkstra to find the longest path in a directed acyclic graph?"
        )
        hits = _active_invalidators(
            graph=graph,
            selected_node_ids=["claim_dijkstra_negative_edge_invalid"],
            question=question,
        )
        self.assertEqual(len(hits), 1)
        src, dst, score = hits[0]
        self.assertEqual(src, "claim_dijkstra_negative_edge_invalid")
        self.assertEqual(dst, "cond_dag_longest_path")
        self.assertGreater(score, 0.10)

    def test_invalidator_does_not_fire_on_unrelated_question(self):
        graph = _graph_with_invalidator()
        question = "What is the area of a triangle with sides 3, 4, 5?"
        hits = _active_invalidators(
            graph=graph,
            selected_node_ids=["claim_dijkstra_negative_edge_invalid"],
            question=question,
        )
        self.assertEqual(hits, [])

    def test_invalidator_ignored_when_source_not_selected(self):
        graph = _graph_with_invalidator()
        question = (
            "Can I use Dijkstra to find the longest path in a directed acyclic graph?"
        )
        hits = _active_invalidators(
            graph=graph,
            selected_node_ids=["fact_negate_weights_dag_longest"],
            question=question,
        )
        self.assertEqual(hits, [])


if __name__ == "__main__":
    unittest.main()
