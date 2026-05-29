from __future__ import annotations

import unittest

from graph_core import Edge, MemoryGraph, Node
from reasoning.scoped_edits import (
    VALIDATION_ACCEPT,
    VALIDATION_NEEDS_REVIEW,
    VALIDATION_REJECT,
    VALIDATION_SOFT_ONLY,
    patches_from_graph_edits,
    summarize_patches,
    validate_patches,
)


def _graph() -> MemoryGraph:
    return MemoryGraph(
        nodes={
            "dijkstra_requires_nonnegative": Node(
                id="dijkstra_requires_nonnegative",
                node_type="claim",
                confidence=0.98,
                text="Dijkstra's algorithm is correct when all edge weights are nonnegative.",
            ),
            "negative_edge_breaks_invariant": Node(
                id="negative_edge_breaks_invariant",
                node_type="explanation",
                confidence=0.96,
                text="A negative edge can improve a distance after Dijkstra has already finalized that vertex.",
            ),
            "shortest_path_grid_apply": Node(
                id="shortest_path_grid_apply",
                node_type="application",
                confidence=0.86,
                text="Use BFS on an unweighted grid to compute shortest paths by layers.",
            ),
            "fenwick_prefix_update": Node(
                id="fenwick_prefix_update",
                node_type="claim",
                confidence=0.93,
                text="Fenwick-tree updates and prefix queries both run in O(log n).",
            ),
        },
        edges=[
            Edge(
                src="dijkstra_requires_nonnegative",
                dst="negative_edge_breaks_invariant",
                relation="support",
            )
        ],
    )


class ScopedEditTests(unittest.TestCase):
    def test_soft_reinforcement_is_soft_only(self) -> None:
        graph = _graph()
        edits = [{
            "op": "increment_meta",
            "node_id": "dijkstra_requires_nonnegative",
            "field": "session_cite_count",
            "delta": 1,
            "tier": "soft",
        }]

        patches = validate_patches(patches_from_graph_edits(edits, graph=graph), graph)

        self.assertEqual(1, len(patches))
        self.assertEqual(VALIDATION_SOFT_ONLY, patches[0].validation.status)
        self.assertEqual("reinforce_existing", patches[0].patch_type)

    def test_supported_claim_can_be_accepted(self) -> None:
        graph = _graph()
        edits = [{
            "op": "add_node",
            "node_id": "claim_dijkstra_negative_not_trusted",
            "node_type": "claim",
            "text": "Dijkstra is not guaranteed to be correct with a negative edge because the edge can improve a finalized distance later.",
            "metadata": {
                "evidence_node_ids": ["negative_edge_breaks_invariant"],
                "source_session": "sess_test",
            },
            "tier": "add",
        }]

        patches = validate_patches(patches_from_graph_edits(edits, graph=graph), graph)

        self.assertEqual(VALIDATION_ACCEPT, patches[0].validation.status)
        self.assertGreaterEqual(patches[0].validation.support_score, 0.12)

    def test_high_risk_claim_without_evidence_needs_review(self) -> None:
        graph = _graph()
        edits = [{
            "op": "add_node",
            "node_id": "claim_no_evidence",
            "node_type": "claim",
            "text": "Dijkstra can safely ignore negative edge weights.",
            "metadata": {"source_session": "sess_test"},
            "tier": "add",
        }]

        patches = validate_patches(patches_from_graph_edits(edits, graph=graph), graph)

        self.assertEqual(VALIDATION_NEEDS_REVIEW, patches[0].validation.status)
        self.assertIn("no explicit evidence", " ".join(patches[0].validation.reasons))

    def test_edge_to_node_added_in_same_batch_is_allowed(self) -> None:
        graph = _graph()
        edits = [
            {
                "op": "add_node",
                "node_id": "atom_negative_edge_failure",
                "node_type": "reasoning_atom",
                "text": "Negative edges can break Dijkstra's finalized-distance invariant.",
                "metadata": {
                    "supporting_node_ids": ["negative_edge_breaks_invariant"],
                    "source_session": "sess_test",
                },
                "tier": "add",
            },
            {
                "op": "add_edge",
                "src": "atom_negative_edge_failure",
                "dst": "negative_edge_breaks_invariant",
                "relation": "derived_from",
                "metadata": {"source_session": "sess_test"},
                "tier": "add",
            },
        ]

        patches = validate_patches(patches_from_graph_edits(edits, graph=graph), graph)

        self.assertNotEqual(VALIDATION_REJECT, patches[1].validation.status)

    def test_edge_missing_endpoint_is_rejected(self) -> None:
        graph = _graph()
        edits = [{
            "op": "add_edge",
            "src": "missing_node",
            "dst": "negative_edge_breaks_invariant",
            "relation": "derived_from",
            "metadata": {"source_session": "sess_test"},
            "tier": "add",
        }]

        patches = validate_patches(patches_from_graph_edits(edits, graph=graph), graph)

        self.assertEqual(VALIDATION_REJECT, patches[0].validation.status)

    def test_noisy_strategy_key_nodes_need_review(self) -> None:
        graph = _graph()
        edits = [{
            "op": "add_node",
            "node_id": "strat_dijkstra_negative",
            "node_type": "strategy",
            "text": "Strategy family: algorithm_applicability\nKeywords: dijkstra, negative edge\nSlot order: verdict, reason, alternative",
            "metadata": {
                "task_family": "algorithm_applicability",
                "slot_order": ["verdict", "reason", "alternative"],
                "key_node_ids": [
                    "negative_edge_breaks_invariant",
                    "shortest_path_grid_apply",
                ],
                "source_session": "sess_test",
            },
            "tier": "add",
        }]

        patches = validate_patches(
            patches_from_graph_edits(edits, graph=graph, question="Can Dijkstra be trusted with one negative edge?"),
            graph,
        )

        self.assertEqual(VALIDATION_NEEDS_REVIEW, patches[0].validation.status)
        self.assertIn("low_relevance_evidence", " ".join(patches[0].validation.warnings))

    def test_duplicate_node_id_becomes_soft_only(self) -> None:
        graph = _graph()
        edits = [{
            "op": "add_node",
            "node_id": "dijkstra_requires_nonnegative",
            "node_type": "claim",
            "text": "Duplicate text",
            "metadata": {"evidence_node_ids": ["negative_edge_breaks_invariant"]},
            "tier": "add",
        }]

        patches = validate_patches(patches_from_graph_edits(edits, graph=graph), graph)
        summary = summarize_patches(patches)

        self.assertEqual(VALIDATION_SOFT_ONLY, patches[0].validation.status)
        self.assertEqual(1, summary["by_status"][VALIDATION_SOFT_ONLY])

    def test_irrelevant_soft_reinforcement_needs_review(self) -> None:
        graph = _graph()
        edits = [{
            "op": "increment_meta",
            "node_id": "shortest_path_grid_apply",
            "field": "session_cite_count",
            "delta": 1,
            "tier": "soft",
        }]

        patches = validate_patches(
            patches_from_graph_edits(
                edits,
                graph=graph,
                question="Design a real-time leaderboard with rank pagination and score updates.",
            ),
            graph,
        )

        self.assertEqual(VALIDATION_NEEDS_REVIEW, patches[0].validation.status)
        self.assertIn("low_relevance_reinforcement", " ".join(patches[0].validation.warnings))

    def test_strategy_edges_inherit_parent_review_status(self) -> None:
        graph = _graph()
        edits = [
            {
                "op": "add_node",
                "node_id": "strat_bad_leaderboard",
                "node_type": "strategy",
                "text": "Strategy family: design_synthesis\nKeywords: leaderboard, rank\nSlot order: core_structure, answer",
                "metadata": {
                    "task_family": "design_synthesis",
                    "slot_order": ["core_structure", "answer"],
                    "key_node_ids": ["shortest_path_grid_apply"],
                    "source_session": "sess_test",
                },
                "tier": "add",
            },
            {
                "op": "add_edge",
                "src": "strat_bad_leaderboard",
                "dst": "shortest_path_grid_apply",
                "relation": "leveraged",
                "metadata": {"source_session": "sess_test"},
                "tier": "add",
            },
        ]

        patches = validate_patches(
            patches_from_graph_edits(
                edits,
                graph=graph,
                question="Design a real-time leaderboard with rank pagination and score updates.",
            ),
            graph,
        )

        self.assertEqual(VALIDATION_NEEDS_REVIEW, patches[0].validation.status)
        self.assertEqual(VALIDATION_NEEDS_REVIEW, patches[1].validation.status)
        self.assertIn("inherits needs_review", " ".join(patches[1].validation.reasons))

    def test_claim_with_unsupported_capability_slots_needs_review(self) -> None:
        graph = _graph()
        edits = [{
            "op": "add_node",
            "node_id": "claim_fenwick_find_kth_pagination",
            "node_type": "claim",
            "text": "A Fenwick tree over score buckets gives O(log n) rank queries and O(log n) find_kth for range pagination.",
            "metadata": {
                "evidence_node_ids": ["fenwick_prefix_update"],
                "source_session": "sess_test",
            },
            "tier": "add",
        }]

        patches = validate_patches(patches_from_graph_edits(edits, graph=graph), graph)

        self.assertEqual(VALIDATION_NEEDS_REVIEW, patches[0].validation.status)
        self.assertIn("find_kth_or_select", " ".join(patches[0].validation.reasons))


if __name__ == "__main__":
    unittest.main()
