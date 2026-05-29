from __future__ import annotations

import unittest

from answerer_v4 import extract_model_patches
from graph_core import MemoryGraph, Node
from reasoning.scoped_edits import (
    VALIDATION_ACCEPT,
    VALIDATION_REJECT,
    patches_from_graph_edits,
    validate_patches,
)


def _graph() -> MemoryGraph:
    return MemoryGraph(
        nodes={
            "claim_dijkstra_negative_edge_invalid": Node(
                id="claim_dijkstra_negative_edge_invalid",
                node_type="claim",
                confidence=0.95,
                text="Dijkstra is invalid when negative edges exist.",
            ),
            "strategy_algorithm_applicability": Node(
                id="strategy_algorithm_applicability",
                node_type="strategy",
                confidence=0.9,
                text="Strategy for evaluating algorithm applicability.",
            ),
        },
        edges=[],
    )


class TestExtractModelPatches(unittest.TestCase):
    def test_epistemic_state_patch_emitted_and_validated(self):
        graph = _graph()
        cot_log = [
            """<patch>
{"op": "add_node", "node_type": "epistemic_state",
 "target_node_id": "claim_dijkstra_negative_edge_invalid",
 "confidence": 0.94,
 "open_questions": ["DAG variant?"],
 "known_risks": ["DAG shortest path may confuse"]}
</patch>"""
        ]
        edits = extract_model_patches(
            cot_log,
            session_id="sess_a",
            graph_nodes=set(graph.nodes.keys()),
        )
        ops = [(e["op"], e.get("node_type") or e.get("relation")) for e in edits]
        self.assertIn(("add_node", "epistemic_state"), ops)
        self.assertIn(("add_edge", "epistemic_of"), ops)
        epi = next(e for e in edits if e.get("node_type") == "epistemic_state")
        self.assertEqual(
            epi["metadata"]["target_node_id"],
            "claim_dijkstra_negative_edge_invalid",
        )
        self.assertIn(
            "claim_dijkstra_negative_edge_invalid",
            epi["metadata"]["evidence_node_ids"],
        )
        self.assertTrue(epi["metadata"]["model_emitted_patch"])

        patches = validate_patches(
            patches_from_graph_edits(
                edits,
                graph=graph,
                learning_report={"question": "Can Dijkstra handle negative edges?"},
                question="Can Dijkstra handle negative edges?",
                task_frame=None,
            ),
            graph,
        )
        epi_patch = next(p for p in patches if p.patch_type == "add_epistemic_state")
        self.assertEqual(epi_patch.validation.status, VALIDATION_ACCEPT)

    def test_v5_edge_relations_extracted(self):
        graph = _graph()
        cot_log = [
            """<patch>
{"op": "add_edge", "src": "strategy_algorithm_applicability",
 "dst": "claim_dijkstra_negative_edge_invalid", "relation": "invalidated_by"}
</patch>
<patch>
{"op": "add_edge", "src": "strategy_algorithm_applicability",
 "dst": "claim_dijkstra_negative_edge_invalid", "relation": "requires_slot"}
</patch>
<patch>
{"op": "add_edge", "src": "strategy_algorithm_applicability",
 "dst": "claim_dijkstra_negative_edge_invalid", "relation": "transfers_to"}
</patch>"""
        ]
        edits = extract_model_patches(
            cot_log,
            session_id="sess_b",
            graph_nodes=set(graph.nodes.keys()),
        )
        relations = {e["relation"] for e in edits if e["op"] == "add_edge"}
        self.assertEqual(
            relations,
            {"invalidated_by", "requires_slot", "transfers_to"},
        )

    def test_malformed_patch_is_dropped_silently(self):
        graph = _graph()
        cot_log = [
            "<patch>{not valid json}</patch>",
            '<patch>{"op": "add_node", "node_type": "unknown_type"}</patch>',
            '<patch>{"op": "add_edge", "src": "x", "dst": "y", "relation": "fake_relation"}</patch>',
        ]
        edits = extract_model_patches(
            cot_log,
            session_id="sess_c",
            graph_nodes=set(graph.nodes.keys()),
        )
        self.assertEqual(edits, [])

    def test_spec_aliases_target_and_id_accepted(self):
        graph = _graph()
        cot_log = [
            """<patch>
{"op": "add_node", "id": "epi_dijkstra_negative_edge_001",
 "node_type": "epistemic_state",
 "target": "claim_dijkstra_negative_edge_invalid",
 "status": "verified", "confidence": 0.94,
 "support_level": "mechanistic + textbook fact",
 "known_risks": ["DAG shortest path may confuse"],
 "invalidators": ["question is about DAG-specific shortest path"],
 "last_verified_by": ["fact_dijkstra_nonnegative"]}
</patch>"""
        ]
        edits = extract_model_patches(
            cot_log,
            session_id="sess_spec",
            graph_nodes=set(graph.nodes.keys()),
        )
        node_edits = [e for e in edits if e["op"] == "add_node"]
        edge_edits = [e for e in edits if e["op"] == "add_edge"]
        self.assertEqual(len(node_edits), 1)
        self.assertEqual(node_edits[0]["node_id"], "epi_dijkstra_negative_edge_001")
        meta = node_edits[0]["metadata"]
        self.assertEqual(meta["target_node_id"], "claim_dijkstra_negative_edge_invalid")
        self.assertEqual(meta["status"], "verified")
        self.assertEqual(meta["support_level"], "mechanistic + textbook fact")
        self.assertEqual(meta["invalidators"], ["question is about DAG-specific shortest path"])
        self.assertEqual(meta["last_verified_by"], ["fact_dijkstra_nonnegative"])
        self.assertEqual(len(edge_edits), 1)
        self.assertEqual(edge_edits[0]["relation"], "epistemic_of")

    def test_sibling_batch_condition_node_plus_invalidated_by_edge(self):
        graph = _graph()
        cot_log = [
            """<patch>
{"op": "add_node", "node_id": "cond_dag_special",
 "node_type": "claim",
 "text": "Question is about DAG-specific shortest path, not general Dijkstra."}
</patch>
<patch>
{"op": "add_edge", "src": "strategy_algorithm_applicability",
 "dst": "cond_dag_special", "relation": "invalidated_by"}
</patch>"""
        ]
        edits = extract_model_patches(
            cot_log,
            session_id="sess_sib",
            graph_nodes=set(graph.nodes.keys()),
        )
        from reasoning.scoped_edits import patches_from_graph_edits, validate_patches
        patches = validate_patches(
            patches_from_graph_edits(
                edits,
                graph=graph,
                learning_report={"question": "test"},
                question="test",
                task_frame=None,
            ),
            graph,
        )
        relation_patches = [p for p in patches if p.patch_type == "add_relation"]
        self.assertEqual(len(relation_patches), 1)
        self.assertNotIn(
            "edge endpoint missing from graph/pre-edit batch: cond_dag_special",
            "; ".join(relation_patches[0].validation.reasons),
        )

    def test_no_target_node_id_does_not_attach_edge_or_evidence(self):
        graph = _graph()
        cot_log = [
            """<patch>
{"op": "add_node", "node_type": "epistemic_state",
 "confidence": 0.7, "open_questions": ["unknown target"]}
</patch>"""
        ]
        edits = extract_model_patches(
            cot_log,
            session_id="sess_d",
            graph_nodes=set(graph.nodes.keys()),
        )
        node_edits = [e for e in edits if e["op"] == "add_node"]
        edge_edits = [e for e in edits if e["op"] == "add_edge"]
        self.assertEqual(len(node_edits), 1)
        self.assertEqual(edge_edits, [])
        self.assertEqual(node_edits[0]["metadata"]["evidence_node_ids"], [])


if __name__ == "__main__":
    unittest.main()
