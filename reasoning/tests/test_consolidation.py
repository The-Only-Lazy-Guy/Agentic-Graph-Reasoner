"""Tests for reasoning/consolidation.py + seed procedure load.

Three concerns:
  1. The seed procedure constructs cleanly and has a validating example.
  2. Consolidator correctly applies the three gates (citations, validating
     example, deps consolidated) and emits the right decision.
  3. session_object nodes never promote — they're per-session by design.
"""
from __future__ import annotations

import unittest

from reasoning.consolidation import Consolidator
from reasoning.procedures.verify_algorithm_preconditions import build_seed_procedure
from reasoning.schemas import (
    FailurePatternNode,
    Provenance,
    SessionEdge,
    SessionObjectNode,
    SessionSubgraph,
)


def _make_session(nodes_list):
    sg = SessionSubgraph(
        session_id="sess_test",
        query="test",
        graph_id="cs4",
    )
    for n in nodes_list:
        sg.nodes[n["id"]] = n
    return sg


class TestSeedProcedure(unittest.TestCase):
    def test_builds_cleanly(self):
        proc = build_seed_procedure()
        self.assertEqual(proc.name, "VerifyAlgorithmPreconditions")
        self.assertEqual(proc.node_type, "procedure")
        self.assertIsNotNone(proc.example_use)
        self.assertGreater(len(proc.provenance.validating_examples), 0,
                           "Seed procedure must have at least one validating example")

    def test_state_schema_has_expected_fields(self):
        proc = build_seed_procedure()
        for field_name in (
            "preconditions_checked",
            "preconditions_violated",
            "preconditions_deferred",
            "evidence_for_violations",
        ):
            self.assertIn(field_name, proc.state_schema)

    def test_example_use_contains_correct_dijkstra_case(self):
        proc = build_seed_procedure()
        ex = proc.example_use
        self.assertEqual(ex["inputs"]["algorithm_name"], "Dijkstra")
        self.assertIn("nonneg", str(ex["final_state"]["preconditions_violated"]).lower())
        self.assertEqual(ex["final_output"]["recommended_alternative"], "Bellman-Ford")

    def test_serialization_roundtrip(self):
        import json
        from reasoning.schemas import ProcedureNode
        proc = build_seed_procedure()
        restored = ProcedureNode.from_dict(json.loads(json.dumps(proc.to_dict())))
        self.assertEqual(restored.to_dict(), proc.to_dict())


class TestConsolidationGates(unittest.TestCase):
    def _proc_node(self, node_id, has_example=True, citation_count=0, depends_on=None):
        return {
            "id": node_id,
            "name": node_id,
            "purpose": "",
            "when_to_use": "",
            "signature": {},
            "state_schema": {},
            "body": "",
            "example_use": {"x": 1} if has_example else None,
            "node_type": "procedure",
            "provenance": Provenance(
                created_in_session_id="sess_test",
                validating_examples=["sess_seed"] if has_example else [],
                depends_on=depends_on or [],
                citation_count=citation_count,
            ).to_dict(),
        }

    def test_node_promotes_when_all_gates_pass(self):
        # 3 prior citations + 1 current = 4 >= threshold of 3
        # validating example present
        # no deps
        sg = _make_session([self._proc_node("p1", has_example=True)])
        cons = Consolidator(promotion_threshold=3)
        decisions = cons.consolidate(sg, prior_citation_counts={"p1": 3})
        self.assertEqual(len(decisions), 1)
        d = decisions[0]
        self.assertEqual(d.decision, "promote")
        self.assertEqual(d.gate_results, {
            "citation_threshold": True,
            "validating_example": True,
            "deps_consolidated": True,
        })
        self.assertIsNotNone(d.node_data)

    def test_gate1_fails_keeps_in_pool(self):
        # only 1 session so far, threshold 3 → gate 1 fails
        sg = _make_session([self._proc_node("p1", has_example=True)])
        cons = Consolidator(promotion_threshold=3)
        decisions = cons.consolidate(sg, prior_citation_counts={})
        self.assertEqual(decisions[0].decision, "keep_in_pool")
        self.assertFalse(decisions[0].gate_results["citation_threshold"])
        self.assertTrue(decisions[0].gate_results["validating_example"])

    def test_gate2_fails_no_example(self):
        sg = _make_session([self._proc_node("p1", has_example=False)])
        cons = Consolidator(promotion_threshold=1)
        decisions = cons.consolidate(sg, prior_citation_counts={})
        self.assertEqual(decisions[0].decision, "keep_in_pool")
        self.assertTrue(decisions[0].gate_results["citation_threshold"])
        self.assertFalse(decisions[0].gate_results["validating_example"])

    def test_gate3_fails_missing_deps(self):
        sg = _make_session([
            self._proc_node("p1", has_example=True, depends_on=["p_missing"]),
        ])
        cons = Consolidator(promotion_threshold=1, consolidated_node_ids=set())
        decisions = cons.consolidate(sg, prior_citation_counts={})
        self.assertEqual(decisions[0].decision, "keep_in_pool")
        self.assertFalse(decisions[0].gate_results["deps_consolidated"])

    def test_gate3_passes_when_deps_consolidated(self):
        sg = _make_session([
            self._proc_node("p1", has_example=True, depends_on=["p_dep"]),
        ])
        cons = Consolidator(promotion_threshold=1, consolidated_node_ids={"p_dep"})
        decisions = cons.consolidate(sg, prior_citation_counts={})
        self.assertEqual(decisions[0].decision, "promote")

    def test_session_object_never_promotes(self):
        node = SessionObjectNode(
            id="so_001",
            procedure_id="proc_p1",
            name="P1",
            state={},
            created_step=0,
            provenance=Provenance(
                created_in_session_id="sess_test",
                validating_examples=["sess_seed"],
                citation_count=99,
            ),
        ).to_dict()
        sg = _make_session([node])
        cons = Consolidator(promotion_threshold=1)
        decisions = cons.consolidate(sg, prior_citation_counts={"so_001": 99})
        self.assertEqual(decisions[0].decision, "expire")
        self.assertEqual(decisions[0].node_type, "session_object")

    def test_non_substrate_nodes_ignored(self):
        # A regular fact node should not appear in decisions
        fact = {
            "id": "fact_001",
            "node_type": "fact",
            "text": "some fact",
        }
        sg = _make_session([fact])
        cons = Consolidator()
        decisions = cons.consolidate(sg)
        self.assertEqual(decisions, [])

    def test_failure_pattern_promotion_path(self):
        fp = FailurePatternNode(
            id="fp_001",
            name="GreedyFails",
            attempted_approach="greedy",
            failure_condition="negative edges",
            failure_mechanism="...",
            replacement="proc_bellman",
            example_failure_case={"observed": "wrong answer"},
            provenance=Provenance(
                created_in_session_id="sess_test",
                validating_examples=["sess_seed"],
            ),
        ).to_dict()
        sg = _make_session([fp])
        cons = Consolidator(promotion_threshold=1)
        decisions = cons.consolidate(sg, prior_citation_counts={})
        self.assertEqual(decisions[0].decision, "promote")
        self.assertEqual(decisions[0].node_type, "failure_pattern")


class TestSeedConsolidation(unittest.TestCase):
    """Verify the actual seed procedure can be consolidated when its
    citation threshold is met. End-to-end-ish smoke test."""

    def test_seed_promotes_when_threshold_low(self):
        proc = build_seed_procedure()
        sg = _make_session([proc.to_dict()])
        cons = Consolidator(promotion_threshold=1)
        decisions = cons.consolidate(sg, prior_citation_counts={})
        self.assertEqual(len(decisions), 1)
        d = decisions[0]
        self.assertEqual(d.decision, "promote",
                         f"Seed procedure should promote at threshold=1; "
                         f"gates: {d.gate_results}")
        self.assertIsNotNone(d.node_data)
        self.assertEqual(d.node_data["name"], "VerifyAlgorithmPreconditions")

    def test_seed_keeps_in_pool_at_default_threshold(self):
        proc = build_seed_procedure()
        sg = _make_session([proc.to_dict()])
        cons = Consolidator(promotion_threshold=3)
        decisions = cons.consolidate(sg, prior_citation_counts={})
        # 0 prior + 1 current = 1 < 3 — gate 1 fails
        self.assertEqual(decisions[0].decision, "keep_in_pool")


if __name__ == "__main__":
    unittest.main()
