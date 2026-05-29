"""Tests for the Phase-2A seed procedures.

Three procedures introduced in Sub-phase 2.7:
  - VerifyNonNegativeEdges       (leaf, no CALL commands)
  - DetectNegativeCycle           (leaf, no CALL commands)
  - VerifyShortestPath            (composer, body emits CALL commands)

Acceptance per sub-phase:
  - Each builds cleanly
  - Each has a populated example_use (passes consolidation gate)
  - State schema fields are sensible
  - Composer's body actually contains CALL commands referencing the
    other two leaves + the Phase-1 procedure
  - Round-trip JSON works for all three (Phase-2A version fields included)
"""
from __future__ import annotations

import json
import re
import unittest

from reasoning.procedures.detect_negative_cycle import build_detect_negative_cycle
from reasoning.procedures.verify_algorithm_preconditions import build_seed_procedure
from reasoning.procedures.verify_nonneg_edges import build_verify_nonneg_edges
from reasoning.procedures.verify_shortest_path import build_verify_shortest_path
from reasoning.schemas import ProcedureNode


class TestLeafProcedures(unittest.TestCase):
    def test_verify_nonneg_edges_builds(self):
        proc = build_verify_nonneg_edges()
        self.assertEqual(proc.name, "VerifyNonNegativeEdges")
        self.assertEqual(proc.node_type, "procedure")
        self.assertIsNotNone(proc.example_use)
        self.assertGreater(len(proc.provenance.validating_examples), 0)
        # Schema: violating_edges + checked_edges
        for field in ("violating_edges", "checked_edges"):
            self.assertIn(field, proc.state_schema)
        # No sub-procedures: body must NOT contain a CALL command
        self.assertNotIn("CALL ", proc.body)

    def test_detect_negative_cycle_builds(self):
        proc = build_detect_negative_cycle()
        self.assertEqual(proc.name, "DetectNegativeCycle")
        self.assertIsNotNone(proc.example_use)
        for field in ("detected_cycles", "checked_paths"):
            self.assertIn(field, proc.state_schema)
        self.assertNotIn("CALL ", proc.body)

    def test_leaf_example_uses_have_final_output(self):
        for proc in [build_verify_nonneg_edges(), build_detect_negative_cycle()]:
            ex = proc.example_use
            self.assertIn("final_output", ex, f"{proc.name} example_use missing final_output")
            self.assertIn("final_state", ex, f"{proc.name} example_use missing final_state")
            self.assertIn("inputs", ex, f"{proc.name} example_use missing inputs")


class TestComposer(unittest.TestCase):
    def test_verify_shortest_path_builds(self):
        proc = build_verify_shortest_path()
        self.assertEqual(proc.name, "VerifyShortestPath")
        self.assertIsNotNone(proc.example_use)
        # Composer's state must capture aggregated verdict info
        for field in ("verdict", "safe_to_apply", "recommended_alternative", "sub_results_summary"):
            self.assertIn(field, proc.state_schema)

    def test_composer_body_contains_call_commands(self):
        """The body must instruct the sub-LLM to emit CALL commands for
        the three sub-procedures it depends on, and must forbid synthesis
        of sub-procedure results in the composer's own state."""
        proc = build_verify_shortest_path()
        body = proc.body
        # Each of the three child names must appear in a CALL line
        self.assertRegex(body, r"CALL\s+VerifyAlgorithmPreconditions")
        self.assertRegex(body, r"CALL\s+VerifyNonNegativeEdges")
        self.assertRegex(body, r"CALL\s+DetectNegativeCycle")
        # Composer still sets its OWN state (verdict / safe_to_apply / alternative)
        self.assertIn("SET state.safe_to_apply", body)
        self.assertIn("SET state.verdict", body)
        self.assertIn("DONE", body)
        # Hardening against the real-LLM regression (2026-05-20 production smoke):
        # composer must explicitly forbid synthesizing sub-procedure results
        # into its own state — that was the failure mode that produced a
        # composer with state but no actual sub_invocation_of edges.
        self.assertIn("MUST emit at least ONE", body,
                      "Composer body must REQUIRE at least one CALL emission")
        self.assertIn("DO NOT write to state.sub_results_summary", body,
                      "Composer body must forbid synthesizing sub-results itself")

    def test_composer_declares_depends_on(self):
        proc = build_verify_shortest_path()
        deps = proc.provenance.depends_on
        # All three referenced procedure IDs are declared in depends_on
        self.assertIn("proc_verify_algorithm_preconditions_v1", deps)
        self.assertIn("proc_verify_nonneg_edges_v1", deps)
        self.assertIn("proc_detect_negative_cycle_v1", deps)

    def test_composer_example_use_dijkstra_unsafe(self):
        """The example_use must capture the canonical Dijkstra-with-neg-edge
        case so consolidation has a worked instance to validate against."""
        proc = build_verify_shortest_path()
        ex = proc.example_use
        self.assertEqual(ex["inputs"]["algorithm_name"], "Dijkstra")
        self.assertFalse(ex["final_state"]["safe_to_apply"])
        self.assertEqual(ex["final_output"]["recommended_alternative"], "Bellman-Ford")


class TestRoundTripWithVersionFields(unittest.TestCase):
    """All three Phase-2A seed procedures must round-trip through JSON
    cleanly, including the new version-chain fields."""

    def _roundtrip(self, proc: ProcedureNode) -> ProcedureNode:
        return ProcedureNode.from_dict(json.loads(json.dumps(proc.to_dict())))

    def test_all_three_roundtrip(self):
        for proc in (
            build_verify_nonneg_edges(),
            build_detect_negative_cycle(),
            build_verify_shortest_path(),
        ):
            restored = self._roundtrip(proc)
            self.assertEqual(restored.to_dict(), proc.to_dict(),
                             f"{proc.name} did not round-trip cleanly")
            # version-chain defaults: every Phase-2A seed is v1 with no chain links
            self.assertEqual(restored.version, 1)
            self.assertIsNone(restored.parent_version_id)
            self.assertIsNone(restored.superseded_by_id)


class TestPhase1AndPhase2ProceduresInteroperate(unittest.TestCase):
    """The composer depends_on the Phase-1 seed. Both must coexist in one
    dispatcher index without collisions."""

    def test_all_four_share_one_dispatcher_index(self):
        from reasoning.dispatcher import Dispatcher
        procs = [
            build_seed_procedure(),                  # Phase 1
            build_verify_nonneg_edges(),             # Phase 2A leaf
            build_detect_negative_cycle(),           # Phase 2A leaf
            build_verify_shortest_path(),            # Phase 2A composer
        ]
        disp = Dispatcher({p.id: p for p in procs})
        # Each is resolvable by name
        for proc in procs:
            self.assertIsNotNone(disp.resolve_name(proc.name),
                                 f"{proc.name} not resolvable")
            self.assertEqual(disp.resolve_name(proc.name).id, proc.id)


if __name__ == "__main__":
    unittest.main()
