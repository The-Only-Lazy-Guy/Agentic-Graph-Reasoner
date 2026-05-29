"""Round-trip JSON tests for reasoning/schemas.py.

Acceptance criterion for Sub-phase 1.1: every dataclass can serialize to
a JSON string and reconstruct itself from that JSON without information
loss.

Run with:
    python -m unittest graph_final/reasoning/tests/test_schemas.py
or via pytest if installed.
"""
from __future__ import annotations

import json
import unittest

from reasoning.schemas import (
    AuditEntry,
    ControlRuleNode,
    FailurePatternNode,
    ProcedureNode,
    Provenance,
    ReasoningAtomNode,
    SignatureFamilyNode,
    SignatureVariantNode,
    SessionEdge,
    SessionObjectNode,
    SessionSubgraph,
    SolvedSubgoalNode,
    StrategyNode,
)


def _roundtrip(obj, cls):
    """Serialize obj -> JSON -> dict -> cls instance. Returns the
    reconstructed object and asserts to_dict matches before/after."""
    payload = json.dumps(obj.to_dict(), ensure_ascii=False, sort_keys=True)
    restored = cls.from_dict(json.loads(payload))
    return restored, payload


class TestProvenance(unittest.TestCase):
    def test_minimal(self):
        p = Provenance(created_in_session_id="sess_abc")
        restored, _ = _roundtrip(p, Provenance)
        self.assertEqual(p.to_dict(), restored.to_dict())

    def test_full(self):
        p = Provenance(
            created_in_session_id="sess_xyz",
            validating_examples=["sess_1", "sess_2"],
            depends_on=["proc_A", "fact_B"],
            citation_count=7,
            citation_decay=0.85,
            last_modified="2026-05-20T10:00:00+00:00",
            deprecated=True,
            deprecation_reason="superseded by v2",
        )
        restored, _ = _roundtrip(p, Provenance)
        self.assertEqual(p.to_dict(), restored.to_dict())
        self.assertEqual(restored.citation_count, 7)
        self.assertEqual(restored.deprecation_reason, "superseded by v2")


class TestProcedureNode(unittest.TestCase):
    def test_roundtrip_with_example_use(self):
        p = ProcedureNode(
            id="proc_001",
            name="VerifyAlgorithmPreconditions",
            purpose="Check stated preconditions of a named algorithm.",
            when_to_use="When question is about algorithm applicability.",
            signature={
                "inputs": [{"name": "algorithm_name", "type": "str"}],
                "outputs": [{"name": "ok", "type": "bool"}],
            },
            state_schema={"checked": "list[str]", "violated": "list[str]"},
            body="Check preconditions of {algorithm_name}.",
            example_use={
                "session_id": "seed",
                "inputs": {"algorithm_name": "Dijkstra"},
                "final_output": {"ok": False},
            },
            provenance=Provenance(created_in_session_id="seed"),
        )
        restored, _ = _roundtrip(p, ProcedureNode)
        self.assertEqual(p.to_dict(), restored.to_dict())
        self.assertEqual(restored.example_use["final_output"]["ok"], False)

    def test_roundtrip_without_example_use(self):
        p = ProcedureNode(
            id="proc_002",
            name="MinimalProc",
            purpose="...",
            when_to_use="...",
            signature={"inputs": [], "outputs": []},
            state_schema={},
            body="do nothing",
            example_use=None,
            provenance=Provenance(created_in_session_id="seed"),
        )
        restored, _ = _roundtrip(p, ProcedureNode)
        self.assertEqual(p.to_dict(), restored.to_dict())
        self.assertIsNone(restored.example_use)

    def test_default_version_fields(self):
        """A freshly-built procedure must default to version=1 with no
        parent/successor links — this is what every Phase-1 procedure should
        report after the schema migration."""
        p = ProcedureNode(
            id="proc_default",
            name="Foo",
            purpose="...",
            when_to_use="...",
            signature={},
            state_schema={},
            body="",
            example_use=None,
            provenance=Provenance(created_in_session_id="seed"),
        )
        self.assertEqual(p.version, 1)
        self.assertIsNone(p.parent_version_id)
        self.assertIsNone(p.superseded_by_id)

    def test_roundtrip_with_version_chain(self):
        """A non-head version of a procedure with explicit parent + successor
        round-trips through JSON without loss."""
        p = ProcedureNode(
            id="proc_001_v1",
            name="VerifyShortestPath",
            purpose="...",
            when_to_use="...",
            signature={},
            state_schema={},
            body="",
            example_use=None,
            provenance=Provenance(created_in_session_id="sess_a"),
            version=1,
            parent_version_id=None,
            superseded_by_id="proc_001_v2",
        )
        restored, _ = _roundtrip(p, ProcedureNode)
        self.assertEqual(restored.version, 1)
        self.assertIsNone(restored.parent_version_id)
        self.assertEqual(restored.superseded_by_id, "proc_001_v2")


class TestStrategyNode(unittest.TestCase):
    def test_roundtrip_with_v2_strategy_fields(self):
        node = StrategyNode(
            id="strat_1",
            question_pattern="How does Dijkstra work?",
            domain_keywords=["dijkstra", "priority", "queue"],
            plan_template=["Read explanation node", "Compose answer"],
            key_node_ids=["dijkstra_greedy_relaxation_explanation"],
            key_node_rationales={"dijkstra_greedy_relaxation_explanation": "cited 2x"},
            workspace_schema=[],
            pitfalls=[{"approach": "negative-edge shortcut", "condition": "wrong question mode", "mechanism": "answer reuse too narrow"}],
            effective_queries=["dijkstra priority queue relax"],
            session_stats={"steps": 2, "tool_calls": 1},
            provenance=Provenance(created_in_session_id="seed"),
            task_family="direct_judgment",
            task_subtype="algorithm_mechanism_explanation",
            question_mode="mechanism_explanation",
            entry_conditions={"algorithm": "Dijkstra"},
            slot_order=["mechanism", "answer"],
            checkpoint_plan=["Read mechanism", "Compose fresh answer"],
            stop_conditions=["Fill slots: mechanism, answer"],
            forbidden_finalize_conditions=["negative edge applicability only"],
            strategy_schema_version=2,
        )
        restored, _ = _roundtrip(node, StrategyNode)
        self.assertEqual(node.to_dict(), restored.to_dict())

        p2 = ProcedureNode(
            id="proc_001_v2",
            name="VerifyShortestPath",
            purpose="refined",
            when_to_use="...",
            signature={},
            state_schema={},
            body="",
            example_use=None,
            provenance=Provenance(created_in_session_id="sess_b"),
            version=2,
            parent_version_id="proc_001_v1",
            superseded_by_id=None,
        )
        restored2, _ = _roundtrip(p2, ProcedureNode)
        self.assertEqual(restored2.version, 2)
        self.assertEqual(restored2.parent_version_id, "proc_001_v1")
        self.assertIsNone(restored2.superseded_by_id)

    def test_backward_compat_loads_legacy_serialized_form(self):
        """Phase-1 procedures were serialized WITHOUT the new version fields.
        Loading such a dict must still produce a valid ProcedureNode with
        sensible defaults — otherwise Phase 1 sessions on disk become
        un-replayable.
        """
        legacy_dict = {
            "id": "proc_legacy",
            "name": "OldProc",
            "purpose": "...",
            "when_to_use": "...",
            "signature": {},
            "state_schema": {},
            "body": "",
            "example_use": None,
            "provenance": Provenance(created_in_session_id="seed").to_dict(),
            "node_type": "procedure",
            # NO version / parent_version_id / superseded_by_id keys
        }
        restored = ProcedureNode.from_dict(legacy_dict)
        self.assertEqual(restored.version, 1)
        self.assertIsNone(restored.parent_version_id)
        self.assertIsNone(restored.superseded_by_id)
        self.assertEqual(restored.name, "OldProc")


class TestFailurePatternNode(unittest.TestCase):
    def test_roundtrip(self):
        f = FailurePatternNode(
            id="fp_001",
            name="GreedySelectionFailsOnNegativeEdges",
            attempted_approach="Greedy edge selection ordered by weight ascending",
            failure_condition="Graph contains at least one edge with negative weight",
            failure_mechanism="Greedy commits to a path before a later negative edge can offer a shorter alternative",
            replacement="proc_bellman_ford",
            example_failure_case={
                "session_id": "seed",
                "observed_failure": "Returned 2 instead of -1 for s->b",
            },
            provenance=Provenance(created_in_session_id="seed"),
        )
        restored, _ = _roundtrip(f, FailurePatternNode)
        self.assertEqual(f.to_dict(), restored.to_dict())
        self.assertEqual(restored.replacement, "proc_bellman_ford")


class TestSignatureNodes(unittest.TestCase):
    def test_signature_family_roundtrip(self):
        node = SignatureFamilyNode(
            id="sigfam_strategy.algorithm_applicability.verdict_reason",
            semantic_type="strategy",
            task_family="algorithm_applicability",
            family_label="strategy:algorithm_applicability:verdict+reason",
            variant_ids=["sigvar_strategy_123"],
            provenance=Provenance(created_in_session_id="seed"),
            contested=True,
            dominant_variant_id=None,
            retrieval_tier="gated",
            support_score=0.5,
            stability_score=1.2,
            risk_score=0.3,
            contradiction_score=0.0,
            bias_score=0.7,
        )
        restored, _ = _roundtrip(node, SignatureFamilyNode)
        self.assertEqual(node.to_dict(), restored.to_dict())

    def test_signature_variant_roundtrip(self):
        node = SignatureVariantNode(
            id="sigvar_strategy_123",
            family_id="sigfam_strategy.algorithm_applicability.verdict_reason",
            semantic_type="strategy",
            semantic_node_id="strat_demo",
            canonical_text="Read failure node then answer.",
            task_family="algorithm_applicability",
            epistemic_status="provisional",
            promotion_state="blocked",
            retrieval_tier="gated",
            provenance=Provenance(created_in_session_id="seed"),
            support_score=0.1,
            stability_score=0.6,
            risk_score=0.2,
            contradiction_score=0.0,
            bias_score=0.25,
            required_slots=["verdict", "reason"],
            support_node_ids=["dijkstra_neg_edges_failure"],
            aliases=["Use negative-edge failure explanation first."],
        )
        restored, _ = _roundtrip(node, SignatureVariantNode)
        self.assertEqual(node.to_dict(), restored.to_dict())

    def test_roundtrip_no_replacement(self):
        f = FailurePatternNode(
            id="fp_002",
            name="UnknownFailure",
            attempted_approach="...",
            failure_condition="...",
            failure_mechanism="...",
            replacement=None,
            example_failure_case=None,
            provenance=Provenance(created_in_session_id="seed"),
        )
        restored, _ = _roundtrip(f, FailurePatternNode)
        self.assertEqual(f.to_dict(), restored.to_dict())
        self.assertIsNone(restored.replacement)
        self.assertIsNone(restored.example_failure_case)


class TestSessionObjectNode(unittest.TestCase):
    def test_roundtrip(self):
        s = SessionObjectNode(
            id="so_001",
            procedure_id="proc_verify_preconditions",
            name="VerifyAlgorithmPreconditions",
            state={
                "preconditions_checked": ["nonneg_edges", "single_source"],
                "preconditions_violated": ["nonneg_edges"],
                "evidence": {"nonneg_edges": "Edge b->c has weight -1"},
            },
            created_step=2,
            provenance=Provenance(created_in_session_id="sess_xyz"),
        )
        restored, _ = _roundtrip(s, SessionObjectNode)
        self.assertEqual(s.to_dict(), restored.to_dict())
        self.assertEqual(restored.state["preconditions_violated"], ["nonneg_edges"])
        self.assertEqual(restored.state["evidence"]["nonneg_edges"], "Edge b->c has weight -1")


class TestSolvedSubgoalNode(unittest.TestCase):
    def test_roundtrip(self):
        node = SolvedSubgoalNode(
            id="ssg_dijkstra_negative",
            summary="Standard Dijkstra is not generally correct with negative edge weights.",
            subgoal_signature="shortest_path.dijkstra.negative_edges.validity",
            question_type="algorithm_applicability",
            input_conditions={"algorithm": "Dijkstra", "graph_property": "negative_edge_weights"},
            output_slots={
                "verdict": "not guaranteed",
                "reason": "negative edge can improve a finalized node later",
                "alternative": "Bellman-Ford",
            },
            valid_when=["standard dijkstra", "general correctness"],
            invalid_when=["specific graph instance", "modified dijkstra variant"],
            supporting_node_ids=[
                "dijkstra_requires_nonnegative_edge_weights",
                "negative_edge_counterexample_test_apply",
            ],
            confidence=0.98,
            source_sessions=["sess_a", "sess_b"],
            provenance=Provenance(created_in_session_id="seed"),
        )
        restored, _ = _roundtrip(node, SolvedSubgoalNode)
        self.assertEqual(node.to_dict(), restored.to_dict())
        self.assertEqual(restored.output_slots["alternative"], "Bellman-Ford")


class TestReasoningAtomNode(unittest.TestCase):
    def test_roundtrip(self):
        node = ReasoningAtomNode(
            id="atom_dijkstra_negative_invariant_break",
            atom_type="invariant_break",
            claim="Negative edges can break Dijkstra's finalized-distance invariant.",
            reusable_for=["algorithm_applicability", "correctness_explanation"],
            dependencies=["dijkstra_greedy_finalization", "negative_edge_can_reduce_distance"],
            supporting_node_ids=["negative_edge_counterexample_test_apply"],
            confidence=0.95,
            provenance=Provenance(created_in_session_id="seed"),
        )
        restored, _ = _roundtrip(node, ReasoningAtomNode)
        self.assertEqual(node.to_dict(), restored.to_dict())
        self.assertIn("algorithm_applicability", restored.reusable_for)


class TestControlRuleNode(unittest.TestCase):
    def test_roundtrip(self):
        node = ControlRuleNode(
            id="ctrl_algorithm_applicability",
            task_family="algorithm_applicability",
            guidance="Answer with verdict, condition, reason, caveat, and alternative before escalating.",
            required_slots=["verdict", "reason", "alternative", "caveat"],
            optional_slots=["counterexample", "proof"],
            forbidden_escalations=["DERIVE"],
            preferred_action_order=["REUSE", "QUERY", "FINALIZE"],
            stop_condition="All required slots filled.",
            provenance=Provenance(created_in_session_id="seed"),
        )
        restored, _ = _roundtrip(node, ControlRuleNode)
        self.assertEqual(node.to_dict(), restored.to_dict())
        self.assertEqual(restored.preferred_action_order[0], "REUSE")


class TestAuditEntry(unittest.TestCase):
    def test_roundtrip_create(self):
        a = AuditEntry(
            session_id="sess_xyz",
            step_index=0,
            object_id="so_001",
            operation="create",
            field_path="",
            old_value=None,
            new_value={"preconditions_checked": [], "violated": []},
            triggered_by_text="I'll create a new VerifyAlgorithmPreconditions object",
            timestamp="2026-05-20T11:00:00+00:00",
        )
        restored, _ = _roundtrip(a, AuditEntry)
        self.assertEqual(a.to_dict(), restored.to_dict())
        self.assertEqual(restored.operation, "create")

    def test_roundtrip_update(self):
        a = AuditEntry(
            session_id="sess_xyz",
            step_index=2,
            object_id="so_001",
            operation="update",
            field_path="state.preconditions_violated",
            old_value=[],
            new_value=["nonneg_edges"],
            triggered_by_text="add nonneg_edges to preconditions_violated",
            timestamp="2026-05-20T11:01:00+00:00",
        )
        restored, _ = _roundtrip(a, AuditEntry)
        self.assertEqual(a.to_dict(), restored.to_dict())
        self.assertEqual(restored.field_path, "state.preconditions_violated")
        self.assertEqual(restored.new_value, ["nonneg_edges"])

    def test_roundtrip_delete(self):
        a = AuditEntry(
            session_id="sess_xyz",
            step_index=4,
            object_id="so_001",
            operation="delete",
            field_path="state.evidence",
            old_value={"nonneg_edges": "weight -1"},
            new_value=None,
            triggered_by_text="clear the evidence map",
            timestamp="2026-05-20T11:02:00+00:00",
        )
        restored, _ = _roundtrip(a, AuditEntry)
        self.assertEqual(a.to_dict(), restored.to_dict())
        self.assertEqual(restored.operation, "delete")
        self.assertIsNone(restored.new_value)


class TestSessionEdge(unittest.TestCase):
    def test_roundtrip(self):
        e = SessionEdge(
            src="so_001",
            dst="proc_bellman_ford",
            relation="recommended_alternative",
            metadata={"confidence": 0.92},
        )
        restored, _ = _roundtrip(e, SessionEdge)
        self.assertEqual(e.to_dict(), restored.to_dict())


class TestSessionSubgraph(unittest.TestCase):
    def test_roundtrip_empty(self):
        sg = SessionSubgraph(
            session_id="sess_empty",
            query="test query",
            graph_id="cs4",
            started_at="2026-05-20T11:00:00+00:00",
        )
        restored, _ = _roundtrip(sg, SessionSubgraph)
        self.assertEqual(sg.to_dict(), restored.to_dict())
        self.assertEqual(restored.step_count, 0)
        self.assertEqual(restored.edges, [])
        self.assertEqual(restored.audit_log, [])

    def test_roundtrip_populated(self):
        prov = Provenance(created_in_session_id="sess_xyz")
        obj = SessionObjectNode(
            id="so_001",
            procedure_id="proc_verify",
            name="VerifyAlgorithmPreconditions",
            state={"preconditions_checked": ["nonneg_edges"]},
            created_step=1,
            provenance=prov,
        )
        edge = SessionEdge(src="so_001", dst="proc_bellman_ford", relation="recommends")
        audit = AuditEntry(
            session_id="sess_xyz",
            step_index=1,
            object_id="so_001",
            operation="create",
            field_path="",
            old_value=None,
            new_value=obj.state,
            triggered_by_text="create",
            timestamp="2026-05-20T11:00:00+00:00",
        )
        sg = SessionSubgraph(
            session_id="sess_xyz",
            query="Can Dijkstra handle negative edges?",
            graph_id="cs4",
            nodes={"so_001": obj.to_dict()},
            edges=[edge],
            audit_log=[audit],
            step_count=3,
            started_at="2026-05-20T11:00:00+00:00",
            ended_at="2026-05-20T11:05:00+00:00",
        )
        restored, _ = _roundtrip(sg, SessionSubgraph)
        self.assertEqual(sg.to_dict(), restored.to_dict())
        self.assertEqual(len(restored.edges), 1)
        self.assertEqual(len(restored.audit_log), 1)
        self.assertEqual(restored.nodes["so_001"]["procedure_id"], "proc_verify")
        self.assertEqual(restored.ended_at, "2026-05-20T11:05:00+00:00")


if __name__ == "__main__":
    unittest.main()
