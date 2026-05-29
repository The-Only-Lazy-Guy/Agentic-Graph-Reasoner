"""Tests for reasoning/substrate_v2.py - Phase 3E-1."""
from __future__ import annotations

import json
import unittest
from pathlib import Path

from reasoning.activation import FrameItem, GraphTaskFrame
from reasoning.schemas import SessionSubgraph
from reasoning.token_estimation import estimate_token_count
from reasoning.substrate_v2 import (
    CheckerRegistry,
    DeltaTransaction,
    FastLoopConfig,
    MissingInfo,
    ReasoningStep,
    SignalNode,
    StateDelta,
    StepContextPacket,
    _sanitize_answer,
    attach_fast_loop_to_session,
    compose_final_answer,
    derive_task_statement_concepts,
    missing_task_statement_concepts,
    packet_from_task_frame,
    parse_step_result,
    project_session_subgraph_to_signals,
    run_fast_step_loop,
    task_concept_constraint,
)
from reasoning.schemas import ProcedureNode
from reasoning.session_subgraph import SessionSubgraphController


ROOT = Path(__file__).resolve().parents[2]


def _load_session(name: str) -> SessionSubgraph:
    path = ROOT / "data" / "session_subgraphs" / name / "subgraph.json"
    return SessionSubgraph.from_dict(json.loads(path.read_text(encoding="utf-8")))


class TestSubstrateV2RoundTrip(unittest.TestCase):
    def test_shared_token_estimator_contract(self):
        self.assertEqual(estimate_token_count(""), 0)
        self.assertEqual(estimate_token_count("a"), 1)
        self.assertEqual(estimate_token_count("abcd"), 1)
        self.assertEqual(estimate_token_count("abcde"), 2)

    def test_signal_node_round_trip(self):
        sig = SignalNode(
            id="sig_1",
            kind="decision",
            text="Use segment tree for online point updates.",
            activation_keys=["segment", "tree", "updates"],
            source_step_id="step_1",
            produced_by="llm_delta",
            state={"status": "supported"},
            evidence_ids=["e1"],
            confidence=0.9,
        )
        restored = SignalNode.from_dict(json.loads(json.dumps(sig.to_dict())))
        self.assertEqual(restored, sig)

    def test_state_delta_to_signal_nodes_tags_regex_fallback(self):
        delta = StateDelta(
            decisions=["Use Bellman-Ford."],
            risks=["Dijkstra is unsafe with negative edges."],
            produced_by="regex_fallback",
            confidence=0.2,
        )
        signals = delta.to_signal_nodes(source_step_id="step_x")
        self.assertEqual({s.produced_by for s in signals}, {"regex_fallback"})
        self.assertEqual({s.kind for s in signals}, {"decision", "risk"})
        self.assertTrue(all(s.confidence == 0.2 for s in signals))

    def test_delta_transaction_round_trip(self):
        txn = DeltaTransaction(
            status="skimmed",
            delta=StateDelta(evidence=["Segment tree has O(log n) updates."], produced_by="regex_fallback"),
            raw_excerpt="answer text",
            parse_error="missing END_STEP_RESULT",
        )
        restored = DeltaTransaction.from_dict(json.loads(json.dumps(txn.to_dict())))
        self.assertEqual(restored.status, "skimmed")
        self.assertEqual(restored.delta.produced_by, "regex_fallback")
        self.assertEqual(restored.parse_error, "missing END_STEP_RESULT")


class TestSessionProjection(unittest.TestCase):
    def test_ioi_synthetic_projects_failed_and_successful_branches(self):
        subgraph = _load_session("sess_phase3d_ioi_synthetic")
        signals = project_session_subgraph_to_signals(subgraph)
        by_source = {sig.source_node_id: sig for sig in signals}

        self.assertEqual(by_source["plan_ioi_kadane_direct"].kind, "repair")
        self.assertIn("Kadane", by_source["plan_ioi_kadane_direct"].text)
        self.assertEqual(by_source["plan_ioi_segment_tree"].kind, "decision")
        self.assertIn("segment tree", by_source["plan_ioi_segment_tree"].text)
        self.assertEqual(by_source["plan_check_ioi_kadane_fails_updates"].kind, "risk")
        self.assertTrue(any(sig.kind == "evidence" and "covered" in sig.text.lower() for sig in signals))

    def test_dijkstra_synthetic_projects_negative_edge_risk_and_bellman_decision(self):
        subgraph = _load_session("sess_phase3d_dijkstra_synthetic")
        signals = project_session_subgraph_to_signals(subgraph)
        by_source = {sig.source_node_id: sig for sig in signals}

        self.assertEqual(by_source["plan_dijkstra_try_dijkstra"].kind, "repair")
        self.assertEqual(by_source["plan_dijkstra_bellman_ford"].kind, "decision")
        self.assertIn("Bellman-Ford", by_source["plan_dijkstra_bellman_ford"].text)
        self.assertEqual(by_source["plan_check_dijkstra_fails_negative_edge"].kind, "risk")
        self.assertIn("negative", by_source["plan_check_dijkstra_fails_negative_edge"].text.lower())

    def test_projection_is_read_only(self):
        subgraph = _load_session("sess_phase3d_ioi_synthetic")
        before = json.dumps(subgraph.to_dict(), sort_keys=True)
        _ = project_session_subgraph_to_signals(subgraph)
        after = json.dumps(subgraph.to_dict(), sort_keys=True)
        self.assertEqual(before, after)


class TestContextPacketAndSteps(unittest.TestCase):
    def test_task_statement_concept_extractor_uses_question_text_only(self):
        concepts = derive_task_statement_concepts(
            "For a directed graph with only nonnegative edge weights, what shortest-path algorithm should be used from one source?"
        )
        self.assertIn("source", concepts)
        self.assertIn("nonnegative", concepts)

        bayes_concepts = derive_task_statement_concepts(
            "In a medical test with rare disease prevalence, why can a positive result still have modest probability of true disease?"
        )
        self.assertIn("positive result", bayes_concepts)
        self.assertIn("prevalence", bayes_concepts)
        self.assertNotIn("posterior", bayes_concepts)

        http_concepts = derive_task_statement_concepts(
            "In HTTP, what does it mean that GET is idempotent?"
        )
        self.assertIn("same effect", http_concepts)
        self.assertIn("does not change", http_concepts)

    def test_task_statement_concept_extractor_covers_deep_questions_without_rubric_leakage(self):
        payment = derive_task_statement_concepts(
            "Design a payment worker for an at-least-once queue. The worker calls an external PSP that supports idempotency keys, and the process can crash after the PSP charge succeeds but before the local database commit. How do you prevent double charge and still converge to the right final state?"
        )
        self.assertIn("at-least-once", payment)
        self.assertIn("idempotency key", payment)
        self.assertIn("local database commit", payment)
        self.assertIn("double charge", payment)
        self.assertNotIn("reconciliation", payment)

        migration = derive_task_statement_concepts(
            "You need to split a monolith Orders table into a new Order service with zero downtime. Reads and writes continue during migration, correctness must be verified before cutover, and rollback must still be possible after cutover."
        )
        self.assertIn("zero downtime", migration)
        self.assertIn("verification before cutover", migration)
        self.assertIn("rollback", migration)
        self.assertNotIn("backfill", migration)

    def test_missing_task_statement_concepts_tracks_visible_answer_terms(self):
        question = "For online undirected connectivity with edge additions and connectivity queries, what structure is appropriate?"
        answer = "Use Union-Find for edge additions."
        missing = missing_task_statement_concepts(question, answer)
        self.assertIn("connectivity queries", missing)
        self.assertNotIn("online", missing)

    def test_task_concept_constraints_compose_into_key_terms(self):
        root = ReasoningStep(
            step_id="step_task_concept_root",
            parent_step_id=None,
            task_id="For a directed graph, what shortest-path algorithm should be used from one source?",
            focus="answer shortest-path question",
            looking_for="final answer",
        )
        response = """STEP_RESULT
status: resolved
result: Use Dijkstra when all edges are nonnegative.
delta:
  decisions:
    - use Dijkstra
END_STEP_RESULT"""
        result = run_fast_step_loop(
            root_step=root,
            llm_call=lambda _prompt: response,
            initial_signals=[
                SignalNode(
                    id="sig_task_source",
                    kind="constraint",
                    text=task_concept_constraint("source"),
                    activation_keys=["source"],
                    produced_by="controller",
                    confidence=0.95,
                )
            ],
            checker_registry=CheckerRegistry(["generic_step_format"]),
            config=FastLoopConfig(max_total_steps=1),
        )
        composed = compose_final_answer(result)
        self.assertIn("shortest paths from one source", composed)

    def test_compose_final_answer_adds_http_same_effect_constraint(self):
        root = ReasoningStep(
            step_id="step_http_root",
            parent_step_id=None,
            task_id="In HTTP, what does it mean that GET is idempotent?",
            focus="answer HTTP idempotence question",
            looking_for="final answer",
        )
        response = """STEP_RESULT
status: resolved
result: GET can be repeated without changing server state.
delta:
  decisions:
    - explain GET idempotence
END_STEP_RESULT"""
        result = run_fast_step_loop(
            root_step=root,
            llm_call=lambda _prompt: response,
            initial_signals=[
                SignalNode(
                    id="sig_http",
                    kind="constraint",
                    text="Repeating GET has the same effect and does not change server or resource state.",
                    activation_keys=["same", "effect", "change", "server", "state"],
                    produced_by="controller",
                    confidence=0.95,
                )
            ],
            checker_registry=CheckerRegistry(["generic_step_format"]),
            config=FastLoopConfig(max_total_steps=1),
        )
        composed = compose_final_answer(result)
        self.assertIn("same effect", composed.lower())

    def test_compose_final_answer_adds_all_negative_maximum_element_constraint(self):
        root = ReasoningStep(
            step_id="step_negative_root",
            parent_step_id=None,
            task_id="When computing maximum non-empty subarray sum, how should an all-negative array be handled?",
            focus="answer all-negative subarray question",
            looking_for="final answer",
        )
        response = """STEP_RESULT
status: resolved
result: For an all-negative array, return the largest (least negative) value.
delta:
  decisions:
    - explain all-negative handling
END_STEP_RESULT"""
        result = run_fast_step_loop(
            root_step=root,
            llm_call=lambda _prompt: response,
            initial_signals=[
                SignalNode(
                    id="sig_negative",
                    kind="risk",
                    text="For non-empty subarrays, all-negative arrays must return the maximum element, not 0.",
                    activation_keys=["negative", "maximum", "element", "not", "0"],
                    produced_by="controller",
                    confidence=0.95,
                )
            ],
            checker_registry=CheckerRegistry(["generic_step_format"]),
            config=FastLoopConfig(max_total_steps=1),
        )
        composed = compose_final_answer(result)
        self.assertIn("maximum element", composed.lower())

    def test_compose_final_answer_adds_shared_state_constraint(self):
        root = ReasoningStep(
            step_id="step_race_root",
            parent_step_id=None,
            task_id="What is a race condition in concurrent programming?",
            focus="answer race condition question",
            looking_for="final answer",
        )
        response = """STEP_RESULT
status: resolved
result: A race condition happens when thread timing changes the program outcome.
delta:
  decisions:
    - explain race condition
END_STEP_RESULT"""
        result = run_fast_step_loop(
            root_step=root,
            llm_call=lambda _prompt: response,
            initial_signals=[
                SignalNode(
                    id="sig_race",
                    kind="constraint",
                    text="Race conditions involve concurrent access to shared state or shared resources.",
                    activation_keys=["race", "condition", "shared", "state", "resources"],
                    produced_by="controller",
                    confidence=0.95,
                )
            ],
            checker_registry=CheckerRegistry(["generic_step_format"]),
            config=FastLoopConfig(max_total_steps=1),
        )
        composed = compose_final_answer(result)
        self.assertIn("shared state", composed.lower())

    def test_packet_from_task_frame_has_stable_cache_key(self):
        frame = GraphTaskFrame(
            session_id="sess_v2",
            constraints=[
                FrameItem("fi_ll", "constraint", "Use long long for sums.", 95, []),
                FrameItem("fi_out", "answer_requirement", "Return C++17 code.", 70, []),
            ],
            pitfalls=[
                FrameItem("fi_neg", "pitfall", "All-negative arrays must not return 0.", 90, []),
            ],
        )
        packet_a = packet_from_task_frame(
            task_summary="Solve dynamic max-subarray.",
            focus="choose data structure",
            looking_for="algorithm decision",
            frame=frame,
            budget_remaining={"calls": 4},
        )
        packet_b = StepContextPacket.from_dict(json.loads(json.dumps(packet_a.to_dict())))
        self.assertEqual(packet_b.cache_key, packet_a.cache_key)
        self.assertIn("Use long long for sums.", packet_a.hard_constraints)
        self.assertIn("All-negative arrays must not return 0.", packet_a.hard_constraints)
        self.assertLessEqual(len(packet_a.active_signals), 6)

    def test_reasoning_step_from_gap_uses_cross_session_canonical_gap_id(self):
        parent = ReasoningStep(
            step_id="step_parent",
            parent_step_id=None,
            task_id="task_1",
            focus="solve task",
            looking_for="final answer",
        )
        gap_a = MissingInfo(
            question="Does Kadane support online point updates?",
            why_needed="Need choose algorithm.",
            expected_shape="decision",
        )
        gap_b = MissingInfo(
            question="  does kadane support online point updates? ",
            why_needed="Different parent wording.",
            expected_shape=" decision ",
        )
        self.assertEqual(gap_a.canonical_id(), gap_b.canonical_id())
        child = ReasoningStep.from_gap(parent, gap_a)
        self.assertEqual(child.parent_step_id, "step_parent")
        self.assertEqual(child.depth, 1)
        self.assertIn(gap_a.canonical_id(), child.step_id)


class TestStepResultParserAndCheckers(unittest.TestCase):
    def test_parse_strict_step_result(self):
        raw = """STEP_RESULT
status: need_info
result: Need derive the merge rule.
missing:
  question: Does segment tree support max-subarray merge?
  why_needed: Need O(log n) online updates.
  expected_shape: evidence
delta:
  decisions:
    - reject Kadane for online updates
  risks:
    - Kadane is O(n) per update
  gaps:
    - segment_tree_merge_rule
END_STEP_RESULT"""
        parsed = parse_step_result(raw)
        self.assertEqual(parsed.status, "need_info")
        self.assertEqual(parsed.delta_transaction.status, "parsed")
        self.assertIsNotNone(parsed.missing)
        assert parsed.missing is not None
        self.assertEqual(parsed.missing.expected_shape, "evidence")
        self.assertIn("reject Kadane", parsed.delta.decisions[0])

    def test_parse_constraints_honored(self):
        parsed = parse_step_result("""STEP_RESULT
status: resolved
result: Use long long for sums.
constraints_honored:
  - Use long long/int64 for large numeric sums.
delta:
  decisions:
    - use wide integer
END_STEP_RESULT""")
        self.assertEqual(parsed.constraints_honored, ["Use long long/int64 for large numeric sums."])

    def test_generic_checker_rejects_unmarked_honored_constraint_claim(self):
        packet = StepContextPacket(
            task_summary="dynamic maximum subarray with large sums",
            focus="solve dynamic maximum subarray",
            looking_for="final answer",
            hard_constraints=["Use long long for sums."],
        )
        parsed = parse_step_result("""STEP_RESULT
status: resolved
result: Use int sums in the segment tree.
constraints_honored:
  - Use long long for sums.
delta:
  decisions:
    - use int sums
END_STEP_RESULT""")
        check = CheckerRegistry(["generic_step_format"]).verify(parsed, packet)
        self.assertFalse(check.passed)
        self.assertIn("honored_constraint_unmarked", {violation.code for violation in check.violations})

    def test_generic_checker_treats_unknown_honored_constraint_claim_as_soft(self):
        packet = StepContextPacket(
            task_summary="short factual answer",
            focus="answer directly",
            looking_for="final answer",
        )
        parsed = parse_step_result("""STEP_RESULT
status: resolved
result: Gradient descent can diverge when the learning rate is too high.
constraints_honored:
  - Result is concise and directly addresses the question.
delta:
  evidence:
    - high learning rate can diverge
END_STEP_RESULT""")
        check = CheckerRegistry(["generic_step_format"]).verify(parsed, packet)
        self.assertTrue(check.passed)
        self.assertIn("honored_constraint_unknown", {violation.code for violation in check.violations})

    def test_generic_checker_treats_blanket_honored_meta_claim_as_soft(self):
        packet = StepContextPacket(
            task_summary="inventory reservation",
            focus="design inventory reservation system",
            looking_for="final answer",
            hard_constraints=["Serialize writes per SKU with single-writer ownership or partition ownership to prevent oversell."],
        )
        parsed = parse_step_result("""STEP_RESULT
status: resolved
result: Use a single-writer partition owner per SKU to prevent oversell.
constraints_honored:
  - All hard constraints are explicitly satisfied in the design.
delta:
  decisions:
    - use single writer per sku
END_STEP_RESULT""")
        check = CheckerRegistry(["generic_step_format"]).verify(parsed, packet)
        self.assertTrue(check.passed)
        self.assertIn("honored_constraint_meta_claim", {violation.code for violation in check.violations})

    def test_parse_malformed_uses_regex_fallback(self):
        parsed = parse_step_result("Use Bellman-Ford because Dijkstra is unsafe with a negative edge.")
        self.assertEqual(parsed.delta_transaction.status, "skimmed")
        self.assertEqual(parsed.delta.produced_by, "regex_fallback")
        self.assertTrue(parsed.delta.decisions)
        self.assertTrue(parsed.delta.risks)

    def test_parse_malformed_delta_fuzz_fails_open_without_missing_child(self):
        malformed_outputs = [
            "",
            "STEP_RESULT\nstatus: maybe\nresult: no\nEND_STEP_RESULT",
            "STEP_RESULT\nstatus: need_info\nresult: missing object\nEND_STEP_RESULT",
            "STEP_RESULT\nstatus: need_info\nmissing:\n  question: only one field\nEND_STEP_RESULT",
            "Random prose: use the textbook definition, no structured block.",
            "STEP_RESULT\nstatus: resolved\nresult: unterminated block",
        ]
        for raw in malformed_outputs:
            with self.subTest(raw=raw):
                parsed = parse_step_result(raw)
                self.assertIn(parsed.delta_transaction.status, {"skimmed", "dropped"})
                self.assertIsNone(parsed.missing)
                if raw.strip():
                    self.assertEqual(parsed.delta.produced_by, "regex_fallback")

    def test_shortest_path_checker_rejects_dijkstra_with_negative_edge(self):
        packet = StepContextPacket(
            task_summary="shortest path with a negative edge",
            focus="choose shortest-path algorithm",
            looking_for="safe algorithm",
            hard_constraints=["Negative edge present; Dijkstra is unsafe."],
        )
        result = parse_step_result("""STEP_RESULT
status: resolved
result: Use Dijkstra from the source.
delta:
  decisions:
    - use Dijkstra
END_STEP_RESULT""")
        check = CheckerRegistry(["shortest_path_safety"]).verify(result, packet)
        self.assertFalse(check.passed)
        self.assertEqual(check.violations[0].code, "dijkstra_negative_edge")

    def test_shortest_path_checker_allows_nonnegative_dijkstra(self):
        packet = StepContextPacket(
            task_summary="single-source shortest paths with nonnegative edges",
            focus="choose shortest-path algorithm from one source",
            looking_for="valid algorithm",
            hard_constraints=["All edge weights are nonnegative."],
        )
        result = parse_step_result("""STEP_RESULT
status: resolved
result: Use Dijkstra's algorithm from one source when all edge weights are nonnegative.
delta:
  decisions:
    - use Dijkstra from one source
END_STEP_RESULT""")
        check = CheckerRegistry(["shortest_path_safety"]).verify(result, packet)
        self.assertTrue(check.passed)
        self.assertEqual(check.violations, [])

    def test_shortest_path_checker_ignores_repair_focus_for_nonnegative_task(self):
        packet = StepContextPacket(
            task_summary="For a directed graph with only nonnegative edge weights, choose the single-source shortest-path algorithm.",
            focus="resolve gap: How should this failed step be repaired? Dijkstra commitment under negative edge signal.",
            looking_for="repair invalid shortest-path answer",
            hard_constraints=["All edge weights must be nonnegative."],
        )
        result = parse_step_result("""STEP_RESULT
status: resolved
result: Use Dijkstra's algorithm from one source when all edge weights are nonnegative.
delta:
  repairs:
    - restate the nonnegative-edge precondition explicitly
END_STEP_RESULT""")
        check = CheckerRegistry(["shortest_path_safety"]).verify(result, packet)
        self.assertTrue(check.passed)
        self.assertEqual(check.violations, [])

    def test_factual_recall_checker_is_soft_and_evidence_anchored(self):
        packet = StepContextPacket(
            task_summary="Define entropy.",
            focus="answer factual question",
            looking_for="short definition",
            active_signals=[
                SignalNode(
                    id="sig_entropy",
                    kind="evidence",
                    text="Entropy is a thermodynamic state function related to multiplicity.",
                    activation_keys=["entropy", "thermodynamic", "state", "multiplicity"],
                    confidence=0.8,
                )
            ],
        )
        anchored = parse_step_result("""STEP_RESULT
status: resolved
result: Entropy is a thermodynamic state function.
delta:
  evidence:
    - entropy is thermodynamic
END_STEP_RESULT""")
        anchored_check = CheckerRegistry(["factual_recall"]).verify(anchored, packet)
        self.assertTrue(anchored_check.passed)
        self.assertEqual(anchored_check.violations, [])

        unrelated = parse_step_result("""STEP_RESULT
status: resolved
result: Photosynthesis converts light into chemical energy.
delta:
  evidence:
    - photosynthesis uses light
END_STEP_RESULT""")
        unrelated_check = CheckerRegistry(["factual_recall"]).verify(unrelated, packet)
        self.assertTrue(unrelated_check.passed)
        self.assertEqual(unrelated_check.violations[0].code, "factual_evidence_unreferenced")
        self.assertEqual(unrelated_check.violations[0].severity, "soft")

    def test_dynamic_checker_requires_segment_tree_aggregate_fields(self):
        packet = StepContextPacket(
            task_summary="dynamic maximum subarray with online updates",
            focus="solve dynamic maximum subarray with online point updates",
            looking_for="final answer",
            hard_constraints=[
                "Online point updates require efficient updates.",
                "Use long long for sums.",
            ],
        )
        vague = parse_step_result("""STEP_RESULT
status: resolved
result: Use a segment tree for maximum subarray queries with point updates.
delta:
  decisions:
    - use segment tree
END_STEP_RESULT""")
        vague_check = CheckerRegistry(["dynamic_max_subarray"]).verify(vague, packet)
        self.assertFalse(vague_check.passed)
        self.assertIn(
            "segment_tree_aggregate_missing",
            {violation.code for violation in vague_check.violations},
        )
        self.assertIn(
            "long_long_missing",
            {violation.code for violation in vague_check.violations},
        )

        concrete = parse_step_result("""STEP_RESULT
status: resolved
result: Use a segment tree storing long long sum, prefix, suffix, and best. Merge left and right nodes with best = max(left.best, right.best, left.suffix + right.prefix).
delta:
  decisions:
    - use segment tree with sum prefix suffix best and left/right merge
END_STEP_RESULT""")
        concrete_check = CheckerRegistry(["dynamic_max_subarray"]).verify(concrete, packet)
        self.assertTrue(concrete_check.passed)

    def test_dynamic_checker_rejects_wrong_kadane_claimed_log_update(self):
        packet = StepContextPacket(
            task_summary="dynamic maximum subarray with online updates",
            focus="solve dynamic maximum subarray with online point updates",
            looking_for="final answer",
            hard_constraints=[
                "Online point updates require efficient updates.",
                "Use long long for sums.",
                "For non-empty subarrays, all-negative arrays must return the maximum element, not 0.",
            ],
        )
        wrong = parse_step_result("""STEP_RESULT
status: resolved
result: Use Kadane after each update and claim O(log n) updates with long long.
delta:
  decisions:
    - use Kadane after each update with O(log n) updates
END_STEP_RESULT""")
        check = CheckerRegistry(["generic_step_format", "algorithm_design", "dynamic_max_subarray"]).verify(wrong, packet)
        self.assertFalse(check.passed)
        self.assertIn("kadane_online", {violation.code for violation in check.violations})
        self.assertIn("segment_tree_missing", {violation.code for violation in check.violations})

    def test_dynamic_connectivity_checker_requires_rollback_and_time_axis(self):
        packet = StepContextPacket(
            task_summary="add(u,v), remove(u,v), connected(u,v) dynamic connectivity",
            focus="design a faster-than-bfs connectivity algorithm with deletions",
            looking_for="final answer",
            hard_constraints=[
                "Solve the add/remove/connectivity task offline with edge-active intervals over time and a rollback-capable DSU.",
            ],
        )
        vague = parse_step_result("""STEP_RESULT
status: resolved
result: Use Union-Find for connectivity and process each operation as it arrives.
delta:
  decisions:
    - use union find
END_STEP_RESULT""")
        vague_check = CheckerRegistry(["dynamic_connectivity_deletions"]).verify(vague, packet)
        self.assertFalse(vague_check.passed)
        self.assertIn("dsu_without_rollback", {violation.code for violation in vague_check.violations})
        self.assertIn("time_axis_missing", {violation.code for violation in vague_check.violations})

        concrete = parse_step_result("""STEP_RESULT
status: resolved
result: Process the queries offline. Give each edge an active interval over time, place those intervals on a segment tree over time, and use a rollback DSU / Union-Find while traversing the tree so add/remove connectivity queries are answered without recomputing BFS per query.
delta:
  decisions:
    - use offline segment tree over time with rollback DSU
END_STEP_RESULT""")
        concrete_check = CheckerRegistry(["dynamic_connectivity_deletions"]).verify(concrete, packet)
        self.assertTrue(concrete_check.passed)

    def test_segment_tree_beats_checker_requires_invariant_bundle(self):
        packet = StepContextPacket(
            task_summary="range_chmin and range_sum design",
            focus="answer range_chmin data structure question",
            looking_for="final answer",
            hard_constraints=[
                "Use segment tree beats or an equivalent max/second-max/count_max/sum state bundle; ordinary lazy propagation is insufficient.",
            ],
        )
        vague = parse_step_result("""STEP_RESULT
status: resolved
result: Use a lazy segment tree for range_chmin and range_sum.
delta:
  decisions:
    - use lazy segment tree
END_STEP_RESULT""")
        vague_check = CheckerRegistry(["segment_tree_beats"]).verify(vague, packet)
        self.assertFalse(vague_check.passed)
        self.assertIn("beats_state_missing", {violation.code for violation in vague_check.violations})

        concrete = parse_step_result("""STEP_RESULT
status: resolved
result: Use segment tree beats. Store max, second max, count_max, and sum at each node, and apply range_chmin lazily only when x lies between the current max and second max so only the current maxima change.
delta:
  decisions:
    - use segment tree beats with second max and count_max
END_STEP_RESULT""")
        concrete_check = CheckerRegistry(["segment_tree_beats"]).verify(concrete, packet)
        self.assertTrue(concrete_check.passed)

    def test_payment_crash_recovery_checker_requires_durable_state_and_reconciliation(self):
        packet = StepContextPacket(
            task_summary="payment worker crash recovery with PSP idempotency",
            focus="design payment worker crash recovery",
            looking_for="final answer",
            hard_constraints=[
                "Persist a durable local payment state machine around the external charge; idempotency key alone is insufficient.",
                "After a crash, reconcile uncertain payment outcomes by querying PSP state before replaying or retrying.",
            ],
        )
        vague = parse_step_result("""STEP_RESULT
status: resolved
result: Use an idempotency key and retry the payment message until it succeeds.
delta:
  decisions:
    - use idempotency key and retry
END_STEP_RESULT""")
        vague_check = CheckerRegistry(["payment_crash_recovery"]).verify(vague, packet)
        self.assertFalse(vague_check.passed)
        self.assertIn("durable_state_missing", {violation.code for violation in vague_check.violations})
        self.assertIn("reconciliation_missing", {violation.code for violation in vague_check.violations})

        concrete = parse_step_result("""STEP_RESULT
status: resolved
result: Keep a durable payment intent state machine with pending/charged states, send PSP requests with an idempotency key, and after any crash query PSP status to reconcile uncertain outcomes before replaying the at-least-once message. Retries use consumer-side dedupe so the charge converges to one final state.
delta:
  decisions:
    - use durable payment intent with PSP reconciliation
END_STEP_RESULT""")
        concrete_check = CheckerRegistry(["payment_crash_recovery"]).verify(concrete, packet)
        self.assertTrue(concrete_check.passed)

    def test_zero_downtime_migration_checker_requires_phased_plan(self):
        packet = StepContextPacket(
            task_summary="zero-downtime orders service extraction",
            focus="outline migration plan",
            looking_for="final answer",
            hard_constraints=[
                "Use ordered migration phases: backfill historical data, capture live writes, verify parity, then cut over.",
                "Keep rollback viable by preserving the old-good source of truth until verification and cutover are complete.",
            ],
        )
        vague = parse_step_result("""STEP_RESULT
status: resolved
result: Use CDC and cut over once the new service is ready.
delta:
  decisions:
    - migrate with CDC then cut over
END_STEP_RESULT""")
        vague_check = CheckerRegistry(["zero_downtime_migration"]).verify(vague, packet)
        self.assertFalse(vague_check.passed)
        self.assertIn("backfill_missing", {violation.code for violation in vague_check.violations})
        self.assertIn("verification_missing", {violation.code for violation in vague_check.violations})

    def test_inventory_reservation_checker_requires_single_writer_and_lifecycle(self):
        packet = StepContextPacket(
            task_summary="flash-sale inventory reservation",
            focus="design inventory reservation system",
            looking_for="final answer",
            hard_constraints=[
                "Serialize writes per SKU with single-writer ownership or partition ownership to prevent oversell.",
                "Model the reservation lifecycle explicitly: hold/reserve, confirm, release/expire, and reconcile from the authoritative source of truth.",
            ],
        )
        vague = parse_step_result("""STEP_RESULT
status: resolved
result: Use a Redis cache counter for inventory and decrement it on every request.
delta:
  decisions:
    - use cache counter
END_STEP_RESULT""")
        vague_check = CheckerRegistry(["inventory_reservation"]).verify(vague, packet)
        self.assertFalse(vague_check.passed)
        codes = {violation.code for violation in vague_check.violations}
        self.assertIn("single_writer_missing", codes)
        self.assertIn("inventory_authority_missing", codes)


class TestFastStepLoop(unittest.TestCase):
    def test_malformed_need_info_does_not_recurse(self):
        root = ReasoningStep(
            step_id="step_fuzz_root",
            parent_step_id=None,
            task_id="fuzz malformed output",
            focus="answer despite malformed structured output",
            looking_for="final answer",
        )

        def llm(_prompt: str) -> str:
            return "STEP_RESULT\nstatus: need_info\nresult: malformed missing object\nEND_STEP_RESULT"

        result = run_fast_step_loop(
            root_step=root,
            llm_call=llm,
            checker_registry=CheckerRegistry(["generic_step_format"]),
            config=FastLoopConfig(max_total_steps=4),
        )
        self.assertEqual(len(result.raw_outputs), 1)
        self.assertEqual(len(result.steps), 1)
        self.assertEqual(result.step_results[0].delta_transaction.status, "skimmed")
        self.assertIsNone(result.step_results[0].missing)

    def test_attach_fast_loop_journals_delta_transaction(self):
        root = ReasoningStep(
            step_id="step_trace_root",
            parent_step_id=None,
            task_id="trace malformed output",
            focus="trace malformed output",
            looking_for="final answer",
        )

        def llm(_prompt: str) -> str:
            return "Use Bellman-Ford because Dijkstra is unsafe with a negative edge."

        result = run_fast_step_loop(
            root_step=root,
            llm_call=llm,
            checker_registry=CheckerRegistry(["generic_step_format"]),
            config=FastLoopConfig(max_total_steps=2),
        )
        session = SessionSubgraphController("sess_v2_trace", "q", "cs4")
        attach_fast_loop_to_session(session, result)
        delta_nodes = [
            node for node in session.subgraph.nodes.values()
            if node.get("node_type") == "substrate_v2_delta"
        ]
        self.assertEqual(len(delta_nodes), 1)
        txn = delta_nodes[0]["delta_transaction"]
        self.assertEqual(txn["status"], "skimmed")
        self.assertEqual(txn["delta"]["produced_by"], "regex_fallback")
        self.assertEqual(txn["parse_error"], "missing STEP_RESULT block")

    def test_checker_hard_failure_opens_repair_child_then_resumes_parent(self):
        root = ReasoningStep(
            step_id="step_ioi_repair_root",
            parent_step_id=None,
            task_id="dynamic max-subarray with online point updates",
            focus="solve dynamic maximum subarray with online updates",
            looking_for="final answer",
        )
        initial_signals = [
            SignalNode(
                id="sig_updates",
                kind="constraint",
                text="Online point updates require efficient updates.",
                activation_keys=["online", "point", "updates", "efficient"],
                confidence=0.95,
            )
        ]
        responses = [
            """STEP_RESULT
status: resolved
result: Use Kadane after each update.
delta:
  decisions:
    - use Kadane after every online update
END_STEP_RESULT""",
            """STEP_RESULT
status: resolved
result: The repair is to use a segment tree storing sum, prefix, suffix, best for O(log n) updates. Merge left and right nodes with best = max(left.best, right.best, left.suffix + right.prefix).
delta:
  repairs:
    - replace Kadane with segment tree aggregate
  evidence:
    - segment tree supports maximum subarray point updates in O(log n)
END_STEP_RESULT""",
            """STEP_RESULT
status: resolved
result: Use a segment tree with sum, prefix, suffix, and best for O(log n) updates. Merge left and right nodes with best = max(left.best, right.best, left.suffix + right.prefix).
delta:
  decisions:
    - use segment tree for online max-subarray updates
END_STEP_RESULT""",
        ]

        def llm(_prompt: str) -> str:
            return responses.pop(0)

        result = run_fast_step_loop(
            root_step=root,
            llm_call=llm,
            initial_signals=initial_signals,
            checker_registry=CheckerRegistry(["generic_step_format", "dynamic_max_subarray"]),
            config=FastLoopConfig(max_total_steps=4, max_child_depth=2),
        )
        self.assertEqual(result.final_step_result.status, "resolved")
        self.assertEqual(len(result.raw_outputs), 3)
        self.assertTrue(any(step.parent_step_id == "step_ioi_repair_root" for step in result.steps))
        self.assertTrue(any(sig.kind == "repair" and "segment tree" in sig.text.lower() for sig in result.signals))

    def test_failed_repair_child_does_not_spawn_repair_grandchild(self):
        root = ReasoningStep(
            step_id="step_ioi_bad_repair_root",
            parent_step_id=None,
            task_id="dynamic max-subarray with online point updates",
            focus="solve dynamic maximum subarray with online updates",
            looking_for="final answer",
        )
        responses = [
            """STEP_RESULT
status: resolved
result: Use Kadane after each update.
delta:
  decisions:
    - use Kadane after every online update
END_STEP_RESULT""",
            """STEP_RESULT
status: resolved
result: Repair: still use Kadane after each update.
delta:
  repairs:
    - keep Kadane for online updates
END_STEP_RESULT""",
            """STEP_RESULT
status: resolved
result: This response must not be consumed.
delta:
  decisions:
    - unexpected grandchild
END_STEP_RESULT""",
        ]

        def llm(_prompt: str) -> str:
            return responses.pop(0)

        result = run_fast_step_loop(
            root_step=root,
            llm_call=llm,
            checker_registry=CheckerRegistry(["generic_step_format", "dynamic_max_subarray"]),
            config=FastLoopConfig(max_total_steps=5, max_child_depth=4, max_children_per_step=3),
        )
        self.assertEqual(result.final_step_result.status, "resolved")
        self.assertEqual(result.root_step.status, "failed")
        self.assertEqual(len(result.raw_outputs), 2)
        self.assertFalse(any(step.depth >= 2 for step in result.steps))
        self.assertEqual(len(responses), 1)
        self.assertTrue(any(sig.kind == "unresolved_gap" for sig in result.signals))

    def test_failed_parent_resume_returns_last_passed_repair_best_effort(self):
        root = ReasoningStep(
            step_id="step_ioi_best_effort_root",
            parent_step_id=None,
            task_id="dynamic max-subarray with online point updates",
            focus="solve dynamic maximum subarray with online updates",
            looking_for="final answer",
        )
        responses = [
            """STEP_RESULT
status: resolved
result: Use a segment tree but omit the aggregate fields.
delta:
  decisions:
    - use segment tree
END_STEP_RESULT""",
            """STEP_RESULT
status: resolved
result: Repair: use a segment tree storing long long sum, prefix, suffix, and best. Merge left and right nodes with best = max(left.best, right.best, left.suffix + right.prefix).
delta:
  repairs:
    - repair with sum prefix suffix best and merge rule
END_STEP_RESULT""",
            """STEP_RESULT
status: resolved
result: Parent resume is still vague and wrong.
delta:
  decisions:
    - vague segment tree
END_STEP_RESULT""",
            """STEP_RESULT
status: resolved
result: Repair child also fails by omitting long long.
delta:
  repairs:
    - incomplete repair
END_STEP_RESULT""",
        ]

        def llm(_prompt: str) -> str:
            return responses.pop(0)

        result = run_fast_step_loop(
            root_step=root,
            llm_call=llm,
            checker_registry=CheckerRegistry(["generic_step_format", "dynamic_max_subarray"]),
            config=FastLoopConfig(max_total_steps=5, max_child_depth=2, max_children_per_step=2),
        )
        self.assertIn("left.suffix + right.prefix", result.final_step_result.result)
        self.assertIn("long long", result.final_step_result.result)
        self.assertNotIn("Parent resume is still vague", result.final_step_result.result)

    def test_fast_loop_instrumentation_records_delta_checker_repair_and_tokens(self):
        root = ReasoningStep(
            step_id="step_instrument_root",
            parent_step_id=None,
            task_id="dynamic max-subarray with online point updates",
            focus="solve dynamic maximum subarray with online updates",
            looking_for="final answer",
        )
        responses = [
            """STEP_RESULT
status: resolved
result: Use Kadane after each update.
delta:
  decisions:
    - use Kadane after every online update
END_STEP_RESULT""",
            """STEP_RESULT
status: resolved
result: Use a segment tree with long long sum, prefix, suffix, and best. Merge left and right nodes with best = max(left.best, right.best, left.suffix + right.prefix).
delta:
  repairs:
    - repair with segment tree sum prefix suffix best
END_STEP_RESULT""",
            """STEP_RESULT
status: resolved
result: Use a segment tree with long long sum, prefix, suffix, and best. Merge left and right nodes with best = max(left.best, right.best, left.suffix + right.prefix).
delta:
  decisions:
    - final segment tree answer
END_STEP_RESULT""",
        ]

        def llm(_prompt: str) -> str:
            return responses.pop(0)

        result = run_fast_step_loop(
            root_step=root,
            llm_call=llm,
            checker_registry=CheckerRegistry(["generic_step_format", "dynamic_max_subarray"]),
            config=FastLoopConfig(max_total_steps=4, max_child_depth=2),
        )

        self.assertEqual(result.delta_status_breakdown["parsed"], 3)
        self.assertEqual(result.delta_status_breakdown["skimmed"], 0)
        self.assertGreaterEqual(result.checker_outcome_breakdown["failed_hard"], 1)
        self.assertGreaterEqual(
            result.checker_outcome_breakdown["passed_strict"]
            + result.checker_outcome_breakdown["passed_soft"],
            1,
        )
        self.assertEqual(result.repair_triggered, 1)
        self.assertEqual(result.repair_succeeded, 1)
        self.assertEqual(len(result.tokens_per_call), len(result.raw_outputs))
        self.assertTrue(all(value > 0 for value in result.tokens_per_call))
        self.assertEqual(set(result.activated_signal_ages), {"min", "median", "max"})

    def test_compose_final_answer_preserves_missing_hard_constraints(self):
        root = ReasoningStep(
            step_id="step_compose_root",
            parent_step_id=None,
            task_id="dynamic max-subarray with online point updates",
            focus="solve dynamic maximum subarray with online updates",
            looking_for="final answer",
        )
        response = """STEP_RESULT
status: resolved
result: Use a segment tree with sum, prefix, suffix, and best.
delta:
  decisions:
    - use segment tree
END_STEP_RESULT"""

        result = run_fast_step_loop(
            root_step=root,
            llm_call=lambda _prompt: response,
            initial_signals=[
                SignalNode(
                    id="sig_long",
                    kind="constraint",
                    text="Use long long/int64 for large numeric sums.",
                    activation_keys=["long", "int64", "sums"],
                    confidence=0.95,
                )
            ],
            checker_registry=CheckerRegistry(["generic_step_format"]),
            config=FastLoopConfig(max_total_steps=1),
        )
        composed = compose_final_answer(result)
        self.assertIn("Use long long/int64 for large numeric sums.", composed)

    def test_composer_does_not_promote_checker_violations_to_answer_constraints(self):
        root = ReasoningStep(
            step_id="step_compose_checker_root",
            parent_step_id=None,
            task_id="union find connectivity",
            focus="answer connectivity data structure question",
            looking_for="final answer",
        )
        response = """STEP_RESULT
status: resolved
result: Use Union-Find for incremental connectivity.
delta:
  decisions:
    - use union find
END_STEP_RESULT"""

        result = run_fast_step_loop(
            root_step=root,
            llm_call=lambda _prompt: response,
            initial_signals=[
                SignalNode(
                    id="sig_checker_noise",
                    kind="risk",
                    text="honored_constraint_unmarked: Claimed honored constraint is not visible in result.",
                    activation_keys=["honored", "constraint", "unmarked"],
                    produced_by="checker",
                    confidence=0.95,
                )
            ],
            checker_registry=CheckerRegistry(["generic_step_format"]),
            config=FastLoopConfig(max_total_steps=1),
        )
        composed = compose_final_answer(result)
        self.assertNotIn("honored_constraint_unmarked", composed)

    def test_ioi_fast_loop_recurses_once_then_finalizes(self):
        root = ReasoningStep(
            step_id="step_ioi_root",
            parent_step_id=None,
            task_id="dynamic max-subarray",
            focus="solve dynamic max-subarray under online updates",
            looking_for="final C++17 answer",
        )
        initial_signals = [
            SignalNode(
                id="sig_updates",
                kind="constraint",
                text="q online point updates require efficient updates.",
                activation_keys=["online", "point", "updates", "efficient"],
                confidence=0.95,
            ),
            SignalNode(
                id="sig_long",
                kind="constraint",
                text="Use long long for sums.",
                activation_keys=["long", "sums"],
                confidence=0.95,
            ),
            SignalNode(
                id="sig_neg",
                kind="risk",
                text="All-negative non-empty arrays must return the maximum element, not 0.",
                activation_keys=["negative", "empty", "maximum", "element"],
                confidence=0.95,
            ),
        ]
        responses = [
            """STEP_RESULT
status: need_info
result: Kadane is not enough; need derive the segment tree merge rule.
missing:
  question: What merge rule supports maximum subarray under point updates?
  why_needed: Need O(log n) updates.
  expected_shape: evidence
delta:
  decisions:
    - reject Kadane for online point updates
  gaps:
    - segment_tree_merge_rule
END_STEP_RESULT""",
            """STEP_RESULT
status: resolved
result: Store sum, prefix, suffix, and best; merge left and right nodes with best = max(left.best, right.best, left.suffix + right.prefix) in O(1).
delta:
  evidence:
    - segment tree merge uses sum, prefix, suffix, best and supports O(log n) point updates
END_STEP_RESULT""",
            """STEP_RESULT
status: resolved
result: Use a segment tree with long long sum, prefix, suffix, and best. Merge left and right nodes with best = max(left.best, right.best, left.suffix + right.prefix). Leaves use the element value, so non-empty all-negative arrays return the maximum element, not 0. Each update is O(log n).
delta:
  decisions:
    - use segment tree for online point updates
  evidence:
    - long long segment tree handles non-empty all-negative arrays
END_STEP_RESULT""",
        ]

        def llm(_prompt: str) -> str:
            return responses.pop(0)

        result = run_fast_step_loop(
            root_step=root,
            llm_call=llm,
            initial_signals=initial_signals,
            checker_registry=CheckerRegistry(["generic_step_format", "algorithm_design", "dynamic_max_subarray"]),
            config=FastLoopConfig(max_total_steps=4, max_child_depth=2),
        )

        self.assertEqual(result.final_step_result.status, "resolved")
        self.assertFalse(result.budget_exhausted)
        self.assertEqual(len(result.raw_outputs), 3)
        self.assertTrue(any(step.parent_step_id == "step_ioi_root" for step in result.steps))
        self.assertTrue(any(sig.kind == "decision" and "segment tree" in sig.text for sig in result.signals))

    def test_dijkstra_fast_loop_resolves_in_one_call(self):
        root = ReasoningStep(
            step_id="step_dijkstra_root",
            parent_step_id=None,
            task_id="shortest path safety",
            focus="choose shortest-path algorithm with a negative edge",
            looking_for="safe algorithm",
        )
        initial_signals = [
            SignalNode(
                id="sig_negative",
                kind="risk",
                text="Negative edge present; Dijkstra is unsafe unless all edges are nonnegative.",
                activation_keys=["negative", "edge", "dijkstra", "unsafe"],
                confidence=0.95,
            )
        ]

        def llm(_prompt: str) -> str:
            return """STEP_RESULT
status: resolved
result: Dijkstra is unsafe with a negative edge; use Bellman-Ford and check for negative cycles if needed.
delta:
  decisions:
    - use Bellman-Ford for negative edge shortest path
  risks:
    - Dijkstra invalid when a negative edge violates nonnegative edge precondition
END_STEP_RESULT"""

        result = run_fast_step_loop(
            root_step=root,
            llm_call=llm,
            initial_signals=initial_signals,
            checker_registry=CheckerRegistry(["generic_step_format", "shortest_path_safety"]),
            config=FastLoopConfig(max_total_steps=4),
        )
        self.assertEqual(result.final_step_result.status, "resolved")
        self.assertEqual(len(result.raw_outputs), 1)
        self.assertTrue(result.checks[-1].passed)

    def test_sanitizer_strips_missing_none(self):
        cleaned = _sanitize_answer(
            "Query PSP state before retrying. missing: none. Use dedupe semantics."
        )
        self.assertNotIn("missing:", cleaned)
        self.assertIn("PSP", cleaned)

    def test_sanitizer_strips_missing_brackets(self):
        cleaned = _sanitize_answer(
            "Use a segment tree. missing: [] Each node stores max."
        )
        self.assertNotIn("missing", cleaned)
        self.assertIn("segment tree", cleaned)

    def test_sanitizer_strips_constraints_honored_section(self):
        cleaned = _sanitize_answer(
            "Use Bellman-Ford.\nconstraints_honored:\n  - long long\nEND_STEP_RESULT"
        )
        self.assertNotIn("constraints_honored", cleaned)
        self.assertNotIn("long long", cleaned)

    def test_composer_dedup_additions(self):
        root = ReasoningStep(
            step_id="step_dedup_root",
            parent_step_id=None,
            task_id="test dedup",
            focus="answer",
            looking_for="final answer",
        )
        response = "STEP_RESULT\nstatus: resolved\nresult: Short answer.\ndelta:\n  decisions:\n    - x\nEND_STEP_RESULT"
        signal_a = SignalNode(
            id="sig_a", kind="constraint",
            text="Explicitly preserve this task-statement concept: source.",
            activation_keys=["source"], produced_by="controller", confidence=0.95,
        )
        signal_b = SignalNode(
            id="sig_b", kind="constraint",
            text="Explicitly preserve this task-statement concept: source.",
            activation_keys=["source"], produced_by="controller", confidence=0.95,
        )
        result = run_fast_step_loop(
            root_step=root,
            llm_call=lambda _p: response,
            initial_signals=[signal_a, signal_b],
            checker_registry=CheckerRegistry(["generic_step_format"]),
            config=FastLoopConfig(max_total_steps=1),
        )
        composed = compose_final_answer(result)
        self.assertEqual(composed.count("shortest paths"), 1)

    def test_repair_resume_produces_distinct_step_occurrences(self):
        root = ReasoningStep(
            step_id="step_rep_root", parent_step_id=None,
            task_id="dynamic max-subarray online with point updates",
            focus="solve dynamic max-subarray with online updates",
            looking_for="final C++ answer",
        )

        responses = [
            "STEP_RESULT\nstatus: resolved\nresult: Use Kadane.\ndelta:\n  decisions:\n    - kadane\nEND_STEP_RESULT",
            "STEP_RESULT\nstatus: resolved\nresult: Use a segment tree with sum, prefix, suffix, best.\ndelta:\n  decisions:\n    - use segment tree\nEND_STEP_RESULT",
            "STEP_RESULT\nstatus: resolved\nresult: Use a segment tree with long long sum, prefix, suffix, and best.\ndelta:\n  decisions:\n    - use segment tree\nEND_STEP_RESULT",
        ]

        def llm(prompt: str) -> str:
            return responses.pop(0)

        result = run_fast_step_loop(
            root_step=root,
            llm_call=llm,
            initial_signals=[
                SignalNode(
                    id="sig_ioi", kind="constraint",
                    text="Use segment tree for online updates.",
                    activation_keys=["segment", "tree"], produced_by="controller",
                    confidence=0.95,
                )
            ],
            checker_registry=CheckerRegistry(["generic_step_format", "dynamic_max_subarray"]),
            config=FastLoopConfig(max_total_steps=5, max_child_depth=2),
        )
        self.assertGreaterEqual(len(result.steps), 3)
        step_ids = [s.step_id for s in result.steps]
        parent_occurrences = [s for s in step_ids if s == "step_rep_root"]
        self.assertGreaterEqual(len(parent_occurrences), 2)
        ages = result.activated_signal_ages
        for key in ("min", "median", "max"):
            self.assertIn(key, ages)
            self.assertGreaterEqual(ages[key], 0.0)

    def test_projection_preserves_stored_activation_keys(self):
        stored_signal = SignalNode(
            id="sig_original", kind="constraint",
            text="Use segment tree beats.",
            activation_keys=["segment", "beats", "tree"],
            produced_by="controller", confidence=0.95,
            scope="session",
        )
        subgraph = {
            "session_id": "test_sess",
            "query": "test query",
            "graph_id": "g1",
            "nodes": {
                "n1": {
                    "id": "n1",
                    "node_type": "substrate_v2_signal",
                    "signal": stored_signal.to_dict(),
                }
            },
            "edges": [],
        }
        from reasoning.schemas import SessionSubgraph
        sess = SessionSubgraph.from_dict(subgraph)
        projected = project_session_subgraph_to_signals(sess)
        self.assertEqual(len(projected), 1)
        self.assertEqual(projected[0].activation_keys, ["segment", "beats", "tree"])


class TestSignalClassification(unittest.TestCase):
    """Tests for post-3E signal quality classification helpers."""

    def test_is_checker_residue_known_codes(self):
        from reasoning.substrate_v2 import _is_checker_residue_text
        self.assertTrue(_is_checker_residue_text("segment_tree_merge_missing"))
        self.assertTrue(_is_checker_residue_text("long_long_missing"))
        self.assertTrue(_is_checker_residue_text("constraint_unaddressed"))

    def test_is_checker_residue_pattern(self):
        from reasoning.substrate_v2 import _is_checker_residue_text
        self.assertTrue(_is_checker_residue_text("segment_tree_aggregate_missing and more"))
        self.assertTrue(_is_checker_residue_text("backfill_missing in migration"))
        self.assertFalse(_is_checker_residue_text("Normal invariant about segment tree"))

    def test_is_checker_residue_clean_signal(self):
        from reasoning.substrate_v2 import _is_checker_residue_text
        self.assertFalse(_is_checker_residue_text("Use segment tree beats with second-max and sum"))
        self.assertFalse(_is_checker_residue_text("The solution requires idempotency keys"))

    def test_is_procedural_fragment(self):
        from reasoning.substrate_v2 import _is_procedural_fragment
        self.assertTrue(_is_procedural_fragment("I'll implement Kadane's algorithm"))
        self.assertTrue(_is_procedural_fragment("Let me explain the approach"))
        self.assertTrue(_is_procedural_fragment("Here's my approach"))
        self.assertTrue(_is_procedural_fragment("My solution is to use a segment tree"))
        self.assertTrue(_is_procedural_fragment("in this step we should"))
        self.assertTrue(_is_procedural_fragment("As a first step, implement rollback"))
        self.assertFalse(_is_procedural_fragment("Use a segment tree with range queries"))
        self.assertFalse(_is_procedural_fragment("We need rollback after cutover"))
        self.assertFalse(_is_procedural_fragment("The idea is to use a segment tree"))
        self.assertFalse(_is_procedural_fragment("This is the authoritative source of truth"))
        self.assertFalse(_is_procedural_fragment("First, we should consider the case"))

    def test_is_compact_invariant(self):
        from reasoning.substrate_v2 import _is_compact_invariant
        self.assertTrue(_is_compact_invariant("Use idempotency keys for payment deduplication"))
        self.assertTrue(_is_compact_invariant("Need single writer per SKU"))
        self.assertTrue(_is_compact_invariant("Backfill + live sync + verification"))
        self.assertFalse(_is_compact_invariant("A" * 250))
        self.assertFalse(_is_compact_invariant("Some unrelated text about algorithms"))

    def test_signal_quality_penalty_checker_residue(self):
        from reasoning.substrate_v2 import _signal_quality_penalty, SignalNode
        sig = SignalNode(id="s1", kind="constraint", text="segment_tree_merge_missing", confidence=0.5)
        self.assertEqual(_signal_quality_penalty(sig), 0.3)

    def test_signal_quality_penalty_procedural(self):
        from reasoning.substrate_v2 import _signal_quality_penalty, SignalNode
        sig = SignalNode(id="s2", kind="decision", text="I'll implement the algorithm", confidence=0.5)
        self.assertEqual(_signal_quality_penalty(sig), 0.15)

    def test_signal_quality_penalty_verbose(self):
        from reasoning.substrate_v2 import _signal_quality_penalty, SignalNode
        sig = SignalNode(id="s3", kind="evidence", text="x" * 350, confidence=0.5)
        self.assertEqual(_signal_quality_penalty(sig), 0.1)

    def test_signal_quality_penalty_none(self):
        from reasoning.substrate_v2 import _signal_quality_penalty, SignalNode
        sig = SignalNode(id="s4", kind="constraint", text="Use exactly-once semantics", confidence=0.5)
        self.assertEqual(_signal_quality_penalty(sig), 0.0)

    def test_signal_quality_bonus_compact_invariant(self):
        from reasoning.substrate_v2 import _signal_quality_bonus, SignalNode
        sig = SignalNode(id="s5", kind="constraint", text="Use idempotency keys for deduplication", confidence=0.5)
        self.assertEqual(_signal_quality_bonus(sig), 0.2)
        # exactly-once is deliberately excluded (payment hard-fail concept)
        sig2 = SignalNode(id="s5b", kind="constraint", text="Use exactly-once semantics", confidence=0.5)
        self.assertEqual(_signal_quality_bonus(sig2), 0.05)

    def test_signal_quality_bonus_zero_for_checker_residue(self):
        from reasoning.substrate_v2 import _signal_quality_bonus, SignalNode
        # checker residue that contains invariant keywords should get zero bonus
        sig = SignalNode(id="s_cr", kind="risk", text="honored_constraint_unmarked: Claimed honored constraint is not visible in result: backfill",
                         confidence=0.9)
        self.assertEqual(_signal_quality_bonus(sig), 0.0, "checker residue must get zero bonus even if text contains invariant keywords")
        # plain checker-residue code
        sig2 = SignalNode(id="s_cr2", kind="constraint", text="segment_tree_merge_missing", confidence=0.5)
        self.assertEqual(_signal_quality_bonus(sig2), 0.0)

    def test_signal_quality_bonus_controller(self):
        from reasoning.substrate_v2 import _signal_quality_bonus, SignalNode
        sig = SignalNode(id="s6", kind="constraint", text="Use segment tree beats", confidence=0.5, produced_by="controller")
        self.assertGreater(_signal_quality_bonus(sig), 0.15)

    def test_signal_quality_bonus_plain_constraint(self):
        from reasoning.substrate_v2 import _signal_quality_bonus, SignalNode
        sig = SignalNode(id="s7", kind="constraint", text="Some constraint", confidence=0.5)
        self.assertEqual(_signal_quality_bonus(sig), 0.05)

    def test_classify_signal_category(self):
        from reasoning.substrate_v2 import _classify_signal_category, SignalNode
        self.assertEqual(_classify_signal_category(SignalNode(id="s", kind="constraint", text="segment_tree_merge_missing")), "checker_residue")
        self.assertEqual(_classify_signal_category(SignalNode(id="s", kind="decision", text="I'll first implement")), "procedural_fragment")
        self.assertEqual(_classify_signal_category(SignalNode(id="s", kind="constraint", text="Use idempotency keys")), "reusable_invariant")
        self.assertEqual(_classify_signal_category(SignalNode(id="s", kind="constraint", text="Hard constraint about X")), "useful_constraint")
        self.assertEqual(_classify_signal_category(SignalNode(id="s", kind="evidence", text="x" * 350)), "verbose_signal")
        self.assertEqual(_classify_signal_category(SignalNode(id="s", kind="evidence", text="plain evidence")), "other")
        # procedural patterns tightened: these are no longer classified as procedural
        self.assertEqual(_classify_signal_category(SignalNode(id="s", kind="constraint", text="We need rollback after cutover")), "reusable_invariant")

    def test_select_active_signals_uses_quality_scoring(self):
        from reasoning.substrate_v2 import select_active_signals, SignalNode
        signals = [
            SignalNode(id="checker_res", kind="constraint", text="segment_tree_merge_missing",
                       activation_keys=["segment", "tree", "merge"], confidence=0.5),
            SignalNode(id="procedural", kind="decision", text="I'll implement the algorithm",
                       activation_keys=["implement", "algorithm"], confidence=0.5),
            SignalNode(id="invariant", kind="constraint", text="Use idempotency keys",
                       activation_keys=["idempotency", "keys"], confidence=0.5),
            SignalNode(id="plain", kind="constraint", text="Normal constraint",
                       activation_keys=["normal", "constraint"], confidence=0.5),
        ]
        selected = select_active_signals(focus="idempotency", looking_for="keys", signals=signals, max_signals=2)
        selected_ids = [s.id for s in selected]
        self.assertIn("invariant", selected_ids[:1], "compact invariant should rank highest")
        self.assertNotIn("checker_res", selected_ids, "checker residue should be suppressed")


if __name__ == "__main__":
    unittest.main()
