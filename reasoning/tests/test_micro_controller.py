from __future__ import annotations

import unittest

from graph_core import Edge, MemoryGraph, Node
from reasoning.micro_controller import (
    build_task_frame,
    build_finalize_user_message,
    compose_answer_from_slots,
    run_micro_epistemic_controller,
)


def _dijkstra_graph() -> MemoryGraph:
    return MemoryGraph(
        nodes={
            "dijkstra_requires_nonnegative_edge_weights": Node(
                id="dijkstra_requires_nonnegative_edge_weights",
                node_type="claim",
                confidence=0.99,
                text="Dijkstra's algorithm requires nonnegative edge weights for its greedy settlement logic to remain correct.",
                metadata={},
            ),
            "negative_edge_counterexample_test_apply": Node(
                id="negative_edge_counterexample_test_apply",
                node_type="application",
                confidence=0.98,
                text="A single negative edge can produce a counterexample where Dijkstra settles a vertex too early and returns the wrong shortest path.",
                metadata={},
            ),
            "bellman_ford_handles_negative_edges": Node(
                id="bellman_ford_handles_negative_edges",
                node_type="claim",
                confidence=0.98,
                text="Bellman-Ford handles negative edge weights by repeated relaxation and is the safe alternative when negative edges exist.",
                metadata={},
            ),
        },
        edges=[
            Edge(
                src="dijkstra_requires_nonnegative_edge_weights",
                dst="negative_edge_counterexample_test_apply",
                relation="support",
            ),
            Edge(
                src="negative_edge_counterexample_test_apply",
                dst="bellman_ford_handles_negative_edges",
                relation="support",
            ),
        ],
    )


def _dijkstra_explanation_graph() -> MemoryGraph:
    graph = _dijkstra_graph()
    graph.nodes["dijkstra_greedy_relaxation_explanation"] = Node(
        id="dijkstra_greedy_relaxation_explanation",
        node_type="explanation",
        confidence=0.97,
        text="Dijkstra repeatedly picks the unsettled node with the smallest tentative distance and relaxes its outgoing edges using a priority queue. It assumes nonnegative edge weights.",
        metadata={},
    )
    graph.nodes["strat_dijkstra_negative_legacy"] = Node(
        id="strat_dijkstra_negative_legacy",
        node_type="strategy",
        confidence=0.5,
        text="Strategy for: Can Dijkstra be trusted with one negative edge?\nKey nodes: negative_edge_counterexample_test_apply",
        metadata={
            "question_pattern": "Can Dijkstra be trusted with one negative edge?",
            "key_node_ids": ["negative_edge_counterexample_test_apply"],
        },
    )
    graph.nodes["strat_dijkstra_mechanism_v2"] = Node(
        id="strat_dijkstra_mechanism_v2",
        node_type="strategy",
        confidence=0.7,
        text="Strategy family: direct_judgment\nStrategy subtype: algorithm_mechanism_explanation",
        metadata={
            "task_family": "direct_judgment",
            "task_subtype": "algorithm_mechanism_explanation",
            "question_mode": "mechanism_explanation",
            "entry_conditions": {"algorithm": "Dijkstra"},
            "key_node_ids": ["dijkstra_greedy_relaxation_explanation"],
            "domain_keywords": ["dijkstra", "priority", "queue", "relaxation"],
            "slot_order": ["mechanism", "answer"],
            "checkpoint_plan": ["Read the mechanism node", "Compose a fresh explanation"],
            "strategy_schema_version": 2,
        },
    )
    return graph


def _kadane_graph() -> MemoryGraph:
    return MemoryGraph(
        nodes={
            "kadane_fails_all_negative_naive_false": Node(
                id="kadane_fails_all_negative_naive_false",
                node_type="claim",
                confidence=0.02,
                text="Misconception: Kadane inherently fails on all-negative arrays and returns 0.",
                metadata={},
            ),
            "kadane_correct_init_first_element": Node(
                id="kadane_correct_init_first_element",
                node_type="claim",
                confidence=0.99,
                text="Refutation: the correct non-empty Kadane initializes both current and best to a[0]. On an all-negative array, it returns the least-negative element rather than 0.",
                metadata={},
            ),
            "subarray_must_be_nonempty": Node(
                id="subarray_must_be_nonempty",
                node_type="claim",
                confidence=0.97,
                text="The standard Maximum Subarray problem requires the chosen subarray to be non-empty.",
                metadata={},
            ),
        },
        edges=[
            Edge(
                src="kadane_correct_init_first_element",
                dst="subarray_must_be_nonempty",
                relation="support",
            ),
        ],
    )


def _leaderboard_graph() -> MemoryGraph:
    return MemoryGraph(
        nodes={
            "cpp_fenwick_tree_template": Node(
                id="cpp_fenwick_tree_template",
                node_type="example",
                confidence=1.0,
                text="Fenwick (Binary Indexed Tree) supporting O(log n) point updates and O(log n) prefix-sum queries on an array of length n.",
                metadata={},
            ),
            "fenwick_update_and_prefix_query_log_n": Node(
                id="fenwick_update_and_prefix_query_log_n",
                node_type="claim",
                confidence=0.99,
                text="Fenwick-tree updates and prefix queries both run in O(log n).",
                metadata={},
            ),
            "db_btree_example": Node(
                id="db_btree_example",
                node_type="example",
                confidence=0.98,
                text="A B-tree is a common database index structure used for ordered access and pagination windows.",
                metadata={},
            ),
            "race_condition_unsynchronized_shared_state": Node(
                id="race_condition_unsynchronized_shared_state",
                node_type="claim",
                confidence=0.99,
                text="Unsynchronized shared mutable state can create race conditions because observed behavior depends on execution interleaving order.",
                metadata={},
            ),
        },
        edges=[],
    )


class TestMicroEpistemicController(unittest.TestCase):
    def test_direct_judgment_task_signature_canonicalizes_vacuum_sound_family(self):
        base = build_task_frame("Why can light travel through space but sound cannot?")
        paraphrases = [
            build_task_frame("If astronauts can see sunlight in space, why can't they hear it there?"),
            build_task_frame("In empty space, why could you see a flash but not hear it?"),
            build_task_frame("Why is sunlight visible in the vacuum of space even though no sound can be heard there?"),
        ]

        self.assertEqual(base.task_family, "direct_judgment")
        self.assertEqual(
            base.task_signature,
            "direct_judgment.sound_requires_medium_vs_light_vacuum",
        )
        for paraphrase in paraphrases:
            self.assertEqual(paraphrase.task_signature, base.task_signature)

    def test_direct_judgment_task_signature_canonicalizes_refraction_frequency_family(self):
        base = build_task_frame("Why does a prism bend light but not change the light's frequency?")
        paraphrases = [
            build_task_frame("Why doesn't refraction change the frequency of light?"),
            build_task_frame("If light slows down in glass, why does its frequency stay the same?"),
            build_task_frame("When a laser enters water, why doesn't its frequency change?"),
            build_task_frame("Light bends when entering glass. Why is the frequency still unchanged?"),
        ]

        self.assertEqual(base.task_family, "direct_judgment")
        self.assertEqual(
            base.task_signature,
            "direct_judgment.refraction_changes_speed_not_frequency",
        )
        for paraphrase in paraphrases:
            self.assertEqual(paraphrase.task_signature, base.task_signature)

    def test_direct_judgment_unrelated_question_does_not_collapse_into_physics_shortcuts(self):
        frame = build_task_frame("Why is the sky blue at noon?")

        self.assertEqual(frame.task_family, "direct_judgment")
        self.assertNotEqual(
            frame.task_signature,
            "direct_judgment.sound_requires_medium_vs_light_vacuum",
        )
        self.assertNotEqual(
            frame.task_signature,
            "direct_judgment.refraction_changes_speed_not_frequency",
        )
        self.assertTrue(frame.task_signature.startswith("direct_judgment."))

    def test_design_synthesis_uses_fine_grained_required_slots(self):
        frame = build_task_frame(
            "Design a real-time leaderboard with rank pagination, ties, and a 100ms propagation budget."
        )

        self.assertEqual("design_synthesis", frame.task_family)
        for slot in (
            "core_structure",
            "rank_query",
            "pagination",
            "tie_policy",
            "scale_architecture",
            "latency_budget",
            "consistency_model",
            "failure_mode_fix",
            "answer",
        ):
            self.assertIn(slot, frame.required_slots)
        self.assertNotIn("approach", frame.required_slots)

    def test_design_synthesis_seeds_question_slots_and_escalates_honestly(self):
        graph = _leaderboard_graph()
        outcome = run_micro_epistemic_controller(
            question=(
                "Design a real-time leaderboard service. Score updates must propagate within 100ms, "
                "rank queries must run in O(log n), and pagination should support rank windows."
            ),
            graph=graph,
            anchor_ids=["cpp_fenwick_tree_template", "fenwick_update_and_prefix_query_log_n"],
        )

        self.assertFalse(outcome.finalizable)
        self.assertIn("leaderboard", outcome.slot_values.get("problem_frame", "").lower())
        self.assertIn("100ms", outcome.slot_values.get("latency_budget", ""))
        self.assertIn("core_structure", outcome.slot_values)
        self.assertIn("rank_query", outcome.slot_values)
        self.assertTrue(outcome.micro_steps)
        self.assertEqual(outcome.micro_steps[0].action.value, "REUSE")
        self.assertNotEqual(outcome.micro_steps[-1].action.value, "REUSE")

    def test_algorithm_applicability_reuses_local_working_memory(self):
        graph = _dijkstra_graph()
        outcome = run_micro_epistemic_controller(
            question="Can Dijkstra be trusted with one negative edge?",
            graph=graph,
            anchor_ids=list(graph.nodes.keys()),
        )

        self.assertTrue(outcome.finalizable)
        self.assertEqual(outcome.task_family, "algorithm_applicability")
        self.assertEqual(outcome.working_set.global_queries_used, 0)
        self.assertGreaterEqual(outcome.subgoal_reuse_count, 1)
        self.assertIn("verdict", outcome.slot_values)
        self.assertIn("reason", outcome.slot_values)
        self.assertIn("alternative", outcome.slot_values)
        self.assertIn("caveat", outcome.slot_values)
        self.assertIn("FINALIZE", outcome.controller_action_counts)
        answer = compose_answer_from_slots(outcome)
        self.assertIn("Bellman-Ford", answer)

    def test_kadane_all_negative_reuses_without_requiring_alternative(self):
        graph = _kadane_graph()
        outcome = run_micro_epistemic_controller(
            question="Does Kadane fail on all-negative arrays?",
            graph=graph,
            anchor_ids=list(graph.nodes.keys()),
        )

        self.assertTrue(outcome.finalizable)
        self.assertEqual(outcome.task_family, "algorithm_applicability")
        self.assertEqual(outcome.working_set.global_queries_used, 0)
        self.assertIn("verdict", outcome.slot_values)
        self.assertIn("reason", outcome.slot_values)
        self.assertIn("caveat", outcome.slot_values)
        self.assertNotIn("alternative", outcome.task_frame.required_slots)
        answer = compose_answer_from_slots(outcome)
        self.assertIn("does not fail", answer.lower())
        self.assertIn("least negative", answer.lower())

    def test_specific_instance_stays_out_of_reuse_shortcut(self):
        graph = _dijkstra_graph()
        outcome = run_micro_epistemic_controller(
            question="Given graph edges (s,a,2), (a,b,-5), (s,b,1), can Dijkstra be trusted here?",
            graph=graph,
            anchor_ids=["dijkstra_requires_nonnegative_edge_weights"],
        )

        self.assertFalse(outcome.finalizable)
        self.assertEqual(outcome.task_family, "procedure_or_instance_verification")
        self.assertTrue(outcome.micro_steps)
        self.assertNotEqual(outcome.micro_steps[-1].action.value, "FINALIZE")
        self.assertTrue(outcome.controller_fallback_used)

    def test_exact_solved_subgoal_match_is_reused(self):
        graph = _dijkstra_graph()
        graph.nodes["ssg_dijkstra_negative"] = Node(
            id="ssg_dijkstra_negative",
            node_type="solved_subgoal",
            text="Standard Dijkstra is not generally correct with negative edge weights.",
            metadata={
                "summary": "Standard Dijkstra is not generally correct with negative edge weights.",
                "subgoal_signature": "shortest_path.dijkstra.negative_edge_weights.validity",
                "question_type": "algorithm_applicability",
                "input_conditions": {
                    "algorithm": "Dijkstra",
                    "graph_property": "negative_edge_weights",
                },
                "output_slots": {
                    "verdict": "Dijkstra is not guaranteed to be correct when negative edge weights are present.",
                    "reason": "A negative edge can improve a node after Dijkstra has already finalized it.",
                    "alternative": "Use Bellman-Ford instead.",
                    "caveat": "It may still work on some inputs, but not as a general guarantee.",
                },
                "valid_when": ["general correctness", "negative edge weights"],
                "invalid_when": ["specific graph instance", "modified dijkstra"],
                "supporting_node_ids": [
                    "dijkstra_requires_nonnegative_edge_weights",
                    "negative_edge_counterexample_test_apply",
                    "bellman_ford_handles_negative_edges",
                ],
            },
        )

        outcome = run_micro_epistemic_controller(
            question="Can Dijkstra be trusted with one negative edge?",
            graph=graph,
            anchor_ids=["ssg_dijkstra_negative"],
        )

        self.assertTrue(outcome.finalizable)
        self.assertTrue(outcome.micro_steps)
        self.assertEqual(outcome.micro_steps[0].action.value, "REUSE")
        self.assertEqual(outcome.micro_steps[0].matched_node_id, "ssg_dijkstra_negative")
        prompt = build_finalize_user_message(
            "Can Dijkstra be trusted with one negative edge?",
            outcome,
            graph,
        )
        self.assertIn("Resolved slots:", prompt)
        self.assertIn("Use Bellman-Ford instead.", prompt)

    def test_related_mechanism_question_does_not_reuse_negative_edge_answer_memory(self):
        graph = _dijkstra_graph()
        graph.nodes["strat_dijkstra_negative_legacy"] = Node(
            id="strat_dijkstra_negative_legacy",
            node_type="strategy",
            confidence=0.5,
            text="Strategy for: Can Dijkstra be trusted with one negative edge?\nKey nodes: negative_edge_counterexample_test_apply",
            metadata={
                "question_pattern": "Can Dijkstra be trusted with one negative edge?",
                "key_node_ids": ["negative_edge_counterexample_test_apply"],
            },
        )
        outcome = run_micro_epistemic_controller(
            question="How does Dijkstra work?",
            graph=graph,
            anchor_ids=["strat_dijkstra_negative_legacy"],
        )

        self.assertEqual(outcome.task_family, "direct_judgment")
        self.assertEqual(outcome.task_frame.context.task_subtype, "algorithm_mechanism_explanation")
        self.assertEqual(outcome.task_frame.context.question_mode, "mechanism_explanation")
        self.assertFalse(outcome.exact_answer_reuse_used)
        self.assertFalse(outcome.strategy_assist_used)
        self.assertFalse(outcome.finalizable)

    def test_v2_strategy_assists_mechanism_question_without_answer_reuse(self):
        graph = _dijkstra_explanation_graph()
        outcome = run_micro_epistemic_controller(
            question="How does Dijkstra work?",
            graph=graph,
            anchor_ids=["strat_dijkstra_mechanism_v2"],
        )

        self.assertTrue(outcome.finalizable)
        self.assertTrue(outcome.strategy_assist_used)
        self.assertFalse(outcome.exact_answer_reuse_used)
        self.assertIn("mechanism", outcome.slot_values)
        self.assertIn("priority queue", compose_answer_from_slots(outcome).lower())


if __name__ == "__main__":
    unittest.main()
