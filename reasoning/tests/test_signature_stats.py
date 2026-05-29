from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from typing import Any, Dict, List, Optional

from graph_core import Node as GraphNode
from reasoning.signature_stats import (
    SignatureCandidate,
    SignatureVariantStats,
    _passes_live_bias_relevance_gate,
    build_shadow_report,
    judge_variant_relation,
    load_live_signature_bias_plan,
    load_signature_stats_index,
    run_signature_shadow_session,
    score_event_impact,
)


class TestSignatureStats(unittest.TestCase):
    def test_run_signature_shadow_session_writes_index_and_events(self) -> None:
        graph_edits = [
            {
                "op": "add_node",
                "node_id": "strat_demo",
                "node_type": "strategy",
                "text": "Strategy family: algorithm_applicability\nSlot order: verdict, reason, alternative",
                "metadata": {
                    "task_family": "algorithm_applicability",
                    "task_subtype": "algorithm_mechanism_explanation",
                    "question_mode": "judgment",
                    "domain_keywords": ["dijkstra", "negative", "weights"],
                    "slot_order": ["verdict", "reason", "alternative"],
                    "key_node_ids": ["dijkstra_neg_edges_failure"],
                    "plan_template": ["Read failure node", "Answer"],
                    "checkpoint_plan": ["Read failure node", "Answer"],
                    "entry_conditions": {"algorithm": "Dijkstra", "condition": "negative edge weights"},
                },
            },
            {
                "op": "add_node",
                "node_id": "ssg_demo",
                "node_type": "solved_subgoal",
                "text": "Dijkstra is not generally correct when negative edges are present.",
                "metadata": {
                    "summary": "Dijkstra is not generally correct when negative edges are present.",
                    "subgoal_signature": "algorithm_applicability.dijkstra.negative_edge_weights.validity",
                    "question_type": "algorithm_applicability",
                    "task_subtype": "algorithm_mechanism_explanation",
                    "question_mode": "judgment",
                    "output_slots": {
                        "verdict": "not guaranteed",
                        "reason": "negative edges break the finalized-distance invariant",
                        "alternative": "Bellman-Ford",
                    },
                    "supporting_node_ids": ["dijkstra_neg_edges_failure", "bellman_ford_apply"],
                    "valid_when": ["asking about standard Dijkstra"],
                    "invalid_when": ["asking about modified variants"],
                    "input_conditions": {"algorithm": "Dijkstra", "condition": "negative edge weights"},
                },
            },
        ]
        scoped_patches = [
            {
                "patch_id": "patch_strategy",
                "patch_type": "add_strategy",
                "target_id": "strat_demo",
                "evidence_node_ids": ["dijkstra_neg_edges_failure", "shortest_path_grid_apply"],
                "affected_node_ids": ["strat_demo"],
                "validation": {
                    "status": "needs_review",
                    "reasons": ["strategy has low-relevance key evidence"],
                    "warnings": ["low_relevance_evidence:shortest_path_grid_apply"],
                },
            },
            {
                "patch_id": "patch_subgoal",
                "patch_type": "add_solved_subgoal",
                "target_id": "ssg_demo",
                "evidence_node_ids": ["dijkstra_neg_edges_failure", "bellman_ford_apply"],
                "affected_node_ids": ["ssg_demo"],
                "validation": {
                    "status": "accept",
                    "reasons": ["slots are graph-supported"],
                    "warnings": [],
                },
            },
        ]
        hypotheses = {
            "h_1": {
                "text": "Use Redis Cluster with consistent hashing for scale.",
                "verdict": "discarded",
                "evidence": "not supported by the current graph",
            }
        }
        final_answer = (
            "Dijkstra is not generally correct when negative edges are present.\n"
            "One possible implementation detail, not directly supported by the current graph, "
            "is Redis sharding; treat that as a hypothesis until verified."
        )

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            result = run_signature_shadow_session(
                session_id="sess_sig_1",
                question="Can Dijkstra be trusted with one negative edge?",
                task_family="algorithm_applicability",
                graph_edits=graph_edits,
                scoped_patches=scoped_patches,
                hypotheses=hypotheses,
                final_answer=final_answer,
                cited_node_ids=["dijkstra_neg_edges_failure", "bellman_ford_apply"],
                finalized=True,
                execution_mode="micro_controller_finalize",
                design_evidence_gate_rounds=1,
                stats_dir=root,
            )

            event_types = {row["event_type"] for row in result["events"]}
            self.assertIn("supported_reuse", event_types)
            self.assertIn("supported_finalize", event_types)
            self.assertIn("scoped_patch_accept", event_types)
            self.assertIn("scoped_patch_needs_review", event_types)
            self.assertIn("low_relevance_retrieval", event_types)
            self.assertIn("hypothesis_discarded", event_types)
            self.assertIn("provisional_used_with_caveat", event_types)
            self.assertIn("answer_gate_rewrite", event_types)
            self.assertGreaterEqual(result["update_summary"]["candidate_count"], 3)

            index_path = Path(result["update_summary"]["index_path"])
            events_path = Path(result["update_summary"]["events_path"])
            self.assertTrue(index_path.exists())
            self.assertTrue(events_path.exists())

            index = load_signature_stats_index(index_path)
            self.assertGreaterEqual(len(index.families), 3)
            self.assertGreaterEqual(len(index.variants), 3)

            shadow = result["shadow_report"]
            self.assertEqual(shadow["mode"], "shadow_only")
            self.assertTrue(shadow["ranking_complete"])
            self.assertGreaterEqual(shadow["candidate_count"], 3)
            self.assertGreaterEqual(len(shadow["baseline_ranking"]), 3)
            self.assertEqual(shadow["baseline_ranking"][0]["baseline_rank"], 1)
            self.assertEqual(shadow["adjusted_ranking"][0]["adjusted_rank"], 1)
            self.assertGreaterEqual(len(shadow["adjusted_family_ranking"]), 1)
            self.assertTrue(shadow["focus_variants"])
            projection = result["graph_projection"]
            self.assertGreaterEqual(projection["summary"]["family_count"], 1)
            self.assertTrue(any(node["node_type"] == "signature_family" for node in projection["nodes"]))
            self.assertTrue(any(node["node_type"] == "signature_variant" for node in projection["nodes"]))
            self.assertTrue(any(edge["relation"] == "has_variant" for edge in projection["edges"]))
            self.assertTrue(any(edge["relation"] == "supported_by" for edge in projection["edges"]))

    def test_shadow_report_reranks_by_bias(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            first = run_signature_shadow_session(
                session_id="sess_a",
                question="Can Dijkstra be trusted with one negative edge?",
                task_family="algorithm_applicability",
                graph_edits=[
                    {
                        "op": "add_node",
                        "node_id": "ssg_a",
                        "node_type": "solved_subgoal",
                        "text": "Dijkstra is unsafe with negative edges.",
                        "metadata": {
                            "summary": "Dijkstra is unsafe with negative edges.",
                            "subgoal_signature": "algorithm_applicability.dijkstra.negative_edge_weights.validity",
                            "question_type": "algorithm_applicability",
                            "output_slots": {"verdict": "unsafe"},
                            "supporting_node_ids": ["dijkstra_neg_edges_failure"],
                        },
                    }
                ],
                scoped_patches=[
                    {
                        "patch_id": "patch_a",
                        "patch_type": "add_solved_subgoal",
                        "target_id": "ssg_a",
                        "evidence_node_ids": ["dijkstra_neg_edges_failure"],
                        "affected_node_ids": ["ssg_a"],
                        "validation": {"status": "accept", "reasons": [], "warnings": []},
                    }
                ],
                hypotheses={},
                final_answer="Dijkstra is unsafe with negative edges.",
                cited_node_ids=["dijkstra_neg_edges_failure"],
                finalized=True,
                execution_mode="micro_controller_finalize",
                stats_dir=root,
            )
            second = run_signature_shadow_session(
                session_id="sess_b",
                question="How does Dijkstra work?",
                task_family="algorithm_applicability",
                graph_edits=[
                    {
                        "op": "add_node",
                        "node_id": "strat_b",
                        "node_type": "strategy",
                        "text": "Strategy family: algorithm_applicability\nSlot order: mechanism, answer",
                        "metadata": {
                            "task_family": "algorithm_applicability",
                            "domain_keywords": ["dijkstra", "mechanism"],
                            "slot_order": ["mechanism", "answer"],
                            "key_node_ids": ["dijkstra_mechanism_summary"],
                            "plan_template": ["Read summary", "Explain"],
                            "checkpoint_plan": ["Read summary", "Explain"],
                        },
                    }
                ],
                scoped_patches=[
                    {
                        "patch_id": "patch_b",
                        "patch_type": "add_strategy",
                        "target_id": "strat_b",
                        "evidence_node_ids": ["dijkstra_mechanism_summary"],
                        "affected_node_ids": ["strat_b"],
                        "validation": {"status": "needs_review", "reasons": ["too narrow"], "warnings": []},
                    }
                ],
                hypotheses={},
                final_answer="Dijkstra repeatedly relaxes edges using a priority queue.",
                cited_node_ids=["dijkstra_mechanism_summary"],
                finalized=True,
                execution_mode="full_loop",
                stats_dir=root,
            )
            index = load_signature_stats_index(Path(second["update_summary"]["index_path"]))
            report = build_shadow_report(
                index=index,
                question="Can Dijkstra be trusted with one negative edge?",
                task_family="algorithm_applicability",
                focus_variant_ids=[],
            )
            self.assertEqual(report["mode"], "shadow_only")
            self.assertTrue(report["ranking_complete"])
            self.assertGreaterEqual(len(report["adjusted_top_k"]), 1)
            self.assertEqual(report["adjusted_ranking"][0]["adjusted_rank"], 1)
            top = report["adjusted_top_k"][0]
            self.assertEqual(top["semantic_type"], "solved_subgoal")

    def test_equivalent_revision_reuses_existing_variant(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            first = run_signature_shadow_session(
                session_id="sess_rev_a",
                question="Can Dijkstra be trusted with one negative edge?",
                task_family="algorithm_applicability",
                graph_edits=[
                    {
                        "op": "add_node",
                        "node_id": "ssg_rev_a",
                        "node_type": "solved_subgoal",
                        "text": "Dijkstra is unsafe with negative edges.",
                        "metadata": {
                            "summary": "Dijkstra is unsafe with negative edges.",
                            "subgoal_signature": "algorithm_applicability.dijkstra.negative_edge_weights.validity",
                            "question_type": "algorithm_applicability",
                            "output_slots": {
                                "verdict": "unsafe",
                                "reason": "negative edges break the finalized-distance invariant",
                            },
                            "supporting_node_ids": ["dijkstra_neg_edges_failure"],
                            "input_conditions": {"algorithm": "Dijkstra", "condition": "negative edge weights"},
                            "valid_when": ["asking about standard Dijkstra"],
                            "invalid_when": [],
                        },
                    }
                ],
                scoped_patches=[
                    {
                        "patch_id": "patch_rev_a",
                        "patch_type": "add_solved_subgoal",
                        "target_id": "ssg_rev_a",
                        "evidence_node_ids": ["dijkstra_neg_edges_failure"],
                        "affected_node_ids": ["ssg_rev_a"],
                        "validation": {"status": "accept", "reasons": [], "warnings": []},
                    }
                ],
                hypotheses={},
                final_answer="Dijkstra is unsafe with negative edges.",
                cited_node_ids=["dijkstra_neg_edges_failure"],
                finalized=True,
                execution_mode="micro_controller_finalize",
                stats_dir=root,
            )
            second = run_signature_shadow_session(
                session_id="sess_rev_b",
                question="Can standard Dijkstra handle negative edge weights?",
                task_family="algorithm_applicability",
                graph_edits=[
                    {
                        "op": "add_node",
                        "node_id": "ssg_rev_b",
                        "node_type": "solved_subgoal",
                        "text": "Standard Dijkstra is not reliable when negative edges exist.",
                        "metadata": {
                            "summary": "Standard Dijkstra is not reliable when negative edges exist.",
                            "subgoal_signature": "algorithm_applicability.dijkstra.negative_edge_weights.validity",
                            "question_type": "algorithm_applicability",
                            "output_slots": {
                                "verdict": "unsafe",
                                "reason": "negative edges break the finalized-distance invariant",
                            },
                            "supporting_node_ids": ["dijkstra_neg_edges_failure"],
                            "input_conditions": {"algorithm": "Dijkstra", "condition": "negative edge weights"},
                            "valid_when": ["asking about standard Dijkstra"],
                            "invalid_when": [],
                        },
                    }
                ],
                scoped_patches=[
                    {
                        "patch_id": "patch_rev_b",
                        "patch_type": "add_solved_subgoal",
                        "target_id": "ssg_rev_b",
                        "evidence_node_ids": ["dijkstra_neg_edges_failure"],
                        "affected_node_ids": ["ssg_rev_b"],
                        "validation": {"status": "accept", "reasons": [], "warnings": []},
                    }
                ],
                hypotheses={},
                final_answer="Standard Dijkstra is not reliable when negative edges exist.",
                cited_node_ids=["dijkstra_neg_edges_failure"],
                finalized=True,
                execution_mode="micro_controller_finalize",
                stats_dir=root,
            )
            first_variant = first["candidates"][0]["variant_id"]
            second_candidate = second["candidates"][0]
            self.assertEqual(second_candidate["variant_id"], first_variant)
            self.assertEqual(second_candidate["variant_resolution"], "equivalent_revision")
            index = load_signature_stats_index(Path(second["update_summary"]["index_path"]))
            self.assertEqual(len(index.variants), 1)

    def test_overlapping_strategy_keeps_sibling_variant(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            run_signature_shadow_session(
                session_id="sess_sib_a",
                question="Can Dijkstra be trusted with one negative edge?",
                task_family="algorithm_applicability",
                graph_edits=[
                    {
                        "op": "add_node",
                        "node_id": "strat_sib_a",
                        "node_type": "strategy",
                        "text": "Strategy family: algorithm_applicability\nCheckpoint plan:\n1. Read failure node\n2. Answer with Bellman-Ford alternative",
                        "metadata": {
                            "task_family": "algorithm_applicability",
                            "task_subtype": "algorithm_mechanism_explanation",
                            "question_mode": "judgment",
                            "domain_keywords": ["dijkstra", "negative", "weights"],
                            "slot_order": ["verdict", "reason", "alternative"],
                            "key_node_ids": ["dijkstra_neg_edges_failure"],
                            "plan_template": ["Read failure node", "Answer with Bellman-Ford alternative"],
                            "checkpoint_plan": ["Read failure node", "Answer with Bellman-Ford alternative"],
                            "entry_conditions": {"algorithm": "Dijkstra", "condition": "negative edge weights"},
                        },
                    }
                ],
                scoped_patches=[
                    {
                        "patch_id": "patch_sib_a",
                        "patch_type": "add_strategy",
                        "target_id": "strat_sib_a",
                        "evidence_node_ids": ["dijkstra_neg_edges_failure"],
                        "affected_node_ids": ["strat_sib_a"],
                        "validation": {"status": "accept", "reasons": [], "warnings": []},
                    }
                ],
                hypotheses={},
                final_answer="Use Bellman-Ford when negative edges exist.",
                cited_node_ids=["dijkstra_neg_edges_failure"],
                finalized=True,
                execution_mode="micro_controller_finalize",
                stats_dir=root,
            )
            second = run_signature_shadow_session(
                session_id="sess_sib_b",
                question="Can Dijkstra be trusted if a graph has a negative edge?",
                task_family="algorithm_applicability",
                graph_edits=[
                    {
                        "op": "add_node",
                        "node_id": "strat_sib_b",
                        "node_type": "strategy",
                        "text": "Strategy family: algorithm_applicability\nCheckpoint plan:\n1. Read counterexample\n2. Explain invariant break\n3. Offer Bellman-Ford",
                        "metadata": {
                            "task_family": "algorithm_applicability",
                            "task_subtype": "algorithm_mechanism_explanation",
                            "question_mode": "judgment",
                            "domain_keywords": ["dijkstra", "negative", "edge"],
                            "slot_order": ["verdict", "reason", "alternative"],
                            "key_node_ids": ["dijkstra_neg_edges_failure"],
                            "plan_template": ["Read counterexample", "Explain invariant break", "Offer Bellman-Ford"],
                            "checkpoint_plan": ["Read counterexample", "Explain invariant break", "Offer Bellman-Ford"],
                            "entry_conditions": {"algorithm": "Dijkstra", "condition": "negative edge weights"},
                        },
                    }
                ],
                scoped_patches=[
                    {
                        "patch_id": "patch_sib_b",
                        "patch_type": "add_strategy",
                        "target_id": "strat_sib_b",
                        "evidence_node_ids": ["dijkstra_neg_edges_failure"],
                        "affected_node_ids": ["strat_sib_b"],
                        "validation": {"status": "accept", "reasons": [], "warnings": []},
                    }
                ],
                hypotheses={},
                final_answer="Negative edges break the invariant, so use Bellman-Ford.",
                cited_node_ids=["dijkstra_neg_edges_failure"],
                finalized=True,
                execution_mode="micro_controller_finalize",
                stats_dir=root,
            )
            candidate = second["candidates"][0]
            self.assertEqual(candidate["family_resolution"], "family_alias")
            self.assertEqual(candidate["variant_resolution"], "sibling_variant")
            self.assertIn(candidate["relation_to_match"], {"overlaps", "entails"})
            index = load_signature_stats_index(Path(second["update_summary"]["index_path"]))
            self.assertEqual(len(index.families), 1)
            self.assertEqual(len(index.variants), 2)
            self.assertEqual(len(index.relations), 1)
            self.assertTrue(second["update_summary"]["created_relation_ids"])
            propagated_variants = [variant for variant in index.variants.values() if variant.propagated_support_score > 0.0]
            self.assertTrue(propagated_variants)
            projection = second["graph_projection"]
            self.assertEqual(projection["summary"]["family_count"], 1)
            self.assertEqual(sum(1 for node in projection["nodes"] if node["node_type"] == "signature_variant"), 2)
            self.assertGreaterEqual(projection["summary"]["relation_edge_count"], 1)
            self.assertTrue(any(edge["relation"] in {"overlaps", "entails"} for edge in projection["edges"]))

    def test_repeated_solved_subgoal_promotes_to_supported(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            first = run_signature_shadow_session(
                session_id="sess_promote_ssg_a",
                question="Can Dijkstra be trusted with one negative edge?",
                task_family="algorithm_applicability",
                graph_edits=[
                    {
                        "op": "add_node",
                        "node_id": "ssg_promote_a",
                        "node_type": "solved_subgoal",
                        "text": "Standard Dijkstra is not generally correct when negative edge weights are present.",
                        "metadata": {
                            "summary": "Standard Dijkstra is not generally correct when negative edge weights are present.",
                            "subgoal_signature": "shortest_path.dijkstra.negative_edge_weights.validity",
                            "question_type": "algorithm_applicability",
                            "output_slots": {
                                "verdict": "no",
                                "reason": "negative edges break the greedy guarantee",
                                "alternative": "Bellman-Ford",
                            },
                            "supporting_node_ids": [
                                "dijkstra_requires_nonnegative_edge_weights",
                                "bellman_ford_handles_negative_edges",
                            ],
                            "input_conditions": {"algorithm": "Dijkstra", "condition": "negative edge weights"},
                            "valid_when": ["asking about standard Dijkstra"],
                            "invalid_when": [],
                        },
                    }
                ],
                scoped_patches=[
                    {
                        "patch_id": "patch_promote_ssg_a",
                        "patch_type": "add_solved_subgoal",
                        "target_id": "ssg_promote_a",
                        "evidence_node_ids": [
                            "dijkstra_requires_nonnegative_edge_weights",
                            "bellman_ford_handles_negative_edges",
                        ],
                        "affected_node_ids": ["ssg_promote_a"],
                        "validation": {"status": "accept", "reasons": [], "warnings": []},
                    }
                ],
                hypotheses={},
                final_answer="Standard Dijkstra is not generally correct when negative edge weights are present. Use Bellman-Ford instead.",
                cited_node_ids=[
                    "dijkstra_requires_nonnegative_edge_weights",
                    "bellman_ford_handles_negative_edges",
                ],
                finalized=True,
                execution_mode="micro_controller_finalize",
                stats_dir=root,
            )
            second = run_signature_shadow_session(
                session_id="sess_promote_ssg_b",
                question="Can standard Dijkstra handle negative edge weights?",
                task_family="algorithm_applicability",
                graph_edits=[
                    {
                        "op": "add_node",
                        "node_id": "ssg_promote_b",
                        "node_type": "solved_subgoal",
                        "text": "Standard Dijkstra is not generally correct when negative edge weights are present.",
                        "metadata": {
                            "summary": "Standard Dijkstra is not generally correct when negative edge weights are present.",
                            "subgoal_signature": "shortest_path.dijkstra.negative_edge_weights.validity",
                            "question_type": "algorithm_applicability",
                            "output_slots": {
                                "verdict": "no",
                                "reason": "negative edges break the greedy guarantee",
                                "alternative": "Bellman-Ford",
                            },
                            "supporting_node_ids": [
                                "dijkstra_requires_nonnegative_edge_weights",
                                "bellman_ford_handles_negative_edges",
                            ],
                            "input_conditions": {"algorithm": "Dijkstra", "condition": "negative edge weights"},
                            "valid_when": ["asking about standard Dijkstra"],
                            "invalid_when": [],
                        },
                    }
                ],
                scoped_patches=[
                    {
                        "patch_id": "patch_promote_ssg_b",
                        "patch_type": "add_solved_subgoal",
                        "target_id": "ssg_promote_b",
                        "evidence_node_ids": [
                            "dijkstra_requires_nonnegative_edge_weights",
                            "bellman_ford_handles_negative_edges",
                        ],
                        "affected_node_ids": ["ssg_promote_b"],
                        "validation": {"status": "accept", "reasons": [], "warnings": []},
                    }
                ],
                hypotheses={},
                final_answer="Standard Dijkstra is not generally correct when negative edge weights are present. Use Bellman-Ford instead.",
                cited_node_ids=[
                    "dijkstra_requires_nonnegative_edge_weights",
                    "bellman_ford_handles_negative_edges",
                ],
                finalized=True,
                execution_mode="micro_controller_finalize",
                stats_dir=root,
            )
            variant_id = first["candidates"][0]["variant_id"]
            index = load_signature_stats_index(Path(second["update_summary"]["index_path"]))
            variant = index.variants[variant_id]
            self.assertEqual(variant.promotion_state, "supported")
            self.assertEqual(variant.epistemic_status, "supported")
            self.assertEqual(variant.retrieval_tier, "normal")
            self.assertGreaterEqual(second["update_summary"]["promotion_event_count"], 1)
            transition_types = {event["event_type"] for event in second["events"]}
            self.assertIn("promoted_to_supported", transition_types)

    def test_repeated_strategy_promotes_to_review_only(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            run_signature_shadow_session(
                session_id="sess_promote_strat_a",
                question="Can Dijkstra be trusted with one negative edge?",
                task_family="algorithm_applicability",
                graph_edits=[
                    {
                        "op": "add_node",
                        "node_id": "strat_promote_a",
                        "node_type": "strategy",
                        "text": "Strategy family: algorithm_applicability\nCheckpoint plan:\n1. Read failure node\n2. Answer with Bellman-Ford alternative",
                        "metadata": {
                            "task_family": "algorithm_applicability",
                            "task_subtype": "algorithm_applicability",
                            "question_mode": "verdict",
                            "domain_keywords": ["dijkstra", "negative", "weights"],
                            "slot_order": ["verdict", "reason", "alternative"],
                            "key_node_ids": ["dijkstra_requires_nonnegative_edge_weights"],
                            "plan_template": ["Read failure node", "Answer with Bellman-Ford alternative"],
                            "checkpoint_plan": ["Read failure node", "Answer with Bellman-Ford alternative"],
                            "entry_conditions": {"algorithm": "Dijkstra", "condition": "negative edge weights"},
                        },
                    }
                ],
                scoped_patches=[
                    {
                        "patch_id": "patch_promote_strat_a",
                        "patch_type": "add_strategy",
                        "target_id": "strat_promote_a",
                        "evidence_node_ids": ["dijkstra_requires_nonnegative_edge_weights"],
                        "affected_node_ids": ["strat_promote_a"],
                        "validation": {"status": "accept", "reasons": [], "warnings": []},
                    }
                ],
                hypotheses={},
                final_answer="Dijkstra is not reliable with negative edge weights; use Bellman-Ford instead.",
                cited_node_ids=["dijkstra_requires_nonnegative_edge_weights"],
                finalized=True,
                execution_mode="full_loop",
                stats_dir=root,
            )
            second = run_signature_shadow_session(
                session_id="sess_promote_strat_b",
                question="Can Dijkstra be trusted if a graph has a negative edge?",
                task_family="algorithm_applicability",
                graph_edits=[
                    {
                        "op": "add_node",
                        "node_id": "strat_promote_b",
                        "node_type": "strategy",
                        "text": "Strategy family: algorithm_applicability\nCheckpoint plan:\n1. Read failure node\n2. Answer with Bellman-Ford alternative",
                        "metadata": {
                            "task_family": "algorithm_applicability",
                            "task_subtype": "algorithm_applicability",
                            "question_mode": "verdict",
                            "domain_keywords": ["dijkstra", "negative", "weights"],
                            "slot_order": ["verdict", "reason", "alternative"],
                            "key_node_ids": ["dijkstra_requires_nonnegative_edge_weights"],
                            "plan_template": ["Read failure node", "Answer with Bellman-Ford alternative"],
                            "checkpoint_plan": ["Read failure node", "Answer with Bellman-Ford alternative"],
                            "entry_conditions": {"algorithm": "Dijkstra", "condition": "negative edge weights"},
                        },
                    }
                ],
                scoped_patches=[
                    {
                        "patch_id": "patch_promote_strat_b",
                        "patch_type": "add_strategy",
                        "target_id": "strat_promote_b",
                        "evidence_node_ids": ["dijkstra_requires_nonnegative_edge_weights"],
                        "affected_node_ids": ["strat_promote_b"],
                        "validation": {"status": "accept", "reasons": [], "warnings": []},
                    }
                ],
                hypotheses={},
                final_answer="Negative edges break Dijkstra's guarantee, so Bellman-Ford is the safer choice.",
                cited_node_ids=["dijkstra_requires_nonnegative_edge_weights"],
                finalized=True,
                execution_mode="full_loop",
                stats_dir=root,
            )
            variant_id = next(cand["variant_id"] for cand in second["candidates"] if cand["semantic_type"] == "strategy")
            index = load_signature_stats_index(Path(second["update_summary"]["index_path"]))
            variant = index.variants[variant_id]
            self.assertEqual(variant.promotion_state, "review")
            self.assertEqual(variant.epistemic_status, "provisional")
            self.assertEqual(variant.retrieval_tier, "gated")
            transition_types = {event["event_type"] for event in second["events"]}
            self.assertIn("promoted_to_review", transition_types)
            self.assertNotIn("promoted_to_supported", transition_types)

    def test_provisional_claim_recurrence_does_not_auto_promote(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            first = run_signature_shadow_session(
                session_id="sess_promote_claim_a",
                question="Design a leaderboard service.",
                task_family="design_synthesis",
                graph_edits=[],
                scoped_patches=[],
                hypotheses={},
                final_answer="One possible implementation detail, not directly supported by the current graph, is Redis sharding until verified.",
                cited_node_ids=[],
                finalized=True,
                execution_mode="full_loop",
                design_evidence_gate_rounds=1,
                stats_dir=root,
            )
            second = run_signature_shadow_session(
                session_id="sess_promote_claim_b",
                question="Design a high-scale leaderboard architecture.",
                task_family="design_synthesis",
                graph_edits=[],
                scoped_patches=[],
                hypotheses={},
                final_answer="One possible implementation detail, not directly supported by the current graph, is Redis sharding until verified.",
                cited_node_ids=[],
                finalized=True,
                execution_mode="full_loop",
                design_evidence_gate_rounds=1,
                stats_dir=root,
            )
            index = load_signature_stats_index(Path(second["update_summary"]["index_path"]))
            provisional_variants = [variant for variant in index.variants.values() if variant.semantic_type == "provisional_claim"]
            self.assertTrue(provisional_variants)
            self.assertTrue(all(variant.promotion_state != "supported" for variant in provisional_variants))
            self.assertTrue(all(variant.epistemic_status == "provisional" for variant in provisional_variants))
            transition_types = {event["event_type"] for event in second["events"]}
            self.assertNotIn("promoted_to_supported", transition_types)

    def test_contradicting_variants_mark_family_contested(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            run_signature_shadow_session(
                session_id="sess_contra_a",
                question="Why can't astronauts hear sunlight in space?",
                task_family="direct_judgment",
                graph_edits=[
                    {
                        "op": "add_node",
                        "node_id": "ssg_contra_a",
                        "node_type": "solved_subgoal",
                        "text": "Sound cannot travel through vacuum because sound needs a material medium.",
                        "metadata": {
                            "summary": "Sound cannot travel through vacuum because sound needs a material medium.",
                            "subgoal_signature": "direct_judgment.sound_vacuum_hearing",
                            "question_type": "direct_judgment",
                            "question_mode": "why",
                            "output_slots": {
                                "answer": "sound cannot travel through vacuum",
                                "reason": "sound needs a material medium",
                            },
                            "supporting_node_ids": [
                                "wave_sound_medium",
                                "electromagnetic_waves_propagate_in_vacuum",
                            ],
                            "input_conditions": {"topic": "sound in vacuum"},
                            "valid_when": ["asking why sound cannot be heard in space"],
                            "invalid_when": [],
                        },
                    }
                ],
                scoped_patches=[
                    {
                        "patch_id": "patch_contra_a",
                        "patch_type": "add_solved_subgoal",
                        "target_id": "ssg_contra_a",
                        "evidence_node_ids": [
                            "wave_sound_medium",
                            "electromagnetic_waves_propagate_in_vacuum",
                        ],
                        "affected_node_ids": ["ssg_contra_a"],
                        "validation": {"status": "accept", "reasons": [], "warnings": []},
                    }
                ],
                hypotheses={},
                final_answer="Sound cannot travel through vacuum because sound needs a material medium.",
                cited_node_ids=["wave_sound_medium", "electromagnetic_waves_propagate_in_vacuum"],
                finalized=True,
                execution_mode="micro_controller_finalize",
                stats_dir=root,
            )
            second = run_signature_shadow_session(
                session_id="sess_contra_b",
                question="Can sound travel through vacuum in space?",
                task_family="direct_judgment",
                graph_edits=[
                    {
                        "op": "add_node",
                        "node_id": "ssg_contra_b",
                        "node_type": "solved_subgoal",
                        "text": "Sound can travel through vacuum without any material medium.",
                        "metadata": {
                            "summary": "Sound can travel through vacuum without any material medium.",
                            "subgoal_signature": "direct_judgment.sound_vacuum_hearing",
                            "question_type": "direct_judgment",
                            "question_mode": "why",
                            "output_slots": {
                                "answer": "sound can travel through vacuum",
                                "reason": "sound self-propagates without a medium",
                            },
                            "supporting_node_ids": [
                                "wave_sound_medium",
                                "electromagnetic_waves_propagate_in_vacuum",
                            ],
                            "input_conditions": {"topic": "sound in vacuum"},
                            "valid_when": ["asking why sound cannot be heard in space"],
                            "invalid_when": [],
                        },
                    }
                ],
                scoped_patches=[
                    {
                        "patch_id": "patch_contra_b",
                        "patch_type": "add_solved_subgoal",
                        "target_id": "ssg_contra_b",
                        "evidence_node_ids": [
                            "wave_sound_medium",
                            "electromagnetic_waves_propagate_in_vacuum",
                        ],
                        "affected_node_ids": ["ssg_contra_b"],
                        "validation": {"status": "accept", "reasons": [], "warnings": []},
                    }
                ],
                hypotheses={},
                final_answer="Sound can travel through vacuum without any material medium.",
                cited_node_ids=["wave_sound_medium", "electromagnetic_waves_propagate_in_vacuum"],
                finalized=True,
                execution_mode="micro_controller_finalize",
                stats_dir=root,
            )
            candidate = second["candidates"][0]
            self.assertEqual(candidate["variant_resolution"], "sibling_variant")
            self.assertEqual(candidate["relation_to_match"], "contradicts")
            index = load_signature_stats_index(Path(second["update_summary"]["index_path"]))
            self.assertEqual(len(index.relations), 1)
            relation = next(iter(index.relations.values()))
            self.assertEqual(relation.relation_type, "contradicts")
            family = next(iter(index.families.values()))
            self.assertTrue(family.contested)
            self.assertGreater(family.effective_contradiction_score, family.contradiction_score)
            gated_variants = [variant for variant in index.variants.values() if variant.retrieval_tier in {"gated", "audit_only"}]
            self.assertTrue(gated_variants)
            projection = second["graph_projection"]
            self.assertTrue(any(edge["relation"] == "contradicts" for edge in projection["edges"]))
            report = build_shadow_report(
                index=index,
                question="Why can't astronauts hear sunlight in space?",
                task_family="direct_judgment",
                focus_variant_ids=[],
            )
            contested_rows = [row for row in report["adjusted_family_ranking"] if row["family_id"] == family.id]
            self.assertTrue(contested_rows)
            self.assertTrue(contested_rows[0]["family_contested"])

    def test_entailment_relation_points_from_specific_variant_to_general_variant(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            first = run_signature_shadow_session(
                session_id="sess_entail_a",
                question="Can standard Dijkstra handle negative edges?",
                task_family="algorithm_applicability",
                graph_edits=[
                    {
                        "op": "add_node",
                        "node_id": "ssg_entail_a",
                        "node_type": "solved_subgoal",
                        "text": "Standard Dijkstra is not correct when negative edge weights are present.",
                        "metadata": {
                            "summary": "Standard Dijkstra is not correct when negative edge weights are present.",
                            "subgoal_signature": "shortest_path.dijkstra.negative_edge_weights.validity",
                            "question_type": "algorithm_applicability",
                            "output_slots": {
                                "verdict": "no",
                                "reason": "negative edges break the greedy guarantee",
                            },
                            "supporting_node_ids": ["dijkstra_requires_nonnegative_edge_weights"],
                            "input_conditions": {"algorithm": "Dijkstra", "condition": "negative edge weights"},
                            "valid_when": ["asking about standard Dijkstra"],
                            "invalid_when": [],
                        },
                    }
                ],
                scoped_patches=[
                    {
                        "patch_id": "patch_entail_a",
                        "patch_type": "add_solved_subgoal",
                        "target_id": "ssg_entail_a",
                        "evidence_node_ids": ["dijkstra_requires_nonnegative_edge_weights"],
                        "affected_node_ids": ["ssg_entail_a"],
                        "validation": {"status": "accept", "reasons": [], "warnings": []},
                    }
                ],
                hypotheses={},
                final_answer="Standard Dijkstra is not correct when negative edge weights are present.",
                cited_node_ids=["dijkstra_requires_nonnegative_edge_weights"],
                finalized=True,
                execution_mode="micro_controller_finalize",
                stats_dir=root,
            )
            second = run_signature_shadow_session(
                session_id="sess_entail_b",
                question="Can Dijkstra be trusted with one negative edge?",
                task_family="algorithm_applicability",
                graph_edits=[
                    {
                        "op": "add_node",
                        "node_id": "ssg_entail_b",
                        "node_type": "solved_subgoal",
                        "text": (
                            "Standard Dijkstra is not correct when negative edge weights are present, "
                            "because negative edges break the greedy guarantee; use Bellman-Ford instead."
                        ),
                        "metadata": {
                            "summary": (
                                "Standard Dijkstra is not correct when negative edge weights are present, "
                                "because negative edges break the greedy guarantee; use Bellman-Ford instead."
                            ),
                            "subgoal_signature": "shortest_path.dijkstra.negative_edge_weights.validity",
                            "question_type": "algorithm_applicability",
                            "output_slots": {
                                "verdict": "no",
                                "reason": "negative edges break the greedy guarantee",
                                "alternative": "Bellman-Ford",
                                "caveat": "may still work on some instances but not generally",
                            },
                            "supporting_node_ids": [
                                "dijkstra_requires_nonnegative_edge_weights",
                                "bellman_ford_handles_negative_edges",
                            ],
                            "input_conditions": {"algorithm": "Dijkstra", "condition": "negative edge weights"},
                            "valid_when": ["asking about standard Dijkstra"],
                            "invalid_when": [],
                        },
                    }
                ],
                scoped_patches=[
                    {
                        "patch_id": "patch_entail_b",
                        "patch_type": "add_solved_subgoal",
                        "target_id": "ssg_entail_b",
                        "evidence_node_ids": [
                            "dijkstra_requires_nonnegative_edge_weights",
                            "bellman_ford_handles_negative_edges",
                        ],
                        "affected_node_ids": ["ssg_entail_b"],
                        "validation": {"status": "accept", "reasons": [], "warnings": []},
                    }
                ],
                hypotheses={},
                final_answer=(
                    "Standard Dijkstra is not correct when negative edge weights are present, "
                    "because negative edges break the greedy guarantee; use Bellman-Ford instead."
                ),
                cited_node_ids=[
                    "dijkstra_requires_nonnegative_edge_weights",
                    "bellman_ford_handles_negative_edges",
                ],
                finalized=True,
                execution_mode="micro_controller_finalize",
                stats_dir=root,
            )
            index = load_signature_stats_index(Path(second["update_summary"]["index_path"]))
            self.assertEqual(len(index.relations), 1)
            relation = next(iter(index.relations.values()))
            first_variant_id = first["candidates"][0]["variant_id"]
            second_variant_id = second["candidates"][0]["variant_id"]
            self.assertEqual(relation.relation_type, "entails")
            self.assertEqual(relation.src_variant_id, second_variant_id)
            self.assertEqual(relation.dst_variant_id, first_variant_id)
            general_variant = index.variants[first_variant_id]
            specific_variant = index.variants[second_variant_id]
            self.assertGreater(general_variant.propagated_support_score, 0.0)
            self.assertGreater(general_variant.effective_support_score, general_variant.support_score)
            self.assertGreater(specific_variant.propagated_support_score, 0.0)
            projection = second["graph_projection"]
            entail_edges = [edge for edge in projection["edges"] if edge["relation"] == "entails"]
            self.assertEqual(len(entail_edges), 1)

    def test_algorithm_applicability_shadow_prefers_supported_solved_subgoal(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            first = run_signature_shadow_session(
                session_id="sess_alg_pref_a",
                question="Can Dijkstra be trusted with one negative edge?",
                task_family="algorithm_applicability",
                graph_edits=[
                    {
                        "op": "add_node",
                        "node_id": "strat_alg_pref",
                        "node_type": "strategy",
                        "text": (
                            "Strategy family: algorithm_applicability\n"
                            "Checkpoint plan:\n"
                            "1. Read controller-selected evidence and finalize\n"
                            "Slot order: verdict, reason, alternative, caveat"
                        ),
                        "metadata": {
                            "task_family": "algorithm_applicability",
                            "task_subtype": "algorithm_applicability",
                            "question_mode": "verdict",
                            "domain_keywords": ["algorithm", "bellman", "bridge", "dijkstra"],
                            "slot_order": ["verdict", "reason", "alternative", "caveat"],
                            "key_node_ids": [
                                "dijkstra_requires_nonnegative_edge_weights",
                                "negative_edge_diagnostics_to_algorithm_choice_bridge",
                                "bellman_ford_handles_negative_edges",
                            ],
                            "plan_template": ["Read controller-selected evidence and finalize"],
                            "checkpoint_plan": ["Read controller-selected evidence and finalize"],
                            "entry_conditions": {
                                "algorithm": "Dijkstra",
                                "condition": "negative edge weights",
                                "scope": "general_correctness",
                            },
                        },
                    },
                    {
                        "op": "add_node",
                        "node_id": "ssg_alg_pref",
                        "node_type": "solved_subgoal",
                        "text": (
                            "No. Standard Dijkstra is not guaranteed to be correct when negative edge weights are present. "
                            "Use Bellman-Ford instead when negative edge weights are present."
                        ),
                        "metadata": {
                            "summary": (
                                "No. Standard Dijkstra is not guaranteed to be correct when negative edge weights are present. "
                                "Use Bellman-Ford instead when negative edge weights are present."
                            ),
                            "subgoal_signature": "shortest_path.dijkstra.negative_edge_weights.validity",
                            "question_type": "algorithm_applicability",
                            "task_subtype": "algorithm_applicability",
                            "question_mode": "verdict",
                            "output_slots": {
                                "verdict": "no",
                                "reason": "negative edges break Dijkstra's greedy guarantee",
                                "alternative": "Bellman-Ford",
                                "caveat": "may still work on some instances but not generally",
                            },
                            "supporting_node_ids": [
                                "dijkstra_requires_nonnegative_edge_weights",
                                "negative_edge_diagnostics_to_algorithm_choice_bridge",
                                "bellman_ford_handles_negative_edges",
                            ],
                            "valid_when": ["asking about standard Dijkstra"],
                            "invalid_when": [],
                            "input_conditions": {
                                "algorithm": "Dijkstra",
                                "condition": "negative edge weights",
                            },
                        },
                    },
                ],
                scoped_patches=[
                    {
                        "patch_id": "patch_alg_pref_strategy",
                        "patch_type": "add_strategy",
                        "target_id": "strat_alg_pref",
                        "evidence_node_ids": [
                            "dijkstra_requires_nonnegative_edge_weights",
                            "negative_edge_diagnostics_to_algorithm_choice_bridge",
                            "bellman_ford_handles_negative_edges",
                        ],
                        "affected_node_ids": ["strat_alg_pref"],
                        "validation": {"status": "accept", "reasons": [], "warnings": []},
                    },
                    {
                        "patch_id": "patch_alg_pref_ssg",
                        "patch_type": "add_solved_subgoal",
                        "target_id": "ssg_alg_pref",
                        "evidence_node_ids": [
                            "dijkstra_requires_nonnegative_edge_weights",
                            "negative_edge_diagnostics_to_algorithm_choice_bridge",
                            "bellman_ford_handles_negative_edges",
                        ],
                        "affected_node_ids": ["ssg_alg_pref"],
                        "validation": {"status": "needs_review", "reasons": ["review later"], "warnings": []},
                    },
                ],
                hypotheses={},
                final_answer=(
                    "No. Standard Dijkstra is not guaranteed to be correct when negative edge weights are present. "
                    "Use Bellman-Ford instead."
                ),
                cited_node_ids=[
                    "dijkstra_requires_nonnegative_edge_weights",
                    "negative_edge_diagnostics_to_algorithm_choice_bridge",
                    "bellman_ford_handles_negative_edges",
                ],
                finalized=True,
                execution_mode="micro_controller_finalize",
                stats_dir=root,
            )
            second = run_signature_shadow_session(
                session_id="sess_alg_pref_b",
                question="Can standard Dijkstra handle negative edge weights?",
                task_family="algorithm_applicability",
                graph_edits=[
                    {
                        "op": "add_node",
                        "node_id": "strat_alg_pref_2",
                        "node_type": "strategy",
                        "text": (
                            "Strategy family: algorithm_applicability\n"
                            "Checkpoint plan:\n"
                            "1. Read controller-selected evidence and finalize\n"
                            "Slot order: verdict, reason, alternative, caveat"
                        ),
                        "metadata": {
                            "task_family": "algorithm_applicability",
                            "task_subtype": "algorithm_applicability",
                            "question_mode": "verdict",
                            "domain_keywords": ["algorithm", "bellman", "bridge", "dijkstra"],
                            "slot_order": ["verdict", "reason", "alternative", "caveat"],
                            "key_node_ids": [
                                "dijkstra_requires_nonnegative_edge_weights",
                                "negative_edge_diagnostics_to_algorithm_choice_bridge",
                                "bellman_ford_handles_negative_edges",
                            ],
                            "plan_template": ["Read controller-selected evidence and finalize"],
                            "checkpoint_plan": ["Read controller-selected evidence and finalize"],
                            "entry_conditions": {
                                "algorithm": "Dijkstra",
                                "condition": "negative edge weights",
                                "scope": "general_correctness",
                            },
                        },
                    },
                    {
                        "op": "add_node",
                        "node_id": "ssg_alg_pref_2",
                        "node_type": "solved_subgoal",
                        "text": (
                            "No. Standard Dijkstra is not guaranteed to be correct when negative edge weights are present. "
                            "Use Bellman-Ford instead when negative edge weights are present."
                        ),
                        "metadata": {
                            "summary": (
                                "No. Standard Dijkstra is not guaranteed to be correct when negative edge weights are present. "
                                "Use Bellman-Ford instead when negative edge weights are present."
                            ),
                            "subgoal_signature": "shortest_path.dijkstra.negative_edge_weights.validity",
                            "question_type": "algorithm_applicability",
                            "task_subtype": "algorithm_applicability",
                            "question_mode": "verdict",
                            "output_slots": {
                                "verdict": "no",
                                "reason": "negative edges break Dijkstra's greedy guarantee",
                                "alternative": "Bellman-Ford",
                                "caveat": "may still work on some instances but not generally",
                            },
                            "supporting_node_ids": [
                                "dijkstra_requires_nonnegative_edge_weights",
                                "negative_edge_diagnostics_to_algorithm_choice_bridge",
                                "bellman_ford_handles_negative_edges",
                            ],
                            "valid_when": ["asking about standard Dijkstra"],
                            "invalid_when": [],
                            "input_conditions": {
                                "algorithm": "Dijkstra",
                                "condition": "negative edge weights",
                            },
                        },
                    },
                ],
                scoped_patches=[
                    {
                        "patch_id": "patch_alg_pref_strategy_2",
                        "patch_type": "add_strategy",
                        "target_id": "strat_alg_pref_2",
                        "evidence_node_ids": [
                            "dijkstra_requires_nonnegative_edge_weights",
                            "negative_edge_diagnostics_to_algorithm_choice_bridge",
                            "bellman_ford_handles_negative_edges",
                        ],
                        "affected_node_ids": ["strat_alg_pref_2"],
                        "validation": {"status": "accept", "reasons": [], "warnings": []},
                    },
                    {
                        "patch_id": "patch_alg_pref_ssg_2",
                        "patch_type": "add_solved_subgoal",
                        "target_id": "ssg_alg_pref_2",
                        "evidence_node_ids": [
                            "dijkstra_requires_nonnegative_edge_weights",
                            "negative_edge_diagnostics_to_algorithm_choice_bridge",
                            "bellman_ford_handles_negative_edges",
                        ],
                        "affected_node_ids": ["ssg_alg_pref_2"],
                        "validation": {"status": "needs_review", "reasons": ["review later"], "warnings": []},
                    },
                ],
                hypotheses={},
                final_answer=(
                    "No. Standard Dijkstra is not guaranteed to be correct when negative edge weights are present. "
                    "Use Bellman-Ford instead."
                ),
                cited_node_ids=[
                    "dijkstra_requires_nonnegative_edge_weights",
                    "negative_edge_diagnostics_to_algorithm_choice_bridge",
                    "bellman_ford_handles_negative_edges",
                ],
                finalized=True,
                execution_mode="micro_controller_finalize",
                stats_dir=root,
            )
            solved_family_id = next(
                cand["family_id"]
                for cand in second["candidates"]
                if cand["semantic_type"] == "solved_subgoal"
            )
            top = second["shadow_report"]["adjusted_top_k"][0]
            self.assertEqual(top["semantic_type"], "solved_subgoal")
            self.assertEqual(top["family_id"], solved_family_id)
            self.assertGreater(top["adjusted_score"], top["baseline_score"])
            first_solved = next(cand for cand in first["candidates"] if cand["semantic_type"] == "solved_subgoal")
            second_solved = next(cand for cand in second["candidates"] if cand["semantic_type"] == "solved_subgoal")
            self.assertEqual(first_solved["variant_id"], second_solved["variant_id"])

    def test_live_signature_bias_plan_uses_graph_backed_support_nodes(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            run_signature_shadow_session(
                session_id="sess_live_bias_seed",
                question="Can Dijkstra be trusted with one negative edge?",
                task_family="algorithm_applicability",
                graph_edits=[
                    {
                        "op": "add_node",
                        "node_id": "strat_live_bias",
                        "node_type": "strategy",
                        "text": (
                            "Strategy family: algorithm_applicability\n"
                            "Checkpoint plan:\n"
                            "1. Read controller-selected evidence and finalize"
                        ),
                        "metadata": {
                            "task_family": "algorithm_applicability",
                            "slot_order": ["verdict", "reason", "alternative", "caveat"],
                            "key_node_ids": [
                                "dijkstra_requires_nonnegative_edge_weights",
                                "negative_edge_counterexample_test_apply",
                                "bellman_ford_handles_negative_edges",
                            ],
                            "checkpoint_plan": ["Read controller-selected evidence and finalize"],
                        },
                    },
                    {
                        "op": "add_node",
                        "node_id": "ssg_live_bias",
                        "node_type": "solved_subgoal",
                        "text": (
                            "No. Standard Dijkstra is not guaranteed to be correct when negative edge weights are present. "
                            "Use Bellman-Ford instead."
                        ),
                        "metadata": {
                            "summary": (
                                "No. Standard Dijkstra is not guaranteed to be correct when negative edge weights are present. "
                                "Use Bellman-Ford instead."
                            ),
                            "subgoal_signature": "shortest_path.dijkstra.negative_edge_weights.validity",
                            "question_type": "algorithm_applicability",
                            "output_slots": {
                                "verdict": "no",
                                "reason": "negative edges break Dijkstra's greedy guarantee",
                                "alternative": "Bellman-Ford",
                                "caveat": "may still work on some instances but not generally",
                            },
                            "supporting_node_ids": [
                                "dijkstra_requires_nonnegative_edge_weights",
                                "negative_edge_counterexample_test_apply",
                                "bellman_ford_handles_negative_edges",
                            ],
                            "valid_when": ["asking about standard Dijkstra"],
                            "invalid_when": [],
                            "input_conditions": {
                                "algorithm": "Dijkstra",
                                "condition": "negative edge weights",
                            },
                        },
                    },
                ],
                scoped_patches=[
                    {
                        "patch_id": "patch_live_bias_strategy",
                        "patch_type": "add_strategy",
                        "target_id": "strat_live_bias",
                        "evidence_node_ids": [
                            "dijkstra_requires_nonnegative_edge_weights",
                            "negative_edge_counterexample_test_apply",
                            "bellman_ford_handles_negative_edges",
                        ],
                        "affected_node_ids": ["strat_live_bias"],
                        "validation": {"status": "accept", "reasons": [], "warnings": []},
                    },
                    {
                        "patch_id": "patch_live_bias_ssg",
                        "patch_type": "add_solved_subgoal",
                        "target_id": "ssg_live_bias",
                        "evidence_node_ids": [
                            "dijkstra_requires_nonnegative_edge_weights",
                            "negative_edge_counterexample_test_apply",
                            "bellman_ford_handles_negative_edges",
                        ],
                        "affected_node_ids": ["ssg_live_bias"],
                        "validation": {"status": "accept", "reasons": [], "warnings": []},
                    },
                ],
                hypotheses={},
                final_answer=(
                    "No. Standard Dijkstra is not guaranteed to be correct when negative edge weights are present. "
                    "Use Bellman-Ford instead."
                ),
                cited_node_ids=[
                    "dijkstra_requires_nonnegative_edge_weights",
                    "negative_edge_counterexample_test_apply",
                    "bellman_ford_handles_negative_edges",
                ],
                finalized=True,
                execution_mode="micro_controller_finalize",
                stats_dir=root,
            )

            graph_nodes = {
                "dijkstra_requires_nonnegative_edge_weights": GraphNode(
                    id="dijkstra_requires_nonnegative_edge_weights",
                    node_type="claim",
                    confidence=0.99,
                    text="Dijkstra requires nonnegative edge weights for its greedy settlement logic to remain correct.",
                ),
                "negative_edge_counterexample_test_apply": GraphNode(
                    id="negative_edge_counterexample_test_apply",
                    node_type="application",
                    confidence=0.98,
                    text="A single negative edge can break the settled-distance invariant and make Dijkstra return the wrong path.",
                ),
                "bellman_ford_handles_negative_edges": GraphNode(
                    id="bellman_ford_handles_negative_edges",
                    node_type="claim",
                    confidence=0.98,
                    text="Bellman-Ford is the safe alternative when negative edges exist.",
                ),
            }
            plan = load_live_signature_bias_plan(
                question="Can standard Dijkstra handle negative edge weights?",
                task_family="algorithm_applicability",
                graph_nodes=graph_nodes,
                stats_dir=root,
                max_anchor_ids=4,
            )
            self.assertTrue(plan.enabled)
            self.assertEqual(plan.reason, "supported_solved_subgoal_family")
            self.assertEqual(plan.semantic_type, "solved_subgoal")
            self.assertTrue(plan.anchor_ids)
            self.assertTrue(set(plan.anchor_ids).issubset(set(graph_nodes.keys())))
            self.assertNotIn("ssg_live_bias", plan.anchor_ids)
            self.assertIn("dijkstra_requires_nonnegative_edge_weights", plan.anchor_ids)
            self.assertIn("bellman_ford_handles_negative_edges", plan.anchor_ids)

    def test_direct_judgment_solved_subgoal_family_separates_unrelated_topics(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            run_signature_shadow_session(
                session_id="sess_vacuum_base",
                question="Why can light travel through space but sound cannot?",
                task_family="direct_judgment",
                graph_edits=[
                    {
                        "op": "add_node",
                        "node_id": "ssg_vacuum",
                        "node_type": "solved_subgoal",
                        "text": "Sound waves require a material medium, but electromagnetic waves can travel through vacuum.",
                        "metadata": {
                            "summary": "Sound waves require a material medium, but electromagnetic waves can travel through vacuum.",
                            "subgoal_signature": "direct_judgment.1c7cdd144554",
                            "question_type": "direct_judgment",
                            "task_subtype": "direct_judgment",
                            "question_mode": "answer_reason",
                            "input_conditions": {"algorithm": "", "condition": "", "artifact": ""},
                            "output_slots": {
                                "answer": "Sound waves require a material medium.",
                                "reason": "Electromagnetic waves can travel through vacuum.",
                            },
                            "valid_when": ["direct_judgment"],
                            "invalid_when": [],
                            "supporting_node_ids": ["wave_sound_medium"],
                        },
                    },
                ],
                scoped_patches=[
                    {
                        "patch_id": "patch_vacuum",
                        "patch_type": "add_solved_subgoal",
                        "target_id": "ssg_vacuum",
                        "evidence_node_ids": ["wave_sound_medium"],
                        "affected_node_ids": ["ssg_vacuum"],
                        "validation": {"status": "accept", "reasons": [], "warnings": []},
                    }
                ],
                hypotheses={},
                final_answer="Sound needs a material medium, but light can travel through vacuum.",
                cited_node_ids=["wave_sound_medium"],
                finalized=True,
                execution_mode="micro_controller_finalize",
                stats_dir=root,
            )

            second = run_signature_shadow_session(
                session_id="sess_prism_base",
                question="Why does a prism bend light but not change the light's frequency?",
                task_family="direct_judgment",
                graph_edits=[
                    {
                        "op": "add_node",
                        "node_id": "ssg_prism",
                        "node_type": "solved_subgoal",
                        "text": "When a wave enters a new medium, its frequency is fixed by the source, while speed and wavelength may change.",
                        "metadata": {
                            "summary": "When a wave enters a new medium, its frequency is fixed by the source, while speed and wavelength may change.",
                            "subgoal_signature": "direct_judgment.caa5fa260901",
                            "question_type": "direct_judgment",
                            "task_subtype": "direct_judgment",
                            "question_mode": "answer_reason",
                            "input_conditions": {"algorithm": "", "condition": "", "artifact": ""},
                            "output_slots": {
                                "answer": "When a wave enters a new medium, its frequency is fixed by the source.",
                                "reason": "Refraction occurs because a wave changes speed when it enters a different medium.",
                            },
                            "valid_when": ["direct_judgment"],
                            "invalid_when": [],
                            "supporting_node_ids": ["wave_frequency_fixed_by_source", "refraction_speed_change_media"],
                        },
                    },
                ],
                scoped_patches=[
                    {
                        "patch_id": "patch_prism",
                        "patch_type": "add_solved_subgoal",
                        "target_id": "ssg_prism",
                        "evidence_node_ids": ["wave_frequency_fixed_by_source", "refraction_speed_change_media"],
                        "affected_node_ids": ["ssg_prism"],
                        "validation": {"status": "accept", "reasons": [], "warnings": []},
                    }
                ],
                hypotheses={},
                final_answer="A prism bends light because speed changes at the boundary, while frequency stays fixed by the source.",
                cited_node_ids=["wave_frequency_fixed_by_source", "refraction_speed_change_media"],
                finalized=True,
                execution_mode="micro_controller_finalize",
                stats_dir=root,
            )

            solved_candidate = next(cand for cand in second["candidates"] if cand["semantic_type"] == "solved_subgoal")
            self.assertEqual(solved_candidate["family_resolution"], "new_family")
            self.assertEqual(solved_candidate["variant_resolution"], "new_variant")
            index = load_signature_stats_index(Path(second["update_summary"]["index_path"]))
            solved_families = [fam for fam in index.families.values() if fam.semantic_type == "solved_subgoal"]
            self.assertEqual(len(solved_families), 2)

    def test_direct_judgment_live_bias_gate_allows_ambiguous_multi_support(self) -> None:
        variant = SignatureVariantStats(
            id="sigvar_prism",
            family_id="sigfam_prism",
            semantic_type="solved_subgoal",
            canonical_text="Frequency stays fixed by the source while refraction changes speed.",
            summary_text="Frequency stays fixed by the source while refraction changes speed.",
            task_family="direct_judgment",
            epistemic_status="supported",
            retrieval_tier="normal",
            top_supporting_node_ids=[
                "wave_frequency_fixed_by_source",
                "refraction_speed_change_media",
            ],
        )
        allowed, reason = _passes_live_bias_relevance_gate(
            question="Why doesn't refraction change the frequency of light?",
            task_family="direct_judgment",
            row={
                "baseline_score": 0.14,
                "baseline_rank": 3,
            },
            anchor_rows=[
                {"node_id": "wave_frequency_fixed_by_source", "lexical_score": 0.12},
                {"node_id": "refraction_speed_change_media", "lexical_score": 0.07},
            ],
            variant=variant,
        )
        self.assertTrue(allowed)
        self.assertEqual(reason, "ambiguous_multi_support")

    def test_direct_judgment_live_bias_gate_allows_high_confidence_single_support(self) -> None:
        variant = SignatureVariantStats(
            id="sigvar_vacuum",
            family_id="sigfam_vacuum",
            semantic_type="solved_subgoal",
            canonical_text="Sound needs a medium, but light can travel through vacuum.",
            summary_text="Sound needs a medium, but light can travel through vacuum.",
            task_family="direct_judgment",
            epistemic_status="supported",
            promotion_state="supported",
            retrieval_tier="normal",
            top_supporting_node_ids=["wave_sound_medium"],
            effective_support_score=6.4,
            effective_stability_score=6.9,
            effective_contradiction_score=0.0,
        )
        allowed, reason = _passes_live_bias_relevance_gate(
            question="If astronauts can see sunlight in space, why can't they hear it there?",
            task_family="direct_judgment",
            row={
                "baseline_score": 0.089,
                "baseline_rank": 2,
            },
            anchor_rows=[
                {"node_id": "wave_sound_medium", "lexical_score": 0.097},
            ],
            variant=variant,
        )
        self.assertTrue(allowed)
        self.assertEqual(reason, "high_confidence_single_support")

    def test_direct_judgment_live_bias_gate_allows_explicit_family_review_fast_track(self) -> None:
        variant = SignatureVariantStats(
            id="sigvar_vacuum",
            family_id="sigfam_solved_subgoal.direct_judgment.sound_requires_medium_vs_light_vacuum",
            semantic_type="solved_subgoal",
            canonical_text="Sound needs a medium, but light can travel through vacuum.",
            summary_text="Sound needs a medium, but light can travel through vacuum.",
            task_family="direct_judgment",
            epistemic_status="supported",
            promotion_state="review",
            retrieval_tier="normal",
            top_supporting_node_ids=["wave_sound_medium"],
            effective_support_score=0.75,
            effective_stability_score=0.30,
            effective_contradiction_score=0.0,
        )
        allowed, reason = _passes_live_bias_relevance_gate(
            question="If astronauts can see sunlight in space, why can't they hear it there?",
            task_family="direct_judgment",
            row={
                "baseline_score": 0.089,
                "baseline_rank": 2,
            },
            anchor_rows=[
                {"node_id": "wave_sound_medium", "lexical_score": 0.097},
            ],
            variant=variant,
        )
        self.assertTrue(allowed)
        self.assertEqual(reason, "explicit_family_review_fast_track")

    def test_direct_judgment_live_bias_gate_rejects_family_question_mismatch(self) -> None:
        variant = SignatureVariantStats(
            id="sigvar_vacuum",
            family_id="sigfam_solved_subgoal.direct_judgment_sound_requires_medium_vs_light_vacuum",
            semantic_type="solved_subgoal",
            canonical_text="Sound needs a medium, but light can travel through vacuum.",
            summary_text="Sound needs a medium, but light can travel through vacuum.",
            task_family="direct_judgment",
            epistemic_status="supported",
            promotion_state="supported",
            retrieval_tier="normal",
            top_supporting_node_ids=["wave_sound_medium"],
            effective_support_score=6.4,
            effective_stability_score=6.9,
            effective_contradiction_score=0.0,
        )
        allowed, reason = _passes_live_bias_relevance_gate(
            question="Why does a prism bend light but not change the light's frequency?",
            task_family="direct_judgment",
            row={
                "baseline_score": 0.09,
                "baseline_rank": 2,
            },
            anchor_rows=[
                {"node_id": "wave_sound_medium", "lexical_score": 0.10},
            ],
            variant=variant,
        )
        self.assertFalse(allowed)
        self.assertEqual(reason, "family_question_mismatch")

    def test_live_bias_skips_strong_matching_direct_judgment_family(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            vacuum = run_signature_shadow_session(
                session_id="sess_vacuum_seed",
                question="Why can light travel through space but sound cannot?",
                task_family="direct_judgment",
                graph_edits=[
                    {
                        "op": "add_node",
                        "node_id": "ssg_vacuum",
                        "node_type": "solved_subgoal",
                        "text": "Sound waves require a material medium, but electromagnetic waves can travel through vacuum.",
                        "metadata": {
                            "summary": "Sound waves require a material medium, but electromagnetic waves can travel through vacuum.",
                            "subgoal_signature": "direct_judgment.1c7cdd144554",
                            "question_type": "direct_judgment",
                            "task_subtype": "direct_judgment",
                            "question_mode": "answer_reason",
                            "input_conditions": {"algorithm": "", "condition": "", "artifact": ""},
                            "output_slots": {
                                "answer": "Sound waves require a material medium.",
                                "reason": "Electromagnetic waves can travel through vacuum.",
                            },
                            "valid_when": ["direct_judgment"],
                            "invalid_when": [],
                            "supporting_node_ids": ["wave_sound_medium"],
                        },
                    },
                ],
                scoped_patches=[
                    {
                        "patch_id": "patch_vacuum",
                        "patch_type": "add_solved_subgoal",
                        "target_id": "ssg_vacuum",
                        "evidence_node_ids": ["wave_sound_medium"],
                        "affected_node_ids": ["ssg_vacuum"],
                        "validation": {"status": "accept", "reasons": [], "warnings": []},
                    }
                ],
                hypotheses={},
                final_answer="Sound needs a material medium, but light can travel through vacuum.",
                cited_node_ids=["wave_sound_medium"],
                finalized=True,
                execution_mode="micro_controller_finalize",
                stats_dir=root,
            )
            prism = run_signature_shadow_session(
                session_id="sess_prism_seed",
                question="Why does a prism bend light but not change the light's frequency?",
                task_family="direct_judgment",
                graph_edits=[
                    {
                        "op": "add_node",
                        "node_id": "ssg_prism",
                        "node_type": "solved_subgoal",
                        "text": "When a wave enters a new medium, its frequency is fixed by the source, while speed and wavelength may change.",
                        "metadata": {
                            "summary": "When a wave enters a new medium, its frequency is fixed by the source, while speed and wavelength may change.",
                            "subgoal_signature": "direct_judgment.caa5fa260901",
                            "question_type": "direct_judgment",
                            "task_subtype": "direct_judgment",
                            "question_mode": "answer_reason",
                            "input_conditions": {"algorithm": "", "condition": "", "artifact": ""},
                            "output_slots": {
                                "answer": "When a wave enters a new medium, its frequency is fixed by the source.",
                                "reason": "Refraction occurs because a wave changes speed when it enters a different medium.",
                            },
                            "valid_when": ["direct_judgment"],
                            "invalid_when": [],
                            "supporting_node_ids": ["wave_frequency_fixed_by_source", "refraction_speed_change_media"],
                        },
                    },
                ],
                scoped_patches=[
                    {
                        "patch_id": "patch_prism",
                        "patch_type": "add_solved_subgoal",
                        "target_id": "ssg_prism",
                        "evidence_node_ids": ["wave_frequency_fixed_by_source", "refraction_speed_change_media"],
                        "affected_node_ids": ["ssg_prism"],
                        "validation": {"status": "accept", "reasons": [], "warnings": []},
                    }
                ],
                hypotheses={},
                final_answer="A prism bends light because speed changes at the boundary, while frequency stays fixed by the source.",
                cited_node_ids=["wave_frequency_fixed_by_source", "refraction_speed_change_media"],
                finalized=True,
                execution_mode="micro_controller_finalize",
                stats_dir=root,
            )
            prism_family_id = next(cand["family_id"] for cand in prism["candidates"] if cand["semantic_type"] == "solved_subgoal")
            vacuum_family_id = next(cand["family_id"] for cand in vacuum["candidates"] if cand["semantic_type"] == "solved_subgoal")
            graph_nodes = {
                "wave_sound_medium": GraphNode(
                    id="wave_sound_medium",
                    node_type="fact",
                    confidence=0.99,
                    text="Sound waves require a material medium, but electromagnetic waves can travel through vacuum.",
                ),
                "wave_frequency_fixed_by_source": GraphNode(
                    id="wave_frequency_fixed_by_source",
                    node_type="principle",
                    confidence=0.99,
                    text="When a wave enters a new medium, its frequency is fixed by the source, while speed and wavelength may change.",
                ),
                "refraction_speed_change_media": GraphNode(
                    id="refraction_speed_change_media",
                    node_type="claim",
                    confidence=0.98,
                    text="Refraction occurs because a wave changes speed when it enters a different medium.",
                ),
                "prism_refraction_example": GraphNode(
                    id="prism_refraction_example",
                    node_type="example",
                    confidence=0.97,
                    text="A prism is a standard example of light refraction.",
                ),
            }
            plan = load_live_signature_bias_plan(
                question="Why doesn't refraction change the frequency of light?",
                task_family="direct_judgment",
                graph_nodes=graph_nodes,
                stats_dir=root,
                max_anchor_ids=4,
            )
            self.assertFalse(plan.enabled)
            self.assertEqual(plan.reason, "no_supported_graph_backed_solved_subgoal")
            self.assertEqual(plan.anchor_ids, [])
            self.assertTrue(plan.skipped_candidates)
            prism_skip = next(
                skipped
                for skipped in plan.skipped_candidates
                if skipped.get("family_id") == prism_family_id
            )
            self.assertEqual(prism_skip["reason"], "insufficient_live_relevance")
            self.assertIn("already_strong_baseline", prism_skip["gate_reason"])
            skipped_family_ids = {str(skipped.get("family_id", "")) for skipped in plan.skipped_candidates}
            self.assertIn(prism_family_id, skipped_family_ids)
            self.assertIn(vacuum_family_id, skipped_family_ids)

    def test_live_bias_rejects_weak_direct_judgment_family_carryover(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            run_signature_shadow_session(
                session_id="sess_vacuum_seed",
                question="Why can light travel through space but sound cannot?",
                task_family="direct_judgment",
                graph_edits=[
                    {
                        "op": "add_node",
                        "node_id": "ssg_vacuum",
                        "node_type": "solved_subgoal",
                        "text": "Sound waves require a material medium, but electromagnetic waves can travel through vacuum.",
                        "metadata": {
                            "summary": "Sound waves require a material medium, but electromagnetic waves can travel through vacuum.",
                            "subgoal_signature": "direct_judgment.1c7cdd144554",
                            "question_type": "direct_judgment",
                            "task_subtype": "direct_judgment",
                            "question_mode": "answer_reason",
                            "input_conditions": {"algorithm": "", "condition": "", "artifact": ""},
                            "output_slots": {
                                "answer": "Sound waves require a material medium.",
                                "reason": "Electromagnetic waves can travel through vacuum.",
                            },
                            "valid_when": ["direct_judgment"],
                            "invalid_when": [],
                            "supporting_node_ids": ["wave_sound_medium"],
                        },
                    },
                ],
                scoped_patches=[
                    {
                        "patch_id": "patch_vacuum",
                        "patch_type": "add_solved_subgoal",
                        "target_id": "ssg_vacuum",
                        "evidence_node_ids": ["wave_sound_medium"],
                        "affected_node_ids": ["ssg_vacuum"],
                        "validation": {"status": "accept", "reasons": [], "warnings": []},
                    }
                ],
                hypotheses={},
                final_answer="Sound needs a material medium, but light can travel through vacuum.",
                cited_node_ids=["wave_sound_medium"],
                finalized=True,
                execution_mode="micro_controller_finalize",
                stats_dir=root,
            )
            graph_nodes = {
                "wave_sound_medium": GraphNode(
                    id="wave_sound_medium",
                    node_type="fact",
                    confidence=0.99,
                    text="Sound waves require a material medium, but electromagnetic waves can travel through vacuum.",
                ),
                "wave_frequency_fixed_by_source": GraphNode(
                    id="wave_frequency_fixed_by_source",
                    node_type="principle",
                    confidence=0.99,
                    text="When a wave enters a new medium, its frequency is fixed by the source, while speed and wavelength may change.",
                ),
                "refraction_speed_change_media": GraphNode(
                    id="refraction_speed_change_media",
                    node_type="claim",
                    confidence=0.98,
                    text="Refraction occurs because a wave changes speed when it enters a different medium.",
                ),
                "prism_refraction_example": GraphNode(
                    id="prism_refraction_example",
                    node_type="example",
                    confidence=0.97,
                    text="A prism is a standard example of light refraction.",
                ),
            }
            plan = load_live_signature_bias_plan(
                question="Why does a prism bend light but not change the light's frequency?",
                task_family="direct_judgment",
                graph_nodes=graph_nodes,
                stats_dir=root,
                max_anchor_ids=4,
            )
            self.assertFalse(plan.enabled)
            self.assertEqual(plan.reason, "no_supported_graph_backed_solved_subgoal")
            self.assertEqual(plan.anchor_ids, [])
            self.assertTrue(plan.skipped_candidates)
            skipped = plan.skipped_candidates[0]
            self.assertEqual(skipped["reason"], "insufficient_live_relevance")
            self.assertEqual(skipped["family_id"], "sigfam_solved_subgoal.direct_judgment_1c7cdd144554")
            self.assertIn("already_strong_baseline", skipped["gate_reason"])

    def test_live_bias_rejects_weak_direct_judgment_carryover_even_for_related_paraphrase(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            run_signature_shadow_session(
                session_id="sess_vacuum_seed",
                question="Why can light travel through space but sound cannot?",
                task_family="direct_judgment",
                graph_edits=[
                    {
                        "op": "add_node",
                        "node_id": "ssg_vacuum",
                        "node_type": "solved_subgoal",
                        "text": "Sound waves require a material medium, but electromagnetic waves can travel through vacuum.",
                        "metadata": {
                            "summary": "Sound waves require a material medium, but electromagnetic waves can travel through vacuum.",
                            "subgoal_signature": "direct_judgment.1c7cdd144554",
                            "question_type": "direct_judgment",
                            "task_subtype": "direct_judgment",
                            "question_mode": "answer_reason",
                            "input_conditions": {"algorithm": "", "condition": "", "artifact": ""},
                            "output_slots": {
                                "answer": "Sound waves require a material medium.",
                                "reason": "Electromagnetic waves can travel through vacuum.",
                            },
                            "valid_when": ["direct_judgment"],
                            "invalid_when": [],
                            "supporting_node_ids": ["wave_sound_medium"],
                        },
                    },
                ],
                scoped_patches=[
                    {
                        "patch_id": "patch_vacuum",
                        "patch_type": "add_solved_subgoal",
                        "target_id": "ssg_vacuum",
                        "evidence_node_ids": ["wave_sound_medium"],
                        "affected_node_ids": ["ssg_vacuum"],
                        "validation": {"status": "accept", "reasons": [], "warnings": []},
                    }
                ],
                hypotheses={},
                final_answer="Sound needs a material medium, but light can travel through vacuum.",
                cited_node_ids=["wave_sound_medium"],
                finalized=True,
                execution_mode="micro_controller_finalize",
                stats_dir=root,
            )
            graph_nodes = {
                "wave_sound_medium": GraphNode(
                    id="wave_sound_medium",
                    node_type="fact",
                    confidence=0.99,
                    text="Sound waves require a material medium, but electromagnetic waves can travel through vacuum.",
                ),
                "electromagnetic_waves_propagate_in_vacuum": GraphNode(
                    id="electromagnetic_waves_propagate_in_vacuum",
                    node_type="claim",
                    confidence=0.99,
                    text="Electromagnetic waves do not require a material medium and can propagate through vacuum.",
                ),
                "visible_light_electromagnetic_wave": GraphNode(
                    id="visible_light_electromagnetic_wave",
                    node_type="claim",
                    confidence=0.98,
                    text="Sunlight is visible electromagnetic radiation.",
                ),
            }
            plan = load_live_signature_bias_plan(
                question="If astronauts can see sunlight in space, why can't they hear it there?",
                task_family="direct_judgment",
                graph_nodes=graph_nodes,
                stats_dir=root,
                max_anchor_ids=4,
            )
            self.assertFalse(plan.enabled)
            self.assertEqual(plan.reason, "no_supported_graph_backed_solved_subgoal")
            self.assertEqual(plan.anchor_ids, [])
            self.assertTrue(plan.skipped_candidates)
            skipped = plan.skipped_candidates[0]
            self.assertEqual(skipped["reason"], "insufficient_live_relevance")
            self.assertIn("already_strong_baseline", skipped["gate_reason"])

    def test_live_bias_rejects_unrelated_multi_support_direct_judgment_family(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            run_signature_shadow_session(
                session_id="sess_prism_seed",
                question="Why does a prism bend light but not change the light's frequency?",
                task_family="direct_judgment",
                graph_edits=[
                    {
                        "op": "add_node",
                        "node_id": "ssg_prism",
                        "node_type": "solved_subgoal",
                        "text": "When a wave enters a new medium, its frequency is fixed by the source, while speed and wavelength may change.",
                        "metadata": {
                            "summary": "When a wave enters a new medium, its frequency is fixed by the source, while speed and wavelength may change.",
                            "subgoal_signature": "direct_judgment.caa5fa260901",
                            "question_type": "direct_judgment",
                            "task_subtype": "direct_judgment",
                            "question_mode": "answer_reason",
                            "input_conditions": {"algorithm": "", "condition": "", "artifact": ""},
                            "output_slots": {
                                "answer": "When a wave enters a new medium, its frequency is fixed by the source.",
                                "reason": "Refraction occurs because a wave changes speed when it enters a different medium.",
                            },
                            "valid_when": ["direct_judgment"],
                            "invalid_when": [],
                            "supporting_node_ids": ["wave_frequency_fixed_by_source", "refraction_speed_change_media"],
                        },
                    },
                ],
                scoped_patches=[
                    {
                        "patch_id": "patch_prism",
                        "patch_type": "add_solved_subgoal",
                        "target_id": "ssg_prism",
                        "evidence_node_ids": ["wave_frequency_fixed_by_source", "refraction_speed_change_media"],
                        "affected_node_ids": ["ssg_prism"],
                        "validation": {"status": "accept", "reasons": [], "warnings": []},
                    }
                ],
                hypotheses={},
                final_answer="A prism bends light because speed changes at the boundary, while frequency stays fixed by the source.",
                cited_node_ids=["wave_frequency_fixed_by_source", "refraction_speed_change_media"],
                finalized=True,
                execution_mode="micro_controller_finalize",
                stats_dir=root,
            )
            graph_nodes = {
                "wave_frequency_fixed_by_source": GraphNode(
                    id="wave_frequency_fixed_by_source",
                    node_type="principle",
                    confidence=0.99,
                    text="When a wave enters a new medium, its frequency is fixed by the source, while speed and wavelength may change.",
                ),
                "refraction_speed_change_media": GraphNode(
                    id="refraction_speed_change_media",
                    node_type="claim",
                    confidence=0.98,
                    text="Refraction occurs because a wave changes speed when it enters a different medium.",
                ),
                "wave_sound_medium": GraphNode(
                    id="wave_sound_medium",
                    node_type="fact",
                    confidence=0.99,
                    text="Sound waves require a material medium, but electromagnetic waves can travel through vacuum.",
                ),
                "visible_light_electromagnetic_wave": GraphNode(
                    id="visible_light_electromagnetic_wave",
                    node_type="claim",
                    confidence=0.99,
                    text="Visible light is an electromagnetic wave.",
                ),
                "bridge_field_wave_unification": GraphNode(
                    id="bridge_field_wave_unification",
                    node_type="bridge",
                    confidence=0.98,
                    text="Electromagnetic waves unify electric and magnetic ideas by propagating changing fields through space as light.",
                ),
            }
            plan = load_live_signature_bias_plan(
                question="If astronauts can see sunlight in space, why can't they hear it there?",
                task_family="direct_judgment",
                graph_nodes=graph_nodes,
                stats_dir=root,
                max_anchor_ids=4,
            )
            self.assertFalse(plan.enabled)
            self.assertEqual(plan.reason, "no_supported_graph_backed_solved_subgoal")
            self.assertEqual(plan.anchor_ids, [])
            self.assertTrue(plan.skipped_candidates)
            skipped = plan.skipped_candidates[0]
            self.assertEqual(skipped["reason"], "insufficient_live_relevance")
            self.assertEqual(skipped["family_id"], "sigfam_solved_subgoal.direct_judgment_caa5fa260901")
            self.assertIn("already_strong_baseline", skipped["gate_reason"])

    def test_live_bias_skips_contested_family_even_with_graph_backed_support(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            run_signature_shadow_session(
                session_id="sess_contra_live_a",
                question="Why can't astronauts hear sunlight in space?",
                task_family="direct_judgment",
                graph_edits=[
                    {
                        "op": "add_node",
                        "node_id": "ssg_contra_live_a",
                        "node_type": "solved_subgoal",
                        "text": "Sound cannot travel through vacuum because it needs a material medium.",
                        "metadata": {
                            "summary": "Sound cannot travel through vacuum because it needs a material medium.",
                            "subgoal_signature": "direct_judgment.sound_vacuum_hearing",
                            "question_type": "direct_judgment",
                            "question_mode": "why",
                            "output_slots": {
                                "answer": "sound cannot travel through vacuum",
                                "reason": "sound needs a material medium",
                            },
                            "supporting_node_ids": ["wave_sound_medium", "electromagnetic_waves_propagate_in_vacuum"],
                            "input_conditions": {"topic": "sound in vacuum"},
                            "valid_when": ["asking why sound cannot be heard in space"],
                            "invalid_when": [],
                        },
                    }
                ],
                scoped_patches=[
                    {
                        "patch_id": "patch_contra_live_a",
                        "patch_type": "add_solved_subgoal",
                        "target_id": "ssg_contra_live_a",
                        "evidence_node_ids": ["wave_sound_medium", "electromagnetic_waves_propagate_in_vacuum"],
                        "affected_node_ids": ["ssg_contra_live_a"],
                        "validation": {"status": "accept", "reasons": [], "warnings": []},
                    }
                ],
                hypotheses={},
                final_answer="Sound cannot travel through vacuum because it needs a material medium.",
                cited_node_ids=["wave_sound_medium", "electromagnetic_waves_propagate_in_vacuum"],
                finalized=True,
                execution_mode="micro_controller_finalize",
                stats_dir=root,
            )
            run_signature_shadow_session(
                session_id="sess_contra_live_b",
                question="Can sound travel through vacuum in space?",
                task_family="direct_judgment",
                graph_edits=[
                    {
                        "op": "add_node",
                        "node_id": "ssg_contra_live_b",
                        "node_type": "solved_subgoal",
                        "text": "Sound can travel through vacuum without any material medium.",
                        "metadata": {
                            "summary": "Sound can travel through vacuum without any material medium.",
                            "subgoal_signature": "direct_judgment.sound_vacuum_hearing",
                            "question_type": "direct_judgment",
                            "question_mode": "why",
                            "output_slots": {
                                "answer": "sound can travel through vacuum",
                                "reason": "sound self-propagates without a medium",
                            },
                            "supporting_node_ids": ["wave_sound_medium", "electromagnetic_waves_propagate_in_vacuum"],
                            "input_conditions": {"topic": "sound in vacuum"},
                            "valid_when": ["asking why sound cannot be heard in space"],
                            "invalid_when": [],
                        },
                    }
                ],
                scoped_patches=[
                    {
                        "patch_id": "patch_contra_live_b",
                        "patch_type": "add_solved_subgoal",
                        "target_id": "ssg_contra_live_b",
                        "evidence_node_ids": ["wave_sound_medium", "electromagnetic_waves_propagate_in_vacuum"],
                        "affected_node_ids": ["ssg_contra_live_b"],
                        "validation": {"status": "accept", "reasons": [], "warnings": []},
                    }
                ],
                hypotheses={},
                final_answer="Sound can travel through vacuum without any material medium.",
                cited_node_ids=["wave_sound_medium", "electromagnetic_waves_propagate_in_vacuum"],
                finalized=True,
                execution_mode="micro_controller_finalize",
                stats_dir=root,
            )
            graph_nodes = {
                "wave_sound_medium": GraphNode(
                    id="wave_sound_medium",
                    node_type="fact",
                    confidence=0.99,
                    text="Sound needs a material medium to propagate.",
                ),
                "electromagnetic_waves_propagate_in_vacuum": GraphNode(
                    id="electromagnetic_waves_propagate_in_vacuum",
                    node_type="claim",
                    confidence=0.99,
                    text="Electromagnetic waves can propagate through vacuum.",
                ),
            }
            plan = load_live_signature_bias_plan(
                question="Why can't astronauts hear sunlight in space?",
                task_family="direct_judgment",
                graph_nodes=graph_nodes,
                stats_dir=root,
                max_anchor_ids=4,
            )
            self.assertFalse(plan.enabled)
            self.assertEqual(plan.reason, "no_supported_graph_backed_solved_subgoal")
            self.assertTrue(plan.skipped_candidates)
            self.assertEqual(plan.skipped_candidates[0]["reason"], "family_contested")


class MockController:
    """Simulates an LLM controller returning a fixed response."""

    def __init__(self, response: str) -> None:
        self._response = response

    def chat_oneshot(self, messages: object) -> Dict[str, Any]:
        return {"choices": [{"message": {"content": self._response}}]}


class FailingController:
    """Simulates an LLM controller that always fails."""

    def chat_oneshot(self, messages: object) -> object:
        raise RuntimeError("LLM unavailable")


class TestNliJudgeAndEventScoring(unittest.TestCase):
    """Targeted unit tests for the new NLI judge and LLM event scoring."""

    _TEXT_A = "Dijkstra is not generally correct when negative edges are present because the greedy assumption fails"
    _TEXT_B = "Dijkstra is not correct when negative edge weights appear since greedy fails"

    def _make_candidate(
        self,
        text: str = _TEXT_A,
        supporting_ids: Optional[List[str]] = None,
    ) -> SignatureCandidate:
        return SignatureCandidate(
            family_id="fam_test",
            variant_id="var_cand",
            semantic_type="solved_subgoal",
            family_label="Test family",
            canonical_text=text,
            summary_text=text[:80],
            task_family="algorithm_applicability",
            required_slots=["verdict", "reason", "alternative"],
            scope={"algorithm": "Dijkstra"},
            supporting_node_ids=supporting_ids or ["node_a", "node_b"],
        )

    def _make_variant(
        self,
        text: str = _TEXT_B,
        supporting_ids: Optional[List[str]] = None,
    ) -> SignatureVariantStats:
        return SignatureVariantStats(
            id="var_existing",
            family_id="fam_test",
            semantic_type="solved_subgoal",
            canonical_text=text,
            summary_text=text[:80],
            task_family="algorithm_applicability",
            required_slots=["verdict", "reason", "alternative"],
            scope={"algorithm": "Dijkstra"},
            top_supporting_node_ids=supporting_ids or ["node_c", "node_d"],
            support_score=0.6,
            stability_score=0.5,
            risk_score=0.2,
            contradiction_score=0.1,
            promotion_state="supported",
            session_ids=["sess_1"],
            evidence_fingerprints=["fp1"],
        )

    # ── judge_variant_relation tests ──────────────────────────────────

    def test_nli_judge_falls_back_below_threshold(self) -> None:
        """text_overlap < 0.30 → _schema_relation fallback."""
        cand = self._make_candidate(
            text="Quantum entanglement enables superdense coding and teleportation"
        )
        variant = self._make_variant(
            text="The quick brown fox jumps over the lazy dog and runs away"
        )
        relation, rationale = judge_variant_relation(cand, variant, controller=None)
        # Matching semantic_type/task_family/slots → "overlaps" via schema
        self.assertEqual(relation, "overlaps")
        self.assertIn("partial structural overlap", rationale)

    def test_nli_judge_falls_back_above_threshold(self) -> None:
        """text_overlap > 0.80 → _schema_relation fallback."""
        identical = "Dijkstra is not generally correct when negative edges are present"
        cand = self._make_candidate(text=identical)
        variant = self._make_variant(text=identical)
        relation, rationale = judge_variant_relation(cand, variant, controller=None)
        self.assertEqual(relation, "equivalent")
        self.assertIn("matching slots/scope with strong overlap", rationale)

    def test_nli_judge_returns_equivalent(self) -> None:
        """LLM returns type=equivalent → mapped correctly."""
        cand = self._make_candidate()
        variant = self._make_variant()
        ctrl = MockController('<relation type="equivalent">essentially the same claim</relation>')
        relation, rationale = judge_variant_relation(cand, variant, ctrl)
        self.assertEqual(relation, "equivalent")
        self.assertIn("nli_judge", rationale)
        self.assertIn("essentially the same claim", rationale)

    def test_nli_judge_returns_contradicts(self) -> None:
        """LLM returns type=contradicts → mapped correctly."""
        cand = self._make_candidate()
        variant = self._make_variant()
        ctrl = MockController('<relation type="contradicts">opposite verdict</relation>')
        relation, rationale = judge_variant_relation(cand, variant, ctrl)
        self.assertEqual(relation, "contradicts")
        self.assertIn("nli_judge", rationale)

    def test_nli_judge_returns_sibling(self) -> None:
        """LLM returns sibling → mapped to overlaps."""
        cand = self._make_candidate()
        variant = self._make_variant()
        ctrl = MockController('<relation type="sibling">related but distinct</relation>')
        relation, rationale = judge_variant_relation(cand, variant, ctrl)
        self.assertEqual(relation, "overlaps")
        self.assertIn("nli_judge_sibling", rationale)

    def test_nli_judge_returns_independent(self) -> None:
        """LLM returns independent → mapped correctly."""
        cand = self._make_candidate()
        variant = self._make_variant()
        ctrl = MockController('<relation type="independent">different topics</relation>')
        relation, rationale = judge_variant_relation(cand, variant, ctrl)
        self.assertEqual(relation, "independent")
        self.assertIn("nli_judge", rationale)

    def test_nli_judge_falls_back_on_controller_error(self) -> None:
        """Controller exception → _schema_relation fallback."""
        cand = self._make_candidate()
        variant = self._make_variant()
        ctrl = FailingController()
        relation, rationale = judge_variant_relation(cand, variant, ctrl)
        self.assertEqual(relation, "overlaps")

    def test_nli_judge_falls_back_on_unparseable_response(self) -> None:
        """LLM response with no regex match → falls back."""
        cand = self._make_candidate()
        variant = self._make_variant()
        ctrl = MockController("I think these are related")
        relation, rationale = judge_variant_relation(cand, variant, ctrl)
        self.assertEqual(relation, "overlaps")

    # ── score_event_impact tests ──────────────────────────────────────

    def test_event_impact_deterministic_events(self) -> None:
        """Deterministic events return hardcoded buckets."""
        cases = [
            ("promoted_to_review", 1.0),
            ("promoted_to_supported", 1.5),
            ("deprecated", 1.5),
            ("contradicted", 2.25),
            ("scoped_patch_reject", 1.5),
        ]
        cand = self._make_candidate()
        for event_type, expected in cases:
            with self.subTest(event_type=event_type):
                score, rationale = score_event_impact(
                    event_type=event_type,
                    candidate=cand,
                    variant=None,
                    session_context={},
                    controller=None,
                )
                self.assertEqual(score, expected)
                self.assertEqual(rationale, "deterministic")

    def test_event_impact_no_variant_context(self) -> None:
        """Non-deterministic event with variant=None → medium."""
        cand = self._make_candidate()
        score, rationale = score_event_impact(
            event_type="supported_reuse",
            candidate=cand,
            variant=None,
            session_context={},
            controller=None,
        )
        self.assertEqual(score, 1.0)
        self.assertEqual(rationale, "no_variant_context")

    def test_event_impact_from_llm(self) -> None:
        """LLM returns valid score → parsed correctly."""
        cand = self._make_candidate()
        variant = self._make_variant()
        ctrl = MockController('<impact score="0.85">moderate contribution</impact>')
        score, rationale = score_event_impact(
            event_type="supported_reuse",
            candidate=cand,
            variant=variant,
            session_context={"finalized": True},
            controller=ctrl,
        )
        self.assertAlmostEqual(score, 0.85, places=4)
        self.assertEqual(rationale, "moderate contribution")

    def test_event_impact_clamps_high_score(self) -> None:
        """Score > 2.5 clamped to 2.5."""
        cand = self._make_candidate()
        variant = self._make_variant()
        ctrl = MockController('<impact score="3.0">very high impact</impact>')
        score, rationale = score_event_impact(
            event_type="supported_reuse",
            candidate=cand,
            variant=variant,
            session_context={},
            controller=ctrl,
        )
        self.assertAlmostEqual(score, 2.5, places=4)

    def test_event_impact_falls_back_on_controller_error(self) -> None:
        """Controller exception → medium fallback."""
        cand = self._make_candidate()
        variant = self._make_variant()
        ctrl = FailingController()
        score, rationale = score_event_impact(
            event_type="supported_reuse",
            candidate=cand,
            variant=variant,
            session_context={},
            controller=ctrl,
        )
        self.assertEqual(score, 1.0)
        self.assertEqual(rationale, "llm_fallback")

    def test_event_impact_falls_back_on_unparseable(self) -> None:
        """Unparseable LLM response → medium fallback."""
        cand = self._make_candidate()
        variant = self._make_variant()
        ctrl = MockController("here is my analysis")
        score, rationale = score_event_impact(
            event_type="supported_reuse",
            candidate=cand,
            variant=variant,
            session_context={},
            controller=ctrl,
        )
        self.assertEqual(score, 1.0)
        self.assertEqual(rationale, "parse_fallback")


if __name__ == "__main__":
    unittest.main()
