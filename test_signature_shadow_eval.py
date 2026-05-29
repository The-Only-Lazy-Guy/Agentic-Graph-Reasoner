from __future__ import annotations

import json
import tempfile
from pathlib import Path

from eval_signature_shadow import evaluate_shadow_labels
from reasoning.signature_stats import run_signature_shadow_session


def _write_packet(path: Path, *, question: str, task_family: str, signature_result: dict | None = None) -> None:
    packet = {
        "question": question,
        "controller_task_family": task_family,
    }
    if signature_result is not None:
        packet.update({
            "signature_candidates": signature_result.get("candidates", []),
            "signature_events": signature_result.get("events", []),
            "signature_stats_update": signature_result.get("update_summary", {}),
            "signature_shadow_report": signature_result.get("shadow_report", {}),
            "signature_graph_projection": signature_result.get("graph_projection", {}),
        })
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(packet, ensure_ascii=False, indent=2), encoding="utf-8")


def test_signature_shadow_eval_scores_and_skips_mixed_packets() -> None:
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        artifacts_root = root / "artifacts"
        stats_root = root / "stats"

        first = run_signature_shadow_session(
            session_id="sess_eval_a",
            question="Can Dijkstra be trusted with one negative edge?",
            task_family="algorithm_applicability",
            graph_edits=[
                {
                    "op": "add_node",
                    "node_id": "ssg_eval_a",
                    "node_type": "solved_subgoal",
                    "text": "Dijkstra is unsafe with negative edges.",
                    "metadata": {
                        "summary": "Dijkstra is unsafe with negative edges.",
                        "subgoal_signature": "algorithm_applicability.dijkstra.negative_edge_weights.validity",
                        "question_type": "algorithm_applicability",
                        "output_slots": {
                            "verdict": "unsafe",
                            "reason": "negative edges break the finalized-distance invariant",
                            "alternative": "Bellman-Ford",
                        },
                        "supporting_node_ids": ["dijkstra_neg_edges_failure", "bellman_ford_apply"],
                        "input_conditions": {"algorithm": "Dijkstra", "condition": "negative edge weights"},
                        "valid_when": ["asking about standard Dijkstra"],
                        "invalid_when": [],
                    },
                }
            ],
            scoped_patches=[
                {
                    "patch_id": "patch_eval_a",
                    "patch_type": "add_solved_subgoal",
                    "target_id": "ssg_eval_a",
                    "evidence_node_ids": ["dijkstra_neg_edges_failure", "bellman_ford_apply"],
                    "affected_node_ids": ["ssg_eval_a"],
                    "validation": {"status": "accept", "reasons": [], "warnings": []},
                }
            ],
            hypotheses={},
            final_answer="Dijkstra is unsafe with negative edges. Use Bellman-Ford instead.",
            cited_node_ids=["dijkstra_neg_edges_failure", "bellman_ford_apply"],
            finalized=True,
            execution_mode="micro_controller_finalize",
            stats_dir=stats_root,
        )
        first_family_id = first["candidates"][0]["family_id"]
        _write_packet(
            artifacts_root / "01_first" / "packet.json",
            question="Can Dijkstra be trusted with one negative edge?",
            task_family="algorithm_applicability",
            signature_result=first,
        )

        second = run_signature_shadow_session(
            session_id="sess_eval_b",
            question="Can standard Dijkstra handle negative edge weights?",
            task_family="algorithm_applicability",
            graph_edits=[
                {
                    "op": "add_node",
                    "node_id": "ssg_eval_b",
                    "node_type": "solved_subgoal",
                    "text": "Standard Dijkstra is not reliable when negative edges exist.",
                    "metadata": {
                        "summary": "Standard Dijkstra is not reliable when negative edges exist.",
                        "subgoal_signature": "algorithm_applicability.dijkstra.negative_edge_weights.validity",
                        "question_type": "algorithm_applicability",
                        "output_slots": {
                            "verdict": "unsafe",
                            "reason": "negative edges break the finalized-distance invariant",
                            "alternative": "Bellman-Ford",
                        },
                        "supporting_node_ids": ["dijkstra_neg_edges_failure", "bellman_ford_apply"],
                        "input_conditions": {"algorithm": "Dijkstra", "condition": "negative edge weights"},
                        "valid_when": ["asking about standard Dijkstra"],
                        "invalid_when": [],
                    },
                }
            ],
            scoped_patches=[
                {
                    "patch_id": "patch_eval_b",
                    "patch_type": "add_solved_subgoal",
                    "target_id": "ssg_eval_b",
                    "evidence_node_ids": ["dijkstra_neg_edges_failure", "bellman_ford_apply"],
                    "affected_node_ids": ["ssg_eval_b"],
                    "validation": {"status": "accept", "reasons": [], "warnings": []},
                }
            ],
            hypotheses={},
            final_answer="Standard Dijkstra is not reliable when negative edges exist.",
            cited_node_ids=["dijkstra_neg_edges_failure", "bellman_ford_apply"],
            finalized=True,
            execution_mode="micro_controller_finalize",
            stats_dir=stats_root,
        )
        _write_packet(
            artifacts_root / "02_revision" / "packet.json",
            question="Can standard Dijkstra handle negative edge weights?",
            task_family="algorithm_applicability",
            signature_result=second,
        )

        run_signature_shadow_session(
            session_id="sess_eval_c_seed",
            question="Can Dijkstra be trusted with one negative edge? strategy seed",
            task_family="algorithm_applicability",
            graph_edits=[
                {
                    "op": "add_node",
                    "node_id": "strat_eval_seed",
                    "node_type": "strategy",
                    "text": "Strategy family: algorithm_applicability\nCheckpoint plan:\n1. Read failure node\n2. Answer with Bellman-Ford alternative",
                    "metadata": {
                        "task_family": "algorithm_applicability",
                        "task_subtype": "algorithm_mechanism_explanation",
                        "question_mode": "judgment",
                        "domain_keywords": ["dijkstra", "negative", "edge"],
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
                    "patch_id": "patch_eval_seed",
                    "patch_type": "add_strategy",
                    "target_id": "strat_eval_seed",
                    "evidence_node_ids": ["dijkstra_neg_edges_failure"],
                    "affected_node_ids": ["strat_eval_seed"],
                    "validation": {"status": "accept", "reasons": [], "warnings": []},
                }
            ],
            hypotheses={},
            final_answer="Use Bellman-Ford when negative edges exist.",
            cited_node_ids=["dijkstra_neg_edges_failure"],
            finalized=True,
            execution_mode="micro_controller_finalize",
            stats_dir=stats_root,
        )

        third = run_signature_shadow_session(
            session_id="sess_eval_c",
            question="Can Dijkstra be trusted if a graph has a negative edge?",
            task_family="algorithm_applicability",
            graph_edits=[
                {
                    "op": "add_node",
                    "node_id": "strat_eval_c",
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
                    "patch_id": "patch_eval_c",
                    "patch_type": "add_strategy",
                    "target_id": "strat_eval_c",
                    "evidence_node_ids": ["dijkstra_neg_edges_failure"],
                    "affected_node_ids": ["strat_eval_c"],
                    "validation": {"status": "accept", "reasons": [], "warnings": []},
                }
            ],
            hypotheses={},
            final_answer="Negative edges break the invariant, so use Bellman-Ford.",
            cited_node_ids=["dijkstra_neg_edges_failure"],
            finalized=True,
            execution_mode="micro_controller_finalize",
            stats_dir=stats_root,
        )
        third_family_id = third["candidates"][0]["family_id"]
        _write_packet(
            artifacts_root / "03_sibling" / "packet.json",
            question="Can Dijkstra be trusted if a graph has a negative edge?",
            task_family="algorithm_applicability",
            signature_result=third,
        )

        _write_packet(
            artifacts_root / "04_legacy" / "packet.json",
            question="Legacy packet without signature shadow",
            task_family="algorithm_applicability",
            signature_result=None,
        )

        labels_path = root / "labels.json"
        labels_path.write_text(
            json.dumps(
                {
                    "cases": [
                        {
                            "id": "family_hit_case",
                            "question": "Can Dijkstra be trusted with one negative edge?",
                            "task_family": "algorithm_applicability",
                            "gold_signature_family_ids": [first_family_id],
                            "gold_signature_variant_ids": [],
                            "unsafe_family_ids": [],
                        },
                        {
                            "id": "revision_case",
                            "question": "Can standard Dijkstra handle negative edge weights?",
                            "task_family": "algorithm_applicability",
                            "gold_signature_family_ids": [first_family_id],
                            "gold_signature_variant_ids": [],
                            "unsafe_family_ids": [],
                            "matching_expectation": {
                                "semantic_type": "solved_subgoal",
                                "expected_variant_resolution": "equivalent_revision",
                                "should_match_existing_family": True,
                            },
                        },
                        {
                            "id": "sibling_case",
                            "question": "Can Dijkstra be trusted if a graph has a negative edge?",
                            "task_family": "algorithm_applicability",
                            "gold_signature_family_ids": [third_family_id],
                            "gold_signature_variant_ids": [],
                            "unsafe_family_ids": [],
                            "matching_expectation": {
                                "semantic_type": "strategy",
                                "expected_variant_resolution": "sibling_variant",
                                "should_match_existing_family": True,
                            },
                        },
                        {
                            "id": "legacy_skip_case",
                            "question": "Legacy packet without signature shadow",
                            "task_family": "algorithm_applicability",
                            "gold_signature_family_ids": ["sigfam_legacy.placeholder"],
                            "gold_signature_variant_ids": [],
                            "unsafe_family_ids": [],
                        },
                    ]
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

        report = evaluate_shadow_labels([artifacts_root], labels_path)

        assert report["packet_count"] == 4
        assert report["matched_case_count"] == 4
        assert report["evaluated_case_count"] == 3
        assert report["skipped_case_count"] == 1
        assert report["skip_reason_counts"]["missing_shadow_report"] == 1
        assert report["metrics"]["equivalent_revision_precision"] == 1.0
        assert report["metrics"]["sibling_variant_precision"] == 1.0
        assert report["metrics"]["new_family_false_split_rate"] == 0.0

        by_label = {row["label_id"]: row for row in report["cases"]}
        assert by_label["family_hit_case"]["family_hit_at_1_adjusted"] == 1
        assert by_label["revision_case"]["selected_candidate"]["variant_resolution"] == "equivalent_revision"
        assert by_label["sibling_case"]["selected_candidate"]["variant_resolution"] == "sibling_variant"
        assert by_label["legacy_skip_case"]["skip_reason"] == "missing_shadow_report"
