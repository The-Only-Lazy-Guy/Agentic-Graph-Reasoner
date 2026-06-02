from __future__ import annotations

from graph_core import MemoryGraph, Node
from v5.graph_grower.audit import AuditConfig, audit_rows


def _graph() -> MemoryGraph:
    return MemoryGraph(
        nodes={
            "support_node": Node(
                id="support_node",
                node_type="claim",
                text="Evidence supports the candidate graph edit.",
                confidence=0.95,
            )
        },
        edges=[],
    )


def _row(patches, *, quality=None, metrics=None):
    return {
        "session_id": "sess_test",
        "input": {"question": "Can this graph edit be promoted?"},
        "quality": quality if quality is not None else {
            "training_eligible": True,
            "v5_label_status": "positive",
            "finalized": True,
        },
        "metrics": metrics if metrics is not None else {
            "finalized": True,
            "controller_task_family": "direct_judgment",
            "controller_fallback_used": False,
            "slot_fill_stats": {
                "required_slots": ["answer"],
                "filled_slots": ["answer"],
                "filled_count": 1,
                "required_count": 1,
            },
            "controller_action_counts": {"VERIFY": 1},
            "scoped_patch_summary": {"patch_count": len(patches), "needs_attention": 0},
        },
        "trace": {"scoped_patches": patches},
    }


def _patch(patch_type, *, status="accept", raw_edit=None, risk_level="medium", support_score=0.9):
    raw = raw_edit or {
        "op": "add_node",
        "node_id": "candidate_fact",
        "node_type": "claim",
        "text": "Candidate fact supported by evidence.",
        "metadata": {"evidence_node_ids": ["support_node"]},
    }
    return {
        "patch_id": "patch_0000",
        "patch_type": patch_type,
        "target_id": raw.get("node_id") or raw.get("dst") or "candidate",
        "text": raw.get("text", ""),
        "risk_level": risk_level,
        "evidence_node_ids": ["support_node"],
        "raw_edit": raw,
        "validation": {
            "status": status,
            "support_score": support_score,
            "reasons": [],
            "warnings": [],
            "duplicate_of": None,
            "conflicts_with": [],
        },
    }


def test_accept_fact_from_positive_row_enters_persistent_queue():
    report = audit_rows([_row([_patch("add_fact")])], _graph())

    assert report["lanes"]["persistent_promote"]["count"] == 1
    assert report["lanes"]["persistent_promote"]["delta_estimate"]["new_nodes"] == 1
    assert report["lanes"]["review"]["count"] == 0


def test_design_row_with_suspicious_slots_is_review_not_persistent():
    metrics = {
        "finalized": True,
        "controller_task_family": "design_synthesis",
        "controller_fallback_used": True,
        "slot_fill_stats": {
            "required_slots": ["problem_frame", "rank_query", "pagination", "answer"],
            "filled_slots": ["problem_frame"],
            "filled_count": 1,
            "required_count": 4,
        },
        "scoped_patch_summary": {"patch_count": 1, "needs_attention": 0},
    }
    report = audit_rows([_row([_patch("add_fact")], metrics=metrics)], _graph())

    assert report["lanes"]["persistent_promote"]["count"] == 0
    assert report["lanes"]["review"]["count"] == 1
    candidate = report["lanes"]["review"]["candidates"][0]
    assert "suspicious_design_slots" in candidate["row_flags"]
    assert "low_slot_coverage" in candidate["row_flags"]


def test_strategy_patch_enters_substrate_not_persistent():
    patch = _patch(
        "add_strategy",
        raw_edit={
            "op": "add_node",
            "node_id": "strategy_candidate",
            "node_type": "strategy",
            "text": "Use a verify-then-answer control recipe.",
            "metadata": {"evidence_node_ids": ["support_node"]},
        },
    )
    report = audit_rows([_row([patch])], _graph(), config=AuditConfig(max_candidates=10))

    assert report["lanes"]["persistent_promote"]["count"] == 0
    assert report["lanes"]["substrate"]["count"] == 1
    assert report["lanes"]["substrate"]["delta_estimate"]["new_nodes"] == 1
