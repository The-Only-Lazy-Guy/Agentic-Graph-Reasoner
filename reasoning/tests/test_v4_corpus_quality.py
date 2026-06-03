from __future__ import annotations

from types import SimpleNamespace

from answerer_v4 import V4Session, V4Tools
from graph_core import MemoryGraph, Node
from reasoning.distillation_corpus import packet_to_corpus_row


def _graph() -> MemoryGraph:
    return MemoryGraph({
        "lock_ordering": Node(
            id="lock_ordering",
            node_type="claim",
            text="A consistent lock ordering prevents circular wait in concurrent systems.",
            confidence=0.99,
        ),
        "local_nn": Node(
            id="local_nn",
            node_type="claim",
            text=(
                "Neural networks can use local auxiliary losses to train layer-wise "
                "representations without relying only on a final backpropagation signal."
            ),
            confidence=0.9,
        ),
    }, [])


def test_verify_hypotheses_rejects_verified_analogy_without_question_overlap():
    session = V4Session(
        question="Design a better neural network update than backpropagation."
    )
    tools = V4Tools(_graph(), session)
    tools.read_node("lock_ordering")
    hyp = tools.hypothesize(
        "Forward-only layer ordering solves the neural network credit assignment problem."
    )

    out = tools.verify_hypotheses([{
        "id": hyp["id"],
        "verdict": "verified",
        "evidence": "confirmed by `lock_ordering`",
    }])

    assert "error" in out["results"][0]
    assert out["remaining_unverified"] == [hyp["id"]]
    assert session.hypotheses[hyp["id"]]["verdict"] is None


def test_verify_hypotheses_accepts_read_question_relevant_evidence():
    session = V4Session(
        question="Design a better neural network update than backpropagation."
    )
    tools = V4Tools(_graph(), session)
    tools.read_node("local_nn")
    hyp = tools.hypothesize(
        "Local auxiliary losses can provide layer-wise update signals."
    )

    out = tools.verify_hypotheses([{
        "id": hyp["id"],
        "verdict": "verified",
        "evidence": "confirmed by `local_nn`",
    }])

    assert out["results"][0]["status"] == "stamped"
    assert out["remaining_unverified"] == []
    quality = session.hypotheses[hyp["id"]]["verification_quality"]
    assert quality["accepted"] is True
    assert quality["cited_node_ids"] == ["local_nn"]


def test_corpus_row_keeps_turn_summaries_raw_trace_and_training_quality():
    pkt = SimpleNamespace(
        anchors=[],
        question="Can Dijkstra handle a negative edge?",
        task_frame_items=0,
        task_type="direct_judgment",
        controller_task_family="algorithm_applicability",
        plan=[],
        plan_tree_summary=None,
        tool_log=[{"name": "read_node", "args": {"node_id": "n1"}, "result_summary": "ok"}],
        cot_log=[
            '<tool>{"name":"read_node","args":{"node_id":"n1"}}</tool>',
            '<evidence_audit>{"claims":[]}</evidence_audit><answer>No.</answer>',
        ],
        hypotheses={},
        failures=[],
        objects={},
        procedure_invocations=[],
        micro_steps=[],
        scoped_patches=[],
        answer_raw="No.",
        answer="No.",
        explanation="",
        reflection=None,
        steps=2,
        max_steps=4,
        tool_call_count=1,
        elapsed_sec=0.1,
        finalized=True,
        execution_mode="loop",
        shortcut_anchor_ids=[],
        citation_warnings=0,
        search_repeats=0,
        activation_signals=0,
        coverage_addressed_pct=1.0,
        coverage_rounds=0,
        subgoal_reuse_count=0,
        slot_fill_stats={},
        controller_action_counts={},
        controller_fallback_used=False,
        polish_applied=False,
        budget_summary=None,
        meta_signals=[],
        graph_edits=[],
        graph_edits_applied=False,
        scoped_patch_summary={},
        reflection_edits=[],
        reflection_applied=False,
        nodes_accessed_log=[],
        session_dir="session",
        controller_raw_trace=[{"mode": "loop", "assistant_text": "hello", "events": [1]}],
        controller_call_count=1,
        controller_total_elapsed_sec=0.2,
        controller_nonempty_turns=1,
        finalization_quality={
            "training_eligible": False,
            "issues": ["weak_verified_hypotheses"],
        },
    )

    row = packet_to_corpus_row(pkt, MemoryGraph({}, []))

    assert row["trace"]["turn_summaries"][0]["tool_names"] == ["read_node"]
    assert row["trace"]["turn_summaries"][1]["has_evidence_audit"] is True
    assert row["trace"]["controller_raw_trace"][0]["assistant_text"] == "hello"
    assert row["trace"]["controller_raw_trace_summary"][0]["assistant_chars"] == 5
    assert row["quality"]["training_eligible"] is False
    assert row["quality"]["v5_label_status"] == "needs_review"


def _grounded_pkt(**over):
    base = dict(
        anchors=["n1"], question="Can Dijkstra handle a negative edge?",
        task_frame_items=0, task_type="direct_judgment",
        controller_task_family="algorithm_applicability", plan=[], plan_tree_summary=None,
        tool_log=[], cot_log=[], hypotheses={}, failures=[], objects={},
        procedure_invocations=[],
        micro_steps=[{"index": 1, "subgoal": "answer_question",
                      "subgoal_signature": "sig", "action": "REUSE",
                      "sufficient": True, "filled_slots": ["answer"]}],
        scoped_patches=[],
        answer_raw="Dijkstra requires non-negative edge weights.",
        answer="Dijkstra requires non-negative edge weights.",
        explanation="", reflection=None, steps=1, max_steps=4, tool_call_count=0,
        elapsed_sec=0.1, finalized=True, execution_mode="finalize",
        shortcut_anchor_ids=["n1"], citation_warnings=0, search_repeats=0,
        activation_signals=0, coverage_addressed_pct=1.0, coverage_rounds=0,
        subgoal_reuse_count=0, slot_fill_stats={}, controller_action_counts={},
        controller_fallback_used=False, polish_applied=False, budget_summary=None,
        meta_signals=[], graph_edits=[], graph_edits_applied=False,
        scoped_patch_summary={}, reflection_edits=[], reflection_applied=False,
        nodes_accessed_log=[{"node_id": "n1", "node_type": "fact", "step": 1,
                             "reason": "anchor_retrieval"}],
        session_dir="session", controller_raw_trace=[], controller_call_count=1,
        controller_total_elapsed_sec=0.2, controller_nonempty_turns=1,
        finalization_quality={"training_eligible": True},
    )
    base.update(over)
    return SimpleNamespace(**base)


def test_v2_grounding_normalizes_brief_support_and_subtasks():
    g = MemoryGraph({"n1": Node(id="n1", node_type="fact",
                                text="Dijkstra requires non-negative edge weights.",
                                confidence=0.9)}, [])
    row = packet_to_corpus_row(_grounded_pkt(), g)
    v2 = row["v2_grounding"]
    assert v2["task"] == "Can Dijkstra handle a negative edge?"
    assert v2["artifact_kind"] == "graph_node"
    assert v2["brief"]["retrieved_ids"] == ["n1"]
    assert v2["brief"]["anchor_ids"] == ["n1"]
    assert v2["brief"]["retrieved_by_reason"]["anchor_retrieval"] == ["n1"]
    assert v2["support_ids"] == ["n1"]            # answer overlaps node -> support kept
    assert v2["overrides_graph"] is False
    assert len(v2["subtasks"]) == 1 and v2["subtasks"][0]["goal"] == "answer_question"
    assert v2["verifier"]["finalized"] is True and v2["verifier"]["label_status"] == "positive"


def test_v2_grounding_drops_support_on_override():
    # answer shares ~no content with the support node -> false grounding
    g = MemoryGraph({"n1": Node(id="n1", node_type="fact",
                                text="Photosynthesis converts light into chemical energy.",
                                confidence=0.9)}, [])
    row = packet_to_corpus_row(_grounded_pkt(), g)
    v2 = row["v2_grounding"]
    assert v2["overrides_graph"] is True
    assert v2["support_ids"] == []                # override -> no support
    assert v2["verifier"]["label_status"] == "unsupported"
