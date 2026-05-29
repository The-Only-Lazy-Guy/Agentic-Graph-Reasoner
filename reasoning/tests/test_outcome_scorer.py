"""Tests for reasoning/outcome_scorer.py."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from reasoning.outcome_scorer import (
    SubstrateOutcomeRow,
    collect_outcome_row,
    outcome_count,
    write_outcome_row,
)
from reasoning.reasoning_loop import ReasoningResult
from reasoning.schemas import SessionSubgraph


def _make_fake_result(
    answer: str = "some answer",
    llm_calls: int = 3,
    step_timing: list[float] | None = None,
) -> ReasoningResult:
    return ReasoningResult(
        answer=answer,
        reasoning_trace="trace",
        raw_outputs=[],
        session_subgraph=SessionSubgraph(session_id="test", query="?", graph_id="test", nodes={}, edges=[]),
        session_subgraph_path=Path("."),
        audit_summary={
            "step_count": 2,
            "checker_outcome_breakdown": {
                "passed_strict": 1,
                "passed_soft": 0,
                "failed_hard": 0,
                "failed_soft": 0,
            },
            "step_timing": step_timing or [0.5, 0.3],
            "debug_signal_dump": [
                {"id": "sig_1"},
                {"id": "sig_2"},
                {"id": "sig_1"},
            ],
        },
        consolidation_decisions=[],
        budget_usage={"llm_calls": {"used": llm_calls}},
        dispatch_outcomes=[],
        anchor_ids=[],
        iterations_completed=2,
    )


class TestCollectOutcomeRow(unittest.TestCase):
    def test_basic_row_structure(self):
        result = _make_fake_result()
        task = {"id": "test_task", "kind": "system_design", "question": "Design X?"}
        row = collect_outcome_row(result, task, elapsed_sec=1.23)
        self.assertIsInstance(row, SubstrateOutcomeRow)
        self.assertEqual(row.task_id, "test_task")
        self.assertEqual(row.task_kind, "system_design")
        self.assertEqual(row.llm_calls, 3)
        self.assertEqual(row.step_count, 2)
        self.assertEqual(row.elapsed_sec, 1.23)
        self.assertEqual(len(row.step_timing), 2)

    def test_row_with_judge(self):
        result = _make_fake_result()
        task = {"id": "t1", "kind": "algorithm", "question": "Solve X?"}
        judge = {"passed": True, "mode": "rubric", "rubric": {"passed": True}}
        row = collect_outcome_row(result, task, elapsed_sec=0.5, judge=judge)
        self.assertTrue(row.outcome_correct)
        self.assertEqual(row.outcome_score, 1.0)
        self.assertEqual(row.outcome_source, "deterministic")

    def test_row_with_failed_judge(self):
        result = _make_fake_result()
        task = {"id": "t1", "kind": "algorithm", "question": "Solve X?"}
        judge = {"passed": False, "mode": "rubric", "rubric": {"passed": False}}
        row = collect_outcome_row(result, task, elapsed_sec=0.5, judge=judge)
        self.assertFalse(row.outcome_correct)
        self.assertEqual(row.outcome_score, 0.0)

    def test_activated_signal_ids_deduped(self):
        result = _make_fake_result()
        task = {"id": "t1", "kind": "system_design", "question": "Design Y?"}
        row = collect_outcome_row(result, task)
        self.assertEqual(row.activated_signal_ids, ["sig_1", "sig_2"])

    def test_no_debug_dump_produces_empty_list(self):
        result = _make_fake_result()
        result.audit_summary["debug_signal_dump"] = []
        task = {"id": "t1", "kind": "system_design", "question": "Design Z?"}
        row = collect_outcome_row(result, task)
        self.assertEqual(row.activated_signal_ids, [])


class TestWriteOutcomeRow(unittest.TestCase):
    def test_writes_to_daily_jsonl(self):
        result = _make_fake_result()
        task = {"id": "test_write", "kind": "algorithm", "question": "Write X?"}
        row = collect_outcome_row(result, task)
        path = write_outcome_row(row)
        self.assertTrue(path.exists())
        content = path.read_text(encoding="utf-8")
        self.assertIn("test_write", content)
        self.assertTrue(content.endswith("\n"))
        # Cleanup
        path.unlink(missing_ok=True)

    def test_multiple_rows_append(self):
        result = _make_fake_result()
        task = {"id": "multi_test", "kind": "algorithm", "question": "Multi?"}
        row1 = collect_outcome_row(result, task)
        row2 = collect_outcome_row(result, task)
        path1 = write_outcome_row(row1)
        write_outcome_row(row2)
        lines = path1.read_text(encoding="utf-8").strip().split("\n")
        self.assertEqual(len(lines), 2)
        path1.unlink(missing_ok=True)

    def test_outcome_count(self):
        before = outcome_count()
        result = _make_fake_result()
        task = {"id": "count_test", "kind": "algorithm", "question": "Count?"}
        row = collect_outcome_row(result, task)
        path = write_outcome_row(row)
        after = outcome_count()
        self.assertEqual(after, before + 1)
        path.unlink(missing_ok=True)
