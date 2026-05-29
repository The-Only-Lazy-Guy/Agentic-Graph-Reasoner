"""Tests for reasoning/deterministic_scorer.py."""
from __future__ import annotations

import json
import platform
import tempfile
from pathlib import Path
from typing import Any, Dict, List

import pytest

from reasoning.deterministic_scorer import (
    ScoreResult,
    _contains_any,
    _plugins_for_task,
    compile_and_run_cpp,
    score_by_checker,
    score_by_compile_run,
    score_by_rubric,
    score_task_answer,
)


# ── _contains_any ─────────────────────────────────────────────────────


class TestContainsAny:
    def test_basic_match(self):
        assert _contains_any("hello world", ["hello"])

    def test_case_insensitive(self):
        assert _contains_any("Hello World", ["hello"])

    def test_no_match(self):
        assert not _contains_any("hello world", ["goodbye"])

    def test_multiple_terms_first_match(self):
        assert _contains_any("hello world", ["goodbye", "hello"])

    def test_multiple_terms_no_match(self):
        assert not _contains_any("hello world", ["goodbye", "foo"])

    def test_empty_terms(self):
        assert not _contains_any("hello", [])

    def test_substring_match(self):
        assert _contains_any("reconciling with the PSP", ["reconciling"])
        assert _contains_any("RECONCILE with the PSP", ["reconcile"])


# ── _plugins_for_task ─────────────────────────────────────────────────


class TestPluginsForTask:
    def test_payment_task(self):
        task = {"question": "A payment worker may crash after sending a charge request to the PSP."}
        plugins = _plugins_for_task(task)
        assert "payment_crash_recovery" in plugins

    def test_migration_task(self):
        task = {"question": "Migrate a monolith orders table with zero downtime and rollback."}
        plugins = _plugins_for_task(task)
        assert "zero_downtime_migration" in plugins

    def test_algorithm_design_kind(self):
        task = {"kind": "algorithm_design", "question": "Design an algorithm."}
        plugins = _plugins_for_task(task)
        assert "algorithm_design" in plugins

    def test_generic_task_returns_no_domain_plugins(self):
        task = {"kind": "factual_recall", "question": "What is entropy?"}
        plugins = _plugins_for_task(task)
        assert "payment_crash_recovery" not in plugins
        assert "algorithm_design" not in plugins

    def test_segment_beats_task(self):
        task = {"question": "Implement range_chmin and range_sum queries with segment tree beats."}
        plugins = _plugins_for_task(task)
        assert "segment_tree_beats" in plugins

    def test_inventory_flash_sale_task(self):
        task = {"question": "Design an inventory reservation system for a flash sale."}
        plugins = _plugins_for_task(task)
        assert "inventory_reservation" in plugins


# ── score_by_rubric ───────────────────────────────────────────────────


class TestScoreByRubric:
    def test_passes_with_all_required_terms(self):
        task = {
            "required_terms": [
                ["idempotency"],
                ["durable", "state machine"],
            ],
        }
        result = score_by_rubric("Use idempotency keys and a durable state machine.", task)
        assert result.correct
        assert result.score == 1.0
        assert result.source == "deterministic"
        assert result.violations == []

    def test_fails_missing_required_terms(self):
        task = {
            "required_terms": [
                ["idempotency"],
                ["durable", "state machine"],
            ],
        }
        result = score_by_rubric("Just use a normal approach.", task)
        assert not result.correct
        assert result.score == 0.0
        assert len(result.violations) > 0

    def test_fails_with_forbidden_terms(self):
        task = {
            "required_terms": [["idempotency"]],
            "forbidden_terms": ["two-phase commit"],
        }
        result = score_by_rubric("Use idempotency with two-phase commit.", task)
        assert not result.correct
        assert result.score == 0.0

    def test_passes_no_terms_at_all(self):
        task: Dict[str, Any] = {}
        result = score_by_rubric("Anything goes.", task)
        assert result.correct
        assert result.score == 1.0

    def test_or_group_any_term_suffices(self):
        task = {
            "required_terms": [
                ["alpha", "beta", "gamma"],
            ],
        }
        result = score_by_rubric("Use gamma rays.", task)
        assert result.correct


# ── score_by_checker ──────────────────────────────────────────────────


class TestScoreByChecker:
    def test_payment_checker_passes_good_answer(self):
        task = {
            "kind": "system_design",
            "question": "A payment worker may crash after sending a charge. How to recover?",
        }
        answer = (
            "The worker persists a durable state machine with idempotency keys. "
            "On restart, it queries the PSP for status before retrying. "
            "Reconciliation with the PSP ensures no double charge. "
            "Retries use deduplication to prevent duplicates."
        )
        result = score_by_checker(answer, task)
        assert result is not None
        assert result.correct
        assert result.source == "deterministic"

    def test_payment_checker_fails_bad_answer(self):
        task = {
            "kind": "system_design",
            "question": "A payment worker may crash after sending a charge. How to recover?",
        }
        answer = "Use exactly-once delivery semantics to avoid double charging."
        result = score_by_checker(answer, task)
        assert result is not None
        assert not result.correct
        assert any("exactly_once" in v for v in result.violations)

    def test_generic_task_returns_none(self):
        task = {
            "kind": "factual_recall",
            "question": "What is entropy?",
        }
        result = score_by_checker("Entropy is disorder.", task)
        assert result is None

    def test_migration_checker_passes_good_answer(self):
        task = {
            "kind": "system_design",
            "question": "Migrate a monolith orders table from MySQL to Spanner with zero downtime.",
        }
        answer = (
            "Phase 1: Backfill historical data from MySQL. "
            "Phase 2: Capture live writes via CDC. "
            "Phase 3: Verify parity between databases. "
            "Phase 4: Cut over application routing. "
            "Phase 5: Rollback if needed by reverting routing."
        )
        result = score_by_checker(answer, task)
        assert result is not None
        assert result.correct

    def test_beats_checker_passes_good_answer(self):
        task = {
            "kind": "algorithm_design",
            "question": "Design a segment tree supporting range_chmin and range_sum queries.",
        }
        answer = (
            "Use a segment tree where each node stores max, second max, count_max, and sum. "
            "A range_chmin only changes current maxima when x lies between max and second max. "
            "The sum is updated during propagation."
        )
        result = score_by_checker(answer, task)
        assert result is not None
        assert result.correct

    def test_beats_checker_fails_ordinary_lazy(self):
        task = {
            "kind": "algorithm_design",
            "question": "Design a segment tree supporting range_chmin and range_sum queries.",
        }
        answer = "Use a standard lazy propagation segment tree with range update."
        result = score_by_checker(answer, task)
        assert result is not None
        assert not result.correct


# ── score_task_answer (integration) ───────────────────────────────────


class TestScoreTaskAnswer:
    def test_rubric_only_task(self):
        task = {
            "kind": "factual_recall",
            "question": "What is HTTP GET idempotency?",
            "required_terms": [["idempotent"], ["safe", "no side effects"]],
        }
        result = score_task_answer(
            "HTTP GET is idempotent and has no side effects.", task
        )
        assert result.correct
        assert result.source == "deterministic"

    def test_checker_plugin_overrides_rubric(self):
        task = {
            "kind": "system_design",
            "question": "A payment worker may crash. Design recovery.",
            "required_terms": [["durable"]],
        }
        # Answer passes rubric (contains "durable") but fails checker
        answer = "Use a durable exactly-once transport to avoid double charges."
        result = score_task_answer(answer, task)
        assert not result.correct  # checker fails even though rubric passes
        assert result.source == "deterministic"

    def test_empty_answer_with_rubric_fails(self):
        task = {
            "kind": "algorithm_design",
            "question": "Design an algorithm.",
            "required_terms": [["segment tree"]],
        }
        result = score_task_answer("", task)
        assert not result.correct

    def test_empty_answer_no_rubric_passes_checker_soft(self):
        task = {"kind": "algorithm_design", "question": "Design an algorithm."}
        result = score_task_answer("", task)
        assert result.correct  # only soft violations (complexity_missing)
        assert result.source == "deterministic"


# ── compile_and_run_cpp ──────────────────────────────────────────────


@pytest.mark.skipif(
    platform.system() == "Windows" and not any(
        p for p in __import__("os").environ.get("PATH", "").split(";")
        if "g++" in p.lower() or "mingw" in p.lower() or "msys" in p.lower()
    ),
    reason="g++ not available on PATH",
)
class TestCompileAndRunCpp:
    def test_hello_world(self):
        source = '#include <iostream>\nint main() { std::cout << "OK"; return 0; }'
        result = compile_and_run_cpp(source, [("", "OK")])
        assert result.passed
        assert result.failures == []

    def test_fails_compilation_error(self):
        source = "this is not valid c++"
        result = compile_and_run_cpp(source, [])
        assert not result.passed
        assert any("compilation_failed" in f for f in result.failures)

    def test_output_mismatch(self):
        source = '#include <iostream>\nint main() { std::cout << "hello"; return 0; }'
        result = compile_and_run_cpp(source, [("", "world")])
        assert not result.passed
        assert any("test_case_0" in f for f in result.failures)

    def test_multiple_cases_all_pass(self):
        source = (
            '#include <iostream>\n'
            'int main() { int x; std::cin >> x; std::cout << x * 2; return 0; }'
        )
        cases = [("3", "6"), ("0", "0"), ("-5", "-10")]
        result = compile_and_run_cpp(source, cases)
        assert result.passed

    def test_multiple_cases_one_fails(self):
        source = (
            '#include <iostream>\n'
            'int main() { int x; std::cin >> x; std::cout << x + 1; return 0; }'
        )
        cases = [("1", "2"), ("5", "7"), ("0", "1")]
        result = compile_and_run_cpp(source, cases)
        assert not result.passed

    def test_timeout_kills(self):
        source = 'int main() { while(true) {} return 0; }'
        result = compile_and_run_cpp(source, [("", "x")], timeout_sec=2)
        assert not result.passed
        assert any("timeout" in f for f in result.failures)


# ── score_by_compile_run (task-level) ────────────────────────────────


class TestScoreByCompileRun:
    def test_no_exec_tests_returns_none(self):
        task = {"kind": "algorithm_design", "question": "Design an algorithm."}
        result = score_by_compile_run("any answer", task)
        assert result is None

    def test_skips_on_non_cpp_task(self):
        task = {
            "kind": "code_generation",
            "question": "Write Python code.",
            "execution_language": "python",
            "execution_tests": [{"input": "1", "expected": "2"}],
        }
        result = score_by_compile_run("print(1+1)", task)
        assert result is not None
        assert not result.correct
        assert any("python" in f for f in result.violations)
