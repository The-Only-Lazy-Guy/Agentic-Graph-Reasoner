"""Deterministic outcome scorer — Phase 3G G2.

Three scoring methods, in precedence order:

1. Rubric match — required_terms / forbidden_terms (always applicable).
2. Checker plugin — domain-specific keyword checkers on final answer text.
3. Compile/run — executable code tasks with inline test cases.

Each method produces a ScoreResult with outcome_source = "deterministic"
(rubric/checker) or "test_runner" (compile/run).
"""
from __future__ import annotations

import os
import re
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from reasoning.substrate_v2 import (
    CheckerRegistry,
    DeltaTransaction,
    StateDelta,
    StepContextPacket,
    StepResult,
)


@dataclass
class ScoreResult:
    """Output of a single deterministic scoring pass."""

    correct: bool
    score: float
    source: str  # "deterministic" | "test_runner"
    violations: List[str]
    details: Dict[str, Any] = field(default_factory=dict)


# ── helpers ─────────────────────────────────────────────────────────────


def _contains_any(text: str, terms: Sequence[str]) -> bool:
    low = text.lower()
    return any(str(t).lower() in low for t in terms)


def _get_task_question(task: Dict[str, Any]) -> str:
    return str(task.get("question", task.get("focus", "")))


# ── 1. Rubric scoring (required_terms / forbidden_terms) ────────────


def score_by_rubric(answer: str, task: Dict[str, Any]) -> ScoreResult:
    """Score answer against the task's required_terms and forbidden_terms.

    This mirrors judge_answer() in run_phase3e_benchmark.py.
    Always succeeds (returns ScoreResult even with empty terms).
    """
    missing_groups: List[str] = []
    for group in task.get("required_terms", []):
        if not _contains_any(answer, group):
            missing_groups.append(str(group[0] if group else "?"))

    forbidden_hits: List[str] = []
    for term in task.get("forbidden_terms", []):
        if str(term).lower() in answer.lower():
            forbidden_hits.append(str(term))

    passed = not missing_groups and not forbidden_hits
    violations: List[str] = []
    if missing_groups:
        violations.append(f"missing_required: {', '.join(missing_groups[:3])}")
    if forbidden_hits:
        violations.append(f"forbidden_hits: {', '.join(forbidden_hits[:3])}")

    return ScoreResult(
        correct=passed,
        score=1.0 if passed else 0.0,
        source="deterministic",
        violations=violations,
        details={
            "missing_required_groups": missing_groups,
            "forbidden_hits": forbidden_hits,
        },
    )


# ── 2. Checker-plugin scoring ────────────────────────────────────────


_FINAL_ANSWER_CHECKER_PLUGINS = {
    "algorithm_design",
    "dynamic_max_subarray",
    "shortest_path_safety",
    "dynamic_connectivity_deletions",
    "segment_tree_beats",
    "payment_crash_recovery",
    "zero_downtime_migration",
    "inventory_reservation",
}

_CHECKER_PLUGIN_RULES: List[Tuple[str, List[str]]] = [
    ("dynamic_max_subarray", ["subarray", "update", "online"]),
    ("dynamic_connectivity_deletions", ["remove(", "connected(", "plain dsu is insufficient", "time-axis structure"]),
    ("segment_tree_beats", ["range_chmin", "range_sum", "per-node state"]),
    ("shortest_path_safety", ["dijkstra", "shortest", "negative edge"]),
    ("payment_crash_recovery", ["payment worker", "psp", "double charge", "idempotency"]),
    ("zero_downtime_migration", ["zero downtime", "cutover", "rollback", "order"]),
    ("inventory_reservation", ["flash-sale", "flash sale", "reservation ttl", "oversell"]),
]


def _plugins_for_task(task: Dict[str, Any]) -> List[str]:
    """Return checker plugin names relevant to this task's question."""
    question = _get_task_question(task).lower()
    kind = str(task.get("kind", "")).lower()
    plugins: List[str] = []

    if kind == "algorithm_design":
        plugins.append("algorithm_design")

    for plugin, keywords in _CHECKER_PLUGIN_RULES:
        if any(kw.lower() in question for kw in keywords):
            plugins.append(plugin)

    return [p for p in plugins if p in _FINAL_ANSWER_CHECKER_PLUGINS]


def _build_packet(task: Dict[str, Any]) -> StepContextPacket:
    """Build a minimal StepContextPacket from task JSON for final answer checking."""
    question = _get_task_question(task)
    return StepContextPacket(
        task_summary=question,
        focus=question,
        looking_for="answer",
        active_signals=[],
        parent_decisions=[],
        open_gaps=[],
        hard_constraints=list(task.get("hard_constraints", [])),
        budget_remaining={},
    )


def _build_step_result(answer: str) -> StepResult:
    """Wrap a final answer string into a StepResult for checker plugins."""
    return StepResult(
        status="resolved",
        result=answer,
        delta_transaction=DeltaTransaction(
            status="parsed",
            delta=StateDelta(
                decisions=[answer[:500]] if answer else [],
            ),
        ),
    )


def score_by_checker(answer: str, task: Dict[str, Any]) -> Optional[ScoreResult]:
    """Score answer using domain-specific checker plugins.

    Returns None if no checker plugin applies to this task.
    """
    plugins = _plugins_for_task(task)
    if not plugins:
        return None

    packet = _build_packet(task)
    step_result = _build_step_result(answer)
    registry = CheckerRegistry(plugins)
    check = registry.verify(step_result, packet)

    violations: List[str] = [f"{v.code}: {v.message}" for v in check.violations]
    hard_count = sum(1 for v in check.violations if v.severity == "hard")

    passed = hard_count == 0
    return ScoreResult(
        correct=passed,
        score=1.0 if passed else 0.0,
        source="deterministic",
        violations=violations,
        details={
            "plugins": plugins,
            "violation_codes": [v.code for v in check.violations],
            "hard_count": hard_count,
            "soft_count": len(check.violations) - hard_count,
        },
    )


# ── 3. Compile / run sandbox (code tasks) ────────────────────────────


@dataclass
class TestResult:
    """Result of one compile-and-run test case."""

    passed: bool
    failures: List[str]


_SANDBOX_DIR = Path("data/sandbox")


def _sandbox_path() -> Path:
    _SANDBOX_DIR.mkdir(parents=True, exist_ok=True)
    return _SANDBOX_DIR


def compile_and_run_cpp(
    source_code: str,
    test_cases: List[Tuple[str, str]],
    *,
    timeout_sec: int = 10,
    compiler: str = "g++",
    std: str = "-std=c++17",
    optimization: str = "-O2",
) -> TestResult:
    """Compile C++ source and run against test cases.

    Each test case is (stdin_input, expected_stdout).
    Returns TestResult with passed=True iff all cases pass.
    """
    tag = f"tmp_{int(time.time() * 1000000)}"
    src_path = _sandbox_path() / f"{tag}.cpp"
    exe_path = _sandbox_path() / f"{tag}.exe"

    failures: List[str] = []

    try:
        src_path.write_text(source_code, encoding="utf-8")

        # Compile
        compile_cmd = [compiler, std, optimization, str(src_path), "-o", str(exe_path)]
        proc = subprocess.run(
            compile_cmd,
            capture_output=True,
            text=True,
            timeout=timeout_sec,
        )
        if proc.returncode != 0:
            failures.append(f"compilation_failed: {proc.stderr[:500]}")
            return TestResult(passed=False, failures=failures)

        # Run each test case
        for i, (stdin_data, expected) in enumerate(test_cases):
            try:
                proc = subprocess.run(
                    str(exe_path),
                    input=stdin_data,
                    capture_output=True,
                    text=True,
                    timeout=timeout_sec,
                )
                actual = proc.stdout
                if actual.rstrip() != expected.rstrip():
                    failures.append(
                        f"test_case_{i}: expected={expected!r}, got={actual!r}"
                    )
                elif proc.returncode != 0:
                    failures.append(
                        f"test_case_{i}: nonzero_exit_code={proc.returncode}, stderr={proc.stderr[:200]}"
                    )
            except subprocess.TimeoutExpired:
                failures.append(f"test_case_{i}: timeout (>{timeout_sec}s)")

    except FileNotFoundError:
        failures.append(f"compiler_not_found: {compiler}")
        return TestResult(passed=False, failures=failures)
    except OSError as e:
        failures.append(f"os_error: {e}")
        return TestResult(passed=False, failures=failures)
    finally:
        # Cleanup
        for p in (src_path, exe_path):
            try:
                p.unlink(missing_ok=True)
            except OSError:
                pass

    return TestResult(passed=not failures, failures=failures)


def score_by_compile_run(answer: str, task: Dict[str, Any]) -> Optional[ScoreResult]:
    """Score a code-generation answer by compiling and running it.

    Returns None if the task has no execution_tests field.
    """
    exec_tests = task.get("execution_tests")
    if not exec_tests:
        return None

    test_cases: List[Tuple[str, str]] = [
        (str(tc.get("input", "")), str(tc.get("expected", "")))
        for tc in exec_tests
    ]
    language = str(task.get("execution_language", "c++")).lower()

    if language == "c++":
        result = compile_and_run_cpp(answer, test_cases)
    else:
        return ScoreResult(
            correct=False,
            score=0.0,
            source="test_runner",
            violations=[f"unsupported_language: {language}"],
        )

    return ScoreResult(
        correct=result.passed,
        score=1.0 if result.passed else 0.0,
        source="test_runner",
        violations=result.failures,
    )


# ── Main entry point ──────────────────────────────────────────────────


def score_task_answer(answer: str, task: Dict[str, Any]) -> ScoreResult:
    """Score a final answer using all available deterministic methods.

    Precedence:
      1. Compile/run (test_runner) — for code-generation tasks, overrides everything
      2. Rubric match (deterministic) — always applicable; baseline pass/fail
      3. Checker plugin (deterministic) — domain-specific; hard violations override rubric pass

    If rubric fails -> answer fails regardless of checker.
    If rubric passes and checker finds hard violations -> answer fails.
    If rubric passes and checker finds only soft violations -> answer passes.
    """
    # 1. Compile/run (highest precedence)
    exec_result = score_by_compile_run(answer, task)
    if exec_result is not None:
        return exec_result

    # 2. Rubric baseline (always applicable)
    rubric_result = score_by_rubric(answer, task)
    if not rubric_result.correct:
        return rubric_result

    # 3. Checker plugin (domain-specific enrichment)
    checker_result = score_by_checker(answer, task)
    if checker_result is None:
        return rubric_result

    # Merge: checker hard violations override rubric pass
    merged_violations = list(rubric_result.violations)
    merged_violations.extend(v for v in checker_result.violations if v not in merged_violations)
    hard_count = checker_result.details.get("hard_count", 0)

    return ScoreResult(
        correct=hard_count == 0,
        score=1.0 if hard_count == 0 else 0.0,
        source="deterministic",
        violations=merged_violations,
        details={
            "rubric": rubric_result.details,
            "checker": checker_result.details,
        },
    )
