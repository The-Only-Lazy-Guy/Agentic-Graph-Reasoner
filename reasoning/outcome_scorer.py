"""Outcome scoring infrastructure for Phase 3G.

Structured per-session outcome rows that form the training dataset for
NeedProbe, topology pruning, threshold calibration, and LoRA distillation.
"""
from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from reasoning.deterministic_scorer import ScoreResult, score_task_answer
from reasoning.reasoning_loop import ReasoningResult


@dataclass
class SubstrateOutcomeRow:
    """Structured outcome for one reasoning session (see §10.1)."""

    packet_id: str
    delta_transaction_id: str
    checker_results: List[Dict[str, Any]]
    final_answer: str
    outcome_correct: Optional[bool]
    outcome_score: float
    outcome_source: str  # "deterministic" | "test_runner" | "llm_judge" | "manual"
    step_count: int
    llm_calls: int
    task_id: str
    task_kind: str
    question: str
    elapsed_sec: float
    step_timing: List[float] = field(default_factory=list)
    activated_signal_ids: List[str] = field(default_factory=list)
    budget_usage: Dict[str, Any] = field(default_factory=dict)
    audit_summary: Dict[str, Any] = field(default_factory=dict)
    timestamp: str = ""


_OUTCOME_DIR = Path("data/outcomes")


def _ensure_dir() -> Path:
    _OUTCOME_DIR.mkdir(parents=True, exist_ok=True)
    return _OUTCOME_DIR


def _daily_path() -> Path:
    return _ensure_dir() / f"outcomes_{time.strftime('%Y%m%d')}.jsonl"


def collect_outcome_row(
    result: ReasoningResult,
    task: Dict[str, Any],
    *,
    elapsed_sec: float = 0.0,
    judge: Optional[Dict[str, Any]] = None,
) -> SubstrateOutcomeRow:
    """Construct an outcome row from a completed reasoning session."""
    audit = result.audit_summary or {}
    task_id = str(task.get("id", ""))
    question = str(task.get("question", ""))
    task_kind = str(task.get("kind", ""))

    outcome_correct: Optional[bool] = None
    outcome_score = 0.0
    outcome_source = "deterministic"

    # Phase 3G G2: deterministic scorer (rubric + checker plugins + compile/run)
    det_result: Optional[ScoreResult] = None
    try:
        det_result = score_task_answer(result.answer, task)
    except Exception:
        pass  # non-fatal; fall back to judge

    if det_result is not None:
        outcome_correct = det_result.correct
        outcome_score = det_result.score
        outcome_source = det_result.source

    if judge is not None:
        passed = judge.get("passed")
        llm_passed = bool(passed) if passed is not None else None
        rubric = judge.get("rubric")
        rubric_passed = rubric.get("passed") if rubric else None

        if outcome_correct is None:
            # Deterministic scorer had no opinion — use judge
            if rubric_passed is not None:
                outcome_correct = bool(rubric_passed)
                outcome_score = 1.0 if rubric_passed else 0.0
                outcome_source = "deterministic"
            elif llm_passed is not None:
                outcome_correct = llm_passed
                outcome_score = 1.0 if llm_passed else 0.0
                outcome_source = "llm_judge"
        elif outcome_source == "deterministic" and rubric_passed is not None:
            # Both deterministic scorer and rubric agree — prefer deterministic label
            if outcome_correct != bool(rubric_passed):
                outcome_source = "llm_judge"
                outcome_correct = llm_passed if llm_passed is not None else outcome_correct
                outcome_score = 1.0 if outcome_correct else 0.0

    checker_results: List[Dict[str, Any]] = []
    checker_break = audit.get("checker_outcome_breakdown", {})
    if checker_break:
        checker_results.append({
            "passed_strict": checker_break.get("passed_strict", 0),
            "passed_soft": checker_break.get("passed_soft", 0),
            "failed_hard": checker_break.get("failed_hard", 0),
            "failed_soft": checker_break.get("failed_soft", 0),
        })

    activated_ids: List[str] = []
    debug_dump = audit.get("debug_signal_dump", [])
    if debug_dump:
        activated_ids = list(dict.fromkeys(r["id"] for r in debug_dump))

    step_timing: List[float] = audit.get("step_timing", [])

    return SubstrateOutcomeRow(
        packet_id=f"{task_id}_{int(time.time())}_{id(result)}",
        delta_transaction_id=task_id,
        checker_results=checker_results,
        final_answer=result.answer,
        outcome_correct=outcome_correct,
        outcome_score=outcome_score,
        outcome_source=outcome_source,
        step_count=audit.get("step_count", 0),
        llm_calls=result.budget_usage.get("llm_calls", {}).get("used", 0),
        task_id=task_id,
        task_kind=task_kind,
        question=question,
        elapsed_sec=elapsed_sec,
        step_timing=step_timing,
        activated_signal_ids=activated_ids,
        budget_usage=result.budget_usage,
        audit_summary=audit,
        timestamp=time.strftime("%Y-%m-%dT%H:%M:%S"),
    )


def write_outcome_row(row: SubstrateOutcomeRow) -> Path:
    """Append one outcome row to the daily JSONL file."""
    path = _daily_path()
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(asdict(row), ensure_ascii=False) + "\n")
    return path


def outcome_count() -> int:
    """Return total number of outcome rows accumulated across all daily logs."""
    total = 0
    if not _OUTCOME_DIR.exists():
        return 0
    for p in sorted(_OUTCOME_DIR.glob("outcomes_*.jsonl")):
        with open(p, encoding="utf-8") as f:
            for _ in f:
                total += 1
    return total
