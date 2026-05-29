"""Session trace JSONL logging for Phase 3B macro observation."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence


DEFAULT_TRACE_ROOT = Path("data/trace_logs")


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@dataclass
class ProcedureCall:
    call_id: str
    procedure_id: str
    procedure_name: str
    parent_call_id: Optional[str]
    args_text: str
    mutations_applied: int
    error: Optional[str]
    elapsed_seconds: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @staticmethod
    def from_dict(d: Mapping[str, Any]) -> "ProcedureCall":
        return ProcedureCall(
            call_id=str(d.get("call_id") or ""),
            procedure_id=str(d.get("procedure_id") or ""),
            procedure_name=str(d.get("procedure_name") or ""),
            parent_call_id=(str(d["parent_call_id"]) if d.get("parent_call_id") is not None else None),
            args_text=str(d.get("args_text") or ""),
            mutations_applied=int(d.get("mutations_applied") or 0),
            error=(str(d["error"]) if d.get("error") is not None else None),
            elapsed_seconds=float(d.get("elapsed_seconds") or 0.0),
        )


@dataclass
class SessionTrace:
    session_id: str
    graph_id: str
    domain: Optional[str]
    question: str
    answer: str
    correct: Optional[bool]
    budget_exhausted: bool
    procedure_errors: int
    contradiction_signals: int
    procedure_calls: List[ProcedureCall]
    budget_usage: Dict[str, Any]
    signal_ids_fired: List[str]
    elapsed_seconds: float = 0.0
    timestamp: str = field(default_factory=_now_iso)

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["procedure_calls"] = [call.to_dict() for call in self.procedure_calls]
        return data

    @staticmethod
    def from_dict(d: Mapping[str, Any]) -> "SessionTrace":
        return SessionTrace(
            session_id=str(d.get("session_id") or ""),
            graph_id=str(d.get("graph_id") or ""),
            domain=(str(d["domain"]) if d.get("domain") is not None else None),
            question=str(d.get("question") or ""),
            answer=str(d.get("answer") or ""),
            correct=(
                bool(d["correct"])
                if d.get("correct") is not None
                else None
            ),
            budget_exhausted=bool(d.get("budget_exhausted") or False),
            procedure_errors=int(d.get("procedure_errors") or 0),
            contradiction_signals=int(d.get("contradiction_signals") or 0),
            procedure_calls=[ProcedureCall.from_dict(x) for x in d.get("procedure_calls") or []],
            budget_usage=dict(d.get("budget_usage") or {}),
            signal_ids_fired=[str(x) for x in d.get("signal_ids_fired") or []],
            elapsed_seconds=float(d.get("elapsed_seconds") or 0.0),
            timestamp=str(d.get("timestamp") or _now_iso()),
        )

    def is_clean(self) -> bool:
        return (
            self.correct is True
            and not self.budget_exhausted
            and self.procedure_errors == 0
            and self.contradiction_signals == 0
        )

    def call_sequence_fingerprint(self) -> str:
        if not self.procedure_calls:
            return ""
        by_parent: Dict[Optional[str], List[ProcedureCall]] = {}
        order = {call.call_id: i for i, call in enumerate(self.procedure_calls)}
        for call in self.procedure_calls:
            by_parent.setdefault(call.parent_call_id, []).append(call)
        for calls in by_parent.values():
            calls.sort(key=lambda c: order.get(c.call_id, 0))

        names: List[str] = []

        def visit(call: ProcedureCall) -> None:
            if call.procedure_name:
                names.append(call.procedure_name)
            for child in by_parent.get(call.call_id, []):
                visit(child)

        for root in by_parent.get(None, []):
            visit(root)
        return " > ".join(names)


class TraceLogger:
    """Append/read SessionTrace rows in daily JSONL files."""

    def __init__(self, trace_root: Path = DEFAULT_TRACE_ROOT) -> None:
        self.trace_root = Path(trace_root)

    def _today_path(self) -> Path:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d")
        return self.trace_root / f"traces_{stamp}.jsonl"

    def append(self, trace: SessionTrace) -> Path:
        self.trace_root.mkdir(parents=True, exist_ok=True)
        path = self._today_path()
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(trace.to_dict(), ensure_ascii=False) + "\n")
        return path

    def read_all(self) -> List[SessionTrace]:
        traces: List[SessionTrace] = []
        if not self.trace_root.exists():
            return traces
        for path in sorted(self.trace_root.glob("traces_*.jsonl")):
            with open(path, "r", encoding="utf-8-sig") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        traces.append(SessionTrace.from_dict(json.loads(line)))
                    except Exception as exc:
                        print(f"TraceLogger.read_all: skip malformed line in {path}: {exc}")
        return traces


@dataclass
class ManualGradeResult:
    grade_rows_seen: int
    valid_grade_rows: int
    invalid_grade_rows: int
    trace_rows_updated: int
    updated_session_ids: List[str]
    missing_session_ids: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def apply_manual_grades(
    *,
    trace_root: Path = DEFAULT_TRACE_ROOT,
    grade_file: Path,
    dry_run: bool = False,
) -> ManualGradeResult:
    grades: Dict[str, bool] = {}
    rows_seen = 0
    invalid = 0
    for line in Path(grade_file).read_text(encoding="utf-8-sig").splitlines():
        if not line.strip():
            continue
        rows_seen += 1
        try:
            row = json.loads(line)
            session_id = str(row["session_id"])
            correct = row["correct"]
            if not isinstance(correct, bool):
                raise ValueError("correct must be bool")
            grades[session_id] = correct
        except Exception:
            invalid += 1

    updated: List[str] = []
    root = Path(trace_root)
    if root.exists():
        for path in sorted(root.glob("traces_*.jsonl")):
            rows: List[Dict[str, Any]] = []
            changed = False
            for line in path.read_text(encoding="utf-8-sig").splitlines():
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except Exception:
                    continue
                sid = str(row.get("session_id") or "")
                if sid in grades:
                    row["correct"] = grades[sid]
                    changed = True
                    if sid not in updated:
                        updated.append(sid)
                rows.append(row)
            if changed and not dry_run:
                path.write_text(
                    "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
                    encoding="utf-8",
                )

    missing = sorted(set(grades) - set(updated))
    return ManualGradeResult(
        grade_rows_seen=rows_seen,
        valid_grade_rows=len(grades),
        invalid_grade_rows=invalid,
        trace_rows_updated=len(updated),
        updated_session_ids=sorted(updated),
        missing_session_ids=missing,
    )


def build_procedure_calls_from_outcomes(outcomes: Sequence[Any]) -> List[ProcedureCall]:
    calls: List[ProcedureCall] = []

    def visit(outcome: Any, parent_call_id: Optional[str]) -> None:
        match = getattr(outcome, "match", None)
        procedure_name = str(
            getattr(outcome, "procedure_name", None)
            or getattr(match, "procedure_name", None)
            or _mapping_get(outcome, "procedure_name")
            or ""
        )
        call_id = str(
            getattr(outcome, "call_id", None)
            or _mapping_get(outcome, "call_id")
            or f"call_{procedure_name or len(calls)}"
        )
        call = ProcedureCall(
            call_id=call_id,
            procedure_id=str(getattr(outcome, "procedure_id", None) or _mapping_get(outcome, "procedure_id") or ""),
            procedure_name=procedure_name,
            parent_call_id=parent_call_id,
            args_text=str(getattr(match, "args_text", None) or _mapping_get(outcome, "args_text") or ""),
            mutations_applied=int(getattr(outcome, "mutations_applied", 0) or _mapping_get(outcome, "mutations_applied") or 0),
            error=(
                str(getattr(outcome, "error", None) or _mapping_get(outcome, "error"))
                if (getattr(outcome, "error", None) or _mapping_get(outcome, "error")) is not None
                else None
            ),
            elapsed_seconds=float(getattr(outcome, "elapsed_seconds", 0.0) or _mapping_get(outcome, "elapsed_seconds") or 0.0),
        )
        calls.append(call)
        for child in getattr(outcome, "sub_outcomes", None) or _mapping_get(outcome, "sub_outcomes") or []:
            visit(child, call_id)

    for outcome in outcomes:
        visit(outcome, None)
    return calls


def count_procedure_errors(outcomes: Iterable[Any]) -> int:
    total = 0
    for outcome in outcomes:
        if getattr(outcome, "error", None) or _mapping_get(outcome, "error"):
            total += 1
        total += count_procedure_errors(getattr(outcome, "sub_outcomes", None) or _mapping_get(outcome, "sub_outcomes") or [])
    return total


def build_session_trace(
    *,
    session_id: str,
    graph_id: str,
    domain: Optional[str],
    question: str,
    answer: str,
    correct: Optional[bool],
    budget_exhausted: bool,
    dispatch_outcomes: Sequence[Any],
    budget_usage: Optional[Dict[str, Any]] = None,
    signal_ids_fired: Optional[Sequence[str]] = None,
    contradiction_signals: int = 0,
    elapsed_seconds: float = 0.0,
) -> SessionTrace:
    return SessionTrace(
        session_id=session_id,
        graph_id=graph_id,
        domain=domain,
        question=question,
        answer=answer,
        correct=correct,
        budget_exhausted=budget_exhausted,
        procedure_errors=count_procedure_errors(dispatch_outcomes),
        contradiction_signals=contradiction_signals,
        procedure_calls=build_procedure_calls_from_outcomes(dispatch_outcomes),
        budget_usage=dict(budget_usage or {}),
        signal_ids_fired=[str(x) for x in signal_ids_fired or []],
        elapsed_seconds=elapsed_seconds,
    )


def _mapping_get(value: Any, key: str) -> Any:
    if isinstance(value, Mapping):
        return value.get(key)
    return None
