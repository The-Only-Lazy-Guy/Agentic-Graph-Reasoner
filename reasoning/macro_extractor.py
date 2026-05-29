"""Macro candidate extraction from Phase 3B-1 session traces.

This module is observation-only. It scans clean `SessionTrace` entries,
groups recurring procedure-call fingerprints by a metadata bucket, and
emits `MacroCandidate` JSON for offline inspection. It does not validate,
install, promote, or propose procedures.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

from reasoning.trace_log import DEFAULT_TRACE_ROOT, SessionTrace, TraceLogger


DEFAULT_CANDIDATE_ROOT = Path("data/macro_candidates")
MIN_SESSIONS = 3
MIN_PRECISION = 1.0


@dataclass(frozen=True)
class CandidateBucket:
    fingerprint: str
    graph_domain: Optional[str]
    top_level_procedure: str
    signal_free: bool
    budget_profile: str

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, object]) -> "CandidateBucket":
        return cls(
            fingerprint=str(data.get("fingerprint") or ""),
            graph_domain=(
                str(data["graph_domain"])
                if data.get("graph_domain") is not None
                else None
            ),
            top_level_procedure=str(data.get("top_level_procedure") or ""),
            signal_free=bool(data.get("signal_free")),
            budget_profile=str(data.get("budget_profile") or ""),
        )


@dataclass
class MacroCandidate:
    fingerprint: str
    fingerprint_hash: str
    bucket: CandidateBucket
    procedure_names: List[str]
    source_session_ids: List[str]
    session_count: int
    precision: float
    proposed_name: str
    status: str = "candidate"

    def to_dict(self) -> Dict[str, object]:
        data = asdict(self)
        data["bucket"] = self.bucket.to_dict()
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, object]) -> "MacroCandidate":
        return cls(
            fingerprint=str(data.get("fingerprint") or ""),
            fingerprint_hash=str(data.get("fingerprint_hash") or ""),
            bucket=CandidateBucket.from_dict(dict(data.get("bucket") or {})),
            procedure_names=[str(x) for x in data.get("procedure_names") or []],
            source_session_ids=[str(x) for x in data.get("source_session_ids") or []],
            session_count=int(data.get("session_count") or 0),
            precision=float(data.get("precision") or 0.0),
            proposed_name=str(data.get("proposed_name") or ""),
            status=str(data.get("status") or "candidate"),
        )


@dataclass
class ScanResult:
    traces_seen: int
    clean_traces_seen: int
    candidates: List[MacroCandidate]
    written_paths: List[Path]


@dataclass
class MacroStatus:
    trace_count: int
    clean_trace_count: int
    candidate_count: int
    source_session_coverage: int
    average_precision: Optional[float]
    candidate_status_counts: Dict[str, int]

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


def budget_profile_for_call_count(call_count: int) -> str:
    if call_count <= 2:
        return "low"
    if call_count <= 4:
        return "mid"
    return "high"


def procedure_names_from_trace(trace: SessionTrace) -> List[str]:
    return [c.procedure_name for c in trace.procedure_calls if c.procedure_name]


def top_level_procedure_name(trace: SessionTrace) -> str:
    for call in trace.procedure_calls:
        if call.parent_call_id is None and call.procedure_name:
            return call.procedure_name
    return ""


def bucket_for_trace(trace: SessionTrace) -> Optional[CandidateBucket]:
    fingerprint = trace.call_sequence_fingerprint()
    top_level = top_level_procedure_name(trace)
    if not fingerprint or not top_level:
        return None
    return CandidateBucket(
        fingerprint=fingerprint,
        graph_domain=trace.domain,
        top_level_procedure=top_level,
        signal_free=(len(trace.signal_ids_fired or []) == 0),
        budget_profile=budget_profile_for_call_count(len(trace.procedure_calls)),
    )


def extract_macro_candidates(
    traces: Sequence[SessionTrace],
    *,
    min_sessions: int = MIN_SESSIONS,
    min_precision: float = MIN_PRECISION,
    graph_id: Optional[str] = None,
    domain: Optional[str] = None,
) -> List[MacroCandidate]:
    """Return recurring macro candidates from clean traces only."""
    grouped: Dict[CandidateBucket, List[SessionTrace]] = {}
    for trace in traces:
        if graph_id and trace.graph_id != graph_id:
            continue
        if domain and trace.domain != domain:
            continue
        if not trace.is_clean():
            continue
        bucket = bucket_for_trace(trace)
        if bucket is None:
            continue
        grouped.setdefault(bucket, []).append(trace)

    candidates: List[MacroCandidate] = []
    for bucket, bucket_traces in grouped.items():
        distinct: Dict[str, SessionTrace] = {
            trace.session_id: trace for trace in bucket_traces
        }
        if len(distinct) < min_sessions:
            continue
        precision = sum(1 for t in distinct.values() if t.correct is True) / len(distinct)
        if precision < min_precision:
            continue
        exemplar = next(iter(distinct.values()))
        names = procedure_names_from_trace(exemplar)
        candidates.append(MacroCandidate(
            fingerprint=bucket.fingerprint,
            fingerprint_hash=_bucket_hash(bucket),
            bucket=bucket,
            procedure_names=names,
            source_session_ids=sorted(distinct),
            session_count=len(distinct),
            precision=precision,
            proposed_name=_proposed_name(names),
            status="candidate",
        ))

    candidates.sort(key=lambda c: (-c.session_count, c.fingerprint_hash))
    return candidates


def scan_trace_logs(
    *,
    trace_root: Path = DEFAULT_TRACE_ROOT,
    candidate_root: Path = DEFAULT_CANDIDATE_ROOT,
    min_sessions: int = MIN_SESSIONS,
    min_precision: float = MIN_PRECISION,
    graph_id: Optional[str] = None,
    domain: Optional[str] = None,
    dry_run: bool = False,
) -> ScanResult:
    logger = TraceLogger(Path(trace_root))
    traces = logger.read_all()
    filtered = [
        trace for trace in traces
        if (not graph_id or trace.graph_id == graph_id)
        and (not domain or trace.domain == domain)
    ]
    clean = [trace for trace in filtered if trace.is_clean()]
    candidates = extract_macro_candidates(
        filtered,
        min_sessions=min_sessions,
        min_precision=min_precision,
        graph_id=graph_id,
        domain=domain,
    )
    written: List[Path] = []
    if not dry_run:
        candidate_root = Path(candidate_root)
        candidate_root.mkdir(parents=True, exist_ok=True)
        for candidate in candidates:
            path = candidate_root / f"{candidate.fingerprint_hash}.json"
            path.write_text(
                json.dumps(candidate.to_dict(), indent=2, sort_keys=True),
                encoding="utf-8",
            )
            written.append(path)
    return ScanResult(
        traces_seen=len(filtered),
        clean_traces_seen=len(clean),
        candidates=candidates,
        written_paths=written,
    )


def load_candidates(candidate_root: Path = DEFAULT_CANDIDATE_ROOT) -> List[MacroCandidate]:
    root = Path(candidate_root)
    if not root.exists():
        return []
    candidates: List[MacroCandidate] = []
    for path in sorted(root.glob("*.json")):
        try:
            candidates.append(MacroCandidate.from_dict(json.loads(path.read_text(encoding="utf-8"))))
        except Exception:
            continue
    return candidates


def collect_status(
    *,
    trace_root: Path = DEFAULT_TRACE_ROOT,
    candidate_root: Path = DEFAULT_CANDIDATE_ROOT,
) -> MacroStatus:
    traces = TraceLogger(Path(trace_root)).read_all()
    candidates = load_candidates(candidate_root)
    source_sessions = {
        sid for candidate in candidates for sid in candidate.source_session_ids
    }
    status_counts: Dict[str, int] = {}
    for candidate in candidates:
        status_counts[candidate.status] = status_counts.get(candidate.status, 0) + 1
    avg_precision = None
    if candidates:
        avg_precision = sum(c.precision for c in candidates) / len(candidates)
    return MacroStatus(
        trace_count=len(traces),
        clean_trace_count=sum(1 for trace in traces if trace.is_clean()),
        candidate_count=len(candidates),
        source_session_coverage=len(source_sessions),
        average_precision=avg_precision,
        candidate_status_counts=status_counts,
    )


def _bucket_hash(bucket: CandidateBucket) -> str:
    payload = json.dumps(bucket.to_dict(), sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]


def _proposed_name(names: Iterable[str]) -> str:
    safe = [str(name).strip() for name in names if str(name).strip()]
    if not safe:
        return "Macro_Empty"
    return "Macro_" + "_".join(safe)
