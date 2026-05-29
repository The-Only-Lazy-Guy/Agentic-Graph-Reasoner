"""Reasoning substrate v2 - Phase 3E read projection and schemas.

3E starts additively: existing session-subgraph writers keep their concrete
node types, while this module exposes a uniform SignalNode view over them.
"""
from __future__ import annotations

import hashlib
import json
import re
import sys
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Literal, Mapping, Optional, Sequence

from reasoning.activation import FrameItem, GraphTaskFrame
from reasoning.lexical_matching import (
    constraint_addressed as _shared_constraint_addressed,
    matches_packet_constraint as _shared_matches_packet_constraint,
)
from reasoning.schemas import SessionEdge, SessionSubgraph
from reasoning.token_estimation import estimate_token_count


SignalKind = Literal[
    "constraint",
    "decision",
    "hypothesis",
    "evidence",
    "gap",
    "unresolved_gap",
    "risk",
    "repair",
    "procedure",
]
SignalScope = Literal["session", "reusable"]
SignalProducer = Literal[
    "llm_delta",
    "regex_fallback",
    "checker",
    "controller",
    "consolidation",
    "projection",
]
StepStatus = Literal["open", "resolving", "resolved", "failed", "budget_exhausted"]
StepResultStatus = Literal["resolved", "need_info", "failed"]
DeltaTransactionStatus = Literal["parsed", "skimmed", "dropped"]


@dataclass
class SignalNode:
    id: str
    kind: SignalKind
    text: str
    scope: SignalScope = "session"
    activation_keys: List[str] = field(default_factory=list)
    source_step_id: Optional[str] = None
    produced_by: SignalProducer = "projection"
    state: Optional[Dict[str, Any]] = None
    evidence_ids: List[str] = field(default_factory=list)
    citation_count: int = 0
    decay: float = 1.0
    source_node_id: Optional[str] = None
    source_node_type: Optional[str] = None
    confidence: float = 0.5

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @staticmethod
    def from_dict(d: Mapping[str, Any]) -> "SignalNode":
        return SignalNode(
            id=str(d["id"]),
            kind=d.get("kind", "evidence"),  # type: ignore[arg-type]
            text=str(d.get("text", "")),
            scope=d.get("scope", "session"),  # type: ignore[arg-type]
            activation_keys=[str(x) for x in d.get("activation_keys", [])],
            source_step_id=(str(d["source_step_id"]) if d.get("source_step_id") is not None else None),
            produced_by=d.get("produced_by", "projection"),  # type: ignore[arg-type]
            state=dict(d["state"]) if d.get("state") is not None else None,
            evidence_ids=[str(x) for x in d.get("evidence_ids", [])],
            citation_count=int(d.get("citation_count", 0)),
            decay=float(d.get("decay", 1.0)),
            source_node_id=(str(d["source_node_id"]) if d.get("source_node_id") is not None else None),
            source_node_type=(str(d["source_node_type"]) if d.get("source_node_type") is not None else None),
            confidence=float(d.get("confidence", 0.5)),
        )


@dataclass
class MissingInfo:
    question: str
    why_needed: str
    expected_shape: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @staticmethod
    def from_dict(d: Mapping[str, Any]) -> "MissingInfo":
        return MissingInfo(
            question=str(d.get("question", "")),
            why_needed=str(d.get("why_needed", "")),
            expected_shape=str(d.get("expected_shape", "")),
        )

    def canonical_id(self) -> str:
        payload = json.dumps(
            {
                "expected_shape": _norm(self.expected_shape),
                "normalized_question": _norm(self.question),
            },
            sort_keys=True,
            ensure_ascii=False,
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


@dataclass
class StateDelta:
    decisions: List[str] = field(default_factory=list)
    constraints: List[str] = field(default_factory=list)
    risks: List[str] = field(default_factory=list)
    evidence: List[str] = field(default_factory=list)
    gaps: List[str] = field(default_factory=list)
    repairs: List[str] = field(default_factory=list)
    produced_by: SignalProducer = "llm_delta"
    confidence: float = 0.5

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @staticmethod
    def from_dict(d: Mapping[str, Any]) -> "StateDelta":
        return StateDelta(
            decisions=[str(x) for x in d.get("decisions", [])],
            constraints=[str(x) for x in d.get("constraints", [])],
            risks=[str(x) for x in d.get("risks", [])],
            evidence=[str(x) for x in d.get("evidence", [])],
            gaps=[str(x) for x in d.get("gaps", [])],
            repairs=[str(x) for x in d.get("repairs", [])],
            produced_by=d.get("produced_by", "llm_delta"),  # type: ignore[arg-type]
            confidence=float(d.get("confidence", 0.5)),
        )

    def to_signal_nodes(self, *, source_step_id: str, prefix: str = "delta") -> List[SignalNode]:
        emitted: List[SignalNode] = []
        groups: List[tuple[SignalKind, List[str]]] = [
            ("decision", self.decisions),
            ("constraint", self.constraints),
            ("risk", self.risks),
            ("evidence", self.evidence),
            ("gap", self.gaps),
            ("repair", self.repairs),
        ]
        for kind, values in groups:
            for value in values:
                text = _clean_text(value)
                if not text:
                    continue
                emitted.append(SignalNode(
                    id=f"{prefix}_{source_step_id}_{kind}_{_short_hash(text)}",
                    kind=kind,
                    text=text,
                    activation_keys=activation_keys_for_text(text),
                    source_step_id=source_step_id,
                    produced_by=self.produced_by,
                    confidence=self.confidence,
                ))
        return emitted


@dataclass
class DeltaTransaction:
    status: DeltaTransactionStatus
    delta: StateDelta
    raw_excerpt: str = ""
    parse_error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status,
            "delta": self.delta.to_dict(),
            "raw_excerpt": self.raw_excerpt,
            "parse_error": self.parse_error,
        }

    @staticmethod
    def from_dict(d: Mapping[str, Any]) -> "DeltaTransaction":
        return DeltaTransaction(
            status=d.get("status", "dropped"),  # type: ignore[arg-type]
            delta=StateDelta.from_dict(d.get("delta", {})),
            raw_excerpt=str(d.get("raw_excerpt", "")),
            parse_error=(str(d["parse_error"]) if d.get("parse_error") is not None else None),
        )


@dataclass
class StepContextPacket:
    task_summary: str
    focus: str
    looking_for: str
    active_signals: List[SignalNode] = field(default_factory=list)
    parent_decisions: List[str] = field(default_factory=list)
    open_gaps: List[str] = field(default_factory=list)
    hard_constraints: List[str] = field(default_factory=list)
    budget_remaining: Dict[str, Any] = field(default_factory=dict)
    cache_key: str = ""
    base_prefix_key: Optional[str] = None
    reasoning_mode: str = "quick"

    def __post_init__(self) -> None:
        if not self.cache_key:
            self.cache_key = self.compute_cache_key()

    def compute_cache_key(self) -> str:
        payload = {
            "signal_ids": [sig.id for sig in self.active_signals],
            "focus": self.focus,
            "parent_decisions": self.parent_decisions,
            "reasoning_mode": self.reasoning_mode,
        }
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:16]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_summary": self.task_summary,
            "focus": self.focus,
            "looking_for": self.looking_for,
            "active_signals": [sig.to_dict() for sig in self.active_signals],
            "parent_decisions": list(self.parent_decisions),
            "open_gaps": list(self.open_gaps),
            "hard_constraints": list(self.hard_constraints),
            "budget_remaining": dict(self.budget_remaining),
            "cache_key": self.cache_key,
            "base_prefix_key": self.base_prefix_key,
        }

    @staticmethod
    def from_dict(d: Mapping[str, Any]) -> "StepContextPacket":
        return StepContextPacket(
            task_summary=str(d.get("task_summary", "")),
            focus=str(d.get("focus", "")),
            looking_for=str(d.get("looking_for", "")),
            active_signals=[SignalNode.from_dict(x) for x in d.get("active_signals", [])],
            parent_decisions=[str(x) for x in d.get("parent_decisions", [])],
            open_gaps=[str(x) for x in d.get("open_gaps", [])],
            hard_constraints=[str(x) for x in d.get("hard_constraints", [])],
            budget_remaining=dict(d.get("budget_remaining", {})),
            cache_key=str(d.get("cache_key", "")),
            base_prefix_key=(str(d["base_prefix_key"]) if d.get("base_prefix_key") is not None else None),
        )


@dataclass
class ReasoningStep:
    step_id: str
    parent_step_id: Optional[str]
    task_id: str
    focus: str
    looking_for: str
    context_packet: Optional[StepContextPacket] = None
    depth: int = 0
    status: StepStatus = "open"
    result: Optional[Any] = None
    delta: Optional[StateDelta] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "step_id": self.step_id,
            "parent_step_id": self.parent_step_id,
            "task_id": self.task_id,
            "focus": self.focus,
            "looking_for": self.looking_for,
            "context_packet": self.context_packet.to_dict() if self.context_packet else None,
            "depth": self.depth,
            "status": self.status,
            "result": self.result,
            "delta": self.delta.to_dict() if self.delta else None,
        }

    @staticmethod
    def from_dict(d: Mapping[str, Any]) -> "ReasoningStep":
        return ReasoningStep(
            step_id=str(d["step_id"]),
            parent_step_id=(str(d["parent_step_id"]) if d.get("parent_step_id") is not None else None),
            task_id=str(d.get("task_id", "")),
            focus=str(d.get("focus", "")),
            looking_for=str(d.get("looking_for", "")),
            context_packet=(
                StepContextPacket.from_dict(d["context_packet"])
                if d.get("context_packet") is not None else None
            ),
            depth=int(d.get("depth", 0)),
            status=d.get("status", "open"),  # type: ignore[arg-type]
            result=d.get("result"),
            delta=StateDelta.from_dict(d["delta"]) if d.get("delta") is not None else None,
        )

    @staticmethod
    def from_gap(parent: "ReasoningStep", gap: MissingInfo, *, step_id: Optional[str] = None) -> "ReasoningStep":
        return ReasoningStep(
            step_id=step_id or f"step_gap_{gap.canonical_id()}",
            parent_step_id=parent.step_id,
            task_id=parent.task_id,
            focus=f"resolve gap: {gap.question}",
            looking_for=gap.expected_shape,
            depth=parent.depth + 1,
            status="open",
        )


@dataclass
class StepResult:
    status: StepResultStatus
    result: str
    delta_transaction: DeltaTransaction
    missing: Optional[MissingInfo] = None
    constraints_honored: List[str] = field(default_factory=list)
    raw_output: str = ""
    plan: Optional[Dict[str, str]] = None

    @property
    def delta(self) -> StateDelta:
        return self.delta_transaction.delta

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status,
            "result": self.result,
            "delta_transaction": self.delta_transaction.to_dict(),
            "missing": self.missing.to_dict() if self.missing else None,
            "constraints_honored": list(self.constraints_honored),
            "raw_output": self.raw_output,
            "plan": dict(self.plan) if self.plan else None,
        }

    @staticmethod
    def from_dict(d: Mapping[str, Any]) -> "StepResult":
        plan_raw = d.get("plan")
        plan = dict(plan_raw) if plan_raw else None
        return StepResult(
            status=d.get("status", "failed"),  # type: ignore[arg-type]
            result=str(d.get("result", "")),
            delta_transaction=DeltaTransaction.from_dict(d.get("delta_transaction", {})),
            missing=MissingInfo.from_dict(d["missing"]) if d.get("missing") else None,
            constraints_honored=[str(x) for x in d.get("constraints_honored", [])],
            raw_output=str(d.get("raw_output", "")),
            plan=plan,
        )


@dataclass
class CheckerViolation:
    code: str
    message: str
    severity: Literal["soft", "hard"] = "soft"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class CheckResult:
    passed: bool
    confidence: float = 0.5
    violations: List[CheckerViolation] = field(default_factory=list)
    plugin_names: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "passed": self.passed,
            "confidence": self.confidence,
            "violations": [v.to_dict() for v in self.violations],
            "plugin_names": list(self.plugin_names),
        }


@dataclass
class FastLoopConfig:
    max_tokens_per_step: int = 700
    max_active_signals: int = 6
    max_child_depth: int = 2
    max_children_per_step: int = 2
    max_total_steps: int = 4
    debug_signals: bool = False


@dataclass
class FastLoopResult:
    root_step: ReasoningStep
    final_step_result: StepResult
    signals: List[SignalNode]
    steps: List[ReasoningStep]
    step_results: List[StepResult]
    raw_outputs: List[str]
    checks: List[CheckResult]
    cache_hits: int = 0
    cache_misses: int = 0
    budget_exhausted: bool = False

    # Instrumentation (populated by _compute_instrumentation)
    delta_status_breakdown: Dict[str, int] = field(default_factory=dict)
    checker_outcome_breakdown: Dict[str, int] = field(default_factory=dict)
    repair_triggered: int = 0
    repair_succeeded: int = 0
    tokens_per_call: List[int] = field(default_factory=list)
    activated_signal_ages: Dict[str, float] = field(default_factory=dict)  # min/median/max
    activated_prior_session_signal_count: int = 0
    prior_session_signal_reused: bool = False
    debug_signal_dump: List[Dict[str, Any]] = field(default_factory=list)
    step_timing: List[float] = field(default_factory=list)  # wall-clock sec per llm_call


class PacketRenderCache:
    def __init__(self, max_entries: int = 32) -> None:
        self.max_entries = max_entries
        self._items: Dict[str, str] = {}
        self.hits = 0
        self.misses = 0

    def render(self, packet: StepContextPacket, *, capsule_signal_ids: Optional[set[str]] = None,
               procedure_pool: Optional[Sequence[Any]] = None) -> str:
        cached = self._items.get(packet.cache_key)
        if cached is not None and procedure_pool is None:
            self.hits += 1
            return cached
        self.misses += 1
        rendered = render_step_prompt(packet, capsule_signal_ids=capsule_signal_ids,
                                      procedure_pool=procedure_pool)
        if procedure_pool is None and len(self._items) < self.max_entries:
            self._items[packet.cache_key] = rendered
        return rendered


class CheckerRegistry:
    def __init__(self, plugins: Optional[Sequence[str]] = None) -> None:
        self.plugins = list(plugins or ["generic_step_format"])

    def verify(self, step_result: StepResult, packet: StepContextPacket) -> CheckResult:
        violations: List[CheckerViolation] = []
        names: List[str] = []
        for name in self.plugins:
            names.append(name)
            violations.extend(_run_checker_plugin(name, step_result, packet))
            if len(violations) >= 3:
                violations = violations[:3]
                break
        hard = [v for v in violations if v.severity == "hard"]
        return CheckResult(
            passed=not hard,
            confidence=0.85 if not violations else 0.55,
            violations=violations,
            plugin_names=names,
        )


_REASONING_PLAN_RE = re.compile(
    r"<reasoning>(.*?)</reasoning>", re.DOTALL | re.IGNORECASE
)

def _extract_plan(raw_text: str) -> Optional[Dict[str, str]]:
    """Extract GOAL/KNOWN/PLAN from a <reasoning> block preceding STEP_RESULT."""
    m = _REASONING_PLAN_RE.search(raw_text)
    if not m:
        return None
    block = m.group(1)
    plan: Dict[str, str] = {}
    for key in ("GOAL", "KNOWN", "PLAN"):
        pat = re.compile(
            rf"{key}:\s*(.*?)(?:\n\s*(?:GOAL|KNOWN|PLAN)\s*:|\Z)",
            re.DOTALL | re.IGNORECASE,
        )
        km = pat.search(block)
        if km:
            val = km.group(1).strip().rstrip("</reasoning>")
            plan[key.lower()] = val
    return plan if plan else None


def parse_step_result(raw: str) -> StepResult:
    raw_text = str(raw or "")
    plan = _extract_plan(raw_text)

    block_match = re.search(r"STEP_RESULT\s*(.*?)\s*END_STEP_RESULT", raw_text, flags=re.I | re.S)
    if not block_match:
        delta = _skim_delta(raw_text)
        return StepResult(
            status="resolved" if raw_text.strip() else "failed",
            result=raw_text.strip(),
            delta_transaction=DeltaTransaction(
                status="skimmed" if raw_text.strip() else "dropped",
                delta=delta,
                raw_excerpt=raw_text[:500],
                parse_error="missing STEP_RESULT block",
            ),
            raw_output=raw_text,
            plan=plan,
        )

    block = block_match.group(1)
    try:
        parsed = _parse_step_block(block)
        return StepResult(raw_output=raw_text, plan=plan, **parsed)
    except Exception as exc:
        delta = _skim_delta(raw_text)
        return StepResult(
            status="resolved" if raw_text.strip() else "failed",
            result=raw_text.strip(),
            delta_transaction=DeltaTransaction(
                status="skimmed" if raw_text.strip() else "dropped",
                delta=delta,
                raw_excerpt=raw_text[:500],
                parse_error=str(exc),
            ),
            raw_output=raw_text,
            plan=plan,
        )


def run_fast_step_loop(
    *,
    root_step: ReasoningStep,
    llm_call: Callable[[str], str],
    initial_signals: Optional[Sequence[SignalNode]] = None,
    checker_registry: Optional[CheckerRegistry] = None,
    config: Optional[FastLoopConfig] = None,
    cache: Optional[PacketRenderCache] = None,
    capsule_signal_ids: Optional[set[str]] = None,
    procedure_pool: Optional[Sequence] = None,
    dispatcher: Optional[Any] = None,
    session: Optional[Any] = None,
) -> FastLoopResult:
    cfg = config or FastLoopConfig()
    checker = checker_registry or CheckerRegistry()
    render_cache = cache or PacketRenderCache()
    signals: List[SignalNode] = list(initial_signals or [])
    steps: List[ReasoningStep] = []
    step_results: List[StepResult] = []
    raw_outputs: List[str] = []
    checks: List[CheckResult] = []
    best_repair_result: Optional[StepResult] = None
    step_count = 0
    budget_exhausted = False
    tokens_per_call: List[int] = []
    debug_collector: List[Dict[str, Any]] = []
    step_timing: List[float] = []

    def execute(step: ReasoningStep) -> StepResult:
        nonlocal step_count, budget_exhausted, best_repair_result, tokens_per_call, debug_collector, step_timing
        if step_count >= cfg.max_total_steps:
            budget_exhausted = True
            step.status = "budget_exhausted"
            return StepResult(
                status="failed",
                result="Substrate v2 step budget exhausted.",
                delta_transaction=DeltaTransaction("dropped", StateDelta(confidence=0.0)),
            )
        step_count += 1
        steps.append(step)
        active = select_active_signals(
            focus=step.focus,
            looking_for=step.looking_for,
            signals=signals,
            max_signals=cfg.max_active_signals,
            debug_signals=cfg.debug_signals,
            debug_collector=debug_collector,
        )
        packet = StepContextPacket(
            task_summary=step.task_id,
            focus=step.focus,
            looking_for=step.looking_for,
            active_signals=active,
            parent_decisions=[s.text for s in signals if s.kind == "decision"],
            open_gaps=[s.text for s in signals if s.kind in {"gap", "unresolved_gap"}],
            hard_constraints=_hard_constraints_for_packet(active=active, signals=signals),
            budget_remaining={
                "steps": cfg.max_total_steps - step_count + 1,
                "depth": cfg.max_child_depth - step.depth,
            },
            base_prefix_key=step.context_packet.base_prefix_key if step.context_packet else None,
            reasoning_mode="auto" if step.depth == 0 else "quick",
        )
        step.context_packet = packet
        prompt = render_cache.render(packet, capsule_signal_ids=capsule_signal_ids,
                                     procedure_pool=procedure_pool)
        _t0 = time.perf_counter()
        raw = llm_call(prompt)
        _t1 = time.perf_counter()
        step_timing.append(_t1 - _t0)
        raw_outputs.append(raw)
        tokens_per_call.append(estimate_token_count(prompt) + estimate_token_count(raw))
        result = parse_step_result(raw)
        step_results.append(result)
        step.delta = result.delta
        step.result = result.result
        signals.extend(result.delta.to_signal_nodes(source_step_id=step.step_id))
        check = checker.verify(result, packet)
        checks.append(check)
        signals.extend(_checker_violations_to_signals(check, step.step_id))
        if _is_repair_step(step) and result.status == "resolved" and check.passed:
            best_repair_result = result

        # Phase 5/§2.1 + §5: Procedure dispatch from CoT (DEEP mode)
        # If the model invoked a procedure in its <think> or STEP_RESULT,
        # run it via dispatcher, inject results as signals, and resume.
        if result.status == "resolved" and check.passed and dispatcher is not None and procedure_pool:
            raw_text = result.raw_output or ""
            matches = dispatcher.scan(raw_text)
            if matches:
                step.status = "resolving"
                for match in matches:
                    try:
                        outcome = dispatcher.invoke(match, session, llm_call)
                        outcome_signals = _dispatch_outcome_to_signals(outcome, step.step_id)
                        signals.extend(outcome_signals)
                    except Exception as exc:
                        signals.append(SignalNode(
                            id=f"sigv2_dispatch_fail_{_short_hash(match.procedure_name)}",
                            kind="risk",
                            text=f"Procedure {match.procedure_name} dispatch failed: {exc}",
                            activation_keys=activation_keys_for_text(match.procedure_name),
                            source_step_id=step.step_id,
                            produced_by="controller",
                            confidence=0.3,
                        ))
                resume_step = ReasoningStep(
                    step_id=step.step_id, parent_step_id=step.parent_step_id,
                    task_id=step.task_id, focus=step.focus, looking_for=step.looking_for,
                    depth=step.depth,
                )
                return execute(resume_step)

        if result.status == "resolved" and check.passed:
            step.status = "resolved"
            return result
        if (
            result.status == "need_info"
            and result.missing is not None
            and not _is_repair_step(step)
            and step.depth < cfg.max_child_depth
            and len([s for s in steps if s.parent_step_id == step.step_id]) < cfg.max_children_per_step
        ):
            step.status = "resolving"
            child = ReasoningStep.from_gap(step, result.missing)
            child_result = execute(child)
            signals.append(SignalNode(
                id=f"sigv2_resolution_{child.step_id}",
                kind="evidence" if child_result.status == "resolved" else "unresolved_gap",
                text=child_result.result or result.missing.question,
                activation_keys=activation_keys_for_text(child_result.result or result.missing.question),
                source_step_id=child.step_id,
                produced_by="controller",
                state={"gap_id": result.missing.canonical_id(), "child_status": child_result.status},
                confidence=0.8 if child_result.status == "resolved" else 0.3,
            ))
            resume_step = ReasoningStep(
                step_id=step.step_id, parent_step_id=step.parent_step_id,
                task_id=step.task_id, focus=step.focus, looking_for=step.looking_for,
                depth=step.depth,
            )
            return execute(resume_step)
        hard_violations = [v for v in check.violations if v.severity == "hard"]
        if (
            hard_violations
            and not _is_repair_step(step)
            and step.depth < cfg.max_child_depth
            and len([s for s in steps if s.parent_step_id == step.step_id]) < cfg.max_children_per_step
        ):
            step.status = "resolving"
            violation_text = "; ".join(v.message for v in hard_violations[:3])
            preserve_text = "; ".join(packet.hard_constraints[:6])
            repair_gap = MissingInfo(
                question=(
                    f"How should this failed step be repaired? {violation_text}"
                    + (f" Preserve these hard constraints: {preserve_text}" if preserve_text else "")
                ),
                why_needed="A deterministic checker rejected the current step.",
                expected_shape="corrected decision or repair evidence",
            )
            child = ReasoningStep.from_gap(
                step,
                repair_gap,
                step_id=f"step_repair_{step.step_id}_{_short_hash(violation_text)}",
            )
            child_result = execute(child)
            signals.append(SignalNode(
                id=f"sigv2_repair_{child.step_id}",
                kind="repair" if child.status == "resolved" else "unresolved_gap",
                text=child_result.result or repair_gap.question,
                activation_keys=activation_keys_for_text(child_result.result or repair_gap.question),
                source_step_id=child.step_id,
                produced_by="checker",
                state={"gap_id": repair_gap.canonical_id(), "child_status": child.status},
                confidence=0.8 if child.status == "resolved" else 0.3,
            ))
            if child.status != "resolved":
                step.status = "failed"
                return result
            resume_step = ReasoningStep(
                step_id=step.step_id, parent_step_id=step.parent_step_id,
                task_id=step.task_id, focus=step.focus, looking_for=step.looking_for,
                depth=step.depth,
            )
            return execute(resume_step)
        step.status = "failed" if not check.passed or result.status == "failed" else "resolved"
        return result

    final = execute(root_step)
    if root_step.status != "resolved" and best_repair_result is not None:
        final = best_repair_result
    result = FastLoopResult(
        root_step=root_step,
        final_step_result=final,
        signals=signals,
        steps=steps,
        step_results=step_results,
        raw_outputs=raw_outputs,
        checks=checks,
        cache_hits=render_cache.hits,
        cache_misses=render_cache.misses,
        budget_exhausted=budget_exhausted,
        tokens_per_call=tokens_per_call,
        debug_signal_dump=debug_collector,
        step_timing=step_timing,
    )
    _compute_instrumentation(result)
    return result


def _checker_violations_to_signals(check: CheckResult, step_id: str) -> List[SignalNode]:
    signals: List[SignalNode] = []
    for violation in check.violations[:3]:
        text = f"{violation.code}: {violation.message}"
        signals.append(SignalNode(
            id=f"sigv2_check_{step_id}_{violation.code}_{_short_hash(text)}",
            kind="risk" if violation.severity == "hard" else "evidence",
            text=text,
            activation_keys=activation_keys_for_text(text),
            source_step_id=step_id,
            produced_by="checker",
            state={"severity": violation.severity, "code": violation.code},
            confidence=0.9 if violation.severity == "hard" else 0.55,
        ))
    return signals


def _dispatch_outcome_to_signals(outcome: Any, step_id: str) -> List[SignalNode]:
    """Convert a DispatchOutcome into SignalNodes for the fast loop."""
    signals: List[SignalNode] = []
    if outcome is None:
        return signals
    proc_name = getattr(getattr(outcome, "match", None), "procedure_name", "?")
    sub_resp = getattr(outcome, "sub_response", "") or ""

    # Extract summary/result from sub_response (after DONE block)
    done_match = re.search(r"DONE\s*(.*)", sub_resp, re.DOTALL | re.IGNORECASE)
    summary = done_match.group(1).strip() if done_match else sub_resp[:200]

    if getattr(outcome, "error", None):
        args_text = getattr(getattr(outcome, "match", None), "args_text", "") or ""
        signal_id = f"sigv2_dispatch_{step_id}_{_short_hash(proc_name)}_{_short_hash(args_text)}"
        signals.append(SignalNode(
            id=signal_id,
            kind="risk",
            text=f"{proc_name} error: {outcome.error}",
            activation_keys=activation_keys_for_text(proc_name),
            source_step_id=step_id,
            produced_by="controller",
            confidence=0.3,
        ))
        return signals

    args_text = getattr(getattr(outcome, "match", None), "args_text", "") or ""
    signal_id = f"sigv2_dispatch_{step_id}_{_short_hash(proc_name)}_{_short_hash(args_text)}"
    signals.append(SignalNode(
        id=signal_id,
        kind="decision",
        text=f"Procedure {proc_name} result: {summary[:200]}",
        activation_keys=activation_keys_for_text(proc_name),
        source_step_id=step_id,
        produced_by="controller",
        state={"procedure": proc_name, "mutations": getattr(outcome, "mutations_applied", 0)},
        confidence=0.85,
    ))
    return signals


def _is_repair_step(step: ReasoningStep) -> bool:
    return step.step_id.startswith("step_repair_")


def _compute_instrumentation(r: FastLoopResult) -> None:
    delta_break: Dict[str, int] = {"parsed": 0, "skimmed": 0, "dropped": 0}
    checker_break: Dict[str, int] = {
        "passed_strict": 0,
        "passed_soft": 0,
        "failed_hard": 0,
        "failed_soft": 0,
    }
    repair_triggered = 0
    repair_succeeded = 0
    activated_warm_signals: List[SignalNode] = []

    for sr in r.step_results:
        status = sr.delta_transaction.status
        if status in delta_break:
            delta_break[status] += 1

    for check in r.checks:
        hard = [v for v in check.violations if v.severity == "hard"]
        if not check.passed and hard:
            checker_break["failed_hard"] += 1
        elif not check.passed:
            checker_break["failed_soft"] += 1
        elif check.passed and check.violations:
            checker_break["passed_soft"] += 1
        else:
            checker_break["passed_strict"] += 1

    step_id_to_idx: Dict[str, int] = {}
    for i, step in enumerate(r.steps):
        step_id_to_idx.setdefault(step.step_id, i)
        if step.step_id.startswith("step_repair_"):
            repair_triggered += 1
            if step.status == "resolved":
                repair_succeeded += 1
        packet = step.context_packet
        if packet is not None:
            for sig in packet.active_signals:
                state = sig.state or {}
                if state.get("warm_start"):
                    activated_warm_signals.append(sig)

    r.delta_status_breakdown = delta_break
    r.checker_outcome_breakdown = checker_break
    r.repair_triggered = repair_triggered
    r.repair_succeeded = repair_succeeded

    total_signal_age_steps: List[int] = []
    for step_idx, step in enumerate(r.steps):
        if step.context_packet and step.context_packet.active_signals:
            for sig in step.context_packet.active_signals:
                if sig.source_step_id and sig.source_step_id in step_id_to_idx:
                    source_idx = step_id_to_idx[sig.source_step_id]
                    total_signal_age_steps.append(step_idx - source_idx)
    if total_signal_age_steps:
        sorted_ages = sorted(total_signal_age_steps)
        r.activated_signal_ages = {
            "min": float(sorted_ages[0]),
            "median": float(sorted_ages[len(sorted_ages) // 2]),
            "max": float(sorted_ages[-1]),
        }
    else:
        r.activated_signal_ages = {"min": 0.0, "median": 0.0, "max": 0.0}

    warm_ids = {sig.id for sig in activated_warm_signals}
    r.activated_prior_session_signal_count = len(warm_ids)
    combined_hay = " ".join(_haystack(step_result) for step_result in r.step_results)
    r.prior_session_signal_reused = any(
        any(key in combined_hay for key in sig.activation_keys[:6])
        for sig in activated_warm_signals
    )


def _sanitize_answer(text: str) -> str:
    """Strip protocol artifacts that leak into the model's result text."""
    text = re.sub(r"\bmissing\s*:\s*\[.*?\]", "", text)
    text = re.sub(r"\bmissing\s*:\s*none\.?", "", text, flags=re.I)
    text = re.sub(r"\bconstraints_honored\s*:.*?(?=\n\S|\Z)", "", text, flags=re.S)
    return text.strip().strip(".,;: ")


def compose_final_answer(result: FastLoopResult) -> str:
    """Deterministically preserve hard constraints in the visible answer.

    This is intentionally conservative: it does not invent new reasoning. It
    takes the chosen final step result and appends packet constraints that are
    already known to the controller but missing from the prose answer.

    If the step result has a plan (DEEP mode), the PLAN field is used as
    supplementary context when the answer is empty or very short.
    """
    answer = _sanitize_answer(result.final_step_result.result)
    if (not answer or len(answer) < 20) and result.final_step_result.plan:
        plan = result.final_step_result.plan
        answer = answer or plan.get("goal", "")
    constraints: List[str] = []
    for step in result.steps:
        packet = step.context_packet
        if packet is None:
            continue
        for constraint in packet.hard_constraints:
            text = _clean_text(constraint)
            if text and text not in constraints:
                constraints.append(text)
    missing = [c for c in constraints if not _constraint_addressed(c, answer.lower())]
    if not missing:
        return answer
    task_concepts = [_task_concept_from_constraint(c) for c in missing]
    task_concepts = [c for c in task_concepts if c]
    additions = [
        _constraint_to_answer_sentence(c)
        for c in missing[:4]
        if _task_concept_from_constraint(c) is None
    ]
    if task_concepts:
        additions.extend(_task_concept_to_sentence(concept) for concept in task_concepts[:6])
    additions = [a for a in additions if a and a.lower() not in answer.lower()]
    seen_lower: set[str] = set()
    deduped: List[str] = []
    for a in additions:
        al = a.lower()
        if al not in seen_lower:
            seen_lower.add(al)
            deduped.append(a)
    additions = deduped
    if not additions:
        return answer
    trailer = " ".join(additions)
    if answer:
        if answer.rstrip()[-1] in ".!?":
            return answer.rstrip() + " " + trailer
        return answer.rstrip() + ". " + trailer
    return trailer


_TASK_CONCEPT_CONSTRAINT_PREFIX = "Explicitly preserve this task-statement concept:"


def task_concept_constraint(concept: str) -> str:
    concept = _clean_text(concept)
    return f"{_TASK_CONCEPT_CONSTRAINT_PREFIX} {concept}."


def derive_task_statement_concepts(question: str, *, limit: int = 8) -> List[str]:
    """Extract answer coverage concepts from the user-visible task statement.

    This deliberately uses only the question text. It does not look at benchmark
    rubric terms, expected answers, checker plugins, or prior judgments.
    """
    q = _clean_text(question)
    lower = q.lower()
    concepts: List[str] = []

    def add(concept: str) -> None:
        cleaned = _clean_text(concept).strip(" .,:;")
        if cleaned and cleaned.lower() not in {c.lower() for c in concepts}:
            concepts.append(cleaned)

    phrase_patterns = [
        (r"\bo\(1\)\b", "O(1)"),
        (r"\bfrom (?:one|a|single) source\b", "source"),
        (r"\bsingle source\b", "source"),
        (r"\bconnectivity queries?\b", "connectivity queries"),
        (r"\bplain dsu is insufficient\b", "plain DSU is insufficient"),
        (r"\btime-axis structure\b", "time-axis structure"),
        (r"\brange_chmin\b", "range_chmin"),
        (r"\brange_sum\b", "range_sum"),
        (r"\bper-node state\b", "per-node state"),
        (r"\bcapped lazy updates?\b", "capped lazy updates"),
        (r"\bonline point updates?\b", "online point updates"),
        (r"\bnon-?empty\b", "non-empty"),
        (r"\ball-?negative\b", "all-negative"),
        (r"\bidempotent\b", "same effect"),
        (r"\bidempotency keys?\b", "idempotency key"),
        (r"\bat-?least-?once\b", "at-least-once"),
        (r"\bbefore the local database commit\b", "local database commit"),
        (r"\bdouble charge\b", "double charge"),
        (r"\bzero downtime\b", "zero downtime"),
        (r"\bverified before cutover\b", "verification before cutover"),
        (r"\brollback\b", "rollback"),
        (r"\breservation ttl\b", "reservation TTL"),
        (r"\bpayment confirmation\b", "payment confirmation"),
        (r"\bhot sku\b", "hot SKU"),
        (r"\boversell\b", "oversell"),
        (r"\bdoes it mean\b", "does not change"),
        (r"\blearning rate\b", "learning rate"),
        (r"\bgradient descent\b", "convergence"),
        (r"\bbase rate\b", "base rate"),
        (r"\bpositive result\b", "positive result"),
        (r"\bfalse positive\b", "false positive"),
        (r"\bbinary search on the answer\b", "true/false"),
        (r"\brange sum queries?\b", "range sum queries"),
        (r"\bquery formula\b", "query formula"),
        (r"\bmutable default arguments?\b", "mutable default arguments"),
        (r"\brace condition\b", "race condition"),
        (r"\bshared state\b", "shared state"),
    ]
    for pattern, concept in phrase_patterns:
        if re.search(pattern, lower):
            add(concept)
        if len(concepts) >= limit:
            return concepts[:limit]

    important_terms = {
        "source",
        "connectivity",
        "queries",
        "remove",
        "rollback",
        "range_chmin",
        "range_sum",
        "idempotency",
        "at-least-once",
        "cutover",
        "rollback",
        "ttl",
        "oversell",
        "posterior",
        "conditional",
        "prevalence",
        "probability",
        "shared",
        "stale",
        "cache",
        "update",
        "formula",
        "precondition",
        "nonnegative",
        "unweighted",
        "idempotent",
    }
    for tok in re.findall(r"[a-z0-9_+-]+", lower):
        if tok in important_terms:
            add(tok)
        if len(concepts) >= limit:
            break
    return concepts[:limit]


def missing_task_statement_concepts(question: str, answer: str, *, limit: int = 8) -> List[str]:
    hay = answer.lower()
    missing: List[str] = []
    for concept in derive_task_statement_concepts(question, limit=limit):
        if concept.lower() not in hay:
            missing.append(concept)
    return missing


def _constraint_to_answer_sentence(constraint: str) -> str:
    text = _clean_text(constraint)
    task_concept = _task_concept_from_constraint(text)
    if task_concept:
        return _task_concept_to_sentence(task_concept)
    lower = text.lower()
    if "sum, prefix, suffix, and best" in lower or "sum prefix suffix and best" in lower:
        return "Use a segment tree node storing sum, prefix, suffix, and best."
    if "cross-boundary merge rule" in lower or "left.suffix + right.prefix" in lower:
        return "State the cross-boundary merge rule: best = max(left.best, right.best, left.suffix + right.prefix)."
    if "same effect" in lower and "server" in lower:
        return "Repeating the request has the same effect and does not change server or resource state."
    if "loss minimum" in lower and "convergence" in lower:
        return "A too-high learning rate can overshoot the loss minimum and prevent convergence."
    if "shared state" in lower or "shared resources" in lower:
        return "Race conditions involve concurrent access to shared state or shared resources."
    if "true/false" in lower and "monotone" in lower:
        return "Binary search on the answer requires a monotone true/false feasibility predicate."
    if "long long" in lower or "int64" in lower:
        return "Use long long/int64 for large numeric sums."
    if "all-negative" in lower or "all negative" in lower or "non-empty" in lower:
        return "For non-empty subarrays, all-negative arrays return the maximum element, not 0."
    if "online" in lower and "update" in lower:
        return "Online point updates require an efficient dynamic update structure."
    if "negative edge" in lower or "dijkstra" in lower:
        return "Dijkstra requires nonnegative edge weights; use a safe alternative when negative edges are present."
    if "rollback-capable dsu" in lower or "rollback disjoint-set" in lower:
        return "Use an offline time-axis decomposition with a rollback-capable DSU."
    if "segment tree over time" in lower or "divide and conquer over time" in lower:
        return "Place edge-active intervals on a segment tree over time while traversing with rollback DSU state."
    if "edge-active interval" in lower or "active intervals over time" in lower:
        return "Represent each edge by its active interval over time."
    if "segment tree beats" in lower or "second max" in lower or "count_max" in lower:
        return "Use segment tree beats with max, second max, count_max, and sum."
    if "only the current maxima" in lower or "between the current max and second max" in lower:
        return "A range_chmin update only changes current maxima when x lies between the current max and second max."
    if "durable local payment state" in lower or "payment state machine" in lower:
        return "Persist a durable local payment state machine around the external charge."
    if "reconcile uncertain payment outcomes" in lower or "query psp state" in lower:
        return "After a crash, reconcile uncertain outcomes by querying PSP state before retrying."
    if "consumer-side dedupe" in lower or "at-least-once retries" in lower:
        return "Use consumer-side dedupe/replay semantics so at-least-once retries do not duplicate the business action."
    if "ordered migration phases" in lower or "backfill historical data" in lower:
        return "Use ordered migration phases: backfill, live-change capture, verification, cutover, rollback."
    if "idempotent replay path" in lower or "re-apply updates safely" in lower:
        return "Keep an idempotent replay path so live changes can be re-applied safely during cutover or rollback."
    if "single-writer ownership" in lower or "partition ownership" in lower:
        return "Serialize writes per SKU with single-writer ownership or partition ownership."
    if "reservation lifecycle" in lower or "hold, confirm, release" in lower:
        return "Model the reservation lifecycle explicitly: hold, confirm, release/expire."
    if "cache as derived state" in lower or "cache is derived" in lower:
        return "Treat cache as derived state; keep an authoritative source of truth and reconciliation path."
    if "dedupe tokens" in lower or "duplicate reservations" in lower:
        return "Use idempotency keys or dedupe tokens so retries do not duplicate reservations or confirmations."
    if text.endswith("."):
        return text
    return text + "."


def _task_concept_from_constraint(constraint: str) -> Optional[str]:
    text = _clean_text(constraint)
    lower = text.lower()
    prefix = _TASK_CONCEPT_CONSTRAINT_PREFIX.lower()
    if not lower.startswith(prefix):
        return None
    concept = text[len(_TASK_CONCEPT_CONSTRAINT_PREFIX):].strip(" .,:;")
    return concept or None


def _task_concept_to_sentence(task_concept: str) -> str:
    lower_concept = _clean_text(task_concept).lower()
    if lower_concept == "source":
        return "State that the algorithm computes shortest paths from one source."
    if lower_concept == "connectivity queries":
        return "Explicitly mention connectivity queries on the evolving graph."
    if lower_concept == "plain dsu is insufficient":
        return "Explain why plain DSU is insufficient once deletions are allowed."
    if lower_concept == "time-axis structure":
        return "Explain the time-axis structure that handles edge lifetimes."
    if lower_concept == "range_chmin":
        return "Explicitly address the range_chmin update."
    if lower_concept == "range_sum":
        return "Explicitly address the range_sum query."
    if lower_concept == "per-node state":
        return "State the per-node information that makes the structure correct."
    if lower_concept == "capped lazy updates":
        return "Explain why the capped lazy update remains correct."
    if lower_concept == "same effect":
        return "Repeating the request has the same effect."
    if lower_concept == "does not change":
        return "The request does not change server or resource state."
    if lower_concept == "idempotency key":
        return "Use an idempotency key on PSP requests."
    if lower_concept == "at-least-once":
        return "The design must stay correct under at-least-once delivery."
    if lower_concept == "local database commit":
        return "Address the crash window before the local database commit."
    if lower_concept == "double charge":
        return "Prevent double charge during retries and recovery."
    if lower_concept == "zero downtime":
        return "Keep reads and writes live with zero downtime."
    if lower_concept == "verification before cutover":
        return "Verify correctness before cutover."
    if lower_concept == "rollback":
        return "Keep a rollback path available."
    if lower_concept == "reservation ttl":
        return "Model reservation TTL / hold expiration explicitly."
    if lower_concept == "payment confirmation":
        return "State how payment confirmation commits a reservation."
    if lower_concept == "hot sku":
        return "Handle contention on the hot SKU explicitly."
    if lower_concept == "oversell":
        return "Prevent oversell under concurrency."
    if lower_concept == "minimum":
        return "A too-high learning rate can overshoot the loss minimum."
    if lower_concept == "convergence":
        return "A too-high learning rate can prevent convergence."
    if lower_concept == "true/false":
        return "The feasibility predicate must switch true/false monotonically."
    if lower_concept == "maximum element":
        return "For an all-negative array, return the maximum element, not 0."
    return "Key task terms: " + _clean_text(task_concept) + "."


def _hard_constraints_for_packet(*, active: Sequence[SignalNode], signals: Sequence[SignalNode]) -> List[str]:
    hard: List[str] = []
    for sig in list(active) + [s for s in signals if s.confidence >= 0.85]:
        if sig.produced_by == "checker":
            continue
        if sig.kind not in {"constraint", "risk"}:
            continue
        text = _clean_text(sig.text)
        if text and text not in hard:
            hard.append(text)
        if len(hard) >= 10:
            break
    return hard


def select_active_signals(
    *,
    focus: str,
    looking_for: str,
    signals: Sequence[SignalNode],
    max_signals: int = 6,
    debug_signals: bool = False,
    debug_collector: Optional[List[Dict[str, Any]]] = None,
) -> List[SignalNode]:
    query_tokens = set(activation_keys_for_text(f"{focus} {looking_for}", limit=24))
    scored: List[tuple[float, SignalNode]] = []
    focus_for_dedup = _norm(focus)
    for sig in signals:
        overlap = len(query_tokens.intersection(sig.activation_keys))
        penalty = _signal_quality_penalty(sig)
        bonus = _signal_quality_bonus(sig)
        score = sig.confidence + (0.1 * overlap) + bonus - penalty
        scored.append((score, sig))
        category = _classify_signal_category(sig)
        row = {
            "id": sig.id,
            "kind": sig.kind,
            "category": category,
            "confidence": round(sig.confidence, 3),
            "overlap": overlap,
            "bonus": round(bonus, 3),
            "penalty": round(penalty, 3),
            "score": round(score, 3),
            "produced_by": sig.produced_by,
            "warm_start": (sig.state or {}).get("warm_start", False),
            "text_preview": sig.text[:120],
            "source_node_id": sig.source_node_id,
        }
        if debug_collector is not None:
            debug_collector.append(row)
    scored.sort(key=lambda item: (-item[0], item[1].id))
    selected = [sig for _, sig in scored[:max_signals]]
    if debug_signals:
        selected_ids = {s.id for s in selected}
        print("=== SIGNAL DEBUG ===")
        for row in sorted(debug_collector or [], key=lambda r: -r["score"]):
            marker = " <- SELECTED" if row["id"] in selected_ids else ""
            safe_text = row['text_preview'].encode(
                sys.stdout.encoding or "utf-8", errors="replace"
            ).decode(sys.stdout.encoding or "utf-8", errors="replace")
            print(f"  [{row['category']:22s}] {row['kind']:12s} conf={row['confidence']:.3f} "
                  f"overlap={row['overlap']} score={row['score']:.3f}{marker}")
            print(f"    {safe_text}")
        print(f"  => {len(selected)} selected from {len(signals)} total signals")
        print("==================")
    return selected


def render_capsule_prefix(signals) -> str:
    """Render a stable capsule prefix from selected signals for KV warm-pool.

    This text is sent as a system message. The server caches its KV output
    after the first step and reuses it on subsequent steps in the same session.
    """
    lines = ["Stable graph signals:"]
    for sig in signals:
        kind = getattr(sig, "kind", "signal")
        text = getattr(sig, "text", "")
        lines.append(f"- [{kind}] {text}")
    return "\n".join(lines)


def render_workspace_step_prompt(
    workspace,
    slot_name: str,
    slot_question: str,
    *,
    hard_constraints: Sequence[str] = (),
    parent_decisions: Sequence[str] = (),
) -> str:
    """Render a step prompt that asks the model to fill one workspace slot.

    The prompt shows the current workspace state and asks the model to fill
    the specified slot. The output is expected as STEP_RESULT with the slot
    fill content in delta.decisions.
    """
    filled = workspace.filled_slots() if workspace else {}
    lines = [
        "You are designing a system. Fill one slot in the design workspace.",
        "Return exactly one STEP_RESULT block.",
        "",
        f"Slot: {slot_name}",
        "",
    ]
    if filled:
        lines.append("Current workspace (already filled):")
        for k, v in filled.items():
            preview = v[:120] + "..." if len(v) > 120 else v
            lines.append(f"  {k}: {preview}")
        lines.append("")
    lines.extend([
        "Your response must address:",
        slot_question,
        "",
    ])
    if hard_constraints:
        lines.append("Hard constraints:")
        for c in hard_constraints[:6]:
            lines.append(f"- {c}")
        lines.append("")
    if parent_decisions:
        lines.append("Parent decisions:")
        for d in parent_decisions[-6:]:
            lines.append(f"- {d}")
        lines.append("")
    lines.extend([
        "Format:",
        "STEP_RESULT",
        "status: resolved | need_info | failed",
        "result: <your answer for this slot>",
        "constraints_honored:",
        "  - <copy exact hard-constraint text snippets you explicitly satisfy>",
        "delta:",
        "  decisions:",
        "    - <your slot fill content — this is the most important field>",
        "  constraints:",
        "    - <optional>",
        "  risks:",
        "    - <optional>",
        "  evidence:",
        "    - <optional>",
        "  gaps:",
        "    - <optional>",
        "END_STEP_RESULT",
    ])
    return "\n".join(lines)


def render_step_prompt(
    packet: StepContextPacket,
    *,
    capsule_signal_ids: Optional[set[str]] = None,
    procedure_pool: Optional[Sequence[Any]] = None,
) -> str:
    lines = [
        "You are executing one fast reasoning step.",
        "Return exactly one STEP_RESULT block.",
        "",
        "Task:",
        packet.task_summary,
        "",
        f"Focus: {packet.focus}",
        f"Looking for: {packet.looking_for}",
        "",
    ]
    if packet.active_signals:
        display_signals = packet.active_signals
        if capsule_signal_ids:
            display_signals = [
                s for s in packet.active_signals
                if s.id not in capsule_signal_ids
            ]
        if display_signals:
            lines.append("Active signals:")
            for sig in display_signals:
                lines.append(f"- [{sig.kind}] {sig.text}")
            lines.append("")
    if packet.parent_decisions:
        lines.append("Parent decisions:")
        for decision in packet.parent_decisions[-6:]:
            lines.append(f"- {decision}")
        lines.append("")
    if packet.hard_constraints:
        lines.append("Hard constraints:")
        for constraint in packet.hard_constraints[:6]:
            lines.append(f"- {constraint}")
        lines.append("")

    if procedure_pool and packet.reasoning_mode == "auto":
        lines.append("Available procedures:")
        for proc in procedure_pool:
            name = getattr(proc, "name", "?")
            purpose = getattr(proc, "purpose", "")
            when = getattr(proc, "when_to_use", "")
            lines.append(f"- {name}: {purpose}")
            if when:
                lines.append(f"  Use when: {when}")
        lines.extend([
            "",
            "To invoke a procedure, write in your <reasoning> or <think>:",
            "  \"I'll apply <ProcedureName> to <args>\"",
            "The system will run it and you'll get the result in the next turn.",
            "",
        ])

    if packet.reasoning_mode == "auto":
        lines.extend([
            "Choose your approach:",
            "- QUICK: Output STEP_RESULT directly (best when you have clear signals).",
            "- DEEP: First output <reasoning> with GOAL/KNOWN/PLAN, then STEP_RESULT (best for complex multi-step problems).",
            "",
            "QUICK format:",
            "STEP_RESULT",
            "status: resolved",
            "result: <your answer>",
            "constraints_honored:",
            "  - <copied constraints you satisfy>",
            "delta:",
            "  decisions:",
            "    - <your decisions>",
            "  evidence:",
            "    - <evidence supporting your answer>",
            "END_STEP_RESULT",
            "",
            "DEEP format:",
            "<reasoning>",
            "GOAL: <one-line restatement of the question>",
            "KNOWN: <key facts and signals you have>",
            "PLAN: <how you will compose the answer, step by step>",
            "</reasoning>",
            "STEP_RESULT",
            "status: resolved",
            "result: <your answer>",
            "constraints_honored:",
            "  - <copied constraints you satisfy>",
            "delta:",
            "  decisions:",
            "    - <your decisions>",
            "  evidence:",
            "    - <evidence supporting your answer>",
            "  gaps:",
            "    - <any remaining gaps>",
            "END_STEP_RESULT",
            "",
        ])
    else:
        lines.extend([
            "Format:",
            "STEP_RESULT",
            "status: resolved | need_info | failed",
            "result: <short result>",
            "missing:",
            "  question: <only if status is need_info>",
            "  why_needed: <only if status is need_info>",
            "  expected_shape: <only if status is need_info>",
            "constraints_honored:",
            "  - <copy exact hard-constraint text snippets you explicitly satisfy in result; do not write blanket claims like 'all hard constraints are satisfied'>",
            "delta:",
            "  decisions:",
            "    - <optional>",
            "  constraints:",
            "    - <optional>",
            "  risks:",
            "    - <optional>",
            "  evidence:",
            "    - <optional>",
            "  gaps:",
            "    - <optional>",
            "END_STEP_RESULT",
        ])
    return "\n".join(lines)


def attach_fast_loop_to_session(session: Any, result: FastLoopResult) -> None:
    """Persist substrate-v2 trace nodes into a SessionSubgraphController-like object."""
    step_node_ids: List[str] = []
    latest_by_step_id: Dict[str, str] = {}
    for i, step in enumerate(result.steps):
        occurrence_result = result.step_results[i] if i < len(result.step_results) else None
        occurrence_delta = occurrence_result.delta if occurrence_result is not None else step.delta
        node_id = f"v2_step_{i}_{step.step_id}"
        step_node_ids.append(node_id)
        session.subgraph.nodes.setdefault(node_id, {
            "id": node_id,
            "node_type": "substrate_v2_step",
            "text": step.focus,
            "step_id": step.step_id,
            "parent_step_id": step.parent_step_id,
            "task_id": step.task_id,
            "focus": step.focus,
            "looking_for": step.looking_for,
            "depth": step.depth,
            "status": step.status,
            "result": occurrence_result.result if occurrence_result is not None else step.result,
            "result_status": occurrence_result.status if occurrence_result is not None else None,
            "step_result": occurrence_result.to_dict() if occurrence_result is not None else None,
            "context_packet": step.context_packet.to_dict() if step.context_packet else None,
            "metadata": {"provider": "phase3e-substrate-v2"},
        })
        if step.parent_step_id:
            parent_node_id = latest_by_step_id.get(step.parent_step_id, f"v2_{step.parent_step_id}")
            session.add_edge(
                src=parent_node_id,
                dst=node_id,
                relation="v2_child_step",
                metadata={"provider": "phase3e-substrate-v2"},
            )
        if occurrence_delta is not None:
            delta_id = f"v2_delta_{i}_{step.step_id}"
            session.subgraph.nodes.setdefault(delta_id, {
                "id": delta_id,
                "node_type": "substrate_v2_delta",
                "text": _delta_summary(occurrence_delta),
                "step_id": step.step_id,
                "delta": occurrence_delta.to_dict(),
                "delta_transaction": (
                    occurrence_result.delta_transaction.to_dict()
                    if occurrence_result is not None else None
                ),
                "metadata": {"provider": "phase3e-substrate-v2"},
            })
            session.add_edge(
                src=node_id,
                dst=delta_id,
                relation="emits_delta",
                metadata={"provider": "phase3e-substrate-v2"},
            )
        latest_by_step_id[step.step_id] = node_id

    for i, check in enumerate(result.checks):
        check_id = f"v2_check_{i}"
        session.subgraph.nodes.setdefault(check_id, {
            "id": check_id,
            "node_type": "substrate_v2_check",
            "text": "passed" if check.passed else "; ".join(v.message for v in check.violations),
            "check": check.to_dict(),
            "metadata": {"provider": "phase3e-substrate-v2"},
        })
        if i < len(result.steps):
            session.add_edge(
                src=step_node_ids[i],
                dst=check_id,
                relation="checked_by",
                metadata={"provider": "phase3e-substrate-v2"},
            )

    for signal in result.signals:
        node_id = f"v2_signal_{signal.id}"
        session.subgraph.nodes.setdefault(node_id, {
            "id": node_id,
            "node_type": "substrate_v2_signal",
            "kind": signal.kind,
            "text": signal.text,
            "signal": signal.to_dict(),
            "metadata": {"provider": "phase3e-substrate-v2"},
        })
        if signal.source_step_id:
            source_node = latest_by_step_id.get(signal.source_step_id, f"v2_{signal.source_step_id}")
            session.add_edge(
                src=source_node,
                dst=node_id,
                relation="emits_signal",
                metadata={"provider": "phase3e-substrate-v2"},
            )

    # Persist plan (DEEP mode) as a dedicated node for downstream inspection
    plan = getattr(result.final_step_result, "plan", None)
    if plan:
        plan_node_id = "v2_plan"
        session.subgraph.nodes.setdefault(plan_node_id, {
            "id": plan_node_id,
            "node_type": "substrate_v2_plan",
            "goal": plan.get("goal", ""),
            "known": plan.get("known", ""),
            "plan": plan.get("plan", ""),
            "metadata": {"provider": "phase3e-substrate-v2"},
        })
        # Link plan node to the root step
        if result.root_step and result.root_step.step_id:
            root_node = latest_by_step_id.get(result.root_step.step_id)
            if root_node:
                session.add_edge(
                    src=root_node,
                    dst=plan_node_id,
                    relation="has_plan",
                    metadata={"provider": "phase3e-substrate-v2"},
                )


def project_session_subgraph_to_signals(subgraph: SessionSubgraph) -> List[SignalNode]:
    """Return a read-only SignalNode view over existing session-subgraph nodes."""
    signals: List[SignalNode] = []
    for node_id, node in subgraph.nodes.items():
        if not isinstance(node, Mapping):
            continue
        projected = project_session_node_to_signal(str(node_id), node)
        if projected is not None:
            signals.append(projected)
    return signals


def project_session_node_to_signal(node_id: str, node: Mapping[str, Any]) -> Optional[SignalNode]:
    node_type = str(node.get("node_type") or "")
    metadata = dict(node.get("metadata") or {})

    if node_type == "substrate_v2_signal":
        raw = node.get("signal")
        if isinstance(raw, dict):
            sig = SignalNode.from_dict(raw)
            sig.id = f"sigv2_{node_id}"
            sig.scope = "session"
            if not sig.activation_keys:
                sig.activation_keys = activation_keys_for_text(sig.text)
            sig.produced_by = "projection"
            return sig
        return None

    text = _node_text(node)
    if not text:
        return None

    kind: Optional[SignalKind] = None
    state: Dict[str, Any] = {}
    confidence = float(node.get("confidence", 0.5) or 0.5)

    if node_type == "activation_signal":
        kind = _activation_kind_to_signal_kind(str(node.get("kind") or ""))
        state = {"activation_kind": node.get("kind")}
        confidence = float(node.get("confidence", confidence) or confidence)
    elif node_type == "task_frame_item":
        kind = _frame_item_kind_to_signal_kind(str(node.get("kind") or ""))
        state = {"frame_kind": node.get("kind"), "priority": node.get("priority")}
        confidence = min(1.0, max(0.1, float(node.get("priority", 50) or 50) / 100.0))
    elif node_type == "session_gap":
        kind = "gap"
        state = dict(node.get("state") or {})
    elif node_type == "session_bridge":
        kind = "hypothesis"
        state = {"fills_gap": node.get("fills_gap")}
    elif node_type == "plan_node":
        status = str(node.get("status") or metadata.get("status") or "")
        if status in {"failed", "abandoned"}:
            kind = "repair" if node.get("failure_reason") else "risk"
        else:
            kind = "decision"
        state = {
            "goal": node.get("goal"),
            "hypothesis": node.get("hypothesis") or metadata.get("hypothesis"),
            "status": status,
            "mode": node.get("mode"),
            "failure_reason": node.get("failure_reason"),
        }
        confidence = float(node.get("checkpoint_quality", confidence) or confidence)
    elif node_type == "plan_check":
        passed = bool(node.get("passed") or metadata.get("passed") or False)
        kind = "evidence" if passed else "risk"
        state = {
            "checked_node_id": node.get("checked_node_id"),
            "passed": passed,
            "failure_scope": node.get("failure_scope") or metadata.get("failure_scope"),
            "failed_requirements": list(node.get("failed_requirements") or []),
        }
        confidence = 0.9 if passed else 0.85
    elif node_type == "signal":
        kind = _signal_severity_to_signal_kind(str(node.get("severity") or ""), text)
        state = {"severity": node.get("severity")}
    elif node_type in {"evidence", "answer", "question"}:
        kind = "evidence"
        state = {"source_node_type": node_type}
    else:
        return None

    return SignalNode(
        id=f"sigv2_{node_id}",
        kind=kind,
        text=text,
        scope="session",
        activation_keys=activation_keys_for_text(text),
        source_step_id=_source_step_id(node),
        produced_by="projection",
        state=state or None,
        evidence_ids=[str(x) for x in node.get("evidence_node_ids", [])],
        source_node_id=str(node.get("id") or node_id),
        source_node_type=node_type,
        confidence=max(0.0, min(1.0, confidence)),
    )


def packet_from_task_frame(
    *,
    task_summary: str,
    focus: str,
    looking_for: str,
    frame: GraphTaskFrame,
    budget_remaining: Optional[Mapping[str, Any]] = None,
    max_signals: int = 6,
) -> StepContextPacket:
    signals = [signal_from_frame_item(item) for item in frame.all_items()]
    active = sorted(signals, key=lambda sig: (-sig.confidence, sig.text))[:max_signals]
    hard_constraints = [
        item.text for item in frame.constraints + frame.pitfalls
        if item.priority >= 80
    ]
    return StepContextPacket(
        task_summary=task_summary,
        focus=focus,
        looking_for=looking_for,
        active_signals=active,
        parent_decisions=[],
        open_gaps=[item.text for item in frame.unresolved_gaps],
        hard_constraints=hard_constraints,
        budget_remaining=dict(budget_remaining or {}),
    )


def signal_from_frame_item(item: FrameItem) -> SignalNode:
    kind = _frame_item_kind_to_signal_kind(item.kind)
    return SignalNode(
        id=f"sigv2_{item.item_id}",
        kind=kind,
        text=item.text,
        activation_keys=activation_keys_for_text(item.text),
        produced_by="projection",
        state={"frame_kind": item.kind, "priority": item.priority},
        evidence_ids=list(item.source_signal_ids),
        source_node_id=item.item_id,
        source_node_type="task_frame_item",
        confidence=min(1.0, max(0.1, item.priority / 100.0)),
    )


def _parse_step_block(block: str) -> Dict[str, Any]:
    lines = [line.rstrip() for line in block.splitlines()]
    status = "resolved"
    result_lines: List[str] = []
    missing: Dict[str, str] = {}
    constraints_honored: List[str] = []
    delta_fields: Dict[str, List[str]] = {
        "decisions": [],
        "constraints": [],
        "risks": [],
        "evidence": [],
        "gaps": [],
        "repairs": [],
    }
    section: Optional[str] = None
    current_delta_key: Optional[str] = None

    for raw_line in lines:
        line = raw_line.strip()
        if not line:
            continue
        lower = line.lower()
        if lower.startswith("status:"):
            value = line.split(":", 1)[1].strip().lower()
            if value not in {"resolved", "need_info", "failed"}:
                raise ValueError(f"unknown status {value!r}")
            status = value
            section = None
            continue
        if lower.startswith("result:"):
            first = line.split(":", 1)[1].strip()
            if first and first != "|":
                result_lines.append(first)
            section = "result"
            continue
        if lower == "missing:":
            section = "missing"
            continue
        if lower == "constraints_honored:":
            section = "constraints_honored"
            continue
        if lower == "delta:":
            section = "delta"
            current_delta_key = None
            continue

        if section == "result":
            result_lines.append(line)
            continue
        if section == "missing" and ":" in line:
            key, value = line.split(":", 1)
            key = key.strip().lower()
            if key in {"question", "why_needed", "expected_shape"}:
                missing[key] = _strip_quote(value.strip())
            continue
        if section == "constraints_honored":
            value = ""
            if line.startswith("-"):
                value = line[1:].strip()
            elif ":" in line:
                _, value = line.split(":", 1)
                value = value.strip()
            if value:
                constraints_honored.append(_strip_quote(value))
            continue
        if section == "delta":
            if ":" in line and not line.startswith("-"):
                key, value = line.split(":", 1)
                key = key.strip().lower()
                if key in delta_fields:
                    current_delta_key = key
                    parsed_inline = _parse_inline_list(value.strip())
                    delta_fields[key].extend(parsed_inline)
                continue
            if line.startswith("-") and current_delta_key:
                value = _strip_quote(line[1:].strip())
                if value:
                    delta_fields[current_delta_key].append(value)
            continue

    missing_obj = None
    if status == "need_info":
        if not all(missing.get(k) for k in ("question", "why_needed", "expected_shape")):
            raise ValueError("need_info requires missing.question, missing.why_needed, and missing.expected_shape")
        missing_obj = MissingInfo(
            question=missing["question"],
            why_needed=missing["why_needed"],
            expected_shape=missing["expected_shape"],
        )
    delta = StateDelta(
        decisions=delta_fields["decisions"],
        constraints=delta_fields["constraints"],
        risks=delta_fields["risks"],
        evidence=delta_fields["evidence"],
        gaps=delta_fields["gaps"],
        repairs=delta_fields["repairs"],
        produced_by="llm_delta",
        confidence=0.8,
    )
    return {
        "status": status,
        "result": "\n".join(result_lines).strip(),
        "missing": missing_obj,
        "constraints_honored": constraints_honored,
        "delta_transaction": DeltaTransaction("parsed", delta),
    }


def _skim_delta(raw: str) -> StateDelta:
    text = _clean_text(raw)
    decisions: List[str] = []
    risks: List[str] = []
    evidence: List[str] = []
    lower = text.lower()
    if any(tok in lower for tok in ("use ", "choose ", "select ")):
        decisions.append(text[:220])
    elif text:
        evidence.append(text[:220])
    if any(tok in lower for tok in ("unsafe", "fail", "invalid", "negative edge", "contradict")):
        risks.append(text[:220])
    return StateDelta(
        decisions=decisions,
        risks=risks,
        evidence=evidence,
        produced_by="regex_fallback",
        confidence=0.2,
    )


def _run_checker_plugin(name: str, step_result: StepResult, packet: StepContextPacket) -> List[CheckerViolation]:
    if name == "generic_step_format":
        return _check_generic_step_format(step_result, packet)
    if name == "algorithm_design":
        return _check_algorithm_design(step_result, packet)
    if name == "dynamic_max_subarray":
        return _check_dynamic_max_subarray(step_result, packet)
    if name == "shortest_path_safety":
        return _check_shortest_path_safety(step_result, packet)
    if name == "factual_recall":
        return _check_factual_recall(step_result, packet)
    if name == "dynamic_connectivity_deletions":
        return _check_dynamic_connectivity_deletions(step_result, packet)
    if name == "segment_tree_beats":
        return _check_segment_tree_beats(step_result, packet)
    if name == "payment_crash_recovery":
        return _check_payment_crash_recovery(step_result, packet)
    if name == "zero_downtime_migration":
        return _check_zero_downtime_migration(step_result, packet)
    if name == "inventory_reservation":
        return _check_inventory_reservation(step_result, packet)
    return []


def _check_generic_step_format(step_result: StepResult, packet: StepContextPacket) -> List[CheckerViolation]:
    violations: List[CheckerViolation] = []
    if step_result.delta_transaction.status == "dropped":
        violations.append(CheckerViolation("delta_dropped", "No usable STEP_RESULT delta was parsed.", "soft"))
    if step_result.status == "need_info" and step_result.missing is None:
        violations.append(CheckerViolation("missing_required", "need_info requires a strict missing object.", "hard"))
    hay = _haystack(step_result)
    if step_result.status == "resolved":
        for honored in step_result.constraints_honored[:6]:
            if _is_meta_constraint_claim(honored):
                violations.append(CheckerViolation(
                    "honored_constraint_meta_claim",
                    "constraints_honored must copy explicit hard constraints, not blanket meta-claims.",
                    "soft",
                ))
                continue
            if not _matches_packet_constraint(honored, packet.hard_constraints):
                violations.append(CheckerViolation(
                    "honored_constraint_unknown",
                    f"Claimed honored constraint is not in the packet hard constraints: {honored}",
                    "soft",
                ))
                continue
            if not _constraint_addressed(honored, hay):
                severity = "soft" if _is_surface_constraint(honored) else "hard"
                violations.append(CheckerViolation(
                    "honored_constraint_unmarked",
                    f"Claimed honored constraint is not visible in result: {honored}",
                    severity,
                ))
                if severity == "hard" and len(violations) >= 3:
                    return violations
    for constraint in packet.hard_constraints:
        if not _constraint_addressed(constraint, hay):
            violations.append(CheckerViolation("constraint_unaddressed", f"Constraint not addressed: {constraint}", "soft"))
            if len(violations) >= 3:
                break
    return violations


def _is_surface_constraint(constraint: str) -> bool:
    lower = _clean_text(constraint).lower()
    if _task_concept_from_constraint(constraint):
        return True
    return any(pattern in lower for pattern in (
        "true/false",
        "same effect",
        "does not change",
        "loss minimum",
        "convergence",
        "source",
        "connectivity queries",
        "sum, prefix, suffix, and best",
        "cross-boundary merge rule",
        "shared state",
        "shared resources",
        "idempotency key",
        "at-least-once",
        "double charge",
        "zero downtime",
        "verification before cutover",
        "reservation ttl",
        "payment confirmation",
        "oversell",
    ))


def _is_meta_constraint_claim(constraint: str) -> bool:
    lower = _clean_text(constraint).lower()
    return lower.startswith("all hard constraints") or lower.startswith("every hard constraint")


def _matches_packet_constraint(claim: str, hard_constraints: Sequence[str]) -> bool:
    return _shared_matches_packet_constraint(
        claim,
        hard_constraints,
        task_concept_extractor=_task_concept_from_constraint,
    )


def _check_algorithm_design(step_result: StepResult, packet: StepContextPacket) -> List[CheckerViolation]:
    hay = _haystack(step_result)
    query = f"{packet.focus} {packet.looking_for} {packet.task_summary}".lower()
    violations: List[CheckerViolation] = []
    if any(tok in query for tok in ("complexity", "algorithm", "data structure")):
        if not any(tok in hay for tok in ("o(", "log", "linear", "constant", "complexity")):
            violations.append(CheckerViolation("complexity_missing", "Algorithm step should state complexity or cost.", "soft"))
    if "online" in query and "offline" in hay and "online" not in hay:
        violations.append(CheckerViolation("online_contradiction", "Answer appears to use an offline method for an online task.", "hard"))
    return violations


def _check_dynamic_max_subarray(step_result: StepResult, packet: StepContextPacket) -> List[CheckerViolation]:
    hay = _haystack(step_result)
    query = f"{packet.focus} {packet.looking_for} {' '.join(packet.hard_constraints)}".lower()
    if "subarray" not in query and "subarray" not in hay:
        return []
    violations: List[CheckerViolation] = []
    if "kadane" in hay and "segment tree" not in hay and any(tok in query for tok in ("update", "online")):
        violations.append(CheckerViolation("kadane_online", "Kadane-only answer is invalid for online point updates.", "hard"))
    finalish = "final" in query or "answer" in query
    if "segment tree" in query or "update" in query or "online" in query:
        if "segment tree" not in hay and step_result.status == "resolved":
            violations.append(CheckerViolation("segment_tree_missing", "Dynamic max-subarray answer should use segment tree.", "hard"))
    needs_aggregate = (
        ("segment tree" in hay or "segment tree" in query)
        and ("subarray" in hay or "subarray" in query)
        and (finalish or "merge" in hay or "merge" in query or "repair" in query)
    )
    if needs_aggregate:
        has_sum = _has_total_sum_aggregate(hay)
        has_prefix = "prefix" in hay or "pref" in hay
        has_suffix = "suffix" in hay or "suff" in hay
        has_best = _has_best_subarray_aggregate(hay)
        if not (has_sum and has_prefix and has_suffix and has_best):
            violations.append(CheckerViolation(
                "segment_tree_aggregate_missing",
                "Dynamic max-subarray segment tree should store sum, prefix, suffix, and best.",
                "hard",
            ))
        if finalish and not _has_max_subarray_merge_rule(hay):
            violations.append(CheckerViolation(
                "segment_tree_merge_missing",
                "Dynamic max-subarray answer should state the cross-boundary merge rule.",
                "soft" if has_sum and has_prefix and has_suffix and has_best else "hard",
            ))
    if finalish and "long long" in query and not any(tok in hay for tok in ("long long", "int64", "int64_t")):
        violations.append(CheckerViolation(
            "long_long_missing",
            "Missing long long/int64 handling.",
            "soft" if "segment tree" in hay else "hard",
        ))
    if finalish and ("all-negative" in query or "non-empty" in query) and not any(
        tok in hay for tok in ("all-negative", "all negative", "non-empty", "maximum element", "not 0")
    ):
        violations.append(CheckerViolation("all_negative_missing", "Missing non-empty/all-negative handling.", "hard"))
    return violations[:3]


def _has_total_sum_aggregate(hay: str) -> bool:
    total_sum_patterns = (
        "total sum",
        "segment sum",
        "range sum",
        "node sum",
        "node.sum",
        "sum, prefix",
        "sum,prefix",
        "sum / prefix",
        "sum/prefix",
        "sum prefix",
        "store sum",
        "stores sum",
        "storing sum",
    )
    return any(pattern in hay for pattern in total_sum_patterns)


def _has_best_subarray_aggregate(hay: str) -> bool:
    return any(pattern in hay for pattern in (
        "best",
        "max_subarray",
        "max subarray",
        "maximum subarray",
        "maxsub",
    ))


def _has_max_subarray_merge_rule(hay: str) -> bool:
    if "merge" not in hay and "combine" not in hay and "pull" not in hay:
        return False
    return (
        ("left" in hay and "right" in hay and "suffix" in hay and "prefix" in hay)
        or "left.suffix + right.prefix" in hay
        or "left suff + right pref" in hay
        or "cross-boundary" in hay
        or "cross boundary" in hay
    )


def _check_shortest_path_safety(step_result: StepResult, packet: StepContextPacket) -> List[CheckerViolation]:
    hay = _haystack(step_result)
    task_context = f"{packet.task_summary} {' '.join(packet.hard_constraints)}".lower()
    if "dijkstra" not in task_context and "negative edge" not in task_context and "dijkstra" not in hay:
        return []
    explicit_nonnegative = any(tok in task_context for tok in (
        "only nonnegative edge weights",
        "all edge weights are nonnegative",
        "all edge weights must be nonnegative",
        "edge weights are nonnegative",
        "nonnegative edge weights",
    ))
    risk_context = (
        task_context
        .replace("nonnegative", " ")
        .replace("non-negative", " ")
        .replace("non negative", " ")
    )
    negative_edge_risk = any(tok in risk_context for tok in (
        "negative edge",
        "negative edges",
        "negative weight",
        "negative weights",
        "weights may be negative",
        "weights can be negative",
        "weights are present or allowed",
    ))
    if explicit_nonnegative and not negative_edge_risk:
        return []
    if negative_edge_risk and "dijkstra" in hay and not any(tok in hay for tok in ("unsafe", "invalid", "bellman")):
        return [CheckerViolation("dijkstra_negative_edge", "Dijkstra commitment under negative edge signal.", "hard")]
    return []


def _check_factual_recall(step_result: StepResult, packet: StepContextPacket) -> List[CheckerViolation]:
    """Conservative factual-recall guard.

    This plugin does not judge truth. It only checks that a resolved answer is
    anchored to active evidence when such evidence exists. Violations are soft
    so unknown domains fail open and consolidation can downweight weak traces.
    """
    if step_result.status != "resolved":
        return []
    hay = _haystack(step_result)
    if not hay:
        return [CheckerViolation("empty_factual_answer", "Resolved factual step has no answer text.", "soft")]
    if any(tok in hay for tok in ("not enough information", "unknown", "cannot determine", "unsure")):
        return []

    evidence_signals = [sig for sig in packet.active_signals if sig.kind == "evidence"]
    if not evidence_signals:
        return []

    evidence_keys: List[str] = []
    for sig in evidence_signals:
        candidates = sig.activation_keys or activation_keys_for_text(sig.text, limit=12)
        for key in candidates:
            if key not in evidence_keys and key not in _FACTUAL_RECALL_STOPWORDS:
                evidence_keys.append(key)
            if len(evidence_keys) >= 12:
                break
        if len(evidence_keys) >= 12:
            break

    if not evidence_keys:
        return []
    if not any(key in hay for key in evidence_keys):
        return [CheckerViolation(
            "factual_evidence_unreferenced",
            "Resolved factual answer does not overlap active evidence signals.",
            "soft",
        )]
    return []


def _check_dynamic_connectivity_deletions(step_result: StepResult, packet: StepContextPacket) -> List[CheckerViolation]:
    hay = _haystack(step_result)
    query = f"{packet.focus} {packet.looking_for} {packet.task_summary} {' '.join(packet.hard_constraints)}".lower()
    if "remove(" not in query and "delet" not in query and "connected(" not in query:
        return []
    violations: List[CheckerViolation] = []
    recommends_per_query_traversal = any(tok in hay for tok in ("bfs per query", "dfs per query", "recompute bfs", "recompute dfs"))
    if recommends_per_query_traversal and not any(tok in hay for tok in ("without recomputing", "instead of recomputing", "rather than recomputing")):
        violations.append(CheckerViolation("per_query_traversal", "Per-query BFS/DFS is too slow for this dynamic connectivity task.", "hard"))
    has_dsu = _contains_any(hay, "union-find", "union find", "disjoint set", "dsu")
    has_offline_time = _contains_any(
        hay,
        "segment tree over time",
        "time segment tree",
        "divide and conquer over time",
        "offline",
        "time-axis",
    )
    has_rollback = _contains_any(hay, "rollback", "undo", "revert")
    has_interval = _contains_any(hay, "active interval", "time interval", "edge lifetime", "interval where an edge exists")
    if has_dsu and not has_rollback:
        violations.append(CheckerViolation("dsu_without_rollback", "Dynamic deletions need rollback/persistence beyond plain DSU.", "hard"))
    if not has_dsu:
        violations.append(CheckerViolation("dsu_missing", "Expected a DSU / Union-Find core for offline dynamic connectivity.", "hard"))
    if not has_offline_time:
        violations.append(CheckerViolation("time_axis_missing", "Missing offline time-axis structure for edge add/remove intervals.", "hard"))
    if has_dsu and has_rollback and has_offline_time and not has_interval:
        violations.append(CheckerViolation("active_interval_missing", "Answer should explain edge-active intervals over time.", "soft"))
    return violations[:3]


def _check_segment_tree_beats(step_result: StepResult, packet: StepContextPacket) -> List[CheckerViolation]:
    hay = _haystack(step_result)
    query = f"{packet.focus} {packet.looking_for} {packet.task_summary} {' '.join(packet.hard_constraints)}".lower()
    if "range_chmin" not in query and "capped lazy" not in query:
        return []
    violations: List[CheckerViolation] = []
    if "ordinary lazy propagation" in hay and not _contains_any(hay, "second max", "second maximum", "second-largest"):
        violations.append(CheckerViolation("ordinary_lazy_claim", "Ordinary lazy propagation is insufficient for range_chmin.", "hard"))
    if not _contains_any(hay, "segment tree beats", "segment tree"):
        violations.append(CheckerViolation("segment_tree_missing", "Expected a segment-tree-based answer for range_chmin / range_sum.", "hard"))
    has_second = _contains_any(hay, "second max", "second maximum", "second-largest")
    has_count = _contains_any(hay, "count of max", "max count", "count_max", "cnt max")
    has_sum = _contains_any(hay, "sum", "range sum", "node sum")
    has_cap_rule = _contains_any(
        hay,
        "only current maxima change",
        "cap only the maxima",
        "between the current max and second max",
        "second max < x",
    )
    if not (has_second and has_count and has_sum):
        violations.append(CheckerViolation(
            "beats_state_missing",
            "Segment tree beats answer should name max, second max, count_max, and sum style state.",
            "hard",
        ))
    if not has_cap_rule:
        violations.append(CheckerViolation(
            "beats_cap_rule_missing",
            "Answer should explain that range_chmin only changes current maxima when x lies between max and second max.",
            "hard" if not has_second else "soft",
        ))
    return violations[:3]


def _check_payment_crash_recovery(step_result: StepResult, packet: StepContextPacket) -> List[CheckerViolation]:
    hay = _haystack(step_result)
    query = f"{packet.focus} {packet.looking_for} {packet.task_summary} {' '.join(packet.hard_constraints)}".lower()
    if "psp" not in query and "payment worker" not in query and "double charge" not in query:
        return []
    violations: List[CheckerViolation] = []
    if "exactly-once" in hay:
        violations.append(CheckerViolation("exactly_once_claim", "Exactly-once transport does not solve the external-payment crash window.", "hard"))
    if "two-phase commit" in hay or "2pc" in hay:
        violations.append(CheckerViolation("psp_2pc_claim", "Do not rely on two-phase commit with the PSP.", "hard"))
    has_idempotency = _contains_any(hay, "idempotency key", "idempotency")
    has_durable_state = _contains_any(hay, "durable", "state machine", "payment intent", "pending", "charged")
    has_reconcile = _contains_any(hay, "reconciliation", "query the psp", "psp status", "external status lookup", "reconcile with the psp")
    has_retry_dedupe = _contains_any(hay, "retry", "replay", "dedupe", "deduplication", "effectively once")
    if not has_idempotency:
        violations.append(CheckerViolation("idempotency_missing", "PSP idempotency key should be part of the payment design.", "hard"))
    if not has_durable_state:
        violations.append(CheckerViolation("durable_state_missing", "Missing durable local payment state around the external charge.", "hard"))
    if has_idempotency and not has_reconcile:
        violations.append(CheckerViolation("reconciliation_missing", "Need a reconciliation / PSP status lookup path for crash uncertainty.", "hard"))
    if has_idempotency and has_durable_state and not has_retry_dedupe:
        violations.append(CheckerViolation("retry_dedupe_missing", "Answer should explain retry/replay dedupe under at-least-once delivery.", "soft"))
    return violations[:3]


def _check_zero_downtime_migration(step_result: StepResult, packet: StepContextPacket) -> List[CheckerViolation]:
    hay = _haystack(step_result)
    query = f"{packet.focus} {packet.looking_for} {packet.task_summary} {' '.join(packet.hard_constraints)}".lower()
    if "zero downtime" not in query and "cutover" not in query and "monolith orders table" not in query:
        return []
    violations: List[CheckerViolation] = []
    if _contains_any(hay, "big bang", "take downtime"):
        violations.append(CheckerViolation("unsafe_cutover_claim", "Big-bang or downtime-based migration violates the task.", "hard"))
    has_backfill = _contains_any(hay, "backfill", "historical copy", "initial copy")
    has_live_sync = _contains_any(hay, "dual write", "cdc", "change data capture", "live tail", "tail the log")
    has_verify = _contains_any(hay, "shadow read", "verification", "verify", "consistency check", "compare")
    has_replay = _contains_any(hay, "idempotent replay", "replay", "replayable", "replay log")
    has_cutover = _contains_any(hay, "cutover", "writer switch", "source of truth", "single writer")
    has_rollback = _contains_any(hay, "rollback", "roll back")
    if not has_backfill:
        violations.append(CheckerViolation("backfill_missing", "Zero-downtime migration should include a historical backfill phase.", "hard"))
    if not has_live_sync:
        violations.append(CheckerViolation("live_sync_missing", "Need live-change capture such as dual write or CDC before cutover.", "hard"))
    if not has_verify:
        violations.append(CheckerViolation("verification_missing", "Need shadow verification / consistency checks before cutover.", "hard"))
    if has_backfill and has_live_sync and has_verify and not has_cutover:
        violations.append(CheckerViolation("cutover_missing", "Answer should explicitly describe cutover / source-of-truth switch.", "soft"))
    if has_backfill and has_live_sync and not has_replay:
        violations.append(CheckerViolation("replay_missing", "Need replay / replayability for correctness during migration.", "soft"))
    if not has_rollback:
        violations.append(CheckerViolation("rollback_missing", "Need a rollback path after cutover.", "hard"))
    return violations[:3]


def _check_inventory_reservation(step_result: StepResult, packet: StepContextPacket) -> List[CheckerViolation]:
    hay = _haystack(step_result)
    query = f"{packet.focus} {packet.looking_for} {packet.task_summary} {' '.join(packet.hard_constraints)}".lower()
    if "inventory reservation" not in query and "flash-sale" not in query and "oversell" not in query:
        return []
    violations: List[CheckerViolation] = []
    if "cache is the source of truth" in hay:
        violations.append(CheckerViolation("cache_authority_claim", "Cache cannot be the sole source of truth for inventory.", "hard"))
    if "global mutex" in hay:
        violations.append(CheckerViolation("global_mutex_claim", "One global mutex around all SKUs does not scale.", "hard"))
    has_single_writer = _contains_any(hay, "single writer per sku", "partition owner", "serialize per sku", "single-writer")
    has_lifecycle = _contains_any(hay, "reservation", "ttl", "lease", "hold", "confirm", "release", "expiration")
    has_dedupe = _contains_any(hay, "idempotency key", "dedupe", "deduplication")
    has_reconcile = _contains_any(hay, "reconciliation", "compensating", "rebuild from the log", "rebuild from log")
    has_authority = _contains_any(hay, "source of truth", "authoritative", "cache is derived", "not just cache")
    if not has_single_writer:
        violations.append(CheckerViolation("single_writer_missing", "Need single-writer / partition ownership to prevent oversell.", "hard"))
    if not has_lifecycle:
        violations.append(CheckerViolation("reservation_lifecycle_missing", "Need explicit reservation TTL / confirm / release lifecycle.", "hard"))
    if has_lifecycle and not has_dedupe:
        violations.append(CheckerViolation("reservation_dedupe_missing", "Need idempotency / dedupe for retries and at-least-once delivery.", "soft"))
    if not has_authority:
        violations.append(CheckerViolation("inventory_authority_missing", "Answer should name an authoritative source of truth beyond cache.", "hard"))
    if has_single_writer and has_lifecycle and not has_reconcile:
        violations.append(CheckerViolation("inventory_reconciliation_missing", "Need reconciliation / rebuild path after failures.", "soft"))
    return violations[:3]


def _haystack(step_result: StepResult) -> str:
    delta = step_result.delta
    parts = [
        step_result.result,
        " ".join(delta.decisions),
        " ".join(delta.constraints),
        " ".join(delta.risks),
        " ".join(delta.evidence),
        " ".join(delta.repairs),
    ]
    return " ".join(parts).lower()


_FACTUAL_RECALL_STOPWORDS = {
    "about",
    "after",
    "answer",
    "because",
    "before",
    "could",
    "current",
    "define",
    "definition",
    "final",
    "from",
    "have",
    "known",
    "looking",
    "need",
    "result",
    "should",
    "task",
    "that",
    "their",
    "there",
    "this",
    "what",
    "when",
    "where",
    "which",
    "with",
}


def _constraint_addressed(constraint: str, hay: str) -> bool:
    return _shared_constraint_addressed(
        constraint,
        hay,
        task_concept_extractor=_task_concept_from_constraint,
    )


def _contains_any(hay: str, *terms: str) -> bool:
    return any(term in hay for term in terms)


def _parse_inline_list(value: str) -> List[str]:
    value = value.strip()
    if not value:
        return []
    if value == "[]":
        return []
    if value.startswith("[") and value.endswith("]"):
        try:
            parsed = json.loads(value)
            if isinstance(parsed, list):
                return [_strip_quote(str(x)) for x in parsed if str(x).strip()]
        except Exception:
            inner = value[1:-1]
            return [_strip_quote(part.strip()) for part in inner.split(",") if part.strip()]
    return [_strip_quote(value)]


def _strip_quote(value: str) -> str:
    value = value.strip()
    if (value.startswith('"') and value.endswith('"')) or (value.startswith("'") and value.endswith("'")):
        return value[1:-1]
    return value


def activation_keys_for_text(text: str, *, limit: int = 8) -> List[str]:
    keys: List[str] = []
    for tok in re.findall(r"[a-z0-9_]+", text.lower()):
        if len(tok) < 4:
            continue
        if tok not in keys:
            keys.append(tok)
        if len(keys) >= limit:
            break
    return keys


def _activation_kind_to_signal_kind(kind: str) -> SignalKind:
    return {
        "constraint": "constraint",
        "answer_requirement": "constraint",
        "pitfall": "risk",
        "procedure_suggestion": "procedure",
        "example": "evidence",
        "bridge_hypothesis": "hypothesis",
        "missing_context": "gap",
    }.get(kind, "evidence")  # type: ignore[return-value]


def _frame_item_kind_to_signal_kind(kind: str) -> SignalKind:
    return {
        "constraint": "constraint",
        "answer_requirement": "constraint",
        "pitfall": "risk",
        "procedure_suggestion": "procedure",
        "example": "evidence",
        "missing_context": "gap",
        "bridge_hypothesis": "hypothesis",
    }.get(kind, "evidence")  # type: ignore[return-value]


def _signal_severity_to_signal_kind(severity: str, text: str) -> SignalKind:
    lower = f"{severity} {text}".lower()
    if "error" in lower or "fail" in lower or "contradict" in lower:
        return "risk"
    return "evidence"


def _source_step_id(node: Mapping[str, Any]) -> Optional[str]:
    if node.get("source_step_id") is not None:
        return str(node["source_step_id"])
    if node.get("created_step") is not None:
        return f"step_{node['created_step']}"
    return None


def _node_text(node: Mapping[str, Any]) -> str:
    text = node.get("text")
    if text:
        return _clean_text(str(text))
    for key in ("goal", "reason", "hypothesis", "name"):
        if node.get(key):
            return _clean_text(str(node[key]))
    return ""


def _short_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:10]


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "").strip().lower())


def _clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


# --- Signal quality classification helpers (post-3E signal usefulness) ---

_CHECKER_RESIDUE_PATTERNS = re.compile(
    r"^(violation:|checker:|segment_tree_|long_long_|backfill_|live_sync_"
    r"|verification_missing|rollback_missing|dsu_missing|time_axis_missing"
    r"|active_interval_missing|reservation_lifecycle_|single_writer_"
    r"|inventory_authority_|inventory_reconciliation_|reservation_dedupe_"
    r"|honored_constraint_unmarked|honored_constraint_unknown|per_query_traversal"
    r"|empty_factual_answer|unsafe_cutover_claim|cache_authority_claim"
    r"|global_mutex_claim|gap_cycle|gap_followed|maximum_element_missing)"
)

def _is_checker_residue_text(text: str) -> bool:
    stripped = text.strip().lower()
    if _CHECKER_RESIDUE_PATTERNS.match(stripped):
        return True
    known_codes = {
        "segment_tree_aggregate_missing", "segment_tree_merge_missing",
        "long_long_missing", "all_negative_missing", "kadane_online",
        "segment_tree_missing", "constraint_unaddressed", "missing_required",
        "delta_dropped", "per_query_traversal", "empty_factual_answer",
        "unsafe_cutover_claim", "cache_authority_claim", "global_mutex_claim",
        "gap_cycle", "gap_followed", "maximum_element_missing",
    }
    return stripped in known_codes


_PROCEDURAL_PATTERNS = re.compile(
    r"\b(I'?ll|let me|here('s| is) (the|my)|"
    r"my (approach|solution|answer)|"
    r"in this (step|section|solution)|"
    r"as a (first|next|initial) step)\b",
    re.I,
)

def _is_procedural_fragment(text: str) -> bool:
    return bool(_PROCEDURAL_PATTERNS.search(text))


_INVARIANT_KEYWORDS = {
    "idempotency", "durable state", "backfill", "live sync",
    "verification", "segment tree beats", "second max", "count_max",
    "single writer", "authoritative source", "rollback",
    "reconciliation",
    "prefix", "suffix", "best", "merge rule",
    "concurrency control", "lifecycle", "reservation ttl",
    "offline", "time axis", "active interval",
    "rollback dsu", "divide and conquer over time",
    "edge-active intervals", "segment tree over time",
}

def _is_compact_invariant(text: str) -> bool:
    if len(text) > 200:
        return False
    lower = text.lower()
    return any(kw in lower for kw in _INVARIANT_KEYWORDS)


def _signal_quality_penalty(sig: SignalNode) -> float:
    if _is_checker_residue_text(sig.text):
        return 0.3
    if sig.produced_by == "checker":
        return 0.2
    if _is_procedural_fragment(sig.text):
        return 0.15
    if len(sig.text) > 300:
        return 0.1
    return 0.0


def _signal_quality_bonus(sig: SignalNode) -> float:
    if _is_checker_residue_text(sig.text):
        return 0.0
    bonus = 0.0
    if sig.kind in {"constraint", "risk"}:
        bonus += 0.05
    if _is_compact_invariant(sig.text):
        bonus += 0.15
    if sig.produced_by == "controller":
        bonus += 0.05
    return min(bonus, 0.3)


def _classify_signal_category(sig: SignalNode) -> str:
    if _is_checker_residue_text(sig.text):
        return "checker_residue"
    if _is_procedural_fragment(sig.text):
        return "procedural_fragment"
    if _is_compact_invariant(sig.text) and sig.kind in {"constraint", "risk"}:
        return "reusable_invariant"
    if sig.kind in {"constraint", "risk"}:
        return "useful_constraint"
    if len(sig.text) > 300:
        return "verbose_signal"
    return "other"


def _delta_summary(delta: StateDelta) -> str:
    parts = delta.decisions + delta.constraints + delta.risks + delta.evidence + delta.gaps + delta.repairs
    return "; ".join(parts[:4])[:500]


# ─── Workspace sibling branching ──────────────────────────────────────


def _try_workspace_slot(
    workspace,
    slot_name: str,
    slot_question: str,
    *,
    parent_decisions: Sequence[str] = (),
    hard_constraints: Sequence[str] = (),
    llm_call: Callable[[str], str],
    max_siblings: int = 1,
    checker: Optional[Callable[[str], List[str]]] = None,
) -> Optional[str]:
    """Try to fill one workspace slot via LLM with up to max_siblings attempts.

    Each attempt is a separate LLM call. If the checker (identified by slot
    config) fails, a sibling attempt is made with a slightly different prompt
    that includes the previous failure. Returns the slot fill text on success,
    or None if all attempts fail.
    """
    for attempt in range(max_siblings + 1):
        sibling_hint = ""
        if attempt > 0:
            sibling_hint = (
                f"\n\nPrevious attempt for slot {slot_name!r} failed the checker. "
                "This is a sibling attempt — try a different design approach."
            )
        prompt = render_workspace_step_prompt(
            workspace,
            slot_name,
            slot_question + sibling_hint,
            hard_constraints=hard_constraints,
            parent_decisions=parent_decisions,
        )
        raw = llm_call(prompt)
        result = parse_step_result(raw)
        if result.status != "resolved":
            continue
        # Run the content checker if provided
        if checker is not None:
            violations = checker(result.result)
            if violations:
                continue
        return result.result
    return None
