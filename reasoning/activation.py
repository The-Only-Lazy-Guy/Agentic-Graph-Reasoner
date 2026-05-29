"""Context-aware graph activation - Phase 3C.

The activation layer is deliberately conservative:
- no LLM calls
- no arbitrary code stored on graph nodes
- no long-term graph mutation

It turns the current question + retrieved anchors into a compact task frame
that can shape a direct answer even when no procedure fires.
"""
from __future__ import annotations

import json
import re
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Literal, Mapping, Optional, Sequence

from graph_core import MemoryGraph, Node, canonical_node_type
from reasoning.lexical_matching import (
    constraint_addressed as _shared_constraint_addressed,
    content_tokens,
    has_token_overlap,
)
from reasoning.schemas import ProcedureNode, SessionEdge
from reasoning.session_subgraph import SessionSubgraphController


DEFAULT_ACTIVATION_TRACE_ROOT = Path("data/activation_traces")

SignalKind = Literal[
    "constraint",
    "pitfall",
    "procedure_suggestion",
    "example",
    "bridge_hypothesis",
    "missing_context",
    "answer_requirement",
]


@dataclass
class ActivationConfig:
    max_hops: int = 2
    max_active_nodes: int = 24
    max_signals: int = 40
    max_provisional_nodes: int = 6
    max_task_frame_items: int = 12
    max_frame_chars: int = 2500


@dataclass(frozen=True)
class ActivationHeuristicConfig:
    """Named constants for Phase 3C lexical activation heuristics.

    These values intentionally preserve the historical behavior. Centralizing
    them makes the debt visible and gives Phase 3E/3F one place to instrument,
    tune, or replace with a learned scorer.
    """
    generic_constraint_min_confidence: float = 0.55
    low_confidence_hint_cutoff: float = 0.55
    low_confidence_hint_confidence: float = 0.50
    claim_pitfall_min_confidence: float = 0.65
    failure_pattern_min_confidence: float = 0.75
    segment_tree_requirement_min_confidence: float = 0.82
    dijkstra_negative_pitfall_min_confidence: float = 0.84
    context_constraint_confidence: float = 0.88
    context_segment_tree_confidence: float = 0.90
    context_wide_int_confidence: float = 0.91
    context_pitfall_confidence: float = 0.92
    context_shortest_distance_confidence: float = 0.84
    procedure_shortest_path_confidence: float = 0.92
    procedure_dijkstra_precondition_confidence: float = 0.84
    procedure_negative_cycle_confidence: float = 0.82
    procedure_negative_edge_confidence: float = 0.82
    provisional_missing_context_confidence: float = 0.65
    provisional_bridge_hypothesis_confidence: float = 0.68
    activation_anchor_bonus: float = 0.15
    activation_signal_bonus: float = 0.10
    overlap_min_hits: int = 1
    content_token_min_chars: int = 4


DEFAULT_HEURISTICS = ActivationHeuristicConfig()


@dataclass
class SessionContext:
    session_id: str
    graph_id: str
    domain: Optional[str]
    question: str
    task_kind: Optional[str]
    constraints: List[str]
    requested_outputs: List[str]
    retrieved_anchor_ids: List[str]
    active_node_ids: List[str]
    missing_context: List[str]
    budget_snapshot: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @staticmethod
    def from_dict(d: Mapping[str, Any]) -> "SessionContext":
        return SessionContext(
            session_id=str(d["session_id"]),
            graph_id=str(d["graph_id"]),
            domain=d.get("domain"),
            question=str(d.get("question", "")),
            task_kind=d.get("task_kind"),
            constraints=list(d.get("constraints", [])),
            requested_outputs=list(d.get("requested_outputs", [])),
            retrieved_anchor_ids=list(d.get("retrieved_anchor_ids", [])),
            active_node_ids=list(d.get("active_node_ids", [])),
            missing_context=list(d.get("missing_context", [])),
            budget_snapshot=dict(d.get("budget_snapshot", {})),
        )


@dataclass
class GraphSignal:
    signal_id: str
    source_node_id: str
    kind: SignalKind
    payload: str
    confidence: float
    evidence_node_ids: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @staticmethod
    def from_dict(d: Mapping[str, Any]) -> "GraphSignal":
        return GraphSignal(
            signal_id=str(d["signal_id"]),
            source_node_id=str(d["source_node_id"]),
            kind=d["kind"],
            payload=str(d.get("payload", "")),
            confidence=float(d.get("confidence", 0.0)),
            evidence_node_ids=list(d.get("evidence_node_ids", [])),
        )


@dataclass
class ActivatedNode:
    node_id: str
    node_type: str
    activation_score: float
    activation_reason: str
    emitted_signal_ids: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @staticmethod
    def from_dict(d: Mapping[str, Any]) -> "ActivatedNode":
        return ActivatedNode(
            node_id=str(d["node_id"]),
            node_type=str(d.get("node_type", "unknown")),
            activation_score=float(d.get("activation_score", 0.0)),
            activation_reason=str(d.get("activation_reason", "")),
            emitted_signal_ids=list(d.get("emitted_signal_ids", [])),
        )


@dataclass
class FrameItem:
    item_id: str
    kind: str
    text: str
    priority: int
    source_signal_ids: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @staticmethod
    def from_dict(d: Mapping[str, Any]) -> "FrameItem":
        return FrameItem(
            item_id=str(d["item_id"]),
            kind=str(d.get("kind", "")),
            text=str(d.get("text", "")),
            priority=int(d.get("priority", 0)),
            source_signal_ids=list(d.get("source_signal_ids", [])),
        )


@dataclass
class GraphTaskFrame:
    session_id: str
    constraints: List[FrameItem] = field(default_factory=list)
    pitfalls: List[FrameItem] = field(default_factory=list)
    suggested_structures: List[FrameItem] = field(default_factory=list)
    relevant_examples: List[FrameItem] = field(default_factory=list)
    procedure_suggestions: List[FrameItem] = field(default_factory=list)
    unresolved_gaps: List[FrameItem] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @staticmethod
    def from_dict(d: Mapping[str, Any]) -> "GraphTaskFrame":
        return GraphTaskFrame(
            session_id=str(d["session_id"]),
            constraints=[FrameItem.from_dict(x) for x in d.get("constraints", [])],
            pitfalls=[FrameItem.from_dict(x) for x in d.get("pitfalls", [])],
            suggested_structures=[FrameItem.from_dict(x) for x in d.get("suggested_structures", [])],
            relevant_examples=[FrameItem.from_dict(x) for x in d.get("relevant_examples", [])],
            procedure_suggestions=[FrameItem.from_dict(x) for x in d.get("procedure_suggestions", [])],
            unresolved_gaps=[FrameItem.from_dict(x) for x in d.get("unresolved_gaps", [])],
        )

    def all_items(self) -> List[FrameItem]:
        return (
            list(self.constraints)
            + list(self.pitfalls)
            + list(self.suggested_structures)
            + list(self.relevant_examples)
            + list(self.procedure_suggestions)
            + list(self.unresolved_gaps)
        )


@dataclass
class GraphActivationTrace:
    session_id: str
    context: SessionContext
    activated_nodes: List[ActivatedNode]
    signals: List[GraphSignal]
    provisional_nodes: List[Dict[str, Any]]
    task_frame: GraphTaskFrame
    coverage_result: Optional[Dict[str, Any]] = None
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="seconds")
    )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "session_id": self.session_id,
            "context": self.context.to_dict(),
            "activated_nodes": [n.to_dict() for n in self.activated_nodes],
            "signals": [s.to_dict() for s in self.signals],
            "provisional_nodes": list(self.provisional_nodes),
            "task_frame": self.task_frame.to_dict(),
            "coverage_result": self.coverage_result,
            "timestamp": self.timestamp,
        }

    @staticmethod
    def from_dict(d: Mapping[str, Any]) -> "GraphActivationTrace":
        return GraphActivationTrace(
            session_id=str(d["session_id"]),
            context=SessionContext.from_dict(d["context"]),
            activated_nodes=[ActivatedNode.from_dict(x) for x in d.get("activated_nodes", [])],
            signals=[GraphSignal.from_dict(x) for x in d.get("signals", [])],
            provisional_nodes=[dict(x) for x in d.get("provisional_nodes", [])],
            task_frame=GraphTaskFrame.from_dict(d["task_frame"]),
            coverage_result=dict(d["coverage_result"]) if d.get("coverage_result") else None,
            timestamp=str(d.get("timestamp", "")),
        )


class ActivationTraceLogger:
    """Append-only JSONL persistence for GraphActivationTrace."""

    def __init__(self, trace_root: Path = DEFAULT_ACTIVATION_TRACE_ROOT) -> None:
        self.trace_root = Path(trace_root)

    def _today_path(self) -> Path:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d")
        return self.trace_root / f"activation_{stamp}.jsonl"

    def append(self, trace: GraphActivationTrace) -> Path:
        self.trace_root.mkdir(parents=True, exist_ok=True)
        path = self._today_path()
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(trace.to_dict(), ensure_ascii=False) + "\n")
        return path

    def read_all(self) -> List[GraphActivationTrace]:
        if not self.trace_root.exists():
            return []
        traces: List[GraphActivationTrace] = []
        for path in sorted(self.trace_root.glob("activation_*.jsonl")):
            for line in path.read_text(encoding="utf-8-sig").splitlines():
                if not line.strip():
                    continue
                traces.append(GraphActivationTrace.from_dict(json.loads(line)))
        return traces


BehaviorFn = Callable[[Node, SessionContext], List[GraphSignal]]


def run_graph_activation(
    *,
    session_id: str,
    graph_id: str,
    question: str,
    graph: MemoryGraph,
    anchor_ids: Sequence[str],
    budget_snapshot: Optional[Dict[str, Any]] = None,
    procedure_pool: Optional[Sequence[ProcedureNode]] = None,
    config: Optional[ActivationConfig] = None,
) -> GraphActivationTrace:
    """Build a context-aware task frame from anchors and nearby graph nodes."""
    cfg = config or ActivationConfig()
    active_ids = graph.local_neighborhood(anchor_ids, max_hops=cfg.max_hops, max_nodes=cfg.max_active_nodes)
    context = build_session_context(
        session_id=session_id,
        graph_id=graph_id,
        question=question,
        graph=graph,
        anchor_ids=list(anchor_ids),
        active_node_ids=active_ids,
        budget_snapshot=budget_snapshot or {},
    )

    signals: List[GraphSignal] = []
    activated: List[ActivatedNode] = []
    for node_id in active_ids:
        node = graph.nodes.get(node_id)
        if node is None:
            continue
        node_signals = emit_signals_for_node(node, context)
        node_signals = _dedupe_signals(node_signals)
        signals.extend(node_signals)
        activated.append(ActivatedNode(
            node_id=node_id,
            node_type=canonical_node_type(node.node_type),
            activation_score=_activation_score(node, context, node_signals),
            activation_reason="retrieved_anchor" if node_id in anchor_ids else "local_neighborhood",
            emitted_signal_ids=[s.signal_id for s in node_signals],
        ))

    signals.extend(_emit_context_signals(context))
    signals.extend(_emit_procedure_signals(context, procedure_pool or []))
    signals = _dedupe_signals(signals)[: cfg.max_signals]

    provisional_nodes = _build_provisional_nodes(context, signals, cfg)
    context.missing_context = [
        n["text"] for n in provisional_nodes if n.get("node_type") == "session_gap"
    ]
    signals.extend(_signals_from_provisional_nodes(provisional_nodes))
    signals = _dedupe_signals(signals)[: cfg.max_signals]

    task_frame = build_task_frame(session_id=session_id, signals=signals, config=cfg)
    return GraphActivationTrace(
        session_id=session_id,
        context=context,
        activated_nodes=activated,
        signals=signals,
        provisional_nodes=provisional_nodes,
        task_frame=task_frame,
    )


def build_session_context(
    *,
    session_id: str,
    graph_id: str,
    question: str,
    graph: MemoryGraph,
    anchor_ids: Sequence[str],
    active_node_ids: Sequence[str],
    budget_snapshot: Mapping[str, Any],
) -> SessionContext:
    return SessionContext(
        session_id=session_id,
        graph_id=graph_id,
        domain=_infer_domain(graph),
        question=question,
        task_kind=_infer_task_kind(question),
        constraints=_extract_constraints(question),
        requested_outputs=_extract_requested_outputs(question),
        retrieved_anchor_ids=list(anchor_ids),
        active_node_ids=list(active_node_ids),
        missing_context=[],
        budget_snapshot=dict(budget_snapshot),
    )


def emit_signals_for_node(node: Node, context: SessionContext) -> List[GraphSignal]:
    node_type = canonical_node_type(node.node_type)
    behavior = BEHAVIOR_REGISTRY.get(node_type, emit_generic_signal)
    return behavior(node, context)


def emit_generic_signal(node: Node, context: SessionContext) -> List[GraphSignal]:
    text = _clean_text(node.text)
    if not text or not _has_overlap(text, context.question):
        return []
    kind: SignalKind = (
        "constraint"
        if node.confidence >= DEFAULT_HEURISTICS.generic_constraint_min_confidence
        else "example"
    )
    return [_signal(node.id, kind, text, node.confidence, [node.id])]


def emit_fact_constraints(node: Node, context: SessionContext) -> List[GraphSignal]:
    text = _clean_text(node.text)
    if not text or not _has_overlap(text, context.question):
        return []
    return [_signal(node.id, "constraint", text, node.confidence, [node.id])]


def emit_claim_constraints_or_warnings(node: Node, context: SessionContext) -> List[GraphSignal]:
    text = _clean_text(node.text)
    if not text or not _has_overlap(text, context.question):
        return []
    lower = text.lower()
    if any(tok in lower for tok in ("fail", "unsafe", "invalid", "negative edge", "contradict")):
        return [_signal(node.id, "pitfall", text, max(
            node.confidence,
            DEFAULT_HEURISTICS.claim_pitfall_min_confidence,
        ), [node.id])]
    return [_signal(node.id, "constraint", text, node.confidence, [node.id])]


def emit_relevant_example_signal(node: Node, context: SessionContext) -> List[GraphSignal]:
    text = _clean_text(node.text)
    if not text or not _has_overlap(text, context.question):
        return []
    signals = [_signal(node.id, "example", text, node.confidence, [node.id])]
    lower = f"{text} {context.question}".lower()
    if "segment" in lower and "subarray" in lower:
        signals.append(_signal(
            node.id,
            "answer_requirement",
            "Use a segment tree node with sum, max_prefix, max_suffix, and max_sub/best.",
            max(node.confidence, DEFAULT_HEURISTICS.segment_tree_requirement_min_confidence),
            [node.id],
        ))
    if "dijkstra" in lower and "negative" in lower:
        signals.append(_signal(
            node.id,
            "pitfall",
            "Dijkstra is unsafe when an input edge has negative weight.",
            max(node.confidence, DEFAULT_HEURISTICS.dijkstra_negative_pitfall_min_confidence),
            [node.id],
        ))
    return signals


def emit_summary_signal(node: Node, context: SessionContext) -> List[GraphSignal]:
    text = _clean_text(node.text)
    if not text or not _has_overlap(text, context.question):
        return []
    return [_signal(node.id, "constraint", text, node.confidence, [node.id])]


def emit_low_confidence_hint(node: Node, context: SessionContext) -> List[GraphSignal]:
    text = _clean_text(node.text)
    if (
        not text
        or node.confidence >= DEFAULT_HEURISTICS.low_confidence_hint_cutoff
        or not _has_overlap(text, context.question)
    ):
        return []
    return [_signal(
        node.id,
        "pitfall",
        f"Low-confidence related hypothesis: {text}",
        DEFAULT_HEURISTICS.low_confidence_hint_confidence,
        [node.id],
    )]


def emit_pitfall_signal(node: Node, context: SessionContext) -> List[GraphSignal]:
    text = _clean_text(node.text)
    if not text or not _has_overlap(text, context.question):
        return []
    return [_signal(node.id, "pitfall", text, max(
        node.confidence,
        DEFAULT_HEURISTICS.failure_pattern_min_confidence,
    ), [node.id])]


BEHAVIOR_REGISTRY: Dict[str, BehaviorFn] = {
    "fact": emit_fact_constraints,
    "claim": emit_claim_constraints_or_warnings,
    "statement": emit_claim_constraints_or_warnings,
    "example": emit_relevant_example_signal,
    "summary": emit_summary_signal,
    "hub": emit_summary_signal,
    "bridge": emit_summary_signal,
    "hypothesis": emit_low_confidence_hint,
    "failure_pattern": emit_pitfall_signal,
}


def build_task_frame(
    *,
    session_id: str,
    signals: Sequence[GraphSignal],
    config: Optional[ActivationConfig] = None,
) -> GraphTaskFrame:
    cfg = config or ActivationConfig()
    frame = GraphTaskFrame(session_id=session_id)
    seen_text: set[str] = set()
    category_caps = {
        "pitfalls": 4,
        "procedure_suggestions": 3,
        "suggested_structures": 4,
        "unresolved_gaps": 2,
        "relevant_examples": 2,
        "constraints": 6,
    }

    ranked = sorted(signals, key=lambda s: (-_signal_priority(s), -s.confidence, s.payload))

    for category in (
        "pitfalls",
        "procedure_suggestions",
        "suggested_structures",
        "unresolved_gaps",
        "constraints",
        "relevant_examples",
    ):
        for sig in ranked:
            text = _frame_text(sig.payload)
            if not text:
                continue
            sig_category = _frame_category(sig, text)
            if sig_category != category:
                continue
            if len(getattr(frame, category)) >= category_caps[category]:
                continue
            if len(frame.all_items()) >= cfg.max_task_frame_items:
                break
            key = _dedupe_key(text)
            if key in seen_text:
                continue
            seen_text.add(key)
            item = FrameItem(
                item_id=f"fi_{uuid.uuid4().hex[:10]}",
                kind=sig.kind,
                text=text,
                priority=_signal_priority(sig),
                source_signal_ids=[sig.signal_id],
            )
            getattr(frame, category).append(item)

    if len(frame.all_items()) < cfg.max_task_frame_items:
        for sig in ranked:
            if len(frame.all_items()) >= cfg.max_task_frame_items:
                break
            text = _frame_text(sig.payload)
            if not text:
                continue
            key = _dedupe_key(text)
            if key in seen_text:
                continue
            seen_text.add(key)
            category = _frame_category(sig, text)
            item = FrameItem(
                item_id=f"fi_{uuid.uuid4().hex[:10]}",
                kind=sig.kind,
                text=text,
                priority=_signal_priority(sig),
                source_signal_ids=[sig.signal_id],
            )
            getattr(frame, category).append(item)

    _trim_frame(frame, cfg.max_frame_chars)
    return frame


def render_task_frame(frame: Optional[GraphTaskFrame]) -> str:
    if frame is None or not frame.all_items():
        return ""
    sections = [
        ("Constraints", frame.constraints),
        ("Pitfalls", frame.pitfalls),
        ("Suggested structures", frame.suggested_structures),
        ("Relevant examples", frame.relevant_examples),
        ("Procedure suggestions", frame.procedure_suggestions),
        ("Unresolved gaps", frame.unresolved_gaps),
    ]
    lines = ["<graph_task_frame>"]
    for title, items in sections:
        if not items:
            continue
        lines.append(f"{title}:")
        for item in sorted(items, key=lambda i: (-i.priority, i.text)):
            lines.append(f"- {item.text}")
        lines.append("")
    if lines[-1] == "":
        lines.pop()
    lines.append("</graph_task_frame>")
    return "\n".join(lines)


def evaluate_coverage(frame: GraphTaskFrame, answer: str) -> Dict[str, Any]:
    answer_l = (answer or "").lower()
    addressed: List[str] = []
    missed: List[str] = []
    details: List[Dict[str, Any]] = []
    for item in frame.all_items():
        ok = _item_addressed(item.text, answer_l)
        (addressed if ok else missed).append(item.item_id)
        details.append({
            "item_id": item.item_id,
            "kind": item.kind,
            "text": item.text,
            "addressed": ok,
        })
    return {
        "addressed_item_ids": addressed,
        "missed_item_ids": missed,
        "coverage": (len(addressed) / len(frame.all_items())) if frame.all_items() else 1.0,
        "items": details,
    }


def attach_activation_to_session(
    session: SessionSubgraphController,
    trace: GraphActivationTrace,
) -> None:
    """Project activation trace into the session subgraph.

    These are observations, not CRUD state mutations, so they intentionally do
    not journal audit entries.
    """
    if any(sig.source_node_id == "session_context" for sig in trace.signals):
        session.subgraph.nodes.setdefault("session_context", {
            "id": "session_context",
            "node_type": "session_context",
            "text": trace.context.question,
            "metadata": {
                "provider": "phase3c-activation",
                "graph_id": trace.context.graph_id,
                "task_kind": trace.context.task_kind,
            },
        })

    for sig in trace.signals:
        session.subgraph.nodes.setdefault(sig.signal_id, {
            "id": sig.signal_id,
            "node_type": "activation_signal",
            "kind": sig.kind,
            "text": sig.payload,
            "source_node_id": sig.source_node_id,
            "confidence": sig.confidence,
            "evidence_node_ids": list(sig.evidence_node_ids),
            "metadata": {"provider": "phase3c-activation"},
        })
        if sig.source_node_id:
            session.add_edge(
                src=sig.source_node_id,
                dst=sig.signal_id,
                relation="emits_signal",
                metadata={"provider": "phase3c-activation"},
            )

    for node in trace.provisional_nodes:
        node_id = str(node["id"])
        session.subgraph.nodes.setdefault(node_id, dict(node))
        for source_id in node.get("derived_from", []) or []:
            session.add_edge(
                src=node_id,
                dst=str(source_id),
                relation="derived_from",
                metadata={"provider": "phase3c-activation"},
            )
        if node.get("node_type") == "session_bridge" and node.get("fills_gap"):
            session.add_edge(
                src=node_id,
                dst=str(node["fills_gap"]),
                relation="fills_gap",
                metadata={"provider": "phase3c-activation"},
            )

    for item in trace.task_frame.all_items():
        session.subgraph.nodes.setdefault(item.item_id, {
            "id": item.item_id,
            "node_type": "task_frame_item",
            "kind": item.kind,
            "text": item.text,
            "priority": item.priority,
            "source_signal_ids": list(item.source_signal_ids),
            "metadata": {"provider": "phase3c-activation"},
        })
        for sig_id in item.source_signal_ids:
            session.add_edge(
                src=item.item_id,
                dst=sig_id,
                relation="frame_includes",
                metadata={"provider": "phase3c-activation"},
            )


def _emit_context_signals(context: SessionContext) -> List[GraphSignal]:
    q = context.question.lower()
    signals: List[GraphSignal] = []
    for constraint in context.constraints:
        signals.append(_signal(
            "session_context",
            "constraint",
            constraint,
            DEFAULT_HEURISTICS.context_constraint_confidence,
            [],
        ))

    if "subarray" in q and "update" in q:
        signals.append(_signal(
            "session_context",
            "answer_requirement",
            "Use a segment tree node with sum, max_prefix, max_suffix, and max_sub/best.",
            DEFAULT_HEURISTICS.context_segment_tree_confidence,
            [],
        ))
    if any(tok in q for tok in ("64-bit", "64 bit", "overflow", "1e9", "10^9")):
        signals.append(_signal(
            "session_context",
            "answer_requirement",
            "Use long long/int64 for sums and segment aggregate fields.",
            DEFAULT_HEURISTICS.context_wide_int_confidence,
            [],
        ))
    if "negative" in q and "subarray" in q:
        signals.append(_signal(
            "session_context",
            "pitfall",
            "For non-empty subarrays, all-negative arrays must return the maximum element, not 0.",
            DEFAULT_HEURISTICS.context_pitfall_confidence,
            [],
        ))
    if "dijkstra" in q and "negative" in q:
        signals.append(_signal(
            "session_context",
            "pitfall",
            "Dijkstra is unsafe when an input edge has negative weight.",
            DEFAULT_HEURISTICS.context_pitfall_confidence,
            [],
        ))
    if "shortest" in q and "distance" in q:
        signals.append(_signal(
            "session_context",
            "answer_requirement",
            "If asked for a shortest distance, include the numeric distance and the path when derivable.",
            DEFAULT_HEURISTICS.context_shortest_distance_confidence,
            [],
        ))
    return signals


def _emit_procedure_signals(
    context: SessionContext,
    procedure_pool: Sequence[ProcedureNode],
) -> List[GraphSignal]:
    q = context.question.lower()
    signals: List[GraphSignal] = []
    for proc in procedure_pool:
        hay = f"{proc.name} {proc.purpose} {proc.when_to_use}".lower()
        confidence = 0.0
        if ("dijkstra" in q or "shortest path" in q or "shortest-path" in q) and "verifyshortestpath" in hay:
            confidence = DEFAULT_HEURISTICS.procedure_shortest_path_confidence
        elif "dijkstra" in q and any(tok in hay for tok in ("precondition", "non-negative", "nonnegative")):
            confidence = DEFAULT_HEURISTICS.procedure_dijkstra_precondition_confidence
        elif "negative cycle" in q and "negative cycle" in hay:
            confidence = DEFAULT_HEURISTICS.procedure_negative_cycle_confidence
        elif "negative edge" in q and "nonnegative" in hay:
            confidence = DEFAULT_HEURISTICS.procedure_negative_edge_confidence
        if confidence > 0.0:
            signals.append(_signal(
                proc.id,
                "procedure_suggestion",
                f"Procedure may apply: {proc.name} - {proc.purpose}",
                confidence,
                [proc.id],
            ))
    return signals


def _build_provisional_nodes(
    context: SessionContext,
    signals: Sequence[GraphSignal],
    config: ActivationConfig,
) -> List[Dict[str, Any]]:
    required = _required_context(context)
    covered_text = " ".join(s.payload.lower() for s in signals)
    nodes: List[Dict[str, Any]] = []
    for label, text, bridge_text in required:
        if _covered(label, covered_text):
            continue
        gap_id = f"gap_{uuid.uuid4().hex[:10]}"
        nodes.append({
            "id": gap_id,
            "node_type": "session_gap",
            "text": text,
            "label": label,
            "derived_from": ["session_context"] + list(context.retrieved_anchor_ids[:3]),
            "metadata": {"provider": "phase3c-activation", "session_scoped": True},
        })
        if len(nodes) >= config.max_provisional_nodes:
            break
        bridge_id = f"bridge_{uuid.uuid4().hex[:10]}"
        nodes.append({
            "id": bridge_id,
            "node_type": "session_bridge",
            "text": bridge_text,
            "fills_gap": gap_id,
            "derived_from": ["session_context"] + list(context.active_node_ids[:3]),
            "metadata": {"provider": "phase3c-activation", "session_scoped": True},
        })
        if len(nodes) >= config.max_provisional_nodes:
            break
    return nodes[: config.max_provisional_nodes]


def _signals_from_provisional_nodes(nodes: Sequence[Mapping[str, Any]]) -> List[GraphSignal]:
    out: List[GraphSignal] = []
    for node in nodes:
        ntype = node.get("node_type")
        if ntype == "session_gap":
            out.append(_signal(
                str(node["id"]),
                "missing_context",
                str(node.get("text", "")),
                DEFAULT_HEURISTICS.provisional_missing_context_confidence,
                [str(node["id"])],
            ))
        elif ntype == "session_bridge":
            out.append(_signal(
                str(node["id"]),
                "bridge_hypothesis",
                str(node.get("text", "")),
                DEFAULT_HEURISTICS.provisional_bridge_hypothesis_confidence,
                [str(node["id"])],
            ))
    return out


def _required_context(context: SessionContext) -> List[tuple[str, str, str]]:
    q = context.question.lower()
    required: List[tuple[str, str, str]] = []
    if "subarray" in q and "negative" in q:
        required.append((
            "all_negative_non_empty",
            "Need all-negative handling for non-empty max subarray.",
            "Non-empty max-subarray combine must avoid empty-subarray clamping.",
        ))
    if "subarray" in q and "update" in q:
        required.append((
            "segment_tree_combine",
            "Need segment-tree combine state for dynamic max subarray.",
            "Use sum, prefix, suffix, and best/max_sub fields per segment.",
        ))
    if any(tok in q for tok in ("200000", "2e5", "10^5", "1e5")) or "long long" in q:
        required.append((
            "wide_integer",
            "Need wide integer handling for large sums.",
            "Use long long/int64 for sums and segment aggregate fields.",
        ))
    if "dijkstra" in q and "negative" in q:
        required.append((
            "negative_edge_safety",
            "Need shortest-path safety check for negative edge weights.",
            "Verify non-negative edges before trusting Dijkstra; use Bellman-Ford when violated.",
        ))
    return required


def _covered(label: str, covered_text: str) -> bool:
    expectations = {
        "all_negative_non_empty": ("all-negative", "non-empty", "maximum element", "not 0"),
        "segment_tree_combine": ("segment tree", "max_prefix", "prefix", "suffix", "max_sub", "best"),
        "wide_integer": ("long long", "int64", "wide integer"),
        "negative_edge_safety": ("negative edge", "non-negative", "bellman-ford", "unsafe"),
    }
    return any(tok in covered_text for tok in expectations.get(label, (label,)))


def _infer_domain(graph: MemoryGraph) -> Optional[str]:
    for key in ("domain", "topic", "graph_domain"):
        if graph.metadata.get(key):
            return str(graph.metadata[key])
    counts: Dict[str, int] = {}
    for node in graph.nodes.values():
        value = node.metadata.get("domain") or node.metadata.get("topic")
        if value:
            counts[str(value)] = counts.get(str(value), 0) + 1
    if not counts:
        return None
    return sorted(counts.items(), key=lambda item: (-item[1], item[0]))[0][0]


def _infer_task_kind(question: str) -> Optional[str]:
    q = question.lower()
    if any(tok in q for tok in ("c++", "complexity", "algorithm", "ioi", "point update", "queries")):
        return "algorithm_design"
    if any(tok in q for tok in ("verify", "safe", "trusted", "precondition")):
        return "verification"
    if any(tok in q for tok in ("explain", "define", "what is")):
        return "explanation"
    return None


def _extract_constraints(question: str) -> List[str]:
    q = question.lower()
    constraints: List[str] = []
    for match in re.finditer(r"\b[nqm]\s*,?\s*[nqm]?\s*<=\s*\d+", question, re.IGNORECASE):
        constraints.append(match.group(0).strip())
    if "negative values" in q or "negative numbers" in q or "negative edge" in q:
        constraints.append("negative values/weights are present or allowed")
    if "non-empty" in q or "non empty" in q:
        constraints.append("subarray must be non-empty")
    if "c++17" in q or "c++" in q:
        constraints.append("C++17 implementation expected")
    if any(tok in q for tok in ("64-bit", "64 bit", "overflow", "1e9", "10^9")):
        constraints.append("use 64-bit integer arithmetic for large sums")
    if "point update" in q or "updates" in q:
        constraints.append("must support updates efficiently")
    if any(tok in q for tok in ("200000", "2e5", "10^5", "1e5")):
        constraints.append("large input size requires logarithmic or near-linear operations")
    return _dedupe_text(constraints)


def _extract_requested_outputs(question: str) -> List[str]:
    q = question.lower()
    outputs: List[str] = []
    if "shortest distance" in q or "distance" in q:
        outputs.append("shortest distance")
    if "path" in q:
        outputs.append("path")
    if "code" in q or "c++" in q:
        outputs.append("implementation")
    if "complexity" in q:
        outputs.append("complexity")
    if "safe" in q or "trusted" in q or "applicable" in q:
        outputs.append("applicability verdict")
    return _dedupe_text(outputs)


def _activation_score(node: Node, context: SessionContext, signals: Sequence[GraphSignal]) -> float:
    base = (float(node.confidence) + float(node.importance)) / 2.0
    if node.id in context.retrieved_anchor_ids:
        base += DEFAULT_HEURISTICS.activation_anchor_bonus
    if signals:
        base += DEFAULT_HEURISTICS.activation_signal_bonus
    return min(1.0, max(0.0, base))


def _signal(
    source_node_id: str,
    kind: SignalKind,
    payload: str,
    confidence: float,
    evidence_node_ids: Sequence[str],
) -> GraphSignal:
    return GraphSignal(
        signal_id=f"gs_{uuid.uuid4().hex[:10]}",
        source_node_id=source_node_id,
        kind=kind,
        payload=_clean_text(payload),
        confidence=min(1.0, max(0.0, float(confidence))),
        evidence_node_ids=list(evidence_node_ids),
    )


def _signal_priority(signal: GraphSignal) -> int:
    priorities = {
        "pitfall": 95,
        "answer_requirement": 90,
        "constraint": 80,
        "procedure_suggestion": 70,
        "bridge_hypothesis": 60,
        "missing_context": 50,
        "example": 40,
    }
    return priorities.get(signal.kind, 10)


def _frame_category(signal: GraphSignal, text: str) -> str:
    lower = text.lower()
    if signal.kind == "pitfall":
        return "pitfalls"
    if signal.kind == "procedure_suggestion":
        return "procedure_suggestions"
    if signal.kind == "example":
        return "relevant_examples"
    if signal.kind in {"missing_context", "bridge_hypothesis"}:
        if any(tok in lower for tok in ("use ", "avoid ", "must ", "verify ", "long long", "segment tree")):
            return "suggested_structures"
        return "unresolved_gaps"
    if any(tok in lower for tok in ("segment tree", "max_prefix", "max_suffix", "max_sub", "best/max_sub", "long long", "int64")):
        return "suggested_structures"
    return "constraints"


def _frame_text(text: str) -> str:
    text = _clean_text(text)
    if len(text) > 240:
        return text[:237].rstrip() + "..."
    return text


def _trim_frame(frame: GraphTaskFrame, max_chars: int) -> None:
    while len(render_task_frame(frame)) > max_chars and frame.all_items():
        buckets = [
            frame.relevant_examples,
            frame.unresolved_gaps,
            frame.constraints,
            frame.suggested_structures,
            frame.procedure_suggestions,
            frame.pitfalls,
        ]
        for bucket in buckets:
            if bucket:
                bucket.pop()
                break
        else:
            break


def _item_addressed(item_text: str, answer_l: str) -> bool:
    text = item_text.lower()
    if "long long" in text or "int64" in text or "wide integer" in text:
        return "long long" in answer_l or "int64" in answer_l or "int64_t" in answer_l
    if "all-negative" in text or ("non-empty" in text and "subarray" in text):
        return (
            "all-negative" in answer_l
            or "all negative" in answer_l
            or "non-empty" in answer_l
            or "maximum element" in answer_l
            or "not 0" in answer_l
        )
    if "segment tree" in text or "max_prefix" in text or "max_suffix" in text or "max_sub" in text:
        return (
            "segment tree" in answer_l
            and any(tok in answer_l for tok in ("prefix", "pref", "suffix", "suff"))
            and any(tok in answer_l for tok in ("best", "max_sub", "maximum subarray"))
        )
    if "dijkstra" in text or "negative edge" in text:
        return "negative" in answer_l and ("dijkstra" in answer_l or "bellman" in answer_l or "unsafe" in answer_l)
    return _shared_constraint_addressed(
        text,
        answer_l,
        min_keyword_chars=5,
        max_keywords=8,
        max_required_hits=3,
    )


def _dedupe_signals(signals: Iterable[GraphSignal]) -> List[GraphSignal]:
    out: List[GraphSignal] = []
    seen: set[tuple[str, str]] = set()
    for sig in signals:
        key = (sig.kind, _dedupe_key(sig.payload))
        if key in seen or not sig.payload:
            continue
        seen.add(key)
        out.append(sig)
    return out


def _dedupe_text(values: Iterable[str]) -> List[str]:
    out: List[str] = []
    seen: set[str] = set()
    for value in values:
        key = _dedupe_key(value)
        if key and key not in seen:
            seen.add(key)
            out.append(value)
    return out


def _dedupe_key(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def _clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def _has_overlap(text: str, question: str, *, min_hits: int = DEFAULT_HEURISTICS.overlap_min_hits) -> bool:
    return has_token_overlap(
        text,
        question,
        min_hits=min_hits,
        min_chars=DEFAULT_HEURISTICS.content_token_min_chars,
    )


def _content_tokens(text: str) -> set[str]:
    return content_tokens(text, min_chars=DEFAULT_HEURISTICS.content_token_min_chars)
