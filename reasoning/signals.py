"""Signals — operational observations emitted by meta-procedures.

Phase 3A. A Signal is what a meta-procedure produces when its trigger
predicate matches state. Signals flow into the next iteration's prompt
as a `# System signals` section; the model reacts via its normal
reasoning, without any extra LLM call dedicated to meta-cognition.

Signals also persist into the session subgraph as nodes with
`node_type="signal"` so they survive the request and the front-end
can render them post-hoc.

See PHASE3A_PLAN.md §3.1 and §6 for the design rationale.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Literal, Mapping


# Severity ordering matters for prompt-rendering and prompt-budget capping
# (errors first, then warnings, then info).
SignalSeverity = Literal["error", "warn", "info"]
SEVERITY_ORDER: Dict[str, int] = {"error": 0, "warn": 1, "info": 2}

# Hook points in the reasoning loop where meta-procedures may fire.
SignalHook = Literal["pre_iter", "post_dispatch", "end_of_session"]


@dataclass
class Signal:
    """One observation by a meta-procedure.

    Persisted to the session subgraph (as node_type="signal") and
    rendered into the model's next prompt section.
    """
    id: str                          # short slug e.g. "cycle_detected_so_abc:0"
    type: str                        # short kind e.g. "cycle_detected"
    severity: SignalSeverity
    message: str                     # one-line natural-language summary for the model
    emitted_at_step: int             # session step when this fired
    emitted_by: str                  # name of the meta-procedure that fired
    related_node_ids: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    # Signals with sticky=True persist across iterations until the
    # session ends. Default sticky behaviour by severity:
    #   error -> sticky (sticky=True)
    #   warn/info -> per-iter (sticky=False)
    # Meta-procedures may override per Signal instance.
    sticky: bool = False
    # When once=True, the substrate suppresses re-emission of any signal
    # whose (type, sorted(related_node_ids)) tuple has already fired in
    # this session — prevents duplicate WARN spam for the same root
    # cause across iterations.
    once: bool = False

    # ---- serialization ------------------------------------------------ #

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @staticmethod
    def from_dict(d: Mapping[str, Any]) -> "Signal":
        return Signal(
            id=d["id"],
            type=d["type"],
            severity=d["severity"],
            message=d["message"],
            emitted_at_step=int(d.get("emitted_at_step", 0)),
            emitted_by=d.get("emitted_by", ""),
            related_node_ids=list(d.get("related_node_ids", [])),
            metadata=dict(d.get("metadata", {})),
            sticky=bool(d.get("sticky", False)),
            once=bool(d.get("once", False)),
        )

    # ---- session-subgraph node form ----------------------------------- #

    def to_node(self) -> Dict[str, Any]:
        """Serialize as a session-subgraph node (node_type='signal').

        The node form differs slightly from to_dict(): it includes a
        node_type field and uses 'text' for the human-readable message
        so the front-end's session renderer (which keys off node_type
        and text) picks it up uniformly with the other node kinds.
        """
        return {
            "id": self.id,
            "node_type": "signal",
            "text": self.message,
            "severity": self.severity,
            "type": self.type,
            "emitted_at_step": self.emitted_at_step,
            "emitted_by": self.emitted_by,
            "related_node_ids": list(self.related_node_ids),
            "metadata": dict(self.metadata),
            "sticky": self.sticky,
            "once": self.once,
        }

    @staticmethod
    def from_node(node: Mapping[str, Any]) -> "Signal":
        """Inverse of to_node(). Used when reloading a persisted session
        and reconstructing the Signal stream for replay or UI rendering.
        """
        return Signal(
            id=node["id"],
            type=node.get("type", ""),
            severity=node.get("severity", "info"),
            message=node.get("text", node.get("message", "")),
            emitted_at_step=int(node.get("emitted_at_step", 0)),
            emitted_by=node.get("emitted_by", ""),
            related_node_ids=list(node.get("related_node_ids", [])),
            metadata=dict(node.get("metadata", {})),
            sticky=bool(node.get("sticky", False)),
            once=bool(node.get("once", False)),
        )


# ---- helpers ---------------------------------------------------------- #

def severity_rank(sev: str) -> int:
    """Lower = more urgent. Used for prompt cap ordering."""
    return SEVERITY_ORDER.get(sev, 99)


def signal_dedupe_key(sig: Signal) -> tuple:
    """The tuple used by `once=True` deduping: meta-procedures that
    target the same root cause (same type + same related_node_ids set)
    only fire once per session."""
    return (sig.type, tuple(sorted(sig.related_node_ids)))


def render_signals_block(signals: List[Signal], max_signals: int = 5) -> str:
    """Render a list of Signals as the `# System signals` prompt section.

    Behaviour:
      - Primary sort: severity (ERROR > WARN > INFO).
      - Secondary sort within severity: NEWEST FIRST (descending
        emitted_at_step). Reasoning: a sticky error that's been
        unresolved for many iterations is presumably one the model
        already failed to address; the more actionable info is the
        recent set. Combined with the carrier-cap "drop oldest"
        semantics, stale signals fade out of the prompt naturally.
      - Capped at max_signals (default 5; resolved decision §11).
      - If more signals exist than the cap allows, an overflow meta-line
        is appended listing how many were suppressed.
      - Returns empty string if there are no signals.

    The model sees these verbatim — the `message` text is the only thing
    each meta-procedure controls about how the signal appears.
    """
    if not signals:
        return ""

    ordered = sorted(signals, key=lambda s: (severity_rank(s.severity), -s.emitted_at_step))
    visible = ordered[:max_signals]
    suppressed = len(ordered) - len(visible)

    lines = ["# System signals (deterministic detectors)", ""]
    for s in visible:
        prefix = s.severity.upper()
        lines.append(f"- {prefix:5s} {s.type}: {s.message}")
    if suppressed > 0:
        lines.append(
            f"- INFO  suppressed_overflow: {suppressed} additional signals "
            f"suppressed for prompt brevity."
        )
    return "\n".join(lines).strip()
