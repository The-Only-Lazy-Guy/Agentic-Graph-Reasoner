"""Capsule store — Phase 3F-alpha KV warm-pool prefix builder.

A capsule bundles related signals into a stable text prefix that is
sent as part of the system message so the LLM server can reuse its
KV cache across steps in the same session.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence


@dataclass
class Capsule:
    """A bundle of signals rendered as a stable text prefix for KV reuse."""
    id: str
    signal_ids: set[str]
    rendered_text: str


def build_capsules(signals: Sequence[Any]) -> List[Capsule]:
    """Build capsules from initial signals.

    Each capsule groups a subset of signals by kind and renders them
    into a stable text block.
    """
    if not signals:
        return []

    capsules: List[Capsule] = []
    constraint_signals: List[str] = []
    constraint_ids: set[str] = set()
    risk_signals: List[str] = []
    risk_ids: set[str] = set()
    other_signals: List[str] = []
    other_ids: set[str] = set()

    for sig in signals:
        kind = getattr(sig, "kind", "signal")
        text = getattr(sig, "text", "")
        sig_id = getattr(sig, "id", f"sig_{id(sig)}")
        if kind == "constraint":
            constraint_signals.append(f"- [{kind}] {text}")
            constraint_ids.add(sig_id)
        elif kind == "risk":
            risk_signals.append(f"- [{kind}] {text}")
            risk_ids.add(sig_id)
        else:
            other_signals.append(f"- [{kind}] {text}")
            other_ids.add(sig_id)

    if constraint_signals:
        capsules.append(Capsule(
            id="capsule_constraints",
            signal_ids=constraint_ids,
            rendered_text="Graph constraints:\n" + "\n".join(constraint_signals),
        ))
    if risk_signals:
        capsules.append(Capsule(
            id="capsule_risks",
            signal_ids=risk_ids,
            rendered_text="Known risks:\n" + "\n".join(risk_signals),
        ))
    if other_signals:
        capsules.append(Capsule(
            id="capsule_other",
            signal_ids=other_ids,
            rendered_text="Session signals:\n" + "\n".join(other_signals),
        ))

    return capsules


def select_capsules(
    question: str,
    capsules: List[Capsule],
    max_chars: int = 1200,
) -> List[Capsule]:
    """Select relevant capsules for the given question.

    Currently returns all capsules up to max_chars. Future versions
    may rank by semantic relevance to the question.
    """
    selected: List[Capsule] = []
    total_chars = 0
    for cap in capsules:
        text_len = len(cap.rendered_text) + 2
        if total_chars + text_len > max_chars:
            break
        selected.append(cap)
        total_chars += text_len
    return selected
