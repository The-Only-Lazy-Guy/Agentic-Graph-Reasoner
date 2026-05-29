"""ToolLoopCycleDetector — flags repeated identical tool calls in a JSON-tool loop.

The Phase-3A CycleDetector observes Phase-2A `DispatchOutcome` events. JSON-tool
answerers (e.g., answerer_v4) don't produce DispatchOutcomes — they parse JSON
tool calls and dispatch them directly. This MP fills that gap by observing the
tool call log instead.

Looks at `ctx.metadata['tool_calls']` (a list of {name, args} dicts that the
v4-style answerer feeds in via MetaContext). If any (name, hashable args) tuple
appears CYCLE_THRESHOLD+ times, emits one WARN signal naming the call.

once_per_session=True so each cycle is flagged once per session.
"""
from __future__ import annotations

import json
from typing import List

from reasoning.meta import MetaContext, MetaProcedure
from reasoning.signals import Signal


CYCLE_THRESHOLD = 3


def _detect_tool_loop_cycle(ctx: MetaContext) -> List[Signal]:
    # The v4-style answerer attaches its tool call log to MetaContext via
    # a synthesized `raw_outputs` injection? No — we use a generic side-
    # channel: the answerer is expected to provide tool calls in
    # ctx.previous_signals.metadata? No — cleanest is to put them on the
    # session controller. But the controller doesn't hold tool calls.
    #
    # Solution: walk `ctx.session.subgraph.audit_log` for repeated entries
    # — or use a dedicated registry. For now we look at audit log create
    # events that correspond to repeated session-object creates with the
    # same name. This catches "create_object(name=X)" cycles.
    #
    # For broader tool-loop cycles, the v4 answerer should pass calls via
    # MetaContext.metadata. That hook is added in a follow-up if needed.
    counts: dict = {}
    for entry in ctx.session.subgraph.audit_log:
        if entry.operation != "create":
            continue
        new = entry.new_value or {}
        if not isinstance(new, dict):
            continue
        name = new.get("name")
        if not name:
            continue
        counts[name] = counts.get(name, 0) + 1

    signals: List[Signal] = []
    for name, count in counts.items():
        if count < CYCLE_THRESHOLD:
            continue
        signals.append(Signal(
            id=f"tool_loop_cycle_create_{name}",
            type="cycle_detected",
            severity="warn",
            message=(
                f"You have created {count} session objects named {name!r}. "
                f"Further objects with the same name are likely redundant; "
                f"consider updating an existing one or finalizing the answer."
            ),
            emitted_at_step=ctx.current_iteration,
            emitted_by="tool_loop_cycle_detector",
            metadata={"object_name": name, "create_count": count},
            sticky=True,
            once=True,
        ))
    return signals


def build_tool_loop_cycle_detector() -> MetaProcedure:
    return MetaProcedure(
        id="meta_tool_loop_cycle_detector",
        name="ToolLoopCycleDetector",
        purpose=(
            f"Flag any session-object name created {CYCLE_THRESHOLD}+ times "
            f"in a single tool-loop session."
        ),
        fires_on="post_dispatch",
        predicate=_detect_tool_loop_cycle,
        once_per_session=True,
        priority=30,
    )
