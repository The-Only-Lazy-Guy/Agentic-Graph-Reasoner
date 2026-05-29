"""CycleDetector — flags procedures invoked repeatedly with identical args.

Fires on `post_dispatch`. If the same (procedure_id, args_text) tuple
appears 3+ times in dispatch_outcomes, a WARN signal is emitted naming
the procedure and the args. once_per_session=True so each cycle gets
flagged exactly once per session (not on every subsequent iteration).

Conservative threshold: 3 invocations rather than 2, because some
legitimate patterns (e.g., parent + one explicit re-check) might
involve a second invocation with the same args. Three identical
invocations clearly suggests a loop.

The predicate is read-only; no action needed beyond signal emission.
"""
from __future__ import annotations

from typing import List

from reasoning.meta import MetaContext, MetaProcedure
from reasoning.signals import Signal


CYCLE_THRESHOLD = 3


def _detect_cycles(ctx: MetaContext) -> List[Signal]:
    counts: dict[tuple, list[str]] = {}
    name_for: dict[str, str] = {}
    for outcome in ctx.dispatch_outcomes:
        if outcome.procedure_id is None or outcome.object_id is None:
            continue
        key = (outcome.procedure_id, outcome.match.args_text or "")
        counts.setdefault(key, []).append(outcome.object_id)
        name_for[outcome.procedure_id] = outcome.match.procedure_name

    signals: List[Signal] = []
    for (proc_id, args), object_ids in counts.items():
        if len(object_ids) < CYCLE_THRESHOLD:
            continue
        proc_name = name_for[proc_id]
        # Stable id within session: procedure + hashed args
        args_hash = abs(hash(args)) % 10_000_000
        signals.append(Signal(
            id=f"cycle_{proc_id}_{args_hash}",
            type="cycle_detected",
            severity="warn",
            message=(
                f"{proc_name} has been invoked {len(object_ids)} times with "
                f"identical arguments. Further invocations with the same args "
                f"are likely redundant; consider a different approach or "
                f"finalize the answer."
            ),
            emitted_at_step=ctx.current_iteration,
            emitted_by="cycle_detector",
            related_node_ids=list(object_ids),
            metadata={"procedure_id": proc_id, "args_text": args,
                      "invocation_count": len(object_ids)},
            # sticky=True: the cycle warning fires at post_dispatch of
            # the offending iteration, so it MUST persist into the next
            # iteration's prompt for the model to actually see and react
            # to it. Without sticky, the signal would land in the
            # subgraph but never reach the model.
            sticky=True,
            once=True,
        ))
    return signals


def build_cycle_detector() -> MetaProcedure:
    return MetaProcedure(
        id="meta_cycle_detector",
        name="CycleDetector",
        purpose=(
            f"Flag any procedure invoked {CYCLE_THRESHOLD}+ times with "
            f"identical arguments in one session."
        ),
        fires_on="post_dispatch",
        predicate=_detect_cycles,
        once_per_session=True,
        priority=30,
    )
