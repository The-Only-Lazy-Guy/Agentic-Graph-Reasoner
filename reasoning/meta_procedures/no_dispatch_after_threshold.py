"""NoDispatchAfterThreshold — INFO if many iterations pass without dispatch.

Fires on `pre_iter` starting at iteration `THRESHOLD` (default 2).
If no procedure has been invoked by then, emits an INFO signal
reminding the model that procedures are available and the question
might benefit from one — OR that it should just finalize a direct
answer.

This addresses the Phase-1 "wasted iterations on conceptual questions"
issue from a different angle than the directive-tightening that
shipped post-hoc: signal the staleness instead of (or in addition to)
restricting the directive.

once_per_session=True so the nudge fires once and doesn't pester.
The model either responds (by invoking a procedure or finalizing)
or it doesn't (predicate already fired, won't re-fire).

REPLACED `RepeatedAnchorObservation` from PHASE3A_PLAN.md §7.5:
RepeatedAnchorObservation would have checked whether anchor_ids
changed across iterations. But the current substrate computes
anchor_ids ONCE outside the reasoning loop — they never change
across iterations within a session. The meta-procedure would have
fired every iteration after the first, with no useful signal content.
This NoDispatchAfterThreshold detector is more useful for the
current substrate.
"""
from __future__ import annotations

from typing import List

from reasoning.meta import MetaContext, MetaProcedure
from reasoning.signals import Signal


DISPATCH_THRESHOLD_ITER = 2


def _detect_no_dispatch_after_threshold(ctx: MetaContext) -> List[Signal]:
    if ctx.current_iteration < DISPATCH_THRESHOLD_ITER:
        return []
    if ctx.dispatch_outcomes:
        return []  # Something has been invoked; not stale

    if not ctx.procedure_names:
        # No procedures registered at all — can't suggest invoking one
        return []

    available_str = ", ".join(sorted(ctx.procedure_names))
    return [Signal(
        id=f"no_dispatch_after_iter_{ctx.current_iteration}",
        type="no_dispatch_stale",
        severity="info",
        message=(
            f"You've completed {ctx.current_iteration} reasoning turns without "
            f"invoking any procedure. If a procedure applies "
            f"(available: {available_str}), invoke it now with "
            f"\"I'll apply <ProcedureName> to <args>\". Otherwise, finalize "
            f"the answer directly in this turn."
        ),
        emitted_at_step=ctx.current_iteration,
        emitted_by="no_dispatch_after_threshold",
        metadata={
            "threshold_iteration": DISPATCH_THRESHOLD_ITER,
            "current_iteration": ctx.current_iteration,
            "available_procedures": sorted(ctx.procedure_names),
        },
        sticky=False,
        once=True,
    )]


def build_no_dispatch_after_threshold() -> MetaProcedure:
    return MetaProcedure(
        id="meta_no_dispatch_after_threshold",
        name="NoDispatchAfterThreshold",
        purpose=(
            f"Nudge the model to either invoke a procedure or finalize the "
            f"answer when {DISPATCH_THRESHOLD_ITER}+ iterations have passed "
            f"with no dispatch."
        ),
        fires_on="pre_iter",
        predicate=_detect_no_dispatch_after_threshold,
        once_per_session=True,
        priority=20,
    )
