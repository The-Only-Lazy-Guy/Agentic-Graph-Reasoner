"""Standard meta-procedures shipped with the Phase-3A substrate.

Each module exports a `build_*()` factory returning a MetaProcedure.
The `build_default_meta_pool()` factory assembles the five into one
pool that the reasoning_loop uses by default when no custom pool is
provided.
"""
from reasoning.meta import MetaPool, MetaProcedure
from reasoning.meta_procedures.budget_warner import build_budget_warner
from reasoning.meta_procedures.contradiction_detector import build_contradiction_detector
from reasoning.meta_procedures.cycle_detector import build_cycle_detector
from reasoning.meta_procedures.dispatch_miss_nudge import build_dispatch_miss_nudge
from reasoning.meta_procedures.no_dispatch_after_threshold import build_no_dispatch_after_threshold


def build_default_meta_pool() -> MetaPool:
    """Construct the default Phase-3A meta-procedure pool.

    Order does not matter for correctness (each MP fires on its
    designated hook); listed by priority for predictability:
      - pre_iter:  BudgetWarner, NoDispatchAfterThreshold
      - post_dispatch: CycleDetector, ContradictionDetector, DispatchMissNudge
    """
    pool = MetaPool()
    pool.register(build_budget_warner())
    pool.register(build_no_dispatch_after_threshold())
    pool.register(build_cycle_detector())
    pool.register(build_contradiction_detector())
    pool.register(build_dispatch_miss_nudge())
    return pool


__all__ = [
    "build_default_meta_pool",
    "build_budget_warner",
    "build_contradiction_detector",
    "build_cycle_detector",
    "build_dispatch_miss_nudge",
    "build_no_dispatch_after_threshold",
]
