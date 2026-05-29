"""BudgetWarner — INFO when any budget axis crosses a threshold.

Fires on `pre_iter`. Checks each of (llm_call, hop, subgraph_size,
fan_out, tokens) against its cap. If usage is at or above
BUDGET_THRESHOLD (default 0.75), emits one INFO signal per crossed
axis with the current usage / cap.

Not once_per_session: each iteration that the threshold holds gets
a fresh signal (with a per-iteration id) so the model sees the
warning while it's still relevant. Signal id includes iteration
number; otherwise stream dedupe would silently drop later emissions
after the first.
"""
from __future__ import annotations

from typing import List

from reasoning.meta import MetaContext, MetaProcedure
from reasoning.signals import Signal


BUDGET_THRESHOLD = 0.75


# (op_key_in_BudgetTracker.used, cap_attr_on_Budgets, human_readable_name)
_AXES = [
    ("llm_call", "max_llm_calls", "LLM calls"),
    ("hop", "max_hops", "graph traversal hops"),
    ("subgraph_size", "max_session_subgraph_size", "session subgraph nodes"),
    ("tokens", "max_total_tokens", "output tokens"),
]


def _detect_budget_pressure(ctx: MetaContext) -> List[Signal]:
    signals: List[Signal] = []
    used = ctx.budget.used
    budgets = ctx.budget.budgets

    for op_key, cap_attr, human in _AXES:
        cap = getattr(budgets, cap_attr, 0)
        current = used.get(op_key, 0)
        if cap <= 0:
            continue
        ratio = current / cap
        if ratio < BUDGET_THRESHOLD:
            continue

        pct = int(round(ratio * 100))
        signals.append(Signal(
            id=f"budget_warning_{op_key}_iter_{ctx.current_iteration}",
            type="budget_at_threshold",
            severity="info",
            message=(
                f"{human}: {current}/{cap} ({pct}%) — approaching the cap. "
                f"Consider wrapping up reasoning soon."
            ),
            emitted_at_step=ctx.current_iteration,
            emitted_by="budget_warner",
            metadata={"axis": op_key, "used": current, "cap": cap, "pct": pct},
        ))
    return signals


def build_budget_warner() -> MetaProcedure:
    return MetaProcedure(
        id="meta_budget_warner",
        name="BudgetWarner",
        purpose=(
            f"Emit INFO when any budget axis reaches {int(BUDGET_THRESHOLD*100)}% "
            f"of its cap."
        ),
        fires_on="pre_iter",
        predicate=_detect_budget_pressure,
        once_per_session=False,
        priority=10,
    )
