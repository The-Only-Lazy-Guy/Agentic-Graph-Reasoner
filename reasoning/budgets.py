"""Budget enforcement for reasoning episodes.

Without budgets, the reasoning loop spirals: composition fans out
unboundedly, recursive procedure calls never terminate, retrieval
hops chain forever. **Bounded cognition makes systems more
intelligent, not less.** This module enforces six hard caps.

When a budget is exhausted, the reasoning loop catches BudgetExhausted,
injects a <budget_exhausted> signal into context, and lets the reasoner
produce a final answer with what it has — graceful degradation rather
than crash.

See PHASE1_PLAN.md §6 / REASONING_ARCHITECTURE.md §4.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Literal

# Names of all budget categories. Used as the `op` argument to consume().
BudgetOp = Literal[
    "llm_call",          # one LLM forward pass
    "hop",               # one graph traversal hop (e.g., follow neighbors)
    "subgraph_size",     # one new node added to session subgraph
    "fan_out",           # one procedure call within a single reasoning step
    "tokens",            # output tokens generated (estimate)
]


@dataclass
class Budgets:
    """Hard caps for one reasoning episode.

    Defaults from the resolved decisions table (PHASE1_PLAN.md §11):
      - max_llm_calls: 3   (default LLM call budget per query)
      - everything else: reasonable starting values, tunable.
    """
    max_llm_calls: int = 3
    max_hops: int = 1
    max_recursion_depth: int = 4
    max_session_subgraph_size: int = 50
    max_composition_fan_out: int = 5
    max_total_tokens: int = 2048

    def cap_for(self, op: str) -> int:
        return {
            "llm_call": self.max_llm_calls,
            "hop": self.max_hops,
            "subgraph_size": self.max_session_subgraph_size,
            "fan_out": self.max_composition_fan_out,
            "tokens": self.max_total_tokens,
        }[op]


class BudgetExhausted(Exception):
    """Raised when a consume() call would exceed a budget.

    The reasoning loop is expected to catch this, mark the session as
    budget-bounded in metadata, inject a <budget_exhausted> hint into
    the prompt for the final turn, and finalize the answer.
    """

    def __init__(self, op: str, requested: int, used: int, cap: int):
        super().__init__(
            f"BudgetExhausted({op}): requested {requested} more, "
            f"already used {used}/{cap}"
        )
        self.op = op
        self.requested = requested
        self.used = used
        self.cap = cap


class BudgetTracker:
    """Stateful tracker. One per reasoning episode.

    Counters and the recursion depth stack live here. The reasoning
    loop calls .consume() before each tracked operation.
    """

    def __init__(self, budgets: Budgets):
        self.budgets = budgets
        self.used: Dict[str, int] = {
            "llm_call": 0,
            "hop": 0,
            "subgraph_size": 0,
            "fan_out": 0,
            "tokens": 0,
        }
        self.recursion_depth = 0
        self.fan_out_this_step = 0
        self._fan_out_step: int = -1                # which step we're tracking

    # ---- check vs consume ----------------------------------------------- #

    def check(self, op: BudgetOp, amount: int = 1) -> bool:
        """Return True if `amount` more of `op` is allowed. Non-destructive."""
        if op == "fan_out":
            return (self.fan_out_this_step + amount) <= self.budgets.max_composition_fan_out
        return (self.used[op] + amount) <= self.budgets.cap_for(op)

    def consume(self, op: BudgetOp, amount: int = 1) -> None:
        """Record consumption. Raises BudgetExhausted if over the cap."""
        if op == "fan_out":
            new = self.fan_out_this_step + amount
            cap = self.budgets.max_composition_fan_out
            if new > cap:
                raise BudgetExhausted(op, amount, self.fan_out_this_step, cap)
            self.fan_out_this_step = new
            return
        new = self.used[op] + amount
        cap = self.budgets.cap_for(op)
        if new > cap:
            raise BudgetExhausted(op, amount, self.used[op], cap)
        self.used[op] = new

    # ---- per-step fan_out reset ---------------------------------------- #

    def on_step_change(self, new_step_index: int) -> None:
        """Called by the reasoning loop when stepping to a new iteration.
        Resets the per-step fan-out counter so each reasoning step gets
        its own composition budget."""
        if new_step_index != self._fan_out_step:
            self.fan_out_this_step = 0
            self._fan_out_step = new_step_index

    # ---- recursion management ------------------------------------------ #

    def push_recursion(self) -> None:
        """Enter a procedure-within-procedure call. Raises if depth would exceed cap."""
        new = self.recursion_depth + 1
        cap = self.budgets.max_recursion_depth
        if new > cap:
            raise BudgetExhausted("recursion", 1, self.recursion_depth, cap)
        self.recursion_depth = new

    def pop_recursion(self) -> None:
        """Exit a procedure-within-procedure call."""
        if self.recursion_depth <= 0:
            return
        self.recursion_depth -= 1

    # ---- summary -------------------------------------------------------- #

    def summary(self) -> Dict[str, Any]:
        """Snapshot of usage vs caps. Goes into session metadata at close."""
        return {
            "llm_calls": {"used": self.used["llm_call"], "cap": self.budgets.max_llm_calls},
            "hops": {"used": self.used["hop"], "cap": self.budgets.max_hops},
            "subgraph_size": {"used": self.used["subgraph_size"], "cap": self.budgets.max_session_subgraph_size},
            "fan_out_max_per_step": {"used": self.fan_out_this_step, "cap": self.budgets.max_composition_fan_out},
            "tokens": {"used": self.used["tokens"], "cap": self.budgets.max_total_tokens},
            "recursion_depth_now": self.recursion_depth,
            "recursion_depth_cap": self.budgets.max_recursion_depth,
        }
