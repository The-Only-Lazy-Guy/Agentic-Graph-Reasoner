"""Meta-procedures — trigger-action rules running outside the LLM.

Phase 3A. A MetaProcedure observes substrate state via a deterministic
Python predicate and reacts by emitting Signals or mutating substrate.
The LLM is never invoked from a meta-procedure; meta-cognition cost is
~0 LLM tokens. The model still REACTS to meta-procedures' effects via
the signal injection in its normal next-iteration prompt.

Three responsibilities live in this module:
  1. `MetaProcedure` schema (trigger + optional action + hook + debounce)
  2. `MetaContext` snapshot type passed to predicates and actions
  3. `MetaPool` runtime that orchestrates the hook firing

See PHASE3A_PLAN.md §3 and §4 for the design rationale.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Set

from reasoning.signals import Signal, SignalHook, signal_dedupe_key

if TYPE_CHECKING:
    # Heavy imports only at type-check time — keeps `meta.py` light.
    from reasoning.budgets import BudgetTracker
    from reasoning.dispatcher import DispatchOutcome
    from reasoning.session_subgraph import SessionSubgraphController


_log = logging.getLogger(__name__)


# ---- Context snapshot ------------------------------------------------- #

@dataclass
class MetaContext:
    """Read-mostly snapshot of substrate state passed to predicates / actions.

    Predicates MUST treat this as read-only. Actions MAY mutate via
    `session` (the SessionSubgraphController) but MUST NOT call the LLM.

    IMPORTANT: within a single hook tick, meta-procedures are isolated.
    The `previous_signals` field is a snapshot taken BEFORE the hook
    runs — so MPs that fire later in the same tick do NOT see signals
    emitted by MPs that fired earlier in that same tick. They will see
    each other's signals on the NEXT hook tick. This is by design:
    keeps the hook deterministic and order-independent. If you need
    one MP to react to another's signal on the same turn, design the
    second MP for a LATER hook (e.g., post_dispatch after a pre_iter MP).
    """
    session: "SessionSubgraphController"
    budget: "BudgetTracker"
    dispatch_outcomes: List["DispatchOutcome"]
    raw_outputs: List[str]
    anchor_ids: List[str]
    current_iteration: int
    # Signals emitted earlier in the same session (across all hooks).
    # Useful for predicates that want to avoid re-flagging an already-
    # surfaced issue, beyond the built-in `once` debounce.
    previous_signals: List[Signal] = field(default_factory=list)
    # Names of procedures available to the dispatcher (snapshot at MetaContext
    # construction). Used by predicates like DispatchMissNudge to check
    # whether a model-mentioned procedure name is known.
    procedure_names: List[str] = field(default_factory=list)
    # Phase 13c: tool call log from JSON-tool-loop answerers (e.g., v4). Each
    # entry is {"name": str, "args": dict, "result_summary": str}. Empty when
    # observed from a free-text-dispatcher answerer.
    tool_call_log: List[Dict[str, Any]] = field(default_factory=list)

    @classmethod
    def for_tool_loop(
        cls,
        *,
        session: "SessionSubgraphController",
        budget: "BudgetTracker",
        current_iteration: int,
        raw_outputs: Optional[List[str]] = None,
        anchor_ids: Optional[List[str]] = None,
        previous_signals: Optional[List[Signal]] = None,
        tool_call_log: Optional[List[Dict[str, Any]]] = None,
    ) -> "MetaContext":
        """Construct a MetaContext for a JSON-tool-loop answerer (e.g., v4).

        These answerers don't run the Phase-1/2A free-text dispatcher, so
        `dispatch_outcomes` and `procedure_names` are empty by default.
        Meta-procedures that depend on dispatcher state (e.g., CycleDetector,
        ContradictionDetector) will simply emit nothing — which is the
        correct behavior when there's no dispatcher to observe.

        `tool_call_log` lets v4-aware MPs (e.g., ToolLoopCycleDetector,
        ExcessiveSearchDetector) inspect the actual JSON tool calls.
        """
        return cls(
            session=session,
            budget=budget,
            dispatch_outcomes=[],
            raw_outputs=raw_outputs or [],
            anchor_ids=anchor_ids or [],
            current_iteration=current_iteration,
            previous_signals=previous_signals or [],
            procedure_names=[],
            tool_call_log=list(tool_call_log or []),
        )


# Signature types for predicate and action callables.
PredicateFn = Callable[[MetaContext], List[Signal]]
ActionFn = Callable[[MetaContext, List[Signal]], None]


@dataclass
class MetaProcedure:
    """One trigger-action rule.

    The predicate is run on the configured hook(s); when it returns a
    non-empty signal list, the optional action runs, and the signals
    are added to the session-wide signal stream.
    """
    id: str
    name: str
    purpose: str                                  # one-line documentation
    fires_on: SignalHook
    predicate: PredicateFn
    action: Optional[ActionFn] = None
    # once_per_session: if True, signals whose (type, sorted(related_node_ids))
    # tuple already fired in this session are suppressed. Useful for things
    # like "contradiction between A and B" that shouldn't re-fire just
    # because the model didn't address them yet.
    once_per_session: bool = False
    # priority (lower = earlier in the iteration order). Used only as a
    # tiebreaker when multiple meta-procedures fire on the same hook;
    # signal ordering in the prompt is still by severity.
    priority: int = 100


# ---- Runtime pool ---------------------------------------------------- #

class MetaPool:
    """Registers MetaProcedures and orchestrates their hook firing.

    One instance per reasoning episode. Constructed by `run_reasoning`,
    populated from a default registry, fed a MetaContext on each hook
    tick. Maintains the per-session dedupe set for once_per_session
    meta-procedures.
    """

    def __init__(self, procedures: Optional[List[MetaProcedure]] = None):
        self._procedures: List[MetaProcedure] = list(procedures or [])
        # Dedupe ledger: (mp_id, signal_dedupe_key) -> True once seen.
        self._fired_dedupe: Set[tuple] = set()
        # Cumulative signal stream emitted so far this session.
        # Deduped by Signal.id: the FIRST emission of a given id wins.
        # Subsequent emissions with the same id are dropped from the
        # stream (and won't re-persist either). Matches the persistence
        # layer's "skip if id exists" semantics for consistency.
        self.signal_stream: List[Signal] = []
        self._seen_signal_ids: Set[str] = set()

    def register(self, mp: MetaProcedure) -> None:
        """Add a meta-procedure to the pool. Late registration is allowed."""
        self._procedures.append(mp)

    def procedures_for_hook(self, hook: SignalHook) -> List[MetaProcedure]:
        """All registered procedures that fire on `hook`, ordered by priority."""
        return sorted(
            [mp for mp in self._procedures if mp.fires_on == hook],
            key=lambda mp: (mp.priority, mp.id),
        )

    def run_hook(self, hook: SignalHook, context: MetaContext) -> List[Signal]:
        """Run all meta-procedures registered for `hook` against `context`.

        Returns the list of signals emitted during this hook tick (after
        dedupe). Signals are also appended to the cumulative
        `signal_stream` for the session.

        Exception handling: if a predicate raises, the failure is logged
        and the meta-procedure is skipped for this tick. Other meta-
        procedures still run. This keeps the meta layer fault-tolerant.
        """
        emitted: List[Signal] = []
        for mp in self.procedures_for_hook(hook):
            try:
                candidate_signals = mp.predicate(context)
            except Exception as exc:                  # noqa: BLE001
                _log.warning(
                    "meta-procedure %r predicate raised %s: %s",
                    mp.id, type(exc).__name__, exc,
                )
                continue
            if not candidate_signals:
                continue

            # Apply once_per_session debounce
            accepted: List[Signal] = []
            for sig in candidate_signals:
                if mp.once_per_session or sig.once:
                    key = (mp.id, signal_dedupe_key(sig))
                    if key in self._fired_dedupe:
                        continue
                    self._fired_dedupe.add(key)
                accepted.append(sig)

            if not accepted:
                continue

            # Stream-level dedupe: drop signals whose id has already been
            # emitted in this session. Keeps the in-memory stream consistent
            # with the persistence layer (which silently skips re-persists
            # on existing ids). If you want to emit a "fresh" signal for
            # the same root cause, give it a new id.
            stream_accepted = [s for s in accepted if s.id not in self._seen_signal_ids]
            for s in stream_accepted:
                self._seen_signal_ids.add(s.id)

            # Optional action runs only on the accepted (post-debounce) set.
            # We pass the dedup-filtered list so actions see the same view
            # as the stream.
            if mp.action is not None:
                try:
                    mp.action(context, stream_accepted)
                except Exception as exc:              # noqa: BLE001
                    _log.warning(
                        "meta-procedure %r action raised %s: %s",
                        mp.id, type(exc).__name__, exc,
                    )
                    # The signals were still validly emitted; action
                    # failure doesn't cancel them.

            emitted.extend(stream_accepted)
            self.signal_stream.extend(stream_accepted)

        return emitted

    # ---- introspection / debug ---------------------------------------- #

    def all_procedures(self) -> List[MetaProcedure]:
        return list(self._procedures)

    def fired_dedupe_keys(self) -> Set[tuple]:
        return set(self._fired_dedupe)
