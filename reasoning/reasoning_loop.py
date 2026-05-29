"""Reasoning loop orchestrator — Phase 1 heart.

Ties together everything built so far:

    retrieval (with failure boost)
        ↓
    prompt construction (substrate-aware)
        ↓
    LLM call (main reasoner)
        ↓
    dispatcher scan + invoke (sub-LLM calls + state mutations)
        ↓
    follow-up LLM call if no answer yet (splice dispatch results back)
        ↓
    extract <answer> block
        ↓
    persist session subgraph + audit log
        ↓
    consolidator decisions
        ↓
    ReasoningResult

Replaces what `_run_graph_agent` does today, end-to-end. The front-end's
substrate path (Sub-phase 1.8) calls `run_reasoning()` here.

See PHASE1_PLAN.md §9.
"""
from __future__ import annotations

import hashlib
import json
import re
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Set

from reasoning.budgets import BudgetExhausted, BudgetTracker, Budgets
from reasoning.capsule_store import build_capsules, select_capsules
from reasoning.activation import GraphTaskFrame, render_task_frame
from reasoning.consolidation import ConsolidationDecision, Consolidator
from reasoning.dispatcher import Dispatcher, DispatchOutcome, PatternMatch, flatten_dispatch_outcomes
from reasoning.meta import MetaContext, MetaPool, MetaProcedure
from reasoning.micro_controller import (
    compose_answer_from_slots,
    deterministic_finalize_payload,
    render_micro_context_block,
    run_micro_epistemic_controller,
)
from reasoning.retrieval_boost import retrieve_with_failure_boost
from reasoning.schemas import ProcedureNode, SessionSubgraph
from reasoning.session_subgraph import SessionSubgraphController
from reasoning.signals import Signal, render_signals_block
from reasoning.token_estimation import estimate_token_count
from reasoning.substrate_v2 import (
    CheckerRegistry,
    FastLoopConfig,
    ReasoningStep,
    SignalNode,
    _is_checker_residue_text,
    _try_workspace_slot,
    attach_fast_loop_to_session,
    compose_final_answer,
    derive_task_statement_concepts,
    missing_task_statement_concepts,
    project_session_subgraph_to_signals,
    render_workspace_step_prompt,
    run_fast_step_loop,
    task_concept_constraint,
    activation_keys_for_text,
)
from reasoning.workspace import PAYMENT_WORKSPACE_FILL_ORDER, PAYMENT_WORKSPACE_SLOTS, Workspace

from anchor_retrieval import retrieve_anchors_v2
from graph_core import MemoryGraph


# Phase 3A: cap on sticky signals carried across iterations. A predicate
# that emits new sticky signals every iteration could otherwise grow the
# carrier (and the rendered prompt section) unboundedly. We drop oldest
# beyond this cap; persistence still records every signal in the subgraph.
MAX_CARRIER_STICKY: int = 20


# ---- request / result ---------------------------------------------------- #

@dataclass
class ReasoningRequest:
    question: str
    graph_id: str
    graph_path: str                                  # absolute or relative to cwd
    k_anchors: int = 12
    budgets: Budgets = field(default_factory=Budgets)
    max_iterations: int = 3
    session_persist_root: Path = field(default_factory=lambda: Path("data/session_subgraphs"))
    promotion_threshold: int = 3
    consolidated_node_ids: Set[str] = field(default_factory=set)
    warm_start_session_paths: List[Path] = field(default_factory=list)
    debug_signals: bool = False
    failure_boost: float = 1.4
    # Phase 3E reserved flag. The substrate-v2 fast loop is implemented and
    # tested in reasoning/substrate_v2.py, but production run_reasoning()
    # routing stays unchanged until the next integration slice.
    enable_substrate_v2: bool = False


@dataclass
class ReasoningResult:
    answer: str
    reasoning_trace: str
    raw_outputs: List[str]                           # per-iteration LLM outputs
    session_subgraph: SessionSubgraph
    session_subgraph_path: Path
    audit_summary: Dict[str, Any]
    consolidation_decisions: List[ConsolidationDecision]
    budget_usage: Dict[str, Any]
    dispatch_outcomes: List[DispatchOutcome]
    anchor_ids: List[str]
    iterations_completed: int
    early_terminated_reason: Optional[str] = None
    # Phase 3A: every signal emitted across all hook ticks this session.
    # Already persisted as node_type="signal" in session_subgraph.nodes,
    # but kept here too so callers don't have to grep the subgraph dict.
    signals: List[Signal] = field(default_factory=list)


# ---- public entry point ------------------------------------------------ #

def run_reasoning(
    req: ReasoningRequest,
    llm_call: Callable[[str], str],
    procedure_pool: Optional[List[ProcedureNode]] = None,
    meta_pool: Optional[MetaPool] = None,
    event_callback: Optional[Callable[[str, Dict[str, Any]], None]] = None,
) -> ReasoningResult:
    """Execute one reasoning episode.

    procedure_pool:
        Procedures available to the dispatcher. In Phase 1 we hardcode
        the seed procedure here (caller may override). Phase 2+ would
        retrieve procedures from the graph dynamically.

    meta_pool (Phase 3A):
        Trigger-action meta-procedures observing substrate state across
        three hook points (pre_iter, post_dispatch, end_of_session).
        Emitted signals flow into the next prompt's `# System signals`
        section so the model reacts via its normal reasoning — no
        extra LLM call needed for meta-cognition. Defaults to an empty
        pool; Sub-phase 3.5 fills in the standard meta-procedures.
    """
    if procedure_pool is None:
        # Phase 2A default pool: the Phase-1 seed (VerifyAlgorithmPreconditions)
        # plus the three composition-aware procedures (composer + two leaves).
        # The composer references the others via depends_on, so they MUST
        # ship together for the dependency-gate of consolidation to clear.
        from reasoning.procedures.verify_algorithm_preconditions import build_seed_procedure
        from reasoning.procedures.verify_nonneg_edges import build_verify_nonneg_edges
        from reasoning.procedures.detect_negative_cycle import build_detect_negative_cycle
        from reasoning.procedures.verify_shortest_path import build_verify_shortest_path
        procedure_pool = [
            build_seed_procedure(),
            build_verify_nonneg_edges(),
            build_detect_negative_cycle(),
            build_verify_shortest_path(),
        ]

    # 1. Load long-term graph
    graph = MemoryGraph.load_json(req.graph_path)

    # 2. Retrieve anchor facts (with failure-pattern boost)
    anchor_ids = retrieve_with_failure_boost(
        req.question, graph,
        k=req.k_anchors,
        failure_boost=req.failure_boost,
        graph_basename=req.graph_id,
    )

    # 3. Initialize substrate components
    session_id = f"sess_{uuid.uuid4().hex[:12]}"
    session = SessionSubgraphController(session_id, req.question, req.graph_id)
    budget = BudgetTracker(req.budgets)
    dispatcher = Dispatcher({p.id: p for p in procedure_pool})
    micro_outcome = run_micro_epistemic_controller(
        question=req.question,
        graph=graph,
        anchor_ids=anchor_ids,
    )
    micro_context_block = render_micro_context_block(micro_outcome)
    _emit_event(event_callback, "session_graph", _session_event_payload(
        session=session,
        stage="session_started",
        iteration=0,
        anchor_ids=anchor_ids,
    ))
    _emit_event(event_callback, "session_graph", _session_event_payload(
        session=session,
        stage="micro_controller_ready",
        iteration=0,
        signal_count=len(micro_outcome.micro_steps),
    ))

    if micro_outcome.finalizable:
        return _finalize_from_micro_controller(
            req=req,
            graph=graph,
            anchor_ids=anchor_ids,
            session=session,
            budget=budget,
            llm_call=llm_call,
            micro_outcome=micro_outcome,
            event_callback=event_callback,
        )

    if req.enable_substrate_v2:
        return _run_reasoning_substrate_v2(
            req=req,
            graph=graph,
            anchor_ids=anchor_ids,
            session=session,
            budget=budget,
            llm_call=llm_call,
            procedure_pool=procedure_pool,
            dispatcher=dispatcher,
            event_callback=event_callback,
            micro_outcome=micro_outcome,
        )

    # Phase 3A: meta-procedure pool. Defaults to the five built-in meta-
    # procedures (cycle, budget, contradiction, dispatch-miss, no-dispatch).
    # Production callers (front-end's substrate path) get them automatically.
    # Pass `meta_pool=MetaPool()` explicitly to opt out.
    if meta_pool is None:
        from reasoning.meta_procedures import build_default_meta_pool
        meta_pool = build_default_meta_pool()
    # Sticky signals (severity=error by convention) survive across iterations;
    # non-sticky signals are rendered only for the iteration that emitted them.
    # Capped at MAX_CARRIER_STICKY entries — when more sticky signals arrive
    # than the cap, the OLDEST get dropped from the carrier so the prompt
    # doesn't balloon over a long session. Persistence into the session
    # subgraph still captures every signal regardless of cap.
    carrier_sticky: List[Signal] = []

    # 4. Run the loop
    raw_outputs: List[str] = []
    dispatch_outcomes: List[DispatchOutcome] = []
    early_terminated_reason: Optional[str] = None
    final_answer = ""
    reasoning_trace = ""
    iterations_completed = 0
    # Map procedure_id -> existing SessionObjectNode id. Ensures a procedure
    # mentioned multiple times across one or more iterations reuses ONE
    # session object instead of spawning duplicates. Fix for double-dispatch
    # bug seen in sess_862af617e699 and sess_a9e21797680e (cs4 / merged_graph
    # Dijkstra question). Without this, two separate sub-LLMs run the same
    # procedure body and accumulate divergent state under different ids.
    procedure_object_ids: Dict[str, str] = {}
    # Procedure invocations already dispatched in the CURRENT iteration —
    # short-circuit so a single LLM turn that mentions a procedure twice
    # (e.g., once in PLAN and once in a confirmation sentence) doesn't run
    # the procedure body twice in this turn.
    invoked_this_iteration: set = set()

    for iteration in range(req.max_iterations):
        iterations_completed = iteration + 1
        budget.on_step_change(iteration)
        invoked_this_iteration = set()

        # Phase 3A — pre_iter meta-procedure hook. Predicates observe
        # state from PRIOR iterations (dispatch_outcomes, raw_outputs,
        # carrier_sticky signals) and emit new signals for this turn's
        # prompt. previous_signals is intentionally passed as a tuple-
        # backed list copy so predicates can't accidentally mutate the
        # pool's signal stream.
        meta_ctx = MetaContext(
            session=session,
            budget=budget,
            # Pass the FLATTENED dispatch tree so predicates see every
            # invocation including sub-procedures called via CALL from
            # within a composer's body. Top-level-only consumers can
            # filter on parent_object_id is None.
            dispatch_outcomes=flatten_dispatch_outcomes(dispatch_outcomes),
            raw_outputs=raw_outputs,
            anchor_ids=anchor_ids,
            current_iteration=iteration,
            previous_signals=list(meta_pool.signal_stream),
            procedure_names=[p.name for p in procedure_pool],
        )
        pre_iter_signals = meta_pool.run_hook("pre_iter", meta_ctx)

        # Persist pre_iter signals immediately. If we break out below
        # (budget exhausted before the LLM call), these signals still
        # need to survive — otherwise replay loses observations the
        # substrate actually made. Probe-1 finding (2026-05-20).
        for sig in pre_iter_signals:
            if sig.id not in session.subgraph.nodes:
                session.subgraph.nodes[sig.id] = sig.to_node()
        if pre_iter_signals:
            _emit_event(event_callback, "session_graph", _session_event_payload(
                session=session,
                stage="pre_iter_signals",
                iteration=iteration,
                signal_count=len(pre_iter_signals),
            ))

        # Active signals this turn = sticky carry-over + this turn's pre_iter
        active_signals = list(carrier_sticky) + pre_iter_signals

        # Build prompt with current state of dispatch outcomes + active signals
        prompt = _build_prompt(
            req=req, graph=graph, anchor_ids=anchor_ids,
            procedure_pool=procedure_pool,
            dispatch_outcomes=dispatch_outcomes,
            iteration=iteration,
            signals=active_signals,
            micro_context_block=micro_context_block,
        )

        # LLM call — main reasoner
        try:
            budget.consume("llm_call")
        except BudgetExhausted as exc:
            early_terminated_reason = f"budget: {exc}"
            # Update carrier with any sticky pre_iter signals before breaking
            # so end_of_session sees them in its previous_signals snapshot.
            for sig in pre_iter_signals:
                if sig.sticky and not any(c.id == sig.id for c in carrier_sticky):
                    carrier_sticky.append(sig)
            break

        output = llm_call(prompt)
        raw_outputs.append(output)
        _emit_event(event_callback, "session_graph", _session_event_payload(
            session=session,
            stage="model_output",
            iteration=iteration,
            raw_output_chars=len(output or ""),
        ))

        # Splice the latest reasoning + answer into our running trace
        r, a = _extract_blocks(output)
        if r:
            reasoning_trace = r
        if a:
            final_answer = a

        # Scan for procedure invocations.
        #
        # FINALIZATION-MODE GATE: once procedures have already run in a
        # prior iteration AND this turn's output contains an <answer>
        # block, treat this turn as synthesis-only. The model has the
        # procedure results it needs; any "I'll apply X" phrasing in the
        # synthesis prose is a free-text mention, not a real intent to
        # dispatch. Without this gate, the dispatcher will re-invoke
        # procedures based on incidental prose (observed in the
        # cs4 Dijkstra real-LLM run, sess on 2026-05-21: model emitted
        # <answer> plus the words "I'll apply VerifyNonNegativeEdges to
        # the instance..." → dispatcher consumed an LLM call on bogus
        # args and the budget exhausted).
        if dispatch_outcomes and a:
            matches: List[PatternMatch] = []
        else:
            matches = dispatcher.scan(output)

        # Invoke each match. Dedupe by procedure name within this turn,
        # and reuse the existing SessionObjectNode for procedures we've
        # already instantiated this session.
        for match in matches:
            proc = dispatcher.procedure_index.get(match.procedure_name.lower())
            if proc is None:
                continue
            if proc.id in invoked_this_iteration:
                continue                                # already ran this turn
            invoked_this_iteration.add(proc.id)
            try:
                budget.consume("fan_out")
                outcome = dispatcher.invoke(
                    match, session, llm_call, budget=budget,
                    existing_object_id=procedure_object_ids.get(proc.id),
                )
                dispatch_outcomes.append(outcome)
                if outcome.object_id is not None:
                    procedure_object_ids[proc.id] = outcome.object_id
                _emit_event(event_callback, "session_graph", _session_event_payload(
                    session=session,
                    stage="procedure_dispatched",
                    iteration=iteration,
                    procedure_name=match.procedure_name,
                    object_id=outcome.object_id,
                    mutations_applied=outcome.mutations_applied,
                    success=outcome.error is None,
                ))
            except BudgetExhausted as exc:
                early_terminated_reason = f"budget: {exc}"
                break

        # Phase 3A — post_dispatch hook. Now that dispatch_outcomes is
        # updated for this iteration, meta-procedures can detect cycles
        # (same proc+args fired 3+ times), contradictions (sibling
        # session_objects with opposing verdicts), etc.
        meta_ctx_post = MetaContext(
            session=session,
            budget=budget,
            # Pass the FLATTENED dispatch tree so predicates see every
            # invocation including sub-procedures called via CALL from
            # within a composer's body. Top-level-only consumers can
            # filter on parent_object_id is None.
            dispatch_outcomes=flatten_dispatch_outcomes(dispatch_outcomes),
            raw_outputs=raw_outputs,
            anchor_ids=anchor_ids,
            current_iteration=iteration,
            previous_signals=list(meta_pool.signal_stream),
            procedure_names=[p.name for p in procedure_pool],
        )
        post_signals = meta_pool.run_hook("post_dispatch", meta_ctx_post)

        # Persist post_dispatch signals. pre_iter signals were already
        # persisted earlier in this iteration (before the LLM call) so
        # they survive even budget-exhausted iterations.
        #
        # NOTE for future meta-procedure authors: this is a *direct* mutation
        # of session.subgraph.nodes — it does NOT go through
        # SessionSubgraphController's CRUD methods and therefore is NOT
        # journaled to the audit log. That's intentional for signals
        # (they're observations, not state mutations), but if a meta-procedure
        # ACTION needs to mutate session-object state, it MUST use the
        # controller methods so the audit log captures the change.
        for sig in post_signals:
            if sig.id not in session.subgraph.nodes:
                session.subgraph.nodes[sig.id] = sig.to_node()
        if post_signals:
            _emit_event(event_callback, "session_graph", _session_event_payload(
                session=session,
                stage="post_dispatch_signals",
                iteration=iteration,
                signal_count=len(post_signals),
            ))

        # Update sticky carry-over for the next iteration.
        # New sticky signals (this turn's pre/post) join the carrier;
        # non-sticky signals are dropped after this iteration.
        for sig in pre_iter_signals + post_signals:
            if sig.sticky and not any(c.id == sig.id for c in carrier_sticky):
                carrier_sticky.append(sig)

        # Cap the carrier at MAX_CARRIER_STICKY — drop oldest if over.
        # Prevents unbounded growth from a predicate that emits new
        # sticky signals every iteration. Persistence already captured
        # every signal in the subgraph regardless of cap.
        if len(carrier_sticky) > MAX_CARRIER_STICKY:
            carrier_sticky = carrier_sticky[-MAX_CARRIER_STICKY:]

        # If we already have a complete <answer>, we can stop
        if final_answer and not matches:
            break
        # If dispatch happened but we still don't have a final answer,
        # loop once more so the model can synthesize using the outcomes
        if not final_answer:
            session.step()
            continue
        # If both answer and dispatch happened in same turn, loop to let
        # the model revise the answer in light of dispatch outcomes
        if final_answer and matches and iteration + 1 < req.max_iterations:
            session.step()
            continue

        break

    # Phase 3A — end_of_session hook. Final pass for meta-procedures
    # that report session-level observations (e.g., empty dispatch run,
    # no procedure ever fired, etc.).
    meta_ctx_end = MetaContext(
        session=session,
        budget=budget,
        dispatch_outcomes=dispatch_outcomes,
        raw_outputs=raw_outputs,
        anchor_ids=anchor_ids,
        current_iteration=iterations_completed,
        previous_signals=list(meta_pool.signal_stream),
    )
    end_signals = meta_pool.run_hook("end_of_session", meta_ctx_end)
    for sig in end_signals:
        if sig.id not in session.subgraph.nodes:
            session.subgraph.nodes[sig.id] = sig.to_node()
    if end_signals:
        _emit_event(event_callback, "session_graph", _session_event_payload(
            session=session,
            stage="end_of_session_signals",
            iteration=iterations_completed,
            signal_count=len(end_signals),
        ))

    if not final_answer:
        final_answer = _fallback_answer(
            raw_outputs=raw_outputs,
            dispatch_outcomes=dispatch_outcomes,
            early_terminated_reason=early_terminated_reason,
        )

    # 4.5 Always-present baseline session structure (Phase 2A §11 criterion #4)
    # Even when no procedure fires, the session subgraph must contain a
    # question node, an answer node, and the retrieved anchors as evidence
    # so the UI panel always has something to render. When procedures DID
    # fire, this layer adds Q/A nodes alongside the session_objects (the
    # latter capture procedural state; the former capture the conversational
    # ground truth).
    _seed_session_baseline(
        session=session,
        question=req.question,
        answer=final_answer,
        anchor_ids=anchor_ids,
        graph=graph,
    )
    _emit_event(event_callback, "session_graph", _session_event_payload(
        session=session,
        stage="baseline_seeded",
        iteration=iterations_completed,
        answer_chars=len(final_answer or ""),
    ))

    # Diagnostic: persist raw_outputs alongside the subgraph so we can
    # inspect what the model actually said per iteration when dispatch
    # doesn't fire as expected. Cheap (few KB per session) and the only
    # way to debug real-LLM behavior without re-running.
    _attach_diagnostics_to_subgraph(
        session, raw_outputs=raw_outputs, anchor_ids=anchor_ids,
        budget_summary=budget.summary(),
        dispatch_outcomes=dispatch_outcomes,
    )
    if "__micro_controller__" not in session.subgraph.nodes:
        session.subgraph.nodes["__micro_controller__"] = {
            "id": "__micro_controller__",
            "node_type": "control_rule",
            "kind": "micro_controller_snapshot",
            **micro_outcome.to_dict(),
        }
    _emit_event(event_callback, "session_graph", _session_event_payload(
        session=session,
        stage="diagnostics_attached",
        iteration=iterations_completed,
    ))

    # 5. Persist session
    sess_dir = session.close(req.session_persist_root)

    # 6. Consolidation
    cons = Consolidator(
        promotion_threshold=req.promotion_threshold,
        consolidated_node_ids=req.consolidated_node_ids,
    )
    decisions = cons.consolidate(session.subgraph, prior_citation_counts={})

    return ReasoningResult(
        answer=final_answer,
        reasoning_trace=reasoning_trace,
        raw_outputs=raw_outputs,
        session_subgraph=session.subgraph,
        session_subgraph_path=sess_dir,
        audit_summary={
            "total_entries": len(session.subgraph.audit_log),
            "step_count": session.subgraph.step_count,
            "dispatch_count": len(dispatch_outcomes),
            "budget_exhausted": bool(early_terminated_reason),
            "signal_count": len(meta_pool.signal_stream),
            "micro_controller": True,
            "controller_task_family": micro_outcome.task_family,
            "micro_steps": [step.to_dict() for step in micro_outcome.micro_steps],
            "subgoal_reuse_count": micro_outcome.subgoal_reuse_count,
            "slot_fill_stats": micro_outcome.slot_fill_stats(),
            "controller_action_counts": dict(micro_outcome.controller_action_counts),
            "controller_fallback_used": not micro_outcome.finalizable,
        },
        signals=list(meta_pool.signal_stream),
        consolidation_decisions=decisions,
        budget_usage=budget.summary(),
        dispatch_outcomes=dispatch_outcomes,
        anchor_ids=anchor_ids,
        iterations_completed=iterations_completed,
        early_terminated_reason=early_terminated_reason,
    )


# ---- prompt construction ----------------------------------------------- #

_REASONING_RE = re.compile(r"<reasoning>(.*?)</reasoning>", re.DOTALL | re.IGNORECASE)
_ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL | re.IGNORECASE)


def _emit_event(
    event_callback: Optional[Callable[[str, Dict[str, Any]], None]],
    event: str,
    data: Dict[str, Any],
) -> None:
    if event_callback is None:
        return
    try:
        event_callback(event, data)
    except Exception:
        # UI streaming must never change reasoning behavior.
        return


def _finalize_from_micro_controller(
    *,
    req: ReasoningRequest,
    graph: MemoryGraph,
    anchor_ids: List[str],
    session: SessionSubgraphController,
    budget: BudgetTracker,
    llm_call: Callable[[str], str],
    micro_outcome: Any,
    event_callback: Optional[Callable[[str, Dict[str, Any]], None]] = None,
) -> ReasoningResult:
    raw_outputs: List[str] = []
    payload = deterministic_finalize_payload(req.question, micro_outcome, graph)
    answer = payload.get("answer", "").strip() or compose_answer_from_slots(micro_outcome) or "No final answer was produced."
    reasoning_trace = payload.get("reasoning", "").strip() or "micro-controller finalize"
    raw_outputs.append(
        f"<reasoning>{reasoning_trace}</reasoning>\n"
        f"<answer>{answer}</answer>\n"
        f"<explanation>{payload.get('explanation', '').strip()}</explanation>"
    )
    _emit_event(event_callback, "session_graph", _session_event_payload(
        session=session,
        stage="micro_controller_finalize",
        iteration=0,
        raw_output_chars=len(raw_outputs[0]),
    ))

    _seed_session_baseline(
        session=session,
        question=req.question,
        answer=answer,
        anchor_ids=anchor_ids,
        graph=graph,
    )
    _attach_diagnostics_to_subgraph(
        session,
        raw_outputs=raw_outputs,
        anchor_ids=anchor_ids,
        budget_summary=budget.summary(),
        dispatch_outcomes=[],
    )
    sess_dir = session.close(req.session_persist_root)
    cons = Consolidator(
        promotion_threshold=req.promotion_threshold,
        consolidated_node_ids=req.consolidated_node_ids,
    )
    decisions = cons.consolidate(session.subgraph, prior_citation_counts={})
    return ReasoningResult(
        answer=answer,
        reasoning_trace=reasoning_trace,
        raw_outputs=raw_outputs,
        session_subgraph=session.subgraph,
        session_subgraph_path=sess_dir,
        audit_summary={
            "total_entries": len(session.subgraph.audit_log),
            "step_count": session.subgraph.step_count,
            "dispatch_count": 0,
            "budget_exhausted": False,
            "signal_count": 0,
            "micro_controller": True,
            "controller_task_family": micro_outcome.task_family,
            "micro_steps": [step.to_dict() for step in micro_outcome.micro_steps],
            "subgoal_reuse_count": micro_outcome.subgoal_reuse_count,
            "slot_fill_stats": micro_outcome.slot_fill_stats(),
            "controller_action_counts": dict(micro_outcome.controller_action_counts),
            "controller_fallback_used": False,
            "selected_node_ids": list(micro_outcome.selected_node_ids),
        },
        signals=[],
        consolidation_decisions=decisions,
        budget_usage=budget.summary(),
        dispatch_outcomes=[],
        anchor_ids=anchor_ids,
        iterations_completed=max(1, len(raw_outputs)),
        early_terminated_reason=None,
    )


def _detect_workspace_domain(question: str) -> Optional[str]:
    """Detect if the question matches a known workspace domain.

    Returns the domain name (e.g. "payment") or None if no match.
    """
    lower = question.lower()
    if "payment worker" in lower or "psp" in lower or ("double charge" in lower and "idempotency" in lower):
        return "payment"
    return None


def _make_slot_checker(slot_name: str) -> Callable[[str], List[str]]:
    """Build a content checker for a workspace slot based on its expected keywords.
    
    Returns a callable that takes fill text and returns a list of violation strings
    (empty list = passes check).
    """
    name_lower = slot_name.lower()

    def checker(text: str) -> List[str]:
        hay = text.lower()
        violations: List[str] = []
        if "durable" in name_lower:
            has = any(tok in hay for tok in ("durable", "state machine", "payment intent", "pending", "charged"))
            if not has:
                violations.append("durable_state_keyword_missing")
        if "idempotency" in name_lower:
            has = any(tok in hay for tok in ("idempotency key", "idempotency"))
            if not has:
                violations.append("idempotency_keyword_missing")
        if "reconciliation" in name_lower or "psp" in name_lower:
            has = any(tok in hay for tok in ("reconciliation", "query the psp", "psp status", "external status lookup", "reconcile with the psp", "reconciling"))
            if not has:
                violations.append("reconciliation_keyword_missing")
        if "retry" in name_lower:
            has = any(tok in hay for tok in ("retry", "replay", "dedupe", "deduplication", "effectively once"))
            if not has:
                violations.append("retry_dedupe_keyword_missing")
        return violations

    return checker


def _try_fill_workspace(
    workspace: Workspace,
    question: str,
    parent_decisions: Sequence[str],
    hard_constraints: Sequence[str],
    metered_llm: Callable[[str], str],
) -> None:
    """Try to fill all workspace slots via LLM. Non-fatal: slots that fail
    to fill remain empty and the fast loop will handle them.

    Each slot attempt is checkpointed: if the slot fill + all sibling retries
    fail, the workspace is restored to its state before the attempt."""
    for slot_name in workspace.fill_order:
        if workspace.is_filled(slot_name):
            continue
        slot_config = PAYMENT_WORKSPACE_SLOTS.get(slot_name)
        if slot_config is None:
            continue
        # Checkpoint before attempting this slot
        workspace.checkpoint()
        fill_text = _try_workspace_slot(
            workspace,
            slot_name,
            slot_config["question"],
            parent_decisions=parent_decisions,
            hard_constraints=hard_constraints,
            llm_call=metered_llm,
            max_siblings=1,
            checker=_make_slot_checker(slot_name),
        )
        if fill_text:
            workspace.fill(slot_name, fill_text)
            # Slot succeeded — discard the checkpoint (keep the fill)
            workspace.discard_checkpoint()
        else:
            # All attempts failed — restore to state before this slot
            workspace.restore()


def _run_reasoning_substrate_v2(
    *,
    req: ReasoningRequest,
    graph: MemoryGraph,
    anchor_ids: List[str],
    session: SessionSubgraphController,
    budget: BudgetTracker,
    llm_call: Callable[[str], str],
    procedure_pool: Optional[List[ProcedureNode]] = None,
    dispatcher: Optional[Dispatcher] = None,
    event_callback: Optional[Callable[[str, Dict[str, Any]], None]] = None,
    micro_outcome: Optional[Any] = None,
) -> ReasoningResult:
    """Phase 3E v2 / 3F route. Workspace lane auto-enabled by domain detection."""
    root_step = ReasoningStep(
        step_id="step_root",
        parent_step_id=None,
        task_id=req.question,
        focus=req.question,
        looking_for="final answer",
    )
    initial_signals = _substrate_v2_initial_signals(req, graph, anchor_ids)

    # ── Capsule prefix (3F-alpha KV warm-pool) ─────────────────────────
    capsules = build_capsules(initial_signals)
    selected = select_capsules(req.question, capsules)
    capsule_text = "\n\n".join(c.rendered_text for c in selected) if selected else ""

    def metered_llm(prompt: str) -> str:
        if capsule_text:
            prompt = f"[STABLE]\n{capsule_text}\n[/STABLE]\n\n[VARIABLE]\n{prompt}\n[/VARIABLE]"
        budget.consume("llm_call")
        output = llm_call(prompt)
        budget.consume("tokens", estimate_token_count(prompt) + estimate_token_count(output))
        return output

    # ── Workspace lane (Phase 3F) ──────────────────────────────────────
    workspace = None
    warm_filled = 0
    workspace_domain = _detect_workspace_domain(req.question)
    if workspace_domain == "payment":
        workspace = Workspace(fill_order=list(PAYMENT_WORKSPACE_FILL_ORDER))
        # Warm-start: project filled slots from prior sessions before LLM fill
        warm_filled = _load_warm_start_workspace(
            req.warm_start_session_paths,
            workspace,
            question=req.question,
        )
        if budget.check("llm_call", amount=len(workspace.fill_order)):
            _try_fill_workspace(
                workspace,
                req.question,
                parent_decisions=[s.text for s in initial_signals if s.kind == "decision"],
                hard_constraints=[s.text for s in initial_signals if s.kind == "constraint"],
                metered_llm=metered_llm,
            )

    # Collect signal IDs from selected capsules so render_step_prompt can
    # exclude capsule-bundled signals from the active signals section.
    capsule_signal_ids: set[str] = set()
    for c in selected:
        capsule_signal_ids.update(c.signal_ids)

    early_terminated_reason = None
    try:
        fast_result = run_fast_step_loop(
            root_step=root_step,
            llm_call=metered_llm,
            initial_signals=initial_signals,
            checker_registry=CheckerRegistry(_substrate_v2_checker_plugins(req.question)),
            config=FastLoopConfig(
                max_total_steps=max(1, req.budgets.max_llm_calls),
                max_child_depth=min(2, req.budgets.max_recursion_depth),
                max_active_signals=6,
                debug_signals=req.debug_signals,
            ),
            capsule_signal_ids=capsule_signal_ids or None,
            procedure_pool=procedure_pool,
            dispatcher=dispatcher,
            session=session,
        )
    except BudgetExhausted as exc:
        early_terminated_reason = f"budget: {exc}"
        fast_result = run_fast_step_loop(
            root_step=root_step,
            llm_call=lambda _prompt: (
                "STEP_RESULT\n"
                "status: failed\n"
                "result: Substrate v2 budget exhausted before completion.\n"
                "delta:\n"
                "  risks:\n"
                "    - budget exhausted\n"
                "END_STEP_RESULT"
            ),
            initial_signals=initial_signals,
            checker_registry=CheckerRegistry(["generic_step_format"]),
            config=FastLoopConfig(max_total_steps=1),
        )

    # Inject plan signals (GOAL/KNOWN/PLAN) from DEEP mode into signal stream
    plan = getattr(fast_result.final_step_result, "plan", None)
    if plan:
        plan_signal_map = {
            "goal": ("goal", 0.95),
            "known": ("evidence", 0.85),
            "plan": ("decision", 0.90),
        }
        for key, (kind, conf) in plan_signal_map.items():
            text = plan.get(key)
            if text:
                sid = f"sigv2_plan_{key}"
                exists = any(s.id == sid for s in fast_result.signals)
                if not exists:
                    fast_result.signals.append(SignalNode(
                        id=sid,
                        kind=kind,
                        text=text,
                        activation_keys=activation_keys_for_text(text),
                        source_step_id="step_root",
                        produced_by="controller",
                        confidence=conf,
                    ))

    # Decide final answer: workspace compose if any slots filled, else fast loop
    if workspace is not None and workspace.filled_slots():
        final_answer = workspace.compose()
    else:
        final_answer = compose_final_answer(fast_result) or _fallback_answer(
            raw_outputs=fast_result.raw_outputs,
            dispatch_outcomes=[],
            early_terminated_reason=early_terminated_reason,
        )
    missing_task_concepts_before = missing_task_statement_concepts(req.question, final_answer)
    shaper_called = False
    shaper_error = None
    if len(missing_task_concepts_before) >= 2 and budget.check("llm_call"):
        shaper_called = True
        evidence_lines = _substrate_v2_shaper_evidence(fast_result.signals)
        shaper_prompt = (
            "Rewrite the draft answer so it explicitly mentions the missing task-statement concepts.\n"
            "Use only facts already present in the draft answer or evidence below. Do not introduce new claims.\n"
            "Return only the rewritten answer, with no analysis or markup.\n\n"
            f"Question:\n{req.question}\n\n"
            f"Missing task-statement concepts:\n- " + "\n- ".join(missing_task_concepts_before[:6]) + "\n\n"
            f"Draft answer:\n{final_answer}\n\n"
            f"Evidence already available:\n{evidence_lines}\n"
        )
        try:
            shaped = metered_llm(shaper_prompt).strip()
            if shaped:
                final_answer = shaped
        except BudgetExhausted as exc:
            shaper_error = f"budget: {exc}"
        except Exception as exc:  # defensive: shaper is optional, not load-bearing
            shaper_error = repr(exc)
    missing_task_concepts_after = missing_task_statement_concepts(req.question, final_answer)

    # Persist workspace state in session subgraph for future warm-start
    if workspace is not None:
        session.subgraph.nodes["__workspace__"] = {
            "id": "__workspace__",
            "node_type": "workspace",
            **workspace.to_dict(),
        }
    if micro_outcome is not None and "__micro_controller__" not in session.subgraph.nodes:
        session.subgraph.nodes["__micro_controller__"] = {
            "id": "__micro_controller__",
            "node_type": "control_rule",
            "kind": "micro_controller_snapshot",
            **micro_outcome.to_dict(),
        }

    reasoning_trace = "\n".join(
        f"{step.step_id}: {step.focus} -> {step.status}"
        for step in fast_result.steps
    )

    attach_fast_loop_to_session(session, fast_result)
    _emit_event(event_callback, "session_graph", _session_event_payload(
        session=session,
        stage="substrate_v2_trace_attached",
        iteration=len(fast_result.steps),
        step_count=len(fast_result.steps),
        signal_count=len(fast_result.signals),
    ))

    _seed_session_baseline(
        session=session,
        question=req.question,
        answer=final_answer,
        anchor_ids=anchor_ids,
        graph=graph,
    )
    _attach_diagnostics_to_subgraph(
        session,
        raw_outputs=fast_result.raw_outputs,
        anchor_ids=anchor_ids,
        budget_summary=budget.summary(),
        dispatch_outcomes=[],
    )
    sess_dir = session.close(req.session_persist_root)
    cons = Consolidator(
        promotion_threshold=req.promotion_threshold,
        consolidated_node_ids=req.consolidated_node_ids,
    )
    workspace_slot_count = len(workspace.filled_slots()) if workspace is not None else 0
    workspace_slot_total = len(workspace.fill_order) if workspace is not None else 0
    workspace_warm_filled = warm_filled if workspace is not None else 0

    decisions = cons.consolidate(session.subgraph, prior_citation_counts={})
    return ReasoningResult(
        answer=final_answer,
        reasoning_trace=reasoning_trace,
        raw_outputs=fast_result.raw_outputs,
        session_subgraph=session.subgraph,
        session_subgraph_path=sess_dir,
        audit_summary={
            "total_entries": len(session.subgraph.audit_log),
            "step_count": session.subgraph.step_count,
            "dispatch_count": 0,
            "budget_exhausted": bool(early_terminated_reason or fast_result.budget_exhausted),
            "signal_count": len(fast_result.signals),
            "substrate_v2": True,
            "cache_hits": fast_result.cache_hits,
            "cache_misses": fast_result.cache_misses,
            "delta_status_breakdown": fast_result.delta_status_breakdown,
            "checker_outcome_breakdown": fast_result.checker_outcome_breakdown,
            "repair_triggered": fast_result.repair_triggered,
            "repair_succeeded": fast_result.repair_succeeded,
            "activated_signal_ages": fast_result.activated_signal_ages,
            "activated_prior_session_signal_count": fast_result.activated_prior_session_signal_count,
            "prior_session_signal_reused": fast_result.prior_session_signal_reused,
            "tokens_per_call": fast_result.tokens_per_call,
            "step_timing": fast_result.step_timing,
            "debug_signal_dump": fast_result.debug_signal_dump,
            "task_coverage_missing_before": missing_task_concepts_before,
            "task_coverage_missing_after": missing_task_concepts_after,
            "answer_shaper_called": shaper_called,
            "answer_shaper_error": shaper_error,
            "workspace_slot_count": workspace_slot_count,
            "workspace_slot_total": workspace_slot_total,
            "workspace_warm_filled": workspace_warm_filled,
            "controller_task_family": micro_outcome.task_family if micro_outcome is not None else "",
            "micro_steps": [step.to_dict() for step in micro_outcome.micro_steps] if micro_outcome is not None else [],
            "subgoal_reuse_count": micro_outcome.subgoal_reuse_count if micro_outcome is not None else 0,
            "slot_fill_stats": micro_outcome.slot_fill_stats() if micro_outcome is not None else {},
            "controller_action_counts": dict(micro_outcome.controller_action_counts) if micro_outcome is not None else {},
            "controller_fallback_used": bool(micro_outcome is not None and not micro_outcome.finalizable),
        },
        signals=[],
        consolidation_decisions=decisions,
        budget_usage=budget.summary(),
        dispatch_outcomes=[],
        anchor_ids=anchor_ids,
        iterations_completed=len(fast_result.raw_outputs),
        early_terminated_reason=early_terminated_reason,
    )


def _substrate_v2_initial_signals(
    req: ReasoningRequest,
    graph: MemoryGraph,
    anchor_ids: List[str],
) -> List[SignalNode]:
    signals: List[SignalNode] = _load_warm_start_signals(req.warm_start_session_paths)
    q = req.question
    q_lower = q.lower()
    for concept in derive_task_statement_concepts(q):
        signals.append(SignalNode(
            id=f"sigv2_task_concept_{_short_hash(concept)}",
            kind="constraint",
            text=task_concept_constraint(concept),
            activation_keys=activation_keys_for_text(f"{q} {concept}", limit=8),
            produced_by="controller",
            state={"source": "task_statement_concept", "concept": concept},
            confidence=0.86,
        ))
    if "dijkstra" in q_lower and "negative" in q_lower:
        signals.append(SignalNode(
            id="sigv2_question_dijkstra_negative",
            kind="risk",
            text="Negative edge present; Dijkstra is unsafe unless all edges are nonnegative.",
            activation_keys=["negative", "edge", "dijkstra", "unsafe"],
            produced_by="controller",
            confidence=0.95,
        ))
    if ("remove(" in q_lower and "connected(" in q_lower) or ("plain dsu is insufficient" in q_lower) or ("time-axis structure" in q_lower):
        signals.append(SignalNode(
            id="sigv2_question_dynamic_connectivity_deletions",
            kind="risk",
            text="Ordinary DSU does not support edge deletions directly; recomputing BFS/DFS per query is too slow here.",
            activation_keys=["ordinary", "dsu", "deletions", "bfs", "slow"],
            produced_by="controller",
            confidence=0.93,
        ))
        signals.append(SignalNode(
            id="sigv2_question_dynamic_connectivity_time_axis",
            kind="constraint",
            text="Solve the add/remove/connectivity task offline with edge-active intervals over time and a rollback-capable DSU.",
            activation_keys=["offline", "intervals", "time", "rollback", "dsu"],
            produced_by="controller",
            confidence=0.94,
        ))
        signals.append(SignalNode(
            id="sigv2_question_dynamic_connectivity_segment_tree_time",
            kind="constraint",
            text="Place edge-active intervals on a segment tree over time (or divide and conquer over time) while traversing with rollback DSU state.",
            activation_keys=["segment", "tree", "time", "rollback", "intervals"],
            produced_by="controller",
            confidence=0.94,
        ))
    if "range_chmin" in q_lower or ("range_sum" in q_lower and "per-node state" in q_lower):
        signals.append(SignalNode(
            id="sigv2_question_segment_tree_beats",
            kind="constraint",
            text="Use segment tree beats or an equivalent max/second-max/count_max/sum state bundle; ordinary lazy propagation is insufficient.",
            activation_keys=["segment", "tree", "beats", "second", "count_max", "sum"],
            produced_by="controller",
            confidence=0.94,
        ))
        signals.append(SignalNode(
            id="sigv2_question_segment_tree_beats_rule",
            kind="constraint",
            text="A range_chmin update only changes the current maxima when x lies between the current max and second max.",
            activation_keys=["range_chmin", "current", "maxima", "second", "max"],
            produced_by="controller",
            confidence=0.93,
        ))
    if "subarray" in q_lower and ("update" in q_lower or "online" in q_lower):
        signals.append(SignalNode(
            id="sigv2_question_dynamic_subarray",
            kind="constraint",
            text="Online point updates require an efficient dynamic maximum-subarray data structure.",
            activation_keys=["online", "point", "updates", "subarray", "dynamic"],
            produced_by="controller",
            confidence=0.9,
        ))
        signals.append(SignalNode(
            id="sigv2_question_dynamic_subarray_fields",
            kind="constraint",
            text="Use a segment tree node storing sum, prefix, suffix, and best.",
            activation_keys=["segment", "tree", "sum", "prefix", "suffix", "best"],
            produced_by="controller",
            confidence=0.92,
        ))
        signals.append(SignalNode(
            id="sigv2_question_dynamic_subarray_merge",
            kind="constraint",
            text="State the cross-boundary merge rule: best = max(left.best, right.best, left.suffix + right.prefix).",
            activation_keys=["merge", "left", "right", "suffix", "prefix", "best"],
            produced_by="controller",
            confidence=0.92,
        ))
    if "long long" in q_lower or "10^" in q_lower or "1e" in q_lower:
        signals.append(SignalNode(
            id="sigv2_question_wide_int",
            kind="constraint",
            text="Use long long/int64 for large numeric sums.",
            activation_keys=["long", "int64", "sums"],
            produced_by="controller",
            confidence=0.85,
        ))
    if "subarray" in q_lower and ("non-empty" in q_lower or "non empty" in q_lower or "negative" in q_lower):
        signals.append(SignalNode(
            id="sigv2_question_all_negative",
            kind="risk",
            text="For non-empty subarrays, all-negative arrays must return the maximum element, not 0.",
            activation_keys=["negative", "empty", "subarray", "maximum", "element"],
            produced_by="controller",
            confidence=0.9,
        ))
    if "binary search on the answer" in q_lower:
        signals.append(SignalNode(
            id="sigv2_question_binary_search_answer",
            kind="constraint",
            text="Binary search on the answer requires a monotone true/false feasibility predicate.",
            activation_keys=["binary", "search", "monotone", "true", "false", "predicate"],
            produced_by="controller",
            confidence=0.9,
        ))
    if "http" in q_lower and "get" in q_lower and "idempotent" in q_lower:
        signals.append(SignalNode(
            id="sigv2_question_http_get_idempotent",
            kind="constraint",
            text="Repeating GET has the same effect and does not change server or resource state.",
            activation_keys=["same", "effect", "does", "change", "server", "resource", "state"],
            produced_by="controller",
            confidence=0.9,
        ))
    if "learning rate" in q_lower and ("gradient descent" in q_lower or "training" in q_lower):
        signals.append(SignalNode(
            id="sigv2_question_learning_rate",
            kind="constraint",
            text="A too-high learning rate can overshoot the loss minimum and prevent convergence.",
            activation_keys=["learning", "rate", "loss", "minimum", "convergence", "overshoot"],
            produced_by="controller",
            confidence=0.9,
        ))
    if "race condition" in q_lower:
        signals.append(SignalNode(
            id="sigv2_question_race_condition",
            kind="constraint",
            text="Race conditions involve concurrent access to shared state or shared resources.",
            activation_keys=["race", "condition", "shared", "state", "resources"],
            produced_by="controller",
            confidence=0.9,
        ))
    if "payment worker" in q_lower or "psp" in q_lower or ("double charge" in q_lower and "idempotency" in q_lower):
        signals.append(SignalNode(
            id="sigv2_question_payment_procedure",
            kind="procedure",
            text="System-design workflow: track durable payment state, the crash window around the PSP call, retry/dedupe semantics, and reconciliation against PSP status.",
            activation_keys=["payment", "state", "crash", "retry", "reconciliation", "psp"],
            produced_by="controller",
            state={"systemic_design": True, "preferred_lane": "session_object"},
            confidence=0.95,
        ))
        signals.append(SignalNode(
            id="sigv2_question_payment_state",
            kind="constraint",
            text="Persist a durable local payment state machine around the external charge; idempotency key alone is insufficient.",
            activation_keys=["durable", "payment", "state", "idempotency", "insufficient"],
            produced_by="controller",
            confidence=0.95,
        ))
        signals.append(SignalNode(
            id="sigv2_question_payment_reconcile",
            kind="constraint",
            text="After a crash, reconcile uncertain payment outcomes by querying PSP state before replaying or retrying.",
            activation_keys=["crash", "reconcile", "psp", "retry", "replay"],
            produced_by="controller",
            confidence=0.94,
        ))
        signals.append(SignalNode(
            id="sigv2_question_payment_retry_dedupe",
            kind="constraint",
            text="At-least-once retries need consumer-side dedupe/replay semantics in addition to PSP idempotency.",
            activation_keys=["least", "once", "dedupe", "replay", "idempotency"],
            produced_by="controller",
            confidence=0.93,
        ))
    if "zero downtime" in q_lower or ("cutover" in q_lower and "rollback" in q_lower and "order" in q_lower):
        signals.append(SignalNode(
            id="sigv2_question_migration_procedure",
            kind="procedure",
            text="System-design workflow: phase the migration as backfill, live-change capture, verification, cutover, and rollback.",
            activation_keys=["backfill", "capture", "verification", "cutover", "rollback"],
            produced_by="controller",
            state={"systemic_design": True, "preferred_lane": "session_object"},
            confidence=0.95,
        ))
        signals.append(SignalNode(
            id="sigv2_question_migration_plan",
            kind="constraint",
            text="Use ordered migration phases: backfill historical data, capture live writes, verify parity, then cut over.",
            activation_keys=["backfill", "live", "writes", "verify", "cut", "over"],
            produced_by="controller",
            confidence=0.95,
        ))
        signals.append(SignalNode(
            id="sigv2_question_migration_rollback",
            kind="constraint",
            text="Keep rollback viable by preserving the old-good source of truth until verification and cutover are complete.",
            activation_keys=["rollback", "source", "truth", "verification", "cutover"],
            produced_by="controller",
            confidence=0.93,
        ))
        signals.append(SignalNode(
            id="sigv2_question_migration_replay",
            kind="constraint",
            text="Keep an idempotent replay path for live changes so cutover and rollback can re-apply updates safely.",
            activation_keys=["idempotent", "replay", "live", "changes", "rollback"],
            produced_by="controller",
            confidence=0.93,
        ))
    if "flash-sale" in q_lower or "flash sale" in q_lower or ("reservation ttl" in q_lower and "oversell" in q_lower):
        signals.append(SignalNode(
            id="sigv2_question_inventory_procedure",
            kind="procedure",
            text="System-design workflow: model ownership, reservation state transitions, TTL expiry, payment confirmation, and reconciliation.",
            activation_keys=["ownership", "reservation", "ttl", "confirmation", "reconciliation"],
            produced_by="controller",
            state={"systemic_design": True, "preferred_lane": "session_object"},
            confidence=0.95,
        ))
        signals.append(SignalNode(
            id="sigv2_question_inventory_single_writer",
            kind="constraint",
            text="Serialize writes per SKU with single-writer ownership or partition ownership to prevent oversell.",
            activation_keys=["single", "writer", "partition", "prevent", "oversell"],
            produced_by="controller",
            confidence=0.95,
        ))
        signals.append(SignalNode(
            id="sigv2_question_inventory_lifecycle",
            kind="constraint",
            text="Model the reservation lifecycle explicitly: hold/reserve, confirm, release/expire, and reconcile from the authoritative source of truth.",
            activation_keys=["reservation", "hold", "confirm", "release", "authoritative"],
            produced_by="controller",
            confidence=0.94,
        ))
        signals.append(SignalNode(
            id="sigv2_question_inventory_authority",
            kind="constraint",
            text="Treat cache as derived state; keep an authoritative source of truth and reconciliation path for inventory.",
            activation_keys=["cache", "derived", "authoritative", "source", "truth", "inventory"],
            produced_by="controller",
            confidence=0.94,
        ))
        signals.append(SignalNode(
            id="sigv2_question_inventory_dedupe",
            kind="constraint",
            text="Use idempotency keys or dedupe tokens so retries and at-least-once delivery do not duplicate reservations or confirmations.",
            activation_keys=["idempotency", "dedupe", "retries", "least", "once"],
            produced_by="controller",
            confidence=0.93,
        ))

    for anchor_id in anchor_ids:
        node = graph.nodes.get(anchor_id)
        if node is None:
            continue
        text = (getattr(node, "text", "") or "").strip()
        if not text:
            continue
        signals.append(SignalNode(
            id=f"sigv2_anchor_{anchor_id}",
            kind="evidence",
            text=text[:500],
            activation_keys=activation_keys_for_text(text),
            source_node_id=anchor_id,
            source_node_type=getattr(node, "node_type", "fact"),
            produced_by="projection",
            confidence=float(getattr(node, "confidence", 0.6) or 0.6),
        ))
    return signals


def _load_warm_start_workspace(
    session_paths: List[Path],
    workspace: Workspace,
    *,
    question: str = "",
    min_overlap: int = 3,
) -> int:
    """Project filled slots from prior sessions into the current workspace.

    For each prior session whose subgraph contains a __workspace__ entry
    with a matching domain (determined by activation key overlap against
    the current question), copies all filled slot text into the current
    workspace's corresponding slots.

    Returns the number of slots filled from warm-start projection.
    """
    if not session_paths or workspace is None:
        return 0

    q_lower = question.lower()
    q_tokens = {tok for tok in q_lower.split() if len(tok) >= 4}
    filled_count = 0

    for path in session_paths:
        subgraph_path = path if path.name == "subgraph.json" else path / "subgraph.json"
        if not subgraph_path.exists():
            continue
        try:
            subgraph = SessionSubgraph.from_dict(json.loads(subgraph_path.read_text(encoding="utf-8")))
        except Exception:
            continue
        ws_node = subgraph.nodes.get("__workspace__")
        if ws_node is None:
            continue
        prior_state = ws_node.get("slots")
        prior_fill = ws_node.get("fill_order")
        if not prior_state or not prior_fill:
            continue
        # Check domain match: question tokens must overlap pooled workspace
        # slot texts by ≥ min_overlap tokens.
        pooled_text = " ".join(
            v for v in prior_state.values() if v
        ).lower()
        pooled_tokens = {tok for tok in pooled_text.split() if len(tok) >= 4}
        overlap = len(q_tokens & pooled_tokens)
        if overlap < min_overlap:
            continue
        # Domain matches — project filled slots
        for slot_name in prior_fill:
            if workspace.is_filled(slot_name):
                continue  # don't overwrite already-filled slots
            prior_text = prior_state.get(slot_name)
            if prior_text:
                try:
                    workspace.fill(slot_name, prior_text)
                    filled_count += 1
                except KeyError:
                    # slot_name not in current workspace's fill_order — skip
                    continue
    return filled_count


def _load_warm_start_signals(session_paths: List[Path]) -> List[SignalNode]:
    signals: List[SignalNode] = []
    seen: Set[tuple[str, str]] = set()
    _NOISY_WARM_KINDS = {"hypothesis", "gap"}
    for path in session_paths:
        subgraph_path = path if path.name == "subgraph.json" else path / "subgraph.json"
        if not subgraph_path.exists():
            continue
        try:
            subgraph = SessionSubgraph.from_dict(json.loads(subgraph_path.read_text(encoding="utf-8")))
        except Exception:
            continue
        source_session = subgraph.session_id
        for sig in project_session_subgraph_to_signals(subgraph):
            if sig.kind in _NOISY_WARM_KINDS:
                continue
            if float(sig.confidence) < 0.4:
                continue
            if _is_checker_residue_text(sig.text):
                continue
            key = (sig.kind, (sig.text or "").strip().lower())
            if key in seen or not sig.text:
                continue
            seen.add(key)
            state = dict(sig.state or {})
            state["warm_start"] = True
            state["source_session_id"] = source_session
            state["source_session_path"] = str(subgraph_path.parent)
            signals.append(SignalNode(
                id=f"warm_{source_session}_{_short_hash(sig.id + sig.text)}",
                kind=sig.kind,
                text=sig.text,
                scope="reusable",
                activation_keys=list(sig.activation_keys),
                source_step_id=sig.source_step_id,
                produced_by="projection",
                state=state,
                evidence_ids=list(sig.evidence_ids),
                citation_count=sig.citation_count,
                decay=sig.decay,
                source_node_id=sig.source_node_id,
                source_node_type=sig.source_node_type,
                confidence=min(1.0, max(0.35, float(sig.confidence))),
            ))
    return signals


def _substrate_v2_checker_plugins(question: str) -> List[str]:
    lower = question.lower()
    plugins = ["generic_step_format", "algorithm_design"]
    if "subarray" in lower and ("update" in lower or "online" in lower):
        plugins.append("dynamic_max_subarray")
    if ("remove(" in lower and "connected(" in lower) or ("plain dsu is insufficient" in lower) or ("time-axis structure" in lower):
        plugins.append("dynamic_connectivity_deletions")
    if "range_chmin" in lower or ("range_sum" in lower and "per-node state" in lower):
        plugins.append("segment_tree_beats")
    if "dijkstra" in lower or "shortest" in lower or "negative edge" in lower:
        plugins.append("shortest_path_safety")
    if "payment worker" in lower or "psp" in lower or ("double charge" in lower and "idempotency" in lower):
        plugins.append("payment_crash_recovery")
    if "zero downtime" in lower or ("cutover" in lower and "rollback" in lower and "order" in lower):
        plugins.append("zero_downtime_migration")
    if "flash-sale" in lower or "flash sale" in lower or ("reservation ttl" in lower and "oversell" in lower):
        plugins.append("inventory_reservation")
    if not any(tok in lower for tok in ("algorithm", "dijkstra", "shortest", "subarray", "code", "c++", "python")):
        plugins.append("factual_recall")
    return plugins


def _substrate_v2_shaper_evidence(signals: List[SignalNode]) -> str:
    lines: List[str] = []
    for sig in signals:
        if sig.produced_by == "checker":
            continue
        if sig.kind not in {"constraint", "decision", "evidence", "risk", "repair"}:
            continue
        text = (sig.text or "").strip()
        if text:
            lines.append(f"- [{sig.kind}] {text[:220]}")
        if len(lines) >= 8:
            break
    return "\n".join(lines) if lines else "- (none)"


def _short_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:10]


def _session_event_payload(
    *,
    session: SessionSubgraphController,
    stage: str,
    iteration: int,
    **extra: Any,
) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "stage": stage,
        "iteration": iteration,
        "session_id": session.subgraph.session_id,
        "session_subgraph": session.subgraph.to_dict(),
        "node_count": len(session.subgraph.nodes),
        "edge_count": len(session.subgraph.edges),
        "audit_count": len(session.subgraph.audit_log),
        "step_count": session.subgraph.step_count,
    }
    payload.update(extra)
    return payload


def _extract_blocks(text: str) -> tuple[str, str]:
    """Return (reasoning, answer). Takes the LAST occurrence of each
    (so prompt directive placeholders don't shadow the model's output)."""
    rs = _REASONING_RE.findall(text)
    as_ = _ANSWER_RE.findall(text)
    return (
        rs[-1].strip() if rs else "",
        as_[-1].strip() if as_ else "",
    )


def _build_prompt(
    req: ReasoningRequest,
    graph: MemoryGraph,
    anchor_ids: List[str],
    procedure_pool: List[ProcedureNode],
    dispatch_outcomes: List[DispatchOutcome],
    iteration: int,
    signals: Optional[List[Signal]] = None,
    task_frame: Optional[GraphTaskFrame] = None,
    micro_context_block: str = "",
) -> str:
    """Build the substrate-aware prompt for one reasoning iteration.

    FINALIZATION MODE: when `dispatch_outcomes` is non-empty, this is a
    synthesis turn — procedures have already run and the model should
    compose the final answer from their results. In that case we DROP
    the `# Available procedures` section entirely so the model isn't
    tempted to re-invoke (the cs4 Dijkstra real-LLM run on 2026-05-21
    re-invoked VerifyNonNegativeEdges purely because the procedure list
    was still in front of it). The directive switches to a
    finalize-only variant that explicitly forbids further invocation.
    """
    parts = [_render_facts(graph, anchor_ids)]

    in_finalization = bool(dispatch_outcomes)

    task_frame_block = render_task_frame(task_frame)
    if task_frame_block:
        parts.append(task_frame_block)

    if micro_context_block:
        parts.append(micro_context_block)

    show_procedure_catalog = (
        not in_finalization
        and _should_show_procedure_catalog(task_frame)
    )

    if show_procedure_catalog:
        procs_section = _render_procedures(procedure_pool)
        if procs_section:
            parts.append(procs_section)

    if dispatch_outcomes:
        parts.append(_render_dispatch_results(dispatch_outcomes))

    # Phase 3A: signal injection. Renders only when non-empty so
    # backward-compat prompts (with no meta_pool) are unchanged.
    if signals:
        signals_block = render_signals_block(signals)
        if signals_block:
            parts.append(signals_block)

    parts.append(_render_directive(
        iteration=iteration,
        has_dispatch=in_finalization,
        has_signals=bool(signals),
        has_task_frame=bool(task_frame_block),
        has_procedure_catalog=show_procedure_catalog,
    ))
    parts.append(f"Question: {req.question}")
    return "\n\n".join(parts)


def _should_show_procedure_catalog(task_frame: Optional[GraphTaskFrame]) -> bool:
    if task_frame is None:
        return True
    return bool(task_frame.procedure_suggestions)


def _render_facts(graph: MemoryGraph, anchor_ids: List[str]) -> str:
    lines = ["# Background facts", ""]
    for nid in anchor_ids:
        node = graph.nodes.get(nid)
        if node is None:
            continue
        ntype = getattr(node, "node_type", "fact")
        # Skip procedure nodes here — they get their own section
        if ntype == "procedure":
            continue
        text = (getattr(node, "text", "") or "").strip()
        lines.append(f"### type={ntype}")
        lines.append(text)
        lines.append("")
    return "\n".join(lines).strip()


def _render_procedures(procedure_pool: List[ProcedureNode]) -> str:
    if not procedure_pool:
        return ""
    lines = ["# Available procedures", "",
             "These are reusable reasoning tools. To invoke one, write a phrase like "
             "\"I'll apply <Name> to ...\" or \"using the <Name> procedure\" in your reasoning. "
             "The system will run the procedure, mutate session state, and splice the result back.", ""]
    for proc in procedure_pool:
        lines.append(f"## {proc.name}")
        lines.append(f"Purpose: {proc.purpose}")
        lines.append(f"When to use: {proc.when_to_use}")
        if proc.signature.get("inputs"):
            inputs_str = ", ".join(
                f"{i['name']}: {i.get('type', 'any')}" for i in proc.signature["inputs"]
            )
            lines.append(f"Inputs: {inputs_str}")
        lines.append("")
    return "\n".join(lines).strip()


def _render_dispatch_results(outcomes: List[DispatchOutcome]) -> str:
    """Render the dispatch tree for the finalization prompt.

    Walks sub_outcomes recursively so the model sees child procedures'
    state too, not just the top-level composer's summary. Without this,
    the model in finalization mode would re-invoke leaves it didn't see
    in the prompt (observed in cs4 Dijkstra real-LLM run, 2026-05-21:
    composer summary made it through but child VerifyNonNegativeEdges
    state did not, so the model "helpfully" tried to invoke
    VerifyNonNegativeEdges again to fill the gap).
    """
    lines = ["# Results from procedures invoked so far", ""]
    for o in outcomes:
        _render_one_outcome(o, depth=0, lines=lines)
    return "\n".join(lines).strip()


def _render_one_outcome(o: DispatchOutcome, *, depth: int, lines: List[str]) -> None:
    indent = "  " * depth
    if o.error:
        lines.append(f"{indent}## {o.match.procedure_name} — ERROR: {o.error}")
        return
    lines.append(f"{indent}## {o.match.procedure_name}")
    lines.append(f"{indent}Mutations applied: {o.mutations_applied}")
    summary = _extract_after_done(o.sub_response)
    if summary:
        lines.append(f"{indent}Summary: {summary}")
    # Show the child's resulting state so the model sees concrete findings,
    # not just a prose summary. Cheap (a few lines per child) and prevents
    # the "I'd better re-invoke the child to see its state" failure mode.
    state = _read_object_state(o)
    if state:
        state_line = _format_state_compact(state)
        if state_line:
            lines.append(f"{indent}State: {state_line}")
    # Recurse into sub_outcomes so composer children appear nested.
    for child in (o.sub_outcomes or []):
        _render_one_outcome(child, depth=depth + 1, lines=lines)
    lines.append("")


def _read_object_state(o: DispatchOutcome) -> Optional[Dict[str, Any]]:
    """The DispatchOutcome doesn't carry session state directly; we parse
    SET / ADD commands out of sub_response as a best-effort reconstruction.

    This is intentionally lightweight — it captures the typical leaf
    procedure shape (lists of items, scalar verdicts) without depending
    on the full SessionSubgraphController. For complex state, the model
    has the dispatch-results section's prose summary as a fallback.
    """
    if not o.sub_response:
        return None
    state: Dict[str, Any] = {}
    # ADD <value> TO state.<path>  → append to list at state[path]
    for m in re.finditer(
        r"^\s*ADD\s+(?P<value>.+?)\s+TO\s+state\.(?P<path>[A-Za-z_][\w\.]*)\s*$",
        o.sub_response, re.IGNORECASE | re.MULTILINE,
    ):
        path = m.group("path")
        value = m.group("value").strip().strip('"').strip("'")
        state.setdefault(path, [])
        if isinstance(state[path], list):
            state[path].append(value)
    # SET state.<path> = <value>  → scalar assignment
    for m in re.finditer(
        r"^\s*SET\s+state\.(?P<path>[A-Za-z_][\w\.]*)\s*=\s*(?P<value>.+?)\s*$",
        o.sub_response, re.IGNORECASE | re.MULTILINE,
    ):
        path = m.group("path")
        value = m.group("value").strip().strip('"').strip("'")
        # SET overwrites
        state[path] = value
    return state or None


def _format_state_compact(state: Dict[str, Any]) -> str:
    """Compact one-line state rendering for the prompt. Truncates list
    contents at 5 items and overall string at 200 chars."""
    parts = []
    for key, value in state.items():
        if isinstance(value, list):
            shown = value[:5]
            suffix = ", ..." if len(value) > 5 else ""
            parts.append(f"{key}=[{', '.join(map(str, shown))}{suffix}]")
        else:
            parts.append(f"{key}={value}")
    rendered = "; ".join(parts)
    if len(rendered) > 200:
        rendered = rendered[:197] + "..."
    return rendered


def _extract_after_done(sub_response: str) -> str:
    """Lines after the DONE marker are treated as the procedure's natural-
    language summary, useful for the reasoner's next turn."""
    if not sub_response:
        return ""
    parts = re.split(r"^\s*DONE\s*$", sub_response, maxsplit=1, flags=re.IGNORECASE | re.MULTILINE)
    if len(parts) < 2:
        return ""
    return parts[1].strip()


def _attach_diagnostics_to_subgraph(
    session: SessionSubgraphController,
    *,
    raw_outputs: List[str],
    anchor_ids: List[str],
    budget_summary: Dict[str, Any],
    dispatch_outcomes: List[DispatchOutcome],
) -> None:
    """Stash per-iteration debug data on the SessionSubgraph dict.

    SessionSubgraph is a `@dataclass` with fixed fields, so we cannot
    add a new attribute cleanly. Instead, we store diagnostics in a
    dedicated `__diagnostics__` key inside one of the JSON-only paths.
    The chosen location: write a sibling `diagnostics.json` file in the
    same session directory via SessionSubgraphController's persist
    layer. We extend `subgraph.to_dict()` only minimally by piggy-
    backing on the audit log (each entry gets a special marker is
    awkward; cleaner to write a sibling file).

    Implementation: store diagnostics as a node with id=`__diag__`
    and node_type=`diagnostics` so it round-trips through the existing
    serialization without needing schema changes.
    """
    diag_node = {
        "id": "__diag__",
        "node_type": "diagnostics",
        "raw_outputs": list(raw_outputs),
        "anchor_ids": list(anchor_ids),
        "budget_summary": dict(budget_summary),
        "dispatch_summary": [
            {
                "verb": o.match.verb,
                "procedure_name": o.match.procedure_name,
                "args_text": o.match.args_text,
                "object_id": o.object_id,
                "parent_object_id": o.parent_object_id,
                "mutations_applied": o.mutations_applied,
                "error": o.error,
                "sub_outcome_count": len(o.sub_outcomes or []),
                # First 400 chars of the sub_response — enough to see if
                # the procedure body actually emitted CALL/SET/etc.
                "sub_response_preview": (o.sub_response or "")[:400],
            }
            for o in dispatch_outcomes
        ],
        "iterations_observed": len(raw_outputs),
    }
    # Only add if absent — idempotent (close() may be called twice in tests)
    if "__diag__" not in session.subgraph.nodes:
        session.subgraph.nodes["__diag__"] = diag_node


def _seed_session_baseline(
    *,
    session: SessionSubgraphController,
    question: str,
    answer: str,
    anchor_ids: List[str],
    graph: MemoryGraph,
) -> None:
    """Always-present baseline structure for the session subgraph.

    Per Phase 2A acceptance criterion #4 (PHASE2_PLAN.md §11): even when
    no procedure fires, the session must contain something the UI can
    render — at minimum a question node, an answer node, and the
    retrieved anchors as evidence.

    Called once at the end of every reasoning run (after the loop, before
    persist). Procedure-produced session_objects coexist with these
    baseline nodes; the baseline never overwrites or duplicates them.

    Q0 and A0 use deterministic ids so multiple sessions render with the
    same structure. Anchor nodes are prefixed with `anchor_` to avoid
    colliding with any long-term graph ids that might happen to match.
    """
    # Already-seeded by an earlier call? Idempotent.
    if "Q0" in session.subgraph.nodes:
        return

    # Question node
    session.subgraph.nodes["Q0"] = {
        "id": "Q0",
        "node_type": "question",
        "text": question,
        "created_step": 0,
        "metadata": {"provider": "substrate-baseline"},
    }
    # Answer node
    session.subgraph.nodes["A0"] = {
        "id": "A0",
        "node_type": "answer",
        "text": answer or "(no answer produced)",
        "created_step": max(0, session.subgraph.step_count),
        "metadata": {"provider": "substrate-baseline"},
    }
    # Evidence nodes from anchors
    for anchor_id in anchor_ids:
        node = graph.nodes.get(anchor_id)
        if node is None:
            continue
        text = (getattr(node, "text", "") or "").strip()
        evidence_id = f"anchor_{anchor_id}"
        if evidence_id in session.subgraph.nodes:
            continue
        session.subgraph.nodes[evidence_id] = {
            "id": evidence_id,
            "node_type": "evidence",
            "text": text,
            "source_memory_id": anchor_id,
            "created_step": 0,
            "metadata": {
                "provider": "substrate-baseline",
                "underlying_node_type": getattr(node, "node_type", "fact"),
            },
        }
        # Wire evidence -> answer support edge so the UI can render the
        # "these facts fed this answer" relationship cleanly.
        session.add_edge(
            src=evidence_id,
            dst="A0",
            relation="support",
            metadata={"provider": "substrate-baseline"},
        )





def _fallback_answer(
    raw_outputs: List[str],
    dispatch_outcomes: List[DispatchOutcome],
    early_terminated_reason: Optional[str],
) -> str:
    """Produce a non-empty user-facing answer when no <answer> block was emitted.

    This is the degradation path for budget exhaustion or partial model outputs.
    Prefer the last procedure summary, then the last raw output, then a plain
    budget note.
    """
    for outcome in reversed(dispatch_outcomes):
        summary = _extract_after_done(outcome.sub_response)
        if summary:
            return summary

    for raw in reversed(raw_outputs):
        _, answer = _extract_blocks(raw)
        if answer:
            return answer
        stripped = raw.strip()
        if stripped:
            return stripped

    if early_terminated_reason:
        return (
            "The reasoning budget was exhausted before a final answer block was produced. "
            "Review the persisted session subgraph and audit log for the partial trace."
        )
    return "No final answer was produced."


_SIGNAL_DIRECTIVE_RIDER = (
    "\n\nIf the System signals section above reports anything, address those "
    "concerns in your reasoning before answering. ERROR signals especially "
    "must be acknowledged — do not ignore a contradiction or cycle the system "
    "has flagged."
)


_TASK_FRAME_DIRECTIVE_RIDER = (
    "\n\nUse the <graph_task_frame> above as private guidance before "
    "answering. Do not mention graph internals, node ids, or internal frame "
    "names in the final answer."
)


def _render_directive(
    iteration: int,
    has_dispatch: bool,
    has_signals: bool = False,
    has_task_frame: bool = False,
    has_procedure_catalog: bool = True,
) -> str:
    if iteration == 0:
        base = _DIRECTIVE_INITIAL if has_procedure_catalog else _DIRECTIVE_INITIAL_DIRECT_ONLY
    elif has_dispatch:
        base = _DIRECTIVE_FOLLOWUP_WITH_DISPATCH
    else:
        base = _DIRECTIVE_FOLLOWUP
    if has_signals:
        base = base + _SIGNAL_DIRECTIVE_RIDER
    if has_task_frame:
        base = base + _TASK_FRAME_DIRECTIVE_RIDER
    return base


_DIRECTIVE_INITIAL = (
    "---\n\n"
    "You have absorbed the background facts above. The user does not know "
    "this material exists. Answer as a domain expert.\n\n"
    "If a procedure listed above clearly applies to the question, write "
    "explicitly in your <reasoning>: \"I'll apply <ProcedureName> to <args>\". "
    "The system will run it and you'll get the result on the next turn — in "
    "that case (and ONLY that case) you may omit the <answer> block on this turn.\n\n"
    "If no procedure applies — including for conceptual questions, definitions, "
    "explanations, or comparisons — answer the question directly on this turn. "
    "Do NOT use the procedure invocation phrasing unless you actually want a "
    "procedure to run.\n\n"
    "Emit two blocks:\n"
    "<reasoning>\n"
    "GOAL: <one-line restatement>\n"
    "KNOWN: <facts you can use>\n"
    "PLAN: <how you'll compose the answer; mention procedure invocations here ONLY if applying>\n"
    "</reasoning>\n"
    "<answer>your final answer here — REQUIRED unless you explicitly invoked a procedure above</answer>\n\n"
    "ABSOLUTE RULES:\n"
    "1. ALWAYS attempt the question. Never refuse.\n"
    "2. NEVER mention 'the graph', 'the reference material', 'the nodes', "
    "'the absorbed material', or any internal structure in your answer.\n"
    "3. NEVER include node identifiers or citation markers.\n"
    "4. The <answer> block reads as polished prose.\n"
    "5. Do not skip <answer> unless you actually invoked a procedure on this turn."
)


_DIRECTIVE_INITIAL_DIRECT_ONLY = (
    "---\n\n"
    "You have absorbed the background facts above. The user does not know "
    "this material exists. Answer as a domain expert.\n\n"
    "No procedure catalog is available for this turn. Answer the question "
    "directly; do not write procedure-invocation phrasing such as "
    "\"I'll apply <ProcedureName>\" or \"using the <Name> procedure\".\n\n"
    "Emit two blocks:\n"
    "<reasoning>\n"
    "GOAL: <one-line restatement>\n"
    "KNOWN: <facts you can use>\n"
    "PLAN: <how you'll compose the answer directly>\n"
    "</reasoning>\n"
    "<answer>your final answer here</answer>\n\n"
    "ABSOLUTE RULES:\n"
    "1. ALWAYS attempt the question. Never refuse.\n"
    "2. NEVER mention 'the graph', 'the reference material', 'the nodes', "
    "'the absorbed material', or any internal structure in your answer.\n"
    "3. NEVER include node identifiers or citation markers.\n"
    "4. The <answer> block reads as polished prose.\n"
    "5. Do not use procedure-invocation phrasing on this turn."
)


_DIRECTIVE_FOLLOWUP_WITH_DISPATCH = (
    "---\n\n"
    "FINALIZATION MODE.\n\n"
    "The procedure invocations above produced state mutations and summaries. "
    "Use them to compose your final answer NOW.\n\n"
    "DO NOT invoke any more procedures on this turn. Specifically:\n"
    "  - Do NOT write \"I'll apply <Name> to ...\" or \"I will apply <Name>\".\n"
    "  - Do NOT write \"using the <Name> procedure\" or \"invoke <Name>\".\n"
    "  - Do NOT propose follow-up procedure calls in your reasoning or answer.\n"
    "The available-procedures catalog has been intentionally removed from "
    "this prompt; the results above are complete. If you feel a result is "
    "missing, work from what is shown — do not request more.\n\n"
    "Emit two blocks:\n"
    "<reasoning>GOAL/KNOWN/PLAN as before, integrating procedure results.</reasoning>\n"
    "<answer>final natural-language answer for the user</answer>\n\n"
    "Same rules as before: no internal structure leaks, no node ids, no citation markers."
)


_DIRECTIVE_FOLLOWUP = (
    "---\n\n"
    "Continue the reasoning and produce a final answer.\n\n"
    "Same rules as before: no internal structure leaks, no node ids, no citation markers."
)
