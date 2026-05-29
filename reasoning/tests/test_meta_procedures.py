"""Unit tests for the five Phase-3A meta-procedures.

Each procedure is tested in isolation against a stub MetaContext.
Integration with the reasoning loop is covered separately in
test_signal_injection.py.
"""
from __future__ import annotations

import unittest
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from reasoning.meta import MetaContext
from reasoning.meta_procedures.budget_warner import (
    build_budget_warner,
    _detect_budget_pressure,
    BUDGET_THRESHOLD,
)
from reasoning.meta_procedures.contradiction_detector import (
    build_contradiction_detector,
    _detect_contradictions,
    KNOWN_VERDICT_FIELDS,
)
from reasoning.meta_procedures.cycle_detector import (
    build_cycle_detector,
    _detect_cycles,
    CYCLE_THRESHOLD,
)
from reasoning.meta_procedures.dispatch_miss_nudge import (
    build_dispatch_miss_nudge,
    _detect_dispatch_misses,
)
from reasoning.meta_procedures.no_dispatch_after_threshold import (
    build_no_dispatch_after_threshold,
    _detect_no_dispatch_after_threshold,
    DISPATCH_THRESHOLD_ITER,
)


# ---- Stubs that mimic the real types just enough for predicates ------- #

@dataclass
class _StubMatch:
    procedure_name: str
    args_text: Optional[str]
    verb: str = "apply_intent"


@dataclass
class _StubOutcome:
    procedure_id: Optional[str]
    object_id: Optional[str]
    match: _StubMatch
    parent_object_id: Optional[str] = None
    error: Optional[str] = None


class _StubBudgets:
    def __init__(self, **caps):
        self.max_llm_calls = caps.get("max_llm_calls", 100)
        self.max_hops = caps.get("max_hops", 100)
        self.max_session_subgraph_size = caps.get("max_session_subgraph_size", 100)
        self.max_total_tokens = caps.get("max_total_tokens", 10000)
        self.max_composition_fan_out = caps.get("max_composition_fan_out", 5)
        self.max_recursion_depth = caps.get("max_recursion_depth", 4)


class _StubBudgetTracker:
    def __init__(self, used: Dict[str, int], budgets: _StubBudgets):
        self.used = used
        self.budgets = budgets


class _StubSession:
    class _SG:
        def __init__(self, nodes: Dict[str, Any], edges: List[Any]):
            self.nodes = nodes
            self.edges = edges

    def __init__(self, nodes=None, edges=None):
        self.subgraph = _StubSession._SG(nodes or {}, edges or [])


def _mk_ctx(
    *,
    dispatch_outcomes: Optional[List[_StubOutcome]] = None,
    raw_outputs: Optional[List[str]] = None,
    iteration: int = 0,
    procedure_names: Optional[List[str]] = None,
    nodes: Optional[Dict[str, Any]] = None,
    used: Optional[Dict[str, int]] = None,
    budget_caps: Optional[Dict[str, int]] = None,
) -> MetaContext:
    return MetaContext(
        session=_StubSession(nodes=nodes or {}),                   # type: ignore[arg-type]
        budget=_StubBudgetTracker(
            used=used or {},
            budgets=_StubBudgets(**(budget_caps or {})),
        ),                                                          # type: ignore[arg-type]
        dispatch_outcomes=dispatch_outcomes or [],                  # type: ignore[arg-type]
        raw_outputs=raw_outputs or [],
        anchor_ids=[],
        current_iteration=iteration,
        previous_signals=[],
        procedure_names=procedure_names or [],
    )


# ============ CycleDetector ============================================ #

class TestCycleDetector(unittest.TestCase):
    def _mk(self, oid, proc_id, proc_name, args):
        return _StubOutcome(
            procedure_id=proc_id,
            object_id=oid,
            match=_StubMatch(procedure_name=proc_name, args_text=args),
        )

    def test_below_threshold_no_signal(self):
        outs = [
            self._mk(f"so_{i}", "proc_x", "X", "same_args")
            for i in range(CYCLE_THRESHOLD - 1)
        ]
        sigs = _detect_cycles(_mk_ctx(dispatch_outcomes=outs))
        self.assertEqual(sigs, [])

    def test_at_threshold_fires(self):
        outs = [
            self._mk(f"so_{i}", "proc_x", "X", "same_args")
            for i in range(CYCLE_THRESHOLD)
        ]
        sigs = _detect_cycles(_mk_ctx(dispatch_outcomes=outs, iteration=2))
        self.assertEqual(len(sigs), 1)
        self.assertEqual(sigs[0].severity, "warn")
        self.assertEqual(sigs[0].type, "cycle_detected")
        self.assertEqual(len(sigs[0].related_node_ids), CYCLE_THRESHOLD)
        self.assertIn(f"{CYCLE_THRESHOLD} times", sigs[0].message)

    def test_different_args_not_a_cycle(self):
        outs = [
            self._mk("so_a", "proc_x", "X", "args_A"),
            self._mk("so_b", "proc_x", "X", "args_B"),
            self._mk("so_c", "proc_x", "X", "args_C"),
        ]
        sigs = _detect_cycles(_mk_ctx(dispatch_outcomes=outs))
        self.assertEqual(sigs, [])

    def test_two_different_cycles_two_signals(self):
        outs = []
        for i in range(CYCLE_THRESHOLD):
            outs.append(self._mk(f"so_x{i}", "proc_x", "X", "args_X"))
        for i in range(CYCLE_THRESHOLD):
            outs.append(self._mk(f"so_y{i}", "proc_y", "Y", "args_Y"))
        sigs = _detect_cycles(_mk_ctx(dispatch_outcomes=outs))
        self.assertEqual(len(sigs), 2)

    def test_signal_id_stable_within_session(self):
        """Same procedure + same args produces stable signal id so the
        once_per_session dedupe in MetaPool actually fires once."""
        outs = [self._mk(f"so_{i}", "proc_x", "X", "args") for i in range(CYCLE_THRESHOLD)]
        sigs1 = _detect_cycles(_mk_ctx(dispatch_outcomes=outs))
        # Add a 4th invocation; predicate may or may not fire (id stable)
        outs.append(self._mk("so_extra", "proc_x", "X", "args"))
        sigs2 = _detect_cycles(_mk_ctx(dispatch_outcomes=outs))
        self.assertEqual(sigs1[0].id, sigs2[0].id,
                         "Same cycle should produce the same signal id")


# ============ BudgetWarner ============================================= #

class TestBudgetWarner(unittest.TestCase):
    def test_below_threshold_no_signal(self):
        # 50% of LLM cap — should NOT fire
        sigs = _detect_budget_pressure(_mk_ctx(
            used={"llm_call": 5},
            budget_caps={"max_llm_calls": 10},
        ))
        self.assertEqual(sigs, [])

    def test_at_threshold_fires(self):
        # 8/10 = 0.8 — clearly above the 0.75 threshold
        sigs = _detect_budget_pressure(_mk_ctx(
            used={"llm_call": 8},
            budget_caps={"max_llm_calls": 10},
        ))
        self.assertEqual(len(sigs), 1)
        self.assertEqual(sigs[0].severity, "info")
        self.assertEqual(sigs[0].type, "budget_at_threshold")
        self.assertIn("LLM calls", sigs[0].message)

    def test_multiple_axes_multiple_signals(self):
        # Both LLM and tokens crossed — two signals
        sigs = _detect_budget_pressure(_mk_ctx(
            used={"llm_call": 8, "tokens": 800},
            budget_caps={"max_llm_calls": 10, "max_total_tokens": 1000},
        ))
        self.assertEqual(len(sigs), 2)
        types_seen = {s.metadata["axis"] for s in sigs}
        self.assertEqual(types_seen, {"llm_call", "tokens"})

    def test_cap_zero_does_not_divide_by_zero(self):
        sigs = _detect_budget_pressure(_mk_ctx(
            used={"llm_call": 0},
            budget_caps={"max_llm_calls": 0},
        ))
        self.assertEqual(sigs, [])

    def test_signal_id_includes_iteration(self):
        """The id must include iteration so stream dedupe doesn't drop
        re-emissions across turns while the threshold is held."""
        ctx_iter0 = _mk_ctx(used={"llm_call": 8}, budget_caps={"max_llm_calls": 10}, iteration=0)
        ctx_iter1 = _mk_ctx(used={"llm_call": 9}, budget_caps={"max_llm_calls": 10}, iteration=1)
        s0 = _detect_budget_pressure(ctx_iter0)
        s1 = _detect_budget_pressure(ctx_iter1)
        self.assertNotEqual(s0[0].id, s1[0].id,
                            "Budget signal ids must vary across iterations")


# ============ ContradictionDetector =================================== #

class TestContradictionDetector(unittest.TestCase):
    def _make_outcome(self, oid, name, parent=None):
        return _StubOutcome(
            procedure_id=f"proc_{name.lower()}",
            object_id=oid,
            match=_StubMatch(procedure_name=name, args_text="args"),
            parent_object_id=parent,
        )

    def _make_node(self, oid, state):
        return {
            "id": oid,
            "node_type": "session_object",
            "state": state,
        }

    def test_no_siblings_no_signal(self):
        # Only one child, no contradiction possible
        out = self._make_outcome("so_child", "ChildProc", parent="so_parent")
        nodes = {"so_child": self._make_node("so_child", {"safe_to_apply": True})}
        sigs = _detect_contradictions(_mk_ctx(
            dispatch_outcomes=[out], nodes=nodes,
        ))
        self.assertEqual(sigs, [])

    def test_siblings_agreeing_no_signal(self):
        outs = [
            self._make_outcome("so_a", "A", parent="so_p"),
            self._make_outcome("so_b", "B", parent="so_p"),
        ]
        nodes = {
            "so_a": self._make_node("so_a", {"safe_to_apply": True}),
            "so_b": self._make_node("so_b", {"safe_to_apply": True}),
        }
        sigs = _detect_contradictions(_mk_ctx(
            dispatch_outcomes=outs, nodes=nodes,
        ))
        self.assertEqual(sigs, [])

    def test_siblings_disagreeing_known_field_fires(self):
        outs = [
            self._make_outcome("so_a", "A", parent="so_p"),
            self._make_outcome("so_b", "B", parent="so_p"),
        ]
        nodes = {
            "so_a": self._make_node("so_a", {"safe_to_apply": True}),
            "so_b": self._make_node("so_b", {"safe_to_apply": False}),
        }
        sigs = _detect_contradictions(_mk_ctx(
            dispatch_outcomes=outs, nodes=nodes,
        ))
        self.assertEqual(len(sigs), 1)
        self.assertEqual(sigs[0].severity, "error")
        self.assertTrue(sigs[0].sticky)
        self.assertEqual(set(sigs[0].related_node_ids), {"so_a", "so_b"})

    def test_siblings_disagreeing_UNKNOWN_field_ignored(self):
        """Unknown bool fields are not in the whitelist → no signal."""
        outs = [
            self._make_outcome("so_a", "A", parent="so_p"),
            self._make_outcome("so_b", "B", parent="so_p"),
        ]
        nodes = {
            "so_a": self._make_node("so_a", {"my_custom_bool": True}),
            "so_b": self._make_node("so_b", {"my_custom_bool": False}),
        }
        sigs = _detect_contradictions(_mk_ctx(
            dispatch_outcomes=outs, nodes=nodes,
        ))
        self.assertEqual(sigs, [])

    def test_known_verdict_fields_whitelist_contains_expected_names(self):
        """Lock the whitelist so accidental removal regresses contradiction
        detection coverage."""
        for must in ("safe_to_apply", "preconditions_satisfied",
                     "has_negative_edge", "has_negative_cycle"):
            self.assertIn(must, KNOWN_VERDICT_FIELDS)


# ============ DispatchMissNudge ======================================= #

class TestDispatchMissNudge(unittest.TestCase):
    def test_known_name_no_signal(self):
        output = "I'll apply VerifyShortestPath to check this."
        sigs = _detect_dispatch_misses(_mk_ctx(
            raw_outputs=[output],
            procedure_names=["VerifyShortestPath", "VerifyNonNegativeEdges"],
        ))
        self.assertEqual(sigs, [])

    def test_unknown_name_fires(self):
        output = "I'll apply VerifyShortestPreprocess to the user's graph."
        sigs = _detect_dispatch_misses(_mk_ctx(
            raw_outputs=[output],
            procedure_names=["VerifyShortestPath", "VerifyNonNegativeEdges"],
        ))
        self.assertEqual(len(sigs), 1)
        self.assertEqual(sigs[0].severity, "info")
        self.assertEqual(sigs[0].type, "dispatch_miss")
        self.assertIn("VerifyShortestPreprocess", sigs[0].message)
        self.assertIn("VerifyShortestPath", sigs[0].message)

    def test_case_insensitive_match(self):
        output = "I'll apply verifyshortestpath to..."   # all lowercase
        sigs = _detect_dispatch_misses(_mk_ctx(
            raw_outputs=[output],
            procedure_names=["VerifyShortestPath"],
        ))
        self.assertEqual(sigs, [])

    def test_no_output_no_signal(self):
        sigs = _detect_dispatch_misses(_mk_ctx(raw_outputs=[]))
        self.assertEqual(sigs, [])

    def test_invoke_pattern_recognized(self):
        output = "Now invoke MysteryProc with args=42"
        sigs = _detect_dispatch_misses(_mk_ctx(
            raw_outputs=[output],
            procedure_names=["KnownProc"],
        ))
        self.assertEqual(len(sigs), 1)
        self.assertIn("MysteryProc", sigs[0].message)

    def test_using_the_pattern_recognized(self):
        output = "Resolved using the FakeProc procedure"
        sigs = _detect_dispatch_misses(_mk_ctx(
            raw_outputs=[output],
            procedure_names=["KnownProc"],
        ))
        self.assertEqual(len(sigs), 1)
        self.assertIn("FakeProc", sigs[0].message)


# ============ NoDispatchAfterThreshold ================================ #

class TestNoDispatchAfterThreshold(unittest.TestCase):
    def test_before_threshold_iter_no_signal(self):
        sigs = _detect_no_dispatch_after_threshold(_mk_ctx(
            iteration=DISPATCH_THRESHOLD_ITER - 1,
            procedure_names=["KnownProc"],
        ))
        self.assertEqual(sigs, [])

    def test_at_threshold_with_no_dispatch_fires(self):
        sigs = _detect_no_dispatch_after_threshold(_mk_ctx(
            iteration=DISPATCH_THRESHOLD_ITER,
            procedure_names=["KnownProc"],
            dispatch_outcomes=[],
        ))
        self.assertEqual(len(sigs), 1)
        self.assertEqual(sigs[0].severity, "info")
        self.assertEqual(sigs[0].type, "no_dispatch_stale")
        self.assertIn("KnownProc", sigs[0].message)

    def test_at_threshold_WITH_dispatch_no_signal(self):
        """If anything has been dispatched already, the staleness check
        is irrelevant — don't fire."""
        out = _StubOutcome(
            procedure_id="proc_x",
            object_id="so_1",
            match=_StubMatch("X", "args"),
        )
        sigs = _detect_no_dispatch_after_threshold(_mk_ctx(
            iteration=DISPATCH_THRESHOLD_ITER,
            procedure_names=["KnownProc"],
            dispatch_outcomes=[out],
        ))
        self.assertEqual(sigs, [])

    def test_no_procedures_available_no_signal(self):
        """If the substrate has no procedures registered, signal would
        be useless — predicate suppresses."""
        sigs = _detect_no_dispatch_after_threshold(_mk_ctx(
            iteration=DISPATCH_THRESHOLD_ITER + 1,
            procedure_names=[],
        ))
        self.assertEqual(sigs, [])


# ============ build_default_meta_pool ================================ #

class TestDefaultMetaPool(unittest.TestCase):
    def test_pool_contains_all_five(self):
        from reasoning.meta_procedures import build_default_meta_pool
        pool = build_default_meta_pool()
        procs = pool.all_procedures()
        names = {p.name for p in procs}
        self.assertEqual(names, {
            "BudgetWarner",
            "NoDispatchAfterThreshold",
            "CycleDetector",
            "ContradictionDetector",
            "DispatchMissNudge",
        })

    def test_pool_distributed_across_hooks(self):
        from reasoning.meta_procedures import build_default_meta_pool
        pool = build_default_meta_pool()
        pre_iter = pool.procedures_for_hook("pre_iter")
        post = pool.procedures_for_hook("post_dispatch")
        self.assertEqual(len(pre_iter), 2)
        self.assertEqual(len(post), 3)


if __name__ == "__main__":
    unittest.main()
