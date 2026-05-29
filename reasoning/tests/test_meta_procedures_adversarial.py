"""Adversarial tests for meta-procedures using realistic GLM-style output.

Real GLM produces nuanced phrasings that simple regexes will get wrong.
This file is designed to FIND those false positives / negatives, not
celebrate that the happy path works.

Three pressure axes:
  1. Adversarial English phrasings (negation, past tense, hypothetical, comparative)
  2. Replay of real persisted sessions through the meta-pool
  3. Multi-MP interaction scenarios

Each test that REVEALS a real limitation is documented in the test
docstring rather than silently fixed. Where the predicate behaviour
is wrong, the test is in @unittest.expectedFailure with a comment.
Where the predicate behaviour is correct but the design is debatable,
the test passes with a doc note.
"""
from __future__ import annotations

import json
import unittest
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from reasoning.meta import MetaContext, MetaPool
from reasoning.meta_procedures import build_default_meta_pool
from reasoning.meta_procedures.cycle_detector import _detect_cycles
from reasoning.meta_procedures.dispatch_miss_nudge import _detect_dispatch_misses
from reasoning.meta_procedures.no_dispatch_after_threshold import (
    _detect_no_dispatch_after_threshold,
    DISPATCH_THRESHOLD_ITER,
)


# ---- Stubs (same shape as test_meta_procedures) ---------------------- #

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
    def __init__(self):
        self.max_llm_calls = 100
        self.max_hops = 100
        self.max_session_subgraph_size = 100
        self.max_total_tokens = 100000


class _StubBudgetTracker:
    def __init__(self):
        self.used = {}
        self.budgets = _StubBudgets()


class _StubSession:
    class _SG:
        def __init__(self):
            self.nodes: Dict[str, Any] = {}
            self.edges: List[Any] = []
    def __init__(self):
        self.subgraph = _StubSession._SG()


def _ctx(
    *,
    raw_outputs: Optional[List[str]] = None,
    iteration: int = 0,
    procedure_names: Optional[List[str]] = None,
    dispatch_outcomes: Optional[List[_StubOutcome]] = None,
) -> MetaContext:
    return MetaContext(
        session=_StubSession(),                                  # type: ignore[arg-type]
        budget=_StubBudgetTracker(),                             # type: ignore[arg-type]
        dispatch_outcomes=dispatch_outcomes or [],               # type: ignore[arg-type]
        raw_outputs=raw_outputs or [],
        anchor_ids=[],
        current_iteration=iteration,
        previous_signals=[],
        procedure_names=procedure_names or [],
    )


# =========================================================================
# Adversarial cases for DispatchMissNudge — the riskiest predicate
# =========================================================================

class TestDispatchMissNudgeFalsePositives(unittest.TestCase):
    """Phrasings where the model mentions a non-existent name in a
    non-invocation context. Some of these are KNOWN false positives in
    the current regex-based design — documented as such."""

    PROCS = ["VerifyShortestPath", "VerifyNonNegativeEdges",
             "VerifyAlgorithmPreconditions", "DetectNegativeCycle"]

    def test_PASS_pure_freetext_mention(self):
        """'The VerifyShortestPath procedure is useful' — no invocation
        verb. Predicate correctly ignores."""
        output = "The VerifyShortestPath procedure is the right tool for this kind of question."
        sigs = _detect_dispatch_misses(_ctx(raw_outputs=[output], procedure_names=self.PROCS))
        self.assertEqual(sigs, [],
                         "Free-text mention without verb should NOT fire dispatch_miss")

    def test_PASS_question_about_a_procedure(self):
        """'What does the FooBar procedure do?' — no invocation. Correctly ignored."""
        output = "What does the BogusProc procedure even do? I don't think it applies here."
        sigs = _detect_dispatch_misses(_ctx(raw_outputs=[output], procedure_names=self.PROCS))
        # "using the X procedure" pattern requires "using" — should NOT match here
        self.assertEqual(sigs, [])

    def test_PASS_negation_does_not_fire(self):
        """Adversarial check: 'I won't apply BogusProc' — the regex
        REQUIRES 'I'll' / 'I will' / 'apply X to Y'. 'won't' doesn't
        match. This is more conservative than I initially assumed in
        the design notes."""
        output = "I won't apply BogusProc here because it doesn't fit the question."
        sigs = _detect_dispatch_misses(_ctx(raw_outputs=[output], procedure_names=self.PROCS))
        self.assertEqual(sigs, [],
                         "Negation 'won't apply' should NOT fire — regex requires I'll/will/apply-to")

    def test_PASS_past_tense_does_not_fire(self):
        """Adversarial check: 'I applied OldProcedure earlier' — the
        regex's `apply_intent` requires 'I'll' or 'I will' + 'apply'.
        'I applied' (past tense) doesn't match. Also more conservative
        than expected."""
        output = (
            "In an earlier turn I applied OldProcedure and got the answer. "
            "Now I'll just summarize."
        )
        sigs = _detect_dispatch_misses(_ctx(raw_outputs=[output], procedure_names=self.PROCS))
        self.assertEqual(sigs, [],
                         "Past-tense 'I applied X' should NOT fire — regex requires I'll/will")

    def test_PASS_comparative_invokes_real(self):
        """'Instead of FakeProc, I'll apply VerifyShortestPath' — apply matches
        VerifyShortestPath (real), so no miss signal."""
        output = "Instead of FakeProc, I'll apply VerifyShortestPath to this problem."
        sigs = _detect_dispatch_misses(_ctx(raw_outputs=[output], procedure_names=self.PROCS))
        self.assertEqual(sigs, [],
                         "Apply phrase points at a REAL procedure; no miss")

    @unittest.expectedFailure  # known FP, accepted for v1
    def test_FAIL_hypothetical_apply_x_to_y(self):
        """KNOWN FALSE POSITIVE: 'If we apply X to Y, we would get...'.
        The `apply X to Y` pattern matches regardless of 'If' prefix
        making this a hypothetical/conditional context, not an actual
        invocation. Current design accepts this FP because:
          (a) the signal message text is non-prescriptive ('if you meant
              to invoke...'), so the model can ignore it safely
          (b) negation-aware / conditional-aware parsing is out of scope
              for v1's regex-based predicate.

        Test stays visible via expectedFailure so it doesn't quietly
        get fixed by accident."""
        output = "If we apply HypotheticalProc to a graph with negative edges, we would get..."
        sigs = _detect_dispatch_misses(_ctx(raw_outputs=[output], procedure_names=self.PROCS))
        self.assertEqual(sigs, [],
                         "FALSE POSITIVE: 'If we apply X to Y' currently fires a miss signal")


class TestDispatchMissNudgeTruePositives(unittest.TestCase):
    """Real cases where the signal SHOULD fire."""

    PROCS = ["VerifyShortestPath", "VerifyNonNegativeEdges"]

    def test_typo_in_procedure_name(self):
        """The actual production case observed (sess_e1023f5801ca's iter 0
        had this typo in the PLAN section)."""
        output = "PLAN: I'll apply VerifyShortestPreprocess to check Dijkstra preconditions."
        sigs = _detect_dispatch_misses(_ctx(raw_outputs=[output], procedure_names=self.PROCS))
        self.assertEqual(len(sigs), 1)
        self.assertIn("VerifyShortestPreprocess", sigs[0].message)

    def test_multiple_typos_in_one_output(self):
        """Two different unknown names should both fire."""
        output = (
            "I'll apply BogusOne to the graph, then invoke BogusTwo for verification."
        )
        sigs = _detect_dispatch_misses(_ctx(raw_outputs=[output], procedure_names=self.PROCS))
        self.assertEqual(len(sigs), 2)
        names_in_signals = {s.metadata["mentioned_name"] for s in sigs}
        self.assertEqual(names_in_signals, {"BogusOne", "BogusTwo"})


# =========================================================================
# Real-session replay — does the pool fire spuriously on actual data?
# =========================================================================

SESSION_ROOT = Path(__file__).resolve().parent.parent.parent / "data" / "session_subgraphs"


class TestRealSessionReplay(unittest.TestCase):
    """Reconstruct MetaContext from each persisted session that has a
    __diag__ node (which carries raw_outputs + dispatch_summary), then
    fire the full meta-pool against that context. Asserts:
      - zero ERROR signals on sessions where there was no real contradiction
      - signals fired make sense given the session's actual content

    Older sessions without __diag__ are skipped — they don't carry the
    raw_outputs we need.
    """

    def _eligible_sessions(self) -> List[Path]:
        eligible = []
        if not SESSION_ROOT.exists():
            return eligible
        for sess in sorted(SESSION_ROOT.iterdir()):
            if not sess.is_dir():
                continue
            sg_path = sess / "subgraph.json"
            if not sg_path.exists():
                continue
            sg = json.loads(sg_path.read_text(encoding="utf-8"))
            if "__diag__" in sg.get("nodes", {}):
                eligible.append(sess)
        return eligible

    def _ctx_from_session(self, sess_path: Path) -> MetaContext:
        sg = json.loads((sess_path / "subgraph.json").read_text(encoding="utf-8"))
        diag = sg["nodes"]["__diag__"]
        raw_outputs = diag.get("raw_outputs", [])
        # Reconstruct minimal dispatch_outcomes from dispatch_summary
        outcomes = []
        for d in diag.get("dispatch_summary", []):
            outcomes.append(_StubOutcome(
                procedure_id=d.get("procedure_id"),
                object_id=d.get("object_id"),
                match=_StubMatch(
                    procedure_name=d.get("procedure_name", ""),
                    args_text=d.get("args_text"),
                    verb=d.get("verb", "apply_intent"),
                ),
                parent_object_id=d.get("parent_object_id"),
                error=d.get("error"),
            ))
        # Hydrate the stub session's subgraph nodes from the real persisted nodes
        stub = _StubSession()
        stub.subgraph.nodes = sg["nodes"]
        return MetaContext(
            session=stub,                                            # type: ignore[arg-type]
            budget=_StubBudgetTracker(),                             # type: ignore[arg-type]
            dispatch_outcomes=outcomes,                              # type: ignore[arg-type]
            raw_outputs=raw_outputs,
            anchor_ids=[],
            current_iteration=len(raw_outputs),
            previous_signals=[],
            # Match the production default procedure pool
            procedure_names=[
                "VerifyAlgorithmPreconditions",
                "VerifyNonNegativeEdges",
                "DetectNegativeCycle",
                "VerifyShortestPath",
            ],
        )

    def test_no_eligible_sessions_warns_but_skips(self):
        eligible = self._eligible_sessions()
        if not eligible:
            self.skipTest(
                "No persisted sessions with __diag__ node found "
                "(need at least one substrate run after diagnostics shipped)"
            )

    def test_pool_fires_zero_errors_on_real_sessions(self):
        """ASSERTION: ERROR-severity signals must not fire on existing
        real sessions. None of the persisted sessions had a genuine
        contradiction, so any ERROR is a false positive.

        Survives schema changes: if a future commit introduces an ERROR
        signal that fires on these sessions, this test surfaces it."""
        eligible = self._eligible_sessions()
        if not eligible:
            self.skipTest("No __diag__ sessions to replay")

        for sess in eligible:
            with self.subTest(session=sess.name):
                ctx = self._ctx_from_session(sess)
                pool = build_default_meta_pool()
                pool.run_hook("pre_iter", ctx)
                pool.run_hook("post_dispatch", ctx)
                pool.run_hook("end_of_session", ctx)

                errors = [s for s in pool.signal_stream if s.severity == "error"]
                self.assertEqual(
                    errors, [],
                    f"FALSE-POSITIVE ERROR signal fired on {sess.name}:\n  "
                    + "\n  ".join(f"{e.type}: {e.message[:100]}" for e in errors)
                )

    def test_clean_composer_session_produces_zero_signals(self):
        """ASSERTION: sess_e1023f5801ca is a healthy composer-with-children
        Phase-2A flow. It should produce zero meta-procedure signals
        because nothing went wrong.

        Locks in the conservative-predicate claim: the predicates do NOT
        fire spuriously on well-formed reasoning."""
        target = SESSION_ROOT / "sess_e1023f5801ca"
        if not target.exists():
            self.skipTest("Reference session sess_e1023f5801ca not present")

        ctx = self._ctx_from_session(target)
        pool = build_default_meta_pool()
        pool.run_hook("pre_iter", ctx)
        pool.run_hook("post_dispatch", ctx)
        pool.run_hook("end_of_session", ctx)
        self.assertEqual(
            pool.signal_stream, [],
            f"Clean composer session produced unexpected signals:\n  "
            + "\n  ".join(f"{s.severity} {s.type}: {s.message[:80]}"
                          for s in pool.signal_stream)
        )

    def test_stale_no_answer_session_fires_no_dispatch_signal(self):
        """ASSERTION: sess_7ddd8fc9d588 is a real coding question where
        the model produced 3 iterations of prose without ever emitting
        an <answer> block AND never invoked a procedure. This is
        EXACTLY what NoDispatchAfterThreshold is designed to catch.

        Verify it fires (and only it — no other signals)."""
        target = SESSION_ROOT / "sess_7ddd8fc9d588"
        if not target.exists():
            self.skipTest("Reference session sess_7ddd8fc9d588 not present")

        ctx = self._ctx_from_session(target)
        pool = build_default_meta_pool()
        pool.run_hook("pre_iter", ctx)
        pool.run_hook("post_dispatch", ctx)
        pool.run_hook("end_of_session", ctx)

        signals = pool.signal_stream
        types = [s.type for s in signals]
        self.assertIn("no_dispatch_stale", types,
                      "Stale 3-iteration session must fire no_dispatch_stale")
        # And only that — no false-positive cycles / contradictions / misses
        unexpected = [s for s in signals if s.type != "no_dispatch_stale"]
        self.assertEqual(
            unexpected, [],
            f"Unexpected meta signals fired on stale coding session:\n  "
            + "\n  ".join(f"{s.severity} {s.type}: {s.message[:80]}"
                          for s in unexpected)
        )


# =========================================================================
# Multi-MP interaction
# =========================================================================

class TestMultiMetaInteraction(unittest.TestCase):
    """Scenarios where 2-3 meta-procedures should fire on the same hook.
    Verify each fires once + signals don't interfere."""

    def test_cycle_and_dispatch_miss_fire_simultaneously(self):
        """One procedure is invoked 3x with same args (cycle) AND the
        model also mentioned a non-existent name in the last output."""
        # Build dispatch_outcomes with a triple-invocation of "VerifyShortestPath"
        outs = [
            _StubOutcome(
                procedure_id="proc_vsp",
                object_id=f"so_{i}",
                match=_StubMatch("VerifyShortestPath", "args_same"),
            )
            for i in range(3)
        ]
        # Most recent raw output also has a typo
        ctx = _ctx(
            dispatch_outcomes=outs,
            raw_outputs=["Now I'll apply BogusName to extend"],
            iteration=2,
            procedure_names=["VerifyShortestPath"],
        )
        pool = build_default_meta_pool()
        signals = pool.run_hook("post_dispatch", ctx)
        types = [s.type for s in signals]
        self.assertIn("cycle_detected", types)
        self.assertIn("dispatch_miss", types)


if __name__ == "__main__":
    unittest.main()
