"""Regression tests for finalization-mode gate (Fix 1 + Fix 2).

Real-world bug — cs4 Dijkstra run, sess on 2026-05-21:
  - turn 0: model invoked VerifyShortestPath (composer) which CALL'd two leaves
  - turn 1 (finalization): model emitted <answer> AND prose
    "I'll apply VerifyNonNegativeEdges to the instance and then use the
    VerifyShortestPath result..."
  - The dispatcher matched the prose, consumed an LLM call on bogus args,
    exhausted the budget (6/6), and created a junk session_object.

Fix 1 — reasoning_loop.run_reasoning(): skip dispatcher.scan when
dispatch_outcomes is already non-empty AND this turn's output contains
<answer>. The model has finalized; any "I'll apply X" prose is incidental.

Fix 2 — reasoning_loop._build_prompt(): drop the `# Available procedures`
section in finalization mode. Use a finalize-only directive that
explicitly forbids further invocation phrasings.

These regression tests deliberately test MULTIPLE prompt phrasings — the
model produces these phrasings in unpredictable ways, so the gate must
hold across the variation surface.
"""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from reasoning.budgets import Budgets
from reasoning.dispatcher import DispatchOutcome, PatternMatch
from reasoning.reasoning_loop import (
    ReasoningRequest,
    _build_prompt,
    run_reasoning,
)
from reasoning.procedures.verify_algorithm_preconditions import build_seed_procedure
from reasoning.procedures.verify_nonneg_edges import build_verify_nonneg_edges


GRAPH_PATH = "graphs/merged_graph.json"


class _ScriptedLLM:
    """Returns a sequence of canned outputs, one per call. Records prompts."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = 0
        self.prompts = []

    def __call__(self, prompt):
        self.prompts.append(prompt)
        self.calls += 1
        if not self.responses:
            return "(scripted LLM exhausted)"
        return self.responses.pop(0)


def _make_request(persist_root: Path) -> ReasoningRequest:
    return ReasoningRequest(
        question="Verify Dijkstra preconditions on a graph with edge b->c weight -1.",
        graph_id="cs4",
        graph_path=GRAPH_PATH,
        k_anchors=4,
        max_iterations=3,
        session_persist_root=persist_root,
        promotion_threshold=1,
        budgets=Budgets(max_llm_calls=6, max_total_tokens=8000),
    )


# ---- variation surface ------------------------------------------------- #
# Six different ways the model has been (or could realistically be)
# observed phrasing a re-invocation in finalization prose. Each one
# should be IGNORED by the finalization gate.
REINVOCATION_PROSE_VARIANTS = [
    # 1. Exact phrasing from the real cs4 run on 2026-05-21.
    "I'll apply VerifyNonNegativeEdges to the instance and then use the "
    "VerifyShortestPath result to compose the answer.",
    # 2. "I will" instead of "I'll".
    "I will apply VerifyNonNegativeEdges to the instance description.",
    # 3. "invoke" verb.
    "We can invoke VerifyNonNegativeEdges with the same args to double-check.",
    # 4. "using the X procedure" phrasing.
    "Drawing on the results from using the VerifyAlgorithmPreconditions procedure earlier.",
    # 5. Mid-sentence intent phrasing.
    "Given those findings, I'll apply VerifyAlgorithmPreconditions to confirm "
    "what we already know.",
    # 6. "Now apply X" — bare imperative.
    "Now apply VerifyNonNegativeEdges to the same instance.",
]


# ---- Fix 1: dispatcher gate -------------------------------------------- #

class TestFinalizationGateSkipsDispatch(unittest.TestCase):
    """When dispatch_outcomes is non-empty AND output contains <answer>,
    the dispatcher must NOT scan the output for new invocations."""

    def _run_two_turn_scenario(self, finalization_prose: str):
        """Standard scenario:
          turn 0 → model invokes VerifyAlgorithmPreconditions
          turn 1 → model emits <answer> + the supplied finalization_prose
        """
        turn_0 = (
            "<reasoning>\n"
            "GOAL: Check Dijkstra safety.\n"
            "PLAN: I'll apply VerifyAlgorithmPreconditions to algorithm "
            "Dijkstra, instance edge b->c weight -1.\n"
            "</reasoning>\n"
        )
        sub_body = (
            "ADD nonneg_edges TO state.preconditions_checked\n"
            "ADD nonneg_edges TO state.preconditions_violated\n"
            'SET state.evidence_for_violations.nonneg_edges = "b->c has weight -1"\n'
            "DONE\nnonneg violated\n"
        )
        turn_1 = (
            "<reasoning>\nKNOWN: nonneg violated.\n</reasoning>\n"
            f"{finalization_prose}\n"
            "<answer>No — Dijkstra is unsafe; use Bellman-Ford.</answer>"
        )

        def routing_stub(prompt: str) -> str:
            if "You are executing the VerifyAlgorithmPreconditions procedure." in prompt:
                return sub_body
            if "FINALIZATION MODE" in prompt:
                return turn_1
            return turn_0

        with tempfile.TemporaryDirectory() as td:
            req = _make_request(Path(td))
            return run_reasoning(req, routing_stub)

    def test_exact_real_world_prose_does_not_redispatch(self):
        """The exact prose from sess on 2026-05-21 cs4 run."""
        result = self._run_two_turn_scenario(REINVOCATION_PROSE_VARIANTS[0])

        # Exactly ONE top-level dispatch: the initial
        # VerifyAlgorithmPreconditions. No second dispatch from the
        # finalization prose.
        top_level = [
            o for o in result.dispatch_outcomes
            if o.parent_object_id is None
        ]
        self.assertEqual(
            len(top_level), 1,
            f"Expected 1 top-level dispatch, got {len(top_level)}: "
            f"{[o.match.procedure_name for o in top_level]}",
        )
        # Final answer survives.
        self.assertIn("Bellman-Ford", result.answer)
        # No budget exhaustion.
        self.assertIsNone(result.early_terminated_reason)

    def test_every_prose_variant_does_not_redispatch(self):
        """All 6 prose variants must be treated as inert in finalization mode."""
        for variant in REINVOCATION_PROSE_VARIANTS:
            with self.subTest(variant=variant[:60]):
                result = self._run_two_turn_scenario(variant)
                top_level = [
                    o for o in result.dispatch_outcomes
                    if o.parent_object_id is None
                ]
                self.assertEqual(
                    len(top_level), 1,
                    f"Variant {variant[:50]!r} produced {len(top_level)} "
                    f"top-level dispatches (expected 1): "
                    f"{[o.match.procedure_name for o in top_level]}",
                )

    def test_budget_stays_under_cap(self):
        """Real cs4 run hit 6/6. With the gate: 1 main + 1 sub + 1 finalize = 3."""
        result = self._run_two_turn_scenario(REINVOCATION_PROSE_VARIANTS[0])
        used = result.budget_usage["llm_calls"]["used"]
        self.assertLessEqual(
            used, 4,
            f"Expected ≤4 LLM calls, got {used}. The bug used 6/6.",
        )
        self.assertIsNone(result.early_terminated_reason)

    def test_no_junk_session_object_created(self):
        """The bad re-invocation in the real run created a session_object with
        violating_edges=[] (empty). With the gate, no such junk exists."""
        result = self._run_two_turn_scenario(REINVOCATION_PROSE_VARIANTS[0])

        nonneg_objects = [
            n for n in result.session_subgraph.nodes.values()
            if n.get("node_type") == "session_object"
            and n.get("name") == "VerifyNonNegativeEdges"
        ]
        # In this scenario VerifyNonNegativeEdges was never invoked at all,
        # so the count must be 0.
        self.assertEqual(
            len(nonneg_objects), 0,
            f"Junk VerifyNonNegativeEdges session_object(s) created: "
            f"{[s for s in nonneg_objects]}",
        )


class TestFinalizationGatePreservesValidDispatch(unittest.TestCase):
    """Fix 1 must not over-fire — valid dispatch patterns must still work."""

    def test_iter0_dispatch_fires_even_with_premature_answer(self):
        """Iter 0 has empty dispatch_outcomes at scan time; the gate
        requires dispatch_outcomes to ALREADY be non-empty. So a turn-0
        invocation must still fire even if the model also wrote <answer>."""
        turn_0 = (
            "<reasoning>\n"
            "PLAN: I'll apply VerifyAlgorithmPreconditions to Dijkstra.\n"
            "</reasoning>\n"
            "<answer>preliminary — subject to procedure result</answer>"
        )
        sub_body = (
            "ADD nonneg TO state.preconditions_checked\n"
            "DONE\nchecked\n"
        )
        turn_1 = "<reasoning>...</reasoning><answer>final with confirmation</answer>"

        stub = _ScriptedLLM([turn_0, sub_body, turn_1])

        with tempfile.TemporaryDirectory() as td:
            req = _make_request(Path(td))
            result = run_reasoning(req, stub)
            self.assertEqual(len(result.dispatch_outcomes), 1)
            self.assertEqual(
                result.dispatch_outcomes[0].match.procedure_name,
                "VerifyAlgorithmPreconditions",
            )

    def test_chained_dispatch_when_followup_has_no_answer(self):
        """If a follow-up iteration legitimately invokes another procedure
        and does NOT emit <answer>, the gate must NOT fire — chained
        dispatch across iterations stays supported."""
        turn_0 = "<reasoning>I'll apply VerifyAlgorithmPreconditions to Dijkstra.</reasoning>"
        sub_0 = "ADD a TO state.preconditions_checked\nDONE\nstep1"
        # No <answer> here.
        turn_1 = "<reasoning>I'll apply VerifyNonNegativeEdges to instance b->c -1.</reasoning>"
        sub_1 = "ADD b->c TO state.violating_edges\nDONE\nstep2"
        turn_2 = "<reasoning>...</reasoning><answer>final</answer>"

        stub = _ScriptedLLM([turn_0, sub_0, turn_1, sub_1, turn_2])

        with tempfile.TemporaryDirectory() as td:
            req = _make_request(Path(td))
            req.budgets = Budgets(max_llm_calls=10, max_total_tokens=8000)
            result = run_reasoning(req, stub)

            top_level = [o for o in result.dispatch_outcomes if o.parent_object_id is None]
            self.assertEqual(
                len(top_level), 2,
                "Chained dispatch across iters (no <answer> in middle iter) "
                "must still work — the gate only triggers when <answer> is present.",
            )

    def test_legitimate_invocation_in_iter0_with_no_answer_still_dispatches(self):
        """Standard iter-0 dispatch: model invokes, no <answer>. Must fire."""
        turn_0 = (
            "<reasoning>\nI'll apply VerifyAlgorithmPreconditions to Dijkstra.\n</reasoning>"
        )
        sub_body = "ADD x TO state.preconditions_checked\nDONE\n"
        turn_1 = "<reasoning>...</reasoning><answer>done</answer>"
        stub = _ScriptedLLM([turn_0, sub_body, turn_1])
        with tempfile.TemporaryDirectory() as td:
            req = _make_request(Path(td))
            result = run_reasoning(req, stub)
            self.assertEqual(len(result.dispatch_outcomes), 1)


# ---- Fix 2: prompt builder --------------------------------------------- #

def _empty_graph():
    class _G:
        def __init__(self):
            self.nodes = {}
    return _G()


def _make_outcome(name: str = "VerifyAlgorithmPreconditions") -> DispatchOutcome:
    match = PatternMatch(
        verb="apply_intent", procedure_name=name,
        args_text="Dijkstra", start=0, end=10,
    )
    return DispatchOutcome(
        match=match, procedure_id="proc_001", object_id="so_001",
        sub_prompt="...",
        sub_response="ADD x TO state.y\nDONE\nDijkstra fails preconditions.",
        mutations_applied=1,
    )


class TestFinalizationPromptBuilder(unittest.TestCase):
    """In finalization mode, the prompt MUST drop the procedure catalog
    and use a finalize-only directive that explicitly forbids invocation
    phrasings the model has been observed producing."""

    def test_no_procedures_section_when_dispatch_outcomes_present(self):
        proc = build_seed_procedure()
        nonneg_proc = build_verify_nonneg_edges()
        req = ReasoningRequest(question="Q?", graph_id="cs4", graph_path="(unused)")
        prompt = _build_prompt(
            req=req, graph=_empty_graph(), anchor_ids=[],
            procedure_pool=[proc, nonneg_proc],
            dispatch_outcomes=[_make_outcome()],
            iteration=1,
        )
        # Catalog header is gone.
        self.assertNotIn("# Available procedures", prompt)
        # No procedure entry headers leak either.
        self.assertNotIn("## VerifyNonNegativeEdges\nPurpose:", prompt)
        self.assertNotIn("To invoke one, write a phrase like", prompt)

    def test_directive_explicitly_forbids_invocation_phrasings(self):
        """The finalize directive must call out the specific phrasings
        the model has been observed using."""
        req = ReasoningRequest(question="Q?", graph_id="cs4", graph_path="(unused)")
        prompt = _build_prompt(
            req=req, graph=_empty_graph(), anchor_ids=[],
            procedure_pool=[build_seed_procedure()],
            dispatch_outcomes=[_make_outcome()],
            iteration=1,
        )
        self.assertIn("FINALIZATION MODE", prompt)
        self.assertIn("DO NOT invoke", prompt)
        # The exact phrasings the cs4 run produced are explicitly forbidden.
        self.assertIn("I'll apply", prompt)
        self.assertIn("using the", prompt)
        self.assertIn("invoke", prompt)

    def test_procedures_section_present_on_iter_0(self):
        """Initial turn (no dispatch yet) must still show the catalog."""
        req = ReasoningRequest(question="Q?", graph_id="cs4", graph_path="(unused)")
        prompt = _build_prompt(
            req=req, graph=_empty_graph(), anchor_ids=[],
            procedure_pool=[build_seed_procedure()],
            dispatch_outcomes=[],
            iteration=0,
        )
        self.assertIn("# Available procedures", prompt)
        self.assertIn("VerifyAlgorithmPreconditions", prompt)
        # Not the finalize directive.
        self.assertNotIn("FINALIZATION MODE", prompt)

    def test_dispatch_results_still_render_in_finalization(self):
        """The dispatch-results section is the WHOLE point of finalization
        — it must survive even though procedures section is dropped."""
        req = ReasoningRequest(question="Q?", graph_id="cs4", graph_path="(unused)")
        prompt = _build_prompt(
            req=req, graph=_empty_graph(), anchor_ids=[],
            procedure_pool=[build_seed_procedure()],
            dispatch_outcomes=[_make_outcome()],
            iteration=1,
        )
        self.assertIn("Results from procedures invoked so far", prompt)
        self.assertIn("Dijkstra fails preconditions", prompt)


# ---- Fix 4: render sub_outcomes ---------------------------------------- #

def _make_outcome_with_state(
    name: str,
    sub_response: str,
    *,
    object_id: str = "so_X",
    parent_object_id=None,
    sub_outcomes=None,
) -> DispatchOutcome:
    match = PatternMatch(
        verb="apply_intent", procedure_name=name,
        args_text="...", start=0, end=10,
    )
    o = DispatchOutcome(
        match=match, procedure_id=f"proc_{name}", object_id=object_id,
        sub_prompt="...", sub_response=sub_response,
        mutations_applied=sub_response.lower().count("\nadd ")
                          + sub_response.lower().count("\nset "),
        parent_object_id=parent_object_id,
    )
    if sub_outcomes:
        o.sub_outcomes = sub_outcomes
    return o


class TestFinalizationPromptRendersSubOutcomes(unittest.TestCase):
    """Fix 4: the finalization prompt's dispatch-results section must
    show child procedures' state too, not just the top-level summary.

    Without this, the model in finalization mode would re-invoke leaves
    whose state it cannot see (the bug we hit on cs4 2026-05-21)."""

    def test_child_summary_appears_in_prompt(self):
        """The cs4-style composer + 1 leaf shape. Child summary must render."""
        nonneg_child = _make_outcome_with_state(
            name="VerifyNonNegativeEdges",
            sub_response=(
                'ADD "b->c" TO state.checked_edges\n'
                'ADD "b->c" TO state.violating_edges\n'
                "DONE\nfound one negative edge\n"
            ),
            object_id="so_child_nonneg",
            parent_object_id="so_composer",
        )
        composer = _make_outcome_with_state(
            name="VerifyShortestPath",
            sub_response=(
                "CALL VerifyNonNegativeEdges WITH instance=...\n"
                "SET state.safe_to_apply = false\n"
                "DONE\nDijkstra unsafe\n"
            ),
            object_id="so_composer",
            sub_outcomes=[nonneg_child],
        )

        req = ReasoningRequest(question="Q?", graph_id="cs4", graph_path="(unused)")
        prompt = _build_prompt(
            req=req, graph=_empty_graph(), anchor_ids=[],
            procedure_pool=[build_seed_procedure()],
            dispatch_outcomes=[composer],
            iteration=1,
        )
        # Composer present
        self.assertIn("VerifyShortestPath", prompt)
        self.assertIn("Dijkstra unsafe", prompt)
        # Child also present
        self.assertIn("VerifyNonNegativeEdges", prompt)
        self.assertIn("found one negative edge", prompt)
        # Child state visible (this is what the model NEEDS to avoid
        # re-invoking the leaf — without it, the prompt only had the
        # composer summary, which is what triggered the cs4 bug).
        self.assertIn("violating_edges", prompt)
        self.assertIn("b->c", prompt)

    def test_multiple_children_all_appear(self):
        """Two-leaf composer (VerifyShortestPath for Bellman-Ford on a
        negative-cycle graph) — both children must render."""
        precond_child = _make_outcome_with_state(
            name="VerifyAlgorithmPreconditions",
            sub_response=(
                'ADD "nonneg" TO state.preconditions_checked\n'
                'ADD "nonneg" TO state.preconditions_violated\n'
                "DONE\nnonneg violated\n"
            ),
            object_id="so_precond",
            parent_object_id="so_composer",
        )
        cycle_child = _make_outcome_with_state(
            name="DetectNegativeCycle",
            sub_response=(
                'ADD "b->c->b" TO state.cycles_found\n'
                "SET state.has_negative_cycle = true\n"
                "DONE\nnegative cycle detected\n"
            ),
            object_id="so_cycle",
            parent_object_id="so_composer",
        )
        composer = _make_outcome_with_state(
            name="VerifyShortestPath",
            sub_response="DONE\nUnsafe due to negative cycle.\n",
            object_id="so_composer",
            sub_outcomes=[precond_child, cycle_child],
        )

        req = ReasoningRequest(question="Q?", graph_id="cs4", graph_path="(unused)")
        prompt = _build_prompt(
            req=req, graph=_empty_graph(), anchor_ids=[],
            procedure_pool=[build_seed_procedure()],
            dispatch_outcomes=[composer],
            iteration=1,
        )
        # Both children's findings are visible
        self.assertIn("VerifyAlgorithmPreconditions", prompt)
        self.assertIn("nonneg violated", prompt)
        self.assertIn("preconditions_violated", prompt)
        self.assertIn("DetectNegativeCycle", prompt)
        self.assertIn("negative cycle detected", prompt)
        self.assertIn("has_negative_cycle", prompt)

    def test_children_render_indented_under_parent(self):
        """Visual nesting: children should appear indented under their
        parent so the model sees the composition structure."""
        child = _make_outcome_with_state(
            name="VerifyNonNegativeEdges",
            sub_response='ADD "b->c" TO state.violating_edges\nDONE\nfound\n',
            object_id="so_child",
            parent_object_id="so_parent",
        )
        composer = _make_outcome_with_state(
            name="VerifyShortestPath",
            sub_response="DONE\ncomposer summary\n",
            object_id="so_parent",
            sub_outcomes=[child],
        )

        req = ReasoningRequest(question="Q?", graph_id="cs4", graph_path="(unused)")
        prompt = _build_prompt(
            req=req, graph=_empty_graph(), anchor_ids=[],
            procedure_pool=[build_seed_procedure()],
            dispatch_outcomes=[composer],
            iteration=1,
        )
        # Indented child header (2-space indent per nesting level)
        self.assertIn("  ## VerifyNonNegativeEdges", prompt)

    def test_long_list_state_is_truncated_to_5(self):
        """Defense against runaway state — long lists are truncated."""
        many_edges = "\n".join(
            f'ADD "edge_{i}" TO state.checked_edges' for i in range(20)
        )
        leaf = _make_outcome_with_state(
            name="VerifyNonNegativeEdges",
            sub_response=many_edges + "\nDONE\ndone\n",
            object_id="so_leaf",
        )
        req = ReasoningRequest(question="Q?", graph_id="cs4", graph_path="(unused)")
        prompt = _build_prompt(
            req=req, graph=_empty_graph(), anchor_ids=[],
            procedure_pool=[build_seed_procedure()],
            dispatch_outcomes=[leaf],
            iteration=1,
        )
        # First five edges visible, then truncation marker
        for i in range(5):
            self.assertIn(f"edge_{i}", prompt)
        # The 20th edge should not appear (truncated)
        self.assertNotIn("edge_19", prompt)
        # Explicit truncation marker
        self.assertIn("...", prompt)

    def test_state_with_only_set_commands_renders(self):
        """SET-only state (no ADD): the verdict-style outputs."""
        verdict_leaf = _make_outcome_with_state(
            name="VerifyShortestPath",
            sub_response=(
                "SET state.safe_to_apply = false\n"
                'SET state.verdict = "Dijkstra unsafe"\n'
                'SET state.recommended_alternative = "Bellman-Ford"\n'
                "DONE\nverdict written\n"
            ),
            object_id="so_verdict",
        )
        req = ReasoningRequest(question="Q?", graph_id="cs4", graph_path="(unused)")
        prompt = _build_prompt(
            req=req, graph=_empty_graph(), anchor_ids=[],
            procedure_pool=[build_seed_procedure()],
            dispatch_outcomes=[verdict_leaf],
            iteration=1,
        )
        self.assertIn("safe_to_apply", prompt)
        self.assertIn("false", prompt)
        self.assertIn("Bellman-Ford", prompt)


# ---- Fix 3: pre-LLM args validation ------------------------------------ #

# Variations of malformed args_text the model might produce. Each one
# should be rejected by validate_args BEFORE any LLM call.
BAD_ARGS_VARIANTS = [
    # 1. Exact prose from real cs4 run on 2026-05-21.
    ("VerifyNonNegativeEdges",
     "the instance and then use the VerifyShortestPath result to compose the answer."),
    # 2. Refers to another procedure's output as input.
    ("VerifyNonNegativeEdges",
     "based on what VerifyAlgorithmPreconditions found earlier"),
    # 3. Bare meta-text mentioning a sibling procedure.
    ("VerifyNonNegativeEdges",
     "see the results from DetectNegativeCycle"),
    # 4. Composer self-reference would be valid; THIS test uses cross-procedure.
    ("VerifyAlgorithmPreconditions",
     "Dijkstra after running VerifyShortestPath"),
]


class TestDispatcherValidateArgs(unittest.TestCase):
    """Fix 3: validate_args must reject obviously malformed args BEFORE
    creating a session object or consuming an LLM call."""

    def _dispatcher_with_pool(self):
        from reasoning.dispatcher import Dispatcher
        from reasoning.procedures.detect_negative_cycle import build_detect_negative_cycle
        from reasoning.procedures.verify_shortest_path import build_verify_shortest_path

        procs = [
            build_seed_procedure(),
            build_verify_nonneg_edges(),
            build_detect_negative_cycle(),
            build_verify_shortest_path(),
        ]
        return Dispatcher({p.id: p for p in procs}), procs

    def _match(self, name: str, args: str):
        return PatternMatch(
            verb="apply_intent", procedure_name=name,
            args_text=args, start=0, end=10,
        )

    def test_real_world_bad_args_rejected(self):
        dispatcher, procs = self._dispatcher_with_pool()
        proc = dispatcher.resolve_name("VerifyNonNegativeEdges")
        match = self._match(
            "VerifyNonNegativeEdges",
            "the instance and then use the VerifyShortestPath result to compose the answer.",
        )
        reason = dispatcher.validate_args(proc, match)
        self.assertIsNotNone(reason)
        self.assertIn("VerifyShortestPath", reason)

    def test_all_bad_args_variants_rejected(self):
        """Each variant should be rejected and the reason should name
        the offending other-procedure."""
        dispatcher, _ = self._dispatcher_with_pool()
        for proc_name, bad_args in BAD_ARGS_VARIANTS:
            with self.subTest(proc=proc_name, args=bad_args[:50]):
                proc = dispatcher.resolve_name(proc_name)
                match = self._match(proc_name, bad_args)
                reason = dispatcher.validate_args(proc, match)
                self.assertIsNotNone(
                    reason,
                    f"args {bad_args!r} should be rejected for {proc_name}",
                )

    def test_empty_args_with_required_inputs_rejected(self):
        dispatcher, _ = self._dispatcher_with_pool()
        proc = dispatcher.resolve_name("VerifyNonNegativeEdges")
        match = self._match("VerifyNonNegativeEdges", "")
        reason = dispatcher.validate_args(proc, match)
        self.assertIsNotNone(reason)
        self.assertIn("empty", reason.lower())

    def test_valid_args_accepted(self):
        """A normal graph description should pass."""
        dispatcher, _ = self._dispatcher_with_pool()
        proc = dispatcher.resolve_name("VerifyNonNegativeEdges")
        match = self._match(
            "VerifyNonNegativeEdges",
            "directed graph with edges (a->b, weight 3), (b->c, weight -1)",
        )
        self.assertIsNone(dispatcher.validate_args(proc, match))

    def test_structured_kv_args_accepted(self):
        """Composer-style key=value args should pass."""
        dispatcher, _ = self._dispatcher_with_pool()
        proc = dispatcher.resolve_name("VerifyAlgorithmPreconditions")
        match = self._match(
            "VerifyAlgorithmPreconditions",
            'algorithm_name="Dijkstra" instance_description="edge b->c weight -1"',
        )
        self.assertIsNone(dispatcher.validate_args(proc, match))

    def test_self_name_reference_in_args_is_not_rejected(self):
        """Mentioning the SAME procedure name in its own args is fine —
        only DIFFERENT procedure names indicate meta-prose."""
        dispatcher, _ = self._dispatcher_with_pool()
        proc = dispatcher.resolve_name("VerifyAlgorithmPreconditions")
        # Args that reference the same procedure: silly but not meta-text.
        match = self._match(
            "VerifyAlgorithmPreconditions",
            "Dijkstra; running VerifyAlgorithmPreconditions on the graph",
        )
        self.assertIsNone(dispatcher.validate_args(proc, match))


class TestDispatcherInvokeRejectsBadArgs(unittest.TestCase):
    """The reject in validate_args must propagate through invoke() —
    no session_object, no LLM call."""

    def test_bad_args_invoke_returns_error_outcome(self):
        """invoke() should return a DispatchOutcome with error= set and
        NO sub_response (no LLM call happened)."""
        from reasoning.dispatcher import Dispatcher
        from reasoning.session_subgraph import SessionSubgraphController

        dispatcher = Dispatcher({
            build_seed_procedure().id: build_seed_procedure(),
            build_verify_nonneg_edges().id: build_verify_nonneg_edges(),
        })
        session = SessionSubgraphController("sess_test", "Q?", "cs4")

        # Track LLM calls
        llm_calls = []
        def stub_llm(prompt):
            llm_calls.append(prompt)
            return "ADD x TO state.y\nDONE\n"

        match = PatternMatch(
            verb="apply_intent",
            procedure_name="VerifyNonNegativeEdges",
            args_text="the instance and then use the VerifyAlgorithmPreconditions result",
            start=0, end=10,
        )
        outcome = dispatcher.invoke(match, session, stub_llm, budget=None)

        # LLM was NOT called
        self.assertEqual(len(llm_calls), 0)
        # Outcome reports the validation failure
        self.assertIsNotNone(outcome.error)
        self.assertIn("args validation failed", outcome.error)
        # No session_object was created
        session_objects = [
            n for n in session.subgraph.nodes.values()
            if isinstance(n, dict) and n.get("node_type") == "session_object"
        ]
        self.assertEqual(len(session_objects), 0)

    def test_bad_args_invoke_does_not_consume_budget(self):
        """When validation fails, the LLM call budget axis should not
        be consumed. This is the user-facing point of Fix 3."""
        from reasoning.dispatcher import Dispatcher
        from reasoning.session_subgraph import SessionSubgraphController
        from reasoning.budgets import BudgetTracker, Budgets

        dispatcher = Dispatcher({
            build_seed_procedure().id: build_seed_procedure(),
            build_verify_nonneg_edges().id: build_verify_nonneg_edges(),
        })
        session = SessionSubgraphController("sess_test", "Q?", "cs4")
        budget = BudgetTracker(Budgets(max_llm_calls=3))

        def stub_llm(prompt):
            return "ADD x TO state.y\nDONE\n"

        match = PatternMatch(
            verb="apply_intent",
            procedure_name="VerifyNonNegativeEdges",
            args_text="see VerifyAlgorithmPreconditions result",
            start=0, end=10,
        )
        outcome = dispatcher.invoke(match, session, stub_llm, budget=budget)
        # Outcome is an error
        self.assertIsNotNone(outcome.error)
        # llm_call axis was not consumed
        self.assertEqual(budget.summary()["llm_calls"]["used"], 0)


if __name__ == "__main__":
    unittest.main()
