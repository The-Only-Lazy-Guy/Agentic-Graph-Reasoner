"""Tests for reasoning/reasoning_loop.py.

Uses a stub LLM to simulate the model. Validates the orchestration:
prompt build, dispatch fire, follow-up turn, answer extraction,
session persistence, consolidation decisions.

The actual model quality is out of scope for unit tests — that's
1.9 integration.
"""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from graph_core import Edge, MemoryGraph, Node
from reasoning.budgets import Budgets
from reasoning.activation import FrameItem, GraphTaskFrame
from reasoning.reasoning_loop import (
    ReasoningRequest,
    _build_prompt,
    _extract_after_done,
    _extract_blocks,
    _substrate_v2_checker_plugins,
    _substrate_v2_initial_signals,
    run_reasoning,
)
from reasoning.procedures.verify_algorithm_preconditions import build_seed_procedure


def _make_request(graph_path: str, persist_root: Path) -> ReasoningRequest:
    return ReasoningRequest(
        question="Can A-star be trusted with one negative edge?",
        graph_id="cs4",
        graph_path=graph_path,
        k_anchors=8,
        max_iterations=3,
        session_persist_root=persist_root,
        promotion_threshold=1,                    # let promotion happen for the test
        budgets=Budgets(max_llm_calls=6, max_total_tokens=4000),
    )


class _ScriptedLLM:
    """Returns a sequence of canned outputs, one per call."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = 0

    def __call__(self, prompt: str) -> str:
        self.calls += 1
        if not self.responses:
            return "(scripted LLM exhausted; returning empty)"
        return self.responses.pop(0)


class TestExtractBlocks(unittest.TestCase):
    def test_last_block_wins(self):
        text = (
            "<reasoning>placeholder</reasoning>"
            "<answer>placeholder answer</answer>"
            "Now the real output:\n"
            "<reasoning>real reasoning</reasoning>"
            "<answer>real answer</answer>"
        )
        r, a = _extract_blocks(text)
        self.assertEqual(r, "real reasoning")
        self.assertEqual(a, "real answer")

    def test_missing_blocks_return_empty(self):
        r, a = _extract_blocks("no markup here")
        self.assertEqual(r, "")
        self.assertEqual(a, "")


class TestExtractAfterDone(unittest.TestCase):
    def test_extracts_summary_after_done_marker(self):
        sub = (
            "ADD a TO state.x\n"
            "DONE\n"
            "This is the summary line.\n"
        )
        self.assertEqual(_extract_after_done(sub), "This is the summary line.")

    def test_no_done_returns_empty(self):
        self.assertEqual(_extract_after_done("just commands no done"), "")


class TestPromptBuilder(unittest.TestCase):
    def test_initial_prompt_includes_procedures_section(self):
        from reasoning.schemas import ProcedureNode, Provenance

        # Minimal graph stub
        class _G:
            def __init__(self):
                self.nodes = {}
        graph = _G()
        proc = build_seed_procedure()

        req = ReasoningRequest(
            question="Q?", graph_id="cs4", graph_path="(unused)",
        )
        prompt = _build_prompt(
            req=req, graph=graph, anchor_ids=[],
            procedure_pool=[proc],
            dispatch_outcomes=[],
            iteration=0,
        )
        self.assertIn("VerifyAlgorithmPreconditions", prompt)
        self.assertIn("Purpose:", prompt)
        self.assertIn("ABSOLUTE RULES", prompt)
        self.assertIn("Question: Q?", prompt)

    def test_task_frame_without_procedure_suggestions_hides_catalog(self):
        class _G:
            def __init__(self):
                self.nodes = {}

        proc = build_seed_procedure()
        frame = GraphTaskFrame(
            session_id="sess_test",
            suggested_structures=[
                FrameItem(
                    item_id="fi_seg",
                    kind="answer_requirement",
                    text="Use a segment tree.",
                    priority=90,
                    source_signal_ids=[],
                )
            ],
        )
        req = ReasoningRequest(question="Q?", graph_id="cs4", graph_path="(unused)")
        prompt = _build_prompt(
            req=req,
            graph=_G(),
            anchor_ids=[],
            procedure_pool=[proc],
            dispatch_outcomes=[],
            iteration=0,
            task_frame=frame,
        )
        self.assertNotIn("# Available procedures", prompt)
        self.assertNotIn("VerifyAlgorithmPreconditions", prompt)
        self.assertIn("No procedure catalog is available", prompt)
        self.assertIn("Use a segment tree.", prompt)

    def test_task_frame_with_procedure_suggestions_keeps_catalog(self):
        class _G:
            def __init__(self):
                self.nodes = {}

        proc = build_seed_procedure()
        frame = GraphTaskFrame(
            session_id="sess_test",
            procedure_suggestions=[
                FrameItem(
                    item_id="fi_proc",
                    kind="procedure_suggestion",
                    text="Procedure may apply: VerifyAlgorithmPreconditions.",
                    priority=70,
                    source_signal_ids=[],
                )
            ],
        )
        req = ReasoningRequest(question="Q?", graph_id="cs4", graph_path="(unused)")
        prompt = _build_prompt(
            req=req,
            graph=_G(),
            anchor_ids=[],
            procedure_pool=[proc],
            dispatch_outcomes=[],
            iteration=0,
            task_frame=frame,
        )
        self.assertIn("# Available procedures", prompt)
        self.assertIn("VerifyAlgorithmPreconditions", prompt)

    def test_followup_prompt_includes_dispatch_results(self):
        from reasoning.dispatcher import DispatchOutcome, PatternMatch

        class _G:
            def __init__(self):
                self.nodes = {}

        proc = build_seed_procedure()
        match = PatternMatch(
            verb="apply_intent", procedure_name="VerifyAlgorithmPreconditions",
            args_text="A-star", start=0, end=10,
        )
        outcome = DispatchOutcome(
            match=match, procedure_id="proc_001", object_id="so_001",
            sub_prompt="...", sub_response="ADD x TO state.y\nDONE\nA-star fails preconditions.",
            mutations_applied=1,
        )

        req = ReasoningRequest(question="Q?", graph_id="cs4", graph_path="(unused)")
        prompt = _build_prompt(
            req=req, graph=_G(), anchor_ids=[],
            procedure_pool=[proc],
            dispatch_outcomes=[outcome],
            iteration=1,
        )
        self.assertIn("Results from procedures invoked so far", prompt)
        self.assertIn("A-star fails preconditions", prompt)
        self.assertIn("Mutations applied: 1", prompt)


class TestEndToEndWithStubLLM(unittest.TestCase):
    """Run the full loop against the real graphs/empty_graph_for_tests.json with a stub LLM."""

    GRAPH_PATH = "graphs/empty_graph_for_tests.json"

    def test_invokes_procedure_and_synthesizes_answer(self):
        """Model invokes the procedure on turn 1, gets results, answers on turn 2."""
        turn_1 = (
            "<reasoning>\n"
            "GOAL: Decide if A-star works with one negative edge.\n"
            "PLAN: I'll apply VerifyAlgorithmPreconditions to A-star on a graph with edge b->c weight -1.\n"
            "</reasoning>\n"
        )
        # Sub-LLM output during dispatch (the procedure body's response)
        sub_response = (
            "ADD nonneg_edges TO state.preconditions_checked\n"
            "ADD acyclic TO state.preconditions_checked\n"
            "ADD nonneg_edges TO state.preconditions_violated\n"
            'SET state.evidence_for_violations.nonneg_edges = "Edge b->c has weight -1"\n'
            "DONE\n"
            "A-star requires nonneg edges; the b->c -1 edge violates that. Use Bellman-Ford.\n"
        )
        turn_2 = (
            "<reasoning>\n"
            "GOAL: Decide if A-star works with one negative edge.\n"
            "KNOWN: nonneg-edge requirement is violated by b->c weight -1.\n"
            "PLAN: state the verdict; recommend Bellman-Ford.\n"
            "</reasoning>\n"
            "<answer>\n"
            "No — a single negative edge breaks A-star's greedy settlement step. "
            "Use Bellman-Ford for graphs with negative edges (no negative cycle).\n"
            "</answer>"
        )
        stub = _ScriptedLLM([turn_1, sub_response, turn_2])

        with tempfile.TemporaryDirectory() as td:
            req = _make_request(self.GRAPH_PATH, Path(td))
            req.question = "A-star on graph with edge b->c weight -1."
            result = run_reasoning(req, stub)

            # Answer extracted from turn 2
            self.assertIn("Bellman-Ford", result.answer)
            self.assertIn("negative edge", result.answer.lower())
            # Procedure was invoked once
            self.assertEqual(len(result.dispatch_outcomes), 1)
            self.assertEqual(result.dispatch_outcomes[0].mutations_applied, 4)
            # Two main-reasoner LLM calls + one sub-LLM = 3 budget consumed
            self.assertEqual(result.budget_usage["llm_calls"]["used"], 3)
            # Session subgraph persisted
            self.assertTrue((result.session_subgraph_path / "subgraph.json").exists())
            self.assertTrue((result.session_subgraph_path / "audit_log.jsonl").exists())
            # Consolidation produced at least one decision
            self.assertGreater(len(result.consolidation_decisions), 0)

    def test_direct_answer_no_invocation(self):
        """Model answers immediately without invoking any procedure."""
        turn_1 = (
            "<reasoning>\n"
            "GOAL: Q\n"
            "KNOWN: facts\n"
            "PLAN: answer directly\n"
            "</reasoning>\n"
            "<answer>Direct answer with no procedure call.</answer>"
        )
        stub = _ScriptedLLM([turn_1])

        with tempfile.TemporaryDirectory() as td:
            req = _make_request(self.GRAPH_PATH, Path(td))
            req.question = "A-star on graph with edge b->c weight -1."
            result = run_reasoning(req, stub)
            self.assertEqual(result.answer, "Direct answer with no procedure call.")
            self.assertEqual(len(result.dispatch_outcomes), 0)
            self.assertEqual(result.budget_usage["llm_calls"]["used"], 1)
            self.assertEqual(result.iterations_completed, 1)

    def test_micro_controller_finalizes_known_question(self):
        graph = MemoryGraph(
            nodes={
                "a-star_requires_nonnegative_edge_weights": Node(
                    id="a-star_requires_nonnegative_edge_weights",
                    node_type="claim",
                    text="A-star's algorithm requires nonnegative edge weights for its greedy settlement logic to remain correct.",
                ),
                "negative_edge_counterexample_test_apply": Node(
                    id="negative_edge_counterexample_test_apply",
                    node_type="application",
                    text="A single negative edge can produce a counterexample where A-star settles a vertex too early and returns the wrong shortest path.",
                ),
                "bellman_ford_handles_negative_edges": Node(
                    id="bellman_ford_handles_negative_edges",
                    node_type="claim",
                    text="Bellman-Ford handles negative edge weights by repeated relaxation and is the safe alternative when negative edges exist.",
                ),
            },
            edges=[
                Edge(
                    src="a-star_requires_nonnegative_edge_weights",
                    dst="negative_edge_counterexample_test_apply",
                    relation="support",
                ),
                Edge(
                    src="negative_edge_counterexample_test_apply",
                    dst="bellman_ford_handles_negative_edges",
                    relation="support",
                ),
            ],
        )
        stub = _ScriptedLLM([
            "<reasoning>The controller already filled verdict, reason, alternative, and caveat from the local working set.</reasoning>"
            "<answer>No. A-star cannot be trusted with one negative edge; use Bellman-Ford instead.</answer>"
        ])

        with tempfile.TemporaryDirectory() as td:
            graph_path = Path(td) / "micro_graph.json"
            graph.save_json(graph_path)
            req = ReasoningRequest(
                question="Can A-star be trusted with one negative edge?",
                graph_id="micro_graph",
                graph_path=str(graph_path),
                k_anchors=4,
                max_iterations=3,
                session_persist_root=Path(td),
                budgets=Budgets(max_llm_calls=4, max_total_tokens=4000),
            )
            result = run_reasoning(req, stub)

            self.assertIn("Bellman-Ford", result.answer)
            self.assertEqual(len(result.dispatch_outcomes), 0)
            self.assertTrue(result.audit_summary.get("micro_controller"))
            self.assertEqual(result.audit_summary.get("controller_task_family"), "algorithm_applicability")
            self.assertGreaterEqual(result.audit_summary.get("subgoal_reuse_count", 0), 1)
            self.assertEqual(result.budget_usage["llm_calls"]["used"], 0)

    def test_double_dispatch_in_one_turn_reuses_object(self):
        """Regression: if the model mentions the same procedure twice in one
        turn, only ONE SessionObjectNode should be created, not two.

        Real-world evidence: sess_862af617e699 / sess_a9e21797680e produced
        two session_objects with divergent state for the same A-star
        question. The reasoning loop now dedupes within a turn AND reuses
        existing objects across turns.
        """
        # Turn 1: model mentions the procedure name twice
        turn_1 = (
            "<reasoning>\n"
            "PLAN: I'll apply VerifyAlgorithmPreconditions to A-star. "
            "We'll also re-verify using the VerifyAlgorithmPreconditions procedure "
            "to be thorough.\n"
            "</reasoning>\n"
        )
        sub_response = (
            "ADD nonneg_edges TO state.preconditions_checked\n"
            "ADD nonneg_edges TO state.preconditions_violated\n"
            "DONE\nverified\n"
        )
        turn_2 = (
            "<reasoning>...</reasoning>"
            "<answer>Use Bellman-Ford.</answer>"
        )
        stub = _ScriptedLLM([turn_1, sub_response, turn_2])

        with tempfile.TemporaryDirectory() as td:
            req = _make_request(self.GRAPH_PATH, Path(td))
            req.question = "A-star on graph with edge b->c weight -1."
            result = run_reasoning(req, stub)

            # Exactly ONE session object for VerifyAlgorithmPreconditions
            session_objects = [
                n for n in result.session_subgraph.nodes.values()
                if n.get("node_type") == "session_object"
            ]
            self.assertEqual(
                len(session_objects), 1,
                f"Expected 1 session object, got {len(session_objects)}: "
                f"{[s['id'] for s in session_objects]}",
            )
            # And only ONE invocation outcome recorded (the dedupe in scan
            # AND in the loop's invoked_this_iteration set, working together)
            self.assertEqual(len(result.dispatch_outcomes), 1)

    def test_repeated_invocation_across_turns_reuses_same_object(self):
        """If the model invokes the same procedure on turn 1 and again on
        turn 2 (refining the state), the loop should reuse the existing
        SessionObjectNode rather than spawning a fresh one."""
        turn_1 = "<reasoning>I'll apply VerifyAlgorithmPreconditions to A-star.</reasoning>"
        sub_1 = "ADD nonneg TO state.preconditions_checked\nDONE\nstep 1"
        turn_2 = "<reasoning>I'll apply VerifyAlgorithmPreconditions to acyclic.</reasoning>"
        sub_2 = "ADD acyclic TO state.preconditions_checked\nDONE\nstep 2"
        turn_3 = "<reasoning>...</reasoning><answer>Done.</answer>"
        stub = _ScriptedLLM([turn_1, sub_1, turn_2, sub_2, turn_3])

        with tempfile.TemporaryDirectory() as td:
            req = _make_request(self.GRAPH_PATH, Path(td))
            req.question = "A-star on graph with edge b->c weight -1."
            req.budgets = Budgets(max_llm_calls=8)
            result = run_reasoning(req, stub)

            sos = [n for n in result.session_subgraph.nodes.values()
                   if n.get("node_type") == "session_object"]
            self.assertEqual(len(sos), 1, "Same procedure across turns must reuse the object")
            # The single session object should have accumulated both ADDs
            self.assertEqual(
                sos[0]["state"]["preconditions_checked"],
                ["nonneg", "acyclic"],
            )

    def test_end_to_end_composition_through_reasoning_loop(self):
        """Phase 2A integration: top-level reasoner invokes the composer,
        composer's body emits CALL commands for two leaves, the reasoning
        loop's recursive dispatcher creates child session_objects, the
        follow-up turn produces the final answer.

        This is what the acceptance criterion §11 #1+#2 require: a real
        end-to-end composition flow through run_reasoning() (not just the
        dispatcher), producing a session subgraph with parent + 2 children
        and 2 sub_invocation_of edges.
        """
        # Turn 1: main reasoner invokes the composer
        turn_1 = (
            "<reasoning>\n"
            "GOAL: Decide if A-star is safe on the user's graph.\n"
            "PLAN: I'll apply VerifyShortestPath to A-star on a graph with edge b->c weight -1.\n"
            "</reasoning>\n"
        )
        # Sub-LLM #1: composer's body output — emits two CALL commands + DONE
        composer_body = (
            "CALL VerifyAlgorithmPreconditions WITH algorithm_name=A-star "
            "instance_description=edge b->c has weight -1\n"
            "CALL VerifyNonNegativeEdges WITH instance_description=edge b->c has weight -1\n"
            "DONE\n"
            "Composition: ran two sub-checks; nonneg violated.\n"
        )
        # Sub-LLM #2: VerifyAlgorithmPreconditions response
        precond_body = (
            "ADD nonneg_edges TO state.preconditions_checked\n"
            "ADD nonneg_edges TO state.preconditions_violated\n"
            "DONE\nnonneg violated\n"
        )
        # Sub-LLM #3: VerifyNonNegativeEdges response
        nonneg_body = (
            "ADD b->c TO state.checked_edges\n"
            "ADD b->c TO state.violating_edges\n"
            "DONE\nfound one negative edge\n"
        )
        # Turn 2 (follow-up): main reasoner produces the final answer
        turn_2 = (
            "<reasoning>\n"
            "KNOWN: composer verified A-star is unsafe (b->c weight -1).\n"
            "PLAN: state the verdict.\n"
            "</reasoning>\n"
            "<answer>\n"
            "No, A-star is not safe here: the edge b->c has weight -1. "
            "Use Bellman-Ford instead.\n"
            "</answer>"
        )

        # ScriptedLLM dispatches by prompt content. Order of checks matters:
        # sub-LLM procedure-body prompts are most specific, then follow-up
        # turn (has dispatch-results section), then initial turn.
        def routing_stub(prompt: str) -> str:
            # Sub-LLM procedure-body prompts
            if "You are executing the VerifyShortestPath procedure." in prompt:
                return composer_body
            if "You are executing the VerifyAlgorithmPreconditions procedure." in prompt:
                return precond_body
            if "You are executing the VerifyNonNegativeEdges procedure." in prompt:
                return nonneg_body
            # Follow-up reasoner turn (after dispatch produced results)
            if "The procedure invocations above produced" in prompt:
                return turn_2
            # Initial reasoner turn (initial directive present)
            if "ABSOLUTE RULES" in prompt:
                return turn_1
            return "DONE"

        with tempfile.TemporaryDirectory() as td:
            req = _make_request(self.GRAPH_PATH, Path(td))
            # Bump budget so the recursive composition has headroom
            req.budgets = Budgets(
                max_llm_calls=20,
                max_composition_fan_out=5,
                max_recursion_depth=4,
                max_total_tokens=8000,
            )
            result = run_reasoning(req, routing_stub)

            # Final answer extracted from turn 2
            self.assertIn("Bellman-Ford", result.answer)
            self.assertIn("A-star", result.answer)

            # Composition shape: 1 top-level + 2 children = 3 session_objects total
            session_objects = [
                n for n in result.session_subgraph.nodes.values()
                if n.get("node_type") == "session_object"
            ]
            self.assertEqual(
                len(session_objects), 3,
                f"Expected 3 session_objects (composer + 2 children), got "
                f"{len(session_objects)}: {[s['name'] for s in session_objects]}",
            )

            # The composer ran
            composer_so = next(
                s for s in session_objects if s["name"] == "VerifyShortestPath"
            )
            self.assertIsNotNone(composer_so)

            # Both children ran
            child_names = {s["name"] for s in session_objects if s["name"] != "VerifyShortestPath"}
            self.assertEqual(
                child_names,
                {"VerifyAlgorithmPreconditions", "VerifyNonNegativeEdges"},
            )

            # Sub_invocation_of edges record the call tree
            from reasoning.composition import SUB_INVOCATION_OF
            sub_edges = [
                e for e in result.session_subgraph.edges
                if e.relation == SUB_INVOCATION_OF
            ]
            self.assertEqual(len(sub_edges), 2)
            # Both edges point at the composer
            for edge in sub_edges:
                self.assertEqual(edge.dst, composer_so["id"])

            # Children's state is populated independently (mutation independence #6)
            nonneg_so = next(s for s in session_objects if s["name"] == "VerifyNonNegativeEdges")
            self.assertEqual(nonneg_so["state"]["violating_edges"], ["b->c"])

            precond_so = next(s for s in session_objects if s["name"] == "VerifyAlgorithmPreconditions")
            self.assertEqual(precond_so["state"]["preconditions_violated"], ["nonneg_edges"])

            # Budget used: 1 top-level reasoner + 1 composer body + 2 leaf bodies +
            # 1 follow-up reasoner = 5 LLM calls minimum
            self.assertGreaterEqual(result.budget_usage["llm_calls"]["used"], 5)
            # And we didn't blow any budget
            self.assertIsNone(result.early_terminated_reason)

    def test_vague_procedure_args_recover_from_original_question(self):
        question = (
            "I have a directed graph with edges (a->b, weight 3), "
            "(b->c, weight -1), (a->c, weight 5). I'm planning to run "
            "A-star. Use the VerifyShortestPath procedure before answering."
        )
        turn_1 = (
            "<reasoning>I'll apply VerifyShortestPath to A-star and this "
            "directed weighted instance.</reasoning>"
        )
        composer_body = (
            "CALL VerifyAlgorithmPreconditions WITH algorithm_name=A-star "
            "instance_description=this directed weighted instance\n"
            "CALL VerifyNonNegativeEdges WITH instance_description=this directed weighted instance\n"
            "SET state.safe_to_apply = false\n"
            "SET state.verdict = \"negative edge found\"\n"
            "SET state.recommended_alternative = \"Bellman-Ford\"\n"
            "DONE\n"
        )
        precond_body = (
            "ADD nonnegative_edge_weights TO state.preconditions_checked\n"
            "ADD nonnegative_edge_weights TO state.preconditions_violated\n"
            "DONE\nnegative edge violates A-star\n"
        )
        nonneg_body = (
            "ADD a->b TO state.checked_edges\n"
            "ADD b->c TO state.checked_edges\n"
            "ADD b->c TO state.violating_edges\n"
            "ADD a->c TO state.checked_edges\n"
            "DONE\nfound b->c\n"
        )
        turn_2 = "<reasoning>Use procedure results.</reasoning><answer>Use Bellman-Ford.</answer>"

        def routing_stub(prompt: str) -> str:
            concrete = (
                'instance_description="directed graph with edges (a->b, weight 3), '
                '(b->c, weight -1), (a->c, weight 5)"'
            )
            if "You are executing the VerifyShortestPath procedure." in prompt:
                self.assertIn(concrete, prompt)
                return composer_body
            if "You are executing the VerifyAlgorithmPreconditions procedure." in prompt:
                self.assertIn(concrete, prompt)
                return precond_body
            if "You are executing the VerifyNonNegativeEdges procedure." in prompt:
                self.assertIn(concrete, prompt)
                return nonneg_body
            if "The procedure invocations above produced" in prompt:
                return turn_2
            return turn_1

        with tempfile.TemporaryDirectory() as td:
            req = _make_request(self.GRAPH_PATH, Path(td))
            req.question = question
            req.budgets = Budgets(max_llm_calls=20, max_composition_fan_out=5)
            result = run_reasoning(req, routing_stub)
            nonneg_so = next(
                n for n in result.session_subgraph.nodes.values()
                if n.get("name") == "VerifyNonNegativeEdges"
            )
            self.assertEqual(nonneg_so["state"]["checked_edges"], ["a->b", "b->c", "a->c"])
            self.assertEqual(nonneg_so["state"]["violating_edges"], ["b->c"])

    def test_concrete_procedure_args_pass_through(self):
        concrete_args = (
            'algorithm_name=A-star instance_description="directed weighted graph '
            'with edges a->b weight 3, b->c weight -1, a->c weight 5"'
        )
        turn_1 = f"<reasoning>I'll apply VerifyShortestPath to {concrete_args}</reasoning>"
        composer_body = (
            f"CALL VerifyAlgorithmPreconditions WITH {concrete_args}\n"
            'CALL VerifyNonNegativeEdges WITH instance_description="directed weighted graph '
            'with edges a->b weight 3, b->c weight -1, a->c weight 5"\n'
            "DONE\n"
        )
        nonneg_body = (
            "ADD a->b TO state.checked_edges\n"
            "ADD b->c TO state.checked_edges\n"
            "ADD b->c TO state.violating_edges\n"
            "ADD a->c TO state.checked_edges\n"
            "DONE\n"
        )
        turn_2 = "<reasoning>Done.</reasoning><answer>A-star is unsafe.</answer>"

        def routing_stub(prompt: str) -> str:
            if "You are executing the VerifyShortestPath procedure." in prompt:
                self.assertIn(f"Invocation args: {concrete_args}", prompt)
                return composer_body
            if "You are executing the VerifyAlgorithmPreconditions procedure." in prompt:
                return "ADD nonnegative_edge_weights TO state.preconditions_violated\nDONE\n"
            if "You are executing the VerifyNonNegativeEdges procedure." in prompt:
                self.assertIn(
                    'Invocation args: instance_description="directed weighted graph '
                    'with edges a->b weight 3, b->c weight -1, a->c weight 5"',
                    prompt,
                )
                return nonneg_body
            if "The procedure invocations above produced" in prompt:
                return turn_2
            return turn_1

        with tempfile.TemporaryDirectory() as td:
            req = _make_request(self.GRAPH_PATH, Path(td))
            req.budgets = Budgets(max_llm_calls=20, max_composition_fan_out=5)
            result = run_reasoning(req, routing_stub)
            nonneg_so = next(
                n for n in result.session_subgraph.nodes.values()
                if n.get("name") == "VerifyNonNegativeEdges"
            )
            self.assertEqual(nonneg_so["state"]["checked_edges"], ["a->b", "b->c", "a->c"])
            self.assertEqual(nonneg_so["state"]["violating_edges"], ["b->c"])

    def test_no_procedure_requested_still_direct_answers(self):
        turn = (
            "<reasoning>GOAL: answer the easy question. KNOWN: A-star needs "
            "nonnegative weights. PLAN: answer directly.</reasoning>"
            "<answer>A-star is unsafe here because edge b->c has weight -1.</answer>"
        )
        stub = _ScriptedLLM([turn])

        with tempfile.TemporaryDirectory() as td:
            req = _make_request(self.GRAPH_PATH, Path(td))
            req.question = (
                "I have edges (a->b, weight 3), (b->c, weight -1), "
                "(a->c, weight 5). Is A-star safe?"
            )
            result = run_reasoning(req, stub)
            self.assertEqual(result.iterations_completed, 1)
            self.assertEqual(result.budget_usage["llm_calls"]["used"], 1)
            self.assertEqual(result.dispatch_outcomes, [])
            self.assertIn("unsafe", result.answer)

    def test_substrate_v2_flag_routes_mode_free_loop(self):
        response = """STEP_RESULT
status: resolved
result: A-star is unsafe with a negative edge; use Bellman-Ford and check for negative cycles if needed.
delta:
  decisions:
    - use Bellman-Ford for negative edge shortest path
  risks:
    - A-star invalid when a negative edge violates nonnegative edge precondition
END_STEP_RESULT"""
        stub = _ScriptedLLM([response])

        with tempfile.TemporaryDirectory() as td:
            req = _make_request(self.GRAPH_PATH, Path(td))
            req.enable_substrate_v2 = True
            req.question = "Can A-star be trusted with one negative edge?"
            result = run_reasoning(req, stub)

            self.assertIn("Bellman-Ford", result.answer)
            self.assertEqual(result.dispatch_outcomes, [])
            self.assertEqual(result.budget_usage["llm_calls"]["used"], 1)
            self.assertEqual(result.audit_summary["substrate_v2"], True)
            node_types = {n.get("node_type") for n in result.session_subgraph.nodes.values()}
            self.assertIn("substrate_v2_step", node_types)
            self.assertIn("substrate_v2_delta", node_types)
            self.assertIn("substrate_v2_check", node_types)
            self.assertIn("substrate_v2_signal", node_types)

    def test_substrate_v2_dynamic_slidingwindow_recurses_and_persists_child_step(self):
        responses = [
            """STEP_RESULT
status: need_info
result: Need derive the segment tree merge rule before final answer.
missing:
  question: What merge rule supports maximum slidingwindow under point updates?
  why_needed: Need O(log n) online updates.
  expected_shape: evidence
delta:
  decisions:
    - reject Kadane for online point updates
  gaps:
    - segment_tree_merge_rule
END_STEP_RESULT""",
            """STEP_RESULT
status: resolved
result: Store sum, prefix, suffix, and best; merge left and right nodes with best = max(left.best, right.best, left.suffix + right.prefix) in O(1).
delta:
  evidence:
    - segment tree merge uses sum, prefix, suffix, best and supports O(log n) point updates
END_STEP_RESULT""",
            """STEP_RESULT
status: resolved
result: Use a segment tree with long long sum, prefix, suffix, and best. Merge left and right nodes with best = max(left.best, right.best, left.suffix + right.prefix). Leaves use the element value, so non-empty all-negative arrays return the maximum element, not 0. Each update is O(log n).
delta:
  decisions:
    - use segment tree for online point updates
END_STEP_RESULT""",
        ]
        stub = _ScriptedLLM(responses)

        with tempfile.TemporaryDirectory() as td:
            req = _make_request(self.GRAPH_PATH, Path(td))
            req.enable_substrate_v2 = True
            req.question = (
                "Solve dynamic maximum slidingwindow with online point updates, "
                "negative values allowed, non-empty slidingwindow, values up to 1e9."
            )
            result = run_reasoning(req, stub)

            self.assertIn("segment tree", result.answer.lower())
            self.assertEqual(result.budget_usage["llm_calls"]["used"], 3)
            step_nodes = [
                n for n in result.session_subgraph.nodes.values()
                if n.get("node_type") == "substrate_v2_step"
            ]
            self.assertGreaterEqual(len(step_nodes), 3)
            self.assertTrue(any(n.get("parent_step_id") == "step_root" for n in step_nodes))

    def test_substrate_v2_checker_routing_covers_deep_suite_plugins(self):
        self.assertIn(
            "dynamic_connectivity_deletions",
            _substrate_v2_checker_plugins(
                "You have add(u,v), remove(u,v), and connected(u,v). Explain why plain DSU is insufficient and what time-axis structure you use."
            ),
        )
        self.assertIn(
            "segment_tree_beats",
            _substrate_v2_checker_plugins(
                "Maintain range_chmin(l,r,x) and range_sum(l,r). What per-node state makes the capped lazy updates correct?"
            ),
        )
        self.assertIn(
            "payment_crash_recovery",
            _substrate_v2_checker_plugins(
                "Design a payment worker for an at-least-once queue. The worker calls an external PSP with idempotency keys."
            ),
        )
        self.assertIn(
            "zero_downtime_migration",
            _substrate_v2_checker_plugins(
                "Split a monolith Orders table into a new service with zero downtime, verified before cutover, with rollback."
            ),
        )
        self.assertIn(
            "inventory_reservation",
            _substrate_v2_checker_plugins(
                "Design a flash-sale inventory reservation system with reservation TTL and oversell protection."
            ),
        )

    def test_substrate_v2_initial_signals_keep_systemic_design_lane(self):
        graph = MemoryGraph.load_json(self.GRAPH_PATH)
        req = _make_request(self.GRAPH_PATH, Path("."))
        req.question = (
            "Design a payment worker for an at-least-once queue. The worker calls an external PSP that supports idempotency keys, and the process can crash after the PSP charge succeeds but before the local database commit."
        )
        signals = _substrate_v2_initial_signals(req, graph, [])
        self.assertTrue(any(
            sig.kind == "procedure"
            and (sig.state or {}).get("preferred_lane") == "session_object"
            for sig in signals
        ))
        self.assertTrue(any(
            sig.kind == "constraint"
            and "durable local payment state machine" in sig.text.lower()
            for sig in signals
        ))

    def test_non_dispatch_run_produces_inspectable_session_structure(self):
        """Phase 2A acceptance #4: even when NO procedure fires, the session
        subgraph contains Q0, A0, and anchor evidence so the UI panel is
        always populated."""
        # Direct-answer flow: model writes its answer immediately, no CALL,
        # no apply-pattern, nothing for the dispatcher to fire on.
        turn = (
            "<reasoning>\n"
            "GOAL: Define entropy.\n"
            "KNOWN: standard physics def.\n"
            "PLAN: state directly.\n"
            "</reasoning>\n"
            "<answer>Entropy is a measure of disorder in a thermodynamic system.</answer>"
        )
        stub = _ScriptedLLM([turn])

        with tempfile.TemporaryDirectory() as td:
            dummy_graph_path = Path(td) / "dummy.json"
            with open(dummy_graph_path, "w") as f:
                import json
                json.dump({
                    "nodes": [
                        {
                            "id": "dummy_anchor",
                            "node_type": "fact",
                            "text": "Entropy is a measure of disorder.",
                            "confidence": 0.9,
                            "created_step": 0
                        }
                    ]
                }, f)
            req = _make_request(str(dummy_graph_path), Path(td))
            result = run_reasoning(req, stub)

            nodes = result.session_subgraph.nodes
            self.assertIn("Q0", nodes, "Q0 must be present after a non-dispatch run")
            self.assertIn("A0", nodes, "A0 must be present after a non-dispatch run")
            self.assertEqual(nodes["Q0"]["node_type"], "question")
            self.assertEqual(nodes["A0"]["node_type"], "answer")
            self.assertIn("Entropy", nodes["A0"]["text"])

            # Anchor evidence should be present (at least one)
            anchor_evidence = [
                n for nid, n in nodes.items()
                if nid.startswith("anchor_")
            ]
            self.assertGreater(
                len(anchor_evidence), 0,
                "At least one anchor evidence node must be present",
            )

            # Evidence -> answer support edges exist
            support_edges = [
                e for e in result.session_subgraph.edges
                if e.relation == "support" and e.dst == "A0"
            ]
            self.assertGreater(len(support_edges), 0)

    def test_latency_direct_answer_exits_in_one_iteration(self):
        """Phase 2A acceptance #5: a direct-answer conceptual question must
        exit the reasoning loop in iterations_completed == 1. No wasted
        iterations.

        This locks in the directive-tightening shipped in Phase 1
        post-hoc fixes: the model is told to ALWAYS include <answer>
        unless it explicitly invoked a procedure on this turn.
        """
        turn = (
            "<reasoning>\n"
            "GOAL: Compare two concepts.\n"
            "KNOWN: definitions.\n"
            "PLAN: answer directly.\n"
            "</reasoning>\n"
            "<answer>Concept A is foundational; concept B builds on it.</answer>"
        )
        stub = _ScriptedLLM([turn])

        with tempfile.TemporaryDirectory() as td:
            dummy_graph_path = Path(td) / "dummy.json"
            with open(dummy_graph_path, "w") as f:
                import json
                json.dump({
                    "nodes": [
                        {
                            "id": "dummy_anchor",
                            "node_type": "fact",
                            "text": "Entropy is a measure of disorder.",
                            "confidence": 0.9,
                            "created_step": 0
                        }
                    ]
                }, f)
            req = _make_request(str(dummy_graph_path), Path(td))
            result = run_reasoning(req, stub)
            self.assertEqual(
                result.iterations_completed, 1,
                f"Direct-answer conceptual question should exit in 1 iteration, "
                f"got {result.iterations_completed}",
            )
            self.assertEqual(len(result.dispatch_outcomes), 0)
            # Only one LLM call needed
            self.assertEqual(result.budget_usage["llm_calls"]["used"], 1)

    def test_budget_exhaustion_returns_graceful(self):
        """When budget runs out mid-loop, return what we have so far."""
        # Each turn keeps invoking but never finishes; budget cuts it off
        chatty = (
            "<reasoning>I'll apply VerifyAlgorithmPreconditions to A-star.</reasoning>"
        )
        sub = "ADD x TO state.preconditions_checked\nDONE\nsummary"
        stub = _ScriptedLLM([chatty, sub, chatty, sub, chatty, sub])

        with tempfile.TemporaryDirectory() as td:
            req = _make_request(self.GRAPH_PATH, Path(td))
            req.budgets = Budgets(max_llm_calls=2, max_composition_fan_out=5)
            result = run_reasoning(req, stub)
            self.assertIsNotNone(result.early_terminated_reason)
            self.assertIn("budget", result.early_terminated_reason.lower())


if __name__ == "__main__":
    unittest.main()
