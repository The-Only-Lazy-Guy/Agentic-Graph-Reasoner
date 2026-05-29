"""Tests for reasoning/dispatcher.py.

Three concerns:
  1. SCAN matches all five pattern variants on realistic reasoner text.
  2. SCAN ignores names that don't resolve to a procedure (silent miss).
  3. INVOKE creates a SessionObjectNode, calls the stub LLM, parses
     mutation commands from the response, applies them to session state.

The stub LLM is a callable that returns canned text — no model needed.
"""
from __future__ import annotations

import unittest
from typing import Callable

from reasoning.budgets import Budgets, BudgetTracker
from reasoning.composition import SUB_INVOCATION_OF
from reasoning.dispatcher import (
    Dispatcher,
    DispatchOutcome,
    PatternMatch,
    _CALL_RE,
    _MUT_ADD,
    _MUT_SET,
    _MUT_DELETE,
    _apply_mutations,
    _build_name_index,
    _parse_value,
    find_existing_sub_invocation,
)
from reasoning.schemas import ProcedureNode, Provenance
from reasoning.session_subgraph import SessionSubgraphController


def _make_seed_proc() -> ProcedureNode:
    return ProcedureNode(
        id="proc_verify_001",
        name="VerifyAlgorithmPreconditions",
        purpose="Check stated preconditions of an algorithm",
        when_to_use="applicability questions",
        signature={"inputs": [{"name": "algo", "type": "str"}], "outputs": []},
        state_schema={
            "preconditions_checked": "list[str]",
            "preconditions_violated": "list[str]",
            "evidence": "dict[str, str]",
        },
        body=(
            "Check preconditions of {args}. Emit ADD/SET/DELETE commands."
        ),
        example_use=None,
        provenance=Provenance(created_in_session_id="seed"),
    )


def _make_index():
    return {"proc_verify_001": _make_seed_proc()}


# ----------- SCAN tests ------------------------------------------------- #

class TestScan(unittest.TestCase):
    def setUp(self):
        self.disp = Dispatcher(_make_index())

    def test_apply_pattern(self):
        text = "First, I'll apply VerifyAlgorithmPreconditions to the user's graph."
        matches = self.disp.scan(text)
        self.assertEqual(len(matches), 1)
        m = matches[0]
        self.assertEqual(m.verb, "apply_intent")
        self.assertEqual(m.procedure_name, "VerifyAlgorithmPreconditions")
        self.assertIn("user's graph", m.args_text)

    def test_invoke_pattern(self):
        text = "Now invoke VerifyAlgorithmPreconditions with algorithm=Dijkstra instance=...."
        matches = self.disp.scan(text)
        self.assertEqual(len(matches), 1)
        self.assertEqual(matches[0].verb, "invoke")

    def test_using_the_pattern(self):
        text = "We resolve this using the VerifyAlgorithmPreconditions procedure."
        matches = self.disp.scan(text)
        self.assertEqual(len(matches), 1)
        self.assertEqual(matches[0].verb, "using_the")

    def test_create_new_pattern(self):
        text = "Create a new VerifyAlgorithmPreconditions object for this question."
        matches = self.disp.scan(text)
        self.assertEqual(len(matches), 1)
        self.assertEqual(matches[0].verb, "create_new")

    def test_unknown_procedure_ignored(self):
        text = "I'll apply BogusProcedureName to anything."
        matches = self.disp.scan(text)
        self.assertEqual(matches, [])

    def test_case_insensitive_lookup(self):
        text = "Apply verifyalgorithmpreconditions to the graph."
        matches = self.disp.scan(text)
        self.assertEqual(len(matches), 1)

    def test_multiple_invocations_in_one_pass(self):
        text = (
            "Step 1: I'll apply VerifyAlgorithmPreconditions to graph X.\n"
            "Later I'll invoke VerifyAlgorithmPreconditions with graph Y.\n"
        )
        matches = self.disp.scan(text)
        self.assertEqual(len(matches), 2)
        self.assertTrue(all(m.procedure_name == "VerifyAlgorithmPreconditions" for m in matches))

    def test_dedup_overlapping_patterns(self):
        # "I'll apply X to Y" can match both apply_intent AND apply patterns
        # at slightly different start positions — dedup keeps the first.
        text = "I'll apply VerifyAlgorithmPreconditions to Dijkstra on this graph."
        matches = self.disp.scan(text)
        self.assertEqual(len(matches), 1)


# ----------- Mutation parser tests -------------------------------------- #

class TestParseValue(unittest.TestCase):
    def test_string_quoted(self):
        self.assertEqual(_parse_value('"hello"'), "hello")

    def test_string_unquoted(self):
        self.assertEqual(_parse_value('nonneg_edges'), "nonneg_edges")

    def test_int(self):
        self.assertEqual(_parse_value("42"), 42)

    def test_list(self):
        self.assertEqual(_parse_value('["a","b"]'), ["a", "b"])

    def test_dict(self):
        self.assertEqual(_parse_value('{"k":"v"}'), {"k": "v"})

    def test_empty(self):
        self.assertEqual(_parse_value(""), "")


class TestMutationRegex(unittest.TestCase):
    def test_add_simple(self):
        text = "ADD nonneg_edges TO state.preconditions_checked"
        ms = list(_MUT_ADD.finditer(text))
        self.assertEqual(len(ms), 1)
        self.assertEqual(ms[0].group("value"), "nonneg_edges")
        self.assertEqual(ms[0].group("path"), "state.preconditions_checked")

    def test_set_with_quotes(self):
        text = 'SET state.evidence.nonneg = "Edge b->c has weight -1"'
        ms = list(_MUT_SET.finditer(text))
        self.assertEqual(len(ms), 1)
        self.assertEqual(ms[0].group("path"), "state.evidence.nonneg")

    def test_delete(self):
        text = "DELETE state.preconditions_deferred"
        ms = list(_MUT_DELETE.finditer(text))
        self.assertEqual(len(ms), 1)

    def test_multiple_mutations(self):
        text = (
            "ADD nonneg TO state.preconditions_checked\n"
            "ADD nonneg TO state.preconditions_violated\n"
            'SET state.evidence.nonneg = "Edge b->c weight -1"\n'
            "DONE\n"
        )
        adds = list(_MUT_ADD.finditer(text))
        sets = list(_MUT_SET.finditer(text))
        self.assertEqual(len(adds), 2)
        self.assertEqual(len(sets), 1)


# ----------- Mutation application -------------------------------------- #

class TestApplyMutations(unittest.TestCase):
    def _setup(self):
        ctrl = SessionSubgraphController("sess1", "Q", "cs4")
        proc = _make_seed_proc()
        # Initial state matches procedure schema
        oid = ctrl.create_object(
            proc,
            {"preconditions_checked": [], "preconditions_violated": [], "evidence": {}},
            "create",
        )
        ctrl.step()
        return ctrl, oid

    def test_add_appends_to_list(self):
        ctrl, oid = self._setup()
        response = "ADD nonneg_edges TO state.preconditions_checked"
        n = _apply_mutations(ctrl, oid, response, "test")
        self.assertEqual(n, 1)
        self.assertEqual(
            ctrl.subgraph.nodes[oid]["state"]["preconditions_checked"],
            ["nonneg_edges"],
        )

    def test_add_multiple_appends_to_same_list(self):
        ctrl, oid = self._setup()
        response = (
            "ADD nonneg TO state.preconditions_checked\n"
            "ADD acyclic TO state.preconditions_checked\n"
        )
        n = _apply_mutations(ctrl, oid, response, "test")
        self.assertEqual(n, 2)
        self.assertEqual(
            ctrl.subgraph.nodes[oid]["state"]["preconditions_checked"],
            ["nonneg", "acyclic"],
        )

    def test_set_with_nested_path(self):
        ctrl, oid = self._setup()
        response = 'SET state.evidence.nonneg = "Edge b->c has weight -1"'
        n = _apply_mutations(ctrl, oid, response, "test")
        self.assertEqual(n, 1)
        self.assertEqual(
            ctrl.subgraph.nodes[oid]["state"]["evidence"]["nonneg"],
            "Edge b->c has weight -1",
        )

    def test_delete_field(self):
        ctrl, oid = self._setup()
        # Pre-populate
        ctrl.update_object(oid, "state.evidence.nonneg", "something", "init")
        response = "DELETE state.evidence.nonneg"
        n = _apply_mutations(ctrl, oid, response, "test")
        self.assertEqual(n, 1)
        self.assertNotIn("nonneg", ctrl.subgraph.nodes[oid]["state"]["evidence"])

    def test_full_seed_procedure_sequence(self):
        """The actual mutation sequence we'd expect for the Dijkstra case."""
        ctrl, oid = self._setup()
        response = (
            "ADD nonneg_edges TO state.preconditions_checked\n"
            "ADD acyclic TO state.preconditions_checked\n"
            "ADD nonneg_edges TO state.preconditions_violated\n"
            'SET state.evidence.nonneg_edges = "Edge b->c has weight -1"\n'
            "DONE\n"
            "(this verifies Dijkstra preconditions; nonneg violated)\n"
        )
        n = _apply_mutations(ctrl, oid, response, "verify")
        self.assertEqual(n, 4)
        state = ctrl.subgraph.nodes[oid]["state"]
        self.assertEqual(state["preconditions_checked"], ["nonneg_edges", "acyclic"])
        self.assertEqual(state["preconditions_violated"], ["nonneg_edges"])
        self.assertEqual(state["evidence"]["nonneg_edges"], "Edge b->c has weight -1")


# ----------- INVOKE end-to-end with stub LLM --------------------------- #

class TestInvoke(unittest.TestCase):
    def _stub_llm_factory(self, response: str) -> Callable[[str], str]:
        def stub(_: str) -> str:
            return response
        return stub

    def test_invoke_creates_object_and_applies_mutations(self):
        ctrl = SessionSubgraphController(
            "sess1",
            "Original question mentions edge b->c weight -1.",
            "cs4",
        )
        disp = Dispatcher(_make_index())
        budget = BudgetTracker(Budgets(max_llm_calls=3))

        match = PatternMatch(
            verb="apply_intent",
            procedure_name="VerifyAlgorithmPreconditions",
            args_text="Dijkstra on graph with edge b->c weight -1",
            start=0,
            end=50,
        )
        canned = (
            "ADD nonneg_edges TO state.preconditions_checked\n"
            "ADD nonneg_edges TO state.preconditions_violated\n"
            'SET state.evidence.nonneg_edges = "weight -1"\n'
            "DONE"
        )
        outcome = disp.invoke(match, ctrl, self._stub_llm_factory(canned), budget=budget)

        self.assertEqual(outcome.procedure_id, "proc_verify_001")
        self.assertIsNotNone(outcome.object_id)
        self.assertEqual(outcome.mutations_applied, 3)
        self.assertEqual(budget.used["llm_call"], 1)
        # Sub-prompt should mention the procedure name and the args
        self.assertIn("VerifyAlgorithmPreconditions", outcome.sub_prompt)
        self.assertIn("Dijkstra", outcome.sub_prompt)
        self.assertIn("Original question mentions edge b->c weight -1", outcome.sub_prompt)
        # Final state correct
        state = ctrl.subgraph.nodes[outcome.object_id]["state"]
        self.assertEqual(state["preconditions_checked"], ["nonneg_edges"])
        self.assertEqual(state["preconditions_violated"], ["nonneg_edges"])
        self.assertEqual(state["evidence"]["nonneg_edges"], "weight -1")

    def test_invoke_unknown_procedure_returns_error(self):
        ctrl = SessionSubgraphController("sess1", "Q", "cs4")
        disp = Dispatcher({})  # empty index
        match = PatternMatch(
            verb="apply_intent",
            procedure_name="MysteryProc",
            args_text="x",
            start=0, end=10,
        )
        outcome = disp.invoke(match, ctrl, self._stub_llm_factory(""))
        self.assertIsNone(outcome.procedure_id)
        self.assertIsNotNone(outcome.error)

    def test_invoke_reuses_existing_object(self):
        ctrl = SessionSubgraphController("sess1", "Q", "cs4")
        disp = Dispatcher(_make_index())

        # First invocation creates the object
        m1 = PatternMatch("apply_intent", "VerifyAlgorithmPreconditions", "first", 0, 10)
        out1 = disp.invoke(
            m1, ctrl,
            self._stub_llm_factory("ADD nonneg TO state.preconditions_checked\nDONE"),
        )

        # Second invocation reuses the same object id (passed explicitly)
        m2 = PatternMatch("apply_intent", "VerifyAlgorithmPreconditions", "second", 20, 30)
        out2 = disp.invoke(
            m2, ctrl,
            self._stub_llm_factory("ADD acyclic TO state.preconditions_checked\nDONE"),
            existing_object_id=out1.object_id,
        )

        self.assertEqual(out1.object_id, out2.object_id)
        # Final list has BOTH appends
        state = ctrl.subgraph.nodes[out1.object_id]["state"]
        self.assertEqual(state["preconditions_checked"], ["nonneg", "acyclic"])


# ----------- Phase 2A: parent_object_id + sub_invocation_of -------------- #

class TestParentObjectId(unittest.TestCase):
    def _stub(self, response: str) -> Callable[[str], str]:
        def f(_: str) -> str:
            return response
        return f

    def _setup_with_parent(self):
        """Create a parent session_object first so we have something to
        attach sub-invocations to."""
        ctrl = SessionSubgraphController("sess1", "Q", "cs4")
        disp = Dispatcher(_make_index())
        parent_match = PatternMatch(
            "apply_intent", "VerifyAlgorithmPreconditions", "parent ctx", 0, 10,
        )
        parent_outcome = disp.invoke(
            parent_match, ctrl,
            self._stub("ADD parent_check TO state.preconditions_checked\nDONE"),
        )
        return ctrl, disp, parent_outcome.object_id

    def test_parent_object_id_set_creates_sub_invocation_edge(self):
        ctrl, disp, parent_id = self._setup_with_parent()
        # Reset the procedure index to a different "child" procedure so
        # the sub-invocation isn't conflated with parent reuse.
        from reasoning.schemas import ProcedureNode, Provenance
        child_proc = ProcedureNode(
            id="proc_child_001",
            name="ChildProc",
            purpose="leaf procedure",
            when_to_use="...",
            signature={},
            state_schema={"checked": "list[str]"},
            body="check whatever",
            example_use=None,
            provenance=Provenance(created_in_session_id="seed"),
        )
        disp_with_child = Dispatcher({**_make_index(), "proc_child_001": child_proc})

        child_match = PatternMatch(
            "apply_intent", "ChildProc", "child ctx", 0, 10,
        )
        out = disp_with_child.invoke(
            child_match, ctrl,
            self._stub("ADD x TO state.checked\nDONE"),
            parent_object_id=parent_id,
        )

        # The DispatchOutcome reflects the parent
        self.assertEqual(out.parent_object_id, parent_id)

        # A sub_invocation_of edge connects child -> parent
        edges = [
            e for e in ctrl.subgraph.edges
            if e.relation == SUB_INVOCATION_OF
        ]
        self.assertEqual(len(edges), 1)
        edge = edges[0]
        self.assertEqual(edge.src, out.object_id)
        self.assertEqual(edge.dst, parent_id)
        # Metadata carries the dedupe-key fields
        self.assertEqual(edge.metadata["procedure_id"], "proc_child_001")
        self.assertEqual(edge.metadata["procedure_name"], "ChildProc")
        self.assertEqual(edge.metadata["args_text"], "child ctx")

    def test_top_level_invocation_does_not_create_sub_edge(self):
        ctrl = SessionSubgraphController("sess1", "Q", "cs4")
        disp = Dispatcher(_make_index())
        match = PatternMatch(
            "apply_intent", "VerifyAlgorithmPreconditions", "test", 0, 10,
        )
        out = disp.invoke(
            match, ctrl,
            self._stub("ADD x TO state.preconditions_checked\nDONE"),
            # parent_object_id NOT passed -> top-level
        )
        self.assertIsNone(out.parent_object_id)
        sub_edges = [e for e in ctrl.subgraph.edges if e.relation == SUB_INVOCATION_OF]
        self.assertEqual(sub_edges, [])

    def test_audit_trigger_text_carries_parent_prefix(self):
        ctrl, disp, parent_id = self._setup_with_parent()
        from reasoning.schemas import ProcedureNode, Provenance
        child_proc = ProcedureNode(
            id="proc_child_002", name="OtherChild",
            purpose="", when_to_use="", signature={}, state_schema={"checked": "list[str]"},
            body="", example_use=None,
            provenance=Provenance(created_in_session_id="seed"),
        )
        disp2 = Dispatcher({**_make_index(), "proc_child_002": child_proc})
        match = PatternMatch("apply_intent", "OtherChild", "x", 0, 5)
        disp2.invoke(
            match, ctrl, self._stub("DONE"),
            parent_object_id=parent_id,
        )
        # Find the create-audit entry for the child
        child_create_entries = [
            e for e in ctrl.subgraph.audit_log
            if e.operation == "create" and e.object_id != parent_id
        ]
        self.assertEqual(len(child_create_entries), 1)
        self.assertIn(f"sub-invocation of {parent_id}", child_create_entries[0].triggered_by_text)


class TestDedupeLookup(unittest.TestCase):
    """find_existing_sub_invocation locates an already-created child of
    a given (parent, procedure, args) tuple. Used by Sub-phase 2.4 to
    prevent duplicate sub-invocations."""

    def _stub(self, response: str) -> Callable[[str], str]:
        def f(_: str) -> str:
            return response
        return f

    def test_returns_existing_child_id_on_exact_match(self):
        ctrl = SessionSubgraphController("sess1", "Q", "cs4")
        # Create a parent + one child via parent_object_id
        disp = Dispatcher(_make_index())
        parent_out = disp.invoke(
            PatternMatch("apply_intent", "VerifyAlgorithmPreconditions", "parent", 0, 10),
            ctrl, self._stub("DONE"),
        )
        from reasoning.schemas import ProcedureNode, Provenance
        child_proc = ProcedureNode(
            id="proc_child", name="ChildProc",
            purpose="", when_to_use="", signature={}, state_schema={"checked": "list[str]"},
            body="", example_use=None,
            provenance=Provenance(created_in_session_id="seed"),
        )
        disp2 = Dispatcher({**_make_index(), "proc_child": child_proc})
        child_out = disp2.invoke(
            PatternMatch("apply_intent", "ChildProc", "args_v1", 0, 10),
            ctrl, self._stub("DONE"),
            parent_object_id=parent_out.object_id,
        )

        # Same (parent, proc_id, args) → finds the existing child
        found = find_existing_sub_invocation(
            ctrl, parent_out.object_id, "proc_child", "args_v1",
        )
        self.assertEqual(found, child_out.object_id)

    def test_returns_none_when_args_differ(self):
        ctrl = SessionSubgraphController("sess1", "Q", "cs4")
        disp = Dispatcher(_make_index())
        parent_out = disp.invoke(
            PatternMatch("apply_intent", "VerifyAlgorithmPreconditions", "parent", 0, 10),
            ctrl, self._stub("DONE"),
        )
        from reasoning.schemas import ProcedureNode, Provenance
        child_proc = ProcedureNode(
            id="proc_child", name="ChildProc",
            purpose="", when_to_use="", signature={}, state_schema={"checked": "list[str]"},
            body="", example_use=None,
            provenance=Provenance(created_in_session_id="seed"),
        )
        disp2 = Dispatcher({**_make_index(), "proc_child": child_proc})
        disp2.invoke(
            PatternMatch("apply_intent", "ChildProc", "args_A", 0, 10),
            ctrl, self._stub("DONE"),
            parent_object_id=parent_out.object_id,
        )

        # Distinct args_text → no match
        found = find_existing_sub_invocation(
            ctrl, parent_out.object_id, "proc_child", "args_B",
        )
        self.assertIsNone(found)

    def test_returns_none_when_no_parent_match(self):
        ctrl = SessionSubgraphController("sess1", "Q", "cs4")
        found = find_existing_sub_invocation(
            ctrl, "so_does_not_exist", "proc_anything", "any args",
        )
        self.assertIsNone(found)


# ----------- Phase 2A: structured CALL parser + recursive invoke -------- #

class TestCallRegex(unittest.TestCase):
    """The CALL pattern must match the structured grammar reliably and
    must NOT collide with the top-level free-text patterns."""

    def test_call_with_args(self):
        text = "CALL VerifyNonNegativeEdges WITH instance=Dijkstra graph"
        matches = list(_CALL_RE.finditer(text))
        self.assertEqual(len(matches), 1)
        self.assertEqual(matches[0].group("name"), "VerifyNonNegativeEdges")
        self.assertEqual(matches[0].group("args"), "instance=Dijkstra graph")

    def test_call_without_args(self):
        text = "CALL DetectNegativeCycle"
        matches = list(_CALL_RE.finditer(text))
        self.assertEqual(len(matches), 1)
        self.assertEqual(matches[0].group("name"), "DetectNegativeCycle")
        self.assertIsNone(matches[0].group("args"))

    def test_call_lowercase(self):
        text = "call SomeProc with x=1"
        matches = list(_CALL_RE.finditer(text))
        self.assertEqual(len(matches), 1)

    def test_multiple_calls_one_response(self):
        text = (
            "CALL A WITH first set of args\n"
            "ADD nothing TO state.x\n"
            "CALL B WITH different args\n"
            "DONE"
        )
        matches = list(_CALL_RE.finditer(text))
        self.assertEqual(len(matches), 2)
        self.assertEqual(matches[0].group("name"), "A")
        self.assertEqual(matches[1].group("name"), "B")

    def test_does_not_match_freetext_apply(self):
        """Free-text "apply X to Y" must NOT trigger the CALL regex.
        The CALL parser only fires on the structured shape."""
        text = "I'll apply Foo to bar"
        matches = list(_CALL_RE.finditer(text))
        self.assertEqual(matches, [])


class TestRecursiveDispatch(unittest.TestCase):
    """The composer pattern: parent procedure's body emits CALL X commands,
    children are dispatched as sub-invocations under the parent's session_object.
    """

    def _make_composer_and_children(self):
        composer = ProcedureNode(
            id="proc_composer",
            name="VerifyShortestPath",
            purpose="composer",
            when_to_use="...",
            signature={},
            state_schema={"verdict": "str"},
            body="delegate",
            example_use=None,
            provenance=Provenance(created_in_session_id="seed"),
        )
        nonneg = ProcedureNode(
            id="proc_nonneg",
            name="VerifyNonNegativeEdges",
            purpose="leaf",
            when_to_use="...",
            signature={},
            state_schema={"violating_edges": "list[str]"},
            body="check edges",
            example_use=None,
            provenance=Provenance(created_in_session_id="seed"),
        )
        cycle = ProcedureNode(
            id="proc_cycle",
            name="DetectNegativeCycle",
            purpose="leaf",
            when_to_use="...",
            signature={},
            state_schema={"detected_cycles": "list[str]"},
            body="check cycles",
            example_use=None,
            provenance=Provenance(created_in_session_id="seed"),
        )
        return composer, nonneg, cycle

    def _scripted_llm(self, responses_by_marker):
        """Stub LLM that returns a canned response based on procedure name
        in the prompt. Lets us script per-procedure-body responses."""
        def stub(prompt: str) -> str:
            for marker, response in responses_by_marker.items():
                if marker in prompt:
                    return response
            return "DONE"
        return stub

    def test_composer_invokes_two_children(self):
        composer, nonneg, cycle = self._make_composer_and_children()
        ctrl = SessionSubgraphController("sess1", "Q", "cs4")
        disp = Dispatcher({composer.id: composer, nonneg.id: nonneg, cycle.id: cycle})
        budget = BudgetTracker(Budgets(max_llm_calls=10, max_composition_fan_out=5))
        budget.on_step_change(0)

        # Composer body output: emit two CALL commands
        composer_body_response = (
            "SET state.verdict = \"in progress\"\n"
            "CALL VerifyNonNegativeEdges WITH instance=Dijkstra example\n"
            "CALL DetectNegativeCycle WITH instance=Dijkstra example\n"
            "DONE"
        )
        # Each leaf's body output: do its thing then DONE
        leaf_nonneg_response = "ADD b->c TO state.violating_edges\nDONE"
        leaf_cycle_response = "DONE\n(no cycles found)"

        stub = self._scripted_llm({
            "VerifyShortestPath": composer_body_response,
            "VerifyNonNegativeEdges": leaf_nonneg_response,
            "DetectNegativeCycle": leaf_cycle_response,
        })

        top_match = PatternMatch(
            "apply_intent", "VerifyShortestPath", "Dijkstra on a neg-edge graph",
            0, 10,
        )
        outcome = disp.invoke(top_match, ctrl, stub, budget=budget)

        # The composer's outcome has 2 sub_outcomes
        self.assertEqual(len(outcome.sub_outcomes), 2)
        self.assertEqual(
            {o.match.procedure_name for o in outcome.sub_outcomes},
            {"VerifyNonNegativeEdges", "DetectNegativeCycle"},
        )
        # Each sub-outcome has parent_object_id set to the composer's
        for o in outcome.sub_outcomes:
            self.assertEqual(o.parent_object_id, outcome.object_id)
        # Two sub_invocation_of edges exist
        edges = [e for e in ctrl.subgraph.edges if e.relation == SUB_INVOCATION_OF]
        self.assertEqual(len(edges), 2)
        # Each child's session_object has its own state, distinct from siblings
        nonneg_outcome = next(o for o in outcome.sub_outcomes if o.match.procedure_name == "VerifyNonNegativeEdges")
        cycle_outcome = next(o for o in outcome.sub_outcomes if o.match.procedure_name == "DetectNegativeCycle")
        self.assertNotEqual(nonneg_outcome.object_id, cycle_outcome.object_id)
        self.assertEqual(
            ctrl.subgraph.nodes[nonneg_outcome.object_id]["state"]["violating_edges"],
            ["b->c"],
        )

    def test_duplicate_call_with_same_args_is_deduped(self):
        """Acceptance criterion #3: a parent body emitting `CALL X WITH foo`
        twice in the same response creates only ONE child."""
        composer, nonneg, _ = self._make_composer_and_children()
        ctrl = SessionSubgraphController("sess1", "Q", "cs4")
        disp = Dispatcher({composer.id: composer, nonneg.id: nonneg})
        budget = BudgetTracker(Budgets(max_llm_calls=10, max_composition_fan_out=5))
        budget.on_step_change(0)

        composer_body = (
            "CALL VerifyNonNegativeEdges WITH instance=foo\n"
            "CALL VerifyNonNegativeEdges WITH instance=foo\n"
            "DONE"
        )
        stub = self._scripted_llm({
            "VerifyShortestPath": composer_body,
            "VerifyNonNegativeEdges": "DONE",
        })
        top_match = PatternMatch("apply_intent", "VerifyShortestPath", "x", 0, 10)
        outcome = disp.invoke(top_match, ctrl, stub, budget=budget)

        # Only ONE child despite two CALL lines
        self.assertEqual(len(outcome.sub_outcomes), 1)
        # And only one sub_invocation_of edge
        edges = [e for e in ctrl.subgraph.edges if e.relation == SUB_INVOCATION_OF]
        self.assertEqual(len(edges), 1)

    def test_distinct_args_produce_distinct_children(self):
        """If the agent legitimately wants the same procedure run on two
        different inputs, distinct args are how it asks for that."""
        composer, nonneg, _ = self._make_composer_and_children()
        ctrl = SessionSubgraphController("sess1", "Q", "cs4")
        disp = Dispatcher({composer.id: composer, nonneg.id: nonneg})
        budget = BudgetTracker(Budgets(max_llm_calls=10, max_composition_fan_out=5))
        budget.on_step_change(0)

        composer_body = (
            "CALL VerifyNonNegativeEdges WITH instance=graphA\n"
            "CALL VerifyNonNegativeEdges WITH instance=graphB\n"
            "DONE"
        )
        stub = self._scripted_llm({
            "VerifyShortestPath": composer_body,
            "VerifyNonNegativeEdges": "DONE",
        })
        outcome = disp.invoke(
            PatternMatch("apply_intent", "VerifyShortestPath", "x", 0, 10),
            ctrl, stub, budget=budget,
        )
        self.assertEqual(len(outcome.sub_outcomes), 2)
        # Distinct object ids
        ids = {o.object_id for o in outcome.sub_outcomes}
        self.assertEqual(len(ids), 2)

    def test_unknown_procedure_in_call_silently_skipped(self):
        composer, nonneg, _ = self._make_composer_and_children()
        ctrl = SessionSubgraphController("sess1", "Q", "cs4")
        disp = Dispatcher({composer.id: composer, nonneg.id: nonneg})
        budget = BudgetTracker(Budgets(max_llm_calls=10, max_composition_fan_out=5))
        budget.on_step_change(0)

        composer_body = (
            "CALL BogusProc WITH something\n"
            "CALL VerifyNonNegativeEdges WITH real args\n"
            "DONE"
        )
        stub = self._scripted_llm({
            "VerifyShortestPath": composer_body,
            "VerifyNonNegativeEdges": "DONE",
        })
        outcome = disp.invoke(
            PatternMatch("apply_intent", "VerifyShortestPath", "x", 0, 10),
            ctrl, stub, budget=budget,
        )
        # The real child fires; the bogus one is silently skipped
        self.assertEqual(len(outcome.sub_outcomes), 1)
        self.assertEqual(outcome.sub_outcomes[0].match.procedure_name, "VerifyNonNegativeEdges")

    def test_fan_out_budget_caps_composition(self):
        """When a composer emits MORE child CALLs than the per-step fan-out
        cap allows, the dispatcher must:
          - Apply children one-by-one until the cap is hit
          - Stop dispatching at that point
          - Leave already-completed children in sub_outcomes
          - Not crash (graceful BudgetExhausted handling).
        """
        composer = ProcedureNode(
            id="proc_composer", name="Composer",
            purpose="big composer", when_to_use="...",
            signature={}, state_schema={}, body="...",
            example_use=None,
            provenance=Provenance(created_in_session_id="seed"),
        )
        # Five distinct leaf procedures
        leaves = []
        for i in range(5):
            leaves.append(ProcedureNode(
                id=f"proc_leaf_{i}", name=f"Leaf{i}",
                purpose="leaf", when_to_use="...",
                signature={}, state_schema={}, body="...",
                example_use=None,
                provenance=Provenance(created_in_session_id="seed"),
            ))

        ctrl = SessionSubgraphController("sess1", "Q", "cs4")
        disp = Dispatcher({p.id: p for p in [composer] + leaves})

        # Cap fan_out at 3. Composer fires 5 CALLs in one body — we expect
        # only 3 children to complete (the 4th attempt blows the budget;
        # 5th is never even tried).
        budget = BudgetTracker(Budgets(
            max_llm_calls=20,
            max_composition_fan_out=3,
            max_recursion_depth=8,
        ))
        budget.on_step_change(0)

        composer_body = "\n".join([
            f"CALL Leaf{i} WITH args_{i}" for i in range(5)
        ]) + "\nDONE"

        def stub(prompt: str) -> str:
            if "Composer" in prompt:
                return composer_body
            return "DONE"

        outcome = disp.invoke(
            PatternMatch("apply_intent", "Composer", "x", 0, 10),
            ctrl, stub, budget=budget,
        )

        # Exactly 3 children completed (fan_out cap)
        self.assertEqual(len(outcome.sub_outcomes), 3,
                         f"Expected 3 children given fan_out cap=3, got {len(outcome.sub_outcomes)}")
        # First three child procedures completed in order: Leaf0, Leaf1, Leaf2
        self.assertEqual(
            [c.match.procedure_name for c in outcome.sub_outcomes],
            ["Leaf0", "Leaf1", "Leaf2"],
        )
        # Budget summary confirms fan_out is at the cap
        self.assertEqual(budget.fan_out_this_step, 3)

    def test_recursion_depth_budget_caps_chain(self):
        """A long chain X -> X -> X -> ... must terminate at the recursion cap,
        not loop forever."""
        # Self-calling procedure: its body emits CALL to itself with new args
        looper = ProcedureNode(
            id="proc_loop", name="Looper",
            purpose="recurses", when_to_use="...",
            signature={}, state_schema={"depth": "int"}, body="...",
            example_use=None,
            provenance=Provenance(created_in_session_id="seed"),
        )
        ctrl = SessionSubgraphController("sess1", "Q", "cs4")
        disp = Dispatcher({looper.id: looper})
        budget = BudgetTracker(Budgets(max_llm_calls=100, max_recursion_depth=2, max_composition_fan_out=10))
        budget.on_step_change(0)

        counter = {"n": 0}
        def stub(prompt: str) -> str:
            counter["n"] += 1
            return f"CALL Looper WITH depth={counter['n']}\nDONE"

        # Top-level invocation — recursion depth starts at 0 here.
        # First sub-call pushes to depth=1, second to depth=2; third should raise.
        outcome = disp.invoke(
            PatternMatch("apply_intent", "Looper", "depth=0", 0, 10),
            ctrl, stub, budget=budget,
        )

        # Walk the chain to verify it terminated cleanly
        depth = 0
        cur = outcome
        while cur.sub_outcomes:
            depth += 1
            self.assertEqual(len(cur.sub_outcomes), 1)
            cur = cur.sub_outcomes[0]
        # Chain depth must not exceed the recursion budget cap
        self.assertLessEqual(depth, 2)


# ----------- Phase 2A: version-chain name resolution -------------------- #

class TestVersionChainResolution(unittest.TestCase):
    """The dispatcher's name index must resolve each name to the active
    head of its version chain — excluding deprecated procedures and
    procedures with a superseded_by_id."""

    def _v(self, vnum, *, deprecated=False, superseded_by_id=None, parent_version_id=None):
        return ProcedureNode(
            id=f"proc_X_v{vnum}",
            name="ExampleProc",
            purpose=f"version {vnum}",
            when_to_use="...",
            signature={},
            state_schema={},
            body="",
            example_use=None,
            provenance=Provenance(
                created_in_session_id="seed",
                deprecated=deprecated,
            ),
            version=vnum,
            parent_version_id=parent_version_id,
            superseded_by_id=superseded_by_id,
        )

    def test_single_version_resolves_to_self(self):
        v1 = self._v(1)
        idx = _build_name_index([v1])
        self.assertEqual(idx["exampleproc"].id, v1.id)

    def test_v1_superseded_by_v2_resolves_to_v2(self):
        v1 = self._v(1, superseded_by_id="proc_X_v2")
        v2 = self._v(2, parent_version_id="proc_X_v1")
        idx = _build_name_index([v1, v2])
        self.assertEqual(idx["exampleproc"].id, "proc_X_v2")

    def test_deprecated_v1_excluded(self):
        v1 = self._v(1, deprecated=True)
        v2 = self._v(2, parent_version_id="proc_X_v1")
        idx = _build_name_index([v1, v2])
        self.assertEqual(idx["exampleproc"].id, "proc_X_v2")

    def test_all_deprecated_excluded_entirely(self):
        v1 = self._v(1, deprecated=True)
        v2 = self._v(2, parent_version_id="proc_X_v1", deprecated=True)
        idx = _build_name_index([v1, v2])
        self.assertNotIn("exampleproc", idx)

    def test_dispatcher_resolve_name(self):
        v1 = self._v(1, superseded_by_id="proc_X_v2")
        v2 = self._v(2, parent_version_id="proc_X_v1")
        disp = Dispatcher({v1.id: v1, v2.id: v2})
        resolved = disp.resolve_name("ExampleProc")
        self.assertIsNotNone(resolved)
        self.assertEqual(resolved.id, "proc_X_v2")
        # Case-insensitive
        self.assertEqual(disp.resolve_name("exampleproc").id, "proc_X_v2")
        # Unknown name
        self.assertIsNone(disp.resolve_name("UnknownProc"))

    def test_top_level_invocation_routes_to_v2(self):
        """When the main reasoner says 'apply ExampleProc to ...', the
        dispatcher should run the v2 procedure body, not v1's."""
        v1 = ProcedureNode(
            id="proc_X_v1", name="ExampleProc", purpose="v1 body",
            when_to_use="", signature={}, state_schema={"called_version": "int"},
            body="v1 instructions", example_use=None,
            provenance=Provenance(created_in_session_id="seed"),
            version=1, superseded_by_id="proc_X_v2",
        )
        v2 = ProcedureNode(
            id="proc_X_v2", name="ExampleProc", purpose="v2 body",
            when_to_use="", signature={}, state_schema={"called_version": "int"},
            body="v2 instructions", example_use=None,
            provenance=Provenance(created_in_session_id="seed"),
            version=2, parent_version_id="proc_X_v1",
        )
        ctrl = SessionSubgraphController("sess1", "Q", "cs4")
        disp = Dispatcher({v1.id: v1, v2.id: v2})

        match = PatternMatch("apply_intent", "ExampleProc", "x", 0, 10)
        outcome = disp.invoke(
            match, ctrl,
            lambda p: "SET state.called_version = 2\nDONE",
        )
        # The procedure_id on the outcome must match v2 (the head), NOT v1
        self.assertEqual(outcome.procedure_id, "proc_X_v2")
        # And the rendered sub-prompt should contain v2's body text
        self.assertIn("v2 instructions", outcome.sub_prompt)
        self.assertNotIn("v1 instructions", outcome.sub_prompt)


if __name__ == "__main__":
    unittest.main()
