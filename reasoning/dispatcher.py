"""Dispatcher: regex-based invocation of procedures from reasoner text.

The trickiest piece of Phase 1. Three responsibilities:

  1. SCAN — pattern-match the reasoner's <reasoning> output for
     procedure invocations. Free-text matching, NOT JSON parsing.
     Resolved decision §5: pattern-match for Phase 1, upgrade to JSON
     grammar in Phase 2.

  2. INVOKE — given a matched invocation, instantiate (or update) a
     SessionObjectNode, render the procedure body as a sub-prompt,
     call the sub-LLM, parse mutation commands from the response, and
     apply them to session state via SessionSubgraphController's
     full-CRUD methods.

  3. MUTATION PARSING — the procedure body should instruct the
     sub-LLM to emit mutations in a constrained grammar:

        ADD <value> TO <field_path>
        SET <field_path> = <json_value>
        DELETE <field_path>
        DONE

     The parser is generous on whitespace but strict on the verbs.
     Anything not matching is ignored (graceful degradation).

This module is the *first* place where we depend on a real LLM. The
contract: callers pass a `llm_call(prompt: str) -> str` callable. Tests
substitute a StubLLM with canned responses.
"""
from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

from reasoning.budgets import BudgetTracker, BudgetExhausted
from reasoning.schemas import ProcedureNode
from reasoning.session_subgraph import SessionSubgraphController


# ---- Pattern catalog ---------------------------------------------------- #
# Each pattern names a 'verb' and captures `name` (procedure) and optional
# `args` (free-text argument span). Patterns are checked in priority order;
# first match wins per text position.

# Note: \w doesn't match dots / underscores in some contexts — we use a
# class that includes letters, digits, and underscore. CamelCase procedure
# names like VerifyAlgorithmPreconditions are matched as a single token.
_PROC_NAME = r"(?P<name>[A-Za-z][A-Za-z0-9_]*)"

_PATTERNS: List[tuple[str, re.Pattern[str]]] = [
    # "apply VerifyAlgorithmPreconditions to ..."
    ("apply", re.compile(
        rf"\bapply\s+{_PROC_NAME}\s+to\s+(?P<args>[^\n]+)",
        re.IGNORECASE,
    )),
    # "I'll apply X" / "I will apply X"
    ("apply_intent", re.compile(
        rf"\bI(?:'?ll|\s+will)\s+(?:now\s+)?apply\s+{_PROC_NAME}(?:\s+to\s+(?P<args>[^\n]+))?",
        re.IGNORECASE,
    )),
    # "invoke X" / "invoke X with ..."
    ("invoke", re.compile(
        rf"\binvoke\s+{_PROC_NAME}(?:\s+with\s+(?P<args>[^\n]+))?",
        re.IGNORECASE,
    )),
    # "using the X procedure"
    ("using_the", re.compile(
        rf"\busing\s+the\s+{_PROC_NAME}\s+procedure",
        re.IGNORECASE,
    )),
    # "create a new X object" / "create new X instance"
    ("create_new", re.compile(
        rf"\bcreate\s+a?\s*new\s+{_PROC_NAME}\s+(?:object|instance)",
        re.IGNORECASE,
    )),
]


@dataclass
class PatternMatch:
    """One detected procedure invocation in reasoner text."""
    verb: str                    # which pattern matched (apply / invoke / create_new / ...)
    procedure_name: str          # the name as it appeared (case preserved)
    args_text: Optional[str]     # free-text after 'to' or 'with', if any
    start: int                   # position in source text
    end: int


@dataclass
class DispatchOutcome:
    """Result of invoking one PatternMatch. Spliced back into reasoner context."""
    match: PatternMatch
    procedure_id: Optional[str]            # None if name didn't resolve
    object_id: Optional[str]               # the SessionObjectNode created/updated
    sub_prompt: str
    sub_response: str
    mutations_applied: int
    error: Optional[str] = None
    elapsed_seconds: float = 0.0
    # Phase 2A: when this invocation was triggered from inside another
    # procedure's body (not from the main reasoner), parent_object_id is
    # the session_object id of that parent. None for top-level invocations.
    parent_object_id: Optional[str] = None
    # Phase 2A: filled by the recursive scan in Sub-phase 2.4 with child
    # DispatchOutcomes when this invocation itself emitted CALL commands.
    # Empty list for leaf procedures.
    sub_outcomes: List["DispatchOutcome"] = None  # type: ignore[assignment]

    def __post_init__(self):
        # `sub_outcomes` defaults to None then mutated to [] for a cleaner
        # JSON serialization story (None is explicit "not populated yet";
        # an empty list means "ran but called nothing").
        if self.sub_outcomes is None:
            self.sub_outcomes = []


# ---- Mutation parsing --------------------------------------------------- #
# These regexes match the constrained mutation grammar inside the sub-LLM's
# response. Verbs are uppercase by convention to make accidental prose
# matches less likely. Case-insensitive in practice — easier on the LLM.

_MUT_ADD = re.compile(
    r"^\s*ADD\s+(?P<value>.+?)\s+TO\s+(?P<path>[A-Za-z_][\w\.]*)\s*$",
    re.IGNORECASE | re.MULTILINE,
)
_MUT_SET = re.compile(
    r"^\s*SET\s+(?P<path>[A-Za-z_][\w\.]*)\s*=\s*(?P<value>.+?)\s*$",
    re.IGNORECASE | re.MULTILINE,
)
_MUT_DELETE = re.compile(
    r"^\s*DELETE\s+(?P<path>[A-Za-z_][\w\.]*)\s*$",
    re.IGNORECASE | re.MULTILINE,
)

# Phase 2A: structured CALL command for sub-procedure invocations from
# within a procedure body. Distinct from the top-level free-text patterns
# (apply / invoke / using_the / etc.). The procedure body's directive
# instructs the sub-LLM to emit this exact shape; reliable parsing.
#
#   CALL VerifyNonNegativeEdges WITH instance=Dijkstra graph
#
# `name` matches the procedure's `.name` attribute (case-insensitive).
# `args` captures everything after WITH up to end-of-line, free-text.
_CALL_RE = re.compile(
    r"^\s*CALL\s+(?P<name>[A-Za-z][A-Za-z0-9_]*)"
    r"(?:\s+WITH\s+(?P<args>[^\n]+))?\s*$",
    re.IGNORECASE | re.MULTILINE,
)

_EDGE_WITH_WEIGHT_RE = re.compile(
    r"\(?\b(?P<src>[A-Za-z0-9_]+)\s*->\s*(?P<dst>[A-Za-z0-9_]+)\b\)?"
    r"(?:\s*,?\s*(?:has\s+)?(?:weight|weights?|w|cost)\s*)"
    r"(?P<weight>-?\d+(?:\.\d+)?)",
    re.IGNORECASE,
)

_ALGORITHM_NAMES = [
    ("Bellman-Ford", re.compile(r"\bBellman\s*-?\s*Ford\b", re.IGNORECASE)),
    ("Floyd-Warshall", re.compile(r"\bFloyd\s*-?\s*Warshall\b", re.IGNORECASE)),
    ("Dijkstra", re.compile(r"\bDijkstra(?:'s)?\b", re.IGNORECASE)),
    ("BFS", re.compile(r"\bBFS\b", re.IGNORECASE)),
    ("DFS", re.compile(r"\bDFS\b", re.IGNORECASE)),
    ("A*", re.compile(r"\bA\s*\*\b|\bA-star\b", re.IGNORECASE)),
    ("Kadane", re.compile(r"\bKadane(?:'s)?\b", re.IGNORECASE)),
]


def _parse_value(raw: str) -> Any:
    """Try JSON; fall back to literal string. Empty input becomes empty string."""
    raw = raw.strip()
    if not raw:
        return ""
    # Strip wrapping quotes if present (LLMs often add them)
    if (raw.startswith('"') and raw.endswith('"')) or (raw.startswith("'") and raw.endswith("'")):
        return raw[1:-1]
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, ValueError):
        return raw


# ---- ProcedureInvocationResolver --------------------------------------- #

def resolve_invocation_args(
    procedure: ProcedureNode,
    raw_args_text: str,
    original_question: str,
) -> dict:
    """Resolve a raw model invocation into canonical procedure args.

    Deterministic v1 resolver: it only knows the argument types that have
    caused real failures so far (`algorithm_name` and `instance_description`).
    Unknown single-input procedures keep the raw args as that input, preserving
    Phase-1 behavior for older tests/procedures.
    """
    raw = (raw_args_text or "").strip()
    question = (original_question or "").strip()
    input_names = _signature_input_names(procedure)
    resolved: Dict[str, str] = {}

    for name in input_names:
        field_value = _extract_field_value(raw, name)
        if field_value:
            if name == "instance_description" and not _has_concrete_edges(field_value):
                continue
            resolved[name] = field_value

    if "algorithm_name" in input_names and "algorithm_name" not in resolved:
        algo = _extract_algorithm(raw) or _extract_algorithm(question)
        if algo:
            resolved["algorithm_name"] = algo

    if "instance_description" in input_names and "instance_description" not in resolved:
        raw_instance = _extract_instance_description(raw)
        question_instance = _extract_instance_description(question)
        if raw_instance:
            resolved["instance_description"] = raw_instance
        elif question_instance:
            resolved["instance_description"] = question_instance

    unknown_inputs = [
        name for name in input_names
        if name not in {"algorithm_name", "instance_description"}
        and name not in resolved
    ]
    if raw and len(input_names) == 1 and unknown_inputs:
        resolved[unknown_inputs[0]] = raw

    return resolved


def _canonicalize_invocation_match(
    procedure: ProcedureNode,
    match: PatternMatch,
    original_question: str,
) -> tuple[Optional[PatternMatch], Optional[str]]:
    raw = (match.args_text or "").strip()
    input_names = _signature_input_names(procedure)
    if not input_names:
        return match, None

    resolved = resolve_invocation_args(procedure, raw, original_question)
    missing = [name for name in input_names if not str(resolved.get(name, "")).strip()]
    if missing:
        payload = {
            "type": "unresolved_procedure_args",
            "procedure": procedure.name,
            "missing": missing,
            "raw_args": raw,
        }
        return None, json.dumps(payload, ensure_ascii=False)

    canonical_args = _format_canonical_args(input_names, resolved, raw)
    return PatternMatch(
        verb=match.verb,
        procedure_name=match.procedure_name,
        args_text=canonical_args,
        start=match.start,
        end=match.end,
    ), None


def _signature_input_names(procedure: ProcedureNode) -> List[str]:
    inputs = (procedure.signature or {}).get("inputs") or []
    names: List[str] = []
    for item in inputs:
        if isinstance(item, dict):
            name = str(item.get("name") or "").strip()
        else:
            name = str(item).strip()
        if name:
            names.append(name)
    return names


def _extract_algorithm(*texts: str) -> Optional[str]:
    for text in texts:
        for canonical, pattern in _ALGORITHM_NAMES:
            if pattern.search(text or ""):
                return canonical
    return None


def _extract_instance_description(text: str) -> Optional[str]:
    text = (text or "").strip()
    if not text:
        return None
    field_value = _extract_field_value(text, "instance_description")
    if field_value and _has_concrete_edges(field_value):
        return field_value
    if _has_concrete_edges(text):
        return _canonical_edge_description(text)
    return None


def _extract_field_value(text: str, field_name: str) -> Optional[str]:
    if not text:
        return None
    pattern = re.compile(
        rf"\b{re.escape(field_name)}\s*=\s*"
        rf"(?P<value>\"[^\"]*\"|'[^']*'|.+?)(?=\s+[A-Za-z_]\w*\s*=|$)",
        re.IGNORECASE | re.DOTALL,
    )
    m = pattern.search(text)
    if not m:
        return None
    return m.group("value").strip().strip('"').strip("'")


def _has_concrete_edges(text: str) -> bool:
    return bool(_EDGE_WITH_WEIGHT_RE.search(text or ""))


def _canonical_edge_description(text: str) -> str:
    edges = []
    seen = set()
    for m in _EDGE_WITH_WEIGHT_RE.finditer(text or ""):
        src = m.group("src")
        dst = m.group("dst")
        weight = m.group("weight")
        key = (src, dst, weight)
        if key in seen:
            continue
        seen.add(key)
        edges.append(f"({src}->{dst}, weight {weight})")
    prefix = "directed graph" if re.search(r"\bdirected\b", text or "", re.IGNORECASE) else "graph"
    return f"{prefix} with edges {', '.join(edges)}" if edges else ""


def _format_canonical_args(input_names: List[str], resolved: Dict[str, str], raw: str) -> str:
    # If the model already emitted keyed concrete args, preserve that string.
    # This keeps clean invocations stable while still letting vague invocations
    # fall back to canonical fields from the original question.
    if raw and _raw_has_complete_concrete_fields(raw, input_names):
        return raw
    parts = []
    for name in input_names:
        value = resolved.get(name, "")
        parts.append(f"{name}={json.dumps(value, ensure_ascii=False)}")
    return " ".join(parts)


def _raw_has_complete_concrete_fields(raw: str, input_names: List[str]) -> bool:
    for name in input_names:
        value = _extract_field_value(raw, name)
        if not value:
            return False
        if name == "instance_description" and not _has_concrete_edges(value):
            return False
    return True


# ---- Version-chain name resolution (Phase 2A §5) ----------------------- #

def _build_name_index(procedures) -> Dict[str, ProcedureNode]:
    """Build a name -> active-head index from a set of procedures.

    Rules (PHASE2_PLAN.md §5.2):
      - Group procedures by lowercased name.
      - Within each group, exclude any whose provenance.deprecated is True.
      - Among the remaining, the active head is the one with no
        superseded_by_id set. If multiple candidates remain (data error),
        pick the highest version number.

    One-hop resolution: we don't walk long chains; we trust each
    ProcedureNode to carry an accurate superseded_by_id pointing at its
    direct successor (None on the head).
    """
    by_name: Dict[str, List[ProcedureNode]] = {}
    for proc in procedures:
        by_name.setdefault(proc.name.lower(), []).append(proc)

    index: Dict[str, ProcedureNode] = {}
    for name, group in by_name.items():
        # Filter deprecated
        live = [p for p in group if not p.provenance.deprecated]
        if not live:
            continue
        # Heads have no successor
        heads = [p for p in live if not p.superseded_by_id]
        if not heads:
            # All live procedures point forward — fall back to highest version
            heads = sorted(live, key=lambda p: -p.version)
        if len(heads) > 1:
            # Multiple heads — pick the highest version (deterministic)
            heads.sort(key=lambda p: -p.version)
        index[name] = heads[0]
    return index


# ---- Dispatcher --------------------------------------------------------- #

class Dispatcher:
    """Scans reasoner text for procedure invocations and executes them."""

    def __init__(self, procedure_index: Dict[str, ProcedureNode]):
        # raw_index keeps the by-id view so version-chain code can walk
        # parent/successor links explicitly.
        self.raw_index = dict(procedure_index)

        # procedure_index maps lowercased name -> the ACTIVE head of the
        # version chain (latest non-deprecated). Phase 2A version-chain
        # resolution lives here.
        self.procedure_index: Dict[str, ProcedureNode] = _build_name_index(
            procedure_index.values()
        )

    def resolve_name(self, name: str) -> Optional[ProcedureNode]:
        """Public version-chain-aware lookup. Returns the active head for
        `name` or None if no live procedure has that name."""
        return self.procedure_index.get(name.lower())

    # ---- pre-LLM args validation (Fix 3) ------------------------------- #

    def validate_args(
        self, proc: ProcedureNode, match: PatternMatch,
    ) -> Optional[str]:
        """Reject obviously malformed args BEFORE consuming an LLM call.

        Real-world bug — cs4 Dijkstra run, 2026-05-21: the model wrote
        finalization prose "I'll apply VerifyNonNegativeEdges to the
        instance and then use the VerifyShortestPath result to compose
        the answer." The apply_intent regex captured everything after
        "to" as args_text. The dispatcher then consumed an LLM call on
        meta-prose that mentioned ANOTHER procedure by name.

        Heuristic: if args_text mentions a DIFFERENT live procedure by
        name, the model is talking ABOUT procedures, not passing real
        arguments. Reject.

        Also reject empty args when the procedure declares required
        inputs — there's nothing for the sub-LLM to bind to.

        Returns None when args look acceptable, an explanation string
        otherwise. The caller wraps the explanation into a DispatchOutcome
        with `error=...` and zero budget consumption.
        """
        args = (match.args_text or "").strip()

        # Check 1: empty args but procedure declares inputs.
        if not args and proc.signature.get("inputs"):
            return "args_text is empty but procedure declares required inputs"

        # Check 2: args_text mentions another procedure by name (meta-prose).
        if args:
            own_name_lower = proc.name.lower()
            for other_proc in self.procedure_index.values():
                if other_proc.name.lower() == own_name_lower:
                    continue
                if re.search(
                    rf"\b{re.escape(other_proc.name)}\b",
                    args,
                    re.IGNORECASE,
                ):
                    return (
                        f"args_text references another procedure "
                        f"({other_proc.name}) — looks like meta-prose about "
                        f"results, not actual arguments"
                    )

        return None

    # ---- SCAN ---------------------------------------------------------- #

    def scan(self, text: str) -> List[PatternMatch]:
        """Find all procedure invocations in `text`. Overlapping matches
        from different patterns are collapsed: the earlier-starting
        (more specific) pattern wins.

        Example: "I'll apply X to Y" matches both `apply_intent`
        (covering "I'll apply X to Y") and `apply` (covering "apply X
        to Y"). The spans overlap; `apply_intent` starts earlier and
        is kept. This is why the regex set is fine even though
        sub-patterns are substrings of super-patterns.
        """
        matches: List[PatternMatch] = []
        for verb, pattern in _PATTERNS:
            for m in pattern.finditer(text):
                name = m.group("name")
                if not name or name.lower() not in self.procedure_index:
                    continue                       # silent miss
                try:
                    args = m.group("args")
                except IndexError:
                    args = None
                matches.append(PatternMatch(
                    verb=verb,
                    procedure_name=name,
                    args_text=(args.strip() if args else None),
                    start=m.start(),
                    end=m.end(),
                ))

        # Span-overlap dedupe. Sort by (start ASC, span_length DESC) so
        # that for identical starts the longest span wins, and for
        # nested matches the outer (earlier-starting) one wins.
        matches.sort(key=lambda x: (x.start, -(x.end - x.start)))
        accepted: List[PatternMatch] = []
        for m in matches:
            overlapping = any(
                not (m.end <= a.start or m.start >= a.end)
                for a in accepted
            )
            if overlapping:
                continue
            accepted.append(m)
        return accepted

    # ---- INVOKE -------------------------------------------------------- #

    def invoke(
        self,
        match: PatternMatch,
        session: SessionSubgraphController,
        llm_call: Callable[[str], str],
        budget: Optional[BudgetTracker] = None,
        existing_object_id: Optional[str] = None,
        parent_object_id: Optional[str] = None,
    ) -> DispatchOutcome:
        """Execute one matched invocation.

        Steps:
          1. Resolve procedure from index
          2. Create or reuse a SessionObjectNode for it
          3. Render procedure body as sub-prompt with bound args + current state
          4. Sub-LLM call (consumes one llm_call from budget)
          5. Parse mutations from response, apply to session state
          6. Return a DispatchOutcome for the reasoning loop to splice

        Phase 2A: when parent_object_id is set, the invocation is a
        sub-invocation called from within another procedure's body. We
        create a `sub_invocation_of` edge from the child's session_object
        to the parent's session_object so the call tree is queryable.
        """
        invoke_started_at = time.monotonic()
        proc = self.procedure_index.get(match.procedure_name.lower())
        if proc is None:
            return DispatchOutcome(
                match=match, procedure_id=None, object_id=None,
                sub_prompt="", sub_response="",
                mutations_applied=0, error="procedure not in index",
                elapsed_seconds=max(0.0, time.monotonic() - invoke_started_at),
                parent_object_id=parent_object_id,
            )

        # Fix 3 (2026-05-21): pre-LLM args validation. Reject obviously
        # malformed args BEFORE creating a session object or consuming
        # an LLM call. The cs4 bug consumed an LLM call (and created a
        # junk session object) on meta-prose args; this gate stops that.
        args_problem = self.validate_args(proc, match)
        if args_problem is not None:
            return DispatchOutcome(
                match=match, procedure_id=proc.id, object_id=None,
                sub_prompt="", sub_response="",
                mutations_applied=0,
                error=f"args validation failed: {args_problem}",
                elapsed_seconds=max(0.0, time.monotonic() - invoke_started_at),
                parent_object_id=parent_object_id,
            )

        # ProcedureInvocationResolver: turn model-written free text into
        # canonical procedure args before any object creation or sub-LLM call.
        # This prevents vague spans like "this directed weighted instance"
        # from replacing concrete user data such as `(b->c, weight -1)`.
        canonical_match, resolve_error = _canonicalize_invocation_match(
            proc, match, session.subgraph.query,
        )
        if resolve_error is not None or canonical_match is None:
            return DispatchOutcome(
                match=match, procedure_id=proc.id, object_id=None,
                sub_prompt="", sub_response="",
                mutations_applied=0,
                error=f"args resolution failed: {resolve_error}",
                elapsed_seconds=max(0.0, time.monotonic() - invoke_started_at),
                parent_object_id=parent_object_id,
            )
        match = canonical_match

        # Phase 2A: tag both the trigger text and the audit log with the
        # parent context, if any, so traceability survives sub-invocations.
        sub_prefix = (
            f"[sub-invocation of {parent_object_id}] " if parent_object_id else ""
        )

        # Create or reuse object
        obj_id = existing_object_id
        if obj_id is None or obj_id not in session.subgraph.nodes:
            initial_state = {field: _default_for_type(t)
                             for field, t in proc.state_schema.items()}
            obj_id = session.create_object(
                proc, initial_state,
                triggered_by=f"{sub_prefix}[{match.verb}] {match.procedure_name}",
            )
            # Wire the sub_invocation_of edge so the call tree is queryable.
            # Metadata carries the invocation args so future dedupe (2.4) can
            # check (parent_object_id, child_procedure_name, args_text) without
            # re-parsing the audit log.
            if parent_object_id is not None:
                from reasoning.composition import SUB_INVOCATION_OF
                session.add_edge(
                    src=obj_id,
                    dst=parent_object_id,
                    relation=SUB_INVOCATION_OF,
                    metadata={
                        "procedure_id": proc.id,
                        "procedure_name": proc.name,
                        "args_text": match.args_text or "",
                        "verb": match.verb,
                    },
                )

        # Render sub-prompt
        sub_prompt = _render_sub_prompt(proc, match, session, obj_id)

        # Budget check for the sub-LLM call
        if budget is not None:
            budget.consume("llm_call")

        # Sub-LLM call
        sub_response = llm_call(sub_prompt)

        # Parse + apply mutations
        n_applied = _apply_mutations(
            session, obj_id, sub_response,
            triggered_by=f"{sub_prefix}{match.procedure_name}[{match.verb}] response",
        )

        # Phase 2A: structured CALL scan. The sub-LLM may have emitted
        # `CALL <child> WITH <args>` lines, requesting sub-procedure
        # invocations from inside this procedure's body. Dispatch each
        # (with dedupe + budget enforcement) and aggregate results into
        # sub_outcomes.
        sub_outcomes = self._dispatch_call_commands(
            parent_object_id=obj_id,
            sub_response=sub_response,
            session=session,
            llm_call=llm_call,
            budget=budget,
        )

        return DispatchOutcome(
            match=match, procedure_id=proc.id, object_id=obj_id,
            sub_prompt=sub_prompt, sub_response=sub_response,
            mutations_applied=n_applied,
            elapsed_seconds=max(0.0, time.monotonic() - invoke_started_at),
            parent_object_id=parent_object_id,
            sub_outcomes=sub_outcomes,
        )

    # ---- direct-invoke API for JSON-tool answerers --------------------- #

    def invoke_by_name(
        self,
        name: str,
        args: Optional[Dict[str, Any]],
        session: SessionSubgraphController,
        llm_call: Callable[[str], str],
        budget: Optional[BudgetTracker] = None,
        existing_object_id: Optional[str] = None,
    ) -> DispatchOutcome:
        """Convenience: invoke a procedure without writing a free-text PatternMatch.

        Synthesizes an `invoke`-verb PatternMatch internally — saves the
        boilerplate JSON-tool answerers (e.g., answerer_v4) would otherwise
        need to fake. `args` is serialized to a JSON args_text string so the
        existing args-validation + canonicalization pipeline still applies.
        """
        # Format args as `key="value" key2="value2"` text — that's what the
        # Phase-2A resolver's _extract_field_value regex expects. JSON form
        # doesn't match because the resolver was built for free-text Option-2
        # invocations.
        args_text: Optional[str]
        if args is None or args == {}:
            args_text = None
        elif isinstance(args, dict):
            parts = []
            for k, v in args.items():
                # Escape inner quotes; the regex tolerates either ' or " bounds.
                s = str(v).replace('"', '\\"')
                parts.append(f'{k}="{s}"')
            args_text = " ".join(parts)
        else:
            args_text = str(args)
        synth_match = PatternMatch(
            verb="invoke",
            procedure_name=name,
            args_text=args_text,
            start=0,
            end=0,
        )
        return self.invoke(
            synth_match,
            session,
            llm_call,
            budget=budget,
            existing_object_id=existing_object_id,
        )

    # ---- structured CALL dispatch (Phase 2A) --------------------------- #

    def _dispatch_call_commands(
        self,
        *,
        parent_object_id: str,
        sub_response: str,
        session: SessionSubgraphController,
        llm_call: Callable[[str], str],
        budget: Optional[BudgetTracker],
    ) -> List[DispatchOutcome]:
        """Scan `sub_response` for `CALL X WITH args` commands and dispatch
        each as a sub-invocation of `parent_object_id`.

        Per acceptance criterion #3, dedupes on
        (parent_object_id, procedure_id, args_text). A second `CALL X WITH
        same args` from the same parent is silently skipped — distinct args
        are how the agent explicitly requests a re-run.

        Budget enforcement: each child consumes one `fan_out` and pushes
        the recursion depth tracker. On `BudgetExhausted`, the remaining
        CALL commands are skipped but already-completed children survive
        in the returned list.
        """
        outcomes: List[DispatchOutcome] = []
        seen_keys: set = set()                # within-this-response dedupe

        for m in _CALL_RE.finditer(sub_response):
            child_name = m.group("name") or ""
            child_args = (m.group("args") or "").strip() or None

            # Resolve the child procedure
            child_proc = self.procedure_index.get(child_name.lower())
            if child_proc is None:
                # Silent miss: unknown procedure name (matches Phase-1 scan behaviour)
                continue

            # Cross-response dedupe: (parent, proc_id, args) already invoked?
            existing_id = find_existing_sub_invocation(
                session, parent_object_id, child_proc.id, child_args,
            )
            if existing_id is not None:
                # Acceptance criterion #3: silently skip duplicate intent
                continue

            # Within-response dedupe (same CALL appears twice in one body's output)
            key = (parent_object_id, child_proc.id, child_args or "")
            if key in seen_keys:
                continue
            seen_keys.add(key)

            # Budget gates
            if budget is not None:
                try:
                    budget.push_recursion()
                except BudgetExhausted:
                    # Recursion cap hit — stop scanning further CALLs cleanly
                    break
                try:
                    budget.consume("fan_out")
                except BudgetExhausted:
                    budget.pop_recursion()
                    break

            try:
                child_match = PatternMatch(
                    verb="call",
                    procedure_name=child_proc.name,
                    args_text=child_args,
                    start=m.start(),
                    end=m.end(),
                )
                child_outcome = self.invoke(
                    child_match, session, llm_call,
                    budget=budget,
                    parent_object_id=parent_object_id,
                )
                outcomes.append(child_outcome)
            except BudgetExhausted:
                # Recursive invoke ran out of budget mid-execution. Stop.
                if budget is not None:
                    budget.pop_recursion()
                break
            else:
                if budget is not None:
                    budget.pop_recursion()

        return outcomes


def flatten_dispatch_outcomes(outcomes: List[DispatchOutcome]) -> List[DispatchOutcome]:
    """Walk dispatch outcomes recursively, yielding each outcome plus all
    of its sub_outcomes (children invoked via CALL within a procedure's
    body). Used to give meta-procedure predicates a flat view of every
    invocation in the session, not just the top-level ones.

    Top-level outcomes have parent_object_id=None; children have a
    non-None parent_object_id pointing at their composer's session_object.
    Predicates that want top-level-only can filter accordingly.
    """
    flat: List[DispatchOutcome] = []
    def _walk(o: DispatchOutcome) -> None:
        flat.append(o)
        for child in o.sub_outcomes or []:
            _walk(child)
    for top in outcomes:
        _walk(top)
    return flat


def find_existing_sub_invocation(
    session: SessionSubgraphController,
    parent_object_id: str,
    procedure_id: str,
    args_text: Optional[str],
) -> Optional[str]:
    """Look up an existing child session_object created as a sub-invocation
    of `parent_object_id` for the given procedure + args.

    The dedupe key per acceptance criterion #3:
        (parent_object_id, child_procedure_name (via procedure_id), args_text)

    Distinct args produce distinct children — that's how the agent
    explicitly requests a re-run with different inputs.

    Returns the child's session_object id if found, None otherwise.
    Used by the recursive scan in Sub-phase 2.4.
    """
    from reasoning.composition import SUB_INVOCATION_OF
    needle = args_text or ""
    for edge in session.subgraph.edges:
        if edge.relation != SUB_INVOCATION_OF:
            continue
        if edge.dst != parent_object_id:
            continue
        meta = edge.metadata or {}
        if meta.get("procedure_id") != procedure_id:
            continue
        if (meta.get("args_text") or "") != needle:
            continue
        return edge.src
    return None


# ---- helpers ------------------------------------------------------------ #

def _default_for_type(type_str: str) -> Any:
    """Best-effort initial value for a field given a Python-style type hint string."""
    t = type_str.strip().lower()
    if t.startswith("list"):
        return []
    if t.startswith("dict") or t.startswith("mapping"):
        return {}
    if t in ("str", "string"):
        return ""
    if t in ("int", "integer"):
        return 0
    if t in ("float", "number"):
        return 0.0
    if t in ("bool", "boolean"):
        return False
    return None


def _render_sub_prompt(
    proc: ProcedureNode,
    match: PatternMatch,
    session: SessionSubgraphController,
    obj_id: str,
) -> str:
    """Build the sub-prompt that asks the sub-LLM to run the procedure body."""
    current_state = session.subgraph.nodes[obj_id]["state"]
    args = match.args_text or ""
    query = session.subgraph.query or ""
    return (
        f"You are executing the {proc.name} procedure.\n\n"
        f"Purpose: {proc.purpose}\n\n"
        f"Body / instructions:\n{proc.body}\n\n"
        f"Original user question/context for resolving references like 'this instance':\n"
        f"{query}\n\n"
        f"Invocation args: {args}\n\n"
        f"Current state of this procedure instance:\n"
        f"{json.dumps(current_state, indent=2)}\n\n"
        f"Emit mutation commands to update the state. Use ONLY these verbs:\n"
        f"  ADD <value> TO <field_path>\n"
        f"  SET <field_path> = <value>\n"
        f"  DELETE <field_path>\n"
        f"  DONE\n"
        f"Output the commands one per line. Use JSON syntax for complex values "
        f"(strings in double quotes, lists in []). End with DONE.\n"
        f"After DONE, you may include a brief natural-language summary on the next line(s).\n"
    )


def _apply_mutations(
    session: SessionSubgraphController,
    obj_id: str,
    response: str,
    triggered_by: str,
) -> int:
    """Parse mutation commands from `response` and apply to session.

    Returns the number of mutations successfully applied. Malformed
    commands are silently skipped (graceful degradation — bad commands
    shouldn't crash the reasoning loop).
    """
    n = 0

    # ADD <value> TO <path>  (append-to-list semantics via read+modify+write)
    for m in _MUT_ADD.finditer(response):
        path = m.group("path")
        value = _parse_value(m.group("value"))
        try:
            current = session.read_object(obj_id, path)
        except Exception:
            current = []
        if not isinstance(current, list):
            current = [current] if current is not None else []
        new_list = current + [value]
        try:
            session.update_object(obj_id, path, new_list, triggered_by)
            n += 1
        except Exception:
            pass

    # SET <path> = <value>
    for m in _MUT_SET.finditer(response):
        path = m.group("path")
        value = _parse_value(m.group("value"))
        try:
            session.update_object(obj_id, path, value, triggered_by)
            n += 1
        except Exception:
            pass

    # DELETE <path>
    for m in _MUT_DELETE.finditer(response):
        path = m.group("path")
        try:
            session.delete_object(obj_id, path, triggered_by)
            n += 1
        except Exception:
            pass

    return n
