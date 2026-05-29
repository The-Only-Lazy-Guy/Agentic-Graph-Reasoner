# Phase-1 implementation plan — reasoning substrate

**Status:** plan phase. No code written yet. Plan must be approved before implementation.

**Parent doc:** `REASONING_ARCHITECTURE.md` — defines the architecture; this doc defines the build.

**Date:** 2026-05-20

---

## 0. Goal of Phase 1

A working substrate where:

1. The reasoner can **create** procedures and stateful objects during a session
2. State can be **mutated** (full CRUD) across reasoning steps with every change journaled to an audit log
3. Procedures **dispatch** via free-text + regex pattern matching from the reasoner's output
4. Each session has its own **subgraph** (separate from long-term memory) that is **always persisted** to disk
5. **Failure patterns** are first-class nodes, retrieval-boosted 1.4× over facts
6. **Budgets** are enforced at dispatch time (LLM calls, hops, recursion, fan-out)
7. **Provenance** is recorded on every created node
8. **Consolidation** from session subgraph → long-term graph runs at session end with the citation+example gate

What's NOT in Phase 1 (deferred to Phase 2/3):
- Procedure composition (calls/inherits/specializes edges)
- Procedure version chains and darwinian decay
- Macro extraction / multi-scale chunking
- Meta-procedures
- Dynamic embeddings (LLM-rerank approximation)
- JSON-grammar dispatch (Phase 1 is regex-based)

Phase 1 ends when the front-end can answer questions through the new substrate alongside the legacy one-shot path, with a toggle, and the seed procedure `VerifyAlgorithmPreconditions` demonstrably runs end-to-end on a real Dijkstra question.

---

## 1. File structure

All new code lives under `graph_final/reasoning/`. Nothing existing is replaced in Phase 1.

```
graph_final/
├── reasoning/                            ← NEW package
│   ├── __init__.py
│   ├── schemas.py                        ← typed dataclasses (Procedure, FailurePattern, etc.)
│   ├── session_subgraph.py               ← per-session graph, mutation interface
│   ├── audit_log.py                      ← journaling of every mutation
│   ├── dispatcher.py                     ← regex pattern matcher → procedure invoker
│   ├── budgets.py                        ← BudgetTracker (call, hop, depth, fan-out)
│   ├── retrieval_boost.py                ← anchor retrieval with failure-pattern weighting
│   ├── consolidation.py                  ← session-end promotion to long-term graph
│   ├── reasoning_loop.py                 ← orchestrator: ties everything together
│   ├── procedures/
│   │   ├── __init__.py
│   │   └── verify_algorithm_preconditions.py   ← seed procedure
│   └── tests/
│       ├── test_session_subgraph.py
│       ├── test_audit_log.py
│       ├── test_dispatcher.py
│       ├── test_budgets.py
│       ├── test_retrieval_boost.py
│       └── test_reasoning_loop.py
│
├── api/                                  ← FRONT-END changes
│   └── frontend_api.py                   ← MODIFIED: toggle to new path via env var
│
├── data/
│   └── session_subgraphs/                ← NEW: per-session JSON dumps (always-persist)
│       └── {session_id}/
│           ├── subgraph.json
│           └── audit_log.jsonl
│
└── REASONING_ARCHITECTURE.md             (existing — design doc)
    PHASE1_PLAN.md                        (this file)
```

---

## 2. Data model (`reasoning/schemas.py`)

Typed dataclasses. Pydantic if it plays nicely with existing project code, else plain `@dataclass` with `dataclasses_json`. All schemas serialize to JSON for persistence.

```python
from dataclasses import dataclass, field
from typing import Any, Literal
from datetime import datetime

NodeType = Literal[
    "fact", "claim", "example", "summary", "hub", "bridge",
    "hypothesis", "application",          # existing
    "procedure", "failure_pattern", "session_object",   # new
]


@dataclass
class Provenance:
    created_in_session_id: str
    validating_examples: list[str] = field(default_factory=list)
    depends_on: list[str] = field(default_factory=list)
    citation_count: int = 0
    citation_decay: float = 1.0
    last_modified: str = ""               # ISO8601
    deprecated: bool = False
    deprecation_reason: str | None = None


@dataclass
class ProcedureNode:
    id: str
    name: str                              # short slug
    purpose: str                           # one sentence
    when_to_use: str                       # paragraph
    signature: dict                        # {inputs: [...], outputs: [...]}
    state_schema: dict                     # field name → JSON type
    body: str                              # sub-prompt template
    example_use: dict | None               # the worked-instance gate
    provenance: Provenance
    node_type: NodeType = "procedure"


@dataclass
class FailurePatternNode:
    id: str
    name: str
    attempted_approach: str
    failure_condition: str
    failure_mechanism: str
    replacement: str | None                # ref to procedure or fact
    example_failure_case: dict | None
    provenance: Provenance
    node_type: NodeType = "failure_pattern"


@dataclass
class SessionObjectNode:
    """A per-session instance of a Procedure with current mutable state."""
    id: str
    procedure_id: str                      # ref to the underlying ProcedureNode
    name: str                              # human-readable, usually procedure.name
    state: dict                            # current state matching procedure.state_schema
    created_step: int                      # which step of the session created this
    provenance: Provenance
    node_type: NodeType = "session_object"


@dataclass
class AuditEntry:
    session_id: str
    step_index: int
    object_id: str
    operation: Literal["create", "read", "update", "delete"]
    field_path: str                        # dotted: "state.visited_nodes"
    old_value: Any                         # null for create
    new_value: Any                         # null for delete/read
    triggered_by_text: str                 # the reasoning snippet
    timestamp: str                         # ISO8601


@dataclass
class SessionSubgraph:
    session_id: str
    query: str
    graph_id: str                          # ref to long-term graph
    nodes: dict[str, Any]                  # node_id → node (any of the new node types)
    edges: list[dict]                      # {src, dst, relation, metadata}
    audit_log: list[AuditEntry]
    step_count: int
    started_at: str
    ended_at: str | None
```

---

## 3. Session subgraph + persistence (`reasoning/session_subgraph.py`)

API:

```python
class SessionSubgraphController:
    def __init__(self, session_id: str, query: str, graph_id: str):
        ...

    # CRUD for session objects
    def create_object(self, procedure: ProcedureNode, initial_state: dict, triggered_by: str) -> str:
        """Returns new SessionObjectNode id. Journals 'create' to audit log."""

    def read_object(self, object_id: str, field_path: str = "") -> Any:
        """Returns the value at the given dotted path. Journals 'read' to audit log."""

    def update_object(self, object_id: str, field_path: str, new_value: Any, triggered_by: str) -> None:
        """Full CRUD update at any dotted path. Journals old + new to audit log."""

    def delete_object(self, object_id: str, field_path: str | None, triggered_by: str) -> None:
        """If field_path None, deletes whole object. Else clears that field. Journals."""

    # Bookkeeping
    def add_failure_pattern(self, failure: FailurePatternNode, triggered_by: str) -> str:
        ...

    def add_edge(self, src: str, dst: str, relation: str) -> None:
        ...

    def step(self) -> None:
        """Advance step counter. Called once per reasoning iteration."""

    # Persistence (always-persist setting)
    def persist(self) -> Path:
        """Writes subgraph.json + audit_log.jsonl to data/session_subgraphs/{session_id}/."""

    def close(self) -> Path:
        """Marks ended_at, persists, returns the persistence path."""
```

**Mutation operations always journal first, then mutate.** If journaling raises, mutation does not occur. This guarantees the audit log is always consistent with state.

Audit log uses **JSONL** (one JSON per line) so streaming/append is cheap and partial reads are safe.

---

## 4. Audit log (`reasoning/audit_log.py`)

Thin module. Two responsibilities:

```python
class AuditLogger:
    def __init__(self, session_id: str, path: Path):
        ...

    def journal(self, entry: AuditEntry) -> None:
        """Atomic-append to {path}/audit_log.jsonl. Flushes immediately."""

    def replay(self, up_to_step: int | None = None) -> list[AuditEntry]:
        """Returns entries in order, optionally truncated."""

    def diff(self, object_id: str, from_step: int, to_step: int) -> list[AuditEntry]:
        """Returns mutations on a specific object across a step range."""

    def reconstruct_state(self, object_id: str, at_step: int) -> dict:
        """Replay mutations up to step, return reconstructed state.
        Used for debugging: 'what was visited_nodes at step 4?'"""
```

`reconstruct_state` is the load-bearing debug primitive for full-CRUD. Without it, a wrong answer 6 months from now is unsolvable.

---

## 5. Dispatcher (`reasoning/dispatcher.py`)

Regex pattern matcher over reasoner output.

```python
@dataclass
class DispatchResult:
    matched: bool
    procedure_name: str | None
    args_text: str | None              # raw text of matched argument span
    invoke_position: tuple[int, int]   # (start, end) in source text


class Dispatcher:
    def __init__(self, procedure_index: dict[str, ProcedureNode]):
        """procedure_index keyed by procedure.name (lowercase)."""

    def scan(self, text: str) -> list[DispatchResult]:
        """
        Scans reasoner output for procedure-invocation patterns.
        Patterns (in priority order):
          - r'apply\s+(?P<name>\w+)\s+to\s+(?P<args>.+?)(?:\n|$)'
          - r'invoke\s+(?P<name>\w+)(?:\s+with\s+(?P<args>.+?))?(?:\n|$)'
          - r'using\s+the\s+(?P<name>\w+)\s+procedure'
          - r'create\s+a?\s*new\s+(?P<name>\w+)\s+(?:object|instance)'
          - r'I\'?ll\s+(?:now\s+)?apply\s+(?P<name>\w+)'

        Match name (lowercased) against procedure_index. Only emit DispatchResult
        for matches that resolve to a known procedure.

        Audit log records every match attempt — including misses — for dispatch coverage.
        """

    def invoke(self, dr: DispatchResult, session: SessionSubgraphController, llm_call: Callable) -> dict:
        """
        Execute the matched procedure:
          1. Look up procedure in index
          2. Bind args_text to procedure.signature.inputs (best-effort)
          3. Render procedure.body as a sub-prompt with bindings
          4. Call llm_call(sub_prompt) → result text
          5. Parse result for state mutations (regex: 'set X to Y', 'add Z to L', etc.)
          6. Apply mutations via session.update_object / create_object / etc.
          7. Return result text for splicing back into reasoner context
        """
```

**Step 5 — mutation parsing — is the fragile part.** Initial version uses naive patterns. If a procedure says "add `node_a` to visited_nodes", the dispatcher will issue a session.update_object call with append semantics. Documented limits: cannot handle arbitrary mutation grammars; procedures should be written with simple, predictable mutation language.

---

## 6. Budgets (`reasoning/budgets.py`)

```python
@dataclass
class Budgets:
    max_llm_calls: int = 3
    max_hops: int = 1
    max_recursion_depth: int = 4
    max_session_subgraph_size: int = 50
    max_composition_fan_out: int = 5
    max_total_tokens: int = 2048


class BudgetTracker:
    def __init__(self, budgets: Budgets):
        self.budgets = budgets
        self.llm_calls_used = 0
        self.hops_used = 0
        self.recursion_depth = 0
        self.subgraph_size = 0
        self.fan_out_this_step = 0
        self.tokens_used = 0

    def check(self, op: str, amount: int = 1) -> bool:
        """Returns True if op (which would consume `amount` of budget) is allowed."""

    def consume(self, op: str, amount: int = 1) -> None:
        """Records consumption. Raises BudgetExhausted if check would fail."""

    def push_recursion(self) -> None:
        """Increments recursion depth (procedure-in-procedure). Raises if exceeded."""

    def pop_recursion(self) -> None:
        ...

    def summary(self) -> dict:
        """Snapshot of current usage. Goes into the session metadata."""


class BudgetExhausted(Exception):
    """Reasoning loop catches this and triggers graceful 'finalize with what we have'."""
```

When `BudgetExhausted` is raised, the reasoning loop catches, injects a `<budget_exhausted>` system message into the next prompt, and lets the reasoner produce its final answer with whatever state it has.

---

## 7. Retrieval boost (`reasoning/retrieval_boost.py`)

Thin wrapper around the existing `anchor_retrieval.retrieve_anchors_v2`. Re-weights failure_pattern nodes by 1.4× before topk selection.

```python
def retrieve_with_failure_boost(
    question: str,
    graph: MemoryGraph,
    k: int,
    failure_boost: float = 1.4,
    **kwargs,
) -> list[str]:
    """
    1. Run normal anchor retrieval → get top-2k candidates with similarity scores
    2. For each candidate whose node_type == 'failure_pattern',
       multiply similarity by failure_boost
    3. Re-sort by adjusted score, take top-k
    """
```

Tuning failure_boost is a knob — start at 1.4, observe how often failure_patterns surface vs ignore-rate, adjust.

---

## 8. Consolidation (`reasoning/consolidation.py`)

End-of-session pipeline.

```python
@dataclass
class ConsolidationDecision:
    node_id: str
    decision: Literal["promote", "keep_in_pool", "expire"]
    reason: str


class Consolidator:
    def __init__(self, long_term_graph: MemoryGraph, promotion_threshold: int = 3):
        ...

    def consolidate(self, session: SessionSubgraph) -> list[ConsolidationDecision]:
        """
        For each new node (procedure, failure_pattern, session_object) in session:
          - Has citation_count across distinct sessions reached promotion_threshold?
            (Uses existing hypothesis_pool's session-counting machinery)
          - Does it have at least one validating example_use?
          - Are all depends_on nodes already in long-term graph?
        If all three: copy node to long_term_graph, increment citation counts on
        referenced long-term nodes, record promotion edge.
        Else: leave in pool (session subgraph remains in cold storage on disk).

        Also handles: deprecation check for procedures whose newer versions have
        dominated. (Phase 1: deprecation is parent-pointer-aware but versioning
        edges are minimal — full version chains are Phase 2.)
        """
```

Promotion threshold M_PROMOTION = 3, deprecation window K = 30 (resolved defaults).

---

## 9. Reasoning loop orchestrator (`reasoning/reasoning_loop.py`)

The main entry point. Replaces what `_run_graph_agent` does today, end-to-end.

```python
@dataclass
class ReasoningRequest:
    question: str
    graph_id: str
    k_anchors: int = 12
    budgets: Budgets = field(default_factory=Budgets)


@dataclass
class ReasoningResult:
    answer: str
    reasoning_trace: str
    session_subgraph_path: Path
    audit_summary: dict
    consolidation_decisions: list[ConsolidationDecision]
    budget_usage: dict


def run_reasoning(req: ReasoningRequest, llm_call: Callable) -> ReasoningResult:
    """
    1. Load graph_id's long-term graph
    2. Retrieve anchors with failure_boost
    3. Build prompt with anchor-filtered context
    4. Initialize SessionSubgraphController, BudgetTracker, Dispatcher (with procedure index
       built from procedures retrieved alongside facts)
    5. Reasoning loop:
       step = 0
       while step < max_iter and not budget_tracker.exhausted():
           prompt = build_step_prompt(req, session, results_so_far)
           output = llm_call(prompt)
           budget_tracker.consume("llm_call")
           dispatch_results = dispatcher.scan(output)
           for dr in dispatch_results:
               result = dispatcher.invoke(dr, session, llm_call)
               results_so_far.append(result)
               budget_tracker.consume("llm_call")
           session.step()
           if final_answer_detected(output):
               break
           step += 1
       answer = extract_answer(output)
    6. session.close() → persists subgraph + audit log
    7. consolidator.consolidate(session) → promotion decisions
    8. Return ReasoningResult
    """
```

**This is the heart of Phase 1.** Everything else is plumbing.

---

## 10. Front-end integration (`api/frontend_api.py` changes)

Toggle via env var `REASONING_MODE` with values `legacy` (one-shot, current behavior) and `substrate` (new path).

```python
# At top of frontend_api.py
REASONING_MODE = os.environ.get("REASONING_MODE", "legacy")

# In _run_graph_agent:
def _run_graph_agent(req: RunRequest, emit: Optional[EmitFn] = None) -> Dict[str, Any]:
    if REASONING_MODE == "substrate":
        from reasoning.reasoning_loop import run_reasoning, ReasoningRequest, Budgets
        from reasoning.schemas import SessionSubgraph
        # ... build ReasoningRequest from RunRequest, call run_reasoning(), adapt result
        # back into the same payload contract used by the front-end UI
        return _adapt_reasoning_result_to_payload(...)
    else:
        # existing one-shot code path, unchanged
        ...
```

Why a toggle vs hard replacement: lets you A/B compare on the same questions in the UI without losing the working baseline. Set `REASONING_MODE=substrate` in run_api.bat to test; unset to revert.

`_adapt_reasoning_result_to_payload` is a small adapter ensuring the new path's output looks structurally identical to the legacy output — same `session`, `trace`, `metrics`, `packet` shapes. The front-end UI doesn't need to know which path produced the answer.

---

## 11. Seed procedure (`reasoning/procedures/verify_algorithm_preconditions.py`)

Stored as a Python dict (or YAML) that's loaded into the long-term graph on first run, then treated like any other procedure node.

```python
VERIFY_ALGORITHM_PRECONDITIONS = ProcedureNode(
    id="proc_verify_algorithm_preconditions_v1",
    name="VerifyAlgorithmPreconditions",
    purpose="Check whether a named algorithm's stated preconditions hold for a given problem instance.",
    when_to_use=(
        "Use when the question asks whether a specific algorithm can be applied to a given "
        "graph, dataset, or input, OR when the user is debugging a suspected algorithm "
        "misuse. Skip if the question is purely about algorithm description or comparison "
        "without a concrete instance."
    ),
    signature={
        "inputs": [
            {"name": "algorithm_name", "type": "str"},
            {"name": "instance_description", "type": "str"},
        ],
        "outputs": [
            {"name": "preconditions_satisfied", "type": "bool"},
            {"name": "violated_preconditions", "type": "list[str]"},
            {"name": "deferred_preconditions", "type": "list[str]"},
            {"name": "recommended_alternative", "type": "str | None"},
        ],
    },
    state_schema={
        "preconditions_checked": "list[str]",
        "preconditions_violated": "list[str]",
        "preconditions_deferred": "list[str]",
        "evidence_for_violations": "dict[str, str]",   # precondition → evidence text
    },
    body=(
        "You are verifying preconditions of the {algorithm_name} algorithm against the "
        "following instance:\n\n{instance_description}\n\n"
        "Step 1: List the algorithm's stated preconditions. For each, mark it as one of:\n"
        "  - SATISFIED if the instance meets it\n"
        "  - VIOLATED if the instance clearly fails it (cite evidence)\n"
        "  - DEFERRED if checking would require information not available\n\n"
        "Step 2: For each VIOLATED precondition, briefly state the evidence in the instance.\n\n"
        "Step 3: If any precondition is VIOLATED, suggest the recommended alternative algorithm "
        "from the background facts.\n\n"
        "Update your verification state:\n"
        "  - add to preconditions_checked: each precondition you considered\n"
        "  - add to preconditions_violated: each precondition that's clearly violated\n"
        "  - add to preconditions_deferred: each precondition you couldn't check\n"
        "  - for each violation, set evidence_for_violations[precondition_name] = <evidence>\n\n"
        "Output:\n"
        "preconditions_satisfied: true if preconditions_violated is empty AND preconditions_deferred "
        "is empty; false otherwise.\n"
        "violated_preconditions: copy of preconditions_violated\n"
        "deferred_preconditions: copy of preconditions_deferred\n"
        "recommended_alternative: the suggested alternative, or null if all preconditions satisfied"
    ),
    example_use={
        "session_id": "<seed>",
        "inputs": {
            "algorithm_name": "Dijkstra",
            "instance_description": "Directed graph with 5 vertices; edges include (a→b, weight 3), (b→c, weight -1), (a→c, weight 5).",
        },
        "final_state": {
            "preconditions_checked": ["all edges nonnegative", "no negative cycles", "single source defined"],
            "preconditions_violated": ["all edges nonnegative"],
            "preconditions_deferred": [],
            "evidence_for_violations": {
                "all edges nonnegative": "Edge b→c has weight -1.",
            },
        },
        "final_output": {
            "preconditions_satisfied": False,
            "violated_preconditions": ["all edges nonnegative"],
            "deferred_preconditions": [],
            "recommended_alternative": "Bellman-Ford",
        },
    },
    provenance=Provenance(
        created_in_session_id="<seed>",
        validating_examples=["<seed>"],
        depends_on=[],
        citation_count=0,
    ),
)
```

This procedure exercises **everything**:
- Multi-step state mutation (preconditions_checked grows step by step)
- Full CRUD (`evidence_for_violations` is a dict, populated key by key)
- Failure-pattern adjacency (when preconditions are violated, the recommended alternative often points at the failure_pattern node)
- Audit log is non-trivial (5+ mutations per invocation)
- Provenance + example_use (already populated, meets the validation gate)

A second seed procedure can be added later, but Phase 1 acceptance requires this one to work end-to-end.

---

## 12. Sub-phases inside Phase 1 (build order)

Strict order. Each sub-phase has a clear acceptance test before moving to the next.

### 1.1 — Skeleton + schemas
Build `reasoning/schemas.py` + tests. Acceptance: round-trip serialize every dataclass through JSON.

### 1.2 — Session subgraph + audit log
Build `session_subgraph.py` + `audit_log.py` + tests. Acceptance: full-CRUD sequence on a SessionObjectNode produces a valid audit log; `reconstruct_state(step=N)` returns correct intermediate state.

### 1.3 — Budgets
Build `budgets.py` + tests. Acceptance: BudgetTracker correctly raises `BudgetExhausted` at each limit; `summary()` matches consumed amounts.

### 1.4 — Retrieval boost
Build `retrieval_boost.py` + tests. Acceptance: on a synthetic mixed-node graph with both facts and failure_patterns, retrieval with boost surfaces failure_patterns more often than baseline.

### 1.5 — Dispatcher
Build `dispatcher.py` + tests. Acceptance: scan() on a hand-written reasoning trace returns expected match list; invoke() with a stub LLM correctly mutates session state.

### 1.6 — Seed procedure + consolidation
Build the seed procedure file + `consolidation.py` + tests. Acceptance: seed procedure can be loaded into a graph and consolidate correctly.

### 1.7 — Reasoning loop orchestrator
Build `reasoning_loop.py` + tests. Acceptance: end-to-end run on the Dijkstra question produces a clean answer plus a session subgraph plus an audit log.

### 1.8 — Front-end toggle
Modify `frontend_api.py` to add `REASONING_MODE` env var + adapter. Acceptance: front-end serves the same question via both paths; outputs are structurally compatible.

### 1.9 — Integration test
Run the same battery of eval questions through both paths. Acceptance: substrate path produces clean (no leak) answers on at least the dijkstra question with VerifyAlgorithmPreconditions visibly fired in the audit log.

---

## 13. Test plan

Three layers:

**Unit tests** (each sub-phase has its own — listed above)

**Integration tests** (after 1.7):
- Run reasoning loop on dijkstra question end-to-end with stub LLM that returns canned outputs simulating "apply VerifyAlgorithmPreconditions"
- Assert: session subgraph contains SessionObjectNode of the procedure; audit log has ≥3 mutation entries; budget tracker shows usage within limits

**End-to-end tests** (after 1.8):
- Through the live front-end with `REASONING_MODE=substrate`
- Same 6 questions from the earlier eval (brendrian, tarsil, naroth × sufficient + partial) plus the dijkstra question
- Manually inspect: no ID leaks, no refusal, audit logs persisted, consolidation decisions logged
- Compare answer quality against legacy path on the same questions

Eval rubric stays mostly the same as `_sft_eval.py`: structural pass, correctness probes, meta-leak, node-id leak. New additions:
- **dispatch_fired**: did the dispatcher actually invoke at least one procedure
- **audit_log_consistent**: every mutation has matching old/new and replay reconstructs the final state
- **budget_within_limits**: no usage exceeded the configured caps

---

## 14. Acceptance criteria for Phase 1 complete

All of the following must hold:

1. The seed procedure `VerifyAlgorithmPreconditions` runs end-to-end on the Dijkstra question, producing a SessionObjectNode with `state.preconditions_violated = ["all edges nonnegative"]` (or equivalent semantic content).
2. The session subgraph is persisted to `data/session_subgraphs/{session_id}/` with valid JSON + JSONL.
3. The audit log has at least 3 mutation entries for the dispatch, and `reconstruct_state` correctly replays them.
4. The front-end toggle works: same `RunRequest` produces compatible payloads from both `legacy` and `substrate` modes; UI can render either without modification.
5. Failure-pattern retrieval boost is observable: when the graph contains a relevant failure_pattern, it surfaces in anchors more often than its raw similarity score would predict.
6. No regression in the existing rubric: substrate-path answers on the 6 eval cells pass structural + meta-leak + node-id-leak checks at the same rate as the legacy path.
7. Budget enforcement is visible in audit summaries: every session reports usage of every budget category.
8. Consolidation runs without crash at every session end and produces a `ConsolidationDecision` list (which may be empty for early sessions).

---

## 15. What we'll know after Phase 1

By the time these acceptance criteria are met, we'll have answered:

- Does the regex dispatcher fire reliably enough to make procedures useful, or do we need to upgrade to JSON-grammar earlier than Phase 2?
- Does full-CRUD mutation get used in ways the agent finds natural, or do we observe a pattern where 95% of mutations are append-only and the choice was overkill?
- Does the seed procedure actually improve answer quality on Dijkstra-class questions, or does the substrate add latency without changing the answer?
- Is the audit log size manageable, or does verbose journaling balloon disk usage faster than expected?
- Are sessions producing consolidation-worthy artifacts at a useful rate, or is the citation threshold M=3 too high/low?

These answers feed directly into Phase-2 design decisions.

---

## 16. Open questions for user before code starts

| Question | Default if not answered |
|---|---|
| Pydantic vs plain `@dataclass` + `dataclasses_json`? | plain @dataclass — fewer dependencies |
| Use `uuid4` for node IDs or human-readable slugs like `proc_verify_algo_preconditions_001`? | uuid4 for runtime, slug for seed procedures |
| Should the dispatcher emit SSE events for the front-end's streaming endpoint, mirroring the existing trace format? | yes — keeps UI's reasoning panel populated |
| Run Phase 1 against the user's local opencode (GLM-4.7-Flash) or wait until a different model is available? | local opencode, same as current front-end |

None of these are blockers — defaults are reasonable. If user has preferences, lock them in now.

---

## 17. Estimated effort

Sub-phase | Lines | Time
---|---|---
1.1 Schemas | ~200 | 0.5 day
1.2 Session subgraph + audit log | ~400 | 1 day
1.3 Budgets | ~150 | 0.5 day
1.4 Retrieval boost | ~100 | 0.5 day
1.5 Dispatcher | ~300 | 1 day
1.6 Seed procedure + consolidation | ~250 | 0.5 day
1.7 Reasoning loop | ~400 | 1.5 days
1.8 Front-end toggle | ~150 | 0.5 day
1.9 Integration test | ~200 | 1 day

**Total: ~2150 lines, ~7 working days** if no significant blockers.

The reasoning loop (1.7) is the heart and the riskiest piece. If dispatcher pattern matching turns out to be too brittle on real GLM output, we'll discover it there and either tighten patterns or fast-forward to JSON-grammar.
