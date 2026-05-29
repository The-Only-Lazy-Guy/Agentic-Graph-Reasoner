# Phase-2A implementation plan — composition (decay deferred)

**Status:** plan phase, narrowed scope. Approved direction; decay moved behind a flag and out of the acceptance bar. No code written yet.

**Parent docs:**
- `REASONING_ARCHITECTURE.md` — three-phase architecture
- `PHASE1_PLAN.md` — Phase-1 build (substrate)
- `PHASE1_PROGRESS.md` — actual Phase-1 outcomes

**Date:** 2026-05-20

**What changed from the first draft of this doc (per user review):**
- Sub-procedure invocations use **structured commands** (`CALL X WITH ...`), not free-text regex. Top-level dispatch from the main reasoner stays regex.
- **Decay is deferred / feature-flagged off.** Schema fields land in Phase 2A so we don't break future compatibility, but the decay pass itself is not active until we have real corpus data to justify the 30-session threshold.
- Acceptance criteria tightened: no duplicate sub-invocations for the same intent; non-dispatch runs still produce inspectable session structure; latency sanity check.
- Stale 113 → 111 test-count fact corrected (Phase-1 backend test count is 113 as of today after the two double-dispatch regression tests landed).

---

## 0. Executive summary

Phase 1 produced a substrate where ONE procedure can be invoked, run, mutate state, and persist. The dispatcher saw flat free-text invocations from the main reasoner.

**Phase 2A's load-bearing claim**: procedures can call other procedures **via structured commands**. The substrate moves from a flat list of tools to a hierarchical library where:

- A high-level procedure (e.g. `VerifyShortestPath`) decomposes into smaller calls (`VerifyNonNegativeEdges`, `DetectNegativeCycle`, `VerifyAlgorithmPreconditions`)
- The composer's body emits explicit `CALL X WITH ...` commands; the dispatcher parses them with the same constrained-grammar style as Phase-1 mutation commands
- Each sub-call is itself a SessionObjectNode with its own state and audit log
- The composition tree is recorded as `sub_invocation_of` edges in the session subgraph (and, on consolidation, in the long-term graph)
- Version chain *schema* lands (so v2 procedures can be authored), but automated decay does NOT run by default

This unlocks the "tons of useful tools" property the user asked for in the original 10-point manifesto, and matches `REASONING_ARCHITECTURE.md` §2.4. Decay (§2.6) ships as Phase 2B once we have real corpus data to calibrate the threshold.

What stays out of Phase 2A: macro extraction (frequency-mined call sequences → named macros), meta-procedures (procedures that reason about procedures), dynamic embeddings, multi-session concurrency, *automated decay*, JSON-grammar upgrade for top-level dispatch.

---

## 1. Goal of Phase 2A

A working substrate where:

1. A procedure body can invoke other procedures and receive their results — via **structured commands**, not free-text regex
2. The dispatcher distinguishes *main-reasoner* invocations (regex over reasoner free-text, unchanged from Phase 1) from *sub-procedure* invocations (parsed from the procedure body's structured-command output)
3. Composition edges (`calls`, `inherits`, `specializes`, `sub_invocation_of`) connect procedures both within the session subgraph and after consolidation
4. **Schema-only version chains**: `parent_version_id` / `superseded_by_id` / `version` fields land on `ProcedureNode` so v2 procedures can be authored. The dispatcher resolves a name to the latest non-deprecated version. **No automated decay pass runs by default.**
5. Composition fan-out budget is actually enforced (Phase 1 had the counter; Phase 2A exercises it)
6. At least 3 new seed procedures exist, at least 2 of which compose (one calls the other)
7. An integration test demonstrates real composition: a high-level procedure runs, invokes 2 sub-procedures via structured commands, the session graph shows the call tree
8. Inspectable session structure even when no procedure fires (Q/A/anchor nodes always present)
9. Latency: a direct-answer conceptual question exits the loop in 1 iteration

What's NOT in Phase 2A:
- **Active decay pass.** Schema is forward-compatible (the fields exist), but no procedure gets auto-deprecated. The decay rule moves to Phase 2B and is gated behind `--enable-decay` until we have real corpus data to justify the threshold.
- Macro extraction (trace mining for repeated call sequences) — Phase 3
- Meta-procedures (procedures that operate on procedures) — Phase 3
- Dynamic embeddings — Phase 3
- Multi-session concurrency — out of scope
- JSON-grammar upgrade for **top-level** dispatch — top-level stays regex this phase; only sub-procedure dispatch becomes structured

---

## 2. File structure

Additions only — no Phase-1 files are renamed or removed.

```
graph_final/
├── reasoning/
│   ├── composition.py                    ← NEW: edge schema + call tree builder
│   ├── version_chain.py                  ← NEW: parent_version + replaces + decay
│   ├── dispatcher.py                     ← MODIFIED: distinguish main vs sub-procedure invocations
│   ├── reasoning_loop.py                 ← MODIFIED: composition fan-out enforcement, call tree
│   ├── schemas.py                        ← MODIFIED: new edge types, version metadata fields
│   ├── procedures/
│   │   ├── verify_algorithm_preconditions.py    (Phase 1 — unchanged)
│   │   ├── verify_nonneg_edges.py               ← NEW: leaf procedure
│   │   ├── detect_negative_cycle.py             ← NEW: leaf procedure
│   │   └── verify_shortest_path.py              ← NEW: composes the leaves
│   └── tests/
│       ├── test_composition.py                  ← NEW
│       ├── test_version_chain.py                ← NEW
│       ├── test_dispatcher_composition.py       ← NEW
│       └── test_reasoning_loop_composition.py   ← NEW (end-to-end)
```

---

## 3. Schema additions (`reasoning/schemas.py`)

### 3.1 Procedure version metadata

`ProcedureNode` gains three optional fields:

```python
@dataclass
class ProcedureNode:
    # ... existing fields ...
    version: int = 1                       # 1, 2, 3, ... within a name family
    parent_version_id: Optional[str] = None  # id of the previous version, if refined
    superseded_by_id: Optional[str] = None   # id of the version that replaces this one
```

Rationale: keeps the existing schema backward-compatible (Phase-1 procedures all have `version=1`, no parent, no successor). Forms a forward+backward linked list per procedure name.

### 3.2 New edge relations

The `SessionEdge` / persisted edge format is unchanged structurally — only the set of valid relation strings expands:

| relation | from → to | meaning |
|---|---|---|
| `calls` | procedure → procedure | A invokes B as a subroutine |
| `inherits` | procedure → procedure | A's behavior generalizes B's |
| `specializes` | procedure → procedure | A is a domain-specific variant of B |
| `replaces` | procedure → procedure | A supersedes B (active) |
| `invoked_in_session` | procedure → session_object | a session_object is an instance of this procedure (already implicit in Phase 1 via `procedure_id` field; this is the explicit edge) |
| `sub_invocation_of` | session_object → session_object | child session_object created during parent's execution |

`sub_invocation_of` is **new and load-bearing**: it captures the call tree at the session-object level (the runtime, not the abstract procedure level).

---

## 4. Composition mechanics

### 4.1 Sub-procedure invocation from within a procedure body

A procedure body (sub-prompt) can now contain its own dispatcher-recognizable invocations. When procedure A's body runs and the sub-LLM emits text like *"I'll apply VerifyNonNegativeEdges to the graph"*, the dispatcher fires recursively.

Each sub-invocation:
1. Consumes one `fan_out` budget from the current iteration's quota
2. Pushes the budget tracker's recursion depth via `push_recursion()`
3. Creates a child SessionObjectNode for the sub-procedure
4. Adds a `sub_invocation_of` edge from the child to the parent's session_object
5. Pops recursion depth when done
6. Returns its result (mutation count + summary) as part of the parent procedure's accumulated context

### 4.2 Composition edges in the long-term graph

When a procedure that uses sub-invocations is *consolidated* (citation gate + worked example gate + deps gate), the `calls` edges between abstract procedure nodes are also promoted. The consolidator checks that all `calls` targets are themselves consolidated (gate 3, depends_on).

### 4.3 Composition is bottom-up, not declarative

Phase 2 does NOT require a procedure to declare "I call these other procedures" up front. The composition emerges from invocations made during execution. The `example_use` field captures the canonical call tree for a worked instance.

This matters: it means a procedure can be authored without knowing in advance which sub-procedures will exist; calls emerge as the corpus grows.

---

## 5. Version chains (schema-only; decay deferred)

Decay is **deferred to Phase 2B** and runs only behind a `--enable-decay` flag once activated. Rationale: we have ~13 real sessions on disk; the proposed K=30 threshold has no empirical justification yet. Better to ship the *schema* so v2 procedures can be authored, watch real corpus dynamics for a while, then turn decay on with real data.

### 5.1 Refinement workflow (between-session only)

When a procedure is refined between sessions (a new author commits a new ProcedureNode), it's created with:

- New id (e.g., `proc_verify_shortest_path_v2`)
- `version = 2`
- `parent_version_id = "proc_verify_shortest_path_v1"`
- Same `name` (the human-facing handle stays stable)

The old version gets `superseded_by_id` set to the new version's id, plus an outgoing `replaces` edge from new → old.

Within-session refinement is **not supported** in Phase 2A — would create audit-log churn without a stable consolidation point.

### 5.2 Dispatcher resolution

When the model writes "I'll apply VerifyShortestPath to ...", the dispatcher resolves the name to **the latest non-deprecated version** in the family (in Phase 2A, since nothing is auto-deprecated, this is always the highest `version` number in the chain). Name-to-id mapping is many-to-one: a name lookup returns the head of the version chain.

Resolution follows **one hop only** (resolved decision §12). A procedure with `parent_version_id` set but no `superseded_by_id` pointing forward is the head; we don't walk farther.

### 5.3 Decay (Phase 2B, behind flag)

Phase 2B will introduce a `Decay` pass on consolidation, gated by an `--enable-decay` flag (off by default). When enabled:

- If `parent_version` has zero citations in the last K sessions and the new version has ≥2× more citations than the parent: mark the parent `deprecated=True` with `deprecation_reason="superseded by {newer_id}"`.
- Deprecated procedures are excluded from the dispatcher index.

K stays unset until we have real corpus data. The acceptance bar for shipping the flag is "100+ real sessions of corpus data exist and the threshold is calibrated against actual citation patterns."

---

## 6. Dispatcher changes (`reasoning/dispatcher.py`)

### 6.1 Two grammars, one dispatcher

The dispatcher gets a **second parse path** for sub-procedure invocations. Top-level free-text invocations from the main reasoner continue to use the Phase-1 regex set (apply / invoke / using_the / create_new / apply_intent).

Sub-procedure invocations — emitted from inside another procedure's body — must use the structured command grammar:

```
CALL <ProcedureName> WITH <free-text args>
WAIT                   (optional barrier; ignored in Phase 2A)
```

The `CALL` verb sits alongside the existing mutation verbs (`ADD`, `SET`, `DELETE`, `DONE`) in the procedure-body grammar. Each `CALL` line triggers a sub-invocation in the order it appears.

Why split: under recursion, regex over free-text compounds in failure modes. Inside a procedure body we already control the directive tightly — telling the sub-LLM "emit `CALL X WITH ...` to invoke a child procedure" is reliable in a way "say it however feels right" isn't.

### 6.2 New invocation context

`Dispatcher.invoke()` gains a `parent_object_id: Optional[str] = None` parameter. When non-None:
- The created/reused SessionObjectNode is connected to the parent via `sub_invocation_of` edge
- The audit log's `triggered_by_text` includes a `[sub-invocation of {parent_object_id}]` prefix for traceability
- The sub-procedure's sub-prompt receives **only** the parent's current state + the args extracted from `CALL X WITH ...`, NOT the full graph anchors (resolved decision §12, option ii)

### 6.3 Structured sub-invocation scan

After the sub-LLM call inside `invoke()`, the sub-response is scanned with the structured `_CALL_RE` regex (matching `CALL <name> WITH <args>` lines), NOT the top-level free-text patterns. Each detected sub-invocation:
- Dedupes by procedure name within the same parent's invocation (acceptance criterion: no duplicate sub-invocation for the same intent unless explicitly requested)
- Checks recursion depth + fan_out budget
- If allowed, recurses into `invoke()` with `parent_object_id` set to the current session_object's id
- Aggregates child DispatchOutcomes into a `sub_outcomes: List[DispatchOutcome]` field on the parent DispatchOutcome

### 6.4 Name resolution with version chain

When either grammar extracts a procedure name, the dispatcher looks up the **active** (non-deprecated, latest version) procedure in the family. A small `_resolve_name(name) -> ProcedureNode` helper handles this. One-hop only: head of the chain (no recursive walks through long chains).

### 6.5 Mutation independence

Cross-procedure state isolation is enforced: a sub-procedure can only mutate its OWN SessionObjectNode. Parent state is read-only from the sub-procedure's perspective; the parent consumes the sub-procedure's results via the returned DispatchOutcome's mutation summary, not by reading the child's state directly. This keeps the audit log replayable per object.

---

## 7. Reasoning loop changes (`reasoning/reasoning_loop.py`)

Minimal surgery — most composition logic lives in the dispatcher. The loop changes:

1. **Build procedure index from the active version of each procedure family** (not raw procedure_pool), so the dispatcher always resolves to the head of the chain
2. **Record sub-invocations in dispatch_outcomes** as a flat list with parent pointers, so the trace can be visualized as a tree
3. **Composition fan-out budget per turn** is enforced at the call site (Phase 1 had the counter; Phase 2 increments it when sub-invocations fire)
4. **Per-iteration recursion budget**: the dispatcher's recursion depth is reset between top-level iterations (sub-procedures within one iteration share a budget, but iteration N+1 gets a fresh recursion depth)

---

## 8. New seed procedures

### 8.1 `VerifyNonNegativeEdges` (leaf)

Single-purpose: given an instance description, check whether the described graph has any negative edges. Returns a boolean + list of violating edge descriptions. No sub-invocations.

State schema: `{"violating_edges": list[str], "checked_edges": list[str]}`.

### 8.2 `DetectNegativeCycle` (leaf)

Given an instance description, check whether the graph has any negative cycle. Returns boolean + cycle description (if found). No sub-invocations.

State schema: `{"detected_cycles": list[str], "checked_paths": list[str]}`.

### 8.3 `VerifyShortestPath` (composer)

The compositional one. Body invokes children via structured `CALL` commands:

```
CALL VerifyAlgorithmPreconditions WITH algorithm_name={algorithm} instance_description={instance}
CALL VerifyNonNegativeEdges WITH instance_description={instance}    (if Dijkstra)
CALL DetectNegativeCycle WITH instance_description={instance}        (if BellmanFord)
```

State schema accumulates results from each sub-invocation. The `example_use` field captures the canonical Dijkstra-with-neg-edge trace, showing two sub-invocations and the resulting aggregated state.

These three procedures + the Phase-1 seed make a **family of four**, enough to exercise composition meaningfully.

---

## 9. Test plan

### Unit tests (per sub-phase)
- `test_composition.py` — edge type validation, sub_invocation_of edge creation
- `test_version_chain.py` — name resolution, parent/successor links, decay rule
- `test_dispatcher_composition.py` — recursive invoke, parent_object_id tracking, budget enforcement under recursion
- `test_reasoning_loop_composition.py` — end-to-end with stub LLM, scripted multi-procedure invocation

### Integration tests
- Composition trace: a session that invokes `VerifyShortestPath` produces a session subgraph with the call tree visible via `sub_invocation_of` edges
- Decay: simulated 31-session run where v2 dominates v1; assert v1 is `deprecated=True` after the threshold

### Manual smoke
- Real-LLM end-to-end through `REASONING_MODE=substrate` on the cs4 Dijkstra question. Verify that with the new procedures available, `VerifyShortestPath` is invoked at the top level and sub-procedures fire underneath.

---

## 10. Phased build order within Phase 2

Strict order. Each sub-phase has a clear acceptance test before moving on.

### 2.1 — Schema + version metadata
Add `version`, `parent_version_id`, `superseded_by_id` to `ProcedureNode`. Round-trip JSON test.

### 2.2 — Composition edges
Define edge type constants, validation helpers. No runtime use yet.

### 2.3 — `Dispatcher.invoke()` accepts `parent_object_id`
Single-level sub-invocations: parent invokes child, child runs, returns. No deeper recursion yet. Tests with stub LLM scripts.

### 2.4 — Structured `CALL` parser inside `invoke()`
Add `_CALL_RE` that matches `CALL <name> WITH <args>` lines. After the sub-LLM call, scan the sub-response with this regex (NOT the top-level free-text patterns). Recurse with budget enforcement and dedupe-by-name within the same parent.

### 2.5 — Version chain in dispatcher name resolution
`_resolve_name(name)` returns the active head of the chain. Tests for name → id mapping including deprecation.

### 2.6 — Reasoning loop: composition fan-out enforcement
Sub-invocations consume `fan_out` budget. Test that hitting the cap mid-composition terminates gracefully.

### 2.7 — Three new seed procedures
`VerifyNonNegativeEdges`, `DetectNegativeCycle`, `VerifyShortestPath`. Each with example_use. Round-trip tests.

### 2.8 — End-to-end composition test
Scripted LLM invocations a `VerifyShortestPath` flow. Inspect the resulting session subgraph for the call tree.

### 2.9 — Inspectable structure on non-dispatch runs
For conceptual questions that never fire a procedure, the session subgraph must STILL contain a minimal record: a question node, an answer node, and the anchors retrieved (as evidence nodes). Phase-1 currently leaves the subgraph empty on these runs. Add a `_seed_empty_session(session, question, answer, anchors)` helper that always runs at session close.

### 2.10 — Latency sanity test
Add `test_conceptual_question_exits_in_one_iteration` that runs a scripted "no procedure needed" flow and asserts `iterations_completed == 1` and `len(dispatch_outcomes) == 0`. Locks in the Phase-1 directive tightening that already shipped.

### 2.11 — Manual real-LLM smoke
Run cs4 Dijkstra through `REASONING_MODE=substrate` with the new procedures available. Inspect the persisted session subgraph for sub_invocation_of edges. Compare answer quality vs. Phase-1 (one-shot) baseline on the same question.

(Decay pass + decay tests deferred to Phase 2B, behind `--enable-decay` flag.)

---

## 11. Acceptance criteria for Phase 2A complete

All must hold:

1. `VerifyShortestPath` runs end-to-end on a Dijkstra-shaped question, invokes `VerifyAlgorithmPreconditions` AND `VerifyNonNegativeEdges` as sub-procedures via structured `CALL` commands.
2. The session subgraph contains:
   - 3+ session_object nodes (parent + 2 children)
   - 2+ `sub_invocation_of` edges
3. **No duplicate sub-invocation for the same intent** — if `VerifyShortestPath`'s body emits two `CALL VerifyNonNegativeEdges WITH ...` lines with the same args within one parent-invoke pass, only ONE child session_object is created. The dedupe key is `(parent_object_id, child_procedure_name, args_text)`. An explicit re-run requires distinct args.
4. **Inspectable session structure even when no procedure fires.** A conceptual question that never dispatches still leaves a session subgraph containing a question node, an answer node, and the retrieved anchors as evidence. Empty subgraphs are no longer a valid outcome.
5. **Latency sanity check.** A direct-answer conceptual question exits the reasoning loop in `iterations_completed == 1`. Locked by a unit test that scripts the flow.
6. Each sub-procedure's state is independent of its siblings' and its parent's. The audit log shows the parent never directly mutates a child's `state.*` path.
7. Recursion depth budget (default 4) is respected — a manufactured deeper-than-4 chain triggers `BudgetExhausted` gracefully and the session still persists.
8. A simulated version chain (v1 → v2) resolves correctly: dispatcher invocations of the name route to v2 by default. **Decay is NOT activated** (Phase 2B with flag).
9. Composition fan-out budget per turn is enforced — manufactured >5 sub-calls in one turn triggers `BudgetExhausted`.
10. **No Phase-1 regression** — all 113 Phase-1 backend tests still pass (count as of 2026-05-20 after the double-dispatch regression tests landed; PHASE1_PROGRESS.md updated separately).
11. Phase 2A adds tests for: composition edges, dispatcher CALL grammar, version-chain resolution, sub-invocation dedupe, non-dispatch inspectability, latency. Target: a new test count delta in `reasoning/tests/`, not a fixed number — concrete coverage matters more than count inflation.

---

## 12. Decisions (resolved 2026-05-20)

| Decision | Resolved as | Notes |
|---|---|---|
| **Sub-invocation grammar** | **Structured `CALL X WITH ...` commands inside procedure bodies; top-level dispatch from main reasoner stays regex** | Per user: regex compounds under recursion. Procedure bodies have tight directive control over the sub-LLM so structured grammar is reliable. |
| **Composition surface to model** | (a) sub-procedures listed in main prompt | The main reasoner can see what's available; the composer body decides what to actually CALL. |
| **Sub-procedure context** | (ii) only parent state + sub args | Composition's whole point is focused sub-execution. Saves token budget. |
| **Mutation independence** | (x) each session_object owns its state, no cross-mutation | Parent reads child results via the returned DispatchOutcome's mutation summary, not by reading the child's state directly. Keeps audit log replayable per object. |
| **Version refinement timing** | between-session only | Within-session refinement would create audit-log churn without a stable consolidation point. |
| **Decay** | **DEFERRED to Phase 2B, flag-gated off** | Per user: 30-session threshold has no empirical justification with only ~13 sessions on disk. Ship schema in 2A so v2 procedures can be authored; activate decay later with real corpus data. |
| **`parent_version` resolution** | one hop only | Simpler, sufficient for v1. |

---

## 13. What this is NOT

Same scope guards as Phase 1, plus:

- **Not macro extraction.** We do NOT detect recurring call sequences and synthesize new macros. That's Phase 3.
- **Not meta-procedures.** Procedures don't reason about procedures yet.
- **Not multi-version simultaneous use.** A name resolves to one version at any time.
- **Not cross-graph composition.** Procedures retrieved for graph A don't compose with procedures retrieved for graph B in the same session.
- **Not a procedure marketplace.** Procedure provenance stays within one repo / one team.

---

## 14. Effort estimate (Phase 2A)

| Sub-phase | Lines | Time |
|---|---|---|
| 2.1 Schema + version metadata | ~100 | 0.5 day |
| 2.2 Composition edges (definitions + validation) | ~80 | 0.25 day |
| 2.3 `Dispatcher.invoke(parent_object_id=)` + dedupe key | ~140 | 0.5 day |
| 2.4 Structured `CALL` parser inside `invoke()` | ~150 | 0.75 day |
| 2.5 Version chain + name resolution (one-hop, no decay) | ~120 | 0.5 day |
| 2.6 Reasoning loop fan-out enforcement | ~80 | 0.25 day |
| 2.7 Three new seed procedures | ~280 | 0.75 day |
| 2.8 End-to-end composition test | ~150 | 0.5 day |
| 2.9 Inspectable structure on non-dispatch runs | ~80 | 0.25 day |
| 2.10 Latency sanity test | ~40 | 0.1 day |
| 2.11 Manual real-LLM smoke | n/a | 0.25 day |

**Total: ~1220 lines, ~4.6 working days for Phase 2A.**

Phase 2B (decay activation) is its own follow-up: ~180 lines, ~0.5 day plus the soak time needed to gather 100+ real sessions before turning the flag on.

---

## 15. What we'll know after Phase 2

By the time these acceptance criteria are met, we'll have answered:

- Does recursive dispatcher invocation work reliably under realistic LLM output, or do we hit a wall at depth 2?
- Does the composition fan-out budget catch bad behavior, or does the model rarely fan-out enough to need it?
- Are procedures actually composable with text-only sub-prompts, or does sub-procedure invocation need JSON grammar to be reliable?
- Does the decay rule (60+ citations on new version, 0 on old over 30 sessions) match real corpus dynamics, or do procedures cycle faster/slower than expected?
- Is `sub_invocation_of` enough to render call trees, or does the UI need more structural metadata?

These feed Phase-3 decisions on macro extraction and meta-procedures.

---

## 16. Status / next move

This doc is the **design phase output**. No code has been written for Phase 2 yet.

**Next step:** user resolves the OPEN decisions in §12. Once those are pinned, we start Sub-phase 2.1 (schema + version metadata). Same discipline as Phase 1: design first, code second, test inspection per sub-phase.
