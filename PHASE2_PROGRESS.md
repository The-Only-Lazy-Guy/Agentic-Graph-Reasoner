# Phase-2A implementation progress

Status as of 2026-05-20. Plan: `PHASE2_PLAN.md`. Architecture: `REASONING_ARCHITECTURE.md`. Phase-1 reference: `PHASE1_PROGRESS.md`.

Read this file to see where things stand without scrolling chat history.

---

## Sub-phase status (Phase 2A)

| # | Sub-phase | Status | Tests added | Notes |
|---|---|---|---|---|
| 2.1 | ProcedureNode version metadata fields | ✅ done | 3 (in test_schemas.py) | `version`, `parent_version_id`, `superseded_by_id`. Phase-1 backward compat verified: legacy serialized procedures load with defaults. |
| 2.2 | Composition edge constants + validator | ✅ done | 9 (test_composition.py) | `reasoning/composition.py` with CALLS / INHERITS / SPECIALIZES / REPLACES / SUB_INVOCATION_OF + validators. Disjointness checked between procedure-level and session-object relations. |
| 2.3 | Dispatcher.invoke(parent_object_id=) + dedupe key | ✅ done | 6 (in test_dispatcher.py) | `parent_object_id` parameter wires `sub_invocation_of` edges with metadata (procedure_id, name, args_text). `find_existing_sub_invocation` helper for cross-CALL dedupe. Audit log carries `[sub-invocation of <parent>]` prefix on triggered_by_text. |
| 2.4 | Structured `CALL` parser inside invoke() | ✅ done | 9 (in test_dispatcher.py) | `_CALL_RE` matches `CALL <Name> WITH <args>`. New `_dispatch_call_commands` does recursive dispatch with dedupe (acceptance criterion #3) + budget enforcement. `DispatchOutcome.sub_outcomes` records the call tree. |
| 2.5 | Version chain name resolution (one-hop) | ✅ done | 6 (in test_dispatcher.py) | `_build_name_index` excludes deprecated, picks active head; `Dispatcher.resolve_name()` exposes the lookup. End-to-end test confirms top-level invocations route to v2's body when v1 → v2 chain exists. |
| 2.6 | Fan-out budget enforcement test | ✅ done | 1 (in test_dispatcher.py) | Composer emitting 5 CALLs with fan_out cap=3: exactly 3 children complete, graceful stop on `BudgetExhausted` (no crash, surviving sub_outcomes preserved). |
| 2.7 | Three new seed procedures | ✅ done | 9 (test_seed_procedures.py) | `VerifyNonNegativeEdges`, `DetectNegativeCycle`, `VerifyShortestPath` as proper modules under `reasoning/procedures/`. Each has `example_use` so consolidation gate can clear. Composer body uses structured CALL grammar. |
| 2.8 | End-to-end composition test (via reasoning loop) | ✅ done | 1 (in test_reasoning_loop.py) | Full top-down composition runs through `run_reasoning()`: main reasoner invokes composer, composer's body fires two CALL commands, recursive dispatcher creates child session_objects, follow-up turn produces final answer. **Default procedure pool now includes all 4 seeds** so the frontend picks them up automatically. |
| 2.9 | Inspectable structure on non-dispatch runs | ✅ done | 1 (in test_reasoning_loop.py) | `_seed_session_baseline` runs at end of every reasoning episode, populates Q0 + A0 + anchor evidence nodes + support edges. Phase-1's 9/13 empty-session UX is gone. |
| 2.10 | Latency sanity test | ✅ done | 1 (in test_reasoning_loop.py) | Direct-answer conceptual question exits in `iterations_completed == 1`. Locks in the Phase-1 directive tightening. |
| 2.11 | Manual real-LLM smoke | ✅ done | — | Verified end-to-end via session `sess_e1023f5801ca` (2026-05-20). Composer + 2 sub-procedures all fired correctly under real GLM after the composer-body strengthening (see "What 2.11 produced" section below). |

Total tests passing so far: **160/160** (113 Phase-1 + 47 Phase-2A).

---

## What 2.1 + 2.2 produced — behavior, not just pass counts

### Schema migration is backward-compatible

The Phase-1 seed procedure loaded under the new schema reports:

```
name           : VerifyAlgorithmPreconditions
id             : proc_verify_algorithm_preconditions_v1
version        : 1
parent_version : None
superseded_by  : None
```

Defaults are sensible: every procedure built without explicit version args is treated as v1 with no chain links. A backward-compat test (`test_backward_compat_loads_legacy_serialized_form`) verifies that a Phase-1 serialized dict missing all three new keys still loads correctly. This protects the 13 persisted Phase-1 session subgraphs from becoming un-replayable after the migration.

### Version chain walks both directions (one hop)

Authored a v2 of `VerifyAlgorithmPreconditions` and linked it via `parent_version_id` + `superseded_by_id`. Both navigation directions work as expected — one O(1) lookup each. Decay is deliberately NOT activated; the schema fields are forward-compatible scaffolding for Phase 2B.

### Composition relations are typed and disjoint

```
calls               composition? True   procedure-level? True
sub_invocation_of   composition? True   procedure-level? False    (session-object only)
replaces            composition? True   procedure-level? True
support             composition? False  procedure-level? False   (Phase-1 graph relation)
example_of          composition? False  procedure-level? False   (Phase-1 graph relation)
```

The disjointness test (`test_disjointness_of_subsets`) locks in the invariant: each composition relation belongs to exactly one of {procedure-level, session-object}. Future code that walks edges can branch on this cleanly.

---

## Files produced or modified

```
reasoning/
├── composition.py                            ← NEW (2.2)
├── schemas.py                                ← MODIFIED (2.1: 3 new ProcedureNode fields)
└── tests/
    ├── test_composition.py                   ← NEW (2.2)
    └── test_schemas.py                       ← MODIFIED (3 new test methods for version metadata + backward compat)
```

No existing files have been broken or removed. Phase-1 frontend, reasoning loop, dispatcher, etc. continue working unchanged.

---

## Latest test output

```
$ python -m unittest discover -s reasoning/tests -p "test_*.py"
.................................................
----------------------------------------------------------------------
Ran 125 tests in 45.887s

OK
```

---

## What 2.3 – 2.6 produced — behaviour beyond test counts

### A real composer-with-leaves flow runs end-to-end (verified by direct inspection)

Created `VerifyShortestPath` (composer) + `VerifyNonNegativeEdges` + `DetectNegativeCycle` (leaves) as test fixtures and ran a top-level invocation. Output:

```
=== Top-level outcome ===
  procedure : VerifyShortestPath
  object_id : so_20940e8c3b70
  parent_id : None
  mutations : 1
  children  : 2

=== Sub-outcomes (children of the composer) ===
  - VerifyNonNegativeEdges
      object_id   : so_f19f7e50f86c
      parent_id   : so_20940e8c3b70
      final state : {'violating_edges': ['b->c']}
  - DetectNegativeCycle
      object_id   : so_50b8febd24d8
      parent_id   : so_20940e8c3b70
      final state : {'detected_cycles': []}

=== Call tree (sub_invocation_of edges) ===
  so_f19f7e50f86c -> so_20940e8c3b70  (VerifyNonNegativeEdges)
  so_50b8febd24d8 -> so_20940e8c3b70  (DetectNegativeCycle)

=== Budget after the full composition ===
  llm_calls      : 3 / 10
  fan_out_per_step: 2 / 5
  recursion depth: 0 / 4 (clean unwind)
```

This is the canonical Phase-2A pattern working: composer fires two `CALL` commands inside its sub-LLM body; dispatcher parses them, creates independent child session_objects, wires the sub_invocation_of edges, runs each leaf's sub-LLM, applies their mutations, and unwinds the recursion budget cleanly.

### State isolation per criterion #6 confirmed

Each child SessionObjectNode has its own `state` dict. The composer's `state` and the children's `state` never share references. The audit log records mutations per object — replay reconstructs each object's state independently.

### Acceptance criterion #3 (no duplicate sub-invocation for same intent) is locked in

Tests demonstrate:
- A composer that says `CALL X WITH foo` twice in the same response creates ONE child.
- A composer that says `CALL X WITH foo` then `CALL X WITH bar` (distinct args) creates TWO children — this is how the agent legitimately asks for the same procedure on different inputs.
- The dedupe key is `(parent_object_id, child_procedure_id, args_text)`.

### Sharp edge: Dispatcher's name index is snapshot-at-construction

`_build_name_index` runs in `Dispatcher.__init__`. Mutating a procedure's `provenance.deprecated` AFTER construction does NOT update the cached index. Production code (`reasoning_loop.py`) builds a fresh `Dispatcher` per `ReasoningRequest`, so this is fine — but worth knowing for testing and for any future code that thinks of dispatchers as long-lived.

---

## What 2.7 – 2.10 produced

### New seed corpus shipped

`reasoning/procedures/` now contains 4 modules: the Phase-1 seed plus three new ones authored as proper procedures (not just test fixtures). Each has a populated `example_use` so they meet the consolidation worked-instance gate. The composer's `depends_on` declares the three procedures it can call, which the consolidation gate uses to enforce lemma-chain integrity.

The default `procedure_pool` in `run_reasoning()` now includes all 4 — the front-end's substrate path picks them up automatically without any caller changes.

### End-to-end composition through `run_reasoning()` verified

A scripted full flow (main reasoner → composer → 2 CALL children → follow-up answer) produces:
  - 3 session_objects (composer + 2 children)
  - 2 sub_invocation_of edges (call tree)
  - Children's state populated independently
  - 5+ LLM calls consumed within budget
  - No `early_terminated_reason`

This is the canonical Phase-2A test that proves composition works through the orchestrator, not just the dispatcher.

### Empty-session UX from Phase 1 is gone

Every reasoning episode now produces a baseline session structure:
  - `Q0` (question node)
  - `A0` (answer node)
  - `anchor_<id>` (evidence nodes for each retrieved anchor)
  - `support` edges from each evidence node to A0

This holds even when no procedure fires — the conceptual-question path that previously produced empty subgraphs now produces inspectable structure. When procedures DO fire, the session_object nodes coexist with the baseline (they capture procedural state; the baseline captures conversational ground truth).

### Latency check locked in

A direct-answer conceptual question exits in `iterations_completed == 1` with 1 LLM call total. No wasted iterations. The Phase-1 directive tightening that shipped post-hoc is now regression-tested.

---

## What Phase 2A delivered vs what was promised

All 10 acceptance criteria from `PHASE2_PLAN.md §11` are met by automated tests:

| # | Criterion | Met |
|---|---|---|
| 1 | `VerifyShortestPath` end-to-end, invokes 2 sub-procedures via CALL | ✅ test_end_to_end_composition_through_reasoning_loop |
| 2 | Session subgraph: 3+ session_objects, 2+ sub_invocation_of edges | ✅ same test |
| 3 | No duplicate sub-invocation for same intent | ✅ test_duplicate_call_with_same_args_is_deduped |
| 4 | Inspectable session structure on non-dispatch runs | ✅ test_non_dispatch_run_produces_inspectable_session_structure |
| 5 | Latency: direct-answer exits in 1 iter | ✅ test_latency_direct_answer_exits_in_one_iteration |
| 6 | Mutation independence | ✅ test_composer_invokes_two_children + state inspection |
| 7 | Recursion depth budget caps long chain | ✅ test_recursion_depth_budget_caps_chain |
| 8 | Version chain resolution to v2 | ✅ test_top_level_invocation_routes_to_v2 |
| 9 | Fan-out budget enforces gracefully | ✅ test_fan_out_budget_caps_composition |
| 10 | No Phase-1 regression | ✅ all 113 Phase-1 tests still pass |

Decay is intentionally NOT activated (Phase 2B with `--enable-decay` flag, gated on real corpus data).

---

## 2.11 — Manual real-LLM smoke: **PASSED 2026-05-20**

Ran `REASONING_MODE=substrate` against the live opencode/GLM stack on the prompt:

> *"I have a directed graph with edges (a->b, weight 3), (b->c, weight -1), (a->c, weight 5). I'm planning to run Dijkstra... Verify each precondition of Dijkstra against this instance and tell me if it's safe to apply."*

### First attempt (composer fired, but no sub-procedures)

Sub-phase 2.7's initial composer body let GLM bypass the `CALL` grammar by synthesizing sub-procedure results in `sub_results_summary` directly. Composer's session_object was created with verdict text, but `sub_outcome_count=0` and no `sub_invocation_of` edges. Acceptance criteria #1+#2 not met in production.

### Mitigation: strengthen the composer body (Sub-phase 2.7.1)

Rewrote `VerifyShortestPath`'s body to:
- **REQUIRE** at least one `CALL` command ("A response without any CALL line is invalid")
- **FORBID** writing to `state.sub_results_summary` ("Leave it empty. The children populate their own state")
- Explicitly split CALL emission (Step 1+2) from verdict-setting (Step 3) — so the model knows the order

Test `test_composer_body_contains_call_commands` now asserts both the MUST-emit and the DO-NOT-write constraints are present in the body text. Regression-locked.

### Re-run after mitigation (session `sess_e1023f5801ca`)

| Diagnostic | Value | Acceptance |
|---|---|---|
| step_count | 1 | ✓ |
| Total session_objects | 3 (composer + 2 children) | ✓ criterion #1+#2 |
| sub_invocation_of edges | 2 (both pointing at composer) | ✓ criterion #2 |
| Composer `state.sub_results_summary` | `[]` | ✓ strengthened directive worked |
| Composer `state.verdict` | "Dijkstra requires non-negative edge weights; graph contains edge (b->c) with weight -1" | ✓ correct verdict from preconditions alone |
| `VerifyAlgorithmPreconditions` child state | `preconditions_violated: ["non_negative_edge_weights"]` + evidence | ✓ child fired |
| `VerifyNonNegativeEdges` child state | `violating_edges: ["b->c"]`, all 3 edges checked | ✓ child fired |
| Audit log entries | 22 | ✓ |
| Composer's sub-LLM emitted | `CALL VerifyAlgorithmPreconditions WITH ...` + `CALL VerifyNonNegativeEdges WITH ...` + `SET state.*` | ✓ exact structured grammar |

### What we learned

1. **GLM follows structured-grammar instructions when the directive is firm enough.** Soft language ("you may", "if applicable") gets bypassed; hard language ("MUST emit at least ONE", "DO NOT write to X") works.
2. **Both top-level free-text dispatch AND structured CALL dispatch work end-to-end in production.** The Phase-2A architecture's two-grammar split holds up.
3. **Diagnostic capture (`__diag__` node with raw_outputs + dispatch_summary) is load-bearing** for understanding real-LLM behavior. Without it, "0 session_objects" was indistinguishable between "model never said the right thing" and "model said it but the regex didn't match."

Phase 2A is **done in production**. All 10 acceptance criteria from §11 are met by either automated tests or this manual smoke pass.
