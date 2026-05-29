# Reasoning Architecture â€” design doc

**Status:** live architecture doc. See §13 for the actual phase status as of
2026-05-24.

**Author of original 10-point manifesto:** user.
**Synthesis + additions:** this doc.

**Date:** 2026-05-19 (original); status table last refreshed 2026-05-24.

---

## 0. Executive summary

Today's Answerer-v2 front-end does **one anchor retrieval, one LLM call, no verification.** The reasoning visible in `<reasoning>` blocks is a single forward pass â€” structured-looking but shallow. The graph is used as *context* for that one call, never as a *medium for computation*.

This design reframes the graph as an **active computational substrate**, not passive storage. Reasoning becomes a loop that creates structured objects, mutates their state, composes them into larger procedures, accumulates failures alongside successes, and consolidates working memory into long-term memory. The transformer remains the fluid-inference engine; the graph supplies persistence, structure, state, and compositional reuse.

Three abstractions are load-bearing:

1. **Stateful objects** â€” graph nodes that carry mutable state across reasoning steps within a session
2. **Session subgraphs** â€” per-query scratch subgraphs that may consolidate into long-term memory
3. **Graph-as-computational-medium** â€” the reframe that turns the other items into consequences rather than features
4. **Context-aware activation** - graph nodes can emit typed signals into the current task frame even when no procedure is explicitly called

The rest of this doc enumerates the full architecture in implementation phases. Phase 1 is the load-bearing substrate; Phase 2 adds composition and evolution; Phase 3 adds meta-cognition, chunking, and context-aware graph activation.

---

## 1. Vision

> Memory is not a database. Memory is a computational substrate.

The graph:
- **stores procedures** as first-class nodes, not just facts
- **routes attention** by determining what gets retrieved next
- **evolves abstractions** by darwinian decay of unused procedures
- **caches reasoning** by reusing successful procedures across sessions
- **preserves state** by externalizing what transformers can't reliably hold
- **encodes failures** alongside successes
- **composes tools** through inheritance, specialization, and call edges
- **shapes future cognition** by influencing which retrievals appear

The transformer/graph split:

| Transformer is good at | Graph is good at |
|---|---|
| Local fluid inference | Persistent state |
| Pattern abstraction | Structural composition |
| Fuzzy synthesis | Explicit relationships |
| Generation | Compositional reuse |
| Generalization within a context window | Cross-session memory |

The hybrid is strictly stronger than either alone. This doc does NOT propose replacing the transformer with symbolic logic â€” it proposes the *seam* between them.

---

## 2. Core abstractions

### 2.1 Stateful objects (procedure-with-state)

A new node_type: `procedure`. Procedures are reusable reasoning patterns AND live cognitive workspaces â€” they carry mutable state across reasoning steps within a session.

**Why this matters:** transformers are bad at stable mutable state. Every long-context-recall paper is fighting this. By externalizing state to a graph-tracked structure, we sidestep the problem entirely. The model reads the current state, proposes mutations, and the system applies them deterministically.

**Retention decision (2026-05-22).** Procedure / session-object nodes stay in
the architecture. They are not the default hot path for direct-answer tasks
anymore, but they remain the right primitive for **systemic design tasks**:
multi-component architecture design, interface negotiation, invariant tracking,
subsystem decomposition, and long-lived design workspaces. `SignalNode` is the
general reasoning substrate; `procedure` and `session_object` remain the
specialized object/workspace lane.

Example:

```yaml
procedure:
  id: <auto>
  name: GraphVerificationContext
  purpose: "Track ongoing verification of properties of a graph being analyzed"
  state_schema:
    visited_nodes: list[node_id]
    detected_cycles: list[cycle]
    violated_constraints: list[constraint]
    assumptions: list[assumption]
    unresolved_questions: list[str]
  signature:
    inputs:
      - graph: GraphDescription
    outputs:
      - verification_report: VerificationReport
  body: |
    Process the input graph one node/edge at a time.
    For each node visited, add to visited_nodes.
    For each cycle detected, add to detected_cycles.
    For each violated constraint encountered, add to violated_constraints.
    If a property cannot be checked without additional assumption, add to assumptions.
    Continue until graph fully traversed or budget exhausted.
  example_use:
    session_id: <ref>
    inputs: {graph: <graph_id>}
    final_state: {...}
    final_output: {...}
  created_in_session: <session_id>
  citation_count: 0
  parent_version: null
```

State mutations across reasoning steps happen via **full CRUD with field-level diffs** (resolved 2026-05-20 â€” see Â§11). Every mutation is journaled in the session audit log (Â§11.1) so the state evolution is replayable and debuggable.

### 2.2 Session subgraphs

Each query gets a **temporary cognitive graph**. Objects/nodes form dynamically during reasoning, accumulate state, may reference each other. At session end:

- Long-term graph remains unchanged unless consolidation criteria are met
- Successful structures (cited, validated, dependencies-clean) propose themselves for consolidation
- The session graph persists in cold storage for replay / debugging / training data

This is the working-memory â†’ long-term-memory consolidation pattern from cognitive science (Atkinson-Shiffrin, etc.). It is **safer** than mutating long-term memory directly â€” bad reasoning in one session can't corrupt the persistent store.

**This already partially exists in answerer_v2** as session graphs with Q/A/evidence nodes. The new ingredient is: objects/procedures/failures created during a session live in the session subgraph until consolidation, not in the persistent graph immediately.

### 2.3 Failure patterns

Critical and absent from most agent systems. A new node_type: `failure_pattern`.

Examples:
- "Greedy edge selection fails under negative weights"
- "Substring-match probe scoring misses numeric answers with units"
- "Single-anchor retrieval misses cross-domain reasoning"

Why important: humans learn heavily from failed abstractions. Without failure storage, the system relives every mistake. Anti-patterns are as valuable as patterns.

Failure patterns are NOT misconceptions (which are claim-level, e.g., "Dijkstra works without negative cycles" is a `_false` claim). Failures are **procedure-level**: "attempting approach X on a problem with property Y fails because Z."

```yaml
failure_pattern:
  id: <auto>
  name: GreedySelectionFailsOnNegativeEdges
  attempted_approach: "Greedy edge selection ordered by weight ascending"
  failure_condition: "Graph contains at least one edge with negative weight"
  failure_mechanism: "Greedy commits to a path before a later negative edge can offer a shorter alternative"
  replacement: <ref to procedure or fact>  # e.g., Bellman-Ford
  example_failure_case:
    session_id: <ref>
    inputs: {...}
    observed_failure: "..."
  created_in_session: <session_id>
  citation_count: 0
```

Retrieval treats failure patterns like any other node â€” they surface in context when relevant, warning the model away from anti-patterns.

### 2.4 Composition & evolution

Procedures call other procedures via explicit edges:

```
procedure: VerifyShortestPath
â”œâ”€â”€ calls VerifyNonNegativeEdges
â”œâ”€â”€ calls DetectNegativeCycle
â””â”€â”€ calls ValidateRelaxationInvariant
```

Edge types between procedures:
- `calls` â€” A invokes B as a subroutine
- `inherits` â€” A's behavior generalizes B's
- `specializes` â€” A is a domain-specific variant of B
- `replaces` â€” A supersedes B (with parent_version chain for darwinian decay)

Versioning: every procedure has a `parent_version` edge. When refined, a new version is created; the old one decays unless still cited.

Granularity heuristic: **let the agent's worked examples implicitly choose granularity.** When a procedure is created, the example_use trace pins what counts as "one step" of that procedure. Compositions emerge bottom-up: if the same sequence of procedure calls recurs across sessions, it becomes a candidate for a macro-procedure (see Phase 3).

### 2.5 Multi-scale memory

Procedures eventually compress into higher-abstraction macros:

| Scale | Example |
|---|---|
| micro | edge relaxation |
| meso | shortest-path verification |
| macro | graph optimization strategy |

This is **chunking** â€” how experts work. They retrieve compressed structures rather than reasoning step-by-step. The system grows this organically through **macro extraction** (Phase 3): detect that `{A, B, C}` is a recurring call sequence, validate that the macro `M â‰¡ Aâˆ˜Bâˆ˜C` is behaviorally equivalent on the worked examples, name M, install M as a new procedure.

Macro extraction is the **hardest unsolved problem** in this design. Equivalence checking is undecidable in general. We will use heuristic detection (frequency thresholds, IO-signature matching, manual validation by an extraction LLM-pass). Phase 3 only.

### 2.6 Meta-procedures

Procedures that reason about procedures. The unlock IS executive function.

Examples of useful meta-procedures:
- `when_to_decompose(question)` â€” heuristic: question has â‰¥3 conjunctive parts, or no single anchor exceeds similarity Î¸
- `when_to_backtrack(state)` â€” heuristic: hypothesis confidence dropped below Î¸ after N steps
- `when_to_compress(memory)` â€” heuristic: working memory exceeds K items
- `when_retrieval_confidence_is_low` â€” when the top-k anchors have mean similarity below Î¸
- `when_to_invoke_verification` â€” gates Phase-1 hypothesis verification

Meta-procedures can recursively invoke themselves. They MUST be guarded by:
- Hard recursion depth caps
- Termination conditions
- The budget system (Â§4 below)

Phase 3 only. Do not bake meta-procedures into v1 â€” you won't know which ones matter until you watch the system reason without them.

### 2.7 Context-aware graph activation

Phase 3B trace work showed a useful boundary: strong models often solve hard direct-answer tasks without invoking any procedure. That is not a failure of macro extraction; it means the graph also needs a non-procedural activation layer.

The intended model:

1. Build a `SessionContext` from the user question, retrieved anchors, graph id, domain, constraints, and observed gaps.
2. Activate nearby graph nodes as typed objects, not just text snippets.
3. Let activated nodes emit `GraphSignal` objects: constraints, pitfalls, relevant procedures, examples, missing-context requests, and bridge hypotheses.
4. If required context is absent, create a provisional adjacent node in the session subgraph, such as `session_gap` or `session_bridge`.
5. Render a compact `GraphTaskFrame` into the prompt before the final reasoning directive.
6. After the answer, run a coverage check that records which frame items were addressed or missed.

This is the user's "nodes as programmatic objects" idea in a constrained form. Nodes do not carry arbitrary executable code. Instead, a small behavior registry defines how each node type can react to a `SessionContext`. The graph behaves like a signal network while remaining inspectable and replayable.

Example:

```yaml
SessionContext:
  graph_id: merged_graph
  domain: computer_science
  task_kind: algorithm_design
  constraints:
    - n,q <= 200000
    - negative values allowed
    - non-empty subarray
    - C++17
  anchors:
    - segment_tree
    - max_subarray

GraphSignal:
  source_node_id: example_max_subarray_segment_tree
  kind: constraint
  payload: "Use sum, max_prefix, max_suffix, and max_sub per segment."
  confidence: 0.91

GraphTaskFrame:
  constraints:
    - non-empty subarray, so all-negative arrays return the maximum element
    - use long long
  pitfalls:
    - do not clamp leaf values to 0; that permits empty subarrays
  suggested_structures:
    - segment tree node with sum/pref/suff/best
```

The provisional nodes are session-scoped by default. Promotion to long-term memory remains a later consolidation problem; Phase 3C only proves that context-aware activation helps the current answer.

### 2.8 Adaptive plan-tree reasoning

Phase 3C gives the model a better task frame, but a weaker/local model can
still choose the wrong approach or drift into the wrong control mode. Phase 3D
therefore treats planning as a **checkpoint tree** rather than a linear plan.

Each plan step is a session-scoped node. If a check fails, the system
backtracks to the best useful checkpoint, marks the failed branch, and creates a
sibling plan. The plan is provisional: wrong branches are evidence, not crashes.

Example:

```text
root: solve dynamic max-subarray
└─ choose_algorithm
   ├─ kadane_direct
   │  └─ failed: point updates make O(n) per query too slow
   └─ segment_tree
      ├─ define_node_state
      ├─ derive_merge_rule
      ├─ verify_all_negative
      └─ final_answer
```

Phase 3D uses the `GraphTaskFrame` and coverage checks from Phase 3C as its
objective function. Procedure calls become one branch type, not the default
mode. This lets the system expose procedure affordances only when the active
plan branch actually needs them.

### 2.9 Reasoning substrate v2

Phase 3E replaces the explicit focus/plan/execute/check/revise/finalize prompt
pipeline with a smaller recursive substrate:

1. Project existing session nodes into uniform `SignalNode` views:
   constraint, decision, hypothesis, evidence, gap, unresolved_gap, risk,
   repair, and procedure.
2. Compile a compact `StepContextPacket` before each call. The model does not
   pick retrievals or dispatch procedures on the hot path.
3. Ask for one `STEP_RESULT` block per reasoning step. The block returns a
   short result, optional strict `missing` object, and a `StateDelta`.
4. Run deterministic checker plugins. Hard checker failures stop the step;
   strict `need_info` opens a child step; malformed deltas degrade into
   low-confidence `DeltaTransaction(status="skimmed")` traces.

This does **not** delete the older procedure/object design. 3E changes what is
load-bearing for fast direct-answer reasoning, but procedure/session-object
nodes are still retained for systemic design tasks where the model benefits
from an explicit persistent workspace object rather than only a short-lived
signal packet.

3E is still prompt-level coupling, not hidden-state coupling. Its job is to
produce reliable `(packet, delta, checker, outcome)` traces for 3F's local
runtime work: KV warm-pool first, learned NeedProbe/adapter later.

---

## 3. Node and edge schemas

### 3.1 Node types (full list)

| node_type | Phase | Purpose |
|---|---|---|
| `fact` | existing | atomic factual content |
| `claim` | existing | assertion (may or may not be true) |
| `example` | existing | worked instance |
| `summary` | existing | aggregated explanation |
| `hub` | existing | high-degree integration node |
| `bridge` | existing | cross-domain connector |
| `hypothesis` | existing | candidate idea pending promotion |
| `application` | existing | usage pattern |
| **`procedure`** | **1** | stateful reasoning template |
| **`failure_pattern`** | **1** | anti-pattern with failure mechanism |
| **`session_object`** | **1** | session-scoped instance of a procedure with current state |
| `macro_procedure` | 3 | compressed composition of procedure calls |
| `meta_procedure` | 3 | procedure that operates on other procedures |
| `activation_signal` | 3C | session-scoped typed signal emitted by an activated graph node |
| `task_frame_item` | 3C | session-scoped prompt item derived from activation signals |
| `session_gap` | 3C | provisional missing-context node created when required context is absent |
| `session_bridge` | 3C | provisional adjacent connector created to relate current context to nearby graph structure |
| `plan_node` | 3D | session-scoped checkpoint in an adaptive reasoning plan tree |
| `plan_check` | 3D | session-scoped result of checking one plan node against task constraints |
| `substrate_v2_step` | 3E | persisted occurrence of one recursive reasoning step |
| `substrate_v2_delta` | 3E | persisted `StateDelta` plus `DeltaTransaction` parse status |
| `substrate_v2_check` | 3E | deterministic checker result for a v2 step |
| `substrate_v2_signal` | 3E | persisted signal emitted or projected during a v2 run |

### 3.2 Edge types (new ones)

| edge | from â†’ to | Phase | Meaning |
|---|---|---|---|
| `calls` | procedure â†’ procedure | 2 | A invokes B as subroutine |
| `inherits` | procedure â†’ procedure | 2 | A generalizes B's behavior |
| `specializes` | procedure â†’ procedure | 2 | A is a domain-specific variant of B |
| `replaces` | procedure â†’ procedure | 2 | A supersedes B (parent_version chain) |
| `applied_in` | procedure â†’ session_object | 1 | session instance of procedure |
| `failure_of` | failure_pattern â†’ procedure | 1 | anti-pattern targets this procedure |
| `replacement_for` | procedure â†’ failure_pattern | 1 | this procedure is the recommended alternative |
| `validates` | example â†’ procedure | 1 | this worked example validates that procedure |
| `depends_on` | procedure â†’ fact/procedure | 1 | A's correctness depends on B |
| `compresses` | macro_procedure â†’ procedure | 3 | macro represents a sequence of procedure calls |
| `meta_targets` | meta_procedure â†’ procedure_class | 3 | meta-procedure operates on procedures matching this pattern |
| `emits_signal` | any node â†’ activation_signal | 3C | activated node produced a typed signal in this session |
| `frame_includes` | task_frame_item â†’ activation_signal | 3C | prompt frame item was derived from this signal |
| `fills_gap` | session_bridge â†’ session_gap | 3C | provisional bridge attempts to satisfy a detected missing-context node |
| `derived_from` | session_gap/session_bridge â†’ source node | 3C | provisional node was synthesized from these retrieved or activated sources |
| `plan_child` | plan_node â†’ plan_node | 3D | child checkpoint in the plan tree |
| `plan_revision_of` | plan_node â†’ plan_node | 3D | sibling branch created after a failed plan |
| `backtracked_to` | plan_check â†’ plan_node | 3D | failed check selected this checkpoint for retry |
| `checked_by` | plan_node â†’ plan_check | 3D | this check evaluated the plan node |

### 3.3 Universal metadata (every node carries)

```yaml
provenance:
  created_in_session_id: <session>
  validating_examples: [<session_id>, ...]  # at least one for promotion
  depends_on: [<node_id>, ...]
  citation_count: int
  citation_decay: float  # half-life
last_modified: timestamp
deprecated: bool
deprecation_reason: str | null
```

**Provenance is critical for debugging months later.** When a wrong answer surfaces, you trace the procedure chain that produced it. Without provenance, this becomes archaeology.

---

## 4. Budget enforcement

**Bounded cognition makes systems more intelligent, not less.** Unconstrained optimization is dumb. Every reasoning episode has hard caps:

| Budget | Default | Hard cap |
|---|---|---|
| Token budget | 2048 | 4096 |
| LLM call budget | 3 | 8 |
| Graph hop budget | 1 | 3 |
| Procedure recursion depth | 4 | 8 |
| Session subgraph size | 50 nodes | 200 nodes |
| Composition fan-out | 5 calls | 12 calls |

Budgets are **enforced at dispatch time**, not as soft suggestions. When a budget is exhausted, the reasoner is given a `<budget_exhausted>` signal in context and must produce a final answer with what it has.

**Bake budgets in from v1.** Retrofitting them after the system spirals is expensive.

---

## 5. Control flow protocol â€” **RESOLVED: free-text + pattern-match for Phase 1**

The biggest decision this doc had to make: **how does the agent invoke procedures and mutate state?** Three options were considered:

### Option (1) â€” Structured LLM commands (JSON / function-call grammar)

```
The agent emits, inside its <reasoning>:
  <action>{"call": "VerifyNonNegativeEdges", "args": {"graph": "$g"}}</action>
System parses, dispatches, returns result via:
  <result>{"output": false, "violating_edge": ...}</result>
```

**Pros:** Reliable dispatch. Exact argument binding. Easy to log and replay.
**Cons:** Model must be reliable at JSON syntax. GLM-Flash is mediocre at this. Probably needs fine-tuning to be solid. May break under sampling temperature variation.

### Option (2) â€” Free-text reference + pattern matching

```
The agent writes naturally:
  "Now I'll apply the non-negative edge check to the user's graph..."
System regex-matches "(apply|use|invoke) X to Y" patterns and dispatches.
```

**Pros:** No syntax requirement on model. Works with any LLM, including ones not fine-tuned for tool use. Graceful degradation: if dispatch misses, model still reasons sensibly.
**Cons:** Fragile on edge cases ("apply Verify..." vs "we should verify..." vs "verification gives us..."). Cannot reliably extract complex argument bindings.

### Option (3) â€” Executive LLM

```
Reasoner LLM produces reasoning text.
Executive LLM watches output, decides what to dispatch.
Result returned to reasoner for next step.
```

**Pros:** Reliable. Decouples reasoning model from dispatch model.
**Cons:** 2Ã— LLM cost per turn. Adds latency. Requires two prompt designs to maintain.

**Resolution: (2) for Phase 1.** Cheapest to prototype, fails gracefully, lets us learn what we actually need before committing to a syntax. Upgrade path to (1) JSON-grammar planned for Phase 2 once we know which procedures get called often enough to justify reliable dispatch.

**Phase 1 dispatcher contract:**
- Regex patterns matched against the reasoner's `<reasoning>` and `<answer>` text
- Match patterns: `apply\s+(\w+)\s+to\s+(...)`, `invoke\s+(\w+)`, `using\s+the\s+(\w+)\s+procedure`, `create\s+a\s+new\s+(\w+)\s+object`
- On match: dispatch via name-lookup against the session subgraph's procedure index â†’ run sub-prompt â†’ splice result back into context for the next reasoning step
- On miss: silent. The model's reasoning continues but the procedure didn't actually fire. This is the graceful-degradation property â€” wrong answer beats crashed loop.
- Audit log records every match attempt (matched + unmatched) for debugging dispatch coverage

---

## 6. Transformer/graph seam

Where in the reasoning loop does the model "consult the graph" vs continue thinking internally?

Two models:

**Fixed consult points** (Phase 1):
1. **Anchor retrieval** (existing, kept) â€” initial context
2. **Hypothesis verification** (new) â€” after reasoning emits hypotheses, system retrieves contradicting evidence and re-prompts
3. **Post-reasoning check** (new) â€” after answer is generated, optional pass that surfaces any contradictions for the answer

**Learned consult policy** (Phase 2+):
The agent decides when to query the graph mid-reasoning. Requires the control-flow decision (Â§5) to be solid first.

**Phase 1 commits to fixed consult points.** This makes the system predictable and debuggable. The "agent decides when to query" pattern is more flexible but harder to make reliable â€” defer.

---

## 7. Consolidation: session â†’ long-term

End-of-session flow:

```
1. Parse session subgraph
2. For each created node (procedure, failure_pattern, session_object):
   - Has it been cited >= M_PROMOTION times across distinct sessions? (use hypothesis_pool's existing machinery)
   - Does it have at least one validated example_use? (the quality gate)
   - Are its dependencies all consolidated? (lemma-chain integrity)
   If all three: promote to long-term graph.
   Else: keep in cold storage (session graph persisted) until citation threshold met or expiration.
3. For procedures with parent_version edges, check if newer version has dominated old:
   - If newer has > 2Ã— citations and old has 0 citations in last K sessions: deprecate old
4. Update citation_count and citation_decay on all referenced long-term nodes.
```

**M_PROMOTION** (citation threshold): start at 3 distinct sessions. Tunable.
**K (deprecation window):** 30 sessions. Tunable.

---

## 8. Phased implementation plan

### Phase 1 â€” substrate (start here)

**Goal:** A system that can create stateful objects, mutate them within a session, reuse procedures within and across sessions, store failures alongside successes. No composition. No meta. No evolution. No chunking.

**Components:**
- New node_types: `procedure`, `failure_pattern`, `session_object`
- Session subgraph data structure (extension of existing answerer_v2 session graphs)
- State mutation operations: `append`, `replace`, `merge`
- Budget enforcement at dispatch (token/call/hop/recursion/subgraph-size/fan-out)
- Provenance fields on every node
- Control flow: Option (2) pattern-matching dispatcher
- Fixed consult points: anchor retrieval + hypothesis verification + post-reasoning check
- Consolidation pipeline (citation threshold + validated example_use gate)

**Output:** working v1 with reusable stateful procedures and failure memory. NO composition yet.

**Estimated effort:** ~1500 lines of Python, ~3-5 design iterations on the dispatcher.

### Phase 2 â€” composition and evolution

**Goal:** Procedures call other procedures, refine each other, decay.

**Components:**
- New edges: `calls`, `inherits`, `specializes`, `replaces`
- Version chains with darwinian decay
- Composition fan-out budget enforcement (already in Phase 1's budget list â€” just gets exercised)
- Possibly upgrade to control flow Option (1) JSON dispatch for reliable argument binding

**Output:** procedures that build on each other and refine over time.

**Estimated effort:** ~800 lines on top of Phase 1.

### Phase 3 â€” meta, chunking, and activation

**Goal:** System gets faster and deeper over time without manual curation.

**Components:**
- Phase 3A: meta-procedures with hard recursion caps
- Phase 3B: macro extraction via trace mining (frequency-based candidate detection + offline validation)
- Phase 3C: context-aware graph activation and provisional adjacent-node synthesis
- Phase 3D: adaptive plan-tree reasoning with checkpoint backtracking
- Phase 3E: signal graph + recursive `STEP_RESULT` substrate behind `enable_substrate_v2`
- `macro_procedure`, `meta_procedure`, `activation_signal`, `task_frame_item`, `session_gap`, `session_bridge`, `plan_node`, and `plan_check` node types
- Dynamic embeddings via LLM-rerank remain a later retrieval improvement, not the core of Phase 3C

**Output:** the system uses cached chunks when procedures recur, and uses active graph signals when the best answer path is direct rather than procedural.

**Estimated effort:** ~2000 lines + nontrivial design for macro extraction. Treat as research.

---

## 9. What this is NOT

To prevent scope creep:

- **Not a replacement for the transformer.** The transformer is the fluid-inference engine; this gives it persistent structured memory.
- **Not symbolic logic.** Procedures are sub-prompts with state, not formal proofs. We're not building Lean.
- **Not a database.** The substrate stores procedures and structure, not arbitrary facts.
- **Not an attempt to internalize the directive in weights.** That was the SFT goal; this is orthogonal. The architecture works regardless of model â€” Qwen3, GLM, Claude, all consume the same prompts.
- **Not a multi-agent system.** One reasoner, one graph, one session at a time. Concurrency is out of scope until far later.

---

## 10. Things deferred (research-grade or premature)

| Topic | Why deferred |
|---|---|
| Truly dynamic embeddings | O(procedures Ã— queries) at retrieval. Use LLM-rerank as cheap approximation. |
| Macro extraction | Equivalence checking undecidable in general. Phase 3 with heuristics. |
| Differentiable graph operations | Symbolic-neural hybrid is a research program. Out of scope for v1-3. |
| Arbitrary executable code on nodes | Too hard to inspect, sandbox, validate, and replay. Phase 3C uses a typed behavior registry instead. |
| Long-term persistence of activation-synthesized nodes | `session_gap` and `session_bridge` nodes stay session-scoped until consolidation criteria are designed and tested. |
| Multi-session concurrency / merge | Adds CRDT-like complexity. One reasoner at a time for now. |
| Cross-graph reasoning (multiple memory graphs at once) | Defer until single-graph reasoning is solid. |
| Procedure marketplaces / sharing across users | Far future. |

---

## 11. Decisions (resolved 2026-05-20)

| Decision | Resolved as | Notes |
|---|---|---|
| **Control flow protocol (Â§5)** | **(2) Free-text + pattern match** | Phase 1 only. Upgrade to JSON-grammar (option 1) in Phase 2 once we know which procedures get called often enough to justify reliable dispatch. |
| **State mutation grammar** | **Full CRUD with field-level diffs** | Departure from initial recommendation (append+replace). Most expressive, highest risk of state corruption. **Forces us to design the audit-log + diff representation from day one** â€” see Â§11.1 below. |
| **Session subgraph persistence** | **Always persist** | One JSON file per session in cold storage. Cheap. Enables debugging, replay, training-data extraction. |
| **Failure-pattern retrieval weight** | **Boosted** | 1.3â€“1.5Ã— similarity multiplier. Tune later based on observed precision/recall on warnings. |
| **Promotion threshold M** | 3 sessions (default) | Tunable knob |
| **Deprecation window K** | 30 sessions (default) | Tunable knob |
| **Default LLM-call budget per query** | 3 (default) | Tunable knob |

### 11.1 Audit log requirement (consequence of full-CRUD choice)

Because state mutations can include arbitrary updates and deletes, every mutation must be journaled. The audit log is **mandatory infrastructure for Phase 1**, not a v2 nice-to-have.

Each mutation entry:
```yaml
mutation_entry:
  session_id: <ref>
  step_index: int
  object_id: <ref>
  operation: "create" | "read" | "update" | "delete"
  field_path: "state.visited_nodes"           # dotted path
  old_value: <any>                            # null for create
  new_value: <any>                            # null for delete/read
  triggered_by_text: "..."                    # the reasoning snippet that produced this mutation
  timestamp: ISO8601
```

The audit log is per-session, persisted alongside the session subgraph. Enables:
- Replay of any session's state evolution step by step
- Rollback if a mutation introduces inconsistency
- Debugging: "what was `assumptions` field at step 4 of session S?"
- Training-data quality assessment: which mutations led to correct vs incorrect answers

---

## 12. Prior art (for orientation, not for direct port)

- **Voyager** (Wang et al. 2023) â€” Minecraft agent with skill library + skill creation. Closest precedent for objects-as-callable-procedures with persistence.
- **MemGPT / Letta** â€” externalized working memory via chat-history paging. Closest precedent for state externalization, but state is unstructured text.
- **Soar, ACT-R** â€” classical cognitive architectures. Procedural + declarative memory split. Closest precedent for the long-term/working-memory hierarchy.
- **HTN planning** â€” hierarchical task networks. Closest precedent for composition (Â§2.4).
- **Lean / Coq tactic libraries** â€” lemma libraries that grow over time with darwinian pressure. Closest precedent for procedure evolution (Â§2.4 + Â§3).
- **Computational rationality (Lieder & Griffiths)** â€” bounded resources make agents more intelligent. Closest precedent for budget design (Â§4).
- **The hypothesis_pool already in this project** â€” same pattern (citation-count based promotion) extended to procedures and failures.

---

## 13. Status / next move

This doc is the live architectural reference. Per-phase detail lives in the
`PHASE*_PROGRESS.md` files. The table below is the authoritative status as of
2026-05-24 — the previous wording in §0 ("partially implemented") was stale
and undercounted what landed.

### Phase status

| Phase | Status | Notes |
|---|---|---|
| 1 — substrate | **done** | All 9 sub-phases shipped; 113 tests; real-LLM smoke battery passed. See `PHASE1_PROGRESS.md`. |
| 2A — composition | **done** | 11 sub-phases, all 10 acceptance criteria met; real-LLM composer smoke passed. See `PHASE2_PROGRESS.md`. Cumulative tests: 160. |
| 2B — darwinian decay | not started | Schema fields are forward-compat scaffolding from 2.1; activation gated on real corpus data. |
| 3A — meta-procedures | **done** | 5 conservative MPs + default-pool wiring; hardening passes 3A.1/3A.2/3A.3 included. Cumulative tests: 260. See `PHASE3A_PROGRESS.md`. |
| 3B-1 — trace mining | **done (code) / gated (corpus)** | Logger + macro extractor + scan/status/grade CLI shipped; 1 live macro candidate from 4 graded sessions. Corpus gate is at 7/10 real graded traces. 3B-2 (proposal/validation/installation/promotion/rollback) is NOT started and remains gated on the corpus threshold. See `PHASE3B_PROGRESS.md`. |
| 3C — graph activation | **done** | `SessionContext`/`GraphTaskFrame`/coverage check landed; real merged_graph + Qwen local smokes passed. Cumulative tests: 311. See `PHASE3C_PROGRESS.md`. |
| 3D-1 — plan-tree (deterministic) | **done** | Plan primitives + backtracking policy + synthetic IOI/Dijkstra drivers. Cumulative tests: 312. See `PHASE3D_PROGRESS.md`. |
| 3D-2 — plan-tree (prompt mode) | not started | Gated behind `enable_adaptive_planning`, off by default. |
| 3E — substrate v2 | **closed, mixed verdict** | Quality gate PASS; cost gate narrow FAIL; **compounding gate FAIL** (mean ratio 1.091 > 0.70 target). Self-verdict: "shippable prompt protocol, not a substrate; 3F remains blocked." See `PHASE3E_PROGRESS.md` final closeout section. |
| 3F — workspace / capsules | **unauthorized; isolated** | Built against the explicit 3E "3F remains blocked" verdict. Code (`workspace.py`, `capsule_store.py`, `test_phase3f_week1.py`) lives on the separate `reasoning-phase3f-unauthorized` git branch and MUST NOT be merged into `reasoning-architecture` without overriding the 3E verdict. |
| Post-3E — signal usefulness | **planned** | The single bounded follow-up authorized by the 3E closeout. See `POST_3E_SIGNAL_USEFULNESS_PLAN.md`. |

### What this doc undersold

The original §0 status was written 2026-05-19 and predicted only Phase 1/2A/3A/3B-1/3E would land. In reality Phase 3C and 3D-1 also shipped (uncovered by the audit on 2026-05-24); Phase 3F was built unauthorized; and the previous status line confused "doc-claimed in flight" with "code state." The doc has been kept in sync since this refresh.

### Current next step

The Phase 3E final verdict is the load-bearing decision the doc has to respect. The only authorized post-3E work is:

1. **Finish the Phase 3B-1 corpus gate** — collect 3+ additional graded real-session traces to reach the >=10 threshold before considering 3B-2.
2. **Execute `POST_3E_SIGNAL_USEFULNESS_PLAN.md`** — the bounded investigation of why compounding failed (signal pollution on warm runs in `payment_psp` and `migration_zd`).
3. **Decide on 3F** — read `PHASE3F_IMPLEMENTATION_PLAN.md` + the workspace/capsule code on the `reasoning-phase3f-unauthorized` branch and either (a) merge with explicit override of the 3E verdict, (b) cherry-pick the parts that are independently valuable, or (c) delete the branch.

Related docs:
- `SFT_DESIGN.md` — sibling design doc for the SFT experiment (orthogonal direction)
- `PROGRESS.md` — chronological experiment log
- `PHASE*_PROGRESS.md` — per-phase implementation notes (authoritative for that phase)
- `PHASE3F_*_PLAN.md` and `PHASE3G_PREPARATION_ROADMAP.md` — on `reasoning-phase3f-unauthorized` branch only; not authorized work

Open ideas not addressed in this doc that may matter later:
- **Trust scores** on procedures (not just citation count, but a quality signal)
- **Procedure retirement** (active deprecation vs passive decay)
- **Cross-domain procedure transfer** (does a procedure from CS reasoning transfer to physics reasoning?)
- **Procedure debugging interface** (when a wrong answer emerges, surface the procedure chain that produced it)

