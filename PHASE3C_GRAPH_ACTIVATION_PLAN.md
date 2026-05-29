# Phase-3C implementation plan - context-aware graph activation

**Status:** first implementation slice complete. Schemas, activation engine, task-frame prompt injection, activation trace persistence, session-subgraph projection, and coverage checks are implemented. Real merged-graph smokes remain next.

**Parent docs:**
- `REASONING_ARCHITECTURE.md` - section 2.7 context-aware graph activation
- `PHASE3B_PLAN.md` / `PHASE3B_PROGRESS.md` - trace evidence that motivates this phase

**Related follow-up:**
- `PHASE3D_ADAPTIVE_PLANNING_PLAN.md` - plan-tree checkpoints,
  backtracking, and plan revision using 3C task frames

**Date:** 2026-05-21

---

## 0. Executive summary

Phase 3B makes procedure reuse measurable. It does not make the graph useful when the model solves a task directly and no procedure fires.

Phase 3C adds a second path: graph nodes act like typed reactive objects. For each session, the system builds a `SessionContext`, activates nearby nodes, converts their reactions into `GraphSignal` objects, creates provisional adjacent nodes when context is missing, and renders a compact `GraphTaskFrame` into the prompt.

This keeps the original graph-as-computational-medium idea while staying bounded: no arbitrary code on nodes, no long-term graph mutation during answer generation, and every activation is traceable.

---

## 1. Goals

1. Make the graph useful in direct-answer sessions where no procedure should fire.
2. Let nodes interact through typed signals, not through raw prompt text alone.
3. Detect missing context and create provisional adjacent session nodes (`session_gap`, `session_bridge`) when useful.
4. Produce a compact `GraphTaskFrame` that shapes the answer before final generation.
5. Log a replayable `GraphActivationTrace`.
6. Add post-answer coverage checks that record which activated constraints, pitfalls, and suggestions were addressed.

---

## 2. Non-goals

- No dynamic embeddings in this phase.
- No arbitrary executable code stored on graph nodes.
- No long-term promotion of synthetic activation nodes in v1.
- No procedure proposal, validation, installation, or promotion work; that remains Phase 3B-2.
- No multi-session merge or concurrency work.

---

## 3. Core schemas

### 3.1 `SessionContext`

```python
@dataclass
class SessionContext:
    session_id: str
    graph_id: str
    domain: Optional[str]
    question: str
    task_kind: Optional[str]          # e.g. "algorithm_design", "verification", "explanation"
    constraints: list[str]
    requested_outputs: list[str]
    retrieved_anchor_ids: list[str]
    active_node_ids: list[str]
    missing_context: list[str]
    budget_snapshot: dict[str, Any]
```

### 3.2 `GraphSignal`

```python
@dataclass
class GraphSignal:
    signal_id: str
    source_node_id: str
    kind: Literal[
        "constraint",
        "pitfall",
        "procedure_suggestion",
        "example",
        "bridge_hypothesis",
        "missing_context",
        "answer_requirement",
    ]
    payload: str
    confidence: float
    evidence_node_ids: list[str]
```

### 3.3 `ActivatedNode`

```python
@dataclass
class ActivatedNode:
    node_id: str
    node_type: str
    activation_score: float
    activation_reason: str
    emitted_signal_ids: list[str]
```

### 3.4 `FrameItem` and `GraphTaskFrame`

```python
@dataclass
class FrameItem:
    item_id: str
    kind: str
    text: str
    priority: int
    source_signal_ids: list[str]

@dataclass
class GraphTaskFrame:
    session_id: str
    constraints: list[FrameItem]
    pitfalls: list[FrameItem]
    suggested_structures: list[FrameItem]
    relevant_examples: list[FrameItem]
    procedure_suggestions: list[FrameItem]
    unresolved_gaps: list[FrameItem]
```

### 3.5 `GraphActivationTrace`

```python
@dataclass
class GraphActivationTrace:
    session_id: str
    context: SessionContext
    activated_nodes: list[ActivatedNode]
    signals: list[GraphSignal]
    provisional_nodes: list[dict[str, Any]]
    task_frame: GraphTaskFrame
    coverage_result: Optional[dict[str, Any]]
```

---

## 4. Activation pipeline

1. Build `SessionContext` from the user question, graph id, retrieved anchors, known domain, task constraints, and current budget state.
2. Seed active nodes from top retrieval anchors.
3. Expand at most `max_hops=2` through high-value edges such as `depends_on`, `validates`, `replacement_for`, `calls`, and existing bridge edges.
4. Run the behavior registry for each active node type.
5. Emit typed `GraphSignal` objects.
6. Compare required context against emitted signals. If context is missing, create session-scoped `session_gap` nodes.
7. If nearby nodes can plausibly connect the gap to the current task, create session-scoped `session_bridge` nodes.
8. Rank and deduplicate signals into a compact `GraphTaskFrame`.
9. Inject the frame into the prompt before the final answer directive.
10. After the answer, run a coverage check and persist `GraphActivationTrace`.

---

## 5. Behavior registry

Nodes do not execute arbitrary node-local programs. Phase 3C uses a small registry owned by the codebase:

```python
BehaviorFn = Callable[[GraphNode, SessionContext], list[GraphSignal]]

BEHAVIOR_REGISTRY = {
    "fact": emit_fact_constraints,
    "claim": emit_claim_constraints_or_warnings,
    "example": emit_relevant_example_signal,
    "summary": emit_summary_signal,
    "hypothesis": emit_low_confidence_hint,
    "procedure": emit_procedure_suggestion,
    "failure_pattern": emit_pitfall_signal,
}
```

This gives node-like reactivity without turning the graph into an unsafe plugin runtime.

---

## 6. Provisional adjacent nodes

When exact context is absent, the activation engine may create session-scoped nodes:

- `session_gap`: a missing requirement, constraint, example type, or bridge needed for the current task.
- `session_bridge`: a tentative relation connecting active graph context to the missing context.

Rules:

- Provisional nodes live only in the session subgraph.
- They must carry `derived_from` edges to the source nodes or context fields that caused creation.
- They must never be promoted automatically.
- Their only immediate use is to improve the current task frame and activation trace.

Example:

```yaml
session_gap:
  text: "Need all-negative handling for non-empty max subarray."
  derived_from:
    - user_constraint_negative_values_allowed
    - user_constraint_non_empty_subarray

session_bridge:
  text: "Segment-tree max-subarray combine must avoid empty-subarray clamp."
  fills_gap: <session_gap_id>
  derived_from:
    - example_max_subarray_segment_tree
```

---

## 7. Prompt integration

The prompt receives a compact frame, not the full activation trace.

Example prompt fragment:

```text
<graph_task_frame>
Constraints:
- Non-empty subarray: all-negative arrays must return the maximum element, not 0.
- Use long long for sums.

Pitfalls:
- Do not clamp leaf best/prefix/suffix to 0; that allows the empty subarray.

Suggested structures:
- Segment tree node fields: sum, pref, suff, best.
</graph_task_frame>
```

The model can still answer directly. Phase 3C is not forcing procedure dispatch; it is supplying active, structured context.

If the task frame contains no procedure suggestions, the reasoning prompt hides
the procedure catalog and switches to direct-answer mode. This prevents weaker
models from inventing irrelevant procedure calls merely because the catalog is
visible.

---

## 8. Defaults and budgets

| Budget | Default |
|---|---:|
| max activation hops | 2 |
| max active nodes | 24 |
| max signals | 40 |
| max provisional nodes | 6 |
| max task-frame items | 12 |
| max frame chars | 2500 |

If budgets are hit, the trace records truncation and the frame favors higher-priority constraints and pitfalls.

---

## 9. Test plan

Unit tests:

1. `SessionContext` round-trips through JSON.
2. `GraphSignal`, `GraphTaskFrame`, and `GraphActivationTrace` round-trip through JSON.
3. Behavior registry emits expected signals for `example`, `procedure`, and `failure_pattern` nodes.
4. Gap detector creates `session_gap` only when required context is not covered by existing signals.
5. Bridge detector creates `session_bridge` only in the session subgraph.
6. Frame ranking deduplicates overlapping signals and respects `max_task_frame_items`.
7. Coverage checker flags a crafted IOI answer that omits `long long` or all-negative handling.

Integration tests:

1. IOI max-subarray task produces a non-empty frame with non-empty-subarray, all-negative, `long long`, and segment-tree combine items.
2. Dijkstra negative-edge task emits both a pitfall and a shortest-path procedure suggestion.
3. A direct-answer session can use the graph frame without any procedure dispatch.
4. Trace persistence is replayable and does not mutate the long-term graph.

---

## 10. Build order

### 3C.1 - schemas and trace persistence

Add dataclasses, JSON round-trip helpers, and `data/activation_traces/activation_<YYYYMMDD>.jsonl`.

Status: implemented in `reasoning/activation.py`.

### 3C.2 - context builder

Extract constraints, requested outputs, task kind, domain, and anchor ids from the existing reasoning-loop inputs.

Status: implemented with heuristic extraction in `run_graph_activation()`.

### 3C.3 - behavior registry

Implement conservative signal emitters for existing node types. Start with `example`, `procedure`, `failure_pattern`, `fact`, and `claim`.

Status: implemented. Procedure suggestions are emitted from the loaded procedure pool; graph-node behaviors are registry-based and do not execute node-local code.

### 3C.4 - activation engine and gap detector

Seed from anchors, expand within budget, emit signals, synthesize session-only gaps and bridges.

Status: implemented. Provisional `session_gap` and `session_bridge` nodes are projected only into the session subgraph.

### 3C.5 - task-frame renderer

Rank, dedupe, and render the compact prompt fragment.

Status: implemented as `<graph_task_frame>` prompt injection.

### 3C.6 - answer coverage check

Compare final answer text against frame items and log addressed vs missed requirements.

Status: implemented with lexical heuristics and persisted on `GraphActivationTrace.coverage_result`.

### 3C.7 - real smokes

Run the Dijkstra and IOI tasks against `merged_graph`. Compare:

- procedure calls
- activation frame contents
- answer quality
- coverage misses
- trace cleanliness

Status: complete for first implementation slice. Real smokes passed on
2026-05-21; see `PHASE3C_PROGRESS.md` for artifact paths and inspection notes.

---

## 11. Acceptance criteria

1. IOI direct-answer task produces a `GraphTaskFrame` with all-negative, non-empty-subarray, `long long`, and segment-tree combine guidance without requiring a procedure call.
2. Dijkstra negative-edge task produces a shortest-path procedure suggestion and a negative-edge pitfall.
3. Missing required context creates `session_gap` nodes only in the session subgraph.
4. Bridge synthesis creates `session_bridge` nodes only in the session subgraph and links them with `fills_gap`.
5. Long-term graph files are unchanged by activation.
6. `GraphActivationTrace` JSONL round-trips and is sufficient to replay activation decisions.
7. Coverage check catches at least one crafted incomplete answer.
8. Full backend suite passes after integration.

---

## 12. Open decisions

| Decision | Default | Notes |
|---|---|---|
| Task-kind classifier | heuristic first | Use question features before adding any LLM call. |
| Activation score threshold | no hard threshold in v1 | Use budgets and rankers first; tune after traces exist. |
| Frame injection point | before final answer directive | Keeps the frame close to answer generation. |
| Coverage checker | lexical + small heuristics | Avoid another model call until we see failures. |
| Promotion of synthetic nodes | deferred | Session-only for Phase 3C. |
| Token-softmax confidence | unavailable in current runtime | The frontend shells out to `opencode run`, which returns text but not logprobs/logits. Add a separate `model_confidence` only if a backend exposes scoring data. |
| Adaptive planning | Phase 3D | Plans should be checkpoint trees that can revise and backtrack when a branch fails. |

---

## 13. Status / next move

This is a design plan, not implementation. The next implementation step remains Phase 3B-1 corpus collection unless the project explicitly pivots to building 3C first.
