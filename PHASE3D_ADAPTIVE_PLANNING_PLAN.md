# Phase-3D implementation plan - adaptive plan-tree reasoning

**Status:** Sub-phase 3D-1 implemented. Deterministic plan-tree data
structures, backtracking policy, coverage-gated finalization, session-subgraph
projection, and unit tests are in place. Prompt-mode integration with
`run_reasoning()` remains gated and not enabled by default.

**Parent docs:**
- `REASONING_ARCHITECTURE.md` - context-aware activation and session subgraphs
- `PHASE3C_GRAPH_ACTIVATION_PLAN.md` / `PHASE3C_PROGRESS.md` - task frame,
  coverage checks, and Qwen smoke observations

**Date:** 2026-05-21

---

## Implementation status

### 3D-1 complete

Files:

- `reasoning/adaptive_planning.py`
- `reasoning/adaptive_planning_examples.py`
- `reasoning/tests/test_adaptive_planning.py`
- `reasoning/schemas.py`
- `run_phase3d_synthetic_drivers.py`

Implemented:

1. `PlanNode`, `PlanCheckResult`, `PlanState`, and `AdaptivePlanTree`.
2. Deterministic checkpoint scoring for `local_step`, `algorithm_choice`,
   `task_interpretation`, and `unknown` failures.
3. Revision flow that records the failed check, marks the failed branch
   abandoned, backtracks to the selected checkpoint, and creates a sibling
   branch.
4. Revision/backtrack/depth budget caps.
5. Coverage-gated finalization using Phase 3C `GraphTaskFrame` priorities.
6. Session-subgraph projection for `plan_node`, `plan_check`, `plan_child`,
   `checked_by`, `failed_because`, `backtracked_to`, and `plan_revision_of`.
7. Synthetic IOI and Dijkstra driver traces, persisted for replay under
   `data/session_subgraphs/`.

Verified:

```powershell
python -m unittest reasoning.tests.test_adaptive_planning
python -m unittest discover reasoning/tests
```

Full reasoning suite result: 293 tests OK, 1 expected failure.

### 3D-1b complete

Artifacts:

- `data/session_subgraphs/sess_phase3d_ioi_synthetic`
- `data/session_subgraphs/sess_phase3d_dijkstra_synthetic`

### 3D-2 pending

Prompt-mode integration should come after the deterministic substrate has one
or two synthetic driver traces. Start behind `enable_adaptive_planning`; do not
turn it on by default.

---

## 0. Executive summary

The current loop is mostly one-shot unless a procedure fires. That works for
strong models, but weaker local models expose a need for a clearer reasoning
control structure.

Phase 3D adds adaptive planning as a **checkpoint tree**, not a linear list.
The model plans one step, executes it, checks it against graph/task constraints,
and either finalizes or backtracks to the best checkpoint and creates a better
sibling plan.

Plans are provisional. A wrong plan is not a crash and not a restart; it is a
failed branch in a session-local plan tree.

---

## 1. Goals

1. Split reasoning into focused modes: focus, plan, execute, check, revise,
   finalize.
2. Represent the plan as a session-scoped tree of checkpoints.
3. Let the model revise a plan when checks fail.
4. Backtrack to the nearest useful checkpoint rather than restarting from the
   root.
5. Use Phase 3C `GraphTaskFrame` and coverage checks as the objective function
   for plan validation.
6. Persist plan nodes, plan edges, failures, and revisions in the session
   subgraph for replay.

---

## 2. Non-goals

- No long-term promotion of plan nodes in v1.
- No free-form multi-agent planner.
- No unbounded tree search.
- No arbitrary executable plan nodes.
- No replacement for procedure dispatch; procedures remain one possible
  execution mode.
- No reliance on hidden chain-of-thought. The loop uses structured visible
  control outputs.

---

## 3. Core idea

Instead of:

```text
plan -> answer
```

Use:

```text
focus -> plan node -> execute node -> check
                         |
                         +-- pass -> finalize
                         |
                         +-- fail -> choose checkpoint -> sibling plan -> execute -> check
```

Example IOI tree:

```text
root: solve dynamic max-subarray
└─ choose_algorithm
   ├─ kadane_direct
   │  └─ failed: does not support q online point updates
   └─ segment_tree
      ├─ define_node_state
      ├─ derive_merge_rule
      ├─ verify_all_negative
      ├─ verify_long_long
      └─ final_answer
```

Example Dijkstra tree:

```text
root: decide if Dijkstra is safe
└─ choose_shortest_path_algorithm
   ├─ dijkstra
   │  └─ failed: negative edge violates nonnegative-weight precondition
   └─ bellman_ford
      └─ final_answer
```

---

## 4. Schemas

### 4.1 `PlanNode`

```python
@dataclass
class PlanNode:
    node_id: str
    parent_id: Optional[str]
    goal: str
    hypothesis: str
    mode: Literal["focus", "plan", "execute", "check", "repair", "finalize"]
    status: Literal["pending", "active", "passed", "failed", "abandoned"]
    checkpoint_quality: float
    failure_reason: Optional[str]
    evidence_ids: list[str]
    created_step: int
```

### 4.2 `PlanEdge`

Use existing `SessionEdge` shape with relations:

- `plan_child`
- `plan_revision_of`
- `backtracked_to`
- `checked_by`
- `failed_because`
- `supports_plan`

### 4.3 `PlanState`

```python
@dataclass
class PlanState:
    session_id: str
    root_node_id: str
    active_node_id: str
    revision_count: int
    max_revisions: int
    finalized: bool
    last_failure_reason: Optional[str]
```

### 4.4 `PlanCheckResult`

```python
@dataclass
class PlanCheckResult:
    checked_node_id: str
    passed: bool
    failure_scope: Literal["local_step", "algorithm_choice", "task_interpretation", "unknown"]
    failed_requirements: list[str]
    suggested_backtrack_node_id: Optional[str]
    reason: str
```

---

## 5. Control outputs

The model should not produce an arbitrary essay during planning. Each mode has
one structured output.

### 5.1 Focus mode

```text
FOCUS_RESULT
task_kind: algorithm_design
required_outputs:
  - C++17 code
  - complexity
constraints:
  - q online updates
  - non-empty subarray
  - negative values allowed
  - 64-bit sums
END_FOCUS_RESULT
```

### 5.2 Plan mode

```text
PLAN_NODE
goal: choose data structure
hypothesis: segment tree with sum/prefix/suffix/best supports point updates
mode: execute
checkpoint_quality: 0.82
END_PLAN_NODE
```

### 5.3 Execute mode

```text
EXECUTE_RESULT
node_id: plan_...
result: defined segment tree node fields and merge rule
new_evidence:
  - merge rule covers cross-boundary subarray
END_EXECUTE_RESULT
```

### 5.4 Check mode

```text
CHECK_RESULT
passed: false
failure_scope: algorithm_choice
failed_requirements:
  - online updates q <= 200000
reason: Kadane is O(n) per query and cannot handle point updates efficiently.
suggested_backtrack: choose_algorithm
END_CHECK_RESULT
```

### 5.5 Revise mode

```text
REVISE_PLAN
backtrack_to: choose_algorithm
abandon: kadane_direct
new_hypothesis: segment tree with max-subarray aggregate
reason: previous branch failed update constraint
END_REVISE_PLAN
```

---

## 6. Backtracking policy

When a check fails, choose the best ancestor checkpoint:

```python
score = (
    checkpoint_quality
    + evidence_support
    + reusable_context_score
    - failure_scope_penalty
    - distance_from_failed_node_penalty
    - repeated_failure_penalty
)
```

Heuristic defaults:

| Failure scope | Preferred checkpoint |
|---|---|
| `local_step` | parent of failed node |
| `algorithm_choice` | nearest ancestor whose goal contains "choose" or whose mode is `plan` |
| `task_interpretation` | root or focus node |
| `unknown` | highest-quality ancestor within last 3 hops |

This keeps the search efficient. A failed merge-rule derivation should not
restart task interpretation; a failed algorithm choice should not keep trying
minor implementation repairs.

---

## 7. Integration with current system

### Phase 3C activation

`GraphTaskFrame` supplies constraints, pitfalls, suggested structures, procedure
suggestions, and unresolved gaps. Phase 3D uses these as the planner's
objective function.

### Procedure dispatcher

Procedure suggestions become candidate plan branches. The planner decides
whether to expose the procedure catalog and which procedure to invoke.

### Coverage checker

Coverage becomes a check signal. Missed critical frame items can force
`REVISE_PLAN` instead of allowing finalization.

### Session subgraph

Every plan node and revision lives in the session subgraph. This gives us a
replayable plan tree and lets the UI show where the system backtracked.

### Trace logs

`SessionTrace` can gain optional fields later:

- `plan_node_count`
- `plan_revision_count`
- `backtrack_count`
- `final_plan_depth`
- `failed_plan_reasons`

Do not add these to Phase 3B macro extraction until 3D has real traces.

---

## 8. Prompting strategy

Use small, mode-specific prompts. The model should focus on one operation:

1. Focus prompt: extract requirements only.
2. Plan prompt: choose next step only.
3. Execute prompt: execute the active node only.
4. Check prompt: compare result to task frame only.
5. Revise prompt: choose checkpoint and sibling branch only.
6. Finalize prompt: answer only from passed nodes.

This is more calls than one-shot prompting, so use budgets:

| Budget | Default |
|---|---:|
| max plan nodes | 12 |
| max revisions | 3 |
| max backtracks | 3 |
| max plan depth | 6 |
| max planning LLM calls | 6 |

For strong models, the loop can collapse focus+plan+execute into one call. For
weaker local models, keep the modes separate.

---

## 9. Optimal application points

Start where the current system shows the most benefit:

1. **Algorithm-design tasks**: Kadane vs segment tree, Dijkstra vs Bellman-Ford,
   binary search vs DP. These have clear failure scopes.
2. **Procedure-dispatch decisions**: avoid showing the procedure catalog unless
   the active plan branch chooses a procedure.
3. **Coverage repair**: if final answer misses `long long`, all-negative, or
   negative-edge caveats, backtrack to the relevant answer-construction node.
4. **Graph activation noise**: if task-kind checks reject a signal, backtrack to
   activation/ranking rather than answer generation.
5. **Macro-candidate validation later**: recurring plan subtrees can become
   better macro candidates than raw procedure call sequences.

---

## 10. Test plan

Unit tests:

1. Plan tree round-trip.
2. Backtracking selects parent for local-step failure.
3. Backtracking selects algorithm-choice ancestor for wrong-algorithm failure.
4. Revision creates sibling node and marks failed branch abandoned.
5. Budget caps stop unbounded revisions.
6. Coverage miss prevents finalization when item priority is critical.

Integration tests:

1. IOI task first proposes Kadane, check fails update constraint, backtracks to
   algorithm choice, selects segment tree, finalizes correctly.
2. Dijkstra task first proposes Dijkstra, check fails negative-edge constraint,
   backtracks to shortest-path choice, selects Bellman-Ford.
3. Direct simple task finalizes without backtracking.
4. Procedure task invokes exactly the chosen procedure and does not expose the
   full procedure catalog when no procedure branch is active.

---

## 11. Acceptance criteria

1. Plan nodes and backtracking edges persist in the session subgraph.
2. A failed plan branch is marked `failed` or `abandoned`, never deleted.
3. The planner can revise an algorithm choice without restarting the whole
   session.
4. IOI and Dijkstra adversarial tests both show at least one successful
   backtrack in synthetic/stubbed tests.
5. Real Qwen smoke shows fewer false procedure invocations than the current
   one-shot prompt.
6. Full reasoning suite passes.

---

## 12. Open decisions

| Decision | Default | Notes |
|---|---|---|
| Enable 3D by default | no | Start behind `enable_adaptive_planning`. |
| Planner output format | tagged blocks | Easier for Qwen than strict JSON, but still parseable. |
| Check implementation | hybrid | Deterministic coverage checks first, optional model check later. |
| Backtrack score | heuristic | Learn/calibrate later from traces. |
| UI rendering | session subgraph | Plan tree can reuse existing graph rendering. |

---

## 13. Next move

3D-1 and 3D-1b are complete. The next slice is prompt-mode integration behind
`enable_adaptive_planning`, still off by default:

1. Add a mode driver for focus, plan, execute, check, revise, and finalize.
2. Run the driver against the existing IOI and Dijkstra scenarios and compare
   the resulting tree against the deterministic synthetic artifacts.
3. Only after the prompt driver reproduces those trees, run Qwen/OpenCode real
   smokes.
