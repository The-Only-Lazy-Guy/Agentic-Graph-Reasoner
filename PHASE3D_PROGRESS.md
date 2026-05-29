# Phase-3D progress - adaptive plan-tree reasoning

Status as of 2026-05-21. Plan: `PHASE3D_ADAPTIVE_PLANNING_PLAN.md`.

---

## Implemented

### Sub-phase 3D-1: deterministic plan-tree substrate

Files:

- `reasoning/adaptive_planning.py`
- `reasoning/adaptive_planning_examples.py`
- `reasoning/tests/test_adaptive_planning.py`
- `reasoning/schemas.py`
- `run_phase3d_synthetic_drivers.py`

What landed:

1. Plan tree primitives: `PlanNode`, `PlanCheckResult`, `PlanState`, and
   `AdaptivePlanTree`.
2. Deterministic backtracking policy:
   - `local_step` prefers the immediate parent repair checkpoint.
   - `algorithm_choice` prefers the nearest choice/plan ancestor.
   - `task_interpretation` prefers root/focus checkpoints.
   - `unknown` uses the general quality/evidence/distance score.
3. Revision flow:
   - records a failed check,
   - marks the failed branch `abandoned`,
   - emits `backtracked_to`,
   - creates a sibling branch from the selected checkpoint,
   - emits `plan_revision_of`.
4. Budget caps for revisions, backtracks, and plan depth.
5. Coverage-gated finalization using Phase 3C `GraphTaskFrame` priorities.
6. Session-subgraph projection for plan/check nodes and replay edges.
7. `plan_node` and `plan_check` added to the typed schema node list.
8. Synthetic IOI and Dijkstra adaptive drivers:
   - IOI: Kadane branch fails online-update constraint, backtracks to algorithm
     choice, segment-tree branch succeeds.
   - Dijkstra: Dijkstra branch fails negative-edge precondition, backtracks to
     shortest-path choice, Bellman-Ford branch succeeds.
9. Replay artifacts persisted under:
   - `data/session_subgraphs/sess_phase3d_ioi_synthetic`
   - `data/session_subgraphs/sess_phase3d_dijkstra_synthetic`

Verification:

```powershell
python -m unittest reasoning.tests.test_adaptive_planning
python -m unittest discover reasoning/tests
```

Result:

- Focused Phase 3D tests: 11 passed.
- Full reasoning suite: 293 tests passed, 1 expected failure.

Additional repair during 3D-1b:

- Restored `reasoning/trace_log.py`, required by Phase 3B macro extraction.
- Repaired a stray duplicate block in `reasoning/reasoning_loop.py`.
- Restored the Phase 3C prompt-builder policy where a task frame with no
  procedure suggestions hides the procedure catalog.

---

## Design notes

The first implementation slice is intentionally model-free. This gives us a
stable object model and replayable tree before prompt integration adds noise.

The key invariant is now covered by tests: a wrong plan branch is retained as
evidence, not deleted, and revision creates a sibling from the best checkpoint
instead of restarting the whole session.

Coverage is already usable as a deterministic check signal. If a final answer
misses a high-priority task-frame item, `try_finalize()` records a failed
`plan_check` and leaves `finalized=False`.

---

## Next move

Recommended next slice:

1. Start 3D-2: mode-specific prompt integration behind
   `enable_adaptive_planning`, still off by default.
2. Feed the prompt driver the same IOI and Dijkstra scenarios first, comparing
   its tree against the deterministic synthetic artifacts.
3. Only then try Qwen/OpenCode real-model smokes.
