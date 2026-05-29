# Phase-3C implementation progress

Status as of 2026-05-21. Plan: `PHASE3C_GRAPH_ACTIVATION_PLAN.md`.

Phase 3C adds context-aware graph activation for sessions where procedure
dispatch is not the only useful path. The first implementation slice is now
in place.

---

## Implemented

| Area | Status | Notes |
|---|---:|---|
| Activation schemas | done | `SessionContext`, `GraphSignal`, `ActivatedNode`, `FrameItem`, `GraphTaskFrame`, and `GraphActivationTrace` live in `reasoning/activation.py`. |
| Activation trace JSONL | done | `ActivationTraceLogger` writes `data/activation_traces/activation_<YYYYMMDD>.jsonl` in production and temp roots in tests. |
| Context builder | done | Heuristically extracts task kind, constraints, requested outputs, domain, anchors, and budget snapshot. |
| Behavior registry | done | Node-type behaviors emit typed signals for facts, claims, examples, summaries, hypotheses, failure patterns, and procedure suggestions. |
| Provisional nodes | done | Missing context can create session-scoped `session_gap` and `session_bridge` nodes with `derived_from` / `fills_gap` edges. |
| Task frame rendering | done | `<graph_task_frame>` is injected into prompts as private guidance. Category quotas reserve room for pitfalls, procedure suggestions, and suggested structures. |
| Coverage check | done | Final answer is checked against frame items and persisted on the activation trace. |
| Reasoning-loop integration | done | `run_reasoning()` builds activation once per session, injects the frame into prompts, projects activation nodes into the session subgraph, and persists the trace. |

---

## Files added or modified

```
reasoning/
├── activation.py                  new - Phase 3C activation engine + schemas
├── reasoning_loop.py              modified - activation prompt injection + trace persistence
├── schemas.py                     modified - Phase 3C node-type literals
└── tests/
    └── test_activation.py         new - round-trip, frame, coverage, projection tests

local_model_command.py/.cmd        new - local llama-server command shim for Qwen/GLM tests
PHASE3C_GRAPH_ACTIVATION_PLAN.md   updated status
PHASE3C_PROGRESS.md                new - this progress log
PHASE3D_ADAPTIVE_PLANNING_PLAN.md  new - follow-up design for plan-tree backtracking
```

---

## Validation

Targeted:

```powershell
python -m unittest reasoning.tests.test_activation reasoning.tests.test_reasoning_loop
```

Result: **24 tests passing**.

Full reasoning suite:

```powershell
python -m unittest discover reasoning/tests
```

Result: **311 tests passing** with **1 documented expected failure**.

`pytest` is not installed in this environment, so validation used `unittest`.

---

## Real merged-graph smoke

Run on 2026-05-21 with `run_frontend_pipeline_debug.py`, `graph_id=merged_graph`,
`reasoning_mode=substrate`.

### Dijkstra negative-edge smoke

Artifact:

`artifacts/frontend_pipeline_debug/frontend_pipeline_debug_20260521_102000.json`

Session:

`data/session_subgraphs/sess_5059f5adcb91`

Result:

- Correct answer: Dijkstra is unsafe because `b->c` has weight `-2`; use
  Bellman-Ford.
- Procedure path fired: `VerifyShortestPath > VerifyAlgorithmPreconditions,
  VerifyNonNegativeEdges`.
- Activation frame included negative-edge pitfalls and procedure suggestions
  for `VerifyShortestPath`, `VerifyAlgorithmPreconditions`, and
  `VerifyNonNegativeEdges`.
- Trace quality: no budget exhaustion, no procedure errors, no contradiction
  signals.

### IOI dynamic max-subarray smoke

Artifact:

`artifacts/frontend_pipeline_debug/frontend_pipeline_debug_20260521_102247.json`

Session:

`data/session_subgraphs/sess_285a7f93947b`

Result:

- Correct direct-answer solution with a segment tree storing `sum`,
  `max_prefix`, `max_suffix`, and `max_sub`.
- Uses `long long`.
- Handles all-negative arrays by initializing leaves to the element value, not
  zero.
- No procedure calls fired.
- Activation frame included segment-tree structure, `long long/int64`, and
  all-negative/non-empty guidance.

Observation:

- The frame still contains some noisy neighboring context for IOI tasks
  (Kadane, divide-and-conquer, binary-search safety). The category quotas keep
  the critical items visible, but Phase 3C should next add task-kind-aware
  suppression/reranking for irrelevant anchors.

### Qwen local-model smoke

Local runtime:

`cache/models/Qwen3-4B-Instruct-2507-Q4_K_M.gguf` through `llama-server` on
`127.0.0.1:6768`, called via `local_model_command.cmd`.

Artifacts:

- Dijkstra: `artifacts/frontend_pipeline_debug/frontend_pipeline_debug_20260521_121557.json`
- IOI after prompt-policy fix: `artifacts/frontend_pipeline_debug/frontend_pipeline_debug_20260521_122147.json`

Results:

- Dijkstra: correct, no visible `<think>`, procedure path fired.
- IOI: correct direct-answer path, one model call, no procedure dispatch.
- The smaller Qwen model exposed a useful prompt bug: when no procedure was
  task-relevant, the generic procedure catalog still invited false dispatch.
  `reasoning_loop._build_prompt()` now hides the procedure catalog whenever a
  Phase 3C task frame has no procedure suggestions.
- The same run motivates Phase 3D: a weaker model benefits from isolated
  planning/checking modes and a plan tree that can revise wrong branches
  instead of blending planning, execution, and final answer in one step.

---

## Confidence note

The current frontend model path shells out to `opencode run` and receives only
text, stderr, return code, and elapsed time. It does not expose token logits or
logprobs, so the visible `confidence` remains a heuristic.

Local Qwen through `llama-server` does expose token logprobs. The added
`local_model_command.py` shim can request and log token-confidence stats when
`LOCAL_LLM_LOGPROBS=1`:

```powershell
$env:MODEL_COMMAND=(Resolve-Path .\local_model_command.cmd).Path
$env:LOCAL_LLM_BASE_URL='http://127.0.0.1:6768'
$env:LOCAL_LLM_LOGPROBS='1'
python run_frontend_pipeline_debug.py --no-attach ...
```

Stats are written to `data/model_confidence/local_model_calls_<YYYYMMDD>.jsonl`
and include mean logprob, p10 logprob, min logprob, and mean top-1/top-2
margin. Keep this separate as `model_confidence`; raw token probability alone
is not a reliable correctness estimate and should be calibrated against graded
traces before driving UI confidence.

Qwen local smoke used:

```powershell
llama-server -m cache/models/Qwen3-4B-Instruct-2507-Q4_K_M.gguf -ngl 99 -c 8192 --host 127.0.0.1 --port 6768
```

OpenCode note: `opencode run` has a `--thinking` flag that shows thinking
blocks, but the frontend's current OpenCode path already receives only text and
does not expose logits/logprobs. Disabling visible thinking there is not enough
to build calibrated confidence.

---

## Current behavior

- Direct-answer sessions can receive graph-derived constraints, pitfalls,
  examples, procedure suggestions, and inferred task requirements without
  forcing procedure dispatch.
- The frame is private guidance. The directive explicitly forbids leaking graph
  internals, node ids, or frame names into the final answer.
- Activation is advisory and fault-tolerant; failures in activation or trace
  persistence do not block the core reasoning loop.
- Long-term graph files are not mutated by activation.

---

## Next move

Planning/design:

1. Phase 3D-1 and 3D-1b are implemented. See `PHASE3D_PROGRESS.md`.
2. Synthetic IOI and Dijkstra adaptive-driver traces now exist under
   `data/session_subgraphs/sess_phase3d_ioi_synthetic` and
   `data/session_subgraphs/sess_phase3d_dijkstra_synthetic`.
3. Keep `run_reasoning()` prompt-mode integration behind
   `enable_adaptive_planning` and off by default until those traces are clean.

Phase 3C tuning:

1. Add task-kind-aware suppression/reranking so static Kadane/binary-search
   neighbors do not crowd dynamic segment-tree tasks.
2. Add an optional `model_confidence` design path if the runtime can expose
   logprobs or a scoring API.
