# Phase-1 implementation progress

Status as of 2026-05-20 after landing Sub-phases 1.7, 1.8, and 1.9.
Plan doc: `PHASE1_PLAN.md`. Architecture doc: `REASONING_ARCHITECTURE.md`.

Important correction: the actual frontend for this project is the sibling repo
`../graph-front-end`, not `graph_final/graph_agent_frontend.py`.

---

## Sub-phase status

| # | Sub-phase | Status | Tests | Notes |
|---|---|---|---|---|
| 1.1 | Schemas + round-trip JSON | done | 13/13 | `reasoning/schemas.py` |
| 1.2 | Session subgraph + audit log | done | 33/33 | `reasoning/{audit_log,session_subgraph}.py`; fixed the shared-reference journaling bug by deep-copying values at journal time |
| 1.3 | Budget tracker | done | 10/10 | `reasoning/budgets.py` |
| 1.4 | Retrieval boost | done | 6/6 | `reasoning/retrieval_boost.py`; failure-pattern nodes get 1.4x score boost before top-k selection |
| 1.5 | Dispatcher | done | 26/26 | `reasoning/dispatcher.py`; regex scan + constrained mutation grammar |
| 1.6 | Seed procedure + consolidation | done | 14/14 | `reasoning/{procedures/verify_algorithm_preconditions,consolidation}.py` |
| 1.7 | Reasoning loop orchestrator | done | 9/9 | `reasoning/reasoning_loop.py`; now returns the in-memory session subgraph as part of `ReasoningResult` and falls back cleanly if no `<answer>` block is produced |
| 1.8 | Front-end toggle | done | covered by 1.9 | Implemented in `../graph-front-end/api/frontend_api.py`; mode selected by `REASONING_MODE` |
| 1.9 | End-to-end integration test | done | 3/3 | `../graph-front-end/api/test_frontend_api.py`; verifies legacy/substrate payload compatibility, `REASONING_MODE=substrate`, and Dijkstra procedure firing through the real frontend API |

Backend reasoning tests passing: **113/113** (as of 2026-05-20, after the two double-dispatch regression tests landed: `test_double_dispatch_in_one_turn_reuses_object` and `test_repeated_invocation_across_turns_reuses_same_object`).
Frontend API integration tests passing: **3/3**.
Real-LLM frontend-pipeline debug runs: **2/2 passing** via `run_frontend_pipeline_debug.py` after configuring OpenCode profile-data access.

---

## Files now in scope

### Backend (`graph_final`)

```
reasoning/
|-- __init__.py
|-- audit_log.py
|-- budgets.py
|-- consolidation.py
|-- dispatcher.py
|-- reasoning_loop.py
|-- retrieval_boost.py
|-- schemas.py
|-- session_subgraph.py
|-- procedures/
|   |-- __init__.py
|   `-- verify_algorithm_preconditions.py
`-- tests/
    |-- __init__.py
    |-- test_audit_log.py
    |-- test_budgets.py
    |-- test_consolidation.py
    |-- test_dispatcher.py
    |-- test_reasoning_loop.py
    |-- test_retrieval_boost.py
    |-- test_schemas.py
    `-- test_session_subgraph.py
```

### Actual frontend integration (`../graph-front-end`)

```
api/
|-- frontend_api.py
`-- test_frontend_api.py

src/
|-- App.tsx
`-- types.ts

run_api.bat
README.md
```

---

## What 1.7, 1.8, and 1.9 added

### 1.7 - Reasoning loop orchestrator

`reasoning/reasoning_loop.py` is now the usable substrate entrypoint:
- retrieves anchors through failure-aware retrieval
- runs the main loop and dispatcher
- persists `subgraph.json` + `audit_log.jsonl`
- runs consolidation at session end
- returns both the persisted path and the in-memory `session_subgraph`
- produces a fallback answer if the model never emits an `<answer>` block

The Dijkstra seed-path test remains the load-bearing acceptance case and is passing.

### 1.8 - Frontend toggle

The real integration point is `../graph-front-end/api/frontend_api.py`:
- `REASONING_MODE=legacy` -> existing one-shot graph-context path
- `REASONING_MODE=substrate` -> `run_reasoning()` path

Implementation details:
- the frontend API now adapts `ReasoningResult` back into the JSON session/trace shape the React client already expects
- `run_api.bat` defaults `REASONING_MODE` to `legacy` if unset
- `/api/health` reports the active `reasoning_mode`
- the React status panel shows the active reasoning mode

### 1.9 - End-to-end integration coverage

`../graph-front-end/api/test_frontend_api.py` now covers the actual frontend API entrypoint:
- legacy mode payload shape still works
- substrate mode payload shape is compatible with the React client
- substrate mode persists artifacts to disk
- substrate trace records `INVOKE_PROCEDURE` on the Dijkstra case
- env-var selection works (`REASONING_MODE=substrate`)

This is now the main protection against the frontend drifting away from the substrate return type.

---

## Manual validation on the real frontend API

Ran real `POST /api/runs` calls through `../graph-front-end/api/frontend_api.py` in
`REASONING_MODE=substrate`. The substrate's own persistence dropped one session
subgraph per call under:

- `graph_final/data/session_subgraphs/sess_<hex>/subgraph.json`
- `graph_final/data/session_subgraphs/sess_<hex>/audit_log.jsonl`

13 such sessions exist on disk as of 2026-05-20. Inspect any with `cat <path>/subgraph.json`.

### Three manually reviewed samples

| Sample | Graph | Result | Notes |
|---|---|---|---|
| `dijkstra_negative_edge` | `cs4` | good answer | Procedure fired and the answer correctly recommends Bellman-Ford. However, the model invoked `VerifyAlgorithmPreconditions` twice in one run, creating two session objects for the same intent. |
| `light_vs_sound` | `physics1` | good answer | Clean conceptual answer. No procedure invocation. The returned substrate session graph is empty, which is structurally thin for UI inspection even though the answer is fine. |
| `heat_vs_temperature` | `physics1` | good answer | Clean conceptual answer. Same structural issue as above: no session nodes/edges unless a procedure fires. |

### Smoke battery (13 real substrate calls under data/session_subgraphs/)

Ran real substrate-mode calls across:

- `cs4`
- `algo3_binary_search`
- `algo2_floyd_cycle`
- `physics1`
- `math1_modexp`
- `sys1_webserver_latency`
- `merged_graph`

Result: **13/13 returned usable answers without API failure**.

Observed behavior from the smoke pass (cross-checked against persisted subgraphs):
- Dijkstra-style instance questions on `cs4`, `algo1_dp`, and `merged_graph` do trigger
  the Phase-1 substrate as intended (4/13 sessions had session_objects).
- Conceptual questions (physics1, math1_modexp, sys1_webserver_latency, algo3_binary_search,
  algo2_floyd_cycle) usually answer correctly but produce **empty session graphs** (9/13).
- Some conceptual questions take **3 reasoning iterations with no dispatch**, which is
  unnecessary latency.
- The Dijkstra case used to **double-dispatch** the same procedure in one run
  (verified in sess_862af617e699 and sess_a9e21797680e on 2026-05-19, two divergent
  session_objects for the same procedure). **Fixed 2026-05-20**: reasoning loop now
  dedupes within a turn and reuses existing SessionObjectNodes across turns. Regression
  test in `test_reasoning_loop::test_double_dispatch_in_one_turn_reuses_object`.

Current read: the substrate is functional, but still needs Phase-1 hardening before Phase 2.

### Real-LLM frontend-pipeline debug run

Added `run_frontend_pipeline_debug.py` to run the same worker used by the
frontend chat and save raw debug JSON. Direct mode imports
`../graph-front-end/api/frontend_api.py`, builds the same `RunRequest`, calls
`_run_graph_agent()`, captures emitted events, and stores the final payload plus
persisted substrate artifacts. HTTP mode can also POST to `/api/runs/stream`
when the API server is already running.

Latest real LLM run:

```
$ python run_frontend_pipeline_debug.py --mode direct --graph-id cs4 --k-anchors 8 --anchor-strategy topk --reasoning-mode substrate --max-iterations 3 --max-llm-calls 6 --model-timeout 300 --no-attach --opencode-dirs profile-data
```

Latest artifact:
`artifacts/frontend_pipeline_debug/frontend_pipeline_debug_20260521_005725.json`

Manual review:
- result: **ok**
- resolved frontend model command: `opencode run -`
- raw model calls captured: 6/6 ok
- event sequence: `ready`, `started`, `action_start`, `action_complete`, `tool_result`, `tool_result`, `final`
- final answer correctly marks Dijkstra unsafe because edge `b->c` has weight `-1`, and recommends Bellman-Ford
- persisted session: `data/session_subgraphs/sess_d43a4a428c9a`
- session graph: 16 nodes, 10 edges, 28 audit entries, diagnostics node present
- trace actions: `RETRIEVE_ANCHORS`, `MODEL_RUN`, `MODEL_RUN`, `INVOKE_PROCEDURE`, `INVOKE_PROCEDURE`, `FINALIZE_ANSWER`

Caveats from the run:
- The default attach URL (`http://localhost:6767`) fails when no OpenCode attach
  server is running (`Session not found`), so this successful run used
  `--no-attach`.
- Fully workspace-local OpenCode dirs have no saved credentials, so real LLM
  calls require `--opencode-dirs profile-data` or equivalent configured
  credentials.
- The run used the full 6/6 LLM-call budget and injected a
  `budget_warning_llm_call_iter_2` signal. The output is correct, but the seed
  procedure path is still latency-heavy.

---

## Current open items / non-blockers

These are not Phase-1 blockers anymore, but they are still worth tracking:

| Item | Where | Why it is still open |
|---|---|---|
| Audit log redundancy | `subgraph.json` embeds `audit_log`, and `audit_log.jsonl` is also written separately | Both are consistent at close time. Can be simplified later if JSONL becomes the sole source of truth. |
| Dotted-path keys containing literal `.` | dotted-path helpers in `audit_log.py` / `session_subgraph.py` | Current mutation grammar assumes path segments do not contain literal dots. |
| Read-journaling verbosity | `SessionSubgraphController.read_object()` | Useful for debugging, but can inflate logs on read-heavy sessions. Candidate Phase-2 toggle. |
| Env-var-only mode switch | `../graph-front-end/api/frontend_api.py` / `run_api.bat` | Matches the plan, but there is no runtime in-app toggle yet. |
| Substrate session visualization is adapted | `../graph-front-end/api/frontend_api.py` | The React client still expects the older session JSON shape, so substrate sessions are adapted into that shape for graph rendering while the raw substrate state is carried separately in the payload. |
| Empty session graphs on non-dispatch runs | `../graph-front-end/api/frontend_api.py` + substrate payload contract | Good answers are being returned, but the inspectable session graph is often empty unless a procedure fires. Tied to the seed corpus being thin (only one procedure exists). Address by growing the procedure corpus in Phase 2 / future work. |
| ~~Duplicate procedure dispatch~~ | ~~`reasoning/reasoning_loop.py` / `reasoning/dispatcher.py`~~ | **FIXED 2026-05-20**. Loop now tracks `procedure_id -> object_id` and dedupes invocations within a turn. Regression tests `test_double_dispatch_in_one_turn_reuses_object` and `test_repeated_invocation_across_turns_reuses_same_object` lock the behavior in. |
| ~~Extra reasoning iterations on conceptual questions~~ | ~~`reasoning/reasoning_loop.py`~~ | **FIXED 2026-05-20**. Directive (`_DIRECTIVE_INITIAL`) tightened: the model is now told explicitly that omitting `<answer>` is only allowed when it actually invoked a procedure on this turn. Conceptual questions should now produce an `<answer>` in iteration 0 and the loop's existing `if final_answer and not matches: break` will exit immediately. |
| Failure-pattern boost untested on real data | `reasoning/retrieval_boost.py` + production graphs | Unit-tested with mocks; no `failure_pattern` node exists in any production graph yet, so the 1.4× boost has never fired end-to-end. Pick this up when authoring failure patterns. |
| No substrate-vs-legacy answer-quality regression | `../graph-front-end/api/test_frontend_api.py` | Existing test checks payload shape, not answer quality. Tie this to the 6-cell eval from `_sft_eval.py` when that work is revisited. |

---

## Latest test outputs

### Backend reasoning suite

```
$ python -m unittest discover -s reasoning/tests -p "test_*.py"
..............................................
----------------------------------------------------------------------
Ran 113 tests in 35.653s

OK
```

(Test count moved from 111 → 113 on 2026-05-20 when the two double-dispatch regression tests landed.)

### Frontend API integration

```
$ cd ../graph-front-end
$ python -m unittest discover -s api -t . -p "test_*.py"
.Warning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN to enable higher rate limits and faster downloads.
Loading weights: 100%|##########| 103/103 [00:00<00:00, 5015.07it/s]
..
----------------------------------------------------------------------
Ran 3 tests in 38.996s

OK
```

### Frontend build

```
$ cd ../graph-front-end
$ npm run build
vite build completed successfully
```
