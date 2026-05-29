# Phase-3B implementation progress

Status as of 2026-05-21. Plan: `PHASE3B_PLAN.md`.

Phase 3B is split into two gated mini-phases. This progress file covers only
3B-1: trace logging plus macro observation. No 3B-2 proposal, validation,
installation, promotion, rollback, or JSON procedure-pool loading code has
been started.

---

## Sub-phase status

| # | Sub-phase | Status | Tests added | Notes |
|---|---|---:|---:|---|
| 3B.1 | Session trace logging | done | 29 trace tests total | `run_reasoning()` appends `SessionTrace` JSONL entries under `data/trace_logs/` in production. Test callers can route traces to a temp root. |
| 3B.2 | Macro extractor + fingerprint | done | 12 | `reasoning/macro_extractor.py` groups only clean traces by fingerprint + domain + top-level procedure + signal-free + budget profile. |
| 3B.3 | CLI scan + status + grade | done | covered in extractor tests | `run_macro_extraction.py scan`, `status`, and `grade` support 3B-1 observation and manual curation. Unsupported 3B-2 commands are intentionally absent. |

Full backend suite after 3B-1: **304/304 passing** with 1 documented expected
failure inherited from Phase 3A.

---

## Files added or modified

```
reasoning/
├── trace_log.py                  modified/active - SessionTrace schema + JSONL logger
├── macro_extractor.py            new - clean-trace macro candidate detection
├── reasoning_loop.py             modified - trace emission at session end
├── dispatcher.py                 modified - DispatchOutcome elapsed_seconds
└── tests/
    ├── test_trace_log.py         expanded - round-trip + run_reasoning trace emission
    └── test_macro_extractor.py   new - bucket grouping, clean filter, scan/status CLI

run_macro_extraction.py           new - scan/status CLI
PHASE3B_PROGRESS.md               new - this progress log
PHASE3B_PLAN.md                   updated status pointer
PHASE3C_GRAPH_ACTIVATION_PLAN.md  new - follow-up plan for context-aware graph activation
data/trace_grades/                new - reviewed trace grading JSONL files
```

---

## Real cs4 smoke

Ran the frontend pipeline in substrate mode against `cs4` twice after the trace
hook landed. Latest artifact:

`artifacts/frontend_pipeline_debug/frontend_pipeline_debug_20260521_071624.json`

Latest persisted session:

`data/session_subgraphs/sess_c26775a6948e`

Latest trace entry:

`data/trace_logs/traces_20260521.jsonl`

Observed trace shape:

| Field | Value |
|---|---|
| graph_id | `cs4` |
| domain | `computer_science` |
| budget_exhausted | `false` |
| procedure_errors | `0` |
| contradiction_signals | `0` |
| signal_ids_fired | `[]` |
| budget llm_calls | `5/6` |
| procedure call tree | `VerifyShortestPath > VerifyAlgorithmPreconditions, VerifyNonNegativeEdges` |
| correct | `null` |

The answer was correct by inspection, but `correct` remains `null` because
`graphs/cs4.json` does not provide an unambiguous `expected_answer`. That is
intentional: ungraded traces are not clean and cannot seed macro candidates.

Running `python run_macro_extraction.py scan --dry-run --json` after the two
real cs4 runs reported:

```json
{
  "candidate_count": 0,
  "clean_traces_seen": 0,
  "dry_run": true,
  "traces_seen": 2,
  "written_paths": []
}
```

This is the expected gate state before manual grading or expected-answer
metadata exists.

---

## Real merged_graph smoke

Ran the same Dijkstra/precondition smoke against `graphs/merged_graph.json`.
Latest artifact:

`artifacts/frontend_pipeline_debug/frontend_pipeline_debug_20260521_080338.json`

Latest persisted session:

`data/session_subgraphs/sess_85cd32718e26`

Trace entry appended to:

`data/trace_logs/traces_20260521.jsonl`

Observed trace shape:

| Field | Value |
|---|---|
| graph_id | `merged_graph` |
| domain | `computer_science` |
| budget_exhausted | `false` |
| procedure_errors | `0` |
| contradiction_signals | `0` |
| signal_ids_fired | `[]` |
| budget llm_calls | `5/6` |
| procedure call tree | `VerifyShortestPath > VerifyAlgorithmPreconditions, VerifyNonNegativeEdges` |
| correct | `null` |

This run is a stronger macro exemplar than the earlier cs4-only trace because
both child procedures emitted structured mutations:

- `VerifyAlgorithmPreconditions`: checked `non_negative_edge_weights`,
  `single_source`, and `no_negative_cycles`; violated
  `non_negative_edge_weights`.
- `VerifyNonNegativeEdges`: checked `a->b`, `b->c`, `a->c`; violated `b->c`.

It is still not eligible for extraction because `correct` is ungraded.

Additional merged_graph variants run on 2026-05-21:

| Session | Case | Verdict | Trace quality |
|---|---|---|---|
| `sess_9eab3d3dca43` | all non-negative edges `s->a=2`, `a->b=4`, `s->b=10` | safe | good: all 3 procedures mutated useful state |
| `sess_ff1961e85ff2` | negative edge `y->z=-2` | unsafe | good: all 3 procedures mutated useful state |

Together with `sess_85cd32718e26`, these form a 3-session real corpus with
the same fingerprint:

`VerifyShortestPath > VerifyAlgorithmPreconditions, VerifyNonNegativeEdges`

Manual grading file:

`data/trace_grades/phase3b_seed_dijkstra_20260521.jsonl`

Command:

```powershell
python run_macro_extraction.py grade data\trace_grades\phase3b_seed_dijkstra_20260521.jsonl --json
```

Result: 3 valid rows, 3 live trace rows updated, no missing session ids.

Scanner result after grading:

```json
{
  "candidate_count": 1,
  "clean_traces_seen": 3,
  "traces_seen": 3
}
```

Candidate emitted:

`data/macro_candidates/3bf0dcd972ff.json`

Candidate summary:

| Field | Value |
|---|---|
| fingerprint | `VerifyShortestPath > VerifyAlgorithmPreconditions, VerifyNonNegativeEdges` |
| bucket | `computer_science`, signal-free, `mid` budget profile |
| source sessions | 3 |
| precision | `1.0` |
| proposed_name | `Macro_VerifyShortestPath_VerifyAlgorithmPreconditions_VerifyNonNegativeEdges` |

Implementation hardening from this test:

- `TraceLogger.read_all()` now reads with `utf-8-sig` so JSONL files with a
  UTF-8 BOM on the first line are not silently undercounted.
- `run_macro_extraction.py grade <file>` applies reviewed correctness labels
  to live trace JSONL while preserving the trace schema.

---

## Current gate state

3B-1 code is implemented and tested, but the corpus gate is not satisfied yet.
The plan requires at least 10 real logged sessions and at least one scanner run
against them before considering 3B-2. Current real trace count is 7, clean
trace count is 5, and one live macro candidate has been emitted. The corpus
gate is partially satisfied; we still need at least 3 more real sessions before
3B-2 should start.

---

## Major test slice: Dijkstra + IOI-style control

Run on 2026-05-21 against `merged_graph`.

### Dijkstra strengthening case

Artifact:

`artifacts/frontend_pipeline_debug/frontend_pipeline_debug_20260521_085655.json`

Session:

`data/session_subgraphs/sess_c883baf4f0c9`

Prompt shape: safe Dijkstra case with edges `p->q=1`, `q->r=2`,
`r->t=3`, `p->t=12`; asks for applicability and shortest distance to `t`.

Result:

- Correct answer: Dijkstra safe, shortest distance `p->t = 6` via
  `p->q->r->t`.
- Procedure tree: `VerifyShortestPath > VerifyAlgorithmPreconditions,
  VerifyNonNegativeEdges`.
- All three procedures mutated useful state.
- No budget exhaustion, procedure errors, or contradiction signals.

### IOI-style difficult control

Artifact:

`artifacts/frontend_pipeline_debug/frontend_pipeline_debug_20260521_090019.json`

Session:

`data/session_subgraphs/sess_aaddb95c7847`

Prompt shape: dynamic maximum subarray sum under point updates, `n,q <= 200000`,
negative values allowed, non-empty subarray, C++17 required.

Result:

- Correct segment-tree solution using `sum`, `max_prefix`, `max_suffix`,
  `max_sub`.
- Handles all-negative arrays by initializing leaves to the element value
  rather than allowing empty subarrays.
- Uses `long long`; complexity is `O(n)` build, `O(log n)` update, `O(1)`
  whole-array query from root.
- No procedures fired, as expected; one benign `no_dispatch_after_iter_2`
  info signal fired. This is a useful direct-answer control and did not create
  a macro candidate.

Design consequence:

- The graph cannot be useful only through procedure dispatch. This control
  motivates Phase 3C: context-aware graph activation should let nearby nodes
  emit constraints, pitfalls, examples, and provisional gap/bridge nodes into a
  task frame even when the final answer is produced directly.
- This is intentionally not being added to Phase 3B. Macro extraction remains
  scoped to clean procedure traces.

Manual grading file:

`data/trace_grades/phase3b_major_test_20260521.jsonl`

Status after grading and scan:

```text
traces=7 clean=5 candidates=1 coverage=4 avg_precision=1.000
candidate_status: candidate=1
```

The live macro candidate remains:

`data/macro_candidates/3bf0dcd972ff.json`

Its `session_count` is now 4, sourced only from Dijkstra/precondition
composition traces. The IOI control did not pollute macro extraction.

Full backend suite after this slice: **304/304 passing** with the inherited
expected failure.
