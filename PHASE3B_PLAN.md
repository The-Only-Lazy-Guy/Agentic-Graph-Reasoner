# Phase-3B implementation plan â€” macro extraction and tool authoring

**Status:** plan phase complete. Phase 3B-1 implementation progress is tracked in `PHASE3B_PROGRESS.md`.

**Parent docs:**
- `REASONING_ARCHITECTURE.md` â€” Â§2.4 composition, Â§2.5 multi-scale memory, Â§3 Phase-3 components
- `PHASE3A_PLAN.md` / `PHASE3A_PROGRESS.md` â€” meta-procedures (trigger-signal-react)
- `PHASE2_PLAN.md` / `PHASE2_PROGRESS.md` â€” composition substrate
- `PHASE1_PLAN.md` / `PHASE1_PROGRESS.md` â€” substrate foundations

**Related follow-up:**
- `PHASE3C_GRAPH_ACTIVATION_PLAN.md` - context-aware graph activation for direct-answer sessions where no procedure fires

**Date:** 2026-05-21

---

## 0. Executive summary

Phase 3A gave the system meta-cognition. The procedure library is still **entirely hand-authored**.

Phase 3B closes that gap in two mini-phases:

**Phase 3B-1 â€” Trace logging + macro observation (build first)**
Collect structured session traces. Scan for recurring call sequences. Answer: *are there patterns worth macro-ing, and how clean are they?* Do **not** install or validate procedures yet.

**Phase 3B-2 â€” Proposal + validation + staged installation (build after corpus exists)**
Add `PROPOSE_PROCEDURE` command, offline validator, JSON-first installer, and promote CLI. Only after 3B-1 has produced real trace data to avoid overfitting to synthetic tests.

The two creation paths:
```
bottom-up: correct traces â†’ macro candidate â†’ validate â†’ staged JSON â†’ promote
top-down:  model PROPOSE_PROCEDURE â†’ procedure_proposal node â†’ validate â†’ staged JSON â†’ promote
```

Both share the same validation + staging pipeline. Procedures are **stored as JSON first**, not generated Python. This keeps installation data-driven, inspectable, and reversible.

What is NOT in Phase 3B:
- Dynamic embeddings (deferred retrieval work, not 3B)
- Context-aware graph activation or provisional adjacent-node synthesis (Phase 3C)
- Consolidation / citation-decay lifecycle (Phase 2B/3D)
- Multi-session concurrency
- Cross-domain procedure transfer
- Auto-promoting without human `--promote` (ever, in v1)

---

## 1. Split: Phase 3B-1 vs 3B-2

### Phase 3B-1 goals (trace logging + observation)

1. Every `run_reasoning()` call appends a `SessionTrace` JSONL entry.
2. Trace contains: session id, graph id, question, answer, correctness, full ordered call tree, budget usage, signal ids fired.
3. A batch `scan` command detects recurring call sequences from clean traces and emits `MacroCandidate` JSON.
4. A `status` command reports candidate counts, precision, session coverage.
5. **No installation, no validation, no `PROPOSE_PROCEDURE`.** Observation only.

Done when: we have â‰¥10 real sessions logged and the scanner has run at least once against them.

### Phase 3B-2 goals (proposal + validation + installation)

1. `PROPOSE_PROCEDURE` parser in dispatcher.
2. `procedure_proposal` node type in schemas.
3. Offline validator with 5 steps + independent replay requirement.
4. JSON installer to `data/procedure_pool/_proposed/`.
5. `--promote` CLI to move staged JSON to live pool.
6. Dispatcher loads both hand-authored Python procedures and validated JSON procedures.
7. Rollback support (`--uninstall <name>`).
8. Proposal pass-rate and rejection-reason tracking.

---

## 2. File structure

Additions only. No existing files renamed or removed.

```
graph_final/
â”œâ”€â”€ reasoning/
â”‚   â”œâ”€â”€ trace_log.py                     â† NEW (3B-1): per-session structured trace recorder
â”‚   â”œâ”€â”€ macro_extractor.py               â† NEW (3B-1): frequency-based candidate detection
â”‚   â”œâ”€â”€ procedure_validator.py           â† NEW (3B-2): validation pass
â”‚   â”œâ”€â”€ procedure_pool_loader.py         â† NEW (3B-2): loads Python + JSON procedures uniformly
â”‚   â”œâ”€â”€ schemas.py                       â† MODIFIED (3B-2): add procedure_proposal node type
â”‚   â”œâ”€â”€ dispatcher.py                    â† MODIFIED (3B-2): parse PROPOSE_PROCEDURE command
â”‚   â”œâ”€â”€ reasoning_loop.py                â† MODIFIED (3B-1): emit trace log at session end
â”‚   â””â”€â”€ tests/
â”‚       â”œâ”€â”€ test_trace_log.py            â† NEW (3B-1)
â”‚       â”œâ”€â”€ test_macro_extractor.py      â† NEW (3B-1)
â”‚       â”œâ”€â”€ test_procedure_validator.py  â† NEW (3B-2)
â”‚       â”œâ”€â”€ test_procedure_pool_loader.pyâ† NEW (3B-2)
â”‚       â””â”€â”€ test_propose_procedure.py   â† NEW (3B-2)
â”œâ”€â”€ data/
â”‚   â”œâ”€â”€ trace_logs/                      â† NEW (3B-1): traces_YYYYMMDD.jsonl
â”‚   â”œâ”€â”€ macro_candidates/                â† NEW (3B-1): <fingerprint_hash>.json
â”‚   â”œâ”€â”€ procedure_proposals/             â† NEW (3B-2): <session_id>_<name>.json
â”‚   â””â”€â”€ procedure_pool/
â”‚       â”œâ”€â”€ _proposed/                   â† NEW (3B-2): staged JSON, not yet live
â”‚       â””â”€â”€ *.json                       â† NEW (3B-2): live validated JSON procedures
â”œâ”€â”€ run_macro_extraction.py              â† NEW (3B-1+): CLI batch job
â””â”€â”€ PHASE3B_PLAN.md                      â† this file
```

---

## 3. Session trace logging â€” `reasoning/trace_log.py` (3B-1)

### 3.1 Schema

```python
@dataclass
class ProcedureCall:
    call_id: str                     # uuid4 short, unique within session
    procedure_id: str
    procedure_name: str
    parent_call_id: Optional[str]    # None = top-level
    args_text: str
    mutations_applied: int
    error: Optional[str]
    elapsed_seconds: float

@dataclass
class SessionTrace:
    session_id: str
    graph_id: str
    domain: Optional[str]            # from graph metadata, if present
    question: str
    answer: str
    correct: Optional[bool]          # None until graded
    budget_exhausted: bool
    procedure_errors: int            # count of DispatchOutcomes with error != None
    contradiction_signals: int       # count of ERROR-severity signal nodes
    procedure_calls: list[ProcedureCall]
    budget_usage: dict
    signal_ids_fired: list[str]
    timestamp: str                   # ISO8601
```

### 3.2 Correctness signal

`correct` is filled by two paths:
- **Auto**: if the session's graph node carries an `expected_answer` field, match against the answer at session end.
- **Manual grading**: `run_macro_extraction.py --grade <jsonl>` back-fills `correct` from a graded file.

### 3.3 Clean-trace eligibility for macro extraction

Macro candidates are only built from **clean traces**:

```
correct == True
AND budget_exhausted == False
AND procedure_errors == 0
AND contradiction_signals == 0
```

A final answer can be correct even if the substrate was messy. We do not macro-extract from messy-but-recovered traces.

### 3.4 Storage

Append-only JSONL: `data/trace_logs/traces_<YYYYMMDD>.jsonl`. One line per session.

---

## 4. Macro candidate detection â€” `reasoning/macro_extractor.py` (3B-1)

### 4.1 Call sequence fingerprint

A call sequence fingerprint is the depth-first traversal string of procedure names:

```
"VerifyShortestPath > VerifyAlgorithmPreconditions, VerifyNonNegativeEdges"
```

Args are NOT part of the fingerprint â€” macros capture structural patterns, not specific inputs.

### 4.2 Metadata buckets (richer than fingerprint alone)

Candidates are grouped by fingerprint AND metadata bucket to prevent merging structurally similar but semantically different sequences:

```python
@dataclass
class CandidateBucket:
    fingerprint: str
    graph_domain: Optional[str]       # e.g. "computer_science", "math"
    top_level_procedure: str          # first procedure name in the sequence
    signal_free: bool                 # all source sessions had 0 signals fired
    budget_profile: str               # "low" (<= 2 calls), "mid" (3-4), "high" (5+)
```

A macro candidate requires â‰¥ MIN_SESSIONS sessions **within the same bucket**, not just the same fingerprint globally.

### 4.3 Thresholds

```python
MIN_SESSIONS  = 3     # distinct clean sessions in same bucket
MIN_PRECISION = 1.0   # all contributing sessions must have correct=True
```

### 4.4 Candidate schema

```python
@dataclass
class MacroCandidate:
    fingerprint: str
    fingerprint_hash: str            # short SHA256
    bucket: CandidateBucket
    procedure_names: list[str]
    source_session_ids: list[str]
    session_count: int
    precision: float
    proposed_name: str               # auto: "Macro_" + "_".join(procedure_names)
    status: str                      # "candidate" | "validated" | "rejected" | "installed"
```

Written to `data/macro_candidates/<fingerprint_hash>.json`.

---

## 5. `PROPOSE_PROCEDURE` command (3B-2)

### 5.1 Grammar

Inside the model's `<reasoning>` block:

```
PROPOSE_PROCEDURE
  name: VerifyBellmanFordSafety
  purpose: Verify whether Bellman-Ford can safely find shortest paths on a given graph.
  inputs:
    - algorithm_name: str
    - instance_description: str
  outputs:
    - safe_to_apply: bool
    - verdict: str
    - detected_issue: str
  body: |
    CALL DetectNegativeCycle WITH instance_description={instance_description}
    SET state.safe_to_apply = true
    SET state.verdict = "Bellman-Ford is safe â€” no negative cycle detected"
    DONE
  example_input: "directed graph (A->B weight 2), (B->C weight -5), (C->A weight 1)"
  example_expected_output: "safe_to_apply=false, detected_issue=negative cycle A->B->C->A"
END_PROPOSE_PROCEDURE
```

Body language: same `CALL` / `SET` / `ADD` / `DELETE` / `DONE` mutation grammar as existing procedure bodies.

### 5.2 Conservative trigger

The directive rider is added **only** when dispatch_outcomes is non-empty AND at least one sub-invocation fired:

> "If you noticed a reusable missing reasoning pattern â€” one that would reduce future call depth or improve correctness, has a clear worked example, and isn't already in the procedure catalog â€” you may propose it using `PROPOSE_PROCEDURE ... END_PROPOSE_PROCEDURE`. Only propose if you used at least one procedure this session and observed a gap. Proposals are recorded for offline review and are never installed immediately."

This prevents noisy proposals during direct-answer sessions (`VerifyBasicAddition`, `ExplainConceptClearly`, etc.).

### 5.3 On parse

The dispatcher:
1. Parses the block into a `ProcedureProposal` dataclass.
2. Creates a `procedure_proposal` node in the session subgraph.
3. Writes `data/procedure_proposals/<session_id>_<name>.json`.
4. Does **NOT** install or validate immediately.
5. Emits INFO acknowledgement next iteration: `"Procedure proposal 'X' recorded â€” pending offline validation."`

### 5.4 `procedure_proposal` node type

```python
NodeType = Literal[..., "signal", "procedure_proposal"]

# node fields:
{
  "node_type": "procedure_proposal",
  "proposal_id": str,
  "proposed_name": str,
  "status": "pending" | "validated" | "rejected",
  "session_id": str,
  "raw_block": str,
}
```

---

## 6. Validation pass â€” `reasoning/procedure_validator.py` (3B-2)

Both macro candidates and model proposals go through the same five-step gate.

### 6.1 Step 1 â€” Parse

Parse into `ProcedureNode`. Reject if:
- Name collision with existing non-deprecated procedure (case-insensitive).
- Required fields missing (`name`, `purpose`, at least one input, `body`).
- Body contains no `CALL` or `SET` commands.
- Name does not match `[A-Z][A-Za-z0-9]+` (PascalCase, no underscores).

### 6.2 Step 2 â€” Dry-run against stated example

Run the body through a stub LLM that echoes `example_expected_output`.
Assert: mutations â‰¥ 1, no `BudgetExhausted`, no error in `DispatchOutcome`.

Stub is deterministic and free. It proves the body parses and dispatches, not that it reasons correctly.

### 6.3 Step 3 â€” Independent replay (new â€” not in original plan)

**A proposal cannot pass validation using only its own stated example.**

For macro candidates: re-run the procedure body against each source session's actual question text (at least one beyond the stated example). Assert output fields declared in `outputs` are populated.

For model proposals: synthesize one adversarial variant of the stated example (flip one edge weight, change algorithm name, etc.) and assert the body produces a different output. If both cases produce identical output, the body is likely static and not actually reasoning.

Pass threshold: â‰¥ MIN_PRECISION of replay cases pass AND at least one independent case passes.

### 6.4 Step 4 â€” Budget regression

Run the body under current budget defaults. Assert: does not exhaust `llm_calls` or `fan_out` on the worked examples.

### 6.5 Step 5 â€” Name safety

PascalCase format. No existing name collision. No shadowing a built-in procedure node.

### 6.6 Validation result

```python
@dataclass
class ValidationResult:
    candidate_id: str
    passed: bool
    steps_passed: list[str]
    steps_failed: list[str]
    failure_reason: Optional[str]
    rejection_reason_code: Optional[str]   # "parse_error" | "no_commands" | "name_collision" |
                                           # "budget_regression" | "independent_replay_failed"
    procedure_node: Optional[ProcedureNode]  # only if passed=True
```

Rejection reason codes are tracked across all proposals. This tells us whether the model is bad at names, bad at body grammar, or bad at generalization.

---

## 7. JSON-first installation (3B-2)

### 7.1 Storage format

Validated procedures are stored as JSON `ProcedureNode` files, **not generated Python**:

```
data/procedure_pool/_proposed/proc_verify_bellman_ford_safety_v1.json  â† staged
data/procedure_pool/proc_verify_bellman_ford_safety_v1.json            â† live (after promote)
```

Rationale: no Python source modification, no `__init__.py` edits, easy to inspect, easy to roll back (delete the JSON file).

### 7.2 Procedure pool loader â€” `reasoning/procedure_pool_loader.py`

Loads all procedures uniformly from two sources:

```python
def load_procedure_pool() -> dict[str, ProcedureNode]:
    procedures = {}
    # Source 1: hand-authored Python factories
    for factory_fn in _PYTHON_FACTORIES:
        proc = factory_fn()
        procedures[proc.id] = proc
    # Source 2: validated JSON pool
    for json_path in Path("data/procedure_pool").glob("*.json"):
        proc = ProcedureNode.from_dict(json.loads(json_path.read_text()))
        procedures[proc.id] = proc
    return procedures
```

The dispatcher and reasoning loop call `load_procedure_pool()` on startup. Python and JSON procedures are **indistinguishable at runtime**.

Each procedure node carries a `source` field in provenance:
```
source: "hand_authored" | "validated_json" | "proposed"
```

### 7.3 Install flow

```
1. Load ValidationResult where passed=True
2. Serialize ProcedureNode to JSON
3. Set provenance.source = "validated_json"
4. Write to data/procedure_pool/_proposed/<id>.json
5. Append entry to data/install_log.jsonl
```

### 7.4 Promote

```
python run_macro_extraction.py --promote VerifyBellmanFordSafety
```

Moves `data/procedure_pool/_proposed/proc_xxx.json` â†’ `data/procedure_pool/proc_xxx.json`.
Procedure is live on next `load_procedure_pool()` call.

### 7.5 Rollback

```
python run_macro_extraction.py --uninstall VerifyBellmanFordSafety
```

Moves live JSON back to `_proposed/`. Procedure removed from pool on next load. No Python files modified in either direction.

---

## 8. CLI batch job â€” `run_macro_extraction.py`

```
Commands (3B-1):
  scan            Scan trace logs, emit macro candidates to data/macro_candidates/
  status          Print: candidate count, proposal count, installed count, pass rate

Commands (3B-2, added after 3B-1 is stable):
  validate        Run validation pass on all candidates/proposals with status=candidate/pending
  install         Write validated items to data/procedure_pool/_proposed/
  promote <name>  Move _proposed/<name>.json to live pool
  uninstall <name> Move live JSON back to _proposed/
  grade <file>    Back-fill correct=True/False from graded JSONL

Flags:
  --min-sessions N     (default 3)
  --min-precision F    (default 1.0)
  --dry-run            Print what would happen, write nothing
  --graph-id ID        Limit trace scan to one graph
  --domain D           Limit trace scan to one domain
```

---

## 9. Test plan

### 3B-1 unit tests
- `test_trace_log.py` â€” round-trip, append, schema validation, `correct` field states, clean-trace filter
- `test_macro_extractor.py` â€” fingerprint generation, bucket grouping, threshold, precision filter, clean-trace-only enforcement

### 3B-2 unit tests
- `test_procedure_validator.py` â€” each step passes/fails in isolation; independent replay rejection; rejection_reason_code coverage
- `test_procedure_pool_loader.py` â€” Python + JSON loaded uniformly; `source` field populated; `_proposed/` excluded from live pool
- `test_propose_procedure.py` â€” block parsing; `procedure_proposal` node in session subgraph; proposal on disk; no immediate dispatch

### Integration tests
- **Macro from synthetic corpus**: 3 synthetic `SessionTrace` JSONL entries, same fingerprint + bucket, all clean â†’ `scan` â†’ `validate` â†’ status=`validated`, `procedure_node` populated.
- **Model proposal happy path**: stub LLM emitting valid block â†’ parsed â†’ independent replay passes â†’ status=`validated` â†’ installer writes to `_proposed/`.
- **Proposal rejected â€” no commands in body**: validator Step 1 catches, `rejection_reason_code="no_commands"`.
- **Proposal rejected â€” independent replay fails**: body produces same output regardless of input â†’ Step 3 rejects, `rejection_reason_code="independent_replay_failed"`.
- **Proposal rejected â€” name collision**: existing procedure name â†’ Step 1 rejects.
- **Budget regression caught**: body exhausts `llm_calls` on example â†’ Step 4 rejects.
- **Promote + load**: after `--promote`, `load_procedure_pool()` returns the new procedure; dispatcher resolves it by name.
- **Rollback**: after `--uninstall`, procedure absent from `load_procedure_pool()`.

---

## 10. Phased build order

### 3B.1 â€” Session trace logging
Wire `run_reasoning()` to emit `SessionTrace` JSONL. Round-trip test. Inspect a real trace from cs4 smoke.

### 3B.2 â€” Macro extractor + fingerprint
Build `macro_extractor.py` with bucket grouping and clean-trace filter. Test with 3-session synthetic corpus.

### 3B.3 â€” CLI scan + status
`run_macro_extraction.py scan` and `status`. Integration test: synthetic corpus â†’ scan â†’ candidate file on disk.

**â€” Gate: run 3B.1â€“3B.3 on real sessions, inspect candidates. Only proceed to 3B.4+ when trace data validates the approach. â€”**

### 3B.4 â€” `procedure_proposal` node type
Add to `schemas.py`. Round-trip JSON test.

### 3B.5 â€” `PROPOSE_PROCEDURE` dispatcher parsing
Parse block, create node, write proposal JSON, emit INFO ack. Unit tests with canned stub outputs.

### 3B.6 â€” Procedure validator (5 steps + independent replay)
Build `procedure_validator.py`. Unit test each step. Test independent replay with adversarial case.

### 3B.7 â€” JSON-first installer + pool loader
`procedure_pool_loader.py` loads Python + JSON. Installer writes to `_proposed/`. `promote` / `uninstall` commands. Test rollback.

### 3B.8 â€” End-to-end smoke
Real LLM session: model proposes a procedure, `procedure_proposal` node appears in subgraph, `validate` + `promote` run manually, new procedure available in next session.

---

## 11. Acceptance criteria for Phase 3B complete

### From original plan (retained):
1. Every `run_reasoning()` appends a `SessionTrace` JSONL entry with full ordered call tree, budget usage, and signal ids.
2. `PROPOSE_PROCEDURE` blocks are parsed and land as `procedure_proposal` nodes in session subgraph and on disk. Not installed immediately.
3. Validator rejects: missing fields, no commands in body, name collision, budget regression.
4. Validator passes well-formed proposal and emits a `ProcedureNode`.
5. `scan` detects recurring call sequence from 3-session synthetic corpus and emits `MacroCandidate`.
6. Full pipeline end-to-end: traces â†’ scan â†’ validate â†’ install â†’ promote â†’ dispatcher resolves new name.
7. Installed procedures (JSON) are indistinguishable at runtime from hand-authored Python ones.
8. Human review gate preserved: installation writes to `_proposed/`, never live until `--promote`.
9. No regression â€” all 260 Phase 1/2A/3A tests still pass.
10. Phase 3B adds â‰¥ 25 new tests.

### Added from review:
11. Macro extraction ignores traces with `budget_exhausted=True`, `procedure_errors > 0`, or `contradiction_signals > 0`.
12. A proposal must pass at least one **independent** replay case (not only its own stated example) to be validated.
13. The installer supports rollback via `--uninstall <name>`.
14. `load_procedure_pool()` reports source (`hand_authored` | `validated_json`) for every loaded procedure.
15. The system tracks proposal pass rate and rejection reason codes across all proposals.
16. Macro candidates are grouped by metadata bucket (domain, signal-free, budget profile) â€” not fingerprint alone.

---

## 12. Decisions (resolved 2026-05-21)

| Decision | Resolved as | Notes |
|---|---|---|
| **`PROPOSE_PROCEDURE` trigger** | **Only after a procedure fired in the session** | Prevents noisy proposals during direct-answer sessions. |
| **`MIN_SESSIONS`** | **3** | Matches `M_PROMOTION` in `REASONING_ARCHITECTURE.md` Â§7. |
| **Validation dry-run** | **Stub + independent replay** | Stub for Step 2 (deterministic, free). Independent replay added as Step 3 so the model can't pass with a static body. |
| **Human review gate** | **Always require `--promote`** | Never auto-promote in v1. Revisit after 20+ promoted procedures. |
| **Correctness source** | **Both auto (graph metadata) + manual grading** | Auto where `expected_answer` exists; manual queue otherwise. |
| **Proposal body language** | **Same CALL/SET/ADD/DELETE grammar** | Keeps validator simple; proposals are machine-parseable. |
| **Installation format** | **JSON-first** (`data/procedure_pool/*.json`) | No Python source modification. Reversible. Inspectable. Python generation deferred until format is stable. |

---

## 13. What this is NOT

- **Not LLM-written Python.** The model proposes a structured spec (name, purpose, body in mutation grammar); the installer serializes it as JSON. No `exec`, no code generation.
- **Not self-modifying weights.** Procedure installation changes the tool library, not the model.
- **Not automatic deprecation.** Version chains and decay remain deferred (Phase 2B).
- **Not equivalence-checked macro synthesis.** The validator uses precision-over-corpus as a proxy for correctness. Formal equivalence is undecidable.
- **Not cross-session state sharing.** Proposals from session A aren't visible in session B until promoted to the live pool.

---

## 14. Prior art

- **Voyager** (Wang et al. 2023) â€” Minecraft agent that writes JavaScript skill functions during play. Closest precedent for top-down proposal path. Key difference: Voyager writes executable code; we write structured sub-prompts in a constrained grammar.
- **ToolFormer** (Schick et al. 2023) â€” model learns API calls by self-supervision. Similar spirit; different mechanism.
- **Program synthesis by example** â€” Step 3's IO-signature matching is a weak form. Used as sanity check, not proof.
- **The hypothesis_pool already in this project** â€” same citation-count promotion pattern, now applied to procedure proposals.

---

## 15. Effort estimate

| Sub-phase | Lines | Time |
|---|---|---|
| 3B.1 Session trace logging | ~200 | 0.5 day |
| 3B.2 Macro extractor + bucket grouping | ~250 | 0.75 day |
| 3B.3 CLI scan + status | ~150 | 0.25 day |
| 3B.4 `procedure_proposal` node type | ~50 | 0.1 day |
| 3B.5 `PROPOSE_PROCEDURE` parser | ~150 | 0.5 day |
| 3B.6 Procedure validator (5 steps + replay) | ~350 | 1.25 day |
| 3B.7 JSON installer + pool loader + rollback | ~200 | 0.5 day |
| 3B.8 End-to-end smoke | n/a | 0.25 day |

**Total: ~1350 lines, ~4.1 working days.**

3B-1 (3B.1â€“3B.3): ~600 lines, ~1.5 days â€” build and run first.
3B-2 (3B.4â€“3B.8): ~750 lines, ~2.6 days â€” only after real trace data validates the approach.

---

## 16. What we'll know after Phase 3B

- Are there recurring call sequences in real sessions worth macro-ing? (Answers whether bottom-up path is viable.)
- What fraction of model proposals pass validation? (Measures prompt quality for the proposal block.)
- Does the independent replay requirement catch bad proposals the stub alone would miss?
- Is the human review gate actually a bottleneck, or do most proposals get promoted quickly? (Informs whether to loosen the gate.)
- Does having more procedures in the pool cause useful new dispatch, or just noise? (Informs Phase 3C retrieval design.)

---

## 17. Lesson feeding Phase 3C

The merged-graph major test produced two useful signals:

- Dijkstra-style tasks can create clean procedural traces and a real macro candidate.
- IOI-style algorithm-design tasks can be solved correctly with no procedure calls at all.

That means graph usefulness cannot depend only on procedure dispatch. Phase 3B should continue measuring procedural recurrence, while Phase 3C adds a separate context-aware activation path: graph nodes emit typed signals into a task frame even when the model answers directly.

The design boundary is intentional:

- 3B asks: "Which recurring procedure call sequences are worth compressing?"
- 3C asks: "How can nearby graph nodes actively shape the answer when no procedure should fire?"

---

## 18. Status / next move

This doc is the **Phase 3B design and implementation contract**. Phase 3B-1 is implemented; current progress and corpus gate status live in `PHASE3B_PROGRESS.md`.

**Next step:** continue 3B-1 real-session collection until the corpus gate is satisfied. Do not start 3B-2 proposal, validation, or installation work until the trace data is large enough to justify it. Do not start Phase 3C code until its plan is reviewed against the 3B evidence.

Same discipline as all prior phases: design first, code second, inspect per sub-phase. Do not start 3B-2 components until 3B.1â€“3B.3 have run on real sessions.

