# Phase-3A implementation progress

Status as of 2026-05-20. Plan: `PHASE3A_PLAN.md`. Architecture: `REASONING_ARCHITECTURE.md`.

Phase 3A is the trigger-signal-react meta-procedure layer: predicates observe substrate state in pure Python, emit signals that flow into the next prompt's `# System signals` section, and the model reacts via its normal reasoning — no extra LLM call for meta-cognition.

---

## Sub-phase status

| # | Sub-phase | Status | Tests added | Notes |
|---|---|---|---|---|
| 3.1 | Signal + MetaProcedure schemas | ✅ done | 22 | `signals.py` (Signal + render) + `meta.py` (MetaProcedure + MetaContext + MetaPool). `NodeType` extended with `"signal"`. |
| 3.2 | MetaPool orchestration | ✅ done | (covered by 3.1) | Folded into `meta.py` alongside the schemas. |
| 3.3 | Reasoning-loop hooks + signal injection | ✅ done | 4 | Three hook points (`pre_iter`, `post_dispatch`, `end_of_session`) wired into `run_reasoning`. Prompt builder takes `active_signals` arg, renders `# System signals` section, appends directive rider only when signals are present. |
| 3.4 | Signal persistence | ✅ done as side-effect | (covered by 3.3) | Signals persist into `session.subgraph.nodes` as `node_type="signal"` via `Signal.to_node()`. |
| 3.5 | Five real meta-procedures | ✅ done | 27 | `reasoning/meta_procedures/` package with `CycleDetector`, `BudgetWarner`, `ContradictionDetector`, `DispatchMissNudge`, `NoDispatchAfterThreshold`. `build_default_meta_pool()` factory. |
| 3.5+ | Adversarial + real-session tests | ✅ done | 13 (1 expectedFailure) | Realistic GLM-style outputs + replay of 2 persisted real sessions through the pool. Validates conservative-predicate claim against production data. |
| 3.6 | Cycle-detection end-to-end test | ✅ done | 1 | Scripted LLM invokes `VerifyNonNegativeEdges` 3× with same args → `CycleDetector` fires WARN sticky → next iter's main prompt contains the signal. |
| 3.7 | Contradiction sticky-signal lifecycle test | ✅ done | 2 | Custom test procedures (`ChildA`, `ChildB`) with whitelisted `safe_to_apply` field — opposing booleans fire ERROR sticky. Inverse test (siblings agreeing) confirms no false positive. |
| 3.5+ wire default pool | ✅ done | — | `run_reasoning()` defaults to `build_default_meta_pool()` instead of empty pool. Production now gets meta-procedures automatically. |
| 3.5++ flatten dispatch_outcomes | ✅ done | — | `flatten_dispatch_outcomes()` helper added to dispatcher. MetaContext receives flattened tree so predicates see sub-procedures invoked via `CALL`, not just top-level. CycleDetector + ContradictionDetector were silently broken before this fix; tests caught it. |
| 3.5++ CycleDetector sticky | ✅ done | — | Was `sticky=False` initially → signal landed in subgraph but never reached the model. Changed to `sticky=True` so the model actually sees the cycle warning next iteration and can react. |
| 3.8 | Manual real-LLM smoke (user-side) | pending | — | Through front-end with `REASONING_MODE=substrate` + default meta-pool now plugged in. |

**Total tests passing: 233/233** (1 documented `@expectedFailure` for a known accepted FP).

---

## What review passes caught (the real value)

### Review pass 1 — initial 3.3 inspection
Found 5 sharp edges:
1. Unbounded carrier-sticky growth
2. Render order showing OLDEST signals when newest are more actionable
3. Mutable `previous_signals` exposed on MetaContext
4. Audit-log invisibility for direct subgraph mutation (signals path)
5. Stream/persistence dedup inconsistency (in-memory had duplicates, persistence didn't)

Fixed in the hardening pass:
- `MAX_CARRIER_STICKY=20` with drop-oldest semantics
- `render_signals_block` sorts by `(severity_rank, -emitted_at_step)` — ERROR first, then newest-within-severity
- `previous_signals` passed via `list(...)` defensive copy
- Docstring note on the signal-persistence direct-mutation pattern
- Stream-level id dedup in `MetaPool.run_hook`

### Review pass 2 — probes targeting specific failure modes
Probe 1: pre_iter signals **lost on budget exhaustion before LLM call** — real bug, signal in memory but never persisted. **Fixed**: persist pre_iter signals immediately after emission, not bundled with post_dispatch. Regression test `test_pre_iter_signals_persist_even_on_budget_exhaust`.

Probe 2: same-id re-emissions had 3-in-stream but 1-in-persistence — **inconsistency**. **Fixed**: `MetaPool` tracks `_seen_signal_ids` and deduplicates the in-memory stream too. Regression test `test_signal_stream_dedupes_same_id_re_emissions`.

Probe 3: same-hook isolation (MP B doesn't see MP A's signals on the same tick) — by design, but undocumented. **Documented** in `MetaContext` docstring.

Probe 4: JSON persistence round-trip with signal nodes — verified working.

### Review pass 3 — adversarial + real-session replay
Tested DispatchMissNudge against GLM-style phrasings:
- Negation (`I won't apply X`): regex correctly ignores. **Not a FP.**
- Past tense (`I applied X earlier`): regex correctly ignores. **Not a FP.**
- Hypothetical (`If we apply X to Y`): regex matches — **real FP**, accepted for v1 (signal text is non-prescriptive). Marked `@expectedFailure`.
- Free-text mention (`The X procedure is useful`): correctly ignored.

Replayed 2 persisted real sessions through the meta-pool:
- `sess_e1023f5801ca` (real Dijkstra composer firing correctly): **0 signals.** Pool correctly identifies clean reasoning.
- `sess_7ddd8fc9d588` (real coding question, model produced prose without `<answer>` tags): exactly **1 `no_dispatch_stale` signal**. Predicate correctly diagnoses the format-compliance issue.

Conservative-predicate claim validated against actual GLM output.

---

## File inventory

```
reasoning/
├── signals.py                                    NEW — Signal dataclass + render_signals_block
├── meta.py                                       NEW — MetaProcedure + MetaContext + MetaPool
├── reasoning_loop.py                             MODIFIED — 3 hook points, signal injection, carrier cap
├── schemas.py                                    MODIFIED — "signal" added to NodeType
├── meta_procedures/                              NEW package
│   ├── __init__.py                               build_default_meta_pool() factory
│   ├── cycle_detector.py                         WARN: same (proc, args) ≥3 times
│   ├── budget_warner.py                          INFO: any budget axis ≥75%
│   ├── contradiction_detector.py                 ERROR sticky: sibling SOs disagreeing on whitelisted bool
│   ├── dispatch_miss_nudge.py                    INFO: model mentions unknown procedure name
│   └── no_dispatch_after_threshold.py            INFO: no dispatch by iter 2
└── tests/
    ├── test_signals.py                           NEW — round-trip, severity, render
    ├── test_meta.py                              NEW — MetaPool orchestration + dedup + fault tolerance
    ├── test_signal_injection.py                  NEW — end-to-end through reasoning_loop
    ├── test_meta_procedures.py                   NEW — each MP in isolation
    └── test_meta_procedures_adversarial.py       NEW — false-positive resistance + real-session replay
```

Plan said ~970 lines + ~30 tests. Actual: ~1100 production lines + 66 tests (incl. 13 adversarial). Slightly over because the adversarial pass added value the original plan didn't anticipate.

---

## Known limitations (accepted for v1)

| Limitation | Mitigation |
|---|---|
| Hypothetical phrasing ("If we apply X to Y") fires DispatchMissNudge false positive | Signal text is non-prescriptive ("if you meant to invoke...") — model can ignore. `@expectedFailure` test keeps the issue visible. |
| MPs within one hook tick don't see each other's signals | By design (context snapshot per tick). Cross-MP chaining works on NEXT tick. Documented in MetaContext docstring. |
| Carrier-sticky cap of 20 means very long sessions could drop early errors | Persistence captures every signal regardless of cap. The cap protects the prompt budget, not the audit trail. |
| NoDispatchAfterThreshold fires on off-domain questions where no procedure applies | Signal text covers both options ("invoke a procedure OR finalize the answer directly"). Real-session replay confirmed the message reads correctly even on coding-question case. |

---

## What's still pending

- **3.8** — manual real-LLM smoke through `REASONING_MODE=substrate` with the meta-pool now active by default. Validates Phase-3A end-to-end on actual GLM output. **User-side validation.**

Sub-phase 3.8 has been partially run: cs4 Dijkstra negative-edge smoke (2026-05-21) returned a correct user-facing answer but exposed three control-hygiene bugs in finalization. All three were fixed in Phase-3A.1 hardening (below).

---

## Phase-3A.1 finalization-hardening pass (2026-05-21)

Triggered by sub-phase 3.8 smoke on cs4 (`frontend_pipeline_debug_20260521_005725.json`). The run produced the correct verdict (Dijkstra unsafe → Bellman-Ford) and correct composition (VerifyShortestPath → VerifyAlgorithmPreconditions + VerifyNonNegativeEdges via CALL), but the finalization turn re-invoked `VerifyNonNegativeEdges` at the top level on prose args (`"the instance and then use the VerifyShortestPath result to compose the answer."`), exhausting the budget (6/6) and producing a junk session_object.

Four fixes landed:

| # | Fix | Location | Description |
|---|---|---|---|
| 1 | Finalization-mode dispatch gate | `reasoning_loop.run_reasoning` | Skip `dispatcher.scan` when `dispatch_outcomes` is already non-empty AND this turn's output contains `<answer>`. Once the model has finalized, any "I'll apply X" prose is incidental, not a real intent. |
| 2 | Drop procedures section in finalization | `reasoning_loop._build_prompt` | When `dispatch_outcomes` is non-empty, omit the `# Available procedures` catalog entirely and use a finalize-only directive that explicitly forbids the phrasings the model is observed using ("I'll apply", "using the", "invoke"). |
| 3 | Pre-LLM args validation | `dispatcher.Dispatcher.validate_args` | Reject obviously-malformed args BEFORE creating a session object or consuming an LLM call. Currently catches: args referencing a DIFFERENT live procedure by name (meta-prose), and empty args on a procedure that declares required inputs. |
| 4 | Render sub_outcomes in finalization prompt | `reasoning_loop._render_dispatch_results` | Walk `sub_outcomes` recursively so child procedures' summaries AND parsed state appear (indented) in the finalize prompt. Without this, the cs4 run only saw the composer's summary in the prompt and re-invoked the leaf to "fill the gap." |

Test additions (24 in `test_finalization_gate.py`):
- 6 finalization-prose variants vs Fix 1's dispatcher gate
- 3 preservation tests (iter-0 dispatch, chained dispatch, premature `<answer>`)
- 4 prompt-builder shape tests (no catalog in finalization, directive content, iter-0 still has catalog, dispatch-results survives)
- 5 sub-outcome rendering tests (single child, multi-child, indentation, list truncation, SET-only state)
- 6 args-validation tests across multiple bad-args variants
- 2 invoke-level tests (no LLM call on rejection, no budget consumption on rejection)

Each fix uses MULTIPLE prompt phrasing variants to defend against the
variation surface real LLMs produce. The exact prose from the cs4 run
is the first variant in each variant list; additional phrasings cover
"I will", "invoke", "using the", mid-sentence, and bare-imperative forms.

### Phase-3A.2 procedure-context propagation (2026-05-21)

Follow-up forced-procedure smoke after Phase-3A.1 validated the finalization
gate but exposed a different substrate weakness: top-level dispatch args can
be deictic (for example, "this directed weighted instance") instead of
restating the concrete edge list. The composer then passed that underspecified
text to child procedures, so `VerifyNonNegativeEdges` could create a real child
session_object but leave `state.violating_edges=[]` even though the original
question contained `(b->c, weight -1)`.

Fix: every procedure sub-prompt now includes the original user question/context
before `Invocation args`. This is a general context-propagation fix, not a
cs4-specific heuristic: child procedures can resolve references like "this
instance" without depending on the main reasoner to restate all facts in the
dispatch span.

Validation smoke `frontend_pipeline_debug_20260521_025810.json`:

| Diagnostic | Value |
|---|---|
| result | ok |
| elapsed | 83.0s |
| LLM calls | 5/6 |
| trace | `RETRIEVE_ANCHORS`, `MODEL_RUN`, `MODEL_RUN`, `INVOKE_PROCEDURE`, `FINALIZE_ANSWER` |
| session_objects | 3: `VerifyShortestPath`, `VerifyAlgorithmPreconditions`, `VerifyNonNegativeEdges` |
| `sub_invocation_of` edges | 2 |
| `VerifyNonNegativeEdges.state.violating_edges` | `["b->c"]` |
| budget_exhausted | false |
| junk extra top-level dispatch | none |

Test hardening: existing dispatcher invoke coverage now asserts that procedure
sub-prompts carry the original question context.

### Phase-3A.3 canonical invocation resolver (2026-05-21)

Focused stabilization patch for argument fidelity. Before any procedure LLM call,
`Dispatcher.invoke()` now runs a deterministic `ProcedureInvocationResolver`:

```text
raw model invocation + original user question + procedure signature
  -> canonical procedure args
```

Current resolver scope is intentionally narrow:
- `algorithm_name`: extract known algorithm names from raw args or original question.
- `instance_description`: prefer concrete edge/weight args from raw text; otherwise
  recover concrete edge/weight data from the original question.
- If required args cannot be resolved, reject before creating a session object or
  consuming a procedure LLM call with `unresolved_procedure_args` dispatch error.

This directly prevents the bad `025332` failure class where `VerifyShortestPath`
and its children received vague args such as `"this directed weighted instance"`
and produced empty/deferred state.

Regression coverage added: exactly 3 tests in `test_reasoning_loop.py`:
- vague args recover concrete graph edges from the original question
- concrete args pass through and produce the same child state
- no-procedure direct answer still exits in one LLM call with no dispatch

Validation reruns:

| Artifact | Case | Result |
|---|---|---|
| `frontend_pipeline_debug_20260521_034707.json` | default cs4 direct task | direct answer, 1/6 LLM calls, 0 session_objects |
| `frontend_pipeline_debug_20260521_034909.json` | forced `VerifyShortestPath` / vague-style model args | 3 session_objects, `checked_edges=[a->b,b->c,a->c]`, `violating_edges=[b->c]`, 5/6 calls |
| `frontend_pipeline_debug_20260521_035106.json` | concrete args | 3 session_objects, `checked_edges=[a->b,b->c,a->c]`, `violating_edges=[b->c]`, 5/6 calls |

Full backend suite after patch: **260/260 passing** with 1 documented expected
failure for the unrelated hypothetical DispatchMissNudge false positive.

---

## Test totals over phases

| Phase | Tests added | Cumulative |
|---|---|---|
| Phase 1 | 113 | 113 |
| Phase 2A | 47 | 160 |
| Phase 3A (3.1–3.5) | 57 | 217 |
| Phase 3A adversarial | 13 | 230 |
| Phase 3A end-to-end (3.6–3.7) | 3 | 233 |
| Phase 3A.1 finalization hardening | 24 | 257 |
| Phase 3A.3 canonical invocation resolver | 3 | 260 |

All passing (1 documented expected-failure for accepted FP).

---

## Bugs caught across review passes — total counts

| Where caught | Bug | Severity |
|---|---|---|
| Review pass 2 (probes) | pre_iter signals lost on budget exhaust | data loss in persistence |
| Review pass 2 (probes) | stream/persistence dedup inconsistency | confusing diagnostics |
| Manual inspection of cap test | render showed OLDEST 5 signals when newest are more actionable | semantic bug |
| 3.6 / 3.7 (end-to-end) | predicates only saw top-level outcomes; sub-CALL invocations invisible | CycleDetector + ContradictionDetector silently broken on composition |
| 3.6 design check | CycleDetector `sticky=False` → signal never reached the model | the WHOLE POINT of signal injection failed |
| 3.8 (real-LLM cs4) | finalization scanned output for new dispatch even after `<answer>` was emitted | budget exhaustion + junk session_object on prose-args invocation |
| 3.8 (real-LLM cs4) | finalize prompt still listed `# Available procedures` and didn't forbid invocation phrasings | model tempted into re-invoking after results already produced |
| 3.8 (real-LLM cs4) | `_render_dispatch_results` only walked top-level outcomes; sub-CALL child state invisible to finalization model | model re-invoked the leaf to "see its state" — directly caused the bad second dispatch |
| 3.8 (real-LLM cs4) | dispatcher consumed an LLM call on args_text that was meta-prose ("the instance and then use the X result") | wasted budget axis on input that could never bind |

Pattern: stubs and unit tests proved the components worked; **probes targeting realistic edges + end-to-end integration caught real bugs.** Each round of review found something the previous round missed. Test counts alone weren't enough.
