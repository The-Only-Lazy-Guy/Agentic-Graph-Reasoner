# Phase 3E â€” Reasoning Substrate v2

**Status:** 3E-1 implemented; 3E-2 implemented behind `enable_substrate_v2`;
3E-3 deterministic guardrails plus final-answer coverage cleanup implemented.
Default `run_reasoning()` behavior remains unchanged when the flag is false.
**Replaces:** the focus/plan/execute/check/revise/finalize mode pipeline as the
default control flow. 3D-1 deterministic substrate is preserved and reused.
**Parent docs:**
- `REASONING_ARCHITECTURE.md`
- `PHASE3C_GRAPH_ACTIVATION_PLAN.md` / `PHASE3C_PROGRESS.md`
- `PHASE3D_ADAPTIVE_PLANNING_PLAN.md` / `PHASE3D_PROGRESS.md`

**Date:** 2026-05-21

---

## Implementation status

### 3E-1 complete

Files:

- `reasoning/substrate_v2.py`
- `reasoning/tests/test_substrate_v2.py`

Implemented:

1. `SignalNode`, `ReasoningStep`, `StateDelta`, `DeltaTransaction`,
   `MissingInfo`, and `StepContextPacket`.
2. Read-only projection over existing session node types:
   `activation_signal`, `task_frame_item`, `session_gap`, `session_bridge`,
   `plan_node`, `plan_check`, `signal`, evidence/question/answer.
3. `packet_from_task_frame()` as the 3E view over Phase 3C `GraphTaskFrame`.
4. Cross-session gap canonicalization with
   `hash(expected_shape, normalized_question)`.
5. `regex_fallback` support on skimmed deltas through `StateDelta.produced_by`.
6. Round-trip tests and projection tests against the synthetic IOI/Dijkstra
   Phase 3D session subgraphs.

Verified:

```powershell
python -m unittest reasoning.tests.test_substrate_v2
python -m unittest discover reasoning/tests
```

Full reasoning suite result: 308 tests OK, 1 expected failure.

### 3E-2a complete

Standalone pieces now implemented in `reasoning/substrate_v2.py`:

1. Strict `STEP_RESULT` parser plus regex-fallback `DeltaTransaction`.
2. `CheckerRegistry` with `generic_step_format`, `algorithm_design`,
   `dynamic_max_subarray`, and `shortest_path_safety` plugins.
3. `PacketRenderCache`.
4. `run_fast_step_loop()` with recursive child-step handling, child-depth and
   total-step caps.
5. Stubbed IOI and Dijkstra fast-loop tests.

### 3E-2b complete

Implemented:

1. `ReasoningRequest.enable_substrate_v2`, default `False`.
2. Flagged `run_reasoning()` route through `run_fast_step_loop()`.
3. Initial controller-produced v2 signals from the question and retrieved
   anchors.
4. V2 trace persistence into the session subgraph:
   `substrate_v2_step`, `substrate_v2_delta`, `substrate_v2_check`, and
   `substrate_v2_signal`.
5. Replayed parent resumes persist as distinct step occurrences instead of
   collapsing onto the same node id.
6. Integration tests for Dijkstra one-call v2 resolution and IOI recursive
   parent-child-parent flow.

Default behavior is still the existing procedure/meta loop unless
`enable_substrate_v2=True`.

### 3E-3 deterministic guardrails complete

Implemented:

1. `factual_recall` checker plugin for broad non-algorithm tasks. It is
   deliberately conservative: it does not judge truth, and only raises soft
   evidence-anchoring violations.
2. Malformed-delta fuzz coverage for missing blocks, invalid statuses,
   incomplete `need_info`, random prose, and unterminated blocks.
3. Strict rule that malformed/skipped `need_info` cannot create recursive
   child steps; a child step requires a parsed `missing` object.
4. Per-step `StepResult` journaling in `FastLoopResult`, so replay can inspect
   `DeltaTransaction.status`, `parse_error`, raw excerpt, and
   `produced_by=regex_fallback` signals from the session subgraph.
5. Checker-hard-failure repair path: deterministic hard violations now create
   a child repair step, then resume the parent with the repair signal.
6. Dynamic max-subarray checker tightened to require the segment-tree aggregate
   (`sum`, `prefix`, `suffix`, `best`) and to keep controller hard constraints
   such as `long long` visible even when they are not in the top active packet.
7. Repair recursion termination: repair steps are terminal for checker
   hard-failure repair. If a repair child itself fails the checker, the
   controller records an unresolved repair signal and returns best-effort
   instead of spawning a repair grandchild.
8. `STEP_RESULT` supports `constraints_honored`, so the model can explicitly
   name controller hard constraints it believes the visible answer satisfies.
9. `generic_step_format` now hard-fails a claimed honored constraint when the
   answer/delta text does not contain a matching marker. This catches
   "the packet knew it, but the final answer dropped it" failures.
10. `compose_final_answer()` runs after the fast loop and deterministically
    appends missing hard constraints from the step lineage into the returned
    visible answer. This is intentionally not new reasoning; it is a
    controller-side preservation pass.
11. Checker-produced violation signals are excluded from packet hard
    constraints. This prevents errors such as `honored_constraint_unmarked`
    from being recycled into final-answer "constraints honored" text.
12. Unknown `constraints_honored` claims are soft violations, while known
    packet hard-constraint claims without answer markers remain hard failures.
13. Task-statement-derived coverage now feeds v2 without benchmark rubric
    leakage: salient concepts are extracted from the user question, emitted as
    controller task-concept signals, preserved by the composer, and optionally
    sent through one shaper call only when two or more task-derived concepts
    remain missing.
14. Lexical matching is centralized in `reasoning/lexical_matching.py`.
    `graph_core.lexical_overlap`, activation overlap checks, and substrate
    constraint matching now delegate to shared deterministic helpers. This
    preserves behavior while creating one future replacement boundary for
    learned scoring.

3E-4 remains the real local-model/Qwen comparison phase.

The 3E-4 comparison suite is frozen in
`bench/core_20.json` before baseline/v2 runs. It contains
20 task samples with deterministic required-term/forbidden-term judge hints:
algorithm-design cases such as `alg_dijkstra_negative_edge` and
`alg_dynamic_max_subarray_online`, factual-recall cases such as
`fact_entropy_definition`, and conceptual-reasoning cases such as
`reason_bayes_base_rate`. Use `run_phase3e_benchmark.py` to produce
per-task answers, call counts, smoke-judge results, and session paths under
`artifacts/phase3e_benchmark_<timestamp>/results.json`.

The first no-anchor benchmark exposed a lexical judge problem: correct answers
were failing for phrasing variants such as "data retrieval operations" vs
"lookup/query" and "same object across function calls" vs "shared/reused".
The deterministic rubric was expanded with synonym groups and the runner now
supports `--rescore-results` so existing JSON results can be rejudged without
spending another model run.

The committed suite paths now match `PHASE3E_SUCCESS_CRITERIA.md`:

- `bench/core_20.json` is the frozen 20-task quality/cost suite.
- `bench/cold_warm_5.json` is the 5-task cold/warm/warm warm-start smoke suite.
  On the current local model it saturates at `1.0` cold calls, so it proves
  warm-start reuse wiring but does not by itself prove call-count compounding.
- A follow-up adversarial cold/warm suite should stay inside repair-capable
  domains so cold runs still have room to collapse from Archetype B/C into A.

Sample coverage for the 3E-3 fixes:

| Fix | Sample / test | Why this sample matters |
|---|---|---|
| `DeltaTransaction(parsed/skimmed/dropped)` fallback | `test_parse_malformed_delta_fuzz_fails_open_without_missing_child` | Exercises malformed structured output without allowing a child step from a bad `need_info`. |
| Persisted transaction journaling | `test_attach_fast_loop_journals_delta_transaction` | Confirms trace replay can inspect parse status and `regex_fallback` provenance. |
| Soft `factual_recall` checker | `test_factual_recall_checker_is_soft_and_evidence_anchored` | Keeps factual recall fail-open instead of acting like a truth oracle. |
| Dynamic max-subarray aggregate requirement | `test_dynamic_checker_requires_segment_tree_aggregate_fields` | Locks the IOI-specific requirement for `sum/prefix/suffix/best` plus integer-width handling. |
| Checker-requested repair recursion | `test_checker_hard_failure_opens_repair_child_then_resumes_parent` | Shows a bad Kadane-online answer becoming a repair child and then a corrected parent answer. |
| Repair recursion termination | `test_failed_repair_child_does_not_spawn_repair_grandchild` | Prevents a failed repair child from creating a repair grandchild. |
| Explicit honored-constraint claims | `test_parse_constraints_honored`, `test_generic_checker_rejects_unmarked_honored_constraint_claim` | Parses `constraints_honored` and rejects a model that claims `long long` while the answer still says `int`. |
| Final-answer constraint preservation | `test_compose_final_answer_preserves_missing_hard_constraints` | Covers the IOI failure mode where the internal repair state had `long long` but the visible final answer dropped it. |
| Checker-noise containment | `test_generic_checker_treats_unknown_honored_constraint_claim_as_soft`, `test_composer_does_not_promote_checker_violations_to_answer_constraints` | Keeps model-added meta claims and checker violation strings from creating repair loops or final-answer pollution. |
| Task-statement coverage without rubric leakage | `test_task_statement_concept_extractor_uses_question_text_only`, `test_missing_task_statement_concepts_tracks_visible_answer_terms`, `test_task_concept_constraints_compose_into_key_terms` | Derives coverage terms from the question text, detects terse answer drops, and preserves missing task concepts in the final answer. |
| Lexical matching centralization | `reasoning.tests.test_lexical_matching` | Locks graph-core lexical overlap, activation token overlap, and substrate constraint matching behind one shared helper module. |
| Dijkstra one-call path | Qwen smoke `alg_dijkstra_negative_edge` | Confirms the predicted 1-call trajectory on the negative-edge case. |
| IOI three-call repair path | Qwen smoke `alg_dynamic_max_subarray_online` before checker tightening | Confirms bad first answer -> repair child -> parent resume. |
| IOI post-tightening acceptance | Qwen smoke `alg_dynamic_max_subarray_online` with `k_anchors=0` | Confirms the stricter checker accepts a valid segment-tree answer rather than false-positive failing it. |

Correction after full-output inspection: the first post-tightening IOI
acceptance above was too lenient. The returned answer mentioned prefix/suffix
and long long but did not explicitly name the full `sum/prefix/suffix/best`
aggregate or the cross-boundary merge rule. The checker was tightened again so
generic "maximum subarray sum" no longer satisfies the total segment-sum field,
and final IOI answers must state a left/right prefix/suffix merge.

Raw strict rerun summary:

```text
negative control:
wrong answer = Kadane after each update with claimed O(log n)
checker = passed false
hard violations = kadane_online, segment_tree_missing

cold/no-anchor IOI:
llm_calls.used = 2 / 8
tokens.used = 1838 / 16000
budget_exhausted = false
step 0 = failed; hard violations segment_tree_aggregate_missing, segment_tree_merge_missing
step 1 repair = failed; hard violations segment_tree_aggregate_missing, long_long_missing
```

Do not count the old 1-call IOI result as a substrate win. The clean statement
is narrower: stricter checking now rejects vague IOI answers and the negative
control; the local model still needs better repair prompting or more substrate
context to pass this task reliably.

Raw deterministic rescore after synonym-rubric expansion:

```text
v2 no-anchor existing results:
tasks: 20
ok: 20
judge_passed: 15
total_llm_calls: 23
mean_llm_calls: 1.15

baseline no-anchor existing results:
tasks: 20
ok: 20
judge_passed: 14
total_llm_calls: 21
mean_llm_calls: 1.05
```

Interpretation: the first 9/20 vs 13/20 result was too noisy to guide
architecture. After deterministic rescore, v2 quality is slightly ahead on
the same already-paid-for outputs, but it still has no call-count win. The
next model run should test the final-answer composer, not storage collapse or
3F.

Raw corrected no-anchor v2 benchmark after final-answer composer and
checker-noise containment:

```text
tasks: 20
ok: 20
judge_passed: 13
total_llm_calls: 21
mean_llm_calls: 1.05
```

Comparison against the rescored no-anchor baseline:

```text
baseline rescored: judge_passed=14/20, total_llm_calls=21, mean_llm_calls=1.05
v2 corrected:      judge_passed=13/20, total_llm_calls=21, mean_llm_calls=1.05
```

The composer fixed the IOI visible-answer constraint-loss symptom
(`alg_dynamic_max_subarray_online` now passes in 2 calls), but the corrected
no-anchor run still does not pass the 3E-4 gate. The current blockers are
terse one-call answers and coverage gaps, not storage layout.

Updated 3E-4 benchmark status (2026-05-22):

```text
core_20, warmed local model, k_anchors=0, dual judge:
baseline: judge_passed=14/20, mean_llm_calls=1.1, median_elapsed_sec=3.9555, p95_elapsed_sec=10.113
v2:       judge_passed=20/20, mean_llm_calls=1.0, median_elapsed_sec=3.2705, p95_elapsed_sec=4.274

cold_warm_5, warmed local model, k_anchors=0, dual judge:
cold_mean_llm_calls=1.0
warm_mean_llm_calls=1.0
warm_over_cold_call_ratio=1.0
warm_runs_with_prior_signal_activation=10/10
warm_runs_with_prior_signal_reuse=10/10
```

Interpretation:

- Quality and cost now pass cleanly on the corrected harness.
- Compounding still fails on call ratio, but the blocker is benchmark
  saturation, not missing warm-start reuse: the current smoke suite resolves in
  one call even when cold.
- That makes 3E shippable as a better prompt protocol, but 3F should stay
  paused until a harder compounding suite exists.

---

## TL;DR

The current loop is slow because we made the model *agentic*: it picks objects,
dispatches procedures, navigates modes. That cost is the problem, not the fix.
Procedures-as-objects also turns out to be too narrow a primitive â€” most
reasoning pressure is not callable code.

This doc proposes **3E: a signal graph + recursive step loop**. One LLM call
per step. The graph activates context *before* the call deterministically; the
model emits a small `state_delta` *immediately after* the call (parsed from a
structured `STEP_RESULT` block); a checker either accepts the result or opens a
child step to resolve a named gap. No mode pipeline, no model-agentic
retrieval, no procedure dispatch on the hot path.

**What 3E is and isn't.** 3E is the *data and control substrate* that makes a
future co-processor trainable. It is **not** itself graph/LLM co-processing â€”
it is a better prompt protocol plus disciplined session-graph mutation. The
truly deep coupling (mid-generation graph signals, hidden-state hooks) lives
in 3F and depends on the dataset 3E produces.

Existing work is not thrown out â€” 3A / 3B / 3C / 3D all become *layers* on top
of the new substrate rather than the substrate itself. **Migration is
additive**, not replacement: 3E-1 introduces `SignalNode` as a projection over
existing `activation_signal` / `session_gap` / `plan_node` types; types are
retired only after 3E-4 ships.

---

## Â§0. What is actually slow

Reading the 3D plan against the 3C frame, the LLM calls a single reasoning
episode pays for a weak local model:

| Mode | Calls | What it produces |
|---|---|---|
| focus | 1 | task_kind, required_outputs, constraints |
| plan | 1 per node | goal, hypothesis, mode |
| execute | 1 per node | result, new_evidence |
| check | 1 per node | passed, failure_scope |
| revise | 1 per failure | backtrack_to, new_hypothesis |
| finalize | 1 | answer |
| procedure dispatch (3A/3B) | +N | each fired procedure |

A merely-plausible 8-node tree pays 25â€“35 calls. Each one re-reads a fat task
frame. The edge model isn't slow *per call* â€” it's slow because the
architecture asked for an order of magnitude too many.

**Diagnosis.** The mode pipeline was designed so the model could be *weak but
careful*. Instead it made every weak call a multiplier on every other weak
call. The right move for a weak model is the opposite: **one call, maximally
pre-loaded**, and let recursion happen only when the call itself reports a
missing piece.

The deeper issue is that `procedure` was the wrong primary node type. Looking
at the 3B trace work, the things that actually carried weight in good reasoning
were almost never callable: they were *constraints noticed*, *risks flagged*,
*decisions taken*, *branches abandoned*, *gaps observed*. Procedures are one
shape of reusable signal, not the genus.

---

## Â§1. Three insights, one design

The meeting transcript surfaced three ideas. Each is correct, and they compose.
The proposal is to ship them in that order.

1. **Signal graph.** Replace procedure/object as the load-bearing node type
   with a uniform `SignalNode` (constraint, decision, hypothesis, evidence,
   gap, risk, repair, procedure). Every LLM call emits a small `state_delta`
   that mutates the session subgraph automatically. No special dispatch.
2. **Recursive step loop.** One step = one call. If the model returns
   `need_info` with a named gap, the controller opens a child step whose only
   job is to resolve that gap. The controller still does bounded retrieval
   (k â‰¤ 6, cheap), but the *model* never picks what to fetch â€” there is no
   agentic search loop. Common case is one call total.
3. **Context compiler now, co-processor later.** The graph compiles context *before* the call,
   deterministically. Near-term: prefix-cached capsules served from a KV
   warm-pool. Mid-term: a learned `NeedProbe` that selects which signals to
   inject. Long-term: hidden-state hooks on a local Qwen runner.

How they compose: â‘  defines *what* the graph stores. â‘¡ defines *when* the
graph is touched. â‘¢ defines *how* the graph reaches the model. 3E ships â‘  and
â‘¡ end-to-end with a string context compiler; 3F is the learned/hooked version
of â‘¢.

---

## Â§2. Architecture at a glance

```text
            â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”    â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”    â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”    â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”
   task â”€â”€â–¶ â”‚   task     â”‚â”€â”€â–¶ â”‚ context compiler â”‚â”€â”€â–¶ â”‚ one LLM call â”‚â”€â”€â–¶ â”‚ checker  â”‚â”€â”€â”
            â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜    â”‚  deterministic   â”‚    â”‚ answer +     â”‚    â”‚ + cov.   â”‚  â”‚
                              â”‚  fuses active    â”‚    â”‚ state_delta  â”‚    â”‚          â”‚  â”‚
                              â”‚     signals      â”‚    â””â”€â”€â”€â”€â”€â”€â”¬â”€â”€â”€â”€â”€â”€â”€â”˜    â””â”€â”€â”€â”€â”¬â”€â”€â”€â”€â”€â”˜  â”‚
                              â””â”€â”€â”€â”€â”€â”€â”€â”€â”¬â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜           â”‚                 â”‚        â”‚
                                       â–² activate            â”‚ state_delta     â”‚ gap?   â”‚
                                       â”‚                     â–¼                 â–¼        â”‚
                              â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”´â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”    â”‚
                              â”‚                    signal graph                    â”‚    â”‚
                              â”‚  constraint Â· decision Â· hypothesis Â· evidence     â”‚    â”‚
                              â”‚  gap Â· risk Â· repair Â· procedure                   â”‚    â”‚
                              â”‚  session subgraph â‡„ long-term                      â”‚    â”‚
                              â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜    â”‚
                                       â–²                                                â”‚
                                       â”‚ child step (only on need_info)                 â”‚
                              â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”´â”€â”€â”€â”€â”€â”€â”€â”€â”€â”                          â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”´â”€â”€â”
                              â”‚   child step     â”‚â—€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”‚   answer     â”‚
                              â”‚ opens only on    â”‚     else                 â”‚ + promoted   â”‚
                              â”‚   need_info      â”‚                          â”‚   signals    â”‚
                              â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜                          â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜
```

The four moving parts:

1. **Signal graph** â€” uniform node type; carries both session-scoped working
   memory and long-term consolidated memory.
2. **Context compiler** â€” deterministic Python. Given the current step's focus
   + the active signals, produces a small `StepContextPacket`. No LLM call, no
   top-k search at this stage.
3. **One LLM call per step** â€” emits a structured `STEP_RESULT` containing the
   answer and a tiny `state_delta`.
4. **Checker** â€” deterministic verification; if it passes, the step finishes;
   if the model reported `need_info` or the checker found a violation, a child
   step is opened.

---

## Â§3. Core schemas

### 3.1 `SignalNode` â€” the new general primitive

Replaces `procedure` / `session_object` / `failure_pattern` /
`activation_signal` / `session_gap` / `session_bridge` as the load-bearing
types. Those become *kinds* of one thing.

```yaml
SignalNode:
  id:              str
  kind:            constraint | decision | hypothesis | evidence
                 | gap | risk | repair | procedure
  text:            str                # compact summary, â‰¤ 200 chars
  scope:           session | reusable
  activation_keys: list[str]          # cheap retrieval anchors
  source_step_id:  str | null
  produced_by:     llm_delta | checker | controller | consolidation
  state:           dict | null        # only populated for procedure / repair
  evidence_ids:    list[node_id]
  citation_count:  int                # for promotion
  decay:           float
```

The seven non-procedure kinds cover what the 3B traces actually emitted.
Procedures stay supported but are now *one kind of signal* rather than the
spine of the system.

Retention decision (2026-05-22): keep the legacy `procedure` /
`session_object` lane for **systemic design tasks**. In those tasks, a
persistent workspace object is still more natural than a pure signal packet.
3E changes the default fast path; it does not delete the object/workspace path.

### 3.2 `ReasoningStep` â€” the new control unit

Replaces the six modes (focus / plan / execute / check / repair / finalize)
with one recursive unit. The mode distinction collapses into `looking_for`.

```python
@dataclass
class ReasoningStep:
    step_id:        str
    parent_step_id: Optional[str]
    task_id:        str
    focus:          str                  # what am I doing right now?
    looking_for:    str                  # what info/result do I need?
    context_packet: StepContextPacket    # compiled by controller, not model
    depth:          int                  # recursion depth
    status:         Literal["open", "resolving", "resolved", "failed", "budget_exhausted"]
    result:         Optional[Any]
    delta:          Optional[StateDelta]
```

### 3.3 `StateDelta` â€” the single output channel

Every LLM call ends with a small, parseable block. This is the only way the
graph mutates from inference. No mid-stream dispatch, no inline tool calls.

```text
STEP_RESULT
status: need_info          # resolved | need_info | failed
result: |
  Kadane is O(n) per point update â€” fails q â‰¤ 2Â·10âµ.
missing:
  question:      "Does segment tree with (sum, pref, suff, best) support online updates?"
  why_needed:    "need an O(log n) per-update data structure"
  expected_shape: decision
delta:
  decisions:    ["reject Kadane for online point updates"]
  constraints:  ["q â‰¤ 2e5 online point updates", "non-empty subarray"]
  risks:        ["all-negative array must not return 0"]
  evidence:     []
  gaps:         ["segment_tree_update_complexity"]
END_STEP_RESULT
```

Three things to notice:

- The model never picks which node to call. It just describes what it concluded
  and what's still missing. The controller decides what becomes a child step.
- `delta` is small by design â€” it is not a chain-of-thought dump. The model can
  still think internally; only the delta is journaled.
- This is the same output shape regardless of task kind. There is no
  algorithm-design prompt and a separate Dijkstra prompt; the spine is uniform.

### 3.4 `StepContextPacket` â€” what reaches the model

Compiled deterministically. This is where 3C's task frame lives now.

```yaml
StepContextPacket:
  task_summary:       str             # 1â€“2 lines
  focus:              str
  looking_for:        str
  active_signals:     list[SignalNode] # â‰¤ 6, sorted by activation score
  parent_decisions:   list[str]        # already-made commitments to honor
  open_gaps:          list[str]        # sibling unresolved gaps
  hard_constraints:   list[str]        # checker will enforce these
  budget_remaining:   { tokens, calls, depth }
  cache_key:          str              # hash(active_signal_ids, focus, parent_decisions)
  base_prefix_key:    str | null       # stable bytes shared with parent step
```

**Why â‰¤ 6 active signals?** 3B traces showed that beyond ~6 items the model
began ignoring or hallucinating reconciliation between them. Keeping the packet
small is not a performance hack; it is a correctness property. The graph
carries more â€” the packet carries only what activated *this* step.

**Honest about retrieval.** The controller still does bounded top-k
activation (`k_max=6`). The claim is *not* that retrieval is gone; it is that
(a) it is cheap, (b) the model does not drive it, and (c) the heuristic is
replaceable by a learned scorer in 3F-Î² without changing the loop.

### 3.5 Packet cache (in-session)

Parent-resume after a child step is the common recursion shape, and naive
implementation re-renders and re-prefills the parent's entire packet. To make
the speed claim hold *in 3E*, not just in 3F:

- Every `StepContextPacket` carries `cache_key` = hash of stable inputs (active
  signal ids, focus, parent decision list).
- The renderer keeps a per-session LRU of `(cache_key â†’ rendered_prompt_bytes)`.
- Parent-resume keeps `base_prefix_key` constant and appends only the new
  child evidence; rendered prefix is reused.
- For local inference servers that expose KV-cache reuse (llama.cpp, vLLM
  prefix-cache), pass `base_prefix_key` through as the server's cache key.
  For hosted APIs without that, the saved render cost alone is still worth it.

This is the minimum cache semantics needed to honor Â§4's call counts. The
bigger `GraphCapsule` warm-pool story remains a 3F-Î± concern, but it is no
longer the *only* thing standing between the design and its claimed speed.

---

## Â§4. The fast loop, fully written out

```python
def run_step(step, graph, budget):
    # 1. activation (deterministic, bounded top-k â€” controller-side, not agentic)
    signals = graph.activate(
        focus=step.focus,
        looking_for=step.looking_for,
        keys=step.activation_keys(),
        k_max=6,
    )
    packet = compile_context(step, signals, graph.parent_chain(step))
    prompt = render_with_cache(packet)     # reuses base prefix via packet.cache_key

    # 2. one LLM call
    raw = llm.complete(prompt, max_tokens=700, cache_key=packet.base_prefix_key)
    result = parse_step_result(raw)        # STEP_RESULT block â€” POST-call delta

    # 3. apply state delta to graph (always; even malformed deltas are journaled)
    graph.apply_delta(result.delta, source_step=step.step_id)

    # 4. deterministic check (see Â§4.1 â€” bounded scope, fail-open by default)
    check = checker.verify(result, packet.hard_constraints)

    if result.status == "resolved" and check.passed:
        return result

    # 5. recurse only if a gap is named or check found violation
    if budget.allows_child(step.depth):
        for gap in result.missing_or_violations(check):
            child = ReasoningStep.from_gap(parent=step, gap=gap)
            child_result = run_step(child, graph, budget)
            graph.attach_resolution(parent=step, child=child_result)
        return run_step(step.with_resumed_context(), graph, budget)

    return result.as_best_effort()
```

**Budgets (hard caps, enforced at dispatch):**

| Budget | Default | Hard cap |
|---|---:|---:|
| tokens per step | 700 | 1200 |
| active signals per packet | 6 | 10 |
| child depth | 2 | 3 |
| children per step | 2 | 3 |
| total steps per session | 4 | 8 |

The common case â€” resolved on the first call, one delta, one checker pass â€”
costs **one LLM call**. The adversarial Kadaneâ†’segment-tree case from the 3D
plan costs **three** (parent, one child to resolve update-complexity, parent
resumes â€” and the parent-resume reuses cached prefix per Â§3.5). The current
mode pipeline pays ~12 calls for the same trajectory.

### Â§4.1 Checker â€” scope and failure modes

The loop puts a lot of weight on `checker.verify(...)`. If the checker is
weak, bad deltas pollute the graph. If it is too strict, recursion explodes.
**This is the single biggest implementation risk in 3E**, and the design
commits to bounding it now rather than treating checker scope as an open
question.

The checker is:

- **A registry of per-`task_kind` plugins**, not a monolith.
  Algorithm-design tasks get a different checker than factual-recall.
  Plugins are registered against activation keys; an unrecognized task kind
  falls back to the trivial "hard_constraints present" check.
- **Deterministic Python.** No LLM call on the hot path. (A model-based
  checker is allowed for promotion/consolidation, but not for the
  pass/recurse decision.)
- **Permitted to:** verify syntactic constraints (long long present,
  non-negative-edge invariant, expected-shape match), check the delta is
  parseable, check coverage of hard constraints from the packet, detect
  contradiction with an already-active `decision` signal.
- **Not permitted to:** evaluate whether the *answer* is right beyond
  declared constraints; invent new gaps the model didn't name; mutate the
  graph (only the controller does that).
- **Fail-open by default.** An unknown task kind or an unparseable
  result returns `passed=true, confidence=low`. The result is journaled
  as low-confidence rather than blocking recursion. This biases the loop
  toward shipping an answer over spinning, and lets the consolidation pass
  later demote low-confidence outputs.
- **Hard cap on violations per step.** A checker that finds more than 3
  violations short-circuits to `failed` rather than spawning 3 child steps;
  this kills the recursion-explosion failure mode.

---

## Â§5. How existing phases collapse into this

| Phase | Was | Becomes |
|---|---|---|
| 3A | Procedure dispatch via pattern-matching on free text | One `kind: procedure` signal. The model invokes it by emitting `delta.decisions = ["apply X"]`; controller executes if a callable body exists, else treats it as a decision. **Off the hot path.** |
| 3B | Macro extraction from procedure-call traces | Mines *signal sequences* in session subgraphs. A recurring decisionâ†’constraintâ†’repair triple becomes a candidate macro signal, not a callable macro procedure. Same trace miner, broader corpus. |
| 3C | `GraphTaskFrame` + coverage check stitched into the prompt | Becomes the `StepContextPacket`. Coverage check becomes part of the deterministic checker in Â§4. |
| 3D | focus/plan/execute/check/revise/finalize mode pipeline with checkpoint tree | The *tree* survives â€” the recursion forest in Â§4 *is* a plan tree, with each step its own checkpoint. The *modes* are deleted. Backtracking = picking a different child step from the parent's gap list. |

None of the 3Aâ€“3D code is wasted: the schemas project onto `SignalNode`, the
synthetic IOI/Dijkstra fixtures still validate the new loop, and the checker
keeps the coverage logic. What disappears is the mode-driver code and the
procedure-dispatch hot path.

**Important: migration is additive, not destructive.**
In 3E-1, `SignalNode` is introduced as a *projection* over the existing
`activation_signal`, `session_gap`, `session_bridge`, `plan_node`, and
`plan_check` types â€” read views, not replacements. The new loop reads through
the projection. Existing writers keep writing to existing types. Only after
3E-4 ships and the loop is in daily use do we collapse the underlying storage.
This keeps 3E-1 a substrate experiment, not a schema migration project.
Additional decision (2026-05-22): storage collapse does **not** mean deleting
legacy `procedure` / `session_object` support. Those nodes remain available as
a specialized subsystem for systemic design reasoning even if they stop being
the mandatory hot path for direct-answer tasks.

**Net effect.** Same architectural ambition (working memory, structured
composition, replayable sessions, consolidation) â€” but the spine is one call +
one delta + one checker, not a six-mode pipeline. Procedures become opt-in,
not the default.

---

## Â§6. The co-processor question

The transcript's third idea â€” wire the graph into the model's hidden state â€”
is the right end goal, but it has to be earned. Three rungs, ship in order:

### 3E (now) â€” substrate, not co-processor

**Be explicit about what 3E is.** It is a better prompt protocol plus
disciplined session-graph mutation, with in-session packet caching. The
"co-processor" framing is aspirational here; the only thing being co-processed
in 3E is the prompt, not the model's hidden state. What 3E earns is the right
shape of data â€” every (packet, delta, checker outcome) tuple becomes a
training example for the real co-processor in 3F.

### 3F-Î± â€” KV warm pool (when 3E proves the loop)

Most of the latency at this stage will be prefill, not generation. Cluster
signals into *capsules*; pre-render each capsule once; ask the inference server
(llama.cpp / vLLM both support this) to keep the capsule's KV cache resident.
Each step still emits one HTTP call, but the prefix is hot.

```yaml
GraphCapsule:
  capsule_id:       str
  domain:           str               # algorithm / dijkstra / dp / ...
  signal_ids:       list[node_id]
  rendered_prefix:  str               # stable bytes
  prefix_hash:      str               # cache key for server
  token_budget:     int               # typically 200â€“500
```

Session start picks â‰¤ 2 capsules from activation keys; the model warms on them
once and reuses across every step in the session. Per-step packets shrink to
focus + delta-since-last.

### 3F-Î² â€” learned NeedProbe + adapter (research)

Once the capsule store is real and we have a corpus of
(packet â†’ useful signal subset) pairs from 3F-Î± traces, train a small probe
that scores signals against the current hidden state, and an adapter that
injects the selected signals as prefix-KV. This is the "graph as live memory
module" the transcript described, but trained on data we have rather than data
we hope to invent.

The honest realism check from the transcript holds: deep injection requires a
custom transformers runner around local Qwen. Don't build 3F-Î² until 3F-Î±
shows clear wins.

---

## Â§7. Worked examples

### 7.1 Trivial direct-answer task

```text
stepâ‚€: focus = "answer user", looking_for = "final answer"
  packet  â†’ 2 activated signals (definition + 1 worked example)
  call    â†’ STEP_RESULT { status: resolved, delta: {decisions: [...]} }
  checker â†’ pass
  done. 1 LLM call total.
```

### 7.2 IOI dynamic max-subarray (adversarial choice)

```text
stepâ‚€: focus = "solve max-subarray under online updates"
  packet  â†’ constraints {qâ‰¤2e5, non-empty, neg ok}, signals {Kadane, seg-tree}
  call    â†’ STEP_RESULT {
              status: need_info,
              delta.decisions: ["reject Kadane for online updates"],
              missing.question: "seg-tree merge rule for max-subarray"
            }
  checker â†’ pass (no violation, just an open gap)

stepâ‚ (child): focus = "derive seg-tree merge for max-subarray"
  packet  â†’ parent decision + (sum, pref, suff, best) hint
  call    â†’ STEP_RESULT { status: resolved, delta.evidence: [merge rule] }

stepâ‚€ resumed:
  packet  â†’ now includes child's evidence
  call    â†’ STEP_RESULT { status: resolved, result: full C++ }
  checker â†’ pass (long long present, non-empty respected)
  done. 3 LLM calls total. Old design: â‰¥ 12.
```

### 7.3 Dijkstra with a negative edge

```text
stepâ‚€: focus = "compute shortest paths"
  packet  â†’ signal {Dijkstra}, signal {non-negative edge invariant}
  call    â†’ STEP_RESULT {
              status: resolved,
              delta.risks: ["graph contains negative edge â†’ Dijkstra invalid"],
              delta.decisions: ["use Bellman-Ford"]
            }
  checker â†’ flags Dijkstra-invariant violation IF model had committed to it
           â†’ here, model already self-corrected, so pass
  done. 1 LLM call. The risk signal promotes into long-term memory.
```

---

## Â§8. Migration plan

| Slice | What ships | Gate to next |
|---|---|---|
| **3E-1** | `SignalNode` + `ReasoningStep` + `StateDelta` schemas as a **read projection** over existing `activation_signal` / `session_gap` / `plan_node` types. `StepContextPacket` (with `cache_key`) as a re-shape of the 3C task frame. No existing writer changes. | Round-trip the 3D synthetic IOI + Dijkstra fixtures through the projection. Unit tests green. No existing test regresses. |
| **3E-2** | Fast loop in Â§4 behind `enable_substrate_v2` flag. One mode-free prompt template. Per-task-kind checker registry (Â§4.1) with fail-open default. In-session packet cache (Â§3.5). | IOI synthetic resolves in â‰¤ 3 LLM calls. Dijkstra negative-edge resolves in â‰¤ 1. Tests green. |
| **3E-3** | Recursive child steps + gap resolution. Budget enforcement. Replay/audit log. Malformed-delta fallback path. Add `factual_recall` checker plugin for non-IOI/general tasks. | Adversarial Kadaneâ†’seg-tree case ends with the right answer and a replayable child trace. No infinite recursion under fuzzed inputs or malformed deltas. |
| **3E-4** | Wire real Qwen behind the loop. Compare against current 3D prompt-mode integration on a 20-task suite. | â‰¥ 3Ã— call reduction at equal answer quality. If yes, schedule storage collapse (retire underlying node types). If not, re-tune packet size and checker strictness before moving on. |
| **3F-Î±** | `GraphCapsule` + KV warm-pool against llama.cpp. Capsule selection at session start. | Measurable prefill latency drop on repeat-domain sessions. |
| **3F-Î²** | Learned `NeedProbe` + prefix-KV adapter. Custom Qwen runner. | Research milestone. Do not start until 3F-Î± is in daily use. |

**What I'd pause.** Phase 3D-2 (prompt-mode integration of
focus/plan/execute/check/revise/finalize). The deterministic 3D-1 substrate is
reusable inside 3E as the plan-tree projection; the prompt driver is the thing
we are explicitly trying not to ship.

---

## Â§9. Decisions formerly open

### 9.1 Delta parser robustness

**Decision:** no retry on the hot path. Use a `DeltaTransaction` with three
outcomes:

1. `parsed`: strict `STEP_RESULT` block parsed cleanly.
2. `skimmed`: strict parse failed, but the answer text can be skimmed into a
   low-confidence delta.
3. `dropped`: neither parse nor skim produced usable state.

Malformed deltas are always journaled with the raw excerpt and
`confidence=low`. They may create `decision`, `risk`, or `evidence` signals
only when the skimmed text is explicit. They may **not** create child steps,
procedure signals, or reusable long-term signals. A child step requires a
strictly parsed `missing` object with `question`, `why_needed`, and
`expected_shape`.

One retry is allowed only outside the latency-sensitive path: offline
validation, test fixtures, or explicit debug mode. In production 3E, bad
formatting should degrade into a low-confidence answer, not another LLM call.

Skimmed deltas must tag every emitted signal with
`produced_by=regex_fallback`. Consolidation and promotion treat this as weaker
than `llm_delta`, `checker`, or `controller`; a regex-fallback signal cannot be
the sole evidence for reusable promotion.

### 9.2 Checker plugins for 3E-2

**Decision:** ship the smallest plugin registry that covers the synthetic gates
and does not pretend to verify arbitrary truth.

3E-2 plugins:

| Plugin | Applies when | Checks |
|---|---|---|
| `generic_step_format` | all tasks | parse status, expected output shape, hard constraints are represented in result/delta |
| `algorithm_design` | `task_kind=algorithm_design` | complexity claim present when required, online/offline constraints not contradicted, required language/output present |
| `dynamic_max_subarray` | activation keys include max-subarray/update/segment-tree | rejects Kadane-only online-update answers, requires segment tree aggregate, `long long`, and non-empty/all-negative handling when those constraints are active |
| `shortest_path_safety` | activation keys include Dijkstra/shortest-path/negative-edge | rejects Dijkstra commitment under negative-edge signal, accepts Bellman-Ford or explicit unsafe-Dijkstra answer |
| `factual_recall` | non-algorithm/general tasks | soft-checks that resolved answers overlap active evidence when evidence is present; never judges arbitrary truth |

Deferred:

- mathematical proof checking,
- compile/run code checking,
- model-based critique,
- procedure safety beyond "do not run procedures on the hot path."

Unknown task kinds use `generic_step_format` only and fail open.

`factual_recall` landed in 3E-3. It is intentionally soft/fail-open so broad
non-algorithm tasks can be traced without pretending that deterministic Python
can verify arbitrary factual truth.

### 9.3 Promotion criteria for non-procedure signals

**Decision:** promotion is per-kind, source-aware, and stricter for model-only
claims. Default scope remains `session`; reusable promotion is opt-in through
consolidation.

Initial thresholds:

| Signal kind | Promotion rule |
|---|---|
| `constraint` | promote after 1 clean high-confidence session if source is checker/controller; require 1 clean high-confidence model-delta session plus no contradiction, or 2 distinct model-delta sessions if ungraded |
| `risk` | promote after 1 clean high-confidence session if source is checker/controller; require 1 clean model-delta session plus checker reference, or 2 distinct model-delta sessions if ungraded |
| `decision` | promote after 2 clean sessions with same activation keys and same successful outcome |
| `repair` | promote after 3 sessions where the repair follows a matching failed branch and final answer passes |
| `gap` | do not promote unresolved gaps; promote resolved gaps only as `evidence` or `repair` |
| `hypothesis` | never auto-promote unless a checker/plugin explicitly references it as supported |
| `evidence` | never auto-promote unless tied to existing graph evidence or checker-confirmed output |
| `procedure` | keep existing stricter procedure validation path |

Promotion requires:

- `correct=True` when a grade exists,
- no budget exhaustion,
- no procedure errors,
- no checker hard failure,
- no malformed-only delta as the sole source.

### 9.4 Unknowable gap termination

**Decision:** add `kind: unresolved_gap` as a signal kind or `gap.state =
"unresolved"` if we keep the enum small. Do not keep retrying equivalent gaps.

Gap identity is canonicalized by:

```text
hash(expected_shape, normalized_question)
```

The smaller key is deliberate: it enables cross-session deduplication when the
same gap appears under different parent task wording. Parent/task/constraint
metadata is still stored on the node for debugging and scoring, but it is not
part of the canonical id.

If the same gap fails twice in one session, or if child depth/call budget is
exhausted, the controller emits an unresolved-gap signal:

```yaml
kind: unresolved_gap
text: "Could not resolve whether X under constraints Y"
state:
  attempts: 2
  failure_reasons: [...]
  parent_step_id: ...
  terminal: true
```

Terminal unresolved gaps block further recursion for that gap id. Parent steps
must either produce a best-effort answer that names the uncertainty, or fail
closed if the expected output cannot be honest without the missing information.

### 9.5 3F-Î± and OpenCode

**Decision:** treat 3F-Î± KV warm-pool as local-runtime-only for now. Do not
design it around OpenCode.

OpenCode may benefit from provider prompt caching if the provider exposes it,
but that is an opportunistic optimization, not an architectural dependency.
For 3E/3F measurements, maintain separate runtime tracks:

| Runtime | Expected cache strategy |
|---|---|
| local Qwen via llama.cpp/vLLM | prefix/KV cache keys, capsule warm-pool |
| hosted/provider through OpenCode | prompt-cache headers if exposed; otherwise no KV assumption |
| plain shell model command | no cache assumption |

The 3F-Î± gate is measured only on local Qwen. OpenCode remains a quality
baseline, not the target runtime for deep graph/LLM coupling.

---

## Â§10. Roadmap beyond 3F: Phase 3G â€” Closed-Loop Substrate

3E gives the system a substrate. 3F connects that substrate to inference. 3G
is where the graph and model start improving each other instead of remaining a
fixed model plus external memory.

The loop:

```text
session run
  -> (packet, delta, checker, answer, outcome)
  -> outcome scorer
  -> substrate gradient
       - NeedProbe online updates
       - signal embedding updates
       - promotion threshold calibration
       - graph topology pruning
  -> periodic LoRA distillation
       high-confidence substrate behavior folds back into model weights
```

### 10.1 Outcome scoring

Every session must produce a training row:

```yaml
SubstrateOutcomeRow:
  packet_id: str
  delta_transaction_id: str
  checker_result_id: str
  final_answer: str
  outcome:
    correct: bool | null
    score: float
    source: deterministic | test_runner | llm_judge | manual
```

Algorithm and code tasks should prefer deterministic scoring: compile/run,
unit tests, known-answer checks, or task-specific validators. Factual/general
tasks can use a small LLM judge or manual grade, but those labels must be
tagged separately from deterministic labels.

This is the dataset 3F-beta needs: which packets and deltas actually helped.

### 10.2 Online NeedProbe and topology pruning

The substrate should stop being write-only.

Weekly or batch updates:

- Edges that never appear in successful packets decay.
- Signals that recur in successful packets cluster.
- Activation scoring is fine-tuned from `(step context -> useful signal ids)`.
- Promotion thresholds are calibrated against outcome scores.
- Dead or misleading topology is pruned or downweighted, not immediately
  deleted.

This is the first place where the graph learns what it should retrieve rather
than merely storing what it was told.

### 10.3 Distillation back into the model

Take high-confidence substrate traces and materialize them as:

```text
prompt -> expected STEP_RESULT delta -> expected final answer
```

Train a small LoRA on the recurring patterns. After distillation, the base
model should need less substrate scaffolding for problems the substrate has
already solved repeatedly. This is the earned version of "the graph changed
the weights": the graph does not write into weights directly; successful graph
behavior becomes training data.

### 10.4 Gates before 3G starts

Do not start 3G until:

1. 3F-alpha is running in daily local-model use.
2. The trace corpus has at least 5,000 labeled outcomes.
3. At least 60% of labels are deterministic, test-runner, or manual rather
   than LLM-judge-only.
4. The activation/packet logs are stable enough that a NeedProbe target means
   the same thing across runs.

Without these gates, 3G is open-ended ML research. With them, it is a closed
feedback loop with a real objective.

---

## Â§11. What this is not

- Not a replacement for the graph â€” it makes the graph carry more, not less.
- Not a rejection of procedures â€” they remain one signal kind, with their
  existing dispatch path intact for cases that genuinely benefit.
- Not a new prompt format â€” `STEP_RESULT` is one block, not a grammar.
- Not the neural co-processor itself â€” 3E is the substrate that makes the
  co-processor trainable later.
- Not multi-agent. One reasoner, one graph, recursive but serial.

---

## Deep-suite note (2026-05-22)

The exploratory `bench/deep_reasoning_5.json` suite is now the right place to
measure whether v2 is becoming a **graph reasoner** rather than only a better
prompt protocol.

What the latest local-model runs showed:

- Baseline still sits at `0/5` strict passes and `1.0` mean calls.
- V2 improved to `1/5` strict passes and forced honest repair traces on the
  system-design tasks, but it is still not a deep-graph win yet.
- On the latest v2-only rerun, mean calls fell from `1.8` to `1.4` after
  tightening compound constraint matching and removing blanket
  `constraints_honored` meta-claims.

Milestone framing:

- A stronger plain 7B model will probably improve answer surface quality.
- But `7B` is **not** the real milestone. The real milestone is:
  1. more strict passes on `deep_reasoning_5`
  2. visible Archetype B/C traces on systemic-design tasks
  3. bounded call count while those traces become more explicit

Residual bottlenecks after this slice:

- payment answers still hide PSP-status lookup / reconciliation too deep in
  prose;
- migration answers still under-surface replayability;
- inventory answers still need a better invariant-first answer surface rather
  than generic system-summary prose;
- dynamic-connectivity answers are checker-clean but still brittle against the
  strict external rubric unless they explicitly name the time-structure in the
  visible answer.

## Closeout note (2026-05-23)

The 3E closeout harness is now materially better, but 3E should still be
treated as **not yet formally closed**.

What improved:

- closeout gates now report `passed` / `failed` / `skipped`;
- `quality_cost` reads benchmark JSON and enforces the implemented numeric
  gates instead of subprocess exit code;
- replay checking was renamed honestly to `replay_artifact_integrity`;
- offline `negative_controls` and `recursion_fuzz` now use task-derived packet
  context instead of empty packets;
- `bench/recursion_fuzz.json` now exceeds the criteria minimum at `106` cases.

Verified offline closeout:

```text
negative_controls: 44 cases, 33 passed, 11 failed
recursion_fuzz: 106 cases, 64 passed, 42 failed
replay_artifact_integrity: 60 sessions, 60 passed, 0 failed
```

What this means:

- The harness is now honest enough to guide work.
- The remaining failures are real signal, not bookkeeping noise.
- But two acceptance-path gaps still matter:

1. The replicated `compounding` gate still needs a final fix so warm/cold
   quality is derived directly from replicate `rows` in the payload. Until that
   lands, the warm-quality clause is not a trustworthy acceptance signal.
2. `recursion_fuzz` still classifies parser/checker output rather than running
   a true recursive loop, so it is not yet a full measurement of recursion
   exhaustion or repair containment.

The current closeout blockers are therefore:

- compounding quality wiring in the replicated path;
- parser-vs-suite contract alignment for malformed `STEP_RESULT` cases;
- checker lexical coverage for a small set of negative controls;
- a real stubbed-loop fuzz harness if we want `recursion_fuzz` to prove the
  discipline/failure-containment claims instead of only parser/checker behavior.

Sample-to-fix map to keep the bigger picture visible:

| Remaining blocker | Sample(s) | Why it matters |
|---|---|---|
| Migration downtime phrase coverage | `nc_migration_big_bang` | `maintenance window` should trigger the same unsafe-cutover signal as `big bang` / `take downtime`. |
| Inventory authority / mutex phrase coverage | `nc_inventory_cache_authority`, `nc_inventory_global_mutex` | These are semantically bad answers that still miss exact checker phrases. |
| Generic malformed-step contract | `nc_generic_missing_required_on_need_info`, `nc_generic_delta_dropped`, `fuzz_need_info_no_missing` | The parser currently falls back to `skimmed/resolved` on malformed structured output, which does not match the stricter suite expectation. |
| Empty factual-answer contract | `nc_factual_empty_answer` | The plugin only flags empty resolved factual answers; empty dropped output currently bypasses it. |
| Parser skim-vs-drop policy | `fuzz_missing_delta_entirely`, `fuzz_delta_no_status`, `fuzz_garbage_text`, `fuzz_unclosed_step_result` | The suite assumes dropping; the parser currently preserves more malformed output as skimmed text. |
| True recursion/repair containment proof | `fuzz_repair_chain_no_resolution`, `fuzz_50_violations_in_one_step`, `fuzz_cyclic_gap_ids` | These need a real loop harness, not only parser/checker classification, to prove the discipline gate honestly. |

So the status line for 3E should remain:

- **quality/cost:** strong
- **closeout harness:** much better and now informative
- **formal acceptance:** not complete yet
- **3F:** still blocked on finishing the closeout truthfulness path

One framing correction is important:

- `core_20` already answers the original operational 3E question:
  the old high-call-count loop was replaced with a cheap local-runtime path.
- It does **not** answer the substrate question, because cold single-session
  tasks have nothing to compound against.

So the project should stop treating "3E original objective" and
"3E formal substrate proof" as the same milestone:

| Question | Best instrument | Current status |
|---|---|---|
| Did 3E collapse edge-runtime cost? | `bench/core_20.json` | **Yes** |
| Did 3E create cross-session compounding? | `bench/cold_warm_adversarial.json` + closeout gates | **Not fully proved yet** |

That split is the right way to decide the next move:

- do not reopen loop/cost architecture work unless compounding evidence says
  the substrate itself is weak;
- do not treat answer-surface tuning alone as proof that the graph compounds;
- use the cold/warm path as the actual go/no-go for the 3E -> 3F decision.
