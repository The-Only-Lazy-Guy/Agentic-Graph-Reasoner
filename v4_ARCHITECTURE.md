# answerer_v4 Architecture

**Last updated:** 2026-05-28  
**Current status:** implemented research pipeline. The graph/control path is now load-bearing, tool use is verified, the micro-controller can force a fast finalize path when required slots are already satisfied, graph-learning candidates now pass through a scoped patch/audit layer before they are treated as trustworthy, and the signature layer now stores explicit family-variant relations (`overlaps` / `entails` / `contradicts`) in addition to shadow/live retrieval signals.

---

## Purpose

V4 is the backend answerer used to collect high-quality behavior from `opencode/big-pickle` and turn it into a smaller local graph-native reasoner later.

The target behavior is not "LLM thinks harder." It is:

```text
query
  -> identify task type
  -> load relevant graph working set
  -> check whether subgoals are already solved
  -> read only required evidence
  -> answer
  -> extract reusable graph learning
```

For already-solved tasks, the desired path is close to:

```text
query -> read graph evidence -> answer
```

The LLM should be an operator inside a controlled loop, not the whole agent.

---

## Runtime Layers

### 1. Long-Term Graph Memory

Persistent graph in `graphs/*.json`.

It stores more than plain facts:

| Memory type | Purpose |
|---|---|
| `claim` / `fact` / `application` | Grounded reusable knowledge |
| `failure_pattern` / deprecated misconception | Known bad reasoning paths |
| `strategy` | Reusable task recipe with key evidence nodes |
| `solved_subgoal` | "This subproblem is already answered" capsule |
| `reasoning_atom` | Reusable mechanism/invariant explanation |
| `control_rule` | Required slots and stopping policy for a task family |
| `procedure` / examples | Verification and implementation building blocks |

The important distinction is that a normal fact can say "Dijkstra requires nonnegative weights," while a solved subgoal can say "the Dijkstra negative-edge applicability subproblem is already solved, with verdict/reason/alternative/caveat filled."

### 2. Local Working Memory

Built once near the beginning of a query from anchors and compatible graph nodes.

Contains:

```text
task frame
anchor node ids
candidate facts
candidate solved_subgoals
candidate reasoning_atoms
candidate strategy nodes
control rules
slot values already filled
```

Micro-steps check this local working set first. Global graph search should only happen when the working set cannot fill a required slot.

### 3. Micro Epistemic Controller

Implemented in `reasoning/micro_controller.py`.

It operates at semantic subgoal granularity:

```text
current subgoal
  -> knownness check
  -> slot sufficiency check
  -> cheapest action
```

Action space:

```text
REUSE
QUERY
DERIVE
VERIFY
ASK
FINALIZE
```

Core output:

```text
task_family
task_signature
required_slots
optional_slots
micro_steps
slot_values
selected_node_ids
controller_action_counts
recommended_action=FINALIZE when all required slots are filled
```

The controller uses slot masks, not free-form vibes:

```text
required_slots subset filled_slots => sufficient
```

When the controller recommends `FINALIZE`, `answerer_v4.py` now enforces that path by reading controller-selected evidence nodes and doing a one-shot finalization call with `MICRO_FINALIZE_SYSTEM_PROMPT`.

### 4. LLM Tool Executor

Implemented in `answerer_v4.py`.

If the micro-controller cannot finalize, the system falls back to the full graph-tool loop. The model can call:

| Tool | Purpose |
|---|---|
| `read_node(id)` | Read full graph evidence |
| `expand_neighbors(id, k)` | Follow graph edges |
| `search_nodes(query, k)` | Search graph when local memory is insufficient |
| `hypothesize(text)` | Record a gap, not answer evidence |
| `create_object` / `update_object` / `read_object` | Stateful task workspace |
| `record_failure` | Persist a failed approach in the session |
| `mark_done(index)` | Track plan progress |
| `verify_hypotheses` | Stamp hypotheses before finalizing |
| `invoke_procedure` | Run opt-in verification procedures |

Loop-mode answers are guarded by `validate_answer_reads(...)`: at least one `read_node` call is required before accepting an answer. This keeps the model from bypassing graph evidence with pure generation.

Important opencode integration detail:

```text
generic controllers emit <tool>{...}</tool>
opencode path emits <graph_action>{...}</graph_action>
```

This avoids opencode mistaking graph-tool requests like `read_node` for native backend tools. The parser accepts both formats, but the opencode prompt now explicitly tells the model to emit plain-text graph actions rather than native tool calls.

### 5. Post-Processing and Learning

After a finalized answer, V4 can extract graph learning:

| Component | File | Purpose |
|---|---|---|
| deterministic report | `reasoning/post_processing.py` | cited nodes, verified claims, failures, synthesized objects |
| graph edits | `reasoning/post_processing.py` | candidate node/edge updates |
| scoped edit patches | `reasoning/scoped_edits.py` | scope/evidence/risk/validation wrapper for graph edits |
| signature stats | `reasoning/signature_stats.py` | family/variant candidates, typed events, persistent stats, shadow rerank, and gated live anchor bias |
| reflection | `reasoning/reflection.py` | LLM-generated learning summary |
| edit applier | `reasoning/graph_editor.py` | deterministic validation/application |
| distillation corpus | `reasoning/distillation_corpus.py` | SFT rows from full sessions |

`apply_graph_edits=False` is the default safety mode. In that mode, edits are produced and dumped, but the graph file is not mutated. When `apply_graph_edits=True`, V4 now applies only raw edits whose scoped patch status is `accept` or `soft_only`; `needs_review` and `reject` patches are held back.

The scoped patch layer is deliberately non-mutating. It converts raw edits into typed patches:

```text
reinforce_existing
add_fact
add_relation
add_strategy
add_solved_subgoal
add_reasoning_atom
add_control_rule
deprecate_fact
```

Each patch carries:

```text
scope
valid_when / invalid_when
evidence_node_ids
affected_node_ids
source_session
confidence
risk_level
validation.status = accept | soft_only | needs_review | reject
```

This gives us a GMeLLo-inspired edit proposal layer while keeping the symbolic graph as the source of truth.

On top of that, V4 now keeps a signature memory layer:

```text
strategy / solved_subgoal / provisional_claim
  -> signature_family
  -> signature_variant
  -> explicit variant relations
  -> typed events
  -> support/stability/risk/bias stats
  -> shadow rerank report
```

By default this layer is shadow-only and auditable. It now also has a narrow opt-in live path: for `algorithm_applicability` and `direct_judgment`, V4 can ask the signature layer for a graph-backed anchor-bias hint before retrieval.

That live path is intentionally conservative:

```text
shadow-ranked signature memory
  -> keep only supported solved_subgoal variants
  -> skip contested families
  -> use graph-backed support node ids only
  -> prepend those anchors to normal retrieval
```

Broad provisional strategy families are not allowed to drive the live bias directly.

The signature layer now also persists explicit sibling-variant relations:

```text
overlaps
entails
contradicts
```

These are stored in the signature stats index and emitted in the graph projection. They do not yet drive live retrieval directly, but they already support:

- contested-family detection from explicit contradictions
- entailment direction from specific variant -> general variant
- bounded score propagation / promotion logic

Current design caveat from the latest broad live runs:

```text
direct_judgment solved-subgoal family signatures are still too coarse
```

The family gating is now much safer than before, but the current canonicalization can still collapse unrelated direct-judgment explanations into one family if the signature is too broad. That means a live bias may still prepend a semantically irrelevant support node even when the final answer remains correct. This needs tighter family signatures before we broaden live use further.

For each run, V4 also emits a graph-shaped shadow projection for the touched signature families:

```text
signature_family --has_variant--> signature_variant
signature_variant --realized_as--> strategy|solved_subgoal|provisional memory node
signature_variant --supported_by--> evidence nodes
signature_variant --overlaps/entails/contradicts--> sibling variants
```

So the family/variant layer is no longer just an index; it now has an explicit graph view we can inspect and later promote.

The score model is also now split into:

```text
direct scores
  support / stability / risk / contradiction / bias

plus bounded relation-propagated scores
  propagated_support
  propagated_stability
  propagated_risk
  propagated_contradiction

which produce
  effective_support
  effective_stability
  effective_risk
  effective_contradiction
  effective_bias
```

Propagation is deliberately shallow and typed:

- `overlaps` shares a weak amount of support/stability
- `entails` sends stronger support/stability from specific -> general
- `contradicts` adds contradiction/risk pressure and blocks promotion more aggressively

This is the first implementation of the "bounded BFS-like propagation" idea, but restricted to one hop and explicit relation types only.

---

## Main Flow

```text
answer_query_v4(question, graph)
  -> classify / auto_config if enabled
  -> optional signature live bias hint
  -> retrieve anchors
  -> run graph activation frame
  -> run micro_epistemic_controller
  -> if recommended_action=FINALIZE:
       read controller-selected evidence
       call one-shot finalizer
       execution_mode = micro_controller_finalize
     else:
       enter full plan/tool/read/search loop
       enforce read_node before answer
       execution_mode = loop
  -> optional answer polish
  -> learning report + graph edits
  -> scoped patches + validation summary
  -> signature candidates + signature events + shadow rerank
  -> optional reflection
  -> optional corpus write
  -> return V4Packet
```

Important packet fields:

| Field | Meaning |
|---|---|
| `execution_mode` | `micro_controller_finalize`, `loop`, `direct_shortcut`, etc. |
| `tool_call_count` | All graph/session tool calls |
| `controller_call_count` | Big-pickle calls made by the controller |
| `controller_raw_trace` | Raw stdin/stdout/stderr for evidence dumps |
| `micro_steps` | Controller decisions at subgoal level |
| `slot_fill_stats` | Required/filled/missing slot accounting |
| `learning_report` / `graph_edits` | Post-processing output |
| `graph_edits_applied` | Whether graph mutation actually happened |
| `scoped_patches` / `scoped_patch_summary` | Scoped validation of proposed graph edits |
| `signature_candidates` / `signature_events` | Shadow signature-memory extraction for this run |
| `signature_stats_update` | Index delta / touched family-variant summary |
| `signature_shadow_report` | Baseline vs bias-adjusted shadow ranking, now including full candidate rankings and collapsed family rankings for Layer 2 eval |
| `signature_graph_projection` | Graph-shaped shadow view of touched signature families/variants |
| `signature_live_bias` | Live anchor-bias audit for the pre-retrieval signature hint path |

---

## Task Families and Slots

The controller currently recognizes these major families:

| Task family | Required slots |
|---|---|
| `algorithm_applicability` | `verdict`, `reason`, `alternative`, `caveat` |
| `direct_judgment` | `answer`, `reason` |
| `algorithm_mechanism_explanation` | currently `mechanism`, `answer` |
| `algorithm_usage_context` | `usage_context`, `answer` |
| `design_synthesis` | `problem_frame`, `core_structure`, `rank_query`, `pagination`, `tie_policy`, `scale_architecture`, `latency_budget`, `consistency_model`, `failure_mode_fix`, `answer` |
| `procedure_or_instance_verification` | `instance_summary`, `precondition_results`, `verdict`, `answer` |
| `relational_explanation` | `relationship`, `explanation` |

Known design correction for later:

```text
algorithm_mechanism_explanation should not allow a shortcut with only:
  ["mechanism", "answer"]

For Dijkstra-like questions, it should require:
  ["mechanism", "relaxation_step", "priority_queue", "nonnegative_precondition", "answer"]
```

This prevents fast answers from omitting the nonnegative-weight precondition.

---

## Strategy Nodes

Strategy nodes are intended to generalize across related questions, not memorize an exact query string.

Good strategy metadata should include:

```text
task_family
task_subtype
question_mode
entry_conditions
required_slots
optional_slots
key_node_ids
checkpoint_plan
stop_conditions
forbidden_finalize_conditions
strategy_schema_version >= 2
```

Example intent:

```text
Question A: "Can Dijkstra be trusted with one negative edge?"
Question B: "Is Dijkstra valid if some edges are negative?"

Both should map to:
  shortest_path.dijkstra.negative_edge_weights.validity
```

The strategy should help B by filling the same subgoal slots, not only by matching A verbatim.

---

## Verified Evidence So Far

### Dijkstra repeat / micro-controller finalize

Artifact:

```text
artifacts/paper_full_dumps/20260527_040440/full_dump_report.md
```

Observed:

| Run | execution_mode | steps | tool calls | controller calls | graph edits |
|---|---:|---:|---:|---:|---:|
| 1 | `micro_controller_finalize` | 1 | 3 | 1 | 3 added |
| 2 | `micro_controller_finalize` | 1 | 3 | 1 | 1 added |

This proves the enforced `recommended_action=FINALIZE` path works and can collapse an already-solved task to read-and-answer behavior.

### Systemic-thinking baseline vs graph

Artifact:

```text
artifacts/systemic_thinking_comparison/20260527_041701/full_comparison_report.md
```

Prompt: real-time competitive-programming leaderboard design.

| Mode | Time | Steps | Tool calls | Notes |
|---|---:|---:|---:|---|
| Raw baseline | 163.1s | N/A | 0 | coherent but made unsupported scaling/linearizability claims |
| Graph pipeline | 163.5s | 10 | 29 | confirmed real tool use and produced 26 candidate graph edits |

The graph run was not faster on this hard design task yet, but it was auditable and tool-grounded. The baseline showed mild-to-moderate hallucination risk, especially the claim that per-user version pinning guarantees linearizable rank reads.

### Scoped edit lab

Artifact:

```text
artifacts/scoped_edit_lab/latest_check/edit_lab_report.md
```

Input artifact:

```text
artifacts/v4_difficulty_sweep/20260527_234038
```

Observed over the trivial/medium/hard sweep:

| Count | Value |
|---|---:|
| cases | 3 |
| raw graph edits | 51 |
| `soft_only` patches | 14 |
| `accept` patches | 21 |
| `needs_review` patches | 16 |

Important signal: the hard leaderboard strategy patch was marked `needs_review` because it included low-relevance evidence node `shortest_path_grid_apply`, and all child `leveraged` edges inherited that status. The lab also held back hard-task claims whose evidence supported Fenwick prefix/update but not stronger claims like `find_kth`, range pagination, score buckets, or tie policy.

### Opencode loop repair

Validated with a live targeted rerun of:

```text
If astronauts can see sunlight in space, why can't they hear it there?
```

Before the protocol fix, the loop prompt made opencode try unavailable native tools like `read_node`, which produced invalid tool attempts and blocked proper graph use.

After switching the opencode path to plain-text `<graph_action>{...}</graph_action>` blocks:

| Mode | finalized | steps | tool calls | Notes |
|---|---:|---:|---:|---|
| before fix | false | 6 | 0 | invalid native tool attempts in raw trace |
| after fix | true | 2 | 6-8 | correct graph-backed loop answer |

This proves the current backend still uses tools in loop mode; the earlier failure was a protocol mismatch, not a loss of tool-use ability.

### Signature relation layer

Artifacts:

```text
artifacts/signature_live_bias_compare_broad/20260528_relations_v1/
```

What is now verified:

- sibling variants persist explicit `overlaps` / `entails` / `contradicts` relation records
- contradiction relations can mark a family contested
- entailment direction points from a more specific variant to a more general variant
- the relation layer is behavior-neutral on the validated live sweep so far

Broad comparison summary:

| Comparison | Result |
|---|---|
| old live (`v4_actions`) -> new live (`relations_v1`) | 10 same, 0 regressions |
| new shadow -> new live | 1 improved, 0 regressed |

Important nuance: the one new shadow-vs-live improvement remained the astronaut paraphrase, but live bias was still disabled there. So that delta came from normal loop/runtime variance, not from the new relation layer itself.

### Relation-aware propagation sweep

Artifacts:

```text
artifacts/signature_live_bias_compare_broad/20260528_relprop_v1/
```

What changed:

- ranking now uses effective bias, not just direct bias
- contradiction relations gate normal retrieval tiers more aggressively
- live bias skips contested families even when they still have graph-backed support nodes

Observed:

- relation-only shadow -> propagation shadow: no behavior changes
- relation-only live -> propagation live: one nominal improvement, no regressions, but no broad quality/step gain
- propagation shadow -> propagation live: one improvement, zero regressions, still mostly neutral

Interpretation:

```text
relation-aware propagation is structurally correct,
but current live-bias selectivity is still the limiting factor
```

So the next practical optimization target is not deeper propagation. It is making live bias more selective for task families like `direct_judgment`, where strong baseline anchors often make extra bias unnecessary.

---

## Current Bottlenecks

1. **Slot schemas are still too coarse for some shortcuts.** Dijkstra mechanism answers need `nonnegative_precondition` as a required slot.
2. **Constraint coverage needs to become a checker-enforced slot.** The leaderboard graph answer missed explicit `100ms` propagation even though the prompt required it.
3. **Graph learning must generalize through task signatures.** Exact-query solved capsules are useful but too narrow.
4. **Hard design tasks still cost many tool calls.** The graph path is auditable but not yet cheaper than raw big-pickle on open-ended architecture prompts.
5. **Graph edits are opt-in.** A run can produce many edits while `graph_edits_applied=False`, meaning the graph does not actually learn from that run.
6. **Direct-judgment family signatures are still too coarse.** The live-bias gates are safer now, but family canonicalization can still over-merge unrelated explanations.
7. **Signature relations are not yet load-bearing.** `overlaps` / `entails` / `contradicts` are now stored and projected, but they do not yet drive propagation, promotion, or live reranking directly.
8. **Scoped patch validation is heuristic.** It catches missing evidence, duplicate IDs, missing endpoints, broad control rules, and low-relevance strategy evidence, but it is not yet a full contradiction/provenance checker.

---

## File Map

### Core

| File | Purpose |
|---|---|
| `answerer_v4.py` | Main pipeline, micro-finalize enforcement, full tool loop, packet |
| `reasoning/micro_controller.py` | Rules-first subgoal knownness controller |
| `reasoning/task_classifier.py` | Optional task complexity routing |
| `reasoning/activation.py` | Graph task frame / constraints / pitfalls |
| `reasoning/retrieval_boost.py` | Anchor retrieval and failure-pattern boost |
| `reasoning/session_subgraph.py` | Audit-logged session state |
| `reasoning/budgets.py` | Budget accounting |

### Learning

| File | Purpose |
|---|---|
| `reasoning/post_processing.py` | Deterministic learning extraction and graph edits |
| `reasoning/scoped_edits.py` | Scoped patch generation and validation |
| `reasoning/reflection.py` | LLM reflection |
| `reasoning/graph_editor.py` | Reflection edit validation/application |
| `reasoning/semantic_dedupe.py` | Duplicate detection |
| `reasoning/edit_judge.py` | LLM quality gate for edits |
| `reasoning/graph_health.py` | Graph structural health |
| `reasoning/consolidation.py` | Promotion decisions |
| `reasoning/distillation_corpus.py` | SFT corpus writer |

### Scripts and Evidence

| File | Purpose |
|---|---|
| `run_systemic_thinking_comparison.py` | Raw baseline vs graph evidence dump |
| `run_scoped_edit_lab.py` | Offline scoped graph-edit audit lab |
| `scripts/process_session.py` | Offline reflection runner |
| `scripts/verify_feedback_loop.py` | Feedback-loop proof runner |
| `_test_v4_mock.py` | Mock regression tests |
| `artifacts/paper_full_dumps/*` | Full Dijkstra run dumps |
| `artifacts/systemic_thinking_comparison/*` | Full baseline-vs-graph dumps |
| `artifacts/scoped_edit_lab/*` | Scoped graph-edit validation reports |

---

## Deployment Target

The final product is still:

```text
local 4B reasoner
  + graph memory
  + micro-controller / decision network
  + small embedder
  + no cloud dependency
```

`opencode/big-pickle` is the teacher/data-collection backend. V4's job is to collect trajectories, tool-use behavior, graph edits, and control traces that can be distilled into the smaller deployment model.

---

## Selective Direct-Judgment Live Bias

The live signature-bias layer now has a stricter policy for `direct_judgment`.

Previous behavior:

```text
if a supported solved_subgoal family matched at all,
enable live bias
```

That was safe, but too eager. It kept prepending direct-judgment anchor nodes
even when the baseline retrieval was already strong, which added latency
without reducing steps.

Current behavior:

```text
algorithm_applicability:
  keep supported solved_subgoal live bias

direct_judgment:
  reject already-strong top-baseline cases
  only allow live bias for ambiguous multi-support / dense-support matches
```

Implementation notes:

- `build_shadow_report(...)` now carries `baseline_rank` into adjusted rows.
- `_passes_live_bias_relevance_gate(...)` now uses:
  - `baseline_rank`
  - `baseline_score`
  - anchor lexical mass
  - support-node count
- this means the controller can ask:

```text
is this family merely relevant,
or is it likely to help enough to justify a live anchor override?
```

On the current broad paired sweep:

- older relation-propagation pair:
  - `live_bias_applied_count = 9`
  - `mean_elapsed_delta = +0.53`
- selective gate pair:
  - `live_bias_applied_count = 3`
  - `mean_steps_delta = -0.1`
  - `mean_tool_calls_delta = -0.1`
  - `mean_elapsed_delta = -5.54`

So the architecture lesson is:

```text
live bias should optimize for helpfulness, not just relevance
```

---

## Explicit Direct-Judgment Task Signatures

Direct-judgment signatures are no longer always opaque hashes.

The controller now emits explicit semantic task signatures for the two main
measured physics families:

- `direct_judgment.sound_requires_medium_vs_light_vacuum`
- `direct_judgment.refraction_changes_speed_not_frequency`

Why this matters:

- family labels become more interpretable in the signature index
- paraphrases map to the same family more reliably
- later promotion / retrieval / audit reports are easier to reason about

Examples:

```text
Why can light travel through space but sound cannot?
If astronauts can see sunlight in space, why can't they hear it there?
  -> direct_judgment.sound_requires_medium_vs_light_vacuum

Why does a prism bend light but not change the light's frequency?
Why doesn't refraction change the frequency of light?
  -> direct_judgment.refraction_changes_speed_not_frequency
```

This is not the final canonicalization story yet. Other direct-explanation
families still fall back to topic-term signatures, but the measured benchmark
families now have stable semantic keys instead of digest-only identifiers.

---

## Promotion Workflow

The signature-memory layer now has an explicit deterministic lifecycle:

```text
blocked
  -> review
  -> supported
```

This is intentionally conservative.

Promotion is **not** based on recurrence alone. It is based on a bounded
combination of:

- distinct session count
- distinct question count
- distinct evidence-fingerprint count
- success-event count
- scoped patch status
- contradiction pressure
- low-relevance penalties
- answer-gate rewrite penalties

Current semantics:

- `solved_subgoal`
  - starts as graph-backed supported memory
  - can move from `review` to `supported` automatically when repeated, clean, and stable
  - remains `normal` retrieval tier when healthy

- `strategy`
  - starts as provisional memory
  - can move into `review` when repeated stable use is observed
  - remains `gated` even after review, so it does not behave like direct fact memory

- `provisional_claim`
  - stays provisional unless a future stronger evidence path is added
  - does not auto-promote to `supported`

Negative guards still win:

- `contradicted`
- `scoped_patch_needs_review`
- `scoped_patch_reject`
- low-relevance strategy reinforcement
- answer-gate rewrites for provisional claims

When a transition happens, the system now emits typed transition events:

- `promoted_to_review`
- `promoted_to_supported`

Those transitions are persisted in:

- the per-run `signature_events`
- `signature_stats_update`
- the global `signature_events.jsonl`

So promotion is now an auditable first-class behavior rather than an implicit
score threshold hidden inside the index.

---

## Broader Direct-Judgment Canonicalization

The original explicit direct-judgment family keys were useful, but still too
dependent on the original benchmark wording. The controller now recognizes the
same two physics families using broader concept bundles instead of near-exact
phrase checks.

Current explicit direct-judgment semantic families:

- `direct_judgment.sound_requires_medium_vs_light_vacuum`
- `direct_judgment.refraction_changes_speed_not_frequency`

These now match not only the original prompts, but also paraphrases like:

```text
In empty space, why could you see a flash but not hear it?
When a laser enters water, why doesn't its frequency change?
Light bends when entering glass. Why is the frequency still unchanged?
```

Architecturally this matters because:

- solved-subgoal family IDs become more semantically stable
- family labels in the signature index become interpretable
- shadow retrieval evaluation measures concept reuse instead of surface-form reuse

Unrelated direct-judgment questions still fall back to hashed topic signatures,
so the broader matching remains bounded to the known semantic families.

---

## Expanded 14-Case Shadow Bank

The shadow evaluation bank is now larger and more diagnostic.

Current broad-v2 bank:

- `5` vacuum/sound/light-space cases
- `5` refraction/frequency cases
- `4` Dijkstra negative-edge applicability cases

The most important artifact is:

```text
artifacts/signature_live_bias_compare_broad_v2/20260528_canonical_v1/shadow_eval.json
```

Key metrics from that run:

- `family_hit_at_1_adjusted = 0.785714`
- `family_hit_at_1_baseline = 0.142857`
- `family_hit_at_3_adjusted = 1.0`
- `delta_family_mrr = 0.392857`

So the family/variant memory layer is now much more clearly helping on the
harder paraphrase bank in **shadow mode**.

This is important because it narrows the remaining problem:

```text
shadow family matching is now strong enough;
the remaining gap is live eligibility and timing
```

---

## Direct-Judgment Live-Bias Safety Gate

The live bias path for `direct_judgment` is now shaped by two ideas:

1. helpfulness-aware gating
2. family-question semantic compatibility

Helpfulness-aware gating already existed in the selective broad-sweep work:

- reject already-strong baseline answers
- allow only cases where a live anchor override is plausibly helpful

The new follow-up adds a stricter compatibility guard for the single-support
escape hatch:

```text
if a direct-judgment family has only one strong support node,
it may still become live-bias-eligible,
but only if the question semantically matches that family
```

That prevents a family like:

- `direct_judgment.sound_requires_medium_vs_light_vacuum`

from incorrectly activating on:

- prism / refraction / unchanged-frequency questions

The main architecture lesson is:

```text
for direct_judgment tasks, live bias needs both
  semantic family alignment
and
  helpfulness evidence
```

Current state after the fix:

- the prism cross-family leak is gone
- `algorithm_applicability` live bias still behaves correctly
- the hard vacuum paraphrases still do not improve live, because the solved-subgoal
  family is not live-eligible early enough inside the same sweep

So the next architecture step is **not** broader semantic matching. It is
making strong direct-judgment solved-subgoal memory available earlier without
reopening cross-family leakage.
