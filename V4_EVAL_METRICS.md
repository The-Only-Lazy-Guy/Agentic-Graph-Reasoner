# V4 Eval Metrics

This document defines the exact evaluation stack for the current V4 graph environment.

The point is to separate:

1. machinery correctness
2. shadow retrieval quality
3. real task behavior

Layer 1 is already covered by unit and mock tests.

This document defines the exact metrics for:

- Layer 2: shadow retrieval eval
- Layer 3: behavior eval

The definitions below are intentionally tied to the artifacts we already produce.

---

## Artifact Sources

Current scripts and their role:

- `run_v4_difficulty_sweep.py`
  - per-case task behavior across trivial / medium / hard
  - writes `packet.json`, `summary.json`, `report.md`
- `run_repeat_learning_experiment.py`
  - same-question run1 vs run2 learning delta
  - writes `run_1_packet.json`, `run_2_packet.json`, `summary.json`
- `run_systemic_thinking_comparison.py`
  - raw baseline vs graph pipeline on one harder synthesis task
  - writes `summary.json`, `graph_packet.json`, baseline raw output

Important packet fields now available:

- `signature_candidates`
- `signature_events`
- `signature_stats_update`
- `signature_shadow_report`
- `signature_graph_projection`

These are the required sources for Layer 2 and the learning-related subset of Layer 3.

---

## Layer 2: Shadow Retrieval Eval

### Goal

Measure whether the signature layer would improve retrieval ordering **without changing live retrieval yet**.

This layer must answer:

- did the right family move up?
- did the right variant move up?
- did unsafe / provisional / contested memory stay controlled?
- did candidate matching collapse true revisions and preserve meaningful siblings?

### Unit of evaluation

One labeled question case.

Each case must define a gold signature target set.

### Label file schema

Recommended JSON shape per case:

```json
{
  "id": "dijkstra_negative_edge_applicability",
  "question": "Can Dijkstra be trusted with one negative edge?",
  "task_family": "algorithm_applicability",
  "gold_signature_family_ids": [
    "sigfam_solved_subgoal.algorithm_applicability_dijkstra_negative_edge_weights_validity"
  ],
  "gold_signature_variant_ids": [],
  "unsafe_family_ids": [],
  "notes": "Any correct Dijkstra-negative-edge family counts."
}
```

Rules:

- `gold_signature_family_ids` is required
- `gold_signature_variant_ids` is optional
- for early evals, family labels are more stable than exact variant labels

### Required artifact fields

From `packet.json`:

- `signature_shadow_report.baseline_ranking`
- `signature_shadow_report.adjusted_ranking`
- `signature_shadow_report.baseline_top_k`
- `signature_shadow_report.adjusted_top_k`
- `signature_shadow_report.rank_movers`
- `signature_stats_update.variant_resolution_counts`
- `signature_stats_update.relation_counts`
- `signature_graph_projection`

Recommended label extensions for matching/revision metrics:

- `matching_expectation.semantic_type`
- `matching_expectation.expected_variant_resolution`
- `matching_expectation.expected_family_resolution`
- `matching_expectation.should_match_existing_family`
- optional `matching_expectation.target_family_ids`
- optional `matching_expectation.target_source_node_ids`

### Ranking metrics

For each labeled case:

#### 1. `family_hit_at_k_baseline`

Definition:

`1` if any item in `baseline_top_k[:k]` has `family_id in gold_signature_family_ids`, else `0`.

Compute for:

- `k = 1`
- `k = 3`
- `k = 5`

#### 2. `family_hit_at_k_adjusted`

Same as above, but on `adjusted_top_k`.

Primary metric:

- `family_hit_at_1_adjusted`

#### 3. `family_rank_baseline`

Definition:

The smallest 1-indexed rank of any gold family in the full baseline ranking.

If absent, use `INF`.

Implementation note:

For the current scorer JSON we store `null` when the ranking is incomplete.
When the full ranking is present but the gold family is absent, the scorer
treats it as rank `len(ranking) + 1` for delta/MRR aggregation while still
reporting MRR `0`.

#### 4. `family_rank_adjusted`

Same on the adjusted ranking.

#### 5. `family_rank_delta`

Definition:

`family_rank_baseline - family_rank_adjusted`

Interpretation:

- positive = the gold family moved up
- zero = unchanged
- negative = moved down

Primary aggregation:

- mean `family_rank_delta`
- median `family_rank_delta`
- win rate: `% of cases where family_rank_delta > 0`

#### 6. `family_mrr_baseline`

Definition:

`1 / family_rank_baseline` if present, else `0`

#### 7. `family_mrr_adjusted`

Same on adjusted ranking.

Primary metric:

- `delta_family_mrr = family_mrr_adjusted - family_mrr_baseline`

#### 8. `variant_hit_at_k_adjusted`

Only if `gold_signature_variant_ids` is non-empty.

Definition:

`1` if any item in adjusted top-k has `variant_id in gold_signature_variant_ids`.

Use:

- `k = 1`
- `k = 3`

### Safety metrics

#### 9. `contested_family_top1_rate`

Definition:

Fraction of cases where `adjusted_top_k[0].family_contested == true`.

Desired:

- low

Rationale:

Contested families should not dominate top-1 casually.

#### 10. `audit_only_or_provisional_top1_rate`

Definition:

Fraction of cases where the adjusted top-1 candidate has:

- `retrieval_tier == "audit_only"`
  or
- `epistemic_status == "provisional"`

Desired:

- near zero on known-answer tasks
- non-zero allowed on open design tasks, but should be explicitly tracked

#### 11. `unsafe_top3_rate`

Definition:

Fraction of cases where any candidate in adjusted top-3 has:

- `family_id in unsafe_family_ids`

If `unsafe_family_ids` is empty, skip this metric.

### Matching / revision metrics

These measure whether the signature layer is grouping memories correctly.

#### 12. `equivalent_revision_precision`

Manual or labeled subset metric.

Definition:

Among cases labeled as “same underlying memory, revised wording”, fraction where:

- `variant_resolution == "equivalent_revision"`

#### 13. `sibling_variant_precision`

Manual or labeled subset metric.

Definition:

Among cases labeled as “same family, meaningfully different variant”, fraction where:

- `variant_resolution == "sibling_variant"`

#### 14. `new_family_false_split_rate`

Definition:

Fraction of cases where the correct outcome should have matched an existing family, but:

- `family_resolution == "new_family"`

Desired:

- low

### Layer 2 aggregate score

Recommended primary dashboard:

- `family_hit_at_1_adjusted`
- `delta_family_mrr`
- `median_family_rank_delta`
- `contested_family_top1_rate`
- `audit_only_or_provisional_top1_rate`
- `equivalent_revision_precision`
- `sibling_variant_precision`

### Layer 2 pass criteria for phase 1

Before enabling live reranking for a task family:

- `family_hit_at_1_adjusted >= family_hit_at_1_baseline`
- `delta_family_mrr > 0`
- `median_family_rank_delta >= 0`
- `contested_family_top1_rate <= 0.10` on known-answer tasks
- `audit_only_or_provisional_top1_rate <= 0.05` on known-answer tasks
- `equivalent_revision_precision >= 0.85` on labeled revision subset
- `sibling_variant_precision >= 0.80` on labeled sibling subset

---

## Layer 3: Behavior Eval

### Goal

Measure whether the whole graph environment helps real task behavior.

This layer must answer:

- are answers better?
- are runs shorter?
- are tool calls better targeted?
- does run 2 get easier than run 1?
- does graph memory help more than raw baseline?

### Eval families

Use 3 current experiment types:

#### A. Difficulty sweep

Script:

```powershell
python run_v4_difficulty_sweep.py
```

Use for:

- trivial
- medium
- hard

#### B. Repeat learning

Script:

```powershell
python run_repeat_learning_experiment.py --question "..."
```

Use for:

- same question run1 vs run2
- graph-learning effect

#### C. Systemic comparison

Script:

```powershell
python run_systemic_thinking_comparison.py --question "..."
```

Use for:

- raw baseline vs graph environment

### Required behavior metrics

### Correctness / quality

#### 1. `answer_score`

Per-case manual or checker-based score:

- `2` = correct, grounded, complete enough
- `1` = partially correct or under-specified, but not materially wrong
- `0` = incorrect, unsupported, or misses core requirement

This is the primary task-quality metric.

Aggregate:

- mean `answer_score`
- `% score == 2`
- `% score >= 1`

#### 2. `unsupported_claim_flag`

Definition:

`1` if final answer contains material unsupported claims.

Use current gates/checkers where available.

Aggregate:

- unsupported claim rate

#### 3. `slot_coverage_rate`

Definition:

From `slot_fill_stats`:

`filled_required / total_required`

Aggregate:

- mean slot coverage
- `% full coverage`

### Efficiency

#### 4. `steps_used`

Definition:

`packet.steps`

Aggregate:

- mean
- median
- p90

#### 5. `tool_call_count`

Definition:

`packet.tool_call_count`

Aggregate:

- mean
- median
- p90

#### 6. `controller_call_count`

Definition:

`packet.controller_call_count`

Aggregate:

- mean
- median

#### 7. `elapsed_sec`

Definition:

`packet.elapsed_sec`

Aggregate:

- mean
- median
- p90

#### 8. `efficiency_score`

Optional derived metric for same-task comparisons only:

```text
efficiency_score =
  0.4 * normalized_steps
  + 0.3 * normalized_tool_calls
  + 0.3 * normalized_elapsed
```

Lower is better.

Use only within the same benchmark set, not across unrelated tasks.

### Controller / reuse behavior

#### 9. `reuse_success_flag`

Definition:

`1` if:

- `execution_mode` is one of
  - `micro_controller_finalize`
  - `micro_controller_reuse`
  or
- `subgoal_reuse_count > 0`

Aggregate:

- reuse success rate

#### 10. `shortcut_precision`

Definition:

Among runs with shortcut / finalize reuse:

fraction where:

- `answer_score == 2`
- `unsupported_claim_flag == 0`

This is extremely important. Fast wrong shortcuts are a failure.

#### 11. `fallback_rate`

Definition:

`packet.controller_fallback_used`

Aggregate:

- fallback rate by task family

Desired:

- low on known-answer tasks
- acceptable on hard design tasks

### Learning / run2 improvement metrics

These apply to `run_repeat_learning_experiment.py`.

#### 12. `repeat_step_delta`

Definition:

`steps_run2 - steps_run1`

Desired:

- negative

#### 13. `repeat_tool_delta`

Definition:

`tool_call_count_run2 - tool_call_count_run1`

Desired:

- negative

#### 14. `repeat_elapsed_delta`

Definition:

`elapsed_sec_run2 - elapsed_sec_run1`

Desired:

- negative

#### 15. `repeat_answer_score_delta`

Definition:

`answer_score_run2 - answer_score_run1`

Desired:

- non-negative

#### 16. `repeat_shortcut_upgrade_flag`

Definition:

`1` if run2 uses a more direct reuse mode than run1, for example:

- run1 = `loop`
- run2 = `micro_controller_finalize` or `micro_controller_reuse`

#### 17. `repeat_signature_reuse_flag`

Definition:

`1` if run2 has either:

- lower `family_rank_adjusted` for the gold family than run1 in shadow eval, or
- `variant_resolution == "equivalent_revision"` for the relevant memory, or
- fewer new variants created while still answering correctly

This is the clearest “the graph learned something reusable” signal in the current architecture.

### Baseline vs graph metrics

These apply to `run_systemic_thinking_comparison.py`.

#### 18. `graph_vs_baseline_answer_score_delta`

Definition:

`answer_score_graph - answer_score_baseline`

Primary quality comparison.

#### 19. `graph_vs_baseline_unsupported_delta`

Definition:

`unsupported_claim_flag_graph - unsupported_claim_flag_baseline`

Desired:

- negative

#### 20. `graph_vs_baseline_latency_ratio`

Definition:

`elapsed_sec_graph / elapsed_sec_baseline`

Use only with answer quality shown alongside it.

Interpretation:

- graph can be slower if quality is materially higher
- graph should be faster only when quality is not worse

### Layer 3 aggregate dashboards

### Dashboard A: Known-answer tasks

Use trivial + medium + repeat-known tasks.

Primary metrics:

- `% answer_score == 2`
- mean `steps_used`
- mean `tool_call_count`
- mean `elapsed_sec`
- `reuse_success_rate`
- `shortcut_precision`
- `repeat_step_delta`
- `repeat_tool_delta`

### Dashboard B: Hard synthesis tasks

Use hard difficulty + systemic comparison.

Primary metrics:

- mean `answer_score`
- unsupported claim rate
- mean slot coverage
- fallback rate
- graph vs baseline answer score delta
- graph vs baseline unsupported delta

### Phase 1 acceptance criteria

Before distillation / small-model eval:

#### Known-answer acceptance

- `% answer_score == 2 >= 0.80`
- `shortcut_precision >= 0.90`
- median `repeat_step_delta <= 0`
- median `repeat_tool_delta <= 0`
- repeat runs must not reduce answer quality

#### Hard synthesis acceptance

- graph answer score must be `>=` raw baseline on average
- graph unsupported claim rate must be `<=` raw baseline
- hard tasks may remain slower, but must show better grounding and slot coverage

---

## Minimal next implementation

To make these metrics runnable with less manual work, the next practical additions should be:

1. a label file for Layer 2 gold families / variants
2. a small scorer script that reads `packet.json` / `summary.json` artifacts
3. a lightweight answer rubric sheet for Layer 3 manual scoring

That is enough to start collecting credible evidence without waiting for live retrieval integration.
