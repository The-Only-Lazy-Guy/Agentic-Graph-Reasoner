# V4 Current Plan

**Last updated:** 2026-05-28

This is the current working plan for the backend reasoning environment in `graph_final`.

## Goal

Make the current graph-native environment work reliably enough that a smaller local model can use it well.

Target behavior:

```text
query
  -> identify task family
  -> retrieve the right graph evidence
  -> reuse solved subgoals when safe
  -> fall back to graph-tool reasoning when needed
  -> answer
  -> learn safely without polluting the graph
```

The near-term goal is not fancy latent graph fusion. It is:

```text
safe + auditable + effective graph use
```

## What Is Already Done

### Core reasoning loop

- micro-controller finalize path
- full graph-tool loop
- enforced graph-read requirement before loop answers
- opencode loop protocol fix using plain-text `<graph_action>` blocks

### Learning and graph safety

- post-processing extraction
- scoped graph-edit patches
- edit validation / audit lab
- hard-task evidence gate

### Signature memory

- shadow signature family/variant memory
- typed event stats index
- shadow retrieval eval
- live bias for:
  - `algorithm_applicability`
  - `direct_judgment`
- selective `direct_judgment` live-bias gating that blocks already-strong baseline cases
- explicit direct-judgment task signatures for:
  - `sound_requires_medium_vs_light_vacuum`
  - `refraction_changes_speed_not_frequency`
- broader paraphrase-aware direct-judgment semantic matching for:
  - vacuum / sound / light / space variants
  - refraction / medium-transition / unchanged-frequency variants
- deterministic promotion workflow:
  - repeated stable memory can move to `review`
  - graph-backed solved subgoals can move to `supported`
  - provisional claims do not auto-promote to `supported`

### Relation layer

- explicit `overlaps`
- explicit `entails`
- explicit `contradicts`
- contested-family logic
- bounded one-hop relation-aware score propagation
- contradiction-aware retrieval gating

### Validation

- deep unit/regression coverage for signature memory and live bias
- focused V4 mock tests
- broad live vs shadow sweep comparisons
- paired broad sweep evidence that reduced live-bias applications from `9` to `3`
- paired broad sweep evidence that changed mean live-vs-shadow elapsed delta from `+0.53s` to `-5.54s`
- end-to-end promotion audit showing real `promoted_to_review` / `promoted_to_supported` transition events in session packets
- expanded 14-case labeled shadow bank with harder paraphrases for:
  - vacuum / sound / light in space
  - refraction / unchanged frequency
  - Dijkstra negative-edge applicability
- fresh shadow retrieval evidence on the 14-case bank:
  - `family_hit_at_1_adjusted = 0.7857`
  - `family_hit_at_1_baseline = 0.1429`
  - `family_hit_at_3_adjusted = 1.0`
  - `delta_family_mrr = 0.392857`
- direct-judgment live gate now checks semantic family/question compatibility before allowing single-support bias, which removed the prism <- vacuum cross-family leak
- added a tightly-scoped `review`-tier live path for explicit direct-judgment semantic families, solving the eligibility timing bottleneck without reopening broad strategy leaks

## What Is Still Left

### 1. Decide whether family/variant wrappers should become real graph nodes

Right now:

- they exist in the signature index
- they are projected into a graph-shaped artifact

Still left:

- decide when to write `signature_family` / `signature_variant` into the main graph itself

### 2. Add smarter borderline comparison

Still missing:

- NLI-assisted revision vs sibling-variant comparison
- better handling of borderline semantic overlap

### 3. Add LLM-scored impact for ambiguous events

Right now:

- event buckets are deterministic

Still left:

- let the model score only ambiguous/context-sensitive events (Completed via `_score_ambiguous_event_batch`)
- keep strict schema and backend-controlled numeric mapping

### 4. Qwen 4B benchmark on the stabilized environment

Only after the current environment is solid:

- run Qwen 4B in the same environment
- measure:
  - tok/s
  - tool usage correctness
  - finalize behavior
  - graph reliance
  - answer quality

## Current Order Of Work

This is the recommended order from here:

1. Re-run the same 14-case shadow/live comparison bank to validate the new review-tier fast-track. (Completed)
2. Decide whether family/variant wrappers should become real graph nodes. (Completed: Decided NO)
3. Add smarter borderline comparison. (Completed: judge_edits_batch fallback in _judge_equivalent_vs_sibling)
4. Benchmark Qwen 4B inside this stabilized environment. (IN PROGRESS)

## Current Honest Status

If we count the full architecture plan:

- about **74% complete**

If we count the signature-memory/live-bias track specifically:

- about **92% complete**

The hard part left is not wiring more components.
It is making the live behavior improve in a measurable way without losing safety.

## Related Files

- `v4_PROGRESS.md`
- `v4_ARCHITECTURE.md`
- `V4_EVAL_METRICS.md`
- `reasoning/signature_stats.py`
- `reasoning/tests/test_signature_stats.py`
