# v4 — Reasoning-as-Traversal (latent-only)

**Status:** design + implementation in progress, 2026-07-08.
**Principle:** The retrieval trajectory itself IS the derivation. h_K at hop k determines
what's retrieved at hop k+1 — entirely in mpnet-768 latent space, no LM generation between
hops. The LM only generates the final answer from accumulated records.

**Builds on v3** (compose_pool proven: refiner beats cosine +0.95 on hard retrieval).
**Fixes v3's gap:** compose was 2 parallel independent hops; v4 adds CONDITIONAL hops
where you can't know you need B until you've retrieved A.

## 1. Why latent-only (Option B, chosen over text-mediated)

Text-mediated (generate next-query text with LM between hops) would work immediately but
isn't new — it's multi-shot RAG with extra LM calls. The novel claim is that the latent
trajectory itself can encode the conditional dependency: h_k (in mpnet space) carries
enough information about the content of the k-th hop's record to guide hop k+1's retrieval
toward the conditionally-correct target, without any token-space LM involvement.

This works when the embedding space naturally correlates keywords across records:
- config.py contains PREFERRED_STRATEGY = "nash" → its embedding encodes "nash"
- nash.py contains "nash" in function names/doco → its embedding also encodes "nash"
- The spec says only "the established strategy preference" → no "nash" in spec embedding
- Hop 1: spec → ranker retrieves config.py (closest match to "strategy" + "preference")
- h_1 refined toward config_emb → h_1 carries the "nash" signal
- Hop 2: search_ctx(h_1) → naturally ranks nash.py above maxmin.py (keyword overlap)
- The conditional dependency is resolved PURELY through latent space geometry

## 2. Architecture

```
                    ┌─── TraversalRanker (latent-only, no LM) ───────────┐
                    │                                                    │
                    │  h = embed(goal)                                   │
                    │  for hop in 1..H:                                  │
                    │    cand = search_ctx(h, pool_k, exclude=seen_ids)  │
                    │    h = Refiner.Net(h, cand_ctx, cand_feats, K=1)   │
                    │    records = top_k(cand, h, k_impl)                │
                    │    all_records += records                          │
                    │    if gap_detector.should_stop(h, hop): break      │
                    │                                                    │
                    │  return (all_records, hops, final_h)               │
                    └─────────────────────┬──────────────────────────────┘
                                          │ accumulated records (text payload)
                                          ▼
                    ┌─────────────────────────────────────────────────────┐
                    │  LM (single generation from payload)                │
                    │  derive answer = comp(p*(1+tax+fee))                │
                    └─────────────────────────────────────────────────────┘
                                          │ answer + sandbox result
                                          ▼
                    ┌─────────────────────────────────────────────────────┐
                    │  Per-subsystem rewards:                             │
                    │    ranker: per-hop retrieval hit-rate               │
                    │    gap: correct stop decision                       │
                    │    LM: grounded+solves+unique (derive_reward)       │
                    └─────────────────────────────────────────────────────┘
```

### 2.1 Key difference from v3

v3 ranker: ONE refinement over ONE static pool → top-k readout.
v4 traversal: SEQUENTIAL hops, each with its OWN pool (excluding prior finds).
h_K after hop k IS the query for hop k+1 — no text re-embedding.

### 2.2 How Refiner.Net is used

v3 uses Refiner.Net with K=3-4 steps over a single pool. v4 uses it at K=1 per hop
(one refinement step per pool). The v1 ablation data (lggn_refine.py) shows K=1 already
beats raw cosine by ~0.03-0.05 — enough for a per-hop signal.

The `feat_proj` (feature fusion adapter) from v3 is preserved: same structural features
(same_file, name_mentioned, kind onehot, concept_confidence) are fused into each hop's
ctx before Refiner.Net sees it.

### 2.3 Gap Detector

A lightweight MLP(h, hop) → P(stop), trained with REINFORCE:

```
gap_net = nn.Sequential(nn.Linear(d + 1, 64), nn.ReLU(), nn.Linear(64, 1))
action = sample(bernoulli(P(stop)))
reward = task_success - 0.1 * extra_hops (if > optimal) - 0.5 * premature_stop (if < optimal)
```

Optimal hop count = number of distinct source_session_idxs for the current session.
Per-subsystem credit: gap detector's reward is CONDITIONAL on retrieval success — if
retrieval hit-rate is 0, the gap detector isn't penalized (wrong stop decision doesn't
matter if the records weren't there to find).

## 3. Conditional Benchmark: `preference` archetype

New archetype in project_gen.py. Structure:

| Session | Kind | File | Key content | source_session_idx |
|---------|------|------|-------------|-------------------|
| 0 | create | config.py | PREFERRED_STRATEGY = "nash" (or "maxmin") + get_strategy() | — |
| 1 | create | nash.py | nash_equilibrium(payoffs) — NE solver | — |
| 2 | create | maxmin.py | maxmin_solution(payoffs) — maxmin solver | — |
| 3 | create | utils.py | helper functions, distractor | — |
| 4 | compose | solver.py | solve(payoffs) — uses the project's PREFERRED_STRATEGY | [0, conditional on 0 → 1 or 2] |

Session 4's spec: "Implement solve(payoffs) that returns the game outcome using this
project's established strategy preference."

The words "nash" and "maxmin" appear ONLY in the respective files, NOT in the spec.
Config.py contains one of these keywords (the correct one). This creates the conditional
chain:
- Retrieve config.py → discover PREFERRED_STRATEGY → this tells you which solver to use
- Retrieve the right solver (nash.py or maxmin.py) → use it in solve()

**Why this tests latent traversal:**
- Hop 1 must retrieve config.py (not nash.py or maxmin.py) — the spec's embedding is in
  generic "strategy preference" space; config.py is the most precise match for "config"
- h_1 after refinement points near config_emb, which carries the keyword signal
- Hop 2: search_ctx(h_1) naturally ranks the matching solver above the non-matching one
  because config_emb's keyword overlap biases the cosine similarity
- The ranker at hop 2 can use both embedding similarity AND structural features to pick
  the right file

## 4. Per-subsystem RL

### 4.1 Subsystems and reward signals

| Subsystem | Reward | Signal | Train via |
|-----------|--------|--------|-----------|
| Traversal ranker (hops) | Per-hop hit-rate: did hop k find one of the needed source records? | 0..1 per hop, averaged | REINFORCE or contrastive (existing v3 loss adapted for multi-hop) |
| Gap detector | Correct stop: stop at optimal hop count ±1? | +1 correct, -0.5 premature (missing sources), -0.1 per extra hop | REINFORCE |
| LM (derive) | Grounded + solves + unique (from derive_reward.py) | +1.5 / +1.0 / 0.0 / -1.0 | GRPO (existing from derive_rl.py) |

### 4.2 Clean credit assignment

The hierarchy:
```
Ranker → Gap Detector → LM

Reward_ranker = PER_HOP_HIT_RATE      (not conditioned on task success)
Reward_gap    = STOP_CORRECTNESS       (conditioned on ranker success)
Reward_lm     = DERIVE_REWARD          (conditioned on both ranker + gap success)
```

A solve failure with perfect retrieval → LM trains, ranker doesn't.
A solve failure with bad retrieval → ranker trains, LM doesn't.
A solve failure with wrong stop → gap detector trains, ranker doesn't.

This prevents subsystem-A from exploiting subsystem-B's reward. If the ranker delivers
wrong records, task failure is attributed to retrieval, not to the LM's composition ability.

### 4.3 Implementation

Rewards are computed in project_loop.py after each session, stored per-subsystem in
results. Training happens:

- Ranker: offline after data collection (as in v3 build_ranker_reprs + _train_ranker)
- Gap detector: online REINFORCE (small MLP, fast to train per-batch)
- LM: offline GRPO (as in derive_rl.py)

For local 0.5B experimentation, the LM is frozen; only the ranker + gap detector train.

## 5. Files changed / created

### New files in v5/:
- `v5/runtime/traversal_ranker.py` — TraversalRanker class
- `v5/runtime/gap_detector.py` — GapDetector + RL training
- `v5/v4_DESIGN.md` — this document

### Modified files in v5/:
- `v5/runtime/project_gen.py` — add `preference` archetype
- `v5/runtime/project_loop.py` — wire traversal + gap detector + per-subsystem rewards
- `v5/runtime/traversal_ranker.py` — build_traversal_reprs + _train_traversal_ranker

### Selftests (no GPU/network):
- `traversal_ranker --selftest` — synthetic multi-hop ranking task
- `gap_detector --selftest` — synthetic stop-decision learning task
- `project_gen --selftest` — new `preference` archetype gold chain + withholding verification
- `project_loop --smoke` — end-to-end with traversal mode, 0.5B

## 6. Local experimentation plan

```
# Selftests
python -m v5.runtime.project_gen --selftest
python -m v5.runtime.traversal_ranker --selftest
python -m v5.runtime.gap_detector --selftest
python -m v5.runtime.project_loop --smoke --model Qwen/Qwen2.5-0.5B

# Build traversal training data
python -m v5.runtime.traversal_ranker --build-reprs --model Qwen/Qwen2.5-0.5B --archetypes preference

# Train traversal ranker + gap detector
python -m v5.runtime.traversal_ranker --train --reprs artifacts/traversal_reprs.npz
python -m v5.runtime.gap_detector --train

# Run with traversal mode
python -m v5.runtime.project_loop --run --arm memory --query-mode traversal --archetypes preference
```

## 7. Measurable gates

| Gate | Condition | Status |
|------|-----------|--------|
| GV1 | preference gold chains pass selftest | not yet |
| GV2 | traversal ranker beats raw cosine on conditional multi-hop | not yet |
| GV3 | gap detector stops at correct hop on held-out chains | not yet |
| GV4 | per-subsystem rewards correctly attribute failures | not yet |
| GV5 | end-to-end solves preference benchmark (dep sessions) | not yet |

## 8. Relationship to v3

v4 does NOT replace v3. The v3 ranker (one-shot, top-k) is still the right solution for
independent multi-source retrieval (compose, compose_pool). v4's traversal is only needed
for CONDITIONAL dependencies where hop 2's target is unknowable without hop 1's content.

In terms of the broader system:
- v3: retrieval amplifier — finds the right records in a noisy pool
- v4: retrieval REASONER — traverses a conditional path through the record graph
- Both use the same Refiner.Net; v4 just wraps it in a multi-hop loop
