# READ_THIS — GRR: Graph Recursive Reasoner (2026-07-13)

> At-a-glance dump of the latest session (raw numbers, decisions, repro commands).
> Updated each working session. Branch: fix/swe-slot-plan-gate-real-file.

## What GRR is
A tiny OWNED reasoner (not a language model) whose vocabulary + memory ARE a verified, self-compressing
graph. Content lives symbolically in the graph (executed); strategy lives in the latent. Rewarded for
its own future capability, not for matching us. Core bet — **memory is load-bearing only for a model
FORCED to compose** — is PROVEN and EMBODIED.

## The stack — all built + real-tested this session (no stubs-echoing-themselves)
| module | what | proof |
|---|---|---|
| algo_quality (GRR-1) | fuzz-generality store-gate | rejected 31% overfit in the real harvest |
| algo_graph_reason.Budget (GRR-2) | budget in wake loop | budget really halts the solve |
| algo_capability (GRR-3) | counterfactual Δcapability | real ablation drops solve; coupled to fuzz bar |
| algo_abstract (GRR-4) | MUTATE + ABSTRACT | mutate repairs 25→100%; str_dp2 fuzz-equiv + MDL-compresses |
| algo_sleep (GRR-5) | gated compress + prune | lossy rewrite VETOED by fuzz-equivalence |
| resolve_deps (GRR-5b) | transitive dep graph-walk | abstractions run + get credited |
| algo_composed (GRR-6.1) | compose-forced solver | **str_dp2: Δ0 (free 4B) → TOP atom Δ+0.50** |
| algo_trm_compose (GRR-6.2/3) | TRM policy + realizer | trained from scratch, drives compose-forced solve |
| algo_dsl + algo_dsl_trm (5b) | combinator DSL + program decoder | 86% synthetic / 100% mpnet |

## Key numbers (real 4B / real mpnet)
```
cross-attend adapter (earlier): compose +43% @200tok, +5% @400tok  = AMORTIZATION, not capability
Δcapability (hard graph, general): only lcs_length/edit_distance load-bearing; rest replaceable
str_dp2 under compose-forcing: Δ+0.50, TOP atom (vs Δ0 free-form 4B)  <-- THE thesis
DSL program decoder (mpnet): 100% held-out INSTANCES (base + dp2 graphs)
compo-gen (leave-one-family-out, mpnet): 0/66 = 0%  <-- imitation is RECALL, not reasoning
RL + dense reward: DISCOVERED FILTER(is_prime)->MAP(digit_sum) for an UNSEEN family (structure!),
                   aggregator max-vs-sum not finalized; HIGH VARIANCE / unstable
```

## Honest frontier
- Imitation → RECALL (memorizes family→program; 0% compo-gen on synthetic AND mpnet).
- RL (GRPO on dense agreement-fraction verify-reward) → the recall→reasoning mechanism: it DISCOVERED a
  novel composition for an unseen family via search+verify. But UNSTABLE (another run wandered).
- Open research: curriculum (short→long, forced novel op-combos) + stable RL (entropy/KL-to-warmstart,
  more seeds) + sharper reward on weakly-distinguished picks + maybe compositional decoder bias.

## Repro (molab, no 4B — mpnet + tiny nets, minutes)
```
python -m v5.runtime.algo_composed --selftest                    # thesis: str_dp2 load-bearing
python -m v5.runtime.algo_dsl_trm --train --hard --graph graphs/algo_reason_hard.json   # 100% (recall)
python -m v5.runtime.algo_dsl_trm --compo-gen --hard --graph graphs/algo_reason_hard.json  # 0% = recall
# every module: python -m v5.runtime.<algo_quality|algo_capability|algo_abstract|algo_sleep|algo_dsl|algo_composed|algo_trm_compose|algo_dsl_trm> --selftest
```

## Files (all v5/runtime/)
algo_quality · algo_capability · algo_abstract · algo_sleep · algo_composed · algo_trm_compose ·
algo_dsl · algo_dsl_trm (+ RL: train_rl/fast_reward/sample). Reuses: algo_graph_mg (MGRetriever.resolve_deps),
algo_compose_tasks (_REF/_NEEDS/ALL_ATOMS/gen), graph_grower, subgraph/gnn_encoder/goal_encoder (read stack).
Trained: artifacts/grr6_trm.pt, artifacts/grr6_dsl.pt.

## Next (own push): curriculum + stable RL to turn RECALL → robust REASONING.
