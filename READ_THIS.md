# READ_THIS — V5 latest raw results & quick reference

> At-a-glance dump of the latest runs (raw outputs, numbers, repro commands) so
> you don't have to dig through commits/logs. Updated each working session.

**Last updated:** 2026-05-30
**HEAD:** `59a449b` · branch `main`

---

## TL;DR claim boundary

- ✅ V5 trains end-to-end on **real graph states + real LM hidden states, including
  planning**, on a substrate-rich V4 corpus.
- ✅ V5 **generates** end-to-end with the adapter live; injection is numerically
  stable (stays coherent even with untrained projections).
- ✅ Random-init injection is **95% non-catastrophic** over 20 questions (perfect
  1/1 hook control) → Stage 2 starts from a stable injected-generation baseline.
- ✅ **Stage 2 core (synthetic)**: residual gate + 2A/2B trainer learns attention
  routing (plan/evid precision → 1.0) with bounded gated write (~11% of ‖h‖).
- ❌ NOT yet: V5 **generalizes** (corpus is 20 traces → train-fit only).
- ❌ NOT yet: V5 **improves** generation (Stage 2 not yet on the real 1536-d adapter;
  LoRA untrained).
- ⏳ Staged from full Stage 2 spec: KL-vs-base-LM stability loss (use
  `perturbation_baseline` on real LM), explicit head-retention loss, "no-graph"
  negative cases, and real-LM post-Stage-2 catastrophic-rate check.

---

## 1. Inference demo (raw) — `python -m v5.infer_demo`

Qwen2.5-1.5B, greedy, binary-search applicability question. **n=1, random-init
projections — anecdotal, NOT a quality claim.**

```
question: Is binary search applicable to find a target in this array,
          and what precondition must hold?

BASELINE (no adapter):
  "[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]\n\nTo determine if binary search is
   applicable to find a target in the given array, we need to check if the
   array is sorted in ascending order. Binary search is an efficient algorithm
   for finding an"   <- rambles, hallucinates array, never answers

V5-INJECTED (untrained projections):
  "The array is sorted in ascending order. Yes, binary search can be applied to
   find a target in a sorted array. The precondition that must hold is that the
   array must be sorted in ascending order. Binary search works by repeatedly
   dividing the search interval in half..."   <- direct: verdict + precondition

hook call counts: {'planning': 1, 'evidence': 1}   (decode steps skipped)
fallback_needed: True   (heads untrained)
DIFF: outputs differ
```

Key finding: injection stayed **coherent** with random `W_o` → residual magnitude
is sane → Stage 2 will *shape* it, not fight catastrophic perturbation.

---

## 1b. Perturbation baseline (raw) — `python -m v5.perturbation_baseline`

20 corpus questions, baseline vs V5-injected (random-init projections), Qwen2.5-1.5B.
Goal: prove the adapter is *usually non-catastrophic* before Stage 2 — NOT improvement.

```
AGGREGATE (n=20):
  hook control ok (1/1)    : 20/20
  baseline gibberish       : 1/20
  injected gibberish       : 1/20
  CATASTROPHIC (inj broke) : 1/20  (5%)
  non-catastrophic rate    : 95%
  mean baseline length     : 272 chars
  mean injected length     : 265 chars   (no length collapse)
  mean semantic sim        : 0.73        (1=identical; moderate drift, stays related)
```

Read: random injection rarely breaks generation (1 catastrophic case, sim→0.43);
moderate drift is expected with untrained projections; hook control perfect.
=> Stage 2 starts from a stable injected-generation baseline.

---

## 1c. Stage 2 core (raw) — `python -m v5.training.stage2`

Synthetic, lm_dim=128, gate_init=0.02. Trains attention routing (2A) then gated
write (2B). NOT answer-quality; proves the projections can learn where to look and
write a bounded residual.

```
--- Stage 2A (learn to LOOK: train Q/K/V; W_o/gate frozen) ---
  BEFORE:  plan_attn 0.10  evid_attn 0.53  write_ratio 0.017
  AFTER 2A: plan_attn 1.00  evid_attn 1.00  write_ratio 0.111

--- Stage 2B (learn to WRITE: train W_o + gate; Q/K/V on) ---
  AFTER 2B: plan_attn 1.00  evid_attn 1.00  write_ratio 0.116
            gates (plan/evid) = 0.002 / 0.023

success criteria: plan>=0.9 OK · evid>=0.9 OK · write bounded (<=0.35) OK
```

Note: lr 1e-3 diverged (attn loss 1.8→9.6) on the small attention pool; lr 2e-4
converges. Residual gate keeps the write ~11% of ‖h‖.

---

## 2. Substrate Population Pass (raw) — `python -m v5.training.substrate`

```
base: 831 nodes, 1454 edges
substrate nodes added: 47
  epistemic_state    27
  strategy            7
  reasoning_atom      5
  solved_subgoal      4
  failure_pattern     4
relations added: 79
total: 878 nodes, 1533 edges  -> graphs/merged_graph_substrate.json
planning-pool substrate nodes added: 16
```

---

## 3. Bridge coverage (raw) — `python -m v5.training.bridge`

```
subgraph size (avg nodes/example):
  anchors-only         : 5.0    (~0.5 edges)
  persisted 1-hop nbhd : 17.8   (real edges -> real R-GCN message passing)

per-head label coverage     base graph   substrate-enriched
  plan                       0/20 (0%)    17/20 (85%)
  evid                      19/20 (95%)   19/20 (95%)
  slot                      20/20 (100%)  20/20 (100%)
  epi                        8/20 (40%)    8/20 (40%)
  inv                        1/20 (5%)     1/20 (5%)
  shortcut                  20/20 (100%)  20/20 (100%)
```

---

## 4. Real Stage 1 training (raw) — `python -m v5.training.stage1_real`

Qwen2.5-1.5B (hidden=1536, anchor_layer=8), substrate-enriched graph, 20 examples,
150 epochs, loss 18.5 → 2.99. **Train-fit, no held-out split.**

```
head        before   after
planning    0.94  -> 1.00   (now supervised via substrate anchors)
evidence    0.32  -> 1.00
slot        0.00  -> 1.00
epistemic   0.00  -> 1.00
shortcut    0.65  -> 1.00
```

Synthetic trainability (`python -m v5.training.trainability_test`): all heads
0.5 → 1.0, fallback_applicable 1.0 → 0.0, fallback_blocked stays 1.0.

---

## 5. Repro commands (Windows PowerShell)

```powershell
python -m v5.smoke_test_toy                         # deterministic invariants (fast)
python -m v5.training.trainability_test             # synthetic head trainability
python -m v5.training.substrate                     # build substrate-enriched graph
python -m v5.training.bridge                        # corpus -> Stage1Example + coverage
python -m v5.training.stage2                         # Stage 2 (2A routing + 2B write), synthetic
$env:KMP_DUPLICATE_LIB_OK="TRUE"; python -u -m v5.realstack_test       # real-stack prefill
$env:KMP_DUPLICATE_LIB_OK="TRUE"; python -u -m v5.training.stage1_real # real Stage 1 (planning incl.)
$env:KMP_DUPLICATE_LIB_OK="TRUE"; python -u -m v5.infer_demo           # baseline vs injected generation
$env:KMP_DUPLICATE_LIB_OK="TRUE"; python -u -m v5.perturbation_baseline --n 20  # non-catastrophic baseline
```

Env note: `sentence_transformers` segfaults when co-loaded with `torch_geometric`
here — we use `transformers.AutoModel` for mpnet. Always set
`KMP_DUPLICATE_LIB_OK=TRUE` for the heavy combos.

---

## 6. Next true milestone

Not architecture. **Scale the V4 corpus** (more traces) → enables an 80/20
held-out split and the first *generalization* metrics: node precision/recall by
pool, slot/epistemic/invalidator/shortcut accuracy, fallback decision accuracy.
Then Stage 2 (train cross-attn projections) + LoRA before any inference-quality
claim. Full detail in `v5_PROGRESS.md`.
