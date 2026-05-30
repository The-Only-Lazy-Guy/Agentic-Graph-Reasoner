# READ_THIS — V5 latest raw results & quick reference

> At-a-glance dump of the latest runs (raw outputs, numbers, repro commands) so
> you don't have to dig through commits/logs. Updated each working session.

**Last updated:** 2026-05-30
**HEAD:** `e1ae5f4` · branch `main`

---

## TL;DR claim boundary

- ✅ V5 trains end-to-end on **real graph states + real LM hidden states, including
  planning**, on a substrate-rich V4 corpus.
- ✅ V5 **generates** end-to-end with the adapter live; injection is numerically
  stable (stays coherent even with untrained projections).
- ❌ NOT yet: V5 **generalizes** (corpus is 20 traces → train-fit only).
- ❌ NOT yet: V5 **improves** generation (cross-attn projections untrained — Stage 2/LoRA needed).

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
$env:KMP_DUPLICATE_LIB_OK="TRUE"; python -u -m v5.realstack_test       # real-stack prefill
$env:KMP_DUPLICATE_LIB_OK="TRUE"; python -u -m v5.training.stage1_real # real Stage 1 (planning incl.)
$env:KMP_DUPLICATE_LIB_OK="TRUE"; python -u -m v5.infer_demo           # baseline vs injected generation
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
