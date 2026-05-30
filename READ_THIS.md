# READ_THIS — V5 latest raw results & quick reference

> At-a-glance dump of the latest runs (raw outputs, numbers, repro commands) so
> you don't have to dig through commits/logs. Updated each working session.

**Last updated:** 2026-05-31
**HEAD:** `7664f99` · branch `main`

---

## TL;DR claim boundary

- ✅ V5 trains end-to-end on **real graph states + real LM hidden states, including
  planning**, on a substrate-rich V4 corpus.
- ✅ V5 **generates** end-to-end with the adapter live; injection is numerically
  stable (stays coherent even with untrained projections).
- ✅ Random-init injection is **95% non-catastrophic** over 20 questions (perfect
  1/1 hook control) → Stage 2 starts from a stable injected-generation baseline.
- ✅ **Stage 2 core (synthetic)**: residual gate + 2A/2B trainer learns attention
  routing (plan/evid precision → 1.0) with bounded gated write (~11% of ‖h‖);
  negatives stay diffuse (entropy ln 3), positives confident, no collapse.
- ✅ **Stage 2A on REAL corpus**: routing plan 0.76→1.00, evid 0.37→1.00;
  perturbation re-check 0/20 catastrophic, hooks 20/20, sim 0.95 (W_o/gate frozen
  → generation untouched, as intended for "learn to look").
- ✅ **Stage 2B on REAL corpus (write-safety)**: write path trained, all 6 gates
  pass — write_ratio 0.047 (negatives lowest 0.034), catastrophic 0/20, hooks
  20/20, sim 0.94. Generation stable with real writing.
- ✅ **Integrated Stage 1→2A→2B (one adapter)**: 7/8 gates. Heads retained
  (head-retention loss fixed epi 0.38→0.88), routing 1.0, write 0.109 (negatives
  least 0.057), catastrophic 0/20, fallback blocked/negative HIGH.
- ⚠️ **Fallback applicable-drop**: only 1.00→0.94 (1/17). The fallback gate needs
  slot≥0.85 AND primary-evidence epi≥0.70; the 20-example corpus doesn't calibrate
  the heads to cross those thresholds. A calibration + corpus-scale issue (motivates
  a support-pointer head), not a training-mechanism failure.
- ✅ **Scaled 20 → 46 traces** (local GGUF gen, 35 finalized, 382 patches) and
  re-ran held-out (10 eval). KEY FINDING: 2.3× data improved slot/shortcut/node
  generalization but did NOT fix **fallback-applicable (still 1.00)** or
  **epistemic generalization (all-node 0.00)** → points to an architecture/label
  problem (the **support-pointer head**), NOT pure data scale. (n=10 still < 100–300
  bar; indicative not conclusive.)
- ⏸️ **Held on purpose**: Stage 3 overlay, Stage 4 LoRA, any quality claim. **Next
  indicated step: support-pointer head**, not just more data.
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

With negatives (15+15 positive, 10 negative): positives confident (entropy
0.00/0.35), negatives maximally diffuse (entropy 1.10 = ln 3), top-1 freq 0.5/0.5
(no collapse).

---

## 1d. Stage 2A on REAL corpus + perturbation re-check — `python -m v5.training.stage2_real`

Qwen2.5-1.5B, substrate graph, gate=0.02, W_o frozen (learn to LOOK only).

```
attention routing (real corpus): plan 0.76 -> 1.00,  evid 0.37 -> 1.00

PERTURBATION RE-CHECK (Stage-2A adapter, 20 questions):
  catastrophic           : 0/20
  non_catastrophic_rate  : 1.000
  hooks_ok               : 20/20
  mean_base_len          : 272
  mean_inj_len           : 265
  mean_sim               : 0.948   (generation barely changes — W_o/gate frozen)
  injected_gibberish     : 0
```

HONEST CONFOUND: this used gate=0.02 vs the random baseline's gate=1.0, so 0%
catastrophic is partly because 2A barely writes by design. Write-safety is only
truly tested in Stage 2B (W_o + gate trained) — intentionally held.

---

## 1e. Stage 2B on REAL corpus — write-safety milestone — `python -m v5.training.stage2b_real`

Qwen2.5-1.5B. Train W_o + gate (Q/K/V lower LR), real positives + real negatives.
NOT a quality milestone — tests whether the adapter can WRITE without breaking safety.

```
per-case-type (after 2B):
  tag          n   write_ratio   fallback
  applicable  17     0.048         1.00
  blocked      3     0.044         1.00
  negative     5     0.034         1.00   <- negatives write LEAST
gates plan/evid: 0.012 / 0.008    overall write_ratio 0.047

perturbation re-check (20q): catastrophic 0/20, hooks 20/20, gibberish 0, sim 0.940

WRITE-SAFETY GATES (all OK): catastrophic ~0 · hooks 20/20 · no gibberish ·
  sim>=0.5 · write<=0.20 · negatives <= positives
```

HONEST CAVEAT: standalone 2B run -> aux heads are random (untrained), so
fallback_needed is 1.0 for ALL case types (retained, not regressed, but NOT the
desired "drops for applicable"). The applicable-fallback-drop needs Stage 1 heads
+ Stage 2 on ONE adapter (pipeline integration) — separate from write-safety.

---

## 1f. Integrated Stage 1->2A->2B — `python -m v5.training.stage_integrated`

One adapter through Stage 1 (heads) -> 2A (routing) -> 2B (write + head-retention).
Qwen2.5-1.5B. 7/8 integrated gates pass.

```
head metrics retained (after 2B): plan 1.0 evid 1.0 slot 1.0 epi 0.88 sc 1.0
  (head-retention loss fixed the regression: epi WAS 0.38 without it)
routing retained: plan 1.00 evid 1.00

per-case-type (write_ratio | fallback before->after):
  applicable  17   0.109   1.00 -> 0.94
  blocked      3   0.117   1.00 -> 1.00
  negative     5   0.057   1.00 -> 1.00     <- negatives write least

perturbation (20q): catastrophic 0/20, hooks 20/20, gibberish 0, sim 0.88

GATES: 7/8 OK
  [OK] head retained · routing retained · write bounded · negatives least ·
       catastrophic<=baseline · fallback blocked HIGH · fallback negative HIGH
  [FAIL] fallback applicable LOW  (0.94 — see below)
```

UNMET GATE: applicable fallback barely drops (1.00->0.94). fallback_needed wants
slot>=0.85 AND primary-evidence epi>=0.70; the 20-example corpus doesn't calibrate
the heads to cross those specific thresholds. Calibration + corpus-scale issue
(motivates an explicit support-pointer head), NOT a training-mechanism failure —
heads, routing, and write all train and retain.

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
$env:KMP_DUPLICATE_LIB_OK="TRUE"; python -u -m v5.training.stage2_real   # real Stage 2A + perturbation re-check
$env:KMP_DUPLICATE_LIB_OK="TRUE"; python -u -m v5.training.stage2b_real  # real Stage 2B write-safety milestone
$env:KMP_DUPLICATE_LIB_OK="TRUE"; python -u -m v5.training.stage_integrated  # integrated 1->2A->2B (7/8 gates)
$env:KMP_DUPLICATE_LIB_OK="TRUE"; python -u -m v5.training.corpus_scaling --corpus <corpus.jsonl>  # scale + held-out metrics
```

## 1g. Corpus scaling 20 -> 46 traces (held-out) — `v5.training.corpus_scaling`

Local GGUF generation (run_gen_llama.py -> llama-server :6768), 46 traces
(35 finalized, 382 patches), 41 train / 10 held-out.

```
coverage: plan 76% · evid 98% · slot 100% · epi 76% · inv 2% · shortcut 100%

HELD-OUT (10 unseen):
  plan node  P@1=0.57  recall=0.63   (n=7)
  evid node  P@1=0.67  recall=0.58   (n=9)
  head acc (strict all-node): slot=0.89  epi=0.00  shortcut=0.89
  fallback:  applicable=1.00  blocked=1.00  negative=1.00
  write:     applicable 0.097 · blocked 0.128 · negative 0.056 (negatives least)
```

vs n=20: node attention now generalizes modestly (the earlier P@1=1.0 was noise);
slot/shortcut generalize (0.89). BUT fallback-applicable stayed 1.00 and epi
stayed 0.00 across BOTH scales -> the epistemic/fallback gate is an architecture/
label issue (support-pointer head), not data-scale. Caveats: n=10 < 100-300;
epi all-node match is strict (per-node metric added).

---

## Scaling the corpus (data-gen on a fresh box / cloud)

```
# fresh environment setup (venv + deps + LLM backend + env vars)
setup_datagen_env.bat            # Windows   (--gpu for CUDA torch, --run to generate)
bash setup_datagen_env.sh        # Linux/cloud

# backend: opencode CLI (npm i -g opencode-ai; opencode auth login)
#          OR llama-server serving a GGUF on :6768  (LOCAL_LLM_BASE_URL)

# generate traces, then scale + measure held-out calibration:
python run_phase15_corpus.py --dataset <questions.json> --graph graphs/merged_graph.json --mode harvest
python -m v5.training.corpus_scaling --corpus artifacts/phase15/phase15_corpus.jsonl
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
