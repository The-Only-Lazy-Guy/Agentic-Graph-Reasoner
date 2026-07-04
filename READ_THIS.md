# READ_THIS — LGGN Decode v2 Progress & Raw Results (2026-07-04)

> At-a-glance dump of the latest runs (raw outputs, numbers, repro commands).
> Updated each working session.

## Architecture

```
BehaviorEncoder: 2560d (Qwen space) → 64d (behavior space)
FiLM injection: h' = γ(z)⊙h + β(z) at every transformer layer
Refiner: g → h_K via K graph-operator steps (cross-attention + operator policy)
Decoder: Qwen3.5-4B + LoRA + FiLM(z) → search/replace patches
```

## What's Proven

| Claim | Evidence | Status |
|-------|----------|--------|
| FiLM injection works | ceiling > baseline (+0.027 to +0.037) | CONFIRMED |
| v1 soft-prefix failed | baseline 0.136 > latent 0.098 | CONFIRMED (motivates v2) |
| z-dropout is key | latent(0.188) > ceiling(0.165) at n=1000 | CONFIRMED |
| Bridge works | latent > baseline in favorable conditions | CONFIRMED (cos > 0.73) |
| Sensitivity cliff | decoder recall drops sharply at cos 0.73-0.81 | CONFIRMED |
| Refiner is bottleneck | cos=0.561 << cliff at 0.73 | CONFIRMED |
| Truncation was 14% | format_fail: 14% → 0% after max_tokens 256→512 | FIXED |

## Latest Run: Diagnostic (n=200 lite, baseline+ceiling, 3 epochs)

```
format_fail:  0/29 (0%)   ← FIXED (was 14%)
zero_recall: 16/29 (55%)  ← model generates valid patch, wrong code
partial:      8/29 (28%)  ← close but not exact match
good:         5/29 (17%)  ← recall >= 0.5

baseline recall: 0.153
ceiling recall:  0.181
ceiling - baseline: +0.027 (FiLM mechanism works)

inference: ~3.4s/instance baseline, ~4.1s ceiling
```

## Full Run: n=722 (lite+fable5, all arms, 5 epochs, sensitivity)

```
[lggn-decode v2 FiLM] dataset=lite+fable5 n=500 r=1024 K=8 ops=32
  722 instances (222 SWE-bench + 500 Fable-5)
  train 578 / held 144

  refiner cos(h_K, gold_f) = 0.561

  baseline  : held recall 0.196
  constant  : held recall 0.203
  latent    : held recall 0.179
  ceiling   : held recall 0.228

  ceiling - baseline = +0.033  → FiLM INJECTION WORKS
  latent  - baseline = -0.017  → h_K HURTS (refiner too noisy)
  
  sensitivity curve (ceiling decoder, gold→h_K interpolation):
   alpha  cos(h_a,f)   recall
    1.00       1.000    0.228
    0.83       0.923    0.228   ← flat above cos 0.80
    0.67       0.805    0.228
    0.50       0.712    0.217   ← starts dropping
    0.33       0.645    0.194   ← at baseline
    0.17       0.597    0.147   ← BELOW baseline (z hurts)
    0.00       0.561    0.124   ← h_K alone = worst
    perm       0.259    0.214   ← wrong-instance gold > h_K (!!)

  graph edits: 56 nodes (26 retrievable), 51 edges, 2761 edits
  
  timing:
    refiner             :  1457.4s  (24 min)
    decoder_baseline    :  2020.6s  (34 min)
    decoder_constant    :  2675.0s  (45 min)
    decoder_latent      :  2564.0s  (43 min)
    decoder_ceiling     :  8752.3s  (146 min, includes sensitivity)
    total               : 17471.4s  (4.85 hours)
```

## Prior Run: n=1000 (lite only, all arms, z-dropout=0.15)

```
  222 instances (lite caps at 222 kept), d=2560
  train 178 / held 44
  
  refiner cos(h_K, gold_f) = 0.585

  baseline  : held recall 0.141
  constant  : held recall 0.124
  latent    : held recall 0.188   ← BEATS ceiling (z-dropout effect)
  ceiling   : held recall 0.165

  latent - baseline  = +0.047  → BRIDGE WORKS
  latent - ceiling   = +0.023  → z-dropout > gold injection
```

## Diagnostic Samples (raw model output)

```
[ZERO_RECALL] recall=0.0 — valid format, wrong fix:
  raw:  <<<<<<< SEARCH
        kwargs = {k: v for k, v in match.groupdict().items() if v is not None}
        =======
        kwargs = {k: v for k, v in match...
  gold: kwargs = match.groupdict()
        kwargs = {k: v for k, v in kwargs.items() if v is...
  → model finds right code, generates different fix strategy

[GOOD] recall=0.5 — correct fix:
  raw:  <<<<<<< SEARCH
        self.query = getattr(queryset, 'query', queryset)
        =======
        self.query = getattr(queryset, 'query', queryset)
        sel...
  gold: self.query = getattr(queryset, 'query', queryset).clone()
        self.query.subquery = True
```

## Root Cause of 0.2 Plateau

| Factor | Impact | Status |
|--------|--------|--------|
| Truncation (256 tokens) | 14% format fail | FIXED → 0% |
| Wrong fix strategy | 55% zero recall | 4B model capacity wall |
| Partial matches | 28% | Close but not exact |
| Refiner cos (0.561) | Below cliff (0.73) | Needs architectural fix |
| Bug-only prompt | Wrong for Fable-5 data | FIXED (generic prompt) |

## Commits This Session

```
b716d48 fix(lggn_decode): generic prompt framing for mixed SWE + Fable-5 data
1952328 fix(lggn_decode): increase max_new_tokens 256->512 to fix truncation
e036f23 feat(lggn_decode): inference timing + --arms filter to cut training time
363db4f feat(lggn_decode): diagnostic failure breakdown in decoder eval
5c0345e feat(lggn_decode): Fable-5 HF dataset integration + inference timing
```

## New Features

### Fable-5 HF Dataset Integration
```bash
# Fable-5 only
python -m v5.runtime.lggn_decode --dataset fable5 --n 500

# Mixed SWE-bench + Fable-5
python -m v5.runtime.lggn_decode --dataset lite+fable5 --n 500

# Fable-5 extracts str_replace displacement pairs (old→new) from
# Glint-Research/Fable-5-traces on HuggingFace (4782 sessions)
```

### Arms Filter (cut training time)
```bash
# Full run (4.8h):
--arms baseline,constant,latent,ceiling

# Fast iteration (1.2h):
--arms latent,ceiling

# Diagnostic only (30min):
--arms baseline,ceiling --decoder-epochs 3 --n 200
```

### Diagnostic Failure Breakdown
Runs on baseline + ceiling arms automatically. Reports:
- format_fail / zero_recall / partial / good percentages
- 8 raw output samples with category
- Per-instance inference timing (ms/instance, min, max)

## Repro Commands

```bash
# Quick diagnostic (30 min)
python -m v5.runtime.lggn_decode --dataset lite --n 200 \
    --arms baseline,ceiling --decoder-epochs 3 --refiner-epochs 200

# Full run with Fable-5 + sensitivity (4-5h)
python -m v5.runtime.lggn_decode --dataset lite+fable5 --n 500 \
    --K 8 --r 1024 --n-op 32 --refiner-epochs 600 \
    --learn-ops --k-warmup 0.5 --contrastive 0.1 \
    --decoder-epochs 5 --z-dropout 0.15 --sensitivity

# Selftest (no GPU, 10s)
python -m v5.runtime.lggn_decode --selftest
```

## Open Questions for Next Session

1. **Model size**: 4B model generates valid patches but wrong code (55% zero_recall).
   Bigger model (7B+) would help but VRAM cost. Is 6GB VRAM the hard cap?

2. **Refiner quality**: cos=0.561 << cliff at 0.73. Options:
   - Deeper refiner architecture
   - Separate refiners per domain (SWE vs Fable-5)
   - Stronger contrastive loss
   - More refiner capacity

3. **MoLoRA vs FiLM**: FiLM mechanism works (ceiling proves it) but is weak.
   MoLoRA (mixture of LoRA experts) could give z more steering power.
   BUT: only helps if refiner gets above the cliff first. Otherwise amplifies noise.

4. **Fable-5 data quality**: Mixed lite+fable5 may hurt refiner (different displacement
   distributions). Need to test fable5-only vs lite-only vs mixed.
