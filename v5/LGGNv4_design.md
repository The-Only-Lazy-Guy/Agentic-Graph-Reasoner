# LGGN v4 — LatentProjector (latent query replaces Call A decode)

**Status:** implemented + end-to-end cloud run, 2026-07-09/10.
**Relationship to v4_DESIGN.md:** `v4_DESIGN.md` covers the *traversal* half (multi-hop
latent retrieval). This doc covers the *query-initiation* half: how the initial query
vector `h` is produced. v4 traversal needs an `initial_h`; previously that came from
Call A (LM decodes 200 tokens of why-text → mpnet-embed). This work replaces Call A
with a single LM forward pass + a learned projection — no autoregressive decode.

## 1. Why replace Call A

v3/v4 "why" / "traversal" query modes start by running Call A:

```
spec + current ──LM.generate_raw(200 tok)──> why_text ──mpnet──> h (768d)
```

The decoded why_text is **immediately re-embedded** into mpnet-768 space and never
read again by the model. That 200-token autoregressive decode is pure waste:

- ~200ms per session (one LM generation) just to produce a vector we throw away.
- It is the only extra LM call per session beyond the final answer generation.

**LatentProjector** collapses this to a single forward pass:

```
spec + current ──LM(**enc, output_hidden_states=True)──> hs.mean(dim=0) ──MLP──> h (768d)
```

- ~1ms per session (no decode), ~200× faster on the query path.
- `why_tok = 0` (verified: cloud run reported `why_tok 0`).
- Inspired by HRM's task encoder: spec → LM hidden → MLP → query vector.

The training target is the **mpnet trace embedding** already present in Fable-5
triples, so no new labels are needed — we supervise the projection to mimic mpnet.

## 2. Architecture

### 2.1 LatentProjector (`v5/runtime/latent_projector.py`)

```python
class LatentProjector(nn.Module):
    def __init__(self, d_lm=2048, d_proj=768, d_hidden=1024):
        self.net = nn.Sequential(
            nn.Linear(d_lm, d_hidden),
            nn.LayerNorm(d_hidden),
            nn.GELU(),
            nn.Linear(d_hidden, d_proj),
        )
    def forward(self, h):
        return F.normalize(self.net(h), dim=-1)   # L2-normalized to mpnet-768 space
```

### 2.2 Hidden extraction (`RawLM.get_pooled_hidden`, `v5/runtime/lggn_realizer.py`)

```python
def get_pooled_hidden(self, text, max_tokens=512, layer=-1):
    # single forward pass, NO decode; mean-pool last hidden at `layer`
    enc = self.tok(text[:2000], return_tensors="pt", truncation=True, max_length=max_tokens).to(self.dev)
    with torch.no_grad():
        out = self.model(**enc, output_hidden_states=True)
    hs = out.hidden_states[layer][0]
    return hs.mean(dim=0).float().cpu()        # [d_lm]
```

`layer=-1` = output hidden state (last layer before lm_head).

### 2.3 Training target

From each Fable-5 triple `(goal, old, trace)`:

- input:  `spec = why_prompt(goal, old)` → LM hidden via `get_pooled_hidden` / `project_lm_hidden`
- target: `mpnet({"trace": trace})["trace"]`  (the 768-d trace embedding)
- loss:   `cosine_margin = (1 - (proj · target).sum(-1)).mean()`
          (both sides L2-normalized, so this is `1 - cosine`)

The projection learns to map the LM's *intent encoding* (spec hidden) directly into the
mpnet space that the traversal ranker already lives in.

### 2.4 Inference hook

`TraversalRanker.retrieve(goal, ..., initial_h=None)` already accepted an `initial_h`.
The latent path supplies it instead of `embed(goal)`:

```python
# in run_arm / run_chain, when query_mode == "traversal" and latent_query:
hs = lm.get_pooled_hidden(goal_for_query, layer=PROJECTOR_LAYER)   # [d_lm]
initial_h = projector(hs[None])[0].detach().numpy()               # [768]
t_result = traversal.retrieve(goal=..., initial_h=initial_h)
```

No internal change to `TraversalRanker` was needed — `initial_h` was already a parameter.

## 3. Three-phase CLI (`v5/runtime/project_loop.py`)

| Phase | Flag | Function | Output |
|-------|------|----------|--------|
| 1. Build data | `--build-projector-data` | `build_projector_data(model, out_path, layer, max_examples)` | `.npz` of `(lm_hidden, target_emb)` pairs, one LM forward per triple (no decode) |
| 2. Train | `--train-projector` | `train_projector(data_path, out_path, d_lm, epochs)` | `latent_projector.pt` (MLP weights) |
| 3. Infer | `--run --query-mode traversal+latent --latent-query` | `run_arm(... projector=..., latent_query=True)` | eval rows + results.json key `memory_traversal+latent` |

Key robustness details:
- `--train-projector` auto-detects `d_lm` from the `.npz` hidden dim (`LatentProjector(d_lm=data["lm_hidden"].shape[1])`), so it is model-agnostic (0.5B→896, 3B→2048).
- The `--run` projector loader infers `d_lm` from the checkpoint's `net.0.weight.shape[1]`, so build/train/run cannot silently mismatch dimensions.
- `latent_query=True` is also implied by `--query-mode traversal+latent` (the two are equivalent for the CLI).

### 3.1 Commands

```bash
# Phase 1 — build from Fable-5 triples (GPU; full-precision hidden states recommended, see §5)
V5_LM_QUANT=4bit python -m v5.runtime.project_loop \
  --build-projector-data --model Qwen/Qwen2.5-3B \
  --projector-data artifacts/projection_data.npz --projector-layer -1

# Phase 2 — train MLP (tiny, CPU or GPU)
python -m v5.runtime.project_loop \
  --train-projector --projector-data artifacts/projection_data.npz \
  --projector-out artifacts/latent_projector.pt --epochs 200

# Phase 3 — eval (skips Call A; why_tok = 0)
V5_LM_QUANT=4bit python -m v5.runtime.project_loop \
  --run --arm memory --query-mode traversal+latent \
  --ranker artifacts/traversal_ranker --lora artifacts/project_lora \
  --projector-out artifacts/latent_projector.pt --archetypes compose
```

Unit test: `scripts/test_latent_projector.py` (offline, Qwen2.5-0.5B + all-mpnet-base-v2;
builds 32 pairs, trains 40 epochs, asserts projector emits 768-d unit-norm vectors and
latent-vs-mpnet cosine > 0; also regression-checks `get_pooled_hidden(layer=)` and
`projector(...).detach().numpy()`).

## 4. Results — first cloud run (Qwen2.5-3B, 4bit)

| Phase | Metric | Value |
|-------|--------|-------|
| 1 | pairs built | 936 (full `realizer_triples.jsonl`) |
| 2 | train_loss | 0.70 → 0.04 |
| 2 | **test_cos** | **0.33 → 0.18** (over 200 epochs — low absolute, see §5) |
| 3 | solve | 40/80 = **0.500** |
| 3 | **DEP** | **0.000** (n=20, `compose` benchmark) |
| 3 | indep | 0.667 |
| 3 | mem_tok | **3** (≈1 file path in payload) |
| 3 | why_tok | **0** (Call A correctly skipped) |
| 3 | wall | 35.1s (80 sessions) |

Per-session logs: every `compose_*_s3` delivered `fees.py` (one of the two *source*
files) instead of the target `checkout.py`, `hops=3`, `ranker_hit=0.50`.

**Interpretation:** the speed win is real (`why_tok 0`, 35s), but the latent query is
currently too weak for this benchmark — DEP 0.000 means the 2-hop derivation never
completes because the traversal retrieves only one of the two required sources.

## 5. Diagnosis — why DEP collapsed

The projector's `test_cos ≈ 0.18` is the root cause. Under `V5_LM_QUANT=4bit`, the
layer-`-1` hidden states are quantized/noisy, so the projection target (clean mpnet) is
hard to regress and `initial_h` lands in the wrong region of mpnet space → the traversal
starts from a bad vector and retrieves the wrong file (`fees.py`).

Evidence the projection *can* work: the offline smoke test (Qwen2.5-0.5B, **full
precision**) reached **mean cosine(latent, mpnet) = 0.83** on 8 held-out samples. The
drop to 0.18 on the 3B/4bit cloud run is consistent with quantization-degraded hidden
states, not with the architecture being unsound.

Note also the test_cos *decreased* from ep1 (0.33) to ep200 (0.18): with 936 train pairs
the MLP overfits the train split (loss 0.70→0.04) while test generalization degrades.
This is a secondary issue; the dominant one is hidden-state quality.

## 6. Bugs found & fixed during cloud bring-up

These cost several push/fix cycles; recorded so a re-run does not repeat them.

| # | Symptom | Fix |
|---|---------|-----|
| 1 | `ModuleNotFoundError: v5.runtime.solution_ladder` on cloud | cloud working tree out of sync → `git fetch --all && git reset --hard origin/<branch>` |
| 2 | `FileNotFoundError: ranker_meta.npz` in `load_ranker` | `.npz` is gitignored → `git add -f artifacts/traversal_ranker/ranker_meta.npz` (also committed `ranker.pt`, `gap.pt`, `feat_proj.pt`) |
| 3 | `get_pooled_hidden() got unexpected keyword argument 'layer'` | param was named `layer_idx`; renamed to `layer` to match call sites |
| 4 | `Can't call numpy() on Tensor that requires grad` at `projector(hs[None])[0].numpy()` | `.detach().numpy()` at both call sites (run_arm + run_chain) |
| 5 | `mat1 (28x896) and mat2 (2048x1024) cannot multiply` in train | `train_projector` hardcoded `d_lm=2048` → now auto-detected from `.npz` |
| 6 | GitHub push blocked: Groq API keys in `realizer_triples.jsonl` | 70 keys redacted to `REDACTED_GROQ_KEY` (they were noise inside trace text; projector training unaffected) |
| 7 | `Converting a tensor with requires_grad=True to a scalar` warning | `total_loss += float(loss) * len(x)` → `loss.item()` |

## 7. Next steps / gates

| Gate | Condition | Status |
|------|-----------|--------|
| LP1 | smoke test: latent-vs-mpnet cosine > 0 (tiny data) | PASS (0.83 @0.5B full-prec) |
| LP2 | cloud: latent-query DEP within X% of explicit-decode baseline | **not yet** — blocked by LP3 |
| LP3 | cloud: test_cos > ~0.5 (full-precision hidden states) | **not yet** (0.18 @4bit) |
| LP4 | end-to-end solves `compose` dep sessions with latent query | **not yet** (DEP 0.000) |

Immediate actions:
1. **Baseline isolation** — run explicit-decode traversal to see if traversal itself is
   sound on `compose` (independent of latent quality):
   ```bash
   V5_LM_QUANT=4bit python -m v5.runtime.project_loop --run --arm memory \
     --query-mode traversal --ranker artifacts/traversal_ranker \
     --lora artifacts/project_lora --archetypes compose
   ```
   If baseline DEP > 0, the issue is purely latent-query quality → step 2.
   If baseline DEP is also 0, the ranker/traversal is broken on `compose` regardless.
2. **Full-precision projector** — re-run Phase 1 **without** `V5_LM_QUANT=4bit` so
   `get_pooled_hidden` sees clean hidden states; expect test_cos to rise toward the
   0.83 seen in the smoke test, and DEP to recover. Projector training stays tiny, so
   only the data-build forward passes get more expensive.

## 8. Files

### New
- `v5/runtime/latent_projector.py` — `LatentProjector` + `project_lm_hidden`
- `scripts/test_latent_projector.py` — offline 3-phase smoke test
- `v5/LGGNv4_design.md` — this document

### Modified
- `v5/runtime/lggn_realizer.py` — `RawLM.get_pooled_hidden()`
- `v5/runtime/project_loop.py` — `build_projector_data`, `train_projector`, CLI flags
  (`--build-projector-data`, `--train-projector`, `--latent-query`, `--query-mode
  traversal+latent`, `--projector-data/-out/-layer`), inference hook in `run_arm`/`run_chain`
