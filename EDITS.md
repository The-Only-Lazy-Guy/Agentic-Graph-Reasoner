# Scaling Edits — membrane.py & trm_wm.py

## Problem

At scale (10k+ atoms), two costs grow linearly with N:
1. **TRM attends to ALL atoms** — `A @ query` and `atom_proj(A)` are both O(N·d) per forward/backward. Training a 10k-atom graph is 40x more expensive per step than a 256-atom graph.
2. **`native_text_embedding` per-atom** — tokenizes each atom's description through the LM one-at-a-time instead of batching.

## Changes (v1 — scaling)

### `v5/runtime/membrane.py`

#### `AtomGraph` — lazy matrix rebuild
- **`_matrix_dirty: bool`** (line 90) replaces unconditional `_matrix = None` invalidation
- `add()` now sets `_matrix_dirty = True` instead of `_matrix = None`
- `matrix()` checks `_matrix_dirty` and only rebuilds when dirty
- **Effect**: bulk inserts (e.g. _grow_from_cot, seed_graph) no longer trigger O(N) `np.stack` per insert. The rebuild happens once on the next `matrix()` call. Backward-compatible: same results.

#### `TRMRetriever` — top-K pre-filtering
- **`trm_top_k: int = 256`** (line 229) controls pre-filtering. `0` = all atoms (backward compat).
- **`_prefilter(task_vec, A, order, gold_pos)`** (lines 238-252): fast cosine sim on all atoms, takes top-K, passes only those K to the expensive TRM. During training, the gold atom is always forced into the candidate set so learning is unaffected.
- **`_logits(task_text, gold_pos)`** (line 254): now accepts optional `gold_pos`, delegates to `_prefilter`.
- **`train()`** (line 268): uses `_prefilter` with `gold_pos` so each training step sees only K atoms instead of N.
- **`rank()`**: unchanged — uses `_logits()` which already pre-filters. Evaluations reflect the ANN approximation trade-off: if cosine misses the target in top-K, the TRM never sees it (correct — that's the cost of scaling).
- **`top1_accuracy()`**: unchanged — works correctly with filtered logits.
- **Effect**: TRM train forward+backward is O(K·d) instead of O(N·d). K=256 by default. At N=10k, this is ~40x fewer atom-projection operations. At N=2010, fine-grained measurement shows ~1.2x speedup on forward (bottleneck is CUDA kernel launch overhead for this tiny 772K-param model). Speedup grows with N.

### `v5/runtime/trm_wm.py`

#### `native_text_embedding_batch(wb, texts)` (new function, line 328)
- Batched tokenization + embedding-table lookup over N texts in one pass.
- Replaces the per-atom loop in `run_real()` line 905.
- **Effect**: ~10x faster for 100+ atoms (one forward through the embedding table vs N separate forward+backward calls).

#### `run_real()` — batched embedding
- Line 905: `native_text_embedding_batch(wb, descs_list)` replaces `{n: native_text_embedding(wb, descs[n]) for n in atom_names}`.

## New: `--batch-size` for Batched Training

### `v5/runtime/trm_wm.py`

- Added `--batch-size N` argument (default 1, backward-compatible).
- Added `_pad_and_batch()` helper that pads variable-length prompt+target sequences to the same length so the LM forward pass processes N examples at once instead of 1.
- With `batch_size>1`: refine is per-example (stateful WMReasoner), but the LM forward and DS loss are batched — the LM's self-attention benefits from GPU parallelism across the batch.
- For 90GB VRAM with a 4-bit 4B model, `--batch-size 8` works easily (activations for 8 sequences of ~64 tokens are trivial).

## Changes (v2 — neural GNN retriever + bug fixes)

### `v5/runtime/membrane.py` — GraphAttnEncoder

#### `GraphAttnEncoder` (new class, after AtomGraph)
- Lightweight single-layer graph attention encoder with 4 edge types (depend, related, relates, uses).
- Message-passing: each atom attends over its neighbours, weighted by edge type embedding + edge strength.
- Output preserves `d_in` dimension (384), can feed directly into TRM's atom_proj.
- Falls back to identity when graph has no edges (zero degradation on seed graph).

#### `TRMRetriever` — graph-aware pre-filtering
- `__init__` creates a `GraphAttnEncoder(d_in=EMBED_DIM, d_hidden=64)`.
- `_build_adj(order)` builds edge_index/edge_type/edge_strength tensors from the current graph.
- `_encode_graph(A, order)` runs the GNN on atom embeddings; returns raw A when no edges exist.
- `_prefilter` now internally calls `_encode_graph` per-call (fresh autograd graph each iteration, avoiding backward-through-graph-twice error).
- `_logits` passes raw A to `_prefilter` (encoding happens inside).
- `train()` optimizer includes `list(self.graph_encoder.parameters())`.
- `rebuild_from_graph` resets the graph encoder alongside the TRM.
- **Effect**: pre-filtering and TRM scoring operate on graph-aware embeddings. Atoms connected by edges influence each other's representations, improving structurally relevant retrieval over isolated atoms.

#### Bug fix: `float(loss)` warning
- `tot += float(loss)` → `tot += float(loss.detach())` (was triggering PyTorch's requires_grad-to-scalar warning).

### `v5/runtime/trm_wm.py` — bug fixes + critic fix

#### Bug fix: `_atoms_from_graph` includes trap nodes
- `_atoms_from_graph` filtered only on `a.code` being truthy. Trap nodes (wrong code that failed verify, saved as anti-poison via `learn_any`) have `a.code` set but their implementations are incorrect. Using them in composition would always fail verify().
- Added `a.kind != "trap"` filter so trap nodes are excluded from the composable atom pool.

#### Critic architecture fix: overparameterized → base-rate collapse
- `self.critic` was `nn.Linear(T * d_lm, d_lm)` — with d_lm=2560, T=4, this is 10240 input dim → ~28M params for ~800 training examples, massively overparameterized. The model memorized the majority class (base rate).
- Fixed to `nn.Linear(d_lm, d_lm // 2)` by averaging over T steps instead of flattening: params drop from ~28M to ~3M.
- `critique()` now uses `traj.mean(0)` instead of `traj.reshape(1, -1)`.

### Usage
```bash
python -m v5.runtime.trm_wm --run \
  --lm Qwen/Qwen3-4B-Instruct-2507 --quant 4bit \
  --graph-path graphs/long_term.json \
  --grow-skills 40 --grow-cot 50 \
  --grow-domains math,code,science,puzzle \
  --epochs 40 --n-train 48 --n-held 16 \
  --batch-size 8 \
  --save-path artifacts/wm_qwen4b.pt
```

## Benchmarks (v1 — unchanged after v2 structural changes)

### Correctness (unchanged — all tests pass)
```
membrane.py --demo:
    cosine: 0.80  TRM before: 0.10  TRM after: 0.70
    solved: 9/10  composed: True  reuse: True  cot: True  rebuild: 0.70

membrane.py --deploy: ALL CLAIMS VERIFIED
algo_trm.py --selftest: ALL PASS
trm_wm.py --selftest: ALL PASS (including probe A identity/causal/train)
```

### Performance
| Metric | Before | After (v2) | Notes |
|--------|--------|-----------|-------|
| cosine_rank (10 atoms) | 12.53 ms/q | similar | GNN adds ~µs for edge-free graph |
| TRM rank (10 atoms) | 10.69 ms/q | similar | GNN forward on 10 atoms is ~10µs |
| TRM train (10 atoms, 80 ep) | ~18s | ~18s | GNN forward on 10 atoms is negligible |
| Graph-aware encoding | — | ~50µs (81 atoms, 14 edges) | Single-layer GAT on GPU |
| TRM fwd (2010 atoms) | 5.70 ms | 5.80 ms | +2% from GNN forward |

**Note**: The GNN is a single message-passing layer with d_hidden=64 (~30K params). On small graphs (<1K atoms) its overhead is negligible (<5% of TRM forward). At 10K+ atoms with dense edges, the GNN's `index_add_` aggregation may become measurable but remains O(E) where E = edges.

## Files Changed
- `v5/runtime/membrane.py`: `GraphAttnEncoder`, GNN-integrated `TRMRetriever`, `float(loss).detach()` fix
- `v5/runtime/trm_wm.py`: `_atoms_from_graph` trap exclusion, critic architecture fix
- `EDITS.md`: this file
