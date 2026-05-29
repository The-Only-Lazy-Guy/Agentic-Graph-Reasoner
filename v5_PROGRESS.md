# V5 GNN + Recurrent Cross-Attention Adapter — Progress Log

**Status:** Architecture implemented, 7 correctness bugs fixed, Phase 15 corpus collected, Phase 16 training pipeline built. Pending: real BERT embedder, Qwen3 hidden-state integration (Phase 17).

**Last updated:** 2026-05-30

**Architecture reference:** `V5_ARCHITECTURE.md`

---

## V5 Design Goals

Replace the V4 micro-controller shortcut path with a learned GNN + recurrent cross-attention module injected into Qwen3-4B. The LM's own weights remain frozen; only the adapter trains (LoRA + aux heads).

```text
Qwen3-4B (frozen)
  Layer 8  <- planning hook  (RecurrentAttentionBlock, r_plan iterations)
  Layer 20 <- evidence hook  (RecurrentAttentionBlock, r_evidence iterations)

MemoryGraph
  -> BERT text embeddings (pre-computed, 768-dim)
  -> RGCNEncoder (2-layer R-GCN, 12 relation types, hidden=256)
  -> GraphMemoryKV (raw node embeddings + planning/evidence masks)
  -> RecurrentAttentionBlock x2
```

Each block runs R iterations of:

```
Q_r = W_q( h_r ‖ goal_vector ‖ slot_state_r )
K_r = base_K + delta_K,  V_r = base_V + delta_V   (StateOverlayHead)
A_r = softmax(Q_r @ K_r.T / sqrt(d)) @ V_r
h_{r+1} = LayerNorm(h_r + W_o(A_r))
[AuxHeads update LoopState]
[ExitCondition check]
```

---

## 2026-05-29–30: V5 Architecture Implementation

### Commits

| Hash | Description |
|---|---|
| `d9bed09` | GNN encoder, goal encoder, loop state, cross-attention adapter |
| `c29d472` | GraphMemoryKV, ActiveSubgraph, typed node pool masks |
| `cd4bff5` | **7 correctness bug fixes** |
| `284360e` | Phase 15 dataset + Phase 16 trainer |

### Modules shipped

#### `v5/gnn_encoder.py`

R-GCN encoder: `text_emb(768) + node_type_emb(64) + epistemic_emb(16) + confidence(1)` → `Linear(849,256)` → 2×`RGCNConv(256,256,num_relations=12)` with residual connections.

- `NODE_TYPE_VOCAB`: fact, claim, strategy, failure_pattern, solved_subgoal, reasoning_atom, control_rule, epistemic_state, procedure, application, reasoning_chain, unknown
- `encode_to_kv(inputs, active_subgraph)` → `GraphMemoryKV` (raw embeddings only; each block projects its own K/V)

#### `v5/goal_encoder.py`

- `GOAL_DIM=128`, `NUM_SLOTS=10`
- `GoalEncoder.forward(family_ids, mode_ids, slot_ids, slot_mask)` → `[B, 128]`
- `encode_task_frame(task_frame, device, goal_encoder)` → `[1, 128]`

#### `v5/subgraph.py`

- `GraphMemoryKV`: node_embeddings `[N, 256]`, node_ids, node_types, planning_mask `[N bool]`, evidence_mask `[N bool]`, invalidator_flags `[N float]`, slot_relevance `[N float]`
- `ActiveSubgraph`: pre-processed subgraph with pool masks; built from a MemoryGraph + pre-computed BERT embeddings
- `build_active_subgraph(graph, node_ids, text_embeddings, device, task_frame)`:
  - planning pool: strategy / failure_pattern / control_rule / reasoning_chain / reasoning_atom + epistemic(uncertain/unknown)
  - evidence pool: fact / claim / application / solved_subgoal / procedure + epistemic(verified/supported)
  - invalidator_flags: 1.0 if node has outgoing `INVALIDATED_BY` or `CONTRADICTS` edge

#### `v5/loop_state.py`

- `LoopState` dataclass: h_r, slot_state_r, node_scores_r, shortcut_validity_r, epistemic_confidence_r, invalidator_flags_r, loop_idx, exit_reason
- `to_log_entry(node_ids, layer)` → structured dict for corpus logging
- 6 auxiliary heads (all `nn.Module`):
  - `SlotHead`: h_r → `[B, NUM_SLOTS]` slot fill confidence
  - `NodeHead`: (h_r, node_embeddings) → `[B, N]` per-node logit adjustment
  - `StateOverlayHead` (TOP_K=16): loop state summary → (delta_K, delta_V) broadcast to all nodes
  - `EpistemicHead`: (h_r, node_embeddings) → `[B, N]` belief confidence
  - `InvalidatorHead`: (h_r, node_embeddings) → `[B, N]` invalidator activation probability
  - `ShortcutHead`: h_r → `[B, 1]` shortcut safety score
- `AuxHeads.update_state(state, node_embeddings, static_inv)`: combines structural gate × dynamic neural activation: `combined_inv = static_inv * dynamic_inv`

#### `v5/exit_condition.py`

Compound exit guard:

```python
should_exit_loop(state, loop_idx, r_max, task_frame) -> (bool, reason_str)
```

Checks (all must pass): low entropy, required slots filled, no invalidators, epistemic ok, shortcut validity.

- `_required_slot_indices(task_frame)`: only checks slots listed in `task_frame.required_slots`, not all NUM_SLOTS
- Hard cap fires at `loop_idx >= r_max - 1`
- `fallback_needed(state)`: True if evidence loop exited via max_loops with incomplete state

#### `v5/cross_attention.py`

- `CrossAttentionProjections.forward(h_r, goal, slot_state, K_r, V_r, node_mask)`:
  - Q = W_q(h_r ‖ goal ‖ slot_state), attention over K_r/V_r, residual + LayerNorm
  - Empty mask safety: falls back to attend-all (prevents NaN from all-masked softmax)
- `RecurrentAttentionBlock`: runs recurrent loop, each block projects its own K/V from raw GNN embeddings
- `V5AttentionAdapter`: planning_block (L8) + evidence_block (L20), shared AuxHeads

#### `v5/adapter.py`

- `GraphAttentionInjector.prepare_session(graph, node_ids, text_embeddings, task_frame, r_plan, r_evidence)`:
  - Builds ActiveSubgraph, runs GNN once per session, caches GraphMemoryKV
- `inject(model)` context manager: registers forward hooks at L8/L20
- Hooks skip decode steps (`seq_len == 1`); use last token `h[:, -1, :]` as anchor
- `_get_transformer_layers(model)`: tries `model.model.layers` first (Qwen3-4B path)

---

## 2026-05-30: 7 Bug Fixes (commit `cd4bff5`)

All bugs identified in architecture review and fixed before training:

| # | Bug | Fix |
|---|---|---|
| 1 | `max_loops_reached` never fired (`>= r_max` never true in 0..r_max-1 loop) | Changed to `>= r_max - 1`; guarantee exit_reason set after loop |
| 2 | Slot fill checked all NUM_SLOTS (too strict when only 2 required) | `_required_slot_indices(task_frame)` — only check required slots |
| 3 | Hooks fired on every decode step (expensive, semantically wrong) | `if h.shape[1] == 1: return output` in both hooks |
| 4 | Hook anchor = `h[:, 0, :]` (BOS token, low context) | Changed to `h[:, -1, :]` (last token, most context-rich during prefill) |
| 5 | K/V projection shared across planning + evidence blocks | Removed K/V from GraphMemoryKV; each RecurrentAttentionBlock projects its own K/V |
| 6 | Neural invalidator head could hallucinate on non-invalidatable nodes | `combined_inv = static_inv * dynamic_inv` (structural gate prevents false firing) |
| 7 | All-masked softmax → NaN when node pool is empty | Added `if not node_mask.any(): node_mask = None` fallback |

Smoke test confirmed all 7 fixes before commit.

---

## 2026-05-30: Phase 15 Corpus Collection

Corpus generated from 20 tasks (harvest mode):

```text
artifacts/phase15/phase15_corpus.jsonl
```

| Metric | Value |
|---|---|
| tasks | 20 |
| scoped patches (total) | 371 |
| epistemic nodes added | 27 |
| patch emission rate | ~80% of sessions |
| avg patches/session | 18.55 |

Patch type distribution:

| Type | Count |
|---|---|
| add_relation | 181 |
| reinforce_existing | 106 |
| add_epistemic_state | 27 |
| add_fact | 21 |
| add_strategy | 17 |
| add_reasoning_atom | 5 |
| add_solved_subgoal | 4 |
| add_control_rule | 4 |
| deprecate_fact | 2 |

Also fixed before corpus run:

- `reasoning/scoped_edits.py`: model-emitted epistemic patches use 0.05 support floor (was 0.08)
- `reasoning/scoped_edits.py`: `GraphEditPatch.to_dict()` exposes top-level `metadata` key

---

## 2026-05-30: Phase 16 Training Pipeline (commit `284360e`)

### `v5/training/dataset.py` — Phase15Dataset

Parses `phase15_corpus.jsonl` into `Phase15Sample` dataclasses with supervision targets:

| Target field | Shape | Head | Source |
|---|---|---|---|
| `anchor_mask` | [N] float | NodeHead | input.anchors — nodes V4 accessed |
| `slot_fill_target` | [NUM_SLOTS] float | SlotHead | metrics.slot_fill_stats.filled_slots |
| `epistemic_target` | [N] float | EpistemicHead | add_epistemic_state patches |
| `invalidator_target` | [N] float | InvalidatorHead | deprecate_fact patches |
| `shortcut_valid` | scalar | ShortcutHead | finalized + steps ≤ 0.5 × max_steps |

`CORPUS_SLOT_ALIAS` maps V4 task-specific slot names (answer, explanation, relationship, …) to canonical `SLOT_VOCAB` entries.

### `v5/training/trainer.py` — Phase16Trainer

- Trains adapter AuxHeads on corpus supervision signals; GNN + GoalEncoder frozen; Qwen3 **not** required
- `FakeEmbedder`: zero BERT embeddings until Phase 17
- `_FakeGraph`: no edges (corpus anchors have no edge info); replaced at Phase 17 with real MemoryGraph
- Per-element `pos_weight` for sparse epistemic/invalidator targets (1–2 positive nodes out of N)
- `w_epistemic=0.1`, `w_invalidator=0.1` until Phase 17 (real LM hidden states needed for meaningful gradients)
- Gradient clip at norm=1.0; AdamW lr=3e-4, weight_decay=1e-2
- **Convergence**: total loss 2.76 → 1.22 over 5 epochs with fake embeddings; slot + shortcut heads converge; node head collapses cleanly

---

## What remains before Phase 17

1. **Real BERT embedder**: replace `FakeEmbedder` with `sentence-transformers` or HuggingFace BERT; provides meaningful `[N, 768]` text embeddings → GNN produces semantically meaningful K/V
2. **Real Qwen3 hidden states**: `h_init` must come from actual Qwen3 prefill via `GraphAttentionInjector`; raise `w_epistemic` and `w_invalidator` back to 1.0 once real h_init is in use
3. **Real MemoryGraph with edges**: load actual session subgraphs from `data/session_subgraphs/`; edges are needed for R-GCN to propagate across relation types
4. **LoRA wrapping**: apply peft LoRA to `W_q` and `W_o` in `CrossAttentionProjections` before Phase 17 training
5. **Qwen3 smoke test**: one toy graph + Dijkstra question + frozen Qwen3 with hooks to verify planning/evidence pool selection and fallback trigger end-to-end

---

## V5 File Tree

```text
v5/
├── __init__.py
├── adapter.py          GraphAttentionInjector — hook injection into Qwen3-4B
├── cross_attention.py  CrossAttentionProjections, RecurrentAttentionBlock, V5AttentionAdapter
├── exit_condition.py   Compound exit guard, fallback_needed
├── gnn_encoder.py      RGCNEncoder (2-layer R-GCN), GraphEncoderInputs, build_encoder_inputs
├── goal_encoder.py     GoalEncoder, encode_task_frame
├── loop_state.py       LoopState, 6 AuxHeads, AuxHeads.update_state
├── subgraph.py         GraphMemoryKV, ActiveSubgraph, build_active_subgraph
└── training/
    ├── __init__.py
    ├── dataset.py      Phase15Dataset, Phase15Sample, CORPUS_SLOT_ALIAS
    └── trainer.py      Phase16Trainer, FakeEmbedder, TrainingConfig
```

## torch_geometric install

```
torch_geometric 2.7.0 + pyg_lib/torch_sparse (pt26cu124) installed.
```

Verified: `RGCNConv` import works, GNN forward pass runs on CPU and CUDA.
