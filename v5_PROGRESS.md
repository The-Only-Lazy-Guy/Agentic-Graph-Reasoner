# V5 GNN + Recurrent Cross-Attention Adapter — Progress Log

**Status:** Architecture implemented, validated end-to-end on a real stack, proven
**trainable** (synthetic teacher-forced: every head 0.5→1.0), and **trained on the
real V4 corpus** with real mpnet embeddings + real frozen-Qwen h_init. The
Substrate Population Pass applies the corpus's own scoped patches into the graph,
taking **planning coverage 0% → 85%**, and real Stage 1 now trains **all six heads
including planning** on real graph states. Remaining work is data scale + the
joint end-to-end recipe (Stages 2–5) + the 4B GGUF path — not architecture.

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
Q_r = W_q( LayerNorm(h_r) ‖ goal_vector ‖ slot_state_r )
K_r = base_K + delta_K,  V_r = base_V + delta_V   (StateOverlayHead)
A_r = softmax(Q_r @ K_r.T / sqrt(d)) @ V_r
h_{r+1} = h_r + W_o(A_r)        # pre-norm residual stream (norm on the Q input only)
[AuxHeads read LayerNorm(h_r) via head_norm, then update LoopState]
[ExitCondition check]
```

> NOTE: the update is **pre-norm** (norm on the query input, residual stream
> carried forward). The earlier post-norm form `LayerNorm(h_r + W_o(A))` was a
> contraction that erased h_init — see the 2026-05-30 trainability section.

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

## 2026-05-30: Consistency pass — device crash + hook run-once guard (commit `c61d942`)

An external review flagged six items against `raw.githubusercontent.com`. Verified
each against **live source** (the CDN serves a cached snapshot that lagged `cd4bff5`):

| # | Reviewer claim | Reality |
|---|---|---|
| 1 | `encode_to_kv` passes `K=K, V=V` | Stale — already raw-embeddings-only |
| 2 | `GraphMemoryKV.device` returns `self.K.device` | **REAL** — crash, fixed |
| 3 | `update_state` missing `static_inv` | Stale — sig already has it |
| 4 | Exit checks all slots | Stale — `_required_slot_indices(task_frame)` gates it |
| 5 | Hook frequency uncontrolled | Partial — decode already skipped; added run-once guard |
| 6 | Anchor uses `h[:, 0, :]` | Stale — already `h[:, -1, :]` |

Fixes landed:

- **`subgraph.py`**: `GraphMemoryKV.device` returned `self.K.device`, but `K` was
  removed when the dataclass went raw-embeddings-only. Now `node_embeddings.device`.
  Previously raised `AttributeError` on any `.device` access.
- **`adapter.py`**: added `run_once_per_session` guard so a second prefill-shaped
  pass (beam search / chunked prefill) does not re-run the loops; decode steps
  were already skipped via `seq_len == 1`. Added `_plan_hook_calls` /
  `_evid_hook_calls` counters + `get_hook_call_counts()` for observability.

Verified: full pipeline runs clean; run-once guard holds at 1/1 across
prefill+decode+reprefill; all six v5 modules `py_compile` clean (the reviewer's
"file is one long line" was a web-tool rendering artifact, not the real file).

---

## 2026-05-30: Behavioral smoke test + node_scores cross-pool leak (commit `3989128`)

`v5/smoke_test_toy.py` — Dijkstra negative-edge toy graph, one TaskFrame, runs
planning + evidence loops, asserts deterministic invariants that hold regardless
of head training:

- planning / evidence / invalidator masks match expected node pools
- attention mass stays within each block's pool (no cross-pool leak)
- both loops always set an exit_reason
- combined invalidator (static × dynamic) fires only on nodes with a structural
  `INVALIDATED_BY` / `CONTRADICTS` edge

**Bug the smoke test caught:** `AuxHeads.update_state` adds the `NodeHead`
adjustment to ALL nodes (`new_scores = node_scores_r + new_node_adj`), so
`node_scores_r` leaked out-of-pool even though the **attention** was masked. This
contaminated the exit-condition top-k and `StateOverlayHead` top-k — a leaked
evidence node could have driven the planning loop's epistemic/invalidator check
onto a node the block cannot attend.

Fix (`cross_attention.py`): re-apply the pool mask to `node_scores_r` after
`update_state` (−1e9 on out-of-pool), keeping the cumulative score in-pool across
iterations.

**Second-order fix (`trainer.py`):** masking makes `sigmoid(node_scores_r)=0` on
out-of-pool nodes → max BCE vs anchor target=1 (node loss exploded 0.1 → 76). A
block can only be supervised on nodes it can attend, so node BCE is now restricted
to each block's own pool via `_masked_bce()`. Node loss finite again; total
converges 2.65 → 1.59 over 5 epochs.

---

## 2026-05-30: Phase 17 minimal real-stack test (commit `df2bc6c`)

`v5/realstack_test.py` — end-to-end test with REAL components, no training, no GPU
spend beyond one frozen prefill:

```text
real mpnet-768 embeddings
  + real base graph (graphs/*.json)        → evidence-pool nodes
  + injected reasoning-substrate nodes      → planning-pool nodes
  + real frozen Qwen2.5-1.5B prefill h_init → via GraphAttentionInjector hooks
  → loop logs
```

Run:

```powershell
$env:KMP_DUPLICATE_LIB_OK="TRUE"; python -u -m v5.realstack_test
python -u -m v5.realstack_test --graph graphs/cs1.json --model Qwen/Qwen2.5-0.5B-Instruct
```

Observed pool routing on real data (real GNN encode):

| node | type | plan | evid | inv |
|---|---|---|---|---|
| bsearch_strategy | strategy | ✓ | | 0 |
| unsorted_array_failure | failure_pattern | ✓ | | **1** |
| bsearch_applicability_epi | epistemic_state (uncertain) | ✓ | | 0 |
| sorted_precondition_verified | epistemic_state (verified) | | ✓ | 0 |
| binary_search_* | fact / claim | | ✓ | 0 |

Result: planning attends only planning-pool nodes, evidence only evidence-pool,
invalidator fires on the structurally-invalidating node, hooks fire **once each**,
exit reasons recorded, `fallback_needed=True` — the correct safe behavior with
untrained heads (V5 defers to the V4 path rather than answering on garbage).

### Fixes landed this milestone

- **Configurable LM hidden dim** (`cross_attention.py`): `lm_hidden_dim` threaded
  to `CrossAttentionProjections` + `AuxHeads`; `q_input_dim` derived from it. Lets
  a non-2560 LM (Qwen2.5-1.5B hidden=1536) run without edits; swap to the 4B is a
  config change.
- **Loop-log `exit_reason` backfill** (`cross_attention.py`): `to_log_entry` was
  appended before the in-iteration exit decision, so corpus logs recorded
  `exit_reason=None` even when the loop stopped via `max_loops_reached`. Now the
  last log entry is backfilled with the final exit reason. Logs are training data —
  exit_reason is a key signal.

### Three findings from this milestone

1. **Node-vocab gap (architectural).** Base graphs (`graphs/*.json`) use node
   types fact/claim/theorem/equation/hub/… that populate only the **evidence**
   pool. V5's planning-pool types (strategy / failure_pattern / control_rule /
   reasoning_chain) are the reasoning substrate V4 *writes into the graph over
   time*, not present in any base graph yet (session subgraphs hold only
   session_object/failure_pattern/signal/etc, and Phase 15 scoped patches that add
   the substrate types were never applied to a persisted graph). The real-stack
   test **injects** a few substrate nodes to mirror how V5 sees the graph in
   production. Real deployment requires V4 to have populated the substrate first.

2. **4B GGUF blocker (Phase 18).** The real target at
   `E:/PROJECT/graph_final/cache/models/Qwen3-4B-Instruct-2507-Q4_K_M.gguf` is
   llama.cpp (GGUF) format. Forward-hook hidden-state extraction
   (`register_forward_hook` on `model.model.layers[8]`) needs HF format. Swapping
   in the 4B needs either an HF-format export of the weights or a llama-cpp
   hidden-state hook path. Adapter dim is now config-driven, so this is the only
   remaining blocker for the 4B.

3. **Env instability (native-lib clash).** `sentence_transformers` segfaults
   (exit 139) when co-loaded with `torch_geometric` / the LM on this machine —
   same class of crash flagged in `v4_PROGRESS.md`. Worked around by loading mpnet
   via `transformers.AutoModel` + mean pooling instead. Run heavy combos with
   `KMP_DUPLICATE_LIB_OK=TRUE`.

---

## 2026-05-30: Teacher-forced trainability test + 4 architecture fixes (commit `be42249`)

`v5/training/trainability_test.py` — the "most important test before GPU spend":
prove the heads can LEARN the intended semantics, not merely run. Two task
families (**applicable** / **blocked**) on one synthetic graph, constructed
per-head ground truth, fixed per-task `h_init` carrying the family signal.

Result — every metric goes 0.5 → 1.0, fallback behaves correctly:

| metric | before | after |
|---|---|---|
| plan_node_acc | 0.60 | **1.00** |
| evid_node_acc | 0.50 | **1.00** |
| slot_acc | 0.50 | **1.00** |
| epi_acc | 0.00 | **1.00** |
| inv_acc | 0.50 | **1.00** |
| shortcut_acc | 0.50 | **1.00** |
| fallback_applicable | 1.00 | **0.00** |
| fallback_blocked | 1.00 | 1.00 |

This covers the reviewer's full checklist: planning weights strategy/failure,
evidence weights facts/verified, slots fill, epistemic rises only for supported
paths, invalidator fires only when context activates it, shortcut rises only
when preconditions match, and fallback drops on easy tasks while staying on
blocked ones.

### Four architecture bugs the trainability test exposed

1. **Post-norm contraction erased the LM state.** The recurrent update was
   `h_new = LayerNorm(h_r + W_o(A))`. With a fixed graph/goal context this is a
   contraction to a fixed point — two very different `h_init` (diff L2 7.1)
   collapsed to the same output (diff 1.7e-6), decoupling graph reasoning from
   the LM hidden state. Switched to a **pre-norm residual stream**
   `h_new = h_r + W_o(A)` (norm on the query input only); `h_init` now persists.

2. **Residual stream saturated the heads.** The pure residual stream grows in
   magnitude, saturating the sigmoid heads (both families read the same value).
   Added a **head_norm** (GPT-style final `ln_f`) so heads read a
   magnitude-bounded, direction-preserving `h`. Also **clamp node_scores in
   StateOverlayHead** — the `-1e9` pool-mask sentinel was feeding the overlay
   MLP and exploding `K_r/V_r` (`h_r` reached 4.6e8).

3. **Bilinear heads too weak for context gating.** `EpistemicHead` and
   `InvalidatorHead` were bilinear `h·n` — cannot hold one node's status constant
   while gating another by context through the same `h`. Upgraded both to a
   **concat-MLP over `[h, n, h⊙n]`**; both reach 1.0.

4. **Exit/fallback epistemic gate too strict.** It required EVERY top-k attended
   node to be epistemically confident, so an attended-but-contradicting node
   (legitimately low confidence) blocked exit and forced fallback forever. Now
   gate on the **primary (top-1) attended node**; the invalidator check stays
   conservative over top-k.

### What's proven vs what remains

- **Proven:** the loop's final `h_r` is 100% family-separable (a fresh linear
  probe fits it at loss 0.002); every head learns its target on that
  representation; the four fixes above are correct (toy + real-stack tests still
  pass).
- **Remaining (training recipe, not capacity):** joint end-to-end training that
  *unfreezes* the loop projections is unstable — the projections shift `h_r`
  while the heads chase it. The trainability test trains heads on the frozen loop
  representation (AdamW + cosine, lr 1e-3). End-to-end training needs lr warmup /
  staged unfreezing — a Phase 16 recipe item.

---

## Staged training plan (the recipe for joint end-to-end)

The trainability test proved capacity by training heads on a *frozen* loop
representation. Real training must unfreeze progressively — the instability is
optimization, not design. Planned stages, each with warmup + low LR on the
recurrent projections:

1. **Stage 1** — freeze LM + freeze GNN, train aux heads only. **Implemented:**
   `v5/training/stage1.py` (`Stage1Trainer`, `prepare_stage1` freeze protocol,
   `Stage1Example` interface, `synthetic_examples` + `corpus_examples` data
   paths). Synthetic smoke trains all heads 0.5→1.0 with fallback dropping on
   applicable tasks. The V4-corpus path (`corpus_examples`) is wired but gated on
   the substrate-graph + real-Qwen-h_init prerequisites below.
2. **Stage 2** — unfreeze the cross-attention projections (`W_q/W_k/W_v/W_o`,
   `K_proj/V_proj`) with LR warmup so they don't shift `h_r` out from under the
   heads.
3. **Stage 3** — unfreeze the `StateOverlayHead`.
4. **Stage 4** — LoRA on selected LM layers (around L8 / L20).
5. **Stage 5** — optional GNN fine-tuning.

> NOTE: this is **adapter** staged training (progressive unfreezing of the V5
> module). It is **not** the same as the graph-edit data staging used in V4
> learning — there the cycle is: generate data → collect scoped edits but do NOT
> apply → train → apply edits → move to the next, harder question batch. Do not
> conflate the two.

### Open architectural item (post-trainability)

- **Support-pointer head.** The fallback/exit epistemic gate currently uses the
  primary (top-1) attention node as the answer's support. Highest-attention and
  "the node the answer rests on" are usually aligned but not guaranteed — the top
  node could be a failure pattern or contradiction. A dedicated
  `answer_support_node` / `primary_support_pointer` head would make this explicit
  and is worth adding before scaled training. Fine as-is for now.

## 2026-05-30: Phase 15/17 bridge — V4 trace → Stage1Example (commit `7ca53dd`)

`v5/training/bridge.py` is the critical-path converter from "V4 produced labeled
traces" to "V5 heads train on real graph states." Per corpus row it:

- parses labels via `Phase15Dataset` (anchor / slot / epistemic / invalidator /
  shortcut);
- builds a graph object + `ActiveSubgraph` and runs the **frozen GNN** to get the
  `GraphMemoryKV`;
- **splits the single anchor mask** into `plan_anchor` vs `evid_anchor` by GNN
  pool membership — a node is supervised only in the pool its block attends;
  `None` when that pool has no anchored node (partial labels, which
  `Stage1Example` supports);
- pulls `h_init` from an injected provider.

**Real logic, swappable inputs.** `gnn` / `embedder` / `h_init_provider` default
to mocks (`ZeroEmbedder`, `MockHInitProvider`) so the converter runs on the real
corpus with no LM; real training passes a frozen `RGCNEncoder`, an mpnet
`AutoModel` embedder, and a frozen-Qwen `h_init` provider.
`stage1.corpus_examples()` now delegates to the bridge.

### Coverage report on the real corpus (measures the substrate gap)

`python -m v5.training.bridge` converts all 20 rows and reports per-head label
coverage:

| head | coverage | note |
|---|---|---|
| plan | **0/20 (0%)** | the substrate gap — no planning-pool anchors yet |
| evid | 19/20 (95%) | base-graph anchors are mostly fact/claim |
| slot | 20/20 (100%) | |
| epi | 8/20 (40%) | from `add_epistemic_state` patches |
| inv | 1/20 (5%) | from `deprecate_fact` patches |
| shortcut | 20/20 (100%) | |

The bridge handles substrate-poor rows gracefully and **reports** the 0% planning
coverage rather than hiding it. This makes the next bottleneck concrete and
measurable: planning labels rise only after V4 writes the reasoning substrate
(strategy / failure_pattern / control_rule / reasoning_chain / solved_subgoal /
epistemic_state) into the graph.

### Persisted-graph neighborhood (real topology) — commit `e42b8f0`

Initial bridge built an anchors-only graph (no edges → isolated nodes, shallow
R-GCN). Upgraded to source the **persisted `MemoryGraph` neighborhood**: resolve
the row's anchors in `graphs/merged_graph.json`, expand to the k-hop neighborhood
(anchors + neighbors **with their edges**), and remap per-anchor labels onto the
expanded node list (anchors keep labels; neighbors are unlabeled context).
Falls back to anchors-only when no graph resolves.

Measured: all 100 anchors resolve in `merged_graph`; subgraph grows from **5.0**
nodes (anchors-only, ~0.5 edges) to **17.8** nodes/example with real edges →
actual message passing. Label coverage is unchanged, which is the point:
neighborhood expansion adds topology and evidence context but **not** planning
labels (merged_graph has no strategy/failure_pattern/epistemic_state nodes). The
substrate gap is therefore conclusively a V4-write problem, not a graph-expansion
one. Next bridge step (when substrate exists): the same neighborhood path will
pick up planning-pool nodes automatically.

---

## 2026-05-30: RealProvider path + first real-corpus Stage 1 (commit `926ec1b`)

`v5/training/providers.py` supplies the real inputs that replace the bridge mocks:

- **`RealEmbedder`** — mpnet-768 via `transformers.AutoModel` + mean pooling
  (canonical home; `realstack_test` re-exports it).
- **`FrozenQwenHInitProvider`** — loads Qwen frozen; per question runs one prefill
  and returns the last-token hidden state at the anchor layer
  (`hidden_states[anchor_layer+1]`), cached per question. Exposes `hidden_size`
  so the adapter is built with `lm_hidden_dim` = the LM width.

`v5/training/stage1_real.py` runs Stage 1 end-to-end on the real Phase 15 corpus:
real mpnet node embeddings + real frozen-Qwen h_init + persisted-graph
neighborhood, training the heads that have corpus labels.

**Result** (Qwen2.5-1.5B, hidden=1536, anchor_layer=8, 20 examples, 150 epochs,
loss 11.9 → 2.26):

| head | before | after |
|---|---|---|
| evidence | 0.42 | **1.00** |
| slot | 0.00 | **1.00** |
| epistemic | 0.00 | **0.88** |
| shortcut | 0.25 | **1.00** |
| planning | — | n/a (0% coverage) |

This is the **first training of V5 heads on real graph states + real LM hidden
states** — the synthetic trainability test proved capacity; this proves the real
pipeline trains end-to-end.

> CAVEAT (honest): train-fit on 20 examples with no held-out split — overfitting
> is expected and this demonstrates the pipeline *learns*, not generalization. A
> held-out eval is meaningful once the corpus is larger / substrate-rich.

The data-improvement loop is now fully tooled and measurable: apply V4 substrate
patches → rebuild persisted neighborhoods → re-run `bridge` coverage (planning
should climb off 0%) → `stage1_real` picks up the planning head automatically.

---

## 2026-05-30: Substrate Population Pass — planning unblocked (commit `0acd967`)

The bridge had measured the one remaining gap: planning coverage 0%, because
planning **labels** come from `anchor_mask` (accessed nodes, all evidence-type),
while the planning substrate V4 wrote lives in each trace's `scoped_patches`. The
pass closes that loop using the patches already in the Phase 15 corpus (no new V4
run needed):

1. **`dataset.py`** — `Phase15Sample.substrate_nodes`: parse safe
   (`accept`/`soft_only`) `add_*` substrate patches per trace into
   `{node_id: {type, text, status}}`. Fresh `epistemic_state` nodes default to
   `uncertain` → planning pool.
2. **`substrate.py`** — `build_substrate_graph()` applies the safe substrate nodes
   + relations into a `merged_graph` copy → `graphs/merged_graph_substrate.json`.
   Added **+47 nodes** (27 epistemic_state, 7 strategy, 5 reasoning_atom,
   4 solved_subgoal, 4 failure_pattern) and **+79 relations** (831→878 nodes).
3. **`bridge.py`** — when the enriched graph is used, append each trace's
   substrate nodes to the node list and mark them as anchors; the pool split
   routes planning-type substrate to `plan_anchor`, evidence-type to `evid_anchor`.

### Bridge coverage: planning 0% → 85%

| head | base graph | substrate-enriched |
|---|---|---|
| **plan** | **0%** | **85% (17/20)** |
| evid | 95% | 95% |
| slot | 100% | 100% |
| epi | 40% | 40% |
| inv | 5% | 5% |
| shortcut | 100% | 100% |

### Real Stage 1 WITH planning (Qwen2.5-1.5B, loss 18.5 → 2.99)

| head | before | after |
|---|---|---|
| **planning** | 0.94 | **1.00** (now supervised) |
| evidence | 0.32 | **1.00** |
| slot | 0.00 | **1.00** |
| epistemic | 0.00 | **1.00** |
| shortcut | 0.65 | **1.00** |

The planning head — blocked at 0% coverage for the whole project — now trains on
real graph states + real LM h_init. Same honest caveat: train-fit on 20 examples,
no held-out split (the corpus is too small for an 80/20 split yet); this proves
the full pipeline incl. planning *learns*, not generalization.

---

## What remains before real training

1. **Substrate-populated graph**: run V4 (or apply Phase 15 scoped patches) so a
   persisted graph actually contains strategy / failure_pattern / solved_subgoal /
   epistemic_state nodes — the planning pool. Proven to flow once present.
2. **Real h_init into the trainer**: `Phase16Trainer` still uses `FakeEmbedder` +
   random `h_init`. Wire `GraphAttentionInjector` real Qwen prefill hidden states
   into training; raise `w_epistemic` / `w_invalidator` back to 1.0 once real
   h_init is in use. (The real flow is now proven by `realstack_test.py`.)
3. **Real embedder in trainer**: replace `FakeEmbedder` with the
   `transformers.AutoModel` mpnet path used by `realstack_test.py`.
4. **LoRA wrapping**: apply peft LoRA to `W_q` / `W_o` in
   `CrossAttentionProjections` before training (Stage 4 above).
5. **4B GGUF path (Phase 18)**: HF export or llama-cpp hidden-state hooks (see
   finding #2 above).

---

## V5 File Tree

```text
v5/
├── __init__.py
├── adapter.py          GraphAttentionInjector — hook injection (run-once guard + hook counters)
├── cross_attention.py  CrossAttentionProjections, RecurrentAttentionBlock, V5AttentionAdapter (configurable lm_hidden_dim)
├── exit_condition.py   Compound exit guard, fallback_needed
├── gnn_encoder.py      RGCNEncoder (2-layer R-GCN), GraphEncoderInputs, build_encoder_inputs
├── goal_encoder.py     GoalEncoder, encode_task_frame
├── loop_state.py       LoopState, 6 AuxHeads, AuxHeads.update_state
├── subgraph.py         GraphMemoryKV, ActiveSubgraph, build_active_subgraph
├── smoke_test_toy.py   Deterministic toy-graph invariant test (masks/gating/exit)
├── realstack_test.py   Phase 17 real-stack test (mpnet + Qwen2.5-1.5B + real graph)
└── training/
    ├── __init__.py
    ├── dataset.py             Phase15Dataset, Phase15Sample, CORPUS_SLOT_ALIAS
    ├── trainer.py             Phase16Trainer, FakeEmbedder, _masked_bce, TrainingConfig
    ├── trainability_test.py   teacher-forced head-trainability proof (synthetic)
    ├── stage1.py              Stage1Trainer (heads-only, frozen loop projections)
    ├── bridge.py              Phase 15/17 bridge: V4 corpus trace -> Stage1Example
    ├── providers.py           RealEmbedder (mpnet) + FrozenQwenHInitProvider
    ├── stage1_real.py         Stage 1 on the real corpus (real embeddings + h_init)
    └── substrate.py           Substrate Population Pass (apply V4 patches -> graph)
```

## Test commands

```powershell
# deterministic invariant test (fast, no model)
python -m v5.smoke_test_toy

# trainability test (synthetic; heads 0.5 -> 1.0, fallback drops)
python -m v5.training.trainability_test

# Stage 1 trainer scaffold (synthetic smoke; heads-only, frozen loop projections)
python -m v5.training.stage1

# Phase 15/17 bridge: convert the real V4 corpus -> Stage1Example + coverage report
python -m v5.training.bridge

# real-stack test (real mpnet + Qwen2.5-1.5B; needs KMP workaround)
$env:KMP_DUPLICATE_LIB_OK="TRUE"; python -u -m v5.realstack_test

# Substrate Population Pass: apply safe V4 patches -> merged_graph_substrate.json
python -m v5.training.substrate

# Stage 1 on the REAL corpus (real mpnet + real frozen-Qwen h_init; uses the
# substrate-enriched graph when present -> planning is supervised)
$env:KMP_DUPLICATE_LIB_OK="TRUE"; python -u -m v5.training.stage1_real
```

## torch_geometric install

```
torch_geometric 2.7.0 + pyg_lib/torch_sparse (pt26cu124) installed.
```

Verified: `RGCNConv` import works, GNN forward pass runs on CPU and CUDA.
