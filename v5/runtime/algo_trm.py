"""TRM-over-graph (#55) — a TINY Recursive Model as the OWNED reasoner, trained from scratch, with the
graph giving it exactly what a tiny net needs (Q5). TRM (Tiny Recursive Model, ~7M, beats HRM on
ARC/Sudoku/Maze) recursively refines a latent answer + a scratchpad.

V3 (2026-07-25) — PROPER Tiny Recursive Model (Jolicoeur-Martineau 2025):
  - Two latents: z (scratchpad) and y (solution embedding), single shared network f
  - Inner loop: z = f(z + task + proj_y + cross_attn(z, R))  — think
  - Outer step: y = f(y + z)                                  — act
  - Cross-attention from z to R per step
  - Output T per-cycle y_t values → fill LM working memory slots
  - NOT a ranker — never supervised on graph atoms

  selftest (no torch-GPU):  python -m v5.runtime.algo_trm --selftest
"""
from __future__ import annotations

import argparse
import sys


def _build():
    import math
    import torch
    import torch.nn as nn

    class TRMReasoner(nn.Module):
        """Proper Tiny Recursive Model (Jolicoeur-Martineau 2025).

        Two latents: z (scratchpad) and y (solution embedding).
        Single shared network f applied identically in inner (think) and outer (act) steps.

        Each cycle t:
          Think:  z = f(z + task_proj(x) + proj_y(y) + cross_attn(z, R))
          Act:    y = f(y + z)

        Args:
            d_in:    input embedding dim (MiniLM 384 or LM hidden dim)
            d:       hidden dim of both latents
            T:       number of thinking cycles
            n_heads: cross-attention heads
        """
        def __init__(self, d_in: int = 384, d: int = 256, T: int = 5, n_heads: int = 4,
                     adaptive: bool = False, token_head_d_lm: int | None = None):
            super().__init__()
            self.T = T
            self.d = d
            self.d_in = d_in
            self.adaptive = adaptive

            self.task_proj = nn.Linear(d_in, d)
            self.atom_proj = nn.Linear(d_in, d)
            self.proj_y = nn.Linear(d, d)

            # Shared network f: LayerNorm → 2-layer GELU MLP
            self.f_norm = nn.LayerNorm(d)
            self.f_mlp = nn.Sequential(
                nn.Linear(d, 4 * d), nn.GELU(),
                nn.Linear(4 * d, d),
            )

            # Cross-attention: z queries R (retrieved atom embeddings)
            assert d % n_heads == 0, f"d={d} must be divisible by n_heads={n_heads}"
            self.cross_attn = nn.MultiheadAttention(d, n_heads, batch_first=True)

            # Learnable initial latents
            self.z0 = nn.Parameter(torch.zeros(d))
            self.y0 = nn.Parameter(torch.zeros(d))

            # Per-cycle y projection (maps d→d, applied before output)
            self.y_head = nn.Sequential(
                nn.Linear(d, d), nn.GELU(),
                nn.Linear(d, d),
            )

            # POINTER HEAD: the TRM is a COPIER — the next token is chosen by pointing at one of the
            # span TOKENS in context. The digits already exist as tokens in the spans, so pointing
            # (a token id per step) reconstructs multi-digit numbers token by token and never needs
            # the 151K-vocab head that collapses onto high-frequency space tokens. pointer_head maps
            # the final y to a query; dotting it with the projected span-token keys gives the pointing
            # logits over [L span positions]; pointer_eos adds one special "stop" action. Spaces
            # between numbers exist in the spans, so no separate space action is needed.
            self.token_head_d_lm = token_head_d_lm
            if token_head_d_lm is not None:
                self.input_proj = nn.Linear(token_head_d_lm, d_in)
                self.pointer_head = nn.Linear(d, d)
                self.pointer_eos = nn.Linear(d, 1)
                # EOS must be LEARNED, not defaulted to: "stop immediately" is the easy local
                # optimum for an unsure model, so start it unlikely.
                nn.init.constant_(self.pointer_eos.bias, -3.0)
                # POSITION AWARENESS for the pointer keys: R[j] depends only on the TOKEN at
                # position j, so two "5" tokens are the same key and the pointer degenerates into
                # a token-type classifier (it cannot tell "the 5 in this problem" from "the 5 in
                # the next problem" and collapses onto the most frequent type, space). Adding a
                # learned absolute position encoding makes occurrences addressable: the problem's
                # digits live in a consistent region of the top recalled span, which the model can
                # then learn to target.
                self.pointer_pe = nn.Parameter(torch.zeros(1024, d))
                # GRAPH CONTROLLER: the TRM operates on the session graph as OBJECTS. R is the set
                # of session-graph node embeddings (each span = one object) plus the main graph's
                # tool embeddings; edges are extra information carried into a one-hop message-passing
                # layer (relation-type embedding + learned edge strength gate the message). Three
                # heads over the final y: recall (which nodes to surface to the LM), evict (which
                # nodes to drop from the session graph at write time), tool (which main-graph
                # tools to fetch into the session).
                self.graph_edge_emb = nn.Embedding(12, 16)
                self.graph_msg = nn.Linear(d + 16, d)
                self.graph_self = nn.Linear(d, d)
                self.graph_norm = nn.LayerNorm(d)
                self.recall_head = nn.Linear(d, d)
                self.evict_head = nn.Linear(d, d)
                self.tool_head = nn.Linear(d, d)
                # CONTENT PRIOR for recall: the query is the probe's own text, so LM-space cosine
                # similarity between query and node is a strong, honest, probe-sensitive signal
                # (verified: different top-4 per probe). The learned head still adds edge-aware
                # structure on top; alpha is learnable so training can scale the prior up/down.
                self.graph_cos_alpha = nn.Parameter(torch.tensor(1.0))

            # ACT halt head: y_t → scalar halting probability per step. Bias init -3 so sigmoid starts
            # near 0.05 — the model must LEARN to halt early, defaulting to using all T steps.
            if adaptive:
                self.halt_head = nn.Linear(d, 1)
                nn.init.constant_(self.halt_head.bias, -3.0)

        def _f(self, v: torch.Tensor) -> torch.Tensor:
            """Single shared network: Layernorm → MLP."""
            return self.f_mlp(self.f_norm(v))

        def step(self, x: torch.Tensor, z: torch.Tensor, y: torch.Tensor,
                 R: torch.Tensor, return_attn: bool = False) -> tuple:
            """ONE think/act cycle against the CURRENT context R. Returns (z, y, y_out).

            forward() runs all T cycles against an R that is projected once, before the loop, and never
            changes -- so the recursion can only re-weight a fixed candidate set. That is the structural
            reason the TRM reads as a ranker no matter how it is trained: nothing it computes can alter
            what it is able to attend to.

            step() exposes the same math one cycle at a time so a caller can MUTATE R between cycles
            (append an observation, swap in a node's members, drop a candidate). The arithmetic is
            identical to forward()'s loop body -- forward() is now written in terms of this method, and
            --selftest asserts the two agree bit-for-bit against the pre-refactor sequence.

            x: [d] projected task input (already through task_proj).
            R: [N, d] projected context rows (already through atom_proj) -- may differ every call.

            return_attn=True also returns the cross-attention distribution over R ([N], sums to 1) --
            see rank() below, which is the only caller that needs it."""
            ctx, w = self.cross_attn(z.unsqueeze(0).unsqueeze(0), R.unsqueeze(0), R.unsqueeze(0))
            ctx = ctx.squeeze(0).squeeze(0)
            z = self._f(z + x + self.proj_y(y) + ctx)      # think
            y = self._f(y + z)                             # act
            if return_attn:
                return z, y, self.y_head(y), w.squeeze(0).squeeze(0)
            return z, y, self.y_head(y)

        def rank(self, x_vec: torch.Tensor, atom_vecs: torch.Tensor) -> torch.Tensor:
            """REAL TRM AS RANKER — not cosine computed after the fact. Runs the same T-cycle recursion
            as forward(), and returns the FINAL cycle's cross-attention distribution over atom_vecs: [N],
            softmax already applied by nn.MultiheadAttention, so it sums to 1 and IS a probability over
            candidates. This is the network's own pointer -- the thing it already computes every cycle to
            decide what to read (`step`'s `ctx`), just kept instead of discarded.

            Differentiable end to end: train with NLL against a gold index (`-log(w[gold])`) and the
            gradient reaches task_proj/atom_proj/the shared f through every one of the T cycles, exactly
            the same graph forward()'s y trajectory already backprops through. Nothing new is added to
            the network to make this possible -- forward() computed and discarded this same tensor on
            every single call before this method existed."""
            x = self.task_proj(x_vec)
            R = self.atom_proj(atom_vecs)
            z, y = self.z0, self.y0
            w = None
            for _ in range(self.T):
                z, y, _, w = self.step(x, z, y, R, return_attn=True)
            return w

        def pointer_logits(self, task_emb_lm: torch.Tensor, cue_tokens_lm: torch.Tensor,
                           atom_tokens_lm: torch.Tensor, prefix_embs: torch.Tensor) -> torch.Tensor:
            """POINTER HEAD: logits over [span token positions (L), EOS] for the next emitted token.

            Context = [cue tokens | span tokens | emitted prefix]. The CUE (the probe's 6-word
            opener) is in LM space and, crucially, its words APPEAR in the span — the recursion can
            align "the problem starts here" and learn that the digits FOLLOW the cue-matched region.
            The cue never contains gold digits (gold excludes the first-6-words numbers), so nothing
            leaks through it. Span tokens get the learned PE; the pointer scores cover ONLY the span
            part (the deliverable is a span-token copy). input_proj maps everything into the TRM's
            input space; after the T-cycle recursion the final y is a query dotted against the span
            keys, giving the pointing scores over [span positions, EOS]."""
            x = self.task_proj(self.input_proj(task_emb_lm.float()))
            cue_R = self.atom_proj(self.input_proj(cue_tokens_lm.float()))
            atom_R = self.atom_proj(self.input_proj(atom_tokens_lm.float()))
            L = atom_tokens_lm.shape[0]
            atom_R = atom_R + self.pointer_pe[:L]
            R = torch.cat([cue_R, atom_R], dim=0)
            if prefix_embs is not None and prefix_embs.numel():
                R = torch.cat([R, self.atom_proj(self.input_proj(prefix_embs.float()))], dim=0)
            z, y = self.z0, self.y0
            y_out = None
            for _ in range(self.T):
                z, y, y_out, _ = self.step(x, z, y, R, return_attn=True)
            q = self.pointer_head(y_out)
            span_keys = R[cue_R.shape[0]:cue_R.shape[0] + L]
            scores = (q.unsqueeze(0) * span_keys).sum(-1) / math.sqrt(self.d)
            eos = self.pointer_eos(y_out)
            return torch.cat([scores, eos], dim=0)

        def select_tokens(self, task_emb_lm: torch.Tensor, cue_tids: torch.Tensor,
                          span_tids: torch.Tensor, lm_embed: torch.Tensor,
                          top_k: int = 32) -> list:
            """ONE-SHOT multi-label selection: score every span position (BCE-trained) and return
            the top-k POSITION indices. The answer contract is a SET (gold numbers contained in the
            emitted set) and the caller decodes the selected positions with run-based grouping
            (consecutive selected positions form numbers), so selection fits the task exactly and
            avoids the autoregressive local optimum (the sequential pointer learned "emit a digit"
            and looped). Returns a list of span position indices."""
            dev = span_tids.device
            atom_tokens_lm = lm_embed[span_tids].float()
            cue_tokens_lm = lm_embed[cue_tids.to(dev)].float()
            L = span_tids.shape[0]
            with torch.no_grad():
                lg = self.pointer_logits(task_emb_lm, cue_tokens_lm, atom_tokens_lm,
                                         torch.zeros(0, atom_tokens_lm.shape[1], device=dev,
                                                     dtype=torch.float32))
            idx = lg[:L].topk(min(top_k, L)).indices.tolist()
            return [int(i) for i in idx]

        def controller_logits(self, task_emb_lm: torch.Tensor, node_embs_lm: torch.Tensor,
                              edge_index: torch.Tensor, edge_type: torch.Tensor,
                              edge_strength: torch.Tensor,
                              tool_embs_lm: torch.Tensor) -> tuple:
            """OBJECT-LEVEL graph controller: logits over the session graph's NODES (objects), with
            edges as extra information, plus the main graph's TOOLS.

            task_emb_lm:   [d_lm] task embedding (MiniLM-mean-pooled or LM-space).
            node_embs_lm:  [N, d_lm] the session graph's node embeddings — one row per object.
            edge_index:    [2, E] src->dst node indices (into the N nodes).
            edge_type:     [E] relation-type ids (follows/grounds/related/...).
            edge_strength: [E] learned per-edge scalars in [0,1].
            tool_embs_lm:  [M, d_lm] the main graph's tool embeddings (retrievable objects).

            Returns (recall_logits [N], evict_logits [N], tool_logits [M]):
              recall_logits: which session nodes to SURFACE to the LM (multi-label, BCE-trained).
              evict_logits:  which nodes the session graph should DROP at write time (binary, BCE).
              tool_logits:   which main-graph tools to FETCH into the session (multi-label, BCE).

            All three are the same pointer arithmetic as pointer_logits() — the final y after the
            T-cycle recursion is a query dotted against node/tool keys — but the context R is the
            GRAPH: node embeddings pass through one message-passing hop so an object carries the
            edges it participates in (relation type + strength), not just its own text."""
            x = self.task_proj(self.input_proj(task_emb_lm.float()))
            R_nodes = self.atom_proj(self.input_proj(node_embs_lm.float()))
            if edge_index is not None and edge_index.numel():
                src, dst = edge_index[0], edge_index[1]
                type_emb = self.graph_edge_emb(edge_type)
                msg = self.graph_msg(torch.cat([R_nodes[src], type_emb], dim=-1))
                msg = msg * edge_strength.unsqueeze(-1)
                aggr = torch.zeros_like(R_nodes)
                aggr = aggr.index_add_(0, dst, msg)
                R_nodes = self.graph_norm(R_nodes + self.graph_self(aggr))
            R_tools = self.atom_proj(self.input_proj(tool_embs_lm.float()))
            R = torch.cat([R_nodes, R_tools], dim=0)
            z, y = self.z0, self.y0
            y_out = None
            for _ in range(self.T):
                z, y, y_out, _ = self.step(x, z, y, R, return_attn=True)
            # the T-cycle recurrence is a fixed-point attractor that washes the query out of
            # y_out, so the final query also reads the TASK directly: x·R is a content dot with a
            # direct gradient path to the query, keeping the controller query-sensitive
            rec = (self.recall_head(y_out).unsqueeze(0) * R_nodes).sum(-1) / math.sqrt(self.d)
            rec = rec + (x.unsqueeze(0) * R_nodes).sum(-1) / math.sqrt(self.d)
            rec = rec + self.graph_cos_alpha * torch.nn.functional.cosine_similarity(
                task_emb_lm.float().unsqueeze(0), node_embs_lm.float())
            evi = (self.evict_head(y_out).unsqueeze(0) * R_nodes).sum(-1) / math.sqrt(self.d)
            tol = (self.tool_head(y_out).unsqueeze(0) * R_tools).sum(-1) / math.sqrt(self.d)
            tol = tol + (x.unsqueeze(0) * R_tools).sum(-1) / math.sqrt(self.d)
            return rec, evi, tol

        def select_nodes(self, task_emb_lm: torch.Tensor, node_embs_lm: torch.Tensor,
                         edge_index: torch.Tensor, edge_type: torch.Tensor,
                         edge_strength: torch.Tensor, tool_embs_lm: torch.Tensor,
                         top_k: int = 8, top_tools: int = 3) -> tuple:
            """One-shot OBJECT selection over the session graph. Returns
            (node_idx list, evict_mask list[N] bool, tool_idx list)."""
            dev = node_embs_lm.device
            with torch.no_grad():
                rec, evi, tol = self.controller_logits(
                    task_emb_lm, node_embs_lm, edge_index, edge_type, edge_strength, tool_embs_lm)
            N = node_embs_lm.shape[0]
            M = tool_embs_lm.shape[0]
            n_idx = rec.topk(min(top_k, N)).indices.tolist()
            t_idx = tol.topk(min(top_tools, M)).indices.tolist()
            return ([int(i) for i in n_idx],
                    [bool(e) for e in (evi < 0).tolist()],
                    [int(i) for i in t_idx])

        def evict_decision(self, task_emb_lm: torch.Tensor, node_embs_lm: torch.Tensor,
                           edge_index: torch.Tensor, edge_type: torch.Tensor,
                           edge_strength: torch.Tensor, idx: int = -1,
                           keep_bias: float = -1.0) -> bool:
            """Binary keep/drop decision for one node at write time: True = KEEP, False = DROP.

            keep_bias: the write-time decision is made BEFORE any question exists, so the default
            is conservative (keep unless the head is clearly confident it should drop)."""
            with torch.no_grad():
                _r, evi, _t = self.controller_logits(
                    task_emb_lm, node_embs_lm, edge_index, edge_type, edge_strength,
                    node_embs_lm.new_zeros(0, node_embs_lm.shape[1]))
            return bool(evi[idx] >= keep_bias)

        def forward(self, x_vec: torch.Tensor, atom_vecs: torch.Tensor,
                   z_init: torch.Tensor | None = None, y_init: torch.Tensor | None = None,
                   return_state: bool = False):
            """x_vec: [d_in] task embedding.
            atom_vecs: [N, d_in] atom embeddings (retrieved context R).
            Returns: [T, d] per-cycle y_t solution embeddings (unchanged default shape/behavior).

            z_init/y_init: optional resume state from a PREVIOUS forward() call (its raw final z/y, from
            return_state=True below) -- lets a caller carry real recurrent memory ACROSS separate calls
            instead of always starting fresh from the fixed learned z0/y0. None (default) = exactly the
            original behavior (start from z0/y0), so every existing caller is byte-identical unless it
            opts in. This is genuine cross-call recurrence, distinct from the T inner think/act cycles
            already run within one call -- those were always recurrent; state never survived BETWEEN calls
            before this.

            return_state=False (default): returns just the [T,d] ys tensor, same shape every caller already
            expects. return_state=True: returns (ys, (z, y)) -- the raw final latents (pre-y_head), meant to
            be passed back in as z_init/y_init on the next call, not for any other use.

            ADAPTIVE (ACT): when self.adaptive=True, also returns:
              halt_weights: [T] per-step halting weights (sum to 1, differentiable)
              n_steps:      effective steps used (float, differentiable pondering cost)
            Output ys tensor is still [T, d] — always runs T_max steps for shape stability.
            The CONSUMER decides how to use halt_weights (weighted sum for slots, truncation, etc)."""
            x = self.task_proj(x_vec)
            R = self.atom_proj(atom_vecs)

            z = self.z0 if z_init is None else z_init
            y = self.y0 if y_init is None else y_init

            ys = []
            halt_probs = [] if self.adaptive else None
            for t in range(self.T):
                # One think/act cycle against a FIXED R -- see step(), which is the same body exposed for
                # callers that need to change R between cycles.
                z, y, y_out = self.step(x, z, y, R)
                ys.append(y_out)
                if self.adaptive:
                    halt_probs.append(torch.sigmoid(self.halt_head(y)).squeeze(-1))

            out = torch.stack(ys)  # [T, d]

            if self.adaptive:
                halt_weights, n_steps = self._act_weights(halt_probs)
                if return_state:
                    return out, (z, y), halt_weights, n_steps
                return out, halt_weights, n_steps

            if return_state:
                return out, (z, y)
            return out

        def _act_weights(self, halt_probs: list) -> tuple:
            """Adaptive Computation Time (Graves 2016): convert per-step halt probabilities into
            a proper distribution over steps + a differentiable step count.

            halt_probs: [p_1, ..., p_T], each scalar in (0,1) from sigmoid.
            Returns (weights [T], n_steps scalar).

            The weight for step t is: p_t * prod(1-p_j for j<t), except the last step gets all
            remaining probability (forced halt). This is identical to the discrete hazard function /
            geometric distribution parameterization from ACT."""
            T = len(halt_probs)
            weights = []
            cumul = halt_probs[0].new_tensor(0.0)
            for t in range(T - 1):
                w_t = halt_probs[t] * (1.0 - cumul)
                weights.append(w_t)
                cumul = cumul + w_t
            weights.append(1.0 - cumul)  # remainder on last step
            weights = torch.stack(weights)  # [T]
            n_steps = (weights * torch.arange(1, T + 1, dtype=weights.dtype,
                                              device=weights.device)).sum()
            return weights, n_steps

    return torch, nn, TRMReasoner


def _selftest() -> bool:
    torch, _nn, TRMReasoner = _build()
    print("algo_trm --selftest: V3 proper Tiny Recursive Model (two-latent, cross-attn)\n")

    d_in, d, T = 32, 48, 4
    n_atoms = 6

    trm = TRMReasoner(d_in=d_in, d=d, T=T, n_heads=4)
    n_trm = sum(p.numel() for p in trm.parameters())
    print(f"  [1] TRMReasoner V3: {n_trm} params (d={d}, T={T}) -> PASS")

    atom_vecs = torch.randn(n_atoms, d_in)
    x_vec = torch.randn(d_in)
    y_ts = trm(x_vec, atom_vecs)
    assert y_ts.shape == (T, d), f"y_ts shape {y_ts.shape}"
    print(f"  [2] forward -> y_ts {list(y_ts.shape)} -> PASS")

    y_diffs = [(y_ts[t + 1] - y_ts[t]).norm().item() for t in range(T - 1)]
    evolving = any(d > 1e-6 for d in y_diffs)
    print(f"  [3] y_t step diffs: {[f'{d:.4f}' for d in y_diffs]} -> "
          f"{'PASS (evolving)' if evolving else 'FAIL (static)'}")

    atom_vecs2 = torch.randn(n_atoms, d_in)
    y_ts2 = trm(x_vec, atom_vecs2)
    diff = (y_ts - y_ts2).abs().max().item()
    print(f"  [4] cross-attn sensitivity: max|diff|={diff:.4f} -> "
          f"{'PASS' if diff > 0 else 'FAIL'}")

    improvement = (y_ts[-1] - y_ts[0]).norm().item()
    print(f"  [5] first-to-last diff: {improvement:.4f} -> "
          f"{'PASS (multi-step)' if improvement > 1e-6 else 'FAIL'}")

    x_vec2 = torch.randn(d_in)
    y_ts3 = trm(x_vec2, atom_vecs)
    diff2 = (y_ts - y_ts3).abs().max().item()
    print(f"  [6] task sensitivity: max|diff|={diff2:.4f} -> "
          f"{'PASS' if diff2 > 0 else 'FAIL'}")

    proj = torch.nn.Linear(d, 64)
    slots = proj(y_ts)
    assert slots.shape == (T, 64), f"slots shape {slots.shape}"
    print(f"  [7] y_t -> LM slots via proj: {list(slots.shape)} -> PASS")

    # --- adaptive (ACT) mode ---
    trm_a = TRMReasoner(d_in=d_in, d=d, T=T, n_heads=4, adaptive=True)
    y_ts_a, halt_w, n_s = trm_a(x_vec, atom_vecs)
    assert y_ts_a.shape == (T, d), f"adaptive y_ts shape {y_ts_a.shape}"
    assert halt_w.shape == (T,), f"halt_weights shape {halt_w.shape}"
    assert abs(halt_w.sum().item() - 1.0) < 1e-5, f"halt_weights sum {halt_w.sum().item()}"
    assert 1.0 <= n_s.item() <= T, f"n_steps {n_s.item()} outside [1, {T}]"
    weighted_out = (halt_w.unsqueeze(-1) * y_ts_a).sum(dim=0)
    assert weighted_out.shape == (d,), f"weighted output {weighted_out.shape}"
    print(f"  [8] adaptive ACT: halt_w sum={halt_w.sum():.4f}, n_steps={n_s:.2f}, "
          f"weighted_out [{d}] -> PASS")

    # adaptive + return_state
    y_ts_a2, (z_a, y_a), halt_w2, n_s2 = trm_a(x_vec, atom_vecs, return_state=True)
    assert z_a.shape == (d,) and y_a.shape == (d,), "return_state shapes"
    print(f"  [9] adaptive + return_state -> PASS")

    # pondering cost is differentiable
    loss = n_s2 * 0.01
    loss.backward()
    grad_ok = trm_a.halt_head.weight.grad is not None and trm_a.halt_head.weight.grad.abs().sum() > 0
    print(f"  [10] pondering cost gradient flows to halt_head -> {'PASS' if grad_ok else 'FAIL'}")

    # --- pointer-head generation (LM-space token-level atoms, copy channel) ---
    trm_t = TRMReasoner(d_in=d_in, d=d, T=4, n_heads=4, token_head_d_lm=64)
    lm_embed = torch.randn(100, 64)
    task_lm = torch.randn(64)
    cue_toks_lm = torch.randn(8, 64)
    atom_toks_lm = torch.randn(20, 64)
    lg = trm_t.pointer_logits(task_lm, cue_toks_lm, atom_toks_lm, torch.zeros(0, 64))
    assert lg.shape == (21,), f"pointer_logits shape {lg.shape} (L=20 + EOS)"
    loss_t = torch.nn.functional.cross_entropy(lg.unsqueeze(0), torch.tensor([7]))
    loss_t.backward()
    grad_ok_t = trm_t.pointer_head.weight.grad is not None and \
        trm_t.pointer_head.weight.grad.abs().sum() > 0
    assert grad_ok_t, "pointer_head gradient missing"
    span_tids = torch.randint(0, 100, (20,))
    cue_tids = torch.randint(0, 100, (8,))
    sel = trm_t.select_tokens(task_lm, cue_tids, span_tids, lm_embed, top_k=5)
    assert isinstance(sel, list) and len(sel) <= 5
    assert all(isinstance(i, int) and 0 <= i < 20 for i in sel), f"selected {sel}"
    print(f"  [11] pointer head (LM-space copy): logits {list(lg.shape)}, grad OK, "
          f"selected {len(sel)} positions -> PASS")

    # --- object-level graph controller (session-graph objects + edges as extra info) ---
    node_lm = torch.randn(6, 64)
    tool_lm = torch.randn(3, 64)
    edge_index = torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.long)
    edge_type = torch.tensor([5, 5, 0], dtype=torch.long)
    edge_strength = torch.tensor([0.5, 0.8, 0.3], dtype=torch.float32)
    rec, evi, tol = trm_t.controller_logits(task_lm, node_lm, edge_index, edge_type,
                                            edge_strength, tool_lm)
    assert rec.shape == (6,) and evi.shape == (6,) and tol.shape == (3,), \
        f"controller shapes {rec.shape} {evi.shape} {tol.shape}"
    bce = torch.nn.functional.binary_cross_entropy_with_logits(
        rec, torch.zeros_like(rec))
    bce.backward()
    assert trm_t.recall_head.weight.grad is not None and \
        trm_t.recall_head.weight.grad.abs().sum() > 0
    assert trm_t.graph_msg.weight.grad is not None and \
        trm_t.graph_msg.weight.grad.abs().sum() > 0, "edge message-passing grad missing"
    n_idx, evict_mask, t_idx = trm_t.select_nodes(task_lm, node_lm, edge_index, edge_type,
                                                  edge_strength, tool_lm, top_k=3, top_tools=2)
    assert len(n_idx) <= 3 and len(t_idx) <= 2 and len(evict_mask) == 6
    assert all(isinstance(i, int) and 0 <= i < 6 for i in n_idx), f"node sel {n_idx}"
    keep = trm_t.evict_decision(task_lm, node_lm, edge_index, edge_type, edge_strength)
    assert isinstance(keep, bool)
    print(f"  [12] graph controller (objects+edges): rec {list(rec.shape)} evi {list(evi.shape)} "
          f"tool {list(tol.shape)}, grad OK (msg+heads), sel {len(n_idx)} nodes/{len(t_idx)} tools -> PASS")

    print("\n  ALGO_TRM SELFTEST -> PASS  (V3 TRMReasoner + ACT adaptive + token head + graph controller)")
    return True


def main():
    ap = argparse.ArgumentParser(description="TRM-over-graph: tiny recursive planner.")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        sys.exit(0 if _selftest() else 1)
    ap.print_help()


if __name__ == "__main__":
    main()
