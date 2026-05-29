"""Recurrent cross-attention block: one iteration of the planning or evidence loop.

Implements the per-iteration update from V5_ARCHITECTURE.md §6:

    Q_r    = W_q(concat(h_r, goal_vector, slot_state_r))
    A_r    = softmax(Q_r @ K_r.T / sqrt(d)) @ V_r
    h_{r+1} = LayerNorm(h_r + W_o(A_r))
    [then aux heads update loop state]

W_q and W_o are LoRA-adapted projections (base weights from Qwen3 frozen;
LoRA delta trained). At inference before fine-tuning, they start as identity-
style projections (random init will be replaced by trained weights).

K_r = base_K + overlay_r.K
V_r = base_V + overlay_r.V

base_K, base_V come from the GNN (run once per session, fixed).
overlay_r comes from StateOverlayHead (cheap, per-loop).
"""
from __future__ import annotations

import math
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from v5.exit_condition import fallback_needed, should_exit_loop
from v5.gnn_encoder import GNN_HIDDEN_DIM
from v5.goal_encoder import GOAL_DIM
from v5.loop_state import AuxHeads, LoopState, LM_HIDDEN_DIM

# Cross-attention projection dim (Q, K, V all projected here)
CROSS_ATTN_DIM = 512
# Input to W_q: h_r || goal_vector || slot_state_r
from v5.goal_encoder import NUM_SLOTS as _NUM_SLOTS
Q_INPUT_DIM = LM_HIDDEN_DIM + GOAL_DIM + _NUM_SLOTS   # 2560 + 128 + 10 = 2698

_NEG_INF = -1e9   # mask value added to logits before softmax


class CrossAttentionProjections(nn.Module):
    """LoRA-wrapped Q, K, V, O projections for one cross-attention block.

    Before fine-tuning: these are plain linear layers (random init).
    After LoRA training: base weights frozen, delta applied.

    For now we implement as plain Linear layers. LoRA wrapping is applied
    by the training pipeline (peft library) before Phase 16 training starts.
    """

    def __init__(
        self,
        q_input_dim: int = Q_INPUT_DIM,
        kv_input_dim: int = GNN_HIDDEN_DIM,
        attn_dim: int = CROSS_ATTN_DIM,
        lm_hidden_dim: int = LM_HIDDEN_DIM,
    ):
        super().__init__()
        self.W_q = nn.Linear(q_input_dim, attn_dim, bias=False)
        self.W_k = nn.Linear(kv_input_dim, attn_dim, bias=False)
        self.W_v = nn.Linear(kv_input_dim, attn_dim, bias=False)
        self.W_o = nn.Linear(attn_dim, lm_hidden_dim, bias=False)
        self.norm = nn.LayerNorm(lm_hidden_dim)
        self.scale = math.sqrt(attn_dim)

    def forward(
        self,
        h_r: Tensor,                    # [B, d_lm]
        goal: Tensor,                   # [B, GOAL_DIM]
        slot_state: Tensor,             # [B, NUM_SLOTS]
        K_r: Tensor,                    # [N, attn_dim]
        V_r: Tensor,                    # [N, attn_dim]
        node_mask: Optional[Tensor] = None,  # [N] bool — True = attend; None = attend all
    ) -> Tuple[Tensor, Tensor]:
        """Return (updated h_{r+1}, attn_weights): ([B, d_lm], [B, N])."""
        q_input = torch.cat([h_r, goal, slot_state], dim=-1)   # [B, Q_INPUT_DIM]
        Q = self.W_q(q_input)                                   # [B, attn_dim]

        logits = Q @ K_r.T / self.scale                        # [B, N]

        # Apply node pool mask: nodes outside the pool get -inf before softmax
        if node_mask is not None:
            # node_mask: [N] bool, True = allowed
            mask_val = torch.zeros_like(logits)
            mask_val[:, ~node_mask] = _NEG_INF
            logits = logits + mask_val

        attn_weights = torch.softmax(logits, dim=-1)           # [B, N]

        A = attn_weights @ V_r                                  # [B, attn_dim]
        h_new = self.norm(h_r + self.W_o(A))                   # [B, d_lm]
        return h_new, attn_weights


class RecurrentAttentionBlock(nn.Module):
    """One full recurrent attention loop (Layer 8 planning OR Layer 20 evidence).

    Runs R iterations. Each iteration:
      1. StateOverlayHead produces (delta_K, delta_V) from current loop state
      2. K_r = base_K + delta_K,  V_r = base_V + delta_V
      3. CrossAttentionProjections attends and updates h_r
      4. AuxHeads update LoopState
      5. Exit condition checked

    Returns final LoopState and log entries for corpus.
    """

    def __init__(
        self,
        projections: CrossAttentionProjections,
        aux_heads: AuxHeads,
        layer_id: int,       # 8 (planning) or 20 (evidence)
        r_max: int = 4,
    ):
        super().__init__()
        self.proj = projections
        self.aux = aux_heads
        self.layer_id = layer_id
        self.r_max = r_max

        # Pre-project base K, V from GNN dim to attn_dim (shared, no LoRA needed)
        self.K_proj = nn.Linear(GNN_HIDDEN_DIM, CROSS_ATTN_DIM, bias=False)
        self.V_proj = nn.Linear(GNN_HIDDEN_DIM, CROSS_ATTN_DIM, bias=False)

    def forward(
        self,
        h_init: Tensor,
        goal: Tensor,
        graph_kv,                      # GraphMemoryKV (or Tensor for backward compat)
        node_ids: Optional[List[str]] = None,
        r_max: Optional[int] = None,
        task_frame: Optional[dict] = None,
    ) -> Tuple[Tensor, LoopState, List[dict]]:
        """Run the recurrent loop.

        Accepts either a GraphMemoryKV (preferred) or a raw [N, GNN_HIDDEN_DIM]
        tensor (backward-compatible path used in tests).

        Returns:
            h_final: [1, d_lm] updated hidden state
            final_state: LoopState at exit
            loop_log: list of per-iteration log dicts for corpus
        """
        from v5.subgraph import GraphMemoryKV

        r_max = r_max if r_max is not None else self.r_max

        # Unpack GraphMemoryKV or fall back to raw tensor
        if isinstance(graph_kv, GraphMemoryKV):
            base_node_embeddings = graph_kv.node_embeddings   # [N, GNN_HIDDEN_DIM]
            base_K = graph_kv.K                               # already projected
            base_V = graph_kv.V
            node_ids = node_ids or graph_kv.node_ids
            # Select correct mask for this block (planning vs evidence)
            if self.layer_id == 8:
                node_mask = graph_kv.planning_mask             # [N] bool
            else:
                node_mask = graph_kv.evidence_mask             # [N] bool
            # Pre-init invalidator flags from static graph structure
            static_inv = graph_kv.invalidator_flags.unsqueeze(0)  # [1, N]
        else:
            # Raw tensor path (tests / pre-GraphMemoryKV callers)
            base_node_embeddings = graph_kv
            base_K = self.K_proj(base_node_embeddings)
            base_V = self.V_proj(base_node_embeddings)
            node_mask = None
            static_inv = None

        N = base_node_embeddings.shape[0]
        device = h_init.device

        # Initialize loop state
        init_inv = static_inv if static_inv is not None else torch.zeros(1, N, device=device)
        state = LoopState(
            h_r=h_init,
            slot_state_r=torch.zeros(1, _NUM_SLOTS, device=device),
            node_scores_r=torch.zeros(1, N, device=device),
            shortcut_validity_r=torch.zeros(1, 1, device=device),
            epistemic_confidence_r=torch.zeros(1, N, device=device),
            invalidator_flags_r=init_inv,
            loop_idx=0,
        )

        loop_log: List[dict] = []

        for r in range(r_max):
            # 1. Dynamic K/V overlay from current loop state
            delta_K, delta_V = self.aux.overlay(state, N)
            K_r = base_K + self.K_proj(delta_K)
            V_r = base_V + self.V_proj(delta_V)

            # 2. Cross-attend with node pool mask
            h_new, attn_weights = self.proj(
                state.h_r, goal, state.slot_state_r, K_r, V_r,
                node_mask=node_mask,
            )

            # 3. Update loop state
            state = LoopState(
                h_r=h_new,
                slot_state_r=state.slot_state_r,
                node_scores_r=attn_weights,
                shortcut_validity_r=state.shortcut_validity_r,
                epistemic_confidence_r=state.epistemic_confidence_r,
                invalidator_flags_r=state.invalidator_flags_r,
                loop_idx=r,
            )
            state = self.aux.update_state(state, base_node_embeddings)

            # 4. Log
            loop_log.append(state.to_log_entry(node_ids or [], layer=self.layer_id))

            # 5. Exit check
            should_exit, reason = should_exit_loop(state, r, r_max, task_frame)
            if should_exit:
                state.exit_reason = reason
                break

        return state.h_r, state, loop_log


class V5AttentionAdapter(nn.Module):
    """Full V5 adapter: two recurrent blocks (L8 planning + L20 evidence).

    Designed to be injected into Qwen3-4B via forward hooks.
    The LM's own weights remain frozen; only this module trains.
    """

    def __init__(
        self,
        r_plan: int = 4,
        r_evidence: int = 6,
    ):
        super().__init__()
        aux_heads = AuxHeads()

        plan_proj = CrossAttentionProjections()
        evid_proj = CrossAttentionProjections()

        self.planning_block = RecurrentAttentionBlock(
            projections=plan_proj,
            aux_heads=aux_heads,
            layer_id=8,
            r_max=r_plan,
        )
        self.evidence_block = RecurrentAttentionBlock(
            projections=evid_proj,
            aux_heads=aux_heads,    # shared aux heads across both blocks
            layer_id=20,
            r_max=r_evidence,
        )
        # Keep a reference so training can access both
        self.aux_heads = aux_heads

    def run_planning(
        self,
        h: Tensor,
        goal: Tensor,
        graph_kv,                      # GraphMemoryKV or raw Tensor
        node_ids: Optional[List[str]] = None,
        r_max: Optional[int] = None,
        task_frame: Optional[dict] = None,
    ) -> Tuple[Tensor, LoopState, List[dict]]:
        return self.planning_block(h, goal, graph_kv, node_ids, r_max, task_frame)

    def run_evidence(
        self,
        h: Tensor,
        goal: Tensor,
        graph_kv,                      # GraphMemoryKV or raw Tensor
        node_ids: Optional[List[str]] = None,
        r_max: Optional[int] = None,
        task_frame: Optional[dict] = None,
    ) -> Tuple[Tensor, LoopState, List[dict]]:
        return self.evidence_block(h, goal, graph_kv, node_ids, r_max, task_frame)
