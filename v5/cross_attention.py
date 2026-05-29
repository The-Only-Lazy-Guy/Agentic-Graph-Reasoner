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
        h_r: Tensor,          # [B, d_lm]
        goal: Tensor,          # [B, GOAL_DIM]
        slot_state: Tensor,    # [B, NUM_SLOTS]
        K_r: Tensor,           # [N, attn_dim]  (pre-projected or raw GNN emb)
        V_r: Tensor,           # [N, attn_dim]
    ) -> Tensor:
        """Return updated h_{r+1}: [B, d_lm]."""
        q_input = torch.cat([h_r, goal, slot_state], dim=-1)   # [B, Q_INPUT_DIM]
        Q = self.W_q(q_input)                                   # [B, attn_dim]

        # K, V are pre-projected from GNN embeddings
        attn_weights = torch.softmax(
            Q @ K_r.T / self.scale, dim=-1
        )                                                       # [B, N]

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
        h_init: Tensor,            # [1, d_lm]   LM hidden state entering this layer
        goal: Tensor,              # [1, GOAL_DIM]
        base_node_embeddings: Tensor,  # [N, GNN_HIDDEN_DIM]  (fixed, from GNN)
        node_ids: List[str],
        r_max: Optional[int] = None,
        task_frame: Optional[dict] = None,
    ) -> Tuple[Tensor, LoopState, List[dict]]:
        """Run the recurrent loop.

        Returns:
            h_final: [1, d_lm] updated hidden state
            final_state: LoopState at exit
            loop_log: list of per-iteration log dicts for corpus
        """
        r_max = r_max if r_max is not None else self.r_max
        N = base_node_embeddings.shape[0]
        device = h_init.device

        # Pre-project base K, V (done once, outside loop)
        base_K = self.K_proj(base_node_embeddings)   # [N, CROSS_ATTN_DIM]
        base_V = self.V_proj(base_node_embeddings)   # [N, CROSS_ATTN_DIM]

        # Initialize loop state
        state = LoopState(
            h_r=h_init,
            slot_state_r=torch.zeros(1, _NUM_SLOTS, device=device),
            node_scores_r=torch.zeros(1, N, device=device),
            shortcut_validity_r=torch.zeros(1, 1, device=device),
            epistemic_confidence_r=torch.zeros(1, N, device=device),
            invalidator_flags_r=torch.zeros(1, N, device=device),
            loop_idx=0,
        )

        loop_log: List[dict] = []

        for r in range(r_max):
            # 1. Compute dynamic K/V overlay from current state
            delta_K, delta_V = self.aux.overlay(state, N)   # [N, gnn_dim]

            # Project overlay to attn_dim and add to base
            # delta_K/V are in gnn_dim; we project via K_proj/V_proj
            K_r = base_K + self.K_proj(delta_K)             # [N, CROSS_ATTN_DIM]
            V_r = base_V + self.V_proj(delta_V)             # [N, CROSS_ATTN_DIM]

            # 2. Cross-attend and update hidden state
            h_new, attn_weights = self.proj(
                state.h_r, goal, state.slot_state_r, K_r, V_r
            )

            # 3. Update loop state with new h and aux head predictions
            state = LoopState(
                h_r=h_new,
                slot_state_r=state.slot_state_r,
                node_scores_r=attn_weights,         # [1, N] — attention weights as scores
                shortcut_validity_r=state.shortcut_validity_r,
                epistemic_confidence_r=state.epistemic_confidence_r,
                invalidator_flags_r=state.invalidator_flags_r,
                loop_idx=r,
            )
            state = self.aux.update_state(state, base_node_embeddings)

            # 4. Log
            loop_log.append(state.to_log_entry(node_ids, layer=self.layer_id))

            # 5. Exit condition
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
        node_embeddings: Tensor,
        node_ids: List[str],
        r_max: Optional[int] = None,
        task_frame: Optional[dict] = None,
    ) -> Tuple[Tensor, LoopState, List[dict]]:
        return self.planning_block(h, goal, node_embeddings, node_ids, r_max, task_frame)

    def run_evidence(
        self,
        h: Tensor,
        goal: Tensor,
        node_embeddings: Tensor,
        node_ids: List[str],
        r_max: Optional[int] = None,
        task_frame: Optional[dict] = None,
    ) -> Tuple[Tensor, LoopState, List[dict]]:
        return self.evidence_block(h, goal, node_embeddings, node_ids, r_max, task_frame)
