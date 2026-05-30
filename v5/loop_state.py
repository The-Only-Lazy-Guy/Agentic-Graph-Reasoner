"""LoopState dataclass and six auxiliary prediction heads.

Auxiliary heads (all small MLPs trained jointly with LoRA):
  slot_head         -> slot fill confidence per slot name
  node_head         -> per-node attention logit adjustment
  state_overlay_head-> additive K, V deltas from loop state (no GNN re-run)
  epistemic_head    -> per-node epistemic confidence
  invalidator_head  -> which invalidator flags are active
  shortcut_head     -> is current top node safe to shortcut?

All heads take h_r [B, d_lm] as primary input and produce their respective
outputs. state_overlay_head additionally takes the full loop state tensors.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import torch
import torch.nn as nn
from torch import Tensor

from v5.goal_encoder import GOAL_DIM, NUM_SLOTS
from v5.gnn_encoder import GNN_HIDDEN_DIM

LM_HIDDEN_DIM = 2560    # Qwen3-4B
SLOT_HEAD_DIM = 64
NODE_HEAD_DIM = 64


@dataclass
class LoopState:
    """Complete mutable state propagated between recurrent loop iterations."""
    h_r: Tensor                                      # [1, d_lm] LM hidden state
    slot_state_r: Tensor                             # [1, NUM_SLOTS] fill confidence 0-1
    node_scores_r: Tensor                            # [1, N] per-node attention logit scores
    shortcut_validity_r: Tensor                      # [1, 1] scalar 0-1
    epistemic_confidence_r: Tensor                   # [1, N] per-node belief confidence
    invalidator_flags_r: Tensor                      # [1, N] float (>0.5 = fired)
    loop_idx: int = 0
    exit_reason: Optional[str] = None

    # Readable snapshot for logging / corpus
    def to_log_entry(
        self,
        node_ids: List[str],
        layer: int,
        top_k: int = 5,
    ) -> Dict:
        scores = self.node_scores_r.squeeze(0).tolist()       # [N]
        epi = self.epistemic_confidence_r.squeeze(0).tolist() # [N]
        inv = self.invalidator_flags_r.squeeze(0).tolist()    # [N]
        slots = self.slot_state_r.squeeze(0).tolist()         # [NUM_SLOTS]

        top_indices = sorted(range(len(scores)), key=lambda i: -scores[i])[:top_k]
        top_nodes = [(node_ids[i], round(scores[i], 4)) for i in top_indices]

        return {
            "layer": layer,
            "loop": self.loop_idx,
            "top_nodes": top_nodes,
            "slot_fill_confidence": [round(s, 3) for s in slots],
            "shortcut_validity": round(float(self.shortcut_validity_r.item()), 4),
            "epistemic_confidence_top": {
                node_ids[i]: round(epi[i], 3) for i in top_indices
            },
            "invalidator_flags_top": {
                node_ids[i]: bool(inv[i] > 0.5) for i in top_indices
            },
            "exit_reason": self.exit_reason,
        }


class SlotHead(nn.Module):
    """h_r -> [B, NUM_SLOTS] slot fill confidence (0-1)."""

    def __init__(self, lm_hidden_dim: int = LM_HIDDEN_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(lm_hidden_dim, SLOT_HEAD_DIM),
            nn.GELU(),
            nn.Linear(SLOT_HEAD_DIM, NUM_SLOTS),
            nn.Sigmoid(),
        )

    def forward(self, h: Tensor) -> Tensor:
        return self.net(h)   # [B, NUM_SLOTS]


class NodeHead(nn.Module):
    """(h_r, node_embeddings) -> [B, N] per-node logit adjustments."""

    def __init__(self, lm_hidden_dim: int = LM_HIDDEN_DIM, gnn_dim: int = GNN_HIDDEN_DIM):
        super().__init__()
        self.h_proj = nn.Linear(lm_hidden_dim, NODE_HEAD_DIM)
        self.n_proj = nn.Linear(gnn_dim, NODE_HEAD_DIM)

    def forward(self, h: Tensor, node_embeddings: Tensor) -> Tensor:
        """
        h: [B, d_lm]
        node_embeddings: [N, gnn_dim]
        returns: [B, N]
        """
        h_p = self.h_proj(h)                    # [B, NODE_HEAD_DIM]
        n_p = self.n_proj(node_embeddings)      # [N, NODE_HEAD_DIM]
        return h_p @ n_p.T                      # [B, N]


class StateOverlayHead(nn.Module):
    """Loop state -> additive (delta_K, delta_V) for K_r/V_r update.

    Inputs: concatenation of [slot_state, node_scores_top, shortcut_validity,
                               epistemic_top, invalidator_top]
    Output: per-node delta projected to GNN_HIDDEN_DIM for both K and V.

    'top' means the top-16 node scores are used to keep input dim fixed
    regardless of subgraph size.
    """
    TOP_K = 16
    STATE_SUMMARY_DIM = NUM_SLOTS + TOP_K + 1 + TOP_K + TOP_K  # slots+scores+shortcut+epi+inv

    def __init__(self, gnn_dim: int = GNN_HIDDEN_DIM):
        super().__init__()
        self.gnn_dim = gnn_dim
        in_dim = self.STATE_SUMMARY_DIM
        self.delta_K = nn.Sequential(
            nn.Linear(in_dim, 128), nn.GELU(), nn.Linear(128, gnn_dim),
        )
        self.delta_V = nn.Sequential(
            nn.Linear(in_dim, 128), nn.GELU(), nn.Linear(128, gnn_dim),
        )

    def forward(self, state: LoopState, N: int) -> tuple[Tensor, Tensor]:
        """Return (delta_K, delta_V), each [N, gnn_dim].

        The overlay is broadcast: same delta added to every node's K and V.
        The node_head provides per-node adjustment; this provides global
        loop-state context.
        """
        K = self.TOP_K
        # node_scores carry a -1e9 sentinel on out-of-pool nodes (pool masking in
        # RecurrentAttentionBlock). Clamp before summarizing so the sentinel does
        # not blow up the delta MLP — feeding -1e9 here makes delta_K/delta_V
        # explode, which in turn explodes K_r/V_r and the residual stream.
        scores = state.node_scores_r.squeeze(0).clamp(-30.0, 30.0)  # [N]
        epi = state.epistemic_confidence_r.squeeze(0) # [N]
        inv = state.invalidator_flags_r.squeeze(0)    # [N]

        # Top-K summary (fixed size regardless of N)
        top_idx = scores.topk(min(K, N)).indices
        scores_top = torch.zeros(K, device=scores.device)
        epi_top = torch.zeros(K, device=epi.device)
        inv_top = torch.zeros(K, device=inv.device)
        scores_top[:len(top_idx)] = scores[top_idx]
        epi_top[:len(top_idx)] = epi[top_idx]
        inv_top[:len(top_idx)] = inv[top_idx]

        summary = torch.cat([
            state.slot_state_r.squeeze(0),  # [NUM_SLOTS]
            scores_top,                      # [K]
            state.shortcut_validity_r.view(1), # [1]
            epi_top,                         # [K]
            inv_top,                         # [K]
        ])                                   # [STATE_SUMMARY_DIM]

        s = summary.unsqueeze(0)             # [1, STATE_SUMMARY_DIM]
        dK = self.delta_K(s).squeeze(0)     # [gnn_dim]
        dV = self.delta_V(s).squeeze(0)     # [gnn_dim]

        # Broadcast to all nodes
        return dK.unsqueeze(0).expand(N, -1), dV.unsqueeze(0).expand(N, -1)


class EpistemicHead(nn.Module):
    """(h_r, node_embeddings) -> [B, N] per-node epistemic confidence (0-1).

    Concat-MLP interaction (not pure bilinear): epistemic status of a node is
    context-gated — e.g. a fact may be 'supported' in one task context and
    'unsupported' in another, while a verified node stays high regardless. A
    bilinear h·n score cannot hold one node constant while gating another by
    context through the same h; the MLP over [h, n, h⊙n] has the capacity to.
    """

    def __init__(self, lm_hidden_dim: int = LM_HIDDEN_DIM, gnn_dim: int = GNN_HIDDEN_DIM):
        super().__init__()
        self.h_proj = nn.Linear(lm_hidden_dim, 64)
        self.n_proj = nn.Linear(gnn_dim, 64)
        self.score = nn.Sequential(
            nn.Linear(64 * 3, 64), nn.GELU(), nn.Linear(64, 1),
        )
        self.sig = nn.Sigmoid()

    def forward(self, h: Tensor, node_embeddings: Tensor) -> Tensor:
        h_p = self.h_proj(h)                          # [B, 64]
        n_p = self.n_proj(node_embeddings)            # [N, 64]
        B, N = h_p.shape[0], n_p.shape[0]
        h_e = h_p.unsqueeze(1).expand(B, N, -1)       # [B, N, 64]
        n_e = n_p.unsqueeze(0).expand(B, N, -1)       # [B, N, 64]
        feat = torch.cat([h_e, n_e, h_e * n_e], dim=-1)  # [B, N, 192]
        return self.sig(self.score(feat).squeeze(-1))    # [B, N]


class InvalidatorHead(nn.Module):
    """(h_r, node_embeddings) -> [B, N] invalidator fire probability (0-1).

    High score = invalidator for this node is active in current context.
    Concat-MLP interaction (same rationale as EpistemicHead): whether a node's
    structural invalidator is ACTIVE is context-gated — the same node may fire in
    one task context and not another. A bilinear h·n score is too weak for this
    per-node context gating; the MLP over [h, n, h⊙n] handles it.
    """

    def __init__(self, lm_hidden_dim: int = LM_HIDDEN_DIM, gnn_dim: int = GNN_HIDDEN_DIM):
        super().__init__()
        self.h_proj = nn.Linear(lm_hidden_dim, 64)
        self.n_proj = nn.Linear(gnn_dim, 64)
        self.score = nn.Sequential(
            nn.Linear(64 * 3, 64), nn.GELU(), nn.Linear(64, 1),
        )
        self.sig = nn.Sigmoid()

    def forward(self, h: Tensor, node_embeddings: Tensor) -> Tensor:
        h_p = self.h_proj(h)                          # [B, 64]
        n_p = self.n_proj(node_embeddings)            # [N, 64]
        B, N = h_p.shape[0], n_p.shape[0]
        h_e = h_p.unsqueeze(1).expand(B, N, -1)
        n_e = n_p.unsqueeze(0).expand(B, N, -1)
        feat = torch.cat([h_e, n_e, h_e * n_e], dim=-1)  # [B, N, 192]
        return self.sig(self.score(feat).squeeze(-1))    # [B, N]


class ShortcutHead(nn.Module):
    """h_r -> [B, 1] shortcut safety score (0-1)."""

    def __init__(self, lm_hidden_dim: int = LM_HIDDEN_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(lm_hidden_dim, 64),
            nn.GELU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(self, h: Tensor) -> Tensor:
        return self.net(h)   # [B, 1]


class AuxHeads(nn.Module):
    """Container for all six auxiliary heads. Trained jointly with LoRA."""

    def __init__(
        self,
        lm_hidden_dim: int = LM_HIDDEN_DIM,
        gnn_dim: int = GNN_HIDDEN_DIM,
    ):
        super().__init__()
        # The recurrent block uses a pre-norm residual stream, so h_r grows in
        # magnitude across iterations. Normalize h before the heads read it
        # (GPT-style final ln_f): bounds magnitude so the sigmoid heads don't
        # saturate, while preserving the direction that carries the LM/family
        # signal. Without this, a large-magnitude h_r drives every head output
        # to the same saturated value regardless of input.
        self.head_norm = nn.LayerNorm(lm_hidden_dim)
        self.slot = SlotHead(lm_hidden_dim)
        self.node = NodeHead(lm_hidden_dim, gnn_dim)
        self.overlay = StateOverlayHead(gnn_dim)
        self.epistemic = EpistemicHead(lm_hidden_dim, gnn_dim)
        self.invalidator = InvalidatorHead(lm_hidden_dim, gnn_dim)
        self.shortcut = ShortcutHead(lm_hidden_dim)

    def update_state(
        self,
        state: LoopState,
        node_embeddings: Tensor,        # [N, gnn_dim]
        static_inv: Optional[Tensor] = None,  # [1, N] from graph structure
    ) -> LoopState:
        """Run all heads and return a new LoopState with updated predictions.

        Invalidator combining: static_inv marks nodes that structurally HAVE an
        outgoing invalidated_by edge. The neural head predicts whether that
        invalidator is ACTIVE in the current context.

            combined = static_inv * dynamic_inv

        This prevents the head from firing on nodes with no structural invalidator,
        and prevents the head from suppressing real structural invalidators entirely.
        """
        h = self.head_norm(state.h_r)                          # [1, d_lm] normalized for heads
        N = node_embeddings.shape[0]

        new_slot = self.slot(h)                                # [1, NUM_SLOTS]
        new_node_adj = self.node(h, node_embeddings)           # [1, N]
        new_shortcut = self.shortcut(h)                        # [1, 1]
        new_epistemic = self.epistemic(h, node_embeddings)     # [1, N]
        dynamic_inv = self.invalidator(h, node_embeddings)     # [1, N]  0-1

        # Combine: structural gate × dynamic activation
        if static_inv is not None:
            combined_inv = static_inv * dynamic_inv            # [1, N]
        else:
            combined_inv = dynamic_inv

        # Node scores = previous scores + learned adjustment
        new_scores = state.node_scores_r + new_node_adj        # [1, N]

        return LoopState(
            h_r=state.h_r,
            slot_state_r=new_slot,
            node_scores_r=new_scores,
            shortcut_validity_r=new_shortcut,
            epistemic_confidence_r=new_epistemic,
            invalidator_flags_r=combined_inv,
            loop_idx=state.loop_idx + 1,
            exit_reason=None,
        )
