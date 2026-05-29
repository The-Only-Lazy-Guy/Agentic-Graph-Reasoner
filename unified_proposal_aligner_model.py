from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import torch
import torch.nn as nn

from pred_model import (
    COMMIT_FAMILIES,
    LOGIT_MASK_VALUE,
    MEM_LINK_KINDS,
    REL_WITH_NONE,
    SPAN_KIND_VOCAB,
    _maybe_emb_proj,
)


UNIFIED_NODE_TYPE_VOCAB = ["concept", "bridge"]
UNIFIED_NODE_TYPE_TO_ID = {x: i for i, x in enumerate(UNIFIED_NODE_TYPE_VOCAB)}


@dataclass
class UnifiedBatch:
    """Unified end-to-end minibatch.

    Shapes:
      signal_bow: [B, H_hash]
      cand_*: [B, C, ...]
      mem_*: [B, M, ...]
      slot_mask: [B, K]
      y_* tensors mirror the corresponding heads.
    """

    signal_bow: torch.Tensor
    cand_bow: torch.Tensor
    cand_emb: torch.Tensor
    cand_kind_ids: torch.Tensor
    cand_feat: torch.Tensor
    cand_pair_feat: torch.Tensor
    cand_mask: torch.Tensor
    mem_bow: torch.Tensor
    mem_emb: torch.Tensor
    mem_feat: torch.Tensor
    mem_mask: torch.Tensor
    mem_pair_feat: torch.Tensor
    slot_ids: torch.Tensor
    slot_mask: torch.Tensor
    y_use: torch.Tensor
    y_span: torch.Tensor
    y_is_bridge: torch.Tensor
    y_commit: torch.Tensor
    y_edge_exist: torch.Tensor
    y_edge_rel: torch.Tensor
    y_mem_kind: torch.Tensor
    y_mem_rel: torch.Tensor
    y_mixed_dst_mem: torch.Tensor
    y_bridge_mem_a: torch.Tensor
    y_bridge_mem_b: torch.Tensor
    edge_pair_feat: torch.Tensor


class UnifiedProposalAlignerNet(nn.Module):
    """Single-model proposer+aligner.

    The model predicts:
      - per-slot use / anchor span / bridge type
      - pairwise directed edges + edge relation
      - slot-to-memory link kind + relation
      - commit family
      - template arguments for synthesis slots

    Slot text is derived outside the model from the predicted pointers.
    """

    def __init__(
        self,
        *,
        hash_dim: int = 512,
        hidden_dim: int = 256,
        k_max: int = 3,
        cand_feat_dim: int = 2,
        cand_pair_feat_dim: int = 3,
        mem_feat_dim: int = 3,
        cand_emb_dim: int = 0,
        mem_emb_dim: int = 0,
        edge_pair_feat_dim: int = 0,
        use_verifier: bool = False,
        attention_heads: int = 4,
        attention_dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.hash_dim = hash_dim
        self.hidden_dim = hidden_dim
        self.use_verifier = use_verifier
        self.k_max = k_max
        self.cand_emb_dim = cand_emb_dim
        self.mem_emb_dim = mem_emb_dim
        self.cand_pair_feat_dim = cand_pair_feat_dim
        self.use_cand_self_attn = True

        self.slot_emb = nn.Embedding(k_max, hidden_dim)
        self.span_kind_emb = nn.Embedding(len(SPAN_KIND_VOCAB), 16)

        self.signal_proj = nn.Sequential(
            nn.Linear(hash_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.cand_proj = nn.Sequential(
            nn.Linear(hash_dim + 16 + cand_feat_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.cand_emb_proj = _maybe_emb_proj(cand_emb_dim, hidden_dim)
        self.mem_proj = nn.Sequential(
            nn.Linear(hash_dim + mem_feat_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.mem_emb_proj = _maybe_emb_proj(mem_emb_dim, hidden_dim)
        self.state_proj = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.slot_cand_attn = nn.MultiheadAttention(
            hidden_dim,
            num_heads=attention_heads,
            dropout=attention_dropout,
            batch_first=True,
        )
        self.cand_self_attn = nn.MultiheadAttention(
            hidden_dim,
            num_heads=attention_heads,
            dropout=attention_dropout,
            batch_first=True,
        )
        self.slot_mem_attn = nn.MultiheadAttention(
            hidden_dim,
            num_heads=attention_heads,
            dropout=attention_dropout,
            batch_first=True,
        )
        self.slot_self_attn = nn.MultiheadAttention(
            hidden_dim,
            num_heads=attention_heads,
            dropout=attention_dropout,
            batch_first=True,
        )
        nn.init.zeros_(self.cand_self_attn.out_proj.weight)
        nn.init.zeros_(self.cand_self_attn.out_proj.bias)
        self.slot_norm1 = nn.LayerNorm(hidden_dim)
        self.slot_norm2 = nn.LayerNorm(hidden_dim)
        self.slot_norm3 = nn.LayerNorm(hidden_dim)
        self.cand_self_norm = nn.LayerNorm(hidden_dim)

        self.use_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )
        self.type_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )
        self.span_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.commit_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, len(COMMIT_FAMILIES)),
        )
        self.edge_exist_head = nn.Sequential(
            nn.Linear(hidden_dim * 5, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.edge_rel_head = nn.Sequential(
            nn.Linear(hidden_dim * 5, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, len(REL_WITH_NONE)),
        )
        self.verifier_head = nn.Sequential(
            nn.Linear(hidden_dim * 6 + 32, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, len(REL_WITH_NONE)),
        )
        self.edge_pair_feat_dim = edge_pair_feat_dim
        if edge_pair_feat_dim > 0:
            self.edge_pair_proj = nn.Sequential(
                nn.Linear(edge_pair_feat_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
            nn.init.zeros_(self.edge_pair_proj[-1].weight)
            nn.init.zeros_(self.edge_pair_proj[-1].bias)
        else:
            self.edge_pair_proj = None
        self.slot_pos_emb = nn.Embedding(k_max, 16)
        self.mem_kind_head = nn.Sequential(
            nn.Linear(hidden_dim * 3 + 2 + 16, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, len(MEM_LINK_KINDS)),
        )
        self.mem_rel_head = nn.Sequential(
            nn.Linear(hidden_dim * 3 + 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, len(REL_WITH_NONE)),
        )

        # Synthesis template argument heads.
        # mixed_add_link new_note -> destination memory
        # multi_region_attach bridge -> two memory arguments
        self.mixed_dst_mem_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.bridge_mem_a_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.bridge_mem_b_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def encode(self, batch: UnifiedBatch) -> Dict[str, torch.Tensor]:
        signal_h = self.signal_proj(batch.signal_bow)

        kind_h = self.span_kind_emb(batch.cand_kind_ids.clamp(min=0, max=len(SPAN_KIND_VOCAB) - 1))
        cand_in = torch.cat([batch.cand_bow, kind_h, batch.cand_feat], dim=-1)
        cand_h = self.cand_proj(cand_in) * batch.cand_mask[..., None].float()
        if self.cand_emb_proj is not None:
            cand_h = cand_h + self.cand_emb_proj(batch.cand_emb[..., : self.cand_emb_dim]) * batch.cand_mask[..., None].float()

        if self.use_cand_self_attn:
            cand_self_pad_mask = ~batch.cand_mask
            no_cand_self = ~batch.cand_mask.any(dim=1)
            if bool(no_cand_self.any().item()):
                cand_self_pad_mask = cand_self_pad_mask.clone()
                cand_self_pad_mask[no_cand_self, 0] = False
            attn_out, _ = self.cand_self_attn(cand_h, cand_h, cand_h, key_padding_mask=cand_self_pad_mask)
            attn_out = attn_out * batch.cand_mask.any(dim=1, keepdim=True)[..., None].float()
            cand_h = self.cand_self_norm(cand_h + attn_out)
            cand_h = cand_h * batch.cand_mask[..., None].float()

        mem_in = torch.cat([batch.mem_bow, batch.mem_feat], dim=-1)
        mem_h = self.mem_proj(mem_in) * batch.mem_mask[..., None].float()
        if self.mem_emb_proj is not None:
            mem_h = mem_h + self.mem_emb_proj(batch.mem_emb[..., : self.mem_emb_dim]) * batch.mem_mask[..., None].float()

        mem_valid = batch.mem_mask[..., None].float()
        pooled_mem = (mem_h * mem_valid).sum(dim=1) / mem_valid.sum(dim=1).clamp_min(1.0)
        # Intentional difference from PredAlignNet: pooled_mem is not detached.
        # In the unified setting we want gradients from commit/edge/memory losses
        # to co-adapt the shared memory encoder instead of isolating it.
        state_h = self.state_proj(torch.cat([signal_h, pooled_mem], dim=-1))

        slot_query = state_h[:, None, :] + self.slot_emb.weight[None, :, :]
        slot_query = slot_query * batch.slot_mask[..., None].float()

        cand_pad_mask = ~batch.cand_mask
        no_cand = ~batch.cand_mask.any(dim=1)
        if bool(no_cand.any().item()):
            cand_pad_mask = cand_pad_mask.clone()
            cand_pad_mask[no_cand, 0] = False
        attn_out, _ = self.slot_cand_attn(slot_query, cand_h, cand_h, key_padding_mask=cand_pad_mask)
        attn_out = attn_out * batch.cand_mask.any(dim=1, keepdim=True)[..., None].float()
        slot_query = self.slot_norm1(slot_query + attn_out)

        mem_pad_mask = ~batch.mem_mask
        no_mem = ~batch.mem_mask.any(dim=1)
        if bool(no_mem.any().item()):
            mem_pad_mask = mem_pad_mask.clone()
            mem_pad_mask[no_mem, 0] = False
        attn_out, _ = self.slot_mem_attn(slot_query, mem_h, mem_h, key_padding_mask=mem_pad_mask)
        attn_out = attn_out * batch.mem_mask.any(dim=1, keepdim=True)[..., None].float()
        slot_query = self.slot_norm2(slot_query + attn_out)

        slot_pad_mask = ~batch.slot_mask
        attn_out, _ = self.slot_self_attn(slot_query, slot_query, slot_query, key_padding_mask=slot_pad_mask)
        slot_query = self.slot_norm3(slot_query + attn_out)

        return {
            "signal_h": signal_h,
            "cand_h": cand_h,
            "mem_h": mem_h,
            "pooled_mem": pooled_mem,
            "state_h": state_h,
            "slot_query": slot_query,
        }

    def forward(self, batch: UnifiedBatch) -> Dict[str, torch.Tensor]:
        enc = self.encode(batch)
        slot_query = enc["slot_query"]
        cand_h = enc["cand_h"]
        mem_h = enc["mem_h"]
        state_h = enc["state_h"]
        signal_h = enc["signal_h"]

        B, K, H = slot_query.shape
        C = cand_h.size(1)
        M = mem_h.size(1)

        use_logits = self.use_head(slot_query).squeeze(-1)
        type_logits = self.type_head(slot_query).squeeze(-1)

        slot_exp = slot_query[:, :, None, :].expand(-1, -1, C, -1)
        cand_exp = cand_h[:, None, :, :].expand(-1, K, -1, -1)
        span_logits = self.span_head(torch.cat([slot_exp, cand_exp], dim=-1)).squeeze(-1)
        span_logits = span_logits.masked_fill(~batch.cand_mask[:, None, :], LOGIT_MASK_VALUE)

        commit_logits = self.commit_head(torch.cat([signal_h, enc["pooled_mem"]], dim=-1))

        left = slot_query[:, :, None, :].expand(-1, K, K, -1)
        right = slot_query[:, None, :, :].expand(-1, K, K, -1)
        sig_pair = signal_h[:, None, None, :].expand(-1, K, K, -1)
        state_pair = state_h[:, None, None, :].expand(-1, K, K, -1)
        interaction = left * right
        if self.edge_pair_proj is not None and batch.edge_pair_feat is not None:
            pair_feat_h = self.edge_pair_proj(batch.edge_pair_feat)
            interaction = interaction + pair_feat_h
        pair_in = torch.cat([left, right, sig_pair, state_pair, interaction], dim=-1)
        edge_exist_logits = self.edge_exist_head(pair_in).squeeze(-1)
        edge_rel_logits = self.edge_rel_head(pair_in)
        diff = left - right
        pos_pair = torch.cat([
            self.slot_pos_emb(batch.slot_ids)[:, :, None, :].expand(-1, -1, K, -1),
            self.slot_pos_emb(batch.slot_ids)[:, None, :, :].expand(-1, K, -1, -1),
        ], dim=-1)
        verifier_in = torch.cat([left, right, interaction, diff, sig_pair, state_pair, pos_pair], dim=-1)
        verifier_logits = self.verifier_head(verifier_in)

        slot_mem = slot_query[:, :, None, :].expand(-1, K, M, -1)
        mem_exp = mem_h[:, None, :, :].expand(-1, K, -1, -1)
        signal_mem = signal_h[:, None, None, :].expand(-1, K, M, -1)
        slot_pos = self.slot_pos_emb(batch.slot_ids)
        slot_pos_exp = slot_pos[:, :, None, :].expand(-1, -1, M, -1)
        slot_mem_in = torch.cat([slot_mem, mem_exp, signal_mem, batch.mem_pair_feat, slot_pos_exp], dim=-1)
        mem_kind_logits = self.mem_kind_head(slot_mem_in)
        mem_rel_logits = self.mem_rel_head(slot_mem_in[..., :-16])  # mem_rel_head doesn't take the slot_pos

        mixed_dst_mem_logits = self.mixed_dst_mem_head(torch.cat([slot_mem, mem_exp], dim=-1)).squeeze(-1)
        bridge_mem_a_logits = self.bridge_mem_a_head(torch.cat([slot_mem, mem_exp], dim=-1)).squeeze(-1)
        bridge_mem_b_logits = self.bridge_mem_b_head(torch.cat([slot_mem, mem_exp], dim=-1)).squeeze(-1)

        edge_mask = batch.slot_mask[:, :, None] & batch.slot_mask[:, None, :]
        diag = torch.eye(K, device=edge_mask.device, dtype=torch.bool)[None, :, :]
        edge_mask = edge_mask & ~diag
        edge_exist_logits = edge_exist_logits.masked_fill(~edge_mask, LOGIT_MASK_VALUE)
        edge_rel_logits = edge_rel_logits.masked_fill(~edge_mask[..., None], LOGIT_MASK_VALUE)
        verifier_logits = verifier_logits.masked_fill(~edge_mask[..., None], LOGIT_MASK_VALUE)
        mem_kind_logits = mem_kind_logits.masked_fill(~batch.mem_mask[:, None, :, None], LOGIT_MASK_VALUE)
        mem_rel_logits = mem_rel_logits.masked_fill(~batch.mem_mask[:, None, :, None], LOGIT_MASK_VALUE)
        mixed_dst_mem_logits = mixed_dst_mem_logits.masked_fill(~batch.mem_mask[:, None, :], LOGIT_MASK_VALUE)
        bridge_mem_a_logits = bridge_mem_a_logits.masked_fill(~batch.mem_mask[:, None, :], LOGIT_MASK_VALUE)
        bridge_mem_b_logits = bridge_mem_b_logits.masked_fill(~batch.mem_mask[:, None, :], LOGIT_MASK_VALUE)

        return {
            "use_logits": use_logits,
            "type_logits": type_logits,
            "span_logits": span_logits,
            "commit_logits": commit_logits,
            "edge_exist_logits": edge_exist_logits,
            "edge_rel_logits": edge_rel_logits,
            "verifier_logits": verifier_logits,
            "mem_kind_logits": mem_kind_logits,
            "mem_rel_logits": mem_rel_logits,
            "mixed_dst_mem_logits": mixed_dst_mem_logits,
            "bridge_mem_a_logits": bridge_mem_a_logits,
            "bridge_mem_b_logits": bridge_mem_b_logits,
        }
