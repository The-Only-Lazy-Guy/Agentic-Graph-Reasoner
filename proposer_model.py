from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping

import torch
import torch.nn as nn
import torch.nn.functional as F

from pred_model import LOGIT_MASK_VALUE, SPAN_KIND_TO_ID, SPAN_KIND_VOCAB, _maybe_emb_proj


PROPOSER_NODE_TYPE_VOCAB = ["concept", "bridge"]
PROPOSER_NODE_TYPE_TO_ID = {x: i for i, x in enumerate(PROPOSER_NODE_TYPE_VOCAB)}


def infer_proposer_arch_from_state(state: Mapping[str, torch.Tensor], hidden_dim: int) -> Dict[str, Any]:
    slot_attention_mode = "detr" if "slot_cand_attn.in_proj_weight" in state else "none"
    span_head_weight = state.get("span_head.0.weight")
    if span_head_weight is None:
        return {
            "slot_attention_mode": slot_attention_mode,
            "span_scorer_mode": "dot",
            "cand_pair_feat_dim": 3,
            "use_ar_span_features": False,
        }
    in_dim = int(span_head_weight.shape[1])
    if in_dim == hidden_dim * 2:
        return {
            "slot_attention_mode": slot_attention_mode,
            "span_scorer_mode": "concat_mlp",
            "cand_pair_feat_dim": 0,
            "use_ar_span_features": False,
        }
    if in_dim == hidden_dim * 3 + 5:
        return {
            "slot_attention_mode": slot_attention_mode,
            "span_scorer_mode": "interaction_mlp",
            "cand_pair_feat_dim": 3,
            "use_ar_span_features": True,
        }
    if in_dim == hidden_dim * 3 + 3:
        return {
            "slot_attention_mode": slot_attention_mode,
            "span_scorer_mode": "interaction_mlp",
            "cand_pair_feat_dim": 3,
            "use_ar_span_features": False,
        }
    pair_dim = max(in_dim - hidden_dim * 3, 0)
    return {
        "slot_attention_mode": slot_attention_mode,
        "span_scorer_mode": "interaction_mlp",
        "cand_pair_feat_dim": pair_dim,
        "use_ar_span_features": False,
    }


@dataclass
class ProposerBatch:
    """Fixed-slot proposer minibatch.

    Tensor shapes:
      signal_bow: [B, H_hash]
      cand_bow/cand_emb/cand_kind_ids/cand_feat/cand_pair_feat/cand_mask: [B, C, ...]
      mem_bow/mem_emb/mem_feat/mem_mask: [B, M, ...]
      y_use/y_span/y_is_bridge/y_slot_mask: [B, K]
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
    y_use: torch.Tensor
    y_span: torch.Tensor
    y_is_bridge: torch.Tensor
    y_slot_mask: torch.Tensor


class ProposerNet(nn.Module):
    """Fixed-slot session-node proposer.

    The model predicts, for each slot:
      - whether the slot is used
      - which candidate span anchors the proposed node
      - whether the node is a bridge
    """

    def __init__(
        self,
        *,
        hash_dim: int = 512,
        hidden_dim: int = 256,
        k_max: int = 3,
        cand_feat_dim: int = 2,
        cand_pair_feat_dim: int = 3,
        spec_emb_dim: int = 0,
        cand_emb_dim: int = 0,
        mem_emb_dim: int = 0,
        use_ar_span_features: bool = True,
        slot_attention_mode: str = "none",
        span_scorer_mode: str = "concat_mlp",
        attention_heads: int = 4,
        attention_dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.hash_dim = hash_dim
        self.hidden_dim = hidden_dim
        self.k_max = k_max
        self.cand_emb_dim = cand_emb_dim
        self.mem_emb_dim = mem_emb_dim
        self.cand_pair_feat_dim = cand_pair_feat_dim
        self.use_ar_span_features = use_ar_span_features
        self.slot_attention_mode = slot_attention_mode
        self.span_scorer_mode = span_scorer_mode
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
            nn.Linear(hash_dim + 3, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.mem_emb_proj = _maybe_emb_proj(mem_emb_dim, hidden_dim)
        self.state_proj = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        if slot_attention_mode == "detr":
            self.slot_cand_attn = nn.MultiheadAttention(
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
            self.slot_norm1 = nn.LayerNorm(hidden_dim)
            self.slot_norm2 = nn.LayerNorm(hidden_dim)
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
        if span_scorer_mode == "concat_mlp":
            self.span_head = nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1),
            )
        elif span_scorer_mode == "interaction_mlp":
            total_pair_feat_dim = cand_pair_feat_dim + (2 if use_ar_span_features else 0)
            self.span_head = nn.Sequential(
                nn.Linear(hidden_dim * 3 + total_pair_feat_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1),
            )
        elif span_scorer_mode == "dot":
            self.span_head = None
        else:
            raise ValueError(f"Unknown span_scorer_mode: {span_scorer_mode}")

    def encode(self, batch: ProposerBatch) -> Dict[str, torch.Tensor]:
        signal_h = self.signal_proj(batch.signal_bow)

        kind_h = self.span_kind_emb(batch.cand_kind_ids.clamp(min=0, max=len(SPAN_KIND_VOCAB) - 1))
        cand_in = torch.cat([batch.cand_bow, kind_h, batch.cand_feat], dim=-1)
        cand_h = self.cand_proj(cand_in) * batch.cand_mask[..., None].float()
        if self.cand_emb_proj is not None:
            cand_h = cand_h + self.cand_emb_proj(batch.cand_emb[..., : self.cand_emb_dim]) * batch.cand_mask[..., None].float()

        mem_in = torch.cat([batch.mem_bow, batch.mem_feat], dim=-1)
        mem_h = self.mem_proj(mem_in) * batch.mem_mask[..., None].float()
        if self.mem_emb_proj is not None:
            mem_h = mem_h + self.mem_emb_proj(batch.mem_emb[..., : self.mem_emb_dim]) * batch.mem_mask[..., None].float()

        mem_valid = batch.mem_mask[..., None].float()
        pooled_mem = (mem_h * mem_valid).sum(dim=1) / mem_valid.sum(dim=1).clamp_min(1.0)
        state_h = self.state_proj(torch.cat([signal_h, pooled_mem.detach()], dim=-1))
        slot_query = state_h[:, None, :] + self.slot_emb.weight[None, :, :]
        slot_query = slot_query * batch.y_slot_mask[..., None].float()
        if self.slot_attention_mode == "detr":
            cand_pad_mask = ~batch.cand_mask
            attn_out, _ = self.slot_cand_attn(slot_query, cand_h, cand_h, key_padding_mask=cand_pad_mask)
            slot_query = self.slot_norm1(slot_query + attn_out)
            slot_pad_mask = ~batch.y_slot_mask
            attn_out, _ = self.slot_self_attn(slot_query, slot_query, slot_query, key_padding_mask=slot_pad_mask)
            slot_query = self.slot_norm2(slot_query + attn_out)
        return {
            "signal_h": signal_h,
            "cand_h": cand_h,
            "mem_h": mem_h,
            "slot_query": slot_query,
        }

    def _build_dynamic_pair_feat(
        self,
        batch: ProposerBatch,
        *,
        use_mask: torch.Tensor,
        span_idx: torch.Tensor,
    ) -> torch.Tensor:
        """
        Build slot-conditioned pair features.

        Features per (slot, candidate):
          - signal/span overlap
          - span start normalized
          - span length ratio
          - previous slot picked this candidate
          - any earlier slot picked this candidate
        """
        B, K = use_mask.shape
        C = batch.cand_mask.size(1)
        base = batch.cand_pair_feat[:, None, :, : self.cand_pair_feat_dim].expand(-1, K, -1, -1)
        if not self.use_ar_span_features:
            return base
        prev_same = torch.zeros((B, K, C), dtype=torch.float32, device=batch.signal_bow.device)
        any_prev_same = torch.zeros((B, K, C), dtype=torch.float32, device=batch.signal_bow.device)
        for k in range(1, K):
            prev_valid = use_mask[:, k - 1] & (span_idx[:, k - 1] >= 0) & (span_idx[:, k - 1] < C)
            if prev_valid.any():
                prev_same_k = torch.zeros((B, C), dtype=torch.float32, device=batch.signal_bow.device)
                prev_same_k[prev_valid] = F.one_hot(span_idx[prev_valid, k - 1], num_classes=C).float()
                prev_same[:, k] = prev_same_k
            for j in range(k):
                prev_any_valid = use_mask[:, j] & (span_idx[:, j] >= 0) & (span_idx[:, j] < C)
                if prev_any_valid.any():
                    any_prev_same[prev_any_valid, k] = torch.maximum(
                        any_prev_same[prev_any_valid, k],
                        F.one_hot(span_idx[prev_any_valid, j], num_classes=C).float(),
                    )
        return torch.cat([base, prev_same[..., None], any_prev_same[..., None]], dim=-1)

    def _compute_span_logits(
        self,
        batch: ProposerBatch,
        *,
        slot_query: torch.Tensor,
        cand_h: torch.Tensor,
        pair_feat: torch.Tensor,
    ) -> torch.Tensor:
        B, K, H = slot_query.shape
        C = cand_h.size(1)
        if self.span_scorer_mode == "dot":
            logits = torch.bmm(slot_query, cand_h.transpose(1, 2)) / (H ** 0.5)
            return logits.masked_fill(~batch.cand_mask[:, None, :], LOGIT_MASK_VALUE)
        slot_exp = slot_query[:, :, None, :].expand(-1, -1, C, -1)
        cand_exp = cand_h[:, None, :, :].expand(-1, K, -1, -1)
        if self.span_scorer_mode == "concat_mlp":
            scorer_in = torch.cat([slot_exp, cand_exp], dim=-1)
        else:
            interaction = slot_exp * cand_exp
            scorer_in = torch.cat([slot_exp, cand_exp, interaction, pair_feat], dim=-1)
        span_logits = self.span_head(scorer_in).squeeze(-1)
        return span_logits.masked_fill(~batch.cand_mask[:, None, :], LOGIT_MASK_VALUE)

    def forward(self, batch: ProposerBatch) -> Dict[str, torch.Tensor]:
        enc = self.encode(batch)
        cand_h = enc["cand_h"]
        slot_query = enc["slot_query"]
        if self.span_scorer_mode == "interaction_mlp":
            teacher_use = batch.y_use > 0.5
            teacher_span = torch.where(batch.y_span >= 0, batch.y_span, torch.zeros_like(batch.y_span))
            pair_feat = self._build_dynamic_pair_feat(batch, use_mask=teacher_use, span_idx=teacher_span)
        else:
            pair_feat = batch.cand_pair_feat[:, None, :, : self.cand_pair_feat_dim].expand(-1, self.k_max, -1, -1)
        span_logits = self._compute_span_logits(batch, slot_query=slot_query, cand_h=cand_h, pair_feat=pair_feat)

        use_logits = self.use_head(slot_query).squeeze(-1)
        type_logits = self.type_head(slot_query).squeeze(-1)
        return {
            "use_logits": use_logits,
            "span_logits": span_logits,
            "type_logits": type_logits,
        }

    def predict(self, batch: ProposerBatch, *, use_threshold: float = 0.5) -> Dict[str, torch.Tensor]:
        if self.span_scorer_mode != "interaction_mlp" or not self.use_ar_span_features:
            out = self.forward(batch)
            use_pred = torch.sigmoid(out["use_logits"]) >= use_threshold
            bridge_pred = (torch.sigmoid(out["type_logits"]) >= 0.5) & use_pred
            span_pred = out["span_logits"].argmax(dim=-1)
            valid_choice = use_pred & batch.cand_mask.any(dim=-1, keepdim=False)[:, None].expand_as(span_pred)
            span_pred = torch.where(valid_choice, span_pred, torch.full_like(span_pred, -1))
            return {
                "use_logits": out["use_logits"],
                "type_logits": out["type_logits"],
                "span_logits": out["span_logits"],
                "use_pred": use_pred,
                "bridge_pred": bridge_pred,
                "span_pred": span_pred,
            }

        enc = self.encode(batch)
        cand_h = enc["cand_h"]
        slot_query = enc["slot_query"]
        use_logits = self.use_head(slot_query).squeeze(-1)
        type_logits = self.type_head(slot_query).squeeze(-1)
        use_pred = torch.sigmoid(use_logits) >= use_threshold
        bridge_pred = (torch.sigmoid(type_logits) >= 0.5) & use_pred

        B, K = use_pred.shape
        C = batch.cand_mask.size(1)
        span_logits = torch.full((B, K, C), LOGIT_MASK_VALUE, dtype=torch.float32, device=batch.signal_bow.device)
        span_pred = torch.full((B, K), -1, dtype=torch.long, device=batch.signal_bow.device)
        for k in range(K):
            use_mask_prefix = use_pred.clone()
            use_mask_prefix[:, k:] = False
            span_idx_prefix = torch.where(span_pred >= 0, span_pred, torch.zeros_like(span_pred))
            pair_feat = self._build_dynamic_pair_feat(batch, use_mask=use_mask_prefix, span_idx=span_idx_prefix)
            slot_logits = self._compute_span_logits(
                batch,
                slot_query=slot_query[:, k : k + 1, :],
                cand_h=cand_h,
                pair_feat=pair_feat[:, k : k + 1, :, :],
            ).squeeze(1)
            chosen = slot_logits.argmax(dim=-1)
            valid_choice = use_pred[:, k] & batch.cand_mask.gather(1, chosen[:, None]).squeeze(1)
            span_pred[valid_choice, k] = chosen[valid_choice]
            span_logits[:, k, :] = slot_logits

        return {
            "use_logits": use_logits,
            "type_logits": type_logits,
            "span_logits": span_logits,
            "use_pred": use_pred,
            "bridge_pred": bridge_pred,
            "span_pred": span_pred,
        }
