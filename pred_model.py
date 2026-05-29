from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Mapping

import torch
import torch.nn as nn

from graph_core import CANONICAL_RELATIONS


SPAN_KIND_VOCAB = ["unknown", "full", "clause", "item", "merged", "synth"]
SPAN_KIND_TO_ID = {x: i for i, x in enumerate(SPAN_KIND_VOCAB)}

SPEC_TYPE_VOCAB = ["concept", "bridge", "claim", "summary", "fact", "unknown"]
SPEC_TYPE_TO_ID = {x: i for i, x in enumerate(SPEC_TYPE_VOCAB)}

COMMIT_FAMILIES = ["no_op", "add_node", "other"]
COMMIT_TO_ID = {x: i for i, x in enumerate(COMMIT_FAMILIES)}

REL_WITH_NONE = ["none"] + list(CANONICAL_RELATIONS)
REL_WITH_NONE_TO_ID = {x: i for i, x in enumerate(REL_WITH_NONE)}
MEM_LINK_KINDS = ["none", "attach", "cover"]
MEM_LINK_KIND_TO_ID = {x: i for i, x in enumerate(MEM_LINK_KINDS)}

LOGIT_MASK_VALUE = -1e9
EDGE_EXIST_THRESHOLD = 0.5


def _maybe_emb_proj(in_dim: int, hidden_dim: int) -> nn.Linear | None:
    if in_dim <= 0:
        return None
    layer = nn.Linear(in_dim, hidden_dim)
    nn.init.zeros_(layer.weight)
    nn.init.zeros_(layer.bias)
    return layer


def infer_edge_rel_pair_feat_dim_from_state(state: Mapping[str, torch.Tensor], hidden_dim: int) -> int:
    weight = state.get("edge_rel_head.0.weight")
    if weight is None:
        return 0
    return max(int(weight.shape[1]) - hidden_dim * 5, 0)


def infer_spec_emb_dim_from_state(state: Mapping[str, torch.Tensor]) -> int:
    weight = state.get("spec_emb_proj.weight")
    if weight is None:
        return 0
    return int(weight.shape[1])


def infer_cand_emb_dim_from_state(state: Mapping[str, torch.Tensor]) -> int:
    weight = state.get("cand_emb_proj.weight")
    if weight is None:
        return 0
    return int(weight.shape[1])


def infer_mem_emb_dim_from_state(state: Mapping[str, torch.Tensor]) -> int:
    weight = state.get("mem_emb_proj.weight")
    if weight is None:
        return 0
    return int(weight.shape[1])


@dataclass
class PredBatch:
    """Goal-conditioned aligner minibatch.

    Tensor shapes:
      signal_bow: [B, H_hash]
      spec_bow/spec_emb/spec_type_ids/spec_mask: [B, S, ...]
      cand_bow/cand_emb/cand_kind_ids/cand_feat: [B, C, ...]
      span_pair_feat/cand_mask: [B, S, C, ...] / [B, S, C]
      mem_bow/mem_emb/mem_feat/mem_mask: [B, M, ...]
      mem_pair_feat: [B, S, M, F_mem]
      edge_rel_pair_feat/edge_mask: [B, S, S, ...] / [B, S, S]
      supervision tensors mirror their corresponding prediction heads.
    """
    signal_bow: torch.Tensor
    spec_bow: torch.Tensor
    spec_emb: torch.Tensor
    spec_type_ids: torch.Tensor
    spec_mask: torch.Tensor
    cand_bow: torch.Tensor
    cand_emb: torch.Tensor
    cand_kind_ids: torch.Tensor
    cand_feat: torch.Tensor
    span_pair_feat: torch.Tensor
    cand_mask: torch.Tensor
    mem_bow: torch.Tensor
    mem_emb: torch.Tensor
    mem_feat: torch.Tensor
    mem_mask: torch.Tensor
    mem_pair_feat: torch.Tensor
    edge_rel_pair_feat: torch.Tensor
    edge_mask: torch.Tensor
    y_span: torch.Tensor
    y_commit: torch.Tensor
    y_edge_exist: torch.Tensor
    y_edge_rel: torch.Tensor
    y_mem_kind: torch.Tensor
    y_mem_rel: torch.Tensor


class PredAlignNet(nn.Module):
    """Goal-conditioned aligner for session specs, edges, and memory links.

    The model assumes session-node specs are already given. It aligns those
    specs to spans, predicts the commit family, predicts directed session edges
    plus relation labels, and predicts session-to-memory link kind/relation.
    """
    def __init__(
        self,
        *,
        hash_dim: int = 512,
        hidden_dim: int = 256,
        cand_feat_dim: int = 2,
        span_pair_feat_dim: int = 1,
        mem_pair_feat_dim: int = 2,
        edge_rel_pair_feat_dim: int = 0,
        spec_emb_dim: int = 0,
        cand_emb_dim: int = 0,
        mem_emb_dim: int = 0,
    ) -> None:
        super().__init__()
        self.hash_dim = hash_dim
        self.hidden_dim = hidden_dim
        self.edge_rel_pair_feat_dim = edge_rel_pair_feat_dim
        self.spec_emb_dim = spec_emb_dim
        self.cand_emb_dim = cand_emb_dim
        self.mem_emb_dim = mem_emb_dim

        self.spec_type_emb = nn.Embedding(len(SPEC_TYPE_VOCAB), 24)
        self.span_kind_emb = nn.Embedding(len(SPAN_KIND_VOCAB), 16)

        self.signal_proj = nn.Sequential(
            nn.Linear(hash_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.spec_proj = nn.Sequential(
            nn.Linear(hash_dim * 2 + 24, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.spec_emb_proj = _maybe_emb_proj(spec_emb_dim, hidden_dim)
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
        self.span_scorer = nn.Sequential(
            nn.Linear(hidden_dim * 2 + span_pair_feat_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.none_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
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
            nn.Linear(hidden_dim * 5 + edge_rel_pair_feat_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, len(REL_WITH_NONE)),
        )
        self.mem_kind_head = nn.Sequential(
            nn.Linear(hidden_dim * 3 + mem_pair_feat_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, len(MEM_LINK_KINDS)),
        )
        self.mem_rel_head = nn.Sequential(
            nn.Linear(hidden_dim * 3 + mem_pair_feat_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, len(REL_WITH_NONE)),
        )

    def encode(self, batch: PredBatch) -> Dict[str, torch.Tensor]:
        signal_h = self.signal_proj(batch.signal_bow)

        sig_exp = batch.signal_bow[:, None, :].expand(-1, batch.spec_bow.size(1), -1)
        spec_type_h = self.spec_type_emb(batch.spec_type_ids.clamp(min=0, max=len(SPEC_TYPE_VOCAB) - 1))
        spec_in = torch.cat([batch.spec_bow, sig_exp, spec_type_h], dim=-1)
        spec_h = self.spec_proj(spec_in)
        spec_h_for_edges = spec_h
        if self.spec_emb_proj is not None:
            spec_h_for_edges = spec_h_for_edges + self.spec_emb_proj(batch.spec_emb[..., :self.spec_emb_dim])
        spec_h = spec_h * batch.spec_mask[..., None].float()
        spec_h_for_edges = spec_h_for_edges * batch.spec_mask[..., None].float()

        kind_h = self.span_kind_emb(batch.cand_kind_ids.clamp(min=0, max=len(SPAN_KIND_VOCAB) - 1))
        cand_in = torch.cat([batch.cand_bow, kind_h, batch.cand_feat], dim=-1)
        cand_valid = batch.cand_mask.any(dim=1)
        cand_h = self.cand_proj(cand_in) * cand_valid[..., None].float()
        cand_h_for_span = cand_h
        if self.cand_emb_proj is not None:
            cand_h_for_span = cand_h_for_span + self.cand_emb_proj(batch.cand_emb[..., :self.cand_emb_dim])
            cand_h_for_span = cand_h_for_span * cand_valid[..., None].float()

        mem_in = torch.cat([batch.mem_bow, batch.mem_feat], dim=-1)
        mem_h = self.mem_proj(mem_in) * batch.mem_mask[..., None].float()
        mem_h_for_rel = mem_h
        if self.mem_emb_proj is not None:
            mem_h_for_rel = mem_h_for_rel + self.mem_emb_proj(batch.mem_emb[..., :self.mem_emb_dim])
            mem_h_for_rel = mem_h_for_rel * batch.mem_mask[..., None].float()

        mem_valid = batch.mem_mask[..., None].float()
        pooled_mem = (mem_h * mem_valid).sum(dim=1) / mem_valid.sum(dim=1).clamp_min(1.0)
        state_h = torch.cat([signal_h, pooled_mem.detach()], dim=-1)
        return {
            "signal_h": signal_h,
            "spec_h": spec_h,
            "spec_h_for_edges": spec_h_for_edges,
            "cand_h": cand_h,
            "cand_h_for_span": cand_h_for_span,
            "mem_h": mem_h,
            "mem_h_for_rel": mem_h_for_rel,
            "state_h": state_h,
        }

    def forward(self, batch: PredBatch) -> Dict[str, torch.Tensor]:
        enc = self.encode(batch)
        signal_h = enc["signal_h"]
        spec_h = enc["spec_h"]
        spec_h_for_edges = enc["spec_h_for_edges"]
        cand_h = enc["cand_h"]
        cand_h_for_span = enc["cand_h_for_span"]
        mem_h = enc["mem_h"]
        mem_h_for_rel = enc["mem_h_for_rel"]
        state_h = enc["state_h"]

        B, C, H = cand_h.shape
        S = spec_h.size(1)
        spec_exp = spec_h[:, :, None, :].expand(-1, -1, C, -1)
        cand_exp = cand_h_for_span[:, None, :, :].expand(-1, S, -1, -1)
        span_logits = self.span_scorer(torch.cat([spec_exp, cand_exp, batch.span_pair_feat], dim=-1)).squeeze(-1)
        span_logits = span_logits.masked_fill(~batch.cand_mask, LOGIT_MASK_VALUE)
        none_logits = self.none_head(spec_h).squeeze(-1)
        span_logits = torch.cat([span_logits, none_logits[:, :, None]], dim=-1)

        commit_logits = self.commit_head(state_h)

        sig_pair = signal_h[:, None, None, :].expand(-1, S, S, -1)
        left = spec_h_for_edges[:, :, None, :].expand(-1, S, S, -1)
        right = spec_h_for_edges[:, None, :, :].expand(-1, S, S, -1)
        state_exp = state_h[:, None, None, :].expand(-1, S, S, -1)
        edge_in = torch.cat([left, right, sig_pair, state_exp], dim=-1)
        edge_exist_logits = self.edge_exist_head(edge_in).squeeze(-1)
        if self.edge_rel_pair_feat_dim > 0:
            edge_rel_in = torch.cat([edge_in, batch.edge_rel_pair_feat[..., :self.edge_rel_pair_feat_dim]], dim=-1)
        else:
            edge_rel_in = edge_in
        edge_rel_logits = self.edge_rel_head(edge_rel_in)
        edge_exist_logits = edge_exist_logits.masked_fill(~batch.edge_mask, LOGIT_MASK_VALUE)
        edge_rel_logits = edge_rel_logits.masked_fill(~batch.edge_mask[..., None], LOGIT_MASK_VALUE)

        M = mem_h.size(1)
        spec_mem_left = spec_h[:, :, None, :].expand(-1, S, M, -1)
        spec_mem_right = mem_h[:, None, :, :].expand(-1, S, -1, -1)
        spec_mem_right_for_rel = mem_h_for_rel[:, None, :, :].expand(-1, S, -1, -1)
        sig_mem = signal_h[:, None, None, :].expand(-1, S, M, -1)
        mem_in = torch.cat([spec_mem_left, spec_mem_right, sig_mem, batch.mem_pair_feat], dim=-1)
        mem_rel_in = torch.cat([spec_mem_left, spec_mem_right_for_rel, sig_mem, batch.mem_pair_feat], dim=-1)
        mem_kind_logits = self.mem_kind_head(mem_in)
        mem_rel_logits = self.mem_rel_head(mem_rel_in)
        mem_kind_logits = mem_kind_logits.masked_fill(~batch.mem_mask[:, None, :, None], LOGIT_MASK_VALUE)
        mem_rel_logits = mem_rel_logits.masked_fill(~batch.mem_mask[:, None, :, None], LOGIT_MASK_VALUE)

        return {
            "span_logits": span_logits,
            "commit_logits": commit_logits,
            "edge_exist_logits": edge_exist_logits,
            "edge_rel_logits": edge_rel_logits,
            "mem_kind_logits": mem_kind_logits,
            "mem_rel_logits": mem_rel_logits,
        }
