from __future__ import annotations

"""
graph_policy_model.py

NGR-v0 graph-native policy model.

Coverage/novelty patch:
- Adds global coverage features computed from candidate node signal-overlap.
- This specifically targets the hard no_op vs add_node boundary:
    high candidate coverage  -> likely no_op/update/link
    low candidate coverage   -> likely add_node
- Still not a language model. It predicts structured decisions only.
"""

import math
import re
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from graph_policy_env import EDIT_TYPES, RELATIONS


NODE_TYPE_VOCAB = [
    "claim", "concept", "summary", "hub", "fact", "definition", "example",
    "hypothesis", "bridge", "application", "unknown"
]
NODE_TYPE_TO_ID = {x: i for i, x in enumerate(NODE_TYPE_VOCAB)}


def stable_hash(text: str, mod: int) -> int:
    h = 2166136261
    for ch in str(text):
        h ^= ord(ch)
        h = (h * 16777619) & 0xFFFFFFFF
    return h % mod


def bow_hash(text: str, dim: int) -> torch.Tensor:
    v = torch.zeros(dim, dtype=torch.float32)
    toks = re.findall(r"[A-Za-z0-9_]+", str(text or "").lower())
    for tok in toks:
        if len(tok) <= 2:
            continue
        v[stable_hash(tok, dim)] += 1.0
    norm = v.norm()
    if norm > 0:
        v = v / norm
    return v


@dataclass
class Batch:
    signal_bow: torch.Tensor
    node_bow: torch.Tensor
    node_scalar: torch.Tensor
    node_type_ids: torch.Tensor
    node_mask: torch.Tensor
    edge_index: torch.Tensor
    edge_rel: torch.Tensor
    edge_mask: torch.Tensor
    y_edit: torch.Tensor
    y_target: torch.Tensor
    y_src: torch.Tensor
    y_dst: torch.Tensor
    y_rel: torch.Tensor


class GraphPolicyNet(nn.Module):
    def __init__(
        self,
        *,
        hash_dim: int = 512,
        node_scalar_dim: int = 4,
        node_type_count: int = len(NODE_TYPE_VOCAB),
        hidden_dim: int = 256,
        gnn_layers: int = 2,
    ) -> None:
        super().__init__()
        self.hash_dim = hash_dim
        self.hidden_dim = hidden_dim

        self.type_emb = nn.Embedding(node_type_count, 24)
        self.rel_emb = nn.Embedding(len(RELATIONS), hidden_dim)

        self.node_in = nn.Sequential(
            nn.Linear(hash_dim * 2 + node_scalar_dim + 24, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.signal_proj = nn.Sequential(
            nn.Linear(hash_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.gnn_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
            for _ in range(gnn_layers)
        ])

        # Coverage features from signal-overlap over candidate nodes:
        # max, mean, top3_mean, top1-top2 gap, count(overlap>.50), count(overlap>.75)
        self.coverage_proj = nn.Sequential(
            nn.Linear(6, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        state_dim = hidden_dim * 3

        self.edit_head = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, len(EDIT_TYPES)),
        )

        self.target_query = nn.Linear(state_dim, hidden_dim)
        self.src_query = nn.Linear(state_dim, hidden_dim)
        self.dst_query = nn.Linear(state_dim + hidden_dim, hidden_dim)

        self.relation_head = nn.Sequential(
            nn.Linear(state_dim + hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, len(RELATIONS)),
        )

        self.value_head = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def coverage_features(self, batch: Batch) -> torch.Tensor:
        # node_scalar[:, :, 2] is signal_overlap.
        overlap = batch.node_scalar[:, :, 2]
        mask = batch.node_mask
        B, N = overlap.shape

        masked = overlap.masked_fill(~mask, -1e9)
        max_ov = masked.max(dim=-1).values.clamp_min(0.0)

        counts = mask.float().sum(dim=-1).clamp_min(1.0)
        mean_ov = (overlap * mask.float()).sum(dim=-1) / counts

        k = min(3, N)
        topk = masked.topk(k, dim=-1).values.clamp_min(0.0)
        top3_mean = topk.mean(dim=-1)

        if N >= 2:
            top2 = masked.topk(2, dim=-1).values.clamp_min(0.0)
            gap = top2[:, 0] - top2[:, 1]
        else:
            gap = max_ov

        count_50 = ((overlap > 0.50) & mask).float().sum(dim=-1) / counts
        count_75 = ((overlap > 0.75) & mask).float().sum(dim=-1) / counts

        feats = torch.stack([max_ov, mean_ov, top3_mean, gap, count_50, count_75], dim=-1)
        return feats

    def encode(self, batch: Batch) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        signal_h = self.signal_proj(batch.signal_bow)

        sig_exp = batch.signal_bow[:, None, :].expand(-1, batch.node_bow.size(1), -1)
        type_h = self.type_emb(batch.node_type_ids.clamp(min=0, max=len(NODE_TYPE_VOCAB) - 1))
        x = torch.cat([batch.node_bow, sig_exp, batch.node_scalar, type_h], dim=-1)
        h = self.node_in(x)
        h = h * batch.node_mask[:, :, None].float()

        for layer in self.gnn_layers:
            msg = torch.zeros_like(h)
            B, E = batch.edge_rel.shape
            for b in range(B):
                valid_edges = torch.nonzero(batch.edge_mask[b], as_tuple=False).flatten()
                if valid_edges.numel() == 0:
                    continue
                src = batch.edge_index[b, valid_edges, 0].long()
                dst = batch.edge_index[b, valid_edges, 1].long()
                rel = batch.edge_rel[b, valid_edges].long().clamp(min=0, max=len(RELATIONS) - 1)

                m = h[b, src] + self.rel_emb(rel)
                msg[b].index_add_(0, dst, m)

                # Also pass reverse messages so local undirected evidence is visible.
                mr = h[b, dst] + self.rel_emb(rel)
                msg[b].index_add_(0, src, mr)

            h = h + layer(torch.cat([h, msg], dim=-1))
            h = h * batch.node_mask[:, :, None].float()

        att = torch.einsum("bnh,bh->bn", h, signal_h) / math.sqrt(h.size(-1))
        att = att.masked_fill(~batch.node_mask, -1e9)
        weights = torch.softmax(att, dim=-1)
        graph_h = torch.einsum("bn,bnh->bh", weights, h)

        coverage_h = self.coverage_proj(self.coverage_features(batch))
        state = torch.cat([graph_h, signal_h, coverage_h], dim=-1)

        return h, state, signal_h

    def pointer_logits(self, query: torch.Tensor, node_h: torch.Tensor, node_mask: torch.Tensor) -> torch.Tensor:
        logits = torch.einsum("bh,bnh->bn", query, node_h) / math.sqrt(node_h.size(-1))
        return logits.masked_fill(~node_mask, -1e9)

    def forward(self, batch: Batch) -> Dict[str, torch.Tensor]:
        node_h, state, _signal_h = self.encode(batch)

        edit_logits = self.edit_head(state)

        target_q = self.target_query(state)
        src_q = self.src_query(state)

        target_logits = self.pointer_logits(target_q, node_h, batch.node_mask)
        src_logits = self.pointer_logits(src_q, node_h, batch.node_mask)

        src_prob = torch.softmax(src_logits, dim=-1)
        src_ctx = torch.einsum("bn,bnh->bh", src_prob, node_h)

        dst_q = self.dst_query(torch.cat([state, src_ctx], dim=-1))
        dst_logits = self.pointer_logits(dst_q, node_h, batch.node_mask)

        dst_prob = torch.softmax(dst_logits, dim=-1)
        dst_ctx = torch.einsum("bn,bnh->bh", dst_prob, node_h)

        relation_logits = self.relation_head(torch.cat([state, src_ctx, dst_ctx], dim=-1))
        value = self.value_head(state).squeeze(-1)

        return {
            "edit_logits": edit_logits,
            "target_logits": target_logits,
            "src_logits": src_logits,
            "dst_logits": dst_logits,
            "relation_logits": relation_logits,
            "value": value,
        }
