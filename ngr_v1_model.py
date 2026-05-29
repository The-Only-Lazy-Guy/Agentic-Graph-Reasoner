from __future__ import annotations

"""
ngr_v1_model.py

NGR-v1a policy model.

Main changes:
- Action-specific pair heads for link / attach / cover / add.
- Global state features over full-graph visible memory plus session state.
- Still graph-native/discrete:
    no language-model planner
    no token-level JSON generation
"""

from dataclasses import dataclass
from typing import Any

import hashlib
import re

import torch
import torch.nn as nn
import torch.nn.functional as F


def bow_hash(text: str, dim: int = 512) -> torch.Tensor:
    """
    Stable hashed bag-of-words vector.
    """
    vec = torch.zeros(dim, dtype=torch.float32)
    toks = re.findall(r"[A-Za-z0-9_]+", str(text or "").lower())
    if not toks:
        return vec

    for tok in toks:
        h = int(hashlib.md5(tok.encode("utf-8")).hexdigest(), 16)
        vec[h % dim] += 1.0

    norm = vec.norm()
    if norm > 0:
        vec = vec / norm
    return vec


@dataclass
class V1Batch:
    signal_bow: torch.Tensor

    span_bow: torch.Tensor
    span_mask: torch.Tensor

    memory_bow: torch.Tensor
    memory_scalar: torch.Tensor
    memory_mask: torch.Tensor

    session_bow: torch.Tensor
    session_scalar: torch.Tensor
    session_mask: torch.Tensor

    action_hist: torch.Tensor
    global_scalar: torch.Tensor

    # Labels are kept for compatibility with older scripts.
    y_action: torch.Tensor
    y_span: torch.Tensor
    y_session: torch.Tensor
    y_session_dst: torch.Tensor
    y_memory: torch.Tensor
    y_relation: torch.Tensor
    y_node_type: torch.Tensor
    y_value: torch.Tensor


PHASES = [
    "create",
    "link",
    "attach",
    "add",
    "cover",
    "noop",
    "stop",
]


class NGRV1PolicyNet(nn.Module):
    def __init__(
        self,
        *,
        hash_dim: int = 512,
        hidden_dim: int = 256,
        action_count: int = 8,
        relation_count: int = 9,
        node_type_count: int = 7,
        phase_count: int = 7,
        action_hist_dim: int = 8,
        memory_scalar_dim: int = 4,
        session_scalar_dim: int = 8,
        global_scalar_dim: int = 16,
    ) -> None:
        super().__init__()

        self.hash_dim = hash_dim
        self.hidden_dim = hidden_dim
        self.action_count = action_count
        self.relation_count = relation_count
        self.node_type_count = node_type_count
        self.phase_count = phase_count
        self.global_scalar_dim = global_scalar_dim

        self.text_proj = nn.Sequential(
            nn.Linear(hash_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.memory_scalar_proj = nn.Sequential(
            nn.Linear(memory_scalar_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )

        self.session_scalar_proj = nn.Sequential(
            nn.Linear(session_scalar_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )

        context_dim = hidden_dim * 4 + action_hist_dim + global_scalar_dim
        self.context_proj = nn.Sequential(
            nn.Linear(context_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )

        self.action_head = nn.Linear(hidden_dim, action_count)
        self.phase_head = nn.Linear(hidden_dim, phase_count)
        self.relation_head = nn.Linear(hidden_dim, relation_count)
        self.node_type_head = nn.Linear(hidden_dim, node_type_count)
        self.value_head = nn.Linear(hidden_dim, 1)

        # Pointer heads.
        self.span_head = nn.Linear(hidden_dim, 1)
        self.session_head = nn.Linear(hidden_dim, 1)
        self.session_dst_head = nn.Linear(hidden_dim, 1)
        self.add_session_head = nn.Linear(hidden_dim, 1)
        self.memory_head = nn.Linear(hidden_dim, 1)

        # Pair heads.
        self.link_src_proj = nn.Linear(hidden_dim, hidden_dim)
        self.link_dst_proj = nn.Linear(hidden_dim, hidden_dim)
        self.link_pair_bias = nn.Linear(hidden_dim, 1)

        self.attach_session_proj = nn.Linear(hidden_dim, hidden_dim)
        self.attach_memory_proj = nn.Linear(hidden_dim, hidden_dim)
        self.attach_pair_bias = nn.Linear(hidden_dim, 1)

        self.cover_session_proj = nn.Linear(hidden_dim, hidden_dim)
        self.cover_memory_proj = nn.Linear(hidden_dim, hidden_dim)
        self.cover_pair_bias = nn.Linear(hidden_dim, 1)

    @staticmethod
    def _pool(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        mask_f = mask.float().unsqueeze(-1)
        denom = mask_f.sum(dim=1).clamp_min(1.0)
        return (x * mask_f).sum(dim=1) / denom

    def encode(self, batch: V1Batch) -> dict[str, torch.Tensor]:
        signal = self.text_proj(batch.signal_bow)

        span = self.text_proj(batch.span_bow)
        memory = self.text_proj(batch.memory_bow) + self.memory_scalar_proj(batch.memory_scalar)
        session = self.text_proj(batch.session_bow) + self.session_scalar_proj(batch.session_scalar)

        span_pool = self._pool(span, batch.span_mask)
        memory_pool = self._pool(memory, batch.memory_mask)
        session_pool = self._pool(session, batch.session_mask)

        ctx_in = torch.cat([
            signal,
            span_pool,
            memory_pool,
            session_pool,
            batch.action_hist,
            batch.global_scalar,
        ], dim=-1)
        ctx = self.context_proj(ctx_in)

        # Contextualize nodes by simple residual context injection.
        span_ctx = span + ctx[:, None, :]
        memory_ctx = memory + ctx[:, None, :]
        session_ctx = session + ctx[:, None, :]

        return {
            "ctx": ctx,
            "span": span_ctx,
            "memory": memory_ctx,
            "session": session_ctx,
        }

    def forward(self, batch: V1Batch) -> dict[str, torch.Tensor]:
        enc = self.encode(batch)
        ctx = enc["ctx"]
        span = enc["span"]
        memory = enc["memory"]
        session = enc["session"]

        action_logits = self.action_head(ctx)
        phase_logits = self.phase_head(ctx)
        relation_logits = self.relation_head(ctx)
        node_type_logits = self.node_type_head(ctx)
        value = self.value_head(ctx).squeeze(-1)

        span_logits = self.span_head(span).squeeze(-1).masked_fill(~batch.span_mask.bool(), -1e9)
        memory_logits = self.memory_head(memory).squeeze(-1).masked_fill(~batch.memory_mask.bool(), -1e9)
        session_logits = self.session_head(session).squeeze(-1).masked_fill(~batch.session_mask.bool(), -1e9)
        session_dst_logits = self.session_dst_head(session).squeeze(-1).masked_fill(~batch.session_mask.bool(), -1e9)
        add_session_logits = self.add_session_head(session).squeeze(-1).masked_fill(~batch.session_mask.bool(), -1e9)

        # LINK pair logits: [B, S, S].
        link_src = self.link_src_proj(session)
        link_dst = self.link_dst_proj(session)
        link_pair = torch.matmul(link_src, link_dst.transpose(1, 2)) / (self.hidden_dim ** 0.5)
        link_pair = link_pair + self.link_pair_bias(ctx).view(-1, 1, 1)

        s_mask = batch.session_mask.bool()
        link_mask = s_mask[:, :, None] & s_mask[:, None, :]
        if link_pair.size(1) == link_pair.size(2):
            eye = torch.eye(link_pair.size(1), dtype=torch.bool, device=link_pair.device)[None, :, :]
            link_mask = link_mask & ~eye
        link_pair_logits = link_pair.masked_fill(~link_mask, -1e9)

        # ATTACH pair logits: [B, S, M].
        attach_s = self.attach_session_proj(session)
        attach_m = self.attach_memory_proj(memory)
        attach_pair = torch.matmul(attach_s, attach_m.transpose(1, 2)) / (self.hidden_dim ** 0.5)
        attach_pair = attach_pair + self.attach_pair_bias(ctx).view(-1, 1, 1)
        attach_mask = s_mask[:, :, None] & batch.memory_mask.bool()[:, None, :]
        attach_pair_logits = attach_pair.masked_fill(~attach_mask, -1e9)

        # COVER pair logits: [B, S, M].
        cover_s = self.cover_session_proj(session)
        cover_m = self.cover_memory_proj(memory)
        cover_pair = torch.matmul(cover_s, cover_m.transpose(1, 2)) / (self.hidden_dim ** 0.5)
        cover_pair = cover_pair + self.cover_pair_bias(ctx).view(-1, 1, 1)
        cover_pair_logits = cover_pair.masked_fill(~attach_mask, -1e9)

        return {
            "action_logits": action_logits,
            "phase_logits": phase_logits,
            "span_logits": span_logits,
            "memory_logits": memory_logits,
            "session_logits": session_logits,
            "session_dst_logits": session_dst_logits,
            "add_session_logits": add_session_logits,
            "link_pair_logits": link_pair_logits,
            "attach_pair_logits": attach_pair_logits,
            "cover_pair_logits": cover_pair_logits,
            "relation_logits": relation_logits,
            "node_type_logits": node_type_logits,
            "value": value,
        }
