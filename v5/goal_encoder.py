"""TaskFrame -> goal vector encoder.

Takes the micro_controller TaskFrame fields and produces a dense goal vector
used to condition the cross-attention Q projection at Layer 8 and Layer 20.

Output: [B, GOAL_DIM] float32 tensor  (GOAL_DIM = 128)

Design:
  - task_family: learned embedding (num_families x 32)
  - question_mode: learned embedding (num_modes x 16)
  - required_slots: bag-of-slots embedding (num_slots x 16) -> mean-pool
  - MLP: Linear(64, 128) -> GELU -> Linear(128, 128) -> LayerNorm
"""
from __future__ import annotations

from typing import Dict, List, Optional

import torch
import torch.nn as nn
from torch import Tensor

GOAL_DIM = 128

# Stable vocab for task_family — append only
TASK_FAMILY_VOCAB = [
    "algorithm_applicability",
    "direct_judgment",
    "procedure_execution",
    "system_design",
    "mathematical_derivation",
    "causal_explanation",
    "logical_deduction",
    "comparison",
    "definition",
    "unknown",
]
TASK_FAMILY_ID: Dict[str, int] = {t: i for i, t in enumerate(TASK_FAMILY_VOCAB)}
NUM_TASK_FAMILIES = len(TASK_FAMILY_VOCAB)

# Stable vocab for question_mode — append only
QUESTION_MODE_VOCAB = [
    "explain", "verify", "compare", "design", "compute", "prove",
    "enumerate", "classify", "debug", "unknown",
]
QUESTION_MODE_ID: Dict[str, int] = {m: i for i, m in enumerate(QUESTION_MODE_VOCAB)}
NUM_QUESTION_MODES = len(QUESTION_MODE_VOCAB)

# Stable vocab for slot names — append only
SLOT_VOCAB = [
    "verdict", "reason", "alternative", "caveat", "proof",
    "complexity", "example", "definition", "condition", "unknown",
]
SLOT_ID: Dict[str, int] = {s: i for i, s in enumerate(SLOT_VOCAB)}
NUM_SLOTS = len(SLOT_VOCAB)

FAMILY_EMBED_DIM = 32
MODE_EMBED_DIM = 16
SLOT_EMBED_DIM = 16
COMBINED_DIM = FAMILY_EMBED_DIM + MODE_EMBED_DIM + SLOT_EMBED_DIM  # 64


def _task_family_id(family: str) -> int:
    return TASK_FAMILY_ID.get(family, TASK_FAMILY_ID["unknown"])


def _question_mode_id(mode: str) -> int:
    return QUESTION_MODE_ID.get(mode, QUESTION_MODE_ID["unknown"])


def _slot_id(slot: str) -> int:
    return SLOT_ID.get(slot, SLOT_ID["unknown"])


class GoalEncoder(nn.Module):
    """Encode a TaskFrame into a fixed-dim goal vector.

    Batched: accepts [B] inputs where each element is one TaskFrame.
    """

    def __init__(self, goal_dim: int = GOAL_DIM):
        super().__init__()
        self.family_embed = nn.Embedding(NUM_TASK_FAMILIES, FAMILY_EMBED_DIM)
        self.mode_embed = nn.Embedding(NUM_QUESTION_MODES, MODE_EMBED_DIM)
        self.slot_embed = nn.Embedding(NUM_SLOTS, SLOT_EMBED_DIM)

        self.mlp = nn.Sequential(
            nn.Linear(COMBINED_DIM, goal_dim),
            nn.GELU(),
            nn.Linear(goal_dim, goal_dim),
            nn.LayerNorm(goal_dim),
        )

    def forward(
        self,
        family_ids: Tensor,     # [B] long
        mode_ids: Tensor,       # [B] long
        slot_ids: Tensor,       # [B, max_slots] long  (0-padded)
        slot_mask: Tensor,      # [B, max_slots] bool  (True = real slot)
    ) -> Tensor:
        """Return [B, GOAL_DIM] goal vectors."""
        f = self.family_embed(family_ids)           # [B, 32]
        m = self.mode_embed(mode_ids)               # [B, 16]

        s_emb = self.slot_embed(slot_ids)           # [B, max_slots, 16]
        # Mean-pool over real slots; fall back to zeros when no slots
        mask_f = slot_mask.float().unsqueeze(-1)    # [B, max_slots, 1]
        slot_sum = (s_emb * mask_f).sum(dim=1)      # [B, 16]
        slot_count = mask_f.sum(dim=1).clamp(min=1) # [B, 1]
        s = slot_sum / slot_count                   # [B, 16]

        combined = torch.cat([f, m, s], dim=-1)     # [B, 64]
        return self.mlp(combined)                   # [B, 128]


def encode_task_frame(
    task_frame: Dict,
    device: torch.device,
    goal_encoder: GoalEncoder,
) -> Tensor:
    """Single-item helper: task_frame dict -> [1, GOAL_DIM] tensor.

    task_frame keys used:
        task_family: str
        question_mode: str
        required_slots: List[str]
    """
    family_str = str(task_frame.get("task_family", "") or "unknown")
    mode_str = str(task_frame.get("question_mode", "") or "unknown")
    slots = [str(s) for s in (task_frame.get("required_slots") or [])]

    family_id = torch.tensor([_task_family_id(family_str)], dtype=torch.long, device=device)
    mode_id = torch.tensor([_question_mode_id(mode_str)], dtype=torch.long, device=device)

    if slots:
        sids = [_slot_id(s) for s in slots[:16]]   # cap at 16 slots
        pad = [0] * (16 - len(sids))
        slot_ids = torch.tensor([sids + pad], dtype=torch.long, device=device)
        slot_mask = torch.tensor(
            [[True] * len(sids) + [False] * len(pad)], dtype=torch.bool, device=device
        )
    else:
        slot_ids = torch.zeros((1, 16), dtype=torch.long, device=device)
        slot_mask = torch.zeros((1, 16), dtype=torch.bool, device=device)

    with torch.no_grad():
        return goal_encoder(family_id, mode_id, slot_ids, slot_mask)  # [1, 128]
