"""Compound exit guard for recurrent attention loops (Layer 8 + Layer 20).

Exit requires ALL conditions to hold — attention stability alone is not
sufficient (prevents false convergence on wrong nodes).

See V5_ARCHITECTURE.md §6.5 for full rationale.

Fixes applied:
  - max_loops_reached: check loop_idx >= r_max - 1 (loop runs 0..r_max-1)
  - Slot fill: only checks required slots from task_frame, not all NUM_SLOTS
"""
from __future__ import annotations

from typing import List, Optional, Tuple

import torch
from torch import Tensor

from v5.loop_state import LoopState

ENTROPY_THRESHOLD = 1.5
SLOT_FILL_THRESHOLD = 0.85
EPISTEMIC_THRESHOLD = 0.70
SHORTCUT_THRESHOLD = 0.85
TOP_K_NODES = 3


def _attention_entropy(node_scores: Tensor) -> float:
    probs = torch.softmax(node_scores.squeeze(0), dim=0)
    entropy = -(probs * (probs + 1e-9).log()).sum()
    return float(entropy.item())


def _required_slot_indices(task_frame: Optional[dict]) -> Optional[List[int]]:
    """Return slot vocab indices for required slots in task_frame, or None."""
    if not task_frame:
        return None
    slots = task_frame.get("required_slots") or []
    if not slots:
        return None
    from v5.goal_encoder import SLOT_ID
    return [SLOT_ID.get(str(s), SLOT_ID["unknown"]) for s in slots]


def _all_slots_filled(
    slot_state: Tensor,
    threshold: float = SLOT_FILL_THRESHOLD,
    required_indices: Optional[List[int]] = None,
) -> bool:
    """True if all required slots have fill-confidence >= threshold.

    If required_indices is None, checks all slots (conservative fallback).
    """
    s = slot_state.squeeze(0)   # [NUM_SLOTS]
    if required_indices is not None and len(required_indices) > 0:
        req = torch.tensor(required_indices, dtype=torch.long, device=s.device)
        return bool((s[req] >= threshold).all().item())
    return bool((s >= threshold).all().item())


def _top_k_indices(node_scores: Tensor, k: int = TOP_K_NODES) -> List[int]:
    scores = node_scores.squeeze(0)
    k = min(k, scores.shape[0])
    return scores.topk(k).indices.tolist()


def should_exit_loop(
    state: LoopState,
    loop_idx: int,
    r_max: int,
    task_frame: Optional[dict] = None,
) -> Tuple[bool, str]:
    """Compound exit guard. Returns (should_exit, reason_string).

    Hard cap fires on the LAST iteration (loop_idx == r_max - 1) so that
    exit_reason is always set inside the loop body.
    """
    # 1. Hard cap — fires on last iteration (loop runs 0..r_max-1)
    if loop_idx >= r_max - 1:
        return True, "max_loops_reached"

    N = state.node_scores_r.shape[-1]
    if N == 0:
        return True, "empty_node_pool"

    top_k = _top_k_indices(state.node_scores_r)
    required_indices = _required_slot_indices(task_frame)

    # 2. Attention stability
    attention_stable = _attention_entropy(state.node_scores_r) < ENTROPY_THRESHOLD

    # 3. Required slots filled
    slots_ok = _all_slots_filled(state.slot_state_r, required_indices=required_indices)

    # 4. No active invalidators on top nodes
    inv = state.invalidator_flags_r.squeeze(0)
    no_invalidators = not any(float(inv[i].item()) > 0.5 for i in top_k)

    # 5. Epistemic confidence on the PRIMARY (highest-attention) node.
    # Checking every top-k node is too strict: attention legitimately includes
    # contradicting / uncertain evidence, which would block exit forever. The
    # answer rests on the primary attended node, so gate on that one.
    epi = state.epistemic_confidence_r.squeeze(0)
    primary = top_k[0] if top_k else 0
    epistemic_ok = float(epi[primary].item()) >= EPISTEMIC_THRESHOLD

    # 6. Shortcut path
    shortcut_val = float(state.shortcut_validity_r.item())
    if shortcut_val > SHORTCUT_THRESHOLD and no_invalidators and epistemic_ok and slots_ok:
        return True, "shortcut_verified"

    if attention_stable and slots_ok and no_invalidators and epistemic_ok:
        return True, "all_conditions_met"

    return False, ""


def fallback_needed(
    state: LoopState,
    task_frame: Optional[dict] = None,
) -> bool:
    """True when max_loops_reached but reasoning still incomplete."""
    if state.exit_reason != "max_loops_reached":
        return False

    required_indices = _required_slot_indices(task_frame)
    if not _all_slots_filled(state.slot_state_r, required_indices=required_indices):
        return True

    top_k = _top_k_indices(state.node_scores_r)
    inv = state.invalidator_flags_r.squeeze(0)
    if any(float(inv[i].item()) > 0.5 for i in top_k):
        return True

    # Primary (highest-attention) node must be epistemically confident; a
    # secondary attended node being uncertain is informative context, not a
    # reason to fall back. Mirrors the should_exit_loop epistemic gate.
    epi = state.epistemic_confidence_r.squeeze(0)
    primary = top_k[0] if top_k else 0
    if float(epi[primary].item()) < EPISTEMIC_THRESHOLD:
        return True

    return False
