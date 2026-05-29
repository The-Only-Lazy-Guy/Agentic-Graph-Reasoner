"""Compound exit guard for recurrent attention loops (Layer 8 + Layer 20).

Exit requires ALL conditions to hold — attention stability alone is not
sufficient (prevents false convergence on wrong nodes).

See V5_ARCHITECTURE.md §6.5 for full rationale.
"""
from __future__ import annotations

from typing import List, Optional, Tuple

import torch
from torch import Tensor

from v5.loop_state import LoopState

ENTROPY_THRESHOLD = 1.5          # nats; lower = more concentrated attention
SLOT_FILL_THRESHOLD = 0.85       # minimum confidence to consider a slot filled
EPISTEMIC_THRESHOLD = 0.70       # minimum epistemic confidence on top nodes
SHORTCUT_THRESHOLD = 0.85        # minimum shortcut_validity to attempt shortcut exit
TOP_K_NODES = 3                  # nodes checked for invalidators and epistemic confidence


def _attention_entropy(node_scores: Tensor) -> float:
    """Shannon entropy of softmax attention distribution over node scores."""
    probs = torch.softmax(node_scores.squeeze(0), dim=0)
    entropy = -(probs * (probs + 1e-9).log()).sum()
    return float(entropy.item())


def _all_slots_filled(slot_state: Tensor, threshold: float = SLOT_FILL_THRESHOLD) -> bool:
    """True if every slot has fill-confidence >= threshold."""
    return bool((slot_state.squeeze(0) >= threshold).all().item())


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
    """Compound exit guard.

    Returns (should_exit, reason_string).

    Compound rule (all must hold for non-shortcut exit):
      1. Hard cap always fires
      2. Attention entropy below threshold
      3. All slots filled with high confidence
      4. No active invalidators on top-K nodes
      5. High epistemic confidence on top-K nodes
      6. Shortcut path: extra guard (shortcut_validity + precondition match)
    """
    # 1. Hard cap
    if loop_idx >= r_max:
        return True, "max_loops_reached"

    N = state.node_scores_r.shape[-1]
    if N == 0:
        return True, "empty_node_pool"

    top_k = _top_k_indices(state.node_scores_r)

    # 2. Attention stability
    entropy = _attention_entropy(state.node_scores_r)
    attention_stable = entropy < ENTROPY_THRESHOLD

    # 3. Slot fill
    slots_ok = _all_slots_filled(state.slot_state_r)

    # 4. No active invalidators on top nodes
    inv = state.invalidator_flags_r.squeeze(0)
    no_invalidators = not any(float(inv[i].item()) > 0.5 for i in top_k)

    # 5. Epistemic confidence on top nodes
    epi = state.epistemic_confidence_r.squeeze(0)
    epistemic_ok = all(
        float(epi[i].item()) >= EPISTEMIC_THRESHOLD for i in top_k
    )

    # 6. Shortcut path (shortcut_validity > threshold + all guards pass)
    shortcut_val = float(state.shortcut_validity_r.item())
    if shortcut_val > SHORTCUT_THRESHOLD and no_invalidators and epistemic_ok and slots_ok:
        return True, "shortcut_verified"

    # Full compound condition
    if attention_stable and slots_ok and no_invalidators and epistemic_ok:
        return True, "all_conditions_met"

    return False, ""


def fallback_needed(
    state: LoopState,
    coverage_threshold: float = 0.6,
    contradiction_threshold: float = 0.4,
) -> bool:
    """True when the loop ended but reasoning is still incomplete.

    Triggers V4 external tool loop as fallback (Deep mode).
    Checked after max_loops_reached exit.
    """
    if state.exit_reason != "max_loops_reached":
        return False

    slots_ok = _all_slots_filled(state.slot_state_r)
    if not slots_ok:
        return True

    top_k = _top_k_indices(state.node_scores_r)
    inv = state.invalidator_flags_r.squeeze(0)
    if any(float(inv[i].item()) > 0.5 for i in top_k):
        return True

    epi = state.epistemic_confidence_r.squeeze(0)
    if any(float(epi[i].item()) < EPISTEMIC_THRESHOLD for i in top_k):
        return True

    return False
