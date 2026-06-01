"""Compound exit guard for recurrent attention loops (Layer 8 + Layer 20).

Exit requires ALL conditions to hold — attention stability alone is not
sufficient (prevents false convergence on wrong nodes).

See V5_ARCHITECTURE.md §6.5 for full rationale.

Fixes applied:
  - max_loops_reached: check loop_idx >= r_max - 1 (loop runs 0..r_max-1)
  - Slot fill: only checks required slots from task_frame, not all NUM_SLOTS
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
from torch import Tensor

from v5.loop_state import LoopState

ENTROPY_THRESHOLD = 1.5
SLOT_FILL_THRESHOLD = 0.85
EPISTEMIC_THRESHOLD = 0.70
SHORTCUT_THRESHOLD = 0.85
TOP_K_NODES = 3
FORCE_FALLBACK_CONTEXTS = frozenset({"no_graph", "weak_evidence", "unsupported"})
NORMAL_ANSWER = "normal_answer"
SOFT_FALLBACK = "soft_fallback"
HARD_FALLBACK = "hard_fallback"


@dataclass(frozen=True)
class FallbackPolicy:
    """Three-state fallback classification for controller policy.

    `fallback_needed()` remains the backward-compatible boolean. This richer
    view lets runtime/controller code distinguish true hard unsafety from soft
    neural underconfidence before deciding whether to answer, caveat, verify, or
    fully fall back.
    """

    state: str
    fallback_needed: bool
    hard_reasons: List[str]
    soft_reasons: List[str]
    primary_idx: Optional[int]
    slots_ok: bool
    no_invalidators: bool
    epistemic_ok: bool

    def to_dict(self) -> dict:
        return {
            "state": self.state,
            "fallback_needed": self.fallback_needed,
            "hard_reasons": list(self.hard_reasons),
            "soft_reasons": list(self.soft_reasons),
            "primary_idx": self.primary_idx,
            "slots_ok": self.slots_ok,
            "no_invalidators": self.no_invalidators,
            "epistemic_ok": self.epistemic_ok,
        }


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


def _force_fallback(task_frame: Optional[dict]) -> bool:
    """True when the controller already knows graph support is unavailable.

    This is used for no-graph / weak-evidence prompts: even if the learned heads
    hallucinate high slot, epistemic, or shortcut confidence on an irrelevant
    node, the graph adapter must not self-certify the answer.
    """
    if not task_frame:
        return False
    if bool(task_frame.get("force_fallback")):
        return True
    context = str(task_frame.get("graph_context") or "").lower()
    return context in FORCE_FALLBACK_CONTEXTS


def _shortcut_exit_allowed(task_frame: Optional[dict]) -> bool:
    if _force_fallback(task_frame):
        return False
    if task_frame and task_frame.get("allow_shortcut_exit") is False:
        return False
    return True


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


def _primary_support_index(state: LoopState, top_k: Optional[List[int]] = None) -> Optional[int]:
    """Return the evidence node used for epistemic/fallback gating.

    The attention top-1 is not always the best support node: diagnostics showed
    many direct_judgment failures where gold evidence was in top-k but not rank
    one. When available, support_scores_r reranks only the existing top-k so the
    gate can choose nearby support without expanding its safety surface.
    """
    top_k = top_k if top_k is not None else _top_k_indices(state.node_scores_r)
    if not top_k:
        return None
    support = state.support_scores_r
    if support is None:
        return int(top_k[0])
    scores = support.squeeze(0)
    if scores.numel() <= max(top_k):
        return int(top_k[0])
    return int(max(top_k, key=lambda idx: float(scores[idx].item())))


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

    if _force_fallback(task_frame):
        return False, ""

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
    primary = _primary_support_index(state, top_k)
    epistemic_ok = primary is not None and float(epi[primary].item()) >= EPISTEMIC_THRESHOLD

    # 6. Shortcut path
    shortcut_val = float(state.shortcut_validity_r.item())
    if (
        _shortcut_exit_allowed(task_frame)
        and shortcut_val > SHORTCUT_THRESHOLD
        and no_invalidators
        and epistemic_ok
        and slots_ok
    ):
        return True, "shortcut_verified"

    if attention_stable and slots_ok and no_invalidators and epistemic_ok:
        return True, "all_conditions_met"

    return False, ""


def fallback_needed(
    state: LoopState,
    task_frame: Optional[dict] = None,
) -> bool:
    """True when max_loops_reached but reasoning still incomplete."""
    return fallback_policy(state, task_frame).fallback_needed


def fallback_policy(
    state: LoopState,
    task_frame: Optional[dict] = None,
) -> FallbackPolicy:
    """Classify fallback as hard safety, soft calibration, or normal answer.

    Hard fallback is reserved for explicit safety blockers: no/weak graph support
    from the controller, empty node pool at the hard cap, or active invalidators.
    Soft fallback means the old boolean gate would still fall back, but only
    because learned confidence heads are below threshold.
    """
    hard_reasons: List[str] = []
    soft_reasons: List[str] = []

    if _force_fallback(task_frame):
        hard_reasons.append("forced_fallback")

    if state.exit_reason != "max_loops_reached":
        if hard_reasons:
            return FallbackPolicy(
                state=HARD_FALLBACK,
                fallback_needed=True,
                hard_reasons=hard_reasons,
                soft_reasons=[],
                primary_idx=None,
                slots_ok=False,
                no_invalidators=True,
                epistemic_ok=False,
            )
        return FallbackPolicy(
            state=NORMAL_ANSWER,
            fallback_needed=False,
            hard_reasons=[],
            soft_reasons=[],
            primary_idx=None,
            slots_ok=True,
            no_invalidators=True,
            epistemic_ok=True,
        )

    if state.node_scores_r.shape[-1] == 0:
        hard_reasons.append("empty_pool")

    required_indices = _required_slot_indices(task_frame)
    slots_ok = _all_slots_filled(state.slot_state_r, required_indices=required_indices)
    if not slots_ok:
        soft_reasons.append("missing_slot")

    top_k = _top_k_indices(state.node_scores_r)
    inv = state.invalidator_flags_r.squeeze(0)
    no_invalidators = not any(float(inv[i].item()) > 0.5 for i in top_k)
    if not no_invalidators:
        hard_reasons.append("invalidator_active")

    # Primary (highest-attention) node must be epistemically confident; a
    # secondary attended node being uncertain is informative context, not a
    # reason to fall back. Mirrors the should_exit_loop epistemic gate.
    epi = state.epistemic_confidence_r.squeeze(0)
    primary = _primary_support_index(state, top_k)
    epistemic_ok = primary is not None and float(epi[primary].item()) >= EPISTEMIC_THRESHOLD
    if not epistemic_ok:
        soft_reasons.append("low_epistemic")

    if hard_reasons:
        return FallbackPolicy(
            state=HARD_FALLBACK,
            fallback_needed=True,
            hard_reasons=hard_reasons,
            soft_reasons=soft_reasons,
            primary_idx=primary,
            slots_ok=slots_ok,
            no_invalidators=no_invalidators,
            epistemic_ok=epistemic_ok,
        )
    if soft_reasons:
        return FallbackPolicy(
            state=SOFT_FALLBACK,
            fallback_needed=True,
            hard_reasons=[],
            soft_reasons=soft_reasons,
            primary_idx=primary,
            slots_ok=slots_ok,
            no_invalidators=no_invalidators,
            epistemic_ok=epistemic_ok,
        )
    return FallbackPolicy(
        state=NORMAL_ANSWER,
        fallback_needed=False,
        hard_reasons=[],
        soft_reasons=[],
        primary_idx=primary,
        slots_ok=slots_ok,
        no_invalidators=no_invalidators,
        epistemic_ok=epistemic_ok,
    )
