from __future__ import annotations

import torch

from v5.exit_condition import fallback_needed, should_exit_loop
from v5.goal_encoder import NUM_SLOTS, SLOT_ID
from v5.loop_state import LoopState


def _confident_state(exit_reason: str | None = None) -> LoopState:
    slots = torch.zeros(1, NUM_SLOTS)
    slots[0, SLOT_ID["verdict"]] = 1.0
    slots[0, SLOT_ID["reason"]] = 1.0
    return LoopState(
        h_r=torch.zeros(1, 8),
        slot_state_r=slots,
        node_scores_r=torch.tensor([[4.0, 1.0, 0.0]]),
        shortcut_validity_r=torch.tensor([[0.99]]),
        epistemic_confidence_r=torch.tensor([[0.99, 0.1, 0.1]]),
        invalidator_flags_r=torch.zeros(1, 3),
        exit_reason=exit_reason,
    )


def test_shortcut_verified_still_exits_for_supported_graph_context():
    task_frame = {"required_slots": ["verdict", "reason"]}
    should_exit, reason = should_exit_loop(_confident_state(), 0, 4, task_frame)

    assert should_exit is True
    assert reason == "shortcut_verified"
    assert fallback_needed(_confident_state(exit_reason="shortcut_verified"), task_frame) is False


def test_no_graph_task_frame_blocks_shortcut_and_forces_fallback():
    task_frame = {
        "required_slots": ["verdict", "reason"],
        "graph_context": "no_graph",
        "allow_shortcut_exit": False,
        "force_fallback": True,
    }

    should_exit, reason = should_exit_loop(_confident_state(), 0, 4, task_frame)
    assert should_exit is False
    assert reason == ""

    should_exit, reason = should_exit_loop(_confident_state(), 3, 4, task_frame)
    assert should_exit is True
    assert reason == "max_loops_reached"
    assert fallback_needed(_confident_state(exit_reason="shortcut_verified"), task_frame) is True
    assert fallback_needed(_confident_state(exit_reason="max_loops_reached"), task_frame) is True
