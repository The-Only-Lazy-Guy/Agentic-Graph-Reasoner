"""Workspace — Phase 3F slot-filling workspace for structured reasoning.

A workspace defines an ordered set of reasoning slots that the model
fills one at a time. Each slot captures a specific piece of reasoning
output (e.g., problem identification, approach, verification, answer).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


PAYMENT_WORKSPACE_FILL_ORDER: List[str] = [
    "problem_type",
    "constraints",
    "approach",
    "verification",
    "answer",
]

PAYMENT_WORKSPACE_SLOTS: Dict[str, Dict[str, Any]] = {
    "problem_type": {
        "label": "Problem type",
        "prompt": "Identify the category of problem this question asks about.",
        "max_attempts": 2,
    },
    "constraints": {
        "label": "Constraints",
        "prompt": "List the key constraints that bound the solution.",
        "max_attempts": 2,
    },
    "approach": {
        "label": "Approach",
        "prompt": "Describe the algorithmic approach that solves this problem under the given constraints.",
        "max_attempts": 3,
    },
    "verification": {
        "label": "Verification",
        "prompt": "Verify that the approach satisfies all constraints and correctly handles edge cases.",
        "max_attempts": 2,
    },
    "answer": {
        "label": "Answer",
        "prompt": "Write the final answer synthesizing all prior reasoning.",
        "max_attempts": 2,
    },
}


@dataclass
class WorkspaceSlot:
    name: str
    content: Optional[str] = None
    filled: bool = False


class Workspace:
    """Ordered workspace of reasoning slots filled by the model."""

    def __init__(self, fill_order: List[str]):
        self.fill_order = list(fill_order)
        self._slots: Dict[str, WorkspaceSlot] = {}
        for name in fill_order:
            self._slots[name] = WorkspaceSlot(name=name)

    def filled_slots(self) -> Dict[str, str]:
        """Return a dict of slot_name -> content for filled slots."""
        return {
            name: slot.content
            for name, slot in self._slots.items()
            if slot.filled and slot.content is not None
        }

    def fill_slot(self, name: str, content: str) -> None:
        """Fill a slot with content."""
        if name not in self._slots:
            raise ValueError(f"Unknown workspace slot: {name!r}")
        self._slots[name].content = content
        self._slots[name].filled = True

    def is_filled(self, name: str) -> bool:
        return self._slots.get(name, WorkspaceSlot(name=name)).filled

    def remaining(self) -> List[str]:
        return [name for name in self.fill_order if not self._slots[name].filled]
