"""Shared cheap token-count estimation for local benchmark telemetry.

Exact accounting depends on the active runtime tokenizer. Phase 3E only needs
a stable prompt+output size signal, so keep the historical 4 chars/token
heuristic in one place until a model-specific tokenizer is wired in.
"""
from __future__ import annotations


def estimate_token_count(text: str) -> int:
    if not text:
        return 0
    return max(1, (len(text) + 3) // 4)
