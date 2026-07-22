"""membrane_session.py — the missing middle tier: long-term graph (persists) <-> session focus (this file,
ephemeral, rebuilt each run) <-> working memory (per-task slots, trm_wm.py). Wraps membrane_edits'
glowing_subgraph into a stateful per-session view that BOOSTS retrieval toward what the current
conversation has actually touched, without ever hard-filtering — a genuinely new topic mid-session must
still be answerable from the full long-term graph.
"""
from __future__ import annotations

import numpy as np


class SessionFocus:
    """A session-scoped activation state over a long-term AtomGraph. `.update(query_text)` re-seeds
    activation from the current turn; `.boost_sims(order, sims)` nudges a cosine-similarity ranking toward
    the currently-activated subgraph. Never removes a candidate — only reorders/boosts."""

    def __init__(self, graph, steps: int = 3, threshold: float = 0.1, boost: float = 0.25):
        self.g = graph
        self.steps = steps
        self.threshold = threshold
        self.boost = boost
        self.focus: set[str] = set()

    def update(self, query_text: str) -> set[str]:
        from v5.runtime.membrane_edits import glowing_subgraph
        self.focus = set(glowing_subgraph(self.g, query_text, steps=self.steps, threshold=self.threshold))
        return self.focus

    def boost_sims(self, order: list[str], sims) -> np.ndarray:
        """order/sims aligned (as from AtomGraph.matrix()). Returns a NEW array with a flat additive bonus
        for nodes in the current focus — a BOOST, not a filter: non-focused nodes keep their real score and
        can still win if they're genuinely more relevant (e.g. a new topic the session hasn't touched yet)."""
        boosted = np.asarray(sims, dtype=np.float32).copy()
        if not self.focus:
            return boosted
        for i, n in enumerate(order):
            if n in self.focus:
                boosted[i] += self.boost
        return boosted
