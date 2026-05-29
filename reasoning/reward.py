"""Reward function for rejection sampling (Stage 2 training).

Scores a v4 session on five dimensions:
  - grounding:  did the answer cite graph nodes?
  - coverage:   did it address task-frame items?
  - completion: did the session finalize within budget?
  - efficiency: fewer steps = better
  - learning:   did graph edits improve health?

The composite reward is used to rank N completions per question and
keep the top-k for SFT re-training.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass
class RewardBreakdown:
    grounding: float = 0.0
    coverage: float = 0.0
    completion: float = 0.0
    efficiency: float = 0.0
    learning: float = 0.0
    total: float = 0.0

    def to_dict(self) -> Dict[str, float]:
        return {
            "grounding": round(self.grounding, 4),
            "coverage": round(self.coverage, 4),
            "completion": round(self.completion, 4),
            "efficiency": round(self.efficiency, 4),
            "learning": round(self.learning, 4),
            "total": round(self.total, 4),
        }


def compute_reward(
    metrics: Dict[str, Any],
    *,
    weights: Optional[Dict[str, float]] = None,
) -> RewardBreakdown:
    """Compute reward from a corpus row's metrics + quality fields.

    Expected keys in `metrics`:
      - steps, max_steps (from metrics)
      - finalized (from quality)
      - coverage_addressed_pct (from metrics)
      - nodes_read (from grounding or tool_call_count as proxy)
      - graph_edits_count (from metrics)
      - health_delta (from graph health, if available)
    """
    w = weights or {
        "grounding": 0.30,
        "coverage": 0.20,
        "completion": 0.20,
        "efficiency": 0.15,
        "learning": 0.15,
    }

    r = RewardBreakdown()

    # Grounding: did the model read nodes before answering?
    nodes_read = metrics.get("nodes_read", metrics.get("tool_call_count", 0))
    if nodes_read > 0:
        r.grounding = min(1.0, nodes_read / 5.0)  # 5+ reads = full score
    else:
        r.grounding = 0.0

    # Coverage: task-frame items addressed
    r.coverage = float(metrics.get("coverage_addressed_pct", metrics.get("coverage_pct", 0.0)))

    # Completion: did the session finalize?
    r.completion = 1.0 if metrics.get("finalized", False) else 0.0

    # Efficiency: fewer steps relative to budget = better
    steps = max(1, metrics.get("steps", 1))
    max_steps = max(1, metrics.get("max_steps", 20))
    r.efficiency = max(0.0, 1.0 - (steps / max_steps))

    # Learning: did graph edits contribute positively?
    edits = metrics.get("graph_edits_count", metrics.get("graph_edits_proposed", 0))
    health_delta = metrics.get("health_delta", 0.0)
    if edits > 0 and health_delta >= 0:
        r.learning = min(1.0, edits / 5.0)
    elif edits > 0 and health_delta < 0:
        r.learning = 0.2  # partial credit for attempting
    else:
        r.learning = 0.0

    # Composite
    r.total = (
        w["grounding"] * r.grounding +
        w["coverage"] * r.coverage +
        w["completion"] * r.completion +
        w["efficiency"] * r.efficiency +
        w["learning"] * r.learning
    )
    return r


def score_corpus(corpus_path: str) -> list:
    """Score every session in the corpus. Returns sorted by reward (descending)."""
    import json
    from pathlib import Path

    rows = []
    with Path(corpus_path).open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            metrics = {**row.get("metrics", {}), **row.get("quality", {})}
            reward = compute_reward(metrics)
            rows.append({
                "session_id": row.get("session_id", ""),
                "question": row.get("input", {}).get("question", "")[:60],
                "reward": reward.to_dict(),
                "total": reward.total,
            })

    return sorted(rows, key=lambda r: -r["total"])
