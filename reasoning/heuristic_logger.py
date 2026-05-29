"""HeuristicLogger — records every heuristic decision with features + outcome.

Every hand-tuned threshold in the system becomes a logged decision point.
Each log entry captures:
  - which heuristic fired
  - what features it saw
  - what decision it made
  - what the outcome was (filled later when the session result is known)

The log accumulates per-session, flushed to JSONL at session close.
Over time, this data trains replacement models for each heuristic.

Usage:
    logger = HeuristicLogger(session_id)
    logger.log("semantic_dedupe",
               features={"cosine_sim": 0.87, "text_len": 45},
               decision="ambiguous",
               threshold_used=0.80)
    # ... later ...
    logger.set_outcome("session_finalized", True)
    logger.flush("data/heuristic_logs/")
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@dataclass
class HeuristicDecision:
    heuristic_name: str
    features: Dict[str, Any]
    decision: Any
    threshold_used: Optional[Any] = None
    alternatives: Optional[Dict[str, Any]] = None
    timestamp: str = field(default_factory=_now_iso)


@dataclass
class HeuristicLog:
    session_id: str
    decisions: List[HeuristicDecision] = field(default_factory=list)
    outcomes: Dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=_now_iso)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "session_id": self.session_id,
            "decisions": [asdict(d) for d in self.decisions],
            "outcomes": self.outcomes,
            "timestamp": self.timestamp,
        }


class HeuristicLogger:
    """Per-session logger for heuristic decisions."""

    def __init__(self, session_id: str):
        self.log = HeuristicLog(session_id=session_id)

    def record(
        self,
        name: str,
        features: Dict[str, Any],
        decision: Any,
        threshold_used: Any = None,
        alternatives: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.log.decisions.append(HeuristicDecision(
            heuristic_name=name,
            features=features,
            decision=decision,
            threshold_used=threshold_used,
            alternatives=alternatives,
        ))

    def set_outcome(self, key: str, value: Any) -> None:
        self.log.outcomes[key] = value

    def set_outcomes(self, outcomes: Dict[str, Any]) -> None:
        self.log.outcomes.update(outcomes)

    def flush(self, log_dir: str | Path) -> Path:
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        out_path = log_dir / "heuristic_decisions.jsonl"
        row = json.dumps(self.log.to_dict(), ensure_ascii=False)
        with out_path.open("a", encoding="utf-8") as f:
            f.write(row + "\n")
        return out_path

    @property
    def decision_count(self) -> int:
        return len(self.log.decisions)

    def decisions_for(self, name: str) -> List[HeuristicDecision]:
        return [d for d in self.log.decisions if d.heuristic_name == name]


# ---------------------------------------------------------------------------
# Analysis utilities (for training replacement models)
# ---------------------------------------------------------------------------

def load_heuristic_logs(log_dir: str | Path) -> List[Dict[str, Any]]:
    """Load all logged sessions from the JSONL file."""
    path = Path(log_dir) / "heuristic_decisions.jsonl"
    if not path.exists():
        return []
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return rows


def extract_training_data(
    logs: List[Dict[str, Any]],
    heuristic_name: str,
) -> List[Dict[str, Any]]:
    """Extract (features, decision, outcome) triples for one heuristic.

    Returns rows suitable for training a replacement model:
    [{"features": {...}, "decision": ..., "outcome": {...}}, ...]
    """
    rows = []
    for session in logs:
        outcomes = session.get("outcomes", {})
        for d in session.get("decisions", []):
            if d.get("heuristic_name") != heuristic_name:
                continue
            rows.append({
                "features": d.get("features", {}),
                "decision": d.get("decision"),
                "threshold_used": d.get("threshold_used"),
                "outcome": outcomes,
                "session_id": session.get("session_id"),
            })
    return rows


def summary_stats(logs: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Quick stats on the heuristic log."""
    total_decisions = 0
    by_name: Dict[str, int] = {}
    sessions_with_outcomes = 0
    for session in logs:
        if session.get("outcomes"):
            sessions_with_outcomes += 1
        for d in session.get("decisions", []):
            name = d.get("heuristic_name", "?")
            by_name[name] = by_name.get(name, 0) + 1
            total_decisions += 1
    return {
        "total_sessions": len(logs),
        "sessions_with_outcomes": sessions_with_outcomes,
        "total_decisions": total_decisions,
        "by_heuristic": dict(sorted(by_name.items(), key=lambda x: -x[1])),
    }
