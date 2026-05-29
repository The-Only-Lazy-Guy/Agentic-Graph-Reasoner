"""Audit log queries + JSONL persistence.

The audit log itself lives on `SessionSubgraph.audit_log` (a list of
AuditEntry). This module provides a stateless query layer over that list:

  - replay(up_to_step)     : entries in order, optionally truncated
  - diff(object_id, ...)   : entries on a specific object in a range
  - reconstruct_state(...) : replays mutations to produce the object state
                             at a given step (the load-bearing debug
                             primitive for full-CRUD)

Persistence is JSONL (one entry per line) so append is atomic and partial
reads are safe.

See PHASE1_PLAN.md §4 + §11.1 for the design rationale.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from reasoning.schemas import AuditEntry


class AuditLogger:
    """Stateless query layer over a list of AuditEntry."""

    def __init__(self, audit_log: List[AuditEntry]):
        self.audit_log = audit_log

    # ---- read queries ---------------------------------------------------- #

    def replay(self, up_to_step: Optional[int] = None) -> List[AuditEntry]:
        """Return entries in original order, optionally truncated to those
        at step_index <= up_to_step."""
        if up_to_step is None:
            return list(self.audit_log)
        return [e for e in self.audit_log if e.step_index <= up_to_step]

    def diff(self, object_id: str, from_step: int, to_step: int) -> List[AuditEntry]:
        """Mutations on a specific object within a step range (inclusive)."""
        return [
            e for e in self.audit_log
            if e.object_id == object_id and from_step <= e.step_index <= to_step
        ]

    def reconstruct_state(self, object_id: str, at_step: int) -> Optional[Dict[str, Any]]:
        """Replay mutations on `object_id` up to and including step `at_step`.

        Returns the full object dict at that point, or None if the object
        did not exist (or had been deleted) at that step.

        This is the debug primitive that justifies full-CRUD: with replay,
        any state at any step is recoverable from the audit log alone.
        """
        state: Optional[Dict[str, Any]] = None
        for entry in self.audit_log:
            if entry.object_id != object_id:
                continue
            if entry.step_index > at_step:
                break
            state = _apply_entry(state, entry)
        return state

    # ---- persistence ---------------------------------------------------- #

    @staticmethod
    def persist_jsonl(audit_log: List[AuditEntry], path: Path) -> None:
        """Atomic-append-safe write of every entry as one JSON object per line."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            for entry in audit_log:
                f.write(json.dumps(entry.to_dict(), ensure_ascii=False))
                f.write("\n")

    @staticmethod
    def load_jsonl(path: Path) -> List[AuditEntry]:
        """Load a JSONL audit log file."""
        entries: List[AuditEntry] = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                entries.append(AuditEntry.from_dict(json.loads(line)))
        return entries


# ---- helpers for the replay logic --------------------------------------- #

def _apply_entry(state: Optional[Dict[str, Any]], entry: AuditEntry) -> Optional[Dict[str, Any]]:
    """Apply one audit entry to a state dict. Returns new state."""
    if entry.operation == "create":
        # The new_value contains the full object dict
        return copy.deepcopy(entry.new_value) if entry.new_value is not None else None

    if entry.operation == "delete":
        if entry.field_path == "":
            return None                                    # whole object gone
        if state is None:
            return None
        new_state = copy.deepcopy(state)
        _delete_dotted(new_state, entry.field_path)
        return new_state

    if entry.operation == "update":
        if state is None:
            # Update on something that doesn't exist yet — skip rather than crash
            return None
        new_state = copy.deepcopy(state)
        if entry.field_path == "":
            return copy.deepcopy(entry.new_value)
        _set_dotted(new_state, entry.field_path, entry.new_value)
        return new_state

    # read: doesn't mutate
    return state


def _get_dotted(obj: Dict[str, Any], path: str) -> Any:
    """Traverse a dotted path in a dict. Raises KeyError if any segment is missing.

    Example: _get_dotted({'state': {'foo': 1}}, 'state.foo') -> 1
    """
    cur: Any = obj
    for part in path.split("."):
        if not isinstance(cur, dict):
            raise KeyError(f"Path {path!r} hits non-dict at segment {part!r}")
        cur = cur[part]
    return cur


def _set_dotted(obj: Dict[str, Any], path: str, value: Any) -> None:
    """Set obj[a][b][c] = value where path == 'a.b.c'. Creates intermediate
    dicts when missing — necessary because the agent may mutate paths that
    don't yet exist in initial state."""
    parts = path.split(".")
    cur: Dict[str, Any] = obj
    for part in parts[:-1]:
        nxt = cur.get(part)
        if not isinstance(nxt, dict):
            nxt = {}
            cur[part] = nxt
        cur = nxt
    cur[parts[-1]] = value


def _delete_dotted(obj: Dict[str, Any], path: str) -> None:
    """Delete the leaf at the given dotted path. Silently no-ops if any
    intermediate segment is missing."""
    parts = path.split(".")
    cur: Any = obj
    for part in parts[:-1]:
        if not isinstance(cur, dict) or part not in cur:
            return
        cur = cur[part]
    if isinstance(cur, dict) and parts[-1] in cur:
        del cur[parts[-1]]
