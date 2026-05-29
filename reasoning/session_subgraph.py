"""Session subgraph controller.

Wraps a `SessionSubgraph` (the data structure) with full-CRUD operations
that journal every mutation to the audit log before applying it.

The dispatcher (Sub-phase 1.5) is the caller — it parses procedure
invocations and state mutations from the reasoner's output and translates
them into calls on this controller.

Persistence: always-persist setting (resolved 2026-05-20). At session
close, writes:
    data/session_subgraphs/{session_id}/subgraph.json
    data/session_subgraphs/{session_id}/audit_log.jsonl
"""
from __future__ import annotations

import copy
import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from reasoning.audit_log import AuditLogger, _delete_dotted, _get_dotted, _set_dotted
from reasoning.schemas import (
    AuditEntry,
    FailurePatternNode,
    ProcedureNode,
    Provenance,
    SessionEdge,
    SessionObjectNode,
    SessionSubgraph,
)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _gen_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


class ObjectNotFound(KeyError):
    """Raised when a CRUD op targets a session object that doesn't exist."""


class SessionSubgraphController:
    """Full CRUD over the per-session subgraph, with audit journaling."""

    def __init__(self, session_id: str, query: str, graph_id: str):
        self.subgraph = SessionSubgraph(
            session_id=session_id,
            query=query,
            graph_id=graph_id,
            started_at=_now_iso(),
        )

    # ---- step management ------------------------------------------------ #

    @property
    def step_index(self) -> int:
        return self.subgraph.step_count

    def step(self) -> None:
        """Advance the session's step counter. Call once per reasoning iteration."""
        self.subgraph.step_count += 1

    # ---- CREATE: object ------------------------------------------------- #

    def create_object(
        self,
        procedure: ProcedureNode,
        initial_state: Dict[str, Any],
        triggered_by: str,
    ) -> str:
        """Instantiate a Procedure as a session-scoped object with state."""
        obj_id = _gen_id("so")
        obj = SessionObjectNode(
            id=obj_id,
            procedure_id=procedure.id,
            name=procedure.name,
            state=copy.deepcopy(initial_state),
            created_step=self.step_index,
            provenance=Provenance(
                created_in_session_id=self.subgraph.session_id,
                last_modified=_now_iso(),
            ),
        )
        obj_dict = obj.to_dict()
        self.subgraph.nodes[obj_id] = obj_dict
        self._journal(
            object_id=obj_id,
            operation="create",
            field_path="",
            old_value=None,
            new_value=obj_dict,
            triggered_by=triggered_by,
        )
        return obj_id

    def from_loose_object(
        self,
        name: str,
        fields: List[str],
        initial_state: Dict[str, Any],
        triggered_by: str = "v4.create_object",
    ) -> str:
        """Convenience for callers that don't have a ProcedureNode in hand.

        Constructs an anonymous ProcedureNode (state_schema inferred as
        'any' for each declared field) and then calls create_object().
        Intended for ad-hoc workspaces from answerers that use loose
        {name, fields, state} dicts rather than the full Procedure schema
        (e.g., answerer_v4's create_object tool).

        Returns the session-object id.
        """
        proc = ProcedureNode(
            id=_gen_id("v4anonproc"),
            name=name,
            purpose=f"Anonymous workspace created by v4 for {name!r}",
            when_to_use="ad-hoc; created via SessionSubgraphController.from_loose_object",
            signature={"inputs": [], "outputs": []},
            state_schema={f: "any" for f in fields},
            body="",
            example_use=None,
            provenance=Provenance(
                created_in_session_id=self.subgraph.session_id,
                last_modified=_now_iso(),
            ),
        )
        # Restrict initial_state to declared fields (same contract as v4's V4Tools).
        seeded = {f: initial_state.get(f) for f in fields}
        return self.create_object(proc, seeded, triggered_by)

    # ---- READ ----------------------------------------------------------- #

    def read_object(self, object_id: str, field_path: str = "") -> Any:
        """Read a field (dotted path) on a session object.

        field_path == ""  ->  returns the whole object dict
        Other paths       ->  returns the value at that path.
        Read is also journaled (high-fidelity audit), but does not mutate.
        """
        self._require_object(object_id)
        obj = self.subgraph.nodes[object_id]
        value: Any
        if field_path == "":
            value = copy.deepcopy(obj)
        else:
            value = copy.deepcopy(_get_dotted(obj, field_path))
        self._journal(
            object_id=object_id,
            operation="read",
            field_path=field_path,
            old_value=value,
            new_value=None,
            triggered_by="(read)",
        )
        return value

    # ---- UPDATE --------------------------------------------------------- #

    def update_object(
        self,
        object_id: str,
        field_path: str,
        new_value: Any,
        triggered_by: str,
    ) -> None:
        """Full-CRUD update at any dotted path. Replaces the value wholesale.

        field_path == "" replaces the entire object dict (rare; usually used
        for create-with-id or full-object-rewrite cases).
        """
        self._require_object(object_id)
        obj = self.subgraph.nodes[object_id]
        try:
            old_value: Any = copy.deepcopy(_get_dotted(obj, field_path)) if field_path else copy.deepcopy(obj)
        except KeyError:
            old_value = None
        self._journal(
            object_id=object_id,
            operation="update",
            field_path=field_path,
            old_value=old_value,
            new_value=copy.deepcopy(new_value),
            triggered_by=triggered_by,
        )
        if field_path == "":
            self.subgraph.nodes[object_id] = copy.deepcopy(new_value)
        else:
            _set_dotted(obj, field_path, copy.deepcopy(new_value))
        # Stamp last_modified on provenance
        try:
            _set_dotted(self.subgraph.nodes[object_id], "provenance.last_modified", _now_iso())
        except Exception:
            pass

    # ---- DELETE --------------------------------------------------------- #

    def delete_object(
        self,
        object_id: str,
        field_path: Optional[str],
        triggered_by: str,
    ) -> None:
        """Delete a field (if field_path) or the whole object (if None or '')."""
        self._require_object(object_id)
        obj = self.subgraph.nodes[object_id]
        if field_path in (None, ""):
            old_value = copy.deepcopy(obj)
            self._journal(
                object_id=object_id,
                operation="delete",
                field_path="",
                old_value=old_value,
                new_value=None,
                triggered_by=triggered_by,
            )
            del self.subgraph.nodes[object_id]
            return

        try:
            old_value = copy.deepcopy(_get_dotted(obj, field_path))
        except KeyError:
            old_value = None
        self._journal(
            object_id=object_id,
            operation="delete",
            field_path=field_path,
            old_value=old_value,
            new_value=None,
            triggered_by=triggered_by,
        )
        _delete_dotted(obj, field_path)

    # ---- failure patterns + edges -------------------------------------- #

    def add_failure_pattern(self, failure: FailurePatternNode, triggered_by: str) -> str:
        """Insert a failure_pattern node into the session subgraph.

        Failure patterns get their own audit entry (operation='create',
        field_path='') so they have the same debuggability as session
        objects. They are not mutable post-creation.
        """
        fp_dict = failure.to_dict()
        self.subgraph.nodes[failure.id] = fp_dict
        self._journal(
            object_id=failure.id,
            operation="create",
            field_path="",
            old_value=None,
            new_value=fp_dict,
            triggered_by=triggered_by,
        )
        return failure.id

    def add_edge(
        self,
        src: str,
        dst: str,
        relation: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.subgraph.edges.append(SessionEdge(
            src=src,
            dst=dst,
            relation=relation,
            metadata=metadata or {},
        ))

    # ---- persistence ---------------------------------------------------- #

    def persist(self, root: Path) -> Path:
        """Write subgraph.json + audit_log.jsonl under root/{session_id}/.
        Returns the session's directory path.
        """
        sess_dir = Path(root) / self.subgraph.session_id
        sess_dir.mkdir(parents=True, exist_ok=True)
        (sess_dir / "subgraph.json").write_text(
            json.dumps(self.subgraph.to_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        AuditLogger.persist_jsonl(self.subgraph.audit_log, sess_dir / "audit_log.jsonl")
        return sess_dir

    def close(self, root: Path) -> Path:
        """Mark session ended and persist. Returns session directory."""
        self.subgraph.ended_at = _now_iso()
        return self.persist(root)

    # ---- queries -------------------------------------------------------- #

    def logger(self) -> AuditLogger:
        """Returns a stateless query view over this session's audit log."""
        return AuditLogger(self.subgraph.audit_log)

    # ---- internals ------------------------------------------------------ #

    def _require_object(self, object_id: str) -> None:
        if object_id not in self.subgraph.nodes:
            raise ObjectNotFound(f"No object {object_id!r} in session subgraph")

    def _journal(
        self,
        object_id: str,
        operation: str,
        field_path: str,
        old_value: Any,
        new_value: Any,
        triggered_by: str,
    ) -> None:
        # Deep-copy old/new values into the audit entry. Without this, the
        # entry shares references with live state and subsequent mutations
        # silently corrupt past snapshots — which would defeat the whole
        # point of journaling. Test coverage: test_intermediate_states.
        self.subgraph.audit_log.append(AuditEntry(
            session_id=self.subgraph.session_id,
            step_index=self.step_index,
            object_id=object_id,
            operation=operation,  # type: ignore[arg-type]
            field_path=field_path,
            old_value=copy.deepcopy(old_value),
            new_value=copy.deepcopy(new_value),
            triggered_by_text=triggered_by,
            timestamp=_now_iso(),
        ))
