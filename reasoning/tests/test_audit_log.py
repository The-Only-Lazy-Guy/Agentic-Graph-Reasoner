"""Tests for reasoning/audit_log.py.

Focus: replay, diff, reconstruct_state (the load-bearing debug primitive),
JSONL persistence round-trip.
"""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from reasoning.audit_log import (
    AuditLogger,
    _delete_dotted,
    _get_dotted,
    _set_dotted,
)
from reasoning.schemas import AuditEntry


def _create(step: int, oid: str, new_value, by: str = "create") -> AuditEntry:
    return AuditEntry(
        session_id="sess_test",
        step_index=step,
        object_id=oid,
        operation="create",
        field_path="",
        old_value=None,
        new_value=new_value,
        triggered_by_text=by,
        timestamp=f"2026-05-20T11:0{step}:00+00:00",
    )


def _update(step: int, oid: str, path: str, old, new, by: str = "update") -> AuditEntry:
    return AuditEntry(
        session_id="sess_test",
        step_index=step,
        object_id=oid,
        operation="update",
        field_path=path,
        old_value=old,
        new_value=new,
        triggered_by_text=by,
        timestamp=f"2026-05-20T11:0{step}:00+00:00",
    )


def _delete(step: int, oid: str, path: str, old) -> AuditEntry:
    return AuditEntry(
        session_id="sess_test",
        step_index=step,
        object_id=oid,
        operation="delete",
        field_path=path,
        old_value=old,
        new_value=None,
        triggered_by_text="delete",
        timestamp=f"2026-05-20T11:0{step}:00+00:00",
    )


class TestDottedHelpers(unittest.TestCase):
    def test_get(self):
        d = {"state": {"foo": [1, 2, 3], "bar": {"nested": "x"}}}
        self.assertEqual(_get_dotted(d, "state.foo"), [1, 2, 3])
        self.assertEqual(_get_dotted(d, "state.bar.nested"), "x")

    def test_set_existing(self):
        d = {"state": {"foo": 1}}
        _set_dotted(d, "state.foo", 42)
        self.assertEqual(d["state"]["foo"], 42)

    def test_set_creates_intermediate(self):
        d = {}
        _set_dotted(d, "a.b.c", 7)
        self.assertEqual(d, {"a": {"b": {"c": 7}}})

    def test_delete(self):
        d = {"state": {"foo": 1, "bar": 2}}
        _delete_dotted(d, "state.foo")
        self.assertEqual(d, {"state": {"bar": 2}})

    def test_delete_missing_no_op(self):
        d = {"state": {"foo": 1}}
        _delete_dotted(d, "state.nonexistent")
        self.assertEqual(d, {"state": {"foo": 1}})


class TestReplay(unittest.TestCase):
    def test_replay_all(self):
        log = [
            _create(0, "so_A", {"id": "so_A", "state": {}}),
            _update(1, "so_A", "state.x", None, 5),
        ]
        logger = AuditLogger(log)
        self.assertEqual(len(logger.replay()), 2)

    def test_replay_truncated(self):
        log = [
            _create(0, "so_A", {"id": "so_A", "state": {}}),
            _update(1, "so_A", "state.x", None, 5),
            _update(3, "so_A", "state.x", 5, 10),
        ]
        logger = AuditLogger(log)
        self.assertEqual(len(logger.replay(up_to_step=1)), 2)
        self.assertEqual(len(logger.replay(up_to_step=2)), 2)
        self.assertEqual(len(logger.replay(up_to_step=3)), 3)


class TestDiff(unittest.TestCase):
    def test_diff_filters_by_object(self):
        log = [
            _create(0, "so_A", {"id": "so_A", "state": {}}),
            _create(0, "so_B", {"id": "so_B", "state": {}}),
            _update(1, "so_A", "state.x", None, 5),
            _update(2, "so_B", "state.y", None, 9),
        ]
        logger = AuditLogger(log)
        diffs = logger.diff("so_A", 0, 5)
        self.assertEqual(len(diffs), 2)
        for d in diffs:
            self.assertEqual(d.object_id, "so_A")

    def test_diff_filters_by_step_range(self):
        log = [
            _create(0, "so_A", {"id": "so_A", "state": {}}),
            _update(1, "so_A", "state.x", None, 5),
            _update(3, "so_A", "state.x", 5, 10),
        ]
        logger = AuditLogger(log)
        diffs = logger.diff("so_A", 1, 2)
        self.assertEqual(len(diffs), 1)
        self.assertEqual(diffs[0].step_index, 1)


class TestReconstructState(unittest.TestCase):
    """The most important test set in this file. Validates that
    full-CRUD replay produces correct intermediate state at any step."""

    def test_create_then_query(self):
        initial = {"id": "so_A", "state": {"counter": 0}}
        log = [_create(0, "so_A", initial)]
        logger = AuditLogger(log)
        self.assertEqual(logger.reconstruct_state("so_A", 0), initial)

    def test_create_update_replay(self):
        initial = {"id": "so_A", "state": {"counter": 0}}
        log = [
            _create(0, "so_A", initial),
            _update(1, "so_A", "state.counter", 0, 5),
            _update(2, "so_A", "state.counter", 5, 7),
        ]
        logger = AuditLogger(log)
        s0 = logger.reconstruct_state("so_A", 0)
        s1 = logger.reconstruct_state("so_A", 1)
        s2 = logger.reconstruct_state("so_A", 2)
        self.assertEqual(s0["state"]["counter"], 0)
        self.assertEqual(s1["state"]["counter"], 5)
        self.assertEqual(s2["state"]["counter"], 7)

    def test_create_nested_path(self):
        initial = {"id": "so_A", "state": {"evidence": {}}}
        log = [
            _create(0, "so_A", initial),
            _update(1, "so_A", "state.evidence.nonneg_edges", None, "Edge b->c weight -1"),
            _update(2, "so_A", "state.evidence.acyclic", None, "Cycle a->b->a detected"),
        ]
        logger = AuditLogger(log)
        s2 = logger.reconstruct_state("so_A", 2)
        self.assertEqual(s2["state"]["evidence"]["nonneg_edges"], "Edge b->c weight -1")
        self.assertEqual(s2["state"]["evidence"]["acyclic"], "Cycle a->b->a detected")

    def test_delete_field(self):
        initial = {"id": "so_A", "state": {"foo": 1, "bar": 2}}
        log = [
            _create(0, "so_A", initial),
            _delete(1, "so_A", "state.foo", 1),
        ]
        logger = AuditLogger(log)
        s1 = logger.reconstruct_state("so_A", 1)
        self.assertNotIn("foo", s1["state"])
        self.assertEqual(s1["state"]["bar"], 2)

    def test_delete_whole_object(self):
        log = [
            _create(0, "so_A", {"id": "so_A", "state": {}}),
            _delete(1, "so_A", "", {"id": "so_A", "state": {}}),
        ]
        logger = AuditLogger(log)
        s1 = logger.reconstruct_state("so_A", 1)
        self.assertIsNone(s1)

    def test_object_not_yet_created(self):
        log = [_create(2, "so_A", {"id": "so_A", "state": {}})]
        logger = AuditLogger(log)
        self.assertIsNone(logger.reconstruct_state("so_A", 0))
        self.assertIsNone(logger.reconstruct_state("so_A", 1))
        self.assertIsNotNone(logger.reconstruct_state("so_A", 2))

    def test_other_object_mutations_dont_affect_target(self):
        log = [
            _create(0, "so_A", {"id": "so_A", "state": {"x": 1}}),
            _create(0, "so_B", {"id": "so_B", "state": {"y": 100}}),
            _update(1, "so_B", "state.y", 100, 999),
        ]
        logger = AuditLogger(log)
        sA = logger.reconstruct_state("so_A", 5)
        self.assertEqual(sA["state"]["x"], 1)


class TestJSONLPersistence(unittest.TestCase):
    def test_roundtrip(self):
        log = [
            _create(0, "so_A", {"id": "so_A", "state": {"x": 1}}),
            _update(1, "so_A", "state.x", 1, 2),
            _delete(2, "so_A", "state.x", 2),
        ]
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "audit_log.jsonl"
            AuditLogger.persist_jsonl(log, p)
            text = p.read_text(encoding="utf-8")
            self.assertEqual(text.count("\n"), 3)
            for line in text.splitlines():
                self.assertTrue(line.startswith("{"))
                json.loads(line)
            restored = AuditLogger.load_jsonl(p)
            self.assertEqual(len(restored), 3)
            for original, reloaded in zip(log, restored):
                self.assertEqual(original.to_dict(), reloaded.to_dict())


if __name__ == "__main__":
    unittest.main()
