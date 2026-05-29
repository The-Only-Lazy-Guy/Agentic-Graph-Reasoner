"""Tests for reasoning/session_subgraph.py.

Focus: full CRUD operations correctly mutate state AND produce
consistent audit log; persistence round-trips; ObjectNotFound raises
on bad object IDs.
"""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from reasoning.session_subgraph import (
    ObjectNotFound,
    SessionSubgraphController,
)
from reasoning.schemas import (
    FailurePatternNode,
    ProcedureNode,
    Provenance,
    SessionSubgraph,
)


def _make_seed_procedure() -> ProcedureNode:
    return ProcedureNode(
        id="proc_verify_001",
        name="VerifyAlgorithmPreconditions",
        purpose="Check algorithm preconditions",
        when_to_use="When question is about algorithm applicability",
        signature={"inputs": [{"name": "algo", "type": "str"}], "outputs": []},
        state_schema={
            "preconditions_checked": "list[str]",
            "preconditions_violated": "list[str]",
            "evidence": "dict[str, str]",
        },
        body="Verify {algo}",
        example_use={"session_id": "seed", "inputs": {}, "final_state": {}},
        provenance=Provenance(created_in_session_id="seed"),
    )


class TestCreate(unittest.TestCase):
    def test_create_object_returns_id_and_records_audit(self):
        ctrl = SessionSubgraphController("sess1", "Q?", "cs4")
        proc = _make_seed_procedure()
        obj_id = ctrl.create_object(proc, {"preconditions_checked": []}, "I'll verify Dijkstra")

        self.assertTrue(obj_id.startswith("so_"))
        self.assertIn(obj_id, ctrl.subgraph.nodes)
        self.assertEqual(len(ctrl.subgraph.audit_log), 1)

        entry = ctrl.subgraph.audit_log[0]
        self.assertEqual(entry.operation, "create")
        self.assertEqual(entry.object_id, obj_id)
        self.assertEqual(entry.field_path, "")
        self.assertIsNone(entry.old_value)
        self.assertEqual(entry.new_value["procedure_id"], "proc_verify_001")
        self.assertEqual(entry.triggered_by_text, "I'll verify Dijkstra")


class TestRead(unittest.TestCase):
    def test_read_whole_object(self):
        ctrl = SessionSubgraphController("sess1", "Q?", "cs4")
        proc = _make_seed_procedure()
        obj_id = ctrl.create_object(proc, {"preconditions_checked": []}, "create")

        obj = ctrl.read_object(obj_id)
        self.assertEqual(obj["procedure_id"], "proc_verify_001")
        self.assertEqual(obj["state"]["preconditions_checked"], [])

    def test_read_dotted_field(self):
        ctrl = SessionSubgraphController("sess1", "Q?", "cs4")
        proc = _make_seed_procedure()
        obj_id = ctrl.create_object(proc, {"preconditions_checked": ["nonneg"]}, "create")

        value = ctrl.read_object(obj_id, "state.preconditions_checked")
        self.assertEqual(value, ["nonneg"])

    def test_read_journals(self):
        ctrl = SessionSubgraphController("sess1", "Q?", "cs4")
        proc = _make_seed_procedure()
        obj_id = ctrl.create_object(proc, {"a": 1}, "create")
        ctrl.read_object(obj_id, "state.a")

        read_entries = [e for e in ctrl.subgraph.audit_log if e.operation == "read"]
        self.assertEqual(len(read_entries), 1)
        self.assertEqual(read_entries[0].old_value, 1)

    def test_read_missing_object_raises(self):
        ctrl = SessionSubgraphController("sess1", "Q?", "cs4")
        with self.assertRaises(ObjectNotFound):
            ctrl.read_object("so_nonexistent")


class TestUpdate(unittest.TestCase):
    def test_update_scalar(self):
        ctrl = SessionSubgraphController("sess1", "Q?", "cs4")
        proc = _make_seed_procedure()
        obj_id = ctrl.create_object(proc, {"counter": 0}, "create")
        ctrl.step()

        ctrl.update_object(obj_id, "state.counter", 5, "set counter to 5")
        self.assertEqual(ctrl.subgraph.nodes[obj_id]["state"]["counter"], 5)

        upd = ctrl.subgraph.audit_log[-1]
        self.assertEqual(upd.operation, "update")
        self.assertEqual(upd.field_path, "state.counter")
        self.assertEqual(upd.old_value, 0)
        self.assertEqual(upd.new_value, 5)
        self.assertEqual(upd.step_index, 1)

    def test_update_replaces_list_wholesale(self):
        ctrl = SessionSubgraphController("sess1", "Q?", "cs4")
        proc = _make_seed_procedure()
        obj_id = ctrl.create_object(proc, {"preconditions_violated": []}, "create")
        ctrl.step()
        ctrl.update_object(obj_id, "state.preconditions_violated", ["nonneg_edges"], "append")
        ctrl.step()
        ctrl.update_object(
            obj_id,
            "state.preconditions_violated",
            ["nonneg_edges", "acyclic"],
            "append",
        )
        self.assertEqual(
            ctrl.subgraph.nodes[obj_id]["state"]["preconditions_violated"],
            ["nonneg_edges", "acyclic"],
        )

    def test_update_nested_dict_creates_path(self):
        ctrl = SessionSubgraphController("sess1", "Q?", "cs4")
        proc = _make_seed_procedure()
        obj_id = ctrl.create_object(proc, {"evidence": {}}, "create")
        ctrl.step()
        ctrl.update_object(obj_id, "state.evidence.nonneg_edges", "weight -1", "add")
        self.assertEqual(
            ctrl.subgraph.nodes[obj_id]["state"]["evidence"]["nonneg_edges"],
            "weight -1",
        )

    def test_update_stamps_last_modified(self):
        """last_modified should be a non-empty ISO timestamp after a mutation.

        We deliberately do NOT assert strict change between two rapid calls:
        Windows wall-clock granularity is ~15ms and create + update can land
        on the same microsecond. For ordering, the audit_log's step_index is
        the authoritative source of truth (strictly monotonic by construction).
        """
        ctrl = SessionSubgraphController("sess1", "Q?", "cs4")
        proc = _make_seed_procedure()
        obj_id = ctrl.create_object(proc, {"x": 1}, "create")
        initial = ctrl.subgraph.nodes[obj_id]["provenance"]["last_modified"]
        self.assertNotEqual(initial, "", "create should stamp last_modified")
        ctrl.update_object(obj_id, "state.x", 2, "update")
        after = ctrl.subgraph.nodes[obj_id]["provenance"]["last_modified"]
        self.assertNotEqual(after, "", "update should also stamp last_modified")


class TestDelete(unittest.TestCase):
    def test_delete_field(self):
        ctrl = SessionSubgraphController("sess1", "Q?", "cs4")
        proc = _make_seed_procedure()
        obj_id = ctrl.create_object(proc, {"foo": 1, "bar": 2}, "create")
        ctrl.step()
        ctrl.delete_object(obj_id, "state.foo", "drop foo")
        self.assertNotIn("foo", ctrl.subgraph.nodes[obj_id]["state"])
        self.assertEqual(ctrl.subgraph.nodes[obj_id]["state"]["bar"], 2)

        entry = ctrl.subgraph.audit_log[-1]
        self.assertEqual(entry.operation, "delete")
        self.assertEqual(entry.old_value, 1)

    def test_delete_whole_object(self):
        ctrl = SessionSubgraphController("sess1", "Q?", "cs4")
        proc = _make_seed_procedure()
        obj_id = ctrl.create_object(proc, {"x": 1}, "create")
        ctrl.step()
        ctrl.delete_object(obj_id, "", "drop everything")
        self.assertNotIn(obj_id, ctrl.subgraph.nodes)
        entry = ctrl.subgraph.audit_log[-1]
        self.assertEqual(entry.operation, "delete")
        self.assertEqual(entry.field_path, "")


class TestFailurePatternsAndEdges(unittest.TestCase):
    def test_add_failure_pattern(self):
        ctrl = SessionSubgraphController("sess1", "Q?", "cs4")
        fp = FailurePatternNode(
            id="fp_001",
            name="GreedyFails",
            attempted_approach="greedy",
            failure_condition="negative edges present",
            failure_mechanism="..",
            replacement="proc_bellman",
            example_failure_case=None,
            provenance=Provenance(created_in_session_id="sess1"),
        )
        fp_id = ctrl.add_failure_pattern(fp, "observed failure")
        self.assertEqual(fp_id, "fp_001")
        self.assertIn("fp_001", ctrl.subgraph.nodes)
        self.assertEqual(ctrl.subgraph.audit_log[-1].operation, "create")

    def test_add_edge(self):
        ctrl = SessionSubgraphController("sess1", "Q?", "cs4")
        ctrl.add_edge("so_A", "fp_001", "replaced_by", {"confidence": 0.9})
        self.assertEqual(len(ctrl.subgraph.edges), 1)
        self.assertEqual(ctrl.subgraph.edges[0].relation, "replaced_by")


class TestPersistence(unittest.TestCase):
    def test_persist_and_reload(self):
        ctrl = SessionSubgraphController("sess1", "test query", "cs4")
        proc = _make_seed_procedure()
        obj_id = ctrl.create_object(proc, {"checked": []}, "create")
        ctrl.step()
        ctrl.update_object(obj_id, "state.checked", ["nonneg"], "add nonneg")

        with tempfile.TemporaryDirectory() as td:
            path = ctrl.close(Path(td))
            self.assertTrue((path / "subgraph.json").exists())
            self.assertTrue((path / "audit_log.jsonl").exists())

            # Reload and compare
            with (path / "subgraph.json").open(encoding="utf-8") as f:
                reloaded = SessionSubgraph.from_dict(json.load(f))
            self.assertEqual(reloaded.session_id, "sess1")
            self.assertEqual(reloaded.query, "test query")
            self.assertEqual(reloaded.step_count, 1)
            self.assertIsNotNone(reloaded.ended_at)
            self.assertIn(obj_id, reloaded.nodes)
            self.assertEqual(
                reloaded.nodes[obj_id]["state"]["checked"],
                ["nonneg"],
            )
            # Audit log written separately as JSONL
            jsonl_lines = (path / "audit_log.jsonl").read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(jsonl_lines), 2)  # create + update


class TestReconstructStateViaController(unittest.TestCase):
    """Bridge test: use controller mutations + logger.reconstruct_state.
    This validates the full integration, not just the components in isolation."""

    def test_replay_matches_final_state(self):
        ctrl = SessionSubgraphController("sess1", "Q", "cs4")
        proc = _make_seed_procedure()
        obj_id = ctrl.create_object(proc, {"checked": [], "violated": []}, "create")
        ctrl.step()
        ctrl.update_object(obj_id, "state.checked", ["nonneg"], "add nonneg")
        ctrl.step()
        ctrl.update_object(obj_id, "state.violated", ["nonneg"], "mark violated")
        ctrl.step()
        ctrl.update_object(obj_id, "state.evidence", {"nonneg": "weight -1"}, "evidence")

        final_state = ctrl.subgraph.nodes[obj_id]
        reconstructed = ctrl.logger().reconstruct_state(obj_id, ctrl.step_index)

        # Reconstruction should equal final state byte-for-byte (modulo
        # last_modified, which is a wall-clock timestamp updated on
        # every mutation but not necessarily journaled at sub-second
        # granularity — we compare state field only).
        self.assertEqual(reconstructed["state"], final_state["state"])

    def test_intermediate_states(self):
        ctrl = SessionSubgraphController("sess1", "Q", "cs4")
        proc = _make_seed_procedure()
        obj_id = ctrl.create_object(proc, {"checked": []}, "create")
        ctrl.step()
        ctrl.update_object(obj_id, "state.checked", ["nonneg"], "add nonneg")
        ctrl.step()
        ctrl.update_object(obj_id, "state.checked", ["nonneg", "acyclic"], "add acyclic")

        logger = ctrl.logger()
        s0 = logger.reconstruct_state(obj_id, 0)
        s1 = logger.reconstruct_state(obj_id, 1)
        s2 = logger.reconstruct_state(obj_id, 2)
        self.assertEqual(s0["state"]["checked"], [])
        self.assertEqual(s1["state"]["checked"], ["nonneg"])
        self.assertEqual(s2["state"]["checked"], ["nonneg", "acyclic"])


if __name__ == "__main__":
    unittest.main()
