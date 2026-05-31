from __future__ import annotations

from pathlib import Path
import torch

from project_corpus_to_v5_targets import _default_out
from v5.training.bridge import MockHInitProvider, ZeroEmbedder, sample_to_stage1_example
from v5.training.dataset import _parse_row
from v5.training.projection import project_corpus_row
from v5.gnn_encoder import RGCNEncoder


def _row() -> dict:
    return {
        "session_id": "sess_1",
        "input": {
            "question": "What is the relation between X and Y?",
            "anchors": [
                {"id": "fact_a", "node_type": "fact", "text": "Fact A"},
                {"id": "fact_b", "node_type": "fact", "text": "Fact B"},
            ],
            "task_type": "direct_relationship",
            "controller_task_family": "relational_explanation",
        },
        "trace": {
            "tool_calls": [
                {"name": "read_node", "args": {"node_id": "fact_a"}},
            ],
            "micro_steps": [
                {
                    "index": 1,
                    "subgoal": "identify_relationship",
                    "subgoal_signature": "rel.rel.identify",
                    "action": "REUSE",
                    "sufficient": True,
                    "filled_slots": ["relationship"],
                    "missing_slots": ["explanation"],
                    "evidence_node_ids": ["fact_a"],
                    "matched_node_id": None,
                    "detail": "reuse stored fact",
                },
                {
                    "index": 2,
                    "subgoal": "finalize_answer",
                    "subgoal_signature": "rel.rel.finalize",
                    "action": "FINALIZE",
                    "sufficient": True,
                    "filled_slots": ["relationship", "explanation"],
                    "missing_slots": [],
                    "evidence_node_ids": ["fact_a"],
                    "matched_node_id": None,
                    "detail": "finalized on stored support",
                },
            ],
            "scoped_patches": [
                {
                    "patch_type": "add_strategy",
                    "target_id": "strat_1",
                    "text": "strategy",
                    "raw_edit": {"node_id": "strat_1"},
                    "validation": {"status": "accept"},
                    "evidence_node_ids": ["fact_a"],
                },
                {
                    "patch_type": "add_solved_subgoal",
                    "target_id": "ssg_1",
                    "text": "solved",
                    "raw_edit": {"node_id": "ssg_1"},
                    "metadata": {"supporting_node_ids": ["fact_a"]},
                    "validation": {"status": "accept"},
                    "evidence_node_ids": ["fact_a"],
                },
                {
                    "patch_type": "add_epistemic_state",
                    "target_id": "epis_1",
                    "text": "verified epi",
                    "raw_edit": {
                        "node_id": "epis_1",
                        "metadata": {"target_node_id": "fact_a", "status": "verified"},
                    },
                    "metadata": {
                        "target_node_id": "fact_a",
                        "status": "verified",
                        "evidence_node_ids": ["fact_a"],
                    },
                    "validation": {"status": "accept"},
                    "evidence_node_ids": ["fact_a"],
                },
            ],
        },
        "metrics": {
            "finalized": True,
            "steps": 1,
            "max_steps": 4,
            "shortcut_anchor_ids": ["fact_a"],
            "slot_fill_stats": {
                "required_slots": ["relationship", "explanation"],
                "filled_slots": ["relationship", "explanation"],
            },
        },
        "v5_trajectory": {"nodes_accessed_log": []},
    }


class _Node:
    def __init__(self, node_type: str, text: str, status: str = "unknown"):
        self.node_type = node_type
        self.text = text
        self.confidence = 0.7
        self.metadata = {"status": status}


class _Graph:
    def __init__(self):
        self.nodes = {
            "fact_a": _Node("fact", "Fact A"),
            "fact_b": _Node("fact", "Fact B"),
            "strat_1": _Node("strategy", "strategy"),
            "ssg_1": _Node("solved_subgoal", "solved"),
            "epis_1": _Node("epistemic_state", "verified epi", status="verified"),
        }
        self.edges = []


def test_project_row_builds_architecture_shaped_targets():
    proj = project_corpus_row(_row())

    assert "strat_1" in proj["planning_target"]
    assert "fact_a" in proj["evidence_target"]
    assert "ssg_1" in proj["evidence_target"]
    assert "fact_a" in proj["support_target"]
    assert "fact_b" in proj["distractor_target"]
    assert "strat_1" in proj["candidate_node_ids"]
    assert proj["evidence_loop_targets"]


def test_bridge_prefers_projected_plan_and_evidence_targets():
    row = _row()
    row["v5_projection"] = project_corpus_row(row)
    sample = _parse_row(row)
    assert sample is not None

    device = torch.device("cpu")
    gnn = RGCNEncoder().to(device).eval()
    for p in gnn.parameters():
        p.requires_grad_(False)
    ex = sample_to_stage1_example(
        sample,
        gnn=gnn,
        embedder=ZeroEmbedder(device),
        h_init_provider=MockHInitProvider(128, device),
        device=device,
        lm_dim=128,
        persisted_graph=_Graph(),
        hops=1,
    )
    assert ex is not None
    assert "strat_1" in ex.node_ids
    assert ex.plan_anchor is not None
    assert ex.evid_anchor is not None

    strat_idx = ex.node_ids.index("strat_1")
    fact_idx = ex.node_ids.index("fact_a")
    assert ex.plan_anchor[0, strat_idx].item() > 0.0
    assert ex.evid_anchor[0, fact_idx].item() > 0.0


def test_default_out_suffix():
    out = _default_out(Path("data/corpus_merged.jsonl"))
    assert str(out).endswith("data\\corpus_merged_v5proj.jsonl") or str(out).endswith("data/corpus_merged_v5proj.jsonl")
