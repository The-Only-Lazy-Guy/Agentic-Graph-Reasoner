from __future__ import annotations

from pathlib import Path
import torch

from project_corpus_to_v5_targets import _default_out
from v5.cross_attention import V5AttentionAdapter
from v5.training.bridge import MockHInitProvider, ZeroEmbedder, sample_to_stage1_example
from v5.training.dataset import SLOT_ID, _parse_row
from v5.training.projection import project_corpus_row
from v5.gnn_encoder import RGCNEncoder
from v5.subgraph import build_active_subgraph
from reasoning.graph_relations import Rel


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


class _Edge:
    def __init__(self, src: str, dst: str, relation: str):
        self.src = src
        self.dst = dst
        self.relation = relation


class _InvalidatorGraph(_Graph):
    def __init__(self):
        super().__init__()
        self.edges = [_Edge("fact_a", "fact_b", Rel.INVALIDATED_BY)]


def test_project_row_builds_architecture_shaped_targets():
    proj = project_corpus_row(_row())

    assert "strat_1" in proj["planning_target"]
    assert "fact_a" in proj["evidence_target"]
    assert "ssg_1" in proj["evidence_target"]
    assert "epis_1" not in proj["evidence_target"]
    assert "fact_a" in proj["support_target"]
    assert "fact_b" in proj["distractor_target"]
    assert "strat_1" in proj["candidate_node_ids"]
    assert proj["evidence_loop_targets"]


def test_epistemic_substrate_preserves_verified_status():
    sample = _parse_row(_row())

    assert sample is not None
    assert sample.substrate_nodes["epis_1"]["status"] == "verified"


def test_bridge_prefers_projected_plan_and_evidence_targets():
    row = _row()
    row["v5_projection"] = project_corpus_row(row)
    sample = _parse_row(row)
    assert sample is not None
    assert sample.task_frame["required_slots"] == ["definition", "reason"]
    assert sample.task_frame["filled_slots"] == ["definition", "reason"]

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

    adapter = V5AttentionAdapter(r_plan=1, r_evidence=1, lm_hidden_dim=128, gate_init=0.02).to(device)
    _, ps, _ = adapter.run_planning(ex.h_init, ex.goal, ex.graph_kv, ex.node_ids, task_frame=ex.task_frame)
    assert ps.write_ratios
    assert ps.write_ratio_tensors
    assert ps.write_ratio_tensors[-1].requires_grad


def test_bridge_masks_projected_targets_to_block_pools():
    row = _row()
    projection = project_corpus_row(row)
    projection["candidate_node_ids"] = ["example_1", "fact_a", "strat_1"]
    projection["planning_target"] = {"fact_a": 99.0, "strat_1": 1.0}
    projection["evidence_target"] = {"example_1": 99.0, "fact_a": 1.0}
    projection["support_target"] = {"example_1": 99.0}
    row["v5_projection"] = projection
    sample = _parse_row(row)
    assert sample is not None

    graph = _Graph()
    graph.nodes["example_1"] = _Node("example", "Context example")

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
        persisted_graph=graph,
        hops=1,
    )

    assert ex is not None
    fact_idx = ex.node_ids.index("fact_a")
    strat_idx = ex.node_ids.index("strat_1")
    example_idx = ex.node_ids.index("example_1")

    assert ex.plan_anchor is not None
    assert ex.plan_anchor[0, strat_idx].item() > 0.0
    assert ex.plan_anchor[0, fact_idx].item() == 0.0

    assert ex.evid_anchor is not None
    assert ex.evid_anchor[0, fact_idx].item() > 0.0
    assert ex.evid_anchor[0, example_idx].item() == 0.0


def test_finalized_required_slots_are_marked_filled_even_if_metric_omits_one():
    row = _row()
    row["metrics"]["finalized"] = True
    row["metrics"]["slot_fill_stats"] = {
        "required_slots": ["relationship", "explanation"],
        "filled_slots": ["relationship"],
    }

    sample = _parse_row(row)

    assert sample is not None
    assert sample.task_frame["required_slots"] == ["definition", "reason"]
    assert sample.task_frame["filled_slots"] == ["definition", "reason"]
    assert sample.slot_fill_target[SLOT_ID["definition"]] == 1.0
    assert sample.slot_fill_target[SLOT_ID["reason"]] == 1.0


def test_blocked_rows_are_marked_for_forced_fallback():
    row = _row()
    row["metrics"]["finalized"] = False

    sample = _parse_row(row)

    assert sample is not None
    assert sample.task_frame["graph_context"] == "weak_evidence"
    assert sample.task_frame["force_fallback"] is True
    assert sample.task_frame["allow_shortcut_exit"] is False


def test_invalidator_candidates_require_well_formed_active_edges():
    device = torch.device("cpu")
    graph = _Graph()
    graph.nodes.update({
        "support_claim": _Node("claim", "Supportive claim"),
        "real_claim": _Node("claim", "Claim with an active condition"),
        "condition": _Node("claim", "Condition node"),
        "missing_condition_claim": _Node("claim", "Claim whose condition is absent"),
        "contradict_src": _Node("claim", "Contradiction source"),
        "contradict_dst": _Node("claim", "Contradiction destination"),
    })
    graph.edges = [
        _Edge("support_claim", "support_claim", Rel.INVALIDATED_BY),
        _Edge("real_claim", "condition", Rel.INVALIDATED_BY),
        _Edge("missing_condition_claim", "missing_condition", Rel.INVALIDATED_BY),
        _Edge("contradict_src", "contradict_dst", Rel.CONTRADICTS),
    ]
    node_ids = [
        "support_claim",
        "real_claim",
        "condition",
        "missing_condition_claim",
        "contradict_src",
        "contradict_dst",
    ]
    text_emb = {nid: [0.0] * 768 for nid in node_ids}
    asg = build_active_subgraph(graph, node_ids, text_emb, device)
    flags = {nid: float(asg.invalidator_flags[i].item()) for i, nid in enumerate(node_ids)}

    assert flags["support_claim"] == 0.0
    assert flags["real_claim"] == 1.0
    assert flags["missing_condition_claim"] == 0.0
    assert flags["contradict_src"] == 1.0
    assert flags["contradict_dst"] == 0.0


def test_bridge_trains_inactive_structural_invalidators_as_zero():
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
        persisted_graph=_InvalidatorGraph(),
        hops=1,
    )
    assert ex is not None
    fact_idx = ex.node_ids.index("fact_a")
    assert ex.struct_inv_mask is not None
    assert ex.inv_target is not None
    assert bool(ex.struct_inv_mask[0, fact_idx].item())
    assert ex.inv_target[0, fact_idx].item() == 0.0


def test_default_out_suffix():
    out = _default_out(Path("data/corpus_merged.jsonl"))
    assert str(out).endswith("data\\corpus_merged_v5proj.jsonl") or str(out).endswith("data/corpus_merged_v5proj.jsonl")
