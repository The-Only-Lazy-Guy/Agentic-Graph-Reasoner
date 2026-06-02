from __future__ import annotations

import json
from pathlib import Path

import pytest

from graph_core import MemoryGraph, Node
from v5.graph_grower.apply import apply_growth


def _write_graph(path: Path) -> None:
    MemoryGraph(
        nodes={
            "support_node": Node(
                id="support_node",
                node_type="claim",
                text="Evidence supports the candidate graph edit.",
                confidence=0.95,
            )
        },
        edges=[],
    ).save_json(str(path))


def _substrate_patch(patch_id, raw_edit):
    return {
        "patch_id": patch_id,
        "patch_type": "add_strategy" if raw_edit["op"] == "add_node" else "add_relation",
        "target_id": raw_edit.get("node_id") or raw_edit.get("dst") or "candidate",
        "text": raw_edit.get("text", ""),
        "risk_level": "medium",
        "evidence_node_ids": ["support_node"],
        "raw_edit": raw_edit,
        "validation": {"status": "accept", "support_score": 0.9,
                       "reasons": [], "warnings": [], "duplicate_of": None, "conflicts_with": []},
    }


def _write_corpus(path: Path) -> None:
    patches = [
        _substrate_patch("p_node", {
            "op": "add_node", "node_id": "strategy_candidate", "node_type": "strategy",
            "text": "Use a verify-then-answer control recipe.", "metadata": {},
        }),
        # edge into the base graph -> resolvable, kept
        _substrate_patch("p_edge_ok", {
            "op": "add_edge", "src": "strategy_candidate", "dst": "support_node",
            "relation": "leveraged", "tier": "add", "metadata": {},
        }),
        # edge to a node that exists nowhere -> dangling, must be dropped
        _substrate_patch("p_edge_dangling", {
            "op": "add_edge", "src": "strategy_candidate", "dst": "ghost_hypothesis_h_1",
            "relation": "leveraged", "tier": "add", "metadata": {},
        }),
    ]
    row = {
        "session_id": "sess_test",
        "input": {"question": "grow?"},
        "quality": {"training_eligible": True, "v5_label_status": "positive", "finalized": True},
        "metrics": {"finalized": True, "controller_task_family": "direct_judgment",
                    "controller_fallback_used": False,
                    "slot_fill_stats": {"required_slots": ["answer"], "filled_slots": ["answer"],
                                        "filled_count": 1, "required_count": 1},
                    "scoped_patch_summary": {"patch_count": len(patches), "needs_attention": 0}},
        "trace": {"scoped_patches": patches},
    }
    path.write_text(json.dumps(row) + "\n", encoding="utf-8")


def test_refuses_to_overwrite_base_graph(tmp_path):
    graph_path = tmp_path / "g.json"
    corpus_path = tmp_path / "c.jsonl"
    _write_graph(graph_path)
    _write_corpus(corpus_path)
    with pytest.raises(ValueError):
        apply_growth(corpus_path=corpus_path, graph_path=graph_path, out_path=graph_path)


def test_substrate_apply_adds_node_drops_dangling_edge_and_stamps_provenance(tmp_path):
    graph_path = tmp_path / "g.json"
    corpus_path = tmp_path / "c.jsonl"
    out_path = tmp_path / "grown.json"
    _write_graph(graph_path)
    _write_corpus(corpus_path)

    result = apply_growth(
        corpus_path=corpus_path, graph_path=graph_path, out_path=out_path, lanes=["substrate"],
    )

    assert result["edit_stats"]["node_edits"] == 1
    assert result["edit_stats"]["edge_edits"] == 1          # only the resolvable one
    assert result["edit_stats"]["dropped_dangling_edges"] == 1
    assert result["gate_passed"] is True
    assert result["persisted"] is True

    grown = MemoryGraph.load_json(str(out_path))
    assert "strategy_candidate" in grown.nodes
    node = grown.nodes["strategy_candidate"]
    assert node.metadata.get("auto_grown") is True
    assert node.metadata.get("grow_lane") == "substrate"
    # base graph untouched
    base = MemoryGraph.load_json(str(graph_path))
    assert "strategy_candidate" not in base.nodes


def test_dry_run_does_not_persist(tmp_path):
    graph_path = tmp_path / "g.json"
    corpus_path = tmp_path / "c.jsonl"
    out_path = tmp_path / "grown.json"
    _write_graph(graph_path)
    _write_corpus(corpus_path)

    result = apply_growth(
        corpus_path=corpus_path, graph_path=graph_path, out_path=out_path,
        lanes=["substrate"], dry_run=True,
    )
    assert result["persisted"] is False
    assert not out_path.exists()
