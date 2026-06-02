from __future__ import annotations

import json

from graph_core import MemoryGraph, Node
from v5.graph_grower.apply import apply_candidates
from v5.graph_grower.extract import (
    Document,
    conform_edits,
    extract_documents,
    parse_extraction,
)


def test_conform_drops_non_atomic_and_falls_back_vocab():
    doc = Document(id="d1", text="x", domain="physics", mode="fact")
    parsed = {
        "nodes": [
            {"id": "a", "node_type": "fact", "text": "A scalar has only magnitude."},
            {"id": "b", "node_type": "fact", "text": "x" * 600},               # non-atomic
            {"id": "c", "node_type": "fact", "text": "no"},                     # too short
            {"id": "d", "node_type": "weird_type", "text": "A vector has magnitude and direction."},
        ],
        "edges": [
            {"src": "a", "dst": "d", "relation": "contrasts_with"},   # unknown -> related
            {"src": "a", "dst": "a", "relation": "related"},          # self-loop dropped
        ],
    }
    out = conform_edits(parsed, doc)
    assert len(out["node_edits"]) == 2                       # a and d kept
    assert out["dropped"]["non_atomic"] == 1
    assert out["dropped"]["empty"] == 1
    # bad node_type fell back to the mode default
    assert all(ne["node_type"] in {"fact", "concept", "claim"} for ne in out["node_edits"])
    assert len(out["edge_edits"]) == 1                       # self-loop dropped
    assert out["edge_edits"][0]["relation"] == "related"     # unknown relation -> fallback


def test_concept_alias_maps_to_design_type():
    doc = Document(id="d", text="x", domain="physics", mode="fact")
    parsed = {"nodes": [{"id": "a", "node_type": "concept",
                         "text": "A scalar is described by magnitude alone."}], "edges": []}
    out = conform_edits(parsed, doc)
    assert out["node_edits"][0]["node_type"] == "claim"   # concept -> claim (a design type)


def test_extractor_node_types_match_graph_design():
    """Drift guard: every type the extractor can emit must (a) be in the GNN
    node-type vocab and (b) fall in a cross-attention pool, else the grown node
    is invisible to the model."""
    from v5.gnn_encoder import NODE_TYPE_VOCAB
    from v5.subgraph import PLANNING_NODE_TYPES, EVIDENCE_NODE_TYPES
    from v5.graph_grower.extract import ALLOWED_NODE_TYPES, NODE_TYPE_ALIASES

    pooled = set(PLANNING_NODE_TYPES) | set(EVIDENCE_NODE_TYPES)
    for t in ALLOWED_NODE_TYPES:
        assert t in NODE_TYPE_VOCAB, f"{t} not in GNN NODE_TYPE_VOCAB"
        assert t in pooled, f"{t} is in no planning/evidence pool -> never attended"
    for target in NODE_TYPE_ALIASES.values():
        assert target in ALLOWED_NODE_TYPES, f"alias target {target} not emittable"


def test_parse_extraction_tolerates_code_fence_and_prose():
    raw = 'Sure!\n```json\n{"nodes":[{"node_type":"fact","text":"Water boils at 100C at sea level."}],"edges":[]}\n```'
    parsed = parse_extraction(raw)
    assert len(parsed["nodes"]) == 1


def _stub(chunk: str, mode: str) -> str:
    return json.dumps({
        "nodes": [
            {"id": "n1", "node_type": "fact", "text": "A scalar quantity has only magnitude."},
            {"id": "n2", "node_type": "fact", "text": "A vector quantity has magnitude and direction."},
        ],
        "edges": [{"src": "n1", "dst": "n2", "relation": "contradicts"}],
    })


def test_extract_documents_emits_apply_compatible_candidates():
    docs = [Document(id="scalar_vector", text="para1\n\npara2", domain="physics", mode="fact")]
    result = extract_documents(docs, extract_fn=_stub)
    cands = result["candidates"]
    # two chunks x (2 nodes + 1 edge) = 6
    assert result["stats"]["nodes"] == 4
    assert result["stats"]["edges"] == 2
    node_c = next(c for c in cands if c["raw_edit"]["op"] == "add_node")
    assert node_c["lane"] == "external"
    assert node_c["raw_edit"]["node_id"].startswith("kb_fact_")
    assert "raw_edit" in node_c and "patch_id" in node_c


def test_extract_then_apply_grows_graph(tmp_path):
    graph_path = tmp_path / "g.json"
    out_path = tmp_path / "grown.json"
    MemoryGraph(nodes={"seed": Node(id="seed", node_type="claim", text="seed", confidence=0.9)},
                edges=[]).save_json(str(graph_path))

    docs = [Document(id="sv", text="p", domain="physics", mode="fact")]
    result = extract_documents(docs, extract_fn=_stub)
    applied = apply_candidates(result["candidates"], graph_path=graph_path, out_path=out_path)

    assert applied["persisted"] is True
    grown = MemoryGraph.load_json(str(out_path))
    assert len(grown.nodes) == 3        # seed + 2 atomic facts (dedup by stable id across chunks)
    kb = [n for n in grown.nodes.values() if n.id.startswith("kb_fact_")]
    assert kb and all(n.metadata.get("auto_grown") for n in kb)
