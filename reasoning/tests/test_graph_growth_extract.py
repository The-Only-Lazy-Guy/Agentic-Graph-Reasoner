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


class _Match:
    def __init__(self, node_id, similarity):
        self.node_id = node_id
        self.similarity = similarity


class _StubIndex:
    """Stub DedupeIndex: returns a fixed classification for every query."""
    def __init__(self, kind, node_id="existing_node", sim=0.85):
        self.kind, self.node_id, self.sim = kind, node_id, sim

    def classify(self, text, dup_threshold=0.92, ambiguous_threshold=0.80):
        if self.kind == "novel":
            return "novel", None
        return self.kind, _Match(self.node_id, self.sim)


def test_link_ambiguous_keeps_node_and_adds_attach_edge():
    docs = [Document(id="d", text="p", domain="physics", mode="fact")]   # 1 chunk
    result = extract_documents(docs, extract_fn=_stub, dedupe_index=_StubIndex("ambiguous"))
    assert result["stats"]["nodes"] == 2                  # both kept
    assert result["stats"]["attached_existing"] == 2      # one attach edge each
    attach = [c for c in result["candidates"]
              if c["raw_edit"]["op"] == "add_edge" and c["raw_edit"]["metadata"].get("kb_link") == "ambiguous"]
    assert len(attach) == 2
    assert all(a["raw_edit"]["dst"] == "existing_node" for a in attach)


def test_link_duplicate_drops_node_and_remaps():
    docs = [Document(id="d", text="p", domain="physics", mode="fact")]
    result = extract_documents(docs, extract_fn=_stub, dedupe_index=_StubIndex("duplicate"))
    assert result["stats"]["nodes"] == 0                  # both dropped as duplicates
    assert result["stats"]["deduped_existing"] == 2
    # the intra-batch edge survives, remapped onto the existing node
    edges = [c for c in result["candidates"] if c["raw_edit"]["op"] == "add_edge"]
    assert edges and all(e["raw_edit"]["dst"] == "existing_node" for e in edges)


def test_judge_merge_into_remaps_and_drops():
    def judge_fn(ne):
        if "scalar" in ne["text"].lower():
            return ("merge_into", "existing_scalar")
        return ("accept", None)
    docs = [Document(id="d", text="p", domain="physics", mode="fact")]
    result = extract_documents(docs, extract_fn=_stub, judge_fn=judge_fn)
    assert result["stats"]["merged_existing"] == 1
    assert result["stats"]["nodes"] == 1                  # scalar merged out, vector kept


def test_collect_sft_builds_chat_pairs():
    docs = [Document(id="d", text="p", domain="physics", mode="fact")]   # 1 chunk
    result = extract_documents(docs, extract_fn=_stub, collect_sft=True)
    pairs = result["sft_pairs"]
    assert len(pairs) == 1
    msgs = pairs[0]["messages"]
    assert [m["role"] for m in msgs] == ["system", "user", "assistant"]
    target = json.loads(msgs[2]["content"])
    assert len(target["nodes"]) == 2 and target["nodes"][0]["id"] == "n0"
    assert target["edges"][0]["relation"] == "contradicts"     # remapped to local ids
    assert {e["src"] for e in target["edges"]} <= {"n0", "n1"}
    assert pairs[0]["meta"]["n_nodes"] == 2


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
