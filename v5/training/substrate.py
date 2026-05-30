"""Substrate Population Pass: apply V4 scoped-patch substrate into merged_graph.

The Phase 15 corpus collected scoped patches that PROPOSE adding reasoning
substrate (strategy / failure_pattern / control_rule / reasoning_atom /
reasoning_chain / solved_subgoal / epistemic_state) and relations, but they were
never applied to a persisted graph. This module applies the SAFE ones
(accept / soft_only) into a copy of merged_graph, producing a substrate-enriched
graph the bridge can pull planning-pool nodes from.

This closes the data loop:
    V4 traces (with patches)
      -> apply safe substrate patches -> merged_graph_substrate.json
      -> bridge pulls substrate planning nodes into neighborhoods + labels them
      -> planning coverage rises off 0%

    python -m v5.training.substrate      # build the enriched graph + report
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Tuple

from graph_core import MemoryGraph, Node, Edge
from v5.training.dataset import Phase15Dataset, SAFE_PATCH_STATUSES

DEFAULT_BASE = "graphs/merged_graph.json"
DEFAULT_OUT = "graphs/merged_graph_substrate.json"
CORPUS = "artifacts/phase15/phase15_corpus.jsonl"


def _safe_relations(corpus_path: str):
    """Yield (src, dst, relation) for safe add_relation patches (raw add_edge)."""
    with open(corpus_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            for p in row.get("trace", {}).get("scoped_patches", []) or []:
                if not isinstance(p, dict) or p.get("patch_type") != "add_relation":
                    continue
                if (p.get("validation") or {}).get("status") not in SAFE_PATCH_STATUSES:
                    continue
                raw = p.get("raw_edit") or {}
                src, dst, rel = raw.get("src"), raw.get("dst"), raw.get("relation")
                if src and dst and rel:
                    yield src, dst, rel


def build_substrate_graph(
    corpus_path: str = CORPUS,
    base_graph_path: str = DEFAULT_BASE,
    out_path: str = DEFAULT_OUT,
) -> Tuple[MemoryGraph, dict]:
    """Apply safe substrate node + relation patches into a merged_graph copy."""
    graph = MemoryGraph.load_json(base_graph_path)
    base_nodes = len(graph.nodes)
    base_edges = len(graph.edges)

    # 1. add substrate nodes (union across all traces)
    added_by_type: Dict[str, int] = {}
    ds = Phase15Dataset(corpus_path)
    for sample in ds.samples:
        for nid, info in sample.substrate_nodes.items():
            if nid in graph.nodes:
                continue
            graph.nodes[nid] = Node(
                id=nid, text=info.get("text", ""), node_type=info["type"],
                confidence=0.6, importance=0.6,
                metadata={"status": info.get("status", "unknown"), "source": "substrate_pass"},
            )
            added_by_type[info["type"]] = added_by_type.get(info["type"], 0) + 1

    # 2. add safe relations whose endpoints now resolve
    added_edges = 0
    for src, dst, rel in _safe_relations(corpus_path):
        if src in graph.nodes and dst in graph.nodes:
            graph.edges.append(Edge(src=src, dst=dst, relation=rel, strength=0.6))
            added_edges += 1

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    graph.save_json(out_path)

    stats = {
        "base_nodes": base_nodes, "base_edges": base_edges,
        "substrate_nodes_added": sum(added_by_type.values()),
        "added_by_type": added_by_type,
        "relations_added": added_edges,
        "total_nodes": len(graph.nodes), "total_edges": len(graph.edges),
        "out_path": out_path,
    }
    return graph, stats


def run():
    graph, stats = build_substrate_graph()
    print("Substrate Population Pass:")
    print(f"  base: {stats['base_nodes']} nodes, {stats['base_edges']} edges")
    print(f"  substrate nodes added: {stats['substrate_nodes_added']}")
    for t, n in sorted(stats["added_by_type"].items(), key=lambda x: -x[1]):
        print(f"    {t:18s} {n}")
    print(f"  relations added: {stats['relations_added']}")
    print(f"  total: {stats['total_nodes']} nodes, {stats['total_edges']} edges")
    print(f"  saved -> {stats['out_path']}")

    # how many substrate nodes are planning-pool types?
    from v5.subgraph import PLANNING_NODE_TYPES
    plan_added = sum(n for t, n in stats["added_by_type"].items() if t in PLANNING_NODE_TYPES)
    print(f"\n  planning-pool substrate nodes added: {plan_added} "
          f"(strategy/failure_pattern/control_rule/reasoning_atom/reasoning_chain)")
    print("  -> these become labeled planning anchors in the bridge")
    print("\nSUBSTRATE PASS COMPLETE")
    return stats


if __name__ == "__main__":
    run()
