"""Bridge: the model's graph edits -> graph_grower.apply_candidates on graph_core.MemoryGraph.

The algorithm-graph work was reinventing node-creation on v5/memory.TotalMemory (flat, no edges, no
health gate). The MATURE mechanism already exists: graph_grower audits/health-gates session-proposed
add_node/add_edge edits onto MemoryGraph (typed nodes incl. `implementation`/`worked_example`, typed
edges, provenance). This routes the MODEL's chosen edits through it — so we get EDGES (composition:
`solution derived_from atom`, `atom part_of concept`) and a HEALTH GATE (refuses degrading edits)
for free, instead of the flat 561-node/0-edge/1-concept store.

  node = {implementation, text=purpose, metadata={code}}    edge = {src, dst, relation}
  model STORE  -> add_node candidate     model COMPOSE -> add_edge candidate
  -> graph_grower.apply_candidates (health-gated, non-destructive) -> grown MemoryGraph

  selftest (no model):  python -m v5.runtime.algo_graph_edits --selftest
"""
from __future__ import annotations

import argparse
import sys


def node_candidate(node_id: str, code: str, purpose: str, session_id: str,
                   node_type: str = "implementation", support: float = 1.0) -> dict:
    """A model-chosen atom -> an add_node candidate (code rides in metadata; the model's purpose is
    the node text = the retrieval key)."""
    return {"raw_edit": {"op": "add_node", "node_id": node_id, "node_type": node_type,
                         "text": purpose, "metadata": {"code": code}, "tier": "add"},
            "lane": "substrate", "session_id": session_id, "patch_id": node_id,
            "target_id": node_id, "support_score": support}


def edge_candidate(src: str, dst: str, relation: str, session_id: str, support: float = 1.0) -> dict:
    """A composition -> an add_edge candidate (e.g. `sol derived_from build_adj`, `build_adj part_of
    graph_algorithms`). Endpoints must resolve in the base graph or the same batch or it's dropped."""
    return {"raw_edit": {"op": "add_edge", "src": src, "dst": dst, "relation": relation,
                         "metadata": {}, "tier": "add"},
            "lane": "substrate", "session_id": session_id, "patch_id": f"{src}->{dst}::{relation}",
            "support_score": support}


def propose_edits(session_id: str, stores: list[tuple], edges: list[tuple]) -> list[dict]:
    """Build the candidate batch from the model's choices.
      stores: [(node_id, code, purpose)]  -- atoms the model chose to STORE (already verified)
      edges:  [(src, dst, relation)]       -- composition edges (derived_from / part_of / depend)"""
    cands = [node_candidate(nid, code, purpose, session_id) for nid, code, purpose in stores]
    cands += [edge_candidate(s, d, r, session_id) for s, d, r in edges]
    return cands


def grow(graph_path: str, out_path: str, candidates: list[dict],
         degradation_threshold: float = -0.02, dry_run: bool = False) -> dict:
    """Health-gated, non-destructive apply of the model's edits onto MemoryGraph (out_path != graph_path)."""
    from v5.graph_grower.apply import apply_candidates
    return apply_candidates(candidates, graph_path=graph_path, out_path=out_path,
                            degradation_threshold=degradation_threshold, dry_run=dry_run)


# ═══════════════════════════════════════════════════════════════════════════════
# SELFTEST (no model) — model edits -> candidates -> graph_grower apply on MemoryGraph
# ═══════════════════════════════════════════════════════════════════════════════

def _selftest() -> bool:
    import json
    import tempfile
    from pathlib import Path
    print("algo_graph_edits --selftest: model edits -> graph_grower.apply_candidates (MemoryGraph)\n")
    with tempfile.TemporaryDirectory() as td:
        base = Path(td) / "base.json"
        base.write_text(json.dumps({"metadata": {}, "edges": [], "nodes": [
            {"id": "concept_graph_algos", "text": "graph algorithms", "node_type": "concept"}]}))
        out = Path(td) / "grown.json"

        # model chose to STORE build_adj, and composed it: build_adj part_of the concept
        cands = propose_edits(
            "sess1",
            stores=[("impl_build_adj", "def build_adj(n, edges):\n    ...", "adjacency list from edges")],
            edges=[("impl_build_adj", "concept_graph_algos", "part_of")])
        res = grow(str(base), str(out), cands)
        assert res["edit_stats"]["node_edits"] == 1 and res["edit_stats"]["edge_edits"] == 1, res
        assert res["graph_after"] == {"nodes": 2, "edges": 1}, res
        assert res["gate_passed"] and res["persisted"], res
        print(f"  [1] add_node(implementation)+add_edge(part_of) -> 2 nodes / 1 EDGE, "
              f"health {res['health_before']}->{res['health_after']} gate PASS -> PASS")

        # the grown graph has the atom with its code in metadata + the composition edge
        from graph_core import MemoryGraph
        g = MemoryGraph.load_json(str(out))
        n = g.nodes["impl_build_adj"]
        assert n.node_type == "implementation" and "build_adj" in n.metadata.get("code", "")
        assert g.edge_between("impl_build_adj", "concept_graph_algos") is not None
        print("  [2] grown MemoryGraph: atom node carries code in metadata + composition edge -> PASS")

        # dangling edge (endpoint missing) is dropped by the grower, not persisted as junk
        c2 = propose_edits("s2", stores=[], edges=[("impl_build_adj", "ghost_node", "depend")])
        r2 = grow(str(out), str(Path(td) / "g2.json"), c2)
        assert r2["edit_stats"]["edge_edits"] == 0 and r2["edit_stats"]["dropped_dangling_edges"] == 1, r2
        print("  [3] dangling edge (missing endpoint) dropped by the grower -> PASS")

    print("\n  ALGO_GRAPH_EDITS SELFTEST -> PASS")
    return True


def main():
    ap = argparse.ArgumentParser(description="Route model graph edits through graph_grower (MemoryGraph).")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        sys.exit(0 if _selftest() else 1)
    ap.print_help()


if __name__ == "__main__":
    main()
