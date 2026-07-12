"""GRR-5: the SLEEP pass — the offline library-learning loop. While the wake loop solves under budget,
sleep periodically reshapes the LIBRARY to maximize future capability:

  compress  apply gated ABSTRACTIONS (GRR-4): mint the higher-order skeleton, rewrite each leaf as a
            one-line instance, add `abstracts` edges. Gated by fuzz-EQUIVALENCE (behavior preserved,
            not just seen cases) + MDL (the library shrinks). This is where IRREPLACEABLE atoms enter.
  prune     retire dead weight: atoms whose Δcapability ~ 0 (used-but-replaceable, or out-of-
            distribution) are weakened/retired. Soft — a different budget/distribution can revive them.

The objective is corpus-level so it can't be an online reward; it's this periodic pass. compress mints
what GRR-3 rated load-bearing; prune drops what it rated worthless. Together the library compounds
toward a smaller, more general, higher-capability basis.

  selftest (no model):  python -m v5.runtime.algo_sleep --selftest
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from graph_core import MemoryGraph
from v5.runtime.algo_abstract import verify_abstraction


def sleep_compress(graph_path: str, out_path: str, abstractions) -> dict:
    """Apply each abstraction that passes the gate (fuzz-equivalence + MDL) to the graph: add the
    skeleton node, rewrite each instance leaf to call it, add `abstracts` edges. abstractions =
    [(skeleton_name, skeleton_code, {instance_name: instance_code}, {instance_name: oracle})]."""
    g = json.loads(Path(graph_path).read_text(encoding="utf-8"))
    by_id = {n["id"]: n for n in g["nodes"]}
    applied, rejected = [], []
    for sk_name, sk_code, instances, oracles in abstractions:
        res = verify_abstraction(sk_code, instances, oracles)
        if not (res["equivalent"] and res["compresses"]):          # equivalence VETOES; MDL must win
            rejected.append((sk_name, res))
            continue
        skid = f"impl_{sk_name}"
        if skid not in by_id:                                      # mint the skeleton
            node = {"id": skid, "text": f"higher-order abstraction: {sk_name}",
                    "node_type": "implementation", "metadata": {"code": sk_code, "kind": "abstraction"}}
            g["nodes"].append(node); by_id[skid] = node
        for inst_name, inst_code in instances.items():             # rewrite leaves + link
            iid = f"impl_{inst_name}"
            if iid in by_id:
                by_id[iid]["metadata"]["code"] = inst_code
                g["edges"].append({"src": skid, "dst": iid, "relation": "abstracts",
                                   "strength": 0.7, "directed": True})
        applied.append(sk_name)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(json.dumps(g), encoding="utf-8")
    return dict(applied=applied, rejected=[r[0] for r in rejected],
                nodes=len(g["nodes"]), edges=len(g["edges"]))


def sleep_prune(ranking, thresh: float = 0.02) -> dict:
    """Given a Δcapability ranking (from algo_capability.rank_atoms), split into keepers (delta>=thresh)
    and prune-bait (delta<thresh = dead weight). Soft: caller weakens/retires, doesn't hard-delete."""
    keep = [r["atom"] for r in ranking if r["delta"] >= thresh]
    prune = [r["atom"] for r in ranking if r["delta"] < thresh]
    return dict(keep=keep, prune=prune)


# ═══════════════════════════════════════════════════════════════════════════════
# SELFTEST — sleep compresses a graph: mints str_dp2, rewrites the leaves, adds abstracts edges,
# behavior preserved; prune splits a Δ-ranking into keepers vs dead weight.
# ═══════════════════════════════════════════════════════════════════════════════

def _selftest() -> bool:
    import tempfile
    from v5.runtime.algo_abstract import STR_DP2, DP2_INSTANCES, _DP2_ORACLE, fuzz_equivalent
    from v5.runtime.algo_compose_tasks import ALL_ATOMS
    print("algo_sleep --selftest: compress (mint skeleton + rewrite leaves + edges) + prune dead weight\n")

    with tempfile.TemporaryDirectory() as td:
        gp = str(Path(td) / "g.json")
        nodes = [{"id": "concept_algorithms", "text": "algorithms", "node_type": "concept"}]
        for a in ("edit_distance", "lcs_length"):                  # the two leaves, FULL DP form
            nodes.append({"id": f"impl_{a}", "text": ALL_ATOMS[a][0], "node_type": "implementation",
                          "metadata": {"code": ALL_ATOMS[a][1]}})
        Path(gp).write_text(json.dumps({"metadata": {}, "nodes": nodes, "edges": []}))
        n0 = len(nodes)

        out = str(Path(td) / "grown.json")
        abstractions = [("str_dp2", STR_DP2, DP2_INSTANCES, _DP2_ORACLE)]
        r = sleep_compress(gp, out, abstractions)
        assert r["applied"] == ["str_dp2"] and r["nodes"] == n0 + 1, r
        print(f"  [1] compress: gate PASS -> minted impl_str_dp2 ({n0}->{r['nodes']} nodes, "
              f"{r['edges']} abstracts edges) -> PASS")

        g = MemoryGraph.load_json(out)
        assert "impl_str_dp2" in g.nodes and g.nodes["impl_str_dp2"].metadata.get("kind") == "abstraction"
        assert g.edge_between("impl_str_dp2", "impl_edit_distance") is not None
        # the rewritten leaf is now the 1-line instance AND still behaves (with the skeleton dep)
        ed_code = g.nodes["impl_edit_distance"].metadata["code"]
        assert "str_dp2(" in ed_code and len(ed_code) < len(ALL_ATOMS["edit_distance"][1])
        assert fuzz_equivalent(ed_code, g.nodes["impl_str_dp2"].metadata["code"], "edit_distance",
                               _DP2_ORACLE["edit_distance"], n=60)
        print(f"  [2] leaves rewritten to instances (edit_distance now {len(ed_code)} chars, calls "
              f"str_dp2) + still fuzz-equivalent + `abstracts` edges -> PASS")

        # a lossy abstraction (wrong on_match) must be VETOED by the equivalence gate
        bad_inst = {"edit_distance": "def edit_distance(a, b):\n    return str_dp2(a, b, lambda i: i, "
                    "lambda j: j, lambda d, u, l: d + 1, lambda d, u, l: 1 + min(d, u, l))"}  # +1 bug
        rbad = sleep_compress(gp, str(Path(td) / "x.json"),
                              [("str_dp2", STR_DP2, bad_inst, {"edit_distance": _DP2_ORACLE["edit_distance"]})])
        assert rbad["applied"] == [] and "str_dp2" in rbad["rejected"], rbad
        print("  [3] lossy rewrite (wrong on_match) -> fuzz-equivalence VETOES it (not applied) -> PASS")

        # prune: split a delta ranking
        ranking = [{"atom": "impl_edit_distance", "delta": 0.12}, {"atom": "impl_lis_length", "delta": 0.0}]
        pr = sleep_prune(ranking)
        assert pr["keep"] == ["impl_edit_distance"] and pr["prune"] == ["impl_lis_length"], pr
        print(f"  [4] prune: keep {pr['keep']} (load-bearing), prune {pr['prune']} (dead weight) -> PASS")

    print("\n  ALGO_SLEEP SELFTEST -> PASS")
    return True


def main():
    ap = argparse.ArgumentParser(description="GRR-5: sleep pass (gated compression + prune).")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        sys.exit(0 if _selftest() else 1)
    ap.print_help()


if __name__ == "__main__":
    main()
