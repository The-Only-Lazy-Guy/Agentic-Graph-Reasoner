"""Retroactively apply AtomGraph._self_organize's guaranteed-connectivity fix to an EXISTING saved graph.
Loading a graph does NOT re-run self-organize on nodes that were already isolated when saved -- this
backfills them without a full retrain (no GPU, no training loop, just re-linking by embedding similarity
already computed at load time).

    python -m scripts.backfill_connectivity --graph-path graphs/long_term.json
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = str(Path(__file__).resolve().parents[1])
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)


def backfill(graph_path: str) -> None:
    from v5.runtime.membrane import AtomGraph

    g = AtomGraph.load(graph_path)
    degree: dict = {n: 0 for n in g.atoms}
    for s, d, _ in g.edges:
        degree[s] += 1
        degree[d] += 1
    isolated_before = [n for n in g.atoms if degree[n] == 0]
    print(f"graph: {graph_path}  nodes: {len(g)}  edges: {len(g.edges)}")
    print(f"isolated before backfill: {len(isolated_before)}/{len(g)}")

    edges_before = len(g.edges)
    for name in isolated_before:
        g._self_organize(g.atoms[name])

    degree2: dict = {n: 0 for n in g.atoms}
    for s, d, _ in g.edges:
        degree2[s] += 1
        degree2[d] += 1
    isolated_after = [n for n in g.atoms if degree2[n] == 0]
    print(f"isolated after backfill:  {len(isolated_after)}/{len(g)}")
    print(f"edges added: {len(g.edges) - edges_before}  (total now: {len(g.edges)})")

    g.save(graph_path)
    print(f"saved -> {graph_path}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Backfill guaranteed-connectivity fix onto an existing graph.")
    ap.add_argument("--graph-path", type=str, default="graphs/long_term.json")
    args = ap.parse_args(argv)
    backfill(args.graph_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
