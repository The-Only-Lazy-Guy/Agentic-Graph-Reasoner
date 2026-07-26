"""Retroactively apply AtomGraph._self_organize's guaranteed-connectivity fix to an EXISTING saved graph.

NOTE: AtomGraph.load() now calls repair_connectivity() automatically (auto_repair=True by default) -- any
future run_real invocation with --graph-path already self-heals on load, no manual step needed. This script
still exists to (a) report an honest before/after on a graph you haven't loaded through the fixed code yet,
and (b) persist the repair to disk immediately without waiting for a full training run's final g.save().

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

    g = AtomGraph.load(graph_path, auto_repair=False)     # raw, so "before" is the true saved-on-disk state
    edges_before = len(g.edges)
    print(f"graph: {graph_path}  nodes: {len(g)}  edges: {edges_before}")

    n_isolated = g.repair_connectivity()
    print(f"isolated before backfill: {n_isolated}/{len(g)}")

    degree: dict = {n: 0 for n in g.atoms}
    for s, d, _ in g.edges:
        degree[s] += 1
        degree[d] += 1
    isolated_after = [n for n in g.atoms if degree[n] == 0]
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
