"""Real connectivity report + visualization for a membrane.py AtomGraph -- checks the actual structure of
the long-term graph directly, instead of trusting a bare "N nodes, M edges" print. Written because a real
training run's fallback paths (record_failure's trap-node creation especially) can plausibly flood a graph
with many low-value, weakly-connected nodes -- this reports whether that's actually happening, with real
numbers and a real picture, not a guess.

    python -m scripts.graph_connectivity_report --graph-path graphs/long_term.json
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

_ROOT = str(Path(__file__).resolve().parents[1])
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)


def report(graph_path: str, out_png: str | None, top_n: int = 15) -> None:
    from v5.runtime.membrane import AtomGraph

    g = AtomGraph.load(graph_path)
    n_atoms = len(g)
    edges = g.edges

    print(f"graph: {graph_path}")
    print(f"  nodes: {n_atoms}   edges: {len(edges)}")

    kinds = Counter(a.kind for a in g.atoms.values())
    print(f"  kind breakdown: {dict(kinds)}")

    rel_counts = Counter(r for _, _, r in edges)
    print(f"  edge relation breakdown: {dict(rel_counts)}")

    # degree (undirected -- an edge (s,d,r) counts toward both s and d)
    degree: Counter = Counter()
    for s, d, _ in edges:
        degree[s] += 1
        degree[d] += 1
    isolated = [n for n in g.atoms if degree[n] == 0]
    degrees = [degree[n] for n in g.atoms]
    avg_deg = sum(degrees) / max(1, len(degrees))
    print(f"\n  isolated nodes (degree 0): {len(isolated)}/{n_atoms}  ({100*len(isolated)/max(1,n_atoms):.0f}%)")
    print(f"  average degree: {avg_deg:.2f}")
    print(f"  max degree: {max(degrees) if degrees else 0}")
    if isolated:
        sample = isolated[:top_n]
        print(f"  sample isolated nodes: {sample}")

    # connected components (undirected, ignores edge direction/relation)
    adj: dict[str, set] = {n: set() for n in g.atoms}
    for s, d, _ in edges:
        if s in adj and d in adj:
            adj[s].add(d)
            adj[d].add(s)
    seen: set = set()
    components: list[list[str]] = []
    for start in g.atoms:
        if start in seen:
            continue
        stack, comp = [start], []
        seen.add(start)
        while stack:
            u = stack.pop()
            comp.append(u)
            for v in adj[u]:
                if v not in seen:
                    seen.add(v)
                    stack.append(v)
        components.append(comp)
    components.sort(key=len, reverse=True)
    print(f"\n  connected components: {len(components)}")
    print(f"  largest component: {len(components[0]) if components else 0} nodes "
          f"({100*len(components[0])/max(1,n_atoms):.0f}% of the graph)" if components else "")
    sizes = Counter(len(c) for c in components)
    print(f"  component size distribution: {dict(sorted(sizes.items()))}")

    # the real, honest verdict this was written to answer
    trap_frac = kinds.get("trap", 0) / max(1, n_atoms)
    print(f"\n  VERDICT:")
    if trap_frac > 0.3:
        print(f"    {trap_frac*100:.0f}% of all nodes are traps (failure records) -- if this keeps growing "
              f"faster than real atoms/concepts, the graph IS being dominated by failure bookkeeping, not "
              f"useful structure. Worth checking whether record_failure's trap creation should dedupe more "
              f"aggressively or decay old traps, not just accumulate them forever.")
    if len(isolated) / max(1, n_atoms) > 0.3:
        print(f"    {100*len(isolated)/max(1,n_atoms):.0f}% of nodes have ZERO edges -- these contribute "
              f"embeddings for retrieval but nothing to spreading-activation/topology-aware retrieval "
              f"(GraphAttnEncoder, session focus). Structure IS sparse for a real fraction of the graph.")
    if components and len(components) > 1 and len(components[0]) / max(1, n_atoms) < 0.5:
        print(f"    Graph is FRAGMENTED: largest connected component is under half the graph "
              f"({len(components)} separate components total). Spreading activation/session-focus "
              f"boosting can't reach across components at all.")
    if trap_frac <= 0.3 and len(isolated) / max(1, n_atoms) <= 0.3 and (not components or len(components[0]) / max(1, n_atoms) >= 0.5):
        print(f"    Looks structurally healthy by these measures -- not dominated by traps, most nodes have "
              f"real edges, one large connected component.")

    if out_png:
        _draw(g, edges, out_png)
        print(f"\n  saved visualization -> {out_png}")


def _draw(g, edges, out_png: str) -> None:
    import matplotlib
    matplotlib.use("Agg")  # no display needed, just write the file
    import matplotlib.pyplot as plt
    import networkx as nx

    G = nx.Graph()
    kind_of = {}
    for name, a in g.atoms.items():
        G.add_node(name)
        kind_of[name] = a.kind
    for s, d, r in edges:
        if s in G and d in G:
            G.add_edge(s, d, relation=r)

    color_map = {"atom": "#4C9F70", "concept": "#4C8DA0", "trap": "#C0392B", "procedure": "#B08D57"}
    colors = [color_map.get(kind_of.get(n, ""), "#999999") for n in G.nodes()]
    sizes = [80 + 40 * G.degree(n) for n in G.nodes()]

    fig, ax = plt.subplots(figsize=(14, 14))
    pos = nx.spring_layout(G, seed=0, k=0.5 / max(1, len(G.nodes()) ** 0.5))
    nx.draw_networkx_edges(G, pos, alpha=0.3, width=0.8, ax=ax)
    nx.draw_networkx_nodes(G, pos, node_color=colors, node_size=sizes, ax=ax)
    isolated = [n for n in G.nodes() if G.degree(n) == 0]
    labels = {n: n for n in G.nodes() if G.degree(n) > 0 or n in isolated[:20]}
    nx.draw_networkx_labels(G, pos, labels=labels, font_size=6, ax=ax)
    ax.set_title(f"AtomGraph: {len(G.nodes())} nodes, {len(G.edges())} edges "
                f"(green=atom, blue=concept, red=trap, tan=procedure; isolated nodes labeled too)")
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Real connectivity report + visualization for an AtomGraph.")
    ap.add_argument("--graph-path", type=str, default="graphs/long_term.json")
    ap.add_argument("--out-png", type=str, default="artifacts/graph_connectivity.png",
                    help="where to save the visualization (set to '' to skip drawing)")
    args = ap.parse_args(argv)
    report(args.graph_path, args.out_png or None)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
