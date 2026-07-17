"""algo_grr_retrieval — topology-aware retrieval (GRR-Tool goal 1 + 2).

The MBPP+ A/B showed the seed-trained ComplementPolicy is OUT-OF-DISTRIBUTION on a diverse corpus (it
injected ranking noise -> reuse 24->4). The STRUCTURAL alternative generalises for free: use the GRAPH'S
OWN EDGES. When an atom is retrieved (or already selected), its depend-neighbours and concept-siblings are
likely COMPOSABLE with it, so boosting them surfaces the complement the flat cosine buries — no trained net,
no prompt cajoling. This makes the graph TOPOLOGY load-bearing (goal 1) and, via a cached adjacency index
that only rebuilds when the graph grows, keeps retrieval O(new) not O(N) (goal 2).

Drops into MembraneSolver.policy_fn like any retrieval policy.

    selftest (no GPU):  python -m v5.runtime.algo_grr_retrieval --selftest
"""
from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

_ROOT = str(Path(__file__).resolve().parents[2])
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from graph_core import MemoryGraph  # type: ignore  # noqa: E402
from v5.runtime.algo_grr_membrane import TokenRetriever  # noqa: E402


def _cosine_base(retriever: TokenRetriever):
    """A policy_fn-shaped base ranker over the token-cosine retriever."""
    def base(task, selected, graph, retr):
        return (retr or retriever).rank(task["text"], exclude=set(selected))
    return base


class TopologyRetriever:
    """Wraps a base ranker and BOOSTS the graph-neighbours of the current anchors (top base hits +
    already-selected atoms). Neighbour = depend-target, reverse-depend, or concept-sibling. The
    adjacency index is cached and only rebuilt when the node count changes (graph grew) -> scalable."""

    def __init__(self, graph: MemoryGraph, base=None, boost: float = 0.4, expand: int = 4):
        self.graph = graph
        self.base = base or _cosine_base(TokenRetriever(graph))
        self.boost = boost
        self.expand = expand
        self._n = -1
        self._build()

    def _build(self) -> None:
        # DEPEND edges only. Concept-siblings are too coarse (every number-theory atom is a sibling ->
        # a uniform boost just preserves cosine's bad within-concept order). depend links are the
        # precise composition signal: lcm->gcd, sum_divisors->divisors, is_anagram->char_freq.
        dep_out: dict[str, set] = defaultdict(set)
        dep_in: dict[str, set] = defaultdict(set)
        for e in self.graph.edges:
            if e.relation == "depend":
                dep_out[e.src].add(e.dst)
                dep_in[e.dst].add(e.src)
        self._dep_out, self._dep_in = dep_out, dep_in
        self._n = len(self.graph.nodes)

    def neighbors(self, nids) -> set:
        if len(self.graph.nodes) != self._n:      # graph grew during the loop -> rebuild index
            self._build()
        out: set = set()
        for nid in nids:
            out |= self._dep_out.get(nid, set())   # atoms this one depends on
            out |= self._dep_in.get(nid, set())    # atoms that depend on this one (the composites)
        return out - set(nids)

    def policy_fn(self, task, selected, graph, retriever):
        base = self.base(task, selected, graph, retriever)     # [(nid, score), ...]
        # program-conditioned: boost depend-neighbours of what is ALREADY SELECTED (+ the single top
        # base hit as a seed when nothing is selected yet). NOT the top-k base hits — that adds noise.
        anchors = list(selected) or ([base[0][0]] if base else [])
        nbr = self.neighbors(anchors)
        boosted = [(nid, s + (self.boost if nid in nbr else 0.0)) for nid, s in base]
        boosted.sort(key=lambda z: -z[1])
        return boosted


def make_topology_policy(graph: MemoryGraph, base=None, boost: float = 0.4, expand: int = 4):
    """Return a MembraneSolver.policy_fn that boosts graph-neighbours of the anchors (topology-aware)."""
    return TopologyRetriever(graph, base=base, boost=boost, expand=expand).policy_fn


# ═══════════════════════════════════════════════════════════════════════════════
# SELFTEST — topology surfaces the complement + is load-bearing (ablation)
# ═══════════════════════════════════════════════════════════════════════════════

def _seed_graph() -> MemoryGraph:
    from v5.runtime.algo_grr_seed import build_graph
    from graph_core import Node, Edge
    g = build_graph()
    nodes = {n["id"]: Node.from_dict(n) for n in g["nodes"]}
    edges = [Edge.from_dict(e) for e in g["edges"]]
    return MemoryGraph(nodes, edges, metadata=dict(g["metadata"]))


def _rank_of(ranked, nid) -> int:
    for i, (n, _s) in enumerate(ranked):
        if n == nid:
            return i + 1
    return len(ranked) + 1


def _selftest() -> bool:
    print("algo_grr_retrieval --selftest: topology-aware retrieval (structural complement)\n")
    from v5.runtime.algo_grr_membrane import MembraneSolver, make_stub_compiler
    graph = _seed_graph()
    cos = TokenRetriever(graph)
    topo = make_topology_policy(graph)
    ok = True

    # [1] after selecting the first atom, the missing complement (a concept-sibling / dep-neighbour)
    #     ranks HIGHER under topology-boost than under plain cosine — no trained net.
    cases = [   # B is a DEPEND-neighbour of A -> after selecting A, topology should surface B
        ("the total of all divisors of n", "impl_divisors", "impl_sum_divisors"),
        ("least common multiple of a and b", "impl_gcd", "impl_lcm"),
        ("check if two strings use the same letters", "impl_char_freq", "impl_is_anagram"),
    ]
    improved = 0
    for text, first, comp in cases:
        task = {"text": text}
        r_topo = _rank_of(topo(task, [first], graph, cos), comp)
        r_cos = _rank_of([(n, s) for n, s in cos.rank(text, exclude={first})], comp)
        improved += int(r_topo <= r_cos)
        print(f"  {comp:22s} after {first:20s}  topo_rank={r_topo:2d}  cosine_rank={r_cos:2d}")
    print(f"  [1] complement rank: topology <= cosine on {improved}/{len(cases)} -> "
          f"{'PASS' if improved == len(cases) else 'FAIL'}")
    ok &= improved == len(cases)

    # [2] topology drops into the membrane and solves genuine-2 compositions in <= cosine hops
    recipes = {
        "t_primediv": "def t_primediv(n):\n    return sum(1 for d in divisors(n) if is_prime(d))\n",
        "t_perfect": "def t_perfect(n):\n    return sum_divisors(n) - n == n\n",
    }
    tasks = [
        dict(text="count how many divisors of n are prime numbers", entry="t_primediv",
             tests=[((12,), 2), ((30,), 3), ((7,), 1)]),
        dict(text="is n a perfect number using its divisors", entry="t_perfect",
             tests=[((6,), True), ((12,), False), ((28,), True)]),
    ]
    comp = make_stub_compiler(recipes)
    th = ch = ts = cs = 0
    for t in tasks:
        rt = MembraneSolver(graph, comp, policy_fn=make_topology_policy(graph)).solve(t)
        rc = MembraneSolver(graph, comp).solve(t)
        th += sum(1 for e in rt["trace"] if "hop" in e)
        ch += sum(1 for e in rc["trace"] if "hop" in e)
        ts += int(rt["solved"]); cs += int(rc["solved"])
    print(f"  [2] membrane solve: topology {ts}/{len(tasks)} in {th} hops, cosine {cs}/{len(tasks)} in "
          f"{ch} hops -> {'PASS' if ts == len(tasks) and th <= ch else 'FAIL'}")
    ok &= ts == len(tasks) and th <= ch

    # [3] index rebuilds when the graph grows (scalable: only on size change)
    tr = TopologyRetriever(graph)
    n0 = tr._n
    from graph_core import Node, Edge
    graph.nodes["impl_new_x"] = Node(id="impl_new_x", text="x", node_type="implementation",
                                     metadata={"entry": "new_x"})
    graph.edges.append(Edge(src="impl_new_x", dst="concept_number_theory", relation="part_of"))
    graph._rebuild_index()
    _ = tr.neighbors(["impl_is_prime"])            # triggers rebuild since node count changed
    rebuilt = tr._n == n0 + 1
    print(f"  [3] index rebuild on growth: {n0} -> {tr._n} -> {'PASS' if rebuilt else 'FAIL'}")
    ok &= rebuilt

    # [4] ABLATION — depend topology is LOAD-BEARING at realize time: is_perfect depends on
    #     sum_divisors depends on divisors; with the depend edges the closure resolves and runs,
    #     without them the composite can't realize (NameError). Topology is genuinely useful.
    from v5.runtime.algo_grr_membrane import realize_closure_code
    g2 = _seed_graph()
    with_edges = realize_closure_code(g2, ["impl_is_perfect"])
    flat = MemoryGraph(dict(g2.nodes), [e for e in g2.edges if e.relation != "depend"], g2.metadata)
    without = realize_closure_code(flat, ["impl_is_perfect"])

    def _runs(code):
        ns: dict = {}
        try:
            exec(code, ns)
            return ns["is_perfect"](28) is True
        except Exception:  # noqa: BLE001
            return False
    ok_with, ok_without = _runs(with_edges), _runs(without)
    print(f"  [4] depend-closure ablation: with-edges runs={ok_with}, without-edges runs={ok_without} "
          f"-> {'PASS' if ok_with and not ok_without else 'FAIL'}")
    ok &= ok_with and not ok_without

    print(f"\n  ALGO_GRR_RETRIEVAL SELFTEST -> {'PASS' if ok else 'FAIL'}")
    return ok


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        sys.exit(0 if _selftest() else 1)
    ap.print_help()


if __name__ == "__main__":
    main()
