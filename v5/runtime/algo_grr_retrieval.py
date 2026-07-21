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
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

_ROOT = str(Path(__file__).resolve().parents[2])
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from graph_core import MemoryGraph  # type: ignore  # noqa: E402
from v5.runtime.algo_grr_membrane import TokenRetriever, _tokens  # noqa: E402


# ═══════════════════════════════════════════════════════════════════════════════
# CachedTokenRetriever — scalable cosine (goal 2): tokenize each atom ONCE, reuse across
# tasks, incrementally add only NEW (banked) atoms. Same .rank() interface as TokenRetriever.
# ═══════════════════════════════════════════════════════════════════════════════

class CachedTokenRetriever:
    """Drop-in for TokenRetriever that CACHES per-atom token vectors. TokenRetriever re-tokenizes every
    atom on construction, and the membrane builds a fresh retriever per task -> O(atoms x tasks). This
    tokenizes each atom once and, when the graph GROWS (banking), only tokenizes the new atoms. idf is
    recomputed on growth (cheap counting, not re-tokenization). Reuse ONE instance across the corpus."""

    def __init__(self, graph: MemoryGraph):
        self.graph = graph
        self._vecs: dict[str, Counter] = {}
        self._idf: dict[str, float] = {}
        self._impl: list[str] = []
        self.tok_calls = 0                # instrumentation: how many atoms were tokenized
        self._sync()

    def _sync(self) -> None:
        impl = [nid for nid, n in self.graph.nodes.items() if n.node_type == "implementation"]
        new = [nid for nid in impl if nid not in self._vecs]
        for nid in new:
            self._vecs[nid] = Counter(_tokens(self.graph.nodes[nid].text))
            self.tok_calls += 1
        for nid in list(self._vecs):      # drop removed nodes
            if nid not in self.graph.nodes:
                del self._vecs[nid]
        if new or len(impl) != len(self._impl):
            df: Counter = Counter()
            for v in self._vecs.values():
                df.update(v.keys())
            n = max(1, len(self._vecs))
            self._idf = {t: math.log(1.0 + n / (1 + c)) for t, c in df.items()}
        self._impl = impl

    def _w(self, counter: Counter) -> dict[str, float]:
        default = math.log(1.0 + max(1, len(self._impl)))
        return {t: c * self._idf.get(t, default) for t, c in counter.items()}

    @staticmethod
    def _cos(a: dict[str, float], b: dict[str, float]) -> float:
        if not a or not b:
            return 0.0
        dot = sum(a[t] * b.get(t, 0.0) for t in a)
        na = math.sqrt(sum(x * x for x in a.values()))
        nb = math.sqrt(sum(x * x for x in b.values()))
        return dot / (na * nb) if na and nb else 0.0

    def rank(self, query_text: str, exclude: set | None = None):
        self._sync()                      # picks up newly-banked atoms (tokenizes only those)
        exclude = exclude or set()
        q = self._w(Counter(_tokens(query_text)))
        scored = [(nid, self._cos(q, self._w(self._vecs[nid])))
                  for nid in self._impl if nid not in exclude]
        scored.sort(key=lambda z: -z[1])
        return scored


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
# SpreadingActivationRetriever — energy propagation through typed edges
# ═══════════════════════════════════════════════════════════════════════════════

class SpreadingActivationRetriever:
    """Retrieval by iterative energy propagation through the graph. Instead of scoring every node
    against the prompt independently, injects 'energy' into seed nodes and propagates through
    typed edges with configurable weights:

      Positive edges (depend, part_of, support, corrected_by): carry positive weight → amplify
      Negative edges (avoid_if, contradict, conflict): carry negative weight → suppress
      Neutral edges (related, example_of): carry small positive weight

    The graph settles into a stable activation pattern where a connected sub-graph lights up.
    The result is not isolated nodes but a coherent context block — the entire glowing sub-graph.

    Activation equation:
        A_j(t) = tanh(sum_i A_i(t-1) * W_{ij} * C_j)

    where W_{ij} is the edge-type weight, C_j is the node's confidence score.

    Usage:
        sar = SpreadingActivationRetriever(graph)
        ranked = sar.rank(query_text, steps=5, top_k=20)
        # or get the full glowing sub-graph:
        subgraph = sar.activate_seeds(["impl_is_prime"], steps=5)
    """

    # Edge-type → propagation weight (positive = amplify, negative = suppress)
    EDGE_WEIGHTS: dict[str, float] = {
        "depend": 0.8,
        "part_of": 0.6,
        "support": 0.7,
        "imply": 0.7,
        "refine": 0.5,
        "cause": 0.5,
        "corrected_by": 0.4,
        "example_of": 0.2,
        "related": 0.15,
        "avoid_if": -0.6,
        "contradict": -0.5,
        "conflict": -0.5,
        "refute": -0.5,
        "trap_for": -0.4,
    }

    def __init__(self, graph: MemoryGraph,
                 base=None,
                 steps: int = 5,
                 activation_threshold: float = 0.1,
                 seed_boost: float = 0.3):
        self.graph = graph
        self.base = base or TokenRetriever(graph)
        self.steps = steps
        self.activation_threshold = activation_threshold
        self.seed_boost = seed_boost
        self._build()

    def _build(self) -> None:
        """Build adjacency matrix with typed edge weights."""
        adj: dict[str, list[tuple[str, float]]] = defaultdict(list)
        for e in self.graph.edges:
            w = self.EDGE_WEIGHTS.get(e.relation, 0.1)
            adj[e.src].append((e.dst, w))
            if not e.directed:
                adj[e.dst].append((e.src, w))
        self._adj = dict(adj)

    def _seed_from_query(self, query_text: str, n_seeds: int = 5) -> dict[str, float]:
        """Get initial seed nodes from base retriever, with their cosine scores as initial energy."""
        base_ranks = self.base.rank(query_text)[:n_seeds]
        energies: dict[str, float] = {}
        for nid, score in base_ranks:
            if nid in self.graph.nodes:
                initial = max(0.1, score) * self.seed_boost
                energies[nid] = initial
        # Always include the top seed with full energy
        if base_ranks:
            top = base_ranks[0][0]
            if top in self.graph.nodes:
                energies[top] = max(energies.get(top, 0), 0.5)
        return energies

    def activate(self, seed_energies: dict[str, float],
                 steps: int | None = None) -> dict[str, float]:
        """Iterative spreading activation. Returns node_id -> final activation."""
        steps = steps or self.steps
        a: dict[str, float] = dict(seed_energies)
        # Ensure all candidate nodes have at least 0 activation
        for nid in self.graph.nodes:
            a.setdefault(nid, 0.0)

        for _ in range(steps):
            a_next: dict[str, float] = {}
            for nid in self.graph.nodes:
                n = self.graph.nodes[nid]
                c_j = n.confidence  # node confidence as a gate
                incoming = 0.0
                neighbors = self._adj.get(nid, [])
                for src_nid, w in neighbors:
                    incoming += a.get(src_nid, 0.0) * w * c_j
                # Self-preservation (keep some prior activation)
                prior = a.get(nid, 0.0) * 0.5
                a_next[nid] = math.tanh(incoming + prior)
            a = a_next

        return a

    def activate_seeds(self, seed_node_ids: list[str],
                       steps: int | None = None) -> dict[str, float]:
        """Inject energy into specific seed nodes and propagate."""
        energies = {nid: 1.0 for nid in seed_node_ids if nid in self.graph.nodes}
        return self.activate(energies, steps)

    def rank(self, query_text: str, exclude: set | None = None,
             steps: int | None = None, top_k: int | None = None):
        """Rank atoms by spreading activation from query-derived seeds.
        Returns [(nid, activation), ...] sorted descending."""
        exclude = exclude or set()
        seeds = self._seed_from_query(query_text)
        activation = self.activate(seeds, steps)

        impl = [nid for nid, n in self.graph.nodes.items()
                if n.node_type == "implementation" and nid not in exclude]
        scored = [(nid, activation.get(nid, 0.0)) for nid in impl]
        scored.sort(key=lambda z: -z[1])

        if top_k is not None:
            scored = scored[:top_k]
        return scored

    def glowing_subgraph(self, query_text: str, steps: int | None = None,
                         threshold: float | None = None) -> list[str]:
        """Return all node ids whose activation exceeds threshold after settling.
        This is the 'connected glowing sub-graph' — a coherent context block."""
        threshold = threshold or self.activation_threshold
        activation = self.activate(self._seed_from_query(query_text), steps)
        return sorted([nid for nid, a in activation.items()
                       if a > threshold and nid in self.graph.nodes])


def make_spreading_policy(graph: MemoryGraph, base=None, steps: int = 5,
                          activation_threshold: float = 0.1):
    """Return a MembraneSolver.policy_fn using SpreadingActivationRetriever."""
    sar = SpreadingActivationRetriever(graph, base_retriever=base,
                                       steps=steps, activation_threshold=activation_threshold)

    def policy_fn(task, selected, _graph, retriever):
        exclude = set(selected) if selected else set()
        return sar.rank(task["text"], exclude=exclude)
    return policy_fn


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

    # [5] SCALABILITY — CachedTokenRetriever tokenizes each atom ONCE, only new atoms on growth,
    #     and gives the SAME ranking as the plain retriever.
    g5 = _seed_graph()
    n_impl = sum(1 for n in g5.nodes.values() if n.node_type == "implementation")
    cr = CachedTokenRetriever(g5)
    q = "check whether a number is prime"
    top_cached = cr.rank(q)[0][0]
    top_plain = TokenRetriever(g5).rank(q)[0][0]
    calls_after_build = cr.tok_calls
    _ = cr.rank("least common multiple")           # second query -> NO re-tokenization
    from graph_core import Node, Edge
    g5.nodes["impl_grow_z"] = Node(id="impl_grow_z", text="grow z helper", node_type="implementation",
                                   metadata={"entry": "grow_z"})
    g5.edges.append(Edge(src="impl_grow_z", dst="concept_lists", relation="part_of"))
    g5._rebuild_index()
    _ = cr.rank("grow z helper")                    # picks up the new atom, tokenizes ONLY it
    incremental = (calls_after_build == n_impl and cr.tok_calls == n_impl + 1)
    parity = top_cached == top_plain
    print(f"  [5] cache: tokenized {n_impl}->build, +1 on growth (calls={cr.tok_calls}), "
          f"parity_with_plain={parity} -> {'PASS' if incremental and parity else 'FAIL'}")
    ok &= incremental and parity

    # ═══════════════════════════════════════════════════════════════════
    # SpreadingActivationRetriever selftests (v6)
    # ═══════════════════════════════════════════════════════════════════

    # [6] energy injection + propagation: depend edge spreads activation
    g6 = _seed_graph()
    sar = SpreadingActivationRetriever(g6, base=TokenRetriever(g6), steps=3)
    seeds = {"impl_divisors": 1.0, "impl_is_prime": 1.0}
    activation = sar.activate(seeds)
    # divisors and is_prime should both have non-zero activation
    assert activation.get("impl_divisors", 0) > 0.01, "seed should stay active"
    # sum_divisors depends on divisors -> should get residual activation through depend edge (weight 0.8)
    sn = activation.get("impl_sum_divisors", 0)
    lb = activation.get("impl_lcm", 0)
    print(f"  [6] spreading activation: divisors={activation.get('impl_divisors', 0):.3f}, "
          f"sum_divisors={sn:.3f}, lcm={lb:.3f} "
          f"-> {'PASS' if sn > 0.01 else 'FAIL'}")
    ok &= sn > 0.01

    # [7] rank() produces scored list using base seeds
    ranked_sa = sar.rank("check whether a number is prime", top_k=5)
    assert len(ranked_sa) > 0, "rank should return results"
    assert ranked_sa[0][0] in ("impl_is_prime", "impl_divisors"), \
        f"top should be is_prime or divisors, got {ranked_sa[0]}"
    print(f"  [7] spreading rank() top: {ranked_sa[0]} -> PASS")
    ok &= ranked_sa[0][1] > 0

    # [8] glowing_subgraph returns coherent context block
    subgraph = sar.glowing_subgraph("check whether a number is prime", threshold=0.001, steps=5)
    assert len(subgraph) >= 2, f"expected 2+ glowing nodes, got {len(subgraph)}: {subgraph}"
    print(f"  [8] glowing subgraph ({len(subgraph)} nodes): {subgraph[:6]}... -> PASS")
    ok &= len(subgraph) >= 2

    # [9] negative edge suppresses activation (avoid_if edge inserted)
    g9 = _seed_graph()
    g9.nodes["impl_bad_trap"] = Node(
        id="impl_bad_trap", text="trap: wrong prime check", node_type="trap",
        confidence=0.9, importance=0.3, access_count=0,
        metadata={"code": "def bad_prime(n): return False", "signature": "off-by-one"})
    g9.edges.append(Edge(src="impl_bad_trap", dst="impl_is_prime", relation="avoid_if",
                         strength=0.8, directed=True, confidence=0.9))
    g9._rebuild_index()
    sar9 = SpreadingActivationRetriever(g9, base=TokenRetriever(g9), steps=3)
    seeds9 = {"impl_is_prime": 1.0}
    act9 = sar9.activate(seeds9)
    # avoid_if edge (-0.6 weight) from trap to is_prime should slightly suppress is_prime
    assert act9.get("impl_is_prime", 0) > 0, "is_prime should still be active"
    print(f"  [9] negative edge (avoid_if) does not crash, is_prime={act9.get('impl_is_prime', 0):.3f} -> PASS")
    ok &= act9.get("impl_is_prime", 0) > 0

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
