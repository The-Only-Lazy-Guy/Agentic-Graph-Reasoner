"""Compose-NECESSARY task family + bootstrap atoms — the setup that actually exercises reasoning-over-
graph. MBPP-random is inline-solvable (why reuse stayed ~8%); these tasks are hard to write inline but
EASY to glue from verified atoms, so composition is the path the search should find.

  ATOMS (bootstrap, tier 0): is_prime, digit_sum, build_adj, dijkstra — each a small verified unit that
                             seeds the graph so the compose search has a candidate set.
  COMPOSE (tier 1): each REQUIRES 2+ atoms glued together, verified by asserts:
      sum_digitsum_primes  = is_prime  + digit_sum
      max_prime_digitsum   = is_prime  + digit_sum
      sum_reachable_costs  = dijkstra  (+ sum)        (dijkstra itself = build_adj + heap)
      count_reachable      = dijkstra  (+ count)

Reference solutions here are the ORACLE (used to prove the asserts are sound + that gluing the atoms
solves each). The real run has the LM glue-realize; verify_fn/asserts decide.

  selftest (no model):  python -m v5.runtime.algo_compose_tasks --selftest
"""
from __future__ import annotations

import argparse
import sys

import numpy as np

from v5.runtime.algo_graph_run import MBPPTask


# ── ATOMS (verified units; seed the graph) ────────────────────────────────────────
ATOMS: dict[str, tuple[str, str]] = {
    "is_prime": ("prime test — True iff n is prime",
                 "def is_prime(n):\n    if n < 2:\n        return False\n    i = 2\n"
                 "    while i * i <= n:\n        if n % i == 0:\n            return False\n"
                 "        i += 1\n    return True"),
    "digit_sum": ("sum of the decimal digits of a non-negative integer",
                  "def digit_sum(n):\n    return sum(int(c) for c in str(abs(n)))"),
    "build_adj": ("adjacency dict {node: [(neighbor, weight)]} from n and an edge list of (u, v, w)",
                  "def build_adj(n, edges):\n    adj = {i: [] for i in range(n)}\n"
                  "    for u, v, w in edges:\n        adj[u].append((v, w))\n    return adj"),
    # NOTE: returns a DICT {reachable_node: dist} — NO -1 sentinel. A sentinel is a composition trap
    # (the 3B composed dijkstra ~55% of samples but every glue failed on summing/counting the -1). A
    # dict makes glue clean: sum(d.values()) / len(d). Atom contracts must be glue-friendly.
    "dijkstra": ("shortest weighted distances from s as a dict {node: dist} over REACHABLE nodes only "
                 "(unreachable nodes are absent; source maps to 0)",
                 "def dijkstra(n, edges, s):\n    import heapq\n    adj = {i: [] for i in range(n)}\n"
                 "    for u, v, w in edges:\n        adj[u].append((v, w))\n    dist = {s: 0}\n"
                 "    h = [(0, s)]\n    while h:\n        d, u = heapq.heappop(h)\n"
                 "        if d > dist.get(u, float('inf')):\n            continue\n"
                 "        for v, w in adj[u]:\n            nd = d + w\n"
                 "            if nd < dist.get(v, float('inf')):\n                dist[v] = nd\n"
                 "                heapq.heappush(h, (nd, v))\n    return dist"),
}

_G = [(0, 1, 2), (0, 2, 5), (1, 2, 1), (2, 3, 3)]        # dijkstra(4,_G,0) = [0, 2, 3, 6]


# ── COMPOSE-NECESSARY tasks (need 2+ atoms) ───────────────────────────────────────
COMPOSE = [
    MBPPTask("sum_digitsum_primes",
             "sum of the digit-sums of the PRIME numbers in a list. needs: a prime test AND a digit sum.",
             ["assert sum_digitsum_primes([11, 4, 23]) == (1+1) + (2+3)",   # primes 11,23 -> 2+5 = 7
              "assert sum_digitsum_primes([8, 9, 10]) == 0"]),
    MBPPTask("max_prime_digitsum",
             "the largest digit-sum among the PRIME numbers in a list, or 0 if none. needs a prime test "
             "AND a digit sum.",
             ["assert max_prime_digitsum([11, 23, 4]) == 5",                # max(digit_sum(11)=2, 23=5)
              "assert max_prime_digitsum([4, 6, 8]) == 0"]),
    MBPPTask("sum_reachable_costs",
             "sum of the shortest-path costs from node s to every reachable node (edges = list of "
             "(u,v,w)). needs shortest-path distances.",
             ["assert sum_reachable_costs(4, [(0,1,2),(0,2,5),(1,2,1),(2,3,3)], 0) == 0+2+3+6",
              "assert sum_reachable_costs(3, [(0,1,4)], 0) == 0+4"]),
    MBPPTask("count_reachable",
             "how many nodes are reachable from s (including s) in a weighted directed graph (edges = "
             "list of (u,v,w)). needs shortest-path distances.",
             ["assert count_reachable(4, [(0,1,2),(0,2,5),(1,2,1),(2,3,3)], 0) == 4",
              "assert count_reachable(3, [(0,1,4)], 0) == 2"]),
    # cross-domain / 3-atom: inlining is hopeless (would re-derive dijkstra AND is_prime AND digit_sum)
    MBPPTask("count_prime_reachable",
             "how many reachable nodes have a PRIME node id, from s in a weighted directed graph "
             "(edges = list of (u,v,w)). needs shortest paths AND a prime test.",
             ["assert count_prime_reachable(4, [(0,1,2),(0,2,5),(1,2,1),(2,3,3)], 0) == 2",
              "assert count_prime_reachable(3, [(0,1,4)], 0) == 0"]),
    MBPPTask("sum_digitsum_reachable",
             "sum of the digit-sums of the shortest-path DISTANCES to all reachable nodes from s "
             "(edges = list of (u,v,w)). needs shortest paths AND a digit sum.",
             ["assert sum_digitsum_reachable(4, [(0,1,2),(0,2,5),(1,2,1),(2,3,3)], 0) == 11",
              "assert sum_digitsum_reachable(3, [(0,1,4)], 0) == 4"]),
    MBPPTask("sum_digitsum_prime_dists",
             "sum of the digit-sums of the shortest-path distances that are PRIME, over reachable "
             "nodes from s (edges = list of (u,v,w)). needs shortest paths, a prime test, AND a digit "
             "sum (three atoms).",
             ["assert sum_digitsum_prime_dists(4, [(0,1,2),(0,2,5),(1,2,1),(2,3,3)], 0) == 5",
              "assert sum_digitsum_prime_dists(4, [(0,1,2),(1,2,9),(2,3,2)], 0) == 8"]),
]

# reference (oracle) solutions — each GLUES atoms; proves the asserts are sound + compose solves it
_REF = {
    "sum_digitsum_primes": "def sum_digitsum_primes(lst):\n    return sum(digit_sum(x) for x in lst if is_prime(x))",
    "max_prime_digitsum": "def max_prime_digitsum(lst):\n    ds = [digit_sum(x) for x in lst if is_prime(x)]\n    return max(ds) if ds else 0",
    "sum_reachable_costs": "def sum_reachable_costs(n, edges, s):\n    return sum(dijkstra(n, edges, s).values())",
    "count_reachable": "def count_reachable(n, edges, s):\n    return len(dijkstra(n, edges, s))",
    "count_prime_reachable": "def count_prime_reachable(n, edges, s):\n    return sum(1 for node in dijkstra(n, edges, s) if is_prime(node))",
    "sum_digitsum_reachable": "def sum_digitsum_reachable(n, edges, s):\n    return sum(digit_sum(d) for d in dijkstra(n, edges, s).values())",
    "sum_digitsum_prime_dists": "def sum_digitsum_prime_dists(n, edges, s):\n    return sum(digit_sum(d) for d in dijkstra(n, edges, s).values() if is_prime(d))",
}

# which atoms each compose task needs (for the selftest + as a soft prior)
_NEEDS = {"sum_digitsum_primes": {"is_prime", "digit_sum"},
          "max_prime_digitsum": {"is_prime", "digit_sum"},
          "sum_reachable_costs": {"dijkstra"}, "count_reachable": {"dijkstra"},
          "count_prime_reachable": {"dijkstra", "is_prime"},
          "sum_digitsum_reachable": {"dijkstra", "digit_sum"},
          "sum_digitsum_prime_dists": {"dijkstra", "is_prime", "digit_sum"}}


# ── parametrized generator — MANY compose-necessary instances (the training-corpus data gate) ─────
# oracle funcs mirror the ATOMS (dict-dijkstra contract); family oracles COMPOSE them.

def _is_prime(x):
    if x < 2:
        return False
    i = 2
    while i * i <= x:
        if x % i == 0:
            return False
        i += 1
    return True


def _digit_sum(x):
    return sum(int(c) for c in str(abs(x)))


def _dijkstra(n, edges, s):
    import heapq
    adj = {i: [] for i in range(n)}
    for u, v, w in edges:
        adj[u].append((v, w))
    dist = {s: 0}
    h = [(0, s)]
    while h:
        d, u = heapq.heappop(h)
        if d > dist.get(u, float("inf")):
            continue
        for v, w in adj[u]:
            nd = d + w
            if nd < dist.get(v, float("inf")):
                dist[v] = nd
                heapq.heappush(h, (nd, v))
    return dist


# family -> (kind, oracle). text/ref/needs reuse the COMPOSE + _REF + _NEEDS above (same fn names).
_FAMILIES = {
    "sum_digitsum_primes": ("list", lambda a: sum(_digit_sum(x) for x in a if _is_prime(x))),
    "max_prime_digitsum": ("list", lambda a: max([_digit_sum(x) for x in a if _is_prime(x)] or [0])),
    "sum_reachable_costs": ("graph", lambda n, e, s: sum(_dijkstra(n, e, s).values())),
    "count_reachable": ("graph", lambda n, e, s: len(_dijkstra(n, e, s))),
    "count_prime_reachable": ("graph", lambda n, e, s: sum(1 for k in _dijkstra(n, e, s) if _is_prime(k))),
    "sum_digitsum_reachable": ("graph", lambda n, e, s: sum(_digit_sum(d) for d in _dijkstra(n, e, s).values())),
    "sum_digitsum_prime_dists": ("graph", lambda n, e, s: sum(_digit_sum(d) for d in _dijkstra(n, e, s).values() if _is_prime(d))),
}
_TEXT = {t.name: t.text for t in COMPOSE}


def _rand_list(rng):
    return [int(rng.integers(2, 40)) for _ in range(int(rng.integers(3, 7)))]


def _rand_graph(rng):
    n = int(rng.integers(3, 7))
    edges = []
    for u in range(n):
        for v in range(u + 1, n):
            if rng.random() < 0.55:
                edges.append((u, v, int(rng.integers(1, 10))))
    return n, edges, 0


def gen_compose_tasks(n: int = 200, seed: int = 0):
    """n parametrized compose-necessary instances (random inputs, oracle-computed asserts). Same fn
    names as the families (so retrieval/atoms line up); asserts are sound by construction. The harvest
    corpus + a bigger, deeper task set than the 7 static COMPOSE."""
    rng = np.random.default_rng(seed)
    fams = list(_FAMILIES)
    tasks = []
    for i in range(n):
        name = fams[i % len(fams)]
        kind, oracle = _FAMILIES[name]
        asserts = []
        tries = 0
        while len(asserts) < 2 and tries < 20:
            tries += 1
            if kind == "list":
                a = _rand_list(rng)
                asserts.append(f"assert {name}({a!r}) == {oracle(a)!r}")
            else:
                gn, e, s = _rand_graph(rng)
                asserts.append(f"assert {name}({gn}, {e!r}, {s}) == {oracle(gn, e, s)!r}")
        tasks.append(MBPPTask(name, _TEXT[name], asserts))
    return tasks


def seed_atom_graph(path: str, concept: str = "concept_algorithms"):
    """Write a MemoryGraph pre-seeded with the verified atoms (bootstrap the candidate set)."""
    import json
    from pathlib import Path
    nodes = [{"id": concept, "text": "algorithms", "node_type": "concept"}]
    edges = []
    for name, (purpose, code) in ATOMS.items():
        nodes.append({"id": f"impl_{name}", "text": purpose, "node_type": "implementation",
                      "metadata": {"code": code}})
        edges.append({"src": f"impl_{name}", "dst": concept, "relation": "part_of",
                      "strength": 0.5, "directed": True})
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps({"metadata": {}, "nodes": nodes, "edges": edges}), encoding="utf-8")


def _selftest() -> bool:
    from v5.runtime.tool_compose import verify_fn  # noqa (imported for parity; using verify_asserts)
    from v5.runtime.algo_graph_run import verify_asserts
    print("algo_compose_tasks --selftest: atoms verify, compose tasks are SOUND + solved by GLUING atoms\n")

    # atoms verify on their own
    for name, (_, code) in ATOMS.items():
        tests = {"is_prime": ["assert is_prime(7) and not is_prime(8) and is_prime(23)"],
                 "digit_sum": ["assert digit_sum(23) == 5 and digit_sum(0) == 0"],
                 "build_adj": ["assert build_adj(2, [(0,1,3)]) == {0: [(1,3)], 1: []}"],
                 "dijkstra": ["assert dijkstra(4, [(0,1,2),(0,2,5),(1,2,1),(2,3,3)], 0) == {0:0,1:2,2:3,3:6}",
                              "assert dijkstra(3, [(0,1,4)], 0) == {0:0, 1:4}"]}[name]
        assert verify_asserts(code, tests), f"atom {name} failed"
    print(f"  [1] all {len(ATOMS)} atoms verify -> PASS")

    # each COMPOSE task: (a) asserts are sound (ref solution + its atom deps passes), (b) the ref
    # GLUES its needed atoms (composition is the path)
    for t in COMPOSE:
        deps = "\n\n".join(ATOMS[a][1] for a in _NEEDS[t.name])
        ref = _REF[t.name]
        assert t.verify(ref, deps), f"{t.name}: ref+atoms must pass its asserts"
        assert all(re_search(a, ref) for a in _NEEDS[t.name]), f"{t.name}: ref must CALL its atoms"
        # and it FAILS without the atoms (genuinely compose-necessary, not inlined)
        assert not t.verify(ref, ""), f"{t.name}: ref must NEED its atoms (fails without them)"
    print(f"  [2] all {len(COMPOSE)} compose tasks ({sum(1 for t in COMPOSE if len(_NEEDS[t.name])>=2)} "
          f"multi-atom, incl 3-atom): sound asserts + solved by GLUING atoms + FAIL without them "
          f"(compose-necessary) -> PASS")

    # [3] generator: many parametrized instances, each SOUND (ref+its atoms passes the random asserts)
    gen = gen_compose_tasks(n=70, seed=1)
    for t in gen:
        deps = "\n\n".join(ATOMS[a][1] for a in _NEEDS[t.name])
        assert t.verify(_REF[t.name], deps), f"generated {t.name} unsound: {t.tests}"
    fam_counts = {f: sum(1 for t in gen if t.name == f) for f in _FAMILIES}
    print(f"  [3] generator: {len(gen)} parametrized instances across {len(_FAMILIES)} families, ALL "
          f"sound (oracle asserts pass the composing ref) -> PASS")

    print("\n  ALGO_COMPOSE_TASKS SELFTEST -> PASS")
    return True


def re_search(atom: str, code: str) -> bool:
    import re
    return bool(re.search(rf"\b{re.escape(atom)}\s*\(", code))


def main():
    ap = argparse.ArgumentParser(description="Compose-necessary tasks + bootstrap atoms.")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        sys.exit(0 if _selftest() else 1)
    ap.print_help()


if __name__ == "__main__":
    main()
