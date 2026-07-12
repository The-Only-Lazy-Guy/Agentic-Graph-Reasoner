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
    "dijkstra": ("shortest weighted distance from s to every node (list; -1 if unreachable)",
                 "def dijkstra(n, edges, s):\n    import heapq\n    adj = {i: [] for i in range(n)}\n"
                 "    for u, v, w in edges:\n        adj[u].append((v, w))\n    INF = float('inf')\n"
                 "    dist = [INF] * n\n    dist[s] = 0\n    h = [(0, s)]\n    while h:\n"
                 "        d, u = heapq.heappop(h)\n        if d > dist[u]:\n            continue\n"
                 "        for v, w in adj[u]:\n            if d + w < dist[v]:\n"
                 "                dist[v] = d + w\n                heapq.heappush(h, (d + w, v))\n"
                 "    return [x if x < INF else -1 for x in dist]"),
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
]

# reference (oracle) solutions — each GLUES atoms; proves the asserts are sound + compose solves it
_REF = {
    "sum_digitsum_primes": "def sum_digitsum_primes(lst):\n    return sum(digit_sum(x) for x in lst if is_prime(x))",
    "max_prime_digitsum": "def max_prime_digitsum(lst):\n    ds = [digit_sum(x) for x in lst if is_prime(x)]\n    return max(ds) if ds else 0",
    "sum_reachable_costs": "def sum_reachable_costs(n, edges, s):\n    return sum(d for d in dijkstra(n, edges, s) if d >= 0)",
    "count_reachable": "def count_reachable(n, edges, s):\n    return sum(1 for d in dijkstra(n, edges, s) if d >= 0)",
}

# which atoms each compose task needs (for the selftest + as a soft prior)
_NEEDS = {"sum_digitsum_primes": {"is_prime", "digit_sum"},
          "max_prime_digitsum": {"is_prime", "digit_sum"},
          "sum_reachable_costs": {"dijkstra"}, "count_reachable": {"dijkstra"}}


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
                 "dijkstra": ["assert dijkstra(4, [(0,1,2),(0,2,5),(1,2,1),(2,3,3)], 0) == [0,2,3,6]"]}[name]
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
    print(f"  [2] all {len(COMPOSE)} compose tasks: sound asserts + solved by GLUING atoms + FAIL "
          f"without them (compose-necessary) -> PASS")

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
