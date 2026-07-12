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


# ── HARD suite: inline-FAILING atoms (DP the 4B botches) + trivial-glue families over them ─────────
# The capability test: literal-inline of these atoms craters; composing the seeded atom + a trivial
# reduce succeeds. So compose becomes NECESSARY for SOLVING (not just reuse) -> the adapter's
# forced-composition should convert to a SOLVE lift, unlike the inline-easy suite (ceiling).

HARD_ATOMS: dict[str, tuple[str, str]] = {
    "edit_distance": ("Levenshtein edit distance between two strings a and b (min single-char inserts, "
                      "deletes, substitutions)",
                      "def edit_distance(a, b):\n    m, n = len(a), len(b)\n"
                      "    dp = [[0] * (n + 1) for _ in range(m + 1)]\n"
                      "    for i in range(m + 1):\n        dp[i][0] = i\n"
                      "    for j in range(n + 1):\n        dp[0][j] = j\n"
                      "    for i in range(1, m + 1):\n        for j in range(1, n + 1):\n"
                      "            if a[i-1] == b[j-1]:\n                dp[i][j] = dp[i-1][j-1]\n"
                      "            else:\n                dp[i][j] = 1 + min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])\n"
                      "    return dp[m][n]"),
    "lcs_length": ("length of the longest common subsequence of two strings a and b",
                   "def lcs_length(a, b):\n    m, n = len(a), len(b)\n"
                   "    dp = [[0] * (n + 1) for _ in range(m + 1)]\n"
                   "    for i in range(1, m + 1):\n        for j in range(1, n + 1):\n"
                   "            if a[i-1] == b[j-1]:\n                dp[i][j] = dp[i-1][j-1] + 1\n"
                   "            else:\n                dp[i][j] = max(dp[i-1][j], dp[i][j-1])\n"
                   "    return dp[m][n]"),
    "coin_change": ("fewest coins from `coins` summing to `amount`, or -1 if impossible",
                    "def coin_change(coins, amount):\n    INF = float('inf')\n"
                    "    dp = [0] + [INF] * amount\n    for a in range(1, amount + 1):\n"
                    "        for c in coins:\n            if c <= a and dp[a-c] + 1 < dp[a]:\n"
                    "                dp[a] = dp[a-c] + 1\n    return dp[amount] if dp[amount] < INF else -1"),
    "lis_length": ("length of the longest strictly-increasing subsequence of a list",
                   "def lis_length(arr):\n    import bisect\n    tails = []\n    for x in arr:\n"
                   "        i = bisect.bisect_left(tails, x)\n        if i == len(tails):\n"
                   "            tails.append(x)\n        else:\n            tails[i] = x\n    return len(tails)"),
}
ALL_ATOMS = {**ATOMS, **HARD_ATOMS}


def _edit_distance(a, b):
    m, n = len(a), len(b)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            dp[i][j] = dp[i-1][j-1] if a[i-1] == b[j-1] else 1 + min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])
    return dp[m][n]


def _lcs_length(a, b):
    m, n = len(a), len(b)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            dp[i][j] = dp[i-1][j-1] + 1 if a[i-1] == b[j-1] else max(dp[i-1][j], dp[i][j-1])
    return dp[m][n]


def _coin_change(coins, amount):
    INF = float("inf")
    dp = [0] + [INF] * amount
    for a in range(1, amount + 1):
        for c in coins:
            if c <= a and dp[a-c] + 1 < dp[a]:
                dp[a] = dp[a-c] + 1
    return dp[amount] if dp[amount] < INF else -1


def _lis_length(arr):
    import bisect
    tails = []
    for x in arr:
        i = bisect.bisect_left(tails, x)
        (tails.append(x) if i == len(tails) else tails.__setitem__(i, x))
    return len(tails)


_FAMILIES_HARD = {
    "sum_edit_distance": ("pairs", lambda ps: sum(_edit_distance(a, b) for a, b in ps)),
    "sum_lcs": ("pairs", lambda ps: sum(_lcs_length(a, b) for a, b in ps)),
    "count_makeable": ("coins", lambda coins, amts: sum(1 for a in amts if _coin_change(coins, a) != -1)),
    "max_lis": ("arrays", lambda arrs: max((_lis_length(a) for a in arrs), default=0)),
}
_TEXT_HARD = {
    "sum_edit_distance": "sum of the edit (Levenshtein) distances over a list of [a,b] string pairs. "
                         "needs an edit-distance routine.",
    "sum_lcs": "sum of the longest-common-subsequence lengths over a list of [a,b] string pairs. needs "
               "an LCS routine.",
    "count_makeable": "how many of the given amounts can be made exactly from the coin set (edges = "
                      "coins list, amounts list). needs a coin-change routine.",
    "max_lis": "the largest longest-increasing-subsequence length among a list of integer arrays. "
               "needs an LIS routine.",
}
_NEEDS_HARD = {"sum_edit_distance": {"edit_distance"}, "sum_lcs": {"lcs_length"},
               "count_makeable": {"coin_change"}, "max_lis": {"lis_length"}}
_REF_HARD = {
    "sum_edit_distance": "def sum_edit_distance(pairs):\n    return sum(edit_distance(a, b) for a, b in pairs)",
    "sum_lcs": "def sum_lcs(pairs):\n    return sum(lcs_length(a, b) for a, b in pairs)",
    "count_makeable": "def count_makeable(coins, amounts):\n    return sum(1 for a in amounts if coin_change(coins, a) != -1)",
    "max_lis": "def max_lis(arrays):\n    return max((lis_length(a) for a in arrays), default=0)",
}
_REF.update(_REF_HARD)
_NEEDS.update(_NEEDS_HARD)
_TEXT.update(_TEXT_HARD)


def _hard_static(name, inputs):
    _, oracle = _FAMILIES_HARD[name]
    asserts = [f"assert {name}({', '.join(repr(x) for x in args)}) == {oracle(*args)!r}" for args in inputs]
    return MBPPTask(name, _TEXT_HARD[name], asserts)


HARD_COMPOSE = [
    _hard_static("sum_edit_distance", [([["kitten", "sitting"], ["ab", "abc"]],), ([["cat", "cat"]],)]),
    _hard_static("sum_lcs", [([["abcde", "ace"], ["abc", "abc"]],), ([["xy", "yx"]],)]),
    _hard_static("count_makeable", [([3, 4], [6, 2, 7, 1]), ([2, 5], [4, 3, 10])]),
    _hard_static("max_lis", [([[3, 1, 2, 4], [5, 4, 3]],), ([[1, 2, 3]],)]),
]


def _rand_strs(rng):
    def one():
        return "".join(chr(97 + int(rng.integers(0, 5))) for _ in range(int(rng.integers(2, 7))))
    return [[one(), one()] for _ in range(int(rng.integers(2, 4)))]


def _rand_arrays(rng):
    return [[int(rng.integers(0, 9)) for _ in range(int(rng.integers(3, 7)))]
            for _ in range(int(rng.integers(2, 4)))]


def _rand_coins(rng):
    coins = sorted({int(rng.integers(2, 7)) for _ in range(int(rng.integers(2, 4)))})
    amounts = [int(rng.integers(1, 16)) for _ in range(int(rng.integers(3, 6)))]
    return coins, amounts


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


def _sample(kind, rng):
    """Random args (a TUPLE) for a family's fn, by input kind."""
    if kind == "list":
        return (_rand_list(rng),)
    if kind == "graph":
        return _rand_graph(rng)                 # (n, edges, s)
    if kind == "pairs":
        return (_rand_strs(rng),)
    if kind == "arrays":
        return (_rand_arrays(rng),)
    if kind == "coins":
        return _rand_coins(rng)                 # (coins, amounts)
    raise ValueError(kind)


def gen_compose_tasks(n: int = 200, seed: int = 0, hard: bool = False):
    """n parametrized compose-necessary instances (random inputs, oracle-computed asserts). Same fn
    names as the families (so retrieval/atoms line up); asserts sound by construction. hard=True uses
    the inline-FAILING DP families (edit_distance/lcs/coin_change/lis)."""
    fams = _FAMILIES_HARD if hard else _FAMILIES
    rng = np.random.default_rng(seed)
    names = list(fams)
    tasks = []
    for i in range(n):
        name = names[i % len(names)]
        kind, oracle = fams[name]
        asserts = []
        for _ in range(2):
            args = _sample(kind, rng)
            argstr = ", ".join(repr(x) for x in args)
            asserts.append(f"assert {name}({argstr}) == {oracle(*args)!r}")
        tasks.append(MBPPTask(name, _TEXT[name], asserts))
    return tasks


def seed_atom_graph(path: str, concept: str = "concept_algorithms", hard: bool = False):
    """Write a MemoryGraph pre-seeded with the verified atoms (bootstrap the candidate set). hard=True
    seeds the BIGGER easy+hard atom set (so retrieval must discriminate -> strategy actually matters)."""
    import json
    from pathlib import Path
    atoms = ALL_ATOMS if hard else ATOMS
    nodes = [{"id": concept, "text": "algorithms", "node_type": "concept"}]
    edges = []
    for name, (purpose, code) in atoms.items():
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
    print(f"  [3] generator: {len(gen)} parametrized instances across {len(_FAMILIES)} families, ALL "
          f"sound (oracle asserts pass the composing ref) -> PASS")

    # [4] HARD suite: inline-failing DP atoms verify; hard families sound + compose-necessary; generator
    for name, (_, code) in HARD_ATOMS.items():
        tests = {"edit_distance": ["assert edit_distance('kitten', 'sitting') == 3 and edit_distance('a','a') == 0"],
                 "lcs_length": ["assert lcs_length('abcde', 'ace') == 3 and lcs_length('xy','yx') == 1"],
                 "coin_change": ["assert coin_change([3,4], 6) == 2 and coin_change([3,4], 2) == -1"],
                 "lis_length": ["assert lis_length([3,1,2,4]) == 3 and lis_length([5,4,3]) == 1"]}[name]
        assert verify_asserts(code, tests), f"hard atom {name} failed"
    for t in HARD_COMPOSE:
        deps = "\n\n".join(HARD_ATOMS[a][1] for a in _NEEDS[t.name])
        assert t.verify(_REF[t.name], deps), f"hard {t.name}: ref+atoms must pass"
        assert not t.verify(_REF[t.name], ""), f"hard {t.name}: must NEED its atom"
    genh = gen_compose_tasks(n=40, seed=2, hard=True)
    for t in genh:
        deps = "\n\n".join(HARD_ATOMS[a][1] for a in _NEEDS[t.name])
        assert t.verify(_REF[t.name], deps), f"gen hard {t.name} unsound: {t.tests}"
    print(f"  [4] HARD suite: {len(HARD_ATOMS)} inline-failing DP atoms verify; {len(HARD_COMPOSE)} "
          f"families sound + compose-necessary; {len(genh)} generated instances sound -> PASS")

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
