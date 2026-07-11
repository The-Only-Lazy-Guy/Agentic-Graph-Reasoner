"""Hard, compositional, VERIFIABLE curriculum for training the artifact-graph proposer to synergize.

Per `v5/artifact_graph_design.md` §6 + the approved curriculum plan: the frozen 3B INLINES (monolithic
solutions, ~1 reuse). We train it to factor / call the library / compose. That needs HARD tasks whose
only tractable path is composition — "design an algorithm" — but still with a computable oracle.

Phase A (this file, step 1): a staged DAG of graph-algorithm tasks that SHARE latent sub-algorithms,
with a cold-unsolvable capstone (Dijkstra). The shared atoms are NEVER named in a task spec — if they
emerge as graph nodes and get reused, the graph invented its own vocabulary and compounding pays off.

  latent shared atoms (helpers a good solution factors, never a task by themselves):
    build_adj(n, edges)   neighbors(adj, u)   visited-set   a BFS/DFS traversal   a min-heap

  stage 0  leaves      neighbors_of · out_degree · edge_weight
  stage 1  2-hop       bfs_order · dfs_order · reachable_count           (reuse adjacency/neighbors)
  stage 2  deep        cc_count · topological_order · shortest_hops      (reuse a traversal)
  stage 3  CAPSTONE    dijkstra_dist                                     (reuse adjacency + heap)

Graphs are DIRECTED, weighted (weight in [1,9]); `edges` is a list of (u, v, w) triples, `n` nodes
labelled 0..n-1. Every task has ONE well-defined answer (deterministic tie-breaks) so verify-by-
execution is exact. Reuses `verify_fn` (tool_compose) at reward time (step 2).

  selftest (no model):  python -m v5.runtime.algo_curriculum --selftest
"""
from __future__ import annotations

import argparse
import hashlib
import heapq
import random
import re
import sys
from collections import deque
from dataclasses import dataclass

from v5.runtime.artifact_graph import (ArtifactGraph, _author_prompt, _calls_in, _causal_credit,
                                        _defs_in, _fingerprint)
from v5.runtime.tool_compose import verify_fn
from v5.runtime.tool_memory import _extract_code


# ═══════════════════════════════════════════════════════════════════════════════
# GRAPH GENERATION  (procedural, deterministic per seed)
# ═══════════════════════════════════════════════════════════════════════════════

def gen_graph(seed: int, n_lo: int = 4, n_hi: int = 7, wlo: int = 1, whi: int = 9,
              p: float = 0.35, dag: bool = False) -> tuple[int, list]:
    """Deterministic directed weighted graph. dag=True keeps only u<v edges (acyclic — a valid topo
    order exists); dag=False is general (often cyclic — exercises the 'no order' branch). No duplicate
    (u, v) pair (so edge_weight/out_degree are unambiguous)."""
    rng = random.Random(seed * 7919 + 13)
    n = rng.randint(n_lo, n_hi)
    edges = []
    for u in range(n):
        for v in range(n):
            if u == v or (dag and u >= v):
                continue
            if rng.random() < p:
                edges.append((u, v, rng.randint(wlo, whi)))
    return n, edges


def _build_adj(n: int, edges: list, undirected: bool = False) -> dict:
    adj = {i: [] for i in range(n)}
    for (u, v, w) in edges:
        adj[u].append((v, w))
        if undirected:
            adj[v].append((u, w))
    return adj


# ═══════════════════════════════════════════════════════════════════════════════
# ORACLES  (ground truth — reference implementations, pure Python)
# ═══════════════════════════════════════════════════════════════════════════════

def _o_neighbors(n, edges, u):
    return sorted({v for v, _ in _build_adj(n, edges)[u]})


def _o_out_degree(n, edges, u):
    return len(_o_neighbors(n, edges, u))


def _o_edge_weight(n, edges, u, v):
    ws = [w for (a, b, w) in edges if a == u and b == v]
    return min(ws) if ws else -1


def _o_bfs_order(n, edges, s):
    adj = _build_adj(n, edges)
    seen, q, order = {s}, deque([s]), []
    while q:
        u = q.popleft(); order.append(u)
        for v in sorted({x for x, _ in adj[u]}):
            if v not in seen:
                seen.add(v); q.append(v)
    return order


def _o_dfs_order(n, edges, s):
    adj = _build_adj(n, edges)
    seen, order = set(), []

    def dfs(u):
        seen.add(u); order.append(u)
        for v in sorted({x for x, _ in adj[u]}):
            if v not in seen:
                dfs(v)
    dfs(s)
    return order


def _o_reachable_count(n, edges, s):
    return len(_o_bfs_order(n, edges, s))


def _o_shortest_hops(n, edges, s, t):
    adj = _build_adj(n, edges)
    dist, q = {s: 0}, deque([s])
    while q:
        u = q.popleft()
        for v in {x for x, _ in adj[u]}:
            if v not in dist:
                dist[v] = dist[u] + 1; q.append(v)
    return dist.get(t, -1)


def _o_cc_count(n, edges):
    """Number of connected components treating edges as UNDIRECTED."""
    adj = _build_adj(n, edges, undirected=True)
    seen, c = set(), 0
    for s in range(n):
        if s in seen:
            continue
        c += 1; stack = [s]
        while stack:
            u = stack.pop()
            if u in seen:
                continue
            seen.add(u)
            for v, _ in adj[u]:
                if v not in seen:
                    stack.append(v)
    return c


def _o_topological_order(n, edges):
    """Lexicographically-smallest valid topological order (Kahn + min-heap), or [] if cyclic."""
    es = {(u, v) for (u, v, _) in edges}
    adj = {i: [] for i in range(n)}
    indeg = [0] * n
    for (u, v) in es:
        adj[u].append(v); indeg[v] += 1
    h = [i for i in range(n) if indeg[i] == 0]
    heapq.heapify(h)
    order = []
    while h:
        u = heapq.heappop(h); order.append(u)
        for v in sorted(adj[u]):
            indeg[v] -= 1
            if indeg[v] == 0:
                heapq.heappush(h, v)
    return order if len(order) == n else []


def _o_dijkstra_dist(n, edges, s):
    """Min weighted distance s->i for each i (list length n), -1 if unreachable. THE CAPSTONE."""
    adj = _build_adj(n, edges)
    INF = float("inf")
    dist = [INF] * n
    dist[s] = 0
    h = [(0, s)]
    while h:
        d, u = heapq.heappop(h)
        if d > dist[u]:
            continue
        for v, w in adj[u]:
            if d + w < dist[v]:
                dist[v] = d + w
                heapq.heappush(h, (d + w, v))
    return [dist[i] if dist[i] < INF else -1 for i in range(n)]


# ═══════════════════════════════════════════════════════════════════════════════
# TASK STREAM  (staged DAG; latent atoms shared across tasks, named in NO spec)
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class Task:
    name: str           # target function the model must write
    stage: int          # curriculum depth (0 leaves ... 3 capstone)
    family: str         # task family (for transfer / related-task grouping)
    text: str           # NL spec (mentions NO latent atom)
    oracle: object      # args-list -> expected
    gen: object         # seed -> args list
    atoms: tuple        # latent atoms a good solution factors (for docs/analysis, NOT given to model)


def _pick(seed, n, k=1):
    r = random.Random(seed * 31 + 7)
    return [r.randrange(n) for _ in range(k)]


STREAM = [
    # ── stage 0: leaves — natural solution factors build_adj / neighbors ────────────
    Task("neighbors_of", 0, "graph",
         "Write `neighbors_of(n, edges, u)`: `edges` is a list of (a,b,w) directed weighted edges over "
         "nodes 0..n-1. Return the sorted list of distinct nodes v such that an edge u->v exists.",
         lambda a: _o_neighbors(*a),
         lambda s: (lambda ne: [ne[0], ne[1], _pick(s, ne[0])[0]])(gen_graph(s)),
         ("build_adj", "neighbors")),
    Task("out_degree", 0, "graph",
         "Write `out_degree(n, edges, u)`: number of DISTINCT nodes reachable from u by a single edge "
         "u->v (edges = list of (a,b,w)).",
         lambda a: _o_out_degree(*a),
         lambda s: (lambda ne: [ne[0], ne[1], _pick(s + 1, ne[0])[0]])(gen_graph(s)),
         ("build_adj", "neighbors")),
    Task("edge_weight", 0, "graph",
         "Write `edge_weight(n, edges, u, v)`: the weight of the edge u->v (edges = list of (a,b,w)), "
         "or -1 if there is no such edge.",
         lambda a: _o_edge_weight(*a),
         lambda s: (lambda ne: [ne[0], ne[1]] + _pick(s + 2, ne[0], 2))(gen_graph(s)),
         ("build_adj",)),

    # ── stage 1: 2-hop — reuse adjacency/neighbors + a traversal ────────────────────
    Task("bfs_order", 1, "graph",
         "Write `bfs_order(n, edges, s)`: the breadth-first-search visit order from node s over the "
         "directed graph (edges = list of (a,b,w)). Break ties by ascending node id. Return the list "
         "of nodes in visit order.",
         lambda a: _o_bfs_order(*a),
         lambda s: (lambda ne: [ne[0], ne[1], _pick(s + 3, ne[0])[0]])(gen_graph(s)),
         ("build_adj", "neighbors", "visited", "bfs")),
    Task("dfs_order", 1, "graph",
         "Write `dfs_order(n, edges, s)`: the depth-first-search PRE-order from node s (edges = list of "
         "(a,b,w)), visiting smaller node ids first. Return the list of nodes in visit order.",
         lambda a: _o_dfs_order(*a),
         lambda s: (lambda ne: [ne[0], ne[1], _pick(s + 4, ne[0])[0]])(gen_graph(s)),
         ("build_adj", "neighbors", "visited", "dfs")),
    Task("reachable_count", 1, "graph",
         "Write `reachable_count(n, edges, s)`: how many nodes are reachable FROM s (including s) in "
         "the directed graph (edges = list of (a,b,w)).",
         lambda a: _o_reachable_count(*a),
         lambda s: (lambda ne: [ne[0], ne[1], _pick(s + 5, ne[0])[0]])(gen_graph(s)),
         ("build_adj", "neighbors", "visited", "bfs")),

    # ── stage 2: deep — reuse a traversal / indegree ────────────────────────────────
    Task("cc_count", 2, "graph",
         "Write `cc_count(n, edges)`: the number of connected components, treating every edge (a,b,w) "
         "as UNDIRECTED.",
         lambda a: _o_cc_count(*a),
         lambda s: list(gen_graph(s)),
         ("build_adj", "visited", "dfs")),
    Task("topological_order", 2, "graph",
         "Write `topological_order(n, edges)`: a valid topological ordering of the directed graph "
         "(edges = list of (a,b,w)); among valid orders return the lexicographically smallest. If the "
         "graph has a cycle, return [].",
         lambda a: _o_topological_order(*a),
         lambda s: list(gen_graph(s, dag=(s % 2 == 0))),
         ("build_adj", "indegree", "heap")),
    Task("shortest_hops", 2, "graph",
         "Write `shortest_hops(n, edges, s, t)`: the minimum number of edges on a path from s to t in "
         "the directed graph (edges = list of (a,b,w)), or -1 if t is unreachable.",
         lambda a: _o_shortest_hops(*a),
         lambda s: (lambda ne: [ne[0], ne[1]] + _pick(s + 6, ne[0], 2))(gen_graph(s)),
         ("build_adj", "neighbors", "visited", "bfs")),

    # ── stage 3: CAPSTONE — cold-unsolvable one-shot, composable from adjacency + heap ─
    Task("dijkstra_dist", 3, "graph",
         "Write `dijkstra_dist(n, edges, s)`: for every node i in 0..n-1 return the minimum total "
         "weight of a path from s to i (edges = list of (a,b,w), weights >= 1), or -1 if i is "
         "unreachable. Return the list of n distances.",
         lambda a: _o_dijkstra_dist(*a),
         lambda s: (lambda ne: [ne[0], ne[1], _pick(s + 7, ne[0])[0]])(gen_graph(s)),
         ("build_adj", "heap", "relax")),
]

BY_NAME = {t.name: t for t in STREAM}
STAGES = sorted({t.stage for t in STREAM})


def cases(task: Task, seeds) -> list:
    """[(args, expected)] for verify-by-execution."""
    return [(task.gen(s), task.oracle(task.gen(s))) for s in seeds]


# ═══════════════════════════════════════════════════════════════════════════════
# TASK FINGERPRINT  — identifies the task FUNCTION (same across seeds, distinct across tasks)
# so amortization counts reuse across BEHAVIORALLY-DISTINCT tasks (anti-memorization).
# ═══════════════════════════════════════════════════════════════════════════════

_TFP_SEEDS = range(900, 912)


def task_fingerprint(task: Task) -> str:
    outs = []
    for s in _TFP_SEEDS:
        try:
            outs.append(repr(task.oracle(task.gen(s))))
        except Exception:
            outs.append("ERR")
    return hashlib.md5("\t".join(outs).encode()).hexdigest()[:12]


# ═══════════════════════════════════════════════════════════════════════════════
# REWARD  (§4 of the plan, + the design-review compression term: reuse != value)
#   R = verified·( w_c + w_r·Σ(1+amort) + w_m·compression + w_f·good_helpers
#                  − w_s·inert_helpers − w_len·length )
# amort  = reuse BREADTH (distinct task fps).           compression = lines SAVED by reusing atoms
# good   = novel helper CALLED by the target.           inert = helper uncalled OR re-implements an
#                                                               existing node (over-specific / memorized)
# ═══════════════════════════════════════════════════════════════════════════════

REWARD_W = dict(correct=1.0, reuse=0.4, compress=0.5, factor=0.3, inert=0.5, length=0.03)
_LEN_NORM = 400.0


def reward(verified: bool, amorts: list, compress_lines: int, good_helpers: int,
           inert_helpers: int, code_len: int, W=REWARD_W):
    if not verified:
        return 0.0, dict(verified=False)
    R = (W["correct"]
         + W["reuse"] * sum(1 + a for a in amorts)
         + W["compress"] * (compress_lines / _LEN_NORM)
         + W["factor"] * good_helpers
         - W["inert"] * inert_helpers
         - W["length"] * (code_len / _LEN_NORM))
    return R, dict(verified=True, reuse=[1 + a for a in amorts], compress=compress_lines,
                   good=good_helpers, inert=inert_helpers, length=code_len, R=round(R, 3))


def score_rollout(graph: ArtifactGraph, task: Task, code: str, adv_names: list, ecases: list):
    """Compute (reward, breakdown, verified, causal_reused, sol_defs) for one solution."""
    sol_defs = _defs_in(code)
    called = [a for a in adv_names
              if a not in sol_defs and re.search(rf"\b{re.escape(a)}\s*\(", code)]
    acc, _, _ = verify_fn(code, task.name, ecases, graph.deps_code(called))
    if acc < 0.999:
        R, bd = reward(False, [], 0, 0, 0, len(code))
        return R, bd, False, [], sol_defs
    causal = _causal_credit(graph, code, task.name, ecases, called)
    amorts = [graph.amort(a) for a in causal]
    compress = sum(len(graph.arts[a].code.splitlines()) for a in causal)   # lines saved by reuse
    good = inert = 0
    tgt_src = sol_defs.get(task.name, "")
    for d, src in sol_defs.items():
        if d == task.name:
            continue
        called_by_target = bool(re.search(rf"\b{re.escape(d)}\s*\(", tgt_src))
        fp = _fingerprint(code, d)
        is_dup = fp is not None and any(a.fp == fp for a in graph.arts.values())
        if called_by_target and not is_dup:
            good += 1                     # a genuine new sub-routine
        else:
            inert += 1                    # uncalled, or re-implements an existing atom (penalized)
    R, bd = reward(True, amorts, compress, good, inert, len(code))
    return R, bd, True, causal, sol_defs


def solve_curriculum_task(graph: ArtifactGraph, gen_fn, task: Task, born: int, tfp: str,
                          vseeds, eseeds, k: int = 32, samples: int = 1, select: str = "verify"):
    """Retrieve -> author `samples` candidates -> SELECT -> score -> credit(by task fp)+register.

    select="verify": pick the highest-verifying candidate (baseline — no reuse preference).
    select="reward": among the candidates that VERIFY, pick the highest-REWARD one (reuse/compression
    preference) — this is the best-of-N gate: does the model already PRODUCE reuse solutions among N
    samples (so RL can amplify them), or never (so RL won't conjure them)?"""
    vcases, ecases = cases(task, vseeds), cases(task, eseeds)
    advertised = graph.retrieve(task.text, k=k)
    adv_names = [a.name for a in advertised]
    cands = []
    for gen in gen_fn([_author_prompt(task, advertised, None, [])] * samples):
        code = _extract_code(gen)
        called = [a for a in adv_names
                  if a not in _defs_in(code) and re.search(rf"\b{re.escape(a)}\s*\(", code)]
        vacc, _, _ = verify_fn(code, task.name, vcases, graph.deps_code(called))
        cands.append((code, vacc))
    if not cands:
        code = ""
    elif select == "reward":
        pool = [c for c, va in cands if va >= 0.999] or [c for c, _ in cands]
        code = max(pool, key=lambda c: score_rollout(graph, task, c, adv_names, ecases)[0])
    else:
        code = max(cands, key=lambda cv: cv[1])[0]
    R, bd, vr, causal, sol_defs = score_rollout(graph, task, code, adv_names, ecases)
    if vr:
        for a in causal:
            graph.credit(a, tfp)                       # credit by TASK FINGERPRINT -> amort breadth
        universe = set(sol_defs) | set(graph.arts)
        for name, src in sol_defs.items():
            graph.register(name, src, task.text[:60], born, _calls_in(src, universe, name))
    return dict(name=task.name, stage=task.stage, verified=vr, reward=round(R, 3),
                causal=causal, breakdown=bd, code=code)


def run_curriculum(gen_fn, stream=STREAM, verify_n=6, eval_n=10, k=32, samples=1, select="verify",
                   graph=None):
    graph = graph or ArtifactGraph()
    vseeds, eseeds = range(300, 300 + verify_n), range(700, 700 + eval_n)
    log = []
    for i, task in enumerate(stream):
        log.append(solve_curriculum_task(graph, gen_fn, task, i, task_fingerprint(task),
                                          vseeds, eseeds, k=k, samples=samples, select=select))
    return graph, log


def best_of_n_gate(gen_fn, n=8, stream=STREAM, verify_n=6, eval_n=10):
    """The pre-RL gate. Run the curriculum twice with best-of-`n` sampling: arm 'verify' selects the
    highest-verifying candidate (baseline); arm 'reward' selects the highest-REWARD verified candidate
    (reuse preference). If 'reward' raises causal reuse + amortization + capstone reachability at
    EQUAL correctness, the behavior is learnable -> RL is worth the GPU. If not, a bigger base is
    needed (project-model-strategy)."""
    out = {}
    for mode in ("verify", "reward"):
        graph, log = run_curriculum(gen_fn, stream=stream, verify_n=verify_n, eval_n=eval_n,
                                    samples=n, select=mode)
        cap = next((r for r in log if r["name"] == "dijkstra_dist"), None)
        out[mode] = dict(
            solved=sum(1 for r in log if r["verified"]), total=len(log),
            reusers=sum(1 for r in log if r["causal"]),
            amort=graph.amort("build_adj") if "build_adj" in graph.arts else 0,
            capstone_composed=bool(cap and cap["verified"] and cap["causal"]),
            mean_reward=round(sum(r["reward"] for r in log) / max(1, len(log)), 2),
            log=log)
    return out


# ═══════════════════════════════════════════════════════════════════════════════
# STUB LM (no GPU) — a COMPETENT proposer: factors `build_adj` and CALLS it when advertised.
# Proves the curriculum is composable + drives the reward selftest. (real LM tested on molab.)
# ═══════════════════════════════════════════════════════════════════════════════

_BUILD_ADJ = ("def build_adj(n, edges):\n    adj = {i: [] for i in range(n)}\n"
              "    for a, b, w in edges:\n        adj[a].append((b, w))\n    return adj")

_STUB_BODY = {
    "neighbors_of": (True, "def neighbors_of(n, edges, u):\n    adj = build_adj(n, edges)\n"
                           "    return sorted({v for v, _ in adj[u]})"),
    "out_degree": (True, "def out_degree(n, edges, u):\n    adj = build_adj(n, edges)\n"
                         "    return len({v for v, _ in adj[u]})"),
    "edge_weight": (True, "def edge_weight(n, edges, u, v):\n    adj = build_adj(n, edges)\n"
                          "    ws = [w for x, w in adj[u] if x == v]\n    return min(ws) if ws else -1"),
    "bfs_order": (True, "def bfs_order(n, edges, s):\n    from collections import deque\n"
                        "    adj = build_adj(n, edges)\n    seen = {s}; q = deque([s]); order = []\n"
                        "    while q:\n        u = q.popleft(); order.append(u)\n"
                        "        for v in sorted({x for x, _ in adj[u]}):\n"
                        "            if v not in seen:\n                seen.add(v); q.append(v)\n"
                        "    return order"),
    "dfs_order": (True, "def dfs_order(n, edges, s):\n    adj = build_adj(n, edges)\n"
                        "    seen = set(); order = []\n    def go(u):\n        seen.add(u); order.append(u)\n"
                        "        for v in sorted({x for x, _ in adj[u]}):\n"
                        "            if v not in seen:\n                go(v)\n    go(s)\n    return order"),
    "reachable_count": (True, "def reachable_count(n, edges, s):\n    from collections import deque\n"
                              "    adj = build_adj(n, edges)\n    seen = {s}; q = deque([s])\n"
                              "    while q:\n        u = q.popleft()\n        for v in {x for x, _ in adj[u]}:\n"
                              "            if v not in seen:\n                seen.add(v); q.append(v)\n"
                              "    return len(seen)"),
    "cc_count": (False, "def cc_count(n, edges):\n    adj = {i: [] for i in range(n)}\n"
                        "    for a, b, w in edges:\n        adj[a].append(b); adj[b].append(a)\n"
                        "    seen = set(); c = 0\n    for s in range(n):\n        if s in seen:\n            continue\n"
                        "        c += 1; st = [s]\n        while st:\n            u = st.pop()\n"
                        "            if u in seen:\n                continue\n            seen.add(u)\n"
                        "            for v in adj[u]:\n                if v not in seen:\n                    st.append(v)\n"
                        "    return c"),
    "topological_order": (True, "def topological_order(n, edges):\n    import heapq\n"
                                "    adj = build_adj(n, edges)\n    indeg = [0] * n\n"
                                "    for u in range(n):\n        for v, _ in adj[u]:\n            indeg[v] += 1\n"
                                "    h = [i for i in range(n) if indeg[i] == 0]; heapq.heapify(h); order = []\n"
                                "    while h:\n        u = heapq.heappop(h); order.append(u)\n"
                                "        for v in sorted({x for x, _ in adj[u]}):\n            indeg[v] -= 1\n"
                                "            if indeg[v] == 0:\n                heapq.heappush(h, v)\n"
                                "    return order if len(order) == n else []"),
    "shortest_hops": (True, "def shortest_hops(n, edges, s, t):\n    from collections import deque\n"
                            "    adj = build_adj(n, edges)\n    dist = {s: 0}; q = deque([s])\n"
                            "    while q:\n        u = q.popleft()\n        for v in {x for x, _ in adj[u]}:\n"
                            "            if v not in dist:\n                dist[v] = dist[u] + 1; q.append(v)\n"
                            "    return dist.get(t, -1)"),
    "dijkstra_dist": (True, "def dijkstra_dist(n, edges, s):\n    import heapq\n"
                            "    adj = build_adj(n, edges)\n    INF = float('inf')\n    dist = [INF] * n\n"
                            "    dist[s] = 0; h = [(0, s)]\n    while h:\n        d, u = heapq.heappop(h)\n"
                            "        if d > dist[u]:\n            continue\n        for v, w in adj[u]:\n"
                            "            if d + w < dist[v]:\n                dist[v] = d + w; heapq.heappush(h, (d + w, v))\n"
                            "    return [x if x < INF else -1 for x in dist]"),
}


def _stub_gen(prompts: list) -> list:
    out = []
    for p in prompts:
        name = re.findall(r"Write `([a-z_][a-z0-9_]*)\(", p)[-1]
        needs, body = _STUB_BODY[name]
        block = ""
        if "already DEFINED" in p:
            block = p.split("already DEFINED", 1)[1].split("If you need a sub-computation", 1)[0]
        pieces = []
        if needs and not re.search(r"\bbuild_adj\s*\(", block):
            pieces.append(_BUILD_ADJ)                 # define inline (first use — not yet advertised)
        pieces.append(body)
        out.append("```python\n" + "\n\n".join(pieces) + "\n```")
    return out


# ═══════════════════════════════════════════════════════════════════════════════
# SELFTEST (no model) — oracle correctness + staged structure
# ═══════════════════════════════════════════════════════════════════════════════

def _selftest() -> bool:
    print("algo_curriculum --selftest: oracles + staged DAG (no model)\n")

    # [1] oracles on a hand-checked graph: 0->1(2) 0->2(5) 1->2(1) 2->3(3)
    n, E = 4, [(0, 1, 2), (0, 2, 5), (1, 2, 1), (2, 3, 3)]
    assert _o_neighbors(n, E, 0) == [1, 2], _o_neighbors(n, E, 0)
    assert _o_out_degree(n, E, 0) == 2
    assert _o_edge_weight(n, E, 0, 2) == 5 and _o_edge_weight(n, E, 0, 3) == -1
    assert _o_bfs_order(n, E, 0) == [0, 1, 2, 3], _o_bfs_order(n, E, 0)
    assert _o_dfs_order(n, E, 0) == [0, 1, 2, 3], _o_dfs_order(n, E, 0)
    assert _o_reachable_count(n, E, 0) == 4
    assert _o_shortest_hops(n, E, 0, 3) == 2 and _o_shortest_hops(n, E, 3, 0) == -1
    assert _o_cc_count(n, E) == 1
    assert _o_topological_order(n, E) == [0, 1, 2, 3], _o_topological_order(n, E)
    assert _o_dijkstra_dist(n, E, 0) == [0, 2, 3, 6], _o_dijkstra_dist(n, E, 0)
    print("  [1] all oracles correct on the hand-checked 4-node graph -> PASS")

    # [2] cyclic graph -> topo returns []; unreachable -> dijkstra -1
    assert _o_topological_order(3, [(0, 1, 1), (1, 2, 1), (2, 0, 1)]) == []
    assert _o_dijkstra_dist(3, [(0, 1, 4)], 0) == [0, 4, -1]
    print("  [2] cyclic -> topo []; unreachable -> dijkstra -1 -> PASS")

    # [3] generator deterministic + no duplicate (u,v) pair + valid node range
    g1, g2 = gen_graph(42), gen_graph(42)
    assert g1 == g2, "gen must be deterministic"
    n3, E3 = gen_graph(42)
    seen = set()
    for (u, v, w) in E3:
        assert 0 <= u < n3 and 0 <= v < n3 and u != v and 1 <= w <= 9
        assert (u, v) not in seen, "duplicate (u,v) edge"
        seen.add((u, v))
    dagn, dage = gen_graph(8, dag=True)
    assert all(u < v for (u, v, _) in dage) and _o_topological_order(dagn, dage) != []
    print(f"  [3] gen deterministic, dedup edges, dag=True acyclic -> PASS")

    # [4] staged DAG structure: stages 0..3 present, capstone at stage 3, atoms shared across tasks
    assert STAGES == [0, 1, 2, 3], STAGES
    assert [t.name for t in STREAM if t.stage == 3] == ["dijkstra_dist"], "one capstone"
    from collections import Counter
    atom_use = Counter(a for t in STREAM for a in t.atoms)
    shared = [a for a, c in atom_use.items() if c >= 3]
    assert {"build_adj", "visited", "neighbors"} <= set(shared), f"atoms not shared enough: {atom_use}"
    print(f"  [4] stages {STAGES}, capstone=dijkstra_dist, shared atoms {sorted(shared)} -> PASS")

    # [5] cases() shape: args + expected, and every task's oracle runs on 8 seeds without error
    for t in STREAM:
        cs = cases(t, range(100, 108))
        assert len(cs) == 8 and all(len(c) == 2 for c in cs)
    print("  [5] cases() build for all tasks over 8 seeds -> PASS")

    # [6] REWARD shape: broad amort > narrow; compression (reuse!=value); good helper > inert; gate
    r_broad, _ = reward(True, [3], 4, 0, 0, 120)
    r_narrow, _ = reward(True, [1], 4, 0, 0, 120)
    assert r_broad > r_narrow, "broadly-amortized reuse must score higher"
    r_big, _ = reward(True, [1], 12, 0, 0, 120)      # rare-but-BIG atom (high compression)
    r_small, _ = reward(True, [1], 1, 0, 0, 120)     # same reuse breadth, tiny atom
    assert r_big > r_small, "compression term must value a big atom more (reuse != value)"
    r_good, _ = reward(True, [], 0, 1, 0, 120)
    r_inert, _ = reward(True, [], 0, 0, 1, 120)
    assert r_good > r_inert, "novel called helper must beat an inert/duplicate one"
    assert reward(False, [3], 9, 1, 0, 120)[0] == 0.0, "unverified -> 0 (correctness is the gate)"
    print("  [6] reward: broad>narrow amort, compression values big atoms, good>inert, gate -> PASS")

    # [7] anti-memorization: crediting the SAME task fp N times -> amort 1; N DISTINCT -> N
    g = ArtifactGraph(); g.register("build_adj", _BUILD_ADJ, "", 0, [])
    for _ in range(5):
        g.credit("build_adj", "same_task")
    assert g.amort("build_adj") == 1, "same task reused 5x must NOT inflate amortization"
    for i in range(5):
        g.credit("build_adj", f"task_{i}")
    assert g.amort("build_adj") == 6, "5 distinct tasks -> amort 6"
    print("  [7] amortization counts DISTINCT tasks only (memorizing 1 task -> amort 1) -> PASS")

    # [8] STUB curriculum run: hard staged DAG solved end-to-end, build_adj EMERGES + amortizes,
    # capstone Dijkstra solved BY COMPOSITION (reuses build_adj) -> compounding is real
    graph, log = run_curriculum(_stub_gen, verify_n=6, eval_n=10)
    unsolved = [r["name"] for r in log if not r["verified"]]
    assert not unsolved, f"stub should solve the whole DAG, missed {unsolved}"
    assert "build_adj" in graph.arts, "the shared atom must emerge as ONE node (never named in a spec)"
    am = graph.amort("build_adj")
    assert am >= 5, f"build_adj must amortize across many distinct tasks, got {am}"
    cap = next(r for r in log if r["name"] == "dijkstra_dist")
    assert cap["verified"] and "build_adj" in cap["causal"], f"capstone must compose build_adj: {cap}"
    reusers = [r["name"] for r in log if r["causal"]]
    print(f"  [8] stub run 9/9 | build_adj amort={am} across {reusers}")
    print(f"      capstone dijkstra_dist solved by REUSING build_adj -> compounding -> PASS")

    # [9] over-specific / memorization PENALTY: on a task where build_adj is advertised, REUSING it
    # scores strictly higher than RE-IMPLEMENTING it inline (inert dup + no reuse/compression credit)
    g2 = ArtifactGraph(); g2.register("build_adj", _BUILD_ADJ, "adjacency", 0, [])
    g2.credit("build_adj", "t_a"); g2.credit("build_adj", "t_b")
    ec = cases(BY_NAME["bfs_order"], range(700, 710))
    reuse_code = _STUB_BODY["bfs_order"][1]                       # calls build_adj (from deps)
    inline_code = _BUILD_ADJ + "\n\n" + _STUB_BODY["bfs_order"][1]  # re-implements build_adj inline
    R_reuse = score_rollout(g2, BY_NAME["bfs_order"], reuse_code, ["build_adj"], ec)[0]
    R_inline = score_rollout(g2, BY_NAME["bfs_order"], inline_code, ["build_adj"], ec)[0]
    assert R_reuse > R_inline, f"reuse {R_reuse:.2f} must beat re-implement {R_inline:.2f}"
    print(f"  [9] reuse ({R_reuse:.2f}) > re-implement inline ({R_inline:.2f}) — over-specific penalized -> PASS")

    # [10] BEST-OF-N selection: given candidates {inline, reuse}, select='reward' surfaces the REUSE
    # solution; select='verify' does not (both verify) — the mechanism the pre-RL gate relies on
    def _mock_two(prompts):
        return ["```python\n" + inline_code + "\n```", "```python\n" + reuse_code + "\n```"]
    vs, es = range(300, 306), range(700, 710)
    gr = ArtifactGraph(); gr.register("build_adj", _BUILD_ADJ, "adjacency", 0, [])
    r_rw = solve_curriculum_task(gr, _mock_two, BY_NAME["bfs_order"], 1, "tfp", vs, es,
                                 samples=2, select="reward")
    gv = ArtifactGraph(); gv.register("build_adj", _BUILD_ADJ, "adjacency", 0, [])
    r_vf = solve_curriculum_task(gv, _mock_two, BY_NAME["bfs_order"], 1, "tfp", vs, es,
                                 samples=2, select="verify")
    assert r_rw["causal"] == ["build_adj"], f"reward-select must pick the reuse candidate: {r_rw}"
    assert r_vf["causal"] == [], f"verify-select has no reuse preference: {r_vf}"
    print("  [10] best-of-N: select='reward' surfaces reuse, select='verify' does not -> PASS")

    print("\n  ALGO_CURRICULUM SELFTEST -> PASS")
    return True


def main():
    ap = argparse.ArgumentParser(description="Hard compositional curriculum (graph algorithms).")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--run-stub", action="store_true",
                    help="drive the curriculum with the competent StubLM (no GPU) and print the "
                         "reward/amortization trace — the compounding demonstration")
    ap.add_argument("--gate", action="store_true",
                    help="best-of-N reuse-preference GATE with a real LM (molab): verify- vs reward-"
                         "selection on causal reuse / amortization / capstone. Run before RL.")
    ap.add_argument("--model", default="Qwen/Qwen2.5-3B")
    ap.add_argument("--n", type=int, default=8, help="best-of-N samples per task for --gate")
    ap.add_argument("--chunk", type=int, default=8)
    a = ap.parse_args()
    if a.selftest:
        sys.exit(0 if _selftest() else 1)
    if a.run_stub:
        graph, log = run_curriculum(_stub_gen)
        print("STUB CURRICULUM (no GPU) — reward + reuse trace\n")
        for r in log:
            print(f"  [s{r['stage']}] {r['name']:20} {'OK ' if r['verified'] else 'FAIL'} "
                  f"reward={r['reward']:+.2f}  reuse={r['causal'] or '-'}")
        print(f"\n  build_adj amortization (distinct tasks reusing it): {graph.amort('build_adj')}")
        print(f"  graph nodes: {sorted(graph.arts)}")
        return
    if a.gate:
        from v5.runtime.artifact_graph import _real_gen_fn
        gen_fn = _real_gen_fn(a.model, a.chunk)
        res = best_of_n_gate(gen_fn, n=a.n)
        print(f"\nBEST-OF-{a.n} GATE ({a.model}) — verify-select vs reward-select\n", file=sys.stderr)
        for mode in ("verify", "reward"):
            r = res[mode]
            print(f"  [{mode:6}] solved {r['solved']}/{r['total']} | reusers {r['reusers']} | "
                  f"build_adj amort {r['amort']} | capstone-composed {r['capstone_composed']} | "
                  f"mean_reward {r['mean_reward']}", file=sys.stderr)
        v, w = res["verify"], res["reward"]
        gain = w["reusers"] - v["reusers"]
        print(f"\n  GATE: reward-select adds {gain:+d} reusers at {w['solved']}/{v['solved']} solved "
              f"(equal correctness). Capstone composed: verify={v['capstone_composed']} "
              f"reward={w['capstone_composed']}.", file=sys.stderr)
        print(f"  => {'PASS (behavior is learnable — RL worth it)' if gain > 0 and w['capstone_composed'] else 'WEAK (model rarely produces reuse in N samples — bigger base?)'}",
              file=sys.stderr)
        return
    print("use --selftest (no GPU) · --run-stub (compounding demo) · --gate (molab pre-RL gate).")


if __name__ == "__main__":
    main()
