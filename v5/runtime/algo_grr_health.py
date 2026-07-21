"""algo_grr_health — graph-health / pollution monitor for the scale-up run (GRR-Tool S3).

As the graph grows over a long stream, pollution can creep in even with the store-gate: the nid-collision
check stops SAME-NAME re-banks, but a helper banked under a DIFFERENT name with the SAME behaviour is a
behavioural duplicate (the class the GRR-1 fuzz-equivalence bar exists to catch). This reports the health
signals to watch during scale-up:
  - exact code duplicates (same source, different node),
  - BEHAVIOURAL duplicates (fuzz-equal: same outputs on shared random inputs) — real pollution,
  - orphan atoms (no part_of/concept),
  - dangling edges (depend/part_of target missing),
  - (optional) dead atoms (banked but never reused), given a reuse set.

    selftest (no GPU):  python -m v5.runtime.algo_grr_health --selftest
    report a graph:     python -m v5.runtime.algo_grr_health --graph graphs/grr_seed_clean.json
"""
from __future__ import annotations

import argparse
import ast
import random
import re
import sys
from collections import defaultdict
from pathlib import Path

_ROOT = str(Path(__file__).resolve().parents[2])
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from graph_core import MemoryGraph  # type: ignore  # noqa: E402
from v5.runtime.algo_grr_membrane import run_with_timeout, realize_closure_code  # noqa: E402


def _norm(code: str) -> str:
    return re.sub(r"\s+", " ", (code or "").strip())


def _arity(code: str) -> int:
    try:
        fn = next(n for n in ast.walk(ast.parse(code)) if isinstance(n, ast.FunctionDef))
        return len(fn.args.posonlyargs) + len(fn.args.args)
    except Exception:  # noqa: BLE001
        return -1


def _safe_call(fn, args):
    def _c():
        try:
            return repr(fn(*args))
        except Exception:  # noqa: BLE001
            return "ERR"
    return run_with_timeout(_c, seconds=1.0, default="ERR")


def _behavior_sig(closure_code: str, entry: str, arity: int, input_set: list) -> tuple | None:
    """Signature = outputs on a SHARED FIXED input set (so same-arity atoms are comparable). `closure_code`
    is the atom PLUS its transitive dependency closure so composed atoms resolve. Two atoms with the same
    signature are behaviourally indistinguishable on this probe = candidate duplicates."""
    ns: dict = {}
    try:
        exec(compile(closure_code, "<sig>", "exec"), ns)
        fn = ns.get(entry)
        if not callable(fn) or arity < 1:
            return None
    except Exception:  # noqa: BLE001
        return None
    outs = [_safe_call(fn, args) for args in input_set]
    ok = [o for o in outs if o != "ERR"]
    # only a DISCRIMINATING fingerprint (>=2 distinct outputs) is usable: a constant-on-probe atom
    # (e.g. is_perfect / is_palindrome_number are ~always-False on random inputs) can't be told apart
    # from other constant-on-probe atoms and must not be flagged as a duplicate of them.
    if len(set(ok)) < 2:
        return None
    return (arity, tuple(outs))


def graph_health(graph: MemoryGraph, reused: set | None = None) -> dict:
    impl = [nid for nid, n in graph.nodes.items() if n.node_type == "implementation"]
    ids = set(graph.nodes)

    by_code: dict[str, list] = defaultdict(list)
    for nid in impl:
        by_code[_norm(graph.nodes[nid].metadata.get("code", ""))].append(nid)
    exact = [g for g in by_code.values() if len(g) > 1]

    rng = random.Random(0)                                    # FIXED input set per arity -> comparable sigs.
    # wide multi-digit range so digit-ops diverge (digit_sum vs reverse_digits agree on 1-digit inputs).
    inputs = {ar: [tuple(rng.randint(2, 9999) for _ in range(ar)) for _ in range(10)] for ar in (1, 2, 3)}
    ent = {nid: graph.nodes[nid].metadata.get("entry", nid) for nid in impl}
    by_sig: dict[tuple, list] = defaultdict(list)
    for nid in impl:
        code = graph.nodes[nid].metadata.get("code", "")
        ar = _arity(code)
        closure = realize_closure_code(graph, [nid])          # atom + transitive deps -> composed atoms resolve
        sig = _behavior_sig(closure or code, ent[nid], ar, inputs.get(ar, []))
        if sig is not None:
            by_sig[sig].append(nid)
    behav = [g for g in by_sig.values() if len(g) > 1]

    part_of_src = {e.src for e in graph.edges if e.relation == "part_of"}
    orphan = [nid for nid in impl if nid not in part_of_src]
    dangling = [f"{e.src}->{e.dst}" for e in graph.edges if e.src not in ids or e.dst not in ids]
    dead = [nid for nid in impl if reused is not None and nid not in reused
            and graph.nodes[nid].metadata.get("origin") == "derived"]

    return {"atoms": len(impl), "exact_dups": exact, "behav_dups": behav,
            "orphans": orphan, "dangling": dangling, "dead": dead}


# ═══════════════════════════════════════════════════════════════════════════════
# PruningMonitor — utility-based pruning with decay (v6)
# ═══════════════════════════════════════════════════════════════════════════════

class PruningMonitor:
    """Tracks node utility over time and identifies pruning candidates.
    Utility = weighted combination of access_count, confidence, and importance.
    Applies decay to unused nodes so 'trivial knowledge' naturally fades.

    Usage:
        pm = PruningMonitor(graph)
        pm.apply_decay()
        candidates = pm.find_prune_candidates(threshold=0.1)
    """

    def __init__(self, graph: MemoryGraph,
                 confidence_weight: float = 0.35,
                 access_weight: float = 0.35,
                 importance_weight: float = 0.30,
                 decay_rate: float = 0.02):
        self.graph = graph
        self.w_conf = confidence_weight
        self.w_access = access_weight
        self.w_import = importance_weight
        self.decay_rate = decay_rate

    def utility(self, nid: str) -> float:
        """Compute utility score for a single node."""
        n = self.graph.nodes.get(nid)
        if n is None:
            return 0.0
        if n.node_type in ("hub", "concept"):
            return 1.0
        max_access = max((nd.access_count for nd in self.graph.nodes.values()), default=1)
        norm_access = n.access_count / max(max_access, 1)
        return (self.w_conf * n.confidence +
                self.w_access * norm_access +
                self.w_import * n.importance)

    def apply_decay(self, exclude_types: set | None = None) -> None:
        """Apply time-decay to every node's confidence and access_count.
        Hub/concept/trap nodes are preserved by default."""
        exclude = exclude_types or {"hub", "concept", "trap"}
        for nid, n in list(self.graph.nodes.items()):
            if n.node_type in exclude:
                continue
            n.confidence = max(0.05, n.confidence * (1.0 - self.decay_rate))
            if n.access_count > 0 and not n.metadata.get("last_success", ""):
                n.access_count = max(0, n.access_count - 1)

    def find_prune_candidates(self, threshold: float = 0.08,
                              min_access: int = 0,
                              exclude_types: set | None = None) -> list[str]:
        """Return node ids with utility below threshold.
        exclude_types: node types that are never pruned (default: hub, concept, trap)."""
        exclude = exclude_types or {"hub", "concept", "trap"}
        candidates = []
        for nid in list(self.graph.nodes):
            n = self.graph.nodes[nid]
            if n.node_type in exclude:
                continue
            if n.access_count <= min_access and self.utility(nid) < threshold:
                candidates.append(nid)
        return candidates

    def scores_report(self, top_n: int = 10) -> list[tuple[str, float]]:
        """Return (nid, utility) pairs sorted ascending (lowest first) for inspection."""
        scores = [(nid, self.utility(nid)) for nid in self.graph.nodes]
        scores.sort(key=lambda z: z[1])
        return scores[:top_n]


def report(graph: MemoryGraph, reused: set | None = None) -> dict:
    h = graph_health(graph, reused)
    print(f"  atoms: {h['atoms']}")
    print(f"  exact code duplicates: {len(h['exact_dups'])} group(s)  {h['exact_dups'][:3]}")
    print(f"  BEHAVIOURAL duplicates: {len(h['behav_dups'])} group(s)  "
          f"{[[graph.nodes[n].metadata.get('entry', n) for n in g] for g in h['behav_dups'][:3]]}")
    print(f"  orphan atoms (no concept): {len(h['orphans'])}  {h['orphans'][:5]}")
    print(f"  dangling edges: {len(h['dangling'])}  {h['dangling'][:3]}")
    if reused is not None:
        print(f"  dead derived atoms (never reused): {len(h['dead'])}  {h['dead'][:5]}")
    return h


def _selftest() -> bool:
    print("algo_grr_health --selftest: pollution monitor\n")
    from v5.runtime.algo_grr_poison_test import load_seed
    from graph_core import Node, Edge
    ok = True

    # [1] the clean seed is HEALTHY: no dups, no orphans, no dangling
    g = load_seed()
    print("  [1] clean seed:")
    h = report(g)
    clean = (not h["exact_dups"] and not h["behav_dups"] and not h["orphans"] and not h["dangling"])
    print(f"      -> {'PASS' if clean else 'FAIL'}")
    ok &= clean

    # [2] behavioural duplicate DETECTED: add a same-behaviour atom under a new name
    g.nodes["impl_is_prime_clone"] = Node(
        id="impl_is_prime_clone", text="tests primality (clone)", node_type="implementation",
        metadata={"entry": "is_prime_clone", "origin": "derived", "code":
                  "def is_prime_clone(n):\n    return n >= 2 and all(n % i for i in range(2, n))\n"})
    g.edges.append(Edge(src="impl_is_prime_clone", dst="concept_number_theory", relation="part_of"))
    g._rebuild_index()
    print("\n  [2] with an is_prime behavioural clone:")
    h2 = graph_health(g)
    caught = any({"is_prime", "is_prime_clone"} <= {g.nodes[n].metadata.get("entry", n) for n in grp}
                 for grp in h2["behav_dups"])
    print(f"      behavioural-dup groups: "
          f"{[[g.nodes[n].metadata.get('entry', n) for n in grp] for grp in h2['behav_dups']]}")
    print(f"      is_prime clone caught={caught} -> {'PASS' if caught else 'FAIL'}")
    ok &= caught

    # ═══════════════════════════════════════════════════════════════════
    # PruningMonitor selftests (v6)
    # ═══════════════════════════════════════════════════════════════════

    # [3] utility scoring
    g3 = load_seed()
    pm = PruningMonitor(g3)
    u = pm.utility("impl_is_prime")
    assert 0.0 < u <= 1.0, f"utility should be in (0,1], got {u}"
    print(f"  [3] PruningMonitor.utility(impl_is_prime) = {u:.3f} -> PASS")
    ok &= True

    # [4] prune candidates exclude structural nodes
    candidates = pm.find_prune_candidates(threshold=0.15)  # low threshold to find some
    hub_pruned = [c for c in candidates if g3.nodes[c].node_type in ("hub", "concept")]
    assert len(hub_pruned) == 0, f"hub/concept should not be pruned: {hub_pruned}"
    print(f"  [4] find_prune_candidates (threshold=0.15): {len(candidates)} candidates, "
          f"0 hubs pruned -> PASS")
    ok &= len(hub_pruned) == 0

    # [5] decay reduces confidence
    g5 = load_seed()
    pm5 = PruningMonitor(g5, decay_rate=0.1)
    before = g5.nodes["impl_is_prime"].confidence
    pm5.apply_decay()
    after = g5.nodes["impl_is_prime"].confidence
    assert after < before, f"decay should reduce confidence: {before} -> {after}"
    print(f"  [5] apply_decay: confidence {before:.3f} -> {after:.3f} (rate=0.1) -> PASS")
    ok &= after < before

    print(f"\n  ALGO_GRR_HEALTH SELFTEST -> {'PASS' if ok else 'FAIL'}")
    return ok


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--graph", default="")
    a = ap.parse_args()
    if a.selftest:
        sys.exit(0 if _selftest() else 1)
    if a.graph:
        report(MemoryGraph.load_json(a.graph))
        return
    ap.print_help()


if __name__ == "__main__":
    main()
