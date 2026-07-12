"""GRR-3: counterfactual delta-capability - the endogenous reward, made measurable. An atom is worth exactly
how much it improves the model's ability to solve its task distribution UNDER BUDGET:

    delta-capability(atom) = solve_rate(full graph)  -  solve_rate(graph with the atom ablated)

Ablate the node (+ its edges) → re-solve a held-out slice under the real budget (GRR-2) → the drop is
its worth. This SUBSUMES reuse / description-length / exec-cost into one signal grounded in the model's
own benefit - no hand-set weights. `if x==17: return 4` has delta~0 (removing it changes nothing) → never
consolidates. A load-bearing atom (dijkstra) has delta>0. rank_atoms() scores the whole library → dead
weight (delta~0) is prune-bait, high-delta atoms are the keepers.

  selftest (no model):  python -m v5.runtime.algo_capability --selftest
  rank a graph (GPU):   python -m v5.runtime.algo_capability --rank --graph graphs/algo_reason_hard.json --hard
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

from graph_core import MemoryGraph


def ablate(graph_path: str, atom_id: str, out_path: str):
    """Write a copy of the graph with `atom_id` (and every edge touching it) removed."""
    g = json.loads(Path(graph_path).read_text(encoding="utf-8"))
    g["nodes"] = [n for n in g["nodes"] if n["id"] != atom_id]
    g["edges"] = [e for e in g.get("edges", [])
                  if e.get("src") != atom_id and e.get("dst") != atom_id]
    Path(out_path).write_text(json.dumps(g), encoding="utf-8")


def solve_rate(graph_path: str, tasks, gen_fn, embed_fn, budget_tokens: int, samples: int = 4,
               k: int = 8, min_cos: float = 0.25, max_rounds: int = 8) -> float:
    """Fraction of tasks solved UNDER the token budget with the given graph."""
    from v5.runtime.algo_graph_mg import MGRetriever
    from v5.runtime.algo_graph_reason import Budget, solve_iterative
    retr = MGRetriever(MemoryGraph.load_json(graph_path), embed_fn)
    solved = 0
    for t in tasks:
        r = solve_iterative(t, retr, gen_fn, embed_fn, max_rounds=max_rounds, samples=samples, k=k,
                            min_cos=min_cos, budget=Budget(budget_tokens))
        solved += int(r["solved"])
    return solved / max(1, len(tasks))


def delta_capability(atom_id: str, graph_path: str, tasks, gen_fn, embed_fn, budget_tokens: int,
                     **kw) -> dict:
    """delta-capability(atom) = solve_rate(full) - solve_rate(ablated). The atom's counterfactual worth."""
    full = solve_rate(graph_path, tasks, gen_fn, embed_fn, budget_tokens, **kw)
    with tempfile.TemporaryDirectory() as td:
        abl = str(Path(td) / "ablated.json")
        ablate(graph_path, atom_id, abl)
        without = solve_rate(abl, tasks, gen_fn, embed_fn, budget_tokens, **kw)
    return dict(atom=atom_id, with_rate=round(full, 3), without_rate=round(without, 3),
                delta=round(full - without, 3))


def rank_atoms(graph_path: str, tasks, gen_fn, embed_fn, budget_tokens: int, samples: int = 4,
               k: int = 8, min_cos: float = 0.25, max_rounds: int = 8) -> list[dict]:
    """Score every impl atom by delta-capability, FAST. Ablating atom A can only change tasks whose
    full-graph solution actually USED A -> solve the set ONCE (recording used atoms), then re-test each
    atom only on its handful of users. ~1 full pass instead of |atoms|+1. delta~0 = dead weight."""
    from v5.runtime.algo_graph_mg import MGRetriever
    from v5.runtime.algo_graph_reason import Budget, solve_iterative
    kw = dict(samples=samples, k=k, min_cos=min_cos, max_rounds=max_rounds)

    retr = MGRetriever(MemoryGraph.load_json(graph_path), embed_fn)
    solved = []                                              # (task, used_atom_names) for SOLVED tasks
    for t in tasks:
        r = solve_iterative(t, retr, gen_fn, embed_fn, budget=Budget(budget_tokens), **kw)
        if r["solved"]:
            solved.append((t, set(r["used"] or [])))
    n = len(tasks)
    full_solved = len(solved)

    g = MemoryGraph.load_json(graph_path)
    atoms = [nid for nid, node in g.nodes.items() if node.node_type == "implementation"]
    out = []
    for aid in atoms:
        aname = aid[len("impl_"):] if aid.startswith("impl_") else aid
        at_risk = [t for t, used in solved if aname in used]     # solved tasks that USED this atom
        if not at_risk:                                          # never used -> ablating it is free -> delta 0
            out.append(dict(atom=aid, delta=0.0, without=round(full_solved / n, 3)))
            continue
        with tempfile.TemporaryDirectory() as td:
            abl = str(Path(td) / "a.json"); ablate(graph_path, aid, abl)
            r2 = MGRetriever(MemoryGraph.load_json(abl), embed_fn)
            still = sum(1 for t in at_risk
                        if solve_iterative(t, r2, gen_fn, embed_fn, budget=Budget(budget_tokens),
                                           **kw)["solved"])
        without = full_solved - len(at_risk) + still            # non-users unchanged; users re-tested
        out.append(dict(atom=aid, delta=round((full_solved - without) / n, 3),
                        without=round(without / n, 3), users=len(at_risk)))
    out.sort(key=lambda d: -d["delta"])
    return out


# ═══════════════════════════════════════════════════════════════════════════════
# SELFTEST - real ablation: removing a LOAD-BEARING atom drops solve (delta>0); removing DEAD WEIGHT
# changes nothing (delta=0). The mechanism is real (ablated atom → missing dep → verify fails).
# ═══════════════════════════════════════════════════════════════════════════════

def _selftest() -> bool:
    from v5.memory.store import make_fake_embedder
    from v5.runtime.algo_compose_tasks import ALL_ATOMS, gen_compose_tasks
    print("algo_capability --selftest: delta-capability - load-bearing atom delta>0, dead weight delta=0\n")
    embed = make_fake_embedder()

    with tempfile.TemporaryDirectory() as td:
        # graph: the two atoms sum_digitsum_primes needs + a USELESS dead-weight atom
        gp = str(Path(td) / "g.json")
        nodes = [{"id": "concept_number_theory", "text": "number theory", "node_type": "concept"}]
        for a in ("is_prime", "digit_sum"):
            nodes.append({"id": f"impl_{a}", "text": ALL_ATOMS[a][0], "node_type": "implementation",
                          "metadata": {"code": ALL_ATOMS[a][1]}})
        nodes.append({"id": "impl_dead", "text": "unused helper", "node_type": "implementation",
                      "metadata": {"code": "def dead(x):\n    return x"}})
        Path(gp).write_text(json.dumps({"metadata": {}, "nodes": nodes, "edges": []}))

        tasks = [t for t in gen_compose_tasks(24, seed=1) if t.name == "sum_digitsum_primes"][:6]

        def _compose_stub(prompts):                          # always emits the composing solution
            return ["```python\ndef sum_digitsum_primes(lst):\n"
                    "    return sum(digit_sum(x) for x in lst if is_prime(x))\n```"] * len(prompts)

        # load-bearing: ablating is_prime removes the dep -> verify fails -> solve drops
        d_prime = delta_capability("impl_is_prime", gp, tasks, _compose_stub, embed, budget_tokens=10000,
                                   min_cos=-1.0, samples=1)
        assert d_prime["with_rate"] == 1.0 and d_prime["without_rate"] == 0.0 and d_prime["delta"] > 0, d_prime
        print(f"  [1] load-bearing is_prime: with={d_prime['with_rate']:.0%} without={d_prime['without_rate']:.0%} "
              f"-> delta={d_prime['delta']:+.2f} (worth keeping) -> PASS")

        # dead weight: ablating the unused atom changes nothing -> delta=0
        d_dead = delta_capability("impl_dead", gp, tasks, _compose_stub, embed, budget_tokens=10000,
                                  min_cos=-1.0, samples=1)
        assert d_dead["delta"] == 0.0, d_dead
        print(f"  [2] dead weight: with={d_dead['with_rate']:.0%} without={d_dead['without_rate']:.0%} "
              f"-> delta={d_dead['delta']:+.2f} (prune-bait) -> PASS")

        # ranking separates them
        ranked = rank_atoms(gp, tasks, _compose_stub, embed, budget_tokens=10000, min_cos=-1.0, samples=1)
        assert ranked[0]["atom"] != "impl_dead" and ranked[-1]["atom"] == "impl_dead", ranked
        print(f"  [3] rank_atoms: top={ranked[0]['atom']} (delta{ranked[0]['delta']:+.2f}), "
              f"bottom={ranked[-1]['atom']} (delta{ranked[-1]['delta']:+.2f}) -> PASS")

    print("\n  ALGO_CAPABILITY SELFTEST -> PASS")
    return True


def main():
    ap = argparse.ArgumentParser(description="GRR-3: counterfactual delta-capability under budget.")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--rank", action="store_true", help="score every atom's delta-capability (GPU)")
    ap.add_argument("--model", default="Qwen/Qwen3-4B")
    ap.add_argument("--graph", default="graphs/algo_reason_hard.json")
    ap.add_argument("--n-tasks", type=int, default=24)
    ap.add_argument("--budget", type=int, default=600)
    ap.add_argument("--hard", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        sys.exit(0 if _selftest() else 1)
    if a.rank:
        from v5.runtime.algo_compose_tasks import gen_compose_tasks, seed_atom_graph
        from v5.runtime.algo_graph_reason import _build_lm
        if not Path(a.graph).exists():
            seed_atom_graph(a.graph, hard=a.hard)
        gen_fn, embed = _build_lm(a.model)
        tasks = gen_compose_tasks(a.n_tasks, seed=99, hard=a.hard)
        print(f"rank_atoms: {a.graph} | budget={a.budget} | {len(tasks)} held-out tasks", flush=True)
        for r in rank_atoms(a.graph, tasks, gen_fn, embed, budget_tokens=a.budget):
            print(f"  {r['atom']:24s} delta-capability={r['delta']:+.2f}  (without={r['without']:.0%})", flush=True)
        return
    ap.print_help()


if __name__ == "__main__":
    main()
