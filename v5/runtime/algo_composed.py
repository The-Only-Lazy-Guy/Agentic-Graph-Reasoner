"""GRR-6 step 1: the COMPOSE-FORCED solver — the thesis validation, no LM, deterministic.

The str_dp2 experiment proved: a frozen 4B INLINES, so abstractions stay d0. GRR-6's whole reason to
exist is a model that CANNOT inline — its solutions must CALL atoms. This module is that solver's
substrate (and the target the TRM will drive): a solution is built in the CALL-the-atom form (the `_REF`
wiring), its transitive deps are resolved through the graph (MGRetriever.resolve_deps — the graph walk),
and it's fuzz-verified. Because the solution ALWAYS calls the atom (never inlines the DP), an abstracted
leaf (edit_distance -> str_dp2) drags in the skeleton, and ABLATING the skeleton breaks it.

Result to prove: on the dp2 graph, `str_dp2` scores d>0 (load-bearing) under compose-forcing, exactly
where the free-form 4B left it at d0. That is the reason to build the compose-forced model.

  selftest (no model):  python -m v5.runtime.algo_composed --selftest
  rank a graph:         python -m v5.runtime.algo_composed --rank --graph graphs/algo_reason_dp2.json --hard
"""
from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

from graph_core import MemoryGraph
from v5.runtime.algo_capability import ablate
from v5.runtime.algo_compose_tasks import _NEEDS, _REF
from v5.runtime.algo_graph_mg import MGRetriever
from v5.runtime.algo_quality import fuzz


def composed_solution(task_name: str):
    """The compose-forced solution for a family = the CALL-the-atom wiring (_REF). It calls the needed
    atoms; it never inlines them. Returns (code, needed_atom_names) or (None, None) if no template."""
    if task_name not in _REF:
        return None, None
    return _REF[task_name], set(_NEEDS.get(task_name, ()))


def composed_solve(graph_path: str, tasks, embed_fn, n_fuzz: int = 40):
    """Solve each task in the compose-forced form, resolving deps through the graph (transitive walk)
    and fuzz-verifying. Returns [(task, used_closure)] for the tasks that solve GENERAL."""
    retr = MGRetriever(MemoryGraph.load_json(graph_path), embed_fn)
    solved = []
    for t in tasks:
        code, needed = composed_solution(t.name)
        if code is None:
            continue
        deps = retr.resolve_deps(needed)                 # graph WALK: pull instances + their skeletons
        passed, total = fuzz(code, t.name, deps, n=n_fuzz)
        if total and passed == total:                    # general, not just benchmark
            solved.append((t, retr.resolve_dep_names(needed)))   # transitive closure = what it truly needs
    return solved


def composed_solve_rate(graph_path: str, tasks, embed_fn) -> float:
    return len(composed_solve(graph_path, tasks, embed_fn)) / max(1, len(tasks))


def rank_atoms_composed(graph_path: str, tasks, embed_fn) -> list[dict]:
    """dcapability under COMPOSE-FORCING: ablate each atom -> re-solve (deterministic, no LM) -> the
    drop is its worth. A transitively-needed skeleton (str_dp2) scores high because ablating it breaks
    every leaf that calls it. This is the same library, judged by a model that must compose."""
    full = composed_solve(graph_path, tasks, embed_fn)
    full_n = len(full)
    n = len(tasks)
    g = MemoryGraph.load_json(graph_path)
    atoms = [nid for nid, node in g.nodes.items() if node.node_type == "implementation"]
    # users = solved tasks whose transitive closure includes this atom (credits skeletons)
    out = []
    for aid in atoms:
        aname = aid[len("impl_"):] if aid.startswith("impl_") else aid
        users = sum(1 for _, used in full if aname in used)
        with tempfile.TemporaryDirectory() as td:
            abl = str(Path(td) / "a.json"); ablate(graph_path, aid, abl)
            without = len(composed_solve(abl, tasks, embed_fn))
        out.append(dict(atom=aid, delta=round((full_n - without) / n, 3),
                        without=round(without / n, 3), users=users))
    out.sort(key=lambda d: -d["delta"])
    return out


# ═══════════════════════════════════════════════════════════════════════════════
# SELFTEST — the thesis: under compose-forcing, str_dp2 is LOAD-BEARING (d>0) where the free-form 4B
# left it at d0. Deterministic, no LM.
# ═══════════════════════════════════════════════════════════════════════════════

def _selftest() -> bool:
    import json
    from v5.memory.store import make_fake_embedder
    from v5.runtime.algo_abstract import DP2_INSTANCES, STR_DP2, _DP2_ORACLE
    from v5.runtime.algo_compose_tasks import ALL_ATOMS, gen_compose_tasks
    from v5.runtime.algo_sleep import sleep_compress
    print("algo_composed --selftest: compose-forcing makes str_dp2 LOAD-BEARING (vs d0 free-form)\n")
    embed = make_fake_embedder()

    with tempfile.TemporaryDirectory() as td:
        # base graph: edit_distance + lcs_length (FULL DP) + the atoms the other hard families need
        gp = str(Path(td) / "g.json")
        nodes = [{"id": "concept_algorithms", "text": "algorithms", "node_type": "concept"}]
        for a in ("edit_distance", "lcs_length", "coin_change", "lis_length"):
            nodes.append({"id": f"impl_{a}", "text": ALL_ATOMS[a][0], "node_type": "implementation",
                          "metadata": {"code": ALL_ATOMS[a][1]}})
        Path(gp).write_text(json.dumps({"metadata": {}, "nodes": nodes, "edges": []}))
        tasks = gen_compose_tasks(24, seed=99, hard=True)          # sum_edit_distance/sum_lcs/count_makeable/max_lis

        # [1] compose-forced solve works and CALLS the atom (leaf still full DP here)
        solved0 = composed_solve(gp, tasks, embed)
        r0 = rank_atoms_composed(gp, tasks, embed)
        base = {d["atom"]: d for d in r0}
        assert len(solved0) >= 12, len(solved0)
        assert base["impl_edit_distance"]["delta"] > 0 and base["impl_lcs_length"]["delta"] > 0, r0
        print(f"  [1] base graph: compose-forced solves {len(solved0)}/{len(tasks)}; edit_distance "
              f"d={base['impl_edit_distance']['delta']:+.2f}, lcs_length d={base['impl_lcs_length']['delta']:+.2f} "
              f"(leaves load-bearing, as always) -> PASS")

        # ABSTRACT: sleep-compress str_dp2 (leaves become 1-line instances calling it)
        dp2 = str(Path(td) / "dp2.json")
        sleep_compress(gp, dp2, [("str_dp2", STR_DP2, DP2_INSTANCES, _DP2_ORACLE)])

        # [2] THE THESIS: on the abstracted graph, str_dp2 is now LOAD-BEARING under compose-forcing
        r = rank_atoms_composed(dp2, tasks, embed)
        d = {x["atom"]: x for x in r}
        sd = d["impl_str_dp2"]
        assert sd["delta"] > 0 and sd["users"] > 0, sd
        assert r[0]["atom"] == "impl_str_dp2", r     # top of the ranking
        print(f"  [2] dp2 graph: str_dp2 d={sd['delta']:+.2f} ({sd['users']} users) = TOP of the ranking "
              f"-> LOAD-BEARING under compose-forcing (free-form 4B had it at d0) -> PASS")

        # [3] ablating str_dp2 really breaks the leaves (they can't resolve their skeleton)
        without = composed_solve_rate(dp2, tasks, embed)
        with tempfile.TemporaryDirectory() as td2:
            abl = str(Path(td2) / "nosk.json"); ablate(dp2, "impl_str_dp2", abl)
            ablated_rate = composed_solve_rate(abl, tasks, embed)
        assert ablated_rate < without, (without, ablated_rate)
        print(f"  [3] ablate str_dp2 -> compose-forced solve {without:.0%} -> {ablated_rate:.0%} "
              f"(leaves can't resolve the skeleton) -> PASS")

    print("\n  ALGO_COMPOSED SELFTEST -> PASS  (the reason to build GRR-6: memory pays when the model MUST compose)")
    return True


def main():
    ap = argparse.ArgumentParser(description="GRR-6 step 1: compose-forced solver + composed dcapability.")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--rank", action="store_true", help="rank atoms by composed dcapability (no LM)")
    ap.add_argument("--graph", default="graphs/algo_reason_hard.json")
    ap.add_argument("--n-tasks", type=int, default=24)
    ap.add_argument("--hard", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        sys.exit(0 if _selftest() else 1)
    if a.rank:
        from v5.memory.store import make_mpnet_embedder
        from v5.runtime.algo_compose_tasks import gen_compose_tasks, seed_atom_graph
        if not Path(a.graph).exists():
            seed_atom_graph(a.graph, hard=a.hard)
        tasks = gen_compose_tasks(a.n_tasks, seed=99, hard=a.hard)
        print(f"rank_atoms_composed: {a.graph} | {len(tasks)} tasks (compose-forced, no LM)", flush=True)
        for r in rank_atoms_composed(a.graph, tasks, make_mpnet_embedder()):
            print(f"  {r['atom']:24s} dcapability={r['delta']:+.2f}  (without={r['without']:.0%}, {r['users']} users)",
                  flush=True)
        return
    ap.print_help()


if __name__ == "__main__":
    main()
