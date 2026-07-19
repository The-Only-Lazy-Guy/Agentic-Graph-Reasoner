"""algo_grr_scaleup — the scale-up run harness (frozen-compiler + membrane, long corpus).

Assembles a big DECOMPOSABLE + DIVERSE corpus (the compose generator for compounding + MBPP+ for
diversity, interleaved), runs the membrane over it with the winning config, and keeps the graph HEALTHY:

  retrieval  = topology (depend-neighbour boost — beat cosine + the trained net on MBPP+),
  verify     = subprocess hard-kill (V5_HARD_VERIFY, auto on --lm; LM code can't hang the run),
  gate       = fuzz-generality (the fixed one; composed helpers bank),
  SLEEP      = helper-granular derive-bank,
  PRUNE      = drop DERIVED atoms never reused after a grace period (kills the dead-atom bloat MBPP+'s
               atomic tasks cause — 41 banked / 8 reused), so the graph doesn't inflate,
  MONITOR    = periodic graph-health (atoms / dead / dup / orphan),
  LM         = FROZEN throughout ("more data" = the GRAPH grows, not the LM).

Compounding target: cross-task reuse + derived_reuse RISE, per-task LM cost FALLS, graph grows but stays
clean (dead-atom rate bounded by prune).

    selftest (no GPU):  python -m v5.runtime.algo_grr_scaleup --selftest
    molab (real 3B):    python -m v5.runtime.algo_grr_scaleup --run --lm Qwen/Qwen2.5-3B-Instruct \
                          --n-compose 120 --mbpp 200 --save graphs/grr_scaleup.json
"""
from __future__ import annotations

import argparse
import os
import sys
from collections import defaultdict
from pathlib import Path

_ROOT = str(Path(__file__).resolve().parents[2])
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from v5.runtime.algo_grr_poison_test import load_seed, bank_helper_granular  # noqa: E402
from v5.runtime.algo_grr_membrane import MembraneSolver, make_stub_compiler  # noqa: E402
from v5.runtime.algo_grr_compose import gen_corpus, gen_corpus_hard  # noqa: E402
from v5.runtime.algo_grr_health import graph_health  # noqa: E402


def assemble_corpus(n_compose: int = 120, mbpp_limit: int = 200, seed: int = 0,
                    hard: bool = False) -> list[dict]:
    """Interleave the decomposable compose corpus (compounding) with MBPP+ (diversity + dead-atom
    pressure). Interleaved so derived prims recur throughout, not front-loaded. hard=True uses the
    HARD-helper corpus (non-trivial algos the 3B fails to inline -> banking is load-bearing vs RAG)."""
    import random
    compose = (gen_corpus_hard if hard else gen_corpus)(n_compose, seed=seed)
    mbpp = []
    if mbpp_limit > 0:
        try:
            from v5.runtime.algo_grr_mbpp import load_mbpp
            mbpp = load_mbpp(limit=mbpp_limit)
        except Exception as e:  # noqa: BLE001
            print(f"(MBPP+ unavailable: {e!r} — compose-only)")
    rng = random.Random(seed)
    stubs = {t["entry"]: t.get("reference", "") for t in compose + mbpp}
    stream = compose + mbpp
    rng.shuffle(stream)
    return stream, stubs


def run_scaleup(graph, tasks: list[dict], compile_fn, policy_fn=None, retriever=None,
                prune_grace: int = 50, report_every: int = 40, save_path: str = "",
                max_hops: int = 4, max_retries: int = 1, verbose: bool = True, bank: bool = True) -> dict:
    # bank=True  -> OURS: verified self-growth (bank atoms + prune + abstract) -> compounds.
    # bank=False -> RAG baseline: STATIC store, retrieval only, NO banking -> flat (the contrast for #4).
    if retriever is None:
        from v5.runtime.algo_grr_retrieval import CachedTokenRetriever
        retriever = CachedTokenRetriever(graph)
    seed_ids = {nid for nid in graph.nodes if graph.nodes[nid].node_type == "implementation"}
    reuse_count: dict[str, int] = defaultdict(int)   # per-atom reuse over the stream
    bank_task: dict[str, int] = {}                   # nid -> task index it was banked at
    solved = reuse = derived_reuse = banked = pruned = lm_calls = 0
    per = []

    for i, t in enumerate(tasks):
        solver = MembraneSolver(graph, compile_fn, retriever=retriever, policy_fn=policy_fn,
                                max_hops=max_hops, max_retries=max_retries)
        r = solver.solve(t)
        lm_calls += len(solver.compile_inputs)
        if r["solved"]:
            solved += 1
            for s in r["selected"]:
                reuse_count[s] += 1
                reuse += 1
                if s not in seed_ids:
                    derived_reuse += 1
            if bank:                                      # OURS self-grows; RAG baseline keeps a static store
                for nid in bank_helper_granular(graph, r["code"], t["entry"], type_pool=t["type_pool"]):
                    bank_task[nid] = i
                    banked += 1

        # PRUNE dead derived atoms: banked >= grace tasks ago, never reused.
        if bank and (i + 1) % prune_grace == 0:
            dead = [nid for nid, bt in list(bank_task.items())
                    if i - bt >= prune_grace and reuse_count.get(nid, 0) == 0
                    and nid in graph.nodes and graph.nodes[nid].metadata.get("origin") == "derived"]
            for nid in dead:
                graph.nodes.pop(nid, None)
                bank_task.pop(nid, None)
                pruned += 1
            if dead:
                graph.edges = [e for e in graph.edges if e.src in graph.nodes and e.dst in graph.nodes]
                graph._rebuild_index()

        if (i + 1) % report_every == 0 or i == len(tasks) - 1:
            n_impl = sum(1 for n in graph.nodes.values() if n.node_type == "implementation")
            derived_live = [nid for nid in graph.nodes if graph.nodes[nid].metadata.get("origin") == "derived"]
            dead_now = sum(1 for nid in derived_live if reuse_count.get(nid, 0) == 0)
            row = dict(upto=i + 1, solved=solved, reuse=reuse, derived_reuse=derived_reuse,
                       banked=banked, pruned=pruned, atoms=n_impl, dead=dead_now,
                       lm_per_task=round(lm_calls / (i + 1), 2))
            per.append(row)
            if verbose:
                print(f"  [{row['upto']:4d}] solved={row['solved']:4d} reuse={row['reuse']:4d} "
                      f"deriv_reuse={row['derived_reuse']:4d} banked={row['banked']:3d} "
                      f"pruned={row['pruned']:3d} atoms={row['atoms']:3d} dead={row['dead']:3d} "
                      f"lm/task={row['lm_per_task']}", flush=True)

    if save_path:
        graph.save_json(save_path)
        if verbose:
            print(f"  saved grown graph -> {save_path}")
    final = dict(n=len(tasks), solved=solved, reuse=reuse, derived_reuse=derived_reuse,
                 banked=banked, pruned=pruned, atoms=sum(1 for n in graph.nodes.values()
                                                         if n.node_type == "implementation"), per=per)
    return final, graph


# ═══════════════════════════════════════════════════════════════════════════════
# SELFTEST — prune kills dead atoms without killing reused ones; graph stays clean
# ═══════════════════════════════════════════════════════════════════════════════

def _selftest() -> bool:
    print("algo_grr_scaleup --selftest: scale-up harness (prune keeps the graph clean)\n")
    stream, stubs = assemble_corpus(n_compose=40, mbpp_limit=40, seed=1)
    compile_fn = make_stub_compiler(stubs)
    ok = True

    # NO-PRUNE baseline (huge grace) vs PRUNE
    g_np = load_seed()
    res_np, g_np = run_scaleup(g_np, stream, compile_fn, prune_grace=10**9, report_every=10**9, verbose=False)
    g_p = load_seed()
    res_p, g_p = run_scaleup(g_p, stream, compile_fn, prune_grace=20, report_every=10**9, verbose=False)
    print(f"  no-prune: {res_np['atoms']} atoms, banked {res_np['banked']}, pruned {res_np['pruned']}")
    print(f"  prune:    {res_p['atoms']} atoms, banked {res_p['banked']}, pruned {res_p['pruned']}, "
          f"derived_reuse {res_p['derived_reuse']}")

    # [1] prune shrinks the graph (removed dead atoms)
    shrank = res_p["atoms"] < res_np["atoms"] and res_p["pruned"] >= 1
    print(f"  [1] prune removes dead atoms: {res_np['atoms']} -> {res_p['atoms']} "
          f"({res_p['pruned']} pruned) -> {'PASS' if shrank else 'FAIL'}")
    ok &= shrank

    # [2] REUSED prims survive the prune — the compose primitives are still present + were reused
    from v5.runtime.algo_grr_poison_test import _atom_entries
    ents = set(_atom_entries(g_p).keys())
    prims_kept = sum(1 for p in ("sum_of_squares", "nth_fibonacci", "factorial", "triangular")
                     if p in ents)
    print(f"  [2] reused compose prims survive: {prims_kept}/4 present + derived_reuse "
          f"{res_p['derived_reuse']} -> {'PASS' if prims_kept >= 3 and res_p['derived_reuse'] >= 10 else 'FAIL'}")
    ok &= prims_kept >= 3 and res_p["derived_reuse"] >= 10

    # [3] final graph is HEALTHY (no orphans / dangling)
    h = graph_health(g_p)
    clean = not h["orphans"] and not h["dangling"]
    print(f"  [3] final health: {h['atoms']} atoms, {len(h['orphans'])} orphans, "
          f"{len(h['dangling'])} dangling, {len(h['behav_dups'])} behav-dups -> "
          f"{'PASS' if clean else 'FAIL'}")
    ok &= clean

    print(f"\n  ALGO_GRR_SCALEUP SELFTEST -> {'PASS' if ok else 'FAIL'}")
    return ok


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--lm", default="")
    ap.add_argument("--n-compose", type=int, default=120)
    ap.add_argument("--mbpp", type=int, default=200)
    ap.add_argument("--topo", action="store_true", default=True)
    ap.add_argument("--no-topo", dest="topo", action="store_false")
    ap.add_argument("--prune-grace", type=int, default=50)
    ap.add_argument("--report-every", type=int, default=40)
    ap.add_argument("--save", default="")
    ap.add_argument("--rag-baseline", action="store_true",
                    help="#4 hero plot: run OURS (self-growing) vs a STATIC-RAG baseline on the same stream")
    ap.add_argument("--hard", action="store_true",
                    help="use the HARD-helper corpus (non-trivial algos the 3B fails to inline) so banking "
                         "is load-bearing vs RAG; pair with --mbpp 0 for a clean compounding signal")
    ap.add_argument("--v2", action="store_true",
                    help="integrated MembraneV2 (reason+author+bank) vs inline-RAG (no reasoner, no memory) "
                         "on the HARD corpus + held-out set; requires --lm. No-GPU check: "
                         "python -m v5.runtime.algo_grr_pipeline --selftest-v2")
    ap.add_argument("--spec-step", action="store_true",
                    help="step-speculation: batch-author all missing atoms in ONE LM call (cuts LM calls ~2x)")
    ap.add_argument("--debug", type=int, nargs="?", const=5, default=0,
                    help="print per-task detail for first N held-out tasks (default 5)")
    a = ap.parse_args()

    if a.selftest:
        sys.exit(0 if _selftest() else 1)

    if a.run and a.v2:                                # ── integrated MembraneV2 vs inline-RAG (the real #4) ──
        if not a.lm:
            raise SystemExit("--v2 needs --lm (real 3B). No-GPU check: "
                             "python -m v5.runtime.algo_grr_pipeline --selftest-v2")
        os.environ["V5_HARD_VERIFY"] = "1"
        from v5.runtime.algo_grr_membrane import make_frozen_gen
        from v5.runtime.algo_grr_pipeline import run_v2_compare, make_lm_author, make_lm_inline, \
            make_lm_batch_author
        gen = make_frozen_gen(a.lm, temperature=0.6, max_new_tokens=320)
        stream = gen_corpus_hard(a.n_compose, seed=0)
        holdout = gen_corpus_hard(40, seed=0, holdout=True)
        spec_tag = " + SPEC-STEP" if a.spec_step else ""
        print(f"#4-v2 INTEGRATED: MembraneV2(reason+author+bank){spec_tag} vs inline-RAG (no reasoner) | "
              f"stream {len(stream)} + held-out {len(holdout)} | lm={a.lm}\n", flush=True)
        author_fn = make_lm_author(gen)
        batch_author_fn = make_lm_batch_author(gen) if a.spec_step else None
        run_v2_compare(stream, holdout, author_fn, make_lm_inline(gen),
                       batch_author_fn=batch_author_fn,
                       report_every=a.report_every,
                       debug_heldout_n=a.debug)
        return

    if a.run:
        stream, stubs = assemble_corpus(a.n_compose, a.mbpp, seed=0, hard=a.hard)
        from v5.runtime.algo_grr_retrieval import CachedTokenRetriever, make_topology_policy
        if a.lm:
            os.environ["V5_HARD_VERIFY"] = "1"            # subprocess verify -> no hangs
            from v5.runtime.algo_grr_membrane import make_frozen_gen, make_lm_compiler
            compile_fn = make_lm_compiler(make_frozen_gen(a.lm, temperature=0.6, max_new_tokens=320))
            print("[hard-verify] subprocess per verify")
        else:
            compile_fn = make_stub_compiler(stubs)
            print("(stub = reference; use --lm for the real scale-up run)")

        if a.rag_baseline:                                # ── #4: OURS vs static-RAG, same stream ──
            def _arm(bank, topo, label):
                print(f"\n  ── running {label} arm: {len(stream)} tasks (progress every {a.report_every}) ──", flush=True)
                g = load_seed()
                return run_scaleup(g, stream, compile_fn,
                                   policy_fn=make_topology_policy(g) if topo else None,
                                   retriever=CachedTokenRetriever(g), prune_grace=a.prune_grace,
                                   report_every=a.report_every, verbose=True, bank=bank)[0]
            print(f"#4 COMPOUNDING vs RAG: {len(stream)} tasks, lm={a.lm or 'stub'}\n")
            ours = _arm(bank=True, topo=True, label="OURS (self-growing + topology)")   # verified self-growth
            rag = _arm(bank=False, topo=False, label="RAG baseline (static store, no banking)")
            print(f"  window |  OURS solved  lm/task |  RAG solved  lm/task")
            for po, pr in zip(ours["per"], rag["per"]):
                print(f"  {po['upto']:>6} |    {po['solved']:>4}     {po['lm_per_task']:>5} |   {pr['solved']:>4}    "
                      f" {pr['lm_per_task']:>5}")
            print(f"\n  FINAL  OURS: solved {ours['solved']}/{ours['n']} | banked {ours['banked']} | "
                  f"DERIVED_REUSE {ours['derived_reuse']} | lm/task {ours['per'][-1]['lm_per_task']}")
            print(f"         RAG : solved {rag['solved']}/{rag['n']} | banked {rag['banked']} | "
                  f"DERIVED_REUSE {rag['derived_reuse']} (static store) | lm/task {rag['per'][-1]['lm_per_task']}")
            print(f"  => COMPOUNDING signal = derived_reuse: OURS {ours['derived_reuse']} vs RAG {rag['derived_reuse']} "
                  f"(RAG cannot reuse — it never banks). With the real 3B, OURS lm/task also falls as banked")
            print(f"     atoms cut compose cost; the stub is cost-flat so lm/task shows the plumbing only.")
            return

        graph = load_seed()
        policy_fn = make_topology_policy(graph) if a.topo else None
        print(f"SCALE-UP: {len(stream)} tasks ({a.n_compose} compose + MBPP+), lm={a.lm or 'stub'}, "
              f"retrieval={'topo' if a.topo else 'cosine'}, prune_grace={a.prune_grace}\n")
        res, _g = run_scaleup(graph, stream, compile_fn, policy_fn=policy_fn,
                              retriever=CachedTokenRetriever(graph),
                              prune_grace=a.prune_grace, report_every=a.report_every, save_path=a.save)
        print(f"\nSCALE-UP DONE: solved {res['solved']}/{res['n']} | reuse {res['reuse']} "
              f"(derived {res['derived_reuse']}) | banked {res['banked']} | pruned {res['pruned']} | "
              f"final atoms {res['atoms']}")
        return

    ap.print_help()


if __name__ == "__main__":
    main()
