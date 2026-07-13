"""GRR-8: THE unified wake/sleep compounding loop — every validated stage, ONE driver over ONE graph.

  WAKE         per task (curriculum: the transform budget grows with rounds): DECODE (recall) -> verify;
               on fail, net-GUIDED SEARCH under a verify budget (algo_dsl_trm.solve_with_search — the
               until-solve inference primitive)
  CONSOLIDATE  SFT the net on every discovery so far (STaR: the decode amortizes the search)
  SLEEP        write each newly-discovered family PROGRAM into the GRAPH as an implementation node —
               code = the realized program (CALLS atoms, never inlines), the pipeline rides SYMBOLICALLY
               in metadata (the graph stores the program's FORM), depend edges to its atom closure —
               via graph_grower's health-gated apply. Then re-index retrieval: the action space IS the
               graph, and it grew.
  MEASURE      per round: net zero-shot solve (decode-only, fresh instances), mean verifies-to-solve,
               graph size. Compounding = zero-shot RISES while verifies-to-solve FALLS.

--rebuild-net  fresh net, NO search: SFT purely from the GRAPH's stored program nodes, then eval.
               Content is symbolic in the graph (survives any net/box reset); the net just re-amortizes.

Discovery is search + oracle-I/O verify only — reference pipelines are NEVER read (algo_dsl._PROGRAMS
supplies family SCOPE and input kind, nothing else). Program nodes are stored as retrievable outputs;
enumerating them as callable atoms (--programs-as-atoms) is the hierarchical-composition lever for the
scale-up dataset (mechanism present, wins need tasks that nest families).

  selftest (no LM):     python -m v5.runtime.algo_grr_loop --selftest
  molab (real mpnet):   python -m v5.runtime.algo_grr_loop --loop --graph graphs/algo_grr_loop.json
                        python -m v5.runtime.algo_grr_loop --rebuild --graph graphs/algo_grr_loop.json
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from graph_core import MemoryGraph
from v5.runtime.algo_dsl import _PROGRAMS, Op, atoms_of, realize_program
from v5.runtime.algo_dsl_trm import (_build, _guided_search, _is_general, _sft_steps, program_to_steps,
                                     solve_with_search)
from v5.runtime.algo_graph_edits import edge_candidate, grow, node_candidate
from v5.runtime.algo_graph_mg import MGRetriever, _fn_name

# family SCOPE + input kind only — the reference pipelines are never read (discovery = search + verify)
FAM_KINDS = {f: (k, None) for f, (k, _p) in _PROGRAMS.items()}


def _graph_atoms(graph: MemoryGraph, embed_fn, programs_as_atoms: bool = False):
    """The action space, read from the GRAPH: implementation nodes with code (program nodes excluded by
    default — see module docstring). Returns (atom_names, atom_idx, atom_vecs)."""
    impls = [(nid, n) for nid, n in graph.nodes.items()
             if n.node_type == "implementation" and n.metadata.get("code")
             and (programs_as_atoms or n.metadata.get("kind") != "program")]
    names = [_fn_name(n.metadata["code"]) or nid for nid, n in impls]
    vecs = embed_fn({nid: n.text for nid, n in impls})
    mat = np.asarray([vecs[nid] for nid, _ in impls], dtype=np.float32)
    return names, {a: i for i, a in enumerate(names)}, mat


def _tasks_by_family(seed: int, n: int = 60):
    from v5.runtime.algo_compose_tasks import gen_compose_tasks
    by = {}
    for t in list(gen_compose_tasks(n, seed=seed)) + list(gen_compose_tasks(n, seed=seed, hard=True)):
        if t.name in FAM_KINDS:
            by.setdefault(t.name, []).append(t)
    return by


def _zero_shot(model, atom_names, atom_vecs, resolve_fn, embed_fn, seed: int, n_per: int = 3,
               n_verify: int = 24):
    """Decode-ONLY solve on fresh instances (no search, no updates) — the recall/amortization measure."""
    import torch
    A = torch.as_tensor(atom_vecs, dtype=torch.float32)
    by = _tasks_by_family(seed)
    fams_solved, inst_solved, inst_total = 0, 0, 0
    for fam, ts in by.items():
        ok = 0
        for t in ts[:n_per]:
            gv = np.asarray(list(embed_fn({"q": t.text}).values())[0], dtype=np.float32)
            pipe = model.decode(torch.as_tensor(gv), A, atom_names)
            ok += int(_is_general(pipe, fam, FAM_KINDS, resolve_fn, n_verify))
        inst_solved += ok; inst_total += min(n_per, len(ts))
        fams_solved += int(ok == min(n_per, len(ts)) and ok > 0)
    return fams_solved, len(by), inst_solved, inst_total


def _sleep_store(graph_path: str, retr: MGRetriever, discoveries: dict, rnd: int,
                 concept: str = "concept_algorithms"):
    """SLEEP write-back: each discovered family program -> an implementation node (kind=program, pipeline
    symbolic in metadata) + depend edges to the atoms it calls + part_of the concept. Health-gated via
    graph_grower. Returns #stored."""
    from v5.runtime.algo_compose_tasks import _TEXT
    g = retr.graph
    cands = []
    for fam, pipe in discoveries.items():
        nid = f"impl_{fam}"
        if nid in g.nodes:                                  # already banked (or name collision) -> skip
            continue
        code = realize_program(fam, FAM_KINDS[fam][0], pipe)
        cands.append(node_candidate(
            nid, code, _TEXT[fam], f"grr8_r{rnd}",
            metadata={"kind": "program", "family": fam, "input_kind": FAM_KINDS[fam][0],
                      "pipeline": [[op.kind, op.arg] for op in pipe]}))
        cands.append(edge_candidate(nid, concept, "part_of", f"grr8_r{rnd}"))
        for atom in sorted(atoms_of(pipe)):
            cands.append(edge_candidate(nid, f"impl_{atom}", "depend", f"grr8_r{rnd}"))
    if not cands:
        return 0
    newp = graph_path + ".grown"
    res = grow(graph_path, newp, cands)
    if res.get("persisted"):
        Path(newp).replace(graph_path)
    return sum(1 for c in cands if c["raw_edit"]["op"] == "add_node")


def wake_sleep_loop(graph_path: str, embed_fn, rounds: int = 3, budget: int = 1500,
                    sft_steps: int = 400, n_wake: int = 2, seed: int = 0, log: bool = True):
    """The loop. Returns (model, hist) with hist rows:
    (round, n_search_discoveries, mean_verifies_to_solve, fams_zero_shot, n_fams, graph_nodes)."""
    import torch
    torch, _nn, ProgramDecoder = _build()
    torch.manual_seed(seed)
    g = MemoryGraph.load_json(graph_path)
    atom_names, atom_idx, atom_vecs = _graph_atoms(g, embed_fn)
    retr = MGRetriever(g, embed_fn)
    resolve_fn = lambda atoms: retr.resolve_deps(atoms) if atoms else ""
    model = ProgramDecoder(d_in=atom_vecs.shape[1], d=64)
    opt = torch.optim.Adam(model.parameters(), lr=5e-4)
    A = torch.as_tensor(atom_vecs, dtype=torch.float32)
    pool, known = [], {}                                    # known: fam -> discovered pipe
    hist = []
    for r in range(rounds):
        mt = 1 if r == 0 else 2                             # curriculum: 2-step programs, then 3-step
        by = _tasks_by_family(seed=100 + r)
        n_disc, verifies = 0, []
        for fam, ts in sorted(by.items()):
            for t in ts[:n_wake]:                           # WAKE
                res = solve_with_search(model, t, FAM_KINDS, atom_names, atom_vecs, resolve_fn,
                                        embed_fn, atom_idx, opt=opt, budget=budget,
                                        max_transforms=mt, seed=seed + r)
                if res["solved"]:
                    verifies.append(res["verifies"])
                    if res["via"] == "search":
                        n_disc += 1
                        known[fam] = res["pipe"]
                    gv = np.asarray(list(embed_fn({"q": t.text}).values())[0], dtype=np.float32)
                    pool.append((gv, program_to_steps(known.get(fam, res["pipe"]), atom_idx)))
        _sft_steps(model, opt, pool, A, sft_steps, seed=seed + r)          # CONSOLIDATE (replay)
        stored = _sleep_store(graph_path, retr, known, r)                  # SLEEP (idempotent per fam)
        if stored:
            retr = MGRetriever(MemoryGraph.load_json(graph_path), embed_fn)
            resolve_fn = lambda atoms: retr.resolve_deps(atoms) if atoms else ""
        fz, nf, iz, it = _zero_shot(model, atom_names, atom_vecs, resolve_fn, embed_fn, seed=900 + r)
        mv = sum(verifies) / max(1, len(verifies))
        hist.append((r, n_disc, mv, fz, nf, len(retr.graph.nodes)))
        if log:
            print(f"  round {r}: mt<={mt} | search-discovered {n_disc} | verifies-to-solve {mv:.1f} | "
                  f"zero-shot {fz}/{nf} fams ({iz}/{it} inst) | graph {len(retr.graph.nodes)} nodes "
                  f"{len(retr.graph.edges)} edges", flush=True)
    return model, hist


def rebuild_net(graph_path: str, embed_fn, sft_steps: int = 800, seed: int = 0, log: bool = True):
    """The persistence proof: a FRESH net, NO search — SFT purely from the graph's stored program nodes
    (pipeline is symbolic in metadata), then decode-only eval on fresh instances. The graph IS the
    memory; the net re-amortizes it in seconds. Returns (fams_solved, n_fams)."""
    import torch
    torch, _nn, ProgramDecoder = _build()
    torch.manual_seed(seed)
    g = MemoryGraph.load_json(graph_path)
    atom_names, atom_idx, atom_vecs = _graph_atoms(g, embed_fn)
    progs = [(nid, n) for nid, n in g.nodes.items() if n.metadata.get("kind") == "program"]
    traces = []
    for nid, n in progs:
        pipe = [Op(k, a) for k, a in n.metadata["pipeline"]]
        gv = np.asarray(list(embed_fn({nid: n.text}).values())[0], dtype=np.float32)
        traces.append((gv, program_to_steps(pipe, atom_idx)))
    model = ProgramDecoder(d_in=atom_vecs.shape[1], d=64)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    _sft_steps(model, opt, traces, torch.as_tensor(atom_vecs, dtype=torch.float32), sft_steps, seed=seed)
    retr = MGRetriever(g, embed_fn)
    resolve_fn = lambda atoms: retr.resolve_deps(atoms) if atoms else ""
    fz, nf, iz, it = _zero_shot(model, atom_names, atom_vecs, resolve_fn, embed_fn, seed=901)
    if log:
        print(f"  rebuild-net: fresh net + {len(traces)} graph-stored programs (no search) -> "
              f"zero-shot {fz}/{nf} fams ({iz}/{it} inst)", flush=True)
    return fz, nf


def _mpnet_embed():
    from v5.memory.store import make_mpnet_embedder
    return make_mpnet_embedder()


# ═══════════════════════════════════════════════════════════════════════════════
# SELFTEST — the WHOLE loop, no LM: cold net discovers by search; consolidation lifts zero-shot;
# sleep banks programs into the graph (retrievable); verifies-to-solve falls; a FRESH net rebuilds
# from the graph alone (the persistence thesis).
# ═══════════════════════════════════════════════════════════════════════════════

def _selftest() -> bool:
    import json
    import tempfile
    from v5.runtime.algo_compose_tasks import ALL_ATOMS
    print("algo_grr_loop --selftest: wake->consolidate->sleep(graph)->measure, + rebuild-from-graph\n")

    atom_names = ["is_prime", "digit_sum", "build_adj", "dijkstra", "edit_distance", "lcs_length",
                  "coin_change", "lis_length"]
    rng = np.random.default_rng(0)
    d_in = 64
    # specific before general: "increasing" must precede "subsequence" (max_lis's text contains
    # "longest-increasing-SUBSEQUENCE" — the general keyword would collapse both fams to one embedding)
    kw = [("largest digit-sum", "max_prime_digitsum"), ("prime", "sum_digitsum_primes"),
          ("edit", "sum_edit_distance"), ("increasing", "max_lis"),
          ("subsequence", "sum_lcs"), ("coin", "count_makeable")]
    fam_base = {f: rng.standard_normal(d_in).astype("float32") for f in FAM_KINDS}

    def embed(d):
        out = {}
        for k, text in d.items():
            f = next((f for w, f in kw if w in text.lower()), None)
            out[k] = ((fam_base[f] if f in fam_base else 0.05 * rng.standard_normal(d_in))
                      + 0.3 * rng.standard_normal(d_in)).astype("float32")
        return out

    with tempfile.TemporaryDirectory() as td:
        gp = str(Path(td) / "g.json")
        nodes = [{"id": "concept_algorithms", "text": "algorithms", "node_type": "concept"}]
        for a in atom_names:
            nodes.append({"id": f"impl_{a}", "text": ALL_ATOMS[a][0], "node_type": "implementation",
                          "metadata": {"code": ALL_ATOMS[a][1]}})
        Path(gp).write_text(json.dumps({"metadata": {}, "nodes": nodes, "edges": []}))

        model, hist = wake_sleep_loop(gp, embed, rounds=3, budget=1500, sft_steps=800, n_wake=4, seed=0)

        # [1] round 0: the COLD net cannot decode-solve -> discoveries come from SEARCH (wake works)
        assert hist[0][1] >= 3, hist[0]
        print(f"\n  [1] round 0: cold net -> {hist[0][1]} families discovered via SEARCH -> PASS")

        # [2] compounding, axis 1: zero-shot (decode-only) RISES to all families
        fz_last, nf = hist[-1][3], hist[-1][4]
        assert fz_last == nf == 6, hist[-1]
        assert hist[0][3] <= hist[-1][3], (hist[0], hist[-1])
        print(f"  [2] zero-shot fams {hist[0][3]}/{nf} (r0) -> {fz_last}/{nf} (r{hist[-1][0]}): the net "
              f"AMORTIZES its own discoveries -> PASS")

        # [3] compounding, axis 2: verifies-to-solve FALLS (search costs vanish once consolidated)
        assert hist[-1][2] < hist[0][2], [h[2] for h in hist]
        print(f"  [3] verifies-to-solve {hist[0][2]:.1f} (r0) -> {hist[-1][2]:.1f} (r{hist[-1][0]}) "
              f"-> search cost collapses -> PASS")

        # [4] SLEEP banked the programs: nodes exist, pipeline symbolic, depend edges, retrievable
        g = MemoryGraph.load_json(gp)
        progs = {nid: n for nid, n in g.nodes.items() if n.metadata.get("kind") == "program"}
        assert len(progs) == 6, list(progs)
        assert all("pipeline" in n.metadata for n in progs.values())
        assert g.edge_between("impl_max_lis", "impl_lis_length") is not None
        retr = MGRetriever(g, embed)
        # keyword-stub embed puts the atom (lis_length) and the program (max_lis) in the SAME region
        # (both texts say "increasing"); the property under test is the program is RETRIEVABLE at all
        got = retr.retrieve("the largest longest-increasing-subsequence length among arrays", k=2, min_cos=-1)
        assert any(name == "max_lis" for name, _ in got), got
        print(f"  [4] sleep: {len(progs)} program nodes banked (pipeline SYMBOLIC in metadata, depend "
              f"edges to atom closure) + retrieval surfaces them -> PASS")

        # [5] persistence: a FRESH net rebuilt from the GRAPH ALONE (no search) recalls everything
        fz, nf = rebuild_net(gp, embed, seed=99)
        assert fz == nf == 6, (fz, nf)
        print(f"  [5] rebuild-net: fresh net + graph-stored programs only -> {fz}/{nf} fams zero-shot "
              f"-> the GRAPH is the memory; the net re-amortizes -> PASS")

    print("\n  ALGO_GRR_LOOP SELFTEST -> PASS  (wake/sleep compounding loop closed: search discovers, "
          "net amortizes, graph REMEMBERS)")
    return True


def main():
    ap = argparse.ArgumentParser(description="GRR-8: unified wake/sleep compounding loop over the graph.")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--loop", action="store_true", help="run the loop with real mpnet (molab)")
    ap.add_argument("--rebuild", action="store_true", help="fresh net from graph-stored programs (molab)")
    ap.add_argument("--graph", default="graphs/algo_grr_loop.json")
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--budget", type=int, default=1500)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    if a.selftest:
        sys.exit(0 if _selftest() else 1)
    if a.loop or a.rebuild:
        from v5.runtime.algo_compose_tasks import seed_atom_graph
        if not Path(a.graph).exists():
            seed_atom_graph(a.graph, hard=True)
        embed = _mpnet_embed()
        if a.loop:
            print(f"GRR-8 loop (real mpnet): {a.graph} | rounds={a.rounds} budget={a.budget}", flush=True)
            wake_sleep_loop(a.graph, embed, rounds=a.rounds, budget=a.budget, seed=a.seed)
        if a.rebuild:
            print(f"GRR-8 rebuild-net (real mpnet): {a.graph}", flush=True)
            rebuild_net(a.graph, embed, seed=a.seed)
        return
    ap.print_help()


if __name__ == "__main__":
    main()
