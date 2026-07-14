"""GRR-8: THE unified wake/sleep compounding loop — every validated stage, ONE driver over ONE graph.

  WAKE         per task (curriculum: the transform budget grows with rounds): DECODE (recall) -> verify;
               on fail, net-GUIDED SEARCH under a verify budget (algo_dsl_trm.solve_with_search — the
               until-solve inference primitive; beam search for deep programs)
  CONSOLIDATE  SFT the net on every discovery so far (STaR: the decode amortizes the search)
  SLEEP        write each newly-discovered family PROGRAM into the GRAPH as an implementation node —
               code = the realized program (CALLS atoms, never inlines), the pipeline rides SYMBOLICALLY
               in metadata (the graph stores the program's FORM), depend edges to its atom closure —
               via graph_grower's health-gated apply. Then re-index retrieval: the action space IS the
               graph, and it grew.
  MEASURE      per round: net zero-shot solve (decode-only, fresh goals), mean verifies-to-solve,
               graph size. Compounding = zero-shot RISES while verifies-to-solve FALLS.

--rebuild-net  fresh net, NO search: SFT purely from the GRAPH's stored program nodes, then eval.
               Content is symbolic in the graph (survives any net/box reset); the net just re-amortizes.

DOMAINS (GRR-9c): the loop is domain-parametrized.
  hand6    the 6 hand-written families (algo_compose_tasks oracles, exhaustive-guided search) — the
           original validation domain, unchanged.
  factory  algo_dsl_gen: N generated families, chain depth to ~6, PARAPHRASED goals (train phrasings !=
           eval phrasings — zero-shot is measured on HELD-OUT phrasings), BEAM search (exhaustive levels
           die combinatorially at this depth), pipe_is_general verifier. The scale-up domain.
Discovery is search + oracle-I/O verify only — reference pipelines are NEVER read by the searcher (they
define each family's oracle BEHAVIOR and the node text, nothing else).

  selftest (no LM):     python -m v5.runtime.algo_grr_loop --selftest
  molab (real mpnet):   python -m v5.runtime.algo_grr_loop --loop --graph graphs/algo_grr_loop.json
                        python -m v5.runtime.algo_grr_loop --loop --factory --families 24 --rounds 5 \
                            --graph graphs/algo_grr_factory.json
                        python -m v5.runtime.algo_grr_loop --rebuild [--factory --families 24] --graph ...
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from graph_core import MemoryGraph
from v5.runtime.algo_dsl import _PROGRAMS, Op, atoms_of, realize_program
from v5.runtime.algo_dsl_trm import (_build, _is_general, _sft_steps, program_to_steps,
                                     solve_with_search)
from v5.runtime.algo_graph_edits import edge_candidate, grow, node_candidate
from v5.runtime.algo_graph_mg import MGRetriever, _fn_name

# hand6 family SCOPE + input kind only — reference pipelines are never read by the search
FAM_KINDS = {f: (k, None) for f, (k, _p) in _PROGRAMS.items()}


# ═══════════════════════════════════════════════════════════════════════════════
# DOMAINS — everything family-specific lives here; the loop below is domain-blind.
# ═══════════════════════════════════════════════════════════════════════════════

def hand6_domain():
    """The original 6-family validation domain (algo_compose_tasks oracles, exhaustive-guided search)."""
    from v5.runtime.algo_compose_tasks import _TEXT, gen_compose_tasks, seed_atom_graph

    def tasks_by_fam(seed):
        by = {}
        for t in list(gen_compose_tasks(60, seed=seed)) + list(gen_compose_tasks(60, seed=seed, hard=True)):
            if t.name in FAM_KINDS:
                by.setdefault(t.name, []).append(t)
        return by

    def eval_texts(seed):
        return {f: [t.text for t in ts[:3]] for f, ts in tasks_by_fam(seed).items()}

    return dict(
        name="hand6", fams=FAM_KINDS, text_of=_TEXT,
        texts_of={f: [_TEXT[f]] for f in FAM_KINDS},
        make_is_general=lambda resolve_fn: (lambda p, f: _is_general(p, f, FAM_KINDS, resolve_fn, 24)),
        tasks_by_fam=tasks_by_fam, eval_texts=eval_texts,
        seed_graph=lambda path: seed_atom_graph(path, hard=True),
        curriculum=lambda r: 1 if r == 0 else 2, beam=0,
    )


def factory_domain(n_families=24, fam_seed=0, para_train=3, para_eval=2, beam=12, max_chain=4,
                   explore=6, tier: str = "method"):
    """The scale-up domain: algo_dsl_gen factory families, paraphrased goals, beam search.
    Zero-shot eval decodes HELD-OUT phrasings (never trained on) — generalization, not point recall.
    tier="intent" swaps EVERY text for the intent tier (WHAT, never HOW — zero method vocabulary):
    first-encounter solve there vs method tier = the measured reasoning-vs-translation delta."""
    from v5.runtime.algo_dsl_gen import (GEN_ATOMS, GenTask, _rand_list, gen_families, gen_tasks,
                                         interpret, pipe_is_general, pipe_text_intent_variants,
                                         pipe_text_variants)
    pipes = gen_families(n_families, seed=fam_seed, max_chain=max_chain)
    fams = {f: ("list", None) for f in pipes}
    variants_fn = pipe_text_intent_variants if tier == "intent" else pipe_text_variants
    allv = {f: variants_fn(p, para_train + para_eval) for f, p in pipes.items()}

    def tasks_by_fam(seed):
        import numpy as _np
        by = {}
        if tier == "intent":                              # instances cycle the intent train phrasings
            rng = _np.random.default_rng(seed)
            for fam, pipe in pipes.items():
                train = allv[fam][:para_train] or allv[fam]      # robust to short variant lists
                for i in range(max(3, para_train)):
                    lst = _rand_list(rng)
                    by.setdefault(fam, []).append(
                        GenTask(fam, train[i % len(train)],
                                [f"assert {fam}({lst!r}) == {interpret(pipe, lst)!r}"]))
        else:
            for t in gen_tasks(pipes, n_per=max(3, para_train), seed=seed, paraphrase_k=para_train):
                by.setdefault(t.name, []).append(t)
        return by

    def seed_graph(path):
        nodes = [{"id": "concept_algorithms", "text": "algorithms", "node_type": "concept"}]
        for a, (text, code, _fn, _role) in GEN_ATOMS.items():
            nodes.append({"id": f"impl_{a}", "text": text, "node_type": "implementation",
                          "metadata": {"code": code}})
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps({"metadata": {}, "nodes": nodes, "edges": []}), encoding="utf-8")

    maxlen = max(len(p) for p in pipes.values())
    return dict(
        name="factory", fams=fams, text_of={f: allv[f][0] for f in pipes},
        texts_of={f: allv[f][:para_train] for f in pipes},   # the goal REGION the graph must remember
        make_is_general=lambda _resolve_fn: (lambda p, f: pipe_is_general(p, pipes, f, n=24)),
        tasks_by_fam=tasks_by_fam,
        # HELD-OUT phrasings (falls back to the last variant if a family produced few — never empty)
        eval_texts=lambda _seed: {f: (allv[f][para_train:] or allv[f][-1:]) for f in pipes},
        seed_graph=seed_graph,
        curriculum=lambda r: min(maxlen - 1, 2 + r), beam=beam, explore=explore,
        all_texts=[t for vs in allv.values() for t in vs],   # warm the embed cache in ONE batched encode
        lm_vocab=([a for a, v in GEN_ATOMS.items() if v[3] == "pred"],
                  [a for a, v in GEN_ATOMS.items() if v[3] == "map"]),   # proposer vocabulary
    )


# ═══════════════════════════════════════════════════════════════════════════════
# The loop (domain-blind)
# ═══════════════════════════════════════════════════════════════════════════════

def _cached_embed(embed_fn):
    """Text-keyed embedding cache. The loop re-embeds the SAME texts constantly (eval phrasings every
    round, wake phrasings cycling) — measured a large share of the 1h40m molab wall time. mpnet is
    deterministic per text, so this is free. Misses are encoded in ONE batched call (single-text mpnet
    calls pay per-call overhead), so warming the cache with every known text upfront is one encode."""
    cache: dict = {}

    def f(d):
        missing = {f"m{i}": t for i, t in enumerate({t for t in d.values() if t not in cache})}
        if missing:
            got = embed_fn(missing)
            for k, t in missing.items():
                cache[t] = got[k]
        return {k: cache[t] for k, t in d.items()}

    return f


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


def _zero_shot(model, atom_names, atom_vecs, embed_fn, chk, texts_by_fam, max_len=8):
    """Decode-ONLY solve on the given goals (no search, no updates) — the recall/amortization measure.
    For the factory domain the goals are HELD-OUT phrasings, so this is paraphrase generalization.
    Returns (fams_solved, n_fams, inst_solved, inst_total, per_fam={fam: (ok, total)})."""
    import torch
    A = torch.as_tensor(atom_vecs, dtype=torch.float32)
    fams_solved, inst_solved, inst_total, per_fam = 0, 0, 0, {}
    for fam, texts in texts_by_fam.items():
        ok = 0
        for text in texts:
            gv = np.asarray(list(embed_fn({"q": text}).values())[0], dtype=np.float32)
            pipe = model.decode(torch.as_tensor(gv), A, atom_names, max_len=max_len)
            ok += int(chk(pipe, fam))
        inst_solved += ok; inst_total += len(texts)
        per_fam[fam] = (ok, len(texts))
        fams_solved += int(ok == len(texts) and ok > 0)
    return fams_solved, len(texts_by_fam), inst_solved, inst_total, per_fam


def _sleep_store(graph_path: str, retr: MGRetriever, discoveries: dict, rnd: int, domain: dict,
                 concept: str = "concept_algorithms", origins: dict | None = None):
    """SLEEP write-back: each discovered family program -> an implementation node (kind=program, pipeline
    symbolic in metadata) + depend edges to the atoms it calls + part_of the concept. Health-gated via
    graph_grower. `origins[fam]` (beam|epsilon|guided) rides in metadata — the provenance behind the
    "how often does exploration discover programs that get reused?" metric. Returns #stored."""
    g = retr.graph
    cands = []
    for fam, pipe in discoveries.items():
        nid = f"impl_{fam}"
        if nid in g.nodes:                                  # already banked (or name collision) -> skip
            continue
        code = realize_program(fam, domain["fams"][fam][0], pipe)
        cands.append(node_candidate(
            nid, code, domain["text_of"][fam], f"grr8_r{rnd}",
            metadata={"kind": "program", "family": fam, "input_kind": domain["fams"][fam][0],
                      "pipeline": [[op.kind, op.arg] for op in pipe],
                      "origin": (origins or {}).get(fam, "unknown"), "found_round": rnd,
                      # the goal REGION, not a point: every train phrasing rides with the program, so a
                      # rebuilt net generalizes to held-out phrasings (one stored phrasing measured 5/24)
                      "texts": domain["texts_of"][fam]}))
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
                    sft_steps: int = 400, n_wake: int = 2, seed: int = 0, log: bool = True,
                    domain: dict | None = None, lm_gen=None, lm_k: int = 6, lm_hint: bool = False):
    """The loop. Returns (model, hist) with hist rows:
    (round, n_search_discoveries, mean_verifies_to_solve, fams_zero_shot, n_fams, graph_nodes).
    lm_gen plugs the GRR-12 escalation ladder: decode -> beam+epsilon -> LM PROPOSER (task text ->
    candidate pipelines, SAME verify gate, origin='lm') — motivated by the measured search ceiling."""
    import torch
    domain = domain or hand6_domain()
    torch, _nn, ProgramDecoder = _build()
    torch.manual_seed(seed)
    embed_fn = _cached_embed(embed_fn)
    if domain.get("all_texts"):                             # one batched encode instead of N singles
        embed_fn({f"w{i}": t for i, t in enumerate(domain["all_texts"])})
    g = MemoryGraph.load_json(graph_path)
    atom_names, atom_idx, atom_vecs = _graph_atoms(g, embed_fn)
    retr = MGRetriever(g, embed_fn)
    resolve_fn = lambda atoms: retr.resolve_deps(atoms) if atoms else ""
    chk = domain["make_is_general"](resolve_fn)
    model = ProgramDecoder(d_in=atom_vecs.shape[1], d=64)
    opt = torch.optim.Adam(model.parameters(), lr=5e-4)
    A = torch.as_tensor(atom_vecs, dtype=torch.float32)
    pool, known, origin_of = [], {}, {}                     # known: fam -> discovered pipe
    hist = []
    for r in range(rounds):
        mt = domain["curriculum"](r)
        by = domain["tasks_by_fam"](100 + r)
        n_disc, verifies = 0, []
        failed_this_round = set()                           # a fam that just failed a FULL search will
        for fam, ts in sorted(by.items()):                  # fail its siblings too (same net, same goal
            for t in ts[:n_wake]:                           # region) -> search once per fam per round
                res = solve_with_search(model, t, domain["fams"], atom_names, atom_vecs, resolve_fn,
                                        embed_fn, atom_idx, opt=opt,
                                        budget=0 if fam in failed_this_round else budget,
                                        max_transforms=mt, seed=seed + r,
                                        is_general=chk, beam=domain["beam"],
                                        explore=domain.get("explore", 0))
                if not res["solved"] and fam not in failed_this_round:
                    failed_this_round.add(fam)
                    if lm_gen is not None and domain.get("lm_vocab") and fam not in known:
                        # GRR-12 ladder rung: search failed -> the LM proposes from the TASK TEXT,
                        # candidates pass the SAME gate (MDL-first). A hit is a discovery like any other.
                        # GRR-16: lm_hint hands the TRM's confidence-gated THOUGHT to the LM as text.
                        from v5.runtime.algo_lm_proposer import propose_and_verify
                        preds, maps = domain["lm_vocab"]
                        sk = None
                        if lm_hint:
                            from v5.runtime.algo_dsl_trm import sketch as _sketch
                            gv_h = np.asarray(list(embed_fn({"q": t.text}).values())[0], dtype=np.float32)
                            sk = _sketch(model, gv_h, atom_names, atom_vecs)
                        pipe_lm, nver = propose_and_verify(lm_gen, t.text, fam, chk, preds, maps,
                                                           k=lm_k, sketch=sk)
                        if pipe_lm is not None:
                            res = dict(solved=True, pipe=pipe_lm, via="search",
                                       verifies=res["verifies"] + nver, origin="lm")
                            _sft_steps(model, opt, [(np.asarray(list(embed_fn({"q": t.text}).values())[0],
                                                                dtype=np.float32),
                                                     program_to_steps(pipe_lm, atom_idx))], A, 150,
                                       seed=seed + r)
                if res["solved"]:
                    verifies.append(res["verifies"])
                    if res["via"] == "search":
                        n_disc += 1
                        if fam not in known:                # first discovery fixes the origin credit
                            origin_of[fam] = res.get("origin") or "search"
                        known[fam] = res["pipe"]
                    gv = np.asarray(list(embed_fn({"q": t.text}).values())[0], dtype=np.float32)
                    pool.append((gv, program_to_steps(known.get(fam, res["pipe"]), atom_idx)))
        _sft_steps(model, opt, pool, A, sft_steps, seed=seed + r)          # CONSOLIDATE (replay)
        stored = _sleep_store(graph_path, retr, known, r, domain, origins=origin_of)
        if stored:
            retr = MGRetriever(MemoryGraph.load_json(graph_path), embed_fn)
            resolve_fn = lambda atoms: retr.resolve_deps(atoms) if atoms else ""
            chk = domain["make_is_general"](resolve_fn)
        fz, nf, iz, it, per_fam = _zero_shot(model, atom_names, atom_vecs, embed_fn, chk,
                                             domain["eval_texts"](900 + r))
        mv = sum(verifies) / max(1, len(verifies))
        hist.append((r, n_disc, mv, fz, nf, len(retr.graph.nodes)))
        if log:
            print(f"  round {r}: mt<={mt} | search-discovered {n_disc} | verifies-to-solve {mv:.1f} | "
                  f"zero-shot {fz}/{nf} fams ({iz}/{it} inst) | graph {len(retr.graph.nodes)} nodes "
                  f"{len(retr.graph.edges)} edges", flush=True)
    if log and origin_of:
        # THE EPSILON-REUSE METRIC: for each discovery origin, how many of its finds are now REUSED
        # zero-shot (the net decodes them on held-out goals with no search)? Exploration that finds
        # load-bearing programs shows up here; junk finds don't get reused.
        print("  --- discovery provenance -> reuse (held-out zero-shot) ---", flush=True)
        for origin in sorted({o for o in origin_of.values()}):
            fams_o = [f for f, o in origin_of.items() if o == origin]
            reused = [f for f in fams_o if per_fam.get(f, (0, 0))[0] > 0]
            inst_ok = sum(per_fam.get(f, (0, 0))[0] for f in fams_o)
            inst_tot = sum(per_fam.get(f, (0, 0))[1] for f in fams_o)
            print(f"  {origin:8s}: discovered {len(fams_o)} fams -> reused {len(reused)} "
                  f"({inst_ok}/{inst_tot} held-out inst) : {', '.join(sorted(reused)) or '-'}", flush=True)
    return model, hist


def rebuild_net(graph_path: str, embed_fn, sft_steps: int = 800, seed: int = 0, log: bool = True,
                domain: dict | None = None):
    """The persistence proof: a FRESH net, NO search — SFT purely from the graph's stored program nodes
    (pipeline is symbolic in metadata), then decode-only eval on fresh goals (factory: HELD-OUT
    phrasings). The graph IS the memory; the net re-amortizes it in seconds. Returns (fams, n_fams)."""
    import torch
    domain = domain or hand6_domain()
    torch, _nn, ProgramDecoder = _build()
    torch.manual_seed(seed)
    embed_fn = _cached_embed(embed_fn)
    g = MemoryGraph.load_json(graph_path)
    atom_names, atom_idx, atom_vecs = _graph_atoms(g, embed_fn)
    progs = [(nid, n) for nid, n in g.nodes.items() if n.metadata.get("kind") == "program"]
    traces = []
    for nid, n in progs:
        pipe = [Op(k, a) for k, a in n.metadata["pipeline"]]
        steps = program_to_steps(pipe, atom_idx)
        for text in n.metadata.get("texts", [n.text]):       # every stored phrasing (the goal REGION)
            gv = np.asarray(list(embed_fn({"q": text}).values())[0], dtype=np.float32)
            traces.append((gv, steps))
    model = ProgramDecoder(d_in=atom_vecs.shape[1], d=64)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    _sft_steps(model, opt, traces, torch.as_tensor(atom_vecs, dtype=torch.float32), sft_steps, seed=seed)
    retr = MGRetriever(g, embed_fn)
    resolve_fn = lambda atoms: retr.resolve_deps(atoms) if atoms else ""
    chk = domain["make_is_general"](resolve_fn)
    fz, nf, iz, it, _pf = _zero_shot(model, atom_names, atom_vecs, embed_fn, chk, domain["eval_texts"](901))
    if log:
        print(f"  rebuild-net: fresh net + {len(progs)} graph-stored programs ({len(traces)} stored "
              f"phrasings, no search, sft={sft_steps}) -> zero-shot {fz}/{nf} fams ({iz}/{it} inst)",
              flush=True)
    return fz, nf


def _mpnet_embed():
    from v5.memory.store import make_mpnet_embedder
    return make_mpnet_embedder()


# ═══════════════════════════════════════════════════════════════════════════════
# SELFTEST — [1-5] the hand6 loop (unchanged semantics), [6] the FACTORY domain at scale-down:
# beam search discovers deep generated families, banks them, zero-shot measured on HELD-OUT phrasings.
# ═══════════════════════════════════════════════════════════════════════════════

def _selftest() -> bool:
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

    # [6] FACTORY domain at scale-down: beam search + paraphrased goals + held-out-phrasing zero-shot
    from v5.runtime.algo_dsl_gen import gen_families, pipe_text_variants
    dom = factory_domain(n_families=10, fam_seed=3, para_train=2, para_eval=1, beam=10, max_chain=3)
    pipes = gen_families(10, seed=3, max_chain=3)
    fam_base2 = {f: rng.standard_normal(d_in).astype("float32") for f in pipes}
    text2fam = {t: f for f, p in pipes.items() for t in pipe_text_variants(p, 6)}

    def embed2(d):
        out = {}
        for k, text in d.items():
            f = text2fam.get(text)
            out[k] = ((fam_base2[f] if f in fam_base2 else 0.05 * rng.standard_normal(d_in))
                      + 0.15 * rng.standard_normal(d_in)).astype("float32")
        return out

    with tempfile.TemporaryDirectory() as td:
        gp2 = str(Path(td) / "gf.json")
        dom["seed_graph"](gp2)
        model2, hist2 = wake_sleep_loop(gp2, embed2, rounds=4, budget=400, sft_steps=600, n_wake=3,
                                        seed=0, domain=dom)
        disc = sum(h[1] for h in hist2)
        assert disc >= 6, hist2                                   # beam search discovers deep families
        assert hist2[-1][3] >= 0.6 * hist2[-1][4], hist2[-1]      # held-out-phrasing zero-shot majority
        g2 = MemoryGraph.load_json(gp2)
        progs2 = [nid for nid, n in g2.nodes.items() if n.metadata.get("kind") == "program"]
        assert len(progs2) >= 6, progs2
        fz2, nf2 = rebuild_net(gp2, embed2, seed=7, domain=dom, log=False)
        assert fz2 >= 0.6 * nf2, (fz2, nf2)
        print(f"  [6] FACTORY domain: beam search discovered {disc} deep families "
              f"(len<={max(len(p) for p in pipes.values())}), zero-shot on HELD-OUT phrasings "
              f"{hist2[-1][3]}/{hist2[-1][4]} fams, {len(progs2)} programs banked, rebuild "
              f"{fz2}/{nf2} -> PASS")

    print("\n  ALGO_GRR_LOOP SELFTEST -> PASS  (wake/sleep compounding loop closed: search discovers, "
          "net amortizes, graph REMEMBERS — hand6 AND factory domains)")
    return True


def main():
    ap = argparse.ArgumentParser(description="GRR-8: unified wake/sleep compounding loop over the graph.")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--loop", action="store_true", help="run the loop with real mpnet (molab)")
    ap.add_argument("--rebuild", action="store_true", help="fresh net from graph-stored programs (molab)")
    ap.add_argument("--factory", action="store_true", help="factory domain (generated fams, paraphrases, beam)")
    ap.add_argument("--families", type=int, default=24)
    ap.add_argument("--fam-seed", type=int, default=0)
    ap.add_argument("--para-train", type=int, default=3)
    ap.add_argument("--para-eval", type=int, default=2)
    ap.add_argument("--beam", type=int, default=12)
    ap.add_argument("--explore", type=int, default=6,
                    help="epsilon slots in the beam (random extensions kept besides the top-B)")
    ap.add_argument("--intent", action="store_true",
                    help="intent-tier texts (WHAT, never HOW) — the reasoning-vs-translation test")
    ap.add_argument("--graph", default="graphs/algo_grr_loop.json")
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--budget", type=int, default=1500)
    ap.add_argument("--n-wake", type=int, default=2)
    ap.add_argument("--sft-steps", type=int, default=400, help="consolidation SFT steps per round")
    ap.add_argument("--lm", default="", help="HF model for the GRR-12 proposer ladder (e.g. Qwen/Qwen2.5-3B)")
    ap.add_argument("--lm-k", type=int, default=6, help="LM proposal samples per stuck family")
    ap.add_argument("--lm-hint", action="store_true",
                    help="GRR-16: include the TRM's confidence-gated sketch in the proposer prompt")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    if a.selftest:
        sys.exit(0 if _selftest() else 1)
    if a.loop or a.rebuild:
        if a.factory:
            domain = factory_domain(a.families, a.fam_seed, a.para_train, a.para_eval, a.beam,
                                    explore=a.explore, tier="intent" if a.intent else "method")
        else:
            domain = hand6_domain()
        if not Path(a.graph).exists():
            domain["seed_graph"](a.graph)
        embed = _mpnet_embed()
        if a.loop:
            lm_gen = None
            if a.lm:
                from v5.runtime.algo_lm_proposer import make_hf_gen
                lm_gen = make_hf_gen(a.lm)
            print(f"GRR-8 loop (real mpnet, domain={domain['name']}): {a.graph} | rounds={a.rounds} "
                  f"budget={a.budget} n_wake={a.n_wake} sft_steps={a.sft_steps} "
                  f"lm={a.lm or 'off'} hint={a.lm_hint}", flush=True)
            wake_sleep_loop(a.graph, embed, rounds=a.rounds, budget=a.budget, n_wake=a.n_wake,
                            sft_steps=a.sft_steps, seed=a.seed, domain=domain,
                            lm_gen=lm_gen, lm_k=a.lm_k, lm_hint=a.lm_hint)
        if a.rebuild:
            print(f"GRR-8 rebuild-net (real mpnet, domain={domain['name']}): {a.graph}", flush=True)
            rebuild_net(a.graph, embed, sft_steps=max(800, a.sft_steps * 2), seed=a.seed, domain=domain)
        return
    ap.print_help()


if __name__ == "__main__":
    main()
