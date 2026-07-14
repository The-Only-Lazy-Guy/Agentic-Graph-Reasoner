"""GRR audit tool — open the box: WHAT is actually in the graph, and HOW does the model solve, raw.

  --census   node-by-node census of a graph: content classes (pure code / NL text / symbolic pipeline /
             mixed), triviality proxies (code size, pipeline depth), provenance, verbatim samples.
             Answers: what % is trivial knowledge, what % pure code, what % NL-about-implementation.
  --trace    RAW step-by-step inference on N tasks: goal embed -> per-emission head logits (top-3 ops /
             atoms / aggs with probabilities) -> chosen step -> realized code -> verify result. On a
             failed decode, the ladder is traced too (beam candidates in score order, verify hits).
  --mbpp-baseline  the unseen-DOMAIN probe: run MBPP+ pipeline-shaped tasks (REAL human tasks, different
             vocabulary/signatures) through the CURRENT decode+beam ladder. The expected mostly-fail
             number is the honest pre-GRR-14 baseline that atom-authoring must beat.

  local, no GPU needed:  python -m v5.runtime.algo_grr_inspect --build --census --trace 3 --mbpp-baseline 40
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from graph_core import MemoryGraph

DEFAULT_GRAPH = "graphs/algo_grr_inspect.json"


# ═══════════════════════════════════════════════════════════════════════════════
# Shared: a local graph + embedder (real mpnet if installed, else the keyword stub)
# ═══════════════════════════════════════════════════════════════════════════════

def _embedder(pipes, d_in=64):
    try:
        from v5.memory.store import make_mpnet_embedder
        print("  [embedder: real mpnet]", flush=True)
        return make_mpnet_embedder()
    except Exception:
        from v5.runtime.algo_dsl_gen import pipe_text_intent_variants, pipe_text_variants
        rng = np.random.default_rng(0)
        base = {f: rng.standard_normal(d_in).astype("float32") for f in pipes}
        t2f = {t: f for f, p in pipes.items()
               for t in pipe_text_variants(p, 6) + pipe_text_intent_variants(p, 6)}
        print("  [embedder: keyword stub (sentence-transformers not installed locally)]", flush=True)

        def embed(d):
            out = {}
            for k, text in d.items():
                f = t2f.get(text)
                out[k] = ((base[f] if f in base else 0.05 * rng.standard_normal(d_in))
                          + 0.15 * rng.standard_normal(d_in)).astype("float32")
            return out

        return embed


def build_local(graph_path=DEFAULT_GRAPH, n_families=16, rounds=4, budget=400, seed=0):
    """Reproduce a factory graph LOCALLY (deterministic search + stub-LM proposer rung)."""
    from v5.runtime.algo_dsl_gen import gen_families
    from v5.runtime.algo_grr_loop import factory_domain, wake_sleep_loop
    from v5.runtime.algo_lm_proposer import make_stub_gen
    dom = factory_domain(n_families=n_families, fam_seed=3, para_train=3, para_eval=2, beam=8,
                         max_chain=4, explore=4)
    pipes = gen_families(n_families, seed=3, max_chain=4)
    embed = _embedder(pipes)
    Path(graph_path).parent.mkdir(parents=True, exist_ok=True)
    if Path(graph_path).exists():
        Path(graph_path).unlink()
    dom["seed_graph"](graph_path)
    _model, hist = wake_sleep_loop(graph_path, embed, rounds=rounds, budget=budget, sft_steps=800,
                                   n_wake=3, seed=seed, domain=dom, lm_gen=make_stub_gen(), lm_k=2)
    return graph_path, dom, embed


# ═══════════════════════════════════════════════════════════════════════════════
# --census
# ═══════════════════════════════════════════════════════════════════════════════

def census(graph_path: str, n_samples: int = 4):
    g = MemoryGraph.load_json(graph_path)
    rows = []
    for nid, n in g.nodes.items():
        md = n.metadata or {}
        code = md.get("code", "")
        rows.append(dict(
            id=nid, type=n.node_type, kind=md.get("kind", "atom" if code else "-"),
            has_code=bool(code), code_chars=len(code),
            has_nl=bool((n.text or "").strip()), nl_chars=len(n.text or ""),
            n_nl_phrasings=len(md.get("texts", [])) or (1 if (n.text or "").strip() else 0),
            has_symbolic=bool(md.get("pipeline")), pipe_len=len(md.get("pipeline") or []),
            origin=md.get("origin", "-"), edges_out=sum(1 for e in g.edges if e.src == nid),
        ))
    N = len(rows)
    n_code = sum(r["has_code"] for r in rows)
    n_nl = sum(r["has_nl"] for r in rows)
    n_sym = sum(r["has_symbolic"] for r in rows)
    n_pure_code = sum(1 for r in rows if r["has_code"] and not r["has_symbolic"])
    n_mixed = sum(1 for r in rows if r["has_code"] and r["has_symbolic"])
    n_nl_only = sum(1 for r in rows if r["has_nl"] and not r["has_code"])
    # triviality proxy: executable content below 60 chars (double = 2*n class) vs real algorithms
    trivial = [r for r in rows if r["has_code"] and r["code_chars"] < 60 and r["pipe_len"] <= 1]
    print(f"\n=== CENSUS: {graph_path} — {N} nodes, {len(g.edges)} edges ===")
    print(f"  content classes:")
    print(f"    NL-only (concept labels)          : {n_nl_only:3d}  ({n_nl_only/N:.0%})")
    print(f"    pure code + NL doc line (atoms)   : {n_pure_code:3d}  ({n_pure_code/N:.0%})")
    print(f"    code + SYMBOLIC pipeline + NL     : {n_mixed:3d}  ({n_mixed/N:.0%})  <- programs")
    print(f"  any code: {n_code}/{N} ({n_code/N:.0%}) | any NL: {n_nl}/{N} ({n_nl/N:.0%}) | "
          f"symbolic form: {n_sym}/{N} ({n_sym/N:.0%})")
    print(f"  NL is DESCRIPTIVE (retrieval keys / task phrasings), never step-by-step instructions: "
          f"the executable knowledge is the CODE, the compositional knowledge is the PIPELINE.")
    print(f"  triviality proxy (code<60 chars, depth<=1): {len(trivial)}/{n_code} of code nodes "
          f"({len(trivial)/max(1,n_code):.0%}) — {', '.join(r['id'] for r in trivial) or '-'}")
    by_origin = {}
    for r in rows:
        if r["kind"] == "program":
            by_origin[r["origin"]] = by_origin.get(r["origin"], 0) + 1
    print(f"  program provenance: {by_origin}")
    print(f"\n  --- verbatim samples ---")
    samples = ([r for r in rows if r["kind"] == "-"][:1]
               + [r for r in rows if r["kind"] == "atom"][:1]
               + [r for r in rows if r["kind"] == "program"][:max(1, n_samples - 2)])
    for r in samples:
        n = g.nodes[r["id"]]
        md = n.metadata or {}
        print(f"\n  [{r['id']}]  type={r['type']} kind={r['kind']} origin={r['origin']}")
        print(f"    text (NL): {n.text!r}")
        if md.get("texts"):
            for t in md["texts"][1:2]:
                print(f"    alt phrasing: {t!r}")
        if md.get("pipeline"):
            print(f"    pipeline (SYMBOLIC): {md['pipeline']}")
        if md.get("code"):
            pad = "\n      ".join(md["code"].splitlines())
            print(f"    code:\n      {pad}")
    return rows


# ═══════════════════════════════════════════════════════════════════════════════
# --trace: raw stepwise inference
# ═══════════════════════════════════════════════════════════════════════════════

def _decode_verbose(model, gv, atom_names, atom_vecs, max_len=8):
    import torch
    from v5.runtime.algo_dsl import Op
    from v5.runtime.algo_dsl_trm import AGGS, OPS
    g = model.goal_proj(torch.as_tensor(gv))
    A = model.atom_proj(torch.as_tensor(atom_vecs, dtype=torch.float32))
    state = torch.zeros(model.d)
    pipe = []
    print("    RAW DECODE (per emission: head distributions -> committed step):")
    for step in range(max_len):
        ctx = torch.cat([g, state])
        op_l = model.op_head(ctx)
        agg_l = model.agg_head(ctx)
        atom_l = A @ model.q_atom(ctx)
        opp = torch.softmax(op_l, -1)
        top_ops = sorted(zip(OPS, opp.tolist()), key=lambda x: -x[1])[:3]
        print(f"      step {step}: op head  -> " + ", ".join(f"{o} {p:.2f}" for o, p in top_ops))
        op = int(op_l.argmax())
        if OPS[op] == "REDUCE":
            ap = torch.softmax(agg_l, -1)
            top_aggs = sorted(zip(AGGS, ap.tolist()), key=lambda x: -x[1])[:3]
            print(f"               agg head -> " + ", ".join(f"{a} {p:.2f}" for a, p in top_aggs))
            pipe.append(Op("REDUCE", AGGS[int(agg_l.argmax())]))
            print(f"               COMMIT REDUCE({pipe[-1].arg}) -> halt")
            break
        atp = torch.softmax(atom_l, -1)
        top_atoms = sorted(zip(atom_names, atp.tolist()), key=lambda x: -x[1])[:3]
        print(f"               atom ptr -> " + ", ".join(f"{a} {p:.2f}" for a, p in top_atoms))
        ai = int(atom_l.argmax())
        pipe.append(Op(OPS[op], atom_names[ai]))
        print(f"               COMMIT {OPS[op]}({atom_names[ai]})")
        import torch as _t
        state = model.rnn((model.op_emb(_t.tensor(op)) + A[ai]).unsqueeze(0), state.unsqueeze(0))[0]
    return pipe


def trace(graph_path: str, dom, embed, n_tasks: int = 3, seed: int = 0):
    """End-to-end raw trace: rebuild the net FROM THE GRAPH (the deployment path), then solve fresh
    tasks stepwise: embed -> decode (logits shown) -> realize -> verify. Ladder shown on failure."""
    import torch
    from v5.runtime.algo_dsl import realize_program, atoms_of
    from v5.runtime.algo_dsl_gen import GEN_ATOMS
    from v5.runtime.algo_dsl_trm import _build, _sft_steps, program_to_steps
    from v5.runtime.algo_grr_loop import _graph_atoms
    from v5.runtime.algo_dsl import Op
    torch_, _nn, ProgramDecoder = _build()
    torch_.manual_seed(seed)
    g = MemoryGraph.load_json(graph_path)
    atom_names, atom_idx, atom_vecs = _graph_atoms(g, embed)
    progs = [(nid, n) for nid, n in g.nodes.items() if n.metadata.get("kind") == "program"]
    traces = []
    for _nid, n in progs:
        pipe = [Op(k, a) for k, a in n.metadata["pipeline"]]
        for text in n.metadata.get("texts", [n.text]):
            gv = np.asarray(list(embed({"q": text}).values())[0], dtype=np.float32)
            traces.append((gv, program_to_steps(pipe, atom_idx)))
    model = ProgramDecoder(d_in=atom_vecs.shape[1], d=64)
    opt = torch_.optim.Adam(model.parameters(), lr=1e-3)
    _sft_steps(model, opt, traces, torch_.as_tensor(atom_vecs, dtype=torch.float32), 1000, seed=seed)
    print(f"\n=== RAW TRACE: net rebuilt from the GRAPH ALONE ({len(progs)} programs, "
          f"{len(traces)} stored phrasings) ===")

    by = dom["tasks_by_fam"](901)                       # fresh eval-side instances
    chk = dom["make_is_general"](lambda a: "")
    shown = 0
    for fam, ts in sorted(by.items()):
        if shown >= n_tasks:
            break
        t = ts[0]
        shown += 1
        print(f"\n  -- task {shown}: family={fam}")
        print(f"    text: {t.text!r}")
        gv = np.asarray(list(embed({"q": t.text}).values())[0], dtype=np.float32)
        print(f"    goal embed: dim={gv.shape[0]}, |v|={float(np.linalg.norm(gv)):.2f}")
        pipe = _decode_verbose(model, gv, atom_names, atom_vecs)
        try:
            code = realize_program(fam, "list", pipe)
            pad = "\n      ".join(code.splitlines())
            print(f"    REALIZED code (calls atoms, never inlines):\n      {pad}")
            deps = sorted(atoms_of(pipe))
            print(f"    deps resolved through the graph: {deps}")
        except Exception as e:
            print(f"    realize failed: {e}")
        ok = chk(pipe, fam)
        print(f"    VERIFY vs oracle on random inputs (two disjoint sets): "
              f"{'GENERAL — SOLVED' if ok else 'FAILED'}")


# ═══════════════════════════════════════════════════════════════════════════════
# --mbpp-baseline: the unseen-DOMAIN probe (real human tasks through the current ladder)
# ═══════════════════════════════════════════════════════════════════════════════

def mbpp_baseline(graph_path: str, embed, n: int = 40,
                  corpus: str = "artifacts/mbpp_plus_prepped.jsonl", seed: int = 0):
    """MBPP+ pipeline-shaped tasks -> decode + beam over OUR atom vocabulary, verified by the task's
    own asserts. Different domain: human phrasing, alien entry-point names/signatures. The honest
    expectation is ~0 — that number is the pre-GRR-14 baseline atom-authoring must beat."""
    import torch
    from v5.runtime.algo_dsl import realize_program
    from v5.runtime.algo_dsl_gen import GEN_ATOMS
    from v5.runtime.algo_dsl_trm import _build, _beam_search, program_to_steps
    from v5.runtime.algo_grr_loop import _graph_atoms
    from v5.runtime.algo_mbpp_prep import load_prepped
    from v5.runtime.algo_dsl import Op, atoms_of
    torch_, _nn, ProgramDecoder = _build()
    torch_.manual_seed(seed)
    g = MemoryGraph.load_json(graph_path)
    atom_names, atom_idx, atom_vecs = _graph_atoms(g, embed)
    model = ProgramDecoder(d_in=atom_vecs.shape[1], d=64)     # cold decode is honest here
    tasks = load_prepped(corpus, limit=n, pipeline_only=True)
    deps_all = "\n\n".join(GEN_ATOMS[a][1] for a in GEN_ATOMS)

    def try_pipe(task, pipe):
        try:
            code = realize_program(task.name, "list", pipe)
        except Exception:
            return False
        return task.verify(code, deps_all)

    solved = 0
    fams = {t.name: ("list", None) for t in tasks}
    for t in tasks:
        gv = np.asarray(list(embed({"q": t.text}).values())[0], dtype=np.float32)
        found, _used, _origin = _beam_search(
            model, gv, t.name, fams, lambda a: deps_all, atom_names,
            atom_vecs, max_transforms=3, budget=150, B=8, explore=4, seed=seed,
            is_general=lambda pipe, fam, _t=t: try_pipe(_t, pipe))
        solved += bool(found)
    print(f"\n=== UNSEEN-DOMAIN BASELINE: MBPP+ pipeline-shaped through the current ladder ===")
    print(f"  {solved}/{len(tasks)} solved ({solved/max(1,len(tasks)):.0%}) — human tasks, alien "
          f"signatures, no authored atoms. This is the number GRR-14 (LM authors atoms) must beat.")
    return solved, len(tasks)


def main():
    ap = argparse.ArgumentParser(description="GRR audit: census / raw trace / unseen-domain baseline.")
    ap.add_argument("--build", action="store_true", help="(re)build the local factory graph first")
    ap.add_argument("--graph", default=DEFAULT_GRAPH)
    ap.add_argument("--census", action="store_true")
    ap.add_argument("--trace", type=int, default=0, metavar="N", help="raw stepwise trace on N tasks")
    ap.add_argument("--mbpp-baseline", type=int, default=0, metavar="N")
    ap.add_argument("--families", type=int, default=16)
    a = ap.parse_args()
    from v5.runtime.algo_dsl_gen import gen_families
    from v5.runtime.algo_grr_loop import factory_domain
    pipes = gen_families(a.families, seed=3, max_chain=4)
    if a.build:
        graph_path, dom, embed = build_local(a.graph, n_families=a.families)
    else:
        dom = factory_domain(n_families=a.families, fam_seed=3, para_train=3, para_eval=2, beam=8,
                             max_chain=4, explore=4)
        embed = _embedder(pipes)
        graph_path = a.graph
    if a.census:
        census(graph_path)
    if a.trace:
        trace(graph_path, dom, embed, n_tasks=a.trace)
    if a.mbpp_baseline:
        mbpp_baseline(graph_path, embed, n=a.mbpp_baseline)


if __name__ == "__main__":
    main()
