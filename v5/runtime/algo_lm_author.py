"""GRR-14: the LM AUTHORS NEW ATOMS — the invention rung, on real open-source tasks.

Measured motivation: MBPP+ pipeline-shaped human tasks through the current ladder = 1/40 (2%). Outside
its atom vocabulary the system is dead — composition cannot invent PRIMITIVES. This module adds the
final rung: when no pipeline over existing atoms can solve a task, the LM WRITES code.

  retrieve   MGRetriever advertises the graph's existing atoms for the task (reuse pressure: the prompt
             SHOWS what already exists, so the LM composes rather than reinvents)
  author     the LM writes the solution function (algo_graph_mg.solve_mg — the validated loop)
  GATE       the task's OWN dense tests (MBPP+ original asserts + the full EvalPlus script, subprocess)
             — GRR-1 epistemics on real data: solutions that only fit one assert die here
  BANK       the verified solution becomes an implementation node (origin="lm_author", text = the human
             task text = the retrieval key) + depend edges to every atom it CALLS + part_of the concept
             — health-gated through graph_grower like every other write
  reuse      later tasks retrieve earlier AUTHORED atoms; when a new solution calls one, that's
             CROSS-TASK REUSE — the compounding question, now on real data. Counted and reported.

Same contract as everywhere: the LM sees the task TEXT + advertised atom signatures — never reference
solutions. The gate, not the model, decides what enters memory.

  selftest (no GPU):  python -m v5.runtime.algo_lm_author --selftest
  molab (real LM):    python -m v5.runtime.algo_lm_author --run --model Qwen/Qwen2.5-3B-Instruct \
                          --limit 60 --graph graphs/algo_mbpp_author.json
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

from graph_core import MemoryGraph
from v5.runtime.algo_graph_edits import edge_candidate, grow, node_candidate
from v5.runtime.algo_graph_mg import MGRetriever, _edits_from_solve, _fn_name, seed_graph, solve_mg


def _author_prompt_purpose(task, advertised, purposes: dict) -> str:
    """The FIXED advertisement (--ad-style purpose): each ad = signature + its node's PURPOSE line
    (the bare-signature prompt showed `get_equal(Input)`-style ads — uninterpretable — plus a HARD
    'do NOT re-implement' directive: an instruction to prefer functions whose behavior the model
    cannot know; suspected driver of the accuracy decline as the graph grows). Soft directive."""
    from v5.runtime.algo_graph_run import _sig
    parts = [task.text]
    if advertised:
        parts.append("\nFunctions already in your library (you MAY call one IF it clearly fits; "
                     "otherwise ignore them and write your own code):")
        for name, code in advertised:
            purpose = (purposes.get(name) or "").strip()
            parts.append(f"  {_sig(code) or name}" + (f"  # {purpose}" if purpose else ""))
    parts.append(f"Write `{task.name}(...)` in ONE Python code block.")
    parts.append("Then, IF you wrote a genuinely reusable helper worth keeping for FUTURE tasks, "
                 "curate your library — one line each, AFTER the code block:\n"
                 "  STORE <helper_name>: <one-line purpose of what it computes>\n"
                 "Only store genuinely reusable helpers; store nothing if none apply (your choice).")
    return "\n\n".join(parts)


def _solve_author(retr: MGRetriever, gen_fn, task, k: int, samples: int, ad_style: str):
    """solve_mg with a pluggable advertisement style (solve_mg hardcodes the bare-sig prompt).
    ad_style: off (k=0) | sig (status quo, delegates to solve_mg) | purpose (fixed ads)."""
    if ad_style == "off":
        return solve_mg(retr, gen_fn, task, k=0, samples=samples)
    if ad_style == "sig":
        return solve_mg(retr, gen_fn, task, k=k, samples=samples)
    # purpose arm — mirror solve_mg's verify/reward flow with the fixed prompt
    import re as _re
    from v5.runtime.algo_graph_run import _task_verify, _top_defs, _code_fingerprint
    from v5.runtime.derive_reward import _def_names, code_reward, grounded_code
    from v5.runtime.tool_memory import _extract_code
    advertised = retr.retrieve(task.text, k=k)
    purposes = {}
    for nid in retr.ids:
        node = retr.graph.nodes[nid]
        fn = _fn_name(node.metadata.get("code", "")) or nid
        purposes[fn] = (node.text or "").splitlines()[0][:120]
    adv_names = [n for n, _ in advertised]
    prompt = _author_prompt_purpose(task, advertised, purposes)
    best = ("", [], False, "")
    for gen in gen_fn([prompt] * samples):
        code = _extract_code(gen)
        defined = _def_names(code)
        called = [n for n in adv_names if n not in defined and _re.search(rf"\b{_re.escape(n)}\s*\(", code)]
        deps = "\n\n".join(c for n, c in advertised if n in called)
        if _task_verify(task, code, deps):
            best = (code, called, True, gen); break
        if not best[0]:
            best = (code, called, False, gen)
    code, reused, verified, raw = best
    _, used = grounded_code(code, adv_names)
    new_helpers = [n for n in _top_defs(code) if n != task.name and n not in adv_names]
    R, _ = code_reward(verified, composed_used=used,
                       authored_new_verified=len(new_helpers) if verified else 0)
    return dict(name=task.name, verified=verified, reward=round(R, 3), reused=used, code=code, raw=raw)


def _called_atoms(code: str, atom_names) -> list:
    """Atoms the solution CALLS (not re-defines) — the depend edges + the reuse signal.
    (?<![\\w.]) guards method calls: `lst.count(x)` must NOT count as calling a banked atom named
    `count` (\\b matches after '.', which produced false-positive reuse events)."""
    defined = set(re.findall(r"def\s+([A-Za-z_]\w*)\s*\(", code or ""))
    return [a for a in atom_names if a not in defined
            and re.search(rf"(?<![\w.]){re.escape(a)}\s*\(", code or "")]


def _bank_solution(graph_path: str, retr: MGRetriever, task, res_solve: dict, called: list,
                   session: str, concept: str = "concept_algorithms"):
    """A VERIFIED solve -> (a) the solution as an implementation node (origin=lm_author, task text =
    retrieval key) + depend edges to called atoms; (b) the model's STORE-action HELPERS as their OWN
    atoms (origin=lm_author_helper) — the reuse-granular units: nobody calls another task's entry
    point, but everybody calls is_prime. Health-gated. Returns (banked_solution, helper_names)."""
    code = res_solve["code"]
    cands, helper_names = [], []
    nid = f"impl_{task.name}"
    if nid not in retr.graph.nodes:
        cands.append(node_candidate(nid, code, task.text.splitlines()[0][:200], session,
                                    metadata={"kind": "authored", "origin": "lm_author"}))
        cands.append(edge_candidate(nid, concept, "part_of", session))
        for a in called:
            cands.append(edge_candidate(nid, f"impl_{a}", "depend", session))
    stores, edges = _edits_from_solve(res_solve, task, concept)     # model-chosen helpers (Fix D)
    for hid, src, purpose in stores:
        if hid != nid and hid not in retr.graph.nodes:
            cands.append(node_candidate(hid, src, purpose, session,
                                        metadata={"kind": "authored", "origin": "lm_author_helper"}))
            cands.append(edge_candidate(hid, concept, "part_of", session))
            helper_names.append(hid[len("impl_"):])
    if not cands:
        return False, []
    newp = graph_path + ".grown"
    r = grow(graph_path, newp, cands)
    if r.get("persisted"):
        Path(newp).replace(graph_path)
        return True, helper_names
    return False, []


def _failure_class(t, code: str, deps: str) -> str:
    """Taxonomy for a miss: no_code | syntax | assert_fail (fails the original asserts) |
    plus_only_fail (passes the original asserts, fails the DENSE EvalPlus script = the gate catching a
    benchmark-overfit — plain-MBPP leaderboards would have counted this one as SOLVED)."""
    from v5.runtime.algo_graph_run import verify_asserts
    if not code or "def " not in code:
        return "no_code"
    try:
        compile(code, "<gen>", "exec")
    except SyntaxError:
        return "syntax"
    originals = [x for x in t.tests if x.lstrip().startswith("assert")]
    plus = [x for x in t.tests if not x.lstrip().startswith("assert")]
    full = (deps + "\n" + code) if deps else code
    if originals and verify_asserts(full, originals, getattr(t, "setup", "")):
        return "plus_only_fail" if plus else "assert_fail"
    return "assert_fail"


def run_author_loop(graph_path: str, embed_fn, gen_fn, tasks, k_retrieve: int = 6, samples: int = 4,
                    reindex_every: int = 5, log: bool = True,
                    failure_log: str = "artifacts/grr14_failures.jsonl", ad_style: str = "sig",
                    shuffle_seed: int | None = None):
    """The loop over real tasks. Returns the report dict. The graph GROWS as it runs — atoms authored
    for early tasks are advertised (and reused) by later ones. k_retrieve=0 disables advertisement
    entirely (the context-pollution ablation: does accuracy stop declining when the prompt stops
    growing with the graph?). Failures are classified + logged; the report prints the solve rate per
    20-task bucket so the over-time curve is visible directly."""
    if not Path(graph_path).exists():
        seed_graph(graph_path, ("concept_algorithms",))
    if shuffle_seed is not None:                        # the corpus-ORDERING control
        import random
        tasks = list(tasks)
        random.Random(shuffle_seed).shuffle(tasks)
    retr = MGRetriever(MemoryGraph.load_json(graph_path), embed_fn)
    solved = banked = 0
    authored_this_run: set = set()
    reuse_events = []                                   # (task, called earlier-authored atoms)
    since_reindex = 0
    buckets: list = []                                  # per-20-task solve counts (the decline curve)
    fail_counts: dict = {}
    Path(failure_log).parent.mkdir(parents=True, exist_ok=True)
    flog = open(failure_log, "w", encoding="utf-8")
    for i, t in enumerate(tasks):
        if i % 20 == 0:
            buckets.append(0)
        res = _solve_author(retr, gen_fn, t, k=k_retrieve, samples=samples, ad_style=ad_style)
        if not res["verified"]:
            cls = _failure_class(t, res.get("code", ""), "")
            fail_counts[cls] = fail_counts.get(cls, 0) + 1
            flog.write(json.dumps({"i": i, "task": t.name, "class": cls,
                                   "graph_nodes": len(retr.graph.nodes),
                                   "code": (res.get("code") or "")[:1500]}) + "\n")
        if res["verified"]:
            solved += 1
            buckets[-1] += 1
            atom_names = [_fn_name(retr.graph.nodes[nid].metadata.get("code", "")) or nid
                          for nid in retr.ids]
            called = _called_atoms(res["code"], atom_names)
            reused_authored = [a for a in called if a in authored_this_run]
            if reused_authored:
                reuse_events.append((t.name, reused_authored))
            ok, helper_names = _bank_solution(graph_path, retr, t, res, called, f"grr14_{i}")
            if ok:
                banked += 1
                authored_this_run.add(t.name)
                authored_this_run.update(helper_names)  # helpers = the reuse-granular atoms
                since_reindex += 1
                if since_reindex >= reindex_every:      # new atoms become retrievable for later tasks
                    retr = MGRetriever(MemoryGraph.load_json(graph_path), embed_fn)
                    since_reindex = 0
        if log and (i + 1) % 10 == 0:
            print(f"  [{i+1}/{len(tasks)}] solved {solved} | banked {banked} | "
                  f"cross-task reuse events {len(reuse_events)}", flush=True)
    if since_reindex:
        retr = MGRetriever(MemoryGraph.load_json(graph_path), embed_fn)
    flog.close()
    bucket_sizes = [min(20, len(tasks) - 20 * b) for b in range(len(buckets))]
    report = dict(tasks=len(tasks), solved=solved, solve_rate=round(solved / max(1, len(tasks)), 3),
                  banked=banked, reuse_events=len(reuse_events),
                  reuse_detail=reuse_events[:10], graph_nodes=len(retr.graph.nodes),
                  curve=[round(s / max(1, n), 2) for s, n in zip(buckets, bucket_sizes)],
                  failures=fail_counts, k_retrieve=k_retrieve, ad_style=ad_style,
                  shuffle_seed=shuffle_seed)
    if log:
        print(f"\n  === GRR-14 report (ad_style={ad_style}, k={k_retrieve}, "
              f"shuffle={shuffle_seed}) ===", flush=True)
        print(f"  solved {solved}/{len(tasks)} ({report['solve_rate']:.0%}) — the no-authoring ladder "
              f"baseline was 2% (1/40)", flush=True)
        print(f"  SOLVE CURVE per 20 tasks (the over-time question): {report['curve']}", flush=True)
        print(f"  FAILURE TAXONOMY: {fail_counts}  (plus_only_fail = passed the original asserts, "
              f"killed by the DENSE gate — a plain-MBPP leaderboard counts those as solved)", flush=True)
        print(f"  atoms banked: {banked} (origin=lm_author, health-gated, depend edges to called atoms)",
              flush=True)
        print(f"  CROSS-TASK REUSE: {len(reuse_events)} events "
              f"{('e.g. ' + '; '.join(f'{n} called {c}' for n, c in reuse_events[:3])) if reuse_events else ''}",
              flush=True)
        print(f"  graph: {report['graph_nodes']} nodes | failure detail -> {failure_log}", flush=True)
    return report


# ═══════════════════════════════════════════════════════════════════════════════
# SELFTEST (no GPU) — a "perfect author" stub proves the loop mechanics: verified solutions bank with
# provenance + depend edges; a later task REUSES an earlier authored atom (the compounding event);
# an unsolvable task banks NOTHING (the gate holds).
# ═══════════════════════════════════════════════════════════════════════════════

def _selftest() -> bool:
    import tempfile
    import numpy as np
    from v5.runtime.algo_graph_run import MBPPTask
    print("algo_lm_author --selftest: author -> dense gate -> bank(origin) -> cross-task reuse\n")

    # keyword-keyed stub embed (a fake random embedder retrieves nothing — t2 must FIND t1's atom)
    rng = np.random.default_rng(0)
    base = rng.standard_normal(64).astype("float32")

    def embed(d):
        out = {}
        for k, t in d.items():
            v = base if "decimal digit" in t.lower() else rng.standard_normal(64).astype("float32")
            out[k] = (v + 0.05 * rng.standard_normal(64)).astype("float32")
        return out

    t1 = MBPPTask("digit_total", "Compute the sum of decimal digits of n.\nWrite `digit_total(n)`.",
                  ["assert digit_total(123) == 6", "assert digit_total(9) == 9",
                   "assert digit_total(4051) == 10"])
    t2 = MBPPTask("digit_total_list",
                  "Sum of decimal digit sums across a list. ctx: sum of decimal digits of n",
                  ["assert digit_total_list([12, 34]) == 10", "assert digit_total_list([5]) == 5"])
    t3 = MBPPTask("impossible", "Return the 10th busy beaver number.\nWrite `impossible()`.",
                  ["assert impossible() == -1"])

    t4 = MBPPTask("count_odds", "Count the odd numbers in a list.\nWrite `count_odds(xs)`.",
                  ["assert count_odds([1, 2, 3]) == 2", "assert count_odds([2, 4]) == 0"])

    CODE1 = "def digit_total(n):\n    return sum(int(c) for c in str(abs(n)))"
    CODE2 = ("def digit_total_list(xs):\n    return sum(digit_total(x) for x in xs)")  # REUSES t1's atom
    CODE4 = ("def is_odd_h(n):\n    return n % 2 == 1\n\n"
             "def count_odds(xs):\n    return sum(1 for x in xs if is_odd_h(x))")

    def stub_gen(prompts):
        outs = []
        for p in prompts:
            if "digit_total_list" in p:
                outs.append(f"```python\n{CODE2}\n```")     # calls the ADVERTISED authored atom
            elif "digit_total" in p:
                outs.append(f"```python\n{CODE1}\n```")
            elif "count_odds" in p:
                # the author CHOOSES to store a helper (Fix D STORE action) -> reuse-granular atom
                outs.append(f"```python\n{CODE4}\n```\nSTORE is_odd_h: odd-number predicate helper")
            else:
                outs.append("```python\ndef impossible():\n    return 42\n```")   # fails its assert
        return outs

    with tempfile.TemporaryDirectory() as td:
        gp = str(Path(td) / "g.json")
        report = run_author_loop(gp, embed, stub_gen, [t1, t4, t2, t3], samples=1, reindex_every=1,
                                 log=False)
        g = MemoryGraph.load_json(gp)

        # [1] verified solutions banked with provenance; the failed one is NOT; the model's STORE
        #     helper became its OWN atom (the reuse-granular unit)
        assert report["solved"] == 3 and report["banked"] == 3, report
        assert "impl_digit_total" in g.nodes and "impl_impossible" not in g.nodes
        assert g.nodes["impl_digit_total"].metadata.get("origin") == "lm_author"
        assert g.nodes["impl_is_odd_h"].metadata.get("origin") == "lm_author_helper"
        print(f"  [1] 3/4 solved -> banked with origin=lm_author; STORE helper is_odd_h banked as its "
              f"OWN atom (origin=lm_author_helper); the gate-failing task banked NOTHING -> PASS")

        # [2] cross-task REUSE: t2's solution CALLS t1's authored atom; depend edge landed
        assert report["reuse_events"] == 1 and report["reuse_detail"][0][1] == ["digit_total"], report
        assert g.edge_between("impl_digit_total_list", "impl_digit_total") is not None
        print(f"  [2] cross-task reuse: digit_total_list CALLS the authored digit_total "
              f"(+ depend edge in the graph) — compounding on authored knowledge -> PASS")

        # [3] the banked code is the runnable knowledge: resolve deps through the graph and execute
        retr = MGRetriever(g, embed)
        deps = retr.resolve_deps(["digit_total_list"])
        ns: dict = {}
        exec(deps, ns)
        assert ns["digit_total_list"]([12, 34]) == 10
        print(f"  [3] graph walk resolves the authored dependency chain; code executes -> PASS")

    print("\n  ALGO_LM_AUTHOR SELFTEST -> PASS  (the LM invents primitives; the gate decides; the "
          "graph compounds)")
    return True


def main():
    ap = argparse.ArgumentParser(description="GRR-14: LM authors new atoms on MBPP+ (gated, banked, reused).")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--run", action="store_true", help="real LM over the prepped MBPP+ corpus (molab)")
    ap.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    ap.add_argument("--corpus", default="artifacts/mbpp_plus_prepped.jsonl")
    ap.add_argument("--graph", default="graphs/algo_mbpp_author.json")
    ap.add_argument("--limit", type=int, default=60)
    ap.add_argument("--samples", type=int, default=4)
    ap.add_argument("--pipeline-only", action="store_true", help="restrict to pipeline-shaped tasks")
    ap.add_argument("--ad-style", default="off", choices=["off", "sig", "purpose"],
                    help="advertisement arm: off=DEFAULT (measured: sig ads cost ~16pp by task 90 and "
                         "cause the over-time decline) | sig=bare-sig status quo | purpose=repaired ads")
    ap.add_argument("--shuffle", type=int, default=-1, help="shuffle tasks with this seed (-1 = corpus order)")
    a = ap.parse_args()
    if a.selftest:
        sys.exit(0 if _selftest() else 1)
    if a.run:
        from v5.memory.store import make_mpnet_embedder
        from v5.runtime.algo_lm_proposer import make_hf_gen
        from v5.runtime.algo_mbpp_prep import load_prepped
        tasks = load_prepped(a.corpus, limit=a.limit, pipeline_only=a.pipeline_only)
        print(f"GRR-14 author loop (real LM {a.model}): {len(tasks)} MBPP+ tasks | graph {a.graph} | "
              f"ad_style={a.ad_style} shuffle={a.shuffle}", flush=True)
        run_author_loop(a.graph, make_mpnet_embedder(), make_hf_gen(a.model, max_new_tokens=400),
                        tasks, samples=a.samples, ad_style=a.ad_style,
                        shuffle_seed=None if a.shuffle < 0 else a.shuffle)
        return
    ap.print_help()


if __name__ == "__main__":
    main()
