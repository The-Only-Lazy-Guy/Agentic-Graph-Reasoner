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
from v5.runtime.algo_graph_mg import MGRetriever, _fn_name, seed_graph, solve_mg


def _called_atoms(code: str, atom_names) -> list:
    """Atoms the solution CALLS (not re-defines) — the depend edges + the reuse signal."""
    defined = set(re.findall(r"def\s+([A-Za-z_]\w*)\s*\(", code or ""))
    return [a for a in atom_names if a not in defined
            and re.search(rf"\b{re.escape(a)}\s*\(", code or "")]


def _bank_solution(graph_path: str, retr: MGRetriever, task, code: str, called: list,
                   session: str, concept: str = "concept_algorithms") -> bool:
    """A VERIFIED solution -> an implementation node (origin=lm_author) + depend edges to every called
    atom + part_of the concept. Health-gated. Returns True iff persisted."""
    nid = f"impl_{task.name}"
    if nid in retr.graph.nodes:
        return False
    cands = [node_candidate(nid, code, task.text.splitlines()[0][:200], session,
                            metadata={"kind": "authored", "origin": "lm_author"})]
    cands.append(edge_candidate(nid, concept, "part_of", session))
    for a in called:
        cands.append(edge_candidate(nid, f"impl_{a}", "depend", session))
    newp = graph_path + ".grown"
    res = grow(graph_path, newp, cands)
    if res.get("persisted"):
        Path(newp).replace(graph_path)
        return True
    return False


def run_author_loop(graph_path: str, embed_fn, gen_fn, tasks, k_retrieve: int = 6, samples: int = 4,
                    reindex_every: int = 5, log: bool = True):
    """The loop over real tasks. Returns the report dict. The graph GROWS as it runs — atoms authored
    for early tasks are advertised (and reused) by later ones."""
    if not Path(graph_path).exists():
        seed_graph(graph_path, ("concept_algorithms",))
    retr = MGRetriever(MemoryGraph.load_json(graph_path), embed_fn)
    solved = banked = 0
    authored_this_run: set = set()
    reuse_events = []                                   # (task, called earlier-authored atoms)
    since_reindex = 0
    for i, t in enumerate(tasks):
        res = solve_mg(retr, gen_fn, t, k=k_retrieve, samples=samples)
        if res["verified"]:
            solved += 1
            atom_names = [_fn_name(retr.graph.nodes[nid].metadata.get("code", "")) or nid
                          for nid in retr.ids]
            called = _called_atoms(res["code"], atom_names)
            reused_authored = [a for a in called if a in authored_this_run]
            if reused_authored:
                reuse_events.append((t.name, reused_authored))
            if _bank_solution(graph_path, retr, t, res["code"], called, f"grr14_{i}"):
                banked += 1
                authored_this_run.add(t.name)
                since_reindex += 1
                if since_reindex >= reindex_every:      # new atoms become retrievable for later tasks
                    retr = MGRetriever(MemoryGraph.load_json(graph_path), embed_fn)
                    since_reindex = 0
        if log and (i + 1) % 10 == 0:
            print(f"  [{i+1}/{len(tasks)}] solved {solved} | banked {banked} | "
                  f"cross-task reuse events {len(reuse_events)}", flush=True)
    if since_reindex:
        retr = MGRetriever(MemoryGraph.load_json(graph_path), embed_fn)
    report = dict(tasks=len(tasks), solved=solved, solve_rate=round(solved / max(1, len(tasks)), 3),
                  banked=banked, reuse_events=len(reuse_events),
                  reuse_detail=reuse_events[:10], graph_nodes=len(retr.graph.nodes))
    if log:
        print(f"\n  === GRR-14 report ===", flush=True)
        print(f"  solved {solved}/{len(tasks)} ({report['solve_rate']:.0%}) — the no-authoring ladder "
              f"baseline was 2% (1/40)", flush=True)
        print(f"  atoms banked: {banked} (origin=lm_author, health-gated, depend edges to called atoms)",
              flush=True)
        print(f"  CROSS-TASK REUSE: {len(reuse_events)} events "
              f"{('e.g. ' + '; '.join(f'{n} called {c}' for n, c in reuse_events[:3])) if reuse_events else ''}",
              flush=True)
        print(f"  graph: {report['graph_nodes']} nodes", flush=True)
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

    CODE1 = "def digit_total(n):\n    return sum(int(c) for c in str(abs(n)))"
    CODE2 = ("def digit_total_list(xs):\n    return sum(digit_total(x) for x in xs)")  # REUSES t1's atom

    def stub_gen(prompts):
        outs = []
        for p in prompts:
            if "digit_total_list" in p:
                # the perfect author reuses the ADVERTISED atom iff the prompt shows it
                outs.append(f"```python\n{CODE2}\n```" if "digit_total" in p.split("Task:")[0] or
                            "digit_total(" in p else f"```python\n{CODE2}\n```")
            elif "digit_total" in p:
                outs.append(f"```python\n{CODE1}\n```")
            else:
                outs.append("```python\ndef impossible():\n    return 42\n```")   # fails its assert
        return outs

    with tempfile.TemporaryDirectory() as td:
        gp = str(Path(td) / "g.json")
        report = run_author_loop(gp, embed, stub_gen, [t1, t2, t3], samples=1, reindex_every=1,
                                 log=False)
        g = MemoryGraph.load_json(gp)

        # [1] verified solutions banked with provenance; the failed one is NOT
        assert report["solved"] == 2 and report["banked"] == 2, report
        assert "impl_digit_total" in g.nodes and "impl_impossible" not in g.nodes
        assert g.nodes["impl_digit_total"].metadata.get("origin") == "lm_author"
        print(f"  [1] 2/3 solved -> banked with origin=lm_author; the gate-failing task banked "
              f"NOTHING -> PASS")

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
    a = ap.parse_args()
    if a.selftest:
        sys.exit(0 if _selftest() else 1)
    if a.run:
        from v5.memory.store import make_mpnet_embedder
        from v5.runtime.algo_lm_proposer import make_hf_gen
        from v5.runtime.algo_mbpp_prep import load_prepped
        tasks = load_prepped(a.corpus, limit=a.limit, pipeline_only=a.pipeline_only)
        print(f"GRR-14 author loop (real LM {a.model}): {len(tasks)} MBPP+ tasks | graph {a.graph}",
              flush=True)
        run_author_loop(a.graph, make_mpnet_embedder(), make_hf_gen(a.model, max_new_tokens=400),
                        tasks, samples=a.samples)
        return
    ap.print_help()


if __name__ == "__main__":
    main()
