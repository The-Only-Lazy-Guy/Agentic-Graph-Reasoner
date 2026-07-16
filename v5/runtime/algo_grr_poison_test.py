"""algo_grr_poison_test — the two-arm poison test (GRR-Tool build B2).

Question (READ_THIS "POISON DIAGNOSIS"): does the frozen-compiler + membrane design avoid the
graph-poisons-the-LM decline that both the old STaR loop and a naive tool-TRM suffer?

Two arms, SAME designed seed curriculum, SAME starting graph (grr_seed_clean), N rounds:

  OLD arm (poison-prone):  raw graph advertised into the prompt (all banked atoms) + whole-solution
                           banking (flat, no depend) + LoRA SFT on own verified traces.
                           -> prompt FLOODS as the graph grows (context poison); LoRA narrows the
                           LM on its own outputs (weight poison); banked whole-solutions don't
                           compose (reuse structurally 0).
  NEW arm (frozen+membrane): frozen LM (never trained) + curated spec (only selected atoms) +
                           helper-granular derive-banking (depend edges). -> prompt BOUNDED; graph
                           growth REDUCES per-task LM work (compounding); banked atoms reuse.

The REAL accuracy-decline signal needs the frozen/LoRA 3B on molab. The STRUCTURAL mechanism —
NEW bounds the prompt and compounds via reuse; OLD floods and cannot reuse — is provable no-GPU
here with the stub compiler. That structural difference IS the poison mechanism, measured directly.

    selftest (no GPU):  python -m v5.runtime.algo_grr_poison_test --selftest
    molab (real 3B):    python -m v5.runtime.algo_grr_poison_test --run --lm Qwen/Qwen2.5-3B-Instruct
"""
from __future__ import annotations

import argparse
import ast
import re
import sys
from pathlib import Path
from typing import Callable

_ROOT = str(Path(__file__).resolve().parents[2])
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from graph_core import MemoryGraph, Node, Edge  # type: ignore  # noqa: E402
from v5.runtime.algo_grr_membrane import (  # noqa: E402
    MembraneSolver, TokenRetriever, make_stub_compiler, make_lm_compiler,
    verify_code, reuse_set, render_compile_prompt,
)


# ═══════════════════════════════════════════════════════════════════════════════
# The designed seed curriculum — progressive rounds over the 21-atom seed domain
#   R1 recall  : single seed atom (wrap)
#   R2 compose : two seed atoms
#   R3 derive  : NO seed atom covers the core -> the LM authors a NEW atom (banked)
#   R4 reuse   : reuses the atoms DERIVED in R3 (the compounding test)
# Each task carries a stub `recipe` (stands in for the frozen LM's output on molab).
# ═══════════════════════════════════════════════════════════════════════════════

def curriculum() -> list[list[dict]]:
    R1 = [
        dict(text="check whether a number is prime", entry="t_isprime",
             tests=[((7,), True), ((8,), False), ((2,), True), ((1,), False)],
             recipe="def t_isprime(n):\n    return is_prime(n)\n"),
        dict(text="greatest common divisor of two integers", entry="t_gcd",
             tests=[((12, 8), 4), ((17, 5), 1)],
             recipe="def t_gcd(a, b):\n    return gcd(a, b)\n"),
        dict(text="reverse the digits of an integer", entry="t_rev",
             tests=[((123,), 321), ((100,), 1)],
             recipe="def t_rev(n):\n    return reverse_digits(n)\n"),
    ]
    R2 = [
        dict(text="count how many divisors of n are prime numbers", entry="t_primediv",
             tests=[((12,), 2), ((30,), 3), ((7,), 1)],
             recipe="def t_primediv(n):\n    return sum(1 for d in divisors(n) if is_prime(d))\n"),
        dict(text="least common multiple of two integers", entry="t_lcm",
             tests=[((4, 6), 12), ((3, 5), 15)],
             recipe="def t_lcm(a, b):\n    return lcm(a, b)\n"),
        dict(text="check if two strings are anagrams", entry="t_anag",
             tests=[(("listen", "silent"), True), (("abc", "abd"), False)],
             recipe="def t_anag(a, b):\n    return is_anagram(a, b)\n"),
    ]
    # R3: the core needs a NEW atom no seed atom provides. The recipe DEFINES that helper +
    # the entry that calls it -> the NEW arm banks the helper (helper-granular).
    R3 = [
        dict(text="sum of the squares of the integers from 1 to n", entry="t_sumsq",
             tests=[((3,), 14), ((1,), 1), ((4,), 30)],
             recipe=("def sum_of_squares(n):\n"
                     "    return sum(i * i for i in range(1, n + 1))\n\n"
                     "def t_sumsq(n):\n    return sum_of_squares(n)\n"),
             derives=[("sum_of_squares", "sum of the squares of 1..n")]),
        dict(text="the n-th fibonacci number (0-indexed, fib(0)=0, fib(1)=1)", entry="t_fib",
             tests=[((0,), 0), ((1,), 1), ((7,), 13), ((10,), 55)],
             recipe=("def nth_fibonacci(n):\n"
                     "    a, b = 0, 1\n"
                     "    for _ in range(n):\n"
                     "        a, b = b, a + b\n"
                     "    return a\n\n"
                     "def t_fib(n):\n    return nth_fibonacci(n)\n"),
             derives=[("nth_fibonacci", "the n-th fibonacci number")]),
    ]
    # R4: REUSES the R3-derived atoms (sum_of_squares, nth_fibonacci) — the compounding payoff.
    R4 = [
        dict(text="sum of the squares of 1..n, then reversed digits", entry="t_sumsq_rev",
             tests=[((3,), 41), ((4,), 3)],   # sumsq(3)=14->41 ; sumsq(4)=30->3
             recipe=("def t_sumsq_rev(n):\n"
                     "    return reverse_digits(sum_of_squares(n))\n")),
        dict(text="the n-th fibonacci number, is it prime", entry="t_fib_prime",
             tests=[((4,), True), ((6,), False), ((5,), True)],  # fib4=3(T),fib6=8(F),fib5=5(T)
             recipe="def t_fib_prime(n):\n    return is_prime(nth_fibonacci(n))\n"),
    ]
    return [R1, R2, R3, R4]


# ═══════════════════════════════════════════════════════════════════════════════
# Seed graph + banking
# ═══════════════════════════════════════════════════════════════════════════════

def load_seed() -> MemoryGraph:
    from v5.runtime.algo_grr_seed import build_graph
    g = build_graph()
    nodes = {n["id"]: Node.from_dict(n) for n in g["nodes"]}
    edges = [Edge.from_dict(e) for e in g["edges"]]
    return MemoryGraph(nodes, edges, metadata=dict(g["metadata"]))


def _classify(text: str) -> str:
    t = text.lower()
    if re.search(r"prime|divisor|gcd|lcm|fib|digit|number|factor|square", t):
        return "concept_number_theory"
    if re.search(r"string|char|anagram|palindrome", t):
        return "concept_strings"
    if re.search(r"list|array|subarray|element", t):
        return "concept_lists"
    if re.search(r"search|sort|merge|binary", t):
        return "concept_search"
    return "concept_number_theory"


def _atom_entries(graph: MemoryGraph) -> dict[str, str]:
    return {graph.nodes[nid].metadata.get("entry", nid): nid
            for nid in graph.nodes if graph.nodes[nid].node_type == "implementation"}


def _extract_defs(code: str) -> dict[str, str]:
    """name -> source block for each top-level FunctionDef."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return {}
    lines = code.splitlines()
    out: dict[str, str] = {}
    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            end = getattr(node, "end_lineno", node.lineno)
            out[node.name] = "\n".join(lines[node.lineno - 1:end])
    return out


def bank_helper_granular(graph: MemoryGraph, code: str, entry: str) -> list[str]:
    """NEW-arm SLEEP: bank each NEW helper (reachable from entry, not the entry, not an existing
    atom) as its OWN verified atom with part_of + depend edges. Returns banked node ids."""
    existing = _atom_entries(graph)          # entry-name -> node id
    defs = _extract_defs(code)
    # call graph (for depend edges + reachability)
    reach = reuse_set(code, entry, set(defs.keys()))   # atoms/helpers reachable from entry
    banked: list[str] = []
    for name in reach:
        if name == entry or name in existing or name not in defs:
            continue
        nid = f"impl_{name}"
        if nid in graph.nodes:
            continue
        concept = _classify(name + " " + defs[name])
        graph.nodes[nid] = Node(
            id=nid, text=f"{name.replace('_', ' ')}", node_type="implementation",
            confidence=0.9, importance=0.5,
            metadata={"code": defs[name], "entry": name, "kind": "atom", "origin": "derived",
                      "concept": concept},
        )
        graph.edges.append(Edge(src=nid, dst=concept, relation="part_of", strength=1.0,
                                metadata={"origin": "derived"}))
        # depend edges: helpers/atoms THIS helper calls
        called = {n.func.id for n in ast.walk(ast.parse(defs[name]))
                  if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}
        for c in called:
            dep_nid = existing.get(c) or (f"impl_{c}" if f"impl_{c}" in graph.nodes else None)
            if dep_nid and dep_nid != nid:
                graph.edges.append(Edge(src=nid, dst=dep_nid, relation="depend", strength=1.0,
                                        metadata={"origin": "derived"}))
        existing[name] = nid
        banked.append(nid)
    graph._rebuild_index()
    return banked


# ═══════════════════════════════════════════════════════════════════════════════
# NEW arm — frozen compiler + membrane + helper-granular derive-banking
# ═══════════════════════════════════════════════════════════════════════════════

def make_curriculum_stub(tasks_flat: list[dict]) -> Callable[[dict], str]:
    """Stub 'frozen compiler': looks up each task's recipe by entry, prepends the curated closure."""
    recipes = {t["entry"]: t["recipe"] for t in tasks_flat}
    base = make_stub_compiler(recipes)

    def compile_fn(spec: dict) -> str:
        return base(spec)
    return compile_fn


def run_new_arm(rounds: list[list[dict]], compile_fn: Callable[[dict], str]) -> list[dict]:
    graph = load_seed()
    flat = [t for r in rounds for t in r]
    metrics = []
    for ri, tasks in enumerate(rounds):
        solved = reuse_events = lm_calls = banked = prompt_atoms = 0
        for t in tasks:
            solver = MembraneSolver(graph, compile_fn)     # rebuilt on the CURRENT graph
            n0 = len(solver.compile_inputs)
            r = solver.solve(t)
            lm_calls += len(solver.compile_inputs) - n0
            prompt_atoms += max((len(s["atoms"]) for s in solver.compile_inputs), default=0)
            if r["solved"]:
                solved += 1
                reuse_events += len(r["selected"])
                banked += len(bank_helper_granular(graph, r["code"], t["entry"]))
        n = len(tasks)
        metrics.append(dict(round=ri + 1, solved=solved, n=n, reuse=reuse_events, banked=banked,
                            avg_lm_calls=lm_calls / n, avg_prompt_atoms=prompt_atoms / n,
                            graph_nodes=len(graph.nodes)))
    return metrics


# ═══════════════════════════════════════════════════════════════════════════════
# OLD arm — raw graph advertised into the prompt + whole-solution banking (poison-prone)
# ═══════════════════════════════════════════════════════════════════════════════

def make_raw_stub(tasks_flat: list[dict]) -> Callable[[str, MemoryGraph, dict], str]:
    """Stub raw-arm 'compiler'. Models the real LM seeing the FLOOD: every advertised atom's code
    is dumped into context, so the solution can inline/call any of it. Faithfully reproduces the
    OLD structural properties (flood grows with the graph; whole-solution banking has no depend
    edges). The accuracy hit from the flood + LoRA is the molab-only signal."""
    recipes = {t["entry"]: t["recipe"] for t in tasks_flat}

    def compile_fn(prompt: str, graph: MemoryGraph, task: dict) -> str:
        # dump ALL advertised atom code (the flood) so recipe calls resolve, then the recipe
        flood = "\n\n".join(graph.nodes[nid].metadata.get("code", "")
                            for nid in graph.nodes
                            if graph.nodes[nid].node_type == "implementation")
        return flood + "\n\n" + recipes.get(task["entry"], "")
    return compile_fn


def _raw_prompt(graph: MemoryGraph, task: dict) -> tuple[str, int]:
    """OLD arm prompt: ADVERTISE every banked atom (the measured net-negative channel) + task.
    Returns (prompt, n_advertised). n_advertised GROWS with the graph -> context flood."""
    impls = [nid for nid in graph.nodes if graph.nodes[nid].node_type == "implementation"]
    ads = "\n".join(f"- {graph.nodes[nid].metadata.get('entry', nid)}: {graph.nodes[nid].text}"
                    for nid in impls)
    prompt = f"Available functions:\n{ads}\n\nTask: {task['text']}\nWrite {task['entry']}."
    return prompt, len(impls)


def bank_whole_solution(graph: MemoryGraph, code: str, task: dict) -> None:
    """OLD-arm banking: the WHOLE solution under the entry name, flat (part_of only, NO depend).
    This is the polluted topology that makes reuse structurally 0."""
    nid = f"impl_{task['entry']}"
    if nid in graph.nodes:
        return
    concept = _classify(task["text"])
    graph.nodes[nid] = Node(id=nid, text=task["text"], node_type="implementation",
                            confidence=0.9, importance=0.5,
                            metadata={"code": code, "entry": task["entry"], "kind": "authored",
                                      "origin": "old_whole"})
    graph.edges.append(Edge(src=nid, dst=concept, relation="part_of", strength=1.0))
    graph._rebuild_index()


def run_old_arm(rounds: list[list[dict]],
                compile_fn: Callable[[str, MemoryGraph, dict], str],
                train_fn: Callable[[list], None] | None = None) -> list[dict]:
    graph = load_seed()
    verified_pool: list[dict] = []
    metrics = []
    for ri, tasks in enumerate(rounds):
        solved = reuse_events = prompt_atoms = 0
        for t in tasks:
            prompt, n_ads = _raw_prompt(graph, t)
            prompt_atoms += n_ads
            code = compile_fn(prompt, graph, t)
            frac, _ = verify_code(code, t["entry"], t["tests"])
            if frac >= 1.0:
                solved += 1
                # reuse under whole-solution banking: does the code CALL a previously-banked
                # whole-solution atom? (structurally ~never — nobody calls another task's entry)
                banked_entries = {graph.nodes[nid].metadata.get("entry", nid)
                                  for nid in graph.nodes
                                  if graph.nodes[nid].metadata.get("origin") == "old_whole"}
                reuse_events += len(reuse_set(code, t["entry"], banked_entries))
                bank_whole_solution(graph, code, t)
                verified_pool.append({"task": t, "code": code})
        if train_fn is not None:            # molab: LoRA SFT on own verified traces (weight poison)
            train_fn(verified_pool)
        n = len(tasks)
        metrics.append(dict(round=ri + 1, solved=solved, n=n, reuse=reuse_events, banked=0,
                            avg_lm_calls=1.0, avg_prompt_atoms=prompt_atoms / n,
                            graph_nodes=len(graph.nodes)))
    return metrics


# ═══════════════════════════════════════════════════════════════════════════════
def _fmt(name: str, m: list[dict]) -> None:
    print(f"  {name}")
    print("    round  solved  reuse  banked  avg_lm_calls  prompt_atoms  graph_nodes")
    for r in m:
        print(f"      {r['round']:d}     {r['solved']}/{r['n']}     {r['reuse']:2d}    "
              f"{r['banked']:2d}       {r['avg_lm_calls']:.2f}         {r['avg_prompt_atoms']:.1f}"
              f"          {r['graph_nodes']}")


def two_arm(verbose: bool = True) -> tuple[list[dict], list[dict]]:
    rounds = curriculum()
    flat = [t for r in rounds for t in r]
    new_m = run_new_arm(rounds, make_curriculum_stub(flat))
    old_m = run_old_arm(rounds, make_raw_stub(flat), train_fn=None)
    if verbose:
        _fmt("NEW (frozen + membrane):", new_m)
        _fmt("OLD (raw graph flood + whole-solution banking):", old_m)
    return new_m, old_m


def _selftest() -> bool:
    print("algo_grr_poison_test --selftest: two-arm structural mechanism (no GPU)\n")
    new_m, old_m = two_arm(verbose=True)
    ok = True

    # [1] both arms solve the curriculum with the stubs (plumbing works)
    new_solved = all(r["solved"] == r["n"] for r in new_m)
    old_solved = all(r["solved"] == r["n"] for r in old_m)
    print(f"\n  [1] curriculum solved: NEW all-rounds={new_solved}, OLD all-rounds={old_solved} -> "
          f"{'PASS' if new_solved and old_solved else 'FAIL'}")
    ok &= new_solved and old_solved

    # [2] NEW compounds: R3 derives+banks new atoms, R4 reuses them
    r3_banked = new_m[2]["banked"] >= 1
    r4_reuse = new_m[3]["reuse"] >= 1
    grew = new_m[3]["graph_nodes"] > new_m[0]["graph_nodes"]
    print(f"  [2] NEW compounding: R3 banked={new_m[2]['banked']}, R4 reuse={new_m[3]['reuse']}, "
          f"graph {new_m[0]['graph_nodes']}->{new_m[3]['graph_nodes']} -> "
          f"{'PASS' if r3_banked and r4_reuse and grew else 'FAIL'}")
    ok &= r3_banked and r4_reuse and grew

    # [3] MEMBRANE bounds the prompt; RAW floods. NEW prompt-atoms bounded + flat vs OLD growing.
    new_bounded = max(r["avg_prompt_atoms"] for r in new_m) <= 6
    old_grows = old_m[-1]["avg_prompt_atoms"] > old_m[0]["avg_prompt_atoms"]
    print(f"  [3] prompt: NEW max_atoms={max(r['avg_prompt_atoms'] for r in new_m):.1f} (bounded), "
          f"OLD {old_m[0]['avg_prompt_atoms']:.0f}->{old_m[-1]['avg_prompt_atoms']:.0f} (flood) -> "
          f"{'PASS' if new_bounded and old_grows else 'FAIL'}")
    ok &= new_bounded and old_grows

    # [4] REUSE asymmetry: NEW reuses banked atoms; OLD whole-solution banking cannot compose (0)
    new_reuse_total = sum(r["reuse"] for r in new_m)
    old_reuse_total = sum(r["reuse"] for r in old_m)
    print(f"  [4] reuse totals: NEW={new_reuse_total} (>0), OLD={old_reuse_total} (structurally 0) "
          f"-> {'PASS' if new_reuse_total > 0 and old_reuse_total == 0 else 'FAIL'}")
    ok &= new_reuse_total > 0 and old_reuse_total == 0

    print(f"\n  ALGO_GRR_POISON_TEST SELFTEST -> {'PASS' if ok else 'FAIL'}")
    return ok


def inspect_derive(lm_name: str) -> None:
    """Raw-pipeline dump on the derive-territory rounds (R3 + R4): print whether derive fired, the
    ACTUAL code the frozen LM produced, and what helper-granular banking extracts. Diagnoses the
    banked=0 / no-compounding gap without arguing from aggregates."""
    from v5.runtime.algo_grr_membrane import make_frozen_gen, make_lm_compiler, MembraneSolver
    graph = load_seed()
    gen = make_frozen_gen(lm_name, temperature=0.6, max_new_tokens=280)
    compile_fn = make_lm_compiler(gen)
    rounds = curriculum()
    for ri in (2, 3):                       # R3 derive, R4 reuse
        for t in rounds[ri]:
            solver = MembraneSolver(graph, compile_fn)
            r = solver.solve(t)
            picked = [graph.nodes[s].metadata.get("entry", s) for s in r["selected"]]
            print("=" * 72)
            print(f"R{ri+1}  {t['entry']}  |  {t['text']}")
            print(f"  solved={r['solved']}  derived={r['derived']}  selected={picked}  "
                  f"lm_calls={len(solver.compile_inputs)}")
            print("  --- CODE PRODUCED ---")
            print("  " + (r["code"] or "").replace("\n", "\n  "))
            if r["solved"]:
                banked = bank_helper_granular(graph, r["code"], t["entry"])
                print(f"  --- BANKED: {[graph.nodes[b].metadata.get('entry', b) for b in banked]} "
                      f"(graph now {len(graph.nodes)} nodes)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--run", action="store_true", help="two-arm run (stub unless --lm given)")
    ap.add_argument("--inspect", action="store_true", help="raw-dump R3/R4 derive code (needs --lm)")
    ap.add_argument("--lm", default="", help="frozen 3B for the NEW arm (molab)")
    a = ap.parse_args()
    if a.selftest:
        sys.exit(0 if _selftest() else 1)
    if a.inspect:
        inspect_derive(a.lm or "Qwen/Qwen2.5-3B-Instruct")
        return
    if a.run:
        if a.lm:
            from v5.runtime.algo_grr_membrane import make_frozen_gen
            rounds = curriculum()
            gen = make_frozen_gen(a.lm, temperature=0.6, max_new_tokens=220)
            new_m = run_new_arm(rounds, make_lm_compiler(gen))
            _fmt("NEW (frozen 3B + membrane):", new_m)
            print("\n(OLD arm with LoRA train_fn = molab, wire algo_star_epoch.train_lora)")
        else:
            two_arm(verbose=True)
        return
    ap.print_help()


if __name__ == "__main__":
    main()
