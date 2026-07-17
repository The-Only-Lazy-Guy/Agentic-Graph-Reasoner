"""algo_grr_ablate — does the graph help because it HAS more atoms, or because the
MEMBRANE picks better?  No-GPU ablation on the seed curriculum (R1-R4, 10 tasks).

Protocol:
  For each (graph, policy) pair, run the membrane solver. The stub compiler always
  produces the correct code, so the ONLY variable is which atoms the membrane selects.
  Ground-truth needed atoms = which seed atoms the recipe actually CALLs.

Measures: selection precision, selection recall, hops wasted (hops beyond the minimum),
          whether the RIGHT atom was ranked #1 on the first hop.

Three graphs:
  [bare]    seed graph (25 nodes)
  [noise]   seed + 20 random MBPP+ whole-solution stubs (irrelevant, no depend edges)
  [grown]   seed + 20 curriculum-derived helpers (relevant, with depend edges)

Three policies:
  [random]  random ranking
  [cosine]  default cosine
  [topo]    TopologyRetriever (depend-neighbour boost)

    python -m v5.runtime.algo_grr_ablate
"""
from __future__ import annotations

import ast
import re
import sys
from pathlib import Path
from typing import Callable

_ROOT = str(Path(__file__).resolve().parents[2])
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from graph_core import MemoryGraph, Node, Edge  # type: ignore
from v5.runtime.algo_grr_membrane import MembraneSolver, TokenRetriever, make_stub_compiler, verify_code
from v5.runtime.algo_grr_poison_test import curriculum, load_seed


# ═══════════════════════════════════════════════════════════════════════════════
# Ground-truth: which seed atoms does each task's recipe actually CALL?
# ═══════════════════════════════════════════════════════════════════════════════

_SEED_ENTRIES = {
    "is_prime", "gcd", "lcm", "divisors", "sum_divisors", "is_perfect",
    "reverse_digits", "is_palindrome_number", "reverse_string", "is_palindrome",
    "char_freq", "is_anagram", "most_common", "count_occurrences",
    "is_even", "is_odd", "factorial", "nth_fibonacci", "sum_of_squares",
    "triangular", "digit_product", "digit_sum",
}


def _called_entries(code: str) -> set[str]:
    """Return the set of seed entry names CALLED in code (not defined)."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return set()
    defs = {node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)}
    calls = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if node.func.id in _SEED_ENTRIES and node.func.id not in defs:
                calls.add(node.func.id)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            if node.func.attr in _SEED_ENTRIES and node.func.attr not in defs:
                calls.add(node.func.attr)
    return calls


def needs_for(tasks: list[dict]) -> dict[str, set[str]]:
    """Map entry -> set of seed atom entries the recipe calls."""
    return {t["entry"]: _called_entries(t["recipe"]) for t in tasks}


# ═══════════════════════════════════════════════════════════════════════════════
# Graph variants
# ═══════════════════════════════════════════════════════════════════════════════

_NOISE_STUBS = [
    ("count_vowels_in_string", "def count_vowels_in_string(s): return sum(1 for c in s.lower() if c in 'aeiou')"),
    ("capitalize_words", "def capitalize_words(s): return ' '.join(w.capitalize() for w in s.split())"),
    ("remove_duplicates", "def remove_duplicates(lst): return list(dict.fromkeys(lst))"),
    ("flatten_nested", "def flatten_nested(lst): return [x for sub in lst for x in (sub if isinstance(sub, list) else [sub])]"),
    ("is_sorted", "def is_sorted(lst): return all(lst[i] <= lst[i+1] for i in range(len(lst)-1))"),
    ("longest_word", "def longest_word(s): return max(s.split(), key=len)"),
    ("list_to_freq_dict", "def list_to_freq_dict(lst): return {x: lst.count(x) for x in set(lst)}"),
    ("unique_pairs", "def unique_pairs(lst): return [(a, b) for i, a in enumerate(lst) for b in lst[i+1:]]"),
    ("running_sum", "def running_sum(lst): return [sum(lst[:i+1]) for i in range(len(lst))]"),
    ("transpose_matrix", "def transpose_matrix(m): return list(map(list, zip(*m)))"),
    ("chunk_list", "def chunk_list(lst, n): return [lst[i:i+n] for i in range(0, len(lst), n)]"),
    ("interleave", "def interleave(a, b): return [x for pair in zip(a, b) for x in pair]"),
    ("batched", "def batched(lst, n): return [lst[i:i+n] for i in range(0, len(lst), n)]"),
    ("cycle", "def cycle(lst): return lst * ((max(1, len(lst)) // len(lst)) + 1)"),
    ("shuffle_peek", "def shuffle_peek(lst): return lst[-1] if lst else None"),
    ("mean", "def mean(lst): return sum(lst) / len(lst) if lst else 0"),
    ("median", "def median(lst): s = sorted(lst); n = len(s); return s[n//2] if n else 0"),
    ("starts_with_vowel", "def starts_with_vowel(s): return s[0].lower() in 'aeiou' if s else False"),
    ("trim_whitespace", "def trim_whitespace(s): return ' '.join(s.split())"),
    ("swap_case", "def swap_case(s): return ''.join(c.swapcase() for c in s)"),
]

_GROWN_HELPERS = [
    ("sum_of_squares", "def sum_of_squares(n): return sum(i*i for i in range(1, n+1))"),
    ("fib", "def fib(n): a, b = 0, 1\n    for _ in range(n): a, b = b, a + b\n    return a"),
    ("is_leap_year", "def is_leap_year(y): return y % 4 == 0 and (y % 100 != 0 or y % 400 == 0)"),
    ("days_in_month", "def days_in_month(m, y): return 31 if m in (1,3,5,7,8,10,12) else 30 if m in (4,6,9,11) else 29 if is_leap_year(y) else 28"),
    ("hcf", "def hcf(a, b): return gcd(a, b)"),
    ("lcm_of_three", "def lcm_of_three(a, b, c): return lcm(lcm(a, b), c)"),
    ("prime_factors", "def prime_factors(n): return [i for i in range(2, n+1) if n % i == 0 and is_prime(i)]"),
    ("count_divisors", "def count_divisors(n): return len(divisors(n))"),
    ("sum_digits_squared", "def sum_digits_squared(n): return sum(int(d)**2 for d in str(n))"),
    ("is_square", "def is_square(n): return int(n**0.5)**2 == n"),
    ("is_cube", "def is_cube(n): return round(n**(1/3))**3 == n"),
    ("digit_reversal_sum", "def digit_reversal_sum(n): return n + reverse_digits(n)"),
    ("is_armstrong", "def is_armstrong(n): return sum(int(d)**len(str(n)) for d in str(n)) == n"),
    ("nth_triangular", "def nth_triangular(n): return n*(n+1)//2"),
    ("sine_approx", "def sine_approx(x): return x - x**3/6 + x**5/120"),
    ("digit_product_nonzero", "def digit_product_nonzero(n): return digit_product(n) or 1"),
    ("is_automorphic", "def is_automorphic(n): return str(n*n).endswith(str(n))"),
    ("alternating_sum", "def alternating_sum(lst): return sum(lst[::2]) - sum(lst[1::2])"),
    ("bisect_left", "def bisect_left(lst, x): lo, hi = 0, len(lst)\n    while lo < hi: mid = (lo+hi)//2\n        if lst[mid] < x: lo = mid+1\n        else: hi = mid\n    return lo"),
    ("has_duplicate", "def has_duplicate(lst): return len(set(lst)) < len(lst)"),
]

_DEPEND_CHAINS = {
    "sum_of_squares": [],
    "fib": [],
    "is_leap_year": [],
    "days_in_month": ["is_leap_year"],
    "hcf": ["gcd"],
    "lcm_of_three": ["lcm"],
    "prime_factors": ["is_prime"],
    "count_divisors": ["divisors"],
    "sum_digits_squared": [],
    "is_square": [],
    "is_cube": [],
    "digit_reversal_sum": ["reverse_digits"],
    "is_armstrong": [],
    "nth_triangular": [],
    "sine_approx": [],
    "digit_product_nonzero": ["digit_product"],
    "is_automorphic": [],
    "alternating_sum": [],
    "bisect_left": [],
    "has_duplicate": [],
}


def _add_stubs(graph: MemoryGraph, stubs: list[tuple[str, str]],
               part_of: str = "noise", base_depends: dict[str, list[str]] | None = None):
    deps = base_depends or {}
    for entry, code in stubs:
        nid = f"impl_{entry}"
        graph.nodes[nid] = Node(
            id=nid, node_type="implementation",
            text=f"{entry} helper",
            metadata={"entry": entry, "code": code, "origin": "stub"},
        )
        graph.edges.append(Edge(src=part_of, dst=nid, relation="part_of"))
        for dep in deps.get(entry, []):
            dep_id = f"impl_{dep}"
            if dep_id in graph.nodes:
                graph.edges.append(Edge(src=nid, dst=dep_id, relation="depend"))


def make_noise_graph() -> MemoryGraph:
    g = load_seed()
    noise_count = max(1, len([n for n in g.nodes if g.nodes[n].node_type == "implementation"]) - 21)
    _add_stubs(g, _NOISE_STUBS[:max(4, noise_count)], part_of="noise_domain")
    return g


def make_grown_graph() -> MemoryGraph:
    g = load_seed()
    _add_stubs(g, _GROWN_HELPERS, part_of="grown_domain", base_depends=_DEPEND_CHAINS)
    return g


# ═══════════════════════════════════════════════════════════════════════════════
# Policies
# ═══════════════════════════════════════════════════════════════════════════════

import random


def make_random_policy(graph: MemoryGraph, seed: int = 0):
    rng = random.Random(seed)

    def policy_fn(task: dict, selected: list[str], graph: MemoryGraph, retriever: TokenRetriever):
        rank = retriever.rank(task["text"], exclude=set(selected))
        rng.shuffle(rank)
        return rank
    return policy_fn


def make_cosine_policy(graph: MemoryGraph):
    def policy_fn(task: dict, selected: list[str], graph: MemoryGraph, retriever: TokenRetriever):
        return retriever.rank(task["text"], exclude=set(selected))
    return policy_fn


def make_topo_policy(graph: MemoryGraph):
    from v5.runtime.algo_grr_retrieval import make_topology_policy
    return make_topology_policy(graph)


# ═══════════════════════════════════════════════════════════════════════════════
# Run and measure
# ═══════════════════════════════════════════════════════════════════════════════

def node_entry(graph, nid):
    return graph.nodes[nid].metadata.get("entry", nid)


def run_ablation(graph: MemoryGraph, policy_fn: Callable | None, label: str,
                 tasks: list[dict], needs: dict[str, set[str]]) -> dict:
    from v5.runtime.algo_grr_retrieval import CachedTokenRetriever
    retriever = CachedTokenRetriever(graph)
    compile_fn = make_stub_compiler({t["entry"]: t["recipe"] for t in tasks})
    total_prec = total_recall = total_hops = total_lm = correct_first = n = 0
    for t in tasks:
        solver = MembraneSolver(graph, compile_fn, retriever=retriever, policy_fn=policy_fn,
                                max_hops=6, max_retries=1)
        r = solver.solve(t)
        selected_entries = {node_entry(graph, s) for s in r["selected"]}
        ground = needs.get(t["entry"], set())
        inter = selected_entries & ground
        prec = len(inter) / len(selected_entries) if selected_entries else 0.0
        rec = len(inter) / len(ground) if ground else 1.0
        total_prec += prec
        total_recall += rec
        total_hops += len(r.get("trace", []))
        total_lm += len(solver.compile_inputs)
        if r.get("trace") and ground:
            first_pick_entries = {node_entry(graph, r["trace"][0]["picked"])} if r["trace"][0].get("picked", "") else set()
            if first_pick_entries & ground:
                correct_first += 1
        n += 1
    return dict(
        label=label,
        n=n,
        solved=n,  # stub always solves
        avg_precision=total_prec / n,
        avg_recall=total_recall / n,
        avg_hops=total_hops / n,
        avg_lm_calls=total_lm / n,
        first_hop_hit=correct_first / n if n else 0.0,
        graph_nodes=len(graph.nodes),
    )


def fmt_row(r: dict) -> str:
    return (f"  {r['label']:25s}  prec={r['avg_precision']:.2f}  recall={r['avg_recall']:.2f}  "
            f"hops={r['avg_hops']:.2f}  lm={r['avg_lm_calls']:.1f}  "
            f"first_hit={r['first_hop_hit']:.2f}  graph={r['graph_nodes']}")


def main() -> None:
    print("algo_grr_ablate — graph knowledge vs retrieval quality (no GPU)\n")
    print("Question: does the graph help because it HAS more atoms (knowledge),")
    print("or because the membrane PICKS better (retrieval skill)?\n")

    tasks = [t for rnd in curriculum() for t in rnd]
    needs = needs_for(tasks)

    graphs = {
        "bare (seed 25n)": load_seed,
        "noise (seed+noise)": make_noise_graph,
        "grown (seed+helpers)": make_grown_graph,
    }
    policies = {
        "random": make_random_policy,
        "cosine": make_cosine_policy,
        "topo": make_topo_policy,
    }

    results = []
    for g_label, g_fn in graphs.items():
        for p_label, p_fn in policies.items():
            g = g_fn()
            pol = p_fn(g)
            label = f"{p_label} / {g_label}"
            r = run_ablation(g, pol, label, tasks, needs)
            results.append(r)
            print(fmt_row(r))

    # isolate the two axes
    print("\n-- ISOLATION --\n")

    # Same policy, different graph sizes
    for pol in ["cosine", "topo"]:
        row_bare = next(r for r in results if r["label"].startswith(f"{pol} / bare"))
        row_grown = next(r for r in results if r["label"].startswith(f"{pol} / grown"))
        dp = row_grown['avg_precision'] - row_bare['avg_precision']
        dr = row_grown['avg_recall'] - row_bare['avg_recall']
        dh = row_grown['avg_hops'] - row_bare['avg_hops']
        df = row_grown['first_hop_hit'] - row_bare['first_hop_hit']
        print(f"  [{pol}] grown - bare:  prec {dp:+.2f}  recall {dr:+.2f}  hops {dh:+.2f}  first_hit {df:+.2f}")

    # Same graph, different policies
    for g in ["bare (seed 25n)", "grown (seed+helpers)"]:
        row_random = next(r for r in results if r["label"] == f"random / {g}")
        row_cosine = next(r for r in results if r["label"] == f"cosine / {g}")
        row_topo = next(r for r in results if r["label"] == f"topo / {g}")
        dpr = row_topo['avg_precision'] - row_random['avg_precision']
        drr = row_topo['avg_recall'] - row_random['avg_recall']
        dpk = row_topo['avg_precision'] - row_cosine['avg_precision']
        drk = row_topo['avg_recall'] - row_cosine['avg_recall']
        print(f"  [{g}] topo - random: prec {dpr:+.2f}  recall {drr:+.2f}")
        print(f"  [{g}] topo - cosine: prec {dpk:+.2f}  recall {drk:+.2f}")

    # noise vs bare (does noise degrade?)
    print("\n-- NOISE EFFECT (does irrelevant knowledge hurt?) --\n")
    for pol in ["cosine", "topo"]:
        row_bare = next(r for r in results if r["label"].startswith(f"{pol} / bare"))
        row_noise = next(r for r in results if r["label"].startswith(f"{pol} / noise"))
        dp = row_noise['avg_precision'] - row_bare['avg_precision']
        dr = row_noise['avg_recall'] - row_bare['avg_recall']
        dh = row_noise['avg_hops'] - row_bare['avg_hops']
        print(f"  [{pol}] noise - bare:  prec {dp:+.2f}  recall {dr:+.2f}  hops {dh:+.2f}")

    print("\nVERDICT:")
    print("  If grown >> bare on the SAME policy:  more atoms help (knowledge effect)")
    print("  If topo >> cosine on the SAME graph:   retrieval skill helps (policy effect)")
    print("  If noise ~= bare:                       membrane filters irrelevant atoms robustly")
    print("  If noise << bare:                       irrelevant atoms confuse the membrane")


if __name__ == "__main__":
    main()
