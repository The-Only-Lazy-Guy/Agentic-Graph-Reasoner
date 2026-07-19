"""algo_grr_compose — a DECOMPOSABLE task corpus (the compounding stress test MBPP+ can't be).

Local analysis found MBPP+'s compounding ceiling is 2% — 370/378 references are monolithic single
functions, so self-derived reuse is corpus-capped regardless of retrieval. This module is the opposite:
tasks built by COMPOSING a small pool of recurring sub-computations `outer(inner(n))`. A handful of
`inner` primitives (sum_of_squares, nth_fibonacci, factorial, ...) each appear across MANY tasks, so once
one is DERIVED and banked it is REUSED by every later task that needs it — the graph compounds. This is
the honest next corpus for demonstrating strong compounding (APPS-class decomposable structure).

    selftest (no GPU):  python -m v5.runtime.algo_grr_compose --selftest
    molab (real 3B):    python -m v5.runtime.algo_grr_compose --run --lm Qwen/Qwen2.5-3B-Instruct --n 80 --topo
"""
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

_ROOT = str(Path(__file__).resolve().parents[2])
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from v5.runtime.algo_grr_poison_test import load_seed, bank_helper_granular  # noqa: E402
from v5.runtime.algo_grr_mbpp import verify_asserts, run_mbpp  # noqa: E402
from v5.runtime.algo_grr_membrane import make_stub_compiler, bankable_pure_defs  # noqa: E402
from v5.runtime.algo_grr_poison_test import _atom_entries  # noqa: E402


# ── primitive pool: (source, oracle fn, NL phrase). INNER = int->int derivable helpers (NOT in the
#    seed, so they must be derived + banked). OUTER = int->int|bool wrappers. ─────────────────────────
def _fib(n):
    a, b = 0, 1
    for _ in range(n):
        a, b = b, a + b
    return a


def _fact(n):
    r = 1
    for i in range(2, n + 1):
        r *= i
    return r


def _digprod(n):
    p = 1
    for c in str(abs(n)):
        p *= int(c)
    return p


def _ndiv(n):
    return sum(1 for i in range(1, abs(n) + 1) if n % i == 0) if n else 0


INNER = {
    "sum_of_squares": ("def sum_of_squares(n):\n    return sum(i * i for i in range(1, n + 1))\n",
                       lambda n: sum(i * i for i in range(1, n + 1)), "the sum of squares of 1..n"),
    "nth_fibonacci": ("def nth_fibonacci(n):\n    a, b = 0, 1\n    for _ in range(n):\n"
                      "        a, b = b, a + b\n    return a\n", _fib, "the n-th fibonacci number"),
    "factorial": ("def factorial(n):\n    r = 1\n    for i in range(2, n + 1):\n        r *= i\n"
                  "    return r\n", _fact, "the factorial of n"),
    "triangular": ("def triangular(n):\n    return n * (n + 1) // 2\n",
                   lambda n: n * (n + 1) // 2, "the n-th triangular number"),
    "digit_product": ("def digit_product(n):\n    p = 1\n    for c in str(abs(n)):\n"
                      "        p *= int(c)\n    return p\n", _digprod, "the product of the digits of n"),
}
OUTER = {
    "is_prime": ("def is_prime(n):\n    if n < 2:\n        return False\n    i = 2\n"
                 "    while i * i <= n:\n        if n % i == 0:\n            return False\n"
                 "        i += 1\n    return True\n",
                 lambda x: x >= 2 and all(x % i for i in range(2, int(x ** 0.5) + 1)),
                 "whether {v} is prime"),
    "reverse_digits": ("def reverse_digits(n):\n    r = 0\n    n = abs(n)\n    while n:\n"
                       "        r = r * 10 + n % 10\n        n //= 10\n    return r\n",
                       lambda x: int(str(abs(x))[::-1]), "the digit-reversal of {v}"),
    "digit_sum": ("def digit_sum(n):\n    return sum(int(c) for c in str(abs(n)))\n",
                  lambda x: sum(int(c) for c in str(abs(x))), "the digit sum of {v}"),
    "num_divisors": ("def num_divisors(n):\n    return sum(1 for i in range(1, abs(n) + 1) "
                     "if n % i == 0) if n else 0\n", _ndiv, "the number of divisors of {v}"),
}


def gen_corpus(n_tasks: int = 60, seed: int = 0) -> list[dict]:
    """Compose outer(inner(n)) tasks with a SMALL primitive pool -> heavy recurrence -> compounding."""
    rng = random.Random(seed)
    combos = [(i, o) for i in INNER for o in OUTER]
    rng.shuffle(combos)
    tasks = []
    for k in range(n_tasks):
        inner, outer = combos[k % len(combos)]
        i_code, i_fn, i_desc = INNER[inner]
        o_code, o_fn, o_desc = OUTER[outer]
        entry = f"t_{k:03d}"
        text = f"{o_desc.format(v=i_desc)}"
        # self-contained reference: define both prims + the entry that COMPOSES them
        ref = f"{i_code}\n{o_code}\ndef {entry}(n):\n    return {outer}({inner}(n))\n"
        asserts = []
        for n in (2, 3, 4, 5):
            try:
                exp = o_fn(i_fn(n))
            except Exception:  # noqa: BLE001
                continue
            asserts.append(f"assert {entry}({n}) == {exp!r}")

        def _mk(a=asserts):
            return lambda code: verify_asserts(code, a)

        tasks.append(dict(text=text, entry=entry, examples=asserts, verify_fn=_mk(),
                          type_pool=[int], tests=[], reference=ref,
                          _prims=(inner, outer)))
    return tasks


# ── HARD reusable helpers: non-trivial algorithms a frozen 3B often FAILS to write inline (partitions,
#    derangements, josephus, ...). Each recurs across many tasks, so a banked+verified copy is
#    LOAD-BEARING: OURS banks once then reuses; RAG must re-derive inline every time and errs. This is
#    the corpus that separates compounding from static-RAG (gen_corpus's easy prims cannot). ───────────
def _partitions(n):
    dp = [1] + [0] * n
    for k in range(1, n + 1):
        for i in range(k, n + 1):
            dp[i] += dp[i - k]
    return dp[n]


def _derangements(n):
    if n == 0:
        return 1
    a, b = 1, 0
    for i in range(2, n + 1):
        a, b = b, (i - 1) * (a + b)
    return b


def _josephus(n):
    r = 0
    for i in range(1, n + 1):
        r = (r + 2) % i
    return r + 1


def _catalan(n):
    from math import comb
    return comb(2 * n, n) // (n + 1)


def _mult_persistence(n):
    s, n = 0, abs(n)
    while n >= 10:
        p = 1
        for c in str(n):
            p *= int(c)
        n = p
        s += 1
    return s


HARD = {
    "num_partitions": ("def num_partitions(n):\n    dp = [1] + [0] * n\n    for k in range(1, n + 1):\n"
                       "        for i in range(k, n + 1):\n            dp[i] += dp[i - k]\n    return dp[n]\n",
                       _partitions, "the number of integer partitions of n"),
    "derangements": ("def derangements(n):\n    if n == 0:\n        return 1\n    a, b = 1, 0\n"
                     "    for i in range(2, n + 1):\n        a, b = b, (i - 1) * (a + b)\n    return b\n",
                     _derangements, "the number of derangements of n items"),
    "josephus": ("def josephus(n):\n    r = 0\n    for i in range(1, n + 1):\n        r = (r + 2) % i\n"
                 "    return r + 1\n", _josephus, "the Josephus survivor position for n people (step 2)"),
    "catalan": ("def catalan(n):\n    from math import comb\n    return comb(2 * n, n) // (n + 1)\n",
                _catalan, "the n-th Catalan number"),
    "mult_persistence": ("def mult_persistence(n):\n    s, n = 0, abs(n)\n    while n >= 10:\n        p = 1\n"
                         "        for c in str(n):\n            p *= int(c)\n        n = p\n        s += 1\n"
                         "    return s\n", _mult_persistence, "the multiplicative persistence of n"),
}


# OUTER wrappers used ONLY in the held-out split: trivially easy (the LM writes them fine), so on a
# held-out task the ONLY hard part is the HARD helper. OURS reuses the BANKED helper -> solves; RAG has
# no memory and must re-derive the hard helper inline -> fails. Isolates "reasoner+memory vs inline".
OUTER_HELD = {
    "is_even": ("def is_even(n):\n    return n % 2 == 0\n", lambda x: x % 2 == 0, "whether {v} is even"),
    "last_digit": ("def last_digit(n):\n    return abs(n) % 10\n", lambda x: abs(x) % 10, "the last digit of {v}"),
    "count_digits": ("def count_digits(n):\n    return len(str(abs(n)))\n", lambda x: len(str(abs(x))),
                     "the number of digits in {v}"),
}


def gen_corpus_hard(n_tasks: int = 120, seed: int = 0, holdout: bool = False) -> list[dict]:
    """Compose outer(HARD(n)) tasks. HARD helpers are non-trivial (a frozen 3B often fails to write them
    inline), and each recurs across many tasks -> once DERIVED+banked it is reused, and the reuse is
    LOAD-BEARING (RAG, which re-derives inline, gets HARD wrong). holdout=True uses UNSEEN easy wrappers
    (OUTER_HELD) over the SAME hard helpers -> a pure test of banked-atom reuse: OURS retrieves the
    verified helper it banked earlier; RAG (no memory, no reasoner) must re-write the hard logic inline."""
    outers = OUTER_HELD if holdout else OUTER
    rng = random.Random(seed + (10_000 if holdout else 0))
    combos = [(h, o) for h in HARD for o in outers]
    rng.shuffle(combos)
    tasks = []
    for k in range(n_tasks):
        hard, outer = combos[k % len(combos)]
        h_code, h_fn, h_desc = HARD[hard]
        o_code, o_fn, o_desc = outers[outer]
        entry = f"{'ho' if holdout else 'h'}_{k:03d}"
        text = f"{o_desc.format(v=h_desc)}"
        ref = f"{h_code}\n{o_code}\ndef {entry}(n):\n    return {outer}({hard}(n))\n"
        asserts = []
        for n in (5, 6, 7, 8):
            try:
                exp = o_fn(h_fn(n))
            except Exception:  # noqa: BLE001
                continue
            asserts.append(f"assert {entry}({n}) == {exp!r}")

        def _mk(a=asserts):
            return lambda code: verify_asserts(code, a)

        tasks.append(dict(text=text, entry=entry, examples=asserts, verify_fn=_mk(),
                          type_pool=[int], tests=[], reference=ref, _prims=(hard, outer),
                          atom_oracles={hard: h_fn}))   # fuzz-generality gate: bank the helper iff general
    return tasks


def gen_corpus_multihard(n_tasks: int = 90, seed: int = 0) -> list[dict]:
    """Each task chains TWO hard helpers: W( h1( josephus(n) ) ) — needs josephus AND h1 authored. Multiple
    missing atoms per task is what makes STEP-SPECULATION pay: the tiny planner proposes the whole K-atom
    program at once, the LM authors+verifies the CHUNK in ONE call instead of one call per atom. josephus
    is a bounded shrinker (≤n) so nested values stay small/computable. Also strengthens the RAG-fails case
    (RAG must inline BOTH hard helpers). `_wprog` = (atoms dep-first, wiring tree) for the realizer."""
    rng = random.Random(seed + 555)
    h1_pool = [h for h in HARD if h != "josephus"]
    tasks = []
    for k in range(n_tasks):
        h1 = h1_pool[k % len(h1_pool)]
        wname = list(OUTER)[k % len(OUTER)]
        h1_code, h1_fn, h1_desc = HARD[h1]
        j_code, j_fn, _ = HARD["josephus"]
        w_code, w_fn, w_desc = OUTER[wname]
        entry = f"m_{k:03d}"
        text = w_desc.format(v=f"{h1_desc} of the Josephus survivor for n people")
        ref = f"{j_code}\n{h1_code}\n{w_code}\ndef {entry}(n):\n    return {wname}({h1}(josephus(n)))\n"
        asserts = []
        for n in (5, 6, 7, 8):
            try:
                exp = w_fn(h1_fn(j_fn(n)))
            except Exception:  # noqa: BLE001
                continue
            asserts.append(f"assert {entry}({n}) == {exp!r}")

        def _mk(a=asserts):
            return lambda code: verify_asserts(code, a)

        wiring = ("call", wname, [("call", h1, [("call", "josephus", ["n"])])])
        tasks.append(dict(text=text, entry=entry, examples=asserts, verify_fn=_mk(),
                          type_pool=[int], tests=[], reference=ref, _prims=(h1, wname),
                          _wprog=(["josephus", h1, wname], wiring),
                          atom_oracles={h1: h1_fn, "josephus": j_fn}))   # fuzz-generality gate
    return tasks


# ═══════════════════════════════════════════════════════════════════════════════
# SELFTEST — the corpus is decomposable (ceiling ~100%) and compounds (derived_reuse >> MBPP+)
# ═══════════════════════════════════════════════════════════════════════════════

def _selftest() -> bool:
    print("algo_grr_compose --selftest: decomposable corpus (compounding stress test)\n")
    tasks = gen_corpus(60, seed=0)
    ok = True

    # [1] references valid (self-contained, pass their own asserts)
    good = sum(1 for t in tasks if t["verify_fn"](t["reference"])[0] >= 1.0)
    print(f"  [1] references valid: {good}/{len(tasks)} pass own asserts -> "
          f"{'PASS' if good == len(tasks) else 'FAIL'}")
    ok &= good == len(tasks)

    # [2] FACTORABILITY CEILING ~100% (vs MBPP+ 2%): every ref factors an INNER primitive
    seed = load_seed()
    existing = set(_atom_entries(seed).keys())
    have = sum(1 for t in tasks if len(bankable_pure_defs(t["reference"], existing | {t["entry"]})) >= 1)
    print(f"  [2] factorability ceiling: {have}/{len(tasks)} refs have >=1 extractable helper "
          f"({100*have//len(tasks)}%  vs MBPP+ 2%) -> {'PASS' if have == len(tasks) else 'FAIL'}")
    ok &= have == len(tasks)

    # [3] COMPOUNDING: stub driver over the stream -> derived_reuse grows large (prims recur)
    graph = load_seed()
    stub = make_stub_compiler({t["entry"]: t["reference"] for t in tasks})
    res = run_mbpp(graph, tasks, stub, chunk=20, verbose=True)
    print(f"  [3] compounding: solved {res['solved']}/{res['n']}, banked {res['banked']}, "
          f"derived_reuse {res['derived_reuse']} (MBPP+ was ~3) -> "
          f"{'PASS' if res['derived_reuse'] >= 15 else 'FAIL'}")
    ok &= res["derived_reuse"] >= 15

    print(f"\n  ALGO_GRR_COMPOSE SELFTEST -> {'PASS' if ok else 'FAIL'}")
    return ok


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--lm", default="")
    ap.add_argument("--n", type=int, default=80)
    ap.add_argument("--topo", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        sys.exit(0 if _selftest() else 1)
    if a.run:
        tasks = gen_corpus(a.n, seed=0)
        graph = load_seed()
        from v5.runtime.algo_grr_retrieval import CachedTokenRetriever
        retriever = CachedTokenRetriever(graph)
        policy_fn = None
        if a.topo:
            from v5.runtime.algo_grr_retrieval import make_topology_policy
            policy_fn = make_topology_policy(graph)
        if a.lm:
            import os
            os.environ["V5_HARD_VERIFY"] = "1"            # subprocess verify -> LM code can't hang the run
            from v5.runtime.algo_grr_membrane import make_frozen_gen, make_lm_compiler
            compile_fn = make_lm_compiler(make_frozen_gen(a.lm, temperature=0.6, max_new_tokens=320))
        else:
            compile_fn = make_stub_compiler({t["entry"]: t["reference"] for t in tasks})
            print("(stub = reference; use --lm for the real run)")
        print(f"decomposable corpus: {len(tasks)} tasks, lm={a.lm or 'stub'}, "
              f"retrieval={'topo' if a.topo else 'cosine'}\n")
        res = run_mbpp(graph, tasks, compile_fn, policy_fn=policy_fn, retriever=retriever)
        print(f"\nSOLVED {res['solved']}/{res['n']} | banked {res['banked']} | cross-task reuse "
              f"{res['reuse']} (DERIVED {res['derived_reuse']}) | graph {res['graph_nodes']} nodes")
        return
    ap.print_help()


if __name__ == "__main__":
    main()
