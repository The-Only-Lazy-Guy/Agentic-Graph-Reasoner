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
