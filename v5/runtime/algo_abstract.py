"""GRR-4: MUTATE + ABSTRACT — the generalization lever. Composition alone can't invent; these are the
operators that can, and Δcapability (GRR-3) is the honest judge of whether the result earns its place.

  MUTATE   edit ONE atom (change a bound / comparison / init) -> a variant. Small leaps + repair.
  ABSTRACT extract a shared control pattern from several atoms into a HIGHER-ORDER atom, rewriting each
           as a one-line instance. This is the move that mints IRREPLACEABLE atoms — the Δcapability
           finding said only EXPENSIVE-to-re-derive atoms are load-bearing, and an abstracted skeleton
           (e.g. the 2D-string-DP that edit_distance AND lcs_length both are) is exactly that.

Every mint is GATED: fuzz-EQUIVALENCE (the rewrite must match the original on random inputs — MDL
proposes, equivalence vetoes; this is the guard against the lossy false-merge we hit before), MDL
(compress the corpus), and downstream Δcapability (irreplaceable = worth keeping).

Showcase: ABSTRACT str_dp2 from {edit_distance, lcs_length} -> both become 1-line instances, behavior
preserved, library compressed, and the skeleton is exactly the expensive/irreplaceable kind.

  selftest (no model):  python -m v5.runtime.algo_abstract --selftest
"""
from __future__ import annotations

import argparse
import sys

import numpy as np

from v5.runtime.algo_compose_tasks import _edit_distance, _lcs_length
from v5.runtime.algo_quality import _count_pass, mdl_size


# ── MUTATE ────────────────────────────────────────────────────────────────────────
def mutate(code: str, subs: dict) -> str:
    """Apply targeted edits (old->new) to an atom -> a variant. e.g. repair the count_makeable truthy
    bug: `!= -1` in place of a bare truthiness; or make A*'s bound f-based for IDA*."""
    for old, new in subs.items():
        code = code.replace(old, new)
    return code


# ── ABSTRACT: the 2D-string-DP skeleton (the irreplaceable higher-order atom) ──────
STR_DP2 = (
    "def str_dp2(a, b, base_i, base_j, on_match, on_nomatch):\n"
    "    m, n = len(a), len(b)\n"
    "    dp = [[0] * (n + 1) for _ in range(m + 1)]\n"
    "    for i in range(m + 1):\n        dp[i][0] = base_i(i)\n"
    "    for j in range(n + 1):\n        dp[0][j] = base_j(j)\n"
    "    for i in range(1, m + 1):\n        for j in range(1, n + 1):\n"
    "            d, u, l = dp[i-1][j-1], dp[i-1][j], dp[i][j-1]\n"
    "            dp[i][j] = on_match(d, u, l) if a[i-1] == b[j-1] else on_nomatch(d, u, l)\n"
    "    return dp[m][n]")

# each leaf atom, rewritten as a ONE-LINE instance of the skeleton (behavior-identical)
DP2_INSTANCES = {
    "edit_distance": "def edit_distance(a, b):\n    return str_dp2(a, b, lambda i: i, lambda j: j, "
                     "lambda d, u, l: d, lambda d, u, l: 1 + min(d, u, l))",
    "lcs_length": "def lcs_length(a, b):\n    return str_dp2(a, b, lambda i: 0, lambda j: 0, "
                  "lambda d, u, l: d + 1, lambda d, u, l: max(u, l))",
}
_DP2_ORACLE = {"edit_distance": _edit_distance, "lcs_length": _lcs_length}


def _rand_strs(rng, n):
    def one():
        return "".join(chr(97 + int(rng.integers(0, 5))) for _ in range(int(rng.integers(0, 8))))
    return [(one(), one()) for _ in range(n)]


def fuzz_equivalent(candidate_code: str, deps: str, name: str, oracle, n: int = 80, seed: int = 0) -> bool:
    """True iff candidate (deps+code) matches the oracle on n random string pairs — the behavior-
    preservation gate for a rewrite/abstraction/merge (NOT just the seen cases -> no lossy merge)."""
    rng = np.random.default_rng(seed)
    asserts = [f"assert {name}({a!r}, {b!r}) == {oracle(a, b)!r}" for a, b in _rand_strs(rng, n)]
    full = (deps + "\n\n" + candidate_code) if deps else candidate_code
    return _count_pass(full, asserts) == len(asserts)


def verify_abstraction(skeleton: str, instances: dict, oracles: dict, n: int = 80) -> dict:
    """Gate an abstraction: every rewritten instance must be fuzz-EQUIVALENT to the original behavior,
    AND the library must COMPRESS (MDL of skeleton+instances < MDL of the originals)."""
    equiv = {name: fuzz_equivalent(inst, skeleton, name, oracles[name], n=n)
             for name, inst in instances.items()}
    mdl_new = mdl_size(skeleton) + sum(mdl_size(c) for c in instances.values())
    # originals reconstructed from the oracle-equivalent leaf forms (what they'd cost stored standalone)
    from v5.runtime.algo_compose_tasks import HARD_ATOMS
    mdl_old = sum(mdl_size(HARD_ATOMS[n][1]) for n in instances if n in HARD_ATOMS)
    return dict(equivalent=all(equiv.values()), per_instance=equiv,
                mdl_new=mdl_new, mdl_old=mdl_old, compresses=mdl_new < mdl_old)


# ═══════════════════════════════════════════════════════════════════════════════
# SELFTEST — MUTATE repairs; ABSTRACT preserves behavior + compresses toward an irreplaceable skeleton
# ═══════════════════════════════════════════════════════════════════════════════

def _selftest() -> bool:
    from v5.runtime.algo_compose_tasks import ALL_ATOMS
    from v5.runtime.algo_quality import fuzz
    print("algo_abstract --selftest: MUTATE repairs a bug; ABSTRACT preserves behavior + compresses\n")

    # [1] MUTATE: repair the count_makeable truthy bug (`if coin_change(...)` counts -1 as truthy)
    buggy = "def count_makeable(coins, amounts):\n    return sum(1 for a in amounts if coin_change(coins, a))"
    fixed = mutate(buggy, {"if coin_change(coins, a)": "if coin_change(coins, a) != -1"})
    deps = ALL_ATOMS["coin_change"][1]
    p_bug, t = fuzz(buggy, "count_makeable", deps, n=60)
    p_fix, _ = fuzz(fixed, "count_makeable", deps, n=60)
    assert p_bug < t and p_fix == t, (p_bug, p_fix, t)
    print(f"  [1] MUTATE repair: buggy agrees {p_bug/t:.0%} of random inputs -> mutated `!= -1` -> "
          f"{p_fix/t:.0%} (fixed) -> PASS")

    # [2] ABSTRACT: str_dp2 rewrites of edit_distance + lcs_length are behavior-EQUIVALENT to originals
    res = verify_abstraction(STR_DP2, DP2_INSTANCES, _DP2_ORACLE, n=80)
    assert res["equivalent"], res["per_instance"]
    print(f"  [2] ABSTRACT: edit_distance + lcs_length rewritten as str_dp2 instances, fuzz-EQUIVALENT "
          f"on 80 random pairs {res['per_instance']} -> PASS")

    # [3] the abstraction COMPRESSES the library (MDL down) and the skeleton is the expensive/
    # irreplaceable kind (a 2D DP the model can't cheaply re-derive under budget -> predicted high delta)
    assert res["compresses"], res
    print(f"  [3] MDL: skeleton+instances={res['mdl_new']} < two full DPs={res['mdl_old']} -> COMPRESSES; "
          f"str_dp2 is the 2D-DP kind delta-capability rated irreplaceable -> PASS")

    print("\n  ALGO_ABSTRACT SELFTEST -> PASS  (delta-capability gate on molab: does str_dp2 out-score the leaves)")
    return True


def main():
    ap = argparse.ArgumentParser(description="GRR-4: MUTATE + ABSTRACT operators (generalization lever).")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        sys.exit(0 if _selftest() else 1)
    ap.print_help()


if __name__ == "__main__":
    main()
