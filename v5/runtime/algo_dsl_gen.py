"""GRR-9a: the HARD family factory — the measuring instrument the 6 hand-written families couldn't be.

Both decoders (GRU and the gen-1 recursive TRM core) saturate the 6-family benchmark at 100%: the task
was too easy to discriminate architectures (gen-1 "wasn't bad, the task was too easy"). This factory
generates N families programmatically with a controllable difficulty axis:

  pipeline (canonical) = FILTER(pred)* (0-2)  ->  MAP(transform) chain (1-4)  ->  REDUCE(agg)

Depth = the MAP-CHAIN length: t3(t2(t1(x))) is a chain of DEPENDENT choices — exactly where a recursive
(think-before-emit) decoder should earn its keep over a single-pass one. Canonical order matches the
realizer's semantics (algo_dsl.realize_program applies FILTER guards to the RAW element and composes
MAPs), so a reference pipeline realizes to code whose behavior IS the oracle's.

  atoms      wider unary pool: predicates (is_prime, is_odd, is_square) + transforms (digit_sum, square,
             reverse_digits, count_divisors, collatz_steps, double). Each has code (for the graph /
             compose-forcing) AND a python oracle fn (for interpretation).
  oracle     the pipeline INTERPRETED with oracle fns (never the realized code — code is what's tested).
  text       synthesized per pipeline from per-atom phrases -> mpnet-embeddable, distinct by construction.
  dedup      behavior fingerprint on shared random inputs (two pipelines that compute the same function
             collapse to one family — no duplicate-behavior families).
  verify     gen_verify(code, deps, fam, n, seed): agreement with the family's oracle on random lists —
             the same epistemics as fast_reward but for FACTORY families (fast_reward's hardcoded
             _FAMILIES stays untouched; GRR-7/8 paths unaffected).

  selftest (no LM):  python -m v5.runtime.algo_dsl_gen --selftest
"""
from __future__ import annotations

import argparse
import sys

import numpy as np

from v5.runtime.algo_dsl import Op, atoms_of, realize_program

# ═══════════════════════════════════════════════════════════════════════════════
# The atom pack: name -> (doc text, code, oracle_fn, role)   role in {pred, map}
# ═══════════════════════════════════════════════════════════════════════════════

def _is_prime(n):
    if n < 2:
        return False
    i = 2
    while i * i <= n:
        if n % i == 0:
            return False
        i += 1
    return True


def _digit_sum(n):
    return sum(int(c) for c in str(abs(n)))


def _reverse_digits(n):
    return int(str(abs(n))[::-1])


def _count_divisors(n):
    n = abs(n)
    if n == 0:
        return 0
    return sum(1 for d in range(1, n + 1) if n % d == 0)


def _collatz_steps(n):
    n = abs(n) or 1
    s = 0
    while n != 1 and s < 200:
        n = n // 2 if n % 2 == 0 else 3 * n + 1
        s += 1
    return s


GEN_ATOMS = {
    # predicates (FILTER)
    "is_prime": ("primality test — True iff n is a prime number",
                 "def is_prime(n):\n    if n < 2:\n        return False\n    i = 2\n"
                 "    while i * i <= n:\n        if n % i == 0:\n            return False\n"
                 "        i += 1\n    return True",
                 _is_prime, "pred"),
    "is_odd": ("odd test — True iff n is odd",
               "def is_odd(n):\n    return n % 2 == 1",
               lambda n: n % 2 == 1, "pred"),
    "is_square": ("perfect-square test — True iff n is a perfect square",
                  "def is_square(n):\n    if n < 0:\n        return False\n    r = int(n ** 0.5)\n"
                  "    return r * r == n or (r + 1) * (r + 1) == n",
                  lambda n: n >= 0 and (int(n ** 0.5)) ** 2 == n or (int(n ** 0.5) + 1) ** 2 == n, "pred"),
    # transforms (MAP)
    "digit_sum": ("digit sum — sum of the decimal digits of n",
                  "def digit_sum(n):\n    return sum(int(c) for c in str(abs(n)))",
                  _digit_sum, "map"),
    "square": ("square — n * n",
               "def square(n):\n    return n * n",
               lambda n: n * n, "map"),
    "reverse_digits": ("digit reversal — the decimal digits of n reversed as an integer",
                       "def reverse_digits(n):\n    return int(str(abs(n))[::-1])",
                       _reverse_digits, "map"),
    "count_divisors": ("divisor count — how many positive integers divide n",
                       "def count_divisors(n):\n    n = abs(n)\n    if n == 0:\n        return 0\n"
                       "    return sum(1 for d in range(1, n + 1) if n % d == 0)",
                       _count_divisors, "map"),
    "collatz_steps": ("Collatz step count — steps for n to reach 1 under the Collatz map",
                      "def collatz_steps(n):\n    n = abs(n) or 1\n    s = 0\n"
                      "    while n != 1 and s < 200:\n        n = n // 2 if n % 2 == 0 else 3 * n + 1\n"
                      "        s += 1\n    return s",
                      _collatz_steps, "map"),
    "double": ("doubling — 2 * n",
               "def double(n):\n    return 2 * n",
               lambda n: 2 * n, "map"),
}

PREDS = [a for a, (_t, _c, _f, r) in GEN_ATOMS.items() if r == "pred"]
MAPS = [a for a, (_t, _c, _f, r) in GEN_ATOMS.items() if r == "map"]
GEN_AGGS = ["sum", "max", "count", "len"]

_PHRASE = {
    "is_prime": "prime", "is_odd": "odd", "is_square": "perfect-square",
    "digit_sum": "digit-sum", "square": "square", "reverse_digits": "digit-reversal",
    "count_divisors": "divisor-count", "collatz_steps": "Collatz-step-count", "double": "double",
}
_AGG_PHRASE = {"sum": "the sum", "max": "the largest value", "count": "the count",
               "len": "the number"}


def pipe_text(pipe) -> str:
    """Synthesize the task text from the pipeline (distinct per family by construction)."""
    preds = [op.arg for op in pipe if op.kind == "FILTER"]
    maps = [op.arg for op in pipe if op.kind == "MAP"]
    agg = pipe[-1].arg
    chain = " of the ".join(_PHRASE[m] for m in reversed(maps))       # t2(t1(x)) reads "t2 of the t1"
    kept = (" and ".join(_PHRASE[p] for p in preds) + " numbers") if preds else "numbers"
    if agg in ("count", "len"):
        return (f"{_AGG_PHRASE[agg]} of {kept} in the list whose {chain} we consider. "
                f"needs {', '.join(sorted(set(preds + maps)))}.")
    return (f"{_AGG_PHRASE[agg]} of the {chain} over the {kept} in the list. "
            f"needs {', '.join(sorted(set(preds + maps)))}.")


# paraphrase pools — the GENERALIZATION axis. With a single fixed text per family, mpnet gives ONE
# deterministic embedding -> the benchmark collapses to memorizing (point -> program) pairs and every
# architecture saturates (measured: 32 fams, 100% everywhere). Variants make a family a REGION in
# embedding space; training on some phrasings and evaluating on HELD-OUT phrasings is a real
# generalization test.
_SYN_PHRASE = {
    "is_prime": ["prime", "prime-number"], "is_odd": ["odd", "non-even"],
    "is_square": ["perfect-square", "square-number"],
    "digit_sum": ["digit-sum", "sum-of-digits"], "square": ["square", "second-power"],
    "reverse_digits": ["digit-reversal", "reversed-digits"],
    "count_divisors": ["divisor-count", "number-of-divisors"],
    "collatz_steps": ["Collatz-step-count", "Collatz-steps"], "double": ["double", "twice-the-value"],
}
_SYN_AGG = {"sum": ["the sum", "the total", "the aggregate"],
            "max": ["the largest value", "the maximum", "the highest result"],
            "count": ["the count", "how many", "the tally"],
            "len": ["the number", "how many", "the tally"]}


def _variant_text(pipe, vi: int) -> str:
    """Deterministic paraphrase #vi of a pipeline's task text. vi=0 == pipe_text (the canonical form).
    Varies: synonyms (per-atom, per-agg), sentence STRUCTURE (nested 'x of the y' vs forward
    'apply y then x'), and whether the 'needs ...' atom hint appears (half the variants drop it)."""
    if vi == 0:
        return pipe_text(pipe)
    rng = np.random.default_rng(10_000 + vi * 977 + len(pipe) * 31 +
                                sum(ord(c) for op in pipe for c in op.arg))
    syn = lambda a: _SYN_PHRASE[a][int(rng.integers(0, len(_SYN_PHRASE[a])))]
    preds = [op.arg for op in pipe if op.kind == "FILTER"]
    maps = [op.arg for op in pipe if op.kind == "MAP"]
    agg = pipe[-1].arg
    aggw = _SYN_AGG[agg][int(rng.integers(0, len(_SYN_AGG[agg])))]
    kept = (" and ".join(syn(p) for p in preds) + " numbers") if preds else "the numbers"
    nested = " of the ".join(syn(m) for m in reversed(maps))
    forward = " then ".join(syn(m) for m in maps)
    hint = f" needs {', '.join(sorted(set(preds + maps)))}." if rng.integers(0, 2) else ""
    if agg in ("count", "len"):
        forms = [f"{aggw} of {kept} in the list.",
                 f"given a list of integers, keep {kept} and report {aggw}.",
                 f"across the list, tally {kept}: {aggw}."]
    else:
        forms = [f"{aggw} of the {nested} over {kept} in the list.",
                 f"given a list of integers, keep {kept}, apply {forward}, then take {aggw}.",
                 f"for each of {kept} compute the {nested}; report {aggw}."]
    return forms[int(rng.integers(0, len(forms)))] + hint


def pipe_text_variants(pipe, k: int) -> list:
    """k DISTINCT paraphrases (variant 0 = the canonical pipe_text). Deterministic."""
    out, vi = [], 0
    while len(out) < k and vi < k * 20:
        t = _variant_text(pipe, vi)
        if t not in out:
            out.append(t)
        vi += 1
    return out


def interpret(pipe, lst) -> int:
    """The ORACLE: interpret the pipeline with oracle fns (canonical semantics — filters on the raw
    element, maps composed — matching realize_program)."""
    preds = [GEN_ATOMS[op.arg][2] for op in pipe if op.kind == "FILTER"]
    maps = [GEN_ATOMS[op.arg][2] for op in pipe if op.kind == "MAP"]
    agg = pipe[-1].arg
    kept = [x for x in lst if all(p(x) for p in preds)]
    if agg in ("count", "len"):
        return len(kept)
    vals = []
    for x in kept:
        for m in maps:
            x = m(x)
        vals.append(x)
    return {"sum": sum(vals), "max": max(vals) if vals else 0}[agg]


def _rand_list(rng):
    return [int(rng.integers(2, 100)) for _ in range(int(rng.integers(4, 9)))]


def _fingerprint(pipe, n: int = 14, seed: int = 5):
    rng = np.random.default_rng(seed)
    return tuple(interpret(pipe, _rand_list(rng)) for _ in range(n))


def gen_families(n_families: int = 24, seed: int = 0, min_chain: int = 1, max_chain: int = 4,
                 max_preds: int = 2):
    """Generate n distinct-BEHAVIOR families. Returns {fam_name: pipeline}. Difficulty = chain length
    (uniform over [min_chain, max_chain]); fam_name encodes the length bucket for reporting."""
    rng = np.random.default_rng(seed)
    fams, seen = {}, set()
    tries = 0
    while len(fams) < n_families and tries < n_families * 200:
        tries += 1
        n_p = int(rng.integers(0, max_preds + 1))
        n_m = int(rng.integers(min_chain, max_chain + 1))
        preds = sorted(rng.choice(PREDS, size=n_p, replace=False).tolist()) if n_p else []
        maps = [MAPS[int(i)] for i in rng.integers(0, len(MAPS), size=n_m)]
        agg = GEN_AGGS[int(rng.integers(0, len(GEN_AGGS)))]
        if agg in ("count", "len"):
            maps = maps[:1]                                   # count ignores the chain -> keep it honest
        pipe = ([Op("FILTER", p) for p in preds] + [Op("MAP", m) for m in maps] + [Op("REDUCE", agg)])
        fp = _fingerprint(pipe)
        if fp in seen or len(set(fp)) <= 1:                   # dup behavior / degenerate (constant) family
            continue
        seen.add(fp)
        L = len(pipe)
        fams[f"gen{L}_{len(fams):03d}"] = pipe
    return fams


class GenTask:
    """Duck-typed like MBPPTask: .name (family), .text (synthesized), .asserts (oracle-computed)."""
    def __init__(self, name, text, asserts):
        self.name, self.text, self.asserts = name, text, asserts


def gen_tasks(fams, n_per: int = 4, seed: int = 0, paraphrase_k: int = 1):
    """Parametrized instances per family (random input lists, oracle-computed asserts). paraphrase_k>1
    cycles the instances through that many distinct phrasings (goal variation — instance i gets variant
    i mod k), so a training pool covers a REGION of goal space, not a point."""
    rng = np.random.default_rng(seed)
    out = []
    for fam, pipe in fams.items():
        texts = pipe_text_variants(pipe, paraphrase_k) if paraphrase_k > 1 else [pipe_text(pipe)]
        for i in range(n_per):
            lst = _rand_list(rng)
            out.append(GenTask(fam, texts[i % len(texts)],
                               [f"assert {fam}({lst!r}) == {interpret(pipe, lst)!r}"]))
    return out


def gen_verify(code: str, deps: str, fams: dict, fam: str, n: int = 32, seed: int = 0) -> float:
    """Agreement fraction of `code` (a candidate realization for `fam`) with the family ORACLE on n
    random lists — fast_reward's epistemics for factory families (reference pipeline only defines the
    oracle BEHAVIOR; candidate code never sees it)."""
    if fam not in fams:
        return 0.0
    ns: dict = {}
    try:
        exec((deps + "\n" + code) if deps else code, ns)
    except Exception:
        return 0.0
    fn = ns.get(fam)
    if fn is None:
        return 0.0
    rng = np.random.default_rng(seed)
    match = 0
    for _ in range(n):
        lst = _rand_list(rng)
        try:
            match += int(fn(lst) == interpret(fams[fam], lst))
        except Exception:
            pass
    return match / n


def pipe_is_general(pipe_candidate, fams: dict, fam: str, n: int = 32) -> bool:
    """Keep-gate for factory families: realize the candidate, fully general on TWO disjoint input sets."""
    try:
        code = realize_program(fam, "list", pipe_candidate)
    except (ValueError, KeyError):
        return False
    deps = "\n\n".join(GEN_ATOMS[a][1] for a in atoms_of(pipe_candidate) if a in GEN_ATOMS)
    return gen_verify(code, deps, fams, fam, n, seed=0) >= 1.0 and \
        gen_verify(code, deps, fams, fam, n, seed=7) >= 1.0


# ═══════════════════════════════════════════════════════════════════════════════
# SELFTEST — coverage (every reference realizes + verifies 1.0), wrong pipe fails, behaviors and
# texts distinct, difficulty axis present (length buckets populated).
# ═══════════════════════════════════════════════════════════════════════════════

def _selftest() -> bool:
    print("algo_dsl_gen --selftest: hard family factory (the discriminating benchmark)\n")
    fams = gen_families(24, seed=0)
    assert len(fams) == 24, len(fams)

    # [1] coverage: every reference pipeline realizes to code whose behavior == the oracle
    for fam, pipe in fams.items():
        assert pipe_is_general(pipe, fams, fam), (fam, pipe)
    lens = sorted({len(p) for p in fams.values()})
    print(f"  [1] 24 families generated; EVERY reference realizes + verifies 1.0 vs its oracle "
          f"(lengths {lens[0]}..{lens[-1]}) -> PASS")

    # [2] difficulty axis: multiple length buckets, including chains >= 3 (the discriminating regime)
    buckets = {}
    for fam, p in fams.items():
        buckets.setdefault(len(p), []).append(fam)
    assert max(buckets) >= 5, buckets.keys()
    print(f"  [2] length buckets {{{', '.join(f'{k}: {len(v)}' for k, v in sorted(buckets.items()))}}} "
          f"(chain depth = dependent choices) -> PASS")

    # [3] a WRONG pipeline (perturbed atom) fails verify -> the choice is real
    fam, pipe = next((f, p) for f, p in fams.items() if len(p) >= 4)
    wrong = list(pipe)
    i = next(i for i, op in enumerate(wrong) if op.kind == "MAP")
    alt = next(m for m in MAPS if m != wrong[i].arg)
    wrong[i] = Op("MAP", alt)
    assert not pipe_is_general(wrong, fams, fam), (fam, wrong)
    print(f"  [3] perturbed pipeline ({wrong[i].arg} in {pipe[i].arg}'s slot) FAILS verify -> PASS")

    # [4] behaviors and texts pairwise distinct (dedup worked; no text collapse)
    fps = [_fingerprint(p) for p in fams.values()]
    texts = [pipe_text(p) for p in fams.values()]
    assert len(set(fps)) == len(fps) and len(set(texts)) == len(texts)
    print(f"  [4] {len(fps)} behavior fingerprints distinct + {len(texts)} texts distinct -> PASS")

    # [5] tasks: oracle-computed asserts execute true against the realized reference
    tasks = gen_tasks(fams, n_per=2, seed=1)
    t = tasks[0]
    pipe = fams[t.name]
    code = realize_program(t.name, "list", pipe)
    deps = "\n\n".join(GEN_ATOMS[a][1] for a in atoms_of(pipe))
    ns: dict = {}
    exec(deps + "\n" + code, ns)
    for a in t.asserts:
        exec(a, ns)
    print(f"  [5] {len(tasks)} parametrized instances; asserts sound by construction -> PASS")

    # [6] paraphrase axis: k distinct variants per family (variant 0 = canonical), deterministic
    for fam, pipe in list(fams.items())[:8]:
        vs = pipe_text_variants(pipe, 5)
        assert len(vs) == 5 and len(set(vs)) == 5 and vs[0] == pipe_text(pipe), (fam, vs)
        assert vs == pipe_text_variants(pipe, 5), "variants must be deterministic"
    print("  [6] paraphrase variants: 5 distinct deterministic phrasings/family (variant 0 = canonical; "
          "synonyms + structure flips + hint dropout) -> PASS")

    print("\n  ALGO_DSL_GEN SELFTEST -> PASS  (a benchmark that can finally discriminate architectures)")
    return True


def main():
    ap = argparse.ArgumentParser(description="GRR-9a: hard parametric DSL family factory.")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        sys.exit(0 if _selftest() else 1)
    ap.print_help()


if __name__ == "__main__":
    main()
