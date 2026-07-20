"""algo_grr_refract — Polymorphic Spec Refraction (the authoring-tax fix, user's Proposal 1).

The frozen 3B's authoring failures are SYSTEMATIC (an attractor collapse): the same prompt maps into the
same broken basin every sample, so Best-of-N resamples the same wrong code (measured: best-of-4 TRIPLED
the floor). We can't touch the weights (on-device, anti-poison). So change HOW the model experiences the
spec: on verify-fail, deterministically MUTATE the specification surface (identifier transmutation, paradigm
inversion) to shift the initial token distribution into a DIFFERENT basin — decorrelated retries, not
resamples. If any refracted variant passes the gate once, it banks and is free forever.

Wire: `MembraneV2(author_fn=make_refraction_author(gen), author_tries=4)` — each of the N tries uses the
NEXT refraction variant (different prompt), not a resample. HONEST BOUND: refraction rescues PROMPT-FRAGILE
failures (the model CAN do it, the phrasing trips it), NOT CAPABILITY-GAP ones (it genuinely can't) — expect
a partial lift, not 100%. The AST-error-injection variant (show wrong code + "don't do this") is OMITTED by
default: priming the wrong pattern is a pink-elephant risk; keep it behind an explicit A/B.

    python -m v5.runtime.algo_grr_refract --selftest   # no-GPU: refraction banks prompt-fragile atoms; Best-of-N can't
    # real 3B (user box): python -m v5.runtime.algo_grr_scale --run --K 100 --T 400 --lm Qwen/Qwen2.5-3B-Instruct \
    #                        --author-tries 4   (with a refraction author wired — see run_refract)
"""
from __future__ import annotations

import argparse
import hashlib
import random
import re
import sys


def refract_specs(name: str, desc: str, k: int = 4) -> list[str]:
    """k DETERMINISTIC prompt variants of the same functional spec, each with a distinct surface form so
    the token path enters a different basin. Variant 0 = canonical; 1..k-1 = refracted."""
    d = desc or name.replace("_", " ")
    return [
        # 0. canonical
        f"Write a single self-contained Python function named `{name}` taking one integer `n` and "
        f"returning {d}. Return ONLY the function definition.",
        # 1. IDENTIFIER TRANSMUTATION: different variable + verbose framing
        f"Define a function `{name}` of one integer argument `value`. It must compute and return "
        f"{d.replace(' n ', ' value ')}. Output ONLY the def, keeping the exact name `{name}`.",
        # 2. PARADIGM INVERSION: force a single closed-form expression (breaks a stuck loop template)
        f"Implement `{name}(n)` as a SINGLE `return` expression (closed form, avoid loops/branches) that "
        f"yields {d}. Return only the def named exactly `{name}`.",
        # 3. PARADIGM INVERSION: restate-then-return (breaks a stuck one-liner template)
        f"Task: {d}. First restate that requirement in a comment, then write `{name}(n)` that returns it. "
        f"Output only the final Python def named exactly `{name}`.",
    ][:k]


def make_refraction_author(gen):
    """author_fn(name, task) that uses the NEXT refraction variant on each call for a given atom (tracked
    per entry+name). Combined with MembraneV2 author_tries=N -> N DECORRELATED prompts, not N resamples."""
    from v5.runtime.algo_grr_membrane import _extract_code
    from v5.runtime.algo_lm_author import repair_code
    state: dict = {}

    def author(name, task):
        key = (task.get("entry"), name)
        i = state.get(key, 0)
        state[key] = i + 1
        variants = refract_specs(name, task.get("text", ""))
        prompt = variants[i % len(variants)]
        return repair_code(_extract_code(gen([prompt])[0]), name)
    return author


# ── no-GPU selftest: a PROMPT-FRAGILE author (systematic basin) — refraction escapes, Best-of-N can't ──
def _polys(K, seed=0):
    rng = random.Random(seed)
    pool, used = {}, set()
    while len(pool) < K:
        a, b, c, m = rng.randint(1, 6), rng.randint(0, 8), rng.randint(0, 8), rng.choice([7, 9, 11, 13, 17])
        if (a, b, c, m) in used:
            continue
        used.add((a, b, c, m))
        n = len(pool)
        pool[f"g{n}"] = (f"def g{n}(n):\n    return ({a} * n * n + {b} * n + {c}) % {m}\n",
                         (lambda a, b, c, m: (lambda x: (a * x * x + b * x + c) % m))(a, b, c, m),
                         f"the value of {a} n squared plus {b} n plus {c} modulo {m}")
    return pool


def _fragile_gen(pool, hard, p_escape=0.5):
    """Systematic-basin author: on a HARD atom it returns WRONG code for the CANONICAL phrasing every time
    (the attractor); a REFRACTED phrasing escapes with prob p_escape (deterministic per prompt). Easy atoms
    always succeed. Models the measured failure mode: resampling the same prompt never escapes; a different
    surface sometimes does."""
    def gen(prompts):
        out = []
        for p in prompts:
            m = re.search(r"`(g\d+)`", p)
            name = m.group(1) if m else None
            src, _orc, _d = pool.get(name, ("", None, ""))
            if name in hard:
                refracted = ("`value`" in p) or ("SINGLE `return`" in p) or ("restate" in p)
                h = int(hashlib.md5(p.encode()).hexdigest(), 16) % 1000 / 1000.0
                out.append(src if (refracted and h < p_escape) else f"def {name}(n):\n    return 0\n")
            else:
                out.append(src)
        return out
    return gen


def _passes(src, name, orc) -> bool:
    try:
        ns: dict = {}
        exec(compile(src, "<a>", "exec"), ns)  # noqa: S102
        fn = ns.get(name)
        return callable(fn) and all(fn(x) == orc(x) for x in (2, 3, 4, 5, 6, 7))
    except Exception:  # noqa: BLE001
        return False


def selftest() -> bool:
    print("algo_grr_refract --selftest: Spec Refraction vs Best-of-N on a SYSTEMATIC (prompt-fragile) author\n")
    K = 60
    pool = _polys(K, seed=0)
    names = list(pool)
    hard = set(names[:int(K * 0.4)])                 # 40% of atoms sit in a bad canonical basin
    gen = _fragile_gen(pool, hard, p_escape=0.5)
    tries = 4

    def bank_count(use_refraction):
        banked = 0
        for name in hard:
            _src, orc, desc = pool[name]
            variants = refract_specs(name, desc)
            ok = False
            for i in range(tries):
                prompt = variants[i % len(variants)] if use_refraction else variants[0]   # refract vs resample-canonical
                if _passes(gen([prompt])[0], name, orc):
                    ok = True
                    break
            banked += int(ok)
        return banked

    bon = bank_count(False)                          # Best-of-N: 4 resamples of the CANONICAL prompt
    ref = bank_count(True)                           # Refraction: 4 DIFFERENT (decorrelated) prompts
    n = len(hard)
    print(f"  hard atoms (bad canonical basin): {n}")
    print(f"  Best-of-{tries} (resample canonical) banked : {bon}/{n}")
    print(f"  Spec Refraction ({tries} variants)   banked : {ref}/{n}")
    ok = bon <= 1 and ref >= 0.5 * n and ref > bon + 5
    print(f"\n  => Best-of-N resamples the SAME broken basin -> ~0 rescued ({bon}/{n}). Refraction shifts the")
    print(f"     surface -> escapes to a different basin -> rescues {ref}/{n} PROMPT-FRAGILE atoms at the same")
    print(f"     call budget. (Capability-gap atoms remain out of reach — an honest partial lift.)")
    print(f"\n  ALGO_GRR_REFRACT SELFTEST -> {'PASS' if ok else 'FAIL'}")
    return ok


def main():
    ap = argparse.ArgumentParser(description="Polymorphic Spec Refraction (authoring-tax fix)")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        sys.exit(0 if selftest() else 1)
    ap.print_help()


if __name__ == "__main__":
    main()
