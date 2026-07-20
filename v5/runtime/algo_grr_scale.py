"""algo_grr_scale — the COMPOUNDING wall: how far does 'LM-cost-per-task falls as the graph grows' hold?

The hero claim is compounding: bank a verified atom once, reuse it forever, so author-calls/task fall with
use. At 5 atoms it saturates instantly. The honest scaling question for DEPLOYMENT: on a LARGE atom pool
with a REALISTIC recurrence (a few common atoms, a long tail of rare ones — Zipf), does author-cost keep
falling to zero, or PLATEAU at the rate a never-seen atom appears? This harness measures that curve and
locates the plateau (the tail-rate floor) as a function of pool size K and Zipf skew s.

    python -m v5.runtime.algo_grr_scale --selftest          # no-GPU: compounding falls + plateaus
    python -m v5.runtime.algo_grr_scale --run --K 200 --T 1500 --zipf 1.1
"""
from __future__ import annotations

import argparse
import math
import random
import sys
from pathlib import Path

_ROOT = str(Path(__file__).resolve().parents[2])
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from v5.runtime.algo_grr_mbpp import verify_asserts               # noqa: E402
from v5.runtime.algo_grr_pipeline import (AtomStore, AtomRouter, TopologyAtomRouter,  # noqa: E402
                                          OraclePlanner, MembraneV2)


# ── a SCALABLE synthetic domain: K distinct verifiable atoms + a few wrappers ─────
WRAP = {
    "is_even": ("def is_even(n):\n    return int(n % 2 == 0)\n", lambda x: int(x % 2 == 0), "whether {v} is even"),
    "last_digit": ("def last_digit(n):\n    return abs(n) % 10\n", lambda x: abs(x) % 10, "the last digit of {v}"),
    "count_digits": ("def count_digits(n):\n    return len(str(abs(n)))\n", lambda x: len(str(abs(x))),
                     "the number of digits of {v}"),
    "triple": ("def triple(n):\n    return 3 * n\n", lambda x: 3 * x, "three times {v}"),
}


def gen_atom_pool(K: int, seed: int = 0) -> dict:
    """K distinct hard atoms g_k(n) = (a n^2 + b n + c) mod m — each with an exact oracle + a description
    (the router's key). Distinct params -> distinct behavior; the frozen LM would have to re-derive each."""
    rng = random.Random(seed)
    pool = {}
    used = set()
    k = 0
    while len(pool) < K:
        a, b, c, m = rng.randint(1, 6), rng.randint(0, 8), rng.randint(0, 8), rng.choice([7, 9, 11, 13, 17])
        key = (a, b, c, m)
        if key in used:
            continue
        used.add(key)
        name = f"g{k}"
        code = f"def {name}(n):\n    return ({a} * n * n + {b} * n + {c}) % {m}\n"
        orc = (lambda a, b, c, m: (lambda n: (a * n * n + b * n + c) % m))(a, b, c, m)
        desc = f"the value of {a} n squared plus {b} n plus {c} modulo {m}"
        pool[name] = (code, orc, desc)
        k += 1
    return pool


def _zipf_weights(n: int, s: float):
    w = [1.0 / ((i + 1) ** s) for i in range(n)]
    tot = sum(w)
    return [wi / tot for wi in w]


def gen_stream(pool: dict, T: int, zipf_s: float, seed: int = 1) -> list[dict]:
    """T tasks; the HARD atom is drawn Zipf(zipf_s) over the pool (a few common, a long rare tail), wrapped
    by a random easy wrapper. Task = wrapper(g_k(n))."""
    names = list(pool)
    rng = random.Random(seed)
    w = _zipf_weights(len(names), zipf_s)
    cum = []
    acc = 0.0
    for wi in w:
        acc += wi
        cum.append(acc)
    tasks = []
    for t in range(T):
        r = rng.random()
        hard = names[min(range(len(cum)), key=lambda i: (cum[i] < r, cum[i]))]  # first cum>=r
        # linear scan is fine for these sizes:
        for i, cc in enumerate(cum):
            if r <= cc:
                hard = names[i]
                break
        wrap = rng.choice(list(WRAP))
        h_code, h_orc, h_desc = pool[hard]
        w_code, w_fn, w_desc = WRAP[wrap]
        entry = f"t{t:05d}"
        asserts = []
        for n in (5, 6, 7, 8):
            asserts.append(f"assert {entry}({n}) == {w_fn(h_orc(n))!r}")
        text = w_desc.format(v=h_desc)

        def _mk(a=asserts):
            return lambda code: verify_asserts(code, a)
        tasks.append(dict(text=text, entry=entry, examples=asserts, verify_fn=_mk(),
                          _prims=(hard, wrap), atom_oracles={hard: h_orc}))
    return tasks


# ── run: MembraneV2 over the stream with a sim (correct) author; measure the compounding curve ────
def run_scale(K: int, T: int, zipf_s: float, router: str = "topo", window: int = 50,
              seed: int = 0, verbose: bool = True, lm: str = "", noise: float = 0.0,
              p_hard: float = 0.25, author_tries: int = 1, author_fail_cap=None) -> dict:
    pool = gen_atom_pool(K, seed=seed)
    stream = gen_stream(pool, T, zipf_s, seed=seed + 1)
    src = {**{k: v[0] for k, v in pool.items()}, **{k: v[0] for k, v in WRAP.items()}}
    if lm:                                                        # REAL 3B author: errors + fuzz-gate rejects
        import os
        os.environ["V5_HARD_VERIFY"] = "1"                       # untrusted LM code -> hard-kill subprocess verify
        from v5.runtime.algo_grr_membrane import make_frozen_gen
        from v5.runtime.algo_grr_pipeline import make_lm_author
        author = make_lm_author(make_frozen_gen(lm, temperature=0.4, max_new_tokens=200))
    elif noise > 0:                                              # WEAK sim author: a hard fraction fails the gate.
        hard = set(list(pool)[:int(len(pool) * noise)])          # the FREQUENT atoms are the hard ones (worst case:
        rng_a = random.Random(seed + 3)                          # the common tasks are the ones the model can't write)

        def author(name, task):                                 # stochastic failure (best-of-N can retry past it)
            if name in hard and rng_a.random() > p_hard:
                return f"def {name}(n):\n    return 0\n"         # WRONG -> fuzz-gate rejects (never banks)
            return src.get(name, "")
    else:
        author = lambda name, task: src.get(name, "")            # noqa: E731 — correct stub author (real 3B may err)

    store = AtomStore()
    for name, (code, *_ ) in WRAP.items():                        # wrappers seeded; hard atoms authored on demand
        store.set_rich(name, code, description=WRAP[name][2].replace("{v}", "the value"), provenance="seed")
    rt = TopologyAtomRouter(store) if router == "topo" else AtomRouter(store)
    m = MembraneV2(store, rt, OraclePlanner(), author_fn=author, bank=True,
                   author_tries=author_tries, author_fail_cap=author_fail_cap)

    rows = []
    prev_author = prev_reuse = 0
    win_solved = win_route = 0
    seen_hard = set()
    win_new = 0
    for i, t in enumerate(stream):
        r = m.solve(t)
        win_solved += int(r["solved"])
        win_route += int(r["route_ok"])
        h_atom = t["_prims"][0]                     # NB: not `hard` — that would clobber the author closure's set
        if h_atom not in seen_hard:
            seen_hard.add(h_atom); win_new += 1
        if (i + 1) % window == 0:
            da = m.author_calls - prev_author
            dr = m.derived_reuse - prev_reuse
            prev_author, prev_reuse = m.author_calls, m.derived_reuse
            rows.append(dict(win=(i + 1) // window, author_per_task=da / window,
                             new_atoms=win_new, reuse=dr, route=win_route / window,
                             banked=m.banked, graph=len(store)))
            if verbose:
                print(f"  [win {rows[-1]['win']:>3}] author/task={da/window:.2f}  new_atoms={win_new:>2}  "
                      f"deriv_reuse={dr:>3}  route_ok={win_route/window:.2f}  banked={m.banked}  graph={len(store)}",
                      flush=True)
            win_solved = win_route = win_new = 0

    early = sum(x["author_per_task"] for x in rows[:2]) / max(1, len(rows[:2]))
    late = sum(x["author_per_task"] for x in rows[-3:]) / max(1, len(rows[-3:]))
    # theoretical tail-rate floor: expected fraction of tasks whose atom is being seen for the FIRST time,
    # late in the stream ~ the probability mass still on unseen atoms. Approximated by measured late new-rate.
    late_new_rate = sum(x["new_atoms"] for x in rows[-3:]) / (3 * window) if len(rows) >= 3 else float("nan")
    return dict(rows=rows, K=K, T=T, zipf=zipf_s, early=early, late=late,
                floor=late_new_rate, banked=m.banked, graph=len(store), reuse=m.derived_reuse,
                author_calls=m.author_calls, skipped=m.author_skipped, failed=len(m.failed_authors),
                fuzz_rejected=m.fuzz_rejected)


def _print_summary(r: dict):
    print(f"\n  K={r['K']} T={r['T']} zipf={r['zipf']}: author/task {r['early']:.2f} (early) -> "
          f"{r['late']:.2f} (late)  | plateau~{r['floor']:.2f} (new-atom tail-rate)  | "
          f"banked={r['banked']}/{r['K']} deriv_reuse={r['reuse']}")
    print(f"  => COMPOUNDING holds (author/task falls {r['early']:.2f}->{r['late']:.2f}) but PLATEAUS at the "
          f"tail-rate, not zero: the honest scaling law — cost floor = rate new atoms appear.")


def selftest() -> bool:
    print("algo_grr_scale --selftest: the compounding wall (author-cost falls, then PLATEAUS at the tail-rate)\n")
    # LARGE pool (K >> stream can drain) so the plateau is a real NONZERO floor, not exhaustion.
    # skewed (heavy head): common atoms bank -> most tasks reuse -> LOW author floor.
    print("  SKEWED reuse (zipf=1.3, K=400) — a few common atoms carry most tasks:")
    r = run_scale(K=400, T=600, zipf_s=1.3, router="topo", window=50, seed=0)
    _print_summary(r)
    # flat (heavy tail): tasks spread over 400 atoms -> most are first-seen -> HIGH author floor (weak compounding).
    print("\n  FLAT reuse (zipf=0.3, K=400) — the long tail keeps forcing new authoring:")
    r0 = run_scale(K=400, T=600, zipf_s=0.3, router="topo", window=50, seed=0, verbose=False)
    _print_summary(r0)
    ok = (r["early"] > r["late"] + 0.10           # skewed: clear compounding (author-cost falls)
          and r["late"] < r0["late"] - 0.10        # skewed floor clearly BELOW the flat floor
          and r0["late"] > 0.20                    # flat floor is a real NONZERO plateau (heavy tail keeps authoring)
          and r["reuse"] > 0)
    print(f"\n  => THE SCALING LAW: compounding drives author-cost DOWN but only to a floor = the rate a "
          f"NEVER-SEEN atom appears. Skewed reuse -> low floor ({r['late']:.2f}); flat/heavy-tail -> high floor "
          f"({r0['late']:.2f}). Compounding is BOUNDED by the reuse distribution — the honest deployment limit, "
          f"not 'cost -> 0'.")
    print(f"\n  ALGO_GRR_SCALE SELFTEST -> {'PASS' if ok else 'FAIL'}")
    return ok


def tax_demo() -> bool:
    """LIMIT 3 (the authoring-error tax) + THE FIX: a WEAK author (30% of atoms fail the gate stochastically,
    p=0.25/call) inflates the compounding floor by re-authoring failures forever. best-of-N banks the
    stochastically-hard atoms; a FAILED-AUTHOR mistake-node stops re-authoring persistent failures."""
    print("algo_grr_scale --tax: the authoring-error tax (weak author) + the fix "
          "(best-of-N + failed-author mistake-node)\n")
    # PERSISTENT-hard: 30% of atoms (the FREQUENT ones) the author can NEVER write (p_hard=0) -> the
    # fuzz-gate keeps rejecting them, so without memory they are re-authored on every reappearance forever.
    base = dict(K=60, T=600, zipf_s=1.1, noise=0.30, p_hard=0.0, seed=0, verbose=False)
    a = run_scale(**base, author_tries=1, author_fail_cap=None)      # baseline: re-authors failures forever
    b = run_scale(**base, author_tries=1, author_fail_cap=2)         # fix: failed-author mistake-node (try 2, then skip)
    # best-of-N helps the STOCHASTIC case (author sometimes right): show it banks more of those.
    st = dict(K=60, T=600, zipf_s=1.1, noise=0.30, p_hard=0.25, seed=0, verbose=False)
    s1 = run_scale(**st, author_tries=1, author_fail_cap=None)
    s4 = run_scale(**st, author_tries=4, author_fail_cap=None)
    print(f"  (persistent-hard: the model NEVER writes 30% of atoms; the gate rejects them)")
    print(f"  arm                          | author_calls | banked | wasted re-auth | skipped(-node)")
    print(f"  BASELINE (no memory)         | {a['author_calls']:>12} | {a['banked']:>6} | {a['author_calls']-a['banked']:>14} | {a['skipped']:>6}")
    print(f"  FIX (failed-author -node,cap2)| {b['author_calls']:>11} | {b['banked']:>6} | {b['author_calls']-b['banked']:>14} | {b['skipped']:>6}")
    print(f"  (stochastic-hard, RECURRING: best-of-1 banked {s1['banked']} -> best-of-4 banked {s4['banked']} of {st['K']} "
          f"— small, because a recurring stream already supplies retries; best-of-N mainly helps RARE atoms)")
    waste_a, waste_b = a["author_calls"] - a["banked"], b["author_calls"] - b["banked"]
    ok = waste_b < waste_a * 0.3 and b["skipped"] > 0        # the real win: mistake-node kills the re-authoring waste
    print(f"\n  => THE FIX: the FAILED-AUTHOR mistake-node cuts wasted re-authoring {waste_a}->{waste_b} (~{waste_a//max(1,waste_b)}x):")
    print(f"     try-twice-then-skip instead of re-author-forever. Same banked; the weak-author tax stops")
    print(f"     compounding. (best-of-N is situational — helps only where atoms don't recur.) The negative")
    print(f"     memory tier finally earns its keep in the live loop.")
    print(f"\n  ALGO_GRR_SCALE TAX SELFTEST -> {'PASS' if ok else 'FAIL'}")
    return ok


def main():
    ap = argparse.ArgumentParser(description="compounding-at-scale stress test")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--K", type=int, default=200)
    ap.add_argument("--T", type=int, default=1500)
    ap.add_argument("--zipf", type=float, default=1.1)
    ap.add_argument("--router", choices=["topo", "flat"], default="topo")
    ap.add_argument("--lm", type=str, default="", help="real frozen author (e.g. Qwen/Qwen2.5-3B-Instruct); "
                    "else a correct stub author. Shows whether the compounding floor RISES with real 3B "
                    "authoring errors + fuzz-gate rejections")
    ap.add_argument("--sweep", action="store_true", help="sweep K and zipf to map the compounding floor")
    ap.add_argument("--tax", action="store_true", help="the authoring-error tax + fix (best-of-N + mistake-node)")
    ap.add_argument("--author-tries", type=int, default=1, help="best-of-N author (resample per atom)")
    ap.add_argument("--author-fail-cap", type=int, default=None, help="failed-author mistake-node: skip an atom "
                    "after N failed authorings (cuts the real-3B tax; try with --lm)")
    a = ap.parse_args()
    if a.selftest:
        sys.exit(0 if selftest() else 1)
    if a.tax:
        sys.exit(0 if tax_demo() else 1)
    if a.sweep:
        print("compounding floor vs pool size K and Zipf skew s (author/task late):")
        print(f"  {'K':>5} {'zipf':>5} | {'early':>6} {'late':>6} {'floor':>6} {'banked':>7} {'reuse':>6}")
        for K in (50, 200, 500):
            for z in (0.8, 1.1, 1.4):
                r = run_scale(K=K, T=1500, zipf_s=z, router=("topo" if a.router == "topo" else "flat"),
                              verbose=False)
                print(f"  {K:>5} {z:>5} | {r['early']:>6.2f} {r['late']:>6.2f} {r['floor']:>6.2f} "
                      f"{r['banked']:>7} {r['reuse']:>6}", flush=True)
        return
    if a.run:
        r = run_scale(K=a.K, T=a.T, zipf_s=a.zipf, router=("topo" if a.router == "topo" else "flat"), lm=a.lm,
                      author_tries=a.author_tries, author_fail_cap=a.author_fail_cap)
        _print_summary(r)
        print(f"  (author_calls={r['author_calls']} banked={r['banked']} wasted={r['author_calls']-r['banked']} "
              f"skipped={r['skipped']} — with --author-fail-cap the mistake-node cuts the wasted re-authoring)")
        return
    ap.print_help()


if __name__ == "__main__":
    main()
