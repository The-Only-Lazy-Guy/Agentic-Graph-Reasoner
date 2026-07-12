"""GRR-1: the fuzz-generality store-gate — verification is the TICKET, not the promotion. An atom is
stored IFF it is:

  verified   passes the task's own asserts            (entry ticket)
  general    correct on N RANDOM inputs vs the oracle  (the killer filter — `if x==17: return 4` dies)
  compact    small AST / low MDL                       (prefer the shortest general form)
  novel      not a behavioral duplicate                (fingerprint on fixed probes)

Correctness alone lets benchmark-overfit garbage into permanent memory. This gate runs the candidate on
many random inputs against the family oracle; a solution that hard-codes the one benchmark answer agrees
on ~0% of random inputs and is rejected. refilter_harvest() re-checks the existing harvest to MEASURE
how much stored code was overfit.

  selftest (no model):  python -m v5.runtime.algo_quality --selftest
  refilter a harvest:   python -m v5.runtime.algo_quality --refilter --harvest artifacts/reason_harvest_hard.jsonl
"""
from __future__ import annotations

import argparse
import ast
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

from v5.runtime.algo_compose_tasks import ALL_ATOMS, _FAMILIES, _FAMILIES_HARD, _sample

FAMS = {**_FAMILIES, **_FAMILIES_HARD}


def _count_pass(full_code: str, asserts: list[str], timeout: float = 10.0) -> int:
    """Run all asserts in ONE subprocess, counting how many pass (agreement with the oracle)."""
    lines = [full_code, "__p = 0"]
    for a in asserts:
        lines += [f"try:\n    {a}\n    __p += 1\nexcept Exception:\n    pass"]
    lines.append("print('__PASS__', __p)")
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "f.py"; p.write_text("\n".join(lines), encoding="utf-8")
        try:
            r = subprocess.run([sys.executable, "-I", str(p)], cwd=td, capture_output=True,
                               text=True, encoding="utf-8", errors="replace", timeout=timeout)
        except subprocess.TimeoutExpired:
            return 0
    for tok in (r.stdout or "").split("\n"):
        if tok.startswith("__PASS__"):
            try:
                return int(tok.split()[1])
            except (IndexError, ValueError):
                return 0
    return 0


def fuzz(code: str, name: str, deps: str = "", n: int = 60, seed: int = 0):
    """(passed, total) over n RANDOM inputs of the family, compared to its oracle. general iff passed==total."""
    if name not in FAMS:
        return 0, 0
    kind, oracle = FAMS[name]
    rng = np.random.default_rng(seed)
    asserts = []
    for _ in range(n):
        args = _sample(kind, rng)
        try:
            out = oracle(*args)
        except Exception:
            continue
        argstr = ", ".join(repr(a) for a in args)
        asserts.append(f"assert {name}({argstr}) == {out!r}")
    if not asserts:
        return 0, 0
    full = (deps + "\n" + code) if deps else code
    return _count_pass(full, asserts), len(asserts)


def mdl_size(code: str) -> int:
    """Description length proxy = AST node count (prefer the shortest general form)."""
    try:
        return sum(1 for _ in ast.walk(ast.parse(code)))
    except SyntaxError:
        return 10 ** 6


def behavior_fp(code: str, name: str, deps: str = "", probes: int = 8) -> str:
    """Behavioral fingerprint = the outputs on a FIXED probe set (seed 12345). Two atoms with the same
    fingerprint are behavioral duplicates -> merge, don't double-store."""
    if name not in FAMS:
        return ""
    kind, _ = FAMS[name]
    rng = np.random.default_rng(12345)
    lines = [(deps + "\n" + code) if deps else code, "import json", "__o = []"]
    for _ in range(probes):
        args = _sample(kind, rng)
        argstr = ", ".join(repr(a) for a in args)
        lines.append(f"try:\n    __o.append({name}({argstr}))\nexcept Exception:\n    __o.append('ERR')")
    lines.append("print('__FP__', json.dumps(__o))")
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "fp.py"; p.write_text("\n".join(lines), encoding="utf-8")
        try:
            r = subprocess.run([sys.executable, "-I", str(p)], cwd=td, capture_output=True, text=True,
                               encoding="utf-8", errors="replace", timeout=10)
        except subprocess.TimeoutExpired:
            return ""
    import hashlib
    for ln in (r.stdout or "").splitlines():
        if ln.startswith("__FP__"):
            return hashlib.sha1(ln[6:].strip().encode()).hexdigest()[:16]
    return ""


def store_gate(code: str, name: str, deps: str = "", existing_fps=(), n: int = 60,
               max_mdl: int = 400) -> dict:
    """The gate. store IFF general (fuzz-correct) AND novel (new behavior) AND compact (under max_mdl)."""
    passed, total = fuzz(code, name, deps, n=n)
    agree = passed / max(1, total)
    general = total > 0 and passed == total
    mdl = mdl_size(code)
    fp = behavior_fp(code, name, deps)
    novel = bool(fp) and fp not in set(existing_fps)
    compact = mdl <= max_mdl
    reasons = []
    if not general:
        reasons.append(f"overfit (agrees {agree:.0%} of random inputs)")
    if not novel:
        reasons.append("behavioral duplicate")
    if not compact:
        reasons.append(f"too large (mdl {mdl}>{max_mdl})")
    return dict(store=general and novel and compact, general=general, agree=round(agree, 3),
                mdl=mdl, novel=novel, fp=fp, reasons=reasons)


def refilter_harvest(harvest_path: str, n: int = 60) -> dict:
    """Re-check every harvested solution through the fuzz gate — how much of what we STORED was overfit?"""
    import json
    rows = [json.loads(l) for l in Path(harvest_path).read_text(encoding="utf-8").splitlines() if l.strip()]
    overfit, general, unknown = [], 0, 0
    for r in rows:
        deps = "\n\n".join(ALL_ATOMS[a][1] for a in r.get("used", []) if a in ALL_ATOMS)
        passed, total = fuzz(r["code"], r["task"], deps, n=n)
        if total == 0:
            unknown += 1
        elif passed == total:
            general += 1
        else:
            overfit.append((r["task"], round(passed / total, 2)))
    print(f"refilter {harvest_path}: {len(rows)} stored -> {general} general, {len(overfit)} OVERFIT, "
          f"{unknown} unknown-family", flush=True)
    for name, ag in overfit[:12]:
        print(f"    OVERFIT {name}: agrees {ag:.0%} of random inputs", flush=True)
    return dict(total=len(rows), general=general, overfit=len(overfit))


# ═══════════════════════════════════════════════════════════════════════════════
# SELFTEST — real code: the general solution passes; the benchmark-overfit hardcode is REJECTED
# ═══════════════════════════════════════════════════════════════════════════════

def _selftest() -> bool:
    print("algo_quality --selftest: fuzz-generality gate keeps general code, rejects overfit hardcodes\n")

    name = "sum_digitsum_primes"
    deps = "\n\n".join(ALL_ATOMS[a][1] for a in ("is_prime", "digit_sum"))
    general = "def sum_digitsum_primes(lst):\n    return sum(digit_sum(x) for x in lst if is_prime(x))"
    # overfit: hard-codes the ONE benchmark answer -> passes that assert, fails random inputs
    overfit = ("def sum_digitsum_primes(lst):\n    if lst == [11, 4, 23]:\n        return 7\n    return 0")

    g = store_gate(general, name, deps, n=80)
    assert g["store"] and g["general"] and g["agree"] == 1.0, g
    print(f"  [1] general solution: agrees {g['agree']:.0%} of random inputs, mdl={g['mdl']} -> STORE -> PASS")

    o = store_gate(overfit, name, deps, n=80)
    assert not o["store"] and not o["general"] and o["agree"] < 0.5, o    # 100% general vs ~22% overfit
    print(f"  [2] benchmark-overfit hardcode: agrees {o['agree']:.0%} of random inputs (vs 100% general) "
          f"-> REJECT ({o['reasons'][0]}) -> PASS")

    # [3] novelty: the same behavior twice is a duplicate (fingerprint), not double-stored
    fp = behavior_fp(general, name, deps)
    dup = store_gate(general, name, deps, existing_fps=[fp], n=40)
    assert not dup["store"] and not dup["novel"], dup
    print(f"  [3] behavioral duplicate (fp {fp}): novel={dup['novel']} -> REJECT -> PASS")

    # [4] compactness: MDL prefers the shorter of two GENERAL forms
    bloated = ("def sum_digitsum_primes(lst):\n    total = 0\n    for x in lst:\n"
               "        if is_prime(x):\n            s = 0\n            for c in str(abs(x)):\n"
               "                s = s + int(c)\n            total = total + s\n    return total")
    assert mdl_size(bloated) > mdl_size(general), (mdl_size(bloated), mdl_size(general))
    print(f"  [4] MDL: general form mdl={mdl_size(general)} < inlined-bloated mdl={mdl_size(bloated)} "
          f"(prefer the compact general form) -> PASS")

    print("\n  ALGO_QUALITY SELFTEST -> PASS")
    return True


def main():
    ap = argparse.ArgumentParser(description="Fuzz-generality store-gate (quality beyond correctness).")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--refilter", action="store_true", help="re-check a harvest through the fuzz gate")
    ap.add_argument("--harvest", default="artifacts/reason_harvest_hard.jsonl")
    ap.add_argument("--n", type=int, default=60)
    a = ap.parse_args()
    if a.selftest:
        sys.exit(0 if _selftest() else 1)
    if a.refilter:
        refilter_harvest(a.harvest, n=a.n)
        return
    ap.print_help()


if __name__ == "__main__":
    main()
