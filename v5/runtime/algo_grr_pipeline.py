"""algo_grr_pipeline — MembraneV2: the integrated pipeline (#1a, WAVE 0).

Turns the validated probes into ONE end-to-end system with three swappable interfaces, so the parallel
tracks (#2 planner, #3 GraphGPS router, #5 step-spec) build against stubs without blocking each other:

    AtomRouter.rank(task_text, atoms)  -> ranked atom names      (#3 improves it: GraphGPS)
    Planner.plan(task, ranked)         -> AtomProgram            (#2 trains it: TRMPlanDecoder)
    realize(AtomProgram, store)        -> code                   (deterministic — never the LM)
    MembraneV2.solve = rank -> plan -> realize -> ratify(LM) -> VERIFY -> bank

WAVE-0 stubs shipped here: token-overlap router + an ORACLE planner (reads the known wiring). Swap in the
trained planner / GPS router later behind the same interfaces (#1b). The reasoner emits STRUCTURE (atom
program); the deterministic realizer builds code (can't inline) — this is why the frozen-LM composition
ceiling (0.03 free-form) becomes 1.00.

    python -m v5.runtime.algo_grr_pipeline --selftest    # no-GPU: end-to-end on the compose corpus
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

_ROOT = str(Path(__file__).resolve().parents[2])
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from v5.runtime.algo_grr_compose import INNER, OUTER  # noqa: E402


# ── AtomProgram + deterministic realizer ────────────────────────────────────────
@dataclass
class AtomProgram:
    """What the reasoner emits: which atoms (dep order) + how they wire. wiring is a tree:
    ('call', atom_name, [arg, ...]) | 'n' (the input leaf). NEVER free-form code."""
    atoms: list
    wiring: object


def _render(w) -> str:
    if isinstance(w, str):
        return w                                   # leaf, e.g. 'n'
    _, name, args = w
    return f"{name}(" + ", ".join(_render(a) for a in args) + ")"


def realize(prog: AtomProgram, store: dict, entry: str) -> str:
    """Deterministic program -> code: prepend the atoms' verified source (deps first), emit the entry
    that wires them. The realizer CANNOT inline -> the atoms are load-bearing."""
    closure = "\n".join(store[a].rstrip("\n") for a in prog.atoms if a in store)
    return f"{closure}\n\ndef {entry}(n):\n    return {_render(prog.wiring)}\n"


# ── Interfaces (stubs here; #2/#3 replace behind them) ──────────────────────────
class AtomStore(dict):
    """name -> verified source. Seeded from the compose primitive pool (+ grows via banking)."""

    @classmethod
    def from_compose(cls):
        s = cls()
        for name, (code, *_ ) in {**INNER, **OUTER}.items():
            s[name] = code
        return s


class AtomRouter:
    """WAVE-0 stub: rank atoms by token overlap with the task text. #3 replaces with GraphGPS features
    (mpnet content + topology MPNN + LapPE/RWSE). Interface is `.rank(task_text) -> [name...]`."""

    def __init__(self, store: dict):
        self.store = store
        self._toks = {n: set(_tok(n)) | set(_tok(_desc(store[n]))) for n in store}

    def rank(self, task_text: str, k: int | None = None):
        q = set(_tok(task_text))
        scored = sorted(self.store, key=lambda n: len(q & self._toks[n]), reverse=True)
        return scored[:k] if k else scored


class OraclePlanner:
    """WAVE-0 stub: reads the KNOWN wiring (task['_prims'] = (inner, outer)) -> the ground-truth program.
    #2 replaces this with the trained TRMPlanDecoder that INFERS the program from the NL task."""

    def plan(self, task: dict, ranked=None) -> AtomProgram:
        inner, outer = task["_prims"]
        return AtomProgram(atoms=[inner, outer],
                           wiring=("call", outer, [("call", inner, ["n"])]))


class MembraneV2:
    """The orchestration: route -> plan -> realize -> LM ratifies glue -> verify -> bank. Frozen LM only
    RATIFIES (stub = identity, since realize already produced verified-atom code). Tracks reuse = the
    compounding signal."""

    def __init__(self, store, router, planner, ratify_fn=None):
        self.store, self.router, self.planner = store, router, planner
        self.ratify_fn = ratify_fn
        self.reuse = {}                            # atom -> times used across tasks

    def solve(self, task: dict) -> dict:
        ranked = self.router.rank(task["text"])
        prog = self.planner.plan(task, ranked)
        code = realize(prog, self.store, task["entry"])
        if self.ratify_fn:                         # real LM writes/ratifies the glue line
            code = self.ratify_fn(code, task, prog)
        ok = task["verify_fn"](code)[0] >= 1.0
        route_ok = all(a in ranked[:6] for a in prog.atoms)   # did the router surface the needed atoms?
        if ok:
            for a in prog.atoms:
                self.reuse[a] = self.reuse.get(a, 0) + 1
        return dict(solved=ok, code=code, program=prog, route_ok=route_ok)


# ── helpers ──────────────────────────────────────────────────────────────────
def _tok(s: str):
    import re
    return [t for t in re.split(r"[^a-z0-9]+", s.lower()) if len(t) > 2]


def _desc(code: str) -> str:
    return code.split("(", 1)[0].replace("def", " ")


# ── selftest (no GPU) ──────────────────────────────────────────────────────────
def _selftest() -> bool:
    print("algo_grr_pipeline --selftest: MembraneV2 end-to-end (route -> plan -> realize -> verify)\n")
    from v5.runtime.algo_grr_compose import gen_corpus
    store = AtomStore.from_compose()
    tasks = gen_corpus(40, seed=0)
    solver = MembraneV2(store, AtomRouter(store), OraclePlanner())

    solved = route_hits = 0
    for t in tasks:
        r = solver.solve(t)
        solved += r["solved"]
        route_hits += r["route_ok"]
    n = len(tasks)
    reuse_total = sum(solver.reuse.values())
    distinct = len(solver.reuse)
    print(f"  [1] end-to-end solved     : {solved}/{n} ({100*solved//n}%)  (route->plan->realize->verify)")
    print(f"  [2] router surfaced atoms : {route_hits}/{n} tasks had needed atoms in top-6")
    print(f"  [3] compounding (reuse)   : {reuse_total} atom-uses over {distinct} distinct prims "
          f"(avg {reuse_total/max(1,distinct):.1f} reuse/prim)")
    # [4] realize is deterministic + correct; a WRONG program must fail verify (realizer isn't a rubber stamp)
    t0 = tasks[0]
    bad = AtomProgram(atoms=list(t0["_prims"]), wiring=("call", t0["_prims"][0], ["n"]))  # drops the outer
    bad_code = realize(bad, store, t0["entry"])
    caught = t0["verify_fn"](bad_code)[0] < 1.0
    print(f"  [4] wrong program fails verify: {'PASS' if caught else 'FAIL'}")

    ok = solved >= 0.95 * n and route_hits >= 0.8 * n and caught
    print(f"\n  MembraneV2 skeleton: interfaces chain, realize is load-bearing, reuse compounds.")
    print(f"  Swap points: AtomRouter->GraphGPS (#3), OraclePlanner->TRMPlanDecoder (#2), ratify_fn->3B (#1b).")
    print(f"\n  ALGO_GRR_PIPELINE SELFTEST -> {'PASS' if ok else 'FAIL'}")
    return ok


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        sys.exit(0 if _selftest() else 1)
    ap.print_help()


if __name__ == "__main__":
    main()
