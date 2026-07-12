"""Decomposable MULTI-STEP tasks (#53) — the enabler for working memory (#50) + mid-reasoning temp
nodes (#51). Each task is an ORDERED chain of sub-goals; every step is independently verifiable and
step N+1 CONSUMES step N's output. So the natural solve is sequential: solve a sub-goal -> store its
result as a (temp) node -> the next sub-goal reuses it -> ... -> final.

Single-glue compose tasks gave a temp node nowhere to live (one function, done). Here the intermediate
result is explicit and checkable, so the loop can (a) verify progress step-by-step (anti-forget), and
(b) write/reuse scratch nodes within the task.

  primes_then_digitsum : keep_primes (is_prime)  ->  total_ds (digit_sum)
  dist_then_digitsum   : reachable_dists (dijkstra) -> sum_vals_ds (digit_sum)
  edits_then_max       : edit_dists (edit_distance) -> list_max (trivial)

  selftest (no model):  python -m v5.runtime.algo_multistep_tasks --selftest
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field

import numpy as np

from v5.runtime.algo_compose_tasks import (_digit_sum, _dijkstra, _edit_distance, _is_prime)
from v5.runtime.algo_graph_run import verify_asserts


@dataclass
class Step:
    name: str
    text: str
    tests: list[str]
    needs: set = field(default_factory=set)      # atoms this step calls


@dataclass
class MultiStepTask:
    name: str
    text: str
    steps: list[Step]
    final_name: str
    final_tests: list[str]

    def verify_step(self, i: int, code: str, deps: str = "") -> bool:
        return verify_asserts((deps + "\n" + code) if deps else code, self.steps[i].tests)

    def verify_final(self, code: str, deps: str = "") -> bool:
        return verify_asserts((deps + "\n" + code) if deps else code, self.final_tests)


# ── reference step + final code (the ORACLE — proves the chain is sound + genuinely sequential) ────
_REF_STEP = {
    "keep_primes": "def keep_primes(lst):\n    return [x for x in lst if is_prime(x)]",
    "total_ds": "def total_ds(lst):\n    return sum(digit_sum(x) for x in lst)",
    "reachable_dists": "def reachable_dists(n, edges, s):\n    return dijkstra(n, edges, s)",
    "sum_vals_ds": "def sum_vals_ds(d):\n    return sum(digit_sum(v) for v in d.values())",
    "edit_dists": "def edit_dists(pairs):\n    return [edit_distance(a, b) for a, b in pairs]",
    "list_max": "def list_max(lst):\n    return max(lst) if lst else 0",
}
_REF_FINAL = {
    "primes_then_digitsum": "def primes_then_digitsum(lst):\n    return total_ds(keep_primes(lst))",
    "dist_then_digitsum": "def dist_then_digitsum(n, edges, s):\n    return sum_vals_ds(reachable_dists(n, edges, s))",
    "edits_then_max": "def edits_then_max(pairs):\n    return list_max(edit_dists(pairs))",
}
# atoms each step needs (for seeding / audit)
_STEP_ATOM = {"keep_primes": {"is_prime"}, "total_ds": {"digit_sum"}, "reachable_dists": {"dijkstra"},
              "sum_vals_ds": {"digit_sum"}, "edit_dists": {"edit_distance"}, "list_max": set()}


# ── families: (final_name, kind, step_names, oracles) ─────────────────────────────
def _o_keep_primes(lst): return [x for x in lst if _is_prime(x)]
def _o_total_ds(lst): return sum(_digit_sum(x) for x in lst)
def _o_reachable_dists(n, e, s): return _dijkstra(n, e, s)
def _o_sum_vals_ds(d): return sum(_digit_sum(v) for v in d.values())
def _o_edit_dists(pairs): return [_edit_distance(a, b) for a, b in pairs]
def _o_list_max(lst): return max(lst) if lst else 0

_FAMILIES = {
    "primes_then_digitsum": dict(kind="list", steps=["keep_primes", "total_ds"],
                                 chain=lambda a: _o_total_ds(_o_keep_primes(a)),
                                 step_in=[lambda a: (a,), lambda a: (_o_keep_primes(a),)],
                                 step_o=[_o_keep_primes, _o_total_ds]),
    "dist_then_digitsum": dict(kind="graph", steps=["reachable_dists", "sum_vals_ds"],
                               chain=lambda n, e, s: _o_sum_vals_ds(_o_reachable_dists(n, e, s)),
                               step_in=[lambda n, e, s: (n, e, s), lambda n, e, s: (_o_reachable_dists(n, e, s),)],
                               step_o=[_o_reachable_dists, _o_sum_vals_ds]),
    "edits_then_max": dict(kind="pairs", steps=["edit_dists", "list_max"],
                           chain=lambda p: _o_list_max(_o_edit_dists(p)),
                           step_in=[lambda p: (p,), lambda p: (_o_edit_dists(p),)],
                           step_o=[_o_edit_dists, _o_list_max]),
}


def _rand_list(rng): return [int(rng.integers(2, 40)) for _ in range(int(rng.integers(3, 7)))]
def _rand_graph(rng):
    n = int(rng.integers(3, 7)); edges = []
    for u in range(n):
        for v in range(u + 1, n):
            if rng.random() < 0.55:
                edges.append((u, v, int(rng.integers(1, 10))))
    return (n, edges, 0)
def _rand_pairs(rng):
    def one(): return "".join(chr(97 + int(rng.integers(0, 5))) for _ in range(int(rng.integers(2, 7))))
    return ([[one(), one()] for _ in range(int(rng.integers(2, 4)))],)


def _sample(kind, rng):
    return {"list": lambda: (_rand_list(rng),), "graph": lambda: _rand_graph(rng),
            "pairs": lambda: _rand_pairs(rng)}[kind]()


def _mk_task(fam: str, args_list) -> MultiStepTask:
    """Build a MultiStepTask instance: per-step asserts (on random inputs, oracle-computed) + final."""
    spec = _FAMILIES[fam]
    steps = []
    for si, sname in enumerate(spec["steps"]):
        tests = []
        for args in args_list:
            sin = spec["step_in"][si](*args)
            sout = spec["step_o"][si](*sin)
            argstr = ", ".join(repr(x) for x in sin)
            tests.append(f"assert {sname}({argstr}) == {sout!r}")
        steps.append(Step(sname, f"step {si+1}: compute {sname}", tests, _STEP_ATOM[sname]))
    final_tests = []
    for args in args_list:
        argstr = ", ".join(repr(x) for x in args)
        final_tests.append(f"assert {fam}({argstr}) == {spec['chain'](*args)!r}")
    text = f"{fam}: solve in order — {' -> '.join(spec['steps'])}, then compose them."
    return MultiStepTask(fam, text, steps, fam, final_tests)


def gen_multistep_tasks(n: int = 60, seed: int = 0):
    rng = np.random.default_rng(seed)
    fams = list(_FAMILIES)
    tasks = []
    for i in range(n):
        fam = fams[i % len(fams)]
        args_list = [_sample(_FAMILIES[fam]["kind"], rng) for _ in range(2)]
        tasks.append(_mk_task(fam, args_list))
    return tasks


# a static set for the base run / selftest
MULTISTEP = [
    _mk_task("primes_then_digitsum", [([11, 4, 23],), ([8, 9, 10],)]),
    _mk_task("dist_then_digitsum", [(4, [(0, 1, 2), (0, 2, 5), (1, 2, 1), (2, 3, 3)], 0), (3, [(0, 1, 4)], 0)]),
    _mk_task("edits_then_max", [([["kitten", "sitting"], ["ab", "abc"]],), ([["a", "a"]],)]),
]


# ═══════════════════════════════════════════════════════════════════════════════
# SELFTEST (no model) — each step's ref passes ITS asserts; the final ref (composing the steps) passes;
# the final NEEDS the steps (fails without them) -> genuinely sequential + decomposable
# ═══════════════════════════════════════════════════════════════════════════════

def _selftest() -> bool:
    from v5.runtime.algo_compose_tasks import ALL_ATOMS
    print("algo_multistep_tasks --selftest: sound per-step + final chain + genuinely sequential\n")

    for t in MULTISTEP:
        # each step verifies with its atoms + the previous step's code available
        prior = ""
        for i, st in enumerate(t.steps):
            deps = "\n\n".join(ALL_ATOMS[a][1] for a in st.needs)
            assert t.verify_step(i, _REF_STEP[st.name], deps), f"{t.name} step {st.name} unsound"
            prior += "\n\n" + _REF_STEP[st.name]
        # final composes ALL step fns (+ their atoms); passes
        atoms = set().union(*(st.needs for st in t.steps))
        alldeps = "\n\n".join(ALL_ATOMS[a][1] for a in atoms) + prior
        assert t.verify_final(_REF_FINAL[t.name], alldeps), f"{t.name} final unsound"
        # final FAILS without the steps (it genuinely depends on the chain, not inlined)
        assert not t.verify_final(_REF_FINAL[t.name], ""), f"{t.name} final must NEED its steps"
    print(f"  [1] {len(MULTISTEP)} static tasks: every step sound + final composes the chain + FAILS "
          f"without the steps (sequential) -> PASS")

    gen = gen_multistep_tasks(30, seed=3)
    for t in gen:
        prior = ""
        for i, st in enumerate(t.steps):
            deps = "\n\n".join(ALL_ATOMS[a][1] for a in st.needs)
            assert t.verify_step(i, _REF_STEP[st.name], deps), f"gen {t.name} step {st.name} unsound"
            prior += "\n\n" + _REF_STEP[st.name]
        atoms = set().union(*(st.needs for st in t.steps))
        alldeps = "\n\n".join(ALL_ATOMS[a][1] for a in atoms) + prior
        assert t.verify_final(_REF_FINAL[t.name], alldeps), f"gen {t.name} final unsound"
    print(f"  [2] generator: {len(gen)} instances across {len(_FAMILIES)} families, all steps + finals "
          f"sound -> PASS")

    steps_total = sum(len(t.steps) for t in MULTISTEP)
    print(f"  [3] shape: {len(MULTISTEP)} tasks, {steps_total} verifiable sub-steps (each an "
          f"intermediate result a temp node can hold) -> PASS")

    print("\n  ALGO_MULTISTEP_TASKS SELFTEST -> PASS")
    return True


def main():
    ap = argparse.ArgumentParser(description="Decomposable multi-step tasks (ordered verifiable sub-goals).")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        sys.exit(0 if _selftest() else 1)
    ap.print_help()


if __name__ == "__main__":
    main()
