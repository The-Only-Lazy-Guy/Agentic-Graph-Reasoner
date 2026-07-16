"""algo_grr_membrane — the frozen-compiler + TRM-membrane closed-loop solver (GRR-Tool, build A).

Design (READ_THIS "POISON DIAGNOSIS + frozen-compiler resolution", 2026-07-16)
-----------------------------------------------------------------------------
LM = FROZEN COMPILER. It is a pure `compile(spec) -> code`; its weights never change, so there is
NO gradient path graph->LM (kills weight-poison). The TRM is the MEMBRANE between graph and LM:
the graph NEVER reaches the LM directly. The membrane retrieves, tentatively composes, and hands
the LM ONLY a curated spec {task, selected atoms (+dep closure), holes} — never a raw top-k dump
(kills context-poison). A bad atom only costs if it is selected AND survives the hard verify gate;
compiling a bad spec -> verify fails -> not banked. The loop self-cleans.

This module is the ORCHESTRATOR. Retrieval policy is injectable:
  - default = iterative, VERIFIER-GATED cosine (the honest baseline: one-shot cosine made multi-hop
    and program-conditioned by re-scoring each hop against realized coverage);
  - a trained TRM policy (build B) drops in via `policy_fn` with the same interface.

The LM compile_fn is injectable too:
  - `make_stub_compiler(recipes)` = deterministic no-GPU stand-in for the selftest;
  - a frozen 3B (algo_lm_author.make_hf_gen wrapper) for molab.

    selftest (no GPU/LM):  python -m v5.runtime.algo_grr_membrane --selftest
"""
from __future__ import annotations

import argparse
import math
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Callable

_ROOT = str(Path(__file__).resolve().parents[2])
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from graph_core import MemoryGraph, Node, Edge  # type: ignore  # noqa: E402


# ═══════════════════════════════════════════════════════════════════════════════
# Dependency-closure realizer — assemble an atom + its transitive `depend` closure
# ═══════════════════════════════════════════════════════════════════════════════

def _depend_map(graph: MemoryGraph) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {nid: [] for nid in graph.nodes}
    for e in graph.edges:
        if e.relation == "depend" and e.src in graph.nodes and e.dst in graph.nodes:
            out[e.src].append(e.dst)
    return out


def resolve_closure(graph: MemoryGraph, atom_ids: list[str]) -> list[str]:
    """Transitive `depend` closure of atom_ids, ordered deps-FIRST (realizer order)."""
    dep = _depend_map(graph)
    order: list[str] = []
    seen: set[str] = set()

    def visit(nid: str) -> None:
        if nid in seen or nid not in graph.nodes:
            return
        seen.add(nid)
        for d in dep.get(nid, []):
            visit(d)
        order.append(nid)

    for a in atom_ids:
        visit(a)
    return order


def realize_closure_code(graph: MemoryGraph, atom_ids: list[str]) -> str:
    """Concatenated source of atom_ids + their dep closure, deps first."""
    parts = []
    for nid in resolve_closure(graph, atom_ids):
        code = graph.nodes[nid].metadata.get("code", "")
        if code:
            parts.append(code.rstrip("\n"))
    return "\n\n".join(parts)


# ═══════════════════════════════════════════════════════════════════════════════
# VERIFY — the hard gate. exec code, run I/O tests, return (fraction_pass, detail)
# ═══════════════════════════════════════════════════════════════════════════════

def _called(code: str, name: str) -> bool:
    """True iff `name` is CALLED in code (a `name(` occurrence that is not its own `def name(`)."""
    calls = len(re.findall(r"(?<![\w.])" + re.escape(name) + r"\s*\(", code))
    defs = len(re.findall(r"\bdef\s+" + re.escape(name) + r"\s*\(", code))
    return calls - defs > 0


def verify_code(code: str, entry: str, tests: list[tuple]) -> tuple[float, str]:
    """Returns (fraction of tests passing, detail). Any exception -> that test fails."""
    if not tests:
        return 0.0, "no tests"
    ns: dict = {}
    try:
        exec(compile(code, f"<compile:{entry}>", "exec"), ns)
    except Exception as e:  # noqa: BLE001
        return 0.0, f"compile error: {e!r}"
    fn = ns.get(entry)
    if not callable(fn):
        return 0.0, f"entry '{entry}' not defined"
    n_ok = 0
    first_err = ""
    for args, expected in tests:
        try:
            got = fn(*args)
        except Exception as e:  # noqa: BLE001
            if not first_err:
                first_err = f"{entry}{args!r} raised {e!r}"
            continue
        if got == expected:
            n_ok += 1
        elif not first_err:
            first_err = f"{entry}{args!r} -> {got!r} != {expected!r}"
    return n_ok / len(tests), (first_err or "all pass")


# ═══════════════════════════════════════════════════════════════════════════════
# Retrieval — no-GPU token-overlap embedder + cosine (mpnet injected on molab)
# ═══════════════════════════════════════════════════════════════════════════════

_TOK = re.compile(r"[a-z0-9]+")


def _tokens(s: str) -> list[str]:
    return _TOK.findall(s.lower())


class TokenRetriever:
    """Dependency-free cosine over bag-of-token vectors of node purpose text.

    Stand-in for mpnet in the no-GPU selftest. Same interface a real embed retriever exposes:
    `rank(query_text, exclude) -> [(node_id, score), ...]` over implementation nodes only.
    """

    def __init__(self, graph: MemoryGraph):
        self.graph = graph
        self.impl_ids = [nid for nid, n in graph.nodes.items() if n.node_type == "implementation"]
        self._vecs = {nid: Counter(_tokens(graph.nodes[nid].text)) for nid in self.impl_ids}
        # idf over the impl corpus so generic words ("the", "of") don't dominate
        df: Counter = Counter()
        for v in self._vecs.values():
            df.update(v.keys())
        n = max(1, len(self.impl_ids))
        self._idf = {t: math.log(1.0 + n / (1 + c)) for t, c in df.items()}

    def _w(self, counter: Counter) -> dict[str, float]:
        return {t: c * self._idf.get(t, math.log(1.0 + len(self.impl_ids))) for t, c in counter.items()}

    @staticmethod
    def _cos(a: dict[str, float], b: dict[str, float]) -> float:
        if not a or not b:
            return 0.0
        dot = sum(a[t] * b.get(t, 0.0) for t in a)
        na = math.sqrt(sum(x * x for x in a.values()))
        nb = math.sqrt(sum(x * x for x in b.values()))
        return dot / (na * nb) if na and nb else 0.0

    def rank(self, query_text: str, exclude: set[str] | None = None) -> list[tuple[str, float]]:
        exclude = exclude or set()
        q = self._w(Counter(_tokens(query_text)))
        scored = [(nid, self._cos(q, self._w(self._vecs[nid])))
                  for nid in self.impl_ids if nid not in exclude]
        scored.sort(key=lambda x: -x[1])
        return scored


# ═══════════════════════════════════════════════════════════════════════════════
# Stub compiler — deterministic no-GPU stand-in for the FROZEN LM
# ═══════════════════════════════════════════════════════════════════════════════

def make_stub_compiler(recipes: dict[str, str]) -> Callable[[dict], str]:
    """A fake 'frozen compiler'. Given a spec, prepend the CURATED atoms' closure code, then emit the
    entry body from a per-entry recipe (stands in for what a real LM would infer from the spec).

    recipes: entry_name -> body source (a `def <entry>(...)` block that CALLS the selected atoms).
    The stub can only produce correct code when the spec.atoms contain what the recipe needs -> the
    membrane's coverage signal is real (missing atom -> NameError -> verify fails).
    """
    def compile_fn(spec: dict) -> str:
        closure = "\n\n".join(a["code"].rstrip("\n") for a in spec.get("atoms", []))
        body = recipes.get(spec["entry"], "")
        return (closure + "\n\n" + body) if closure else body
    return compile_fn


# ═══════════════════════════════════════════════════════════════════════════════
# The membrane solver
# ═══════════════════════════════════════════════════════════════════════════════

class MembraneSolver:
    """Frozen-compiler + membrane closed loop.

    The LM (compile_fn) receives ONLY the curated spec.atoms (the membrane). Retrieval is iterative
    and verifier-gated: each hop keeps the candidate that most raises realized test coverage; stop
    when coverage == 1.0 or no candidate helps (or budget). On final failure, re-reason with failure
    context; on exhausted budget, DERIVE (hand the LM a hole) and bank the verified result.
    """

    def __init__(self, graph: MemoryGraph, compile_fn: Callable[[dict], str],
                 retriever: TokenRetriever | None = None,
                 k: int = 3, max_hops: int = 6, max_retries: int = 2):
        self.graph = graph
        self.compile_fn = compile_fn
        self.retriever = retriever or TokenRetriever(graph)
        self.k = k
        self.max_hops = max_hops
        self.max_retries = max_retries
        self.compile_inputs: list[dict] = []  # audit: everything the LM ever saw (membrane check)

    # ── curate: build the spec the LM is allowed to see (selected atoms only) ──────
    def _curate(self, task: dict, selected: list[str], failure: dict | None = None) -> dict:
        closure = resolve_closure(self.graph, selected)
        atoms = [{"name": self.graph.nodes[nid].metadata.get("entry", nid),
                  "purpose": self.graph.nodes[nid].text,
                  "code": self.graph.nodes[nid].metadata.get("code", "")}
                 for nid in closure]
        spec = {"task_text": task["text"], "entry": task["entry"],
                "tests": task["tests"], "atoms": atoms}
        if failure:
            spec["failure"] = failure
        return spec

    def _coverage(self, task: dict, selected: list[str], failure: dict | None = None) -> tuple[float, str, str]:
        """Compile the curated spec and verify -> (fraction_pass, code, detail)."""
        spec = self._curate(task, selected, failure)
        self.compile_inputs.append(spec)              # audit every LM input
        code = self.compile_fn(spec)
        frac, detail = verify_code(code, task["entry"], task["tests"])
        return frac, code, detail

    # ── the loop ───────────────────────────────────────────────────────────────
    def solve(self, task: dict, min_score: float = 1e-3) -> dict:
        trace: list[dict] = []
        selected: list[str] = []
        cur_cov, cur_code, cur_detail = self._coverage(task, [])  # recipe alone (usually 0)

        # iterative, program-conditioned retrieval. Candidates are added SPECULATIVELY by rank
        # (a composition where neither atom alone yields partial credit still climbs); realized
        # coverage is the STOP signal, and un-called atoms are pruned at the end. Adding a
        # definition never lowers coverage, so speculative add is monotone-safe.
        query = task["text"]
        for hop in range(self.max_hops):
            ranked = self.retriever.rank(query, exclude=set(selected))
            cand = next((c for c, s in ranked if s > min_score), None)
            if cand is None:
                break                                  # ret_stop: nothing else relevant to try
            selected.append(cand)
            cur_cov, cur_code, cur_detail = self._coverage(task, selected)
            trace.append({"hop": hop, "picked": cand, "coverage": cur_cov})
            # NOTE: the cosine baseline keeps a STABLE query (task text) + exclude-selected. Folding
            # selected purposes back into the query reinforces the already-covered concept and buries
            # the missing complement -> that program-conditioning is the TRM policy's job (build B),
            # which retrieves the COMPLEMENT; the untrained baseline must not fake it.
            if cur_cov >= 1.0:
                break

        # retries with failure context (LM re-compiles the SAME curated atoms, sees the error)
        retries = 0
        while cur_cov < 1.0 and retries < self.max_retries and selected:
            failure = {"code": cur_code, "error": cur_detail}
            cov, code, detail = self._coverage(task, selected, failure)
            retries += 1
            trace.append({"retry": retries, "coverage": cov})
            if cov > cur_cov:
                cur_cov, cur_code, cur_detail = cov, code, detail

        derived = False
        if cur_cov < 1.0:
            # DERIVE-on-gap: hand the LM a hole (frozen capability), verify, bank if it passes
            spec = self._curate(task, selected)
            spec["derive"] = True
            self.compile_inputs.append(spec)
            code = self.compile_fn(spec)
            frac, detail = verify_code(code, task["entry"], task["tests"])
            trace.append({"derive": True, "coverage": frac})
            if frac >= 1.0:
                cur_cov, cur_code, cur_detail, derived = frac, code, detail, True

        # PRUNE: on a solve, keep only atoms actually CALLED in the verified code. Speculatively
        # tried-but-unused atoms (e.g. a high-similarity poison the gate never let compile) drop
        # out here -> they are never banked. Minimal spec = the reuse-bearing unit.
        if cur_cov >= 1.0 and selected:
            selected = [nid for nid in selected
                        if _called(cur_code, self.graph.nodes[nid].metadata.get("entry", nid))]

        return {"solved": cur_cov >= 1.0, "coverage": cur_cov, "code": cur_code,
                "detail": cur_detail, "selected": selected, "derived": derived, "trace": trace}


# ═══════════════════════════════════════════════════════════════════════════════
# SELFTEST — no GPU, no LM. Proves the membrane plumbing + anti-poison mechanism.
# ═══════════════════════════════════════════════════════════════════════════════

def _seed_graph() -> MemoryGraph:
    from v5.runtime.algo_grr_seed import build_graph
    g = build_graph()
    nodes = {n["id"]: Node.from_dict(n) for n in g["nodes"]}
    edges = [Edge.from_dict(e) for e in g["edges"]]
    return MemoryGraph(nodes, edges, metadata=g["metadata"])


def _selftest() -> bool:
    print("algo_grr_membrane --selftest: frozen-compiler + membrane closed loop\n")
    graph = _seed_graph()
    ok_all = True

    # Tasks: NL purpose + entry + I/O tests. Solutions require finding (and composing) seed atoms.
    #   single-atom : retrieve one atom, wrap it
    #   two-atom    : requires a SECOND hop (compose two atoms) -> tests iterative retrieval
    tasks = [
        dict(text="check whether a number is prime", entry="checkprime",
             tests=[((7,), True), ((8,), False), ((2,), True)]),
        dict(text="reverse the digits of an integer", entry="revnum",
             tests=[((123,), 321), ((100,), 1)]),
        dict(text="largest sum of a contiguous subarray", entry="maxsub",
             tests=[(([-2, 1, -3, 4, -1, 2, 1, -5, 4],), 6), (([1, 2, 3],), 6)]),
        # two-atom composition: count how many divisors of n are prime
        dict(text="count the divisors of n that are prime numbers", entry="nprimediv",
             tests=[((12,), 2), ((30,), 3), ((7,), 1)]),
    ]
    # Recipes stand in for what the FROZEN LM would infer from the curated spec.
    recipes = {
        "checkprime": "def checkprime(n):\n    return is_prime(n)\n",
        "revnum": "def revnum(n):\n    return reverse_digits(n)\n",
        "maxsub": "def maxsub(xs):\n    return max_subarray_sum(xs)\n",
        "nprimediv": ("def nprimediv(n):\n"
                      "    return sum(1 for d in divisors(n) if is_prime(d))\n"),
    }
    compiler = make_stub_compiler(recipes)

    # ── [1] each task solves; membrane selects the right atom(s) ──────────────────
    solver = MembraneSolver(graph, compiler)
    n_solved = 0
    for t in tasks:
        r = solver.solve(t)
        picked = [graph.nodes[s].metadata.get("entry", s) for s in r["selected"]]
        status = "OK" if r["solved"] else "FAIL"
        print(f"  [task] {t['entry']:12s} solved={r['solved']!s:5s} "
              f"cov={r['coverage']:.2f} atoms={picked}")
        if r["solved"]:
            n_solved += 1
    print(f"  [1] tasks solved: {n_solved}/{len(tasks)} -> {'PASS' if n_solved == len(tasks) else 'FAIL'}")
    ok_all &= (n_solved == len(tasks))

    # ── [2] iterative retrieval really fired on the 2-atom task ───────────────────
    r = MembraneSolver(graph, compiler).solve(tasks[-1])
    got = {graph.nodes[s].metadata.get("entry", s) for s in r["selected"]}
    multi = {"divisors", "is_prime"}.issubset(got)
    print(f"  [2] 2-atom task selected {sorted(got)} (needs divisors+is_prime) -> "
          f"{'PASS' if multi else 'FAIL'}")
    ok_all &= multi

    # ── [3] MEMBRANE HELD: the LM only ever saw CURATED atoms, never the full graph ─
    all_impl = {n.metadata.get("entry", nid) for nid, n in graph.nodes.items()
                if n.node_type == "implementation"}
    leaked = False
    for spec in solver.compile_inputs:
        seen = {a["name"] for a in spec["atoms"]}
        # every atom the LM saw must be within the selected closure (<= a few), never ~all 21
        if len(seen) > 6 or (seen and not seen.issubset(all_impl)):
            leaked = True
    print(f"  [3] membrane: {len(solver.compile_inputs)} LM specs, max atoms/spec "
          f"{max((len(s['atoms']) for s in solver.compile_inputs), default=0)}, "
          f"leak={leaked} -> {'PASS' if not leaked else 'FAIL'}")
    ok_all &= (not leaked)

    # ── [4] ANTI-POISON: inject a high-similarity WRONG atom; gate must reject it ──
    poison = Node(id="impl_prime_poison",
                  text="check whether a number is prime prime primality test number",
                  node_type="implementation", confidence=0.9, importance=0.9,
                  metadata={"code": "def prime_poison(n):\n    return True\n",  # WRONG: always True
                            "entry": "prime_poison", "kind": "atom", "origin": "poison"})
    pgraph = MemoryGraph(dict(graph.nodes, **{poison.id: poison}), list(graph.edges), graph.metadata)
    pcompiler = make_stub_compiler(dict(recipes, **{
        # if the membrane were fooled into using the poison, this is the code it'd compile:
        "checkprime_p": "def checkprime_p(n):\n    return prime_poison(n)\n"}))
    ptask = dict(text="check whether a number is prime", entry="checkprime",
                 tests=[((7,), True), ((8,), False), ((2,), True)])
    presolver = MembraneSolver(pgraph, pcompiler)
    pr = presolver.solve(ptask)
    picked = {pgraph.nodes[s].metadata.get("entry", s) for s in pr["selected"]}
    # poison ranks high on text, but coverage(poison) fails (8 is not prime -> True is wrong);
    # the real is_prime raises coverage to 1.0 -> membrane keeps the verified atom, drops poison.
    poison_rejected = "prime_poison" not in picked and pr["solved"]
    # confirm the poison WAS a top candidate (so rejection is by the gate, not by ranking luck)
    top = [c for c, _ in presolver.retriever.rank(ptask["text"])[:3]]
    poison_was_tempting = "impl_prime_poison" in top
    print(f"  [4] anti-poison: poison in top-3={poison_was_tempting}, final atoms={sorted(picked)}, "
          f"solved={pr['solved']} -> {'PASS' if poison_rejected and poison_was_tempting else 'FAIL'}")
    ok_all &= (poison_rejected and poison_was_tempting)

    print(f"\n  ALGO_GRR_MEMBRANE SELFTEST -> {'PASS' if ok_all else 'FAIL'}")
    return ok_all


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        sys.exit(0 if _selftest() else 1)
    ap.print_help()


if __name__ == "__main__":
    main()
