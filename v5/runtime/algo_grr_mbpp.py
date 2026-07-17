"""algo_grr_mbpp — scale the frozen-compiler + membrane to MBPP+ (checklist #3).

The seed curriculum is small and related, so reuse fires easily. MBPP+ (378 real open-source tasks) is
the GENERALIZATION proof + the factoring/reuse STRESS TEST: starting from the clean 25-atom seed, does the
frozen 3B + membrane (a) solve real tasks, (b) FACTOR reusable helpers the fuzz-gate accepts, and (c) show
CROSS-TASK REUSE rising as the graph grows? Cross-task reuse on MBPP was 0 in every prior design — this is
the measurement that matters.

MBPP+ records: {text, code(reference), asserts[], plus_test, setup, name(entry), pipeline_shaped}. We grade
with the base `asserts` (fast, in-process); the reference already passed the full EvalPlus plus_test at prep.

    selftest (no GPU):  python -m v5.runtime.algo_grr_mbpp --selftest
    molab (real 3B):    python -m v5.runtime.algo_grr_mbpp --run --lm Qwen/Qwen2.5-3B-Instruct --limit 120 [--policy]
"""
from __future__ import annotations

import argparse
import ast
import json
import os
import subprocess
import sys
from pathlib import Path

_ROOT = str(Path(__file__).resolve().parents[2])
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from v5.runtime.algo_grr_poison_test import load_seed, bank_helper_granular  # noqa: E402
from v5.runtime.algo_grr_membrane import MembraneSolver, make_stub_compiler  # noqa: E402

CORPUS = "artifacts/mbpp_plus_prepped.jsonl"
COMMON = ("import math\nimport re\nimport itertools\nimport functools\n"
          "from collections import Counter, defaultdict, deque\n")


# ═══════════════════════════════════════════════════════════════════════════════
# Verify via the task's own asserts (the hard gate; the LM never writes these)
# ═══════════════════════════════════════════════════════════════════════════════

# Worker executed in a FRESH subprocess (see hard_verify_asserts). It reads {full, asserts} on stdin,
# runs the asserts, prints "FRAC <frac> <detail>". A C-level runaway (factorial(10**9), 'x'*10**9,
# catastrophic regex) is killed by the parent's subprocess timeout — which SIGALRM cannot do (signals
# only fire between Python bytecodes, never inside a C call).
_HARD_WORKER = r"""
import sys, json
if hasattr(sys, "set_int_max_str_digits"):
    sys.set_int_max_str_digits(200000)
d = json.load(sys.stdin)
ns = {}
try:
    exec(compile(d["full"], "<v>", "exec"), ns)
except Exception as e:
    print("FRAC 0.0 compile:" + repr(e)[:80]); sys.exit()
n, first = 0, ""
A = d["asserts"]
for a in A:
    try:
        exec(a, ns); n += 1
    except Exception as e:
        if not first: first = repr(e)[:80]
print("FRAC", n / max(1, len(A)), first or "ok")
"""


def hard_verify_asserts(full_code: str, asserts: list[str], timeout: float) -> tuple[float, str]:
    """Run the asserts in a FRESH python subprocess with a real kill-on-timeout. Isolated interpreter
    (no CUDA inherited), so a C-level runaway is hard-killed instead of hanging the corpus run."""
    try:
        r = subprocess.run([sys.executable, "-c", _HARD_WORKER],
                           input=json.dumps({"full": full_code, "asserts": asserts}),
                           capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return 0.0, "timeout (hard-killed subprocess)"
    line = next((ln for ln in reversed(r.stdout.splitlines()) if ln.startswith("FRAC ")), None)
    if not line:
        return 0.0, f"no result (crash: {r.stderr.strip()[:80]})"
    parts = line.split(" ", 2)
    try:
        return float(parts[1]), (parts[2] if len(parts) > 2 else "")
    except ValueError:
        return 0.0, "parse error"


def verify_asserts(code: str, asserts: list[str], setup: str = "",
                   timeout: float = 6.0) -> tuple[float, str]:
    """Grade LM code against the task's own asserts. LM code can infinite-loop OR do a C-level runaway.
    With V5_HARD_VERIFY=1 (auto-set on --lm runs) each verify runs in a hard-killable SUBPROCESS; else
    the fast in-process SIGALRM path (fine for trusted stub/selftest code)."""
    if not asserts:
        return 0.0, "no asserts"
    full = COMMON + (setup or "") + "\n" + code
    if os.environ.get("V5_HARD_VERIFY") == "1":
        return hard_verify_asserts(full, asserts, timeout)

    from v5.runtime.algo_grr_membrane import run_with_timeout

    def _run():
        ns: dict = {}
        exec(compile(full, "<mbpp>", "exec"), ns)
        n_ok, first = 0, ""
        for a in asserts:
            try:
                exec(a, ns)
                n_ok += 1
            except Exception as e:  # noqa: BLE001
                if not first:
                    first = f"{a[:70]} -> {e!r}"
        return n_ok / len(asserts), (first or "all pass")

    try:
        return run_with_timeout(_run, seconds=timeout, default=(0.0, "timeout (runaway code)"))
    except Exception as e:  # noqa: BLE001
        return 0.0, f"compile: {e!r}"


def _type_pool_from_asserts(asserts: list[str], entry: str) -> list:
    """Infer arg types for the fuzz-gate by literal-eval'ing the entry's call args in the asserts."""
    for a in asserts:
        try:
            tree = ast.parse(a)
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == entry:
                types = []
                for arg in node.args:
                    try:
                        t = type(ast.literal_eval(arg))
                        if t not in types:
                            types.append(t)
                    except Exception:  # noqa: BLE001
                        pass
                if types:
                    return types
    return [int]


def load_mbpp(path: str = CORPUS, limit: int | None = None) -> list[dict]:
    tasks = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        asserts, entry, setup = r["asserts"], r["name"], r.get("setup", "")

        def mk(asserts=asserts, setup=setup):
            return lambda code: verify_asserts(code, asserts, setup)

        tasks.append(dict(text=r["text"], entry=entry, examples=asserts, verify_fn=mk(),
                          type_pool=_type_pool_from_asserts(asserts, entry), tests=[],
                          reference=r["code"], pipeline_shaped=r.get("pipeline_shaped", False)))
        if limit and len(tasks) >= limit:
            break
    return tasks


# ═══════════════════════════════════════════════════════════════════════════════
# Corpus driver — membrane over the stream; measures cross-task reuse as the graph grows
# ═══════════════════════════════════════════════════════════════════════════════

def run_mbpp(graph, tasks: list[dict], compile_fn, policy_fn=None, retriever=None, chunk: int = 30,
             max_hops: int = 4, max_retries: int = 1, verbose: bool = True) -> dict:
    # trimmed hop/retry budget: most MBPP+ tasks don't match a seed atom, so an unbounded retrieval
    # loop just burns LM calls before deriving. Fewer hops -> faster corpus pass; the derive path
    # (LM writes from scratch) is where diverse tasks are solved anyway.
    # ONE cached retriever reused across the whole corpus (goal 2) — no per-task O(atoms) rebuild.
    if retriever is None:
        from v5.runtime.algo_grr_retrieval import CachedTokenRetriever
        retriever = CachedTokenRetriever(graph)
    seed_ids = {nid for nid in graph.nodes if graph.nodes[nid].node_type == "implementation"}
    solved = reuse = banked = derived_reuse = 0
    per = []
    for i, t in enumerate(tasks):
        solver = MembraneSolver(graph, compile_fn, retriever=retriever, policy_fn=policy_fn,
                                max_hops=max_hops, max_retries=max_retries)
        r = solver.solve(t)
        if verbose:
            n_lm = len(solver.compile_inputs)
            print(f"  [{i+1:4d}/{len(tasks)}] {t['entry']:40s} "
                  f"{'OK' if r['solved'] else 'FAIL':4s}  lm_calls={n_lm}", flush=True)
        if r["solved"]:
            solved += 1
            # cross-task reuse = graph atoms CALLED in the verified solution
            reuse += len(r["selected"])
            # of which, reuse of atoms DERIVED earlier in the run (the compounding signal)
            derived_reuse += sum(1 for s in r["selected"] if s not in seed_ids)
            banked += len(bank_helper_granular(graph, r["code"], t["entry"], type_pool=t["type_pool"]))
        if (i + 1) % chunk == 0 or i == len(tasks) - 1:
            per.append(dict(upto=i + 1, solved=solved, reuse=reuse, derived_reuse=derived_reuse,
                            banked=banked, graph=len(graph.nodes)))
            if verbose:
                p = per[-1]
                print(f"  [{p['upto']:4d}] solved={p['solved']:4d} reuse={p['reuse']:4d} "
                      f"derived_reuse={p['derived_reuse']:4d} banked={p['banked']:4d} graph={p['graph']}")
    return dict(n=len(tasks), solved=solved, reuse=reuse, derived_reuse=derived_reuse,
                banked=banked, graph_nodes=len(graph.nodes), per=per)


# ═══════════════════════════════════════════════════════════════════════════════
# SELFTEST — no GPU: loader + assert-verify + driver plumbing (stub = reference code)
# ═══════════════════════════════════════════════════════════════════════════════

def _selftest() -> bool:
    print("algo_grr_mbpp --selftest: MBPP+ loader + assert-verify + membrane driver (no GPU)\n")
    ok = True
    if not Path(CORPUS).exists():
        print(f"  corpus {CORPUS} missing -> SKIP (present on the box)")
        return True

    tasks = load_mbpp(limit=8)
    print(f"  [0] loaded {len(tasks)} tasks; entries: {[t['entry'] for t in tasks][:4]}...")

    # [1] each REFERENCE solution passes its own asserts (verify harness is correct)
    ref_pass = sum(1 for t in tasks if t["verify_fn"](t["reference"])[0] >= 1.0)
    print(f"  [1] reference passes own asserts: {ref_pass}/{len(tasks)} -> "
          f"{'PASS' if ref_pass == len(tasks) else 'FAIL'}")
    ok &= ref_pass == len(tasks)

    # [2] type inference from asserts (non-trivial)
    pools = {t["entry"]: [x.__name__ for x in t["type_pool"]] for t in tasks}
    non_default = sum(1 for t in tasks if t["type_pool"] != [int])
    print(f"  [2] type pools inferred (e.g. {list(pools.items())[:3]}), {non_default}/{len(tasks)} "
          f"non-default -> {'PASS' if non_default >= 1 else 'FAIL'}")
    ok &= non_default >= 1

    # [3] driver runs end-to-end with a stub compiler = the reference code -> all solve
    graph = load_seed()
    stub = make_stub_compiler({t["entry"]: t["reference"] for t in tasks})
    res = run_mbpp(graph, tasks, stub, chunk=4, verbose=False)
    print(f"  [3] driver (stub=reference): solved {res['solved']}/{res['n']}, banked {res['banked']}, "
          f"graph {res['graph_nodes']} -> {'PASS' if res['solved'] == res['n'] else 'FAIL'}")
    ok &= res["solved"] == res["n"]

    print(f"\n  ALGO_GRR_MBPP SELFTEST -> {'PASS' if ok else 'FAIL'}")
    return ok


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--lm", default="", help="frozen 3B (molab); omit = stub=reference smoke")
    ap.add_argument("--limit", type=int, default=120)
    ap.add_argument("--policy", action="store_true", help="use trained ComplementPolicy retrieval")
    ap.add_argument("--topo", action="store_true",
                    help="use topology-aware retrieval (depend-neighbour boost — structural, no net)")
    ap.add_argument("--corpus", default=CORPUS)
    a = ap.parse_args()

    if a.selftest:
        sys.exit(0 if _selftest() else 1)

    if a.run:
        tasks = load_mbpp(a.corpus, limit=a.limit)
        graph = load_seed()
        from v5.runtime.algo_grr_retrieval import CachedTokenRetriever
        retriever = CachedTokenRetriever(graph)          # one cached retriever for the whole corpus
        policy_fn = None
        if a.topo:                                       # structural, generalising (goal 1)
            from v5.runtime.algo_grr_retrieval import make_topology_policy
            policy_fn = make_topology_policy(graph)
            print("[topo] depend-neighbour boost -> membrane retrieval")
        elif a.policy:                                   # seed-trained net (OOD on MBPP+; see READ_THIS)
            from v5.runtime.algo_grr_policy import train_and_make_policy
            _m, policy_fn = train_and_make_policy(load_seed())
            print("[policy] ComplementPolicy on seed graph -> membrane retrieval")
        if a.lm:
            os.environ["V5_HARD_VERIFY"] = "1"            # subprocess verify -> real LM code can't hang the run
            from v5.runtime.algo_grr_membrane import make_frozen_gen, make_lm_compiler
            compile_fn = make_lm_compiler(make_frozen_gen(a.lm, temperature=0.6, max_new_tokens=320))
            print("[hard-verify] each verify runs in a hard-killable subprocess")
        else:
            compile_fn = make_stub_compiler({t["entry"]: t["reference"] for t in tasks})
            print("(stub = reference code; use --lm for the real generalization test)")
        print(f"MBPP+ run: {len(tasks)} tasks, lm={a.lm or 'stub'}, "
              f"retrieval={'topo' if a.topo else 'policy' if a.policy else 'cosine'}\n")
        res = run_mbpp(graph, tasks, compile_fn, policy_fn=policy_fn, retriever=retriever)
        print(f"\nSOLVED {res['solved']}/{res['n']} ({100*res['solved']//res['n']}%) | "
              f"banked {res['banked']} atoms | cross-task reuse {res['reuse']} "
              f"(of derived atoms: {res['derived_reuse']}) | graph {res['graph_nodes']} nodes")
        return

    ap.print_help()


if __name__ == "__main__":
    main()
