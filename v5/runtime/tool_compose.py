"""Compositional Tool-Graph: the model induces VERIFIED primitive tools, then COMPOSES them into
a new algorithm by CALLING the stored ones — new knowledge built from atomic knowledge.

Builds on tool_memory (induce -> verify-by-execution -> diagnostic-refine). Here the memory is a
GRAPH: nodes = verified tools, edges = "calls". The demonstration:

  1. induce primitive `nash_solver(R, C)`  -> the equilibrium cell            (a hard one)
  2. induce primitive `payoff(R, C, cell)` -> the payoffs at a cell           (an easy one)
  3. induce COMPOSITE `equilibrium_row_payoff(R, C)` that must CALL both stored primitives
     (payoff(R, C, nash_solver(R, C))[0]) — verified by executing it WITH the primitives in scope.

Why it matters: (3) is a NEW algorithm constructed from stored verified atoms, not retrieved and
not re-derived. The "which tools apply to this task" step is the retrieve-and-compose policy —
exactly the relevance-reasoning validated on compose_pool, now over ALGORITHMS not facts (the
natural home for the HRM-latent selector). learning_graph (learn-algorithms) + tool_memory
(impl-algorithms) unify under this graph substrate.

  selftest (no model):  python -m v5.runtime.tool_compose --selftest
  run (GPU):            V5_LM_TRUST_REMOTE_CODE=1 python -m v5.runtime.tool_compose --size 3
"""
from __future__ import annotations

import argparse
import ast
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from v5.runtime.reason_rl import make_game, pure_ne, EVAL_SEEDS, TRAIN_SEEDS, batch_generate
from v5.runtime.tool_memory import _extract_code, _log, _ne_explanation

_SENTINEL = "FNRESULT"


# ═══════════════════════════════════════════════════════════════════════════════
# GENERAL VERIFY-BY-EXECUTION  — call fn_name(*args) per case, with deps in scope
# ═══════════════════════════════════════════════════════════════════════════════

def verify_fn(code: str, fn_name: str, cases: list, deps_code: str = "",
              timeout: float = 8.0, max_fails: int = 6) -> tuple[float, list, str]:
    """Execute `fn_name(*args)` for each (args, expected) case, with `deps_code` (verified
    primitives) prepended so a COMPOSITE can call them. Returns (accuracy, fails, error).
    Comparison normalizes tuples<->lists so (0,1)==[0,1]."""
    if f"def {fn_name}" not in (code or ""):
        return 0.0, [], f"no {fn_name} defined"
    harness = "\n".join([
        deps_code, "", code, "",
        "def _norm(x):",
        "    return [_norm(v) for v in x] if isinstance(x, (list, tuple)) else x",
        "if True:",
        f"    _cases = {cases!r}",
        "    _ok, _fails = 0, []",
        "    for _args, _exp in _cases:",
        "        try:",
        f"            _a = {fn_name}(*_args)",
        "            _good = _norm(_a) == _norm(_exp)",
        "        except Exception as _e:",
        "            _good, _a = False, 'ERR:' + type(_e).__name__",
        "        _ok += 1 if _good else 0",
        f"        if not _good and len(_fails) < {max_fails}:",
        "            _fails.append((_args, _exp, _a if isinstance(_a,(list,tuple,int,float)) else str(_a)))",
        f"    print('{_SENTINEL}', _ok, len(_cases), repr(_fails))",
    ])
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "fn.py"
        p.write_text(harness, encoding="utf-8")
        try:
            proc = subprocess.run([sys.executable, "-I", str(p)], cwd=td, capture_output=True,
                                  text=True, encoding="utf-8", errors="replace", timeout=timeout)
        except subprocess.TimeoutExpired:
            return 0.0, [], "timeout"
    line = next((ln for ln in reversed((proc.stdout or "").splitlines())
                 if ln.startswith(_SENTINEL)), "")
    if not line:
        return 0.0, [], f"crash: {(proc.stderr or '')[-160:]}"
    try:
        _, ok, tot, fr = line.split(" ", 3)
        return int(ok) / max(1, int(tot)), ast.literal_eval(fr), ""
    except Exception as e:
        return 0.0, [], f"parse error: {e}"


# ═══════════════════════════════════════════════════════════════════════════════
# TOOL SPECS  — task text, case generator, and a per-tool failure diagnostic
# ═══════════════════════════════════════════════════════════════════════════════

def _nash_cases(games):
    return [([g["row_pay"], g["col_pay"]], list(g["ne"])) for g in games]


def _payoff_cases(games):
    # test payoff at an arbitrary cell (deterministic per game), not just the NE
    out = []
    for g in games:
        i = g["seed"] % g["size"]; j = (g["seed"] // 2) % g["size"]
        out.append(([g["row_pay"], g["col_pay"], [i, j]],
                    [g["row_pay"][i][j], g["col_pay"][i][j]]))
    return out


def _composite_cases(games):
    return [([g["row_pay"], g["col_pay"]], g["row_pay"][g["ne"][0]][g["ne"][1]]) for g in games]


def _generic_diag(args, exp, got):
    return f"inputs {args} -> your output {got}, correct output {exp}"


def _nash_diag(args, exp, got):
    R, C = args[0], args[1]
    return f"your output {got}; " + _ne_explanation(R, C, tuple(exp))


NASH_TASK = (
    "Write `nash_solver(R, C)` returning the pure-strategy Nash equilibrium cell (i, j) of a "
    "2-player game. R[i][j] is the ROW player's payoff, C[i][j] the COL player's. A cell is a NE "
    "iff R[i][j] is the max DOWN its COLUMN j AND C[i][j] is the max ACROSS its ROW i.")

PAYOFF_TASK = (
    "Write `payoff(R, C, cell)` where cell=(i, j). Return the tuple (R[i][j], C[i][j]) — the row "
    "player's and col player's payoffs at that cell.")

SPECS = {
    "nash_solver": dict(task=NASH_TASK, cases=_nash_cases, diag=_nash_diag),
    "payoff": dict(task=PAYOFF_TASK, cases=_payoff_cases, diag=_generic_diag),
}


# ═══════════════════════════════════════════════════════════════════════════════
# GENERAL INDUCE LOOP  (induce -> verify -> diagnostic-refine -> hill-climb)
# ═══════════════════════════════════════════════════════════════════════════════

def _write_prompt(task: str, fn_name: str, avail: list[tuple[str, str]],
                  prev_code: str | None, fails: list, prev_acc, diag) -> str:
    parts = [task]
    if avail:
        parts.append("\nVERIFIED tools already available (already defined — CALL them, do NOT "
                     "re-implement):")
        for sig, desc in avail:
            parts.append(f"  {sig}  -> {desc}")
    if prev_code:
        parts.append(f"\nYour current `{fn_name}` scored {prev_acc:.0%}:\n```python\n{prev_code}\n```")
        if fails:
            parts.append("It got these WRONG — here is the correct behaviour for each:")
            for args, exp, got in fails[:4]:
                parts.append("  " + diag(args, exp, got))
        parts.append(f"Write an IMPROVED `{fn_name}`.")
    else:
        parts.append(f"\nWrite `{fn_name}`.")
    parts.append("Output ONLY a Python code block with the function.")
    return "\n\n".join(parts)


def induce(model, tok, dev, fn_name: str, task: str, cases: list, diag, *, avail=None,
           deps_code: str = "", rounds: int = 8, samples: int = 4, chunk: int = 8,
           eval_cases: list | None = None, seed_code: str | None = None) -> tuple[str, float]:
    """Induce ONE tool (primitive or composite). avail = [(sig, desc)] of callable stored tools;
    deps_code = their source, prepended at verify-time so the tool can call them. seed_code = an
    initial draft to refine from round 0 (e.g. a composite that already CALLS the right tools) so
    the loop doesn't re-derive a non-calling version."""
    avail = avail or []
    best_code, best_acc = None, 0.0
    cur_code, cur_fails, cur_acc = None, [], 0.0
    if seed_code and f"def {fn_name}" in seed_code:
        a, f, _ = verify_fn(seed_code, fn_name, cases, deps_code)
        if a > 0:                                    # only seed from a draft that runs — a 0% seed
            cur_code, cur_fails, cur_acc = seed_code, f, a   # (crashes) just anchors the loop to junk
            best_code, best_acc = seed_code, a
            _log(f"\n── inducing {fn_name} (seeded from draft @ {a:.0%}) ──")
        else:
            _log(f"\n── inducing {fn_name} (draft seed @ 0%, starting fresh) ──")
    else:
        _log(f"\n── inducing {fn_name} ──")
    if best_acc >= 0.999:
        _log(f"  [{fn_name}] seed already correct, stop")
        return best_code, best_acc
    for rnd in range(rounds):
        prompt = _write_prompt(task, fn_name, avail, cur_code, cur_fails, cur_acc, diag)
        gens = batch_generate(model, tok, [prompt] * samples, dev, max_new=380,
                              sample=True, temperature=1.0, chunk=chunk)
        rb = None
        for gen in gens:
            code = _extract_code(gen)
            acc, fails, err = verify_fn(code, fn_name, cases, deps_code)
            if rb is None or acc > rb[0]:
                rb = (acc, code, fails, err)
        acc, code, fails, err = rb
        cur_code, cur_fails, cur_acc = code, fails, acc
        if acc > best_acc + 1e-9:
            best_code, best_acc = code, acc
        ev = ""
        if eval_cases and best_code:
            ea, _, _ = verify_fn(best_code, fn_name, eval_cases, deps_code)
            ev = f" | held-out {ea:.0%}"
        _log(f"  [{fn_name} r{rnd}] round-best {acc:.0%} (best {best_acc:.0%}){ev}"
             + (f"  [{err}]" if (acc == 0 and err) else ""))
        if best_acc >= 0.999:
            _log(f"  [{fn_name}] VERIFIED CORRECT, stop")
            break
    return (best_code or ""), best_acc      # "" (never None) so composition/deps assembly is safe


# ═══════════════════════════════════════════════════════════════════════════════
# THE RUN: induce primitives -> COMPOSE them into a new verified tool -> tool graph
# ═══════════════════════════════════════════════════════════════════════════════

def run_compose(model_name: str, size: int, rounds: int, samples: int, chunk: int,
                verify_n: int, eval_n: int):
    from transformers import AutoTokenizer
    from v5.lm_loader import load_frozen_lm

    trust = os.environ.get("V5_LM_TRUST_REMOTE_CODE", "0").lower() in ("1", "true", "yes")
    _log("  [compose] loading model...")
    model = load_frozen_lm(model_name); model.eval()
    tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=trust)
    dev = next(model.parameters()).device
    vg = [make_game(s, size=size) for s in list(TRAIN_SEEDS)[:verify_n]]
    eg = [make_game(s, size=size) for s in list(EVAL_SEEDS)[:eval_n]]
    _log(f"COMPOSE on {size}x{size} | model={model_name} | rounds={rounds} samples={samples}\n")

    graph: list[dict] = []          # nodes: {name, code, acc, sig, desc, calls}
    # ── Phase 1: induce the two primitives ──────────────────────────────────────
    prims = {}
    for name, sig, desc in [("nash_solver", "nash_solver(R, C)", "the pure Nash equilibrium cell (i, j)"),
                            ("payoff", "payoff(R, C, cell)", "the (row_payoff, col_payoff) at cell=(i,j)")]:
        spec = SPECS[name]
        code, acc = induce(model, tok, dev, name, spec["task"], spec["cases"](vg), spec["diag"],
                           rounds=rounds, samples=samples, chunk=chunk,
                           eval_cases=spec["cases"](eg))
        prims[name] = dict(code=code, acc=acc, sig=sig, desc=desc)
        graph.append(dict(name=name, code=code, acc=acc, sig=sig, desc=desc, calls=[]))
        if not code or acc < 0.999:
            _log(f"\n[!] primitive {name} only reached {acc:.0%} — composition needs it correct; "
                 f"continuing but the composite may fail.")

    # ── Phase 2: COMPOSE — write a tool that CALLS the two verified primitives ────
    deps = prims["nash_solver"]["code"] + "\n\n" + prims["payoff"]["code"]
    avail = [(prims["nash_solver"]["sig"], prims["nash_solver"]["desc"]),
             (prims["payoff"]["sig"], prims["payoff"]["desc"])]
    comp_task = ("Write `equilibrium_row_payoff(R, C)`: return the ROW player's payoff AT the "
                 "game's Nash equilibrium. Build it by CALLING the verified tools above — do not "
                 "re-implement the equilibrium or payoff logic yourself.")
    comp_code, comp_acc = induce(model, tok, dev, "equilibrium_row_payoff", comp_task,
                                 _composite_cases(vg), _generic_diag, avail=avail, deps_code=deps,
                                 rounds=rounds, samples=samples, chunk=chunk,
                                 eval_cases=_composite_cases(eg))
    # detect which primitives the composite actually CALLS (graph edges)
    calls = [n for n in prims if comp_code and (n + "(") in comp_code]
    graph.append(dict(name="equilibrium_row_payoff", code=comp_code, acc=comp_acc,
                      sig="equilibrium_row_payoff(R, C)", desc="row payoff at the equilibrium",
                      calls=calls))
    final_eval, _, _ = verify_fn(comp_code, "equilibrium_row_payoff", _composite_cases(eg), deps) \
        if comp_code else (0.0, [], "")

    _log("\n" + "=" * 60)
    _log(f"=== TOOL GRAPH ({len(graph)} nodes) ===")
    for n in graph:
        edge = f"  --calls--> {n['calls']}" if n.get("calls") else ""
        _log(f"  [{n['name']}] verified {n['acc']:.0%}{edge}")
    _log(f"\nCOMPOSITION: equilibrium_row_payoff held-out eval {final_eval:.0%}, "
         f"calls {calls or 'NOTHING (re-implemented — not composition!)'}")
    if comp_code:
        _log("\n  ── the composite the model wrote ──\n" + comp_code)
        Path("artifacts").mkdir(exist_ok=True)
        Path("artifacts/composed_tool.py").write_text(deps + "\n\n" + comp_code, encoding="utf-8")
        _log("\n  graph saved -> artifacts/composed_tool.py")
    return final_eval, bool(calls)


# ═══════════════════════════════════════════════════════════════════════════════
# SELFTEST (no model)
# ═══════════════════════════════════════════════════════════════════════════════

def _selftest() -> bool:
    print("tool_compose --selftest: general verify + composition-by-calling (no model)\n")
    games = [make_game(s, size=3) for s in range(20)]

    nash = ("def nash_solver(R, C):\n"
            "    m, n = len(R), len(R[0])\n"
            "    for i in range(m):\n"
            "        for j in range(n):\n"
            "            if R[i][j] == max(R[r][j] for r in range(m)) and \\\n"
            "               C[i][j] == max(C[i][c] for c in range(n)):\n"
            "                return (i, j)\n")
    pay = ("def payoff(R, C, cell):\n    i, j = cell\n    return (R[i][j], C[i][j])\n")
    a1, _, e1 = verify_fn(nash, "nash_solver", _nash_cases(games))
    a2, _, e2 = verify_fn(pay, "payoff", _payoff_cases(games))
    assert a1 == 1.0 and a2 == 1.0, f"primitives should verify 100%: nash {a1} {e1}, payoff {a2} {e2}"
    print(f"  [1] primitives verify: nash {a1:.0%}, payoff {a2:.0%} -> PASS")

    # a COMPOSITE that CALLS the primitives (with deps in scope) verifies correctly
    comp = ("def equilibrium_row_payoff(R, C):\n"
            "    return payoff(R, C, nash_solver(R, C))[0]\n")
    ac, _, ec = verify_fn(comp, "equilibrium_row_payoff", _composite_cases(games), nash + "\n" + pay)
    assert ac == 1.0, f"composite calling primitives should be 100%, got {ac:.0%} {ec}"
    print(f"  [2] composite CALLS primitives -> {ac:.0%} -> PASS")

    # without the deps in scope, the same composite CRASHES (proves it truly depends on them)
    ax, _, _ = verify_fn(comp, "equilibrium_row_payoff", _composite_cases(games), deps_code="")
    assert ax == 0.0, "composite must fail without its primitives -> it genuinely composes"
    print(f"  [3] composite needs its primitives (fails w/o them: {ax:.0%}) -> PASS")

    # a wrong primitive drops accuracy; diagnostic carries the correct behaviour
    bad = "def nash_solver(R, C):\n    return (0, 0)\n"
    ab, fb, _ = verify_fn(bad, "nash_solver", _nash_cases(games))
    assert ab < 0.5 and fb
    d = _nash_diag(fb[0][0], fb[0][1], fb[0][2])
    assert "COLUMN" in d and "ROW" in d
    print(f"  [4] wrong primitive {ab:.0%}, nash diagnostic carries column/row axis -> PASS")

    print("\n  TOOL_COMPOSE SELFTEST -> PASS")
    return True


def main():
    ap = argparse.ArgumentParser(description="Compositional tool-graph: induce primitives, compose them.")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--model", default="Qwen/Qwen2.5-3B")
    ap.add_argument("--size", type=int, default=3)
    ap.add_argument("--rounds", type=int, default=8)
    ap.add_argument("--samples", type=int, default=4)
    ap.add_argument("--chunk", type=int, default=8)
    ap.add_argument("--verify-n", type=int, default=40)
    ap.add_argument("--eval-n", type=int, default=40)
    a = ap.parse_args()
    if a.selftest:
        sys.exit(0 if _selftest() else 1)
    run_compose(a.model, a.size, a.rounds, a.samples, a.chunk, a.verify_n, a.eval_n)


if __name__ == "__main__":
    main()
