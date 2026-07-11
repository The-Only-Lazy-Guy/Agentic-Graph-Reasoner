"""Tool-Induction Memory: the model bootstraps a VERIFIED reasoning TOOL into memory.

The failure this solves: a small model can't reason a hard task instance-by-instance (3x3 Nash:
~18%, ~chance), and static worked-example memory just makes it ANCHOR to a frequent answer. But
it does NOT have to reason each instance. It can write a general METHOD once — an executable
solver — and the system VERIFIES that method by EXECUTION against ground truth, keeps it only if
it beats the current best, and hands the model its own FAILING CASES to refine it. Reuse is then
"execute the tool", not "re-reason".

So memory here is a growing, verified library of self-built tools (graph nodes with confidence =
measured accuracy, lineage = refine history). The loop:

  round r:
    1. the model WRITES / REVISES `solve_game(R, C)` (given the task + its own failing cases)
    2. VERIFY: execute the candidate on a held-out batch, score vs the oracle -> accuracy + fails
    3. KEEP if accuracy > best (write-back to the tool bank; new node, parent = prev best)
    4. the best tool is the method it has proven; eval accuracy tracks the bootstrap

Novelty vs static few-shot: (a) the artifact is a METHOD, not an instance to copy -> no anchoring;
(b) the verification gate discards hallucinated/wrong methods; (c) execution makes reuse exact;
(d) refinement is grounded in the model's own measured failures, not a template. A model that
reasons at chance can end at 100% once it has induced+verified the right tool.

  selftest (no model):  python -m v5.runtime.tool_memory --selftest
  run (GPU):            V5_LM_TRUST_REMOTE_CODE=1 python -m v5.runtime.tool_memory --rounds 8 --size 3
"""
from __future__ import annotations

import argparse
import ast
import os
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from v5.runtime.reason_rl import make_game, pure_ne, EVAL_SEEDS, TRAIN_SEEDS, batch_generate

_SENTINEL = "TOOLRESULT"


# ═══════════════════════════════════════════════════════════════════════════════
# VERIFY-BY-EXECUTION  — run the model's candidate tool against the oracle
# ═══════════════════════════════════════════════════════════════════════════════

def _extract_code(gen: str) -> str:
    """Pull the solver code out of a model generation (fenced block, or from `def solve_game`)."""
    m = re.search(r"```(?:python)?\s*(.*?)```", gen, flags=re.DOTALL)
    if m:
        return m.group(1).strip()
    i = gen.find("def solve_game")
    return gen[i:].strip() if i >= 0 else gen.strip()


def verify_solver(code: str, games: list[dict], timeout: float = 8.0,
                  max_fails: int = 6) -> tuple[float, list, str]:
    """Execute `solve_game(R, C)` on `games` in a subprocess; return (accuracy, failing_cases,
    error). failing_cases = [(R, C, expected, got)] for the model to learn from."""
    if "def solve_game" not in (code or ""):
        return 0.0, [], "no solve_game defined"
    payload = [(g["row_pay"], g["col_pay"], list(g["ne"])) for g in games]
    harness = "\n".join([
        code,
        "",
        "if True:",
        f"    _games = {payload!r}",
        "    _ok, _fails = 0, []",
        "    for _R, _C, _ne in _games:",
        "        try:",
        "            _a = solve_game(_R, _C)",
        "            _good = (list(_a) == _ne)",
        "        except Exception as _e:",
        "            _good, _a = False, 'ERR:' + type(_e).__name__",
        "        _ok += 1 if _good else 0",
        f"        if not _good and len(_fails) < {max_fails}:",
        "            _fails.append((_R, _C, _ne, _a if isinstance(_a,(list,tuple)) else str(_a)))",
        f"    print('{_SENTINEL}', _ok, len(_games), repr(_fails))",
    ])
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "tool.py"
        p.write_text(harness, encoding="utf-8")
        try:
            proc = subprocess.run([sys.executable, "-I", str(p)], cwd=td, capture_output=True,
                                  text=True, encoding="utf-8", errors="replace", timeout=timeout)
        except subprocess.TimeoutExpired:
            return 0.0, [], "timeout (likely an infinite loop in the tool)"
    line = next((ln for ln in reversed((proc.stdout or "").splitlines())
                 if ln.startswith(_SENTINEL)), "")
    if not line:
        return 0.0, [], f"crash: {(proc.stderr or '')[-160:]}"
    try:
        _, ok, tot, fails_repr = line.split(" ", 3)
        return int(ok) / max(1, int(tot)), ast.literal_eval(fails_repr), ""
    except Exception as e:
        return 0.0, [], f"parse error: {e}"


# ═══════════════════════════════════════════════════════════════════════════════
# THE MODEL WRITES / REVISES THE TOOL
# ═══════════════════════════════════════════════════════════════════════════════

TASK = ("You are writing one general Python function that solves a class of game-theory "
        "problems.\n"
        "Write `solve_game(R, C)` that returns the pure-strategy Nash equilibrium cell (i, j) of "
        "a 2-player game. R[i][j] is the ROW player's payoff, C[i][j] is the COL player's payoff "
        "(square matrices). A cell (i, j) is a Nash equilibrium iff R[i][j] is the maximum in its "
        "COLUMN j (row player won't switch rows) AND C[i][j] is the maximum in its ROW i (col "
        "player won't switch cols). Return the (i, j) tuple. The matrices may be 2x2 or 3x3.")


def write_tool_prompt(examples: list[dict], prev_code: str | None,
                      fails: list, prev_acc: float | None) -> str:
    parts = [TASK]
    if examples:
        parts.append("\nA few instances (with the correct equilibrium cell):")
        for g in examples[:3]:
            parts.append(f"R={g['row_pay']} C={g['col_pay']}  -> NE {g['ne']}")
    if prev_code:
        parts.append(f"\nYour current solver scored {prev_acc:.0%}:\n```python\n{prev_code}\n```")
        if fails:
            parts.append("It got these WRONG (fix them without breaking the rest):")
            for R, C, exp, got in fails[:5]:
                parts.append(f"  R={R} C={C}: your answer {got}, correct {tuple(exp)}")
        parts.append("Write an IMPROVED `solve_game(R, C)`.")
    else:
        parts.append("\nWrite `solve_game(R, C)`.")
    parts.append("Output ONLY a Python code block with the function.")
    return "\n\n".join(parts)


# ═══════════════════════════════════════════════════════════════════════════════
# THE LOOP: induce -> verify -> keep-if-better -> refine from own failures
# ═══════════════════════════════════════════════════════════════════════════════

def run_tool_induction(model_name: str, rounds: int, size: int, verify_n: int, eval_n: int,
                       samples: int, chunk: int):
    from transformers import AutoTokenizer
    from v5.lm_loader import load_frozen_lm

    trust = os.environ.get("V5_LM_TRUST_REMOTE_CODE", "0").lower() in ("1", "true", "yes")
    model = load_frozen_lm(model_name); model.eval()
    tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=trust)
    dev = next(model.parameters()).device

    verify_games = [make_game(s, size=size) for s in list(TRAIN_SEEDS)[:verify_n]]
    eval_games = [make_game(s, size=size) for s in list(EVAL_SEEDS)[:eval_n]]
    examples = [make_game(s, size=size) for s in list(TRAIN_SEEDS)[verify_n:verify_n + 3]]

    print(f"TOOL-INDUCTION on {size}x{size} Nash | model={model_name} | rounds={rounds} "
          f"samples/round={samples} verify={verify_n} eval={eval_n}\n", flush=True)

    tool_bank: list[dict] = []          # verified tools: {code, acc, round, parent}
    best_code, best_acc, best_fails = None, 0.0, []
    for rnd in range(rounds):
        prompt = write_tool_prompt(examples, best_code, best_fails, best_acc)
        # SAMPLE several candidate tools this round (the model proposes; verification selects)
        gens = batch_generate(model, tok, [prompt] * samples, dev, max_new=420,
                              sample=(samples > 1), temperature=0.8, chunk=chunk)
        round_best = None
        for gen in gens:
            code = _extract_code(gen)
            acc, fails, err = verify_solver(code, verify_games)
            if round_best is None or acc > round_best[0]:
                round_best = (acc, code, fails, err)
        acc, code, fails, err = round_best
        improved = acc > best_acc + 1e-9
        tag = "KEPT" if improved else "rejected"
        if improved:
            tool_bank.append(dict(code=code, acc=acc, round=rnd,
                                  parent=len(tool_bank) - 1 if tool_bank else None))
            best_code, best_acc, best_fails = code, acc, fails
        eval_acc, _, _ = verify_solver(best_code, eval_games) if best_code else (0.0, [], "")
        note = err if (acc == 0 and err) else ""
        print(f"[round {rnd}] candidate verify-acc {acc:.0%} -> {tag} (best {best_acc:.0%}) | "
              f"held-out eval {eval_acc:.0%} {note}", flush=True)

    print(f"\n=== DONE === bootstrapped {len(tool_bank)} verified tool(s); "
          f"best verify-acc {best_acc:.0%}", flush=True)
    if best_code:
        final_eval, _, _ = verify_solver(best_code, eval_games)
        print(f"  final held-out eval accuracy: {final_eval:.0%}\n", flush=True)
        print("  ── the method the model discovered ──\n" + best_code, flush=True)
        Path("artifacts").mkdir(exist_ok=True)
        Path("artifacts/induced_tool.py").write_text(best_code, encoding="utf-8")
        print("\n  tool saved -> artifacts/induced_tool.py", flush=True)
    return best_acc


# ═══════════════════════════════════════════════════════════════════════════════
# SELFTEST (no model)
# ═══════════════════════════════════════════════════════════════════════════════

def _selftest() -> bool:
    print("tool_memory --selftest: verify-by-execution (no model)\n")
    games = [make_game(s, size=2) for s in range(20)] + [make_game(s, size=3) for s in range(20)]

    correct = (
        "def solve_game(R, C):\n"
        "    m, n = len(R), len(R[0])\n"
        "    for i in range(m):\n"
        "        for j in range(n):\n"
        "            if R[i][j] >= max(R[r][j] for r in range(m)) and \\\n"
        "               C[i][j] >= max(C[i][c] for c in range(n)):\n"
        "                return (i, j)\n"
        "    return None\n")
    acc, fails, err = verify_solver(correct, games)
    assert acc == 1.0 and not fails, f"correct tool should score 100%, got {acc:.0%} err={err}"
    print(f"  [1] correct solver -> {acc:.0%}, 0 failures -> PASS")

    wrong = "def solve_game(R, C):\n    return (0, 0)\n"
    acc_w, fails_w, _ = verify_solver(wrong, games)
    assert acc_w < 0.5 and fails_w, f"constant tool should score low + expose fails, got {acc_w:.0%}"
    print(f"  [2] constant (0,0) solver -> {acc_w:.0%}, {len(fails_w)} failing cases exposed -> PASS")

    # robustness: crash / infinite-loop / no-def don't take the harness down
    assert verify_solver("def solve_game(R,C):\n    return R[9][9]\n", games[:3])[0] == 0.0
    assert verify_solver("def solve_game(R,C):\n    while True: pass\n", games[:2],
                         timeout=3.0)[2].startswith("timeout")
    assert verify_solver("print('hi')", games[:2])[2] == "no solve_game defined"
    print("  [3] crash / timeout / no-def handled gracefully -> PASS")

    # code extraction from a fenced block
    ex = _extract_code("Here:\n```python\ndef solve_game(R,C):\n    return (1,1)\n```\ndone")
    assert ex.startswith("def solve_game") and "return (1,1)" in ex
    print("  [4] code extraction from fenced block -> PASS")

    # the refine prompt carries the failing cases so the model can fix them
    p = write_tool_prompt(games[:3], wrong, fails_w, acc_w)
    assert "scored" in p and "WRONG" in p and "R=" in p
    print("  [5] refine prompt carries measured failures -> PASS")

    print("\n  TOOL_MEMORY SELFTEST -> PASS")
    return True


def main():
    ap = argparse.ArgumentParser(description="Tool-Induction Memory: model bootstraps a verified solver.")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--model", default="Qwen/Qwen2.5-3B")
    ap.add_argument("--rounds", type=int, default=8)
    ap.add_argument("--size", type=int, default=3, help="game size (3x3 is where instance-reasoning fails)")
    ap.add_argument("--verify-n", type=int, default=40, help="held-out games the tool is verified on")
    ap.add_argument("--eval-n", type=int, default=40, help="disjoint held-out games for the reported eval")
    ap.add_argument("--samples", type=int, default=4, help="candidate tools sampled per round")
    ap.add_argument("--chunk", type=int, default=8)
    a = ap.parse_args()
    if a.selftest:
        sys.exit(0 if _selftest() else 1)
    run_tool_induction(a.model, a.rounds, a.size, a.verify_n, a.eval_n, a.samples, a.chunk)


if __name__ == "__main__":
    main()
