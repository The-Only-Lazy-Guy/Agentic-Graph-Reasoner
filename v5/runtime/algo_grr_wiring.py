"""algo_grr_wiring — the COMPOSITION-CEILING probe: where the frozen LM mis-wires, and the planner fixes it.

Weakness #1 in the honest gap map: the frozen LM is the composition ceiling. The graph delivers verified
atoms; the LM must WIRE them. Simple nesting f(g(n)) is trivial, so we test STRUCTURE INFERENCE: a nested
arithmetic expression over n, described in words, that the LM must parse into correctly-wired atom calls.
As the expression DEPTH grows, the frozen LM mis-parses / mis-routes -> solve-rate drops.

  Arm A (free-form): the LM reads the word description + atom helpers, writes the code itself.
  Arm B (planned):   the ground-truth structure (the expression TREE) is realized DETERMINISTICALLY into
                     atom calls -> 100% by construction, at any depth.

The DELTA at each depth = the LM's structure-inference/wiring ceiling that a PLANNER removes. This is NOT
the str_dp2 result (a single abstraction's Δcapability on algorithmic tasks) — it isolates compositional
DEPTH. Honest scope: Arm B is GIVEN the tree; a TRM that INFERS the tree from NL is the future planner
(the hard part). What is shown: given the plan, realization is perfect; the frozen LM alone is not.

    python -m v5.runtime.algo_grr_wiring --selftest   # no-GPU: realizer solves all depths; harness works
    python -m v5.runtime.algo_grr_wiring --run --lm Qwen/Qwen2.5-3B-Instruct   # free-form LM vs planned
"""
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

_ROOT = str(Path(__file__).resolve().parents[2])
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

# atom name -> (python fn, helper-name, word-template)
_UNARY = {
    "double": (lambda x: 2 * x, "dbl", "double of {0}"),
    "increment": (lambda x: x + 1, "inc", "one more than {0}"),
    "square": (lambda x: x * x, "sq", "the square of {0}"),
    "negate": (lambda x: -x, "neg", "the negation of {0}"),
}
_BINARY = {
    "sum": (lambda x, y: x + y, "add", "the sum of {0} and {1}"),
    "product": (lambda x, y: x * y, "mul", "the product of {0} and {1}"),
    "difference": (lambda x, y: x - y, "sub", "the difference of {0} and {1}"),
}
_HELPERS = {
    "dbl": "def dbl(x):\n    return 2*x\n",
    "inc": "def inc(x):\n    return x+1\n",
    "sq": "def sq(x):\n    return x*x\n",
    "neg": "def neg(x):\n    return -x\n",
    "add": "def add(x,y):\n    return x+y\n",
    "mul": "def mul(x,y):\n    return x*y\n",
    "sub": "def sub(x,y):\n    return x-y\n",
}
_HELP_DESC = {"dbl": "double x", "inc": "x plus 1", "sq": "x times x", "neg": "minus x",
              "add": "x plus y", "mul": "x times y", "sub": "x minus y"}


def gen_expr(depth: int, rng: random.Random):
    """A random expression tree over the single input n. Leaves = n; internal nodes = unary/binary atoms."""
    if depth <= 0:
        return ("n",)
    if rng.random() < 0.5:
        op = rng.choice(list(_UNARY))
        return (op, gen_expr(depth - 1, rng))
    op = rng.choice(list(_BINARY))
    # keep one branch shallower so trees don't blow up exponentially
    return (op, gen_expr(depth - 1, rng), gen_expr(rng.randint(0, depth - 1), rng))


def to_words(t) -> str:
    if t[0] == "n":
        return "n"
    if t[0] in _UNARY:
        return _UNARY[t[0]][2].format(f"({to_words(t[1])})")
    tmpl = _BINARY[t[0]][2]
    return tmpl.format(f"({to_words(t[1])})", f"({to_words(t[2])})")


def to_code(t) -> str:
    """DETERMINISTIC realization of the tree into atom calls (the planned arm)."""
    if t[0] == "n":
        return "n"
    if t[0] in _UNARY:
        return f"{_UNARY[t[0]][1]}({to_code(t[1])})"
    name = _BINARY[t[0]][1]
    return f"{name}({to_code(t[1])}, {to_code(t[2])})"


def oracle(t, n: int) -> int:
    if t[0] == "n":
        return n
    if t[0] in _UNARY:
        return _UNARY[t[0]][0](oracle(t[1], n))
    return _BINARY[t[0]][0](oracle(t[1], n), oracle(t[2], n))


def _closure() -> str:
    return "\n".join(_HELPERS[h] for h in _HELPERS)


def verify(entry_body: str, t, entry="solve", tests=(2, 3, 5, 7, 11)) -> bool:
    """Prepend the verified atom closure, run entry(n), compare to the oracle. entry_body = the glue only."""
    code = _closure() + "\n\n" + entry_body
    ns: dict = {}
    try:
        exec(compile(code, "<wiring>", "exec"), ns)  # noqa: S102
        fn = ns.get(entry)
        if fn is None:
            return False
        for n in tests:
            if fn(n) != oracle(t, n):
                return False
        return True
    except Exception:  # noqa: BLE001
        return False


def realize_planned(t, entry="solve") -> str:
    """Arm B: given the tree (the plan), emit correct glue deterministically."""
    return f"def {entry}(n):\n    return {to_code(t)}\n"


def freeform_prompt(t, entry="solve") -> str:
    helpers = "\n".join(f"  {h}(...) : {_HELP_DESC[h]}" for h in _HELPERS)
    return (f"You may call these helper functions (already defined):\n{helpers}\n\n"
            f"Write a Python function `{entry}(n)` that returns {to_words(t)}.\n"
            f"Use ONLY the helpers above; output only the function definition.")


# ═══════════════════════════════════════════════════════════════════════════════
# SELFTEST (no GPU): the planned realizer solves EVERY depth; the harness verifies correctly.
# ═══════════════════════════════════════════════════════════════════════════════

def _selftest() -> bool:
    print("algo_grr_wiring --selftest: composition-ceiling harness (no GPU)\n")
    ok = True
    rng = random.Random(0)
    for depth in (1, 2, 3, 4, 5, 6):
        solved = 0
        n_tasks = 40
        for i in range(n_tasks):
            t = gen_expr(depth, random.Random(1000 * depth + i))
            if verify(realize_planned(t), t):
                solved += 1
        rate = solved / n_tasks
        print(f"  depth {depth}: planned realizer solves {solved}/{n_tasks} = {rate:.2f}")
        ok &= rate == 1.0
    # a deliberately WRONG wiring must FAIL (verifier is real, not a rubber stamp)
    t = gen_expr(3, random.Random(7))
    wrong = "def solve(n):\n    return inc(n)\n"                 # ignores the tree
    caught = not verify(wrong, t)
    print(f"  verifier rejects a wrong wiring: {'PASS' if caught else 'FAIL'}")
    ok &= caught
    print(f"\n  Planned arm = 100% at every depth by construction (deterministic realize). The --run test\n"
          f"  measures whether the FREE-FORM frozen LM holds or DROPS as depth grows (its wiring ceiling).")
    print(f"\n  ALGO_GRR_WIRING SELFTEST -> {'PASS' if ok else 'FAIL'}")
    return ok


def _run(a) -> bool:
    import os
    import torch
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    from transformers import AutoTokenizer
    from v5.lm_loader import load_frozen_lm
    from v5.runtime.algo_grr_membrane import _extract_code, strip_module_exec

    tok = AutoTokenizer.from_pretrained(a.lm)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = load_frozen_lm(a.lm).eval()

    def gen(prompt):
        msg = tok.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False,
                                      add_generation_prompt=True)
        ids = tok(msg, return_tensors="pt", add_special_tokens=False).input_ids.to(model.device)
        with torch.no_grad():
            out = model.generate(input_ids=ids, do_sample=False, max_new_tokens=160,
                                 pad_token_id=tok.pad_token_id)
        return tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True)

    depths = a.depths
    print(f"  depth | free-form {a.lm.split('/')[-1]} | planned   ({a.trials} tasks each)")
    rows = []
    for depth in depths:
        ff = pl = 0
        for i in range(a.trials):
            t = gen_expr(depth, random.Random(9000 * depth + i))
            body = strip_module_exec(_extract_code(gen(freeform_prompt(t))))
            if verify(body, t):
                ff += 1
            if verify(realize_planned(t), t):
                pl += 1
        aff, apl = ff / a.trials, pl / a.trials
        rows.append((aff, apl))
        print(f"  {depth:>5} |   {aff:.2f}              |  {apl:.2f}", flush=True)
    ff_lo, ff_hi = rows[0][0], rows[-1][0]
    ok = ff_hi < ff_lo - 0.15 and rows[-1][1] > 0.95
    print(f"\n  free-form LM: {ff_lo:.2f} (shallow) -> {ff_hi:.2f} (deep) = WIRING CEILING")
    print(f"  planned: {rows[-1][1]:.2f} at depth {depths[-1]} = structure removes the ceiling")
    print(f"  -> {'PASS' if ok else 'INCONCLUSIVE'}: the frozen LM's composition ceiling is {'real' if ok else 'higher than tested'}.")
    return ok


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--lm", default="Qwen/Qwen2.5-3B-Instruct")
    ap.add_argument("--depths", type=int, nargs="*", default=[1, 2, 3, 4, 5])
    ap.add_argument("--trials", type=int, default=30)
    a = ap.parse_args()
    if a.selftest:
        sys.exit(0 if _selftest() else 1)
    if a.run:
        sys.exit(0 if _run(a) else 1)
    ap.print_help()


if __name__ == "__main__":
    main()
