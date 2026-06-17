"""RL reward for the DERIVE step — reward a derived slot fill ONLY when it is GROUNDED in upstream,
SOLVES the task, and is slightly UNIQUE (a real composition, not a copy). Ungrounded fills are
PUNISHED even when they happen to be right (no rewarding lucky hallucination — the failure mode the
boundary control exposed: derive invented '7000').

The reward IS the RL signal, so it is validated OFFLINE on known derive cases BEFORE any policy is
optimized against it (a wrong reward => RL optimizes the wrong thing). `grounded_fn`/`solves_fn` are
pluggable: arithmetic now (recompute from upstream), verifier/entailment later for SWE.

  validate (no model):  python -m v5.runtime.derive_reward
"""
from __future__ import annotations

import itertools
import operator
import re
from typing import Callable, Sequence

_OPS = {"+": operator.add, "-": operator.sub, "*": operator.mul}


def _nums(s) -> list[int]:
    return [int(x) for x in re.findall(r"-?\d+", str(s))]


def grounded_arith(value, upstream_values: Sequence[str], op: str = "+") -> tuple[bool, str]:
    """Is `value` reconstructible from upstream by the slot's DECLARED transform `op`? (recompute —
    don't trust the model's word, and DON'T accept 'any op that happens to hit', which is
    reward-hackable: with +,-,* over a couple numbers almost any value is reachable). op in {+,-,*};
    a value equal to an upstream number is a grounded COPY. Returns (grounded, why)."""
    vn = _nums(value)
    if not vn:
        return False, "no number in the derived value"
    target = vn[0]
    ups = [n for u in upstream_values for n in _nums(u)]
    if not ups:
        return False, "no upstream numbers to ground in"
    if target in ups:
        return True, f"equals upstream {target} (grounded but a COPY)"
    if op in _OPS and len(ups) >= 2:
        from functools import reduce
        expect = reduce(_OPS[op], ups)
        if target == expect:
            return True, f"{ups} via '{op}' = {target}"
        return False, f"{target} != upstream {ups} via '{op}' (={expect}); ungrounded/wrong-op"
    return False, f"cannot ground {target} in {ups} with op '{op}'"


def grounded_formula(value, upstream_named: dict, formula: Callable[[dict], float]) -> tuple[bool, str]:
    """GROUNDED via the slot's DECLARED multi-step formula over NAMED upstream values (recompute, don't
    trust the model). upstream_named = {name: value_str}; formula(nums:{name->int}) -> number. Handles
    distractor-robust selection + multi-step ops: only the named upstream feed the formula."""
    nums = {k: _nums(v)[0] for k, v in upstream_named.items() if _nums(v)}
    if len(nums) < len(upstream_named):
        return False, f"missing upstream numbers (have {nums})"
    vn = _nums(value)
    if not vn:
        return False, "no number in the derived value"
    try:
        expect = int(formula(nums))
    except Exception as e:                                    # malformed formula -> not gradeable
        return False, f"formula error: {e}"
    if vn[0] == expect:
        return True, f"formula({nums}) = {expect}"
    if vn[0] in nums.values():
        return True, f"copy of an upstream value {vn[0]} (grounded, not the formula)"
    return False, f"{vn[0]} != formula({nums}) = {expect}; ungrounded/wrong-compose"


def is_unique(value, upstream_values: Sequence[str], retrieved_texts: Sequence[str] = ()) -> bool:
    """slightly UNIQUE = the derived value is NOT a verbatim copy of an upstream value or a retrieved
    fact — it composes something new (else derive added nothing over retrieval)."""
    vn = _nums(value)
    if not vn:
        return False
    ups = [n for u in upstream_values for n in _nums(u)]
    ret = [n for t in retrieved_texts for n in _nums(t)]
    return vn[0] not in ups and vn[0] not in ret


def derive_reward(value, upstream_values: Sequence[str], gold=None, op: str = "+",
                  retrieved_texts: Sequence[str] = (), upstream_named: dict | None = None,
                  formula: Callable | None = None,
                  grounded_fn: Callable = grounded_arith,
                  solves_fn: Callable | None = None,
                  base: float = 1.0, uniq_bonus: float = 0.5, hallucinate_penalty: float = 1.0):
    """The RL reward. Order matters: GROUNDEDNESS is the gate — an ungrounded fill is PUNISHED even if
    it equals gold (no lucky-hallucination credit). Then solve + uniqueness shape the positive reward.
    Pass (upstream_named, formula) for multi-step grounding, else (upstream_values, op) for single-op.
      returns (reward: float, breakdown: dict)."""
    if formula is not None and upstream_named is not None:
        g, why = grounded_formula(value, upstream_named, formula)
        uvals = list(upstream_named.values())
    else:
        g, why = grounded_fn(value, upstream_values, op)
        uvals = list(upstream_values)
    if not g:
        return -hallucinate_penalty, {"grounded": False, "why": why, "verdict": "PUNISH (ungrounded)"}
    if solves_fn is not None:
        solved = bool(solves_fn(value))
    else:                                                      # default: arithmetic correctness vs gold
        vn = _nums(value)
        solved = bool(vn) and gold is not None and vn[0] == int(gold)
    uniq = is_unique(value, uvals, retrieved_texts)
    if solved:
        r = base + (uniq_bonus if uniq else 0.0)
        return r, {"grounded": True, "solved": True, "unique": uniq, "why": why,
                   "verdict": "REWARD" + ("+unique" if uniq else " (copy, no bonus)")}
    return 0.1, {"grounded": True, "solved": False, "unique": uniq, "why": why,
                 "verdict": "small (grounded but unsolved)"}


# ── validate the reward on KNOWN derive cases (no model). The reward MUST reward grounded+solves+
# unique and PUNISH the hallucinated boundary case — else RL would optimize the wrong signal. ──
def _validate():
    # (name, value, upstream, op, gold, expect)  expect: '+' big / '-' punish / 'copy' +1.0 / '0' small
    CASES = [
        ("derivable Fizzbolt 7+5", "12", ["Widget costs 7 credits", "Widget costs 5 credits"], "+", 12, "+"),
        ("derivable Quorblade 13+8", "21", ["costs 13 credits", "costs 8 credits"], "+", 21, "+"),
        ("derivable Vexrod 20+14", "34", ["costs 20 credits", "costs 14 credits"], "+", 34, "+"),
        ("BOUNDARY hallucinated code", "7000", ["Widget costs 7 credits"], "+", None, "-"),
        ("LUCKY right but UNSUPPORTED (12 from 3,100)", "12", ["costs 3 credits", "costs 100 credits"], "+", 12, "-"),
        ("WRONG-OP (35=7*5 when task is +)", "35", ["costs 7 credits", "costs 5 credits"], "+", 12, "-"),
        ("grounded+solves but NOT unique (12+0)", "12", ["costs 12 credits", "costs 0 credits"], "+", 12, "copy"),
        ("COPY that does not solve", "7", ["costs 7 credits", "costs 5 credits"], "+", 12, "0"),
    ]
    results = []
    print("DERIVE REWARD validation (no model) — reward grounded+solves+unique, punish hallucination.\n")
    for name, val, ups, op, gold, expect in CASES:
        r, b = derive_reward(val, ups, gold=gold, op=op)
        if expect == "+":   ok = r > 1.0
        elif expect == "-": ok = r < 0
        elif expect == "copy": ok = abs(r - 1.0) < 1e-9 and not b.get("unique", True)   # grounded+solves, NO uniq bonus
        else:               ok = 0 <= r <= 0.5                                            # grounded but unsolved
        results.append(ok)
        print(f"  [{'PASS' if ok else 'FAIL'}] {name:44} reward={r:+.2f}  {b['verdict']}  ({b.get('why','')[:42]})")
    n = sum(results)
    print(f"\n=== DERIVE REWARD: {n}/{len(results)} PASS ===")
    print("  grounded+solves+unique -> +1.5 | copy -> +1.0 | grounded-unsolved -> +0.1 | UNGROUNDED -> -1.0")
    print("  the LUCKY-hallucination row (value==gold but NOT derivable from upstream) is PUNISHED -> RL")
    print("  cannot win by inventing the right-looking number; it must actually compose from the graph.")
    return n == len(results)


if __name__ == "__main__":
    import sys
    sys.exit(0 if _validate() else 1)
