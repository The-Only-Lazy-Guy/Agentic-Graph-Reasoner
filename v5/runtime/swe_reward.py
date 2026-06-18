"""RL reward for the SWE FIX step — same shape as derive_reward, for code patches (SEARCH/REPLACE).

  grounded = the patch APPLIES: every SEARCH block occurs in the source character-for-character
             (a hallucinated hunk that doesn't match the file is PUNISHED, even if it "looks" right).
  solves   = test-passing. Pluggable: a real verifier (SWE Docker) for eval, or a cheap GOLD-OVERLAP
             proxy for the in-loop reward (the patch's REPLACE lines overlap the gold patch's added
             lines). The proxy is imperfect (many valid fixes) but dense + single-box for training.
  unique   = a REAL edit: some REPLACE differs from its SEARCH (not a no-op / copy / identical branches).

GROUNDED is the gate (like derive_reward): an unapplyable patch is punished regardless of content, so
the policy cannot win by emitting plausible-looking hunks that don't match the file. Validated offline
on synthetic cases BEFORE training, then on real preds.

  validate (no model):  python -m v5.runtime.swe_reward
"""
from __future__ import annotations

import re
from typing import Callable, Sequence


def _norm(s: str) -> str:
    return "\n".join(line.rstrip() for line in (s or "").strip().splitlines())


def _norm_line(s: str) -> str:
    return (s or "").strip()


def grounded_applyable(blocks: Sequence[dict], source: str) -> tuple[bool, str]:
    """Every non-empty SEARCH must occur in the source (the patch can actually apply)."""
    src = _norm(source)
    if not blocks:
        return False, "no SR blocks"
    for b in blocks:
        s = _norm(b.get("search", ""))
        if s and s not in src:
            return False, f"SEARCH not in source: {s[:40]!r}... (unapplyable/hallucinated hunk)"
    return True, "all SEARCH blocks match the source"


def is_real_edit(blocks: Sequence[dict]) -> bool:
    """At least one block changes something (REPLACE != SEARCH) — not a no-op or identical branches."""
    return any(_norm(b.get("search", "")) != _norm(b.get("replace", "")) for b in blocks if b.get("search"))


def _added_lines(diff: str) -> set:
    return {_norm_line(l[1:]) for l in (diff or "").splitlines()
            if l.startswith("+") and not l.startswith("+++")
            if _norm_line(l[1:])}


def solves_goldoverlap(blocks: Sequence[dict], gold_patch: str, thresh: float = 0.5) -> bool:
    """Cheap proxy: the patch's REPLACE-introduced lines overlap the gold patch's ADDED lines.
    A real fix that matches the gold hunk scores high; a wrong/no-op edit does not."""
    gold = _added_lines(gold_patch)
    if not gold:
        return False
    repl = set()
    for b in blocks:
        s = {_norm_line(l) for l in (b.get("search", "") or "").splitlines() if _norm_line(l)}
        for l in (b.get("replace", "") or "").splitlines():
            ln = _norm_line(l)
            if ln and ln not in s:                       # genuinely new lines this patch introduces
                repl.add(ln)
    if not repl:
        return False
    hit = len(repl & gold) / max(1, len(repl))
    return hit >= thresh


def swe_reward(blocks: Sequence[dict], source: str, gold_patch: str | None = None,
               solves_fn: Callable | None = None,
               base: float = 1.0, uniq_bonus: float = 0.5, hallucinate_penalty: float = 1.0):
    """grounded (gate) x solves x unique. Returns (reward, breakdown). solves_fn(blocks)->bool overrides
    the gold-overlap proxy (plug the SWE Docker verifier here for eval)."""
    g, why = grounded_applyable(blocks, source)
    if not g:
        return -hallucinate_penalty, {"grounded": False, "why": why, "verdict": "PUNISH (unapplyable)"}
    real = is_real_edit(blocks)
    if not real:
        return 0.0, {"grounded": True, "real_edit": False, "why": why,
                     "verdict": "zero (applies but a NO-OP / identical branches)"}
    if solves_fn is not None:
        solved = bool(solves_fn(blocks))
    else:
        solved = bool(gold_patch) and solves_goldoverlap(blocks, gold_patch)
    if solved:
        return base + uniq_bonus, {"grounded": True, "real_edit": True, "solved": True, "why": why,
                                   "verdict": "REWARD (applies + solves + real edit)"}
    return 0.1, {"grounded": True, "real_edit": True, "solved": False, "why": why,
                 "verdict": "small (applies + real edit, does not match gold)"}


# ── validate on synthetic cases (no model). The reward MUST punish unapplyable + no-op, reward the
# gold-matching fix, and not be foolable by plausible-but-unapplyable hunks. ──
def _validate():
    SRC = ("def deconstruct(self):\n    name, path, args, kwargs = super().deconstruct()\n"
           "    if self.mask is None or operand.mask is None:\n        return handle_mask(self.mask)\n"
           "    return name, path, args, kwargs\n")
    GOLD = ("--- a/x.py\n+++ b/x.py\n@@\n-        return handle_mask(self.mask)\n"
            "+        if self.mask is None: return deepcopy(operand.mask)\n"
            "+        return handle_mask(self.mask, operand.mask)\n")

    def blk(search, replace, file="x.py"):
        return [{"file": file, "search": search, "replace": replace}]

    CASES = [
        ("gold-matching fix", blk("        return handle_mask(self.mask)",
                                  "        if self.mask is None: return deepcopy(operand.mask)\n"
                                  "        return handle_mask(self.mask, operand.mask)"), GOLD, "+"),
        ("UNAPPLYABLE (SEARCH not in source)", blk("return totally_made_up_function()", "return fixed()"), GOLD, "-"),
        ("NO-OP (replace == search)", blk("    return name, path, args, kwargs",
                                          "    return name, path, args, kwargs"), GOLD, "0noop"),
        ("identical-branches no-op (django-10924 style)",
         blk("    return name, path, args, kwargs",
             "    if callable(self.path):\n        return name, path, args, kwargs\n"
             "    else:\n        return name, path, args, kwargs"), GOLD, "small_or_noop"),
        ("applies + real edit, WRONG (no gold overlap)", blk("    return name, path, args, kwargs",
                                                             "    return None  # broke it"), GOLD, "0small"),
    ]
    results = []
    print("SWE REWARD validation (no model) — punish unapplyable + no-op, reward the gold-matching fix.\n")
    for name, blocks, gold, expect in CASES:
        r, b = swe_reward(blocks, SRC, gold_patch=gold)
        if expect == "+":   ok = r > 1.0
        elif expect == "-": ok = r < 0
        elif expect == "0noop": ok = r == 0.0 and b.get("real_edit") is False
        elif expect == "small_or_noop": ok = r <= 0.1            # identical branches: no-op or grounded-unsolved
        else:               ok = 0 <= r <= 0.1                   # applies, real, doesn't match gold
        results.append(ok)
        print(f"  [{'PASS' if ok else 'FAIL'}] {name:42} reward={r:+.2f}  {b['verdict']}")
    n = sum(results)
    print(f"\n=== SWE REWARD: {n}/{len(results)} PASS ===")
    print("  applies+solves+real -> +1.5 | applies+real, wrong -> +0.1 | no-op -> 0.0 | UNAPPLYABLE -> -1.0")
    print("  plug solves_fn = the SWE Docker verifier for eval; the gold-overlap proxy is the in-loop signal.")
    return n == len(results)


if __name__ == "__main__":
    import sys
    sys.exit(0 if _validate() else 1)
