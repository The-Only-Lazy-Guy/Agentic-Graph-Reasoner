"""Composition tasks for v6, built so the experiment CAN measure memory.

Every constraint here exists because v5's version could not, and the failures were only visible
once the controls worked:

1. OPAQUE NAMES. v5 generated task text from phrases describing exactly the operation each atom was
   NAMED after ("the square of the nth Fibonacci number" -> square(fibonacci(n))), and verified
   behaviourally. Observed in a real run:
       target : def task(n): return square(fibonacci(n))
       model  : def task(n): return fibonacci(n)**2      <- PASSED, using neither composed atom
   When the prompt contains a complete solution path, WM and DERANGED cannot differ and the
   derangement arm detects nothing. Here the English says WHAT to compute; only memory says WHICH
   identifier implements it.

2. CODE-SHAPED PROMPT. v5's prompt ended in "Explanation:\\n" while the target was a bare
   " return {expr}" fragment, so the ablated arm scored 0 by OBEYING its instruction and the
   adapter's entire measured gain was "emit code instead of prose". The prompt here ends where the
   answer begins.

3. THE VERIFIER REQUIRES THE ATOMS TO BE CALLED. Behavioural equality alone lets an inlined
   reimplementation pass, which again makes memory unnecessary.

4. ENOUGH TASKS FOR RL. v5's generator capped at 64 pairs from 10 atoms, leaving 32 held after
   training -- badly underpowered for a binary terminal reward, and the binding constraint on every
   confidence interval measured this session. 28 atoms give ~750 ordered pairs.

Split is BY PAIR with no train/held overlap, and the atom POOL is shared: the model must generalise
composition, not memorise a new atom.
"""
from __future__ import annotations

import random

# name -> (english description, python body over n). Names are assigned opaquely below; these keys
# are internal only and never reach the model.
_OPS: dict[str, tuple[str, str]] = {
    "square":        ("the square of a number",                              "n * n"),
    "cube":          ("the cube of a number",                                "n ** 3"),
    "double":        ("twice a number",                                      "2 * n"),
    "triple":        ("three times a number",                                "3 * n"),
    "succ":          ("one more than a number",                              "n + 1"),
    "pred":          ("one less than a number",                              "n - 1"),
    "negate":        ("the negation of a number",                            "-n"),
    "absval":        ("the absolute value of a number",                      "abs(n)"),
    "halve":         ("a number integer-divided by two",                     "n // 2"),
    "digit_sum":     ("the sum of the decimal digits of a number",           "sum(int(c) for c in str(abs(n)))"),
    "digit_count":   ("how many decimal digits a number has",                "len(str(abs(n)))"),
    "reverse_dig":   ("the number with its decimal digits reversed",         "int(str(abs(n))[::-1])"),
    "count_bits":    ("the number of one bits in the binary representation", "bin(abs(n)).count('1')"),
    "bit_length":    ("how many bits are needed to represent a number",      "abs(n).bit_length()"),
    "sum_to_n":      ("the sum of all integers from 1 to a number",          "n * (n + 1) // 2"),
    "num_divisors":  ("how many positive divisors a number has",             "sum(1 for i in range(1, abs(n) + 1) if n % i == 0) if n else 0"),
    "largest_digit": ("the largest decimal digit of a number",               "max(int(c) for c in str(abs(n)))"),
    "smallest_digit":("the smallest decimal digit of a number",              "min(int(c) for c in str(abs(n)))"),
    "digit_product": ("the product of the decimal digits of a number",       "__import__('math').prod(int(c) for c in str(abs(n)))"),
    "mod_seven":     ("the remainder of a number divided by seven",          "n % 7"),
    "mod_three":     ("the remainder of a number divided by three",          "n % 3"),
    "next_even":     ("the next even number at or above a number",           "n + (n % 2)"),
    "is_even":       ("one if a number is even and zero otherwise",          "int(n % 2 == 0)"),
    "is_prime":      ("one if a number is prime and zero otherwise",         "int(n >= 2 and all(n % i for i in range(2, int(abs(n) ** 0.5) + 1)))"),
    "sign_of":       ("the sign of a number as -1, 0 or 1",                  "(n > 0) - (n < 0)"),
    "tens_digit":    ("the tens digit of a number",                          "(abs(n) // 10) % 10"),
    "ones_digit":    ("the ones digit of a number",                          "abs(n) % 10"),
    "third_of":      ("a number integer-divided by three",                   "n // 3"),
}

_TOKENS = ["qx7", "vk2", "zn9", "hb4", "tm6", "pj1", "wd8", "rc5", "gf3", "ly0",
           "as4", "mn3", "eu6", "kt8", "bz1", "cw5", "df9", "hj2", "ip7", "or0",
           "sv3", "ux4", "yb6", "ea8", "fn1", "gm5", "ho2", "jr9"]


def build_atoms(seed: int = 0) -> tuple[dict[str, str], dict[str, str], dict[str, str]]:
    """Returns (descriptions, code, mapping internal_name -> opaque_name).

    Opaque names are ASSIGNED AT RANDOM per seed so no run can accidentally depend on a particular
    name/description pairing that a tokenizer happens to find easy.
    """
    rng = random.Random(seed)
    keys = list(_OPS)
    toks = _TOKENS[:len(keys)]
    rng.shuffle(toks)
    mapping = {k: f"op_{t}" for k, t in zip(keys, toks)}
    descs = {mapping[k]: _OPS[k][0] for k in keys}
    codes = {mapping[k]: f"def {mapping[k]}(n): return {_OPS[k][1]}" for k in keys}
    return descs, codes, mapping


def make_tasks(n_train: int = 200, n_held: int = 200, seed: int = 0):
    """Composition tasks outer(inner(n)). Returns (train, held, descs, codes).

    Each task is (text, atoms_needed, target_expr). Split is BY PAIR and disjoint; the atom pool is
    shared so the model must generalise composition rather than memorise an unseen atom.
    """
    descs, codes, _ = build_atoms(seed)
    names = list(descs)
    rng = random.Random(seed + 1)
    pairs = [(o, i) for o in names for i in names if o != i]
    rng.shuffle(pairs)
    want = min(len(pairs), n_train + n_held)
    pairs = pairs[:want]
    cut = int(round(len(pairs) * n_train / max(1, n_train + n_held)))

    def mk(pr):
        out = []
        for o, i in pr:
            text = f"{descs[o]}, applied to {descs[i]}"
            out.append((text, [i, o], f"{o}({i}(n))"))
        return out

    return mk(pairs[:cut]), mk(pairs[cut:]), descs, codes


def make_tests(expr: str, codes: dict[str, str], n_probe: int = 6, seed: int = 0):
    """(input, expected) pairs from executing the TARGET expression. Skips inputs that raise."""
    ns: dict = {}
    for src in codes.values():
        exec(src, ns)
    rng = random.Random(seed)
    tests = []
    for _ in range(n_probe * 4):
        if len(tests) >= n_probe:
            break
        x = rng.randint(1, 60)
        try:
            tests.append((x, eval(expr.replace("(n)", f"({x})"), dict(ns))))
        except Exception:
            continue
    return tests


def verify(generated: str, atoms_needed: list[str], tests: list, codes: dict[str, str]) -> bool:
    """Behavioural equality AND the required atoms actually CALLED.

    The call check is not pedantry. v5 verified behaviour only, so an inlined reimplementation
    passed without touching the composed atoms -- which makes working memory unnecessary and the
    derangement arm blind. Both conditions must hold.
    """
    expr = _first_expr(generated)
    if not expr:
        return False
    if not all(f"{a}(" in expr for a in atoms_needed):
        return False
    ns: dict = {}
    for src in codes.values():
        exec(src, ns)
    try:
        exec(f"def _t(n):\n    return {expr}\n", ns)
        fn = ns["_t"]
        return all(fn(x) == y for x, y in tests)
    except Exception:
        return False


def _first_expr(text: str) -> str | None:
    """Pull the first complete expression out of a raw continuation.

    Greedy decoding with no stop criterion tends to repeat the same clause, so take the first line
    and cut at obvious continuation boundaries.
    """
    t = text.strip()
    if t.startswith("return "):
        t = t[7:]
    for cut in ("\n", "#", "  return", "\treturn"):
        i = t.find(cut)
        if i > 0:
            t = t[:i]
    t = t.strip().rstrip(".").rstrip()
    return t or None


def prompt_for(text: str) -> str:
    """Ends where the answer begins, so a continuation IS the target and ablated measures ability."""
    return f"# {text}\ndef task(n):\n    return "
