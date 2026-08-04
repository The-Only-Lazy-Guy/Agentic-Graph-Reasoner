"""algo_grr_reuse -- DO SELF-AUTHORED, VERIFIER-GATED TOOLS TRANSFER TO NEW BUGS?

The claim under test, and the only part of this stack that is not a generic coding-agent harness:
a small LM authors a repair TOOL, the tool is gated and banked in the graph, and LATER instances are
repaired by REPLAYING a banked tool with ZERO LM CALLS. If that works, the system gets cheaper as it
sees more bugs. Every coding agent re-prompts from scratch every time; none of them accumulate
verified, reusable transformations.

WHY THE PREVIOUS TOOLS COULD NEVER TRANSFER. The authored tools hardcoded the instance:
    old = "        if isinstance(value, bytes):"
    return text.replace(old, new, 1)
`old` is one exact line of one exact file, so the bank was write-only by construction. This module
asks the LM for PARAMETERISED tools (a pattern, not a literal) and then measures transfer honestly.

THE CORRECTNESS SIGNAL, and its limits. Only 3 SWE-bench Docker images are cached here, so real test
execution cannot measure transfer across hundreds of instances. Instead each instance's REAL gold diff
is parsed into (before_hunk, after_hunk) and a tool is correct on that instance when
    tool(before_hunk) == after_hunk
i.e. it reproduces the maintainer's actual fix. This is:
  * REAL data -- the gold patch, not a synthetic mutation;
  * STRICTER than the test suite (a different-but-valid fix would pass tests and fail this), so it is
    a LOWER BOUND on correctness, never an inflated one;
  * HUNK-level, not whole-file: it proves the transformation is right where the fix lives, and does
    not prove the tool leaves the rest of the file alone. The 3 cached instances are still run through
    the real Docker verifier so the two measures can be compared on common ground.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
from pathlib import Path

_ROOT = str(Path(__file__).resolve().parents[2])
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
os.environ.setdefault("HF_HOME", r"E:\cache\hf")

import numpy as np

from v5.runtime.algo_grr_toolsmith import ToolBank, run_edit_tool, strip_fences


# ── real gold diffs -> (before, after) hunks ─────────────────────────────────────────────────────
def parse_hunks(diff: str) -> list:
    """Split a unified diff into (before, after) text pairs, one per hunk.

    Context lines belong to BOTH sides; '-' lines only to before; '+' lines only to after. A tool that
    turns before into after has reproduced the maintainer's edit exactly."""
    out, before, after, inhunk = [], [], [], False
    for ln in (diff or "").splitlines():
        if ln.startswith("@@"):
            if inhunk and (before or after):
                out.append(("\n".join(before), "\n".join(after)))
            before, after, inhunk = [], [], True
            continue
        if not inhunk:
            continue
        if ln.startswith("--- ") or ln.startswith("+++ ") or ln.startswith("diff "):
            continue
        if ln.startswith("-"):
            before.append(ln[1:])
        elif ln.startswith("+"):
            after.append(ln[1:])
        elif ln.startswith(" ") or ln == "":
            before.append(ln[1:] if ln else "")
            after.append(ln[1:] if ln else "")
    if inhunk and (before or after):
        out.append(("\n".join(before), "\n".join(after)))
    return [(b, a) for b, a in out if b != a]


_HUNKS = Path(_ROOT) / "artifacts" / "swebench_gold_hunks.json"


def load_gold(n: int = 400, repo: str = "django/django", seed: int = 0) -> list:
    """Real SWE-bench instances with their real gold patches, reduced to single-hunk single-file
    cases. Multi-hunk fixes are dropped rather than half-scored -- a tool that reproduces one hunk of
    three has not made the repair, and counting it would inflate every number in this file.

    Reads the prepped json rather than the parquet: this module imports torch (via toolsmith), and
    pyarrow after torch segfaults in this environment -- it exits SILENTLY with no traceback, which is
    exactly how it presented. Run scripts/prep_gold_hunks.py to regenerate."""
    if not _HUNKS.exists():
        raise FileNotFoundError(f"{_HUNKS} missing -- run: python scripts/prep_gold_hunks.py")
    rows = [r for r in json.loads(_HUNKS.read_text(encoding="utf-8"))
            if (repo in (None, "", "all") or r["repo"] == repo)]
    random.Random(seed).shuffle(rows)
    return rows[:n]


# ── authoring: ask for a PATTERN, not a literal ──────────────────────────────────────────────────
PROMPT = """You are writing a REUSABLE Python repair tool.

BUG REPORT:
{issue}

THE CODE THAT MUST CHANGE:
{before}

Write a function:

def edit(text: str) -> str

It receives the code above and returns it with the bug fixed.

CRITICAL -- this tool is stored and REPLAYED on OTHER bugs later, so write it to GENERALISE:
- Prefer a `re.sub` PATTERN over an exact string literal.
- Capture the parts that vary (names, arguments) with groups and put them back with \\1, \\2.
- Do not embed this file's specific indentation or variable names unless the fix truly requires it.
- Change only what the bug requires.

Output ONLY the function. No markdown fences, no commentary.

Example of the GENERALISING style that is wanted:

def edit(text: str) -> str:
    import re
    return re.sub(r"isinstance\\((\\w+), bytes\\)", r"isinstance(\\1, (bytes, memoryview))", text, count=1)
"""


ial_PROMPT = """You are fixing a bug by writing a small Python function.

BUG REPORT:
{issue}

THE CODE THAT MUST CHANGE:
{before}

Write a function:

def edit(text: str) -> str

It receives the code above and returns it with the bug fixed. Use an exact string replacement:
copy the line to change VERBATIM from the code above, including its leading indentation.
Change only what the bug requires.

Output ONLY the function. No markdown fences, no commentary.

Example:

def edit(text: str) -> str:
    old = "    if isinstance(value, bytes):"
    new = "    if isinstance(value, (bytes, memoryview)):"
    return text.replace(old, new, 1)
"""


def author_tool(lm, issue: str, before: str, feedback: str = "", style: str = "param") -> str:
    """style='param' asks for a REUSABLE pattern; style='literal' asks for an exact replacement.

    The literal arm is the CONTROL that makes the parameterised result interpretable. Asking for a
    generalising regex is strictly harder than asking for a string replace, so a null in the
    parameterised arm alone cannot distinguish "tools do not transfer" from "this model cannot write
    the harder kind of tool at all". Running both on the SAME instances separates them.
    """
    tmpl = PROMPT if style == "param" else ial_PROMPT
    p = tmpl.format(issue=issue[:900], before=before[:1500])
    if feedback:
        p += f"\nYour previous attempt failed: {feedback}\nFix exactly that.\n"
    try:
        return strip_fences(str(lm.generate_chat(p, max_new=300, temperature=0.6)))
    except Exception as e:                                         # noqa: BLE001
        return ""


def tool_fixes(code: str, before: str, after: str) -> tuple:
    """VERIFIED correctness: does the tool turn the real before-hunk into the real after-hunk?"""
    if "def edit" not in (code or ""):
        return False, "no `def edit`"
    ok, res = run_edit_tool(code, before)
    if not ok:
        return False, str(res)[:120]
    if res == before:
        return False, "tool made no change"
    if res.strip() == after.strip():
        return True, "reproduces the gold fix exactly"
    return False, "changed the code but not into the gold fix"


def is_parameterised(code: str) -> bool:
    """A crude but honest proxy: does the tool use a PATTERN rather than only literal replacement?
    Reported alongside transfer so the two can be compared -- if parameterised tools transfer and
    literal ones do not, that is the mechanism, not a coincidence."""
    return bool(re.search(r"re\.(sub|compile|search|match)", code or ""))


# ── the experiment: reuse FIRST, author only on miss ─────────────────────────────────────────────
def run(lm, rows: list, bank: ToolBank, tries: int = 2, verbose: bool = True,
        style: str = "param") -> dict:
    """Stream instances in order. For each: try every banked tool first (ZERO LM calls); only if all
    fail does the LM author a new one. Verified tools are banked. The claim lives or dies on whether
    the replay column stays at zero."""
    stats = {"n": 0, "replay": 0, "authored": 0, "lm_calls": 0, "solved": 0, "param": 0}
    curve = []
    for row in rows:
        stats["n"] += 1
        before, after = row["before"], row["after"]

        # 1) REPLAY -- retrieve by meaning from the graph, run each candidate, no LM involved
        hit = None
        for name, code, cos in bank.retrieve(row["problem"], k=5):
            good, _ = tool_fixes(code, before, after)
            if good:
                hit = (name, cos)
                break
        if hit:
            stats["replay"] += 1
            stats["solved"] += 1
            curve.append((stats["n"], stats["replay"], stats["lm_calls"]))
            if verbose:
                print(f"  [{stats['n']:3d}] {row['instance_id'][:34]:34s} REPLAY  {hit[0]} "
                      f"(cos {hit[1]:.2f})  LM calls: 0", flush=True)
            continue

        # 2) AUTHOR -- the LM only runs when the bank could not do it
        fb, done = "", False
        for _ in range(tries):
            stats["lm_calls"] += 1
            code = author_tool(lm, row["problem"], before, fb, style=style)
            good, why = tool_fixes(code, before, after)
            if good:
                stats["authored"] += 1
                stats["solved"] += 1
                stats["param"] += int(is_parameterised(code))
                bank.bank(row["problem"], row["repo"], row["instance_id"], code, "gold-verified")
                done = True
                if verbose:
                    print(f"  [{stats['n']:3d}] {row['instance_id'][:34]:34s} AUTHORED"
                          f"{' (parameterised)' if is_parameterised(code) else ' (literal)'} "
                          f"-> banked, size {len(bank)}", flush=True)
                break
            fb = why
        if not done and verbose:
            print(f"  [{stats['n']:3d}] {row['instance_id'][:34]:34s} failed: {fb[:60]}", flush=True)
        curve.append((stats["n"], stats["replay"], stats["lm_calls"]))
    stats["curve"] = curve
    return stats


def _selftest() -> bool:
    print("algo_grr_reuse --selftest: real gold diffs, real transfer accounting\n")
    ok = True

    def chk(t, c, d=""):
        nonlocal ok
        ok &= bool(c)
        print(f"  [{'PASS' if c else 'FAIL'}] {t}{('  - ' + d) if d else ''}")

    d = ("--- a/f.py\n+++ b/f.py\n@@ -1,3 +1,3 @@\n ctx\n-    if isinstance(v, bytes):\n"
         "+    if isinstance(v, (bytes, memoryview)):\n more\n")
    hs = parse_hunks(d)
    chk("[1] a real unified diff splits into before/after hunks", len(hs) == 1, f"{len(hs)} hunk")
    b, a = hs[0]
    chk("[2] context lines appear on BOTH sides; -/+ only on their own",
        "ctx" in b and "ctx" in a and "(bytes, memoryview)" in a and "(bytes, memoryview)" not in b)

    good, why = tool_fixes("def edit(text):\n    import re\n    return re.sub(r'bytes\\)', "
                           "'(bytes, memoryview))', text, count=1)\n", b, a)
    chk("[3] a tool reproducing the GOLD fix is accepted", good, why)
    bad, why2 = tool_fixes("def edit(text):\n    return text.replace('ctx','CTX')\n", b, a)
    chk("[4] a tool that changes the code but NOT into the gold fix is REJECTED", not bad, why2)
    bad2, why3 = tool_fixes("def edit(text):\n    return text\n", b, a)
    chk("[5] a no-op tool is REJECTED", not bad2, why3)

    chk("[6] parameterised vs literal is detected",
        is_parameterised("import re\nre.sub(r'a','b',t)") and
        not is_parameterised("t.replace('a','b')"))

    rows = load_gold(n=5)
    chk("[7] real SWE-bench gold patches load and reduce to single-hunk cases",
        len(rows) > 0 and all(r["before"] != r["after"] for r in rows),
        f"{len(rows)} instances, e.g. {rows[0]['instance_id'] if rows else '-'}")

    print(f"\n  ALGO_GRR_REUSE -> {'PASS' if ok else 'FAIL'}")
    return ok


def main():
    ap = argparse.ArgumentParser(description="Do self-authored verified tools transfer?")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--lm", type=str, default="Qwen/Qwen2.5-3B-Instruct")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--style", choices=["param", "literal"], default="param",
                    help="literal = the CONTROL arm: exact string replacement, easier to "
                         "author but cannot transfer by construction")
    a = ap.parse_args()
    if a.selftest:
        sys.exit(0 if _selftest() else 1)
    if not a.run:
        ap.print_help(); return

    rows = load_gold(n=a.n, seed=a.seed)
    print(f"{len(rows)} real single-hunk django instances (gold patches as ground truth)\n")
    from v5.runtime.dcpd_latent import WhiteBox
    lm = WhiteBox(a.lm, quant="4bit")
    bank = ToolBank()
    st = run(lm, rows, bank, style=a.style)

    n = max(1, st["n"])
    print(f"\n{'=' * 74}")
    print(f"TRANSFER OF SELF-AUTHORED, VERIFIED TOOLS   (n={n}, {a.lm}, style={a.style})")
    print(f"  repaired in total                 : {st['solved']}/{n}")
    print(f"  ...by REPLAY of a banked tool     : {st['replay']}   <- ZERO LM calls")
    print(f"  ...by authoring a new tool        : {st['authored']}")
    print(f"  tools banked                      : {len(bank)}")
    print(f"  of those, parameterised (re.*)    : {st['param']}/{max(1, st['authored'])}")
    print(f"  total LM calls                    : {st['lm_calls']}")
    print(f"  LM calls per repair               : "
          f"{st['lm_calls'] / max(1, st['solved']):.2f}")
    half = n // 2
    e1 = sum(1 for i, r, _ in st["curve"] if i <= half and r)
    e2 = st["replay"] - e1
    print(f"\n  COMPOUNDING CHECK (does replay rise as the bank grows?)")
    print(f"    replays in first half : {e1}")
    print(f"    replays in second half: {e2}")
    print(f"    -> {'RISES' if e2 > e1 else 'does NOT rise'}")
    print(f"{'=' * 74}")


if __name__ == "__main__":
    main()
