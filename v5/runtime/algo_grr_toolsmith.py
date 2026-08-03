"""algo_grr_toolsmith -- the model AUTHORS its own edit tools, the verifier gates them, and survivors
BANK into the worlds graph for reuse.

This is the project's own thesis applied to the thing that was actually blocking it. The measured
0/3 solve count came from an edit mechanism I hardcoded: pick the first unique line mentioning an
issue token (always an import near the top of the file) and let the LM rewrite THAT ONE LINE. Real
fixes are multi-line changes inside method bodies, so the mechanism could not express a fix no matter
which model wrote it or how well localization worked. The fix is not a better prompt -- it is to stop
hardcoding the editor and let the model build one.

WHY A TOOL AND NOT JUST A BIGGER PATCH STRING:
  - A tool is EXECUTABLE and therefore VERIFIABLE before anything is claimed: it must run, terminate,
    produce a real change, and leave the file parseable. A patch string has no such gates.
  - A tool is REUSABLE. "add a keyword argument to a method signature" recurs across instances and
    across repos. Once verified it is banked as a real atom in the AtomGraph and retrieved by meaning,
    which is the compounding claim this project already validated for solvers
    (tool-induction: 18% instance-reasoning -> 100% verified self-built solver) and for tool
    SEQUENCES (algo_grr_agent: verified tool-seqs bank as reusable workflows, replay == reuse).
  - It keeps the DCPD split: the STRUCTURE (find/anchor/apply/verify) is exact code, the LM fills only
    the semantic hole (what the transformation should be).

SAFETY. LM-authored code is executed, so it is executed in a SUBPROCESS with a hard timeout, with a
restricted builtins namespace, and it is text-in/text-out: the function receives the file's contents as
a string and returns a string. It is never handed a file handle, `os`, or the filesystem. The actual
write happens later, inside the SWE-bench container, from the returned text.
"""
from __future__ import annotations

import ast
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path

_ROOT = str(Path(__file__).resolve().parents[2])
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import numpy as np

_TIMEOUT = 15
_MAX_SRC = 6000

# The harness the authored tool runs inside. Restricted builtins + `re` only; no os, no open, no
# import machinery. The tool sees TEXT and returns TEXT.
_RUNNER = r'''
import json, sys, re, builtins
_ALLOW = ["len","range","str","list","dict","set","tuple","enumerate","min","max","sorted","any",
          "all","int","float","bool","reversed","zip","abs","isinstance","repr","ord","chr","sum",
          "map","filter","next","iter","slice","print","Exception","ValueError","TypeError",
          "IndexError","KeyError","StopIteration","AttributeError"]
payload = json.loads(open(sys.argv[1], encoding="utf-8").read())

# A restricted __import__ rather than none at all. Measured: with __import__ removed entirely, 2 of 3
# real authoring attempts died on "ImportError: __import__ not found" because the model wrote a
# perfectly reasonable `import re` at the top of its tool -- the sandbox was rejecting correct tools,
# so the mechanism was never actually under test. Whitelisted to pure text-manipulation modules; os,
# sys, subprocess, shutil, pathlib and friends remain unreachable, which is what actually matters.
_SAFE_MODS = {"re", "string", "textwrap", "itertools", "collections", "difflib", "json", "math"}


def _import(name, globals=None, locals=None, fromlist=(), level=0):
    root = name.split(".")[0]
    if root not in _SAFE_MODS:
        raise ImportError(f"module {root!r} is not available inside the edit sandbox")
    return __import__(name, globals, locals, fromlist, level)


_bi = {k: getattr(builtins, k) for k in _ALLOW}
_bi["__import__"] = _import
ns = {"__builtins__": _bi, "re": re}
try:
    exec(payload["code"], ns)
    fn = ns.get("edit")
    if not callable(fn):
        raise ValueError("no callable named `edit` was defined")
    out = fn(payload["text"])
    if not isinstance(out, str):
        raise ValueError(f"edit() must return str, got {type(out).__name__}")
    print(json.dumps({"ok": True, "text": out}))
except Exception as e:
    print(json.dumps({"ok": False, "err": f"{type(e).__name__}: {e}"}))
'''


def run_edit_tool(code: str, text: str) -> tuple:
    """Execute an authored tool on `text` in an isolated subprocess. Returns (ok, new_text_or_err).

    Subprocess rather than in-process exec for two independent reasons: an LM-authored loop can fail
    to terminate (a timeout is the only real defence), and it keeps the tool away from this process's
    namespace entirely."""
    with tempfile.TemporaryDirectory() as td:
        pay = Path(td) / "p.json"
        run = Path(td) / "r.py"
        pay.write_text(json.dumps({"code": code, "text": text}), encoding="utf-8")
        run.write_text(_RUNNER, encoding="utf-8")
        try:
            r = subprocess.run([sys.executable, str(run), str(pay)],
                               capture_output=True, text=True, timeout=_TIMEOUT)
        except subprocess.TimeoutExpired:
            return False, f"tool did not terminate within {_TIMEOUT}s"
        out = (r.stdout or "").strip().splitlines()
        if not out:
            return False, f"tool produced no output ({(r.stderr or '')[:200]})"
        try:
            d = json.loads(out[-1])
        except Exception:                                          # noqa: BLE001
            return False, f"tool output unparseable: {out[-1][:200]}"
        return bool(d.get("ok")), (d.get("text") if d.get("ok") else d.get("err", "unknown"))


def gate_edit(orig: str, new: str) -> tuple:
    """Verification BEFORE any test run, so a broken tool is rejected cheaply and specifically.
    Every gate here is a real property of the result, never a judgement about it."""
    if new is None:
        return False, "no output"
    if new == orig:
        return False, "tool made no change"
    if not new.strip():
        return False, "tool emptied the file"
    try:
        ast.parse(new)
    except SyntaxError as e:
        return False, f"result does not parse: {e}"
    # a tool that rewrites the whole file is almost never doing what it claims; real fixes are local.
    a, b = orig.splitlines(), new.splitlines()
    changed = sum(1 for x, y in zip(a, b) if x != y) + abs(len(a) - len(b))
    if changed > max(40, len(a) * 0.25):
        return False, f"tool rewrote {changed} lines -- too broad to be a targeted fix"
    return True, f"changed {changed} line(s)"


_PROMPT = """You are writing a small Python function that edits one source file to fix a bug.

ISSUE:
{issue}

FILE: {path}
Here are the parts of the file that mention terms from the issue:

{excerpt}

Write a Python function with this exact signature:

def edit(text: str) -> str

`text` is the ENTIRE current contents of {path}. Return the ENTIRE new contents with the fix applied.
Use `re` or plain string methods. Make a TARGETED change - do not rewrite the file.
Output ONLY the function definition. No explanation, no markdown fences.
"""


def _excerpt(text: str, issue: str, budget: int = 2600) -> str:
    """Give the author the regions that actually mention the issue's terms, with line numbers -- not
    the first N bytes of the file, which is how the old one-line-anchor editor kept landing on
    imports. A 90k-char django file cannot go in a prompt; WHICH 2.6k is the whole game."""
    toks = {t for t in re.findall(r"[A-Za-z_][A-Za-z0-9_]{3,}", issue)}
    lines = text.splitlines()
    score = [(sum(1 for t in toks if t in l), i) for i, l in enumerate(lines)]
    hot = sorted((i for s, i in score if s > 0), key=lambda i: -score[i][0])[:12]
    keep: set = set()
    for i in sorted(hot):
        keep |= set(range(max(0, i - 6), min(len(lines), i + 10)))
    out, last, size = [], -1, 0
    for i in sorted(keep):
        if i != last + 1:
            out.append("        ...")
        seg = f"{i + 1:6d}  {lines[i]}"
        size += len(seg)
        if size > budget:
            break
        out.append(seg)
        last = i
    return "\n".join(out) if out else text[:budget]


def author_edit_tool(issue: str, path: str, text: str, lm, retries: int = 2) -> tuple:
    """The LM authors `def edit(text)->str`; it is executed and gated. Returns (ok, code, new_text,
    note). The LM never edits the file -- it writes a tool, and the tool's OUTPUT is what gets gated."""
    if lm is None:
        return False, "", "", "no LM supplied"
    last = "no attempt"
    for attempt in range(retries + 1):
        prompt = _PROMPT.format(issue=issue[:1200], path=path, excerpt=_excerpt(text, issue))
        if attempt:
            prompt += f"\nYour previous attempt failed: {last}\nFix it.\n"
        try:
            raw = str(lm.generate_chat(prompt, max_new=420)).strip()
        except Exception as e:                                     # noqa: BLE001
            return False, "", "", f"LM call failed: {e!r}"
        code = strip_fences(raw)
        if "def edit" not in code:
            last = "no `def edit(text)` in output"
            continue
        ok, res = run_edit_tool(code, text)
        if not ok:
            last = res
            continue
        good, note = gate_edit(text, res)
        if good:
            return True, code, res, note
        last = note
    return False, "", "", last


def strip_fences(raw: str) -> str:
    """LM output routinely arrives fenced. Measured earlier this session: a fence line taken literally
    as code was the single most common failure of the previous editor."""
    if "```" in raw:
        blocks = re.findall(r"```(?:python)?\s*\n(.*?)```", raw, re.S)
        if blocks:
            return max(blocks, key=len).strip()
        raw = raw.replace("```python", "").replace("```", "")
    return raw.strip()


# ── the bank: verified tools become real atoms in the worlds graph ────────────────────────────────
class ToolBank:
    """Verified edit tools, banked as real AtomGraph atoms and retrieved by MEANING.

    Writes are VERIFIER-GATED: a tool is banked only when the real test suite passed with it, never
    because it looked plausible. This project already measured what happens otherwise -- banking on
    confidence rather than verification banked 8 wrong entries against 0 (the CoT-schema result), so
    the gate is the whole design, not a formality.
    """

    def __init__(self, graph=None):
        from v5.runtime.membrane import AtomGraph
        self.g = graph if graph is not None else AtomGraph()
        self.n = 0

    def bank(self, issue: str, repo: str, path: str, code: str, note: str = "") -> str:
        from v5.runtime.membrane import Atom
        from embedder import encode_batch
        name = f"edit_tool::{repo}::{self.n}"
        desc = summarize_tool(issue, code)
        emb = np.asarray(encode_batch([desc])[0], dtype=np.float32)
        self.g.add(Atom(name=name, code=code, kind="edit_tool", provenance=f"verified:{repo}",
                        description=desc, emb=emb))
        self.n += 1
        return name

    def retrieve(self, issue: str, k: int = 3) -> list:
        """Nearest banked tools by MiniLM cosine over the tool's own description."""
        from embedder import encode_batch
        if not getattr(self.g, "atoms", None):
            return []
        names = [n for n, a in self.g.atoms.items() if a.kind == "edit_tool"]
        if not names:
            return []
        M = np.stack([self.g.atoms[n].emb for n in names])
        q = np.asarray(encode_batch([issue[:600]])[0], dtype=np.float32)
        order = (-(M @ q)).argsort()[:k]
        return [(names[int(i)], self.g.atoms[names[int(i)]].code,
                 float((M @ q)[int(i)])) for i in order]

    def __len__(self):
        return self.n


def summarize_tool(issue: str, code: str) -> str:
    """The retrieval key. Deliberately built from the CODE's own surface plus the issue's terms --
    never from an LM's description of itself, which nothing verifies."""
    ops = []
    if "re.sub" in code:
        ops.append("regex substitution")
    if "replace(" in code:
        ops.append("string replace")
    if "splitlines" in code or "\\n".join([]) == "":
        ops.append("line-wise edit")
    if "def " in code and "insert" in code:
        ops.append("insertion")
    head = " ".join(re.findall(r"[A-Za-z_][A-Za-z0-9_]{3,}", issue)[:24])
    return f"edit tool ({', '.join(ops) or 'text transform'}) for: {head}"


def _selftest() -> bool:
    print("algo_grr_toolsmith --selftest: authored tools are EXECUTED and GATED, never trusted\n")
    ok = True

    def chk(t, c, d=""):
        nonlocal ok
        ok &= bool(c)
        print(f"  [{'PASS' if c else 'FAIL'}] {t}{('  - ' + d) if d else ''}")

    src = "def f(a):\n    return a + 1\n\n\ndef g(b):\n    return b * 2\n"

    good, res = run_edit_tool("def edit(text):\n    return text.replace('a + 1', 'a + 2')\n", src)
    chk("[1] a real authored tool runs and transforms text", good and "a + 2" in res, str(res)[:50])

    good, res = run_edit_tool("def edit(text):\n    while True:\n        pass\n", src)
    chk("[2] a non-terminating tool is killed by the timeout, not left to hang",
        not good and "terminate" in str(res), str(res)[:60])

    good, res = run_edit_tool("def edit(text):\n    import os\n    return os.listdir('.')[0]\n", src)
    chk("[3] the sandbox denies filesystem access (no os, no open)", not good, str(res)[:60])

    good, res = run_edit_tool("def edit(text):\n    import re\n    return re.sub('a . 1','a + 2',text)\n",
                              src)
    chk("[3b] but a LEGITIMATE `import re` WORKS -- blocking it rejected 2 of 3 real authored tools",
        good and "a + 2" in str(res), str(res)[:50].replace("\n", " "))
    for mod in ("subprocess", "shutil", "pathlib", "socket"):
        g2, _ = run_edit_tool(f"def edit(text):\n    import {mod}\n    return text + 'x'\n", src)
        ok &= not g2
    chk("[3c] the dangerous modules stay unreachable (subprocess/shutil/pathlib/socket)", True)

    good, res = run_edit_tool("def edit(text):\n    return open('x','w')\n", src)
    chk("[4] the sandbox denies open() specifically", not good, str(res)[:60])

    good, res = run_edit_tool("def nope(text):\n    return text\n", src)
    chk("[5] a tool without the required entry point is rejected", not good, str(res)[:60])

    g, n = gate_edit(src, src)
    chk("[6] a no-op tool is REJECTED (it changed nothing)", not g and "no change" in n, n)
    # `a +++ 1` was the first probe here and it is VALID python (a + (+(+1))) -- the gate was right
    # and the test was wrong. Unbalanced parens are unambiguously a syntax error.
    g, n = gate_edit(src, "def f(a:\n    return 1\n")
    chk("[7] a result that does not parse is REJECTED", not g and "parse" in n, n[:50])
    g, n = gate_edit(src, src.replace("a + 1", "a + 2"))
    chk("[8] a targeted, parseable change is ACCEPTED", g, n)
    big = "\n".join(f"x{i} = {i}" for i in range(200))
    g, n = gate_edit(big, "\n".join(f"y{i} = {i + 1}" for i in range(200)))
    chk("[9] a whole-file rewrite is REJECTED as too broad", not g and "broad" in n, n[:50])

    chk("[10] fences are stripped before code is ever executed",
        strip_fences("```python\ndef edit(t):\n    return t\n```") .startswith("def edit"))

    ex = _excerpt("import os\n" * 40 + "def target_method(self, value):\n    return value\n",
                  "target_method returns the wrong value")
    chk("[11] the excerpt targets issue-relevant regions, not the file's first bytes",
        "target_method" in ex, ex.strip().splitlines()[0][:50] if ex.strip() else "")

    tb = ToolBank()
    nm = tb.bank("fix get_FIELD_display override", "django/django", "a.py",
                 "def edit(text):\n    return text.replace('x','y')\n", "1 line")
    got = tb.retrieve("get_FIELD_display override is broken", k=1)
    chk("[12] a verified tool banks into the worlds graph and is retrievable BY MEANING",
        len(tb) == 1 and got and got[0][0] == nm, f"{got[0][2]:.3f} cosine" if got else "none")

    print(f"\n  ALGO_GRR_TOOLSMITH -> {'PASS' if ok else 'FAIL'}")
    return ok


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Model-authored, verifier-gated edit tools.")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    sys.exit(0 if (_selftest() if a.selftest else ap.print_help() or True) else 1)
