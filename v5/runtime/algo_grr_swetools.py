"""algo_grr_swetools — REAL agentic tools over a REAL repo, for algo_grr_agent's loop.

algo_grr_agent already implements the loop this project argues for ("TRM CALLS graph tools, the LM
only SPEAKS") and its own docstring names the missing step: "Deployment: swap the arithmetic
ToolRegistry for real agentic tools (search / call_api / run_code) -- the loop is identical; only the
tool set changes." That swap is this file. Nothing here re-implements the loop.

WHERE THE LM ACTUALLY GOES, since this keeps being ambiguous:
  - WHICH TOOL to call is a discrete choice over a registry. No decoding, no parsing, no
    hallucinated tool names. The controller/TRM picks it.
  - The LM decodes only FREE-TEXT ARGUMENTS that nothing else can produce -- the replacement code in
    an edit, a search string. That is the DCPD split: the structure is exact, the LM fills holes.
  - The LM never decides whether a step succeeded. Tools return real observations and the verifier is
    real test execution.

STATE is a dict threaded through the tools, so an observation from one tool is visible to the next --
which is the property every ablation in this project kept finding absent.
"""
from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

_ROOT = str(Path(__file__).resolve().parents[2])
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

SRC_ROOT = os.environ.get("SWE_SRC", r"E:\swebench_src")
_MAX_OBS = 4000


def _repo_dir(repo: str) -> Path:
    return Path(SRC_ROOT) / repo.replace("/", "_")


def _clip(s: str) -> str:
    s = s or ""
    return s if len(s) <= _MAX_OBS else s[:_MAX_OBS] + f"\n...[{len(s) - _MAX_OBS} more chars]"


# ── the tools. every one returns (ok, observation_text) and MUTATES state ────────────────────────
def t_list_dir(state, arg):
    d = _repo_dir(state["repo"]) / (arg or "")
    if not d.is_dir():
        return False, f"not a directory: {arg}"
    names = sorted(p.name + ("/" if p.is_dir() else "") for p in d.iterdir())
    return True, _clip(" ".join(names[:400]))


def t_read_file(state, arg):
    p = _repo_dir(state["repo"]) / (arg or "")
    if not p.is_file():
        return False, f"no such file: {arg}"
    txt = p.read_text(encoding="utf-8", errors="ignore")
    state["open_file"] = str(arg)
    state["open_text"] = txt
    return True, _clip(txt[:_MAX_OBS])


def t_grep(state, arg):
    """Search the repo for a literal string. This is the tool that turns 'the issue quotes an error
    message' into 'the one file that defines it' -- measured earlier: 39.5% of issues name a symbol
    unique to exactly one file."""
    root = _repo_dir(state["repo"])
    if not arg:
        return False, "empty pattern"
    hits = []
    for p in root.rglob("*.py"):
        try:
            txt = p.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        if arg in txt:
            ln = txt[:txt.index(arg)].count("\n") + 1
            hits.append(f"{p.relative_to(root).as_posix()}:{ln}")
            if len(hits) >= 40:
                break
    state["last_grep"] = hits
    return bool(hits), _clip("\n".join(hits) or "no matches")


def t_find_def(state, arg):
    """Locate where a symbol is DEFINED (not merely mentioned) -- the difference between 'this file
    uses it' and 'this file owns it'."""
    root = _repo_dir(state["repo"])
    pat = re.compile(rf"^[ \t]*(?:async +)?(?:def|class)[ \t]+{re.escape(arg or '')}\b", re.M)
    hits = []
    for p in root.rglob("*.py"):
        try:
            txt = p.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        m = pat.search(txt)
        if m:
            hits.append(f"{p.relative_to(root).as_posix()}:{txt[:m.start()].count(chr(10)) + 1}")
    return bool(hits), _clip("\n".join(hits[:40]) or f"no definition of {arg}")


def t_edit(state, arg):
    """Apply a replacement to the open file. `arg` is (old, new) -- `new` is the ONE thing here the LM
    produces, and it is checked: the edit must apply uniquely and the result must still PARSE, so a
    syntactically broken patch is rejected by the tool rather than by a test run later."""
    import ast
    if not state.get("open_file"):
        return False, "no file open; read_file first"
    old, new = (arg if isinstance(arg, (tuple, list)) and len(arg) == 2 else (None, None))
    if old is None:
        return False, "edit needs (old, new)"
    txt = state["open_text"]
    n = txt.count(old)
    if n == 0:
        return False, "old text not found"
    if n > 1:
        return False, f"old text is ambiguous ({n} occurrences)"
    cand = txt.replace(old, new)
    try:
        ast.parse(cand)
    except SyntaxError as e:
        return False, f"edit would not parse: {e}"
    state["patched_text"] = cand
    state["patch"] = (state["open_file"], old, new)
    return True, f"edit applies cleanly to {state['open_file']} and still parses"


def t_run_tests(state, arg):
    """THE verifier: run the instance's real tests in its SWE-bench container. Everything else in
    this file is navigation; this is the only step that can say the work was right."""
    inst = state.get("instance_id")
    if not inst:
        return False, "no instance_id in state"
    if not state.get("patch"):
        return False, "no patch to test"
    img = f"swebench/sweb.eval.x86_64.{inst.replace('__', '_1776_')}:latest"
    path, old, new = state["patch"]
    # `set -o pipefail`: without it, `cmd | tail -40`'s exit status is tail's (almost always 0),
    # not the test command's -- a real bug found live: django-11999 exited "successfully" from a
    # command that was actually `No module named pytest` (django doesn't ship pytest; it uses
    # tests/runtests.py), and the exit code alone said nothing was wrong.
    script = (f"cd /testbed && python - <<'EOF'\n"
              f"import io\np={path!r}\ns=open(p,encoding='utf-8').read()\n"
              f"s=s.replace({old!r},{new!r})\nopen(p,'w',encoding='utf-8').write(s)\nEOF\n"
              f"cd /testbed && set -o pipefail && ({arg or 'python -m pytest -x -q'}) 2>&1 | tail -40")
    try:
        # --pull=never: fail fast on an uncached image instead of silently pulling several GB. Only a
        # handful of images are cached locally (checked before this was added: 3 django instances) --
        # without this flag, calling this tool on any other instance would trigger a multi-GB download.
        r = subprocess.run(["wsl", "-d", os.environ.get("WSL_DISTRO", "UbuntuE"),
                            "docker", "run", "--rm", "--pull", "never", "--entrypoint", "bash", img,
                            "-lc", script],
                           capture_output=True, text=True, timeout=900)
        out = (r.stdout or "") + (r.stderr or "")
    except Exception as e:                                     # noqa: BLE001
        return False, f"test run failed to start: {e!r}"
    # exit code is the primary signal now that pipefail makes it trustworthy. The text markers are a
    # second, independent gate -- a real false positive was found where the process could plausibly
    # exit 0 from a no-op (e.g. an empty test selection) with no real pass evidence at all; " passed"
    # must actually be present, and "no module named"/"command not found" hard-fail regardless of
    # exit code, since those mean the command never ran as intended.
    low = out.lower()
    hard_fail = "no module named" in low or "command not found" in low
    ok = (r.returncode == 0) and (" passed" in out) and not hard_fail
    state["last_tests"] = out[-2000:]
    state["test_returncode"] = r.returncode
    return ok, _clip(out)


def swe_registry():
    """A ToolRegistry of the real tools. Same shape algo_grr_agent already consumes."""
    from v5.runtime.algo_grr_agent import ToolRegistry
    reg = ToolRegistry()
    reg.add("list_dir", t_list_dir, "list a directory in the repo")
    reg.add("grep", t_grep, "find files containing a literal string")
    reg.add("find_def", t_find_def, "find where a symbol is defined")
    reg.add("read_file", t_read_file, "open a file and read its text")
    reg.add("edit", t_edit, "apply a (old,new) replacement to the open file; must parse")
    reg.add("run_tests", t_run_tests, "run the instance's real tests in its SWE-bench container")
    return reg


def _selftest() -> bool:
    """Real tools on the real django checkout. No LM, no Docker -- those are separate arms."""
    print("algo_grr_swetools --selftest: real tools over a real repo\n")
    ok = True

    def chk(t, c, d=""):
        nonlocal ok
        ok &= bool(c)
        print(f"  [{'PASS' if c else 'FAIL'}] {t}{('  - ' + d) if d else ''}")

    st = {"repo": "django/django", "instance_id": "django__django-11999"}
    d = _repo_dir(st["repo"])
    chk("[1] real repo checkout present", d.is_dir(), str(d))
    if not d.is_dir():
        print("\n  MEMBRANE_SWETOOLS -> FAIL (no checkout)")
        return False

    good, obs = t_list_dir(st, "django/db/models")
    chk("[2] list_dir returns real entries", good and "query.py" in obs, obs[:60])

    # QuerySet is a real top-level class. The first probe here used get_FIELD_display, which django
    # generates dynamically rather than defining -- the tool was right and the test was wrong.
    good, obs = t_find_def(st, "QuerySet")
    chk("[3] find_def locates a real definition site (not a mere mention)",
        good and "query.py" in obs, obs.splitlines()[0] if good else obs)

    good, obs = t_read_file(st, "django/db/models/query.py")
    chk("[4] read_file opens a file and puts it in STATE",
        good and st.get("open_file") and len(st.get("open_text", "")) > 1000,
        f"{len(st.get('open_text',''))} chars in state")

    bad, obs = t_edit(st, ("def __init__(self", "def __init__(self"))
    chk("[5] edit REJECTS an ambiguous match rather than guessing",
        not bad and "ambiguous" in obs, obs[:60])

    txt = st["open_text"]
    uniq = "class QuerySet"
    good, obs = t_edit(st, (uniq, uniq)) if txt.count(uniq) == 1 else (False, "not unique")
    chk("[6] edit accepts a unique match and re-parses the result",
        good and st.get("patch") is not None, obs[:70])

    bad, obs = t_edit(st, (uniq, "class QuerySet(:::"))
    chk("[7] edit REJECTS a patch that would not parse (syntax gate, not a test run)",
        not bad and "not parse" in obs, obs[:60])

    reg = swe_registry()
    chk("[8] registry exposes the real tools to algo_grr_agent's loop",
        set(reg) == {"list_dir", "grep", "find_def", "read_file", "edit", "run_tests"},
        " ".join(sorted(reg)))

    print(f"\n  MEMBRANE_SWETOOLS -> {'PASS' if ok else 'FAIL'}")
    return ok


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Real SWE tools for the agentic loop.")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    sys.exit(0 if (_selftest() if a.selftest else ap.print_help() or True) else 1)
