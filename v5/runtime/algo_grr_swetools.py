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

import json
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
_TESTS_JSON = Path(_ROOT) / "artifacts" / "swebench_tests.json"
_TESTS: dict | None = None
# django's test id in FAIL_TO_PASS reads "test_name (dotted.Class)"; its runner wants dotted.Class.test_name
_DJ_ID = re.compile(r"^(\S+)\s+\(([^)]+)\)\s*$")


def _repo_dir(repo: str) -> Path:
    return Path(SRC_ROOT) / repo.replace("/", "_")


def _clip(s: str) -> str:
    s = s or ""
    return s if len(s) <= _MAX_OBS else s[:_MAX_OBS] + f"\n...[{len(s) - _MAX_OBS} more chars]"


def instance_tests(instance_id: str) -> dict:
    """The REAL FAIL_TO_PASS / PASS_TO_PASS directives SWE-bench ships, from the cached dataset
    (scripts/prep_swe_tests.py extracts them). Empty dict if unavailable -- callers must treat that as
    'cannot verify', never as 'passed'."""
    global _TESTS
    if _TESTS is None:
        try:
            _TESTS = json.loads(_TESTS_JSON.read_text(encoding="utf-8"))
        except Exception:                                          # noqa: BLE001
            _TESTS = {}
    return _TESTS.get(instance_id, {})


def test_command(instance_id: str, repo: str = "") -> str:
    """Build the instance's REAL test command from its REAL FAIL_TO_PASS ids.

    Not a guess at a runner: `python -m pytest` was the previous default and is simply wrong for
    django, which ships no pytest at all (measured live: "No module named pytest"). django uses
    tests/runtests.py with dotted test paths; the rest of these repos are pytest projects.
    Returns "" when the directives are unknown -- the caller must then refuse to claim a pass.
    """
    meta = instance_tests(instance_id)
    f2p = meta.get("FAIL_TO_PASS") or []
    if not f2p:
        return ""
    repo = repo or meta.get("repo", "")
    if repo == "django/django":
        ids = []
        for t in f2p:
            m = _DJ_ID.match(t.strip())
            ids.append(f"{m.group(2)}.{m.group(1)}" if m else t.strip())
        return ("./tests/runtests.py --verbosity 2 --settings=test_sqlite --parallel 1 "
                + " ".join(ids))
    return "python -m pytest --no-header -rA -p no:cacheprovider " + " ".join(
        f"'{t}'" for t in f2p)


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


_CFILE_CACHE: dict = {}


def container_file_text(instance_id: str, path: str) -> str:
    """The file EXACTLY as the container has it, at the instance's base commit.

    This exists because the local checkout at E:/swebench_src is at HEAD while every container is
    pinned to its instance's base commit, and the gap is not cosmetic: on django__django-11133 the
    model proposed `        if isinstance(value, bytes):` -- the REAL gold anchor, verified to pass in
    the container -- and the local pre-filter rejected it as "not in the file verbatim" because HEAD
    had since changed that line. The harness was throwing away correct fixes. Anything that shows the
    model code, or checks the model's `old` string, must read from HERE, not from the local clone.
    """
    key = (instance_id, path)
    if key in _CFILE_CACHE:
        return _CFILE_CACHE[key]
    img = f"swebench/sweb.eval.x86_64.{instance_id.replace('__', '_1776_')}:latest"
    try:
        # binary pipe + explicit utf-8 decode. Letting subprocess decode with the Windows default
        # (cp1252) crashed on a real source file containing a 0x9d byte, losing an instance outright:
        # "UnicodeDecodeError: 'charmap' codec can't decode byte 0x9d".
        r = subprocess.run(["wsl", "-d", os.environ.get("WSL_DISTRO", "UbuntuE"),
                            "docker", "run", "--rm", "--pull", "never", "--entrypoint", "bash", img,
                            "-lc", f"cat /testbed/{path}"],
                           capture_output=True, timeout=300)
        txt = r.stdout.decode("utf-8", errors="replace") if r.returncode == 0 else ""
    except Exception:                                              # noqa: BLE001
        txt = ""
    _CFILE_CACHE[key] = txt
    return txt


def t_run_tests(state, arg):
    """THE verifier: run the instance's real tests in its SWE-bench container. Everything else in
    this file is navigation; this is the only step that can say the work was right."""
    inst = state.get("instance_id")
    if not inst:
        return False, "no instance_id in state"
    if not state.get("patch") and not state.get("patch_tool"):
        return False, "no patch to test"
    img = f"swebench/sweb.eval.x86_64.{inst.replace('__', '_1776_')}:latest"
    path, old, new = state.get("patch") or (state["patch_tool"][0], "", "")
    # `set -o pipefail`: without it, `cmd | tail -40`'s exit status is tail's (almost always 0),
    # not the test command's -- a real bug found live: django-11999 exited "successfully" from a
    # command that was actually `No module named pytest` (django doesn't ship pytest; it uses
    # tests/runtests.py), and the exit code alone said nothing was wrong.
    # the REAL command from the instance's REAL FAIL_TO_PASS directives; `arg` may override for
    # debugging, but is never controller-authored in the agent loop.
    cmd = arg or test_command(inst, state.get("repo", ""))
    if not cmd:
        return False, (f"no FAIL_TO_PASS directives known for {inst} -- cannot verify. "
                       f"(run scripts/prep_swe_tests.py)")
    # SWE-bench protocol, and it is NOT optional: the FAIL_TO_PASS tests are ADDED by test_patch.
    # Skipping it made even the REAL GOLD PATCH score as a failure ("type object
    # 'GetFieldDisplayTests' has no attribute 'test_overriding_FIELD_display'") -- i.e. the verifier
    # was structurally incapable of ever reporting a pass. Caught by a positive control, not by
    # reading the code. test_patch touches ONLY test files, never the source under repair, so it
    # cannot leak the fix.
    tp = instance_tests(inst).get("test_patch") or ""
    apply_tp = ""
    if tp:
        apply_tp = ("cd /testbed && git checkout -- . && cat > /tmp/test.patch <<'TESTPATCH_EOF'\n"
                    + tp + "\nTESTPATCH_EOF\ngit apply -v /tmp/test.patch && ")
    if state.get("patch_tool"):
        # SHIP THE TOOL, NOT THE TEXT. Writing the model's whole rewritten file into the container
        # was unsound: `open_text` comes from the local HEAD checkout while the container is pinned at
        # the instance's BASE COMMIT, so overwriting produced a file from the wrong django version
        # ("ImportError: cannot import name '_lazy_re_compile'"). Running the authored tool INSIDE the
        # container against the container's OWN file removes the version mismatch entirely -- and it
        # is what "the model authored an edit tool" should mean anyway: the tool is the artifact.
        # base64 for both payloads so no file content or code is ever parsed by the shell (a heredoc
        # here previously executed a line of response.py as a command: "filelike: command not found").
        import base64
        fpath, code = state["patch_tool"]
        runner = (
            "import sys, re, builtins, base64\n"
            "code = base64.b64decode(sys.argv[1]).decode('utf-8')\n"
            "p = sys.argv[2]\n"
            "src = open(p, encoding='utf-8').read()\n"
            "ns = {'re': re}\n"
            "exec(code, ns)\n"
            "out = ns['edit'](src)\n"
            "assert isinstance(out, str) and out.strip() and out != src, 'tool made no valid change'\n"
            "open(p, 'w', encoding='utf-8').write(out)\n"
            "print('TOOL_APPLIED_IN_CONTAINER')\n")
        rb64 = base64.b64encode(runner.encode("utf-8")).decode("ascii")
        cb64 = base64.b64encode(code.encode("utf-8")).decode("ascii")
        write = (f"cd /testbed && echo {rb64} | base64 -d > /tmp/_apply.py && "
                 f"python /tmp/_apply.py {cb64} {fpath} && ")
    else:
        write = (f"cd /testbed && python - <<'EOF'\n"
                 f"import io\np={path!r}\ns=open(p,encoding='utf-8').read()\n"
                 f"s=s.replace({old!r},{new!r})\nopen(p,'w',encoding='utf-8').write(s)\nEOF\n")
    script = (apply_tp + write
              + f"cd /testbed && set -o pipefail && ({cmd}) 2>&1 | tail -40")
    try:
        # --pull=never: fail fast on an uncached image instead of silently pulling several GB. Only a
        # handful of images are cached locally (checked before this was added: 3 django instances) --
        # without this flag, calling this tool on any other instance would trigger a multi-GB download.
        # decode explicitly as utf-8; the Windows default (cp1252) raises on real test output
        # containing non-cp1252 bytes and would turn a genuine result into a crash.
        r = subprocess.run(["wsl", "-d", os.environ.get("WSL_DISTRO", "UbuntuE"),
                            "docker", "run", "--rm", "--pull", "never", "--entrypoint", "bash", img,
                            "-lc", script],
                           capture_output=True, timeout=900)
        out = (r.stdout.decode("utf-8", errors="replace")
               + r.stderr.decode("utf-8", errors="replace"))
    except Exception as e:                                     # noqa: BLE001
        return False, f"test run failed to start: {e!r}"
    # exit code is the primary signal now that pipefail makes it trustworthy. The text markers are a
    # second, independent gate -- a real false positive was found where the process could plausibly
    # exit 0 from a no-op (e.g. an empty test selection) with no real pass evidence at all; " passed"
    # must actually be present, and "no module named"/"command not found" hard-fail regardless of
    # exit code, since those mean the command never ran as intended.
    low = out.lower()
    hard_fail = ("no module named" in low or "command not found" in low
                 or "no such file or directory" in low)
    # two runners, two success markers: pytest says "N passed", django's tests/runtests.py is
    # unittest-style and says "OK" on its own line after "Ran N tests". Requiring pytest's marker
    # alone would false-NEGATIVE every django instance, the mirror of the false-positive already
    # fixed. Scan ALL lines, not just the last: `git apply -v`'s stderr interleaves AFTER the test
    # output, so the final line is routinely "Applied patch ... cleanly." -- checking only the last
    # line made a genuinely passing gold patch (rc=0) report as a failure.
    passed = (" passed" in low) or re.search(r"^OK\b", out, re.M) is not None
    ok = (r.returncode == 0) and passed and not hard_fail
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

    # the verifier's command construction, checked without Docker (the full ground-truth calibration
    # is --calibrate, which needs the images). django is the case that matters: it ships NO pytest.
    cmd = test_command("django__django-11999", "django/django")
    chk("[9] django's test command uses runtests.py with a converted dotted id, NOT pytest",
        "runtests.py" in cmd and "pytest" not in cmd
        and "model_fields.tests.GetFieldDisplayTests.test_overriding_FIELD_display" in cmd,
        cmd[:100])
    chk("[10] test_patch is available -- FAIL_TO_PASS tests do not EXIST without it",
        bool(instance_tests("django__django-11999").get("test_patch")))
    chk("[11] an unknown instance yields NO command, so the caller cannot claim a pass",
        test_command("not__a-real-instance", "x/y") == "")

    print(f"\n  MEMBRANE_SWETOOLS -> {'PASS' if ok else 'FAIL'}")
    return ok


def _calibrate() -> bool:
    """GROUND-TRUTH calibration of the verifier: for each locally cached image, a no-op edit must
    FAIL the instance's real FAIL_TO_PASS test and the REAL GOLD PATCH must PASS it.

    This is the check that actually matters, and it caught three real bugs that code review did not:
    a false PASS from a command that never ran, a missing test_patch that made even the gold patch
    unpassable, and a marker scan that only read the last line (which is git-apply stderr). A verifier
    that cannot separate a no-op from the true fix produces confident, wrong numbers.
    Needs the SWE-bench Docker images; run with --calibrate.
    """
    import subprocess as _sp
    cached = ["django__django-10924", "django__django-11133", "django__django-11999"]
    print("algo_grr_swetools --calibrate: verifier vs ground truth (needs cached Docker images)\n")
    ok = True
    for inst in cached:
        meta = instance_tests(inst)
        cmd, tp = test_command(inst, "django/django"), meta.get("test_patch") or ""
        if not cmd or not tp:
            print(f"  [SKIP] {inst}: no directives/test_patch")
            continue
        st = {"repo": "django/django", "instance_id": inst,
              "patch": ("django/__init__.py", "from django.utils.version import get_version",
                         "from django.utils.version import get_version")}
        neg, _ = t_run_tests(st, None)
        img = f"swebench/sweb.eval.x86_64.{inst.replace('__', '_1776_')}:latest"
        gold = meta.get("gold_patch") or _gold_patch(inst)
        pos = False
        if gold:
            script = ("cd /testbed && git checkout -- . && cat > /tmp/t.patch <<'TP'\n" + tp
                      + "\nTP\ngit apply -v /tmp/t.patch && cat > /tmp/g.patch <<'GP'\n" + gold
                      + "\nGP\ngit apply -v /tmp/g.patch && "
                      + f"set -o pipefail && ({cmd}) 2>&1 | tail -25")
            r = _sp.run(["wsl", "-d", os.environ.get("WSL_DISTRO", "UbuntuE"), "docker", "run",
                         "--rm", "--pull", "never", "--entrypoint", "bash", img, "-lc", script],
                        capture_output=True, timeout=900)
            o = (r.stdout.decode("utf-8", errors="replace")
                 + r.stderr.decode("utf-8", errors="replace"))
            pos = (r.returncode == 0) and ((" passed" in o.lower())
                                            or re.search(r"^OK\b", o, re.M) is not None)
        good = (not neg) and pos
        ok &= good
        print(f"  [{'PASS' if good else 'FAIL'}] {inst}: no-op edit -> "
              f"{'FAIL' if not neg else 'PASS(BAD)'}, gold patch -> {'PASS' if pos else 'FAIL(BAD)'}")
    print(f"\n  SWETOOLS_CALIBRATE -> {'PASS' if ok else 'FAIL'}")
    return ok


def _gold_patch(inst: str) -> str:
    """The instance's real reference patch, read from the cached SWE-bench parquet. Used ONLY by
    --calibrate as a positive control -- never by any tool the agent can call."""
    try:
        import pyarrow.parquet as pq
        p = (Path(r"E:\cache\hf\hub\datasets--princeton-nlp--SWE-bench\snapshots")
             / "e48e2bd1e9fecd5bbd641e9414ac59da9f2e69f6" / "data" / "test-00000-of-00001.parquet")
        t = pq.read_table(p).select(["instance_id", "patch"]).to_pydict()
        return dict(zip(t["instance_id"], t["patch"])).get(inst, "")
    except Exception:                                              # noqa: BLE001
        return ""


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Real SWE tools for the agentic loop.")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--calibrate", action="store_true",
                    help="verify the test verifier against ground truth (needs Docker images)")
    a = ap.parse_args()
    if a.calibrate:
        sys.exit(0 if _calibrate() else 1)
    sys.exit(0 if (_selftest() if a.selftest else ap.print_help() or True) else 1)
