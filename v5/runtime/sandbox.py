"""Local test sandbox for the v3 agent loop — millisecond verification, no Docker.

Runs candidate code + assert-style tests in ONE `python -I` subprocess (isolated mode:
no user site, no env inheritance), tempdir cwd, hard timeout. Per-assert granularity via
an exec harness appended to the candidate module; a sentinel line carries the result out.

SECURITY NOTE: this is for TRUSTED local task suites (MBPP + our own mutants). It is
process isolation + timeout, not a jail — do not feed untrusted code.

Result schema (always all keys):
  {passed, n_pass, n_total, first_fail, stderr_tail, dur_ms}

  python -m v5.runtime.sandbox --selftest
"""
from __future__ import annotations

import subprocess
import sys
import tempfile
import time
from pathlib import Path

SENTINEL = "SBX_RESULT"


def _harness(tests: list[str], setup: str = "") -> str:
    """Appended below the candidate code: run each assert, count passes, report first fail."""
    lines = [
        "",
        "if True:",
        "    import traceback as _tb",
        f"    _setup = {setup!r}",
        f"    _tests = {tests!r}",
        "    _p, _ff = 0, ''",
        "    try:",
        "        if _setup: exec(compile(_setup, '<setup>', 'exec'), globals())",
        "    except Exception as _e:",
        "        _ff = 'setup: %s: %s' % (type(_e).__name__, _e)",
        "    if not _ff:",
        "        for _i, _src in enumerate(_tests):",
        "            try:",
        "                exec(compile(_src, '<test%d>' % _i, 'exec'), globals())",
        "                _p += 1",
        "            except Exception as _e:",
        "                if not _ff:",
        "                    _ff = 'test%d: %s: %s' % (_i, type(_e).__name__, _e)",
        f"    print('\\n' + {SENTINEL!r}, _p, len(_tests), repr(_ff[:200]))",
    ]
    return "\n".join(lines)


def run(code: str, tests: list[str], setup: str = "", timeout: float = 5.0) -> dict:
    res = {"passed": False, "n_pass": 0, "n_total": len(tests), "first_fail": "",
           "stderr_tail": "", "dur_ms": 0}
    t0 = time.time()
    with tempfile.TemporaryDirectory() as td:
        script = Path(td) / "cand.py"
        script.write_text((code or "") + "\n" + _harness(tests, setup), encoding="utf-8")
        try:
            proc = subprocess.run(
                [sys.executable, "-I", str(script)], cwd=td, capture_output=True,
                text=True, encoding="utf-8", errors="replace", timeout=timeout)
        except subprocess.TimeoutExpired as exc:
            res["first_fail"] = "timeout"
            res["stderr_tail"] = (exc.stderr or "")[-400:] if isinstance(exc.stderr, str) else ""
            res["dur_ms"] = int((time.time() - t0) * 1000)
            return res
    res["dur_ms"] = int((time.time() - t0) * 1000)
    res["stderr_tail"] = (proc.stderr or "")[-400:]
    line = next((ln for ln in reversed((proc.stdout or "").splitlines())
                 if ln.startswith(SENTINEL)), "")
    if not line:                                        # crashed before the harness ran
        res["first_fail"] = f"crash: exit {proc.returncode}"
        return res
    try:
        _, p, tot, ff_repr = line.split(" ", 3)
        res["n_pass"], res["n_total"] = int(p), int(tot)
        import ast
        res["first_fail"] = ast.literal_eval(ff_repr)
    except Exception:                                   # noqa: BLE001
        res["first_fail"] = "sentinel parse error"
        return res
    res["passed"] = res["n_pass"] == res["n_total"] and res["n_total"] > 0 \
        and not res["first_fail"]
    return res


def obs_text(res: dict) -> str:
    """The failure observation fed back into the retry prompt's trace slot (DATA)."""
    if res["passed"]:
        return ""
    parts = [f"{res['n_pass']}/{res['n_total']} tests pass"]
    if res["first_fail"]:
        parts.append(res["first_fail"])
    if res["stderr_tail"] and "crash" in (res["first_fail"] or ""):
        parts.append(res["stderr_tail"][-200:])
    return " | ".join(parts)[:280]


# ── selftest ────────────────────────────────────────────────────────────────────

def _selftest() -> bool:
    print("sandbox --selftest: pass / partial fail / crash / timeout / unicode (this machine)\n")
    ok = run("def add(a, b):\n    return a + b\n", ["assert add(1, 2) == 3", "assert add(0, 0) == 0"])
    assert ok["passed"] and ok["n_pass"] == 2 and ok["first_fail"] == "", ok
    print(f"  [1] pass ({ok['dur_ms']}ms) -> PASS")

    part = run("def add(a, b):\n    return a - b\n", ["assert add(1, 0) == 1", "assert add(1, 2) == 3"])
    assert not part["passed"] and part["n_pass"] == 1 and "test1" in part["first_fail"], part
    print("  [2] partial fail + first_fail -> PASS")

    crash = run("def broken(:\n", ["assert True"])
    assert not crash["passed"] and "crash" in crash["first_fail"] and crash["stderr_tail"], crash
    print("  [3] syntax crash + stderr tail -> PASS")

    tmo = run("while True:\n    pass\n", ["assert True"], timeout=2)
    assert not tmo["passed"] and tmo["first_fail"] == "timeout" and tmo["dur_ms"] >= 1900, tmo
    print(f"  [4] timeout ({tmo['dur_ms']}ms) -> PASS")

    uni = run("def greet():\n    return 'héllo — ünïcode'\n", ["assert 'héllo' in greet()"])
    assert uni["passed"], uni
    o = obs_text(part)
    assert "1/2" in o and "test1" in o
    print("  [5] unicode + obs_text -> PASS")

    print("\n  SANDBOX SELFTEST -> PASS")
    return True


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    if ap.parse_args().selftest:
        raise SystemExit(0 if _selftest() else 1)
    ap.print_help()
