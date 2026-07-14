"""GRR-13b: MBPP+ preprocessing — the HARDER, OPEN-SOURCE dataset, preprocessed and validated FIRST.

Why MBPP+ (EvalPlus) and not plain MBPP: our whole epistemics rides on the GENERALITY gate (GRR-1 —
correct on many inputs, not just the benchmark). Plain MBPP ships 3 asserts per problem: a gate that
weak invites exactly the `if x==17: return 4` overfit the store-gate exists to kill. MBPP+ augments
each problem with ~35x test inputs — dense enough for the gate to mean something on real data.

The pipeline (all offline after the one download):
  load    HF `evalplus/mbppplus` (fallback: google-research-datasets/mbpp sanitized, flagged weak)
  norm    defensive field mapping -> {name (entry fn), text (the human prompt), asserts (original),
          plus_test (the augmented test script), n_plus (assert-density estimate)}
  VALIDATE the reference solution must pass BOTH the original asserts and the plus script in an
          isolated subprocess (drops broken/underspecified records — no silent junk enters the corpus)
  cache   artifacts/mbpp_plus_prepped.jsonl + a stats report (task counts, assert density, and the
          heuristic split: pipeline-shaped candidates vs LM-author-territory)

Consumers: the GRR ladder's LM-author rung (step 2) trains/validates on load_prepped(); MBPPTask's
verify (subprocess asserts) is the gate.

  selftest (offline):  python -m v5.runtime.algo_mbpp_prep --selftest
  molab (downloads):   python -m v5.runtime.algo_mbpp_prep --prep --out artifacts/mbpp_plus_prepped.jsonl
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

from v5.runtime.algo_graph_run import MBPPTask, verify_asserts


def _entry_name(rec: dict) -> str:
    """Entry-point fn. PRIORITY: a name that is BOTH defined in the reference code AND called in the
    asserts (raw-dump bug: `assert set(similar_elements(...))` extracted `set` — a wrapper builtin —
    so the prompt asked the model to write set(...) while the tests called similar_elements)."""
    defs = re.findall(r"def\s+([A-Za-z_]\w*)\s*\(", rec.get("code") or "")
    called = []
    for t in rec.get("asserts") or []:
        called += re.findall(r"([A-Za-z_]\w*)\s*\(", t)
    import builtins
    blt = set(dir(builtins))
    for name in called:                              # first assert-called name that the reference DEFINES
        if name in defs:
            return name
    for name in called:                              # else: first non-builtin called name
        if name not in blt and name != "assert":
            return name
    return defs[-1] if defs else ""


def normalize(raw: dict) -> dict | None:
    """Defensive field mapping across mbppplus/mbpp schemas -> the prepped record shape (or None)."""
    text = (raw.get("prompt") or raw.get("text") or "").strip()
    code = (raw.get("code") or "").strip()
    asserts = list(raw.get("test_list") or [])
    plus = (raw.get("test") or "").strip()          # EvalPlus: a script with dense assertion checks
    setup = "\n".join(raw.get("test_imports") or [])
    if not text or not code or not (asserts or plus):
        return None
    rec = {"text": text, "code": code, "asserts": asserts, "plus_test": plus, "setup": setup,
           "n_plus": plus.count("assert")}
    rec["name"] = _entry_name(rec)
    if not rec["name"]:
        return None
    return rec


def validate(rec: dict, timeout: float = 20.0) -> bool:
    """The soundness gate: the REFERENCE solution must pass the original asserts AND the plus script.
    A record whose own reference fails is broken/underspecified -> dropped, never enters the corpus."""
    tests = list(rec["asserts"]) + ([rec["plus_test"]] if rec["plus_test"] else [])
    return verify_asserts(rec["code"], tests, setup=rec.get("setup", ""), timeout=timeout)


def _pipeline_shaped(rec: dict) -> bool:
    """Heuristic split for the report: single-argument function whose reference reads like a
    list-transform (the DSL rung's candidates); everything else is LM-author territory."""
    m = re.search(rf"def\s+{re.escape(rec['name'])}\s*\(([^)]*)\)", rec["code"])
    if not m:
        return False
    args = [a.strip() for a in m.group(1).split(",") if a.strip() and a.strip() != "self"]
    body = rec["code"]
    return len(args) == 1 and any(k in body for k in ("for ", "sum(", "max(", "min(", "len(", "filter", "map"))


def prep(out_path: str = "artifacts/mbpp_plus_prepped.jsonl", limit: int = 0,
         repo: str = "evalplus/mbppplus", timeout: float = 20.0) -> dict:
    """Download -> normalize -> VALIDATE -> cache. Returns the stats report (also printed)."""
    from datasets import load_dataset
    try:
        ds = load_dataset(repo, split="test")
        source = repo
    except Exception as e:                          # fallback: plain MBPP (weak gate — flagged)
        print(f"  [{repo} unavailable: {type(e).__name__}] falling back to plain MBPP (3-assert gate "
              f"is WEAK — prefer mbppplus)", flush=True)
        ds = load_dataset("google-research-datasets/mbpp", "sanitized", split="test")
        source = "google-research-datasets/mbpp"
    kept, dropped_norm, dropped_val, pipe_shaped = [], 0, 0, 0
    for raw in ds:
        rec = normalize(dict(raw))
        if rec is None:
            dropped_norm += 1
            continue
        if not validate(rec, timeout=timeout):
            dropped_val += 1
            continue
        rec["pipeline_shaped"] = _pipeline_shaped(rec)
        pipe_shaped += int(rec["pipeline_shaped"])
        kept.append(rec)
        if limit and len(kept) >= limit:
            break
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for rec in kept:
            f.write(json.dumps(rec) + "\n")
    n_plus = [r["n_plus"] for r in kept]
    report = dict(source=source, kept=len(kept), dropped_normalize=dropped_norm,
                  dropped_validation=dropped_val, pipeline_shaped=pipe_shaped,
                  lm_author_territory=len(kept) - pipe_shaped,
                  mean_plus_asserts=round(sum(n_plus) / max(1, len(n_plus)), 1), out=out_path)
    print("  MBPP+ prep report:", flush=True)
    for k, v in report.items():
        print(f"    {k}: {v}", flush=True)
    return report


def load_prepped(path: str = "artifacts/mbpp_plus_prepped.jsonl", limit: int = 0,
                 pipeline_only: bool = False):
    """Prepped jsonl -> MBPPTask list (tests = original asserts + the plus script: the DENSE gate)."""
    tasks = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            if pipeline_only and not rec.get("pipeline_shaped"):
                continue
            tests = list(rec["asserts"]) + ([rec["plus_test"]] if rec["plus_test"] else [])
            tasks.append(MBPPTask(rec["name"], f"{rec['text']}\nWrite `{rec['name']}(...)`.",
                                  tests, rec.get("setup", "")))
            if limit and len(tasks) >= limit:
                break
    return tasks


# ═══════════════════════════════════════════════════════════════════════════════
# SELFTEST (offline) — the pipeline on fabricated records: normalize maps fields, VALIDATION drops a
# record whose reference fails its own plus tests, cache round-trips, loaded task verifies.
# ═══════════════════════════════════════════════════════════════════════════════

def _selftest() -> bool:
    import tempfile
    print("algo_mbpp_prep --selftest: normalize -> validate -> cache -> load (offline, fabricated)\n")

    good = {"prompt": "Write a function to sum the squares of a list.",
            "code": "def sum_squares(xs):\n    return sum(x * x for x in xs)",
            "test_list": ["assert sum_squares([1, 2]) == 5"],
            "test": "\n".join(f"assert sum_squares({xs!r}) == {sum(x*x for x in xs)}"
                              for xs in ([1, 2, 3], [0], [7, 7], list(range(9)))),
            "test_imports": []}
    broken = {"prompt": "Write a function to double a number.",
              "code": "def dbl(n):\n    return n + 1",                    # reference is WRONG
              "test_list": ["assert dbl(2) == 4"], "test": "assert dbl(5) == 10"}
    malformed = {"prompt": "", "code": "def f(): pass", "test_list": []}

    r_good = normalize(good)
    assert r_good and r_good["name"] == "sum_squares" and r_good["n_plus"] == 4, r_good
    assert normalize(malformed) is None
    print("  [1] normalize: fields mapped, entry fn extracted, plus-assert density counted, malformed "
          "dropped -> PASS")

    assert validate(r_good)
    r_broken = normalize(broken)
    assert r_broken is not None and not validate(r_broken)
    print("  [2] VALIDATION gate: sound reference passes; a record whose own reference FAILS its tests "
          "is dropped (no silent junk) -> PASS")

    assert _pipeline_shaped(r_good)
    with tempfile.TemporaryDirectory() as td:
        p = str(Path(td) / "prep.jsonl")
        r_good["pipeline_shaped"] = True
        with open(p, "w", encoding="utf-8") as f:
            f.write(json.dumps(r_good) + "\n")
        tasks = load_prepped(p)
        assert len(tasks) == 1 and tasks[0].name == "sum_squares" and len(tasks[0].tests) == 2
        assert tasks[0].verify(r_good["code"])                 # dense gate: asserts + plus script
        assert not tasks[0].verify("def sum_squares(xs):\n    return 5")   # benchmark-only overfit dies
        print("  [3] cache round-trip -> MBPPTask; the DENSE gate passes the reference and kills the "
              "single-assert overfit -> PASS")

    print("\n  ALGO_MBPP_PREP SELFTEST -> PASS  (real open-source tasks, preprocessed + validated, "
          "gate-compatible — the LM-author rung's corpus)")
    return True


def main():
    ap = argparse.ArgumentParser(description="GRR-13b: MBPP+ preprocessing (normalize/validate/cache).")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--prep", action="store_true", help="download + preprocess + cache (molab)")
    ap.add_argument("--out", default="artifacts/mbpp_plus_prepped.jsonl")
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    if a.selftest:
        sys.exit(0 if _selftest() else 1)
    if a.prep:
        prep(a.out, limit=a.limit)
        return
    ap.print_help()


if __name__ == "__main__":
    main()
