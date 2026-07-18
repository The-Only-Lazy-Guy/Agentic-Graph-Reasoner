"""prep_mhpp — download MHPP (Mostly Hard Python Problems), normalise to our JSONL, and SELF-VALIDATE the
harness on the canonical solutions BEFORE any model run ("build before verify").

MHPP is HumanEval-format (function completion + a `check(candidate)` test). This script:
  1. loads MHPP (HF dataset name OR a raw JSONL url OR an existing local file),
  2. writes artifacts/mhpp.jsonl with the fields our loader expects:
       {prompt, entry_point, test, canonical_solution, task_id?}
  3. VALIDATES: runs each CANONICAL solution through verify_check -> they must (nearly) all pass. Any
     failure means a schema/harness mismatch to fix BEFORE trusting a model's score.

    # on molab (network):
    python -m v5.runtime.prep_mhpp --hf <mhpp-hf-path> --out artifacts/mhpp.jsonl
    python -m v5.runtime.prep_mhpp --url <raw-jsonl-url> --out artifacts/mhpp.jsonl
    # validate an already-downloaded file:
    python -m v5.runtime.prep_mhpp --validate artifacts/mhpp.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_ROOT = str(Path(__file__).resolve().parents[2])
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

# HARD benchmark sources. NOTE: MHPP (SparksofAGI/MHPP) withholds tests+solutions (submit-to-server) ->
# it does NOT work with our LOCAL verify gate. Use BigCodeBench (public unittest tests) instead — our
# loader routes its unittest-style tests to verify_unittest. Pass --hf to override.
_HF_CANDIDATES = ["bigcode/bigcodebench", "SparksofAGI/MHPP"]


def _norm_row(r: dict) -> dict | None:
    """Map a hard-benchmark row (BigCodeBench / HumanEval+ / MHPP) to our schema. Returns None if unusable."""
    prompt = (r.get("complete_prompt") or r.get("instruct_prompt") or r.get("prompt") or r.get("problem")
              or r.get("text") or r.get("question") or "")
    entry = r.get("entry_point") or r.get("entry") or r.get("name") or ""
    test = r.get("test") or r.get("test_code") or r.get("test_list") or ""
    sol = r.get("canonical_solution") or r.get("solution") or r.get("code") or r.get("answer") or ""
    if isinstance(test, list):                       # some sets ship asserts as a list
        return {"prompt": prompt, "entry_point": entry, "asserts": test, "canonical_solution": sol,
                "task_id": r.get("task_id", "")}
    if not (entry and test):
        return None
    return {"prompt": prompt, "entry_point": entry, "test": test, "canonical_solution": sol,
            "task_id": r.get("task_id", "")}


def _load_rows(a) -> list[dict]:
    if a.validate:
        return [json.loads(l) for l in Path(a.validate).read_text(encoding="utf-8").splitlines() if l.strip()]
    if a.local:
        return [json.loads(l) for l in Path(a.local).read_text(encoding="utf-8").splitlines() if l.strip()]
    if a.url:
        import urllib.request
        print(f"[prep] downloading {a.url}")
        txt = urllib.request.urlopen(a.url).read().decode("utf-8")  # noqa: S310
        try:
            data = json.loads(txt)
            return data if isinstance(data, list) else [data]
        except json.JSONDecodeError:
            return [json.loads(l) for l in txt.splitlines() if l.strip()]
    # HF datasets
    from datasets import load_dataset
    names = [a.hf] if a.hf else _HF_CANDIDATES
    for name in names:
        try:
            print(f"[prep] trying HF dataset '{name}' split='{a.split}'")
            ds = load_dataset(name, split=a.split)
            return [dict(r) for r in ds]
        except Exception as e:  # noqa: BLE001
            print(f"       -> {e!r}")
            # split name wrong (e.g. BigCodeBench uses versioned splits v0.1.x) -> auto-pick the latest
            try:
                dd = load_dataset(name)                    # DatasetDict of all splits
                split = sorted(dd.keys())[-1]              # latest (e.g. v0.1.4)
                print(f"       -> auto-using split '{split}'")
                return [dict(r) for r in dd[split]]
            except Exception as e2:  # noqa: BLE001
                print(f"       -> {e2!r}")
    raise SystemExit("Could not load the dataset. Pass --hf <path> --split <name>, --url <jsonl>, or --local <file>.")


def _validate(path: str, limit: int = 0) -> bool:
    """Run each canonical solution through the loader's verify_fn — they must (nearly) all pass. Use
    `limit` to sample (BCB is 1140 heavy-lib subprocess verifies -> full validate takes 15+ min)."""
    from v5.runtime.algo_grr_mbpp import load_mhpp
    import os
    os.environ.setdefault("V5_HARD_VERIFY", "1")     # subprocess hard-verify (untrusted-code safe)
    tasks = load_mhpp(path, limit=limit or None)
    ok = miss = 0
    fails = []
    for t in tasks:
        ref = t.get("reference", "")
        if not ref:
            miss += 1
            continue
        score = t["verify_fn"](ref)[0]
        if score >= 1.0:
            ok += 1
        else:
            fails.append((t["entry"], score))
    n = len(tasks)
    print(f"\n[validate] {n} tasks | canonical PASS {ok}/{n} | no-solution {miss} | fail {len(fails)}")
    for e, s in fails[:10]:
        print(f"   FAIL {e}: score {s:.2f}")
    good = n > 0 and ok >= 0.9 * (n - miss)
    print(f"[validate] harness {'OK — safe to run the model' if good else 'MISMATCH — fix schema before running a model'}")
    return good


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf", default="", help="HF dataset path (overrides the best-effort candidates)")
    ap.add_argument("--url", default="", help="raw JSONL/JSON url")
    ap.add_argument("--local", default="", help="an existing raw file to normalise")
    ap.add_argument("--split", default="test")
    ap.add_argument("--out", default="artifacts/mhpp.jsonl")
    ap.add_argument("--validate", default="", help="validate an already-normalised jsonl (skip download)")
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()

    if a.validate:
        sys.exit(0 if _validate(a.validate, limit=a.limit) else 1)

    rows = _load_rows(a)
    norm = [x for x in (_norm_row(r) for r in rows) if x]
    if a.limit:
        norm = norm[:a.limit]
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        for x in norm:
            f.write(json.dumps(x) + "\n")
    print(f"[prep] wrote {len(norm)} tasks -> {a.out} (from {len(rows)} raw rows)")
    _validate(a.out, limit=a.limit or 50)            # sample-validate by default (full 1140 = 15+ min)


if __name__ == "__main__":
    main()
