"""Capture the FAIL_TO_PASS test-failure (the debug OBSERVATION) per SWE instance -> failure-rich data
to test T.9: does the test-failure help predict the gold operator, beyond the issue? (Fork A.)

Runs the REAL harness on the UNPATCHED repo (empty prediction) -> the FAIL_TO_PASS failures fail by
definition -> _failure_feedback() reads the harness report = the observation a debugger would see.

NEEDS DOCKER (run in WSL where swebench works). VALIDATE on --n 2 first (confirm test_failure is
non-empty + names real failing tests) before scaling.

  python -m v5.training.capture_failures --dataset lite --n 2  --out data/traj/swe_failures.jsonl   # validate
  python -m v5.training.capture_failures --dataset lite --n 50 --out data/traj/swe_failures.jsonl   # scale
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main():
    ap = argparse.ArgumentParser(description="Capture SWE FAIL_TO_PASS failures (the debug observation).")
    ap.add_argument("--dataset", default="lite")
    ap.add_argument("--split", default="test")
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--backend", default="docker")
    ap.add_argument("--run-id", default="capture_fail")
    ap.add_argument("--out", default="data/traj/swe_failures.jsonl")
    ap.add_argument("--max-workers", type=int, default=4)
    a = ap.parse_args()
    from v5.graph_grower.swe_load import load_instances
    from v5.graph_grower import swe_verify as V
    insts = list(load_instances(name=a.dataset, split=a.split, limit=a.n))
    ids = [i["instance_id"] for i in insts]
    out_dir = "artifacts/graph_growth/swe_capture"
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    preds_path = f"{out_dir}/_capture_{a.run_id}.jsonl"
    V.write_predictions({iid: "" for iid in ids}, preds_path)        # empty -> harness runs tests on unpatched
    print(f"[capture] running harness on {len(ids)} UNPATCHED instances (FAIL_TO_PASS should fail)...", flush=True)
    try:
        V._verify(preds_path, a.dataset, a.run_id, a.backend, ids, a.max_workers, out_dir, a.split)
    except Exception as e:                                           # noqa: BLE001 -- report still written per-instance
        print(f"[capture] _verify raised ({e}); collecting whatever reports were written", flush=True)
    rows, got = [], 0
    for i in insts:
        fb = V._failure_feedback(a.run_id, i["instance_id"])
        has = "FAIL" in fb or "still" in fb.lower()
        got += int(has)
        rows.append({"iid": i["instance_id"], "issue": (i.get("problem_statement") or "")[:2500],
                     "test_failure": fb[:2500], "has_failure": has, "gold_patch": i.get("patch", "")})
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    print(f"[capture] -> {a.out}  ({len(rows)} instances, {got} with a real FAIL_TO_PASS failure captured)")
    if got == 0:
        print("[capture] WARNING: 0 failures captured -> empty-patch may not trigger the harness here; "
              "tell me and I'll switch to a no-op patch that applies cleanly.")
    elif rows:
        print(f"[capture] sample test_failure: {rows[0]['test_failure'][:200]!r}")


if __name__ == "__main__":
    main()
