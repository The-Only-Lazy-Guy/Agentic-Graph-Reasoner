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


def _noop_patch(iid: str) -> str:
    """A NON-EMPTY patch that creates a fresh file -> always applies cleanly, behavior-neutral, so the
    FAIL_TO_PASS tests still fail on the (otherwise unpatched) code. (Empty patches are dropped by the
    swebench harness.)"""
    fn = f"_noop_probe_{iid.replace('/', '_')}.txt"
    return (f"diff --git a/{fn} b/{fn}\n"
            f"new file mode 100644\n"
            f"--- /dev/null\n"
            f"+++ b/{fn}\n"
            f"@@ -0,0 +1 @@\n"
            f"+noop\n")


def main():
    ap = argparse.ArgumentParser(description="Capture SWE FAIL_TO_PASS failures (the debug observation).")
    ap.add_argument("--dataset", default="lite")
    ap.add_argument("--split", default="test")
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--backend", default="docker")
    ap.add_argument("--run-id", default="capture_fail")
    ap.add_argument("--out", default="data/traj/swe_failures.jsonl")
    ap.add_argument("--max-workers", type=int, default=2)   # 137/OOM: few parallel heavy containers
    ap.add_argument("--batch", type=int, default=5, help="instances per harness call (incremental save)")
    a = ap.parse_args()
    from v5.graph_grower.swe_load import load_instances
    from v5.graph_grower import swe_verify as V
    insts = list(load_instances(name=a.dataset, split=a.split, limit=a.n))
    outp = Path(a.out); outp.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if outp.exists():                                        # RESUME: skip already-captured (survives crash/sleep)
        done = {json.loads(l)["iid"] for l in outp.read_text(encoding="utf-8").splitlines() if l.strip()}
        print(f"[capture] resume: {len(done)} already captured, skipping them", flush=True)
    todo = [i for i in insts if i["instance_id"] not in done]
    out_dir = "artifacts/graph_growth/swe_capture"; Path(out_dir).mkdir(parents=True, exist_ok=True)
    got = 0
    with open(outp, "a", encoding="utf-8") as f:             # APPEND: incremental, per-batch durability
        for bi in range(0, len(todo), a.batch):
            chunk = todo[bi:bi + a.batch]; ids = [i["instance_id"] for i in chunk]
            rid = f"{a.run_id}_{bi // a.batch}"
            preds_path = f"{out_dir}/_capture_{rid}.jsonl"
            V.write_predictions({iid: _noop_patch(iid) for iid in ids}, preds_path)
            print(f"[capture] batch {bi // a.batch + 1}/{-(-len(todo)//a.batch)}: {len(ids)} instances "
                  f"(workers={a.max_workers}) starting {ids[0]}...", flush=True)
            try:
                V._verify(preds_path, a.dataset, rid, a.backend, ids, a.max_workers, out_dir, a.split)
            except Exception as e:                           # noqa: BLE001 -- report still written per-instance
                print(f"[capture] _verify raised ({e}); collecting per-instance reports", flush=True)
            for i in chunk:
                fb = V._failure_feedback(rid, i["instance_id"])
                has = "FAIL" in fb or "still" in fb.lower()
                got += int(has)
                f.write(json.dumps({"iid": i["instance_id"], "issue": (i.get("problem_statement") or "")[:2500],
                                    "test_failure": fb[:2500], "has_failure": has,
                                    "gold_patch": i.get("patch", "")}) + "\n"); f.flush()
            print(f"[capture] batch done | cumulative failures captured this run: {got}", flush=True)
    total = len(done) + len(todo)
    print(f"[capture] -> {a.out}  (total {total} instances; this run {len(todo)}, {got} FAIL_TO_PASS captured)")


if __name__ == "__main__":
    main()
