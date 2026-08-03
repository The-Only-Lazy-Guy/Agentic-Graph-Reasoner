"""Extract each SWE-bench instance's REAL test metadata (FAIL_TO_PASS / PASS_TO_PASS / base commit)
from the cached princeton-nlp/SWE-bench parquet, into a small json the runtime can read.

Separate torch-free process on purpose: pyarrow after torch segfaults in this environment, the same
conflict this repo already documents for --grow-cot-docs-path / --math-cot-docs-path / prep_hotpot.py.

Why this exists: algo_grr_swetools.t_run_tests defaulted to `python -m pytest`, which is simply WRONG
for django (ships no pytest at all -- its runner is tests/runtests.py). A real run reported
"No module named pytest" and an earlier version of the ok-heuristic misread that as a PASS. The fix is
not a better guess at a command; it is using the test directives SWE-bench actually ships.
"""
import json
import sys
from pathlib import Path

PARQUET = (Path(r"E:\cache\hf\hub\datasets--princeton-nlp--SWE-bench\snapshots")
           / "e48e2bd1e9fecd5bbd641e9414ac59da9f2e69f6" / "data" / "test-00000-of-00001.parquet")
OUT = Path(r"E:\PROJECT\graph_v5\artifacts\swebench_tests.json")


def main() -> int:
    import pyarrow.parquet as pq
    if not PARQUET.exists():
        print(f"missing {PARQUET}", file=sys.stderr)
        return 1
    t = pq.read_table(PARQUET)
    cols = t.column_names
    print(f"columns: {cols}")
    # test_patch is REQUIRED, not optional: the FAIL_TO_PASS tests are ADDED by it. Without applying
    # it the test literally does not exist in the container ("type object 'GetFieldDisplayTests' has
    # no attribute 'test_overriding_FIELD_display'") and even the REAL gold patch scores as a failure.
    need = ["instance_id", "repo", "FAIL_TO_PASS", "PASS_TO_PASS", "base_commit",
            "environment_setup_commit", "test_patch"]
    have = [c for c in need if c in cols]
    d = t.select(have).to_pydict()
    n = len(d["instance_id"])
    out = {}
    for i in range(n):
        row = {k: d[k][i] for k in have}
        # FAIL_TO_PASS / PASS_TO_PASS ship as JSON-encoded lists of test node ids
        for k in ("FAIL_TO_PASS", "PASS_TO_PASS"):
            v = row.get(k)
            if isinstance(v, str):
                try:
                    row[k] = json.loads(v)
                except Exception:                                  # noqa: BLE001
                    row[k] = [v]
        out[row["instance_id"]] = row
    OUT.write_text(json.dumps(out), encoding="utf-8")
    f2p = sum(1 for v in out.values() if v.get("FAIL_TO_PASS"))
    print(f"wrote {OUT}  ({n} instances, {f2p} with FAIL_TO_PASS)")
    ex = out.get("django__django-11999")
    if ex:
        print("\nexample django__django-11999:")
        print("  FAIL_TO_PASS:", ex.get("FAIL_TO_PASS"))
        print("  PASS_TO_PASS:", (ex.get("PASS_TO_PASS") or [])[:3], "...")
    return 0


if __name__ == "__main__":
    sys.exit(main())
