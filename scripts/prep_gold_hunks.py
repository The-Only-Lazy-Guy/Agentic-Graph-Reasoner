"""Extract single-hunk gold patches from the cached SWE-bench parquet into a small json.

Separate torch-free process, per this repo's documented pyarrow-after-torch segfault (the same reason
prep_hotpot.py and prep_swe_tests.py exist). algo_grr_reuse imports torch transitively via
algo_grr_toolsmith, so it CANNOT read the parquet itself -- doing so exits silently with no traceback.
"""
import json
import random
import sys
from pathlib import Path

PARQUET = (Path(r"E:\cache\hf\hub\datasets--princeton-nlp--SWE-bench\snapshots")
           / "e48e2bd1e9fecd5bbd641e9414ac59da9f2e69f6" / "data" / "test-00000-of-00001.parquet")
OUT = Path(r"E:\PROJECT\graph_v5\artifacts\swebench_gold_hunks.json")


def parse_hunks(diff: str) -> list:
    out, before, after, inhunk = [], [], [], False
    for ln in (diff or "").splitlines():
        if ln.startswith("@@"):
            if inhunk and (before or after):
                out.append(("\n".join(before), "\n".join(after)))
            before, after, inhunk = [], [], True
            continue
        if not inhunk or ln.startswith(("--- ", "+++ ", "diff ")):
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


def main() -> int:
    import pyarrow.parquet as pq
    if not PARQUET.exists():
        print(f"missing {PARQUET}", file=sys.stderr)
        return 1
    t = pq.read_table(PARQUET).select(
        ["instance_id", "repo", "problem_statement", "patch"]).to_pydict()
    rows = []
    for i in range(len(t["instance_id"])):
        diff = t["patch"][i]
        if diff.count("+++ b/") != 1:                 # single FILE only
            continue
        hs = parse_hunks(diff)
        if len(hs) != 1:                              # single HUNK only -- see algo_grr_reuse docstring
            continue
        b, a = hs[0]
        if not b.strip() or not a.strip() or len(b) > 2500:
            continue
        rows.append({"instance_id": t["instance_id"][i], "repo": t["repo"][i],
                     "problem": t["problem_statement"][i], "before": b, "after": a})
    random.Random(0).shuffle(rows)
    OUT.write_text(json.dumps(rows), encoding="utf-8")
    from collections import Counter
    c = Counter(r["repo"] for r in rows)
    print(f"wrote {OUT}  ({len(rows)} single-hunk instances of {len(t['instance_id'])} total)")
    for repo, k in c.most_common(6):
        print(f"  {repo:34s} {k}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
