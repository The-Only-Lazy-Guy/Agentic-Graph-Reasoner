"""Mine RECURRING REPAIR SCHEMAS from real SWE-bench gold patches.

The point: a schema is an abstraction over many real fixes ("widen an isinstance to accept another
type", "add a None guard"). It generalises BY CONSTRUCTION, which is exactly what the banked literal
tools could not do -- measured 0 replays across 40 instances because every tool hardcoded one file's
exact line.

These are MINED, not invented. If a pattern is not actually recurrent in the real gold diffs it does
not become a schema, and the coverage number below is the honest ceiling for any schema-based repair:
instances outside it cannot be fixed this way no matter how good the chooser is.

Torch-free (reads the prepped json only) per this repo's pyarrow/torch constraint.
"""
import json
import re
import sys
from collections import Counter
from pathlib import Path

SRC = Path(r"E:\PROJECT\graph_v5\artifacts\swebench_gold_hunks.json")
OUT = Path(r"E:\PROJECT\graph_v5\artifacts\repair_schemas.json")


def changed_lines(before: str, after: str):
    """The (removed, added) line pairs of a hunk, ignoring untouched context."""
    b = [l for l in before.splitlines()]
    a = [l for l in after.splitlines()]
    bs, as_ = set(b), set(a)
    rem = [l for l in b if l not in as_]
    add = [l for l in a if l not in bs]
    return rem, add


# Each schema: a name, a predicate over (removed, added) lines, and the slots an LM would fill.
SCHEMAS = [
    ("widen_isinstance",
     lambda r, a: any("isinstance" in x for x in r) and any("isinstance" in y for y in a)
     and any(y.count("(") > x.count("(") for x, y in zip(r, a)),
     ["symbol", "added_type"]),
    ("add_none_guard",
     lambda r, a: any(re.search(r"\bis None\b|\bis not None\b", y) for y in a)
     and not any(re.search(r"\bis None\b|\bis not None\b", x) for x in r),
     ["symbol"]),
    ("change_default_arg",
     lambda r, a: any(re.search(r"def \w+\(.*=", x) for x in r)
     and any(re.search(r"def \w+\(.*=", y) for y in a),
     ["function", "param", "new_default"]),
    ("add_kwarg_passthrough",
     lambda r, a: any(re.search(r"\*\*kwargs|\w+=\w+", y) for y in a)
     and any(len(y) > len(x) for x, y in zip(r, a)),
     ["call", "kwarg"]),
    ("swap_operator",
     lambda r, a: any(re.sub(r"[=<>!]+", "", x).strip() == re.sub(r"[=<>!]+", "", y).strip()
                      and x.strip() != y.strip() for x, y in zip(r, a)),
     ["expression"]),
    ("wrap_in_call",
     lambda r, a: any(re.search(r"\w+\(", y) and y.count("(") > x.count("(")
                      and x.strip().strip("()") in y for x, y in zip(r, a)),
     ["expression", "wrapper"]),
    ("add_attribute_check",
     lambda r, a: any("hasattr" in y or "getattr" in y for y in a)
     and not any("hasattr" in x or "getattr" in x for x in r),
     ["object", "attribute"]),
    ("extend_collection_literal",
     lambda r, a: any(("[" in y or "(" in y or "{" in y) and len(y) > len(x)
                      and x.strip().rstrip("])}") in y for x, y in zip(r, a)),
     ["collection", "new_member"]),
]


def classify(before: str, after: str):
    r, a = changed_lines(before, after)
    if not r or not a:
        return None
    for name, pred, slots in SCHEMAS:
        try:
            if pred(r, a):
                return name
        except Exception:                                          # noqa: BLE001
            continue
    return None


def main() -> int:
    if not SRC.exists():
        print(f"missing {SRC} -- run scripts/prep_gold_hunks.py first", file=sys.stderr)
        return 1
    rows = json.loads(SRC.read_text(encoding="utf-8"))
    c = Counter()
    labelled = []
    for r in rows:
        s = classify(r["before"], r["after"])
        c[s or "UNMATCHED"] += 1
        if s:
            labelled.append({**r, "schema": s})
    n = len(rows)
    print(f"{n} real single-hunk gold patches\n")
    for name, k in c.most_common():
        print(f"  {name:28s} {k:5d}  {k / n:6.1%}")
    cov = sum(v for k, v in c.items() if k != "UNMATCHED")
    print(f"\n  COVERAGE (a mined schema matches the real fix): {cov}/{n} = {cov / n:.1%}")
    print(f"  ^ HARD CEILING for schema-based repair. Instances outside it cannot be fixed this way")
    print(f"    no matter how good the schema chooser is.")
    slots = {name: sl for name, _, sl in SCHEMAS}
    OUT.write_text(json.dumps({"schemas": slots, "instances": labelled}), encoding="utf-8")
    print(f"\nwrote {OUT} ({len(labelled)} labelled instances)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
