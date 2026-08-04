"""VERIFIED schema mining: a hunk earns a schema label ONLY if that schema, with some concrete
parameters, PROVABLY reproduces the real gold fix.

Why this replaces mine_repair_schemas.py: that version labelled by surface predicates ("isinstance
appears on both sides"), which is a guess. Measured precision on the widen_isinstance class was
2/15 = 13% -- with the ORACLE schema AND oracle parameters derived from the gold diff, 13 of 15 could
not be reproduced at all, because the predicate had matched restructures that merely mention
isinstance. So the previous "16.5% coverage" was inflated ~7x, and the schema-thinker was being
trained to predict mislabels: its held-out 0.250 was real skill aimed at a corrupted target.

This is the same generate-and-verify discipline the rest of this stack already runs on, applied to the
LABELS. Candidate parameters are enumerated from the diff itself, every candidate is EXECUTED, and the
label is kept only when the output equals the gold `after` exactly. A schema that cannot reproduce the
fix is not that fix's schema, whatever it looks like.

MORE DATA: multi-hunk patches are no longer dropped. Each hunk is an independent labelling target, so
a 3-hunk patch contributes up to 3 examples. Single-hunk-only was a restriction of the repair
experiment (you must fix every hunk to fix the bug), not of schema CLASSIFICATION.

Torch-free per this repo's pyarrow/torch constraint.
"""
import json
import re
import sys
from collections import Counter
from pathlib import Path

PARQUET = (Path(r"E:\cache\hf\hub\datasets--princeton-nlp--SWE-bench\snapshots")
           / "e48e2bd1e9fecd5bbd641e9414ac59da9f2e69f6" / "data" / "test-00000-of-00001.parquet")
OUT = Path(r"E:\PROJECT\graph_v5\artifacts\repair_schemas_verified.json")
IDENT = re.compile(r"[A-Za-z_][A-Za-z0-9_\.]*")


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


# ── executable schemas (kept byte-identical in behaviour to algo_grr_schema.apply_schema) ─────────
def apply_schema(name, p, text):
    s = p.get("symbol") or p.get("expression") or p.get("object") or ""
    if name == "widen_isinstance":
        t = p.get("added_type", "")
        if not s or not t:
            return text
        def _w(m):
            arg = m.group(2).strip()
            inner = arg[1:-1].strip() if arg.startswith("(") and arg.endswith(")") else arg
            return f"isinstance({m.group(1)}, ({inner}, {t}))"
        return re.sub(rf"isinstance\(\s*({re.escape(s)})\s*,\s*(\([^)]*\)|[\w\.]+)\s*\)", _w, text, count=1)
    if name == "add_none_guard":
        if not s:
            return text
        return re.sub(rf"^(\s*)(return\s+{re.escape(s)}\b.*)$",
                      rf"\1if {s} is None:\n\1    return None\n\1\2", text, count=1, flags=re.M)
    if name == "add_attribute_check":
        o, at = p.get("object", ""), p.get("attribute", "")
        if not o or not at:
            return text
        return re.sub(rf"^(\s*)(.*\b{re.escape(o)}\.{re.escape(at)}\b.*)$",
                      rf"\1if hasattr({o}, '{at}'):\n\1    \2", text, count=1, flags=re.M)
    if name == "extend_collection_literal":
        c, m2 = p.get("collection", ""), p.get("new_member", "")
        if not c or not m2:
            return text
        return re.sub(rf"({re.escape(c)}\s*=\s*[\[\(\{{][^\]\)\}}]*)([\]\)\}}])", rf"\1, {m2}\2",
                      text, count=1)
    if name == "wrap_in_call":
        e, w = p.get("expression", ""), p.get("wrapper", "")
        if not e or not w:
            return text
        return text.replace(e, f"{w}({e})", 1)
    if name == "replace_line":
        o, n = p.get("old_line", ""), p.get("new_line", "")
        if not o:
            return text
        return text.replace(o, n, 1)
    return text


def changed(before, after):
    bs, as_ = set(before.splitlines()), set(after.splitlines())
    return ([l for l in before.splitlines() if l not in as_],
            [l for l in after.splitlines() if l not in bs])


def candidates(name, before, after):
    """Enumerate plausible parameter sets FROM THE DIFF. Bounded on purpose -- this is a search for
    whether the schema CAN express the fix, not an attempt to make every hunk match something."""
    rem, add = changed(before, after)
    if name == "widen_isinstance":
        for m in re.finditer(r"isinstance\(\s*([\w\.]+)\s*,\s*\(([^)]*)\)\s*\)", after):
            for t in [x.strip() for x in m.group(2).split(",")]:
                yield {"symbol": m.group(1), "added_type": t}
    elif name == "add_none_guard":
        for l in add:
            m = re.search(r"if\s+([\w\.]+)\s+is None", l)
            if m:
                yield {"symbol": m.group(1)}
    elif name == "add_attribute_check":
        for l in add:
            m = re.search(r"hasattr\(\s*([\w\.]+)\s*,\s*['\"](\w+)['\"]", l)
            if m:
                yield {"object": m.group(1), "attribute": m.group(2)}
    elif name == "extend_collection_literal":
        for l in add:
            m = re.search(r"([\w\.]+)\s*=\s*[\[\(\{]", l)
            if m:
                for tok in IDENT.findall(l)[-4:]:
                    yield {"collection": m.group(1), "new_member": tok}
    elif name == "wrap_in_call":
        for x, y in zip(rem, add):
            m = re.search(r"([\w\.]+)\(", y)
            if m:
                yield {"expression": x.strip(), "wrapper": m.group(1)}
    elif name == "replace_line":
        # the DEGENERATE schema: exactly one line changed. Included deliberately as a BASELINE --
        # it always reproduces a 1-line fix, so it measures how much of the corpus is "just one line"
        # and cannot be claimed as an abstraction. Reported separately, never mixed in.
        if len(rem) == 1 and len(add) == 1:
            yield {"old_line": rem[0], "new_line": add[0]}


SCHEMAS = ["widen_isinstance", "add_none_guard", "add_attribute_check",
           "extend_collection_literal", "wrap_in_call", "replace_line"]


def label(before, after):
    """Try every schema with every enumerated parameter set. Keep only a PROVEN reproduction."""
    for name in SCHEMAS:
        for p in candidates(name, before, after):
            try:
                if apply_schema(name, p, before).strip() == after.strip():
                    return name, p
            except Exception:                                      # noqa: BLE001
                continue
    return None, None


def main() -> int:
    import pyarrow.parquet as pq
    t = pq.read_table(PARQUET).select(
        ["instance_id", "repo", "problem_statement", "patch"]).to_pydict()
    rows, n_hunks, c = [], 0, Counter()
    for i in range(len(t["instance_id"])):
        for b, a in parse_hunks(t["patch"][i]):
            if not b.strip() or not a.strip() or len(b) > 3000:
                continue
            n_hunks += 1
            name, p = label(b, a)
            c[name or "UNMATCHED"] += 1
            if name:
                rows.append({"instance_id": t["instance_id"][i], "repo": t["repo"][i],
                             "problem": t["problem_statement"][i], "before": b, "after": a,
                             "schema": name, "params": p})
    print(f"{n_hunks} real hunks from {len(t['instance_id'])} SWE-bench instances "
          f"(multi-hunk patches now contribute every hunk)\n")
    for k, v in c.most_common():
        print(f"  {k:28s} {v:5d}  {v / n_hunks:6.1%}")
    real = [r for r in rows if r["schema"] != "replace_line"]
    print(f"\n  VERIFIED coverage, ABSTRACT schemas only : {len(real)}/{n_hunks} = {len(real)/n_hunks:.1%}")
    print(f"  (+ degenerate one-line replace_line       : {c['replace_line']}/{n_hunks} = "
          f"{c['replace_line']/n_hunks:.1%}  <- a baseline, NOT an abstraction)")
    print(f"\n  previous UNVERIFIED miner claimed 16.5%; its widen_isinstance precision measured 13%.")
    OUT.write_text(json.dumps({"instances": rows}), encoding="utf-8")
    print(f"\nwrote {OUT} ({len(rows)} verified-labelled hunks, {len(real)} abstract)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
