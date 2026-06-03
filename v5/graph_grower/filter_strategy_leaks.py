"""Retroactive generality filter for swe_strategy candidate queues.

The 300-instance #3 run was generated BEFORE the in-code generality gate
(swe_strategy._is_leaky / _LINT_TYPES) existed. This applies the same type-aware
filter to existing candidate jsonl(s): drop repo-specific RETRIEVAL-ENTRY nodes
(strategy / reasoning_atom / chain) + their edges; KEEP solved_subgoal (instance
resolution, grounded by edge). Deterministic, no teacher calls. See READ_THIS 03b.

  python -m v5.graph_grower.filter_strategy_leaks \
    --in artifacts/graph_growth/swe_strategy_candidates_s0.jsonl \
         artifacts/graph_growth/swe_strategy_candidates_s1.jsonl \
    --out artifacts/graph_growth/swe_strategy_candidates_clean.jsonl
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from v5.graph_grower.swe_strategy import _is_leaky, _LINT_TYPES


def filter_files(in_paths: Sequence[str], out_path: str) -> dict:
    rows = []
    for p in in_paths:
        for line in Path(p).read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                rows.append(json.loads(line))

    dropped = set()
    drop_by: dict = {}
    for r in rows:
        e = r.get("raw_edit", {})
        if e.get("op") != "add_node" or not e.get("metadata", {}).get("swe_strategy"):
            continue
        nt = e.get("node_type")
        if nt in _LINT_TYPES and _is_leaky(e.get("text", ""), e.get("metadata", {}).get("kb_domain", "")):
            dropped.add(e["node_id"]); drop_by[nt] = drop_by.get(nt, 0) + 1

    kept, dn, de = [], 0, 0
    for r in rows:
        e = r.get("raw_edit", {}); op = e.get("op")
        if op == "add_node":
            if e["node_id"] in dropped:
                dn += 1; continue
            kept.append(r)
        elif op == "add_edge":
            if e.get("src") in dropped or e.get("dst") in dropped:
                de += 1; continue
            kept.append(r)
        else:
            kept.append(r)

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as w:
        for r in kept:
            w.write(json.dumps(r, ensure_ascii=False) + "\n")
    return {"dropped_nodes": dn, "drop_by": drop_by, "dropped_edges": de,
            "kept": len(kept), "out": out_path}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Type-aware generality filter for strategy candidates.")
    ap.add_argument("--in", dest="in_paths", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    r = filter_files(args.in_paths, args.out)
    print(f"dropped {r['dropped_nodes']} nodes {r['drop_by']} + {r['dropped_edges']} edges "
          f"-> kept {r['kept']} -> {r['out']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
