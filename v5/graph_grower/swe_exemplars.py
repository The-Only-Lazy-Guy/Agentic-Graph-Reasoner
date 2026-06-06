"""Worked-exemplar nodes for retrieve-and-adapt (plan Phase A).

Stage-4 manual inspection showed the weak 4B localizes (right file) but can't SYNTHESIZE
the correct surgical fix. The design's answer is worked exemplars: hand the model a
near-correct SIMILAR diff to ADAPT instead of inventing one. This builds the exemplar
corpus -- one record per SWE instance: (issue -> gold diff + touched-symbol names) -- the
executable knowledge the graph was missing. retrieve_adapt.py then, for a NEW issue,
retrieves the nearest OTHER exemplar (exclude self = no leakage) and adapts it.

  python -m v5.graph_grower.swe_exemplars \
    --traces data/swe/grounded_traces.jsonl data/swe/grounded_traces_verified.jsonl \
    --nodes artifacts/graph_growth/swe_code_candidates.jsonl \
            artifacts/graph_growth/swe_code_candidates_verified.jsonl \
    --dataset lite --out data/swe/exemplars.jsonl
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from v5.graph_grower.swe_load import load_instances
from v5.graph_grower.swe_probe import load_traces, load_node_texts, _symbol_name


def build(traces_p: Sequence[str], nodes_p: Sequence[str], datasets: Sequence[str],
          split: str, max_diff: int = 3000) -> list:
    traces = load_traces(traces_p)
    id2text = load_node_texts(nodes_p)
    insts = {}
    for ds in datasets:
        insts.update({t["instance_id"]: t for t in load_instances(ds, split, limit=0)})
    rows = []
    for iid, t in traces.items():
        inst = insts.get(iid)
        if not inst:
            continue
        diff = (inst.get("patch") or "").strip()
        issue = (t.get("issue") or "").strip()
        if not (diff and issue):
            continue
        syms = [n for s in t["support_ids"] if s in id2text and (n := _symbol_name(id2text[s]))]
        rows.append({"instance_id": iid, "repo": inst.get("repo", ""),
                     "issue": issue[:2500], "diff": diff[:max_diff], "symbols": syms})
    return rows


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Build worked-exemplar corpus (issue -> gold diff).")
    ap.add_argument("--traces", nargs="+", default=["data/swe/grounded_traces.jsonl"])
    ap.add_argument("--nodes", nargs="+", default=["artifacts/graph_growth/swe_code_candidates.jsonl"])
    ap.add_argument("--dataset", nargs="+", default=["lite"])
    ap.add_argument("--split", default="test")
    ap.add_argument("--out", default="data/swe/exemplars.jsonl")
    args = ap.parse_args(argv)
    rows = build(args.traces, args.nodes, args.dataset, args.split)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as w:
        for r in rows:
            w.write(json.dumps(r, ensure_ascii=False) + "\n")
    avg = sum(len(r["diff"]) for r in rows) / max(1, len(rows))
    print(f"wrote {len(rows)} exemplars -> {args.out}  (avg diff {avg:.0f} chars, "
          f"{sum(1 for r in rows if r['symbols'])} with symbols)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
