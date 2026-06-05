"""Project SWE code grounded_traces -> the phase15 training-corpus rows the V5 injection
trainer (stage1/2A/2B) consumes. The trainer was built for STEM V4 traces (anchors +
v5_projection); the code traces only carry a `v2_grounding` block, so this is the missing
adapter that lets the L8/L20 reasoning-grounding heads train on CODE.

Mapping (V5_V2_DESIGN stage 6):
  - subgraph nodes (input.anchors) = the brief's retrieved symbols + the support symbols,
    typed `fact` so they land in the EVIDENCE pool (raw `symbol` isn't pooled -> invisible
    to cross-attention). Strategy nodes -> `strategy` (PLANNING pool) when present.
  - v5_projection.support_target / evidence_target = the support symbols (the fix touched).
  - v5_projection.distractor_target = retrieved non-support symbols (trained as negatives:
    the evidence head must concentrate on support, not the rest of the brief).
  - outputs.answer = the gold solution/patch; metrics.finalized = True (gold resolved it).

  python -m v5.graph_grower.code_to_corpus \
    --traces data/swe/grounded_traces.jsonl data/swe/grounded_traces_verified.jsonl \
    --nodes artifacts/graph_growth/swe_code_candidates.jsonl \
            artifacts/graph_growth/swe_code_candidates_verified.jsonl \
    --strategy artifacts/graph_growth/swe_strategy_candidates_clean.jsonl \
    --out data/swe/code_phase15_corpus.jsonl
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from v5.graph_grower.swe_probe import load_node_texts


PLANNING_TYPES = {"strategy", "reasoning_atom", "reasoning_chain"}   # -> PLANNING pool (L8)


def load_strategy_by_instance(paths: Sequence[str]):
    """strategy candidates -> ({instance_id: [(nid, node_type)]}, {nid: text}). Lets each
    code row attach its OWN distilled strategy/reasoning nodes as PLANNING anchors so the
    planning head (L8) trains (was 0 — code rows had no planning targets)."""
    by_inst: Dict[str, List] = {}
    id2text: Dict[str, str] = {}
    for p in paths:
        for line in Path(p).read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            e = json.loads(line).get("raw_edit", {})
            if e.get("op") != "add_node":
                continue
            meta = e.get("metadata", {}) or {}
            iid = meta.get("instance_id")
            nid, nt = e.get("node_id"), e.get("node_type", "")
            if iid and nid:
                by_inst.setdefault(iid, []).append((nid, nt))
                id2text[nid] = e.get("text", "") or ""
    return by_inst, id2text


def build_rows(trace_paths: Sequence[str], id2text: Dict[str, str],
               strat_by_inst: Dict[str, List], max_answer: int = 4000) -> List[dict]:
    rows: List[dict] = []
    for tp in trace_paths:
        for line in Path(tp).read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            v = r.get("v2_grounding") or {}
            iid = r.get("instance_id") or v.get("instance_id") or ""
            issue = str(v.get("task", "")).strip()
            support = [s for s in (v.get("support_ids") or []) if s in id2text]
            retrieved = [s for s in ((v.get("brief") or {}).get("retrieved_ids") or []) if s in id2text]
            if not (issue and support):
                continue
            # this instance's distilled strategy/reasoning nodes -> PLANNING anchors (L8)
            strat_nodes = [(nid, nt) for nid, nt in strat_by_inst.get(iid, []) if nid in id2text]
            strat_type = {nid: nt for nid, nt in strat_nodes}
            planning_ids = [nid for nid, nt in strat_nodes if nt in PLANNING_TYPES]
            sup_set = set(support)
            # subgraph node pool = support + retrieved brief + this instance's strategy nodes
            cand = list(dict.fromkeys(support + retrieved + [nid for nid, _ in strat_nodes]))
            anchors = [{"id": nid, "text": id2text.get(nid, ""),
                        "node_type": strat_type.get(nid, "fact")} for nid in cand]
            solution = v.get("solution")
            answer = (solution if isinstance(solution, str) else json.dumps(solution))[:max_answer]
            rows.append({
                "session_id": iid,
                "input": {"question": issue, "anchors": anchors,
                          "task_family": "code_fix"},
                "outputs": {"answer_polished": answer},
                "metrics": {"finalized": True, "steps": 1, "max_steps": 4},
                "trace": {"scoped_patches": []},
                "v5_projection": {
                    "candidate_node_ids": cand,
                    "support_target": {s: 1.0 for s in support},
                    "evidence_target": {s: 1.0 for s in support},
                    "distractor_target": {c: 1.0 for c in retrieved if c not in sup_set and c not in strat_type},
                    "planning_target": {nid: 1.0 for nid in planning_ids},
                    "diagnostics": {"source": "swe_code", "instance_id": iid,
                                    "n_support": len(support), "n_cand": len(cand),
                                    "n_planning": len(planning_ids)},
                },
            })
    return rows


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Project SWE code traces -> phase15 training corpus.")
    ap.add_argument("--traces", nargs="+", required=True)
    ap.add_argument("--nodes", nargs="+", required=True, help="code symbol add_node candidates")
    ap.add_argument("--strategy", nargs="*", default=[], help="strategy candidates (planning nodes)")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default="data/swe/code_phase15_corpus.jsonl")
    args = ap.parse_args(argv)

    sym_text = load_node_texts(args.nodes)
    strat_by_inst, strat_text = load_strategy_by_instance(args.strategy) if args.strategy else ({}, {})
    id2text = {**strat_text, **sym_text}          # both symbol + strategy texts for anchors
    rows = build_rows(args.traces, id2text, strat_by_inst)
    if args.limit:
        rows = rows[: args.limit]
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as w:
        for r in rows:
            w.write(json.dumps(r, ensure_ascii=False) + "\n")
    n_sup = sum(len(r["v5_projection"]["support_target"]) for r in rows)
    n_dis = sum(len(r["v5_projection"]["distractor_target"]) for r in rows)
    n_pl = sum(len(r["v5_projection"]["planning_target"]) for r in rows)
    n_with_pl = sum(1 for r in rows if r["v5_projection"]["planning_target"])
    print(f"wrote {len(rows)} code rows -> {args.out}  (avg support {n_sup/max(1,len(rows)):.1f}, "
          f"avg distractors {n_dis/max(1,len(rows)):.1f}, avg planning {n_pl/max(1,len(rows)):.1f}, "
          f"rows-with-planning {n_with_pl}/{len(rows)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
