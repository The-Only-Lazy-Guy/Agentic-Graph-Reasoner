"""Learning capture (V4 `post_processing`, for code) — close the loop: solved run -> grow graph.

V4's loop ends by turning a solved SESSION into durable graph growth (LearningReport ->
graph_edits, gated). For code, the analog: a SOLVED instance (issue + the working patch +
the symbols it touched) -> an EXEMPLAR node + `leveraged` edges to those symbols, applied to
the live graph. Then the NEXT similar issue retrieves that exemplar (retrieve_adapt) and
adapts it -> the system gets better with every solved task. This is the piece that closes the
V4->V5 loop; it reuses the grower (stitch -> hub-wire -> gated apply).

"Solved" gate is external: pass a predictions jsonl of patches the verifier PASSED
(--solved), or default to GOLD patches (guaranteed-correct exemplars) for a clean demo.

  # from gold (clean demo): builds an exemplar per traced instance
  python -m v5.graph_grower.code_post_processing \
    --traces data/swe/grounded_traces.jsonl --nodes artifacts/graph_growth/swe_code_candidates.jsonl \
    --out artifacts/graph_growth/exemplar_candidates.jsonl
  # then grow the graph:
  python -m v5.graph_grower.apply --candidates artifacts/graph_growth/exemplar_candidates.jsonl \
    --graph graphs/grown_graph4.json --out graphs/grown_graph5.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Dict, List, Sequence

from v5.graph_grower.extract import (stitch_candidates, wire_hubs, write_candidate_queue,
                                     _to_candidate, Document)
from v5.graph_grower.swe_load import load_instances
from v5.graph_grower.swe_probe import load_traces, load_node_texts, _symbol_name


def _h(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()[:12]


def _issue_gist(issue: str) -> str:
    return re.split(r"[.\n]", (issue or "").strip())[0][:160]


def capture(instance_id: str, issue: str, model_patch: str, support_ids: Sequence[str],
            id2text: Dict[str, str]) -> List[dict]:
    """One solved instance -> exemplar node + `leveraged` edges to the support symbols."""
    syms = [n for s in support_ids if s in id2text and (n := _symbol_name(id2text[s]))]
    ex_id = f"exemplar_{_h(instance_id)}"
    # text = what we RETRIEVE on (issue-level) so a future similar issue finds this exemplar
    text = f"Worked fix for: {_issue_gist(issue)}" + (f" (touches {', '.join(syms[:4])})" if syms else "")
    doc = Document(id=instance_id, text="", domain="code", mode="cot")
    node = {"op": "add_node", "node_id": ex_id, "node_type": "exemplar", "text": text, "tier": "add",
            "metadata": {"swe_exemplar": True, "instance_id": instance_id,
                         "diff": (model_patch or "")[:4000], "symbols": syms}}
    cands = [_to_candidate(node, doc, 0)]
    for i, s in enumerate(support_ids):           # exemplar grounds onto the real code symbols
        link = {"op": "add_edge", "src": ex_id, "dst": s, "relation": "leveraged", "tier": "add",
                "metadata": {"swe_exemplar_link": True}}
        cands.append(_to_candidate(link, doc, 100 + i))
    return cands


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Learning capture: solved run -> exemplar graph edits.")
    ap.add_argument("--traces", nargs="+", default=["data/swe/grounded_traces.jsonl"])
    ap.add_argument("--nodes", nargs="+", default=["artifacts/graph_growth/swe_code_candidates.jsonl"])
    ap.add_argument("--solved", default="",
                    help="predictions jsonl {instance_id, model_patch} the verifier PASSED. "
                         "Default: GOLD patches (clean demo).")
    ap.add_argument("--dataset", default="lite")
    ap.add_argument("--split", default="test")
    ap.add_argument("--out", default="artifacts/graph_growth/exemplar_candidates.jsonl")
    args = ap.parse_args(argv)

    traces = load_traces(args.traces)
    id2text = load_node_texts(args.nodes)
    # patch source: solved predictions, else gold
    if args.solved:
        patch_of = {}
        for line in Path(args.solved).read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line); patch_of[r["instance_id"]] = r.get("model_patch", "")
    else:
        patch_of = {t["instance_id"]: t.get("patch", "") for t in load_instances(args.dataset, args.split, limit=0)}

    cands: List[dict] = []
    n = 0
    for iid, t in traces.items():
        patch = patch_of.get(iid)
        if not (patch and patch.strip()):
            continue
        support = t.get("support_ids") or []
        c = capture(iid, t.get("issue", ""), patch, support, id2text)
        if c:
            cands.extend(c); n += 1

    stitch_candidates(cands)
    wire_hubs(cands)
    out = write_candidate_queue(cands, args.out)
    n_nodes = sum(1 for x in cands if x["raw_edit"]["op"] == "add_node")
    print(f"captured {n} solved instances -> {n_nodes} exemplar nodes -> {out}")
    print(f"  grow the graph: python -m v5.graph_grower.apply --candidates {out} "
          f"--graph graphs/grown_graph4.json --out graphs/grown_graph5.json")
    print("  then retrieve_adapt / the loop retrieves these exemplars -> the system improves.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
