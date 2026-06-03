"""Distill reusable debugging STRATEGY nodes from SWE gold fixes (V4 session-graph).

The cheap rung gives SYMBOL nodes (what exists). This adds STRATEGY / reasoning nodes
(HOW to fix a bug-class), distilled by the teacher (opencode/codex) from
(issue + gold patch + touched signatures) -- one call per instance, NO Docker/verifier
(the gold patch is given, so the teacher just summarizes the fix into a reusable
lesson). Strategy nodes link (`leveraged`) to the support symbols, so approach-grounding
sits on top of token-grounding. Conform/stitch/hub-wire reuse the grower. Runs on a
cheap cloud box (opencode = cloud model, no GPU). See V5_V2_DESIGN.md (strategy nodes,
declarative -- the model attends them, it does NOT execute them).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from v5.graph_grower.extract import (
    Document, parse_extraction, conform_edits, stitch_candidates, wire_hubs,
    write_candidate_queue, _to_candidate, _build_opencode_controller,
)
from v5.graph_grower.swe_load import load_instances, SMALL_REPOS
from v5.graph_grower.swe_grounded import build_instance

STRATEGY_SYS = (
    "You distill a REUSABLE debugging strategy from a resolved bug fix, as atomic "
    "knowledge-graph nodes. You are given a bug report, the fix (diff), and the "
    "signatures of the functions it touched. Extract:\n"
    "  - the root cause as reasoning_atom node(s) (what was actually wrong);\n"
    "  - the fix STRATEGY as a `strategy` node: a GENERALIZABLE lesson of the form "
    "'to fix <bug-class>, <approach>' -- reusable on OTHER code, not specific to this repo;\n"
    "  - the resolved approach as a `solved_subgoal` node.\n"
    "Each node: one atomic, self-contained idea. Prefer general lessons over instance "
    "details (no file paths / variable names unless essential).\n"
    "node_type one of: reasoning_atom, reasoning_chain, strategy, solved_subgoal.\n"
    "relation one of: chain_step, leveraged, supports, transfers_to.\n"
    'Output ONLY JSON: {"nodes":[{"id":"slug","node_type":"...","text":"..."}],'
    '"edges":[{"src":"slug","dst":"slug","relation":"..."}]}'
)


def _build_user(inst: Dict[str, Any], support_sigs: List[str]) -> str:
    issue = (inst.get("problem_statement") or "")[:2000]
    patch = (inst.get("patch") or "")[:2500]
    sigs = "\n".join(f"  - {s}" for s in support_sigs[:12])
    return (f"BUG REPORT:\n{issue}\n\nTOUCHED FUNCTIONS:\n{sigs}\n\n"
            f"FIX (diff):\n{patch}\n\nExtract the reusable strategy as JSON.")


def _make_controller(backend: str, opencode_config_dir: str, model: Optional[str]):
    if backend == "codex":
        from answerer_v4 import V4CodexController
        return V4CodexController(model=model, print_raw_output=False)
    return _build_opencode_controller(model, opencode_config_dir)


def extract_strategy(inst: Dict[str, Any], controller, repo_root: str) -> List[Dict[str, Any]]:
    """One instance -> strategy/reasoning candidate edits linked to the support symbols."""
    r = build_instance(inst, repo_root)   # checkout + symbols + support (reused)
    if r is None:
        return []
    support_ids = r["trace"]["v2_grounding"]["support_ids"]
    id2text = {n["node_id"]: n.get("text", "") for n in r["nodes"]}
    support_sigs = [id2text[s] for s in support_ids if s in id2text]

    resp = controller.chat_oneshot([
        {"role": "system", "content": STRATEGY_SYS},
        {"role": "user", "content": _build_user(inst, support_sigs)},
    ])
    raw = resp["choices"][0]["message"]["content"]
    doc = Document(id=inst["instance_id"], text="", domain=inst["repo"], mode="cot")
    conformed = conform_edits(parse_extraction(raw), doc)

    cands: List[Dict[str, Any]] = []
    for idx, ne in enumerate(conformed["node_edits"]):
        ne["metadata"] = {**ne.get("metadata", {}), "swe_strategy": True,
                          "instance_id": inst["instance_id"]}
        cands.append(_to_candidate(ne, doc, idx))
    for idx, ee in enumerate(conformed["edge_edits"]):
        cands.append(_to_candidate(ee, doc, idx + len(conformed["node_edits"])))
    # link each new strategy/atom -> the support symbols (approach grounds onto code)
    new_ids = [ne["node_id"] for ne in conformed["node_edits"]]
    for i, nid in enumerate(new_ids):
        for sid in support_ids:
            link = {"op": "add_edge", "src": nid, "dst": sid, "relation": "leveraged",
                    "tier": "add", "metadata": {"swe_strategy_link": True}}
            cands.append(_to_candidate(link, doc, 10000 + i))
    return cands


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Distill SWE gold fixes -> strategy nodes (teacher).")
    ap.add_argument("--dataset", default="lite")
    ap.add_argument("--split", default="test")
    ap.add_argument("--limit", type=int, default=20)
    ap.add_argument("--small", action="store_true")
    ap.add_argument("--backend", choices=["opencode", "codex"], default="opencode")
    ap.add_argument("--opencode-config-dir", default="pure-opencode")
    ap.add_argument("--model", default=None)
    ap.add_argument("--repo-root", default="data/swe_repos")
    ap.add_argument("--shard-index", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--out", default="artifacts/graph_growth/swe_strategy_candidates.jsonl")
    args = ap.parse_args(argv)

    repos = SMALL_REPOS if args.small else None
    insts = load_instances(args.dataset, args.split, limit=0, repos=repos)
    insts = [t for i, t in enumerate(insts) if i % args.num_shards == args.shard_index]
    if args.limit:
        insts = insts[: args.limit]
    print(f"strategy distill: {len(insts)} instances  backend={args.backend}")

    controller = _make_controller(args.backend, args.opencode_config_dir, args.model)
    cands: List[Dict[str, Any]] = []
    ok = 0
    for inst in insts:
        try:
            c = extract_strategy(inst, controller, args.repo_root)
        except Exception as e:   # noqa: BLE001
            print(f"  ERR {inst['instance_id']}: {repr(e)[:80]}"); continue
        if c:
            ok += 1
            n_nodes = sum(1 for x in c if x["raw_edit"]["op"] == "add_node")
            cands.extend(c)
            print(f"  {inst['instance_id']:30} +{n_nodes} strategy nodes")

    stitch_candidates(cands)
    wire_hubs(cands)
    out = write_candidate_queue(cands, args.out)
    n_nodes = sum(1 for x in cands if x["raw_edit"]["op"] == "add_node")
    print(f"\n{ok}/{len(insts)} instances -> {n_nodes} strategy nodes -> {out}")
    print(f"  apply: python -m v5.graph_grower.apply --candidates {out} "
          f"--graph <code_graph.json> --out <grown_code_graph.json>")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
