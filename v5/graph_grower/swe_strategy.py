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
import re
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
    "Each node: one atomic, self-contained idea.\n"
    "HARD RULE -- node text must TRANSFER to other codebases. The node TEXT must NOT "
    "contain: file paths or '.py', function/class/variable names from this repo, "
    "'self.', copied 'def ...(' signatures, or import lines. State the lesson in terms "
    "of the BUG-CLASS and the APPROACH, not this repo's identifiers. The repo-specific "
    "link is carried by the `leveraged` EDGE to the touched symbols, NOT by the text.\n"
    "  GOOD strategy: 'to fix a name-collision when a key contains a separator char, "
    "escape or reject the separator before using the key as a lookup path'.\n"
    "  BAD strategy: 'fix register_blueprint in flask/app.py so self.blueprints handles "
    "dotted names' (repo identifiers -> rejected).\n"
    "node_type one of: reasoning_atom, reasoning_chain, strategy, solved_subgoal.\n"
    "relation one of: chain_step, leveraged, supports, transfers_to.\n"
    'Output ONLY JSON: {"nodes":[{"id":"slug","node_type":"...","text":"..."}],'
    '"edges":[{"src":"slug","dst":"slug","relation":"..."}]}'
)


# High-precision leak markers: a strategy node TEXT containing any of these is
# repo-specific (won't transfer) -> drop the node, keep the leveraged edge to the
# real symbol (the edge is allowed to be specific; the text must be general).
_LEAK_PAT = re.compile(
    r"\.py\b"                                  # file extension
    r"|\bself\."                               # attribute access
    r"|\bdef\s+\w+\s*\("                       # copied signature
    r"|\bimport\s+[\w.]+|\bfrom\s+[\w.]+\s+import\b"   # import lines
    r"|/[\w./-]+\.\w+"                         # path-with-extension
    r"|\b[a-zA-Z_]\w*\.[a-zA-Z_]\w*\.[a-zA-Z_]\w*",    # dotted module path a.b.c (alpha)
    re.IGNORECASE,
)


# Node types that are RETRIEVAL ENTRIES (must transfer across repos) -> linted.
# solved_subgoal is deliberately excluded: it's the instance resolution, grounded by edge.
_LINT_TYPES = {"strategy", "reasoning_atom", "reasoning_chain"}


def _is_leaky(text: str, repo: str = "") -> bool:
    """True if node text carries repo-specific identifiers (path/self./signature/import/
    dotted-module) or the repo's own org/project name verbatim -> not transferable."""
    if not text:
        return False
    if _LEAK_PAT.search(text):
        return True
    low = text.lower()
    return any(t in low for t in (p for p in re.split(r"[/_-]", repo.lower()) if len(p) > 3))


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

    # GENERALITY GATE (type-aware): retrieval-entry nodes (strategy / reasoning_atom /
    # chain) MUST transfer -> drop if repo-specific. solved_subgoal is the instance
    # RESOLUTION (specificity is correct, grounded by its leveraged edge) -> not linted.
    # Dropping a node drops its edges with it. See V5_V2_DESIGN.md / READ_THIS.
    repo = inst.get("repo", "")
    leaky_ids = {ne["node_id"] for ne in conformed["node_edits"]
                 if ne.get("node_type") in _LINT_TYPES and _is_leaky(ne.get("text", ""), repo)}
    kept_nodes = [ne for ne in conformed["node_edits"] if ne["node_id"] not in leaky_ids]
    if leaky_ids:
        print(f"    [leak] dropped {len(leaky_ids)}/{len(conformed['node_edits'])} "
              f"repo-specific nodes ({inst['instance_id']})")
    if not kept_nodes:
        return []   # whole extraction leaked -> nothing transferable

    cands: List[Dict[str, Any]] = []
    for idx, ne in enumerate(kept_nodes):
        ne["metadata"] = {**ne.get("metadata", {}), "swe_strategy": True,
                          "instance_id": inst["instance_id"]}
        cands.append(_to_candidate(ne, doc, idx))
    for idx, ee in enumerate(conformed["edge_edits"]):
        if ee.get("src") in leaky_ids or ee.get("dst") in leaky_ids:
            continue   # edge touching a dropped node
        cands.append(_to_candidate(ee, doc, idx + len(kept_nodes)))
    # link each SURVIVING strategy/atom -> the support symbols (approach grounds onto code)
    new_ids = [ne["node_id"] for ne in kept_nodes]
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
