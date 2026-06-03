"""Build grounded code traces from SWE-bench gold patches (the cheap rung).

DATA only, no Docker, no teacher run. For each SWE instance:
  checkout repo@base_commit -> parse the patch's touched files to symbol nodes
  -> map the gold patch's changed lines to the enclosing symbols (the SUPPORT)
  -> emit a `v2_grounding` row (task=issue, support=touched symbols, brief=neighborhood,
     solution=gold patch, verifier=FAIL_TO_PASS), plus the code-graph nodes and the
     retrieval gold (issue -> touched symbols).

See SWE_DATA_PIPELINE.md. Reuses swe_load (load+checkout) + code_extract (ast parse
+ symbol_at_line). artifact_kind="code_symbol" -> the row is born v2-shaped.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from v5.graph_grower.swe_load import load_instances, checkout_repo, patch_files, SMALL_REPOS
from v5.graph_grower.code_extract import extract_paths, symbol_at_line


def parse_patch_hunks(patch: str) -> Dict[str, List[tuple]]:
    """Gold patch -> {file: [(old_start, old_len), ...]} (old-side = base_commit lines)."""
    res: Dict[str, List[tuple]] = {}
    cur: Optional[str] = None
    for line in (patch or "").splitlines():
        m = re.match(r"^diff --git a/(.+?) b/(.+?)$", line)
        if m:
            cur = m.group(2); res.setdefault(cur, []); continue
        h = re.match(r"^@@ -(\d+)(?:,(\d+))? \+\d+(?:,\d+)? @@", line)
        if h and cur is not None:
            res[cur].append((int(h.group(1)), int(h.group(2) or 1)))
    return res


def touched_symbols(nodes: List[Dict[str, Any]], hunks: Dict[str, List[tuple]]) -> List[str]:
    """Symbols whose span contains any changed (old-side) line of the gold patch."""
    sup: set = set()
    for f, hs in hunks.items():
        for start, length in hs:
            for ln in range(start, start + max(length, 1)):
                sid = symbol_at_line(nodes, f, ln)
                if sid:
                    sup.add(sid)
    return sorted(sup)


def build_instance(inst: Dict[str, Any], repo_root: str | Path) -> Optional[Dict[str, Any]]:
    """Checkout + parse + map -> {trace, gold, nodes, edges} or None on failure."""
    files = [f for f in patch_files(inst["patch"]) if f.endswith(".py")]
    if not files:
        return None
    dest = Path(repo_root) / inst["repo"].replace("/", "__")
    ok, _ = checkout_repo(inst["repo"], inst["base_commit"], dest)
    if not ok:
        return None
    nodes, edges = extract_paths(dest, files, repo=inst["repo"])
    if not nodes:
        return None
    hunks = parse_patch_hunks(inst["patch"])
    support = touched_symbols(nodes, hunks)
    brief = [n["node_id"] for n in nodes
             if n["node_type"] == "symbol" and n["metadata"].get("file") in files]

    trace = {
        "lane": "swe", "instance_id": inst["instance_id"], "repo": inst["repo"],
        "v2_grounding": {
            "task": inst["problem_statement"],
            "artifact_kind": "code_symbol",
            "brief": {"retrieved_ids": brief, "touched_files": files},
            "support_ids": support,
            "overrides_graph": False,
            "solution": inst["patch"],
            "verifier": {"FAIL_TO_PASS": inst.get("FAIL_TO_PASS"),
                         "PASS_TO_PASS": inst.get("PASS_TO_PASS")},
        },
    }
    gold = {"question": inst["problem_statement"], "support_ids": support,
            "instance_id": inst["instance_id"]}
    return {"trace": trace, "gold": gold, "nodes": nodes, "edges": edges}


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Build grounded code traces from SWE-bench gold patches.")
    ap.add_argument("--dataset", default="lite")
    ap.add_argument("--split", default="test")
    ap.add_argument("--limit", type=int, default=5)
    ap.add_argument("--small", action="store_true", help="only small repos (fast)")
    ap.add_argument("--repo-root", default="data/swe_repos")
    ap.add_argument("--out-traces", default="data/swe/grounded_traces.jsonl")
    ap.add_argument("--out-gold", default="data/swe/retrieval_gold_code.jsonl")
    ap.add_argument("--out-graph", default="artifacts/graph_growth/swe_code_candidates.jsonl")
    args = ap.parse_args(argv)

    repos = SMALL_REPOS if args.small else None
    insts = load_instances(args.dataset, args.split, limit=args.limit, repos=repos)
    print(f"instances: {len(insts)}")

    traces, golds, gnodes = [], [], []
    seen_node = set()
    ok = 0
    for inst in insts:
        r = build_instance(inst, args.repo_root)
        if r is None:
            print(f"  SKIP {inst['instance_id']} (no py files / checkout / parse)"); continue
        ok += 1
        traces.append(r["trace"]); golds.append(r["gold"])
        for n in r["nodes"] + r["edges"]:
            key = json.dumps(n, sort_keys=True)
            if key not in seen_node:
                seen_node.add(key); gnodes.append(n)
        sup = r["trace"]["v2_grounding"]["support_ids"]
        brief = r["trace"]["v2_grounding"]["brief"]["retrieved_ids"]
        print(f"  {inst['instance_id']:30} support={len(sup)}/{len(brief)} symbols  files={r['trace']['v2_grounding']['brief']['touched_files']}")

    for path, rows in [(args.out_traces, traces), (args.out_gold, golds)]:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as h:
            for x in rows:
                h.write(json.dumps(x, ensure_ascii=False) + "\n")
    Path(args.out_graph).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_graph, "w", encoding="utf-8") as h:
        for n in gnodes:
            h.write(json.dumps(n, ensure_ascii=False) + "\n")

    print(f"\n{ok}/{len(insts)} instances -> {len(traces)} grounded traces, {len(golds)} gold, "
          f"{len([n for n in gnodes if n.get('op')=='add_node'])} code nodes")
    print(f"  traces -> {args.out_traces}\n  gold   -> {args.out_gold}\n  graph  -> {args.out_graph}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
