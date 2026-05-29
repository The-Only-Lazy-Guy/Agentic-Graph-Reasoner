"""Small batch live test: count <patch> emissions across 5 mixed questions."""

import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("LOCAL_LLM_BASE_URL", "http://127.0.0.1:6768")
os.environ.setdefault("LOCAL_LLM_MAX_TOKENS", "2400")
os.environ.setdefault("LOCAL_LLM_TEMPERATURE", "0.2")

from answerer_v4 import V4OpencodeController, answer_query_v4, _PATCH_BLOCK_RE
from graph_core import MemoryGraph

GRAPH_PATH = Path("graphs/merged_graph.json")

QUESTIONS = [
    ("dijkstra_neg_edge",
     "Can I use Dijkstra's algorithm if there are negative edges but no negative cycle?"),
    ("dijkstra_longest_dag",
     "Can I use Dijkstra to find the longest path in a directed acyclic graph?"),
    ("astronaut_weightless",
     "Why do astronauts in the ISS experience weightlessness despite being only 400km above Earth?"),
    ("ball_bat_math",
     "A bat and a ball cost $1.10 in total. The bat costs $1.00 more than the ball. How much does the ball cost?"),
    ("bst_complexity",
     "What is the time complexity of searching for an element in a balanced binary search tree?"),
]

V5_NODE_TYPES = {"epistemic_state", "solved_subgoal", "strategy", "claim", "fact",
                 "reasoning_atom", "control_rule", "failure_pattern"}
V5_EDGES = {"epistemic_of", "invalidated_by", "requires_slot", "transfers_to",
            "overlaps", "entails", "contradicts"}


def run_one(qid, q):
    graph = MemoryGraph.load_json(GRAPH_PATH)
    controller = V4OpencodeController(config_dir="pure-opencode")
    t0 = time.time()
    pkt = answer_query_v4(
        question=q,
        graph=graph,
        controller=controller,
        max_steps=8,
        collect_corpus=False,
    )
    elapsed = time.time() - t0

    patch_blocks = []
    for turn in (pkt.cot_log or []):
        if isinstance(turn, str):
            patch_blocks.extend(list(_PATCH_BLOCK_RE.finditer(turn)))
    epi_node_edits = [e for e in (pkt.graph_edits or [])
                      if e.get("op") == "add_node"
                      and e.get("node_type") == "epistemic_state"
                      and e.get("metadata", {}).get("model_emitted_patch")]
    v5_edges = [e for e in (pkt.graph_edits or [])
                if e.get("op") == "add_edge"
                and e.get("relation") in V5_EDGES
                and e.get("metadata", {}).get("model_emitted_patch")]
    summary = pkt.scoped_patch_summary or {}
    by_type = summary.get("by_type", {})

    return {
        "qid": qid,
        "execution_mode": pkt.execution_mode,
        "steps": pkt.steps,
        "elapsed_sec": round(elapsed, 1),
        "patch_blocks": len(patch_blocks),
        "epi_node_edits": len(epi_node_edits),
        "v5_edge_edits": len(v5_edges),
        "scoped_epi_accept": (
            sum(1 for p in (pkt.scoped_patches or [])
                if p.get("patch_type") == "add_epistemic_state"
                and p.get("validation", {}).get("status") == "accept")
        ),
        "by_type_epi": by_type.get("add_epistemic_state", 0),
    }


def main():
    results = []
    print(f"{'qid':25s} {'mode':28s} {'steps':>5s} {'elap':>6s} "
          f"{'<patch>':>7s} {'epi':>4s} {'edge':>4s} {'accept':>6s}")
    for qid, q in QUESTIONS:
        try:
            r = run_one(qid, q)
        except Exception as e:
            print(f"{qid:25s} FAILED: {e}")
            continue
        results.append(r)
        print(f"{r['qid']:25s} {r['execution_mode']:28s} "
              f"{r['steps']:>5d} {r['elapsed_sec']:>6.1f} "
              f"{r['patch_blocks']:>7d} {r['epi_node_edits']:>4d} "
              f"{r['v5_edge_edits']:>4d} {r['scoped_epi_accept']:>6d}")

    print("\n=== AGGREGATE ===")
    n = len(results)
    if not n:
        print("no results")
        return 1
    total_patches = sum(r["patch_blocks"] for r in results)
    total_epi = sum(r["epi_node_edits"] for r in results)
    total_edges = sum(r["v5_edge_edits"] for r in results)
    total_accept = sum(r["scoped_epi_accept"] for r in results)
    runs_with_patch = sum(1 for r in results if r["patch_blocks"] > 0)
    print(json.dumps({
        "runs": n,
        "runs_with_any_patch": runs_with_patch,
        "patch_emission_rate": round(runs_with_patch / n, 2),
        "total_patch_blocks": total_patches,
        "total_epistemic_node_edits": total_epi,
        "total_v5_edge_edits": total_edges,
        "total_scoped_epi_accept": total_accept,
    }, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
