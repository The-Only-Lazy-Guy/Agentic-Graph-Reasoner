"""Verify the feedback loop proves: never done -> derives -> learns -> reuses faster.

Test sequence:
  1. Confirm the graph does NOT have the target implementation
  2. Phase 1: model derives the implementation from graph building blocks
  3. Reflection adds the implementation to the graph as a new node
  4. Phase 2: model finds the learned implementation, cites it, answers FASTER

The loop is closed when Phase 2 reads Phase 1's new node and finishes faster.

Usage:
    python scripts/verify_feedback_loop.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from graph_core import MemoryGraph
from answerer_v4 import V4OpencodeController, answer_query_v4

SERVER = "http://127.0.0.1:4096"
CONFIG_DIR = r"C:\Users\Ace\AppData\Local\Temp\opencode-empty-config"
MODEL = "opencode/big-pickle"
GRAPH_PATH = "graphs/merged_graph.json"

# The question must target something the graph CAN'T do yet but CAN derive
# from building blocks it already has.
QUESTION = (
    "Implement find_kth (order statistic query) for a Fenwick tree using "
    "binary lifting. The function should find the smallest index such that "
    "prefix_sum(index) >= k, in O(log n). Show complete C++ code."
)

# Building blocks the graph HAS:
EXPECTED_BUILDING_BLOCKS = [
    "cpp_fenwick_tree_template",    # Fenwick tree struct
    "cpp_lowbit_idiom",             # i & -i trick
    "fenwick_update_and_prefix_query_log_n",  # O(log n) operations
]


def run_session(question, graph, label, **kwargs):
    ctrl = V4OpencodeController(
        model=MODEL, server_url=SERVER, config_dir=CONFIG_DIR, timeout=180,
    )
    t0 = time.time()
    pkt = answer_query_v4(
        question=question, graph=graph, controller=ctrl,
        max_steps=15, k_anchors=5,
        anonymize_ids=False,  # real IDs so we can track what gets read
        graph_only_answer=True,
        collect_corpus=True,
        controller_label=MODEL,
        polish_answer=False,
        apply_graph_edits=True,
        run_reflection_inline=True,
        **kwargs,
    )
    wall = time.time() - t0
    return pkt, wall


def nodes_read(tool_log):
    return [e["args"]["node_id"] for e in tool_log
            if e.get("name") == "read_node" and "node_id" in e.get("args", {})]


def main():
    print("="*70)
    print("FEEDBACK LOOP VERIFICATION")
    print("Proving: never done -> derives -> learns -> reuses faster")
    print("="*70)

    graph = MemoryGraph.load_json(GRAPH_PATH)
    print(f"\nGraph: {len(graph.nodes)} nodes, {len(graph.edges)} edges")

    # Step 1: Confirm the target is NOT in the graph
    print(f"\nQuestion: {QUESTION[:80]}...")
    print(f"\n--- STEP 1: Confirm graph does NOT have find_kth ---")
    has_find_kth = any(
        "find_kth" in n.text.lower() or "find kth" in n.text.lower() or "order statistic" in n.text.lower()
        for n in graph.nodes.values()
    )
    has_blocks = [bid for bid in EXPECTED_BUILDING_BLOCKS if bid in graph.nodes]
    print(f"  find_kth/order-statistic in graph: {has_find_kth}")
    print(f"  Building blocks present: {has_blocks}")
    if has_find_kth:
        print("  WARNING: graph already has find_kth content. Results may not prove derivation.")

    # Step 2: Phase 1 -- derive the implementation
    print(f"\n--- STEP 2: PHASE 1 -- Derive implementation from building blocks ---")
    nodes_before = set(graph.nodes.keys())
    pkt1, wall1 = run_session(QUESTION, graph, "PHASE 1")
    nodes_after_p1 = set(graph.nodes.keys())
    new_nodes_p1 = nodes_after_p1 - nodes_before
    reads_p1 = nodes_read(pkt1.tool_log)
    blocks_read = [bid for bid in EXPECTED_BUILDING_BLOCKS if bid in reads_p1]

    print(f"\n  Phase 1 results:")
    print(f"    finalized: {pkt1.finalized}")
    print(f"    steps: {pkt1.steps}  |  tool calls: {pkt1.tool_call_count}  |  wall: {wall1:.1f}s")
    print(f"    answer: {len(pkt1.answer)} chars")
    print(f"    building blocks read: {blocks_read}")
    print(f"    nodes read total: {len(reads_p1)}")
    print(f"    NEW NODES ADDED: {len(new_nodes_p1)}")
    for nid in sorted(new_nodes_p1)[:5]:
        n = graph.nodes[nid]
        print(f"      {nid} [{n.node_type}]: {n.text[:100]}")

    if not new_nodes_p1:
        print("\n  NO NEW NODES from Phase 1. Reflection may not have fired or")
        print("  produced no add_node edits. Loop test cannot proceed.")
        # Show what edits were proposed
        print(f"  graph_edits proposed: {len(pkt1.graph_edits)}")
        for e in pkt1.graph_edits[:5]:
            print(f"    {e.get('op')} / {e.get('tier')}: {e.get('text','')[:60]}")
        if pkt1.reflection:
            print(f"  reflection new_facts: {len(pkt1.reflection.get('new_facts', []))}")
            print(f"  reflection implementations: {len(pkt1.reflection.get('implementations', []))}")
        return

    # Step 3: Phase 2 -- reuse the learned implementation
    print(f"\n--- STEP 3: PHASE 2 -- Reuse the learned implementation ---")
    print(f"  Graph now has {len(graph.nodes)} nodes (+{len(new_nodes_p1)} from Phase 1)")
    pkt2, wall2 = run_session(QUESTION, graph, "PHASE 2")
    reads_p2 = nodes_read(pkt2.tool_log)
    new_nodes_read_in_p2 = [nid for nid in reads_p2 if nid in new_nodes_p1]

    print(f"\n  Phase 2 results:")
    print(f"    finalized: {pkt2.finalized}")
    print(f"    steps: {pkt2.steps}  |  tool calls: {pkt2.tool_call_count}  |  wall: {wall2:.1f}s")
    print(f"    answer: {len(pkt2.answer)} chars")
    print(f"    nodes read total: {len(reads_p2)}")
    print(f"    PHASE 1 NODES READ: {new_nodes_read_in_p2}")

    # Also check if Phase 1's node content appears in Phase 2's answer
    content_in_answer = False
    for nid in new_nodes_p1:
        node_text = graph.nodes[nid].text[:50].lower()
        if node_text in pkt2.answer.lower():
            content_in_answer = True
            break

    # Step 4: Verdict
    print(f"\n{'='*70}")
    print(f"FEEDBACK LOOP VERDICT")
    print(f"{'='*70}")

    checks = {
        "1. Graph lacked find_kth before": not has_find_kth,
        "2. Phase 1 derived implementation": pkt1.finalized and len(pkt1.answer) > 100,
        "3. Phase 1 added new nodes to graph": len(new_nodes_p1) > 0,
        "4. Phase 2 read Phase 1's nodes": len(new_nodes_read_in_p2) > 0 or content_in_answer,
        "5. Phase 2 was FASTER (fewer steps)": pkt2.steps < pkt1.steps,
        "6. Phase 2 was FASTER (wall time)": wall2 < wall1,
    }
    for check, passed in checks.items():
        print(f"  [{'PASS' if passed else 'FAIL'}] {check}")
        if "Phase 1 added" in check and passed:
            print(f"        ({len(new_nodes_p1)} new nodes)")
        if "Phase 2 read" in check:
            print(f"        (direct reads: {new_nodes_read_in_p2}, content match: {content_in_answer})")
        if "fewer steps" in check:
            print(f"        (P1: {pkt1.steps} steps, P2: {pkt2.steps} steps)")
        if "wall time" in check:
            print(f"        (P1: {wall1:.1f}s, P2: {wall2:.1f}s)")

    all_pass = all(checks.values())
    print(f"\n  LOOP {'CLOSED' if all_pass else 'NOT FULLY CLOSED'}")
    if all_pass:
        print("  The model derived an implementation, learned it, and reused it faster.")
    else:
        failed = [k for k, v in checks.items() if not v]
        print(f"  Failed checks: {failed}")


if __name__ == "__main__":
    main()
