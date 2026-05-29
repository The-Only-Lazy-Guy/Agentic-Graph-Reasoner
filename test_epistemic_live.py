"""Single-question live opencode test for V5 epistemic patch wiring.

Runs ONE Dijkstra-style question through the V4OpencodeController +
answer_query_v4, then dumps:
  - whether the model emitted <patch> blocks
  - extracted graph_edits filtered to V5 node/edge types
  - scoped_patches with status (accept / soft_only / needs_review / reject)
"""

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

os.environ.setdefault("LOCAL_LLM_BASE_URL", "http://127.0.0.1:6768")
os.environ.setdefault("LOCAL_LLM_MAX_TOKENS", "2400")
os.environ.setdefault("LOCAL_LLM_TEMPERATURE", "0.2")

from answerer_v4 import V4OpencodeController, answer_query_v4, _PATCH_BLOCK_RE
from graph_core import MemoryGraph

GRAPH_PATH = Path("graphs/merged_graph.json")

DEFAULT_QUESTION = (
    "Can I use Dijkstra's algorithm if there are negative edges but no negative "
    "cycle? Explain why or why not, and give a safer alternative if any."
)

V5_NODE_TYPES = {"epistemic_state", "solved_subgoal", "strategy",
                 "claim", "fact", "reasoning_atom",
                 "control_rule", "failure_pattern"}
V5_EDGES = {"epistemic_of", "invalidated_by", "requires_slot",
            "transfers_to", "overlaps", "entails", "contradicts"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--question", type=str, default=DEFAULT_QUESTION)
    ap.add_argument("--loop", action="store_true",
                    help="Disable micro_controller finalize shortcut; force tool loop")
    ap.add_argument("--max-steps", type=int, default=6)
    ap.add_argument("--config-dir", type=str, default="pure-opencode")
    ap.add_argument("--apply", action="store_true",
                    help="apply_graph_edits=True to verify on-disk mutation path")
    args = ap.parse_args()

    if not GRAPH_PATH.exists():
        print(f"Graph not found: {GRAPH_PATH}")
        return 1

    cfg_path = Path(args.config_dir).resolve()
    print(f"opencode config_dir: {cfg_path}  exists={cfg_path.exists()}")
    if cfg_path.exists():
        contents = sorted(p.name for p in cfg_path.iterdir() if not p.name.startswith("node_modules"))
        print(f"  config_dir entries: {contents}")

    print(f"Loading graph from {GRAPH_PATH} ...")
    graph = MemoryGraph.load_json(GRAPH_PATH)
    print(f"Graph nodes: {len(graph.nodes)}")

    controller = V4OpencodeController(config_dir=args.config_dir)
    mode_label = "LOOP" if args.loop else "FINALIZE_SHORTCUT_ALLOWED"
    print(f"Mode: {mode_label}")
    print(f"Question: {args.question}\n")
    pre_node_count = len(graph.nodes)
    pre_edge_count = len(graph.edges)
    t0 = time.time()
    packet = answer_query_v4(
        question=args.question,
        graph=graph,
        controller=controller,
        max_steps=args.max_steps,
        collect_corpus=False,
        enforce_recommended_finalize=not args.loop,
        apply_graph_edits=args.apply,
    )
    elapsed = time.time() - t0
    post_node_count = len(graph.nodes)
    post_edge_count = len(graph.edges)
    print(f"graph mutation: +{post_node_count - pre_node_count} nodes, "
          f"+{post_edge_count - pre_edge_count} edges "
          f"(apply_graph_edits={args.apply})")
    new_epistemic_nodes = [
        nid for nid, n in graph.nodes.items()
        if n.node_type == "epistemic_state"
        and (n.metadata or {}).get("model_emitted_patch")
    ]
    print(f"new epistemic_state nodes in graph: {len(new_epistemic_nodes)}")
    for nid in new_epistemic_nodes[:3]:
        meta = graph.nodes[nid].metadata or {}
        print(f"  {nid} -> target={meta.get('target_node_id')} "
              f"confidence={meta.get('confidence')} status={meta.get('status')}")

    print(f"\n=== RUN SUMMARY ===")
    print(f"finalized: {packet.finalized}")
    print(f"execution_mode: {packet.execution_mode}")
    print(f"steps: {packet.steps}")
    print(f"tool_call_count: {packet.tool_call_count}")
    print(f"elapsed_sec: {elapsed:.1f}")
    print(f"answer[:300]: {packet.answer[:300]}")

    # cot_log inspection
    cot_log = packet.cot_log if hasattr(packet, "cot_log") else []
    print(f"\n=== COT_LOG ===")
    print(f"assistant_turns: {len(cot_log)}")
    patch_blocks = []
    for i, turn in enumerate(cot_log):
        if not isinstance(turn, str):
            continue
        for m in _PATCH_BLOCK_RE.finditer(turn):
            patch_blocks.append((i, m.group(1)))
    print(f"raw <patch> blocks captured: {len(patch_blocks)}")
    for i, raw in patch_blocks:
        try:
            obj = json.loads(raw)
            print(f"  turn {i}: op={obj.get('op')} "
                  f"node_type={obj.get('node_type')} "
                  f"relation={obj.get('relation')}")
        except Exception:
            print(f"  turn {i}: MALFORMED JSON: {raw[:120]}")

    # graph_edits filter for V5 items
    print(f"\n=== GRAPH EDITS (V5 only) ===")
    v5_edits = []
    for e in packet.graph_edits or []:
        if e.get("op") == "add_node" and e.get("node_type") in V5_NODE_TYPES \
                and e.get("metadata", {}).get("model_emitted_patch"):
            v5_edits.append(e)
        elif e.get("op") == "add_edge" and e.get("relation") in V5_EDGES \
                and e.get("metadata", {}).get("model_emitted_patch"):
            v5_edits.append(e)
    print(f"V5 model-emitted edits: {len(v5_edits)}")
    for e in v5_edits[:10]:
        print(f"  {json.dumps(e)[:200]}")

    # scoped patches with epistemic / V5 relation patch types
    print(f"\n=== SCOPED PATCHES ===")
    scoped = packet.scoped_patches or []
    epi_patches = [p for p in scoped if p.get("patch_type") == "add_epistemic_state"]
    rel_patches = [p for p in scoped if p.get("patch_type") == "add_relation"
                   and p.get("metadata", {}).get("relation") in V5_EDGES]
    print(f"add_epistemic_state patches: {len(epi_patches)}")
    for p in epi_patches:
        print(f"  status={p.get('validation', {}).get('status')} "
              f"target={p.get('metadata', {}).get('target_node_id')} "
              f"reasons={p.get('validation', {}).get('reasons')}")
    print(f"V5 add_relation patches: {len(rel_patches)}")
    for p in rel_patches:
        print(f"  rel={p.get('metadata', {}).get('relation')} "
              f"status={p.get('validation', {}).get('status')} "
              f"src={p.get('source_id')} dst={p.get('target_id')}")

    print(f"\n=== SUMMARY COUNTS ===")
    summary = packet.scoped_patch_summary or {}
    print(json.dumps({
        "by_status": summary.get("by_status"),
        "by_type": summary.get("by_type"),
        "patch_count": summary.get("patch_count"),
    }, indent=2))

    print("\n=== VERDICT ===")
    if patch_blocks and v5_edits:
        print("PASS: model emitted <patch>, parser extracted V5 edits, scoped layer ran.")
    elif patch_blocks and not v5_edits:
        print("PARTIAL: model emitted <patch> but no V5 edits passed extraction. "
              "Check JSON shape / node_type values.")
    else:
        print("NO PATCH EMISSION: model did not produce <patch> blocks. "
              "Likely needs prompt nudge, or question didn't warrant epistemic state.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
