"""Test: does the model DERIVE implementations from graph evidence, or just recall from training?

Runs 5 implementation-focused questions with anonymized IDs. For each, inspects:
  1. Did the model read graph nodes BEFORE writing code?
  2. Does the implementation cite specific node IDs as its source?
  3. Did the model hypothesize vs directly recall?
  4. Is the code structurally similar to graph content or memorized?

Comparison: also runs each question BARE (no graph) to see if the model
produces the same code — if identical, it's recalling, not learning.

Usage:
    python scripts/test_implementation_learning.py
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
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
GRAPH = "graphs/merged_graph.json"

IMPL_QUESTIONS = [
    {
        "q": "Implement a Fenwick tree in C++ that supports point update and prefix sum query. Show the complete struct with update() and query() methods.",
        "label": "fenwick_impl",
        "graph_has": True,  # graph has cpp_fenwick_tree_template
    },
    {
        "q": "Implement Kadane's algorithm in C++ for maximum contiguous subarray sum. Handle the all-negative case correctly.",
        "label": "kadane_impl",
        "graph_has": True,  # graph has cpp_kadane_template_apply
    },
    {
        "q": "Implement a function that detects whether a directed weighted graph contains a negative-weight cycle using the Bellman-Ford relaxation principle.",
        "label": "neg_cycle_impl",
        "graph_has": True,  # graph has detect_negative_cycle procedure
    },
    {
        "q": "Implement a binary search function in C++ that returns the index of the target in a sorted vector, or -1 if not found. Use the overflow-safe midpoint idiom.",
        "label": "bsearch_impl",
        "graph_has": True,  # graph has cpp_binary_search_apply
    },
    {
        "q": "Implement a trie (prefix tree) in C++ that supports insert(word) and search(word). This is NOT in the knowledge graph — you'll need to reason from general principles.",
        "label": "trie_impl",
        "graph_has": False,  # graph does NOT have trie
    },
]


def run_bare(question: str) -> dict:
    """Run the question through bare opencode (no graph)."""
    exe = shutil.which("opencode") or shutil.which("opencode.cmd")
    env = dict(os.environ)
    env["OPENCODE_CONFIG_DIR"] = CONFIG_DIR
    cmd = [exe, "run", "--model", MODEL, "--format", "json"]
    if SERVER:
        cmd += ["--attach", SERVER]
    prompt = f"Answer this implementation question. Show complete working code.\n\n{question}"
    t0 = time.time()
    proc = subprocess.run(
        cmd, input=prompt, capture_output=True, text=True, timeout=120,
        encoding="utf-8", errors="replace",
        shell=(sys.platform == "win32" and exe.endswith(".cmd")),
        env=env,
    )
    elapsed = time.time() - t0
    parts = []
    for line in proc.stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            evt = json.loads(line)
        except json.JSONDecodeError:
            continue
        if evt.get("type") == "text":
            parts.append(evt.get("part", {}).get("text", ""))
    return {"answer": "".join(parts).strip(), "elapsed": round(elapsed, 1)}


def extract_code_blocks(text: str) -> list:
    """Extract ```...``` code blocks from text."""
    return re.findall(r"```(?:\w+)?\n(.*?)```", text, re.DOTALL)


def count_node_refs(text: str) -> int:
    """Count anonymous node references (node_NNN) in text."""
    return len(re.findall(r"node_\d{3}", text))


def main():
    graph = MemoryGraph.load_json(GRAPH)
    print(f"Graph loaded: {len(graph.nodes)} nodes")
    print(f"Questions: {len(IMPL_QUESTIONS)}\n")

    for i, item in enumerate(IMPL_QUESTIONS):
        q = item["q"]
        label = item["label"]
        has_graph = item["graph_has"]

        print(f"\n{'='*70}")
        print(f"[{i+1}/{len(IMPL_QUESTIONS)}] {label} (graph_has={has_graph})")
        print(f"Q: {q[:80]}")
        print(f"{'='*70}")

        # ── BARE model (no graph) ──
        print("\n--- BARE MODEL ---")
        bare = run_bare(q)
        bare_code = extract_code_blocks(bare["answer"])
        print(f"  elapsed: {bare['elapsed']}s  |  answer: {len(bare['answer'])} chars  |  code blocks: {len(bare_code)}")
        if bare_code:
            print(f"  first code block ({len(bare_code[0])} chars):")
            print("  " + bare_code[0][:300].replace("\n", "\n  "))

        # ── V4 with graph + anonymization ──
        print("\n--- V4 (graph + anonymized IDs) ---")
        ctrl = V4OpencodeController(model=MODEL, server_url=SERVER, config_dir=CONFIG_DIR, timeout=180)
        t0 = time.time()
        pkt = answer_query_v4(
            question=q, graph=graph, controller=ctrl,
            max_steps=12, k_anchors=5,
            enable_activation=False, polish_answer=False, collect_corpus=False,
            anonymize_ids=True,
        )
        v4_elapsed = time.time() - t0
        v4_code = extract_code_blocks(pkt.answer)
        node_refs = count_node_refs(pkt.answer)
        cot_node_refs = count_node_refs("\n".join(pkt.cot_log))

        # Count read_node calls (evidence gathering before implementation)
        reads_before_answer = sum(1 for e in pkt.tool_log if e.get("name") == "read_node")
        hypotheses = len(pkt.hypotheses)
        verified = sum(1 for h in pkt.hypotheses.values() if h.get("verdict") == "verified")

        print(f"  elapsed: {v4_elapsed:.1f}s  |  steps: {pkt.steps}  |  finalized: {pkt.finalized}")
        print(f"  answer: {len(pkt.answer)} chars  |  code blocks: {len(v4_code)}")
        print(f"  read_node calls: {reads_before_answer}  |  node refs in answer: {node_refs}")
        print(f"  node refs in CoT: {cot_node_refs}  |  hypotheses: {hypotheses} ({verified} verified)")

        if v4_code:
            print(f"  first code block ({len(v4_code[0])} chars):")
            print("  " + v4_code[0][:300].replace("\n", "\n  "))

        # ── COMPARISON ──
        print("\n--- ANALYSIS ---")
        if reads_before_answer > 0 and node_refs > 0:
            print("  [GRAPH-DERIVED] Model read graph nodes AND cited them in the answer.")
        elif reads_before_answer > 0 and node_refs == 0:
            print("  [GRAPH-INFORMED] Model read nodes but didn't cite in the answer (may have internalized).")
        elif reads_before_answer == 0:
            print("  [RECALL-ONLY] Model wrote code without reading any graph nodes.")

        if bare_code and v4_code:
            # Crude similarity: shared lines
            bare_lines = set(bare_code[0].strip().splitlines())
            v4_lines = set(v4_code[0].strip().splitlines())
            if bare_lines and v4_lines:
                overlap = len(bare_lines & v4_lines) / min(len(bare_lines), len(v4_lines))
                print(f"  Code overlap (bare vs v4): {overlap*100:.0f}%")
                if overlap > 0.8:
                    print("  [MEMORIZED] Code is near-identical — model likely recalled from training.")
                elif overlap > 0.4:
                    print("  [MIXED] Some overlap — model used graph evidence to modify a known pattern.")
                else:
                    print("  [NOVEL] Implementations differ significantly — graph shaped the output.")

    print(f"\n{'='*70}")
    print("DONE")


if __name__ == "__main__":
    main()
