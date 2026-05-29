"""Run the full question bank through v4 with graph edits ON.

The graph is loaded ONCE and stays in memory — edits accumulate across
sessions, so later questions benefit from earlier sessions' learning.
A backup is written before any edits.

Usage:
    python scripts/run_batch.py
    python scripts/run_batch.py --limit 5    # quick test
    python scripts/run_batch.py --dry-run    # no graph edits
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from graph_core import MemoryGraph
from answerer_v4 import V4OpencodeController, V4RemoteController, answer_query_v4
from reasoning.task_classifier import TaskClassifier
from reasoning.distillation_corpus import corpus_stats

GRAPH_PATH = Path("graphs/merged_graph.json")
SERVER = "http://127.0.0.1:4096"
CONFIG_DIR = r"C:\Users\Ace\AppData\Local\Temp\opencode-empty-config"
MODEL = "opencode/big-pickle"

QUESTION_BANK = [
    # CS / Algorithms
    {"q": "What is binary search and what is its time complexity?", "domain": "cs"},
    {"q": "Explain why Dijkstra's algorithm fails on graphs with negative edge weights.", "domain": "cs"},
    {"q": "What is a Fenwick tree and how does the lowbit trick (i & -i) work?", "domain": "cs"},
    {"q": "Explain the difference between a segment tree and a Fenwick tree. When would you choose one over the other?", "domain": "cs"},
    {"q": "What is Kadane's algorithm for maximum subarray sum? Walk through it on [-2, 1, -3, 4, -1, 2, 1, -5, 4].", "domain": "cs"},
    {"q": "Design a system that can answer 'what is my rank?' queries in O(log n) for 500K concurrent users with real-time score updates.", "domain": "cs"},
    # CS / Systems
    {"q": "What is a race condition? Give a concrete example with shared mutable state.", "domain": "cs_systems"},
    {"q": "Explain the CAP theorem and give a real-world example of each trade-off.", "domain": "cs_systems"},
    {"q": "What is consistent hashing and why is it used in distributed systems?", "domain": "cs_systems"},
    # Math
    {"q": "Prove that the square root of 2 is irrational.", "domain": "math"},
    {"q": "What is the difference between pointwise and uniform convergence of a sequence of functions?", "domain": "math"},
    {"q": "Explain the Banach fixed-point theorem and give an example application.", "domain": "math"},
    # Physics
    {"q": "Why can light travel through space but sound cannot?", "domain": "physics"},
    {"q": "Explain the photoelectric effect and why it supports the particle theory of light.", "domain": "physics"},
    {"q": "What is renormalization in quantum field theory and why is it necessary?", "domain": "physics"},
    # Chemistry
    {"q": "What determines whether a chemical reaction is exothermic or endothermic?", "domain": "chemistry"},
    {"q": "Explain Le Chatelier's principle with a concrete equilibrium example.", "domain": "chemistry"},
    # Biology
    {"q": "What is the difference between mitosis and meiosis?", "domain": "biology"},
    # Software Engineering
    {"q": "What is the difference between unit tests and integration tests? When would you skip one?", "domain": "software_eng"},
    {"q": "Explain SOLID principles. Which one is most frequently violated in practice?", "domain": "software_eng"},
    # Out-of-domain
    {"q": "Explain how a transformer neural network's self-attention mechanism works, including the Q/K/V matrices.", "domain": "ml"},
    {"q": "What is the difference between REST and GraphQL? When would you choose one over the other?", "domain": "web"},
    {"q": "Explain how Bitcoin's proof-of-work consensus mechanism works and its energy implications.", "domain": "crypto"},
    {"q": "What is the Riemann hypothesis and why does it matter for prime number distribution?", "domain": "math"},
    {"q": "Explain the Byzantine Generals Problem and how PBFT solves it.", "domain": "distributed"},
]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--dry-run", action="store_true", help="skip graph edits")
    ap.add_argument("--no-anonymize", action="store_true", help="disable ID anonymization")
    ap.add_argument("--graph", default=str(GRAPH_PATH))
    args = ap.parse_args()

    bank = QUESTION_BANK
    if args.limit > 0:
        bank = bank[:args.limit]

    apply_edits = not args.dry_run
    anonymize = not args.no_anonymize
    graph_path = Path(args.graph)

    print(f"=== BATCH RUN: {len(bank)} questions ===")
    print(f"  graph: {graph_path}")
    print(f"  apply_graph_edits: {apply_edits}")
    print(f"  anonymize_ids: {anonymize}")
    print(f"  model: {MODEL}")
    print()

    # Backup the graph before any edits
    if apply_edits:
        backup_path = graph_path.with_suffix(".pre_batch_backup.json")
        shutil.copy2(graph_path, backup_path)
        print(f"  Graph backup: {backup_path}")

    graph = MemoryGraph.load_json(str(graph_path))
    nodes_before = len(graph.nodes)
    edges_before = len(graph.edges)
    print(f"  Graph loaded: {nodes_before} nodes, {edges_before} edges")

    classifier = TaskClassifier.load()
    corpus_before = corpus_stats()
    print(f"  Corpus before: {corpus_before.get('rows', 0)} rows\n")

    results = []
    total_t0 = time.time()

    for i, item in enumerate(bank):
        q = item["q"]
        domain = item.get("domain", "?")
        print(f"[{i+1}/{len(bank)}] ({domain}) {q[:65]}{'...' if len(q) > 65 else ''}")

        ctrl = V4RemoteController(model_label=MODEL)
        t0 = time.time()
        try:
            pkt = answer_query_v4(
                question=q,
                graph=graph,
                controller=ctrl,
                auto_config=True,
                classifier=classifier,
                collect_corpus=True,
                controller_label=MODEL,
                apply_graph_edits=apply_edits,
                anonymize_ids=anonymize,
            )
            elapsed = time.time() - t0
            row = {
                "index": i,
                "question": q,
                "domain": domain,
                "finalized": pkt.finalized,
                "steps": pkt.steps,
                "tool_calls": pkt.tool_call_count,
                "elapsed_sec": round(elapsed, 1),
                "answer_len": len(pkt.answer),
                "graph_edits": len(pkt.graph_edits),
                "graph_edits_applied": pkt.graph_edits_applied,
                "scoped_patches": len(getattr(pkt, "scoped_patches", []) or []),
                "scoped_patch_summary": getattr(pkt, "scoped_patch_summary", {}) or {},
                "coverage_pct": round(pkt.coverage_addressed_pct * 100, 1),
                "session_dir": pkt.session_dir,
                "error": None,
            }
            status = "OK" if pkt.finalized else "NOT FINALIZED"
            print(f"  {status} | steps={pkt.steps} tools={pkt.tool_call_count} "
                  f"edits={len(pkt.graph_edits)}(applied={pkt.graph_edits_applied}) "
                  f"patches={len(getattr(pkt, 'scoped_patches', []) or [])} "
                  f"elapsed={elapsed:.1f}s answer={len(pkt.answer)}ch")
        except Exception as e:
            elapsed = time.time() - t0
            row = {
                "index": i, "question": q, "domain": domain,
                "finalized": False, "error": f"{type(e).__name__}: {str(e)[:200]}",
                "elapsed_sec": round(elapsed, 1),
            }
            print(f"  ERROR: {type(e).__name__}: {str(e)[:100]} ({elapsed:.1f}s)")
        results.append(row)

    total_elapsed = time.time() - total_t0

    # Save the updated graph if edits were applied
    nodes_after = len(graph.nodes)
    edges_after = len(graph.edges)
    if apply_edits and (nodes_after != nodes_before or edges_after != edges_before):
        graph.save_json(str(graph_path))
        print(f"\n  Graph saved: {graph_path}")

    # Write batch report
    report_path = Path("data/batch_report.json")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "total_questions": len(bank),
        "finalized": sum(1 for r in results if r.get("finalized")),
        "errors": sum(1 for r in results if r.get("error")),
        "total_elapsed_sec": round(total_elapsed, 1),
        "mean_elapsed_sec": round(total_elapsed / len(bank), 1),
        "graph_before": {"nodes": nodes_before, "edges": edges_before},
        "graph_after": {"nodes": nodes_after, "edges": edges_after},
        "nodes_added": nodes_after - nodes_before,
        "edges_added": edges_after - edges_before,
        "apply_graph_edits": apply_edits,
        "results": results,
    }
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    # Summary
    corpus_after = corpus_stats()
    print(f"\n{'='*60}")
    print(f"=== BATCH COMPLETE ===")
    print(f"{'='*60}")
    print(f"  questions:   {len(bank)}")
    print(f"  finalized:   {report['finalized']}/{len(bank)}")
    print(f"  errors:      {report['errors']}")
    print(f"  total time:  {total_elapsed:.0f}s ({total_elapsed/60:.1f}min)")
    print(f"  mean time:   {total_elapsed/len(bank):.1f}s per question")
    print(f"  graph delta: {nodes_before} -> {nodes_after} nodes (+{nodes_after - nodes_before})")
    print(f"               {edges_before} -> {edges_after} edges (+{edges_after - edges_before})")
    print(f"  corpus:      {corpus_before.get('rows', 0)} -> {corpus_after.get('rows', 0)} rows")
    print(f"  report:      {report_path}")

    failed = [r for r in results if not r.get("finalized")]
    if failed:
        print(f"\n  Not finalized ({len(failed)}):")
        for r in failed:
            print(f"    [{r['index']+1}] {r['question'][:60]} — {r.get('error', 'max steps reached')}")


if __name__ == "__main__":
    main()
