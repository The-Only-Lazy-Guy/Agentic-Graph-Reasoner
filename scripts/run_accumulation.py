"""Multi-round accumulation: run question bank N times, graph grows each round.

Each round:
  1. Load the current graph (with previous rounds' edits)
  2. Run a subset of questions from the question bank
  3. Apply graph edits (health-gated, auto-connected)
  4. Save the graph for the next round
  5. Report: finalization rate, graph growth, health delta

After all rounds, format the accumulated corpus for SFT.

Usage:
    python scripts/run_accumulation.py --rounds 3 --per-round 50
    python scripts/run_accumulation.py --rounds 3 --per-round 25 --graph-only
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from graph_core import MemoryGraph
from answerer_v4 import V4OpencodeController, V4RemoteController, answer_query_v4
from reasoning.task_classifier import TaskClassifier
from reasoning.graph_health import compute_health
from reasoning.distillation_corpus import corpus_stats

SERVER = "http://127.0.0.1:4096"
CONFIG_DIR = r"C:\Users\Ace\AppData\Local\Temp\opencode-empty-config"
MODEL = "opencode/big-pickle"
GRAPH_PATH = Path("graphs/merged_graph.json")
QUESTION_BANK = Path("data/question_bank.jsonl")


def load_question_bank(path: Path) -> list:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return rows


def run_round(
    round_num: int,
    questions: list,
    graph: MemoryGraph,
    *,
    graph_only: bool = False,
    anonymize: bool = True,
) -> dict:
    """Run one round of questions on the current graph state."""
    classifier = TaskClassifier.load()
    health_before = compute_health(graph)
    nodes_before = len(graph.nodes)
    edges_before = len(graph.edges)
    t0 = time.time()

    results = []
    for i, item in enumerate(questions):
        q = item["q"]
        domain = item.get("domain", "?")
        diff = item.get("difficulty", "?")
        print(f"  [{i+1}/{len(questions)}] ({domain}/{diff}) {q[:55]}{'...' if len(q)>55 else ''}", end="", flush=True)

        ctrl = V4RemoteController(model_label=f"{MODEL}_round{round_num}")
        try:
            pkt = answer_query_v4(
                question=q, graph=graph, controller=ctrl,
                auto_config=True, classifier=classifier,
                collect_corpus=True, controller_label=f"{MODEL}_round{round_num}",
                apply_graph_edits=True,
                anonymize_ids=anonymize,
                graph_only_answer=graph_only,
            )
            print(f" => {'OK' if pkt.finalized else 'INCOMPLETE'} ({pkt.steps}s/{pkt.elapsed_sec:.0f}s)")
            results.append({
                "question": q, "domain": domain, "difficulty": diff,
                "finalized": pkt.finalized, "steps": pkt.steps,
                "elapsed": pkt.elapsed_sec, "answer_len": len(pkt.answer),
                "edits": len(pkt.graph_edits),
            })
        except Exception as e:
            print(f" => ERROR: {type(e).__name__}")
            results.append({
                "question": q, "domain": domain, "difficulty": diff,
                "finalized": False, "error": str(e)[:100],
            })

    elapsed = time.time() - t0
    health_after = compute_health(graph)

    return {
        "round": round_num,
        "questions": len(questions),
        "finalized": sum(1 for r in results if r.get("finalized")),
        "errors": sum(1 for r in results if r.get("error")),
        "elapsed_sec": round(elapsed, 1),
        "graph_before": {"nodes": nodes_before, "edges": edges_before},
        "graph_after": {"nodes": len(graph.nodes), "edges": len(graph.edges)},
        "health_before": round(health_before.health_score, 4),
        "health_after": round(health_after.health_score, 4),
        "health_delta": round(health_after.health_score - health_before.health_score, 4),
        "results": results,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--per-round", type=int, default=50, help="questions per round")
    ap.add_argument("--graph", default=str(GRAPH_PATH))
    ap.add_argument("--bank", default=str(QUESTION_BANK))
    ap.add_argument("--graph-only", action="store_true")
    ap.add_argument("--no-anonymize", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed)
    bank = load_question_bank(Path(args.bank))
    print(f"Question bank: {len(bank)} questions")

    graph = MemoryGraph.load_json(args.graph)
    # Backup
    backup = Path(args.graph).with_suffix(".pre_accumulation_backup.json")
    graph.save_json(str(backup))
    print(f"Graph: {len(graph.nodes)} nodes (backed up to {backup})")
    print(f"Rounds: {args.rounds}, questions/round: {args.per_round}")
    print(f"Graph-only: {args.graph_only}, Anonymize: {not args.no_anonymize}\n")

    round_reports = []
    corpus_start = corpus_stats().get("rows", 0)

    for r in range(1, args.rounds + 1):
        print(f"\n{'='*60}")
        print(f"ROUND {r}/{args.rounds}")
        print(f"{'='*60}")

        # Sample questions for this round (different each round for diversity)
        sample = random.sample(bank, min(args.per_round, len(bank)))

        report = run_round(
            r, sample, graph,
            graph_only=args.graph_only,
            anonymize=not args.no_anonymize,
        )
        round_reports.append(report)

        # Save graph after each round
        graph.save_json(args.graph)

        print(f"\n  Round {r} summary:")
        print(f"    finalized: {report['finalized']}/{report['questions']}")
        print(f"    graph: {report['graph_before']['nodes']} -> {report['graph_after']['nodes']} "
              f"(+{report['graph_after']['nodes'] - report['graph_before']['nodes']})")
        print(f"    health: {report['health_before']} -> {report['health_after']} "
              f"({report['health_delta']:+.4f})")
        print(f"    elapsed: {report['elapsed_sec']:.0f}s")

    # Final summary
    corpus_end = corpus_stats().get("rows", 0)
    print(f"\n{'='*60}")
    print(f"ACCUMULATION COMPLETE")
    print(f"{'='*60}")
    print(f"  Rounds:     {args.rounds}")
    print(f"  Total Qs:   {sum(r['questions'] for r in round_reports)}")
    print(f"  Finalized:  {sum(r['finalized'] for r in round_reports)}")
    print(f"  Corpus:     {corpus_start} -> {corpus_end} rows (+{corpus_end - corpus_start})")
    print(f"\n  Round-by-round:")
    print(f"  {'round':>5s}  {'finalized':>10s}  {'nodes':>8s}  {'health':>8s}  {'delta':>8s}")
    for rr in round_reports:
        print(f"  {rr['round']:>5d}  {rr['finalized']:>5d}/{rr['questions']:<4d}  "
              f"{rr['graph_after']['nodes']:>8d}  {rr['health_after']:>8.4f}  {rr['health_delta']:>+8.4f}")

    # Save accumulation report
    report_path = Path("data/accumulation_report.json")
    report_path.write_text(json.dumps(round_reports, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n  Report: {report_path}")

    # Auto-format SFT data
    print(f"\n  Formatting SFT dataset...")
    from scripts.format_sft_data import main as format_main
    sys.argv = ["format_sft_data"]
    try:
        format_main()
    except SystemExit:
        pass


if __name__ == "__main__":
    main()
