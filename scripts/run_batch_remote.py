"""Batch runner for remote endpoint.

Stops on HTTP error, logs progress to Discord webhook every N samples,
and sends a final summary when done or stopped.

Usage:
    python scripts/run_batch_remote.py --apply-edits
    python scripts/run_batch_remote.py --limit 100 --apply-edits
    python scripts/run_batch_remote.py --start-from 50 --apply-edits  # resume
    python scripts/run_batch_remote.py --log-every 5  # discord update every 5
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from graph_core import MemoryGraph
from answerer_v4 import V4RemoteController, answer_query_v4
from reasoning.task_classifier import TaskClassifier
from reasoning.distillation_corpus import corpus_stats

GRAPH_PATH = Path("graphs/merged_graph.json")
QUESTION_BANK = Path("data/question_bank_v2.jsonl")
WEBHOOK_URL = "https://discord.com/api/webhooks/1499461304220516413/Ssfbyj-DfqDl_y7i04LyrhAOc4-sFyyzfk6SXBRwk7q7JkFXlbsvz2-OjyhSWmzs2l63"


def send_discord(message: str, webhook: str = WEBHOOK_URL) -> None:
    """Send a message to Discord webhook. Fails silently."""
    try:
        import requests
        # Discord limit is 2000 chars
        msg = message[:1950]
        requests.post(webhook, json={"content": msg}, timeout=10)
    except Exception:
        pass


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--limit", type=int, default=0, help="max questions (0=all)")
    ap.add_argument("--start-from", type=int, default=0, help="skip first N (resume)")
    ap.add_argument("--graph", default=str(GRAPH_PATH))
    ap.add_argument("--bank", default=str(QUESTION_BANK))
    ap.add_argument("--max-steps", type=int, default=12)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--apply-edits", action="store_true", help="apply graph edits")
    ap.add_argument("--log-every", type=int, default=10, help="discord log interval")
    ap.add_argument("--webhook", default=WEBHOOK_URL)
    args = ap.parse_args()

    random.seed(args.seed)

    bank = []
    with Path(args.bank).open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    bank.append(json.loads(line))
                except json.JSONDecodeError:
                    continue

    bank = bank[args.start_from:]
    if args.limit > 0:
        bank = bank[:args.limit]

    graph = MemoryGraph.load_json(args.graph)
    classifier = TaskClassifier.load()
    corpus_before = corpus_stats().get("rows", 0)

    print(f"=== BATCH RUN (remote) ===")
    print(f"  Questions: {len(bank)} (start_from={args.start_from})")
    print(f"  Graph: {len(graph.nodes)} nodes")
    print(f"  Corpus before: {corpus_before} rows")
    print(f"  Apply edits: {args.apply_edits}")
    print(f"  Discord log every: {args.log_every}")
    print()

    send_discord(
        f"**Batch started**\n"
        f"Questions: {len(bank)} (from #{args.start_from})\n"
        f"Graph: {len(graph.nodes)} nodes\n"
        f"Corpus: {corpus_before} rows"
    )

    ok = 0
    incomplete = 0
    t0 = time.time()
    stopped_at = args.start_from + len(bank)  # will update if stopped early

    for i, item in enumerate(bank):
        q = item["q"]
        domain = item.get("domain", "?")
        diff = item.get("difficulty", "?")
        qtype = item.get("type", "?")
        idx = args.start_from + i
        print(f"[{idx+1}] ({domain}/{qtype}) {q[:55]}...", end="", flush=True)

        ctrl = V4RemoteController()
        try:
            pkt = answer_query_v4(
                question=q, graph=graph, controller=ctrl,
                auto_config=True, classifier=classifier,
                max_steps=args.max_steps, k_anchors=5,
                collect_corpus=True,
                controller_label="remote/big-pickle",
                apply_graph_edits=args.apply_edits,
                anonymize_ids=True,
                polish_answer=False,
            )
            if pkt.finalized:
                print(f" => OK (steps={pkt.steps}, {len(pkt.answer)}ch)")
                ok += 1
            else:
                print(f" => INCOMPLETE (steps={pkt.steps})")
                incomplete += 1
        except KeyboardInterrupt:
            stopped_at = idx
            print(f" => INTERRUPTED")
            send_discord(
                f"**Batch interrupted** by user at #{idx+1}\n"
                f"OK: {ok} | Incomplete: {incomplete}\n"
                f"Resume with: `--start-from {idx}`"
            )
            break
        except Exception as e:
            stopped_at = idx
            err_type = type(e).__name__
            err_msg = str(e)[:200]
            print(f" => ERROR: {err_type}: {err_msg}")

            # Save graph before stopping
            if args.apply_edits:
                graph.save_json(args.graph)

            cs = corpus_stats()
            elapsed = time.time() - t0
            send_discord(
                f"**Batch STOPPED** at #{idx+1} -- {err_type}\n"
                f"```{err_msg[:300]}```\n"
                f"OK: {ok} | Incomplete: {incomplete}\n"
                f"Corpus: {corpus_before} -> {cs.get('rows', 0)} (+{cs.get('rows', 0) - corpus_before})\n"
                f"Elapsed: {elapsed:.0f}s\n"
                f"**Resume with:** `--start-from {idx}`"
            )
            print(f"\n  STOPPED. Resume with: --start-from {idx}")
            break

        # Periodic discord log
        if (i + 1) % args.log_every == 0:
            cs = corpus_stats()
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed * 60 if elapsed > 0 else 0
            send_discord(
                f"**Progress** #{idx+1}/{args.start_from + len(bank)}\n"
                f"OK: {ok} | Incomplete: {incomplete}\n"
                f"Corpus: {cs.get('rows', 0)} rows | Graph: {len(graph.nodes)} nodes\n"
                f"Rate: {rate:.1f} q/min | Elapsed: {elapsed:.0f}s"
            )
    else:
        # Completed all questions (no break)
        stopped_at = args.start_from + len(bank)

    # Final save + report
    if args.apply_edits:
        graph.save_json(args.graph)

    elapsed = time.time() - t0
    cs = corpus_stats()
    corpus_after = cs.get("rows", 0)

    summary = (
        f"**Batch complete**\n"
        f"Range: #{args.start_from+1} to #{stopped_at}\n"
        f"OK: {ok} | Incomplete: {incomplete}\n"
        f"Corpus: {corpus_before} -> {corpus_after} (+{corpus_after - corpus_before})\n"
        f"Graph: {len(graph.nodes)} nodes\n"
        f"Time: {elapsed:.0f}s ({elapsed/60:.1f}min)\n"
        f"Rate: {(ok+incomplete)/elapsed*60:.1f} q/min" if elapsed > 0 else ""
    )
    print(f"\n{'='*50}")
    print(summary)
    send_discord(summary)


if __name__ == "__main__":
    main()
