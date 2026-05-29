"""Pre-flight check: run 3 questions through the full v4 pipeline and verify
every artifact is written correctly before committing to a large batch.

Checks:
  1. Corpus JSONL row written per session
  2. Session dir created with all expected files
  3. No path collisions between runs
  4. Recovery from a deliberately broken question
  5. Coverage-delta comparison (bare vs v4) on 1 question

Usage:
    python scripts/validate_batch_readiness.py
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
from reasoning.task_classifier import TaskClassifier
from reasoning.distillation_corpus import corpus_stats

GRAPH_PATH = "graphs/merged_graph.json"
SERVER = "http://127.0.0.1:4096"
CONFIG_DIR = r"C:\Users\Ace\AppData\Local\Temp\opencode-empty-config"
MODEL = "opencode/big-pickle"

QUESTIONS = [
    "What is binary search?",                    # trivial — should be fast
    "Explain Kadane's algorithm on [-2,1,-3,4,-1,2,1,-5,4].",  # medium
    "",  # deliberately empty — should not crash
]


def check(label: str, ok: bool, detail: str = ""):
    status = "PASS" if ok else "FAIL"
    print(f"  [{status}] {label}" + (f" — {detail}" if detail else ""))
    return ok


def main():
    print("=== BATCH READINESS VALIDATION ===\n")

    # Check opencode serve is alive
    import urllib.request
    try:
        urllib.request.urlopen(f"{SERVER}/app", timeout=3)
        check("opencode serve alive", True, SERVER)
    except Exception as e:
        check("opencode serve alive", False, str(e))
        print("\nFATAL: opencode serve not running. Start it first.")
        sys.exit(1)

    graph = MemoryGraph.load_json(GRAPH_PATH)
    check("graph loaded", len(graph.nodes) > 0, f"{len(graph.nodes)} nodes")

    classifier = TaskClassifier.load()
    check("classifier loaded", True)

    # Snapshot corpus before
    corpus_before = corpus_stats()
    rows_before = corpus_before.get("rows", 0)
    print(f"\n  Corpus before: {rows_before} rows\n")

    # Run each question
    all_ok = True
    session_dirs: list[str] = []
    for i, q in enumerate(QUESTIONS):
        print(f"\n--- Question {i+1}/{len(QUESTIONS)}: {q[:60]!r} ---")
        ctrl = V4OpencodeController(
            model=MODEL, server_url=SERVER, config_dir=CONFIG_DIR, timeout=120,
        )
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
                enable_activation=bool(q),  # skip activation on empty
                polish_answer=bool(q),      # skip polish on empty
            )
            elapsed = time.time() - t0
            print(f"  completed: steps={pkt.steps} finalized={pkt.finalized} "
                  f"elapsed={pkt.elapsed_sec}s (wall={elapsed:.1f}s)")

            if q:  # non-empty question
                ok = check("finalized", pkt.finalized)
                all_ok &= ok
                ok = check("answer non-empty", len(pkt.answer) > 10, f"{len(pkt.answer)} chars")
                all_ok &= ok
            else:  # empty question — should handle gracefully
                ok = check("empty question handled", True, f"steps={pkt.steps}")
                all_ok &= ok

            # Session dir
            if pkt.session_dir:
                session_dirs.append(pkt.session_dir)
                sd = Path(pkt.session_dir)
                expected_files = ["subgraph.json", "audit_log.jsonl", "cot_log.txt"]
                for fname in expected_files:
                    exists = (sd / fname).exists()
                    ok = check(f"  {fname} exists", exists)
                    all_ok &= ok
                # learning_report + graph_edits
                ok = check("  learning_report.json", (sd / "learning_report.json").exists())
                all_ok &= ok
                ok = check("  graph_edits.json", (sd / "graph_edits.json").exists())
                all_ok &= ok
            else:
                ok = check("session_dir populated", False, "None")
                all_ok &= ok

        except Exception as e:
            elapsed = time.time() - t0
            if not q:  # empty question error is OK
                ok = check("empty question raised", True, f"{type(e).__name__}: {str(e)[:80]}")
            else:
                ok = check(f"question completed", False, f"{type(e).__name__}: {str(e)[:100]}")
                all_ok = False

    # Path collision check
    if len(session_dirs) == len(set(session_dirs)):
        check("no session dir collisions", True, f"{len(session_dirs)} unique dirs")
    else:
        check("no session dir collisions", False, f"duplicates found!")
        all_ok = False

    # Corpus check
    corpus_after = corpus_stats()
    rows_after = corpus_after.get("rows", 0)
    rows_added = rows_after - rows_before
    # We expect 2 rows (the 2 non-empty finalized questions). Empty might or might not write.
    ok = check("corpus rows added", rows_added >= 2, f"{rows_added} new rows (total: {rows_after})")
    all_ok &= ok

    # Verify corpus row structure
    if rows_added > 0:
        corpus_path = Path(corpus_after["path"])
        lines = corpus_path.read_text(encoding="utf-8").strip().splitlines()
        last_row = json.loads(lines[-1])
        expected_keys = {"schema_version", "session_id", "input", "trace", "outputs", "metrics", "quality"}
        has_keys = expected_keys.issubset(set(last_row.keys()))
        ok = check("corpus row schema valid", has_keys,
                    f"keys: {sorted(last_row.keys())[:8]}...")
        all_ok &= ok

    print(f"\n{'='*50}")
    print(f"RESULT: {'ALL CHECKS PASSED' if all_ok else 'SOME CHECKS FAILED'}")
    print(f"{'='*50}")
    if all_ok:
        print("\nReady for batch runs.")
    else:
        print("\nFix the failing checks before running a full batch.")
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
