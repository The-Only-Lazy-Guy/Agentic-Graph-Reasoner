"""Daily corpus accumulation runner — Phase 3G G4 + session-to-graph.

Loads all benchmark suites, deduplicates tasks by id, runs them with
cold/warm compounding, and batch-consolidates session memories into the
graph at the end.

Usage:
    python run_daily_corpus.py                              # single run
    python run_daily_corpus.py --daemon --interval 24       # loop every 24h
    python run_daily_corpus.py --dry-run                    # simulate, no LLM
    python run_daily_corpus.py --recall "segment tree"      # query memories
    python run_daily_corpus.py --recall-count               # count memories
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

os.environ.setdefault("LOCAL_LLM_BASE_URL", "http://127.0.0.1:6768")
os.environ.setdefault("LOCAL_LLM_MAX_TOKENS", "2400")
os.environ.setdefault("LOCAL_LLM_TEMPERATURE", "0.2")


# ── Suites and defaults ──────────────────────────────────────────────────


DEFAULT_SUITES: List[str] = [
    "bench/core_20.json",
    "bench/deep_reasoning_5.json",
    "bench/cold_warm_5.json",
]

GRAPH_ID = "merged_graph"
GRAPH_PATH = "graphs/merged_graph.json"
OUTPUT_ROOT = Path("data/daily_corpus")
K_ANCHORS = 3
MAX_LLM_CALLS = 6


# ── Suite loading ────────────────────────────────────────────────────────


def _load_suite(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def merge_suites(suite_paths: List[str]) -> Dict[str, Any]:
    """Load multiple suite files and merge their tasks (deduplicating by id)."""
    seen: set = set()
    merged_tasks: List[Dict[str, Any]] = []
    for sp in suite_paths:
        suite = _load_suite(Path(sp))
        for task in suite.get("tasks", []):
            tid = str(task.get("id", ""))
            if tid and tid not in seen:
                seen.add(tid)
                merged_tasks.append(task)
    merged: Dict[str, Any] = {"tasks": merged_tasks}
    max_runs = 1
    for sp in suite_paths:
        s = _load_suite(Path(sp))
        max_runs = max(max_runs, int(s.get("runs_per_task", 1)))
    merged["runs_per_task"] = max_runs
    return merged


# ── Recall queries ───────────────────────────────────────────────────────


def recall(query: str, graph_path: str, max_results: int = 10) -> int:
    """Query session_memory nodes in the graph matching *query*."""
    from graph_core import MemoryGraph

    path = Path(graph_path)
    if not path.exists():
        print(f"Graph not found: {path}")
        return 0

    graph = MemoryGraph.load_json(path)
    low = query.lower()
    matches: List[tuple] = []
    for nid, node in graph.nodes.items():
        if node.node_type != "session_memory":
            continue
        score = low in node.text.lower()
        if score:
            matches.append((nid, node))

    matches.sort(key=lambda x: x[1].timestamp if hasattr(x[1], "timestamp") and x[1].metadata.get("timestamp") else "")
    print(f"Found {len(matches)} session memories matching '{query}':")
    for nid, node in matches[:max_results]:
        ts = node.metadata.get("timestamp", "?")
        score = node.metadata.get("outcome_score", "?")
        correct = node.metadata.get("outcome_correct", "?")
        print(f"  [{ts}] score={score} correct={correct}  {node.text[:120]}")
    if len(matches) > max_results:
        print(f"  ... and {len(matches) - max_results} more")
    return len(matches)


def recall_count(graph_path: str) -> int:
    """Count total session_memory nodes in the graph."""
    from graph_core import MemoryGraph

    path = Path(graph_path)
    if not path.exists():
        print(f"Graph not found: {path}")
        return 0

    graph = MemoryGraph.load_json(path)
    count = sum(1 for n in graph.nodes.values() if n.node_type == "session_memory")
    print(f"Graph has {count} session_memory nodes ({len(graph.nodes)} total nodes, {len(graph.edges)} edges)")
    return count


# ── LLM warmup with retry ────────────────────────────────────────────────


def _llm_warmup_with_retry(*, judge_mode: str, max_retries: int = 3, retry_delay: float = 30.0) -> Dict[str, Any]:
    """Call _warm_up_local_model with retries on connection failures."""
    from run_phase3e_benchmark import _warm_up_local_model as warmup

    last_exc: Optional[Exception] = None
    for attempt in range(1, max_retries + 1):
        try:
            return warmup(judge_mode=judge_mode)
        except Exception as exc:
            last_exc = exc
            msg = str(exc).lower()
            is_connection = any(kw in msg for kw in ("connection", "refused", "timeout", "unreachable"))
            if not is_connection or attempt == max_retries:
                raise
            print(f"LLM connection failed (attempt {attempt}/{max_retries}): {exc}")
            print(f"Retrying in {retry_delay}s ...")
            time.sleep(retry_delay)
    raise RuntimeError(f"LLM unreachable after {max_retries} attempts") from last_exc


# ── Dry-run mock ─────────────────────────────────────────────────────────


def _dry_run_mock_result(task: Dict[str, Any]) -> Dict[str, Any]:
    """Generate a mock result dict for dry-run mode (no LLM call)."""
    from reasoning.reasoning_loop import ReasoningResult, SessionSubgraph

    result = ReasoningResult(
        answer=f"Mock answer for {task.get('id', '?')}",
        reasoning_trace="",
        raw_outputs=[],
        session_subgraph=SessionSubgraph(session_id="dry", query="?", graph_id="dry"),
        session_subgraph_path=Path("."),
        audit_summary={
            "debug_signal_dump": [
                {"id": "sigv2_dry", "source_node_id": None, "text_preview": "dry run signal"},
            ],
            "step_count": 0,
            "tokens_per_call": [],
            "checker_outcome_breakdown": {},
            "repair_triggered": 0,
            "repair_succeeded": 0,
            "step_timing": [],
        },
        consolidation_decisions=[],
        budget_usage={"llm_calls": {"used": 0}},
        dispatch_outcomes=[],
        anchor_ids=[],
        iterations_completed=1,
    )
    return result


def run_dry(
    suite_paths: List[str],
    *,
    graph_path: str,
) -> Dict[str, Any]:
    """Simulate a daily run without hitting the LLM."""
    from reasoning.outcome_scorer import outcome_count
    from reasoning.session_to_graph import batch_consolidate

    suite = merge_suites(suite_paths)
    tasks = suite.get("tasks", [])
    runs_per_task = int(suite.get("runs_per_task", 1))

    print(f"[dry-run] Suites: {suite_paths}")
    print(f"[dry-run] {len(tasks)} deduplicated tasks × {runs_per_task} runs = {len(tasks) * runs_per_task} outcomes")
    print(f"[dry-run] Current outcome count: {outcome_count()}")

    # Generate mock results
    mock_results: list = []
    for task in tasks:
        for run_index in range(1, runs_per_task + 1):
            result = _dry_run_mock_result(task)
            mock_results.append((result, task))

    print(f"[dry-run] Generated {len(mock_results)} mock session results")

    # Test consolidation path
    added = batch_consolidate(mock_results, graph_path, dry_run=True)
    print(f"[dry-run] Consolidation would add {added} session_memory nodes")
    print("[dry-run] Dry run PASSED")
    return {"tasks": len(tasks), "runs_per_task": runs_per_task, "mock_results": len(mock_results), "dry_consolidation_nodes": added}


# ── Live run ─────────────────────────────────────────────────────────────


def run_daily(
    suite_paths: List[str],
    *,
    output_root: Path,
    graph_id: str,
    graph_path: str,
    k_anchors: int,
    max_llm_calls: int,
    dry_run: bool = False,
) -> Dict[str, Any]:
    """Execute one daily corpus run."""
    if dry_run:
        return run_dry(suite_paths, graph_path=graph_path)

    from run_phase3e_benchmark import run_cold_warm_suite
    from reasoning.outcome_scorer import outcome_count

    suite = merge_suites(suite_paths)
    tasks = suite.get("tasks", [])
    runs_per_task = int(suite.get("runs_per_task", 1))

    print(f"Daily corpus — {len(tasks)} tasks × {runs_per_task} runs = {len(tasks) * runs_per_task} outcomes expected")
    print(f"Pre-run outcome count: {outcome_count()}")

    stamp = time.strftime("%Y%m%d_%H%M%S")
    run_root = output_root / stamp
    run_root.mkdir(parents=True, exist_ok=True)

    warmup = _llm_warmup_with_retry(judge_mode="rubric")
    print(json.dumps({"warmup": warmup}, ensure_ascii=False))

    started = time.perf_counter()
    pre_count = outcome_count()
    rows = run_cold_warm_suite(
        suite=suite,
        output_root=run_root,
        graph_id=graph_id,
        graph_path=graph_path,
        k_anchors=k_anchors,
        max_llm_calls=max_llm_calls,
        judge_mode="rubric",
        debug_signals=True,
        consolidate_graph=True,
    )
    elapsed = time.perf_counter() - started

    post_count = outcome_count()
    result = {
        "stamp": stamp,
        "suites": suite_paths,
        "tasks": len(tasks),
        "runs_per_task": runs_per_task,
        "rows_ok": sum(1 for r in rows if r.get("ok")),
        "rows_total": len(rows),
        "judge_passed": sum(1 for r in rows if r.get("judge", {}).get("passed")),
        "elapsed_sec": round(elapsed, 1),
        "pre_outcome_count": pre_count,
        "post_outcome_count": post_count,
        "new_outcomes": post_count - pre_count,
    }
    (run_root / "summary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"WROTE {run_root / 'summary.json'}")
    print(f"Total outcomes to date: {outcome_count()}")
    return result


# ── Daemon loop ──────────────────────────────────────────────────────────


def daemon_loop(
    suite_paths: List[str],
    *,
    interval_hours: float = 24,
    output_root: Path = OUTPUT_ROOT,
    graph_id: str = GRAPH_ID,
    graph_path: str = GRAPH_PATH,
    k_anchors: int = K_ANCHORS,
    max_llm_calls: int = MAX_LLM_CALLS,
) -> None:
    """Run daily corpus on a loop with configurable interval."""
    interval_sec = interval_hours * 3600
    iteration = 0
    while True:
        iteration += 1
        print(f"\n{'='*60}")
        print(f"Daily corpus iteration {iteration} — {time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*60}")
        try:
            run_daily(
                suite_paths,
                output_root=output_root,
                graph_id=graph_id,
                graph_path=graph_path,
                k_anchors=k_anchors,
                max_llm_calls=max_llm_calls,
            )
        except Exception as exc:
            print(f"Iteration {iteration} failed: {exc}")
            import traceback
            traceback.print_exc()
        next_time = time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime(time.time() + interval_sec))
        print(f"Next run in {interval_hours}h ({next_time})")
        time.sleep(interval_sec)


# ── CLI ──────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description="Daily corpus accumulation")
    parser.add_argument("--once", action="store_true", help="Run once and exit")
    parser.add_argument("--daemon", action="store_true", help="Run on a loop")
    parser.add_argument("--interval", type=float, default=24, help="Hours between daemon runs (default: 24)")
    parser.add_argument("--suites", nargs="*", default=DEFAULT_SUITES, help="Suite files to include")
    parser.add_argument("--out", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--graph-id", default=GRAPH_ID)
    parser.add_argument("--graph-path", default=GRAPH_PATH)
    parser.add_argument("--k-anchors", type=int, default=K_ANCHORS)
    parser.add_argument("--max-llm-calls", type=int, default=MAX_LLM_CALLS)
    parser.add_argument("--dry-run", action="store_true", help="Simulate without LLM calls")
    parser.add_argument("--recall", type=str, default=None, metavar="QUERY", help="Query session memories in the graph")
    parser.add_argument("--recall-count", action="store_true", help="Count session memory nodes in the graph")
    args = parser.parse_args()

    if args.recall is not None:
        recall(args.recall, args.graph_path)
        return

    if args.recall_count:
        recall_count(args.graph_path)
        return

    if args.daemon:
        daemon_loop(
            args.suites,
            interval_hours=args.interval,
            output_root=args.out,
            graph_id=args.graph_id,
            graph_path=args.graph_path,
            k_anchors=args.k_anchors,
            max_llm_calls=args.max_llm_calls,
        )
    else:
        run_daily(
            args.suites,
            output_root=args.out,
            graph_id=args.graph_id,
            graph_path=args.graph_path,
            k_anchors=args.k_anchors,
            max_llm_calls=args.max_llm_calls,
            dry_run=args.dry_run,
        )


if __name__ == "__main__":
    main()
