"""Run the frozen Phase 3E baseline/v2 comparison suite.

The suite is intentionally data-driven so the 3E-4 gate cannot drift while the
substrate is tuned. This runner records answers, call counts, and deterministic
required-term/forbidden-term smoke judgments. Manual review remains the quality
authority for borderline answers.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import statistics
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List

from local_model_command import _extract_content, _post_chat
from reasoning.budgets import Budgets
from reasoning.outcome_scorer import SubstrateOutcomeRow, collect_outcome_row, write_outcome_row
from reasoning.reasoning_loop import ReasoningRequest, run_reasoning


DEFAULT_TASKS = Path("bench/core_20.json")


def _load_suite(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _contains_any(answer: str, terms: Iterable[str]) -> bool:
    lower = answer.lower()
    return any(str(term).lower() in lower for term in terms)


def judge_answer(answer: str, task: Dict[str, Any]) -> Dict[str, Any]:
    missing_groups: List[List[str]] = []
    for group in task.get("required_terms", []):
        if not _contains_any(answer, group):
            missing_groups.append([str(x) for x in group])
    forbidden_hits = [
        str(term)
        for term in task.get("forbidden_terms", [])
        if str(term).lower() in answer.lower()
    ]
    return {
        "passed": not missing_groups and not forbidden_hits,
        "missing_required_groups": missing_groups,
        "forbidden_hits": forbidden_hits,
    }


def _llm_call(prompt: str) -> str:
    return _extract_content(_post_chat(prompt)).strip()


def _llm_judge_call(prompt: str) -> str:
    old_temp = os.environ.get("LOCAL_LLM_TEMPERATURE")
    old_max_tokens = os.environ.get("LOCAL_LLM_MAX_TOKENS")
    try:
        os.environ["LOCAL_LLM_TEMPERATURE"] = "0"
        os.environ["LOCAL_LLM_MAX_TOKENS"] = "600"
        return _extract_content(_post_chat(prompt)).strip()
    finally:
        if old_temp is None:
            os.environ.pop("LOCAL_LLM_TEMPERATURE", None)
        else:
            os.environ["LOCAL_LLM_TEMPERATURE"] = old_temp
        if old_max_tokens is None:
            os.environ.pop("LOCAL_LLM_MAX_TOKENS", None)
        else:
            os.environ["LOCAL_LLM_MAX_TOKENS"] = old_max_tokens


def _warm_up_local_model(*, judge_mode: str) -> Dict[str, Any]:
    started = time.perf_counter()
    answer = _llm_call("Reply with exactly: OK")
    judge = None
    if judge_mode in {"dual", "llm"}:
        judge = _llm_judge_call("Return JSON only: {\"passed\": true}")
    return {
        "elapsed_sec": round(time.perf_counter() - started, 3),
        "answer_preview": answer[:80],
        "judge_preview": (judge or "")[:80],
    }


def _extract_json_object(text: str) -> Dict[str, Any]:
    text = text.strip()
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
    except Exception:
        pass
    match = re.search(r"\{.*\}", text, flags=re.S)
    if not match:
        raise ValueError("no JSON object found in judge output")
    parsed = json.loads(match.group(0))
    if not isinstance(parsed, dict):
        raise ValueError("judge JSON was not an object")
    return parsed


def judge_answer_llm(answer: str, task: Dict[str, Any]) -> Dict[str, Any]:
    required = task.get("required_terms", [])
    forbidden = task.get("forbidden_terms", [])
    prompt_lines = [
        "You are a strict benchmark judge.",
        "Evaluate whether the answer satisfies each required concept group semantically, not by exact string match.",
        "A required group passes if the answer clearly expresses at least one concept in that group.",
        "A forbidden hit should only be listed if the answer semantically asserts that forbidden content.",
        "Return JSON only with this schema:",
        '{"passed": true|false, "required_groups": [{"terms": ["..."], "passed": true|false, "reason": "short"}], "forbidden_hits": ["..."], "summary": "short"}',
        "",
        f"Question:\n{task.get('question', '')}",
        "",
        "Required concept groups:",
    ]
    prompt_lines.extend(f"- {json.dumps(group, ensure_ascii=False)}" for group in required)
    prompt_lines.extend([
        "",
        "Forbidden terms:",
    ])
    if forbidden:
        prompt_lines.extend(f"- {term}" for term in forbidden)
    else:
        prompt_lines.append("- []")
    prompt_lines.extend([
        "",
        f"Answer:\n{answer}",
    ])
    prompt = "\n".join(prompt_lines)
    try:
        parsed = _extract_json_object(_llm_judge_call(prompt))
        group_results = parsed.get("required_groups", [])
        forbidden_hits = [str(x) for x in parsed.get("forbidden_hits", [])]
        passed = bool(parsed.get("passed", False))
        return {
            "passed": passed,
            "required_groups": group_results if isinstance(group_results, list) else [],
            "forbidden_hits": forbidden_hits,
            "summary": str(parsed.get("summary", "")),
            "parse_error": None,
        }
    except Exception as exc:
        return {
            "passed": False,
            "required_groups": [],
            "forbidden_hits": [],
            "summary": "",
            "parse_error": repr(exc),
        }


def judge_row(answer: str, task: Dict[str, Any], judge_mode: str) -> Dict[str, Any]:
    rubric = judge_answer(answer, task)
    llm = None
    if judge_mode in {"dual", "llm"}:
        llm = judge_answer_llm(answer, task)
    if judge_mode == "rubric":
        return {
            "passed": rubric["passed"],
            "agreed": True,
            "mode": judge_mode,
            "missing_required_groups": rubric["missing_required_groups"],
            "forbidden_hits": rubric["forbidden_hits"],
            "rubric": rubric,
            "llm": None,
        }
    if judge_mode == "llm":
        return {
            "passed": bool(llm and llm["passed"]),
            "agreed": True,
            "mode": judge_mode,
            "missing_required_groups": rubric["missing_required_groups"],
            "forbidden_hits": llm["forbidden_hits"] if llm else [],
            "rubric": rubric,
            "llm": llm,
        }
    agreed = bool(llm) and (rubric["passed"] == llm["passed"])
    return {
        "passed": bool(rubric["passed"] and llm and llm["passed"]),
        "agreed": bool(agreed),
        "mode": judge_mode,
        "missing_required_groups": rubric["missing_required_groups"],
        "forbidden_hits": rubric["forbidden_hits"],
        "rubric": rubric,
        "llm": llm,
    }


def _write_raw_trace(
    task_root: Path,
    result: Any,
    task: Dict[str, Any],
) -> None:
    audit = result.audit_summary or {}
    calls = audit.get("tokens_per_call", [])
    trace_lines = [
        f"# Raw Trace — {task.get('kind', '?')} task {task['id']}",
        "",
        f"**Answer**: {result.answer}",
        f"**LLM calls**: {result.budget_usage.get('llm_calls', {}).get('used')}",
        f"**Budget usage**: {result.budget_usage}",
        "",
        "## Instrumentation",
        f"- delta_status_breakdown: {audit.get('delta_status_breakdown', {})}",
        f"- checker_outcome_breakdown: {audit.get('checker_outcome_breakdown', {})}",
        f"- repair_triggered: {audit.get('repair_triggered')}",
        f"- repair_succeeded: {audit.get('repair_succeeded')}",
        f"- Activated signal ages: {audit.get('activated_signal_ages', {})}",
        f"- tokens_per_call: {calls}",
        "",
        "## Session subgraph",
        f"Path: `{result.session_subgraph_path}`",
        "",
    ]
    (task_root / "trace.md").write_text("\n".join(trace_lines), encoding="utf-8")


def run_one_task(
    *,
    task: Dict[str, Any],
    mode: str,
    output_root: Path,
    graph_id: str,
    graph_path: str,
    k_anchors: int,
    max_llm_calls: int,
    judge_mode: str,
    warm_start_session_paths: List[Path] | None = None,
    debug_signals: bool = False,
    consolidation_queue: List | None = None,
) -> Dict[str, Any]:
    enabled = mode == "v2"
    task_root = output_root / mode / str(task["id"])
    task_root.mkdir(parents=True, exist_ok=True)
    req = ReasoningRequest(
        question=str(task["question"]),
        graph_id=graph_id,
        graph_path=graph_path,
        k_anchors=k_anchors,
        max_iterations=3,
        session_persist_root=task_root,
        budgets=Budgets(max_llm_calls=max_llm_calls, max_total_tokens=16000),
        enable_substrate_v2=enabled,
        warm_start_session_paths=list(warm_start_session_paths or []),
        debug_signals=debug_signals,
    )
    started = time.perf_counter()
    try:
        result = run_reasoning(req, _llm_call)
        elapsed = time.perf_counter() - started
        judged = judge_row(result.answer, task, judge_mode)
        if enabled:
            _write_raw_trace(task_root, result, task)
        row: SubstrateOutcomeRow | None = None
        try:
            row = collect_outcome_row(result, task, elapsed_sec=elapsed, judge=judged)
            write_outcome_row(row)
        except Exception:
            pass  # outcome collection is non-fatal
        if consolidation_queue is not None and enabled:
            consolidation_queue.append((result, task, row))
        return {
            "task_id": task["id"],
            "kind": task.get("kind"),
            "mode": mode,
            "ok": True,
            "judge": judged,
            "elapsed_sec": round(elapsed, 3),
            "llm_calls": result.budget_usage.get("llm_calls", {}).get("used"),
            "answer": result.answer,
            "budget_usage": result.budget_usage,
            "audit_summary": result.audit_summary,
            "step_timing": (result.audit_summary or {}).get("step_timing", []),
            "session_subgraph_path": str(result.session_subgraph_path),
            "warm_start_session_paths": [str(p) for p in (warm_start_session_paths or [])],
        }
    except Exception as exc:
        return {
            "task_id": task["id"],
            "kind": task.get("kind"),
            "mode": mode,
            "ok": False,
            "elapsed_sec": round(time.perf_counter() - started, 3),
            "error": repr(exc),
        }


def _nearest_rank_percentile(sorted_values: List[float], percentile: float) -> float | None:
    if not sorted_values:
        return None
    if percentile <= 0:
        return sorted_values[0]
    if percentile >= 1:
        return sorted_values[-1]
    rank = max(1, math.ceil(percentile * len(sorted_values)))
    index = min(len(sorted_values) - 1, rank - 1)
    return sorted_values[index]


def summarize(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    by_mode: Dict[str, List[Dict[str, Any]]] = {}
    for row in rows:
        by_mode.setdefault(str(row["mode"]), []).append(row)
    summary: Dict[str, Any] = {}
    for mode, mode_rows in by_mode.items():
        ok_rows = [r for r in mode_rows if r.get("ok")]
        judged_rows = [r for r in ok_rows if r.get("judge", {}).get("passed")]
        rubric_rows = [r for r in ok_rows if r.get("judge", {}).get("rubric", {}).get("passed")]
        llm_rows = [
            r for r in ok_rows
            if r.get("judge", {}).get("llm") is not None
            and r["judge"]["llm"].get("passed")
        ]
        disagreements = [
            r for r in ok_rows
            if r.get("judge", {}).get("llm") is not None
            and not r.get("judge", {}).get("agreed", False)
        ]
        calls = [int(r.get("llm_calls") or 0) for r in ok_rows]
        elapsed = sorted(float(r.get("elapsed_sec") or 0.0) for r in ok_rows)
        median_elapsed = statistics.median(elapsed) if elapsed else None
        p95_elapsed = _nearest_rank_percentile(elapsed, 0.95)
        summary[mode] = {
            "tasks": len(mode_rows),
            "ok": len(ok_rows),
            "judge_passed": len(judged_rows),
            "judge_passed_rubric": len(rubric_rows),
            "judge_passed_llm": len(llm_rows),
            "judge_disagreements": len(disagreements),
            "total_llm_calls": sum(calls),
            "mean_llm_calls": (sum(calls) / len(calls)) if calls else None,
            "median_elapsed_sec": median_elapsed,
            "p95_elapsed_sec": p95_elapsed,
        }
    if "baseline" in summary and "v2" in summary:
        base_calls = summary["baseline"]["total_llm_calls"]
        v2_calls = summary["v2"]["total_llm_calls"]
        summary["comparison"] = {
            "call_reduction": (base_calls / v2_calls) if v2_calls else None,
            "quality_equal_by_smoke": (
                summary["baseline"]["judge_passed"] == summary["v2"]["judge_passed"]
            ),
        }
    cold_rows = [r for r in rows if int(r.get("run_index") or 0) == 1 and r.get("ok")]
    warm_rows = [r for r in rows if int(r.get("run_index") or 0) >= 2 and r.get("ok")]
    if cold_rows and warm_rows:
        cold_calls = [int(r.get("llm_calls") or 0) for r in cold_rows]
        warm_calls = [int(r.get("llm_calls") or 0) for r in warm_rows]
        warm_activated = [
            r for r in warm_rows
            if int((r.get("audit_summary") or {}).get("activated_prior_session_signal_count") or 0) >= 1
        ]
        warm_reused = [
            r for r in warm_rows
            if bool((r.get("audit_summary") or {}).get("prior_session_signal_reused"))
        ]
        summary["cold_warm"] = {
            "cold_tasks": len(cold_rows),
            "warm_tasks": len(warm_rows),
            "cold_mean_llm_calls": (sum(cold_calls) / len(cold_calls)) if cold_calls else None,
            "warm_mean_llm_calls": (sum(warm_calls) / len(warm_calls)) if warm_calls else None,
            "warm_over_cold_call_ratio": ((sum(warm_calls) / len(warm_calls)) / (sum(cold_calls) / len(cold_calls))) if cold_calls and warm_calls else None,
            "cold_judge_passed": sum(1 for r in cold_rows if r.get("judge", {}).get("passed")),
            "warm_judge_passed": sum(1 for r in warm_rows if r.get("judge", {}).get("passed")),
            "warm_runs_with_prior_signal_activation": len(warm_activated),
            "warm_runs_with_prior_signal_reuse": len(warm_reused),
        }
    return summary


def run_cold_warm_suite(
    *,
    suite: Dict[str, Any],
    output_root: Path,
    graph_id: str,
    graph_path: str,
    k_anchors: int,
    max_llm_calls: int,
    judge_mode: str,
    debug_signals: bool = False,
    consolidate_graph: bool = True,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    runs_per_task = int(suite.get("runs_per_task", 3))
    consolidation_queue: List = []
    for task in suite["tasks"]:
        prior_paths: List[Path] = []
        for run_index in range(1, runs_per_task + 1):
            row = run_one_task(
                task=task,
                mode="v2",
                output_root=output_root / f"run_{run_index}",
                graph_id=graph_id,
                graph_path=graph_path,
                k_anchors=k_anchors,
                max_llm_calls=max_llm_calls,
                judge_mode=judge_mode,
                warm_start_session_paths=prior_paths if run_index >= 2 else [],
                debug_signals=debug_signals,
                consolidation_queue=consolidation_queue,
            )
            audit = row.get("audit_summary") or {}
            row["run_index"] = run_index
            row["run_kind"] = "cold" if run_index == 1 else "warm"
            row["workspace_warm_filled"] = audit.get("workspace_warm_filled", 0)
            rows.append(row)
            print(json.dumps({
                "task_id": row["task_id"],
                "run_index": run_index,
                "run_kind": row["run_kind"],
                "ok": row["ok"],
                "judge_passed": row.get("judge", {}).get("passed"),
                "llm_calls": row.get("llm_calls"),
                "warm_start_count": len(prior_paths) if run_index >= 2 else 0,
                "workspace_warm_filled": row["workspace_warm_filled"],
            }, ensure_ascii=False))
            if row.get("ok") and row.get("session_subgraph_path"):
                prior_paths.append(Path(str(row["session_subgraph_path"])))
    # Phase 3G: batch-consolidate session memories into the graph
    if consolidate_graph and graph_path and consolidation_queue:
        try:
            from reasoning.session_to_graph import batch_consolidate
            n = batch_consolidate(
                [(r, t) for r, t, _ in consolidation_queue],
                graph_path,
                outcome_rows=[row for _, _, row in consolidation_queue],
            )
            if n:
                print(f"Consolidated {n} session memories into {graph_path}")
        except Exception as exc:
            print(f"Graph consolidation skipped: {exc}")

    return rows


def replicate_cold_warm(
    *,
    suite: Dict[str, Any],
    output_root: Path,
    graph_id: str,
    graph_path: str,
    k_anchors: int,
    max_llm_calls: int,
    judge_mode: str,
    replicates: int,
    debug_signals: bool = False,
) -> Dict[str, Any]:
    ratios: List[float] = []
    all_rows: List[Dict[str, Any]] = []
    failed_replicates: List[int] = []
    for rep in range(1, replicates + 1):
        rep_root = output_root / f"rep_{rep}"
        rep_root.mkdir(parents=True, exist_ok=True)
        print(f"\n--- Replicate {rep}/{replicates} ---")
        rows = run_cold_warm_suite(
            suite=suite, output_root=rep_root,
            graph_id=graph_id, graph_path=graph_path,
            k_anchors=k_anchors, max_llm_calls=max_llm_calls,
            judge_mode=judge_mode, debug_signals=debug_signals,
        )
        for row in rows:
            row["replicate"] = rep
        summary = summarize(rows)
        ratio = summary.get("cold_warm", {}).get("warm_over_cold_call_ratio")
        if ratio is not None:
            ratios.append(ratio)
        else:
            failed_replicates.append(rep)
        all_rows.extend(rows)
    ratios_sorted = sorted(ratios)
    n_completed = len(ratios_sorted)
    result = {
        "replicates": n_completed,
        "replicates_requested": replicates,
        "replicates_failed": len(failed_replicates),
        "failed_replicate_indices": failed_replicates,
        "ratios": ratios_sorted,
        "mean_ratio": statistics.mean(ratios_sorted) if n_completed else None,
        "median_ratio": statistics.median(ratios_sorted) if n_completed else None,
        "min_ratio": ratios_sorted[0] if n_completed else None,
        "max_ratio": ratios_sorted[-1] if n_completed else None,
        "pass_count_0_70": sum(1 for r in ratios_sorted if r <= 0.70),
        "pass_rate_0_70": (sum(1 for r in ratios_sorted if r <= 0.70) / n_completed) if n_completed else None,
        "raw_rows": all_rows,
    }
    return result


def print_replicate_summary(rep_result: Dict[str, Any]) -> None:
    print("\n=== REPLICATED COMPOUNDING SUMMARY ===")
    print(f"  Replicates:          {rep_result['replicates']}")
    print(f"  Mean ratio:          {rep_result['mean_ratio']:.4f}" if rep_result['mean_ratio'] is not None else "  Mean ratio:          N/A")
    print(f"  Median ratio:        {rep_result['median_ratio']:.4f}" if rep_result['median_ratio'] is not None else "  Median ratio:        N/A")
    print(f"  Min ratio:           {rep_result['min_ratio']:.4f}" if rep_result['min_ratio'] is not None else "  Min ratio:           N/A")
    print(f"  Max ratio:           {rep_result['max_ratio']:.4f}" if rep_result['max_ratio'] is not None else "  Max ratio:           N/A")
    print(f"  Pass count (≤0.70):  {rep_result['pass_count_0_70']}")
    print(f"  Pass rate (≤0.70):   {rep_result['pass_rate_0_70']:.1%}" if rep_result['pass_rate_0_70'] is not None else "  Pass rate (≤0.70):   N/A")
    print("======================================")


def rescore_results(
    results_path: Path,
    tasks_path: Path,
    output_path: Path | None = None,
    judge_mode: str = "rubric",
) -> Dict[str, Any]:
    suite = _load_suite(tasks_path)
    tasks_by_id = {str(task["id"]): task for task in suite["tasks"]}
    payload = json.loads(results_path.read_text(encoding="utf-8"))
    rows = payload.get("rows", [])
    for row in rows:
        task = tasks_by_id.get(str(row.get("task_id")))
        if not task or not row.get("ok"):
            continue
        row["judge"] = judge_row(str(row.get("answer", "")), task, judge_mode)
    payload["judge_contract"] = suite.get("judge_contract")
    payload["rescore_source"] = str(results_path)
    payload["judge_mode"] = judge_mode
    payload["summary"] = summarize(rows)
    out = output_path or results_path.with_name(results_path.stem + "_rescored.json")
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"output": str(out), "summary": payload["summary"]}


def main() -> int:
    import sys as _sys
    try:
        _sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass
    parser = argparse.ArgumentParser(description="Run frozen Phase 3E benchmark tasks.")
    parser.add_argument("--tasks", type=Path, default=DEFAULT_TASKS)
    parser.add_argument("--rescore-results", type=Path, default=None)
    parser.add_argument("--rescore-out", type=Path, default=None)
    parser.add_argument("--judge-mode", choices=["rubric", "dual", "llm"], default="rubric")
    parser.add_argument("--mode", choices=["baseline", "v2", "both"], default="both")
    parser.add_argument("--cold-warm", action="store_true", help="Run the cold/warm compounding suite instead of the core benchmark")
    parser.add_argument("--replicate", type=int, default=0, help="Number of cold/warm replicates (only with --cold-warm; reports aggregated stats)")
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--graph-id", default="cs4")
    parser.add_argument("--graph-path", default="graphs/cs4.json")
    parser.add_argument("--k-anchors", type=int, default=8)
    parser.add_argument("--max-llm-calls", type=int, default=8)
    parser.add_argument("--base-url", default="http://127.0.0.1:6768")
    parser.add_argument("--max-tokens", type=int, default=2400)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--enable-thinking", action="store_true")
    parser.add_argument("--skip-warmup", action="store_true", help="Do not pre-warm the local model before timing benchmark tasks")
    parser.add_argument("--debug-signals", action="store_true", help="Print detailed signal activation info per step during warm runs")
    args = parser.parse_args()

    if args.cold_warm and args.tasks == DEFAULT_TASKS:
        args.tasks = Path("bench/cold_warm_adversarial.json")

    os.environ["LOCAL_LLM_BASE_URL"] = args.base_url
    os.environ["LOCAL_LLM_MAX_TOKENS"] = str(args.max_tokens)
    os.environ["LOCAL_LLM_TEMPERATURE"] = str(args.temperature)
    os.environ["LOCAL_LLM_ENABLE_THINKING"] = "1" if args.enable_thinking else "0"

    if args.rescore_results is not None:
        result = rescore_results(args.rescore_results, args.tasks, args.rescore_out, args.judge_mode)
        print(json.dumps(result["summary"], ensure_ascii=False, indent=2))
        print(f"WROTE {result['output']}")
        return 0

    suite = _load_suite(args.tasks)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    output_root = args.out or Path("artifacts") / f"phase3e_benchmark_{stamp}"
    output_root.mkdir(parents=True, exist_ok=True)
    warmup = None
    if not args.skip_warmup:
        warmup = _warm_up_local_model(judge_mode=args.judge_mode)
        print(json.dumps({"warmup": warmup}, ensure_ascii=False))

    if args.cold_warm:
        if args.replicate > 1:
            rep_result = replicate_cold_warm(
                suite=suite,
                output_root=output_root,
                graph_id=args.graph_id,
                graph_path=args.graph_path,
                k_anchors=args.k_anchors,
                max_llm_calls=args.max_llm_calls,
                judge_mode=args.judge_mode,
                replicates=args.replicate,
                debug_signals=args.debug_signals,
            )
            rep_payload = {
                "suite_version": suite.get("version"),
                "judge_contract": suite.get("judge_contract"),
                "warmup": warmup,
                "replicates": rep_result["replicates"],
                "replicates_requested": rep_result["replicates_requested"],
                "replicates_failed": rep_result["replicates_failed"],
                "failed_replicate_indices": rep_result["failed_replicate_indices"],
                "ratios": rep_result["ratios"],
                "mean_ratio": rep_result["mean_ratio"],
                "median_ratio": rep_result["median_ratio"],
                "min_ratio": rep_result["min_ratio"],
                "max_ratio": rep_result["max_ratio"],
                "pass_count_0_70": rep_result["pass_count_0_70"],
                "pass_rate_0_70": rep_result["pass_rate_0_70"],
                "rows": rep_result["raw_rows"],
            }
            (output_root / "replicate_summary.json").write_text(
                json.dumps(rep_payload, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            print(json.dumps(rep_payload, ensure_ascii=False, indent=2))
            print(f"WROTE {output_root / 'replicate_summary.json'}")
            print_replicate_summary(rep_result)
            return 0

        rows = run_cold_warm_suite(
            suite=suite,
            output_root=output_root,
            graph_id=args.graph_id,
            graph_path=args.graph_path,
            k_anchors=args.k_anchors,
            max_llm_calls=args.max_llm_calls,
            judge_mode=args.judge_mode,
            debug_signals=args.debug_signals,
        )
        payload = {
            "suite_version": suite.get("version"),
            "judge_contract": suite.get("judge_contract"),
            "warmup": warmup,
            "rows": rows,
            "summary": summarize(rows),
        }
        (output_root / "results.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(json.dumps(payload["summary"], ensure_ascii=False, indent=2))
        print(f"WROTE {output_root / 'results.json'}")
        return 0

    modes = ["baseline", "v2"] if args.mode == "both" else [args.mode]
    rows: List[Dict[str, Any]] = []
    for mode in modes:
        for task in suite["tasks"]:
            row = run_one_task(
                task=task,
                mode=mode,
                output_root=output_root,
                graph_id=args.graph_id,
                graph_path=args.graph_path,
                k_anchors=args.k_anchors,
                max_llm_calls=args.max_llm_calls,
                judge_mode=args.judge_mode,
            )
            rows.append(row)
            print(json.dumps({
                "task_id": row["task_id"],
                "mode": row["mode"],
                "ok": row["ok"],
                "judge_passed": row.get("judge", {}).get("passed"),
                "llm_calls": row.get("llm_calls"),
            }, ensure_ascii=False))

    payload = {
        "suite_version": suite.get("version"),
        "judge_contract": suite.get("judge_contract"),
        "warmup": warmup,
        "rows": rows,
        "summary": summarize(rows),
    }
    (output_root / "results.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(payload["summary"], ensure_ascii=False, indent=2))
    print(f"WROTE {output_root / 'results.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
