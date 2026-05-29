from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import asdict, is_dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping

from answerer_v4 import V4OpencodeController, answer_query_v4
from graph_core import MemoryGraph


DEFAULT_GRAPH = Path("graphs/merged_graph.json")
DEFAULT_OUT_ROOT = Path("artifacts/v4_difficulty_sweep")
DEFAULT_CONFIG_DIR = "pure-opencode"

DEFAULT_CASES: List[Dict[str, Any]] = [
    {
        "id": "trivial_physics_medium_requirement",
        "difficulty": "trivial",
        "question": "Why can light travel through space but sound cannot?",
        "max_steps": 6,
        "expected": "Known direct explanation should reuse graph facts or finalize quickly.",
    },
    {
        "id": "medium_physics_multi_fact",
        "difficulty": "medium",
        "question": "Why does a prism bend light but not change the light's frequency?",
        "max_steps": 8,
        "expected": "Needs two related graph facts: refraction/speed change and source-fixed frequency.",
    },
    {
        "id": "hard_system_design",
        "difficulty": "hard",
        "question": (
            "Design a real-time leaderboard service for a competitive programming platform. "
            "Requirements: 500K+ concurrent users, score updates must propagate within 100ms, "
            "rank queries must run in O(log n), range pagination must support ranks 1000-1020, "
            "ties must be handled correctly, and the design should name one subtle correctness issue with a fix."
        ),
        "max_steps": 12,
        "expected": "Should fall back to the full graph-tool loop and use memory as scaffolding.",
    },
]


def _now_slug() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return _jsonable(asdict(value))
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    return value


def _packet_summary(packet: Any) -> Dict[str, Any]:
    return {
        "question": packet.question,
        "answer": packet.answer,
        "answer_raw": packet.answer_raw,
        "execution_mode": packet.execution_mode,
        "task_type": packet.task_type,
        "controller_task_family": packet.controller_task_family,
        "steps": packet.steps,
        "max_steps": packet.max_steps,
        "tool_call_count": packet.tool_call_count,
        "elapsed_sec": packet.elapsed_sec,
        "finalized": packet.finalized,
        "anchors": list(packet.anchors),
        "shortcut_reason": packet.shortcut_reason,
        "shortcut_anchor_ids": list(packet.shortcut_anchor_ids),
        "controller_fallback_used": packet.controller_fallback_used,
        "subgoal_reuse_count": packet.subgoal_reuse_count,
        "slot_fill_stats": _jsonable(packet.slot_fill_stats),
        "controller_action_counts": dict(packet.controller_action_counts),
        "controller_call_count": packet.controller_call_count,
        "controller_total_elapsed_sec": packet.controller_total_elapsed_sec,
        "controller_nonempty_turns": packet.controller_nonempty_turns,
        "controller_raw_trace": _jsonable(packet.controller_raw_trace),
        "session_dir": packet.session_dir,
        "plan": _jsonable(packet.plan),
        "tool_log": _jsonable(packet.tool_log),
        "cot_log": _jsonable(packet.cot_log),
        "task_frame": _jsonable(packet.task_frame),
        "task_frame_rendered": packet.task_frame_rendered,
        "coverage": _jsonable(packet.coverage),
        "coverage_addressed_pct": packet.coverage_addressed_pct,
        "hypotheses": _jsonable(packet.hypotheses),
        "procedure_invocations": _jsonable(packet.procedure_invocations),
        "learning_report": _jsonable(packet.learning_report),
        "graph_edits": _jsonable(packet.graph_edits),
        "graph_edits_applied": packet.graph_edits_applied,
        "scoped_patches": _jsonable(getattr(packet, "scoped_patches", [])),
        "scoped_patch_summary": _jsonable(getattr(packet, "scoped_patch_summary", {})),
        "signature_candidates": _jsonable(getattr(packet, "signature_candidates", [])),
        "signature_events": _jsonable(getattr(packet, "signature_events", [])),
        "signature_stats_update": _jsonable(getattr(packet, "signature_stats_update", {})),
        "signature_shadow_report": _jsonable(getattr(packet, "signature_shadow_report", {})),
        "signature_graph_projection": _jsonable(getattr(packet, "signature_graph_projection", {})),
        "signature_live_bias": _jsonable(getattr(packet, "signature_live_bias", {})),
        "micro_steps": _jsonable(packet.micro_steps),
        "explanation": packet.explanation,
        "polish_applied": packet.polish_applied,
    }


def _write_text(path: Path, lines: Iterable[str]) -> None:
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_cot_log(path: Path, cot_log: List[str]) -> None:
    lines: List[str] = []
    for i, chunk in enumerate(cot_log, start=1):
        lines.append(f"==================== TURN {i} ====================")
        lines.append(chunk)
        lines.append("")
    _write_text(path, lines)


def _raw_trace_to_text(raw_trace: List[Mapping[str, Any]]) -> str:
    lines: List[str] = []
    for entry in raw_trace:
        lines.append(f"==================== CALL {entry.get('call_index', '?')} ====================")
        lines.append(f"mode: {entry.get('mode', '')}")
        lines.append(f"started_at: {entry.get('started_at', '')}")
        lines.append(f"ended_at: {entry.get('ended_at', '')}")
        lines.append(f"elapsed_sec: {entry.get('elapsed_sec', '')}")
        lines.append(f"session_id_before: {entry.get('session_id_before', '')}")
        lines.append(f"session_id_after: {entry.get('session_id_after', '')}")
        lines.append(f"model: {entry.get('model', '')}")
        lines.append(f"server_url: {entry.get('server_url', '')}")
        lines.append("")
        lines.append("--- stdin_message ---")
        lines.append(str(entry.get("stdin_message", "") or ""))
        lines.append("")
        lines.append("--- raw_stdout ---")
        lines.append(str(entry.get("raw_stdout", "") or ""))
        lines.append("")
        lines.append("--- raw_stderr ---")
        lines.append(str(entry.get("raw_stderr", "") or ""))
        lines.append("")
        lines.append("--- assistant_text ---")
        lines.append(str(entry.get("assistant_text", "") or ""))
        lines.append("")
    return "\n".join(lines)


def _write_raw_trace_text(path: Path, raw_trace: List[Mapping[str, Any]]) -> None:
    path.write_text(_raw_trace_to_text(raw_trace), encoding="utf-8")


def _copy_session_dir(session_dir: str | None, dst_root: Path) -> None:
    if not session_dir:
        return
    src = Path(session_dir)
    if not src.exists():
        return
    dst_root.mkdir(parents=True, exist_ok=True)
    dst = dst_root / src.name
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)


def _case_dir_name(index: int, case: Mapping[str, Any]) -> str:
    safe_id = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in str(case["id"]))
    return f"{index:02d}_{case['difficulty']}_{safe_id}"


def _run_case(
    *,
    index: int,
    case: Mapping[str, Any],
    graph_path: Path,
    graph_id: str,
    model: str,
    server_url: str,
    config_dir: str,
    timeout: float,
    out_dir: Path,
    apply_graph_edits: bool,
    enable_signature_live_bias: bool,
    signature_stats_dir: str | Path,
) -> Dict[str, Any]:
    case_dir = out_dir / _case_dir_name(index, case)
    case_dir.mkdir(parents=True, exist_ok=True)

    graph = MemoryGraph.load_json(graph_path)
    controller = V4OpencodeController(
        model=model,
        server_url=server_url,
        config_dir=config_dir,
        timeout=timeout,
        print_raw_output=False,
    )
    packet = answer_query_v4(
        question=str(case["question"]),
        graph=graph,
        controller=controller,
        graph_id=graph_id,
        max_steps=int(case["max_steps"]),
        k_anchors=5,
        auto_config=False,
        enable_activation=True,
        enable_procedures=True,
        polish_answer=False,
        collect_corpus=False,
        apply_graph_edits=apply_graph_edits,
        enable_signature_live_bias=enable_signature_live_bias,
        signature_stats_dir=signature_stats_dir,
    )

    packet_data = _packet_summary(packet)
    raw_trace = list(packet_data.get("controller_raw_trace", []))
    raw_trace_text = _raw_trace_to_text(raw_trace)
    cot_log = list(packet_data.get("cot_log", []))

    packet_json = json.dumps(_jsonable(packet_data), ensure_ascii=False, indent=2)
    (case_dir / "packet.json").write_text(packet_json, encoding="utf-8")
    (case_dir / "raw_trace.json").write_text(json.dumps(_jsonable(raw_trace), ensure_ascii=False, indent=2), encoding="utf-8")
    (case_dir / "raw_trace.txt").write_text(raw_trace_text, encoding="utf-8")
    _write_cot_log(case_dir / "cot_log.txt", cot_log)
    (case_dir / "answer.txt").write_text(str(packet.answer or ""), encoding="utf-8")
    graph.save_json(case_dir / "graph_after.json")
    _copy_session_dir(packet.session_dir, case_dir / "session_dirs")

    result = {
        "case": dict(case),
        "case_dir": str(case_dir),
        "packet": packet_data,
        "packet_json": packet_json,
        "raw_trace_text": raw_trace_text,
        "cot_log_text": (case_dir / "cot_log.txt").read_text(encoding="utf-8"),
    }
    return result


def _compact_case_summary(result: Mapping[str, Any]) -> Dict[str, Any]:
    case = result["case"]
    packet = result["packet"]
    slot_stats = packet.get("slot_fill_stats") or {}
    return {
        "id": case.get("id"),
        "difficulty": case.get("difficulty"),
        "question": case.get("question"),
        "expected": case.get("expected"),
        "execution_mode": packet.get("execution_mode"),
        "task_type": packet.get("task_type"),
        "controller_task_family": packet.get("controller_task_family"),
        "finalized": packet.get("finalized"),
        "steps": packet.get("steps"),
        "max_steps": packet.get("max_steps"),
        "tool_call_count": packet.get("tool_call_count"),
        "controller_call_count": packet.get("controller_call_count"),
        "controller_nonempty_turns": packet.get("controller_nonempty_turns"),
        "elapsed_sec": packet.get("elapsed_sec"),
        "anchors": packet.get("anchors", []),
        "shortcut_reason": packet.get("shortcut_reason", ""),
        "controller_fallback_used": packet.get("controller_fallback_used"),
        "subgoal_reuse_count": packet.get("subgoal_reuse_count"),
        "slot_fill_stats": slot_stats,
        "controller_action_counts": packet.get("controller_action_counts", {}),
        "graph_edits_applied": packet.get("graph_edits_applied"),
        "graph_edits_count": len(packet.get("graph_edits") or []),
        "scoped_patches_count": len(packet.get("scoped_patches") or []),
        "scoped_patch_summary": packet.get("scoped_patch_summary") or {},
        "signature_candidate_count": len(packet.get("signature_candidates") or []),
        "signature_event_count": len(packet.get("signature_events") or []),
        "signature_focus_family_count": int((((packet.get("signature_graph_projection") or {}).get("summary") or {}).get("family_count") or 0)),
        "signature_focus_variant_count": len((((packet.get("signature_graph_projection") or {}).get("focus_variant_ids") or []))),
        "signature_resolution_counts": (packet.get("signature_stats_update") or {}).get("variant_resolution_counts", {}),
        "signature_live_bias_enabled": bool((packet.get("signature_live_bias") or {}).get("enabled")),
        "signature_live_bias_applied": bool((packet.get("signature_live_bias") or {}).get("applied")),
        "signature_live_bias_reason": str((packet.get("signature_live_bias") or {}).get("reason", "") or ""),
        "signature_live_bias_family_id": str((packet.get("signature_live_bias") or {}).get("family_id", "") or ""),
        "signature_live_bias_variant_id": str((packet.get("signature_live_bias") or {}).get("variant_id", "") or ""),
        "signature_live_bias_anchor_ids": list((packet.get("signature_live_bias") or {}).get("applied_anchor_ids", [])),
        "answer_chars": len(str(packet.get("answer", ""))),
    }


def _write_full_report(out_dir: Path, results: List[Mapping[str, Any]], summary_rows: List[Mapping[str, Any]]) -> None:
    lines: List[str] = [
        "# V4 Difficulty Sweep",
        "",
        "This run probes whether the current pipeline scales work with task difficulty.",
        "Graph edits are disabled unless the run command explicitly enabled them.",
        "",
        "## Summary",
        "",
        "| Difficulty | Case | Mode | Steps | Tools | Controller Calls | Elapsed | Slots | Live Bias | Edits | Patches |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | --- | --- | ---: | ---: |",
    ]
    for row in summary_rows:
        slots = row.get("slot_fill_stats") or {}
        slot_text = f"{slots.get('filled_count', 0)}/{slots.get('required_count', 0)}"
        lines.append(
            "| {difficulty} | {id} | {mode} | {steps}/{max_steps} | {tools} | {calls} | {elapsed} | {slots} | {live_bias} | {edits} | {patches} |".format(
                difficulty=row.get("difficulty", ""),
                id=row.get("id", ""),
                mode=row.get("execution_mode", ""),
                steps=row.get("steps", ""),
                max_steps=row.get("max_steps", ""),
                tools=row.get("tool_call_count", ""),
                calls=row.get("controller_call_count", ""),
                elapsed=row.get("elapsed_sec", ""),
                slots=slot_text,
                live_bias=(
                    "applied"
                    if row.get("signature_live_bias_applied")
                    else ("eligible" if row.get("signature_live_bias_enabled") else "off")
                ),
                edits=row.get("graph_edits_count", ""),
                patches=row.get("scoped_patches_count", ""),
            )
        )

    lines.extend(["", "## Case Details", ""])
    for result, row in zip(results, summary_rows):
        case = result["case"]
        packet = result["packet"]
        lines.extend([
            f"### {case['difficulty'].title()}: {case['id']}",
            "",
            f"- Question: {case['question']}",
            f"- Expected behavior: {case.get('expected', '')}",
            f"- Case directory: {result['case_dir']}",
            f"- execution_mode: {row.get('execution_mode')}",
            f"- task_type: {row.get('task_type')}",
            f"- controller_task_family: {row.get('controller_task_family')}",
            f"- finalized: {row.get('finalized')}",
            f"- steps: {row.get('steps')}/{row.get('max_steps')}",
            f"- tool_call_count: {row.get('tool_call_count')}",
            f"- controller_call_count: {row.get('controller_call_count')}",
            f"- controller_nonempty_turns: {row.get('controller_nonempty_turns')}",
            f"- elapsed_sec: {row.get('elapsed_sec')}",
            f"- anchors: {', '.join(str(x) for x in row.get('anchors', []))}",
            f"- shortcut_reason: {row.get('shortcut_reason') or '(none)'}",
            f"- controller_fallback_used: {row.get('controller_fallback_used')}",
            f"- subgoal_reuse_count: {row.get('subgoal_reuse_count')}",
            f"- slot_fill_stats: `{json.dumps(_jsonable(row.get('slot_fill_stats')), ensure_ascii=False)}`",
            f"- controller_action_counts: `{json.dumps(_jsonable(row.get('controller_action_counts')), ensure_ascii=False)}`",
            f"- graph_edits_applied: {row.get('graph_edits_applied')}",
            f"- graph_edits_count: {row.get('graph_edits_count')}",
            f"- scoped_patches_count: {row.get('scoped_patches_count')}",
            f"- scoped_patch_summary: `{json.dumps(_jsonable(row.get('scoped_patch_summary')), ensure_ascii=False)}`",
            f"- signature_live_bias_enabled: {row.get('signature_live_bias_enabled')}",
            f"- signature_live_bias_applied: {row.get('signature_live_bias_applied')}",
            f"- signature_live_bias_reason: {row.get('signature_live_bias_reason') or '(none)'}",
            f"- signature_live_bias_family_id: {row.get('signature_live_bias_family_id') or '(none)'}",
            f"- signature_live_bias_anchor_ids: {', '.join(str(x) for x in row.get('signature_live_bias_anchor_ids', [])) or '(none)'}",
            "",
            "#### Answer",
            "",
            "```text",
            str(packet.get("answer", "")).rstrip(),
            "```",
            "",
            "#### Micro Steps",
            "",
            "```json",
            json.dumps(_jsonable(packet.get("micro_steps", [])), ensure_ascii=False, indent=2),
            "```",
            "",
            "#### Tool Log",
            "",
            "```json",
            json.dumps(_jsonable(packet.get("tool_log", [])), ensure_ascii=False, indent=2),
            "```",
            "",
            "#### Raw Big-Pickle Trace",
            "",
            "```text",
            str(result.get("raw_trace_text", "")).rstrip(),
            "```",
            "",
            "#### CoT / Tool Transcript",
            "",
            "```text",
            str(result.get("cot_log_text", "")).rstrip(),
            "```",
            "",
            "#### Full Packet JSON",
            "",
            "```json",
            str(result.get("packet_json", "")).rstrip(),
            "```",
            "",
        ])
    _write_text(out_dir / "full_difficulty_sweep.md", lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run answerer_v4 across trivial/medium/hard tasks and dump everything.")
    parser.add_argument("--graph", type=Path, default=DEFAULT_GRAPH)
    parser.add_argument("--graph-id", default="merged_graph")
    parser.add_argument("--model", default="opencode/big-pickle")
    parser.add_argument("--server-url", default="http://127.0.0.1:4096")
    parser.add_argument("--config-dir", default=DEFAULT_CONFIG_DIR)
    parser.add_argument("--timeout", type=float, default=360.0)
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    parser.add_argument("--apply-graph-edits", action="store_true")
    parser.add_argument("--enable-signature-live-bias", action="store_true")
    parser.add_argument("--signature-stats-dir", default="data/signature_stats")
    parser.add_argument(
        "--cases-json",
        type=Path,
        default=None,
        help="Optional JSON file containing a list of case objects with id, difficulty, question, max_steps, expected.",
    )
    args = parser.parse_args()

    cases = DEFAULT_CASES
    if args.cases_json is not None:
        cases = json.loads(args.cases_json.read_text(encoding="utf-8"))

    out_dir = args.out_root / _now_slug()
    out_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(args.graph, out_dir / "graph_start.json")

    results: List[Mapping[str, Any]] = []
    summary_rows: List[Mapping[str, Any]] = []
    for index, case in enumerate(cases, start=1):
        print(f"[difficulty-sweep] running {case['difficulty']}: {case['id']}", flush=True)
        result = _run_case(
            index=index,
            case=case,
            graph_path=args.graph,
            graph_id=args.graph_id,
            model=args.model,
            server_url=args.server_url,
            config_dir=args.config_dir,
            timeout=args.timeout,
            out_dir=out_dir,
            apply_graph_edits=args.apply_graph_edits,
            enable_signature_live_bias=args.enable_signature_live_bias,
            signature_stats_dir=args.signature_stats_dir,
        )
        row = _compact_case_summary(result)
        results.append(result)
        summary_rows.append(row)
        print(json.dumps(_jsonable(row), ensure_ascii=False, indent=2), flush=True)

    summary = {
        "out_dir": str(out_dir),
        "graph": str(args.graph),
        "model": args.model,
        "server_url": args.server_url,
        "config_dir": args.config_dir,
        "apply_graph_edits": bool(args.apply_graph_edits),
        "enable_signature_live_bias": bool(args.enable_signature_live_bias),
        "signature_stats_dir": str(args.signature_stats_dir),
        "cases": _jsonable(summary_rows),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_full_report(out_dir, results, summary_rows)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
