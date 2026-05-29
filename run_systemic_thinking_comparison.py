from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from dataclasses import asdict, is_dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping

from answerer_v4 import V4OpencodeController, answer_query_v4
from graph_core import MemoryGraph


DEFAULT_QUESTION = (
    "Design a real-time leaderboard service for a competitive programming platform. "
    "Requirements: 500K+ concurrent users, score updates must propagate within 100ms, "
    "'what is my current rank?' queries must run in O(log n), and the API must support "
    "range pagination for ranks 1000-1020. Walk through the full design: choose the core "
    "data structure, handle ties, sketch the distributed architecture, and identify one "
    "subtle correctness issue with a fix."
)
DEFAULT_GRAPH = Path("graphs/merged_graph.json")
DEFAULT_OUT_ROOT = Path("artifacts/systemic_thinking_comparison")
DEFAULT_CONFIG_DIR = "pure-opencode"


BASELINE_SYSTEM_PROMPT = """\
You are a senior systems and algorithms designer.

This is the RAW BASELINE condition. Do not use tools, files, graph memory, or external lookup.

Do exactly one reasoning pass, then final answer:
<reasoning>
Concise chain-of-thought style reasoning in one pass only. Plan the system, choose tradeoffs, and sanity-check correctness.
</reasoning>

<answer>
The final answer.
</answer>
"""


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
        "explanation": packet.explanation,
        "polish_applied": packet.polish_applied,
        "micro_steps": _jsonable(packet.micro_steps),
    }


def _write_text(path: Path, lines: Iterable[str]) -> None:
    path.write_text("\n".join(lines), encoding="utf-8")


def _find_opencode() -> str:
    for candidate in ("opencode", "opencode.cmd"):
        found = shutil.which(candidate)
        if found:
            return found
    raise FileNotFoundError("opencode not found on PATH")


def _parse_opencode_text(stdout: str) -> tuple[str, List[Dict[str, Any]]]:
    text_parts: List[str] = []
    events: List[Dict[str, Any]] = []
    for raw_line in stdout.splitlines():
        raw_line = raw_line.strip()
        if not raw_line:
            continue
        try:
            event = json.loads(raw_line)
        except json.JSONDecodeError:
            continue
        if not isinstance(event, dict):
            continue
        events.append(event)
        if event.get("type") == "text":
            text_parts.append(event.get("part", {}).get("text", ""))
    return "".join(text_parts), events


def run_raw_baseline(
    *,
    question: str,
    model: str,
    server_url: str,
    config_dir: str,
    timeout: float,
) -> Dict[str, Any]:
    exe = _find_opencode()
    env = dict(os.environ)
    env["OPENCODE_CONFIG_DIR"] = config_dir
    cmd = [exe, "run", "--model", model, "--format", "json"]
    if server_url:
        cmd += ["--attach", server_url]

    prompt = f"{BASELINE_SYSTEM_PROMPT}\n---\n\nQuestion:\n{question}"
    t0 = time.time()
    proc = subprocess.run(
        cmd,
        input=prompt,
        capture_output=True,
        text=True,
        timeout=timeout,
        encoding="utf-8",
        errors="replace",
        shell=False,
        env=env,
    )
    elapsed = round(time.time() - t0, 3)
    assistant_text, events = _parse_opencode_text(proc.stdout)
    if not assistant_text and proc.returncode != 0:
        raise RuntimeError(f"opencode failed: {proc.stderr[:500]}")
    return {
        "cmd": cmd,
        "prompt": prompt,
        "raw_stdout": proc.stdout,
        "raw_stderr": proc.stderr,
        "events": events,
        "assistant_text": assistant_text,
        "returncode": proc.returncode,
        "elapsed_sec": elapsed,
    }


def _write_raw_baseline_text(path: Path, payload: Mapping[str, Any]) -> None:
    lines = [
        "==================== RAW BASELINE ====================",
        f"elapsed_sec: {payload.get('elapsed_sec', '')}",
        f"returncode: {payload.get('returncode', '')}",
        "cmd: " + " ".join(str(x) for x in payload.get("cmd", [])),
        "",
        "--- prompt ---",
        str(payload.get("prompt", "")),
        "",
        "--- raw_stdout ---",
        str(payload.get("raw_stdout", "")),
        "",
        "--- raw_stderr ---",
        str(payload.get("raw_stderr", "")),
        "",
        "--- assistant_text ---",
        str(payload.get("assistant_text", "")),
    ]
    _write_text(path, lines)


def _write_raw_trace_text(path: Path, raw_trace: List[Mapping[str, Any]]) -> None:
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
    _write_text(path, lines)


def _write_cot_log(path: Path, cot_log: List[str]) -> None:
    lines: List[str] = []
    for i, chunk in enumerate(cot_log, start=1):
        lines.append(f"==================== TURN {i} ====================")
        lines.append(chunk)
        lines.append("")
    _write_text(path, lines)


def run_graph_pipeline(
    *,
    question: str,
    graph_path: Path,
    graph_id: str,
    model: str,
    server_url: str,
    config_dir: str,
    timeout: float,
    max_steps: int,
    apply_graph_edits: bool,
    enable_signature_live_bias: bool,
    signature_stats_dir: str | Path,
) -> Dict[str, Any]:
    graph = MemoryGraph.load_json(graph_path)
    controller = V4OpencodeController(
        model=model,
        server_url=server_url,
        config_dir=config_dir,
        timeout=timeout,
        print_raw_output=False,
    )
    packet = answer_query_v4(
        question=question,
        graph=graph,
        controller=controller,
        graph_id=graph_id,
        max_steps=max_steps,
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
    return {
        "packet": _packet_summary(packet),
        "graph": graph,
    }


def _write_report(
    *,
    out_dir: Path,
    question: str,
    model: str,
    baseline: Mapping[str, Any],
    graph_packet: Mapping[str, Any],
    baseline_text: str,
    graph_trace_text: str,
    graph_cot_text: str,
    graph_packet_json: str,
) -> None:
    lines = [
        "# Raw Baseline vs Graph Pipeline",
        "",
        f"- Question: {question}",
        f"- Model: {model}",
        "- Baseline condition: raw opencode, no tools, exactly one requested reasoning pass.",
        "- Graph condition: answerer_v4 with graph tools and full packet/raw trace capture.",
        "",
        "## Summary",
        "",
        f"- baseline elapsed_sec: {baseline.get('elapsed_sec')}",
        f"- baseline assistant chars: {len(str(baseline.get('assistant_text', '')))}",
        f"- graph execution_mode: {graph_packet.get('execution_mode')}",
        f"- graph finalized: {graph_packet.get('finalized')}",
        f"- graph steps: {graph_packet.get('steps')}/{graph_packet.get('max_steps')}",
        f"- graph tool_call_count: {graph_packet.get('tool_call_count')}",
        f"- graph controller_call_count: {graph_packet.get('controller_call_count')}",
        f"- graph elapsed_sec: {graph_packet.get('elapsed_sec')}",
        f"- graph anchors: {', '.join(graph_packet.get('anchors', []))}",
        f"- graph graph_edits_applied: {graph_packet.get('graph_edits_applied')}",
        f"- graph graph_edits_count: {len(graph_packet.get('graph_edits', []))}",
        f"- graph scoped_patches_count: {len(graph_packet.get('scoped_patches', []))}",
        f"- graph scoped_patch_summary: `{json.dumps(_jsonable(graph_packet.get('scoped_patch_summary', {})), ensure_ascii=False)}`",
        f"- graph signature_live_bias: `{json.dumps(_jsonable(graph_packet.get('signature_live_bias', {})), ensure_ascii=False)}`",
        "",
        "## Baseline Assistant Text",
        "",
        "```text",
        str(baseline.get("assistant_text", "")).rstrip(),
        "```",
        "",
        "## Graph Answer",
        "",
        "```text",
        str(graph_packet.get("answer", "")).rstrip(),
        "```",
        "",
        "## Baseline Full Raw Dump",
        "",
        "```text",
        baseline_text.rstrip(),
        "```",
        "",
        "## Graph Raw Big-Pickle Dump",
        "",
        "```text",
        graph_trace_text.rstrip(),
        "```",
        "",
        "## Graph CoT / Tool Transcript",
        "",
        "```text",
        graph_cot_text.rstrip(),
        "```",
        "",
        "## Graph Full Packet JSON",
        "",
        "```json",
        graph_packet_json.rstrip(),
        "```",
    ]
    _write_text(out_dir / "full_comparison_report.md", lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare raw one-pass opencode baseline against graph pipeline.")
    parser.add_argument("--question", default=DEFAULT_QUESTION)
    parser.add_argument("--graph", type=Path, default=DEFAULT_GRAPH)
    parser.add_argument("--graph-id", default="merged_graph")
    parser.add_argument("--model", default="opencode/big-pickle")
    parser.add_argument("--server-url", default="http://127.0.0.1:4096")
    parser.add_argument("--config-dir", default=DEFAULT_CONFIG_DIR)
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--max-steps", type=int, default=12)
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    parser.add_argument("--apply-graph-edits", action="store_true")
    parser.add_argument("--enable-signature-live-bias", action="store_true")
    parser.add_argument("--signature-stats-dir", default="data/signature_stats")
    args = parser.parse_args()

    out_dir = args.out_root / _now_slug()
    out_dir.mkdir(parents=True, exist_ok=True)

    baseline = run_raw_baseline(
        question=args.question,
        model=args.model,
        server_url=args.server_url,
        config_dir=args.config_dir,
        timeout=args.timeout,
    )
    baseline_json = json.dumps(_jsonable(baseline), ensure_ascii=False, indent=2)
    (out_dir / "baseline_raw.json").write_text(baseline_json, encoding="utf-8")
    _write_raw_baseline_text(out_dir / "baseline_raw.txt", baseline)

    graph_result = run_graph_pipeline(
        question=args.question,
        graph_path=args.graph,
        graph_id=args.graph_id,
        model=args.model,
        server_url=args.server_url,
        config_dir=args.config_dir,
        timeout=args.timeout,
        max_steps=args.max_steps,
        apply_graph_edits=args.apply_graph_edits,
        enable_signature_live_bias=args.enable_signature_live_bias,
        signature_stats_dir=args.signature_stats_dir,
    )
    graph_packet = graph_result["packet"]
    graph_packet_json = json.dumps(_jsonable(graph_packet), ensure_ascii=False, indent=2)
    (out_dir / "graph_packet.json").write_text(graph_packet_json, encoding="utf-8")
    _write_raw_trace_text(out_dir / "graph_raw_trace.txt", list(graph_packet.get("controller_raw_trace", [])))
    (out_dir / "graph_raw_trace.json").write_text(
        json.dumps(_jsonable(graph_packet.get("controller_raw_trace", [])), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    _write_cot_log(out_dir / "graph_cot_log.txt", list(graph_packet.get("cot_log", [])))
    (out_dir / "summary.json").write_text(
        json.dumps({
            "question": args.question,
            "model": args.model,
            "out_dir": str(out_dir),
            "baseline": {
                "elapsed_sec": baseline.get("elapsed_sec"),
                "assistant_chars": len(str(baseline.get("assistant_text", ""))),
                "returncode": baseline.get("returncode"),
            },
            "graph": {
                "execution_mode": graph_packet.get("execution_mode"),
                "finalized": graph_packet.get("finalized"),
                "steps": graph_packet.get("steps"),
                "tool_call_count": graph_packet.get("tool_call_count"),
                "controller_call_count": graph_packet.get("controller_call_count"),
                "elapsed_sec": graph_packet.get("elapsed_sec"),
                "answer_chars": len(str(graph_packet.get("answer", ""))),
                "graph_edits_applied": graph_packet.get("graph_edits_applied"),
                "graph_edits_count": len(graph_packet.get("graph_edits", [])),
                "scoped_patches_count": len(graph_packet.get("scoped_patches", [])),
                "scoped_patch_summary": graph_packet.get("scoped_patch_summary", {}),
                "signature_live_bias": graph_packet.get("signature_live_bias", {}),
            },
            "enable_signature_live_bias": bool(args.enable_signature_live_bias),
            "signature_stats_dir": str(args.signature_stats_dir),
        }, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    baseline_text = (out_dir / "baseline_raw.txt").read_text(encoding="utf-8")
    graph_trace_text = (out_dir / "graph_raw_trace.txt").read_text(encoding="utf-8")
    graph_cot_text = (out_dir / "graph_cot_log.txt").read_text(encoding="utf-8")
    _write_report(
        out_dir=out_dir,
        question=args.question,
        model=args.model,
        baseline=baseline,
        graph_packet=graph_packet,
        baseline_text=baseline_text,
        graph_trace_text=graph_trace_text,
        graph_cot_text=graph_cot_text,
        graph_packet_json=graph_packet_json,
    )

    print(json.dumps(json.loads((out_dir / "summary.json").read_text(encoding="utf-8")), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    main()
