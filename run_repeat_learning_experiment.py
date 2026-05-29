from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import asdict, is_dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping

from answerer_v4 import V4OpencodeController, answer_query_v4
from graph_core import Edge, MemoryGraph, Node


DEFAULT_QUESTION = "Design a C++ data structure supporting point updates and prefix sum queries."
DEFAULT_GRAPH = Path("graphs/merged_graph.json")
DEFAULT_OUT_ROOT = Path("artifacts/repeat_learning")


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
    }


def _node_snapshot(graph: MemoryGraph) -> Dict[str, Dict[str, Any]]:
    return {
        nid: {
            "node_type": node.node_type,
            "confidence": node.confidence,
            "text": node.text,
            "metadata": dict(node.metadata or {}),
        }
        for nid, node in graph.nodes.items()
    }


def _edge_snapshot(graph: MemoryGraph) -> List[Dict[str, Any]]:
    return [
        {
            "src": edge.src,
            "dst": edge.dst,
            "relation": edge.relation,
            "metadata": dict(edge.metadata or {}),
        }
        for edge in graph.edges
    ]


def _graph_diff(before_nodes: Mapping[str, Dict[str, Any]], after_graph: MemoryGraph, before_edge_count: int) -> Dict[str, Any]:
    added_nodes: List[Dict[str, Any]] = []
    deprecated_nodes: List[str] = []
    for nid, node in after_graph.nodes.items():
        if nid not in before_nodes:
            added_nodes.append({
                "id": nid,
                "node_type": node.node_type,
                "confidence": node.confidence,
                "text": node.text,
                "metadata": dict(node.metadata or {}),
            })
        before_meta = before_nodes.get(nid, {}).get("metadata", {})
        after_meta = dict(node.metadata or {})
        if not before_meta.get("deprecated") and after_meta.get("deprecated"):
            deprecated_nodes.append(nid)
    return {
        "node_count_before": len(before_nodes),
        "node_count_after": len(after_graph.nodes),
        "edge_count_before": before_edge_count,
        "edge_count_after": len(after_graph.edges),
        "added_nodes": added_nodes,
        "deprecated_nodes": deprecated_nodes,
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
        lines.append(f"variant: {entry.get('variant', '')}")
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


def _copy_session_dir(session_dir: str | None, dst_root: Path) -> None:
    if not session_dir:
        return
    src = Path(session_dir)
    if not src.exists():
        return
    dst = dst_root / src.name
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)


def _make_controller(model: str, server_url: str, timeout: float) -> V4OpencodeController:
    return V4OpencodeController(model=model, server_url=server_url, timeout=timeout)


def _run_once(
    *,
    run_index: int,
    question: str,
    graph: MemoryGraph,
    graph_input_path: Path,
    graph_id: str,
    model: str,
    server_url: str,
    timeout: float,
    out_dir: Path,
    enable_signature_live_bias: bool,
    signature_stats_dir: str | Path,
) -> Dict[str, Any]:
    before_nodes = _node_snapshot(graph)
    before_edge_count = len(graph.edges)
    controller = _make_controller(model=model, server_url=server_url, timeout=timeout)
    packet = answer_query_v4(
        question=question,
        graph=graph,
        controller=controller,
        graph_id=graph_id,
        max_steps=10,
        enable_activation=False,
        polish_answer=False,
        collect_corpus=False,
        apply_graph_edits=True,
        auto_config=False,
        enable_signature_live_bias=enable_signature_live_bias,
        signature_stats_dir=signature_stats_dir,
    )
    packet_data = _packet_summary(packet)
    graph_diff = _graph_diff(before_nodes, graph, before_edge_count)
    graph_path = out_dir / f"graph_after_run_{run_index}.json"
    graph.save_json(graph_path)
    packet_path = out_dir / f"run_{run_index}_packet.json"
    packet_path.write_text(json.dumps(_jsonable(packet_data), ensure_ascii=False, indent=2), encoding="utf-8")
    _write_cot_log(out_dir / f"run_{run_index}_cot_log.txt", list(packet.cot_log))
    raw_trace = list(packet_data.get("controller_raw_trace", []))
    (out_dir / f"run_{run_index}_raw_trace.json").write_text(
        json.dumps(_jsonable(raw_trace), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    _write_raw_trace_text(out_dir / f"run_{run_index}_raw_trace.txt", raw_trace)
    (out_dir / f"run_{run_index}_answer.txt").write_text(packet.answer or "", encoding="utf-8")
    diff_path = out_dir / f"run_{run_index}_graph_diff.json"
    diff_path.write_text(json.dumps(_jsonable(graph_diff), ensure_ascii=False, indent=2), encoding="utf-8")
    _copy_session_dir(packet.session_dir, out_dir / "session_dirs")
    return {
        "packet": packet_data,
        "graph_diff": graph_diff,
        "graph_input_path": str(graph_input_path),
        "graph_path": str(graph_path),
    }


def _write_compare_report(
    *,
    out_dir: Path,
    question: str,
    model: str,
    graph_source: Path,
    run1: Mapping[str, Any],
    run2: Mapping[str, Any],
) -> None:
    p1 = run1["packet"]
    p2 = run2["packet"]
    d1 = run1["graph_diff"]
    d2 = run2["graph_diff"]
    lines = [
        "# Repeat Learning Experiment",
        "",
        f"- Question: {question}",
        f"- Model: {model}",
        f"- Source graph: {graph_source}",
        "- Persistence mode: snapshot-only; run 1 writes to a copied graph, not back to the source graph file.",
        "",
        "## Run 1",
        f"- graph_used: {run1['graph_input_path']}",
        f"- execution_mode: {p1['execution_mode']}",
        f"- steps: {p1['steps']}",
        f"- tool_call_count: {p1['tool_call_count']}",
        f"- elapsed_sec: {p1['elapsed_sec']}",
        f"- controller_call_count: {p1['controller_call_count']}",
        f"- controller_total_elapsed_sec: {p1['controller_total_elapsed_sec']}",
        f"- controller_nonempty_turns: {p1['controller_nonempty_turns']}/{p1['controller_call_count']}",
        f"- anchors: {', '.join(p1['anchors'])}",
        f"- graph_edits_applied: {p1['graph_edits_applied']}",
        f"- scoped_patches_count: {len(p1.get('scoped_patches', []))}",
        f"- scoped_patch_summary: `{json.dumps(_jsonable(p1.get('scoped_patch_summary', {})), ensure_ascii=False)}`",
        f"- signature_live_bias: `{json.dumps(_jsonable(p1.get('signature_live_bias', {})), ensure_ascii=False)}`",
        f"- added_nodes: {len(d1['added_nodes'])}",
        f"- deprecated_nodes: {', '.join(d1['deprecated_nodes']) or '(none)'}",
        "",
        "## Run 2",
        f"- graph_used: {run2['graph_input_path']}",
        f"- execution_mode: {p2['execution_mode']}",
        f"- steps: {p2['steps']}",
        f"- tool_call_count: {p2['tool_call_count']}",
        f"- elapsed_sec: {p2['elapsed_sec']}",
        f"- controller_call_count: {p2['controller_call_count']}",
        f"- controller_total_elapsed_sec: {p2['controller_total_elapsed_sec']}",
        f"- controller_nonempty_turns: {p2['controller_nonempty_turns']}/{p2['controller_call_count']}",
        f"- anchors: {', '.join(p2['anchors'])}",
        f"- graph_edits_applied: {p2['graph_edits_applied']}",
        f"- scoped_patches_count: {len(p2.get('scoped_patches', []))}",
        f"- scoped_patch_summary: `{json.dumps(_jsonable(p2.get('scoped_patch_summary', {})), ensure_ascii=False)}`",
        f"- signature_live_bias: `{json.dumps(_jsonable(p2.get('signature_live_bias', {})), ensure_ascii=False)}`",
        f"- added_nodes: {len(d2['added_nodes'])}",
        f"- deprecated_nodes: {', '.join(d2['deprecated_nodes']) or '(none)'}",
        "",
        "## Comparison",
        f"- steps_delta: {p2['steps'] - p1['steps']}",
        f"- tool_call_delta: {p2['tool_call_count'] - p1['tool_call_count']}",
        f"- elapsed_delta: {round(float(p2['elapsed_sec']) - float(p1['elapsed_sec']), 3)}",
        f"- controller_call_delta: {p2['controller_call_count'] - p1['controller_call_count']}",
        f"- controller_elapsed_delta: {round(float(p2['controller_total_elapsed_sec']) - float(p1['controller_total_elapsed_sec']), 3)}",
        f"- empty_turn_delta: {(p2['controller_call_count'] - p2['controller_nonempty_turns']) - (p1['controller_call_count'] - p1['controller_nonempty_turns'])}",
        f"- run2_new_anchors_from_run1: {', '.join([nid for nid in p2['anchors'] if nid not in p1['anchors']]) or '(none)'}",
        "",
        "## Answer 1",
        p1["answer"],
        "",
        "## Answer 2",
        p2["answer"],
        "",
        "## Files",
        "- run_1_packet.json",
        "- run_2_packet.json",
        "- run_1_cot_log.txt",
        "- run_2_cot_log.txt",
        "- run_1_raw_trace.json",
        "- run_2_raw_trace.json",
        "- run_1_raw_trace.txt",
        "- run_2_raw_trace.txt",
        "- run_1_graph_diff.json",
        "- run_2_graph_diff.json",
        "- graph_after_run_1.json",
        "- graph_after_run_2.json",
    ]
    _write_text(out_dir / "compare.md", lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the same question twice and compare learning artifacts.")
    parser.add_argument("--question", default=DEFAULT_QUESTION)
    parser.add_argument("--graph", type=Path, default=DEFAULT_GRAPH)
    parser.add_argument("--graph-id", default="repeat_learning_graph")
    parser.add_argument("--model", default="opencode/big-pickle")
    parser.add_argument("--server-url", default="http://127.0.0.1:4096")
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    parser.add_argument("--enable-signature-live-bias", action="store_true")
    parser.add_argument("--signature-stats-dir", default="data/signature_stats")
    args = parser.parse_args()

    out_dir = args.out_root / _now_slug()
    out_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(args.graph, out_dir / "graph_start.json")

    graph1 = MemoryGraph.load_json(args.graph)
    run1 = _run_once(
        run_index=1,
        question=args.question,
        graph=graph1,
        graph_input_path=args.graph,
        graph_id=args.graph_id,
        model=args.model,
        server_url=args.server_url,
        timeout=args.timeout,
        out_dir=out_dir,
        enable_signature_live_bias=args.enable_signature_live_bias,
        signature_stats_dir=args.signature_stats_dir,
    )

    graph2 = MemoryGraph.load_json(out_dir / "graph_after_run_1.json")
    run2 = _run_once(
        run_index=2,
        question=args.question,
        graph=graph2,
        graph_input_path=out_dir / "graph_after_run_1.json",
        graph_id=args.graph_id,
        model=args.model,
        server_url=args.server_url,
        timeout=args.timeout,
        out_dir=out_dir,
        enable_signature_live_bias=args.enable_signature_live_bias,
        signature_stats_dir=args.signature_stats_dir,
    )

    summary = {
        "question": args.question,
        "model": args.model,
        "graph_source": str(args.graph),
        "out_dir": str(out_dir),
        "enable_signature_live_bias": bool(args.enable_signature_live_bias),
        "signature_stats_dir": str(args.signature_stats_dir),
        "run_1": run1,
        "run_2": run2,
    }
    (out_dir / "summary.json").write_text(json.dumps(_jsonable(summary), ensure_ascii=False, indent=2), encoding="utf-8")
    _write_compare_report(
        out_dir=out_dir,
        question=args.question,
        model=args.model,
        graph_source=args.graph,
        run1=run1,
        run2=run2,
    )
    print(json.dumps({
        "out_dir": str(out_dir),
        "run_1": {
            "execution_mode": run1["packet"]["execution_mode"],
            "steps": run1["packet"]["steps"],
            "tool_call_count": run1["packet"]["tool_call_count"],
            "elapsed_sec": run1["packet"]["elapsed_sec"],
            "anchors": run1["packet"]["anchors"],
            "added_nodes": [n["id"] for n in run1["graph_diff"]["added_nodes"]],
        },
        "run_2": {
            "execution_mode": run2["packet"]["execution_mode"],
            "steps": run2["packet"]["steps"],
            "tool_call_count": run2["packet"]["tool_call_count"],
            "elapsed_sec": run2["packet"]["elapsed_sec"],
            "anchors": run2["packet"]["anchors"],
            "added_nodes": [n["id"] for n in run2["graph_diff"]["added_nodes"]],
        },
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
