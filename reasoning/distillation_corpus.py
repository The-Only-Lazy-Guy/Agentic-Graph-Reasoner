"""Distillation corpus writer.

Goal: every successful v4 session becomes a row of training data that we
can later use to distill v4's behavior into a smaller deployable model.

Format (single rich JSONL per session, append-only):
    data/distillation_corpus/sessions.jsonl

Each row is a JSON object with everything a downstream trainer might want:

  - session metadata (id, timestamp, graph_id, controller info)
  - input (question, retrieved anchors with snippets)
  - task frame (Phase 2 activation block, if any)
  - full trace (plan, cot_log, tool_call_log, hypotheses, failures,
    session objects, plan_tree summary)
  - outputs (raw answer, polished answer, explanation, reflection)
  - metrics (steps, tool calls, elapsed, coverage %, citation warns,
    search repeats, budget summary, meta signals)
  - quality signals (coverage_pct, finalized flag) — used downstream to
    decide which rows to include in SFT

This is deliberately PERMISSIVE — extra fields are fine; the trainer
picks what it needs. We never delete fields once written (would break
older snapshots), only add new optional ones.
"""
from __future__ import annotations

import dataclasses
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from answerer_v4 import V4Packet
    from graph_core import MemoryGraph


DEFAULT_CORPUS_ROOT = Path("data/distillation_corpus")
DEFAULT_CORPUS_FILE = "sessions.jsonl"

CORPUS_SCHEMA_VERSION = 1


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _serialize_obj(obj: Any) -> Any:
    """Best-effort JSON-safe serialization of session objects, FailureRecords, etc."""
    if dataclasses.is_dataclass(obj):
        return dataclasses.asdict(obj)
    if hasattr(obj, "to_dict"):
        try:
            return obj.to_dict()
        except Exception:
            pass
    # Mappings, lists, primitives
    if isinstance(obj, dict):
        return {k: _serialize_obj(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_serialize_obj(v) for v in obj]
    return obj


def packet_to_corpus_row(
    pkt: "V4Packet",
    graph: "MemoryGraph",
    *,
    controller_label: str = "",
    extra_metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Turn a V4Packet into the canonical corpus row dict.

    `controller_label` is a short tag like "opencode:big-pickle" so the
    trainer can stratify samples by model when needed.
    """
    # Resolve anchor snippets so the row is self-contained (you can train on
    # it without re-loading the original graph).
    anchor_records = []
    for aid in pkt.anchors:
        n = graph.nodes.get(aid) if hasattr(graph, "nodes") else None
        snippet = n.text if n else ""
        ntype = getattr(n, "node_type", "") if n else ""
        anchor_records.append({"id": aid, "node_type": ntype, "text": snippet[:600]})

    row: Dict[str, Any] = {
        # ── envelope ──
        "schema_version": CORPUS_SCHEMA_VERSION,
        "session_id": getattr(pkt, "session_dir", "") or "",
        "timestamp": _now_iso(),
        "graph_id": (graph.metadata.get("title")
                     if getattr(graph, "metadata", None) else "graph"),
        "controller": controller_label,

        # ── input ──
        "input": {
            "question": pkt.question,
            "anchors": anchor_records,
            "task_frame_items": pkt.task_frame_items,
            "task_type": getattr(pkt, "task_type", ""),
            "controller_task_family": getattr(pkt, "controller_task_family", ""),
        },

        # ── full trace ──
        "trace": {
            "plan": [{"text": sg.text, "done": sg.done} for sg in pkt.plan],
            "plan_tree": pkt.plan_tree_summary,
            "tool_calls": _serialize_obj(pkt.tool_log),
            "cot_log": list(pkt.cot_log),
            "hypotheses": dict(pkt.hypotheses),
            "failures": [_serialize_obj(f) for f in pkt.failures],
            "session_objects": {
                v4id: _serialize_obj(obj)
                for v4id, obj in pkt.objects.items()
            },
            "procedure_invocations": _serialize_obj(pkt.procedure_invocations),
            "micro_steps": _serialize_obj(getattr(pkt, "micro_steps", [])),
            "scoped_patches": _serialize_obj(getattr(pkt, "scoped_patches", [])),
        },

        # ── outputs ──
        "outputs": {
            "answer_raw": pkt.answer_raw,
            "answer_polished": pkt.answer,
            "explanation": pkt.explanation,
            "reflection": pkt.reflection,
        },

        # ── metrics ──
        "metrics": {
            "steps": pkt.steps,
            "max_steps": pkt.max_steps,
            "tool_call_count": pkt.tool_call_count,
            "elapsed_sec": pkt.elapsed_sec,
            "finalized": pkt.finalized,
            "execution_mode": getattr(pkt, "execution_mode", "loop"),
            "controller_task_family": getattr(pkt, "controller_task_family", ""),
            "shortcut_anchor_ids": list(getattr(pkt, "shortcut_anchor_ids", []) or []),
            "citation_warnings": pkt.citation_warnings,
            "search_repeats": pkt.search_repeats,
            "task_frame_items": pkt.task_frame_items,
            "activation_signals": pkt.activation_signals,
            "coverage_addressed_pct": pkt.coverage_addressed_pct,
            "coverage_rounds": pkt.coverage_rounds,
            "subgoal_reuse_count": getattr(pkt, "subgoal_reuse_count", 0),
            "slot_fill_stats": _serialize_obj(getattr(pkt, "slot_fill_stats", {})),
            "controller_action_counts": _serialize_obj(getattr(pkt, "controller_action_counts", {})),
            "controller_fallback_used": bool(getattr(pkt, "controller_fallback_used", False)),
            "polish_applied": pkt.polish_applied,
            "budget_summary": pkt.budget_summary,
            "meta_signals_count": len(pkt.meta_signals),
            "graph_edits_proposed": len(pkt.graph_edits),
            "graph_edits_applied": pkt.graph_edits_applied,
            "scoped_patch_summary": _serialize_obj(getattr(pkt, "scoped_patch_summary", {})),
            "reflection_edits_proposed": len(pkt.reflection_edits),
            "reflection_applied": pkt.reflection_applied,
        },

        # ── quality signals (for downstream filtering) ──
        "quality": {
            "finalized": pkt.finalized,
            "coverage_pct": pkt.coverage_addressed_pct,
            "had_polish": pkt.polish_applied,
            # Crude "interesting" heuristic: hard tasks tend to produce
            # objects/hypotheses/failures; easy tasks don't.
            "complexity_proxy_score": (
                len(pkt.objects) + len(pkt.hypotheses) + len(pkt.failures)
                + (1 if pkt.plan_tree_summary else 0)
            ),
        },
    }
    if extra_metadata:
        row["extra"] = dict(extra_metadata)
    return row


def append_session_to_corpus(
    pkt: "V4Packet",
    graph: "MemoryGraph",
    *,
    corpus_root: Path = DEFAULT_CORPUS_ROOT,
    corpus_file: str = DEFAULT_CORPUS_FILE,
    controller_label: str = "",
    extra_metadata: Optional[Dict[str, Any]] = None,
) -> Path:
    """Append one session as a JSONL row. Returns the file path."""
    corpus_root = Path(corpus_root)
    corpus_root.mkdir(parents=True, exist_ok=True)
    out_path = corpus_root / corpus_file
    row = packet_to_corpus_row(
        pkt, graph,
        controller_label=controller_label,
        extra_metadata=extra_metadata,
    )
    line = json.dumps(row, ensure_ascii=False) + "\n"
    with out_path.open("a", encoding="utf-8") as f:
        f.write(line)
    return out_path


def corpus_stats(
    corpus_root: Path = DEFAULT_CORPUS_ROOT,
    corpus_file: str = DEFAULT_CORPUS_FILE,
) -> Dict[str, Any]:
    """Quick health-check stats on the persisted corpus."""
    p = Path(corpus_root) / corpus_file
    if not p.exists():
        return {"rows": 0, "path": str(p), "exists": False}
    n = 0
    finalized = 0
    total_complexity = 0
    by_controller: Dict[str, int] = {}
    with p.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            n += 1
            if row.get("quality", {}).get("finalized"):
                finalized += 1
            total_complexity += row.get("quality", {}).get("complexity_proxy_score", 0)
            ctrl = row.get("controller", "")
            by_controller[ctrl] = by_controller.get(ctrl, 0) + 1
    return {
        "rows": n,
        "finalized": finalized,
        "mean_complexity_proxy": (total_complexity / n) if n else 0.0,
        "by_controller": by_controller,
        "path": str(p),
        "size_bytes": p.stat().st_size,
        "exists": True,
    }
