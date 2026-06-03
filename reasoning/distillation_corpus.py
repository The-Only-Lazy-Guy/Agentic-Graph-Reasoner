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
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, TYPE_CHECKING

from reasoning.lexical_matching import lexical_overlap

if TYPE_CHECKING:
    from answerer_v4 import V4Packet
    from graph_core import MemoryGraph


DEFAULT_CORPUS_ROOT = Path("data/distillation_corpus")
DEFAULT_CORPUS_FILE = "sessions.jsonl"

# V1: SFT distillation fields (answer, trace, metrics)
# V2: V5 cross-attention trajectory fields (nodes_accessed_log, node_type_breakdown,
#     trajectory_reward_score). V2 is a strict superset of V1 — old readers
#     that ignore unknown fields remain compatible.
CORPUS_SCHEMA_VERSION = 2

_TOOL_BLOCK_RE = re.compile(r"<(?:tool|graph_action)>\s*(.*?)\s*</(?:tool|graph_action)>", re.DOTALL)
_ANSWER_BLOCK_RE = re.compile(r"<answer\b", re.IGNORECASE)
_PATCH_BLOCK_RE = re.compile(r"<patch\b", re.IGNORECASE)
_EVIDENCE_AUDIT_BLOCK_RE = re.compile(r"<evidence_audit\b", re.IGNORECASE)


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


def _build_node_type_counts(nodes_accessed_log: list) -> Dict[str, int]:
    """Summarise nodes_accessed_log into a {node_type: count} dict."""
    counts: Dict[str, int] = {}
    for entry in nodes_accessed_log:
        ntype = entry.get("node_type", "unknown") if isinstance(entry, dict) else "unknown"
        counts[ntype] = counts.get(ntype, 0) + 1
    return counts


def _tool_names_from_turn(text: str) -> list[str]:
    names: list[str] = []
    for match in _TOOL_BLOCK_RE.finditer(str(text or "")):
        try:
            payload = json.loads(match.group(1))
        except json.JSONDecodeError:
            names.append("<parse_error>")
            continue
        name = payload.get("name")
        names.append(str(name) if isinstance(name, str) else "<shape_error>")
    return names


def _turn_summaries(cot_log: list[str]) -> list[Dict[str, Any]]:
    summaries: list[Dict[str, Any]] = []
    for idx, text in enumerate(cot_log):
        raw = str(text or "")
        tool_names = _tool_names_from_turn(raw)
        counts: Dict[str, int] = {}
        for name in tool_names:
            counts[name] = counts.get(name, 0) + 1
        summaries.append({
            "turn_index": idx,
            "chars": len(raw),
            "tool_call_count": len(tool_names),
            "tool_names": tool_names[:20],
            "tool_counts": counts,
            "has_answer": bool(_ANSWER_BLOCK_RE.search(raw)),
            "has_patch": bool(_PATCH_BLOCK_RE.search(raw)),
            "has_evidence_audit": bool(_EVIDENCE_AUDIT_BLOCK_RE.search(raw)),
            "preview": raw[:500],
        })
    return summaries


def _controller_raw_trace_summary(raw_trace: list[Dict[str, Any]]) -> list[Dict[str, Any]]:
    out: list[Dict[str, Any]] = []
    for idx, entry in enumerate(raw_trace):
        if not isinstance(entry, dict):
            out.append({
                "turn_index": idx,
                "mode": None,
                "elapsed_sec": None,
                "session_id_after": None,
                "assistant_chars": 0,
                "assistant_nonempty": False,
                "event_count": 0,
            })
            continue
        assistant_text = str(entry.get("assistant_text", "") or "")
        out.append({
            "turn_index": idx,
            "mode": entry.get("mode"),
            "elapsed_sec": entry.get("elapsed_sec"),
            "session_id_after": entry.get("session_id_after"),
            "assistant_chars": len(assistant_text),
            "assistant_nonempty": bool(assistant_text.strip()),
            "event_count": len(entry.get("events", []) or []),
        })
    return out



def _build_v2_grounding(
    pkt: "V4Packet",
    anchor_records: list,
    answer_support_ids: list,
    answer_overrides_graph: bool,
    label_status: str,
) -> Dict[str, Any]:
    """Normalize the trace into the V5 v2 ``(task + brief -> solution + support)`` shape.

    Additive view (does not touch existing fields). Re-projects the already-computed
    micro_steps + nodes_accessed_log + answer_support_ids into the form v2 training
    consumes: a per-subtask ledger, the assembled BRIEF (everything retrieved), and
    the SUPPORT subset the solution actually used (override-aware). During the STEM
    grounding phase a graph NODE is the artifact stand-in (``artifact_kind``); when the
    grower re-points at code these become api_card/symbol ids with the same shape.
    See V5_V2_DESIGN.md §6/§9.
    """
    nlog = list(getattr(pkt, "nodes_accessed_log", []) or [])
    retrieved_ids = list(dict.fromkeys(
        e.get("node_id") for e in nlog if isinstance(e, dict) and e.get("node_id")))
    by_reason: Dict[str, list] = {}
    for e in nlog:
        if isinstance(e, dict) and e.get("node_id"):
            by_reason.setdefault(str(e.get("reason", "?")), []).append(e["node_id"])

    subtasks = []
    for ms in (getattr(pkt, "micro_steps", []) or []):
        m = ms if isinstance(ms, dict) else _serialize_obj(ms)
        subtasks.append({
            "index": m.get("index"),
            "goal": m.get("subgoal"),
            "signature": m.get("subgoal_signature"),
            "action": m.get("action"),
            "sufficient": m.get("sufficient"),
            "filled_slots": m.get("filled_slots", []),
        })

    return {
        "task": getattr(pkt, "question", ""),
        # STEM phase: a graph node stands in for a code artifact. Re-pointing the
        # grower at code swaps this to api_card/symbol/exemplar, same schema.
        "artifact_kind": "graph_node",
        "brief": {
            "anchor_ids": [a["id"] for a in anchor_records],
            "retrieved_ids": retrieved_ids,
            "retrieved_by_reason": by_reason,
        },
        "support_ids": list(answer_support_ids),
        "overrides_graph": bool(answer_overrides_graph),
        "subtasks": subtasks,
        "solution": getattr(pkt, "answer", "") or getattr(pkt, "answer_raw", ""),
        "verifier": {
            "finalized": bool(getattr(pkt, "finalized", False)),
            "coverage_pct": getattr(pkt, "coverage_addressed_pct", None),
            "label_status": label_status,
        },
    }


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

    V2 adds: v5_trajectory block with nodes_accessed_log, node_type_breakdown,
    and trajectory_reward_score. These fields are None-safe: if graph is
    unavailable or nodes_accessed_log is empty, fields are written as empty.
    """
    finalization_quality = _serialize_obj(getattr(pkt, "finalization_quality", {}) or {})
    training_eligible = bool(
        finalization_quality.get("training_eligible", getattr(pkt, "finalized", False))
    )
    controller_raw_trace = _serialize_obj(getattr(pkt, "controller_raw_trace", []) or [])

    # Resolve anchor snippets so the row is self-contained (you can train on
    # it without re-loading the original graph).
    anchor_records = []
    for aid in pkt.anchors:
        n = graph.nodes.get(aid) if hasattr(graph, "nodes") else None
        snippet = n.text if n else ""
        ntype = getattr(n, "node_type", "") if n else ""
        anchor_records.append({"id": aid, "node_type": ntype, "text": snippet[:600]})

    # ── answer-grounded SUPPORT label ───────────────────────────────────────
    # The nodes the ANSWER actually rests on (not the full tool-search trajectory):
    #   - finalize path: shortcut_anchor_ids (the evidence the controller read)
    #   - loop path:     anchors whose id appears in the answer/explanation text
    # This is the clean support/evidence target for V5 (fixes the trajectory-vs-
    # citation label noise). De-noises support-pointer + epistemic supervision.
    _answer_text = " ".join(str(t) for t in (
        getattr(pkt, "answer_raw", ""), getattr(pkt, "answer", ""),
        getattr(pkt, "explanation", "")) if t)
    answer_support_ids = list(dict.fromkeys(getattr(pkt, "shortcut_anchor_ids", []) or []))
    for aid in pkt.anchors:
        if aid not in answer_support_ids and aid in _answer_text:
            answer_support_ids.append(aid)
    # Loop-FINALIZED fallback: finalize-shortcut traces carry clean
    # shortcut_anchor_ids, but loop-finalized answers cite by CONTENT (no node-id
    # string), so the scan above is empty. Use the nodes actually read_node'd as
    # the evidence base. Gated on finalized so non-finalized/weak-evidence traces
    # correctly keep empty support.
    if getattr(pkt, "finalized", False) and not answer_support_ids:
        for c in (getattr(pkt, "tool_log", []) or []):
            if isinstance(c, dict) and c.get("name") == "read_node":
                nid = (c.get("args") or {}).get("node_id")
                if nid and nid not in answer_support_ids:
                    answer_support_ids.append(nid)

    # ── override detection: answer ignored the graph ────────────────────────
    # If a FINALIZED answer reuses (almost) no content from its support nodes,
    # the model answered from its own parametric knowledge and OVERRODE the
    # graph (retrieval missed / the read node was a category mismatch). Recording
    # those nodes as "support" is a FALSE grounding label that poisons the V5
    # epistemic/support heads. Detect via lexical overlap of the polished answer
    # vs the support-node content; on override drop the support label and mark
    # the row UNSUPPORTED — still useful: it's a clean negative that calibrates
    # the fallback gate (epistemic SHOULD fire, fallback SHOULD trigger).
    _OVERRIDE_OVERLAP_THRESHOLD = 0.08
    answer_overrides_graph = False
    if answer_support_ids and getattr(pkt, "finalized", False) and hasattr(graph, "nodes"):
        _support_text = " ".join(
            graph.nodes[aid].text
            for aid in answer_support_ids
            if aid in graph.nodes and getattr(graph.nodes[aid], "text", "")
        )
        _final_answer_text = str(getattr(pkt, "answer", "") or getattr(pkt, "answer_raw", "") or "")
        if _support_text and _final_answer_text:
            _grounding_overlap = lexical_overlap(_final_answer_text, _support_text)
            if _grounding_overlap < _OVERRIDE_OVERLAP_THRESHOLD:
                answer_overrides_graph = True
                answer_support_ids = []   # false grounding -> no support label

    v5_label_status = (
        "unsupported" if answer_overrides_graph
        else "positive" if training_eligible
        else "needs_review"
    )

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
            "turn_summaries": _turn_summaries(list(pkt.cot_log)),
            "controller_raw_trace": controller_raw_trace,
            "controller_raw_trace_summary": _controller_raw_trace_summary(
                controller_raw_trace if isinstance(controller_raw_trace, list) else []
            ),
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
            # answer-grounded support: the nodes the answer rests on (for V5
            # support/evidence targets — see projection).
            "answer_support_ids": answer_support_ids,
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
            "controller_call_count": getattr(pkt, "controller_call_count", 0),
            "controller_total_elapsed_sec": getattr(pkt, "controller_total_elapsed_sec", 0.0),
            "controller_nonempty_turns": getattr(pkt, "controller_nonempty_turns", 0),
            "finalization_quality": finalization_quality,
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
            "training_eligible": training_eligible,
            "v5_label_status": v5_label_status,
            "answer_overrides_graph": answer_overrides_graph,
            "finalization_quality": finalization_quality,
            # Crude "interesting" heuristic: hard tasks tend to produce
            # objects/hypotheses/failures; easy tasks don't.
            "complexity_proxy_score": (
                len(pkt.objects) + len(pkt.hypotheses) + len(pkt.failures)
                + (1 if pkt.plan_tree_summary else 0)
            ),
        },

        # ── V5: cross-attention trajectory data (Schema V2) ──────────────────
        # These fields are the primary supervision signal for training the
        # GNN + LoRA cross-attention adapters in Phase 16.
        "v5_trajectory": {
            # Every node explicitly accessed during this session, in order.
            # Format: [{"node_id": str, "node_type": str, "step": int, "reason": str}]
            # Reasons: "anchor_retrieval", "tool_read", "shortcut",
            #          "neighbor_expand", "subgoal_lookup"
            "nodes_accessed_log": list(getattr(pkt, "nodes_accessed_log", []) or []),

            # Summary: how many nodes of each type were accessed.
            # Derived from nodes_accessed_log for fast filtering.
            # Format: {"strategy": 2, "fact": 5, "failure_pattern": 1, ...}
            "node_type_counts": _build_node_type_counts(
                getattr(pkt, "nodes_accessed_log", []) or []
            ),

            # Trajectory reward score (0.0–1.0). Computed post-hoc using
            # reasoning.scoring.trajectory_reward(). None if not yet scored.
            # Phase 16 training pipeline fills this field in a separate pass.
            "trajectory_reward_score": None,

            # Whether this trajectory has been labelled as a high-quality
            # training example (manually reviewed or auto-approved).
            "approved_for_training": False,

            # Schema version so Phase 16 pipeline can detect V1 vs V2 rows.
            "corpus_schema_version": CORPUS_SCHEMA_VERSION,
        },

        # ── V5 v2: normalized (task + brief -> solution + support) grounding view ──
        # Additive re-projection of the trace into the shape v2 training consumes
        # (per-subtask ledger + assembled brief + override-aware support). Graph
        # node = code-artifact stand-in for now. See V5_V2_DESIGN.md.
        "v2_grounding": _build_v2_grounding(
            pkt, anchor_records, answer_support_ids, answer_overrides_graph, v5_label_status
        ),
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
