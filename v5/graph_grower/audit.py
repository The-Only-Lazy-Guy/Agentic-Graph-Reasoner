"""Audit session graph edits into graph-growth queues.

This module is the first, non-mutating slice of the V5 graph grower.  It reads
distillation/session JSONL rows, classifies scoped patches into:

* persistent promotion candidates: strict, long-term graph candidates
* substrate candidates: V5-only planning/evidence substrate candidates
* review candidates: unsafe, ambiguous, or row-quality-blocked candidates

It never mutates ``graphs/merged_graph.json``.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

from graph_core import MemoryGraph
from reasoning.graph_health import compute_health


SAFE_STATUSES = frozenset({"accept", "soft_only"})
REVIEW_STATUSES = frozenset({"needs_review", "reject"})

SUBSTRATE_PATCH_TYPE_TO_NODE = {
    "add_strategy": "strategy",
    "add_failure_pattern": "failure_pattern",
    "add_control_rule": "control_rule",
    "add_reasoning_atom": "reasoning_atom",
    "add_reasoning_chain": "reasoning_chain",
    "add_solved_subgoal": "solved_subgoal",
    "add_epistemic_state": "epistemic_state",
}
SUBSTRATE_PATCH_TYPES = frozenset(SUBSTRATE_PATCH_TYPE_TO_NODE) | frozenset({"add_relation"})

PERSISTENT_PATCH_TYPES = frozenset({
    "add_fact",
    "add_failure_pattern",
    "add_epistemic_state",
    "add_relation",
})

PERSISTENT_BLOCKING_ROW_FLAGS = frozenset({
    "missing_quality_schema",
    "training_not_eligible",
    "weak_label_status",
    "answer_overrides_graph",
    "not_finalized",
    "controller_fallback_used",
    "low_slot_coverage",
    "suspicious_design_slots",
})

SUSPICIOUS_DESIGN_SLOTS = frozenset({
    "rank_query",
    "pagination",
    "tie_policy",
    "latency_budget",
    "consistency_model",
})


@dataclass(frozen=True)
class AuditConfig:
    min_persistent_support: float = 0.12
    min_slot_coverage: float = 0.50
    max_review_ratio: float = 0.25
    allow_old_schema_persistent: bool = False
    max_candidates: int = 50


def load_jsonl(path: str | Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def audit_corpus(
    *,
    corpus_path: str | Path,
    graph_path: str | Path,
    config: Optional[AuditConfig] = None,
) -> Dict[str, Any]:
    rows = load_jsonl(corpus_path)
    graph = MemoryGraph.load_json(graph_path)
    report = audit_rows(rows, graph, config=config)
    report["corpus_path"] = str(corpus_path)
    report["graph_path"] = str(graph_path)
    return report


def audit_rows(
    rows: Sequence[Mapping[str, Any]],
    graph: MemoryGraph,
    *,
    config: Optional[AuditConfig] = None,
) -> Dict[str, Any]:
    cfg = config or AuditConfig()
    health = compute_health(graph, sample_bfs=20)

    patch_counts: Dict[str, Counter] = {
        "by_status": Counter(),
        "by_type": Counter(),
        "by_risk": Counter(),
        "by_task_family": Counter(),
    }
    row_flag_counts: Counter = Counter()
    session_summaries: List[Dict[str, Any]] = []
    patch_records: List[Dict[str, Any]] = []

    for row_index, row in enumerate(rows):
        row_info = _row_info(row, row_index, cfg)
        row_flag_counts.update(row_info["flags"])
        patches = _scoped_patches(row)
        status_counts = Counter()
        type_counts = Counter()
        for patch_index, patch in enumerate(patches):
            if not isinstance(patch, Mapping):
                continue
            record = _patch_record(row_info, patch, patch_index)
            patch_records.append(record)
            status_counts[record["status"]] += 1
            type_counts[record["patch_type"]] += 1
            patch_counts["by_status"][record["status"]] += 1
            patch_counts["by_type"][record["patch_type"]] += 1
            patch_counts["by_risk"][record["risk_level"]] += 1
            patch_counts["by_task_family"][row_info["task_family"]] += 1
        if patches or row_info["flags"]:
            session_summaries.append({
                "row_index": row_index,
                "session_id": row_info["session_id"],
                "task_family": row_info["task_family"],
                "question": row_info["question"],
                "flags": row_info["flags"],
                "slot_coverage": row_info["slot_coverage"],
                "patch_count": len(patches),
                "patch_statuses": dict(sorted(status_counts.items())),
                "patch_types": dict(sorted(type_counts.items())),
            })

    persistent = _persistent_candidates(patch_records, graph, cfg)
    substrate = _substrate_candidates(patch_records, graph)
    review = _review_candidates(patch_records, cfg)

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "config": asdict(cfg),
        "graph": {
            "node_count": len(graph.nodes),
            "edge_count": len(graph.edges),
            "health": health.to_dict(),
            "health_score": round(health.health_score, 4),
        },
        "corpus": {
            "row_count": len(rows),
            "rows_with_patches": sum(1 for row in rows if _scoped_patches(row)),
            "row_flags": dict(sorted(row_flag_counts.items())),
        },
        "patch_counts": {
            name: dict(sorted(counter.items()))
            for name, counter in patch_counts.items()
        },
        "lanes": {
            "persistent_promote": _lane_report(persistent, graph, cfg.max_candidates),
            "substrate": _lane_report(substrate, graph, cfg.max_candidates),
            "review": _lane_report(review, graph, cfg.max_candidates),
        },
        "sessions": session_summaries[: cfg.max_candidates],
    }


def write_report(report: Mapping[str, Any], out_path: str | Path, *, write_queues: bool = True) -> Dict[str, str]:
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(_report_without_full_queues(report), ensure_ascii=False, indent=2), encoding="utf-8")
    written = {"report": str(out)}
    if not write_queues:
        return written

    lanes = report.get("lanes") if isinstance(report.get("lanes"), Mapping) else {}
    for lane_name, file_name in (
        ("persistent_promote", "promotion_queue.jsonl"),
        ("substrate", "substrate_queue.jsonl"),
        ("review", "review_queue.jsonl"),
    ):
        lane = lanes.get(lane_name) if isinstance(lanes.get(lane_name), Mapping) else {}
        queue_path = out.parent / file_name
        with queue_path.open("w", encoding="utf-8") as handle:
            for candidate in lane.get("all_candidates", []) or []:
                handle.write(json.dumps(candidate, ensure_ascii=False) + "\n")
        written[lane_name] = str(queue_path)
    return written


def _report_without_full_queues(report: Mapping[str, Any]) -> Dict[str, Any]:
    """Keep reports compact while queue JSONL files carry the full candidate sets."""
    clean = dict(report)
    lanes = clean.get("lanes") if isinstance(clean.get("lanes"), Mapping) else {}
    clean_lanes: Dict[str, Any] = {}
    for lane_name, lane in lanes.items():
        if not isinstance(lane, Mapping):
            clean_lanes[lane_name] = lane
            continue
        lane_clean = dict(lane)
        lane_clean.pop("all_candidates", None)
        clean_lanes[lane_name] = lane_clean
    clean["lanes"] = clean_lanes
    return clean


def _row_info(row: Mapping[str, Any], row_index: int, cfg: AuditConfig) -> Dict[str, Any]:
    quality = row.get("quality") if isinstance(row.get("quality"), Mapping) else {}
    metrics = row.get("metrics") if isinstance(row.get("metrics"), Mapping) else {}
    inp = row.get("input") if isinstance(row.get("input"), Mapping) else {}
    slot_stats = metrics.get("slot_fill_stats") if isinstance(metrics.get("slot_fill_stats"), Mapping) else {}
    required_slots = [str(s) for s in slot_stats.get("required_slots", []) or []]
    filled_count = _safe_int(slot_stats.get("filled_count"))
    required_count = _safe_int(slot_stats.get("required_count"))
    slot_coverage = (filled_count / required_count) if required_count > 0 else None
    task_family = str(
        metrics.get("controller_task_family")
        or _mapping_get(inp.get("task_frame"), "task_family")
        or _mapping_get(inp.get("task_frame"), "family")
        or "unknown"
    )
    flags: List[str] = []

    has_training_eligible = "training_eligible" in quality
    has_label_status = "v5_label_status" in quality
    if not (has_training_eligible and has_label_status) and not cfg.allow_old_schema_persistent:
        flags.append("missing_quality_schema")
    if has_training_eligible and not bool(quality.get("training_eligible")):
        flags.append("training_not_eligible")
    label_status = str(quality.get("v5_label_status", "") or "")
    if has_label_status and label_status != "positive":
        flags.append("weak_label_status")
    if bool(quality.get("answer_overrides_graph", False)):
        flags.append("answer_overrides_graph")
    if not bool(metrics.get("finalized", quality.get("finalized", False))):
        flags.append("not_finalized")
    if bool(metrics.get("controller_fallback_used", False)):
        flags.append("controller_fallback_used")
    if slot_coverage is not None and slot_coverage < cfg.min_slot_coverage:
        flags.append("low_slot_coverage")
    if task_family == "design_synthesis" and SUSPICIOUS_DESIGN_SLOTS.intersection(required_slots):
        flags.append("suspicious_design_slots")
    patch_summary = metrics.get("scoped_patch_summary")
    if isinstance(patch_summary, Mapping):
        patch_count = _safe_int(patch_summary.get("patch_count") or patch_summary.get("total"))
        needs_attention = _safe_int(
            patch_summary.get("needs_attention")
            or patch_summary.get("needs_review")
            or patch_summary.get("review")
        )
        if patch_count > 0 and needs_attention / patch_count > cfg.max_review_ratio:
            flags.append("high_review_ratio")

    return {
        "row_index": row_index,
        "session_id": _session_id(row),
        "question": str(inp.get("question", "") or row.get("question", "") or ""),
        "task_family": task_family,
        "flags": sorted(set(flags)),
        "persistent_blocked": bool(PERSISTENT_BLOCKING_ROW_FLAGS.intersection(flags)),
        "slot_coverage": None if slot_coverage is None else round(slot_coverage, 3),
    }


def _patch_record(row_info: Mapping[str, Any], patch: Mapping[str, Any], patch_index: int) -> Dict[str, Any]:
    validation = patch.get("validation") if isinstance(patch.get("validation"), Mapping) else {}
    raw_edit = patch.get("raw_edit") if isinstance(patch.get("raw_edit"), Mapping) else {}
    return {
        "row_index": row_info["row_index"],
        "patch_index": patch_index,
        "session_id": row_info["session_id"],
        "question": row_info["question"],
        "task_family": row_info["task_family"],
        "row_flags": list(row_info["flags"]),
        "row_persistent_blocked": bool(row_info["persistent_blocked"]),
        "slot_coverage": row_info["slot_coverage"],
        "patch_id": str(patch.get("patch_id", f"patch_{patch_index:04d}")),
        "patch_type": str(patch.get("patch_type", "unknown")),
        "target_id": str(patch.get("target_id") or raw_edit.get("node_id") or ""),
        "text": str(patch.get("text", "") or raw_edit.get("text", "")),
        "status": str(validation.get("status", "unknown")),
        "risk_level": str(patch.get("risk_level", "medium")),
        "support_score": _safe_float(validation.get("support_score")),
        "duplicate_of": validation.get("duplicate_of"),
        "conflicts_with": list(validation.get("conflicts_with", []) or []),
        "validation_reasons": list(validation.get("reasons", []) or []),
        "validation_warnings": list(validation.get("warnings", []) or []),
        "evidence_node_ids": list(patch.get("evidence_node_ids", []) or []),
        "valid_when": list(patch.get("valid_when", []) or []),
        "invalid_when": list(patch.get("invalid_when", []) or []),
        "raw_edit": dict(raw_edit),
    }


def _persistent_candidates(
    records: Sequence[Mapping[str, Any]],
    graph: MemoryGraph,
    cfg: AuditConfig,
) -> List[Dict[str, Any]]:
    proposed_node_ids: Set[str] = set()
    preliminary: List[Tuple[Mapping[str, Any], List[str]]] = []
    for record in records:
        reasons = _persistent_reject_reasons(record, graph, cfg, set())
        if not reasons:
            preliminary.append((record, reasons))
            raw = record.get("raw_edit") if isinstance(record.get("raw_edit"), Mapping) else {}
            if raw.get("op") == "add_node":
                node_id = str(raw.get("node_id") or record.get("target_id") or "")
                if node_id:
                    proposed_node_ids.add(node_id)

    out: List[Dict[str, Any]] = []
    for record, _ in preliminary:
        reasons = _persistent_reject_reasons(record, graph, cfg, proposed_node_ids)
        if reasons:
            continue
        out.append(_candidate(record, "persistent_promote", ["strict persistent gate passed"]))
    return out


def _persistent_reject_reasons(
    record: Mapping[str, Any],
    graph: MemoryGraph,
    cfg: AuditConfig,
    proposed_node_ids: Set[str],
) -> List[str]:
    reasons: List[str] = []
    patch_type = str(record.get("patch_type", ""))
    raw = record.get("raw_edit") if isinstance(record.get("raw_edit"), Mapping) else {}
    status = str(record.get("status", ""))
    if patch_type not in PERSISTENT_PATCH_TYPES:
        reasons.append(f"patch_type_not_persistent:{patch_type}")
    if status != "accept":
        reasons.append(f"status_not_accept:{status or 'unknown'}")
    if record.get("risk_level") == "high":
        reasons.append("high_risk_patch")
    if _safe_float(record.get("support_score")) < cfg.min_persistent_support:
        reasons.append("support_below_persistent_threshold")
    if record.get("duplicate_of"):
        reasons.append("duplicate_patch")
    if record.get("conflicts_with"):
        reasons.append("conflict_patch")
    for flag in record.get("row_flags", []) or []:
        if flag in PERSISTENT_BLOCKING_ROW_FLAGS:
            reasons.append(f"row_flag:{flag}")
    op = str(raw.get("op", "") or "")
    if patch_type == "add_relation" or op == "add_edge":
        missing = [
            nid for nid in (str(raw.get("src", "") or ""), str(raw.get("dst", "") or ""))
            if nid and nid not in graph.nodes and nid not in proposed_node_ids
        ]
        if missing:
            reasons.append("missing_relation_endpoint:" + ",".join(missing))
    if op == "increment_meta":
        reasons.append("soft_reinforcement_not_persistent_promotion")
    return reasons


def _substrate_candidates(records: Sequence[Mapping[str, Any]], graph: MemoryGraph) -> List[Dict[str, Any]]:
    substrate_node_ids: Set[str] = {
        str((record.get("raw_edit") or {}).get("node_id") or record.get("target_id") or "")
        for record in records
        if isinstance(record.get("raw_edit"), Mapping)
        and record.get("patch_type") in SUBSTRATE_PATCH_TYPE_TO_NODE
        and record.get("status") in SAFE_STATUSES
    }
    out: List[Dict[str, Any]] = []
    for record in records:
        patch_type = str(record.get("patch_type", ""))
        status = str(record.get("status", ""))
        if status not in SAFE_STATUSES or patch_type not in SUBSTRATE_PATCH_TYPES:
            continue
        raw = record.get("raw_edit") if isinstance(record.get("raw_edit"), Mapping) else {}
        reasons = ["safe V5 substrate patch"]
        warnings: List[str] = []
        if record.get("row_flags"):
            warnings.append("row_flags:" + ",".join(record.get("row_flags", []) or []))
        if patch_type == "add_relation" or raw.get("op") == "add_edge":
            missing = [
                nid for nid in (str(raw.get("src", "") or ""), str(raw.get("dst", "") or ""))
                if nid and nid not in graph.nodes and nid not in substrate_node_ids
            ]
            if missing:
                warnings.append("substrate_relation_missing_endpoint:" + ",".join(missing))
        out.append(_candidate(record, "substrate", reasons, warnings=warnings))
    return out


def _review_candidates(records: Sequence[Mapping[str, Any]], cfg: AuditConfig) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    empty_graph = MemoryGraph(nodes={}, edges=[])
    for record in records:
        reasons: List[str] = []
        status = str(record.get("status", ""))
        if status in REVIEW_STATUSES or status not in SAFE_STATUSES:
            reasons.append(f"validation_status:{status or 'unknown'}")
        persistent_reasons = _persistent_reject_reasons(record, empty_graph, cfg, set())
        row_block_reasons = [r for r in persistent_reasons if r.startswith("row_flag:")]
        if row_block_reasons and record.get("patch_type") in PERSISTENT_PATCH_TYPES:
            reasons.extend(row_block_reasons)
        if record.get("validation_warnings"):
            reasons.append("validation_warnings_present")
        if reasons:
            out.append(_candidate(record, "review", _unique(reasons)))
    return out


def _candidate(
    record: Mapping[str, Any],
    lane: str,
    reasons: Sequence[str],
    *,
    warnings: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    raw = record.get("raw_edit") if isinstance(record.get("raw_edit"), Mapping) else {}
    return {
        "lane": lane,
        "row_index": record.get("row_index"),
        "patch_index": record.get("patch_index"),
        "session_id": record.get("session_id"),
        "task_family": record.get("task_family"),
        "question": record.get("question"),
        "row_flags": list(record.get("row_flags", []) or []),
        "slot_coverage": record.get("slot_coverage"),
        "patch_id": record.get("patch_id"),
        "patch_type": record.get("patch_type"),
        "target_id": record.get("target_id"),
        "status": record.get("status"),
        "risk_level": record.get("risk_level"),
        "support_score": record.get("support_score"),
        "evidence_node_ids": list(record.get("evidence_node_ids", []) or []),
        "reasons": list(reasons),
        "warnings": list(warnings or []),
        "validation_reasons": list(record.get("validation_reasons", []) or []),
        "validation_warnings": list(record.get("validation_warnings", []) or []),
        "text": record.get("text"),
        "raw_edit": dict(raw),
    }


def _lane_report(
    candidates: Sequence[Mapping[str, Any]],
    graph: MemoryGraph,
    max_candidates: int,
) -> Dict[str, Any]:
    by_type = Counter(str(c.get("patch_type", "unknown")) for c in candidates)
    by_status = Counter(str(c.get("status", "unknown")) for c in candidates)
    by_family = Counter(str(c.get("task_family", "unknown")) for c in candidates)
    return {
        "count": len(candidates),
        "by_type": dict(sorted(by_type.items())),
        "by_status": dict(sorted(by_status.items())),
        "by_task_family": dict(sorted(by_family.items())),
        "delta_estimate": _delta_estimate(candidates, graph),
        "candidates": list(candidates[:max_candidates]),
        "all_candidates": list(candidates),
    }


def _delta_estimate(candidates: Sequence[Mapping[str, Any]], graph: MemoryGraph) -> Dict[str, Any]:
    existing_edges = {(edge.src, edge.dst, edge.relation) for edge in graph.edges}
    proposed_nodes: Set[str] = set()
    proposed_edges: Set[Tuple[str, str, str]] = set()
    unresolved_edges: List[Dict[str, Any]] = []

    for candidate in candidates:
        raw = candidate.get("raw_edit") if isinstance(candidate.get("raw_edit"), Mapping) else {}
        if raw.get("op") == "add_node":
            node_id = str(raw.get("node_id") or candidate.get("target_id") or "")
            if node_id and node_id not in graph.nodes:
                proposed_nodes.add(node_id)

    for candidate in candidates:
        raw = candidate.get("raw_edit") if isinstance(candidate.get("raw_edit"), Mapping) else {}
        if raw.get("op") != "add_edge":
            continue
        src = str(raw.get("src", "") or "")
        dst = str(raw.get("dst", "") or "")
        rel = str(raw.get("relation", "related") or "related")
        if not src or not dst:
            continue
        missing = [nid for nid in (src, dst) if nid not in graph.nodes and nid not in proposed_nodes]
        if missing:
            unresolved_edges.append({
                "patch_id": candidate.get("patch_id"),
                "missing": missing,
                "src": src,
                "dst": dst,
            })
            continue
        key = (src, dst, rel)
        if key not in existing_edges:
            proposed_edges.add(key)

    return {
        "new_nodes": len(proposed_nodes),
        "new_edges": len(proposed_edges),
        "unresolved_edges": len(unresolved_edges),
        "unresolved_edge_examples": unresolved_edges[:10],
    }


def _scoped_patches(row: Mapping[str, Any]) -> List[Any]:
    trace = row.get("trace") if isinstance(row.get("trace"), Mapping) else {}
    patches = trace.get("scoped_patches")
    if patches is None:
        patches = row.get("scoped_patches")
    return list(patches or [])


def _session_id(row: Mapping[str, Any]) -> str:
    raw = str(row.get("session_id", "") or "")
    if "\\" in raw or "/" in raw:
        return Path(raw).name
    return raw or "unknown_session"


def _mapping_get(value: Any, key: str) -> Any:
    return value.get(key) if isinstance(value, Mapping) else None


def _safe_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _safe_int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _unique(values: Iterable[str]) -> List[str]:
    seen: Set[str] = set()
    out: List[str] = []
    for value in values:
        if value not in seen:
            seen.add(value)
            out.append(value)
    return out


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Audit V4 session graph edits into V5 graph-growth queues.")
    parser.add_argument("--corpus", default="data/distillation_corpus/sessions.jsonl")
    parser.add_argument("--graph", default="graphs/merged_graph.json")
    parser.add_argument("--out", default="artifacts/graph_growth/graph_growth_audit.json")
    parser.add_argument("--min-persistent-support", type=float, default=AuditConfig.min_persistent_support)
    parser.add_argument("--min-slot-coverage", type=float, default=AuditConfig.min_slot_coverage)
    parser.add_argument("--max-review-ratio", type=float, default=AuditConfig.max_review_ratio)
    parser.add_argument("--max-candidates", type=int, default=AuditConfig.max_candidates)
    parser.add_argument("--allow-old-schema-persistent", action="store_true")
    parser.add_argument("--no-queues", action="store_true")
    args = parser.parse_args(argv)

    cfg = AuditConfig(
        min_persistent_support=args.min_persistent_support,
        min_slot_coverage=args.min_slot_coverage,
        max_review_ratio=args.max_review_ratio,
        allow_old_schema_persistent=args.allow_old_schema_persistent,
        max_candidates=args.max_candidates,
    )
    report = audit_corpus(corpus_path=args.corpus, graph_path=args.graph, config=cfg)
    written = write_report(report, args.out, write_queues=not args.no_queues)

    lanes = report["lanes"]
    print("Graph Growth Audit")
    print(f"  rows: {report['corpus']['row_count']}  rows_with_patches: {report['corpus']['rows_with_patches']}")
    print(f"  graph: {report['graph']['node_count']} nodes / {report['graph']['edge_count']} edges")
    print(f"  persistent: {lanes['persistent_promote']['count']} candidates")
    print(f"  substrate:   {lanes['substrate']['count']} candidates")
    print(f"  review:      {lanes['review']['count']} candidates")
    print(f"  report: {written['report']}")
    if not args.no_queues:
        print(f"  queues: {written['persistent_promote']}, {written['substrate']}, {written['review']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
