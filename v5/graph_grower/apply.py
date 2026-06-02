"""Apply audited graph-growth queues to a grown graph (gated, non-destructive).

Phase B of the V5 graph grower. Consumes the audit lanes (see ``audit.py``) and
applies their ``raw_edit``s to a COPY of the persisted graph, producing a new
``grown_graph.json`` with:

* provenance stamps on every grown node/edge (``auto_grown``, ``batch_id``,
  ``lane``, ``source_session``),
* a health gate (reuses ``apply_edits_offline`` health-before/after; refuses to
  persist a batch that degrades health beyond ``degradation_threshold`` unless
  ``--force``),
* dangling-edge filtering (edges whose endpoints resolve to neither the base
  graph nor a node added in the same batch are skipped — e.g. the session-local
  ``claim_v4_*_h_1`` hypothesis endpoints flagged by the audit).

It NEVER overwrites the V4-generation base graph: ``--out`` must differ from
``--graph`` (guarded). Substrate is the permissive lane (V5-only topology);
persistent is the strict lane (audit already gated it).
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Set, Tuple

from graph_core import MemoryGraph
from reasoning.graph_editor import apply_edits_offline
from reasoning.graph_health import compute_health

from v5.graph_grower.audit import AuditConfig, audit_corpus

# Permissive for substrate (V5-only), strict ops still gated by the audit lane.
DEFAULT_ALLOWED_TIERS = ("soft", "add", "mutate")
DEFAULT_DEGRADATION_THRESHOLD = -0.02
LANE_FILES = ("persistent_promote", "substrate", "review")


def _batch_id() -> str:
    return "grow_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _candidates_for_lanes(report: Mapping[str, Any], lanes: Sequence[str]) -> List[Dict[str, Any]]:
    """Pull the full candidate sets for the selected lanes from an audit report."""
    out: List[Dict[str, Any]] = []
    report_lanes = report.get("lanes") if isinstance(report.get("lanes"), Mapping) else {}
    for lane in lanes:
        lane_report = report_lanes.get(lane) if isinstance(report_lanes.get(lane), Mapping) else {}
        for cand in lane_report.get("all_candidates", []) or []:
            if isinstance(cand, Mapping):
                out.append(dict(cand))
    return out


def _edits_from_candidates(
    candidates: Sequence[Mapping[str, Any]],
    graph: MemoryGraph,
    *,
    batch_id: str,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Build a deduped, provenance-stamped, endpoint-validated edit list.

    Returns (edits, stats). Nodes are ordered before edges so add_edge endpoints
    that are added in this same batch resolve. Edges with unresolved endpoints
    are dropped (counted in stats).
    """
    node_edits: Dict[str, Dict[str, Any]] = {}
    edge_edits: Dict[Tuple[str, str, str], Dict[str, Any]] = {}
    dropped_dangling: List[Dict[str, Any]] = []
    skipped_existing_nodes = 0

    # Pass 1: nodes (so we know the full post-add node id set).
    for cand in candidates:
        raw = cand.get("raw_edit") if isinstance(cand.get("raw_edit"), Mapping) else {}
        if raw.get("op") != "add_node":
            continue
        nid = str(raw.get("node_id") or cand.get("target_id") or "")
        if not nid:
            continue
        if nid in graph.nodes:
            skipped_existing_nodes += 1
            continue
        if nid in node_edits:
            continue
        edit = dict(raw)
        edit["metadata"] = _stamp(edit.get("metadata"), cand, batch_id)
        node_edits[nid] = edit

    added_node_ids: Set[str] = set(node_edits)
    resolvable: Set[str] = set(graph.nodes) | added_node_ids

    # Pass 2: edges (endpoints must resolve to base graph or a node added above).
    for cand in candidates:
        raw = cand.get("raw_edit") if isinstance(cand.get("raw_edit"), Mapping) else {}
        if raw.get("op") != "add_edge":
            continue
        src = str(raw.get("src", "") or "")
        dst = str(raw.get("dst", "") or "")
        rel = str(raw.get("relation", "related") or "related")
        if not src or not dst:
            continue
        missing = [nid for nid in (src, dst) if nid not in resolvable]
        if missing:
            dropped_dangling.append({"patch_id": cand.get("patch_id"), "missing": missing,
                                     "src": src, "dst": dst})
            continue
        key = (src, dst, rel)
        if key in edge_edits:
            continue
        edit = dict(raw)
        edit["metadata"] = _stamp(edit.get("metadata"), cand, batch_id)
        edge_edits[key] = edit

    edits = list(node_edits.values()) + list(edge_edits.values())
    stats = {
        "node_edits": len(node_edits),
        "edge_edits": len(edge_edits),
        "skipped_existing_nodes": skipped_existing_nodes,
        "dropped_dangling_edges": len(dropped_dangling),
        "dropped_dangling_examples": dropped_dangling[:10],
    }
    return edits, stats


def _stamp(metadata: Any, cand: Mapping[str, Any], batch_id: str) -> Dict[str, Any]:
    md = dict(metadata) if isinstance(metadata, Mapping) else {}
    md["auto_grown"] = True
    md["batch_id"] = batch_id
    md["grow_lane"] = cand.get("lane")
    md["grow_source_session"] = cand.get("session_id")
    md["grow_patch_id"] = cand.get("patch_id")
    if cand.get("support_score") is not None:
        md["grow_support_score"] = cand.get("support_score")
    return md


def apply_growth(
    *,
    corpus_path: str | Path,
    graph_path: str | Path,
    out_path: str | Path,
    lanes: Sequence[str] = ("substrate",),
    audit_config: Optional[AuditConfig] = None,
    allowed_tiers: Sequence[str] = DEFAULT_ALLOWED_TIERS,
    degradation_threshold: float = DEFAULT_DEGRADATION_THRESHOLD,
    force: bool = False,
    dry_run: bool = False,
) -> Dict[str, Any]:
    """Audit -> build edits -> health-gated apply -> persist a grown graph.

    Non-destructive: refuses to write onto ``graph_path``. Returns a report dict.
    """
    graph_path = Path(graph_path)
    out_path = Path(out_path)
    if out_path.resolve() == graph_path.resolve():
        raise ValueError(
            f"refusing to overwrite the base graph {graph_path} -- choose a different --out"
        )

    batch_id = _batch_id()
    report = audit_corpus(corpus_path=corpus_path, graph_path=graph_path, config=audit_config)
    candidates = _candidates_for_lanes(report, lanes)

    graph = MemoryGraph.load_json(str(graph_path))
    health_before = compute_health(graph)
    nodes_before = len(graph.nodes)
    edges_before = len(graph.edges)
    edits, edit_stats = _edits_from_candidates(candidates, graph, batch_id=batch_id)

    apply_summary = apply_edits_offline(
        graph,
        edits,
        allowed_tiers=allowed_tiers,
        degradation_threshold=degradation_threshold,
    )
    health_after = compute_health(graph)
    health_delta = round(health_after.health_score - health_before.health_score, 4)
    gate_passed = health_delta >= degradation_threshold

    persisted = False
    backup = None
    if not dry_run and (gate_passed or force):
        out_path.parent.mkdir(parents=True, exist_ok=True)
        if out_path.exists():
            backup = str(out_path.with_suffix(out_path.suffix + f".bak_{batch_id}"))
            MemoryGraph.load_json(str(out_path)).save_json(backup)
        graph.save_json(str(out_path))
        persisted = True

    result = {
        "batch_id": batch_id,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "corpus_path": str(corpus_path),
        "graph_path": str(graph_path),
        "out_path": str(out_path),
        "lanes": list(lanes),
        "candidate_count": len(candidates),
        "edit_stats": edit_stats,
        "apply_summary": apply_summary,
        "health_before": round(health_before.health_score, 4),
        "health_after": round(health_after.health_score, 4),
        "health_delta": health_delta,
        "degradation_threshold": degradation_threshold,
        "gate_passed": gate_passed,
        "forced": bool(force and not gate_passed),
        "dry_run": dry_run,
        "persisted": persisted,
        "backup": backup,
        "graph_before": {"nodes": nodes_before, "edges": edges_before},
        "graph_after": {"nodes": len(graph.nodes), "edges": len(graph.edges)},
    }
    return result


def _write_report(result: Mapping[str, Any], report_path: str | Path) -> str:
    p = Path(report_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return str(p)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Apply audited graph-growth queues to a grown graph (gated).")
    parser.add_argument("--corpus", default="data/distillation_corpus/sessions.jsonl")
    parser.add_argument("--graph", default="graphs/merged_graph.json")
    parser.add_argument("--out", default="graphs/grown_graph.json")
    parser.add_argument("--report", default="artifacts/graph_growth/graph_growth_apply.json")
    parser.add_argument("--lanes", default="substrate",
                        help="comma-separated: substrate,persistent_promote")
    parser.add_argument("--degradation-threshold", type=float, default=DEFAULT_DEGRADATION_THRESHOLD)
    parser.add_argument("--force", action="store_true", help="persist even if the health gate fails")
    parser.add_argument("--dry-run", action="store_true", help="audit + apply in memory, do not write the graph")
    args = parser.parse_args(argv)

    lanes = [s.strip() for s in args.lanes.split(",") if s.strip()]
    result = apply_growth(
        corpus_path=args.corpus,
        graph_path=args.graph,
        out_path=args.out,
        lanes=lanes,
        degradation_threshold=args.degradation_threshold,
        force=args.force,
        dry_run=args.dry_run,
    )
    report_path = _write_report(result, args.report)

    es = result["edit_stats"]
    print("Graph Growth Apply")
    print(f"  batch: {result['batch_id']}  lanes: {','.join(result['lanes'])}")
    print(f"  candidates: {result['candidate_count']}  "
          f"node_edits: {es['node_edits']}  edge_edits: {es['edge_edits']}")
    print(f"  skipped existing nodes: {es['skipped_existing_nodes']}  "
          f"dropped dangling edges: {es['dropped_dangling_edges']}")
    print(f"  graph after: {result['graph_after']['nodes']} nodes / {result['graph_after']['edges']} edges")
    print(f"  health: {result['health_before']} -> {result['health_after']} "
          f"(delta {result['health_delta']}, gate {'PASS' if result['gate_passed'] else 'FAIL'})")
    print(f"  persisted: {result['persisted']}"
          + (f" -> {result['out_path']}" if result['persisted'] else " (not written)"))
    print(f"  report: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
