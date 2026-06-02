"""Project V4 corpus rows into V5-native supervision targets.

The raw distillation corpus captures V4 behavior in a tool-using form. V5 does
not want to imitate the textual tool-call interface directly; it wants training
targets aligned to its architecture:

  question -> candidate subgraph -> planning loop -> evidence loop -> answer

This module turns one corpus row into architecture-shaped supervision:

  - outer retrieval target     : weak teacher for candidate-subgraph selection
  - planning target            : soft node weights for Layer-8 planning
  - evidence target            : soft node weights for Layer-20 evidence
  - support target             : nodes the final answer appears to rest on
  - distractor target          : retrieved-but-unused / contradicted nodes
  - per-loop targets           : best-effort loop-shaped supervision

The projection is intentionally heuristic. It prefers "which graph objects V4
actually relied on" over "which tools V4 called." It uses:

  - anchors / shortcut anchors
  - micro_steps.evidence_node_ids
  - read_node tool calls
  - safe scoped patches (strategy / solved_subgoal / epistemic / ...)
  - nodes_accessed_log loop entries, when they contain usable top_nodes

Rows remain fully backward-compatible: the caller can attach the resulting dict
under row["v5_projection"] and keep the original corpus untouched otherwise.
"""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path
import json
from typing import Any, DefaultDict, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

SAFE_PATCH_STATUSES = frozenset({"accept", "soft_only"})

PLANNING_PATCH_WEIGHTS: Dict[str, float] = {
    "add_strategy": 3.0,
    "add_failure_pattern": 2.5,
    "add_control_rule": 2.0,
    "add_reasoning_chain": 1.5,
    "add_reasoning_atom": 1.5,
}
EVIDENCE_PATCH_WEIGHTS: Dict[str, float] = {
    "add_solved_subgoal": 2.5,
}
ACTION_EVIDENCE_WEIGHTS: Dict[str, float] = {
    "REUSE": 2.5,
    "QUERY": 2.0,
    "DERIVE": 1.5,
    "VERIFY": 2.0,
    "FINALIZE": 3.0,
}
ACTION_SUPPORT_WEIGHTS: Dict[str, float] = {
    "REUSE": 2.0,
    "QUERY": 1.5,
    "DERIVE": 1.0,
    "VERIFY": 1.5,
    "FINALIZE": 2.5,
}
TOOL_REASON_WEIGHTS: Dict[str, float] = {
    "anchor_retrieval": 1.0,
    "neighbor_expand": 0.5,
    "tool_read": 1.5,
    "shortcut": 2.0,
    "subgoal_lookup": 2.0,
}
LOOP_LAYER_PLANNING = 8
LOOP_LAYER_EVIDENCE = 20
PROJECTION_SCHEMA_VERSION = 1


def _add(counter: MutableMapping[str, float], node_id: Optional[str], weight: float) -> None:
    if not node_id or weight <= 0:
        return
    counter[str(node_id)] = float(counter.get(str(node_id), 0.0)) + float(weight)


def _add_many(counter: MutableMapping[str, float], node_ids: Iterable[str], weight: float) -> None:
    for node_id in node_ids:
        _add(counter, node_id, weight)


def _ordered_unique(node_ids: Iterable[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for node_id in node_ids:
        nid = str(node_id)
        if not nid or nid in seen:
            continue
        seen.add(nid)
        out.append(nid)
    return out


def _sorted_weight_map(counter: Mapping[str, float], *, ndigits: int = 6) -> Dict[str, float]:
    items = [
        (node_id, round(float(weight), ndigits))
        for node_id, weight in counter.items()
        if float(weight) > 0.0
    ]
    items.sort(key=lambda item: (-item[1], item[0]))
    return dict(items)


def _candidate_order(
    anchors: Sequence[str],
    support: Mapping[str, float],
    evidence: Mapping[str, float],
    planning: Mapping[str, float],
    distractor: Mapping[str, float],
) -> List[str]:
    ranked_tail: List[str] = []
    for source in (support, evidence, planning, distractor):
        ranked_tail.extend(
            node_id for node_id, _ in sorted(source.items(), key=lambda item: (-item[1], item[0]))
        )
    return _ordered_unique(list(anchors) + ranked_tail)


def _safe_patch_status(patch: Mapping[str, Any]) -> bool:
    return ((patch.get("validation") or {}).get("status") in SAFE_PATCH_STATUSES)


def _node_ids_from_patch(patch: Mapping[str, Any]) -> List[str]:
    raw = patch.get("raw_edit") or {}
    meta = patch.get("metadata") or {}
    payload_meta = ((patch.get("payload") or {}).get("metadata") or {})

    node_ids: List[str] = []
    for key in ("target_id",):
        if patch.get(key):
            node_ids.append(str(patch[key]))
    for key in ("node_id", "target_node_id", "src", "dst"):
        if raw.get(key):
            node_ids.append(str(raw[key]))
        if meta.get(key):
            node_ids.append(str(meta[key]))
        if payload_meta.get(key):
            node_ids.append(str(payload_meta[key]))
    for key in ("evidence_node_ids", "affected_node_ids", "supporting_node_ids", "last_verified_by", "key_node_ids"):
        for source in (patch, meta, payload_meta, raw):
            for node_id in (source.get(key) or []):
                node_ids.append(str(node_id))
    return _ordered_unique(node_ids)


def _epistemic_status(patch: Mapping[str, Any]) -> str:
    raw = patch.get("raw_edit") or {}
    meta = patch.get("metadata") or {}
    payload_meta = ((patch.get("payload") or {}).get("metadata") or {})
    for source in (raw.get("metadata") or {}, meta, payload_meta):
        status = source.get("status")
        if isinstance(status, str) and status:
            return status
    return "unknown"


def _target_node_from_epistemic_patch(patch: Mapping[str, Any]) -> Optional[str]:
    raw = patch.get("raw_edit") or {}
    meta = patch.get("metadata") or {}
    payload_meta = ((patch.get("payload") or {}).get("metadata") or {})
    for source in (raw.get("metadata") or {}, meta, payload_meta, raw):
        target = source.get("target_node_id")
        if isinstance(target, str) and target:
            return target
    return None


def _loop_target_from_top_nodes(top_nodes: Sequence[Any]) -> Dict[str, float]:
    weights: DefaultDict[str, float] = defaultdict(float)
    for item in top_nodes:
        if isinstance(item, (list, tuple)) and len(item) >= 2:
            node_id = str(item[0])
            try:
                weight = float(item[1])
            except (TypeError, ValueError):
                weight = 0.0
            _add(weights, node_id, weight)
    return _sorted_weight_map(weights)


def _micro_step_loop_targets(micro_steps: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    loops: List[Dict[str, Any]] = []
    for idx, step in enumerate(micro_steps):
        evidence_node_ids = [
            str(node_id)
            for node_id in (step.get("evidence_node_ids") or [])
            if isinstance(node_id, str) and node_id
        ]
        if not evidence_node_ids:
            continue
        target = {node_id: 1.0 for node_id in _ordered_unique(evidence_node_ids)}
        loops.append({
            "loop": idx,
            "source": f"micro_step:{step.get('action') or 'unknown'}",
            "target": target,
        })
    return loops


def project_corpus_row(row: Mapping[str, Any]) -> Dict[str, Any]:
    """Return a V5-native projection dict for one corpus row."""
    inp = row.get("input", {}) or {}
    trace = row.get("trace", {}) or {}
    metrics = row.get("metrics", {}) or {}
    outputs = row.get("outputs", {}) or {}
    v5_traj = row.get("v5_trajectory", {}) or {}

    anchors = inp.get("anchors") or []
    anchor_ids = _ordered_unique(
        a.get("id")
        for a in anchors
        if isinstance(a, Mapping) and isinstance(a.get("id"), str)
    )
    shortcut_anchor_ids = _ordered_unique(metrics.get("shortcut_anchor_ids") or [])
    tool_calls = trace.get("tool_calls") or []
    micro_steps = [ms for ms in (trace.get("micro_steps") or []) if isinstance(ms, Mapping)]
    patches = [p for p in (trace.get("scoped_patches") or []) if isinstance(p, Mapping)]
    nodes_accessed_log = [e for e in (v5_traj.get("nodes_accessed_log") or []) if isinstance(e, Mapping)]

    outer: DefaultDict[str, float] = defaultdict(float)
    planning: DefaultDict[str, float] = defaultdict(float)
    evidence: DefaultDict[str, float] = defaultdict(float)
    support: DefaultDict[str, float] = defaultdict(float)
    distractor: DefaultDict[str, float] = defaultdict(float)

    for node_id in anchor_ids:
        _add(outer, node_id, 1.0)
    _add_many(outer, shortcut_anchor_ids, 1.0)
    _add_many(evidence, shortcut_anchor_ids, 2.0)
    _add_many(support, shortcut_anchor_ids, 3.0)

    # Answer-grounded support: nodes the FINAL answer rests on (V4-side label,
    # outputs.answer_support_ids). Highest-priority support/evidence signal —
    # dominates trajectory-derived weights so V5 support/epistemic targets match
    # what the answer actually cited.
    answer_support_ids = _ordered_unique(outputs.get("answer_support_ids") or [])
    _add_many(outer, answer_support_ids, 1.5)
    _add_many(evidence, answer_support_ids, 4.0)
    _add_many(support, answer_support_ids, 5.0)

    explicit_read_nodes: List[str] = []
    for tool_call in tool_calls:
        name = str(tool_call.get("name") or "")
        args = tool_call.get("args") or {}
        node_id = args.get("node_id") if isinstance(args, Mapping) else None
        if name == "read_node" and isinstance(node_id, str):
            explicit_read_nodes.append(node_id)
            _add(outer, node_id, 1.25)
            _add(evidence, node_id, 1.5)
        elif name == "expand_neighbors" and isinstance(node_id, str):
            _add(outer, node_id, 0.75)
        elif name == "list_anchors" and isinstance(node_id, str):
            _add(outer, node_id, 0.5)

    last_support_nodes: List[str] = []
    for step_index, step in enumerate(micro_steps):
        action = str(step.get("action") or "unknown")
        evidence_node_ids = _ordered_unique(step.get("evidence_node_ids") or [])
        matched_node_id = step.get("matched_node_id")
        if evidence_node_ids:
            step_boost = 1.0 + (step_index / max(1, len(micro_steps)))
            _add_many(evidence, evidence_node_ids, ACTION_EVIDENCE_WEIGHTS.get(action, 1.0) * step_boost)
            _add_many(outer, evidence_node_ids, 0.5)
            last_support_nodes = evidence_node_ids
            if action in ACTION_SUPPORT_WEIGHTS:
                _add_many(support, evidence_node_ids, ACTION_SUPPORT_WEIGHTS[action] * step_boost)
        if isinstance(matched_node_id, str) and matched_node_id:
            _add(evidence, matched_node_id, ACTION_EVIDENCE_WEIGHTS.get(action, 1.0))
            _add(outer, matched_node_id, 0.5)

    if last_support_nodes:
        _add_many(support, last_support_nodes, 1.5)

    planning_loops: List[Dict[str, Any]] = []
    evidence_loops: List[Dict[str, Any]] = []
    for entry in nodes_accessed_log:
        if "reason" in entry and isinstance(entry.get("node_id"), str):
            node_id = str(entry["node_id"])
            reason = str(entry.get("reason") or "")
            weight = TOOL_REASON_WEIGHTS.get(reason, 0.75)
            _add(outer, node_id, weight)
            if reason in {"tool_read", "shortcut", "subgoal_lookup"}:
                _add(evidence, node_id, weight)
            if reason == "shortcut":
                _add(support, node_id, weight + 0.5)
        top_nodes = entry.get("top_nodes") or []
        if "layer" in entry and top_nodes:
            loop_target = _loop_target_from_top_nodes(top_nodes)
            if not loop_target:
                continue
            layer = int(entry.get("layer") or 0)
            loop_payload = {
                "loop": int(entry.get("loop") or 0),
                "source": "nodes_accessed_log",
                "target": loop_target,
            }
            if layer == LOOP_LAYER_PLANNING:
                planning_loops.append(loop_payload)
                for node_id, weight in loop_target.items():
                    _add(planning, node_id, weight)
            elif layer == LOOP_LAYER_EVIDENCE:
                evidence_loops.append(loop_payload)
                for node_id, weight in loop_target.items():
                    _add(evidence, node_id, weight)

    for patch in patches:
        patch_type = str(patch.get("patch_type") or "")
        safe = _safe_patch_status(patch)
        node_ids = _node_ids_from_patch(patch)

        if safe and patch_type in PLANNING_PATCH_WEIGHTS:
            target_id = (patch.get("raw_edit") or {}).get("node_id") or patch.get("target_id")
            if isinstance(target_id, str):
                _add(planning, target_id, PLANNING_PATCH_WEIGHTS[patch_type])

        if safe and patch_type in EVIDENCE_PATCH_WEIGHTS:
            target_id = (patch.get("raw_edit") or {}).get("node_id") or patch.get("target_id")
            if isinstance(target_id, str):
                _add(evidence, target_id, EVIDENCE_PATCH_WEIGHTS[patch_type])

        if safe and patch_type == "add_solved_subgoal":
            meta = patch.get("metadata") or {}
            supporting = _ordered_unique(meta.get("supporting_node_ids") or [])
            _add_many(evidence, supporting, 1.75)
            _add_many(support, supporting, 2.0)

        if safe and patch_type == "add_reasoning_atom":
            meta = patch.get("metadata") or {}
            supporting = _ordered_unique(meta.get("supporting_node_ids") or [])
            _add_many(evidence, supporting, 1.25)
            _add_many(support, supporting, 1.5)

        if safe and patch_type == "add_epistemic_state":
            status = _epistemic_status(patch).lower()
            target_node_id = _target_node_from_epistemic_patch(patch)
            evidence_node_ids = _ordered_unique((patch.get("evidence_node_ids") or []))
            if status in {"verified", "supported"}:
                if target_node_id:
                    _add(evidence, target_node_id, 1.75)
                    _add(support, target_node_id, 2.0)
                _add_many(evidence, evidence_node_ids, 1.5)
                _add_many(support, evidence_node_ids, 1.5)
            else:
                target_id = (patch.get("raw_edit") or {}).get("node_id") or patch.get("target_id")
                if isinstance(target_id, str):
                    _add(planning, target_id, 1.0)
                if target_node_id and status in {"contradicted", "mismatched"}:
                    _add(distractor, target_node_id, 1.75)
                    _add_many(distractor, evidence_node_ids, 1.0)

        if safe and patch_type == "add_relation":
            raw = patch.get("raw_edit") or {}
            relation = str(raw.get("relation") or "")
            src = raw.get("src")
            if relation == "invalidated_by" and isinstance(src, str):
                _add(distractor, src, 1.0)

        if safe:
            for node_id in node_ids:
                _add(outer, node_id, 0.1)

    # Override row: the answer ignored the graph (false grounding, flagged by
    # distillation_corpus.answer_overrides_graph). Keep the candidate pool +
    # planning attention, but ZERO support/evidence so this trains as a clean
    # negative (epistemic ~0 on attended nodes -> fallback gate learns to fire).
    overridden = bool((row.get("quality") or {}).get("answer_overrides_graph"))
    if overridden:
        evidence.clear()
        support.clear()

    for node_id, weight in list(support.items()):
        _add(evidence, node_id, max(1.0, 0.5 * weight))

    positively_used = set(evidence) | set(support) | set(planning)
    for node_id in anchor_ids:
        if node_id not in positively_used:
            _add(distractor, node_id, 0.75)

    evidence_loop_targets = [] if overridden else (evidence_loops or _micro_step_loop_targets(micro_steps))
    if not planning_loops and planning:
        planning_loops = [{
            "loop": 0,
            "source": "aggregate",
            "target": _sorted_weight_map(planning),
        }]
    if not evidence_loop_targets and evidence:
        evidence_loop_targets = [{
            "loop": 0,
            "source": "aggregate",
            "target": _sorted_weight_map(evidence),
        }]

    outer_map = _sorted_weight_map(outer)
    planning_map = _sorted_weight_map(planning)
    evidence_map = _sorted_weight_map(evidence)
    support_map = _sorted_weight_map(support)
    distractor_map = _sorted_weight_map(distractor)

    candidate_node_ids = _candidate_order(
        anchor_ids,
        support_map,
        evidence_map,
        planning_map,
        distractor_map,
    )

    nonempty_loop_logs = sum(
        1 for entry in nodes_accessed_log
        if entry.get("top_nodes")
    )
    return {
        "schema_version": PROJECTION_SCHEMA_VERSION,
        "projection_method": "phase15_v5_projection_v1",
        "candidate_node_ids": candidate_node_ids,
        "outer_retrieval_target": outer_map,
        "planning_target": planning_map,
        "evidence_target": evidence_map,
        "support_target": support_map,
        "answer_support_ids": answer_support_ids,
        "distractor_target": distractor_map,
        "planning_loop_targets": planning_loops,
        "evidence_loop_targets": evidence_loop_targets,
        "diagnostics": {
            "anchor_count": len(anchor_ids),
            "shortcut_anchor_count": len(shortcut_anchor_ids),
            "answer_support_count": len(answer_support_ids),
            "explicit_read_count": len(explicit_read_nodes),
            "micro_step_count": len(micro_steps),
            "safe_patch_count": sum(1 for patch in patches if _safe_patch_status(patch)),
            "nodes_accessed_log_count": len(nodes_accessed_log),
            "nonempty_loop_log_count": nonempty_loop_logs,
            "has_projection_signal": bool(planning_map or evidence_map or support_map),
        },
    }


def project_corpus_file(corpus_path: str | Path, out_path: str | Path) -> Dict[str, Any]:
    """Attach `v5_projection` to every row in `corpus_path` and write `out_path`."""
    corpus_path = Path(corpus_path)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rows = 0
    planning_rows = 0
    evidence_rows = 0
    support_rows = 0
    loop_rows = 0
    candidate_total = 0

    with corpus_path.open(encoding="utf-8") as src, out_path.open("w", encoding="utf-8") as dst:
        for line in src:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            projection = project_corpus_row(row)
            row["v5_projection"] = projection
            dst.write(json.dumps(row, ensure_ascii=False) + "\n")

            rows += 1
            planning_rows += int(bool(projection["planning_target"]))
            evidence_rows += int(bool(projection["evidence_target"]))
            support_rows += int(bool(projection["support_target"]))
            loop_rows += int(bool(projection["planning_loop_targets"] or projection["evidence_loop_targets"]))
            candidate_total += len(projection["candidate_node_ids"])

    return {
        "rows": rows,
        "planning_rows": planning_rows,
        "evidence_rows": evidence_rows,
        "support_rows": support_rows,
        "loop_rows": loop_rows,
        "mean_candidate_nodes": (candidate_total / rows) if rows else 0.0,
        "corpus_path": str(corpus_path),
        "out_path": str(out_path),
    }
