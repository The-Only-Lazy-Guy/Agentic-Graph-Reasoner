from __future__ import annotations
import os

os.environ["HF_HOME"] = os.path.join(os.getcwd(), "cache")

import json
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from graph_core import MemoryGraph, canonical_relation


@dataclass
class ActionSpec:
    action_type: str
    target_nodes: List[str] = field(default_factory=list)
    target_edges: List[Tuple[str, str]] = field(default_factory=list)
    target_edge_relations: List[str] = field(default_factory=list)
    target_edge_sources: List[str] = field(default_factory=list)
    relation: Optional[str] = None
    text: Optional[str] = None
    new_node_id: Optional[str] = None
    confidence: float = 0.5
    source: str = "decision_classifier"
    planner_action: Optional[str] = None
    planner_reasoning: Optional[str] = None
    planner_proposed: Optional[Dict[str, Any]] = None
    planner_validation: Optional[Dict[str, Any]] = None
    tool_calls: List[Dict[str, Any]] = field(default_factory=list)
    tool_results: List[Dict[str, Any]] = field(default_factory=list)
    used_tool_result_ids: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["target_edges"] = [[a, b] for a, b in self.target_edges]
        d["target_edge_relations"] = [canonical_relation(r) for r in self.target_edge_relations]
        d["target_edge_sources"] = [str(x) for x in self.target_edge_sources]
        return d


def _fallback_text(signal_text: str, action_type: str) -> str:
    s = " ".join(str(signal_text or "").strip().split())
    if not s:
        return "The graph should be updated based on the provided signal."
    if len(s) > 240:
        s = s[:237].rstrip() + "..."
    return s


def make_node_id(text: str, prefix: str = "generated") -> str:
    import re
    toks = re.findall(r"[A-Za-z0-9]+", text.lower())[:8]
    return prefix + "_" + "_".join(toks or ["node"])


def compile_action_spec(spec: ActionSpec, *, graph: MemoryGraph, signal_text: str) -> Dict[str, Any]:
    action = spec.action_type
    nodes = [n for n in spec.target_nodes if n in graph.nodes]
    edges = [(a, b) for a, b in spec.target_edges if a in graph.nodes and b in graph.nodes]
    text = spec.text or _fallback_text(signal_text, action)
    relation = canonical_relation(spec.relation or "related")

    plan: Dict[str, Any] = {
        "task_type": action,
        "confidence": float(spec.confidence),
        "used_nodes": nodes,
        "used_edges": [[a, b] for a, b in edges],
        "reasoning_summary": f"Compiled from classifier decision: {action}.",
        "proposed_action": {"type": action, "target_nodes": nodes, "target_edges": [[a, b] for a, b in edges]},
        "needs_tool_check": action not in {"no_op", "retrieve_context"},
        "tool_trace": {
            "tool_calls": [dict(x) for x in spec.tool_calls],
            "tool_results": [dict(x) for x in spec.tool_results],
            "used_tool_result_ids": [str(x) for x in spec.used_tool_result_ids if str(x)],
        },
    }
    pa = plan["proposed_action"]

    if action == "add_node":
        new_id = spec.new_node_id or make_node_id(text, "add")
        edges_to = []
        seen = set()
        for idx, nid in enumerate(nodes):
            if nid == new_id or nid in seen:
                continue
            edge_rel = spec.target_edge_relations[idx] if idx < len(spec.target_edge_relations) else relation
            edge_source = str(spec.target_edge_sources[idx]) if idx < len(spec.target_edge_sources) and str(spec.target_edge_sources[idx]) else "planner"
            edges_to.append({
                "dst": nid,
                "relation": canonical_relation(edge_rel or relation),
                "source": edge_source,
            })
            seen.add(nid)
        pa.update({
            "new_node": {"id": new_id, "text": text, "node_type": "claim", "confidence": 0.75, "importance": 0.75, "metadata": {"status": "proposed"}},
            "edges_to": edges_to,
            "reason": "The signal contains information not fully represented by the retrieved local component.",
        })
    elif action == "update_node":
        target = nodes[0] if nodes else None
        pa.update({"target_node_id": target, "proposed_text": text, "reason": "The signal refines an existing node."})
    elif action == "link_nodes":
        if len(nodes) >= 2:
            src, dst = nodes[0], nodes[1]
        elif edges:
            src, dst = edges[0]
        else:
            src = dst = None
        pa.update({"src": src, "dst": dst, "relation": relation, "reason": "The signal makes this relation explicit."})
    elif action == "create_bridge":
        new_id = spec.new_node_id or make_node_id(text, "bridge")
        pa.update({
            "bridge_node": {"id": new_id, "text": text, "node_type": "claim", "confidence": 0.70, "importance": 0.80, "metadata": {"kind": "bridge_insight", "status": "proposed"}},
            "connects": nodes,
            "reason": "The signal connects ideas across the retrieved local component.",
        })
    elif action == "resolve_conflict":
        pa.update({"resolution": text, "reason": "The signal resolves a conflict or misconception in the retrieved evidence."})
    elif action == "summarize_cluster":
        target = nodes[-1] if nodes else None
        pa.update({"target_summary_node_id": target, "summary_text": text, "reason": "The signal compresses a component into a reusable summary."})
    elif action == "retrieve_context":
        pa.update({"covered_by": nodes, "reason": "The request asks for context rather than a graph mutation."})
        plan["needs_tool_check"] = False
    else:  # no_op
        pa.update({"covered_by": nodes, "reason": "The signal appears covered by the retrieved graph evidence."})
        plan["needs_tool_check"] = False

    return plan


def validate_agent_plan(plan: Dict[str, Any], graph: MemoryGraph) -> Dict[str, Any]:
    errors: List[str] = []
    warnings: List[str] = []
    action = str(plan.get("task_type", ""))
    pa = plan.get("proposed_action", {}) or {}
    if not action:
        errors.append("missing task_type")
    if pa.get("type") != action:
        errors.append("proposed_action.type does not match task_type")

    for nid in plan.get("used_nodes", []) or []:
        if nid not in graph.nodes:
            errors.append(f"used node does not exist: {nid}")

    if action == "update_node":
        target = pa.get("target_node_id")
        if not target or target not in graph.nodes:
            errors.append("update_node requires existing target_node_id")
        if not pa.get("proposed_text"):
            errors.append("update_node requires proposed_text")
    elif action == "add_node":
        new_node = pa.get("new_node")
        if not isinstance(new_node, dict) or not new_node.get("id") or not new_node.get("text"):
            errors.append("add_node requires new_node.id and new_node.text")
        for edge in pa.get("edges_to", []) or []:
            if not isinstance(edge, dict):
                errors.append("add_node.edges_to entries must be objects")
                continue
            if edge.get("dst") not in graph.nodes:
                errors.append(f"add_node edge target does not exist: {edge.get('dst')}")
    elif action == "link_nodes":
        src, dst = pa.get("src"), pa.get("dst")
        if src not in graph.nodes or dst not in graph.nodes:
            errors.append("link_nodes requires existing src and dst")
        if not pa.get("relation"):
            errors.append("link_nodes requires relation")
    elif action == "create_bridge":
        b = pa.get("bridge_node")
        if not isinstance(b, dict) or not b.get("id") or not b.get("text"):
            errors.append("create_bridge requires bridge_node.id and bridge_node.text")
        if not pa.get("connects"):
            warnings.append("create_bridge has no connects list")
    elif action == "resolve_conflict":
        if not pa.get("resolution"):
            errors.append("resolve_conflict requires resolution")
    elif action == "summarize_cluster":
        if not pa.get("summary_text"):
            errors.append("summarize_cluster requires summary_text")
    elif action in {"no_op", "retrieve_context"}:
        pass
    else:
        errors.append(f"unknown action type: {action}")

    return {"ok": not errors, "errors": errors, "warnings": warnings}
