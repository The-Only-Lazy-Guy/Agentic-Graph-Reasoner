from __future__ import annotations

"""
graph_policy_env.py — NGR-v0 hard-validity environment.

No language-model planner. No fallback action. The learned policy chooses typed
edit actions and node/relation pointers; this environment only validates and
serializes them.
"""

import argparse
import json
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from graph_core import CANONICAL_RELATIONS, MemoryGraph, canonical_relation, lexical_overlap

EDIT_TYPES = ["no_op", "add_node", "update_node", "link_nodes", "resolve_conflict"]
RELATIONS = list(CANONICAL_RELATIONS)
EDIT_TO_ID = {x: i for i, x in enumerate(EDIT_TYPES)}
ID_TO_EDIT = {i: x for x, i in EDIT_TO_ID.items()}
REL_TO_ID = {x: i for i, x in enumerate(RELATIONS)}
ID_TO_REL = {i: x for x, i in REL_TO_ID.items()}


@dataclass
class GraphPolicyEnvConfig:
    max_candidates: int = 64
    candidate_hops: int = 1
    allow_self_link: bool = False
    invalid_reward: float = -1.0


@dataclass
class GraphPolicyState:
    signal: str = ""
    candidate_node_ids: List[str] = field(default_factory=list)
    step: int = 0
    done: bool = False
    last_action: Dict[str, Any] = field(default_factory=dict)
    trajectory: List[Dict[str, Any]] = field(default_factory=list)


def safe_node_id(text: str, prefix: str = "ngr") -> str:
    toks = re.findall(r"[a-z0-9]+", str(text or "").lower())[:8]
    return prefix + "_" + "_".join(toks or ["node"])


class GraphPolicyEnv:
    def __init__(self, graph: MemoryGraph, config: Optional[GraphPolicyEnvConfig] = None) -> None:
        self.graph = graph
        self.config = config or GraphPolicyEnvConfig()
        self.state = GraphPolicyState()

    def reset(self, signal: str, *, candidate_node_ids: Optional[Sequence[str]] = None, task: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
        if candidate_node_ids is None and task is not None:
            candidate_node_ids = task.get("candidate_node_ids") or []
        ids = [str(x) for x in (candidate_node_ids or []) if str(x) in self.graph.nodes]
        if not ids:
            ids = self._default_candidates(signal)
        seen, clean_ids = set(), []
        for nid in ids:
            if nid in self.graph.nodes and nid not in seen:
                seen.add(nid); clean_ids.append(nid)
            if len(clean_ids) >= self.config.max_candidates:
                break
        self.state = GraphPolicyState(signal=str(signal), candidate_node_ids=clean_ids)
        return self.observe()

    def _default_candidates(self, signal: str) -> List[str]:
        # Candidate construction only; this never chooses the final action.
        scored: List[Tuple[float, str]] = []
        for nid, node in self.graph.nodes.items():
            score = float(lexical_overlap(signal, f"{nid.replace('_', ' ')} {node.text}"))
            score += 0.02 * float(getattr(node, "importance", 0.5))
            score += 0.01 * float(getattr(node, "confidence", 0.5))
            if score > 0:
                scored.append((score, nid))
        scored.sort(reverse=True)
        seeds = [nid for _, nid in scored[: max(8, self.config.max_candidates // 4)]]
        expanded: List[str] = []
        for sid in seeds:
            expanded.extend(self.graph.local_neighborhood([sid], max_hops=self.config.candidate_hops, max_nodes=16))
        expanded.extend(seeds)
        return expanded[: self.config.max_candidates]

    def observe(self) -> Dict[str, Any]:
        nodes = []
        id_to_idx = {nid: i for i, nid in enumerate(self.state.candidate_node_ids)}
        for idx, nid in enumerate(self.state.candidate_node_ids):
            n = self.graph.nodes[nid]
            nodes.append({
                "idx": idx, "id": nid, "text": str(n.text), "node_type": str(n.node_type),
                "confidence": float(n.confidence), "importance": float(n.importance),
                "signal_overlap": float(lexical_overlap(self.state.signal, f"{nid.replace('_', ' ')} {n.text}")),
            })
        edges = []
        for e in self.graph.iter_local_edges(self.state.candidate_node_ids):
            if e.src in id_to_idx and e.dst in id_to_idx:
                edges.append({
                    "src_idx": id_to_idx[e.src], "dst_idx": id_to_idx[e.dst],
                    "src": e.src, "dst": e.dst, "relation": canonical_relation(e.relation),
                    "strength": float(e.strength), "directed": bool(e.directed),
                })
        return {
            "signal": self.state.signal,
            "step": self.state.step,
            "done": self.state.done,
            "candidate_nodes": nodes,
            "candidate_edges": edges,
            "edit_types": EDIT_TYPES,
            "relations": RELATIONS,
        }

    def valid_action_mask(self) -> Dict[str, Any]:
        n = len(self.state.candidate_node_ids)
        return {"edit_type": [1] * len(EDIT_TYPES), "node_pointer": [1] * n, "relation": [1] * len(RELATIONS)}

    def _node_id(self, idx: Optional[int]) -> Optional[str]:
        try:
            i = int(idx)  # type: ignore[arg-type]
        except Exception:
            return None
        if i < 0 or i >= len(self.state.candidate_node_ids):
            return None
        return self.state.candidate_node_ids[i]

    def validate_action(self, action: Mapping[str, Any]) -> Tuple[bool, List[str]]:
        errors: List[str] = []
        edit = str(action.get("edit_type", ""))
        if edit not in EDIT_TO_ID:
            return False, [f"unknown edit_type: {edit}"]
        if edit == "no_op":
            return True, errors
        if edit == "add_node":
            for idx in action.get("evidence_node_indices", []) or []:
                if self._node_id(idx) is None:
                    errors.append(f"add_node evidence index out of range: {idx}")
            return not errors, errors
        if edit in {"update_node", "resolve_conflict"}:
            if self._node_id(action.get("target_idx")) is None:
                errors.append(f"{edit} requires valid target_idx")
            return not errors, errors
        if edit == "link_nodes":
            src = self._node_id(action.get("src_idx")); dst = self._node_id(action.get("dst_idx"))
            rel = canonical_relation(action.get("relation", ""))
            if src is None: errors.append("link_nodes requires valid src_idx")
            if dst is None: errors.append("link_nodes requires valid dst_idx")
            if src is not None and dst is not None and src == dst and not self.config.allow_self_link:
                errors.append("link_nodes src and dst must differ")
            if rel not in REL_TO_ID: errors.append(f"invalid relation: {action.get('relation')}")
            return not errors, errors
        return False, [f"unsupported edit_type: {edit}"]

    def serialize_action(self, action: Mapping[str, Any]) -> Dict[str, Any]:
        edit = str(action.get("edit_type", "no_op"))
        ok, errors = self.validate_action(action)
        if not ok:
            return {"action": "invalid", "errors": errors, "raw": dict(action)}
        conf = float(action.get("confidence", 0.5))
        if edit == "no_op":
            return {"action": "no_op", "confidence": conf, "proposed": {"reason": "policy selected no_op"}}
        if edit == "add_node":
            ev = [nid for idx in (action.get("evidence_node_indices", []) or []) if (nid := self._node_id(idx))]
            return {"action": "add_node", "confidence": conf, "used_tool_result_ids": ev,
                    "proposed": {"id": safe_node_id(self.state.signal, "ngr_add"), "text": self.state.signal,
                                 "node_type": str(action.get("node_type", "claim")),
                                 "edges_to": [{"dst": nid, "relation": canonical_relation(action.get("relation", "related"))} for nid in ev]}}
        if edit == "update_node":
            target = self._node_id(action.get("target_idx"))
            return {"action": "update_node", "confidence": conf, "used_tool_result_ids": [target] if target else [],
                    "proposed": {"target_id": target, "new_text": self.state.signal}}
        if edit == "resolve_conflict":
            target = self._node_id(action.get("target_idx"))
            return {"action": "resolve_conflict", "confidence": conf, "used_tool_result_ids": [target] if target else [],
                    "proposed": {"target_id": target, "resolution_text": self.state.signal}}
        if edit == "link_nodes":
            src = self._node_id(action.get("src_idx")); dst = self._node_id(action.get("dst_idx"))
            rel = canonical_relation(action.get("relation", "related"))
            return {"action": "link_nodes", "confidence": conf, "used_tool_result_ids": [x for x in [src, dst] if x],
                    "proposed": {"src": src, "dst": dst, "relation": rel}}
        raise AssertionError(edit)

    def step(self, action: Mapping[str, Any]) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        serialized = self.serialize_action(action)
        ok = serialized.get("action") != "invalid"
        reward = 0.0 if ok else self.config.invalid_reward
        self.state.step += 1; self.state.done = True; self.state.last_action = serialized; self.state.trajectory.append(serialized)
        return self.observe(), reward, self.state.done, {"serialized_action": serialized, "valid": ok}

    def serialize_trajectory(self) -> Dict[str, Any]:
        return {"signal": self.state.signal, "candidate_node_ids": list(self.state.candidate_node_ids),
                "trajectory": list(self.state.trajectory), "final_action": self.state.last_action}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--graph", required=True)
    ap.add_argument("--signal", required=True)
    ap.add_argument("--max-candidates", type=int, default=32)
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args()
    graph = MemoryGraph.load_json(args.graph)
    env = GraphPolicyEnv(graph, GraphPolicyEnvConfig(max_candidates=args.max_candidates))
    obs = env.reset(args.signal)
    print(json.dumps({"nodes": len(graph.nodes), "edges": len(graph.edges), "candidate_count": len(obs["candidate_nodes"]),
                      "candidate_nodes": obs["candidate_nodes"][:10], "candidate_edges": obs["candidate_edges"][:10],
                      "valid_action_mask": env.valid_action_mask()}, ensure_ascii=False, indent=2))
    if args.debug:
        _, _, _, info = env.step({"edit_type": "no_op", "confidence": 0.5})
        print(json.dumps(info, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
