from __future__ import annotations
import os

os.environ["HF_HOME"] = os.path.join(os.getcwd(), "cache")
"""Graph-consistency reward for self-improving graph-edit policies.

This module is intentionally label-free. It scores whether a proposed graph edit
makes the graph cleaner/more coherent, not whether it matches a human action
label. Use it for offline regret/preference generation before trying online RL.

Expected project files in the same directory:
  - graph_core.py
  - editor.py
"""

import math
import re
from dataclasses import asdict, dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from graph_core import CONTRADICTION_RELATIONS, MemoryGraph, canonical_relation, lexical_overlap


_STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "of", "to", "in", "on", "for", "with", "and", "or", "but", "that", "this",
    "it", "as", "at", "by", "from", "into", "about", "than", "then", "when", "while",
}

_CONFLICT_CUES = {
    "false", "wrong", "incorrect", "misconception", "contradict", "contradicts",
    "refute", "refutes", "not true", "does not", "do not", "cannot", "can't",
    "counterexample", "disprove", "disproves", "not always", "invalid",
}

_SUMMARY_CUES = {
    "mostly about", "overall", "summarize", "summary", "overview", "cluster",
    "reasoning is about", "reasoning is mostly about", "connects", "compares",
    "tradeoff", "trade-off", "organizes", "matching",
}

_BRIDGE_CUES = {"bridge", "connect", "connection", "links", "link between", "across", "relate", "relates"}


@dataclass
class ConsistencyRewardConfig:
    # Global graph score weights.
    edge_coherence_weight: float = 0.65
    connectivity_weight: float = 0.20
    contradiction_quality_weight: float = 0.20
    duplicate_penalty_weight: float = 0.35
    contradiction_density_weight: float = 0.15
    weak_edge_penalty_weight: float = 0.25

    # Action-specific shaping.
    covered_noop_bonus: float = 0.03
    useful_edit_bonus: float = 0.03
    no_op_missed_signal_penalty: float = 0.15
    add_duplicate_penalty: float = 0.45
    update_overwrite_penalty: float = 0.55
    update_covered_penalty: float = 0.18
    conflict_misuse_penalty: float = 0.80
    bridge_misuse_penalty: float = 0.35
    summary_misuse_penalty: float = 0.45
    invalid_plan_penalty: float = 1.00
    add_isolated_penalty: float = 0.30
    add_hub_only_penalty: float = 0.12
    add_false_target_penalty: float = 0.30
    add_low_info_penalty: float = 0.12
    add_useful_edge_bonus: float = 0.030
    add_nonhub_edge_bonus: float = 0.040

    # Thresholds.
    duplicate_overlap_threshold: float = 0.86
    covered_overlap_threshold: float = 0.72
    update_min_old_new_overlap: float = 0.15
    weak_edge_overlap_threshold: float = 0.04


# -----------------------------------------------------------------------------
# Text / graph helpers
# -----------------------------------------------------------------------------


def clean(text: Any) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def content_tokens(text: str) -> List[str]:
    out: List[str] = []
    for t in re.findall(r"[A-Za-z0-9_]+", str(text or "").lower()):
        if t in _STOPWORDS or len(t) < 2:
            continue
        if t.endswith("ies") and len(t) > 4:
            t = t[:-3] + "y"
        elif t.endswith("s") and len(t) > 4 and not t.endswith(("ss", "us")):
            t = t[:-1]
        out.append(t)
    return out


def token_set(text: str) -> set[str]:
    return set(content_tokens(text))


def jaccard(a: Iterable[str], b: Iterable[str]) -> float:
    sa, sb = set(a), set(b)
    if not sa and not sb:
        return 1.0
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / max(len(sa | sb), 1)


def has_any_phrase(text: str, phrases: Iterable[str]) -> bool:
    low = str(text or "").lower()
    return any(p in low for p in phrases)


def has_conflict_cue(text: str) -> bool:
    return has_any_phrase(text, _CONFLICT_CUES)


def has_summary_cue(text: str) -> bool:
    return has_any_phrase(text, _SUMMARY_CUES)


def has_bridge_cue(text: str) -> bool:
    return has_any_phrase(text, _BRIDGE_CUES)


def node_meta(node: Any) -> Dict[str, Any]:
    meta = getattr(node, "metadata", {}) or {}
    return meta if isinstance(meta, dict) else {}


def node_descriptor(nid: str, node: Any) -> str:
    meta = node_meta(node)
    return " ".join([
        str(nid),
        str(getattr(node, "node_type", "")),
        str(meta.get("kind", "")),
        str(meta.get("status", "")),
        str(meta.get("polarity", "")),
        str(meta.get("truth", "")),
    ]).lower()


def is_false_or_hypothesis_node(nid: str, graph: MemoryGraph) -> bool:
    if nid not in graph.nodes:
        return False
    raw = node_descriptor(nid, graph.nodes[nid])
    return any(x in raw for x in ("false", "misconception", "hypothesis", "uncertain", "polarity false"))


def is_summary_node(nid: str, graph: MemoryGraph) -> bool:
    if nid not in graph.nodes:
        return False
    return str(getattr(graph.nodes[nid], "node_type", "")).lower() in {"summary", "hub", "overview"}


def degree_map(graph: MemoryGraph) -> Dict[str, int]:
    deg = {nid: 0 for nid in graph.nodes}
    for e in graph.edges:
        if e.src in deg:
            deg[e.src] += 1
        if e.dst in deg:
            deg[e.dst] += 1
    return deg


def best_node_overlap(text: str, graph: MemoryGraph) -> Tuple[float, Optional[str]]:
    best = 0.0
    best_id: Optional[str] = None
    for nid, node in graph.nodes.items():
        ov = float(lexical_overlap(text, node.text))
        if ov > best:
            best, best_id = ov, nid
    return best, best_id


def added_node_ids(before: MemoryGraph, after: MemoryGraph) -> List[str]:
    return [nid for nid in after.nodes if nid not in before.nodes]


def changed_node_ids(before: MemoryGraph, after: MemoryGraph) -> List[str]:
    out: List[str] = []
    for nid, node in after.nodes.items():
        if nid in before.nodes and clean(before.nodes[nid].text) != clean(node.text):
            out.append(nid)
    return out


def added_edges(before: MemoryGraph, after: MemoryGraph) -> List[Tuple[str, str, str]]:
    old = {(e.src, e.dst, canonical_relation(e.relation)) for e in before.edges}
    return [(e.src, e.dst, canonical_relation(e.relation)) for e in after.edges if (e.src, e.dst, canonical_relation(e.relation)) not in old]


# -----------------------------------------------------------------------------
# Global graph consistency score
# -----------------------------------------------------------------------------


def duplicate_penalty(graph: MemoryGraph, *, threshold: float) -> float:
    ids = list(graph.nodes.keys())
    if len(ids) < 2:
        return 0.0
    # O(n^2) is fine for small/medium memory graphs. Cap for giant wiki tests.
    max_pairs = 250_000
    checked = 0
    dup = 0.0
    for i, a in enumerate(ids):
        ta = graph.nodes[a].text
        for b in ids[i + 1:]:
            checked += 1
            if checked > max_pairs:
                return dup / max(checked, 1)
            ov = float(lexical_overlap(ta, graph.nodes[b].text))
            if ov >= threshold:
                dup += ov
    return dup / max(checked, 1)


def edge_quality_components(graph: MemoryGraph, *, weak_overlap_threshold: float) -> Dict[str, float]:
    if not graph.edges:
        return {
            "edge_coherence": 0.0,
            "weak_edge_penalty": 0.0,
            "contradiction_quality": 0.0,
            "contradiction_density": 0.0,
        }

    coherence_values: List[float] = []
    contradiction_values: List[float] = []
    weak_edges = 0
    contradiction_edges = 0

    for e in graph.edges:
        if e.src not in graph.nodes or e.dst not in graph.nodes:
            continue
        rel = canonical_relation(e.relation)
        src = graph.nodes[e.src]
        dst = graph.nodes[e.dst]
        ov = float(lexical_overlap(src.text, dst.text))

        if rel in CONTRADICTION_RELATIONS:
            contradiction_edges += 1
            # A contradiction edge is useful when at least one endpoint is explicitly false/hypothesis
            # or the text itself carries conflict language.
            has_false_endpoint = is_false_or_hypothesis_node(e.src, graph) or is_false_or_hypothesis_node(e.dst, graph)
            text_has_conflict = has_conflict_cue(src.text) or has_conflict_cue(dst.text)
            contradiction_values.append(1.0 if (has_false_endpoint or text_has_conflict) else 0.25)
            # Don't judge contradiction coherence by positive lexical similarity as strongly.
            coherence_values.append(min(1.0, ov / 0.18) * 0.5 + 0.25)
        else:
            q = min(1.0, ov / 0.18)
            coherence_values.append(q)
            if ov < weak_overlap_threshold and rel in {"support", "refine", "related", "depend", "cause"}:
                weak_edges += 1

    total_edges = max(len(graph.edges), 1)
    return {
        "edge_coherence": sum(coherence_values) / max(len(coherence_values), 1),
        "weak_edge_penalty": weak_edges / total_edges,
        "contradiction_quality": sum(contradiction_values) / max(len(contradiction_values), 1) if contradiction_values else 0.5,
        "contradiction_density": contradiction_edges / total_edges,
    }


def graph_consistency_score(graph: MemoryGraph, cfg: Optional[ConsistencyRewardConfig] = None) -> Tuple[float, Dict[str, float]]:
    cfg = cfg or ConsistencyRewardConfig()
    deg = degree_map(graph)
    n = max(len(graph.nodes), 1)
    isolated_frac = sum(1 for d in deg.values() if d == 0) / n
    connectedness = 1.0 - isolated_frac

    edge_parts = edge_quality_components(graph, weak_overlap_threshold=cfg.weak_edge_overlap_threshold)
    dup = duplicate_penalty(graph, threshold=cfg.duplicate_overlap_threshold)

    raw = (
        cfg.edge_coherence_weight * edge_parts["edge_coherence"]
        + cfg.connectivity_weight * connectedness
        + cfg.contradiction_quality_weight * edge_parts["contradiction_quality"]
        - cfg.duplicate_penalty_weight * dup
        - cfg.contradiction_density_weight * edge_parts["contradiction_density"]
        - cfg.weak_edge_penalty_weight * edge_parts["weak_edge_penalty"]
    )
    comps = {
        "score": float(raw),
        "connectedness": float(connectedness),
        "isolated_frac": float(isolated_frac),
        "duplicate_penalty": float(dup),
        **{k: float(v) for k, v in edge_parts.items()},
        "node_count": float(len(graph.nodes)),
        "edge_count": float(len(graph.edges)),
    }
    return float(raw), comps


# -----------------------------------------------------------------------------
# Action-specific reward shaping
# -----------------------------------------------------------------------------


def _plan_action(plan: Mapping[str, Any]) -> str:
    return str(plan.get("task_type", (plan.get("proposed_action", {}) or {}).get("type", "")))


def _plan_nodes(plan: Mapping[str, Any]) -> List[str]:
    nodes = plan.get("used_nodes", []) or []
    pa = plan.get("proposed_action", {}) or {}
    for key in ("target_nodes", "connects"):
        if isinstance(pa.get(key), list):
            nodes += pa[key]
    if pa.get("target_node_id"):
        nodes.append(pa["target_node_id"])
    for edge in pa.get("edges_to", []) or []:
        if isinstance(edge, Mapping) and edge.get("dst"):
            nodes.append(edge["dst"])
    return [str(x) for x in nodes if str(x)]


def action_penalties_and_bonus(
    before: MemoryGraph,
    after: MemoryGraph,
    *,
    signal_text: str,
    plan: Mapping[str, Any],
    validation: Optional[Mapping[str, Any]],
    cfg: ConsistencyRewardConfig,
) -> Tuple[float, Dict[str, float]]:
    action = _plan_action(plan)
    pa = plan.get("proposed_action", {}) or {}
    signal = clean(signal_text)
    best_before_overlap, best_before_id = best_node_overlap(signal, before)

    invalid = 0.0 if (validation is None or validation.get("ok", True)) else 1.0
    useful_edit_bonus = 0.0 if action in {"no_op", "retrieve_context"} else cfg.useful_edit_bonus
    covered_noop_bonus = 0.0
    no_op_missed_signal = 0.0
    add_duplicate = 0.0
    add_isolated = 0.0
    add_hub_only = 0.0
    add_false_target = 0.0
    add_low_info = 0.0
    add_edge_bonus = 0.0
    update_overwrite = 0.0
    update_covered = 0.0
    conflict_misuse = 0.0
    bridge_misuse = 0.0
    summary_misuse = 0.0

    if action in {"no_op", "retrieve_context"}:
        if best_before_overlap >= cfg.covered_overlap_threshold:
            covered_noop_bonus = cfg.covered_noop_bonus
        else:
            # If the graph does not cover a factual-looking signal, no_op should not be free.
            if len(content_tokens(signal)) >= 5:
                no_op_missed_signal = cfg.no_op_missed_signal_penalty

    if action == "add_node":
        new_text = ""
        if isinstance(pa.get("new_node"), Mapping):
            new_text = str(pa.get("new_node", {}).get("text", ""))
        new_text = new_text or str(pa.get("text", "")) or signal
        ov, _ = best_node_overlap(new_text, before)
        if ov >= cfg.duplicate_overlap_threshold:
            add_duplicate = cfg.add_duplicate_penalty * ov
        if len(content_tokens(signal)) <= 4:
            add_low_info = cfg.add_low_info_penalty
        edge_targets = [
            str(e.get("dst", ""))
            for e in pa.get("edges_to", []) or []
            if isinstance(e, Mapping) and str(e.get("dst", "")) in before.nodes
        ]
        if not edge_targets:
            add_isolated = cfg.add_isolated_penalty
        else:
            nonhub_targets = [
                nid for nid in edge_targets
                if str(getattr(before.nodes[nid], "node_type", "")).lower() not in {"hub", "summary", "overview"}
            ]
            if not nonhub_targets:
                add_hub_only = cfg.add_hub_only_penalty
            if any(is_false_or_hypothesis_node(nid, before) for nid in edge_targets):
                add_false_target = cfg.add_false_target_penalty
            add_edge_bonus = min(0.090, cfg.add_useful_edge_bonus * len(set(edge_targets)))
            add_edge_bonus += min(0.120, cfg.add_nonhub_edge_bonus * len(set(nonhub_targets)))

    if action == "update_node":
        target = str(pa.get("target_node_id", ""))
        proposed = clean(pa.get("proposed_text", signal))
        if target in before.nodes:
            old = clean(before.nodes[target].text)
            old_new = float(lexical_overlap(old, proposed))
            signal_old = float(lexical_overlap(signal, old))
            if old_new < cfg.update_min_old_new_overlap:
                update_overwrite = cfg.update_overwrite_penalty * (1.0 - old_new)
            # If old node already covered the signal, updating is usually duplicate churn.
            if signal_old >= cfg.covered_overlap_threshold:
                update_covered = cfg.update_covered_penalty * signal_old

    if action == "resolve_conflict":
        targets = _plan_nodes(plan)
        has_false_target = any(is_false_or_hypothesis_node(t, before) for t in targets)
        if not has_false_target and not has_conflict_cue(signal):
            conflict_misuse = cfg.conflict_misuse_penalty
        elif not has_conflict_cue(signal):
            # False nodes near normal factual text should not automatically trigger conflict.
            false_overlap = max(
                [float(lexical_overlap(signal, before.nodes[t].text)) for t in targets if t in before.nodes and is_false_or_hypothesis_node(t, before)]
                or [0.0]
            )
            if false_overlap < 0.16:
                conflict_misuse = 0.35 * cfg.conflict_misuse_penalty

    if action == "create_bridge":
        if not has_bridge_cue(signal):
            bridge_misuse = cfg.bridge_misuse_penalty

    if action == "summarize_cluster":
        targets = _plan_nodes(plan)
        if not has_summary_cue(signal):
            summary_misuse = cfg.summary_misuse_penalty
        if len(set(targets)) < 3:
            summary_misuse += 0.25

    bonus = useful_edit_bonus + covered_noop_bonus + add_edge_bonus
    penalty = (
        cfg.invalid_plan_penalty * invalid
        + no_op_missed_signal
        + add_duplicate
        + add_isolated
        + add_hub_only
        + add_false_target
        + add_low_info
        + update_overwrite
        + update_covered
        + conflict_misuse
        + bridge_misuse
        + summary_misuse
    )
    comps = {
        "invalid_plan_penalty": cfg.invalid_plan_penalty * invalid,
        "useful_edit_bonus": useful_edit_bonus,
        "covered_noop_bonus": covered_noop_bonus,
        "no_op_missed_signal_penalty": no_op_missed_signal,
        "add_duplicate_penalty": add_duplicate,
        "add_isolated_penalty": add_isolated,
        "add_hub_only_penalty": add_hub_only,
        "add_false_target_penalty": add_false_target,
        "add_low_info_penalty": add_low_info,
        "add_edge_bonus": add_edge_bonus,
        "update_overwrite_penalty": update_overwrite,
        "update_covered_penalty": update_covered,
        "conflict_misuse_penalty": conflict_misuse,
        "bridge_misuse_penalty": bridge_misuse,
        "summary_misuse_penalty": summary_misuse,
        "best_before_overlap": best_before_overlap,
        "best_before_node": best_before_id or "",
        "total_action_bonus": bonus,
        "total_action_penalty": penalty,
    }
    return float(bonus - penalty), comps


def graph_score_cache_key(graph: MemoryGraph) -> Tuple[int, int, int, int]:
    """Cheap structural fingerprint for reward-score caching.

    The graph object is mutated across commits, so object identity is not enough.
    This O(n+e) signature is much cheaper than the O(n^2) duplicate scan and
    changes when node text/types or edge triples change.
    """
    node_sig = 0
    for nid, node in graph.nodes.items():
        node_sig ^= hash((
            str(nid),
            clean(getattr(node, "text", "")),
            str(getattr(node, "node_type", "")),
            float(getattr(node, "confidence", 0.0)),
            float(getattr(node, "importance", 0.0)),
        ))
    edge_sig = 0
    for e in graph.edges:
        edge_sig ^= hash((str(e.src), str(e.dst), canonical_relation(e.relation), round(float(e.strength), 6)))
    return (len(graph.nodes), len(graph.edges), node_sig, edge_sig)


class GraphConsistencyReward:
    """Label-free reward based on graph coherence and action sanity.

    Signature mirrors GraphUsefulnessReward enough to be dropped into bandit loops.
    """

    def __init__(self, config: Optional[ConsistencyRewardConfig] = None) -> None:
        self.config = config or ConsistencyRewardConfig()
        self._score_cache: Dict[Tuple[int, int, int, int], Tuple[float, Dict[str, float]]] = {}

    def _graph_consistency_score_cached(self, graph: MemoryGraph) -> Tuple[float, Dict[str, float]]:
        key = graph_score_cache_key(graph)
        cached = self._score_cache.get(key)
        if cached is not None:
            return cached
        scored = graph_consistency_score(graph, self.config)
        if len(self._score_cache) > 2048:
            self._score_cache.clear()
        self._score_cache[key] = scored
        return scored

    def __call__(
        self,
        before: MemoryGraph,
        after: MemoryGraph,
        *,
        signal_text: str,
        plan: Mapping[str, Any],
        hidden_queries: Optional[Sequence[str]] = None,
        qa_items: Optional[Sequence[Mapping[str, Any]]] = None,
        validation: Optional[Mapping[str, Any]] = None,
    ) -> Tuple[float, Dict[str, Any]]:
        cfg = self.config
        before_score, before_debug = self._graph_consistency_score_cached(before)
        after_score, after_debug = self._graph_consistency_score_cached(after)
        delta = after_score - before_score
        action_adj, action_debug = action_penalties_and_bonus(
            before,
            after,
            signal_text=signal_text,
            plan=plan,
            validation=validation,
            cfg=cfg,
        )
        reward = delta + action_adj
        debug = {
            "reward": float(reward),
            "components": {
                "consistency_delta": float(delta),
                **action_debug,
            },
            "before_graph_score": before_debug,
            "after_graph_score": after_debug,
            "deltas": {
                "added_nodes": added_node_ids(before, after),
                "changed_nodes": changed_node_ids(before, after),
                "added_edges": added_edges(before, after),
            },
            "config": asdict(cfg),
        }
        return float(reward), debug
