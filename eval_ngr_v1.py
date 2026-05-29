from __future__ import annotations

"""
eval_ngr_v1.py

NGR-v1a evaluator for multi-mode rollout diagnostics.

Current scope:
1. guided_exact_progress upper-bound diagnostic
2. phase_guided middle-ground rollout metric
3. policy_only real rollout metric
4. dead-end and link-rank probes for policy diagnosis
5. covered-task rollout metrics for no-op reachability
"""

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F

from graph_core import MemoryGraph, lexical_overlap
from ngr_v1_env import (
    NGRV1Env,
    NGRV1Config,
    V1_ACTIONS,
    V1_RELATIONS,
    V1_NODE_TYPES,
    ACTION_TO_ID,
    ID_TO_ACTION,
    ID_TO_REL,
    ID_TO_NODE_TYPE,
    split_signal_spans,
)
from ngr_v1_model import NGRV1PolicyNet, V1Batch, bow_hash, PHASES


GLOBAL_DIM = 16
SESSION_SCALAR_DIM = 8
PHASE_TO_ID = {name: i for i, name in enumerate(PHASES)}
ACTION_TO_PHASE = {
    "CREATE_SESSION_NODE": "create",
    "LINK_SESSION_NODES": "link",
    "PROPOSE_LINK_SESSION_TO_MEMORY": "attach",
    "PROPOSE_ADD_SESSION_NODE": "add",
    "MARK_COVERED": "cover",
    "PROPOSE_NO_OP": "noop",
    "STOP": "stop",
}
PHASE_TO_ACTION = {
    "create": "CREATE_SESSION_NODE",
    "link": "LINK_SESSION_NODES",
    "attach": "PROPOSE_LINK_SESSION_TO_MEMORY",
    "add": "PROPOSE_ADD_SESSION_NODE",
    "cover": "MARK_COVERED",
    "noop": "PROPOSE_NO_OP",
    "stop": "STOP",
}


def phase_entropy_from_logits(logits: torch.Tensor) -> float:
    probs = torch.softmax(logits, dim=-1)
    ent = -(probs * (probs.clamp_min(1e-12).log())).sum()
    return float(ent.detach().cpu())


def top_phase_names(logits: torch.Tensor, k: int) -> List[str]:
    probs = torch.softmax(logits, dim=-1)
    topk = min(max(k, 1), probs.numel())
    ids = torch.topk(probs, k=topk).indices.tolist()
    return [PHASES[int(i)] for i in ids]


def policy_only_should_preserve_link_phase(obs: Mapping[str, Any], progress: Mapping[str, bool]) -> bool:
    if progress.get("is_noop_goal", False):
        return False
    session_nodes = obs.get("session_nodes", []) or []
    if len(session_nodes) < 2:
        return False
    session_edges = obs.get("session_edges", []) or []
    # Structural heuristic only: if the session graph is still sparse, do not
    # let policy-only top-k filtering delete the entire link action family.
    return len(session_edges) < max(1, len(session_nodes) - 1)


def structurally_required_phase(progress: Mapping[str, bool], obs: Mapping[str, Any]) -> str:
    if progress.get("is_noop_goal", False):
        if not progress.get("create_complete", False):
            return "create"
        if not progress.get("coverage_complete", False):
            return "cover"
        if not bool(obs.get("proposed_no_op", False)):
            return "noop"
        return "stop"

    if not progress.get("create_complete", False):
        return "create"
    if progress.get("has_edges", False) and not progress.get("edge_complete", False):
        return "link"
    if progress.get("has_attachments", False) and not progress.get("attach_complete", False):
        return "attach"
    if progress.get("has_adds", False) and not progress.get("add_complete", False):
        return "add"
    return "stop"


def policy_only_allowed_phases(
    obs: Mapping[str, Any],
    out: Mapping[str, torch.Tensor],
    progress: Mapping[str, bool],
    *,
    topk: int,
    protect_link_phase: bool,
    protect_structural_phase: bool,
    soften_topk_on_create_cover: bool,
) -> set[str]:
    structural_phase = structurally_required_phase(progress, obs)
    if progress.get("is_noop_goal", False):
        if topk <= 0:
            allowed = {"create", "cover", "noop", "stop"}
        else:
            allowed = set(top_phase_names(out["phase_logits"][0], topk))
            allowed &= {"create", "cover", "noop", "stop"}
        if not progress.get("create_complete", False):
            allowed.discard("cover")
            allowed.discard("noop")
            allowed.discard("stop")
            allowed.add("create")
            return allowed or {"create"}
        if not progress.get("coverage_complete", False):
            allowed.discard("noop")
            allowed.discard("stop")
            allowed.add("cover")
            return allowed or {"cover"}
        if not bool(obs.get("proposed_no_op", False)):
            allowed.discard("stop")
            allowed.add("noop")
            return allowed or {"noop"}
        return {"stop"}
    if soften_topk_on_create_cover and structural_phase in {"create", "cover"}:
        return set(PHASES)
    if topk <= 0:
        return set(PHASES)
    allowed = set(top_phase_names(out["phase_logits"][0], topk))
    if protect_link_phase and policy_only_should_preserve_link_phase(obs, progress):
        allowed.add("link")
    if protect_structural_phase:
        allowed.add(structural_phase)
    return allowed


def read_jsonl(path: str | Path) -> List[Dict[str, Any]]:
    rows = []
    with Path(path).open("r", encoding="utf-8-sig") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def mean(xs: Sequence[float]) -> float:
    return sum(xs) / max(len(xs), 1)


def clean_text(x: Any, max_len: int = 360) -> str:
    return " ".join(str(x or "").split())[:max_len]


def best_span_idx(signal: str, span_text: str) -> int:
    spans = split_signal_spans(signal)
    if not spans:
        return 0
    target = clean_text(span_text)
    best_i = 0
    best_score = -1.0
    for i, sp in enumerate(spans):
        score = lexical_overlap(target, clean_text(sp.text))
        if score > best_score:
            best_i = i
            best_score = score
    return best_i


def f1_from_counts(tp: int, pred_n: int, gold_n: int) -> Dict[str, float]:
    p = tp / max(pred_n, 1)
    r = tp / max(gold_n, 1)
    return {
        "precision": p,
        "recall": r,
        "f1": 0.0 if p + r == 0 else 2 * p * r / (p + r),
        "tp": tp,
        "pred_n": pred_n,
        "gold_n": gold_n,
    }


def greedy_match_texts(pred_texts: Sequence[str], gold_texts: Sequence[str], threshold: float = 0.45) -> Tuple[int, Dict[int, int]]:
    pairs = []
    for i, p in enumerate(pred_texts):
        for j, g in enumerate(gold_texts):
            sc = float(lexical_overlap(p, g))
            if sc >= threshold:
                pairs.append((sc, i, j))
    pairs.sort(reverse=True)

    used_p = set()
    used_g = set()
    mapping = {}
    for _sc, i, j in pairs:
        if i in used_p or j in used_g:
            continue
        used_p.add(i)
        used_g.add(j)
        mapping[i] = j

    return len(mapping), mapping


def score_session_nodes(cp, goal):
    pred = cp.get("session_nodes", []) or []
    gold = goal.get("session_nodes", []) or []
    tp, mapping = greedy_match_texts(
        [str(x.get("text", "")) for x in pred],
        [str(x.get("span_text", "")) for x in gold],
    )
    out = f1_from_counts(tp, len(pred), len(gold))
    out["pred_to_gold"] = mapping
    return out


def goal_session_nodes_for_runtime(goal: Mapping[str, Any]) -> List[Dict[str, Any]]:
    session_nodes = list(goal.get("session_nodes", []) or [])
    if session_nodes:
        return session_nodes
    covered = goal.get("covered_mappings", []) or []
    return [
        {
            "name": f"covered_{i}",
            "span_text": str(cov.get("span_text", "")),
            "node_type": "concept",
        }
        for i, cov in enumerate(covered)
    ]


def runtime_pred_to_gold(obs: Mapping[str, Any], goal: Mapping[str, Any], threshold: float = 0.45) -> Dict[int, int]:
    pred = obs.get("session_nodes", []) or []
    gold = goal_session_nodes_for_runtime(goal)
    _tp, mapping = greedy_match_texts(
        [str(x.get("text", "")) for x in pred],
        [str(x.get("span_text", "")) for x in gold],
        threshold=threshold,
    )
    return mapping


def goal_edge_structures(goal: Mapping[str, Any]) -> Tuple[set[Tuple[int, int, str]], Dict[Tuple[int, int], set[str]]]:
    goal_nodes = goal_session_nodes_for_runtime(goal)
    gold_name = {str(g.get("name", f"s{i}")): i for i, g in enumerate(goal_nodes)}
    gold_edges: set[Tuple[int, int, str]] = set()
    gold_pair_to_rels: Dict[Tuple[int, int], set[str]] = defaultdict(set)
    for e in goal.get("session_edges", []) or []:
        src = gold_name.get(str(e.get("src", "")))
        dst = gold_name.get(str(e.get("dst", "")))
        rel = str(e.get("relation", "related"))
        if src is None or dst is None:
            continue
        gold_edges.add((src, dst, rel))
        gold_pair_to_rels[(src, dst)].add(rel)
    return gold_edges, gold_pair_to_rels


def mapped_runtime_edge_set(obs: Mapping[str, Any], pred_to_gold: Mapping[int, int]) -> set[Tuple[int, int, str]]:
    pred_edges: set[Tuple[int, int, str]] = set()
    for e in obs.get("session_edges", []) or []:
        src = int(e.get("src", -1))
        dst = int(e.get("dst", -1))
        rel = str(e.get("relation", "related"))
        if src in pred_to_gold and dst in pred_to_gold:
            pred_edges.add((pred_to_gold[src], pred_to_gold[dst], rel))
    return pred_edges


def missing_gold_link_runtime_targets(
    obs: Mapping[str, Any],
    goal: Mapping[str, Any],
) -> Dict[str, Any]:
    pred_to_gold = runtime_pred_to_gold(obs, goal)
    gold_edges, _gold_pair_to_rels = goal_edge_structures(goal)
    existing = mapped_runtime_edge_set(obs, pred_to_gold)
    missing_gold_edges = gold_edges - existing

    gold_to_preds: Dict[int, List[int]] = defaultdict(list)
    for pred_idx, gold_idx in pred_to_gold.items():
        gold_to_preds[int(gold_idx)].append(int(pred_idx))

    runtime_gold_tuples: set[Tuple[int, int, str]] = set()
    runtime_gold_pairs: set[Tuple[int, int]] = set()
    runtime_gold_pair_to_rels: Dict[Tuple[int, int], set[str]] = defaultdict(set)
    missing_due_to_node_match = 0

    for gold_src, gold_dst, rel in missing_gold_edges:
        src_preds = gold_to_preds.get(int(gold_src), [])
        dst_preds = gold_to_preds.get(int(gold_dst), [])
        if not src_preds or not dst_preds:
            missing_due_to_node_match += 1
            continue
        for src_pred in src_preds:
            for dst_pred in dst_preds:
                if src_pred == dst_pred:
                    continue
                runtime_gold_tuples.add((src_pred, dst_pred, rel))
                runtime_gold_pairs.add((src_pred, dst_pred))
                runtime_gold_pair_to_rels[(src_pred, dst_pred)].add(rel)

    return {
        "pred_to_gold": dict(pred_to_gold),
        "missing_gold_edges": sorted(missing_gold_edges),
        "runtime_gold_tuples": runtime_gold_tuples,
        "runtime_gold_pairs": runtime_gold_pairs,
        "runtime_gold_pair_to_rels": runtime_gold_pair_to_rels,
        "no_gold_node_match": missing_due_to_node_match > 0 and not runtime_gold_tuples,
        "missing_due_to_node_match_count": int(missing_due_to_node_match),
    }


def score_session_edges(cp, goal, pred_to_gold):
    gold_nodes = goal.get("session_nodes", []) or []
    gold_name = {str(g.get("name", f"s{i}")): i for i, g in enumerate(gold_nodes)}

    gold_set = set()
    for e in goal.get("session_edges", []) or []:
        s = gold_name.get(str(e.get("src", "")))
        d = gold_name.get(str(e.get("dst", "")))
        r = str(e.get("relation", "related"))
        if s is not None and d is not None:
            gold_set.add((s, d, r))

    pred_set = set()
    for e in cp.get("session_edges", []) or []:
        s = int(e.get("src", -1))
        d = int(e.get("dst", -1))
        r = str(e.get("relation", "related"))
        if s in pred_to_gold and d in pred_to_gold:
            pred_set.add((pred_to_gold[s], pred_to_gold[d], r))

    return f1_from_counts(len(pred_set & gold_set), len(pred_set), len(gold_set))


def score_attachments(cp, goal, pred_to_gold):
    gold_nodes = goal.get("session_nodes", []) or []
    gold_name = {str(g.get("name", f"s{i}")): i for i, g in enumerate(gold_nodes)}

    gold_set = set()
    for a in goal.get("memory_attachments", []) or []:
        s = gold_name.get(str(a.get("session", "")))
        mem = str(a.get("memory_id", ""))
        r = str(a.get("relation", "related"))
        if s is not None and mem:
            gold_set.add((s, mem, r))

    pred_set = set()
    for a in cp.get("attachments", []) or []:
        ps = int(a.get("session_idx", -1))
        mem = str(a.get("memory_id", ""))
        r = str(a.get("relation", "related"))
        if ps in pred_to_gold and mem:
            pred_set.add((pred_to_gold[ps], mem, r))

    return f1_from_counts(len(pred_set & gold_set), len(pred_set), len(gold_set))


def commit_items(cp, goal, pred_to_gold):
    gold_nodes = goal.get("session_nodes", []) or []
    gold_name = {str(g.get("name", f"s{i}")): i for i, g in enumerate(gold_nodes)}

    pred_items = set()
    for e in cp.get("final_memory_edits", []) or []:
        action = str(e.get("action", ""))
        if action == "add_node":
            ps = e.get("from_session_idx")
            if ps is not None and int(ps) in pred_to_gold:
                pred_items.add(("add_node", pred_to_gold[int(ps)]))
        elif action == "link_nodes":
            if "from_attachment" in e:
                ps = int(e.get("from_attachment"))
                prop = e.get("proposed", {}) or {}
                if ps in pred_to_gold:
                    pred_items.add(("link_session_memory", pred_to_gold[ps], str(prop.get("dst", "")), str(prop.get("relation", "related"))))
            elif "from_session_edge" in e:
                edge = e.get("from_session_edge", {}) or {}
                prop = e.get("proposed", {}) or {}
                s = int(edge.get("src", -1))
                d = int(edge.get("dst", -1))
                if s in pred_to_gold and d in pred_to_gold:
                    pred_items.add(("link_session_session", pred_to_gold[s], pred_to_gold[d], str(prop.get("relation", "related"))))
        elif action == "no_op":
            pred_items.add(("no_op",))

    gold_items = set()
    for f in goal.get("final_commits", []) or []:
        action = str(f.get("action", ""))
        if action == "add_node":
            si = gold_name.get(str(f.get("session", "")))
            if si is not None:
                gold_items.add(("add_node", si))
        elif action == "link_nodes":
            si = gold_name.get(str(f.get("session", "")))
            mem = str(f.get("memory_id", ""))
            r = str(f.get("relation", "related"))
            if si is not None and mem:
                gold_items.add(("link_session_memory", si, mem, r))
        elif action == "no_op":
            gold_items.add(("no_op",))

    return pred_items, gold_items


def score_commit(cp, goal, pred_to_gold):
    pred, gold = commit_items(cp, goal, pred_to_gold)
    return f1_from_counts(len(pred & gold), len(pred), len(gold))


def score_no_op(cp, goal, task):
    gold_noop = any(str(f.get("action", "")) == "no_op" for f in goal.get("final_commits", []) or [])
    if not gold_noop and task != "covered_long_signal":
        return None
    pred_noop = any(str(e.get("action", "")) == "no_op" for e in cp.get("final_memory_edits", []) or [])
    return int(pred_noop == gold_noop)


def action_hist_vector(action_history: Sequence[Mapping[str, Any]]) -> torch.Tensor:
    v = torch.zeros(len(V1_ACTIONS), dtype=torch.float32)
    for a in action_history or []:
        aid = ACTION_TO_ID.get(str(a.get("action", "")))
        if aid is not None:
            v[aid] += 1.0
    if v.sum() > 0:
        v = v / v.sum().clamp_min(1.0)
    return v


def global_scalar_from_obs(obs: Mapping[str, Any]) -> torch.Tensor:
    hist = action_hist_vector(obs.get("action_history", []) or [])
    sess = obs.get("session_nodes", []) or []
    edges = obs.get("session_edges", []) or []
    atts = obs.get("attachments", []) or []
    covered = [s for s in sess if s.get("covered_by")]
    proposed_adds = [s for s in sess if s.get("proposed_add")]

    vals = [
        len(sess) / 16.0,
        len(edges) / 32.0,
        len(atts) / 32.0,
        len(proposed_adds) / 16.0,
        len(covered) / 16.0,
        1.0 if obs.get("proposed_no_op") else 0.0,
        len(obs.get("memory_nodes", []) or []) / 512.0,
        len(obs.get("spans", []) or []) / 32.0,
        len(obs.get("action_history", []) or []) / 12.0,
    ]

    for name in [
        "CREATE_SESSION_NODE",
        "LINK_SESSION_NODES",
        "PROPOSE_LINK_SESSION_TO_MEMORY",
        "MARK_COVERED",
        "PROPOSE_ADD_SESSION_NODE",
        "PROPOSE_NO_OP",
        "STOP",
    ]:
        vals.append(float(hist[ACTION_TO_ID[name]]))

    vals = vals[:GLOBAL_DIM]
    vals.extend([0.0] * (GLOBAL_DIM - len(vals)))
    return torch.tensor(vals, dtype=torch.float32)


def build_batch(env: NGRV1Env, hash_dim: int, device: torch.device) -> V1Batch:
    obs = env.observe()

    signal_bow = bow_hash(obs["signal"], hash_dim)[None, :]

    spans = obs.get("spans", []) or []
    if spans:
        span_bow = torch.stack([bow_hash(s.get("text", ""), hash_dim) for s in spans])[None, :, :]
        span_mask = torch.ones(1, len(spans), dtype=torch.bool)
    else:
        span_bow = torch.zeros(1, 1, hash_dim)
        span_mask = torch.zeros(1, 1, dtype=torch.bool)

    mems = obs.get("memory_nodes", []) or []
    if mems:
        memory_bow = torch.stack([
            bow_hash(f"{m.get('id','')} {m.get('text','')}", hash_dim)
            for m in mems
        ])[None, :, :]
        memory_scalar = torch.tensor([[
            [
                float(m.get("confidence", 0.5)),
                float(m.get("importance", 0.5)),
                float(m.get("signal_overlap", 0.0)),
                float(m.get("retrieval_score", 0.0)),
            ]
            for m in mems
        ]], dtype=torch.float32)
        memory_mask = torch.ones(1, len(mems), dtype=torch.bool)
    else:
        memory_bow = torch.zeros(1, 1, hash_dim)
        memory_scalar = torch.zeros(1, 1, 4)
        memory_mask = torch.zeros(1, 1, dtype=torch.bool)

    sess = obs.get("session_nodes", []) or []
    edges = obs.get("session_edges", []) or []
    atts = obs.get("attachments", []) or []

    if sess:
        in_deg = [0] * len(sess)
        out_deg = [0] * len(sess)
        attach_count = [0] * len(sess)

        for e in edges:
            s = int(e.get("src", -1))
            d = int(e.get("dst", -1))
            if 0 <= s < len(sess):
                out_deg[s] += 1
            if 0 <= d < len(sess):
                in_deg[d] += 1

        for a in atts:
            s = int(a.get("session_idx", -1))
            if 0 <= s < len(sess):
                attach_count[s] += 1

        session_bow = torch.stack([bow_hash(s.get("text", ""), hash_dim) for s in sess])[None, :, :]
        session_scalar = torch.tensor([[
            [
                1.0,
                1.0 if s.get("covered_by") else 0.0,
                1.0 if s.get("proposed_add") else 0.0,
                min(attach_count[i], 8) / 8.0,
                min(in_deg[i], 8) / 8.0,
                min(out_deg[i], 8) / 8.0,
                (sum(s.get("span_indices", [0])) / max(len(s.get("span_indices", [0])), 1)) / max(len(spans), 1),
                i / max(len(sess), 1),
            ]
            for i, s in enumerate(sess)
        ]], dtype=torch.float32)
        session_mask = torch.ones(1, len(sess), dtype=torch.bool)
    else:
        session_bow = torch.zeros(1, 1, hash_dim)
        session_scalar = torch.zeros(1, 1, SESSION_SCALAR_DIM)
        session_mask = torch.zeros(1, 1, dtype=torch.bool)

    hist = action_hist_vector(obs.get("action_history", []) or [])[None, :]
    global_scalar = global_scalar_from_obs(obs)[None, :]

    zlong = torch.zeros(1, dtype=torch.long)
    zfloat = torch.zeros(1)

    batch = V1Batch(
        signal_bow=signal_bow,
        span_bow=span_bow,
        span_mask=span_mask,
        memory_bow=memory_bow,
        memory_scalar=memory_scalar,
        memory_mask=memory_mask,
        session_bow=session_bow,
        session_scalar=session_scalar,
        session_mask=session_mask,
        action_hist=hist,
        global_scalar=global_scalar,
        y_action=zlong,
        y_span=zlong,
        y_session=zlong,
        y_session_dst=zlong,
        y_memory=zlong,
        y_relation=zlong,
        y_node_type=zlong,
        y_value=zfloat,
    )

    return V1Batch(**{k: getattr(batch, k).to(device) for k in batch.__dataclass_fields__})


def is_noop_goal(goal: Mapping[str, Any]) -> bool:
    return any(str(f.get("action", "")) == "no_op" for f in goal.get("final_commits", []) or [])


def goal_session_name_map(goal: Mapping[str, Any]) -> Dict[str, Mapping[str, Any]]:
    return {str(s.get("name", f"s{i}")): s for i, s in enumerate(goal_session_nodes_for_runtime(goal))}


def env_state_sets(obs: Mapping[str, Any]) -> Dict[str, Any]:
    session_nodes = obs.get("session_nodes", []) or []
    session_edges = obs.get("session_edges", []) or []
    attachments = obs.get("attachments", []) or []
    memory_nodes = obs.get("memory_nodes", []) or []

    return {
        "session_nodes": session_nodes,
        "session_edges": session_edges,
        "attachments": attachments,
        "memory_nodes": memory_nodes,
        "covered_count": sum(1 for s in session_nodes if s.get("covered_by")),
        "proposed_add_count": sum(1 for s in session_nodes if s.get("proposed_add")),
        "proposed_no_op": bool(obs.get("proposed_no_op", False)),
    }


def coverage_complete_env(row: Mapping[str, Any], obs: Mapping[str, Any]) -> bool:
    goal = row.get("goal", {}) or {}
    covs = goal.get("covered_mappings", []) or []
    if not covs:
        return False
    pred_to_gold = runtime_pred_to_gold(obs, goal)
    pred_set = set()
    for pred_idx, gold_idx in pred_to_gold.items():
        covered_by = str((obs.get("session_nodes", []) or [])[pred_idx].get("covered_by", ""))
        if covered_by:
            pred_set.add((gold_idx, covered_by))
    gold_set = {(i, str(cov.get("memory_id", ""))) for i, cov in enumerate(covs) if str(cov.get("memory_id", ""))}
    return gold_set.issubset(pred_set)


def goal_progress_env(row: Mapping[str, Any], obs: Mapping[str, Any]) -> Dict[str, bool]:
    goal = row.get("goal", {}) or {}
    pred_to_gold = runtime_pred_to_gold(obs, goal)
    goal_nodes = goal_session_nodes_for_runtime(goal)
    gold_name = {str(g.get("name", f"s{i}")): i for i, g in enumerate(goal_nodes)}

    create_complete = len(pred_to_gold) >= len(goal_nodes)

    pred_edge_set = set()
    for e in obs.get("session_edges", []) or []:
        src = int(e.get("src", -1))
        dst = int(e.get("dst", -1))
        rel = str(e.get("relation", "related"))
        if src in pred_to_gold and dst in pred_to_gold:
            pred_edge_set.add((pred_to_gold[src], pred_to_gold[dst], rel))

    gold_edge_set = set()
    for e in goal.get("session_edges", []) or []:
        src = gold_name.get(str(e.get("src", "")))
        dst = gold_name.get(str(e.get("dst", "")))
        rel = str(e.get("relation", "related"))
        if src is not None and dst is not None:
            gold_edge_set.add((src, dst, rel))
    edge_complete = gold_edge_set.issubset(pred_edge_set)

    pred_att_set = set()
    for a in obs.get("attachments", []) or []:
        session_idx = int(a.get("session_idx", -1))
        mem = str(a.get("memory_id", ""))
        rel = str(a.get("relation", "related"))
        if session_idx in pred_to_gold and mem:
            pred_att_set.add((pred_to_gold[session_idx], mem, rel))

    gold_att_set = set()
    for a in goal.get("memory_attachments", []) or []:
        session_idx = gold_name.get(str(a.get("session", "")))
        mem = str(a.get("memory_id", ""))
        rel = str(a.get("relation", "related"))
        if session_idx is not None and mem:
            gold_att_set.add((session_idx, mem, rel))
    attach_complete = gold_att_set.issubset(pred_att_set)

    pred_add_set = {
        pred_to_gold[i]
        for i, sn in enumerate(obs.get("session_nodes", []) or [])
        if i in pred_to_gold and bool(sn.get("proposed_add"))
    }
    gold_add_set = {
        gold_name[str(f.get("session", ""))]
        for f in goal.get("final_commits", []) or []
        if str(f.get("action", "")) == "add_node" and str(f.get("session", "")) in gold_name
    }
    add_complete = gold_add_set.issubset(pred_add_set)

    return {
        "create_complete": create_complete,
        "edge_complete": edge_complete,
        "attach_complete": attach_complete,
        "add_complete": add_complete,
        "has_edges": bool(gold_edge_set),
        "has_attachments": bool(gold_att_set),
        "has_adds": bool(gold_add_set),
        "coverage_complete": coverage_complete_env(row, obs),
        "is_noop_goal": is_noop_goal(goal),
        "final_complete": final_commit_complete_env(row, obs),
    }


def gold_phase_for_obs(row: Mapping[str, Any], obs: Mapping[str, Any]) -> str:
    progress = goal_progress_env(row, obs)
    if progress["is_noop_goal"]:
        if not progress["create_complete"]:
            return "create"
        if not progress["coverage_complete"]:
            return "cover"
        if not bool(obs.get("proposed_no_op", False)):
            return "noop"
        return "stop"

    if not progress["create_complete"]:
        return "create"
    if progress["has_edges"] and not progress["edge_complete"]:
        return "link"
    if progress["has_attachments"] and not progress["attach_complete"]:
        return "attach"
    if progress["has_adds"] and not progress["add_complete"]:
        return "add"
    return "stop"


def exact_progress_tuples(row: Mapping[str, Any], env: NGRV1Env, obs: Mapping[str, Any], progress: Mapping[str, bool]) -> List[Dict[str, Any]]:
    goal = row.get("goal", {}) or {}
    signal = str(row.get("signal", ""))
    mem_ids = [str(m.get("id", "")) for m in (obs.get("memory_nodes", []) or []) if str(m.get("id", ""))]
    pred_to_gold = runtime_pred_to_gold(obs, goal)
    gold_to_pred = {gold_idx: pred_idx for pred_idx, gold_idx in pred_to_gold.items()}
    goal_nodes = goal_session_nodes_for_runtime(goal)
    gold_name = {str(g.get("name", f"s{i}")): i for i, g in enumerate(goal_nodes)}

    candidates: List[Dict[str, Any]] = []

    if progress["is_noop_goal"]:
        if not progress["create_complete"]:
            matched_gold = set(pred_to_gold.values())
            for gold_idx, spec in enumerate(goal_nodes):
                if gold_idx in matched_gold:
                    continue
                candidates.append({
                    "action": "CREATE_SESSION_NODE",
                    "span_indices": [best_span_idx(signal, str(spec.get("span_text", "")))],
                    "node_type": str(spec.get("node_type", "concept")),
                })
        elif not progress["coverage_complete"]:
            covered = set()
            for pred_idx, gold_idx in pred_to_gold.items():
                covered_by = str((obs.get("session_nodes", []) or [])[pred_idx].get("covered_by", ""))
                if covered_by:
                    covered.add((gold_idx, covered_by))
            for i, cov in enumerate(goal.get("covered_mappings", []) or []):
                mem = str(cov.get("memory_id", ""))
                pred_idx = gold_to_pred.get(i)
                if pred_idx is None or not mem or (i, mem) in covered:
                    continue
                candidates.append({
                    "action": "MARK_COVERED",
                    "session_idx": pred_idx,
                    "memory_idx": int(mem_ids.index(mem)) if mem in mem_ids else -1,
                })
        elif not bool(obs.get("proposed_no_op", False)):
            candidates.append({"action": "PROPOSE_NO_OP"})
        else:
            candidates.append({"action": "STOP"})
    else:
        if not progress["create_complete"]:
            matched_gold = set(pred_to_gold.values())
            for gold_idx, spec in enumerate(goal_nodes):
                if gold_idx in matched_gold:
                    continue
                candidates.append({
                    "action": "CREATE_SESSION_NODE",
                    "span_indices": [best_span_idx(signal, str(spec.get("span_text", "")))],
                    "node_type": str(spec.get("node_type", "concept")),
                })
        elif progress["has_edges"] and not progress["edge_complete"]:
            existing = set()
            for e in obs.get("session_edges", []) or []:
                src = int(e.get("src", -1))
                dst = int(e.get("dst", -1))
                rel = str(e.get("relation", "related"))
                if src in pred_to_gold and dst in pred_to_gold:
                    existing.add((pred_to_gold[src], pred_to_gold[dst], rel))
            for e in goal.get("session_edges", []) or []:
                src_gold = gold_name.get(str(e.get("src", "")))
                dst_gold = gold_name.get(str(e.get("dst", "")))
                rel = str(e.get("relation", "related"))
                if src_gold is None or dst_gold is None or (src_gold, dst_gold, rel) in existing:
                    continue
                if src_gold in gold_to_pred and dst_gold in gold_to_pred:
                    candidates.append({
                        "action": "LINK_SESSION_NODES",
                        "src_session_idx": gold_to_pred[src_gold],
                        "dst_session_idx": gold_to_pred[dst_gold],
                        "relation": rel,
                    })
        elif progress["has_attachments"] and not progress["attach_complete"]:
            existing = set()
            for a in obs.get("attachments", []) or []:
                session_idx = int(a.get("session_idx", -1))
                mem = str(a.get("memory_id", ""))
                rel = str(a.get("relation", "related"))
                if session_idx in pred_to_gold and mem:
                    existing.add((pred_to_gold[session_idx], mem, rel))
            for a in goal.get("memory_attachments", []) or []:
                session_gold = gold_name.get(str(a.get("session", "")))
                mem = str(a.get("memory_id", ""))
                rel = str(a.get("relation", "related"))
                if session_gold is None or not mem or (session_gold, mem, rel) in existing:
                    continue
                pred_idx = gold_to_pred.get(session_gold)
                if pred_idx is not None and mem in mem_ids:
                    candidates.append({
                        "action": "PROPOSE_LINK_SESSION_TO_MEMORY",
                        "session_idx": pred_idx,
                        "memory_idx": int(mem_ids.index(mem)),
                        "relation": rel,
                    })
        elif progress["has_adds"] and not progress["add_complete"]:
            for f in goal.get("final_commits", []) or []:
                if str(f.get("action", "")) != "add_node":
                    continue
                session_gold = gold_name.get(str(f.get("session", "")))
                if session_gold is None:
                    continue
                pred_idx = gold_to_pred.get(session_gold)
                if pred_idx is None:
                    continue
                sn = (obs.get("session_nodes", []) or [])[pred_idx]
                if not bool(sn.get("proposed_add")):
                    candidates.append({"action": "PROPOSE_ADD_SESSION_NODE", "session_idx": pred_idx})
        else:
            candidates.append({"action": "STOP"})

    out: List[Dict[str, Any]] = []
    seen = set()
    for cand in candidates:
        valid, _errors = env.validate_action(cand)
        if not valid:
            continue
        key = json.dumps(cand, sort_keys=True)
        if key not in seen:
            seen.add(key)
            out.append(cand)
    return out


def final_commit_complete_env(row: Mapping[str, Any], obs: Mapping[str, Any]) -> bool:
    """
    Runtime completion gate.

    Runtime session nodes do not preserve goal names, so recover a session-to-goal
    mapping by text overlap and require the exact goal structure under that mapping.
    """
    goal = row.get("goal", {}) or {}
    pred_to_gold = runtime_pred_to_gold(obs, goal)
    goal_nodes = goal_session_nodes_for_runtime(goal)

    if is_noop_goal(goal):
        return bool(obs.get("proposed_no_op", False)) and coverage_complete_env(row, obs)

    if len(pred_to_gold) < len(goal_nodes):
        return False

    gold_name = {str(g.get("name", f"s{i}")): i for i, g in enumerate(goal_nodes)}

    pred_edge_set = set()
    for e in obs.get("session_edges", []) or []:
        src = int(e.get("src", -1))
        dst = int(e.get("dst", -1))
        rel = str(e.get("relation", "related"))
        if src in pred_to_gold and dst in pred_to_gold:
            pred_edge_set.add((pred_to_gold[src], pred_to_gold[dst], rel))

    gold_edge_set = set()
    for e in goal.get("session_edges", []) or []:
        src = gold_name.get(str(e.get("src", "")))
        dst = gold_name.get(str(e.get("dst", "")))
        rel = str(e.get("relation", "related"))
        if src is not None and dst is not None:
            gold_edge_set.add((src, dst, rel))
    if not gold_edge_set.issubset(pred_edge_set):
        return False

    pred_att_set = set()
    for a in obs.get("attachments", []) or []:
        session_idx = int(a.get("session_idx", -1))
        mem = str(a.get("memory_id", ""))
        rel = str(a.get("relation", "related"))
        if session_idx in pred_to_gold and mem:
            pred_att_set.add((pred_to_gold[session_idx], mem, rel))

    gold_att_set = set()
    for a in goal.get("memory_attachments", []) or []:
        session_idx = gold_name.get(str(a.get("session", "")))
        mem = str(a.get("memory_id", ""))
        rel = str(a.get("relation", "related"))
        if session_idx is not None and mem:
            gold_att_set.add((session_idx, mem, rel))
    if not gold_att_set.issubset(pred_att_set):
        return False

    pred_add_set = {
        pred_to_gold[i]
        for i, sn in enumerate(obs.get("session_nodes", []) or [])
        if i in pred_to_gold and bool(sn.get("proposed_add"))
    }
    gold_add_set = {
        gold_name[str(f.get("session", ""))]
        for f in goal.get("final_commits", []) or []
        if str(f.get("action", "")) == "add_node" and str(f.get("session", "")) in gold_name
    }
    if not gold_add_set.issubset(pred_add_set):
        return False

    return True


def topk_from_logits(logits: torch.Tensor, mask: Sequence[int], k: int) -> List[int]:
    logits = logits.detach().clone().flatten()
    m = torch.tensor(mask, dtype=torch.bool, device=logits.device)
    if m.numel() < logits.numel():
        m = torch.cat([m, torch.zeros(logits.numel() - m.numel(), dtype=torch.bool, device=logits.device)])
    m = m[: logits.numel()]
    logits = logits.masked_fill(~m, -1e9)
    valid = int(m.sum().item())
    if valid <= 0:
        return []
    k = min(max(1, k), valid)
    return [int(x) for x in logits.topk(k).indices.detach().cpu().tolist()]


def topk_pair_indices(logits: torch.Tensor, k: int) -> List[Tuple[int, int]]:
    flat = logits.detach().flatten()
    valid = torch.isfinite(flat) & (flat > -1e8)
    if int(valid.sum().item()) <= 0:
        return []
    masked = flat.masked_fill(~valid, -1e9)
    k = min(k, int(valid.sum().item()))
    idxs = [int(x) for x in masked.topk(k).indices.detach().cpu().tolist()]
    cols = logits.size(1)
    return [(i // cols, i % cols) for i in idxs]


def logp(logits: torch.Tensor, idx: int) -> float:
    if idx < 0 or idx >= logits.numel():
        return -1e9
    return float(F.log_softmax(logits, dim=-1)[idx].detach().cpu())


def logp_pair(logits: torch.Tensor, i: int, j: int) -> float:
    if i < 0 or j < 0 or i >= logits.size(0) or j >= logits.size(1):
        return -1e9
    return float(F.log_softmax(logits.flatten(), dim=-1)[i * logits.size(1) + j].detach().cpu())


def evidence_ok_for_attach(obs: Mapping[str, Any], session_idx: int, memory_idx: int, threshold: float) -> bool:
    if threshold <= 0:
        return True

    sess = obs.get("session_nodes", []) or []
    mems = obs.get("memory_nodes", []) or []
    if not (0 <= session_idx < len(sess) and 0 <= memory_idx < len(mems)):
        return False

    st = str(sess[session_idx].get("text", ""))
    mt = f"{mems[memory_idx].get('id','')} {mems[memory_idx].get('text','')}"
    overlap = float(lexical_overlap(st, mt))
    retrieval_score = float(mems[memory_idx].get("retrieval_score", 0.0))
    signal_overlap = float(mems[memory_idx].get("signal_overlap", 0.0))

    return max(overlap, retrieval_score, signal_overlap) >= threshold


def evidence_ok_for_cover(obs: Mapping[str, Any], session_idx: int, memory_idx: int, threshold: float) -> bool:
    sess = obs.get("session_nodes", []) or []
    mems = obs.get("memory_nodes", []) or []
    if not (0 <= session_idx < len(sess) and 0 <= memory_idx < len(mems)):
        return False
    st = clean_text(sess[session_idx].get("text", ""))
    mt = f"{mems[memory_idx].get('id','')} {mems[memory_idx].get('text','')}"
    overlap = float(lexical_overlap(st, clean_text(mt)))
    retrieval_score = float(mems[memory_idx].get("retrieval_score", 0.0))
    signal_overlap = float(mems[memory_idx].get("signal_overlap", 0.0))
    return max(overlap, retrieval_score, signal_overlap) >= threshold


def runtime_action_valid(
    row: Mapping[str, Any],
    obs: Mapping[str, Any],
    masks: Mapping[str, Sequence[int]],
    *,
    eval_mode: str,
) -> Tuple[List[int], Dict[str, bool], Dict[str, Any]]:
    goal = row.get("goal", {}) or {}
    action_valid = list(masks["action"])
    progress = goal_progress_env(row, obs)

    if not (progress["is_noop_goal"] and progress["coverage_complete"] and not obs.get("proposed_no_op", False)):
        action_valid[ACTION_TO_ID["PROPOSE_NO_OP"]] = 0

    if not (goal.get("covered_mappings") or []):
        action_valid[ACTION_TO_ID["MARK_COVERED"]] = 0

    action_valid[ACTION_TO_ID["PROPOSE_LINK_MEMORY_TO_MEMORY"]] = 0

    if eval_mode in {"phase_guided", "guided_exact_progress"}:
        if progress["is_noop_goal"]:
            if not progress["create_complete"]:
                for action_name in ["LINK_SESSION_NODES", "PROPOSE_LINK_SESSION_TO_MEMORY", "PROPOSE_ADD_SESSION_NODE", "MARK_COVERED", "PROPOSE_NO_OP", "STOP"]:
                    action_valid[ACTION_TO_ID[action_name]] = 0
            elif not progress["coverage_complete"]:
                for action_name in ["CREATE_SESSION_NODE", "LINK_SESSION_NODES", "PROPOSE_LINK_SESSION_TO_MEMORY", "PROPOSE_ADD_SESSION_NODE", "PROPOSE_NO_OP", "STOP"]:
                    action_valid[ACTION_TO_ID[action_name]] = 0
            elif not bool(obs.get("proposed_no_op", False)):
                for action_name in ["CREATE_SESSION_NODE", "LINK_SESSION_NODES", "PROPOSE_LINK_SESSION_TO_MEMORY", "PROPOSE_ADD_SESSION_NODE", "MARK_COVERED", "STOP"]:
                    action_valid[ACTION_TO_ID[action_name]] = 0
            else:
                for action_name in ["CREATE_SESSION_NODE", "LINK_SESSION_NODES", "PROPOSE_LINK_SESSION_TO_MEMORY", "PROPOSE_ADD_SESSION_NODE", "MARK_COVERED", "PROPOSE_NO_OP"]:
                    action_valid[ACTION_TO_ID[action_name]] = 0
        else:
            if not progress["create_complete"]:
                for action_name in ["LINK_SESSION_NODES", "PROPOSE_LINK_SESSION_TO_MEMORY", "PROPOSE_ADD_SESSION_NODE", "MARK_COVERED", "PROPOSE_NO_OP", "STOP"]:
                    action_valid[ACTION_TO_ID[action_name]] = 0
            elif progress["has_edges"] and not progress["edge_complete"]:
                for action_name in ["CREATE_SESSION_NODE", "PROPOSE_LINK_SESSION_TO_MEMORY", "PROPOSE_ADD_SESSION_NODE", "MARK_COVERED", "PROPOSE_NO_OP", "STOP"]:
                    action_valid[ACTION_TO_ID[action_name]] = 0
            elif progress["has_attachments"] and not progress["attach_complete"]:
                for action_name in ["CREATE_SESSION_NODE", "LINK_SESSION_NODES", "PROPOSE_ADD_SESSION_NODE", "MARK_COVERED", "PROPOSE_NO_OP", "STOP"]:
                    action_valid[ACTION_TO_ID[action_name]] = 0
            elif progress["has_adds"] and not progress["add_complete"]:
                for action_name in ["CREATE_SESSION_NODE", "LINK_SESSION_NODES", "PROPOSE_LINK_SESSION_TO_MEMORY", "MARK_COVERED", "PROPOSE_NO_OP", "STOP"]:
                    action_valid[ACTION_TO_ID[action_name]] = 0
            else:
                for action_name in ["CREATE_SESSION_NODE", "LINK_SESSION_NODES", "PROPOSE_LINK_SESSION_TO_MEMORY", "PROPOSE_ADD_SESSION_NODE", "MARK_COVERED", "PROPOSE_NO_OP"]:
                    action_valid[ACTION_TO_ID[action_name]] = 0

    if not progress["final_complete"]:
        action_valid[ACTION_TO_ID["STOP"]] = 0

    if int(obs.get("budget_left", 0)) <= 1:
        action_valid[ACTION_TO_ID["STOP"]] = 1

    return action_valid, progress, goal


def candidate_action_counts(candidates: Sequence[Mapping[str, Any]]) -> Dict[str, int]:
    counts = Counter(str(c.get("action", "")) for c in candidates)
    return dict(counts)


def exhaustive_valid_action_counts(
    row: Mapping[str, Any],
    env: NGRV1Env,
    *,
    eval_mode: str,
    attach_evidence_threshold: float,
) -> Dict[str, Any]:
    obs = env.observe()
    masks = env.valid_action_mask()
    action_valid, progress, goal = runtime_action_valid(row, obs, masks, eval_mode=eval_mode)

    n_spans = len(obs.get("spans", []) or [])
    n_session = len(obs.get("session_nodes", []) or [])
    n_memory = len(obs.get("memory_nodes", []) or [])
    action_idxs = [i for i, v in enumerate(action_valid) if v]

    spans = list(range(n_spans))
    sessions = list(range(n_session))
    memories = list(range(n_memory))
    link_pairs = [(s, d) for s in sessions for d in sessions if s != d]
    attach_pairs = [(s, m) for s in sessions for m in memories]
    rels = list(V1_RELATIONS)
    nts = list(V1_NODE_TYPES)

    counts = Counter()
    seen = set()

    def add(cand: Dict[str, Any]) -> None:
        valid, _errors = env.validate_action(cand)
        if not valid:
            return
        key = json.dumps(cand, sort_keys=True)
        if key in seen:
            return
        seen.add(key)
        counts[str(cand.get("action", ""))] += 1

    for aid in action_idxs:
        name = ID_TO_ACTION[aid]
        if name == "CREATE_SESSION_NODE":
            for sp in spans:
                for nt in nts:
                    add({"action": name, "span_indices": [sp], "node_type": nt})
        elif name == "LINK_SESSION_NODES":
            for s, d in link_pairs:
                for rel in rels:
                    add({"action": name, "src_session_idx": s, "dst_session_idx": d, "relation": rel})
        elif name == "MARK_COVERED":
            for s, m in attach_pairs:
                if is_noop_goal(goal) and not evidence_ok_for_cover(obs, s, m, attach_evidence_threshold):
                    continue
                add({"action": name, "session_idx": s, "memory_idx": m})
        elif name == "PROPOSE_ADD_SESSION_NODE":
            if is_noop_goal(goal):
                continue
            for s in sessions:
                add({"action": name, "session_idx": s})
        elif name == "PROPOSE_LINK_SESSION_TO_MEMORY":
            if not (goal.get("memory_attachments") or []):
                continue
            for s, m in attach_pairs:
                if not evidence_ok_for_attach(obs, s, m, attach_evidence_threshold):
                    continue
                for rel in rels:
                    add({"action": name, "session_idx": s, "memory_idx": m, "relation": rel})
        elif name == "PROPOSE_NO_OP":
            add({"action": name})
        elif name == "STOP":
            add({"action": name})

    return {
        "total": int(sum(counts.values())),
        "action_counts": dict(counts),
        "progress": progress,
    }


def dead_end_probe_step(
    *,
    row: Mapping[str, Any],
    env: NGRV1Env,
    out: Mapping[str, torch.Tensor],
    eval_mode: str,
    beam_per_head: int,
    relation_k: int,
    node_type_k: int,
    attach_evidence_threshold: float,
    policy_only_phase_topk: int,
    policy_only_protect_link_phase: bool,
    policy_only_protect_structural_phase: bool,
    policy_only_link_pair_k: int,
    policy_only_cover_pair_k: int,
    policy_only_link_all_relations: bool,
    policy_only_soften_topk_on_create_cover: bool,
    pred_phase_top1: str,
) -> Dict[str, Any]:
    obs = env.observe()
    progress = goal_progress_env(row, obs)
    structural_phase = structurally_required_phase(progress, obs)

    no_topk_candidates = enumerate_valid_tuples(
        row,
        env,
        out,
        eval_mode=eval_mode,
        beam_per_head=beam_per_head,
        relation_k=relation_k,
        node_type_k=node_type_k,
        attach_evidence_threshold=attach_evidence_threshold,
        policy_only_phase_topk=0 if eval_mode == "policy_only" else policy_only_phase_topk,
        policy_only_protect_link_phase=policy_only_protect_link_phase,
        policy_only_protect_structural_phase=policy_only_protect_structural_phase,
        policy_only_link_pair_k=policy_only_link_pair_k,
        policy_only_cover_pair_k=policy_only_cover_pair_k,
        policy_only_link_all_relations=policy_only_link_all_relations,
        policy_only_soften_topk_on_create_cover=policy_only_soften_topk_on_create_cover,
    )
    no_topk_counts = candidate_action_counts(no_topk_candidates)
    exhaustive = exhaustive_valid_action_counts(
        row,
        env,
        eval_mode=eval_mode,
        attach_evidence_threshold=attach_evidence_threshold,
    )
    exhaustive_counts = dict(exhaustive.get("action_counts", {}))
    required_action = PHASE_TO_ACTION.get(structural_phase, "")
    required_exhaustive_count = int(exhaustive_counts.get(required_action, 0))

    if eval_mode == "policy_only" and policy_only_phase_topk > 0 and no_topk_candidates:
        reason = "phase_topk_pruned_all_candidates"
    elif required_exhaustive_count > 0 and not no_topk_candidates:
        reason = "scorer_beam_missed_candidates"
    elif structural_phase == "link" and required_exhaustive_count <= 0:
        reason = "no_session_pair_available"
    elif structural_phase in {"attach", "cover"} and required_exhaustive_count <= 0:
        reason = "no_memory_target_available"
    elif exhaustive.get("total", 0) > 0 and not no_topk_candidates:
        reason = "scorer_beam_missed_candidates"
    else:
        reason = "no_structural_candidate"

    return {
        "reason": reason,
        "pred_phase_top1": pred_phase_top1,
        "structural_phase": structural_phase,
        "required_action_family": required_action,
        "session_node_count": len(obs.get("session_nodes", []) or []),
        "session_edge_count": len(obs.get("session_edges", []) or []),
        "attachment_count": len(obs.get("attachments", []) or []),
        "memory_node_count": len(obs.get("memory_nodes", []) or []),
        "span_count": len(obs.get("spans", []) or []),
        "budget_left": int(obs.get("budget_left", 0)),
        "progress": dict(progress),
        "candidate_family_counts_no_topk": no_topk_counts,
        "candidate_total_no_topk": len(no_topk_candidates),
        "candidate_family_counts_exhaustive": exhaustive_counts,
        "candidate_total_exhaustive": int(exhaustive.get("total", 0)),
        "required_action_count_exhaustive": required_exhaustive_count,
    }


def enumerate_valid_tuples(
    row: Mapping[str, Any],
    env: NGRV1Env,
    out: Mapping[str, torch.Tensor],
    *,
    eval_mode: str,
    beam_per_head: int,
    relation_k: int,
    node_type_k: int,
    attach_evidence_threshold: float,
    policy_only_phase_topk: int,
    policy_only_protect_link_phase: bool,
    policy_only_protect_structural_phase: bool,
    policy_only_link_pair_k: int,
    policy_only_cover_pair_k: int,
    policy_only_link_all_relations: bool,
    policy_only_soften_topk_on_create_cover: bool,
) -> List[Dict[str, Any]]:
    obs = env.observe()
    masks = env.valid_action_mask()
    action_valid, progress, goal = runtime_action_valid(row, obs, masks, eval_mode=eval_mode)

    n_spans = len(obs.get("spans", []) or [])
    n_session = len(obs.get("session_nodes", []) or [])
    n_memory = len(obs.get("memory_nodes", []) or [])

    if eval_mode == "guided_exact_progress":
        exact_candidates = exact_progress_tuples(row, env, obs, progress)
        if exact_candidates:
            return exact_candidates

    policy_allowed_phases: set[str] = set()
    if eval_mode == "policy_only":
        policy_allowed_phases = policy_only_allowed_phases(
            obs,
            out,
            progress,
            topk=policy_only_phase_topk,
            protect_link_phase=policy_only_protect_link_phase,
            protect_structural_phase=policy_only_protect_structural_phase,
            soften_topk_on_create_cover=policy_only_soften_topk_on_create_cover,
        )

    action_idxs = [i for i, v in enumerate(action_valid) if v]

    span_idxs = topk_from_logits(out["span_logits"][0], masks.get("span", []), beam_per_head) if n_spans else []
    session_idxs = topk_from_logits(out["session_logits"][0], masks.get("session", []), beam_per_head) if n_session else []
    add_session_idxs = topk_from_logits(out["add_session_logits"][0], masks.get("session", []), beam_per_head) if n_session else []
    memory_idxs = topk_from_logits(out["memory_logits"][0], masks.get("memory", []), beam_per_head) if n_memory else []
    link_pair_budget = beam_per_head * beam_per_head
    if eval_mode == "policy_only" and "link" in policy_allowed_phases:
        link_pair_budget = max(link_pair_budget, policy_only_link_pair_k)
    cover_pair_budget = beam_per_head * beam_per_head
    if eval_mode == "policy_only" and progress["is_noop_goal"] and "cover" in policy_allowed_phases:
        if policy_only_cover_pair_k > 0:
            cover_pair_budget = max(cover_pair_budget, policy_only_cover_pair_k)
        else:
            cover_pair_budget = max(cover_pair_budget, n_session * n_memory)
    link_pairs = topk_pair_indices(out["link_pair_logits"][0], link_pair_budget) if n_session >= 2 else []
    attach_pairs = topk_pair_indices(out["attach_pair_logits"][0], beam_per_head * beam_per_head) if n_session and n_memory else []
    cover_pairs = topk_pair_indices(out["cover_pair_logits"][0], cover_pair_budget) if n_session and n_memory else []

    rel_idxs = topk_from_logits(out["relation_logits"][0], [1] * len(V1_RELATIONS), relation_k)
    nt_idxs = topk_from_logits(out["node_type_logits"][0], [1] * len(V1_NODE_TYPES), node_type_k)

    rels = [(ID_TO_REL.get(i, "related"), i) for i in rel_idxs]
    nts = [(ID_TO_NODE_TYPE.get(i, "concept"), i) for i in nt_idxs]

    candidates: List[Dict[str, Any]] = []

    def add(a: Dict[str, Any]) -> None:
        valid, _errors = env.validate_action(a)
        if valid:
            candidates.append(a)

    for aid in action_idxs:
        name = ID_TO_ACTION[aid]

        if name == "CREATE_SESSION_NODE":
            for sp in span_idxs:
                for nt, _ in nts:
                    add({"action": name, "span_indices": [sp], "node_type": nt})

        elif name == "LINK_SESSION_NODES":
            link_rels = rels
            if eval_mode == "policy_only" and "link" in policy_allowed_phases and policy_only_link_all_relations:
                link_rels = [(rel, rid) for rid, rel in enumerate(V1_RELATIONS)]
            for s, d in link_pairs:
                if s == d:
                    continue
                for rel, _ in link_rels:
                    add({"action": name, "src_session_idx": s, "dst_session_idx": d, "relation": rel})

        elif name == "MARK_COVERED":
            for s, m in cover_pairs:
                if is_noop_goal(goal) and not evidence_ok_for_cover(obs, s, m, attach_evidence_threshold):
                    continue
                add({"action": name, "session_idx": s, "memory_idx": m})

        elif name == "PROPOSE_ADD_SESSION_NODE":
            # Do not propose add in true no-op tasks.
            if is_noop_goal(goal):
                continue
            for s in add_session_idxs:
                add({"action": name, "session_idx": s})

        elif name == "PROPOSE_LINK_SESSION_TO_MEMORY":
            # Avoid attachment spam in pure long_decompose rows with no memory attachments.
            if not (goal.get("memory_attachments") or []):
                continue
            for s, m in attach_pairs:
                if not evidence_ok_for_attach(obs, s, m, attach_evidence_threshold):
                    continue
                for rel, _ in rels:
                    add({"action": name, "session_idx": s, "memory_idx": m, "relation": rel})

        elif name == "PROPOSE_LINK_MEMORY_TO_MEMORY":
            for src in memory_idxs:
                for dst in memory_idxs:
                    if src == dst:
                        continue
                    for rel, _ in rels:
                        add({"action": name, "src_memory_idx": src, "dst_memory_idx": dst, "relation": rel})

        elif name == "PROPOSE_NO_OP":
            add({"action": name})

        elif name == "STOP":
            add({"action": name})

    seen = set()
    out_c = []
    for c in candidates:
        if eval_mode == "policy_only" and policy_only_phase_topk > 0:
            action_phase = ACTION_TO_PHASE.get(str(c.get("action", "")), "unknown")
            if action_phase not in policy_allowed_phases:
                continue
        key = json.dumps(c, sort_keys=True)
        if key not in seen:
            seen.add(key)
            out_c.append(c)

    return out_c


def tuple_score(
    out: Mapping[str, torch.Tensor],
    tup: Mapping[str, Any],
    *,
    arg_weight: float,
    normalize_args: bool,
    stop_penalty: float,
    phase_guidance_weight: float,
    compat_guidance_weight: float,
) -> float:
    action = str(tup.get("action", ""))
    if action not in ACTION_TO_ID:
        return -1e9

    action_lp = logp(out["action_logits"][0], ACTION_TO_ID[action])
    args: List[float] = []

    if action == "CREATE_SESSION_NODE":
        spans = tup.get("span_indices", [])
        if spans:
            args.append(logp(out["span_logits"][0], int(spans[0])))
        nt = str(tup.get("node_type", "concept"))
        args.append(logp(out["node_type_logits"][0], V1_NODE_TYPES.index(nt) if nt in V1_NODE_TYPES else 0))

    elif action == "LINK_SESSION_NODES":
        args.append(logp_pair(out["link_pair_logits"][0], int(tup["src_session_idx"]), int(tup["dst_session_idx"])))
        rel = str(tup.get("relation", "related"))
        args.append(logp(out["relation_logits"][0], V1_RELATIONS.index(rel) if rel in V1_RELATIONS else 0))

    elif action == "MARK_COVERED":
        args.append(logp_pair(out["cover_pair_logits"][0], int(tup["session_idx"]), int(tup["memory_idx"])))

    elif action == "PROPOSE_ADD_SESSION_NODE":
        args.append(logp(out["add_session_logits"][0], int(tup["session_idx"])))

    elif action == "PROPOSE_LINK_SESSION_TO_MEMORY":
        args.append(logp_pair(out["attach_pair_logits"][0], int(tup["session_idx"]), int(tup["memory_idx"])))
        rel = str(tup.get("relation", "related"))
        args.append(logp(out["relation_logits"][0], V1_RELATIONS.index(rel) if rel in V1_RELATIONS else 0))

    elif action == "PROPOSE_LINK_MEMORY_TO_MEMORY":
        args.append(logp(out["memory_logits"][0], int(tup["src_memory_idx"])))
        args.append(logp(out["memory_logits"][0], int(tup["dst_memory_idx"])))
        rel = str(tup.get("relation", "related"))
        args.append(logp(out["relation_logits"][0], V1_RELATIONS.index(rel) if rel in V1_RELATIONS else 0))

    arg_score = 0.0
    if args:
        arg_score = sum(args) / len(args) if normalize_args else sum(args)

    phase_bonus = 0.0
    phase_name = ACTION_TO_PHASE.get(action)
    if phase_name is not None and phase_guidance_weight > 0.0:
        phase_bonus = phase_guidance_weight * logp(out["phase_logits"][0], PHASE_TO_ID[phase_name])

    compat_bonus = 0.0
    if phase_name is not None and compat_guidance_weight > 0.0:
        pred_phase = top_phase_names(out["phase_logits"][0], 1)[0]
        compat_bonus = compat_guidance_weight * (1.0 if pred_phase == phase_name else 0.0)

    score = action_lp + arg_weight * arg_score + phase_bonus + compat_bonus
    if action == "STOP":
        score -= stop_penalty
    return score


def all_valid_link_candidates(env: NGRV1Env) -> List[Dict[str, Any]]:
    obs = env.observe()
    n_session = len(obs.get("session_nodes", []) or [])
    candidates: List[Dict[str, Any]] = []
    seen = set()
    if n_session < 2:
        return candidates
    for src in range(n_session):
        for dst in range(n_session):
            if src == dst:
                continue
            for rel in V1_RELATIONS:
                cand = {
                    "action": "LINK_SESSION_NODES",
                    "src_session_idx": src,
                    "dst_session_idx": dst,
                    "relation": rel,
                }
                valid, _errors = env.validate_action(cand)
                if not valid:
                    continue
                key = json.dumps(cand, sort_keys=True)
                if key in seen:
                    continue
                seen.add(key)
                candidates.append(cand)
    return candidates


def rank_of_first_match(
    ranked: Sequence[Tuple[float, Dict[str, Any]]],
    predicate,
) -> Optional[int]:
    for idx, (_score, cand) in enumerate(ranked, start=1):
        if predicate(cand):
            return idx
    return None


def relation_rank_within_pair(
    ranked: Sequence[Tuple[float, Dict[str, Any]]],
    pair_to_rels: Mapping[Tuple[int, int], set[str]],
) -> Optional[int]:
    best_rank: Optional[int] = None
    for pair, gold_rels in pair_to_rels.items():
        pair_rank = 0
        for _score, cand in ranked:
            cand_pair = (int(cand.get("src_session_idx", -1)), int(cand.get("dst_session_idx", -1)))
            if cand_pair != pair:
                continue
            pair_rank += 1
            if str(cand.get("relation", "related")) in gold_rels:
                if best_rank is None or pair_rank < best_rank:
                    best_rank = pair_rank
                break
    return best_rank


def classify_link_probe_choice(
    action: Mapping[str, Any],
    *,
    runtime_gold_tuples: set[Tuple[int, int, str]],
    runtime_gold_pairs: set[Tuple[int, int]],
    runtime_gold_pair_to_rels: Mapping[Tuple[int, int], set[str]],
    no_gold_node_match: bool,
) -> Tuple[bool, Optional[str]]:
    if str(action.get("action", "")) == "LINK_SESSION_NODES":
        src = int(action.get("src_session_idx", -1))
        dst = int(action.get("dst_session_idx", -1))
        rel = str(action.get("relation", "related"))
        if (src, dst, rel) in runtime_gold_tuples:
            return True, None
        if (src, dst) in runtime_gold_pairs:
            return False, "wrong_relation"
        if (dst, src) in runtime_gold_pairs:
            return False, "wrong_direction"
        return False, "wrong_pair"

    if runtime_gold_tuples:
        return False, "wrong_pair"
    if no_gold_node_match:
        return False, "no_gold_node_match"
    return False, "gold_absent"


def link_rank_probe_step(
    *,
    row: Mapping[str, Any],
    env: NGRV1Env,
    out: Mapping[str, torch.Tensor],
    chosen_action: Mapping[str, Any],
    arg_weight: float,
    normalize_args: bool,
    stop_penalty: float,
    phase_guidance_weight: float,
    compat_guidance_weight: float,
) -> Dict[str, Any]:
    obs = env.observe()
    goal = row.get("goal", {}) or {}
    target = missing_gold_link_runtime_targets(obs, goal)
    runtime_gold_tuples = target["runtime_gold_tuples"]
    runtime_gold_pairs = target["runtime_gold_pairs"]
    runtime_gold_pair_to_rels = target["runtime_gold_pair_to_rels"]
    no_gold_node_match = bool(target["no_gold_node_match"])

    candidates = all_valid_link_candidates(env)
    ranked = sorted(
        [
            (
                tuple_score(
                    out,
                    cand,
                    arg_weight=arg_weight,
                    normalize_args=normalize_args,
                    stop_penalty=stop_penalty,
                    phase_guidance_weight=phase_guidance_weight,
                    compat_guidance_weight=compat_guidance_weight,
                ),
                cand,
            )
            for cand in candidates
        ],
        key=lambda x: x[0],
        reverse=True,
    )

    best_gold_link_rank = rank_of_first_match(
        ranked,
        lambda cand: (
            int(cand.get("src_session_idx", -1)),
            int(cand.get("dst_session_idx", -1)),
            str(cand.get("relation", "related")),
        ) in runtime_gold_tuples,
    )
    best_gold_pair_rank = rank_of_first_match(
        ranked,
        lambda cand: (
            int(cand.get("src_session_idx", -1)),
            int(cand.get("dst_session_idx", -1)),
        ) in runtime_gold_pairs,
    )
    best_gold_relation_rank = relation_rank_within_pair(ranked, runtime_gold_pair_to_rels)

    chosen_link_is_gold, chosen_wrong_reason = classify_link_probe_choice(
        chosen_action,
        runtime_gold_tuples=runtime_gold_tuples,
        runtime_gold_pairs=runtime_gold_pairs,
        runtime_gold_pair_to_rels=runtime_gold_pair_to_rels,
        no_gold_node_match=no_gold_node_match,
    )

    return {
        "gold_link_present": bool(best_gold_link_rank is not None),
        "best_gold_link_rank": best_gold_link_rank,
        "best_gold_pair_rank": best_gold_pair_rank,
        "best_gold_relation_rank": best_gold_relation_rank,
        "chosen_link_is_gold": bool(chosen_link_is_gold),
        "chosen_wrong_reason": chosen_wrong_reason,
        "link_candidate_count": len(candidates),
        "missing_gold_edge_count": len(target["missing_gold_edges"]),
        "missing_due_to_node_match_count": int(target["missing_due_to_node_match_count"]),
    }


def decode_tuple_beam(
    row,
    env,
    out,
    *,
    eval_mode,
    beam_per_head,
    relation_k,
    node_type_k,
    arg_weight,
    normalize_args,
    stop_penalty,
    attach_evidence_threshold,
    phase_guidance_weight,
    compat_guidance_weight,
    policy_only_phase_topk,
    policy_only_protect_link_phase,
    policy_only_protect_structural_phase,
    policy_only_link_pair_k,
    policy_only_cover_pair_k,
    policy_only_link_all_relations,
    policy_only_soften_topk_on_create_cover,
):
    candidates = enumerate_valid_tuples(
        row,
        env,
        out,
        eval_mode=eval_mode,
        beam_per_head=beam_per_head,
        relation_k=relation_k,
        node_type_k=node_type_k,
        attach_evidence_threshold=attach_evidence_threshold,
        policy_only_phase_topk=policy_only_phase_topk,
        policy_only_protect_link_phase=policy_only_protect_link_phase,
        policy_only_protect_structural_phase=policy_only_protect_structural_phase,
        policy_only_link_pair_k=policy_only_link_pair_k,
        policy_only_cover_pair_k=policy_only_cover_pair_k,
        policy_only_link_all_relations=policy_only_link_all_relations,
        policy_only_soften_topk_on_create_cover=policy_only_soften_topk_on_create_cover,
    )
    if not candidates:
        return {"action": "__NO_VALID_TUPLE__"}, {
            "candidate_count": 0,
            "fallback_mode": "none_dead_end",
            "forced_fallback": False,
        }

    scored = [(
        tuple_score(
            out,
            c,
            arg_weight=arg_weight,
            normalize_args=normalize_args,
            stop_penalty=stop_penalty,
            phase_guidance_weight=phase_guidance_weight,
            compat_guidance_weight=compat_guidance_weight,
        ),
        c,
    ) for c in candidates]
    scored.sort(key=lambda x: x[0], reverse=True)
    return scored[0][1], {
        "candidate_count": len(candidates),
        "fallback_mode": "beam",
        "best_score": float(scored[0][0]),
        "top_action_candidates": dict(Counter(str(c.get("action", "")) for _s, c in scored[:20])),
    }


@torch.no_grad()
def rollout_one(
    model,
    row,
    *,
    device,
    hash_dim,
    max_steps,
    weak_retrieval_threshold,
    eval_mode,
    beam_per_head,
    relation_k,
    node_type_k,
    arg_weight,
    normalize_args,
    stop_penalty,
    attach_evidence_threshold,
    phase_guidance_weight,
    compat_guidance_weight,
    policy_only_phase_topk,
    policy_only_protect_link_phase,
    policy_only_protect_structural_phase,
    policy_only_link_pair_k,
    policy_only_cover_pair_k,
    policy_only_link_all_relations,
    policy_only_soften_topk_on_create_cover,
    link_rank_probe,
    dead_end_probe,
):
    graph = MemoryGraph.load_json(str(row["graph_path"]))
    env = NGRV1Env(graph, NGRV1Config(max_steps=max_steps, weak_retrieval_threshold=weak_retrieval_threshold))
    env.reset(str(row["signal"]), task=row)

    invalid = repeated = 0
    seen = Counter()
    action_counts = Counter()
    repeat_action_counts = Counter()
    invalid_errors = Counter()
    candidate_counts = []
    phase_trace = []

    for _ in range(max_steps):
        obs_before = env.observe()
        batch = build_batch(env, hash_dim, device)
        out = model(batch)

        phase_logits = out["phase_logits"][0]
        phase_probs = torch.softmax(phase_logits, dim=-1)
        topk = min(3, phase_probs.numel())
        top_phase_ids = torch.topk(phase_probs, k=topk).indices.tolist()
        pred_phase_top1 = PHASES[int(top_phase_ids[0])] if top_phase_ids else "unknown"
        pred_phase_top3 = [PHASES[int(i)] for i in top_phase_ids]
        gold_phase = gold_phase_for_obs(row, obs_before)
        progress_before = goal_progress_env(row, obs_before)

        action, dbg = decode_tuple_beam(
            row,
            env,
            out,
            eval_mode=eval_mode,
            beam_per_head=beam_per_head,
            relation_k=relation_k,
            node_type_k=node_type_k,
            arg_weight=arg_weight,
            normalize_args=normalize_args,
            stop_penalty=stop_penalty,
            attach_evidence_threshold=attach_evidence_threshold,
            phase_guidance_weight=phase_guidance_weight,
            compat_guidance_weight=compat_guidance_weight,
            policy_only_phase_topk=policy_only_phase_topk,
            policy_only_protect_link_phase=policy_only_protect_link_phase,
            policy_only_protect_structural_phase=policy_only_protect_structural_phase,
            policy_only_link_pair_k=policy_only_link_pair_k,
            policy_only_cover_pair_k=policy_only_cover_pair_k,
            policy_only_link_all_relations=policy_only_link_all_relations,
            policy_only_soften_topk_on_create_cover=policy_only_soften_topk_on_create_cover,
        )
        candidate_counts.append(float(dbg.get("candidate_count", 0)))
        chosen_action_phase = ACTION_TO_PHASE.get(str(action.get("action", "")), "unknown")
        chosen_action_name = str(action.get("action", ""))
        phase_compatible = int(chosen_action_phase == gold_phase)
        stop_premature = int(chosen_action_name == "STOP" and not bool(progress_before.get("final_complete", False)))
        attach_before_edge_complete = int(
            chosen_action_name == "PROPOSE_LINK_SESSION_TO_MEMORY"
            and bool(progress_before.get("has_edges", False))
            and not bool(progress_before.get("edge_complete", False))
        )
        is_covered_task = bool(progress_before.get("is_noop_goal", False))
        all_cover_nodes_created = bool(is_covered_task and progress_before.get("create_complete", False))
        coverage_complete = bool(is_covered_task and progress_before.get("coverage_complete", False))
        noop_available = bool(
            is_covered_task
            and coverage_complete
            and not bool(obs_before.get("proposed_no_op", False))
        )
        noop_available_not_chosen = int(noop_available and chosen_action_name != "PROPOSE_NO_OP")
        dead_end = None
        if dead_end_probe and chosen_action_name == "__NO_VALID_TUPLE__":
            dead_end = dead_end_probe_step(
                row=row,
                env=env,
                out=out,
                eval_mode=eval_mode,
                beam_per_head=beam_per_head,
                relation_k=relation_k,
                node_type_k=node_type_k,
                attach_evidence_threshold=attach_evidence_threshold,
                policy_only_phase_topk=policy_only_phase_topk,
                policy_only_protect_link_phase=policy_only_protect_link_phase,
                policy_only_protect_structural_phase=policy_only_protect_structural_phase,
                policy_only_link_pair_k=policy_only_link_pair_k,
                policy_only_cover_pair_k=policy_only_cover_pair_k,
                policy_only_link_all_relations=policy_only_link_all_relations,
                policy_only_soften_topk_on_create_cover=policy_only_soften_topk_on_create_cover,
                pred_phase_top1=pred_phase_top1,
            )

        action_counts[str(action.get("action", ""))] += 1
        key = json.dumps(action, sort_keys=True)
        if seen[key] > 0:
            repeated += 1
            repeat_action_counts[str(action.get("action", ""))] += 1
        seen[key] += 1

        phase_trace.append({
            "step": len(phase_trace),
            "gold_phase_if_available": gold_phase,
            "pred_phase_top1": pred_phase_top1,
            "pred_phase_top3": pred_phase_top3,
            "chosen_action": chosen_action_name,
            "chosen_action_phase": chosen_action_phase,
            "phase_compatible": phase_compatible,
            "stop_premature": stop_premature,
            "attach_before_edge_complete": attach_before_edge_complete,
            "is_covered_task": is_covered_task,
            "all_cover_nodes_created": all_cover_nodes_created,
            "coverage_complete": coverage_complete,
            "noop_available": noop_available,
            "noop_available_not_chosen": noop_available_not_chosen,
            "phase_entropy": phase_entropy_from_logits(phase_logits),
            "candidate_count": dbg.get("candidate_count", 0),
            "fallback_mode": dbg.get("fallback_mode", ""),
            "top_action_candidates": dbg.get("top_action_candidates", {}),
            "dead_end_probe": dead_end,
        })
        if link_rank_probe and gold_phase == "link":
            phase_trace[-1]["link_rank_probe"] = link_rank_probe_step(
                row=row,
                env=env,
                out=out,
                chosen_action=action,
                arg_weight=arg_weight,
                normalize_args=normalize_args,
                stop_penalty=stop_penalty,
                phase_guidance_weight=phase_guidance_weight,
                compat_guidance_weight=compat_guidance_weight,
            )

        _obs, _reward, done, info = env.step(action)
        if not info.get("valid", False):
            invalid += 1
            for err in info.get("errors", []):
                invalid_errors[str(err)] += 1

        if done:
            break

    traj = env.serialize_trajectory()
    cp = traj["commit_plan"]
    goal = row.get("goal", {}) or {}

    node = score_session_nodes(cp, goal)
    pred_to_gold = node.get("pred_to_gold", {})

    return {
        "eval_mode": eval_mode,
        "id": row.get("id"),
        "task_type": row.get("task_type", "unknown"),
        "steps": len(traj.get("steps", [])),
        "invalid": invalid,
        "repeated": repeated,
        "early_stop": int(len(traj.get("steps", [])) <= 1),
        "session_node": node,
        "session_edge": score_session_edges(cp, goal, pred_to_gold),
        "attachment": score_attachments(cp, goal, pred_to_gold),
        "commit": score_commit(cp, goal, pred_to_gold),
        "no_op_ok": score_no_op(cp, goal, str(row.get("task_type", ""))),
        "action_counts": dict(action_counts),
        "repeat_action_counts": dict(repeat_action_counts),
        "invalid_errors": dict(invalid_errors),
        "avg_tuple_candidates": mean(candidate_counts),
        "phase_trace": phase_trace,
        "phase_entropy": mean([float(x.get("phase_entropy", 0.0)) for x in phase_trace]) if phase_trace else 0.0,
        "trajectory": traj,
    }


def aggregate(results):
    by_task = defaultdict(list)
    for r in results:
        by_task[str(r.get("task_type", "unknown"))].append(r)

    def agg(group):
        noops = [r["no_op_ok"] for r in group if r.get("no_op_ok") is not None]
        steps_total = sum(float(r.get("steps", 0)) for r in group)
        action_counts = Counter()
        repeat_counts = Counter()
        invalid_errors = Counter()
        phase_confusion = defaultdict(Counter)
        action_by_pred_phase = defaultdict(Counter)
        task_by_pred_phase = defaultdict(Counter)
        phase_entropies = []
        phase_compatible_hits = 0
        phase_step_total = 0
        premature_stop_count = 0
        attach_before_edge_complete_count = 0
        noop_available_not_chosen_count = 0
        covered_rows = 0
        covered_reaches_cover_complete = 0
        covered_reaches_noop_available = 0
        covered_noop_chosen_when_available = 0
        covered_premature_stop_count = 0
        covered_create_after_all_nodes_present_count = 0
        covered_link_on_noop_goal_count = 0
        dead_end_steps = 0
        dead_end_reasons = Counter()
        dead_end_by_pred_phase = Counter()
        dead_end_by_structural_phase = Counter()
        dead_end_no_topk_action_counts = Counter()
        dead_end_exhaustive_action_counts = Counter()
        link_probe_steps = 0
        link_gold_present = 0
        link_gold_top1 = 0
        link_gold_top5 = 0
        link_pair_top1 = 0
        link_pair_top5 = 0
        link_relation_rank1 = 0
        link_relation_denom = 0
        chosen_link_gold = 0
        wrong_pair = 0
        wrong_direction = 0
        wrong_relation = 0
        gold_absent = 0
        no_gold_node_match = 0

        for r in group:
            action_counts.update(r.get("action_counts", {}))
            repeat_counts.update(r.get("repeat_action_counts", {}))
            invalid_errors.update(r.get("invalid_errors", {}))
            task_name = str(r.get("task_type", "unknown"))
            if task_name == "covered_long_signal":
                covered_rows += 1
                steps_for_row = r.get("phase_trace", []) or []
                if any(bool(step.get("coverage_complete", False)) for step in steps_for_row):
                    covered_reaches_cover_complete += 1
                if any(bool(step.get("noop_available", False)) for step in steps_for_row):
                    covered_reaches_noop_available += 1
                if any(bool(step.get("noop_available", False)) and str(step.get("chosen_action", "")) == "PROPOSE_NO_OP" for step in steps_for_row):
                    covered_noop_chosen_when_available += 1
            for step in r.get("phase_trace", []) or []:
                pred = str(step.get("pred_phase_top1", "unknown"))
                gold = str(step.get("gold_phase_if_available", "unknown"))
                action = str(step.get("chosen_action", "unknown"))
                phase_confusion[pred][gold] += 1
                action_by_pred_phase[pred][action] += 1
                task_by_pred_phase[pred][str(r.get("task_type", "unknown"))] += 1
                phase_entropies.append(float(step.get("phase_entropy", 0.0)))
                phase_compatible_hits += int(step.get("phase_compatible", 0))
                phase_step_total += 1
                premature_stop_count += int(step.get("stop_premature", 0))
                attach_before_edge_complete_count += int(step.get("attach_before_edge_complete", 0))
                noop_available_not_chosen_count += int(step.get("noop_available_not_chosen", 0))
                if task_name == "covered_long_signal":
                    covered_premature_stop_count += int(step.get("stop_premature", 0))
                    covered_create_after_all_nodes_present_count += int(
                        bool(step.get("all_cover_nodes_created", False))
                        and action == "CREATE_SESSION_NODE"
                    )
                    covered_link_on_noop_goal_count += int(action == "LINK_SESSION_NODES")
                dead_end = step.get("dead_end_probe")
                if dead_end:
                    dead_end_steps += 1
                    dead_end_reasons[str(dead_end.get("reason", "unknown"))] += 1
                    dead_end_by_pred_phase[pred] += 1
                    dead_end_by_structural_phase[str(dead_end.get("structural_phase", "unknown"))] += 1
                    for action_name, count in (dead_end.get("candidate_family_counts_no_topk", {}) or {}).items():
                        dead_end_no_topk_action_counts[str(action_name)] += int(count)
                    for action_name, count in (dead_end.get("candidate_family_counts_exhaustive", {}) or {}).items():
                        dead_end_exhaustive_action_counts[str(action_name)] += int(count)
                probe = step.get("link_rank_probe")
                if probe:
                    link_probe_steps += 1
                    best_gold_link_rank = probe.get("best_gold_link_rank")
                    best_gold_pair_rank = probe.get("best_gold_pair_rank")
                    best_gold_relation_rank = probe.get("best_gold_relation_rank")
                    if bool(probe.get("gold_link_present", False)):
                        link_gold_present += 1
                    if best_gold_link_rank == 1:
                        link_gold_top1 += 1
                    if isinstance(best_gold_link_rank, int) and best_gold_link_rank <= 5:
                        link_gold_top5 += 1
                    if best_gold_pair_rank == 1:
                        link_pair_top1 += 1
                    if isinstance(best_gold_pair_rank, int) and best_gold_pair_rank <= 5:
                        link_pair_top5 += 1
                    if best_gold_relation_rank is not None:
                        link_relation_denom += 1
                        if best_gold_relation_rank == 1:
                            link_relation_rank1 += 1
                    if bool(probe.get("chosen_link_is_gold", False)):
                        chosen_link_gold += 1
                    reason = str(probe.get("chosen_wrong_reason", "") or "")
                    if reason == "wrong_pair":
                        wrong_pair += 1
                    elif reason == "wrong_direction":
                        wrong_direction += 1
                    elif reason == "wrong_relation":
                        wrong_relation += 1
                    elif reason == "gold_absent":
                        gold_absent += 1
                    elif reason == "no_gold_node_match":
                        no_gold_node_match += 1

        return {
            "n": len(group),
            "avg_steps": mean([float(r.get("steps", 0)) for r in group]),
            "invalid_action_rate": sum(float(r.get("invalid", 0)) for r in group) / max(steps_total, 1.0),
            "repeated_action_rate": sum(float(r.get("repeated", 0)) for r in group) / max(steps_total, 1.0),
            "early_stop_rate": mean([float(r.get("early_stop", 0)) for r in group]),
            "session_node_f1": mean([float(r["session_node"]["f1"]) for r in group]),
            "session_edge_f1": mean([float(r["session_edge"]["f1"]) for r in group]),
            "memory_attachment_f1": mean([float(r["attachment"]["f1"]) for r in group]),
            "commit_f1": mean([float(r["commit"]["f1"]) for r in group]),
            "no_op_accuracy": mean([float(x) for x in noops]) if noops else None,
            "avg_tuple_candidates": mean([float(r.get("avg_tuple_candidates", 0.0)) for r in group]),
            "phase_entropy": mean(phase_entropies) if phase_entropies else 0.0,
            "phase_compatible_rate": (phase_compatible_hits / phase_step_total) if phase_step_total else 0.0,
            "premature_stop_count": int(premature_stop_count),
            "attach_before_edge_complete_count": int(attach_before_edge_complete_count),
            "noop_available_not_chosen_count": int(noop_available_not_chosen_count),
            "covered_reaches_cover_complete_rate": (covered_reaches_cover_complete / covered_rows) if covered_rows else 0.0,
            "covered_reaches_noop_available_rate": (covered_reaches_noop_available / covered_rows) if covered_rows else 0.0,
            "covered_noop_chosen_when_available_rate": (covered_noop_chosen_when_available / covered_rows) if covered_rows else 0.0,
            "covered_premature_stop_count": int(covered_premature_stop_count),
            "covered_create_after_all_nodes_present_count": int(covered_create_after_all_nodes_present_count),
            "covered_link_on_noop_goal_count": int(covered_link_on_noop_goal_count),
            "phase_confusion": {pred: dict(cnt) for pred, cnt in sorted(phase_confusion.items())},
            "action_by_pred_phase": {pred: dict(cnt.most_common()) for pred, cnt in sorted(action_by_pred_phase.items())},
            "task_by_pred_phase": {pred: dict(cnt.most_common()) for pred, cnt in sorted(task_by_pred_phase.items())},
            "dead_end_probe": {
                "steps": int(dead_end_steps),
                "step_rate": (dead_end_steps / steps_total) if steps_total else 0.0,
                "reason_counts": dict(dead_end_reasons.most_common()),
                "pred_phase_counts": dict(dead_end_by_pred_phase.most_common()),
                "structural_phase_counts": dict(dead_end_by_structural_phase.most_common()),
                "candidate_family_counts_no_topk": dict(dead_end_no_topk_action_counts.most_common()),
                "candidate_family_counts_exhaustive": dict(dead_end_exhaustive_action_counts.most_common()),
            },
            "action_counts": dict(action_counts.most_common()),
            "repeat_action_counts": dict(repeat_counts.most_common()),
            "invalid_errors": dict(invalid_errors.most_common(10)),
            "link_rank_probe": {
                "steps": int(link_probe_steps),
                "link_gold_present_rate": (link_gold_present / link_probe_steps) if link_probe_steps else None,
                "link_gold_top1_rate": (link_gold_top1 / link_probe_steps) if link_probe_steps else None,
                "link_gold_top5_rate": (link_gold_top5 / link_probe_steps) if link_probe_steps else None,
                "link_pair_top1_rate": (link_pair_top1 / link_probe_steps) if link_probe_steps else None,
                "link_pair_top5_rate": (link_pair_top5 / link_probe_steps) if link_probe_steps else None,
                "link_relation_accuracy_when_pair_correct": (link_relation_rank1 / link_relation_denom) if link_relation_denom else None,
                "chosen_link_gold_rate": (chosen_link_gold / link_probe_steps) if link_probe_steps else None,
                "wrong_pair_count": int(wrong_pair),
                "wrong_direction_count": int(wrong_direction),
                "wrong_relation_count": int(wrong_relation),
                "gold_absent_count": int(gold_absent),
                "no_gold_node_match_count": int(no_gold_node_match),
            },
        }

    return {
        "overall": agg(results),
        "by_task": {task: agg(group) for task, group in sorted(by_task.items())},
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--val-jsonl", required=True)
    ap.add_argument("--max-rollout-rows", type=int, default=200)
    ap.add_argument("--max-steps", type=int, default=12)
    ap.add_argument("--weak-retrieval-threshold", type=float, default=0.12)
    ap.add_argument("--beam-per-head", type=int, default=4)
    ap.add_argument("--relation-k", type=int, default=3)
    ap.add_argument("--node-type-k", type=int, default=2)
    ap.add_argument("--arg-weight", type=float, default=1.0)
    ap.add_argument("--no-normalize-args", action="store_true")
    ap.add_argument("--stop-penalty", type=float, default=0.0)
    ap.add_argument("--attach-evidence-threshold", type=float, default=0.08)
    ap.add_argument("--eval-modes", default="guided_exact_progress,phase_guided,policy_only")
    ap.add_argument("--phase-guidance-weight", type=float, default=0.75)
    ap.add_argument("--policy-only-phase-prior", type=float, default=0.0)
    ap.add_argument("--policy-only-compat-weight", type=float, default=0.0)
    ap.add_argument("--policy-only-phase-topk", type=int, default=0)
    ap.add_argument("--policy-only-protect-link-phase", action="store_true")
    ap.add_argument("--policy-only-protect-structural-phase", action="store_true")
    ap.add_argument("--policy-only-link-pair-k", type=int, default=64)
    ap.add_argument("--policy-only-cover-pair-k", type=int, default=256)
    ap.add_argument("--policy-only-link-all-relations", action="store_true")
    ap.add_argument("--policy-only-soften-topk-on-create-cover", action="store_true")
    ap.add_argument("--link-rank-probe", action="store_true")
    ap.add_argument("--dead-end-probe", action="store_true")
    ap.add_argument("--save-rollouts-jsonl", default="")
    ap.add_argument("--cpu", action="store_true")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")

    ckpt = torch.load(args.checkpoint, map_location=device)
    cfg = ckpt.get("args", {})

    hash_dim = int(cfg.get("hash_dim", 512))
    hidden_dim = int(cfg.get("hidden_dim", 256))

    model = NGRV1PolicyNet(hash_dim=hash_dim, hidden_dim=hidden_dim).to(device)
    load_result = model.load_state_dict(ckpt["model"], strict=False)
    model.eval()

    rows = read_jsonl(args.val_jsonl)[: args.max_rollout_rows]

    modes = [m.strip() for m in str(args.eval_modes or "").split(",") if m.strip()]
    all_results: List[Dict[str, Any]] = []
    mode_results: Dict[str, Any] = {}

    for eval_mode in modes:
        results = []
        for row in rows:
            try:
                phase_weight = args.phase_guidance_weight if eval_mode == "phase_guided" else (args.policy_only_phase_prior if eval_mode == "policy_only" else 0.0)
                compat_weight = args.policy_only_compat_weight if eval_mode == "policy_only" else 0.0
                phase_topk = args.policy_only_phase_topk if eval_mode == "policy_only" else 0
                results.append(rollout_one(
                    model,
                    row,
                    device=device,
                    hash_dim=hash_dim,
                    max_steps=args.max_steps,
                    weak_retrieval_threshold=args.weak_retrieval_threshold,
                    eval_mode=eval_mode,
                    beam_per_head=args.beam_per_head,
                    relation_k=args.relation_k,
                    node_type_k=args.node_type_k,
                    arg_weight=args.arg_weight,
                    normalize_args=not args.no_normalize_args,
                    stop_penalty=args.stop_penalty,
                    attach_evidence_threshold=args.attach_evidence_threshold,
                    phase_guidance_weight=phase_weight,
                    compat_guidance_weight=compat_weight,
                    policy_only_phase_topk=phase_topk,
                    policy_only_protect_link_phase=args.policy_only_protect_link_phase,
                    policy_only_protect_structural_phase=args.policy_only_protect_structural_phase,
                    policy_only_link_pair_k=args.policy_only_link_pair_k,
                    policy_only_cover_pair_k=args.policy_only_cover_pair_k,
                    policy_only_link_all_relations=args.policy_only_link_all_relations,
                    policy_only_soften_topk_on_create_cover=bool(args.policy_only_soften_topk_on_create_cover),
                    link_rank_probe=bool(args.link_rank_probe),
                    dead_end_probe=bool(args.dead_end_probe),
                ))
            except Exception as exc:
                results.append({
                    "eval_mode": eval_mode,
                    "id": row.get("id"),
                    "task_type": row.get("task_type", "unknown"),
                    "error": str(exc),
                    "steps": 0,
                    "invalid": 1,
                    "repeated": 0,
                    "early_stop": 1,
                    "session_node": {"f1": 0.0},
                    "session_edge": {"f1": 0.0},
                    "attachment": {"f1": 0.0},
                    "commit": {"f1": 0.0},
                    "no_op_ok": None,
                    "action_counts": {},
                    "repeat_action_counts": {},
                    "invalid_errors": {str(exc): 1},
                    "avg_tuple_candidates": 0.0,
                    "phase_trace": [],
                    "trajectory": {},
                })
        all_results.extend(results)
        mode_results[eval_mode] = {
            "decoder": f"v1a6_{eval_mode}_tuple_beam",
            "decoder_config": {
                "eval_mode": eval_mode,
                "beam_per_head": args.beam_per_head,
                "relation_k": args.relation_k,
                "node_type_k": args.node_type_k,
                "arg_weight": args.arg_weight,
                "phase_guidance_weight": args.phase_guidance_weight if eval_mode == "phase_guided" else 0.0,
                "policy_only_phase_prior": args.policy_only_phase_prior if eval_mode == "policy_only" else 0.0,
                "policy_only_compat_weight": args.policy_only_compat_weight if eval_mode == "policy_only" else 0.0,
                "policy_only_phase_topk": args.policy_only_phase_topk if eval_mode == "policy_only" else 0,
                "policy_only_protect_link_phase": bool(args.policy_only_protect_link_phase) if eval_mode == "policy_only" else False,
                "policy_only_protect_structural_phase": bool(args.policy_only_protect_structural_phase) if eval_mode == "policy_only" else False,
                "policy_only_link_pair_k": args.policy_only_link_pair_k if eval_mode == "policy_only" else 0,
                "policy_only_cover_pair_k": args.policy_only_cover_pair_k if eval_mode == "policy_only" else 0,
                "policy_only_link_all_relations": bool(args.policy_only_link_all_relations) if eval_mode == "policy_only" else False,
                "policy_only_soften_topk_on_create_cover": bool(args.policy_only_soften_topk_on_create_cover) if eval_mode == "policy_only" else False,
                "link_rank_probe": bool(args.link_rank_probe),
                "dead_end_probe": bool(args.dead_end_probe),
                "normalize_args": not args.no_normalize_args,
                "stop_penalty": args.stop_penalty,
                "attach_evidence_threshold": args.attach_evidence_threshold,
                "noop_gate": "goal_noop_and_coverage_complete",
                "stop_gate": "final_commit_complete",
            },
            "rollout_metrics": aggregate(results),
            "rollout_rows": len(results),
        }

    out = {
        "checkpoint": args.checkpoint,
        "val_jsonl": args.val_jsonl,
        "device": str(device),
        "eval_modes": modes,
        "phase_guidance_weight": args.phase_guidance_weight,
        "checkpoint_load": {
            "missing_keys": list(load_result.missing_keys),
            "unexpected_keys": list(load_result.unexpected_keys),
        },
        "mode_results": mode_results,
    }

    print(json.dumps(out, indent=2, ensure_ascii=False))

    if args.save_rollouts_jsonl:
        path = Path(args.save_rollouts_jsonl)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            for r in all_results:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
