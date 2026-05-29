from __future__ import annotations

"""
train_ngr_v1.py

NGR-v1a trainer for no-retrieval full-graph edit-program learning.

Current scope:
1. candidate-set CE over task-valid tuples
2. phase head and action-family auxiliary losses
3. link-pair auxiliary loss and reverse-direction penalty
4. covered-task create/cover/noop control losses
5. covered/noop validation metrics for clean-state supervision
"""

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from graph_core import MemoryGraph, lexical_overlap
from ngr_v1_env import (
    ACTION_TO_ID,
    REL_TO_ID,
    NODE_TYPE_TO_ID,
    V1_ACTIONS,
    V1_RELATIONS,
    V1_NODE_TYPES,
    split_signal_spans,
)
from ngr_v1_model import NGRV1PolicyNet, V1Batch, bow_hash


GLOBAL_DIM = 16
SESSION_SCALAR_DIM = 8
PHASES = ["create", "link", "attach", "add", "cover", "noop", "stop"]
PHASE_TO_ID = {name: i for i, name in enumerate(PHASES)}
PHASE_TO_ACTION = {
    "create": "CREATE_SESSION_NODE",
    "link": "LINK_SESSION_NODES",
    "attach": "PROPOSE_LINK_SESSION_TO_MEMORY",
    "add": "PROPOSE_ADD_SESSION_NODE",
    "cover": "MARK_COVERED",
    "noop": "PROPOSE_NO_OP",
    "stop": "STOP",
}
DEFAULT_PHASE_ACTION_AUX_WEIGHTS = {
    "create": 1.0,
    "link": 1.0,
    "attach": 1.0,
    "add": 1.0,
    "cover": 1.0,
    "noop": 1.0,
    "stop": 1.0,
}


@dataclass
class ProgressBatch:
    signal_bow: torch.Tensor
    span_bow: torch.Tensor
    span_mask: torch.Tensor
    memory_bow: torch.Tensor
    memory_scalar: torch.Tensor
    memory_mask: torch.Tensor
    session_bow: torch.Tensor
    session_scalar: torch.Tensor
    session_mask: torch.Tensor
    action_hist: torch.Tensor
    global_scalar: torch.Tensor
    allowed_next: List[List[Dict[str, Any]]]
    candidate_next: List[List[Dict[str, Any]]]
    task_type: List[str]
    goals: List[Mapping[str, Any]]
    states: List[Mapping[str, Any]]
    y_value: torch.Tensor
    y_phase: torch.Tensor


def read_jsonl(path: str | Path) -> List[Dict[str, Any]]:
    rows = []
    with Path(path).open("r", encoding="utf-8-sig") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def pad_stack(xs: Sequence[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    max_n = max(x.size(0) for x in xs)
    dim = xs[0].size(-1)
    out = torch.zeros(len(xs), max_n, dim, dtype=xs[0].dtype)
    mask = torch.zeros(len(xs), max_n, dtype=torch.bool)
    for i, x in enumerate(xs):
        n = x.size(0)
        out[i, :n] = x
        mask[i, :n] = True
    return out, mask


def to_device(batch: ProgressBatch, device: torch.device) -> ProgressBatch:
    return ProgressBatch(
        signal_bow=batch.signal_bow.to(device),
        span_bow=batch.span_bow.to(device),
        span_mask=batch.span_mask.to(device),
        memory_bow=batch.memory_bow.to(device),
        memory_scalar=batch.memory_scalar.to(device),
        memory_mask=batch.memory_mask.to(device),
        session_bow=batch.session_bow.to(device),
        session_scalar=batch.session_scalar.to(device),
        session_mask=batch.session_mask.to(device),
        action_hist=batch.action_hist.to(device),
        global_scalar=batch.global_scalar.to(device),
        allowed_next=batch.allowed_next,
        candidate_next=batch.candidate_next,
        task_type=batch.task_type,
        goals=batch.goals,
        states=batch.states,
        y_value=batch.y_value.to(device),
        y_phase=batch.y_phase.to(device),
    )


def action_hist_vector(action_history: Sequence[Mapping[str, Any]]) -> torch.Tensor:
    v = torch.zeros(len(V1_ACTIONS), dtype=torch.float32)
    for a in action_history or []:
        aid = ACTION_TO_ID.get(str(a.get("action", "")))
        if aid is not None:
            v[aid] += 1.0
    if v.sum() > 0:
        v = v / v.sum().clamp_min(1.0)
    return v


def global_scalar_from_state(state: Mapping[str, Any], span_count: int) -> torch.Tensor:
    session_nodes = state.get("session_nodes", []) or []
    session_edges = state.get("session_edges", []) or []
    attachments = state.get("attachments", []) or []
    proposed_adds = state.get("proposed_adds", []) or []
    covered = state.get("covered", []) or []
    memory_nodes = state.get("memory_node_ids", []) or []
    actions = state.get("action_history", []) or []

    hist = action_hist_vector(actions)

    vals = [
        len(session_nodes) / 16.0,
        len(session_edges) / 32.0,
        len(attachments) / 32.0,
        len(proposed_adds) / 16.0,
        len(covered) / 16.0,
        1.0 if state.get("proposed_no_op") else 0.0,
        len(memory_nodes) / 512.0,
        span_count / 32.0,
        len(actions) / 12.0,
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


def goal_session_name_map(goal: Mapping[str, Any]) -> Dict[str, Mapping[str, Any]]:
    out: Dict[str, Mapping[str, Any]] = {}
    for i, s in enumerate(goal.get("session_nodes", []) or []):
        out[str(s.get("name", f"s{i}"))] = s
    return out


def tuple_key(t: Mapping[str, Any]) -> tuple:
    """
    Canonical key for comparing allowed tuples against candidates.
    Only operative fields are used.
    """
    action = str(t.get("action", ""))

    if action == "CREATE_SESSION_NODE":
        # Data rows use span_idx. Runtime can use span_indices. Normalize to span_idx.
        if "span_idx" in t:
            sp = int(t.get("span_idx", 0))
        else:
            inds = t.get("span_indices", []) or [0]
            sp = int(inds[0])
        return (action, sp, str(t.get("node_type", "concept")))

    if action == "LINK_SESSION_NODES":
        return (
            action,
            int(t.get("src_session_idx", -1)),
            int(t.get("dst_session_idx", -1)),
            str(t.get("relation", "related")),
        )

    if action == "MARK_COVERED":
        return (
            action,
            int(t.get("session_idx", -1)),
            int(t.get("memory_idx", -1)),
        )

    if action == "PROPOSE_ADD_SESSION_NODE":
        return (
            action,
            int(t.get("session_idx", -1)),
        )

    if action == "PROPOSE_LINK_SESSION_TO_MEMORY":
        return (
            action,
            int(t.get("session_idx", -1)),
            int(t.get("memory_idx", -1)),
            str(t.get("relation", "related")),
        )

    if action == "PROPOSE_LINK_MEMORY_TO_MEMORY":
        return (
            action,
            int(t.get("src_memory_idx", -1)),
            int(t.get("dst_memory_idx", -1)),
            str(t.get("relation", "related")),
        )

    if action in {"PROPOSE_NO_OP", "STOP"}:
        return (action,)

    return (action, json.dumps(t, sort_keys=True))


def is_noop_goal(goal: Mapping[str, Any]) -> bool:
    return any(str(f.get("action", "")) == "no_op" for f in goal.get("final_commits", []) or [])


def state_sets(state: Mapping[str, Any]) -> Dict[str, Any]:
    session_nodes = state.get("session_nodes", []) or []
    name_to_idx = {str(s.get("name", "")): i for i, s in enumerate(session_nodes)}
    created_names = set(name_to_idx)

    linked = {
        (
            str(e.get("src_name", "")),
            str(e.get("dst_name", "")),
            str(e.get("relation", "related")),
        )
        for e in state.get("session_edges", []) or []
    }

    attachments = {
        (
            str(a.get("session_name", "")),
            str(a.get("memory_id", "")),
            str(a.get("relation", "related")),
        )
        for a in state.get("attachments", []) or []
    }

    covered = {
        (
            str(c.get("session_name", "")),
            str(c.get("memory_id", "")),
        )
        for c in state.get("covered", []) or []
    }

    proposed_adds = set(str(x) for x in state.get("proposed_adds", []) or [])
    memory_ids = list(state.get("memory_node_ids", []) or [])
    memory_idx = {m: i for i, m in enumerate(memory_ids)}

    return {
        "session_nodes": session_nodes,
        "created_names": created_names,
        "name_to_idx": name_to_idx,
        "linked": linked,
        "attachments": attachments,
        "covered": covered,
        "proposed_adds": proposed_adds,
        "memory_ids": memory_ids,
        "memory_idx": memory_idx,
    }


def coverage_complete(goal: Mapping[str, Any], state: Mapping[str, Any]) -> bool:
    ss = state_sets(state)
    covered = ss["covered"]
    covs = goal.get("covered_mappings", []) or []
    if not covs:
        return False
    for i, cov in enumerate(covs):
        name = f"covered_{i}"
        mem = str(cov.get("memory_id", ""))
        if (name, mem) not in covered:
            return False
    return True


def final_commit_complete(goal: Mapping[str, Any], state: Mapping[str, Any]) -> bool:
    """
    Stop is allowed only when the final edit program is complete.

    For no-op tasks:
      proposed_no_op must be true.

    For edit tasks:
      all session nodes exist,
      all required session edges exist,
      all memory attachments exist,
      all final add_node sessions are proposed.
    """
    if is_noop_goal(goal):
        return bool(state.get("proposed_no_op", False))

    ss = state_sets(state)

    # All goal session nodes exist.
    for name in goal_session_name_map(goal):
        if name not in ss["created_names"]:
            return False

    # All session edges exist.
    for e in goal.get("session_edges", []) or []:
        key = (str(e.get("src", "")), str(e.get("dst", "")), str(e.get("relation", "related")))
        if key not in ss["linked"]:
            return False

    # All session-memory attachments exist.
    for a in goal.get("memory_attachments", []) or []:
        key = (str(a.get("session", "")), str(a.get("memory_id", "")), str(a.get("relation", "related")))
        if key not in ss["attachments"]:
            return False

    # All final add_node commits are proposed.
    for f in goal.get("final_commits", []) or []:
        if str(f.get("action", "")) == "add_node":
            if str(f.get("session", "")) not in ss["proposed_adds"]:
                return False

    return True


def covered_all_nodes_created(goal: Mapping[str, Any], state: Mapping[str, Any]) -> bool:
    covs = goal.get("covered_mappings", []) or []
    if not covs:
        return False
    created = {str(s.get("name", "")) for s in state.get("session_nodes", []) or []}
    return all(f"covered_{i}" in created for i in range(len(covs)))


def covered_structural_phase(goal: Mapping[str, Any], state: Mapping[str, Any]) -> str:
    if not covered_all_nodes_created(goal, state):
        return "create"
    if not coverage_complete(goal, state):
        return "cover"
    if not bool(state.get("proposed_no_op", False)):
        return "noop"
    return "stop"


def build_task_candidates(row: Mapping[str, Any], state: Mapping[str, Any]) -> List[Dict[str, Any]]:
    """
    Candidate universe used by candidate-set CE.

    v1a.5.3 fix:
    - v1a.5.2 added useful same-action negatives, but polluted terminal
      phases with too many unrelated actions.
    - This version is phase-aware.

    Phase rules:
      create:
        CREATE_SESSION_NODE candidates only
        includes wrong spans / node types

      link:
        LINK_SESSION_NODES candidates with wrong pairs/directions/relations
        plus a small PROPOSE_ADD_SESSION_NODE shortcut distractor

      attach:
        PROPOSE_LINK_SESSION_TO_MEMORY candidates with wrong pairs/relations
        plus a small PROPOSE_ADD_SESSION_NODE shortcut distractor

      add:
        PROPOSE_ADD_SESSION_NODE for all created session nodes
        only same-action add negatives

      cover:
        MARK_COVERED candidates with wrong session-memory pairs

      noop:
        PROPOSE_NO_OP only, gated by no-op goal + coverage complete

      stop:
        STOP only, gated by final work complete

    This keeps hard negatives local to the current phase, instead of injecting
    random phase-breaking actions everywhere.
    """
    goal = row.get("goal", {}) or {}
    signal = str(row.get("signal", ""))
    phase = str(row.get("phase", ""))
    ss = state_sets(state)

    candidates: List[Dict[str, Any]] = []
    session_nodes = list(ss["session_nodes"])
    memory_ids = list(ss["memory_ids"])

    session_pair_key, session_memory_key = candidate_sort_key_by_text_pair(session_nodes, memory_ids)

    def dedupe_and_return(xs: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
        seen = set()
        out: List[Dict[str, Any]] = []
        for c in xs:
            key = tuple_key(c)
            if key not in seen:
                seen.add(key)
                out.append(dict(c))
        return out or [{"action": "STOP"}]

    def add_small_add_distractors(limit: int = 3) -> None:
        if is_noop_goal(goal):
            return
        added = 0
        for i, sn in enumerate(session_nodes):
            sname = str(sn.get("name", ""))
            if sname and sname in ss["proposed_adds"]:
                continue
            if sn.get("proposed_add"):
                continue
            candidates.append({
                "action": "PROPOSE_ADD_SESSION_NODE",
                "session_idx": i,
            })
            added += 1
            if added >= limit:
                break

    # ------------------------------------------------------------------
    # CREATE phase: same-action negatives only.
    # ------------------------------------------------------------------
    if phase == "create":
        spans = split_signal_spans(signal)
        used_span_idxs = set()
        for sn in session_nodes:
            for x in sn.get("span_indices", []) or []:
                try:
                    used_span_idxs.add(int(x))
                except Exception:
                    pass

        gold_create_types = []
        for name, spec in goal_session_name_map(goal).items():
            if name not in ss["created_names"]:
                gold_create_types.append(str(spec.get("node_type", "concept")))

        node_type_pool = []
        for nt in gold_create_types + ["concept", "claim", "fact", "summary"]:
            if nt not in node_type_pool:
                node_type_pool.append(nt)

        create_span_candidates = [i for i in range(len(spans)) if i not in used_span_idxs]
        for sp in create_span_candidates[:MAX_CREATE_CANDIDATES]:
            for nt in node_type_pool[:2]:
                candidates.append({
                    "action": "CREATE_SESSION_NODE",
                    "span_idx": sp,
                    "node_type": nt,
                })

        if is_noop_goal(goal):
            variant = str(row.get("state_variant", ""))
            if variant in {"cover_create_incomplete", "false_terminal_drift"}:
                if session_nodes and memory_ids:
                    sm_pairs = [(i, j) for i in range(len(session_nodes)) for j in range(len(memory_ids))]
                    sm_pairs.sort(key=session_memory_key)
                    covered_existing = ss["covered"]
                    for i, j in sm_pairs[:2]:
                        sname = str(session_nodes[i].get("name", ""))
                        mem = memory_ids[j]
                        if sname and (sname, mem) in covered_existing:
                            continue
                        candidates.append({
                            "action": "MARK_COVERED",
                            "session_idx": i,
                            "memory_idx": j,
                        })
                all_pairs = []
                for i in range(len(session_nodes)):
                    for j in range(len(session_nodes)):
                        if i != j:
                            all_pairs.append((i, j))
                all_pairs.sort(key=session_pair_key)
                for i, j in all_pairs[:2]:
                    candidates.append({
                        "action": "LINK_SESSION_NODES",
                        "src_session_idx": i,
                        "dst_session_idx": j,
                        "relation": "related",
                    })
                candidates.append({"action": "STOP"})
                candidates.append({"action": "PROPOSE_NO_OP"})

        return dedupe_and_return(candidates)

    # ------------------------------------------------------------------
    # LINK phase: link hard negatives + tiny add shortcut distractor.
    # ------------------------------------------------------------------
    if phase == "link":
        gold_link_rels = [str(e.get("relation", "related")) for e in goal.get("session_edges", []) or []]
        link_rel_pool = relation_candidates(*gold_link_rels)[:4]

        all_pairs = []
        for i in range(len(session_nodes)):
            for j in range(len(session_nodes)):
                if i == j:
                    continue
                all_pairs.append((i, j))
        all_pairs.sort(key=session_pair_key)

        linked_existing = ss["linked"]

        for i, j in all_pairs[:MAX_LINK_PAIR_CANDIDATES]:
            src_name = str(session_nodes[i].get("name", ""))
            dst_name = str(session_nodes[j].get("name", ""))
            for rel in link_rel_pool:
                if src_name and dst_name and (src_name, dst_name, rel) in linked_existing:
                    continue
                candidates.append({
                    "action": "LINK_SESSION_NODES",
                    "src_session_idx": i,
                    "dst_session_idx": j,
                    "relation": rel,
                })

        add_small_add_distractors(limit=3)
        return dedupe_and_return(candidates)

    # ------------------------------------------------------------------
    # ATTACH phase: attach hard negatives + tiny add shortcut distractor.
    # ------------------------------------------------------------------
    if phase == "attach":
        if session_nodes and memory_ids:
            gold_attach_rels = [str(a.get("relation", "related")) for a in goal.get("memory_attachments", []) or []]
            attach_rel_pool = relation_candidates(*gold_attach_rels)[:4]

            sm_pairs = [(i, j) for i in range(len(session_nodes)) for j in range(len(memory_ids))]
            sm_pairs.sort(key=session_memory_key)

            attached_existing = ss["attachments"]

            for i, j in sm_pairs[:MAX_ATTACH_PAIR_CANDIDATES]:
                sname = str(session_nodes[i].get("name", ""))
                mem = memory_ids[j]
                for rel in attach_rel_pool:
                    if sname and (sname, mem, rel) in attached_existing:
                        continue
                    candidates.append({
                        "action": "PROPOSE_LINK_SESSION_TO_MEMORY",
                        "session_idx": i,
                        "memory_idx": j,
                        "relation": rel,
                    })

        add_small_add_distractors(limit=3)
        return dedupe_and_return(candidates)

    # ------------------------------------------------------------------
    # ADD phase: add same-action negatives only.
    #
    # v1a.5.4 fix:
    # The old branch filtered out too much, so candidate_next often equaled
    # allowed_next. For add ranking, the model must compare every created
    # session node and learn which ones deserve memory commits.
    #
    # Therefore, include PROPOSE_ADD_SESSION_NODE(i) for every created session
    # node. If a node is already proposed in the synthetic state, skip it;
    # otherwise it is a valid same-action candidate. Gold add sessions remain
    # allowed_next; non-gold session nodes become hard negatives.
    # ------------------------------------------------------------------
    if phase == "add":
        if not is_noop_goal(goal):
            for i, sn in enumerate(session_nodes):
                if sn.get("proposed_add"):
                    continue
                candidates.append({
                    "action": "PROPOSE_ADD_SESSION_NODE",
                    "session_idx": i,
                })

        return dedupe_and_return(candidates)

    # ------------------------------------------------------------------
    # COVER phase: cover same-action negatives only.
    # ------------------------------------------------------------------
    if phase == "cover":
        if is_noop_goal(goal) and session_nodes and memory_ids:
            sm_pairs = [(i, j) for i in range(len(session_nodes)) for j in range(len(memory_ids))]
            sm_pairs.sort(key=session_memory_key)

            covered_existing = ss["covered"]

            for i, j in sm_pairs[:MAX_COVER_PAIR_CANDIDATES]:
                sname = str(session_nodes[i].get("name", ""))
                mem = memory_ids[j]
                if sname and (sname, mem) in covered_existing:
                    continue
                candidates.append({
                    "action": "MARK_COVERED",
                    "session_idx": i,
                    "memory_idx": j,
                })
            variant = str(row.get("state_variant", ""))
            if variant in {"cover_incomplete", "false_terminal_drift"}:
                spans = split_signal_spans(signal)
                used_span_idxs = set()
                for sn in session_nodes:
                    for x in sn.get("span_indices", []) or []:
                        try:
                            used_span_idxs.add(int(x))
                        except Exception:
                            pass
                for sp in [i for i in range(len(spans)) if i not in used_span_idxs][:2]:
                    candidates.append({
                        "action": "CREATE_SESSION_NODE",
                        "span_idx": sp,
                        "node_type": "concept",
                    })
                all_pairs = []
                for i in range(len(session_nodes)):
                    for j in range(len(session_nodes)):
                        if i != j:
                            all_pairs.append((i, j))
                all_pairs.sort(key=session_pair_key)
                for i, j in all_pairs[:3]:
                    candidates.append({
                        "action": "LINK_SESSION_NODES",
                        "src_session_idx": i,
                        "dst_session_idx": j,
                        "relation": "related",
                    })
                if variant == "false_terminal_drift":
                    candidates.append({"action": "STOP"})

        return dedupe_and_return(candidates)

    # ------------------------------------------------------------------
    # NOOP phase: terminal action only.
    # ------------------------------------------------------------------
    if phase == "noop":
        if is_noop_goal(goal) and coverage_complete(goal, state) and not bool(state.get("proposed_no_op", False)):
            candidates.append({"action": "PROPOSE_NO_OP"})
            if str(row.get("state_variant", "")) in {"cover_complete_no_noop", "false_terminal_drift", "imperfect_recovery"}:
                spans = split_signal_spans(signal)
                used_span_idxs = set()
                for sn in session_nodes:
                    for x in sn.get("span_indices", []) or []:
                        try:
                            used_span_idxs.add(int(x))
                        except Exception:
                            pass
                for sp in [i for i in range(len(spans)) if i not in used_span_idxs][:2]:
                    candidates.append({
                        "action": "CREATE_SESSION_NODE",
                        "span_idx": sp,
                        "node_type": "concept",
                    })
                all_pairs = []
                for i in range(len(session_nodes)):
                    for j in range(len(session_nodes)):
                        if i != j:
                            all_pairs.append((i, j))
                all_pairs.sort(key=session_pair_key)
                for i, j in all_pairs[:3]:
                    candidates.append({
                        "action": "LINK_SESSION_NODES",
                        "src_session_idx": i,
                        "dst_session_idx": j,
                        "relation": "related",
                    })
                if session_nodes and memory_ids:
                    sm_pairs = [(i, j) for i in range(len(session_nodes)) for j in range(len(memory_ids))]
                    sm_pairs.sort(key=session_memory_key)
                    for i, j in sm_pairs[:2]:
                        candidates.append({
                            "action": "MARK_COVERED",
                            "session_idx": i,
                            "memory_idx": j,
                        })
                add_small_add_distractors(limit=2)
                if str(row.get("state_variant", "")) == "false_terminal_drift":
                    candidates.append({"action": "STOP"})
        return dedupe_and_return(candidates)

    # ------------------------------------------------------------------
    # STOP phase: terminal action only.
    # ------------------------------------------------------------------
    if phase == "stop":
        if final_commit_complete(goal, state):
            candidates.append({"action": "STOP"})
        return dedupe_and_return(candidates)

    # ------------------------------------------------------------------
    # Fallback for unknown phase:
    # Use safe terminal gate, then gold allowed tuples will be added by
    # V1ProgressDataset.__getitem__ if absent.
    # ------------------------------------------------------------------
    if final_commit_complete(goal, state):
        candidates.append({"action": "STOP"})
    return dedupe_and_return(candidates)


# Candidate generation controls.
# These are deliberately modest so training stays fast, but large enough to
# contain same-action wrong tuples.
MAX_CREATE_CANDIDATES = 10
MAX_LINK_PAIR_CANDIDATES = 18
MAX_ATTACH_PAIR_CANDIDATES = 18
MAX_COVER_PAIR_CANDIDATES = 18

RELATION_CANDIDATE_POOL = [
    "related",
    "part_of",
    "precede",
    "depend",
    "support",
    "refine",
    "example_of",
    "cause",
    "contradict",
]


def relation_candidates(*gold_relations: str) -> List[str]:
    rels: List[str] = []
    for r in gold_relations:
        r = str(r or "").strip()
        if r and r not in rels:
            rels.append(r)
    for r in RELATION_CANDIDATE_POOL:
        if r not in rels:
            rels.append(r)
    return rels


def candidate_sort_key_by_text_pair(session_nodes: Sequence[Mapping[str, Any]], memory_ids: Sequence[str] | None = None):
    """
    Deterministic sorting key builder used to avoid random candidate sampling.

    For session-session pairs:
      prefer close span positions, but include wrong directions/pairs.

    For session-memory pairs:
      prefer low index first. The model still sees wrong pairs because we
      include all pairs up to a cap before gold tuples are reinserted later.
    """
    def span_pos(s: Mapping[str, Any]) -> float:
        inds = s.get("span_indices", []) or [0]
        try:
            return float(sum(int(x) for x in inds)) / max(len(inds), 1)
        except Exception:
            return 0.0

    pos = [span_pos(s) for s in session_nodes]

    def session_pair_key(pair: tuple[int, int]) -> tuple:
        i, j = pair
        return (abs(pos[i] - pos[j]), min(i, j), max(i, j), i, j)

    def session_memory_key(pair: tuple[int, int]) -> tuple:
        i, j = pair
        return (i + j, i, j)

    return session_pair_key, session_memory_key


def candidate_stats_for_debug(row: Mapping[str, Any], state: Mapping[str, Any], candidates: Sequence[Mapping[str, Any]], allowed: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    allowed_keys = {tuple_key(a) for a in allowed}
    cand_keys = {tuple_key(c) for c in candidates}
    return {
        "phase": row.get("phase"),
        "task_type": row.get("task_type"),
        "candidate_len": len(candidates),
        "allowed_len": len(allowed),
        "same_tuple_set": cand_keys == allowed_keys,
        "candidate_actions": sorted({str(c.get("action", "")) for c in candidates}),
        "allowed_actions": sorted({str(a.get("action", "")) for a in allowed}),
    }


class V1ProgressDataset(Dataset):
    def __init__(
        self,
        rows: Sequence[Mapping[str, Any]],
        *,
        hash_dim: int = 512,
        max_spans: int = 32,
        max_memory: int = 512,
        max_session: int = 16,
    ) -> None:
        self.rows = list(rows)
        self.hash_dim = hash_dim
        self.max_spans = max_spans
        self.max_memory = max_memory
        self.max_session = max_session
        self.graph_cache: Dict[str, MemoryGraph] = {}

    def __len__(self) -> int:
        return len(self.rows)

    def graph(self, path: str) -> MemoryGraph:
        if path not in self.graph_cache:
            self.graph_cache[path] = MemoryGraph.load_json(path)
        return self.graph_cache[path]

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.rows[idx]
        graph = self.graph(str(row["graph_path"]))
        signal = str(row["signal"])
        goal = row.get("goal", {}) or {}
        state = row.get("state", {}) or {}

        spans = split_signal_spans(signal, self.max_spans)
        if not spans:
            spans = split_signal_spans(signal or "empty", self.max_spans)
        span_bow = torch.stack([bow_hash(s.text, self.hash_dim) for s in spans[: self.max_spans]])

        memory_ids = [
            str(x)
            for x in state.get("memory_node_ids", []) or []
            if str(x) in graph.nodes
        ][: self.max_memory]
        if not memory_ids:
            memory_ids = list(graph.nodes.keys())[:1]

        memory_bow = []
        memory_scalar = []
        for mid in memory_ids:
            n = graph.nodes[mid]
            txt = f"{mid.replace('_', ' ')} {n.text}"
            memory_bow.append(bow_hash(txt, self.hash_dim))
            memory_scalar.append(torch.tensor([
                float(getattr(n, "confidence", 0.5)),
                float(getattr(n, "importance", 0.5)),
                float(lexical_overlap(signal, txt)),
                1.0,
            ], dtype=torch.float32))
        memory_bow = torch.stack(memory_bow) if memory_bow else torch.zeros(1, self.hash_dim)
        memory_scalar = torch.stack(memory_scalar) if memory_scalar else torch.zeros(1, 4)

        session_nodes = list(state.get("session_nodes", []) or [])[: self.max_session]
        session_edges = state.get("session_edges", []) or []
        attachments = state.get("attachments", []) or []
        covered = state.get("covered", []) or []

        in_deg = [0] * max(len(session_nodes), 1)
        out_deg = [0] * max(len(session_nodes), 1)
        attach_count = [0] * max(len(session_nodes), 1)
        covered_names = {str(c.get("session_name", "")) for c in covered}

        for e in session_edges:
            s = int(e.get("src_session_idx", -1))
            d = int(e.get("dst_session_idx", -1))
            if 0 <= s < len(out_deg):
                out_deg[s] += 1
            if 0 <= d < len(in_deg):
                in_deg[d] += 1

        for a in attachments:
            s = int(a.get("session_idx", -1))
            if 0 <= s < len(attach_count):
                attach_count[s] += 1

        session_bow = []
        session_scalar = []
        for i, sn in enumerate(session_nodes):
            spans_idx = sn.get("span_indices", []) or [0]
            span_pos = float(sum(int(x) for x in spans_idx) / max(len(spans_idx), 1)) / max(len(spans), 1)
            session_bow.append(bow_hash(str(sn.get("text", "")), self.hash_dim))
            session_scalar.append(torch.tensor([
                1.0,
                1.0 if sn.get("covered_by") or str(sn.get("name", "")) in covered_names else 0.0,
                1.0 if sn.get("proposed_add") else 0.0,
                min(attach_count[i], 8) / 8.0,
                min(in_deg[i], 8) / 8.0,
                min(out_deg[i], 8) / 8.0,
                span_pos,
                i / max(len(session_nodes), 1),
            ], dtype=torch.float32))

        if not session_bow:
            session_bow = [torch.zeros(self.hash_dim)]
            session_scalar = [torch.zeros(SESSION_SCALAR_DIM)]

        allowed_next = list(row.get("allowed_next", []) or [{"action": "STOP"}])
        candidate_next = build_task_candidates(row, state)

        # Ensure allowed tuples are in the candidate universe. If the generator
        # produced a legal allowed tuple not in task candidates, include it.
        c_keys = {tuple_key(c) for c in candidate_next}
        for a in allowed_next:
            if tuple_key(a) not in c_keys:
                candidate_next.append(a)
                c_keys.add(tuple_key(a))

        y_value = 1.0 if any(a.get("action") == "STOP" for a in allowed_next) else 0.5

        return {
            "signal_bow": bow_hash(signal, self.hash_dim),
            "span_bow": span_bow,
            "memory_bow": memory_bow,
            "memory_scalar": memory_scalar,
            "session_bow": torch.stack(session_bow),
            "session_scalar": torch.stack(session_scalar),
            "action_hist": action_hist_vector(state.get("action_history", []) or []),
            "global_scalar": global_scalar_from_state(state, len(spans)),
            "allowed_next": allowed_next,
            "candidate_next": candidate_next,
            "task_type": str(row.get("task_type", "unknown")),
            "goal_raw": goal,
            "state_raw": state,
            "y_value": torch.tensor(y_value, dtype=torch.float32),
            "y_phase": torch.tensor(PHASE_TO_ID.get(str(row.get("phase", "")), 0), dtype=torch.long),
        }


def collate(items: Sequence[Mapping[str, Any]]) -> ProgressBatch:
    span_bow, span_mask = pad_stack([x["span_bow"] for x in items])
    memory_bow, memory_mask = pad_stack([x["memory_bow"] for x in items])
    memory_scalar, _ = pad_stack([x["memory_scalar"] for x in items])
    session_bow, session_mask = pad_stack([x["session_bow"] for x in items])
    session_scalar, _ = pad_stack([x["session_scalar"] for x in items])

    return ProgressBatch(
        signal_bow=torch.stack([x["signal_bow"] for x in items]),
        span_bow=span_bow,
        span_mask=span_mask,
        memory_bow=memory_bow,
        memory_scalar=memory_scalar,
        memory_mask=memory_mask,
        session_bow=session_bow,
        session_scalar=session_scalar,
        session_mask=session_mask,
        action_hist=torch.stack([x["action_hist"] for x in items]),
        global_scalar=torch.stack([x["global_scalar"] for x in items]),
        allowed_next=[x["allowed_next"] for x in items],
        candidate_next=[x["candidate_next"] for x in items],
        task_type=[str(x["task_type"]) for x in items],
        goals=[x["goal_raw"] for x in items],
        states=[x["state_raw"] for x in items],
        y_value=torch.stack([x["y_value"] for x in items]),
        y_phase=torch.stack([x["y_phase"] for x in items]),
    )


def batch_to_v1(batch: ProgressBatch) -> V1Batch:
    n = batch.signal_bow.size(0)
    zlong = torch.zeros(n, dtype=torch.long, device=batch.signal_bow.device)
    zfloat = torch.zeros(n, dtype=torch.float32, device=batch.signal_bow.device)

    return V1Batch(
        signal_bow=batch.signal_bow,
        span_bow=batch.span_bow,
        span_mask=batch.span_mask,
        memory_bow=batch.memory_bow,
        memory_scalar=batch.memory_scalar,
        memory_mask=batch.memory_mask,
        session_bow=batch.session_bow,
        session_scalar=batch.session_scalar,
        session_mask=batch.session_mask,
        action_hist=batch.action_hist,
        global_scalar=batch.global_scalar,
        y_action=zlong,
        y_span=zlong,
        y_session=zlong,
        y_session_dst=zlong,
        y_memory=zlong,
        y_relation=zlong,
        y_node_type=zlong,
        y_value=zfloat,
    )


def parse_action_weights(raw: str) -> Dict[str, float]:
    weights = {a: 1.0 for a in V1_ACTIONS}
    weights.update({
        "CREATE_SESSION_NODE": 0.8,
        "LINK_SESSION_NODES": 3.0,
        "MARK_COVERED": 3.0,
        "PROPOSE_ADD_SESSION_NODE": 3.0,
        "PROPOSE_LINK_SESSION_TO_MEMORY": 2.5,
        "PROPOSE_LINK_MEMORY_TO_MEMORY": 1.5,
        "PROPOSE_NO_OP": 3.0,
        "STOP": 1.0,
    })
    if raw:
        for part in raw.split(","):
            if not part.strip():
                continue
            k, v = part.split("=", 1)
            k = k.strip()
            if k not in weights:
                raise ValueError(f"Unknown action weight: {k}")
            weights[k] = float(v)
    return weights


def parse_phase_action_aux_weights(raw: str) -> Dict[str, float]:
    weights = dict(DEFAULT_PHASE_ACTION_AUX_WEIGHTS)
    if raw:
        for part in raw.split(","):
            if not part.strip():
                continue
            k, v = part.split("=", 1)
            k = k.strip()
            if k not in weights:
                raise ValueError(f"Unknown phase action aux weight: {k}")
            weights[k] = float(v)
    return weights


def row_weight(allowed: Sequence[Mapping[str, Any]], weights: Mapping[str, float]) -> float:
    vals = [weights.get(str(a.get("action", "")), 1.0) for a in allowed]
    return max(vals) if vals else 1.0


def safe_logp(logits: torch.Tensor, idx: int) -> torch.Tensor | None:
    if idx < 0 or idx >= logits.numel():
        return None
    return F.log_softmax(logits, dim=-1)[idx]


def tuple_score(
    out: Mapping[str, torch.Tensor],
    b: int,
    tup: Mapping[str, Any],
    *,
    arg_weight: float,
    normalize_args: bool,
) -> torch.Tensor | None:
    action = str(tup.get("action", ""))
    if action not in ACTION_TO_ID:
        return None

    action_lp = safe_logp(out["action_logits"][b], ACTION_TO_ID[action])
    if action_lp is None:
        return None

    arg_lps: List[torch.Tensor] = []

    def add(x: torch.Tensor | None) -> bool:
        if x is None:
            return False
        arg_lps.append(x)
        return True

    if action == "CREATE_SESSION_NODE":
        if not add(safe_logp(out["span_logits"][b], int(tup.get("span_idx", 0)))):
            return None
        nt = str(tup.get("node_type", "concept"))
        nid = NODE_TYPE_TO_ID.get(nt, NODE_TYPE_TO_ID["concept"])
        if not add(safe_logp(out["node_type_logits"][b], nid)):
            return None

    elif action == "LINK_SESSION_NODES":
        s = int(tup.get("src_session_idx", -1))
        d = int(tup.get("dst_session_idx", -1))
        logits = out["link_pair_logits"][b]
        if s < 0 or d < 0 or s >= logits.size(0) or d >= logits.size(1):
            return None
        arg_lps.append(F.log_softmax(logits.flatten(), dim=-1)[s * logits.size(1) + d])
        rid = REL_TO_ID.get(str(tup.get("relation", "related")))
        if rid is None or not add(safe_logp(out["relation_logits"][b], rid)):
            return None

    elif action == "MARK_COVERED":
        s = int(tup.get("session_idx", -1))
        m = int(tup.get("memory_idx", -1))
        logits = out["cover_pair_logits"][b]
        if s < 0 or m < 0 or s >= logits.size(0) or m >= logits.size(1):
            return None
        arg_lps.append(F.log_softmax(logits.flatten(), dim=-1)[s * logits.size(1) + m])

    elif action == "PROPOSE_ADD_SESSION_NODE":
        if not add(safe_logp(out["add_session_logits"][b], int(tup.get("session_idx", -1)))):
            return None

    elif action == "PROPOSE_LINK_SESSION_TO_MEMORY":
        s = int(tup.get("session_idx", -1))
        m = int(tup.get("memory_idx", -1))
        logits = out["attach_pair_logits"][b]
        if s < 0 or m < 0 or s >= logits.size(0) or m >= logits.size(1):
            return None
        arg_lps.append(F.log_softmax(logits.flatten(), dim=-1)[s * logits.size(1) + m])
        rid = REL_TO_ID.get(str(tup.get("relation", "related")))
        if rid is None or not add(safe_logp(out["relation_logits"][b], rid)):
            return None

    elif action == "PROPOSE_LINK_MEMORY_TO_MEMORY":
        if not add(safe_logp(out["memory_logits"][b], int(tup.get("src_memory_idx", -1)))):
            return None
        if not add(safe_logp(out["memory_logits"][b], int(tup.get("dst_memory_idx", -1)))):
            return None
        rid = REL_TO_ID.get(str(tup.get("relation", "related")))
        if rid is not None:
            add(safe_logp(out["relation_logits"][b], rid))

    elif action in {"PROPOSE_NO_OP", "STOP"}:
        pass

    if arg_lps:
        args = torch.stack(arg_lps)
        arg_score = args.mean() if normalize_args else args.sum()
        return action_lp + arg_weight * arg_score

    return action_lp


def link_pair_aux_loss(
    out: Mapping[str, torch.Tensor],
    batch: ProgressBatch,
    *,
    reverse_margin: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    pair_losses: List[torch.Tensor] = []
    reverse_losses: List[torch.Tensor] = []

    for b, allowed in enumerate(batch.allowed_next):
        if PHASES[int(batch.y_phase[b].item())] != "link":
            continue

        allowed_pairs = {
            (int(a.get("src_session_idx", -1)), int(a.get("dst_session_idx", -1)))
            for a in allowed
            if str(a.get("action", "")) == "LINK_SESSION_NODES"
        }
        if not allowed_pairs:
            continue

        logits = out["link_pair_logits"][b]
        valid_mask = torch.isfinite(logits) & (logits > -1e8)
        if int(valid_mask.sum().item()) <= 0:
            continue

        flat = logits.flatten().masked_fill(~valid_mask.flatten(), -1e9)
        log_probs = F.log_softmax(flat, dim=0)

        width = logits.size(1)
        allowed_idx: List[int] = []
        for s, d in allowed_pairs:
            if 0 <= s < logits.size(0) and 0 <= d < logits.size(1) and bool(valid_mask[s, d]):
                allowed_idx.append(s * width + d)
        if not allowed_idx:
            continue

        allowed_t = torch.tensor(allowed_idx, dtype=torch.long, device=log_probs.device)
        pair_losses.append(-torch.logsumexp(log_probs[allowed_t], dim=0))

        for s, d in allowed_pairs:
            if not (0 <= s < logits.size(0) and 0 <= d < logits.size(1) and bool(valid_mask[s, d])):
                continue
            if (d, s) in allowed_pairs:
                continue
            if not (0 <= d < logits.size(0) and 0 <= s < logits.size(1) and bool(valid_mask[d, s])):
                continue
            reverse_losses.append(F.relu(torch.tensor(reverse_margin, device=logits.device) - (logits[s, d] - logits[d, s])))

    zero = out["link_pair_logits"].sum() * 0.0
    pair_loss = torch.stack(pair_losses).mean() if pair_losses else zero
    reverse_loss = torch.stack(reverse_losses).mean() if reverse_losses else zero
    return pair_loss, reverse_loss


def candidate_set_ce_loss(
    out: Mapping[str, torch.Tensor],
    batch: ProgressBatch,
    weights: Mapping[str, float],
    *,
    arg_weight: float,
    normalize_args: bool,
) -> torch.Tensor:
    losses = []

    for b, candidates in enumerate(batch.candidate_next):
        scores: List[torch.Tensor] = []
        keys: List[tuple] = []

        for cand in candidates:
            score = tuple_score(out, b, cand, arg_weight=arg_weight, normalize_args=normalize_args)
            if score is None:
                continue
            scores.append(score)
            keys.append(tuple_key(cand))

        if not scores:
            continue

        allowed_keys = {tuple_key(a) for a in batch.allowed_next[b]}
        allowed_idx = [i for i, k in enumerate(keys) if k in allowed_keys]

        if not allowed_idx:
            # This should not happen because dataset inserts allowed into candidate set.
            continue

        score_t = torch.stack(scores)
        log_probs = F.log_softmax(score_t, dim=0)
        allowed_logprob = torch.logsumexp(log_probs[torch.tensor(allowed_idx, device=log_probs.device)], dim=0)
        loss = -allowed_logprob

        loss = loss * row_weight(batch.allowed_next[b], weights)
        losses.append(loss)

    if not losses:
        return out["action_logits"].sum() * 0.0

    return torch.stack(losses).mean()


def tuple_hit(out: Mapping[str, torch.Tensor], batch: ProgressBatch, *, arg_weight: float, normalize_args: bool) -> tuple[int, int]:
    hits = total = 0

    for b, candidates in enumerate(batch.candidate_next):
        scored = []
        for cand in candidates:
            score = tuple_score(out, b, cand, arg_weight=arg_weight, normalize_args=normalize_args)
            if score is not None:
                scored.append((float(score.detach().cpu()), tuple_key(cand)))
        if not scored:
            continue

        scored.sort(reverse=True)
        pred_key = scored[0][1]
        allowed_keys = {tuple_key(a) for a in batch.allowed_next[b]}
        hits += int(pred_key in allowed_keys)
        total += 1

    return hits, total


def link_pair_hit(out: Mapping[str, torch.Tensor], batch: ProgressBatch) -> tuple[int, int]:
    hits = total = 0
    for b, allowed in enumerate(batch.allowed_next):
        if PHASES[int(batch.y_phase[b].item())] != "link":
            continue
        allowed_pairs = {
            (int(a.get("src_session_idx", -1)), int(a.get("dst_session_idx", -1)))
            for a in allowed
            if str(a.get("action", "")) == "LINK_SESSION_NODES"
        }
        if not allowed_pairs:
            continue
        logits = out["link_pair_logits"][b]
        valid_mask = torch.isfinite(logits) & (logits > -1e8)
        if int(valid_mask.sum().item()) <= 0:
            continue
        masked = logits.flatten().masked_fill(~valid_mask.flatten(), -1e9)
        pred = int(masked.argmax().item())
        width = logits.size(1)
        pred_pair = (pred // width, pred % width)
        hits += int(pred_pair in allowed_pairs)
        total += 1
    return hits, total


def action_family_aux_loss(
    out: Mapping[str, torch.Tensor],
    batch: ProgressBatch,
    *,
    phase_action_aux_weights: Mapping[str, float],
    link_attach_margin: float,
    link_attach_margin_weight: float,
    attach_link_margin: float,
    attach_link_margin_weight: float,
    non_attach_attach_margin: float,
    non_attach_attach_margin_weight: float,
    noop_stop_margin: float,
    noop_stop_margin_weight: float,
    premature_stop_margin: float,
    premature_stop_margin_weight: float,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    action_logits = out["action_logits"]
    compatible_actions = torch.tensor(
        [ACTION_TO_ID[PHASE_TO_ACTION[PHASES[int(i)]]] for i in batch.y_phase.detach().cpu().tolist()],
        dtype=torch.long,
        device=batch.y_phase.device,
    )
    per_row_ce = F.cross_entropy(action_logits, compatible_actions, reduction="none")
    row_weights = torch.tensor(
        [
            0.0 if str(batch.task_type[b]) == "covered_long_signal" else phase_action_aux_weights.get(PHASES[int(batch.y_phase[b].item())], 1.0)
            for b in range(batch.y_phase.size(0))
        ],
        dtype=per_row_ce.dtype,
        device=per_row_ce.device,
    )
    weighted_ce = (per_row_ce * row_weights).mean()

    zero = action_logits.sum() * 0.0
    link_attach_losses: List[torch.Tensor] = []
    attach_link_losses: List[torch.Tensor] = []
    non_attach_attach_losses: List[torch.Tensor] = []
    noop_stop_losses: List[torch.Tensor] = []
    premature_stop_losses: List[torch.Tensor] = []

    link_id = ACTION_TO_ID["LINK_SESSION_NODES"]
    attach_id = ACTION_TO_ID["PROPOSE_LINK_SESSION_TO_MEMORY"]
    noop_id = ACTION_TO_ID["PROPOSE_NO_OP"]
    stop_id = ACTION_TO_ID["STOP"]

    for b in range(batch.y_phase.size(0)):
        if str(batch.task_type[b]) == "covered_long_signal":
            continue
        phase_name = PHASES[int(batch.y_phase[b].item())]
        gold_action_id = compatible_actions[b]
        if phase_name == "link":
            link_attach_losses.append(F.relu(torch.tensor(link_attach_margin, device=action_logits.device) - (action_logits[b, link_id] - action_logits[b, attach_id])))
        if phase_name == "attach":
            attach_link_losses.append(F.relu(torch.tensor(attach_link_margin, device=action_logits.device) - (action_logits[b, attach_id] - action_logits[b, link_id])))
        if phase_name not in {"attach", "stop"}:
            non_attach_attach_losses.append(F.relu(torch.tensor(non_attach_attach_margin, device=action_logits.device) - (action_logits[b, gold_action_id] - action_logits[b, attach_id])))
        if phase_name == "noop":
            noop_stop_losses.append(F.relu(torch.tensor(noop_stop_margin, device=action_logits.device) - (action_logits[b, noop_id] - action_logits[b, stop_id])))
        if phase_name != "stop":
            premature_stop_losses.append(F.relu(torch.tensor(premature_stop_margin, device=action_logits.device) - (action_logits[b, gold_action_id] - action_logits[b, stop_id])))

    loss_link_attach = torch.stack(link_attach_losses).mean() if link_attach_losses else zero
    loss_attach_link = torch.stack(attach_link_losses).mean() if attach_link_losses else zero
    loss_non_attach_attach = torch.stack(non_attach_attach_losses).mean() if non_attach_attach_losses else zero
    loss_noop_stop = torch.stack(noop_stop_losses).mean() if noop_stop_losses else zero
    loss_premature_stop = torch.stack(premature_stop_losses).mean() if premature_stop_losses else zero

    total = (
        weighted_ce
        + link_attach_margin_weight * loss_link_attach
        + attach_link_margin_weight * loss_attach_link
        + non_attach_attach_margin_weight * loss_non_attach_attach
        + noop_stop_margin_weight * loss_noop_stop
        + premature_stop_margin_weight * loss_premature_stop
    )
    return total, {
        "loss_action_phase_ce_weighted": weighted_ce,
        "loss_link_attach_margin": loss_link_attach,
        "loss_attach_link_margin": loss_attach_link,
        "loss_non_attach_attach_margin": loss_non_attach_attach,
        "loss_noop_stop_margin": loss_noop_stop,
        "loss_premature_stop_margin": loss_premature_stop,
    }


def covered_task_control_loss(
    out: Mapping[str, torch.Tensor],
    batch: ProgressBatch,
    *,
    covered_create_margin_weight: float,
    covered_cover_margin_weight: float,
    covered_noop_margin_weight: float,
    covered_premature_stop_margin_weight: float,
    covered_negative_action_margin_weight: float,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    action_logits = out["action_logits"]
    zero = action_logits.sum() * 0.0
    loss_create_ce: List[torch.Tensor] = []
    losses: List[torch.Tensor] = []
    loss_cover_ce: List[torch.Tensor] = []
    loss_noop_ce: List[torch.Tensor] = []
    loss_stop_ce: List[torch.Tensor] = []
    loss_negative_margin: List[torch.Tensor] = []
    loss_premature_stop: List[torch.Tensor] = []

    create_id = ACTION_TO_ID["CREATE_SESSION_NODE"]
    cover_id = ACTION_TO_ID["MARK_COVERED"]
    noop_id = ACTION_TO_ID["PROPOSE_NO_OP"]
    stop_id = ACTION_TO_ID["STOP"]
    neg_ids_create = [ACTION_TO_ID["MARK_COVERED"], ACTION_TO_ID["LINK_SESSION_NODES"], ACTION_TO_ID["PROPOSE_NO_OP"], ACTION_TO_ID["STOP"]]
    neg_ids_common = [ACTION_TO_ID["CREATE_SESSION_NODE"], ACTION_TO_ID["LINK_SESSION_NODES"], ACTION_TO_ID["STOP"]]
    neg_ids_noop = [ACTION_TO_ID["CREATE_SESSION_NODE"], ACTION_TO_ID["LINK_SESSION_NODES"], ACTION_TO_ID["MARK_COVERED"], ACTION_TO_ID["STOP"]]

    for b, task in enumerate(batch.task_type):
        if str(task) != "covered_long_signal":
            continue
        goal = batch.goals[b]
        state = batch.states[b]
        phase_name = covered_structural_phase(goal, state)
        logits = action_logits[b]
        if phase_name == "create":
            loss_create_ce.append(F.cross_entropy(logits.unsqueeze(0), torch.tensor([create_id], device=logits.device)))
            for neg_id in neg_ids_create:
                loss_negative_margin.append(F.relu(torch.tensor(0.5, device=logits.device) - (logits[create_id] - logits[neg_id])))
            loss_premature_stop.append(F.relu(torch.tensor(0.5, device=logits.device) - (logits[create_id] - logits[stop_id])))
        elif phase_name == "cover":
            loss_cover_ce.append(F.cross_entropy(logits.unsqueeze(0), torch.tensor([cover_id], device=logits.device)))
            for neg_id in neg_ids_common:
                loss_negative_margin.append(F.relu(torch.tensor(0.5, device=logits.device) - (logits[cover_id] - logits[neg_id])))
            loss_premature_stop.append(F.relu(torch.tensor(0.5, device=logits.device) - (logits[cover_id] - logits[stop_id])))
        elif phase_name == "noop":
            loss_noop_ce.append(F.cross_entropy(logits.unsqueeze(0), torch.tensor([noop_id], device=logits.device)))
            for neg_id in neg_ids_noop:
                loss_negative_margin.append(F.relu(torch.tensor(0.5, device=logits.device) - (logits[noop_id] - logits[neg_id])))
            loss_premature_stop.append(F.relu(torch.tensor(0.5, device=logits.device) - (logits[noop_id] - logits[stop_id])))
        elif phase_name == "stop":
            loss_stop_ce.append(F.cross_entropy(logits.unsqueeze(0), torch.tensor([stop_id], device=logits.device)))

    create_ce = torch.stack(loss_create_ce).mean() if loss_create_ce else zero
    cover_ce = torch.stack(loss_cover_ce).mean() if loss_cover_ce else zero
    noop_ce = torch.stack(loss_noop_ce).mean() if loss_noop_ce else zero
    stop_ce = torch.stack(loss_stop_ce).mean() if loss_stop_ce else zero
    neg_margin = torch.stack(loss_negative_margin).mean() if loss_negative_margin else zero
    premature_stop = torch.stack(loss_premature_stop).mean() if loss_premature_stop else zero
    total = (
        covered_create_margin_weight * create_ce
        + covered_cover_margin_weight * cover_ce
        + covered_noop_margin_weight * noop_ce
        + 1.0 * stop_ce
        + covered_negative_action_margin_weight * neg_margin
        + covered_premature_stop_margin_weight * premature_stop
    )
    return total, {
        "loss_covered_create_ce": create_ce,
        "loss_covered_cover_ce": cover_ce,
        "loss_covered_noop_ce": noop_ce,
        "loss_covered_stop_ce": stop_ce,
        "loss_covered_negative_margin": neg_margin,
        "loss_covered_premature_stop": premature_stop,
    }


def loss_fn(out, batch, *, value_weight, phase_weight, action_phase_weight, phase_action_aux_weights, link_attach_margin, link_attach_margin_weight, attach_link_margin, attach_link_margin_weight, non_attach_attach_margin, non_attach_attach_margin_weight, noop_stop_margin, noop_stop_margin_weight, premature_stop_margin, premature_stop_margin_weight, covered_create_margin_weight, covered_cover_margin_weight, covered_noop_margin_weight, covered_premature_stop_margin_weight, covered_negative_action_margin_weight, link_pair_aux_weight, link_reverse_margin_weight, link_reverse_margin, action_weights, arg_weight, normalize_args):
    loss_tuple = candidate_set_ce_loss(
        out,
        batch,
        action_weights,
        arg_weight=arg_weight,
        normalize_args=normalize_args,
    )
    loss_value = F.mse_loss(out["value"], batch.y_value)
    loss_phase = F.cross_entropy(out["phase_logits"], batch.y_phase)
    loss_link_pair_aux, loss_link_reverse = link_pair_aux_loss(
        out,
        batch,
        reverse_margin=link_reverse_margin,
    )
    loss_action_phase, action_phase_parts = action_family_aux_loss(
        out,
        batch,
        phase_action_aux_weights=phase_action_aux_weights,
        link_attach_margin=link_attach_margin,
        link_attach_margin_weight=link_attach_margin_weight,
        attach_link_margin=attach_link_margin,
        attach_link_margin_weight=attach_link_margin_weight,
        non_attach_attach_margin=non_attach_attach_margin,
        non_attach_attach_margin_weight=non_attach_attach_margin_weight,
        noop_stop_margin=noop_stop_margin,
        noop_stop_margin_weight=noop_stop_margin_weight,
        premature_stop_margin=premature_stop_margin,
        premature_stop_margin_weight=premature_stop_margin_weight,
    )
    loss_covered_control, covered_parts = covered_task_control_loss(
        out,
        batch,
        covered_create_margin_weight=covered_create_margin_weight,
        covered_cover_margin_weight=covered_cover_margin_weight,
        covered_noop_margin_weight=covered_noop_margin_weight,
        covered_premature_stop_margin_weight=covered_premature_stop_margin_weight,
        covered_negative_action_margin_weight=covered_negative_action_margin_weight,
    )
    return (
        loss_tuple
        + value_weight * loss_value
        + phase_weight * loss_phase
        + action_phase_weight * loss_action_phase
        + loss_covered_control
        + link_pair_aux_weight * loss_link_pair_aux
        + link_reverse_margin_weight * loss_link_reverse
    ), {
        "loss_tuple": float(loss_tuple.detach().cpu()),
        "loss_value": float(loss_value.detach().cpu()),
        "loss_phase": float(loss_phase.detach().cpu()),
        "loss_action_phase": float(loss_action_phase.detach().cpu()),
        "loss_action_phase_ce_weighted": float(action_phase_parts["loss_action_phase_ce_weighted"].detach().cpu()),
        "loss_link_attach_margin": float(action_phase_parts["loss_link_attach_margin"].detach().cpu()),
        "loss_attach_link_margin": float(action_phase_parts["loss_attach_link_margin"].detach().cpu()),
        "loss_non_attach_attach_margin": float(action_phase_parts["loss_non_attach_attach_margin"].detach().cpu()),
        "loss_noop_stop_margin": float(action_phase_parts["loss_noop_stop_margin"].detach().cpu()),
        "loss_premature_stop_margin": float(action_phase_parts["loss_premature_stop_margin"].detach().cpu()),
        "loss_covered_control": float(loss_covered_control.detach().cpu()),
        "loss_covered_create_ce": float(covered_parts["loss_covered_create_ce"].detach().cpu()),
        "loss_covered_cover_ce": float(covered_parts["loss_covered_cover_ce"].detach().cpu()),
        "loss_covered_noop_ce": float(covered_parts["loss_covered_noop_ce"].detach().cpu()),
        "loss_covered_stop_ce": float(covered_parts["loss_covered_stop_ce"].detach().cpu()),
        "loss_covered_negative_margin": float(covered_parts["loss_covered_negative_margin"].detach().cpu()),
        "loss_covered_premature_stop": float(covered_parts["loss_covered_premature_stop"].detach().cpu()),
        "loss_link_pair_aux": float(loss_link_pair_aux.detach().cpu()),
        "loss_link_reverse": float(loss_link_reverse.detach().cpu()),
    }


@torch.no_grad()
def evaluate(model, loader, device, *, value_weight, phase_weight, action_phase_weight, phase_action_aux_weights, link_attach_margin, link_attach_margin_weight, attach_link_margin, attach_link_margin_weight, non_attach_attach_margin, non_attach_attach_margin_weight, noop_stop_margin, noop_stop_margin_weight, premature_stop_margin, premature_stop_margin_weight, covered_create_margin_weight, covered_cover_margin_weight, covered_noop_margin_weight, covered_premature_stop_margin_weight, covered_negative_action_margin_weight, link_pair_aux_weight, link_reverse_margin_weight, link_reverse_margin, action_weights, arg_weight, normalize_args):
    model.eval()
    hits = total = 0
    link_hits = link_total = 0
    loss_sum = 0.0
    steps = 0
    phase_hits = phase_total = 0
    covered_create_hits = covered_create_total = 0
    covered_cover_hits = covered_cover_total = 0
    covered_noop_hits = covered_noop_total = 0
    covered_stop_hits = covered_stop_total = 0
    covered_cover_phase_hits = covered_cover_phase_total = 0
    covered_noop_phase_hits = covered_noop_phase_total = 0
    covered_stop_phase_hits = covered_stop_phase_total = 0

    for batch in loader:
        batch = to_device(batch, device)
        out = model(batch_to_v1(batch))
        loss, _ = loss_fn(
            out,
            batch,
            value_weight=value_weight,
            phase_weight=phase_weight,
            action_phase_weight=action_phase_weight,
            phase_action_aux_weights=phase_action_aux_weights,
            link_attach_margin=link_attach_margin,
            link_attach_margin_weight=link_attach_margin_weight,
            attach_link_margin=attach_link_margin,
            attach_link_margin_weight=attach_link_margin_weight,
            non_attach_attach_margin=non_attach_attach_margin,
            non_attach_attach_margin_weight=non_attach_attach_margin_weight,
            noop_stop_margin=noop_stop_margin,
            noop_stop_margin_weight=noop_stop_margin_weight,
            premature_stop_margin=premature_stop_margin,
            premature_stop_margin_weight=premature_stop_margin_weight,
            covered_create_margin_weight=covered_create_margin_weight,
            covered_cover_margin_weight=covered_cover_margin_weight,
            covered_noop_margin_weight=covered_noop_margin_weight,
            covered_premature_stop_margin_weight=covered_premature_stop_margin_weight,
            covered_negative_action_margin_weight=covered_negative_action_margin_weight,
            link_pair_aux_weight=link_pair_aux_weight,
            link_reverse_margin_weight=link_reverse_margin_weight,
            link_reverse_margin=link_reverse_margin,
            action_weights=action_weights,
            arg_weight=arg_weight,
            normalize_args=normalize_args,
        )
        h, t = tuple_hit(out, batch, arg_weight=arg_weight, normalize_args=normalize_args)
        hits += h
        total += t
        lh, lt = link_pair_hit(out, batch)
        link_hits += lh
        link_total += lt
        phase_pred = out["phase_logits"].argmax(dim=-1)
        action_pred = out["action_logits"].argmax(dim=-1)
        phase_hits += int((phase_pred == batch.y_phase).sum().item())
        phase_total += int(batch.y_phase.numel())
        for b, task in enumerate(batch.task_type):
            if str(task) != "covered_long_signal":
                continue
            phase_name = covered_structural_phase(batch.goals[b], batch.states[b])
            phase_id = PHASE_TO_ID.get(phase_name, -1)
            if phase_name == "create":
                covered_create_total += 1
                covered_create_hits += int(int(action_pred[b].item()) == ACTION_TO_ID["CREATE_SESSION_NODE"])
            elif phase_name == "cover":
                covered_cover_total += 1
                covered_cover_hits += int(int(action_pred[b].item()) == ACTION_TO_ID["MARK_COVERED"])
                if phase_id >= 0:
                    covered_cover_phase_total += 1
                    covered_cover_phase_hits += int(int(phase_pred[b].item()) == phase_id)
            elif phase_name == "noop":
                covered_noop_total += 1
                covered_noop_hits += int(int(action_pred[b].item()) == ACTION_TO_ID["PROPOSE_NO_OP"])
                if phase_id >= 0:
                    covered_noop_phase_total += 1
                    covered_noop_phase_hits += int(int(phase_pred[b].item()) == phase_id)
            elif phase_name == "stop":
                covered_stop_total += 1
                covered_stop_hits += int(int(action_pred[b].item()) == ACTION_TO_ID["STOP"])
                if phase_id >= 0:
                    covered_stop_phase_total += 1
                    covered_stop_phase_hits += int(int(phase_pred[b].item()) == phase_id)
        loss_sum += float(loss.detach().cpu())
        steps += 1

    return {
        "tuple_candidate_hit": hits / max(total, 1),
        "tuple_rows": total,
        "link_pair_candidate_hit": link_hits / max(link_total, 1),
        "link_pair_rows": link_total,
        "val_candidate_ce": loss_sum / max(steps, 1),
        "phase_accuracy": phase_hits / max(phase_total, 1),
        "covered_create_action_accuracy": covered_create_hits / max(covered_create_total, 1),
        "covered_create_rows": covered_create_total,
        "covered_cover_action_accuracy": covered_cover_hits / max(covered_cover_total, 1),
        "covered_cover_rows": covered_cover_total,
        "covered_noop_action_accuracy": covered_noop_hits / max(covered_noop_total, 1),
        "covered_noop_rows": covered_noop_total,
        "covered_stop_action_accuracy": covered_stop_hits / max(covered_stop_total, 1),
        "covered_stop_rows": covered_stop_total,
        "covered_cover_phase_accuracy": covered_cover_phase_hits / max(covered_cover_phase_total, 1),
        "covered_noop_phase_accuracy": covered_noop_phase_hits / max(covered_noop_phase_total, 1),
        "covered_stop_phase_accuracy": covered_stop_phase_hits / max(covered_stop_phase_total, 1),
    }


def train(args):
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    action_weights = parse_action_weights(args.action_weights)
    phase_action_aux_weights = parse_phase_action_aux_weights(args.phase_action_aux_weights)

    train_rows = read_jsonl(args.train_jsonl)
    val_rows = read_jsonl(args.val_jsonl)

    train_ds = V1ProgressDataset(train_rows, hash_dim=args.hash_dim, max_spans=args.max_spans, max_memory=args.max_memory, max_session=args.max_session)
    val_ds = V1ProgressDataset(val_rows, hash_dim=args.hash_dim, max_spans=args.max_spans, max_memory=args.max_memory, max_session=args.max_session)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate, num_workers=0)

    model = NGRV1PolicyNet(hash_dim=args.hash_dim, hidden_dim=args.hidden_dim).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(json.dumps({
        "device": str(device),
        "train_rows": len(train_rows),
        "val_rows": len(val_rows),
        "epochs": args.epochs,
        "mode": "v1a13 covered-noop-recovery no-retrieval full-graph candidate-set CE",
        "action_weights": action_weights,
        "arg_weight": args.arg_weight,
        "phase_weight": args.phase_weight,
        "action_phase_weight": args.action_phase_weight,
        "phase_action_aux_weights": phase_action_aux_weights,
        "link_attach_margin_weight": args.link_attach_margin_weight,
        "link_attach_margin": args.link_attach_margin,
        "attach_link_margin_weight": args.attach_link_margin_weight,
        "attach_link_margin": args.attach_link_margin,
        "non_attach_attach_margin_weight": args.non_attach_attach_margin_weight,
        "non_attach_attach_margin": args.non_attach_attach_margin,
        "noop_stop_margin_weight": args.noop_stop_margin_weight,
        "noop_stop_margin": args.noop_stop_margin,
        "premature_stop_margin_weight": args.premature_stop_margin_weight,
        "premature_stop_margin": args.premature_stop_margin,
        "link_pair_aux_weight": args.link_pair_aux_weight,
        "link_reverse_margin_weight": args.link_reverse_margin_weight,
        "link_reverse_margin": args.link_reverse_margin,
        "covered_create_margin_weight": args.covered_create_margin_weight,
        "covered_cover_margin_weight": args.covered_cover_margin_weight,
        "covered_noop_margin_weight": args.covered_noop_margin_weight,
        "covered_premature_stop_margin_weight": args.covered_premature_stop_margin_weight,
        "covered_negative_action_margin_weight": args.covered_negative_action_margin_weight,
        "normalize_args": not args.no_normalize_args,
        "expected_loss": "non_negative",
    }, indent=2))

    best = -1e9

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        part_sum: Dict[str, float] = {}
        steps = skipped = 0

        for batch in train_loader:
            batch = to_device(batch, device)
            out = model(batch_to_v1(batch))
            loss, parts = loss_fn(
                out,
                batch,
                value_weight=args.value_weight,
                phase_weight=args.phase_weight,
                action_phase_weight=args.action_phase_weight,
                phase_action_aux_weights=phase_action_aux_weights,
                link_attach_margin=args.link_attach_margin,
                link_attach_margin_weight=args.link_attach_margin_weight,
                attach_link_margin=args.attach_link_margin,
                attach_link_margin_weight=args.attach_link_margin_weight,
                non_attach_attach_margin=args.non_attach_attach_margin,
                non_attach_attach_margin_weight=args.non_attach_attach_margin_weight,
                noop_stop_margin=args.noop_stop_margin,
                noop_stop_margin_weight=args.noop_stop_margin_weight,
                premature_stop_margin=args.premature_stop_margin,
                premature_stop_margin_weight=args.premature_stop_margin_weight,
                covered_create_margin_weight=args.covered_create_margin_weight,
                covered_cover_margin_weight=args.covered_cover_margin_weight,
                covered_noop_margin_weight=args.covered_noop_margin_weight,
                covered_premature_stop_margin_weight=args.covered_premature_stop_margin_weight,
                covered_negative_action_margin_weight=args.covered_negative_action_margin_weight,
                link_pair_aux_weight=args.link_pair_aux_weight,
                link_reverse_margin_weight=args.link_reverse_margin_weight,
                link_reverse_margin=args.link_reverse_margin,
                action_weights=action_weights,
                arg_weight=args.arg_weight,
                normalize_args=not args.no_normalize_args,
            )

            if not torch.isfinite(loss):
                skipped += 1
                continue

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            opt.step()

            total_loss += float(loss.detach().cpu())
            for k, v in parts.items():
                part_sum[k] = part_sum.get(k, 0.0) + v
            steps += 1

        metrics = evaluate(
            model,
            val_loader,
            device,
            value_weight=args.value_weight,
            phase_weight=args.phase_weight,
            action_phase_weight=args.action_phase_weight,
            phase_action_aux_weights=phase_action_aux_weights,
            link_attach_margin=args.link_attach_margin,
            link_attach_margin_weight=args.link_attach_margin_weight,
            attach_link_margin=args.attach_link_margin,
            attach_link_margin_weight=args.attach_link_margin_weight,
            non_attach_attach_margin=args.non_attach_attach_margin,
            non_attach_attach_margin_weight=args.non_attach_attach_margin_weight,
            noop_stop_margin=args.noop_stop_margin,
            noop_stop_margin_weight=args.noop_stop_margin_weight,
            premature_stop_margin=args.premature_stop_margin,
            premature_stop_margin_weight=args.premature_stop_margin_weight,
            covered_create_margin_weight=args.covered_create_margin_weight,
            covered_cover_margin_weight=args.covered_cover_margin_weight,
            covered_noop_margin_weight=args.covered_noop_margin_weight,
            covered_premature_stop_margin_weight=args.covered_premature_stop_margin_weight,
            covered_negative_action_margin_weight=args.covered_negative_action_margin_weight,
            link_pair_aux_weight=args.link_pair_aux_weight,
            link_reverse_margin_weight=args.link_reverse_margin_weight,
            link_reverse_margin=args.link_reverse_margin,
            action_weights=action_weights,
            arg_weight=args.arg_weight,
            normalize_args=not args.no_normalize_args,
        )
        score = metrics["tuple_candidate_hit"] + 0.10 * metrics["phase_accuracy"] - 0.05 * metrics["val_candidate_ce"]

        log = {
            "epoch": epoch,
            "train_loss": total_loss / max(steps, 1),
            "skipped_nonfinite": skipped,
            **{k: v / max(steps, 1) for k, v in sorted(part_sum.items())},
            **metrics,
            "score": score,
        }
        print(json.dumps(log, indent=2))

        if score > best:
            best = score
            torch.save({
                "model": model.state_dict(),
                "args": vars(args),
                "actions": V1_ACTIONS,
                "relations": V1_RELATIONS,
                "node_types": V1_NODE_TYPES,
                "best_score": best,
                "action_weights": action_weights,
                "model_version": "ngr_v1a13_covered_noop_recovery",
            }, out_dir / "best_ngr_v1a.pt")

    print(f"saved best checkpoint to {out_dir / 'best_ngr_v1a.pt'}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-jsonl", required=True)
    ap.add_argument("--val-jsonl", required=True)
    ap.add_argument("--out-dir", default="out_ngr_v1a6_no_retrieval")
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--hash-dim", type=int, default=512)
    ap.add_argument("--hidden-dim", type=int, default=256)
    ap.add_argument("--max-spans", type=int, default=32)
    ap.add_argument("--max-memory", type=int, default=512)
    ap.add_argument("--max-session", type=int, default=16)
    ap.add_argument("--value-weight", type=float, default=0.05)
    ap.add_argument("--phase-weight", type=float, default=0.5)
    ap.add_argument("--action-phase-weight", type=float, default=0.5)
    ap.add_argument("--phase-action-aux-weights", default="")
    ap.add_argument("--link-attach-margin-weight", type=float, default=1.0)
    ap.add_argument("--link-attach-margin", type=float, default=0.5)
    ap.add_argument("--attach-link-margin-weight", type=float, default=0.75)
    ap.add_argument("--attach-link-margin", type=float, default=0.5)
    ap.add_argument("--non-attach-attach-margin-weight", type=float, default=0.75)
    ap.add_argument("--non-attach-attach-margin", type=float, default=0.5)
    ap.add_argument("--noop-stop-margin-weight", type=float, default=3.0)
    ap.add_argument("--noop-stop-margin", type=float, default=0.5)
    ap.add_argument("--premature-stop-margin-weight", type=float, default=1.0)
    ap.add_argument("--premature-stop-margin", type=float, default=0.5)
    ap.add_argument("--covered-create-margin-weight", type=float, default=2.0)
    ap.add_argument("--covered-cover-margin-weight", type=float, default=2.5)
    ap.add_argument("--covered-noop-margin-weight", type=float, default=3.0)
    ap.add_argument("--covered-premature-stop-margin-weight", type=float, default=3.0)
    ap.add_argument("--covered-negative-action-margin-weight", type=float, default=2.0)
    ap.add_argument("--link-pair-aux-weight", type=float, default=0.75)
    ap.add_argument("--link-reverse-margin-weight", type=float, default=0.50)
    ap.add_argument("--link-reverse-margin", type=float, default=0.50)
    ap.add_argument("--action-weights", default="")
    ap.add_argument("--arg-weight", type=float, default=1.0)
    ap.add_argument("--no-normalize-args", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--cpu", action="store_true")
    args = ap.parse_args()
    train(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
