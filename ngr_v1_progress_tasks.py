from __future__ import annotations

"""
ngr_v1_progress_tasks.py

NGR-v1a progress-state generator.

Main changes:
- Keeps phase-exclusive rows.
- Uses full-graph memory visibility in synthetic states.
- Adds realistic action_history for each phase.
"""

import argparse
import copy
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

from graph_core import MemoryGraph, lexical_overlap
from ngr_v1_env import split_signal_spans


def read_jsonl(path: str | Path) -> List[Dict[str, Any]]:
    rows = []
    with Path(path).open("r", encoding="utf-8-sig") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: str | Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def clean_text(x: Any, max_len: int = 360) -> str:
    return " ".join(str(x or "").split())[:max_len].rstrip()


def best_span_idx(signal: str, span_text: str) -> int:
    spans = split_signal_spans(signal)
    if not spans:
        return 0
    best_i, best_s = 0, -1.0
    for i, sp in enumerate(spans):
        s = float(lexical_overlap(span_text, sp.text))
        if s > best_s:
            best_i, best_s = i, s
    return best_i


def goal_session_name_map(goal: Mapping[str, Any]) -> Dict[str, Mapping[str, Any]]:
    out = {}
    for i, s in enumerate(goal.get("session_nodes", []) or []):
        out[str(s.get("name", f"s{i}"))] = s
    return out


def goal_memory_ids(goal: Mapping[str, Any], row: Mapping[str, Any]) -> List[str]:
    ids = [str(x) for x in row.get("initial_memory_node_ids", []) or []]
    for att in goal.get("memory_attachments", []) or []:
        mem = str(att.get("memory_id", ""))
        if mem:
            ids.append(mem)
    for cov in goal.get("covered_mappings", []) or []:
        mem = str(cov.get("memory_id", ""))
        if mem:
            ids.append(mem)
    seen = set()
    return [x for x in ids if x and not (x in seen or seen.add(x))]


_GRAPH_CACHE: Dict[str, MemoryGraph] = {}


def full_graph_memory_ids(row: Mapping[str, Any], signal: str) -> List[str]:
    path = str(row.get("graph_path", ""))
    if path not in _GRAPH_CACHE:
        _GRAPH_CACHE[path] = MemoryGraph.load_json(path)
    graph = _GRAPH_CACHE[path]
    scored = []
    for nid in graph.nodes:
        txt = f"{nid.replace('_', ' ')} {graph.nodes[nid].text}"
        scored.append((float(lexical_overlap(signal, txt)), nid))
    scored.sort(key=lambda x: (-x[0], x[1]))
    return [nid for _score, nid in scored]


def pseudo_cover_goal(row: Mapping[str, Any]) -> Dict[str, Any]:
    goal = dict(row.get("goal", {}) or {})
    if goal.get("session_nodes"):
        return goal
    goal["session_nodes"] = [
        {
            "name": f"covered_{i}",
            "span_text": clean_text(cov.get("span_text", row.get("signal", ""))),
            "node_type": "concept",
        }
        for i, cov in enumerate(goal.get("covered_mappings", []) or [])
    ]
    return goal


def session_node_from_goal(signal: str, name: str, spec: Mapping[str, Any]) -> Dict[str, Any]:
    text = clean_text(spec.get("span_text", ""))
    return {
        "name": name,
        "span_indices": [best_span_idx(signal, text)],
        "text": text,
        "node_type": str(spec.get("node_type", "concept")),
        "covered_by": None,
        "proposed_add": False,
    }


def hist_actions(*names: str) -> List[Dict[str, Any]]:
    return [{"action": n, "valid": True, "synthetic_history": True, "step": i} for i, n in enumerate(names)]


def history_for_phase(phase: str, created_count: int = 0, edge_count: int = 0, attach_count: int = 0, cover_count: int = 0) -> List[Dict[str, Any]]:
    h: List[str] = []
    h.extend(["CREATE_SESSION_NODE"] * max(0, created_count))
    if phase in {"attach", "add", "stop"}:
        h.extend(["LINK_SESSION_NODES"] * max(0, edge_count))
    if phase in {"add", "stop"}:
        h.extend(["PROPOSE_LINK_SESSION_TO_MEMORY"] * max(0, attach_count))
    if phase in {"cover", "noop", "stop"}:
        h.extend(["MARK_COVERED"] * max(0, cover_count))
    if phase == "stop":
        pass
    return hist_actions(*h)


def materialize_state(
    row: Mapping[str, Any],
    *,
    created_names: Sequence[str],
    linked_edges: Sequence[Tuple[str, str, str]],
    attachments_done: Sequence[Tuple[str, str, str]],
    covered_done: Sequence[Tuple[str, str]],
    proposed_adds: Sequence[str],
    proposed_no_op: bool,
    include_memory_ids: Sequence[str],
    phase: str,
) -> Dict[str, Any]:
    signal = str(row["signal"])
    goal = row.get("goal", {}) or {}
    gs = goal_session_name_map(goal)

    proposed_add_set = set(proposed_adds)
    covered_map = {s: m for s, m in covered_done}

    session_nodes = []
    name_to_idx = {}
    for name in created_names:
        if name not in gs:
            continue
        sn = session_node_from_goal(signal, name, gs[name])
        sn["covered_by"] = covered_map.get(name)
        sn["proposed_add"] = name in proposed_add_set
        name_to_idx[name] = len(session_nodes)
        session_nodes.append(sn)

    session_edges = []
    for src, dst, rel in linked_edges:
        if src in name_to_idx and dst in name_to_idx:
            session_edges.append({
                "src_name": src,
                "dst_name": dst,
                "src_session_idx": name_to_idx[src],
                "dst_session_idx": name_to_idx[dst],
                "relation": rel,
            })

    attachments = []
    for sname, mem, rel in attachments_done:
        if sname in name_to_idx:
            attachments.append({
                "session_name": sname,
                "session_idx": name_to_idx[sname],
                "memory_id": mem,
                "relation": rel,
                "proposed": True,
            })

    memory_ids = list(include_memory_ids)
    for _s, mem, _r in attachments_done:
        memory_ids.append(mem)
    for _s, mem in covered_done:
        memory_ids.append(mem)
    seen = set()
    memory_ids = [m for m in memory_ids if m and not (m in seen or seen.add(m))]

    action_history = history_for_phase(
        phase,
        created_count=len(session_nodes),
        edge_count=len(session_edges),
        attach_count=len(attachments),
        cover_count=len(covered_done),
    )
    if proposed_no_op:
        action_history.append({
            "action": "PROPOSE_NO_OP",
            "valid": True,
            "synthetic_history": True,
            "step": len(action_history),
        })

    return {
        "memory_node_ids": memory_ids,
        "session_nodes": session_nodes,
        "session_edges": session_edges,
        "attachments": attachments,
        "covered": [{"session_name": s, "memory_id": m} for s, m in covered_done],
        "proposed_adds": list(proposed_adds),
        "proposed_no_op": bool(proposed_no_op),
        "action_history": action_history,
    }


def synthetic_extra_session_node(signal: str, idx: int = 0) -> Dict[str, Any]:
    spans = split_signal_spans(signal)
    span_idx = min(max(idx, 0), max(len(spans) - 1, 0))
    text = clean_text(spans[span_idx].text if spans else signal)
    return {
        "name": f"extra_{span_idx}",
        "text": text,
        "span_indices": [span_idx],
        "node_type": "concept",
        "covered_by": None,
        "proposed_add": False,
    }


def add_extra_session_node(state: Mapping[str, Any], signal: str, *, idx: int = 0) -> Dict[str, Any]:
    out = copy.deepcopy(dict(state))
    session_nodes = list(out.get("session_nodes", []) or [])
    extra = synthetic_extra_session_node(signal, idx=idx)
    existing_names = {str(s.get("name", "")) for s in session_nodes}
    if extra["name"] not in existing_names:
        session_nodes.append(extra)
    out["session_nodes"] = session_nodes
    return out


def recovery_rows_for_goal(row: Mapping[str, Any], rng: random.Random) -> List[Dict[str, Any]]:
    raw_goal = row.get("goal", {}) or {}
    task_type = str(row.get("task_type", "unknown"))
    goal = pseudo_cover_goal(row) if task_type == "covered_long_signal" else raw_goal
    row2 = dict(row)
    row2["goal"] = goal
    signal = str(row2.get("signal", ""))
    base_memory = full_graph_memory_ids(row2, signal)
    names = list(goal_session_name_map(goal).keys())
    out: List[Dict[str, Any]] = []

    if task_type == "covered_long_signal":
        covs = raw_goal.get("covered_mappings", []) or []
        if covs:
            partial_created = [f"covered_{i}" for i in range(max(len(covs) - 1, 0))]
            create_state = materialize_state(
                row2,
                created_names=partial_created,
                linked_edges=[],
                attachments_done=[],
                covered_done=[],
                proposed_adds=[],
                proposed_no_op=False,
                include_memory_ids=base_memory,
                phase="create",
            )
            create_state = add_extra_session_node(create_state, signal, idx=2)
            allowed = exclusive_actions_for_phase(row2, create_state, "create")
            if allowed:
                out.append({
                    "id": f"{row.get('id', 'row')}::progress_recovery::cover_create_incomplete",
                    "source_goal_id": row.get("id"),
                    "graph_path": row.get("graph_path"),
                    "task_type": row.get("task_type", "unknown"),
                    "signal": row.get("signal"),
                    "spans": row.get("spans"),
                    "goal": goal,
                    "state": create_state,
                    "allowed_next": allowed,
                    "phase": "create",
                    "phase_exclusive": True,
                    "v1a6_action_history": True,
                    "state_variant": "cover_create_incomplete",
                })
            covered_done = [
                (f"covered_{i}", str(covs[i].get("memory_id", "")))
                for i in range(max(len(covs) - 1, 0))
            ]
            state = materialize_state(
                row2,
                created_names=[f"covered_{i}" for i in range(len(covs))],
                linked_edges=[],
                attachments_done=[],
                covered_done=covered_done,
                proposed_adds=[],
                proposed_no_op=False,
                include_memory_ids=base_memory,
                phase="cover",
            )
            state = add_extra_session_node(state, signal, idx=0)
            allowed = exclusive_actions_for_phase(row2, state, "cover")
            if allowed:
                out.append({
                    "id": f"{row.get('id', 'row')}::progress_recovery::cover_incomplete",
                    "source_goal_id": row.get("id"),
                    "graph_path": row.get("graph_path"),
                    "task_type": row.get("task_type", "unknown"),
                    "signal": row.get("signal"),
                    "spans": row.get("spans"),
                    "goal": goal,
                    "state": state,
                    "allowed_next": allowed,
                    "phase": "cover",
                    "phase_exclusive": True,
                    "v1a6_action_history": True,
                    "state_variant": "cover_incomplete",
                })
            state = materialize_state(
                row2,
                created_names=[f"covered_{i}" for i in range(len(covs))],
                linked_edges=[],
                attachments_done=[],
                covered_done=[(f"covered_{i}", str(covs[i].get("memory_id", ""))) for i in range(len(covs))],
                proposed_adds=[],
                proposed_no_op=False,
                include_memory_ids=base_memory,
                phase="noop",
            )
            state = add_extra_session_node(state, signal, idx=0)
            allowed = exclusive_actions_for_phase(row2, state, "noop")
            if allowed:
                out.append({
                    "id": f"{row.get('id', 'row')}::progress_recovery::cover_complete_no_noop",
                    "source_goal_id": row.get("id"),
                    "graph_path": row.get("graph_path"),
                    "task_type": row.get("task_type", "unknown"),
                    "signal": row.get("signal"),
                    "spans": row.get("spans"),
                    "goal": goal,
                    "state": state,
                    "allowed_next": allowed,
                    "phase": "noop",
                    "phase_exclusive": True,
                    "v1a6_action_history": True,
                    "state_variant": "cover_complete_no_noop",
                })
            drift_phase = "cover" if len(covs) > 1 else "noop"
            drift_state = materialize_state(
                row2,
                created_names=[f"covered_{i}" for i in range(len(covs))],
                linked_edges=[],
                attachments_done=[],
                covered_done=covered_done if drift_phase == "cover" else [(f"covered_{i}", str(covs[i].get("memory_id", ""))) for i in range(len(covs))],
                proposed_adds=[],
                proposed_no_op=False,
                include_memory_ids=base_memory,
                phase=drift_phase,
            )
            drift_state = add_extra_session_node(drift_state, signal, idx=1)
            allowed = exclusive_actions_for_phase(row2, drift_state, drift_phase)
            if allowed:
                out.append({
                    "id": f"{row.get('id', 'row')}::progress_recovery::false_terminal_drift",
                    "source_goal_id": row.get("id"),
                    "graph_path": row.get("graph_path"),
                    "task_type": row.get("task_type", "unknown"),
                    "signal": row.get("signal"),
                    "spans": row.get("spans"),
                    "goal": goal,
                    "state": drift_state,
                    "allowed_next": allowed,
                    "phase": drift_phase,
                    "phase_exclusive": True,
                    "v1a6_action_history": True,
                    "state_variant": "false_terminal_drift",
                })
        return out

    if names and goal.get("session_edges"):
        link_edges = [(str(e.get("src", "")), str(e.get("dst", "")), str(e.get("relation", "related"))) for e in goal.get("session_edges", []) or []]
        missing = rng.randrange(len(link_edges))
        present_edges = [e for i, e in enumerate(link_edges) if i != missing]
        state = materialize_state(
            row2,
            created_names=list(names),
            linked_edges=present_edges,
            attachments_done=[(str(a.get("session", "")), str(a.get("memory_id", "")), str(a.get("relation", "related"))) for a in goal.get("memory_attachments", []) or []],
            covered_done=[],
            proposed_adds=[str(goal.get("final_commits", [])[0].get("session", names[0]))] if goal.get("final_commits") else [],
            proposed_no_op=False,
            include_memory_ids=base_memory,
            phase="link",
        )
        state = add_extra_session_node(state, signal, idx=1)
        allowed = exclusive_actions_for_phase(row2, state, "link")
        if allowed:
            out.append({
                "id": f"{row.get('id', 'row')}::progress_recovery::link",
                "source_goal_id": row.get("id"),
                "graph_path": row.get("graph_path"),
                "task_type": row.get("task_type", "unknown"),
                "signal": row.get("signal"),
                "spans": row.get("spans"),
                "goal": goal,
                "state": state,
                "allowed_next": allowed,
                "phase": "link",
                "phase_exclusive": True,
                "v1a6_action_history": True,
                "state_variant": "imperfect_recovery",
            })

    if names and goal.get("memory_attachments"):
        all_atts = [(str(a.get("session", "")), str(a.get("memory_id", "")), str(a.get("relation", "related"))) for a in goal.get("memory_attachments", []) or []]
        missing = rng.randrange(len(all_atts))
        present_atts = [a for i, a in enumerate(all_atts) if i != missing]
        state = materialize_state(
            row2,
            created_names=list(names),
            linked_edges=[(str(e.get("src", "")), str(e.get("dst", "")), str(e.get("relation", "related"))) for e in goal.get("session_edges", []) or []],
            attachments_done=present_atts,
            covered_done=[],
            proposed_adds=[str(goal.get("final_commits", [])[0].get("session", names[0]))] if goal.get("final_commits") else [],
            proposed_no_op=False,
            include_memory_ids=base_memory,
            phase="attach",
        )
        state = add_extra_session_node(state, signal, idx=2)
        allowed = exclusive_actions_for_phase(row2, state, "attach")
        if allowed:
            out.append({
                "id": f"{row.get('id', 'row')}::progress_recovery::attach",
                "source_goal_id": row.get("id"),
                "graph_path": row.get("graph_path"),
                "task_type": row.get("task_type", "unknown"),
                "signal": row.get("signal"),
                "spans": row.get("spans"),
                "goal": goal,
                "state": state,
                "allowed_next": allowed,
                "phase": "attach",
                "phase_exclusive": True,
                "v1a6_action_history": True,
                "state_variant": "imperfect_recovery",
                })

    add_targets = [str(f.get("session", "")) for f in goal.get("final_commits", []) or [] if str(f.get("action", "")) == "add_node"]
    if add_targets:
        missing = rng.randrange(len(add_targets))
        proposed_adds = [s for i, s in enumerate(add_targets) if i != missing]
        state = materialize_state(
            row2,
            created_names=list(names),
            linked_edges=[(str(e.get("src", "")), str(e.get("dst", "")), str(e.get("relation", "related"))) for e in goal.get("session_edges", []) or []],
            attachments_done=[(str(a.get("session", "")), str(a.get("memory_id", "")), str(a.get("relation", "related"))) for a in goal.get("memory_attachments", []) or []],
            covered_done=[],
            proposed_adds=proposed_adds,
            proposed_no_op=False,
            include_memory_ids=base_memory,
            phase="add",
        )
        state = add_extra_session_node(state, signal, idx=3)
        allowed = exclusive_actions_for_phase(row2, state, "add")
        if allowed:
            out.append({
                "id": f"{row.get('id', 'row')}::progress_recovery::add",
                "source_goal_id": row.get("id"),
                "graph_path": row.get("graph_path"),
                "task_type": row.get("task_type", "unknown"),
                "signal": row.get("signal"),
                "spans": row.get("spans"),
                "goal": goal,
                "state": state,
                "allowed_next": allowed,
                "phase": "add",
                "phase_exclusive": True,
                "v1a6_action_history": True,
                "state_variant": "imperfect_recovery",
            })

    return out


def exclusive_actions_for_phase(row: Mapping[str, Any], state: Mapping[str, Any], phase: str) -> List[Dict[str, Any]]:
    goal = row.get("goal", {}) or {}
    signal = str(row.get("signal", ""))

    session_nodes = state.get("session_nodes", []) or []
    created_names = {str(s.get("name", "")) for s in session_nodes}
    name_to_idx = {str(s.get("name", "")): i for i, s in enumerate(session_nodes)}
    memory_ids = list(state.get("memory_node_ids", []) or [])
    memory_idx = {m: i for i, m in enumerate(memory_ids)}

    linked = {(str(e.get("src_name", "")), str(e.get("dst_name", "")), str(e.get("relation", "related"))) for e in state.get("session_edges", []) or []}
    attachments = {(str(a.get("session_name", "")), str(a.get("memory_id", "")), str(a.get("relation", "related"))) for a in state.get("attachments", []) or []}
    covered = {(str(c.get("session_name", "")), str(c.get("memory_id", ""))) for c in state.get("covered", []) or []}
    proposed_adds = set(str(x) for x in state.get("proposed_adds", []) or [])

    actions: List[Dict[str, Any]] = []

    if phase == "create":
        for name, spec in goal_session_name_map(goal).items():
            if name in created_names:
                continue
            span_text = clean_text(spec.get("span_text", ""))
            actions.append({
                "action": "CREATE_SESSION_NODE",
                "session_name": name,
                "span_idx": best_span_idx(signal, span_text),
                "span_text": span_text,
                "node_type": str(spec.get("node_type", "concept")),
            })

    elif phase == "link":
        for e in goal.get("session_edges", []) or []:
            src, dst, rel = str(e.get("src", "")), str(e.get("dst", "")), str(e.get("relation", "related"))
            if src in name_to_idx and dst in name_to_idx and (src, dst, rel) not in linked:
                actions.append({
                    "action": "LINK_SESSION_NODES",
                    "src_session_name": src,
                    "dst_session_name": dst,
                    "src_session_idx": name_to_idx[src],
                    "dst_session_idx": name_to_idx[dst],
                    "relation": rel,
                })

    elif phase == "attach":
        for a in goal.get("memory_attachments", []) or []:
            sname, mem, rel = str(a.get("session", "")), str(a.get("memory_id", "")), str(a.get("relation", "related"))
            if sname in name_to_idx and mem in memory_idx and (sname, mem, rel) not in attachments:
                actions.append({
                    "action": "PROPOSE_LINK_SESSION_TO_MEMORY",
                    "session_name": sname,
                    "session_idx": name_to_idx[sname],
                    "memory_id": mem,
                    "memory_idx": memory_idx[mem],
                    "relation": rel,
                })

    elif phase == "add":
        for f in goal.get("final_commits", []) or []:
            if str(f.get("action", "")) != "add_node":
                continue
            sname = str(f.get("session", ""))
            if sname in name_to_idx and sname not in proposed_adds:
                actions.append({
                    "action": "PROPOSE_ADD_SESSION_NODE",
                    "session_name": sname,
                    "session_idx": name_to_idx[sname],
                })

    elif phase == "cover":
        for i, cov in enumerate(goal.get("covered_mappings", []) or []):
            sname = f"covered_{i}"
            mem = str(cov.get("memory_id", ""))
            if sname in name_to_idx and mem in memory_idx and (sname, mem) not in covered:
                actions.append({
                    "action": "MARK_COVERED",
                    "session_name": sname,
                    "session_idx": name_to_idx[sname],
                    "memory_id": mem,
                    "memory_idx": memory_idx[mem],
                })

    elif phase == "noop":
        if not bool(state.get("proposed_no_op", False)):
            actions.append({"action": "PROPOSE_NO_OP"})

    elif phase == "stop":
        actions.append({"action": "STOP"})

    return actions


def make_state_for_phase(row: Mapping[str, Any], phase: str, rng: random.Random) -> Dict[str, Any]:
    raw_goal = row.get("goal", {}) or {}
    task_type = str(row.get("task_type", "unknown"))

    goal = pseudo_cover_goal(row) if task_type == "covered_long_signal" else raw_goal
    row2 = dict(row)
    row2["goal"] = goal

    names = list(goal_session_name_map(goal).keys())
    base_memory = full_graph_memory_ids(row2, str(row2.get("signal", "")))

    created: List[str] = []
    linked: List[Tuple[str, str, str]] = []
    atts: List[Tuple[str, str, str]] = []
    covered: List[Tuple[str, str]] = []
    proposed_adds: List[str] = []
    proposed_no_op = False

    if task_type == "covered_long_signal":
        covs = raw_goal.get("covered_mappings", []) or []
        if phase == "create":
            if len(covs) <= 1:
                created = []
            else:
                k = rng.randint(0, len(covs) - 1)
                picked = sorted(rng.sample(range(len(covs)), k=k)) if k else []
                created = [f"covered_{i}" for i in picked]
        elif phase == "cover":
            created = [f"covered_{i}" for i in range(len(covs))]
            if covs:
                k = rng.randint(0, len(covs) - 1)
                picked = sorted(rng.sample(range(len(covs)), k=k)) if k else []
                covered = [
                    (f"covered_{i}", str(covs[i].get("memory_id", "")))
                    for i in picked
                ]
        elif phase == "noop":
            created = [f"covered_{i}" for i in range(len(covs))]
            covered = [(f"covered_{i}", str(covs[i].get("memory_id", ""))) for i in range(len(covs))]
        elif phase == "stop":
            created = [f"covered_{i}" for i in range(len(covs))]
            covered = [(f"covered_{i}", str(covs[i].get("memory_id", ""))) for i in range(len(covs))]
            proposed_no_op = True

    else:
        if phase == "create":
            if len(names) <= 1:
                created = []
            else:
                k = rng.randint(0, max(0, len(names) - 1))
                created = rng.sample(names, k=k) if k else []
        elif phase == "link":
            created = list(names)
        elif phase == "attach":
            created = list(names)
            for e in goal.get("session_edges", []) or []:
                linked.append((str(e.get("src", "")), str(e.get("dst", "")), str(e.get("relation", "related"))))
        elif phase == "add":
            created = list(names)
            for e in goal.get("session_edges", []) or []:
                linked.append((str(e.get("src", "")), str(e.get("dst", "")), str(e.get("relation", "related"))))
            for a in goal.get("memory_attachments", []) or []:
                atts.append((str(a.get("session", "")), str(a.get("memory_id", "")), str(a.get("relation", "related"))))
        elif phase == "stop":
            created = list(names)
            for e in goal.get("session_edges", []) or []:
                linked.append((str(e.get("src", "")), str(e.get("dst", "")), str(e.get("relation", "related"))))
            for a in goal.get("memory_attachments", []) or []:
                atts.append((str(a.get("session", "")), str(a.get("memory_id", "")), str(a.get("relation", "related"))))
            for f in goal.get("final_commits", []) or []:
                if str(f.get("action", "")) == "add_node":
                    proposed_adds.append(str(f.get("session", "")))

    state = materialize_state(
        row2,
        created_names=created,
        linked_edges=linked,
        attachments_done=atts,
        covered_done=covered,
        proposed_adds=proposed_adds,
        proposed_no_op=proposed_no_op,
        include_memory_ids=base_memory,
        phase=phase,
    )
    return {"row": row2, "goal": goal, "state": state}


PHASES_NON_NOOP = ["create", "link", "link", "link", "attach", "attach", "add", "stop"]
PHASES_COVERED = ["create", "cover", "cover", "cover", "noop", "noop", "noop", "stop"]


def available_phases(row: Mapping[str, Any]) -> List[str]:
    goal = row.get("goal", {}) or {}
    task = str(row.get("task_type", "unknown"))
    if task == "covered_long_signal":
        return PHASES_COVERED

    phases = ["create", "add", "stop"]
    if goal.get("session_edges"):
        phases.extend(["link", "link", "link"])
    if goal.get("memory_attachments"):
        phases.extend(["attach", "attach"])
    return phases


def make_progress_states_for_row(row: Mapping[str, Any], *, states_per_goal: int, rng: random.Random) -> List[Dict[str, Any]]:
    phases = available_phases(row)
    out = []
    attempts = max(states_per_goal * 4, len(phases))
    for i in range(attempts):
        if len(out) >= states_per_goal:
            break
        phase = phases[i % len(phases)]
        built = make_state_for_phase(row, phase, rng)
        row2 = built["row"]
        state = built["state"]
        state_variant = None
        if phase == "add":
            state = add_extra_session_node(state, str(row.get("signal", "")), idx=3)
        if phase == "create" and str(row.get("task_type", "")) == "covered_long_signal":
            state = add_extra_session_node(state, str(row.get("signal", "")), idx=1)
            state_variant = "cover_create_incomplete"
        if phase == "noop":
            state = add_extra_session_node(state, str(row.get("signal", "")), idx=0)
            state_variant = "cover_complete_no_noop" if str(row.get("task_type", "")) == "covered_long_signal" else "imperfect_recovery"
        allowed = exclusive_actions_for_phase(row2, state, phase)
        if not allowed:
            continue
        rec = {
            "id": f"{row.get('id', 'row')}::progress::{len(out):02d}",
            "source_goal_id": row.get("id"),
            "graph_path": row.get("graph_path"),
            "task_type": row.get("task_type", "unknown"),
            "signal": row.get("signal"),
            "spans": row.get("spans"),
            "goal": built["goal"],
            "state": state,
            "allowed_next": allowed,
            "phase": phase,
            "phase_exclusive": True,
            "v1a6_action_history": True,
        }
        if state_variant:
            rec["state_variant"] = state_variant
        out.append(rec)
    return out


def convert(rows: Sequence[Mapping[str, Any]], *, states_per_goal: int, seed: int) -> List[Dict[str, Any]]:
    rng = random.Random(seed)
    out = []
    for row in rows:
        out.extend(make_progress_states_for_row(row, states_per_goal=states_per_goal, rng=rng))
        out.extend(recovery_rows_for_goal(row, rng))
    rng.shuffle(out)
    return out


def summarize(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    tasks = Counter()
    phases = Counter()
    actions = Counter()
    multi_action_type_rows = 0
    rows_with_action_history = 0
    imperfect_recovery_rows = 0

    for r in rows:
        tasks[str(r.get("task_type", "unknown"))] += 1
        phases[str(r.get("phase", "unknown"))] += 1
        if str(r.get("state_variant", "")) == "imperfect_recovery":
            imperfect_recovery_rows += 1
        if r.get("state", {}).get("action_history"):
            rows_with_action_history += 1
        action_types = {str(a.get("action", "unknown")) for a in r.get("allowed_next", []) or []}
        if len(action_types) > 1:
            multi_action_type_rows += 1
        for a in r.get("allowed_next", []) or []:
            actions[str(a.get("action", "unknown"))] += 1

    return {
        "rows": len(rows),
        "task_counts": dict(tasks),
        "phase_counts": dict(phases),
        "allowed_action_counts": dict(actions),
        "rows_with_action_history": rows_with_action_history,
        "imperfect_recovery_rows": imperfect_recovery_rows,
        "progress_state_rows": True,
        "phase_exclusive": True,
        "multi_action_type_rows": multi_action_type_rows,
        "allowed_next_is_action_tuples": True,
        "v1a6_action_history": True,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--goal-train-jsonl", required=True)
    ap.add_argument("--goal-val-jsonl", required=True)
    ap.add_argument("--out-dir", default="artifacts/tasks_v1a6_no_retrieval")
    ap.add_argument("--states-per-goal", type=int, default=12)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    train_goal = read_jsonl(args.goal_train_jsonl)
    val_goal = read_jsonl(args.goal_val_jsonl)

    train = convert(train_goal, states_per_goal=args.states_per_goal, seed=args.seed)
    val = convert(val_goal, states_per_goal=args.states_per_goal, seed=args.seed + 999)

    out_dir = Path(args.out_dir)
    write_jsonl(out_dir / "ngr_v1_progress_train.jsonl", train)
    write_jsonl(out_dir / "ngr_v1_progress_val.jsonl", val)

    summary = {
        "train": summarize(train),
        "val": summarize(val),
        "states_per_goal": args.states_per_goal,
        "source_train": args.goal_train_jsonl,
        "source_val": args.goal_val_jsonl,
        "v1a6": {
            "no_retrieval": True,
            "full_graph_memory": True,
            "pair_head_policy": True,
            "phase_exclusive_rows": True,
            "synthetic_action_history": True,
            "goal": "teach link-vs-attach-vs-cover-vs-noop without retrieval collapse",
        },
    }

    (out_dir / "ngr_v1_progress_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
