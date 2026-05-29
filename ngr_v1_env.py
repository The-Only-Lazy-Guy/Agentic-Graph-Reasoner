from __future__ import annotations

"""
ngr_v1_env.py

NGR-v1a no-retrieval environment.

Current behavior:
- Full graph memory pool is visible from reset.
- No learned retrieval action exists.
- Duplicate/useless structural actions are blocked in validity checks.

This keeps the active diagnosis focused on edit-program behavior instead of
memory-pool expansion.
"""

import argparse
import json
import re
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from graph_core import MemoryGraph, canonical_relation, lexical_overlap


ACTIONS = [
    "CREATE_SESSION_NODE",
    "LINK_SESSION_NODES",
    "MARK_COVERED",
    "PROPOSE_ADD_SESSION_NODE",
    "PROPOSE_LINK_SESSION_TO_MEMORY",
    "PROPOSE_LINK_MEMORY_TO_MEMORY",
    "PROPOSE_NO_OP",
    "STOP",
]

RELATIONS = [
    "related",
    "depend",
    "part_of",
    "precede",
    "cause",
    "support",
    "contradict",
    "refine",
    "example_of",
]

NODE_TYPES = [
    "concept",
    "claim",
    "fact",
    "summary",
    "hypothesis",
    "bridge",
    "unknown",
]

ACTION_TO_ID = {a: i for i, a in enumerate(ACTIONS)}
ID_TO_ACTION = {i: a for a, i in ACTION_TO_ID.items()}
REL_TO_ID = {r: i for i, r in enumerate(RELATIONS)}
ID_TO_REL = {i: r for r, i in REL_TO_ID.items()}
NODE_TYPE_TO_ID = {t: i for i, t in enumerate(NODE_TYPES)}
ID_TO_NODE_TYPE = {i: t for t, i in NODE_TYPE_TO_ID.items()}

V1_ACTIONS = ACTIONS
V1_RELATIONS = RELATIONS
V1_NODE_TYPES = NODE_TYPES


@dataclass
class SignalSpan:
    id: str
    text: str
    start: int = 0
    end: int = 0
    used_count: int = 0
    span_kind: str = "chunk"


@dataclass
class SessionNode:
    id: str
    span_indices: List[int]
    text: str
    node_type: str = "concept"
    created_step: int = 0
    covered_by: Optional[str] = None
    proposed_add: bool = False


@dataclass
class SessionEdge:
    src: int
    dst: int
    relation: str
    created_step: int = 0


@dataclass
class MemoryAttachment:
    session_idx: int
    memory_id: str
    relation: str
    proposed: bool = False


@dataclass
class MemoryMemoryProposal:
    src_memory_id: str
    dst_memory_id: str
    relation: str


@dataclass
class RetrievalRecord:
    query_source: str
    query_idx: int
    query_text: str
    returned_ids: List[str]
    returned_scores: List[float]
    max_score: float
    weak: bool
    step: int


@dataclass
class NGRV1Config:
    max_steps: int = 12
    max_spans: int = 32
    max_session_nodes: int = 16
    max_memory_nodes: int = 512
    min_steps_before_stop: int = 1
    weak_retrieval_threshold: float = 0.12
    enable_comma_item_spans: bool = True
    keep_full_signal_span: bool = True
    keep_parent_clause_spans: bool = True
    min_item_chars: int = 3


@dataclass
class NGRV1State:
    signal: str = ""
    spans: List[SignalSpan] = field(default_factory=list)
    memory_node_ids: List[str] = field(default_factory=list)
    memory_scores: Dict[str, float] = field(default_factory=dict)
    session_nodes: List[SessionNode] = field(default_factory=list)
    session_edges: List[SessionEdge] = field(default_factory=list)
    attachments: List[MemoryAttachment] = field(default_factory=list)
    memory_link_proposals: List[MemoryMemoryProposal] = field(default_factory=list)
    retrieval_history: List[RetrievalRecord] = field(default_factory=list)
    action_history: List[Dict[str, Any]] = field(default_factory=list)
    proposed_no_op: bool = False
    step: int = 0
    done: bool = False


def norm(text: Any) -> str:
    return " ".join(str(text or "").split())


def normalize_ws(text: Any) -> str:
    return norm(text)


def text_key(text: Any) -> str:
    return re.sub(r"[^a-z0-9_ ]+", "", norm(text).lower())


def make_session_id(step: int, span_indices: Sequence[int]) -> str:
    joined = "_".join(str(i) for i in span_indices)
    return f"s{step}_{joined}"


def _add_span_unique(
    spans: List[SignalSpan],
    seen: set[str],
    *,
    text: str,
    start: int,
    end: int,
    span_kind: str,
    max_spans: int,
) -> None:
    text = norm(text)
    if not text:
        return
    key = text.lower()
    if key in seen or len(spans) >= max_spans:
        return
    seen.add(key)
    spans.append(SignalSpan(
        id=f"span_{len(spans)}",
        text=text,
        start=max(0, int(start)),
        end=max(0, int(end)),
        span_kind=span_kind,
    ))


def _find_span_bounds(full_text: str, sub: str, cursor: int = 0) -> tuple[int, int]:
    sub = str(sub)
    pos = full_text.find(sub, cursor)
    if pos < 0:
        pos = full_text.lower().find(sub.lower(), cursor)
    if pos < 0:
        pos = cursor
    return pos, pos + len(sub)


def _split_intro_list(clause: str) -> tuple[str, List[str]]:
    c = norm(clause)
    lower = c.lower()
    markers = [
        " include ", " includes ", " including ",
        " consist of ", " consists of ",
        " contain ", " contains ",
        " are ", " is ",
        " involve ", " involves ",
    ]
    marker_pos = None
    marker = None
    for m in markers:
        idx = lower.find(m)
        if idx >= 0:
            marker_pos = idx
            marker = m
            break
    if marker_pos is None or marker is None:
        return "", []

    intro = c[:marker_pos].strip(" :,-")
    rest = c[marker_pos + len(marker):].strip(" :,-")
    if "," not in rest and " and " not in rest:
        return intro, []

    rest = re.sub(r"\s*,?\s+and\s+", ", ", rest)
    rest = re.sub(r"\s*,?\s+or\s+", ", ", rest)
    items = [norm(x.strip(" :,-")) for x in rest.split(",")]
    items = [x for x in items if x and len(x) >= 3]
    if len(items) < 2:
        return intro, []
    return intro, items


def split_signal_spans(signal: str, max_spans: int = 32, config: Optional[NGRV1Config] = None) -> List[SignalSpan]:
    cfg = config or NGRV1Config(max_spans=max_spans)
    max_spans = max_spans or cfg.max_spans
    text = str(signal or "").strip()
    if not text:
        return []

    spans: List[SignalSpan] = []
    seen: set[str] = set()

    if cfg.keep_full_signal_span:
        _add_span_unique(
            spans, seen,
            text=norm(text.rstrip(".")),
            start=0,
            end=len(text),
            span_kind="full",
            max_spans=max_spans,
        )

    pattern = r"(?:\n+|(?:^|\s)(?:\d+\)|\d+\.|[-*•])\s+|[.;])"
    raw_parts = [norm(x) for x in re.split(pattern, text) if norm(x)]
    if not raw_parts:
        raw_parts = [norm(text)]

    cursor = 0
    for part in raw_parts:
        start, end = _find_span_bounds(text, part, cursor)
        cursor = end

        if cfg.keep_parent_clause_spans:
            _add_span_unique(
                spans, seen,
                text=part,
                start=start,
                end=end,
                span_kind="clause",
                max_spans=max_spans,
            )

        if cfg.enable_comma_item_spans:
            intro, items = _split_intro_list(part)
            if intro and len(intro) >= cfg.min_item_chars:
                intro_start, intro_end = _find_span_bounds(text, intro, start)
                _add_span_unique(
                    spans, seen,
                    text=intro,
                    start=intro_start,
                    end=intro_end,
                    span_kind="item",
                    max_spans=max_spans,
                )
            local_cursor = start
            for item in items:
                if len(item) < cfg.min_item_chars:
                    continue
                item_start, item_end = _find_span_bounds(text, item, local_cursor)
                local_cursor = item_end
                _add_span_unique(
                    spans, seen,
                    text=item,
                    start=item_start,
                    end=item_end,
                    span_kind="item",
                    max_spans=max_spans,
                )

    base_indices = [i for i, s in enumerate(spans) if s.span_kind in {"item", "clause"}]
    for a, b in zip(base_indices, base_indices[1:]):
        if len(spans) >= max_spans:
            break
        sa, sb = spans[a], spans[b]
        _add_span_unique(
            spans, seen,
            text=norm(sa.text + " " + sb.text),
            start=sa.start,
            end=sb.end,
            span_kind="merged",
            max_spans=max_spans,
        )

    for i, s in enumerate(spans):
        s.id = f"span_{i}"
    return spans[:max_spans]


def parse_span_indices(action: Mapping[str, Any]) -> List[int]:
    if "span_indices" in action and action.get("span_indices") is not None:
        raw = action.get("span_indices")
        if isinstance(raw, list):
            return [int(x) for x in raw]
        return [int(raw)]
    if "span_idx" in action and action.get("span_idx") is not None:
        return [int(action.get("span_idx"))]
    raise ValueError("CREATE_SESSION_NODE requires span_indices or span_idx")


class NGRV1Env:
    def __init__(self, graph: MemoryGraph, config: Optional[NGRV1Config] = None) -> None:
        self.graph = graph
        self.config = config or NGRV1Config()
        self.state = NGRV1State()

    def reset(self, signal: str, task: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
        spans = split_signal_spans(signal, self.config.max_spans, self.config)
        scored = []
        for nid in self.graph.nodes:
            scored.append((self.score_node_for_query(str(signal or ""), nid), nid))
        scored.sort(key=lambda x: (-x[0], x[1]))
        initial_memory = [nid for _score, nid in scored]
        initial_scores = {nid: float(score) for score, nid in scored}

        self.state = NGRV1State(
            signal=str(signal),
            spans=spans,
            memory_node_ids=self._dedupe_cap(initial_memory),
            memory_scores=initial_scores,
        )
        return self.observe()

    def _dedupe_cap(self, ids: Sequence[str]) -> List[str]:
        seen = set()
        out = []
        for nid in ids:
            nid = str(nid)
            if nid in self.graph.nodes and nid not in seen:
                seen.add(nid)
                out.append(nid)
            if len(out) >= self.config.max_memory_nodes:
                break
        return out

    def score_node_for_query(self, query: str, nid: str) -> float:
        n = self.graph.nodes[nid]
        txt = f"{nid.replace('_', ' ')} {n.text}"
        score = float(lexical_overlap(query, txt))
        score += 0.02 * float(getattr(n, "importance", 0.5))
        score += 0.01 * float(getattr(n, "confidence", 0.5))
        return float(score)

    def observe(self) -> Dict[str, Any]:
        memory_nodes = []
        mem_index = {nid: i for i, nid in enumerate(self.state.memory_node_ids)}

        for i, nid in enumerate(self.state.memory_node_ids):
            n = self.graph.nodes[nid]
            memory_nodes.append({
                "idx": i,
                "id": nid,
                "text": str(n.text),
                "node_type": str(n.node_type),
                "confidence": float(n.confidence),
                "importance": float(n.importance),
                "signal_overlap": float(lexical_overlap(self.state.signal, f"{nid.replace('_', ' ')} {n.text}")),
                "retrieval_score": float(self.state.memory_scores.get(nid, 0.0)),
            })

        memory_edges = []
        try:
            local_edges = self.graph.iter_local_edges(self.state.memory_node_ids)
        except Exception:
            local_edges = []

        for e in local_edges:
            if e.src in mem_index and e.dst in mem_index:
                memory_edges.append({
                    "src_idx": mem_index[e.src],
                    "dst_idx": mem_index[e.dst],
                    "src": e.src,
                    "dst": e.dst,
                    "relation": canonical_relation(e.relation),
                    "strength": float(getattr(e, "strength", 1.0)),
                })

        last_retrieval = self.state.retrieval_history[-1] if self.state.retrieval_history else None

        return {
            "signal": self.state.signal,
            "step": self.state.step,
            "budget_left": self.config.max_steps - self.state.step,
            "done": self.state.done,
            "spans": [asdict(s) for s in self.state.spans],
            "memory_nodes": memory_nodes,
            "memory_edges": memory_edges,
            "session_nodes": [asdict(s) for s in self.state.session_nodes],
            "session_edges": [asdict(e) for e in self.state.session_edges],
            "attachments": [asdict(a) for a in self.state.attachments],
            "memory_link_proposals": [asdict(p) for p in self.state.memory_link_proposals],
            "retrieval_history": [asdict(r) for r in self.state.retrieval_history],
            "last_retrieval_weak": bool(last_retrieval.weak) if last_retrieval else None,
            "last_retrieval_max_score": float(last_retrieval.max_score) if last_retrieval else None,
            "action_history": list(self.state.action_history),
            "proposed_no_op": self.state.proposed_no_op,
            "actions": ACTIONS,
            "relations": RELATIONS,
            "node_types": NODE_TYPES,
        }

    def _session_span_set_exists(self, span_indices: Sequence[int]) -> bool:
        sset = set(int(x) for x in span_indices)
        return any(set(sn.span_indices) == sset for sn in self.state.session_nodes)

    def _session_text_exists(self, text: str) -> bool:
        key = text_key(text)
        return any(text_key(sn.text) == key for sn in self.state.session_nodes)

    def _has_new_creatable_span(self) -> bool:
        if len(self.state.session_nodes) >= self.config.max_session_nodes:
            return False
        existing_sets = {tuple(sorted(sn.span_indices)) for sn in self.state.session_nodes}
        existing_texts = {text_key(sn.text) for sn in self.state.session_nodes}
        for i, sp in enumerate(self.state.spans):
            if (i,) not in existing_sets and text_key(sp.text) not in existing_texts:
                return True
        return False

    def _has_uncovered_session(self) -> bool:
        if not self.state.session_nodes:
            return False
        if not self.state.memory_node_ids:
            return False
        for sn in self.state.session_nodes:
            if sn.covered_by is None:
                return True
            if any(mem != sn.covered_by for mem in self.state.memory_node_ids):
                return True
        return False

    def _has_unproposed_add_session(self) -> bool:
        return any((not sn.proposed_add) and sn.covered_by is None for sn in self.state.session_nodes)

    def _has_possible_new_attachment(self) -> bool:
        if not self.state.session_nodes or not self.state.memory_node_ids:
            return False
        existing = {(a.session_idx, a.memory_id, a.relation) for a in self.state.attachments}
        for si, _sn in enumerate(self.state.session_nodes):
            for mem in self.state.memory_node_ids:
                for rel in RELATIONS:
                    if (si, mem, rel) not in existing:
                        return True
        return False

    def _has_possible_new_session_edge(self) -> bool:
        if len(self.state.session_nodes) < 2:
            return False
        existing = {(e.src, e.dst, e.relation) for e in self.state.session_edges}
        for i in range(len(self.state.session_nodes)):
            for j in range(len(self.state.session_nodes)):
                if i == j:
                    continue
                for rel in RELATIONS:
                    if (i, j, rel) not in existing:
                        return True
        return False

    def valid_action_mask(self) -> Dict[str, Any]:
        s = self.state
        n_spans = len(s.spans)
        n_session = len(s.session_nodes)
        n_memory = len(s.memory_node_ids)

        action_valid = {a: True for a in ACTIONS}

        if n_spans == 0 or not self._has_new_creatable_span():
            action_valid["CREATE_SESSION_NODE"] = False

        if not self._has_possible_new_session_edge():
            action_valid["LINK_SESSION_NODES"] = False

        if n_session < 1 or n_memory < 1 or not self._has_uncovered_session():
            action_valid["MARK_COVERED"] = False

        if n_session < 1 or not self._has_unproposed_add_session():
            action_valid["PROPOSE_ADD_SESSION_NODE"] = False

        if n_session < 1 or n_memory < 1 or not self._has_possible_new_attachment():
            action_valid["PROPOSE_LINK_SESSION_TO_MEMORY"] = False

        if n_memory < 2:
            action_valid["PROPOSE_LINK_MEMORY_TO_MEMORY"] = False

        if s.proposed_no_op:
            action_valid["PROPOSE_NO_OP"] = False

        if s.step < self.config.min_steps_before_stop:
            action_valid["STOP"] = False

        if s.step >= self.config.max_steps:
            action_valid = {a: False for a in ACTIONS}
            action_valid["STOP"] = True

        return {
            "action": [1 if action_valid[a] else 0 for a in ACTIONS],
            "span": [0 if self._session_span_set_exists([i]) or self._session_text_exists(sp.text) else 1 for i, sp in enumerate(s.spans)],
            "session": [1] * n_session,
            "memory": [1] * n_memory,
            "relation": [1] * len(RELATIONS),
            "node_type": [1] * len(NODE_TYPES),
        }

    def validate_action(self, action: Mapping[str, Any]) -> Tuple[bool, List[str]]:
        errors: List[str] = []
        a = str(action.get("action", ""))

        if a not in ACTION_TO_ID:
            return False, [f"unknown action: {a}"]

        mask = self.valid_action_mask()
        if not mask["action"][ACTION_TO_ID[a]]:
            errors.append(f"action not currently valid: {a}")

        def check_idx(name: str, idx: Any, n: int) -> None:
            try:
                i = int(idx)
            except Exception:
                errors.append(f"{name} must be int")
                return
            if i < 0 or i >= n:
                errors.append(f"{name} out of range: {i}")

        if a == "CREATE_SESSION_NODE":
            try:
                inds = sorted(set(parse_span_indices(action)))
                if not inds:
                    errors.append("CREATE_SESSION_NODE requires non-empty span_indices")
                for x in inds:
                    check_idx("span_idx", x, len(self.state.spans))
                if not errors:
                    txt = norm(" ".join(self.state.spans[i].text for i in inds))
                    if self._session_span_set_exists(inds):
                        errors.append(f"duplicate session span set: {inds}")
                    if self._session_text_exists(txt):
                        errors.append(f"duplicate session node text: {txt}")
            except Exception as exc:
                errors.append(str(exc))

        elif a == "LINK_SESSION_NODES":
            check_idx("src_session_idx", action.get("src_session_idx"), len(self.state.session_nodes))
            check_idx("dst_session_idx", action.get("dst_session_idx"), len(self.state.session_nodes))
            if action.get("src_session_idx") == action.get("dst_session_idx"):
                errors.append("session src/dst must differ")
            rel = str(action.get("relation", "related"))
            rel = rel if rel in RELATIONS else canonical_relation(rel)
            if rel not in RELATIONS:
                errors.append(f"bad relation: {action.get('relation')}")
            elif not errors:
                src = int(action["src_session_idx"])
                dst = int(action["dst_session_idx"])
                if any(e.src == src and e.dst == dst and e.relation == rel for e in self.state.session_edges):
                    errors.append(f"duplicate session edge: {src}->{dst}:{rel}")

        elif a == "MARK_COVERED":
            check_idx("session_idx", action.get("session_idx"), len(self.state.session_nodes))
            check_idx("memory_idx", action.get("memory_idx"), len(self.state.memory_node_ids))
            if not errors:
                si = int(action["session_idx"])
                target_mem = self.state.memory_node_ids[int(action["memory_idx"])]
                if self.state.session_nodes[si].covered_by == target_mem:
                    errors.append("session node already marked covered by target memory")
                if self.state.session_nodes[si].proposed_add:
                    errors.append("cannot mark covered after proposed add")

        elif a == "PROPOSE_ADD_SESSION_NODE":
            check_idx("session_idx", action.get("session_idx"), len(self.state.session_nodes))
            if not errors:
                si = int(action["session_idx"])
                sn = self.state.session_nodes[si]
                if sn.proposed_add:
                    errors.append("session node already proposed for add")
                if sn.covered_by is not None:
                    errors.append("cannot propose add for covered session node")

        elif a == "PROPOSE_LINK_SESSION_TO_MEMORY":
            check_idx("session_idx", action.get("session_idx"), len(self.state.session_nodes))
            check_idx("memory_idx", action.get("memory_idx"), len(self.state.memory_node_ids))
            rel = str(action.get("relation", "related"))
            rel = rel if rel in RELATIONS else canonical_relation(rel)
            if rel not in RELATIONS:
                errors.append(f"bad relation: {action.get('relation')}")
            if not errors:
                si = int(action["session_idx"])
                mem = self.state.memory_node_ids[int(action["memory_idx"])]
                if any(att.session_idx == si and att.memory_id == mem and att.relation == rel for att in self.state.attachments):
                    errors.append(f"duplicate session-memory attachment: {si}->{mem}:{rel}")

        elif a == "PROPOSE_LINK_MEMORY_TO_MEMORY":
            check_idx("src_memory_idx", action.get("src_memory_idx"), len(self.state.memory_node_ids))
            check_idx("dst_memory_idx", action.get("dst_memory_idx"), len(self.state.memory_node_ids))
            if action.get("src_memory_idx") == action.get("dst_memory_idx"):
                errors.append("memory src/dst must differ")
            rel = str(action.get("relation", "related"))
            rel = rel if rel in RELATIONS else canonical_relation(rel)
            if rel not in RELATIONS:
                errors.append(f"bad relation: {action.get('relation')}")
            if not errors:
                src = self.state.memory_node_ids[int(action["src_memory_idx"])]
                dst = self.state.memory_node_ids[int(action["dst_memory_idx"])]
                if any(p.src_memory_id == src and p.dst_memory_id == dst and p.relation == rel for p in self.state.memory_link_proposals):
                    errors.append(f"duplicate memory-memory proposal: {src}->{dst}:{rel}")

        elif a == "PROPOSE_NO_OP":
            if self.state.proposed_no_op:
                errors.append("no_op already proposed")

        elif a == "STOP":
            pass

        return not errors, errors

    def _record_action(self, action: Mapping[str, Any], valid: bool, errors: List[str]) -> None:
        rec = dict(action)
        rec["valid"] = valid
        if errors:
            rec["errors"] = errors
        rec["step"] = self.state.step
        self.state.action_history.append(rec)

    def step(self, action: Mapping[str, Any]) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        valid, errors = self.validate_action(action)
        reward = 0.0

        if not valid:
            reward = -1.0
            self._record_action(action, valid, errors)
            self.state.step += 1
            if self.state.step >= self.config.max_steps:
                self.state.done = True
            return self.observe(), reward, self.state.done, {"valid": False, "errors": errors}

        a = str(action["action"])

        if a == "CREATE_SESSION_NODE":
            inds = sorted(set(parse_span_indices(action)))
            txt = norm(" ".join(self.state.spans[i].text for i in inds))
            nt = str(action.get("node_type", "concept"))
            nt = nt if nt in NODE_TYPES else "concept"

            for i in inds:
                self.state.spans[i].used_count += 1

            self.state.session_nodes.append(SessionNode(
                id=make_session_id(self.state.step, inds),
                span_indices=inds,
                text=txt,
                node_type=nt,
                created_step=self.state.step,
            ))

        elif a == "LINK_SESSION_NODES":
            src = int(action["src_session_idx"])
            dst = int(action["dst_session_idx"])
            rel = str(action.get("relation", "related"))
            rel = rel if rel in RELATIONS else canonical_relation(rel)
            if rel not in RELATIONS:
                rel = "related"
            self.state.session_edges.append(SessionEdge(src=src, dst=dst, relation=rel, created_step=self.state.step))

        elif a == "MARK_COVERED":
            si = int(action["session_idx"])
            mi = int(action["memory_idx"])
            self.state.session_nodes[si].covered_by = self.state.memory_node_ids[mi]

        elif a == "PROPOSE_ADD_SESSION_NODE":
            si = int(action["session_idx"])
            self.state.session_nodes[si].proposed_add = True

        elif a == "PROPOSE_LINK_SESSION_TO_MEMORY":
            si = int(action["session_idx"])
            mi = int(action["memory_idx"])
            rel = str(action.get("relation", "related"))
            rel = rel if rel in RELATIONS else canonical_relation(rel)
            if rel not in RELATIONS:
                rel = "related"
            self.state.attachments.append(MemoryAttachment(
                session_idx=si,
                memory_id=self.state.memory_node_ids[mi],
                relation=rel,
                proposed=True,
            ))

        elif a == "PROPOSE_LINK_MEMORY_TO_MEMORY":
            src = self.state.memory_node_ids[int(action["src_memory_idx"])]
            dst = self.state.memory_node_ids[int(action["dst_memory_idx"])]
            rel = str(action.get("relation", "related"))
            rel = rel if rel in RELATIONS else canonical_relation(rel)
            if rel not in RELATIONS:
                rel = "related"
            self.state.memory_link_proposals.append(MemoryMemoryProposal(src, dst, rel))

        elif a == "PROPOSE_NO_OP":
            self.state.proposed_no_op = True

        elif a == "STOP":
            self.state.done = True

        self._record_action(action, True, [])
        self.state.step += 1

        if self.state.step >= self.config.max_steps:
            self.state.done = True

        return self.observe(), reward, self.state.done, {"valid": True}

    def serialize_commit_plan(self) -> Dict[str, Any]:
        edits: List[Dict[str, Any]] = []
        session_to_ref: Dict[int, str] = {}

        for i, sn in enumerate(self.state.session_nodes):
            if sn.proposed_add:
                new_id = "new_" + re.sub(r"[^a-z0-9_]+", "_", sn.id.lower()).strip("_")
                session_to_ref[i] = new_id
                edits.append({
                    "action": "add_node",
                    "from_session_idx": i,
                    "proposed": {"id": new_id, "text": sn.text, "node_type": sn.node_type},
                })
            elif sn.covered_by:
                session_to_ref[i] = sn.covered_by

        for e in self.state.session_edges:
            if e.src in session_to_ref and e.dst in session_to_ref:
                src_ref = session_to_ref[e.src]
                dst_ref = session_to_ref[e.dst]
                if src_ref != dst_ref:
                    edits.append({
                        "action": "link_nodes",
                        "from_session_edge": {"src": e.src, "dst": e.dst},
                        "proposed": {"src": src_ref, "dst": dst_ref, "relation": e.relation},
                    })

        for att in self.state.attachments:
            if att.session_idx in session_to_ref:
                edits.append({
                    "action": "link_nodes",
                    "from_attachment": att.session_idx,
                    "proposed": {
                        "src": session_to_ref[att.session_idx],
                        "dst": att.memory_id,
                        "relation": att.relation,
                    },
                })

        for p in self.state.memory_link_proposals:
            edits.append({
                "action": "link_nodes",
                "from_memory_proposal": True,
                "proposed": {"src": p.src_memory_id, "dst": p.dst_memory_id, "relation": p.relation},
            })

        if self.state.proposed_no_op and not edits:
            edits.append({"action": "no_op", "proposed": {"reason": "all relevant session concepts marked covered or no long-term edit needed"}})

        return {
            "final_memory_edits": edits,
            "session_nodes": [asdict(s) for s in self.state.session_nodes],
            "session_edges": [asdict(e) for e in self.state.session_edges],
            "attachments": [asdict(a) for a in self.state.attachments],
            "proposed_no_op": self.state.proposed_no_op,
        }

    def serialize_trajectory(self) -> Dict[str, Any]:
        return {
            "signal": self.state.signal,
            "steps": list(self.state.action_history),
            "retrieval_history": [asdict(r) for r in self.state.retrieval_history],
            "commit_plan": self.serialize_commit_plan(),
        }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--graph", required=True)
    ap.add_argument("--signal", required=True)
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args()

    graph = MemoryGraph.load_json(args.graph)
    env = NGRV1Env(graph)
    obs = env.reset(args.signal)

    print(json.dumps({
        "spans": obs["spans"],
        "memory_nodes": obs["memory_nodes"][:8],
        "last_retrieval_weak": obs["last_retrieval_weak"],
        "last_retrieval_max_score": obs["last_retrieval_max_score"],
        "valid_action_mask": env.valid_action_mask(),
    }, indent=2, ensure_ascii=False))

    if args.debug:
        item_span_indices = [i for i, s in enumerate(env.state.spans) if s.span_kind == "item"]
        for idx in item_span_indices[:3] or ([0] if env.state.spans else []):
            _, _, _, info = env.step({"action": "CREATE_SESSION_NODE", "span_indices": [idx], "node_type": "concept"})
            print(f"debug create span {idx}:", json.dumps(info, ensure_ascii=False))

        if len(env.state.session_nodes) >= 2:
            _, _, _, info = env.step({"action": "LINK_SESSION_NODES", "src_session_idx": 0, "dst_session_idx": 1, "relation": "precede"})
            print("debug link session:", json.dumps(info, ensure_ascii=False))

        if env.state.session_nodes:
            _, _, _, info = env.step({"action": "PROPOSE_ADD_SESSION_NODE", "session_idx": 0})
            print("debug propose add:", json.dumps(info, ensure_ascii=False))

        _, _, _, info = env.step({"action": "STOP"})
        print("debug stop:", json.dumps(info, ensure_ascii=False))
        print(json.dumps(env.serialize_trajectory(), indent=2, ensure_ascii=False))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
