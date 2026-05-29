from __future__ import annotations
import os

os.environ["HF_HOME"] = os.path.join(os.getcwd(), "cache")

import json
import re
from collections import deque
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from reasoning.lexical_matching import lexical_overlap as _shared_lexical_overlap
from reasoning.lexical_matching import lexical_tokens as _shared_lexical_tokens


# -----------------------------
# Label normalization
# -----------------------------

NODE_TYPE_ALIAS_MAP: Dict[str, str] = {
    "facts": "fact", "statements": "statement", "summaries": "summary", "overviews": "summary",
    "examples": "example", "instances": "example", "claims": "claim", "hypotheses": "hypothesis",
    "theorems": "theorem", "lemmas": "lemma", "equations": "equation", "definitions": "definition",
    "proofs": "proof", "theory": "theory", "theories": "theory", "explanation": "explanation",
    "explanations": "explanation", "bridges": "bridge", "entities": "entity", "concepts": "concept",
    "hubs": "hub", "summary_hub": "hub",
}

RELATION_ALIAS_MAP: Dict[str, str] = {
    "supports": "support", "supported": "support", "evidence": "support", "evidence_for": "support",
    "proves": "support", "prove": "support", "supported_by": "support", "implies": "imply",
    "implied_by": "imply", "causes": "cause", "caused_by": "cause", "leads_to": "cause",
    "results_in": "cause", "contradicts": "contradict", "conflicts_with": "conflict",
    "refutes": "refute", "part_of": "part_of", "is_part_of": "part_of", "example_of": "example_of",
    "instances_of": "example_of", "instance_of": "example_of", "refines": "refine", "related_to": "related",
    "connects": "related", "connect": "related", "links": "related", "link": "related",
    "associated_with": "related", "depends_on": "depend", "dependent_on": "depend",
    "precedes": "precede", "follows": "follow",
}

NODE_TYPE_FAMILY_MAP: Dict[str, str] = {
    "summary": "summary", "hub": "summary", "overview": "summary",
    "concept": "concept", "definition": "concept", "theorem": "concept", "lemma": "concept",
    "proof": "concept", "law": "concept", "bridge": "concept", "theory": "concept",
    "equation": "fact", "fact": "fact", "statement": "fact", "explanation": "fact",
    "claim": "claim", "hypothesis": "claim", "example": "example", "entity": "entity", "issue": "entity",
    "unknown": "unknown",
}

RELATION_FAMILY_MAP: Dict[str, str] = {
    "support": "support", "imply": "support", "refine": "support",
    "cause": "causal", "depend": "causal",
    "contradict": "contradict", "conflict": "contradict", "refute": "contradict",
    "part_of": "part_of", "example_of": "part_of", "related": "related",
    "precede": "temporal", "follow": "temporal", "unknown": "unknown",
}

CONTRADICTION_RELATIONS = {"contradict", "conflict", "refute"}
SUPPORT_RELATIONS = {"support", "imply", "part_of", "example_of", "refine", "cause", "depend"}
CANONICAL_RELATIONS = ["support", "contradict", "refine", "depend", "cause", "part_of", "example_of", "related"]
RELATION_TO_ID = {r: i for i, r in enumerate(CANONICAL_RELATIONS)}
ID_TO_RELATION = {i: r for r, i in RELATION_TO_ID.items()}


def normalize_label(text: str) -> str:
    text = str(text or "").strip().lower()
    if not text:
        return "unknown"
    text = text.replace("-", "_").replace(" ", "_").replace("/", "_")
    text = re.sub(r"[^a-z0-9_]+", "", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return text or "unknown"


def _safe_singularize(label: str) -> str:
    if label.endswith("ies") and len(label) > 3:
        return label[:-3] + "y"
    if label.endswith("sses") and len(label) > 4:
        return label[:-2]
    if label.endswith("ses") and len(label) > 3:
        return label[:-2]
    if label.endswith("s") and len(label) > 3 and not label.endswith(("ss", "us", "is")):
        return label[:-1]
    return label


def canonical_node_type(raw: str) -> str:
    label = _safe_singularize(normalize_label(raw))
    return NODE_TYPE_ALIAS_MAP.get(label, label or "unknown")


def canonical_relation(raw: str) -> str:
    label = _safe_singularize(normalize_label(raw))
    return RELATION_ALIAS_MAP.get(label, label or "unknown")


def node_type_family(node_type: str) -> str:
    return NODE_TYPE_FAMILY_MAP.get(canonical_node_type(node_type), "unknown")


def relation_family(relation: str) -> str:
    return RELATION_FAMILY_MAP.get(canonical_relation(relation), "unknown")


def relation_polarity(relation: str) -> float:
    return -1.0 if canonical_relation(relation) in CONTRADICTION_RELATIONS else 1.0


# -----------------------------
# Graph model
# -----------------------------

@dataclass
class Node:
    id: str
    text: str
    node_type: str = "claim"
    confidence: float = 0.5
    importance: float = 0.5
    metadata: Dict[str, Any] = field(default_factory=dict)

    @staticmethod
    def from_dict(d: Mapping[str, Any]) -> "Node":
        return Node(
            id=str(d["id"]),
            text=str(d.get("text", "")),
            node_type=str(d.get("node_type", d.get("type", "claim"))),
            confidence=float(d.get("confidence", 0.5)),
            importance=float(d.get("importance", 0.5)),
            metadata=dict(d.get("metadata", {}) or {}),
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class Edge:
    src: str
    dst: str
    relation: str = "related"
    strength: float = 0.5
    directed: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)

    @staticmethod
    def from_dict(d: Mapping[str, Any]) -> "Edge":
        return Edge(
            src=str(d.get("src", d.get("source", ""))),
            dst=str(d.get("dst", d.get("target", ""))),
            relation=str(d.get("relation", "related")),
            strength=float(d.get("strength", 0.5)),
            directed=bool(d.get("directed", True)),
            metadata=dict(d.get("metadata", {}) or {}),
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class MemoryGraph:
    def __init__(self, nodes: Mapping[str, Node], edges: Sequence[Edge], metadata: Optional[Dict[str, Any]] = None) -> None:
        self.nodes: Dict[str, Node] = dict(nodes)
        self.edges: List[Edge] = list(edges)
        self.metadata: Dict[str, Any] = dict(metadata or {})
        self._adj: Dict[str, List[str]] = {}
        self._adj_out: Dict[str, List[str]] = {}
        self._edge_by_pair: Dict[Tuple[str, str], Edge] = {}
        self._rebuild_index()

    @staticmethod
    def load_json(path: str | Path) -> "MemoryGraph":
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        nodes = {str(n["id"]): Node.from_dict(n) for n in raw.get("nodes", [])}
        edges = []
        for e in raw.get("edges", []) or []:
            edge = Edge.from_dict(e)
            if edge.src in nodes and edge.dst in nodes:
                edges.append(edge)
        return MemoryGraph(nodes, edges, metadata=dict(raw.get("metadata", {}) or {}))

    def save_json(self, path: str | Path) -> None:
        obj = {
            "metadata": self.metadata,
            "nodes": [n.to_dict() for n in self.nodes.values()],
            "edges": [e.to_dict() for e in self.edges],
        }
        Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")

    def _rebuild_index(self) -> None:
        adj: Dict[str, List[str]] = {nid: [] for nid in self.nodes}
        adj_out: Dict[str, List[str]] = {nid: [] for nid in self.nodes}
        self._edge_by_pair = {}
        for e in self.edges:
            if e.src not in self.nodes or e.dst not in self.nodes:
                continue
            adj.setdefault(e.src, []).append(e.dst)
            adj.setdefault(e.dst, []).append(e.src)
            adj_out.setdefault(e.src, []).append(e.dst)
            if not e.directed:
                adj_out.setdefault(e.dst, []).append(e.src)
            self._edge_by_pair[(e.src, e.dst)] = e
        self._adj = adj
        self._adj_out = adj_out

    def edge_between(self, src: str, dst: str) -> Optional[Edge]:
        return self._edge_by_pair.get((src, dst)) or self._edge_by_pair.get((dst, src))

    def directed_edge_between(self, src: str, dst: str) -> Optional[Edge]:
        edge = self._edge_by_pair.get((src, dst))
        if edge is not None:
            return edge
        reverse = self._edge_by_pair.get((dst, src))
        if reverse is not None and not reverse.directed:
            return reverse
        return None

    def out_neighbors(self, node_id: str) -> List[str]:
        return list(self._adj_out.get(node_id, []))

    def local_neighborhood(self, seeds: Sequence[str], *, max_hops: int = 2, max_nodes: int = 32) -> List[str]:
        out: List[str] = []
        seen = set()
        q = deque()
        for s in seeds:
            if s in self.nodes and s not in seen:
                seen.add(s); q.append((s, 0)); out.append(s)
        while q and len(out) < max_nodes:
            u, d = q.popleft()
            if d >= max_hops:
                continue
            for v in sorted(self._adj.get(u, []), key=lambda x: -self.nodes[x].importance):
                if v in seen:
                    continue
                seen.add(v); out.append(v); q.append((v, d + 1))
                if len(out) >= max_nodes:
                    break
        return out

    def iter_local_edges(self, node_ids: Sequence[str]) -> List[Edge]:
        s = set(node_ids)
        return [e for e in self.edges if e.src in s and e.dst in s]


def resolve_graph_path(graph_id: str, graph_dir: str | Path) -> Path:
    graph_id = str(graph_id)
    base = Path(graph_dir)
    candidates = [base / graph_id]
    if not graph_id.endswith(".json"):
        candidates.append(base / f"{graph_id}.json")
    # also tolerate uploaded file suffixes: cs4(9).json -> cs4.json in rows
    stem = Path(graph_id).stem
    candidates.extend(sorted(base.glob(f"{stem}*.json")))
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError(f"Cannot resolve graph_id={graph_id!r} under {graph_dir}")


# -----------------------------
# Text/features
# -----------------------------

def build_semantic_node_text(node: Node) -> str:
    fam = node_type_family(node.node_type)
    status = str(node.metadata.get("status", node.metadata.get("polarity", "")))
    cluster = str(node.metadata.get("cluster", ""))
    return (
        f"Node id: {node.id}. Type: {node.node_type}. Family: {fam}. "
        f"Status: {status}. Cluster: {cluster}. Text: {node.text}"
    )


def build_semantic_relation_text(relation: str) -> str:
    rel = canonical_relation(relation)
    fam = relation_family(rel)
    return f"Relation: {rel}. Family: {fam}. This edge describes how two graph nodes are connected."


def node_role_features(node: Node) -> List[float]:
    fam = node_type_family(node.node_type)
    meta = node.metadata or {}
    status = str(meta.get("status", "")).lower()
    polarity = str(meta.get("polarity", "")).lower()
    kind = str(meta.get("kind", "")).lower()
    text = node.text.lower()
    return [
        1.0,  # bias
        float(node.confidence),
        float(node.importance),
        1.0 if fam == "summary" else 0.0,
        1.0 if fam == "concept" else 0.0,
        1.0 if fam == "fact" else 0.0,
        1.0 if fam == "claim" else 0.0,
        1.0 if fam == "example" else 0.0,
        1.0 if status == "accepted" else 0.0,
        1.0 if polarity == "false" or "false" in kind or node.confidence < 0.15 else 0.0,
        1.0 if "hypothesis" in kind or canonical_node_type(node.node_type) == "hypothesis" or status == "uncertain" else 0.0,
        1.0 if "bridge" in kind or canonical_node_type(node.node_type) == "bridge" else 0.0,
        1.0 if any(w in text for w in ("not ", "false", "wrong", "cannot", "contradict")) else 0.0,
        min(len(node.text) / 220.0, 1.0),
    ]


def edge_features(edge: Edge) -> List[float]:
    rel = canonical_relation(edge.relation)
    fam = relation_family(rel)
    return [
        1.0,
        float(edge.strength),
        1.0 if edge.directed else 0.0,
        relation_polarity(rel),
        1.0 if fam == "support" else 0.0,
        1.0 if fam == "contradict" else 0.0,
        1.0 if fam == "causal" else 0.0,
        1.0 if fam == "part_of" else 0.0,
    ]


def lexical_tokens(text: str) -> set[str]:
    return _shared_lexical_tokens(text, min_chars=3)


def lexical_overlap(a: str, b: str) -> float:
    return _shared_lexical_overlap(a, b, min_chars=3)
