from __future__ import annotations

import argparse
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Set, Tuple

from graph_core import Edge, MemoryGraph, Node, canonical_relation
from planner import choose_attachment_edges, flatten_graph_packet, retrieve_graph_packet


STOPWORDS: Set[str] = {
    "a", "an", "and", "are", "as", "at", "be", "by", "can", "for", "from", "has",
    "but", "have", "in", "into", "is", "it", "its", "never", "of", "on", "or", "over", "that",
    "the", "their", "this", "to", "using", "with", "without",
}


@dataclass
class QAProbe:
    question: str
    expected_answer: str
    must_cover_terms: List[str]
    source: str = "deterministic_signal_probe"


def normalize_text(text: Any) -> str:
    return " ".join(str(text or "").strip().split())


def content_tokens(text: str) -> List[str]:
    toks = re.findall(r"[a-z0-9]+", str(text or "").lower())
    return [t for t in toks if len(t) > 2 and t not in STOPWORDS]


def token_set(text: str) -> Set[str]:
    return set(content_tokens(text))


def lexical_recall(expected: str, observed: str) -> float:
    exp = token_set(expected)
    if not exp:
        return 0.0
    obs = token_set(observed)
    return len(exp & obs) / len(exp)


def generate_qa_probes(signal_text: str, *, max_terms: int = 10) -> List[QAProbe]:
    signal = normalize_text(signal_text)
    terms = content_tokens(signal)
    seen: Set[str] = set()
    must_terms: List[str] = []
    for term in terms:
        if term in seen:
            continue
        seen.add(term)
        must_terms.append(term)
        if len(must_terms) >= max_terms:
            break
    topic = " ".join(must_terms[:4]) if must_terms else "the signal"
    question = f"What should the graph know about {topic}?"
    return [QAProbe(question=question, expected_answer=signal, must_cover_terms=must_terms)]


def score_probe(
    graph: MemoryGraph,
    probe: QAProbe,
    *,
    top_k: int = 8,
    hops: int = 1,
) -> Dict[str, Any]:
    # This is an offline evaluator: the expected answer is known because the
    # critic generated the probe from the signal. Include it in retrieval so
    # the score measures graph answerability, not only question phrasing.
    retrieval_query = f"{probe.question} {probe.expected_answer}"
    packet = retrieve_graph_packet(graph, retrieval_query, top_k=top_k, hops=hops, mode="general")
    retrieved = []
    seen_ids: Set[str] = set()
    # Preserve lexical anchors first. For QA evaluation, a high-recall answer
    # node should not be displaced by structurally boosted hubs.
    for section in ("anchors", "suggested_attach", "neighbors", "parents", "same_cluster", "children"):
        for item in packet.get(section, []) or []:
            nid = str(item.get("id", ""))
            if not nid or nid in seen_ids:
                continue
            seen_ids.add(nid)
            retrieved.append(item)
            if len(retrieved) >= top_k:
                break
        if len(retrieved) >= top_k:
            break
    for item in flatten_graph_packet(packet, top_k=top_k):
        nid = str(item.get("id", ""))
        if nid and nid not in seen_ids:
            seen_ids.add(nid)
            retrieved.append(item)
            if len(retrieved) >= top_k:
                break
    texts: List[str] = []
    node_scores: List[Tuple[str, float]] = []
    for item in retrieved:
        nid = str(item.get("id", ""))
        node = graph.nodes.get(nid)
        if node is None:
            continue
        text = normalize_text(node.text)
        texts.append(text)
        node_scores.append((nid, lexical_recall(probe.expected_answer, text)))

    combined = "\n".join(texts)
    answer_recall = lexical_recall(probe.expected_answer, combined)
    must = set(probe.must_cover_terms)
    observed = token_set(combined)
    must_cover_recall = (len(must & observed) / len(must)) if must else 0.0
    best_single_node_recall = max((score for _, score in node_scores), default=0.0)

    useful_edge_bonus = 0.0
    retrieved_ids = [str(item.get("id", "")) for item in retrieved if str(item.get("id", "")) in graph.nodes]
    retrieved_set = set(retrieved_ids)
    for edge in graph.edges:
        if edge.src in retrieved_set and edge.dst in retrieved_set:
            useful_edge_bonus = min(1.0, useful_edge_bonus + 0.10)

    score = (
        0.45 * answer_recall
        + 0.25 * must_cover_recall
        + 0.20 * best_single_node_recall
        + 0.10 * useful_edge_bonus
    )
    missing_terms = sorted(must - observed)
    return {
        "question": probe.question,
        "expected_answer": probe.expected_answer,
        "must_cover_terms": probe.must_cover_terms,
        "score": float(score),
        "answer_recall": float(answer_recall),
        "must_cover_recall": float(must_cover_recall),
        "best_single_node_recall": float(best_single_node_recall),
        "useful_edge_bonus": float(useful_edge_bonus),
        "missing_terms": missing_terms,
        "retrieved_ids": retrieved_ids,
        "top_node_recalls": [{"id": nid, "recall": score} for nid, score in sorted(node_scores, key=lambda x: (-x[1], x[0]))[:5]],
    }


def score_graph_for_signal(graph: MemoryGraph, signal_text: str, *, top_k: int = 8, hops: int = 1) -> Dict[str, Any]:
    probes = generate_qa_probes(signal_text)
    scored = [score_probe(graph, probe, top_k=top_k, hops=hops) for probe in probes]
    avg = sum(float(x["score"]) for x in scored) / len(scored) if scored else 0.0
    return {
        "qa_score": float(avg),
        "probes": [asdict(p) for p in probes],
        "probe_scores": scored,
    }


def clone_graph(graph: MemoryGraph) -> MemoryGraph:
    return MemoryGraph(
        {nid: Node.from_dict(node.to_dict()) for nid, node in graph.nodes.items()},
        [Edge.from_dict(edge.to_dict()) for edge in graph.edges],
        metadata=dict(graph.metadata),
    )


def simulate_add_node(
    graph: MemoryGraph,
    *,
    node_id: str,
    node_text: str,
    max_edges: int = 3,
) -> MemoryGraph:
    after = clone_graph(graph)
    safe_id = re.sub(r"[^a-z0-9_]+", "_", node_id.strip().lower()).strip("_") or "qa_probe_simulated_node"
    if safe_id in after.nodes:
        suffix = 2
        base = safe_id
        while f"{base}_{suffix}" in after.nodes:
            suffix += 1
        safe_id = f"{base}_{suffix}"
    after.nodes[safe_id] = Node(
        id=safe_id,
        text=normalize_text(node_text),
        node_type="concept",
        confidence=0.75,
        importance=0.75,
        metadata={"status": "qa_probe_simulated"},
    )
    packet = retrieve_graph_packet(after, node_text, top_k=8, hops=1, mode="general")
    for edge in choose_attachment_edges(after, node_text, packet, max_edges=max_edges):
        dst = str(edge.get("dst", ""))
        if dst in after.nodes and dst != safe_id:
            after.edges.append(Edge(src=safe_id, dst=dst, relation=canonical_relation(edge.get("relation", "related")), strength=0.7))
    after._rebuild_index()
    return after


def main() -> int:
    ap = argparse.ArgumentParser(description="Score whether a graph can answer QA probes implied by a signal.")
    ap.add_argument("--graph", required=True, help="Path to graph JSON.")
    ap.add_argument("--signal", required=True, help="Signal text to probe.")
    ap.add_argument("--top-k", type=int, default=8)
    ap.add_argument("--hops", type=int, default=1)
    ap.add_argument("--simulate-add-node-id", default="")
    ap.add_argument("--simulate-add-node-text", default="")
    ap.add_argument("--out-json", default="")
    args = ap.parse_args()

    graph = MemoryGraph.load_json(args.graph)
    before = score_graph_for_signal(graph, args.signal, top_k=args.top_k, hops=args.hops)
    result: Dict[str, Any] = {
        "signal": args.signal,
        "before": before,
    }
    if args.simulate_add_node_text:
        node_id = args.simulate_add_node_id or "qa_probe_simulated_node"
        after_graph = simulate_add_node(graph, node_id=node_id, node_text=args.simulate_add_node_text)
        after = score_graph_for_signal(after_graph, args.signal, top_k=args.top_k, hops=args.hops)
        result["after_simulated_add"] = after
        result["qa_delta"] = float(after["qa_score"]) - float(before["qa_score"])

    text = json.dumps(result, ensure_ascii=False, indent=2)
    if args.out_json:
        Path(args.out_json).write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
