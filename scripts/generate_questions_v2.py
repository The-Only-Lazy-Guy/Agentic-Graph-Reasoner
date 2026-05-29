"""Generate diverse, reasoning-intensive questions from graph structure.

V2: focuses on question types that force multi-hop graph reasoning,
not just recall. Each question type produces different training signals:

  - debug: model must find contradictions between nodes
  - transfer: model must bridge concepts across domains
  - edge_case: model must find failure patterns and boundary conditions
  - chain: model must follow multi-hop paths
  - adversarial: model must verify or refute claims using graph evidence
  - synthesis: model must combine 2+ nodes into something new
  - teaching: model must restructure graph content for a specific audience

Usage:
    python scripts/generate_questions_v2.py
    python scripts/generate_questions_v2.py --per-hub 10
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path
from typing import List, Dict, Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from graph_core import MemoryGraph

GRAPH_PATH = "graphs/merged_graph.json"
OUTPUT_PATH = "data/question_bank_v2.jsonl"


def get_neighborhood(graph: MemoryGraph, node_id: str, max_depth: int = 2) -> List[Dict[str, Any]]:
    """Get nodes within max_depth hops."""
    visited = {node_id}
    frontier = [node_id]
    result = []
    for depth in range(max_depth):
        next_frontier = []
        for nid in frontier:
            for e in graph.edges:
                other = None
                if e.src == nid:
                    other = e.dst
                elif e.dst == nid:
                    other = e.src
                if other and other not in visited:
                    visited.add(other)
                    next_frontier.append(other)
                    n = graph.nodes.get(other)
                    if n:
                        result.append({
                            "id": other,
                            "text": n.text[:100],
                            "type": n.node_type,
                            "depth": depth + 1,
                        })
        frontier = next_frontier
    return result


def domain_from_hub(hub_id: str) -> str:
    hid = hub_id.lower()
    if any(k in hid for k in ("cs", "algo", "comp", "program", "system", "data", "cpp", "software", "network")):
        return "cs"
    if any(k in hid for k in ("math", "algebra", "analysis", "geometry", "number", "prob", "logic")):
        return "math"
    if any(k in hid for k in ("physics", "thermo", "quantum", "mechanics")):
        return "physics"
    if any(k in hid for k in ("chem", "reaction", "acid", "kinetic")):
        return "chemistry"
    if any(k in hid for k in ("bio", "cell", "gene")):
        return "biology"
    if any(k in hid for k in ("engineer", "test", "security", "architect", "workflow")):
        return "software_eng"
    return "general"


def generate_debug_questions(nodes: list, domain: str, hub_id: str) -> list:
    """'My X doesn't work because Y. What's wrong?' — forces contradiction finding."""
    qs = []
    false_nodes = [n for n in nodes if "_false" in n["id"]]
    for fn in false_nodes[:2]:
        qs.append({
            "q": f"Someone told me: '{fn['text']}' Is this correct? If not, explain what's actually true and why this misconception exists.",
            "domain": domain, "difficulty": "medium", "type": "adversarial",
            "hub": hub_id, "source_nodes": [fn["id"]],
        })
    return qs


def generate_chain_questions(graph: MemoryGraph, nodes: list, domain: str, hub_id: str) -> list:
    """Follow a 2-3 hop path and ask about the connection."""
    qs = []
    # Find pairs that are 2 hops apart
    for n1 in nodes[:5]:
        neighbors = get_neighborhood(graph, n1["id"], max_depth=1)
        for n2 in neighbors[:3]:
            if n1["type"] in ("claim", "fact") and n2["type"] in ("claim", "fact", "application"):
                qs.append({
                    "q": f"How does '{n1['text'][:60]}' connect to '{n2['text'][:60]}'? Trace the reasoning chain through the graph.",
                    "domain": domain, "difficulty": "medium", "type": "chain",
                    "hub": hub_id, "source_nodes": [n1["id"], n2["id"]],
                })
    return qs[:3]


def generate_edge_case_questions(nodes: list, domain: str, hub_id: str) -> list:
    """When does X fail? — forces failure pattern discovery."""
    qs = []
    claims = [n for n in nodes if n["type"] in ("claim", "application")]
    for c in claims[:3]:
        qs.append({
            "q": f"Under what conditions does '{c['text'][:60]}' break down or produce incorrect results? Give specific failure scenarios.",
            "domain": domain, "difficulty": "hard", "type": "edge_case",
            "hub": hub_id, "source_nodes": [c["id"]],
        })
    return qs[:2]


def generate_synthesis_questions(nodes: list, domain: str, hub_id: str) -> list:
    """Combine 2+ concepts into something new."""
    qs = []
    concepts = [n for n in nodes if n["type"] in ("claim", "fact", "application", "example")]
    if len(concepts) >= 2:
        pairs = [(concepts[i], concepts[j]) for i in range(len(concepts)) for j in range(i+1, len(concepts))]
        random.shuffle(pairs)
        for a, b in pairs[:2]:
            qs.append({
                "q": f"Can you combine the idea from '{a['text'][:50]}' with '{b['text'][:50]}' to solve a novel problem? Describe the problem and the combined solution.",
                "domain": domain, "difficulty": "hard", "type": "synthesis",
                "hub": hub_id, "source_nodes": [a["id"], b["id"]],
            })
    return qs


def generate_teaching_questions(nodes: list, domain: str, hub_id: str) -> list:
    """Explain X to audience Y — forces restructuring."""
    audiences = ["a high school student", "a professional in a different field",
                 "someone who only knows the basics", "a visual learner"]
    qs = []
    for n in nodes[:2]:
        if n["type"] in ("claim", "fact", "theorem", "definition"):
            audience = random.choice(audiences)
            qs.append({
                "q": f"Explain '{n['text'][:60]}' to {audience}. Use analogies grounded in the graph's knowledge.",
                "domain": domain, "difficulty": "medium", "type": "teaching",
                "hub": hub_id, "source_nodes": [n["id"]],
            })
    return qs


def generate_transfer_questions(graph: MemoryGraph, hub_id: str, all_hubs: list) -> list:
    """Bridge concepts across domains — forces cross-hub reasoning."""
    qs = []
    hub_domain = domain_from_hub(hub_id)
    hub_node = graph.nodes.get(hub_id)
    if not hub_node:
        return qs
    # Find hubs from different domains
    other_hubs = [h for h in all_hubs if domain_from_hub(h) != hub_domain]
    random.shuffle(other_hubs)
    for other_id in other_hubs[:2]:
        other_node = graph.nodes.get(other_id)
        if not other_node:
            continue
        qs.append({
            "q": f"What principles from '{hub_node.text[:50]}' could transfer to '{other_node.text[:50]}'? Find connections through the graph.",
            "domain": f"{hub_domain}+{domain_from_hub(other_id)}", "difficulty": "hard", "type": "transfer",
            "hub": hub_id, "source_nodes": [hub_id, other_id],
        })
    return qs


def generate_implementation_deep(nodes: list, domain: str, hub_id: str) -> list:
    """Step-by-step implementation with justification at each step."""
    qs = []
    examples = [n for n in nodes if n["type"] in ("example", "application")]
    for ex in examples[:2]:
        qs.append({
            "q": f"Implement '{ex['text'][:50]}' step by step. At each step, justify your choice by citing specific graph evidence.",
            "domain": domain, "difficulty": "hard", "type": "implementation",
            "hub": hub_id, "source_nodes": [ex["id"]],
        })
    return qs


def main():
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--per-hub", type=int, default=8, help="target questions per hub")
    ap.add_argument("--graph", default=GRAPH_PATH)
    ap.add_argument("--output", default=OUTPUT_PATH)
    ap.add_argument("--seed", type=int, default=123)
    args = ap.parse_args()

    random.seed(args.seed)
    graph = MemoryGraph.load_json(args.graph)
    hub_ids = [nid for nid, n in graph.nodes.items() if n.node_type == "hub"]
    print(f"Graph: {len(graph.nodes)} nodes, {len(hub_ids)} hubs")

    all_questions = []
    from collections import Counter
    type_counter = Counter()

    for hub_id in hub_ids:
        domain = domain_from_hub(hub_id)
        neighbors = get_neighborhood(graph, hub_id, max_depth=2)
        if not neighbors:
            continue

        qs = []
        qs.extend(generate_debug_questions(neighbors, domain, hub_id))
        qs.extend(generate_chain_questions(graph, neighbors, domain, hub_id))
        qs.extend(generate_edge_case_questions(neighbors, domain, hub_id))
        qs.extend(generate_synthesis_questions(neighbors, domain, hub_id))
        qs.extend(generate_teaching_questions(neighbors, domain, hub_id))
        qs.extend(generate_transfer_questions(graph, hub_id, hub_ids))
        qs.extend(generate_implementation_deep(neighbors, domain, hub_id))

        random.shuffle(qs)
        qs = qs[:args.per_hub]
        all_questions.extend(qs)

        for q in qs:
            type_counter[q.get("type", "?")] += 1

    # Dedupe
    seen = set()
    deduped = []
    for q in all_questions:
        key = q["q"].lower().strip()[:80]
        if key not in seen:
            seen.add(key)
            deduped.append(q)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for q in deduped:
            f.write(json.dumps(q, ensure_ascii=False) + "\n")

    by_domain = Counter(q["domain"] for q in deduped)
    print(f"\nGenerated: {len(deduped)} unique questions")
    print(f"By type: {dict(type_counter)}")
    print(f"By domain: {dict(by_domain)}")
    print(f"Output: {out_path}")


if __name__ == "__main__":
    main()
