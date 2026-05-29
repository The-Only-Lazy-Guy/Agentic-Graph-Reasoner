"""Generate diverse questions from graph hub nodes.

Walks each hub node, reads its neighbors, and generates questions at
three difficulty levels that test different reasoning patterns:
  - recall: "What is X?"
  - comparison: "Compare X and Y"
  - application: "Design/implement using X"

Output: data/question_bank.jsonl (one question per line)

Usage:
    python scripts/generate_questions.py
    python scripts/generate_questions.py --per-hub 8  # more per hub
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from graph_core import MemoryGraph

GRAPH_PATH = "graphs/merged_graph.json"
OUTPUT_PATH = "data/question_bank.jsonl"

RECALL_TEMPLATES = [
    "What is {concept} and why does it matter?",
    "Explain {concept} in one paragraph suitable for a CS student.",
    "Define {concept} and give its key properties.",
]

COMPARISON_TEMPLATES = [
    "Compare {concept_a} and {concept_b}. When would you choose one over the other?",
    "What is the difference between {concept_a} and {concept_b}?",
    "Explain the relationship between {concept_a} and {concept_b}.",
]

APPLICATION_TEMPLATES = [
    "Implement {concept} in C++. Show complete working code.",
    "Design a system that uses {concept} to solve {problem}.",
    "Walk through how {concept} works on a concrete example.",
    "What happens when you apply {concept} to {scenario}?",
]

HARD_TEMPLATES = [
    "Design a system that combines {concept_a} and {concept_b} to handle {constraint}.",
    "What are the correctness pitfalls when implementing {concept}? Identify at least two.",
    "Prove or disprove: {claim}.",
    "Given {scenario}, which approach is optimal and why? Consider {concept_a} vs {concept_b}.",
]

# Domain-specific scenarios for hard questions
SCENARIOS = {
    "cs": ["500K concurrent users", "real-time updates within 100ms", "O(log n) query time"],
    "math": ["a non-trivial edge case", "the general case with constraints", "a counterexample"],
    "physics": ["extreme conditions", "the quantum regime", "macroscopic systems"],
    "chemistry": ["equilibrium shift", "reaction rate change", "phase transition"],
    "biology": ["cell division errors", "gene expression changes", "evolutionary pressure"],
    "software_eng": ["legacy codebase migration", "microservice decomposition", "CI/CD pipeline"],
}


def extract_concepts(graph: MemoryGraph, hub_id: str, max_concepts: int = 10):
    """Get concept names from a hub's neighborhood."""
    concepts = []
    node = graph.nodes.get(hub_id)
    if not node:
        return concepts

    # Direct neighbors
    neighbor_ids = set()
    for e in graph.edges:
        if e.src == hub_id:
            neighbor_ids.add(e.dst)
        elif e.dst == hub_id:
            neighbor_ids.add(e.src)

    for nid in neighbor_ids:
        n = graph.nodes.get(nid)
        if n and n.node_type in ("claim", "fact", "example", "application", "theorem", "definition"):
            # Extract a short concept name from the node text
            text = n.text.strip()
            # Take the first sentence or first 60 chars
            short = text.split(".")[0][:60].strip()
            if len(short) > 10:
                concepts.append({"id": nid, "text": short, "type": n.node_type})
        if len(concepts) >= max_concepts:
            break
    return concepts


def domain_from_hub(hub_id: str) -> str:
    """Infer domain from hub node id."""
    hid = hub_id.lower()
    if any(k in hid for k in ("cs", "algo", "comp", "program", "system", "data", "cpp", "software", "network")):
        return "cs"
    if any(k in hid for k in ("math", "algebra", "analysis", "geometry", "number", "prob", "logic")):
        return "math"
    if any(k in hid for k in ("physics", "thermo", "quantum", "mechanics", "electro")):
        return "physics"
    if any(k in hid for k in ("chem", "reaction", "acid", "kinetic")):
        return "chemistry"
    if any(k in hid for k in ("bio", "cell", "gene", "evolution")):
        return "biology"
    if any(k in hid for k in ("engineer", "test", "security", "architect", "workflow")):
        return "software_eng"
    return "general"


def generate_for_hub(graph: MemoryGraph, hub_id: str, per_hub: int = 6) -> list:
    """Generate questions from one hub's neighborhood."""
    concepts = extract_concepts(graph, hub_id)
    if not concepts:
        return []

    domain = domain_from_hub(hub_id)
    hub_text = graph.nodes[hub_id].text[:80]
    questions = []

    # Recall questions (easy)
    for c in concepts[:3]:
        tmpl = random.choice(RECALL_TEMPLATES)
        questions.append({
            "q": tmpl.format(concept=c["text"]),
            "domain": domain,
            "difficulty": "easy",
            "hub": hub_id,
            "source_nodes": [c["id"]],
        })

    # Comparison questions (medium)
    if len(concepts) >= 2:
        pairs = [(concepts[i], concepts[j])
                 for i in range(len(concepts)) for j in range(i+1, len(concepts))]
        random.shuffle(pairs)
        for a, b in pairs[:2]:
            tmpl = random.choice(COMPARISON_TEMPLATES)
            questions.append({
                "q": tmpl.format(concept_a=a["text"], concept_b=b["text"]),
                "domain": domain,
                "difficulty": "medium",
                "hub": hub_id,
                "source_nodes": [a["id"], b["id"]],
            })

    # Application questions (medium-hard)
    for c in concepts[:2]:
        tmpl = random.choice(APPLICATION_TEMPLATES)
        scenario_pool = SCENARIOS.get(domain, ["a concrete problem"])
        questions.append({
            "q": tmpl.format(concept=c["text"], problem=random.choice(scenario_pool),
                             scenario=random.choice(scenario_pool)),
            "domain": domain,
            "difficulty": "hard",
            "hub": hub_id,
            "source_nodes": [c["id"]],
        })

    # Hard design/proof questions
    if len(concepts) >= 2:
        tmpl = random.choice(HARD_TEMPLATES)
        a, b = random.sample(concepts[:5], 2)
        constraint = random.choice(SCENARIOS.get(domain, ["efficiency"]))
        # Find a claim to use
        claim = next((c["text"] for c in concepts if c["type"] == "claim"), concepts[0]["text"])
        questions.append({
            "q": tmpl.format(concept=a["text"], concept_a=a["text"], concept_b=b["text"],
                             constraint=constraint, claim=claim, scenario=constraint),
            "domain": domain,
            "difficulty": "hard",
            "hub": hub_id,
            "source_nodes": [a["id"], b["id"]],
        })

    random.shuffle(questions)
    return questions[:per_hub]


def main():
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--per-hub", type=int, default=6, help="questions per hub node")
    ap.add_argument("--graph", default=GRAPH_PATH)
    ap.add_argument("--output", default=OUTPUT_PATH)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed)
    graph = MemoryGraph.load_json(args.graph)
    hubs = [(nid, n) for nid, n in graph.nodes.items() if n.node_type == "hub"]
    print(f"Graph: {len(graph.nodes)} nodes, {len(hubs)} hubs")

    all_questions = []
    for hub_id, hub_node in hubs:
        qs = generate_for_hub(graph, hub_id, per_hub=args.per_hub)
        all_questions.extend(qs)
        if qs:
            print(f"  {hub_id}: {len(qs)} questions ({domain_from_hub(hub_id)})")

    # Dedupe by question text
    seen = set()
    deduped = []
    for q in all_questions:
        key = q["q"].lower().strip()
        if key not in seen:
            seen.add(key)
            deduped.append(q)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for q in deduped:
            f.write(json.dumps(q, ensure_ascii=False) + "\n")

    # Stats
    from collections import Counter
    by_domain = Counter(q["domain"] for q in deduped)
    by_diff = Counter(q["difficulty"] for q in deduped)
    print(f"\nGenerated: {len(deduped)} unique questions")
    print(f"By domain: {dict(by_domain)}")
    print(f"By difficulty: {dict(by_diff)}")
    print(f"Output: {out_path}")


if __name__ == "__main__":
    main()
