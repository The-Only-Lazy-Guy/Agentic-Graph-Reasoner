"""External-knowledge -> graph-edit extractor (shared facts + CoT modes).

Source B of the V5 graph grower. Turns external documents (Wikipedia/paragraph
text OR chain-of-thought traces) into ATOMIC graph-edit candidates in the same
``raw_edit`` schema the audit/apply path already consumes. So the extractor is
just a new *source*; dedup, judge, validate, health-gate, staging, provenance,
and apply are all reused (see ``apply.py:apply_candidates``).

Pipeline (per document):
  chunk -> LLM extract (mode-aware) -> conform (vocab + atomicity + stable ids)
        -> [optional] entity-resolve vs existing graph (semantic_dedupe.classify)
        -> [optional] LLM judge (edit_judge) -> candidate queue jsonl

The LLM call is injectable (``extract_fn``) so the core (parsing, conformance,
candidate shaping) is testable offline with a stub. The default backend mirrors
data-gen: a V4 opencode controller (big default model, no --model).

Two modes, one extractor:
  * fact : paragraph -> atomic claim/concept nodes + entails/supports/contradicts/related
  * cot  : reasoning steps -> reasoning_atom/chain_step nodes + chain_step/leveraged/transfers_to
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence

from reasoning.graph_relations import RELATION_TYPE_ID

# ── vocabulary ───────────────────────────────────────────────────────────────
ALLOWED_RELATIONS = frozenset(RELATION_TYPE_ID)        # the 12 canonical relations
FALLBACK_RELATION = "related"

# Conservative synonym map for relations LLMs commonly emit; anything else that
# is not in ALLOWED_RELATIONS falls back to "related".
RELATION_ALIASES: Dict[str, str] = {
    "is_a": "related", "instance_of": "related", "part_of": "related",
    "causes": "entails", "implies": "entails", "leads_to": "entails",
    "because": "supports", "evidence_for": "supports", "supported_by": "supports",
    "contradicted_by": "contradicts", "refutes": "contradicts",
    "derived_from": "leveraged", "uses": "leveraged", "applies": "leveraged",
    "next_step": "chain_step", "then": "chain_step", "step": "chain_step",
    "generalizes_to": "transfers_to", "transfers": "transfers_to",
    "requires": "requires_slot", "depends_on": "requires_slot",
}

# Node types MUST match the graph design: only types in v5.gnn_encoder.
# NODE_TYPE_VOCAB get a real type embedding, and only types in a
# v5.subgraph pool (PLANNING_NODE_TYPES / EVIDENCE_NODE_TYPES) are ever attended.
# An off-vocab type (e.g. "concept") becomes "unknown" AND falls in no pool ->
# the node is invisible to both cross-attention layers. So we emit ONLY pooled
# design types and alias common LLM outputs onto them.
#   fact mode -> EVIDENCE pool : fact, claim   (declarative atomic knowledge)
#   cot  mode -> PLANNING pool : reasoning_atom, reasoning_chain, strategy
#                EVIDENCE pool : solved_subgoal (the resolved conclusion)
MODE_DEFAULT_NODE_TYPE = {"fact": "fact", "cot": "reasoning_atom"}
MODE_NODE_TYPES = {
    "fact": frozenset({"fact", "claim"}),
    "cot": frozenset({"reasoning_atom", "reasoning_chain", "strategy", "solved_subgoal"}),
}
ALLOWED_NODE_TYPES = frozenset().union(*MODE_NODE_TYPES.values())

# Map common LLM node-type outputs onto the design vocab (applied before the
# per-mode allow check; anything still off-vocab falls back to the mode default).
NODE_TYPE_ALIASES: Dict[str, str] = {
    "concept": "claim", "definition": "claim", "principle": "claim",
    "theorem": "fact", "law": "fact", "equation": "fact", "rule": "fact",
    "chain_step": "reasoning_chain", "step": "reasoning_chain",
    "reasoning_step": "reasoning_chain", "inference": "reasoning_atom",
    "subgoal": "solved_subgoal", "conclusion": "solved_subgoal", "plan": "strategy",
}

# atomicity bounds: a node should be one self-contained claim, not a blob
MIN_TEXT_CHARS = 12
MAX_TEXT_CHARS = 400


@dataclass
class Document:
    id: str
    text: str
    domain: str = ""
    mode: str = "fact"


# ── source + chunking ────────────────────────────────────────────────────────
def load_documents(path: str | Path) -> List[Document]:
    """Load documents from a jsonl: {id, text, domain?, mode?}."""
    docs: List[Document] = []
    for i, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines()):
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        docs.append(Document(
            id=str(row.get("id") or f"doc_{i}"),
            text=str(row.get("text") or ""),
            domain=str(row.get("domain") or ""),
            mode=str(row.get("mode") or "fact"),
        ))
    return docs


def chunk_document(doc: Document, *, max_chunk_chars: int = 1200) -> List[str]:
    """Split a document into extraction units.

    fact mode -> paragraphs (blank-line separated, length-capped).
    cot  mode -> reasoning steps (numbered/marker separated, else paragraphs).
    """
    text = doc.text.strip()
    if not text:
        return []
    if doc.mode == "cot":
        parts = re.split(r"\n\s*(?:step\s*\d+[:.)]|\d+[.)]|-)\s+", text, flags=re.IGNORECASE)
        parts = [p.strip() for p in parts if p.strip()]
        if len(parts) > 1:
            return parts
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    chunks: List[str] = []
    for p in (paragraphs or [text]):
        if len(p) <= max_chunk_chars:
            chunks.append(p)
        else:
            for s in range(0, len(p), max_chunk_chars):
                chunks.append(p[s:s + max_chunk_chars])
    return chunks


# ── extraction prompt + parsing ──────────────────────────────────────────────
def _system_prompt(mode: str) -> str:
    node_hint = (
        "node_type one of: fact, claim (declarative, decomposed atomic knowledge)"
        if mode == "fact" else
        "node_type one of: reasoning_atom, reasoning_chain, strategy, solved_subgoal"
    )
    rel_hint = (
        "relation one of: entails, supports, contradicts, related"
        if mode == "fact" else
        "relation one of: chain_step, leveraged, transfers_to, supports"
    )
    return (
        "You convert text into ATOMIC knowledge-graph edits. Decompose into the "
        "SMALLEST self-contained claims: one idea per node, each independently "
        "true/checkable. Do NOT copy whole sentences with multiple facts.\n"
        f"{node_hint}.\n{rel_hint}; edges connect node ids you emit.\n"
        "Output ONLY a JSON object:\n"
        '{"nodes":[{"id":"short_slug","node_type":"...","text":"one atomic claim"}],'
        '"edges":[{"src":"slug","dst":"slug","relation":"..."}]}'
    )


def _llm_extract(chunk: str, mode: str, controller: Any) -> str:
    resp = controller.chat_oneshot([
        {"role": "system", "content": _system_prompt(mode)},
        {"role": "user", "content": chunk},
    ])
    return resp["choices"][0]["message"]["content"]


_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


def parse_extraction(raw: str) -> Dict[str, Any]:
    """Parse the model's JSON (tolerant of surrounding prose / code fences)."""
    if not raw:
        return {"nodes": [], "edges": []}
    text = raw.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\n?|\n?```$", "", text).strip()
    m = _JSON_RE.search(text)
    if not m:
        return {"nodes": [], "edges": []}
    try:
        obj = json.loads(m.group(0))
    except json.JSONDecodeError:
        return {"nodes": [], "edges": []}
    return {
        "nodes": obj.get("nodes") or [],
        "edges": obj.get("edges") or [],
    }


# ── conformance: vocab + atomicity + stable ids ──────────────────────────────
def _slug(text: str, mode: str) -> str:
    h = hashlib.sha1(text.strip().lower().encode("utf-8")).hexdigest()[:12]
    return f"kb_{mode}_{h}"


def conform_edits(parsed: Mapping[str, Any], doc: Document) -> Dict[str, Any]:
    """Map raw LLM nodes/edges to validated, atomic, stable-id raw_edits.

    Returns {"node_edits": [...], "edge_edits": [...], "id_map": {llm_id: stable_id},
    "dropped": {...}}.
    """
    mode = doc.mode if doc.mode in MODE_DEFAULT_NODE_TYPE else "fact"
    allowed_types = MODE_NODE_TYPES.get(mode, ALLOWED_NODE_TYPES)
    default_type = MODE_DEFAULT_NODE_TYPE[mode]

    id_map: Dict[str, str] = {}
    node_edits: List[Dict[str, Any]] = []
    dropped = {"non_atomic": 0, "empty": 0, "bad_edge": 0}

    for n in parsed.get("nodes", []) or []:
        if not isinstance(n, Mapping):
            continue
        text = str(n.get("text") or "").strip()
        if len(text) < MIN_TEXT_CHARS:
            dropped["empty"] += 1
            continue
        if len(text) > MAX_TEXT_CHARS:
            dropped["non_atomic"] += 1
            continue
        ntype = str(n.get("node_type") or "").strip().lower()
        ntype = NODE_TYPE_ALIASES.get(ntype, ntype)
        if ntype not in allowed_types:
            ntype = default_type
        stable = _slug(text, mode)
        llm_id = str(n.get("id") or stable)
        id_map[llm_id] = stable
        id_map.setdefault(stable, stable)
        node_edits.append({
            "op": "add_node",
            "node_id": stable,
            "node_type": ntype,
            "text": text,
            "tier": "add",
            "metadata": {"kb_source_doc": doc.id, "kb_domain": doc.domain, "kb_mode": mode},
        })

    edge_edits: List[Dict[str, Any]] = []
    for e in parsed.get("edges", []) or []:
        if not isinstance(e, Mapping):
            continue
        src = id_map.get(str(e.get("src") or ""))
        dst = id_map.get(str(e.get("dst") or ""))
        if not src or not dst or src == dst:
            dropped["bad_edge"] += 1
            continue
        rel = str(e.get("relation") or "").strip().lower()
        rel = rel if rel in ALLOWED_RELATIONS else RELATION_ALIASES.get(rel, FALLBACK_RELATION)
        edge_edits.append({
            "op": "add_edge",
            "src": src,
            "dst": dst,
            "relation": rel,
            "tier": "add",
            "metadata": {"kb_source_doc": doc.id, "kb_mode": mode},
        })
    return {"node_edits": node_edits, "edge_edits": edge_edits, "id_map": id_map, "dropped": dropped}


def _to_candidate(raw_edit: Mapping[str, Any], doc: Document, idx: int) -> Dict[str, Any]:
    """Shape a raw_edit into the candidate dict apply.py consumes."""
    op = raw_edit.get("op")
    target = raw_edit.get("node_id") if op == "add_node" else raw_edit.get("dst")
    return {
        "lane": "external",
        "session_id": doc.id,
        "task_family": doc.domain or "external_kb",
        "patch_id": f"{doc.id}_{op}_{idx:04d}",
        "patch_type": "add_fact" if op == "add_node" else "add_relation",
        "target_id": target,
        "status": "accept",
        "risk_level": "medium",
        "support_score": None,
        "text": raw_edit.get("text", ""),
        "raw_edit": dict(raw_edit),
        "reasons": ["external_kb_extraction"],
        "warnings": [],
    }


# ── orchestration ────────────────────────────────────────────────────────────
def _attach_edge(new_id: str, existing_id: str, similarity: float, doc: Document) -> Dict[str, Any]:
    """Edge from a new node to a near-but-not-duplicate existing graph node, so
    the new knowledge joins the topology instead of forming an island."""
    return {
        "op": "add_edge", "src": new_id, "dst": existing_id,
        "relation": "related", "tier": "add",
        "metadata": {"kb_source_doc": doc.id, "kb_link": "ambiguous",
                     "kb_link_sim": round(float(similarity), 4)},
    }


def extract_documents(
    docs: Sequence[Document],
    *,
    controller: Any = None,
    extract_fn: Optional[Callable[[str, str], str]] = None,
    dedupe_index: Any = None,
    dup_threshold: float = 0.92,
    attach_threshold: float = 0.80,
    judge_fn: Optional[Callable[[Mapping[str, Any]], Any]] = None,
    collect_sft: bool = False,
) -> Dict[str, Any]:
    """Run the extractor over documents -> candidates + stats.

    extract_fn(chunk, mode) -> raw model string. If None, uses the LLM controller.

    dedupe_index (optional, semantic_dedupe.DedupeIndex over the TARGET graph):
      - cosine >= dup_threshold  -> "duplicate": drop the new node, remap its
        edges onto the existing node.
      - attach_threshold..dup    -> "ambiguous": KEEP the new node AND add an
        attach edge (related) to the nearest existing node, so it joins the
        graph topology rather than islanding.
    judge_fn (optional) -> object/tuple with (decision, merge_target):
      reject -> drop; merge_into -> drop + remap edges onto merge_target; accept -> keep.
    """
    if extract_fn is None:
        if controller is None:
            raise ValueError("provide either extract_fn or controller")
        extract_fn = lambda chunk, mode: _llm_extract(chunk, mode, controller)

    candidates: List[Dict[str, Any]] = []
    sft_pairs: List[Dict[str, Any]] = []
    stats = {"docs": len(docs), "chunks": 0, "nodes": 0, "edges": 0,
             "dropped_non_atomic": 0, "dropped_empty": 0, "dropped_bad_edge": 0,
             "deduped_existing": 0, "attached_existing": 0, "merged_existing": 0,
             "judged_out": 0, "sft_pairs": 0}
    # map a dropped/duplicate/merged stable id -> existing graph node id, to remap edges
    remap: Dict[str, str] = {}

    for doc in docs:
        for chunk in chunk_document(doc):
            stats["chunks"] += 1
            parsed = parse_extraction(extract_fn(chunk, doc.mode))
            conformed = conform_edits(parsed, doc)
            stats["dropped_non_atomic"] += conformed["dropped"]["non_atomic"]
            stats["dropped_empty"] += conformed["dropped"]["empty"]
            stats["dropped_bad_edge"] += conformed["dropped"]["bad_edge"]

            # Distillation: capture (chunk -> conformed extraction) as an SFT pair
            # for training a local Qwen-0.5B extractor to replace the big teacher.
            if collect_sft:
                pair = _sft_pair(chunk, doc, conformed)
                if pair is not None:
                    sft_pairs.append(pair)
                    stats["sft_pairs"] += 1

            kept_nodes: List[Dict[str, Any]] = []
            attach_edges: List[Dict[str, Any]] = []
            for ne in conformed["node_edits"]:
                nid = ne["node_id"]
                ambiguous_match = None
                # entity resolution vs existing graph
                if dedupe_index is not None:
                    kind, match = dedupe_index.classify(
                        ne["text"], dup_threshold=dup_threshold, ambiguous_threshold=attach_threshold)
                    if kind == "duplicate" and match is not None:
                        remap[nid] = match.node_id
                        stats["deduped_existing"] += 1
                        continue
                    if kind == "ambiguous" and match is not None:
                        ambiguous_match = match
                # optional LLM judge gate
                if judge_fn is not None:
                    decision, target = _judge_decision(judge_fn(ne))
                    if decision == "reject":
                        stats["judged_out"] += 1
                        continue
                    if decision == "merge_into" and target:
                        remap[nid] = target
                        stats["merged_existing"] += 1
                        continue
                kept_nodes.append(ne)
                if ambiguous_match is not None:
                    attach_edges.append(_attach_edge(nid, ambiguous_match.node_id,
                                                      ambiguous_match.similarity, doc))
                    stats["attached_existing"] += 1

            for idx, ne in enumerate(kept_nodes):
                candidates.append(_to_candidate(ne, doc, idx))
                stats["nodes"] += 1
            for idx, ee in enumerate(list(conformed["edge_edits"]) + attach_edges):
                ee = dict(ee)
                ee["src"] = remap.get(ee["src"], ee["src"])
                ee["dst"] = remap.get(ee["dst"], ee["dst"])
                candidates.append(_to_candidate(ee, doc, idx + len(kept_nodes)))
                stats["edges"] += 1

    return {"candidates": candidates, "stats": stats, "sft_pairs": sft_pairs}


def _sft_pair(chunk: str, doc: Document, conformed: Mapping[str, Any]) -> Optional[Dict[str, Any]]:
    """Build one (chunk -> conformed extraction) chat SFT pair for distillation.

    The target is the CLEAN conformed extraction (atomic, vocab-correct), with
    local node ids (n0, n1, ...) so the student learns structure, not hashes.
    Returns None for empty extractions (nothing to learn)."""
    node_edits = conformed.get("node_edits") or []
    if not node_edits:
        return None
    stable_to_local = {ne["node_id"]: f"n{i}" for i, ne in enumerate(node_edits)}
    nodes = [{"id": stable_to_local[ne["node_id"]], "node_type": ne["node_type"], "text": ne["text"]}
             for ne in node_edits]
    edges = []
    for ee in conformed.get("edge_edits") or []:
        src, dst = stable_to_local.get(ee["src"]), stable_to_local.get(ee["dst"])
        if src and dst:
            edges.append({"src": src, "dst": dst, "relation": ee["relation"]})
    target = json.dumps({"nodes": nodes, "edges": edges}, ensure_ascii=False)
    return {
        "messages": [
            {"role": "system", "content": _system_prompt(doc.mode)},
            {"role": "user", "content": chunk},
            {"role": "assistant", "content": target},
        ],
        "meta": {"doc_id": doc.id, "mode": doc.mode, "teacher": "opencode",
                 "n_nodes": len(nodes), "n_edges": len(edges)},
    }


def _judge_decision(verdict: Any) -> tuple:
    """Normalize a judge return (JudgeVerdict | tuple | str) -> (decision, target)."""
    if isinstance(verdict, tuple):
        return (verdict[0], verdict[1] if len(verdict) > 1 else None)
    if isinstance(verdict, str):
        return (verdict, None)
    return (getattr(verdict, "decision", "accept"), getattr(verdict, "merge_target", None))


def write_candidate_queue(candidates: Sequence[Mapping[str, Any]], path: str | Path) -> str:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as h:
        for c in candidates:
            h.write(json.dumps(c, ensure_ascii=False) + "\n")
    return str(p)


def _build_opencode_controller(model: Optional[str], config_dir: str):
    from answerer_v4 import V4OpencodeController
    return V4OpencodeController(config_dir=config_dir, model=model, server_url=None)


def _build_dedupe_index(graph_path: str):
    """Build a DedupeIndex over the target graph (loads embeddings)."""
    from graph_core import MemoryGraph
    from reasoning.semantic_dedupe import build_dedupe_index
    graph = MemoryGraph.load_json(graph_path)
    basename = Path(graph_path).stem
    return graph, build_dedupe_index(graph, graph_basename=basename)


def _make_judge_fn(graph, dedupe_index, controller):
    """Wrap edit_judge.judge_edit -> (decision, merge_target)."""
    from reasoning.edit_judge import judge_edit
    def fn(node_edit: Mapping[str, Any]):
        v = judge_edit(dict(node_edit), graph, dedupe_index, controller)
        return v.decision, v.merge_target
    return fn


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Extract external knowledge (facts/CoT) into graph-edit candidates.")
    parser.add_argument("--docs", required=True, help="input jsonl: {id, text, domain?, mode?}")
    parser.add_argument("--out", default="artifacts/graph_growth/external_candidates.jsonl")
    parser.add_argument("--backend", choices=["opencode"], default="opencode")
    parser.add_argument("--opencode-config-dir", default="pure-opencode")
    parser.add_argument("--opencode-model", default=None)
    parser.add_argument("--limit", type=int, default=0, help="cap number of documents (0 = all)")
    parser.add_argument("--graph", default="graphs/merged_graph.json",
                        help="target graph for linking (the graph apply will grow)")
    parser.add_argument("--link-graph", action="store_true",
                        help="dedup new nodes against --graph (drop exact dups, attach near-dups)")
    parser.add_argument("--dup-threshold", type=float, default=0.92)
    parser.add_argument("--attach-threshold", type=float, default=0.80)
    parser.add_argument("--judge", action="store_true",
                        help="LLM-judge each new node (accept/reject/merge_into); implies --link-graph")
    parser.add_argument("--collect-sft", action="store_true",
                        help="append (chunk -> conformed extraction) chat pairs for Qwen-0.5B distillation")
    parser.add_argument("--sft-out", default="data/external_kb/extractor_sft.jsonl",
                        help="SFT corpus path (appended across runs)")
    args = parser.parse_args(argv)

    docs = load_documents(args.docs)
    if args.limit:
        docs = docs[: args.limit]
    controller = _build_opencode_controller(args.opencode_model, args.opencode_config_dir)

    dedupe_index = None
    judge_fn = None
    if args.link_graph or args.judge:
        graph, dedupe_index = _build_dedupe_index(args.graph)
        if args.judge:
            judge_fn = _make_judge_fn(graph, dedupe_index, controller)

    result = extract_documents(
        docs, controller=controller, dedupe_index=dedupe_index,
        dup_threshold=args.dup_threshold, attach_threshold=args.attach_threshold,
        judge_fn=judge_fn, collect_sft=args.collect_sft,
    )
    out = write_candidate_queue(result["candidates"], args.out)

    sft_out = None
    if args.collect_sft and result["sft_pairs"]:
        sft_out = Path(args.sft_out)
        sft_out.parent.mkdir(parents=True, exist_ok=True)
        with sft_out.open("a", encoding="utf-8") as h:   # append across runs
            for pair in result["sft_pairs"]:
                h.write(json.dumps(pair, ensure_ascii=False) + "\n")

    s = result["stats"]
    print("External Knowledge Extraction")
    print(f"  docs: {s['docs']}  chunks: {s['chunks']}")
    print(f"  candidates: {len(result['candidates'])}  (nodes {s['nodes']} / edges {s['edges']})")
    print(f"  dropped: non_atomic {s['dropped_non_atomic']}, empty {s['dropped_empty']}, "
          f"bad_edge {s['dropped_bad_edge']}")
    print(f"  linking: deduped {s['deduped_existing']}, attached {s['attached_existing']}, "
          f"merged {s['merged_existing']}, judged_out {s['judged_out']}"
          + ("" if (dedupe_index is not None) else "  (linking OFF - pass --link-graph)"))
    if args.collect_sft:
        print(f"  sft pairs: {s['sft_pairs']}" + (f" -> appended {sft_out}" if sft_out else " (none)"))
    print(f"  queue: {out}")
    print(f"  next: python -m v5.graph_grower.apply --candidates {out} --graph {args.graph} --out graphs/grown_graph.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
