from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

from consistency_regret_loop import (
    add_node_attachment_targets,
    apply_plan,
    attachment_specific_overlap,
    attachment_specific_terms,
    attachment_topic_flags,
    clone_graph,
    commit_spec,
    is_summary_node,
    retrieve_context,
    row_to_action_spec,
)
from critic import deterministic_graph_checks
from graph_core import MemoryGraph, lexical_overlap
from qa_probe_reward import content_tokens, score_graph_for_signal


GENERIC_EVIDENCE_TERMS: Set[str] = {
    "algorithm", "application", "case", "change", "complex", "concept", "correct",
    "data", "deep", "different", "example", "function", "input", "layer", "learn",
    "method", "model", "network", "node", "output", "pattern", "process", "result",
    "structure", "system", "task", "value", "using", "used", "related", "average",
    "descent", "gradient",
}

STRONG_SINGLETON_EVIDENCE: Set[str] = {
    "activation", "bayes", "bloom", "bridge", "backpropagation", "derivative",
    "dijkstra", "entropy", "fenwick", "hashing", "matrix", "prefix", "regression",
    "search", "signal", "summary", "weighted", "neural", "nonlinearity", "impulse",
    "momentum", "diffusion", "combustion", "atp", "sql", "btree",
}

RELATION_QUESTION_TEMPLATES: Dict[str, str] = {
    "related": "How is {source} related to {target}?",
    "part_of": "How is {source} part of {target}?",
    "support": "How does {source} support {target}?",
    "depend": "How does {source} depend on {target}?",
    "example_of": "How is {source} an example of {target}?",
    "refine": "How does {source} refine {target}?",
    "contradict": "Why does {source} contradict {target}?",
}

RELATION_EXPECTED_ANCHORS: Dict[str, Set[str]] = {
    "related": {"related", "connection", "shared", "similar", "topic"},
    "part_of": {"part", "component", "member", "section", "cluster", "summary", "layer", "module", "step", "stage"},
    "support": {"support", "enable", "allow", "cause", "explain", "help", "introduce", "reduce", "model", "track"},
    "depend": {"depend", "require", "prerequisite", "need", "based on", "relies on", "requires"},
    "example_of": {"example", "instance", "application", "case", "implement", "implementation"},
    "refine": {"refine", "specialize", "narrow", "constrain", "precision", "improve"},
    "contradict": {"contradict", "false", "wrong", "incorrect", "refute", "counterexample", "not", "cannot"},
}

RELATION_CUE_TERMS: Dict[str, Set[str]] = {
    "part_of": {"part", "component", "section", "layer", "module", "stage", "step", "summary", "hub", "contains"},
    "support": {"support", "enable", "allow", "cause", "explain", "help", "introduce", "reduce", "model", "track", "result"},
    "depend": {"depend", "depends", "require", "requires", "need", "needs", "prerequisite", "based on"},
    "example_of": {"example", "examples", "application", "applications", "instance", "instances", "case", "cases", "implement", "implementation"},
    "refine": {"refine", "refines", "specialize", "specializes", "narrow", "narrows", "constrain", "constrains", "improve", "improves"},
    "contradict": {"contradict", "contradicts", "contradiction", "false", "wrong", "incorrect", "refute", "refutes", "counterexample", "cannot", "not"},
}

PART_OF_CONTAINER_TERMS: Set[str] = {
    "part", "component", "section", "layer", "module", "stage", "step", "summary", "hub", "contains", "container",
    "within", "inside", "belongs", "group", "cluster", "parent", "ancestor",
}

FLAG_EVIDENCE_TERMS: Dict[str, str] = {
    "ml_ai": "machine learning",
    "math_analysis": "mathematical computation",
    "range_query": "range query",
    "graph_algo": "graph algorithm",
    "debug": "debugging",
    "database": "database",
    "hci": "interface design",
    "biology": "biology",
    "physics": "physics",
    "chemistry": "chemistry",
    "networking": "networking",
}


def read_jsonl(path: str | Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: str | Path, rows: Iterable[Mapping[str, Any]]) -> None:
    with Path(path).open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def commit_key(row: Mapping[str, Any]) -> Tuple[Any, ...]:
    return (
        row.get("raw_text_id"),
        row.get("sub_signal_index"),
        row.get("chunk_index"),
        row.get("chunk_sub_signal_index"),
    )


def scored_commit_key(row: Mapping[str, Any]) -> Tuple[Any, ...]:
    return (
        row.get("raw_text_id"),
        row.get("sub_signal_index"),
        row.get("chunk_index"),
        row.get("chunk_sub_signal_index"),
    )


def safe_node_id(text: str, *, fallback: str = "repaired_node") -> str:
    toks = re.findall(r"[a-z0-9]+", str(text or "").lower())[:7]
    return "_".join(toks) if toks else fallback


def concept_label(text: str, *, max_terms: int = 4) -> str:
    terms = [t for t in content_tokens(text) if t not in GENERIC_EVIDENCE_TERMS]
    if not terms:
        terms = content_tokens(text)
    return " ".join(terms[:max_terms]) if terms else "concept"


def relation_specific_evidence_terms(source_text: str, target_text: str, relation: str) -> List[str]:
    rel = normalize_relation(relation)
    source_terms = content_tokens(source_text)
    target_terms = content_tokens(target_text)
    shared = clean_evidence_terms(attachment_specific_terms(source_text) & attachment_specific_terms(target_text))

    terms: List[str] = []
    relation_cues = RELATION_CUE_TERMS.get(rel, set())

    def first_cue(words: Sequence[str]) -> str:
        for word in words:
            if word in relation_cues:
                return word
        return ""

    source_cue = first_cue(source_terms)
    target_cue = first_cue(target_terms)
    if source_cue:
        terms.append(source_cue)
    if target_cue and target_cue != source_cue:
        terms.append(target_cue)

    if rel == "contradict":
        contradiction_cues = [t for t in source_terms + target_terms if t in RELATION_CUE_TERMS["contradict"]]
        for cue in contradiction_cues:
            if cue not in terms:
                terms.append(cue)
        if not terms:
            for token in source_terms + target_terms:
                if token in {"not", "false", "wrong", "incorrect", "cannot", "counterexample", "refute"}:
                    terms.append(token)
                    break

    if not terms:
        terms.extend(shared)

    if not terms and rel in {"support", "depend", "part_of", "example_of", "refine"}:
        for token in source_terms + target_terms:
            if token not in GENERIC_EVIDENCE_TERMS and len(token) >= 6:
                terms.append(token)
                break

    return clean_evidence_terms(terms + shared)


def clean_evidence_terms(terms: Iterable[str]) -> List[str]:
    out: List[str] = []
    seen: Set[str] = set()
    for term in terms:
        t = str(term or "").strip().lower()
        if not t or t in seen or t in GENERIC_EVIDENCE_TERMS:
            continue
        if len(t) <= 2:
            continue
        seen.add(t)
        out.append(t)
        if len(out) >= 5:
            break
    return out


def evidence_terms_are_strong(
    terms: Sequence[str],
    *,
    source_text: str,
    target_text: str,
) -> bool:
    filtered = clean_evidence_terms(terms)
    if len(filtered) >= 2:
        return True
    if len(filtered) != 1:
        return False
    term = filtered[0]
    if term not in STRONG_SINGLETON_EVIDENCE:
        return False
    source_l = str(source_text or "").lower()
    target_l = str(target_text or "").lower()
    return term in source_l and term in target_l


def normalize_relation(relation: str) -> str:
    rel = str(relation or "related").strip().lower()
    if rel in {"contradict", "contradicts", "contradiction", "conflict", "conflicts", "conflict_with", "refute", "refutes"}:
        return "contradict"
    if rel in {"supports", "supporting", "enables", "enable", "causes", "cause", "allows", "allow"}:
        return "support"
    if rel in {"depends", "depends_on", "depends-on", "requires", "require", "prerequisite"}:
        return "depend"
    if rel in {"part", "member_of", "member-of", "component_of", "component-of", "contains"}:
        return "part_of"
    if rel in {"example", "instance_of", "instance-of", "application_of", "application-of"}:
        return "example_of"
    if rel in {"refines", "specializes", "specialise", "specializes"}:
        return "refine"
    return rel or "related"


def relation_question(relation: str, source_label: str, target_label: str) -> str:
    rel = normalize_relation(relation)
    template = RELATION_QUESTION_TEMPLATES.get(rel, RELATION_QUESTION_TEMPLATES["related"])
    return template.format(source=source_label, target=target_label)


def relation_anchor_terms(relation: str) -> Set[str]:
    return RELATION_EXPECTED_ANCHORS.get(normalize_relation(relation), RELATION_EXPECTED_ANCHORS["related"])


def is_true_container_target(graph: MemoryGraph, target_id: str) -> bool:
    node = graph.nodes.get(target_id)
    if node is None:
        return False
    node_type = str(getattr(node, "node_type", "")).lower()
    if node_type not in {"summary", "hub", "overview"}:
        return False
    text = f"{target_id.replace('_', ' ')} {getattr(node, 'text', '')}".lower()
    return any(token in text for token in (
        "hub", "summary", "overview", "category", "field",
        "topic", "cluster", "family", "class", "type",
    ))


def is_bridge_target(graph: MemoryGraph, target_id: str) -> bool:
    node = graph.nodes.get(target_id)
    if node is None:
        return "bridge" in target_id.lower()
    return "bridge" in target_id.lower() or str(getattr(node, "node_type", "")).lower() == "bridge"


def specific_shared_terms(source_text: str, target_text: str) -> List[str]:
    return clean_evidence_terms(attachment_specific_terms(source_text) & attachment_specific_terms(target_text))


def related_evidence_ok(
    *,
    evidence_terms: Sequence[str],
    source_text: str,
    target_text: str,
    target_id: str,
    graph: MemoryGraph,
    article_local_node_ids: Optional[Set[str]],
) -> Tuple[bool, Dict[str, Any]]:
    shared = specific_shared_terms(source_text, target_text)
    target_is_local = bool(article_local_node_ids and target_id in article_local_node_ids)
    target_is_bridge = is_bridge_target(graph, target_id)
    singleton_ok = evidence_terms_are_strong(
        evidence_terms,
        source_text=source_text,
        target_text=target_text,
    )

    if target_is_local:
        ok = len(shared) >= 2 or singleton_ok
        return ok, {
            "target_is_local": True,
            "target_is_bridge": target_is_bridge,
            "shared_terms": shared,
            "singleton_ok": singleton_ok,
            "rule": "local_requires_two_shared_or_one_strong_singleton",
        }

    ok = len(shared) >= 2
    return ok, {
        "target_is_local": False,
        "target_is_bridge": target_is_bridge,
        "shared_terms": shared,
        "singleton_ok": singleton_ok,
        "rule": "global_or_bridge_requires_two_shared_terms",
    }


def relation_debug_reason(
    *,
    relation: str,
    semantic_ok: bool,
    hard_conflict: bool,
    structural_target: bool = False,
    container_evidence: bool = False,
    relation_anchor_hit: bool = False,
    evidence_terms: Sequence[str] = (),
    source_flags: Sequence[str] = (),
    target_flags: Sequence[str] = (),
) -> str:
    rel = normalize_relation(relation)
    if semantic_ok:
        return "ok"
    if hard_conflict:
        return "hard_domain_conflict"
    if rel == "part_of":
        if not structural_target and not container_evidence:
            return "part_of_needs_structural_target_or_container_evidence"
        if not relation_anchor_hit:
            return "part_of_missing_container_anchor"
    if rel in {"support", "depend"} and not relation_anchor_hit:
        return f"{rel}_too_weak_to_keep"
    if rel in {"support", "depend"} and evidence_terms and len(evidence_terms) < 2:
        return f"{rel}_downgraded_to_related"
    if rel == "contradict" and not evidence_terms:
        return "contradict_missing_refutation_evidence"
    if target_flags and source_flags and not (set(source_flags) & set(target_flags)):
        return "cross_topic_or_unshared_domain"
    return "missing_specific_edge_evidence"


def maybe_downgrade_relation(
    source_text: str,
    target_text: str,
    relation: str,
) -> Tuple[str, bool, Dict[str, Any]]:
    rel = normalize_relation(relation)
    if rel not in {"support", "depend"}:
        return rel, False, {"downgraded": False, "reason": "not_weak_relation"}

    evidence_terms = relation_specific_evidence_terms(source_text, target_text, rel)
    shared = clean_evidence_terms(attachment_specific_terms(source_text) & attachment_specific_terms(target_text))
    weak = len(evidence_terms) < 2 or not any(term in relation_anchor_terms(rel) for term in evidence_terms)
    if weak:
        return "related", True, {
            "downgraded": True,
            "from_relation": rel,
            "to_relation": "related",
            "reason": "weak_support_or_depend_relation",
            "evidence_terms": evidence_terms,
            "shared_terms": shared,
        }
    return rel, False, {
        "downgraded": False,
        "reason": "strong_enough_relation",
        "evidence_terms": evidence_terms,
        "shared_terms": shared,
    }


def action_type(row: Mapping[str, Any]) -> str:
    return str((row.get("action_spec", {}) or {}).get("action_type", ""))


def signal_copy_overlap(row: Mapping[str, Any]) -> float:
    spec = row.get("action_spec", {}) or {}
    if str(spec.get("action_type", "")) != "add_node":
        return 0.0
    return float(lexical_overlap(str(row.get("signal_text", "")), str(spec.get("text", ""))))


def is_copy_like_node(graph: MemoryGraph, nid: str) -> bool:
    node = graph.nodes.get(nid)
    if node is None:
        return True
    text = str(getattr(node, "text", "") or "")
    readable_id = nid.replace("_", " ")
    if float(lexical_overlap(readable_id, text)) >= 0.72:
        return True
    if nid.startswith(("cons_", "repair_")) and float(lexical_overlap(readable_id, text)) >= 0.55:
        return True
    return False


def repair_topic_flags(text: str) -> Set[str]:
    flags = set(attachment_topic_flags(text))
    s = str(text or "").lower().replace("_", " ")
    if any(k in s for k in (
        "neural network", "neural nets", "neuron", "neurons", "backpropagation",
        "gradient descent", "gradients", "loss function", "cross entropy", "optimizer",
        "activation function", "activation functions", "hidden layer", "output layer",
        "weights and biases", "weighted sum", "weight matrix", "nonlinearity",
        "relu", "gelu", "silu", "sigmoid", "tanh", "dropout", "batch normalization",
        "attention", "embedding", "transformer",
    )):
        flags.add("ml_ai")
    if any(k in s for k in ("weighted sum", "nonlinearity", "gradient", "derivative")):
        flags.add("math_analysis")
    if "activation energy" in s or any(k in s for k in ("chemical activation", "chemical reaction", "catalyst", "enzyme")):
        flags.add("chemistry")
    if any(k in s for k in ("diffusion", "atp", "active transport", "protein", "cell")):
        flags.add("biology")
    if any(k in s for k in ("momentum", "impulse", "kinetic", "temperature")):
        flags.add("physics")
    if any(k in s for k in ("stress test", "randomized", "debug", "wrong answer")):
        flags.add("debug")
    if any(k in s for k in ("contrast", "legibility", "visibility", "feedback")):
        flags.add("hci")
    if any(k in s for k in ("methane", "combustion", "reaction", "molecule")):
        flags.add("chemistry")
    if any(k in s for k in ("b-tree", "database", "table", "sql")):
        flags.add("database")
    return flags


def edge_candidate_ok(graph: MemoryGraph, signal: str, nid: str, *, allow_structural: bool = True) -> bool:
    node = graph.nodes.get(nid)
    if node is None:
        return False
    if is_copy_like_node(graph, nid):
        return False
    structural = is_summary_node(nid, graph) or "hub" in nid.lower()
    if structural and not allow_structural:
        return False
    text = f"{nid.replace('_', ' ')} {node.text}"
    overlap = float(lexical_overlap(signal, text))
    specific = float(attachment_specific_overlap(signal, text))
    src_flags = repair_topic_flags(signal)
    dst_flags = repair_topic_flags(text)
    hard_domains = {"biology", "physics", "chemistry", "database", "hci", "debug", "networking"}
    if "ml_ai" in src_flags and (dst_flags & hard_domains) and "ml_ai" not in dst_flags:
        return False
    if src_flags and dst_flags and not (src_flags & dst_flags):
        return False
    if src_flags and dst_flags and (src_flags & dst_flags):
        return True
    if specific > 0.0:
        return True
    if structural:
        return overlap >= 0.06
    return overlap >= 0.16


def strict_target_ok(graph: MemoryGraph, signal: str, nid: str) -> bool:
    node = graph.nodes.get(nid)
    if node is None:
        return False
    text = f"{nid.replace('_', ' ')} {node.text}"
    overlap = float(lexical_overlap(signal, text))
    specific = float(attachment_specific_overlap(signal, text))
    structural = is_summary_node(nid, graph) or "hub" in nid.lower()
    src_flags = repair_topic_flags(signal)
    dst_flags = repair_topic_flags(text)
    hard_domains = {"biology", "physics", "chemistry", "database", "hci", "debug", "networking"}
    if "ml_ai" in src_flags and (dst_flags & hard_domains) and "ml_ai" not in dst_flags:
        return False
    if src_flags and dst_flags and not (src_flags & dst_flags):
        return False
    if src_flags and dst_flags and (src_flags & dst_flags):
        return True
    if specific > 0.0:
        return True
    if structural:
        return overlap >= 0.06
    return overlap >= 0.16


def rebuild_edges_for_repair(
    graph: MemoryGraph,
    *,
    signal: str,
    repaired_text: str,
    row: Mapping[str, Any],
    max_edges: int = 3,
) -> Tuple[List[str], List[str], Dict[str, Any]]:
    article_local_ids = set(str(x) for x in (row.get("article_local_node_ids", []) or []) if str(x) in graph.nodes)
    retrieved = retrieve_context(graph, f"{signal} {repaired_text}", top_k=12)
    raw_targets, raw_rels = add_node_attachment_targets(
        graph=graph,
        signal_text=f"{signal} {repaired_text}",
        retrieved=retrieved,
        article_local_node_ids=article_local_ids,
        max_targets=max_edges * 3,
    )

    original_spec = row.get("action_spec", {}) or {}
    original_rels = original_spec.get("target_edge_relations", []) or []
    for idx, original_nid in enumerate(original_spec.get("target_nodes", []) or []):
        nid = str(original_nid)
        if nid in graph.nodes and nid not in raw_targets:
            raw_targets.append(nid)
            raw_rels.append(str(original_rels[idx]) if idx < len(original_rels) else str(original_spec.get("relation", "related")))

    chosen: List[str] = []
    rels_out: List[str] = []
    rejected: List[Dict[str, Any]] = []
    for idx, nid in enumerate(raw_targets):
        if nid in chosen:
            continue
        if not edge_candidate_ok(graph, signal, nid, allow_structural=True):
            rejected.append({"dst": nid, "reason": "weak_or_copy_like_target"})
            continue
        chosen.append(nid)
        rels_out.append(str(raw_rels[idx]) if idx < len(raw_rels) else "related")
        if len(chosen) >= max_edges:
            break

    if not chosen:
        signal_flags = repair_topic_flags(f"{signal} {repaired_text}")
        fallback_rows: List[Tuple[float, str]] = []
        if signal_flags:
            for nid, node in graph.nodes.items():
                if nid in chosen or is_copy_like_node(graph, nid):
                    continue
                node_blob = f"{nid.replace('_', ' ')} {node.text}"
                node_flags = repair_topic_flags(node_blob)
                if not (signal_flags & node_flags):
                    continue
                overlap = float(lexical_overlap(signal, node_blob))
                specific = float(attachment_specific_overlap(signal, node_blob))
                structural_bonus = 0.04 if (is_summary_node(nid, graph) or "hub" in nid.lower()) else 0.0
                score = overlap + specific + structural_bonus + 0.02 * float(getattr(node, "importance", 0.0) or 0.0)
                fallback_rows.append((score, nid))
        fallback_rows.sort(key=lambda x: (-x[0], x[1]))
        for _, nid in fallback_rows:
            if not edge_candidate_ok(graph, signal, nid, allow_structural=True):
                rejected.append({"dst": nid, "reason": "topic_fallback_rejected"})
                continue
            chosen.append(nid)
            rels_out.append("related")
            if len(chosen) >= min(max_edges, 2):
                break

    filtered_targets: List[str] = []
    filtered_rels: List[str] = []
    for idx, nid in enumerate(chosen):
        if strict_target_ok(graph, signal, nid):
            filtered_targets.append(nid)
            filtered_rels.append(rels_out[idx] if idx < len(rels_out) else "related")
        else:
            rejected.append({"dst": nid, "reason": "dropped_by_strict_edge_filter"})
    if filtered_targets:
        chosen = filtered_targets[:max_edges]
        rels_out = filtered_rels[:max_edges]

    semantic_targets: List[str] = []
    semantic_rels: List[str] = []
    for idx, nid in enumerate(chosen):
        rel = rels_out[idx] if idx < len(rels_out) else "related"
        target_text = f"{nid.replace('_', ' ')} {graph.nodes[nid].text}" if nid in graph.nodes else nid.replace("_", " ")
        rel, downgraded, downgrade_debug = maybe_downgrade_relation(repaired_text, target_text, rel)
        if downgraded:
            rejected.append({"dst": nid, "reason": "relation_downgraded_to_related", "downgrade_debug": downgrade_debug})
        check = semantic_edge_check(
            graph,
            source_text=repaired_text,
            target_id=nid,
            relation=rel,
            article_local_node_ids=article_local_ids,
        )
        if check.get("semantic_ok"):
            semantic_targets.append(nid)
            semantic_rels.append(rel)
        else:
            rejected.append({"dst": nid, "reason": "dropped_by_semantic_edge_filter", "semantic_check": check})
    if semantic_targets:
        chosen = semantic_targets[:max_edges]
        rels_out = semantic_rels[:max_edges]

    return chosen, rels_out, {
        "retrieved_ids": [str(r.get("id", "")) for r in retrieved],
        "raw_targets": raw_targets,
        "chosen_targets": chosen,
        "rejected_targets": rejected,
        "article_local_node_ids": sorted(article_local_ids),
        "article_local_node_count": len(article_local_ids),
    }


def strict_edge_quality(graph: MemoryGraph, signal: str, plan: Mapping[str, Any]) -> Tuple[bool, Dict[str, Any]]:
    pa = plan.get("proposed_action", {}) if isinstance(plan, Mapping) else {}
    if not isinstance(pa, Mapping) or str(plan.get("task_type", "")) != "add_node":
        return True, {"reason": "not_add_node"}
    edges = [e for e in pa.get("edges_to", []) or [] if isinstance(e, Mapping)]
    if not edges:
        return False, {"reason": "no_edges_to", "bad_targets": []}
    bad_targets: List[Dict[str, Any]] = []
    useful_targets = 0
    for edge in edges:
        dst = str(edge.get("dst", ""))
        node = graph.nodes.get(dst)
        if node is None:
            bad_targets.append({"dst": dst, "reason": "missing"})
            continue
        overlap = float(lexical_overlap(signal, f"{dst.replace('_', ' ')} {node.text}"))
        structural = str(getattr(node, "node_type", "")).lower() in {"summary", "hub", "overview"} or "hub" in dst.lower()
        if strict_target_ok(graph, signal, dst):
            useful_targets += 1
            continue
        if structural and overlap >= 0.06:
            useful_targets += 1
            continue
        bad_targets.append({"dst": dst, "reason": "low_signal_overlap", "overlap": overlap, "node_type": getattr(node, "node_type", "")})
    ok = useful_targets > 0 and not bad_targets
    return ok, {
        "reason": "ok" if ok else "bad_or_weak_edges",
        "useful_targets": useful_targets,
        "edge_count": len(edges),
        "bad_targets": bad_targets,
    }


def semantic_edge_check(
    graph: MemoryGraph,
    *,
    source_text: str,
    target_id: str,
    relation: str,
    article_local_node_ids: Optional[Set[str]] = None,
) -> Dict[str, Any]:
    node = graph.nodes.get(target_id)
    if node is None:
        return {
            "dst": target_id,
            "relation": relation,
            "semantic_ok": False,
            "reason": "missing_target",
            "justification": "",
            "evidence_terms": [],
            "edge_qa": {},
        }

    target_text = f"{target_id.replace('_', ' ')} {node.text}"
    src_flags = repair_topic_flags(source_text)
    dst_flags = repair_topic_flags(target_text)
    hard_domains = {"biology", "physics", "chemistry", "database", "hci", "debug", "networking"}
    hard_conflict = bool("ml_ai" in src_flags and (dst_flags & hard_domains) and "ml_ai" not in dst_flags)

    source_label = concept_label(source_text)
    target_label = concept_label(f"{target_id.replace('_', ' ')} {node.text}")
    relation = normalize_relation(relation)
    relation_plausible = relation in {"related", "part_of", "support", "depend", "example_of", "refine", "contradict"}
    mentions_both = bool(source_label and target_label)
    qa_question = relation_question(relation, source_label, target_label)
    expected_anchor_terms = relation_anchor_terms(relation)
    evidence_terms = relation_specific_evidence_terms(source_text, target_text, relation)
    shared_flags = sorted(src_flags & dst_flags)
    relation_anchor_hit = any(term in expected_anchor_terms for term in evidence_terms)
    structural_target = is_true_container_target(graph, target_id)
    container_evidence = relation == "part_of" and any(
        term in PART_OF_CONTAINER_TERMS for term in evidence_terms
    )
    related_debug: Dict[str, Any] = {}
    if relation == "part_of":
        relation_plausible = relation_plausible and structural_target and container_evidence
    elif relation == "related":
        related_ok, related_debug = related_evidence_ok(
            evidence_terms=evidence_terms,
            source_text=source_text,
            target_text=target_text,
            target_id=target_id,
            graph=graph,
            article_local_node_ids=article_local_node_ids,
        )
        relation_plausible = relation_plausible and related_ok
    elif relation != "related":
        relation_plausible = relation_plausible and relation_anchor_hit
    qa_expected = ", ".join(evidence_terms)
    if relation == "related":
        strength_ok = related_ok
    else:
        strength_ok = evidence_terms_are_strong(
            evidence_terms,
            source_text=source_text,
            target_text=target_text,
        )
    qa_score = 1.0 if strength_ok and mentions_both and relation_plausible and not hard_conflict else 0.0
    semantic_ok = bool(qa_score >= 1.0)
    if relation == "part_of" and not semantic_ok and not structural_target and not container_evidence:
        reason = "part_of_needs_structural_target_or_container_evidence"
    elif relation == "part_of" and not semantic_ok:
        reason = "part_of_missing_container_anchor"
    elif relation == "related" and not semantic_ok:
        if related_debug.get("target_is_local"):
            reason = "related_local_needs_stronger_evidence"
        elif related_debug.get("target_is_bridge"):
            reason = "related_bridge_needs_two_shared_terms"
        else:
            reason = "related_global_needs_two_shared_terms"
    elif relation in {"support", "depend"} and not semantic_ok and relation_anchor_hit:
        reason = f"{relation}_weak_downgrade_candidate"
    elif relation in {"support", "depend"} and not semantic_ok:
        reason = f"{relation}_too_weak"
    else:
        reason = relation_debug_reason(
            relation=relation,
            semantic_ok=semantic_ok,
            hard_conflict=hard_conflict,
            structural_target=structural_target,
            container_evidence=container_evidence,
            relation_anchor_hit=relation_anchor_hit,
            evidence_terms=evidence_terms,
            source_flags=src_flags,
            target_flags=dst_flags,
        )
    justification = (
        f"{source_label} connects to {target_label} through {qa_expected}."
        if evidence_terms
        else ""
    )
    return {
        "dst": target_id,
        "relation": relation,
        "semantic_ok": semantic_ok,
        "reason": reason,
        "justification": justification,
        "evidence_terms": evidence_terms,
        "relation_anchor_hit": relation_anchor_hit,
        "structural_target": structural_target,
        "container_evidence": container_evidence,
        "relation_expected_anchors": sorted(expected_anchor_terms),
        "mentions_source_and_target": mentions_both,
        "relation_plausible": relation_plausible,
        "related_debug": related_debug if relation == "related" else {},
        "source_flags": sorted(src_flags),
        "target_flags": sorted(dst_flags),
        "edge_qa": {
            "question": qa_question,
            "expected": qa_expected,
            "score": qa_score,
        },
    }


def semantic_edge_checks(graph: MemoryGraph, source_text: str, plan: Mapping[str, Any]) -> Tuple[bool, List[Dict[str, Any]]]:
    return semantic_edge_checks_with_locals(graph, source_text, plan, article_local_node_ids=None)


def semantic_edge_checks_with_locals(
    graph: MemoryGraph,
    source_text: str,
    plan: Mapping[str, Any],
    *,
    article_local_node_ids: Optional[Set[str]],
) -> Tuple[bool, List[Dict[str, Any]]]:
    pa = plan.get("proposed_action", {}) if isinstance(plan, Mapping) else {}
    if not isinstance(pa, Mapping) or str(plan.get("task_type", "")) != "add_node":
        return True, []
    rows = []
    for edge in pa.get("edges_to", []) or []:
        if not isinstance(edge, Mapping):
            continue
        rows.append(semantic_edge_check(
            graph,
            source_text=source_text,
            target_id=str(edge.get("dst", "")),
            relation=str(edge.get("relation", "related")),
            article_local_node_ids=article_local_node_ids,
        ))
    return bool(rows and all(bool(r.get("semantic_ok")) for r in rows)), rows


def compact_concept_text(signal_text: str, *, max_terms: int = 8) -> str:
    terms: List[str] = []
    seen = set()
    for term in content_tokens(signal_text):
        if term in seen:
            continue
        seen.add(term)
        terms.append(term)
        if len(terms) >= max_terms:
            break
    if not terms:
        return "Reusable concept from the current signal."
    if len(terms) <= 3:
        return "Concept: " + ", ".join(terms) + "."
    head = " ".join(terms[:3])
    tail = ", ".join(terms[3:])
    return f"{head}: {tail}."


def repair_add_node_copy(
    graph: MemoryGraph,
    row: Mapping[str, Any],
    *,
    copy_threshold: float = 0.92,
    max_edges: int = 3,
) -> Optional[Dict[str, Any]]:
    spec = dict(row.get("action_spec", {}) or {})
    if str(spec.get("action_type", "")) != "add_node":
        return None
    signal = str(row.get("signal_text", ""))
    original_text = str(spec.get("text", ""))
    if float(lexical_overlap(signal, original_text)) < copy_threshold:
        return None

    # Try shorter concept texts until the copy-overlap guard is cleared. Keep
    # enough content terms to preserve QA coverage, but do not copy the sentence.
    variants: List[str] = []
    for max_terms in (8, 7, 6, 5, 4):
        variants.append(compact_concept_text(signal, max_terms=max_terms))
    variants.append("Reusable concept: " + ", ".join(content_tokens(signal)[:4]) + ".")

    best: Optional[Dict[str, Any]] = None
    for text in variants:
        overlap = float(lexical_overlap(signal, text))
        if overlap >= copy_threshold:
            continue
        repaired = dict(spec)
        repaired["text"] = text
        old_id = str(repaired.get("new_node_id") or "")
        if not old_id or float(lexical_overlap(old_id.replace("_", " "), signal)) >= 0.70:
            repaired["new_node_id"] = "repair_" + safe_node_id(text)
        targets, rels, edge_debug = rebuild_edges_for_repair(
            graph,
            signal=signal,
            repaired_text=text,
            row=row,
            max_edges=max_edges,
        )
        repaired["target_nodes"] = targets
        repaired["target_edge_relations"] = rels
        repaired["relation"] = rels[0] if rels else "related"
        repaired["article_local_node_ids"] = sorted(edge_debug.get("article_local_node_ids", []))
        candidate = {
            "action_spec": repaired,
            "repair_text": text,
            "repair_signal_copy_overlap": overlap,
            "repair_reason": "add_node copied the signal too closely; compressed into a concept-style node",
            "edge_repair_debug": edge_debug,
        }
        if best is None or overlap > float(best.get("repair_signal_copy_overlap", 0.0)):
            # Prefer the richest repaired text that still clears the guard.
            best = candidate
    return best


def score_action_spec(graph: MemoryGraph, signal: str, spec: Mapping[str, Any], *, top_k: int, hops: int) -> Dict[str, Any]:
    before = score_graph_for_signal(graph, signal, top_k=top_k, hops=hops)
    after_graph = clone_graph(graph)
    action_spec = row_to_action_spec(spec, default_text=signal)
    committed, plan, validation = commit_spec(after_graph, signal_text=signal, spec=action_spec)
    after = score_graph_for_signal(after_graph, signal, top_k=top_k, hops=hops) if committed else before
    proposed = {}
    pa = plan.get("proposed_action", {}) if isinstance(plan, Mapping) else {}
    if isinstance(pa, Mapping):
        if plan.get("task_type") == "add_node":
            new_node = pa.get("new_node", {}) if isinstance(pa.get("new_node"), Mapping) else {}
            proposed = {
                "id": new_node.get("id"),
                "text": new_node.get("text"),
                "edges_to": pa.get("edges_to", []),
            }
        else:
            proposed = dict(pa)
    checks, explanations = deterministic_graph_checks(
        signal_text=signal,
        context_nodes=[],
        planner_action=str(plan.get("task_type", "")),
        planner_proposed=proposed,
        planner_validation={"valid": bool(validation.get("ok")), "errors": validation.get("errors", [])},
        graph=graph,
    )
    strict_edges_ok, strict_edges = strict_edge_quality(graph, signal, plan)
    source_text = str(((plan.get("proposed_action", {}) or {}).get("new_node", {}) or {}).get("text", signal))
    article_local_node_ids = {
        str(x) for x in (spec.get("article_local_node_ids", []) or [])
        if str(x) in graph.nodes
    } if isinstance(spec, Mapping) else set()
    semantic_edges_ok, semantic_edges = semantic_edge_checks_with_locals(
        graph,
        source_text,
        plan,
        article_local_node_ids=article_local_node_ids,
    )
    return {
        "qa_before": float(before.get("qa_score", 0.0)),
        "qa_after": float(after.get("qa_score", 0.0)),
        "qa_delta": float(after.get("qa_score", 0.0)) - float(before.get("qa_score", 0.0)),
        "commit_simulated": bool(committed),
        "plan": plan,
        "validation": validation,
        "guard_checks": checks,
        "guard_explanations": explanations,
        "strict_edges_ok": bool(strict_edges_ok),
        "strict_edges": strict_edges,
        "semantic_edges_ok": bool(semantic_edges_ok),
        "semantic_edges": semantic_edges,
        "guard_passes": all(bool(v) for v in checks.values()) and bool(strict_edges_ok) and bool(semantic_edges_ok),
    }


def should_attempt_repair(row: Mapping[str, Any], *, min_qa_delta: float, copy_threshold: float) -> bool:
    return (
        action_type(row) == "add_node"
        and float(row.get("qa_delta", 0.0)) >= min_qa_delta
        and signal_copy_overlap(row) >= copy_threshold
    )


def main() -> int:
    ap = argparse.ArgumentParser(description="Logging-only action repair for high-QA add_node candidates that fail copy/quality guards.")
    ap.add_argument("--seed-graph", required=True)
    ap.add_argument("--comparison-jsonl", required=True, help="Per-candidate rows from compare_reward_vs_qa.py.")
    ap.add_argument("--commits-jsonl", default="", help="Optional commits file to replay graph state between candidate groups.")
    ap.add_argument("--out-json", default="")
    ap.add_argument("--out-jsonl", required=True)
    ap.add_argument("--min-qa-delta", type=float, default=0.08)
    ap.add_argument("--copy-overlap-threshold", type=float, default=0.92)
    ap.add_argument("--top-k", type=int, default=8)
    ap.add_argument("--hops", type=int, default=1)
    ap.add_argument("--max-repairs", type=int, default=0)
    ap.add_argument("--repair-max-edges", type=int, default=3)
    args = ap.parse_args()

    graph = MemoryGraph.load_json(args.seed_graph)
    rows = read_jsonl(args.comparison_jsonl)
    commits_by_key: Dict[Tuple[Any, ...], Mapping[str, Any]] = {}
    if args.commits_jsonl:
        for row in read_jsonl(args.commits_jsonl):
            commits_by_key[commit_key(row)] = row

    out_rows: List[Dict[str, Any]] = []
    attempted = 0
    successful = 0
    kept_positive_qa = 0
    guard_passed = 0
    action_counts: Counter[str] = Counter()
    accepted_target_counts: Counter[str] = Counter()
    accepted_nonlocal_target_counts: Counter[str] = Counter()

    current_key: Optional[Tuple[Any, ...]] = None
    for row in rows:
        key = scored_commit_key(row)
        if current_key is not None and key != current_key:
            commit = commits_by_key.get(current_key)
            if commit and commit.get("committed") and isinstance(commit.get("commit_plan"), Mapping):
                apply_plan(graph, commit.get("commit_plan", {}) or {})
        current_key = key

        if args.max_repairs and attempted >= args.max_repairs:
            break
        if not should_attempt_repair(row, min_qa_delta=args.min_qa_delta, copy_threshold=args.copy_overlap_threshold):
            continue

        attempted += 1
        signal = str(row.get("signal_text", ""))
        repair = repair_add_node_copy(graph, row, copy_threshold=args.copy_overlap_threshold, max_edges=args.repair_max_edges)
        if repair is None:
            out_rows.append({
                "repair_attempted": True,
                "repair_success": False,
                "failure_reason": "no_repair_candidate",
                "original": row,
            })
            continue

        repaired_score = score_action_spec(graph, signal, repair["action_spec"], top_k=args.top_k, hops=args.hops)
        action_counts[str((repair["action_spec"] or {}).get("action_type", ""))] += 1
        positive = float(repaired_score.get("qa_delta", 0.0)) >= float(args.min_qa_delta)
        passes = bool(repaired_score.get("guard_passes"))
        if positive:
            kept_positive_qa += 1
        if passes:
            guard_passed += 1
        ok = positive and passes
        if ok:
            successful += 1
            local_ids = {
                str(x) for x in (((repair.get("action_spec", {}) or {}).get("article_local_node_ids", []) or []))
            }
            for nid in ((repair.get("action_spec", {}) or {}).get("target_nodes", []) or []):
                nid_s = str(nid)
                accepted_target_counts[nid_s] += 1
                if nid_s not in local_ids:
                    accepted_nonlocal_target_counts[nid_s] += 1
        out_rows.append({
            "repair_attempted": True,
            "repair_success": bool(ok),
            "failure_reason": None if ok else ("guard_failed" if not passes else "qa_delta_too_low"),
            "raw_text_id": row.get("raw_text_id"),
            "sub_signal_index": row.get("sub_signal_index"),
            "candidate_index": row.get("candidate_index"),
            "signal_text": signal,
            "original_action_spec": row.get("action_spec", {}),
            "original_reward": row.get("reward"),
            "original_qa_delta": row.get("qa_delta"),
            "original_signal_copy_overlap": signal_copy_overlap(row),
            "repaired_action_spec": repair["action_spec"],
            "repair_reason": repair["repair_reason"],
            "repair_signal_copy_overlap": repair["repair_signal_copy_overlap"],
            "edge_repair_debug": repair.get("edge_repair_debug", {}),
            **repaired_score,
        })

    report = {
        "comparison_jsonl": args.comparison_jsonl,
        "repairs_attempted": attempted,
        "repair_successes": successful,
        "repair_success_rate": (successful / attempted) if attempted else None,
        "repairs_with_positive_qa_delta": kept_positive_qa,
        "repairs_guard_passed": guard_passed,
        "min_qa_delta": args.min_qa_delta,
        "copy_overlap_threshold": args.copy_overlap_threshold,
        "repaired_action_distribution": dict(action_counts),
        "accepted_target_frequency": dict(accepted_target_counts.most_common(10)),
        "accepted_nonlocal_target_frequency": dict(accepted_nonlocal_target_counts.most_common(10)),
        "accepted_nonlocal_target_overuse": {
            nid: (count / successful)
            for nid, count in accepted_nonlocal_target_counts.items()
            if successful and (count / successful) > 0.20
        },
        "examples": out_rows[:10],
    }
    write_jsonl(args.out_jsonl, out_rows)
    text = json.dumps(report, ensure_ascii=False, indent=2)
    if args.out_json:
        Path(args.out_json).write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
