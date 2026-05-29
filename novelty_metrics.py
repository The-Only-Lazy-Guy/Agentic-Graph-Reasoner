"""
Graph-based novelty metrics for Answerer-v2 evaluation.

These look at the FINAL session graph and the action trace, not just the
answer text. Each metric answers a specific question about whether the
session graph is actually carrying reasoning weight or whether the model
is just producing nice-looking grounded summaries.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Set

import numpy as np

from answerer_v1 import SessionGraph
from answerer_v2 import (
    NODE_EVIDENCE, NODE_NOTE, NODE_CONCLUSION,
    get_plan_steps, nodes_for_plan_step, has_evidence_grounding,
    _incoming,
)


# ---------------------------------------------------------------------------
# Embedding-based anti-claim check
# ---------------------------------------------------------------------------

from embedder import encode_batch as _encode  # shared MiniLM


# Negation / refutation markers. Includes:
#  - bare negators:    not, never, cannot, n't, without
#  - "no <noun>":      no medium, no negative cycle, ...
#  - refutation words: false, wrong, incorrect, untrue, mistaken,
#                      misconception, fallacy, myth, refute(d|s)?,
#                      debunk(ed|s)?, disprove(d|s|n)?
# Limitation: this catches lexical refutation, not directional opposition
# (e.g. "from hot to cold" vs "from cold to hot" share no negator token).
# Those cases will still false-positive when claim and chunk are topic-
# similar but directionally opposite. Treat the gate as a sensitive
# detector and inspect violation_details when it fires.
_NEG_RE = re.compile(
    r"\b(not|never|cannot|n't|without|"
    r"false|falsehood|wrong(ly)?|incorrect(ly)?|untrue|mistaken|"
    r"misconception|fallac(y|ies)|myth|"
    r"refute[ds]?|debunk(ed|s)?|disprov(e[ds]?|en|ing)|"
    # Equivalence-negators only — these reliably negate an "X is Y" claim:
    # "X and Y are distinct/different/distinguishable"
    # Excluded: unlike / contrary / contrast — these are CONTRASTIVE in
    # context ("X requires a medium, unlike Y") not equivalence-negating,
    # and they false-positive on assertions of a positive claim about X
    # that happens to mention Y for contrast.
    r"distinct(ly)?|differ(s|ent|ent\s+from)?|distinguish(ed|es|able)?|"
    r"contradic(t|ts|ted|tion))\b"
    r"|\bno\s+\w+",
    re.IGNORECASE,
)


def _has_negation(text: str) -> bool:
    return bool(_NEG_RE.search(text or ""))


def _negation_polarity(text: str) -> int:
    """Polarity = count of negation tokens mod 2. Two negations cancel
    (double negative → positive). One negation flips polarity.

    Handles cases like Q2 Stage 2:
        "...with a negative-weight edge (but no negative cycle),
         you cannot trust the result"
    which contains TWO negation matches ("no negative", "cannot"), so
    its polarity is 0 (positive in the double-negative sense — the
    sentence asserts "you cannot trust" which IS a positive claim that
    Dijkstra is unsafe). Comparing this to the claim
        "Dijkstra is safe as long as there is no negative cycle"
    which has ONE negation match → polarity 1 → polarities differ → the
    chunk is refuting the claim, not asserting it.
    """
    return len(_NEG_RE.findall(text or "")) % 2


def detect_claim_violations(
    answer_text: str,
    must_not_claim: List[str],
    *,
    threshold: float = 0.65,
) -> List[Dict[str, Any]]:
    """Returns list of {claim, max_similarity, matched_chunk} for any
    must_not_claim entry whose semantic similarity to some sentence in the
    answer exceeds the threshold. Falls back to substring on error.

    Substring is too literal — it missed Q2's
    "Dijkstra's result is trustworthy ... no negative cycle"
    against
    "Dijkstra is safe as long as there is no negative cycle".
    Embedding cosine catches that paraphrase.
    """
    if not must_not_claim or not (answer_text or "").strip():
        return []
    chunks = [s.strip() for s in re.split(r"(?<=[.!?])\s+", answer_text) if s.strip()]
    if not chunks:
        return []
    try:
        import numpy as np
        all_texts = chunks + [str(c) for c in must_not_claim]
        embs = _encode(all_texts)
        chunk_embs = embs[: len(chunks)]
        claim_embs = embs[len(chunks):]
        sims = chunk_embs @ claim_embs.T  # [n_chunks, n_claims]
        violations: List[Dict[str, Any]] = []
        for j, claim in enumerate(must_not_claim):
            claim_neg = _has_negation(str(claim))
            best_i = int(np.argmax(sims[:, j]))
            best_sim = float(sims[best_i, j])
            if best_sim < threshold:
                continue
            chunk_neg = _has_negation(chunks[best_i])
            # If exactly one of {claim, chunk} contains a negation token,
            # they're expressing opposing propositions (chunk refutes
            # claim or vice versa). Treat as agreement, not violation.
            #
            # Was polarity-mod-2 (count negations mod 2), which over-
            # corrected on chunks like "X are distinct, X are distinct"
            # where 2 negators cancelled to polarity 0 even though the
            # chunk clearly refutes a claim with 0 negators. Binary
            # has-negation is simpler and matches every real case in
            # our suite.
            if claim_neg != chunk_neg:
                continue
            violations.append({
                "claim": claim,
                "max_similarity": round(best_sim, 3),
                "matched_chunk": chunks[best_i][:200],
                "claim_negated": claim_neg,
                "chunk_negated": chunk_neg,
            })
        return violations
    except Exception as ex:
        ans = answer_text.lower()
        return [
            {"claim": c, "max_similarity": None, "matched_chunk": None, "fallback": str(ex)}
            for c in must_not_claim if str(c).lower() in ans
        ]


# ---------------------------------------------------------------------------
# Support chain inspection
# ---------------------------------------------------------------------------

def support_chain_depth(
    session: SessionGraph, node_id: str, _visited: Set[str] = None,
) -> int:
    """Length of the longest support chain from node_id back to an evidence
    node. Evidence = depth 1; evidence -> conclusion = depth 2;
    evidence -> note -> conclusion = depth 3."""
    if _visited is None:
        _visited = set()
    if node_id in _visited:
        return 0
    _visited.add(node_id)
    node = session.nodes.get(node_id)
    if node is None:
        return 0
    if node.node_type == NODE_EVIDENCE:
        return 1
    parents = _incoming(session, node_id, "supports") + _incoming(session, node_id, "derived_from")
    if not parents:
        return 1
    best = 0
    for p in parents:
        d = support_chain_depth(session, p, set(_visited))
        if d > best:
            best = d
    return 1 + best


def transitive_evidence_ancestors(
    session: SessionGraph, node_id: str, _visited: Set[str] = None,
) -> Set[str]:
    """All evidence node ids reachable backward via supports/derived_from."""
    if _visited is None:
        _visited = set()
    if node_id in _visited:
        return set()
    _visited.add(node_id)
    node = session.nodes.get(node_id)
    if node is None:
        return set()
    if node.node_type == NODE_EVIDENCE:
        return {node_id}
    result: Set[str] = set()
    parents = _incoming(session, node_id, "supports") + _incoming(session, node_id, "derived_from")
    for p in parents:
        result |= transitive_evidence_ancestors(session, p, _visited)
    return result


# ---------------------------------------------------------------------------
# usage_coverage: does each support_usage string semantically appear in
# the support node's actual text?
# ---------------------------------------------------------------------------

USAGE_COSINE_THRESHOLD = 0.45  # below this, the usage string is treated
                               # as inconsistent with the support text


def compute_usage_coverage(session: SessionGraph) -> Dict[str, Any]:
    """For every successful NOTE/CONCLUDE written by the controller,
    check that each `support_usage[support_id]` string is actually
    consistent with `session.nodes[support_id].text`. Returns the
    aggregate coverage plus a per-write breakdown.

    Cheap approximation: cosine of MiniLM mean-pool embeddings between
    the usage string and the support text. Below USAGE_COSINE_THRESHOLD
    we consider the usage to be unsupported by the cited node — i.e.
    the model declared "I used X from this support" but X is not really
    in the support's text.
    """
    writes: List[Dict[str, Any]] = []
    for node in session.nodes.values():
        if node.node_type not in (NODE_NOTE, NODE_CONCLUSION):
            continue
        usage = (node.metadata or {}).get("support_usage") or {}
        if not usage:
            continue
        for sid, used in usage.items():
            sup = session.nodes.get(sid)
            if sup is None or not str(used).strip() or not (sup.text or "").strip():
                continue
            writes.append({
                "write_id": node.id,
                "support_id": sid,
                "usage": used,
                "support_text": sup.text,
            })

    if not writes:
        return {
            "usage_coverage": 1.0,  # nothing to check -> not failing
            "n_usage_pairs": 0,
            "n_weak": 0,
            "weak_examples": [],
        }

    # Batched embed of all usage strings + support texts.
    texts = []
    for w in writes:
        texts.append(w["usage"])
        texts.append(w["support_text"])
    embs = _encode(texts)
    sims: List[float] = []
    weak_examples: List[Dict[str, Any]] = []
    for i, w in enumerate(writes):
        u_emb = embs[2 * i]
        s_emb = embs[2 * i + 1]
        sim = float(np.dot(u_emb, s_emb))
        sims.append(sim)
        if sim < USAGE_COSINE_THRESHOLD:
            weak_examples.append({
                "write_id": w["write_id"],
                "support_id": w["support_id"],
                "usage": w["usage"][:120],
                "support_text": w["support_text"][:120],
                "sim": round(sim, 3),
            })

    n_strong = sum(1 for s in sims if s >= USAGE_COSINE_THRESHOLD)
    return {
        "usage_coverage": round(n_strong / len(sims), 3) if sims else 1.0,
        "n_usage_pairs": len(sims),
        "n_weak": len(weak_examples),
        "weak_examples": weak_examples[:5],  # truncate for log size
    }


# ---------------------------------------------------------------------------
# Per-question metrics
# ---------------------------------------------------------------------------

def evaluate_run(
    session: SessionGraph,
    trace: List[Dict[str, Any]],
    answer_text: str,
    question_row: Dict[str, Any],
) -> Dict[str, Any]:
    """Compute graph-grounded metrics for a single eval row."""
    required = set(question_row.get("required_evidence", []))
    forbidden = set(question_row.get("forbidden_shortcut_nodes", []))
    rubric = question_row.get("rubric", {}) or {}
    min_depth = int(question_row.get("min_evidence_depth", 2))

    plan_steps = get_plan_steps(session)
    conclusions = [n for n in session.nodes.values() if n.node_type == NODE_CONCLUSION]
    notes = [n for n in session.nodes.values() if n.node_type == NODE_NOTE]

    # Evidence nodes touched anywhere in the support ancestry of any
    # grounded conclusion. This is what the answer actually leans on.
    cited_evidence: Set[str] = set()
    for c in conclusions:
        if has_evidence_grounding(session, c.id):
            cited_evidence |= transitive_evidence_ancestors(session, c.id)

    # 1. graph_dependency: fraction of required_evidence used somewhere in
    # the support ancestry of a grounded conclusion.
    if required:
        hit = required & cited_evidence
        graph_dependency = len(hit) / len(required)
    else:
        graph_dependency = 1.0
    missing_evidence = sorted(required - cited_evidence)

    # 2. path_depth: max chain depth across grounded conclusions.
    max_depth = 0
    for c in conclusions:
        if has_evidence_grounding(session, c.id):
            d = support_chain_depth(session, c.id)
            if d > max_depth:
                max_depth = d
    depth_ok = max_depth >= min_depth

    # 3. support_minimality: fraction of writes whose supports list has
    # 1..4 ids. The executor enforces this hard-rule; we record it for
    # diagnostic continuity.
    note_or_concl_writes = [
        e for e in trace
        if e.get("action") in {"NOTE", "CONCLUDE"} and e.get("result", {}).get("success")
    ]
    minimal = sum(
        1 for e in note_or_concl_writes
        if 1 <= len(e.get("args", {}).get("supports", []) or []) <= 4
    )
    support_minimality = (minimal / len(note_or_concl_writes)) if note_or_concl_writes else 1.0
    avg_supports = (
        sum(len(e.get("args", {}).get("supports", []) or []) for e in note_or_concl_writes)
        / len(note_or_concl_writes)
    ) if note_or_concl_writes else 0.0

    # 4. no_shortcut: did the model avoid citing any forbidden_shortcut_nodes
    # as a direct support of a successful write?
    shortcut_violations = 0
    for e in note_or_concl_writes:
        sups = set(e.get("args", {}).get("supports", []) or [])
        if sups & forbidden:
            shortcut_violations += 1
    no_shortcut = shortcut_violations == 0

    # 5. contradiction_handling: did any successful write cite both
    # endpoints of a contradict edge?
    contradict_pairs = [
        (e.src, e.dst) for e in session.edges if e.relation == "contradict"
    ]
    contradict_violations = 0
    for entry in note_or_concl_writes:
        sups = set(entry.get("args", {}).get("supports", []) or [])
        for a, b in contradict_pairs:
            if a in sups and b in sups:
                contradict_violations += 1
                break
    contradiction_clean = contradict_violations == 0

    # 6. rubric.must_mention coverage in the final answer text
    must_mention = [str(s).lower() for s in rubric.get("must_mention", [])]
    answer_lower = (answer_text or "").lower()
    hit_kw = [k for k in must_mention if k in answer_lower]
    keyword_coverage = (len(hit_kw) / len(must_mention)) if must_mention else 1.0

    # 6b. usage_coverage: does each support_usage string semantically
    # appear in the support node's text? Catches "model cites a real
    # evidence node but invents content not present in that node."
    usage_info = compute_usage_coverage(session)

    # 7. rubric.must_not_claim violation: embedding cosine similarity
    # (paraphrase-aware), with substring fallback on error.
    # NOTE: strip the readout's "[supports: id1, id2, ...]" annotations
    # before scoring so that node-id strings that lexically resemble a
    # must_not_claim don't trigger false positives. Those annotations are
    # readout metadata, not part of the model's actual claim.
    forbidden_claims_raw = [str(s) for s in rubric.get("must_not_claim", []) if str(s).strip()]
    answer_for_claim_check = re.sub(
        r"\s*\[supports:[^\]]*\]", "", answer_text or "", flags=re.IGNORECASE,
    )
    claim_violation_records = detect_claim_violations(
        answer_for_claim_check, forbidden_claims_raw, threshold=0.65,
    )
    claim_violations = [r["claim"] for r in claim_violation_records]
    no_false_claims = not claim_violations

    # 8. Reject rate during exploration — high = model struggled
    rejected_writes = sum(
        1 for e in trace
        if e.get("action") in {"NOTE", "CONCLUDE"} and not e.get("result", {}).get("success")
    )

    # 9. Plan coverage already in packet.confidence; recompute here for self-containment
    cov_done = sum(
        1 for ps in plan_steps
        if any(
            has_evidence_grounding(session, c.id)
            for c in nodes_for_plan_step(session, ps.id, NODE_CONCLUSION)
        )
    )
    plan_coverage = cov_done / len(plan_steps) if plan_steps else 0.0

    # Composite "novelty pass" gate. Every guard must hold.
    novelty_pass = bool(
        graph_dependency >= 0.5
        and depth_ok
        and no_shortcut
        and contradiction_clean
        and no_false_claims
        and plan_coverage >= 0.5
    )

    return {
        "id": question_row.get("id"),
        "category": question_row.get("category"),
        "graph_dependency": round(graph_dependency, 3),
        "missing_required_evidence": missing_evidence,
        "cited_evidence_count": len(cited_evidence),
        "max_support_depth": max_depth,
        "depth_ok": depth_ok,
        "support_minimality": round(support_minimality, 3),
        "avg_supports_per_write": round(avg_supports, 2),
        "no_shortcut": no_shortcut,
        "shortcut_violations": shortcut_violations,
        "contradiction_clean": contradiction_clean,
        "contradict_violations": contradict_violations,
        "keyword_coverage": round(keyword_coverage, 3),
        "hit_keywords": hit_kw,
        "missed_keywords": [k for k in must_mention if k not in answer_lower],
        "no_false_claims": no_false_claims,
        "violated_claims": claim_violations,
        "violation_details": claim_violation_records,
        "usage_coverage": usage_info["usage_coverage"],
        "usage_pairs": usage_info["n_usage_pairs"],
        "usage_weak": usage_info["n_weak"],
        "usage_weak_examples": usage_info["weak_examples"],
        "rejected_writes": rejected_writes,
        "plan_coverage": round(plan_coverage, 3),
        "novelty_pass": novelty_pass,
    }


def aggregate(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not results:
        return {}
    n = len(results)
    keys_to_avg = [
        "graph_dependency", "max_support_depth", "support_minimality",
        "avg_supports_per_write", "keyword_coverage", "plan_coverage",
        "anchor_quality", "usage_coverage",
    ]
    keys_to_count_true = [
        "depth_ok", "no_shortcut", "contradiction_clean",
        "no_false_claims", "novelty_pass",
    ]
    out: Dict[str, Any] = {"n": n}
    for k in keys_to_avg:
        out[f"avg_{k}"] = round(sum(r.get(k, 0) or 0 for r in results) / n, 3)
    for k in keys_to_count_true:
        out[f"frac_{k}"] = round(sum(1 for r in results if r.get(k)) / n, 3)
    # Per-category breakdown
    by_cat: Dict[str, Dict[str, Any]] = {}
    for r in results:
        cat = r.get("category", "?")
        by_cat.setdefault(cat, {"n": 0, "pass": 0})
        by_cat[cat]["n"] += 1
        if r.get("novelty_pass"):
            by_cat[cat]["pass"] += 1
    out["by_category"] = by_cat
    return out
