"""Trajectory scoring utilities for V5 corpus collection and cross-attention training.

answer_score(answer, question) → float [0.0, 1.0]
    Scores how well a final answer addresses the question.
    Used by the trajectory reward function in V5_ARCHITECTURE.md §8.

trajectory_reward(traj, graph) → float [0.0, 1.0]
    Full trajectory reward: node selection quality + answer correctness.
    This is the primary training signal for the cross-attention LoRA adapters.

Design principles:
    - Fast enough to run on every corpus session (no GPU required at collection time).
    - Calibrated: 0.0 = completely wrong, 1.0 = perfect. Avoid saturating at extremes.
    - Pluggable: the LLM judge path is available for borderline cases.
    - Conservative: when uncertain, score lower rather than higher (avoids training
      on false positives).

See V5_ARCHITECTURE.md §8 for the full reward function specification.
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from graph_core import MemoryGraph

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Minimum answer length to be considered a real answer (not a refusal/error).
_MIN_ANSWER_TOKENS = 20

# Penalty applied when the answer contains known refusal / error phrases.
_REFUSAL_PHRASES = [
    "i don't know",
    "i cannot",
    "i do not have",
    "unable to answer",
    "no information",
    "not enough context",
    "i'm not sure",
    "cannot determine",
]

# Node types that qualify as good Layer-8 planning nodes (Strategy pass).
PLANNING_NODE_TYPES = {"strategy", "failure_pattern", "control_rule", "reasoning_chain"}

# Node types that qualify as good Layer-20 evidence nodes (Evidence pass).
EVIDENCE_NODE_TYPES = {"fact", "claim", "application", "solved_subgoal"}


# ---------------------------------------------------------------------------
# answer_score() — primary scoring function
# ---------------------------------------------------------------------------

def answer_score(
    answer: str,
    question: str,
    *,
    method: str = "heuristic",
    llm_judge_fn: Optional[Any] = None,
    threshold_for_judge: float = 0.45,
) -> float:
    """Score how well `answer` addresses `question`.

    Args:
        answer: The model's final polished answer string.
        question: The original question string.
        method: One of:
            "heuristic" — fast rule-based scoring (default, used during collection).
            "embedding"  — cosine similarity between question and answer embeddings.
            "llm"        — LLM-as-judge (expensive; use only for validation passes).
        llm_judge_fn: Optional callable(question, answer) → float. Required when
            method="llm" or when the heuristic score falls in the borderline range
            [threshold_for_judge - 0.1, threshold_for_judge + 0.1] and a judge fn
            is provided.
        threshold_for_judge: Heuristic scores in ±0.1 of this value trigger the
            LLM judge when llm_judge_fn is provided.

    Returns:
        Float in [0.0, 1.0]. Higher = better answer.
    """
    if method == "llm" and llm_judge_fn is not None:
        return _llm_score(question, answer, llm_judge_fn)
    if method == "embedding":
        return _embedding_score(question, answer)

    # Default: heuristic
    score = _heuristic_score(answer, question)

    # Escalate to judge for borderline scores if available
    if llm_judge_fn is not None:
        low = threshold_for_judge - 0.1
        high = threshold_for_judge + 0.1
        if low <= score <= high:
            judge_score = _llm_score(question, answer, llm_judge_fn)
            # Blend: 30% heuristic, 70% judge to smooth noise
            score = 0.3 * score + 0.7 * judge_score

    return round(max(0.0, min(1.0, score)), 4)


def _heuristic_score(answer: str, question: str) -> float:
    """Fast rule-based answer quality score."""
    if not answer or not answer.strip():
        return 0.0

    tokens = answer.split()
    score = 0.5  # neutral baseline

    # 1. Length signal: very short answers are usually incomplete
    if len(tokens) < _MIN_ANSWER_TOKENS:
        score -= 0.20
    elif len(tokens) >= 80:
        score += 0.10  # substantive answer bonus

    # 2. Refusal / uncertainty penalty
    answer_lower = answer.lower()
    for phrase in _REFUSAL_PHRASES:
        if phrase in answer_lower:
            score -= 0.30
            break

    # 3. Question keyword coverage: does the answer address the question's key terms?
    question_words = set(_content_words(question))
    answer_words = set(_content_words(answer))
    if question_words:
        overlap = len(question_words & answer_words) / len(question_words)
        # Overlap contributes up to ±0.20
        score += 0.20 * (overlap - 0.5)  # 0.5 overlap = neutral

    # 4. Structure signal: answers with paragraphs, lists, or headers are usually better
    if "\n" in answer or "- " in answer or "## " in answer or "* " in answer:
        score += 0.05

    # 5. Finalization proxy: does it include a clear concluding sentence?
    last_sentence = answer.strip().rsplit(".", 1)[-1].strip()
    if len(last_sentence.split()) >= 5:
        score += 0.05

    return score


def _content_words(text: str) -> List[str]:
    """Extract non-stopword lowercase tokens from text."""
    _STOP = {
        "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
        "have", "has", "had", "do", "does", "did", "will", "would", "could",
        "should", "may", "might", "shall", "can", "to", "of", "in", "on",
        "at", "by", "for", "with", "about", "as", "from", "that", "this",
        "it", "its", "and", "or", "but", "not", "no", "so", "if", "what",
        "how", "why", "when", "where", "which", "who",
    }
    return [w for w in re.findall(r"[a-z]+", text.lower()) if w not in _STOP and len(w) > 2]


def _embedding_score(question: str, answer: str) -> float:
    """Cosine similarity between question and answer using sentence-transformers."""
    try:
        from sentence_transformers import SentenceTransformer
        import numpy as np
        model = SentenceTransformer("all-MiniLM-L6-v2")
        q_emb, a_emb = model.encode([question, answer])
        cos = float(np.dot(q_emb, a_emb) / (np.linalg.norm(q_emb) * np.linalg.norm(a_emb) + 1e-9))
        # Map cosine [-1, 1] → score [0, 1], but cosine of (question, good answer)
        # is typically 0.3–0.8. We use a soft sigmoid to spread the range.
        return round(max(0.0, min(1.0, 0.5 + cos * 0.5)), 4)
    except ImportError:
        # Fall back to heuristic if sentence-transformers not available
        return _heuristic_score(answer, question)


def _llm_score(question: str, answer: str, judge_fn: Any) -> float:
    """Delegate to a user-supplied LLM judge function."""
    try:
        raw = judge_fn(question, answer)
        if isinstance(raw, (int, float)):
            return round(max(0.0, min(1.0, float(raw))), 4)
        # If judge returns a dict with a "score" key
        if isinstance(raw, dict):
            return round(max(0.0, min(1.0, float(raw.get("score", 0.5)))), 4)
    except Exception:
        pass
    return 0.5  # safe default on judge failure


# ---------------------------------------------------------------------------
# trajectory_reward() — full V5 training signal
# ---------------------------------------------------------------------------

def trajectory_reward(
    trajectory: Dict[str, Any],
    graph: "MemoryGraph",
    *,
    answer_score_fn=None,
    llm_judge_fn=None,
) -> float:
    """Compute the full trajectory reward for a training example.

    Args:
        trajectory: Dict with keys:
            "question": str
            "task_frame": dict (task_family, question_mode, required_slots)
            "layer8_node_ids": List[str]  — nodes selected at planning pass
            "layer20_node_ids": List[str] — nodes selected at evidence pass
            "final_answer": str
        graph: MemoryGraph to look up node types and metadata.
        answer_score_fn: Optional override for answer_score(). Signature:
            (answer: str, question: str) → float
        llm_judge_fn: Optional LLM judge. Forwarded to answer_score().

    Returns:
        Float in [0.0, 1.0].

    Reward components (from V5_ARCHITECTURE.md §8):
        - Answer correctness:  up to 1.0   (primary signal, 60% weight)
        - Layer 8 node quality: up to 0.30  (planning pass quality)
        - Layer 20 node quality: up to 0.40 (evidence pass quality)
        - Contradiction penalty: up to -0.30
        Final score is clipped to [0.0, 1.0].
    """
    question = trajectory.get("question", "")
    final_answer = trajectory.get("final_answer", "")
    task_frame = trajectory.get("task_frame", {})
    task_family = task_frame.get("task_family", "")
    layer8_ids = list(trajectory.get("layer8_node_ids", []))
    layer20_ids = list(trajectory.get("layer20_node_ids", []))

    # --- Component 1: Answer correctness (primary signal) ---
    if answer_score_fn is not None:
        a_score = answer_score_fn(final_answer, question)
    else:
        a_score = answer_score(final_answer, question, llm_judge_fn=llm_judge_fn)
    r = a_score * 0.60

    # --- Component 2: Layer 8 (planning pass) node quality ---
    for nid in layer8_ids:
        node = graph.nodes.get(nid)
        if node is None:
            continue
        if node.node_type in PLANNING_NODE_TYPES:
            # Bonus if the strategy/chain matches the current task family
            node_family = (node.context_guard or {}).get("task_family", "")
            if not node_family or node_family == task_family:
                r += 0.10  # matched planning node
            else:
                r += 0.03  # some credit for finding a planning node (wrong family)

    # Cap Layer 8 contribution at 0.30
    r = min(r, a_score * 0.60 + 0.30)

    # --- Component 3: Layer 20 (evidence pass) node quality ---
    for nid in layer20_ids:
        node = graph.nodes.get(nid)
        if node is None:
            continue
        if node.node_type in EVIDENCE_NODE_TYPES:
            # Bonus proportional to node confidence and access_count
            node_confidence = getattr(node, "confidence", 0.5)
            r += 0.08 * node_confidence

    # --- Component 4: Contradiction penalty ---
    all_ids = layer8_ids + layer20_ids
    for nid in all_ids:
        node = graph.nodes.get(nid)
        if node is None:
            continue
        # Check for contradiction edges from this node
        for edge in graph.edges:
            if edge.relation == "contradicts" and (edge.src == nid or edge.dst == nid):
                other_id = edge.dst if edge.src == nid else edge.src
                if other_id in set(all_ids):
                    # Two mutually contradicting nodes were both attended to
                    r -= 0.15

    return round(max(0.0, min(1.0, r)), 4)


# ---------------------------------------------------------------------------
# node_type_breakdown() — helper for corpus V2 schema
# ---------------------------------------------------------------------------

def node_type_breakdown(node_ids: List[str], graph: "MemoryGraph") -> Dict[str, List[str]]:
    """Group a list of node IDs by their node_type.

    Used by distillation_corpus.py to write the V5 trajectory fields.

    Returns:
        Dict mapping node_type string to list of node IDs of that type.
        Example: {"strategy": ["strat_abc"], "fact": ["fact_1", "fact_2"]}
    """
    result: Dict[str, List[str]] = {}
    for nid in node_ids:
        node = graph.nodes.get(nid)
        if node is None:
            result.setdefault("unknown", []).append(nid)
        else:
            result.setdefault(node.node_type, []).append(nid)
    return result
