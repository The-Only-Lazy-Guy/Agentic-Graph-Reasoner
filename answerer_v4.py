from __future__ import annotations

"""
answerer_v4.py -- Graph-grounded iterative reasoning.

Extends v3 with:
  1. Enforced <plan> on step 0; <replan> replaces plan any time thereafter
  2. Session objects (create_object / update_object / read_object / list_objects)
     for tracking multi-part state across steps
  3. CoT citation check -- model is warned when its reasoning lacks node citations
  4. Open-ended multi-step loop -- no fixed EXPLORE/CLOSE phases; model iterates
     until it emits <answer> or hits max_steps
  5. Budget visibility -- step N/max and state header injected every turn
"""

import json
import re
import time
import urllib.request
import urllib.error
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Set, Tuple

from graph_core import MemoryGraph
from anchor_retrieval import retrieve_anchors_v2
from reasoning.retrieval_boost import retrieve_with_failure_boost
from reasoning.activation import (
    ActivationConfig,
    GraphActivationTrace,
    GraphTaskFrame,
    evaluate_coverage,
    render_task_frame,
    run_graph_activation,
)
from reasoning.session_subgraph import SessionSubgraphController
from reasoning.consolidation import Consolidator
from reasoning.adaptive_planning import (
    AdaptivePlanTree,
    PlanCheckResult,
    PlanningBudgetExceeded,
)
from reasoning.dispatcher import Dispatcher
from reasoning.procedures.verify_algorithm_preconditions import build_seed_procedure
from reasoning.procedures.verify_nonneg_edges import build_verify_nonneg_edges
from reasoning.procedures.detect_negative_cycle import build_detect_negative_cycle
from reasoning.procedures.verify_shortest_path import build_verify_shortest_path
from reasoning.post_processing import (
    LearningReport,
    apply_graph_edits as _apply_graph_edits,
    extract_learning_report,
    produce_graph_edits,
)
from reasoning.scoped_edits import (
    approved_raw_edits_from_patches,
    patches_from_graph_edits,
    patches_to_dicts,
    summarize_patches,
    validate_patches,
)
from reasoning.reflection import ReflectionResult, run_reflection
from reasoning.graph_editor import edits_from_reflection, edits_from_reflection_v2, apply_edits as _apply_reflection_edits
from reasoning.distillation_corpus import append_session_to_corpus
from reasoning.heuristic_logger import HeuristicLogger
from reasoning.schemas import FailurePatternNode, Provenance
from reasoning.signals import Signal
from reasoning.meta import MetaContext, MetaPool
from reasoning.meta_procedures.budget_warner import build_budget_warner
from reasoning.semantic_dedupe import build_dedupe_index
from reasoning.edit_judge import judge_edits_batch
from reasoning.meta_procedures.tool_loop_cycle_detector import build_tool_loop_cycle_detector
from reasoning.meta_procedures.excessive_search_detector import build_excessive_search_detector
from reasoning.budgets import BudgetTracker, Budgets
from reasoning.lexical_matching import content_tokens, lexical_overlap, normalize_text
from reasoning.micro_controller import (
    MICRO_FINALIZE_SYSTEM_PROMPT,
    build_finalize_user_message,
    cheap_anchor_candidates,
    compose_answer_from_slots,
    deterministic_finalize_payload,
    finalize_evidence_node_ids,
    infer_task_family,
    propose_control_memory_edits,
    render_micro_context_block,
    run_micro_epistemic_controller,
)
from reasoning.signature_stats import (
    load_live_signature_bias_plan,
    run_signature_shadow_session,
)
from pathlib import Path

if TYPE_CHECKING:
    from reasoning.task_classifier import TaskClassifier


DEFAULT_LLAMA_SERVER_URL = "http://127.0.0.1:6767"

_TOOL_BLOCK_RE    = re.compile(r"<(?:tool|graph_action)>\s*(\{.*?\})\s*</(?:tool|graph_action)>", re.DOTALL)
_ANSWER_BLOCK_RE  = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)
_PLAN_BLOCK_RE    = re.compile(r"<plan>(.*?)</plan>", re.DOTALL)
_REPLAN_BLOCK_RE  = re.compile(r"<replan>(.*?)</replan>", re.DOTALL)
# Thinking models (Qwen3-Thinking, etc.) wrap internal monologue in <think>
_THINK_BLOCK_RE   = re.compile(r"<think>.*?</think>", re.DOTALL)


def _visible(text: str) -> str:
    """Strip <think>...</think> blocks -- structured tags only appear outside them."""
    return _THINK_BLOCK_RE.sub("", text)


def _action_tag_for_controller(controller: Any | None) -> str:
    if controller is not None and controller.__class__.__name__ == "V4OpencodeController":
        return "graph_action"
    return "tool"


def _render_action_protocol_prompt(base_prompt: str, *, action_tag: str) -> str:
    if action_tag == "tool":
        return base_prompt
    adapted = base_prompt.replace("<tool>", f"<{action_tag}>").replace("</tool>", f"</{action_tag}>")
    adapted = adapted.replace("GRAPH TOOLS", "GRAPH ACTIONS")
    adapted = adapted.replace("TOOL SYNTAX", "GRAPH ACTION SYNTAX")
    adapted = adapted.replace("Available tools:", "Available graph actions:")
    adapted = adapted.replace("Use tools to explore", "Use graph actions to explore")
    adapted = adapted.replace("do not call further graph or workspace tools", "do not emit further graph or workspace actions")
    adapted = adapted.replace("Multiple tool calls in one response are fine.", f"Multiple <{action_tag}> blocks in one response are fine.")
    opencode_note = (
        "OPENCODE BACKEND NOTE:\n"
        f"- Do NOT use any native opencode tool or function-calling interface.\n"
        f"- When you need a graph operation, emit a plain text <{action_tag}>{{...}}</{action_tag}> block.\n"
        "- The backend executes those graph actions locally and returns the results next turn.\n"
        "- If you call a native tool instead, the run will fail.\n"
    )
    return f"{opencode_note}\n{adapted}"


# ---------------------------------------------------------------------------
# Session data structures
# ---------------------------------------------------------------------------

@dataclass
class PlanSubgoal:
    text: str
    done: bool = False


@dataclass
class SessionObject:
    id: str
    name: str
    fields: List[str]
    state: Dict[str, Any]
    created_at_step: int


@dataclass
class FailureRecord:
    approach: str
    condition: str
    mechanism: str
    recorded_at_step: int


@dataclass
class DirectAnswerPlan:
    task_type: str
    anchor_ids: List[str]
    coverage: float
    reason: str


@dataclass
class V4Session:
    question: str
    anchors: List[str]             = field(default_factory=list)
    # Phase 4: hypotheses is now {hid: {"text", "verdict", "evidence"}} where
    # verdict is None until verify_hypotheses stamps it.
    hypotheses: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    plan: List[PlanSubgoal]        = field(default_factory=list)
    objects: Dict[str, SessionObject] = field(default_factory=dict)
    failures: List[FailureRecord]  = field(default_factory=list)
    step: int                      = 0
    planned: bool                  = False
    session_id: str                = ""
    activation: Optional[GraphActivationTrace] = None
    coverage_rounds: int           = 0
    coverage: Optional[Dict[str, Any]] = None
    design_evidence_gate_rounds: int = 0
    hypothesis_verify_prompted: bool = False    # Phase 4: have we asked the model to verify?
    read_grounding_prompted: bool = False       # have we asked the model to read nodes before answering?
    graph_only_answer: bool = False            # strict mode: answer from graph content only
    heuristic_logger: Optional[HeuristicLogger] = None
    controller: Optional[SessionSubgraphController] = None  # Phase 5: write-through audit + persist
    session_dir: Optional[str]     = None        # Phase 5: where the session was persisted
    meta_pool: Optional[MetaPool]  = None        # Phase 6: meta-procedure pool
    budget_tracker: Optional[BudgetTracker] = None  # Phase 6+7: budget accounting
    sticky_signals: List[Signal]   = field(default_factory=list)  # Phase 6: cap 20
    budget_exhausted_for: Optional[str] = None     # Phase 7: which budget axis hit cap
    plan_tree: Optional[AdaptivePlanTree] = None   # Phase 9: opt-in adaptive plan tree
    plan_tree_subgoal_ids: List[str] = field(default_factory=list)  # Phase 9: tree node ids by linear plan index
    dispatcher: Optional[Dispatcher] = None        # Phase 10: procedure dispatcher
    procedure_invocations: List[Dict[str, Any]] = field(default_factory=list)  # Phase 10
    _obj_counter: int              = field(default=0, repr=False)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _truncate(text: str, n: int) -> str:
    text = text or ""
    return text if len(text) <= n else text[: n - 1] + "…"


def _summarize(result: Dict[str, Any]) -> str:
    if "error" in result:
        return f"error: {result['error']}"
    parts = []
    for k, v in sorted(result.items()):
        if isinstance(v, list):
            parts.append(f"{k}=[{len(v)} items]")
        elif isinstance(v, str):
            parts.append(f"{k}={_truncate(v, 60)!r}")
        else:
            parts.append(f"{k}={v!r}")
    return "; ".join(parts)


_DIRECT_RELATIONSHIP_RE = re.compile(
    r"\b(relationship between|difference between|compare|contrast|how .* relate)\b",
    re.IGNORECASE,
)
_DIRECT_DESIGN_RE = re.compile(
    r"\b(design|architect|system|distributed|migration|migrate|pipeline|service|workflow|rollout|backend|frontend)\b",
    re.IGNORECASE,
)
_DIRECT_CODE_RE = re.compile(
    r"\b(implement|implementation|write code|function|class|template|pseudocode|refactor|build)\b",
    re.IGNORECASE,
)
_DIRECT_PROCEDURE_RE = re.compile(
    r"\b(before answering|precondition|preconditions|apply\s+verify|verify each|given graph|instance\b|edges?\s*\(|source\s+[A-Za-z0-9_]+)\b",
    re.IGNORECASE,
)
_DIRECT_JUDGMENT_RE = re.compile(r"^\s*(can|is|are|does|do|did|should|will|would|could)\b", re.IGNORECASE)
_DIRECT_LOOKUP_PREFIXES = (
    "what is",
    "what are",
    "why is",
    "why does",
    "when should",
    "when is",
    "explain",
    "describe",
)
_DIRECT_ELIGIBLE_TASK_TYPES = {
    "direct_judgment",
    "direct_lookup",
    "direct_relationship",
}
_DIRECT_MAX_ANCHOR_SCAN = 8
_DIRECT_MAX_READS = 3
_DIRECT_NODE_TYPE_PRIOR = {
    "strategy": 0.20,
    "claim": 0.18,
    "definition": 0.18,
    "explanation": 0.16,
    "concept": 0.14,
    "application": 0.13,
    "example": 0.11,
    "summary": 0.08,
    "bridge": 0.06,
}
_DIRECT_ANSWER_SYSTEM_PROMPT = """\
You are in FAST ANSWER mode.

The retrieval layer has already selected the small set of graph nodes that
most directly answers the user's question. Do NOT search, call tools,
create workspace objects, expand neighbors, or hypothesize.

Use only the evidence provided. If an evidence node includes a WARNING,
treat that warning as binding.

Output EXACTLY these blocks:

<reasoning>
2-5 concise sentences. You may cite node ids here.
</reasoning>

<answer>
Final user-facing answer. No node ids, no graph references.
</answer>

<explanation>
One short paragraph on how the answer was grounded. No node ids.
</explanation>
"""


def _infer_question_task_type(question: str) -> str:
    q = (question or "").strip()
    ql = q.lower()
    if not ql:
        return "analysis"
    if _DIRECT_PROCEDURE_RE.search(ql):
        return "procedure_or_instance"
    if _DIRECT_DESIGN_RE.search(ql) or _DIRECT_CODE_RE.search(ql):
        return "design_or_synthesis"
    if _DIRECT_RELATIONSHIP_RE.search(ql):
        return "direct_relationship"
    if _DIRECT_JUDGMENT_RE.match(ql):
        return "direct_judgment"
    if ql.startswith(_DIRECT_LOOKUP_PREFIXES):
        return "direct_lookup"
    return "analysis"


def _direct_node_usable(node: Any) -> bool:
    if node is None:
        return False
    meta = getattr(node, "metadata", {}) or {}
    if meta.get("deprecated"):
        return False
    conf = float(getattr(node, "confidence", 0.0) or 0.0)
    if conf < 0.15:
        return False
    node_id = str(getattr(node, "id", "") or "")
    if node_id.endswith("_false"):
        return False
    return True


def _merge_anchor_lists(*parts: List[str], limit: Optional[int] = None) -> List[str]:
    merged: List[str] = []
    seen: Set[str] = set()
    for part in parts:
        for raw in part:
            node_id = str(raw or "").strip()
            if not node_id or node_id in seen:
                continue
            seen.add(node_id)
            merged.append(node_id)
            if limit is not None and len(merged) >= limit:
                return merged
    return merged


def _strategy_key_node_ids(node: Any, graph: MemoryGraph) -> List[str]:
    meta = getattr(node, "metadata", {}) or {}
    if int(meta.get("strategy_schema_version", 0) or 0) < 2:
        return []
    key_ids = meta.get("key_node_ids", [])
    if not isinstance(key_ids, list):
        return []
    return [str(nid) for nid in key_ids if str(nid) in graph.nodes]


def _direct_node_score(question: str, node: Any, task_type: str) -> float:
    text = str(getattr(node, "text", "") or "")
    node_type = str(getattr(node, "node_type", "") or "").lower()
    q_tokens = content_tokens(question, min_chars=4)
    t_tokens = content_tokens(text, min_chars=4)
    coverage = 0.0 if not q_tokens else len(q_tokens & t_tokens) / max(len(q_tokens), 1)
    score = (
        0.80 * lexical_overlap(question, text, min_chars=3)
        + 0.35 * coverage
        + _DIRECT_NODE_TYPE_PRIOR.get(node_type, 0.0)
        + 0.15 * max(0.0, min(float(getattr(node, "confidence", 0.0) or 0.0), 1.0))
    )
    low = text.lower()
    if task_type == "direct_judgment" and any(
        cue in low for cue in ("requires", "cannot", "fails", "unsafe", "negative edge", "nonnegative")
    ):
        score += 0.08
    if task_type == "direct_relationship" and any(
        cue in low for cue in ("extends", "inherits", "relates", "connect", "difference", "compare")
    ):
        score += 0.08
    if node_type == "strategy":
        meta = getattr(node, "metadata", {}) or {}
        if int(meta.get("strategy_schema_version", 0) or 0) < 2:
            score -= 0.20
        else:
            task_family = str(meta.get("task_family", "") or "")
            if task_family and task_family in task_type:
                score += 0.08
        kws = {
            str(k).lower()
            for k in meta.get("domain_keywords", [])
            if isinstance(k, (str, int, float))
        }
        score += 0.03 * min(len(kws & q_tokens), 4)
    return score


def _direct_evidence_role(node: Any) -> str:
    text = str(getattr(node, "text", "") or "").lower()
    if any(cue in text for cue in ("counterexample", "fails", "unsafe", "breaks", "wrong shortest path", "misuse")):
        return "failure"
    if any(cue in text for cue in ("bellman-ford", "alternative", "instead", "safe choice", "recommended")):
        return "alternative"
    if any(cue in text for cue in ("requires", "must", "nonnegative", "only when", "correct when")):
        return "rule"
    return "core"


def _build_direct_evidence_record(
    graph: MemoryGraph,
    node_id: str,
    *,
    anon_fn: Optional[Any] = None,
) -> Optional[Dict[str, Any]]:
    node = graph.nodes.get(node_id)
    if node is None:
        return None
    out: Dict[str, Any] = {
        "id": anon_fn(node.id) if anon_fn is not None else node.id,
        "text": node.text,
        "node_type": node.node_type,
        "confidence": node.confidence,
    }
    if node.id.endswith("_false") or node.confidence < 0.1:
        out["WARNING"] = (
            "THIS IS A KNOWN MISCONCEPTION (confidence={:.2f}). "
            "Do NOT use it as supporting evidence."
        ).format(node.confidence)
    return out


def _select_direct_answer_plan(
    question: str,
    graph: MemoryGraph,
    anchors: List[str],
    *,
    max_reads: int = _DIRECT_MAX_READS,
) -> Optional[DirectAnswerPlan]:
    task_type = _infer_question_task_type(question)
    if task_type not in _DIRECT_ELIGIBLE_TASK_TYPES:
        return None

    q_tokens = content_tokens(question, min_chars=4)
    if len(q_tokens) > 24:
        return None

    scored: List[Tuple[float, str, str]] = []
    seen: Set[str] = set()
    for rank, nid in enumerate(anchors[:_DIRECT_MAX_ANCHOR_SCAN]):
        node = graph.nodes.get(nid)
        if node is None:
            continue
        if str(getattr(node, "node_type", "")).lower() == "strategy":
            for key_id in _strategy_key_node_ids(node, graph):
                if key_id in seen:
                    continue
                key_node = graph.nodes.get(key_id)
                if not _direct_node_usable(key_node):
                    continue
                scored.append((
                    _direct_node_score(question, key_node, task_type) + 0.20,
                    key_id,
                    f"strategy:{nid}",
                ))
                seen.add(key_id)
        if not _direct_node_usable(node):
            continue
        scored.append((
            _direct_node_score(question, node, task_type) - (0.02 * rank),
            nid,
            "anchor",
        ))
        seen.add(nid)

    scored.sort(key=lambda item: item[0], reverse=True)
    selected: List[str] = []
    details: List[str] = []
    if task_type == "direct_judgment":
        for wanted_role in ("rule", "failure", "alternative"):
            role_candidates: List[Tuple[float, str, str]] = []
            for score, nid, source in scored:
                if score < 0.22 or nid in selected:
                    continue
                node = graph.nodes.get(nid)
                if node is None or _direct_evidence_role(node) != wanted_role:
                    continue
                bonus = 0.0
                node_type = str(node.node_type).lower()
                if wanted_role == "failure" and node_type in {"application", "claim", "example"}:
                    bonus += 0.05
                if wanted_role == "alternative" and node_type == "claim":
                    bonus += 0.03
                role_candidates.append((score + bonus, nid, source))
            if role_candidates:
                role_candidates.sort(key=lambda item: item[0], reverse=True)
                best_score, best_nid, best_source = role_candidates[0]
                selected.append(best_nid)
                details.append(f"{best_nid}:{best_score:.2f}:{best_source}:{wanted_role}")
                if len(selected) >= max_reads:
                    break
    for score, nid, source in scored:
        if len(selected) >= max_reads:
            break
        if score < 0.22 or nid in selected:
            continue
        selected.append(nid)
        details.append(f"{nid}:{score:.2f}:{source}")
    if not selected:
        return None

    union_tokens: Set[str] = set()
    max_single_overlap = 0.0
    has_supporting_claim = False
    for nid in selected:
        node = graph.nodes.get(nid)
        if node is None:
            continue
        union_tokens |= content_tokens(node.text or "", min_chars=4)
        max_single_overlap = max(
            max_single_overlap,
            lexical_overlap(question, node.text or "", min_chars=3),
        )
        if str(node.node_type).lower() in {
            "claim", "definition", "explanation", "concept",
            "application", "example", "summary", "strategy",
        }:
            has_supporting_claim = True

    coverage = 0.0 if not q_tokens else len(union_tokens & q_tokens) / max(len(q_tokens), 1)
    if not has_supporting_claim:
        return None
    if coverage < 0.34 and max_single_overlap < 0.28:
        return None
    if task_type == "direct_judgment" and coverage < 0.40 and max_single_overlap < 0.32:
        return None

    return DirectAnswerPlan(
        task_type=task_type,
        anchor_ids=selected,
        coverage=coverage,
        reason=(
            f"task_type={task_type}; coverage={coverage:.2f}; "
            f"selected={', '.join(details[:3])}"
        ),
    )


def _render_direct_answer_user_message(
    question: str,
    task_type: str,
    evidence_records: List[Dict[str, Any]],
) -> str:
    task_note = {
        "direct_judgment": "Lead with a clear yes/no verdict. If the evidence names a safer alternative, include it.",
        "direct_relationship": "State the relationship explicitly, then explain it briefly.",
        "direct_lookup": "Answer directly and concisely from the evidence.",
    }.get(task_type, "Answer directly from the evidence.")
    lines = [
        f"Question:\n{question}",
        "",
        f"Detected task type: {task_type}",
        task_note,
        "The graph appears to already contain the direct answer. Use the evidence below and answer now.",
        "",
        "Evidence nodes:",
    ]
    for rec in evidence_records:
        conf = rec.get("confidence")
        conf_text = f"{float(conf):.2f}" if isinstance(conf, (int, float)) else "n/a"
        lines.append(f"### {rec.get('id')} [{rec.get('node_type', 'unknown')}] confidence={conf_text}")
        if rec.get("WARNING"):
            lines.append(f"WARNING: {rec['WARNING']}")
        lines.append(str(rec.get("text", "") or ""))
        lines.append("")
    return "\n".join(lines).strip()


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------

class V4Tools:
    def __init__(
        self,
        graph: MemoryGraph,
        session: V4Session,
        *,
        snippet_chars: int = 220,
        use_failure_boost: bool = True,
        llm_controller: Optional["V4LlamaServerController"] = None,
        anonymize_ids: bool = False,
    ):
        self.graph = graph
        self.session = session
        self.snippet_chars = snippet_chars
        self.use_failure_boost = use_failure_boost
        self.llm_controller = llm_controller
        self.anonymize_ids = anonymize_ids
        self.call_log: List[Dict[str, Any]] = []
        # ID anonymization: model sees "node_001" instead of "cpp_fenwick_template".
        # Bidirectional maps built lazily as nodes are encountered.
        self._real_to_anon: Dict[str, str] = {}
        self._anon_to_real: Dict[str, str] = {}
        self._anon_counter: int = 0
        # search_nodes dedupe: token-set + k -> result IDs. Jaccard sim against
        # past keys lets us catch semantic duplicates with reworded queries.
        # Each entry: (token_set, k, list_of_hit_ids, original_query)
        self._search_cache: List[Tuple[frozenset, int, List[str], str]] = []
        self._search_repeats: int = 0
        # Jaccard similarity threshold for "this is the same query as before".
        # 0.6 catches reworded duplicates without over-matching unrelated queries.
        self._search_jaccard_threshold: float = 0.6
        # Phase 5 write-through: map v4 obj id ("obj_1") to controller id ("so_xxx")
        self._v4_to_ctrl: Dict[str, str] = {}
        self._last_triggered_by: str = ""  # main loop sets this before tools fire

    def _rec(self, name: str, args: Dict[str, Any], result: Dict[str, Any]) -> None:
        self.call_log.append({"name": name, "args": args, "result_summary": _summarize(result)})

    # ── ID anonymization ────────────────────────────────────────────────────

    def _anon(self, real_id: str) -> str:
        """Map a real node ID to an anonymous one. Lazy -- assigns on first encounter."""
        if not self.anonymize_ids:
            return real_id
        if real_id not in self._real_to_anon:
            self._anon_counter += 1
            anon_id = f"node_{self._anon_counter:03d}"
            self._real_to_anon[real_id] = anon_id
            self._anon_to_real[anon_id] = real_id
        return self._real_to_anon[real_id]

    def _deanon(self, maybe_anon_id: str) -> str:
        """Map an anonymous ID back to real. When anonymization is on,
        REJECTS IDs not in the mapping -- prevents the model from bypassing
        anonymization by guessing real IDs from training knowledge."""
        if not self.anonymize_ids:
            return maybe_anon_id
        if maybe_anon_id in self._anon_to_real:
            return self._anon_to_real[maybe_anon_id]
        # Model sent a raw ID that's not in our mapping -- could be a
        # real ID the model guessed from training data. Block it.
        return f"__blocked_{maybe_anon_id}"

    def _blocked_error(self, attempted_id: str) -> Dict[str, Any]:
        """Error returned when the model tries an ID not in the anon mapping."""
        known = sorted(self._anon_to_real.keys())[:10]
        return {
            "error": (
                f"ID {attempted_id!r} is not recognized. Node IDs in this session are "
                f"anonymous (e.g., {', '.join(known[:3])}). Use ONLY IDs from: "
                f"(1) seed nodes in the first message, (2) search_nodes results, "
                f"(3) expand_neighbors results. Do NOT guess IDs from topic names."
            ),
        }

    def get_id_mapping(self) -> Dict[str, str]:
        """Return the anon→real mapping for provenance/debugging."""
        return dict(self._anon_to_real)

    # Tokens shorter than this or in this stoplist don't contribute to search-
    # dedupe Jaccard similarity. Short connectives swamp the union otherwise.
    _SEARCH_DEDUPE_STOPWORDS: Set[str] = frozenset({
        "the", "a", "an", "of", "for", "and", "or", "in", "on", "to", "by",
        "with", "is", "are", "be", "as", "via", "from", "this", "that",
        "n", "k", "log", "via",
    })

    def _search_tokens(self, query: str) -> frozenset:
        """Normalize a search query into a token set for Jaccard dedupe."""
        if not query:
            return frozenset()
        toks = re.findall(r"[A-Za-z][A-Za-z0-9_]+", query.lower())
        return frozenset(
            t for t in toks
            if len(t) >= 3 and t not in self._SEARCH_DEDUPE_STOPWORDS
        )

    @staticmethod
    def _jaccard(a: frozenset, b: frozenset) -> float:
        if not a or not b:
            return 0.0
        inter = len(a & b)
        if inter == 0:
            return 0.0
        return inter / len(a | b)

    # ── graph read ──────────────────────────────────────────────────────────

    def read_node(self, node_id: str) -> Dict[str, Any]:
        real_id = self._deanon(node_id)
        if self.anonymize_ids and real_id.startswith("__blocked_"):
            out = self._blocked_error(node_id)
            self._rec("read_node", {"node_id": node_id}, out)
            return out
        node = self.graph.nodes.get(real_id)
        if node is None and real_id in self.session.hypotheses:
            hyp = self.session.hypotheses[real_id]
            out: Dict[str, Any] = {
                "id": node_id,  # return the ID the model used (may be anon)
                "text": hyp["text"],
                "node_type": "hypothesis",
                "verdict": hyp.get("verdict"),
                "evidence": hyp.get("evidence"),
            }
        elif node is None:
            out = {"error": f"node {node_id!r} not found"}
        else:
            out = {"id": self._anon(node.id), "text": node.text, "node_type": node.node_type,
                   "confidence": node.confidence}
            if node.id.endswith("_false") or node.confidence < 0.1:
                out["WARNING"] = "THIS IS A KNOWN MISCONCEPTION (confidence={:.2f}). The claim above is FALSE. Do NOT cite it as evidence — cite the node that CONTRADICTS it instead.".format(node.confidence)
        self._rec("read_node", {"node_id": node_id}, out)
        return out

    def expand_neighbors(self, node_id: str, k: int = 5) -> Dict[str, Any]:
        real_id = self._deanon(node_id)
        if self.anonymize_ids and real_id.startswith("__blocked_"):
            out = self._blocked_error(node_id)
            self._rec("expand_neighbors", {"node_id": node_id, "k": k}, out)
            return out
        # Phase 7: budget -- each expand_neighbors is one graph hop.
        if self.session.budget_tracker is not None:
            try:
                self.session.budget_tracker.consume("hop", 1)
            except Exception as e:
                out = {"error": f"budget exhausted: {e}", "_budget_exhausted": "hop"}
                self._rec("expand_neighbors", {"node_id": node_id, "k": k}, out)
                return out
        if real_id not in self.graph.nodes:
            out = {"error": f"node {node_id!r} not found"}
            self._rec("expand_neighbors", {"node_id": node_id, "k": k}, out)
            return out
        neighbors: List[Dict[str, Any]] = []
        for e in self.graph.edges:
            other: Optional[str] = None
            if e.src == real_id:
                other = e.dst
            elif e.dst == real_id and not e.directed:
                other = e.src
            if other is None:
                continue
            n = self.graph.nodes.get(other)
            if n is None:
                continue
            neighbors.append({
                "id": self._anon(n.id),
                "relation": e.relation,
                "snippet": _truncate(n.text, self.snippet_chars),
            })
            if len(neighbors) >= k:
                break
        out = {"neighbors": neighbors}
        self._rec("expand_neighbors", {"node_id": node_id, "k": k}, out)
        return out

    def search_nodes(self, query: str, k: int = 5) -> Dict[str, Any]:
        new_tokens = self._search_tokens(query)
        # Semantic dedupe: find the highest-Jaccard prior search at this k.
        best_sim = 0.0
        best_entry: Optional[Tuple[frozenset, int, List[str], str]] = None
        for entry in self._search_cache:
            prev_tokens, prev_k, _, _ = entry
            if prev_k != k:
                continue
            sim = self._jaccard(new_tokens, prev_tokens)
            if sim > best_sim:
                best_sim = sim
                best_entry = entry

        # Log the dedupe decision
        if self.session.heuristic_logger:
            self.session.heuristic_logger.record(
                "search_jaccard_dedupe",
                features={"query": (query or "")[:60], "best_sim": round(best_sim, 3),
                           "k": k, "cache_size": len(self._search_cache)},
                decision="cache_hit" if (best_entry and best_sim >= self._search_jaccard_threshold) else "miss",
                threshold_used=self._search_jaccard_threshold,
            )
        if best_entry is not None and best_sim >= self._search_jaccard_threshold:
            _, _, cached_ids, prev_query = best_entry
            self._search_repeats += 1
            out = {
                "hits": [
                    {"id": self._anon(h), "snippet": _truncate(self.graph.nodes[h].text, self.snippet_chars)}
                    for h in cached_ids if h in self.graph.nodes
                ],
                "_warning": (
                    f"This query is semantically similar (Jaccard={best_sim:.2f}) to a "
                    f"previous one: {prev_query!r}. Returning cached results. "
                    "If the graph doesn't have what you're looking for, consider "
                    "record_failure(approach=..., condition=..., mechanism=...) "
                    "instead of continuing to search."
                ),
            }
            self._rec("search_nodes", {"query": query, "k": k, "_cached_jaccard": best_sim}, out)
            return out

        if self.use_failure_boost:
            hits = retrieve_with_failure_boost(query, self.graph, k=k)
        else:
            hits = retrieve_anchors_v2(query, self.graph, k=k, strategy="topk")
        self._search_cache.append((new_tokens, k, list(hits), query))
        out = {
            "hits": [
                {"id": h, "snippet": _truncate(self.graph.nodes[h].text, self.snippet_chars)}
                for h in hits if h in self.graph.nodes
            ]
        }
        self._rec("search_nodes", {"query": query, "k": k}, out)
        return out

    def hypothesize(self, text: str) -> Dict[str, Any]:
        text = (text or "").strip()
        if not text:
            out = {"error": "hypothesize requires non-empty text"}
            self._rec("hypothesize", {"text": text}, out)
            return out
        hid = f"h_{len(self.session.hypotheses) + 1}"
        self.session.hypotheses[hid] = {"text": text, "verdict": None, "evidence": None}
        if self.session.graph_only_answer:
            out = {"id": hid, "status": "gap_recorded",
                   "WARNING": "This is a GAP RECORD. This content MUST NOT appear in your <answer>. "
                   "Only graph content from read_node may appear in the answer."}
        else:
            out = {"id": hid, "status": "recorded",
                   "next_step": "before finalizing, verify or discard this hypothesis with verify_hypotheses"}
        self._rec("hypothesize", {"text": text}, out)
        return out

    def verify_hypotheses(self, verdicts: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Phase 4: stamp each hypothesis with verdict='verified'|'discarded' + evidence.

        verdicts: [{id: 'h_1', verdict: 'verified'|'discarded', evidence: 'why'}, ...]
        """
        if not isinstance(verdicts, list):
            out = {"error": "verdicts must be a list of {id, verdict, evidence} objects"}
            self._rec("verify_hypotheses", {"verdicts_type": type(verdicts).__name__}, out)
            return out
        results: List[Dict[str, Any]] = []
        for v in verdicts:
            if not isinstance(v, dict):
                results.append({"error": f"not an object: {v!r}"})
                continue
            hid = v.get("id")
            verdict = v.get("verdict")
            evidence = (v.get("evidence") or "").strip()
            if hid not in self.session.hypotheses:
                results.append({"id": hid, "error": "unknown hypothesis id"})
                continue
            if verdict not in ("verified", "discarded"):
                results.append({"id": hid, "error": "verdict must be 'verified' or 'discarded'"})
                continue
            if not evidence:
                results.append({"id": hid, "error": "evidence is required"})
                continue
            self.session.hypotheses[hid]["verdict"] = verdict
            self.session.hypotheses[hid]["evidence"] = evidence
            results.append({"id": hid, "status": "stamped", "verdict": verdict})
        unverified = [hid for hid, h in self.session.hypotheses.items() if h.get("verdict") is None]
        out = {"results": results, "remaining_unverified": unverified}
        self._rec("verify_hypotheses", {"count": len(verdicts)}, out)
        return out

    def list_anchors(self) -> Dict[str, Any]:
        out = {
            "anchors": [
                {"id": self._anon(a), "snippet": _truncate(self.graph.nodes[a].text, self.snippet_chars)}
                for a in self.session.anchors if a in self.graph.nodes
            ]
        }
        self._rec("list_anchors", {}, out)
        return out

    # ── plan ────────────────────────────────────────────────────────────────

    def mark_done(self, index: int) -> Dict[str, Any]:
        if not self.session.plan:
            out = {"error": "no plan exists yet"}
            self._rec("mark_done", {"index": index}, out)
            return out
        if not isinstance(index, int) or not (0 <= index < len(self.session.plan)):
            out = {"error": f"index {index!r} out of range [0, {len(self.session.plan) - 1}]"}
            self._rec("mark_done", {"index": index}, out)
            return out
        self.session.plan[index].done = True
        # Phase 9: also mark the corresponding tree node passed (when tree exists).
        if (
            self.session.plan_tree is not None
            and 0 <= index < len(self.session.plan_tree_subgoal_ids)
        ):
            try:
                tree_node_id = self.session.plan_tree_subgoal_ids[index]
                self.session.plan_tree.mark_passed(tree_node_id)
                # Activate next pending subgoal, if any.
                for next_id in self.session.plan_tree_subgoal_ids[index + 1:]:
                    next_node = self.session.plan_tree.nodes.get(next_id)
                    if next_node and next_node.status == "pending":
                        self.session.plan_tree._deactivate_current()
                        self.session.plan_tree.state.active_node_id = next_id
                        next_node.status = "active"
                        break
            except Exception:
                pass
        out = {"status": "done", "subgoal": self.session.plan[index].text}
        self._rec("mark_done", {"index": index}, out)
        return out

    # ── session objects ──────────────────────────────────────────────────────

    def create_object(self, name: str, fields: List[str], initial_state: Dict[str, Any]) -> Dict[str, Any]:
        if not name or not isinstance(name, str):
            out = {"error": "name must be a non-empty string"}
            self._rec("create_object", {"name": name}, out)
            return out
        if not isinstance(fields, list) or not all(isinstance(f, str) for f in fields):
            out = {"error": "fields must be a list of strings"}
            self._rec("create_object", {"name": name, "fields": fields}, out)
            return out
        if not isinstance(initial_state, dict):
            out = {"error": "initial_state must be an object"}
            self._rec("create_object", {"name": name}, out)
            return out
        # Phase 7: budget -- each create_object adds one node to session subgraph.
        if self.session.budget_tracker is not None:
            try:
                self.session.budget_tracker.consume("subgraph_size", 1)
            except Exception as e:
                out = {"error": f"budget exhausted: {e}", "_budget_exhausted": "subgraph_size"}
                self._rec("create_object", {"name": name}, out)
                return out
        self.session._obj_counter += 1
        obj_id = f"obj_{self.session._obj_counter}"
        state = {f: initial_state.get(f) for f in fields}
        obj = SessionObject(
            id=obj_id,
            name=name,
            fields=fields,
            state=state,
            created_at_step=self.session.step,
        )
        self.session.objects[obj_id] = obj
        # Phase 5: mirror to audit-logged subgraph controller
        if self.session.controller is not None:
            try:
                ctrl_id = self.session.controller.from_loose_object(
                    name=name,
                    fields=list(fields),
                    initial_state=state,
                    triggered_by=self._last_triggered_by or f"step {self.session.step}",
                )
                self._v4_to_ctrl[obj_id] = ctrl_id
            except Exception:
                pass  # never break the loop on audit failure
        out = {"id": obj_id, "name": name, "state": state}
        self._rec("create_object", {"name": name, "fields": fields}, out)
        return out

    def update_object(self, obj_id: str, field: str, value: Any) -> Dict[str, Any]:
        obj = self.session.objects.get(obj_id)
        if obj is None:
            out = {"error": f"object {obj_id!r} not found. Existing: {list(self.session.objects)}"}
            self._rec("update_object", {"obj_id": obj_id, "field": field}, out)
            return out
        if field not in obj.fields:
            out = {"error": f"field {field!r} not in schema {obj.fields}"}
            self._rec("update_object", {"obj_id": obj_id, "field": field}, out)
            return out
        obj.state[field] = value
        # Phase 5: mirror to audit-logged subgraph controller
        if self.session.controller is not None and obj_id in self._v4_to_ctrl:
            try:
                self.session.controller.update_object(
                    object_id=self._v4_to_ctrl[obj_id],
                    field_path=f"state.{field}",
                    new_value=value,
                    triggered_by=self._last_triggered_by or f"step {self.session.step}",
                )
            except Exception:
                pass
        out = {"id": obj_id, "field": field, "value": value, "status": "updated"}
        self._rec("update_object", {"obj_id": obj_id, "field": field, "value": value}, out)
        return out

    def read_object(self, obj_id: str) -> Dict[str, Any]:
        obj = self.session.objects.get(obj_id)
        if obj is None:
            out = {"error": f"object {obj_id!r} not found. Existing: {list(self.session.objects)}"}
            self._rec("read_object", {"obj_id": obj_id}, out)
            return out
        out = {"id": obj_id, "name": obj.name, "fields": obj.fields, "state": obj.state}
        self._rec("read_object", {"obj_id": obj_id}, out)
        return out

    def list_objects(self) -> Dict[str, Any]:
        out = {
            "objects": [
                {"id": o.id, "name": o.name, "state": o.state}
                for o in self.session.objects.values()
            ]
        }
        self._rec("list_objects", {}, out)
        return out

    # ── Phase 10: invoke_procedure JSON tool wrapping Phase-2A Dispatcher ───

    def invoke_procedure(self, name: str, args: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        if self.session.dispatcher is None or self.session.controller is None:
            out = {"error": "invoke_procedure: dispatcher not enabled (pass enable_procedures=True)"}
            self._rec("invoke_procedure", {"name": name}, out)
            return out
        if self.llm_controller is None:
            out = {"error": "invoke_procedure: no llm_controller available for sub-prompt"}
            self._rec("invoke_procedure", {"name": name}, out)
            return out

        # Build a one-shot llm_call(prompt) -> response_text closure over the
        # main controller. Each invocation is its own messages list -- the
        # procedure's body template carries all needed context.
        def llm_call(prompt: str) -> str:
            try:
                resp = self.llm_controller.chat([
                    {"role": "system", "content": "You are a procedure executor. Follow the prompt's instructions exactly and emit SET/ADD/DELETE mutation lines as directed."},
                    {"role": "user", "content": prompt},
                ])
                return resp["choices"][0]["message"]["content"]
            except Exception as e:
                return f"(llm_call error: {e})"

        try:
            outcome = self.session.dispatcher.invoke_by_name(
                name=name,
                args=args or {},
                session=self.session.controller,
                llm_call=llm_call,
                budget=self.session.budget_tracker,
            )
        except Exception as e:
            out = {"error": f"invoke_procedure crashed: {type(e).__name__}: {e}"}
            self._rec("invoke_procedure", {"name": name}, out)
            return out

        # Pull current object state for the response payload (if creation succeeded).
        obj_state: Optional[Dict[str, Any]] = None
        if outcome.object_id and outcome.object_id in self.session.controller.subgraph.nodes:
            node = self.session.controller.subgraph.nodes[outcome.object_id]
            if isinstance(node, dict):
                obj_state = node.get("state")

        out = {
            "procedure": name,
            "procedure_id": outcome.procedure_id,
            "object_id": outcome.object_id,
            "mutations_applied": outcome.mutations_applied,
            "state": obj_state,
            "error": outcome.error,
            "elapsed_sec": round(outcome.elapsed_seconds, 3),
        }
        self.session.procedure_invocations.append(out)
        self._rec("invoke_procedure", {"name": name, "args": args}, out)
        return out

    # ── Phase 9 plan-tree tools (no-op when plan_tree disabled) ─────────────

    def _require_plan_tree(self, op: str) -> Optional[Dict[str, Any]]:
        if self.session.plan_tree is None:
            return {"error": f"{op}: plan_tree is not enabled this session (pass enable_plan_tree=True)"}
        return None

    def plan_add_child(self, parent_id: str, goal: str, hypothesis: str = "", mode: str = "execute") -> Dict[str, Any]:
        err = self._require_plan_tree("plan_add_child")
        if err:
            self._rec("plan_add_child", {"parent_id": parent_id}, err); return err
        try:
            child_id = self.session.plan_tree.add_child(
                parent_id, goal=goal, hypothesis=hypothesis, mode=mode,
            )
            out = {"node_id": child_id, "status": "active"}
        except (PlanningBudgetExceeded, ValueError, KeyError) as e:
            out = {"error": f"plan_add_child failed: {e}"}
        self._rec("plan_add_child", {"parent_id": parent_id, "goal": goal[:60]}, out)
        return out

    def plan_record_check(
        self, node_id: str, passed: bool, failure_scope: str = "local_step",
        reason: str = "", failed_requirements: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        err = self._require_plan_tree("plan_record_check")
        if err:
            self._rec("plan_record_check", {"node_id": node_id}, err); return err
        try:
            check = PlanCheckResult(
                checked_node_id=node_id,
                passed=bool(passed),
                failure_scope=failure_scope,  # type: ignore
                reason=reason,
                failed_requirements=list(failed_requirements or []),
            )
            check_id = self.session.plan_tree.record_check(check)
            out = {"check_id": check_id, "passed": passed}
        except (KeyError, ValueError) as e:
            out = {"error": f"plan_record_check failed: {e}"}
        self._rec("plan_record_check", {"node_id": node_id, "passed": passed}, out)
        return out

    def plan_mark_passed(self, node_id: str) -> Dict[str, Any]:
        err = self._require_plan_tree("plan_mark_passed")
        if err:
            self._rec("plan_mark_passed", {"node_id": node_id}, err); return err
        try:
            self.session.plan_tree.mark_passed(node_id)
            out = {"node_id": node_id, "status": "passed"}
        except KeyError as e:
            out = {"error": str(e)}
        self._rec("plan_mark_passed", {"node_id": node_id}, out)
        return out

    def plan_mark_failed(self, node_id: str, reason: str = "") -> Dict[str, Any]:
        err = self._require_plan_tree("plan_mark_failed")
        if err:
            self._rec("plan_mark_failed", {"node_id": node_id}, err); return err
        try:
            self.session.plan_tree.mark_failed(node_id, reason)
            out = {"node_id": node_id, "status": "failed", "reason": reason}
        except KeyError as e:
            out = {"error": str(e)}
        self._rec("plan_mark_failed", {"node_id": node_id}, out)
        return out

    def plan_revise(
        self, failed_id: str, new_goal: str, new_hypothesis: str = "",
        mode: str = "execute", reason: str = "",
    ) -> Dict[str, Any]:
        err = self._require_plan_tree("plan_revise")
        if err:
            self._rec("plan_revise", {"failed_id": failed_id}, err); return err
        try:
            check = PlanCheckResult(
                checked_node_id=failed_id, passed=False,
                failure_scope="local_step", reason=reason or "model-judged failure",
            )
            new_id = self.session.plan_tree.revise_from_failure(
                failed_id, check, new_goal=new_goal, new_hypothesis=new_hypothesis, mode=mode,
            )
            out = {
                "new_active_node_id": new_id,
                "revision_count": self.session.plan_tree.state.revision_count,
            }
        except (PlanningBudgetExceeded, KeyError, ValueError) as e:
            out = {"error": str(e)}
        self._rec("plan_revise", {"failed_id": failed_id, "new_goal": new_goal[:60]}, out)
        return out

    def record_failure(self, approach: str, condition: str, mechanism: str) -> Dict[str, Any]:
        approach  = (approach  or "").strip()
        condition = (condition or "").strip()
        mechanism = (mechanism or "").strip()
        if not approach or not condition or not mechanism:
            out = {"error": "approach, condition, and mechanism are all required"}
            self._rec("record_failure", {}, out)
            return out
        # Phase 7: budget -- failure pattern adds one node to session subgraph.
        if self.session.budget_tracker is not None:
            try:
                self.session.budget_tracker.consume("subgraph_size", 1)
            except Exception as e:
                out = {"error": f"budget exhausted: {e}", "_budget_exhausted": "subgraph_size"}
                self._rec("record_failure", {"approach": approach[:80]}, out)
                return out
        rec = FailureRecord(
            approach=approach,
            condition=condition,
            mechanism=mechanism,
            recorded_at_step=self.session.step,
        )
        self.session.failures.append(rec)
        # Phase 5: mirror as a FailurePatternNode in the audit-logged subgraph
        if self.session.controller is not None:
            try:
                fp = FailurePatternNode(
                    id=f"fp_v4_{len(self.session.failures)}_{uuid.uuid4().hex[:8]}",
                    name=approach[:80],
                    attempted_approach=approach,
                    failure_condition=condition,
                    failure_mechanism=mechanism,
                    replacement=None,
                    example_failure_case=None,
                    provenance=Provenance(
                        created_in_session_id=self.session.controller.subgraph.session_id,
                        last_modified=datetime.now(timezone.utc).isoformat(),
                    ),
                )
                self.session.controller.add_failure_pattern(
                    fp, triggered_by=self._last_triggered_by or f"step {self.session.step}",
                )
            except Exception:
                pass
        out = {"status": "recorded", "failure_index": len(self.session.failures) - 1}
        self._rec("record_failure", {"approach": approach[:80]}, out)
        return out

    # ── web search ─────────────────────────────────────────────────────────

    def web_search(self, query: str, k: int = 3) -> Dict[str, Any]:
        """Search the web when the graph lacks coverage on a topic."""
        import urllib.request
        import urllib.parse
        try:
            from duckduckgo_search import DDGS
            with DDGS() as ddgs:
                raw = list(ddgs.text(query, max_results=k))
            results = [{"title": r.get("title", ""), "snippet": r.get("body", ""), "url": r.get("href", "")} for r in raw]
            out: Dict[str, Any] = {"results": results}
        except ImportError:
            # Fallback: DuckDuckGo Instant Answer API (limited but no dependency)
            try:
                url = "https://api.duckduckgo.com/?q={}&format=json&no_html=1&skip_disambig=1".format(
                    urllib.parse.quote(query))
                with urllib.request.urlopen(url, timeout=10) as resp:
                    data = json.loads(resp.read().decode("utf-8"))
                results = []
                if data.get("Abstract"):
                    results.append({"title": data.get("Heading", ""), "snippet": data["Abstract"],
                                    "url": data.get("AbstractURL", "")})
                for topic in data.get("RelatedTopics", [])[:k]:
                    if isinstance(topic, dict) and topic.get("Text"):
                        results.append({"title": "", "snippet": topic["Text"],
                                        "url": topic.get("FirstURL", "")})
                out = {"results": results[:k]} if results else {"results": [], "note": "No results. Try a more specific query."}
            except Exception as e:
                out = {"error": f"Web search failed: {str(e)[:100]}"}
        except Exception as e:
            out = {"error": f"Web search failed: {str(e)[:100]}"}
        self._rec("web_search", {"query": query, "k": k}, out)
        return out


TOOL_DISPATCH: Dict[str, Any] = {
    "read_node":        lambda t, a: t.read_node(**a),
    "expand_neighbors": lambda t, a: t.expand_neighbors(**a),
    "search_nodes":     lambda t, a: t.search_nodes(**a),
    "hypothesize":      lambda t, a: t.hypothesize(**a),
    "verify_hypotheses": lambda t, a: t.verify_hypotheses(**a),
    "list_anchors":     lambda t, a: t.list_anchors(),
    "mark_done":        lambda t, a: t.mark_done(**a),
    "create_object":    lambda t, a: t.create_object(**a),
    "update_object":    lambda t, a: t.update_object(**a),
    "read_object":      lambda t, a: t.read_object(**a),
    "list_objects":     lambda t, a: t.list_objects(),
    "record_failure":   lambda t, a: t.record_failure(**a),
    # Phase 9 plan-tree tools (no-ops unless enable_plan_tree=True)
    "plan_add_child":   lambda t, a: t.plan_add_child(**a),
    "plan_record_check": lambda t, a: t.plan_record_check(**a),
    "plan_mark_passed": lambda t, a: t.plan_mark_passed(**a),
    "plan_mark_failed": lambda t, a: t.plan_mark_failed(**a),
    "plan_revise":      lambda t, a: t.plan_revise(**a),
    # Phase 10 dispatcher tool (no-op unless enable_procedures=True)
    "invoke_procedure": lambda t, a: t.invoke_procedure(**a),
    # Web search
    "web_search":       lambda t, a: t.web_search(**a),
}


# ---------------------------------------------------------------------------
# Plan parsing
# ---------------------------------------------------------------------------

def _parse_plan_text(raw: str) -> List[PlanSubgoal]:
    subgoals: List[PlanSubgoal] = []
    for line in raw.strip().splitlines():
        line = line.strip()
        if not line:
            continue
        cleaned = re.sub(r"^[\d]+\.\s*|^[-*•]\s*", "", line).strip()
        if cleaned:
            subgoals.append(PlanSubgoal(text=cleaned))
    return subgoals


def parse_plan(text: str) -> Optional[List[PlanSubgoal]]:
    m = _PLAN_BLOCK_RE.search(_visible(text))
    return _parse_plan_text(m.group(1)) if m else None


def parse_replan(text: str) -> Optional[List[PlanSubgoal]]:
    m = _REPLAN_BLOCK_RE.search(_visible(text))
    return _parse_plan_text(m.group(1)) if m else None


# ---------------------------------------------------------------------------
# Tool call parsing and execution
# ---------------------------------------------------------------------------

def parse_tool_calls(text: str) -> List[Dict[str, Any]]:
    calls: List[Dict[str, Any]] = []
    for m in _TOOL_BLOCK_RE.finditer(_visible(text)):
        raw = m.group(1)
        try:
            obj = json.loads(raw)
        except json.JSONDecodeError:
            calls.append({"_parse_error": True, "raw": raw})
            continue
        name = obj.get("name")
        args = obj.get("args", {})
        if not isinstance(name, str) or not isinstance(args, dict):
            calls.append({"_shape_error": True, "raw": raw})
            continue
        calls.append({"name": name, "args": args})
    return calls


def parse_answer(text: str) -> Optional[str]:
    m = _ANSWER_BLOCK_RE.search(_visible(text))
    return m.group(1).strip() if m else None


def execute_tool(tools: V4Tools, call: Dict[str, Any]) -> Dict[str, Any]:
    if "_parse_error" in call:
        return {"error": f"tool call JSON did not parse: {call.get('raw', '')[:120]}"}
    if "_shape_error" in call:
        return {"error": f"tool call must have 'name' (string) and 'args' (object): {call.get('raw', '')[:120]}"}
    name = call["name"]
    fn = TOOL_DISPATCH.get(name)
    if fn is None:
        return {"error": f"unknown tool {name!r}. Available: {sorted(TOOL_DISPATCH)}"}
    try:
        return fn(tools, call["args"])
    except TypeError as e:
        return {"error": f"bad args for {name}: {e}"}
    except Exception as e:
        return {"error": f"{type(e).__name__} in {name}: {e}"}


# ---------------------------------------------------------------------------
# CoT citation check
# ---------------------------------------------------------------------------

def _strip_structured_tags(text: str) -> str:
    text = _visible(text)
    text = _TOOL_BLOCK_RE.sub("", text)
    text = _ANSWER_BLOCK_RE.sub("", text)
    text = _PLAN_BLOCK_RE.sub("", text)
    text = _REPLAN_BLOCK_RE.sub("", text)
    return text.strip()


def has_graph_citation(response: str, node_ids: Set[str]) -> bool:
    # Check both the visible portion and the think block -- model may cite nodes while thinking
    cot = _strip_structured_tags(response)
    if len(cot) < 200:
        return True  # too short to require citations
    for nid in node_ids:
        if nid in cot:
            return True
    return False


# ---------------------------------------------------------------------------
# Per-turn state header
# ---------------------------------------------------------------------------

def _render_state_header(session: V4Session, step: int, max_steps: int) -> str:
    remaining = max_steps - step
    pct = step / max_steps if max_steps > 0 else 0

    if pct >= 0.90:
        budget_note = f"  URGENT: {remaining} steps left -- finalize soon."
    elif pct >= 0.75:
        budget_note = f"  {remaining} steps left -- prioritize remaining subgoals."
    else:
        budget_note = ""

    lines = [f"[Step {step}/{max_steps} -- {remaining} remaining]{budget_note}"]

    if session.plan:
        lines.append("")
        lines.append("Plan:")
        for i, sg in enumerate(session.plan):
            mark = "x" if sg.done else " "
            lines.append(f"  [{mark}] {i + 1}. {sg.text}")
        # Stalled-plan nudge: if we're past step 5 and zero subgoals are done,
        # the model probably forgot to call mark_done.
        if step >= 5 and not any(sg.done for sg in session.plan):
            lines.append(
                "  ⚠ NOTE: no subgoals marked done after 5+ steps. If you've completed "
                "any of them, call mark_done(index=N) now."
            )

    if session.objects:
        lines.append("")
        lines.append("Active objects:")
        for obj in session.objects.values():
            lines.append(f"  - {obj.name} ({obj.id}): {json.dumps(obj.state, ensure_ascii=False)}")

    if session.failures:
        lines.append("")
        lines.append("Recorded failures:")
        for i, fr in enumerate(session.failures):
            lines.append(f"  {i + 1}. {fr.approach!r} -- condition: {fr.condition!r}")

    if session.plan_tree is not None:
        st = session.plan_tree.state
        active = session.plan_tree.nodes.get(st.active_node_id)
        active_label = f"{active.mode}: {active.goal[:60]!r}" if active else "(none)"
        lines.append("")
        lines.append(
            f"Plan tree: {len(session.plan_tree.nodes)} nodes, "
            f"revisions={st.revision_count}/{st.max_revisions}, "
            f"backtracks={st.backtrack_count}/{st.max_backtracks}, "
            f"finalized={st.finalized}"
        )
        lines.append(f"  Active node: `{st.active_node_id}` -- {active_label}")
        if st.last_failure_reason:
            lines.append(f"  Last failure: {st.last_failure_reason!r}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Phase 6: meta-procedure signal rendering
# ---------------------------------------------------------------------------

# Cap on sticky signal carry -- mirrors reasoning_loop.MAX_CARRIER_STICKY.
_V4_MAX_STICKY_SIGNALS = 20
# Cap on how many signals we surface to the model per turn.
_V4_MAX_SIGNALS_PER_TURN = 3

_SEV_ORDER = {"error": 0, "warn": 1, "info": 2}


def _render_signals_section(signals: List[Signal]) -> str:
    """Render top-severity signals as a '## Notes from your own process' block."""
    if not signals:
        return ""
    ranked = sorted(signals, key=lambda s: (_SEV_ORDER.get(s.severity, 9), s.id))[
        :_V4_MAX_SIGNALS_PER_TURN
    ]
    lines = ["## Notes from your own process"]
    lines.append("(Observations about your behavior this session. You decide whether to act on them.)")
    lines.append("")
    for s in ranked:
        lines.append(f"- [{s.severity.upper()}] {s.message}")
    return "\n".join(lines)


_POLISH_SYSTEM_PROMPT = """\
You are an editor producing the FINAL answer for an end user.

You will be given the user's question, a draft answer, and a brief reasoning
summary. Rewrite the draft as clean, polished prose that:

  - Does NOT reference any internal identifiers -- no `snake_case_ids`,
    no `xyz_apply` / `_concept` / `_bridge` style tokens, no backtick-
    wrapped node names.
  - Does NOT mention "the graph", "knowledge graph", "nodes", "edges",
    "anchors", "the substrate", "session", "tool calls", or any internal
    architecture concept.
  - Preserves ALL technical content, code blocks, algorithms, and
    correctness arguments from the draft.
  - Directly addresses the user's question.

Then produce a brief reasoning summary (1-2 short paragraphs) describing
HOW the answer was reached, in user-facing language. The reasoning summary
must obey the same restrictions (no internal identifiers, no graph talk).

Output format EXACTLY:

<answer>
Your polished answer here.
</answer>

<explanation>
1-2 paragraphs explaining the reasoning approach.
</explanation>
"""


def _strip_node_id_citations(text: str, node_ids: set) -> str:
    """Regex safety net: remove backtick-wrapped tokens that are real node ids,
    and clean up the boilerplate around them.

    Examples removed/cleaned:
      "According to `cpp_kadane_apply`:"     → ""
      "From `xxx` (type: fact):"              → ""
      "The graph node `xxx` shows ..."        → "..."
    Falls back to a token-only replacement when the surrounding clause
    can't be cleanly identified.
    """
    if not text or not node_ids:
        return text
    # Lead-in phrases that reference a node -- remove the whole clause up to
    # the colon. Conservative: only when the back-ticked token is a real id.
    def _drop_lead(match):
        ref = match.group("ref")
        return "" if ref in node_ids else match.group(0)
    text = re.sub(
        r"(?im)(?:According to|From|Per|Cited in|Node|Per node)\s+`(?P<ref>[A-Za-z_][\w]*)`[^:.\n]*[:.](?:\s+)?",
        _drop_lead,
        text,
    )
    # Bare backtick tokens -- strip backticks if the token is a real id.
    def _drop_token(match):
        tok = match.group(1)
        return "" if tok in node_ids else f"`{tok}`"
    text = re.sub(r"`([A-Za-z_][\w]*)`", _drop_token, text)
    # Clean up double spaces / orphan punctuation left behind.
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"  +", " ", text)
    return text.strip()


_POLISH_ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL | re.IGNORECASE)
_POLISH_EXPLANATION_RE = re.compile(r"<explanation>(.*?)</explanation>", re.DOTALL | re.IGNORECASE)


def polish_final_answer(
    *,
    question: str,
    raw_answer: str,
    session: V4Session,
    controller: "V4LlamaServerController",
    node_ids: set,
) -> Dict[str, str]:
    """One extra LLM call that rewrites the answer for the end user.

    Returns {"answer": str, "explanation": str}. On failure, returns the
    raw answer (with regex-stripped citations) and an empty explanation.
    """
    # Build a compact reasoning summary the polisher can lean on.
    summary_parts: List[str] = []
    if session.activation is not None and session.activation.task_frame.all_items():
        items = session.activation.task_frame.all_items()
        summary_parts.append("Key constraints addressed:")
        for it in items[:8]:
            summary_parts.append(f"  - ({it.kind}) {it.text}")
    verified = [h for h in session.hypotheses.values() if h.get("verdict") == "verified"]
    if verified:
        summary_parts.append("\nVerified findings:")
        for h in verified:
            summary_parts.append(f"  - {h.get('text','')} (evidence: {h.get('evidence','')[:120]})")
    if session.failures:
        summary_parts.append("\nApproaches ruled out:")
        for fr in session.failures:
            summary_parts.append(f"  - {fr.approach!r} fails when {fr.condition!r}: {fr.mechanism}")
    summary = "\n".join(summary_parts) or "(no structured summary)"

    user_msg = (
        f"User question:\n{question}\n\n"
        f"=== Draft answer (rewrite this) ===\n{raw_answer}\n\n"
        f"=== Reasoning summary (for your reference) ===\n{summary}\n\n"
        "Produce the polished <answer> and <explanation> now."
    )
    messages = [
        {"role": "system", "content": _POLISH_SYSTEM_PROMPT},
        {"role": "user", "content": user_msg},
    ]
    try:
        # Use chat_oneshot so stateful controllers (e.g., V4OpencodeController)
        # don't reuse the main-loop session -- the polish prompt needs its own
        # system message, not the v4 reasoning system prompt.
        chat_fn = getattr(controller, "chat_oneshot", controller.chat)
        resp = chat_fn(messages)
        content = resp["choices"][0]["message"]["content"]
    except Exception:
        # Fall back to regex-cleaned raw answer.
        return {
            "answer": _strip_node_id_citations(raw_answer, node_ids),
            "explanation": "",
        }

    am = _POLISH_ANSWER_RE.search(content)
    em = _POLISH_EXPLANATION_RE.search(content)
    polished = am.group(1).strip() if am else content.strip()
    explanation = em.group(1).strip() if em else ""

    # Regex safety net -- even if the polisher slipped, scrub remaining ids.
    polished = _strip_node_id_citations(polished, node_ids)
    explanation = _strip_node_id_citations(explanation, node_ids)
    return {"answer": polished, "explanation": explanation}


def validate_answer_grounding(
    answer: str,
    session: V4Session,
    tool_log: List[Dict[str, Any]],
) -> Optional[str]:
    """In graph_only_answer mode, verify the answer is grounded in graph reads.

    Returns an error message if validation fails, or None if the answer passes.
    """
    if not session.graph_only_answer:
        return None

    # Check 1: at least one read_node call must have happened
    read_count = sum(1 for e in tool_log if e.get("name") == "read_node")
    if read_count == 0:
        return (
            "Your answer must be grounded in graph content, but you never called "
            "read_node. Read at least one node before answering."
        )

    # Check 2: hypothesis text must not appear in the answer
    for hid, hyp in session.hypotheses.items():
        hyp_text = (hyp.get("text") or "").strip()
        if hyp_text and len(hyp_text) > 20 and hyp_text.lower() in answer.lower():
            return (
                f"Your answer contains hypothesis text ({hid}: {hyp_text[:60]!r}...) "
                f"which is a GAP RECORD, not a verified claim. Remove it from your "
                f"answer -- only content from read_node results may appear."
            )

    return None


def validate_answer_reads(
    answer: str,
    tool_log: List[Dict[str, Any]],
) -> Optional[str]:
    """Require at least one graph read before accepting a loop-mode answer."""
    _ = answer
    read_count = sum(1 for e in tool_log if e.get("name") == "read_node")
    if read_count == 0:
        return (
            "Before finalizing, read at least one graph node with read_node. "
            "Use the seed list or the recommended evidence nodes, then re-emit your <answer>."
        )
    return None


_DESIGN_EVIDENCE_HEDGE_CUES = (
    "one possible",
    "possible implementation",
    "possible option",
    "plausible",
    "could use",
    "might use",
    "may use",
    "hypothesis",
    "not directly supported",
    "if needed",
    "if we need",
    "depending on",
    "assume",
    "assuming",
    "for example",
)

_DESIGN_EVIDENCE_GROUPS: List[Dict[str, Any]] = [
    {
        "name": "tie_policy_detail",
        "triggers": ["timestamp", "secondary sort", "compound key", "monotonic counter", "tiebreaker"],
        "support_terms": ["timestamp", "secondary sort", "compound key", "tiebreaker", "tie policy"],
    },
    {
        "name": "distributed_infra_detail",
        "triggers": [
            "redis",
            "zadd",
            "zrank",
            "zrange",
            "sorted set",
            "consistent hashing",
            "shard",
            "replica",
            "replication",
            "coordinator",
            "router",
            "session stickiness",
            "read-your-writes",
            "eventual consistency",
            "wal",
            "write-ahead log",
        ],
        "support_terms": [
            "redis",
            "zadd",
            "zrank",
            "zrange",
            "sorted set",
            "consistent hashing",
            "shard",
            "replica",
            "replication",
            "coordinator",
            "router",
            "read-your-writes",
            "eventual consistency",
            "wal",
        ],
    },
    {
        "name": "atomic_fix_detail",
        "triggers": [
            "lua",
            "multi/exec",
            "compare-and-swap",
            "compare and swap",
            "cas",
            "mutex",
            "2-phase commit",
            "two-phase commit",
            "pending-transition bitmap",
            "write-lock",
            "write lock",
        ],
        "support_terms": [
            "lua",
            "multi/exec",
            "compare-and-swap",
            "compare and swap",
            "cas",
            "mutex",
            "2-phase commit",
            "two-phase commit",
            "bitmap",
            "write-lock",
            "write lock",
        ],
    },
    {
        "name": "quantitative_latency_detail",
        "triggers": [
            "microsecond",
            "microseconds",
            "~1-5ms",
            "~ 1-5ms",
            "~1–5ms",
            "~ 1–5ms",
            "<1ms",
            "< 1ms",
            "≈",
            "intra-dc",
            "intra dc",
            "50ns",
            "2.5μs",
            "2.5us",
        ],
        "support_terms": [
            "microsecond",
            "microseconds",
            "latency",
            "1-5ms",
            "50ns",
            "2.5us",
            "2.5μs",
            "propagate",
        ],
    },
]


def _answer_lines_outside_code(answer: str) -> List[str]:
    lines: List[str] = []
    in_code = False
    for raw in answer.splitlines():
        stripped = raw.strip()
        if stripped.startswith("```"):
            in_code = not in_code
            continue
        if in_code or not stripped:
            continue
        if stripped.startswith("#"):
            continue
        if stripped.startswith("- "):
            stripped = stripped[2:].strip()
        stripped = re.sub(r"`([^`]+)`", r"\1", stripped)
        stripped = re.sub(r"\s+", " ", stripped).strip()
        if stripped:
            lines.append(stripped)
    return lines


def _design_line_has_hedge(line: str) -> bool:
    low = line.lower()
    return any(cue in low for cue in _DESIGN_EVIDENCE_HEDGE_CUES)


def _read_node_ids_from_tool_log(tool_log: List[Dict[str, Any]]) -> List[str]:
    seen: Set[str] = set()
    ordered: List[str] = []
    for entry in tool_log:
        if entry.get("name") != "read_node":
            continue
        args = entry.get("args") if isinstance(entry.get("args"), dict) else {}
        node_id = str(args.get("node_id", "") or "")
        if node_id and node_id not in seen:
            seen.add(node_id)
            ordered.append(node_id)
    return ordered


def _build_design_support_corpus(
    *,
    graph: MemoryGraph,
    tool_log: List[Dict[str, Any]],
) -> str:
    parts: List[str] = []
    for node_id in _read_node_ids_from_tool_log(tool_log):
        node = graph.nodes.get(node_id)
        if node is None:
            continue
        parts.append(f"{node.id} {node.node_type} {node.text}")
    return normalize_text(" ".join(parts)).lower()


def _unsupported_design_answer_issues(
    answer: str,
    *,
    graph: MemoryGraph,
    tool_log: List[Dict[str, Any]],
) -> List[Dict[str, str]]:
    support_corpus = _build_design_support_corpus(graph=graph, tool_log=tool_log)
    if not support_corpus:
        return []
    issues: List[Dict[str, str]] = []
    seen_lines: Set[Tuple[str, str]] = set()
    for line in _answer_lines_outside_code(answer):
        low_line = normalize_text(line).lower()
        if _design_line_has_hedge(low_line):
            continue
        for group in _DESIGN_EVIDENCE_GROUPS:
            triggers = group["triggers"]
            if not any(term in low_line for term in triggers):
                continue
            if any(term in support_corpus for term in group["support_terms"]):
                continue
            key = (group["name"], line)
            if key in seen_lines:
                continue
            seen_lines.add(key)
            issues.append({
                "group": str(group["name"]),
                "line": line,
            })
    return issues


def _design_evidence_gate_message(issues: List[Dict[str, str]]) -> str:
    lines = [
        "Unsupported design details detected in your <answer>.",
        "Keep only implementation details grounded in graph nodes you actually read.",
        "For any unsupported detail, either:",
        "1. rewrite it as an explicit caveat/hypothesis",
        "2. or remove it",
        "",
        "Unsupported lines:",
    ]
    for issue in issues[:8]:
        lines.append(f"- ({issue['group']}) {issue['line']}")
    lines.extend([
        "",
        "Example rewrite:",
        '  "One possible implementation detail, not directly supported by the current graph, is Redis sharding..."',
        "",
        "Re-emit <answer>...</answer>.",
    ])
    return "\n".join(lines)


def _strip_unsupported_design_lines(answer: str, issues: List[Dict[str, str]]) -> str:
    blocked = {normalize_text(issue["line"]).lower() for issue in issues if issue.get("line")}
    if not blocked:
        return answer
    out_lines: List[str] = []
    in_code = False
    removed_any = False
    for raw in answer.splitlines():
        stripped = raw.strip()
        if stripped.startswith("```"):
            in_code = not in_code
            out_lines.append(raw)
            continue
        if not in_code:
            candidate = stripped
            if candidate.startswith("- "):
                candidate = candidate[2:].strip()
            candidate = re.sub(r"`([^`]+)`", r"\1", candidate)
            candidate = normalize_text(candidate).lower()
            if candidate and candidate in blocked:
                removed_any = True
                continue
        out_lines.append(raw)
    cleaned = "\n".join(out_lines).strip()
    if removed_any:
        cleaned = cleaned.rstrip() + "\n\nCaveat: some infrastructure-specific details were omitted because they were not directly supported by the current graph evidence."
    return cleaned


def validate_design_answer_support(
    answer: str,
    *,
    graph: MemoryGraph,
    tool_log: List[Dict[str, Any]],
    task_family: str,
) -> Optional[Dict[str, Any]]:
    if task_family != "design_synthesis":
        return None
    issues = _unsupported_design_answer_issues(answer, graph=graph, tool_log=tool_log)
    if not issues:
        return None
    return {
        "issues": issues,
        "message": _design_evidence_gate_message(issues),
    }


def _micro_recommended_finalize(outcome: Any) -> bool:
    if not getattr(outcome, "finalizable", False):
        return False
    steps = getattr(outcome, "micro_steps", []) or []
    if not steps:
        return False
    action = getattr(steps[-1], "action", "")
    action_value = getattr(action, "value", action)
    return str(action_value) == "FINALIZE"


def _seed_plan_tree(session: V4Session, question: str, subgoals: List[PlanSubgoal]) -> None:
    """Phase 9: build an AdaptivePlanTree from the question + linear subgoals.

    Root = question (mode='focus'), children = subgoals (mode='execute').
    Stores the per-subgoal tree node ids on session.plan_tree_subgoal_ids
    so the linear `mark_done(index)` tool can update the tree in lockstep.
    """
    tree = AdaptivePlanTree(
        session_id=session.session_id or "v4_session",
        root_goal=question,
        root_hypothesis="",
    )
    subgoal_ids: List[str] = []
    for sg in subgoals:
        try:
            cid = tree.add_child(
                tree.state.root_node_id,
                goal=sg.text,
                hypothesis="",
                mode="execute",
                activate=False,
            )
            subgoal_ids.append(cid)
        except (PlanningBudgetExceeded, ValueError):
            break
    # Activate the first subgoal so the model has a clear current focus.
    if subgoal_ids:
        try:
            tree._deactivate_current()
            tree.state.active_node_id = subgoal_ids[0]
            tree.nodes[subgoal_ids[0]].status = "active"
        except Exception:
            pass
    session.plan_tree = tree
    session.plan_tree_subgoal_ids = subgoal_ids


def _add_sticky(session: V4Session, new_signals: List[Signal]) -> None:
    """Append new sticky signals to carrier; cap and FIFO-evict."""
    existing_ids = {s.id for s in session.sticky_signals}
    for sig in new_signals:
        if sig.sticky and sig.id not in existing_ids:
            session.sticky_signals.append(sig)
            existing_ids.add(sig.id)
    if len(session.sticky_signals) > _V4_MAX_STICKY_SIGNALS:
        session.sticky_signals = session.sticky_signals[-_V4_MAX_STICKY_SIGNALS:]


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

V4_SYSTEM_PROMPT = """\
You are a graph-reasoning agent. The knowledge graph is your external memory.
You explore the graph with tools, accumulate evidence in session objects, and
ground claims in nodes you read.

When the graph does NOT cover the exact topic, DO NOT refuse to answer.
Instead:
  1. Search for RELATED or analogous concepts the graph does have.
  2. Read those related nodes -- they may contain transferable principles.
  3. Use hypothesize() for any claim you cannot directly ground in a node.
     Build your answer from graph evidence + informed hypotheses.
  4. NEVER say "the graph doesn't have this, I can't answer." Always
     reason from what IS available, even if it requires bridging.

The goal: every answer is grounded in the graph as much as possible,
extended by explicit hypotheses where the graph has gaps.

━━━ PLANNING (required, but proportional) ━━━

Your very first response MUST begin with a plan. Match plan size to question
complexity:

  Simple factual question -> 1-2 subgoals:
  <plan>
  1. Read the relevant node(s) and answer
  </plan>

  Complex design question -> more subgoals:
  <plan>
  1. Investigate candidate data structures
  2. Design the update algorithm
  3. Sketch distributed architecture
  </plan>

The plan is a live scratchpad -- update it freely at any step.
Mark subgoals done as you complete them:
  <tool>{"name": "mark_done", "args": {"index": 0}}</tool>

DO NOT over-engineer simple questions. If the graph already contains the
direct answer, read it and answer immediately. Do not create workspace
objects, hypothesize, or expand neighbors unless the question actually
requires multi-step reasoning.

━━━ GRAPH TOOLS ━━━

You start with 5 seed nodes. They are entry points, not the full answer.
Use tools to explore -- read full content, follow edges, search for what you need.

  read_node(node_id)            -- full text, type, confidence of a node
  expand_neighbors(node_id, k)  -- edges + connected nodes with snippets
  search_nodes(query, k)        -- semantic search across the whole graph
  list_anchors()                -- re-list your 5 seed nodes
  web_search(query, k)          -- search the web when graph lacks coverage

  <tool>{"name": "read_node", "args": {"node_id": "some_node_id"}}</tool>
  <tool>{"name": "expand_neighbors", "args": {"node_id": "some_node_id", "k": 5}}</tool>
  <tool>{"name": "search_nodes", "args": {"query": "O(log n) rank query sorted set", "k": 5}}</tool>
  <tool>{"name": "web_search", "args": {"query": "llama.cpp inference server API", "k": 3}}</tool>

Use web_search ONLY when graph search returns nothing relevant. The graph is
the primary source; the web fills gaps the graph does not cover yet.

━━━ SESSION OBJECTS ━━━

For multi-part tasks, create a workspace object to accumulate findings across steps.
It persists -- read it, update field by field, reference it in your answer.

  <tool>{"name": "create_object", "args": {
    "name": "analysis",
    "fields": ["data_structure", "justification", "open_questions"],
    "initial_state": {"data_structure": null, "justification": null, "open_questions": []}
  }}</tool>

  <tool>{"name": "update_object", "args": {
    "obj_id": "obj_1", "field": "data_structure", "value": "Fenwick tree"
  }}</tool>

  <tool>{"name": "read_object", "args": {"obj_id": "obj_1"}}</tool>
  <tool>{"name": "list_objects", "args": {}}</tool>

━━━ HYPOTHESES ━━━

Track ungrounded ideas explicitly -- do not assert them as facts:
  <tool>{"name": "hypothesize", "args": {"text": "Fenwick tree may not support range pagination directly"}}</tool>

EVERY hypothesis you record must be either verified or discarded before you
finalize. Use verify_hypotheses with concrete evidence (a node id you read,
or a short reason for discarding):
  <tool>{"name": "verify_hypotheses", "args": {"verdicts": [
    {"id": "h_1", "verdict": "verified", "evidence": "confirmed by `fenwick_pagination_apply`"},
    {"id": "h_2", "verdict": "discarded", "evidence": "no graph support; out of scope"}
  ]}}</tool>

If you try to emit <answer> with unverified hypotheses, the loop will block
and ask you to verify them first.

━━━ FAILURE RECORDING ━━━

When an approach fails, record it -- you will see it in the state header on future steps:
  <tool>{"name": "record_failure", "args": {
    "approach": "segment tree for per-user rank",
    "condition": "need position of individual element, not aggregate",
    "mechanism": "segment tree computes range aggregates, not element ranks"
  }}</tool>

━━━ CITATIONS ━━━

In your REASONING (private working memory), cite graph nodes so your
work is auditable:
  "According to `dijkstra_requires_nonnegative_edge_weights`: the algorithm
   requires nonnegative edge weights..."

In your <answer> (shown to the user), write naturally. Do NOT include
node IDs, parenthetical citations, or graph references. The answer should
read like a clear explanation, not an annotated bibliography.

The rule: read a node before asserting its content in reasoning. But
the final answer is clean prose grounded in what you discovered.

━━━ GRAPH TASK FRAME ━━━

If a `<graph_task_frame>` block appears in the first user message, its
constraints and pitfalls are binding requirements derived from the graph.
Your answer must address every constraint and avoid every pitfall.
If you intentionally do not address an item, state why explicitly.

━━━ PROCEDURES (opt-in) ━━━

When the dispatcher is enabled this session, you can invoke deterministic
verification procedures via:

  invoke_procedure(name, args)

Available procedures (use only when relevant to the question):
  - VerifyNonNegativeEdges(instance) -- check whether a graph has any negative edges
  - DetectNegativeCycle(instance)    -- check for negative cycles
  - VerifyShortestPath(instance, source, target) -- verify a claimed shortest-path solution
  - VerifyAlgorithmPreconditions(algorithm, instance) -- general precondition check

Each invocation creates a session object capturing the result. Cite the
result object's state in your answer.

━━━ ADAPTIVE PLAN TREE (opt-in) ━━━

If the state header mentions a "Plan tree:" line, the linear plan above is
also tracked as a checkpoint tree. You can use these tools to evolve it:

  plan_add_child(parent_id, goal, hypothesis, mode)
    modes: "focus" | "plan" | "execute" | "check" | "repair" | "finalize"

  plan_record_check(node_id, passed, failure_scope, reason, failed_requirements=[])
    failure_scope: "local_step" | "algorithm_choice" | "task_interpretation"

  plan_mark_passed(node_id)
  plan_mark_failed(node_id, reason)

  plan_revise(failed_id, new_goal, new_hypothesis, mode, reason)
    Creates a sibling branch after a failed check. Use this when an
    approach turns out wrong (e.g., picked Dijkstra but graph has negative
    edges) -- call plan_revise with the corrected approach.

The tree is OPT-IN. If no "Plan tree:" line appears in the state header,
ignore these tools and use the linear plan with mark_done(index).

━━━ META-OBSERVATIONS ━━━

You may occasionally receive a `## Notes from your own process` block in a
user message. These are observations about YOUR behavior in this session
(e.g., "you've created the same workspace 3 times" or "you've used 80% of
your budget"). They are not commands -- you decide whether to change course.

If you see a `<budget_exhausted budget="...">` tag, a hard cap was reached.
Finalize your answer with what you have; do not call further graph or
workspace tools.

━━━ TOOL SYNTAX ━━━

  <tool>{"name": "tool_name", "args": {...}}</tool>

Multiple tool calls in one response are fine. Results arrive in the next message.

━━━ FINISHING ━━━

When your plan is complete, write your final answer:

  <answer>
  Clear, natural explanation. No node IDs or graph references.
  </answer>

Example of a GOOD answer:
  "No. Dijkstra cannot be trusted with even a single negative edge.
   The algorithm's greedy settlement logic requires all edge weights
   to be nonnegative. A single negative edge breaks the invariant that
   a settled vertex has its final shortest distance."

Example of a BAD answer:
  "No (dijkstra_requires_nonnegative_edge_weights). The algorithm
   fails (negative_edge_counterexample_test_apply)."
  <-- node IDs leak into the answer; user does not need these

All reasoning, plans, tool calls, and objects are private working memory.
Only the <answer> block is shown to the user.
"""

# Graph-only preamble: replaces the first section of V4_SYSTEM_PROMPT when
# graph_only_answer=True. The rest of the prompt (tools, planning) stays.
_HYBRID_PREAMBLE = """\
You are a graph-reasoning agent. The knowledge graph is your external memory.
You explore the graph with tools, accumulate evidence in session objects, and
ground claims in nodes you read.

When the graph does NOT cover the exact topic, DO NOT refuse to answer.
Instead:
  1. Search for RELATED or analogous concepts the graph does have.
  2. Read those related nodes -- they may contain transferable principles.
  3. Use hypothesize() for any claim you cannot directly ground in a node.
     Build your answer from graph evidence + informed hypotheses.
  4. NEVER say "the graph doesn't have this, I can't answer." Always
     reason from what IS available, even if it requires bridging.

The goal: every answer is grounded in the graph as much as possible,
extended by explicit hypotheses where the graph has gaps."""

_GRAPH_ONLY_PREAMBLE = """\
You are a graph-reasoning agent with two strict roles:

NAVIGATOR -- your general knowledge helps you explore EFFICIENTLY:
  - Use what you know to decide what to search for
  - Pick smart search queries (e.g., you know "Fenwick" relates to "lowbit")
  - Choose which nodes to read first, which neighborhoods to expand
  - Plan your investigation using your background knowledge as a compass

SYNTHESIZER -- your reasoning AND answer come ONLY from graph content:
  - Every claim in reasoning AND <answer> must have an inline citation:
    "X is true (node_003). Y follows because Z (node_007)."
  - If the graph doesn't cover something, say "not available in the graph"
  - Do NOT fill gaps from your own knowledge
  - An answer without (node_NNN) citations will be REJECTED

hypothesize() records GAPS -- things the graph is missing. Hypotheses are
logged for future sessions to fill. They MUST NOT appear in your <answer>.

The graph is the SOURCE. Your knowledge is the COMPASS.
You are a researcher who can only cite from the library (graph), but uses
expertise to know which shelves to check first."""


def _select_system_prompt(graph_only: bool) -> str:
    """Swap the preamble section based on mode."""
    if not graph_only:
        return V4_SYSTEM_PROMPT
    return V4_SYSTEM_PROMPT.replace(_HYBRID_PREAMBLE, _GRAPH_ONLY_PREAMBLE)


def _build_first_user_message(
    question: str,
    anchors: List[str],
    graph: MemoryGraph,
    task_frame_block: str = "",
    micro_context_block: str = "",
    anon_fn: Optional[Any] = None,
    complexity: str = "",
    action_tag: str = "tool",
) -> str:
    """Build the opening user message: question + seed snippets + optional task frame."""
    _a = anon_fn or (lambda x: x)
    lines = []
    lines.append("[Step 0]")
    lines.append("")
    lines.append("## Question")
    lines.append(question)
    lines.append("")
    lines.append("## Seed nodes -- entry points into the graph")
    lines.append("Use read_node to get full content. Use expand_neighbors to follow edges.")
    lines.append("Use search_nodes if you need something not covered here.")
    if action_tag != "tool":
        lines.append(
            f"IMPORTANT: emit plain text <{action_tag}>{{...}}</{action_tag}> blocks for graph operations. "
            "Do NOT use any native opencode tool interface."
        )
    if anon_fn is not None:
        lines.append("NOTE: Node IDs are anonymous (node_001, node_002...). Use ONLY the IDs shown here or returned by tools. Do NOT guess IDs from topic names.")
    lines.append("")
    for a in anchors:
        n = graph.nodes.get(a)
        if not n:
            continue
        # Show only ID + type. Do NOT show text — force the model to call
        # read_node to get content. This ensures tool calls happen and
        # graph reads are logged for training data + graph edits.
        lines.append(f"  `{_a(a)}` [{n.node_type}]  (call read_node to see content)")
    lines.append("")
    if task_frame_block:
        lines.append("## Graph task frame")
        lines.append("Treat constraints and pitfalls below as binding. Address each item in your answer.")
        lines.append("")
        lines.append(task_frame_block)
        lines.append("")
    if micro_context_block:
        lines.append("## Micro controller state")
        lines.append("Treat this as the current subgoal ledger. Reuse filled slots before searching again.")
        lines.append("")
        lines.append(micro_context_block)
        lines.append("")
    lines.append("Begin with <plan>.")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Controller -- OpenAI-compatible chat completions (works with llama-server,
# opencode provider proxy, or any /v1/chat/completions endpoint)
# ---------------------------------------------------------------------------

@dataclass
class V4ControllerConfig:
    base_url: str      = DEFAULT_LLAMA_SERVER_URL
    temperature: float = 0.3
    max_tokens: int    = 8192
    timeout: float     = 600.0
    api_key: str       = "none"    # set to real key when using a cloud provider
    # llama.cpp only -- ignored when talking to a generic provider
    enable_thinking: bool = False
    llamacpp_mode: bool   = True   # False = generic OpenAI-compatible (opencode, etc.)


class V4LlamaServerController:
    def __init__(self, config: Optional[V4ControllerConfig] = None) -> None:
        self.config = config or V4ControllerConfig()
        self._checked = False
        if self.config.llamacpp_mode:
            self._guard_localhost(self.config.base_url)

    @staticmethod
    def _guard_localhost(url: str) -> None:
        from urllib.parse import urlparse
        host = (urlparse(url).hostname or "").lower()
        if host not in ("127.0.0.1", "localhost", "::1"):
            raise ValueError(f"Controller refuses non-localhost URL in llama.cpp mode: {url!r}")

    def _ensure_reachable(self) -> None:
        if self._checked:
            return
        if self.config.llamacpp_mode:
            url = self.config.base_url.rstrip("/") + "/health"
            try:
                with urllib.request.urlopen(url, timeout=5) as r:
                    r.read(64)
            except (urllib.error.URLError, OSError) as e:
                raise RuntimeError(f"llama-server unreachable at {self.config.base_url!r}: {e}")
        self._checked = True

    def chat(self, messages: List[Dict[str, str]]) -> Dict[str, Any]:
        self._ensure_reachable()
        body: Dict[str, Any] = {
            "messages": messages,
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
        }
        if self.config.llamacpp_mode:
            # llama.cpp-specific fields
            body["cache_prompt"] = True
            if self.config.enable_thinking:
                body["chat_template_kwargs"] = {"enable_thinking": True}
            else:
                body["chat_template_kwargs"] = {"enable_thinking": False}
                body["reasoning_effort"] = "none"

        headers: Dict[str, str] = {"Content-Type": "application/json"}
        if self.config.api_key and self.config.api_key != "none":
            headers["Authorization"] = f"Bearer {self.config.api_key}"

        payload = json.dumps(body).encode("utf-8")
        req = urllib.request.Request(
            self.config.base_url.rstrip("/") + "/v1/chat/completions",
            data=payload,
            headers=headers,
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=self.config.timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))

    def chat_oneshot(self, messages: List[Dict[str, str]]) -> Dict[str, Any]:
        """Stateless: just call chat. Llama-server has no per-controller session."""
        return self.chat(messages)


# ---------------------------------------------------------------------------
# Opencode controller -- runs via `opencode run` subprocess, maintains a
# session across steps so the provider sees the full conversation natively.
# ---------------------------------------------------------------------------

import subprocess as _subprocess
import shutil as _shutil
import sys as _sys
import os as _os
import copy as _copy

def _opencode_exe() -> str:
    """Find the opencode executable, accounting for Windows .cmd wrappers."""
    if "OPENCODE_EXE_PATH" in _os.environ:
        return _os.environ["OPENCODE_EXE_PATH"]
    for candidate in ("opencode", "opencode.cmd"):
        found = _shutil.which(candidate)
        if found:
            return found
    raise FileNotFoundError(
        "opencode not found on PATH. Install with: npm install -g opencode-ai"
    )

_OPENCODE_EXE = None  # resolved lazily


class V4OpencodeController:
    """
    Uses `opencode run --attach <url> --format json` to talk to any provider
    configured in opencode (Google, OpenAI, etc.).

    On the first chat() call the system prompt + first user message are sent
    together to bootstrap the session. On subsequent calls only the newest
    user message is forwarded -- the provider session already holds the prior
    assistant turns.
    """

    def __init__(
        self,
        model: str = "google/gemini-2.5-flash",
        server_url: Optional[str] = None,
        config_dir: Optional[str] = None,  # OPENCODE_CONFIG_DIR override (empty dir = no agents/AGENTS.md)
        variant: Optional[str] = None,
        timeout: float = 600.0,
        trace_mode_default: str = "chat",
        print_raw_output: bool = True,
    ) -> None:
        self.model = model
        self.server_url = server_url
        self.config_dir = config_dir
        self.variant = variant
        self.timeout = timeout
        self.trace_mode_default = trace_mode_default
        self.print_raw_output = print_raw_output
        self._session_id: Optional[str] = None
        self._sent_count: int = 0
        self._system_prompt: Optional[str] = None
        self._raw_trace: List[Dict[str, Any]] = []
        self._call_counter: int = 0

    def _merge_child_trace(self, entries: List[Dict[str, Any]]) -> None:
        for entry in entries:
            item = _copy.deepcopy(entry)
            self._call_counter += 1
            item["call_index"] = self._call_counter
            self._raw_trace.append(item)

    def get_raw_trace(self) -> List[Dict[str, Any]]:
        return _copy.deepcopy(self._raw_trace)

    def _print_opencode_raw_output(
        self,
        *,
        stdout: str,
        stderr: str,
        cmd: List[str],
        mode: str,
        returncode: int,
    ) -> None:
        """Print OpenCode subprocess output exactly as received.

        This intentionally does NOT truncate stdout or stderr. Debug output is
        written to stderr so normal program stdout can remain reserved for the
        final answer / CLI contract.
        """
        if not self.print_raw_output:
            return

        _sys.stderr.write("\n===== OPENCODE RAW OUTPUT BEGIN =====\n")
        _sys.stderr.write(f"mode={mode} returncode={returncode}\n")
        _sys.stderr.write("cmd=" + " ".join(cmd) + "\n")

        _sys.stderr.write("----- STDOUT BEGIN -----\n")
        _sys.stderr.write(stdout or "")
        if stdout and not stdout.endswith("\n"):
            _sys.stderr.write("\n")
        _sys.stderr.write("----- STDOUT END -----\n")

        _sys.stderr.write("----- STDERR BEGIN -----\n")
        _sys.stderr.write(stderr or "")
        if stderr and not stderr.endswith("\n"):
            _sys.stderr.write("\n")
        _sys.stderr.write("----- STDERR END -----\n")
        _sys.stderr.write("===== OPENCODE RAW OUTPUT END =====\n")
        _sys.stderr.flush()

    def _run_opencode(self, message: str, *, mode: Optional[str] = None) -> str:
        """Submit one user message to the opencode session, return assistant text."""
        global _OPENCODE_EXE
        if _OPENCODE_EXE is None:
            _OPENCODE_EXE = _opencode_exe()

        mode = mode or self.trace_mode_default
        started_at = datetime.now(timezone.utc).isoformat()
        started_ts = time.time()
        session_id_before = self._session_id
        cmd = [
            _OPENCODE_EXE, "run",
            "--model", self.model,
            "--format", "json",
        ]
        if self.server_url:
            cmd += ["--attach", self.server_url]
        if self._session_id:
            cmd += ["--session", self._session_id, "--continue"]
        if self.variant:
            cmd += ["--variant", self.variant]
        # Message via stdin -- avoids Windows cmd.exe 8KB arg-length limit

        # Windows can execute the resolved .cmd wrapper directly with argv.
        # Avoid shell=True here: list-argv + shell=True is fragile on Windows
        # and caused the opencode wrapper path to hang even on tiny prompts.
        use_shell = False
        env = dict(_os.environ)
        if self.config_dir:
            env["OPENCODE_CONFIG_DIR"] = self.config_dir
        proc = _subprocess.run(
            cmd,
            input=message,
            capture_output=True,
            text=True,
            timeout=self.timeout,
            encoding="utf-8",
            errors="replace",
            shell=use_shell,
            env=env,
        )
        ended_at = datetime.now(timezone.utc).isoformat()
        elapsed_sec = round(time.time() - started_ts, 3)

        self._print_opencode_raw_output(
            stdout=proc.stdout,
            stderr=proc.stderr,
            cmd=cmd,
            mode=mode,
            returncode=proc.returncode,
        )

        text_parts: List[str] = []
        session_id: Optional[str] = session_id_before
        events: List[Dict[str, Any]] = []

        for raw_line in proc.stdout.splitlines():
            raw_line = raw_line.strip()
            if not raw_line:
                continue
            try:
                event = json.loads(raw_line)
            except json.JSONDecodeError:
                continue
            if not isinstance(event, dict):
                continue
            events.append(event)

            # Capture session ID from any event (all carry it)
            session_id = event.get("sessionID") or session_id

            if event.get("type") == "text":
                part = event.get("part", {})
                text_parts.append(part.get("text", ""))

        if session_id:
            self._session_id = session_id

        assistant_text = "".join(text_parts)
        self._call_counter += 1
        self._raw_trace.append({
            "call_index": self._call_counter,
            "mode": mode,
            "session_id_before": session_id_before,
            "session_id_after": self._session_id,
            "model": self.model,
            "server_url": self.server_url,
            "variant": self.variant,
            "stdin_message": message,
            "raw_stdout": proc.stdout,
            "raw_stderr": proc.stderr,
            "events": events,
            "assistant_text": assistant_text,
            "returncode": proc.returncode,
            "started_at": started_at,
            "ended_at": ended_at,
            "elapsed_sec": elapsed_sec,
        })

        if not text_parts and proc.returncode != 0:
            raise RuntimeError(
                f"opencode run failed (exit {proc.returncode}):\n{proc.stderr}"
            )

        return assistant_text

    def chat(self, messages: List[Dict[str, str]]) -> Dict[str, Any]:
        """
        Accepts the full message history (OpenAI format).
        Sends only new user messages to the opencode session.
        System prompt is prepended to the very first user message.
        """
        # Extract system prompt once
        if self._system_prompt is None:
            for m in messages:
                if m["role"] == "system":
                    self._system_prompt = m["content"]
                    break

        new_messages = messages[self._sent_count:]
        self._sent_count = len(messages)

        assistant_text = ""
        for msg in new_messages:
            if msg["role"] != "user":
                # assistant turns are already stored in the provider session
                continue

            content = msg["content"]

            # Bootstrap: rules injected into the first user turn.
            # No custom tool protocol -- graph content is pre-expanded in the message,
            # so the model reasons over provided context without needing to call tools.
            if self._session_id is None and self._system_prompt:
                # Inject full system prompt as bootstrap (sent via stdin, no length limit)
                content = (
                    f"{self._system_prompt}\n"
                    "---\n"
                    f"{content}"
                )

            assistant_text = self._run_opencode(content, mode=self.trace_mode_default)

        return {"choices": [{"message": {"content": assistant_text}}]}

    def chat_oneshot(self, messages: List[Dict[str, str]]) -> Dict[str, Any]:
        """One-off conversation in a FRESH opencode session.

        Use this for sub-calls (e.g., Phase-12 answer polishing) that need their
        own system prompt and must not be polluted by, or pollute, the main
        reasoning session's state. A throwaway controller is constructed
        per-call so no shared state leaks.
        """
        sub = V4OpencodeController(
            model=self.model,
            server_url=self.server_url,
            config_dir=self.config_dir,
            variant=self.variant,
            timeout=self.timeout,
            trace_mode_default="chat_oneshot",
            print_raw_output=self.print_raw_output,
        )
        result = sub.chat(messages)
        self._merge_child_trace(sub.get_raw_trace())
        return result


# ---------------------------------------------------------------------------
# Gemini direct controller -- calls Google's generativelanguage API natively.
# Reads the API key from opencode's auth.json so no separate setup is needed.
# Supports proper system messages and full conversation history.
# ---------------------------------------------------------------------------

def _load_google_api_key() -> str:
    """Read the Google API key from opencode's auth store."""
    candidates = [
        _os.path.expanduser("~/.local/share/opencode/auth.json"),
        _os.path.expanduser("~\\AppData\\Roaming\\opencode\\auth.json"),
        _os.path.expanduser("~\\.local\\share\\opencode\\auth.json"),
    ]
    # Also try opencode's default config location
    xdg = _os.environ.get("XDG_DATA_HOME", "")
    if xdg:
        candidates.append(_os.path.join(xdg, "opencode", "auth.json"))

    for path in candidates:
        if _os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                data = json.loads(f.read())
            key = (data.get("google") or {}).get("key", "")
            if key:
                return key

    env_key = _os.environ.get("GOOGLE_API_KEY", "") or _os.environ.get("GEMINI_API_KEY", "")
    if env_key:
        return env_key

    raise RuntimeError(
        "Google API key not found. Expected in opencode auth.json under google.key, "
        "or set GOOGLE_API_KEY env var."
    )


class V4GeminiController:
    """
    Calls Google Gemini via the generativelanguage REST API directly.
    Uses the API key from opencode's auth.json -- no extra setup needed.

    Supports proper system_instruction so the model reliably follows our
    plan/tool/answer protocol without role confusion.
    """

    GEMINI_URL = (
        "https://generativelanguage.googleapis.com/v1beta/models"
        "/{model}:generateContent?key={key}"
    )

    def __init__(
        self,
        model: str = "gemini-2.5-flash",
        temperature: float = 0.4,
        max_output_tokens: int = 8192,
        timeout: float = 120.0,
        api_key: Optional[str] = None,
    ) -> None:
        self.model = model
        self.temperature = temperature
        self.max_output_tokens = max_output_tokens
        self.timeout = timeout
        self._api_key = api_key or _load_google_api_key()
        self._history: List[Dict[str, Any]] = []  # Gemini-format turns
        self._system_prompt: Optional[str] = None
        # Token accounting (accumulated across all steps)
        self.total_input_tokens: int = 0
        self.total_output_tokens: int = 0

    def chat(self, messages: List[Dict[str, str]]) -> Dict[str, Any]:
        """
        Accepts full OpenAI-format message list.
        Converts to Gemini format and maintains history server-side.
        """
        # Extract system prompt (Gemini handles it separately)
        if self._system_prompt is None:
            for m in messages:
                if m["role"] == "system":
                    self._system_prompt = m["content"]
                    break

        # Sync: find messages not yet in our history
        # Our _history has (user, model) pairs; messages has system+user+assistant interleaved
        non_system = [m for m in messages if m["role"] != "system"]
        already_sent = len(self._history)  # each entry is one turn

        new_turns = non_system[already_sent:]

        for turn in new_turns:
            role = "user" if turn["role"] == "user" else "model"
            self._history.append({
                "role": role,
                "parts": [{"text": turn["content"]}],
            })

        # Build request body
        body: Dict[str, Any] = {
            "contents": self._history,
            "generationConfig": {
                "temperature": self.temperature,
                "maxOutputTokens": self.max_output_tokens,
            },
        }
        if self._system_prompt:
            body["system_instruction"] = {
                "parts": [{"text": self._system_prompt}]
            }

        url = self.GEMINI_URL.format(model=self.model, key=self._api_key)
        payload = json.dumps(body).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        # Retry on 429 with exponential backoff (free tier = 10 RPM)
        import time as _time
        raw: Dict[str, Any] = {}
        for attempt in range(5):
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    raw = json.loads(resp.read().decode("utf-8"))
                break
            except urllib.error.HTTPError as e:
                if e.code in (429, 503) and attempt < 4:
                    wait = 15 * (2 ** attempt)  # 15s, 30s, 60s, 120s
                    print(f"  [Gemini {e.code}] retrying in {wait}s (attempt {attempt+1}/5)")
                    _time.sleep(wait)
                    continue
                raise

        # Extract text from response
        try:
            parts = raw["candidates"][0]["content"]["parts"]
            text = "".join(p.get("text", "") for p in parts)
        except (KeyError, IndexError) as e:
            raise RuntimeError(f"Unexpected Gemini response shape: {e}; got {raw!r}")

        # Accumulate token usage
        usage = raw.get("usageMetadata", {})
        self.total_input_tokens  += usage.get("promptTokenCount", 0)
        self.total_output_tokens += usage.get("candidatesTokenCount", 0)

        # Append assistant turn to history for next call
        self._history.append({
            "role": "model",
            "parts": [{"text": text}],
        })

        return {"choices": [{"message": {"content": text}}]}

    def chat_oneshot(self, messages: List[Dict[str, str]]) -> Dict[str, Any]:
        """Fresh sub-conversation: no shared history, no system-prompt lock-in."""
        sub = V4GeminiController(
            model=self.model,
            api_key=self._api_key,
            timeout=self.timeout,
        )
        return sub.chat(messages)


# ---------------------------------------------------------------------------
# Remote controller -- talks to a Cloudflare tunnel / remote opencode server
# via a simple POST /run endpoint. Config lives in E:/PROJECT/remote_config.json.
# ---------------------------------------------------------------------------

_REMOTE_CONFIG_PATH = Path("E:/PROJECT/remote_config.json")


def _load_remote_config() -> Dict[str, Any]:
    if _REMOTE_CONFIG_PATH.exists():
        return json.loads(_REMOTE_CONFIG_PATH.read_text(encoding="utf-8"))
    return {"endpoint": "http://localhost:4096/run", "timeout": 300}


V4_COMPACT_PROMPT = """\
You are a graph-reasoning agent. A knowledge graph is loaded and available \
through tools. You must read the graph before answering.

TOOL PROTOCOL:
Write <tool> blocks. The system executes them and returns results in the \
next message. You then continue. This repeats until you write <answer>.

Available tools:
  read_node(node_id)            -- read full content of a node
  search_nodes(query, k)        -- semantic search across the graph
  expand_neighbors(node_id, k)  -- follow edges from a node
  mark_done(index)              -- mark a plan subgoal as completed
  hypothesize(text)             -- record a gap (cannot appear in answer)
  record_failure(approach, condition, mechanism) -- log a failed approach
  create_object(name, fields, initial_state) -- workspace for multi-part tasks
  update_object(obj_id, field, value)        -- update workspace

Syntax: <tool>{"name": "read_node", "args": {"node_id": "node_001"}}</tool>

YOUR JOB:
1. Write a <plan> with subgoals tailored to the question.
2. Read seed nodes and explore the graph to gather evidence.
3. Reason about what you find. Search deeper if needed.
4. Write <answer> with inline citations: "X is true (node_001)."

RULES:
- The seed list shows IDs only. Call read_node to see content.
- Do NOT answer without reading nodes first.
- Your knowledge helps you decide WHERE to look. The graph is the SOURCE.
- EVERY claim in <answer> needs a (node_NNN) citation.
- Decide your own plan based on the question. Hard questions need more steps.
"""


class V4RemoteController:
    """Talks to a remote opencode instance via HTTP POST.

    The remote server accepts:
        POST /run  {"prompt": "..."}
    Returns:
        {"stdout": "...", "stderr": "...", "returncode": 0}

    Each chat() call sends the full conversation as a single prompt.
    Stateless -- every call is independent (no session continuity).
    """

    def __init__(
        self,
        endpoint: Optional[str] = None,
        timeout: float = 300,
        model_label: str = "remote/big-pickle",
    ):
        cfg = _load_remote_config()
        self.endpoint = endpoint or cfg.get("endpoint", "http://localhost:4096/run")
        self.timeout = timeout or cfg.get("timeout", 300)
        self.model = model_label

    def _call(self, prompt: str) -> str:
        """POST prompt to the remote endpoint, return stdout."""
        import requests
        resp = requests.post(
            self.endpoint,
            json={"prompt": prompt},
            timeout=self.timeout,
        )
        resp.raise_for_status()
        data = resp.json()
        text = data.get("stdout", "").strip()
        if not text and data.get("returncode", 1) != 0:
            raise RuntimeError(f"Remote call failed: {data.get('stderr', '')[:300]}")
        return text

    def chat(self, messages: List[Dict[str, str]]) -> Dict[str, Any]:
        """Pack the full conversation into a single self-contained prompt.

        Since the remote endpoint is stateless (opencode run), every call
        must include the complete context. We compress older turns to keep
        the prompt manageable while preserving the tool-call protocol.
        """
        system = ""
        turns: List[str] = []

        for m in messages:
            role = m.get("role", "user")
            content = m.get("content", "")
            if role == "system":
                system = content
            elif role == "user":
                turns.append(f"USER:\n{content}")
            elif role == "assistant":
                turns.append(f"ASSISTANT:\n{content}")

        # Build the prompt: system first, then conversation history,
        # then an explicit continuation instruction.
        parts = []
        if system:
            parts.append(system)
            parts.append("---")

        # Compress older turns: keep first 2 and last 4 in full,
        # summarize middle turns to save tokens.
        if len(turns) <= 6:
            for t in turns:
                parts.append(t)
        else:
            # First 2 turns (question + initial plan)
            for t in turns[:2]:
                parts.append(t)
            # Middle turns compressed
            mid = turns[2:-4]
            compressed = []
            for t in mid:
                # Keep only the first 200 chars of each middle turn
                lines = t.split("\n")
                role_line = lines[0] if lines else "TURN:"
                body = "\n".join(lines[1:])[:200]
                compressed.append(f"{role_line}\n{body}...")
            parts.append("--- PRIOR CONVERSATION (compressed) ---")
            parts.extend(compressed)
            parts.append("--- END COMPRESSED ---")
            # Last 4 turns in full (most recent context)
            for t in turns[-4:]:
                parts.append(t)

        parts.append("\n---\nContinue as ASSISTANT. Follow the tool-call protocol from the system prompt.")

        prompt = "\n\n".join(parts)
        text = self._call(prompt)
        return {"choices": [{"message": {"content": text}}]}

    def chat_oneshot(self, messages: List[Dict[str, str]]) -> Dict[str, Any]:
        """Stateless by design -- same as chat()."""
        return self.chat(messages)


# ---------------------------------------------------------------------------
# Result packet

@dataclass
class V4Packet:
    question: str
    answer: str
    task_type: str
    execution_mode: str
    steps: int
    max_steps: int
    tool_call_count: int
    tool_log: List[Dict[str, Any]]
    cot_log: List[str]
    plan: List[PlanSubgoal]
    objects: Dict[str, SessionObject]
    failures: List[FailureRecord]
    elapsed_sec: float
    finalized: bool
    anchors: List[str]
    citation_warnings: int
    search_repeats: int    # how many search_nodes calls hit the dedupe cache
    task_frame_items: int = 0     # Phase 2: number of items in the activation task frame
    activation_signals: int = 0   # Phase 2: number of GraphSignal objects emitted
    task_frame: Optional[Dict[str, Any]] = None         # Phase 2: structured task frame
    task_frame_rendered: str = ""                       # Phase 2: rendered <graph_task_frame> block
    coverage: Optional[Dict[str, Any]] = None   # Phase 3: final coverage report
    coverage_addressed_pct: float = 1.0          # Phase 3: % frame items addressed in final answer
    coverage_rounds: int = 0                     # Phase 3: number of self-judged revise rounds taken
    hypotheses: Dict[str, Dict[str, Any]] = field(default_factory=dict)  # Phase 4: full hypothesis records
    session_dir: Optional[str] = None    # Phase 5: where the session subgraph + audit log were persisted
    meta_signals: List[Dict[str, Any]] = field(default_factory=list)  # Phase 6: all emitted signals
    budget_summary: Optional[Dict[str, Any]] = None        # Phase 7: BudgetTracker.summary() snapshot
    consolidation_decisions: List[Dict[str, Any]] = field(default_factory=list)  # Phase 8: promotion decisions
    plan_tree_summary: Optional[Dict[str, Any]] = None     # Phase 9: AdaptivePlanTree.to_dict() if enabled
    procedure_invocations: List[Dict[str, Any]] = field(default_factory=list)  # Phase 10
    learning_report: Optional[Dict[str, Any]] = None    # Phase 11
    graph_edits: List[Dict[str, Any]] = field(default_factory=list)  # Phase 11
    graph_edits_applied: bool = False                   # Phase 11
    scoped_patches: List[Dict[str, Any]] = field(default_factory=list)  # Phase 11b
    scoped_patch_summary: Dict[str, Any] = field(default_factory=dict)  # Phase 11b
    answer_raw: str = ""                                # Phase 12: pre-polish answer
    explanation: str = ""                               # Phase 12: rationale paragraph
    polish_applied: bool = False                        # Phase 12
    shortcut_reason: str = ""                           # direct shortcut: why it triggered
    shortcut_anchor_ids: List[str] = field(default_factory=list)  # direct shortcut evidence reads
    controller_task_family: str = ""
    micro_steps: List[Dict[str, Any]] = field(default_factory=list)
    subgoal_reuse_count: int = 0
    slot_fill_stats: Dict[str, Any] = field(default_factory=dict)
    controller_action_counts: Dict[str, int] = field(default_factory=dict)
    controller_fallback_used: bool = False
    controller_raw_trace: List[Dict[str, Any]] = field(default_factory=list)
    controller_call_count: int = 0
    controller_total_elapsed_sec: float = 0.0
    controller_nonempty_turns: int = 0
    reflection: Optional[Dict[str, Any]] = None         # Phase 14: parsed ReflectionResult.to_dict()
    reflection_edits: List[Dict[str, Any]] = field(default_factory=list)  # Phase 14
    reflection_applied: bool = False                    # Phase 14
    signature_candidates: List[Dict[str, Any]] = field(default_factory=list)  # Phase 18
    signature_events: List[Dict[str, Any]] = field(default_factory=list)  # Phase 18
    signature_stats_update: Dict[str, Any] = field(default_factory=dict)  # Phase 18
    signature_shadow_report: Dict[str, Any] = field(default_factory=dict)  # Phase 18
    signature_graph_projection: Dict[str, Any] = field(default_factory=dict)  # Phase 18
    signature_live_bias: Dict[str, Any] = field(default_factory=dict)  # Phase 19
    # ---------------------------------------------------------------------------
    # V5 additions — required for cross-attention trajectory training corpus
    # ---------------------------------------------------------------------------
    # Ordered log of every graph node explicitly accessed during this session.
    # Each entry: {"node_id": str, "node_type": str, "step": int, "reason": str}
    # where "reason" is one of: "anchor_retrieval", "tool_read", "shortcut",
    # "neighbor_expand", "subgoal_lookup".
    # This is the primary trajectory data for Layer-8 / Layer-20 supervision.
    nodes_accessed_log: List[Dict[str, Any]] = field(default_factory=list)  # V5


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def answer_query_v4(
    *,
    question: str,
    graph: MemoryGraph,
    controller: Optional[V4LlamaServerController] = None,
    max_steps: int = 20,
    k_anchors: int = 5,
    use_failure_boost: bool = True,
    enable_activation: bool = True,
    graph_id: Optional[str] = None,
    enable_plan_tree: bool = False,
    enable_procedures: bool = False,
    apply_graph_edits: bool = False,
    polish_answer: bool = True,
    run_reflection_inline: bool = False,
    collect_corpus: bool = True,
    controller_label: str = "",
    auto_config: bool = False,
    classifier: Optional["TaskClassifier"] = None,
    anonymize_ids: bool = False,
    graph_only_answer: bool = False,
    enable_direct_shortcut: bool = False,
    enable_preloop_finalize: bool = False,
    enforce_recommended_finalize: bool = True,
    max_direct_reads: int = _DIRECT_MAX_READS,
    enable_signature_live_bias: bool = False,
    signature_stats_dir: str | Path = "data/signature_stats",
) -> V4Packet:
    controller = controller or V4LlamaServerController()
    _hlog = HeuristicLogger(f"v4_{uuid.uuid4().hex[:12]}")

    # Phase 17: auto-configure pipeline based on question complexity.
    _classified_level: Optional[str] = None
    if auto_config and classifier is not None:
        _classified_level, _conf, _cfg = classifier.classify(question)
        _hlog.record("task_classifier",
                     features={"question_len": len(question), "question_tokens": len(question.split())},
                     decision=_classified_level,
                     threshold_used={"score_cutoffs": [0.2, 0.6], "confidence": round(_conf, 3)},
                     alternatives={"max_steps": _cfg.max_steps, "activation": _cfg.enable_activation})
        # Apply classifier config, but let explicit caller overrides win.
        # We only override params that are still at their DEFAULTS.
        if max_steps == 20:  # caller didn't override
            max_steps = _cfg.max_steps
        if enable_activation is True:
            enable_activation = _cfg.enable_activation
        if polish_answer is True:
            polish_answer = _cfg.polish_answer
        if enable_plan_tree is False:
            enable_plan_tree = _cfg.enable_plan_tree
        if enable_procedures is False:
            enable_procedures = _cfg.enable_procedures
        if run_reflection_inline is False:
            run_reflection_inline = _cfg.run_reflection_inline

    task_type = _infer_question_task_type(question)
    signature_pre_task_family = infer_task_family(question)
    micro_outcome = None
    anchor_strategy = "full_retrieval"
    signature_live_bias_plan = None
    signature_live_bias: Dict[str, Any] = {
        "enabled": False,
        "reason": "feature_flag_disabled",
        "question": question,
        "task_family": signature_pre_task_family,
        "applied": False,
        "applied_anchor_ids": [],
    }
    if enable_signature_live_bias:
        try:
            signature_live_bias_plan = load_live_signature_bias_plan(
                question=question,
                task_family=signature_pre_task_family,
                graph_nodes=graph.nodes,
                stats_dir=signature_stats_dir,
                max_anchor_ids=max(k_anchors, _DIRECT_MAX_ANCHOR_SCAN),
            )
            signature_live_bias = {
                **signature_live_bias_plan.to_dict(),
                "applied": False,
                "applied_anchor_ids": [],
            }
        except Exception as exc:
            signature_live_bias = {
                "enabled": False,
                "reason": "live_bias_load_failed",
                "question": question,
                "task_family": signature_pre_task_family,
                "error": str(exc),
                "applied": False,
                "applied_anchor_ids": [],
            }

    t0 = time.time()
    if task_type in _DIRECT_ELIGIBLE_TASK_TYPES:
        cheap_anchors = cheap_anchor_candidates(
            question,
            graph,
            k=max(k_anchors, _DIRECT_MAX_ANCHOR_SCAN),
        )
        if signature_live_bias.get("enabled") and signature_live_bias.get("anchor_ids"):
            cheap_anchors = _merge_anchor_lists(
                list(signature_live_bias.get("anchor_ids", [])),
                list(cheap_anchors),
                limit=max(k_anchors, _DIRECT_MAX_ANCHOR_SCAN),
            )
        cheap_micro_outcome = run_micro_epistemic_controller(
            question=question,
            graph=graph,
            anchor_ids=list(cheap_anchors),
        )
        if cheap_micro_outcome.finalizable and cheap_micro_outcome.selected_node_ids:
            anchors = [str(a) for a in cheap_anchors[:k_anchors]]
            micro_outcome = cheap_micro_outcome
            anchor_strategy = (
                "signature_live_bias+cheap_lexical_reuse"
                if signature_live_bias.get("enabled") and signature_live_bias.get("anchor_ids")
                else "cheap_lexical_reuse"
            )
        else:
            if use_failure_boost:
                anchors = retrieve_with_failure_boost(question, graph, k=k_anchors)
                anchor_strategy = "failure_boost"
            else:
                anchors = retrieve_anchors_v2(question, graph, k=k_anchors, strategy="topk")
                anchor_strategy = "topk"
    else:
        if use_failure_boost:
            anchors = retrieve_with_failure_boost(question, graph, k=k_anchors)
            anchor_strategy = "failure_boost"
        else:
            anchors = retrieve_anchors_v2(question, graph, k=k_anchors, strategy="topk")
            anchor_strategy = "topk"
    if signature_live_bias.get("enabled") and signature_live_bias.get("anchor_ids"):
        anchors = _merge_anchor_lists(
            list(signature_live_bias.get("anchor_ids", [])),
            [str(a) for a in anchors],
            limit=k_anchors,
        )
        signature_live_bias["applied"] = True
        signature_live_bias["applied_anchor_ids"] = list(signature_live_bias.get("anchor_ids", []))
        if not anchor_strategy.startswith("signature_live_bias+"):
            anchor_strategy = f"signature_live_bias+{anchor_strategy}"
    anchors = [str(a) for a in anchors]
    if signature_live_bias.get("applied"):
        applied_set = set(anchors)
        signature_live_bias["applied_anchor_ids"] = [
            str(node_id)
            for node_id in signature_live_bias.get("anchor_ids", [])
            if str(node_id) in applied_set
        ]
        signature_live_bias["anchor_strategy"] = anchor_strategy

    session_id = _hlog.log.session_id
    session = V4Session(question=question, anchors=list(anchors), session_id=session_id,
                        graph_only_answer=graph_only_answer, heuristic_logger=_hlog)
    resolved_graph_id = graph_id or (graph.metadata.get("title", "graph") if graph.metadata else "graph")
    # Phase 5: write-through audit + persist
    session.controller = SessionSubgraphController(
        session_id=session_id,
        query=question,
        graph_id=resolved_graph_id,
    )
    # Phase 6+7: budget tracker + meta-procedure pool
    session.budget_tracker = BudgetTracker(Budgets(
        max_llm_calls=max_steps,
        max_hops=max_steps * 5,
        max_session_subgraph_size=max_steps * 4,
        max_total_tokens=max_steps * 2048,
    ))
    session.meta_pool = MetaPool()
    session.meta_pool.register(build_budget_warner())
    session.meta_pool.register(build_tool_loop_cycle_detector())
    session.meta_pool.register(build_excessive_search_detector())
    # Phase 10: optional procedure dispatcher (4 concrete procedures)
    if enable_procedures:
        proc_list = [
            build_seed_procedure(),
            build_verify_nonneg_edges(),
            build_detect_negative_cycle(),
            build_verify_shortest_path(),
        ]
        proc_index = {p.name.lower(): p for p in proc_list}
        session.dispatcher = Dispatcher(procedure_index=proc_index)
    tools = V4Tools(graph, session, use_failure_boost=use_failure_boost, llm_controller=controller, anonymize_ids=anonymize_ids)
    node_ids: Set[str] = set(graph.nodes.keys())

    if micro_outcome is None:
        micro_outcome = run_micro_epistemic_controller(
            question=question,
            graph=graph,
            anchor_ids=list(anchors),
        )
    controller_task_family = micro_outcome.task_family
    micro_context_block = render_micro_context_block(micro_outcome)
    _hlog.record(
        "task_shape_router",
        features={
            "question_len": len(question),
            "question_tokens": len(question.split()),
            "newlines": question.count("\n"),
            "question_marks": question.count("?"),
        },
        decision=task_type,
        threshold_used={"direct_max_question_tokens": 24},
    )
    _hlog.record(
        "micro_epistemic_controller",
        features={
            "task_family": controller_task_family,
            "anchor_count": len(anchors),
            "anchor_strategy": anchor_strategy,
            "selected_nodes": len(micro_outcome.selected_node_ids),
            "filled_slots": len(micro_outcome.slot_values),
            "micro_steps": len(micro_outcome.micro_steps),
        },
        decision="finalizable" if micro_outcome.finalizable else "fallback",
        threshold_used={
            "max_graph_queries": micro_outcome.task_frame.policy.max_graph_queries,
            "max_derivations": micro_outcome.task_frame.policy.max_derivations,
        },
    )

    cot_log: List[str] = []
    finalized = False
    final_answer: Optional[str] = None
    citation_warnings = 0
    plan_retries = 0
    execution_mode = "loop"
    shortcut_reason = ""
    shortcut_anchor_ids: List[str] = []
    shortcut_explanation = ""
    task_frame_block = ""
    messages: List[Dict[str, str]] = []
    controller_fallback_used = not micro_outcome.finalizable
    micro_shortcut_used = False
    action_tag = _action_tag_for_controller(controller)

    direct_plan = (
        _select_direct_answer_plan(
            question, graph, list(anchors), max_reads=max_direct_reads,
        )
        if enable_direct_shortcut else None
    )
    _hlog.record(
        "direct_answer_shortcut",
        features={
            "task_type": task_type,
            "anchor_count": len(anchors),
            "selected_reads": len(direct_plan.anchor_ids) if direct_plan else 0,
        },
        decision="hit" if direct_plan is not None else "miss",
        threshold_used={"max_reads": max_direct_reads, "max_anchor_scan": _DIRECT_MAX_ANCHOR_SCAN},
    )

    if enforce_recommended_finalize and _micro_recommended_finalize(micro_outcome):
        finalize_ids = finalize_evidence_node_ids(
            micro_outcome,
            max_nodes=max(2, min(max_direct_reads, 4)),
        )
        try:
            session.plan = [PlanSubgoal(text="Read controller-selected evidence and finalize", done=True)]
            session.planned = True
            for nid in finalize_ids:
                read_arg = tools._anon(nid) if anonymize_ids else nid
                tools.read_node(read_arg)
            if session.budget_tracker is not None:
                session.budget_tracker.consume("llm_call", 1)
            chat_fn = getattr(controller, "chat_oneshot", controller.chat)
            finalize_resp = chat_fn([
                {"role": "system", "content": MICRO_FINALIZE_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": build_finalize_user_message(
                        question,
                        micro_outcome,
                        graph,
                        evidence_node_ids=finalize_ids,
                    ),
                },
            ])
            finalize_content = finalize_resp["choices"][0]["message"]["content"]
            finalized_answer = parse_answer(finalize_content) or compose_answer_from_slots(micro_outcome)
            if finalized_answer:
                cot_log.append(finalize_content)
                final_answer = finalized_answer
                finalized = True
                execution_mode = "micro_controller_finalize"
                shortcut_reason = (
                    "recommended_action=FINALIZE; "
                    f"task_family={controller_task_family}; "
                    f"filled_slots={','.join(sorted(micro_outcome.slot_values.keys()))}"
                )
                shortcut_anchor_ids = list(finalize_ids)
                micro_shortcut_used = True
                controller_fallback_used = False
                em = _POLISH_EXPLANATION_RE.search(finalize_content)
                shortcut_explanation = em.group(1).strip() if em else ""
        except Exception:
            finalized = False
            final_answer = None
            cot_log = []
            citation_warnings = 0
            session.plan = []
            session.planned = False

    if (
        enable_preloop_finalize
        and
        micro_outcome.finalizable
        and micro_outcome.selected_node_ids
        and (
            micro_outcome.exact_answer_reuse_used
            or (
                micro_outcome.task_family == "algorithm_applicability"
                and not micro_outcome.strategy_assist_used
            )
        )
    ):
        try:
            direct_payload = deterministic_finalize_payload(question, micro_outcome, graph)
            direct_answer = direct_payload.get("answer", "").strip()
            if direct_answer:
                session.plan = [PlanSubgoal(text="Reuse solved subgoals and finalize", done=True)]
                session.planned = True
                for nid in micro_outcome.selected_node_ids:
                    read_arg = tools._anon(nid) if anonymize_ids else nid
                    tools.read_node(read_arg)
                direct_content = (
                    f"<reasoning>{direct_payload.get('reasoning', '').strip()}</reasoning>\n"
                    f"<answer>{direct_answer}</answer>\n"
                    f"<explanation>{direct_payload.get('explanation', '').strip()}</explanation>"
                )
                cot_log.append(direct_content)
                final_answer = direct_answer
                finalized = True
                execution_mode = "micro_controller_reuse"
                shortcut_reason = (
                    f"task_family={controller_task_family}; "
                    f"filled_slots={','.join(sorted(micro_outcome.slot_values.keys()))}"
                )
                shortcut_anchor_ids = list(micro_outcome.selected_node_ids)
                micro_shortcut_used = True
                controller_fallback_used = False
                shortcut_explanation = direct_payload.get("explanation", "").strip()
        except Exception:
            finalized = False
            final_answer = None
            cot_log = []
            citation_warnings = 0
            session.plan = []
            session.planned = False

    if enable_preloop_finalize and not finalized and direct_plan is not None:
        evidence_records = [
            _build_direct_evidence_record(
                graph, nid, anon_fn=tools._anon if anonymize_ids else None,
            )
            for nid in direct_plan.anchor_ids
        ]
        evidence_records = [rec for rec in evidence_records if rec is not None]
        if evidence_records:
            try:
                if session.budget_tracker is not None:
                    session.budget_tracker.consume("llm_call", 1)
                chat_fn = getattr(controller, "chat_oneshot", controller.chat)
                direct_resp = chat_fn([
                    {"role": "system", "content": _DIRECT_ANSWER_SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": _render_direct_answer_user_message(
                            question, direct_plan.task_type, evidence_records,
                        ),
                    },
                ])
                direct_content = direct_resp["choices"][0]["message"]["content"]
                direct_answer = parse_answer(direct_content)
                if direct_answer:
                    session.plan = [PlanSubgoal(text="Read the strongest direct evidence and answer", done=True)]
                    session.planned = True
                    for nid in direct_plan.anchor_ids:
                        read_arg = tools._anon(nid) if anonymize_ids else nid
                        tools.read_node(read_arg)
                    cot_log.append(direct_content)
                    final_answer = direct_answer
                    finalized = True
                    execution_mode = "direct_shortcut"
                    shortcut_reason = direct_plan.reason
                    shortcut_anchor_ids = list(direct_plan.anchor_ids)
                    controller_fallback_used = True
                    if (
                        not has_graph_citation(direct_content, node_ids)
                        and len(_strip_structured_tags(direct_content)) > 200
                    ):
                        citation_warnings += 1
                    em = _POLISH_EXPLANATION_RE.search(direct_content)
                    shortcut_explanation = em.group(1).strip() if em else ""
            except Exception:
                finalized = False
                final_answer = None
                cot_log = []
                citation_warnings = 0
                session.plan = []
                session.planned = False

    if not finalized:
        # Phase 2: build the activation task frame from anchors + nearby graph.
        if enable_activation and anchors:
            try:
                session.activation = run_graph_activation(
                    session_id=session_id,
                    graph_id=resolved_graph_id,
                    question=question,
                    graph=graph,
                    anchor_ids=list(anchors),
                    config=ActivationConfig(max_frame_chars=1800),
                )
                task_frame_block = render_task_frame(session.activation.task_frame)
            except Exception:
                # Activation should never block the loop. Log and continue without a frame.
                session.activation = None
                task_frame_block = ""

        messages = [
            {"role": "system",  "content": (
                _render_action_protocol_prompt(V4_COMPACT_PROMPT, action_tag=action_tag)
                if isinstance(controller, V4RemoteController)
                else _render_action_protocol_prompt(_select_system_prompt(graph_only_answer), action_tag=action_tag)
            )},
            {"role": "user",    "content": _build_first_user_message(
                question, anchors, graph, task_frame_block=task_frame_block,
                micro_context_block=micro_context_block,
                anon_fn=tools._anon if anonymize_ids else None,
                complexity=_classified_level or "",
                action_tag=action_tag,
            )},
        ]

    for step in range(max_steps if not finalized else 0):
        session.step = step
        # Phase 5: advance audit step counter alongside the v4 step counter.
        if session.controller is not None:
            session.controller.step()

        # Phase 6: pre_iter meta hook. Count an LLM call against the budget so
        # BudgetWarner has something to observe.
        pre_iter_signals: List[Signal] = []
        if session.meta_pool is not None and session.controller is not None and session.budget_tracker is not None:
            try:
                session.budget_tracker.consume("llm_call", 1)
            except Exception:
                pass
            try:
                ctx_pre = MetaContext.for_tool_loop(
                    session=session.controller,
                    budget=session.budget_tracker,
                    current_iteration=step,
                    raw_outputs=cot_log,
                    anchor_ids=session.anchors,
                    previous_signals=list(session.sticky_signals),
                    tool_call_log=tools.call_log,
                )
                pre_iter_signals = session.meta_pool.run_hook("pre_iter", ctx_pre)
            except Exception:
                pre_iter_signals = []

        resp_data = controller.chat(messages)
        try:
            content: str = resp_data["choices"][0]["message"]["content"]
        except (KeyError, IndexError) as e:
            raise RuntimeError(f"unexpected llama-server response: {e}; got {resp_data!r}")

        cot_log.append(content)
        messages.append({"role": "assistant", "content": content})

        # 1. Parse plan / replan (before tool execution so plan is enforced)
        if not session.planned:
            plan = parse_plan(content)
            if plan:
                session.plan = plan
                session.planned = True
                # Phase 9: seed the plan_tree from the linear subgoals when enabled.
                if enable_plan_tree:
                    _seed_plan_tree(session, question, plan)
            else:
                plan_retries += 1
                if plan_retries <= 2:
                    messages.append({
                        "role": "user",
                        "content": (
                            "Write a plan before calling any tools:\n\n"
                            "  <plan>\n  1. First subgoal\n  2. Second subgoal\n  </plan>"
                        ),
                    })
                    continue
                # Auto-stub after 2 retries; don't block forever
                session.plan = [PlanSubgoal(text="Explore the graph and answer the question")]
                session.planned = True
                if enable_plan_tree:
                    _seed_plan_tree(session, question, session.plan)
        else:
            replan = parse_replan(content)
            if replan:
                session.plan = replan
                if enable_plan_tree:
                    # Re-plan: replace existing tree with a fresh one from the new linear plan.
                    _seed_plan_tree(session, question, replan)

        # 2. CoT citation check
        cited = has_graph_citation(content, node_ids)
        cot_len = len(_strip_structured_tags(content))
        if _hlog:
            _hlog.record("citation_check",
                         features={"cot_length": cot_len, "cited": cited, "step": step},
                         decision="pass" if cited else "warn",
                         threshold_used=200)
        if not cited:
            citation_warnings += 1

        # 3. Execute tool calls (always, even when answer is also present in this response)
        # Phase 5: stamp triggered_by source for write-through audit entries.
        tools._last_triggered_by = _strip_structured_tags(content)[:200] or f"step {step}"
        tool_calls = parse_tool_calls(content)
        tool_result_parts: List[str] = []
        for call in tool_calls:
            result = execute_tool(tools, call)
            label = call.get("name", "<bad_call>")
            # Phase 7: catch budget exhaustion. After hitting a cap, the next
            # turn will instruct the model to finalize.
            if isinstance(result, dict) and "_budget_exhausted" in result:
                session.budget_exhausted_for = result["_budget_exhausted"]
            tool_result_parts.append(
                f"[{label}]\n{json.dumps(result, ensure_ascii=False, indent=2)}"
            )

        # 4. Check for final answer (after tools run so state mutations take effect)
        ans = parse_answer(content)
        if ans is not None:
            read_grounding_error = validate_answer_reads(ans, tools.call_log)
            if read_grounding_error:
                session.read_grounding_prompted = True
                guidance = read_grounding_error
                if micro_outcome.selected_node_ids:
                    guidance += "\nRecommended first reads: " + ", ".join(micro_outcome.selected_node_ids[:5])
                messages.append({"role": "user", "content": guidance})
                continue

            # Graph-only grounding validation: reject answers that contain
            # hypothesis text or have zero read_node calls.
            grounding_error = validate_answer_grounding(ans, session, tools.call_log)
            if grounding_error and not session.hypothesis_verify_prompted:
                session.hypothesis_verify_prompted = True  # reuse flag to avoid infinite loop
                messages.append({"role": "user", "content": grounding_error + "\nRe-emit your <answer> using only graph-read content."})
                continue

            # Phase 4: deterministic gate. If any hypothesis is still unverified,
            # block finalization until verify_hypotheses is called.
            unverified = [hid for hid, h in session.hypotheses.items() if h.get("verdict") is None]
            if unverified and not session.hypothesis_verify_prompted:
                session.hypothesis_verify_prompted = True
                hyp_lines = "\n".join(
                    f"  - {hid}: {session.hypotheses[hid]['text']}" for hid in unverified
                )
                verify_msg = (
                    "Before finalizing: the following hypotheses are still unverified. "
                    "For each, call verify_hypotheses with a verdict ('verified' or "
                    "'discarded') and concrete evidence (a node id or short reason). "
                    "Then re-emit your <answer>.\n\n"
                    f"Unverified hypotheses:\n{hyp_lines}\n\n"
                    f'Example: <{action_tag}>{{"name": "verify_hypotheses", "args": {{"verdicts": '
                    f'[{{"id": "h_1", "verdict": "verified", "evidence": "confirmed by '
                    f'node `xyz_apply`"}}]}}}}</{action_tag}>'
                )
                messages.append({"role": "user", "content": verify_msg})
                continue  # don't finalize; loop back for verification

            design_support_issue = validate_design_answer_support(
                ans,
                graph=graph,
                tool_log=tools.call_log,
                task_family=controller_task_family,
            )
            if design_support_issue is not None:
                if session.design_evidence_gate_rounds == 0 and step + 1 < max_steps:
                    session.design_evidence_gate_rounds += 1
                    messages.append({"role": "user", "content": design_support_issue["message"]})
                    continue
                ans = _strip_unsupported_design_lines(ans, design_support_issue["issues"])

            # Phase 3: surface coverage to the model once. Model self-judges whether
            # to revise. Cap at 1 round (budget enforcement, not heuristic gate).
            if (
                session.activation is not None
                and session.activation.task_frame.all_items()
                and session.coverage_rounds == 0
            ):
                cov = evaluate_coverage(session.activation.task_frame, ans)
                session.coverage = cov
                session.coverage_rounds += 1
                missed_ids = cov.get("missed_item_ids", [])
                pct = round(cov.get("coverage", 1.0) * 100)
                items_by_id = {item.item_id: item for item in session.activation.task_frame.all_items()}
                missed_texts = [
                    f"- ({items_by_id[mid].kind}) {items_by_id[mid].text}"
                    for mid in missed_ids if mid in items_by_id
                ]
                if missed_texts:
                    if step + 1 >= max_steps:
                        # Do not lose a valid final answer emitted on the last
                        # allowed turn. Coverage revision needs another model
                        # call, so at the budget boundary we accept the answer
                        # and preserve the coverage report for audit.
                        final_answer = ans
                        finalized = True
                        break
                    cov_msg = (
                        f"Coverage report on your <answer>: {pct}% of task-frame items addressed.\n"
                        f"Items not addressed:\n" + "\n".join(missed_texts) + "\n\n"
                        "If you intentionally chose not to address an item, state why in your "
                        "answer. Otherwise, revise and re-emit <answer>...</answer> with the "
                        "missing items covered. You get one revision."
                    )
                    messages.append({"role": "user", "content": cov_msg})
                    continue  # let the model decide; don't finalize yet
            final_answer = ans
            finalized = True
            break

        # 5. No answer: send tool results back (or nudge if no tools either)
        if not tool_calls:
            nudge = (
                "No tool calls and no <answer> in your response. "
                "Call a tool or emit <answer>...</answer>."
            )
            if not cited and len(_strip_structured_tags(content)) > 200:
                nudge += (
                    " Also: your reasoning made claims without citing any graph nodes -- "
                    "use read_node and reference nodes by ID."
                )
            messages.append({"role": "user", "content": nudge})
            continue

        # Phase 6: post_dispatch meta hook (after tools have run).
        post_dispatch_signals: List[Signal] = []
        if session.meta_pool is not None and session.controller is not None and session.budget_tracker is not None:
            try:
                ctx_post = MetaContext.for_tool_loop(
                    session=session.controller,
                    budget=session.budget_tracker,
                    current_iteration=step,
                    raw_outputs=cot_log,
                    anchor_ids=session.anchors,
                    previous_signals=list(session.sticky_signals),
                    tool_call_log=tools.call_log,
                )
                post_dispatch_signals = session.meta_pool.run_hook("post_dispatch", ctx_post)
            except Exception:
                post_dispatch_signals = []
        # Merge sticky carry-over + this turn's signals for next-message rendering.
        signals_this_turn = list(session.sticky_signals) + pre_iter_signals + post_dispatch_signals
        # Update sticky carrier with new sticky signals.
        _add_sticky(session, pre_iter_signals + post_dispatch_signals)

        next_step = step + 1
        state_hdr = _render_state_header(session, next_step, max_steps)
        result_parts: List[str] = [state_hdr, "", "Tool results:"] + tool_result_parts

        if not cited and len(_strip_structured_tags(content)) > 200:
            result_parts.append(
                "\nReminder: cite graph node IDs in your reasoning "
                "(e.g., 'According to `node_id`: ...')."
            )

        signals_block = _render_signals_section(signals_this_turn)
        if signals_block:
            result_parts.append(signals_block)

        # Phase 7: if a budget was exhausted this turn, tell the model to finalize.
        if session.budget_exhausted_for is not None:
            result_parts.append(
                f"<budget_exhausted budget=\"{session.budget_exhausted_for}\"/>\n"
                "A budget axis has been exhausted. Finalize your answer NOW "
                "with what you have. Do not call further graph or workspace tools."
            )

        messages.append({"role": "user", "content": "\n\n".join(result_parts)})

    elapsed = time.time() - t0

    if final_answer is None:
        fallback_answer = compose_answer_from_slots(micro_outcome).strip()
        final_answer = fallback_answer or "(no final answer -- max_steps reached)"

    if session.activation is not None:
        task_frame_items_count = len(session.activation.task_frame.all_items())
        activation_signals_count = len(session.activation.signals)
    else:
        task_frame_items_count = 0
        activation_signals_count = 0

    # Phase 5: persist the session subgraph + audit log.
    consolidation_decisions: List[Dict[str, Any]] = []
    if session.controller is not None:
        try:
            sess_dir = session.controller.close(Path("data/session_subgraphs"))
            session.session_dir = str(sess_dir)
            # Also persist the raw CoT log so reasoning can be inspected later.
            try:
                cot_path = Path(session.session_dir) / "cot_log.txt"
                lines = []
                for i, c in enumerate(cot_log):
                    lines.append(f"==================== STEP {i} ====================")
                    lines.append(c)
                    lines.append("")
                cot_path.write_text("\n".join(lines), encoding="utf-8")
            except Exception:
                pass
        except Exception:
            session.session_dir = None
            sess_dir = None

        # Phase 8: consolidation -- produce decisions, write decisions.json.
        if session.session_dir is not None:
            try:
                consolidator = Consolidator(promotion_threshold=3)
                decisions = consolidator.consolidate(
                    session.controller.subgraph, prior_citation_counts={}
                )
                consolidation_decisions = [
                    {
                        "node_id": d.node_id,
                        "node_type": d.node_type,
                        "decision": d.decision,
                        "reason": d.reason,
                        "gate_results": d.gate_results,
                    }
                    for d in decisions
                ]
                (Path(session.session_dir) / "decisions.json").write_text(
                    json.dumps(consolidation_decisions, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
            except Exception:
                consolidation_decisions = []

    # Phase 11: extract learning report + produce graph_edits.json (dry-run default).
    learning_report_dict: Optional[Dict[str, Any]] = None
    graph_edits_list: List[Dict[str, Any]] = []
    graph_edits_applied_flag = False
    scoped_patches_list: List[Dict[str, Any]] = []
    scoped_patch_summary: Dict[str, Any] = {}
    signature_candidates_list: List[Dict[str, Any]] = []
    signature_events_list: List[Dict[str, Any]] = []
    signature_stats_update: Dict[str, Any] = {}
    signature_shadow_report: Dict[str, Any] = {}
    signature_graph_projection: Dict[str, Any] = {}
    try:
        v4_to_ctrl_map = tools._v4_to_ctrl if session.controller is not None else {}
        subgraph_nodes = (
            session.controller.subgraph.nodes if session.controller is not None else {}
        )
        report = extract_learning_report(
            session_id=session_id,
            question=question,
            graph_id=resolved_graph_id,
            main_graph=graph,
            hypotheses=session.hypotheses,
            failures=session.failures,
            objects=session.objects,
            tool_log=tools.call_log,
            v4_to_ctrl_id=v4_to_ctrl_map,
            subgraph_nodes=subgraph_nodes,
        )
        # Extract strategy from successful sessions
        strategy_node = None
        if finalized:
            try:
                from reasoning.post_processing import extract_strategy
                strategy_node = extract_strategy(
                    report=report,
                    plan=session.plan,
                    tool_log=tools.call_log,
                    question=question,
                    task_frame=micro_outcome.task_frame,
                    finalized=finalized,
                    steps=session.step + 1,
                    tool_call_count=len(tools.call_log),
                    elapsed_sec=round(time.time() - t0, 1),
                )
            except Exception:
                pass
        graph_edits_list = produce_graph_edits(
            report, graph=graph, promotion_decisions=consolidation_decisions,
            strategy=strategy_node,
        )
        graph_edits_list.extend(
            propose_control_memory_edits(
                outcome=micro_outcome,
                question=question,
                session_id=session_id,
                graph=graph,
            )
        )
        learning_report_dict = report.to_dict()
        learning_report_dict = report.to_dict()
        try:
            _dedupe_idx = build_dedupe_index(graph)
            graph_edits_list = judge_edits_batch(
                graph_edits_list, graph, _dedupe_idx, controller
            )
        except Exception as e:
            logger.warning(f"Failed to judge deterministic graph edits: {e}")
            
        scoped_patch_objs = validate_patches(
            patches_from_graph_edits(
                graph_edits_list,
                graph=graph,
                learning_report=learning_report_dict,
                question=question,
                task_frame=(
                    micro_outcome.task_frame.to_dict()
                    if hasattr(micro_outcome.task_frame, "to_dict") else None
                ),
            ),
            graph,
        )
        scoped_patches_list = patches_to_dicts(scoped_patch_objs)
        scoped_patch_summary = summarize_patches(scoped_patch_objs)
        graph_edits_to_apply = approved_raw_edits_from_patches(scoped_patch_objs)
        if session.session_dir is not None:
            (Path(session.session_dir) / "learning_report.json").write_text(
                json.dumps(learning_report_dict, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            (Path(session.session_dir) / "graph_edits.json").write_text(
                json.dumps(graph_edits_list, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            (Path(session.session_dir) / "scoped_patches.json").write_text(
                json.dumps(scoped_patches_list, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            (Path(session.session_dir) / "scoped_patch_summary.json").write_text(
                json.dumps(scoped_patch_summary, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        if apply_graph_edits and graph_edits_list:
            backup_path = (
                Path(session.session_dir) / "graph_backup_pre_edits.json"
                if session.session_dir is not None else None
            )
            apply_summary = _apply_graph_edits(
                graph, graph_edits_to_apply,
                dry_run=False,
                backup_path=backup_path,
                allowed_tiers=("soft", "add", "promote"),
            )
            apply_summary["scoped_filter"] = {
                "raw_edits": len(graph_edits_list),
                "approved_edits": len(graph_edits_to_apply),
                "held_for_review": max(0, len(graph_edits_list) - len(graph_edits_to_apply)),
            }
            graph_edits_applied_flag = apply_summary.get("applied", 0) > 0
            if session.session_dir is not None:
                (Path(session.session_dir) / "graph_edits_apply_summary.json").write_text(
                    json.dumps(apply_summary, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
    except Exception:
        # Never let post-processing break the main flow.
        pass

    # Phase 18: signature family / variant stats in shadow mode only.
    try:
        signature_task_family = (
            controller_task_family
            or getattr(micro_outcome, "task_family", "")
            or task_type
        )
        signature_result = run_signature_shadow_session(
            session_id=session_id,
            question=question,
            task_family=signature_task_family,
            graph_edits=graph_edits_list,
            scoped_patches=scoped_patches_list,
            hypotheses=session.hypotheses,
            final_answer=final_answer,
            cited_node_ids=((learning_report_dict or {}).get("cited_node_ids", []) if isinstance(learning_report_dict, dict) else []),
            finalized=finalized,
            execution_mode=execution_mode,
            design_evidence_gate_rounds=session.design_evidence_gate_rounds,
            stats_dir=signature_stats_dir,
        )
        signature_candidates_list = list(signature_result.get("candidates", []))
        signature_events_list = list(signature_result.get("events", []))
        signature_stats_update = dict(signature_result.get("update_summary", {}))
        signature_shadow_report = dict(signature_result.get("shadow_report", {}))
        signature_graph_projection = dict(signature_result.get("graph_projection", {}))
        if session.session_dir is not None:
            (Path(session.session_dir) / "signature_live_bias.json").write_text(
                json.dumps(signature_live_bias, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            (Path(session.session_dir) / "signature_candidates.json").write_text(
                json.dumps(signature_candidates_list, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            (Path(session.session_dir) / "signature_events.json").write_text(
                json.dumps(signature_events_list, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            (Path(session.session_dir) / "signature_stats_update.json").write_text(
                json.dumps(signature_stats_update, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            (Path(session.session_dir) / "signature_shadow_report.json").write_text(
                json.dumps(signature_shadow_report, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            (Path(session.session_dir) / "signature_graph_projection.json").write_text(
                json.dumps(signature_graph_projection, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
    except Exception:
        pass

    # Phase 3: recompute coverage on the FINAL answer (which may differ from the
    # first-pass answer if the model revised in response to the coverage report).
    final_coverage = session.coverage
    final_coverage_pct = 1.0
    if (
        session.activation is not None
        and session.activation.task_frame.all_items()
        and finalized
    ):
        final_coverage = evaluate_coverage(session.activation.task_frame, final_answer)
        final_coverage_pct = final_coverage.get("coverage", 1.0)

    # Phase 12: polish the answer (strip node IDs + graph references; produce explanation).
    polished_answer = final_answer
    explanation_text = shortcut_explanation
    polish_applied = False
    raw_answer = final_answer
    if polish_answer and finalized and execution_mode != "direct_shortcut":
        try:
            polish = polish_final_answer(
                question=question,
                raw_answer=final_answer,
                session=session,
                controller=controller,
                node_ids=node_ids,
            )
            polished_answer = polish["answer"] or final_answer
            explanation_text = polish["explanation"]
            polish_applied = True
        except Exception:
            polished_answer = _strip_node_id_citations(final_answer, node_ids)
            polish_applied = False

    # Phase 14: optional inline reflection LLM call. Off by default -- the
    # canonical reflection path is offline via scripts/process_session.py.
    reflection_dict: Optional[Dict[str, Any]] = None
    reflection_edits: List[Dict[str, Any]] = []
    reflection_applied = False
    if run_reflection_inline and finalized:
        try:
            refl = run_reflection(
                controller=controller,
                session_id=session_id,
                question=question,
                anchors=session.anchors,
                tool_log=tools.call_log,
                hypotheses=session.hypotheses,
                failures=session.failures,
                objects=session.objects,
                polished_answer=polished_answer,
            )
            reflection_dict = refl.to_dict()
            if session.session_dir:
                (Path(session.session_dir) / "reflection.json").write_text(
                    json.dumps(reflection_dict, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
            try:
                _dedupe_idx = build_dedupe_index(graph)
                reflection_edits = edits_from_reflection_v2(refl, graph=graph, dedupe_index=_dedupe_idx)
                reflection_edits = judge_edits_batch(
                    reflection_edits, graph, _dedupe_idx, controller
                )
            except Exception as e:
                logger.warning(f"Failed to judge reflection edits: {e}")
                reflection_edits = edits_from_reflection(refl, graph=graph)
            if session.session_dir:
                (Path(session.session_dir) / "reflection_graph_edits.json").write_text(
                    json.dumps(reflection_edits, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
            if apply_graph_edits and reflection_edits:
                backup_path = (
                    Path(session.session_dir) / "graph_backup_pre_reflection_edits.json"
                    if session.session_dir is not None else None
                )
                apply_summary = _apply_reflection_edits(
                    graph, reflection_edits,
                    dry_run=False, backup_path=backup_path,
                    allowed_tiers=("soft", "add"),
                )
                reflection_applied = apply_summary.get("applied", 0) > 0
                if session.session_dir:
                    (Path(session.session_dir) / "reflection_apply_summary.json").write_text(
                        json.dumps(apply_summary, ensure_ascii=False, indent=2),
                        encoding="utf-8",
                    )
        except Exception:
            pass

    # Flush heuristic log with session outcomes
    _hlog.set_outcomes({
        "finalized": finalized,
        "steps": session.step + 1,
        "tool_calls": len(tools.call_log),
        "answer_length": len(polished_answer),
        "coverage_pct": final_coverage_pct,
        "graph_edits_count": len(graph_edits_list),
        "elapsed_sec": round(time.time() - t0, 1),
    })
    try:
        _hlog.flush("data/heuristic_logs")
    except Exception:
        pass

    controller_raw_trace: List[Dict[str, Any]] = []
    if hasattr(controller, "get_raw_trace"):
        try:
            controller_raw_trace = list(getattr(controller, "get_raw_trace")())
        except Exception:
            controller_raw_trace = []
    controller_call_count = len(controller_raw_trace)
    controller_total_elapsed_sec = round(
        sum(float(entry.get("elapsed_sec", 0.0) or 0.0) for entry in controller_raw_trace),
        3,
    )
    controller_nonempty_turns = sum(
        1 for entry in controller_raw_trace
        if str(entry.get("assistant_text", "") or "").strip()
    )

    # Phase 15: append to distillation corpus.
    _pkt = V4Packet(
        question=question,
        answer=polished_answer,
        task_type=task_type,
        execution_mode=execution_mode,
        steps=session.step + 1,
        max_steps=max_steps,
        tool_call_count=len(tools.call_log),
        tool_log=tools.call_log,
        cot_log=cot_log,
        plan=session.plan,
        objects=session.objects,
        failures=session.failures,
        elapsed_sec=round(elapsed, 1),
        finalized=finalized,
        anchors=list(anchors),
        citation_warnings=citation_warnings,
        search_repeats=tools._search_repeats,
        task_frame_items=task_frame_items_count,
        activation_signals=activation_signals_count,
        task_frame=(
            session.activation.task_frame.to_dict()
            if session.activation is not None and session.activation.task_frame.all_items()
            else None
        ),
        task_frame_rendered=task_frame_block,
        coverage=final_coverage,
        coverage_addressed_pct=final_coverage_pct,
        coverage_rounds=session.coverage_rounds,
        hypotheses=session.hypotheses,
        session_dir=session.session_dir,
        meta_signals=(
            [s.to_dict() for s in session.meta_pool.signal_stream]
            if session.meta_pool is not None else []
        ),
        budget_summary=(
            session.budget_tracker.summary() if session.budget_tracker is not None else None
        ),
        consolidation_decisions=consolidation_decisions,
        plan_tree_summary=(
            session.plan_tree.to_dict() if session.plan_tree is not None else None
        ),
        procedure_invocations=session.procedure_invocations,
        learning_report=learning_report_dict,
        graph_edits=graph_edits_list,
        graph_edits_applied=graph_edits_applied_flag,
        scoped_patches=scoped_patches_list,
        scoped_patch_summary=scoped_patch_summary,
        answer_raw=raw_answer,
        explanation=explanation_text,
        polish_applied=polish_applied,
        shortcut_reason=shortcut_reason,
        shortcut_anchor_ids=shortcut_anchor_ids,
        controller_task_family=controller_task_family,
        micro_steps=[step.to_dict() for step in micro_outcome.micro_steps],
        subgoal_reuse_count=micro_outcome.subgoal_reuse_count,
        slot_fill_stats=micro_outcome.slot_fill_stats(),
        controller_action_counts=dict(micro_outcome.controller_action_counts),
        controller_fallback_used=controller_fallback_used and not micro_shortcut_used,
        controller_raw_trace=controller_raw_trace,
        controller_call_count=controller_call_count,
        controller_total_elapsed_sec=controller_total_elapsed_sec,
        controller_nonempty_turns=controller_nonempty_turns,
        reflection=reflection_dict,
        reflection_edits=reflection_edits,
        reflection_applied=reflection_applied,
        signature_candidates=signature_candidates_list,
        signature_events=signature_events_list,
        signature_stats_update=signature_stats_update,
        signature_shadow_report=signature_shadow_report,
        signature_graph_projection=signature_graph_projection,
        signature_live_bias=signature_live_bias,
    )
    if collect_corpus and _pkt.finalized:
        try:
            append_session_to_corpus(
                _pkt, graph,
                controller_label=controller_label,
            )
        except Exception:
            pass
    return _pkt
