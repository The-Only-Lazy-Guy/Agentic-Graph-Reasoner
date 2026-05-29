"""Shadow-mode signature family / variant stats for graph learning.

Phase 1 scope:
  - strategy
  - solved_subgoal
  - provisional_claim

The module does not change live retrieval. It collects end-of-run events,
updates a persistent signature index, and emits a shadow rerank report so we
can inspect what *would* move once retrieval integration is turned on.
"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from reasoning.lexical_matching import content_tokens, lexical_overlap, normalize_text


PHASE1_SIGNATURE_TYPES = frozenset({"strategy", "solved_subgoal", "provisional_claim"})
EVENT_TYPES = frozenset({
    "supported_reuse",
    "supported_finalize",
    "provisional_used_with_caveat",
    "hypothesis_discarded",
    "answer_gate_rewrite",
    "scoped_patch_accept",
    "scoped_patch_soft_only",
    "scoped_patch_needs_review",
    "scoped_patch_reject",
    "low_relevance_retrieval",
    "contradicted",
    "promoted_to_review",
    "promoted_to_supported",
    "deprecated",
})
IMPACT_BUCKET_WEIGHTS = {
    "tiny": 0.25,
    "low": 0.50,
    "medium": 1.00,
    "high": 1.50,
    "critical": 2.25,
}
EVENT_BASE_DELTAS: Dict[str, Dict[str, float]] = {
    "supported_reuse": {"support": 0.30, "stability": 0.95, "risk": -0.05},
    "supported_finalize": {"support": 0.70, "stability": 1.10, "risk": -0.05},
    "provisional_used_with_caveat": {"stability": 0.30, "risk": 0.12},
    "hypothesis_discarded": {"support": -0.60, "stability": -0.20, "risk": 0.20},
    "answer_gate_rewrite": {"support": -0.80, "stability": -0.25, "risk": 0.40},
    "scoped_patch_accept": {"support": 0.75, "stability": 0.30},
    "scoped_patch_soft_only": {"support": 0.25, "stability": 0.10},
    "scoped_patch_needs_review": {"support": -0.85, "stability": -0.30, "risk": 0.45},
    "scoped_patch_reject": {"support": -1.20, "stability": -0.50, "risk": 0.65, "contradiction": 0.60},
    "low_relevance_retrieval": {"stability": -0.35, "risk": 0.20},
    "contradicted": {"support": -1.75, "stability": -0.80, "risk": 0.60, "contradiction": 1.75},
    "promoted_to_review": {"support": 0.40, "stability": 0.20, "risk": -0.05},
    "promoted_to_supported": {"support": 1.00, "stability": 0.60, "risk": -0.10},
    "deprecated": {"support": -1.00, "stability": -0.50, "risk": 0.55, "contradiction": 0.50},
}

_HEDGE_CUES = (
    "possible",
    "possibly",
    "may",
    "might",
    "could",
    "hypothesis",
    "tentative",
    "one possible",
    "not directly supported",
    "until verified",
)
_NEGATION_TERMS = frozenset({"not", "never", "cannot", "can't", "unsafe", "invalid", "incorrect", "fails", "failure"})
_AFFIRMATION_TERMS = frozenset({"can", "safe", "valid", "correct", "works", "guaranteed", "succeeds", "reliable"})
_STATUS_EVENT_BY_PATCH = {
    "accept": "scoped_patch_accept",
    "soft_only": "scoped_patch_soft_only",
    "needs_review": "scoped_patch_needs_review",
    "reject": "scoped_patch_reject",
}
_PROMOTION_BLOCKING_EVENTS = frozenset({
    "scoped_patch_needs_review",
    "scoped_patch_reject",
    "contradicted",
    "deprecated",
})
_REVIEW_SENSITIVE_EVENTS = frozenset({
    "low_relevance_retrieval",
    "answer_gate_rewrite",
})

# --- NLI judge for borderline sibling-variant comparison ---

_NLI_JUDGE_SYSTEM = """\
You are a knowledge-graph variant relation judge. You will be shown two memory
variants from the same semantic family. Determine their relationship.

Output EXACTLY one relation tag:

  <relation type="equivalent">Rationale: same memory, reworded.</relation>
  <relation type="sibling">Rationale: same family, meaningfully different.</relation>
  <relation type="contradicts">Rationale: opposite or incompatible claims.</relation>
  <relation type="independent">Rationale: too different to relate.</relation>

Rules:
  - If both variants express the same core fact/conclusion with different wording, use equivalent.
  - If they address the same topic but add different details or perspectives, use sibling.
  - If one affirms and the other denies the same claim, use contradicts.
  - If they are too different to meaningfully relate, use independent.
"""

_NLI_RELATION_RE = re.compile(
    r'<relation\s+type="(?P<type>equivalent|sibling|contradicts|independent)"'
    r'\s*>(?P<rationale>.*?)</relation>',
    re.DOTALL | re.IGNORECASE,
)

# --- LLM event impact scoring ---

_EVENT_IMPACT_SYSTEM = """\
You are a memory-event impact scorer. Given a signature event and its context,
output a numeric impact score from 0.0 (negligible) to 2.5 (critical).

Consider:
  - How much this event changes the variant's trustworthiness
  - Whether it affects final answer quality
  - How many sessions/evidence support this pattern

Output EXACTLY one tag:

  <impact score="X.XX">Brief rationale.</impact>

Rules:
  - Score near 0.0 for routine repetitions with no new information
  - Score near 0.5 for mild reinforcement
  - Score near 1.0 for meaningful new evidence
  - Score near 1.5 for strong trust-building events
  - Score near 2.0 for critical quality gates
  - Score near 2.5 for paradigm-shifting discoveries
"""

_IMPACT_SCORE_RE = re.compile(
    r'<impact\s+score="(?P<score>-?[0-9]+\.?[0-9]*)"\s*>(?P<rationale>.*?)</impact>',
    re.DOTALL | re.IGNORECASE,
)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _slug(text: Any) -> str:
    slug = re.sub(r"[^a-zA-Z0-9_]+", "_", str(text or "").strip().lower())
    slug = re.sub(r"_+", "_", slug).strip("_")
    return slug or "unknown"


def _short_hash(payload: Any) -> str:
    raw = payload if isinstance(payload, str) else json.dumps(payload, sort_keys=True, ensure_ascii=False)
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:12]


def _stable_list(values: Iterable[Any]) -> List[str]:
    seen = set()
    out: List[str] = []
    for value in values:
        text = str(value or "").strip()
        if not text or text in seen:
            continue
        seen.add(text)
        out.append(text)
    return out


def _top_tokens(text: str, *, limit: int = 4) -> List[str]:
    toks = sorted(content_tokens(text, min_chars=4))
    if toks:
        return toks[:limit]
    raw = re.findall(r"[A-Za-z0-9_]+", normalize_text(text).lower())
    return sorted(set(t for t in raw if len(t) >= 4))[:limit]


def _question_fingerprint(question: str) -> str:
    return _short_hash(sorted(content_tokens(question, min_chars=4)))


def _evidence_fingerprint(evidence_node_ids: Sequence[str]) -> str:
    return _short_hash(sorted(set(str(x) for x in evidence_node_ids if x)))


def _merge_lists(*parts: Iterable[Any]) -> List[str]:
    merged: List[str] = []
    for part in parts:
        merged.extend(str(x) for x in part if str(x or "").strip())
    return _stable_list(merged)


def _jaccard_strings(a: Sequence[str], b: Sequence[str]) -> float:
    sa = {str(x).strip().lower() for x in a if str(x or "").strip()}
    sb = {str(x).strip().lower() for x in b if str(x or "").strip()}
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def _scope_pairs(scope: Mapping[str, Any]) -> List[str]:
    pairs: List[str] = []
    for key, value in sorted((scope or {}).items()):
        if value in ("", None, [], {}):
            continue
        if isinstance(value, Mapping):
            for skey, sval in sorted(value.items()):
                if sval in ("", None, [], {}):
                    continue
                pairs.append(f"{key}.{skey}={normalize_text(sval).lower()}")
        elif isinstance(value, (list, tuple, set)):
            for item in value:
                text = normalize_text(item).lower()
                if text:
                    pairs.append(f"{key}[]={text}")
        else:
            pairs.append(f"{key}={normalize_text(value).lower()}")
    return pairs


def _relation_record_id(
    *,
    family_id: str,
    src_variant_id: str,
    dst_variant_id: str,
    relation_type: str,
    symmetric: bool,
) -> str:
    if symmetric:
        src_variant_id, dst_variant_id = sorted([src_variant_id, dst_variant_id])
    return f"sigrel_{_slug(relation_type)}_{_short_hash([family_id, src_variant_id, dst_variant_id, symmetric])}"


@dataclass
class SignatureCandidate:
    family_id: str
    variant_id: str
    semantic_type: str
    family_label: str
    canonical_text: str
    summary_text: str
    task_family: str
    task_subtype: str = ""
    question_mode: str = ""
    epistemic_status: str = "provisional"
    promotion_state: str = "blocked"
    retrieval_tier: str = "gated"
    source_node_ids: List[str] = field(default_factory=list)
    linked_node_ids: List[str] = field(default_factory=list)
    supporting_node_ids: List[str] = field(default_factory=list)
    required_slots: List[str] = field(default_factory=list)
    scope: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    proposed_family_id: str = ""
    proposed_variant_id: str = ""
    matched_family_id: str = ""
    matched_variant_id: str = ""
    family_resolution: str = "new_family"
    variant_resolution: str = "new_variant"
    relation_to_match: str = "independent"
    family_match_score: float = 0.0
    variant_match_score: float = 0.0
    resolution_reason: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class SignatureEvent:
    event_id: str
    session_id: str
    question: str
    question_fingerprint: str
    signature_family_id: str
    signature_variant_id: str
    semantic_type: str
    event_type: str
    impact_bucket: str
    event_reason: str = ""
    ambiguous: bool = False
    impact_multiplier: float = 1.0
    task_family: str = ""
    evidence_node_ids: List[str] = field(default_factory=list)
    linked_node_ids: List[str] = field(default_factory=list)
    affected_node_ids: List[str] = field(default_factory=list)
    affected_final_answer: bool = False
    output_caveated: bool = False
    evidence_fingerprint: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @staticmethod
    def from_dict(d: Mapping[str, Any]) -> "SignatureEvent":
        return SignatureEvent(
            event_id=str(d.get("event_id", "")),
            session_id=str(d.get("session_id", "")),
            question=str(d.get("question", "")),
            question_fingerprint=str(d.get("question_fingerprint", "")),
            signature_family_id=str(d.get("signature_family_id", "")),
            signature_variant_id=str(d.get("signature_variant_id", "")),
            semantic_type=str(d.get("semantic_type", "")),
            event_type=str(d.get("event_type", "")),
            impact_bucket=str(d.get("impact_bucket", "medium")),
            event_reason=str(d.get("event_reason", "")),
            ambiguous=bool(d.get("ambiguous", False)),
            impact_multiplier=float(d.get("impact_multiplier", 1.0)),
            task_family=str(d.get("task_family", "")),
            evidence_node_ids=list(d.get("evidence_node_ids", [])),
            linked_node_ids=list(d.get("linked_node_ids", [])),
            affected_node_ids=list(d.get("affected_node_ids", [])),
            affected_final_answer=bool(d.get("affected_final_answer", False)),
            output_caveated=bool(d.get("output_caveated", False)),
            evidence_fingerprint=str(d.get("evidence_fingerprint", "")),
            metadata=dict(d.get("metadata", {})),
        )


@dataclass
class SignatureVariantStats:
    id: str
    family_id: str
    semantic_type: str
    canonical_text: str
    summary_text: str
    task_family: str
    task_subtype: str = ""
    question_mode: str = ""
    epistemic_status: str = "provisional"
    promotion_state: str = "blocked"
    retrieval_tier: str = "gated"
    support_score: float = 0.0
    stability_score: float = 0.0
    risk_score: float = 0.0
    contradiction_score: float = 0.0
    bias_score: float = 0.0
    propagated_support_score: float = 0.0
    propagated_stability_score: float = 0.0
    propagated_risk_score: float = 0.0
    propagated_contradiction_score: float = 0.0
    effective_support_score: float = 0.0
    effective_stability_score: float = 0.0
    effective_risk_score: float = 0.0
    effective_contradiction_score: float = 0.0
    effective_bias_score: float = 0.0
    positive_events: int = 0
    negative_events: int = 0
    event_counts: Dict[str, int] = field(default_factory=dict)
    relation_counts: Dict[str, int] = field(default_factory=dict)
    source_node_ids: List[str] = field(default_factory=list)
    linked_node_ids: List[str] = field(default_factory=list)
    top_supporting_node_ids: List[str] = field(default_factory=list)
    question_fingerprints: List[str] = field(default_factory=list)
    evidence_fingerprints: List[str] = field(default_factory=list)
    session_ids: List[str] = field(default_factory=list)
    required_slots: List[str] = field(default_factory=list)
    scope: Dict[str, Any] = field(default_factory=dict)
    aliases: List[str] = field(default_factory=list)
    last_updated_at: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @staticmethod
    def from_dict(d: Mapping[str, Any]) -> "SignatureVariantStats":
        return SignatureVariantStats(
            id=str(d.get("id", "")),
            family_id=str(d.get("family_id", "")),
            semantic_type=str(d.get("semantic_type", "")),
            canonical_text=str(d.get("canonical_text", "")),
            summary_text=str(d.get("summary_text", "")),
            task_family=str(d.get("task_family", "")),
            task_subtype=str(d.get("task_subtype", "")),
            question_mode=str(d.get("question_mode", "")),
            epistemic_status=str(d.get("epistemic_status", "provisional")),
            promotion_state=str(d.get("promotion_state", "blocked")),
            retrieval_tier=str(d.get("retrieval_tier", "gated")),
            support_score=float(d.get("support_score", 0.0)),
            stability_score=float(d.get("stability_score", 0.0)),
            risk_score=float(d.get("risk_score", 0.0)),
            contradiction_score=float(d.get("contradiction_score", 0.0)),
            bias_score=float(d.get("bias_score", 0.0)),
            propagated_support_score=float(d.get("propagated_support_score", 0.0)),
            propagated_stability_score=float(d.get("propagated_stability_score", 0.0)),
            propagated_risk_score=float(d.get("propagated_risk_score", 0.0)),
            propagated_contradiction_score=float(d.get("propagated_contradiction_score", 0.0)),
            effective_support_score=float(d.get("effective_support_score", d.get("support_score", 0.0))),
            effective_stability_score=float(d.get("effective_stability_score", d.get("stability_score", 0.0))),
            effective_risk_score=float(d.get("effective_risk_score", d.get("risk_score", 0.0))),
            effective_contradiction_score=float(d.get("effective_contradiction_score", d.get("contradiction_score", 0.0))),
            effective_bias_score=float(d.get("effective_bias_score", d.get("bias_score", 0.0))),
            positive_events=int(d.get("positive_events", 0)),
            negative_events=int(d.get("negative_events", 0)),
            event_counts=dict(d.get("event_counts", {})),
            relation_counts=dict(d.get("relation_counts", {})),
            source_node_ids=list(d.get("source_node_ids", [])),
            linked_node_ids=list(d.get("linked_node_ids", [])),
            top_supporting_node_ids=list(d.get("top_supporting_node_ids", [])),
            question_fingerprints=list(d.get("question_fingerprints", [])),
            evidence_fingerprints=list(d.get("evidence_fingerprints", [])),
            session_ids=list(d.get("session_ids", [])),
            required_slots=list(d.get("required_slots", [])),
            scope=dict(d.get("scope", {})),
            aliases=list(d.get("aliases", [])),
            last_updated_at=str(d.get("last_updated_at", "")),
        )


@dataclass
class SignatureRelationStats:
    id: str
    family_id: str
    src_variant_id: str
    dst_variant_id: str
    relation_type: str
    symmetric: bool = False
    observation_count: int = 0
    source_session_ids: List[str] = field(default_factory=list)
    reasons: List[str] = field(default_factory=list)
    last_match_score: float = 0.0
    last_updated_at: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @staticmethod
    def from_dict(d: Mapping[str, Any]) -> "SignatureRelationStats":
        return SignatureRelationStats(
            id=str(d.get("id", "")),
            family_id=str(d.get("family_id", "")),
            src_variant_id=str(d.get("src_variant_id", "")),
            dst_variant_id=str(d.get("dst_variant_id", "")),
            relation_type=str(d.get("relation_type", "")),
            symmetric=bool(d.get("symmetric", False)),
            observation_count=int(d.get("observation_count", 0)),
            source_session_ids=list(d.get("source_session_ids", [])),
            reasons=list(d.get("reasons", [])),
            last_match_score=float(d.get("last_match_score", 0.0)),
            last_updated_at=str(d.get("last_updated_at", "")),
        )


@dataclass
class SignatureFamilyStats:
    id: str
    semantic_type: str
    family_label: str
    task_family: str
    variant_ids: List[str] = field(default_factory=list)
    support_score: float = 0.0
    stability_score: float = 0.0
    risk_score: float = 0.0
    contradiction_score: float = 0.0
    bias_score: float = 0.0
    effective_support_score: float = 0.0
    effective_stability_score: float = 0.0
    effective_risk_score: float = 0.0
    effective_contradiction_score: float = 0.0
    effective_bias_score: float = 0.0
    contested: bool = False
    dominant_variant_id: Optional[str] = None
    event_counts: Dict[str, int] = field(default_factory=dict)
    relation_counts: Dict[str, int] = field(default_factory=dict)
    last_updated_at: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @staticmethod
    def from_dict(d: Mapping[str, Any]) -> "SignatureFamilyStats":
        return SignatureFamilyStats(
            id=str(d.get("id", "")),
            semantic_type=str(d.get("semantic_type", "")),
            family_label=str(d.get("family_label", "")),
            task_family=str(d.get("task_family", "")),
            variant_ids=list(d.get("variant_ids", [])),
            support_score=float(d.get("support_score", 0.0)),
            stability_score=float(d.get("stability_score", 0.0)),
            risk_score=float(d.get("risk_score", 0.0)),
            contradiction_score=float(d.get("contradiction_score", 0.0)),
            bias_score=float(d.get("bias_score", 0.0)),
            effective_support_score=float(d.get("effective_support_score", d.get("support_score", 0.0))),
            effective_stability_score=float(d.get("effective_stability_score", d.get("stability_score", 0.0))),
            effective_risk_score=float(d.get("effective_risk_score", d.get("risk_score", 0.0))),
            effective_contradiction_score=float(d.get("effective_contradiction_score", d.get("contradiction_score", 0.0))),
            effective_bias_score=float(d.get("effective_bias_score", d.get("bias_score", 0.0))),
            contested=bool(d.get("contested", False)),
            dominant_variant_id=d.get("dominant_variant_id"),
            event_counts=dict(d.get("event_counts", {})),
            relation_counts=dict(d.get("relation_counts", {})),
            last_updated_at=str(d.get("last_updated_at", "")),
        )


@dataclass
class SignatureStatsIndex:
    version: int = 2
    updated_at: str = ""
    total_events: int = 0
    families: Dict[str, SignatureFamilyStats] = field(default_factory=dict)
    variants: Dict[str, SignatureVariantStats] = field(default_factory=dict)
    relations: Dict[str, SignatureRelationStats] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "version": self.version,
            "updated_at": self.updated_at,
            "total_events": self.total_events,
            "families": {k: v.to_dict() for k, v in self.families.items()},
            "variants": {k: v.to_dict() for k, v in self.variants.items()},
            "relations": {k: v.to_dict() for k, v in self.relations.items()},
        }

    @staticmethod
    def from_dict(d: Mapping[str, Any]) -> "SignatureStatsIndex":
        return SignatureStatsIndex(
            version=int(d.get("version", 1)),
            updated_at=str(d.get("updated_at", "")),
            total_events=int(d.get("total_events", 0)),
            families={
                str(k): SignatureFamilyStats.from_dict(v)
                for k, v in dict(d.get("families", {})).items()
                if isinstance(v, Mapping)
            },
            variants={
                str(k): SignatureVariantStats.from_dict(v)
                for k, v in dict(d.get("variants", {})).items()
                if isinstance(v, Mapping)
            },
            relations={
                str(k): SignatureRelationStats.from_dict(v)
                for k, v in dict(d.get("relations", {})).items()
                if isinstance(v, Mapping)
            },
        )


@dataclass
class LiveSignatureBiasPlan:
    enabled: bool = False
    reason: str = ""
    question: str = ""
    task_family: str = ""
    family_id: str = ""
    family_label: str = ""
    variant_id: str = ""
    semantic_type: str = ""
    baseline_rank: int = 0
    adjusted_rank: int = 0
    adjusted_score: float = 0.0
    baseline_score: float = 0.0
    bias_score: float = 0.0
    shadow_adjustment: float = 0.0
    anchor_ids: List[str] = field(default_factory=list)
    anchor_rows: List[Dict[str, Any]] = field(default_factory=list)
    support_node_ids: List[str] = field(default_factory=list)
    linked_node_ids: List[str] = field(default_factory=list)
    skipped_candidates: List[Dict[str, Any]] = field(default_factory=list)
    candidate_count: int = 0
    stats_index_path: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def default_signature_stats_dir(root: str | Path = "data/signature_stats") -> Path:
    return Path(root)


def load_signature_stats_index(path: str | Path) -> SignatureStatsIndex:
    p = Path(path)
    if not p.exists():
        return SignatureStatsIndex(updated_at=_now_iso())
    payload = json.loads(p.read_text(encoding="utf-8"))
    return SignatureStatsIndex.from_dict(payload)


def save_signature_stats_index(index: SignatureStatsIndex, path: str | Path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(index.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")


def append_signature_events_jsonl(events: Sequence[SignatureEvent], path: str | Path) -> None:
    if not events:
        return
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as fh:
        for event in events:
            fh.write(json.dumps(event.to_dict(), ensure_ascii=False) + "\n")


def _strategy_candidate_from_edit(edit: Mapping[str, Any], *, question: str) -> SignatureCandidate:
    meta = dict(edit.get("metadata") or {})
    task_family = str(meta.get("task_family", "") or "generic")
    slot_order = list(meta.get("slot_order", []) or meta.get("required_slots", []))
    slot_key = "+".join(_slug(s) for s in slot_order[:5]) or "generic_slots"
    keywords = list(meta.get("domain_keywords", []) or _top_tokens(question, limit=3))
    concept_key = "+".join(_slug(k) for k in keywords[:3]) or "generic_concept"
    family_id = f"sigfam_strategy.{_slug(task_family)}.{slot_key}.{concept_key}"
    variant_payload = {
        "plan_template": list(meta.get("plan_template", [])),
        "checkpoint_plan": list(meta.get("checkpoint_plan", [])),
        "entry_conditions": dict(meta.get("entry_conditions", {})),
        "question_mode": meta.get("question_mode", ""),
    }
    variant_id = f"sigvar_strategy_{_short_hash([family_id, variant_payload, edit.get('text', '')])}"
    return SignatureCandidate(
        family_id=family_id,
        variant_id=variant_id,
        semantic_type="strategy",
        family_label=f"strategy:{task_family}:{slot_key}",
        canonical_text=str(edit.get("text", "") or ""),
        summary_text=str(meta.get("question_pattern", "") or edit.get("text", "") or ""),
        task_family=task_family,
        task_subtype=str(meta.get("task_subtype", "") or ""),
        question_mode=str(meta.get("question_mode", "") or ""),
        epistemic_status="provisional",
        promotion_state="blocked",
        retrieval_tier="gated",
        source_node_ids=[str(edit.get("node_id", ""))],
        linked_node_ids=_stable_list(meta.get("key_node_ids", [])),
        supporting_node_ids=_stable_list(meta.get("key_node_ids", [])),
        required_slots=[str(x) for x in slot_order if str(x).strip()],
        scope={
            "task_family": task_family,
            "task_subtype": str(meta.get("task_subtype", "") or ""),
            "question_mode": str(meta.get("question_mode", "") or ""),
            "entry_conditions": dict(meta.get("entry_conditions", {})),
        },
        metadata=meta,
        proposed_family_id=family_id,
        proposed_variant_id=variant_id,
    )


def _solved_subgoal_candidate_from_edit(edit: Mapping[str, Any]) -> SignatureCandidate:
    meta = dict(edit.get("metadata") or {})
    task_family = str(meta.get("question_type", "") or meta.get("task_family", "") or "generic")
    subgoal_signature = str(meta.get("subgoal_signature", "") or edit.get("node_id", ""))
    slot_names = sorted(str(k) for k in dict(meta.get("output_slots", {})).keys())
    slot_key = "+".join(_slug(s) for s in slot_names[:5]) or "generic_slots"
    family_id = f"sigfam_solved_subgoal.{_slug(subgoal_signature)}"
    variant_payload = {
        "output_slots": dict(meta.get("output_slots", {})),
        "valid_when": list(meta.get("valid_when", [])),
        "invalid_when": list(meta.get("invalid_when", [])),
    }
    variant_id = f"sigvar_solved_subgoal_{_short_hash([family_id, variant_payload, edit.get('text', '')])}"
    return SignatureCandidate(
        family_id=family_id,
        variant_id=variant_id,
        semantic_type="solved_subgoal",
        family_label=f"solved_subgoal:{subgoal_signature}",
        canonical_text=str(meta.get("summary", "") or edit.get("text", "") or ""),
        summary_text=str(meta.get("summary", "") or edit.get("text", "") or ""),
        task_family=task_family,
        task_subtype=str(meta.get("task_subtype", "") or ""),
        question_mode=str(meta.get("question_mode", "") or ""),
        epistemic_status="supported",
        promotion_state="review",
        retrieval_tier="normal",
        source_node_ids=[str(edit.get("node_id", ""))],
        linked_node_ids=_stable_list(meta.get("supporting_node_ids", [])),
        supporting_node_ids=_stable_list(meta.get("supporting_node_ids", [])),
        required_slots=slot_names,
        scope={
            "subgoal_signature": subgoal_signature,
            "input_conditions": dict(meta.get("input_conditions", {})),
            "valid_when": list(meta.get("valid_when", [])),
            "invalid_when": list(meta.get("invalid_when", [])),
        },
        metadata=meta,
        proposed_family_id=family_id,
        proposed_variant_id=variant_id,
    )


def _provisional_claim_candidate(
    *,
    text: str,
    question: str,
    task_family: str,
    evidence_node_ids: Sequence[str],
    reason: str,
) -> SignatureCandidate:
    concepts = _top_tokens(text or question, limit=3)
    concept_key = "+".join(_slug(x) for x in concepts[:3]) or "generic_claim"
    family_id = f"sigfam_provisional_claim.{_slug(task_family or 'generic')}.{concept_key}"
    variant_id = f"sigvar_provisional_claim_{_short_hash([family_id, normalize_text(text).lower()])}"
    return SignatureCandidate(
        family_id=family_id,
        variant_id=variant_id,
        semantic_type="provisional_claim",
        family_label=f"provisional_claim:{task_family}:{concept_key}",
        canonical_text=str(text or "").strip(),
        summary_text=str(text or "").strip(),
        task_family=str(task_family or "generic"),
        epistemic_status="provisional",
        promotion_state="blocked",
        retrieval_tier="gated",
        source_node_ids=[],
        linked_node_ids=_stable_list(evidence_node_ids),
        supporting_node_ids=_stable_list(evidence_node_ids),
        required_slots=[],
        scope={"reason": reason},
        metadata={"origin_reason": reason},
        proposed_family_id=family_id,
        proposed_variant_id=variant_id,
    )


def _merge_candidate(dst: SignatureCandidate, src: SignatureCandidate) -> None:
    dst.source_node_ids = _merge_lists(dst.source_node_ids, src.source_node_ids)
    dst.linked_node_ids = _merge_lists(dst.linked_node_ids, src.linked_node_ids)
    dst.supporting_node_ids = _merge_lists(dst.supporting_node_ids, src.supporting_node_ids)
    dst.required_slots = _merge_lists(dst.required_slots, src.required_slots)
    if not dst.summary_text and src.summary_text:
        dst.summary_text = src.summary_text
    if len(src.canonical_text) > len(dst.canonical_text):
        dst.canonical_text = src.canonical_text
    dst.metadata.update(src.metadata)


def _family_basis_text(family: SignatureFamilyStats, index: SignatureStatsIndex) -> str:
    parts = [family.family_label, family.task_family]
    for vid in family.variant_ids[:3]:
        variant = index.variants.get(vid)
        if variant is None:
            continue
        parts.extend([variant.canonical_text, variant.summary_text, " ".join(variant.required_slots)])
    return " ".join(p for p in parts if p)


def _family_support_node_ids(family: SignatureFamilyStats, index: SignatureStatsIndex, *, limit_variants: int = 4) -> List[str]:
    support_ids: List[str] = []
    for vid in family.variant_ids[:limit_variants]:
        variant = index.variants.get(vid)
        if variant is None:
            continue
        support_ids.extend(variant.top_supporting_node_ids)
        support_ids.extend(variant.linked_node_ids)
    return _stable_list(support_ids)


def _family_prefilter_score(candidate: SignatureCandidate, family: SignatureFamilyStats, index: SignatureStatsIndex) -> float:
    if family.semantic_type != candidate.semantic_type:
        return -1.0
    if family.task_family != candidate.task_family:
        return -1.0
    basis = _family_basis_text(family, index)
    family_text = " ".join([candidate.family_label, candidate.canonical_text, " ".join(candidate.required_slots)])
    ref_slots = [
        slot
        for vid in family.variant_ids[:3]
        for slot in (index.variants.get(vid).required_slots if index.variants.get(vid) is not None else [])
    ]
    ref_support = _family_support_node_ids(family, index)
    slot_j = _jaccard_strings(candidate.required_slots, ref_slots)
    text_overlap = lexical_overlap(candidate.canonical_text, basis, min_chars=3)
    support_j = _jaccard_strings(candidate.supporting_node_ids, ref_support)
    if candidate.semantic_type == "solved_subgoal":
        if family.task_family == "direct_judgment" and support_j <= 0.0 and text_overlap < 0.25:
            return -1.0
        score = 0.0
        score += 1.60 * support_j
        score += 1.00 * text_overlap
        score += 0.20 * slot_j
        if candidate.question_mode and any((index.variants.get(vid) and index.variants[vid].question_mode == candidate.question_mode) for vid in family.variant_ids[:3]):
            score += 0.15
        if candidate.task_subtype and any((index.variants.get(vid) and index.variants[vid].task_subtype == candidate.task_subtype) for vid in family.variant_ids[:3]):
            score += 0.15
        return round(score, 4)
    score = 0.0
    score += 1.10 * slot_j
    score += 0.85 * lexical_overlap(family_text, basis, min_chars=3)
    if candidate.question_mode and any((index.variants.get(vid) and index.variants[vid].question_mode == candidate.question_mode) for vid in family.variant_ids[:3]):
        score += 0.20
    if candidate.task_subtype and any((index.variants.get(vid) and index.variants[vid].task_subtype == candidate.task_subtype) for vid in family.variant_ids[:3]):
        score += 0.20
    return round(score, 4)


def _variant_prefilter_score(candidate: SignatureCandidate, variant: SignatureVariantStats) -> float:
    if variant.semantic_type != candidate.semantic_type:
        return -1.0
    if variant.family_id != candidate.family_id:
        return -1.0
    score = 0.0
    if variant.task_family == candidate.task_family:
        score += 0.35
    if candidate.question_mode and variant.question_mode == candidate.question_mode:
        score += 0.20
    if candidate.task_subtype and variant.task_subtype == candidate.task_subtype:
        score += 0.20
    score += 1.00 * _jaccard_strings(candidate.required_slots, variant.required_slots)
    score += 0.65 * _jaccard_strings(_scope_pairs(candidate.scope), _scope_pairs(variant.scope))
    score += 0.70 * lexical_overlap(candidate.canonical_text, variant.canonical_text, min_chars=3)
    score += 0.35 * _jaccard_strings(candidate.supporting_node_ids, variant.top_supporting_node_ids)
    if variant.epistemic_status == candidate.epistemic_status:
        score += 0.10
    return round(score, 4)


def _polarity_profile(text: str) -> Tuple[bool, bool]:
    low = normalize_text(text).lower()
    tokens = set(re.findall(r"[A-Za-z']+", low))
    has_neg = any(tok in tokens or tok in low for tok in _NEGATION_TERMS)
    has_aff = any(tok in tokens or tok in low for tok in _AFFIRMATION_TERMS)
    return has_neg, has_aff



def _judge_equivalent_vs_sibling(
    candidate: SignatureCandidate,
    variant: SignatureVariantStats,
    controller: Any,
) -> bool:
    if not controller:
        return False
    system_prompt = """You are an expert judge comparing two reasoning signature variants.
Your job is to decide whether a NEW candidate is just an 'equivalent_revision' of an EXISTING variant, or if it is a distinct 'sibling_variant'.
If the candidate is just phrasing the same underlying concept/strategy differently, return <verdict>equivalent</verdict>.
If the candidate introduces a distinct condition, edge case, or structural change that warrants tracking separately, return <verdict>sibling</verdict>.
"""
    user_prompt = f"""
EXISTING VARIANT:
{variant.canonical_text}

NEW CANDIDATE:
{candidate.canonical_text}

Are these fundamentally the same core idea (equivalent), or distinct enough to track separately (sibling)?
"""
    try:
        chat_fn = getattr(controller, "chat_oneshot", getattr(controller, "chat", None))
        if not chat_fn:
            return False
        resp = chat_fn([
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ])
        content = resp["choices"][0]["message"]["content"]
        if "<verdict>equivalent</verdict>" in content.lower():
            return True
    except Exception:
        pass
    return False

def _schema_relation(candidate: SignatureCandidate, variant: SignatureVariantStats, controller: Any = None) -> Tuple[str, str]:
    if candidate.semantic_type != variant.semantic_type:
        return ("independent", "semantic_type mismatch")
    if candidate.task_family != variant.task_family:
        return ("independent", "task_family mismatch")
    slot_j = _jaccard_strings(candidate.required_slots, variant.required_slots)
    scope_j = _jaccard_strings(_scope_pairs(candidate.scope), _scope_pairs(variant.scope))
    text_overlap = lexical_overlap(candidate.canonical_text, variant.canonical_text, min_chars=3)
    support_j = _jaccard_strings(candidate.supporting_node_ids, variant.top_supporting_node_ids)
    same_slots = (
        set(str(x).strip().lower() for x in candidate.required_slots if str(x).strip())
        == set(str(x).strip().lower() for x in variant.required_slots if str(x).strip())
    )
    same_scope = scope_j >= 0.80 or (not _scope_pairs(candidate.scope) and not _scope_pairs(variant.scope))
    cand_neg, cand_aff = _polarity_profile(candidate.canonical_text)
    var_neg, var_aff = _polarity_profile(variant.canonical_text)
    opposite_polarity = text_overlap >= 0.55 and ((cand_neg and var_aff and not var_neg) or (var_neg and cand_aff and not cand_neg))
    if opposite_polarity:
        return ("contradicts", f"shared topic but opposite polarity (text_overlap={text_overlap:.2f})")
    if candidate.semantic_type == "strategy":
        if same_slots and same_scope and text_overlap >= 0.82:
            return ("equivalent", f"matching strategy structure with strong text overlap (text_overlap={text_overlap:.2f})")
    elif same_slots and same_scope and (text_overlap >= 0.88 or support_j >= 0.85):
        return ("equivalent", f"matching slots/scope with strong overlap (text_overlap={text_overlap:.2f}, support_overlap={support_j:.2f})")
    elif same_slots and same_scope and 0.80 <= text_overlap < 0.88:
        if controller and _judge_equivalent_vs_sibling(candidate, variant, controller):
            return ("equivalent", f"LLM judged as equivalent revision (text_overlap={text_overlap:.2f})")
    cand_slots = {str(x).strip().lower() for x in candidate.required_slots if str(x).strip()}
    var_slots = {str(x).strip().lower() for x in variant.required_slots if str(x).strip()}
    if (
        same_scope
        and cand_slots
        and var_slots
        and (cand_slots.issubset(var_slots) or var_slots.issubset(cand_slots))
        and (text_overlap >= 0.60 or support_j >= 0.50)
    ):
        return ("entails", f"same scope and one slot set subsumes the other (slot_overlap={slot_j:.2f}, support_overlap={support_j:.2f})")
    if text_overlap >= 0.40 or slot_j >= 0.50 or scope_j >= 0.50:
        return ("overlaps", f"same family with partial structural overlap (text_overlap={text_overlap:.2f}, slot_overlap={slot_j:.2f}, scope_overlap={scope_j:.2f})")
    return ("independent", f"low structural overlap (text_overlap={text_overlap:.2f}, slot_overlap={slot_j:.2f}, scope_overlap={scope_j:.2f})")


def judge_variant_relation(
    candidate: SignatureCandidate,
    variant: SignatureVariantStats,
    controller: Any,
) -> Tuple[str, str]:
    """LLM NLI judge for borderline sibling-variant comparison.

    Called when the symbolic heuristic gives 'overlaps' but similarity is in
    the ambiguous zone (0.40-0.75 text overlap). Returns (relation, rationale).
    Falls back to symbolic heuristic on any failure.
    """
    text_overlap = lexical_overlap(candidate.canonical_text, variant.canonical_text, min_chars=3)
    if text_overlap < 0.30 or text_overlap > 0.80:
        return _schema_relation(candidate, variant)

    user_msg = (
        f"## Variant A (candidate)\n"
        f"  semantic_type: {candidate.semantic_type}\n"
        f"  task_family: {candidate.task_family}\n"
        f"  text: {candidate.canonical_text[:400]}\n"
        f"  summary: {candidate.summary_text[:200]}\n"
        f"  required_slots: {', '.join(candidate.required_slots[:5])}\n\n"
        f"## Variant B (existing)\n"
        f"  semantic_type: {variant.semantic_type}\n"
        f"  task_family: {variant.task_family}\n"
        f"  text: {variant.canonical_text[:400]}\n"
        f"  summary: {variant.summary_text[:200]}\n"
        f"  required_slots: {', '.join(variant.required_slots[:5])}\n\n"
        f"## Context\n"
        f"  text_overlap: {text_overlap:.3f}\n"
        f"  slot_overlap: {_jaccard_strings(candidate.required_slots, variant.required_slots):.3f}\n"
        f"  scope_overlap: {_jaccard_strings(_scope_pairs(candidate.scope), _scope_pairs(variant.scope)):.3f}\n\n"
        f"Produce your <relation> now."
    )
    try:
        resp = controller.chat_oneshot([
            {"role": "system", "content": _NLI_JUDGE_SYSTEM},
            {"role": "user", "content": user_msg},
        ])
        content = resp["choices"][0]["message"]["content"]
    except Exception:
        return _schema_relation(candidate, variant)

    m = _NLI_RELATION_RE.search(content)
    if not m:
        return _schema_relation(candidate, variant)

    relation_type = m.group("type")
    rationale = m.group("rationale").strip()
    if relation_type == "equivalent":
        return ("equivalent", f"nli_judge: {rationale}")
    if relation_type == "contradicts":
        return ("contradicts", f"nli_judge: {rationale}")
    if relation_type == "sibling":
        return ("overlaps", f"nli_judge_sibling: {rationale}")
    return ("independent", f"nli_judge: {rationale}")


def score_event_impact(
    *,
    event_type: str,
    candidate: SignatureCandidate,
    variant: Optional[SignatureVariantStats],
    session_context: Dict[str, Any],
    controller: Any,
) -> Tuple[float, str]:
    """LLM-scored event impact for ambiguous events.

    Returns (impact_score, rationale). Falls back to deterministic bucket
    on any failure.
    """
    deterministic_events = {
        "promoted_to_review", "promoted_to_supported", "deprecated",
        "contradicted", "scoped_patch_reject",
    }
    if event_type in deterministic_events:
        bucket = IMPACT_BUCKET_WEIGHTS.get(
            {"promoted_to_review": "medium", "promoted_to_supported": "high",
             "deprecated": "high", "contradicted": "critical",
             "scoped_patch_reject": "high"}.get(event_type, "medium"),
            1.0,
        )
        return bucket, "deterministic"

    if variant is None:
        bucket = IMPACT_BUCKET_WEIGHTS.get("medium", 1.0)
        return bucket, "no_variant_context"

    user_msg = (
        f"## Event\n"
        f"  type: {event_type}\n"
        f"  task_family: {candidate.task_family}\n"
        f"  semantic_type: {candidate.semantic_type}\n"
        f"  variant_text: {candidate.canonical_text[:200]}\n\n"
        f"## Variant state\n"
        f"  promotion_state: {variant.promotion_state}\n"
        f"  support_score: {variant.support_score:.2f}\n"
        f"  stability_score: {variant.stability_score:.2f}\n"
        f"  risk_score: {variant.risk_score:.2f}\n"
        f"  contradiction_score: {variant.contradiction_score:.2f}\n"
        f"  session_count: {len(variant.session_ids)}\n"
        f"  evidence_count: {len(variant.evidence_fingerprints)}\n\n"
        f"## Session context\n"
        f"  finalized: {session_context.get('finalized', False)}\n"
        f"  execution_mode: {session_context.get('execution_mode', '')}\n"
        f"  answer_score: {session_context.get('answer_score', 'N/A')}\n\n"
        f"Produce your <impact> now."
    )
    try:
        resp = controller.chat_oneshot([
            {"role": "system", "content": _EVENT_IMPACT_SYSTEM},
            {"role": "user", "content": user_msg},
        ])
        content = resp["choices"][0]["message"]["content"]
    except Exception:
        bucket_name = "medium"
        return IMPACT_BUCKET_WEIGHTS.get(bucket_name, 1.0), "llm_fallback"

    m = _IMPACT_SCORE_RE.search(content)
    if not m:
        bucket_name = "medium"
        return IMPACT_BUCKET_WEIGHTS.get(bucket_name, 1.0), "parse_fallback"

    try:
        score = max(0.0, min(2.5, float(m.group("score"))))
    except (ValueError, TypeError):
        score = 1.0
    rationale = m.group("rationale").strip()
    return score, rationale


def _resolve_candidate_family(candidate: SignatureCandidate, index: SignatureStatsIndex) -> SignatureCandidate:
    candidate.matched_family_id = candidate.family_id
    if candidate.family_id in index.families:
        candidate.family_resolution = "exact_family"
        return candidate
    family_rows: List[Tuple[float, SignatureFamilyStats]] = []
    for family in index.families.values():
        score = _family_prefilter_score(candidate, family, index)
        if score <= 0.0:
            continue
        family_rows.append((score, family))
    family_rows.sort(key=lambda item: item[0], reverse=True)
    if not family_rows:
        candidate.family_resolution = "new_family"
        return candidate
    best_score, best_family = family_rows[0]
    candidate.family_match_score = best_score
    if best_score >= 1.15:
        candidate.matched_family_id = best_family.id
        candidate.family_id = best_family.id
        candidate.family_resolution = "family_alias"
        candidate.resolution_reason = f"aliased family to {best_family.id} via symbolic prefilter score {best_score:.2f}"
    else:
        candidate.family_resolution = "new_family"
    return candidate


def _resolve_candidate_variant(
    candidate: SignatureCandidate,
    index: SignatureStatsIndex,
    controller: Any = None,
) -> SignatureCandidate:
    candidate.matched_variant_id = candidate.variant_id
    family = index.families.get(candidate.family_id)
    if family is None:
        candidate.variant_resolution = "new_variant"
        return candidate
    if candidate.variant_id in index.variants:
        candidate.variant_resolution = "exact_variant"
        candidate.relation_to_match = "equivalent"
        return candidate
    rows: List[Tuple[float, SignatureVariantStats]] = []
    for vid in family.variant_ids:
        variant = index.variants.get(vid)
        if variant is None:
            continue
        score = _variant_prefilter_score(candidate, variant)
        if score <= 0.0:
            continue
        rows.append((score, variant))
    rows.sort(key=lambda item: item[0], reverse=True)
    if not rows:
        candidate.variant_resolution = "new_variant"
        return candidate
    best_relation = "independent"
    best_reason = ""
    best_match_id = ""
    best_match_score = 0.0
    relation_priority = {"equivalent": 4, "contradicts": 3, "entails": 2, "overlaps": 1, "independent": 0}
    for score, variant in rows[:3]:
        relation, reason = _schema_relation(candidate, variant, controller=controller)
        if relation == "overlaps" and controller is not None:
            text_overlap = lexical_overlap(candidate.canonical_text, variant.canonical_text, min_chars=3)
            if 0.30 <= text_overlap <= 0.80:
                relation, reason = judge_variant_relation(candidate, variant, controller)
        if relation_priority[relation] > relation_priority[best_relation] or (relation == best_relation and score > best_match_score):
            best_relation = relation
            best_reason = reason
            best_match_id = variant.id
            best_match_score = score
    candidate.variant_match_score = best_match_score
    candidate.matched_variant_id = best_match_id or candidate.variant_id
    candidate.relation_to_match = best_relation
    if best_relation == "equivalent" and best_match_id:
        candidate.variant_resolution = "equivalent_revision"
        candidate.variant_id = best_match_id
        candidate.resolution_reason = best_reason
    elif best_relation in {"overlaps", "entails", "contradicts"} and best_match_id:
        candidate.variant_resolution = "sibling_variant"
        if not candidate.resolution_reason:
            candidate.resolution_reason = best_reason
    else:
        candidate.variant_resolution = "new_variant"
        if not candidate.resolution_reason:
            candidate.resolution_reason = best_reason
    return candidate


def _relation_endpoints(
    candidate: SignatureCandidate,
    matched_variant: SignatureVariantStats,
) -> Optional[Tuple[str, str, bool]]:
    relation = str(candidate.relation_to_match or "")
    if relation not in {"overlaps", "entails", "contradicts"}:
        return None
    if relation in {"overlaps", "contradicts"}:
        src_variant_id, dst_variant_id = sorted([candidate.variant_id, matched_variant.id])
        return (src_variant_id, dst_variant_id, True)
    candidate_slots = {str(x).strip().lower() for x in candidate.required_slots if str(x or "").strip()}
    matched_slots = {str(x).strip().lower() for x in matched_variant.required_slots if str(x or "").strip()}
    if candidate_slots and matched_slots:
        if candidate_slots > matched_slots:
            return (candidate.variant_id, matched_variant.id, False)
        if matched_slots > candidate_slots:
            return (matched_variant.id, candidate.variant_id, False)
    return (candidate.variant_id, matched_variant.id, False)


def _upsert_variant_relation(
    *,
    index: SignatureStatsIndex,
    candidate: SignatureCandidate,
    session_id: str,
) -> Optional[str]:
    if candidate.variant_resolution != "sibling_variant":
        return None
    matched_variant = index.variants.get(candidate.matched_variant_id)
    current_variant = index.variants.get(candidate.variant_id)
    if matched_variant is None or current_variant is None:
        return None
    if matched_variant.family_id != current_variant.family_id:
        return None
    endpoints = _relation_endpoints(candidate, matched_variant)
    if endpoints is None:
        return None
    src_variant_id, dst_variant_id, symmetric = endpoints
    relation_id = _relation_record_id(
        family_id=current_variant.family_id,
        src_variant_id=src_variant_id,
        dst_variant_id=dst_variant_id,
        relation_type=candidate.relation_to_match,
        symmetric=symmetric,
    )
    relation = index.relations.get(relation_id)
    if relation is None:
        relation = SignatureRelationStats(
            id=relation_id,
            family_id=current_variant.family_id,
            src_variant_id=src_variant_id,
            dst_variant_id=dst_variant_id,
            relation_type=candidate.relation_to_match,
            symmetric=symmetric,
        )
        index.relations[relation_id] = relation
    relation.observation_count += 1
    relation.source_session_ids = _merge_lists(relation.source_session_ids, [session_id])
    if candidate.resolution_reason:
        relation.reasons = _merge_lists(relation.reasons, [candidate.resolution_reason])[:5]
    relation.last_match_score = max(float(relation.last_match_score or 0.0), float(candidate.variant_match_score or 0.0))
    relation.last_updated_at = _now_iso()
    return relation_id


def _resolve_candidate_against_index(
    candidate: SignatureCandidate,
    index: SignatureStatsIndex,
    controller: Any = None,
) -> SignatureCandidate:
    candidate.proposed_family_id = candidate.proposed_family_id or candidate.family_id
    candidate.proposed_variant_id = candidate.proposed_variant_id or candidate.variant_id
    candidate = _resolve_candidate_family(candidate, index)
    candidate = _resolve_candidate_variant(candidate, index, controller=controller)
    return candidate


def _resolution_summary(candidates: Mapping[str, SignatureCandidate]) -> Dict[str, Any]:
    family_counts: Dict[str, int] = {}
    variant_counts: Dict[str, int] = {}
    relation_counts: Dict[str, int] = {}
    for candidate in candidates.values():
        family_counts[candidate.family_resolution] = family_counts.get(candidate.family_resolution, 0) + 1
        variant_counts[candidate.variant_resolution] = variant_counts.get(candidate.variant_resolution, 0) + 1
        relation_counts[candidate.relation_to_match] = relation_counts.get(candidate.relation_to_match, 0) + 1
    return {
        "family_resolution_counts": family_counts,
        "variant_resolution_counts": variant_counts,
        "relation_counts": relation_counts,
    }


def collect_signature_candidates(
    *,
    index: Optional[SignatureStatsIndex],
    question: str,
    task_family: str,
    graph_edits: Sequence[Mapping[str, Any]],
    hypotheses: Mapping[str, Mapping[str, Any]],
    final_answer: str,
    cited_node_ids: Sequence[str],
    controller: Any = None,
) -> Dict[str, SignatureCandidate]:
    raw_candidates: List[SignatureCandidate] = []
    for edit in graph_edits:
        if str(edit.get("op", "")) != "add_node":
            continue
        node_type = str(edit.get("node_type", "") or "")
        if node_type == "strategy":
            cand = _strategy_candidate_from_edit(edit, question=question)
        elif node_type == "solved_subgoal":
            cand = _solved_subgoal_candidate_from_edit(edit)
        else:
            continue
        raw_candidates.append(cand)

    for hid, payload in hypotheses.items():
        text = str(payload.get("text", "") or "").strip()
        if not text or str(payload.get("verdict", "")) != "discarded":
            continue
        cand = _provisional_claim_candidate(
            text=text,
            question=question,
            task_family=task_family,
            evidence_node_ids=cited_node_ids,
            reason=f"discarded_hypothesis:{hid}",
        )
        raw_candidates.append(cand)

    for line in _answer_lines_outside_code(final_answer):
        low = normalize_text(line).lower()
        if not any(cue in low for cue in _HEDGE_CUES):
            continue
        cand = _provisional_claim_candidate(
            text=line,
            question=question,
            task_family=task_family,
            evidence_node_ids=cited_node_ids,
            reason="caveated_final_answer",
        )
        raw_candidates.append(cand)

    by_variant: Dict[str, SignatureCandidate] = {}
    for cand in raw_candidates:
        if index is not None:
            cand = _resolve_candidate_against_index(cand, index, controller=controller)
        existing = by_variant.get(cand.variant_id)
        if existing is None:
            by_variant[cand.variant_id] = cand
        else:
            _merge_candidate(existing, cand)
    return by_variant


def _answer_lines_outside_code(answer: str) -> List[str]:
    lines: List[str] = []
    in_code = False
    for raw in str(answer or "").splitlines():
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


def _make_event(
    *,
    session_id: str,
    question: str,
    candidate: SignatureCandidate,
    event_type: str,
    impact_bucket: str,
    event_reason: str,
    ambiguous: bool = False,
    impact_multiplier: float = 1.0,
    evidence_node_ids: Sequence[str] = (),
    linked_node_ids: Sequence[str] = (),
    affected_node_ids: Sequence[str] = (),
    affected_final_answer: bool = False,
    output_caveated: bool = False,
    metadata: Optional[Mapping[str, Any]] = None,
    index: int = 0,
) -> SignatureEvent:
    if event_type not in EVENT_TYPES:
        raise ValueError(f"unknown signature event_type: {event_type}")
    bucket = impact_bucket if impact_bucket in IMPACT_BUCKET_WEIGHTS else "medium"
    ev_ids = _stable_list(evidence_node_ids)
    meta = {
        "proposed_text": candidate.canonical_text,
        "proposed_variant_id": candidate.proposed_variant_id or candidate.variant_id,
        "proposed_family_id": candidate.proposed_family_id or candidate.family_id,
        "variant_resolution": candidate.variant_resolution,
        "family_resolution": candidate.family_resolution,
        "relation_to_match": candidate.relation_to_match,
    }
    meta.update(dict(metadata or {}))
    return SignatureEvent(
        event_id=f"sig_evt_{session_id}_{index:03d}_{_slug(event_type)}",
        session_id=session_id,
        question=question,
        question_fingerprint=_question_fingerprint(question),
        signature_family_id=candidate.family_id,
        signature_variant_id=candidate.variant_id,
        semantic_type=candidate.semantic_type,
        event_type=event_type,
        impact_bucket=bucket,
        event_reason=event_reason,
        task_family=candidate.task_family,
        evidence_node_ids=ev_ids,
        linked_node_ids=_stable_list(linked_node_ids),
        affected_node_ids=_stable_list(affected_node_ids),
        affected_final_answer=affected_final_answer,
        output_caveated=output_caveated,
        evidence_fingerprint=_evidence_fingerprint(ev_ids),
        metadata=meta,
    )


def collect_signature_events(
    *,
    session_id: str,
    question: str,
    task_family: str,
    candidates: Mapping[str, SignatureCandidate],
    hypotheses: Mapping[str, Mapping[str, Any]],
    scoped_patches: Sequence[Mapping[str, Any]],
    final_answer: str,
    finalized: bool,
    execution_mode: str,
    design_evidence_gate_rounds: int = 0,
) -> List[SignatureEvent]:
    events: List[SignatureEvent] = []
    by_node_id: Dict[str, SignatureCandidate] = {}
    for cand in candidates.values():
        for node_id in cand.source_node_ids:
            by_node_id[node_id] = cand

    next_index = 0
    for cand in candidates.values():
        if not finalized:
            continue
        if cand.semantic_type == "strategy":
            events.append(_make_event(
                session_id=session_id,
                question=question,
                candidate=cand,
                event_type="supported_finalize",
                impact_bucket="medium",
                event_reason=f"successful session produced reusable strategy via {execution_mode or 'full_loop'}",
                evidence_node_ids=cand.supporting_node_ids,
                linked_node_ids=cand.linked_node_ids,
                affected_node_ids=cand.source_node_ids,
                affected_final_answer=True,
                metadata={"execution_mode": execution_mode},
                index=next_index,
            ))
            next_index += 1
        elif cand.semantic_type == "solved_subgoal":
            event_type = "supported_reuse" if execution_mode.startswith("micro_controller") else "supported_finalize"
            impact_bucket = "high" if execution_mode.startswith("micro_controller") else "medium"
            events.append(_make_event(
                session_id=session_id,
                question=question,
                candidate=cand,
                event_type=event_type,
                impact_bucket=impact_bucket,
                event_reason=f"final answer grounded in solved_subgoal path via {execution_mode or 'full_loop'}",
                evidence_node_ids=cand.supporting_node_ids,
                linked_node_ids=cand.linked_node_ids,
                affected_node_ids=cand.source_node_ids,
                affected_final_answer=True,
                metadata={"execution_mode": execution_mode},
                index=next_index,
            ))
            next_index += 1

    for hid, payload in hypotheses.items():
        text = str(payload.get("text", "") or "").strip()
        if not text or str(payload.get("verdict", "")) != "discarded":
            continue
        cand = next((c for c in candidates.values() if c.semantic_type == "provisional_claim" and normalize_text(c.canonical_text).lower() == normalize_text(text).lower()), None)
        if cand is None:
            continue
        events.append(_make_event(
            session_id=session_id,
            question=question,
            candidate=cand,
            event_type="hypothesis_discarded",
            impact_bucket="medium",
            event_reason=str(payload.get("evidence", "") or f"hypothesis {hid} was discarded"),
            evidence_node_ids=cand.supporting_node_ids,
            linked_node_ids=cand.linked_node_ids,
            affected_final_answer=False,
            output_caveated=False,
            metadata={"hypothesis_id": hid},
            index=next_index,
        ))
        next_index += 1

    caveated_lines = {
        normalize_text(line).lower()
        for line in _answer_lines_outside_code(final_answer)
        if any(cue in normalize_text(line).lower() for cue in _HEDGE_CUES)
    }
    for cand in candidates.values():
        if cand.semantic_type != "provisional_claim":
            continue
        norm = normalize_text(cand.canonical_text).lower()
        if norm not in caveated_lines:
            continue
        events.append(_make_event(
            session_id=session_id,
            question=question,
            candidate=cand,
            event_type="provisional_used_with_caveat",
            impact_bucket="low",
            event_reason="final answer retained this detail only as a caveated hypothesis",
            evidence_node_ids=cand.supporting_node_ids,
            linked_node_ids=cand.linked_node_ids,
            affected_final_answer=True,
            output_caveated=True,
            metadata={"execution_mode": execution_mode},
            index=next_index,
        ))
        next_index += 1
        if design_evidence_gate_rounds > 0:
            events.append(_make_event(
                session_id=session_id,
                question=question,
                candidate=cand,
                event_type="answer_gate_rewrite",
                impact_bucket="medium",
                event_reason="design evidence gate forced this detail to remain tentative",
                evidence_node_ids=cand.supporting_node_ids,
                linked_node_ids=cand.linked_node_ids,
                affected_final_answer=True,
                output_caveated=True,
                metadata={"design_evidence_gate_rounds": design_evidence_gate_rounds},
                index=next_index,
            ))
            next_index += 1

    for patch in scoped_patches:
        target_id = str(patch.get("target_id", "") or "")
        cand = by_node_id.get(target_id)
        if cand is None:
            continue
        validation = patch.get("validation") if isinstance(patch.get("validation"), Mapping) else {}
        status = str(validation.get("status", "") or "")
        event_type = _STATUS_EVENT_BY_PATCH.get(status)
        if not event_type:
            continue
        reasons = list(validation.get("reasons", []))
        warnings = list(validation.get("warnings", []))
        impact_bucket = "medium"
        if status in {"needs_review", "reject"}:
            impact_bucket = "high"
        elif status == "soft_only":
            impact_bucket = "low"
        events.append(_make_event(
            session_id=session_id,
            question=question,
            candidate=cand,
            event_type=event_type,
            impact_bucket=impact_bucket,
            event_reason="; ".join([str(x) for x in (reasons + warnings) if x]) or f"scoped patch status={status}",
            evidence_node_ids=list(patch.get("evidence_node_ids", [])),
            linked_node_ids=cand.linked_node_ids,
            affected_node_ids=list(patch.get("affected_node_ids", [])),
            metadata={"patch_id": patch.get("patch_id", ""), "patch_type": patch.get("patch_type", "")},
            index=next_index,
        ))
        next_index += 1
        if any(str(w).startswith("low_relevance_") for w in warnings):
            events.append(_make_event(
                session_id=session_id,
                question=question,
                candidate=cand,
                event_type="low_relevance_retrieval",
                impact_bucket="medium",
                event_reason="; ".join(str(w) for w in warnings if str(w).startswith("low_relevance_")),
                evidence_node_ids=list(patch.get("evidence_node_ids", [])),
                linked_node_ids=cand.linked_node_ids,
                affected_node_ids=list(patch.get("affected_node_ids", [])),
                metadata={"patch_id": patch.get("patch_id", "")},
                index=next_index,
            ))
            next_index += 1

    return events


def _default_variant_record(candidate: SignatureCandidate) -> SignatureVariantStats:
    return SignatureVariantStats(
        id=candidate.variant_id,
        family_id=candidate.family_id,
        semantic_type=candidate.semantic_type,
        canonical_text=candidate.canonical_text,
        summary_text=candidate.summary_text,
        task_family=candidate.task_family,
        task_subtype=candidate.task_subtype,
        question_mode=candidate.question_mode,
        epistemic_status=candidate.epistemic_status,
        promotion_state=candidate.promotion_state,
        retrieval_tier=candidate.retrieval_tier,
        source_node_ids=list(candidate.source_node_ids),
        linked_node_ids=list(candidate.linked_node_ids),
        top_supporting_node_ids=list(candidate.supporting_node_ids),
        required_slots=list(candidate.required_slots),
        scope=dict(candidate.scope),
        aliases=[candidate.summary_text] if candidate.summary_text and candidate.summary_text != candidate.canonical_text else [],
        effective_support_score=0.0,
        effective_stability_score=0.0,
        effective_risk_score=0.0,
        effective_contradiction_score=0.0,
        effective_bias_score=0.0,
        last_updated_at=_now_iso(),
    )


def _default_family_record(candidate: SignatureCandidate) -> SignatureFamilyStats:
    return SignatureFamilyStats(
        id=candidate.family_id,
        semantic_type=candidate.semantic_type,
        family_label=candidate.family_label,
        task_family=candidate.task_family,
        variant_ids=[candidate.variant_id],
        last_updated_at=_now_iso(),
    )


def _task_family_weights(task_family: str) -> Dict[str, float]:
    if task_family == "design_synthesis":
        return {"support": 0.35, "stability": 0.25, "risk": 0.45, "contradiction": 0.75, "provisional_penalty": 0.35}
    if task_family == "algorithm_applicability":
        return {"support": 0.55, "stability": 0.35, "risk": 0.25, "contradiction": 0.60, "provisional_penalty": 0.20}
    return {"support": 0.45, "stability": 0.30, "risk": 0.35, "contradiction": 0.65, "provisional_penalty": 0.25}


def _bias_from_scores(
    *,
    task_family: str,
    epistemic_status: str,
    retrieval_tier: str,
    support_score: float,
    stability_score: float,
    risk_score: float,
    contradiction_score: float,
) -> float:
    weights = _task_family_weights(task_family)
    bias = (
        support_score * weights["support"]
        + stability_score * weights["stability"]
        - risk_score * weights["risk"]
        - contradiction_score * weights["contradiction"]
    )
    if epistemic_status == "provisional":
        bias -= weights["provisional_penalty"]
    if retrieval_tier == "audit_only":
        bias -= 1.50
    return round(bias, 4)


def _bias_for(variant: SignatureVariantStats) -> float:
    return _bias_from_scores(
        task_family=variant.task_family,
        epistemic_status=variant.epistemic_status,
        retrieval_tier=variant.retrieval_tier,
        support_score=variant.support_score,
        stability_score=variant.stability_score,
        risk_score=variant.risk_score,
        contradiction_score=variant.contradiction_score,
    )


def _relation_neighbor_rows(index: SignatureStatsIndex, variant: SignatureVariantStats) -> List[Tuple[SignatureRelationStats, SignatureVariantStats, str]]:
    rows: List[Tuple[SignatureRelationStats, SignatureVariantStats, str]] = []
    for relation in index.relations.values():
        if relation.family_id != variant.family_id:
            continue
        if relation.src_variant_id == variant.id:
            neighbor = index.variants.get(relation.dst_variant_id)
            if neighbor is not None:
                rows.append((relation, neighbor, "outgoing"))
        elif relation.dst_variant_id == variant.id:
            neighbor = index.variants.get(relation.src_variant_id)
            if neighbor is not None:
                rows.append((relation, neighbor, "incoming"))
    return rows


def _relation_propagation_for_variant(index: SignatureStatsIndex, variant: SignatureVariantStats) -> Dict[str, Any]:
    support = 0.0
    stability = 0.0
    risk = 0.0
    contradiction = 0.0
    relation_counts: Dict[str, int] = {}
    evidence: List[Dict[str, Any]] = []
    for relation, neighbor, direction in _relation_neighbor_rows(index, variant):
        observation_scale = min(1.0, 0.35 * max(1, int(relation.observation_count or 0)))
        relation_counts[relation.relation_type] = int(relation_counts.get(relation.relation_type, 0)) + 1
        row_support = 0.0
        row_stability = 0.0
        row_risk = 0.0
        row_contradiction = 0.0
        neighbor_supported = neighbor.epistemic_status == "supported" and neighbor.retrieval_tier == "normal"
        confidence_boost = 1.3 if neighbor_supported else 1.0
        if relation.relation_type == "overlaps":
            row_support += neighbor.support_score * 0.15 * confidence_boost
            row_stability += neighbor.stability_score * 0.12 * confidence_boost
            row_risk += neighbor.risk_score * 0.05
        elif relation.relation_type == "entails":
            if direction == "incoming":
                row_support += neighbor.support_score * 0.35 * confidence_boost
                row_stability += neighbor.stability_score * 0.25 * confidence_boost
            else:
                row_support += neighbor.support_score * 0.08 * confidence_boost
                row_stability += neighbor.stability_score * 0.05 * confidence_boost
        elif relation.relation_type == "contradicts":
            row_risk += max(neighbor.risk_score * 0.12, 0.15 * observation_scale)
            row_contradiction += max(neighbor.support_score * 0.08, 0.40 * observation_scale)
        support += row_support
        stability += row_stability
        risk += row_risk
        contradiction += row_contradiction
        evidence.append({
            "relation_id": relation.id,
            "relation_type": relation.relation_type,
            "direction": direction,
            "neighbor_variant_id": neighbor.id,
            "support_delta": round(row_support, 4),
            "stability_delta": round(row_stability, 4),
            "risk_delta": round(row_risk, 4),
            "contradiction_delta": round(row_contradiction, 4),
        })
    return {
        "support": round(support, 4),
        "stability": round(stability, 4),
        "risk": round(risk, 4),
        "contradiction": round(contradiction, 4),
        "relation_counts": relation_counts,
        "evidence": evidence,
    }


def _refresh_effective_variant_metrics(index: SignatureStatsIndex, family_id: str) -> None:
    family = index.families.get(family_id)
    if family is None:
        return
    for variant_id in family.variant_ids:
        variant = index.variants.get(variant_id)
        if variant is None:
            continue
        propagated = _relation_propagation_for_variant(index, variant)
        variant.propagated_support_score = float(propagated["support"])
        variant.propagated_stability_score = float(propagated["stability"])
        variant.propagated_risk_score = float(propagated["risk"])
        variant.propagated_contradiction_score = float(propagated["contradiction"])
        variant.effective_support_score = round(variant.support_score + variant.propagated_support_score, 4)
        variant.effective_stability_score = round(variant.stability_score + variant.propagated_stability_score, 4)
        variant.effective_risk_score = round(variant.risk_score + variant.propagated_risk_score, 4)
        variant.effective_contradiction_score = round(variant.contradiction_score + variant.propagated_contradiction_score, 4)
        variant.relation_counts = dict(propagated["relation_counts"])
        variant.effective_bias_score = _bias_from_scores(
            task_family=variant.task_family,
            epistemic_status=variant.epistemic_status,
            retrieval_tier=variant.retrieval_tier,
            support_score=variant.effective_support_score,
            stability_score=variant.effective_stability_score,
            risk_score=variant.effective_risk_score,
            contradiction_score=variant.effective_contradiction_score,
        )
        if variant.relation_counts.get("contradicts", 0) > 0:
            variant.promotion_state = "blocked"
            if variant.effective_contradiction_score >= 1.25:
                variant.retrieval_tier = "audit_only"
            elif variant.retrieval_tier == "normal":
                variant.retrieval_tier = "gated"
            variant.effective_bias_score = _bias_from_scores(
                task_family=variant.task_family,
                epistemic_status=variant.epistemic_status,
                retrieval_tier=variant.retrieval_tier,
                support_score=variant.effective_support_score,
                stability_score=variant.effective_stability_score,
                risk_score=variant.effective_risk_score,
                contradiction_score=variant.effective_contradiction_score,
            )


def _should_count_positive(event_type: str) -> bool:
    base = EVENT_BASE_DELTAS.get(event_type, {})
    return (base.get("support", 0.0) + base.get("stability", 0.0)) >= (base.get("risk", 0.0) + base.get("contradiction", 0.0))


def _distinct_count(values: Sequence[str]) -> int:
    return len({str(value or "").strip() for value in values if str(value or "").strip()})


def _promotion_snapshot(variant: SignatureVariantStats) -> Dict[str, Any]:
    event_counts = dict(variant.event_counts or {})
    return {
        "distinct_sessions": _distinct_count(variant.session_ids),
        "distinct_questions": _distinct_count(variant.question_fingerprints),
        "distinct_evidence": _distinct_count(variant.evidence_fingerprints),
        "support_nodes": _distinct_count(variant.top_supporting_node_ids),
        "success_events": int(event_counts.get("supported_reuse", 0)) + int(event_counts.get("supported_finalize", 0)),
        "accept_events": int(event_counts.get("scoped_patch_accept", 0)),
        "soft_accept_events": int(event_counts.get("scoped_patch_soft_only", 0)),
        "needs_review_events": int(event_counts.get("scoped_patch_needs_review", 0)),
        "reject_events": int(event_counts.get("scoped_patch_reject", 0)),
        "low_relevance_events": int(event_counts.get("low_relevance_retrieval", 0)),
        "answer_gate_rewrite_events": int(event_counts.get("answer_gate_rewrite", 0)),
        "caveated_events": int(event_counts.get("provisional_used_with_caveat", 0)),
        "contradicted_events": int(event_counts.get("contradicted", 0)),
    }


def _promotion_block_reason(variant: SignatureVariantStats, snapshot: Mapping[str, Any]) -> str:
    if variant.contradiction_score >= 1.0 or variant.effective_contradiction_score >= 1.0:
        return "contradiction_pressure"
    if int(snapshot.get("reject_events", 0) or 0) > 0:
        return "scoped_reject"
    if int(snapshot.get("contradicted_events", 0) or 0) > 0:
        return "contradicted"
    if variant.semantic_type in {"strategy", "provisional_claim"} and int(snapshot.get("needs_review_events", 0) or 0) > 0:
        return "needs_review"
    if variant.semantic_type == "strategy" and int(snapshot.get("low_relevance_events", 0) or 0) > 0:
        return "low_relevance"
    if variant.semantic_type == "provisional_claim" and int(snapshot.get("answer_gate_rewrite_events", 0) or 0) > 0:
        return "answer_gate_rewrite"
    return ""


def _review_thresholds(variant: SignatureVariantStats) -> Dict[str, float]:
    if variant.semantic_type == "solved_subgoal":
        return {"support": 1.0, "stability": 1.0, "risk": 0.55, "sessions": 1, "questions": 1, "evidence": 1, "successes": 1}
    if variant.semantic_type == "strategy":
        return {"support": 1.2, "stability": 1.25, "risk": 0.45, "sessions": 2, "questions": 2, "evidence": 1, "successes": 2}
    return {"support": 1.1, "stability": 0.9, "risk": 0.35, "sessions": 2, "questions": 2, "evidence": 2, "successes": 0}


def _supported_thresholds(variant: SignatureVariantStats) -> Dict[str, float]:
    if variant.semantic_type == "solved_subgoal":
        return {"support": 2.2, "stability": 2.0, "risk": 0.45, "sessions": 2, "questions": 2, "evidence": 1, "successes": 2}
    if variant.semantic_type == "strategy":
        return {"support": 2.6, "stability": 2.4, "risk": 0.30, "sessions": 3, "questions": 3, "evidence": 2, "successes": 3}
    return {"support": 9.9, "stability": 9.9, "risk": 0.0, "sessions": 99, "questions": 99, "evidence": 99, "successes": 99}


def _qualifies_for_review(variant: SignatureVariantStats, snapshot: Mapping[str, Any]) -> bool:
    if _promotion_block_reason(variant, snapshot):
        return False
    thresholds = _review_thresholds(variant)
    if variant.support_score < thresholds["support"] or variant.stability_score < thresholds["stability"]:
        return False
    if variant.risk_score > thresholds["risk"]:
        return False
    if int(snapshot.get("distinct_sessions", 0) or 0) < int(thresholds["sessions"]):
        return False
    if int(snapshot.get("distinct_questions", 0) or 0) < int(thresholds["questions"]):
        return False
    if int(snapshot.get("distinct_evidence", 0) or 0) < int(thresholds["evidence"]):
        return False
    if int(snapshot.get("success_events", 0) or 0) < int(thresholds["successes"]):
        return False
    if variant.semantic_type == "provisional_claim" and int(snapshot.get("accept_events", 0) or 0) <= 0:
        return False
    return True


def _qualifies_for_supported(variant: SignatureVariantStats, snapshot: Mapping[str, Any]) -> bool:
    if variant.promotion_state not in {"review", "supported"}:
        return False
    if _promotion_block_reason(variant, snapshot):
        return False
    thresholds = _supported_thresholds(variant)
    if variant.support_score < thresholds["support"] or variant.stability_score < thresholds["stability"]:
        return False
    if variant.risk_score > thresholds["risk"]:
        return False
    if variant.effective_contradiction_score > 0.45:
        return False
    if int(snapshot.get("distinct_sessions", 0) or 0) < int(thresholds["sessions"]):
        return False
    if int(snapshot.get("distinct_questions", 0) or 0) < int(thresholds["questions"]):
        return False
    if int(snapshot.get("distinct_evidence", 0) or 0) < int(thresholds["evidence"]):
        return False
    if int(snapshot.get("success_events", 0) or 0) < int(thresholds["successes"]):
        return False
    if int(snapshot.get("accept_events", 0) or 0) <= 0:
        return False
    if variant.semantic_type == "strategy" and int(snapshot.get("soft_accept_events", 0) or 0) > 0:
        return False
    if variant.semantic_type == "provisional_claim":
        return False
    return True


def _promotion_target_state(variant: SignatureVariantStats) -> Tuple[str, str]:
    if variant.semantic_type == "solved_subgoal":
        return ("supported", "normal")
    if variant.semantic_type == "strategy":
        return ("supported", "gated")
    return ("provisional", "gated")


def _update_variant_from_event(variant: SignatureVariantStats, event: SignatureEvent) -> None:
    base = EVENT_BASE_DELTAS.get(event.event_type, {})
    weight = IMPACT_BUCKET_WEIGHTS.get(event.impact_bucket, 1.0) * getattr(event, 'impact_multiplier', 1.0)
    variant.support_score = round(variant.support_score + base.get("support", 0.0) * weight, 4)
    variant.stability_score = round(variant.stability_score + base.get("stability", 0.0) * weight, 4)
    variant.risk_score = round(max(0.0, variant.risk_score + base.get("risk", 0.0) * weight), 4)
    variant.contradiction_score = round(max(0.0, variant.contradiction_score + base.get("contradiction", 0.0) * weight), 4)
    if _should_count_positive(event.event_type):
        variant.positive_events += 1
    else:
        variant.negative_events += 1
    variant.event_counts[event.event_type] = int(variant.event_counts.get(event.event_type, 0)) + 1
    variant.source_node_ids = _merge_lists(variant.source_node_ids, event.affected_node_ids)
    variant.linked_node_ids = _merge_lists(variant.linked_node_ids, event.linked_node_ids)
    variant.top_supporting_node_ids = _merge_lists(variant.top_supporting_node_ids, event.evidence_node_ids)
    variant.session_ids = _merge_lists(variant.session_ids, [event.session_id])
    variant.question_fingerprints = _merge_lists(variant.question_fingerprints, [event.question_fingerprint])
    if event.evidence_fingerprint:
        variant.evidence_fingerprints = _merge_lists(variant.evidence_fingerprints, [event.evidence_fingerprint])
    if event.output_caveated:
        variant.retrieval_tier = "gated"
    if event.event_type == "promoted_to_review":
        variant.promotion_state = "review"
        if variant.semantic_type == "solved_subgoal":
            variant.retrieval_tier = "normal"
        elif variant.retrieval_tier != "audit_only":
            variant.retrieval_tier = "gated"
    if event.event_type == "promoted_to_supported":
        target_status, target_tier = _promotion_target_state(variant)
        variant.promotion_state = "supported"
        variant.epistemic_status = target_status
        variant.retrieval_tier = target_tier
    elif event.event_type in {"scoped_patch_accept", "promoted_to_supported"} and variant.semantic_type == "solved_subgoal":
        variant.promotion_state = "review"
        if variant.epistemic_status != "provisional":
            variant.retrieval_tier = "normal"
    if event.event_type in {"scoped_patch_needs_review", "scoped_patch_reject", "deprecated"}:
        variant.promotion_state = "blocked"
    if event.event_type in {"contradicted", "deprecated"} and variant.contradiction_score >= 1.5:
        variant.retrieval_tier = "audit_only"
    proposed_text = str(event.metadata.get("proposed_text", "") or "")
    if proposed_text and proposed_text not in variant.aliases and proposed_text != variant.canonical_text:
        variant.aliases = _merge_lists(variant.aliases, [proposed_text])
    variant.bias_score = _bias_for(variant)
    variant.last_updated_at = _now_iso()


def _dominance_thresholds(task_family: str) -> Dict[str, float]:
    if task_family == "design_synthesis":
        return {"support": 2.0, "stability": 1.0, "margin": 0.75, "max_contradiction": 0.5}
    if task_family == "algorithm_applicability":
        return {"support": 1.5, "stability": 0.75, "margin": 0.50, "max_contradiction": 0.5}
    return {"support": 2.5, "stability": 1.0, "margin": 1.0, "max_contradiction": 0.5}


def _dominant_variant_id(family: SignatureFamilyStats, index: SignatureStatsIndex) -> Optional[str]:
    variants = [index.variants.get(vid) for vid in family.variant_ids]
    variants = [v for v in variants if v is not None]
    if not variants:
        return None
    ranked = sorted(
        variants,
        key=lambda v: (v.effective_bias_score, v.effective_support_score, v.effective_stability_score, v.bias_score),
        reverse=True,
    )
    leader = ranked[0]
    thresholds = _dominance_thresholds(family.task_family)
    if leader.epistemic_status != "supported":
        return None
    if leader.effective_support_score < thresholds["support"] or leader.effective_stability_score < thresholds["stability"]:
        return None
    if leader.effective_contradiction_score > thresholds["max_contradiction"]:
        return None
    runner_up = ranked[1] if len(ranked) > 1 else None
    if runner_up is not None and (leader.effective_bias_score - runner_up.effective_bias_score) < thresholds["margin"]:
        return None
    return leader.id


def _candidate_from_variant(variant: SignatureVariantStats, family: Optional[SignatureFamilyStats]) -> SignatureCandidate:
    return SignatureCandidate(
        family_id=variant.family_id,
        variant_id=variant.id,
        semantic_type=variant.semantic_type,
        family_label=family.family_label if family is not None else variant.family_id,
        canonical_text=variant.canonical_text,
        summary_text=variant.summary_text,
        task_family=variant.task_family,
        task_subtype=variant.task_subtype,
        question_mode=variant.question_mode,
        epistemic_status=variant.epistemic_status,
        promotion_state=variant.promotion_state,
        retrieval_tier=variant.retrieval_tier,
        source_node_ids=list(variant.source_node_ids),
        linked_node_ids=list(variant.linked_node_ids),
        supporting_node_ids=list(variant.top_supporting_node_ids),
        required_slots=list(variant.required_slots),
        scope=dict(variant.scope),
    )


def _promotion_event_reason(target: str, snapshot: Mapping[str, Any], variant: SignatureVariantStats) -> str:
    return (
        f"automatic_{target}: sessions={int(snapshot.get('distinct_sessions', 0) or 0)}, "
        f"questions={int(snapshot.get('distinct_questions', 0) or 0)}, "
        f"evidence={int(snapshot.get('distinct_evidence', 0) or 0)}, "
        f"support={variant.support_score:.2f}, stability={variant.stability_score:.2f}, "
        f"risk={variant.risk_score:.2f}, contradiction={variant.effective_contradiction_score:.2f}"
    )


def _make_promotion_event(
    *,
    variant: SignatureVariantStats,
    family: Optional[SignatureFamilyStats],
    session_id: str,
    question: str,
    event_type: str,
    impact_bucket: str,
    event_reason: str,
    metadata: Mapping[str, Any],
    index: int,
) -> SignatureEvent:
    candidate = _candidate_from_variant(variant, family)
    return _make_event(
        session_id=session_id,
        question=question,
        candidate=candidate,
        event_type=event_type,
        impact_bucket=impact_bucket,
        event_reason=event_reason,
        ambiguous=False,
        impact_multiplier=1.0,
        evidence_node_ids=variant.top_supporting_node_ids,
        linked_node_ids=variant.linked_node_ids,
        affected_node_ids=variant.source_node_ids,
        metadata=dict(metadata),
        index=index,
    )


def _derive_promotion_events(
    *,
    index: SignatureStatsIndex,
    variant_ids: Sequence[str],
    session_id: str,
    question: str,
    start_index: int,
) -> List[SignatureEvent]:
    events: List[SignatureEvent] = []
    next_index = start_index
    for variant_id in _stable_list(variant_ids):
        variant = index.variants.get(variant_id)
        if variant is None:
            continue
        family = index.families.get(variant.family_id)
        snapshot = _promotion_snapshot(variant)
        block_reason = _promotion_block_reason(variant, snapshot)
        if block_reason:
            continue
        if variant.promotion_state == "blocked" and _qualifies_for_review(variant, snapshot):
            events.append(_make_promotion_event(
                variant=variant,
                family=family,
                session_id=session_id,
                question=question,
                event_type="promoted_to_review",
                impact_bucket="medium",
                event_reason=_promotion_event_reason("review", snapshot, variant),
                metadata={
                    "promotion_target": "review",
                    "snapshot": dict(snapshot),
                },
                index=next_index,
            ))
            next_index += 1
            continue
        if (
            variant.promotion_state in {"review", "supported"}
            and variant.promotion_state != "supported"
            and _qualifies_for_supported(variant, snapshot)
        ):
            target_status, target_tier = _promotion_target_state(variant)
            events.append(_make_promotion_event(
                variant=variant,
                family=family,
                session_id=session_id,
                question=question,
                event_type="promoted_to_supported",
                impact_bucket="high",
                event_reason=_promotion_event_reason("supported", snapshot, variant),
                metadata={
                    "promotion_target": "supported",
                    "target_epistemic_status": target_status,
                    "target_retrieval_tier": target_tier,
                    "snapshot": dict(snapshot),
                },
                index=next_index,
            ))
            next_index += 1
    return events


def _family_contested(family: SignatureFamilyStats, index: SignatureStatsIndex) -> bool:
    relation_conflicts = set()
    family_variant_ids = set(str(vid) for vid in family.variant_ids if str(vid or "").strip())
    for relation in index.relations.values():
        if relation.family_id != family.id or relation.relation_type != "contradicts":
            continue
        if relation.observation_count <= 0:
            continue
        if relation.src_variant_id in family_variant_ids and relation.dst_variant_id in family_variant_ids:
            relation_conflicts.update([relation.src_variant_id, relation.dst_variant_id])
    variants = [index.variants.get(vid) for vid in family.variant_ids]
    active = [v for v in variants if v is not None and v.negative_events > 0 and v.effective_contradiction_score > 0.0]
    if len(relation_conflicts) < 2 and len(active) < 2:
        return False
    return _dominant_variant_id(family, index) is None


def _refresh_family(index: SignatureStatsIndex, family_id: str) -> None:
    family = index.families.get(family_id)
    if family is None:
        return
    _refresh_effective_variant_metrics(index, family_id)
    variants = [index.variants[vid] for vid in family.variant_ids if vid in index.variants]
    if not variants:
        return
    family.support_score = round(max(v.support_score for v in variants), 4)
    family.stability_score = round(max(v.stability_score for v in variants), 4)
    family.risk_score = round(max(v.risk_score for v in variants), 4)
    family.contradiction_score = round(max(v.contradiction_score for v in variants), 4)
    family.bias_score = round(max(v.bias_score for v in variants), 4)
    family.effective_support_score = round(max(v.effective_support_score for v in variants), 4)
    family.effective_stability_score = round(max(v.effective_stability_score for v in variants), 4)
    family.effective_risk_score = round(max(v.effective_risk_score for v in variants), 4)
    family.effective_contradiction_score = round(max(v.effective_contradiction_score for v in variants), 4)
    family.effective_bias_score = round(max(v.effective_bias_score for v in variants), 4)
    counts: Dict[str, int] = {}
    relation_counts: Dict[str, int] = {}
    for variant in variants:
        for key, value in variant.event_counts.items():
            counts[key] = counts.get(key, 0) + int(value)
        for key, value in variant.relation_counts.items():
            relation_counts[key] = relation_counts.get(key, 0) + int(value)
    family.event_counts = counts
    family.relation_counts = relation_counts
    family.dominant_variant_id = _dominant_variant_id(family, index)
    family.contested = _family_contested(family, index)
    family.last_updated_at = _now_iso()



def _score_ambiguous_event_batch(
    events: List[SignatureEvent],
    candidates: Mapping[str, SignatureCandidate],
    controller: Any,
) -> None:
    if not controller:
        return
    ambiguous_events = [e for e in events if e.ambiguous]
    if not ambiguous_events:
        return
    
    system_prompt = """You are an expert judge evaluating the impact of reasoning events.
For each event, evaluate its context and return an impact multiplier between 0.5 and 1.5.
For example, a 'supported_finalize' on a complex 6-step problem is stronger (1.5) than on a trivial 1-step direct lookup (0.5).
Return ONLY valid JSON like: {"event_id_1": 1.5, "event_id_2": 0.5}.
"""
    
    user_prompt = "EVENTS:\n"
    for ev in ambiguous_events:
        cand = candidates.get(ev.signature_variant_id)
        text = cand.canonical_text if cand else "Unknown"
        user_prompt += f"- ID: {ev.event_id}\n  Type: {ev.event_type}\n  Variant: {text}\n  Reason: {ev.event_reason}\n\n"
        
    try:
        import json
        chat_fn = getattr(controller, "chat_oneshot", getattr(controller, "chat", None))
        if not chat_fn:
            return
        resp = chat_fn([
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ])
        content = resp["choices"][0]["message"]["content"]
        # Very simple extraction of JSON
        start = content.find("{")
        end = content.rfind("}")
        if start != -1 and end != -1:
            data = json.loads(content[start:end+1])
            for ev in ambiguous_events:
                if ev.event_id in data:
                    ev.impact_multiplier = max(0.5, min(1.5, float(data[ev.event_id])))
    except Exception:
        pass

def apply_signature_events(
    *,
    index: SignatureStatsIndex,
    candidates: Mapping[str, SignatureCandidate],
    events: Sequence[SignatureEvent],
    controller: Any = None,
) -> Dict[str, Any]:
    created_families: List[str] = []
    created_variants: List[str] = []
    created_relations: List[str] = []
    touched_variants: List[str] = []
    touched_families: List[str] = []
    touched_relations: List[str] = []
    promotion_events: List[SignatureEvent] = []
    session_ids = _stable_list(event.session_id for event in events)
    last_session_id = session_ids[-1] if session_ids else ""
    last_question = events[-1].question if events else ""

    for event in events:
        candidate = candidates.get(event.signature_variant_id)
        if candidate is None:
            continue
        if event.signature_family_id not in index.families:
            index.families[event.signature_family_id] = _default_family_record(candidate)
            created_families.append(event.signature_family_id)
        family = index.families[event.signature_family_id]
        if event.signature_variant_id not in index.variants:
            index.variants[event.signature_variant_id] = _default_variant_record(candidate)
            created_variants.append(event.signature_variant_id)
        variant = index.variants[event.signature_variant_id]
        if event.signature_variant_id not in family.variant_ids:
            family.variant_ids.append(event.signature_variant_id)
        _update_variant_from_event(variant, event)
        touched_variants.append(event.signature_variant_id)
        touched_families.append(event.signature_family_id)
        index.total_events += 1

    for family_id in sorted(set(touched_families)):
        _refresh_family(index, family_id)
    for candidate in candidates.values():
        if candidate.variant_id not in index.variants:
            continue
        relation_id = _upsert_variant_relation(
            index=index,
            candidate=candidate,
            session_id=session_ids[0] if session_ids else "",
        )
        if not relation_id:
            continue
        relation = index.relations.get(relation_id)
        if relation is None:
            continue
        touched_relations.append(relation_id)
        if relation.observation_count == 1:
            created_relations.append(relation_id)
        _refresh_family(index, relation.family_id)

    base_event_count = len(events)
    for _round in range(3):
        new_promotion_events = _derive_promotion_events(
            index=index,
            variant_ids=touched_variants,
            session_id=last_session_id,
            question=last_question,
            start_index=base_event_count + len(promotion_events),
        )
        if not new_promotion_events:
            break
        promotion_events.extend(new_promotion_events)
        for event in new_promotion_events:
            candidate = candidates.get(event.signature_variant_id)
            if candidate is None:
                family = index.families.get(event.signature_family_id)
                variant = index.variants.get(event.signature_variant_id)
                if variant is None:
                    continue
                candidate = _candidate_from_variant(variant, family)
            if event.signature_family_id not in index.families:
                index.families[event.signature_family_id] = _default_family_record(candidate)
                created_families.append(event.signature_family_id)
            family = index.families[event.signature_family_id]
            if event.signature_variant_id not in index.variants:
                index.variants[event.signature_variant_id] = _default_variant_record(candidate)
                created_variants.append(event.signature_variant_id)
            variant = index.variants[event.signature_variant_id]
            if event.signature_variant_id not in family.variant_ids:
                family.variant_ids.append(event.signature_variant_id)
            _update_variant_from_event(variant, event)
            touched_variants.append(event.signature_variant_id)
            touched_families.append(event.signature_family_id)
            index.total_events += 1
        for family_id in sorted(set(touched_families)):
            _refresh_family(index, family_id)
    index.updated_at = _now_iso()
    return {
        "created_family_ids": _stable_list(created_families),
        "created_variant_ids": _stable_list(created_variants),
        "created_relation_ids": _stable_list(created_relations),
        "touched_family_ids": _stable_list(touched_families),
        "touched_variant_ids": _stable_list(touched_variants),
        "touched_relation_ids": _stable_list(touched_relations),
        "event_count": len(events) + len(promotion_events),
        "base_event_count": len(events),
        "promotion_event_count": len(promotion_events),
        "promotion_transition_counts": {
            "promoted_to_review": sum(1 for event in promotion_events if event.event_type == "promoted_to_review"),
            "promoted_to_supported": sum(1 for event in promotion_events if event.event_type == "promoted_to_supported"),
        },
        "promotion_events": [event.to_dict() for event in promotion_events],
        "total_index_events": index.total_events,
        **_resolution_summary(candidates),
    }


def build_shadow_report(
    *,
    index: SignatureStatsIndex,
    question: str,
    task_family: str,
    focus_variant_ids: Sequence[str],
    top_k: int = 5,
) -> Dict[str, Any]:
    def _has_supported_solved_subgoal(pool: Sequence[SignatureVariantStats]) -> bool:
        return any(
            variant.semantic_type == "solved_subgoal"
            and variant.epistemic_status == "supported"
            and variant.retrieval_tier == "normal"
            for variant in pool
        )

    def _shadow_retrieval_adjustment(
        variant: SignatureVariantStats,
        family: Optional[SignatureFamilyStats],
        *,
        has_supported_solved_subgoal: bool,
    ) -> float:
        tf = str(variant.task_family or "")
        relation_counts = dict(variant.relation_counts or {})
        if tf not in {"algorithm_applicability", "direct_judgment"}:
            if variant.semantic_type == "provisional_claim" and variant.retrieval_tier != "normal":
                return -0.35
            adjustment = 0.0
            if relation_counts.get("contradicts", 0) > 0:
                adjustment -= 0.25 * float(relation_counts.get("contradicts", 0))
            if relation_counts.get("entails", 0) > 0 and family is not None and family.dominant_variant_id == variant.id:
                adjustment += 0.15
            if variant.propagated_support_score > 0.1:
                adjustment += min(0.20, variant.propagated_support_score * 0.3)
            return round(adjustment, 4)

        adjustment = 0.0
        event_counts = dict(variant.event_counts or {})

        if has_supported_solved_subgoal:
            if variant.semantic_type == "solved_subgoal" and variant.epistemic_status == "supported":
                adjustment += 0.95
                if variant.retrieval_tier == "normal":
                    adjustment += 0.25
                adjustment += min(0.75, 0.25 * float(event_counts.get("supported_reuse", 0)))
                if variant.propagated_support_score > 0.2:
                    adjustment += min(0.30, variant.propagated_support_score * 0.4)
            elif variant.semantic_type == "strategy":
                if variant.epistemic_status == "provisional":
                    adjustment -= 1.00
                if variant.retrieval_tier != "normal":
                    adjustment -= 0.45
                if "controller-selected evidence and finalize" in normalize_text(variant.canonical_text).lower():
                    adjustment -= 0.25
                if variant.propagated_support_score > 0.3:
                    adjustment += min(0.15, variant.propagated_support_score * 0.2)
            elif variant.semantic_type == "provisional_claim":
                adjustment -= 0.80
                if variant.retrieval_tier != "normal":
                    adjustment -= 0.25
        else:
            if variant.semantic_type == "provisional_claim" and variant.retrieval_tier != "normal":
                adjustment -= 0.35

        if relation_counts.get("contradicts", 0) > 0:
            adjustment -= 0.35 * float(relation_counts.get("contradicts", 0))
        if relation_counts.get("overlaps", 0) > 0 and variant.semantic_type == "strategy" and variant.epistemic_status == "provisional":
            adjustment -= min(0.30, 0.10 * float(relation_counts.get("overlaps", 0)))
        if relation_counts.get("entails", 0) > 0 and family is not None and family.dominant_variant_id == variant.id:
            adjustment += min(0.25, 0.12 * float(relation_counts.get("entails", 0)))

        return round(adjustment, 4)

    def _annotate_rank(rows: Sequence[Mapping[str, Any]], rank_key: str) -> List[Dict[str, Any]]:
        ranked_rows: List[Dict[str, Any]] = []
        for i, row in enumerate(rows, start=1):
            ranked_rows.append({
                **dict(row),
                rank_key: i,
            })
        return ranked_rows

    def _collapse_family(rows: Sequence[Mapping[str, Any]], rank_key: str) -> List[Dict[str, Any]]:
        family_rows: List[Dict[str, Any]] = []
        seen_family_ids = set()
        for row in rows:
            family_id = str(row.get("family_id", "") or "")
            if not family_id or family_id in seen_family_ids:
                continue
            seen_family_ids.add(family_id)
            family = index.families.get(family_id)
            family_rows.append({
                "family_id": family_id,
                "family_label": family.family_label if family is not None else "",
                "semantic_type": row.get("semantic_type", ""),
                "task_family": row.get("task_family", ""),
                "best_variant_id": row.get("variant_id", ""),
                "family_contested": bool(row.get("family_contested", False)),
                "dominant_variant_id": family.dominant_variant_id if family is not None else None,
                "variant_count": len(family.variant_ids) if family is not None else 0,
                "epistemic_status": row.get("epistemic_status", ""),
                "retrieval_tier": row.get("retrieval_tier", ""),
                "baseline_score": row.get("baseline_score", 0.0),
                "bias_score": row.get("bias_score", 0.0),
                "effective_bias_score": row.get("effective_bias_score", 0.0),
                "effective_support_score": family.effective_support_score if family is not None else row.get("effective_support_score", 0.0),
                "effective_stability_score": family.effective_stability_score if family is not None else row.get("effective_stability_score", 0.0),
                "effective_contradiction_score": family.effective_contradiction_score if family is not None else row.get("effective_contradiction_score", 0.0),
                "adjusted_score": row.get("adjusted_score", 0.0),
                "relation_counts": dict(family.relation_counts) if family is not None else {},
                rank_key: int(row.get(rank_key, 0) or 0),
            })
        return family_rows

    candidate_variants: List[SignatureVariantStats] = []
    for variant in index.variants.values():
        if variant.semantic_type not in PHASE1_SIGNATURE_TYPES:
            continue
        if task_family and variant.task_family and variant.task_family != task_family:
            continue
        candidate_variants.append(variant)

    has_supported_solved_subgoal = _has_supported_solved_subgoal(candidate_variants)
    candidates: List[Dict[str, Any]] = []
    for variant in candidate_variants:
        base_text = " ".join([
            variant.canonical_text,
            variant.summary_text,
            variant.task_family,
            " ".join(variant.required_slots),
        ])
        lexical_score = lexical_overlap(question, base_text, min_chars=3)
        if task_family == "direct_judgment" and _direct_judgment_family_matches_question(question, variant.family_id):
            lexical_score = max(lexical_score, 1.0)
        family = index.families.get(variant.family_id)
        family_penalty = -0.35 if family is not None and family.contested else 0.0
        shadow_adjustment = _shadow_retrieval_adjustment(
            variant,
            family,
            has_supported_solved_subgoal=has_supported_solved_subgoal,
        )
        effective_bias = float(variant.effective_bias_score or variant.bias_score)
        adjusted = round(lexical_score + effective_bias + family_penalty + shadow_adjustment, 4)
        candidates.append({
            "variant_id": variant.id,
            "family_id": variant.family_id,
            "semantic_type": variant.semantic_type,
            "canonical_text": variant.canonical_text,
            "task_family": variant.task_family,
            "epistemic_status": variant.epistemic_status,
            "retrieval_tier": variant.retrieval_tier,
            "baseline_score": round(lexical_score, 4),
            "bias_score": variant.bias_score,
            "effective_bias_score": effective_bias,
            "effective_support_score": variant.effective_support_score,
            "effective_stability_score": variant.effective_stability_score,
            "effective_risk_score": variant.effective_risk_score,
            "effective_contradiction_score": variant.effective_contradiction_score,
            "relation_counts": dict(variant.relation_counts or {}),
            "shadow_adjustment": shadow_adjustment,
            "adjusted_score": adjusted,
            "family_contested": bool(family.contested) if family is not None else False,
            "dominant_variant_id": family.dominant_variant_id if family is not None else None,
        })

    baseline = _annotate_rank(
        sorted(candidates, key=lambda row: row["baseline_score"], reverse=True),
        "baseline_rank",
    )
    adjusted = _annotate_rank(
        sorted(candidates, key=lambda row: row["adjusted_score"], reverse=True),
        "adjusted_rank",
    )
    baseline_rank = {row["variant_id"]: int(row["baseline_rank"]) for row in baseline}
    adjusted_rank = {row["variant_id"]: int(row["adjusted_rank"]) for row in adjusted}
    for row in adjusted:
        row["baseline_rank"] = baseline_rank.get(str(row.get("variant_id", "")), 0)
        row["rank_delta"] = row["baseline_rank"] - int(row.get("adjusted_rank", 0) or 0)
    movers: List[Dict[str, Any]] = []
    for row in adjusted:
        vid = row["variant_id"]
        before = baseline_rank.get(vid, 0)
        after = adjusted_rank.get(vid, 0)
        if before != after:
            movers.append({
                "variant_id": vid,
                "family_id": row["family_id"],
                "semantic_type": row["semantic_type"],
                "baseline_rank": before,
                "adjusted_rank": after,
                "rank_delta": before - after,
                "bias_score": row["bias_score"],
                "family_contested": row["family_contested"],
            })
    focus_rows = [row for row in adjusted if row["variant_id"] in set(focus_variant_ids)]
    baseline_family = _collapse_family(baseline, "baseline_rank")
    adjusted_family = _collapse_family(adjusted, "adjusted_rank")
    return {
        "mode": "shadow_only",
        "question": question,
        "task_family": task_family,
        "ranking_complete": True,
        "candidate_count": len(candidates),
        "family_count": len({row["family_id"] for row in candidates if str(row.get("family_id", "")).strip()}),
        "baseline_top_k": baseline[:top_k],
        "adjusted_top_k": adjusted[:top_k],
        "baseline_ranking": baseline,
        "adjusted_ranking": adjusted,
        "baseline_family_ranking": baseline_family,
        "adjusted_family_ranking": adjusted_family,
        "rank_movers": sorted(movers, key=lambda row: abs(int(row["rank_delta"])), reverse=True)[:top_k],
        "focus_variants": focus_rows,
    }


_LIVE_BIAS_TASK_FAMILIES = frozenset({"algorithm_applicability", "direct_judgment"})


def _rank_live_anchor_rows(
    *,
    question: str,
    graph_nodes: Mapping[str, Any],
    support_node_ids: Sequence[str],
    linked_node_ids: Sequence[str],
    source_node_ids: Sequence[str],
) -> List[Dict[str, Any]]:
    support_set = {str(x) for x in support_node_ids if str(x or "").strip()}
    linked_set = {str(x) for x in linked_node_ids if str(x or "").strip()}
    source_set = {str(x) for x in source_node_ids if str(x or "").strip()}
    candidate_ids = _stable_list([*support_node_ids, *linked_node_ids, *source_node_ids])
    rows: List[Dict[str, Any]] = []
    for node_id in candidate_ids:
        node = graph_nodes.get(node_id)
        if node is None:
            continue
        text = str(getattr(node, "text", "") or "")
        node_type = str(getattr(node, "node_type", "") or "")
        lexical_score = lexical_overlap(question, " ".join([node_id, text]), min_chars=3)
        support_bonus = 0.80 if node_id in support_set else 0.0
        linked_bonus = 0.25 if node_id in linked_set else 0.0
        source_bonus = 0.10 if node_id in source_set else 0.0
        metadata = getattr(node, "metadata", {}) or {}
        deprecated_penalty = -1.0 if metadata.get("deprecated") else 0.0
        total_score = round(lexical_score + support_bonus + linked_bonus + source_bonus + deprecated_penalty, 4)
        rows.append({
            "node_id": node_id,
            "node_type": node_type,
            "lexical_score": round(lexical_score, 4),
            "support_bonus": support_bonus,
            "linked_bonus": linked_bonus,
            "source_bonus": source_bonus,
            "score": total_score,
            "text_preview": text[:160],
        })
    rows.sort(
        key=lambda row: (
            float(row.get("score", 0.0)),
            float(row.get("lexical_score", 0.0)),
            str(row.get("node_id", "")),
        ),
        reverse=True,
    )
    return rows


def _contains_any_phrase(text: str, phrases: Sequence[str]) -> bool:
    low = str(text or "").lower()
    return any(str(phrase or "").lower() in low for phrase in phrases)


def _direct_judgment_family_matches_question(question: str, family_id: str) -> bool:
    fam = str(family_id or "")
    if not fam:
        return True
    low = str(question or "").lower()
    if fam.endswith("direct_judgment_sound_requires_medium_vs_light_vacuum"):
        mentions_sound = _contains_any_phrase(
            low,
            ("sound", "hear", "hearing", "audible", "audio", "acoustic", "sonic", "noise"),
        )
        mentions_sight_or_light = _contains_any_phrase(
            low,
            ("light", "sunlight", "starlight", "laser", "flash", "visible", "see", "seeing", "sight", "star", "stars", "sun"),
        )
        mentions_vacuum = _contains_any_phrase(
            low,
            ("vacuum", "space", "outer space", "empty space", "airless", "without air", "no air"),
        )
        return mentions_sound and mentions_sight_or_light and mentions_vacuum
    if fam.endswith("direct_judgment_refraction_changes_speed_not_frequency"):
        mentions_frequency = "frequency" in low
        mentions_optical_signal = _contains_any_phrase(
            low,
            ("light", "laser", "beam", "photon", "sunlight", "starlight", "visible", "prism"),
        )
        mentions_medium_or_boundary = _contains_any_phrase(
            low,
            ("refraction", "refract", "refracted", "refracts", "prism", "glass", "water", "medium", "boundary", "interface"),
        )
        mentions_transition_or_speed_change = _contains_any_phrase(
            low,
            ("enter", "enters", "entering", "goes into", "passes into", "slows down", "speed changes", "changes speed", "bend", "bends", "wavelength"),
        )
        return mentions_frequency and mentions_optical_signal and (mentions_medium_or_boundary or mentions_transition_or_speed_change)
    return True


def _passes_live_bias_relevance_gate(
    *,
    question: str,
    task_family: str,
    row: Mapping[str, Any],
    anchor_rows: Sequence[Mapping[str, Any]],
    variant: SignatureVariantStats,
) -> Tuple[bool, str]:
    if task_family != "direct_judgment":
        return True, "not_required"
    if not _direct_judgment_family_matches_question(question, variant.family_id):
        return False, "family_question_mismatch"
    baseline_score = float(row.get("baseline_score", 0.0) or 0.0)
    baseline_rank = int(row.get("baseline_rank", 0) or 0)
    best_anchor_lexical = float(anchor_rows[0].get("lexical_score", 0.0) or 0.0) if anchor_rows else 0.0
    second_anchor_lexical = float(anchor_rows[1].get("lexical_score", 0.0) or 0.0) if len(anchor_rows) > 1 else 0.0
    anchor_lexical_mass = round(
        sum(float(anchor_row.get("lexical_score", 0.0) or 0.0) for anchor_row in anchor_rows[:3]),
        4,
    )
    support_node_count = len({str(node_id) for node_id in variant.top_supporting_node_ids if str(node_id or "").strip()})
    # Check stronger threshold first so already_strong_baseline (>= 0.24) wins
    # over already_top_baseline (>= 0.18) when both apply.
    if baseline_rank in {1, 2} and baseline_score >= 0.24:
        return False, f"already_strong_baseline(rank={baseline_rank}, baseline={baseline_score:.4f})"
    if baseline_rank == 1 and baseline_score >= 0.18:
        return False, f"already_top_baseline(rank={baseline_rank}, baseline={baseline_score:.4f})"
    if (
        support_node_count >= 2
        and baseline_score <= 0.18
        and best_anchor_lexical >= 0.11
        and anchor_lexical_mass >= 0.14
    ):
        return True, "ambiguous_multi_support"
    if (
        support_node_count >= 3
        and baseline_score <= 0.22
        and best_anchor_lexical >= 0.08
        and anchor_lexical_mass >= 0.18
    ):
        return True, "dense_multi_support"
    is_explicit_dj_family = variant.family_id.startswith("sigfam_solved_subgoal.direct_judgment.")
    is_review_fast_track = (
        variant.promotion_state in ("supported", "review") 
        and is_explicit_dj_family
    )
    has_strong_propagation = variant.propagated_support_score >= 0.3
    if (
        support_node_count == 1
        and baseline_rank >= 2
        and baseline_score <= 0.12
        and best_anchor_lexical >= 0.09
        and anchor_lexical_mass >= 0.09
        and (variant.promotion_state == "supported" or is_review_fast_track or has_strong_propagation)
        and (variant.effective_support_score >= 4.5 or is_review_fast_track or has_strong_propagation)
        and (variant.effective_stability_score >= 4.5 or is_review_fast_track or has_strong_propagation)
        and variant.effective_contradiction_score <= 0.20
    ):
        reason = "explicit_family_review_fast_track" if variant.promotion_state == "review" else "high_confidence_single_support"
        return True, reason
    return False, (
        "weak_match("
        f"rank={baseline_rank}, "
        f"baseline={baseline_score:.4f}, "
        f"anchor_lexical={best_anchor_lexical:.4f}, "
        f"anchor_lexical_2={second_anchor_lexical:.4f}, "
        f"anchor_mass={anchor_lexical_mass:.4f}, "
        f"support_nodes={support_node_count})"
    )


def build_live_signature_bias_plan(
    *,
    index: SignatureStatsIndex,
    question: str,
    task_family: str,
    graph_nodes: Mapping[str, Any],
    max_anchor_ids: int = 5,
    stats_index_path: str | Path = "",
) -> LiveSignatureBiasPlan:
    plan = LiveSignatureBiasPlan(
        enabled=False,
        reason="not_applicable",
        question=question,
        task_family=task_family,
        stats_index_path=str(stats_index_path or ""),
    )
    if task_family not in _LIVE_BIAS_TASK_FAMILIES:
        plan.reason = "task_family_not_enabled"
        return plan
    if not index.variants:
        plan.reason = "empty_signature_index"
        return plan

    shadow_report = build_shadow_report(
        index=index,
        question=question,
        task_family=task_family,
        focus_variant_ids=[],
        top_k=max(5, max_anchor_ids),
    )
    plan.candidate_count = int(shadow_report.get("candidate_count", 0) or 0)
    adjusted_rows = list(shadow_report.get("adjusted_ranking", []))
    if not adjusted_rows:
        plan.reason = "no_shadow_candidates"
        return plan

    skipped: List[Dict[str, Any]] = []
    for row in adjusted_rows:
        variant_id = str(row.get("variant_id", "") or "")
        family_id = str(row.get("family_id", "") or "")
        variant = index.variants.get(variant_id)
        family = index.families.get(family_id)
        if variant is None or family is None:
            skipped.append({
                "variant_id": variant_id,
                "family_id": family_id,
                "reason": "missing_variant_or_family",
            })
            continue
        if variant.semantic_type != "solved_subgoal":
            skipped.append({
                "variant_id": variant_id,
                "family_id": family_id,
                "reason": "semantic_type_not_live_bias_eligible",
                "semantic_type": variant.semantic_type,
            })
            continue
        if variant.epistemic_status != "supported":
            skipped.append({
                "variant_id": variant_id,
                "family_id": family_id,
                "reason": "variant_not_supported",
                "epistemic_status": variant.epistemic_status,
            })
            continue
        if family.contested:
            skipped.append({
                "variant_id": variant_id,
                "family_id": family_id,
                "reason": "family_contested",
            })
            continue
        if variant.retrieval_tier != "normal":
            skipped.append({
                "variant_id": variant_id,
                "family_id": family_id,
                "reason": "variant_not_normal_tier",
                "retrieval_tier": variant.retrieval_tier,
            })
            continue
        if family.dominant_variant_id and family.dominant_variant_id != variant.id:
            skipped.append({
                "variant_id": variant_id,
                "family_id": family_id,
                "reason": "non_dominant_family_variant",
                "dominant_variant_id": family.dominant_variant_id,
            })
            continue
        if variant.effective_contradiction_score > 0.9:
            skipped.append({
                "variant_id": variant_id,
                "family_id": family_id,
                "reason": "effective_contradiction_too_high",
                "effective_contradiction_score": variant.effective_contradiction_score,
            })
            continue

        anchor_rows = _rank_live_anchor_rows(
            question=question,
            graph_nodes=graph_nodes,
            support_node_ids=variant.top_supporting_node_ids,
            linked_node_ids=variant.linked_node_ids,
            source_node_ids=variant.source_node_ids,
        )
        if not anchor_rows:
            skipped.append({
                "variant_id": variant_id,
                "family_id": family_id,
                "reason": "no_graph_backed_anchor_nodes",
            })
            continue
        passes_relevance_gate, gate_reason = _passes_live_bias_relevance_gate(
            question=question,
            task_family=task_family,
            row=row,
            anchor_rows=anchor_rows,
            variant=variant,
        )
        if not passes_relevance_gate:
            skipped.append({
                "variant_id": variant_id,
                "family_id": family_id,
                "reason": "insufficient_live_relevance",
                "gate_reason": gate_reason,
                "baseline_score": float(row.get("baseline_score", 0.0) or 0.0),
                "anchor_lexical_score": float(anchor_rows[0].get("lexical_score", 0.0) or 0.0),
                "support_node_count": len({str(node_id) for node_id in variant.top_supporting_node_ids if str(node_id or "").strip()}),
            })
            continue

        plan.enabled = True
        plan.reason = "supported_solved_subgoal_family"
        plan.family_id = family.id
        plan.family_label = family.family_label
        plan.variant_id = variant.id
        plan.semantic_type = variant.semantic_type
        plan.baseline_rank = int(row.get("baseline_rank", 0) or 0)
        plan.adjusted_rank = int(row.get("adjusted_rank", 0) or 0)
        plan.adjusted_score = float(row.get("adjusted_score", 0.0) or 0.0)
        plan.baseline_score = float(row.get("baseline_score", 0.0) or 0.0)
        plan.bias_score = float(row.get("bias_score", 0.0) or 0.0)
        plan.shadow_adjustment = float(row.get("shadow_adjustment", 0.0) or 0.0)
        plan.anchor_rows = anchor_rows[:max_anchor_ids]
        plan.anchor_ids = [str(anchor_row["node_id"]) for anchor_row in plan.anchor_rows]
        plan.support_node_ids = [
            str(node_id)
            for node_id in variant.top_supporting_node_ids
            if str(node_id) in graph_nodes
        ]
        plan.linked_node_ids = [
            str(node_id)
            for node_id in variant.linked_node_ids
            if str(node_id) in graph_nodes
        ]
        if gate_reason != "not_required":
            plan.reason = f"{plan.reason}:{gate_reason}"
        plan.skipped_candidates = skipped[:8]
        return plan

    plan.reason = "no_supported_graph_backed_solved_subgoal"
    plan.skipped_candidates = skipped[:8]
    return plan


def load_live_signature_bias_plan(
    *,
    question: str,
    task_family: str,
    graph_nodes: Mapping[str, Any],
    stats_dir: str | Path = "data/signature_stats",
    max_anchor_ids: int = 5,
) -> LiveSignatureBiasPlan:
    root = default_signature_stats_dir(stats_dir)
    index_path = root / "signature_stats_index.json"
    index = load_signature_stats_index(index_path)
    return build_live_signature_bias_plan(
        index=index,
        question=question,
        task_family=task_family,
        graph_nodes=graph_nodes,
        max_anchor_ids=max_anchor_ids,
        stats_index_path=index_path,
    )


def _family_projection_text(family: SignatureFamilyStats, index: SignatureStatsIndex) -> str:
    parts = [
        f"Signature family: {family.family_label}",
        f"Semantic type: {family.semantic_type}",
        f"Task family: {family.task_family}",
    ]
    if family.contested:
        parts.append("Status: contested")
    if family.dominant_variant_id:
        parts.append(f"Dominant variant: {family.dominant_variant_id}")
    if family.variant_ids:
        parts.append(f"Variants: {len(family.variant_ids)}")
    top_variant = index.variants.get(family.dominant_variant_id or (family.variant_ids[0] if family.variant_ids else ""))
    if top_variant is not None and top_variant.canonical_text:
        parts.append(f"Representative: {top_variant.canonical_text}")
    return "\n".join(parts)


def _variant_projection_text(variant: SignatureVariantStats) -> str:
    parts = [
        f"Signature variant: {variant.id}",
        f"Semantic type: {variant.semantic_type}",
    ]
    if variant.canonical_text:
        parts.append(variant.canonical_text)
    if variant.required_slots:
        parts.append("Required slots: " + ", ".join(variant.required_slots))
    if variant.aliases:
        parts.append("Aliases: " + "; ".join(variant.aliases[:3]))
    return "\n".join(parts)


def build_signature_graph_projection(
    *,
    index: SignatureStatsIndex,
    focus_variant_ids: Sequence[str],
) -> Dict[str, Any]:
    focus_set = {str(x) for x in focus_variant_ids if str(x or "").strip()}
    focus_family_ids = {
        index.variants[vid].family_id
        for vid in focus_set
        if vid in index.variants
    }
    nodes: List[Dict[str, Any]] = []
    edges: List[Dict[str, Any]] = []
    seen_nodes = set()
    seen_edges = set()

    def add_node(node: Dict[str, Any]) -> None:
        nid = str(node.get("id", "") or "")
        if not nid or nid in seen_nodes:
            return
        seen_nodes.add(nid)
        nodes.append(node)

    def add_edge(src: str, dst: str, relation: str, metadata: Optional[Mapping[str, Any]] = None) -> None:
        key = (src, dst, relation)
        if not src or not dst or key in seen_edges:
            return
        seen_edges.add(key)
        edges.append({
            "src": src,
            "dst": dst,
            "relation": relation,
            "metadata": dict(metadata or {}),
        })

    for family_id in sorted(focus_family_ids):
        family = index.families.get(family_id)
        if family is None:
            continue
        add_node({
            "id": family.id,
            "node_type": "signature_family",
            "text": _family_projection_text(family, index),
            "metadata": {
                "semantic_type": family.semantic_type,
                "task_family": family.task_family,
                "family_label": family.family_label,
                "variant_ids": list(family.variant_ids),
                "support_score": family.support_score,
                "stability_score": family.stability_score,
                "risk_score": family.risk_score,
                "contradiction_score": family.contradiction_score,
                "bias_score": family.bias_score,
                "effective_support_score": family.effective_support_score,
                "effective_stability_score": family.effective_stability_score,
                "effective_risk_score": family.effective_risk_score,
                "effective_contradiction_score": family.effective_contradiction_score,
                "effective_bias_score": family.effective_bias_score,
                "relation_counts": dict(family.relation_counts),
                "contested": family.contested,
                "dominant_variant_id": family.dominant_variant_id,
                "retrieval_tier": "gated" if family.contested else "normal",
            },
        })
        if family.dominant_variant_id:
            add_edge(family.id, family.dominant_variant_id, "dominant_variant", {"contested": family.contested})
        for vid in family.variant_ids:
            variant = index.variants.get(vid)
            if variant is None:
                continue
            add_node({
                "id": variant.id,
                "node_type": "signature_variant",
                "text": _variant_projection_text(variant),
                "metadata": {
                    "family_id": family.id,
                    "semantic_type": variant.semantic_type,
                    "task_family": variant.task_family,
                    "epistemic_status": variant.epistemic_status,
                    "promotion_state": variant.promotion_state,
                    "retrieval_tier": variant.retrieval_tier,
                    "support_score": variant.support_score,
                    "stability_score": variant.stability_score,
                    "risk_score": variant.risk_score,
                    "contradiction_score": variant.contradiction_score,
                    "bias_score": variant.bias_score,
                    "propagated_support_score": variant.propagated_support_score,
                    "propagated_stability_score": variant.propagated_stability_score,
                    "propagated_risk_score": variant.propagated_risk_score,
                    "propagated_contradiction_score": variant.propagated_contradiction_score,
                    "effective_support_score": variant.effective_support_score,
                    "effective_stability_score": variant.effective_stability_score,
                    "effective_risk_score": variant.effective_risk_score,
                    "effective_contradiction_score": variant.effective_contradiction_score,
                    "effective_bias_score": variant.effective_bias_score,
                    "relation_counts": dict(variant.relation_counts),
                    "required_slots": list(variant.required_slots),
                    "source_node_ids": list(variant.source_node_ids),
                    "support_node_ids": list(variant.top_supporting_node_ids),
                    "aliases": list(variant.aliases),
                },
            })
            add_edge(family.id, variant.id, "has_variant", {"contested_family": family.contested})
            for semantic_node_id in variant.source_node_ids:
                add_edge(variant.id, semantic_node_id, "realized_as", {"semantic_type": variant.semantic_type})
            for support_node_id in variant.top_supporting_node_ids:
                add_edge(variant.id, support_node_id, "supported_by", {"family_id": family.id})
    for relation in index.relations.values():
        if relation.family_id not in focus_family_ids:
            continue
        if relation.src_variant_id not in seen_nodes or relation.dst_variant_id not in seen_nodes:
            continue
        add_edge(
            relation.src_variant_id,
            relation.dst_variant_id,
            relation.relation_type,
            {
                "relation_id": relation.id,
                "family_id": relation.family_id,
                "symmetric": relation.symmetric,
                "observation_count": relation.observation_count,
                "last_match_score": relation.last_match_score,
                "reasons": list(relation.reasons),
            },
        )

    return {
        "focus_variant_ids": sorted(focus_set),
        "focus_family_ids": sorted(focus_family_ids),
        "nodes": nodes,
        "edges": edges,
        "summary": {
            "family_count": len(focus_family_ids),
            "node_count": len(nodes),
            "edge_count": len(edges),
            "relation_edge_count": sum(1 for edge in edges if edge.get("relation") in {"overlaps", "entails", "contradicts"}),
        },
    }


def _score_events_with_llm(
    *,
    events: List[SignatureEvent],
    candidates: Mapping[str, SignatureCandidate],
    index: SignatureStatsIndex,
    session_id: str,
    question: str,
    finalized: bool,
    execution_mode: str,
    controller: Any,
) -> List[SignatureEvent]:
    """Score ambiguous events using LLM judge. Returns updated event list."""
    deterministic_events = {
        "promoted_to_review", "promoted_to_supported", "deprecated",
        "contradicted", "scoped_patch_reject",
    }
    session_context = {
        "finalized": finalized,
        "execution_mode": execution_mode,
    }
    scored_events: List[SignatureEvent] = []
    for event in events:
        if event.event_type in deterministic_events:
            scored_events.append(event)
            continue
        candidate = candidates.get(event.signature_variant_id)
        if candidate is None:
            scored_events.append(event)
            continue
        variant = index.variants.get(event.signature_variant_id)
        impact_score, rationale = score_event_impact(
            event_type=event.event_type,
            candidate=candidate,
            variant=variant,
            session_context=session_context,
            controller=controller,
        )
        if impact_score < 0.25:
            new_bucket = "tiny"
        elif impact_score < 0.75:
            new_bucket = "low"
        elif impact_score < 1.25:
            new_bucket = "medium"
        elif impact_score < 1.75:
            new_bucket = "high"
        else:
            new_bucket = "critical"
        event.impact_bucket = new_bucket
        event.metadata["llm_impact_score"] = round(impact_score, 4)
        event.metadata["llm_impact_rationale"] = rationale
        scored_events.append(event)
    return scored_events


def run_signature_shadow_session(
    *,
    session_id: str,
    question: str,
    task_family: str,
    graph_edits: Sequence[Mapping[str, Any]],
    scoped_patches: Sequence[Mapping[str, Any]],
    hypotheses: Mapping[str, Mapping[str, Any]],
    final_answer: str,
    cited_node_ids: Sequence[str],
    finalized: bool,
    execution_mode: str,
    design_evidence_gate_rounds: int = 0,
    stats_dir: str | Path = "data/signature_stats",
    controller: Optional[Any] = None,
) -> Dict[str, Any]:
    root = default_signature_stats_dir(stats_dir)
    index_path = root / "signature_stats_index.json"
    events_path = root / "signature_events.jsonl"
    index = load_signature_stats_index(index_path)
    candidates = collect_signature_candidates(
        index=index,
        question=question,
        task_family=task_family,
        graph_edits=graph_edits,
        hypotheses=hypotheses,
        final_answer=final_answer,
        cited_node_ids=cited_node_ids,
        controller=controller,
    )
    events = collect_signature_events(
        session_id=session_id,
        question=question,
        task_family=task_family,
        candidates=candidates,
        hypotheses=hypotheses,
        scoped_patches=scoped_patches,
        final_answer=final_answer,
        finalized=finalized,
        execution_mode=execution_mode,
        design_evidence_gate_rounds=design_evidence_gate_rounds,
    )
    if controller is not None:
        events = _score_events_with_llm(
            events=events,
            candidates=candidates,
            index=index,
            session_id=session_id,
            question=question,
            finalized=finalized,
            execution_mode=execution_mode,
            controller=controller,
        )
    update_summary = apply_signature_events(index=index, candidates=candidates, events=events)
    all_events = list(events) + [
        SignatureEvent.from_dict(payload)
        for payload in list(update_summary.get("promotion_events", []))
        if isinstance(payload, Mapping)
    ]
    save_signature_stats_index(index, index_path)
    append_signature_events_jsonl(all_events, events_path)
    shadow_report = build_shadow_report(
        index=index,
        question=question,
        task_family=task_family,
        focus_variant_ids=update_summary.get("touched_variant_ids", []),
    )
    graph_projection = build_signature_graph_projection(
        index=index,
        focus_variant_ids=update_summary.get("touched_variant_ids", []),
    )
    return {
        "candidates": [cand.to_dict() for cand in candidates.values()],
        "events": [event.to_dict() for event in all_events],
        "update_summary": {
            **update_summary,
            "index_path": str(index_path),
            "events_path": str(events_path),
            "candidate_count": len(candidates),
        },
        "shadow_report": shadow_report,
        "graph_projection": graph_projection,
    }
