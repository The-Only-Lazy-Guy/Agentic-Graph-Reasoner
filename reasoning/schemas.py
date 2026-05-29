"""Typed dataclasses for the reasoning substrate.

Every dataclass exposes:
  - to_dict() — full serialization (via dataclasses.asdict)
  - from_dict() — reconstruction from a plain dict

Matches the convention used by graph_core.Node / graph_core.Edge so the
substrate plays nicely with the rest of the project.

See PHASE1_PLAN.md §2 for the design rationale and field-by-field semantics.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Literal, Mapping, Optional

# Node-type enum. Existing node_types come from graph_core; the three new ones
# (procedure, failure_pattern, session_object) are Phase-1 additions.
NodeType = Literal[
    "fact",
    "claim",
    "example",
    "summary",
    "hub",
    "bridge",
    "hypothesis",
    "application",
    "procedure",
    "failure_pattern",
    "session_object",
    "signal",                  # Phase 3A: meta-procedure observation persisted to subgraph
    "activation_signal",       # Phase 3C: typed graph-activation observation
    "task_frame_item",         # Phase 3C: prompt-frame item derived from activation signals
    "session_gap",             # Phase 3C: session-scoped missing-context node
    "session_bridge",          # Phase 3C: session-scoped provisional connector
    "plan_node",               # Phase 3D: adaptive planning checkpoint
    "plan_check",              # Phase 3D: deterministic/model-assisted plan validation result
    "strategy",                # Proven reasoning recipe from a successful session
    "solved_subgoal",          # Persistent reusable answer to a typed micro-subproblem
    "reasoning_atom",          # Reusable reasoning fragment / explanation atom
    "control_rule",            # Persistent controller policy for a task family
    "signature_family",        # Family-level wrapper for related learned memory variants
    "signature_variant",       # Variant-level wrapper linked to a semantic memory node
NodeType = Literal[
    "fact",
    "claim",
    "example",
    "summary",
    "hub",
    "bridge",
    "hypothesis",
    "application",
    "procedure",
    "failure_pattern",
    "session_object",
    "signal",                  # Phase 3A: meta-procedure observation persisted to subgraph
    "activation_signal",       # Phase 3C: typed graph-activation observation
    "task_frame_item",         # Phase 3C: prompt-frame item derived from activation signals
    "session_gap",             # Phase 3C: session-scoped missing-context node
    "session_bridge",          # Phase 3C: session-scoped provisional connector
    "plan_node",               # Phase 3D: adaptive planning checkpoint
    "plan_check",              # Phase 3D: deterministic/model-assisted plan validation result
    "strategy",                # Proven reasoning recipe from a successful session
    "solved_subgoal",          # Persistent reusable answer to a typed micro-subproblem
    "reasoning_atom",          # Reusable reasoning fragment / explanation atom
    "control_rule",            # Persistent controller policy for a task family
    "signature_family",        # Family-level wrapper for related learned memory variants
    "signature_variant",       # Variant-level wrapper linked to a semantic memory node
    "reasoning_chain",         # V5: Multi-hop deductive path (A→B→C logic) as a named reusable node
    "epistemic_state",         # V5: Belief-status node (how strongly known, what invalidates it)
]

# CRUD operations on session-object state, captured in the audit log.
MutationOp = Literal["create", "read", "update", "delete"]


# ---------- Provenance ---------------------------------------------------- #

@dataclass
class Provenance:
    """Universal metadata carried by every reasoning-substrate node.

    Without provenance, debugging a wrong answer six months from now becomes
    archaeology. Every procedure/failure_pattern/session_object MUST carry
    this. See PHASE1_PLAN.md §11 / REASONING_ARCHITECTURE.md §3.3.
    """
    created_in_session_id: str
    validating_examples: List[str] = field(default_factory=list)
    depends_on: List[str] = field(default_factory=list)
    citation_count: int = 0
    citation_decay: float = 1.0
    last_modified: str = ""           # ISO8601
    deprecated: bool = False
    deprecation_reason: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @staticmethod
    def from_dict(d: Mapping[str, Any]) -> "Provenance":
        return Provenance(
            created_in_session_id=d["created_in_session_id"],
            validating_examples=list(d.get("validating_examples", [])),
            depends_on=list(d.get("depends_on", [])),
            citation_count=int(d.get("citation_count", 0)),
            citation_decay=float(d.get("citation_decay", 1.0)),
            last_modified=d.get("last_modified", ""),
            deprecated=bool(d.get("deprecated", False)),
            deprecation_reason=d.get("deprecation_reason"),
        )


# ---------- Procedure ----------------------------------------------------- #

@dataclass
class ProcedureNode:
    """A reusable reasoning template with optional internal state.

    Procedures are graph nodes (node_type='procedure'). They expose a
    signature (inputs/outputs), a state_schema describing the shape of any
    mutable workspace they maintain, and a body which is a sub-prompt
    template rendered with bound inputs at invocation time.

    Instantiated copies during a session live as SessionObjectNode (with
    actual state values), referencing this ProcedureNode by id.

    Phase-2A added version-chain fields. All optional with safe defaults so
    Phase-1 serialized procedures (no version field) load unchanged.
    See PHASE2_PLAN.md §3.1 + §5.
    """
    id: str
    name: str                         # short slug, e.g. 'VerifyAlgorithmPreconditions'
    purpose: str                      # one-sentence summary
    when_to_use: str                  # paragraph explaining trigger conditions
    signature: Dict[str, Any]         # {'inputs': [...], 'outputs': [...]}
    state_schema: Dict[str, str]      # field_name -> JSON-type string
    body: str                         # sub-prompt template with {placeholders}
    example_use: Optional[Dict[str, Any]]  # the quality gate; None = not promotable
    provenance: Provenance
    node_type: NodeType = "procedure"
    # ---- version chain (Phase 2A) ---- #
    # `version` is the position in the name family: 1 for the original, 2 for
    # the first refinement, etc. Multiple ProcedureNodes can share the same
    # `name`; they form a chain via parent_version_id / superseded_by_id.
    version: int = 1
    # parent_version_id points BACKWARD to the version this one refined.
    # None on the original (version=1).
    parent_version_id: Optional[str] = None
    # superseded_by_id points FORWARD to the version that replaces this one.
    # None on the head of the chain (latest version). Set when a successor
    # ProcedureNode is created.
    superseded_by_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @staticmethod
    def from_dict(d: Mapping[str, Any]) -> "ProcedureNode":
        return ProcedureNode(
            id=d["id"],
            name=d["name"],
            purpose=d["purpose"],
            when_to_use=d["when_to_use"],
            signature=dict(d.get("signature", {})),
            state_schema=dict(d.get("state_schema", {})),
            body=d["body"],
            example_use=dict(d["example_use"]) if d.get("example_use") else None,
            provenance=Provenance.from_dict(d["provenance"]),
            node_type=d.get("node_type", "procedure"),
            # Defaults preserve Phase-1 backward compatibility when these
            # fields are absent in serialized form.
            version=int(d.get("version", 1)),
            parent_version_id=d.get("parent_version_id"),
            superseded_by_id=d.get("superseded_by_id"),
        )


# ---------- Failure pattern ----------------------------------------------- #

@dataclass
class FailurePatternNode:
    """An anti-pattern: a documented failure mode of some approach.

    Distinct from a misconception (which is claim-level, e.g. 'Dijkstra
    works without negative cycles'). Failure patterns are procedure-level:
    'attempting approach X on problem-class Y fails because Z'.

    Retrieval boosts these (1.4x by default) so they surface in context
    when relevant, warning the reasoner away from anti-patterns.
    """
    id: str
    name: str
    attempted_approach: str
    failure_condition: str
    failure_mechanism: str
    replacement: Optional[str]        # node-id of recommended alternative (procedure/fact)
    example_failure_case: Optional[Dict[str, Any]]
    provenance: Provenance
    node_type: NodeType = "failure_pattern"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @staticmethod
    def from_dict(d: Mapping[str, Any]) -> "FailurePatternNode":
        return FailurePatternNode(
            id=d["id"],
            name=d["name"],
            attempted_approach=d["attempted_approach"],
            failure_condition=d["failure_condition"],
            failure_mechanism=d["failure_mechanism"],
            replacement=d.get("replacement"),
            example_failure_case=dict(d["example_failure_case"])
                if d.get("example_failure_case") else None,
            provenance=Provenance.from_dict(d["provenance"]),
            node_type=d.get("node_type", "failure_pattern"),
        )


# ---------- Session object ------------------------------------------------ #

@dataclass
class SessionObjectNode:
    """A live, stateful instance of a Procedure within one session.

    Created when the dispatcher fires a procedure invocation. Carries the
    current value of every field in procedure.state_schema. Mutated via
    full CRUD operations, each journaled to the audit log.

    Lives in the session subgraph. May be referenced by other session
    objects within the same session. Does not directly persist to long-term
    memory — the session subgraph is the persistence unit, and consolidation
    decides whether parts of it graduate.
    """
    id: str
    procedure_id: str                 # ref to underlying ProcedureNode.id
    name: str                         # usually procedure.name
    state: Dict[str, Any]             # current state, keyed by state_schema fields
    created_step: int                 # session step at which this object was created
    provenance: Provenance
    node_type: NodeType = "session_object"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @staticmethod
    def from_dict(d: Mapping[str, Any]) -> "SessionObjectNode":
        return SessionObjectNode(
            id=d["id"],
            procedure_id=d["procedure_id"],
            name=d["name"],
            state=dict(d.get("state", {})),
            created_step=int(d["created_step"]),
            provenance=Provenance.from_dict(d["provenance"]),
            node_type=d.get("node_type", "session_object"),
        )


# ---------- Audit entry --------------------------------------------------- #

@dataclass
class AuditEntry:
    """One mutation event in a session's audit log.

    Journaled JSONL-style (one entry per line) for atomic-append safety
    and partial-read tolerance. Every CRUD operation on any
    SessionObjectNode produces exactly one of these.

    The triggered_by_text field captures the reasoning snippet that
    caused the mutation — load-bearing for debugging six months later
    when an answer turned out to be wrong and we need to trace the
    procedure chain.
    """
    session_id: str
    step_index: int
    object_id: str
    operation: MutationOp
    field_path: str                   # dotted, e.g. 'state.visited_nodes'
    old_value: Any                    # None for create
    new_value: Any                    # None for delete/read
    triggered_by_text: str
    timestamp: str                    # ISO8601

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @staticmethod
    def from_dict(d: Mapping[str, Any]) -> "AuditEntry":
        return AuditEntry(
            session_id=d["session_id"],
            step_index=int(d["step_index"]),
            object_id=d["object_id"],
            operation=d["operation"],
            field_path=d["field_path"],
            old_value=d.get("old_value"),
            new_value=d.get("new_value"),
            triggered_by_text=d.get("triggered_by_text", ""),
            timestamp=d.get("timestamp", ""),
        )


# ---------- Session subgraph --------------------------------------------- #

@dataclass
class SessionEdge:
    """A relationship inside the session subgraph. Mirrors graph_core.Edge
    shape so the front-end's session-rendering code can consume both."""
    src: str
    dst: str
    relation: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @staticmethod
    def from_dict(d: Mapping[str, Any]) -> "SessionEdge":
        return SessionEdge(
            src=d["src"],
            dst=d["dst"],
            relation=d["relation"],
            metadata=dict(d.get("metadata", {})),
        )


@dataclass
class StrategyNode:
    """A proven reasoning recipe extracted from a successful session."""
    id: str
    question_pattern: str
    domain_keywords: List[str]
    plan_template: List[str]
    key_node_ids: List[str]
    key_node_rationales: Dict[str, str]
    workspace_schema: List[Dict[str, Any]]
    pitfalls: List[Dict[str, str]]
    effective_queries: List[str]
    session_stats: Dict[str, Any]
    provenance: Provenance
    task_family: str = ""
    task_subtype: str = ""
    question_mode: str = ""
    entry_conditions: Dict[str, Any] = field(default_factory=dict)
    slot_order: List[str] = field(default_factory=list)
    checkpoint_plan: List[str] = field(default_factory=list)
    stop_conditions: List[str] = field(default_factory=list)
    forbidden_finalize_conditions: List[str] = field(default_factory=list)
    strategy_schema_version: int = 2
    node_type: NodeType = "strategy"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @staticmethod
    def from_dict(d: Mapping[str, Any]) -> "StrategyNode":
        return StrategyNode(
            id=d["id"],
            question_pattern=d["question_pattern"],
            domain_keywords=list(d.get("domain_keywords", [])),
            plan_template=list(d.get("plan_template", [])),
            key_node_ids=list(d.get("key_node_ids", [])),
            key_node_rationales=dict(d.get("key_node_rationales", {})),
            workspace_schema=list(d.get("workspace_schema", [])),
            pitfalls=list(d.get("pitfalls", [])),
            effective_queries=list(d.get("effective_queries", [])),
            session_stats=dict(d.get("session_stats", {})),
            provenance=Provenance.from_dict(d["provenance"]),
            task_family=d.get("task_family", ""),
            task_subtype=d.get("task_subtype", ""),
            question_mode=d.get("question_mode", ""),
            entry_conditions=dict(d.get("entry_conditions", {})),
            slot_order=list(d.get("slot_order", [])),
            checkpoint_plan=list(d.get("checkpoint_plan", [])),
            stop_conditions=list(d.get("stop_conditions", [])),
            forbidden_finalize_conditions=list(d.get("forbidden_finalize_conditions", [])),
            strategy_schema_version=int(d.get("strategy_schema_version", 2)),
            node_type=d.get("node_type", "strategy"),
        )


@dataclass
class SolvedSubgoalNode:
    """A reusable answer to a typed subproblem with context guards."""
    id: str
    summary: str
    subgoal_signature: str
    question_type: str
    input_conditions: Dict[str, Any]
    output_slots: Dict[str, Any]
    valid_when: List[str]
    invalid_when: List[str]
    supporting_node_ids: List[str]
    confidence: float
    source_sessions: List[str]
    provenance: Provenance
    node_type: NodeType = "solved_subgoal"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @staticmethod
    def from_dict(d: Mapping[str, Any]) -> "SolvedSubgoalNode":
        return SolvedSubgoalNode(
            id=d["id"],
            summary=d.get("summary", ""),
            subgoal_signature=d.get("subgoal_signature", ""),
            question_type=d.get("question_type", ""),
            input_conditions=dict(d.get("input_conditions", {})),
            output_slots=dict(d.get("output_slots", {})),
            valid_when=list(d.get("valid_when", [])),
            invalid_when=list(d.get("invalid_when", [])),
            supporting_node_ids=list(d.get("supporting_node_ids", [])),
            confidence=float(d.get("confidence", 0.5)),
            source_sessions=list(d.get("source_sessions", [])),
            provenance=Provenance.from_dict(d["provenance"]),
            node_type=d.get("node_type", "solved_subgoal"),
        )


@dataclass
class ReasoningAtomNode:
    """A reusable reasoning fragment that can fill explanation-style slots."""
    id: str
    atom_type: str
    claim: str
    reusable_for: List[str]
    dependencies: List[str]
    supporting_node_ids: List[str]
    confidence: float
    provenance: Provenance
    node_type: NodeType = "reasoning_atom"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @staticmethod
    def from_dict(d: Mapping[str, Any]) -> "ReasoningAtomNode":
        return ReasoningAtomNode(
            id=d["id"],
            atom_type=d.get("atom_type", ""),
            claim=d.get("claim", ""),
            reusable_for=list(d.get("reusable_for", [])),
            dependencies=list(d.get("dependencies", [])),
            supporting_node_ids=list(d.get("supporting_node_ids", [])),
            confidence=float(d.get("confidence", 0.5)),
            provenance=Provenance.from_dict(d["provenance"]),
            node_type=d.get("node_type", "reasoning_atom"),
        )


@dataclass
class ControlRuleNode:
    """A reusable controller policy for a task family."""
    id: str
    task_family: str
    guidance: str
    required_slots: List[str]
    optional_slots: List[str]
    forbidden_escalations: List[str]
    preferred_action_order: List[str]
    stop_condition: str
    provenance: Provenance
    node_type: NodeType = "control_rule"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @staticmethod
    def from_dict(d: Mapping[str, Any]) -> "ControlRuleNode":
        return ControlRuleNode(
            id=d["id"],
            task_family=d.get("task_family", ""),
            guidance=d.get("guidance", ""),
            required_slots=list(d.get("required_slots", [])),
            optional_slots=list(d.get("optional_slots", [])),
            forbidden_escalations=list(d.get("forbidden_escalations", [])),
            preferred_action_order=list(d.get("preferred_action_order", [])),
            stop_condition=d.get("stop_condition", ""),
            provenance=Provenance.from_dict(d["provenance"]),
            node_type=d.get("node_type", "control_rule"),
        )


@dataclass
class SignatureFamilyNode:
    """Family-level wrapper for related strategy/subgoal/provisional variants."""
    id: str
    semantic_type: str
    task_family: str
    family_label: str
    variant_ids: List[str]
    provenance: Provenance
    contested: bool = False
    dominant_variant_id: Optional[str] = None
    retrieval_tier: str = "gated"
    support_score: float = 0.0
    stability_score: float = 0.0
    risk_score: float = 0.0
    contradiction_score: float = 0.0
    bias_score: float = 0.0
    node_type: NodeType = "signature_family"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @staticmethod
    def from_dict(d: Mapping[str, Any]) -> "SignatureFamilyNode":
        return SignatureFamilyNode(
            id=d["id"],
            semantic_type=d.get("semantic_type", ""),
            task_family=d.get("task_family", ""),
            family_label=d.get("family_label", ""),
            variant_ids=list(d.get("variant_ids", [])),
            provenance=Provenance.from_dict(d["provenance"]),
            contested=bool(d.get("contested", False)),
            dominant_variant_id=d.get("dominant_variant_id"),
            retrieval_tier=d.get("retrieval_tier", "gated"),
            support_score=float(d.get("support_score", 0.0)),
            stability_score=float(d.get("stability_score", 0.0)),
            risk_score=float(d.get("risk_score", 0.0)),
            contradiction_score=float(d.get("contradiction_score", 0.0)),
            bias_score=float(d.get("bias_score", 0.0)),
            node_type=d.get("node_type", "signature_family"),
        )


@dataclass
class SignatureVariantNode:
    """Variant-level wrapper linked to a semantic strategy/subgoal/provisional node."""
    id: str
    family_id: str
    semantic_type: str
    semantic_node_id: str
    canonical_text: str
    task_family: str
    epistemic_status: str
    promotion_state: str
    retrieval_tier: str
    provenance: Provenance
    support_score: float = 0.0
    stability_score: float = 0.0
    risk_score: float = 0.0
    contradiction_score: float = 0.0
    bias_score: float = 0.0
    required_slots: List[str] = field(default_factory=list)
    support_node_ids: List[str] = field(default_factory=list)
    aliases: List[str] = field(default_factory=list)
    node_type: NodeType = "signature_variant"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @staticmethod
    def from_dict(d: Mapping[str, Any]) -> "SignatureVariantNode":
        return SignatureVariantNode(
            id=d["id"],
            family_id=d.get("family_id", ""),
            semantic_type=d.get("semantic_type", ""),
            semantic_node_id=d.get("semantic_node_id", ""),
            canonical_text=d.get("canonical_text", ""),
            task_family=d.get("task_family", ""),
            epistemic_status=d.get("epistemic_status", "provisional"),
            promotion_state=d.get("promotion_state", "blocked"),
            retrieval_tier=d.get("retrieval_tier", "gated"),
            provenance=Provenance.from_dict(d["provenance"]),
            support_score=float(d.get("support_score", 0.0)),
            stability_score=float(d.get("stability_score", 0.0)),
            risk_score=float(d.get("risk_score", 0.0)),
            contradiction_score=float(d.get("contradiction_score", 0.0)),
            bias_score=float(d.get("bias_score", 0.0)),
            required_slots=list(d.get("required_slots", [])),
            support_node_ids=list(d.get("support_node_ids", [])),
            aliases=list(d.get("aliases", [])),
            node_type=d.get("node_type", "signature_variant"),
        )


@dataclass
class SessionSubgraph:
    """The scratch graph for one reasoning session.

    Holds dynamically-created procedure invocations (SessionObjectNodes),
    failure patterns observed during reasoning, edges between them, and
    the audit log of every state mutation.

    Persists as one JSON file (subgraph.json) plus one JSONL file
    (audit_log.jsonl) under data/session_subgraphs/{session_id}/.
    """
    session_id: str
    query: str
    graph_id: str                     # ref to the long-term graph this session ran against
    nodes: Dict[str, Any] = field(default_factory=dict)  # node_id -> node dict
    edges: List[SessionEdge] = field(default_factory=list)
    audit_log: List[AuditEntry] = field(default_factory=list)
    step_count: int = 0
    started_at: str = ""
    ended_at: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "session_id": self.session_id,
            "query": self.query,
            "graph_id": self.graph_id,
            "nodes": self.nodes,
            "edges": [e.to_dict() for e in self.edges],
            "audit_log": [a.to_dict() for a in self.audit_log],
            "step_count": self.step_count,
            "started_at": self.started_at,
            "ended_at": self.ended_at,
        }

    @staticmethod
    def from_dict(d: Mapping[str, Any]) -> "SessionSubgraph":
        return SessionSubgraph(
            session_id=d["session_id"],
            query=d["query"],
            graph_id=d["graph_id"],
            nodes=dict(d.get("nodes", {})),
            edges=[SessionEdge.from_dict(e) for e in d.get("edges", [])],
            audit_log=[AuditEntry.from_dict(a) for a in d.get("audit_log", [])],
            step_count=int(d.get("step_count", 0)),
            started_at=d.get("started_at", ""),
            ended_at=d.get("ended_at"),
        )


# ---------------------------------------------------------------------------
# V5: ReasoningChainNode
# ---------------------------------------------------------------------------

@dataclass
class ReasoningChainNode:
    """A multi-hop deductive path captured as a named, reusable graph node.

    Distinct from StrategyNode (which is a *recipe* for how to reason) and
    SolvedSubgoalNode (which is a *cached answer* to a sub-problem).
    A ReasoningChainNode captures the *logical structure* of a deduction:

        premise_a  → (entails)  →  intermediate_b  → (entails)  →  conclusion_c

    This makes multi-hop reasoning patterns reusable. If the same A→B→C chain
    recurs across different questions, the model can attend to this node and
    shortcut the full derivation.

    GNN role:
        - Attended to during the Layer 8 planning pass (provides deductive
          scaffolding alongside StrategyNodes and FailurePatternNodes).
        - Edges: chain_step edges (relation="chain_step", ordered by step_index)
          connect this node to each intermediate fact/claim node.
        - context_guard on the Node itself restricts attention to matching
          task families.

    Lifecycle:
        Created by post_processing.extract_reasoning_chain() on successful
        multi-step sessions (>= 2 graph reads that form an entailment path).
        Requires human review (tier="review") before promotion to "supported".
    """
    id: str

    # Human-readable description of the full deductive path.
    chain_text: str

    # Ordered list of node IDs that form the chain: [premise_id, step1_id, ..., conclusion_id]
    # Minimum 2 nodes (premise + conclusion). Each consecutive pair must have an
    # 'entails' or 'supports' edge in the main MemoryGraph.
    chain_step_ids: List[str]

    # The final conclusion stated as a standalone claim.
    conclusion: str

    # Domain keywords for TF-IDF retrieval matching (same role as StrategyNode.domain_keywords).
    domain_keywords: List[str] = field(default_factory=list)

    # Which task families this chain is valid for. Empty = all families.
    applicable_task_families: List[str] = field(default_factory=list)

    # Number of times this chain was successfully used across sessions.
    # Incremented by post_processing at session end (same as Node.access_count).
    access_count: int = 0

    # Session that first produced this chain.
    source_session_id: str = ""

    # Schema version for forward-compatibility.
    chain_schema_version: int = 1

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @staticmethod
    def from_dict(d: Mapping[str, Any]) -> "ReasoningChainNode":
        return ReasoningChainNode(
            id=d["id"],
            chain_text=d.get("chain_text", ""),
            chain_step_ids=list(d.get("chain_step_ids", [])),
            conclusion=d.get("conclusion", ""),
            domain_keywords=list(d.get("domain_keywords", [])),
            applicable_task_families=list(d.get("applicable_task_families", [])),
            access_count=int(d.get("access_count", 0)),
            source_session_id=d.get("source_session_id", ""),
            chain_schema_version=int(d.get("chain_schema_version", 1)),
        )
