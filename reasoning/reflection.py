"""Session reflection — an LLM call that articulates what the model learned.

This module decouples post-processing from the inline reasoning loop. The
deterministic Phase-11 extractor (`extract_learning_report`) walks the
session mechanically; this reflection step asks the model to STATE what it
learned in structured form. The two are complementary:

  - Extractor catches things the model did (cite, hypothesize, fail-record).
  - Reflection catches things the model REALIZED (new conclusions, novel
    relationships) that aren't visible in the tool log alone.

The reflection output is a structured block (XML tags inside <learning>...
</learning>) that a downstream graph editor (`graph_editor.py`) parses into
edit operations. The model does NOT directly mutate the graph; it only
articulates candidates. Application is a separate, deterministic step.

This file is the LLM-facing half. The deterministic parser + applier lives
in `reasoning/graph_editor.py`.
"""
from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Protocol, Sequence


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class LLMController(Protocol):
    """Minimal interface a controller must expose for reflection.

    answerer_v4's V4LlamaServerController / V4OpencodeController / V4GeminiController
    all conform. The key requirement is chat_oneshot (so reflection runs in
    its OWN session, not polluted by the main reasoning loop's history).
    """
    def chat_oneshot(self, messages: List[Dict[str, str]]) -> Dict[str, Any]: ...


# ---------------------------------------------------------------------------
# ReflectionResult dataclass + serialization
# ---------------------------------------------------------------------------

@dataclass
class NewFact:
    """A claim the model wants added to the graph as a new node."""
    text: str
    evidence_node_ids: List[str] = field(default_factory=list)
    confidence: str = "medium"          # "low" | "medium" | "high"


@dataclass
class NewRelationship:
    """A relationship the model wants added as an edge between existing nodes."""
    src: str
    dst: str
    relation: str
    rationale: str = ""


@dataclass
class FailedApproach:
    """An approach that was tried and didn't work. Same shape as Phase-11 failure."""
    approach: str
    condition: str
    mechanism: str
    replacement_suggestion: Optional[str] = None  # what to do instead


@dataclass
class ReinforcedNode:
    """An existing node the model wants to flag as further validated by this session."""
    node_id: str
    rationale: str = ""


@dataclass
class NodeUpdate:
    """Proposed enrichment or correction to an existing node."""
    node_id: str
    mode: str       # "append" | "replace"
    text: str


@dataclass
class NodeDeprecation:
    """Proposed deprecation of an existing node."""
    node_id: str
    reason: str
    successor_id: Optional[str] = None


@dataclass
class Implementation:
    """Code or algorithm template discovered during reasoning."""
    text: str
    language: str = "pseudocode"
    evidence_node_ids: List[str] = field(default_factory=list)


@dataclass
class WorkedExample:
    """Step-by-step walkthrough the model produced."""
    text: str
    problem: str = ""
    evidence_node_ids: List[str] = field(default_factory=list)


@dataclass
class ReflectionResult:
    session_id: str
    new_facts: List[NewFact] = field(default_factory=list)
    new_relationships: List[NewRelationship] = field(default_factory=list)
    failed_approaches: List[FailedApproach] = field(default_factory=list)
    reinforced_nodes: List[ReinforcedNode] = field(default_factory=list)
    updates: List[NodeUpdate] = field(default_factory=list)
    deprecations: List[NodeDeprecation] = field(default_factory=list)
    implementations: List[Implementation] = field(default_factory=list)
    worked_examples: List[WorkedExample] = field(default_factory=list)
    raw_reflection_text: str = ""
    parse_errors: List[str] = field(default_factory=list)
    timestamp: str = field(default_factory=_now_iso)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "session_id": self.session_id,
            "new_facts": [asdict(f) for f in self.new_facts],
            "new_relationships": [asdict(r) for r in self.new_relationships],
            "failed_approaches": [asdict(f) for f in self.failed_approaches],
            "updates": [asdict(u) for u in self.updates],
            "deprecations": [asdict(d) for d in self.deprecations],
            "implementations": [asdict(i) for i in self.implementations],
            "worked_examples": [asdict(w) for w in self.worked_examples],
            "reinforced_nodes": [asdict(r) for r in self.reinforced_nodes],
            "raw_reflection_text": self.raw_reflection_text,
            "parse_errors": list(self.parse_errors),
            "timestamp": self.timestamp,
        }


# ---------------------------------------------------------------------------
# Reflection prompt
# ---------------------------------------------------------------------------

_REFLECTION_SYSTEM = """\
You are a knowledge-distillation assistant. A reasoning session has just
finished. Your job is to ARTICULATE what new knowledge from that session
should be added to the long-term knowledge graph for future sessions to
benefit from.

Be exhaustive. Err on the side of listing MORE rather than less. A separate
deterministic component will validate and apply the edits — your job is to
SURFACE candidates, not to filter them.

Output exactly this structure (no other commentary):

<learning>

<new_fact confidence="high|medium|low" evidence="node_id_a,node_id_b">
Plain-language claim that should be added as a new graph node. State it as
a complete sentence with concrete content. Evidence is a comma-separated
list of existing graph node IDs that support this claim (from the trace).
</new_fact>

<new_fact confidence="..." evidence="...">
Another claim.
</new_fact>

<new_relationship src="existing_node_id_a" dst="existing_node_id_b" relation="supports|contradicts|specializes|generalizes|example_of|prerequisite_for">
One-line rationale for why this edge should exist.
</new_relationship>

<failed_approach approach="..." condition="..." mechanism="..." replacement="optional_node_id_or_text">
An approach that was tried in this session and failed. Replacement is
optional — point to a node or describe the better approach if you know it.
</failed_approach>

<reinforced node_id="existing_node_id">
This existing node was further validated in this session. One-line note on
how (e.g., "used in a passing chain of reasoning for the leaderboard design").
</reinforced>

<update_node node_id="existing_node_id" mode="append|replace">
Additional detail, correction, or enrichment for an existing node that is
currently too sparse or slightly wrong. Use mode="append" to add to the
existing text, or mode="replace" to correct it.
</update_node>

<deprecate node_id="existing_node_id" successor="optional_replacement_node_id">
Reason why this node is wrong, outdated, or superseded by a better version.
</deprecate>

<implementation language="c++|python|pseudocode" evidence="node_id_a,node_id_b">
A code template, algorithm implementation, or concrete procedure that was
produced or discovered during this session. Include the full code.
</implementation>

<worked_example evidence="node_id_a" problem="short problem description">
A step-by-step walkthrough produced during reasoning that demonstrates how
to solve a specific problem. Include the full walkthrough.
</worked_example>

</learning>

Rules:
  - Use ONLY the structure above. No extra prose outside <learning>.
  - For evidence / node_id attributes, use node IDs you actually saw in the
    trace (read_node calls, anchors list, citations). Do not invent IDs.
  - It is fine to emit zero of any block type. An empty <learning></learning>
    is valid when nothing new was learned.
  - new_fact texts should be ATOMIC (one claim each), not paragraphs.
  - implementation and worked_example texts CAN be multi-line (include full code/walkthrough).
  - update_node MUST reference an existing node_id from the trace.
"""


def _format_reflection_input(
    *,
    question: str,
    anchors: Sequence[str],
    tool_log: Sequence[Mapping[str, Any]],
    hypotheses: Mapping[str, Mapping[str, Any]],
    failures: Sequence[Any],
    objects: Mapping[str, Any],
    polished_answer: str,
) -> str:
    """Assemble the user message for the reflection call."""
    parts: List[str] = []
    parts.append(f"# Session question\n{question}\n")
    if anchors:
        parts.append(f"# Initial anchors\n{', '.join(anchors)}\n")

    if tool_log:
        parts.append("# Tool calls (chronological)")
        for entry in tool_log:
            name = entry.get("name", "?")
            args = entry.get("args", {})
            args_short = json.dumps(args, ensure_ascii=False)[:200]
            summary = (entry.get("result_summary") or "")[:200]
            parts.append(f"  - {name}({args_short}) → {summary}")
        parts.append("")

    if hypotheses:
        parts.append("# Hypotheses raised")
        for hid, h in hypotheses.items():
            verdict = h.get("verdict") or "(unverified)"
            ev = (h.get("evidence") or "")[:200]
            parts.append(f"  - {hid} [{verdict}]: {h.get('text','')[:200]}")
            if ev:
                parts.append(f"      evidence: {ev}")
        parts.append("")

    if failures:
        parts.append("# Failures recorded during the session")
        for fr in failures:
            approach = getattr(fr, "approach", "")
            condition = getattr(fr, "condition", "")
            mechanism = getattr(fr, "mechanism", "")
            parts.append(f"  - {approach!r} fails when {condition!r}: {mechanism}")
        parts.append("")

    if objects:
        parts.append("# Session workspace objects (final state)")
        for v4_id, obj in objects.items():
            name = getattr(obj, "name", "")
            state = getattr(obj, "state", {})
            state_short = json.dumps(state, ensure_ascii=False)[:400]
            parts.append(f"  - {name} ({v4_id}): {state_short}")
        parts.append("")

    if polished_answer:
        parts.append(f"# Final answer delivered to the user\n{polished_answer}\n")

    parts.append("Now produce the <learning>...</learning> block.")
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

_LEARNING_RE = re.compile(r"<learning>(.*?)</learning>", re.DOTALL | re.IGNORECASE)

_NEW_FACT_RE = re.compile(
    r"<new_fact(?P<attrs>[^>]*)>(?P<body>.*?)</new_fact>", re.DOTALL | re.IGNORECASE,
)
_NEW_REL_RE = re.compile(
    r"<new_relationship(?P<attrs>[^>]*)>(?P<body>.*?)</new_relationship>",
    re.DOTALL | re.IGNORECASE,
)
_FAILED_RE = re.compile(
    r"<failed_approach(?P<attrs>[^>]*?)\s*/?>(?P<body>.*?)(?:</failed_approach>|(?=<|$))",
    re.DOTALL | re.IGNORECASE,
)
_REINF_RE = re.compile(
    r"<reinforced(?P<attrs>[^>]*?)\s*/?>(?P<body>.*?)(?:</reinforced>|(?=<|$))",
    re.DOTALL | re.IGNORECASE,
)
_UPDATE_RE = re.compile(
    r"<update_node(?P<attrs>[^>]*)>(?P<body>.*?)</update_node>",
    re.DOTALL | re.IGNORECASE,
)
_DEPRECATE_RE = re.compile(
    r"<deprecate(?P<attrs>[^>]*?)\s*/?>(?P<body>.*?)(?:</deprecate>|(?=<|$))",
    re.DOTALL | re.IGNORECASE,
)
_IMPL_RE = re.compile(
    r"<implementation(?P<attrs>[^>]*)>(?P<body>.*?)</implementation>",
    re.DOTALL | re.IGNORECASE,
)
_EXAMPLE_RE = re.compile(
    r"<worked_example(?P<attrs>[^>]*)>(?P<body>.*?)</worked_example>",
    re.DOTALL | re.IGNORECASE,
)
_ATTR_RE = re.compile(r'(\w+)\s*=\s*"([^"]*)"', re.IGNORECASE)


def _attrs(s: str) -> Dict[str, str]:
    return {k.lower(): v for k, v in _ATTR_RE.findall(s or "")}


def parse_reflection(text: str, session_id: str) -> ReflectionResult:
    """Parse the model's <learning>...</learning> block into a ReflectionResult."""
    result = ReflectionResult(session_id=session_id, raw_reflection_text=text)
    m = _LEARNING_RE.search(text or "")
    if not m:
        result.parse_errors.append("no <learning>...</learning> block found")
        return result
    inner = m.group(1)

    for fm in _NEW_FACT_RE.finditer(inner):
        attrs = _attrs(fm.group("attrs"))
        body = (fm.group("body") or "").strip()
        if not body:
            continue
        ev = [e.strip() for e in (attrs.get("evidence") or "").split(",") if e.strip()]
        result.new_facts.append(NewFact(
            text=body,
            evidence_node_ids=ev,
            confidence=(attrs.get("confidence") or "medium").lower(),
        ))

    for rm in _NEW_REL_RE.finditer(inner):
        attrs = _attrs(rm.group("attrs"))
        src = attrs.get("src", "").strip()
        dst = attrs.get("dst", "").strip()
        if not src or not dst:
            result.parse_errors.append(f"new_relationship missing src/dst: {attrs}")
            continue
        result.new_relationships.append(NewRelationship(
            src=src, dst=dst,
            relation=(attrs.get("relation") or "related").strip(),
            rationale=(rm.group("body") or "").strip(),
        ))

    for fa in _FAILED_RE.finditer(inner):
        attrs = _attrs(fa.group("attrs"))
        approach = (attrs.get("approach") or "").strip()
        condition = (attrs.get("condition") or "").strip()
        mechanism = (attrs.get("mechanism") or "").strip()
        if not approach:
            continue
        result.failed_approaches.append(FailedApproach(
            approach=approach,
            condition=condition,
            mechanism=mechanism,
            replacement_suggestion=attrs.get("replacement") or None,
        ))

    for rn in _REINF_RE.finditer(inner):
        attrs = _attrs(rn.group("attrs"))
        nid = (attrs.get("node_id") or "").strip()
        if not nid:
            continue
        result.reinforced_nodes.append(ReinforcedNode(
            node_id=nid,
            rationale=(rn.group("body") or "").strip(),
        ))

    for um in _UPDATE_RE.finditer(inner):
        attrs = _attrs(um.group("attrs"))
        nid = (attrs.get("node_id") or "").strip()
        mode = (attrs.get("mode") or "append").strip().lower()
        body = (um.group("body") or "").strip()
        if not nid or not body:
            continue
        result.updates.append(NodeUpdate(node_id=nid, mode=mode, text=body))

    for dm in _DEPRECATE_RE.finditer(inner):
        attrs = _attrs(dm.group("attrs"))
        nid = (attrs.get("node_id") or "").strip()
        body = (dm.group("body") or "").strip()
        if not nid:
            continue
        result.deprecations.append(NodeDeprecation(
            node_id=nid, reason=body,
            successor_id=attrs.get("successor") or None,
        ))

    for im in _IMPL_RE.finditer(inner):
        attrs = _attrs(im.group("attrs"))
        body = (im.group("body") or "").strip()
        if not body:
            continue
        ev = [e.strip() for e in (attrs.get("evidence") or "").split(",") if e.strip()]
        result.implementations.append(Implementation(
            text=body,
            language=(attrs.get("language") or "pseudocode").strip(),
            evidence_node_ids=ev,
        ))

    for wm in _EXAMPLE_RE.finditer(inner):
        attrs = _attrs(wm.group("attrs"))
        body = (wm.group("body") or "").strip()
        if not body:
            continue
        ev = [e.strip() for e in (attrs.get("evidence") or "").split(",") if e.strip()]
        result.worked_examples.append(WorkedExample(
            text=body,
            problem=(attrs.get("problem") or "").strip(),
            evidence_node_ids=ev,
        ))
    return result


# ---------------------------------------------------------------------------
# Public entry points
# ---------------------------------------------------------------------------

def run_reflection(
    *,
    controller: LLMController,
    session_id: str,
    question: str,
    anchors: Sequence[str],
    tool_log: Sequence[Mapping[str, Any]],
    hypotheses: Mapping[str, Mapping[str, Any]],
    failures: Sequence[Any],
    objects: Mapping[str, Any],
    polished_answer: str,
) -> ReflectionResult:
    """Run the reflection LLM call inline. Returns a parsed ReflectionResult.

    Catches all exceptions — reflection failures should never break the
    main flow. On failure the result has parse_errors populated.
    """
    try:
        user_msg = _format_reflection_input(
            question=question,
            anchors=anchors,
            tool_log=tool_log,
            hypotheses=hypotheses,
            failures=failures,
            objects=objects,
            polished_answer=polished_answer,
        )
        resp = controller.chat_oneshot([
            {"role": "system", "content": _REFLECTION_SYSTEM},
            {"role": "user", "content": user_msg},
        ])
        content = resp["choices"][0]["message"]["content"]
        return parse_reflection(content, session_id=session_id)
    except Exception as e:
        result = ReflectionResult(session_id=session_id, raw_reflection_text="")
        result.parse_errors.append(f"reflection call failed: {type(e).__name__}: {e}")
        return result


def reflect_from_session_dir(
    session_dir: Path,
    controller: LLMController,
) -> ReflectionResult:
    """Offline entry: re-process a persisted session by reading its files.

    Used by scripts/process_session.py to apply a new reflection prompt
    against an old session, or to re-run reflection after a prompt update.
    """
    session_dir = Path(session_dir)
    subgraph = json.loads((session_dir / "subgraph.json").read_text(encoding="utf-8"))
    session_id = subgraph.get("session_id", session_dir.name)
    question = subgraph.get("query", "")

    # Try to also load the polished answer + tool log from extra artifacts.
    polished_answer = ""
    polish_path = session_dir / "polish_post_strip.txt"
    if polish_path.exists():
        polished_answer = polish_path.read_text(encoding="utf-8")

    # Reconstruct tool_log + hypotheses + failures + objects from learning_report
    # (the audit log is mutation-only; the learning report carries the v4 view).
    tool_log: List[Dict[str, Any]] = []
    hypotheses: Dict[str, Dict[str, Any]] = {}
    failures: List[Any] = []
    objects: Dict[str, Any] = {}
    anchors: List[str] = []
    lr_path = session_dir / "learning_report.json"
    if lr_path.exists():
        lr = json.loads(lr_path.read_text(encoding="utf-8"))
        anchors = lr.get("cited_node_ids", [])
        for vc in lr.get("verified_claims", []):
            hypotheses[vc["hypothesis_id"]] = {
                "text": vc["text"],
                "verdict": "verified",
                "evidence": vc.get("evidence", ""),
            }
        for fr in lr.get("recorded_failures", []):
            # Synthetic object that has .approach / .condition / .mechanism for
            # _format_reflection_input.
            class _F: pass
            f = _F()
            f.approach = fr.get("approach", "")
            f.condition = fr.get("condition", "")
            f.mechanism = fr.get("mechanism", "")
            failures.append(f)
        for so in lr.get("synthesized_objects", []):
            class _O: pass
            o = _O()
            o.name = so.get("name", "")
            o.state = so.get("state", {})
            o.fields = so.get("fields", [])
            objects[so.get("v4_id", "?")] = o

    result = run_reflection(
        controller=controller,
        session_id=session_id,
        question=question,
        anchors=anchors,
        tool_log=tool_log,
        hypotheses=hypotheses,
        failures=failures,
        objects=objects,
        polished_answer=polished_answer,
    )

    # Persist the reflection for posterity.
    (session_dir / "reflection.json").write_text(
        json.dumps(result.to_dict(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return result
