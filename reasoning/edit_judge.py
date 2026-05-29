"""LLM-as-judge for proposed graph edits.

Before a non-trivial edit (add_node, update_node, deprecate_node) is applied,
a short LLM call evaluates: is this edit novel, correct, and consistent with
the existing graph neighborhood?

The judge sees the proposed edit + the 3 nearest existing nodes (from the
semantic dedupe index) + their 1-hop edges. It returns accept / reject /
merge_into_existing with a rationale.

This replaces heuristic gates with principled model-based evaluation.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Protocol, Sequence

from graph_core import MemoryGraph


class LLMController(Protocol):
    def chat_oneshot(self, messages: List[Dict[str, str]]) -> Dict[str, Any]: ...


@dataclass
class JudgeVerdict:
    decision: str          # "accept" | "reject" | "merge_into"
    merge_target: Optional[str] = None   # node_id to merge into
    rationale: str = ""


_JUDGE_SYSTEM = """\
You are a knowledge-graph quality judge. You will be shown a proposed edit
to a knowledge graph and the 3 most similar existing nodes.

Evaluate:
  (a) Is this edit NOVEL — does it add information not already captured?
  (b) Is it CORRECT — is the content factually accurate on its face?
  (c) Is it CONSISTENT — does it align with or properly extend existing nodes?

Output EXACTLY one verdict tag:

  <verdict decision="accept">Short rationale.</verdict>
  <verdict decision="reject">Why: duplicate / incorrect / inconsistent.</verdict>
  <verdict decision="merge_into" merge_target="existing_node_id">
    This content should be merged into the existing node instead of added separately.
  </verdict>

Rules:
  - If the proposed text is a near-paraphrase of an existing node, use merge_into.
  - If the proposed text contradicts an existing node without explanation, reject.
  - If the proposed text adds genuinely new, accurate information, accept.
  - When merging, set merge_target to the most relevant existing node's ID.
"""

_VERDICT_RE = re.compile(
    r'<verdict\s+decision="(?P<decision>accept|reject|merge_into)"'
    r'(?:\s+merge_target="(?P<merge>[^"]*)")?'
    r'\s*>(?P<rationale>.*?)</verdict>',
    re.DOTALL | re.IGNORECASE,
)


def _format_judge_input(
    edit: Dict[str, Any],
    neighbors: List[Dict[str, Any]],
) -> str:
    """Build the user message for the judge."""
    parts = []
    parts.append("## Proposed edit")
    parts.append(f"  op: {edit.get('op')}")
    parts.append(f"  node_type: {edit.get('node_type', '?')}")
    parts.append(f"  text: {edit.get('text', '')[:500]}")
    if edit.get("metadata"):
        meta_short = {k: v for k, v in edit["metadata"].items()
                      if k in ("confidence", "attempted_approach", "failure_condition", "language", "problem")}
        if meta_short:
            parts.append(f"  metadata: {json.dumps(meta_short, ensure_ascii=False)}")
    parts.append("")
    parts.append("## 3 most similar existing nodes")
    for i, nb in enumerate(neighbors):
        parts.append(f"  [{i+1}] id={nb['id']}  type={nb.get('type', '?')}  sim={nb.get('sim', 0):.3f}")
        parts.append(f"      text: {nb.get('text', '')[:300]}")
        if nb.get("edges"):
            parts.append(f"      edges: {', '.join(nb['edges'][:5])}")
    parts.append("")
    parts.append("Produce your <verdict> now.")
    return "\n".join(parts)


def judge_edit(
    edit: Dict[str, Any],
    graph: MemoryGraph,
    dedupe_index: Any,  # DedupeIndex
    controller: LLMController,
) -> JudgeVerdict:
    """Run the LLM judge on a single proposed edit.

    Returns JudgeVerdict. On failure (parse error, LLM error), defaults to
    accept — fail-open so edits aren't silently dropped by judge bugs.
    """
    text = edit.get("text", "")
    if not text:
        return JudgeVerdict(decision="accept", rationale="no text to judge")

    # Get the 3 nearest neighbors for context.
    top3 = dedupe_index.query_topk(text, k=3)
    neighbors = []
    for m in top3:
        node = graph.nodes.get(m.node_id)
        if node is None:
            continue
        # 1-hop edges for context
        edge_descs = []
        for e in graph.edges:
            if e.src == m.node_id:
                edge_descs.append(f"--{e.relation}--> {e.dst}")
            elif e.dst == m.node_id:
                edge_descs.append(f"<--{e.relation}-- {e.src}")
            if len(edge_descs) >= 5:
                break
        neighbors.append({
            "id": m.node_id,
            "type": node.node_type,
            "sim": m.similarity,
            "text": node.text,
            "edges": edge_descs,
        })

    user_msg = _format_judge_input(edit, neighbors)
    try:
        resp = controller.chat_oneshot([
            {"role": "system", "content": _JUDGE_SYSTEM},
            {"role": "user", "content": user_msg},
        ])
        content = resp["choices"][0]["message"]["content"]
    except Exception as e:
        return JudgeVerdict(decision="accept", rationale=f"judge call failed: {e}")

    m = _VERDICT_RE.search(content)
    if not m:
        return JudgeVerdict(decision="accept", rationale=f"judge returned unparseable: {content[:200]}")

    return JudgeVerdict(
        decision=m.group("decision"),
        merge_target=m.group("merge") or None,
        rationale=m.group("rationale").strip(),
    )


def judge_edits_batch(
    edits: List[Dict[str, Any]],
    graph: MemoryGraph,
    dedupe_index: Any,
    controller: LLMController,
    *,
    judge_ops: Sequence[str] = ("add_node", "update_node", "deprecate_node"),
) -> List[Dict[str, Any]]:
    """Judge a batch of edits, returning the filtered + modified list.

    Edits with ops NOT in judge_ops pass through unchanged.
    Rejected edits are removed. merge_into edits are converted to update_node.
    """
    result = []
    for edit in edits:
        if edit.get("op") not in judge_ops or not edit.get("needs_judge"):
            result.append(edit)
            continue
        verdict = judge_edit(edit, graph, dedupe_index, controller)
        edit["_judge_verdict"] = verdict.decision
        edit["_judge_rationale"] = verdict.rationale
        if verdict.decision == "reject":
            continue
        if verdict.decision == "merge_into" and verdict.merge_target:
            result.append({
                "op": "update_node",
                "node_id": verdict.merge_target,
                "append_text": edit.get("text", ""),
                "metadata_patch": {
                    "merged_from_session": edit.get("metadata", {}).get("source_session", ""),
                    "merged_edit_type": edit.get("node_type", "claim"),
                },
                "tier": "mutate",
                "_judge_verdict": "merge_into",
                "_judge_rationale": verdict.rationale,
            })
        else:
            result.append(edit)
    return result
