from __future__ import annotations

import json
import os
import re
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Set

from graph_core import MemoryGraph, lexical_overlap

os.environ.setdefault("HF_HOME", os.path.join(os.getcwd(), "cache"))


_CRITIC_SCHEMA = """\
{
  "verdict": "accept | revise | reject",
  "should_reject": false,
  "confidence": 0.0,
  "checks": {
    "action_matches_signal": true,
    "new_node_is_not_duplicate": true,
    "uses_existing_ids_only": true,
    "edges_are_topic_compatible": true,
    "not_cross_domain": true,
    "not_signal_restatement": true
  },
  "check_explanations": {
    "action_matches_signal": "one short reason",
    "new_node_is_not_duplicate": "one short reason",
    "uses_existing_ids_only": "one short reason",
    "edges_are_topic_compatible": "one short reason",
    "not_cross_domain": "one short reason",
    "not_signal_restatement": "one short reason"
  },
  "mistakes": [
    {
      "type": "duplicate | wrong_action | bad_edge | missing_edge | hallucinated_id | cross_domain | conflict_missed | no_op_missed | weak_reasoning",
      "evidence": "short evidence grounded in the signal or retrieved node ids",
      "affected_ids": ["id1", "id2"]
    }
  ],
  "suggested_fix": {
    "action": "same | no_op | add_node | update_node | link_nodes | create_bridge | resolve_conflict | summarize_cluster",
    "proposed": {}
  }
}
"""


_CRITIC_SYSTEM = """\
You are a strict graph-edit critic. Your job is to find mistakes in a proposed knowledge-graph edit.

You are NOT the planner. Do not design a new graph from scratch. Review only the proposed action.

Check for these mistakes:
- wrong_action: action type does not match the signal and retrieved graph context
- duplicate: add_node repeats an existing node instead of adding new knowledge
- bad_edge: edge target is weak, unrelated, hub-only, or cross-domain
- missing_edge: add_node should attach to a useful retrieved non-hub node but does not
- hallucinated_id: proposal references a node id not shown in the retrieved context
- conflict_missed: false or contradicted signal should resolve_conflict instead of add_node
- no_op_missed: signal is already covered but proposal edits the graph
- weak_reasoning: reasoning does not support the final action

For every check, set true only when the proposed edit clearly improves graph quality.
If any check is false, explain it in check_explanations and add a matching mistake.
Do not copy placeholder strings from the schema. Every explanation must name a concrete signal phrase or node id.
Output ONLY valid JSON. The JSON MUST include top-level "verdict", "should_reject", and "checks" fields.
"""


_CRITIC_USER = """\
Signal:
{signal}

Retrieved context:
{context}

Planner action:
{action}

Planner reasoning:
{reasoning}

Planner proposed JSON:
{proposed}

Planner validation:
{validation}

Review the planner proposal. Output only the JSON review with this exact schema:
{schema}
"""


@dataclass
class CriticMistake:
    type: str
    evidence: str = ""
    affected_ids: List[str] = field(default_factory=list)


@dataclass
class CriticResult:
    verdict: str = "accept"
    mistakes: List[CriticMistake] = field(default_factory=list)
    suggested_fix: Dict[str, Any] = field(default_factory=lambda: {"action": "same", "proposed": {}})
    checks: Dict[str, bool] = field(default_factory=dict)
    check_explanations: Dict[str, str] = field(default_factory=dict)
    deterministic_checks: Dict[str, bool] = field(default_factory=dict)
    deterministic_check_explanations: Dict[str, str] = field(default_factory=dict)
    should_reject: bool = False
    confidence: float = 0.0
    raw_output: str = ""
    parse_ok: bool = True
    validation_errors: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class GraphCritic:
    def __init__(
        self,
        model_name: str,
        *,
        device: str = "auto",
        quantize_4bit: bool = True,
        cache_dir: str = "./cache",
        max_new_tokens: int = 512,
        local_files_only: bool = False,
    ) -> None:
        self.model_name = model_name
        self.device = device
        self.quantize_4bit = quantize_4bit
        self.cache_dir = cache_dir
        self.max_new_tokens = max_new_tokens
        self.local_files_only = local_files_only
        self._model = None
        self._tokenizer = None

    def load(self) -> "GraphCritic":
        if self._model is None:
            self._load_model()
        return self

    def _load_model(self) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        print(f"[critic] loading {self.model_name!r}  4bit={self.quantize_4bit}", flush=True)
        t0 = time.perf_counter()
        load_kw: Dict[str, Any] = {
            "trust_remote_code": True,
            "cache_dir": self.cache_dir,
            "local_files_only": self.local_files_only,
        }
        if self.quantize_4bit and torch.cuda.is_available():
            try:
                from transformers import BitsAndBytesConfig

                load_kw["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                )
                load_kw["device_map"] = self.device if self.device != "cpu" else None
            except Exception as exc:
                print(f"[critic] bitsandbytes unavailable ({exc!r}); using fp16", flush=True)
                load_kw["dtype"] = torch.float16
                self.quantize_4bit = False
        else:
            load_kw["dtype"] = torch.float16 if torch.cuda.is_available() else torch.float32
            if torch.cuda.is_available() and self.device not in {"cpu", None}:
                load_kw["device_map"] = self.device

        self._model = AutoModelForCausalLM.from_pretrained(self.model_name, **load_kw)
        if not load_kw.get("device_map"):
            target = self.device if self.device not in {"auto", None} else ("cuda" if torch.cuda.is_available() else "cpu")
            self._model = self._model.to(target)
        self._model.eval()

        self._tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True,
            cache_dir=self.cache_dir,
            local_files_only=self.local_files_only,
        )
        if self._tokenizer.pad_token_id is None:
            self._tokenizer.pad_token_id = self._tokenizer.eos_token_id
        print(f"[critic] ready in {time.perf_counter() - t0:.1f}s", flush=True)

    def review(
        self,
        *,
        signal_text: str,
        context_nodes: Sequence[Mapping[str, Any]],
        planner_action: str,
        planner_reasoning: str,
        planner_proposed: Mapping[str, Any],
        planner_validation: Mapping[str, Any],
        graph: Optional[MemoryGraph] = None,
    ) -> CriticResult:
        if self._model is None:
            self._load_model()
        context = self._format_context(context_nodes)
        user = _CRITIC_USER.format(
            signal=signal_text,
            context=context,
            action=planner_action,
            reasoning=planner_reasoning or "",
            proposed=json.dumps(dict(planner_proposed or {}), ensure_ascii=False, indent=2),
            validation=json.dumps(dict(planner_validation or {}), ensure_ascii=False, indent=2),
            schema=_CRITIC_SCHEMA,
        )
        raw = self._generate([
            {"role": "system", "content": _CRITIC_SYSTEM},
            {"role": "user", "content": user},
        ])
        result = self._parse_result(raw)
        if graph is not None:
            det_checks, det_explanations = deterministic_graph_checks(
                signal_text=signal_text,
                context_nodes=context_nodes,
                planner_action=planner_action,
                planner_proposed=planner_proposed,
                planner_validation=planner_validation,
                graph=graph,
            )
            result.deterministic_checks = det_checks
            result.deterministic_check_explanations = det_explanations
        return result

    def _format_context(self, nodes: Sequence[Mapping[str, Any]]) -> str:
        if not nodes:
            return "(none)"
        lines: List[str] = []
        for node in nodes[:10]:
            lines.append(
                f"- id={node.get('id')} type={node.get('node_type')} "
                f"score={float(node.get('score', 0.0) or 0.0):.3f} text={node.get('text')}"
            )
        return "\n".join(lines)

    def _is_thinking_model(self) -> bool:
        return "qwen3" in self.model_name.lower()

    def _generate(self, messages: List[Dict[str, str]]) -> str:
        import torch

        try:
            kw: Dict[str, Any] = {}
            if self._is_thinking_model():
                # Critic always runs in non-thinking mode: deterministic JSON output
                kw["enable_thinking"] = False
            prompt = self._tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True, **kw
            )
        except Exception:
            prompt = ""
            for msg in messages:
                prompt += f"### {msg.get('role', 'user').capitalize()}\n{msg.get('content', '')}\n\n"
            prompt += "### Assistant\n"

        device = next(self._model.parameters()).device
        inputs = self._tokenizer(prompt, return_tensors="pt", truncation=True, max_length=16384, padding=False)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        prompt_len = inputs["input_ids"].shape[-1]
        t0 = time.perf_counter()
        with torch.no_grad():
            out = self._model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                temperature=None,
                top_p=None,
                top_k=None,
                pad_token_id=self._tokenizer.eos_token_id,
            )
        elapsed = time.perf_counter() - t0
        tokens_out = out.shape[-1] - prompt_len
        print(f"[critic] generated {tokens_out} tokens in {elapsed:.1f}s  ({tokens_out / max(elapsed, 0.01):.0f} tok/s)", flush=True)
        return self._tokenizer.decode(out[0][prompt_len:], skip_special_tokens=True).strip()

    def _parse_result(self, raw: str) -> CriticResult:
        obj = _parse_json_object(raw)
        if not isinstance(obj, dict):
            return CriticResult(
                verdict="accept",
                confidence=0.0,
                raw_output=raw,
                parse_ok=False,
                validation_errors=["json_parse_failed"],
            )
        if "verdict" not in obj or "checks" not in obj:
            return CriticResult(
                verdict="accept",
                confidence=0.0,
                raw_output=raw,
                parse_ok=False,
                validation_errors=["critic_schema_missing_required_fields"],
            )
        verdict = str(obj.get("verdict", "accept")).strip().lower()
        if verdict not in {"accept", "revise", "reject"}:
            verdict = "accept"
        mistakes: List[CriticMistake] = []
        for item in obj.get("mistakes", []) or []:
            if not isinstance(item, Mapping):
                continue
            affected = [str(x) for x in item.get("affected_ids", []) or [] if str(x)]
            mistakes.append(CriticMistake(
                type=str(item.get("type", "weak_reasoning")),
                evidence=str(item.get("evidence", "")),
                affected_ids=affected,
            ))
        fix = obj.get("suggested_fix") if isinstance(obj.get("suggested_fix"), Mapping) else {}
        action = str(fix.get("action", "same")).strip().lower()
        if action not in {"same", "no_op", "add_node", "update_node", "link_nodes", "create_bridge", "resolve_conflict", "summarize_cluster"}:
            action = "same"
        proposed = fix.get("proposed") if isinstance(fix.get("proposed"), Mapping) else {}
        checks_raw = obj.get("checks") if isinstance(obj.get("checks"), Mapping) else {}
        checks = {str(k): bool(v) for k, v in checks_raw.items()}
        explanations_raw = obj.get("check_explanations") if isinstance(obj.get("check_explanations"), Mapping) else {}
        explanations = {str(k): str(v) for k, v in explanations_raw.items()}
        should_reject = bool(obj.get("should_reject", verdict == "reject"))
        try:
            confidence = float(obj.get("confidence", 0.0) or 0.0)
        except Exception:
            confidence = 0.0
        validation_errors: List[str] = []
        placeholder_text = json.dumps(obj, ensure_ascii=False).lower()
        if (
            "one short reason" in placeholder_text
            or "short evidence grounded" in placeholder_text
            or "duplicate | wrong_action" in placeholder_text
            or "same | no_op" in placeholder_text
        ):
            validation_errors.append("critic_schema_placeholder_output")
        return CriticResult(
            verdict=verdict,
            mistakes=mistakes,
            suggested_fix={"action": action, "proposed": dict(proposed)},
            checks=checks,
            check_explanations=explanations,
            should_reject=should_reject,
            confidence=max(0.0, min(1.0, confidence)),
            raw_output=raw,
            parse_ok=not bool(validation_errors),
            validation_errors=validation_errors,
        )


def _parse_json_object(raw: str) -> Optional[Dict[str, Any]]:
    text = str(raw or "").strip()
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.MULTILINE).strip()
    text = re.sub(r"```\s*$", "", text, flags=re.MULTILINE).strip()
    candidates: List[str] = [text]
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        candidates.append(match.group())
    for candidate in candidates:
        try:
            obj = json.loads(candidate)
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            continue
    return None


_STOP = {
    "the", "and", "for", "with", "that", "this", "from", "into", "using", "used",
    "uses", "can", "will", "should", "when", "where", "each", "than", "then",
    "data", "structure", "node", "graph", "signal", "action", "proposed",
}

_STRUCTURAL_TYPES = {"hub", "summary", "overview"}


def _tokens(text: Any) -> Set[str]:
    return {t for t in re.findall(r"[a-z0-9]+", str(text or "").lower()) if len(t) > 2 and t not in _STOP}


def _is_structural(graph: MemoryGraph, nid: str) -> bool:
    node = graph.nodes.get(nid)
    return bool(node and str(getattr(node, "node_type", "")).lower() in _STRUCTURAL_TYPES)


def _topic_flags(text: str, nid: str = "") -> Set[str]:
    s = f"{text} {nid}".lower().replace("_", " ")
    flags: Set[str] = set()
    if any(k in s for k in ("bloom", "hash", "set membership", "bit array", "false positive")):
        flags.add("hashing")
    if any(k in s for k in ("range query", "range sum", "prefix", "fenwick", "segment tree", "sparse table", "rmq")):
        flags.add("range_query")
    if any(k in s for k in ("bfs", "dijkstra", "shortest path", "graph algorithm", "edge weight")):
        flags.add("graph_algo")
    if any(k in s for k in ("neural network", "activation", "backpropagation", "gradient", "optimizer", "transformer")):
        flags.add("ml_ai")
    if any(k in s for k in ("database", "sql", "index", "b-tree", "transaction")):
        flags.add("database")
    if any(k in s for k in ("cell", "gene", "protein", "osmosis", "mutation")):
        flags.add("biology")
    if any(k in s for k in ("force", "temperature", "motion", "voltage", "thermodynamics")):
        flags.add("physics")
    return flags


def _edge_targets(action: str, proposed: Mapping[str, Any]) -> List[str]:
    if action == "add_node":
        return [str(e.get("dst", "")) for e in proposed.get("edges_to", []) or [] if isinstance(e, Mapping)]
    if action == "link_nodes":
        return [str(proposed.get("src", "")), str(proposed.get("dst", ""))]
    if action == "create_bridge":
        return [str(x) for x in proposed.get("connects", []) or []]
    if action in {"update_node", "resolve_conflict"}:
        return [str(proposed.get("target_id", ""))]
    if action == "summarize_cluster":
        return [str(x) for x in proposed.get("covers", []) or []]
    return []


def deterministic_graph_checks(
    *,
    signal_text: str,
    context_nodes: Sequence[Mapping[str, Any]],
    planner_action: str,
    planner_proposed: Mapping[str, Any],
    planner_validation: Mapping[str, Any],
    graph: MemoryGraph,
) -> tuple[Dict[str, bool], Dict[str, str]]:
    action = str(planner_action or "").lower()
    proposed = dict(planner_proposed or {})
    retrieved_ids = {str(r.get("id", "")) for r in context_nodes if str(r.get("id", ""))}
    targets = [x for x in _edge_targets(action, proposed) if x]
    target_set = set(targets)

    signal_tokens = _tokens(signal_text)
    new_text = str(proposed.get("text") or proposed.get("new_text") or proposed.get("resolution_text") or "")
    restatement_overlap = float(lexical_overlap(signal_text, new_text)) if new_text else 0.0
    max_duplicate = 0.0
    duplicate_id = ""
    if action == "add_node" and new_text:
        for nid, node in graph.nodes.items():
            ov = float(lexical_overlap(new_text, node.text))
            if ov > max_duplicate:
                max_duplicate = ov
                duplicate_id = nid

    uses_existing = all(t in graph.nodes and (not retrieved_ids or t in retrieved_ids or t in graph.nodes) for t in targets)
    signal_flags = _topic_flags(signal_text)
    edge_topic_ok = True
    cross_domain_ok = True
    bad_edges: List[str] = []
    for tid in targets:
        node = graph.nodes.get(tid)
        if not node:
            edge_topic_ok = False
            cross_domain_ok = False
            bad_edges.append(tid)
            continue
        overlap = float(lexical_overlap(signal_text, node.text))
        shared = signal_tokens & _tokens(f"{node.text} {tid}")
        target_flags = _topic_flags(node.text, tid)
        if not _is_structural(graph, tid) and overlap < 0.12 and not shared:
            edge_topic_ok = False
            bad_edges.append(tid)
        if signal_flags and target_flags and not (signal_flags & target_flags) and not shared:
            cross_domain_ok = False
            bad_edges.append(tid)

    action_ok = True
    if action == "no_op":
        action_ok = any(float(r.get("score", 0.0) or 0.0) >= 0.55 for r in context_nodes)
    elif action in {"update_node", "link_nodes", "resolve_conflict", "create_bridge", "summarize_cluster"}:
        action_ok = bool(targets)
    elif action == "add_node":
        action_ok = bool(new_text)

    checks = {
        "action_matches_signal": bool(action_ok),
        "new_node_is_not_duplicate": not (action == "add_node" and max_duplicate >= 0.86),
        "uses_existing_ids_only": bool(uses_existing),
        "edges_are_topic_compatible": bool(edge_topic_ok),
        "not_cross_domain": bool(cross_domain_ok),
        "not_signal_restatement": not (action == "add_node" and restatement_overlap >= 0.95),
    }
    explanations = {
        "action_matches_signal": f"action={action}; targets={targets or []}",
        "new_node_is_not_duplicate": f"max_duplicate_overlap={max_duplicate:.3f}; duplicate_id={duplicate_id}",
        "uses_existing_ids_only": f"targets={targets or []}; missing={[t for t in targets if t not in graph.nodes]}",
        "edges_are_topic_compatible": f"bad_edges={sorted(set(bad_edges))}",
        "not_cross_domain": f"signal_flags={sorted(signal_flags)}; bad_edges={sorted(set(bad_edges))}",
        "not_signal_restatement": f"signal_new_text_overlap={restatement_overlap:.3f}",
    }
    if isinstance(planner_validation, Mapping) and planner_validation.get("valid") is False:
        checks["uses_existing_ids_only"] = False
        explanations["uses_existing_ids_only"] += "; planner validation invalid"
    if action == "add_node" and not targets:
        checks["edges_are_topic_compatible"] = False
        explanations["edges_are_topic_compatible"] = "add_node has no explicit planner edges_to"
    return checks, explanations
