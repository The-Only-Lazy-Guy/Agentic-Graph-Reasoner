"""
Answerer-v3: COT + tools paradigm.

Departure from v2's structured-action protocol. Instead of forcing the
LLM through PLAN/EXPAND/NOTE/CONCLUDE/FINALIZE rituals, the LLM reasons
in free-form chain-of-thought and embeds tool calls when it needs to
read or modify the graph. The harness parses the tool calls, executes
them, and feeds results back. The final answer is the model's prose
inside an <answer>...</answer> block.

The graph is USED DURING REASONING (the model reads/expands/searches it
to ground its COT), but the answer text is the model's natural prose.
This is the trade-off the user asked for: simpler protocol, more
natural reasoning, lose the literal "graph IS the answer" property.

Tools:
  read_node(node_id)               -> full text and type of a node
  expand_neighbors(node_id, k=5)   -> up to k connected nodes (id, rel, text snippet)
  search_nodes(query, k=5)         -> semantic search over the main graph
  hypothesize(text)                -> record a claim the graph doesn't contain
  list_anchors()                   -> the seed anchors retrieved at question time

Tool call syntax (model emits this in its assistant message):
  <tool>{"name": "read_node", "args": {"node_id": "cpp_kadane_template_apply"}}</tool>

Final answer syntax (model emits this when ready):
  <answer>... the answer text ...</answer>
"""
from __future__ import annotations

import json
import re
import time
import urllib.request
import urllib.error
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from graph_core import MemoryGraph
from anchor_retrieval import retrieve_anchors_v2


DEFAULT_LLAMA_SERVER_URL = "http://127.0.0.1:6767"


# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------

@dataclass
class V3Session:
    question: str
    anchors: List[str] = field(default_factory=list)
    hypotheses: Dict[str, str] = field(default_factory=dict)  # h_id -> text


# ---------------------------------------------------------------------------
# Tools — pure functions over (graph, session)
# ---------------------------------------------------------------------------

class V3Tools:
    """Tool dispatcher. Each method returns a JSON-serializable dict."""

    def __init__(self, graph: MemoryGraph, session: V3Session, *, snippet_chars: int = 220):
        self.graph = graph
        self.session = session
        self.snippet_chars = snippet_chars
        self.call_log: List[Dict[str, Any]] = []

    def _record(self, name: str, args: Dict[str, Any], result: Dict[str, Any]) -> None:
        self.call_log.append({"name": name, "args": args, "result_summary": _summarize(result)})

    def read_node(self, node_id: str) -> Dict[str, Any]:
        node = self.graph.nodes.get(node_id)
        if node is None and node_id in self.session.hypotheses:
            out = {"id": node_id, "text": self.session.hypotheses[node_id], "node_type": "hypothesis"}
            self._record("read_node", {"node_id": node_id}, out)
            return out
        if node is None:
            out = {"error": f"node {node_id!r} not found"}
            self._record("read_node", {"node_id": node_id}, out)
            return out
        out = {"id": node.id, "text": node.text, "node_type": node.node_type}
        self._record("read_node", {"node_id": node_id}, out)
        return out

    def expand_neighbors(self, node_id: str, k: int = 5) -> Dict[str, Any]:
        if node_id not in self.graph.nodes:
            out = {"error": f"node {node_id!r} not found"}
            self._record("expand_neighbors", {"node_id": node_id, "k": k}, out)
            return out
        neighbors: List[Dict[str, Any]] = []
        for e in self.graph.edges:
            other = None
            if e.src == node_id:
                other = e.dst
            elif e.dst == node_id and not e.directed:
                other = e.src
            if other is None:
                continue
            n = self.graph.nodes.get(other)
            if n is None:
                continue
            neighbors.append({
                "id": n.id,
                "relation": e.relation,
                "snippet": _truncate(n.text, self.snippet_chars),
            })
            if len(neighbors) >= k:
                break
        out = {"neighbors": neighbors}
        self._record("expand_neighbors", {"node_id": node_id, "k": k}, out)
        return out

    def search_nodes(self, query: str, k: int = 5) -> Dict[str, Any]:
        hits = retrieve_anchors_v2(query, self.graph, k=k, strategy="topk")
        out = {
            "hits": [
                {"id": h, "snippet": _truncate(self.graph.nodes[h].text, self.snippet_chars)}
                for h in hits if h in self.graph.nodes
            ]
        }
        self._record("search_nodes", {"query": query, "k": k}, out)
        return out

    def hypothesize(self, text: str) -> Dict[str, Any]:
        text = (text or "").strip()
        if not text:
            out = {"error": "hypothesize requires non-empty text"}
            self._record("hypothesize", {"text": text}, out)
            return out
        hid = f"h_{len(self.session.hypotheses) + 1}"
        self.session.hypotheses[hid] = text
        out = {"id": hid, "status": "recorded"}
        self._record("hypothesize", {"text": text}, out)
        return out

    def list_anchors(self) -> Dict[str, Any]:
        out = {
            "anchors": [
                {"id": a, "snippet": _truncate(self.graph.nodes[a].text, self.snippet_chars)}
                for a in self.session.anchors if a in self.graph.nodes
            ]
        }
        self._record("list_anchors", {}, out)
        return out


def _truncate(text: str, n: int) -> str:
    text = text or ""
    if len(text) <= n:
        return text
    return text[:n - 1] + "…"


def _summarize(result: Dict[str, Any]) -> str:
    """Compact summary of a tool result for the call log."""
    if "error" in result:
        return f"error: {result['error']}"
    keys = sorted(result.keys())
    parts = []
    for k in keys:
        v = result[k]
        if isinstance(v, list):
            parts.append(f"{k}=[{len(v)} items]")
        elif isinstance(v, str):
            parts.append(f"{k}={_truncate(v, 60)!r}")
        else:
            parts.append(f"{k}={v!r}")
    return "; ".join(parts)


TOOL_DISPATCH = {
    "read_node": lambda tools, args: tools.read_node(**args),
    "expand_neighbors": lambda tools, args: tools.expand_neighbors(**args),
    "search_nodes": lambda tools, args: tools.search_nodes(**args),
    "hypothesize": lambda tools, args: tools.hypothesize(**args),
    "list_anchors": lambda tools, args: tools.list_anchors(**args),
}


# ---------------------------------------------------------------------------
# Parsing tool calls and final-answer blocks from model output
# ---------------------------------------------------------------------------

_TOOL_BLOCK_RE = re.compile(r"<tool>\s*(\{.*?\})\s*</tool>", re.DOTALL)
_ANSWER_BLOCK_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)


def parse_tool_calls(text: str) -> List[Dict[str, Any]]:
    calls: List[Dict[str, Any]] = []
    for m in _TOOL_BLOCK_RE.finditer(text):
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
    m = _ANSWER_BLOCK_RE.search(text)
    if not m:
        return None
    return m.group(1).strip()


def execute_tool(tools: V3Tools, call: Dict[str, Any]) -> Dict[str, Any]:
    if "_parse_error" in call:
        return {"error": f"tool call JSON did not parse: {call.get('raw','')[:120]}"}
    if "_shape_error" in call:
        return {"error": f"tool call must have 'name' (string) and 'args' (object): {call.get('raw','')[:120]}"}
    name = call["name"]
    args = call["args"]
    fn = TOOL_DISPATCH.get(name)
    if fn is None:
        return {"error": f"unknown tool {name!r}. Available: {list(TOOL_DISPATCH)}"}
    try:
        return fn(tools, args)
    except TypeError as e:
        return {"error": f"bad args for {name}: {e}"}
    except Exception as e:
        return {"error": f"{type(e).__name__} in {name}: {e}"}


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

V3_SYSTEM_PROMPT = """You are a reasoning assistant with access to a knowledge graph.

You can think freely. When you need information from the graph, call a tool. Tool results come back as user messages and you can continue reasoning or call more tools. When you are ready to answer the user, write the answer inside <answer>...</answer>.

# Tools

- read_node(node_id) — get the full text and type of a node by id.
- expand_neighbors(node_id, k=5) — list up to k nodes connected to node_id, with the relation and a text snippet.
- search_nodes(query, k=5) — semantic search the graph for nodes matching a free-text query.
- hypothesize(text) — record a claim the graph does not contain. Use this only when search_nodes and expand_neighbors do not surface the fact you need.
- list_anchors() — re-list the initial anchors that were retrieved for this question.

# Tool call syntax

Emit a tool call as a single tag on its own line:

  <tool>{"name": "read_node", "args": {"node_id": "cpp_kadane_template_apply"}}</tool>

You can emit MULTIPLE tool calls in one response. After your response, every tool call you emitted will be executed and the results will come back as the NEXT user message. Then you can continue.

# When to stop

When you have read enough of the graph to answer the user's question, emit your final answer in:

  <answer>... your full answer here, with all required content, code, math, and reasoning ...</answer>

The user only sees what is inside the <answer> tags. Your reasoning before that is for your own bookkeeping.

# How to use the graph well

- The graph is your authoritative source. If it has the implementation, copy the code into <answer> verbatim — do not paraphrase code.
- If a node id ends with `_apply`, it contains canonical implementation code. Always read it before showing implementation.
- If a node id ends with `_false`, it is a misconception node — the text states the WRONG view. Do not cite it as truth; instead, read its contradict pair and refute the misconception in your answer.
- If you cannot find a needed fact via search_nodes or expand_neighbors, hypothesize it explicitly so the user knows it is not graph-grounded.

# Starting state

You are given the initial anchors for this question below. Read the ones that look relevant first. You do NOT need to read every anchor.

"""


def _build_first_user_message(question: str, anchors: List[str], graph: MemoryGraph) -> str:
    lines = ["Anchors retrieved for your question:"]
    for a in anchors:
        n = graph.nodes.get(a)
        if not n:
            continue
        lines.append(f"  - {a}  [{n.node_type}]  {_truncate(n.text, 140)}")
    lines.append("")
    lines.append(f"Question: {question}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Controller — wraps llama-server chat API
# ---------------------------------------------------------------------------

@dataclass
class V3ControllerConfig:
    base_url: str = DEFAULT_LLAMA_SERVER_URL
    temperature: float = 0.3
    max_tokens: int = 1024
    timeout: float = 360.0
    enable_thinking: bool = False


class V3LlamaServerController:
    def __init__(self, config: Optional[V3ControllerConfig] = None) -> None:
        self.config = config or V3ControllerConfig()
        self._checked = False
        self._guard_localhost(self.config.base_url)

    @staticmethod
    def _guard_localhost(url: str) -> None:
        from urllib.parse import urlparse
        host = (urlparse(url).hostname or "").lower()
        if host not in ("127.0.0.1", "localhost", "::1"):
            raise ValueError(f"V3LlamaServerController refuses non-localhost URL: {url!r}")

    def _ensure_server_reachable(self) -> None:
        if self._checked:
            return
        url = self.config.base_url.rstrip("/") + "/health"
        try:
            with urllib.request.urlopen(url, timeout=5) as r:
                _ = r.read(64)
            self._checked = True
        except (urllib.error.URLError, OSError) as e:
            raise RuntimeError(f"llama-server unreachable at {self.config.base_url!r}: {e}")

    def chat(self, messages: List[Dict[str, str]]) -> Dict[str, Any]:
        self._ensure_server_reachable()
        payload = json.dumps({
            "messages": messages,
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
            "chat_template_kwargs": {"enable_thinking": self.config.enable_thinking},
            "reasoning_effort": "none",
            "cache_prompt": True,
        }).encode("utf-8")
        req = urllib.request.Request(
            self.config.base_url.rstrip("/") + "/v1/chat/completions",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=self.config.timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

@dataclass
class V3Packet:
    question: str
    answer: str
    iterations: int
    tool_call_count: int
    tool_log: List[Dict[str, Any]]
    cot_log: List[str]            # one entry per assistant response
    elapsed_sec: float
    finalized: bool                # True if the model emitted <answer>...</answer>
    anchors: List[str]


def answer_query_v3(
    *,
    question: str,
    graph: MemoryGraph,
    controller: Optional[V3LlamaServerController] = None,
    max_iterations: int = 8,
    k_anchors: int = 8,
) -> V3Packet:
    controller = controller or V3LlamaServerController()

    t0 = time.time()
    anchors = retrieve_anchors_v2(question, graph, k=k_anchors, strategy="topk")
    session = V3Session(question=question, anchors=list(anchors))
    tools = V3Tools(graph, session)

    messages: List[Dict[str, str]] = [
        {"role": "system", "content": V3_SYSTEM_PROMPT},
        {"role": "user", "content": _build_first_user_message(question, anchors, graph)},
    ]
    cot_log: List[str] = []

    finalized = False
    final_answer: Optional[str] = None
    iterations = 0

    for it in range(max_iterations):
        iterations = it + 1
        response = controller.chat(messages)
        try:
            content = response["choices"][0]["message"]["content"]
        except (KeyError, IndexError) as e:
            raise RuntimeError(f"unexpected llama-server response shape: {e}; got {response!r}")
        cot_log.append(content)
        messages.append({"role": "assistant", "content": content})

        ans = parse_answer(content)
        if ans is not None:
            final_answer = ans
            finalized = True
            break

        tool_calls = parse_tool_calls(content)
        if not tool_calls:
            # Model produced text but neither tool calls nor a final answer.
            # Nudge it to do one or the other; cap retries via max_iterations.
            messages.append({
                "role": "user",
                "content": (
                    "Your previous response had no tool calls and no <answer>...</answer>. "
                    "Either call a tool with <tool>{\"name\":..., \"args\":...}</tool> "
                    "or emit your final answer inside <answer>...</answer>."
                ),
            })
            continue

        # Execute each tool call and feed results back as a single user message
        result_lines = []
        for call in tool_calls:
            result = execute_tool(tools, call)
            label = call.get("name", "<bad_call>")
            result_lines.append(f"[tool {label}] result:\n{json.dumps(result, ensure_ascii=False, indent=2)}")
        messages.append({"role": "user", "content": "\n\n".join(result_lines)})

    elapsed = time.time() - t0

    if final_answer is None:
        # Did not finalize within max_iterations.
        final_answer = "(no final answer produced; max_iterations reached)"

    return V3Packet(
        question=question,
        answer=final_answer,
        iterations=iterations,
        tool_call_count=len(tools.call_log),
        tool_log=tools.call_log,
        cot_log=cot_log,
        elapsed_sec=round(elapsed, 1),
        finalized=finalized,
        anchors=list(anchors),
    )
