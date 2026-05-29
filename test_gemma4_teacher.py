#!/usr/bin/env python3
"""
test_gemma4_teacher.py  —  Gate 11: Gemma 4 capability test

Loads google/gemma-4-31B-it (or any HF model) locally and runs the full
2-stage planner protocol across 5 test cases.

VRAM note: 31B at 4-bit ≈ 15.5 GB. RTX 4050 has 6 GB VRAM.
device_map="auto" is always used — layers spill to CPU RAM automatically.
Generation will be slow (~1–3 min/call) but correct for batch teacher work.

Capability gate (need >= 4/5 per criterion to PASS as teacher):
  1. JSON schema compliance       (action/reasoning/confidence/used_tool_result_ids/proposed)
  2. used_tool_result_ids non-empty  (model cites retrieved node IDs)
  3. Reasoning quality            (>= 10 tokens, not blank)
  4. No hallucinated node IDs     (all cited IDs present in provided context)
  5. Correct action type          (add_node / no_op / resolve_conflict as expected)

VERDICT:
  TEACHER CAPABLE   → generate 200-500 synthetic gold trajectories
  TEACHER FAILED    → promote Gemma 4 to STUDENT (replace Qwen 2.5-1.5B, run SFT/DPO/RL)

Usage:
  python test_gemma4_teacher.py
  python test_gemma4_teacher.py --hf-model google/gemma-4-31B-it --cache-dir ./cache
  python test_gemma4_teacher.py --no-4bit          # fp16, needs more VRAM
  python test_gemma4_teacher.py --backend api       # LM Studio fallback (localhost:1234)

  # Direct GGUF inference; no LM Studio server, no external API.
  python test_gemma4_teacher.py --backend llama --llama-model-path ./models/gemma4-IQ2_M.gguf
  python test_gemma4_teacher.py --backend llama --llama-repo-id bartowski/google_gemma-4-26B-A4B-it-GGUF --llama-filename '*IQ2_M.gguf'
"""

import argparse
import json
import sys
import time
import urllib.request
from pathlib import Path
from typing import Any, Optional

# ── Planner prompts (mirror planner.py exactly) ──────────────────────────────

_TOOL_PLAN_SYSTEM_GRAPH_WALK = """\
You are choosing one graph-walk tool before a graph edit.
Reply with exactly one short line and nothing else.

Allowed replies:
none
route_regions
expand_frontier | EXACT_NODE_ID | relation1,relation2 | depth
inspect_path | SRC_ID | DST_ID
find_conflicts | EXACT_NODE_ID[,EXACT_NODE_ID]

Rules:
- FIRST CALL: always reply route_regions. Do NOT reply none.
- SECOND CALL: use expand_frontier, inspect_path, or find_conflicts to explore the regions found.
- THIRD CALL: Only reply none if you have enough evidence to choose a final graph action.
- find_conflicts is for contradiction or misconception signals.
- expand_frontier is for exploring a local area around a specific retrieved node.
- inspect_path is for checking whether two nodes are meaningfully connected."""

_TOOL_PLAN_USER_GRAPH_WALK = """\
## Current graph nodes
{context}

## Signal
{signal}

Think as if you only know the graph shown here. Choose one graph-walk tool line."""

_EXECUTOR_SYSTEM = """\
You are a JSON generator for a knowledge graph editor.
Output ONLY a single valid JSON object — no markdown, no explanation, no text outside JSON.
Use EXACT node IDs from the context. Follow the action schema exactly."""

_EXECUTOR_USER = """\
## Graph nodes (use EXACT IDs shown)
{context}

## Signal (this is the NEW information — use it as new_text when updating)
{signal}

## Reasoning (use this to decide the action and which node IDs to reference)
{think}

Action schemas — follow EXACTLY:
  add_node:         {{"id":"<snake_id>","text":"<text from signal>","node_type":"<type>","edges_to":[{{"dst":"<EXACT_ID>","relation":"<rel>"}}]}}
  update_node:      {{"target_id":"<EXACT_ID>","new_text":"<text from signal>"}}
    WRONG: {{"some_node_id":"text"}}  ← do NOT use the node ID as a key
  link_nodes:       {{"src":"<EXACT_ID>","dst":"<DIFFERENT_EXACT_ID>","relation":"<rel>"}}
    NOTE: src and dst MUST be different node IDs
  create_bridge:    {{"id":"<NEW_id>","text":"<sentence>","connects":["<EXACT_ID>","<EXACT_ID>"]}}
  resolve_conflict: {{"target_id":"<EXACT_ID>","resolution_text":"<sentence>"}}
  summarize_cluster:{{"summary_text":"<sentence>","covers":["<EXACT_ID>","<EXACT_ID>",...]}}
  no_op:            {{"reason":"<why>"}}

Output: {{"action":"<action>","reasoning":"<one sentence>","confidence":<0.0-1.0>,"used_tool_result_ids":["<EXACT_ID>"],"proposed":{{...}}}}"""

# ── Test cases ────────────────────────────────────────────────────────────────

TEST_CASES = [
    {
        "id": "TC1_add_node_lazy_propagation",
        "desc": "add_node — new concept; must cite retrieved IDs in used_tool_result_ids",
        "signal": (
            "Lazy propagation defers pending range updates in a segment tree; "
            "each node stores a lazy tag that is pushed down before any child is accessed."
        ),
        "context": {
            "segment_tree_range_query": (
                "Segment trees support range queries in O(log n) by storing "
                "aggregated values at each internal node."
            ),
            "segment_tree_build_on_array": (
                "A segment tree can be built over an array in O(n) by constructing "
                "leaf nodes first then propagating values upward."
            ),
            "fenwick_update_prefix_query": (
                "Fenwick tree updates and prefix queries both run in O(log n) "
                "using bit manipulation on indices."
            ),
        },
        "sim_tool_call": "expand_frontier | segment_tree_range_query | related,part_of | 2",
        "sim_tool_result_ids": ["segment_tree_range_query", "segment_tree_build_on_array"],
        "sim_tool_text": (
            "Expanded from segment_tree_range_query (depth=2):\n"
            "- segment_tree_range_query: Segment trees support range queries in O(log n).\n"
            "- segment_tree_build_on_array: A segment tree can be built over an array in O(n)."
        ),
        "expected_action": "add_node",
    },
    {
        "id": "TC2_no_op_fenwick_covered",
        "desc": "no_op — signal fully covered; must cite covering node IDs",
        "signal": "Fenwick tree updates and prefix queries both run in O(log n).",
        "context": {
            "fenwick_update_prefix_query": (
                "Fenwick tree updates and prefix queries both run in O(log n) "
                "using bit manipulation on indices."
            ),
            "fenwick_binary_indexed_tree": (
                "A Fenwick tree (Binary Indexed Tree) stores cumulative frequencies "
                "for efficient prefix sum computation."
            ),
            "segment_tree_range_query": (
                "Segment trees support range queries in O(log n) by storing "
                "aggregated values at each internal node."
            ),
        },
        "sim_tool_call": "expand_frontier | fenwick_update_prefix_query | related,part_of | 1",
        "sim_tool_result_ids": ["fenwick_update_prefix_query", "fenwick_binary_indexed_tree"],
        "sim_tool_text": (
            "Expanded from fenwick_update_prefix_query (depth=1):\n"
            "- fenwick_update_prefix_query: Fenwick tree updates and prefix queries O(log n).\n"
            "- fenwick_binary_indexed_tree: BIT stores cumulative frequencies for prefix sums."
        ),
        "expected_action": "no_op",
    },
    {
        "id": "TC3_add_node_gospers_hack",
        "desc": "add_node — novel concept; edges_to must only reference real node IDs",
        "signal": (
            "Gosper's Hack computes the next integer with the same popcount, "
            "enabling O(2^n) enumeration of all k-bit submasks in competitive programming."
        ),
        "context": {
            "bit_manipulation_isolate_lowest": (
                "The expression x & -x isolates the lowest set bit "
                "and drives Fenwick tree index arithmetic."
            ),
            "sos_dp_sum_over_subsets": (
                "SOS (Sum over Subsets) DP fills a table over all subsets in "
                "O(n·2^n) by processing one bit dimension at a time."
            ),
            "union_find_amortized_alpha": (
                "Union-Find with path compression and union by rank achieves "
                "amortized O(α(n)) per operation."
            ),
        },
        "sim_tool_call": "expand_frontier | bit_manipulation_isolate_lowest | related | 1",
        "sim_tool_result_ids": ["bit_manipulation_isolate_lowest", "sos_dp_sum_over_subsets"],
        "sim_tool_text": (
            "Expanded from bit_manipulation_isolate_lowest (depth=1):\n"
            "- bit_manipulation_isolate_lowest: x & -x isolates the lowest set bit.\n"
            "- sos_dp_sum_over_subsets: SOS DP O(n·2^n) for all subsets."
        ),
        "expected_action": "add_node",
    },
    {
        "id": "TC4_resolve_conflict_bfs_false",
        "desc": "resolve_conflict — false-claim node surfaced; target_id must be the false node",
        "signal": (
            "BFS does NOT correctly compute shortest paths on weighted graphs "
            "when edge weights are unequal — it treats all edges as unit-weight."
        ),
        "context": {
            "bfs_on_weighted_graph_false": (
                "[FALSE CLAIM] BFS correctly computes shortest paths on weighted graphs."
            ),
            "dijkstra_nonneg_edge_weights": (
                "Dijkstra's algorithm correctly computes shortest paths for graphs "
                "with non-negative edge weights using a priority queue."
            ),
            "bfs_unweighted_shortest_path": (
                "BFS finds shortest paths in unweighted graphs by exploring level by level."
            ),
        },
        "sim_tool_call": "find_conflicts | bfs_on_weighted_graph_false",
        "sim_tool_result_ids": ["bfs_on_weighted_graph_false"],
        "sim_tool_text": (
            "Conflicts around bfs_on_weighted_graph_false:\n"
            "- bfs_on_weighted_graph_false [FALSE CLAIM]: BFS correctly computes shortest paths "
            "on weighted graphs. Contradicts dijkstra_nonneg_edge_weights and "
            "bfs_unweighted_shortest_path (BFS is for unweighted graphs only)."
        ),
        "expected_action": "resolve_conflict",
    },
    {
        "id": "TC5_no_op_union_find_covered",
        "desc": "no_op — strong multi-node coverage; reasoning must be specific and non-blank",
        "signal": (
            "Union-Find with path compression and union by rank achieves "
            "amortized O(α(n)) per operation, where α is the inverse Ackermann function."
        ),
        "context": {
            "union_find_path_compression": (
                "Union-Find path compression flattens the tree during find operations, "
                "giving near-O(1) subsequent finds."
            ),
            "union_find_union_by_rank": (
                "Union by rank attaches the shorter tree under the taller, "
                "preventing O(n) height in Union-Find."
            ),
            "union_find_amortized_alpha": (
                "Union-Find with path compression and union by rank achieves "
                "amortized O(α(n)) per operation."
            ),
        },
        "sim_tool_call": "expand_frontier | union_find_amortized_alpha | related,part_of | 1",
        "sim_tool_result_ids": [
            "union_find_path_compression",
            "union_find_union_by_rank",
            "union_find_amortized_alpha",
        ],
        "sim_tool_text": (
            "Expanded from union_find_amortized_alpha (depth=1):\n"
            "- union_find_amortized_alpha: O(α(n)) per operation via path compression + union by rank.\n"
            "- union_find_path_compression: path compression flattens tree during find.\n"
            "- union_find_union_by_rank: union by rank prevents tall trees."
        ),
        "expected_action": "no_op",
    },
]

REQUIRED_KEYS = {"action", "reasoning", "confidence", "used_tool_result_ids", "proposed"}


# ── Backend: local HuggingFace ────────────────────────────────────────────────

def load_hf_model(model_name: str, cache_dir: str, quantize_4bit: bool):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"[model] loading {model_name!r}  4bit={quantize_4bit}  device_map=auto", flush=True)
    if torch.cuda.is_available():
        vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"[model] GPU: {torch.cuda.get_device_name(0)}  VRAM: {vram:.1f} GB", flush=True)
        if vram < 16 and not quantize_4bit:
            print("[model] WARNING: fp16 for a 31B model needs ~62 GB VRAM. "
                  "Re-run with --quantize-4bit (default) or expect CPU-only.", flush=True)
    else:
        print("[model] WARNING: No CUDA. Running entirely on CPU — expect very slow inference.", flush=True)

    t0 = time.perf_counter()
    load_kw: dict = {
        "trust_remote_code": True,
        "cache_dir":         cache_dir,
        "device_map":        "auto",   # spread across GPU + CPU RAM automatically
    }

    if quantize_4bit and torch.cuda.is_available():
        from transformers import BitsAndBytesConfig
        load_kw["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
    else:
        load_kw["torch_dtype"] = torch.float16 if torch.cuda.is_available() else torch.float32

    model = AutoModelForCausalLM.from_pretrained(model_name, **load_kw)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(
        model_name, trust_remote_code=True, cache_dir=cache_dir
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    print(f"[model] ready in {time.perf_counter() - t0:.1f}s", flush=True)
    return model, tokenizer


def generate_local(model, tokenizer, messages: list, max_new_tokens: int, temperature: float) -> str:
    import torch

    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    # put inputs on the same device as the model's embeddings
    first_device = next(iter(model.hf_device_map.values())) if hasattr(model, "hf_device_map") else (
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    inputs = tokenizer(text, return_tensors="pt").to(first_device)

    gen_kw: dict = {"max_new_tokens": max_new_tokens, "pad_token_id": tokenizer.eos_token_id}
    if temperature > 0:
        gen_kw["do_sample"] = True
        gen_kw["temperature"] = temperature
    else:
        gen_kw["do_sample"] = False

    with torch.no_grad():
        output = model.generate(**inputs, **gen_kw)

    new_tokens = output[0][inputs.input_ids.shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


# ── Backend: direct llama.cpp / GGUF ──────────────────────────────────────────

def _messages_to_plain_prompt(messages: list) -> str:
    """Fallback prompt formatter if the GGUF chat template is unavailable."""
    parts = []
    for m in messages:
        role = (m.get("role") or "user").strip().lower()
        content = (m.get("content") or "").strip()
        if role == "system":
            parts.append(f"System:\n{content}")
        elif role == "assistant":
            parts.append(f"Assistant:\n{content}")
        else:
            parts.append(f"User:\n{content}")
    parts.append("Assistant:")
    return "\n\n".join(parts)


def load_llama_model(
    model_path: Optional[str],
    repo_id: Optional[str],
    filename: Optional[str],
    n_ctx: int,
    n_gpu_layers: int,
    n_threads: Optional[int],
    verbose: bool,
):
    """
    Load a quantized GGUF model directly through llama-cpp-python.

    Notes:
    - This does NOT use LM Studio or a localhost API server.
    - Quantization comes from the GGUF file itself, e.g. IQ2_M, Q4_K_M, Q5_K_M.
    - On a 6 GB RTX 4050, start with IQ2_M and low n_gpu_layers, then increase slowly.
    """
    try:
        from llama_cpp import Llama
    except ImportError as exc:
        raise RuntimeError(
            "Missing llama-cpp-python. Install it first:\n"
            "  pip install llama-cpp-python\n\n"
            "For CUDA acceleration on Windows, you may need a CUDA-enabled build."
        ) from exc

    kwargs: dict = {
        "n_ctx": n_ctx,
        "n_gpu_layers": n_gpu_layers,
        "verbose": verbose,
    }
    if n_threads is not None and n_threads > 0:
        kwargs["n_threads"] = n_threads

    t0 = time.perf_counter()

    if model_path:
        p = Path(model_path)
        if not p.exists():
            raise FileNotFoundError(f"GGUF model file not found: {p}")
        print(
            f"[llama] loading local GGUF: {p}\n"
            f"[llama] n_ctx={n_ctx}  n_gpu_layers={n_gpu_layers}  n_threads={n_threads or 'auto'}",
            flush=True,
        )
        llm = Llama(model_path=str(p), **kwargs)
    else:
        if not repo_id or not filename:
            raise ValueError(
                "For --backend llama, provide either --llama-model-path OR both "
                "--llama-repo-id and --llama-filename."
            )
        print(
            f"[llama] downloading/loading GGUF from HF repo={repo_id!r} file={filename!r}\n"
            f"[llama] n_ctx={n_ctx}  n_gpu_layers={n_gpu_layers}  n_threads={n_threads or 'auto'}",
            flush=True,
        )
        llm = Llama.from_pretrained(
            repo_id=repo_id,
            filename=filename,
            **kwargs,
        )

    print(f"[llama] ready in {time.perf_counter() - t0:.1f}s", flush=True)
    return llm


def generate_llama(llm, messages: list, max_new_tokens: int, temperature: float) -> str:
    """Generate with llama-cpp-python using chat completion first, then fallback to raw completion."""
    temp = max(float(temperature), 0.0)

    try:
        resp = llm.create_chat_completion(
            messages=messages,
            temperature=temp,
            max_tokens=max_new_tokens,
        )
        msg = resp["choices"][0].get("message", {}).get("content", "")
        if msg and msg.strip():
            return msg.strip()
    except Exception as chat_exc:
        print(f"[llama] chat completion failed; falling back to plain prompt: {chat_exc}", flush=True)

    prompt = _messages_to_plain_prompt(messages)
    resp = llm(
        prompt,
        temperature=temp,
        max_tokens=max_new_tokens,
        stop=["User:", "System:"],
    )
    return resp["choices"][0].get("text", "").strip()


# ── Backend: LM Studio API ────────────────────────────────────────────────────

def api_post(base_url: str, model: str, messages: list, temperature: float, max_tokens: int) -> str:
    payload = json.dumps({
        "model": model, "messages": messages,
        "temperature": temperature, "max_tokens": max_tokens,
    }).encode()
    req = urllib.request.Request(
        f"{base_url}/v1/chat/completions", data=payload,
        headers={"Content-Type": "application/json"}, method="POST",
    )
    with urllib.request.urlopen(req, timeout=300) as resp:
        return json.loads(resp.read())["choices"][0]["message"]["content"].strip()


def get_api_models(base_url: str) -> list:
    req = urllib.request.Request(f"{base_url}/v1/models", method="GET")
    with urllib.request.urlopen(req, timeout=10) as resp:
        return [m["id"] for m in json.loads(resp.read()).get("data", [])]


# ── Context formatter ─────────────────────────────────────────────────────────

def fmt_context(nodes: dict) -> str:
    return "\n".join(f"[{nid}] {text}" for nid, text in nodes.items())


# ── Checks ────────────────────────────────────────────────────────────────────

def check_schema(p: dict) -> tuple:
    missing = REQUIRED_KEYS - set(p.keys())
    if missing:
        return False, f"missing keys: {sorted(missing)}"
    if not isinstance(p.get("used_tool_result_ids"), list):
        return False, "used_tool_result_ids is not a list"
    return True, "ok"


def check_no_hallucinated_ids(p: dict, allowed: set) -> tuple:
    bad = [i for i in p.get("used_tool_result_ids", []) if i not in allowed]
    proposed = p.get("proposed", {})
    edge_ids = [e.get("dst", "") for e in proposed.get("edges_to", [])]
    for field in ("target_id", "src", "dst"):
        if field in proposed:
            edge_ids.append(proposed[field])
    bad_edges = [i for i in edge_ids if i and i not in allowed]
    all_bad = list(set(bad + bad_edges))
    if all_bad:
        return False, f"hallucinated IDs: {all_bad}"
    return True, "ok"


def check_reasoning(p: dict) -> tuple:
    r = (p.get("reasoning") or "").strip()
    if not r:
        return False, "blank"
    tokens = r.split()
    if len(tokens) < 10:
        return False, f"too short ({len(tokens)} tokens): {r!r}"
    return True, f"{len(tokens)} tokens"


def check_ids_nonempty(p: dict) -> tuple:
    ids = p.get("used_tool_result_ids", [])
    return (bool(ids), str(ids) if ids else "empty — model did not cite any tool results")


def check_action(p: dict, expected: str) -> tuple:
    got = p.get("action", "")
    return got == expected, f"got={got!r}  expected={expected!r}"


# ── Per-case runner ───────────────────────────────────────────────────────────

def run_case(case: dict, call_fn, temperature: float) -> dict:
    r: dict = {
        "id": case["id"], "desc": case["desc"],
        "stage1_raw": None, "stage2_raw": None,
        "parsed": None, "checks": {}, "errors": [], "pass": False,
    }

    context_str = fmt_context(case["context"])
    all_ids = set(case["context"].keys()) | set(case["sim_tool_result_ids"])

    # Stage 1: tool choice
    s1_msgs = [
        {"role": "system", "content": _TOOL_PLAN_SYSTEM_GRAPH_WALK},
        {"role": "user",   "content": _TOOL_PLAN_USER_GRAPH_WALK.format(
            context=context_str, signal=case["signal"])},
    ]
    try:
        r["stage1_raw"] = call_fn(s1_msgs, max_tokens=48)
    except Exception as e:
        r["errors"].append(f"stage1: {e}")
        return r

    # Simulate tool execution
    think = f"Tool call: {case['sim_tool_call']}\n\nTool results:\n{case['sim_tool_text']}"

    # Stage 2: action JSON
    s2_msgs = [
        {"role": "system", "content": _EXECUTOR_SYSTEM},
        {"role": "user",   "content": _EXECUTOR_USER.format(
            context=context_str, signal=case["signal"], think=think)},
    ]
    try:
        r["stage2_raw"] = call_fn(s2_msgs, max_tokens=512)
    except Exception as e:
        r["errors"].append(f"stage2: {e}")
        return r

    # Parse JSON — strip markdown fences, find first { ... }
    raw = (r["stage2_raw"] or "").strip()
    if raw.startswith("```"):
        raw = "\n".join(ln for ln in raw.splitlines() if not ln.startswith("```"))
    s, e_pos = raw.find("{"), raw.rfind("}") + 1
    if s >= 0 and e_pos > s:
        raw = raw[s:e_pos]
    try:
        parsed = json.loads(raw)
        r["parsed"] = parsed
        r["checks"]["json_parse"] = (True, "ok")
    except json.JSONDecodeError as exc:
        r["checks"]["json_parse"] = (False, str(exc))
        r["errors"].append(f"JSON parse: {exc}")
        return r

    r["checks"]["schema"]               = check_schema(parsed)
    r["checks"]["no_hallucinated_ids"]  = check_no_hallucinated_ids(parsed, all_ids)
    r["checks"]["used_tool_result_ids"] = check_ids_nonempty(parsed)
    r["checks"]["reasoning_quality"]    = check_reasoning(parsed)
    r["checks"]["action_type"]          = check_action(parsed, case["expected_action"])

    if case["expected_action"] == "resolve_conflict":
        target  = parsed.get("proposed", {}).get("target_id", "")
        correct = case["sim_tool_result_ids"][0]
        r["checks"]["resolve_target_id"] = (target == correct, f"target_id={target!r} want={correct!r}")

    critical = ["json_parse", "schema", "no_hallucinated_ids", "used_tool_result_ids", "reasoning_quality"]
    r["pass"] = all(r["checks"].get(k, (False,))[0] for k in critical)
    return r


# ── Pretty printer ────────────────────────────────────────────────────────────

def print_result(r: dict) -> None:
    bar = "=" * 72
    print(f"\n{bar}")
    print(f"[{'PASS' if r['pass'] else 'FAIL'}]  {r['id']}")
    print(f"         {r['desc']}")
    print(bar)
    if r["stage1_raw"] is not None:
        print(f"  Stage 1 tool reply : {r['stage1_raw']!r}")
    if r["stage2_raw"] is not None:
        print(f"  Stage 2 raw (220ch): {(r['stage2_raw'] or '')[:220].replace(chr(10),' ')}")
    if r["parsed"]:
        p = r["parsed"]
        print(f"  action             : {p.get('action')}")
        print(f"  reasoning          : {(p.get('reasoning') or '')[:120]!r}")
        print(f"  used_tool_ids      : {p.get('used_tool_result_ids')}")
        print(f"  confidence         : {p.get('confidence')}")
    print("  Checks:")
    for k, v in r["checks"].items():
        ok, msg = v if isinstance(v, tuple) else (v, "")
        print(f"    {'✓' if ok else '✗'}  {k:<28} {msg}")
    for e in r["errors"]:
        print(f"  !! ERROR: {e}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(description="Gate 11 — Gemma 4 teacher/student capability test")
    ap.add_argument("--backend",      default="local",        choices=["local", "api", "llama"],
                    help="local=HuggingFace transformers, api=LM Studio localhost, llama=direct GGUF via llama-cpp-python")
    ap.add_argument("--hf-model",     default="google/gemma-4-26B-A4B-it",
                    help="HuggingFace model ID (local backend only)")
    ap.add_argument("--cache-dir",    default="./cache",
                    help="HF cache directory (default: ./cache)")
    ap.add_argument("--quantize-4bit", default=True, action=argparse.BooleanOptionalAction,
                    help="4-bit QLoRA quantization (default: on; requires bitsandbytes + CUDA)")
    ap.add_argument("--temperature",  default=0.1,  type=float)
    ap.add_argument("--api-base-url", default="http://localhost:1234", help="LM Studio base URL")
    ap.add_argument("--api-model",    default=None, help="LM Studio model ID (auto-detected)")

    # Direct llama.cpp / GGUF backend. This avoids LM Studio and avoids HF BitsAndBytes loading.
    ap.add_argument("--llama-model-path", default=None,
                    help="Path to a local .gguf file. If set, this overrides --llama-repo-id/--llama-filename.")
    ap.add_argument("--llama-repo-id", default="bartowski/google_gemma-4-26B-A4B-it-GGUF",
                    help="HF repo containing GGUF files for llama backend.")
    ap.add_argument("--llama-filename", default="*IQ2_M.gguf",
                    help="GGUF filename or glob inside --llama-repo-id, e.g. '*IQ2_M.gguf' or '*Q4_K_M.gguf'.")
    ap.add_argument("--llama-n-ctx", default=4096, type=int,
                    help="Context length for llama backend. Lower it if RAM/VRAM is tight.")
    ap.add_argument("--llama-n-gpu-layers", default=10, type=int,
                    help="Number of GGUF layers to offload to GPU. Start low on 6 GB VRAM; try 0, 5, 10, 15.")
    ap.add_argument("--llama-n-threads", default=None, type=int,
                    help="CPU threads for llama backend. Default lets llama.cpp choose.")
    ap.add_argument("--llama-verbose", action="store_true",
                    help="Print verbose llama.cpp load/inference logs.")
    args = ap.parse_args()

    print("Gate 11 — Gemma 4 teacher/student capability test")

    # ── Build call_fn ─────────────────────────────────────────────────────────
    if args.backend == "local":
        print(f"Backend: HuggingFace local  model={args.hf_model}  4bit={args.quantize_4bit}")
        try:
            hf_model, hf_tokenizer = load_hf_model(args.hf_model, args.cache_dir, args.quantize_4bit)
        except OSError as e:
            if "not found" in str(e).lower() or "no such file" in str(e).lower():
                print(f"\nERROR: model not found locally. Download it first:", file=sys.stderr)
                print(f"  python -c \"from huggingface_hub import snapshot_download; "
                      f"snapshot_download('{args.hf_model}', cache_dir='{args.cache_dir}')\"",
                      file=sys.stderr)
                print("  (Requires: pip install huggingface_hub  and  HF_TOKEN set or huggingface-cli login)",
                      file=sys.stderr)
            raise

        def call_fn(messages: list, max_tokens: int) -> str:
            return generate_local(hf_model, hf_tokenizer, messages, max_tokens, args.temperature)

    elif args.backend == "api":
        print(f"Backend: LM Studio API  base={args.api_base_url}")
        api_model = args.api_model
        if not api_model:
            try:
                models = get_api_models(args.api_base_url)
            except Exception as e:
                print(f"\nERROR: Cannot reach LM Studio at {args.api_base_url}: {e}", file=sys.stderr)
                sys.exit(1)
            if not models:
                print("ERROR: No models loaded in LM Studio.", file=sys.stderr)
                sys.exit(1)
            api_model = models[0]
            print(f"API model (auto-detected): {api_model}")

        def call_fn(messages: list, max_tokens: int) -> str:
            return api_post(args.api_base_url, api_model, messages, args.temperature, max_tokens)

    elif args.backend == "llama":
        print("Backend: direct llama.cpp / GGUF via llama-cpp-python")
        print("        Quantization is from the GGUF file itself, e.g. IQ2_M/Q4_K_M/Q5_K_M.")
        llama_model = load_llama_model(
            model_path=args.llama_model_path,
            repo_id=args.llama_repo_id,
            filename=args.llama_filename,
            n_ctx=args.llama_n_ctx,
            n_gpu_layers=args.llama_n_gpu_layers,
            n_threads=args.llama_n_threads,
            verbose=args.llama_verbose,
        )

        def call_fn(messages: list, max_tokens: int) -> str:
            return generate_llama(llama_model, messages, max_tokens, args.temperature)

    else:
        raise ValueError(f"Unknown backend: {args.backend}")

    # ── Run all cases ─────────────────────────────────────────────────────────
    results = []
    for i, case in enumerate(TEST_CASES, 1):
        print(f"\nRunning {case['id']} ({i}/{len(TEST_CASES)})...", end=" ", flush=True)
        t0 = time.perf_counter()
        r = run_case(case, call_fn, args.temperature)
        elapsed = time.perf_counter() - t0
        print(f"{'PASS' if r['pass'] else 'FAIL'}  ({elapsed:.1f}s)")
        results.append(r)
        print_result(r)

    # ── Aggregate ─────────────────────────────────────────────────────────────
    n = len(results)
    gate = {
        "json_parse":           sum(1 for r in results if r["checks"].get("json_parse",           (False,))[0]),
        "schema_ok":            sum(1 for r in results if r["checks"].get("schema",               (False,))[0]),
        "used_tool_result_ids": sum(1 for r in results if r["checks"].get("used_tool_result_ids", (False,))[0]),
        "reasoning_quality":    sum(1 for r in results if r["checks"].get("reasoning_quality",    (False,))[0]),
        "no_hallucinated_ids":  sum(1 for r in results if r["checks"].get("no_hallucinated_ids",  (False,))[0]),
    }
    passed_cases = sum(1 for r in results if r["pass"])
    print(f"\n{'='*72}")
    print(f"SUMMARY: {passed_cases}/{n} cases fully passed")
    print(f"{'='*72}")
    print("\nCapability gate (need >= 4/5 per criterion):")
    hard_pass = True
    for criterion, count in gate.items():
        ok = count >= 4
        if not ok:
            hard_pass = False
        verdict = "PASS" if ok else ("WARN" if count >= 3 else "FAIL")
        print(f"  {verdict}  {criterion:<28} {count}/{n}")

    print(f"\n{'='*72}")
    if hard_pass:
        print("VERDICT: TEACHER CAPABLE")
        print("  Gemma 4 follows the planner protocol reliably.")
        print("  Next: generate 200-500 synthetic gold trajectories targeting:")
        print("    • used_tool_result_ids populated correctly")
        print("    • non-blank, specific reasoning")
        print("    • freshness-gap no_op handling")
        print("  Output: artifacts/training/gemma4_gold_sft.jsonl")
        print("  Then: SFT distill into Qwen 2.5-1.5B from gate10e DPO adapter.")
    else:
        print("VERDICT: TEACHER FAILED")
        print("  Gemma 4 does not reliably follow the planner protocol as a teacher.")
        print()
        print("  NEXT: promote Gemma 4 to STUDENT (replace Qwen 2.5-1.5B).")
        print("  Gemma 4 (4B active / 31B MoE) has far more capacity — run full")
        print("  SFT → DPO → RL pipeline directly on it:")
        print()
        print("  Step 1 — Add OpenAI-compatible backend to planner.py:")
        print("    • --planner-backend openai  +  --planner-api-base http://localhost:1234")
        print("    • OR: load via llama-cpp-python GGUF directly, same as --backend llama")
        print("    • OR: load a smaller model via transformers directly")
        print()
        print("  Step 2 — Run consistency_regret_loop.py with Gemma 4 as planner:")
        print(f"    python consistency_regret_loop.py \\")
        print(f"      --planner-model {args.hf_model} \\")
        print(f"      --planner-backend llm \\")
        print(f"      --reasoning-mode hybrid \\")
        print(f"      --signals artifacts/tmp/cs_math_crossref_plannerpos_30.jsonl \\")
        print(f"      --graph artifacts/p5/graph_gate10_crossref_p1.json \\")
        print(f"      ... (standard flags)")
        print()
        print("  Step 3 — SFT/DPO on Gemma 4 with the seed137 gold data +")
        print("           any new planner_commit rows collected above.")
    print(f"{'='*72}")

    sys.exit(0 if hard_pass else 1)


if __name__ == "__main__":
    main()
