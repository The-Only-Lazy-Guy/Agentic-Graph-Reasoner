"""Benchmark llama-cpp-python inference configs for GLM-4.7-Flash GGUF.

This is intentionally a small harness: it measures the exact controller-like
request shape this repo uses, records failures, and writes JSONL for later
comparison. It does not try to hide model load time or parse llama.cpp perf
logs; use llama-bench/llama-server for deeper backend profiling.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

os.environ.setdefault("HF_HOME", str(Path.cwd() / "cache"))


GGML_F16 = 1
GGML_Q4_0 = 2
GGML_Q8_0 = 8


PRESETS: Dict[str, Dict[str, Any]] = {
    "baseline": {
        "flash_attn": False,
        "type_k": GGML_F16,
        "type_v": GGML_F16,
    },
    "flash": {
        "flash_attn": True,
        "type_k": GGML_F16,
        "type_v": GGML_F16,
    },
    "flash_op": {
        "flash_attn": True,
        "op_offload": True,
        "type_k": GGML_F16,
        "type_v": GGML_F16,
    },
    "batch1024": {
        "flash_attn": True,
        "n_batch": 1024,
        "n_ubatch": 512,
        "type_k": GGML_F16,
        "type_v": GGML_F16,
    },
    "ubatch256": {
        "flash_attn": True,
        "n_batch": 512,
        "n_ubatch": 256,
        "type_k": GGML_F16,
        "type_v": GGML_F16,
    },
    "kv_q8": {
        "flash_attn": True,
        "type_k": GGML_Q8_0,
        "type_v": GGML_Q8_0,
    },
    "kv_q4": {
        "flash_attn": True,
        "type_k": GGML_Q4_0,
        "type_v": GGML_Q4_0,
    },
    "nommap": {
        "use_mmap": False,
        "flash_attn": True,
        "type_k": GGML_F16,
        "type_v": GGML_F16,
    },
}


DEFAULT_MESSAGES = [
    {
        "role": "system",
        "content": (
            "You are a graph-reasoning controller. Output compact JSON with "
            "thought_summary, action, args, and reason."
        ),
    },
    {
        "role": "user",
        "content": (
            "Question: Light travels at a finite speed in vacuum.; Sound requires "
            "a medium to propagate.\n\n"
            "Session state: anchors imported, frontier has expandable science "
            "nodes, no final answer yet.\n\n"
            "Choose one valid action from EXPAND_NODE, PROPOSE_CLAIM, STOP."
        ),
    },
]


def _load_prompt_file(path: Path) -> List[Dict[str, str]]:
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".json":
        data = json.loads(text)
        if isinstance(data, list):
            return data
        if isinstance(data, dict) and isinstance(data.get("messages"), list):
            return data["messages"]
        raise ValueError("JSON prompt file must be a message list or {messages: [...]}")
    return [
        {
            "role": "system",
            "content": "You are a graph-reasoning controller. Output compact JSON.",
        },
        {"role": "user", "content": text},
    ]


def _build_graph_messages(graph_path: str, question: str, k_anchors: int) -> List[Dict[str, str]]:
    from answerer_v1 import SessionGraph, init_frontier_from_anchors, retrieve_anchors
    from answerer_v2 import serialize_session_state
    from graph_core import MemoryGraph

    graph = MemoryGraph.load_json(graph_path)
    session = SessionGraph(question)
    anchors = retrieve_anchors(question, graph, k=k_anchors)
    session.import_memory_nodes(anchors, graph)
    init_frontier_from_anchors(session, anchors)
    state = serialize_session_state(question, session, max_paths=5)
    return [
        {
            "role": "system",
            "content": "You are a graph-reasoning controller. Output compact JSON.",
        },
        {
            "role": "user",
            "content": (
                f"State:\n{state}\n\n"
                "Choose one action. Output JSON with thought_summary, action, args, reason."
            ),
        },
    ]


def build_messages(args: argparse.Namespace) -> List[Dict[str, str]]:
    if args.prompt_file:
        return _load_prompt_file(Path(args.prompt_file))
    if args.graph:
        return _build_graph_messages(args.graph, args.question, args.k_anchors)
    return DEFAULT_MESSAGES


def selected_presets(raw: str) -> Iterable[str]:
    names = list(PRESETS) if raw == "all" else [x.strip() for x in raw.split(",") if x.strip()]
    unknown = [name for name in names if name not in PRESETS]
    if unknown:
        raise SystemExit(f"Unknown preset(s): {', '.join(unknown)}. Available: {', '.join(PRESETS)}")
    return names


def run_config(
    name: str,
    overrides: Dict[str, Any],
    args: argparse.Namespace,
    messages: List[Dict[str, str]],
) -> Dict[str, Any]:
    from llama_cpp import Llama

    kwargs: Dict[str, Any] = {
        "model_path": args.model_path,
        "n_gpu_layers": args.n_gpu_layers,
        "n_ctx": args.n_ctx,
        "n_batch": args.n_batch,
        "n_ubatch": args.n_ubatch,
        "use_mmap": args.use_mmap,
        "use_mlock": args.use_mlock,
        "offload_kqv": args.offload_kqv,
        "verbose": args.verbose,
    }
    if args.n_threads is not None:
        kwargs["n_threads"] = args.n_threads
    if args.n_threads_batch is not None:
        kwargs["n_threads_batch"] = args.n_threads_batch
    kwargs.update(overrides)

    record: Dict[str, Any] = {
        "preset": name,
        "kwargs": kwargs,
        "runs": [],
    }

    llm: Optional[Llama] = None
    try:
        t0 = time.perf_counter()
        llm = Llama(**kwargs)
        record["load_seconds"] = time.perf_counter() - t0

        for i in range(args.warmup + args.runs):
            t0 = time.perf_counter()
            result = llm.create_chat_completion(
                messages=messages,
                response_format={"type": "json_object"},
                max_tokens=args.max_tokens,
                temperature=args.temperature,
            )
            elapsed = time.perf_counter() - t0
            usage = result.get("usage", {})
            completion_tokens = int(usage.get("completion_tokens") or 0)
            prompt_tokens = int(usage.get("prompt_tokens") or 0)
            run_record = {
                "warmup": i < args.warmup,
                "seconds": elapsed,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "completion_tok_s": completion_tokens / elapsed if elapsed > 0 else 0.0,
                "total_tok_s": (prompt_tokens + completion_tokens) / elapsed if elapsed > 0 else 0.0,
                "preview": result["choices"][0]["message"]["content"][:160],
            }
            record["runs"].append(run_record)
    except Exception as exc:
        record["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        if llm is not None:
            close = getattr(llm, "close", None)
            if callable(close):
                close()
        del llm
        gc.collect()
    return record


def summarize(record: Dict[str, Any]) -> str:
    if "error" in record:
        return f"{record['preset']}: ERROR {record['error']}"
    measured = [r for r in record["runs"] if not r["warmup"]]
    if not measured:
        return f"{record['preset']}: no measured runs"
    avg_sec = sum(r["seconds"] for r in measured) / len(measured)
    avg_decode = sum(r["completion_tok_s"] for r in measured) / len(measured)
    avg_total = sum(r["total_tok_s"] for r in measured) / len(measured)
    return (
        f"{record['preset']}: load={record.get('load_seconds', 0):.2f}s "
        f"avg={avg_sec:.2f}s decode={avg_decode:.2f} tok/s total={avg_total:.2f} tok/s"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark GLM GGUF llama-cpp-python configs")
    parser.add_argument("--model-path", default="cache/models/GLM-4.7-Flash-Q2_K.gguf")
    parser.add_argument("--presets", default="baseline,flash,flash_op,batch1024,kv_q8")
    parser.add_argument("--out", default="artifacts/glm_inference_bench.jsonl")
    parser.add_argument("--graph", default="", help="Optional graph JSON for controller-like prompt")
    parser.add_argument("--question", default=DEFAULT_MESSAGES[1]["content"])
    parser.add_argument("--k-anchors", type=int, default=8)
    parser.add_argument("--prompt-file", default="", help="Optional text or JSON messages prompt")
    parser.add_argument("--runs", type=int, default=2)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.3)
    parser.add_argument("--n-gpu-layers", type=int, default=20)
    parser.add_argument("--n-ctx", type=int, default=4096)
    parser.add_argument("--n-batch", type=int, default=512)
    parser.add_argument("--n-ubatch", type=int, default=512)
    parser.add_argument("--n-threads", type=int, default=None)
    parser.add_argument("--n-threads-batch", type=int, default=None)
    parser.add_argument("--no-mmap", dest="use_mmap", action="store_false", default=True)
    parser.add_argument("--mlock", dest="use_mlock", action="store_true", default=False)
    parser.add_argument("--no-offload-kqv", dest="offload_kqv", action="store_false", default=True)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    if not Path(args.model_path).exists():
        raise SystemExit(f"Model not found: {args.model_path}")

    messages = build_messages(args)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", encoding="utf-8") as f:
        for name in selected_presets(args.presets):
            record = run_config(name, PRESETS[name], args, messages)
            print(summarize(record), flush=True)
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
