from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
FRONTEND_DIR = ROOT.parent / "graph-front-end"
DEFAULT_OUT_DIR = ROOT / "artifacts" / "frontend_pipeline_debug"
DEFAULT_QUESTION = (
    "I have a directed graph with edges (a->b, weight 3), "
    "(b->c, weight -1), (a->c, weight 5). I'm planning to run Dijkstra. "
    "Verify each precondition of Dijkstra against this instance and tell me "
    "if it's safe to apply."
)


def _jsonable(value: Any) -> Any:
    try:
        json.dumps(value)
        return value
    except TypeError:
        if hasattr(value, "to_dict"):
            return value.to_dict()
        if hasattr(value, "__dict__"):
            return {k: _jsonable(v) for k, v in value.__dict__.items()}
        return str(value)


def _event(event: str, data: dict[str, Any], started: float) -> dict[str, Any]:
    return {
        "event": event,
        "t_rel_sec": round(time.perf_counter() - started, 3),
        "data": _jsonable(data),
    }


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[Any]:
    rows = []
    if not path.exists():
        return rows
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def _load_session_artifacts(final_payload: dict[str, Any]) -> dict[str, Any]:
    substrate = final_payload.get("substrate") or {}
    raw_path = substrate.get("session_subgraph_path")
    if not raw_path:
        return {}
    session_dir = Path(raw_path)
    out: dict[str, Any] = {"session_dir": str(session_dir)}
    subgraph_path = session_dir / "subgraph.json"
    audit_path = session_dir / "audit_log.jsonl"
    if subgraph_path.exists():
        out["subgraph_json"] = _read_json(subgraph_path)
    if audit_path.exists():
        out["audit_log_jsonl"] = _read_jsonl(audit_path)
    return out


def _summarize(final_payload: dict[str, Any] | None, events: list[dict[str, Any]]) -> dict[str, Any]:
    if not final_payload:
        return {
            "ok": False,
            "event_count": len(events),
            "error_events": [e for e in events if e.get("event") == "error"],
        }

    session = final_payload.get("session") or {}
    substrate = final_payload.get("substrate") or {}
    subgraph = substrate.get("session_subgraph") or {}
    nodes = subgraph.get("nodes") or session.get("nodes") or {}
    edges = subgraph.get("edges") or session.get("edges") or []
    node_types: dict[str, int] = {}
    for node in nodes.values():
        ntype = str(node.get("node_type", "unknown"))
        node_types[ntype] = node_types.get(ntype, 0) + 1

    trace = final_payload.get("trace") or []
    return {
        "ok": True,
        "run_id": final_payload.get("run_id"),
        "graph_id": final_payload.get("graph_id"),
        "graph_file": final_payload.get("graph_file"),
        "answer_chars": len(final_payload.get("answer") or ""),
        "steps_taken": final_payload.get("steps_taken"),
        "elapsed": final_payload.get("elapsed"),
        "event_count": len(events),
        "trace_actions": [row.get("action") for row in trace],
        "metrics": final_payload.get("metrics"),
        "node_count": len(nodes),
        "edge_count": len(edges),
        "node_types": node_types,
        "substrate_audit_summary": substrate.get("audit_summary"),
        "substrate_budget_usage": substrate.get("budget_usage"),
        "session_subgraph_path": substrate.get("session_subgraph_path"),
        "signal_nodes": [
            node_id for node_id, node in nodes.items()
            if isinstance(node, dict) and node.get("node_type") == "signal"
        ],
        "diagnostics_present": "__diag__" in nodes,
    }


def run_direct(args: argparse.Namespace) -> dict[str, Any]:
    if str(FRONTEND_DIR) not in sys.path:
        sys.path.insert(0, str(FRONTEND_DIR))

    import api.frontend_api as frontend_api  # type: ignore
    runtime_config = {
        "model_command": frontend_api.MODEL_COMMAND,
        "model_attach_url": frontend_api.MODEL_ATTACH_URL,
        "resolved_model_command": frontend_api._model_command(),
    }
    model_calls: list[dict[str, Any]] = []
    original_call_model = frontend_api._call_model

    def call_model_with_debug(prompt: str, *, timeout: float = frontend_api.MODEL_TIMEOUT) -> dict[str, Any]:
        call_started = time.perf_counter()
        record: dict[str, Any] = {
            "index": len(model_calls),
            "prompt": prompt,
            "prompt_chars": len(prompt or ""),
            "timeout": timeout,
        }
        try:
            result = original_call_model(prompt, timeout=timeout)
            record.update({
                "ok": True,
                "elapsed": result.get("elapsed"),
                "returncode": result.get("returncode"),
                "answer": result.get("answer"),
                "answer_chars": len(result.get("answer") or ""),
                "stderr": result.get("stderr"),
            })
            return result
        except Exception as exc:
            record.update({
                "ok": False,
                "elapsed": round(time.perf_counter() - call_started, 3),
                "error_type": type(exc).__name__,
                "error": str(exc),
                "traceback": traceback.format_exc(),
            })
            raise
        finally:
            model_calls.append(_jsonable(record))

    frontend_api._call_model = call_model_with_debug

    started = time.perf_counter()
    events: list[dict[str, Any]] = []
    final_payload: dict[str, Any] | None = None
    error: dict[str, Any] | None = None

    req = frontend_api.RunRequest(
        question=args.question,
        graph_id=args.graph_id,
        k_anchors=args.k_anchors,
        anchor_strategy=args.anchor_strategy,
    )

    def emit(name: str, data: dict[str, Any]) -> None:
        events.append(_event(name, data, started))

    health = frontend_api.health()
    events.append(_event("ready", {"ok": True, "provider": "model", "health": health}, started))
    try:
        final_payload = frontend_api._run_graph_agent(req, emit=emit)
        events.append(_event("final", final_payload, started))
    except Exception as exc:  # intentionally raw debug
        error = {
            "type": type(exc).__name__,
            "message": str(exc),
            "traceback": traceback.format_exc(),
        }
        events.append(_event("error", {"message": f"{type(exc).__name__}: {exc}"}, started))

    return {
        "mode": "direct_frontend_api_worker",
        "runtime_config": runtime_config,
        "health": _jsonable(health),
        "model_calls": model_calls,
        "events": events,
        "final_payload": _jsonable(final_payload),
        "session_artifacts": _load_session_artifacts(final_payload or {}),
        "error": error,
        "summary": _summarize(final_payload, events),
    }


def _parse_sse_block(block: str) -> dict[str, Any] | None:
    event_name = "message"
    data_lines: list[str] = []
    for raw in block.splitlines():
        line = raw.rstrip("\r")
        if line.startswith("event:"):
            event_name = line[6:].strip()
        elif line.startswith("data:"):
            data_lines.append(line[5:].lstrip())
    if not data_lines:
        return None
    data_raw = "\n".join(data_lines)
    try:
        data = json.loads(data_raw)
    except json.JSONDecodeError:
        data = {"raw": data_raw}
    return {"event": event_name, "data": data}


def run_http(args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    events: list[dict[str, Any]] = []
    final_payload: dict[str, Any] | None = None
    error: dict[str, Any] | None = None
    payload = json.dumps({
        "question": args.question,
        "graph_id": args.graph_id,
        "k_anchors": args.k_anchors,
        "anchor_strategy": args.anchor_strategy,
    }).encode("utf-8")
    req = urllib.request.Request(
        args.api_url.rstrip("/") + "/api/runs/stream",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    buffer = ""
    try:
        with urllib.request.urlopen(req, timeout=args.http_timeout) as response:
            while True:
                chunk = response.read(4096)
                if not chunk:
                    break
                buffer += chunk.decode("utf-8", errors="replace").replace("\r\n", "\n")
                while "\n\n" in buffer:
                    block, buffer = buffer.split("\n\n", 1)
                    parsed = _parse_sse_block(block)
                    if not parsed:
                        continue
                    events.append(_event(parsed["event"], parsed["data"], started))
                    if parsed["event"] == "final":
                        final_payload = parsed["data"]
                    elif parsed["event"] == "error":
                        error = {"type": "stream_error", "message": parsed["data"]}
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        error = {"type": "HTTPError", "status": exc.code, "body": body}
        events.append(_event("error", {"message": body, "status": exc.code}, started))
    except Exception as exc:
        error = {
            "type": type(exc).__name__,
            "message": str(exc),
            "traceback": traceback.format_exc(),
        }
        events.append(_event("error", {"message": f"{type(exc).__name__}: {exc}"}, started))

    tail = buffer.strip()
    if tail:
        parsed = _parse_sse_block(tail)
        if parsed:
            events.append(_event(parsed["event"], parsed["data"], started))
            if parsed["event"] == "final":
                final_payload = parsed["data"]

    return {
        "mode": "http_sse_frontend_chat",
        "api_url": args.api_url,
        "events": events,
        "final_payload": _jsonable(final_payload),
        "session_artifacts": _load_session_artifacts(final_payload or {}),
        "error": error,
        "summary": _summarize(final_payload, events),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the same graph-agent pipeline used by the frontend chat and save raw debug JSON."
    )
    parser.add_argument("--question", default=DEFAULT_QUESTION)
    parser.add_argument("--graph-id", default="cs4")
    parser.add_argument("--k-anchors", type=int, default=8)
    parser.add_argument("--anchor-strategy", choices=["topk", "mmr", "legacy"], default="topk")
    parser.add_argument("--mode", choices=["direct", "http"], default="direct")
    parser.add_argument("--api-url", default="http://127.0.0.1:8787")
    parser.add_argument("--http-timeout", type=float, default=600)
    parser.add_argument("--model-timeout", type=float, default=300)
    parser.add_argument("--reasoning-mode", choices=["legacy", "substrate"], default="substrate")
    parser.add_argument("--max-iterations", type=int, default=3)
    parser.add_argument("--max-llm-calls", type=int, default=6)
    parser.add_argument("--max-total-tokens", type=int, default=4096)
    parser.add_argument(
        "--attach-url",
        default=None,
        help="Override MODEL_ATTACH_URL/OPENCODE_ATTACH_URL before importing the frontend API.",
    )
    parser.add_argument(
        "--no-attach",
        action="store_true",
        help="Disable the frontend API --attach argument for direct-mode local runs.",
    )
    parser.add_argument(
        "--opencode-dirs",
        choices=["workspace", "profile-data", "inherit"],
        default="workspace",
        help=(
            "workspace keeps OpenCode config/data/state/cache inside this repo; "
            "profile-data uses repo-local config but the normal user-profile data/state/cache; "
            "inherit leaves XDG dirs exactly as provided by the parent process."
        ),
    )
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    args = parser.parse_args()

    if args.opencode_dirs == "workspace":
        os.environ.setdefault("XDG_CONFIG_HOME", str(ROOT / ".opencode_config"))
        os.environ.setdefault("XDG_DATA_HOME", str(ROOT / ".opencode_data"))
        os.environ.setdefault("XDG_STATE_HOME", str(ROOT / ".opencode_state"))
        os.environ.setdefault("XDG_CACHE_HOME", str(ROOT / ".opencode_cache"))
        for key in ("XDG_CONFIG_HOME", "XDG_DATA_HOME", "XDG_STATE_HOME", "XDG_CACHE_HOME"):
            Path(os.environ[key]).mkdir(parents=True, exist_ok=True)
    elif args.opencode_dirs == "profile-data":
        os.environ.setdefault("XDG_CONFIG_HOME", str(ROOT / ".opencode_config"))
        Path(os.environ["XDG_CONFIG_HOME"]).mkdir(parents=True, exist_ok=True)
        for key in ("XDG_DATA_HOME", "XDG_STATE_HOME", "XDG_CACHE_HOME"):
            os.environ.pop(key, None)
    os.environ["REASONING_MODE"] = args.reasoning_mode
    os.environ["MODEL_TIMEOUT"] = str(args.model_timeout)
    os.environ["REASONING_MAX_ITERATIONS"] = str(args.max_iterations)
    os.environ["REASONING_MAX_LLM_CALLS"] = str(args.max_llm_calls)
    os.environ["REASONING_MAX_TOTAL_TOKENS"] = str(args.max_total_tokens)
    if args.no_attach:
        # Python can preserve an empty env var for module-level config, while
        # PowerShell removes empty env vars before the child process starts.
        os.environ["MODEL_ATTACH_URL"] = ""
        os.environ["OPENCODE_ATTACH_URL"] = ""
    elif args.attach_url is not None:
        os.environ["MODEL_ATTACH_URL"] = args.attach_url
        os.environ["OPENCODE_ATTACH_URL"] = args.attach_url

    started_iso = datetime.now().isoformat(timespec="seconds")
    if args.mode == "http":
        debug = run_http(args)
    else:
        debug = run_direct(args)

    debug["request"] = {
        "question": args.question,
        "graph_id": args.graph_id,
        "k_anchors": args.k_anchors,
        "anchor_strategy": args.anchor_strategy,
        "reasoning_mode": args.reasoning_mode,
        "max_iterations": args.max_iterations,
        "max_llm_calls": args.max_llm_calls,
        "max_total_tokens": args.max_total_tokens,
        "attach_url": None if args.no_attach else args.attach_url,
        "no_attach": args.no_attach,
        "opencode_dirs": args.opencode_dirs,
        "started_at": started_iso,
        "xdg_config_home": os.environ.get("XDG_CONFIG_HOME"),
        "xdg_data_home": os.environ.get("XDG_DATA_HOME"),
        "xdg_state_home": os.environ.get("XDG_STATE_HOME"),
        "xdg_cache_home": os.environ.get("XDG_CACHE_HOME"),
    }
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = out_dir / f"frontend_pipeline_debug_{stamp}.json"
    out_path.write_text(json.dumps(debug, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps({
        "debug_path": str(out_path),
        "summary": debug.get("summary"),
        "error": debug.get("error"),
    }, ensure_ascii=False, indent=2))
    return 0 if debug.get("summary", {}).get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
