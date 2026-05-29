"""Small local-model command shim for frontend debug runs.

The frontend API expects a command shaped like:

    <MODEL_COMMAND> run [--attach URL] -

OpenCode implements that shape. This shim implements the same minimal surface
but posts the prompt to a local OpenAI-compatible llama-server instead. It is
intended for Qwen/GLM GGUF smoke tests where we want to bypass OpenCode.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _read_prompt(args: list[str]) -> str:
    if "-" in args or not args:
        return sys.stdin.read()
    return " ".join(args)


def _post_chat(prompt: str, *, system_msg: str | None = None) -> dict[str, Any]:
    base_url = os.environ.get("LOCAL_LLM_BASE_URL", "http://127.0.0.1:6768").rstrip("/")
    temperature = float(os.environ.get("LOCAL_LLM_TEMPERATURE", "0.2"))
    max_tokens = int(os.environ.get("LOCAL_LLM_MAX_TOKENS", "8192"))
    timeout = float(os.environ.get("LOCAL_LLM_TIMEOUT", "240"))
    enable_thinking = os.environ.get("LOCAL_LLM_ENABLE_THINKING", "1").lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    reasoning_effort = os.environ.get("LOCAL_LLM_REASONING_EFFORT", "high")
    messages = []
    if system_msg:
        messages.append({"role": "system", "content": system_msg})
    messages.append({"role": "user", "content": prompt})
    payload = {
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "chat_template_kwargs": {"enable_thinking": enable_thinking},
        "reasoning_effort": reasoning_effort,
        "cache_prompt": True,
    }
    if os.environ.get("LOCAL_LLM_LOGPROBS", "0").lower() in {"1", "true", "yes", "on"}:
        payload["logprobs"] = True
        payload["top_logprobs"] = int(os.environ.get("LOCAL_LLM_TOP_LOGPROBS", "5"))
    req = urllib.request.Request(
        base_url + "/v1/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _extract_content(response: dict[str, Any]) -> str:
    try:
        msg = response["choices"][0]["message"]
        content = str(msg.get("content") or "")
        reasoning = str(msg.get("reasoning_content") or "")
        if reasoning:
            content = f"<think>\n{reasoning}\n</think>\n{content}"
        return content
    except Exception as exc:
        raise RuntimeError(f"unexpected local model response shape: {response!r}") from exc


def _confidence_stats(response: dict[str, Any]) -> dict[str, Any]:
    logprob_items = (
        ((response.get("choices") or [{}])[0].get("logprobs") or {}).get("content")
        or []
    )
    values = [
        float(item["logprob"])
        for item in logprob_items
        if item.get("logprob") is not None and str(item.get("token", "")) != ""
    ]
    margins: list[float] = []
    for item in logprob_items:
        top = item.get("top_logprobs") or []
        if len(top) < 2:
            continue
        try:
            margins.append(float(top[0]["logprob"]) - float(top[1]["logprob"]))
        except Exception:
            continue
    if not values:
        return {"available": False}
    ordered = sorted(values)
    p10_idx = max(0, min(len(ordered) - 1, int(0.10 * (len(ordered) - 1))))
    return {
        "available": True,
        "token_count": len(values),
        "mean_logprob": sum(values) / len(values),
        "min_logprob": min(values),
        "p10_logprob": ordered[p10_idx],
        "mean_top1_top2_margin": (sum(margins) / len(margins)) if margins else None,
    }


def _maybe_log_call(prompt: str, response: dict[str, Any], content: str) -> None:
    if os.environ.get("LOCAL_LLM_LOGPROBS", "0").lower() not in {"1", "true", "yes", "on"}:
        return
    root = Path(os.environ.get("LOCAL_LLM_CONFIDENCE_LOG_ROOT", "data/model_confidence"))
    root.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d")
    path = root / f"local_model_calls_{stamp}.jsonl"
    row = {
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "base_url": os.environ.get("LOCAL_LLM_BASE_URL", "http://127.0.0.1:6768"),
        "prompt_sha256": hashlib.sha256(prompt.encode("utf-8", errors="replace")).hexdigest(),
        "prompt_chars": len(prompt),
        "answer_chars": len(content),
        "confidence_stats": _confidence_stats(response),
        "usage": response.get("usage"),
        "model": response.get("model"),
    }
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def main(argv: list[str] | None = None) -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    parser = argparse.ArgumentParser(description="Run a local llama-server model.")
    sub = parser.add_subparsers(dest="command")
    run_p = sub.add_parser("run")
    run_p.add_argument("message", nargs="*")
    run_p.add_argument("--attach", default=None)  # accepted for frontend compatibility; ignored
    run_p.add_argument("--format", default=None)  # accepted for rough opencode compatibility; ignored
    args, unknown = parser.parse_known_args(argv)

    if args.command != "run":
        parser.print_help(sys.stderr)
        return 2

    prompt = _read_prompt(list(args.message) + unknown)
    try:
        response = _post_chat(prompt)
        content = _extract_content(response).strip()
        _maybe_log_call(prompt, response, content)
        sys.stdout.write(content + "\n")
        return 0
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        sys.stderr.write(f"local model HTTP {exc.code}: {body[:2000]}\n")
        return 1
    except Exception as exc:
        sys.stderr.write(f"local model error: {exc}\n")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
