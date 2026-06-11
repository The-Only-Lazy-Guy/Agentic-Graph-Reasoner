"""Real localization — retrieve support symbols from the issue (NOT the gold patch).

The eval pipeline uses ORACLE support (`swe_grounded` AST-maps the GOLD patch's changed lines
to symbols). That measures synthesis GIVEN the location; it does NOT test the graph's actual job
(find the right symbols among the whole repo). This module does the real thing:

  repo @ base_commit -> extract ALL symbols of the main package(s) (the retrieval pool)
    -> embed the issue + every symbol signature -> cosine rank -> top-K = support

Then the loop reads + injects THOSE (not gold). Reports recall@K vs the gold-touched symbols
(localization quality) so we can separate "did it find the bug" from "did it write the fix".
Tractable: capped file/symbol budget; one embed pass per instance.
"""
from __future__ import annotations

import glob
import os
from pathlib import Path
from typing import Dict, List

import numpy as np

from v5.graph_grower.code_extract import extract_paths


def _unit(v):
    a = np.asarray(v, dtype=np.float32)
    return a / (np.linalg.norm(a) + 1e-9)


def _pool_files(dest: str, max_files: int = 400) -> List[str]:
    """Main-package .py files (the realistic retrieval pool) — exclude tests/docs/build."""
    dest = Path(dest)
    skip = ("/test", "/tests/", "/docs/", "/doc/", "/build/", "/.git/", "/examples/", "/benchmarks/")
    out = []
    for p in glob.glob(str(dest / "**" / "*.py"), recursive=True):
        rel = os.path.relpath(p, dest).replace("\\", "/")
        low = "/" + rel.lower()
        if any(s in low for s in skip):
            continue
        out.append(rel)
    out.sort(key=lambda r: (r.count("/"), len(r)))     # shallow/core files first
    return out[:max_files]


def retrieve_support(dest: str, issue: str, embedder, k: int = 8,
                     max_files: int = 400, max_syms: int = 1500) -> List[dict]:
    """Issue -> top-K symbol nodes by cosine(issue, signature). Returns node dicts
    ({node_id, text, metadata:{file,lineno}}). NO gold knowledge."""
    files = _pool_files(dest, max_files)
    nodes, _ = extract_paths(dest, files, repo="")
    syms = [n for n in nodes if n.get("node_type") == "symbol" and (n.get("text") or "").strip()]
    if not syms:
        return []
    syms = syms[:max_syms]
    qv = _unit(embedder.embed_nodes({"q": issue[:1500]})["q"])
    embs = embedder.embed_nodes({n["node_id"]: n["text"] for n in syms})
    ranked = sorted(syms, key=lambda n: float(np.dot(qv, _unit(embs[n["node_id"]]))), reverse=True)
    return ranked[:k]


def to_meta(nodes: List[dict]) -> Dict[str, dict]:
    """Retrieved nodes -> the {node_id: {file, lineno, text}} shape the loop expects."""
    out = {}
    for n in nodes:
        m = n.get("metadata") or {}
        if m.get("file") and m.get("lineno"):
            out[n["node_id"]] = {"file": m["file"], "lineno": m["lineno"], "text": n.get("text", "")}
    return out
