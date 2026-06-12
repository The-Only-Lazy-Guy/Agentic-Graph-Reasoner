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
import torch
import torch.nn as nn

from v5.graph_grower.code_extract import extract_paths


def _unit(v):
    a = np.asarray(v, dtype=np.float32)
    return a / (np.linalg.norm(a, axis=-1, keepdims=True) + 1e-9)


class SymDelta(nn.Module):
    """symbol_emb -> symbol_emb + zero-init MLP(symbol_emb). delta=0 at init == naive cosine.
    Query side stays the frozen embedding (preserves the shared-embedding alignment)."""
    def __init__(self, dim, hidden=512):
        super().__init__()
        self.f = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, dim))
        nn.init.zeros_(self.f[-1].weight); nn.init.zeros_(self.f[-1].bias)

    def forward(self, s):
        out = s + self.f(s)
        return out / (out.norm(dim=-1, keepdim=True) + 1e-9)


def load_delta(path, dim, device):
    m = SymDelta(dim).to(device)
    m.load_state_dict(torch.load(path, map_location=device)); m.eval()
    return m


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
                     max_files: int = 400, max_syms: int = 1500, delta=None) -> List[dict]:
    """Issue -> top-K symbol nodes by cosine(issue, signature). delta = trained SymDelta (applied
    to symbol embeddings) or None (naive cosine). Returns node dicts. NO gold knowledge."""
    files = _pool_files(dest, max_files)
    nodes, _ = extract_paths(dest, files, repo="")
    syms = [n for n in nodes if n.get("node_type") == "symbol" and (n.get("text") or "").strip()]
    if not syms:
        return []
    syms = syms[:max_syms]
    qv = _unit(embedder.embed_nodes({"q": issue[:1500]})["q"]).reshape(-1)
    embs = embedder.embed_nodes({n["node_id"]: n["text"] for n in syms})
    pool = _unit(np.stack([embs[n["node_id"]] for n in syms]))         # [N, D]
    if delta is not None:
        dev = next(delta.parameters()).device
        with torch.no_grad():
            pool = delta(torch.tensor(pool, device=dev)).cpu().numpy()
    scores = pool @ qv
    order = np.argsort(-scores)[:k]
    return [syms[i] for i in order]


def to_meta(nodes: List[dict]) -> Dict[str, dict]:
    """Retrieved nodes -> the {node_id: {file, lineno, text}} shape the loop expects."""
    out = {}
    for n in nodes:
        m = n.get("metadata") or {}
        if m.get("file") and m.get("lineno"):
            out[n["node_id"]] = {"file": m["file"], "lineno": m["lineno"], "text": n.get("text", "")}
    return out
