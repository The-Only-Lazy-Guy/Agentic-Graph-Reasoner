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
import re
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn

from v5.graph_grower.code_extract import extract_paths

_STOP = {"this", "that", "with", "from", "have", "when", "should", "which", "will", "python",
         "code", "error", "issue", "test", "tests", "class", "function", "method", "value",
         "return", "self", "none", "true", "false", "would", "could", "there", "their", "where",
         "what", "into", "also", "only", "like", "used", "using", "example", "expected", "actual"}


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


def _all_py(dest: str) -> List[str]:
    """All repo .py files (relative), excluding tests/docs/build."""
    dest = Path(dest)
    skip = ("/test", "/tests/", "/docs/", "/doc/", "/build/", "/.git/", "/examples/", "/benchmarks/")
    out = []
    for p in glob.glob(str(dest / "**" / "*.py"), recursive=True):
        rel = os.path.relpath(p, dest).replace("\\", "/")
        if not any(s in "/" + rel.lower() for s in skip):
            out.append(rel)
    return out


def _issue_keywords(issue: str) -> List[str]:
    """Distinctive identifiers/words from the issue — the cheap file-filter signal."""
    toks = re.findall(r"[A-Za-z_][A-Za-z0-9_]{3,}", issue or "")
    return list({t for t in toks if t.lower() not in _STOP})[:60]


def _rank_files(dest: str, issue: str, file_k: int = 40) -> List[str]:
    """STAGE 1 (cheap, no embedding): rank repo files by issue-keyword occurrences -> top file_k.
    The issue names the buggy area's identifiers, which appear in the gold file -> good coverage."""
    kws = _issue_keywords(issue)
    files = _all_py(dest)
    if not kws or not files:
        files.sort(key=lambda r: (r.count("/"), len(r)))
        return files[:file_k]
    scored = []
    for rel in files:
        try:
            txt = (Path(dest) / rel).read_text(encoding="utf-8", errors="ignore")
        except Exception:   # noqa: BLE001
            continue
        s = sum(txt.count(k) for k in kws)
        if s > 0:
            scored.append((s, rel))
    scored.sort(reverse=True)
    return [rel for _, rel in scored[:file_k]] or files[:file_k]


# back-compat: shallow pool (unused by the two-stage path)
def _pool_files(dest: str, max_files: int = 400) -> List[str]:
    files = _all_py(dest)
    files.sort(key=lambda r: (r.count("/"), len(r)))
    return files[:max_files]


def load_st_ranker(path):
    """Load the trained bi-encoder (sentence-transformers) — the DESIGN's locked ranker."""
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(path, trust_remote_code=True)


def _body_lex_scores(dest, syms, kws):
    """Issue-keyword occurrences in each symbol's BODY (richer than the signature). Cheap."""
    cache = {}
    out = np.zeros(len(syms), dtype=np.float32)
    for i, n in enumerate(syms):
        m = n.get("metadata") or {}
        f, ln = m.get("file"), m.get("lineno")
        if not f or not ln:
            continue
        if f not in cache:
            try:
                cache[f] = (Path(dest) / f).read_text(encoding="utf-8", errors="ignore").splitlines()
            except Exception:   # noqa: BLE001
                cache[f] = []
        lo, hi = ln[0], ln[-1]
        body = "\n".join(cache[f][max(0, lo - 1):hi])
        out[i] = sum(body.count(k) for k in kws)
    return out


def retrieve_support(dest: str, issue: str, embedder, k: int = 8, max_files: int = 40,
                     max_syms: int = 1500, delta=None, st_model=None, hybrid_lex: float = 0.5) -> List[dict]:
    """Two-stage: STAGE 1 lexical file-filter -> STAGE 2 rank top-file symbols by
    EMBEDDING (signature, st_model/embedder) + hybrid_lex * BODY-keyword match (the body carries
    the bug-relevant identifiers the signature omits). NO gold knowledge."""
    files = _rank_files(dest, issue, max_files)
    nodes, _ = extract_paths(dest, files, repo="")
    syms = [n for n in nodes if n.get("node_type") == "symbol" and (n.get("text") or "").strip()]
    if not syms:
        return []
    syms = syms[:max_syms]
    texts = [n["text"] for n in syms]
    if st_model is not None:
        pool = _unit(np.asarray(st_model.encode(texts, batch_size=64, show_progress_bar=False)))
        qv = _unit(np.asarray(st_model.encode([issue[:1500]]))[0])
    else:
        qv = _unit(embedder.embed_nodes({"q": issue[:1500]})["q"]).reshape(-1)
        embs = embedder.embed_nodes({n["node_id"]: t for n, t in zip(syms, texts)})
        pool = _unit(np.stack([embs[n["node_id"]] for n in syms]))
        if delta is not None:
            dev = next(delta.parameters()).device
            with torch.no_grad():
                pool = delta(torch.tensor(pool, device=dev)).cpu().numpy()
    emb = pool @ qv
    score = emb
    if hybrid_lex > 0:
        lex = _body_lex_scores(dest, syms, _issue_keywords(issue))
        if lex.max() > 0:                                   # min-max normalize both, then blend
            lex = lex / lex.max()
            emb = (emb - emb.min()) / (emb.max() - emb.min() + 1e-9)
            score = emb + hybrid_lex * lex
    order = np.argsort(-score)[:k]
    return [syms[i] for i in order]


def to_meta(nodes: List[dict]) -> Dict[str, dict]:
    """Retrieved nodes -> the {node_id: {file, lineno, text}} shape the loop expects."""
    out = {}
    for n in nodes:
        m = n.get("metadata") or {}
        if m.get("file") and m.get("lineno"):
            out[n["node_id"]] = {"file": m["file"], "lineno": m["lineno"], "text": n.get("text", "")}
    return out
