"""L0 SYNTAX memory — symbols/signatures/identifiers ("every single bit, even syntax").

Wraps code_extract.extract_paths (AST symbols: signature + summary + body[:1000], stable
sym_<sha1> ids) into a persistent store: JSONL records + mpnet embeddings. Also owns
`ident_overlap` — the local-fit half of the memory read path (does a candidate
implementation talk about the SAME identifiers as the code in front of the model?) —
and the per-file identifier vocab used later for constrained decode.

  python -m v5.memory.syntax --selftest
"""
from __future__ import annotations

import re
from pathlib import Path

from v5.memory.store import EmbStore, JsonlWal

_IDENT_RE = re.compile(r"\b[A-Za-z_][A-Za-z0-9_]{2,}\b")
_STOP = {"def", "class", "return", "self", "None", "True", "False", "import", "from",
         "for", "while", "if", "elif", "else", "try", "except", "with", "raise", "pass",
         "and", "not", "in", "is", "or", "lambda", "yield", "assert", "del", "global",
         "the", "this", "that", "function", "value", "values", "int", "str", "list",
         "dict", "print", "len", "range"}


def idents(text: str) -> set[str]:
    return {t for t in _IDENT_RE.findall(text or "") if t not in _STOP}


def ident_overlap(a_text: str, b_text: str) -> float:
    """Identifier compatibility in [0,1]: |A∩B| / sqrt(|A||B|) (cosine on binary sets)."""
    a, b = idents(a_text), idents(b_text)
    if not a or not b:
        return 0.0
    return len(a & b) / ((len(a) * len(b)) ** 0.5)


class SyntaxStore:
    """Persistent symbol store: records WAL + signature embeddings."""

    def __init__(self, root: str | Path = "data/memory", embed_fn=None):
        self.root = Path(root)
        self.wal = JsonlWal(self.root / "l0_syntax.jsonl")
        self.emb = EmbStore(self.root, "l0")
        self.embed_fn = embed_fn
        self.records: dict[str, dict] = {r["id"]: r for r in self.wal.read_all()}

    def __len__(self) -> int:
        return len(self.records)

    def scan_files(self, repo_dir: str, files: list[str], repo: str = "") -> int:
        """AST-extract symbols from files; add NEW ones (content-addressed ids dedup
        naturally: an edited symbol gets a new id, the stale one just stops being cited)."""
        from v5.graph_grower.code_extract import extract_paths
        nodes, _edges = extract_paths(repo_dir, files, repo=repo)
        fresh = []
        for n in nodes:
            if n.get("node_type") != "symbol" or n["node_id"] in self.records:
                continue
            md = n.get("metadata", {})
            rec = {"id": n["node_id"], "name": md.get("name", ""), "kind": md.get("kind", ""),
                   "file": md.get("file", ""), "repo": md.get("repo", repo),
                   "text": n.get("text", "")}
            fresh.append(rec)
        return self._add(fresh)

    def add_snippets(self, snippets: list[dict]) -> int:
        """Non-AST route (task specs, gold snippets): dicts with id/name/kind/file/text."""
        return self._add([s for s in snippets if s["id"] not in self.records])

    def _add(self, recs: list[dict]) -> int:
        if not recs:
            return 0
        if self.embed_fn is not None:
            vecs = self.embed_fn({r["id"]: r["text"] for r in recs})
            self.emb.add([r["id"] for r in recs], [vecs[r["id"]] for r in recs])
        for r in recs:
            self.wal.append(r)
            self.records[r["id"]] = r
        return len(recs)

    def vocab_for_file(self, file: str) -> set[str]:
        """Identifier vocabulary of one file's symbols (constrained-decode source)."""
        out: set[str] = set()
        for r in self.records.values():
            if r.get("file") == file:
                out |= idents(r["text"])
        return out

    def search(self, query_vec, k: int = 8) -> list[dict]:
        return [dict(self.records[i], score=s) for i, s in self.emb.search(query_vec, k)
                if i in self.records]


# ── selftest ────────────────────────────────────────────────────────────────────

def _selftest() -> bool:
    import tempfile
    from v5.memory.store import make_fake_embedder
    print("memory.syntax --selftest: idents, overlap, scan, vocab, persist (no model)\n")

    assert idents("def get_url(retry_count): return _raw(url)") >= {"get_url", "retry_count", "_raw", "url"}
    assert "def" not in idents("def foo(): pass")
    hi = ident_overlap("wss.on('error', handler)", "the wss server error handler closes")
    lo = ident_overlap("wss.on('error', handler)", "matrix multiply kernel stride")
    assert hi > lo and hi > 0.3, f"overlap ordering hi={hi:.2f} lo={lo:.2f}"
    assert ident_overlap("", "x") == 0.0
    print("  [1] idents + ident_overlap -> PASS")

    with tempfile.TemporaryDirectory() as td:
        repo = Path(td) / "repo"
        repo.mkdir()
        (repo / "app.py").write_text(
            "def get_url(retry_count):\n    return _raw(url)\n\n"
            "class Client:\n    def send(self, payload):\n        return post(payload)\n",
            encoding="utf-8")
        st = SyntaxStore(Path(td) / "mem", embed_fn=make_fake_embedder())
        n = st.scan_files(str(repo), ["app.py"], repo="test")
        assert n >= 3, f"expected >=3 symbols (get_url, Client, send), got {n}"
        assert st.scan_files(str(repo), ["app.py"], repo="test") == 0, "rescan dedups"
        vocab = st.vocab_for_file("app.py")
        assert {"get_url", "payload"} <= vocab, vocab
        st2 = SyntaxStore(Path(td) / "mem", embed_fn=make_fake_embedder())
        assert len(st2) == len(st) and len(st2.emb) == len(st.emb), "reload persists"
        got = st2.search(st2.emb.get([next(iter(st2.records))])[0], k=1)
        assert got and got[0]["id"] in st2.records
        print(f"  [2] scan({n} syms) + dedup + vocab + reload + search -> PASS")

    print("\n  MEMORY.SYNTAX SELFTEST -> PASS")
    return True


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    if ap.parse_args().selftest:
        raise SystemExit(0 if _selftest() else 1)
    ap.print_help()
