"""Parse Python source -> symbol nodes (raw_edit candidates) for the v2 code graph.

DATA only. Python `ast` first (the SWE-bench distribution). Each module/class/function
becomes a node with a terse `text` (signature + 1-line docstring) for embedding +
grounding; edges are `contains` (module->symbol, class->method). Scoped to given
files (the patch neighborhood), not the whole repo. See SWE_DATA_PIPELINE.md.

Emits the same `raw_edit` envelope the grower's apply path consumes (op/add_node,
op/add_edge), so code nodes flow through stitch -> hub-wire -> gated apply like KB
nodes. node_type `symbol`/`module` are code-graph types (extend the GNN vocab when
wiring to the attended graph; for now this just builds + inspects the graph).
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from v5.operator_schema import op_kind_for, edge_op_for   # locked Operator-Attention schema (torch-free)

MAX_TEXT = 1000   # signature + summary + body (enriched) — body carries the bug-relevant code


def _h(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()[:12]


def _signature(node: ast.AST) -> str:
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        a = node.args
        parts = [arg.arg for arg in (a.posonlyargs + a.args)]
        if a.vararg:
            parts.append("*" + a.vararg.arg)
        parts += [arg.arg for arg in a.kwonlyargs]
        if a.kwarg:
            parts.append("**" + a.kwarg.arg)
        prefix = "async def " if isinstance(node, ast.AsyncFunctionDef) else "def "
        return f"{prefix}{node.name}({', '.join(parts)})"
    if isinstance(node, ast.ClassDef):
        try:
            bases = [ast.unparse(b) for b in node.bases]
        except Exception:
            bases = []
        return f"class {node.name}" + (f"({', '.join(bases)})" if bases else "")
    return getattr(node, "name", "?")


def _summary(node: ast.AST) -> str:
    try:
        doc = ast.get_docstring(node)
    except Exception:
        doc = None
    if not doc:
        return ""
    lines = doc.strip().splitlines()   # whitespace-only docstring -> empty list
    return (lines[0] if lines else "")[:160]


def extract_file(abs_path: str | Path, repo: str = "", rel: Optional[str] = None
                 ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Parse one Python file -> (node_edits, edge_edits)."""
    rel = rel or str(abs_path)
    try:
        src = Path(abs_path).read_text(encoding="utf-8", errors="replace")
        tree = ast.parse(src)
    except (SyntaxError, ValueError, OSError):
        return [], []

    nodes: List[Dict[str, Any]] = []
    edges: List[Dict[str, Any]] = []
    file_id = f"file_{_h(repo + rel)}"
    nodes.append({"op": "add_node", "node_id": file_id, "node_type": "module",
                  "text": rel, "tier": "add",
                  "metadata": {"repo": repo, "file": rel, "kind": "module",
                               "op_kind": op_kind_for("module")}})

    def visit(parent: ast.AST, parent_id: str, qual: str) -> None:
        for child in ast.iter_child_nodes(parent):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                name = f"{qual}.{child.name}" if qual else child.name
                sid = f"sym_{_h(repo + rel + name)}"
                kind = ("class" if isinstance(child, ast.ClassDef)
                        else "method" if qual else "function")
                sig, summ = _signature(child), _summary(child)
                try:                          # enrich with the BODY -> the embedding matches the
                    body = ast.get_source_segment(src, child) or ""   # bug-relevant code, not just the signature
                except Exception:             # noqa: BLE001
                    body = ""
                head = sig + (" — " + summ if summ else "")
                text = (head + ("\n" + body if body else ""))[:MAX_TEXT]
                nodes.append({"op": "add_node", "node_id": sid, "node_type": "symbol",
                              "text": text, "tier": "add",
                              "metadata": {"repo": repo, "file": rel, "name": name,
                                           "kind": kind, "op_kind": op_kind_for("symbol"),
                                           "lineno": [child.lineno,
                                                      getattr(child, "end_lineno", child.lineno)]}})
                edges.append({"op": "add_edge", "src": parent_id, "dst": sid,
                              "relation": "contains", "tier": "add",
                              "metadata": {"edge_op": edge_op_for("contains")}})
                if isinstance(child, ast.ClassDef):   # recurse for methods
                    visit(child, sid, name)
    visit(tree, file_id, "")
    return nodes, edges


def extract_paths(repo_dir: str | Path, files: Sequence[str], repo: str = ""
                  ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Parse a list of repo-relative files -> combined nodes + edges."""
    repo_dir = Path(repo_dir)
    all_nodes: List[Dict[str, Any]] = []
    all_edges: List[Dict[str, Any]] = []
    for rel in files:
        if not rel.endswith(".py"):
            continue
        p = repo_dir / rel
        if not p.exists():
            continue
        n, e = extract_file(p, repo=repo, rel=rel)
        all_nodes.extend(n)
        all_edges.extend(e)
    return all_nodes, all_edges


def symbol_at_line(nodes: List[Dict[str, Any]], rel: str, line: int) -> Optional[str]:
    """Smallest symbol whose [lineno..end] span contains `line` in file `rel` -> node_id."""
    best, best_span = None, 1 << 30
    for n in nodes:
        m = n.get("metadata", {})
        if n.get("node_type") == "symbol" and m.get("file") == rel:
            lo, hi = m.get("lineno", [0, 0])
            if lo <= line <= hi and (hi - lo) < best_span:
                best, best_span = n["node_id"], hi - lo
    return best


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Parse Python files -> code-graph symbol nodes.")
    ap.add_argument("--repo-dir", required=True)
    ap.add_argument("--repo", default="")
    ap.add_argument("--files", nargs="*", default=[], help="repo-relative .py files (default: patch files)")
    ap.add_argument("--out", default="", help="write nodes+edges jsonl here (optional)")
    args = ap.parse_args(argv)

    nodes, edges = extract_paths(args.repo_dir, args.files, repo=args.repo)
    syms = [n for n in nodes if n["node_type"] == "symbol"]
    print(f"parsed {len(args.files)} files -> {len(nodes)} nodes ({len(syms)} symbols) / {len(edges)} edges")
    import collections
    kinds = collections.Counter(n["metadata"].get("kind") for n in nodes)
    print("  kinds:", dict(kinds))
    print("  sample symbols:")
    for n in syms[:12]:
        m = n["metadata"]
        print(f"    [{m['kind']:8}] {m['file']}:{m['lineno'][0]:>4}  {n['text'][:80]}")
    if args.out:
        with open(args.out, "w", encoding="utf-8") as h:
            for x in nodes + edges:
                h.write(json.dumps(x, ensure_ascii=False) + "\n")
        print(f"  wrote -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
