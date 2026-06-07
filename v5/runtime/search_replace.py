"""Search/Replace edit format — the weak-model-friendly emission contract (plan: format fix).

The 4B can't reliably emit a unified diff (line-number/hunk-header math -> `@@ -XXX +XXX @@`
garbage, ~15/20 applyable at best). Search/Replace blocks remove that tax: the model writes
EXACT old code -> new code (pure Python, its strength), and a tiny applier deterministically
produces a real, applyable patch. Same idea as Aider/Cursor for weak-ish models.

Format (one block per edit):
    path/to/file.py
    <<<<<<< SEARCH
    <exact existing code>
    =======
    <replacement code>
    >>>>>>> REPLACE

This module: SR_SYS prompt + a robust state-machine parser + an applier (apply to a checked-out
repo -> git diff) + format/adherence metrics. No GPU; parser is unit-tested locally.
"""
from __future__ import annotations

import re
import subprocess
from pathlib import Path
from typing import Dict, List

SR_SYS = (
    "You are fixing a bug in a Python project. Output ONLY search/replace edit blocks in "
    "EXACTLY this format — NO unified diff, NO line numbers, NO prose:\n"
    "path/to/file.py\n"
    "<<<<<<< SEARCH\n"
    "<the exact existing code to find>\n"
    "=======\n"
    "<the replacement code>\n"
    ">>>>>>> REPLACE\n"
    "The SEARCH text must match the existing source EXACTLY (copy it). One block per edit; "
    "keep edits minimal."
)

_M_SEARCH = re.compile(r"^<{5,}\s*SEARCH\s*$")
_M_DIV = re.compile(r"^={5,}\s*$")
_M_REPL = re.compile(r"^>{5,}\s*REPLACE\s*$")
_FILE_RE = re.compile(r"([\w./\-]+\.\w+)")


def _file_of(line: str) -> str:
    m = _FILE_RE.search(line or "")
    return m.group(1) if m else ""


def parse_sr(text: str) -> List[Dict[str, str]]:
    """Robust state-machine parse -> [{file, search, replace}]. The file is taken from the
    last meaningful line before the SEARCH marker (handles `path`, `### path`, `file: path`)."""
    lines = (text or "").splitlines()
    blocks: List[Dict[str, str]] = []
    i, last = 0, ""
    while i < len(lines):
        if _M_SEARCH.match(lines[i].strip()):
            fpath = _file_of(last)
            i += 1
            search = []
            while i < len(lines) and not _M_DIV.match(lines[i].strip()):
                search.append(lines[i]); i += 1
            i += 1  # skip divider
            repl = []
            while i < len(lines) and not _M_REPL.match(lines[i].strip()):
                repl.append(lines[i]); i += 1
            i += 1  # skip REPLACE marker
            blocks.append({"file": fpath, "search": "\n".join(search), "replace": "\n".join(repl)})
        else:
            if lines[i].strip():
                last = lines[i]
            i += 1
    return blocks


def apply_sr(repo_dir: str, blocks: List[Dict[str, str]]) -> tuple:
    """Apply blocks to a checked-out repo (exact-string replace) -> (applied_count, git_diff)."""
    applied = 0
    for b in blocks:
        fp = Path(repo_dir) / b["file"]
        if not fp.exists() or not b["search"].strip():
            continue
        txt = fp.read_text(encoding="utf-8", errors="ignore")
        if b["search"] in txt:
            fp.write_text(txt.replace(b["search"], b["replace"], 1), encoding="utf-8")
            applied += 1
    diff = subprocess.run(["git", "-C", repo_dir, "diff"], capture_output=True, text=True).stdout
    return applied, diff


def sr_metrics(blocks: List[Dict[str, str]], gold_files: List[str], gold_syms: List[str]) -> dict:
    """Format/adherence on parsed blocks (no repo needed). well_formed = emitted >=1 block."""
    gnames = {Path(g).name for g in gold_files}
    file_hit = any(Path(b["file"]).name in gnames for b in blocks if b["file"]) if gnames else False
    blob = "\n".join(b["search"] + "\n" + b["replace"] for b in blocks)
    sym_hit = sum(1 for s in gold_syms if s and re.search(rf"\b{re.escape(s)}\b", blob))
    return {"n_blocks": len(blocks),
            "well_formed": 1.0 if blocks else 0.0,
            "file_cov": 1.0 if file_hit else 0.0,
            "edit_cov": sym_hit / max(1, len(gold_syms))}
