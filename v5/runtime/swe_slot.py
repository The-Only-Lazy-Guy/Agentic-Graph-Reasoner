"""#9 (synthesis-focused, ENGINE-WIRED) — slot-graph DIAGNOSE->PLAN->FIX vs one-shot, through SlotGraph.solve.

Localization held FIXED (gold support symbols) to isolate SYNTHESIS from the ~0.30 localization wall.
The SLOT path now runs the REAL engine (slot_coder.SlotGraph): DIAGNOSE (revise='rederive') ->
PLAN (quote the exact source anchor + intended change) -> FIX. An ungrounded PLAN or an unapplyable /
misaligned FIX -> INSUFFICIENT -> dependency-directed BACKTRACK to the nearest upstream slot, then
re-plan / re-diagnose deeper, to a fixpoint (or max_steps). Compared to ONE-SHOT (single SR emit).

`slot_solve()` is shared by the real 4B run AND a no-model `--selftest` that PROVES the engine wiring
(DIAGNOSE->PLAN->FIX->INSUFFICIENT->backtrack->re-plan->anchored-fix->fixpoint) without a GPU. This is
the fix for the earlier mistake where swe_slot bypassed the engine entirely (see memory verify-wiring-not-proxy).

  4B (A40): V5_LM_TRUST_REMOTE_CODE=1 python -m v5.runtime.swe_slot --n-eval 24
  session : ... --session-out-dir artifacts/swe_slot_sessions --session-name lite_n24_run1
  exact   : ... --exact-verify --verify-backend docker    # gold-sanity + exact resolve on this box
  wiring  : python -m v5.runtime.swe_slot --selftest      # no model, proves the slot-graph is invoked
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import subprocess
import time
from pathlib import Path

from v5.runtime.swe_exact_verify import SWEExactVerifier
from v5.runtime.slot_coder import SlotGraph, SlotSpec, Pool


def _canon_path(p: str) -> str:
    return (p or "").replace("\\", "/").strip()


def _same_file(a: str, b: str) -> bool:
    aa, bb = _canon_path(a), _canon_path(b)
    return bool(aa and bb) and (aa == bb or aa.endswith("/" + bb) or bb.endswith("/" + aa))


def _split_src_files(src: str) -> dict[str, str]:
    parts: dict[str, list[str]] = {}
    cur_file = ""
    cur_lines: list[str] = []
    for line in (src or "").splitlines():
        m = re.match(r"^# ([\w./\-]+\.\w+)\s*$", line)
        if m:
            if cur_file:
                parts[cur_file] = cur_lines[:]
            cur_file = _canon_path(m.group(1))
            cur_lines = []
            continue
        if cur_file:
            cur_lines.append(line)
    if cur_file:
        parts[cur_file] = cur_lines[:]
    return {k: "\n".join(v).rstrip("\n") for k, v in parts.items()}


def _format_plan(plan: dict[str, str]) -> str:
    return f"FILE: {plan['file']}\nSEARCH:\n{plan['search']}\nCHANGE:\n{plan['change']}\n"


def _render_fix_plan(plan_text: str) -> str:
    plan = _parse_plan(plan_text)
    if not any(plan.values()):
        return plan_text
    parts = []
    if plan["file"]:
        parts.append(f"TARGET FILE:\n{plan['file']}")
    if plan["search"]:
        parts.append(f"TARGET SEARCH ANCHOR:\n{plan['search']}")
    if plan["change"]:
        parts.append(f"EDIT INTENT:\n{plan['change']}")
    return "\n\n".join(parts)


DIAG_SYS = (
    "You are a terse debugging assistant. Use only the retrieved file set. "
    "Follow the user's FILE/FOCUS/WHY/CHANGE schema exactly, with no bullets, no numbered list, "
    "no markdown fences, and no 'Thinking Process' or speculation about files not shown."
)


PLAN_SYS = (
    "You are a strict patch planner. Output ONLY FILE/SEARCH/CHANGE with no preface or trailing commentary. "
    "FILE must stay inside the authoritative file set shown by the user prompt."
)


def _tokset(text: str) -> set[str]:
    toks = set()
    for m in re.finditer(r"[A-Za-z_][A-Za-z0-9_./-]*", text or ""):
        raw = m.group(0).lower()
        if len(raw) >= 3:
            toks.add(raw)
        pieces = [p for p in re.split(r"[^a-z0-9]+", raw) if len(p) >= 3]
        toks.update(pieces)
        for piece in pieces:
            toks.update(
                part.lower()
                for part in re.findall(r"[A-Z]?[a-z]+|[A-Z]+(?=[A-Z]|$)|\d+", piece)
                if len(part) >= 3
            )
    return toks


def _source_text_no_header(text: str) -> str:
    lines = (text or "").splitlines()
    if lines and re.match(r"^# [\w./\-]+\.\w+\s*$", lines[0].strip()):
        lines = lines[1:]
    return "\n".join(lines)


def _extract_symbol_label(text: str) -> str:
    body = _source_text_no_header(text)
    lines = body.splitlines()
    for idx, line in enumerate(lines):
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("@") and idx + 1 < len(lines):
            nxt = lines[idx + 1].strip()
            if re.match(r"^(def|class)\s+\w+", nxt):
                return f"{stripped} {nxt}".strip()
        if re.match(r"^(def|class)\s+\w+", stripped):
            return stripped.rstrip(":")
    for line in lines:
        stripped = line.strip()
        if stripped:
            return stripped[:100]
    return ""


def _source_label(source: dict) -> str:
    label = (source.get("label") or "").strip()
    if label:
        return label
    return _extract_symbol_label(source.get("text", ""))


def _source_span(source: dict) -> int:
    loc = source.get("lineno")
    if isinstance(loc, (list, tuple)) and len(loc) >= 2:
        try:
            return max(1, int(loc[1]) - int(loc[0]) + 1)
        except Exception:
            pass
    return max(1, len(_source_text_no_header(source.get("text", "")).splitlines()))


def _source_has_definition(source: dict) -> bool:
    return bool(re.search(r"(?m)^\s*(?:def|class)\s+\w+", _source_text_no_header(source.get("text", ""))))


def _source_has_executable(source: dict) -> bool:
    return any(
        re.search(r"^\s*(if|elif|else|for|while|with|try|except|return|raise|assert|yield)\b", line)
        or ("=" in line and "==" not in line)
        for line in _source_text_no_header(source.get("text", "")).splitlines()
    )


def _compose_src(chunks: list[dict]) -> str:
    parts = []
    seen = set()
    for chunk in chunks or []:
        cid = chunk.get("id")
        text = (chunk.get("text") or "").strip("\n")
        if not text or cid in seen:
            continue
        seen.add(cid)
        parts.append(text)
    return "\n\n".join(parts)


def _focus_body(gate_src: str, diag: str, max_lines: int = 120) -> str:
    """Verbatim body of the DIAGNOSE-named FOCUS function, read from the REAL files (gate_src). The
    sparse retrieved support often omits the exact call to edit, so the model regenerates it from
    memory and diverges (applyable-but-wrong); feeding the real body gives it the source to COPY."""
    if not (gate_src and diag):
        return ""
    m_focus = re.search(r'FOCUS:\s*([^\n(]+)', diag)
    if not m_focus:
        return ""
    focus = m_focus.group(1).strip().split()[0].strip("`'\":()")
    if not focus:
        return ""
    files = _split_src_files(gate_src)
    m_file = re.search(r'FILE:\s*(\S+)', diag)
    fpath = _resolve_src_file(m_file.group(1), files) if m_file else ""
    pat = re.compile(rf'^(\s*)(?:def|class)\s+{re.escape(focus)}\b')
    if not (fpath and fpath in files):
        fpath = next((p for p, b in files.items() if pat.search(b)), "")
    if not (fpath and fpath in files):
        return ""
    lines = files[fpath].splitlines()
    start = next((i for i, ln in enumerate(lines) if pat.match(ln)), None)
    if start is None:
        return ""
    indent = len(pat.match(lines[start]).group(1))
    out = [lines[start]]
    for ln in lines[start + 1: start + max_lines]:
        if ln.strip() and (len(ln) - len(ln.lstrip())) <= indent and not ln.lstrip().startswith(("@", "#")):
            break
        out.append(ln)
    return f"# {fpath}\n" + "\n".join(out).rstrip()


def _augment_src(slot_src: str, gate_src: str, diag: str) -> str:
    """Prepend the diagnosed function's REAL body so PLAN/FIX see the exact edit-site source to copy."""
    fb = _focus_body(gate_src, diag)
    if not fb or fb in slot_src:        # skip only if the WHOLE body is already shown (not just its signature)
        return slot_src
    return fb + "\n\n" + slot_src


def _unit(vec) -> list[float]:
    vals = [float(x) for x in (vec or [])]
    norm = math.sqrt(sum(v * v for v in vals))
    if norm <= 1e-9:
        return [0.0 for _ in vals]
    return [v / norm for v in vals]


def _dot(a, b) -> float:
    return float(sum(float(x) * float(y) for x, y in zip(a or [], b or [])))


def _rank_src_chunks(query: str, sources: list[dict], top_k: int = 2) -> list[dict]:
    qtok = _tokset(query)
    ranked = []
    for idx, source in enumerate(sources or []):
        text = f"{source.get('file', '')}\n{source.get('text', '')}"
        stok = _tokset(text)
        overlap = len(qtok & stok)
        lines = max(1, len((source.get("text") or "").splitlines()))
        locality = max(
            (len(qtok & _tokset(line)) for line in (source.get("text") or "").splitlines()),
            default=0,
        )
        score = overlap * 15.0 + locality * 12.0 - min(lines, 200) * 2.0
        ranked.append((score, locality, overlap, lines, idx, source))
    if not ranked:
        return []
    ranked.sort(key=lambda row: (-row[0], -row[1], -row[2], row[3], row[4]))
    return [row[5] for row in ranked[:max(1, top_k)]]


def _rank_diagnose_sources(issue: str, sources: list[dict]) -> list[dict]:
    qtok = _tokset(issue)
    ranked = []
    for idx, source in enumerate(sources or []):
        label = _source_label(source)
        text = _source_text_no_header(source.get("text", ""))
        full = "\n".join(part for part in (source.get("file", ""), label, text) if part)
        stok = _tokset(full)
        overlap = len(qtok & stok)
        locality = max((len(qtok & _tokset(line)) for line in full.splitlines()), default=0)
        span = _source_span(source)
        has_def = 1 if _source_has_definition(source) else 0
        has_exec = 1 if _source_has_executable(source) else 0
        size_bonus = 10 if 8 <= span <= 80 else (4 if span <= 140 else -4)
        score = (
            overlap * 16.0
            + locality * 12.0
            + has_def * 18.0
            + has_exec * 6.0
            + size_bonus
            - max(span - 120, 0) * 0.15
        )
        ranked.append((score, locality, overlap, -has_def, span, idx, source))
    ranked.sort(key=lambda row: (-row[0], -row[1], -row[2], row[3], row[4], row[5]))
    return [row[6] for row in ranked]


def _authoritative_files_text(sources: list[dict]) -> str:
    files = []
    seen = set()
    for source in sources or []:
        fpath = _canon_path(source.get("file", ""))
        if fpath and fpath not in seen:
            files.append(fpath)
            seen.add(fpath)
    if not files:
        return ""
    if len(files) == 1:
        return f"AUTHORITATIVE FILE:\n- {files[0]}"
    rows = "\n".join(f"- {f}" for f in files[:6])
    return f"AUTHORITATIVE FILES (stay inside these unless a hint explicitly names another file):\n{rows}"


def _parse_plan(text: str) -> dict[str, str]:
    file_match = re.search(r"(?mi)^FILE:\s*(.+?)\s*$", text or "")
    # Accept BOTH next-line (`SEARCH:\n<code>`) and INLINE (`SEARCH: <code>`) forms: the model
    # often emits the anchor inline, and the old `^SEARCH:\s*\n` required a newline -> empty search
    # -> plan judged insufficient -> endless DIAGNOSE/PLAN backtrack, never reaching FIX. Stop SEARCH
    # at the next `^CHANGE:` line regardless of which form CHANGE uses.
    search_match = re.search(r"(?mis)^SEARCH:[ \t]*\n?(.*?)(?:^CHANGE:|\Z)", text or "")
    change_match = re.search(r"(?mis)^CHANGE:[ \t]*\n?(.*)\Z", text or "")
    return {
        "file": (file_match.group(1).strip() if file_match else ""),
        "search": (search_match.group(1).strip("\n") if search_match else ""),
        "change": (change_match.group(1).strip() if change_match else ""),
    }


def _best_search_anchor_match(file_body: str, search: str) -> tuple[str, int, int]:
    file_lines = file_body.splitlines()
    groups: list[list[str]] = []
    cur: list[str] = []
    for line in (search or "").splitlines():
        stripped = line.strip()
        if stripped and "..." not in stripped:
            cur.append(line.rstrip())
        elif cur:
            groups.append(cur[:])
            cur = []
    if cur:
        groups.append(cur[:])
    best_lines = 0
    best_chars = 0
    best_start = -1
    for group in groups:
        for offset in range(len(group)):
            probe = group[offset:]
            for start in range(len(file_lines)):
                matched = 0
                while matched < len(probe) and start + matched < len(file_lines):
                    if file_lines[start + matched].strip() != probe[matched].strip():
                        break
                    matched += 1
                if matched <= 0:
                    continue
                chars = sum(len(file_lines[start + i].strip()) for i in range(matched))
                if matched > best_lines or (matched == best_lines and chars > best_chars):
                    best_lines = matched
                    best_chars = chars
                    best_start = start
    if best_start < 0:
        return "", 0, 0
    anchor = "\n".join(file_lines[best_start: best_start + best_lines]).rstrip("\n")
    return anchor, best_lines, best_chars


def _best_search_anchor(file_body: str, search: str) -> str:
    return _best_search_anchor_match(file_body, search)[0]


def _match_exact_span(file_body: str, search: str) -> tuple[int, int]:
    file_lines = file_body.splitlines()
    search_lines = search.splitlines()
    if not search_lines:
        return -1, 0
    for start in range(len(file_lines) - len(search_lines) + 1):
        if file_lines[start:start + len(search_lines)] == search_lines:
            return start, len(search_lines)
    return -1, 0


def _trim_search_anchor(file_body: str, search: str, hint: str, max_lines: int = 8) -> str:
    start, span_len = _match_exact_span(file_body, search)
    if start < 0:
        return search
    hint_toks = _tokset(hint)
    if not hint_toks:
        return search
    file_lines = file_body.splitlines()
    span = file_lines[start:start + span_len]
    has_exec = any(
        re.match(r"^\s*(if|elif|else|return|raise|assert|for|while|with|try|except)\b", line)
        or "=" in line
        for line in span
    )
    if span_len <= max_lines and has_exec:
        return search
    if not has_exec:
        stop = min(len(file_lines), start + max(span_len, max_lines) + 24)
        expanded = file_lines[start:stop]
        first_exec = next(
            (
                idx for idx, line in enumerate(expanded)
                if re.match(r"^\s*(if|elif|else|return|raise|assert|for|while|with|try|except)\b", line)
                or "=" in line
            ),
            0,
        )
        span = expanded[first_exec:] if first_exec < len(expanded) else expanded
    best = search
    best_score = float("-inf")
    for i in range(len(span)):
        for j in range(i + 1, min(len(span), i + max_lines) + 1):
            window_lines = span[i:j]
            wtext = "\n".join(window_lines).rstrip("\n")
            wtoks = _tokset(wtext)
            overlap = len(hint_toks & wtoks)
            if overlap <= 0:
                continue
            code_bonus = sum(
                1
                for line in window_lines
                if re.match(r"^\s*(if|elif|else|return|raise|assert|for|while|with|try|except)\b", line)
                or "=" in line
            )
            prose_penalty = sum(1 for line in window_lines if '"""' in line or "'''" in line)
            score = overlap * 20.0 + code_bonus * 4.0 - len(window_lines) * 1.5 - prose_penalty * 8.0
            if score > best_score:
                best_score = score
                best = wtext
    return best


def _change_from_hint(text: str) -> str:
    raw = (text or "").strip()
    if not raw:
        return ""
    first = raw.splitlines()[0].strip()
    first = re.sub(r"\s+", " ", first)
    if len(first) > 220:
        first = first[:220].rsplit(" ", 1)[0].rstrip(" ,.;:")
    return first


def _looks_codeish_change_line(line: str) -> bool:
    stripped = (line or "").strip()
    if not stripped:
        return False
    if stripped.startswith(("`", "#")):
        return False
    if re.search(r"\br[\"'].*[\"']", stripped):
        return True
    if any(tok in stripped for tok in (" = ", " =r", "==", "return ", "raise ", "yield ", "assert ", ":", "->")):
        return True
    if re.match(r"^[\w.\[\]()'\"\\/+*@<>= -]+$", stripped) and any(ch in stripped for ch in "=()[]'\""):
        return True
    return False


def _is_repeat_instruction(line: str) -> bool:
    return bool(re.search(r"\b(each|every|all|both|matching occurrence|matching site|repeated site)\b", line or "", re.I))


def _extract_change_code_lines(lines: list[str], max_lines: int = 4) -> list[str]:
    out: list[str] = []
    for line in lines:
        stripped = line.strip()
        if _looks_codeish_change_line(stripped):
            out.append(stripped)
        elif out:
            break
        if len(out) >= max_lines:
            break
    return out


def _normalize_change(change: str, hint_text: str = "") -> str:
    raw = (change or "").strip()
    if not raw:
        return _change_from_hint(hint_text)
    lines = [ln.strip() for ln in raw.splitlines() if ln.strip() and not ln.strip().startswith("```")]
    if not lines:
        return _change_from_hint(hint_text)
    action_idx = next(
        (idx for idx, ln in enumerate(lines)
         if ln.lower().startswith(("replace ", "change ", "update ", "use ", "set ", "remove ", "add "))),
        None,
    )
    if action_idx is not None:
        action = lines[action_idx]
        code_lines = _extract_change_code_lines(lines[action_idx + 1:])
        if code_lines:
            extras = [ln for ln in lines[action_idx + 1:] if _is_repeat_instruction(ln)]
            return "\n".join([action] + code_lines + extras[:2])
    if any(_looks_codeish_change_line(ln) for ln in lines):
        compact = []
        for ln in lines[:5]:
            compact.append(ln)
        return "\n".join(compact)
    if "```" in raw or len(lines) > 4:
        return _change_from_hint(raw) or _change_from_hint(hint_text)
    return "\n".join(lines)


def _normalize_search(search: str) -> str:
    lines = [ln.rstrip() for ln in (search or "").splitlines() if not ln.strip().startswith("```")]
    return "\n".join(lines).strip("\n")


def _resolve_src_file(path: str, src_files: dict[str, str]) -> str:
    fpath = _canon_path(path)
    if not fpath:
        return ""
    if fpath in src_files:
        return fpath
    cands = [cand for cand in src_files if _same_file(cand, fpath)]
    return cands[0] if len(cands) == 1 else ""


def _repeat_search_count(file_body: str, search: str) -> int:
    search = (search or "").strip()
    if not search:
        return 0
    return file_body.count(search)


def _augment_repeat_change(change: str, file_body: str, search: str) -> str:
    count = _repeat_search_count(file_body, search)
    if count <= 1:
        return change
    if re.search(r"\b(each|every|all|both|matching occurrence|matching site|repeated site)\b", change, re.I):
        return change
    suffix = f" Apply the same edit to all {count} matching occurrences in this file."
    return (change or "").rstrip() + suffix


def _looks_placeholder_path(path: str) -> bool:
    p = _canon_path(path).lower()
    return (
        not p
        or "path/to/" in p
        or "path/from/" in p
        or p in {"source.py", "file.py"}
        or p.startswith("/path/")
    )


def _repair_plan_to_src(plan_text: str, src: str, hint_text: str = "") -> tuple[str, bool]:
    plan = _parse_plan(plan_text)
    src_files = _split_src_files(src)
    fpath = _resolve_src_file(plan["file"], src_files) or _canon_path(plan["file"])
    search = _normalize_search(plan["search"] or "")
    plan["change"] = _normalize_change(plan["change"], hint_text)
    plan["file"] = fpath
    plan["search"] = search
    if not plan["change"].strip():
        plan["change"] = _change_from_hint(hint_text)
    if len(src_files) == 1:
        only_file = next(iter(src_files))
        if fpath != only_file:
            fpath = only_file
    if _looks_placeholder_path(fpath):
        if len(src_files) == 1:
            fpath = next(iter(src_files))
        else:
            best_file = ""
            best_lines = 0
            best_chars = 0
            for cand, file_body in src_files.items():
                _anchor, lines, chars = _best_search_anchor_match(file_body, search)
                if lines > best_lines or (lines == best_lines and chars > best_chars):
                    best_file = cand
                    best_lines = lines
                    best_chars = chars
            if best_file:
                fpath = best_file
    fpath = _resolve_src_file(fpath, src_files) or fpath
    if not (fpath and plan["change"].strip()):
        normalized = _format_plan({"file": fpath, "search": search, "change": plan["change"]})
        return normalized, normalized != plan_text
    file_body = src_files.get(fpath, "")
    if not file_body:
        return plan_text, False
    compact = [ln for ln in search.splitlines() if ln.strip()]
    if (
        _plan_sufficient(_format_plan({"file": fpath, "search": search, "change": plan["change"]}), src)
        and len(compact) <= 8
        and "\n\n" not in search
        and "..." not in search
        and not _looks_placeholder_path(plan["file"])
    ):
        plan["change"] = _augment_repeat_change(plan["change"], file_body, search)
        normalized = _format_plan(plan)
        return normalized, normalized != plan_text
    repaired = _best_search_anchor(file_body, search)
    if not repaired:
        return plan_text, False
    repaired = _trim_search_anchor(file_body, repaired, hint_text or plan["change"])
    plan["file"] = fpath
    plan["search"] = repaired
    plan["change"] = _augment_repeat_change(plan["change"], file_body, repaired)
    new_text = _format_plan(plan)
    return new_text, new_text != plan_text


def _plan_sufficient(plan_text: str, src: str) -> bool:
    plan = _parse_plan(plan_text)
    src_files = _split_src_files(src)
    fpath = _resolve_src_file(plan["file"], src_files) or _canon_path(plan["file"])
    search = _normalize_search(plan["search"] or "")
    search_lines = [ln for ln in search.splitlines() if ln.strip()]
    requires_code_change = len(search_lines) >= 8
    has_code_change = bool(_extract_change_code_lines((plan["change"] or "").splitlines()))
    return bool(
        fpath
        and not _looks_placeholder_path(fpath)
        and search
        and "..." not in search
        and plan["change"].strip()
        and (has_code_change or not requires_code_change)
        and fpath in src_files
        and search in src_files.get(fpath, "")
    )


def _blocks_match_plan(blocks: list[dict], plan_text: str) -> bool:
    plan = _parse_plan(plan_text)
    fpath = plan["file"]
    search = (plan["search"] or "").strip()
    if not (fpath and search):
        return False
    for b in blocks or []:
        if _same_file(b.get("file", ""), fpath):
            bsearch = (b.get("search") or "").strip()
            if search and (search in bsearch or bsearch in search):
                return True
    return False


def _select_slot_evidence(slot_name: str, ev: list[dict], sources: list[dict]) -> list[dict]:
    if not ev:
        return []
    cap = {"DIAGNOSE": 4, "PLAN": 4, "FIX": 4}.get(slot_name, 2)
    out: list[dict] = []
    seen: set[str] = set()

    def add(source: dict):
        sid = source.get("id")
        if sid and sid not in seen and len(out) < cap:
            out.append(source)
            seen.add(sid)

    if slot_name == "DIAGNOSE":
        ranked = _rank_diagnose_sources("", list(sources or []))
        primary_file = _canon_path(ev[0].get("file", "")) if ev else ""
        for source in ev[:2]:
            add(source)
        if primary_file:
            for source in ranked:
                if _same_file(source.get("file", ""), primary_file):
                    add(source)
        for source in ranked:
            add(source)
        return out

    for source in ev[:2]:
        add(source)
    primary_file = _canon_path(ev[0].get("file", "")) if ev else ""
    if primary_file:
        for source in sources or []:
            if _same_file(source.get("file", ""), primary_file):
                add(source)
    return out[:cap]


def _unmatched(blocks, dest):
    from v5.runtime.sr_withcode import _file_text
    return [b for b in blocks if (b.get("search") or "").strip()
            and (b.get("search") or "").strip() not in _file_text(dest, b.get("file"))]


def _leading_ws(line: str) -> int:
    return len(line) - len(line.lstrip(" "))


def _reindent(text: str, delta: int) -> str:
    """Shift leading-space indentation of every non-blank line by delta (+add / -remove up to)."""
    if delta == 0 or not text:
        return text
    out = []
    for ln in text.splitlines():
        if not ln.strip():
            out.append(ln)
        elif delta > 0:
            out.append(" " * delta + ln)
        else:
            k = 0
            while k < -delta and k < len(ln) and ln[k] == " ":
                k += 1
            out.append(ln[k:])
    return "\n".join(out)


def _repair_sr_to_src(blocks, dest):
    """Goal-#1 (B): snap each SR block's SEARCH to the EXACT source span so `search in file_text`
    holds by construction (the applyable condition). `_best_search_anchor_match` matches the model's
    SEARCH to real source lines WHITESPACE-INSENSITIVELY and returns the verbatim file lines; we adopt
    that as SEARCH and re-indent REPLACE by the leading-ws delta so the edit stays valid. Confidence-
    gated (strong match only) -- never corrupt a low-confidence block. Fixes the near-miss verbatim
    failure (right lines / wrong indentation), NOT truncation or wrong localization."""
    from v5.runtime.sr_withcode import _file_text
    out = []
    for b in blocks:
        search = b.get("search") or ""
        file_text = _file_text(dest, b.get("file"))
        if not search.strip() or not file_text or search.strip() in file_text:
            out.append(b)                                  # empty / no file / already applyable
            continue
        anchor, n_lines, _ = _best_search_anchor_match(file_text, search)
        nonblank = [ln for ln in search.splitlines() if ln.strip()]
        need = max(2, math.ceil(0.6 * len(nonblank)))
        if not anchor or n_lines < need:
            out.append(b)                                  # weak match -> leave untouched
            continue
        s0 = next((ln for ln in search.splitlines() if ln.strip()), "")
        a0 = next((ln for ln in anchor.splitlines() if ln.strip()), "")
        nb = dict(b)
        nb["search"] = anchor
        nb["replace"] = _reindent(b.get("replace") or "", _leading_ws(a0) - _leading_ws(s0))
        out.append(nb)
    return out


def _patch(blocks, dest):                     # applyable -> git diff (the swebench prediction), then restore
    from v5.runtime.search_replace import apply_sr
    if not (bool(blocks) and not _unmatched(blocks, dest)):
        return ""
    _, p = apply_sr(dest, blocks)
    subprocess.run(["git", "-C", dest, "checkout", "--", "."], capture_output=True)
    return p


def fix_user(issue, src, diagnosis="", plan="", hints=None, test_failure=""):
    s = f"ISSUE:\n{issue[:1400]}\n\nRELEVANT SOURCE (the bug is in here):\n{src}\n\n"
    s += _strategy_hint_block(hints or [], title="RELATED STRATEGY HINTS", max_items=2)
    if diagnosis and not plan:
        s += f"ROOT-CAUSE DIAGNOSIS (use it, do not invent a different bug):\n{diagnosis}\n\n"
    if plan:
        s += f"EDIT PLAN (authoritative; realize it literally):\n{_render_fix_plan(plan)}\n\n"
    if test_failure:
        s += (f"PREVIOUS ATTEMPT APPLIED BUT FAILED THE REAL TESTS:\n{test_failure}\n\n"
              "Your previous patch did not fix the bug. Diagnose why from the failure above and change "
              "the fix accordingly; do not just resubmit the same edit.\n\n")
    return (s + "Fix the exact line(s) causing the bug. Output ONLY search/replace blocks. Do NOT output the plan "
            "schema (`FILE`, `SEARCH`, `CHANGE`, `TARGET FILE`, `TARGET SEARCH ANCHOR`, or `EDIT INTENT`). The line before each "
            "SEARCH marker must be the exact file path from the plan. SEARCH must copy the source EXACTLY "
            "(character-for-character); REPLACE must DIFFER. Keep it minimal. If CHANGE gives explicit replacement "
            "code or says the same edit applies to multiple matching sites in the same file, realize that literally "
            "with one block per site or another exact minimal grounding. Do not touch unrelated code.")


def plan_user(issue, src, diagnosis, attempt, hints=None):
    nudge = "" if attempt == 0 else (
        " NOTE: the previous plan/fix missed the exact anchor. Copy the indentation EXACTLY and choose a "
        "smaller anchor from the shown source.")
    file_guard = _authoritative_files_text(
        [{"file": fpath, "text": body} for fpath, body in _split_src_files(src).items()]
    )
    hint_block = _strategy_hint_block(hints or [], title="RELATED STRATEGY HINTS", max_items=3)
    return (
        f"ISSUE:\n{issue[:1400]}\n\n{file_guard}\n\n{hint_block}RELEVANT SOURCE:\n{src}\n\nROOT-CAUSE DIAGNOSIS:\n{diagnosis}\n\n"
        "Plan the edit before patching. Output ONLY this format:\n"
        "FILE: path/from/source.py\nSEARCH:\n<exact existing code copied verbatim from the source>\n"
        "CHANGE:\n<one sentence saying what should change and why>\n"
        "CHANGE may continue on the next lines with the exact replacement line(s) when you know them.\n"
        "Use the diagnosis literally: preserve its file/focus unless the shown source proves the anchor needs to be narrower. "
        "The SEARCH block must be the SMALLEST exact source anchor that pins the buggy location (prefer 1-8 lines, "
        "keep the original indentation, do not quote a whole class/function or docstring when a smaller executable "
        "snippet will do). "
        "For algorithmic or control-flow edits, CHANGE must include the exact replacement line(s), not only prose. "
        "If the same edit repeats in one file, choose one representative exact anchor in source order and mention "
        "the repeated sites in CHANGE. FILE must match a shown source header and stay inside the authoritative file set above. Do not use ellipses, markdown fences, "
        "or placeholder paths; CHANGE must stay concrete and executable."
        f"{nudge}"
    )


def _support_brief(chunks: list[dict], max_items: int = 6, max_preview: int = 180) -> str:
    rows = []
    for chunk in (chunks or [])[:max_items]:
        file = chunk.get("file") or chunk.get("id") or "unknown"
        loc = chunk.get("lineno") or []
        if isinstance(loc, (list, tuple)) and len(loc) >= 2:
            file = f"{file}:{loc[0]}-{loc[1]}"
        label = _source_label(chunk)
        preview = re.sub(r"\s+", " ", _source_text_no_header(chunk.get("text", ""))).strip()
        if len(preview) > max_preview:
            preview = preview[:max_preview].rsplit(" ", 1)[0].rstrip(" ,.;:") + "..."
        prefix = f"- {file}"
        if label:
            prefix += f" [{label}]"
        rows.append(f"{prefix} :: {preview}")
    return "\n".join(rows)


def _strategy_hint_block(hints: list[dict], title: str = "STRATEGY HINTS", max_items: int = 3) -> str:
    if not hints:
        return ""
    body = _support_brief(hints, max_items=max_items, max_preview=220)
    return f"{title} (approach-level only; adapt to the shown source, do not copy literally):\n{body}\n\n"


def diag_user(issue, src, attempt):
    nudge = "" if attempt == 0 else (
        " NOTE: a previous diagnosis led to an UNAPPLYABLE or no-op fix. Be more specific — name the "
        "EXACT function and the EXACT line/token to change, copied verbatim from the source above.")
    files = _authoritative_files_text([{"file": fpath, "text": body} for fpath, body in _split_src_files(src).items()])
    return (
        f"ISSUE:\n{issue[:1400]}\n\n{files}\n\nSOURCE:\n{src}\n\n"
        "Output ONLY this format:\n"
        "FILE: path/from/source.py\n"
        "FOCUS: exact function/class/helper name or a short exact anchor from the source\n"
        "WHY: one sentence root cause\n"
        "CHANGE: one sentence describing the exact change\n"
        "FILE must match a shown source header. FOCUS must stay inside the shown source and name something real from it. "
        f"Do not mention any other file families.{nudge}"
    )


def diag_user_injected(issue, sources, attempt):
    nudge = "" if attempt == 0 else (
        " NOTE: a previous diagnosis led to an UNAPPLYABLE or no-op fix. Be more specific — name the "
        "EXACT function and the EXACT line/token to change.")
    ranked = _rank_diagnose_sources(issue, sources)
    file_guard = _authoritative_files_text(ranked)
    brief = _support_brief(ranked)
    return (
        f"ISSUE:\n{issue[:1400]}\n\n"
        f"{file_guard}\n\n"
        f"RETRIEVED SUPPORT HINTS (higher-priority hints first; full code evidence is supplied through the graph channel):\n{brief}\n\n"
        "Output ONLY this format:\n"
        "FILE: one authoritative file from the hints above\n"
        "FOCUS: exact function/class/helper name from the hints\n"
        "WHY: one sentence root cause\n"
        "CHANGE: one sentence describing the exact change\n"
        "If all hints point to one file, FILE must be that file. Do not mention any other file family "
        f"unless one of the hints explicitly names it.{nudge}"
    )


def slot_solve(issue, src, diagnose_fn, plan_fn, fix_fn, max_steps=8, log=None, capture_evidence=False):
    """The SHARED slot-graph: DIAGNOSE -> PLAN -> FIX.
      diagnose_fn(issue, src, attempt)              -> diagnosis text
      plan_fn(issue, src, diagnosis, attempt)       -> FILE/SEARCH/CHANGE plan text
      fix_fn(issue, src, diagnosis, plan)           -> (APPLYABLE patch text, parsed SR blocks)
    Returns (patch, trace, fixpoint, steps). Same engine for the 4B run and the selftest."""
    attempts = {"DIAGNOSE": 0, "PLAN": 0}
    trace = {"diagnoses": [], "plans": [], "fix_attempts": [], "retrievals": []}
    fix_meta = {"blocks": [], "plan": ""}
    sources = []
    if isinstance(src, dict):
        full_src = src.get("full", "")
        sources = list(src.get("sources") or [])
    else:
        full_src = src
    # GATE/snap plans against the REAL repo files (full content), not the truncated shown
    # window: a correct verbatim anchor often lives in a function outside the top-N slot_src
    # chunks, so checking `search in full_src` wrongly rejects valid plans. Falls back to
    # full_src when no real checkout is provided (e.g. the no-model selftest).
    gate_src = (src.get("gate") if isinstance(src, dict) else "") or full_src

    last_query = {"text": ""}

    def retr(q, kind):
        last_query["text"] = q
        if sources:
            return _rank_src_chunks(q, sources, top_k=2)
        return [{"id": "src", "text": full_src}]              # fallback: one coarse source blob

    def filler(slot, ev, pool):
        slot_ev = ev
        slot_ev = _select_slot_evidence(slot.name, ev, sources)
        slot_src = _compose_src(slot_ev) or full_src
        trace["retrievals"].append({
            "slot": slot.name,
            "query": last_query["text"],
            "ids": [e.get("id") for e in slot_ev],
            "files": [e.get("file") for e in slot_ev],
            "evidence": _support_digest(slot_ev) if capture_evidence else [],
        })
        if slot.name == "DIAGNOSE":
            n = attempts["DIAGNOSE"]; attempts["DIAGNOSE"] = n + 1
            d = diagnose_fn(issue, slot_src, n)
            trace["diagnoses"].append(d)
            return d
        if slot.name == "PLAN":
            diag = pool.get("DIAGNOSE")
            n = attempts["PLAN"]; attempts["PLAN"] = n + 1
            psrc = _augment_src(slot_src, gate_src, diag)   # show the real edit-site function body
            plan = ""
            for retry in range(2):
                raw_plan = plan_fn(issue, psrc, diag, n + retry)
                plan, _ = _repair_plan_to_src(raw_plan, gate_src, diag)
                ok = _plan_sufficient(plan, gate_src)
                if os.environ.get("SWE_PLAN_DEBUG"):
                    pp = _parse_plan(plan); gsf = _split_src_files(gate_src)
                    fb = gsf.get(_resolve_src_file(pp["file"], gsf) or pp["file"], "")
                    anc, nl, _c = _best_search_anchor_match(fb, pp["search"] or "")
                    print(f"[PLAN-DEBUG retry{retry}] sufficient={ok} file={pp['file']!r} "
                          f"file_in_gate={bool(fb)} search_in_src={(pp['search'] or '') in fb}\n"
                          f"  SEARCH(repr)={pp['search']!r}\n"
                          f"  best_anchor(lines={nl})={anc!r}", flush=True)
                if ok:
                    break
            trace["plans"].append(plan)
            return plan
        diag = pool.get("DIAGNOSE")
        plan = pool.get("PLAN")
        fsrc = _augment_src(slot_src, gate_src, diag)        # show the real edit-site function body
        patch, blocks, raw_fix = "", [], ""
        for retry in range(2):
            out = fix_fn(issue, fsrc, diag, plan)            # "" if unapplyable/no-op -> INSUFFICIENT
            if isinstance(out, tuple) and len(out) == 3:
                patch, blocks, raw_fix = out
            else:
                patch, blocks = out
                raw_fix = ""
            fix_meta["blocks"] = list(blocks or [])
            fix_meta["plan"] = plan
            trace["fix_attempts"].append({
                "applyable": bool(patch),
                "anchored": _blocks_match_plan(fix_meta["blocks"], plan),
                "diag_used": diag[:160],
                "n_blocks": len(blocks or []),
                "files": [b.get("file", "") for b in (blocks or [])],
                "raw": raw_fix[:1600],
            })
            if patch and _blocks_match_plan(fix_meta["blocks"], plan):
                break
            # Preserve graph-level backtracking for wrong-scope applyable fixes; only use the
            # local retry to recover from empty/unparseable emissions.
            if patch or retry >= 1:
                break
        return patch

    specs = [
        SlotSpec("DIAGNOSE", [], "src", "ASSERT", query=lambda p: issue[:240], revise="rederive"),
        SlotSpec("PLAN", ["DIAGNOSE"], "src", "ASSERT",
                 query=lambda p: f"{issue[:220]}\n{(p['DIAGNOSE'].value or '')[:220]}",
                 revise="rederive",
                 sufficient=lambda slot, pool: _plan_sufficient(slot.value, gate_src)),
        SlotSpec("FIX", ["DIAGNOSE", "PLAN"], "src", "TRANSFORM",
                 query=lambda p: f"{(p['PLAN'].value or '')[:260]}\n{(p['DIAGNOSE'].value or '')[:180]}",
                 sufficient=lambda slot, pool: bool(slot.value)
                 and _blocks_match_plan(fix_meta["blocks"], fix_meta["plan"])),
    ]
    sg = SlotGraph(specs)
    pool = Pool(specs, context={"issue": issue, "src": full_src})
    ok, steps = sg.solve(pool, retr, filler, max_steps=max_steps, log=log)
    return pool.slots["FIX"].value, trace, ok, steps


def kv_solve(model, tok, dev, issue, gate_src, dest, verifier, task, a, vsecs=None):
    """Phase-A latent solve (gen-minimize): prime [issue + source] ONCE into a shared KV cache, then
    chain DIAGNOSE -> PLAN -> FIX as TERSE turns that build on the cache — the heavy context and prior
    stages are never re-encoded (the latent state-carry). Linear (backtrack deferred). Reuses the same
    parse/repair/snap/patch/verify as slot_solve. Returns (patch, trace)."""
    from v5.runtime.prefix_session import ChatKV
    from v5.runtime.search_replace import parse_sr
    sysmsg = ("You are a precise Python debugging assistant. Use only the shown source. Work step by "
              "step; be terse; never repeat an earlier step.")
    ctx = (f"ISSUE:\n{issue[:1400]}\n\nSOURCE (the bug is in here):\n{gate_src}\n\n"
           "STEP 1 — DIAGNOSE. Output ONLY:\nFILE: <path from the source>\n"
           "FOCUS: <exact function/class name from the source>\nWHY: <one-sentence root cause>")
    chat = ChatKV(model, tok, dev, sysmsg, ctx)
    diag = chat.step("", 200)
    plan_raw = chat.step(
        "STEP 2 — PLAN (use the diagnosis above; do NOT repeat it). Output ONLY:\n"
        "FILE: <path>\nSEARCH:\n<the exact existing code copied verbatim from the source>\n"
        "CHANGE:\n<the exact replacement line(s)>", 240)
    plan = _repair_plan_to_src(plan_raw, gate_src, diag)[0]
    trace = {"diagnoses": [diag], "plans": [plan], "fix_attempts": [], "retrievals": [], "kv": True}
    test_failure = ""; patch = ""; blocks = []; g = ""
    for attempt in range(max(1, a.test_feedback_retries if a.test_feedback else 1)):
        if not test_failure:
            fix_instr = ("STEP 3 — FIX (realize the PLAN above). Output ONLY search/replace blocks, NO prose:\n"
                         "path/from/source.py\n<<<<<<< SEARCH\n<exact existing code>\n=======\n<replacement>\n"
                         ">>>>>>> REPLACE\nSEARCH must copy the source character-for-character.")
        else:
            fix_instr = ("STEP 3 (RETRY) — the previous patch APPLIED but FAILED the tests:\n"
                         f"{test_failure[:1200]}\nDiagnose why from that and CHANGE the fix; do not resubmit the "
                         "same edit. Output ONLY the search/replace block(s).")
        g = chat.step(fix_instr, a.max_new)
        blocks = parse_sr(g)
        if not blocks:                                     # FIX often continues the PLAN FILE/SEARCH/CHANGE
            ms = re.search(r"(?ms)^[ \t]*SEARCH:[ \t]*\n(.*?)\n[ \t]*CHANGE:", g)   # format (cached) -> SR block.
            mc = re.search(r"(?ms)^[ \t]*CHANGE:[ \t]*\n(.*)\Z", g)                 # keep indentation (no strip)
            mf = re.search(r"(?mi)^[ \t]*FILE:[ \t]*(.+)$", g)
            if ms and mc and mf:                            # --sr-snap then snaps SEARCH to exact source + re-indents
                blocks = [{"file": mf.group(1).strip(), "search": ms.group(1).rstrip(),
                           "replace": mc.group(1).rstrip()}]
        if a.sr_snap:
            blocks = _repair_sr_to_src(blocks, str(dest))
        patch = _patch(blocks, str(dest))
        trace["fix_attempts"].append({"applyable": bool(patch),
                                      "anchored": _blocks_match_plan(blocks, plan),
                                      "diag_used": diag[:160], "n_blocks": len(blocks or []),
                                      "files": [b.get("file", "") for b in (blocks or [])],
                                      "raw": g[:1200]})
        ready = bool(a.test_feedback and verifier is not None and patch and _blocks_match_plan(blocks, plan))
        if not ready:
            break
        _vt = time.time()
        resolved, feedback = verifier.verify_patch_feedback(task, patch, tag=f"kv_{task.get('iid','')}")
        if vsecs is not None:
            vsecs[0] += time.time() - _vt
        trace["fix_attempts"][-1]["resolved"] = resolved
        if resolved:
            break
        test_failure = feedback
    return patch, trace


def _parse_instance_ids(text: str) -> list[str]:
    if not (text or "").strip():
        return []
    out = []
    seen = set()
    for part in re.split(r"[\s,]+", text.strip()):
        iid = part.strip()
        if iid and iid not in seen:
            out.append(iid)
            seen.add(iid)
    return out


def _support_digest(sources: list[dict], max_chars: int = 240) -> list[dict]:
    out = []
    for source in sources or []:
        text = (_source_text_no_header(source.get("text", "")) or "").strip()
        compact = re.sub(r"\s+", " ", text)
        if len(compact) > max_chars:
            compact = compact[:max_chars].rsplit(" ", 1)[0] + "..."
        out.append({
            "id": source.get("id"),
            "file": source.get("file"),
            "lineno": source.get("lineno"),
            "label": _source_label(source),
            "text": compact,
        })
    return out


def _strategy_sources(
    iid: str,
    issue: str,
    traces: dict,
    strategy_items_by_inst: dict,
    issue_vecs: dict[str, list[float]],
    strategy_source: str,
    strat_topk: int,
    strat_hints: int,
) -> list[dict]:
    if strategy_source == "off" or not strategy_items_by_inst:
        return []
    chosen: list[dict] = []
    if strategy_source == "own":
        chosen = list(strategy_items_by_inst.get(iid, []))
    else:
        qv = issue_vecs.get(iid)
        if not qv:
            return []
        scored_instances: list[tuple[float, str]] = []
        for sid, items in strategy_items_by_inst.items():
            if sid == iid or sid not in traces:
                continue
            best = max((_dot(qv, item.get("vec", [])) for item in items), default=float("-inf"))
            scored_instances.append((best, sid))
        scored_instances.sort(reverse=True)
        neighbors = [sid for _score, sid in scored_instances[:max(1, strat_topk)]]
        for sid in neighbors[:max(1, strat_topk)]:
            ranked = sorted(
                strategy_items_by_inst.get(sid, []),
                key=lambda item: _dot(qv, item.get("vec", [])),
                reverse=True,
            )
            chosen.extend(ranked)
    out = []
    seen_texts = set()
    for item in chosen:
        clean = (item.get("text") or "").strip()
        if not clean:
            continue
        sig = re.sub(r"\s+", " ", clean.lower())
        if sig in seen_texts:
            continue
        seen_texts.add(sig)
        out.append({
            "id": item.get("id"),
            "file": "",
            "lineno": [],
            "label": item.get("node_type", ""),
            "node_type": item.get("node_type", ""),
            "text": clean,
        })
        if len(out) >= max(0, strat_hints):
            break
    return out


def _infer_model_hidden_size(model) -> int:
    for name in ("hidden_size", "d_model", "n_embd"):
        val = getattr(getattr(model, "config", None), name, None)
        if isinstance(val, int) and val > 0:
            return val
    emb = model.get_input_embeddings()
    if emb is not None and hasattr(emb, "weight"):
        return int(emb.weight.shape[1])
    raise ValueError("could not infer LM hidden size from model config or embeddings")


def _infer_adapter_lm_hidden(adapter_state: dict) -> int:
    for key in (
        "planning_block.aux.node.h_proj.weight",
        "evidence_block.aux.node.h_proj.weight",
        "aux_heads.node.h_proj.weight",
    ):
        if key in adapter_state:
            return int(adapter_state[key].shape[1])
    raise ValueError("could not infer LM hidden size from adapter checkpoint")


class _InjectedRuntime:
    def __init__(self, model_name: str, adapter_ckpt: str, gnn_ckpt: str, trust_remote_code: bool):
        import torch
        from transformers import AutoTokenizer
        from v5.adapter import GraphAttentionInjector
        from v5.cross_attention import V5AttentionAdapter
        from v5.gnn_encoder import RGCNEncoder
        from v5.goal_encoder import GoalEncoder
        from v5.graph_grower.constrained_decode import make_inpatch_processor
        from v5.lm_loader import load_frozen_lm
        from v5.training.providers import RealEmbedder
        from v5.training.stage4_generate import _gen, _stub_graph

        self._torch = torch
        self._make_inpatch_processor = make_inpatch_processor
        self._gen_fn = _gen
        self._stub_graph = _stub_graph
        self.model = load_frozen_lm(model_name)
        self.model.eval()
        self.tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=trust_remote_code)
        self.device = next(self.model.parameters()).device
        self.lm_hidden_dim = _infer_model_hidden_size(self.model)
        state = torch.load(adapter_ckpt, map_location=self.device)
        adapter_dim = _infer_adapter_lm_hidden(state)
        if adapter_dim != self.lm_hidden_dim:
            raise ValueError(
                f"adapter/model hidden mismatch: adapter expects {adapter_dim}, model has {self.lm_hidden_dim}"
            )
        self.embedder = RealEmbedder(self.device)
        self.gnn = RGCNEncoder().to(self.device)
        if gnn_ckpt:
            self.gnn.load_state_dict(torch.load(gnn_ckpt, map_location=self.device))
        self.gnn.eval()
        self.goal_enc = GoalEncoder().to(self.device).eval()
        self.adapter = V5AttentionAdapter(r_plan=3, r_evidence=4, lm_hidden_dim=self.lm_hidden_dim).to(self.device)
        self.adapter.load_state_dict(state)
        self.adapter.eval()
        self.injector = GraphAttentionInjector(self.adapter, self.gnn, self.goal_enc, device=self.device)
        self.task_frame = {"task_family": "code_fix", "required_slots": []}

    def prepare_session(self, support_ids: list[str], support_sources: list[dict], meta: dict):
        node_ids = support_ids[:24]
        node_texts = {}
        source_by_id = {s.get("id"): s.get("text", "") for s in support_sources or []}
        for sid in node_ids:
            text = source_by_id.get(sid) or meta[sid].get("text", "")
            if text:
                node_texts[sid] = text
        if not node_texts:
            return
        text_emb = self.embedder.embed_nodes(node_texts)
        stub = self._stub_graph(node_ids, node_texts, {sid: "fact" for sid in node_ids})
        if not getattr(self, "_structure_warned", False):
            # Phase-0b telemetry: prove empirically what STRUCTURE reaches the GNN. Today it is a
            # stub -- all node_type="fact", edges=[] -- so the "structure-aware" injection is hollow.
            ntypes = {getattr(n, "node_type", "?") for n in stub.nodes.values()}
            print(f"[inject-structure] nodes={len(stub.nodes)} node_types={sorted(ntypes)} "
                  f"edges={len(stub.edges)}  (stub: structure discarded before the GNN)", flush=True)
            self._structure_warned = True
        self.injector.prepare_session(
            stub,
            node_ids,
            text_emb,
            self.task_frame,
            r_plan=3,
            r_evidence=4,
        )

    def generate(self, system: str, user: str, ntok: int, constrain_symbols: list[str] | None = None) -> str:
        msgs = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        text = self._gen_fn(
            self.model,
            self.tok,
            msgs,
            self.device,
            self.injector,
            True,
            ntok,
            constrain_symbols=constrain_symbols or None,
        )
        return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


# ── no-model wiring proof: the engine MUST reject an applyable-but-misaligned fix, then re-plan and recover ──
def _selftest():
    print("swe_slot --selftest: proving the SLOT path runs SlotGraph.solve (no model).\n")
    ranked = _rank_src_chunks(
        "bitwise mask operand none propagation",
        [
            {"id": "broad", "file": "x.py", "text": "# x.py\nclass Big:\n    mask = True\n    propagation = True\n"},
            {"id": "tight", "file": "x.py", "text": "# x.py\ndef _arithmetic_mask(self, operand):\n    return operand.mask\n"},
        ],
        top_k=1,
    )
    retrieval_ok = bool(ranked) and ranked[0]["id"] == "tight"
    print(f"   slot retrieval (narrowing) : {'PASS' if retrieval_ok else 'FAIL'}")
    plan_src = (
        "# x.py\n"
        "def f():\n"
        "    if cond:\n"
        "        return bad()\n"
        "\n"
        "# y.py\n"
        "class Alpha:\n"
        "    value = 1\n"
        "\n"
        "class Beta:\n"
        "    value = 1\n"
    )
    raw_indent = (
        "FILE: x.py\nSEARCH:\n        if cond:\n            return bad()\nCHANGE:\n"
        "Use the fixed return.\n"
    )
    fixed_indent, repaired_indent = _repair_plan_to_src(raw_indent, plan_src)
    indent_ok = repaired_indent and _plan_sufficient(fixed_indent, plan_src)
    raw_order = (
        "FILE: y.py\nSEARCH:\nclass Beta:\n    value = 1\n\nclass Alpha:\n    value = 1\nCHANGE:\n"
        "Update both values.\n"
    )
    fixed_order, repaired_order = _repair_plan_to_src(raw_order, plan_src)
    order_ok = repaired_order and _plan_sufficient(fixed_order, plan_src)
    raw_trunc = (
        "FILE: x.py\nSEARCH:\ndef f():\n    if cond:\n        return bad(\nCHANGE:\n"
        "Close the call and fix the value.\n"
    )
    fixed_trunc, repaired_trunc = _repair_plan_to_src(raw_trunc, plan_src)
    trunc_ok = _plan_sufficient(fixed_trunc, plan_src)
    raw_change = (
        "Replace the regex with:\n"
        "regex = r'\\A[\\w.@+-]+\\Z'\n"
        "Apply the same edit to both validators.\n"
    )
    norm_change = _normalize_change(raw_change, "")
    change_ok = "regex =" in norm_change and "both validators" in norm_change
    multi_sources = [
        {"id": "u", "file": "django/contrib/auth/validators.py", "text": "# django/contrib/auth/validators.py\nclass UnicodeUsernameValidator:\n    regex = r'^[\\w.@+-]+$'"},
        {"id": "a", "file": "django/contrib/auth/validators.py", "text": "# django/contrib/auth/validators.py\nclass ASCIIUsernameValidator:\n    regex = r'^[\\w.@+-]+$'"},
        {"id": "other", "file": "django/forms/forms.py", "text": "# django/forms/forms.py\nclass Form:\n    pass"},
    ]
    plan_ev = _select_slot_evidence("PLAN", [multi_sources[0]], multi_sources)
    evidence_ok = len(plan_ev) >= 2 and all(_same_file(s["file"], "django/contrib/auth/validators.py") for s in plan_ev[:2])
    repeat_plan = (
        "FILE: django/contrib/auth/validators.py\n"
        "SEARCH:\nregex = r'^[\\w.@+-]+$'\n"
        "CHANGE:\nReplace the regex with:\nregex = r'\\A[\\w.@+-]+\\Z'\n"
    )
    fixed_repeat, repaired_repeat = _repair_plan_to_src(
        repeat_plan,
        "# django/contrib/auth/validators.py\nclass ASCIIUsernameValidator:\n    regex = r'^[\\w.@+-]+$'\n\nclass UnicodeUsernameValidator:\n    regex = r'^[\\w.@+-]+$'\n",
        "Replace the regex with the anchored form.",
    )
    repeat_ok = repaired_repeat and "matching occurrences" in fixed_repeat
    print(f"   plan repair (indent drift) : {'PASS' if indent_ok else 'FAIL'}")
    print(f"   plan repair (order drift)  : {'PASS' if order_ok else 'FAIL'}")
    print(f"   plan repair (truncation)   : {'PASS' if trunc_ok else 'FAIL'}"
          f"{' (snapped)' if repaired_trunc else ' (already sufficient)'}")
    print(f"   change preservation        : {'PASS' if change_ok else 'FAIL'}")
    print(f"   same-file support keep     : {'PASS' if evidence_ok else 'FAIL'}")
    print(f"   repeat-site hinting        : {'PASS' if repeat_ok else 'FAIL'}")

    src = "# x.py\ndef check(x, y):\n    if y < 5:\n        return y < 5\n    return x < 5\n"
    def diagnose_fn(issue, src, attempt):
        return "PRECISE: change `return x < 5` to `return x <= 5` in x.py"
    def plan_fn(issue, src, diagnosis, attempt):
        raw = "FILE: x.py\nSEARCH:\nreturn x < 5\nCHANGE:\nChange `<` to `<=` in the return line.\n"
        return _repair_plan_to_src(raw, src)[0]
    fix_attempts = {"n": 0}
    def fix_fn(issue, src, diagnosis, plan):
        fix_attempts["n"] += 1
        if fix_attempts["n"] == 1:
            blocks = [{"file": "x.py", "search": "return y < 5", "replace": "return y <= 5"}]
            return "diff --git a/x.py b/x.py\n+wrong-scope\n", blocks
        blocks = [{"file": "x.py", "search": "return x < 5", "replace": "return x <= 5"}]
        return "diff --git a/x.py b/x.py\n+fixed\n", blocks
    log = []
    patch, trace, ok, steps = slot_solve("issue", src, diagnose_fn, plan_fn, fix_fn, log=log)
    for row in log:
        print("   ", row)
    print(f"\n   diagnoses          : {trace['diagnoses']}")
    print(f"   plans              : {trace['plans']}")
    print(f"   fix attempts       : {[(a['applyable'], a['anchored']) for a in trace['fix_attempts']]}")
    print(f"   fixpoint={ok} steps={steps}  final patch applyable={bool(patch)}")
    backtracked = any(r[0] == "BACKTRACK" for r in log)
    replanned = len(trace["plans"]) >= 2
    anchored = trace["fix_attempts"][-1]["anchored"] if trace["fix_attempts"] else False
    rejected_wrong_scope = any(a["applyable"] and not a["anchored"] for a in trace["fix_attempts"][:-1])
    ok_wired = (
        retrieval_ok and indent_ok and order_ok and trunc_ok and change_ok and evidence_ok and repeat_ok
        and bool(patch) and ok and backtracked and replanned
        and anchored and rejected_wrong_scope
    )
    print(f"\n   WIRING PROOF: backtrack-fired={backtracked}  re-planned={replanned}  "
          f"rejected-applyable-wrong-scope={rejected_wrong_scope}  recovered-anchored={anchored}"
          f"  -> {'PASS' if ok_wired else 'FAIL'}")
    print("   (attempt-1 fix was applyable but ignored the planned anchor -> INSUFFICIENT -> backtrack")
    print("    to PLAN -> attempt-2 follows the quoted source anchor -> fixpoint. The engine enforced scope.)")
    return ok_wired


def _exact_resolve_rate(verifier: SWEExactVerifier | None, name: str,
                        task_patches: list[tuple[dict, str]], scored: int):
    if verifier is None or scored <= 0:
        return None
    res = verifier.verify_task_batch_unique(task_patches, tag=name)
    resolved = sum(1 for task, _patch in task_patches if res.get(task["iid"], False))
    emitted = len(task_patches)
    return resolved / scored, emitted / scored, resolved, emitted


def _slug(text: str) -> str:
    keep = [c.lower() if c.isalnum() else "_" for c in (text or "session")]
    s = "".join(keep).strip("_")
    return s[:80] or "session"


def _prepare_outputs(args):
    if args.session_out_dir:
        tag = _slug(args.session_name or f"swe_slot_{args.dataset}_{args.split}_{time.strftime('%Y%m%d_%H%M%S')}")
        bundle = Path(args.session_out_dir) / tag
        bundle.mkdir(parents=True, exist_ok=True)
        return {
            "name": tag,
            "bundle": bundle,
            "dump": bundle / "dump.txt",
            "oneshot": bundle / "oneshot.jsonl",
            "slot": bundle / "slot.jsonl",
            "summary": bundle / "summary.json",
        }
    Path(args.dump).parent.mkdir(parents=True, exist_ok=True)
    Path("artifacts").mkdir(parents=True, exist_ok=True)
    return {
        "name": "",
        "bundle": None,
        "dump": Path(args.dump),
        "oneshot": Path("artifacts/swe_oneshot_preds.jsonl"),
        "slot": Path("artifacts/swe_slot_preds.jsonl"),
        "summary": None,
    }


def _verify_run_ids(outputs, dataset: str, split: str) -> tuple[str, str]:
    if outputs["name"]:
        base = outputs["name"]
    else:
        base = _slug(f"swe_slot_{dataset}_{split}")
    return f"{base}_oneshot", f"{base}_slot"


def _build_support_sources(dest: Path, support: list[str], meta: dict, src_lines: int) -> list[dict]:
    from v5.runtime.sr_withcode import read_body
    out = []
    for sid in support:
        m = meta[sid]
        body = read_body(str(dest), m["file"], m["lineno"], src_lines)
        if not body:
            continue
        label = _extract_symbol_label(m.get("text", "")) or _extract_symbol_label(body)
        out.append({
            "id": sid,
            "file": m["file"],
            "lineno": m["lineno"],
            "label": label,
            "text": f"# {m['file']}\n{body}",
        })
    return out


def _full_files_src(dest: Path, support: list[str], meta: dict) -> str:
    """Header-formatted FULL contents of the real repo files referenced by the support.
    Used to GATE/snap plans against ground-truth source (not the truncated shown window),
    so a correct verbatim anchor in a function outside the top-N slot_src chunks still passes."""
    seen: set[str] = set()
    parts: list[str] = []
    for sid in support:
        f = meta.get(sid, {}).get("file")
        if not f or f in seen:
            continue
        seen.add(f)
        try:
            body = (Path(dest) / f).read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        if body.strip():
            parts.append(f"# {_canon_path(f)}\n{body}")
    return "\n\n".join(parts)


def _ast_funcs(src_text: str):
    """[(name, class, start, end, calls_set)] for every function in the source (real AST)."""
    import ast
    out = []
    try:
        tree = ast.parse(src_text)
    except Exception:
        return out

    class _V(ast.NodeVisitor):
        def __init__(s):
            s.cls = None

        def visit_ClassDef(s, n):
            prev = s.cls; s.cls = n.name
            for c in n.body:
                s.visit(c)
            s.cls = prev

        def _fn(s, n):
            calls = set()
            for x in ast.walk(n):
                if isinstance(x, ast.Call):
                    nm = getattr(x.func, "attr", None) or getattr(x.func, "id", None)
                    if nm:
                        calls.add(nm)
            out.append((n.name, s.cls, n.lineno, getattr(n, "end_lineno", n.lineno), calls))

        def visit_FunctionDef(s, n):
            s._fn(n)

        def visit_AsyncFunctionDef(s, n):
            s._fn(n)

    _V().visit(tree)
    return out


def _traverse_support(dest, support, meta, max_add: int = 6):
    """Expand the flat seed support with structurally-linked sites (same_class / calls), read from
    the REAL checked-out source via AST. The graph is `contains`-only, so co-edited functions that
    aren't textually similar to the issue are invisible to flat retrieval — this follows the edges
    an array/flat-index can't. Returns support + up to max_add structural neighbors (deduped)."""
    import re as _re
    seed_by_file: dict = {}
    for sid in support:
        m = meta.get(sid) or {}
        f = m.get("file"); mm = _re.search(r'(?:async def|def|class)\s+(\w+)', m.get("text", "") or "")
        if f and mm:
            seed_by_file.setdefault(f, set()).add(mm.group(1))
    if not seed_by_file:
        return support
    sym_by: dict = {}
    for sid, m in meta.items():
        f = m.get("file"); mm = _re.search(r'(?:async def|def|class)\s+(\w+)', m.get("text", "") or "")
        if f and mm:
            sym_by.setdefault((f, mm.group(1)), sid)
    add = []
    for f, seeds in seed_by_file.items():
        fp = Path(dest) / f
        if not fp.exists():
            continue
        funcs = _ast_funcs(fp.read_text(encoding="utf-8", errors="ignore"))
        seed_funcs = [fn for fn in funcs if fn[0] in seeds]
        seed_classes = {fn[1] for fn in seed_funcs if fn[1]}
        for name, cls, s, e, calls in funcs:
            if name in seeds:
                continue
            linked = (cls is not None and cls in seed_classes)                       # same_class
            if not linked:
                linked = any(name in sf[4] for sf in seed_funcs) or any(sd in calls for sd in seeds)  # calls (either dir)
            if linked:
                sid = sym_by.get((f, name))
                if sid and sid not in support and sid not in add:
                    add.append(sid)
    return support + add[:max_add]


def _apply_smoke_overrides(args):
    if not args.smoke:
        return args
    orig_n = args.n_eval
    orig_gold = args.verify_gold_sanity
    args.n_eval = min(args.n_eval, max(1, args.smoke_n_eval))
    if args.exact_verify:
        args.verify_gold_sanity = min(args.verify_gold_sanity, args.n_eval, max(1, args.smoke_gold_sanity))
    print(f"[SMOKE] preflight enabled: n_eval {orig_n} -> {args.n_eval}", flush=True)
    if args.exact_verify:
        print(f"[SMOKE] verifier preflight: gold_sanity {orig_gold} -> {args.verify_gold_sanity}", flush=True)
    return args


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true", help="no-model proof the SLOT path uses the engine")
    ap.add_argument("--smoke", action="store_true",
                    help="cheap preflight: run a tiny generation slice before the full expensive session")
    ap.add_argument("--smoke-n-eval", type=int, default=2,
                    help="when --smoke is set, cap n_eval to this many instances")
    ap.add_argument("--smoke-gold-sanity", type=int, default=2,
                    help="when --smoke and --exact-verify are both set, cap gold-sanity to this many instances")
    ap.add_argument("--model", default="Qwen/Qwen3.5-4B")
    ap.add_argument("--injector-mode", choices=["off", "slot", "all"], default="off",
                    help="use the graph cross-attention injector for slot generations (`slot`) or both one-shot and slot (`all`)")
    ap.add_argument("--adapter-ckpt", default="artifacts/stage_cache/adapter_code_s3.pt",
                    help="cross-attention adapter checkpoint for injector mode")
    ap.add_argument("--gnn-ckpt", default="",
                    help="optional trained GNN checkpoint for injector mode; else random-frozen GNN")
    ap.add_argument("--inject-fix", action="store_true",
                    help="CHEAP-MEASURE: inject graph evidence at the FIX stage (off by default; FIX is "
                         "uninjected today). Builds the injector even with --injector-mode off, and injects "
                         "ONLY at FIX so off-vs-inject isolates whether cross-attn changes code emission.")
    ap.add_argument("--sr-snap", action="store_true",
                    help="goal-#1 (B): snap each FIX SEARCH block to its EXACT source span before "
                         "applying (verbatim-anchor repair), so a near-miss SEARCH (right lines / wrong "
                         "whitespace) still applies. Off = today's raw parse_sr->_patch (for A/B).")
    ap.add_argument("--fix-constrain", action="store_true",
                    help="constrain one-shot decoding toward graph symbols from the retrieved support")
    ap.add_argument("--slot-fix-constrain", action="store_true",
                    help="experimentally constrain slot FIX decoding too; keep off unless a smoke shows it helps")
    ap.add_argument("--traces", default="data/swe/grounded_traces.jsonl")
    ap.add_argument("--nodes", default="artifacts/graph_growth/swe_code_candidates.jsonl")
    ap.add_argument("--strategy-nodes", nargs="+",
                    default=["artifacts/graph_growth/swe_strategy_candidates_clean.jsonl"],
                    help="planning-pool strategy/reasoning nodes for PLAN/FIX hints")
    ap.add_argument("--strategy-source", choices=["off", "retrieved", "own"], default="retrieved",
                    help="off = no strategy hints; retrieved = nearest other-task strategy hints; own = upper-bound debug")
    ap.add_argument("--strat-topk", type=int, default=2,
                    help="for strategy-source=retrieved, pull hints from this many nearest other tasks")
    ap.add_argument("--strat-hints", type=int, default=4,
                    help="cap strategy hints per instance to keep PLAN/FIX context short")
    ap.add_argument("--dataset", default="lite")
    ap.add_argument("--split", default="test")
    ap.add_argument("--instance-ids", default="",
                    help="optional comma/space-separated SWE instance ids to run exactly (overrides n_eval selection)")
    ap.add_argument("--repo-root", default="data/swe_repos")
    ap.add_argument("--n-eval", type=int, default=10)
    ap.add_argument("--src-bodies", type=int, default=4)
    ap.add_argument("--src-lines", type=int, default=70)
    ap.add_argument("--traverse-edges", action="store_true",
                    help="expand flat-retrieved support with same_class/calls neighbors from real AST "
                         "(the graph is contains-only; this follows the code edges flat search can't)")
    ap.add_argument("--audit-support-text", action="store_true",
                    help="for tiny instance audits, include compact retrieved support text in the trace dump")
    ap.add_argument("--max-new", type=int, default=400)
    ap.add_argument("--max-steps", type=int, default=8)
    ap.add_argument("--exact-verify", action="store_true",
                    help="run exact SWE verification for one-shot and slot outputs after emission")
    ap.add_argument("--verify-backend", choices=["docker", "sbcli"], default="docker")
    ap.add_argument("--verify-out-dir", default="artifacts/graph_growth/swe_verify")
    ap.add_argument("--verify-max-workers", type=int, default=4)
    ap.add_argument("--verify-timeout", type=int, default=1800)
    ap.add_argument("--verify-poll-secs", type=int, default=20)
    ap.add_argument("--verify-gold-sanity", type=int, default=5,
                    help="when exact verify is active, require this many gold patches to resolve first")
    ap.add_argument("--test-feedback", action="store_true",
                    help="TIER-2: after an applyable/anchored FIX, run the REAL test harness (Docker, "
                         "expensive) on it; if it doesn't resolve, feed the harness's own failure detail "
                         "back into the next FIX attempt instead of declaring success on syntax-match alone")
    ap.add_argument("--test-feedback-retries", type=int, default=2,
                    help="max FIX attempts per instance when --test-feedback is on (1 = no retry, just score)")
    ap.add_argument("--prefix-kv", action="store_true",
                    help="Phase-A gen-minimize: prime [issue+source] ONCE into a shared KV cache and run "
                         "DIAGNOSE->PLAN->FIX as terse chained turns (no re-encoding context/prior stages). "
                         "Linear (no backtrack). A/B vs the default slot_solve on apply/resolve/TIME.")
    ap.add_argument("--fix-samples", type=int, default=1,
                    help="pass@k: generate K INDEPENDENT sampled FIX candidates, verify each (needs a verifier), "
                         "keep the first that RESOLVES (else the first applyable). Set SWE_TEMP>0 so samples "
                         "differ; exploits the 'explore past the wrong first guess' effect. K=1 = today's behavior.")
    ap.add_argument("--session-out-dir", default="",
                    help="optional directory to write a per-run session bundle (predictions + dump + summary)")
    ap.add_argument("--session-name", default="",
                    help="optional session bundle name; default = swe_slot_<dataset>_<split>_<timestamp>")
    ap.add_argument("--dump", default="artifacts/swe_slot_dump.txt")
    a = ap.parse_args()
    if a.selftest:
        raise SystemExit(0 if _selftest() else 1)
    a = _apply_smoke_overrides(a)
    outputs = _prepare_outputs(a)
    oneshot_run_id, slot_run_id = _verify_run_ids(outputs, a.dataset, a.split)

    from transformers import AutoTokenizer
    from v5.lm_loader import load_frozen_lm
    from v5.graph_grower.swe_load import load_instances, checkout_repo
    from v5.graph_grower.swe_probe import load_traces, _symbol_name
    from v5.runtime.sr_withcode import load_symbol_meta
    from v5.runtime.verifier_retry import load_strategy_meta
    from v5.training.providers import RealEmbedder
    from v5.runtime.search_replace import SR_SYS, parse_sr
    from v5.graph_grower.swe_verify import write_predictions

    trust = os.environ.get("V5_LM_TRUST_REMOTE_CODE", "0").lower() in ("1", "true", "yes")
    traces = load_traces([a.traces])
    meta = load_symbol_meta([a.nodes])
    strategy_meta = load_strategy_meta(a.strategy_nodes) if a.strategy_source != "off" else {}
    strategy_items_by_inst = {}
    insts = {t["instance_id"]: t for t in load_instances(a.dataset, a.split, limit=0)}
    requested_ids = _parse_instance_ids(a.instance_ids)
    if requested_ids:
        ids = [i for i in requested_ids if i in traces and i in insts and all(s in meta for s in traces[i]["support_ids"])]
        missing = [i for i in requested_ids if i not in ids]
        if missing:
            print(f"warning: skipped unavailable/unsupported instance ids: {missing}", flush=True)
    else:
        ids = [i for i in traces if i in insts and all(s in meta for s in traces[i]["support_ids"])][:a.n_eval]
    print(f"instances={len(ids)} | symbol meta={len(meta)}", flush=True)
    issue_vecs = {}
    if strategy_meta:
        import torch
        embedder = RealEmbedder(torch.device("cpu"))
        strat_texts = {}
        for sid, items in strategy_meta.items():
            for nid, text, _node_type in items:
                clean = (text or "").strip()
                if clean:
                    strat_texts[nid] = clean
        strat_vecs = {
            nid: _unit(vec)
            for nid, vec in embedder.embed_nodes(strat_texts).items()
        }
        strategy_items_by_inst = {
            sid: [
                {"id": nid, "text": text or "", "node_type": node_type, "vec": strat_vecs.get(nid, [])}
                for nid, text, node_type in items
                if (text or "").strip()
            ]
            for sid, items in strategy_meta.items()
        }
        issue_texts = {
            sid: traces[sid]["issue"]
            for sid in strategy_meta
            if sid in traces
        }
        issue_vecs = {
            sid: _unit(vec)
            for sid, vec in embedder.embed_nodes(issue_texts).items()
        }
        print(
            f"strategy instances={len(strategy_meta)} | strategy nodes={len(strat_texts)} "
            f"| strategy-source={a.strategy_source}",
            flush=True,
        )

    inj_runtime = _InjectedRuntime(a.model, a.adapter_ckpt, a.gnn_ckpt, trust) \
        if (a.injector_mode != "off" or a.inject_fix) else None
    if inj_runtime is not None:
        model = inj_runtime.model
        tok = inj_runtime.tok
        dev = inj_runtime.device
    else:
        import torch
        model = load_frozen_lm(a.model); model.eval()
        tok = AutoTokenizer.from_pretrained(a.model, trust_remote_code=trust)
        dev = next(model.parameters()).device

    _think = bool(os.environ.get("SWE_THINK"))           # reasoning-depth probe: enable <think> + capture
    _think_cap = os.environ.get("SWE_THINK", "")
    _temp = float(os.environ.get("SWE_TEMP", "0") or 0)  # >0 = sample (probe greedy-loop vs reasoning-limit)
    def gen(system, user, ntok, *, inject=False, constrain_symbols=None):
        if inject and inj_runtime is not None:
            return inj_runtime.generate(system, user, ntok, constrain_symbols=constrain_symbols)
        import torch
        from transformers import LogitsProcessorList
        from v5.graph_grower.constrained_decode import make_inpatch_processor
        msgs = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        kw = dict(add_generation_prompt=True, return_tensors="pt", return_dict=True)
        try:
            enc = tok.apply_chat_template(msgs, enable_thinking=_think, **kw).to(dev)
        except TypeError:
            enc = tok.apply_chat_template(msgs, **kw).to(dev)
        procs = None
        if constrain_symbols:
            plen = enc["input_ids"].shape[1]
            procs = LogitsProcessorList([make_inpatch_processor(tok, constrain_symbols, plen)])
        _samp = {"do_sample": True, "temperature": _temp, "top_p": 0.95} if _temp > 0 else {"do_sample": False}
        with torch.no_grad():
            out = model.generate(
                **enc,
                max_new_tokens=ntok + (2200 if _think else 0),   # room for the reasoning chain
                **_samp,
                logits_processor=procs,
                pad_token_id=tok.eos_token_id,
            )
        t = tok.decode(out[0, enc["input_ids"].shape[1]:], skip_special_tokens=True)
        if _think and _think_cap not in ("1", "true", "yes"):
            with open(_think_cap, "a", encoding="utf-8") as _f:
                _f.write(f"\n\n######## STAGE sys={system[:40]!r} ########\n{t}\n")
        return re.sub(r"<think>.*?</think>", "", t, flags=re.DOTALL).strip()

    def should_inject(stage: str) -> bool:
        if inj_runtime is None:
            return False
        if stage == "diagnose":
            return a.injector_mode in ("slot", "all")
        if stage == "oneshot":
            return a.injector_mode == "all"
        if stage == "fix":
            return a.inject_fix
        return False
    verifier = SWEExactVerifier(a.dataset, a.split, a.verify_backend, a.verify_out_dir,
                                max_workers=a.verify_max_workers, timeout=a.verify_timeout,
                                poll_secs=a.verify_poll_secs, model_name="swe_slot") \
        if (a.exact_verify or a.test_feedback) else None
    if a.test_feedback:
        # mid-loop test-feedback only means something if the harness itself is trustworthy here ->
        # prove it BEFORE the loop (not just at the end), else every feedback round this run is noise.
        gold_n = min(a.verify_gold_sanity, len(ids))
        gold_tasks = [{"iid": i, "gold": insts[i].get("patch", "")} for i in ids[:gold_n]]
        ok, total = verifier.run_gold_sanity(gold_tasks, gold_n, tag="slot_pretest_gold_sanity")
        print(f"[test-feedback] pre-loop gold-sanity: {ok}/{total} gold patches resolved", flush=True)
        if ok != total:
            raise SystemExit("gold-sanity failed; refusing to trust --test-feedback on a broken harness/env")

    dump = open(outputs["dump"], "w", encoding="utf-8")
    oneshot_app = slot_app = scored = 0
    oneshot_preds, slot_preds = {}, {}
    eval_tasks, oneshot_eval, slot_eval = [], [], []
    for k, iid in enumerate(ids):
        t = traces[iid]; inst = insts[iid]
        support = [s for s in t["support_ids"] if s in meta]
        dest = Path(a.repo_root) / inst["repo"].replace("/", "__")
        ok, _ = checkout_repo(inst["repo"], inst["base_commit"], dest, timeout=1800)
        if not ok:
            print(f"  [{k+1}] {iid} checkout FAILED"); continue
        if a.traverse_edges:
            n0 = len(support)
            support = _traverse_support(dest, support, meta)
            if len(support) > n0:
                print(f"  [{k+1}] {iid} traverse: +{len(support)-n0} structural neighbors", flush=True)
        support_sources = _build_support_sources(dest, support, meta, a.src_lines)
        src = _compose_src(support_sources[:a.src_bodies])
        if not src.strip():
            print(f"  [{k+1}] {iid} no source read"); continue
        scored += 1
        issue = t["issue"]
        strategy_sources = _strategy_sources(
            iid,
            issue,
            traces,
            strategy_items_by_inst,
            issue_vecs,
            a.strategy_source,
            a.strat_topk,
            a.strat_hints,
        )
        support_symbols = sorted({
            name for sid in support
            if (name := _symbol_name(meta[sid].get("text", "")))
        })
        if inj_runtime is not None:
            inj_runtime.prepare_session(support, support_sources, meta)
        task = {"iid": iid, "gold": inst.get("patch", "")}
        eval_tasks.append(task)

        # ONE-SHOT baseline (single SR emit)
        g1 = gen(
            SR_SYS,
            fix_user(issue, src, hints=strategy_sources),
            a.max_new,
            inject=should_inject("oneshot"),
            constrain_symbols=(support_symbols if a.fix_constrain else None),
        )
        b1 = parse_sr(g1); app1 = bool(b1) and not _unmatched(b1, str(dest))
        oneshot_app += app1
        p1 = _patch(b1, str(dest))
        if p1.strip():
            oneshot_preds[iid] = p1
            oneshot_eval.append((task, p1))

        # SLOT path THROUGH THE ENGINE (DIAGNOSE -> PLAN -> FIX, backtrack on ungrounded plans / wrong-scope fixes)
        def diagnose_fn(issue, src, attempt):
            prompt = (
                diag_user_injected(issue, support_sources, attempt)
                if should_inject("diagnose")
                else diag_user(issue, src, attempt)
            )
            return gen(
                DIAG_SYS,
                prompt,
                160,
                inject=should_inject("diagnose"),
            )
        def plan_fn(issue, src, diagnosis, attempt):
            raw = gen(
                PLAN_SYS,
                plan_user(issue, src, diagnosis, attempt, hints=strategy_sources),
                220,
                inject=False,
            )
            return _repair_plan_to_src(raw, src, diagnosis)[0]
        tf_log = []
        _vsecs = [0.0]                                       # Docker-verify wall (to split gen vs verify)
        def fix_fn(issue, src, diagnosis, plan):
            test_failure = ""
            K = max(1, a.fix_samples)                          # pass@k: K independent sampled candidates
            rounds = max(K, a.test_feedback_retries if a.test_feedback else 1)
            multi = K > 1 or a.test_feedback                   # K==1 + no tf -> preserve single-shot return
            best = ("", [], "")                                # first APPLYABLE candidate (fallback if none resolve)
            patch = ""; blocks = []; g = ""
            for attempt in range(rounds):
                g = gen(
                    SR_SYS,
                    fix_user(issue, src, diagnosis, plan, hints=strategy_sources, test_failure=test_failure),
                    a.max_new,
                    inject=should_inject("fix"),
                    constrain_symbols=(support_symbols if a.slot_fix_constrain else None),
                )
                blocks = parse_sr(g)
                if a.sr_snap:
                    blocks = _repair_sr_to_src(blocks, str(dest))   # goal-#1 (B): verbatim-anchor snap
                patch = _patch(blocks, str(dest))              # "" unless applyable
                if patch and not best[0]:
                    best = (patch, blocks, g)                  # remember first applyable as the @k fallback
                ready = bool(verifier is not None and patch and _blocks_match_plan(blocks, plan))
                if ready:
                    _vt = time.time()
                    resolved, feedback = verifier.verify_patch_feedback(task, patch, tag=f"tf_{iid}")
                    _vsecs[0] += time.time() - _vt
                    tf_log.append({"attempt": attempt, "ran_tests": True, "resolved": resolved,
                                  "feedback": feedback[:1500], "raw": g[:1200]})
                    if resolved:
                        return patch, blocks, g                # a resolving sample -> done (best@k hit)
                    test_failure = feedback if a.test_feedback else ""   # pass@k stays INDEPENDENT; tf chains
                elif a.test_feedback or K > 1:
                    tf_log.append({"attempt": attempt, "ran_tests": False, "resolved": False,
                                  "reason": "not applyable/anchored", "raw": g[:1200]})
                if not multi:
                    return patch, blocks, g                    # original single-shot behavior preserved
            return best if best[0] else (patch, blocks, g)     # @k: prefer an applyable candidate
        log = []
        gate_src = _full_files_src(dest, support, meta)
        _t0 = time.time()
        if a.prefix_kv:
            p2, trace = kv_solve(model, tok, dev, issue, gate_src, dest, verifier, task, a, vsecs=_vsecs)
            fp = False; steps = len(trace.get("fix_attempts", []))
        else:
            p2, trace, fp, steps = slot_solve(issue, {"full": src, "sources": support_sources, "gate": gate_src}, diagnose_fn, plan_fn, fix_fn,
                                              max_steps=a.max_steps, log=log, capture_evidence=a.audit_support_text)
        _wall = time.time() - _t0; _gen = _wall - _vsecs[0]
        app2 = bool(p2.strip())
        slot_app += app2
        if app2:
            slot_preds[iid] = p2
            slot_eval.append((task, p2))

        print(f"  [{k+1}/{len(ids)}] {iid:28} oneshot_app={app1} slot_app={app2} "
              f"slot_steps={steps} diag_attempts={len(trace['diagnoses'])} "
              f"plan_attempts={len(trace['plans'])} fixpoint={fp} "
              f"| TIME slot={_wall:.1f}s gen={_gen:.1f}s verify={_vsecs[0]:.1f}s", flush=True)
        dump.write(f"\n===== {iid} =====\nISSUE: {issue[:200]}\n\n"
                   f"SUPPORT IDS: {support}\n"
                   f"SUPPORT SOURCES: {_support_digest(support_sources) if a.audit_support_text else [s['id'] for s in support_sources]}\n\n"
                   f"STRATEGY HINTS ({a.strategy_source}): {_support_digest(strategy_sources) if a.audit_support_text else [s['id'] for s in strategy_sources]}\n\n"
                   f"SLOT log: {log}\n"
                   f"RETRIEVALS: {trace['retrievals']}\n"
                   f"DIAGNOSES ({len(trace['diagnoses'])} attempts):\n" +
                   "\n".join(f"  [{i}] {d}" for i, d in enumerate(trace['diagnoses'])) +
                   f"\nPLANS ({len(trace['plans'])} attempts):\n" +
                   "\n".join(f"  [{i}] {p}" for i, p in enumerate(trace['plans'])) +
                   f"\nFIX attempts (applyable, anchored): "
                   f"{[(x['applyable'], x['anchored']) for x in trace['fix_attempts']]}\n"
                   f"RAW FIXES:\n" +
                   "\n".join(
                       f"  [{i}] blocks={x['n_blocks']} files={x['files']}\n{x['raw']}"
                       for i, x in enumerate(trace["fix_attempts"])
                   ) +
                   "\n"
                   f"TEST-FEEDBACK rounds ({len(tf_log)}):\n" +
                   "\n".join(
                       f"  [{r['attempt']}] ran_tests={r['ran_tests']} resolved={r['resolved']}"
                       + (f" reason={r['reason']}" if "reason" in r else f" feedback={r.get('feedback','')}")
                       + f"\n      raw={r['raw']}"
                       for r in tf_log
                   ) +
                   "\n"
                   f"ONESHOT applyable={app1}:\n{g1[:500]}\n")
    dump.close()
    oneshot_path = str(outputs["oneshot"])
    slot_path = str(outputs["slot"])
    n1 = write_predictions(oneshot_preds, oneshot_path, "v5_oneshot")
    n2 = write_predictions(slot_preds, slot_path, "v5_slot")
    print(f"\n=== #9 SYNTHESIS (engine-wired DIAGNOSE->PLAN->FIX vs one-shot, given support) ===")
    print(f"  applyable@1:  ONE-SHOT {oneshot_app}/{scored}  |  SLOT(engine) {slot_app}/{scored}")
    print(f"  emitted predictions: oneshot {n1} -> {oneshot_path} | slot {n2} -> {slot_path}")
    print(f"  dump (MANUALLY INSPECT the diagnoses + retry behavior) -> {outputs['dump']}")
    exact1 = exact2 = None
    if verifier is not None:
        gold_n = min(a.verify_gold_sanity, len(eval_tasks))
        if gold_n > 0:
            ok, total = verifier.run_gold_sanity(eval_tasks, gold_n, tag="slot_gold_sanity")
            print(f"  gold-sanity: {ok}/{total} gold patches resolved", flush=True)
            if ok != total:
                raise SystemExit("gold-sanity failed; refusing to trust exact SWE verifier results")
        exact1 = _exact_resolve_rate(verifier, "slot_oneshot", oneshot_eval, scored)
        exact2 = _exact_resolve_rate(verifier, "slot_graph", slot_eval, scored)
        if exact1 is not None and exact2 is not None:
            r1, e1, ok1, emit1 = exact1
            r2, e2, ok2, emit2 = exact2
            print(f"  exact resolve: ONE-SHOT {ok1}/{scored} ({r1:.0%}) | SLOT(engine) {ok2}/{scored} ({r2:.0%})")
            print(f"  patch emission: ONE-SHOT {emit1}/{scored} ({e1:.0%}) | SLOT(engine) {emit2}/{scored} ({e2:.0%})")
    else:
        print("  exact verify not run here. To score these predictions on the verifier box:")
        smoke_gold = min(2, max(1, scored))
        smoke_oneshot = min(2, max(1, n1)) if n1 else 0
        smoke_slot = min(2, max(1, n2)) if n2 else 0
        print("  quick smoke first (cheap sanity before full Docker run):")
        print(f"    python -m v5.graph_grower.swe_verify --gold-sanity --dataset {a.dataset} --split {a.split} --limit {smoke_gold}")
        if smoke_oneshot:
            print(f"    python -m v5.graph_grower.swe_verify --predictions {oneshot_path} --dataset {a.dataset} --split {a.split} --run-id {oneshot_run_id}_smoke --predictions-limit {smoke_oneshot}")
        if smoke_slot:
            print(f"    python -m v5.graph_grower.swe_verify --predictions {slot_path} --dataset {a.dataset} --split {a.split} --run-id {slot_run_id}_smoke --predictions-limit {smoke_slot}")
        print("  then the full batch:")
        print(f"    python -m v5.graph_grower.swe_verify --gold-sanity --dataset {a.dataset} --split {a.split} --limit 5")
        print(f"    python -m v5.graph_grower.swe_verify --predictions {oneshot_path} --dataset {a.dataset} --split {a.split} --run-id {oneshot_run_id}")
        print(f"    python -m v5.graph_grower.swe_verify --predictions {slot_path} --dataset {a.dataset} --split {a.split} --run-id {slot_run_id}")
    if outputs["summary"] is not None:
        summary = {
            "session_name": outputs["name"],
            "dataset": a.dataset,
            "split": a.split,
            "instance_ids": requested_ids,
            "model": a.model,
            "n_eval_requested": a.n_eval,
            "n_eval_scored": scored,
            "max_steps": a.max_steps,
            "max_new": a.max_new,
            "audit_support_text": a.audit_support_text,
            "predictions": {
                "oneshot": oneshot_path,
                "slot": slot_path,
            },
            "dump_path": str(outputs["dump"]),
            "applyable": {
                "oneshot": {"count": oneshot_app, "total": scored},
                "slot": {"count": slot_app, "total": scored},
            },
            "verify_commands": {
                "gold_sanity_smoke": f"python -m v5.graph_grower.swe_verify --gold-sanity --dataset {a.dataset} --split {a.split} --limit {min(2, max(1, scored))}",
                "oneshot_smoke": (f"python -m v5.graph_grower.swe_verify --predictions {oneshot_path} --dataset {a.dataset} "
                                   f"--split {a.split} --run-id {oneshot_run_id}_smoke --predictions-limit {min(2, max(1, n1))}")
                                   if n1 else "",
                "slot_smoke": (f"python -m v5.graph_grower.swe_verify --predictions {slot_path} --dataset {a.dataset} "
                                f"--split {a.split} --run-id {slot_run_id}_smoke --predictions-limit {min(2, max(1, n2))}")
                                if n2 else "",
                "gold_sanity": f"python -m v5.graph_grower.swe_verify --gold-sanity --dataset {a.dataset} --split {a.split} --limit 5",
                "oneshot": f"python -m v5.graph_grower.swe_verify --predictions {oneshot_path} --dataset {a.dataset} --split {a.split} --run-id {oneshot_run_id}",
                "slot": f"python -m v5.graph_grower.swe_verify --predictions {slot_path} --dataset {a.dataset} --split {a.split} --run-id {slot_run_id}",
            },
        }
        if exact1 is not None and exact2 is not None:
            summary["exact_resolve"] = {
                "oneshot": {"resolved": exact1[2], "emitted": exact1[3], "total": scored},
                "slot": {"resolved": exact2[2], "emitted": exact2[3], "total": scored},
            }
        with open(outputs["summary"], "w", encoding="utf-8") as w:
            json.dump(summary, w, indent=2)
        print(f"  session bundle -> {outputs['bundle']}")
        print(f"  session summary -> {outputs['summary']}")


if __name__ == "__main__":
    main()
