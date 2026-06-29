"""Ingest Glint-Research/Fable-5-traces -> LGGN realizer / operator-mining corpus.

Fable-5 = Fable-5 coding-agent sessions (strong-model CoT), greenfield feature-BUILDING (not bug-fix).
Broadens the corpus beyond SWE for: (a) realizer SFT (intent -> edit), (b) operator MINING.
No FAIL_TO_PASS verifier -> weak/self-supervised, NOT outcome-graded like SWE.

LICENSE: AGPL-3.0 — flag for any commercial/redistribution use. Research/eval only by default.

SCHEMA (from --probe): the HF files mix two record shapes, so `datasets` CANNOT cast them to one
schema (CastError). We bypass it — download the raw JSON files (huggingface_hub) and parse manually.
  * session row : {messages: [ {role, content:[parts]} ... ], prompt, tools, ...}
  * event row   : {message: {role, content:[parts]}, modelId, thinkingLevel, ...}
  * a content part: {type, text, thinking, id, name, arguments:{file_path, ...}} — TOOL-CALL parts
    (have `name` + `arguments`) are the EDITS; text/thinking parts are the CoT/intent.

  python -m v5.training.ingest_fable5 --probe --n 3        # files + a parsed sample
  python -m v5.training.ingest_fable5 --ingest --limit 0   # -> data/fable5/realizer_corpus.jsonl
  python -m v5.training.ingest_fable5 --selftest           # synthetic parse, no network
"""
from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path

DATASET = "Glint-Research/Fable-5-traces"
OUT = "data/fable5/realizer_corpus.jsonl"

# tool names that denote a code edit (substring match, case-insensitive)
_EDIT_TOOLS = ("edit", "write", "str_replace", "apply_patch", "create_file", "patch",
               "search_replace", "replace", "new_file", "save")
# argument fields that carry the edit payload, in preference order
_EDIT_FIELDS = ("diff", "new_string", "content", "new_str", "code", "text", "patch", "replacement")


def _download_files():
    """Download the dataset's raw JSON files (bypasses datasets' schema cast). Returns list of paths."""
    from huggingface_hub import snapshot_download
    root = snapshot_download(repo_id=DATASET, repo_type="dataset",
                             allow_patterns=["*.json", "*.jsonl", "*.json.gz", "*.jsonl.gz"])
    files = sorted(p for p in Path(root).rglob("*") if p.suffix in (".json", ".jsonl", ".gz"))
    return files


def _iter_rows(limit: int = 0):
    """Yield raw dict rows from the downloaded files. Each file is JSONL or a JSON array."""
    n = 0
    for fp in _download_files():
        op = gzip.open if fp.suffix == ".gz" else open
        try:
            with op(fp, "rt", encoding="utf-8") as f:
                head = f.read(1)
                f.seek(0)
                if head == "[":                       # JSON array
                    rows = json.load(f)
                    for r in rows:
                        yield r; n += 1
                        if limit and n >= limit:
                            return
                else:                                  # JSONL
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            yield json.loads(line); n += 1
                        except Exception:
                            continue
                        if limit and n >= limit:
                            return
        except Exception as e:
            print(f"  [skip {fp.name}: {e}]")


def _parts(row: dict):
    """Normalize either record shape -> a flat list of content parts (dicts)."""
    out = []
    msgs = row.get("messages")
    if isinstance(msgs, list):
        for m in msgs:
            c = (m or {}).get("content") if isinstance(m, dict) else None
            if isinstance(c, list):
                out.extend(p for p in c if isinstance(p, dict))
            elif isinstance(c, str):
                out.append({"type": "text", "text": c})
    msg = row.get("message")
    if isinstance(msg, dict):
        c = msg.get("content")
        if isinstance(c, list):
            out.extend(p for p in c if isinstance(p, dict))
        elif isinstance(c, str):
            out.append({"type": "text", "text": c})
    return out


def _is_edit(part: dict) -> bool:
    name = (part.get("name") or "").lower()
    return bool(name) and any(t in name for t in _EDIT_TOOLS)


def _edit_payload(part: dict) -> str:
    args = part.get("arguments")
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except Exception:
            return args[:6000]
    if isinstance(args, dict):
        for k in _EDIT_FIELDS:
            if args.get(k):
                fp = args.get("file_path") or args.get("path") or ""
                return (f"# file: {fp}\n" if fp else "") + str(args[k])[:6000]
        return json.dumps(args)[:6000]
    return ""


def parse_session(row: dict) -> list[dict]:
    """row -> [{goal, intent, edit, tool, file_path, source}]. intent = CoT/text immediately before
    the edit tool-call (the plan the realizer would execute)."""
    goal = (row.get("prompt") or row.get("goal") or row.get("task") or "")[:2000]
    sid = str(row.get("session_id") or row.get("id") or row.get("parentId") or "")
    out, intent = [], ""
    for p in _parts(row):
        txt = p.get("text") or p.get("thinking") or ""
        if _is_edit(p):
            args = p.get("arguments") if isinstance(p.get("arguments"), dict) else {}
            out.append({
                "goal": goal,
                "intent": intent[:2000],
                "edit": _edit_payload(p),
                "tool": (p.get("name") or "").lower(),
                "file_path": (args.get("file_path") or args.get("path") or ""),
                "source": "fable5", "session_id": sid,
            })
        elif isinstance(txt, str) and txt.strip():
            intent = txt
    return out


def probe(n: int) -> None:
    files = _download_files()
    print(f"[fable5] {len(files)} data files; first few: {[f.name for f in files[:5]]}")
    shown = 0
    for row in _iter_rows(limit=200):
        if not isinstance(row, dict):
            continue
        if shown == 0:
            print(f"  row keys: {list(row.keys())}")
            ps = _parts(row)
            print(f"  parts in row[0]: {len(ps)} | part keys sample: {list(ps[0].keys()) if ps else '—'}")
            tool_parts = [p for p in ps if p.get("name")]
            print(f"  tool-call parts: {[p.get('name') for p in tool_parts][:8]}")
        recs = parse_session(row)
        if recs:
            print(f"\n  === parsed edit record (row {shown}) ===")
            r = recs[0]
            print(f"  tool={r['tool']} file={r['file_path']}")
            print(f"  intent: {r['intent'][:200]}")
            print(f"  edit:   {r['edit'][:200]}")
            shown += 1
            if shown >= n:
                return
    if shown == 0:
        print("  (0 edit records — adapt _EDIT_TOOLS/_EDIT_FIELDS to the tool names printed above)")


def ingest(limit: int, out: str) -> None:
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    n_rows = n_edits = 0
    with open(out, "w", encoding="utf-8") as f:
        for row in _iter_rows(limit=limit):
            if not isinstance(row, dict):
                continue
            n_rows += 1
            for rec in parse_session(row):
                if rec["edit"].strip():
                    f.write(json.dumps(rec) + "\n"); n_edits += 1
    print(f"ingested {n_rows} rows -> {n_edits} (intent,edit) realizer records -> {out}")
    print("AGPL-3.0 — research/eval only by default. Next: label_gold over edits -> operator-plan SFT (mixed w/ SWE).")
    if n_edits == 0:
        print("WARN: 0 edits — run --probe, adapt _EDIT_TOOLS/_EDIT_FIELDS to the real tool names.")


def _selftest() -> bool:
    print("ingest_fable5 --selftest: synthetic parse of both record shapes (no network)\n")
    session_row = {
        "session_id": "s1", "prompt": "Add a retry decorator to the http client",
        "messages": [
            {"role": "user", "content": [{"type": "text", "text": "Add retry with backoff."}]},
            {"role": "assistant", "content": [
                {"type": "thinking", "thinking": "I'll wrap get() in a retry(3) decorator."},
                {"type": "tool_use", "name": "str_replace", "arguments":
                    {"file_path": "http.py", "old_string": "return _raw(url)", "new_string": "return retry(3)(_raw)(url)"}},
            ]},
        ],
    }
    event_row = {
        "id": "e1", "modelId": "fable-5",
        "message": {"role": "assistant", "content": [
            {"type": "text", "text": "Create the test file."},
            {"type": "tool_use", "name": "create_file", "arguments": {"file_path": "test_http.py", "content": "def test_retry(): ..."}},
        ]},
    }
    r1 = parse_session(session_row); r2 = parse_session(event_row)
    print(f"  session-shape -> {len(r1)} edits | tool={r1[0]['tool']} file={r1[0]['file_path']}")
    print(f"  event-shape   -> {len(r2)} edits | tool={r2[0]['tool']} file={r2[0]['file_path']}")
    assert len(r1) == 1 and r1[0]["tool"] == "str_replace" and "retry(3)" in r1[0]["edit"], "session-shape edit"
    assert "retry" in r1[0]["intent"], "intent = preceding CoT"
    assert len(r2) == 1 and r2[0]["tool"] == "create_file" and "test_retry" in r2[0]["edit"], "event-shape edit"
    print("\n  FABLE5 INGEST SELFTEST -> PASS")
    return True


def main():
    ap = argparse.ArgumentParser(description="Ingest Fable-5 traces -> realizer/operator-mining corpus.")
    ap.add_argument("--probe", action="store_true", help="download raw files + print schema + parsed sample")
    ap.add_argument("--ingest", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--n", type=int, default=3, help="probe: edit records to dump")
    ap.add_argument("--limit", type=int, default=0, help="ingest: max rows (0=all)")
    ap.add_argument("--out", default=OUT)
    a = ap.parse_args()
    if a.selftest:
        raise SystemExit(0 if _selftest() else 1)
    if a.probe:
        probe(a.n)
    elif a.ingest:
        ingest(a.limit, a.out)
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
