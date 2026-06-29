"""Ingest Glint-Research/Fable-5-traces -> LGGN realizer / operator-mining corpus.

Fable-5 = Fable-5 coding-agent sessions (strong-model CoT), greenfield feature-BUILDING (not bug-fix).
Broadens the corpus beyond SWE for: (a) realizer SFT (intent -> edit), (b) operator MINING.
No FAIL_TO_PASS verifier -> weak/self-supervised, NOT outcome-graded like SWE.

LICENSE: AGPL-3.0 — flag for any commercial/redistribution use. Research/eval only by default.

LAYOUT (from --probe): 4782 files. `history.jsonl` is a SESSION INDEX (skip it). Each per-session
`<uuid>.jsonl` is a stream of EVENT lines; an event = {type, message:{role, content:[parts]}, ...}.
A content part = {type, text, thinking, id, name, arguments:{file_path, ...}}. TOOL-CALL parts (have
`name` + `arguments`) are the EDITS; text/thinking parts are the CoT/intent. `datasets` can't cast the
mixed file schemas -> we download raw + parse manually, per session.

  python -m v5.training.ingest_fable5 --probe --n 3        # parse a few SESSION files
  python -m v5.training.ingest_fable5 --ingest --limit 0   # -> data/fable5/realizer_corpus.jsonl
  python -m v5.training.ingest_fable5 --selftest           # synthetic session, no network
"""
from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path

DATASET = "Glint-Research/Fable-5-traces"
OUT = "data/fable5/realizer_corpus.jsonl"
_SKIP = ("history.jsonl",)                                 # session index, not traces

_EDIT_TOOLS = ("edit", "write", "str_replace", "apply_patch", "create_file", "patch",
               "search_replace", "replace", "new_file", "save", "fs_write", "create", "update_file")
_EDIT_FIELDS = ("diff", "new_string", "content", "new_str", "code", "text", "patch", "replacement", "file_text")


def _download_files():
    from huggingface_hub import snapshot_download
    root = snapshot_download(repo_id=DATASET, repo_type="dataset",
                             allow_patterns=["*.json", "*.jsonl", "*.json.gz", "*.jsonl.gz"])
    return [p for p in sorted(Path(root).rglob("*"))
            if p.suffix in (".json", ".jsonl", ".gz") and p.name not in _SKIP]


def _read_events(fp: Path) -> list:
    """A per-session file -> list of event dicts (JSONL or JSON array)."""
    op = gzip.open if fp.suffix == ".gz" else open
    evs = []
    try:
        with op(fp, "rt", encoding="utf-8") as f:
            head = f.read(1); f.seek(0)
            if head == "[":
                evs = [e for e in json.load(f) if isinstance(e, dict)]
            else:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            evs.append(json.loads(line))
                        except Exception:
                            pass
    except Exception:
        pass
    return [e for e in evs if isinstance(e, dict)]


def _parts(ev: dict):
    """Event -> its content parts (handles message{} | messages[] | a bare content list)."""
    out = []
    for holder in (ev.get("message"), *(ev.get("messages") or [])):
        if isinstance(holder, dict):
            c = holder.get("content")
            if isinstance(c, list):
                out.extend(p for p in c if isinstance(p, dict))
            elif isinstance(c, str):
                out.append({"type": "text", "text": c})
    if isinstance(ev.get("content"), list):                # some events carry content directly
        out.extend(p for p in ev["content"] if isinstance(p, dict))
    return out


def _is_edit(part: dict) -> bool:
    name = (part.get("name") or "").lower()
    return bool(name) and any(t in name for t in _EDIT_TOOLS)


def _edit_payload(part: dict) -> str:
    args = part.get("arguments") or part.get("input")
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except Exception:
            return args[:6000]
    if isinstance(args, dict):
        fp = args.get("file_path") or args.get("path") or ""
        for k in _EDIT_FIELDS:
            if args.get(k):
                return (f"# file: {fp}\n" if fp else "") + str(args[k])[:6000]
        return json.dumps(args)[:6000]
    return ""


def _goal(events: list) -> str:
    for ev in events:
        if ev.get("prompt"):
            return str(ev["prompt"])[:2000]
        for p in _parts(ev):
            role = ((ev.get("message") or {}).get("role") or "").lower()
            if role in ("user", "human") and p.get("text"):
                return p["text"][:2000]
    return ""


def parse_session(events: list, sid: str = "") -> list[dict]:
    """A session's event stream -> [{goal, intent, edit, tool, file_path, source, session_id}].
    intent = the CoT/text immediately before each edit tool-call (the plan the realizer executes)."""
    goal = _goal(events)
    out, intent = [], ""
    for ev in events:
        for p in _parts(ev):
            txt = p.get("text") or p.get("thinking") or ""
            if _is_edit(p):
                args = p.get("arguments") if isinstance(p.get("arguments"), dict) else {}
                out.append({
                    "goal": goal, "intent": intent[:2000], "edit": _edit_payload(p),
                    "tool": (p.get("name") or "").lower(),
                    "file_path": (args.get("file_path") or args.get("path") or ""),
                    "source": "fable5", "session_id": sid,
                })
            elif isinstance(txt, str) and txt.strip():
                intent = txt
    return out


def probe(n: int) -> None:
    files = _download_files()
    print(f"[fable5] {len(files)} session files (history.jsonl skipped); first: {[f.name for f in files[:3]]}")
    tool_names, shown = set(), 0
    for fp in files[:300]:
        events = _read_events(fp)
        if not events:
            continue
        if shown == 0:
            print(f"  {fp.name}: {len(events)} events | event0 keys: {list(events[0].keys())}")
            for ev in events[:6]:
                for p in _parts(ev):
                    if p.get("name"):
                        tool_names.add(p["name"])
            print(f"  tool names seen (first events): {sorted(tool_names)}")
        recs = parse_session(events, fp.stem)
        if recs:
            r = recs[0]
            print(f"\n  === edit record ({fp.name}) ===  tool={r['tool']} file={r['file_path']}")
            print(f"  goal:   {r['goal'][:160]}")
            print(f"  intent: {r['intent'][:160]}")
            print(f"  edit:   {r['edit'][:200]}")
            shown += 1
            if shown >= n:
                return
    if shown == 0:
        print(f"  (0 edit records; tool names seen across files: {sorted(tool_names)} -> add the edit ones to _EDIT_TOOLS)")


def ingest(limit: int, out: str) -> None:
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    files = _download_files()
    if limit:
        files = files[:limit]
    n_sess = n_edits = 0
    with open(out, "w", encoding="utf-8") as f:
        for fp in files:
            events = _read_events(fp)
            if not events:
                continue
            n_sess += 1
            for rec in parse_session(events, fp.stem):
                if rec["edit"].strip():
                    f.write(json.dumps(rec) + "\n"); n_edits += 1
    print(f"ingested {n_sess} sessions -> {n_edits} (intent,edit) realizer records -> {out}")
    print("AGPL-3.0 — research/eval only by default. Next: label_gold over edits -> operator-plan SFT (mixed w/ SWE).")
    if n_edits == 0:
        print("WARN: 0 edits — run --probe, add the real edit-tool names to _EDIT_TOOLS.")


def _selftest() -> bool:
    print("ingest_fable5 --selftest: synthetic session event-stream parse (no network)\n")
    events = [
        {"type": "msg", "message": {"role": "user", "content": [{"type": "text", "text": "Add retry with backoff to the http client."}]}},
        {"type": "msg", "message": {"role": "assistant", "content": [
            {"type": "thinking", "thinking": "I'll wrap get() in a retry(3) decorator."},
            {"type": "tool_use", "name": "str_replace", "arguments":
                {"file_path": "http.py", "old_string": "return _raw(url)", "new_string": "return retry(3)(_raw)(url)"}}]}},
        {"type": "msg", "message": {"role": "assistant", "content": [
            {"type": "text", "text": "Now create the test."},
            {"type": "tool_use", "name": "create_file", "arguments": {"file_path": "test_http.py", "content": "def test_retry(): ..."}}]}},
    ]
    recs = parse_session(events, "sess1")
    for r in recs:
        print(f"  tool={r['tool']:12} file={r['file_path']:14} intent='{r['intent'][:34]}...'")
    assert len(recs) == 2, f"expected 2 edits, got {len(recs)}"
    assert recs[0]["tool"] == "str_replace" and "retry(3)" in recs[0]["edit"], "edit payload"
    assert "retry" in recs[0]["intent"], "intent = preceding CoT"
    assert recs[0]["goal"].startswith("Add retry"), "goal from first user text"
    print("\n  FABLE5 INGEST SELFTEST -> PASS")
    return True


def main():
    ap = argparse.ArgumentParser(description="Ingest Fable-5 traces -> realizer/operator-mining corpus.")
    ap.add_argument("--probe", action="store_true", help="parse a few SESSION files; print tool names + samples")
    ap.add_argument("--ingest", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--n", type=int, default=3, help="probe: edit records to dump")
    ap.add_argument("--limit", type=int, default=0, help="ingest: max sessions (0=all)")
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
