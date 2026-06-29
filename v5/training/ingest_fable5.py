"""Ingest Glint-Research/Fable-5-traces -> LGGN realizer / operator-mining corpus.

Fable-5 = ~4665 Fable-5 coding-agent sessions (strong-model CoT), greenfield feature-BUILDING (not
bug-fixing). Use for LGGN as a BROAD source beyond SWE: (a) realizer SFT (plan/intent -> edit), and
(b) operator MINING (cluster recurring edits -> typed operators, the discovery crux in LGGN_DESIGN.md).
No FAIL_TO_PASS verifier (greenfield) -> weak/self-supervised, NOT outcome-graded like SWE.

LICENSE: Fable-5-traces is **AGPL-3.0** — flag for any commercial use. Research/eval only by default.

Runs on a box with HF access (datasets). Two modes:
  # 1) SEE the real schema first (don't guess) -> paste it back so we finalize the extractor:
  python -m v5.training.ingest_fable5 --probe --n 3
  # 2) extract realizer records (adapt field names from the probe if needed):
  python -m v5.training.ingest_fable5 --ingest --limit 2000 --out data/fable5/realizer_corpus.jsonl
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

DATASET = "Glint-Research/Fable-5-traces"

# edit signals we look for in trace content (any of these = an "edit event" to extract)
_DIFF_RE = re.compile(r"(?m)^(?:diff --git |--- a/|\+\+\+ b/|@@ )")
_SR_RE = re.compile(r"<{5,7}\s*SEARCH|={5,7}\s*$|>{5,7}\s*REPLACE")
_CODEBLOCK_RE = re.compile(r"```[\w.+-]*\n(.*?)```", re.S)


def _load(split: str, streaming: bool):
    from datasets import load_dataset
    return load_dataset(DATASET, split=split, streaming=streaming)


def _truncate(v, n=300):
    s = v if isinstance(v, str) else json.dumps(v, default=str)
    return s[:n] + (" …" if len(s) > n else "")


def probe(n: int, split: str) -> None:
    """Print the real schema of the first n examples — so we adapt the extractor to ground truth,
    not a guess. Paste this output back."""
    ds = _load(split, streaming=True)
    for i, ex in enumerate(ds):
        if i >= n:
            break
        print(f"\n===== example {i} =====")
        for k, v in ex.items():
            t = type(v).__name__
            extra = ""
            if isinstance(v, list) and v:
                extra = f" [len={len(v)}, item0_type={type(v[0]).__name__}"
                if isinstance(v[0], dict):
                    extra += f", item0_keys={list(v[0].keys())}"
                extra += "]"
            print(f"  {k}: <{t}>{extra}  = {_truncate(v)}")


def _iter_text_events(ex: dict):
    """Yield (role, text) from whatever trace shape this row uses. Handles common forms:
    messages/conversation/events list of {role/content} or {type/text}, or a flat 'text'/'output'."""
    for key in ("messages", "conversation", "events", "trace", "steps", "turns"):
        seq = ex.get(key)
        if isinstance(seq, list):
            for it in seq:
                if isinstance(it, dict):
                    role = it.get("role") or it.get("type") or it.get("speaker") or "?"
                    text = it.get("content") or it.get("text") or it.get("message") or it.get("output") or ""
                    if isinstance(text, list):  # content sometimes = list of parts
                        text = "\n".join(p.get("text", "") if isinstance(p, dict) else str(p) for p in text)
                    if isinstance(text, str) and text.strip():
                        yield role, text
            return
    for key in ("text", "output", "completion", "response", "body"):
        if isinstance(ex.get(key), str) and ex[key].strip():
            yield "?", ex[key]
            return


def _extract_edits(events: list[tuple[str, str]]) -> list[dict]:
    """An edit event = text containing a diff / SR block / fenced code. Its INTENT = the most recent
    preceding assistant/CoT text. -> (intent, edit) realizer pair."""
    out = []
    prev_text = ""
    for role, text in events:
        is_edit = bool(_DIFF_RE.search(text) or _SR_RE.search(text))
        code = None
        if is_edit:
            code = text
        else:
            m = _CODEBLOCK_RE.search(text)
            if m and len(m.group(1).strip()) > 40:   # a substantial code block = candidate edit
                code, is_edit = m.group(0), True
        if is_edit and code:
            out.append({"intent": prev_text.strip()[:1200], "edit": code.strip()[:2000], "role": role})
        prev_text = text
    return out


def ingest(limit: int, out: str, split: str) -> None:
    ds = _load(split, streaming=True)
    n_rows = n_edits = 0
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        for ex in ds:
            if limit and n_rows >= limit:
                break
            n_rows += 1
            events = list(_iter_text_events(ex))
            if not events:
                continue
            sid = ex.get("id") or ex.get("session_id") or ex.get("uuid") or f"fable5_{n_rows}"
            for j, rec in enumerate(_extract_edits(events)):
                rec.update({"session_id": str(sid), "edit_idx": j, "source": "fable5"})
                f.write(json.dumps(rec) + "\n")
                n_edits += 1
    print(f"ingested {n_rows} sessions -> {n_edits} (intent,edit) realizer records -> {out}")
    print("AGPL-3.0 source — research/eval only by default; flag for commercial.")
    if n_edits == 0:
        print("WARN: 0 edits extracted -> the trace schema differs. Run --probe and adapt "
              "_iter_text_events / _extract_edits to the real field names.")


def main():
    ap = argparse.ArgumentParser(description="Ingest Fable-5 traces -> realizer/operator-mining corpus.")
    ap.add_argument("--probe", action="store_true", help="print the real dataset schema (do this first)")
    ap.add_argument("--ingest", action="store_true")
    ap.add_argument("--n", type=int, default=3, help="probe: examples to dump")
    ap.add_argument("--limit", type=int, default=0, help="ingest: max sessions (0=all)")
    ap.add_argument("--split", default="train")
    ap.add_argument("--out", default="data/fable5/realizer_corpus.jsonl")
    a = ap.parse_args()
    if a.probe:
        probe(a.n, a.split)
    elif a.ingest:
        ingest(a.limit, a.out, a.split)
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
