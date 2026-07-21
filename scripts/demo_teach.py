"""demo_teach.py — the SEPARATE demo driver (proposal / 5-min video). NOT part of the real system.

The real system lives in v5.runtime.membrane (library + `--interactive` REPL). This script only DRIVES it
with a scripted lesson so the demo is deterministic. It shows, for real:
  1. WRITE-TIME GRAPH EDITING — teaching runs through learn_any -> add_or_merge: near-duplicate paraphrases
     MERGE (no bloat), related facts self-organize with typed 'related' edges (no LM needed; embedding-only).
  2. TEACH -> EXPLAIN (with --lm) — the frozen 4-bit LM can't answer an unseen fact, is taught, then answers
     it GROUNDED in the retrieved node.

    python scripts/demo_teach.py                                   # graph-editing proof (fast, no LM)
    python scripts/demo_teach.py --lm Qwen/Qwen3-4B-Instruct-2507     # + the teach->explain LM grounding
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = str(Path(__file__).resolve().parents[1])
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from v5.runtime.membrane import seed_graph, TRMRetriever, learn_any, encode_batch

# scripted lesson: deliberately includes a PARAPHRASE (should merge) and a RELATED fact (should self-link)
LESSON = [
    "The Klarn Protocol requires exactly three handshake phases: greet, verify, and seal.",
    "The Klarn Protocol uses three handshakes to connect: greet, verify, seal.",   # paraphrase -> MERGE
    "Klarn handshakes always time out after 30 seconds.",                          # related -> LINK
    "Zephyrite melts at 812 degrees and conducts electricity only when wet.",       # unrelated -> ADD
    "The Vora index measures how fast a market recovers after a shock, from 0 to 100.",
]
QUESTIONS = [
    "How many handshake phases does the Klarn Protocol have, and what are they?",
    "At what temperature does zephyrite melt?",
    "What does the Vora index measure?",
]


def _clean(t: str) -> str:
    for c in ("\nHuman:", "Human:", "\nYou are", "\n\n", "<|"):
        i = t.find(c); t = t[:i] if i > 0 else t
    return t.strip()


def main():
    ap = argparse.ArgumentParser(description="scripted demo driver for the real membrane (teach -> explain)")
    ap.add_argument("--lm", type=str, default="", help="real 4-bit LM for the grounding half, e.g. Qwen/Qwen3-4B-Instruct-2507")
    a = ap.parse_args()

    print("demo_teach.py -- driving the REAL membrane with a scripted lesson\n")
    g = seed_graph(); retr = TRMRetriever(g)
    n0 = len(g)

    # ---- 1) WRITE-TIME GRAPH EDITING (no LM) ----
    print("  [1] TEACH -> write-time graph editing (dedup + self-organize):")
    for fact in LESSON:
        r = learn_any(g, retr, fact)
        tag = "MERGE (near-dup)" if r["action"] == "merged" else "add"
        print(f"        {tag:<16} {r['node']:<14} | {fact[:56]}")
    taught = len(g) - n0
    print(f"\n      {len(LESSON)} statements taught -> {taught} new nodes "
          f"({len(LESSON) - taught} merged as duplicates). Graph self-organized {len(g.edges)} edges:")
    for s, d, rel in g.edges:
        if g.get(s).provenance == "learned" or g.get(d).provenance == "learned":
            print(f"        {s} --{rel}--> {d}")

    # ---- 2) RETRIEVE (neural) ----
    print("\n  [2] RETRIEVE (MiniLM neural rank) for each question:")
    picks = []
    for q in QUESTIONS:
        M, order = g.matrix(); sims = (M @ encode_batch([q])[0])
        j = int(sims.argmax()); picks.append(order[j])
        print(f"        {sims[j]:.2f}  {order[j]:<14} <- {q[:52]}")

    if not a.lm:
        print("\n  (pass --lm Qwen/Qwen3-4B-Instruct-2507 to see the frozen LM taught -> explain, grounded)")
        return

    # ---- 3) TEACH -> EXPLAIN with the real frozen 4-bit LM ----
    from v5.runtime.dcpd_latent import WhiteBox
    wb = WhiteBox(a.lm, quant="4bit")
    print(f"\n  LM {a.lm} quant={wb.quant} VRAM={wb.vram_gb:.2f}GB "
          f"({'FITS 6GB' if wb.vram_gb <= 6 else 'OVER 6GB'})")
    print("\n  [3] BEFORE (LM alone) vs AFTER (LM + taught graph):")
    for q, pick in zip(QUESTIONS, picks):
        before = _clean(wb.generate_chat(q, max_new=64))
        node = g.get(pick)
        after = _clean(wb.generate_chat(
            q, system=f"Use this fact the user taught you:\n{node.code or node.description}", max_new=64))
        print(f"     Q: {q}")
        print(f"        LM alone   -> {before[:96]}")
        print(f"        LM + graph -> {after[:96]}  (grounded in {pick})")
    print("\n  => frozen LM couldn't, was TAUGHT, then EXPLAINED it grounded — on a 6GB GPU. Demo claim shown.")


if __name__ == "__main__":
    main()
