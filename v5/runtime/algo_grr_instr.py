"""algo_grr_instr -- SELF-WRITTEN INSTRUCTIONS banked in the worlds graph and retrieved for new bugs.

THE CLAIM. After seeing a solved bug, the LM writes a short, GENERAL instruction ("for this kind of
failure, widen the type guard rather than the caller"). The instruction is banked as a real atom in the
AtomGraph, keyed by the issue's embedding. On a NEW bug, the nearest instruction is retrieved and put
in the prompt. If that helps, the system accumulates transferable procedure -- which is the compounding
property this project has been after, and which no coding-agent harness has.

WHY THIS AND NOT MORE SLOTS. Measured today, in order:
  * self-authored TOOLS banked and replayed: 0 replays / 40 real instances. A tool hardcodes
    `old = "        if isinstance(value, bytes):"` and can only ever fire on that exact line.
  * latent slot memory, two interfaces, four controls: instance-specific signal 0.0045-0.0075 CE
    against its own derangement noise of +-0.002-0.005. It compresses context; it does not remember.
  * the PROMPT channel, by contrast, is worth 1.4940 CE.
An instruction is natural language: it transfers by construction (no code generalisation needed from a
3B, which measurably cannot write one), and it enters through the channel that demonstrably works.

CONTROLS, PRESENT FROM THE FIRST RUN. Every retracted result today came from adding a control after
the fact, so the arms are:
    B  none      no instruction in the prompt                     (the incumbent)
    R  random    a RANDOM banked instruction                      (does ANY instruction help?)
    N  nearest   the instruction retrieved by issue similarity    (the claim)
    O  oracle    the instruction written FOR THIS VERY BUG        (ceiling; leaks by construction,
                 reported only to bound how much a perfect retrieval could ever buy)
N > R is the only comparison that supports retrieval. N == R means instructions help generically and
the graph is doing nothing -- exactly the shape of every null this session.

NO LEAKAGE: train/held are split BY INSTANCE (row-splitting leaked 64% of issue texts here earlier),
instructions are written only from TRAIN bugs, and arm O is labelled as a ceiling, never as a result.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path

_ROOT = str(Path(__file__).resolve().parents[2])
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
os.environ.setdefault("HF_HOME", r"E:\cache\hf")

import numpy as np
import torch
import torch.nn.functional as F

from v5.runtime.algo_grr_nstm_sup import group_split, load_pairs

WRITE_PROMPT = """A bug was fixed by changing one line.

BUG REPORT:
{issue}

BEFORE: {old}
AFTER:  {new}

Write ONE short instruction (max 25 words) that would help someone fix a DIFFERENT bug of the same
kind. Describe the KIND of change, not this file's names or values. Start with a verb.
Output only the instruction."""

USE_PROMPT_N = """{instr}

This line is wrong:
{old}
Write the corrected line, and nothing else:
"""

USE_PROMPT_B = """This line is wrong:
{old}
Write the corrected line, and nothing else:
"""


def write_instruction(lm, r) -> str:
    """The LM writes its own instruction from a SOLVED bug. Nothing is verified here -- the value of
    an instruction is decided later, by whether it helps on a DIFFERENT bug."""
    try:
        out = str(lm.generate_chat(
            WRITE_PROMPT.format(issue=r["issue"][:600], old=r["old"][:150], new=r["new"][:150]),
            max_new=48, temperature=0.3)).strip()
    except Exception:                                              # noqa: BLE001
        return ""
    out = out.splitlines()[0].strip() if out else ""
    return out[:200]


def build_bank(lm, rows, graph):
    """Bank each instruction as a real atom in the worlds graph, keyed by the ISSUE embedding -- the
    same AtomGraph the rest of this project uses, not a private dict."""
    from embedder import encode_batch
    from v5.runtime.membrane import Atom
    kept = []
    for i, r in enumerate(rows):
        ins = write_instruction(lm, r)
        if len(ins) < 12:
            continue
        emb = np.asarray(encode_batch([r["issue"][:600]])[0], dtype=np.float32)
        graph.add(Atom(name=f"instr::{r['instance_id']}::{i}", code="", kind="instruction",
                       provenance="self-written", description=ins, emb=emb))
        kept.append((r["instance_id"], ins, emb))
        if (i + 1) % 25 == 0:
            print(f"    wrote {i+1}/{len(rows)}  banked {len(kept)}", flush=True)
    return kept


def retrieve(bank, issue_emb, mode: str, rng):
    """mode: nearest | random. Random is the control that decides whether RETRIEVAL matters at all."""
    if not bank:
        return ""
    if mode == "random":
        return bank[rng.randrange(len(bank))][1]
    M = np.stack([b[2] for b in bank])
    M = M / (np.linalg.norm(M, axis=1, keepdims=True) + 1e-9)
    q = issue_emb / (np.linalg.norm(issue_emb) + 1e-9)
    return bank[int((M @ q).argmax())][1]


def ce_on_target(lm, prompt: str, target: str) -> float:
    """Teacher-forced CE on the REAL gold line. Same alignment as the prefix experiment, which was
    verified against an independent HF `labels` implementation to 1e-7."""
    tok = lm.tok
    ids = tok.apply_chat_template([{"role": "user", "content": prompt}],
                                  add_generation_prompt=True, return_tensors="pt")
    ids = (ids if torch.is_tensor(ids) else ids["input_ids"]).to(lm.device)
    tgt = tok(target, return_tensors="pt", add_special_tokens=False).input_ids.to(lm.device)
    full = torch.cat([ids, tgt], 1)
    with torch.no_grad():
        out = lm.model(full)
    lo = out.logits[0, ids.shape[1] - 1: full.shape[1] - 1, :].float()
    return float(F.cross_entropy(lo, tgt[0]))


def main():
    ap = argparse.ArgumentParser(description="Self-written instructions banked in the worlds graph.")
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--lm", type=str, default="Qwen/Qwen2.5-3B-Instruct")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    rows = load_pairs(a.n, seed=a.seed)
    tr, held = group_split(rows, seed=a.seed)
    print(f"{len(rows)} deduped real single-line fixes | train {len(tr)} held {len(held)} "
          f"(split BY INSTANCE)\n", flush=True)
    random.seed(a.seed)
    from v5.runtime.dcpd_latent import WhiteBox
    from v5.runtime.membrane import AtomGraph
    lm = WhiteBox(a.lm, quant="4bit")
    graph = AtomGraph()

    print("PHASE 1 -- the LM writes its own instructions from SOLVED train bugs", flush=True)
    bank = build_bank(lm, tr, graph)
    print(f"  banked {len(bank)} instructions as atoms in the worlds graph "
          f"({len(graph.atoms)} atoms)\n", flush=True)
    for _, ins, _ in bank[:5]:
        print(f"    e.g. {ins[:110]}", flush=True)

    from embedder import encode_batch
    print("\nPHASE 2 -- retrieve for HELD bugs the bank has never seen", flush=True)
    rng = random.Random(1234)
    tot = {"none": 0.0, "random": 0.0, "nearest": 0.0, "oracle": 0.0}
    for i, r in enumerate(held):
        e = np.asarray(encode_batch([r["issue"][:600]])[0], dtype=np.float32)
        tot["none"] += ce_on_target(lm, USE_PROMPT_B.format(old=r["old"]), r["new"])
        for mode in ("random", "nearest"):
            ins = retrieve(bank, e, mode, rng)
            tot[mode] += ce_on_target(lm, USE_PROMPT_N.format(instr=ins, old=r["old"]), r["new"])
        # ORACLE: an instruction written for THIS bug. Leaks by construction -- a ceiling, not a result.
        oi = write_instruction(lm, r)
        tot["oracle"] += ce_on_target(lm, USE_PROMPT_N.format(instr=oi or "Fix the bug.",
                                                              old=r["old"]), r["new"])
        if (i + 1) % 20 == 0:
            print(f"    {i+1}/{len(held)}", flush=True)

    n = max(1, len(held))
    B, R, N, O = (tot["none"] / n, tot["random"] / n, tot["nearest"] / n, tot["oracle"] / n)
    print(f"\n{'=' * 74}")
    print(f"SELF-WRITTEN INSTRUCTIONS FROM THE WORLDS GRAPH  (held {n}, split by instance)")
    print(f"  B none      (no instruction)              CE {B:.4f}")
    print(f"  R random    (a RANDOM banked instruction) CE {R:.4f}   vs none {B - R:+.4f}")
    print(f"  N nearest   (retrieved by similarity)     CE {N:.4f}   vs none {B - N:+.4f}")
    print(f"  O oracle    (written FOR this bug)        CE {O:.4f}   [CEILING, leaks by construction]")
    print(f"\n  RETRIEVAL SIGNAL  N vs R: {R - N:+.4f} CE")
    print(f"    -> {'RETRIEVAL WORKS -- the graph is doing something' if R - N > 0.02 else 'retrieval adds NOTHING over a random instruction'}")
    print(f"  headroom if retrieval were perfect (O vs N): {N - O:+.4f}")
    print(f"{'=' * 74}")


if __name__ == "__main__":
    main()
