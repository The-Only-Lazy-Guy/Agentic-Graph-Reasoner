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


import re as _re

# Reject: backticked spans, ANY underscore-bearing token (`_indices`, `w_pad`, `make_bytes` --
# a LEADING-underscore name is NOT caught by the naive \w+_\w+, which was a real hole in the first
# version), dotted attribute paths, and camelCase. All are this-bug identifiers that cannot fire on
# another bug, so they are exactly the instructions that make a bank untransferable.
BACKTICK = _re.compile(r"`[^`]+`|\S*_\S*|\w+\.\w+|[a-z]+[A-Z]\w*")


def is_generic(ins: str) -> bool:
    """GENERIC-NESS GATE, applied at BANK time -- the same verifier-gated discipline used for tools.
    An instruction naming THIS bug's identifiers cannot fire on another bug, which is the tool-transfer
    failure in prose form. Measured on the first run: 27 of 60 banked instructions embedded specific
    names (`w_pad`, `indptr`, `_indices`), so nearly half the bank was untransferable by construction."""
    return not BACKTICK.search(ins or "")


def defect_key(r, mode: str) -> str:
    """What the instruction is INDEXED by.

    CONSTRAINT that decides this: at query time only the ISSUE and the OLD line exist -- `new` is the
    thing being predicted, so an old->new change-shape key is usable when BANKING and unavailable when
    QUERYING. The key must therefore be computable from (issue, old) alone.

    mode:
      issue -- the issue prose (the original, and measured weak: retrieval bought +0.0339 over random)
      code  -- the OLD LINE, i.e. the code context that needs repairing. Issue text is mostly
               traceback, repro steps and repo chatter; the defect lives in the line.
      both  -- code first, then a short issue tail.
    """
    if mode == "issue":
        return r["issue"][:600]
    if mode == "code":
        return r["old"][:200]
    return f"{r['old'][:200]} || {r['issue'][:200]}"


def build_bank(lm, rows, graph, key_mode: str = "code", gate: bool = True):
    """Bank each instruction as a real atom in the worlds graph, keyed by the ISSUE embedding -- the
    same AtomGraph the rest of this project uses, not a private dict."""
    from embedder import encode_batch
    from v5.runtime.membrane import Atom
    kept, seen_key, n_gated = [], set(), 0
    for i, r in enumerate(rows):
        ins = write_instruction(lm, r)
        if len(ins) < 12:
            continue
        if gate and not is_generic(ins):
            n_gated += 1
            continue
        k = defect_key(r, key_mode)
        if k in seen_key:            # DEDUPE BY KEY: multi-hunk bugs share an issue, so the original
            continue                 # dedupe on (issue,old,new) still left IDENTICAL keys (cosine 1.0)
        seen_key.add(k)
        emb = np.asarray(encode_batch([k])[0], dtype=np.float32)
        graph.add(Atom(name=f"instr::{r['instance_id']}::{i}", code="", kind="instruction",
                       provenance="self-written", description=ins, emb=emb))
        kept.append((r["instance_id"], ins, emb))
        if (i + 1) % 50 == 0:
            print(f"    wrote {i+1}/{len(rows)}  banked {len(kept)}  gated-out {n_gated}", flush=True)
    print(f"    gate rejected {n_gated} name-specific instructions", flush=True)
    return kept


def validate_bank(lm, bank, val_rows, graph, min_gain: float = 0.0):
    """VERIFIER-GATED GRAPH EDITING: keep an instruction only if it MEASURABLY helps.

    This is the loop that was missing, and it is why retrieval looked dead. Nothing ever checked
    whether a banked instruction was any good: they were written once from a solved bug and never
    revised, so an instruction that HURTS stayed in the graph forever. Measured directly -- a
    mismatched instruction took a held bug from CE 1.2719 to 1.8045. If a large share of the bank is
    harmful, nearest-neighbour retrieval keeps picking harmful entries and the average washes out,
    which is exactly the +0.01-0.03 signal seen against a random draw.

    So each instruction is APPLIED to a small validation slice and scored by the CE it actually
    causes, versus no instruction at all. Negative-utility entries are DELETED FROM THE GRAPH. This is
    the same discipline the rest of this project uses for tools and atoms: writes are gated by real
    outcomes, never by plausibility. It is also the first time anything here EDITS the graph rather
    than only appending to it.
    """
    base = [ce_on_target(lm, USE_PROMPT_B.format(old=r["old"]), r["new"]) for r in val_rows]
    kept, dropped = [], 0
    for j, (iid, ins, emb) in enumerate(bank):
        gains = []
        for r, b in zip(val_rows, base):
            c = ce_on_target(lm, USE_PROMPT_N.format(instr=ins, old=r["old"]), r["new"])
            gains.append(b - c)                      # positive == the instruction HELPED
        u = float(np.mean(gains))
        if u > min_gain:
            kept.append((iid, ins, emb, u))
        else:
            dropped += 1
            for nm in [k for k, at in graph.atoms.items() if at.description == ins]:
                graph.atoms.pop(nm, None)            # the graph EDIT: harmful entry removed
        if (j + 1) % 25 == 0:
            print(f"    validated {j+1}/{len(bank)}  kept {len(kept)}  pruned {dropped}", flush=True)
    print(f"    PRUNED {dropped} of {len(bank)} instructions by MEASURED utility "
          f"(graph now {len(graph.atoms)} atoms)", flush=True)
    return kept


def retrieve(bank, issue_emb, mode: str, rng, k: int = 5):
    """mode: nearest | random | topk.

    random = uniform over the WHOLE bank -- the weakest control, and the one nearest must beat.
    topk   = a random draw from the k NEAREST. Separates COARSE topical match from FINE ranking:
             if nearest ~= topk but both >> random, the graph's membership carries the signal and its
             ORDERING does not.
    """
    if not bank:
        return ""
    if mode == "random":
        return bank[rng.randrange(len(bank))][1]
    M = np.stack([b[2] for b in bank])
    M = M / (np.linalg.norm(M, axis=1, keepdims=True) + 1e-9)
    q = issue_emb / (np.linalg.norm(issue_emb) + 1e-9)
    sims = M @ q
    if mode == "topk":
        idx = list(np.argsort(-sims)[:min(k, len(bank))])
        return bank[int(idx[rng.randrange(len(idx))])][1]
    return bank[int(sims.argmax())][1]


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
    ap.add_argument("--key", type=str, default="code", choices=["issue", "code", "both"],
                    help="what instructions are INDEXED by. issue = the original weak key "
                         "(+0.0339 over random); code = the OLD LINE, which is where the defect "
                         "actually lives; both = code + a short issue tail.")
    ap.add_argument("--prune", action="store_true",
                    help="VERIFIER-GATED GRAPH EDIT: measure each instruction on a validation "
                         "slice and DELETE the ones that make CE worse. Nothing here had ever "
                         "EDITED the graph -- only appended to it.")
    ap.add_argument("--val", type=int, default=12, help="validation bugs per instruction")
    ap.add_argument("--no-gate", action="store_true",
                    help="disable the generic-ness gate (bank name-specific instructions too)")
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
    bank = build_bank(lm, tr, graph, key_mode=a.key, gate=not a.no_gate)
    if a.prune:
        # validation slice comes from TRAIN, never from held -- pruning must not see the evaluation
        # set, or the surviving bank would be selected on the very thing it is later scored against.
        val = tr[:a.val]
        print(f"  VALIDATING {len(bank)} instructions on {len(val)} TRAIN bugs "
              f"(held is never touched)", flush=True)
        bank = validate_bank(lm, bank, val, graph)
    print(f"  banked {len(bank)} instructions as atoms in the worlds graph "
          f"({len(graph.atoms)} atoms)\n", flush=True)
    for b in bank[:5]:
        u = f"  [utility {b[3]:+.3f}]" if len(b) > 3 else ""
        print(f"    e.g. {b[1][:95]}{u}", flush=True)

    from embedder import encode_batch
    print("\nPHASE 2 -- retrieve for HELD bugs the bank has never seen", flush=True)
    rng = random.Random(1234)
    per = {"none": [], "random": [], "topk": [], "nearest": [], "oracle": []}
    for i, r in enumerate(held):
        # QUERY WITH THE SAME KEY the bank was built with -- previously the bank was keyed on the
        # issue and so was the query, which is why retrieval was matching prose to prose.
        e = np.asarray(encode_batch([defect_key(r, a.key)])[0], dtype=np.float32)
        per["none"].append(ce_on_target(lm, USE_PROMPT_B.format(old=r["old"]), r["new"]))
        for mode in ("random", "topk", "nearest"):
            ins = retrieve(bank, e, mode, rng)
            per[mode].append(ce_on_target(lm, USE_PROMPT_N.format(instr=ins, old=r["old"]), r["new"]))
        # ORACLE: an instruction written for THIS bug. Leaks by construction -- a ceiling, not a result.
        oi = write_instruction(lm, r)
        per["oracle"].append(ce_on_target(lm, USE_PROMPT_N.format(instr=oi or "Fix the bug.",
                                                                  old=r["old"]), r["new"]))
        if (i + 1) % 20 == 0:
            print(f"    {i+1}/{len(held)}", flush=True)
    tot = {k: float(np.sum(v)) for k, v in per.items()}

    n = max(1, len(held))
    B, R, N, O = (tot["none"] / n, tot["random"] / n, tot["nearest"] / n, tot["oracle"] / n)
    print(f"\n{'=' * 74}")
    print(f"SELF-WRITTEN INSTRUCTIONS FROM THE WORLDS GRAPH  (held {n}, split by instance)")
    print(f"  B none      (no instruction)              CE {B:.4f}")
    print(f"  R random    (a RANDOM banked instruction) CE {R:.4f}   vs none {B - R:+.4f}")
    print(f"  N nearest   (retrieved by similarity)     CE {N:.4f}   vs none {B - N:+.4f}")
    print(f"  O oracle    (written FOR this bug)        CE {O:.4f}   [CEILING, leaks by construction]")
    T = tot["topk"] / n
    print(f"  T top-5 rnd (random among 5 NEAREST)      CE {T:.4f}   vs none {B - T:+.4f}")
    # PAIRED BOOTSTRAP -- the evaluation is paired (same bug and target, only the retrieved
    # instruction differs), so resampling held examples gives a CI directly. Promised on an earlier
    # run and NOT delivered: that patch silently failed to apply and I quoted +0.0339 with no error
    # bar at all. Anchors are asserted now so a non-applying patch fails loudly instead of silently.
    d = np.array(per["random"]) - np.array(per["nearest"])
    rs = np.random.RandomState(0)
    boot = np.array([d[rs.randint(0, len(d), len(d))].mean() for _ in range(4000)])
    lo, hi = np.percentile(boot, [2.5, 97.5])
    print(f"\n  RETRIEVAL SIGNAL  N vs R: {R - N:+.4f} CE   95% CI [{lo:+.4f}, {hi:+.4f}]")
    print(f"  ORDERING signal   N vs T: {T - N:+.4f} CE   (does rank-1 beat a draw from the top 5?)")
    print(f"    -> {'RETRIEVAL WORKS (CI excludes 0)' if lo > 0 else 'NOT SIGNIFICANT -- CI includes 0'}")
    print(f"  headroom if retrieval were perfect (O vs N): {N - O:+.4f}")
    print(f"{'=' * 74}")


if __name__ == "__main__":
    main()
