"""algo_grr_qatools -- REAL tools over REAL HotpotQA paragraphs, for the SAME ThinkerController that
drives algo_grr_thinkctl's SWE domain. This file plays the exact role algo_grr_swetools plays for
code: it is the ONLY thing that knows about HotpotQA; the controller, observation encoding, and
training loop in thinkctl.py never import anything from here directly except through a Domain.

Retrieval reuses membrane.multihop_retrieve's VALIDATED conditioning idea (this session confirmed:
conditioning on what was already found helps BRIDGE questions and does nothing for COMPARISON, exactly
as predicted) rather than reimplementing it, but as a per-step POINTED query instead of a fixed
question+mean(observed) formula -- the controller decides what to search for at each hop, which is
strictly more expressive than a hard-coded conditioning rule.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def _encode(texts: list) -> np.ndarray:
    from embedder import encode_batch
    return np.asarray(encode_batch(texts), dtype=np.float32)


def t_retrieve(state: dict, arg: str):
    """Cosine search over this question's real paragraphs for the query text `arg`. Returns the best
    NOT-YET-RETRIEVED paragraph -- repeats are useless so the tool refuses to hand back the same one."""
    paras = state["paras"]
    if state.get("_emb") is None:
        state["_emb"] = _encode([f"{t}. {b}"[:1000] for t, b in paras])
    E = state["_emb"]
    q = _encode([(arg or "")[:300]])[0]
    picked = state.setdefault("picked", [])
    for j in (-(E @ q)).argsort():
        if int(j) not in picked:
            picked.append(int(j))
            t, b = paras[int(j)]
            state.setdefault("retrieved", []).append((t, b))
            return True, f"{t}: {b[:300]}"
    return False, "nothing new to retrieve"


def t_read(state: dict, arg: str):
    """Open one already-retrieved paragraph by title -- the QA analogue of read_file. Rejects a title
    that was never retrieved rather than guessing which one was meant."""
    for t, b in state.get("retrieved", []):
        if t == arg or arg in t or t in arg:
            state["open_para"] = t
            state["open_text"] = b
            return True, b[:600]
    return False, f"not retrieved yet: {arg}"


def t_answer(state: dict, arg: str):
    """Commit to an answer. `arg` is POINTED AT from text the tools already surfaced -- never
    generated -- the same discipline as the SWE domain's edit anchor. Verification happens outside,
    in the domain's reward/metrics functions, never here."""
    state["answer"] = arg
    return True, f"answer: {arg}"


def qa_registry():
    """A ToolRegistry of the real tools, same shape swe_registry() gives algo_grr_agent's loop."""
    from v5.runtime.algo_grr_agent import ToolRegistry
    reg = ToolRegistry()
    reg.add("retrieve", t_retrieve, "retrieve the paragraph best matching a query")
    reg.add("read", t_read, "open a retrieved paragraph by title")
    reg.add("answer", t_answer, "commit to an answer string")
    return reg


def _selftest() -> bool:
    print("algo_grr_qatools --selftest: real tools over real HotpotQA paragraphs\n")
    ok = True

    def chk(t, c, d=""):
        nonlocal ok
        ok &= bool(c)
        print(f"  [{'PASS' if c else 'FAIL'}] {t}{('  - ' + d) if d else ''}")

    p = Path(__file__).resolve().parents[2] / "artifacts" / "hotpot_multihop.json"
    if not p.exists():
        print(f"  {p} missing; skipping (run scripts/prep_hotpot.py)")
        return False
    row = json.loads(p.read_text(encoding="utf-8"))[0]
    paras = [(t, b) for t, b in row["paras"]]
    state = {"paras": paras}

    ok1, obs = t_retrieve(state, row["q"][:200])
    chk("[1] retrieve returns a REAL paragraph", ok1 and len(obs) > 10, obs[:60])
    chk("[2] the retrieval is tracked in state for read/answer to point at",
        len(state.get("retrieved", [])) == 1)

    ok2, obs2 = t_retrieve(state, row["q"][:200])
    chk("[3] retrieve does not hand back the same paragraph twice",
        ok2 and state["retrieved"][1][0] != state["retrieved"][0][0], obs2[:60])

    title = state["retrieved"][0][0]
    ok3, obs3 = t_read(state, title)
    chk("[4] read opens a retrieved paragraph by title",
        ok3 and state.get("open_para") == title, obs3[:60])

    bad, obsb = t_read(state, "definitely not a real title")
    chk("[5] read REJECTS a title that was never retrieved", not bad, obsb[:50])

    ok4, _ = t_answer(state, "some entity")
    chk("[6] answer commits a pointed-at string; verification lives OUTSIDE the tool",
        ok4 and state["answer"] == "some entity")

    reg = qa_registry()
    chk("[7] registry exposes the real tools", set(reg) == {"retrieve", "read", "answer"},
        " ".join(sorted(reg)))

    print(f"\n  MEMBRANE_QATOOLS -> {'PASS' if ok else 'FAIL'}")
    return ok


if __name__ == "__main__":
    import sys
    sys.exit(0 if _selftest() else 1)
