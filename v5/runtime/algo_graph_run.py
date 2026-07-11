"""The unified algorithm-graph loop — ON the existing stack (no parallel island).

Retire path for artifact_graph/algo_curriculum: the "store/retrieve/reward/credit" they hand-rolled
now come from the validated modules —

    retrieve  -> v5.memory.memory.TotalMemory.read  (two-hop concept routing + ranker query_fn)
    author    -> the frozen LM (gen_fn)             (representation-open: it may CALL a stored node,
                                                      DEFINE a new reusable helper, or note a strategy)
    verify    -> v5.runtime.tool_compose.verify_fn  (execution = the code solves_fn)
    reward    -> v5.runtime.derive_reward.code_reward + grounded_code  (compose>novel>bare, fail<0)
    write-back-> TotalMemory.write(form="code")     (L1 record + L2 graph_edits lifecycle: MINT/
                                                      STRENGTHEN/MERGE, poison-gate, confidence)

The node is REPRESENTATION-OPEN (the model chooses the form; §2 of the plan). This first build proves
the loop with CODE nodes on the graph-algorithm curriculum (execution = tight grounding); nl/latent/
lora forms plug into the SAME `write(form=...)` + outcome-credit with no rework.

  selftest (no model):  python -m v5.runtime.algo_graph_run --selftest
  run (GPU):            V5_LM_TRUST_REMOTE_CODE=1 python -m v5.runtime.algo_graph_run --model Qwen/Qwen2.5-3B
"""
from __future__ import annotations

import argparse
import ast
import re
import sys

from v5.memory.memory import TotalMemory
from v5.runtime.derive_reward import _def_names, code_reward, grounded_code
from v5.runtime.tool_compose import verify_fn
from v5.runtime.tool_memory import _extract_code


def _top_defs(code: str) -> dict[str, str]:
    """Top-level `def name` -> source segment (for write-back: store each authored function)."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return {}
    out = {}
    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            seg = ast.get_source_segment(code, node)
            if seg:
                out[node.name] = seg
    return out


def _sig(code: str) -> str:
    m = re.search(r"def\s+\w+\s*\([^)]*\)", code or "")
    return m.group(0) if m else ""


def _author_prompt(task, advertised: list[tuple[str, str]]) -> str:
    """Light supervision: OFFER the model options (call a stored node / define a reusable helper /
    note a strategy) — never force one. `advertised` = [(name, code)] retrieved from memory."""
    parts = [task.text]
    if advertised:
        parts.append("\nReusable functions already in your library (already DEFINED — CALL them by "
                     "name, do NOT re-implement their logic):")
        for name, code in advertised:
            parts.append(f"  {_sig(code) or name}")
    parts.append("\nYou may CALL the functions above, DEFINE a new small reusable helper for any "
                 "sub-computation, and/or leave a short reusable strategy note — your choice.")
    parts.append(f"Write `{task.name}(...)`. Output ONLY a Python code block.")
    return "\n\n".join(parts)


def solve_with_memory(tm: TotalMemory, gen_fn, task, vseeds, eseeds, k: int = 6, samples: int = 1,
                      writeback: bool = True):
    """One task: retrieve nodes from memory -> author -> verify by execution -> reward -> write back
    verified nodes (representation-open, form=code). Returns a per-task dict."""
    from v5.runtime.algo_curriculum import cases           # task family (moves here at retirement)
    vcases, ecases = cases(task, vseeds), cases(task, eseeds)

    hit = tm.read(goal=task.text, span=task.text, k_impl=k)
    advertised = [(r.get("task_id") or "", r["new"]) for r in hit.impls
                  if r.get("form") == "code" and r.get("task_id") and r.get("new")]
    adv_names = [n for n, _ in advertised]

    best = (0.0, "", [])
    for gen in gen_fn([_author_prompt(task, advertised)] * samples):
        code = _extract_code(gen)
        defined = _def_names(code)
        called = [n for n in adv_names if n not in defined and re.search(rf"\b{re.escape(n)}\s*\(", code)]
        deps = "\n\n".join(c for n, c in advertised if n in called)
        acc, _, _ = verify_fn(code, task.name, vcases, deps)
        if acc > best[0]:
            best = (acc, code, called)
    _, code, called = best
    deps = "\n\n".join(c for n, c in advertised if n in called)
    verified = bool(code) and verify_fn(code, task.name, ecases, deps)[0] >= 0.999

    composed, used = grounded_code(code, adv_names)          # composition (call a stored node, no shadow)
    new_helpers = [n for n in _top_defs(code) if n != task.name and n not in adv_names]
    R, bd = code_reward(verified, composed_used=used,
                        authored_new_verified=len(new_helpers) if verified else 0)

    written = []
    if verified and writeback:
        for name, src in _top_defs(code).items():           # store each authored function as a node
            iid = tm.write(goal=task.text, old="", new=src, trace=f"{task.name} solution",
                           verified=True, task_id=name, form="code")
            if iid:
                written.append(name)
    return dict(name=task.name, verified=verified, reward=round(R, 3), reused=used,
                authored=new_helpers, written=written, breakdown=bd)


def run_stream(tm: TotalMemory, gen_fn, stream=None, verify_n=6, eval_n=10, k=6, samples=1):
    from v5.runtime.algo_curriculum import STREAM
    stream = stream or STREAM
    vseeds, eseeds = range(300, 300 + verify_n), range(700, 700 + eval_n)
    log = []
    for task in stream:
        log.append(solve_with_memory(tm, gen_fn, task, vseeds, eseeds, k=k, samples=samples))
    return log


# ═══════════════════════════════════════════════════════════════════════════════
# STUB LM (no GPU) — composes a stored build_adj when advertised, else defines it inline
# ═══════════════════════════════════════════════════════════════════════════════

def _stub_gen(prompts: list[str]) -> list[str]:
    from v5.runtime.algo_curriculum import _BUILD_ADJ, _STUB_BODY
    out = []
    for p in prompts:
        name = re.findall(r"Write `([a-z_][a-z0-9_]*)\(", p)[-1]
        needs, body = _STUB_BODY[name]
        block = p.split("Reusable functions", 1)[1].split("You may CALL", 1)[0] if "Reusable functions" in p else ""
        pieces = []
        if needs and not re.search(r"\bbuild_adj\s*\(", block):
            pieces.append(_BUILD_ADJ)
        pieces.append(body)
        out.append("```python\n" + "\n\n".join(pieces) + "\n```")
    return out


# ═══════════════════════════════════════════════════════════════════════════════
# SELFTEST (no model) — the unified loop end-to-end on the real memory stack
# ═══════════════════════════════════════════════════════════════════════════════

def _selftest() -> bool:
    import tempfile
    from v5.memory.store import make_fake_embedder
    from v5.runtime.algo_curriculum import BY_NAME, _BUILD_ADJ
    print("algo_graph_run --selftest: retrieve(TotalMemory) -> author -> verify -> reward -> "
          "write-back, on the real stack (no model)\n")
    vs, es = range(300, 306), range(700, 710)

    with tempfile.TemporaryDirectory() as td:
        tm = TotalMemory(td, mode="flat", embed_fn=make_fake_embedder())
        # seed build_adj as a CODE node. ctx_text = the query task's text so the hash-exact fake
        # embedder retrieves it (semantic retrieval quality is mpnet's job, covered by memory's own
        # selftests) — here we prove the LOOP mechanism, not the embedding.
        ntask = BY_NAME["neighbors_of"]
        tm.write(goal=ntask.text, old="", new=_BUILD_ADJ, trace="adjacency list",
                 verified=True, task_id="build_adj", form="code")
        assert tm.impls.get(next(iter(tm.impls.records)))["form"] == "code"
        print("  [1] seeded build_adj as a form=code node in TotalMemory -> PASS")

        # solve neighbors_of: must RETRIEVE build_adj, COMPOSE it, verify, reward>bare, write back
        res = solve_with_memory(tm, _stub_gen, ntask, vs, es)
        assert res["verified"], f"should solve: {res}"
        assert res["reused"] == ["build_adj"], f"should compose the retrieved build_adj: {res}"
        assert res["reward"] > 1.0, f"compose reward should beat bare solve: {res}"
        assert "neighbors_of" in res["written"], f"solution written back as a node: {res}"
        print(f"  [2] retrieve->compose build_adj->verify->reward {res['reward']:+.2f}->write-back -> PASS")

        # the written node is now in the graph with a concept (L2 lifecycle fired via write)
        assert tm.stats()["impls"] == 2, tm.stats()
        rec = next(r for r in tm.impls.records.values() if r["task_id"] == "neighbors_of")
        assert rec["form"] == "code" and rec["verified"] == "strong"
        print(f"  [3] write-back -> L1 record (form=code) + L2 lifecycle ({tm.stats()['concepts']} concepts) -> PASS")

        # inline baseline scores LESS than compose (GRPO would demote it): solve with NO memory ->
        # the stub defines build_adj inline -> not composed -> bare-solve reward
        tm_empty = TotalMemory(td + "_e", mode="flat", embed_fn=make_fake_embedder())
        res_inline = solve_with_memory(tm_empty, _stub_gen, ntask, vs, es, writeback=False)
        assert res_inline["verified"] and res_inline["reused"] == [], res_inline
        assert res_inline["reward"] < res["reward"], \
            f"inline ({res_inline['reward']}) must score below compose ({res['reward']})"
        print(f"  [4] inline-solve reward {res_inline['reward']:+.2f} < compose {res['reward']:+.2f} "
              f"(GRPO demotes inline) -> PASS")

    print("\n  ALGO_GRAPH_RUN SELFTEST -> PASS")
    return True


def _real_gen_fn(model_name: str, chunk: int):
    import os
    from transformers import AutoTokenizer
    from v5.lm_loader import load_frozen_lm
    from v5.runtime.reason_rl import batch_generate
    trust = os.environ.get("V5_LM_TRUST_REMOTE_CODE", "0").lower() in ("1", "true", "yes")
    model = load_frozen_lm(model_name); model.eval()
    tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=trust)
    dev = next(model.parameters()).device

    def gen_fn(prompts):
        return batch_generate(model, tok, prompts, dev, max_new=420, sample=True,
                              temperature=1.0, chunk=chunk)
    return gen_fn


def main():
    ap = argparse.ArgumentParser(description="Unified algorithm-graph loop on TotalMemory + derive_reward.")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--model", default="Qwen/Qwen2.5-3B")
    ap.add_argument("--root", default="data/algo_graph", help="TotalMemory root (persists the graph)")
    ap.add_argument("--samples", type=int, default=4)
    ap.add_argument("--chunk", type=int, default=8)
    a = ap.parse_args()
    if a.selftest:
        sys.exit(0 if _selftest() else 1)
    from v5.memory.store import make_mpnet_embedder
    tm = TotalMemory(a.root, mode="concept", embed_fn=make_mpnet_embedder())
    log = run_stream(tm, _real_gen_fn(a.model, a.chunk), samples=a.samples)
    solved = sum(1 for r in log if r["verified"])
    reusers = sum(1 for r in log if r["reused"])
    print(f"\n=== UNIFIED RUN === solved {solved}/{len(log)} | reusers {reusers} | "
          f"graph {tm.stats()}", file=sys.stderr)
    for r in log:
        print(f"  {r['name']:22} {'OK ' if r['verified'] else 'FAIL'} R={r['reward']:+.2f} "
              f"reuse={r['reused'] or '-'} wrote={r['written'] or '-'}", file=sys.stderr)


if __name__ == "__main__":
    main()
