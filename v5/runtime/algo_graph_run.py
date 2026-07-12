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


# ── store-gate: store only STANDALONE reusable ATOMS, deduped by behavior ──────────
# probe a def over mixed arg-tuples (int / list / graph / str). A genuine atom (build_adj) runs on
# SOME probe -> a fingerprint. A composite task-solution (bfs_order calling an undefined build_adj)
# errors on ALL probes -> None -> NOT stored (it's a task answer, not a reusable atom). This alone
# kills the 561-node flood: only atoms enter the graph, and identical atoms collapse to one.
_FP_ARGS = [(0,), (2,), (3,), (4,), (10,), (121,), ([1, 2, 3],), ([2, 3, 5, 7],), ("aba",),
            (3, [(0, 1, 2), (1, 2, 3)]),
            (4, [(0, 1, 2), (0, 2, 5), (1, 2, 1), (2, 3, 3)], 0),
            (4, [(0, 1, 2), (0, 2, 5), (1, 2, 1)], 0, 3)]


def _code_fingerprint(src: str, fn_name: str, timeout: float = 5.0) -> str | None:
    """md5 of the def's outputs over the probe set (STANDALONE — no deps). None if it never
    evaluates (all-ERR) => a composite / non-atom, which we DON'T store as a reusable node."""
    import hashlib
    import subprocess
    import tempfile
    from pathlib import Path
    harness = "\n".join([
        src, "", "if True:", f"    _p = {_FP_ARGS!r}", "    _o = []", "    for _a in _p:",
        "        try:", f"            _o.append(repr({fn_name}(*_a)))",
        "        except Exception as _e:", "            _o.append('ERR:' + type(_e).__name__)",
        "    print('FP', '\\t'.join(_o))"])
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "fp.py"; p.write_text(harness, encoding="utf-8")
        try:
            r = subprocess.run([sys.executable, "-I", str(p)], cwd=td, capture_output=True,
                               text=True, encoding="utf-8", errors="replace", timeout=timeout)
        except subprocess.TimeoutExpired:
            return None
    line = next((l for l in reversed((r.stdout or "").splitlines()) if l.startswith("FP")), "")
    if not line:
        return None
    sig = line[3:]
    if all(t.startswith("ERR:") for t in sig.split("\t")):
        return None                              # never ran on any probe -> not a standalone atom
    return hashlib.md5(sig.encode()).hexdigest()


def writeback_atoms(tm: TotalMemory, code: str, task) -> tuple[list, list]:
    """Store-gate: from a verified solution, store ONLY standalone reusable atoms, deduped by
    behavioral fingerprint, keyed by PURPOSE (sig) not the birth-task ctx (so cross-task retrieval
    can surface them). Returns (written_names, skipped_names). Composites/task-solutions are skipped."""
    seen = {r.get("content") for r in tm.impls.records.values()
            if r.get("form") == "code" and r.get("content")}
    written, skipped = [], []
    for name, src in _top_defs(code).items():
        fp = _code_fingerprint(src, name)
        if fp is None or fp in seen:             # composite (None) or behavioral duplicate -> skip
            skipped.append(name); continue
        purpose = f"reusable helper {_sig(src) or name}"     # PURPOSE-keyed ctx (not the task text)
        iid = tm.write(goal=purpose, old="", new=src, trace=purpose, verified=True,
                       task_id=name, form="code", content=fp)
        if iid:
            written.append(name); seen.add(fp)
    return written, skipped


# ═══════════════════════════════════════════════════════════════════════════════
# TASK FAMILIES — task-agnostic verify. A task has .name/.text and verifies a solution by execution.
#   curriculum (algo_curriculum) : verify_fn over generated (args, expected) cases
#   MBPP+ (open-source, HARDER)  : run the problem's assert tests  (inline-unsolvable -> compose)
# ═══════════════════════════════════════════════════════════════════════════════

def verify_asserts(code: str, tests: list[str], setup: str = "", timeout: float = 10.0) -> bool:
    """Run `code` + the problem's `assert` tests in an isolated subprocess. True iff ALL pass
    (a sentinel print only reached if no assertion/exception fired)."""
    import subprocess
    import tempfile
    from pathlib import Path
    if not code or "def " not in code:
        return False
    harness = "\n".join([setup, code, *tests, "print('__ALLPASS__')"])
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "t.py"; p.write_text(harness, encoding="utf-8")
        try:
            r = subprocess.run([sys.executable, "-I", str(p)], cwd=td, capture_output=True,
                               text=True, encoding="utf-8", errors="replace", timeout=timeout)
        except subprocess.TimeoutExpired:
            return False
    return "__ALLPASS__" in (r.stdout or "")


class MBPPTask:
    """One MBPP/MBPP+ problem: prompt + entry-point name + assert tests."""
    def __init__(self, name: str, text: str, tests: list[str], setup: str = ""):
        self.name, self.text, self.tests, self.setup = name, text, tests, setup

    def verify(self, code: str, deps_code: str = "") -> bool:
        return verify_asserts((deps_code + "\n" + code) if deps_code else code, self.tests, self.setup)


def load_mbpp(limit: int = 0, sanitized: bool = True, split: str = "test",
              repo: str = "google-research-datasets/mbpp"):
    """Load MBPP from HF datasets (molab has network; the local selftest uses a hardcoded MBPPTask).
    Uses the NAMESPACED repo id — the bare `mbpp` alias no longer resolves in newer datasets/hf_hub.
    Defensive about field names across the sanitized/full configs."""
    from datasets import load_dataset
    ds = load_dataset(repo, "sanitized" if sanitized else "full", split=split)
    tasks = []
    for r in ds:
        text = r.get("prompt") or r.get("text") or ""
        tests = r.get("test_list") or []
        setup = "\n".join(r.get("test_imports") or []) or (r.get("test_setup_code") or "")
        name = r.get("entry_point") or ""
        if not name:                                    # derive entry point from the first assert
            m = re.search(r"assert\s+([A-Za-z_]\w*)\s*\(", " ".join(tests))
            name = m.group(1) if m else ""
        if name and tests and text:
            tasks.append(MBPPTask(name, f"{text}\nWrite `{name}(...)` (only the function).", tests, setup))
        if limit and len(tasks) >= limit:
            break
    return tasks


def _task_verify(task, code: str, deps_code: str = "") -> bool:
    """Task-agnostic verification. MBPPTask carries its own asserts; a curriculum Task verifies via
    generated cases (fixed eval seeds)."""
    if hasattr(task, "verify"):
        return task.verify(code, deps_code)
    from v5.runtime.algo_curriculum import cases
    return bool(code) and verify_fn(code, task.name, cases(task, range(700, 712)), deps_code)[0] >= 0.999


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


def _retrieve_atoms(tm: TotalMemory, task, k: int):
    hit = tm.read(goal=task.text, span=task.text, k_impl=k)
    advertised = [(r.get("task_id") or "", r["new"]) for r in hit.impls
                  if r.get("form") == "code" and r.get("task_id") and r.get("new")]
    return advertised, [n for n, _ in advertised]


def _called_and_deps(code: str, advertised, adv_names):
    defined = _def_names(code)
    called = [n for n in adv_names if n not in defined and re.search(rf"\b{re.escape(n)}\s*\(", code)]
    return called, "\n\n".join(c for n, c in advertised if n in called)


def solve_with_memory(tm: TotalMemory, gen_fn, task, k: int = 6, samples: int = 1,
                      writeback: bool = True):
    """One task: retrieve atoms -> author -> verify by execution (task-agnostic) -> reward -> write
    back atoms (store-gated). Works for the curriculum AND MBPP (task.verify)."""
    advertised, adv_names = _retrieve_atoms(tm, task, k)
    best = ("", [], False)                                   # code, called, verified
    for gen in gen_fn([_author_prompt(task, advertised)] * samples):
        code = _extract_code(gen)
        called, deps = _called_and_deps(code, advertised, adv_names)
        if _task_verify(task, code, deps):
            best = (code, called, True); break               # first verified wins (all-or-nothing)
        if not best[0]:
            best = (code, called, False)
    code, called, verified = best
    _, used = grounded_code(code, adv_names)
    new_helpers = [n for n in _top_defs(code) if n != task.name and n not in adv_names]
    R, bd = code_reward(verified, composed_used=used,
                        authored_new_verified=len(new_helpers) if verified else 0)
    written, skipped = writeback_atoms(tm, code, task) if (verified and writeback) else ([], [])
    return dict(name=task.name, verified=verified, reward=round(R, 3), reused=used,
                authored=new_helpers, written=written, skipped=skipped, breakdown=bd)


def run_stream(tm: TotalMemory, gen_fn, tasks=None, k=6, samples=1):
    if tasks is None:
        from v5.runtime.algo_curriculum import STREAM
        tasks = STREAM
    return [solve_with_memory(tm, gen_fn, t, k=k, samples=samples) for t in tasks]


def load_tasks(kind: str = "curriculum", limit: int = 0):
    """Select the task family. curriculum = synthetic graph-algo (inline-solvable — a control);
    mbpp = open-source Python problems (HARDER, inline-unsolvable -> composition necessary)."""
    if kind == "mbpp":
        return load_mbpp(limit=limit or 200)
    from v5.runtime.algo_curriculum import STREAM
    return list(STREAM)


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

        # solve neighbors_of: must RETRIEVE build_adj, COMPOSE it, verify, reward>bare. The solution
        # (a composite calling build_adj) is a task-answer, NOT a reusable atom -> store-gate SKIPS it.
        res = solve_with_memory(tm, _stub_gen, ntask)
        assert res["verified"], f"should solve: {res}"
        assert res["reused"] == ["build_adj"], f"should compose the retrieved build_adj: {res}"
        assert res["reward"] > 1.0, f"compose reward should beat bare solve: {res}"
        assert res["written"] == [] and res["skipped"] == ["neighbors_of"], \
            f"store-gate skips the composite task-solution: {res}"
        print(f"  [2] retrieve->compose build_adj->verify->reward {res['reward']:+.2f} "
              f"(composite skipped by gate) -> PASS")

        # graph did NOT grow from a pure-reuse solve (no pollution): still just the atom build_adj
        assert tm.stats()["impls"] == 1, f"no pollution — only the atom persists: {tm.stats()}"
        print(f"  [3] store-gate: pure-reuse solve adds NO node (no 561-flood) -> PASS")

        # inline baseline scores LESS than compose (GRPO would demote it)
        tm_empty = TotalMemory(td + "_e", mode="flat", embed_fn=make_fake_embedder())
        res_inline = solve_with_memory(tm_empty, _stub_gen, ntask, writeback=False)
        assert res_inline["verified"] and res_inline["reused"] == [], res_inline
        assert res_inline["reward"] < res["reward"], \
            f"inline ({res_inline['reward']}) must score below compose ({res['reward']})"
        print(f"  [4] inline-solve reward {res_inline['reward']:+.2f} < compose {res['reward']:+.2f} "
              f"(GRPO demotes inline) -> PASS")

        # store-gate directly: a solution that DEFINES a new atom + a composite stores ONLY the atom,
        # deduped — the fix for the 561-node/1-concept pollution that killed reuse.
        tm_g = TotalMemory(td + "_g", mode="flat", embed_fn=make_fake_embedder())
        sol = _BUILD_ADJ + "\n\ndef bfs_order(n, edges, s):\n    adj = build_adj(n, edges)\n    return sorted(adj)"
        w, sk = writeback_atoms(tm_g, sol, BY_NAME["bfs_order"])
        assert "build_adj" in w and "bfs_order" in sk, f"store atom, skip composite: {w} {sk}"
        assert tm_g.stats()["impls"] == 1, "only the standalone atom stored"
        w2, sk2 = writeback_atoms(tm_g, sol, BY_NAME["bfs_order"])
        assert w2 == [] and "build_adj" in sk2, f"behavioral dedup on re-store: {w2} {sk2}"
        print("  [5] store-gate: stores standalone atom, skips composite, dedups re-store -> PASS")

        # MBPP task path (open-source harder family): assert-based verify + task.verify + solve loop,
        # no network (hardcoded problem). Proves the loop is task-family-agnostic.
        mtask = MBPPTask("add_one", "Write `add_one(x)` returning x+1.",
                         ["assert add_one(3) == 4", "assert add_one(-1) == 0"])
        assert mtask.verify("def add_one(x):\n    return x + 1"), "correct solution passes asserts"
        assert not mtask.verify("def add_one(x):\n    return x + 2"), "wrong solution fails asserts"

        def _mstub(prompts):
            return ["```python\ndef add_one(x):\n    return x + 1\n```"] * len(prompts)
        tm_m = TotalMemory(td + "_m", mode="flat", embed_fn=make_fake_embedder())
        res_m = solve_with_memory(tm_m, _mstub, mtask)
        assert res_m["verified"], f"MBPP task solved via the same loop: {res_m}"
        print("  [6] MBPP path: verify_asserts + task.verify + task-agnostic solve loop -> PASS")

    print("\n  ALGO_GRAPH_RUN SELFTEST -> PASS")
    return True


# ═══════════════════════════════════════════════════════════════════════════════
# GRPO TRAIN (LoRA) — reuse derive_rl's loop shape (curriculum ramp + group advantages), reward =
# code_reward on retrieve->author->verify, write-back grows the graph across steps (compounding).
# ═══════════════════════════════════════════════════════════════════════════════

def train(model_name, steps, K, lr, r_lora, seed, layers, eval_every, ent_coef, temperature,
          chunk, root, tasks=None):
    import os
    import random
    import torch
    import torch.nn as nn
    from transformers import AutoTokenizer
    from peft import LoraConfig, get_peft_model
    from v5.lm_loader import load_frozen_lm
    from v5.memory.store import make_mpnet_embedder
    from v5.runtime.derive_rl import advantages
    if tasks is None:
        from v5.runtime.algo_curriculum import STREAM
        tasks = list(STREAM)

    trust = os.environ.get("V5_LM_TRUST_REMOTE_CODE", "0").lower() in ("1", "true", "yes")
    tm = TotalMemory(root, mode="concept", embed_fn=make_mpnet_embedder())
    base = load_frozen_lm(model_name)
    tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=trust)
    dev = next(base.parameters()).device
    leaf = sorted({n.split(".")[-1] for n, m in base.named_modules()
                   if isinstance(m, nn.Linear) and ".layers." in n
                   and not any(x in n.lower() for x in ("lm_head", "embed"))})
    cfg = LoraConfig(r=r_lora, lora_alpha=2 * r_lora, lora_dropout=0.0, task_type="CAUSAL_LM",
                     target_modules=leaf, layers_to_transform=layers)
    model = get_peft_model(base, cfg); model.train()
    trainable = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(trainable, lr=lr)
    print(f"LoRA r={r_lora} layers={layers} | trainable={sum(p.numel() for p in trainable):,} | "
          f"graph root={root}", flush=True)

    rng = random.Random(seed)
    stages = sorted({t.stage for t in tasks if hasattr(t, "stage")})   # curriculum has stages; MBPP doesn't
    held = list(tasks)[:24]

    def sample_task(step):
        if not stages:                                   # MBPP etc.: uniform sample (no stage ramp)
            return rng.choice(tasks)
        # curriculum ramp (derive_rl's p_hard idea): unlock harder stages as training progresses.
        frac = min(1.0, step / max(1, 0.6 * steps))
        max_stage = min(max(stages), int(frac * (max(stages) + 1)))
        return rng.choice([t for t in tasks if getattr(t, "stage", 0) <= max_stage])

    def encode(prompt):
        m = [{"role": "user", "content": prompt}]
        kw = dict(add_generation_prompt=True, return_tensors="pt", return_dict=True)
        try:
            enc = tok.apply_chat_template(m, enable_thinking=False, **kw)
        except TypeError:
            enc = tok.apply_chat_template(m, **kw)
        return enc["input_ids"].to(dev)

    def seq_logprob(pids, comp):
        full = torch.cat([pids, comp], dim=1)
        logits = model(full).logits[:, :-1]
        logp = torch.log_softmax(logits.float(), dim=-1)
        start = pids.shape[1] - 1
        span = logp[:, start:start + comp.shape[1]]
        sel = span.gather(-1, comp.unsqueeze(-1)).squeeze(-1).sum(-1)
        ent = -(span.exp() * span).sum(-1).mean()
        return sel, ent

    def score(task, code, advertised, adv_names):
        called, deps = _called_and_deps(code, advertised, adv_names)
        verified = _task_verify(task, code, deps)
        _, used = grounded_code(code, adv_names)
        new_h = [n for n in _top_defs(code) if n != task.name and n not in adv_names]
        r, _ = code_reward(verified, composed_used=used, authored_new_verified=len(new_h) if verified else 0)
        return r, verified, used

    @torch.no_grad()
    def evaluate():
        model.eval()
        solved = reused = 0
        for task in held:
            hit = tm.read(goal=task.text, span=task.text, k_impl=6)
            adv = [(r.get("task_id") or "", r["new"]) for r in hit.impls
                   if r.get("form") == "code" and r.get("task_id")]
            pids = encode(_author_prompt(task, adv))
            out = model.generate(pids, do_sample=False, max_new_tokens=420, pad_token_id=tok.eos_token_id)
            code = _extract_code(tok.decode(out[0, pids.shape[1]:], skip_special_tokens=True))
            r, ok, used = score(task, code, adv, [n for n, _ in adv])
            solved += ok; reused += bool(used)
        model.train()
        return solved / len(held), reused / len(held)

    bs, br = evaluate()
    print(f"[eval @0] solve={bs:.0%} reuse={br:.0%}", flush=True)
    for step in range(1, steps + 1):
        task = sample_task(step)
        hit = tm.read(goal=task.text, span=task.text, k_impl=6)
        advertised = [(r.get("task_id") or "", r["new"]) for r in hit.impls
                      if r.get("form") == "code" and r.get("task_id")]
        adv_names = [n for n, _ in advertised]
        pids = encode(_author_prompt(task, advertised))
        with torch.no_grad():
            outs = model.generate(pids, do_sample=True, temperature=temperature, top_p=0.95,
                                  max_new_tokens=420, num_return_sequences=K, pad_token_id=tok.eos_token_id)
        comp_all = outs[:, pids.shape[1]:]
        comps, rewards, best = [], [], None
        for k in range(K):
            comp = comp_all[k:k + 1]
            code = _extract_code(tok.decode(comp[0], skip_special_tokens=True))
            r, ok, used = score(task, code, advertised, adv_names)
            comps.append(comp); rewards.append(r)
            if ok and best is None:
                best = code
        if best:                                            # write-back grows the graph (gated: atoms only)
            writeback_atoms(tm, best, task)
        mean_r = sum(rewards) / K
        r_std = (sum((r - mean_r) ** 2 for r in rewards) / K) ** 0.5
        if r_std < 1e-9:
            if step % 20 == 0:
                print(f"[step {step:3}] {task.name:20} mean_r={mean_r:+.2f} r_std=0 SKIP", flush=True)
            continue
        advs = advantages(rewards)
        loss = 0.0
        for comp, a in zip(comps, advs):
            lp, ent = seq_logprob(pids, comp)
            loss = loss - a * lp - ent_coef * ent
        loss = loss / K
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        opt.step(); opt.zero_grad()
        if step % 20 == 0:
            print(f"[step {step:3}] {task.name:20} mean_r={mean_r:+.2f} r_std={r_std:.2f} "
                  f"graph={tm.stats()['impls']} nodes", flush=True)
        if step % eval_every == 0:
            s, r = evaluate()
            print(f"[eval @{step}] solve={s:.0%} reuse={r:.0%} (base {bs:.0%}/{br:.0%})", flush=True)
    fs, fr = evaluate()
    print(f"\n=== RL DONE === solve {bs:.0%}->{fs:.0%} | reuse {br:.0%}->{fr:.0%} | graph {tm.stats()}",
          flush=True)
    model.save_pretrained("artifacts/algo_graph_lora")
    print("  LoRA saved -> artifacts/algo_graph_lora", flush=True)


def inspect_graph(tm: TotalMemory, sample_tasks, k: int = 6):
    """Dump the stored graph + show what tm.read RETRIEVES for sample tasks — are the retrieved nodes
    usable atoms, or junk / same-task solutions? Answers 'retrieved but unusable?' directly."""
    from collections import Counter
    recs = list(tm.impls.records.values())
    forms = Counter(r.get("form", "") for r in recs)
    print(f"\n=== GRAPH: {len(recs)} nodes | forms {dict(forms)} | {tm.stats().get('concepts')} concepts ===")
    code = [r for r in recs if r.get("form") == "code"]
    by_id = Counter(r.get("task_id", "") for r in code)
    # OLD curriculum pollution vs MBPP-era atoms
    try:
        from v5.runtime.algo_curriculum import BY_NAME
        curric = set(BY_NAME) | {"build_adj", "build_edge_weight_matrix"}
    except Exception:
        curric = set()
    old = sum(1 for r in code if r.get("task_id") in curric)
    print(f"  code nodes {len(code)} | distinct task_ids {len(by_id)} | "
          f"OLD-curriculum-pollution {old} | top ids {by_id.most_common(8)}")
    print("\n  -- sample stored code nodes (id :: first line) --")
    for r in code[:18]:
        first = ((r.get("new") or "").strip().splitlines() or [""])[0]
        print(f"    [{(r.get('task_id') or '?'):26}] {first[:74]}")

    print("\n=== RETRIEVAL for sample tasks — ARE THE RETRIEVED NODES USABLE? ===")
    for task in sample_tasks:
        hit = tm.read(goal=task.text, span=task.text, k_impl=k)
        print(f"\n[{task.name}] retrieved {len(hit.impls)} (fit-gated):")
        for r in hit.impls:
            first = ((r.get("new") or "").strip().splitlines() or [""])[0]
            print(f"    form={r.get('form',''):5} id={(r.get('task_id') or '?'):22} :: {first[:70]}")
        if not hit.impls:
            print("    (nothing cleared MIN_FIT — retrieval delivered NOTHING)")


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
    ap.add_argument("--inspect", action="store_true", help="dump the stored graph + show retrieval for sample tasks")
    ap.add_argument("--train", action="store_true", help="GRPO LoRA training (curriculum + write-back)")
    ap.add_argument("--model", default="Qwen/Qwen2.5-3B")
    ap.add_argument("--root", default="data/algo_graph", help="TotalMemory root (persists the graph)")
    ap.add_argument("--samples", type=int, default=4)
    ap.add_argument("--chunk", type=int, default=8)
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--k", type=int, default=8, help="GRPO rollouts per task")
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--r-lora", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--layers", type=int, nargs="+", default=[20, 22, 24, 26, 28])
    ap.add_argument("--eval-every", type=int, default=50)
    ap.add_argument("--ent-coef", type=float, default=0.005)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--tasks", default="curriculum", choices=["curriculum", "mbpp"],
                    help="curriculum (synthetic graph algos, inline-solvable control) or mbpp (harder)")
    ap.add_argument("--task-limit", type=int, default=0, help="cap number of tasks (0 = default)")
    a = ap.parse_args()
    if a.selftest:
        sys.exit(0 if _selftest() else 1)
    tasks = load_tasks(a.tasks, a.task_limit)
    print(f"tasks: {a.tasks} ({len(tasks)} loaded)", file=sys.stderr)
    if a.inspect:
        from v5.memory.store import make_mpnet_embedder
        tm = TotalMemory(a.root, mode="concept", embed_fn=make_mpnet_embedder())
        inspect_graph(tm, tasks[:6])
        return
    if a.train:
        train(a.model, a.steps, a.k, a.lr, a.r_lora, a.seed, a.layers, a.eval_every,
              a.ent_coef, a.temperature, a.chunk, a.root, tasks=tasks)
        return
    from v5.memory.store import make_mpnet_embedder
    tm = TotalMemory(a.root, mode="concept", embed_fn=make_mpnet_embedder())
    log = run_stream(tm, _real_gen_fn(a.model, a.chunk), tasks=tasks, samples=a.samples)
    solved = sum(1 for r in log if r["verified"])
    reusers = sum(1 for r in log if r["reused"])
    print(f"\n=== UNIFIED RUN === solved {solved}/{len(log)} | reusers {reusers} | "
          f"graph {tm.stats()}", file=sys.stderr)
    for r in log:
        print(f"  {r['name']:22} {'OK ' if r['verified'] else 'FAIL'} R={r['reward']:+.2f} "
              f"reuse={r['reused'] or '-'} wrote={r['written'] or '-'}", file=sys.stderr)


if __name__ == "__main__":
    main()
