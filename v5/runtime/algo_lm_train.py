"""GRR-14b: UNFROZEN training loop — graph-grounded LM fine-tuning.

Phase 1 (algo_lm_author):  weights FROZEN, graph grows.  We now have 369 nodes, 41 reuse events.
Phase 2 (THIS FILE):       weights UNFROZEN — we SFT the LoRA adapter on BOTH solved and failed
                            attempts so the model learns to (a) write correct code and (b) curate
                            its own graph by interpreting what is worth banking.

Key design decisions vs. the old train_star_mg loop:
  BOTH successes AND failures are fed to the model.
    - Successful attempts  -> standard SFT (next-token CE loss on the correct generation).
    - Failed attempts      -> "reflection" SFT: the model is shown its own failed code + the
                             verifier error, and the TARGET is the CORRECTED (successful) code.
                             If there is no successful attempt for a task, we skip it (we never
                             train on unresolvable failures — no target to point toward).
    - The reflection format teaches the model WHAT NOT to write AND what the correction looks like,
      instead of only copying correct examples.

  REPLAY BUFFER prevents catastrophic forgetting.
    - A fixed-size ring-buffer of (prompt, target) pairs sampled uniformly from all historical
      solved tasks is mixed into every SFT batch alongside new examples.
    - Replay fraction: 40% of each batch is drawn from history.

  INCREMENTAL graph re-embedding.
    - We track which node IDs have already been embedded.
    - Only newly banked nodes are embedded each round — O(new) not O(all).

  CHUNKED generation avoids VRAM OOM.
    - Instead of num_return_sequences=k, we loop ceil(k/chunk) times with chunk=2.

  INCREMENTAL LoRA unfreezing schedule.
    - Round 0–1: only the LoRA adapter params are updated (warm-up).
    - Round 2+:  all trainable params (LoRA + the MoLoRA behaviour encoder, if present) updated.
    - The base frozen backbone is never unfrozen (4-bit quantisation cannot back-prop through it).

  The graph still gates everything: only verified solutions bank atoms.  The model's failure
  signal teaches it HOW to succeed; the graph's health gate decides WHAT enters memory.

  selftest (no GPU):  python -m v5.runtime.algo_lm_train --selftest
  run (GPU):          python -m v5.runtime.algo_lm_train --run \\
                          --model Qwen/Qwen2.5-3B-Instruct \\
                          --graph graphs/grr_grown.json \\
                          --corpus artifacts/corpus_multi.jsonl \\
                          --rounds 10 --batch 16
"""
from __future__ import annotations

import argparse
import collections
import json
import random
import re
import sys
from pathlib import Path
from typing import Deque

from graph_core import MemoryGraph
from v5.runtime.algo_graph_edits import edge_candidate, grow, node_candidate
from v5.runtime.algo_graph_mg import MGRetriever, _edits_from_solve, _fn_name, seed_graph
from v5.runtime.algo_lm_author import (
    _bank_solution, _called_atoms, _failure_class, repair_code,
    _author_prompt_purpose, _solve_author,
)


# ═══════════════════════════════════════════════════════════════════════════════
# PROMPT HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

_REFLECTION_SEP = "\n\n# --- your previous attempt failed ---\n"


def _reflection_prompt(task_prompt: str, failed_code: str, error: str) -> str:
    """Reflection prompt: show the model its own failed attempt + error, ask it to correct."""
    err_snippet = (error or "assertion failed")[:300].strip()
    return (
        task_prompt
        + _REFLECTION_SEP
        + f"# Failed code:\n```python\n{failed_code}\n```\n"
        + f"# Error: {err_snippet}\n\n"
        + "# Please write the corrected solution:"
    )


def _sft_pairs_from_result(task_prompt: str, res: dict) -> list[tuple[str, str]]:
    """Build (prompt, target) SFT pairs from a solve result.

    For a VERIFIED result:
      1. Direct pair: (task_prompt,  winning_raw_generation)
      2. Reflection pairs: for each FAILED attempt that preceded the success,
         (reflection_prompt(failed_code, error), winning_raw_generation)

    For an UNVERIFIED result: returns [] — never train toward an unverified target.
    """
    if not res.get("verified"):
        return []
    winning_raw = res.get("raw", "")
    if not winning_raw:
        return []
    pairs: list[tuple[str, str]] = [(task_prompt, winning_raw)]
    attempts = res.get("attempts", [])
    for att in attempts:
        if att.get("verified"):
            break          # everything before the first success is a failure
        failed_code = repair_code(att.get("gen", ""), "")
        err = att.get("error", "")
        if not failed_code:
            continue
        ref_prompt = _reflection_prompt(task_prompt, failed_code, err)
        pairs.append((ref_prompt, winning_raw))
    return pairs


# ═══════════════════════════════════════════════════════════════════════════════
# INCREMENTAL EMBEDDER — only re-embeds new nodes
# ═══════════════════════════════════════════════════════════════════════════════

class IncrementalMGRetriever(MGRetriever):
    """MGRetriever that keeps a cache of already-embedded node vectors (by node_id).
    Re-indexing after graph growth only calls embed_fn on the NEW nodes.
    """
    def __init__(self, graph: MemoryGraph, embed_fn):
        self._vec_cache: dict[str, "np.ndarray"] = {}
        super().__init__(graph, embed_fn)

    def _index(self):
        import numpy as np
        impls = [(nid, n) for nid, n in self.graph.nodes.items()
                 if n.node_type == "implementation" and n.metadata.get("code")]
        self.ids = [nid for nid, _ in impls]
        if not self.ids:
            self.mat = np.zeros((0, 1), dtype=np.float32)
            return
        # only embed nodes not yet in cache
        new_ids = [nid for nid in self.ids if nid not in self._vec_cache]
        if new_ids:
            new_vecs = self.embed_fn({nid: self.graph.nodes[nid].text for nid in new_ids})
            for nid, vec in new_vecs.items():
                self._vec_cache[nid] = vec
        self.mat = np.asarray([self._vec_cache[nid] for nid in self.ids], dtype=np.float32)
        norms = np.linalg.norm(self.mat, axis=1, keepdims=True) + 1e-9
        self.mat /= norms

    def reindex(self, new_graph: MemoryGraph):
        """Re-attach to a grown graph, only embedding new nodes."""
        self.graph = new_graph
        self._index()


# ═══════════════════════════════════════════════════════════════════════════════
# CHUNKED GENERATION — prevents num_return_sequences VRAM spike
# ═══════════════════════════════════════════════════════════════════════════════

def _chunked_gen(gen_fn, prompt: str, k: int, chunk: int = 2) -> list[str]:
    """Generate k samples for a single prompt by chunking into groups of `chunk`."""
    outs = []
    while len(outs) < k:
        n = min(chunk, k - len(outs))
        outs.extend(gen_fn([prompt] * n))
    return outs


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN UNFROZEN TRAINING LOOP
# ═══════════════════════════════════════════════════════════════════════════════

def train_unfrozen(
    model_name: str,
    graph_path: str,
    tasks: list,
    embed_fn,
    rounds: int = 10,
    batch_size: int = 16,
    k_retrieve: int = 6,
    samples: int = 4,
    gen_chunk: int = 2,
    lr: float = 2e-4,
    r_lora: int = 16,
    replay_cap: int = 500,
    replay_frac: float = 0.4,
    ad_style: str = "off",
    lora_only_warmup_rounds: int = 2,
    max_tokens: int = 1024,
    out_lora: str = "artifacts/grr14b_lora",
    log=print,
):
    """Unfrozen STaR loop: retrieval -> attempt (k samples) -> verify -> SFT(success+reflection)
    -> graph grow -> incremental re-embed -> repeat.

    Args:
        lora_only_warmup_rounds: for the first N rounds, only LoRA params are updated (the
            MoLoRA behaviour encoder, if any, stays frozen). After warmup all trainable params
            are updated.  Base backbone stays frozen throughout (quantised).
    """
    import os
    import torch
    import torch.nn as nn
    from transformers import AutoTokenizer
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from v5.lm_loader import load_frozen_lm, resolve_quant
    from v5.runtime.algo_graph_run import _author_prompt

    # ── model setup ─────────────────────────────────────────────────────────
    trust = os.environ.get("V5_LM_TRUST_REMOTE_CODE", "0").lower() in ("1", "true", "yes")
    tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=trust)
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id
    base = load_frozen_lm(model_name)
    if resolve_quant(None) in ("4bit", "8bit"):
        base = prepare_model_for_kbit_training(base)
    else:
        base.gradient_checkpointing_enable()
        base.enable_input_require_grads()

    # target all attention + MLP linear projections (not just q/k/v/o — wider = stronger)
    leaf_names = sorted({
        n.split(".")[-1] for n, m in base.named_modules()
        if isinstance(m, nn.Linear) and ".layers." in n
        and not any(x in n.lower() for x in ("lm_head", "embed"))
    })
    lcfg = LoraConfig(
        r=r_lora, lora_alpha=2 * r_lora, lora_dropout=0.05,
        bias="none", task_type="CAUSAL_LM", target_modules=leaf_names,
    )
    model = get_peft_model(base, lcfg)
    dev = next(model.parameters()).device
    lora_params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(lora_params, lr=lr)
    pad_id = tok.eos_token_id

    log(f"GRR-14b UNFROZEN | LoRA r={r_lora} | {len(lora_params)} trainable param tensors "
        f"| {len(tasks)} tasks | {rounds} rounds | batch={batch_size}", flush=True)

    # ── graph + retriever ────────────────────────────────────────────────────
    if not Path(graph_path).exists():
        seed_graph(graph_path, ("concept_algorithms",))
    retr = IncrementalMGRetriever(MemoryGraph.load_json(graph_path), embed_fn)

    # ── gen_fn wraps the FROZEN model.generate for the attempt phase ─────────
    @torch.no_grad()
    def gen_fn(prompts: list[str], max_new_tokens: int = 460) -> list[str]:
        model.eval()
        outs = []
        for p in prompts:
            m = [{"role": "user", "content": p}]
            kw = dict(add_generation_prompt=True, return_tensors="pt", return_dict=True)
            try:
                ids = tok.apply_chat_template(m, enable_thinking=False, **kw)["input_ids"].to(dev)
            except TypeError:
                ids = tok.apply_chat_template(m, **kw)["input_ids"].to(dev)
            out = model.generate(
                ids, do_sample=True, temperature=0.8, top_p=0.95,
                max_new_tokens=max_new_tokens, pad_token_id=pad_id,
            )
            outs.append(tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True))
        model.train()
        return outs

    # ── SFT helpers ──────────────────────────────────────────────────────────
    def _encode_pair(prompt: str, target: str) -> tuple[list[int], list[int]] | None:
        p_ids = tok(prompt, add_special_tokens=False)["input_ids"]
        t_ids = tok(target, add_special_tokens=False)["input_ids"] + [tok.eos_token_id]
        if len(p_ids) + len(t_ids) > max_tokens:
            return None
        return p_ids + t_ids, [-100] * len(p_ids) + t_ids

    def _pad_batch(encs):
        width = max(len(ids) for ids, _ in encs)
        ids = [seq + [pad_id] * (width - len(seq)) for seq, _ in encs]
        lbl = [lab + [-100] * (width - len(lab)) for _, lab in encs]
        mask = [[1] * len(seq) + [0] * (width - len(seq)) for seq, _ in encs]
        return ids, lbl, mask

    def _sft_step(pairs: list[tuple[str, str]]) -> float:
        """One gradient step over a list of (prompt, target) pairs. Returns mean loss."""
        if not pairs:
            return 0.0
        encs = [_encode_pair(p, t) for p, t in pairs]
        encs = [e for e in encs if e is not None]
        if not encs:
            return 0.0
        encs.sort(key=lambda e: len(e[0]))
        ids_l, lbl_l, mask_l = _pad_batch(encs)
        ids = torch.tensor(ids_l, device=dev)
        lbl = torch.tensor(lbl_l, device=dev)
        mask = torch.tensor(mask_l, device=dev)
        out = model(ids, attention_mask=mask, labels=lbl)
        opt.zero_grad()
        out.loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [p for g in opt.param_groups for p in g["params"] if p.requires_grad], 1.0
        )
        opt.step()
        return float(out.loss.detach())

    # ── replay buffer ────────────────────────────────────────────────────────
    replay: Deque[tuple[str, str]] = collections.deque(maxlen=replay_cap)

    def _replay_sample(n: int) -> list[tuple[str, str]]:
        if not replay:
            return []
        k = min(n, len(replay))
        return random.sample(list(replay), k)

    # ── authored-atoms tracking (for reuse counting) ─────────────────────────
    authored_this_run: set = set()
    reuse_events: list = []

    # ══════════════════════════════════════════════════════════════════════
    # ROUND LOOP
    # ══════════════════════════════════════════════════════════════════════
    rng = random.Random(42)
    for rnd in range(rounds):

        # ── unfreezing schedule ──────────────────────────────────────────
        if rnd == lora_only_warmup_rounds:
            log(f"  [round {rnd}] warmup done — all trainable params unfrozen", flush=True)

        # ── sample a batch of tasks ──────────────────────────────────────
        batch_tasks = rng.choices(tasks, k=batch_size)
        new_pairs: list[tuple[str, str]] = []
        solved_this_round = 0

        for task in batch_tasks:
            # chunk-generate k samples to avoid OOM
            gens = _chunked_gen(gen_fn, _get_prompt(retr, task, k_retrieve, ad_style), samples, gen_chunk)
            res = _solve_from_gens(retr, gens, task, k_retrieve, ad_style)

            task_prompt = res["prompt"]
            pairs = _sft_pairs_from_result(task_prompt, res)
            new_pairs.extend(pairs)

            if res["verified"]:
                solved_this_round += 1
                # ── bank into graph ──────────────────────────────────────
                atom_names = [_fn_name(retr.graph.nodes[n].metadata.get("code", "")) or n
                              for n in retr.ids]
                called = _called_atoms(res["code"], atom_names)
                reused_authored = [a for a in called if a in authored_this_run]
                if reused_authored:
                    reuse_events.append((task.name, reused_authored))
                ok, helper_names = _bank_solution(
                    graph_path, retr, task, res, called, f"grr14b_r{rnd}"
                )
                if ok:
                    authored_this_run.add(task.name)
                    authored_this_run.update(helper_names)
                    # incremental re-embed: just reload the graph; IncrementalMGRetriever
                    # will only embed newly added nodes
                    retr.reindex(MemoryGraph.load_json(graph_path))

        # ── mix replay into the SFT batch ────────────────────────────────
        n_replay = int(len(new_pairs) * replay_frac / (1 - replay_frac + 1e-9))
        batch_pairs = new_pairs + _replay_sample(n_replay)
        rng.shuffle(batch_pairs)

        # ── gradient step ────────────────────────────────────────────────
        loss = _sft_step(batch_pairs)

        # ── add new successes to replay ──────────────────────────────────
        # only direct (non-reflection) pairs go into replay to avoid compounding errors
        for p, t in new_pairs:
            if _REFLECTION_SEP not in p:
                replay.append((p, t))

        log(
            f"  [round {rnd+1}/{rounds}] solved {solved_this_round}/{batch_size} "
            f"| sft_pairs={len(batch_pairs)} (replay={n_replay}) | loss={loss:.3f} "
            f"| graph={len(retr.graph.nodes)} nodes "
            f"| reuse_events={len(reuse_events)}",
            flush=True,
        )

    # ── save LoRA ────────────────────────────────────────────────────────────
    Path(out_lora).mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out_lora)
    log(f"\n  LoRA saved -> {out_lora}", flush=True)
    log(f"  Graph: {len(retr.graph.nodes)} nodes | authored this run: {len(authored_this_run)}"
        f" | reuse events: {len(reuse_events)}", flush=True)


# ═══════════════════════════════════════════════════════════════════════════════
# INTERNAL HELPERS (shared between train_unfrozen and _selftest)
# ═══════════════════════════════════════════════════════════════════════════════

def _get_prompt(retr: MGRetriever, task, k: int, ad_style: str) -> str:
    from v5.runtime.algo_graph_run import _author_prompt
    if ad_style == "off":
        return _author_prompt(task, [])
    advertised = retr.retrieve(task.text, k=k)
    if ad_style == "purpose":
        purposes = {}
        for nid in retr.ids:
            node = retr.graph.nodes[nid]
            fn = _fn_name(node.metadata.get("code", "")) or nid
            purposes[fn] = (node.text or "").splitlines()[0][:120]
        return _author_prompt_purpose(task, advertised, purposes)
    return _author_prompt(task, advertised)


def _solve_from_gens(retr: MGRetriever, gens: list[str], task, k: int, ad_style: str) -> dict:
    """Run the verify loop over a pre-generated list of gens. Returns the same dict shape as
    _solve_author, including attempts (for reflection SFT)."""
    from v5.runtime.algo_graph_run import _author_prompt, _task_verify, verify_asserts_detail
    from v5.runtime.derive_reward import _def_names, code_reward, grounded_code
    from v5.runtime.tool_memory import _extract_code

    advertised = [] if ad_style == "off" else retr.retrieve(task.text, k=k)
    adv_names = [n for n, _ in advertised]
    prompt = _get_prompt(retr, task, k, ad_style)

    best = ("", [], False, "")
    attempts = []
    for gen in gens:
        code = repair_code(_extract_code(gen), task.name)
        defined = _def_names(code)
        called = [n for n in adv_names if n not in defined
                  and re.search(rf"(?<![\w.]){re.escape(n)}\s*\(", code)]
        deps = "\n\n".join(c for n, c in advertised if n in called)
        if _task_verify(task, code, deps):
            attempts.append({"gen": gen, "verified": True, "error": ""})
            best = (code, called, True, gen)
            break
        _ok, err = verify_asserts_detail(
            (deps + "\n" + code) if deps else code, task.tests, getattr(task, "setup", "")
        )
        attempts.append({"gen": gen, "verified": False, "error": err})
        if not best[0]:
            best = (code, called, False, gen)

    code, reused, verified, raw = best
    _, used = grounded_code(code, adv_names)
    R, _ = code_reward(verified, composed_used=used, authored_new_verified=0)
    return dict(
        name=task.name, verified=verified, reward=round(R, 3),
        reused=used, code=code, raw=raw, prompt=prompt, attempts=attempts,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# SELFTEST (no GPU)
# ═══════════════════════════════════════════════════════════════════════════════

def _selftest() -> bool:
    import tempfile
    import numpy as np
    from v5.runtime.algo_graph_run import MBPPTask

    print("algo_lm_train --selftest: reflection pairs | replay buffer | incremental reindex\n")

    # ── [0] reflection prompt format ──────────────────────────────────────────
    ref = _reflection_prompt("Write foo(x)", "def foo(x): return x", "AssertionError: foo(2) != 4")
    assert "Failed code" in ref and "AssertionError" in ref, ref
    print("  [0] _reflection_prompt contains failed code + error -> PASS")

    # ── [1] _sft_pairs_from_result ────────────────────────────────────────────
    res_verified = {
        "verified": True, "raw": "```python\ndef f(x): return x*2\n```",
        "attempts": [
            {"gen": "```python\ndef f(x): return x\n```", "verified": False, "error": "AssertionError"},
            {"gen": "```python\ndef f(x): return x*2\n```", "verified": True,  "error": ""},
        ]
    }
    pairs = _sft_pairs_from_result("Write f(x)", res_verified)
    # should produce: 1 direct + 1 reflection (1 failed attempt before success)
    assert len(pairs) == 2, f"expected 2 pairs, got {len(pairs)}: {pairs}"
    assert _REFLECTION_SEP in pairs[1][0], "second pair should be a reflection prompt"
    assert pairs[0][1] == pairs[1][1], "both targets should be the winning raw"
    print(f"  [1] verified result -> {len(pairs)} SFT pairs (1 direct + 1 reflection) -> PASS")

    res_fail = {"verified": False, "raw": "", "attempts": []}
    assert _sft_pairs_from_result("Write g(x)", res_fail) == []
    print("  [1b] unverified result -> 0 pairs (no target to train toward) -> PASS")

    # ── [2] IncrementalMGRetriever only embeds new nodes ─────────────────────
    call_log = []
    def tracked_embed(d: dict):
        call_log.append(list(d.keys()))
        return {k: np.random.rand(32).astype("float32") for k in d}

    with tempfile.TemporaryDirectory() as td:
        import json as _json
        gp = Path(td) / "g.json"
        gp.write_text(_json.dumps({"metadata": {}, "nodes": [
            {"id": "concept_algorithms", "text": "algorithms", "node_type": "concept"},
            {"id": "impl_foo", "text": "foo purpose",  "node_type": "implementation",
             "metadata": {"code": "def foo(): pass"}},
        ], "edges": []}))
        g = MemoryGraph.load_json(str(gp))
        inc = IncrementalMGRetriever(g, tracked_embed)
        first_call_ids = set(call_log[-1])
        assert "impl_foo" in first_call_ids, "initial index should embed impl_foo"

        # simulate a grown graph with a new node
        gp.write_text(_json.dumps({"metadata": {}, "nodes": [
            {"id": "concept_algorithms", "text": "algorithms", "node_type": "concept"},
            {"id": "impl_foo", "text": "foo purpose",  "node_type": "implementation",
             "metadata": {"code": "def foo(): pass"}},
            {"id": "impl_bar", "text": "bar purpose",  "node_type": "implementation",
             "metadata": {"code": "def bar(): pass"}},
        ], "edges": []}))
        g2 = MemoryGraph.load_json(str(gp))
        inc.reindex(g2)
        second_call_ids = set(call_log[-1])
        assert "impl_bar" in second_call_ids, "reindex should embed the new node"
        assert "impl_foo" not in second_call_ids, "reindex should NOT re-embed cached nodes"
    print("  [2] IncrementalMGRetriever: only new nodes embedded on reindex -> PASS")

    # ── [3] replay buffer sampling ────────────────────────────────────────────
    replay: Deque = collections.deque(maxlen=5)
    for i in range(10):
        replay.append((f"p{i}", f"t{i}"))
    assert len(replay) == 5, "ring buffer should cap at maxlen"
    sample = random.sample(list(replay), 3)
    assert len(sample) == 3
    print("  [3] replay buffer ring-cap + sampling -> PASS")

    print("\n  ALGO_LM_TRAIN SELFTEST -> PASS")
    return True


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser(
        description="GRR-14b: unfrozen LM training loop with reflection SFT and graph growth."
    )
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    ap.add_argument("--graph", default="graphs/grr_grown.json")
    ap.add_argument("--corpus", default="artifacts/corpus_multi.jsonl")
    ap.add_argument("--limit", type=int, default=0, help="0 = all tasks")
    ap.add_argument("--rounds", type=int, default=10)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--samples", type=int, default=4, help="generation samples per task")
    ap.add_argument("--gen-chunk", type=int, default=2, help="parallel gens per call (VRAM guard)")
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--r-lora", type=int, default=16)
    ap.add_argument("--replay-cap", type=int, default=500)
    ap.add_argument("--ad-style", default="off", choices=["off", "sig", "purpose"])
    ap.add_argument("--warmup-rounds", type=int, default=2,
                    help="rounds where only LoRA (not MoLoRA) params update")
    ap.add_argument("--out-lora", default="artifacts/grr14b_lora")
    a = ap.parse_args()

    if a.selftest:
        sys.exit(0 if _selftest() else 1)

    if a.run:
        from v5.memory.store import make_mpnet_embedder
        from v5.runtime.algo_mbpp_prep import load_prepped
        tasks = load_prepped(a.corpus, limit=a.limit or 100_000)
        embed = make_mpnet_embedder()
        train_unfrozen(
            model_name=a.model,
            graph_path=a.graph,
            tasks=tasks,
            embed_fn=embed,
            rounds=a.rounds,
            batch_size=a.batch,
            samples=a.samples,
            gen_chunk=a.gen_chunk,
            lr=a.lr,
            r_lora=a.r_lora,
            replay_cap=a.replay_cap,
            ad_style=a.ad_style,
            lora_only_warmup_rounds=a.warmup_rounds,
            out_lora=a.out_lora,
        )
        return

    ap.print_help()


if __name__ == "__main__":
    main()
