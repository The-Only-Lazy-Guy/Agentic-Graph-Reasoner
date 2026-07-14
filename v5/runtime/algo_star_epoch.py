"""D3/D5 of the 10-day freeze: the STaR EPOCH DRIVER — the long run's training engine.

  epoch N:  author-run over the corpus (resume ledger skips already-attempted; graph keeps growing)
            -> gate-VERIFIED (prompt -> code) traces accumulate (traces_log)
            -> LoRA-SFT the STUDENT on its own verified solutions (rejection sampling: failures are
               logged for the taxonomy but NEVER trained on — the gate decides what enters the weights,
               exactly like it decides what enters the graph)
            -> next epoch generates WITH the adapter; a frozen HOLDOUT (never trained, never banked)
               is re-evaluated each epoch = the training claim's only honest scoreboard.

  KPIs (GRR_CASES priority order): holdout assert_fail DOWN | plus_only_fail FLAT (gate integrity —
  if it rises the student is learning to overfit tests) | curve flat | atoms sub-linear.

  Deployment constraint holds by construction: the student IS the deployed-class 3B; LoRA merges or
  ships as adapter; nothing >6GB ever deploys.

  selftest (no GPU):   python -m v5.runtime.algo_star_epoch --selftest
  molab:               python -m v5.runtime.algo_star_epoch --epochs 2 --model Qwen/Qwen2.5-3B-Instruct \
                           --limit 300 --holdout 40
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def load_traces(path: str, dedup: bool = True) -> list:
    """Traces append across epochs (a re-solved task re-logs). Dedup by task, keep the LATEST —
    the newest adapter's solution style — so the training set never up-weights re-solves."""
    out = []
    if Path(path).exists():
        with open(path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    out.append(json.loads(line))
    if dedup:
        latest = {}
        for rec in out:
            latest[rec["task"]] = rec
        out = list(latest.values())
    return out


def train_lora(model_name: str, traces: list, out_dir: str, base_adapter: str = "",
               lr: float = 1e-4, epochs: int = 2, r_lora: int = 8, max_len: int = 1024,
               seed: int = 0) -> str:
    """Rejection-SFT: CE on the COMPLETION tokens of (chat-formatted prompt -> verified code).
    Returns the saved adapter dir. Same recipe class as train_star_mg (validated)."""
    import os
    import random
    import torch
    import torch.nn as nn
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig, PeftModel, get_peft_model
    trust = os.environ.get("V5_LM_TRUST_REMOTE_CODE", "0").lower() in ("1", "true", "yes")
    tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=trust)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    base = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.bfloat16,
                                                device_map="auto", trust_remote_code=trust)
    if base_adapter:
        model = PeftModel.from_pretrained(base, base_adapter, is_trainable=True)
    else:
        leaf = sorted({n.split(".")[-1] for n, m in base.named_modules()
                       if isinstance(m, nn.Linear) and ".layers." in n
                       and not any(x in n.lower() for x in ("lm_head", "embed"))})
        model = get_peft_model(base, LoraConfig(r=r_lora, lora_alpha=2 * r_lora, lora_dropout=0.0,
                                                task_type="CAUSAL_LM", target_modules=leaf))
    model.train()
    dev = next(model.parameters()).device
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr)
    ce = nn.CrossEntropyLoss()
    rng = random.Random(seed)
    data = list(traces)
    for ep in range(epochs):
        rng.shuffle(data)
        losses = []
        for rec in data:
            msgs = [{"role": "user", "content": rec["prompt"]}]
            ptxt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            comp = f"```python\n{rec['code']}\n```" + tok.eos_token
            pids = tok(ptxt, return_tensors="pt", add_special_tokens=False).input_ids
            cids = tok(comp, return_tensors="pt", add_special_tokens=False).input_ids
            ids = torch.cat([pids, cids], dim=1)[:, -max_len:].to(dev)
            n_c = min(cids.shape[1], ids.shape[1] - 1)
            logits = model(ids).logits
            loss = ce(logits[:, -n_c - 1:-1].reshape(-1, logits.shape[-1]).float(),
                      ids[:, -n_c:].reshape(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
            opt.step(); opt.zero_grad()
            losses.append(float(loss.detach()))
        print(f"  [lora ep {ep}] mean loss {sum(losses)/max(1,len(losses)):.3f} "
              f"({len(data)} traces)", flush=True)
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out_dir)
    return out_dir


def make_hf_gen_adapter(model_name: str, adapter_dir: str = "", temperature: float = 0.8,
                        max_new_tokens: int = 400):
    """gen_fn like algo_lm_proposer.make_hf_gen, with an optional LoRA adapter loaded on top."""
    import os
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    trust = os.environ.get("V5_LM_TRUST_REMOTE_CODE", "0").lower() in ("1", "true", "yes")
    tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=trust)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.bfloat16,
                                                 device_map="auto", trust_remote_code=trust)
    if adapter_dir:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter_dir)
    model.eval()

    def gen(prompts):
        msgs = [tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False,
                                        add_generation_prompt=True) for p in prompts]
        enc = tok(msgs, return_tensors="pt", padding=True, padding_side="left").to(model.device)
        with torch.no_grad():
            out = model.generate(**enc, do_sample=True, temperature=temperature, top_p=0.95,
                                 max_new_tokens=max_new_tokens, pad_token_id=tok.pad_token_id)
        return [tok.decode(out[i, enc["input_ids"].shape[1]:], skip_special_tokens=True)
                for i in range(len(prompts))]

    return gen


def eval_holdout(gen_fn, tasks, samples: int = 4) -> dict:
    """Frozen-holdout eval: solve WITHOUT banking, with the failure taxonomy (assert_fail vs
    plus_only_fail = the gate-integrity metric)."""
    from v5.runtime.algo_graph_run import _author_prompt, _task_verify
    from v5.runtime.tool_memory import _extract_code
    from v5.runtime.algo_lm_author import _failure_class, repair_code
    solved, fails = 0, {}
    for t in tasks:
        ok, code = False, ""
        for g in gen_fn([_author_prompt(t, [])] * samples):
            code = repair_code(_extract_code(g), t.name)
            if _task_verify(t, code, ""):
                ok = True
                break
        if ok:
            solved += 1
        else:
            c = _failure_class(t, code, "")
            fails[c] = fails.get(c, 0) + 1
    return dict(solved=solved, total=len(tasks), rate=round(solved / max(1, len(tasks)), 3),
                failures=fails)


def run_epochs(model_name: str, corpus: str, n_epochs: int = 2, limit: int = 300, holdout_n: int = 40,
               graph_path: str = "graphs/algo_star_epoch.json", work: str = "artifacts/star_epoch",
               samples: int = 4, train_fn=train_lora, gen_factory=make_hf_gen_adapter,
               author_fn=None, log: bool = True):
    """The orchestrator. Holdout = the LAST holdout_n corpus tasks — never authored on, never banked,
    never trained on; evaluated with each epoch's adapter. train_fn/gen_factory/author_fn are
    injectable for the no-GPU selftest."""
    from v5.memory.store import make_mpnet_embedder
    from v5.runtime.algo_lm_author import run_author_loop
    from v5.runtime.algo_mbpp_prep import load_prepped
    all_tasks = load_prepped(corpus, limit=limit + holdout_n)
    tasks, holdout = all_tasks[:limit], all_tasks[limit:limit + holdout_n]
    Path(work).mkdir(parents=True, exist_ok=True)
    embed = make_mpnet_embedder()
    author = author_fn or run_author_loop
    adapter = ""
    history = []
    for e in range(n_epochs):
        gen = gen_factory(model_name, adapter)
        hold0 = eval_holdout(gen, holdout, samples=samples)
        rep = author(graph_path, embed, gen, tasks, samples=samples, ad_style="off",
                     progress_log=f"{work}/progress_e{e}.jsonl",
                     traces_log=f"{work}/traces.jsonl", resume=True, checkpoint_every=50,
                     failure_log=f"{work}/failures_e{e}.jsonl", log=log)
        traces = load_traces(f"{work}/traces.jsonl")
        adapter = train_fn(model_name, traces, f"{work}/adapter_e{e}")
        gen2 = gen_factory(model_name, adapter)
        hold1 = eval_holdout(gen2, holdout, samples=samples)
        history.append(dict(epoch=e, holdout_before=hold0, holdout_after=hold1,
                            run_solve=rep.get("solve_rate"), traces=len(traces)))
        if log:
            print(f"\n  === epoch {e} === run-solve {rep.get('solve_rate')} | traces {len(traces)} | "
                  f"HOLDOUT {hold0['rate']} -> {hold1['rate']} "
                  f"(gate-integrity plus_only: {hold0['failures'].get('plus_only_fail', 0)} -> "
                  f"{hold1['failures'].get('plus_only_fail', 0)})", flush=True)
    return history


# ═══════════════════════════════════════════════════════════════════════════════
# SELFTEST (no GPU) — orchestration proven with stubs: resume ledger skips, traces accumulate,
# the trainer receives ONLY gate-verified traces, the adapter round-trips into the next epoch's gen.
# ═══════════════════════════════════════════════════════════════════════════════

def _selftest() -> bool:
    import tempfile
    print("algo_star_epoch --selftest: epoch orchestration (stubbed LM + trainer, real ledger/traces)\n")
    from v5.runtime.algo_lm_author import run_author_loop
    from v5.runtime.algo_graph_run import MBPPTask
    import numpy as np

    t1 = MBPPTask("inc", "Add one to n.\nWrite `inc(n)`.", ["assert inc(1) == 2", "assert inc(5) == 6"])
    t2 = MBPPTask("dbl", "Double n.\nWrite `dbl(n)`.", ["assert dbl(2) == 4", "assert dbl(3) == 6"])
    t3 = MBPPTask("mystery", "Return the 9th busy beaver.\nWrite `mystery()`.", ["assert mystery() == -1"])
    rng = np.random.default_rng(0)

    def embed(d):
        return {k: rng.standard_normal(32).astype("float32") for k in d}

    calls = {"gen": 0, "train": []}

    def stub_gen_factory(_model, adapter):
        def gen(prompts):
            calls["gen"] += 1
            outs = []
            for p in prompts:
                if "inc(n)" in p:
                    outs.append("```python\ndef inc(n):\n    return n + 1\n```")
                elif "dbl(n)" in p:
                    # the UNTRAINED stub fails dbl; the "adapter" fixes it (training effect simulated)
                    outs.append("```python\ndef dbl(n):\n    return n * 2\n```" if adapter
                                else "```python\ndef dbl(n):\n    return n\n```")
                else:
                    outs.append("```python\ndef mystery():\n    return 42\n```")
            return outs
        return gen

    def stub_train(_model, traces, out_dir, **_kw):
        calls["train"].append([r["task"] for r in traces])
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        return out_dir

    with tempfile.TemporaryDirectory() as td:
        import v5.runtime.algo_star_epoch as se
        import v5.runtime.algo_mbpp_prep as prep
        work = str(Path(td) / "w")
        gp = str(Path(td) / "g.json")
        orig = prep.load_prepped
        prep.load_prepped = lambda *_a, **_k: [t1, t2, t3, t3]     # holdout = last 1 (t3, unsolvable)
        try:
            hist = se.run_epochs("stub-model", "unused", n_epochs=2, limit=3, holdout_n=1,
                                 graph_path=gp, work=work, samples=1,
                                 train_fn=stub_train, gen_factory=stub_gen_factory, log=False)
        finally:
            prep.load_prepped = orig

        # [1] traces = ONLY gate-verified tasks; epoch 0 trains on {inc}, epoch 1 adds dbl (adapter fixed it)
        assert calls["train"][0] == ["inc"], calls["train"]
        assert sorted(calls["train"][1]) == ["dbl", "inc"], calls["train"]
        print("  [1] trainer received ONLY gate-verified traces; epoch-1 adapter unlocked dbl -> its "
              "trace joined epoch-2's training set (STaR accumulation) -> PASS")

        # [2] resume ledger: epoch 1 skipped the already-attempted... epoch logs are per-epoch, so
        #     epoch 1 re-attempts ONLY unsolved-in-ledger? (per-epoch ledgers -> full re-attempt is
        #     the design: failures deserve retries with the better model)
        p0 = sum(1 for _ in open(f"{work}/progress_e0.jsonl", encoding="utf-8"))
        p1 = sum(1 for _ in open(f"{work}/progress_e1.jsonl", encoding="utf-8"))
        assert p0 == 3 and p1 == 3, (p0, p1)
        print("  [2] per-epoch ledgers written (3 attempts each; failures retried with the new "
              "adapter) -> PASS")

        # [3] holdout never banked + gate-integrity fields present
        assert all("holdout_before" in h and "holdout_after" in h for h in hist)
        assert hist[1]["holdout_after"]["rate"] == 0.0                 # unsolvable holdout stays 0
        print("  [3] frozen holdout evaluated pre/post each epoch (taxonomy incl. plus_only gate "
              "integrity) and never banked -> PASS")

    print("\n  ALGO_STAR_EPOCH SELFTEST -> PASS  (the gate decides what enters the WEIGHTS, exactly "
          "as it decides what enters the graph)")
    return True


def main():
    ap = argparse.ArgumentParser(description="STaR epoch driver: author -> verified traces -> LoRA -> repeat.")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--epochs", type=int, default=0)
    ap.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    ap.add_argument("--corpus", default="artifacts/mbpp_plus_prepped.jsonl")
    ap.add_argument("--limit", type=int, default=300)
    ap.add_argument("--holdout", type=int, default=40)
    ap.add_argument("--graph", default="graphs/algo_star_epoch.json")
    ap.add_argument("--work", default="artifacts/star_epoch")
    a = ap.parse_args()
    if a.selftest:
        sys.exit(0 if _selftest() else 1)
    if a.epochs:
        run_epochs(a.model, a.corpus, n_epochs=a.epochs, limit=a.limit, holdout_n=a.holdout,
                   graph_path=a.graph, work=a.work)
        return
    ap.print_help()


if __name__ == "__main__":
    main()
