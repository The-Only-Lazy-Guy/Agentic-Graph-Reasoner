"""RL loop for the DERIVE step — GRPO over a LoRA, rewarded by the validated derive_reward.

The baseline (derive_hard) showed the frozen 4B's gap = MIXED-OP composition (single-op fine; sum+sub,
mul+add fail). This trains a LoRA to close it: for each procedurally-generated mixed-op task, sample K
derive completions, score each with derive_reward (grounded x solves x unique; hallucination punished),
and do a group-baseline policy-gradient step on the LoRA (GRPO without the clip). The 4B base stays
frozen — the LoRA learns the SKILL of composing grounded from upstream (knowledge stays in the graph).

The reward GATES on groundedness: the policy cannot win by emitting a gold-looking number, only by
actually composing it -> RL can't reward-hack (validated in derive_reward).

  train (A40):   V5_LM_TRUST_REMOTE_CODE=1 python -m v5.runtime.derive_rl --steps 200 --k 8
  selftest (no model):  python -m v5.runtime.derive_rl --selftest
"""
from __future__ import annotations

import argparse
import os
import random
import re

from v5.runtime.derive_reward import derive_reward, _nums

# mixed-op formula templates over named upstream A,B,C (the gap = composing DIFFERENT ops).
TEMPLATES = [
    ("sum3",     lambda a, b, c: a + b + c,        "the total cost including all three parts"),
    ("sum_disc", lambda a, b, c: a + b - c,        "the subtotal of the two items after the discount"),
    ("prod_add", lambda a, b, c: a * b + c,        "the cost of the units plus the flat fee"),
    ("sum_prod", lambda a, b, c: (a + b) * c,      "the total when the two parts are bought in that quantity"),
    ("prod_sub", lambda a, b, c: a * b - c,        "the cost of the units after the rebate"),
]
NOISE = ["The order id is 5567.", "It ships from aisle 12.", "Founded in 2019.",
         "There are 100 reviews.", "This is the tier 3 plan.", "Catalog page 88."]


def gen_task(rng: random.Random) -> dict:
    name, formula, instr = rng.choice(TEMPLATES)
    a, b, c = rng.randint(2, 15), rng.randint(2, 15), rng.randint(1, 6)
    named = {"A": str(a), "B": str(b), "C": str(c)}
    gold = int(formula(a, b, c))
    noise = rng.sample(NOISE, 2)
    givens = f"A = {a}; B = {b}; C = {c}"
    prompt = (f"Given values: {givens}. (Notes, irrelevant: {' '.join(noise)}) "
              f"Compute {instr}. Use ONLY the given values; ignore the notes. Answer with only the number:")
    return {"name": name, "prompt": prompt, "named": named, "gold": gold,
            "formula": (lambda nm, f=formula: f(int(nm["A"]), int(nm["B"]), int(nm["C"])))}


def score(value, task) -> tuple[float, dict]:
    return derive_reward(value, [], gold=task["gold"], upstream_named=task["named"], formula=task["formula"])


def advantages(rewards: list[float]) -> list[float]:
    """GROUP baseline: advantage = reward - group mean, /std (GRPO). Centered -> sums ~0."""
    n = len(rewards)
    mean = sum(rewards) / n
    var = sum((r - mean) ** 2 for r in rewards) / n
    std = var ** 0.5 or 1.0
    return [(r - mean) / std for r in rewards]


# ── no-model proof of the RL MATH: task gen valid + reward discriminates + advantages correct ──
def _selftest():
    print("derive_rl --selftest: prove the RL math (task gen + reward + GRPO advantage), no model.\n")
    rng = random.Random(0)
    tasks = [gen_task(rng) for _ in range(6)]
    ok_tasks = all(_nums(str(t["gold"])) and t["formula"](t["named"]) == t["gold"] for t in tasks)
    print(f"  task-gen: {len(tasks)} mixed-op tasks, gold==formula(named) -> {ok_tasks}")
    for t in tasks[:3]:
        print(f"    [{t['name']:9}] named={t['named']} gold={t['gold']}")
    # reward discriminates: the gold value scores high, a wrong/hallucinated value is punished.
    t = tasks[0]
    r_good, _ = score(str(t["gold"]), t)
    r_bad, _ = score("9999", t)
    r_lucky_ungrounded, b = score(str(t["gold"]), {**t, "named": {"A": "1", "B": "1", "C": "1"}})  # gold but not from these
    disc = r_good > 1.0 and r_bad < 0
    print(f"  reward: gold->{r_good:+.2f}  hallucinated->{r_bad:+.2f}  (discriminates={disc})")
    print(f"  anti-hack: same gold value but ungrounded in upstream -> {r_lucky_ungrounded:+.2f} ({b['verdict']})")
    # advantage: higher reward -> higher advantage, group-centered ~0.
    advs = advantages([1.5, 1.5, -1.0, -1.0])
    centered = abs(sum(advs)) < 1e-6
    monotone = advs[0] > advs[2]
    print(f"  GRPO advantages([1.5,1.5,-1,-1]) = {[round(x,2) for x in advs]}  centered={centered} monotone={monotone}")
    ok = ok_tasks and disc and r_lucky_ungrounded < 0 and centered and monotone
    print(f"\n  RL-MATH SELFTEST -> {'PASS' if ok else 'FAIL'}  (gen valid, reward gated on grounding, advantages correct)")
    return ok


def train(model_name, steps, K, lr, r_lora, seed, layers, eval_every):
    import torch
    import torch.nn as nn
    from transformers import AutoTokenizer
    from peft import LoraConfig, get_peft_model
    from v5.lm_loader import load_frozen_lm

    trust = os.environ.get("V5_LM_TRUST_REMOTE_CODE", "0").lower() in ("1", "true", "yes")
    base = load_frozen_lm(model_name)
    tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=trust)
    dev = next(base.parameters()).device
    # auto-detect the Linear leaf-module names inside the transformer layers (Qwen3.5 is a custom
    # linear-attention arch -> NOT q/k/v/o_proj). Architecture-agnostic; print what we adapt.
    leaf = sorted({n.split(".")[-1] for n, m in base.named_modules()
                   if isinstance(m, nn.Linear) and ".layers." in n
                   and not any(x in n.lower() for x in ("lm_head", "embed"))})
    if not leaf:
        raise RuntimeError("no Linear leaf modules found inside .layers.* — inspect base.named_modules()")
    print(f"LoRA target leaf modules (auto-detected): {leaf}", flush=True)
    cfg = LoraConfig(r=r_lora, lora_alpha=2 * r_lora, lora_dropout=0.0, task_type="CAUSAL_LM",
                     target_modules=leaf, layers_to_transform=layers)   # target the compose/inject band
    model = get_peft_model(base, cfg); model.train()
    trainable = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(trainable, lr=lr)
    print(f"LoRA r={r_lora} on layers {layers} | trainable params={sum(p.numel() for p in trainable):,}", flush=True)
    rng = random.Random(seed)
    held = [gen_task(random.Random(10_000 + i)) for i in range(20)]   # fixed held-out eval set

    def gen_ids(prompt_ids, sample):
        with torch.no_grad():
            return model.generate(prompt_ids, do_sample=sample, temperature=0.9 if sample else None,
                                  top_p=0.95 if sample else None, max_new_tokens=12,
                                  pad_token_id=tok.eos_token_id)

    def encode(prompt):
        m = [{"role": "user", "content": prompt}]
        try:
            return tok.apply_chat_template(m, add_generation_prompt=True, enable_thinking=False, return_tensors="pt").to(dev)
        except TypeError:
            return tok.apply_chat_template(m, add_generation_prompt=True, return_tensors="pt").to(dev)

    def seq_logprob(prompt_ids, comp_ids):
        full = torch.cat([prompt_ids, comp_ids], dim=1)
        logits = model(full).logits[:, :-1]
        logp = torch.log_softmax(logits.float(), dim=-1)
        start = prompt_ids.shape[1] - 1
        sel = logp[:, start:start + comp_ids.shape[1]].gather(-1, comp_ids.unsqueeze(-1)).squeeze(-1)
        return sel.sum(-1)                                   # total log-prob of the sampled completion

    @torch.no_grad()
    def evaluate(tasks):
        model.eval(); rs = []
        for t in tasks:
            pids = encode(t["prompt"])
            out = gen_ids(pids, sample=False)
            val = _nums(tok.decode(out[0, pids.shape[1]:], skip_special_tokens=True))
            rs.append(score(str(val[0]) if val else "", t)[0])
        model.train()
        return sum(rs) / len(rs), sum(1 for r in rs if r < 0) / len(rs)

    base_mean, base_hall = evaluate(held)
    print(f"[eval @0] held-out mean reward={base_mean:+.3f} hallucination={base_hall:.0%}", flush=True)
    for step in range(1, steps + 1):
        t = gen_task(rng)
        pids = encode(t["prompt"])
        comps, rewards = [], []
        for _ in range(K):
            out = gen_ids(pids, sample=True)
            comp = out[:, pids.shape[1]:]
            val = _nums(tok.decode(comp[0], skip_special_tokens=True))
            comps.append(comp); rewards.append(score(str(val[0]) if val else "", t)[0])
        advs = advantages(rewards)
        loss = 0.0
        for comp, a in zip(comps, advs):
            loss = loss - a * seq_logprob(pids, comp)
        loss = loss / K
        loss.backward(); torch.nn.utils.clip_grad_norm_(trainable, 1.0); opt.step(); opt.zero_grad()
        if step % 20 == 0:
            print(f"[step {step}] task={t['name']:9} mean_reward={sum(rewards)/K:+.2f} loss={float(loss):+.3f}", flush=True)
        if step % eval_every == 0:
            m, h = evaluate(held)
            print(f"[eval @{step}] held-out mean reward={m:+.3f} hallucination={h:.0%}  (base {base_mean:+.3f}/{base_hall:.0%})", flush=True)
    m, h = evaluate(held)
    print(f"\n=== RL DONE === held-out mean reward {base_mean:+.3f}->{m:+.3f} | hallucination {base_hall:.0%}->{h:.0%}")
    print("  REAL only if the lift is on HELD-OUT mixed-op tasks (not the trained ones) and hallucination drops.")
    out = "artifacts/derive_lora"
    model.save_pretrained(out); print(f"  LoRA saved -> {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--model", default="Qwen/Qwen3.5-4B")
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--k", type=int, default=8, help="rollouts per task (GRPO group size)")
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--r-lora", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--layers", type=int, nargs="+", default=[20, 22, 24, 26, 28],
                    help="layers to put LoRA on (the compose/inject band)")
    ap.add_argument("--eval-every", type=int, default=50)
    a = ap.parse_args()
    if a.selftest:
        import sys
        sys.exit(0 if _selftest() else 1)
    train(a.model, a.steps, a.k, a.lr, a.r_lora, a.seed, a.layers, a.eval_every)


if __name__ == "__main__":
    main()
