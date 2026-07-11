"""Reasoning + RL: does the system REASON (not retrieve), and does MEMORY lift reasoning?

Task: find the pure-strategy Nash equilibrium of a random 2-player payoff bimatrix. The oracle
is computable and the answer is verifiable, but every instance is NEW — the model cannot
retrieve an answer, it must reason (check mutual best responses). This is the "harder task":
genuine computation, not retrieve-and-paste.

Three arms answer the core question ("does the system reason, or just its LM? does memory
help?"):
  - COLD    : the LM solves the instance alone.
  - +MEMORY : K retrieved WORKED EXAMPLES (a similar game + its best-response derivation + answer)
              are prepended — testing whether showing the METHOD lifts reasoning on a new game.
  - +RL     : GRPO over a LoRA, reward = answer correct (verifiable), optionally with memory.

Memory here is NOT answer-lookup (each game is new) — it is method demonstration. The sharp
test: do RELEVANT worked examples (same game TYPE) beat generic ones beat none?

Self-contained (inline instance generation; lazy model imports). No dependency on the memory
package for the benchmark itself — the "+memory" arm retrieves from an in-process solved-bank.

  selftest (no model):  python -m v5.runtime.reason_rl --selftest
  baseline (frozen):    python -m v5.runtime.reason_rl --baseline --n-baseline 40 --shots 0
  baseline +memory:     python -m v5.runtime.reason_rl --baseline --n-baseline 40 --shots 3
  train (GPU):          V5_LM_TRUST_REMOTE_CODE=1 python -m v5.runtime.reason_rl --steps 300 --k 12 --shots 3
"""
from __future__ import annotations

import argparse
import os
import random
import re
import sys
import time

# ═══════════════════════════════════════════════════════════════════════════════
# GAME GENERATION + ORACLE
# ═══════════════════════════════════════════════════════════════════════════════

GAME_TYPES = ("dominance", "unique", "coordination")   # structural families (memory-relevance)


def pure_ne(row_pay: list[list[int]], col_pay: list[list[int]]) -> list[tuple[int, int]]:
    """All pure-strategy Nash equilibria: cells that are a mutual best response.
    row chooses the row (max over rows within the column); col chooses the col (max over cols
    within the row)."""
    m, n = len(row_pay), len(row_pay[0])
    out = []
    for i in range(m):
        for j in range(n):
            row_best = row_pay[i][j] >= max(row_pay[r][j] for r in range(m))
            col_best = col_pay[i][j] >= max(col_pay[i][c] for c in range(n))
            if row_best and col_best:
                out.append((i, j))
    return out


def _row_dominated(row_pay: list[list[int]]) -> bool:
    """True if some row strictly dominates another (a dominance-solvable structure)."""
    m, n = len(row_pay), len(row_pay[0])
    for a in range(m):
        for b in range(m):
            if a != b and all(row_pay[a][j] > row_pay[b][j] for j in range(n)):
                return True
    return False


def _classify(row_pay, col_pay) -> str:
    ne = pure_ne(row_pay, col_pay)
    if len(ne) >= 2:
        return "coordination"
    if _row_dominated(row_pay):
        return "dominance"
    return "unique"


def make_game(seed: int, size: int = 2, want_type: str | None = None,
              lo: int = 0, hi: int = 9, balance: bool = True) -> dict:
    """Deterministic game with EXACTLY ONE pure NE (reject-sampled), distinct payoffs.

    balance=True forces the NE cell to a seed-cycled target so the NE is UNIFORM over cells.
    This kills the constant-answer confound: 'always guess cell X' then scores exactly 1/cells
    (25% at 2x2, 11% at 3x3), the floor a reasoning model must BEAT. Without it, a model that
    anchors to a frequent cell looks like it's 'reasoning'."""
    rng = random.Random(seed * 7919 + 13)
    cells = [(i, j) for i in range(size) for j in range(size)]
    target = cells[seed % len(cells)] if balance else None
    for _ in range(8000):
        rp = [[rng.randint(lo, hi) for _ in range(size)] for _ in range(size)]
        cp = [[rng.randint(lo, hi) for _ in range(size)] for _ in range(size)]
        ne = pure_ne(rp, cp)
        if len(ne) != 1:
            continue
        if target is not None and ne[0] != target:
            continue
        typ = _classify(rp, cp)
        if want_type and typ != want_type:
            continue
        return dict(seed=seed, size=size, row_pay=rp, col_pay=cp, ne=ne[0], gtype=typ,
                    instance_id=f"game_{size}x{size}_{seed}")
    raise RuntimeError(f"could not generate a unique-NE game (seed={seed}, type={want_type})")


# ═══════════════════════════════════════════════════════════════════════════════
# PROMPT + WORKED-EXAMPLE (memory) + REWARD
# ═══════════════════════════════════════════════════════════════════════════════

def format_game(inst: dict) -> str:
    rp, cp = inst["row_pay"], inst["col_pay"]
    lines = [f"A 2-player game. Cell (i,j): row player's payoff = R[i][j], col player's payoff = C[i][j]."]
    lines.append(f"R (row payoffs) = {rp}")
    lines.append(f"C (col payoffs) = {cp}")
    return "\n".join(lines)


def worked_example(inst: dict) -> str:
    """A METHOD demonstration for the +memory arm: the game, the best-response check for the NE
    cell, and the answer. Teaches HOW to solve, not just the answer."""
    i, j = inst["ne"]
    rp, cp = inst["row_pay"], inst["col_pay"]
    m, n = inst["size"], inst["size"]
    col_of_ne = [rp[r][j] for r in range(m)]
    row_of_ne = [cp[i][c] for c in range(n)]
    return (
        f"{format_game(inst)}\n"
        f"Reasoning: a pure Nash equilibrium is a cell where neither player gains by deviating.\n"
        f"  Cell ({i},{j}): row payoffs down column {j} are {col_of_ne}; {rp[i][j]} is the max, "
        f"so the row player will not switch rows.\n"
        f"  col payoffs across row {i} are {row_of_ne}; {cp[i][j]} is the max, so the col player "
        f"will not switch columns.\n"
        f"  Both are best responses -> equilibrium.\n"
        f"ANSWER: ({i}, {j})"
    )


def build_prompt(inst: dict, examples: list[dict] | None = None) -> str:
    head = ("You find the pure-strategy Nash equilibrium of a 2-player game. Check, for each "
            "cell, whether the row player's payoff is the max in its column AND the col player's "
            "payoff is the max in its row. Output the single equilibrium cell as `ANSWER: (i, j)`.")
    parts = [head]
    if examples:
        parts.append("\nWorked examples:")
        for ex in examples:
            parts.append(worked_example(ex))
    parts.append("\nNow solve this game:")
    parts.append(format_game(inst))
    parts.append("Show brief reasoning, then end with `ANSWER: (i, j)`.")
    return "\n\n".join(parts)


_ANS_RE = re.compile(r"ANSWER\s*:\s*\(?\s*(\d+)\s*,\s*(\d+)\s*\)?")
_TUPLE_RE = re.compile(r"\(\s*(\d+)\s*,\s*(\d+)\s*\)")


def parse_answer(gen: str) -> tuple[int, int] | None:
    """Extract the model's final answer. Prefer the LAST `ANSWER: (i, j)`; else fall back to the
    LAST parenthesized (i, j) tuple anywhere. The fallback matters: a model can REASON to the
    right cell without emitting the exact `ANSWER:` template, and punishing that as 'no answer'
    conflates formatting with reasoning (it deflated the cold arm)."""
    gen = gen or ""
    m = _ANS_RE.findall(gen)
    if m:
        return (int(m[-1][0]), int(m[-1][1]))
    t = _TUPLE_RE.findall(gen)
    if t:
        return (int(t[-1][0]), int(t[-1][1]))
    return None


PASS_REWARD = 1.0
WRONG_PENALTY = 1.0
NOANS_PENALTY = 1.0


def reason_reward(gen: str, inst: dict) -> tuple[float, dict]:
    ans = parse_answer(gen)
    if ans is None:
        return -NOANS_PENALTY, {"verdict": "PUNISH (no ANSWER)"}
    m = inst["size"]
    if not (0 <= ans[0] < m and 0 <= ans[1] < m):
        return -WRONG_PENALTY, {"verdict": f"PUNISH (out of range {ans})"}
    if ans == inst["ne"]:
        return PASS_REWARD, {"verdict": f"REWARD (correct {ans})"}
    return -WRONG_PENALTY, {"verdict": f"PUNISH (wrong {ans}, gold {inst['ne']})"}


def advantages(rewards: list[float]) -> list[float]:
    n = len(rewards)
    mean = sum(rewards) / n
    var = sum((r - mean) ** 2 for r in rewards) / n
    std = var ** 0.5 or 1.0
    return [(r - mean) / std for r in rewards]


def _clean(txt: str) -> str:
    return re.sub(r"<think>.*?</think>", "", txt or "", flags=re.DOTALL)


def _chat_text(tok, prompt: str) -> str:
    m = [{"role": "user", "content": prompt}]
    try:
        return tok.apply_chat_template(m, tokenize=False, add_generation_prompt=True,
                                       enable_thinking=False)
    except TypeError:
        return tok.apply_chat_template(m, tokenize=False, add_generation_prompt=True)


def batch_generate(model, tok, prompts: list[str], dev, max_new: int = 400,
                   sample: bool = False, temperature: float = 1.0, chunk: int = 16) -> list[str]:
    """Generate for a LIST of prompts in left-padded batches — uses the big VRAM instead of one
    generation at a time. Returns cleaned completions in prompt order."""
    import torch
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    old_side = tok.padding_side
    tok.padding_side = "left"
    outs: list[str] = []
    try:
        for s in range(0, len(prompts), chunk):
            texts = [_chat_text(tok, p) for p in prompts[s:s + chunk]]
            enc = tok(texts, return_tensors="pt", padding=True,
                      add_special_tokens=False).to(dev)
            with torch.no_grad():
                gen = model.generate(**enc, do_sample=sample,
                                     temperature=temperature if sample else None,
                                     top_p=0.95 if sample else None,
                                     max_new_tokens=max_new, pad_token_id=tok.eos_token_id)
            new = gen[:, enc["input_ids"].shape[1]:]
            outs.extend(_clean(tok.decode(row, skip_special_tokens=True)) for row in new)
    finally:
        tok.padding_side = old_side
    return outs


# ═══════════════════════════════════════════════════════════════════════════════
# MEMORY: a solved-bank of worked examples, retrieved by game TYPE (relevant) or random
# ═══════════════════════════════════════════════════════════════════════════════

TRAIN_SEEDS = range(1000, 1400)
EVAL_SEEDS = range(0, 60)
BANK_SEEDS = range(2000, 2200)          # disjoint pool the memory arm retrieves worked examples from


def build_bank(size: int) -> list[dict]:
    return [make_game(s, size=size) for s in BANK_SEEDS]


def retrieve_examples(inst: dict, bank: list[dict], k: int, mode: str = "relevant",
                      rng: random.Random | None = None) -> list[dict]:
    """Pick k worked examples from the bank. mode='relevant' -> same game TYPE (tests whether
    method-matched memory helps more); 'random' -> generic few-shot; 'none' -> [].
    Excludes the instance's own seed (no answer leakage)."""
    if k <= 0 or mode == "none":
        return []
    rng = rng or random.Random(inst["seed"])
    pool = [b for b in bank if b["seed"] != inst["seed"]]
    if mode == "relevant":
        same = [b for b in pool if b["gtype"] == inst["gtype"]]
        pool = same or pool
    rng.shuffle(pool)
    return pool[:k]


# ═══════════════════════════════════════════════════════════════════════════════
# BASELINE EVAL (frozen model) — cold vs +memory
# ═══════════════════════════════════════════════════════════════════════════════

def run_baseline(model_name: str, n: int, size: int, shots: int, mem_mode: str, chunk: int = 16):
    from transformers import AutoTokenizer
    from v5.lm_loader import load_frozen_lm

    trust = os.environ.get("V5_LM_TRUST_REMOTE_CODE", "0").lower() in ("1", "true", "yes")
    model = load_frozen_lm(model_name); model.eval()
    tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=trust)
    dev = next(model.parameters()).device
    bank = build_bank(size) if shots > 0 else []

    print(f"FROZEN baseline: {n} games {size}x{size}, shots={shots} mem={mem_mode} "
          f"chunk={chunk} (model={model_name})\n", flush=True)
    insts = [make_game(s, size=size) for s in list(EVAL_SEEDS)[:n]]
    prompts = [build_prompt(g, retrieve_examples(g, bank, shots, mem_mode)) for g in insts]
    gens = batch_generate(model, tok, prompts, dev, max_new=400, sample=False, chunk=chunk)

    solved, by_type, preds = 0, {}, []
    for inst, gen in zip(insts, gens):
        r, info = reason_reward(gen, inst)
        ok = r >= PASS_REWARD - 0.01
        solved += ok
        preds.append(parse_answer(gen))
        d = by_type.setdefault(inst["gtype"], [0, 0]); d[0] += ok; d[1] += 1
        print(f"  [{inst['instance_id']}:{inst['gtype']}] {info['verdict']}", flush=True)
    total = len(insts)
    # floors: the score to BEAT. const = best single-cell guess on THIS eval set; random = 1/cells.
    from collections import Counter
    ne_counts = Counter(g["ne"] for g in insts)
    const_floor = max(ne_counts.values()) / total
    rand_floor = 1.0 / (size * size)
    # anchoring diagnostic: is the model just repeating ONE cell?
    pred_counts = Counter(p for p in preds if p is not None)
    top_pred, top_n = (pred_counts.most_common(1)[0] if pred_counts else (None, 0))
    noans = sum(1 for p in preds if p is None)
    print(f"\n  solve {solved}/{total} = {solved/max(1,total):.0%}  "
          + " ".join(f"{t}:{c}/{n_}" for t, (c, n_) in by_type.items()), flush=True)
    print(f"  floors: const-cell {const_floor:.0%} | random {rand_floor:.0%}   "
          f"(reasoning only if solve > const)", flush=True)
    print(f"  anchoring: most-common pred {top_pred} used {top_n}/{total}, no-answer {noans}/{total}",
          flush=True)
    return solved / max(1, total)


# ═══════════════════════════════════════════════════════════════════════════════
# GRPO TRAIN (LoRA) — reward = answer correct; optional memory in the prompt
# ═══════════════════════════════════════════════════════════════════════════════

def train(model_name, steps, K, lr, r_lora, seed, layers, eval_every, ent_coef,
          temperature, size, shots, mem_mode, sft_steps, chunk=16):
    import torch
    import torch.nn as nn
    from transformers import AutoTokenizer
    from peft import LoraConfig, get_peft_model
    from v5.lm_loader import load_frozen_lm

    trust = os.environ.get("V5_LM_TRUST_REMOTE_CODE", "0").lower() in ("1", "true", "yes")
    base = load_frozen_lm(model_name)
    tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=trust)
    dev = next(base.parameters()).device

    leaf = sorted({n.split(".")[-1] for n, m in base.named_modules()
                   if isinstance(m, nn.Linear) and ".layers." in n
                   and not any(x in n.lower() for x in ("lm_head", "embed"))})
    cfg = LoraConfig(r=r_lora, lora_alpha=2 * r_lora, lora_dropout=0.0,
                     task_type="CAUSAL_LM", target_modules=leaf, layers_to_transform=layers)
    model = get_peft_model(base, cfg); model.train()
    trainable = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(trainable, lr=lr)
    print(f"LoRA r={r_lora} layers={layers} | trainable={sum(p.numel() for p in trainable):,} | "
          f"{size}x{size} shots={shots} mem={mem_mode}", flush=True)

    rng = random.Random(seed)
    bank = build_bank(size) if shots > 0 else []
    held = [make_game(s, size=size) for s in EVAL_SEEDS][:40]
    train_seeds = list(TRAIN_SEEDS)

    def encode(prompt):
        m = [{"role": "user", "content": prompt}]
        kw = dict(add_generation_prompt=True, return_tensors="pt", return_dict=True)
        try:
            return tok.apply_chat_template(m, enable_thinking=False, **kw)["input_ids"].to(dev)
        except TypeError:
            return tok.apply_chat_template(m, **kw)["input_ids"].to(dev)

    def seq_logprob(pids, comp):
        full = torch.cat([pids, comp], dim=1)
        logits = model(full).logits[:, :-1]
        logp = torch.log_softmax(logits.float(), dim=-1)
        start = pids.shape[1] - 1
        span = logp[:, start:start + comp.shape[1]]
        sel = span.gather(-1, comp.unsqueeze(-1)).squeeze(-1).sum(-1)
        ent = -(span.exp() * span).sum(-1).mean()
        return sel, ent

    def evaluate():
        model.eval()
        prompts = [build_prompt(g, retrieve_examples(g, bank, shots, mem_mode)) for g in held]
        gens = batch_generate(model, tok, prompts, dev, max_new=400, sample=False, chunk=chunk)
        rs = [reason_reward(g, inst)[0] for inst, g in zip(held, gens)]
        model.train()
        return sum(1 for r in rs if r >= PASS_REWARD - 0.01) / max(1, len(rs))

    base_solve = evaluate()
    print(f"[eval @0] held-out solve={base_solve:.0%}", flush=True)

    for step in range(1, steps + 1):
        inst = make_game(rng.choice(train_seeds), size=size)
        ex = retrieve_examples(inst, bank, shots, mem_mode)
        pids = encode(build_prompt(inst, ex))
        # all K rollouts in ONE batched generate (uses the big VRAM) instead of K calls
        with torch.no_grad():
            out = model.generate(pids, do_sample=True, temperature=temperature, top_p=0.95,
                                 max_new_tokens=400, num_return_sequences=K,
                                 pad_token_id=tok.eos_token_id)
        comp_all = out[:, pids.shape[1]:]     # [K, new]
        comps, rewards = [], []
        for k in range(K):
            comp = comp_all[k:k + 1]
            gen = _clean(tok.decode(comp[0], skip_special_tokens=True))
            comps.append(comp)
            r, _ = reason_reward(gen, inst); rewards.append(r)
        mean_r = sum(rewards) / K
        r_std = (sum((r - mean_r) ** 2 for r in rewards) / K) ** 0.5
        if r_std < 1e-9:
            if step % 20 == 0:
                print(f"[step {step:3}] mean_r={mean_r:+.2f} r_std=0 SKIP", flush=True)
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
            solve = sum(1 for r in rewards if r >= PASS_REWARD - 0.01) / K
            print(f"[step {step:3}] mean_r={mean_r:+.2f} r_std={r_std:.2f} "
                  f"solve={solve:.0%} rewards={[round(r,1) for r in rewards]}", flush=True)
        if step % eval_every == 0:
            print(f"[eval @{step}] held-out solve={evaluate():.0%}  (base {base_solve:.0%})", flush=True)

    final = evaluate()
    print(f"\n=== RL DONE === held-out solve {base_solve:.0%} -> {final:.0%}", flush=True)
    model.save_pretrained("artifacts/reason_lora")
    print("  LoRA saved -> artifacts/reason_lora", flush=True)


# ═══════════════════════════════════════════════════════════════════════════════
# SELFTEST (no model)
# ═══════════════════════════════════════════════════════════════════════════════

def _selftest() -> bool:
    print("reason_rl --selftest: oracle, unique-NE generation, reward, memory retrieval (no model)\n")
    ok = True

    # 1. oracle correctness on a known game (Prisoner's Dilemma-ish, unique NE)
    rp = [[3, 0], [5, 1]]; cp = [[3, 5], [0, 1]]   # (1,1) is the unique pure NE (defect,defect)
    ne = pure_ne(rp, cp)
    assert ne == [(1, 1)], f"oracle wrong on PD: {ne}"
    print(f"  [1] oracle: PD unique NE = {ne} -> PASS")

    # 2. generator: exactly ONE pure NE, and NE cell is BALANCED (uniform over cells) so the
    # constant-answer floor is 1/cells, not something a guesser can exploit.
    from collections import Counter
    n_by_type = {}
    cell_hist = Counter()
    for s in range(48):
        inst = make_game(s, size=2)
        assert len(pure_ne(inst["row_pay"], inst["col_pay"])) == 1, f"seed {s}: not unique"
        assert inst["ne"] == pure_ne(inst["row_pay"], inst["col_pay"])[0]
        n_by_type[inst["gtype"]] = n_by_type.get(inst["gtype"], 0) + 1
        cell_hist[inst["ne"]] += 1
        make_game(s, size=3)   # 3x3 also generates
    # balance: all 4 cells appear, none dominates (each ~12/48)
    assert len(cell_hist) == 4 and max(cell_hist.values()) <= 16, f"NE not balanced: {dict(cell_hist)}"
    assert len(n_by_type) >= 2, f"want variety of game types, got {n_by_type}"
    print(f"  [2] generator: unique-NE, balanced cells {dict(cell_hist)}, types {n_by_type} -> PASS")

    # 3. reward discriminates: correct answer > wrong > no-answer
    inst = make_game(3, size=2); i, j = inst["ne"]
    r_ok, _ = reason_reward(f"reasoning... ANSWER: ({i}, {j})", inst)
    wi = (i + 1) % 2
    r_wrong, _ = reason_reward(f"ANSWER: ({wi}, {j})", inst)
    r_none, _ = reason_reward("no idea", inst)
    assert r_ok > 0 and r_wrong < 0 and r_none < 0, (r_ok, r_wrong, r_none)
    # parse takes the LAST answer (model's final); ANSWER: wins over bare tuples; and a bare
    # tuple with NO `ANSWER:` still parses (don't punish reasoning for formatting).
    assert parse_answer(f"ANSWER: (0,0)\n...revised...\nANSWER: ({i}, {j})") == (i, j)
    assert parse_answer("checking cells... the equilibrium is (1, 0).") == (1, 0)   # fallback
    assert parse_answer("R = [[3,1],[0,2]]\nso ANSWER: (0, 1)") == (0, 1)            # ANSWER wins
    assert parse_answer("no cell here") is None
    print(f"  [3] reward: correct {r_ok:+.0f} / wrong {r_wrong:+.0f} / none {r_none:+.0f}, "
          f"parse ANSWER>tuple>none -> PASS")

    # 4. worked example teaches the method AND states the right answer (no leakage of a wrong one)
    we = worked_example(inst)
    assert f"ANSWER: ({i}, {j})" in we and "best response" in we.lower()
    print(f"  [4] worked_example: method + correct answer -> PASS")

    # 5. memory retrieval: relevant = same type, excludes own seed, respects k
    bank = [make_game(s, size=2) for s in range(2000, 2040)]
    tgt = make_game(5, size=2)
    rel = retrieve_examples(tgt, bank, 3, "relevant")
    assert len(rel) == 3 and all(e["seed"] != tgt["seed"] for e in rel)
    assert all(e["gtype"] == tgt["gtype"] for e in rel) or \
        not [b for b in bank if b["gtype"] == tgt["gtype"] and b["seed"] != tgt["seed"]]
    assert retrieve_examples(tgt, bank, 0, "relevant") == []
    assert retrieve_examples(tgt, bank, 3, "none") == []
    print(f"  [5] memory retrieval: relevant=same-type, k-capped, no self-leak -> PASS")

    # 6. prompt format: cold has no examples; +memory prepends worked examples
    p_cold = build_prompt(tgt, [])
    p_mem = build_prompt(tgt, rel)
    assert "Worked examples" not in p_cold and "ANSWER: (i, j)" in p_cold
    assert "Worked examples" in p_mem and p_mem.count("ANSWER:") >= 3
    print(f"  [6] prompt: cold vs +memory ({len(rel)} worked examples) -> PASS")

    print(f"\n  REASON_RL SELFTEST -> {'PASS' if ok else 'FAIL'}")
    return ok


def main():
    ap = argparse.ArgumentParser(description="Reasoning + RL: Nash equilibrium, cold vs memory vs RL.")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--baseline", action="store_true", help="frozen-model eval (no RL)")
    ap.add_argument("--n-baseline", type=int, default=40)
    ap.add_argument("--model", default="Qwen/Qwen2.5-3B")
    ap.add_argument("--size", type=int, default=2, help="game size NxN (difficulty knob)")
    ap.add_argument("--shots", type=int, default=0, help="worked examples from memory (0=cold)")
    ap.add_argument("--mem", default="relevant", choices=["relevant", "random", "none"])
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--k", type=int, default=12, help="rollouts per task (GRPO group)")
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--r-lora", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--layers", type=int, nargs="+", default=[20, 22, 24, 26, 28])
    ap.add_argument("--eval-every", type=int, default=50)
    ap.add_argument("--ent-coef", type=float, default=0.005)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--sft-steps", type=int, default=0)
    ap.add_argument("--chunk", type=int, default=16,
                    help="batched-generation chunk size (raise on big VRAM, e.g. 64 on 90GB)")
    a = ap.parse_args()

    if a.selftest:
        sys.exit(0 if _selftest() else 1)
    if a.baseline:
        return run_baseline(a.model, a.n_baseline, a.size, a.shots, a.mem, a.chunk)
    train(a.model, a.steps, a.k, a.lr, a.r_lora, a.seed, a.layers, a.eval_every,
          a.ent_coef, a.temperature, a.size, a.shots, a.mem, a.sft_steps, a.chunk)


if __name__ == "__main__":
    main()
