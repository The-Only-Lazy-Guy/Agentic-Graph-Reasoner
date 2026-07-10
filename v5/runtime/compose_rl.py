"""GRPO RL for the COMPOSE code generation step — fully self-contained.

The compose task: three source files (tax.py, fees.py, catalog.py) are created, then the
model must generate checkout.py that composes constants from tax.py and fees.py into a
combined formula. The spec WITHHOLDS the actual rate values — the model must infer which
files to use and what they contain.

Self-contained: sandbox runner and compose instances are inlined (no module-level
dependency on v5.runtime modules that may not exist on all branch states). The only
external imports are at function-call time (lazy) for training with a model.

Usage:
  selftest (no model):  python -m v5.runtime.compose_rl --selftest
  baseline eval:        python -m v5.runtime.compose_rl --baseline
  train (A40):          V5_LM_TRUST_REMOTE_CODE=1 python -m v5.runtime.compose_rl \\
                          --steps 300 --k 16 --sft-steps 80 --model Qwen/Qwen2.5-3B
"""
from __future__ import annotations

import argparse
import ast
import os
import random
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

# ═══════════════════════════════════════════════════════════════════════════════
# INLINE SANDBOX  — self-contained test runner (no v5.runtime.sandbox dependency)
# ═══════════════════════════════════════════════════════════════════════════════

_SENTINEL = "SBX_RESULT"


def _sandbox_run(code: str, tests: list[str], setup: str = "",
                 timeout: float = 5.0) -> dict:
    """Single-file test.  Same result schema as v5.runtime.sandbox.run."""
    harness_lines = [
        "",
        "if True:",
        "    import sys as _sys, os as _os",
        "    _sys.path.insert(0, _os.getcwd())",
        "    import traceback as _tb",
        f"    _setup = {setup!r}",
        f"    _tests = {tests!r}",
        "    _p, _ff = 0, ''",
        "    try:",
        "        if _setup: exec(compile(_setup, '<setup>', 'exec'), globals())",
        "    except Exception as _e:",
        "        _ff = 'setup: %s: %s' % (type(_e).__name__, _e)",
        "    if not _ff:",
        "        for _i, _src in enumerate(_tests):",
        "            try:",
        "                exec(compile(_src, '<test%d>' % _i, 'exec'), globals())",
        "                _p += 1",
        "            except Exception as _e:",
        "                if not _ff:",
        "                    _ff = 'test%d: %s: %s' % (_i, type(_e).__name__, _e)",
        f"    print('\\n' + {_SENTINEL!r}, _p, len(_tests), repr(_ff[:200]))",
    ]
    return _run_subprocess((code or "") + "\n" + "\n".join(harness_lines),
                           timeout=timeout)


def _sandbox_run_project(files: dict[str, str], tests: list[str],
                         timeout: float = 5.0) -> dict:
    """Multi-file test.  Writes all files into a tempdir, runs test harness."""
    harness = "\n".join([
        "",
        "if True:",
        "    import sys as _sys, os as _os",
        "    _sys.path.insert(0, _os.getcwd())",
        "    import traceback as _tb",
        f"    _tests = {tests!r}",
        "    _p, _ff = 0, ''",
        "    for _i, _src in enumerate(_tests):",
        "        try:",
        "            exec(compile(_src, '<test%d>' % _i, 'exec'), globals())",
        "            _p += 1",
        "        except Exception as _e:",
        "            if not _ff:",
        "                _ff = 'test%d: %s: %s' % (_i, type(_e).__name__, _e)",
        f"    print('\\n' + {_SENTINEL!r}, _p, len(_tests), repr(_ff[:200]))",
    ])
    with tempfile.TemporaryDirectory() as td:
        for rel, content in files.items():
            p = Path(td) / rel
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(content or "", encoding="utf-8")
        runner = Path(td) / "_runner.py"
        runner.write_text(harness, encoding="utf-8")
        res = _run_script(str(runner), cwd=td, timeout=timeout)
    return res


def _run_subprocess(script_text: str, timeout: float) -> dict:
    with tempfile.TemporaryDirectory() as td:
        script = Path(td) / "cand.py"
        script.write_text(script_text, encoding="utf-8")
        return _run_script(str(script), cwd=td, timeout=timeout)


def _run_script(path: str, cwd: str, timeout: float) -> dict:
    res = {"passed": False, "n_pass": 0, "n_total": 0,
           "first_fail": "", "stderr_tail": "", "dur_ms": 0}
    t0 = time.time()
    try:
        proc = subprocess.run(
            [sys.executable, "-I", path], cwd=cwd, capture_output=True,
            text=True, encoding="utf-8", errors="replace", timeout=timeout)
    except subprocess.TimeoutExpired as exc:
        res["first_fail"] = "timeout"
        res["stderr_tail"] = (exc.stderr or "")[-400:] if isinstance(exc.stderr, str) else ""
        res["dur_ms"] = int((time.time() - t0) * 1000)
        return res
    res["dur_ms"] = int((time.time() - t0) * 1000)
    res["stderr_tail"] = (proc.stderr or "")[-400:]
    line = next((ln for ln in reversed((proc.stdout or "").splitlines())
                 if ln.startswith(_SENTINEL)), "")
    if not line:
        res["first_fail"] = f"crash: exit {proc.returncode}"
        return res
    try:
        _, p, tot, ff_repr = line.split(" ", 3)
        res["n_pass"], res["n_total"] = int(p), int(tot)
        res["first_fail"] = ast.literal_eval(ff_repr)
    except Exception:
        res["first_fail"] = "sentinel parse error"
        return res
    res["passed"] = res["n_pass"] == res["n_total"] and res["n_total"] > 0 \
        and not res["first_fail"]
    return res


# ═══════════════════════════════════════════════════════════════════════════════
# INLINE COMPOSE INSTANCES  — self-contained task generation
# ═══════════════════════════════════════════════════════════════════════════════

_LINE_FMTS = [
    ('"ITEM-{:03d}".format(n)', 'like "ITEM-007"'),
    ('"item_{}".format(n)', 'like "item_7"'),
]
_TAXES = [0.05, 0.06, 0.07, 0.08, 0.09, 0.11, 0.12]
_FEES = [0.02, 0.03, 0.04, 0.10, 0.15]
_ID_FMTS = _LINE_FMTS


def _fmt_apply(expr_template: str, **kw) -> str:
    return eval(expr_template, {}, kw)


def _make_compose_instance(seed: int) -> dict:
    """Deterministic compose instance (same shape as project_gen._compose(0 distractors))."""
    rng = random.Random(seed)
    tax = rng.choice(_TAXES)
    fee = rng.choice(_FEES)
    while fee == tax:
        fee = rng.choice(_FEES)
    low = rng.randint(2, 9)
    id_i = rng.randrange(len(_ID_FMTS))
    id_expr, id_hint = _ID_FMTS[id_i]

    tax_gold = f"TAX_RATE = {tax}\n\ndef taxed(amount):\n    return round(amount * (1 + TAX_RATE), 2)\n"
    fees_gold = f"FEE_RATE = {fee}\n\ndef service_fee(amount):\n    return round(amount * FEE_RATE, 2)\n"
    catalog_gold = (
        f"def make_sku(n):\n"
        f"    return {id_expr}\n\n"
        f"def is_low(stock, name):\n"
        f"    return stock.get(name, 0) < {low}\n"
    )
    checkout_gold = f"def final_price(p):\n    return round(p * (1 + {tax} + {fee}), 2)\n"

    def oracle(p):
        return round(p * (1 + tax + fee), 2)

    ts = int(time.time() * 1000000) % 1000000
    sessions = [
        dict(kind="create", target_file="tax.py",
             spec=f"Create tax.py. Constant TAX_RATE = {tax}. taxed(amount) returns amount*(1+TAX_RATE) rounded to 2 decimals.",
             tests=[f"import tax\nassert tax.taxed(100) == {round(100 * (1 + tax), 2)}",
                    f"import tax\nassert tax.TAX_RATE == {tax}"],
             gold={"tax.py": tax_gold}, withheld=[], source_session_idx=None,
             source_session_idxs=[], sid=f"s0_{seed}_{ts}"),
        dict(kind="create", target_file="fees.py",
             spec=f"Create fees.py. Constant FEE_RATE = {fee}. service_fee(amount) returns amount*FEE_RATE rounded to 2 decimals.",
             tests=[f"import fees\nassert fees.service_fee(100) == {round(100 * fee, 2)}",
                    f"import fees\nassert fees.FEE_RATE == {fee}"],
             gold={"fees.py": fees_gold}, withheld=[], source_session_idx=None,
             source_session_idxs=[], sid=f"s1_{seed}_{ts}"),
        dict(kind="create", target_file="catalog.py",
             spec=f"Create catalog.py. make_sku(n) returns an id string {id_hint}. is_low(stock, name) is True when stock is strictly below {low}.",
             tests=[f"import catalog\nassert catalog.make_sku(7) == {_fmt_apply(id_expr, n=7)!r}",
                    f"import catalog\nassert catalog.is_low({{'a': {low - 1}}}, 'a') is True"],
             gold={"catalog.py": catalog_gold}, withheld=[], source_session_idx=None,
             source_session_idxs=[], sid=f"s2_{seed}_{ts}"),
        dict(kind="compose", target_file="checkout.py",
             spec=("Create checkout.py. final_price(p) returns the base price with this project's tax "
                   "charge on p plus this project's service fee charge on p — that is, the base price "
                   "with BOTH established rates applied to it and summed (base + tax-on-base + fee-on-base), "
                   "rounded to 2 decimals. Do NOT restate the rates; use the two values this project "
                   "already established."),
             tests=[f"import checkout\nassert checkout.final_price(100) == {oracle(100)}",
                    f"import checkout\nassert checkout.final_price(50) == {oracle(50)}",
                    f"import checkout\nassert checkout.final_price(8) == {oracle(8)}"],
             gold={"checkout.py": checkout_gold},
             withheld=[str(tax), str(fee)],
             source_session_idx=0, source_session_idxs=[0, 1],
             sid=f"s3_{seed}_{ts}"),
    ]
    return dict(archetype="compose", instance_id=f"compose_{seed}", sessions=sessions,
                params=dict(tax=tax, fee=fee, low=low, id_i=id_i))


# ═══════════════════════════════════════════════════════════════════════════════
# CORE LOGIC
# ═══════════════════════════════════════════════════════════════════════════════

TRAIN_SEEDS = range(100, 130)
EVAL_SEEDS = range(0, 20)

# Reward constants
PASS_REWARD = 1.0
HALLUCINATE_PENALTY = 1.0
PARTIAL_BASE = 0.3


def _source_file_paths(inst: dict) -> list[str]:
    s = inst["sessions"][3]
    idxs = list(s.get("source_session_idxs") or [])
    si = s.get("source_session_idx")
    if si is not None and si not in idxs:
        idxs = [si] + idxs
    seen, out = set(), []
    for idx in idxs:
        fp = inst["sessions"][idx].get("target_file", "")
        if fp and fp not in seen:
            out.append(fp); seen.add(fp)
    return out


def _repo_before_compose(inst: dict) -> dict[str, str]:
    repo = {}
    for i in range(3):
        repo.update(inst["sessions"][i].get("gold") or {})
    return repo


def build_compose_prompt(inst: dict, file_paths: list[str]) -> str:
    s = inst["sessions"][3]
    fps = ", ".join(file_paths)
    return (
        f"{s['spec']}\n\n"
        f"The following files already exist in the project: {fps}.\n"
        f"Write the requested file as a self-contained module that imports and uses "
        f"the constants and functions from the existing files as needed. "
        f"Output only the code, no explanation."
    )


def compute_code_reward(gen: str, repo: dict, tests: list[str],
                        target_file: str = "checkout.py") -> tuple[float, dict]:
    if not gen or not gen.strip():
        return -HALLUCINATE_PENALTY, {"verdict": "PUNISH (empty generation)"}
    try:
        res = _sandbox_run_project({**repo, target_file: gen}, tests)
    except Exception as e:
        return -HALLUCINATE_PENALTY, {"verdict": f"PUNISH (sandbox error: {e})"}
    n_pass, n_total = res["n_pass"], res["n_total"]
    if res["passed"] and n_total > 0:
        return PASS_REWARD, {"verdict": "REWARD (all pass)"}
    if n_pass > 0:
        frac = n_pass / n_total
        r = PARTIAL_BASE + (PASS_REWARD - PARTIAL_BASE) * frac
        return r, {"verdict": f"partial ({n_pass}/{n_total})"}
    fail = res.get("first_fail", "") or "no tests pass"
    return -HALLUCINATE_PENALTY, {"verdict": f"PUNISH ({fail})"}


def advantages(rewards: list[float]) -> list[float]:
    n = len(rewards)
    mean = sum(rewards) / n
    var = sum((r - mean) ** 2 for r in rewards) / n
    std = var ** 0.5 or 1.0
    return [(r - mean) / std for r in rewards]


# ═══════════════════════════════════════════════════════════════════════════════
# BASELINE EVAL  (frozen model, no LoRA)
# ═══════════════════════════════════════════════════════════════════════════════

def run_baseline_eval(model_name: str, n: int):
    import torch
    from transformers import AutoTokenizer
    from v5.lm_loader import load_frozen_lm

    trust = os.environ.get("V5_LM_TRUST_REMOTE_CODE", "0").lower() in ("1", "true", "yes")
    model = load_frozen_lm(model_name)
    model.eval()
    tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=trust)
    dev = next(model.parameters()).device

    print(f"FROZEN baseline on {n} compose instances (model={model_name})\n")
    passed, total = 0, 0
    rewards = []
    for seed in range(100, 100 + n):
        inst = _make_compose_instance(seed)
        fps = _source_file_paths(inst)
        prompt = build_compose_prompt(inst, fps)
        m = [{"role": "user", "content": prompt}]
        kw = dict(add_generation_prompt=True, return_tensors="pt", return_dict=True)
        try:
            enc = tok.apply_chat_template(m, enable_thinking=False, **kw)
        except TypeError:
            enc = tok.apply_chat_template(m, **kw)
        ids = enc["input_ids"].to(dev)
        with torch.no_grad():
            out_ids = model.generate(ids, do_sample=False, max_new_tokens=512,
                                     pad_token_id=tok.eos_token_id)
        gen = tok.decode(out_ids[0, ids.shape[1]:], skip_special_tokens=True).strip()
        gen = re.sub(r"<think>.*?</think>", "", gen, flags=re.DOTALL).strip()

        s = inst["sessions"][3]
        r, info = compute_code_reward(gen, _repo_before_compose(inst), s["tests"])
        rewards.append(r)
        total += 1
        if r >= PASS_REWARD - 0.01:
            passed += 1
        fp = " ;; ".join(fps)
        print(f"  [{inst['instance_id']}] sources={fp!r}  reward={r:+.2f} {info['verdict']}")

    mean_r = sum(rewards) / max(1, total)
    hallu = sum(1 for r in rewards if r < 0) / max(1, total)
    print(f"\n  mean reward={mean_r:+.2f} | solve-rate={passed}/{total}={passed/max(1,total):.0%} | "
          f"hallucination={hallu:.0%}")
    return mean_r


# ═══════════════════════════════════════════════════════════════════════════════
# TRAINING  (SFT + GRPO over LoRA)
# ═══════════════════════════════════════════════════════════════════════════════

def train(model_name, steps, K, lr, r_lora, seed, layers, eval_every, ent_coef,
          temperature, sft_steps):
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
    if not leaf:
        raise RuntimeError("no Linear leaf modules found inside .layers.*")
    print(f"LoRA target leaf modules (auto-detected): {leaf}", flush=True)
    cfg = LoraConfig(r=r_lora, lora_alpha=2 * r_lora, lora_dropout=0.0,
                     task_type="CAUSAL_LM",
                     target_modules=leaf, layers_to_transform=layers)
    model = get_peft_model(base, cfg); model.train()
    trainable = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(trainable, lr=lr)
    print(f"LoRA r={r_lora} on layers {layers} | "
          f"trainable params={sum(p.numel() for p in trainable):,}", flush=True)

    rng = random.Random(seed)
    held = [_make_compose_instance(s) for s in EVAL_SEEDS]
    train_seed_list = list(TRAIN_SEEDS)

    def encode(prompt):
        m = [{"role": "user", "content": prompt}]
        kw = dict(add_generation_prompt=True, return_tensors="pt", return_dict=True)
        try:
            enc = tok.apply_chat_template(m, enable_thinking=False, **kw)
        except TypeError:
            enc = tok.apply_chat_template(m, **kw)
        return enc["input_ids"].to(dev)

    def gen_ids(prompt_ids, sample):
        with torch.no_grad():
            return model.generate(prompt_ids, do_sample=sample,
                                  temperature=temperature if sample else None,
                                  top_p=0.95 if sample else None,
                                  max_new_tokens=512,
                                  pad_token_id=tok.eos_token_id)

    def seq_logprob(prompt_ids, comp_ids):
        full = torch.cat([prompt_ids, comp_ids], dim=1)
        logits = model(full).logits[:, :-1]
        logp = torch.log_softmax(logits.float(), dim=-1)
        start = prompt_ids.shape[1] - 1
        span = logp[:, start:start + comp_ids.shape[1]]
        sel = span.gather(-1, comp_ids.unsqueeze(-1)).squeeze(-1).sum(-1)
        ent = -(span.exp() * span).sum(-1).mean()
        return sel, ent

    @torch.no_grad()
    def evaluate():
        model.eval()
        rs = []
        for inst in held:
            fps = _source_file_paths(inst)
            p = build_compose_prompt(inst, fps)
            s = inst["sessions"][3]
            pids = encode(p)
            out = gen_ids(pids, sample=False)
            gen = tok.decode(out[0, pids.shape[1]:], skip_special_tokens=True).strip()
            gen = re.sub(r"<think>.*?</think>", "", gen, flags=re.DOTALL).strip()
            r, _ = compute_code_reward(gen, _repo_before_compose(inst), s["tests"])
            rs.append(r)
        model.train()
        return sum(rs) / max(1, len(rs)), sum(1 for r in rs if r < 0) / max(1, len(rs))

    base_mean, base_hall = evaluate()
    print(f"[eval @0] held-out mean reward={base_mean:+.3f} "
          f"hallucination={base_hall:.0%}", flush=True)

    # SFT warm-start
    if sft_steps > 0:
        ce = torch.nn.CrossEntropyLoss()
        for s in range(1, sft_steps + 1):
            seed_i = rng.choice(train_seed_list)
            inst = _make_compose_instance(seed_i)
            sess = inst["sessions"][3]
            target = sess["target_file"]
            gold_code = sess["gold"][target]
            fps = _source_file_paths(inst)
            p = build_compose_prompt(inst, fps)
            pids = encode(p)
            gids = tok(gold_code, return_tensors="pt", add_special_tokens=False).input_ids.to(dev)
            full = torch.cat([pids, gids], dim=1)
            logits = model(full).logits
            start = pids.shape[1]
            pred = logits[:, start - 1:start - 1 + gids.shape[1]].reshape(-1, logits.shape[-1])
            loss = ce(pred.float(), gids.reshape(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            opt.step(); opt.zero_grad()
            if s <= 5 or s % 25 == 0:
                print(f"[sft {s:3}] gold={target} ce_loss={float(loss.detach()):.3f}", flush=True)
        sm, sh = evaluate()
        print(f"[eval after SFT] held-out mean reward={sm:+.3f} "
              f"hallucination={sh:.0%}  (base {base_mean:+.3f}/{base_hall:.0%})", flush=True)

    # GRPO RL
    zero_var = 0
    for step in range(1, steps + 1):
        seed_i = rng.choice(train_seed_list)
        inst = _make_compose_instance(seed_i)
        sess = inst["sessions"][3]
        repo = _repo_before_compose(inst)
        tests = sess["tests"]
        fps = _source_file_paths(inst)
        prompt = build_compose_prompt(inst, fps)
        pids = encode(prompt)

        comps, rewards = [], []
        for _ in range(K):
            out = gen_ids(pids, sample=True)
            comp = out[:, pids.shape[1]:]
            gen = tok.decode(comp[0], skip_special_tokens=True).strip()
            gen = re.sub(r"<think>.*?</think>", "", gen, flags=re.DOTALL).strip()
            comps.append(comp)
            r, _ = compute_code_reward(gen, repo, tests)
            rewards.append(r)

        mean_r = sum(rewards) / K
        r_std = (sum((r - mean_r) ** 2 for r in rewards) / K) ** 0.5
        if r_std < 1e-9:
            zero_var += 1
            if steps <= 40 or step % 20 == 0:
                print(f"[step {step:3}] sources={' ;; '.join(fps)} "
                      f"mean_r={mean_r:+.2f} r_std=0 SKIP "
                      f"rewards={[round(r, 1) for r in rewards]}", flush=True)
            continue

        advs = advantages(rewards)
        loss = 0.0
        ent_total = 0.0
        for comp, a in zip(comps, advs):
            lp, ent = seq_logprob(pids, comp)
            loss = loss - a * lp - ent_coef * ent
            ent_total += float(ent.detach())
        loss = loss / K
        loss.backward()
        gnorm = torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        opt.step(); opt.zero_grad()

        if steps <= 40 or step % 20 == 0:
            print(f"[step {step:3}] sources={' ;; '.join(fps)} "
                  f"mean_r={mean_r:+.2f} r_std={r_std:.2f} "
                  f"ent={ent_total/K:.2f} loss={float(loss.detach()):+.3f} "
                  f"gnorm={float(gnorm):.3f} "
                  f"rewards={[round(r, 1) for r in rewards]}", flush=True)
        if step % eval_every == 0:
            m, h = evaluate()
            print(f"[eval @{step}] held-out mean reward={m:+.3f} "
                  f"hallucination={h:.0%}  (base {base_mean:+.3f}/{base_hall:.0%})", flush=True)

    m, h = evaluate()
    print(f"\n=== RL DONE === held-out mean reward {base_mean:+.3f}->{m:+.3f} | "
          f"hallucination {base_hall:.0%}->{h:.0%}")
    print(f"  zero-variance groups = {zero_var}/{steps} ({zero_var/steps:.0%})")
    out = "artifacts/compose_lora"
    model.save_pretrained(out)
    print(f"  LoRA saved -> {out}")


# ═══════════════════════════════════════════════════════════════════════════════
# SELFTEST  (no model, no external v5 modules)
# ═══════════════════════════════════════════════════════════════════════════════

def _selftest() -> bool:
    print("compose_rl --selftest: prove RL math (gen + reward + advantage), no model.\n")

    insts = [_make_compose_instance(s) for s in (100, 101, 102)]
    assert len(insts) == 3
    ok_gen = True
    for inst in insts:
        s = inst["sessions"][3]
        assert s["kind"] == "compose"
        assert s["target_file"] == "checkout.py"
        src = _source_file_paths(inst)
        assert "tax.py" in src and "fees.py" in src
        repo = _repo_before_compose(inst)
        assert "tax.py" in repo and "fees.py" in repo and "catalog.py" in repo
        target = s["target_file"]
        gold = s["gold"][target]
        res = _sandbox_run_project({**repo, target: gold}, s["tests"])
        if not res["passed"]:
            print(f"  [FAIL] gold fails: {res.get('first_fail','?')}")
            ok_gen = False
        res_fail = _sandbox_run_project({**repo, target: ""}, s["tests"])
        if res_fail["passed"]:
            print(f"  [FAIL] empty passes")
            ok_gen = False
    print(f"  [1] compose instances: valid={ok_gen} (gold passes / empty fails)")

    inst = insts[0]
    s = inst["sessions"][3]
    repo = _repo_before_compose(inst)
    gold = s["gold"][s["target_file"]]
    tests = s["tests"]
    r_gold, _ = compute_code_reward(gold, repo, tests)
    r_empty, _ = compute_code_reward("", repo, tests)
    r_garb, _ = compute_code_reward("def garbage():\n    pass\n", repo, tests)
    disc = r_gold > 0.5 and r_empty < 0 and r_garb < 0
    print(f"  [2] reward: gold->{r_gold:+.2f} empty->{r_empty:+.2f} garbage->{r_garb:+.2f}  disc={disc}")

    advs = advantages([1.0, 1.0, -1.0, -1.0])
    centered = abs(sum(advs)) < 1e-6
    monotone = advs[0] > advs[2]
    print(f"  [3] GRPO advantages([1,1,-1,-1]) = {[round(x,2) for x in advs]}  "
          f"centered={centered} monotone={monotone}")

    fps = _source_file_paths(inst)
    has_both = "tax.py" in fps and "fees.py" in fps
    print(f"  [4] source_file_paths -> {fps}  has_both={has_both}")

    prompt = build_compose_prompt(inst, fps)
    ok_fmt = s["spec"][:20] in prompt and "tax.py" in prompt and "already exist" in prompt.lower()
    print(f"  [5] prompt format: spec paths nl -> {'OK' if ok_fmt else 'BAD'}")

    ok = ok_gen and disc and centered and monotone and has_both and ok_fmt
    print(f"\n  COMPOSE RL SELFTEST -> {'PASS' if ok else 'FAIL'}")
    return ok


# ═══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser(
        description="GRPO RL for the COMPOSE generation step (fully self-contained).")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--baseline", action="store_true", help="eval frozen model (no RL)")
    ap.add_argument("--n-baseline", type=int, default=10)
    ap.add_argument("--model", default="Qwen/Qwen2.5-3B")
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--k", type=int, default=8, help="rollouts per task (GRPO group size)")
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--r-lora", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--layers", type=int, nargs="+", default=[20, 22, 24, 26, 28],
                    help="layers to put LoRA on")
    ap.add_argument("--eval-every", type=int, default=50)
    ap.add_argument("--ent-coef", type=float, default=0.005)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--sft-steps", type=int, default=0, help="SFT warm-start steps before RL")
    a = ap.parse_args()

    if a.selftest:
        sys.exit(0 if _selftest() else 1)
    if a.baseline:
        return run_baseline_eval(a.model, a.n_baseline)
    train(a.model, a.steps, a.k, a.lr, a.r_lora, a.seed, a.layers,
          a.eval_every, a.ent_coef, a.temperature, a.sft_steps)


if __name__ == "__main__":
    main()
