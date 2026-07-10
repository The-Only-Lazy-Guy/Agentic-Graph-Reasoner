"""GRPO RL for the COMPOSE code generation step (full traversal->compose pipeline).

The compose task (src: project_gen._compose): three source files are created (tax.py, fees.py,
catalog.py), then the model must generate checkout.py that composes constants from tax.py and
fees.py into a combined formula. The spec WITHHOLDS the actual rate values — the model must
infer which files to use and what they contain.

Architecture:
  1. Generate compose instances from project_gen, or run TraversalRanker for retrieval
  2. Build prompt: spec + source file_path hints
  3. Sample K completions (code) from the LM via LoRA
  4. Test each completion via sandbox.run_project (multi-file with sibling modules)
  5. Reward = test pass rate (0.0-1.0), with -hallucinate_penalty for failed
  6. GRPO update on the LoRA

Two retrieval modes:
  --use-retrieval : run TraversalRanker to retrieve source file_paths (needs --ranker + --embed)
  default         : use gold source file paths directly (cleaner RL signal, no model needed)

Usage:
  selftest (no model):  python -m v5.runtime.compose_rl --selftest
  baseline eval:        python -m v5.runtime.compose_rl --baseline
  train (A40):          V5_LM_TRUST_REMOTE_CODE=1 python -m v5.runtime.compose_rl \\
                          --steps 200 --k 8 --sft-steps 50
  with retrieval:       ... --use-retrieval --ranker artifacts/traversal_ranker
"""
from __future__ import annotations

import argparse
import os
import random
import re
from pathlib import Path

from v5.runtime.project_gen import make_split
from v5.runtime.sandbox import run_project


# ── constants ────────────────────────────────────────────────────────────────────

TRAIN_SEEDS = range(100, 130)        # training seeds (30 instances)
EVAL_SEEDS = range(0, 20)           # held-out eval seeds (20 instances, disjoint from train)
COMPOSE_SESSION_DEPTH = 3           # compose is at depth 3 (after tax, fees, catalog)

# Reward constants
PASS_REWARD = 1.0
HALLUCINATE_PENALTY = 1.0
PARTIAL_BASE = 0.3                  # reward floor when at least some tests pass


def _source_file_paths(inst: dict) -> list[str]:
    """Extract the source file paths needed by the compose session."""
    s = inst["sessions"][COMPOSE_SESSION_DEPTH]
    idxs = s.get("source_session_idxs") or []
    if s.get("source_session_idx") is not None:
        if s["source_session_idx"] not in idxs:
            idxs = [s["source_session_idx"]] + idxs
    seen = set()
    paths = []
    for idx in idxs:
        fp = inst["sessions"][idx].get("target_file", "")
        if fp and fp not in seen:
            paths.append(fp)
            seen.add(fp)
    return paths


def _repo_before_compose(inst: dict) -> dict[str, str]:
    """Build the repo state BEFORE the compose session — gold files for all earlier sessions."""
    repo = {}
    for i in range(COMPOSE_SESSION_DEPTH):
        s = inst["sessions"][i]
        repo.update(s.get("gold") or {})
    return repo


def build_compose_prompt(inst: dict, file_paths: list[str]) -> str:
    """Build a natural-language prompt for the compose task (chat-template friendly).

    Describes the task, which files exist in the project, and asks for the code.
    """
    s = inst["sessions"][COMPOSE_SESSION_DEPTH]
    spec = s["spec"]
    fps_joined = ", ".join(file_paths)
    return (
        f"{spec}\n\n"
        f"The following files already exist in the project: {fps_joined}.\n"
        f"Write the requested file as a self-contained module that imports and uses "
        f"the constants and functions from the existing files as needed. "
        f"Output only the code, no explanation."
    )


def compute_code_reward(gen: str, repo: dict, tests: list[str],
                        target_file: str = "checkout.py") -> tuple[float, dict]:
    """Score a generated code completion by running tests via sandbox.

    Args:
        gen: generated code content.
        repo: dict of {file_path: content} for sibling modules.
        tests: list of test assert strings.
        target_file: the filename the generated code belongs to (default checkout.py).

    Reward:
      +1.0  all tests pass
       0.3+P  some tests pass (partial credit scales with fraction)
      -1.0   code crashes / syntax error / no tests pass
    """
    if not gen or not gen.strip():
        return -HALLUCINATE_PENALTY, {"verdict": "PUNISH (empty generation)"}
    try:
        files = {**repo, target_file: gen}
        res = run_project(files, tests)
    except Exception as e:
        return -HALLUCINATE_PENALTY, {"verdict": f"PUNISH (sandbox error: {e})"}
    n_pass, n_total = res["n_pass"], res["n_total"]
    passed = res["passed"]
    if passed and n_total > 0:
        return PASS_REWARD, {"verdict": "REWARD (all pass)", "n_pass": n_pass, "n_total": n_total}
    if n_pass > 0:
        frac = n_pass / n_total
        r = PARTIAL_BASE + (PASS_REWARD - PARTIAL_BASE) * frac
        return r, {"verdict": f"partial ({n_pass}/{n_total})", "n_pass": n_pass, "n_total": n_total}
    fail_msg = res.get("first_fail", "") or "no tests pass"
    return -HALLUCINATE_PENALTY, {"verdict": f"PUNISH ({fail_msg})"}


def advantages(rewards: list[float]) -> list[float]:
    """GROUP baseline: advantage = reward - group mean, /std (GRPO). Reused from derive_rl."""
    n = len(rewards)
    mean = sum(rewards) / n
    var = sum((r - mean) ** 2 for r in rewards) / n
    std = var ** 0.5 or 1.0
    return [(r - mean) / std for r in rewards]


# ── baseline eval: frozen model (no LoRA) performance ───────────────────────────

def run_baseline_eval(model_name: str, n: int, use_retrieval: bool = False,
                      ranker_path: str = "", seed: int = 0):
    """Evaluate the FROZEN model on n compose instances. Measures how often the base model
    solves the compose task without any RL training."""
    import torch
    from transformers import AutoTokenizer
    from v5.lm_loader import load_frozen_lm

    trust = os.environ.get("V5_LM_TRUST_REMOTE_CODE", "0").lower() in ("1", "true", "yes")
    model = load_frozen_lm(model_name)
    model.eval()
    tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=trust)
    dev = next(model.parameters()).device

    insts = make_split(archetypes=("compose",), seeds=range(100, 100 + n))
    print(f"FROZEN baseline on {len(insts)} compose instances (model={model_name})\n")

    retrieval = _make_retrieval(ranker_path) if use_retrieval else None

    passed, total = 0, 0
    rewards = []
    for inst in insts:
        repo = _repo_before_compose(inst)
        if use_retrieval and retrieval is not None:
            file_paths = _retrieve_sources(inst, retrieval)
        else:
            file_paths = _source_file_paths(inst)
        prompt = build_compose_prompt(inst, file_paths)

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

        s = inst["sessions"][COMPOSE_SESSION_DEPTH]
        r, info = compute_code_reward(gen, repo, s["tests"])
        rewards.append(r)
        total += 1
        if r >= PASS_REWARD - 0.01:
            passed += 1
        fp = " ;; ".join(file_paths)
        print(f"  [{inst['instance_id']}] sources={fp!r}")
        print(f"    reward={r:+.2f} {info['verdict']}")

    mean_r = sum(rewards) / max(1, total)
    hallu = sum(1 for r in rewards if r < 0) / max(1, total)
    print(f"\n  mean reward={mean_r:+.2f} | solve-rate={passed}/{total}={passed/max(1,total):.0%} | "
          f"hallucination={hallu:.0%}")
    return mean_r


# ── retrieval helpers (optional traversal path) ──────────────────────────────────

class _Retrieval:
    """Container for trained ranker components needed at inference."""
    def __init__(self, ranker, embed_fn, gap_detector=None):
        self.ranker = ranker
        self.embed_fn = embed_fn
        self.gap_detector = gap_detector


def _make_retrieval(ranker_path: str):
    """Load trained ranker and embedder for traversal retrieval."""
    from v5.memory.store import make_mpnet_embedder
    from v5.runtime.memory_refiner import load_ranker

    ranker = load_ranker(ranker_path)
    embed_fn = make_mpnet_embedder()
    gap_detector = None
    gap_path = os.path.join(ranker_path, "gap.pt")
    if os.path.exists(gap_path):
        from v5.runtime.gap_detector import GapDetector
        import torch
        gap_detector = GapDetector(d_hidden=256, d_in=768)
        gap_detector.load_state_dict(
            torch.load(gap_path, weights_only=True, map_location="cpu"))
        gap_detector.eval()
    return _Retrieval(ranker, embed_fn, gap_detector)


def _retrieve_sources(inst: dict, retrieval: _Retrieval) -> list[str]:
    """Run TraversalRanker to retrieve source file paths for a compose instance.

    Builds a temporary TotalMemory with gold source sessions written, runs traversal,
    returns the file_paths of retrieved records.
    """
    import shutil
    from v5.memory.memory import TotalMemory
    from v5.runtime.traversal_ranker import TraversalRanker

    tmp = Path("__compose_rl_tmp") / inst["instance_id"]
    if tmp.exists():
        shutil.rmtree(tmp)
    memory = TotalMemory(tmp / "mem", mode="concept", embed_fn=retrieval.embed_fn)

    net, feat_proj, ops, K_r = retrieval.ranker
    traversal = TraversalRanker(memory.impls, memory.concepts, retrieval.embed_fn,
                                net, feat_proj, ops,
                                gap_detector=retrieval.gap_detector)

    repo = {}
    for i in range(COMPOSE_SESSION_DEPTH):
        s = inst["sessions"][i]
        target = s["target_file"]
        body = s["gold"][target]
        repo[target] = body
        memory.write(goal=s["spec"], old="", new=body,
                     trace=s["spec"][:400], verified=True,
                     file_path=target, task_id=s["sid"], kind=s["kind"])
    import time
    time.sleep(0.1)                                               # let memory index settle

    sess = inst["sessions"][COMPOSE_SESSION_DEPTH]
    t_result = traversal.retrieve(
        goal=sess["spec"],
        span=sess["spec"],
        file_path=sess["target_file"])

    shutil.rmtree(tmp, ignore_errors=True)
    return [str(r.get("file_path", "")) for r in t_result.records if r.get("file_path")]


# ── training ─────────────────────────────────────────────────────────────────────

def train(model_name, steps, K, lr, r_lora, seed, layers, eval_every, ent_coef,
          temperature, sft_steps, use_retrieval, ranker_path):
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
    held = make_split(archetypes=("compose",), seeds=EVAL_SEEDS)
    train_seed_list = list(TRAIN_SEEDS)

    retrieval = _make_retrieval(ranker_path) if use_retrieval else None

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
    def evaluate(held_insts):
        model.eval()
        rs = []
        for inst in held_insts:
            if use_retrieval and retrieval is not None:
                fps = _retrieve_sources(inst, retrieval)
            else:
                fps = _source_file_paths(inst)
            p = build_compose_prompt(inst, fps)
            s = inst["sessions"][COMPOSE_SESSION_DEPTH]
            pids = encode(p)
            out = gen_ids(pids, sample=False)
            gen = tok.decode(out[0, pids.shape[1]:], skip_special_tokens=True).strip()
            gen = re.sub(r"<think>.*?</think>", "", gen, flags=re.DOTALL).strip()
            r, _ = compute_code_reward(gen, _repo_before_compose(inst), s["tests"])
            rs.append(r)
        model.train()
        return sum(rs) / max(1, len(rs)), sum(1 for r in rs if r < 0) / max(1, len(rs))

    base_mean, base_hall = evaluate(held)
    print(f"[eval @0] held-out mean reward={base_mean:+.3f} "
          f"hallucination={base_hall:.0%}", flush=True)

    # ── SFT WARM-START ──
    if sft_steps > 0:
        ce = torch.nn.CrossEntropyLoss()
        for s in range(1, sft_steps + 1):
            seed_i = rng.choice(train_seed_list)
            inst = make_split(archetypes=("compose",), seeds=range(seed_i, seed_i + 1))[0]
            sess = inst["sessions"][COMPOSE_SESSION_DEPTH]
            target = sess["target_file"]
            gold_code = sess["gold"][target]
            if use_retrieval and retrieval is not None:
                fps = _retrieve_sources(inst, retrieval)
            else:
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
                print(f"[sft {s:3}] gold={target} ce_loss={float(loss.detach()):.3f}",
                      flush=True)
        sm, sh = evaluate(held)
        print(f"[eval after SFT] held-out mean reward={sm:+.3f} "
              f"hallucination={sh:.0%}  (base {base_mean:+.3f}/{base_hall:.0%})",
              flush=True)

    # ── GRPO RL ──
    zero_var = 0
    for step in range(1, steps + 1):
        seed_i = rng.choice(train_seed_list)
        inst = make_split(archetypes=("compose",), seeds=range(seed_i, seed_i + 1))[0]
        sess = inst["sessions"][COMPOSE_SESSION_DEPTH]
        repo = _repo_before_compose(inst)
        tests = sess["tests"]

        if use_retrieval and retrieval is not None:
            file_paths = _retrieve_sources(inst, retrieval)
        else:
            file_paths = _source_file_paths(inst)
        prompt = build_compose_prompt(inst, file_paths)
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
                print(f"[step {step:3}] sources={' ;; '.join(file_paths)} "
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
            print(f"[step {step:3}] sources={' ;; '.join(file_paths)} "
                  f"mean_r={mean_r:+.2f} r_std={r_std:.2f} "
                  f"ent={ent_total/K:.2f} loss={float(loss.detach()):+.3f} "
                  f"gnorm={float(gnorm):.3f} "
                  f"rewards={[round(r, 1) for r in rewards]}", flush=True)
        if step % eval_every == 0:
            m, h = evaluate(held)
            print(f"[eval @{step}] held-out mean reward={m:+.3f} "
                  f"hallucination={h:.0%}  (base {base_mean:+.3f}/{base_hall:.0%})",
                  flush=True)

    m, h = evaluate(held)
    print(f"\n=== RL DONE === held-out mean reward {base_mean:+.3f}->{m:+.3f} | "
          f"hallucination {base_hall:.0%}->{h:.0%}")
    print(f"  zero-variance groups = {zero_var}/{steps} ({zero_var/steps:.0%})")
    out = "artifacts/compose_lora"
    model.save_pretrained(out)
    print(f"  LoRA saved -> {out}")


# ── selftest (no model) ──────────────────────────────────────────────────────────

def _selftest() -> bool:
    """Validate the RL math: task gen, repo structure, reward discrimination, GRPO advantages.

    No model required — tests the data pipeline and reward function offline.
    """
    print("compose_rl --selftest: prove RL math (gen + reward + advantage), no model.\n")

    # 1. Compose instances are valid
    insts = make_split(archetypes=("compose",), seeds=range(100, 103))
    assert len(insts) == 3, f"expected 3 compose instances, got {len(insts)}"
    ok_gen = True
    for inst in insts:
        s = inst["sessions"][COMPOSE_SESSION_DEPTH]
        assert s["kind"] == "compose", f"session {COMPOSE_SESSION_DEPTH} should be compose"
        assert s["target_file"] == "checkout.py", f"target should be checkout.py"
        src = _source_file_paths(inst)
        assert "tax.py" in src and "fees.py" in src, \
            f"compose should need tax.py and fees.py, got {src}"
        repo = _repo_before_compose(inst)
        assert "tax.py" in repo and "fees.py" in repo and "catalog.py" in repo, \
            f"repo should contain all source files"
        # Gold code passes its own tests
        target = s["target_file"]
        gold_code = s["gold"][target]
        res = run_project({**repo, target: gold_code}, s["tests"])
        if not res["passed"]:
            print(f"  [FAIL] {inst['instance_id']}: gold code does not pass tests: "
                  f"{res.get('first_fail', '?')}")
            ok_gen = False
        # Buggy code should fail (empty generation)
        res_fail = run_project({**repo, target: ""}, s["tests"])
        if res_fail["passed"]:
            print(f"  [FAIL] {inst['instance_id']}: empty code passed tests (impossible)")
            ok_gen = False
    print(f"  [1] compose instance gen: valid={ok_gen} ({len(insts)} instances, "
          f"gold passes / empty fails)")

    # 2. Reward discriminates
    inst = insts[0]
    s = inst["sessions"][COMPOSE_SESSION_DEPTH]
    repo = _repo_before_compose(inst)
    target = s["target_file"]
    gold_code = s["gold"][target]
    tests = s["tests"]

    r_gold, info_gold = compute_code_reward(gold_code, repo, tests)
    r_empty, info_empty = compute_code_reward("", repo, tests)
    r_garbage, info_garb = compute_code_reward("def garbage():\n    pass\n", repo, tests)
    disc = r_gold > 0.5 and r_empty < 0 and r_garbage < 0
    print(f"  [2] reward: gold->{r_gold:+.2f} ({info_gold['verdict']})  "
          f"empty->{r_empty:+.2f} ({info_empty['verdict']})  "
          f"garbage->{r_garbage:+.2f} ({info_garb['verdict']})  "
          f"discriminates={disc}")

    # 3. GRPO advantages
    advs = advantages([PASS_REWARD, PASS_REWARD, -HALLUCINATE_PENALTY, -HALLUCINATE_PENALTY])
    centered = abs(sum(advs)) < 1e-6
    monotone = advs[0] > advs[2]
    print(f"  [3] GRPO advantages([1,1,-1,-1]) = {[round(x, 2) for x in advs]}  "
          f"centered={centered} monotone={monotone}")

    # 4. Source file paths extraction
    fps = _source_file_paths(inst)
    has_both = "tax.py" in fps and "fees.py" in fps
    print(f"  [4] source_file_paths -> {fps}  has_both={has_both}")

    # 5. Build prompt format
    prompt = build_compose_prompt(inst, fps)
    has_spec = s["spec"][:20] in prompt
    has_paths = "tax.py" in prompt and "fees.py" in prompt
    has_nl = "already exist" in prompt.lower()
    fmt_ok = has_spec and has_paths and has_nl
    print(f"  [5] prompt format: spec={has_spec} paths={has_paths} natural_language={has_nl} -> "
          f"{'OK' if fmt_ok else 'BAD'}")

    ok = ok_gen and disc and centered and monotone and has_both and fmt_ok
    print(f"\n  COMPOSE RL SELFTEST -> {'PASS' if ok else 'FAIL'}")
    return ok


# ── entry point ──────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="GRPO RL for the COMPOSE generation step (traversal->compose pipeline).")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--baseline", action="store_true",
                    help="eval frozen model on compose instances (no RL)")
    ap.add_argument("--n-baseline", type=int, default=10)
    ap.add_argument("--model", default="Qwen/Qwen2.5-3B")
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--k", type=int, default=8, help="rollouts per task (GRPO group size)")
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--r-lora", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--layers", type=int, nargs="+", default=[20, 22, 24, 26, 28],
                    help="layers to put LoRA on (the compose/inject band)")
    ap.add_argument("--eval-every", type=int, default=50)
    ap.add_argument("--ent-coef", type=float, default=0.005)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--sft-steps", type=int, default=0,
                    help="SFT warm-start steps before RL")
    ap.add_argument("--use-retrieval", action="store_true",
                    help="run TraversalRanker for source retrieval (needs --ranker)")
    ap.add_argument("--ranker", default="",
                    help="memory_refiner checkpoint dir (needed for --use-retrieval)")
    a = ap.parse_args()

    if a.selftest:
        import sys
        sys.exit(0 if _selftest() else 1)
    if a.baseline:
        return run_baseline_eval(a.model, a.n_baseline, a.use_retrieval, a.ranker, a.seed)
    if a.use_retrieval and not a.ranker:
        ap.error("--use-retrieval requires --ranker <checkpoint dir>")
    train(a.model, a.steps, a.k, a.lr, a.r_lora, a.seed, a.layers,
          a.eval_every, a.ent_coef, a.temperature, a.sft_steps,
          a.use_retrieval, a.ranker)


if __name__ == "__main__":
    main()
