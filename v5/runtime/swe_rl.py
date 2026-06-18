"""SWE RL — SFT-then-GRPO over a LoRA, on real SWE-bench coding tasks, scored by swe_reward.

Same recipe that closed the arithmetic gap (derive_rl), now on code: per instance the frozen 4B emits
SEARCH/REPLACE patches; reward = swe_reward (applies x solves x real-edit; unapplyable hunks punished).
SFT warm-start on the gold patch (gold-diff -> SR), then GRPO refines the LoRA.

Two reward/eval modes are supported:
  - proxy    : cheap in-loop reward via gold-overlap (default; the practical A40 mode)
  - verifier : exact pass/fail via the real SWE verifier (Docker or hosted sb-cli)

That lets the same loop run in the realistic split-box setup:
  * A40 training box:       keep --reward-mode proxy, optionally do no exact eval here
  * Docker / sb-cli box:    run exact held-out eval (--verify-every / final verify)
  * unified GPU+verifier:   use --reward-mode verifier if you truly want the apex in-loop reward
  * split boxes:            emit held predictions during training, then score them with swe_verify elsewhere

  selftest (no model):  python -m v5.runtime.swe_rl --selftest
  proxy train (A40):    V5_LM_TRUST_REMOTE_CODE=1 python -m v5.runtime.swe_rl --sft-steps 120 --steps 120 --k 6
  exact eval/reward:    ... --reward-mode verifier --verify-backend docker
"""
from __future__ import annotations

import argparse
import hashlib
import os
import subprocess
import sys
import time
from pathlib import Path

from v5.graph_grower.swe_verify import DATASET_MAP, parse_report, run_sbcli, write_predictions
from v5.runtime.derive_rl import advantages       # reuse the proven GRPO advantage
from v5.runtime.search_replace import SR_SYS, apply_sr, parse_sr
from v5.runtime.swe_reward import swe_reward


def gold_to_sr(diff: str) -> list[dict]:
    """unified diff -> SEARCH/REPLACE blocks. SEARCH = context+removed (the original), REPLACE =
    context+added (the new), per hunk; file = the +++ path."""
    blocks, file, lines, i = [], None, diff.splitlines(), 0
    while i < len(lines):
        l = lines[i]
        if l.startswith("+++ "):
            file = l[4:].strip().split("\t")[0]
            if file.startswith("b/"):
                file = file[2:]
            i += 1
        elif l.startswith("@@"):
            search, replace = [], []
            i += 1
            while i < len(lines) and not lines[i].startswith(("@@", "--- ", "diff ")):
                ln = lines[i]
                if ln.startswith("+"):
                    replace.append(ln[1:])
                elif ln.startswith("-"):
                    search.append(ln[1:])
                elif ln.startswith(" "):
                    search.append(ln[1:]); replace.append(ln[1:])
                else:
                    break
                i += 1
            if file and (search or replace):
                blocks.append({"file": file, "search": "\n".join(search), "replace": "\n".join(replace)})
        else:
            i += 1
    return blocks


def sr_to_text(blocks: list[dict]) -> str:
    return "\n\n".join(f"{b['file']}\n<<<<<<< SEARCH\n{b['search']}\n=======\n{b['replace']}\n>>>>>>> REPLACE"
                       for b in blocks)


def _restore_repo(repo_dir: str) -> None:
    subprocess.run(["git", "-C", repo_dir, "checkout", "--", "."], capture_output=True, text=True)


def materialize_patch(task: dict, blocks: list[dict]) -> tuple[int, str]:
    """Apply SR blocks to the checked-out repo, capture the git diff, then restore the checkout."""
    dest = task["dest"]
    _restore_repo(dest)
    try:
        applied, patch = apply_sr(dest, blocks)
    finally:
        _restore_repo(dest)
    return applied, patch


class SWEExactVerifier:
    """Small adapter around the existing swe_verify harness, with preflight + caching.

    `verify_patch()` is for one patch on one instance (used by exact rollout reward).
    `verify_task_batch_unique()` is for one patch per unique instance (used by held-out eval / gold sanity).
    """

    def __init__(self, dataset: str, split: str, backend: str, out_dir: str,
                 max_workers: int = 4, timeout: int = 1800, poll_secs: int = 20,
                 model_name: str = "swe_rl") -> None:
        self.dataset = dataset
        self.split = split
        self.backend = backend
        self.out_dir = Path(out_dir)
        self.max_workers = max_workers
        self.timeout = timeout
        self.poll_secs = poll_secs
        self.model_name = model_name
        self.cache: dict[tuple[str, str], bool] = {}
        self._counter = 0
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self._preflight()

    def _preflight(self) -> None:
        if self.backend == "docker":
            try:
                import swebench  # noqa: F401
            except ImportError as exc:  # pragma: no cover - environment-dependent
                raise SystemExit("FATAL: swebench not installed; exact verifier mode needs `pip install swebench`.") from exc
            if subprocess.run(["docker", "info"], capture_output=True, text=True).returncode != 0:
                raise SystemExit("FATAL: Docker daemon not reachable; use --verify-backend sbcli or stay in proxy mode.")
        else:
            if subprocess.run(["sb-cli", "--help"], capture_output=True, text=True).returncode != 0:
                raise SystemExit("FATAL: sb-cli not installed; exact verifier mode needs `pip install sb-cli`.")
            if not os.environ.get("SWEBENCH_API_KEY"):
                print("WARN: SWEBENCH_API_KEY unset; hosted sb-cli submit may fail.", flush=True)

    def _slug(self, text: str) -> str:
        keep = [c.lower() if c.isalnum() else "_" for c in (text or "run")]
        s = "".join(keep).strip("_")
        return s[:48] or "run"

    def _next_run_id(self, tag: str) -> str:
        self._counter += 1
        return f"{self._slug(tag)}_{int(time.time())}_{self._counter:05d}"

    def _pred_path(self, run_id: str) -> Path:
        return self.out_dir / f"{run_id}.predictions.jsonl"

    def _run_docker(self, preds_path: str, run_id: str, instance_ids: list[str]) -> dict[str, bool]:
        ds = DATASET_MAP.get(self.dataset, self.dataset)
        cmd = [sys.executable, "-m", "swebench.harness.run_evaluation",
               "--dataset_name", ds, "--predictions_path", preds_path,
               "--run_id", run_id, "--max_workers", str(self.max_workers),
               "--cache_level", "env"]
        if instance_ids:
            cmd += ["--instance_ids", *instance_ids]
        proc = subprocess.run(cmd, text=True, capture_output=True, timeout=self.timeout)
        if proc.returncode != 0:
            print(f"WARN: swebench harness exited {proc.returncode} for run_id={run_id}", flush=True)
            if proc.stdout:
                sys.stdout.write(proc.stdout[-3000:])
            if proc.stderr:
                sys.stderr.write(proc.stderr[-1500:])
        return parse_report(self.model_name, run_id)

    def _run(self, preds_path: str, run_id: str, instance_ids: list[str]) -> dict[str, bool]:
        if self.backend == "sbcli":
            return run_sbcli(preds_path, self.dataset, run_id, split=self.split,
                             out_dir=str(self.out_dir), submit=True,
                             poll_secs=self.poll_secs, max_wait=self.timeout)
        return self._run_docker(preds_path, run_id, instance_ids)

    def verify_patch(self, task: dict, patch: str, tag: str = "reward") -> bool:
        key = (task["iid"], hashlib.sha1(patch.encode("utf-8")).hexdigest())
        if key in self.cache:
            return self.cache[key]
        run_id = self._next_run_id(f"{tag}_{task['iid']}")
        preds_path = self._pred_path(run_id)
        n = write_predictions({task["iid"]: patch}, str(preds_path), model_name=self.model_name)
        if n <= 0:
            self.cache[key] = False
            return False
        res = self._run(str(preds_path), run_id, [task["iid"]])
        ok = bool(res.get(task["iid"], False))
        self.cache[key] = ok
        return ok

    def verify_task_batch_unique(self, task_patches: list[tuple[dict, str]], tag: str = "eval") -> dict[str, bool]:
        results: dict[str, bool] = {}
        id2patch: dict[str, str] = {}
        key_by_iid: dict[str, tuple[str, str]] = {}
        for task, patch in task_patches:
            key = (task["iid"], hashlib.sha1(patch.encode("utf-8")).hexdigest())
            if key in self.cache:
                results[task["iid"]] = self.cache[key]
                continue
            if task["iid"] in id2patch:
                raise ValueError("verify_task_batch_unique requires at most one patch per instance_id")
            id2patch[task["iid"]] = patch
            key_by_iid[task["iid"]] = key
        if id2patch:
            run_id = self._next_run_id(tag)
            preds_path = self._pred_path(run_id)
            n = write_predictions(id2patch, str(preds_path), model_name=self.model_name)
            if n > 0:
                fresh = self._run(str(preds_path), run_id, list(id2patch))
            else:
                fresh = {}
            for iid, key in key_by_iid.items():
                ok = bool(fresh.get(iid, False))
                self.cache[key] = ok
                results[iid] = ok
        return results


def _verify_blocks_exact(task: dict, blocks: list[dict], verifier: SWEExactVerifier) -> bool:
    applied, patch = materialize_patch(task, blocks)
    if applied <= 0 or not patch.strip():
        return False
    return verifier.verify_patch(task, patch, tag="reward")


def score(completion: str, task: dict, reward_mode: str = "proxy",
          verifier: SWEExactVerifier | None = None):
    blocks = parse_sr(completion)
    if reward_mode == "verifier":
        if verifier is None:
            raise ValueError("reward_mode='verifier' requires an exact verifier")
        return swe_reward(blocks, task["src"], solves_fn=lambda _blocks: _verify_blocks_exact(task, blocks, verifier))
    return swe_reward(blocks, task["src"], gold_patch=task["gold"])


def load_swe_tasks(n, traces_p, nodes_p, dataset, repo_root, src_bodies, src_lines):
    from v5.graph_grower.swe_load import load_instances, checkout_repo
    from v5.graph_grower.swe_probe import load_traces
    from v5.runtime.sr_withcode import load_symbol_meta, read_body
    traces = load_traces([traces_p]); meta = load_symbol_meta([nodes_p])
    insts = {t["instance_id"]: t for t in load_instances(dataset, "test", limit=0)}
    ids = [i for i in traces if i in insts and all(s in meta for s in traces[i]["support_ids"])]
    tasks = []
    for iid in ids[:n]:
        t = traces[iid]; inst = insts[iid]
        dest = Path(repo_root) / inst["repo"].replace("/", "__")
        ok, _ = checkout_repo(inst["repo"], inst["base_commit"], dest, timeout=1800)
        if not ok:
            continue
        support = [s for s in t["support_ids"] if s in meta]
        src = "\n\n".join(f"# {meta[s]['file']}\n{body}" for s in support[:src_bodies]
                          if (body := read_body(str(dest), meta[s]["file"], meta[s]["lineno"], src_lines)))
        gold_sr = gold_to_sr(inst.get("patch", ""))
        if not src.strip() or not gold_sr:
            continue
        tasks.append({"iid": iid, "issue": t["issue"], "src": src, "gold": inst.get("patch", ""),
                      "gold_sr_text": sr_to_text(gold_sr), "dest": str(dest)})
    return tasks


def _selftest():
    print("swe_rl --selftest: gold-diff -> SR roundtrip + reward + GRPO advantage (no model).\n")
    SRC = ("class A:\n    def deconstruct(self):\n        return handle_mask(self.mask)\n    def other(self):\n        pass\n")
    GOLD = ("--- a/x.py\n+++ b/x.py\n@@ -2,2 +2,3 @@\n     def deconstruct(self):\n"
            "-        return handle_mask(self.mask)\n"
            "+        if self.mask is None: return deepcopy(operand.mask)\n"
            "+        return handle_mask(self.mask, operand.mask)\n")
    blocks = gold_to_sr(GOLD)
    print(f"  gold_to_sr -> {len(blocks)} block(s); file={blocks[0]['file'] if blocks else None}")
    text = sr_to_text(blocks)
    from v5.runtime.search_replace import parse_sr
    reparsed = parse_sr(text)
    roundtrip = len(reparsed) == len(blocks) and reparsed[0].get("search", "").strip() in SRC
    r, b = swe_reward(blocks, SRC, gold_patch=GOLD)
    advs = advantages([1.5, 1.5, -1.0, 0.0])
    ok = (len(blocks) == 1 and roundtrip and r > 1.0 and abs(sum(advs)) < 1e-6)
    print(f"  roundtrip (gold SR re-parses + SEARCH in source): {roundtrip}")
    print(f"  reward on the gold patch: {r:+.2f}  {b['verdict']}")
    print(f"  GRPO advantages([1.5,1.5,-1,0]) = {[round(x,2) for x in advs]} (centered={abs(sum(advs))<1e-6})")
    print(f"\n  SWE-RL SELFTEST -> {'PASS' if ok else 'FAIL'}")
    return ok


def train(model_name, tasks, steps, K, lr, r_lora, seed, layers, eval_every, ent_coef, temperature,
          sft_steps, max_new, reward_mode: str = "proxy", verifier: SWEExactVerifier | None = None,
          verify_every: int = 0, verify_gold_sanity: int = 0,
          emit_preds_dir: str = "", emit_preds_every: int = 0):
    import random, torch, torch.nn as nn
    from transformers import AutoTokenizer
    from peft import LoraConfig, get_peft_model
    try:
        from peft import prepare_model_for_kbit_training
    except ImportError:  # older PEFTs
        prepare_model_for_kbit_training = None
    from v5.lm_loader import load_frozen_lm, resolve_dtype, resolve_quant
    from v5.runtime.swe_slot import fix_user

    trust = os.environ.get("V5_LM_TRUST_REMOTE_CODE", "0").lower() in ("1", "true", "yes")
    base = load_frozen_lm(model_name)
    tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=trust)
    dev = next(base.parameters()).device
    qmode = resolve_quant(None)
    dtype = resolve_dtype(dev)
    if qmode in ("4bit", "8bit") and prepare_model_for_kbit_training is not None:
        base = prepare_model_for_kbit_training(base)
    leaf = sorted({n.split(".")[-1] for n, m in base.named_modules()
                   if isinstance(m, nn.Linear) and ".layers." in n and not any(x in n.lower() for x in ("lm_head", "embed"))})
    cfg = LoraConfig(r=r_lora, lora_alpha=2 * r_lora, lora_dropout=0.0, task_type="CAUSAL_LM",
                     target_modules=leaf, layers_to_transform=layers)
    model = get_peft_model(base, cfg); model.train()
    if hasattr(model, "config") and hasattr(model.config, "use_cache"):
        model.config.use_cache = False
    if hasattr(model, "gradient_checkpointing_enable"):
        try:
            model.gradient_checkpointing_enable()
        except Exception:
            pass
    trainable = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(trainable, lr=lr)
    print(f"LoRA r={r_lora} layers {layers} | trainable {sum(p.numel() for p in trainable):,} | tasks {len(tasks)}", flush=True)
    print(f"base load: quant={qmode} dtype={dtype} device={dev} | K={K} max_new={max_new}", flush=True)
    if qmode == "none":
        print("WARN: V5_LM_QUANT is unset -> full-precision base load. For the 4B on rented GPUs, prefer V5_LM_QUANT=4bit.", flush=True)
    if reward_mode == "verifier":
        print("reward_mode=verifier -> exact SWE tests are inside the rollout reward; this is the apex path and will be slow.", flush=True)
    rng = random.Random(seed)
    n_held = max(4, len(tasks) // 5)
    held, train_tasks = tasks[:n_held], tasks[n_held:]

    def encode(system, user):
        m = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        kw = dict(add_generation_prompt=True, return_tensors="pt", return_dict=True)
        try:
            enc = tok.apply_chat_template(m, enable_thinking=False, **kw)
        except TypeError:
            enc = tok.apply_chat_template(m, **kw)
        return enc["input_ids"].to(dev)

    def gen_ids(pids, sample):
        with torch.no_grad():
            return model.generate(pids, do_sample=sample, temperature=temperature if sample else None,
                                  top_p=0.95 if sample else None, max_new_tokens=max_new,
                                  pad_token_id=tok.eos_token_id, use_cache=True)

    def seq_logprob(pids, comp):
        full = torch.cat([pids, comp], dim=1)
        logp = torch.log_softmax(model(full, use_cache=False).logits[:, :-1].float(), dim=-1)
        start = pids.shape[1] - 1
        span = logp[:, start:start + comp.shape[1]]
        sel = span.gather(-1, comp.unsqueeze(-1)).squeeze(-1).sum(-1)
        ent = -(span.exp() * span).sum(-1).mean()
        return sel, ent

    @torch.no_grad()
    def evaluate(ts):
        model.eval(); rs, app, solv = [], 0, 0
        for t in ts:
            pids = encode(SR_SYS, fix_user(t["issue"], t["src"]))
            out = gen_ids(pids, sample=False)
            r, b = score(tok.decode(out[0, pids.shape[1]:], skip_special_tokens=True), t,
                         reward_mode=reward_mode, verifier=verifier)
            rs.append(r); app += int(b.get("grounded", False)); solv += int(b.get("solved", False))
        model.train()
        return sum(rs) / len(ts), app / len(ts), solv / len(ts)

    @torch.no_grad()
    def evaluate_exact(ts, tag):
        if verifier is None:
            return None
        model.eval()
        task_patches: list[tuple[dict, str]] = []
        applyable = 0
        for t in ts:
            pids = encode(SR_SYS, fix_user(t["issue"], t["src"]))
            out = gen_ids(pids, sample=False)
            completion = tok.decode(out[0, pids.shape[1]:], skip_special_tokens=True)
            blocks = parse_sr(completion)
            applied, patch = materialize_patch(t, blocks)
            if applied > 0 and patch.strip():
                task_patches.append((t, patch))
                applyable += 1
        res = verifier.verify_task_batch_unique(task_patches, tag=tag)
        model.train()
        resolved = sum(1 for t in ts if res.get(t["iid"], False))
        return resolved / len(ts), applyable / len(ts)

    def run_gold_sanity(ts, n):
        if verifier is None or n <= 0:
            return
        gold = [(t, t["gold"]) for t in ts[:n] if (t.get("gold") or "").strip()]
        if not gold:
            raise SystemExit("exact verifier requested, but no gold patches available for sanity check")
        res = verifier.verify_task_batch_unique(gold, tag="gold_sanity")
        ok = sum(1 for task, _patch in gold if res.get(task["iid"], False))
        print(f"[gold-sanity] resolved {ok}/{len(gold)} gold patches", flush=True)
        if ok != len(gold):
            raise SystemExit("gold-sanity failed; refusing to trust exact SWE verifier results")

    @torch.no_grad()
    def emit_predictions(ts, tag):
        if not emit_preds_dir:
            return None
        model.eval()
        task_patches: list[tuple[dict, str]] = []
        for t in ts:
            pids = encode(SR_SYS, fix_user(t["issue"], t["src"]))
            out = gen_ids(pids, sample=False)
            completion = tok.decode(out[0, pids.shape[1]:], skip_special_tokens=True)
            blocks = parse_sr(completion)
            applied, patch = materialize_patch(t, blocks)
            if applied > 0 and patch.strip():
                task_patches.append((t, patch))
        out_dir = Path(emit_preds_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        pred_path = out_dir / f"{tag}.jsonl"
        n = write_predictions({task["iid"]: patch for task, patch in task_patches}, str(pred_path), model_name="swe_rl")
        model.train()
        print(f"[emit {tag}] wrote {n}/{len(ts)} held predictions -> {pred_path}", flush=True)
        return str(pred_path)

    if verifier is not None and (reward_mode == "verifier" or verify_every > 0):
        run_gold_sanity(held, min(verify_gold_sanity, len(held)))

    bm, ba, bs = evaluate(held)
    print(f"[eval @0] held mean_reward={bm:+.3f} applyable={ba:.0%} gold-solve={bs:.0%}", flush=True)
    if verifier is not None:
        exact0 = evaluate_exact(held, "verify_0")
        if exact0 is not None:
            vr, vp = exact0
            print(f"[verify @0] held exact_resolve={vr:.0%} patch_emission={vp:.0%}", flush=True)

    if sft_steps > 0:                                      # SFT warm-start on the gold patch (gold SR)
        ce = torch.nn.CrossEntropyLoss()
        for s in range(1, sft_steps + 1):
            t = rng.choice(train_tasks)
            pids = encode(SR_SYS, fix_user(t["issue"], t["src"]))
            gids = tok(t["gold_sr_text"], return_tensors="pt", add_special_tokens=False).input_ids.to(dev)[:, :max_new]
            full = torch.cat([pids, gids], dim=1)
            logits = model(full, use_cache=False).logits
            st = pids.shape[1]
            loss = ce(logits[:, st - 1:st - 1 + gids.shape[1]].reshape(-1, logits.shape[-1]).float(), gids.reshape(-1))
            loss.backward(); torch.nn.utils.clip_grad_norm_(trainable, 1.0); opt.step(); opt.zero_grad()
            if s <= 3 or s % 25 == 0:
                print(f"[sft {s:3}] {t['iid']:26} ce_loss={float(loss.detach()):.3f}", flush=True)
        sm, sa, ss = evaluate(held)
        print(f"[eval after SFT] held mean_reward={sm:+.3f} applyable={sa:.0%} gold-solve={ss:.0%} (base {bm:+.3f}/{ba:.0%}/{bs:.0%})\n", flush=True)
        if verifier is not None:
            exact_sft = evaluate_exact(held, "verify_after_sft")
            if exact_sft is not None:
                vr, vp = exact_sft
                print(f"[verify after SFT] held exact_resolve={vr:.0%} patch_emission={vp:.0%}\n", flush=True)
        if emit_preds_dir:
            emit_predictions(held, "held_after_sft")

    zero_var = 0
    for step in range(1, steps + 1):
        t = rng.choice(train_tasks)
        pids = encode(SR_SYS, fix_user(t["issue"], t["src"]))
        comps, rewards = [], []
        for _ in range(K):
            out = gen_ids(pids, sample=True)
            comp = out[:, pids.shape[1]:]
            completion = tok.decode(comp[0], skip_special_tokens=True)
            comps.append(comp)
            rewards.append(score(completion, t, reward_mode=reward_mode, verifier=verifier)[0])
        mean_r = sum(rewards) / K
        r_std = (sum((r - mean_r) ** 2 for r in rewards) / K) ** 0.5
        if r_std < 1e-9:
            zero_var += 1
            if step % 10 == 0:
                print(f"[step {step:3}] {t['iid']:24} mean_r={mean_r:+.2f} r_std=0 SKIP", flush=True)
            continue
        advs = advantages(rewards)
        opt.zero_grad(set_to_none=True)
        loss_sum = 0.0
        ent_sum = 0.0
        for comp, adv in zip(comps, advs):
            lp, ent = seq_logprob(pids, comp)
            sample_loss = (-adv * lp - ent_coef * ent) / K
            loss_sum += float(sample_loss.detach())
            ent_sum += float(ent.detach())
            sample_loss.backward()
        gnorm = torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        opt.step()
        opt.zero_grad(set_to_none=True)
        if step % 10 == 0:
            print(f"[step {step:3}] {t['iid']:24} mean_r={mean_r:+.2f} r_std={r_std:.2f} ent={ent_sum/K:.2f} loss={loss_sum:+.3f} gnorm={float(gnorm):.3f} rewards={[round(r,1) for r in rewards]}", flush=True)
        if step % eval_every == 0:
            m, ap, sv = evaluate(held)
            print(f"[eval @{step}] held mean_reward={m:+.3f} applyable={ap:.0%} gold-solve={sv:.0%}", flush=True)
        if verifier is not None and verify_every > 0 and step % verify_every == 0:
            exact = evaluate_exact(held, f"verify_{step}")
            if exact is not None:
                vr, vp = exact
                print(f"[verify @{step}] held exact_resolve={vr:.0%} patch_emission={vp:.0%}", flush=True)
        if emit_preds_dir and emit_preds_every > 0 and step % emit_preds_every == 0:
            emit_predictions(held, f"held_step_{step:04d}")

    m, ap, sv = evaluate(held)
    print(f"\n=== SWE-RL DONE === held mean_reward {bm:+.3f}->{m:+.3f} | applyable {ba:.0%}->{ap:.0%} | gold-solve {bs:.0%}->{sv:.0%}")
    if verifier is not None:
        exact_final = evaluate_exact(held, "verify_final")
        if exact_final is not None:
            vr, vp = exact_final
            print(f"  final exact_resolve={vr:.0%} | patch_emission={vp:.0%}")
    else:
        print("  exact SWE resolve not run here (no verifier configured).")
    if emit_preds_dir:
        emit_predictions(held, "held_final")
    print(f"  zero-variance groups {zero_var}/{steps}.")
    model.save_pretrained("artifacts/swe_lora"); print("  LoRA saved -> artifacts/swe_lora")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--model", default="Qwen/Qwen3.5-4B")
    ap.add_argument("--traces", default="data/swe/grounded_traces.jsonl")
    ap.add_argument("--nodes", default="artifacts/graph_growth/swe_code_candidates.jsonl")
    ap.add_argument("--dataset", default="lite")
    ap.add_argument("--repo-root", default="data/swe_repos")
    ap.add_argument("--n-tasks", type=int, default=40)
    ap.add_argument("--src-bodies", type=int, default=4)
    ap.add_argument("--src-lines", type=int, default=70)
    ap.add_argument("--steps", type=int, default=120)
    ap.add_argument("--k", type=int, default=6)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--r-lora", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--layers", type=int, nargs="+", default=[20, 22, 24, 26, 28])
    ap.add_argument("--eval-every", type=int, default=30)
    ap.add_argument("--ent-coef", type=float, default=0.005)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--sft-steps", type=int, default=120)
    ap.add_argument("--max-new", type=int, default=320)
    ap.add_argument("--reward-mode", choices=["proxy", "verifier"], default="proxy",
                    help="proxy = gold-overlap reward; verifier = exact SWE pass/fail reward")
    ap.add_argument("--verify-every", type=int, default=0,
                    help="run exact held-out SWE verification every N GRPO steps (0 = off)")
    ap.add_argument("--verify-backend", choices=["docker", "sbcli"], default="docker")
    ap.add_argument("--verify-out-dir", default="artifacts/graph_growth/swe_verify")
    ap.add_argument("--verify-max-workers", type=int, default=4)
    ap.add_argument("--verify-timeout", type=int, default=1800)
    ap.add_argument("--verify-poll-secs", type=int, default=20)
    ap.add_argument("--verify-gold-sanity", type=int, default=5,
                    help="when exact verify is active, require this many gold patches to resolve first")
    ap.add_argument("--emit-preds-dir", default="",
                    help="optional dir to write held-out prediction jsonl files for a separate swe_verify box")
    ap.add_argument("--emit-preds-every", type=int, default=0,
                    help="emit held prediction jsonl every N GRPO steps (0 = off); final emit happens if dir is set")
    a = ap.parse_args()
    if a.selftest:
        import sys
        sys.exit(0 if _selftest() else 1)
    tasks = load_swe_tasks(a.n_tasks, a.traces, a.nodes, a.dataset, a.repo_root, a.src_bodies, a.src_lines)
    print(f"loaded {len(tasks)} SWE tasks", flush=True)
    if len(tasks) < 8:
        raise SystemExit("too few SWE tasks loaded (need traces + nodes + checkouts)")
    need_verifier = (a.reward_mode == "verifier") or (a.verify_every > 0)
    verifier = SWEExactVerifier(a.dataset, "test", a.verify_backend, a.verify_out_dir,
                                max_workers=a.verify_max_workers, timeout=a.verify_timeout,
                                poll_secs=a.verify_poll_secs) if need_verifier else None
    train(a.model, tasks, a.steps, a.k, a.lr, a.r_lora, a.seed, a.layers, a.eval_every,
          a.ent_coef, a.temperature, a.sft_steps, a.max_new,
          reward_mode=a.reward_mode, verifier=verifier,
          verify_every=a.verify_every, verify_gold_sanity=a.verify_gold_sanity,
          emit_preds_dir=a.emit_preds_dir, emit_preds_every=a.emit_preds_every)


if __name__ == "__main__":
    main()
