"""LGGN v2 M1 — the REALIZER: (reasoning trace + old span) -> new span. Raw completion, no chat
template, no instruction text (the whole prompt is data: the span and the trace, joined by two
fixed separators). Trained on REAL (thinking -> str_replace) pairs from Fable-5 sessions.

Why: lggn_decode proved z (2048d -> 64d bottleneck) cannot carry literal code content (ceiling
recall 0.12 with GOLD fix reprs). Split the job: the SPAN carries identifiers, the TRACE carries
what-to-do, z (M2 tracer) only ever has to reach the trace. M1 is the falsifiable half: do real
reasoning traces steer realization at all?

Arms (fresh LoRA each): trace (old+trace -> new) vs notrace (old -> new), plus an EVAL-TIME
shuffled-trace control (derangement of held traces through the trace model).
Gates: G1a trace added_recall >= 0.15 | G1b trace-notrace >= +0.05 every seed | G1c true-shuffled
>= +0.05 and shuffled ~ notrace. NO-GO on G1b/G1c falsifies the architecture premise -> stop.

  data    : python -m v5.training.ingest_fable5 --ingest-triples
  stats   : python -m v5.runtime.lggn_realizer --stats
  smoke   : python -m v5.runtime.lggn_realizer --smoke            # tiny model, real loop, local
  full    : V5_LM_QUANT=4bit python -m v5.runtime.lggn_realizer --seeds 3 --save-ckpt   (molab)
  wiring  : python -m v5.runtime.lggn_realizer --selftest         # no GPU, no model
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import time
from pathlib import Path

from v5.runtime.solution_ladder import _norm_lines

TRIPLES = "data/fable5/realizer_triples.jsonl"
CAVEAT = "<local-command-caveat>"

# ── format: the ONLY scaffolding the LM ever sees (~5 tokens, zero instructions) ──
SEP_T = "\n###T\n"          # trace follows
SEP_N = "\n###N\n"          # new span follows


def realizer_prompt(old: str, trace: str | None) -> str:
    """trace=None -> the notrace arm. Everything up to the final separator is data."""
    if trace is None:
        return old + SEP_N
    return old + SEP_T + trace + SEP_N


def tracer_prompt(old: str) -> str:
    """M2: strict prefix of realizer_prompt(old, trace) — leaves a future single-pass merge open."""
    return old + SEP_T


# ── data ────────────────────────────────────────────────────────────────────────

def load_triples(path: str = TRIPLES, trace_chars: int = 400, max_old_chars: int = 1800,
                 max_new_chars: int = 1800, fresh_only: bool = True, limit: int = 0,
                 stats: bool = False, log=print) -> list[dict]:
    """realizer_triples.jsonl -> filtered [{goal, trace, old, new, session_id}].
    Caps are FILTERS, not truncation — truncating a supervised pair corrupts it. The one
    transform: trace = intent TAIL (the text just before the tool call is the concrete plan;
    earlier text is exploration), sized to the M2 tracer's generation budget."""
    p = Path(path)
    if not p.exists():
        log(f"  [triples] {path} missing -> building via ingest_fable5 (network)...")
        from v5.training.ingest_fable5 import ingest_triples
        ingest_triples(0, path)
    rows = []
    with open(p, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    funnel = [("raw", len(rows))]

    rows = [r for r in rows if (r.get("goal") or "").strip()
            and not r["goal"].lstrip().startswith(CAVEAT)]
    funnel.append(("goal_ok", len(rows)))

    rows = [r for r in rows if len((r.get("intent") or "").strip()) >= 10]
    funnel.append(("intent_ok", len(rows)))

    rows = [r for r in rows if (r.get("old") or "").strip() and (r.get("new") or "").strip()
            and r["old"].strip() != r["new"].strip()]
    funnel.append(("edit_ok", len(rows)))

    n_fresh = sum(1 for r in rows if r.get("fresh"))
    if fresh_only:
        rows = [r for r in rows if r.get("fresh")]
    funnel.append((f"fresh_only={fresh_only}", len(rows)))

    seen, dedup = set(), []
    for r in rows:
        key = (r["old"].strip(), r["new"].strip())
        if key not in seen:
            seen.add(key)
            dedup.append(r)
    rows = dedup
    funnel.append(("dedup", len(rows)))

    rows = [r for r in rows if len(r["old"]) <= max_old_chars and len(r["new"]) <= max_new_chars]
    funnel.append((f"caps<=({max_old_chars},{max_new_chars})", len(rows)))

    def _trace(intent: str) -> str:
        # collapse whitespace runs BEFORE the tail-slice so the char budget buys content, not
        # indentation (a tail landing inside a code block otherwise yields a near-blank trace)
        return re.sub(r"[ \t]{4,}", " ", intent.strip())[-trace_chars:]

    out = [{"goal": r["goal"], "trace": _trace(r["intent"]),
            "old": r["old"], "new": r["new"], "session_id": r.get("session_id", "")}
           for r in rows]
    out = [t for t in out if len(t["trace"].strip()) >= 10]
    funnel.append(("trace_ok", len(out)))
    if limit:
        out = out[:limit]
    if stats:
        log("  [triples] filter funnel:")
        for name, n in funnel:
            log(f"    {name:24} {n}")
        log(f"    (fresh available before fresh_only: {n_fresh})")
        if out:
            import numpy as np
            for field in ("trace", "old", "new"):
                lens = np.array([len(t[field]) for t in out])
                log(f"    len({field:5}) p10/p50/p90/max = "
                    f"{np.percentile(lens, 10):.0f}/{np.percentile(lens, 50):.0f}/"
                    f"{np.percentile(lens, 90):.0f}/{lens.max()}")
            n_goals = len({_goal_key(t['goal']) for t in out})
            log(f"    unique goals: {n_goals} ({len(out)/max(1, n_goals):.1f} triples/goal)")
    return out


def _goal_key(goal: str) -> str:
    """Stable goal identity for split grouping. NOT builtin hash() — that is salted per process."""
    return hashlib.md5((goal or "")[:400].encode("utf-8", "ignore")).hexdigest()


def split_by_goal(triples: list[dict], seed: int, held_frac: float = 0.2) -> tuple[list[int], list[int]]:
    """Goal-level split (stricter than session: the same goal recurs across sessions, and its
    edits are near-duplicates). Whole goal-groups go to held until >= held_frac of triples."""
    import numpy as np
    groups: dict[str, list[int]] = {}
    for i, t in enumerate(triples):
        groups.setdefault(_goal_key(t["goal"]), []).append(i)
    keys = sorted(groups)
    order = np.random.RandomState(seed).permutation(len(keys))
    target = held_frac * len(triples)
    he, n_he = [], 0
    for k in order:
        if n_he >= target:
            break
        he.append(keys[k])
        n_he += len(groups[keys[k]])
    he_set = set(he)
    tr_idx = [i for k in keys if k not in he_set for i in groups[k]]
    he_idx = [i for k in keys if k in he_set for i in groups[k]]
    assert not (set(tr_idx) & set(he_idx)), "split leak"
    return tr_idx, he_idx


# ── masking (no chat template) ──────────────────────────────────────────────────

def _encode_pair(tok, prompt: str, target: str, max_tokens: int = 1024):
    """Tokenize prompt and target SEPARATELY and concatenate: the mask boundary is exact and
    matches inference (where the prompt is tokenized alone). Single-pass tokenization can merge
    tokens across the boundary. EOS is appended AND supervised so generation learns to stop.
    Returns (input_ids, labels) as plain lists, or None if over budget."""
    p_ids = tok(prompt, add_special_tokens=False)["input_ids"]
    t_ids = tok(target, add_special_tokens=False)["input_ids"] + [tok.eos_token_id]
    if len(p_ids) + len(t_ids) > max_tokens:
        return None
    return p_ids + t_ids, [-100] * len(p_ids) + t_ids


def _pad_batch(encs: list[tuple[list[int], list[int]]], pad_id: int):
    """Right-pad a batch of (input_ids, labels) to the batch max. Pads get label -100 and
    mask 0. Mask comes from LENGTHS, never from `ids != pad` — the supervised EOS equals pad."""
    width = max(len(ids) for ids, _ in encs)
    ids = [seq + [pad_id] * (width - len(seq)) for seq, _ in encs]
    lbl = [lab + [-100] * (width - len(lab)) for _, lab in encs]
    mask = [[1] * len(seq) + [0] * (width - len(seq)) for seq, _ in encs]
    return ids, lbl, mask


# ── metrics ─────────────────────────────────────────────────────────────────────

def added_recall(gen: str, old: str, gold_new: str) -> float | None:
    """Recall over gold lines ABSENT from the input span. Copy-the-input scores exactly 0
    (plain _fidelity rewards copying because old/new overlap heavily). None -> the edit adds
    no new lines (pure rearrange/delete) — skip the instance."""
    old_set = set(_norm_lines(old))
    added = [ln for ln in _norm_lines(gold_new) if ln not in old_set]
    if not added:
        return None
    gen_set = set(_norm_lines(gen))
    return sum(1 for ln in added if ln in gen_set) / len(added)


def is_copy(gen: str, old: str) -> bool:
    return set(_norm_lines(gen)) == set(_norm_lines(old))


def _derange(n: int, rng: random.Random) -> list[int]:
    """Permutation of range(n) with no fixed points (each instance gets a WRONG trace)."""
    idx = list(range(n))
    if n < 2:
        return idx
    for _ in range(200):
        rng.shuffle(idx)
        if all(i != j for i, j in enumerate(idx)):
            return idx
    return [(i + 1) % n for i in range(n)]


# ── the LM (shared by M1 realizer and M2 tracer) ────────────────────────────────

class RawLM:
    """_Decoder (lggn_decode) minus chat template minus FiLM: 4-bit-or-env base + LoRA(r=16,
    q/k/v/o), raw-completion encode, optional MoLoRA hooks for the M2 tracer."""

    def __init__(self, model_name: str, d_latent: int | None = None, use_molora: bool = False,
                 n_experts: int = 4, expert_r: int = 8, bottleneck: int = 64, lr: float = 2e-4):
        import os
        import torch
        from transformers import AutoTokenizer
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
        from v5.lm_loader import load_frozen_lm, resolve_quant

        self.torch = torch
        trust = os.environ.get("V5_LM_TRUST_REMOTE_CODE", "0").lower() in ("1", "true", "yes")
        self.tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=trust)
        base = load_frozen_lm(model_name)
        if resolve_quant(None) in ("4bit", "8bit"):
            base = prepare_model_for_kbit_training(base)
        else:
            # non-quant path (local dev): still need activation checkpointing for batched
            # training, and input grads so LoRA gets gradients through the frozen embed
            base.gradient_checkpointing_enable()
            base.enable_input_require_grads()
        lcfg = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, bias="none",
                          task_type="CAUSAL_LM",
                          target_modules=["q_proj", "k_proj", "v_proj", "o_proj"])
        self.model = get_peft_model(base, lcfg)
        self.dev = next(self.model.parameters()).device
        self.d_model = self.model.get_input_embeddings().weight.shape[1]
        self.use_molora = use_molora

        self.molora = None
        self._z_raw = None
        self._hooks = []
        if use_molora:
            from v5.runtime.lggn_decode import _MoLoRAModule, _find_layers
            assert d_latent, "use_molora needs d_latent"
            layers = _find_layers(self.model)
            cdt = next(p for p in self.model.parameters() if p.is_floating_point()).dtype
            self.molora = _MoLoRAModule(self.d_model, d_latent, len(layers),
                                        n_experts, expert_r, bottleneck).to(self.dev, cdt)
            self._install_molora_hooks(layers)

        lora_params = [p for p in self.model.parameters() if p.requires_grad]
        param_groups = [{"params": lora_params, "lr": lr}]
        if self.molora is not None:
            param_groups.append({"params": list(self.molora.parameters()), "lr": lr * 0.5})
        self.opt = torch.optim.AdamW(param_groups)
        self.max_grad_norm = 1.0

    def _install_molora_hooks(self, layers):
        for l, layer in enumerate(layers):
            def hook(module, input, output, layer_idx=l):
                if self._z_raw is None:
                    return output
                b = self.molora.encode(self._z_raw)
                h = self.molora.modulate(output[0], b, layer_idx)
                if isinstance(output, tuple):
                    return (h,) + output[1:]
                return h
            self._hooks.append(layer.register_forward_hook(hook))

    def _set_z(self, v):
        if v is None or self.molora is None:
            self._z_raw = None
            return
        cdt = next(self.molora.parameters()).dtype
        self._z_raw = self.torch.as_tensor(v, dtype=cdt, device=self.dev).unsqueeze(0)

    def _set_z_batch(self, zs: list):
        """Per-row z for a batch; None rows become zero vectors (NOTE: in the batched path a
        dropped z is a ZEROS vector through the encoder, not the hook bypass of _set_z(None) —
        same 'no signal' intent, slightly different mechanics)."""
        if self.molora is None or all(z is None for z in zs):
            self._z_raw = None
            return
        torch = self.torch
        cdt = next(self.molora.parameters()).dtype
        d = next(self.molora.behavior_encoder.parameters()).shape[1]
        rows = [torch.zeros(d) if z is None else torch.as_tensor(z, dtype=torch.float32)
                for z in zs]
        self._z_raw = torch.stack(rows).to(self.dev, cdt)

    def train_on(self, pairs: list[tuple[str, str]], latents=None, epochs: int = 2,
                 warmup: int = 1, z_dropout: float = 0.0, max_tokens: int = 1024,
                 batch_size: int = 8, log=print):
        """pairs: [(prompt, target)] already formatted; latents: optional per-pair z (M2).
        Batched: pairs are packed batch_size at a time (right pad, labels -100 on pads)."""
        torch = self.torch
        _drop_rng = random.Random(42)
        self.model.train()
        if self.molora is not None:
            self.molora.train()
        pad = self.tok.eos_token_id
        encs = []                                          # [(pair_idx, (ids, labels))]
        n_skip = 0
        for k, (prompt, target) in enumerate(pairs):
            e = _encode_pair(self.tok, prompt, target, max_tokens)
            if e is None:
                n_skip += 1
                continue
            encs.append((k, e))
        if n_skip:
            log(f"      [skipped {n_skip} over-budget pairs (> {max_tokens} tok)]")
        # length-sorted batches: uniform lengths -> minimal padding, bounded peak memory.
        # Batch ORDER is shuffled per epoch so sorting doesn't become a length curriculum.
        encs.sort(key=lambda ke: len(ke[1][0]))
        batches = [encs[i:i + batch_size] for i in range(0, len(encs), batch_size)]
        _order_rng = random.Random(43)
        for ep in range(epochs):
            _order_rng.shuffle(batches)
            if self.molora is not None:
                frozen = ep < warmup
                for p in self.molora.parameters():
                    p.requires_grad_(not frozen)
                if ep == warmup:
                    log(f"      [MoLoRA unfrozen at epoch {ep+1}]")
            tot, n_used, n_dropped = 0.0, 0, 0
            t0 = time.time()
            every = max(1, len(batches) // 8)
            for bi, batch in enumerate(batches):
                ids_l, lbl_l, mask_l = _pad_batch([e for _, e in batch], pad)
                ids = torch.tensor(ids_l, device=self.dev)
                lbl = torch.tensor(lbl_l, device=self.dev)
                mask = torch.tensor(mask_l, device=self.dev)
                if latents is not None:
                    zs = []
                    for k, _ in batch:
                        if z_dropout > 0 and _drop_rng.random() < z_dropout:
                            zs.append(None)
                            n_dropped += 1
                        else:
                            zs.append(latents[k])
                    self._set_z_batch(zs)
                else:
                    self._set_z(None)
                out = self.model(ids, attention_mask=mask, labels=lbl)
                self.opt.zero_grad()
                out.loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    [p for g in self.opt.param_groups for p in g["params"] if p.requires_grad],
                    self.max_grad_norm)
                self.opt.step()
                tot += float(out.loss.detach()) * len(batch)
                n_used += len(batch)
                if (bi + 1) % every == 0:
                    log(f"      ep {ep+1} batch {bi+1}/{len(batches)}: loss {tot/max(1, n_used):.3f} "
                        f"({time.time()-t0:.0f}s)")
            drop = f" (z_drop={n_dropped})" if n_dropped else ""
            log(f"      epoch {ep+1}/{epochs}: loss {tot/max(1, n_used):.3f} on {n_used} pairs{drop}")
        if self.molora is not None:
            for p in self.molora.parameters():
                p.requires_grad_(True)
        self._set_z(None)

    def generate_raw(self, prompt: str, z=None, max_new_tokens: int = 512) -> str:
        """Greedy completion. The generated text up to EOS IS the output — no parsing."""
        torch = self.torch
        self.model.eval()
        if self.molora is not None:
            self.molora.eval()
        self._set_z(z)
        with torch.no_grad():
            ids = torch.tensor([self.tok(prompt, add_special_tokens=False)["input_ids"]],
                               device=self.dev)
            out = self.model.generate(input_ids=ids, attention_mask=torch.ones_like(ids),
                                      max_new_tokens=max_new_tokens,
                                      min_new_tokens=2, do_sample=False,
                                      pad_token_id=self.tok.eos_token_id)
            raw = self.tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True)
        self._set_z(None)
        return raw

    def generate_raw_batch(self, prompts: list[str], zs=None, max_new_tokens: int = 512) -> list[str]:
        """Batched greedy completion. LEFT padding so every prompt ends at the same position;
        position ids follow the attention mask (Qwen2 honors it)."""
        torch = self.torch
        self.model.eval()
        if self.molora is not None:
            self.molora.eval()
        pad = self.tok.eos_token_id
        enc = [self.tok(p, add_special_tokens=False)["input_ids"] for p in prompts]
        width = max(len(e) for e in enc)
        ids = torch.tensor([[pad] * (width - len(e)) + e for e in enc], device=self.dev)
        mask = torch.tensor([[0] * (width - len(e)) + [1] * len(e) for e in enc], device=self.dev)
        if zs is not None:
            self._set_z_batch(zs)
        else:
            self._set_z(None)
        with torch.no_grad():
            out = self.model.generate(input_ids=ids, attention_mask=mask,
                                      max_new_tokens=max_new_tokens, min_new_tokens=2,
                                      do_sample=False, pad_token_id=pad)
        self._set_z(None)
        return [self.tok.decode(out[i, width:], skip_special_tokens=True)
                for i in range(len(prompts))]

    def save_checkpoint(self, path: str):
        Path(path).mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(str(Path(path) / "lora"))
        if self.molora is not None:
            self.torch.save(self.molora.state_dict(), str(Path(path) / "molora.pt"))

    @classmethod
    def load_checkpoint(cls, model_name: str, path: str, d_latent: int | None = None,
                        n_experts: int = 4, expert_r: int = 8, bottleneck: int = 64):
        from peft import PeftModel
        has_molora = (Path(path) / "molora.pt").exists()
        lm = cls(model_name, d_latent=d_latent, use_molora=has_molora,
                 n_experts=n_experts, expert_r=expert_r, bottleneck=bottleneck)
        lm.model = PeftModel.from_pretrained(lm.model.get_base_model(), str(Path(path) / "lora"))
        lm.model.eval()
        if has_molora and lm.molora is not None:
            from v5.runtime.lggn_decode import _find_layers
            lm.molora.load_state_dict(
                lm.torch.load(str(Path(path) / "molora.pt"), map_location=lm.dev, weights_only=True))
            lm.molora.eval()
            for h in lm._hooks:
                h.remove()
            lm._hooks.clear()
            lm._install_molora_hooks(_find_layers(lm.model))
        return lm

    def cleanup(self):
        import gc
        for h in self._hooks:
            h.remove()
        self._hooks.clear()
        del self.model
        if self.molora is not None:
            del self.molora
        del self.opt
        gc.collect()
        if self.torch.cuda.is_available():
            self.torch.cuda.empty_cache()


# ── M1 experiment ───────────────────────────────────────────────────────────────

def _eval_arm(lm: RawLM, triples, he_idx, traces_for, max_new: int, eval_batch: int = 8, log=print):
    """traces_for(j, i) -> trace or None for held position j / triple index i."""
    import numpy as np
    recs, copies, gens = [], 0, []
    t0 = time.time()
    for b0 in range(0, len(he_idx), eval_batch):
        chunk = he_idx[b0:b0 + eval_batch]
        prompts = [realizer_prompt(triples[i]["old"], traces_for(b0 + j, i))
                   for j, i in enumerate(chunk)]
        for g, i in zip(lm.generate_raw_batch(prompts, max_new_tokens=max_new), chunk):
            gens.append(g)
            r = added_recall(g, triples[i]["old"], triples[i]["new"])
            if r is not None:
                recs.append(r)
            copies += int(is_copy(g, triples[i]["old"]))
        done = b0 + len(chunk)
        log(f"      {done}/{len(he_idx)} eval'd, added_recall {np.mean(recs) if recs else 0.0:.3f} "
            f"({(time.time()-t0)/done:.1f}s/inst)")
    mean = float(np.mean(recs)) if recs else 0.0
    return mean, recs, copies / max(1, len(he_idx)), gens


def run_m1(model_name: str, triples: list[dict], seeds: list[int], arms: set[str],
           epochs: int, eval_n: int, held_frac: float, max_new: int, max_tokens: int,
           save_ckpt: bool, batch_size: int = 8, eval_batch: int = 8,
           out_dir: str = "artifacts/lggn_realizer", log=print):
    import numpy as np
    results: dict[str, dict[str, list]] = {}
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    for seed in seeds:
        tr_idx, he_idx = split_by_goal(triples, seed, held_frac)
        he_eval = he_idx[:eval_n]
        log(f"\n--- seed {seed}: train {len(tr_idx)} / held {len(he_idx)} (eval {len(he_eval)}) ---")
        random.Random(seed).shuffle(tr_idx)
        shuffle_map = _derange(len(he_eval), random.Random(1000 + seed))
        seed_res = results.setdefault(str(seed), {})

        for arm in ("trace", "notrace"):
            if arm not in arms:
                continue
            log(f"  [{arm}] training ({epochs} epochs, {len(tr_idx)} pairs)...")
            lm = RawLM(model_name)
            pairs = [(realizer_prompt(triples[i]["old"],
                                      triples[i]["trace"] if arm == "trace" else None),
                      triples[i]["new"]) for i in tr_idx]
            lm.train_on(pairs, epochs=epochs, max_tokens=max_tokens, batch_size=batch_size, log=log)

            mean, recs, cp, gens = _eval_arm(
                lm, triples, he_eval,
                (lambda j, i: triples[i]["trace"]) if arm == "trace" else (lambda j, i: None),
                max_new, eval_batch, log)
            log(f"  [{arm}] added_recall={mean:.3f} copy_rate={cp:.2f} (n={len(recs)})")
            seed_res[arm] = {"added_recall": mean, "n": len(recs), "copy_rate": cp}

            if arm == "trace":
                s_mean, s_recs, s_cp, s_gens = _eval_arm(
                    lm, triples, he_eval,
                    lambda j, i: triples[he_eval[shuffle_map[j]]]["trace"], max_new, eval_batch, log)
                log(f"  [trace@shuffled] added_recall={s_mean:.3f} copy_rate={s_cp:.2f}")
                seed_res["shuffled"] = {"added_recall": s_mean, "n": len(s_recs), "copy_rate": s_cp}
                if save_ckpt:
                    ck = f"{out_dir}/seed{seed}_trace"
                    lm.save_checkpoint(ck)
                    log(f"  [trace] checkpoint -> {ck}")

            with open(f"{out_dir.rstrip('/')}/samples_seed{seed}_{arm}.jsonl", "w",
                      encoding="utf-8") as w:
                for j, i in enumerate(he_eval[:40]):
                    t = triples[i]
                    w.write(json.dumps({
                        "idx": i, "trace": t["trace"][:400], "old": t["old"][:600],
                        "gold_new": t["new"][:600], "gen": gens[j][:600],
                        "added_recall": added_recall(gens[j], t["old"], t["new"]),
                    }, ensure_ascii=False) + "\n")
            lm.cleanup()

    _report_m1(results, log)
    with open("artifacts/lggn_realizer_results.json", "w", encoding="utf-8") as w:
        json.dump(results, w, indent=2)
    log("  results -> artifacts/lggn_realizer_results.json")
    return results


def _report_m1(results: dict, log=print):
    import numpy as np
    log("\n=== M1 REALIZER (trace+span -> new span; added-line recall) ===")
    arms = sorted({a for s in results.values() for a in s})
    per_arm = {a: [s[a]["added_recall"] for s in results.values() if a in s] for a in arms}
    for a in arms:
        v = per_arm[a]
        log(f"  {a:10}: added_recall = {np.mean(v):.3f} +/- {np.std(v):.3f}  ({len(v)} seeds)")

    def delta(a, b):
        d = [s[a]["added_recall"] - s[b]["added_recall"]
             for s in results.values() if a in s and b in s]
        return (float(np.mean(d)), float(np.std(d)), all(x > 0 for x in d)) if d else None

    log("")
    if "trace" in per_arm:
        g1a = float(np.mean(per_arm["trace"])) >= 0.15
        log(f"  G1a trace >= 0.15          : {np.mean(per_arm['trace']):.3f}  -> {'PASS' if g1a else 'FAIL'}")
    d = delta("trace", "notrace")
    if d:
        log(f"  G1b trace - notrace        : {d[0]:+.3f} +/- {d[1]:.3f} all_pos={d[2]}  "
            f"-> {'PASS' if d[0] >= 0.05 and d[2] else 'FAIL'}")
    d = delta("trace", "shuffled")
    if d:
        log(f"  G1c true - shuffled        : {d[0]:+.3f} +/- {d[1]:.3f}  "
            f"-> {'PASS' if d[0] >= 0.05 else 'FAIL'}")
    d = delta("shuffled", "notrace")
    if d:
        log(f"      shuffled - notrace     : {d[0]:+.3f} (should be ~0, +/-0.03)")


# ── selftest (no GPU, no model) ─────────────────────────────────────────────────

class _FakeTok:
    """Deterministic whitespace tokenizer: one id per unique word."""
    eos_token_id = 0

    def __init__(self):
        self.vocab = {}

    def __call__(self, text, add_special_tokens=False):
        ids = [self.vocab.setdefault(w, len(self.vocab) + 1) for w in text.split()]
        return {"input_ids": ids}


def _selftest() -> bool:
    print("lggn_realizer --selftest: data funnel, masking, metrics, split, format (no model)\n")

    # 1. filter funnel on synthetic rows
    import tempfile
    rows = [
        {"goal": "Add retry", "intent": "wrap get in retry decorator", "old": "a = 1", "new": "a = 2", "fresh": True, "session_id": "s1"},
        {"goal": CAVEAT + " noise", "intent": "long enough intent", "old": "b = 1", "new": "b = 2", "fresh": True},      # caveat -> drop
        {"goal": "Add retry", "intent": "x", "old": "c = 1", "new": "c = 2", "fresh": True},                             # intent<10 -> drop
        {"goal": "Add retry", "intent": "identical old and new here", "old": "d = 1", "new": "d = 1", "fresh": True},    # no-op -> drop
        {"goal": "Add retry", "intent": "stale thinking block reused", "old": "e = 1", "new": "e = 2", "fresh": False},  # stale -> drop
        {"goal": "Add retry", "intent": "wrap get in retry decorator", "old": "a = 1", "new": "a = 2", "fresh": True},   # dup -> drop
        {"goal": "Add retry", "intent": "way too big an edit here", "old": "f" * 3000, "new": "f = 2", "fresh": True},   # cap -> drop
        {"goal": "Fix timeout", "intent": "bump the timeout to 30s", "old": "T = 5", "new": "T = 30", "fresh": True},
    ]
    with tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False, encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
        tmp = f.name
    trips = load_triples(tmp, stats=True)
    assert len(trips) == 2, f"funnel: expected 2 survivors, got {len(trips)}"
    assert trips[0]["old"] == "a = 1" and trips[1]["new"] == "T = 30"
    print("  [1] filter funnel -> PASS")

    # 2. masking with fake tokenizer
    tok = _FakeTok()
    prompt, target = "old span here" + SEP_T + "the trace" + SEP_N, "new span here"
    enc = _encode_pair(tok, prompt, target)
    assert enc is not None
    ids, labels = enc
    p_len = len(tok(prompt)["input_ids"])
    assert labels[:p_len] == [-100] * p_len, "prompt fully masked"
    assert labels[p_len:] == ids[p_len:], "target supervised"
    assert ids[-1] == tok.eos_token_id and labels[-1] == tok.eos_token_id, "EOS appended + supervised"
    assert _encode_pair(tok, "w " * 2000, "x", max_tokens=64) is None, "over-budget -> None"
    e1 = _encode_pair(tok, "a b c", "d e")
    e2 = _encode_pair(tok, "a b c d e f", "g")
    ids, lbl, mask = _pad_batch([e1, e2], pad_id=tok.eos_token_id)
    w = len(e2[0])
    assert all(len(r) == w for r in ids + lbl + mask), "padded to batch max"
    assert ids[0][len(e1[0]):] == [tok.eos_token_id] * (w - len(e1[0])), "pad id fills"
    assert lbl[0][len(e1[0]):] == [-100] * (w - len(e1[0])), "pads not supervised"
    assert mask[0] == [1] * len(e1[0]) + [0] * (w - len(e1[0])), "mask from lengths"
    assert mask[1] == [1] * w and lbl[1][-1] == tok.eos_token_id, "full row: no pads, EOS supervised"
    print("  [2] masking + batch padding -> PASS")

    # 3. metrics
    old, new = "keep_a\nkeep_b", "keep_a\nkeep_b\nadded_line_1\nadded_line_2"
    assert added_recall(old, old, new) == 0.0, "copying input scores 0"
    assert added_recall(new, old, new) == 1.0, "exact gold scores 1"
    assert added_recall("added_line_1", old, new) == 0.5
    assert added_recall("anything", old, "keep_b\nkeep_a") is None, "no added lines -> None"
    assert is_copy(old + "\n", old) and not is_copy(new, old)
    print("  [3] added_recall -> PASS")

    # 4. goal-level split: no leakage, deterministic
    trips4 = [{"goal": f"g{k}", "trace": "t", "old": f"o{k}_{j}", "new": "n", "session_id": ""}
              for k in range(10) for j in range(3)]
    for seed in (0, 1, 2):
        tr, he = split_by_goal(trips4, seed)
        tr_goals = {trips4[i]["goal"] for i in tr}
        he_goals = {trips4[i]["goal"] for i in he}
        assert not (tr_goals & he_goals), "goal leaked across split"
        assert len(he) >= 0.2 * len(trips4)
        tr2, he2 = split_by_goal(trips4, seed)
        assert tr == tr2 and he == he2, "split not deterministic"
    print("  [4] goal split -> PASS")

    # 5. derangement + format prefix property
    for n in (2, 5, 30):
        d = _derange(n, random.Random(7))
        assert sorted(d) == list(range(n)) and all(i != j for i, j in enumerate(d))
    assert realizer_prompt("O", "T").startswith(tracer_prompt("O")), "tracer prompt must be a strict prefix"
    assert realizer_prompt("O", None) == "O" + SEP_N
    print("  [5] derangement + format -> PASS")

    print("\n  LGGN_REALIZER SELFTEST -> PASS")
    return True


# ── CLI ─────────────────────────────────────────────────────────────────────────

def main():
    import sys
    # molab pipes stdout -> Python block-buffers print() (8KB) while tqdm/warnings on stderr
    # stream through -> the run LOOKS hung for a whole epoch. Force line buffering.
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except Exception:
        pass
    ap = argparse.ArgumentParser(description="LGGN v2 M1 — realizer: (trace, old span) -> new span.")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--stats", action="store_true", help="print filter funnel + length percentiles, exit")
    ap.add_argument("--smoke", action="store_true",
                    help="tiny local run of the REAL loop: 0.5B model, 24 train / 8 eval, 1 seed")
    ap.add_argument("--model", default="Qwen/Qwen2.5-3B")
    ap.add_argument("--triples", default=TRIPLES)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--arms", default="trace,notrace")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--trace-chars", type=int, default=400)
    ap.add_argument("--eval-n", type=int, default=200)
    ap.add_argument("--held-frac", type=float, default=0.2)
    ap.add_argument("--max-new", type=int, default=512)
    ap.add_argument("--max-tokens", type=int, default=1024, help="train pair budget; longer pairs skipped")
    ap.add_argument("--batch-size", type=int, default=8, help="train batch (A40: 8-16; local 6GB: 2-4)")
    ap.add_argument("--eval-batch", type=int, default=8, help="generation batch at eval")
    ap.add_argument("--limit", type=int, default=0, help="cap triples (smoke/debug)")
    ap.add_argument("--no-fresh-only", action="store_true", help="keep stale-intent rows too")
    ap.add_argument("--save-ckpt", action="store_true", help="save trace-arm checkpoints (M2 needs them)")
    a = ap.parse_args()

    if a.selftest:
        raise SystemExit(0 if _selftest() else 1)

    triples = load_triples(a.triples, trace_chars=a.trace_chars, fresh_only=not a.no_fresh_only,
                           limit=a.limit, stats=True)
    print(f"  {len(triples)} usable triples")
    if a.stats:
        return
    if a.smoke:
        a.model = "Qwen/Qwen2.5-0.5B" if a.model == "Qwen/Qwen2.5-3B" else a.model
        triples = triples[:32]
        print(f"[SMOKE] model={a.model} n={len(triples)} 1 seed, 1 epoch, eval 8")
        run_m1(a.model, triples, [0], {"trace", "notrace"}, epochs=1, eval_n=8,
               held_frac=0.25, max_new=min(a.max_new, 256), max_tokens=a.max_tokens,
               save_ckpt=False, batch_size=min(a.batch_size, 2), eval_batch=min(a.eval_batch, 2))
        return
    print(f"[lggn-realizer M1] model={a.model} seeds={a.seeds} arms={a.arms} epochs={a.epochs} "
          f"trace_chars={a.trace_chars} eval_n={a.eval_n} batch={a.batch_size}/{a.eval_batch}")
    run_m1(a.model, triples, list(range(a.seeds)), set(a.arms.split(",")), epochs=a.epochs,
           eval_n=a.eval_n, held_frac=a.held_frac, max_new=a.max_new, max_tokens=a.max_tokens,
           save_ckpt=a.save_ckpt, batch_size=a.batch_size, eval_batch=a.eval_batch)


if __name__ == "__main__":
    main()
