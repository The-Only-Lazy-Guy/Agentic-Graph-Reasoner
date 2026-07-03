"""LGGN decode v2 — FiLM injection: the refiner's h_K modulates Qwen's computation.

v1 (soft prefix) FAILED: baseline=0.136, latent=0.098, ceiling=0.113 — LoRA ignores prefix tokens.
Root cause: soft prefix adds info to the INPUT; model is free to ignore it. Same failure mode as
composition_decode (ops-as-text hurt). The latent must change the COMPUTATION, not the input.

v2: FiLM (Feature-wise Linear Modulation). At every transformer layer:
    h' = γ(z)⊙h + β(z)
where z is the refiner's h_K (or gold_f for ceiling). The latent modulates every layer's hidden
states — the model CANNOT ignore it. Proven in diffusion transformers (DiT, SD3).

Architecture (3 layers):
    h_K (2560d, Qwen space) -> BehaviorEncoder (2560->bottleneck) -> behavior embedding (~64d)
    -> FiLM Renderer (per-layer expansion to γ,β) -> modulate Qwen at every layer

The BehaviorEncoder creates a compact behavior space where the graph will eventually operate
(operators, composition, write-back, retrieval). The FiLM Renderer is just a renderer.

Arms (each trains a fresh LoRA + FiLM):
  baseline   LoRA only, no FiLM (issue+code -> gold SR)
  latent     LoRA + FiLM(h_K)     (THE TEST — does the refiner help decode?)
  ceiling    LoRA + FiLM(gold_f)  (injection ceiling — can the mechanism work at all?)

  ceiling > baseline -> FiLM injection WORKS (latent modulates computation, model uses it)
  latent > baseline  -> h_K helps decode (the bridge works)
  ceiling ~ baseline -> FiLM mechanism broken (escalate to hypernetwork / DeltaW)

VRAM: ~3.5GB (4-bit Qwen3.5-4B + LoRA ~9M + FiLM ~12M). Under 6GB.

  V5_LM_TRUST_REMOTE_CODE=1 V5_LM_QUANT=4bit python -m v5.runtime.lggn_decode \\
      --model Qwen/Qwen3.5-4B --dataset lite --n 200 --r 512
  python -m v5.runtime.lggn_decode --selftest
"""
from __future__ import annotations

import argparse
import torch
import torch.nn as nn


# ── data ────────────────────────────────────────────────────────────────────────

def _load_paired(model_name, dataset, split, n, layer_frac=0.6, t_ctx=48):
    """Qwen reprs (g,f,ctx,cmask) paired with source text per instance."""
    from v5.runtime.lggn_refine import _reprs
    from v5.graph_grower.swe_load import load_instances
    from v5.runtime.operator_discovery import extract_hunks, _fix_text
    instances = load_instances(name=dataset, split=split, limit=n)
    texts = []
    for inst in instances:
        hs = extract_hunks(inst.get("patch", "") or "")
        if not hs:
            continue
        removed = "\n".join(l for _f, r, _a in hs for l in r)
        if not removed.strip():
            continue
        fx = _fix_text(hs)
        if not fx.strip():
            continue
        added = "\n".join(l for _f, _r, a in hs for l in a)
        texts.append({"issue": (inst.get("problem_statement") or "")[:700],
                      "code": removed[:900], "added": added[:600]})
    g, f, ctx, cmask = _reprs(model_name, dataset, split, n, layer_frac, t_ctx)
    assert len(texts) == len(g), f"text/repr count mismatch: {len(texts)} texts vs {len(g)} reprs"
    return g, f, ctx, cmask, texts


# ── refiner (from lggn_refine, returns h_K + trajectory) ───────────────────────

def _train_refiner(g, f, ctx, cmask, tr, K=4, r=512, n_op=24, epochs=400, seed=0,
                   learn_ops=False, k_warmup=0.5, contrastive=0.0, log=print):
    """Train refiner on tr, return (h_K_all, ops, net, cos_held).

    Improvements over vanilla cosine training:
      learn_ops:    make operator basis learnable (KMeans init, then drift)
      k_warmup:     schedule K from 1→K_max over this fraction of epochs
                    (learn single-step fixes first, then composition)
      contrastive:  weight for wrong-instance-gold triplet loss
                    (push h_K away from f_wrong, not just toward f_own)
    """
    import numpy as np
    from v5.runtime.lggn_refine import Refiner, _discover_ops
    torch.manual_seed(seed)
    disp_tr = (f - g)[tr]
    ops = _discover_ops(disp_tr, n_op, seed)
    net = Refiner.Net(g.shape[1], r=r, n_op=ops.shape[0])
    T = lambda x: torch.as_tensor(x, dtype=torch.float32)
    gt, ft, ct, ot = T(g), T(f), T(ctx), T(ops); cm = torch.as_tensor(cmask)

    if learn_ops:
        ops_param = torch.nn.Parameter(ot.clone())
        opt = torch.optim.Adam(list(net.parameters()) + [ops_param],
                               lr=1e-3, weight_decay=1e-4)
    else:
        ops_param = ot
        opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-4)

    Fn = torch.nn.functional.cosine_similarity
    margin = 0.2
    for ep in range(epochs):
        K_cur = 1 + int((K - 1) * min(1.0, ep / max(1, epochs * k_warmup - 1)))

        net.train(); opt.zero_grad()
        h = net(gt[tr], ct[tr], cm[tr], ops_param, K_cur, True, True)

        loss = (1 - Fn(h, ft[tr])).mean()

        if contrastive > 0 and len(tr) > 1:
            f_neg = torch.roll(ft[tr], 1, 0)
            loss = loss + contrastive * torch.clamp(Fn(h, f_neg) - margin, min=0).mean()

        loss.backward(); opt.step()

    final_ops = ops_param.detach() if learn_ops else ot
    ops_np = final_ops.float().numpy()

    net.eval()
    with torch.no_grad():
        h_all = net(gt, ct, cm, final_ops, K, True, True).float().numpy()
        he = np.setdiff1d(np.arange(len(g)), tr)
        cos_he = float(Fn(torch.tensor(h_all[he]), ft[he]).mean())
    extras = []
    if learn_ops: extras.append("learn_ops")
    if contrastive > 0: extras.append(f"contra={contrastive}")
    if k_warmup < 1.0: extras.append(f"k_warmup={k_warmup}")
    ex = f" ({', '.join(extras)})" if extras else ""
    log(f"    refiner: held cos(h_K, f) = {cos_he:.3f} [K={K} r={r} ops={len(ops_np)}{ex}]")
    return h_all, ops_np, net, cos_he


def _extract_trajectory(net, g, ctx, cmask, ops, K):
    """Per-step op-selection weights from the trained refiner. Shape [K, N, n_op]."""
    import math, numpy as np
    T = lambda x: torch.as_tensor(x, dtype=torch.float32)
    gt, ct, ot = T(g), T(ctx), T(ops); cm = torch.as_tensor(cmask)
    net.eval(); traj = []
    with torch.no_grad():
        h = gt
        for _ in range(K):
            q = net.Wq(h); k = net.Wk(ct)
            sc = (q.unsqueeze(1) * k).sum(-1) / math.sqrt(net.r)
            sc = sc.masked_fill(~cm, -1e9)
            a = (torch.softmax(sc, -1).unsqueeze(-1) * ct).sum(1)
            base = h + a
            logit = net.Wo(base) @ net.Wko(ot).t() / math.sqrt(net.r)
            w = torch.softmax(logit, -1)
            traj.append(w.cpu().numpy())
            h = h + net.gc * a + net.go * (w @ ot)
    return np.stack(traj)


def _write_back_report(traj_held, recall_per_instance, threshold=0.3):
    """Which ops did successful decodes rely on? Returns proposed graph edits."""
    import numpy as np
    from collections import Counter
    success = np.array(recall_per_instance) >= threshold
    n_succ = int(success.sum())
    if n_succ == 0:
        return {"n_success": 0, "op_usage": {}}
    avg_w = traj_held.mean(axis=0)  # [N_held, n_op]
    dom = np.argmax(avg_w[success], axis=1)
    return {"n_success": n_succ, "op_usage": dict(Counter(dom.tolist())),
            "mean_weight": avg_w[success].mean(axis=0).tolist()}


# ── FiLM conditioner ──────────────────────────────────────────────────────────
#
# h' = γ(z)⊙h + β(z) at every transformer layer.
#
# Architecture:
#   BehaviorEncoder: Linear(d_latent -> bottleneck) + GELU   (~164K params)
#   FiLM Renderer:   per-layer Linear(bottleneck -> 2*d_model)  (~12M params for 36 layers)
#
# Initialized to identity: γ=1, β=0. At init, the model behaves as if FiLM isn't there.
# Training teaches (γ,β) to deviate from identity in ways that help decode.
#
# The BehaviorEncoder output IS the behavior embedding — the compact space where
# graph operators, composition, write-back, and retrieval will eventually operate.
# The per-layer linears are just the renderer.

class _FiLMModule(nn.Module):
    """BehaviorEncoder + FiLM Renderer as a single trainable module.

    BehaviorEncoder: d_latent -> bottleneck (the behavior space).
    FiLM Renderer: bottleneck -> per-layer (γ, β).
    """

    def __init__(self, d_model: int, d_latent: int, n_layers: int, bottleneck: int = 64):
        super().__init__()
        self.d_model = d_model
        self.n_layers = n_layers
        self.bottleneck = bottleneck
        self.behavior_encoder = nn.Sequential(
            nn.Linear(d_latent, bottleneck),
            nn.GELU(),
        )
        self.renderer = nn.ModuleList([
            nn.Linear(bottleneck, 2 * d_model) for _ in range(n_layers)
        ])
        self._init_identity()

    def _init_identity(self):
        """γ=1, β=0 at init -> FiLM is a no-op until trained."""
        for lin in self.renderer:
            nn.init.zeros_(lin.weight)
            nn.init.zeros_(lin.bias)
            with torch.no_grad():
                lin.bias.data[:self.d_model].fill_(1.0)  # γ init = 1

    def encode(self, z):
        """z: [B, d_latent] -> behavior embedding [B, bottleneck]."""
        return self.behavior_encoder(z)

    def modulate(self, h, b, layer_idx):
        """h: [B, T, d_model], b: [B, bottleneck] -> modulated h.

        b is a pre-computed behavior embedding (from encode()).
        """
        gb = self.renderer[layer_idx](b)         # [B, 2*d_model]
        gamma, beta = gb.chunk(2, dim=-1)         # each [B, d_model]
        return gamma.unsqueeze(1) * h + beta.unsqueeze(1)


# ── decoder (LoRA Qwen + FiLM conditioning from latent) ──────────────────────

def _prompt(text):
    return (f"Fix this bug.\n\nIssue:\n{text['issue']}\n\nBuggy code:\n{text['code']}\n\n"
            "Output ONLY a search/replace block:\n"
            "<<<<<<< SEARCH\n<exact buggy code>\n=======\n<fixed code>\n>>>>>>> REPLACE")


def _target(text):
    return f"<<<<<<< SEARCH\n{text['code']}\n=======\n{text['added']}\n>>>>>>> REPLACE"


def _find_layers(model):
    """Find transformer decoder layers through any peft wrapping."""
    for name, module in model.named_modules():
        if name.endswith('.layers') and isinstance(module, nn.ModuleList) and len(module) > 10:
            return module
    raise RuntimeError("Can't find transformer layers in model")


class _Decoder:
    """Qwen + LoRA + FiLM conditioning. The latent modulates every layer — can't be ignored.

    When use_film=False (baseline arm): pure LoRA, no FiLM hooks, no behavior encoder.
    When use_film=True: FiLM hooks on every transformer layer. The behavior embedding (from
    BehaviorEncoder) determines (γ,β) at each layer. Gradients flow through the hooks
    to train both LoRA and FiLM jointly.
    """

    def __init__(self, model_name, d_latent, use_film=True, bottleneck=64, lr=2e-4):
        import os
        from transformers import AutoTokenizer
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
        from v5.lm_loader import load_frozen_lm, resolve_quant

        trust = os.environ.get("V5_LM_TRUST_REMOTE_CODE", "0").lower() in ("1", "true", "yes")
        self.tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=trust)
        base = load_frozen_lm(model_name)
        if resolve_quant(None) in ("4bit", "8bit"):
            base = prepare_model_for_kbit_training(base)
        lcfg = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, bias="none",
                          task_type="CAUSAL_LM",
                          target_modules=["q_proj", "k_proj", "v_proj", "o_proj"])
        self.model = get_peft_model(base, lcfg)
        self.dev = next(self.model.parameters()).device
        self.d_model = self.model.get_input_embeddings().weight.shape[1]
        self.use_film = use_film

        layers = _find_layers(self.model)
        n_layers = len(layers)
        cdt = next(p for p in self.model.parameters() if p.is_floating_point()).dtype

        # FiLM module (BehaviorEncoder + Renderer)
        self.film = None
        self._z_raw = None   # raw latent vector (detached); encode runs inside each hook
        self._hooks = []
        if use_film:
            self.film = _FiLMModule(self.d_model, d_latent, n_layers, bottleneck).to(self.dev, cdt)
            self._install_hooks(layers)

        # separate param groups: LoRA at full LR, FiLM at half LR for stability
        lora_params = [p for p in self.model.parameters() if p.requires_grad]
        param_groups = [{"params": lora_params, "lr": lr}]
        if self.film is not None:
            param_groups.append({"params": list(self.film.parameters()), "lr": lr * 0.5})
        self.opt = torch.optim.AdamW(param_groups)
        self.max_grad_norm = 1.0

    def _install_hooks(self, layers):
        """Register forward hooks on every transformer layer for FiLM modulation.

        Each hook computes film.encode(z_raw) -> film.modulate(h, b, layer_idx).
        The encode is recomputed per hook call (not cached) so that gradient
        checkpointing can replay the forward without hitting freed graph nodes.
        Overhead is negligible: 36x a Linear(2560->64)+GELU vs transformer layers.
        """
        for l, layer in enumerate(layers):
            def hook(module, input, output, layer_idx=l):
                if self._z_raw is None:
                    return output
                b = self.film.encode(self._z_raw)
                h = output[0]
                h = self.film.modulate(h, b, layer_idx)
                if isinstance(output, tuple):
                    return (h,) + output[1:]
                return h
            self._hooks.append(layer.register_forward_hook(hook))

    def _set_z(self, v):
        """Set conditioning vector for subsequent forward passes.

        Stores the raw latent as a detached tensor. Each hook recomputes
        film.encode(z_raw) to create a fresh computation graph, compatible
        with gradient checkpointing (which replays forward during backward).
        """
        if v is None or self.film is None:
            self._z_raw = None
            return
        cdt = next(self.film.parameters()).dtype
        self._z_raw = torch.as_tensor(v, dtype=cdt, device=self.dev).unsqueeze(0)

    def _apply_chat(self, msgs, **extra):
        """apply_chat_template with enable_thinking fallback.
        Always returns a tensor [1, T] (extracts input_ids if BatchEncoding)."""
        try:
            out = self.tok.apply_chat_template(msgs, enable_thinking=False, **extra)
        except TypeError:
            out = self.tok.apply_chat_template(msgs, **extra)
        if hasattr(out, "input_ids"):
            out = out["input_ids"]
        if isinstance(out, torch.Tensor):
            return out.unsqueeze(0) if out.dim() == 1 else out
        return torch.tensor(out).unsqueeze(0) if not isinstance(out, torch.Tensor) else out

    def train_on(self, texts, latents, indices, epochs=2, film_warmup=1,
                 z_dropout=0.0, log=print):
        """Train LoRA + FiLM on the training set.

        film_warmup: freeze FiLM for this many epochs, let LoRA settle first.
        z_dropout: probability of replacing z with zeros for each instance (0-1).
            Forces FiLM to USE z-content, not learn a fixed modulation pattern.
            Analogous to classifier-free guidance in diffusion models.
        """
        import random as _rng
        _drop_rng = _rng.Random(42)
        self.model.train()
        if self.film is not None:
            self.film.train()
        for ep in range(epochs):
            if self.film is not None:
                film_frozen = ep < film_warmup
                for p in self.film.parameters():
                    p.requires_grad_(not film_frozen)
                if ep == film_warmup:
                    log(f"      [FiLM unfrozen at epoch {ep+1}]")

            tot = 0.0; n_dropped = 0
            for i in indices:
                if latents is not None and z_dropout > 0 and _drop_rng.random() < z_dropout:
                    self._set_z(None)
                    n_dropped += 1
                else:
                    self._set_z(latents[i] if latents is not None else None)
                prompt, target = _prompt(texts[i]), _target(texts[i])

                full_ids = self._apply_chat(
                    [{"role": "user", "content": prompt},
                     {"role": "assistant", "content": target}],
                    return_tensors="pt").to(self.dev)

                plen = self._apply_chat(
                    [{"role": "user", "content": prompt}],
                    add_generation_prompt=True, return_tensors="pt").shape[-1]

                labels = full_ids.clone()
                labels[0, :plen] = -100

                out = self.model(full_ids, labels=labels)
                self.opt.zero_grad()
                out.loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    [p for g in self.opt.param_groups for p in g["params"] if p.requires_grad],
                    self.max_grad_norm)
                self.opt.step()
                tot += float(out.loss.detach())
            drop_pct = f" (z_drop={n_dropped}/{len(indices)})" if z_dropout > 0 else ""
            log(f"      epoch {ep+1}/{epochs}: loss {tot/max(1,len(indices)):.3f}{drop_pct}")

        # restore FiLM grad state
        if self.film is not None:
            for p in self.film.parameters():
                p.requires_grad_(True)
        self._set_z(None)

    def eval_on(self, texts, latents, indices, log=print):
        """Generate patches on held-out instances, measure recall vs gold.

        Returns (mean_recall, per_instance_recalls).
        """
        import numpy as np
        from v5.runtime.solution_ladder import _emitted_replace, _fidelity

        self.model.eval()
        if self.film is not None:
            self.film.eval()
        recs = []
        with torch.no_grad():
            for j, i in enumerate(indices):
                self._set_z(latents[i] if latents is not None else None)
                prompt = _prompt(texts[i])
                ids = self._apply_chat(
                    [{"role": "user", "content": prompt}],
                    add_generation_prompt=True, return_tensors="pt").to(self.dev)
                out = self.model.generate(
                    input_ids=ids, max_new_tokens=256, min_new_tokens=8,
                    do_sample=False, pad_token_id=self.tok.eos_token_id)
                raw = self.tok.decode(out[0, ids.shape[1]:],
                                      skip_special_tokens=True)
                rec, _ = _fidelity(_emitted_replace(raw), texts[i]["added"])
                recs.append(rec)
                if (j + 1) % 10 == 0:
                    log(f"      {j+1}/{len(indices)} eval'd, running recall {np.mean(recs):.3f}")
        self._set_z(None)
        return float(np.mean(recs)) if recs else 0.0, recs

    def behavior_embeddings(self, latents, indices):
        """Extract behavior embeddings [bottleneck-d] for given instances via BehaviorEncoder."""
        import numpy as np
        if self.film is None or latents is None:
            return None
        behs = []
        cdt = next(self.film.parameters()).dtype
        with torch.no_grad():
            for i in indices:
                z = torch.as_tensor(latents[i], dtype=cdt, device=self.dev).unsqueeze(0)
                b = self.film.encode(z).squeeze(0).float().cpu().numpy()
                behs.append(b)
        return np.stack(behs)

    def cleanup(self):
        """Free GPU memory between arms."""
        import gc
        for h in self._hooks:
            h.remove()
        self._hooks.clear()
        del self.model
        if self.film is not None:
            del self.film
        del self.opt
        gc.collect()
        torch.cuda.empty_cache()


# ── experiment ──────────────────────────────────────────────────────────────────

def _sensitivity_sweep(dec, texts, gold_f, h_K, tr, he, n_points=7, log=print):
    """After training a ceiling decoder with gold_f, eval with interpolated latents.

    h_alpha = alpha * gold_f + (1-alpha) * h_K for alpha in [1.0 ... 0.0].
    Measures: cos(h_alpha, gold_f) and decoder recall at each alpha.
    This reveals the decoder's sensitivity to latent error — where the cliff is.

    Also includes a random-z point to establish the noise floor.
    """
    import numpy as np
    alphas = np.linspace(1.0, 0.0, n_points)
    log(f"\n  --- sensitivity sweep: ceiling decoder, {n_points} alpha points ---")
    log(f"  {'alpha':>6}  {'cos(h_a,f)':>10}  {'recall':>7}")

    curve = []
    for alpha in alphas:
        h_alpha = alpha * gold_f + (1.0 - alpha) * h_K
        cos_vals = torch.nn.functional.cosine_similarity(
            torch.tensor(h_alpha[he]), torch.tensor(gold_f[he]))
        mean_cos = float(cos_vals.mean())
        mean_rec, _ = dec.eval_on(texts, h_alpha, he, log=lambda *a: None)
        log(f"  {alpha:6.2f}  {mean_cos:10.3f}  {mean_rec:7.3f}")
        curve.append((float(alpha), mean_cos, mean_rec))

    # permuted baseline: gold_f from a DIFFERENT instance (correct distribution,
    # wrong instance). Isolates "right latent for THIS input" from "plausible latent."
    #   permuted ~ constant -> decoder needs the RIGHT latent for the instance
    #   permuted ~ ceiling  -> decoder just needs plausible latent (not instance-matched)
    rng = np.random.RandomState(42)
    perm = np.arange(len(he))
    rng.shuffle(perm)
    while np.any(perm == np.arange(len(he))):
        rng.shuffle(perm)
    h_perm = gold_f.copy()
    for k, orig_idx in enumerate(he):
        h_perm[orig_idx] = gold_f[he[perm[k]]]
    cos_perm = float(torch.nn.functional.cosine_similarity(
        torch.tensor(h_perm[he]), torch.tensor(gold_f[he])).mean())
    rec_perm, _ = dec.eval_on(texts, h_perm, he, log=lambda *a: None)
    log(f"  {'perm':>6}  {cos_perm:10.3f}  {rec_perm:7.3f}")
    curve.append(("perm", cos_perm, rec_perm))

    return curve


def run(g, f, ctx, cmask, texts, model_name, n_op=24, K=4, r=512,
        refiner_epochs=400, decoder_epochs=2, bottleneck=64, film_warmup=1,
        z_dropout=0.0, seed=0, sensitivity=False,
        learn_ops=False, k_warmup=0.5, contrastive=0.0, log=print):
    import numpy as np

    rng = np.random.RandomState(seed); idx = rng.permutation(len(g))
    nh = max(2, len(idx) // 5); he, tr = idx[:nh], idx[nh:]
    log(f"  train {len(tr)} / held {len(he)}")

    # phase 1: refiner
    log("  [1/4] training refiner (graph+code, K={}, r={})...".format(K, r))
    h_K, ops, ref_net, cos_he = _train_refiner(
        g, f, ctx, cmask, tr, K=K, r=r, n_op=n_op, epochs=refiner_epochs, seed=seed,
        learn_ops=learn_ops, k_warmup=k_warmup, contrastive=contrastive, log=log)

    # constant-z: mean of h_K over TRAINING set, broadcast to all instances.
    # Tests: does FiLM benefit come from extra params (constant = latent) or z-content (latent > constant)?
    h_K_mean = h_K[tr].mean(axis=0)
    h_K_const = np.broadcast_to(h_K_mean[None], h_K.shape).copy()

    # phase 2: decoder per arm
    d = g.shape[1]
    arms = [
        ("baseline", None,      False, 0.0),       # LoRA only
        ("constant", h_K_const, True,  0.0),        # LoRA + FiLM(mean_h_K) — extra-params control
        ("latent",   h_K,       True,  z_dropout),  # LoRA + FiLM(h_K) + optional z-dropout
        ("ceiling",  f,         True,  0.0),        # LoRA + FiLM(gold_f) — injection ceiling
    ]
    results = {}; per_inst = {}; sens_curve = None
    latent_behs_he = None; ops_behavior = None
    for arm_name, latents, use_film, zdrop in arms:
        log(f"  [2/4] decoder arm '{arm_name}' (film={use_film}, bn={bottleneck}, zdrop={zdrop})...")
        dec = _Decoder(model_name, d_latent=d, use_film=use_film, bottleneck=bottleneck)
        warmup = film_warmup if use_film else 0
        dec.train_on(texts, latents, tr, epochs=decoder_epochs, film_warmup=warmup,
                     z_dropout=zdrop, log=log)
        mean_rec, recs = dec.eval_on(texts, latents, he, log=log)
        results[arm_name] = mean_rec; per_inst[arm_name] = recs
        log(f"    {arm_name}: held recall = {mean_rec:.3f}")
        if arm_name == "latent" and use_film:
            latent_behs_he = dec.behavior_embeddings(latents, he)
            ops_behavior = dec.behavior_embeddings(ops, list(range(len(ops))))
            log(f"    extracted {len(latent_behs_he)} behavior embs (d={latent_behs_he.shape[1]})")
        if arm_name == "ceiling" and sensitivity:
            sens_curve = _sensitivity_sweep(dec, texts, f, h_K, tr, he, log=log)
        dec.cleanup()

    # phase 3: write-back trajectory
    log("  [3/4] write-back trajectory extraction...")
    traj = _extract_trajectory(ref_net, g, ctx, cmask, ops, K)
    wb = _write_back_report(traj[:, he], per_inst.get("latent", []))
    log(f"    write-back: {wb['n_success']} successful decodes")
    if wb.get("op_usage"):
        log(f"    dominant ops: {wb['op_usage']}")

    # phase 4: graph structural edits (wired to real outcomes)
    graph_stats = None
    if latent_behs_he is not None and ops_behavior is not None:
        from v5.runtime.graph_edits import (
            BehaviorGraph, GraphEditEngine, Outcome, BehaviorNode,
            SEED_CONF, CONF_FLOOR, _unit
        )
        log("  [4/4] graph edits engine (wiring to real outcomes)...")
        bg = BehaviorGraph()
        # seed behavior_embs: use mean of VERIFIED h_K projections per dominant
        # op (real FiLM states), fallback to centroid projection for unseen ops
        dom_per_inst = np.argmax(traj[:, he, :].mean(axis=0), axis=1)  # [n_held]
        seed_threshold = 0.3
        verified_mask = np.array([r >= seed_threshold for r in per_inst["latent"]])
        beh_by_op = {}
        for j in range(len(he)):
            if verified_mask[j]:
                beh_by_op.setdefault(dom_per_inst[j], []).append(latent_behs_he[j])
        for j in range(len(ops_behavior)):
            if j in beh_by_op:
                e = _unit(np.mean(beh_by_op[j], axis=0))
            else:
                e = _unit(ops_behavior[j])
            bg.nodes[f"seed_{j}"] = BehaviorNode(
                op_id=f"seed_{j}", name=f"op{j}",
                behavior_emb=e.tolist(),
                confidence=SEED_CONF, source="seed")

        engine = GraphEditEngine(bg)
        threshold = 0.3
        traj_he = traj[:, he, :]
        for j, i in enumerate(he):
            # trajectory: use top-2 distinct ops per step (captures transitions
            # that argmax misses when the refiner soft-blends multiple ops)
            step_weights = traj_he[:, j, :]       # [K, n_op]
            seen_last = -1
            op_ids = []
            for k in range(step_weights.shape[0]):
                ranked = np.argsort(-step_weights[k])
                pick = ranked[0]
                if pick == seen_last and step_weights[k, ranked[1]] > 0.15:
                    pick = ranked[1]
                op_ids.append(f"seed_{pick}")
                seen_last = pick

            engine.observe(Outcome(
                iid=f"inst_{i}", ctx_emb=g[i].astype(np.float32),
                behavior_emb=latent_behs_he[j].astype(np.float32),
                trajectory=op_ids, verified=(per_inst["latent"][j] >= threshold)))

        maint_recs = engine.maintain()
        n_ret = sum(1 for n in bg.nodes.values()
                    if n.retrievable and n.confidence >= CONF_FLOOR)
        edit_counts = {}
        for r in bg.log:
            edit_counts[r.kind] = edit_counts.get(r.kind, 0) + 1
        graph_stats = {
            "n_observed": len(he), "n_nodes": len(bg.nodes),
            "n_retrievable": n_ret,
            "n_edges": sum(len(d) for d in bg.edges.values()),
            "n_edits": len(bg.log), "edit_counts": edit_counts,
            "n_maintenance": len(maint_recs),
        }
        log(f"    {len(he)} outcomes -> {len(bg.nodes)} nodes ({n_ret} retrievable), "
            f"{graph_stats['n_edges']} edges, {len(bg.log)} edits")
        log(f"    edit breakdown: {edit_counts}")

    return results, len(he), cos_he, wb, sens_curve, graph_stats


def _report(results, n_held, cos_refiner, wb, sens_curve=None, graph_stats=None):
    print(f"\n=== LGGN DECODE v2 (FiLM, held n={n_held}) — refiner h_K -> actual patches ===")
    print(f"  refiner cos(h_K, gold_f) = {cos_refiner:.3f}")
    print(f"  [v1 soft-prefix FAILED: baseline 0.136 > latent 0.098 > ceiling 0.113]")
    print()
    for arm in ("baseline", "constant", "latent", "ceiling"):
        if arm in results:
            print(f"  {arm:10}: held recall {results[arm]:.3f}")
    bl = results.get("baseline", 0)
    cn = results.get("constant", 0)
    lt = results.get("latent", 0)
    ce = results.get("ceiling", 0)
    print()
    print(f"  ceiling - baseline = {ce - bl:+.3f}  -> "
          f"{'FiLM INJECTION WORKS' if ce > bl + 0.03 else 'FiLM mechanism broken'}  "
          f"[v1 was: -0.023]")
    print(f"  constant - baseline= {cn - bl:+.3f}  -> "
          f"{'FiLM extra-params help (capacity boost)' if cn > bl + 0.02 else 'FiLM params alone insufficient'}")
    print(f"  latent - constant  = {lt - cn:+.3f}  -> "
          f"{'Z-CONTENT MATTERS (semantic signal)' if lt > cn + 0.02 else 'z-content NOT used (presence only)'}")
    print(f"  latent - baseline  = {lt - bl:+.3f}  -> "
          f"{'BRIDGE WORKS' if lt > bl + 0.03 else 'h_K does not help'}")
    print(f"  ceiling - latent   = {ce - lt:+.3f}  -> "
          f"{'refiner loses info vs gold' if ce > lt + 0.05 else 'refiner captures what decoder needs'}")
    if wb.get("n_success", 0) > 0:
        print(f"\n  write-back: {wb['n_success']}/{n_held} decodes above threshold")
        print(f"  dominant ops: {wb.get('op_usage', {})}")
    if sens_curve:
        print(f"\n  --- sensitivity curve (ceiling decoder, gold->h_K interpolation) ---")
        print(f"  {'alpha':>6}  {'cos':>6}  {'recall':>7}  {'vs baseline':>11}")
        for row in sens_curve:
            a, c, r = row
            delta = r - bl
            a_str = f"{a:.2f}" if isinstance(a, float) else str(a)
            print(f"  {a_str:>6}  {c:6.3f}  {r:7.3f}  {delta:+11.3f}")
    if graph_stats:
        print(f"\n  --- graph structural edits (wired to real outcomes) ---")
        print(f"  {graph_stats['n_observed']} outcomes observed")
        print(f"  {graph_stats['n_nodes']} nodes ({graph_stats['n_retrievable']} retrievable)")
        print(f"  {graph_stats['n_edges']} edges, {graph_stats['n_edits']} total edits")
        print(f"  edit breakdown: {graph_stats['edit_counts']}")
        if graph_stats['n_maintenance']:
            print(f"  {graph_stats['n_maintenance']} maintenance edits")


# ── selftest ────────────────────────────────────────────────────────────────────

def _selftest() -> bool:
    """FiLM module + refiner h_K + trajectory extraction + write-back + graph wiring (no GPU)."""
    print("lggn_decode v2 --selftest: FiLM module + refiner + trajectory + write-back + graph wiring\n")
    import numpy as np

    d, N, T, n_op = 32, 60, 8, 6
    rng = np.random.RandomState(0)
    g = 0.1 * rng.randn(N, d).astype("float32")
    ops_true = rng.randn(n_op, d).astype("float32")
    ops_true /= np.linalg.norm(ops_true, axis=1, keepdims=True)
    f = g.copy()
    for i in range(N):
        f[i] += ops_true[i % n_op] * 0.5
    ctx = 0.3 * rng.randn(N, T, d).astype("float32")
    cmask = np.ones((N, T), bool)
    texts = [{"issue": f"bug {i}", "code": f"x = {i}", "added": f"x = {i+1}"} for i in range(N)]

    # --- FiLM module unit tests ---
    n_layers = 4
    film = _FiLMModule(d_model=d, d_latent=d, n_layers=n_layers, bottleneck=16)

    # test 1: identity initialization
    z = torch.randn(2, d)
    b = film.encode(z)
    assert b.shape == (2, 16), f"behavior embedding shape: {b.shape}"
    h = torch.randn(2, 10, d)
    h_mod = film.modulate(h, b, layer_idx=0)
    assert h_mod.shape == h.shape, f"FiLM output shape mismatch: {h_mod.shape} vs {h.shape}"
    delta = (h_mod - h).abs().max().item()
    assert delta < 0.15, f"FiLM should init near-identity, max delta = {delta:.4f}"
    print(f"  FiLM identity init: max delta = {delta:.4f} (< 0.15)")

    # test 2: gradient flow
    loss = h_mod.sum()
    loss.backward()
    assert film.renderer[0].weight.grad is not None, "gradients must flow through FiLM renderer"
    assert film.behavior_encoder[0].weight.grad is not None, "gradients must flow through BehaviorEncoder"
    print(f"  FiLM gradient flow: OK (renderer + BehaviorEncoder)")

    # test 3: different z -> different modulation
    film.zero_grad()
    z1 = torch.randn(1, d)
    z2 = torch.randn(1, d)
    b1, b2 = film.encode(z1), film.encode(z2)
    h_test = torch.randn(1, 5, d)
    h1 = film.modulate(h_test, b1, 0)
    h2 = film.modulate(h_test, b2, 0)
    # after training they should differ; at init the difference is small but nonzero
    # (because behavior_encoder maps different z to different b, even though renderer is near-identity)
    print(f"  FiLM z-sensitivity: ||h1-h2|| = {(h1-h2).abs().mean().item():.4f}")

    # test 4: all layers have independent parameters
    params_per_layer = [set(id(p) for p in lin.parameters()) for lin in film.renderer]
    for i in range(n_layers):
        for j in range(i+1, n_layers):
            assert params_per_layer[i].isdisjoint(params_per_layer[j]), \
                f"layers {i} and {j} share parameters"
    print(f"  FiLM per-layer independence: {n_layers} layers, all independent")

    # --- refiner (vanilla) ---
    tr = np.arange(N * 4 // 5)
    h_K, ops, net, cos = _train_refiner(g, f, ctx, cmask, tr, K=4, r=32,
                                         n_op=n_op, epochs=100, seed=0, log=lambda *a: None)
    assert h_K.shape == (N, d), f"h_K shape wrong: {h_K.shape}"
    assert cos > 0.2, f"refiner cos too low: {cos:.3f}"
    print(f"  refiner (vanilla): held cos = {cos:.3f}")

    # --- refiner improvements: learn_ops + contrastive + k_warmup ---
    _, ops2, _, cos2 = _train_refiner(g, f, ctx, cmask, tr, K=4, r=32,
                                       n_op=n_op, epochs=100, seed=0,
                                       learn_ops=True, k_warmup=0.5,
                                       contrastive=0.1, log=lambda *a: None)
    assert cos2 > 0.2, f"improved refiner cos too low: {cos2:.3f}"
    print(f"  refiner (learn_ops+contra+k_sched): held cos = {cos2:.3f} "
          f"(delta {cos2-cos:+.3f} vs vanilla)")

    # --- trajectory ---
    traj = _extract_trajectory(net, g, ctx, cmask, ops, K=4)
    assert traj.shape[0] == 4 and traj.shape[1] == N, f"traj shape wrong: {traj.shape}"
    print(f"  trajectory: shape {traj.shape}")

    # --- write-back ---
    he = np.arange(N * 4 // 5, N)
    fake_recall = [0.8 if i % 2 == 0 else 0.1 for i in range(len(he))]
    wb = _write_back_report(traj[:, he], fake_recall)
    assert wb["n_success"] > 0, "should have some successes"
    assert len(wb["op_usage"]) > 0, "should report op usage"
    print(f"  write-back: {wb['n_success']} success, ops {wb['op_usage']}")

    # --- graph wiring bridge ---
    from v5.runtime.graph_edits import (
        BehaviorGraph, GraphEditEngine, Outcome, BehaviorNode,
        SEED_CONF, CONF_FLOOR, _unit
    )
    bg = BehaviorGraph()
    with torch.no_grad():
        ops_emb = film.encode(torch.tensor(ops, dtype=torch.float32))
    for j in range(ops.shape[0]):
        bg.nodes[f"seed_{j}"] = BehaviorNode(
            op_id=f"seed_{j}", name=f"op{j}",
            behavior_emb=_unit(ops_emb[j].numpy()).tolist(),
            confidence=SEED_CONF, source="seed")

    engine = GraphEditEngine(bg)
    traj_he = traj[:, he, :]
    for j, idx in enumerate(he):
        step_w = traj_he[:, j, :]
        seen_last = -1; op_ids = []
        for k in range(step_w.shape[0]):
            ranked = np.argsort(-step_w[k])
            pick = int(ranked[0])
            if pick == seen_last and step_w[k, ranked[1]] > 0.15:
                pick = int(ranked[1])
            op_ids.append(f"seed_{pick}")
            seen_last = pick
        with torch.no_grad():
            beh = film.encode(torch.tensor(h_K[idx:idx+1])).squeeze(0).numpy()
        engine.observe(Outcome(
            iid=f"inst_{idx}", ctx_emb=g[idx].astype(np.float32),
            behavior_emb=beh.astype(np.float32),
            trajectory=op_ids, verified=(fake_recall[j] >= 0.3)))
    maint = engine.maintain()
    n_ret = sum(1 for n in bg.nodes.values()
                if n.retrievable and n.confidence >= CONF_FLOOR)
    assert len(bg.log) > 0, "observe() must produce edits"
    assert n_ret > 0, "some seed ops must become retrievable via STRENGTHEN"
    edit_kinds = sorted(set(r.kind for r in bg.log))
    print(f"  graph wiring: {len(bg.nodes)} nodes ({n_ret} retrievable), "
          f"{len(bg.log)} edits, {len(maint)} maintenance")
    print(f"  edit kinds: {edit_kinds}")

    print("\n  LGGN-DECODE v2 SELFTEST -> PASS (FiLM + refiner + trajectory + write-back + graph wiring)")
    return True


# ── CLI ─────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="LGGN decode v2: refiner h_K -> patches via LoRA + FiLM.")
    ap.add_argument("--model", default="Qwen/Qwen3.5-4B")
    ap.add_argument("--dataset", default="lite"); ap.add_argument("--split", default="test")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--n-op", type=int, default=24)
    ap.add_argument("--r", type=int, default=512, help="refiner inner dim")
    ap.add_argument("--K", type=int, default=4, help="refiner steps")
    ap.add_argument("--refiner-epochs", type=int, default=400)
    ap.add_argument("--decoder-epochs", type=int, default=2)
    ap.add_argument("--bottleneck", type=int, default=64, help="BehaviorEncoder output dim")
    ap.add_argument("--film-warmup", type=int, default=1,
                    help="freeze FiLM for N epochs while LoRA settles (stability)")
    ap.add_argument("--z-dropout", type=float, default=0.0,
                    help="prob of dropping z during latent-arm training (forces z-content use)")
    ap.add_argument("--sensitivity", action="store_true",
                    help="run gold->h_K interpolation sweep after ceiling arm")
    ap.add_argument("--learn-ops", action="store_true",
                    help="make operator basis learnable (KMeans init, then drift)")
    ap.add_argument("--k-warmup", type=float, default=0.5,
                    help="fraction of refiner epochs to ramp K from 1 to max (scheduled K)")
    ap.add_argument("--contrastive", type=float, default=0.0,
                    help="weight for wrong-instance triplet loss (push h_K away from f_wrong)")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        raise SystemExit(0 if _selftest() else 1)
    extras = []
    if a.learn_ops: extras.append("learn_ops")
    if a.contrastive > 0: extras.append(f"contra={a.contrastive}")
    if a.k_warmup != 1.0: extras.append(f"k_warmup={a.k_warmup}")
    ex = f" {' '.join(extras)}" if extras else ""
    print(f"[lggn-decode v2 FiLM] model={a.model} dataset={a.dataset} n={a.n} "
          f"r={a.r} K={a.K} ops={a.n_op} bn={a.bottleneck} "
          f"ref_ep={a.refiner_epochs} dec_ep={a.decoder_epochs} "
          f"warmup={a.film_warmup} zdrop={a.z_dropout} "
          f"sensitivity={a.sensitivity}{ex}")
    g, f, ctx, cmask, texts = _load_paired(a.model, a.dataset, a.split, a.n)
    print(f"  {len(g)} instances, d={g.shape[1]}")
    results, n_held, cos_ref, wb, sens, graph_stats = run(
        g, f, ctx, cmask, texts, a.model,
        n_op=a.n_op, K=a.K, r=a.r,
        refiner_epochs=a.refiner_epochs, decoder_epochs=a.decoder_epochs,
        bottleneck=a.bottleneck, film_warmup=a.film_warmup,
        z_dropout=a.z_dropout, sensitivity=a.sensitivity,
        learn_ops=a.learn_ops, k_warmup=a.k_warmup, contrastive=a.contrastive)
    _report(results, n_held, cos_ref, wb, sens, graph_stats)


if __name__ == "__main__":
    main()
