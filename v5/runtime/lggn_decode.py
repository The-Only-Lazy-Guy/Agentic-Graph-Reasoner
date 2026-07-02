"""LGGN decode — does the refiner's h_K help a trained decoder emit actual patches?

lggn_refine proved: h_K reaches cos=0.593 with the gold fix in Qwen space (all 4 pillars PASS at
r=512). But reaching != decoding. The frozen LLM can't cash in the composition (composition_decode:
ops HURT). This lifts both confounds: train the LM (LoRA) AND inject h_K as a LATENT soft prefix
(not text). h_K is already in Qwen hidden space, so the projection is natural (same dim).

Pipeline: extract Qwen reprs (cached) -> train refiner -> h_K per instance -> train LoRA decoder
with soft_prefix(h_K) + issue+code prompt -> gold SR block -> eval held recall.

Arms (each trains a fresh LoRA adapter):
  baseline   LoRA only, no soft prefix (issue+code -> gold SR)
  latent     LoRA + soft_prefix(h_K)  (THE TEST — does the refiner help decode?)
  ceiling    LoRA + soft_prefix(gold_f)  (injection ceiling — can the mechanism work at all?)

  latent > baseline -> h_K helps decode (the bridge works)
  ceiling >> latent -> refiner loses info vs gold (room to improve refiner)
  ceiling ~ baseline -> injection mechanism broken (LoRA can't read the prefix)

Write-back (post-decode): extract the refiner's per-step op-trajectory for successful decodes.
Report which ops were used. Scaffold for actual graph edits via lggn.py OperatorLibrary.

VRAM: ~3.5GB (4-bit Qwen3.5-4B + LoRA + projection). Under 6GB.

  V5_LM_TRUST_REMOTE_CODE=1 V5_LM_QUANT=4bit python -m v5.runtime.lggn_decode \\
      --model Qwen/Qwen3.5-4B --dataset lite --n 200 --r 512
  python -m v5.runtime.lggn_decode --selftest
"""
from __future__ import annotations

import argparse


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

def _train_refiner(g, f, ctx, cmask, tr, K=4, r=512, n_op=24, epochs=400, seed=0, log=print):
    """Train refiner on tr, return (h_K_all, ops, net, cos_held)."""
    import torch, numpy as np
    from v5.runtime.lggn_refine import Refiner, _discover_ops
    torch.manual_seed(seed)
    disp_tr = (f - g)[tr]
    ops = _discover_ops(disp_tr, n_op, seed)
    net = Refiner.Net(g.shape[1], r=r, n_op=ops.shape[0])
    T = lambda x: torch.as_tensor(x, dtype=torch.float32)
    gt, ft, ct, ot = T(g), T(f), T(ctx), T(ops); cm = torch.as_tensor(cmask)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-4)
    Fn = torch.nn.functional.cosine_similarity
    for _ in range(epochs):
        net.train(); opt.zero_grad()
        h = net(gt[tr], ct[tr], cm[tr], ot, K, True, True)
        loss = (1 - Fn(h, ft[tr])).mean(); loss.backward(); opt.step()
    net.eval()
    with torch.no_grad():
        h_all = net(gt, ct, cm, ot, K, True, True).float().numpy()
        he = np.setdiff1d(np.arange(len(g)), tr)
        cos_he = float(Fn(torch.tensor(h_all[he]), ft[he]).mean())
    log(f"    refiner: held cos(h_K, f) = {cos_he:.3f}")
    return h_all, ops, net, cos_he


def _extract_trajectory(net, g, ctx, cmask, ops, K):
    """Per-step op-selection weights from the trained refiner. Shape [K, N, n_op]."""
    import torch, math, numpy as np
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


# ── decoder (LoRA Qwen + soft prefix from h_K) ─────────────────────────────────

def _prompt(text):
    return (f"Fix this bug.\n\nIssue:\n{text['issue']}\n\nBuggy code:\n{text['code']}\n\n"
            "Output ONLY a search/replace block:\n"
            "<<<<<<< SEARCH\n<exact buggy code>\n=======\n<fixed code>\n>>>>>>> REPLACE")


def _target(text):
    return f"<<<<<<< SEARCH\n{text['code']}\n=======\n{text['added']}\n>>>>>>> REPLACE"


class _Decoder:
    """Qwen + LoRA + learned soft-prefix projection from a latent (h_K or f)."""

    def __init__(self, model_name, d_latent, n_soft=4, lr=2e-4):
        import os, torch, torch.nn as nn
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
        self.n_soft = n_soft
        cdt = next(p for p in self.model.parameters() if p.is_floating_point()).dtype
        self.proj = None
        if n_soft > 0:
            self.proj = nn.Sequential(nn.Linear(d_latent, 512), nn.GELU(),
                                      nn.Linear(512, n_soft * self.d_model)).to(self.dev, cdt)
        params = [p for p in self.model.parameters() if p.requires_grad]
        if self.proj:
            params += list(self.proj.parameters())
        self.opt = torch.optim.AdamW(params, lr=lr)
        self._torch = torch

    def _build(self, prompt_text, v=None, target_text=None):
        """inputs_embeds = [soft prefix?] + [prompt tokens] (+ [target tokens]); labels mask prompt."""
        t = self._torch
        msgs = [{"role": "user", "content": prompt_text}]
        kw = dict(add_generation_prompt=True, return_tensors="pt", return_dict=True)
        try:
            enc = self.tok.apply_chat_template(msgs, enable_thinking=False, **kw)
        except TypeError:
            enc = self.tok.apply_chat_template(msgs, **kw)
        pids = enc["input_ids"].to(self.dev)
        emb = self.model.get_input_embeddings()
        parts, n_pfx = [], 0
        if v is not None and self.proj is not None:
            cdt = self.proj[0].weight.dtype
            vt = t.as_tensor(v, dtype=cdt, device=self.dev)
            parts.append(self.proj(vt).view(1, self.n_soft, self.d_model))
            n_pfx = self.n_soft
        parts.append(emb(pids))
        n_ctx = n_pfx + pids.shape[1]
        labels = t.full((1, n_ctx), -100, dtype=t.long, device=self.dev)
        if target_text is not None:
            tids = self.tok(target_text + self.tok.eos_token, return_tensors="pt",
                            add_special_tokens=False).input_ids.to(self.dev)
            parts.append(emb(tids))
            labels = t.cat([labels, tids], 1)
        inp = t.cat(parts, 1)
        attn = t.ones(inp.shape[:2], dtype=t.long, device=self.dev)
        return inp, attn, labels

    def train_on(self, texts, latents, indices, epochs=2, log=print):
        self.model.train()
        for ep in range(epochs):
            tot = 0.0
            for i in indices:
                v = latents[i] if latents is not None else None
                inp, attn, labels = self._build(_prompt(texts[i]), v, _target(texts[i]))
                out = self.model(inputs_embeds=inp, attention_mask=attn, labels=labels)
                self.opt.zero_grad(); out.loss.backward(); self.opt.step()
                tot += float(out.loss.detach())
            log(f"      epoch {ep+1}/{epochs}: loss {tot/max(1,len(indices)):.3f}")

    def eval_on(self, texts, latents, indices, log=print):
        """Returns (mean_recall, per_instance_recalls)."""
        import numpy as np
        from v5.runtime.solution_ladder import _emitted_replace, _fidelity
        self.model.eval(); recs = []
        with self._torch.no_grad():
            for j, i in enumerate(indices):
                v = latents[i] if latents is not None else None
                inp, attn, _ = self._build(_prompt(texts[i]), v)
                out = self.model.generate(inputs_embeds=inp, attention_mask=attn,
                                          max_new_tokens=256, min_new_tokens=8,
                                          do_sample=False, pad_token_id=self.tok.eos_token_id)
                raw = self.tok.decode(out[0], skip_special_tokens=True)
                rec, _ = _fidelity(_emitted_replace(raw), texts[i]["added"])
                recs.append(rec)
                if (j + 1) % 10 == 0:
                    log(f"      {j+1}/{len(indices)} eval'd, running recall {np.mean(recs):.3f}")
        return float(np.mean(recs)) if recs else 0.0, recs

    def cleanup(self):
        import gc
        del self.model, self.proj, self.opt
        gc.collect()
        self._torch.cuda.empty_cache()


# ── experiment ──────────────────────────────────────────────────────────────────

def run(g, f, ctx, cmask, texts, model_name, n_op=24, K=4, r=512,
        refiner_epochs=400, decoder_epochs=2, n_soft=4, seed=0, log=print):
    import numpy as np, torch

    rng = np.random.RandomState(seed); idx = rng.permutation(len(g))
    nh = max(2, len(idx) // 5); he, tr = idx[:nh], idx[nh:]
    log(f"  train {len(tr)} / held {len(he)}")

    # phase 1: refiner
    log("  [1/3] training refiner (graph+code, K={}, r={})...".format(K, r))
    h_K, ops, ref_net, cos_he = _train_refiner(
        g, f, ctx, cmask, tr, K=K, r=r, n_op=n_op, epochs=refiner_epochs, seed=seed, log=log)

    # phase 2: decoder per arm
    d = g.shape[1]
    arms = [("baseline", None, 0), ("latent", h_K, n_soft), ("ceiling", f, n_soft)]
    results = {}; per_inst = {}
    for arm_name, latents, ns in arms:
        log(f"  [2/3] decoder arm '{arm_name}' (n_soft={ns})...")
        dec = _Decoder(model_name, d_latent=d, n_soft=ns)
        dec.train_on(texts, latents, tr, epochs=decoder_epochs, log=log)
        mean_rec, recs = dec.eval_on(texts, latents, he, log=log)
        results[arm_name] = mean_rec; per_inst[arm_name] = recs
        log(f"    {arm_name}: held recall = {mean_rec:.3f}")
        dec.cleanup()

    # phase 3: write-back trajectory
    log("  [3/3] write-back trajectory extraction...")
    traj = _extract_trajectory(ref_net, g, ctx, cmask, ops, K)
    wb = _write_back_report(traj[:, he], per_inst.get("latent", []))
    log(f"    write-back: {wb['n_success']} successful decodes")
    if wb.get("op_usage"):
        log(f"    dominant ops: {wb['op_usage']}")

    return results, len(he), cos_he, wb


def _report(results, n_held, cos_refiner, wb):
    print(f"\n=== LGGN DECODE (held n={n_held}) — refiner h_K -> actual patches ===")
    print(f"  refiner cos(h_K, gold_f) = {cos_refiner:.3f}")
    for arm in ("baseline", "latent", "ceiling"):
        if arm in results:
            print(f"  {arm:10}: held recall {results[arm]:.3f}")
    bl, lt, ce = results.get("baseline", 0), results.get("latent", 0), results.get("ceiling", 0)
    print()
    print(f"  latent - baseline  = {lt - bl:+.3f}  -> "
          f"{'BRIDGE WORKS (h_K helps decode)' if lt > bl + 0.03 else 'h_K does not help the trained decoder'}")
    print(f"  ceiling - latent   = {ce - lt:+.3f}  -> "
          f"{'refiner loses info vs gold (improve refiner)' if ce > lt + 0.05 else 'refiner captures what decoder needs'}")
    print(f"  ceiling - baseline = {ce - bl:+.3f}  -> "
          f"{'INJECTION WORKS' if ce > bl + 0.03 else 'injection mechanism broken'}")
    if wb.get("n_success", 0) > 0:
        print(f"\n  write-back: {wb['n_success']}/{n_held} decodes above threshold")
        print(f"  dominant ops: {wb.get('op_usage', {})}")
        print(f"  (wire to lggn.py OperatorLibrary.add_or_strengthen for actual graph edits)")


# ── selftest ────────────────────────────────────────────────────────────────────

def _selftest() -> bool:
    """Data pairing + refiner h_K + trajectory extraction + write-back (no GPU)."""
    print("lggn_decode --selftest: refiner h_K + trajectory extraction + write-back report\n")
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

    tr = np.arange(N * 4 // 5)
    h_K, ops, net, cos = _train_refiner(g, f, ctx, cmask, tr, K=4, r=32,
                                         n_op=n_op, epochs=100, seed=0, log=lambda *a: None)
    assert h_K.shape == (N, d), f"h_K shape wrong: {h_K.shape}"
    assert cos > 0.2, f"refiner cos too low: {cos:.3f}"
    print(f"  refiner: h_K shape {h_K.shape}, held cos = {cos:.3f}")

    traj = _extract_trajectory(net, g, ctx, cmask, ops, K=4)
    assert traj.shape[0] == 4 and traj.shape[1] == N, f"traj shape wrong: {traj.shape}"
    print(f"  trajectory: shape {traj.shape}")

    he = np.arange(N * 4 // 5, N)
    fake_recall = [0.8 if i % 2 == 0 else 0.1 for i in range(len(he))]
    wb = _write_back_report(traj[:, he], fake_recall)
    assert wb["n_success"] > 0, "should have some successes"
    assert len(wb["op_usage"]) > 0, "should report op usage"
    print(f"  write-back: {wb['n_success']} success, ops {wb['op_usage']}")

    print("\n  LGGN-DECODE SELFTEST -> PASS (refiner h_K + trajectory + write-back)")
    return True


# ── CLI ─────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="LGGN decode: refiner h_K -> actual patches via LoRA + soft prefix.")
    ap.add_argument("--model", default="Qwen/Qwen3.5-4B")
    ap.add_argument("--dataset", default="lite"); ap.add_argument("--split", default="test")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--n-op", type=int, default=24)
    ap.add_argument("--r", type=int, default=512, help="refiner inner dim")
    ap.add_argument("--K", type=int, default=4, help="refiner steps")
    ap.add_argument("--refiner-epochs", type=int, default=400)
    ap.add_argument("--decoder-epochs", type=int, default=2)
    ap.add_argument("--n-soft", type=int, default=4, help="soft prefix tokens from h_K")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        raise SystemExit(0 if _selftest() else 1)
    print(f"[lggn-decode] model={a.model} dataset={a.dataset} n={a.n} r={a.r} K={a.K}")
    g, f, ctx, cmask, texts = _load_paired(a.model, a.dataset, a.split, a.n)
    print(f"  {len(g)} instances, d={g.shape[1]}")
    results, n_held, cos_ref, wb = run(
        g, f, ctx, cmask, texts, a.model,
        n_op=a.n_op, K=a.K, r=a.r,
        refiner_epochs=a.refiner_epochs, decoder_epochs=a.decoder_epochs,
        n_soft=a.n_soft)
    _report(results, n_held, cos_ref, wb)


if __name__ == "__main__":
    main()
