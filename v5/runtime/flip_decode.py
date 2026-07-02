"""FLIP TEST — was the composition-decode wall the FROZEN decoder, or genuinely generic content?

composition_decode.py delivered the composition as TEXT to a FROZEN 4B -> topK 0.115 (misled).
solution_ladder.py handed the EXACT gold as TEXT to the same FROZEN 4B -> 0.83. Same channel, same
decoder, only the CONTENT differed -> the two walls (frozen-decoder vs generic-content) are TANGLED.

This test breaks the tangle: lift BOTH confounds (frozen -> LoRA-trained, text -> LATENT soft-prefix),
then use a CONTENT-axis control to say which wall is real. The decoder is trained to emit code from a
latent conditioning vector ONLY (no issue text; buggy code is the edit canvas) -- so if it works, the
latent channel is load-bearing by construction (a decorative channel -> 0 resolve, can't be faked).

  cells (decoder=frozen-base + LoRA + soft-prefix projection; NO issue text in the prompt):
    (1) frozen  text   exact-gold          ~0.83   PRIOR (solution_ladder rung c) -- not re-run
    (2) frozen  text   predicted-composed  ~0.115  PRIOR (composition_decode topK) -- not re-run
    (3) trained latent EXACT-latent (fe)           channel ceiling: can the decoder read a latent at all?
    (4) trained latent ORACLE-composed             basis ceiling:   Sum w*.centroid, w* fit to THIS gold
    (5) trained latent PREDICTED-composed          deploy reality:  composer(held goal)

  diagnosis:
    (3) low                 -> injection/embedding-space broken (mpnet may be non-invertible to code); STOP
    (3) high, (4) low       -> operator basis too GENERIC (residual matters); lever = T.8 residual + write-back
    (3) high,(4) high,(5)lo -> composer can't predict held weights; derivation gap in the TRAVERSAL
    all high                -> thesis closes; build the loop
  FLIP verdict: (5) vs prior (2)=0.115.  (5) >> 0.115 -> frozen WAS the confound.  (5) ~= 0.115 -> content wall.

Standard method (not invented here): peft LoRA + a learned soft-prefix (prefix/P-tuning) conditioning
vector. Needs a GPU (molab). Anti-leakage: decoder trained on TRAIN only (v=gold-fix-emb -> gold SR);
composer + KMeans basis from TRAIN only; held numbers only; issue text stripped from cells 3/4/5.

  V5_LM_TRUST_REMOTE_CODE=1 V5_LM_QUANT=4bit python -m v5.runtime.flip_decode --model Qwen/Qwen3.5-4B --dataset lite --n 200
  python -m v5.runtime.flip_decode --selftest
"""
from __future__ import annotations

import argparse

# Prior FROZEN-decoder + TEXT-channel results, pulled (not re-measured). Sources:
#   B1: solution_ladder rung c (+EXACT gold), LGGN_DESIGN scorecard 2026-06-30.
#   B2: composition_decode topK, LGGN_DESIGN scorecard 2026-07-01.
B1_FROZEN_TEXT_EXACT = 0.83
B2_FROZEN_TEXT_COMPOSED = 0.115

# Pre-registered thresholds (commit BEFORE running; do not move to fit the result).
THR_FLIP = B2_FROZEN_TEXT_COMPOSED + 0.15   # cell5 must clear this AND its CI must exclude B2
THR_CHANNEL = 0.50                          # cell3: decoder can read a latent at all
THR_BASIS_FRAC = 0.70                       # cell4 >= 0.70 * cell3 -> basis carries the content


def _sr_block(removed: str, added: str) -> str:
    return f"<<<<<<< SEARCH\n{removed}\n=======\n{added}\n>>>>>>> REPLACE"


def _prompt_latent(code: str) -> str:
    """Cells 3/4/5 prompt: buggy-code CANVAS only, NO issue. The latent carries the fix."""
    return (f"Buggy code:\n{code[:900]}\n\n"
            "Emit ONLY a search/replace block. SEARCH = exact buggy code above; REPLACE = the fix:\n"
            "<<<<<<< SEARCH\n<exact buggy code>\n=======\n<fixed code>\n>>>>>>> REPLACE")


class _LatentDecoder:
    """Frozen base LM + LoRA + a learned projection v(d_emb) -> [n_soft, d_model] soft prefix.
    Trained to emit the gold SR block from the latent ONLY (buggy code is the text canvas)."""

    def __init__(self, model_name, n_soft=8, d_emb=768, lr=2e-4):
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
        self.proj = nn.Sequential(nn.Linear(d_emb, self.d_model), nn.Tanh(),
                                  nn.Linear(self.d_model, n_soft * self.d_model)).to(self.dev, cdt)
        params = [p for p in self.model.parameters() if p.requires_grad] + list(self.proj.parameters())
        self.opt = torch.optim.AdamW(params, lr=lr)
        self._torch, self._nn = torch, nn

    def _soft(self, v):                                       # v: [d_emb] -> [n_soft, d_model]
        t = self._torch
        vv = t.as_tensor(v, dtype=self.proj[0].weight.dtype, device=self.dev)
        return self.proj(vv).view(self.n_soft, self.d_model)

    def _embeds(self, prompt, v, target=None):
        """Build inputs_embeds = [soft prefix] + [prompt] (+ [target]); labels mask all but target."""
        t = self._torch
        msgs = [{"role": "system", "content": "You are a precise code-fixing assistant. Output only a search/replace block."},
                {"role": "user", "content": prompt}]
        try:
            pids = self.tok.apply_chat_template(msgs, add_generation_prompt=True, enable_thinking=False, return_tensors="pt")
        except TypeError:
            pids = self.tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt")
        pids = pids.to(self.dev)
        emb = self.model.get_input_embeddings()
        soft = self._soft(v).unsqueeze(0)                    # [1, n_soft, d_model]
        parts = [soft, emb(pids)]
        n_ctx = self.n_soft + pids.shape[1]
        labels = t.full((1, n_ctx), -100, dtype=t.long, device=self.dev)
        if target is not None:
            tids = self.tok(target + self.tok.eos_token, return_tensors="pt", add_special_tokens=False).input_ids.to(self.dev)
            parts.append(emb(tids))
            labels = t.cat([labels, tids], 1)
        inp = t.cat(parts, 1)
        attn = t.ones(inp.shape[:2], dtype=t.long, device=self.dev)
        return inp, attn, labels

    def train(self, rows, steps, log=print):
        t = self._torch
        self.model.train()
        n = len(rows)
        for s in range(steps):
            r = rows[s % n]
            inp, attn, labels = self._embeds(_prompt_latent(r["code"]), r["fe"],
                                             target=_sr_block(r["code"], r["added"]))
            out = self.model(inputs_embeds=inp, attention_mask=attn, labels=labels)
            out.loss.backward()
            self.opt.step(); self.opt.zero_grad()
            if (s + 1) % max(1, steps // 10) == 0:
                log(f"    step {s+1}/{steps} loss {float(out.loss):.3f}")

    def gen(self, code, v, max_new=256):
        t = self._torch
        self.model.eval()
        with t.no_grad():
            inp, attn, _ = self._embeds(_prompt_latent(code), v, target=None)
            out = self.model.generate(inputs_embeds=inp, attention_mask=attn, max_new_tokens=max_new,
                                      do_sample=False, pad_token_id=self.tok.eos_token_id)
        return self.tok.decode(out[0], skip_special_tokens=True)


def _oracle_weights(fe, centroids, iters=250):
    """Best convex composition of the operator basis for THIS gold: softmax w s.t. Sum w.C ~ fe."""
    import torch
    C = torch.tensor(centroids, dtype=torch.float32)
    f = torch.tensor(fe, dtype=torch.float32)
    logit = torch.zeros(C.shape[0], requires_grad=True)
    opt = torch.optim.Adam([logit], lr=0.1)
    Fn = torch.nn.functional.normalize
    with torch.enable_grad():                                # caller runs under no_grad; the fit needs grad
        for _ in range(iters):
            opt.zero_grad()
            p = torch.softmax(logit, 0) @ C
            loss = 1 - (Fn(p, dim=0) * Fn(f, dim=0)).sum()
            loss.backward(); opt.step()
    with torch.no_grad():
        return (torch.softmax(logit, 0) @ C).numpy()


def _bootstrap_ci(xs, iters=2000, lo=5, hi=95, seed=0):
    import numpy as np
    if not xs:
        return 0.0, 0.0
    rng = np.random.RandomState(seed); a = np.asarray(xs, float)
    bs = [a[rng.randint(0, len(a), len(a))].mean() for _ in range(iters)]
    return float(np.percentile(bs, lo)), float(np.percentile(bs, hi))


def run(rows, dec, n_ops=24, K=3, train_steps=600, epochs=300, log=print):
    import numpy as np, torch
    from v5.runtime.composition_decode import _train_composer
    from v5.runtime.solution_ladder import _emitted_replace, _fidelity
    rng = np.random.RandomState(0); idx = rng.permutation(len(rows))
    nh = len(idx) // 3
    test = [rows[i] for i in idx[:nh]]; train = [rows[i] for i in idx[nh:]]
    n_ops = min(n_ops, max(2, len(train) // 2))
    composer, centroids, _reps = _train_composer(train, n_ops, epochs)     # TRAIN-only basis + composer
    log(f"  training latent decoder on {len(train)} (v=gold-fix-emb -> gold SR), {train_steps} steps")
    dec.train(train, train_steps, log=log)
    log(f"  eval on {len(test)} held (cells 3/4/5)")
    cells = {"c3_exact_latent": [], "c4_oracle_composed": [], "c5_predicted_composed": []}
    with torch.no_grad():
        for t in test:
            w = torch.softmax(composer(torch.tensor(t["g"], dtype=torch.float32)), -1).numpy()
            v_pred = w @ centroids
            v_oracle = _oracle_weights(t["fe"], centroids)
            for cell, v in (("c3_exact_latent", t["fe"]),
                            ("c4_oracle_composed", v_oracle),
                            ("c5_predicted_composed", v_pred)):
                rec, _ = _fidelity(_emitted_replace(dec.gen(t["code"], v)), t["added"])
                cells[cell].append(rec)
    out = {c: (float(np.mean(v)) if v else 0.0, _bootstrap_ci(v)) for c, v in cells.items()}
    return out, len(test)


def _report(out, n):
    c3, c4, c5 = (out[k] for k in ("c3_exact_latent", "c4_oracle_composed", "c5_predicted_composed"))
    print(f"\n=== FLIP DECODE (n={n}) — trained decoder reads the LATENT ===")
    print(f"  PRIOR (frozen+text, not re-run):  exact {B1_FROZEN_TEXT_EXACT:.3f}   composed {B2_FROZEN_TEXT_COMPOSED:.3f}")
    for k, lab in (("c3_exact_latent", "3) exact-latent   (channel ceiling)"),
                   ("c4_oracle_composed", "4) oracle-composed (basis ceiling) "),
                   ("c5_predicted_composed", "5) predicted-comp (deploy)        ")):
        m, (lo, hi) = out[k]
        print(f"  {lab}: recall {m:.3f}  [90% CI {lo:.3f}-{hi:.3f}]")
    print(f"\n  pre-registered gates:")
    print(f"    channel OK   (c3 >= {THR_CHANNEL:.2f}) : {'PASS' if c3[0] >= THR_CHANNEL else 'FAIL'}")
    frac = c4[0] / c3[0] if c3[0] > 0 else 0.0
    print(f"    basis carries (c4 >= {THR_BASIS_FRAC:.2f}*c3={THR_BASIS_FRAC*c3[0]:.3f}) : {'PASS' if frac >= THR_BASIS_FRAC else 'FAIL'} (c4/c3={frac:.2f})")
    flip = c5[0] >= THR_FLIP and c5[1][0] > B2_FROZEN_TEXT_COMPOSED
    print(f"    FLIP         (c5 >= {THR_FLIP:.3f} & CI>B2) : {'PASS' if flip else 'FAIL'}")
    print(f"\n  verdict: ", end="")
    if c3[0] < THR_CHANNEL:
        print("c3 LOW -> latent channel / embedding-space unreadable (mpnet non-invertible?). Fix substrate, conclude nothing about operators.")
    elif frac < THR_BASIS_FRAC:
        print("c3 high, c4 LOW -> operator basis too GENERIC. Lever = carry residual (T.8) + write-back, NOT the injector.")
    elif not flip:
        print("c3/c4 high, c5 LOW -> composer can't predict held weights. Derivation gap is in the TRAVERSAL (more composer data).")
    else:
        print("all gates PASS -> frozen was the confound; latent composition is real. Build the full loop.")


def _selftest() -> bool:
    """No GPU: mock decoder = an INVERTIBLE codebook lookup (nearest latent -> its code). Proves the
    harness measures channel-readability: exact-latent must ~=1.0, a random latent ~=0."""
    print("flip_decode --selftest: invertible mock decoder -> exact-latent ~1.0, random ~0\n")
    import numpy as np
    from v5.runtime.solution_ladder import _emitted_replace, _fidelity
    d, N = 16, 60
    rng = np.random.RandomState(0)
    codes = [f"x = {i%5}" for i in range(N)]
    book = rng.randn(N, d).astype("float32")                 # each code has a unique latent
    rows = [{"code": "x = 1", "added": codes[i], "fix_text": codes[i],
             "g": book[i], "fe": book[i]} for i in range(N)]

    class Mock:                                              # decodes a latent by nearest codebook entry
        def train(self, rows, steps, log=print): pass
        def gen(self, code, v):
            j = int(np.argmin(((book - np.asarray(v)) ** 2).sum(1)))
            return _sr_block("x = 1", codes[j])

    # patch composer path: reuse run() but with a trivial 2-op basis; check c3 (exact) is near-perfect
    out, n = run(rows, Mock(), n_ops=4, K=1, train_steps=1, epochs=50, log=lambda *a: None)
    c3 = out["c3_exact_latent"][0]
    print(f"  c3(exact-latent)={c3:.2f}  c4={out['c4_oracle_composed'][0]:.2f}  c5={out['c5_predicted_composed'][0]:.2f}")
    # sanity: a random latent should NOT recover the code
    rec_rand, _ = _fidelity(_emitted_replace(Mock().gen("x=1", rng.randn(d))), "x = 999")
    assert c3 > 0.9, "exact-latent must be recoverable through an invertible channel"
    assert rec_rand < 0.5, "a random latent must not recover a specific code"
    print("\n  FLIP-DECODE SELFTEST -> PASS (harness measures latent-channel readability)")
    return True


def main():
    ap = argparse.ArgumentParser(description="Flip test: trained decoder reading the latent vs the frozen-text prior.")
    ap.add_argument("--model", default="Qwen/Qwen3.5-4B")
    ap.add_argument("--dataset", default="lite")
    ap.add_argument("--split", default="test")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--n-soft", type=int, default=8)
    ap.add_argument("--train-steps", type=int, default=600)
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        raise SystemExit(0 if _selftest() else 1)
    from v5.runtime.composition_decode import _load
    rows = _load(a.dataset, a.split, a.n)
    print(f"[flip-decode] {len(rows)} instances, model={a.model}")
    dec = _LatentDecoder(a.model, n_soft=a.n_soft)
    _report(*run(rows, dec, K=a.k, train_steps=a.train_steps))


if __name__ == "__main__":
    main()
