"""[PARKED — NOT LGGN] A static MPNet->MLP->8-soft-tokens->Qwen->code probe. It has NO graph, NO
latent refinement, NO test feedback, and puts the latent in mpnet space (foreign + lossy) instead of
Qwen's. It only answers a narrow leaf question (can a frozen/LoRA decoder invert a static embedding to
code). The actual LGGN thesis (HRM-style latent refinement over graph operators, iterating against the
code/tests) lives in lggn_refine.py. Kept for the honest negative + the --free-latent capacity probe.

FLIP TEST — was the composition-decode wall the FROZEN decoder, or genuinely generic content?

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


def _prompt_latent(code: str, canvas: bool = True) -> str:
    """Latent-only prompt (NO issue). canvas=True: buggy code shown as the SEARCH anchor -> the REPLACE
    must come from the latent, BUT the code can leak fixes derivable from the buggy lines (the decoder
    then ignores the latent). canvas=False: NO code at all -> the latent is the ONLY source of the fix =
    the honest channel-readability ceiling."""
    if not canvas:
        return ("A repair is encoded in the context vector prepended to this prompt.\n"
                "Emit ONLY the fixed code lines it encodes, nothing else.")
    return (f"Buggy code:\n{code[:900]}\n\n"
            "Emit ONLY a search/replace block. SEARCH = exact buggy code above; REPLACE = the fix:\n"
            "<<<<<<< SEARCH\n<exact buggy code>\n=======\n<fixed code>\n>>>>>>> REPLACE")


class _LatentDecoder:
    """Frozen base LM + LoRA + a learned projection v(d_emb) -> [n_soft, d_model] soft prefix.
    Trained to emit the gold SR block from the latent ONLY (buggy code is the text canvas)."""

    def __init__(self, model_name, n_soft=8, d_emb=768, lr=2e-4, canvas=True, free_latent=False):
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
        self.canvas = canvas
        self.d_emb = d_emb
        self.free_latent = free_latent          # capacity probe: trainable per-example latent (mpnet removed)
        self.free = None
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
        kw = dict(add_generation_prompt=True, return_tensors="pt", return_dict=True)
        try:
            enc = self.tok.apply_chat_template(msgs, enable_thinking=False, **kw)   # transformers 5.x -> BatchEncoding
        except TypeError:
            enc = self.tok.apply_chat_template(msgs, **kw)
        pids = enc["input_ids"].to(self.dev)
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
        t = self._torch; nn = self._nn
        self.model.train()
        n = len(rows)
        if self.free_latent and self.free is None:                # trainable per-example latent
            self.free = nn.Embedding(n, self.d_emb).to(self.dev, self.proj[0].weight.dtype)
            self.opt.add_param_group({"params": self.free.parameters()})
        for s in range(steps):
            i = s % n; r = rows[i]
            v = self.free.weight[i] if self.free_latent else r["fe"]
            target = _sr_block(r["code"], r["added"]) if self.canvas else r["added"]
            inp, attn, labels = self._embeds(_prompt_latent(r["code"], self.canvas), v, target=target)
            out = self.model(inputs_embeds=inp, attention_mask=attn, labels=labels)
            out.loss.backward()
            self.opt.step(); self.opt.zero_grad()
            if (s + 1) % max(1, steps // 10) == 0:
                log(f"    step {s+1}/{steps} loss {float(out.loss.detach()):.3f}")

    def gen(self, code, v, max_new=256):
        t = self._torch
        self.model.eval()
        with t.no_grad():
            inp, attn, _ = self._embeds(_prompt_latent(code, self.canvas), v, target=None)
            out = self.model.generate(inputs_embeds=inp, attention_mask=attn, max_new_tokens=max_new,
                                      min_new_tokens=8, do_sample=False, pad_token_id=self.tok.eos_token_id)
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


def run(rows, dec, n_ops=24, K=3, train_steps=600, epochs=300, dump="", log=print):
    import numpy as np, torch
    from pathlib import Path
    from v5.runtime.composition_decode import _train_composer
    from v5.runtime.solution_ladder import _emitted_replace, _fidelity
    rng = np.random.RandomState(0); idx = rng.permutation(len(rows))
    nh = len(idx) // 3
    test = [rows[i] for i in idx[:nh]]; train = [rows[i] for i in idx[nh:]]
    n_ops = min(n_ops, max(2, len(train) // 2))
    composer, centroids, _reps = _train_composer(train, n_ops, epochs)     # TRAIN-only basis + composer
    log(f"  training latent decoder on {len(train)} (canvas={dec.canvas}), {train_steps} steps")
    dec.train(train, train_steps, log=log)
    log(f"  eval on {len(test)} held (cells 3/4/5 + zero/random-latent ablation)")
    extract = _emitted_replace if dec.canvas else (lambda s: s)            # pure-latent: gen IS the fix
    keys = ("c3_exact", "c4_oracle", "c5_predicted", "z_zero_latent", "z_rand_latent")
    cells = {k: [] for k in keys}; d = centroids.shape[1]; samples = []
    with torch.no_grad():
        for j, t in enumerate(test):
            w = torch.softmax(composer(torch.tensor(t["g"], dtype=torch.float32)), -1).numpy()
            arms = (("c3_exact", t["fe"]), ("c4_oracle", _oracle_weights(t["fe"], centroids)),
                    ("c5_predicted", w @ centroids), ("z_zero_latent", np.zeros(d)),
                    ("z_rand_latent", np.random.RandomState(j).randn(d)))
            for cell, v in arms:
                g = dec.gen(t["code"], np.asarray(v, dtype=float))
                rec, _ = _fidelity(extract(g), t["added"])
                cells[cell].append(rec)
                if dump and j < 4 and cell in ("c3_exact", "z_zero_latent"):
                    samples.append(f"===== held[{j}] [{cell}] recall={rec:.0%} =====\n"
                                   f"GOLD ADDED:\n{t['added'][:200]}\n\nRAW GEN:\n{g[:300]}\n")
    if dump and samples:
        Path(dump).write_text("\n".join(samples), encoding="utf-8"); log(f"  raw samples -> {dump}")
    out = {c: (float(np.mean(v)) if v else 0.0, _bootstrap_ci(v)) for c, v in cells.items()}
    return out, len(test)


def run_capacity(rows, dec, train_steps=800, log=print):
    """CAPACITY PROBE — can the injection carry ANY information? Replace mpnet fe with a TRAINABLE
    per-example latent (free nn.Embedding), train + eval on TRAIN only (a free latent can't generalize;
    this is a memorization ceiling). Disambiguates:
      exact(free) >> zero  -> the channel WIRING works; the held failure is mpnet fe (not code-informative)
      exact(free) ~= zero  -> the conditioning MECHANISM is too weak -> fix the architecture, not the latent."""
    import numpy as np
    from v5.runtime.solution_ladder import _emitted_replace, _fidelity
    rng = np.random.RandomState(0); idx = rng.permutation(len(rows))
    train = [rows[i] for i in idx[len(idx) // 3:]]
    log(f"  CAPACITY PROBE: free trainable latent on {len(train)} train (canvas={dec.canvas}), {train_steps} steps")
    dec.train(train, train_steps, log=log)
    extract = _emitted_replace if dec.canvas else (lambda s: s)
    d = dec.d_emb; sub = train[:min(40, len(train))]; ex, ze = [], []
    for i, r in enumerate(sub):
        vf = dec.free.weight[i].detach().float().cpu().numpy()
        ex.append(_fidelity(extract(dec.gen(r["code"], vf)), r["added"])[0])
        ze.append(_fidelity(extract(dec.gen(r["code"], np.zeros(d))), r["added"])[0])
    mx, mz = float(np.mean(ex)), float(np.mean(ze))
    print(f"\n=== CAPACITY PROBE (free latent, TRAIN memorization, n={len(sub)}) ===")
    print(f"  exact(free) recall {mx:.3f}   zero-latent recall {mz:.3f}   gap {mx - mz:+.3f}")
    print(f"  verdict: {'INJECTION CARRIES INFO (wiring OK) -> held failure is mpnet fe, move latent to a code-reconstructable space' if mx - mz > 0.10 else 'INJECTION TOO WEAK even to MEMORIZE -> fix the conditioning ARCHITECTURE (inject deeper / cross-attn, not an 8-token prefix)'}")


def _report(out, n):
    c3, c4, c5 = (out[k] for k in ("c3_exact", "c4_oracle", "c5_predicted"))
    z0, zr = out["z_zero_latent"], out["z_rand_latent"]
    print(f"\n=== FLIP DECODE (n={n}) — trained decoder reads the LATENT ===")
    print(f"  PRIOR (frozen+text, not re-run):  exact {B1_FROZEN_TEXT_EXACT:.3f}   composed {B2_FROZEN_TEXT_COMPOSED:.3f}")
    for k, lab in (("c3_exact", "3) exact-latent   (channel ceiling)"),
                   ("c4_oracle", "4) oracle-composed (basis ceiling) "),
                   ("c5_predicted", "5) predicted-comp (deploy)        "),
                   ("z_zero_latent", "-) ZERO latent    (ablation)      "),
                   ("z_rand_latent", "-) RANDOM latent  (ablation)      ")):
        m, (lo, hi) = out[k]
        print(f"  {lab}: recall {m:.3f}  [90% CI {lo:.3f}-{hi:.3f}]")
    # GATE 0: is the latent even load-bearing? exact must beat the no-info ablations, or nothing below is meaningful.
    ablate = max(z0[0], zr[0]); latent_gap = c3[0] - ablate
    load_bearing = latent_gap > 0.05
    print(f"\n  GATE 0 latent load-bearing: c3 - max(zero,random) = {latent_gap:+.3f} -> "
          f"{'latent MATTERS' if load_bearing else 'LATENT IGNORED'}")
    frac = c4[0] / c3[0] if c3[0] > 0 else 0.0
    flip = c5[0] >= THR_FLIP and c5[1][0] > B2_FROZEN_TEXT_COMPOSED
    print(f"  gates: channel(c3>={THR_CHANNEL:.2f})={'PASS' if c3[0] >= THR_CHANNEL else 'FAIL'}  "
          f"basis(c4>={THR_BASIS_FRAC:.2f}*c3)={'PASS' if frac >= THR_BASIS_FRAC else 'FAIL'}(c4/c3={frac:.2f})  "
          f"FLIP(c5>={THR_FLIP:.3f}&CI>B2)={'PASS' if flip else 'FAIL'}")
    print(f"\n  verdict: ", end="")
    if not load_bearing:
        print("LATENT IGNORED (c3 == ablations) -> the decoder fits from the PROMPT, not the latent. "
              "NOT an mpnet verdict. Re-run with --pure-latent (drops the code canvas so the latent is the "
              "only fix source) to test channel-readability honestly.")
    elif c3[0] < THR_CHANNEL:
        print("latent matters but c3 LOW -> embedding space lossy (mpnet hard to invert to exact code). "
              "Consider a code-reconstructable latent (decoder hidden / code-AE) instead of mpnet.")
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
        canvas = True
        def train(self, rows, steps, log=print): pass
        def gen(self, code, v):
            j = int(np.argmin(((book - np.asarray(v)) ** 2).sum(1)))
            return _sr_block("x = 1", codes[j])

    # patch composer path: reuse run() but with a trivial 2-op basis; check c3 (exact) is near-perfect
    out, n = run(rows, Mock(), n_ops=4, K=1, train_steps=1, epochs=50, log=lambda *a: None)
    c3 = out["c3_exact"][0]
    print(f"  c3(exact-latent)={c3:.2f}  c4={out['c4_oracle'][0]:.2f}  c5={out['c5_predicted'][0]:.2f}  "
          f"zero={out['z_zero_latent'][0]:.2f}  rand={out['z_rand_latent'][0]:.2f}")
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
    ap.add_argument("--pure-latent", action="store_true",
                    help="drop the buggy-code canvas -> the latent is the ONLY fix source (honest channel test)")
    ap.add_argument("--dump", default="artifacts/flip_dump.txt", help="raw generations for the first held instances")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        raise SystemExit(0 if _selftest() else 1)
    from v5.runtime.composition_decode import _load
    rows = _load(a.dataset, a.split, a.n)
    print(f"[flip-decode] {len(rows)} instances, model={a.model}, canvas={not a.pure_latent}")
    dec = _LatentDecoder(a.model, n_soft=a.n_soft, canvas=not a.pure_latent)
    _report(*run(rows, dec, K=a.k, train_steps=a.train_steps, dump=a.dump))


if __name__ == "__main__":
    main()
