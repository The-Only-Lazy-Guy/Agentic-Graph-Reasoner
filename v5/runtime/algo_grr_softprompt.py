"""algo_grr_softprompt — Option 2: the owned LATENT reasoner (TRM -> soft prompt -> FROZEN LM).

The z-wall was measured for a foreign single-layer latent injected into a frozen LM. This routes the
TRM's reasoning latent into the LM's NATIVE input space instead: the TRM produces z, a small owned
projection turns z into K "virtual tokens" (soft-prompt embeddings) prepended to the LM input, and the
FROZEN LM attends to them like real tokens. Only the TRM + projection train (gradients flow THROUGH the
frozen LM into the prefix, never into the LM's weights) -> the LM's knowledge is untouched, so poison is
CONTAINED in the owned, resettable adapter. STaR-trained on VERIFIED code (verifier-grounded). On a failed
attempt, the error feeds back into the TRM -> new z -> new soft prompt -> retry (a LATENT feedback loop,
not a text handoff).

Open question this tests: does a TRM-computed soft prompt add CAPABILITY the frozen LM lacks, or does it
SATURATE (like the earlier L26 reader)? If it lifts solve-rate vs no-prefix, the owned latent reasoner
works and the TRM is the thing to scale. Verify gates every output, so it can never produce wrong code —
worst case it just doesn't help.

    selftest (no GPU):  python -m v5.runtime.algo_grr_softprompt --selftest
      proves the PLUMBING on a tiny frozen stub LM: (1) gradients reach the prefix, NOT the LM;
      (2) prefix-training conditions a FROZEN LM's output (loss drops from baseline); (3) failure
      feedback changes the latent. The capability A/B is molab (real 3B).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = str(Path(__file__).resolve().parents[2])
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)


def _build():
    import torch
    import torch.nn as nn

    class SoftPromptTRM(nn.Module):
        """Owned reasoner: (task, atoms, optional failure) -> latent z -> K soft-prompt vectors in the
        LM's embedding space. This is the ONLY thing that trains (with the tiny projection)."""

        def __init__(self, d_in: int, d_model: int, d: int = 128, K: int = 8, T: int = 3):
            super().__init__()
            self.K, self.d_model, self.T = K, d_model, T
            self.task = nn.Linear(d_in, d)
            self.atom = nn.Linear(d_in, d)
            self.fail = nn.Linear(d_in, d)
            self.q = nn.Linear(2 * d, d)
            self.f = nn.Linear(4 * d, d)                          # [task, atom-summary, z, failure]
            self.ln = nn.LayerNorm(d)                             # keeps z responsive (no tanh saturation)
            self.z0 = nn.Parameter(torch.zeros(d))
            self.proj = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, K * d_model))

        def forward(self, x_vec, atom_vecs, fail_vec=None):
            x = self.task(x_vec)                                  # [d]
            A = self.atom(atom_vecs)                              # [N, d]
            ft = self.fail(fail_vec) if fail_vec is not None else torch.zeros_like(x)
            z = self.z0
            for _ in range(self.T):
                y = torch.softmax((A @ self.q(torch.cat([x, z]))) / (A.shape[-1] ** 0.5), dim=0)
                asum = y @ A                                     # [d]
                z = self.ln(self.f(torch.cat([x, asum, z, ft])))     # failure is a first-class input
            sp = self.proj(z).view(self.K, self.d_model)          # [K, d_model]
            return sp, z

    class StubLM(nn.Module):
        """Tiny FROZEN causal LM for the no-GPU plumbing proof: embed -> 1 self-attn block -> lm_head.
        The soft-prompt tokens sit at the front and attend-mix into later positions (causal), so the
        prefix genuinely conditions the OUTPUT — a faithful mini-proxy for a real frozen LM."""

        def __init__(self, vocab: int, d_model: int):
            super().__init__()
            self.embed = nn.Embedding(vocab, d_model)
            self.attn = nn.MultiheadAttention(d_model, num_heads=2, batch_first=True)
            self.ln = nn.LayerNorm(d_model)
            self.head = nn.Linear(d_model, vocab)

        def forward_embeds(self, inputs_embeds):                  # [1, S, d_model] -> [1, S, vocab]
            S = inputs_embeds.shape[1]
            mask = torch.triu(torch.ones(S, S), diagonal=1).bool()
            h, _ = self.attn(inputs_embeds, inputs_embeds, inputs_embeds, attn_mask=mask)
            return self.head(self.ln(inputs_embeds + h))

    return torch, nn, SoftPromptTRM, StubLM


def _prefix_loss(torch, nn, lm, soft_prompt, input_ids, target_ids):
    """Next-token CE on target_ids, with the soft prompt prepended to the input embeddings. Differentiable
    through the FROZEN lm into soft_prompt (and thus the TRM + projection)."""
    emb = lm.embed(input_ids)                                     # [1, L, d]
    full = torch.cat([soft_prompt.unsqueeze(0), emb], dim=1)      # [1, K+L, d]
    logits = lm.forward_embeds(full)                             # [1, K+L, vocab]
    K = soft_prompt.shape[0]
    Lt = target_ids.shape[1]
    # positions that predict each target token: the K+L-1 ... slice covering the last Lt targets
    pred = logits[:, K + input_ids.shape[1] - Lt - 1: K + input_ids.shape[1] - 1, :]
    return nn.functional.cross_entropy(pred.reshape(-1, logits.shape[-1]), target_ids.reshape(-1))


# ═══════════════════════════════════════════════════════════════════════════════
# SELFTEST — plumbing proof on a frozen stub LM (no GPU)
# ═══════════════════════════════════════════════════════════════════════════════

def _selftest() -> bool:
    print("algo_grr_softprompt --selftest: owned latent reasoner plumbing (frozen stub LM)\n")
    torch, nn, SoftPromptTRM, StubLM = _build()
    torch.manual_seed(0)
    d_in, d_model, vocab, K = 32, 48, 40, 6
    ok = True

    lm = StubLM(vocab, d_model)
    for p in lm.parameters():
        p.requires_grad_(False)                                   # FROZEN LM
    sptrm = SoftPromptTRM(d_in, d_model, d=64, K=K, T=3)

    # a synthetic "task": a task vector + atom vectors + a target token sequence the LM should produce
    x = torch.randn(d_in)
    A = torch.randn(5, d_in)
    input_ids = torch.randint(0, vocab, (1, 6))
    target_ids = torch.randint(0, vocab, (1, 4))

    # ── [1] gradient isolation: grads reach the prefix (TRM+proj), NOT the LM's weights ──────────
    sp, _z = sptrm(x, A)
    loss = _prefix_loss(torch, nn, lm, sp, input_ids, target_ids)
    loss.backward()
    trm_grad = any(p.grad is not None and p.grad.abs().sum() > 0 for p in sptrm.parameters())
    lm_grad = any(p.grad is not None for p in lm.parameters())
    print(f"  [1] gradient isolation: TRM+proj get grad={trm_grad}, LM gets grad={lm_grad} -> "
          f"{'PASS' if trm_grad and not lm_grad else 'FAIL'}")
    ok &= trm_grad and not lm_grad

    # ── [2] CONDITIONING: training only the prefix steers the FROZEN LM's output toward the target ─
    torch.manual_seed(0)
    lm = StubLM(vocab, d_model)
    for p in lm.parameters():
        p.requires_grad_(False)
    sptrm = SoftPromptTRM(d_in, d_model, d=64, K=K, T=3)
    with torch.no_grad():
        base = _prefix_loss(torch, nn, lm, torch.zeros(K, d_model), input_ids, target_ids).item()
    opt = torch.optim.Adam(sptrm.parameters(), lr=5e-3)
    for _ in range(150):
        sp, _z = sptrm(x, A)
        loss = _prefix_loss(torch, nn, lm, sp, input_ids, target_ids)
        opt.zero_grad(); loss.backward(); opt.step()
    trained = loss.item()
    print(f"  [2] conditioning a FROZEN LM: loss {base:.3f} (no prefix) -> {trained:.3f} (trained prefix) "
          f"-> {'PASS' if trained < base * 0.5 else 'FAIL'}")
    ok &= trained < base * 0.5

    # ── [3] FEEDBACK: a failure vector changes the latent z (the retry loop has an effect) ──────────
    sp0, z0 = sptrm(x, A)
    sp1, z1 = sptrm(x, A, fail_vec=torch.randn(d_in))
    delta = (z1 - z0).abs().sum().item()
    print(f"  [3] failure feedback shifts z by {delta:.3f} -> {'PASS' if delta > 1e-3 else 'FAIL'}")
    ok &= delta > 1e-3

    # ── [4] param budget: the owned adapter is TINY (deployable) ────────────────────────────────
    n_params = sum(p.numel() for p in sptrm.parameters())
    print(f"  [4] owned adapter size: {n_params/1e3:.1f}k params (frozen LM untouched) -> "
          f"{'PASS' if n_params < 5e6 else 'FAIL'}")
    ok &= n_params < 5e6

    print(f"\n  ALGO_GRR_SOFTPROMPT SELFTEST -> {'PASS' if ok else 'FAIL'}")
    print("  (plumbing proven no-GPU. CAPABILITY A/B — does the soft prompt lift a real 3B's solve rate")
    print("   vs no-prefix, or saturate? — is the molab experiment.)")
    return ok


# ═══════════════════════════════════════════════════════════════════════════════
# REAL-3B CAPABILITY A/B (molab) — does the soft prompt lift solve-rate, or saturate?
# ═══════════════════════════════════════════════════════════════════════════════

def make_lm_bundle(model_name: str):
    """Load the FROZEN LM once. Returns (model, tok, embed_layer, d_model, device)."""
    import os
    import torch
    from transformers import AutoTokenizer
    from v5.lm_loader import load_frozen_lm
    trust = os.environ.get("V5_LM_TRUST_REMOTE_CODE", "0").lower() in ("1", "true", "yes")
    tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=trust)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = load_frozen_lm(model_name)                    # weights frozen (requires_grad False)
    embed_layer = model.get_input_embeddings()
    d_model = embed_layer.embedding_dim
    return model, tok, embed_layer, d_model, next(model.parameters()).device


def _prompt_ids(tok, text, device):
    msg = tok.apply_chat_template([{"role": "user", "content": text}], tokenize=False,
                                  add_generation_prompt=True)
    return tok(msg, return_tensors="pt", add_special_tokens=False).input_ids.to(device)


def softprompt_generate(bundle, sptrm, task_text, atom_vecs, embed_fn, prompt_text,
                        fail_vec=None, max_new_tokens=256, temperature=0.6):
    """Generate code from the FROZEN LM conditioned on the TRM's soft prompt (prepended in embed space)."""
    import torch
    model, tok, embed_layer, d_model, dev = bundle
    x = torch.as_tensor(embed_fn(task_text), dtype=torch.float32, device=dev)
    A = torch.as_tensor(atom_vecs, dtype=torch.float32, device=dev)
    fv = None if fail_vec is None else torch.as_tensor(fail_vec, dtype=torch.float32, device=dev)
    with torch.no_grad():
        sp, _z = sptrm(x, A, fv)                          # [K, d_model]
        ids = _prompt_ids(tok, prompt_text, dev)
        emb = embed_layer(ids)                            # [1, L, d_model]
        full = torch.cat([sp.to(emb.dtype).unsqueeze(0), emb], dim=1)
        out = model.generate(inputs_embeds=full, do_sample=True, temperature=temperature,
                             top_p=0.95, max_new_tokens=max_new_tokens, pad_token_id=tok.pad_token_id)
    return tok.decode(out[0], skip_special_tokens=True)


def train_softprompt(bundle, sptrm, traces, embed_fn, render_prompt, steps=600, lr=3e-3, seed=0):
    """STaR/prefix-tuning: teach the TRM+projection to make the FROZEN LM produce the verified code.
    Gradients flow THROUGH the frozen LM into the prefix ONLY (LM weights never update)."""
    import random
    import torch
    import torch.nn as nn
    model, tok, embed_layer, d_model, dev = bundle
    sptrm.to(dev)
    opt = torch.optim.Adam(sptrm.parameters(), lr=lr)
    rng = random.Random(seed)
    K = sptrm.K
    for step in range(steps):
        tr = traces[rng.randrange(len(traces))]           # (task_text, atom_vecs, prompt_text, code)
        x = torch.as_tensor(embed_fn(tr["task"]), dtype=torch.float32, device=dev)
        A = torch.as_tensor(tr["atoms"], dtype=torch.float32, device=dev)
        sp, _z = sptrm(x, A)                               # [K, d_model]
        p_ids = _prompt_ids(tok, tr["prompt"], dev)
        c_ids = tok(tr["code"] + tok.eos_token, return_tensors="pt",
                    add_special_tokens=False).input_ids.to(dev)
        ids = torch.cat([p_ids, c_ids], dim=1)[:, -1024:]
        emb = embed_layer(ids)                            # [1, L, d_model]
        full = torch.cat([sp.to(emb.dtype).unsqueeze(0), emb], dim=1)   # [1, K+L, d_model]
        logits = model(inputs_embeds=full).logits         # [1, K+L, vocab]
        n_c = min(c_ids.shape[1], ids.shape[1] - 1)
        pred = logits[:, K + ids.shape[1] - n_c - 1: K + ids.shape[1] - 1, :]
        loss = nn.functional.cross_entropy(pred.reshape(-1, logits.shape[-1]).float(),
                                           ids[:, -n_c:].reshape(-1))
        opt.zero_grad(); loss.backward(); opt.step()
        if step % 100 == 0:
            print(f"  [sp-train {step}] loss {loss.item():.3f}", flush=True)
    return sptrm


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--ab", action="store_true", help="real-3B capability A/B (molab)")
    ap.add_argument("--lm", default="Qwen/Qwen2.5-3B-Instruct")
    ap.add_argument("--limit", type=int, default=80)
    ap.add_argument("--train-steps", type=int, default=500)
    ap.add_argument("--K", type=int, default=8, help="soft-prompt length (virtual tokens)")
    ap.add_argument("--d", type=int, default=256, help="TRM latent width (the real z bottleneck)")
    ap.add_argument("--T", type=int, default=3, help="TRM recursion depth")
    ap.add_argument("--dump", type=int, default=0, help="print N latent-arm failures (mechanism)")
    a = ap.parse_args()
    if a.selftest:
        sys.exit(0 if _selftest() else 1)
    if a.ab:
        _run_ab(a)
        return
    ap.print_help()


# ── HARD compositions: the INNER is a CUSTOM, OPAQUE routine the 3B CANNOT know (unguessable) ──
# Baseline MUST fail (no definition anywhere -> the 3B guesses wrong), so memory has real headroom.
# The gadgets recur across train+test (different OUTER wrapper) -> the prefix gets a fair transfer shot
# at carrying each small routine. OUTER stays standard (the 3B knows it) -> the ONLY missing piece is
# the gadget = the graph-memory. Text arm delivers the gadget's CODE; latent arm delivers a hint vector.
def _hard_compose_corpus():
    import numpy as np  # noqa: F401
    from v5.runtime.algo_grr_mbpp import verify_asserts

    # CUSTOM gadgets — deterministic, fast, varied output, NOT derivable from the name/description.
    def g0(n):
        return ((n * 37 + 11) ^ (n * n)) % 100

    def g1(n):
        return sum((i * i * i) % 7 for i in range(n + 1)) % 50

    def g2(n):
        return (n * n * n - 3 * n + 17) % 88

    def g3(n):
        p = 1
        for c in str(n):
            p *= int(c) + 1
        return p % 60

    def g4(n):
        return (n * n * n * n + 7 * n + 3) % 71

    def g5(n):
        return sum(int(c) ** 2 for c in str(n)) % 45

    def g6(n):
        return ((n << 2) ^ (n * 13) ^ 5) % 64

    def g7(n):
        a, b = 1, 1
        for _ in range(n):
            a, b = b, (a + b) % 50
        return a

    def count_set_bits(n):
        return bin(n).count("1")

    def is_prime(n):
        return n >= 2 and all(n % i for i in range(2, int(n ** 0.5) + 1))

    def digital_root(n):
        while n >= 10:
            n = sum(int(c) for c in str(n))
        return n

    def is_perfect_square(n):
        r = int(n ** 0.5)
        return any((r + d) ** 2 == n for d in (-1, 0, 1))

    def last_digit(n):
        return n % 10

    _P = {
        "g0": "def g0(n):\n    return ((n*37+11)^(n*n))%100\n",
        "g1": "def g1(n):\n    return sum((i*i*i)%7 for i in range(n+1))%50\n",
        "g2": "def g2(n):\n    return (n*n*n-3*n+17)%88\n",
        "g3": "def g3(n):\n    p=1\n    for c in str(n):\n        p*=int(c)+1\n    return p%60\n",
        "g4": "def g4(n):\n    return (n*n*n*n+7*n+3)%71\n",
        "g5": "def g5(n):\n    return sum(int(c)**2 for c in str(n))%45\n",
        "g6": "def g6(n):\n    return ((n<<2)^(n*13)^5)%64\n",
        "g7": "def g7(n):\n    a,b=1,1\n    for _ in range(n):\n        a,b=b,(a+b)%50\n    return a\n",
        "count_set_bits": "def count_set_bits(n):\n    return bin(n).count('1')\n",
        "is_prime": "def is_prime(n):\n    return n>=2 and all(n%i for i in range(2,int(n**0.5)+1))\n",
        "digital_root": "def digital_root(n):\n    while n>=10:\n        n=sum(int(c) for c in str(n))\n    return n\n",
        "is_perfect_square": "def is_perfect_square(n):\n    r=int(n**0.5)\n    return any((r+d)**2==n for d in (-1,0,1))\n",
        "last_digit": "def last_digit(n):\n    return n%10\n",
    }
    # INNER: opaque custom routine — the description NAMES it but hides the formula (memory-only).
    _gd = "the output of internal routine {g} (a fixed proprietary integer transform) applied to n"
    INNER = {g: (fn, _gd.format(g=g)) for g, fn in
             [("g0", g0), ("g1", g1), ("g2", g2), ("g3", g3), ("g4", g4), ("g5", g5), ("g6", g6), ("g7", g7)]}
    # OUTER: standard, the 3B knows it — so the gadget is the ONLY missing piece.
    OUTER = {
        "is_prime": (is_prime, "whether {v} is prime"),
        "count_set_bits": (count_set_bits, "the number of 1-bits of {v}"),
        "digital_root": (digital_root, "the digital root of {v}"),
        "is_perfect_square": (is_perfect_square, "whether {v} is a perfect square"),
        "last_digit": (last_digit, "the last decimal digit of {v}"),
    }
    tasks = []
    k = 0
    for iname, (ifn, idesc) in INNER.items():
        for oname, (ofn, ophr) in OUTER.items():
            entry = f"h_{k:02d}"; k += 1
            ref = f"{_P[iname]}\n{_P[oname]}\ndef {entry}(n):\n    return {oname}({iname}(n))\n"
            asserts = []
            for n in (5, 9, 14, 23, 31, 40, 52, 63):
                try:
                    asserts.append(f"assert {entry}({n}) == {ofn(ifn(n))!r}")
                except Exception:  # noqa: BLE001
                    pass
            text = (f"Given a positive integer n, let x be {idesc}. Return {ophr.format(v='x')}. "
                    f"(g0..g7 are fixed internal routines — use the one named.)")
            atom_specs = [
                {"name": iname, "purpose": idesc, "code": _P[iname]},
                {"name": oname, "purpose": ophr.format(v="a value x"), "code": _P[oname]},
            ]

            # DROP degenerate tasks: if every asserted output is identical, a constant guess passes
            # without any memory (verify needs ALL asserts) -> the gadget wouldn't be load-bearing.
            if len({a.split("==", 1)[1].strip() for a in asserts}) < 2:
                k -= 1
                continue

            def mk(a=asserts):
                return lambda code: verify_asserts(code, a)
            tasks.append(dict(text=text, entry=entry, examples=asserts, verify_fn=mk(),
                              reference=ref, atom_texts=[idesc, ophr.format(v="a value x")],
                              atom_specs=atom_specs))
    return tasks


def _run_ab(a):
    """FAIR capability A/B (retest): MiniLM embeddings + GRAPH-MEMORY objective on HARD compositions over
    obscure primitives (where the frozen 3B fails alone). The TRM gets the GROUND-TRUTH needed-atom memory
    (best case for the latent). Train the soft prompt to inject that memory -> verified composition; test
    held-out: plain 3B (task text only) vs 3B + memory-soft-prompt."""
    import os
    import numpy as np
    import torch as _t
    os.environ["V5_HARD_VERIFY"] = "1"
    torch, nn, SoftPromptTRM, _StubLM = _build()
    from v5.runtime.algo_grr_membrane import (render_compile_prompt, _extract_code, strip_module_exec,
                                              _strip_redefs)
    from embedder import encode_one                        # real MiniLM (384-d) — the semantic embedder

    def embed(text):
        return encode_one(text)

    tasks = _hard_compose_corpus()
    import random
    random.Random(0).shuffle(tasks)
    half = len(tasks) // 2
    train_tasks, test_tasks = tasks[:half], tasks[half:]
    bundle = make_lm_bundle(a.lm)
    d_model = bundle[3]
    sptrm = SoftPromptTRM(d_in=384, d_model=d_model, d=a.d, K=a.K, T=a.T)
    _np = sum(p.numel() for p in sptrm.parameters())

    def _atom_vecs(t):
        return np.stack([embed(s) for s in t["atom_texts"]]).astype(np.float32)

    traces = []
    for t in train_tasks:
        spec = {"task_text": t["text"], "entry": t["entry"], "atoms": [], "examples": t["examples"]}
        traces.append({"task": t["text"], "atoms": _atom_vecs(t),
                       "prompt": render_compile_prompt(spec), "code": t["reference"]})
    print(f"[FAIR A/B] {len(tasks)} hard compositions; train {len(traces)}, test {len(test_tasks)}; "
          f"MiniLM embed, ground-truth atom memory; TRM K={a.K} d={a.d} T={a.T} ({_np/1e3:.1f}k params)\n")
    train_softprompt(bundle, sptrm, traces, embed, render_compile_prompt, steps=a.train_steps)

    _fails = []                                           # (gadget, entry, real_code, generated_code) latent misses

    def _gen_plain(prompt):
        ids = _prompt_ids(bundle[1], prompt, bundle[4])
        with _t.no_grad():
            out = bundle[0].generate(input_ids=ids, do_sample=True, temperature=0.6, top_p=0.95,
                                     max_new_tokens=256, pad_token_id=bundle[1].pad_token_id)
        return bundle[1].decode(out[0, ids.shape[1]:], skip_special_tokens=True)

    def _solve(task, arm):
        # arm: 'none' = task only; 'text' = atoms as TEXT (membrane spec, code prepended); 'latent' = memory soft-prompt
        atoms = task["atom_specs"] if arm == "text" else []
        spec = {"task_text": task["text"], "entry": task["entry"], "atoms": atoms, "examples": task["examples"]}
        prompt = render_compile_prompt(spec)
        if arm == "latent":
            code = softprompt_generate(bundle, sptrm, task["text"], _atom_vecs(task), embed, prompt)
        else:
            code = _gen_plain(prompt)
        code = strip_module_exec(_extract_code(code))
        if arm == "text":                                 # mirror make_lm_compiler: strip the LM's redefs of
            names = {a["name"] for a in task["atom_specs"]}   # provided atoms, THEN prepend the verified closure
            code = _strip_redefs(code, names)             # (else a wrong guessed g0 shadows the real gadget)
            closure = "\n\n".join(a["code"].rstrip("\n") for a in task["atom_specs"])
            code = closure + "\n\n" + code
        ok = task["verify_fn"](code)[0] >= 1.0
        if arm == "latent" and not ok:                    # capture HOW the latent reconstruction missed
            _fails.append((task["atom_specs"][0]["name"], task["entry"], task["atom_specs"][0]["code"], code))
        return ok

    none = sum(_solve(t, "none") for t in test_tasks)
    txt = sum(_solve(t, "text") for t in test_tasks)
    lat = sum(_solve(t, "latent") for t in test_tasks)
    n = len(test_tasks)
    print(f"\n[FAIR A/B RESULT] held-out {n} hard compositions (obscure-primitive pipelines):")
    print(f"  A. plain 3B, no memory      : {none}/{n} ({100*none//n}%)")
    print(f"  B. 3B + atoms as TEXT        : {txt}/{n} ({100*txt//n}%)   (membrane: verified code in prompt)")
    print(f"  C. 3B + memory soft-prompt   : {lat}/{n} ({100*lat//n}%)   (latent: same atoms via K virtual tokens)")
    print(f"  latent lift  (C-A) = {lat - none:+d}   ->  {'latent injects memory' if lat > none else 'latent adds nothing over no-memory'}")
    print(f"  text lift    (B-A) = {txt - none:+d}")
    print(f"  latent vs text (C-B) = {lat - txt:+d}   ->  "
          f"{'latent matches/beats text' if lat >= txt else 'TEXT delivers code the latent cannot (capacity limit) -> structural/membrane TRM is the answer'}")
    if a.dump and _fails:
        print(f"\n[LATENT FAILURES] {len(_fails)} misses — how the neural reconstruction broke "
              f"(real gadget vs what the soft-prompt made the frozen LM emit):")
        for gname, ent, real, gen in _fails[:a.dump]:
            print(f"\n  ── {ent} (needs {gname}) ──")
            print("  REAL gadget:\n    " + real.strip().replace("\n", "\n    "))
            print("  LM emitted (soft-prompt only):\n    " + gen.strip()[:500].replace("\n", "\n    "))


if __name__ == "__main__":
    main()
