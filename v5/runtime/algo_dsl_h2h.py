"""GRR-9b: gen-1 vs gen-2 HEAD-TO-HEAD on the hard factory benchmark — the fair fight the 6-family
suite couldn't referee ("gen-1 wasn't bad, the task was too easy": both saturate at 100% there).

  gen-2  ProgramDecoder (algo_dsl_trm): single-pass GRU + pointer — the current production decoder.
  gen-1  RecursiveProgramDecoder (HERE): the TRMReasoner refinement loop (algo_trm.py) ported into the
         SAME autoregressive interface (loss/sample/decode): before EACH emission, a scratchpad z is
         refined T times (soft-read the atoms -> update z -> re-query), THEN the heads fire. Deep
         supervision over the inner steps (the TRM recipe). Think-before-emit.

The benchmark (algo_dsl_gen): N families, canonical FILTER* -> MAP-chain -> REDUCE pipelines, chain
depth up to 4 = DEPENDENT sequential choices — the regime gen-1's own selftest predicted recursion is
for ("its value is DEPENDENT plans"). Both nets train on the SAME traces, same budget, multi-seed; we
report decode-solve per LENGTH BUCKET at training checkpoints (capacity + sample-efficiency for the
production job: amortizing many long discovered programs). Verdict comes from the table, not a prior.

  selftest (no LM, mechanism):  python -m v5.runtime.algo_dsl_h2h --selftest
  the real fight (molab mpnet): python -m v5.runtime.algo_dsl_h2h --h2h --families 32 --steps 6000
"""
from __future__ import annotations

import argparse
import sys

import numpy as np

from v5.runtime.algo_dsl import Op
from v5.runtime.algo_dsl_gen import (GEN_ATOMS, gen_families, gen_tasks, pipe_is_general, pipe_text,
                                     pipe_text_variants)
from v5.runtime.algo_dsl_trm import AGGS, OPS, _REDUCE, _sft_steps, program_to_steps


def _build_recursive():
    import torch
    import torch.nn as nn

    class RecursiveProgramDecoder(nn.Module):
        """gen-1's recursion in gen-2's slot. Same external contract as ProgramDecoder: loss(goal_vec,
        atom_vecs, steps) / sample(...) / decode(...). Internally, each emission runs T inner refinement
        iterations of a scratchpad z (soft atom read -> refine -> re-query, algo_trm.TRMReasoner's loop)
        before the op/agg/atom heads fire; a GRU carries the program-so-far exactly like gen-2."""

        def __init__(self, d_in=768, d=64, T=3):
            super().__init__()
            self.T = T
            self.goal_proj = nn.Linear(d_in, d)
            self.atom_proj = nn.Linear(d_in, d)
            self.z0 = nn.Parameter(torch.zeros(d))
            self.f = nn.Sequential(nn.Linear(4 * d, d), nn.GELU(), nn.Linear(d, d))    # refine z
            self.q = nn.Sequential(nn.Linear(3 * d, d), nn.GELU(), nn.Linear(d, d))    # atom query
            self.op_emb = nn.Embedding(len(OPS), d)
            self.agg_emb = nn.Embedding(len(AGGS), d)
            self.rnn = nn.GRUCell(d, d)
            self.op_head = nn.Sequential(nn.Linear(3 * d, d), nn.GELU(), nn.Linear(d, len(OPS)))
            self.agg_head = nn.Sequential(nn.Linear(3 * d, d), nn.GELU(), nn.Linear(d, len(AGGS)))
            self.d = d

        def _refine(self, g, h, A, return_all=False):
            """T inner iterations -> per-iteration (op_logits, agg_logits, atom_logits)."""
            import torch
            z = self.z0
            y = torch.zeros(A.shape[0])
            outs = []
            for _ in range(self.T):
                ysum = torch.softmax(y, dim=0) @ A                      # soft read of current pick
                z = self.f(torch.cat([g, ysum, z, h]))                  # refine the scratchpad
                y = A @ self.q(torch.cat([g, z, h])) / (self.d ** 0.5)  # re-point at the atoms
                ctx = torch.cat([g, z, h])
                outs.append((self.op_head(ctx), self.agg_head(ctx), y))
            return outs if return_all else outs[-1]

        def loss(self, goal_vec, atom_vecs, steps):
            """Teacher-forced CE with DEEP SUPERVISION: every inner refinement step is supervised (the
            TRM recipe), averaged, so early iterations learn to move toward the answer."""
            import torch
            import torch.nn as nn
            ce = nn.CrossEntropyLoss()
            g = self.goal_proj(goal_vec)
            A = self.atom_proj(atom_vecs)
            h = torch.zeros(self.d)
            total = torch.zeros(())
            for op_i, atom_i, agg_i in steps:
                outs = self._refine(g, h, A, return_all=True)
                step_loss = torch.zeros(())
                for op_l, agg_l, atom_l in outs:
                    step_loss = step_loss + ce(op_l.unsqueeze(0), torch.tensor([op_i]))
                    if op_i == _REDUCE:
                        step_loss = step_loss + ce(agg_l.unsqueeze(0), torch.tensor([agg_i]))
                    else:
                        step_loss = step_loss + ce(atom_l.unsqueeze(0), torch.tensor([atom_i]))
                total = total + step_loss / len(outs)
                if op_i == _REDUCE:
                    break
                arg = A[atom_i]
                h = self.rnn((self.op_emb(torch.tensor(op_i)) + arg).unsqueeze(0), h.unsqueeze(0))[0]
            return total

        def sample(self, goal_vec, atom_vecs, atom_names, temp=1.0, max_len=8):
            import torch
            g = self.goal_proj(goal_vec)
            A = self.atom_proj(atom_vecs)
            h = torch.zeros(self.d)
            pipe, logp = [], torch.zeros(())
            for _ in range(max_len):
                op_l, agg_l, atom_l = self._refine(g, h, A)
                op = torch.distributions.Categorical(logits=op_l / temp).sample()
                logp = logp + torch.log_softmax(op_l, -1)[op]
                if OPS[int(op)] == "REDUCE":
                    agg = torch.distributions.Categorical(logits=agg_l / temp).sample()
                    logp = logp + torch.log_softmax(agg_l, -1)[agg]
                    pipe.append(Op("REDUCE", AGGS[int(agg)])); break
                ai = torch.distributions.Categorical(logits=atom_l / temp).sample()
                logp = logp + torch.log_softmax(atom_l, -1)[ai]
                pipe.append(Op(OPS[int(op)], atom_names[int(ai)]))
                h = self.rnn((self.op_emb(op) + A[ai]).unsqueeze(0), h.unsqueeze(0))[0]
            return pipe, logp

        def decode(self, goal_vec, atom_vecs, atom_names, max_len=8):
            import torch
            with torch.no_grad():
                g = self.goal_proj(goal_vec)
                A = self.atom_proj(atom_vecs)
                h = torch.zeros(self.d)
                pipe = []
                for _ in range(max_len):
                    op_l, agg_l, atom_l = self._refine(g, h, A)
                    op = int(op_l.argmax())
                    if OPS[op] == "REDUCE":
                        pipe.append(Op("REDUCE", AGGS[int(agg_l.argmax())])); break
                    ai = int(atom_l.argmax())
                    pipe.append(Op(OPS[op], atom_names[ai]))
                    h = self.rnn((self.op_emb(torch.tensor(op)) + A[ai]).unsqueeze(0), h.unsqueeze(0))[0]
            return pipe

    return RecursiveProgramDecoder


# ═══════════════════════════════════════════════════════════════════════════════
# The head-to-head harness — same traces, same budget, multi-seed, per-length-bucket report.
# ═══════════════════════════════════════════════════════════════════════════════

def _build_vib(beta: float = 1e-2):
    """User objective: MINIMIZE I(z; task) while MAXIMIZING I(z; solution) — the information
    bottleneck, in its variational form (VIB): the goal ENCODER becomes stochastic
    z ~ N(mu(x), sigma(x)); the program-decode CE (already the training loss) lower-bounds
    I(z; solution); beta * KL(q(z|x) || N(0, I)) upper-bounds I(z; task) — bits about the task
    only flow through z if they pay for themselves in solution accuracy. Predicted effect: z sheds
    task-SURFACE (phrasing idiosyncrasies) -> better held-out-paraphrase generalization; the h2h
    paraphrase regime is the exact instrument that tests it."""
    import torch
    import torch.nn as nn
    from v5.runtime.algo_dsl_trm import _build as _build_gru
    _t, _nn, ProgramDecoder = _build_gru()

    class VIBProgramDecoder(ProgramDecoder):
        def __init__(self, d_in=768, d=64, beta=beta):
            super().__init__(d_in=d_in, d=d)
            self.logvar_head = nn.Linear(d_in, d)          # sigma(x): the bottleneck's width
            self.beta = beta

        def _encode(self, goal_vec, stochastic: bool):
            mu = self.goal_proj(goal_vec)
            if not stochastic:
                return mu, torch.zeros(())
            logvar = self.logvar_head(goal_vec).clamp(-8, 4)
            z = mu + torch.exp(0.5 * logvar) * torch.randn_like(mu)
            kl = 0.5 * (torch.exp(logvar) + mu ** 2 - 1.0 - logvar).sum()
            return z, kl

        def loss(self, goal_vec, atom_vecs, steps):
            # base CE computed with a SAMPLED z + the KL bits-tax (train-time only)
            ce = nn.CrossEntropyLoss()
            g, kl = self._encode(goal_vec, stochastic=True)
            A = self.atom_proj(atom_vecs)
            state = torch.zeros(self.d)
            total = torch.zeros(())
            from v5.runtime.algo_dsl_trm import _REDUCE
            for op_i, atom_i, agg_i in steps:
                op_l, agg_l, atom_l = self._heads(g, state, A)
                total = total + ce(op_l.unsqueeze(0), torch.tensor([op_i]))
                if op_i == _REDUCE:
                    total = total + ce(agg_l.unsqueeze(0), torch.tensor([agg_i]))
                    break
                total = total + ce(atom_l.unsqueeze(0), torch.tensor([atom_i]))
                state = self.rnn((self.op_emb(torch.tensor(op_i)) + A[atom_i]).unsqueeze(0),
                                 state.unsqueeze(0))[0]
            return total + self.beta * kl

        # decode/sample: deterministic mu (inference reads the bottleneck's center) — inherited
        # methods call goal_proj directly, which IS mu. Nothing to override.

    return VIBProgramDecoder


def _build_vq(K: int = 32, beta_commit: float = 0.25, resid_scale: float = 0.25):
    """GRR-17 row 5 (reviewer table): separate memorizing reusable CONCEPTS (good) from memorizing
    dataset INSTANCES (bad). Mechanism: the goal encodes against a learned CONCEPT CODEBOOK — z =
    prototype + a SHRUNK residual. Prototypes are free memory (concepts); instance detail only
    survives through the small residual channel. VQ-VAE straight-through + commitment loss; decode
    path = same heads, so inference is architecture-identical to gru reading a cleaner z."""
    import torch
    import torch.nn as nn
    from v5.runtime.algo_dsl_trm import _build as _build_gru
    _t, _nn, ProgramDecoder = _build_gru()

    class VQProgramDecoder(ProgramDecoder):
        def __init__(self, d_in=768, d=64, K=K):
            super().__init__(d_in=d_in, d=d)
            self.codebook = nn.Parameter(torch.randn(K, d) * 0.5)

        def _quantize(self, mu):
            dists = ((mu.unsqueeze(0) - self.codebook) ** 2).sum(-1)
            j = int(dists.argmin())
            proto = self.codebook[j]
            # straight-through: forward uses prototype + shrunk residual; grads flow to mu
            z = mu + (proto - mu).detach() + resid_scale * (mu - mu.detach())
            commit = ((mu - proto.detach()) ** 2).sum() + beta_commit * ((proto - mu.detach()) ** 2).sum()
            return z, commit

        def loss(self, goal_vec, atom_vecs, steps, usage_w: float = 0.0):
            ce = nn.CrossEntropyLoss()
            mu = self.goal_proj(goal_vec)
            g, commit = self._quantize(mu)
            A = self.atom_proj(atom_vecs)
            state = torch.zeros(self.d)
            total = torch.zeros(())
            from v5.runtime.algo_dsl_trm import _REDUCE
            for op_i, atom_i, agg_i in steps:
                op_l, agg_l, atom_l = self._heads(g, state, A)
                total = total + ce(op_l.unsqueeze(0), torch.tensor([op_i]))
                if op_i == _REDUCE:
                    total = total + ce(agg_l.unsqueeze(0), torch.tensor([agg_i]))
                    break
                total = total + ce(atom_l.unsqueeze(0), torch.tensor([atom_i]))
                state = self.rnn((self.op_emb(torch.tensor(op_i)) + A[atom_i]).unsqueeze(0),
                                 state.unsqueeze(0))[0]
            return total + 0.1 * commit

        def decode(self, goal_vec, atom_vecs, atom_names, max_len=8):
            # base decode loop, but reading the QUANTIZED goal (the concept, not the instance)
            import torch as _tt
            from v5.runtime.algo_dsl import Op as _Op
            from v5.runtime.algo_dsl_trm import AGGS as _AGGS, OPS as _OPS
            with _tt.no_grad():
                g, _ = self._quantize(self.goal_proj(goal_vec))
                A = self.atom_proj(atom_vecs)
                state = _tt.zeros(self.d)
                pipe = []
                for _ in range(max_len):
                    op_l, agg_l, atom_l = self._heads(g, state, A)
                    op = int(op_l.argmax())
                    if _OPS[op] == "REDUCE":
                        pipe.append(_Op("REDUCE", _AGGS[int(agg_l.argmax())])); break
                    ai = int(atom_l.argmax())
                    pipe.append(_Op(_OPS[op], atom_names[ai]))
                    state = self.rnn((self.op_emb(_tt.tensor(op)) + A[ai]).unsqueeze(0),
                                     state.unsqueeze(0))[0]
            return pipe

    return VQProgramDecoder


def _setup(fams, embed_fn):
    atom_names = list(GEN_ATOMS)
    atom_idx = {a: i for i, a in enumerate(atom_names)}
    vecs = embed_fn({a: GEN_ATOMS[a][0] for a in atom_names})
    atom_vecs = np.asarray([vecs[a] for a in atom_names], dtype=np.float32)
    return atom_names, atom_idx, atom_vecs


def _traces(fams, atom_idx, embed_fn, n_per=4, seed=0):
    out = []
    for t in gen_tasks(fams, n_per=n_per, seed=seed):
        gv = np.asarray(list(embed_fn({"q": t.text}).values())[0], dtype=np.float32)
        out.append((gv, program_to_steps(fams[t.name], atom_idx)))
    return out


def _bucket_solve(model, fams, atom_names, atom_vecs, embed_fn, max_len=8, texts_by_fam=None):
    """Decode each family's goal(s) -> pipe_is_general. texts_by_fam overrides the canonical text with a
    LIST of phrasings per family (the held-out-paraphrase eval). Returns {length_bucket: (solved, total)}."""
    import torch
    A = torch.as_tensor(atom_vecs, dtype=torch.float32)
    buckets = {}
    for fam, pipe in fams.items():
        texts = texts_by_fam[fam] if texts_by_fam else [pipe_text(pipe)]
        L = len(pipe)
        for text in texts:
            gv = np.asarray(list(embed_fn({"q": text}).values())[0], dtype=np.float32)
            got = model.decode(torch.as_tensor(gv), A, atom_names, max_len=max_len)
            ok = pipe_is_general(got, fams, fam)
            s, t = buckets.get(L, (0, 0))
            buckets[L] = (s + int(ok), t + 1)
    return buckets


def h2h(n_families=24, steps=4000, seeds=(1, 2, 3), d=64, T=3, embed_fn=None, fam_seed=0, log=True,
        para=None):
    """Train BOTH decoders on the same factory traces; report per-length-bucket decode-solve at
    checkpoints. para=(k_train, k_eval) switches on the PARAPHRASE regime: each family gets
    k_train+k_eval distinct phrasings; training sees only the first k_train, eval decodes the HELD-OUT
    phrasings (generalization, not point-recall — with one fixed text per family mpnet gives one
    deterministic embedding and any net just memorizes 32 points: measured 100% everywhere).
    Returns {arch: {seed: {checkpoint: buckets}}} (buckets = held-out eval when para is set)."""
    import torch
    from v5.runtime.algo_dsl_trm import _build as _build_gru
    RecursiveProgramDecoder = _build_recursive()
    _torch, _nn, ProgramDecoder = _build_gru()
    fams = gen_families(n_families, seed=fam_seed)
    if embed_fn is None:
        from v5.memory.store import make_mpnet_embedder
        embed_fn = make_mpnet_embedder()
    atom_names, atom_idx, atom_vecs = _setup(fams, embed_fn)
    A = torch.as_tensor(atom_vecs, dtype=torch.float32)
    d_in = atom_vecs.shape[1]
    train_texts = eval_texts = None
    if para:
        k_train, k_eval = para
        allv = {f: pipe_text_variants(p, k_train + k_eval) for f, p in fams.items()}
        train_texts = {f: v[:k_train] for f, v in allv.items()}
        eval_texts = {f: v[k_train:] for f, v in allv.items()}
    VIBProgramDecoder = _build_vib()
    VQProgramDecoder = _build_vq()
    archs = {"gru": lambda: ProgramDecoder(d_in=d_in, d=d),
             "recursive": lambda: RecursiveProgramDecoder(d_in=d_in, d=d, T=T),
             "vib": lambda: VIBProgramDecoder(d_in=d_in, d=d),
             "vq": lambda: VQProgramDecoder(d_in=d_in, d=d)}
    if log:
        for name, mk in archs.items():
            torch.manual_seed(0)
            print(f"  {name}: {sum(p.numel() for p in mk().parameters())} params", flush=True)
        if para:
            print(f"  paraphrase regime: train on {para[0]} phrasings/family, eval on {para[1]} "
                  f"HELD-OUT phrasings/family", flush=True)
    results = {a: {} for a in archs}
    half = steps // 2
    for arch, mk in archs.items():
        for sd in seeds:
            torch.manual_seed(sd)
            model = mk()
            opt = torch.optim.Adam(model.parameters(), lr=1e-3)
            if para:
                traces = [(np.asarray(list(embed_fn({"q": t}).values())[0], dtype=np.float32),
                           program_to_steps(fams[f], atom_idx))
                          for f in fams for t in train_texts[f]]
            else:
                traces = _traces(fams, atom_idx, embed_fn, seed=sd)
            _sft_steps(model, opt, traces, A, half, seed=sd)
            b_half = _bucket_solve(model, fams, atom_names, atom_vecs, embed_fn, texts_by_fam=eval_texts)
            _sft_steps(model, opt, traces, A, steps - half, seed=sd + 100)
            b_full = _bucket_solve(model, fams, atom_names, atom_vecs, embed_fn, texts_by_fam=eval_texts)
            results[arch][sd] = {half: b_half, steps: b_full}
            if log:
                tot = lambda b: (sum(s for s, _ in b.values()), sum(t for _, t in b.values()))
                s1, t1 = tot(b_half); s2, t2 = tot(b_full)
                print(f"  {arch:10s} seed {sd}: @{half} {s1}/{t1}  @{steps} {s2}/{t2}  "
                      f"by-len@{steps} {{{', '.join(f'{k}:{v[0]}/{v[1]}' for k, v in sorted(b_full.items()))}}}",
                      flush=True)
    if log:
        tag = "held-out PHRASINGS" if para else "canonical text (recall)"
        print(f"\n  === VERDICT (mean solve over seeds, final checkpoint, {tag}) ===", flush=True)
        for arch in archs:
            per_len = {}
            for sd in seeds:
                for L, (s, t) in results[arch][sd][steps].items():
                    a, b = per_len.get(L, (0, 0))
                    per_len[L] = (a + s, b + t)
            overall = sum(s for s, _ in per_len.values()) / max(1, sum(t for _, t in per_len.values()))
            print(f"  {arch:10s} overall {overall:.0%} | "
                  f"{', '.join(f'len{L} {s}/{t}' for L, (s, t) in sorted(per_len.items()))}", flush=True)
    return results


# ═══════════════════════════════════════════════════════════════════════════════
# SELFTEST — mechanism only: both archs expose the same contract, both TRAIN on the hard benchmark,
# recursion's inner loop demonstrably runs. The VERDICT is the molab table, not an assert.
# ═══════════════════════════════════════════════════════════════════════════════

def _selftest() -> bool:
    import torch
    print("algo_dsl_h2h --selftest: gen-1 recursive decoder vs gen-2 GRU — mechanism + a small fight\n")
    rng = np.random.default_rng(0)
    d_in = 64
    fams = gen_families(12, seed=3, max_chain=3)
    base = {f: rng.standard_normal(d_in).astype("float32") for f in fams}
    # any phrasing of a family -> that family's base + noise (a stub of what mpnet does with paraphrases)
    texts = {t: f for f, p in fams.items() for t in pipe_text_variants(p, 6)}

    def embed(d):
        out = {}
        for k, text in d.items():
            f = texts.get(text)
            out[k] = ((base[f] if f in base else 0.05 * rng.standard_normal(d_in))
                      + 0.15 * rng.standard_normal(d_in)).astype("float32")
        return out

    # [1] contract: recursive decoder trains + decodes a syntactically valid program
    RecursiveProgramDecoder = _build_recursive()
    torch.manual_seed(0)
    m = RecursiveProgramDecoder(d_in=d_in, d=48, T=3)
    atom_names, atom_idx, atom_vecs = _setup(fams, embed)
    A = torch.as_tensor(atom_vecs, dtype=torch.float32)
    tr = _traces(fams, atom_idx, embed, n_per=2, seed=0)
    l0 = float(m.loss(torch.as_tensor(tr[0][0]), A, tr[0][1]))
    opt = torch.optim.Adam(m.parameters(), lr=1e-3)
    _sft_steps(m, opt, tr, A, 600, seed=0)
    l1 = float(m.loss(torch.as_tensor(tr[0][0]), A, tr[0][1]))
    pipe = m.decode(torch.as_tensor(tr[0][0]), A, atom_names)
    assert l1 < l0 * 0.5 and pipe and pipe[-1].kind == "REDUCE", (l0, l1, pipe)
    print(f"  [1] recursive decoder: loss {l0:.2f} -> {l1:.2f}, decodes valid program "
          f"{[(o.kind, o.arg) for o in pipe]} -> PASS")

    # [2] the inner loop is live: T=3 refinement iterations produce DIFFERENT logits per iteration
    outs = m._refine(m.goal_proj(torch.as_tensor(tr[0][0])), torch.zeros(m.d), m.atom_proj(A),
                     return_all=True)
    a0, a2 = outs[0][2], outs[-1][2]
    assert len(outs) == 3 and float((a0 - a2).abs().max()) > 1e-4
    print(f"  [2] recursion live: 3 inner iterations, atom logits move between them -> PASS")

    # [2b] VIB (the min I(z;task) / max I(z;solution) bottleneck): trains (CE+KL falls, KL finite),
    #      decodes a valid program deterministically from mu
    VIB = _build_vib(beta=1e-2)
    torch.manual_seed(1)
    mv = VIB(d_in=d_in, d=48)
    lv0 = float(mv.loss(torch.as_tensor(tr[0][0]), A, tr[0][1]))
    optv = torch.optim.Adam(mv.parameters(), lr=1e-3)
    _sft_steps(mv, optv, tr, A, 500, seed=1)
    lv1 = float(mv.loss(torch.as_tensor(tr[0][0]), A, tr[0][1]))
    pv = mv.decode(torch.as_tensor(tr[0][0]), A, atom_names)
    assert lv1 < lv0 and pv and pv[-1].kind == "REDUCE", (lv0, lv1, pv)
    print(f"  [2b] VIB bottleneck: CE+beta*KL {lv0:.2f} -> {lv1:.2f}, deterministic decode valid "
          f"(min I(z;task), max I(z;solution)) -> PASS")

    # [2c] VQ concept-bottleneck: trains (CE+commit falls), decode-through-prototype yields a valid
    #      program (concepts free, instance detail shrunk)
    VQ = _build_vq(K=16)
    torch.manual_seed(2)
    mq = VQ(d_in=d_in, d=48)
    lq0 = float(mq.loss(torch.as_tensor(tr[0][0]), A, tr[0][1]))
    optq = torch.optim.Adam(mq.parameters(), lr=1e-3)
    _sft_steps(mq, optq, tr, A, 500, seed=2)
    lq1 = float(mq.loss(torch.as_tensor(tr[0][0]), A, tr[0][1]))
    pq = mq.decode(torch.as_tensor(tr[0][0]), A, atom_names)
    assert lq1 < lq0 and pq and pq[-1].kind == "REDUCE", (lq0, lq1, pq)
    print(f"  [2c] VQ concept bottleneck: CE+commit {lq0:.2f} -> {lq1:.2f}, decode-through-prototype "
          f"valid (concepts memorize free, instances pay) -> PASS")

    # [3] a small head-to-head runs end-to-end; both archs learn SOMETHING (>25% at this starved
    #     budget) and the benchmark DISCRIMINATES (nobody saturates — unlike the 6-family suite)
    res = h2h(n_families=12, steps=1600, seeds=(1,), d=48, T=3, embed_fn=embed, fam_seed=3, log=True)
    for arch in ("gru", "recursive"):
        b = res[arch][1][1600]
        solved = sum(s for s, _ in b.values()); total = sum(t for _, t in b.values())
        assert 0.25 < solved / total < 1.0, (arch, b)
    print(f"  [3] h2h runs end-to-end; both learn, NEITHER saturates (the benchmark finally "
          f"discriminates) -> PASS")

    # [4] paraphrase regime smoke: train on 2 phrasings/family, eval on 1 HELD-OUT phrasing —
    #     the anti-point-memorization eval (mpnet 32-fam recall saturated at 100% without it)
    resp = h2h(n_families=12, steps=800, seeds=(1,), d=48, T=3, embed_fn=embed, fam_seed=3, log=False,
               para=(2, 1))
    for arch in ("gru", "recursive"):
        b = resp[arch][1][800]
        assert sum(t for _, t in b.values()) == 12, b          # 1 held-out phrasing per family evaluated
    print(f"  [4] paraphrase regime: trains on seen phrasings, evals HELD-OUT phrasings "
          f"(gru {sum(s for s,_ in resp['gru'][1][800].values())}/12, "
          f"recursive {sum(s for s,_ in resp['recursive'][1][800].values())}/12 at smoke budget) -> PASS")

    print("\n  ALGO_DSL_H2H SELFTEST -> PASS  (mechanism proven; the VERDICT is the molab mpnet table)")
    return True


def main():
    ap = argparse.ArgumentParser(description="GRR-9b: GRU vs recursive-TRM decoder, hard factory benchmark.")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--h2h", action="store_true", help="the real fight with mpnet (molab)")
    ap.add_argument("--families", type=int, default=24)
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--d", type=int, default=64)
    ap.add_argument("--T", type=int, default=3)
    ap.add_argument("--paraphrase", action="store_true",
                    help="train on --para-train phrasings/family, eval HELD-OUT phrasings (the real test)")
    ap.add_argument("--para-train", type=int, default=3)
    ap.add_argument("--para-eval", type=int, default=2)
    a = ap.parse_args()
    if a.selftest:
        sys.exit(0 if _selftest() else 1)
    if a.h2h:
        para = (a.para_train, a.para_eval) if a.paraphrase else None
        print(f"h2h (real mpnet): families={a.families} steps={a.steps} seeds={a.seeds} d={a.d} T={a.T} "
              f"paraphrase={para}", flush=True)
        h2h(n_families=a.families, steps=a.steps, seeds=tuple(range(1, a.seeds + 1)), d=a.d, T=a.T,
            para=para)
        return
    ap.print_help()


if __name__ == "__main__":
    main()
