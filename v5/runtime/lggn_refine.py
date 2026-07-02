"""LGGN reasoning test — HRM-style latent refinement, done right.

The thesis (HRM + graph, per LGGN_DESIGN 4c/4d, the 2026-07-01 correction):
  - reason in a latent state h_t that lives in QWEN's own space (NOT mpnet -- mpnet is a foreign, lossy
    sentence embedder; flip_decode's strawman);
  - refine h_t over K steps (HRM recurrence), each step a MOVE = a graph operator (a latent displacement
    direction discovered in Qwen space);
  - iterate AGAINST THE CODE -- the code-token hidden states are the constraint environment (HRM's
    "grid"). iterative_refine.py showed refinement on a STATIC goal is FLAT because a fixed input has no
    constraints to propagate; giving h_t something to attend to each step is the missing piece.
  - measure whether the latent REACHES the gold-fix representation. This is the REASONING metric,
    decoupled from the leaf decode (a separate wall). resolve is downstream.

  h_0 = goal (Qwen hidden of the issue)
  step k:  a = CrossAttn(h_k, code_ctx)          # attend to the constraints (the environment)
           Delta = Sum softmax(policy(h_k+a)) . operator_dirs   # a graph move
           h_{k+1} = h_k + gc.a + go.Delta
  readout h_K ; loss/metric = cos(h_K, gold_fix)                # all in Qwen space

Ablations = the kill-test (each config trains its own refiner, no-leakage: ops + basis from TRAIN):
  raw_g        cos(g, f)                       floor (no refinement)
  direct       K1, no code, no graph           a single free update (non-iterative baseline ~ latent_compose A)
  K1_full / K4_full                            RECURRENCE value (does K>1 help? HRM's contested claim)
  K4_nocode                                    CONSTRAINT value (does attending to code unflatten iteration?)
  K4_nograph                                   GRAPH value (ops vs a free MLP update of equal capacity)
  K4_randops                                   operator LEARNEDNESS (real dirs vs random, V7-flavored)

No leaf, no Docker. Real reprs need Qwen (molab); reprs are cached to artifacts/lggn_reprs_*.npz.

  V5_LM_TRUST_REMOTE_CODE=1 V5_LM_QUANT=4bit python -m v5.runtime.lggn_refine --model Qwen/Qwen3.5-4B --dataset lite --n 200
  python -m v5.runtime.lggn_refine --selftest
"""
from __future__ import annotations

import argparse


def _reprs(model_name, dataset, split, n, layer_frac=0.6, t_ctx=48, cache="artifacts/lggn_reprs.npz"):
    """Qwen-native reprs per instance: g (issue hidden), ctx (code-token hiddens = the environment),
    f (gold-fix hidden). Cached — the only GPU cost."""
    import os, numpy as np, torch
    from pathlib import Path
    key = f"{cache[:-4]}_{model_name.split('/')[-1]}_{dataset}_{split}_n{n}_L{layer_frac}_T{t_ctx}.npz"
    if Path(key).exists():
        d = np.load(key); print(f"  cached reprs <- {key}")
        return d["g"], d["f"], d["ctx"], d["cmask"]
    from transformers import AutoTokenizer
    from v5.lm_loader import load_frozen_lm
    from v5.graph_grower.swe_load import load_instances
    from v5.runtime.operator_discovery import extract_hunks, _fix_text
    trust = os.environ.get("V5_LM_TRUST_REMOTE_CODE", "0").lower() in ("1", "true", "yes")
    tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=trust)
    lm = load_frozen_lm(model_name); lm.eval()
    dev = next(lm.parameters()).device
    L = max(1, int(layer_frac * lm.config.num_hidden_layers))

    @torch.no_grad()
    def hid(text, seq=False):
        enc = tok(text[:2000], return_tensors="pt", truncation=True, max_length=256).to(dev)
        hs = lm(**enc, output_hidden_states=True).hidden_states[L][0]     # [T, d]
        return hs.float().cpu().numpy() if seq else hs[-1].float().cpu().numpy()

    G, F, CTX, MASK = [], [], [], []
    for i in load_instances(name=dataset, split=split, limit=n):
        hs = extract_hunks(i.get("patch", "") or "")
        if not hs:
            continue
        removed = "\n".join(l for _f, r, _a in hs for l in r)
        fx = _fix_text(hs)
        if not fx.strip() or not removed.strip():
            continue
        G.append(hid(i.get("problem_statement") or ""))
        F.append(hid(fx))
        c = hid("Buggy code:\n" + removed, seq=True)                      # [Tc, d] the constraint set
        idx = np.linspace(0, len(c) - 1, min(t_ctx, len(c))).astype(int)
        c = c[idx]
        pad = np.zeros((t_ctx - len(c), c.shape[1]), c.dtype)
        CTX.append(np.concatenate([c, pad], 0)); MASK.append([1] * len(idx) + [0] * (t_ctx - len(idx)))
    g, f, ctx, cmask = (np.asarray(G), np.asarray(F), np.asarray(CTX), np.asarray(MASK, dtype=bool))
    Path(key).parent.mkdir(parents=True, exist_ok=True)
    np.savez(key, g=g, f=f, ctx=ctx, cmask=cmask); print(f"  reprs -> {key}  ({len(g)} inst, d={g.shape[1]}, L={L})")
    return g, f, ctx, cmask


class Refiner:
    import torch.nn as _nn

    class Net(_nn.Module):
        def __init__(s, d, r=256, n_op=24):
            import torch.nn as nn
            super().__init__()
            s.Wq = nn.Linear(d, r, bias=False); s.Wk = nn.Linear(d, r, bias=False)   # code cross-attn
            s.Wo = nn.Linear(d, r, bias=False); s.Wko = nn.Linear(d, r, bias=False)  # operator policy
            s.A = nn.Linear(d, r); s.B = nn.Linear(r, d)                              # free update (no-graph)
            import torch
            s.gc = nn.Parameter(torch.tensor(0.3)); s.go = nn.Parameter(torch.tensor(0.3))
            s.r = r

        def forward(s, g, ctx, cmask, ops, K, use_code, use_graph):
            import torch, math
            h = g
            for _ in range(K):
                if use_code:
                    q = s.Wq(h); k = s.Wk(ctx)                                        # [N,r],[N,T,r]
                    sc = (q.unsqueeze(1) * k).sum(-1) / math.sqrt(s.r)                # [N,T]
                    sc = sc.masked_fill(~cmask, -1e9); w = torch.softmax(sc, -1)
                    a = (w.unsqueeze(-1) * ctx).sum(1)                                # [N,d]
                else:
                    a = torch.zeros_like(h)
                base = h + a
                if use_graph:
                    logit = s.Wo(base) @ s.Wko(ops).t() / math.sqrt(s.r)             # [N,n_op]
                    delta = torch.softmax(logit, -1) @ ops                            # [N,d] a graph move
                else:
                    delta = s.B(torch.relu(s.A(base)))                               # free update, ops-free
                h = h + s.gc * a + s.go * delta
            return h

    @staticmethod
    def train_eval(g, f, ctx, cmask, ops, tr, he, K, use_code, use_graph, epochs=400, seed=0, log=print):
        import torch
        torch.manual_seed(seed)
        d = g.shape[1]; net = Refiner.Net(d, n_op=ops.shape[0])
        T = lambda x: torch.as_tensor(x, dtype=torch.float32)
        g, f, ctx, ops = T(g), T(f), T(ctx), T(ops); cm = torch.as_tensor(cmask)
        opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-4)
        Fn = torch.nn.functional.cosine_similarity
        for _ in range(epochs):
            net.train(); opt.zero_grad()
            h = net(g[tr], ctx[tr], cm[tr], ops, K, use_code, use_graph)
            loss = (1 - Fn(h, f[tr])).mean(); loss.backward(); opt.step()
        net.eval()
        with torch.no_grad():
            h = net(g[he], ctx[he], cm[he], ops, K, use_code, use_graph)
            return float(Fn(h, f[he]).mean())


def _discover_ops(disp_tr, n_op, seed=0):
    """Operators = displacement directions (f-g) clustered in QWEN space, TRAIN only (no leakage)."""
    import numpy as np
    from sklearn.cluster import KMeans
    n_op = min(n_op, max(2, len(disp_tr) // 2))
    return KMeans(n_op, n_init=10, random_state=seed).fit(disp_tr).cluster_centers_.astype("float32")


def run(g, f, ctx, cmask, n_op=24, epochs=400, seed=0, ops_override=None, log=print):
    import numpy as np
    rng = np.random.RandomState(seed); idx = rng.permutation(len(g)); nh = max(2, len(idx) // 5)
    he, tr = idx[:nh], idx[nh:]
    ops = ops_override.astype("float32") if ops_override is not None else _discover_ops((f - g)[tr], n_op, seed)
    rand = rng.randn(*ops.shape).astype("float32"); rand *= np.linalg.norm(ops, axis=1, keepdims=True) / (np.linalg.norm(rand, axis=1, keepdims=True) + 1e-9)
    import torch
    cos = torch.nn.functional.cosine_similarity
    raw = float(cos(torch.tensor(g[he]), torch.tensor(f[he])).mean())
    E = lambda **kw: Refiner.train_eval(g, f, ctx, cmask, ops, tr, he, epochs=epochs, seed=seed, log=log, **kw)
    log(f"  train {len(tr)} / held {len(he)}; ops={ops.shape[0]} (Qwen-space displacement clusters)")
    out = {"raw_g": raw,
           "direct": E(K=1, use_code=False, use_graph=False),
           "K1_full": E(K=1, use_code=True, use_graph=True),
           "K4_full": E(K=4, use_code=True, use_graph=True),
           "K4_nocode": E(K=4, use_code=False, use_graph=True),
           "K4_nograph": E(K=4, use_code=True, use_graph=False),
           "K4_randops": Refiner.train_eval(g, f, ctx, cmask, rand, tr, he, K=4, use_code=True, use_graph=True, epochs=epochs, seed=seed, log=log)}
    return out, len(he)


def _report(out, n):
    print(f"\n=== LGGN REFINE (n={n}) — latent reaches the gold fix, in Qwen space ===")
    for k in ("raw_g", "direct", "K1_full", "K4_full", "K4_nocode", "K4_nograph", "K4_randops"):
        print(f"  {k:12}: held cos(h_K, gold-fix) = {out[k]:.3f}")
    rec = out["K4_full"] - out["K1_full"]
    con = out["K4_full"] - out["K4_nocode"]
    grp = out["K4_full"] - out["K4_nograph"]
    lrn = out["K4_full"] - out["K4_randops"]
    base = out["K4_full"] - out["direct"]
    print(f"\n  recurrence  K4-K1        = {rec:+.3f}  -> {'HRM recurrence helps' if rec > 0.02 else 'recurrence flat'}")
    print(f"  constraints K4-nocode     = {con:+.3f}  -> {'attending to CODE unflattens iteration' if con > 0.02 else 'code attention adds nothing (still flat)'}")
    print(f"  graph       K4-nograph    = {grp:+.3f}  -> {'operators beat a free update' if grp > 0.02 else 'graph decorative (free MLP ties it)'}")
    print(f"  learnedness K4-randops    = {lrn:+.3f}  -> {'real operator dirs matter' if lrn > 0.02 else 'random dirs tie (ops not load-bearing)'}")
    print(f"  vs non-iter K4-direct     = {base:+.3f}")
    ok = rec > 0.02 and con > 0.02 and grp > 0.02 and lrn > 0.02
    print(f"\n  verdict: {'LGGN STRUCTURE is load-bearing (recurrence + constraints + graph + learned ops all pay) -> build the outer test-feedback loop' if ok else 'at least one pillar is flat -> that pillar is decorative here (honest negative); read the deltas'}")


def _synth(d=32, N=600, n_op=6, T=8, chain=3, seed=0):
    """PATH-DEPENDENT fix (so recurrence is genuinely needed): a chain s0 -> s1=next(s0) -> s2=next(s1),
    f = g + ops[s0]+ops[s1]+ops[s2]. Only s0 is revealed (a CUE in ctx, distinct from the op vector so the
    move must come from the graph BASIS, not ctx). Steps 2..K must infer the transition from state ->
    needs K>1 (recurrence), the code cue (constraints), and the real op basis (learnedness)."""
    import numpy as np
    rng = np.random.RandomState(seed)
    ops = rng.randn(n_op, d).astype("float32"); ops /= np.linalg.norm(ops, axis=1, keepdims=True)
    cue = rng.randn(n_op, d).astype("float32"); cue /= np.linalg.norm(cue, axis=1, keepdims=True)  # distinct from ops
    g = 0.1 * rng.randn(N, d).astype("float32")
    ctx = 0.3 * rng.randn(N, T, d).astype("float32")
    f = g.copy()
    for i in range(N):
        s = rng.randint(n_op)
        ctx[i, 0] = cue[s] + 0.1 * rng.randn(d)            # only the START is revealed
        for _ in range(chain):
            f[i] += ops[s]; s = (s + 1) % n_op             # transition = the graph topology to be learned
    return g, f.astype("float32"), ctx, np.ones((N, T), bool), ops


def _selftest() -> bool:
    print("lggn_refine --selftest: fix = g + ops revealed by ctx -> full(K>1,code,graph) must beat "
          "K1 / no-code / random-ops\n")
    g, f, ctx, cmask, ops = _synth()
    out, n = run(g, f, ctx, cmask, ops_override=ops, epochs=400, log=lambda *a: None)
    print(f"  raw={out['raw_g']:.2f} direct={out['direct']:.2f} K1={out['K1_full']:.2f} "
          f"K4={out['K4_full']:.2f} nocode={out['K4_nocode']:.2f} randops={out['K4_randops']:.2f}")
    assert out["K4_full"] > out["K1_full"] + 0.03, "recurrence must help when the fix needs multiple moves"
    assert out["K4_full"] > out["K4_nocode"] + 0.03, "attending to code (constraints) must help"
    assert out["K4_full"] > out["K4_randops"] + 0.03, "real operator dirs must beat random"
    print("\n  LGGN-REFINE SELFTEST -> PASS (harness detects recurrence + constraint + operator value)")
    return True


def main():
    ap = argparse.ArgumentParser(description="LGGN: HRM latent refinement over graph operators, in Qwen space.")
    ap.add_argument("--model", default="Qwen/Qwen3.5-4B")
    ap.add_argument("--dataset", default="lite")
    ap.add_argument("--split", default="test")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--n-op", type=int, default=24)
    ap.add_argument("--layer-frac", type=float, default=0.6)
    ap.add_argument("--epochs", type=int, default=400)
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        raise SystemExit(0 if _selftest() else 1)
    print(f"[lggn-refine] model={a.model} dataset={a.dataset} n={a.n}")
    g, f, ctx, cmask = _reprs(a.model, a.dataset, a.split, a.n, layer_frac=a.layer_frac)
    print(f"  {len(g)} instances, Qwen dim={g.shape[1]}")
    _report(*run(g, f, ctx, cmask, n_op=a.n_op, epochs=a.epochs))


if __name__ == "__main__":
    main()
