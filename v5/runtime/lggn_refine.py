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

Extended tests (--extended): the graph's OTHER levers beyond per-step discrete selection:
  K4_topo      topology prior: op→op transition bias (does sequencing structure help?)
  K4_hybrid    graph prior + free residual (does the graph ADD value when combined with free MLP?)
  K4_ops_half  compounding: ops from 50% train data (does more experience → better ops?)

No leaf, no Docker. Real reprs need Qwen (molab); reprs are cached to artifacts/lggn_reprs_*.npz.

  V5_LM_TRUST_REMOTE_CODE=1 V5_LM_QUANT=4bit python -m v5.runtime.lggn_refine --model Qwen/Qwen3.5-4B --dataset lite --n 200
  V5_LM_TRUST_REMOTE_CODE=1 V5_LM_QUANT=4bit python -m v5.runtime.lggn_refine --model Qwen/Qwen3.5-4B --dataset lite --n 200 --extended --r 512
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
    instances = load_instances(name=dataset, split=split, limit=n)
    print(f"  extracting reprs from {len(instances)} raw instances (3 Qwen fwd each, layer {L})...")
    for ii, i in enumerate(instances):
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
        if (ii + 1) % 10 == 0 or ii == len(instances) - 1:
            print(f"    {ii+1}/{len(instances)} instances processed, {len(G)} kept", flush=True)
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
    def train_eval(g, f, ctx, cmask, ops, tr, he, K, use_code, use_graph, epochs=400, seed=0, r=256, log=print):
        import torch
        torch.manual_seed(seed)
        d = g.shape[1]; net = Refiner.Net(d, r=r, n_op=ops.shape[0])
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


    class TopologyNet(_nn.Module):
        """Net + learned op→op transition prior. Tests: does sequencing structure help?"""
        def __init__(s, d, r=256, n_op=24):
            import torch.nn as nn, torch
            super().__init__()
            s.Wq = nn.Linear(d, r, bias=False); s.Wk = nn.Linear(d, r, bias=False)
            s.Wo = nn.Linear(d, r, bias=False); s.Wko = nn.Linear(d, r, bias=False)
            s.gc = nn.Parameter(torch.tensor(0.3)); s.go = nn.Parameter(torch.tensor(0.3))
            s.trans = nn.Parameter(0.01 * torch.randn(n_op, n_op))
            s.r = r

        def forward(s, g, ctx, cmask, ops, K):
            import torch, math
            h = g; prev_w = None
            for _ in range(K):
                q = s.Wq(h); k = s.Wk(ctx)
                sc = (q.unsqueeze(1) * k).sum(-1) / math.sqrt(s.r)
                sc = sc.masked_fill(~cmask, -1e9)
                a = (torch.softmax(sc, -1).unsqueeze(-1) * ctx).sum(1)
                base = h + a
                logit = s.Wo(base) @ s.Wko(ops).t() / math.sqrt(s.r)
                if prev_w is not None:
                    logit = logit + prev_w @ s.trans
                op_w = torch.softmax(logit, -1); prev_w = op_w
                h = h + s.gc * a + s.go * (op_w @ ops)
            return h

    class HybridNet(_nn.Module):
        """Graph prior + free residual. Tests: does the graph ADD value when not the sole mechanism?"""
        def __init__(s, d, r=256, n_op=24):
            import torch.nn as nn, torch
            super().__init__()
            s.Wq = nn.Linear(d, r, bias=False); s.Wk = nn.Linear(d, r, bias=False)
            s.Wo = nn.Linear(d, r, bias=False); s.Wko = nn.Linear(d, r, bias=False)
            s.A = nn.Linear(d, r); s.B = nn.Linear(r, d)
            s.gc = nn.Parameter(torch.tensor(0.3))
            s.go_g = nn.Parameter(torch.tensor(0.15)); s.go_f = nn.Parameter(torch.tensor(0.15))
            s.r = r

        def forward(s, g, ctx, cmask, ops, K):
            import torch, math
            h = g
            for _ in range(K):
                q = s.Wq(h); k = s.Wk(ctx)
                sc = (q.unsqueeze(1) * k).sum(-1) / math.sqrt(s.r)
                sc = sc.masked_fill(~cmask, -1e9)
                a = (torch.softmax(sc, -1).unsqueeze(-1) * ctx).sum(1)
                base = h + a
                delta_g = torch.softmax(s.Wo(base) @ s.Wko(ops).t() / math.sqrt(s.r), -1) @ ops
                delta_f = s.B(torch.relu(s.A(base)))
                h = h + s.gc * a + s.go_g * delta_g + s.go_f * delta_f
            return h

    @staticmethod
    def _train_net(cls, g, f, ctx, cmask, ops, tr, he, K, epochs=400, seed=0, r=256):
        """Train+eval any net with forward(g, ctx, cmask, ops, K)."""
        import torch
        torch.manual_seed(seed)
        d = g.shape[1]; net = cls(d, r=r, n_op=ops.shape[0])
        T = lambda x: torch.as_tensor(x, dtype=torch.float32)
        gt, ft, ct, ot = T(g), T(f), T(ctx), T(ops); cm = torch.as_tensor(cmask)
        opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-4)
        Fn = torch.nn.functional.cosine_similarity
        for _ in range(epochs):
            net.train(); opt.zero_grad()
            h = net(gt[tr], ct[tr], cm[tr], ot, K)
            loss = (1 - Fn(h, ft[tr])).mean(); loss.backward(); opt.step()
        net.eval()
        with torch.no_grad():
            return float(Fn(net(gt[he], ct[he], cm[he], ot, K), ft[he]).mean())


def _discover_ops(disp_tr, n_op, seed=0):
    """Operators = displacement directions (f-g) clustered in QWEN space, TRAIN only (no leakage)."""
    import numpy as np
    from sklearn.cluster import KMeans
    n_op = min(n_op, max(2, len(disp_tr) // 2))
    return KMeans(n_op, n_init=10, random_state=seed).fit(disp_tr).cluster_centers_.astype("float32")


def run(g, f, ctx, cmask, n_op=24, epochs=400, seed=0, ops_override=None, r=256, extended=False, log=print):
    import numpy as np
    rng = np.random.RandomState(seed); idx = rng.permutation(len(g)); nh = max(2, len(idx) // 5)
    he, tr = idx[:nh], idx[nh:]
    disp_tr = (f - g)[tr]
    ops = ops_override.astype("float32") if ops_override is not None else _discover_ops(disp_tr, n_op, seed)
    rand = rng.randn(*ops.shape).astype("float32"); rand *= np.linalg.norm(ops, axis=1, keepdims=True) / (np.linalg.norm(rand, axis=1, keepdims=True) + 1e-9)
    import torch
    cos = torch.nn.functional.cosine_similarity
    raw = float(cos(torch.tensor(g[he]), torch.tensor(f[he])).mean())
    E = lambda **kw: Refiner.train_eval(g, f, ctx, cmask, ops, tr, he, epochs=epochs, seed=seed, r=r, log=log, **kw)
    log(f"  train {len(tr)} / held {len(he)}; ops={ops.shape[0]}; r={r}")
    out = {"raw_g": raw,
           "direct": E(K=1, use_code=False, use_graph=False),
           "K1_full": E(K=1, use_code=True, use_graph=True),
           "K4_full": E(K=4, use_code=True, use_graph=True),
           "K4_nocode": E(K=4, use_code=False, use_graph=True),
           "K4_nograph": E(K=4, use_code=True, use_graph=False),
           "K4_randops": Refiner.train_eval(g, f, ctx, cmask, rand, tr, he, K=4, use_code=True, use_graph=True, epochs=epochs, seed=seed, r=r, log=log)}
    if extended:
        TN = lambda cls, o=ops: Refiner._train_net(cls, g, f, ctx, cmask, o, tr, he, K=4, epochs=epochs, seed=seed, r=r)
        log("  [ext] topology...")
        out["K4_topo"] = TN(Refiner.TopologyNet)
        log("  [ext] hybrid (graph prior + free residual)...")
        out["K4_hybrid"] = TN(Refiner.HybridNet)
        n_half = max(2, len(disp_tr) // 2)
        ops_half = _discover_ops(disp_tr[:n_half], n_op, seed)
        log(f"  [ext] compounding (ops from {n_half}/{len(disp_tr)} train instances)...")
        out["K4_ops_half"] = Refiner.train_eval(g, f, ctx, cmask, ops_half, tr, he, K=4, use_code=True, use_graph=True, epochs=epochs, seed=seed, r=r, log=log)
    return out, len(he)


def _report(outs, n):
    """outs = list of per-seed result dicts. Report means + PAIRED deltas (mean +/- std across seeds), so
    a small delta is not over-read as a verdict (the n=29 single-seed trap)."""
    import numpy as np
    keys = ["raw_g", "direct", "K1_full", "K4_full", "K4_nocode", "K4_nograph", "K4_randops"]
    ext_keys = ["K4_topo", "K4_hybrid", "K4_ops_half"]
    has_ext = any(k in outs[0] for k in ext_keys)
    if has_ext:
        keys += [k for k in ext_keys if k in outs[0]]
    S = len(outs)
    print(f"\n=== LGGN REFINE (held n~{n}, {S} seeds) — latent reaches the gold fix, in Qwen space ===")
    for k in keys:
        v = [o[k] for o in outs]
        print(f"  {k:14}: cos = {np.mean(v):.3f} +/- {np.std(v):.3f}")
    dl = {"recurrence  K4-K1       ": [o["K4_full"] - o["K1_full"] for o in outs],
          "constraints K4-nocode   ": [o["K4_full"] - o["K4_nocode"] for o in outs],
          "graph       K4-nograph  ": [o["K4_full"] - o["K4_nograph"] for o in outs],
          "learnedness K4-randops  ": [o["K4_full"] - o["K4_randops"] for o in outs],
          "vs_noniter  K4-direct   ": [o["K4_full"] - o["direct"] for o in outs]}
    if has_ext:
        if "K4_topo" in outs[0]:
            dl["topology   topo-full   "] = [o["K4_topo"] - o["K4_full"] for o in outs]
            dl["topo_vs_free topo-nogr  "] = [o["K4_topo"] - o["K4_nograph"] for o in outs]
        if "K4_hybrid" in outs[0]:
            dl["hybrid     hybr-nogr   "] = [o["K4_hybrid"] - o["K4_nograph"] for o in outs]
            dl["hybr_lift  hybr-full   "] = [o["K4_hybrid"] - o["K4_full"] for o in outs]
        if "K4_ops_half" in outs[0]:
            dl["compound   full-half   "] = [o["K4_full"] - o["K4_ops_half"] for o in outs]
    print()
    verd = {}
    for name, d in dl.items():
        m, s = float(np.mean(d)), float(np.std(d))
        sig = m - s > 0.0
        verd[name.strip().split()[0]] = sig and m > 0.02
        tag = "PASS" if (sig and m > 0.02) else ("flat/noisy" if m > 0 else "negative")
        print(f"  {name} = {m:+.3f} +/- {s:.3f}  -> {tag}")
    print()
    # ── base verdict ──
    print("  base: ", end="")
    if all(verd.get(p) for p in ("recurrence", "constraints", "graph", "learnedness")):
        print("all LGGN pillars load-bearing.")
    elif verd.get("recurrence") and verd.get("constraints") and verd.get("learnedness") and not verd.get("graph"):
        print("HRM validated (recurrence+constraints+real-dirs), graph-as-discrete-basis FLAT.")
    else:
        print("mixed -> read per-delta signs/stds.")
    # ── extended verdicts ──
    if has_ext:
        print("  extended:", end="")
        parts = []
        if verd.get("topology"):
            parts.append("TOPOLOGY helps (op→op transition prior is load-bearing)")
        elif "topology" in verd:
            parts.append("topology flat (op→op prior doesn't help)")
        if verd.get("hybrid"):
            parts.append("HYBRID wins (graph adds value as prior alongside free MLP)")
        elif "hybrid" in verd:
            parts.append("hybrid flat (graph adds nothing even as prior)")
        if verd.get("compound"):
            parts.append("COMPOUNDING real (more experience → better ops)")
        elif "compound" in verd:
            parts.append("compounding flat (op quality doesn't scale with experience)")
        print(" " + "; ".join(parts) if parts else " no clear signal.")


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
    out, n = run(g, f, ctx, cmask, ops_override=ops, epochs=400, seed=0, log=lambda *a: None)
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
    ap.add_argument("--seeds", type=int, default=5, help="train/held splits + inits averaged for CIs")
    ap.add_argument("--r", type=int, default=256, help="refiner inner dim (VRAM-free, increase for expressiveness)")
    ap.add_argument("--extended", action="store_true", help="run topology + hybrid + compounding tests")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        raise SystemExit(0 if _selftest() else 1)
    print(f"[lggn-refine] model={a.model} dataset={a.dataset} n={a.n} r={a.r}"
          f"{' +extended' if a.extended else ''}")
    g, f, ctx, cmask = _reprs(a.model, a.dataset, a.split, a.n, layer_frac=a.layer_frac)
    print(f"  {len(g)} instances, Qwen dim={g.shape[1]}")
    outs = []
    for s in range(a.seeds):
        print(f"\n--- seed {s+1}/{a.seeds} ---")
        out, nh = run(g, f, ctx, cmask, n_op=a.n_op, epochs=a.epochs, seed=s,
                      r=a.r, extended=a.extended, log=print)
        outs.append(out)
    _report(outs, nh)


if __name__ == "__main__":
    main()
