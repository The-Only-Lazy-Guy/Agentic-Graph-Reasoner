"""HRM's core claim on OUR data: does ITERATIVE continuous-latent refinement predict the fix better
than one-shot? Weight-TIED recurrence (same params, K steps) vs K=1 -- so any gain is ITERATION, not
capacity. goal embedding -> K refine steps -> predicted fix embedding; held cosine to the gold fix.

If held-cosine RISES with K -> latent iteration is the missing piece (build the full trained reasoner).
If FLAT -> iteration doesn't help here. No 4B, no Docker.

  python -m v5.runtime.iterative_refine            # SWE golds, mpnet
  python -m v5.runtime.iterative_refine --selftest
"""
from __future__ import annotations

import argparse


def _data():
    import numpy as np, torch
    from v5.graph_grower.swe_load import load_instances
    from v5.runtime.operator_discovery import extract_hunks, _fix_text
    from v5.training.providers import RealEmbedder
    emb = RealEmbedder(torch.device("cpu"))

    def E(texts, bs=48):
        out = []
        for i in range(0, len(texts), bs):
            ch = texts[i:i + bs]; d = emb.embed_nodes({str(j): t[:1000] for j, t in enumerate(ch)})
            out.extend(d[str(j)] for j in range(len(ch)))
        return np.asarray(out, dtype=float)
    goals, fixes = [], []
    for i in load_instances(name="lite", split="test", limit=0):
        hs = extract_hunks(i.get("patch", "") or "")
        if not hs:
            continue
        code = "\n".join(l for _f, r, _a in hs for l in r)[:900]
        goals.append(((i.get("problem_statement") or "")[:900]) + "\nCODE:\n" + code)
        fixes.append(_fix_text(hs))
    return E(goals), E(fixes)


def _refiner(d_in, d, K):
    import torch, torch.nn as nn

    class Refiner(nn.Module):
        """h0 = proj(g); h <- GRUCell(g, h) x K (weight-TIED) ; fix = out(h). K=1 == one-shot."""
        def __init__(self):
            super().__init__()
            self.proj = nn.Linear(d_in, d)
            self.cell = nn.GRUCell(d_in, d)
            self.g_in = nn.Linear(d_in, d_in)
            self.out = nn.Linear(d, d_in)
            self.K = K

        def forward(self, g):
            h = torch.tanh(self.proj(g))
            gi = self.g_in(g)
            for _ in range(self.K):
                h = self.cell(gi, h)
            return self.out(h)
    return Refiner()


def run(G, F, Ks=(1, 2, 4, 8), epochs=300, seed=0, log=print):
    import numpy as np, torch
    rng = np.random.RandomState(seed)
    idx = rng.permutation(len(G)); nh = len(idx) // 5
    he, tr = idx[:nh], idx[nh:]
    Gt = torch.tensor(G[tr], dtype=torch.float32); Ft = torch.tensor(F[tr], dtype=torch.float32)
    Gh = torch.tensor(G[he], dtype=torch.float32); Fh = torch.tensor(F[he], dtype=torch.float32)
    Fn = torch.nn.functional.normalize
    res = {}
    for K in Ks:
        torch.manual_seed(seed)
        m = _refiner(G.shape[1], 256, K)
        opt = torch.optim.Adam(m.parameters(), lr=1e-3, weight_decay=1e-4)
        for ep in range(epochs):
            m.train(); opt.zero_grad()
            pred = m(Gt)
            loss = (1 - (Fn(pred) * Fn(Ft)).sum(1)).mean()          # cosine loss
            loss.backward(); opt.step()
        m.eval()
        with torch.no_grad():
            heldcos = (Fn(m(Gh)) * Fn(Fh)).sum(1).mean().item()
        res[K] = heldcos
        log(f"  K={K}: held cos(pred, gold-fix) = {heldcos:.3f}")
    return res


def _report(res):
    base = res[min(res)]
    best = max(res, key=res.get)
    print(f"\n=== ITERATIVE REFINEMENT (held cosine to gold fix) ===")
    for K, c in res.items():
        print(f"  K={K:2}: {c:.3f}  {'(one-shot baseline)' if K == min(res) else f'{c-base:+.3f} vs one-shot'}")
    gain = res[best] - base
    print(f"\n  best K={best}: {gain:+.3f} over one-shot -> "
          f"{'ITERATION HELPS (latent refinement is the missing piece)' if gain > 0.02 else 'iteration does NOT help (flat)'}")


def _selftest() -> bool:
    print("iterative_refine --selftest: iteration must fit a K-step-structured target (synthetic)\n")
    import numpy as np, torch
    rng = np.random.RandomState(0)
    d = 32; N = 400
    G = rng.randn(N, d).astype("float32")
    W = rng.randn(d, d) * 0.3
    F = G.copy()
    for _ in range(6):                                            # target = 6 applications of W (needs iteration)
        F = np.tanh(F @ W)
    res = run(G, F, Ks=(1, 6), epochs=400, log=lambda *a: None)
    print(f"  K=1 {res[1]:.3f} | K=6 {res[6]:.3f}")
    assert res[6] > res[1] + 0.015, "iteration must beat one-shot on an iteration-structured target"
    print("\n  ITERATIVE REFINE SELFTEST -> PASS")
    return True


def main():
    ap = argparse.ArgumentParser(description="Does latent iteration predict the fix better than one-shot? (HRM core)")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        raise SystemExit(0 if _selftest() else 1)
    G, F = _data()
    print(f"[iter-refine] {len(G)} SWE golds, dim={G.shape[1]}")
    _report(run(G, F))


if __name__ == "__main__":
    main()
