"""The LGGN CORE test: does LATENT TRAVERSAL (compose operators in the manifold) predict the fix better
than direct one-shot? Reasoning happens in the OPERATOR BASIS, not free embedding space.

  A) direct     : goal -> MLP -> fix_emb                       (no operators)
  B) compose    : goal -> weights over operators -> Sum w*centroid  (latent traversal, 1 step)
  C) traverse   : B + K residual refinement steps              (multi-step latent traversal)

Held cosine to the gold fix. If B/C >> A -> the operator manifold IS the right reasoning substrate (the
latent traversal composes toward the fix). If B/C <= A -> operators don't help even at the latent level.
No 4B, no Docker. (Decode of the reached fix-emb to CODE is a separate, LLM-side question.)

  python -m v5.runtime.latent_compose
  python -m v5.runtime.latent_compose --selftest
"""
from __future__ import annotations

import argparse


def _models(d, centroids, K_steps):
    import torch, torch.nn as nn
    C = torch.tensor(centroids, dtype=torch.float32)                 # [Nop, d] fixed operator basis

    class Direct(nn.Module):                                          # A: free embedding regression
        def __init__(s):
            super().__init__(); s.net = nn.Sequential(nn.Linear(d, 256), nn.ReLU(), nn.Linear(256, d))
        def forward(s, g): return s.net(g)

    class Compose(nn.Module):                                         # B: goal -> op weights -> Sum w*centroid
        def __init__(s):
            super().__init__(); s.w = nn.Sequential(nn.Linear(d, 256), nn.ReLU(), nn.Linear(256, C.shape[0]))
        def forward(s, g): return torch.softmax(s.w(g), -1) @ C

    class Traverse(nn.Module):                                        # C: compose + K residual latent steps
        def __init__(s):
            super().__init__()
            s.w = nn.Sequential(nn.Linear(d, 256), nn.ReLU(), nn.Linear(256, C.shape[0]))
            s.step = nn.Sequential(nn.Linear(2 * d, 256), nn.ReLU(), nn.Linear(256, C.shape[0]))
        def forward(s, g):
            h = torch.softmax(s.w(g), -1) @ C
            for _ in range(K_steps):
                h = h + torch.softmax(s.step(torch.cat([g, h], -1)), -1) @ C   # add another operator move
            return h
    return {"A_direct": Direct(), "B_compose": Compose(), "C_traverse": Traverse()}


def run(G, F, centroids=None, epochs=300, K_steps=3, seed=0, log=print):
    import numpy as np, torch
    rng = np.random.RandomState(seed); idx = rng.permutation(len(G)); nh = len(idx) // 5
    he, tr = idx[:nh], idx[nh:]
    if centroids is None:                                            # NO-LEAKAGE: operator basis from TRAIN fixes only
        from sklearn.cluster import KMeans
        centroids = KMeans(24, n_init=10, random_state=0).fit(F[tr]).cluster_centers_
        log(f"  (operator basis = {len(centroids)} clusters from TRAIN fixes only -- no held leakage)")
    Gt, Ft = torch.tensor(G[tr], dtype=torch.float32), torch.tensor(F[tr], dtype=torch.float32)
    Gh, Fh = torch.tensor(G[he], dtype=torch.float32), torch.tensor(F[he], dtype=torch.float32)
    Fn = torch.nn.functional.normalize
    out = {}
    for name, m in _models(G.shape[1], centroids, K_steps).items():
        torch.manual_seed(seed)
        opt = torch.optim.Adam(m.parameters(), lr=1e-3, weight_decay=1e-4)
        for _ in range(epochs):
            m.train(); opt.zero_grad()
            loss = (1 - (Fn(m(Gt)) * Fn(Ft)).sum(1)).mean(); loss.backward(); opt.step()
        m.eval()
        with torch.no_grad():
            out[name] = (Fn(m(Gh)) * Fn(Fh)).sum(1).mean().item()
        log(f"  {name}: held cos(pred, gold-fix) = {out[name]:.3f}")
    return out


def _report(out):
    print(f"\n=== LATENT TRAVERSAL (held cosine to gold fix) ===")
    a = out["A_direct"]
    for k in ("A_direct", "B_compose", "C_traverse"):
        tag = "(direct baseline)" if k == "A_direct" else f"{out[k]-a:+.3f} vs direct"
        print(f"  {k:12}: {out[k]:.3f}  {tag}")
    best = max(("B_compose", "C_traverse"), key=lambda k: out[k])
    print(f"\n  best operator-basis ({best}) vs direct = {out[best]-a:+.3f} -> "
          f"{'OPERATOR MANIFOLD is the right substrate (latent traversal composes toward the fix)' if out[best] > a + 0.02 else 'operators do NOT beat direct at the latent level'}")


def _swe_data():
    import numpy as np, torch, json
    from pathlib import Path
    from v5.runtime.iterative_refine import _data
    G, F = _data()
    ops = [json.loads(l) for l in Path("data/swe/discovered_ops_emb.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    C = np.array([o["_centroid"] for o in ops if o.get("_centroid")], dtype=float)
    return G, F, C


def _selftest() -> bool:
    print("latent_compose --selftest: fix IS a combination of operators -> compose must beat direct-from-noise\n")
    import numpy as np
    rng = np.random.RandomState(0); d, Nop, N = 32, 10, 400
    C = rng.randn(Nop, d).astype("float32")
    W = rng.rand(N, Nop).astype("float32"); W /= W.sum(1, keepdims=True)
    F = W @ C                                                          # fix = convex combo of operators
    G = W + 0.02 * rng.randn(N, Nop).astype("float32")                # goal ~ the weights (recoverable)
    G = np.hstack([G, rng.randn(N, d - Nop).astype("float32")])       # pad goal to width d
    out = run(G, F, C, epochs=250, log=lambda *a: None)
    print(f"  A={out['A_direct']:.2f} B={out['B_compose']:.2f} C={out['C_traverse']:.2f}")
    assert out["B_compose"] > out["A_direct"] + 0.02, "compose must beat direct when fix IS a combo of operators"
    print("\n  LATENT COMPOSE SELFTEST -> PASS (operator-basis composition wins when the fix lives in that basis)")
    return True


def main():
    ap = argparse.ArgumentParser(description="LGGN latent traversal: does operator composition predict the fix?")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--k-steps", type=int, default=3)
    a = ap.parse_args()
    if a.selftest:
        raise SystemExit(0 if _selftest() else 1)
    G, F, _C = _swe_data()
    print(f"[latent-compose] {len(G)} SWE golds, dim={G.shape[1]} (operator basis from TRAIN fixes -- no leakage)")
    _report(run(G, F, centroids=None, K_steps=a.k_steps))          # train-only basis


if __name__ == "__main__":
    main()
