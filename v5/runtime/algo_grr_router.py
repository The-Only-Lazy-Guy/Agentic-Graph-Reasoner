"""algo_grr_router — the NEURAL piece that actually helps the text-membrane.

Lesson from the soft-prompt experiments: a learned LATENT that DELIVERS code fails (lossy reconstruction
-> routing collapse; scaling made it worse, 73%->15%). Text delivers code losslessly (100%). So the neural
model must NOT transport code. Its job is ROUTING: given a task, decide WHICH atoms the membrane should
fetch. It emits a DISCRETE pointer (indices into the graph); the membrane delivers those atoms' exact code
as text. Pointer != code -> the collapse cannot happen.

Why a neural router beats plain cosine RAG: cosine ranks atoms by TEXT similarity to the task. But a needed
atom is often NOT text-similar -- a helper the solution CALLS, an "opposite/contradicts" node, a structural
dependency. Cosine is blind to those. A router trained on VERIFIED (task -> atoms-that-solved-it) pairs
learns the dependency STRUCTURE from data, so it retrieves the text-dissimilar-but-required atoms that
cosine misses. This is the learned generalisation of the hand-built topology retriever.

    neural WHERE it helps (discrete routing)  +  text WHERE it must be exact (delivery)

    selftest (no GPU):  python -m v5.runtime.algo_grr_router --selftest
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

    class NeuralRouter(nn.Module):
        """Scores each atom for a task from pairwise features [task, atom, task*atom]. Trained on verified
        (task -> used-atoms) with BCE (needed=1, distractor=0). Learns dependency structure cosine can't see.
        Output is a RANKING -> top-k SELECTION (discrete pointers), never code."""

        def __init__(self, d: int, h: int = 96):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(3 * d, h), nn.ReLU(),
                nn.Linear(h, h), nn.ReLU(),
                nn.Linear(h, 1),
            )

        def forward(self, t, A):                       # t:[d]  A:[N,d]  ->  [N] scores
            T = t.unsqueeze(0).expand_as(A)
            x = torch.cat([T, A, T * A], dim=-1)
            return self.net(x).squeeze(-1)

    return torch, nn, NeuralRouter


def train_router(router, tasks, atoms, epochs: int = 400, lr: float = 3e-3, seed: int = 0):
    """tasks = list of (task_vec, needed_idx_set). atoms = [N,d]. BCE over all atoms per task."""
    import torch
    torch.manual_seed(seed)
    A = torch.as_tensor(atoms, dtype=torch.float32)
    opt = torch.optim.Adam(router.parameters(), lr=lr)
    lossf = torch.nn.BCEWithLogitsLoss()
    N = A.shape[0]
    for ep in range(epochs):
        tot = 0.0
        for tvec, needed in tasks:
            t = torch.as_tensor(tvec, dtype=torch.float32)
            y = torch.zeros(N)
            for j in needed:
                y[j] = 1.0
            logits = router(t, A)
            loss = lossf(logits, y)
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item()
        if ep % 100 == 0 or ep == epochs - 1:
            print(f"  [router ep {ep:3d}] bce {tot/len(tasks):.4f}", flush=True)
    return router


def _recall_at_k(scores, needed, k):
    import torch
    top = set(torch.topk(torch.as_tensor(scores), k).indices.tolist())
    return len(top & set(needed)) / len(needed)


def make_router_policy(router, atoms):
    """Adapt the router to the MembraneSolver policy_fn seam: given the task vector + candidate atom vecs,
    return a score per candidate (higher = fetch first). Delivery stays TEXT (membrane prepends the code)."""
    import torch
    A_all = torch.as_tensor(atoms, dtype=torch.float32)

    def policy_fn(task_vec, cand_vecs):
        t = torch.as_tensor(task_vec, dtype=torch.float32)
        C = torch.as_tensor(cand_vecs, dtype=torch.float32)
        with torch.no_grad():
            return router(t, C).tolist()
    return policy_fn, A_all


# ═══════════════════════════════════════════════════════════════════════════════
# SELFTEST — a controlled corpus where the needed helper is TEXT-DISSIMILAR to the
# task (a structural dependency). Cosine (and |cosine|) miss it; the router learns it.
# ═══════════════════════════════════════════════════════════════════════════════

def _synth_corpus(n_atoms=24, d=16, alpha=2.0, n_train=200, n_test=80, seed=0):
    """Atoms sit on a circle (embedded in R^d). A task needs its GOAL atom (same angle -> cosine finds it)
    AND a HELPER atom rotated by a FIXED arbitrary angle alpha (text-DISSIMILAR -> cosine misses it, and
    alpha != pi so |cosine| misses it too). The router must LEARN the alpha-rotation dependency from data.
    Held-out tasks use unseen angles -> tests generalisation of the learned rule, not memorisation."""
    import numpy as np
    rng = np.random.default_rng(seed)
    B = rng.standard_normal((2, d)); B /= np.linalg.norm(B, axis=1, keepdims=True)  # 2 random dirs -> a plane

    def emb(theta, noise=0.0):
        v = np.cos(theta) * B[0] + np.sin(theta) * B[1]
        v = v + noise * rng.standard_normal(d)
        return (v / (np.linalg.norm(v) + 1e-9)).astype(np.float32)

    atom_ang = np.linspace(0, 2 * np.pi, n_atoms, endpoint=False)
    atoms = np.stack([emb(a) for a in atom_ang]).astype(np.float32)

    def nearest(theta):
        return int(np.argmin(np.abs(np.angle(np.exp(1j * (atom_ang - theta))))))

    def gen(n, s):
        r = np.random.default_rng(s)
        out = []
        for _ in range(n):
            phi = r.uniform(0, 2 * np.pi)
            goal = nearest(phi); helper = nearest(phi + alpha)
            if goal == helper:
                helper = nearest(phi + alpha + 0.3)
            out.append((emb(phi, noise=0.03), {goal, helper}))
        return out

    return atoms, gen(n_train, seed + 1), gen(n_test, seed + 2)


def _selftest() -> bool:
    print("algo_grr_router --selftest: NEURAL routing on the membrane (discrete pointer, text delivery)\n")
    torch, nn, NeuralRouter = _build()
    import numpy as np
    atoms, train, test = _synth_corpus()
    d = atoms.shape[1]; k = 2
    A = torch.as_tensor(atoms)

    # baselines that use only the embedding geometry (what plain graph-RAG has)
    def cos_scores(t):
        return (A @ torch.as_tensor(t)).tolist()

    def abscos_scores(t):
        return (A @ torch.as_tensor(t)).abs().tolist()

    router = NeuralRouter(d)
    n_params = sum(p.numel() for p in router.parameters())
    print(f"  corpus: {atoms.shape[0]} atoms, {len(train)} train / {len(test)} held-out tasks; "
          f"router {n_params/1e3:.1f}k params\n")
    train_router(router, train, atoms, epochs=400)

    def rec(score_fn, kk):
        import numpy as _np
        return float(_np.mean([_recall_at_k(score_fn(t), nd, kk) for t, nd in test]))

    def rou_scores(t):
        with torch.no_grad():
            return router(torch.as_tensor(t), A).tolist()

    print("\n  Recall@k on HELD-OUT tasks (fraction of the needed atoms retrieved):")
    print(f"    {'method':<20} {'@2':>6} {'@3':>6}")
    print(f"    {'cosine RAG':<20} {rec(cos_scores,2):>6.2f} {rec(cos_scores,3):>6.2f}   text-similarity only -> gets goal, MISSES the helper")
    print(f"    {'|cosine|':<20} {rec(abscos_scores,2):>6.2f} {rec(abscos_scores,3):>6.2f}   fixed metric -> misses the alpha-rotated helper")
    print(f"    {'NEURAL router (ours)':<20} {rec(rou_scores,2):>6.2f} {rec(rou_scores,3):>6.2f}   learned the structural dependency from verified solves")
    cos_r, rou_r = rec(cos_scores, 2), rec(rou_scores, 2)
    lift = rou_r - cos_r
    ok = rou_r >= 0.80 and lift >= 0.25
    print(f"\n    Recall@2 lift over cosine RAG = {lift:+.2f}  ->  {'PASS' if ok else 'FAIL'}")
    print("\n  Mechanism: the neural model picks WHICH atoms (discrete pointer); the membrane delivers their\n"
          "  exact code as TEXT. Neural where it helps (routing), text where it must be exact (delivery).")
    print(f"\n  ALGO_GRR_ROUTER SELFTEST -> {'PASS' if ok else 'FAIL'}")
    return ok


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        sys.exit(0 if _selftest() else 1)
    ap.print_help()


if __name__ == "__main__":
    main()
