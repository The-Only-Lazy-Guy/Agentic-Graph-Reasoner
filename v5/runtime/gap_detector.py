"""v4 — Gap Detector (REINFORCE-based stopping policy for traversal).

Trained to predict P(stop | h, hop) — when the latent has converged enough that
further hops won't improve retrieval quality. Uses REINFORCE with a reward
defined as +1 for stopping at-or-after the gold hop where all target records
are found, -1 for stopping too early.

The gap detector's training signal is clean because:
  - hop 0 precision is computed for each hop's retrieval against the gold set
  - gold_stop_hop = first hop where precision >= 0.5 (or max_hops if never)
  - REINFORCE: sample stop decision, reward = +1 if stop_hop >= gold_stop_hop, -1 otherwise

Architecture: MLP(768 + 1) → 1 (logit). Sigmoid output = P(stop).

  python -m v5.runtime.gap_detector --selftest
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


class GapDetector(nn.Module):
    """Simple MLP deciding whether to stop traversal at current hop.

    Input: [h (768-dim) ∥ hop_norm (scalar = hop / max_hops)]
    Output: P(stop) via sigmoid.

    Training: REINFORCE with stop-correct reward (+1 stop >= gold, -1 otherwise).
    """
    def __init__(self, d_hidden: int = 256, d_in: int = 768):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in + 1, d_hidden),
            nn.ReLU(),
            nn.Linear(d_hidden, d_hidden),
            nn.ReLU(),
            nn.Linear(d_hidden, 1),
        )

    def forward(self, h: torch.Tensor, hop: torch.Tensor,
                max_hops: torch.Tensor) -> torch.Tensor:
        """Returns P(stop) in [0, 1]."""
        hop_norm = (hop.float() / max_hops.float()).unsqueeze(-1)
        x = torch.cat([h, hop_norm], dim=-1)
        return torch.sigmoid(self.net(x)).squeeze(-1)

    def sample_stop(self, h: torch.Tensor, hop: torch.Tensor,
                    max_hops: torch.Tensor,
                    threshold: float | None = None) -> torch.Tensor:
        """Sample stop decision during training (threshold=None) or
        deterministic at inference (threshold=0.5). Returns bool tensor."""
        p = self.forward(h, hop, max_hops)
        if threshold is not None:
            return p >= threshold
        return torch.bernoulli(p).bool()

    def should_stop(self, h: np.ndarray, hop: int, max_hops: int,
                    threshold: float = 0.5) -> bool:
        """Deterministic stop decision for inference (called from TraversalRanker.retrieve).
        Returns True if P(stop | h, hop) >= threshold."""
        h_t = torch.as_tensor(h[None], dtype=torch.float32)
        hop_t = torch.tensor([hop], dtype=torch.float32)
        mh_t = torch.tensor([max_hops], dtype=torch.float32)
        return bool(self.sample_stop(h_t, hop_t, mh_t, threshold=threshold).item())


def train_gap_detector(gap: GapDetector, h: np.ndarray, gold_stop: np.ndarray,
                        max_hops: int, n_epochs: int = 200, lr: float = 1e-3,
                        seed: int = 0, log=print) -> GapDetector:
    """REINFORCE training for GapDetector.

    Args:
        gap: GapDetector instance
        h: (N, d) latent vectors at each step (per-hop)
        gold_stop: (N,) gold stop hop for each trajectory
        max_hops: maximum traversal hops
        n_epochs: REINFORCE epochs
        lr: learning rate
        seed: random seed
        log: logging function

    Returns:
        trained GapDetector (in eval mode)
    """
    torch.manual_seed(seed)
    opt = torch.optim.Adam(gap.parameters(), lr=lr)
    N, d = h.shape
    device = next(gap.parameters()).device

    h_t = torch.as_tensor(h, dtype=torch.float32, device=device)
    gs_t = torch.as_tensor(gold_stop, dtype=torch.long, device=device)
    mh_t = torch.as_tensor(max_hops, dtype=torch.long, device=device)

    for ep in range(n_epochs):
        # For each trajectory, uniformly sample a random hop ∈ [0, max_hops-1]
        # Use REINFORCE to learn P(stop) at that hop based on current h
        hop = torch.randint(0, max_hops, (N,), device=device)
        stop = gap.sample_stop(h_t, hop.float(), mh_t.float())   # (N,) bool

        correct = stop & (hop >= gs_t)          # stop correctly
        wrong_stop = stop & (hop < gs_t)         # stop too early
        # Rewire: keep running (not stop) at this hop is "correct" too, but only
        # REINFORCE on stop actions. Non-stops have zero advantage.
        reward = torch.where(correct, 1.0, torch.where(wrong_stop, -1.0, 0.0))

        logp = gap.forward(h_t, hop.float(), mh_t.float())
        logp = torch.clamp(logp, 1e-7, 1 - 1e-7)
        loss = -(reward * torch.where(stop, logp.log(), (1 - logp).log())).mean()

        opt.zero_grad()
        loss.backward()
        opt.step()

        if (ep + 1) % 50 == 0 or ep == n_epochs - 1:
            with torch.no_grad():
                p_stop = gap.forward(h_t, hop.float(), mh_t.float())
                acc_correct = ((p_stop >= 0.5) & (hop >= gs_t)).float().mean()
            log(f"  [gap-train] ep {ep+1}/{n_epochs} loss={loss.item():.4f} "
                f"stop_acc={acc_correct.item():.3f}")

    gap.eval()
    return gap


def _build_gold_stop(all_h: list[list[np.ndarray]],
                     all_gold_ids: list[set[str]],
                     all_retrieved_ids: list[list[list[str]]],
                     max_hops: int) -> np.ndarray:
    """Determine gold stop hop per trajectory.

    gold_stop = first hop where precision >= 0.5 against gold set.
    If never found, gold_stop = max_hops - 1 (always run to end).

    Args:
        all_h: list of [hop_vecs (list of np.ndarray)] per trajectory
        all_gold_ids: list of gold ID sets per trajectory
        all_retrieved_ids: list of [hop_retrieved_ids (list of str lists)] per trajectory
        max_hops: max traversal depth

    Returns:
        (total_steps,) array of gold stop labels
    """
    gold = []
    for gold_set, hop_retrieved in zip(all_gold_ids, all_retrieved_ids):
        best_hop = max_hops - 1
        for hop_i, ret_ids in enumerate(hop_retrieved):
            ret_set = set(ret_ids)
            if len(ret_set) == 0:
                continue
            precision = len(gold_set & ret_set) / len(ret_set)
            if precision >= 0.5:
                best_hop = hop_i
                break
        gold.append(best_hop)
    return np.asarray(gold, dtype=np.int64)


# ── selftest ─────────────────────────────────────────────────────────────────────

def _selftest() -> bool:
    print("gap_detector --selftest: REINFORCE training, stop prediction, "
          "gold_stop assignment\n")

    N, d, max_hops = 200, 32, 3
    rng = np.random.RandomState(42)

    # Simulate h vectors that converge toward gold targets → later hops have higher
    # precision. h at hop 0 is random, h at hop 2 is close to gold.
    gold_center = rng.randn(d).astype("float32")
    gold_center /= np.linalg.norm(gold_center)

    h_traj = []
    gold_stop = []
    gold_ids_list = []
    ret_ids_list = []

    for _ in range(N):
        gs = rng.randint(max_hops)  # 0, 1, or 2
        gold_stop.append(gs)
        traj = []
        for hop_i in range(max_hops):
            if hop_i < gs:
                vec = 0.8 * rng.randn(d).astype("float32")
            else:
                vec = gold_center + 0.1 * rng.randn(d).astype("float32")
            traj.append(vec)
            vec /= np.linalg.norm(vec)
        h_traj.append(traj)
        # Synthesize gold/retrieved IDs consistent with gs
        gold_ids_list.append({"g0"})
        ret = []
        for hop_i in range(max_hops):
            if hop_i >= gs:
                ret.append(["g0", f"d{hop_i}"])
            else:
                ret.append(["d0"])
        ret_ids_list.append(ret)

    gold_stop_arr = _build_gold_stop(h_traj, gold_ids_list, ret_ids_list, max_hops)
    assert gold_stop_arr.shape == (N,)
    assert (gold_stop_arr == np.asarray(gold_stop)).all(), \
        "gold_stop should match synthesized labels"
    print("  [1] gold_stop assignment -> PASS")

    # Flatten h per-step for training
    h_flat = np.concatenate(h_traj, 0).astype("float32")  # (N*max_hops, d)
    gs_flat = np.repeat(gold_stop_arr, max_hops)

    gap = GapDetector(d_hidden=64, d_in=d)
    train_gap_detector(gap, h_flat, gs_flat, max_hops, n_epochs=100, log=lambda *a: None)

    # Evaluate: at each hop, predict P(stop). Should be low at hop 0, high at hop max_hops-1.
    with torch.no_grad():
        h_t = torch.as_tensor(h_flat, dtype=torch.float32)
        hop_t = torch.arange(max_hops, dtype=torch.float32).repeat(N)
        mh_t = torch.tensor(max_hops, dtype=torch.float32)
        p_stop = gap.forward(h_t, hop_t, mh_t)

    # By construction: hop 0 should have low P(stop), hop 2 should have high P(stop)
    p_by_hop = p_stop.reshape(N, max_hops).mean(0)
    print(f"    mean P(stop) by hop: {p_by_hop.numpy().round(3)}  "
          f"(expect rising)")

    assert p_by_hop[0] < p_by_hop[-1], \
        f"P(stop) should increase with hop: {p_by_hop.numpy()}"
    print(f"  [2] P(stop) rises with hop -> PASS")
    assert p_by_hop[-1] > 0.5, \
        f"P(stop) at final hop should be >0.5, got {p_by_hop[-1]:.3f}"
    print(f"  [3] P(stop) at final hop >0.5 -> PASS")

    # Deterministic inference
    decisions = gap.sample_stop(
        h_t, hop_t, torch.full_like(hop_t, max_hops, dtype=torch.float32),
        threshold=0.5)
    assert decisions.shape == (N * max_hops,)
    assert decisions.dtype == torch.bool
    print("  [4] deterministic inference -> PASS")

    # Save/load roundtrip
    import tempfile, os
    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, "gap.pt")
        torch.save(gap.state_dict(), p)
        gap2 = GapDetector(d_hidden=64, d_in=d)
        gap2.load_state_dict(torch.load(p, weights_only=True))
        gap2.eval()
        with torch.no_grad():
            p2 = gap2.forward(h_t, hop_t, mh_t)
        assert torch.allclose(p_stop, p2, atol=1e-6)
    print("  [5] save/load roundtrip -> PASS")

    print("\n  GAP DETECTOR SELFTEST -> PASS")
    return True


def train_gap_from_traversal(ranker_dir: str, model_name: str,
                              archetypes: tuple = ("preference",),
                              n_seeds: int = 20, epochs: int = 200, d: int = 768,
                              pool_k: int = 16, k_impl: int = 2, max_hops: int = 3,
                              lr: float = 1e-3, seed: int = 0, log=print) -> str:
    """Collect REAL (h, gold_stop) pairs by running TraversalRanker on eval chains, then
    train the gap detector with REINFORCE. This is the proper training path (not synthetic
    random data): gold_stop = number of source sessions the dependency needs, h = the actual
    latent after each hop's refinement. Saves gap.pt into ranker_dir. Returns the path."""
    import os
    import tempfile
    import shutil
    import torch
    from v5.memory.store import make_mpnet_embedder
    from v5.runtime.memory_refiner import load_ranker
    from v5.memory.memory import TotalMemory
    from v5.runtime.traversal_ranker import TraversalRanker
    from v5.runtime.project_gen import make_instance

    net, feat_proj, ops, K = load_ranker(ranker_dir)
    embed_fn = make_mpnet_embedder()

    H, GS = [], []
    for seed in range(n_seeds):
        inst = make_instance(archetypes[0], seed)
        td = tempfile.mkdtemp()
        try:
            mem = TotalMemory(td, mode="concept", embed_fn=embed_fn)
            repo = {}
            for s in inst["sessions"]:
                t = s["target_file"]
                if s.get("buggy"):
                    repo[t] = s["buggy"][t]
                mem.write(goal=s["spec"], old=repo.get(t, ""), new=s["gold"][t],
                          trace=s["spec"][:400], verified=True, file_path=t,
                          task_id=s["sid"], kind=s["kind"])
                repo[t] = s["gold"][t]
            for s in inst["sessions"]:
                if not s.get("withheld"):
                    continue
                src_idxs = s.get("source_session_idxs")
                if not src_idxs:
                    idx = s.get("source_session_idx")
                    src_idxs = [idx] if idx is not None else []
                if not src_idxs:
                    continue
                rk = TraversalRanker(mem.impls, mem.concepts, embed_fn, net,
                                     feat_proj, ops, pool_k=pool_k, k_impl=k_impl,
                                     max_hops=max_hops)
                res = rk.retrieve(s["spec"], "", s["target_file"])
                gold_stop = len(src_idxs)
                for h_vec in res.hop_hs:
                    H.append(h_vec)
                    GS.append(gold_stop)
        finally:
            shutil.rmtree(td, ignore_errors=True)

    if not H:
        log("  [gap-train] no dependency sessions collected, aborting")
        return ""
    H_arr = np.asarray(H, dtype=np.float32)
    GS_arr = np.asarray(GS, dtype=np.int64)
    gap = GapDetector(d_hidden=256, d_in=d)
    train_gap_detector(gap, H_arr, GS_arr, max_hops, n_epochs=epochs, lr=lr,
                       seed=seed, log=log)
    out = os.path.join(ranker_dir, "gap.pt")
    torch.save(gap.state_dict(), out)
    log(f"  [gap-train] {len(H)} (h, gold_stop) pairs from {n_seeds} seeds -> {out}")
    return out


def main():
    import argparse
    ap = argparse.ArgumentParser(description="v4 — Gap Detector (REINFORCE stopping)")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--train-gap", action="store_true",
                    help="train gap detector on REAL traversal h vectors")
    ap.add_argument("--ranker", default="artifacts/traversal_ranker",
                    help="trained ranker dir (loaded to run traversal for data collection)")
    ap.add_argument("--model", default="Qwen/Qwen2.5-3B")
    ap.add_argument("--archetypes", default="preference")
    ap.add_argument("--n-seeds", type=int, default=20)
    ap.add_argument("--epochs", type=int, default=200)
    a = ap.parse_args()
    if a.selftest:
        raise SystemExit(0 if _selftest() else 1)
    if a.train_gap:
        archetypes = tuple(x.strip() for x in a.archetypes.split(",") if x.strip())
        train_gap_from_traversal(a.ranker, a.model, archetypes=archetypes,
                                 n_seeds=a.n_seeds, epochs=a.epochs, log=print)
        return
    ap.print_help()


if __name__ == "__main__":
    main()
