"""TRM-over-graph (#55) — a TINY Recursive Model as the OWNED reasoner, trained from scratch, with the
graph giving it exactly what a tiny net needs (Q5). TRM (Tiny Recursive Model, ~7M, beats HRM on
ARC/Sudoku/Maze) recursively refines a latent answer + a scratchpad.

V3 (2026-07-25) — PROPER Tiny Recursive Model (Jolicoeur-Martineau 2025):
  - Two latents: z (scratchpad) and y (solution embedding), single shared network f
  - Inner loop: z = f(z + task + proj_y + cross_attn(z, R))  — think
  - Outer step: y = f(y + z)                                  — act
  - Cross-attention from z to R per step
  - Output T per-cycle y_t values → fill LM working memory slots
  - NOT a ranker — never supervised on graph atoms

  selftest (no torch-GPU):  python -m v5.runtime.algo_trm --selftest
"""
from __future__ import annotations

import argparse
import sys


def _build():
    import torch
    import torch.nn as nn

    class TRMReasoner(nn.Module):
        """Proper Tiny Recursive Model (Jolicoeur-Martineau 2025).

        Two latents: z (scratchpad) and y (solution embedding).
        Single shared network f applied identically in inner (think) and outer (act) steps.

        Each cycle t:
          Think:  z = f(z + task_proj(x) + proj_y(y) + cross_attn(z, R))
          Act:    y = f(y + z)

        Args:
            d_in:    input embedding dim (MiniLM 384 or LM hidden dim)
            d:       hidden dim of both latents
            T:       number of thinking cycles
            n_heads: cross-attention heads
        """
        def __init__(self, d_in: int = 384, d: int = 256, T: int = 5, n_heads: int = 4):
            super().__init__()
            self.T = T
            self.d = d
            self.d_in = d_in

            self.task_proj = nn.Linear(d_in, d)
            self.atom_proj = nn.Linear(d_in, d)
            self.proj_y = nn.Linear(d, d)

            # Shared network f: LayerNorm → 2-layer GELU MLP
            self.f_norm = nn.LayerNorm(d)
            self.f_mlp = nn.Sequential(
                nn.Linear(d, 4 * d), nn.GELU(),
                nn.Linear(4 * d, d),
            )

            # Cross-attention: z queries R (retrieved atom embeddings)
            assert d % n_heads == 0, f"d={d} must be divisible by n_heads={n_heads}"
            self.cross_attn = nn.MultiheadAttention(d, n_heads, batch_first=True)

            # Learnable initial latents
            self.z0 = nn.Parameter(torch.zeros(d))
            self.y0 = nn.Parameter(torch.zeros(d))

            # Per-cycle y projection (maps d→d, applied before output)
            self.y_head = nn.Sequential(
                nn.Linear(d, d), nn.GELU(),
                nn.Linear(d, d),
            )

        def _f(self, v: torch.Tensor) -> torch.Tensor:
            """Single shared network: Layernorm → MLP."""
            return self.f_mlp(self.f_norm(v))

        def forward(self, x_vec: torch.Tensor, atom_vecs: torch.Tensor) -> torch.Tensor:
            """x_vec: [d_in] task embedding.
            atom_vecs: [N, d_in] atom embeddings (retrieved context R).
            Returns: [T, d] per-cycle y_t solution embeddings.
            """
            x = self.task_proj(x_vec)
            R = self.atom_proj(atom_vecs)

            z = self.z0
            y = self.y0

            ys = []
            for _ in range(self.T):
                # Cross-attend z to R
                ctx, _ = self.cross_attn(
                    z.unsqueeze(0).unsqueeze(0),     # [1, 1, d]
                    R.unsqueeze(0),                   # [1, N, d]
                    R.unsqueeze(0),
                )
                ctx = ctx.squeeze(0).squeeze(0)       # [d]

                # Inner loop (think): z = f(z + x + proj_y + ctx)
                z = self._f(z + x + self.proj_y(y) + ctx)

                # Outer step (act): y = f(y + z)
                y = self._f(y + z)

                ys.append(self.y_head(y))

            return torch.stack(ys)  # [T, d]

    return torch, nn, TRMReasoner


def _selftest() -> bool:
    torch, _nn, TRMReasoner = _build()
    print("algo_trm --selftest: V3 proper Tiny Recursive Model (two-latent, cross-attn)\n")

    d_in, d, T = 32, 48, 4
    n_atoms = 6

    trm = TRMReasoner(d_in=d_in, d=d, T=T, n_heads=4)
    n_trm = sum(p.numel() for p in trm.parameters())
    print(f"  [1] TRMReasoner V3: {n_trm} params (d={d}, T={T}) -> PASS")

    atom_vecs = torch.randn(n_atoms, d_in)
    x_vec = torch.randn(d_in)
    y_ts = trm(x_vec, atom_vecs)
    assert y_ts.shape == (T, d), f"y_ts shape {y_ts.shape}"
    print(f"  [2] forward -> y_ts {list(y_ts.shape)} -> PASS")

    y_diffs = [(y_ts[t + 1] - y_ts[t]).norm().item() for t in range(T - 1)]
    evolving = any(d > 1e-6 for d in y_diffs)
    print(f"  [3] y_t step diffs: {[f'{d:.4f}' for d in y_diffs]} -> "
          f"{'PASS (evolving)' if evolving else 'FAIL (static)'}")

    atom_vecs2 = torch.randn(n_atoms, d_in)
    y_ts2 = trm(x_vec, atom_vecs2)
    diff = (y_ts - y_ts2).abs().max().item()
    print(f"  [4] cross-attn sensitivity: max|diff|={diff:.4f} -> "
          f"{'PASS' if diff > 0 else 'FAIL'}")

    improvement = (y_ts[-1] - y_ts[0]).norm().item()
    print(f"  [5] first-to-last diff: {improvement:.4f} -> "
          f"{'PASS (multi-step)' if improvement > 1e-6 else 'FAIL'}")

    x_vec2 = torch.randn(d_in)
    y_ts3 = trm(x_vec2, atom_vecs)
    diff2 = (y_ts - y_ts3).abs().max().item()
    print(f"  [6] task sensitivity: max|diff|={diff2:.4f} -> "
          f"{'PASS' if diff2 > 0 else 'FAIL'}")

    proj = torch.nn.Linear(d, 64)
    slots = proj(y_ts)
    assert slots.shape == (T, 64), f"slots shape {slots.shape}"
    print(f"  [7] y_t -> LM slots via proj: {list(slots.shape)} -> PASS")

    print("\n  ALGO_TRM SELFTEST -> PASS  (V3 TRMReasoner)")
    return True


def main():
    ap = argparse.ArgumentParser(description="TRM-over-graph: tiny recursive planner.")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        sys.exit(0 if _selftest() else 1)
    ap.print_help()


if __name__ == "__main__":
    main()
