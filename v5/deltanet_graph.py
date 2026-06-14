"""DeltaNet graph-memory: compile graph nodes into an EXTERNAL additive fast-weight state
read in parallel with the frozen LM's GatedDeltaNet layers — NOT folded into the live
recurrent state (which would entangle, per the delta rule's kv_mem subtraction).

Layer math (transformers Qwen3_5GatedDeltaNet, torch_recurrent_gated_delta_rule):
    state S : [b, h, k_dim, v_dim]   (outer-product fast weights)
    write   : S += k ⊗ ((v - (S*k).sum(-2)) * beta)      # delta rule (entangling!)
    read    : out = (S * q[...,None]).sum(-2)             # = Σ over k_dim

External graph memory (this module):
    S_graph = Σ_i beta_i * (k_i ⊗ v_i)                   # additive, per-node terms kept
    graph_read(q) = (S_graph * q[...,None]).sum(-2) = Σ_i beta_i (k_i·q) v_i

Two-state injection:  core_attn_out += graph_read(q_t)   (S_graph NEVER enters the live S_lm)
Unlearn node i     :  drop beta_i k_i⊗v_i from the sum   (EXACT — proven in tests below)

This file is pure-tensor + the real layer kernel; runs on CPU in ms. No LM needed.
Run:  PYTHONPATH=E:\\PROJECT\\graph_v5 python -m v5.deltanet_graph
"""
from __future__ import annotations

import torch


# ── external additive graph memory ──────────────────────────────────────────────
def build_graph_state(keys, values, betas):
    """keys [N,dk], values [N,dv], betas [N] -> S_graph [dk,dv] = Σ beta_i k_i⊗v_i."""
    return torch.einsum("n,nk,nv->kv", betas, keys, values)


def graph_read(S, q):
    """S [dk,dv], q [...,dk] -> [...,dv]; matches the layer read (S*q[...,None]).sum(-2)."""
    return (S.unsqueeze(0) * q.unsqueeze(-1)).sum(-2) if q.dim() == 2 else (S * q.unsqueeze(-1)).sum(-2)


def node_term(k, v, beta):
    """Single node's rank-1 contribution beta * k⊗v -> [dk,dv]."""
    return beta * torch.outer(k, v)


# ── faithful replica of the live layer kernel (for the entanglement contrast) ────
def _delta_rule_seq(keys, values, betas, queries, gates=None):
    """Feed nodes as a token sequence through the REAL delta rule, then read with `queries`
    appended as no-write probe steps (beta=0). Returns probe readouts. Mirrors
    transformers.models.qwen3_5.modeling_qwen3_5.torch_recurrent_gated_delta_rule."""
    N, dk = keys.shape
    dv = values.shape[1]
    P = queries.shape[0]
    S = torch.zeros(dk, dv)
    g = torch.zeros(N) if gates is None else gates           # exp(0)=1 -> no decay
    # write phase
    for i in range(N):
        S = S * g[i].exp()
        kv_mem = (S * keys[i].unsqueeze(-1)).sum(0)           # [dv]
        delta = (values[i] - kv_mem) * betas[i]
        S = S + keys[i].unsqueeze(-1) * delta.unsqueeze(0)
    # probe phase (beta=0 -> no write, just read)
    outs = []
    for p in range(P):
        outs.append((S * queries[p].unsqueeze(-1)).sum(0))
    return torch.stack(outs)                                  # [P, dv]


def _cross_check_real_kernel():
    """Confirm our read matches the actual transformers kernel on a tiny case."""
    try:
        from transformers.models.qwen3_5.modeling_qwen3_5 import torch_recurrent_gated_delta_rule
    except Exception as e:                                    # noqa: BLE001
        return None, f"(skip: {e})"
    torch.manual_seed(0)
    b, seq, h, dk, dv = 1, 4, 1, 8, 8
    q = torch.randn(b, seq, h, dk); k = torch.randn(b, seq, h, dk)
    v = torch.randn(b, seq, h, dv); beta = torch.ones(b, seq, h); g = torch.zeros(b, seq, h)
    out, S = torch_recurrent_gated_delta_rule(q, k, v, g, beta, None, True)
    # replicate read at the last position from the returned final state S [b,h,dk,dv]
    q_last = q[0, -1, 0] * (1 / dk ** 0.5)                    # kernel scales q by 1/sqrt(dk)
    ours = (S[0, 0] * q_last.unsqueeze(-1)).sum(0)
    return torch.allclose(out[0, -1, 0], ours, atol=1e-4), ""


# ── tests ────────────────────────────────────────────────────────────────────────
def _ok(name, cond, extra=""):
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}{('  ' + extra) if extra else ''}")
    return cond


def run_tests():
    torch.manual_seed(1)
    dk = dv = 16
    N = 5
    keys = torch.randn(N, dk); keys = keys / keys.norm(dim=-1, keepdim=True)   # overlapping (NOT orthogonal)
    values = torch.randn(N, dv)
    betas = torch.rand(N) + 0.5
    q = torch.randn(dk)
    allp = True

    # T1: additive readout == Σ_i beta_i (k_i·q) v_i
    S = build_graph_state(keys, values, betas)
    manual = sum(betas[i] * (keys[i] @ q) * values[i] for i in range(N))
    allp &= _ok("T1 additive readout = Σ beta_i (k_i·q) v_i",
                torch.allclose(graph_read(S, q), manual, atol=1e-5))

    # T2: drop node j -> EXACT clean removal (the two-state unlearn claim)
    j = 2
    S_minus = S - node_term(keys[j], values[j], betas[j])
    expect = graph_read(S, q) - betas[j] * (keys[j] @ q) * values[j]
    rebuilt = build_graph_state(torch.cat([keys[:j], keys[j+1:]]),
                                torch.cat([values[:j], values[j+1:]]),
                                torch.cat([betas[:j], betas[j+1:]]))
    allp &= _ok("T2 drop term == read minus node-j contribution (EXACT)",
                torch.allclose(graph_read(S_minus, q), expect, atol=1e-6))
    allp &= _ok("T2 drop term == rebuild-without-j (states identical)",
                torch.allclose(S_minus, rebuilt, atol=1e-6))

    # T3: REVERSIBILITY contrast (the real reason for two-state) ---------------------
    # additive: subtract node-j term == build-from-scratch-without-j  (EXACT, reversible) [shown in T2]
    # delta-rule (folding writes into the live state): naive subtract of j's k⊗v does NOT
    # recover the from-scratch-without-j state -> writes are entangled, NOT reversible.
    Sd_full = torch.zeros(dk, dv)
    for i in range(N):
        kv = (Sd_full * keys[i].unsqueeze(-1)).sum(0)
        Sd_full = Sd_full + keys[i].unsqueeze(-1) * ((values[i] - kv) * betas[i]).unsqueeze(0)
    keepi = [i for i in range(N) if i != j]
    Sd_scratch = torch.zeros(dk, dv)
    for i in keepi:
        kv = (Sd_scratch * keys[i].unsqueeze(-1)).sum(0)
        Sd_scratch = Sd_scratch + keys[i].unsqueeze(-1) * ((values[i] - kv) * betas[i]).unsqueeze(0)
    Sd_naive_remove = Sd_full - node_term(keys[j], values[j], betas[j])
    delta_revert_err = (Sd_naive_remove - Sd_scratch).norm().item()
    add_revert_err = (S_minus - rebuilt).norm().item()        # ~0 from T2
    allp &= _ok("T3 additive removal IS reversible (subtract == scratch)", add_revert_err < 1e-6,
                f"err={add_revert_err:.2e}")
    allp &= _ok("T3 delta-rule folding is NOT reversible (naive subtract != scratch)",
                delta_revert_err > 1e-3, f"err={delta_revert_err:.4f}")

    # T3b: SELECTIVITY depends on key orthogonality (a REQUIREMENT on the projector) ----
    b_idx = 4                                                 # a node != j(=2) to probe
    Q, _ = torch.linalg.qr(torch.randn(dk, N))                # orthonormal node keys
    okeys = Q.T[:N]
    So = build_graph_state(okeys, values, betas)
    drift_ortho = (graph_read(So, okeys[b_idx])
                   - graph_read(So - node_term(okeys[j], values[j], betas[j]), okeys[b_idx])).norm().item()
    drift_overlap = (graph_read(S, keys[b_idx])
                     - graph_read(S - node_term(keys[j], values[j], betas[j]), keys[b_idx])).norm().item()
    allp &= _ok("T3b orthogonal keys -> removing node-j leaves node-b read ~unchanged (selective)",
                drift_ortho < 1e-5, f"ortho_drift={drift_ortho:.2e}  overlap_drift={drift_overlap:.4f}")
    print(f"       => projector MUST produce ~orthogonal node keys for interference-free selective unlearn")

    # T4: read matches the REAL transformers kernel
    res, msg = _cross_check_real_kernel()
    if res is None:
        print(f"  [SKIP] T4 real-kernel cross-check {msg}")
    else:
        allp &= _ok("T4 graph_read matches transformers torch kernel final-state read", res)

    print(f"\n{'ALL PASS — two-state additive memory is exact + selective; delta-rule entangles (as warned)' if allp else 'SOME FAILED'}")
    return allp


if __name__ == "__main__":
    raise SystemExit(0 if run_tests() else 1)
