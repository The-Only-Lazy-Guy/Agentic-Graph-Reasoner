"""REAL single-layer integration test for the two-state graph-memory injection.

Instantiates an actual Qwen3_5GatedDeltaNet (random weights — we test PLUMBING, not grounding),
wraps its delta-rule kernel to add an external graph read to core_attn_out (before norm/out_proj),
WITHOUT touching the recurrent state. Verifies the integration assumptions actually hold:

  P1  injection is additive at the core: S_graph=0  ->  output == baseline (identity)
  P2  injection changes the layer output: S_graph(all) != baseline
  P3  unlearn is clean: removing node j == building S_graph from scratch without j
      (rebuild == subtract per-node term, in the real per-head shape)
  P4  removing a node changes the output (behavior depends on which nodes are present)
  P5  two-state: the wrapper leaves last_recurrent_state untouched (graph never enters live S_lm)

This needs NO model download — one small layer, random init. Runs on CPU.
Run: PYTHONPATH=E:\\PROJECT\\graph_v5 python -m v5.deltanet_layer_test
"""
from __future__ import annotations

import torch

from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5GatedDeltaNet
try:
    from transformers.models.qwen3_5.modeling_qwen3_5 import l2norm
except Exception:                                   # noqa: BLE001
    def l2norm(x, dim=-1, eps=1e-6):
        return x / (x.norm(dim=dim, keepdim=True) + eps)


def _make_layer():
    from transformers import Qwen3_5Config
    cfg = Qwen3_5Config(
        hidden_size=128,
        linear_num_value_heads=4, linear_num_key_heads=2,
        linear_key_head_dim=16, linear_value_head_dim=16,
        linear_conv_kernel_dim=4,
        hidden_act="silu", rms_norm_eps=1e-6,
    )
    layer = Qwen3_5GatedDeltaNet(cfg, layer_idx=0).eval()
    return layer, cfg


def _build_S(keys, values, betas):
    """per-head S_graph: keys/values [N, H, dk]/[N, H, dv], betas [N, H] -> [H, dk, dv]."""
    return torch.einsum("nh,nhk,nhv->hkv", betas, keys, values)


def _install_graph_read(layer, S_graph_ref):
    """Wrap the chunk kernel: out += read(S_graph, q); state passes through untouched (two-state)."""
    orig = layer.chunk_gated_delta_rule

    def wrapped(query, key, value, g=None, beta=None, initial_state=None,
                output_final_state=False, use_qk_l2norm_in_kernel=False, **kw):
        out, state = orig(query, key, value, g=g, beta=beta, initial_state=initial_state,
                          output_final_state=output_final_state,
                          use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel, **kw)
        S = S_graph_ref[0]
        if S is not None:
            q = query
            if use_qk_l2norm_in_kernel:
                q = l2norm(q, dim=-1)
            q = q * (1.0 / query.shape[-1] ** 0.5)          # kernel's q-scale
            gr = torch.einsum("bshk,hkv->bshv", q.float(), S.float()).to(out.dtype)
            out = out + gr
        return out, state                                    # state UNCHANGED -> two-state
    layer.chunk_gated_delta_rule = wrapped


def _ok(name, cond, extra=""):
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}{('  ' + extra) if extra else ''}")
    return cond


def run():
    torch.manual_seed(0)
    layer, cfg = _make_layer()
    H, dk, dv = cfg.linear_num_value_heads, cfg.linear_key_head_dim, cfg.linear_value_head_dim
    b, seq = 1, 6
    hs = torch.randn(b, seq, cfg.hidden_size)

    with torch.no_grad():
        baseline = layer(hs)                                 # prefill, no cache -> chunk path

    # graph nodes -> per-head (k,v)
    N, j = 4, 1
    keys = torch.randn(N, H, dk); values = torch.randn(N, H, dv); betas = torch.rand(N, H) + 0.5
    S_all = _build_S(keys, values, betas)
    keep = [i for i in range(N) if i != j]
    S_scratch = _build_S(keys[keep], values[keep], betas[keep])
    S_subtract = S_all - torch.einsum("h,hk,hv->hkv", betas[j], keys[j], values[j])

    ref = [None]
    _install_graph_read(layer, ref)
    allp = True

    with torch.no_grad():
        ref[0] = None;       out_zero = layer(hs)
        ref[0] = S_all;      out_all = layer(hs)
        ref[0] = S_scratch;  out_drop_scratch = layer(hs)
        ref[0] = S_subtract; out_drop_sub = layer(hs)

    allp &= _ok("P1 S_graph=0 -> output == baseline (injection is additive/identity at empty)",
                torch.allclose(out_zero, baseline, atol=1e-5),
                f"max|d|={ (out_zero-baseline).abs().max():.2e}")
    allp &= _ok("P2 S_graph(all) changes the layer output",
                not torch.allclose(out_all, baseline, atol=1e-4),
                f"max|d|={ (out_all-baseline).abs().max():.4f}")
    allp &= _ok("P3 unlearn clean: rebuild-without-j == subtract-j (per-head, real shape)",
                torch.allclose(S_scratch, S_subtract, atol=1e-5))
    allp &= _ok("P3b -> same layer output for rebuild vs subtract",
                torch.allclose(out_drop_scratch, out_drop_sub, atol=1e-5))
    allp &= _ok("P4 removing node j changes the output (behavior depends on present nodes)",
                not torch.allclose(out_drop_scratch, out_all, atol=1e-4),
                f"max|d|={ (out_drop_scratch-out_all).abs().max():.4f}")

    # P5: two-state — the wrapped kernel must return the SAME recurrent state as the raw kernel
    # (graph read added to the OUTPUT only, never to last_recurrent_state). Compare directly on
    # identical q,k,v,g,beta: raw kernel vs the installed wrapper (ref[0]=S_all).
    from transformers.models.qwen3_5.modeling_qwen3_5 import (
        chunk_gated_delta_rule as _fa, torch_chunk_gated_delta_rule)
    raw_kernel = _fa or torch_chunk_gated_delta_rule
    q = torch.randn(b, seq, H, dk); k = torch.randn(b, seq, H, dk); v = torch.randn(b, seq, H, dv)
    g = torch.zeros(b, seq, H); beta = torch.rand(b, seq, H)
    o0, s0 = raw_kernel(q, k, v, g=g, beta=beta, initial_state=None, output_final_state=True,
                        use_qk_l2norm_in_kernel=True)
    ref[0] = S_all
    o1, s1 = layer.chunk_gated_delta_rule(q, k, v, g=g, beta=beta, initial_state=None,
                                          output_final_state=True, use_qk_l2norm_in_kernel=True)
    allp &= _ok("P5 two-state: recurrent state identical with/without graph read",
                (s0 is None and s1 is None) or torch.allclose(s0, s1, atol=1e-6))
    allp &= _ok("P5b graph read DID change the kernel output (sanity: o1 != o0)",
                not torch.allclose(o0, o1, atol=1e-4))

    print(f"\n{'ALL PASS — injection point + two-state plumbing hold on the REAL layer' if allp else 'SOME FAILED — integration assumption broken'}")
    return allp


if __name__ == "__main__":
    raise SystemExit(0 if run() else 1)
