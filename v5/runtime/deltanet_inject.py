"""DeltaNet graph-memory injector + projector (the fast-weight alternative to per-node LoRAs).

Compiles graph nodes into an EXTERNAL additive fast-weight state S_graph = Σ_i beta_i k_i⊗v_i,
read in PARALLEL with the frozen LM's Qwen3_5GatedDeltaNet layers (added to core_attn_out before
norm/out_proj), NEVER folded into the live recurrent state. Plumbing validated in
v5/deltanet_layer_test.py (prefill+decode, two-state non-leak). The OPEN question this enables:
does a trained projector make the frozen LM actually READ the written nodes (ground), and is
node add/remove clean? -> train the projector (deltanet_ground.py) and run the killer test.

  DeltaNetGraphProjector : node_emb -> per-head (k, v), keys l2-normed; orthogonality loss helper
                           (T3b: selective unlearn needs ~orthogonal node keys)
  DeltaNetGraphInjector  : find GatedDeltaNet layers, wrap both kernels, manage per-layer S_graph,
                           set_nodes / drop_node / clear  (drop = rebuild from kept per-node terms)

Self-test (no model download):  python -m v5.runtime.deltanet_inject --selftest
"""
from __future__ import annotations

from typing import Dict, List, Optional

import torch
import torch.nn as nn

try:
    from transformers.models.qwen3_5.modeling_qwen3_5 import l2norm
except Exception:                                   # noqa: BLE001
    def l2norm(x, dim=-1, eps=1e-6):
        return x / (x.norm(dim=dim, keepdim=True) + eps)


class DeltaNetGraphProjector(nn.Module):
    """node_emb [N, E] -> keys [N, H, dk] (l2-normed per head), values [N, H, dv]."""
    def __init__(self, emb_dim: int, num_heads: int, head_k_dim: int, head_v_dim: int):
        super().__init__()
        self.H, self.dk, self.dv = num_heads, head_k_dim, head_v_dim
        self.to_k = nn.Linear(emb_dim, num_heads * head_k_dim)
        self.to_v = nn.Linear(emb_dim, num_heads * head_v_dim)

    def forward(self, node_emb: torch.Tensor):
        N = node_emb.shape[0]
        k = self.to_k(node_emb).reshape(N, self.H, self.dk)
        v = self.to_v(node_emb).reshape(N, self.H, self.dv)
        k = l2norm(k, dim=-1)                            # unit keys -> dot = cosine (selectivity)
        return k, v


def key_orthogonality_loss(keys: torch.Tensor) -> torch.Tensor:
    """keys [N, H, dk] -> mean off-diagonal |K Kᵀ| over heads. 0 == nodes have orthogonal keys.
    T3b: additive memory is interference-free for selective unlearn only when keys are ~orthogonal."""
    N = keys.shape[0]
    if N < 2:
        return keys.sum() * 0.0
    G = torch.einsum("nhk,mhk->hnm", keys, keys)        # [H, N, N] per-head Gram
    eye = torch.eye(N, device=keys.device).unsqueeze(0)
    off = (G * (1 - eye)).abs().sum(dim=(1, 2)) / (N * (N - 1))
    return off.mean()


def _wrap_kernel(orig, s_ref):
    """out += read(S_graph, q); state untouched (two-state). s_ref[0] is [H, dk, dv] or None."""
    def wrapped(query, key, value, g=None, beta=None, initial_state=None,
                output_final_state=False, use_qk_l2norm_in_kernel=False, **kw):
        out, state = orig(query, key, value, g=g, beta=beta, initial_state=initial_state,
                          output_final_state=output_final_state,
                          use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel, **kw)
        S = s_ref[0]
        if S is not None:
            q = query
            if use_qk_l2norm_in_kernel:
                q = l2norm(q, dim=-1)
            q = q * (1.0 / query.shape[-1] ** 0.5)
            out = out + torch.einsum("bshk,hkv->bshv", q.float(), S.float()).to(out.dtype)
        return out, state
    return wrapped


class DeltaNetGraphInjector:
    """Installs the two-state graph read on chosen GatedDeltaNet layers and manages S_graph.

    projectors: one DeltaNetGraphProjector per injected layer (head dims may differ; here equal).
    """
    def __init__(self, model, layer_indices: List[int], emb_dim: int, device=None):
        self.device = device or next(model.parameters()).device
        self.gdn: Dict[int, nn.Module] = {}
        self._srefs: Dict[int, list] = {}
        self._node_terms: Dict[int, Optional[torch.Tensor]] = {}   # per-layer [N, H, dk, dv]
        self.projectors = nn.ModuleDict()
        layers = self._decoder_layers(model)
        for i in layer_indices:
            gdn = getattr(layers[i], "linear_attn", None)
            if gdn is None:
                raise ValueError(f"layer {i} has no linear_attn (not a GatedDeltaNet layer)")
            self.gdn[i] = gdn
            self._srefs[i] = [None]
            gdn.chunk_gated_delta_rule = _wrap_kernel(gdn.chunk_gated_delta_rule, self._srefs[i])
            gdn.recurrent_gated_delta_rule = _wrap_kernel(gdn.recurrent_gated_delta_rule, self._srefs[i])
            self.projectors[str(i)] = DeltaNetGraphProjector(
                emb_dim, gdn.num_v_heads, gdn.head_k_dim, gdn.head_v_dim).to(self.device)

    @staticmethod
    def _decoder_layers(model):
        for path in (("model", "layers"), ("model", "model", "layers"), ("layers",)):
            m = model
            try:
                for a in path:
                    m = getattr(m, a)
                if len(m) > 0:
                    return list(m)
            except AttributeError:
                continue
        raise ValueError("cannot locate decoder layers")

    @staticmethod
    def gdn_layer_indices(model) -> List[int]:
        """Indices of layers that are GatedDeltaNet (linear-attn), not full attention."""
        return [i for i, l in enumerate(DeltaNetGraphInjector._decoder_layers(model))
                if getattr(l, "linear_attn", None) is not None]

    def set_nodes(self, node_emb: torch.Tensor, betas: Optional[torch.Tensor] = None):
        """Write nodes into every injected layer's S_graph. Stores per-node terms for clean drop."""
        node_emb = node_emb.to(self.device)
        N = node_emb.shape[0]
        betas = torch.ones(N, device=self.device) if betas is None else betas.to(self.device)
        for i, gdn in self.gdn.items():
            k, v = self.projectors[str(i)](node_emb)                 # [N,H,dk],[N,H,dv]
            terms = torch.einsum("n,nhk,nhv->nhkv", betas, k, v)     # [N,H,dk,dv] per-node rank-1
            self._node_terms[i] = terms
            self._srefs[i][0] = terms.sum(0)                         # S_graph [H,dk,dv]

    def drop_node(self, idx: int):
        """Selective unlearn: rebuild S_graph from the KEPT per-node terms (exact, reversible)."""
        for i in self.gdn:
            terms = self._node_terms[i]
            keep = torch.cat([terms[:idx], terms[idx + 1:]], dim=0)
            self._node_terms[i] = keep
            self._srefs[i][0] = keep.sum(0) if keep.shape[0] > 0 else None

    def clear(self):
        for i in self.gdn:
            self._srefs[i][0] = None
            self._node_terms[i] = None

    def keys_for_ortho_loss(self, node_emb: torch.Tensor):
        """Per-layer keys for the orthogonality regularizer during projector training."""
        node_emb = node_emb.to(self.device)
        return {i: self.projectors[str(i)](node_emb)[0] for i in self.gdn}


# ── self-test on a single real layer (no model download) ───────────────────────────
def _selftest():
    from transformers import Qwen3_5Config
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5GatedDeltaNet

    cfg = Qwen3_5Config(hidden_size=128, linear_num_value_heads=4, linear_num_key_heads=2,
                        linear_key_head_dim=16, linear_value_head_dim=16, linear_conv_kernel_dim=4,
                        hidden_act="silu", rms_norm_eps=1e-6)

    class _Stub(nn.Module):                              # minimal model: one GDN decoder layer
        def __init__(s):
            super().__init__()
            lyr = nn.Module(); lyr.linear_attn = Qwen3_5GatedDeltaNet(cfg, 0)
            s.model = nn.Module(); s.model.layers = nn.ModuleList([lyr])
    torch.manual_seed(0)
    stub = _Stub().eval()
    gdn = stub.model.layers[0].linear_attn
    inj = DeltaNetGraphInjector(stub, [0], emb_dim=64)
    hs = torch.randn(1, 6, cfg.hidden_size)
    ok = True

    with torch.no_grad():
        base = gdn(hs)
        emb = torch.randn(5, 64)
        inj.set_nodes(emb)
        out_all = gdn(hs)
        inj.drop_node(1)
        out_drop = gdn(hs)
        inj.clear()
        out_clear = gdn(hs)

    def _ok(n, c):
        print(f"  [{'PASS' if c else 'FAIL'}] {n}"); return c
    ok &= _ok("clear() == baseline (empty graph is identity)", torch.allclose(out_clear, base, atol=1e-5))
    ok &= _ok("set_nodes changes output", not torch.allclose(out_all, base, atol=1e-4))
    ok &= _ok("drop_node changes output (vs all-nodes)", not torch.allclose(out_drop, out_all, atol=1e-4))
    # orthogonality loss runs + is non-negative
    kl = inj.keys_for_ortho_loss(emb)
    loss = sum(key_orthogonality_loss(k) for k in kl.values())
    ok &= _ok(f"orthogonality loss computes ({float(loss):.4f}) >= 0", float(loss) >= 0)
    print("SELFTEST", "PASS" if ok else "FAIL")
    return ok


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        raise SystemExit(0 if _selftest() else 1)
    print("use --selftest")
