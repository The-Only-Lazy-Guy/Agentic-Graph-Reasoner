"""Multi-layer Operator injector — the ADAPTER FIX experiment.

The single-layer OperatorInjector adds ONE steering vector at ONE layer: it carries direction + the
first token, but not exact multi-token content (RAG, which attends to the real tokens, does). This
injects the op-signed node shift at SEVERAL layers at once — more of the node's signal, closer to what
attention delivers — to test whether a richer injection converts belief-shift into exact reasoning /
emission (the agentic-coder need: the right symbol/patch, not just the right direction).

Drop-in: same combine()/inject()/answer_logits() API as OperatorInjector, but the steer is a
per-layer dict. Each hooked layer captures its own last-token hidden, so combine builds the shift at
every layer in one forward.

  layers e.g. "8,14,20,26" (4B) or "6,10,14" (1.5B).
"""
from __future__ import annotations

import contextlib
from typing import List, Tuple

import torch

SIGN = {"ASSERT": 1.0, "INVALIDATE": -1.0, "TRANSFORM": 1.0, "GATE": 0.0, "SLOT": 0.0}


def _layers(model):
    m = model
    for attr in ("model", "layers"):
        m = getattr(m, attr)
    return m


class OperatorInjectorML:
    def __init__(self, model, tok, layers: List[int], alpha: float = 1.0):
        self.model, self.tok, self.alpha = model, tok, alpha
        self.L = list(layers)
        self.dev = next(model.parameters()).device
        self.blocks = _layers(model)
        self._cap = {l: None for l in self.L}
        self._steer = {l: None for l in self.L}
        for l in self.L:
            self.blocks[l].register_forward_hook(self._mk_hook(l))

    def _mk_hook(self, l):
        def hook(mod, inp, out):
            is_tup = isinstance(out, tuple)
            h = out[0] if is_tup else out
            self._cap[l] = h.detach()
            if self._steer[l] is not None:
                h = h + self._steer[l].to(h.dtype)
                return ((h,) + tuple(out[1:])) if is_tup else h
            return out
        return hook

    @torch.no_grad()
    def _hlast_all(self, text):
        for l in self.L:
            self._steer[l] = None
        ids = self.tok(text, return_tensors="pt").input_ids.to(self.dev)
        self.model(ids)
        return {l: self._cap[l][0, -1].float() for l in self.L}

    def combine(self, nodes: List[Tuple[str, str]], query: str, normalize: bool = False):
        base = self._hlast_all(query)
        v = {l: torch.zeros_like(base[l]) for l in self.L}
        for text, op in nodes:
            s = SIGN.get(op, 0.0)
            if s == 0.0:
                continue
            nh = self._hlast_all(f"{text}\n{query}")
            for l in self.L:
                shift = nh[l] - base[l]
                if normalize:
                    shift = shift / (shift.norm() + 1e-6) * base[l].norm()
                v[l] = v[l] + s * self.alpha * shift
        return v

    @contextlib.contextmanager
    def inject(self, v):
        for l in self.L:
            self._steer[l] = v[l] if v is not None else None
        try:
            yield
        finally:
            for l in self.L:
                self._steer[l] = None

    @torch.no_grad()
    def answer_logits(self, query, v=None):
        with self.inject(v) if v is not None else contextlib.nullcontext():
            ids = self.tok(query, return_tensors="pt").input_ids.to(self.dev)
            return self.model(ids).logits[0, -1].float()
