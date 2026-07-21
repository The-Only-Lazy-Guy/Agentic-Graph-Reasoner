"""trm_wm.py — the TRM as a REAL reasoner COUPLED to the frozen LM (design (b)), with a WORKING MEMORY.

The prior TRM was a ranker: it reordered atoms and never touched the LM. This wires it into the LM's
computation so its reasoning actually MODULATES generation.

  LONG-TERM memory = the graph                       (unchanged; grows without training)
  WORKING memory   = K slots the TRM refines over T recursion steps, INITIALIZED from the top-K retrieved
                     atoms  -> grounded in real content, not a free latent (this is what killed soft-prompt)
  COUPLING to LM   = a GATED CROSS-ATTENTION adapter (Flamingo-style, tanh-gate init 0) inserted at a few
                     frozen-LM layers: the LM's hidden states ATTEND to the working-memory slots.

Only the adapter + the slot-refiner (the "TRM") train, on VERIFIED answers; the LM never moves (frozen ->
anti-poison preserved, same sanction as trainer.py). The tanh gate starts at 0 so at init the LM is
BITWISE-identical to the base model; the adapter can only *add* signal once it earns lower loss.

Mechanism proven by --selftest on distilgpt2 (no Qwen needed):
  (i)   gate=0  -> LM logits identical to base            (identity at init; can't wreck fluency)
  (ii)  gate!=0 -> LM logits change                       (working memory is causally wired into the LM)
  (iii) train   -> a token placed ONLY in the slots (absent from the prompt) becomes the LM's answer, and
                   it GENERALIZES to HELD-OUT slot content it never trained on  (real copy-from-memory,
                   not memorization) — with the gate ABLATED to 0 the effect vanishes (proves causality).

    python -m v5.runtime.trm_wm --selftest                                    # mechanism, distilgpt2, local
    python -m v5.runtime.trm_wm --run --lm Qwen/Qwen3-4B-Instruct-2507        # real experiment (their GPU)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = str(Path(__file__).resolve().parents[2])
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import torch
import torch.nn as nn

from embedder import encode_batch, EMBED_DIM


# ================================================================================================
# gated cross-attention adapter — the LM's hidden states attend to the working-memory slots
# ================================================================================================
class GatedCrossAttn(nn.Module):
    """h (LM hidden [B,S,d]) attends to slots [K,d]; output = h + tanh(g)*proj(attn). g init 0 -> identity."""

    def __init__(self, d: int, n_heads: int = 4):
        super().__init__()
        assert d % n_heads == 0
        self.h, self.dh = n_heads, d // n_heads
        self.q = nn.Linear(d, d)
        self.k = nn.Linear(d, d)
        self.v = nn.Linear(d, d)
        self.o = nn.Linear(d, d)
        self.g = nn.Parameter(torch.zeros(1))            # tanh gate -> 0 at init = identity

    def forward(self, h: torch.Tensor, slots: torch.Tensor) -> torch.Tensor:
        B, S, d = h.shape
        if slots.dim() == 2:                                              # [K,d] shared across the batch
            slots = slots.unsqueeze(0).expand(B, -1, -1)
        Bk, K, _ = slots.shape                                           # [B,K,d] per-example working memory
        q = self.q(h).view(B, S, self.h, self.dh).transpose(1, 2)          # [B,h,S,dh]
        k = self.k(slots).view(Bk, K, self.h, self.dh).permute(0, 2, 1, 3)  # [B,h,K,dh]
        v = self.v(slots).view(Bk, K, self.h, self.dh).permute(0, 2, 1, 3)
        att = torch.softmax((q @ k.transpose(-1, -2)) / (self.dh ** 0.5), dim=-1)  # [B,h,S,K]
        ctx = (att @ v).transpose(1, 2).reshape(B, S, d)                   # [B,S,d]
        return h + torch.tanh(self.g) * self.o(ctx)


# ================================================================================================
# WMReasoner — the working memory + its recursive refinement + the coupling hooks
# ================================================================================================
class WMReasoner(nn.Module):
    """K working-memory slots, initialized from retrieved atom embeddings, refined by T recursion steps
    (the real TRM role: iterative refinement), then read by the LM via gated cross-attention adapters."""

    def __init__(self, d_lm: int, couple_layers, d_emb: int = EMBED_DIM, T: int = 4, n_heads: int = 4):
        super().__init__()
        self.T = T
        self.proj_atom = nn.Linear(d_emb, d_lm)          # slot init from a retrieved atom
        self.proj_task = nn.Linear(d_emb, d_lm)          # the query conditions the refinement
        self.upd = nn.Sequential(nn.Linear(3 * d_lm, d_lm), nn.GELU(), nn.Linear(d_lm, d_lm))
        self.norm = nn.LayerNorm(d_lm)
        self.adapters = nn.ModuleList([GatedCrossAttn(d_lm, n_heads) for _ in couple_layers])
        self.couple_layers = list(couple_layers)
        self._slots = None                               # set per forward via set_context()

    def refine(self, task_emb: torch.Tensor, atom_embs: torch.Tensor) -> torch.Tensor:
        """task_emb [d_emb], atom_embs [K,d_emb] -> working memory [K,d_lm] after T recursion steps."""
        q = self.proj_task(task_emb)                     # [d_lm]
        z = self.proj_atom(atom_embs)                    # [K,d_lm]  (grounded slot init)
        for _ in range(self.T):
            ctx = z.mean(0, keepdim=True).expand_as(z)   # global summary of the working memory
            qb = q.unsqueeze(0).expand_as(z)
            z = self.norm(z + self.upd(torch.cat([z, qb, ctx], dim=-1)))   # recursive refinement
        return z

    def set_context(self, task_emb, atom_embs):
        te = torch.as_tensor(task_emb, dtype=torch.float32, device=self._device())
        ae = torch.as_tensor(atom_embs, dtype=torch.float32, device=self._device())
        if ae.dim() == 1:
            ae = ae.unsqueeze(0)
        self._slots = self.refine(te, ae)

    def set_slots_direct(self, slots: torch.Tensor):
        """Set the working memory directly (already in d_lm). Accepts [d], [K,d], or [B,K,d] (per-example)."""
        self._slots = slots.unsqueeze(0) if slots.dim() == 1 else slots

    def clear(self):
        self._slots = None

    def _device(self):
        return self.proj_atom.weight.device

    def couple(self, wb) -> list:
        """Register forward hooks so the coupled LM layers add gated cross-attention to the slots."""
        handles = []
        for a_i, L in enumerate(self.couple_layers):
            handles.append(wb.layers[L].register_forward_hook(self._mk_hook(a_i)))
        return handles

    def _mk_hook(self, idx):
        def hook(_mod, _inp, out):
            if self._slots is None:
                return None
            h = out[0] if isinstance(out, tuple) else out
            h2 = self.adapters[idx](h.float(), self._slots.float()).to(h.dtype)   # adapter in fp32, cast back
            if isinstance(out, tuple):
                return (h2,) + tuple(out[1:])
            return h2
        return hook


# ================================================================================================
# selftest — prove the mechanism on distilgpt2 (identity / causal / trainable+generalizing)
# ================================================================================================
def _vocab_words(tok, n: int = 120):
    """Pull n single-token lowercase words straight from the vocab (so we have ENOUGH data to force the
    adapter to learn the general copy rule instead of memorizing a tiny lookup)."""
    words = []
    for tid in range(len(tok)):
        s = tok.decode([tid])
        if s[:1] == " " and s[1:].isalpha() and s[1:].islower() and len(s) >= 4:
            words.append((s[1:], tid))
        if len(words) >= n:
            break
    return words


def _probe(wb, R, pids, targets, slot_of, steps=200, lr=5e-3):
    """Train the adapter (+ whatever slot_of uses) so the frozen LM emits each target token given ONLY its
    slot. slot_of(word) -> [K,d_lm]. 80/20 train/held-out split -> held-out tests GENERALIZATION, not memory.
    Batched (all words in one LM forward). Returns (train, held-out, held-out-ablated, final-gate, loss)."""
    k = max(1, int(0.8 * len(targets)))
    train, test = targets[:k], targets[k:]
    for a in R.adapters:
        with torch.no_grad():
            a.g.fill_(1.5)                               # warm-start (gate=0 freezes the interior)
    opt = torch.optim.Adam([p for p in R.parameters() if p.requires_grad], lr=lr, weight_decay=1e-2)

    def slots_of(split):                                 # -> [B,K,d_lm] (fresh each call, keeps grad)
        return torch.stack([slot_of(w) for w, _ in split], dim=0)

    def batch_ids(split):
        return pids.expand(len(split), -1)

    def acc(split, ablate=False):
        R.set_slots_direct(slots_of(split).detach())
        gs = [float(a.g) for a in R.adapters]
        if ablate:
            for a in R.adapters:
                with torch.no_grad():
                    a.g.zero_()
        with torch.no_grad():
            preds = wb.model(batch_ids(split)).logits[:, -1].argmax(-1).tolist()
        if ablate:
            for a, gv in zip(R.adapters, gs):
                with torch.no_grad():
                    a.g.fill_(gv)
        return sum(int(p == t) for p, (_, t) in zip(preds, split)) / len(split)

    tgt = torch.tensor([t for _, t in train], device=wb.device)
    R.train()
    last = float("nan")
    for _ in range(steps):
        R.set_slots_direct(slots_of(train))
        logits = wb.model(batch_ids(train)).logits[:, -1]     # [B,V]
        loss = nn.functional.cross_entropy(logits, tgt)
        opt.zero_grad(); loss.backward(); opt.step()
        last = float(loss)
    R.eval()
    return acc(train), acc(test), acc(test, ablate=True), float(R.adapters[0].g), last


def selftest():
    from v5.runtime.dcpd_latent import WhiteBox
    torch.manual_seed(0)
    print("trm_wm.py --selftest : TRM working memory coupled to a FROZEN distilgpt2 (mechanism proof)\n")
    wb = WhiteBox("distilgpt2", quant="fp32")
    d_lm = wb.d_model
    couple = [4, 5]                                       # the LAST two layers -> most direct control of output
    R = WMReasoner(d_lm, couple_layers=couple).to(wb.device)
    for p in wb.model.parameters():
        p.requires_grad_(False)                          # FROZEN LM
    handles = R.couple(wb)

    prompt = "The answer is"
    pids = wb.tok(prompt, return_tensors="pt").input_ids.to(wb.device)
    words = _vocab_words(wb.tok, 1000)                    # OVER-constrain the 768x768 map: 800 train / 200 held-out
    wid = dict(words)

    # (i) identity at init: slots present but gate=0 -> logits identical to the base model
    R.clear()
    base = wb.model(pids).logits.detach()
    R.set_context(encode_batch([prompt])[0], encode_batch(["banana"])[0])
    id_diff = (base - wb.model(pids).logits.detach()).abs().max().item()
    print(f"  (i)   identity@init   max|base - withslots(gate=0)| = {id_diff:.2e}   "
          f"{'PASS' if id_diff < 1e-4 else 'FAIL'}")
    # (ii) causal: bump a gate -> logits must change
    with torch.no_grad():
        R.adapters[0].g.fill_(1.0)
    ch = (base - wb.model(pids).logits.detach()).abs().max().item()
    print(f"  (ii)  causal wiring   max|base - withslots(gate=1)| = {ch:.2e}   "
          f"{'PASS' if ch > 1e-3 else 'FAIL'}\n")
    with torch.no_grad():
        R.adapters[0].g.zero_()

    # PROBE A — WIRING: slot = the LM's OWN token embedding of the target (same space). Isolates the adapter.
    lm_emb = wb.model.get_input_embeddings().weight                        # [V, d_lm]
    tr_a, te_a, ab_a, g_a, l_a = _probe(
        wb, R, pids, words, lambda w: lm_emb[wid[w]].unsqueeze(0), steps=400)
    print(f"  (A) WIRING  (slot = LM's own embedding):  train {tr_a:.2f}  HELD-OUT {te_a:.2f}  "
          f"ablate->0 {ab_a:.2f}  gate {g_a:+.2f}  loss {l_a:.3f}")

    # reset the reasoner so probe B is independent
    R2 = WMReasoner(d_lm, couple_layers=couple).to(wb.device)
    for h in handles:
        h.remove()
    handles = R2.couple(wb)
    task_emb = encode_batch([prompt])[0]
    memb = {w: encode_batch([w])[0] for w, _ in words}
    # PROBE B — GRAPH BRIDGE: slot from MiniLM via the refiner (the real path). Can the tiny LM read graph space?
    tr_b, te_b, ab_b, g_b, l_b = _probe(
        wb, R2, pids, words, lambda w: R2.refine(torch.as_tensor(task_emb, dtype=torch.float32, device=wb.device),
                                                 torch.as_tensor(memb[w], dtype=torch.float32, device=wb.device).unsqueeze(0)),
        steps=400)
    print(f"  (B) BRIDGE  (slot from MiniLM graph emb):  train {tr_b:.2f}  HELD-OUT {te_b:.2f}  "
          f"ablate->0 {ab_b:.2f}  gate {g_b:+.2f}  loss {l_b:.3f}")

    print()
    if te_a >= 0.5 and ab_a < te_a:
        print("  => WIRING PROVEN: the working memory causally + GENERALIZABLY drives the frozen LM (probe A).")
        if te_b >= 0.5:
            print("     BRIDGE also works on distilgpt2 -- graph-space slots read too.")
        else:
            print("     BRIDGE (reading MiniLM graph space) did NOT generalize on distilgpt2 -- expected: the tiny")
            print("     LM can't decode a foreign embedding space. This is the job of the capable 4B (--run).")
    else:
        print("  => WIRING still not generalizing -- the adapter architecture needs more work (report honest).")
    for h in handles:
        h.remove()


def run_real(lm_name: str):
    """The real experiment on the deployment LM: teach facts into the graph, then show the working memory
    lets the frozen LM answer a COMPOSITION the base LM cannot — trained on verified answers only."""
    print(f"run_real: wiring the TRM working memory into {lm_name} (4-bit). "
          f"This needs the real GPU; build the compose set + train the adapter here.")
    print("  (scaffold — the selftest proves the mechanism; this drives it on the deployment model.)")
    # left as the entry point for the user's GPU; the module above is the reusable machinery.


def main():
    ap = argparse.ArgumentParser(description="TRM working memory coupled to a frozen LM (real reasoner, design b)")
    ap.add_argument("--selftest", action="store_true", help="prove the mechanism on distilgpt2 (local, fast)")
    ap.add_argument("--run", action="store_true", help="real experiment on --lm")
    ap.add_argument("--lm", type=str, default="Qwen/Qwen3-4B-Instruct-2507")
    a = ap.parse_args()
    if a.run:
        run_real(a.lm)
    else:
        selftest()


if __name__ == "__main__":
    main()
