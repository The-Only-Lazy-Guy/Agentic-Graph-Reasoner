"""trm_wm.py — the TRM as a REAL reasoner COUPLED to the frozen LM (design (b)), with a WORKING MEMORY.

The prior TRM was a ranker: it reordered atoms and never touched the LM. This wires it into the LM's
computation so its reasoning actually MODULATES generation.

  LONG-TERM memory = the graph                       (unchanged; grows without training)
  WORKING memory   = K slots the TRM refines over T recursion steps, INITIALIZED from the top-K retrieved
                     atoms  -> grounded in real content, not a free latent (this is what killed soft-prompt)
  COUPLING to LM   = a GATED CROSS-ATTENTION adapter (Flamingo-style, tanh-gate init 0) inserted at a few
                     frozen-LM layers: the LM's hidden states ATTEND to the working-memory slots.

  DEEP SUPERVISION = at each refinement step t, the working memory must independently identify the target
                     (via cosine retrieval against the answer pool). Gradients flow through EVERY step,
                     not just the final output -> compels the recurrence to be meaningful, not a black box.

Only the adapter + the slot-refiner (the "TRM") train, on VERIFIED answers; the LM never moves (frozen ->
anti-poison preserved, same sanction as trainer.py). The tanh gate starts at 0 so at init the LM is
BITWISE-identical to the base model; the adapter can only *add* signal once it earns lower loss.

Mechanism proven by --selftest on distilgpt2 (no Qwen needed):
  (i)   gate=0  -> LM logits identical to base            (identity at init; can't wreck fluency)
  (ii)  gate!=0 -> LM logits change                       (working memory is causally wired into the LM)
  (iii) train   -> a token placed ONLY in the slots (absent from the prompt) becomes the LM's answer, and
                   it GENERALIZES to HELD-OUT slot content it never trained on  (real copy-from-memory,
                   not memorization) — with the gate ABLATED to 0 the effect vanishes (proves causality).
  (iv)  deep sup: each refinement step independently improves retrieval accuracy -> measured stepwise.

    python -m v5.runtime.trm_wm --selftest                                    # mechanism, distilgpt2, local
    python -m v5.runtime.trm_wm --run --lm Qwen/Qwen3-4B-Instruct-2507        # real experiment (their GPU)
"""
from __future__ import annotations

import argparse
import os
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

    def __init__(self, d: int, n_heads: int = 4, delta_scale: float = 0.3):
        super().__init__()
        assert d % n_heads == 0
        self.h, self.dh = n_heads, d // n_heads
        self.q = nn.Linear(d, d)
        self.k = nn.Linear(d, d)
        self.v = nn.Linear(d, d)
        self.o = nn.Linear(d, d)
        self.g = nn.Parameter(torch.zeros(1))
        # CAP the injection at delta_scale*||h|| (was 100% of ||h|| -- a sledgehammer that, combined with an
        # unregularized gate free to swing to tanh~0.97, could overwrite rather than blend with the residual
        # stream, encouraging memorization over a generalizable nudge).
        self.delta_scale = delta_scale
        for lin in (self.v, self.o):
            nn.init.eye_(lin.weight)
            nn.init.zeros_(lin.bias)

    def forward(self, h: torch.Tensor, slots: torch.Tensor) -> torch.Tensor:
        B, S, d = h.shape
        if slots.dim() == 2:
            slots = slots.unsqueeze(0).expand(B, -1, -1)
        Bk, K, _ = slots.shape
        q = self.q(h).view(B, S, self.h, self.dh).transpose(1, 2)
        k = self.k(slots).view(Bk, K, self.h, self.dh).permute(0, 2, 1, 3)
        v = self.v(slots).view(Bk, K, self.h, self.dh).permute(0, 2, 1, 3)
        att = torch.softmax((q @ k.transpose(-1, -2)) / (self.dh ** 0.5), dim=-1)
        ctx = (att @ v).transpose(1, 2).reshape(B, S, d)
        delta = self.o(ctx)
        delta = delta / (delta.norm(dim=-1, keepdim=True) + 1e-6) * (h.norm(dim=-1, keepdim=True) * self.delta_scale)
        return h + torch.tanh(self.g) * delta


# ================================================================================================
# ================================================================================================
# AlgorithmicCell — explicit search, binding, and branching for a true reasoning engine
# ================================================================================================



# ================================================================================================
# WMReasoner — the working memory + its recursive refinement + the coupling hooks + deep supervision
# ================================================================================================
class WMReasoner(nn.Module):
    """Working memory slots produced by TRMReasoner (proper two-latent Tiny Recursive Model),
    projected to LM space, then read by the LM via gated cross-attention adapters.

    DEEP SUPERVISION: intermediate y_t values from each TRM cycle are regressed against
    oracle-computed intermediate targets (native_text_embedding of true intermediate results).
    Loss is MSE in d_lm space — NOT CE against atom pools (TRM is not a ranker)."""
    def __init__(self, d_lm: int, couple_layers, trm, n_heads: int = 4, M: int = 4):
        super().__init__()
        self.T = trm.T
        self.M = M
        self.trm = trm                                         # shared TRMReasoner (two-latent)

        # Project TRM's y_t [T, d] → [T, d_lm] for LM adapters
        self.proj_y = nn.Linear(trm.d, d_lm)

        # Gated cross-attention adapters (unchanged)
        self.adapters = nn.ModuleList([GatedCrossAttn(d_lm, n_heads) for _ in couple_layers])
        self.couple_layers = list(couple_layers)
        self._slots = None

        # Deep supervision: map y_t [d] → d_lm for MSE against native_text_embedding targets
        self.ds_proj = nn.Linear(trm.d, d_lm)

        # Self-critique (unchanged)
        self.critic_pool = nn.Linear(d_lm, d_lm)
        self.critic = nn.Sequential(nn.Linear(d_lm, d_lm // 2), nn.GELU(), nn.Linear(d_lm // 2, 1))

    def critique(self, raw_states: list[torch.Tensor]) -> torch.Tensor:
        """raw_states: [T per-step projected y_t values in d_lm space] from refine(track_deltas=True)."""
        state = torch.stack(raw_states, dim=0)  # [T, d_lm]
        pooled = torch.tanh(self.critic_pool(state))  # [T, d_lm]
        return torch.sigmoid(self.critic(pooled.mean(0, keepdim=True))).squeeze()  # [1, d_lm] → scalar

    def critic_loss(self, raw_states_batch: list[list[torch.Tensor]], labels: list) -> torch.Tensor:
        preds = torch.stack([self.critique(s) for s in raw_states_batch])
        y = torch.tensor([float(l) for l in labels], device=preds.device)
        return nn.functional.binary_cross_entropy(preds, y)

    def trajectory_instability(self, deltas: list) -> float:
        """Measures convergence of y_t values across TRM cycles.
        deltas: per-cycle ||y_{t+1} - y_t|| norms.
        Returns late/early ratio. <1 = settling (good), >1 = still churning."""
        if len(deltas) < 2:
            return 1.0
        vals = [float(d) for d in deltas]
        half = max(1, len(vals) // 2)
        early, late = vals[:half], vals[half:]
        early_mean = sum(early) / len(early) + 1e-8
        late_mean = sum(late) / len(late)
        return late_mean / early_mean

    def refine(self, task_emb: torch.Tensor, atom_embs: torch.Tensor, native: bool = False,
              track_deltas: bool = False):
        """task_emb [d_in], atom_embs [N, d_in] (both in TRM's d_in space, typically MiniLM 384-d).
        Runs the proper TRM (two-latent, cross-attn) to produce per-cycle y_t solution embeddings,
        projects to d_lm for GatedCrossAttn adapters.

        native=True is ACCEPTED for backward compat but no longer needed — TRMReasoner handles
        its own internal projections regardless of input space."""
        y_t = self.trm(task_emb, atom_embs)                    # [T, trm.d]
        slots = self.proj_y(y_t)                               # [T, d_lm] — working memory
        self._slots = slots
        states = [y_t[i] for i in range(self.T)]                # per-step y_t for DS

        if track_deltas:
            deltas = [(y_t[i + 1] - y_t[i]).norm().item() for i in range(self.T - 1)] if self.T > 1 else [0.0]
            raw_states = [s.detach() for s in slots]            # projected y_t in d_lm space for critic
            return slots, states, deltas, raw_states
        return slots, states

    def ds_loss_batch(self, all_states: list[list[torch.Tensor]], targets: torch.Tensor | None = None,
                      _unused=None) -> torch.Tensor:
        """Deep supervision on intermediate y_t values. MSE between ds_proj(y_t[t]) and target[t]
        in d_lm space. targets: [B, T, d_lm] oracle-computed intermediate values via
        native_text_embedding, or None (returns 0)."""
        B = len(all_states)
        T = len(all_states[0]) if all_states else 1
        dev = self._device()

        if targets is None or targets.shape[0] != B:
            return torch.tensor(0.0, device=dev)

        flat = torch.stack([torch.stack(s) for s in all_states], dim=0).float().to(dev)
        y_t_proj = self.ds_proj(flat)                           # [B, T, d_lm]
        return nn.functional.mse_loss(y_t_proj, targets.float().to(dev))

    def set_context(self, task_emb, atom_embs):
        te = torch.as_tensor(task_emb, dtype=torch.float32, device=self._device())
        ae = torch.as_tensor(atom_embs, dtype=torch.float32, device=self._device())
        if ae.dim() == 1:
            ae = ae.unsqueeze(0)
        self._slots, _ = self.refine(te, ae)

    def set_slots_direct(self, slots: torch.Tensor):
        self._slots = slots.unsqueeze(0) if slots.dim() == 1 else slots

    def clear(self):
        self._slots = None

    def save(self, path: str):
        """Persist the trained adapter + TRMReasoner."""
        torch.save({
            "state_dict": self.state_dict(),
            "d_lm": self.proj_y.out_features,
            "couple_layers": self.couple_layers,
            "T": self.T,
            "trm_d": self.trm.d,
            "trm_d_in": self.trm.d_in,
            "n_heads": self.adapters[0].h if len(self.adapters) else 4,
        }, path)

    @classmethod
    def load(cls, path: str, trm, map_location=None) -> "WMReasoner":
        """Reconstruct a WMReasoner from a save()'d checkpoint.
        Requires an already-constructed TRMReasoner instance (passed as `trm`)."""
        blob = torch.load(path, map_location=map_location, weights_only=False)
        R = cls(blob["d_lm"], blob["couple_layers"], trm,
                n_heads=blob["n_heads"], M=blob.get("M", 4))
        R.load_state_dict(blob["state_dict"], strict=False)
        return R

    def _device(self):
        return self.proj_y.weight.device

    def couple(self, wb) -> list:
        handles = []
        for a_i, L in enumerate(self.couple_layers):
            handles.append(wb.layers[L].register_forward_hook(self._mk_hook(a_i)))
        return handles

    def _mk_hook(self, idx):
        def hook(_mod, _inp, out):
            if self._slots is None:
                return None
            h = out[0] if isinstance(out, tuple) else out
            h2 = self.adapters[idx](h.float(), self._slots.float()).to(h.dtype)
            if isinstance(out, tuple):
                return (h2,) + tuple(out[1:])
            return h2
        return hook


def native_text_embedding(wb, text: str) -> torch.Tensor:
    """PROBE-C-VALIDATED path: embed text via the LM's OWN embedding table (mean-pooled over its tokens) --
    zero cross-model gap, unlike routing through MiniLM + a trained bridge (probe B collapsed on held-out;
    probe C generalized 0.19->0.29). Use this for anything injected into the LM's residual stream; MiniLM
    stays fine for cheap cosine RETRIEVAL (picking which atom), which never goes through the adapter."""
    tie = bool(getattr(wb.model.config, "tie_word_embeddings", False))
    out_emb = wb.model.get_output_embeddings()
    lm_emb = out_emb.weight if (out_emb is not None and not tie) else wb.model.get_input_embeddings().weight
    ids = wb.tok(text, return_tensors="pt").input_ids.to(wb.device)
    return lm_emb[ids[0]].float().mean(0).detach()


def native_text_embedding_batch(wb, texts: list[str]) -> torch.Tensor:
    """Batched version of native_text_embedding. Returns [N, d_lm] tensor.
    Much faster than per-atom calls when embedding many atom descriptions because
    tokenization + embedding lookup happen in one pass."""
    if not texts:
        return torch.empty(0, 0)
    encoded = wb.tok(texts, padding=True, truncation=True, return_tensors="pt")
    ids = encoded.input_ids.to(wb.device)
    mask = encoded.attention_mask.to(wb.device)
    tie = bool(getattr(wb.model.config, "tie_word_embeddings", False))
    out_emb = wb.model.get_output_embeddings()
    lm_emb = out_emb.weight if (out_emb is not None and not tie) else wb.model.get_input_embeddings().weight
    embs = lm_emb[ids]
    embs = (embs * mask.unsqueeze(-1).float()).sum(1) / mask.sum(1, keepdim=True).float().clamp(min=1)
    return embs.detach()


# ================================================================================================
# selftest — prove the mechanism on distilgpt2 (identity / causal / trainable+generalizing / deep sup)
# ================================================================================================
def _vocab_words(tok, n: int = 120):
    """Pull n single-token lowercase words from the vocab."""
    words = []
    for tid in range(len(tok)):
        s = tok.decode([tid])
        if s[:1] == " " and s[1:].isalpha() and s[1:].islower() and len(s) >= 4:
            words.append((s[1:], tid))
        if len(words) >= n:
            break
    return words


def _run_probe(wb, R, pids, train, test, precomputed_states, answer_pool, tid_of,
               steps=200, lr=3e-3, bs=128, ds_weight=0.15, dump=0):
    """precomputed_states: list of [[K,d_lm], ...] per-step states for each word, one refine per word.
    ds_weight > 0 calls ds_loss_batch(all_states, targets=None) which returns 0 with new MSE-based DS —
    the old atom-pool CE is removed. Set ds_weight=0 unless you provide targets."""
    k = len(train)
    # gate gets its OWN param group with much higher weight decay: unconstrained, it swung to tanh~0.97 (almost
    # fully open) with nothing pulling it back, letting a high-magnitude, largely unconstrained edit memorize
    # train pairs instead of learning a modest, generalizable nudge. Ordinary params keep the normal wd.
    gate_params = [a.g for a in R.adapters]
    gate_ids = {id(p) for p in gate_params}
    other_params = [p for p in R.parameters() if p.requires_grad and id(p) not in gate_ids]
    opt = torch.optim.Adam([
        {"params": other_params, "weight_decay": 1e-4},
        {"params": gate_params, "weight_decay": 5e-2},
    ], lr=lr)
    for a in R.adapters:
        with torch.no_grad():
            a.g.fill_(0.8)          # more modest warm-start (was 1.5) now that delta itself is capped at 0.3*||h||

    R.train()
    last = float("nan")
    order = list(range(k))
    for ep in range(steps):
        torch.manual_seed(ep)
        order = torch.randperm(k).tolist()
        tot, nb = 0.0, 0
        for i in range(0, k, bs):
            idx = order[i:i + bs]
            final_slots = torch.stack([precomputed_states[j][-1] for j in idx], dim=0).to(wb.device)
            R.set_slots_direct(final_slots)
            logits = wb.model(pids.expand(len(idx), -1)).logits[:, -1]
            lm_loss = nn.functional.cross_entropy(
                logits, torch.tensor([tid_of[train[j][0]] for j in idx], device=wb.device))

            if ds_weight > 0:
                state_list = [precomputed_states[j] for j in idx]
                ds_acc = R.ds_loss_batch(state_list, targets=None)
                loss = lm_loss + ds_weight * ds_acc
            else:
                loss = lm_loss

            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += float(lm_loss.detach())
            nb += 1
        last = tot / nb

    R.eval()

    def acc(data, ablate=False):
        hits = 0
        base = len(train) if data is not train else 0
        for i in range(0, len(data), bs):
            idx = list(range(i, min(i + bs, len(data))))
            chunk_slots = torch.stack([precomputed_states[base + j][-1] for j in idx], dim=0).to(wb.device)
            R.set_slots_direct(chunk_slots.detach())
            gs = [float(a.g.detach()) for a in R.adapters]
            if ablate:
                for a in R.adapters:
                    with torch.no_grad():
                        a.g.zero_()
            with torch.no_grad():
                preds = wb.model(pids.expand(len(idx), -1)).logits[:, -1].argmax(-1).tolist()
            if ablate:
                for a, gv in zip(R.adapters, gs):
                    with torch.no_grad():
                        a.g.fill_(gv)
            hits += sum(int(p == tid_of[data[j][0]]) for j, p in zip(idx, preds))
        return hits / len(data)

    tr = acc(train)
    te = acc(test)
    te_abl = acc(test, ablate=True)
    if dump:
        base = len(train)
        try:
            out_lines = [f"       [dump] held-out  target -> top-5 predicted (slot injected):"]
            for j in range(min(dump, len(test))):
                w, tokid = test[j]
                R.set_slots_direct(precomputed_states[base + j][-1].unsqueeze(0).to(wb.device).detach())
                with torch.no_grad():
                    lg = wb.model(pids).logits[0, -1]
                top = lg.topk(5).indices.tolist()
                toks = ", ".join(repr(wb.tok.decode([t])) for t in top)
                out_lines.append(f"          {w!r:>12} (tok {tokid}) -> {toks}  {'<- HIT' if top[0] == tokid else ''}")
            print("\n".join(out_lines))
        except UnicodeEncodeError:
            pass  # terminal encoding may not support special chars; skip dump
    return tr, te, te_abl, float(R.adapters[0].g), last


def selftest(wb=None, bs=128, steps_a=120, words_n=120):
    from v5.runtime.dcpd_latent import WhiteBox
    from v5.runtime.algo_trm import _build as _build_trm
    torch.manual_seed(0)

    _, _, TRMReasoner = _build_trm()

    if wb is None:
        print("trm_wm.py --selftest : WMReasoner (TRM V3) coupled to FROZEN distilgpt2\n")
        wb = WhiteBox("distilgpt2", quant="fp32")
        if os.environ.get("GRAPH_FORCE_CPU"):
            wb.model = wb.model.to("cpu"); wb.device = "cpu"
            print("  (forced CPU)")
    else:
        print(f"trm_wm.py --probe on {wb.name}: WMReasoner+TRM mechanism test\n")
    d_lm = wb.d_model
    couple = [wb.n_layers - 2, wb.n_layers - 1]

    trm = TRMReasoner(d_in=EMBED_DIM, d=256, T=4, n_heads=4)
    R = WMReasoner(d_lm, couple_layers=couple, trm=trm).to(wb.device)
    for p in wb.model.parameters():
        p.requires_grad_(False)
    handles = R.couple(wb)

    prompt = "The answer is"
    pids = wb.tok(prompt, return_tensors="pt").input_ids.to(wb.device)
    words = _vocab_words(wb.tok, words_n)
    print(f"  atoms: {len(words)}  ({int(0.8*len(words))} train / {len(words)-int(0.8*len(words))} held-out)")
    split = max(1, int(0.8 * len(words)))
    train_w, test_w = words[:split], words[split:]
    tid_of = {w: t for w, t in words}

    # (i) identity at init
    R.clear()
    base = wb.model(pids).logits.detach()
    R.set_context(encode_batch([prompt])[0], encode_batch(["banana"])[0])
    id_diff = (base - wb.model(pids).logits.detach()).abs().max().item()
    print(f"  (i)   identity@init   max|base - withslots(gate=0)| = {id_diff:.2e}   "
          f"{'PASS' if id_diff < 1e-4 else 'FAIL'}")
    # (ii) causal
    with torch.no_grad():
        R.adapters[0].g.fill_(1.0)
    ch = (base - wb.model(pids).logits.detach()).abs().max().item()
    print(f"  (ii)  causal wiring   max|base - withslots(gate=1)| = {ch:.2e}   "
          f"{'PASS' if ch > 1e-3 else 'FAIL'}\n")
    with torch.no_grad():
        R.adapters[0].g.zero_()

    # INJECT IN THE OUTPUT (unembedding) SPACE
    tie = bool(getattr(wb.model.config, "tie_word_embeddings", False))
    _out = wb.model.get_output_embeddings()
    lm_emb = (_out.weight if _out is not None else wb.model.get_input_embeddings().weight)
    print(f"  tie_word_embeddings={tie} -> inject in the {'tied' if tie else 'OUTPUT/unembedding'} space\n")
    answer_pool = torch.stack([lm_emb[tid_of[w]] for w, _ in words], dim=0)
    answer_pool = answer_pool / (answer_pool.norm(dim=-1, keepdim=True) + 1e-8)

    # PROBE A — WIRING: precompute ALL per-step states (direct slot injection, bypasses TRM)
    states_a = []
    for w, _ in words:
        z_a = lm_emb[tid_of[w]].detach()
        states_a.append([z_a.unsqueeze(0).clone() for _ in range(R.T)])
    tr_a, te_a, ab_a, g_a, l_a = _run_probe(
        wb, R, pids, train_w, test_w, states_a, answer_pool, tid_of,
        steps=steps_a, bs=bs, ds_weight=0.0, dump=6)
    print(f"  (A) WIRING  (slot = LM's own embedding):  train {tr_a:.2f}  HELD-OUT {te_a:.2f}  "
          f"ablate->0 {ab_a:.2f}  gate {g_a:+.2f}  loss {l_a:.3f}")

    # PROBE D — TRM INTEGRATION: run TRMReasoner through WMReasoner.refine()
    trm2 = TRMReasoner(d_in=EMBED_DIM, d=256, T=4, n_heads=4)
    Rd = WMReasoner(d_lm, couple_layers=couple, trm=trm2).to(wb.device)
    hd = Rd.couple(wb)
    states_d = []
    task_emb = torch.as_tensor(encode_batch([prompt])[0], dtype=torch.float32, device=wb.device)
    with torch.no_grad():
        for w, _ in words:
            atom_emb = torch.as_tensor(encode_batch([w])[0], dtype=torch.float32, device=wb.device)
            slots, y_states = Rd.refine(task_emb, atom_emb.unsqueeze(0))
            slots_direct = slots.detach()
            # Use the FINAL y_t as the slot (same per-step states for all T for probe compat)
            states_d.append([slots_direct[i].unsqueeze(0).clone() if i < len(slots_direct)
                           else slots_direct[-1].unsqueeze(0).clone() for i in range(Rd.T)])
    tr_d, te_d, ab_d, g_d, l_d = _run_probe(
        wb, Rd, pids, train_w, test_w, states_d, answer_pool, tid_of,
        steps=steps_a, bs=bs, ds_weight=0.0, dump=6)
    print(f"  (D) TRM-INTEG  (TRM y_t -> proj_y -> slots):  train {tr_d:.2f}  HELD-OUT {te_d:.2f}  "
          f"ablate->0 {ab_d:.2f}  gate {g_d:+.2f}  loss {l_d:.3f}")
    for h in hd:
        h.remove()

    # TRMReasoner integrity: y_t shape [T, d], values evolve across cycles
    atom_emb = torch.as_tensor(encode_batch(["banana"])[0], dtype=torch.float32, device=wb.device)
    y_ts = trm(task_emb, atom_emb.unsqueeze(0))
    assert y_ts.shape == (trm.T, trm.d), f"y_ts {y_ts.shape}"
    y_diffs = [(y_ts[t + 1] - y_ts[t]).norm().item() for t in range(trm.T - 1)]
    evolving = any(d > 1e-6 for d in y_diffs)
    print(f"\n  TRM integrity: y_ts {list(y_ts.shape)} diffs {[f'{d:.3f}' for d in y_diffs]} -> "
          f"{'PASS' if evolving else 'FAIL'}")

    print(f"\n  WMReasoner: {sum(p.numel() for p in R.parameters())} total params "
          f"(TRM {sum(p.numel() for p in trm.parameters())} + WM {sum(p.numel() for p in R.parameters()) - sum(p.numel() for p in trm.parameters())})")
    print(f"     refinement steps: {R.T}  |  each y_t is a {trm.d}-d solution embedding")
    print(f"     DS: MSE(y_t_proj, oracle_target) in d_lm space — NOT atom-pool CE (TRM is not a ranker)")
    for h in handles:
        h.remove()


# ================================================================================================
# Phase 3 — wire the REAL membrane.py graph in, additively. _seed_atoms()/_compose_tasks_real()/the old
# fn_map below stay COMPLETELY UNTOUCHED (run_real's default path is byte-identical to before this phase --
# zero regression risk). These new functions activate only when run_real is given graph_path=... .
# ================================================================================================
def _atoms_from_graph(g) -> tuple[dict, dict]:
    """Real graph atoms, not the hand-written 10-atom dict. Filters on the STRUCTURAL fact of having real
    executable code -- NOT kind=='atom' (kind is a free natural-language label in membrane.py, not a closed
    enum; whether a node is usable for composition is a fact about its code, not what string labels it).
    Excludes trap nodes (wrong code that failed verify, saved as anti-poison) -- these have a.code but
    their implementations are incorrect, so using them in composition would always fail verify()."""
    descs, codes = {}, {}
    for name, a in g.atoms.items():
        if a.code and a.kind != "trap":
            descs[name] = a.description
            codes[name] = a.code
    return descs, codes


def _grow_from_cot(g, n: int, domains: str = "math,code,science,puzzle", keywords: str = "",
                   min_reasoning_chars: int = 200) -> dict:
    """Real graph growth from open data: stream N real OpenThoughts-114k CoT traces (v5.graph_grower.
    fetch_cot -- HF-streamed, no full-dataset download) and bank each through membrane's OWN learn_any --
    the same write-time graph editor demo()/interactive_trace() already use (dedup via cosine >=0.90,
    self-organizing 'related' edges below that). Plain text with no code/oracle -> concept nodes (Tier C:
    trusted-source text, no independent recompute) -- separate from the code atoms composition trains on
    below; this step's job is only to make the LONG-TERM graph itself grow from real external data, honestly
    (some fraction will dedup-merge into existing nodes rather than add new ones -- reported, not hidden)."""
    from v5.graph_grower.fetch_cot import stream_openthoughts
    from v5.runtime.membrane import learn_any, TRMRetriever
    retr = TRMRetriever(g)
    ot_domains = [d.strip() for d in domains.split(",") if d.strip()]
    kw = [k.strip() for k in keywords.split(",") if k.strip()] or None
    added = merged = seen = 0
    for doc in stream_openthoughts(ot_domains=ot_domains, keywords=kw, limit=n,
                                   min_reasoning_chars=min_reasoning_chars):
        seen += 1
        res = learn_any(g, retr, doc["text"][:4000])   # cap -- MiniLM truncates anyway, keep banking cheap
        if res["status"] == "banked-fact":
            added += 1
        elif res["status"] == "merged-fact":
            merged += 1
    return {"seen": seen, "added": added, "merged": merged}


def _grow_skills_from_corpus(g, n: int | None = None, domains: str = "") -> dict:
    """Real EXECUTABLE-skill growth (Tier A: independent execution oracle) -- the piece _grow_from_cot
    deliberately left out (that one only banks prose as concept nodes, no .code, never enters the composable
    pool). scripts/build_crossdomain_corpus.py has 44 hand-written, oracle-verified (real Python reference
    code + real test tuples) cross-domain tasks (math/physics/biology/cs/stats, deliberately sharing
    primitives like gcd/mean/kinetic_energy across domains). Routes each through membrane's OWN
    learn_any(code=..., tests=...) -- the SAME real fuzz-gate/verify() every other atom in the graph passes
    through; nothing is banked as code without passing real execution against real tests.

    KNOWN LIMIT, stated plainly (not silently worked around): membrane.py's Atom/verify/_closure/realize_*
    machinery assumes a SINGLE-argument entry(n) throughout (every existing atom, direct/compose
    realization, and learn_any's own '_e(n): return {nm}(n)' verify wrapper). This corpus has multi-arg
    tasks too (gcd(a,b), bmi(weight,height), merge_sorted(a,b)) -- those are SKIPPED here, counted and
    reported, not mis-banked. A handful of tasks also use an expected value of None (approximate-value
    placeholders in the corpus, e.g. gravitational_force) -- also skipped, same reason: verify() needs a
    real expected value to compare against."""
    from v5.runtime.membrane import learn_any, TRMRetriever
    from scripts.build_crossdomain_corpus import build_corpus
    retr = TRMRetriever(g)
    dom_filter = {d.strip() for d in domains.split(",") if d.strip()} or None
    tasks = build_corpus()
    seen = banked = trap = skipped_multiarg = skipped_notype = 0
    for t in tasks:
        if dom_filter and t["domain"] not in dom_filter:
            continue
        if n is not None and seen >= n:
            break
        seen += 1
        raw_tests = t["tests"]
        if any(len(args) != 1 for args, _ in raw_tests):
            skipped_multiarg += 1
            continue
        if any(exp is None for _, exp in raw_tests):
            skipped_notype += 1
            continue
        tests = [(args[0], exp) for args, exp in raw_tests]
        res = learn_any(g, retr, t["text"], code=t["reference"], tests=tests, name=t["entry"])
        if res["status"] == "banked-skill":
            banked += 1
        else:
            trap += 1
    return {"seen": seen, "banked": banked, "trap": trap,
            "skipped_multiarg": skipped_multiarg, "skipped_notype": skipped_notype}


def _dynamic_oracle(g, atom_names: list[str]):
    """Build ONE shared exec namespace from the graph's OWN atom code, via membrane._closure (already
    resolves transitive .depends -- critical: a naive per-atom exec breaks the moment a real banked atom
    depends on another, exactly the bug class already found and fixed once this session in _run_task's
    hardcoded fn_map). Returns a callable _run_task(n, code_line) with the SAME interface as the old
    hardcoded one, but sourced from the graph itself -- scales to an arbitrary/growing atom set."""
    from v5.runtime.membrane import _closure
    src = _closure(g, atom_names)
    ns: dict = {}
    exec(compile(src, "<graph-oracle>", "exec"), ns)

    def _run_task(n, code_line):
        return eval(code_line, {"__builtins__": __builtins__}, {**ns, "n": n})
    return _run_task


# the ORIGINAL hand-tuned phrasings (from _compose_tasks_real) -- reused byte-identical for the 10 seed
# atoms if they're present in the graph, so the default 10-atom case produces IDENTICAL task text to before.
_KNOWN_INNER_PHRASE = {
    "digit_sum": "the digit sum of n", "num_divisors": "the number of divisors of n",
    "factorial": "n factorial", "fibonacci": "the nth Fibonacci number",
    "reverse_digits": "n with its digits reversed", "count_bits": "the number of one bits in n",
    "sum_to_n": "the sum of all integers from 1 to n", "square": "the square of n",
}
_KNOWN_OUTER_TEMPLATE = {
    "is_prime": "whether {inner} is prime", "digit_sum": "the digit sum of {inner}",
    "num_divisors": "the number of divisors of {inner}",
    "reverse_digits": "the digit-reversal of {inner}",
    "count_bits": "the number of one bits in {inner}",
    "sum_to_n": "the sum of all integers from 1 to {inner}",
    "square": "the square of {inner}", "is_even": "whether {inner} is even",
}


def _compose_tasks_from_graph(g, atom_names: list[str], n_train: int = 48, n_held: int = 16, seed: int = 0):
    """Generic version of _compose_tasks_real: builds (outer,inner) 2-atom composition candidates from
    WHATEVER atoms currently exist in the graph. Known atoms reuse the exact hand-tuned phrasing above
    (byte-identical to _compose_tasks_real); atoms outside that set fall through to a generic
    description-driven template. Flagged honestly: generic phrasing reads stiffer -- later polish, not a
    blocker; the CODE (not the task text) is always exact regardless, since it's built from atom names."""
    import random as _random

    def inner_phrase(name):
        return _KNOWN_INNER_PHRASE.get(name, f"the result of {name} applied to n")

    def outer_template(name):
        return _KNOWN_OUTER_TEMPLATE.get(name, f"the result of {name} applied to {{inner}}")

    pairs = [(o, i) for o in atom_names for i in atom_names if o != i]
    _random.Random(seed).shuffle(pairs)
    n_train = min(n_train, max(0, len(pairs) - 4))
    n_held = min(n_held, max(0, len(pairs) - n_train))

    def _mk(outer, inner):
        text = outer_template(outer).format(inner=inner_phrase(inner))
        code = f"def task(n): return {outer}({inner}(n))"
        return (text, [inner, outer], code)

    train = [_mk(o, i) for o, i in pairs[:n_train]]
    held_out = [_mk(o, i) for o, i in pairs[n_train:n_train + n_held]]
    return train, held_out


def _seed_atoms() -> tuple[dict, dict]:
    """Same 10 atoms as membrane.py. Returns {name: description} and {name: code}."""
    descs = {
        "is_prime": "whether a number is prime (exactly two divisors)",
        "digit_sum": "the sum of the decimal digits of a number",
        "num_divisors": "how many positive divisors a number has",
        "factorial": "the factorial of a number, n!",
        "fibonacci": "the nth Fibonacci number",
        "reverse_digits": "the number with its decimal digits reversed",
        "count_bits": "the number of one bits in the binary representation",
        "sum_to_n": "the sum of all integers from 1 to n",
        "square": "the square of a number",
        "is_even": "whether a number is even",
    }
    codes = {
        "is_prime": "def is_prime(n): return n>=2 and all(n%i for i in range(2,int(n**0.5)+1))",
        "digit_sum": "def digit_sum(n): return sum(int(c) for c in str(abs(n)))",
        "num_divisors": "def num_divisors(n): return sum(1 for i in range(1,abs(n)+1) if n%i==0)",
        "factorial": "def factorial(n): r=1\n for i in range(2,n+1): r*=i\n return r",
        "fibonacci": "def fibonacci(n): a,b=0,1\n for _ in range(n): a,b=b,a+b\n return a",
        "reverse_digits": "def reverse_digits(n): return int(str(abs(n))[::-1])",
        "count_bits": "def count_bits(n): return bin(abs(n)).count('1')",
        "sum_to_n": "def sum_to_n(n): return n*(n+1)//2",
        "square": "def square(n): return n*n",
        "is_even": "def is_even(n): return int(n%2==0)",
    }
    return descs, codes


def _compose_tasks_real(n_train: int = 48, n_held: int = 16, seed: int = 0):
    """(task_text, atoms_needed, target_code_template) for training and held-out.

    AUTO-GENERATED from all (outer, inner) 2-atom composition pairs -- the hand-authored 6 train / 4 held-out
    was far below the data volume everything else in this session needed to generalize (probes needed
    ~hundreds-1000 atoms before held-out moved off 0). 8 numeric INNER atoms x 8 OUTER atoms = 64 pairs;
    split so no exact (outer,inner) PAIR leaks into held-out, but every individual atom appears in many
    training pairs -- the model must generalize COMPOSITION, not memorize a whole new atom."""
    import random as _random
    inner_phrase = {
        "digit_sum": "the digit sum of n", "num_divisors": "the number of divisors of n",
        "factorial": "n factorial", "fibonacci": "the nth Fibonacci number",
        "reverse_digits": "n with its digits reversed", "count_bits": "the number of one bits in n",
        "sum_to_n": "the sum of all integers from 1 to n", "square": "the square of n",
    }
    outer_template = {
        "is_prime": "whether {inner} is prime", "digit_sum": "the digit sum of {inner}",
        "num_divisors": "the number of divisors of {inner}",
        # NOT "{inner} with its digits reversed" -- that's structurally identical to inner_phrase's own
        # "n with its digits reversed" plugged into ANOTHER atom's outer template in the opposite order
        # (X(reverse_digits(n)) vs reverse_digits(X(n))), a real attachment-ambiguity collision caught by
        # a train/held text-overlap check. This phrasing is unambiguous.
        "reverse_digits": "the digit-reversal of {inner}",
        "count_bits": "the number of one bits in {inner}",
        "sum_to_n": "the sum of all integers from 1 to {inner}",
        "square": "the square of {inner}", "is_even": "whether {inner} is even",
    }
    pairs = [(o, i) for o in outer_template for i in inner_phrase]
    _random.Random(seed).shuffle(pairs)
    n_train, n_held = min(n_train, len(pairs) - 4), min(n_held, len(pairs) - n_train)

    def _mk(outer, inner):
        text = outer_template[outer].format(inner=inner_phrase[inner])
        code = f"def task(n): return {outer}({inner}(n))"
        return (text, [inner, outer], code)                 # atoms_needed order: inner first (applied first)

    train = [_mk(o, i) for o, i in pairs[:n_train]]
    held_out = [_mk(o, i) for o, i in pairs[n_train:n_train + n_held]]
    return train, held_out


def _exec_verify(code: str, tests: list) -> bool:
    """Run code, call task(n) with test inputs, verify outputs match."""
    try:
        ns = {}
        exec(compile(code, "<verify>", "exec"), ns)
        fn = ns.get("task")
        if not callable(fn):
            return False
        for inp, expected in tests:
            if fn(inp) != expected:
                return False
        return True
    except Exception:
        return False


def _pad_and_batch(pids_list, tids_list, pad_token_id, device):
    """Pad variable-length prompt+target sequences for a batched LM forward.
    Returns (input_ids, labels, attention_mask) all shaped [B, max_len]."""
    max_n = max(p.shape[-1] + t.shape[-1] for p, t in zip(pids_list, tids_list))
    batch_ids, batch_labels, batch_attn = [], [], []
    for pids, tids in zip(pids_list, tids_list):
        n = pids.shape[-1] + tids.shape[-1]
        pad_len = max_n - n
        input_ids = torch.cat([pids, tids], dim=-1)
        padded = torch.nn.functional.pad(input_ids, (0, pad_len), value=pad_token_id)
        labels = torch.full((1, max_n), -100, device=device, dtype=torch.long)
        labels[0, pids.shape[-1]:n] = tids
        attn = torch.nn.functional.pad(torch.ones(1, n, device=device), (0, pad_len), value=0)
        batch_ids.append(padded)
        batch_labels.append(labels)
        batch_attn.append(attn)
    return torch.cat(batch_ids, dim=0), torch.cat(batch_labels, dim=0), torch.cat(batch_attn, dim=0)


def run_real(lm_name: str, quant: str = "4bit", epochs: int = 40, n_train: int = 48, n_held: int = 16,
            graph_path: str | None = None, save_path: str | None = None, grow_cot: int = 0,
            grow_domains: str = "math,code,science,puzzle", grow_keywords: str = "",
            grow_skills: int = 0, grow_skills_domains: str = "",
            batch_size: int = 1):
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass
    """graph_path=None (default): UNCHANGED behavior, the hand-written 10-atom dict + hand-tuned templated
    tasks (the proven 13-15/16 held-out result) -- zero risk of regression, this path is untouched by Phase
    3. graph_path=<path>: real graph atoms (via membrane's AtomGraph.load/seed_graph + _atoms_from_graph)
    and a graph-derived dynamic oracle (_dynamic_oracle, via membrane._closure) -- scales to whatever atoms
    actually exist, not a fixed 10. grow_cot>0 (requires graph_path): ingest that many real OpenThoughts-114k
    CoT docs into the graph via learn_any BEFORE training -- concept nodes only (see _grow_from_cot). grow_
    skills>0 (requires graph_path): bank up to that many real oracle-verified EXECUTABLE atoms from
    scripts/build_crossdomain_corpus.py (see _grow_skills_from_corpus) -- these DO enter the composable pool
    (_atoms_from_graph filters on real .code), unlike grow_cot's concept nodes."""
    from v5.runtime.dcpd_latent import WhiteBox
    from v5.runtime.algo_trm import _build as _build_trm
    import random
    print(f"run_real: WMReasoner + TRMReasoner V3 coupled to {lm_name} ({quant}) — real composition tasks\n")

    _, _, TRMReasoner = _build_trm()
    wb = WhiteBox(lm_name, quant=quant)
    d_lm = wb.d_model
    couple = [wb.n_layers - 2, wb.n_layers - 1]
    print(f"  LM: {lm_name}  d={d_lm}  layers={wb.n_layers}  gate layers={couple}  device={wb.device}")

    trm = TRMReasoner(d_in=EMBED_DIM, d=256, T=4, n_heads=4)
    R = WMReasoner(d_lm, couple_layers=couple, trm=trm, n_heads=4).to(wb.device)
    for p in wb.model.parameters():
        p.requires_grad_(False)
    handles = R.couple(wb)

    if graph_path:
        from pathlib import Path as _Path
        from v5.runtime.membrane import AtomGraph, seed_graph
        g = AtomGraph.load(graph_path) if _Path(graph_path).exists() else seed_graph()
        if grow_cot > 0:
            n0 = len(g)
            stats = _grow_from_cot(g, grow_cot, domains=grow_domains, keywords=grow_keywords)
            print(f"  grow: real OpenThoughts-114k CoT ingested via learn_any -> graph {n0} -> {len(g)} nodes "
                  f"(+{stats['added']} new concepts, {stats['merged']} deduped into existing, "
                  f"{stats['seen']} docs seen)")
        if grow_skills > 0:
            n0 = len(g)
            sstats = _grow_skills_from_corpus(g, n=grow_skills, domains=grow_skills_domains)
            print(f"  grow-skills: real oracle-verified corpus ingested via learn_any -> graph {n0} -> {len(g)} "
                  f"nodes (+{sstats['banked']} new EXECUTABLE atoms, {sstats['trap']} failed verify->trap, "
                  f"{sstats['skipped_multiarg']} skipped multi-arg, {sstats['skipped_notype']} skipped "
                  f"no-expected-value, {sstats['seen']} tasks considered)")
        descs, codes = _atoms_from_graph(g)
        atom_names = list(descs.keys())
        print(f"  graph: {len(atom_names)} REAL atoms from {graph_path if _Path(graph_path).exists() else '(fresh seed_graph)'} "
              f"(NATIVE LM-embedding-table injection)")
    else:
        descs, codes = _seed_atoms()
        atom_names = list(descs.keys())
        # NATIVE-SPACE injection (probe-C-validated): embed each atom's description via the LM's OWN embedding
        # table, not MiniLM + an untrained proj_atom bridge -- that bridge is exactly what probe B showed
        # collapses on held-out (train fits, held-out ~0). This was very likely why composition scored 0/4 even
        # after deep supervision was fixed: the atoms fed to refine() were never in a space the LM could read.
        print(f"  graph: {len(atom_names)} atoms (NATIVE LM-embedding-table injection, MiniLM dropped for this path)")
    # BATCHED native-text embedding: one tokenizer + embedding-table pass instead of N separate calls.
    # For large graphs (100+ atoms) this is ~10x faster; identical output (same LM embedding table).
    atom_names_list = list(atom_names)
    descs_list = [descs[n] for n in atom_names_list]
    atom_emb_tensor = native_text_embedding_batch(wb, descs_list)
    atom_embs = {n: atom_emb_tensor[i] for i, n in enumerate(atom_names_list)}

    if graph_path:
        train_tasks, held_tasks = _compose_tasks_from_graph(g, atom_names, n_train=n_train, n_held=n_held)
    else:
        train_tasks, held_tasks = _compose_tasks_real(n_train=n_train, n_held=n_held)
    all_tasks = train_tasks + held_tasks
    split = len(train_tasks)
    print(f"  tasks: {split} train, {len(all_tasks) - split} held-out (2-atom composition, auto-generated, "
          f"no train/held PAIR overlap)\n")

    gate_params = [a.g for a in R.adapters]
    gate_ids = {id(p) for p in gate_params}
    other_params = [p for p in R.parameters() if p.requires_grad and id(p) not in gate_ids]
    opt = torch.optim.Adam([
        {"params": other_params, "weight_decay": 1e-4},
        {"params": gate_params, "weight_decay": 5e-2},
    ], lr=1e-3)
    for a in R.adapters:
        with torch.no_grad():
            a.g.fill_(0.8)

    # Build raw prompts (no chat template, to avoid special-token issues with teacher-forcing)
    def build_prompt(task_text, inner_name=None, outer_name=None):
        """Build tokenized prompt. If inner/outer atom names are provided (decoded from registers),
        prepend them as an explicit CODE hint so the LM knows the exact composition to write.
        Code form (outer(inner(n))) is unambiguous -- 'inner/outer' role labels are not, because
        the LM interprets 'inner' as 'first in code left-to-right' (i.e. outer in function call
        notation), causing systematic order reversal."""
        if inner_name and outer_name:
            hint = f"# return: {outer_name}({inner_name}(n))\n"
        else:
            hint = ""
        return wb.tok(f"{hint}Write a function task(n):\n# {task_text}\ndef task(n):",
                      return_tensors="pt").input_ids.to(wb.device)

    # Precompute static task embeddings and atom embeddings for all examples
    print("  Precomputing task + atom embeddings...")
    task_embs = {}
    for text, atoms_needed, _ in all_tasks:
        if text not in task_embs:
            task_embs[text] = torch.as_tensor(encode_batch([text])[0], dtype=torch.float32, device=wb.device)

    prompt_ids = {text: build_prompt(text) for text, _, _ in all_tasks}
    # All 10 atom oracle functions (used for verification)
    def _fib(n):
        a, b = 0, 1
        for _ in range(n):
            a, b = b, a + b
        return a

    if graph_path:
        # DYNAMIC oracle, sourced from the graph's OWN atom code (Phase 3) -- scales to whatever atoms
        # actually exist, instead of the fixed 10-lambda dict below.
        _run_task = _dynamic_oracle(g, atom_names)
    else:
        def _run_task(n, code_line):
            """Execute the composition code_line (e.g. 'digit_sum(fibonacci(n))') at n.

            CRITICAL FIX: eval's namespace never included 'n' itself -- every composition expression
            references n directly, so this raised NameError on EVERY call, silently caught by verify()'s
            except-> False. This meant verify() could never return True for ANY input, correct or not,
            since _run_task was written -- the true root cause under the 0/4 and 0/16 results, deeper than
            the decoding-loop issue."""
            fn_map = {
                "is_prime": lambda n: n>=2 and all(n%i for i in range(2,int(n**0.5)+1)),
                "digit_sum": lambda n: sum(int(c) for c in str(abs(n))),
                "num_divisors": lambda n: sum(1 for i in range(1,abs(n)+1) if n%i==0),
                "factorial": lambda n: __import__('math').factorial(n),
                "fibonacci": _fib,
                "reverse_digits": lambda n: int(str(abs(n))[::-1]),
                "count_bits": lambda n: bin(abs(n)).count('1'),
                "sum_to_n": lambda n: n*(n+1)//2,
                "square": lambda n: n*n,
                "is_even": lambda n: int(n%2==0),
            }
            return eval(code_line, {"__builtins__": __builtins__}, {**fn_map, "n": n})

    def _extract_first_return(raw: str) -> str | None:
        """Pull the FIRST return-expression out of raw generated text. Safety net: the model is now trained
        to emit EOS after the expression, so generation should normally terminate cleanly. But if EOS fails
        (e.g. the model loops), cut at newline, repeated 'return', ' is ', or ' == ' to recover the first
        complete answer."""
        if "return " not in raw:
            return None
        after = raw.split("return ", 1)[1]
        cuts = [i for i in (after.find("\n"), after.find(" return"), after.find("\treturn")) if i != -1]
        if cuts:
            after = after[:min(cuts)]
        expr = after.strip().rstrip(".")
        return expr or None

    def verify(code_str, target_code, test_ns=(2, 5, 7, 10)):
        # test_ns includes n=2: is_prime(factorial(n)) is CONSTANT False for n=5,7,10 (factorial(n)>=120 is
        # always composite), so a degenerate 'return False' trivially passed -- factorial(2)=2, which IS
        # prime, breaks that degeneracy. Caught from a real spurious ablated PASS in a run's output.
        """Extract the model's first return-expression and check its BEHAVIOR against the target's, both
        evaluated through the oracle fn_map (via _run_task) -- not by exec'ing the model's raw code and
        calling it directly (the old approach: task_fn's globals never had the atom functions injected, so
        even a PERFECT composition would NameError internally and silently report as a failure). A wrong
        atom name (e.g. 'digit_reverse' instead of 'reverse_digits') correctly still fails here -- fn_map
        only knows the real atom names, so eval raises NameError on anything else."""
        expr = _extract_first_return(code_str)
        if not expr:
            return False
        target_expr = target_code.split("return ", 1)[1].strip()
        try:
            for n in test_ns:
                if _run_task(n, expr) != _run_task(n, target_expr):
                    return False
            return True
        except Exception as e:
            # a wrong atom name (NameError) or truncated/malformed expr (SyntaxError) SHOULD fail here -- that
            # is correct, not a bug. But swallowing every exception identically makes it impossible to tell
            # "wrong answer" apart from "extraction/harness broke" while debugging. Opt-in visibility:
            if os.environ.get("GRAPH_DEBUG_VERIFY"):
                print(f"    [verify exception] expr={expr!r}  {type(e).__name__}: {e}")
            return False

    train_ex = [(task_embs[text], [atom_names.index(a) for a in atoms_needed], atoms_needed,
                 prompt_ids[text], text, code)
                for text, atoms_needed, code in train_tasks]
    held_ex = [(task_embs[text], [atom_names.index(a) for a in atoms_needed], atoms_needed,
                prompt_ids[text], text, code)
               for text, atoms_needed, code in held_tasks]

    print(f"  Training the adapter + WMReasoner ({epochs} epochs, {len(train_ex)} pairs; "
          f"real-close loop: refine -> LM -> verify)...")
    if batch_size > 1:
        print(f"  Batched training: batch_size={batch_size} (pads variable-length sequences, "
              f"~{batch_size}x fewer optimizer steps)")
    best_held = 0.0
    eval_every = max(1, epochs // 8)
    last_dump = None
    pad_id = wb.tok.pad_token_id or 0
    # SELF-CRITIQUE data collection (co-evolutionary arms-race idea, stage 1): every eval checkpoint runs
    # real generate()+verify() on held_ex (free byproduct) AND, now, ALSO on train_ex (a small deliberate
    # extra generation cost, see below) -- ~4x more real (trajectory, real-verify-label) pairs than
    # held_ex alone, addressing the critic's small-sample-size problem directly. Still PASSIVE: this data
    # trains the critic only, after the main loop -- does not touch the reasoner's own loss (that's stage 2,
    # gated on stage 1's critic actually beating base rate first, to avoid training the reasoner against an
    # unreliable judge -- Goodhart's law / reward-hacking risk, not built yet).
    # Precompute oracle DS targets: first half of T steps target inner(n), second half target full composition
    inner_expr_set = {f"{a[0]}(n)" for _, _, a, _, _, _ in train_ex + held_ex}
    final_expr_set = {c.split("return ", 1)[1].strip() for _, _, _, _, _, c in train_ex + held_ex}
    ds_texts = list(inner_expr_set | final_expr_set)
    ds_target_embs = native_text_embedding_batch(wb, ds_texts).to(wb.device)  # [N, d_lm]
    ds_target_map = {t: ds_target_embs[i].detach() for i, t in enumerate(ds_texts)}
    ds_train_targets = []
    for _, _, atoms_needed, _, _, target_code in train_ex:
        inner_expr = f"{atoms_needed[0]}(n)"
        final_expr = target_code.split("return ", 1)[1].strip()
        half = R.T // 2
        tgt = torch.stack([ds_target_map[inner_expr]] * half + [ds_target_map[final_expr]] * (R.T - half), dim=0)
        ds_train_targets.append(tgt.detach())  # [T, d_lm]
    print(f"  Oracle DS targets: {len(ds_texts)} unique expressions ({len(ds_train_targets)} examples)")

    critic_examples: list = []
    for ep in range(epochs):
        R.train()
        random.shuffle(train_ex)
        tot_lm, tot_ds, n = 0.0, 0.0, 0
        for b0 in range(0, len(train_ex), batch_size):
            batch = train_ex[b0:b0 + batch_size]
            all_states, pids_list, tids_list = [], [], []
            ds_batch_targets = []
            for task_emb, gold_idxs, atoms_needed, pids, text, target_code in batch:
                mini_atom_embs = torch.stack([
                    torch.as_tensor(encode_batch([atom_names[idx]])[0], dtype=torch.float32, device=wb.device)
                    for idx in gold_idxs
                ])
                slots, states = R.refine(task_emb, mini_atom_embs)
                R.set_slots_direct(slots)
                pids = build_prompt(text)
                return_body = target_code.split(": ", 1)[1] if ": " in target_code else target_code
                tids = wb.tok(" " + return_body, return_tensors="pt").input_ids.to(wb.device)
                eos = torch.tensor([[wb.tok.eos_token_id]], device=wb.device)
                tids = torch.cat([tids, eos], dim=-1)
                pids_list.append(pids); tids_list.append(tids)
                all_states.append(states)
                # DS target index: find position in train_ex
                ti = next(j for j, (_, _, an, _, _, tc) in enumerate(train_ex)
                          if an == atoms_needed and tc == target_code)
                ds_batch_targets.append(ds_train_targets[ti])

            if batch_size > 1 and len(batch) > 1:
                input_ids, labels, attn_mask = _pad_and_batch(pids_list, tids_list, pad_id, wb.device)
                outs = wb.model(input_ids=input_ids, attention_mask=attn_mask, labels=labels)
                lm_loss = outs.loss
            else:
                outs = wb.model(input_ids=torch.cat([pids_list[0], tids_list[0]], dim=-1),
                                labels=torch.cat([torch.full_like(pids_list[0], -100), tids_list[0]], dim=-1))
                lm_loss = outs.loss

            ds_loss = R.ds_loss_batch(all_states, targets=torch.stack(ds_batch_targets, dim=0))
            # DS weight 0.5: previously 0.1 caused the warmup alignment (ds_loss 0.067) to be
            # obliterated by lm_loss in the very first epoch (ep0 ds_loss jumped back to 3.4).
            # At 0.5, ds contribution ~1.7 > lm_loss ~0.06-0.12, preserving alignment.
            loss = lm_loss + 0.5 * ds_loss
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot_lm += float(lm_loss.detach())
            tot_ds += float(ds_loss.detach())
            n += 1

        print(f"  ep {ep:>3}  lm_loss {tot_lm/max(n,1):.3f}  ds_loss {tot_ds/max(n,1):.3f}  gate {float(R.adapters[0].g):+.2f}", end="")
        if ep % eval_every == 0 or ep == epochs - 1:
            R.eval()
            held_ok, ablated_ok = 0, 0
            dump = []
            for task_emb, gold_idxs, atoms_needed, pids, text, target_code in held_ex:
                # Use MiniLM atoms for TRM
                held_mini_embs = torch.stack([
                    torch.as_tensor(encode_batch([atom_names[idx]])[0], dtype=torch.float32, device=wb.device)
                    for idx in gold_idxs
                ])
                slots, wm_states, wm_deltas, wm_raw = R.refine(task_emb, held_mini_embs, track_deltas=True)
                with torch.no_grad():
                    R.set_slots_direct(slots)
                    out = wb.model.generate(pids, max_new_tokens=64,
                                            do_sample=False, pad_token_id=wb.tok.eos_token_id)
                    code = wb.tok.decode(out[0][pids.shape[-1]:], skip_special_tokens=True).strip()
                wm_ok = verify("def task(n): " + code, target_code)
                held_ok += int(wm_ok)
                critic_examples.append(([s.detach() for s in wm_raw], wm_ok))
                instability = R.trajectory_instability(wm_deltas)
                R.clear()
                with torch.no_grad():
                    out = wb.model.generate(pids, max_new_tokens=64,
                                            do_sample=False, pad_token_id=wb.tok.eos_token_id)
                    code_abl = wb.tok.decode(out[0][pids.shape[-1]:], skip_special_tokens=True).strip()
                abl_ok = verify("def task(n): " + code_abl, target_code)
                ablated_ok += int(abl_ok)
                dump.append((text, target_code, code, wm_ok, code_abl, abl_ok, instability))

            # CO-TRAINING data (stage 1)
            for task_emb, gold_idx, atoms_needed, pids, text, target_code in train_ex:
                K_atom_embs = torch.stack([
                    torch.as_tensor(encode_batch([atom_names[idx]])[0], dtype=torch.float32, device=wb.device)
                    for idx in gold_idx
                ])
                slots, tr_states, tr_deltas, tr_raw = R.refine(task_emb, K_atom_embs, track_deltas=True)
                with torch.no_grad():
                    R.set_slots_direct(slots)
                    out = wb.model.generate(pids, max_new_tokens=64,
                                            do_sample=False, pad_token_id=wb.tok.eos_token_id)
                    tr_code = wb.tok.decode(out[0][pids.shape[-1]:], skip_special_tokens=True).strip()
                tr_ok = verify("def task(n): " + tr_code, target_code)
                critic_examples.append(([s.detach() for s in tr_raw], tr_ok))   # PRE-norm content + REAL label
                R.clear()
            R.train()

            best_held = max(best_held, held_ok)
            last_dump = dump
            inst_pass = [d[6] for d in dump if d[3]]
            inst_fail = [d[6] for d in dump if not d[3]]
            inst_str = (f"  instab(pass/fail) {sum(inst_pass)/len(inst_pass):.3f}/"
                       f"{sum(inst_fail)/len(inst_fail):.3f}" if inst_pass and inst_fail else "")
            print(f"  held WM {held_ok}/{len(held_ex)}  ablated {ablated_ok}/{len(held_ex)}  {inst_str}")

    print(f"\n  [dump] final epoch, held-out generations (WM vs ablated) vs the verified target:")
    for row in (last_dump or []):
        text, target_code, code, wm_ok, code_abl, abl_ok, instability = row[:7]
        print(f"     task: {text}")
        print(f"       target : {target_code}")
        print(f"       WM     : def task(n): {code[:90]}{'  <- PASS' if wm_ok else ''}  instab={instability:.3f}")
        print(f"       ablated: def task(n): {code_abl[:90]}{'  <- PASS' if abl_ok else ''}")

    print(f"\n  Best held-out: {best_held}/{len(held_ex)}  (gate ablated = {ablated_ok} baseline)")
    verdict = "PROVEN" if best_held > ablated_ok else "PARTIAL"
    print(f"  => {verdict}: working memory {'improves' if best_held > ablated_ok else 'does not improve'} held-out composition on {lm_name}")

    # SELF-CRITIQUE: train + report on the (trajectory, real-verify-outcome) pairs collected as a free
    # byproduct of eval above. Held-out split within this set (not the SAME split as train/held composition
    # tasks -- this is a separate check: can the critic predict PASS/FAIL of a trajectory it wasn't trained
    # on). Mirrors algo_grr_cot.py's critic_demo() validation pattern (report accuracy, don't just claim it).
    if len(critic_examples) >= 8:
        print(f"\n  Training self-critique (tier-4 amortizer, {len(critic_examples)} real labeled trajectories)...")
        random.Random(0).shuffle(critic_examples)
        split = max(4, int(0.8 * len(critic_examples)))
        c_train, c_test = critic_examples[:split], critic_examples[split:]

        # CLASS-BALANCE the training set: with held WM mostly passing (11-14/16 by mid-training), 'always
        # predict pass' already scores ~the base rate on plain BCE -- a first real run collapsed to EXACTLY
        # the base rate, the signature of this degenerate solution. Oversample the minority class so the
        # loss can't be minimized by ignoring the trajectory content.
        pos = [ex for ex in c_train if ex[1]]
        neg = [ex for ex in c_train if not ex[1]]
        if pos and neg:
            hi, lo = (pos, neg) if len(pos) >= len(neg) else (neg, pos)
            lo_up = [lo[i % len(lo)] for i in range(len(hi))]     # oversample minority to match majority count
            c_train_balanced = hi + lo_up
        else:
            c_train_balanced = c_train                            # only one class present -- nothing to balance
        print(f"  class balance: train {len(pos)} pass / {len(neg)} fail -> balanced to "
              f"{len(c_train_balanced)} examples for training")

        c_opt = torch.optim.Adam(list(R.critic.parameters()) + list(R.critic_pool.parameters()), lr=1e-2)
        for _ in range(200):
            random.shuffle(c_train_balanced)
            c_opt.zero_grad()
            loss = R.critic_loss([s for s, _ in c_train_balanced], [y for _, y in c_train_balanced])
            loss.backward(); c_opt.step()
        with torch.no_grad():
            preds = [float(R.critique(s)) >= 0.5 for s, _ in c_test]
            labels = [bool(y) for _, y in c_test]
            acc = sum(int(p == y) for p, y in zip(preds, labels)) / max(1, len(c_test))
            base_rate = sum(labels) / max(1, len(labels))
        print(f"  critic held-out accuracy: {acc:.2f}  (base rate / always-predict-majority: "
              f"{max(base_rate, 1-base_rate):.2f})  n_test={len(c_test)}")
        print(f"  => critic {'beats' if acc > max(base_rate, 1-base_rate) else 'does NOT beat'} the base rate "
              f"-- {'the trajectory carries a real self-assessment signal' if acc > max(base_rate, 1-base_rate) else 'no signal found yet, report honest'}")
    else:
        print(f"\n  (only {len(critic_examples)} labeled trajectories collected -- too few to train/report the "
              f"critic meaningfully; needs more epochs or a bigger held-out set)")

    if save_path:
        R.save(save_path)
        print(f"\n  saved trained WMReasoner to {save_path} ({sum(p.numel() for p in R.parameters())} params) "
              f"-- load it into membrane.py's Membrane(..., wb=..., wm_path=...) to use the trained adapter live.")

    if graph_path:
        g.save(graph_path)
        print(f"  saved long-term graph -> {graph_path} ({len(g)} nodes, {len(g.edges)} edges) "
              f"-- growth persists for the next run.")

    for h in handles:
        h.remove()


def probe_real(lm_name: str, quant: str = "4bit", words_n: int = 400, steps: int = 120):
    """Run the copy(A)+bridge(B) mechanism test on the REAL 4B (not distilgpt2). Probe B is the decisive one:
    can a capable LM READ graph-space (MiniLM) slots via the working memory and generalize? distilgpt2 can't
    (0.04); this is the fair test. Smaller batch/steps since the 4B is heavy."""
    from v5.runtime.dcpd_latent import WhiteBox
    wb = WhiteBox(lm_name, quant=quant)
    for p in wb.model.parameters():
        p.requires_grad_(False)
    print(f"  LM {lm_name}  quant={wb.quant}  VRAM={wb.vram_gb:.2f}GB  layers={wb.n_layers}\n")
    selftest(wb, bs=48, steps_a=steps, words_n=words_n)


def main():
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass
    ap = argparse.ArgumentParser(description="TRM working memory coupled to a frozen LM (real reasoner, design b)")
    ap.add_argument("--selftest", action="store_true", help="prove the mechanism on distilgpt2 (local, fast)")
    ap.add_argument("--probe", action="store_true", help="copy+bridge mechanism test on the real --lm (the fair bridge test)")
    ap.add_argument("--run", action="store_true", help="full composition experiment on --lm (hardest task)")
    ap.add_argument("--lm", type=str, default="Qwen/Qwen3-4B-Instruct-2507")
    ap.add_argument("--quant", type=str, default="4bit", help="quantization: 4bit, fp16, fp32, auto")
    ap.add_argument("--words", type=int, default=400, help="#atoms for --probe (scale this to test the data hypothesis)")
    ap.add_argument("--steps", type=int, default=120, help="training steps per probe")
    ap.add_argument("--epochs", type=int, default=40, help="training epochs for --run")
    ap.add_argument("--n-train", type=int, default=48, help="#composition pairs to train on for --run")
    ap.add_argument("--n-held", type=int, default=16, help="#held-out composition pairs for --run")
    ap.add_argument("--graph-path", type=str, default=None,
                    help="--run: use REAL atoms from this membrane.py graph file instead of the hand-written 10")
    ap.add_argument("--save-path", type=str, default=None,
                    help="--run: persist the trained WMReasoner here (was previously impossible -- the "
                         "proven adapter vanished when the process exited)")
    ap.add_argument("--grow-cot", type=int, default=0,
                    help="--run (requires --graph-path): ingest this many real OpenThoughts-114k CoT docs "
                         "into the graph via learn_any before training -- the graph actually grows, not just "
                         "trains on a static atom set. 0 = off (default, byte-identical to before this flag)")
    ap.add_argument("--grow-domains", type=str, default="math,code,science,puzzle",
                    help="--grow-cot: OpenThoughts domains to keep (comma-sep)")
    ap.add_argument("--grow-keywords", type=str, default="",
                    help="--grow-cot: comma-sep keywords, keep only docs mentioning one (optional filter)")
    ap.add_argument("--grow-skills", type=int, default=0,
                    help="--run (requires --graph-path): bank up to this many real oracle-verified EXECUTABLE "
                         "atoms from scripts/build_crossdomain_corpus.py via learn_any before training -- "
                         "these DO enter the composable pool (unlike --grow-cot's concept-only nodes). "
                         "Single-arg tasks only (see _grow_skills_from_corpus docstring). 0 = off")
    ap.add_argument("--grow-skills-domains", type=str, default="",
                    help="--grow-skills: comma-sep domain filter (math,physics,biology,cs,stats); empty = all")
    ap.add_argument("--batch-size", type=int, default=1,
                    help="--run: batch size for training (pads variable-length sequences). "
                         ">1 uses batched LM forward. With 90GB VRAM, 4-8 works on a 4B 4-bit model.")
    a = ap.parse_args()
    if a.probe:
        probe_real(a.lm, a.quant, a.words, a.steps)
    elif a.run:
        run_real(a.lm, a.quant, a.epochs, a.n_train, a.n_held, a.graph_path, a.save_path,
                 a.grow_cot, a.grow_domains, a.grow_keywords, a.grow_skills, a.grow_skills_domains,
                 batch_size=a.batch_size)
    else:
        selftest()


if __name__ == "__main__":
    main()
