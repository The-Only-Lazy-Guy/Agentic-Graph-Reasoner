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
# WMReasoner — the working memory + its recursive refinement + the coupling hooks + deep supervision
# ================================================================================================
class WMReasoner(nn.Module):
    """K working-memory slots, initialized from retrieved atom embeddings, refined by T recursion steps
    (the real TRM role: iterative refinement), then read by the LM via gated cross-attention adapters.

    DEEP SUPERVISION: a lightweight retrieval head reads the working memory at each refinement step t
    and scores all candidate answers (cosine sim against answer_pool). CE loss at every step forces the
    entire refinement trajectory to converge toward the target, not just the final state."""
    def __init__(self, d_lm: int, couple_layers, d_emb: int = EMBED_DIM, T: int = 4, n_heads: int = 4):
        super().__init__()
        self.T = T
        # MODALITY PROJECTOR (2-layer GELU): translate KG/MiniLM geometry -> the decoder's native space, so
        # the cross-attention isn't handed 'foreign noise'. A single Linear is too weak to bridge the gap.
        self.proj_atom = nn.Sequential(nn.Linear(d_emb, d_lm), nn.GELU(), nn.Linear(d_lm, d_lm))
        self.proj_task = nn.Linear(d_emb, d_lm)
        self.upd = nn.Sequential(nn.Linear(3 * d_lm, d_lm), nn.GELU(), nn.Linear(d_lm, d_lm))
        self.norm = nn.LayerNorm(d_lm)
        self.adapters = nn.ModuleList([GatedCrossAttn(d_lm, n_heads) for _ in couple_layers])
        self.couple_layers = list(couple_layers)
        self._slots = None

        self.ds_pool = nn.Linear(d_lm, d_lm)
        self.ds_proj = nn.Linear(d_lm, d_lm)
        # temperature for the DS cosine logits. q and pool are L2-normalized -> logits in [-1,1]; dividing by
        # sqrt(d_lm) (~50) flattens them to ~uniform (ds_loss stuck at ln N, no gradient). 0.07 SHARPENS them.
        self._ds_scale = 0.07

        # SELF-CRITIQUE (tier-4 amortizer -- same doctrine as algo_grr_cot.py's critic): predicts the REAL
        # verifier's verdict FROM THE REASONING TRAJECTORY itself ("does this look right"), not by calling
        # the oracle. Trained supervised on (trajectory, real verify() outcome) pairs -- never the model's
        # own unverified guess (same anti-poison target discipline as trainer.py, applied to the critic).
        # NEVER certifies a trace for banking on its own -- amortizes which attempts are worth a real verify
        # call / flags likely mistakes for retry. Needs no answer_pool -- works for free-form answers too.
        # OWN pooling head (was reusing ds_pool): ds_pool is trained for a DIFFERENT objective (retrieval
        # margin against an answer pool), so its representation isn't necessarily informative about
        # generation-cleanliness (whether decoding stays valid vs degenerates/hallucinates a wrong name) --
        # a first real run collapsed to exactly the base rate, and a shared, objective-fighting pooling
        # layer was one of two diagnosed causes (the other: class imbalance, fixed at the training-loop level).
        self.critic_pool = nn.Linear(d_lm, d_lm)
        self.critic = nn.Sequential(nn.Linear(self.T * d_lm, d_lm), nn.GELU(), nn.Linear(d_lm, 1))

    def critique(self, states: list[torch.Tensor]) -> torch.Tensor:
        """states: the [T per-step states] from refine() for ONE example. Returns a scalar in [0,1] -- the
        critic's own estimate of 'will this pass the real verifier', built PURELY from how the working
        memory's reasoning evolved across its T recursion steps -- not from re-checking the answer."""
        traj = torch.stack([torch.tanh(self.critic_pool(s.mean(0, keepdim=True))) for s in states])  # [T,1,d_lm]
        flat = traj.reshape(1, -1)                                        # [1, T*d_lm]
        return torch.sigmoid(self.critic(flat)).squeeze()

    def critic_loss(self, states_batch: list[list[torch.Tensor]], labels: list) -> torch.Tensor:
        """Supervised BCE against REAL verify() outcomes (0/1) -- the target is always the verifier's own
        past label, never the model's own guess, exactly the anti-poison discipline already established for
        the LM (trainer.py), now applied to the critic."""
        preds = torch.stack([self.critique(s) for s in states_batch])
        y = torch.tensor([float(l) for l in labels], device=preds.device)
        return nn.functional.binary_cross_entropy(preds, y)

    def trajectory_instability(self, deltas: list) -> float:
        """FAST, LABEL-FREE mistake signal (per user request: 'fast noticeable mistakes', not the slower
        learned critic which needs GPU rounds just to validate it exists). No training, no labels.

        deltas: the per-step PRE-LayerNorm update magnitudes from refine(track_deltas=True).

        FIX (v2): the first version measured cosine-distance between POST-LayerNorm pooled states -- but
        refine() applies LayerNorm every step, which renormalizes z's scale back to a fixed range
        REGARDLESS of content, actively erasing the 'how much did this step want to move' signal this
        metric needs. Empirically confirmed broken: 3 different random tasks at real model scale (d_lm=2560)
        landed within 0.0005 of each other, and a real GPU run showed literally EVERY held-out task at the
        exact same instab=0.059 regardless of pass/fail -- a metric that's constant regardless of input is
        not measuring anything. Now measures the RAW update norm BEFORE LayerNorm erases it: does the
        proposed change SHRINK over the recursion (settling = converged = low instability) or stay as large
        as it started (still churning = hasn't converged = high instability)? Returns a ratio: late-step
        mean / early-step mean. >1 = not settling (bad), <1 = settling (good)."""
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
        """task_emb [d_emb], atom_embs [K,d_emb] (or [K,d_lm] if native=True) -> (working_memory [K,d_lm],
        per_step_states [[K,d_lm],...]) or, if track_deltas=True, a 3rd return value: per-step PRE-LayerNorm
        update magnitudes (see trajectory_instability -- LayerNorm renormalizes z's scale every step, which
        erases the 'how much did this step actually want to move' signal; deltas capture it before that
        erasure). Backward-compatible: default False keeps the old 2-value return, every existing caller
        unaffected. native=True: atom_embs are ALREADY in the LM's own embedding space (see
        native_text_embedding) -- skip proj_atom's MiniLM->LM bridge, which probe B showed collapses on
        held-out; probe C showed native-space injection generalizes (0.19->0.29 under dilution)."""
        q = self.proj_task(task_emb)
        if native:
            d_lm = self.upd[0].in_features // 3
            assert atom_embs.shape[-1] == d_lm, (
                f"refine(native=True) requires atom_embs already in the LM's own d_lm space "
                f"(got last dim {atom_embs.shape[-1]}, expected {d_lm}) -- produce it via native_text_embedding, "
                f"not a foreign encoder like MiniLM (that bridge collapses on held-out, see probe B).")
        z = atom_embs if native else self.proj_atom(atom_embs)
        states = []
        deltas = [] if track_deltas else None
        for _ in range(self.T):
            ctx = z.mean(0, keepdim=True).expand_as(z)
            qb = q.unsqueeze(0).expand_as(z)
            raw_update = self.upd(torch.cat([z, qb, ctx], dim=-1))
            z = self.norm(z + raw_update)
            states.append(z.clone())
            if track_deltas:
                deltas.append(raw_update.norm(dim=-1).mean().detach())
        if track_deltas:
            return z, states, deltas
        return z, states

    def ds_loss_batch(self, all_states: list[list[torch.Tensor]], answer_pool: torch.Tensor, gold_idxs: list[int]) -> torch.Tensor:
        """Batch deep supervision: for each example, at each refinement step, pool K slots -> query ->
        score against all candidates in answer_pool. CE on gold_idx at every step.

        all_states: [[T states], ...] per-example per-step states
        answer_pool: [N, d_lm] candidate embeddings
        gold_idxs:   [B] correct answer indices

        Vectorized: all examples + all steps are batched into one loss computation.
        """
        dev = answer_pool.device
        B = len(all_states)
        T = len(all_states[0]) if all_states else 1
        K = all_states[0][0].shape[0]
        # [B, T, K, d_lm] -> [B*T, K, d_lm]
        flat = torch.stack([torch.stack(s) for s in all_states], dim=0).float().to(dev)
        flat = flat.view(B * T, K, -1)
        pooled = torch.tanh(self.ds_pool(flat.mean(1)))  # [B*T, d_lm]
        q = self.ds_proj(pooled)
        q = q / (q.norm(dim=-1, keepdim=True) + 1e-8)
        logits = q @ answer_pool.float().T / self._ds_scale  # [B*T, N]
        gold = torch.tensor(gold_idxs, device=dev).repeat_interleave(T)
        return nn.functional.cross_entropy(logits, gold)

    def align_projector(self, kg_embs: torch.Tensor, lm_targets: torch.Tensor,
                        steps: int = 300, lr: float = 1e-3) -> float:
        """MODALITY PRE-ALIGNMENT (CLIP): train proj_atom so KG (MiniLM) embeddings land in the LM's OWN
        representation space -> cross-attention stops seeing foreign noise. kg_embs [N,d_emb], lm_targets
        [N,d_lm] (the LM's embedding of each atom). Trains proj_atom only; symmetric InfoNCE."""
        opt = torch.optim.Adam(self.proj_atom.parameters(), lr=lr, weight_decay=1e-2)  # resist memorizing the pairs
        T = lm_targets / (lm_targets.norm(dim=-1, keepdim=True) + 1e-8)
        last = float("nan")
        for _ in range(steps):
            P = self.proj_atom(kg_embs)
            P = P / (P.norm(dim=-1, keepdim=True) + 1e-8)
            logits = P @ T.t() / 0.07
            labels = torch.arange(P.shape[0], device=P.device)
            loss = 0.5 * (nn.functional.cross_entropy(logits, labels) +
                          nn.functional.cross_entropy(logits.t(), labels))
            opt.zero_grad(); loss.backward(); opt.step()
            last = float(loss.detach())
        return last

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

    def _device(self):
        return self.proj_task.weight.device

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
    """precomputed_states: list of [[K,d_lm], ...] per-step states for each word, one refine per word."""
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

            if ds_weight > 0 and answer_pool is not None:
                state_list = []
                gold_list = []
                for bi, j in enumerate(idx):
                    state_list.append(precomputed_states[j])
                    gold_list.append(j)
                ds_acc = R.ds_loss_batch(state_list, answer_pool.to(wb.device), gold_list)
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
        print(f"       [dump] held-out  target -> top-5 predicted (slot injected):")
        for j in range(min(dump, len(test))):
            w, tokid = test[j]
            R.set_slots_direct(precomputed_states[base + j][-1].unsqueeze(0).to(wb.device).detach())
            with torch.no_grad():
                lg = wb.model(pids).logits[0, -1]
            top = lg.topk(5).indices.tolist()
            toks = ", ".join(repr(wb.tok.decode([t])) for t in top)
            print(f"          {w!r:>12} (tok {tokid}) -> {toks}  {'<- HIT' if top[0] == tokid else ''}")
    return tr, te, te_abl, float(R.adapters[0].g), last


def selftest(wb=None, bs=128, steps_a=120, steps_b=250, words_n=120):
    from v5.runtime.dcpd_latent import WhiteBox
    torch.manual_seed(0)
    if wb is None:
        print("trm_wm.py --selftest : TRM working memory coupled to a FROZEN distilgpt2 (mechanism proof)\n")
        wb = WhiteBox("distilgpt2", quant="fp32")
        if os.environ.get("GRAPH_FORCE_CPU"):
            wb.model = wb.model.to("cpu"); wb.device = "cpu"
            print("  (forced CPU)")
    else:
        print(f"trm_wm.py --probe on {wb.name} (real LM): copy(A) + bridge(B) mechanism test on a capable model\n")
    d_lm = wb.d_model
    couple = [wb.n_layers - 2, wb.n_layers - 1]         # last two layers (was [4,5] for distilgpt2's 6)
    R = WMReasoner(d_lm, couple_layers=couple).to(wb.device)
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

    # INJECT IN THE OUTPUT (unembedding) SPACE. logit_w = final_hidden . lm_head[w]; to make the LM emit w you
    # must push the hidden toward the OUTPUT embedding of w, NOT the input embedding. They're equal only when
    # the model TIES them (distilgpt2 does -> probe A "worked"); Qwen UNTIES them -> input-emb probe A can't
    # generalize (0.88 on distilgpt2 was a tying artifact). get_output_embeddings==input for tied models.
    tie = bool(getattr(wb.model.config, "tie_word_embeddings", False))
    _out = wb.model.get_output_embeddings()
    lm_emb = (_out.weight if _out is not None else wb.model.get_input_embeddings().weight)
    print(f"  tie_word_embeddings={tie} -> inject in the {'tied' if tie else 'OUTPUT/unembedding'} space\n")
    ans_idx = {w: i for i, (w, _) in enumerate(words)}
    answer_pool = torch.stack([lm_emb[tid_of[w]] for w, _ in words], dim=0)
    answer_pool = answer_pool / (answer_pool.norm(dim=-1, keepdim=True) + 1e-8)

    # PROBE A — WIRING: precompute ALL per-step states at once
    states_a = []
    for w, _ in words:
        z_a = lm_emb[tid_of[w]].detach()  # [d_lm]
        states_a.append([z_a.unsqueeze(0).clone() for _ in range(R.T)])
    tr_a, te_a, ab_a, g_a, l_a = _run_probe(
        wb, R, pids, train_w, test_w, states_a, answer_pool, tid_of, steps=steps_a, bs=bs, ds_weight=0.15, dump=6)
    print(f"  (A) WIRING  (slot = LM's own embedding):  train {tr_a:.2f}  HELD-OUT {te_a:.2f}  "
          f"ablate->0 {ab_a:.2f}  gate {g_a:+.2f}  loss {l_a:.3f}")

    # PROBE B — BRIDGE (slot from MiniLM graph space). Run RAW vs CLIP-ALIGNED to isolate the modality gap.
    for h in handles:
        h.remove()
    task_emb = torch.as_tensor(encode_batch([prompt])[0], dtype=torch.float32, device=wb.device)
    memb = {w: torch.as_tensor(encode_batch([w])[0], dtype=torch.float32, device=wb.device) for w, _ in words}
    align_kg = torch.stack([memb[w] for w, _ in train_w])                    # MiniLM of TRAIN atoms only
    align_tgt = torch.stack([lm_emb[tid_of[w]].float() for w, _ in train_w]) # the LM's own embedding of each
    te_b, bridge_res = 0.0, {}
    for tag, do_align in (("raw", False), ("CLIP-aligned", True)):
        Rb = WMReasoner(d_lm, couple_layers=couple).to(wb.device)
        hb = Rb.couple(wb)
        al = Rb.align_projector(align_kg, align_tgt, steps=300) if do_align else None
        states_b = []
        with torch.no_grad():
            for w, _ in words:
                # feed the projector output DIRECTLY as the slot (parallel to probe A's e_w) -- do NOT push it
                # through the random-init recursion, which scrambles the alignment before the adapter reads it.
                z = Rb.proj_atom(memb[w].unsqueeze(0)).detach()              # [1,d_lm]; aligned -> ~ e_w
                states_b.append([z.clone() for _ in range(Rb.T)])
        tr_b, te_b, ab_b, g_b, l_b = _run_probe(
            wb, Rb, pids, train_w, test_w, states_b, answer_pool, tid_of, steps=steps_b, bs=bs, ds_weight=0.15,
            dump=(6 if do_align else 0))
        bridge_res[tag] = te_b
        alstr = f"  align_loss {al:.2f}" if al is not None else ""
        print(f"  (B:{tag:>12}) BRIDGE MiniLM->LM:  train {tr_b:.2f}  HELD-OUT {te_b:.2f}  "
              f"ablate {ab_b:.2f}  gate {g_b:+.2f}{alstr}")
        if do_align:
            # RECURSED variant: push the ALIGNED slot through the actual T-step refine() this time (per the
            # finding that probe B previously bypassed the TRM's own recursion entirely). upd/norm are still
            # RANDOM/untrained here (states are precomputed once, detached, like every other probe) -- this
            # isolates whether random recursion further scrambles the aligned signal, or leaves it roughly
            # intact, without yet committing to training the recursion weights themselves.
            states_b_rec = []
            with torch.no_grad():
                for w, _ in words:
                    _, states = Rb.refine(task_emb, memb[w].unsqueeze(0))
                    states_b_rec.append([s.clone().detach() for s in states])
            tr_r, te_r, ab_r, g_r, l_r = _run_probe(
                wb, Rb, pids, train_w, test_w, states_b_rec, answer_pool, tid_of,
                steps=steps_b, bs=bs, ds_weight=0.15, dump=6)
            bridge_res["CLIP-aligned+recursed"] = te_r
            print(f"  (B:CLIP+recursed) BRIDGE through TRM's OWN recursion:  train {tr_r:.2f}  HELD-OUT {te_r:.2f}  "
                  f"ablate {ab_r:.2f}  gate {g_r:+.2f}")
        for h in hb:
            h.remove()
    handles = []

    # PROBE C — NATIVE-SPACE, MULTI-TOKEN: mean-pool a short carrier phrase's tokens through the LM's OWN
    # embedding table (zero cross-model gap, unlike B) -- tests whether probe A's win survives when the
    # target's signal is DILUTED by surrounding phrase tokens, the realistic shape of a graph atom's
    # natural-language description (not a bare single-token embedding).
    Rc = WMReasoner(d_lm, couple_layers=couple).to(wb.device)
    hc = Rc.couple(wb)
    states_c = []
    with torch.no_grad():
        for w, _ in words:
            ids = wb.tok(f"the concept of {w}", return_tensors="pt").input_ids.to(wb.device)
            z_c = lm_emb[ids[0]].mean(0).detach()          # mean-pool -> ONE vector, still fully native space
            states_c.append([z_c.unsqueeze(0).clone() for _ in range(Rc.T)])
    tr_c, te_c, ab_c, g_c, l_c = _run_probe(
        wb, Rc, pids, train_w, test_w, states_c, answer_pool, tid_of, steps=steps_a, bs=bs, ds_weight=0.15, dump=6)
    print(f"  (C) NATIVE-PHRASE (mean-pooled LM-own embedding, diluted):  train {tr_c:.2f}  HELD-OUT {te_c:.2f}  "
          f"ablate->0 {ab_c:.2f}  gate {g_c:+.2f}  loss {l_c:.3f}")
    for h in hc:
        h.remove()

    print()
    raw_b, al_b = bridge_res.get("raw", 0.0), bridge_res.get("CLIP-aligned", 0.0)
    print(f"  => probe A (wiring) held-out {te_a:.2f}  |  probe B bridge held-out: raw {raw_b:.2f} -> CLIP-aligned {al_b:.2f}")
    rec_b = bridge_res.get("CLIP-aligned+recursed")
    if al_b >= 0.5 and al_b > raw_b:
        print(f"     BRIDGE WORKS on {wb.name}: CLIP-aligned graph slots read + GENERALIZE. Modality gap CLOSED.")
    elif al_b > raw_b + 0.1:
        print(f"     BRIDGE PARTIAL on {wb.name}: alignment helps ({raw_b:.2f}->{al_b:.2f}) but not solved.")
    else:
        print(f"     BRIDGE FAILS on {wb.name}: alignment did not transfer to held-out -> graph slots stay foreign.")
    if rec_b is not None:
        if abs(rec_b - al_b) < 0.05:
            print(f"     recursion is roughly NEUTRAL through the TRM loop ({al_b:.2f} -> {rec_b:.2f}).")
        elif rec_b > al_b:
            print(f"     recursion HELPS ({al_b:.2f} -> {rec_b:.2f}) -- worth training the recursion weights too.")
        else:
            print(f"     recursion HURTS ({al_b:.2f} -> {rec_b:.2f}) -- random untrained recursion scrambles alignment.")
    if te_c >= 0.5 * te_a:
        print(f"     PROBE C: native-space injection SURVIVES dilution ({te_a:.2f} single-token -> {te_c:.2f} phrase)"
              f" -- inject via the LM's OWN embedding table for real atom text, skip the MiniLM bridge entirely.")
    else:
        print(f"     PROBE C: native-space injection DEGRADES under dilution ({te_a:.2f} -> {te_c:.2f})"
              f" -- even native-space needs a sharp single-vector signal, not a diluted phrase mean.")

    print(f"\n  DEEP SUPERVISION (ds_weight=0.15):")
    print(f"     refinement steps: {R.T}  |  ds_head params: {sum(p.numel() for p in R.ds_pool.parameters()) + sum(p.numel() for p in R.ds_proj.parameters())}")
    print(f"     each step's working memory must independently retrieve the target from {len(words)} candidates")
    print(f"     -> gradients flow through ALL T steps, not just the final output")
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
    enum; whether a node is usable for composition is a fact about its code, not what string labels it)."""
    descs, codes = {}, {}
    for name, a in g.atoms.items():
        if a.code:
            descs[name] = a.description
            codes[name] = a.code
    return descs, codes


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


def run_real(lm_name: str, quant: str = "4bit", epochs: int = 40, n_train: int = 48, n_held: int = 16,
            graph_path: str | None = None):
    """graph_path=None (default): UNCHANGED behavior, the hand-written 10-atom dict + hand-tuned templated
    tasks (the proven 13-15/16 held-out result) -- zero risk of regression, this path is untouched by Phase
    3. graph_path=<path>: real graph atoms (via membrane's AtomGraph.load/seed_graph + _atoms_from_graph)
    and a graph-derived dynamic oracle (_dynamic_oracle, via membrane._closure) -- scales to whatever atoms
    actually exist, not a fixed 10."""
    from v5.runtime.dcpd_latent import WhiteBox
    import random
    print(f"run_real: WMReasoner coupled to {lm_name} ({quant}) — real composition tasks\n")

    wb = WhiteBox(lm_name, quant=quant)
    d_lm = wb.d_model
    couple = [wb.n_layers - 2, wb.n_layers - 1]
    print(f"  LM: {lm_name}  d={d_lm}  layers={wb.n_layers}  gate layers={couple}  device={wb.device}")

    R = WMReasoner(d_lm, couple_layers=couple).to(wb.device)
    for p in wb.model.parameters():
        p.requires_grad_(False)
    handles = R.couple(wb)

    if graph_path:
        from pathlib import Path as _Path
        from v5.runtime.membrane import AtomGraph, seed_graph
        g = AtomGraph.load(graph_path) if _Path(graph_path).exists() else seed_graph()
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
    atom_embs = {n: native_text_embedding(wb, descs[n]) for n in atom_names}

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
    def build_prompt(task_text):
        return wb.tok(f"Write a function task(n):\n# {task_text}\ndef task(n):", return_tensors="pt").input_ids.to(wb.device)

    # Precompute static task embeddings and atom embeddings for all examples
    print("  Precomputing task + atom embeddings...")
    task_embs = {}
    for text, atoms_needed, _ in all_tasks:
        if text not in task_embs:
            task_embs[text] = torch.as_tensor(encode_batch([text])[0], dtype=torch.float32, device=wb.device)

    atom_stack = torch.stack([atom_embs[n] for n in atom_names], dim=0).to(wb.device)  # [N, d_lm] NATIVE now

    prompt_ids = {}
    for text, _, _ in all_tasks:
        prompt_ids[text] = build_prompt(text)

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
        """Pull the FIRST return-expression out of raw generated text. Greedy decoding with no stopping
        criterion tends to loop the same 'return EXPR' clause until max_new_tokens cuts it off mid-token --
        an eval-harness artifact, not a reasoning failure. Cut at the next newline OR the next repeated
        'return' (the loop) so the extracted expression is the model's actual (usually complete) first answer."""
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

    train_ex = [(task_embs[text], atom_names.index(atoms_needed[0]), atoms_needed,
                 prompt_ids[text], text, code)
                for text, atoms_needed, code in train_tasks]
    held_ex = [(task_embs[text], atom_names.index(atoms_needed[0]), atoms_needed,
                prompt_ids[text], text, code)
               for text, atoms_needed, code in held_tasks]

    print(f"  Training the adapter + WMReasoner ({epochs} epochs, {len(train_ex)} pairs; "
          f"real-close loop: refine -> LM -> verify)...")
    best_held = 0.0
    eval_every = max(1, epochs // 8)
    last_dump = None
    # SELF-CRITIQUE data collection (co-evolutionary arms-race idea, stage 1): every eval checkpoint runs
    # real generate()+verify() on held_ex (free byproduct) AND, now, ALSO on train_ex (a small deliberate
    # extra generation cost, see below) -- ~4x more real (trajectory, real-verify-label) pairs than
    # held_ex alone, addressing the critic's small-sample-size problem directly. Still PASSIVE: this data
    # trains the critic only, after the main loop -- does not touch the reasoner's own loss (that's stage 2,
    # gated on stage 1's critic actually beating base rate first, to avoid training the reasoner against an
    # unreliable judge -- Goodhart's law / reward-hacking risk, not built yet).
    critic_examples: list = []
    for ep in range(epochs):
        R.train()
        # DS candidate pool: atom_stack is ALREADY native d_lm (native_text_embedding) -- no projection needed.
        train_pool = (atom_stack / (atom_stack.norm(dim=-1, keepdim=True) + 1e-8)).detach()
        random.shuffle(train_ex)
        tot_lm, tot_ds, n = 0.0, 0.0, 0
        for task_emb, gold_idx, atoms_needed, pids, text, target_code in train_ex:
            K_atom_embs = torch.stack([atom_embs[n] for n in atoms_needed], dim=0)
            slots, states = R.refine(task_emb, K_atom_embs, native=True)
            R.set_slots_direct(slots)
            return_body = target_code.split(": ", 1)[1] if ": " in target_code else target_code
            tids = wb.tok(" " + return_body, return_tensors="pt").input_ids.to(wb.device)
            input_ids = torch.cat([pids, tids], dim=-1)
            labels = torch.full_like(input_ids, -100)
            labels[:, pids.shape[-1]:] = tids
            outs = wb.model(input_ids=input_ids, labels=labels)
            lm_loss = outs.loss
            state_list = [states]
            ds_loss = R.ds_loss_batch(state_list, train_pool, [gold_idx])
            loss = lm_loss + 0.1 * ds_loss
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot_lm += float(lm_loss.detach())
            tot_ds += float(ds_loss.detach())
            n += 1

        if ep % eval_every == 0 or ep == epochs - 1:
            R.eval()
            held_ok, ablated_ok = 0, 0
            dump = []
            for task_emb, gold_idx, atoms_needed, pids, text, target_code in held_ex:
                K_atom_embs = torch.stack([atom_embs[n] for n in atoms_needed], dim=0)
                slots, wm_states, wm_deltas = R.refine(task_emb, K_atom_embs, native=True, track_deltas=True)
                with torch.no_grad():
                    R.set_slots_direct(slots)
                    # NO repetition_penalty: tried it to stop the 'return EXPR return EXPR...' loop, but it
                    # penalizes reusing ANY prior token -- including the literal atom-name tokens the working
                    # memory taught it. Composing e.g. count_bits(count_bits(n)) requires REPEATING a name;
                    # even single-composition cases got pushed toward hallucinated paraphrases ('num_bits',
                    # 'count_bits_to_n') instead of the real atom names. verify()'s extraction (stop at the
                    # first newline or repeated 'return') already tolerates the loop without hurting quality.
                    out = wb.model.generate(pids, max_new_tokens=64,
                                            do_sample=False, pad_token_id=wb.tok.eos_token_id)
                    code = wb.tok.decode(out[0][pids.shape[-1]:], skip_special_tokens=True).strip()
                wm_ok = verify("def task(n): " + code, target_code)
                held_ok += int(wm_ok)
                critic_examples.append(([s.detach() for s in wm_states], wm_ok))   # REAL trajectory + REAL label
                instability = R.trajectory_instability(wm_deltas)   # FAST, no-training mistake signal (v2: pre-norm deltas)
                R.clear()
                with torch.no_grad():
                    out = wb.model.generate(pids, max_new_tokens=64,
                                            do_sample=False, pad_token_id=wb.tok.eos_token_id)
                    code_abl = wb.tok.decode(out[0][pids.shape[-1]:], skip_special_tokens=True).strip()
                abl_ok = verify("def task(n): " + code_abl, target_code)
                ablated_ok += int(abl_ok)
                dump.append((text, target_code, code, wm_ok, code_abl, abl_ok, instability))

            # CO-TRAINING data (stage 1, per user's arms-race idea): the held_ex loop above is the ONLY place
            # real generate()+verify() happens, so it's the only place a real pass/fail label exists -- the
            # main training loop is teacher-forced (target always fed as ground truth), so there's no
            # natural label there. Extend the SAME real eval mechanism to train_ex too, at the SAME checkpoint
            # cadence (9 times, not every epoch) -- ~4x more critic data (48+16 tasks vs 16), ~2x more
            # generation cost (skip the ablated comparison here, not needed for critic data), NOT a 40x or
            # 50x blowup. Still purely PASSIVE/monitoring -- does not touch the reasoner's own loss (stage 2,
            # gated on stage 1 actually beating base rate, not built yet).
            for task_emb, gold_idx, atoms_needed, pids, text, target_code in train_ex:
                K_atom_embs = torch.stack([atom_embs[n] for n in atoms_needed], dim=0)
                slots, tr_states = R.refine(task_emb, K_atom_embs, native=True)
                with torch.no_grad():
                    R.set_slots_direct(slots)
                    out = wb.model.generate(pids, max_new_tokens=64,
                                            do_sample=False, pad_token_id=wb.tok.eos_token_id)
                    tr_code = wb.tok.decode(out[0][pids.shape[-1]:], skip_special_tokens=True).strip()
                tr_ok = verify("def task(n): " + tr_code, target_code)
                critic_examples.append(([s.detach() for s in tr_states], tr_ok))
                R.clear()
            R.train()

            best_held = max(best_held, held_ok)
            last_dump = dump
            inst_pass = [d[6] for d in dump if d[3]]
            inst_fail = [d[6] for d in dump if not d[3]]
            inst_str = (f"  instab(pass/fail) {sum(inst_pass)/len(inst_pass):.3f}/"
                       f"{sum(inst_fail)/len(inst_fail):.3f}" if inst_pass and inst_fail else "")
            print(f"  ep {ep:>3}  lm_loss {tot_lm/max(n,1):.3f}  ds_loss {tot_ds/max(n,1):.3f}  "
                  f"held WM {held_ok}/{len(held_ex)}  ablated {ablated_ok}/{len(held_ex)}  {inst_str}  "
                  f"gate {float(R.adapters[0].g):+.2f}")

    print(f"\n  [dump] final epoch, held-out generations (WM vs ablated) vs the verified target:")
    for text, target_code, code, wm_ok, code_abl, abl_ok, instability in (last_dump or []):
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
    selftest(wb, bs=48, steps_a=steps, steps_b=steps, words_n=words_n)


def main():
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
    a = ap.parse_args()
    if a.probe:
        probe_real(a.lm, a.quant, a.words, a.steps)
    elif a.run:
        run_real(a.lm, a.quant, a.epochs, a.n_train, a.n_held, a.graph_path)
    else:
        selftest()


if __name__ == "__main__":
    main()
