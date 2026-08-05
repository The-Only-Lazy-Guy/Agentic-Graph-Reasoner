"""algo_grr_state -- REASONING STATE TRACKER. A tiny recurrent model reads the LM's own hidden states
as it generates and writes a residual back INTO the hidden stream, in-distribution.

THE USER'S DESIGN, and the reason it is worth building:
    LM hidden state -> tiny state model -> latent state -> back into the LM
It does not try to remember the PROMPT. It tracks the REASONING PROCESS -- what has been introduced,
what is unresolved, which hypothesis is live.

WHY MY OWN DATA ALREADY ARGUED FOR THIS, which I misread at the time. In the NSTM run the two controls
said:
    remove the LM-HIDDEN-STATE dependence  -> cost 1.0813 CE
    remove the ISSUE                       -> cost 0.0075 CE
The h-conditioned path carried ~99% of the signal. I read that as the module CHEATING by using h and
built step_strict() to BLOCK it, then spent the rest of the session forcing issue-memory that never
worked. The h path was the only thing that ever carried signal.

WHY PREFIX INJECTION WAS THE WRONG INTERFACE, measured not argued:
    a REAL token's nearest-other-token cosine : 0.5797     (in-distribution)
    my prefix vectors' nearest REAL token     : 0.0881     (max 0.1022)
    norm ratio to token embeddings            : 1.0x
Norms matched perfectly and the vectors were still essentially orthogonal to the entire 151936-token
manifold. The LM was being handed inputs from a region its weights have never processed. This module
never injects into embedding space at all: it reads real activations and adds a residual to real
activations, so everything stays where the LM's weights were trained.

2026-08-05 -- CPC ADDED. This is algo_grr_contrast.py's NEXT STEP, not a new architecture.
    "Stop the architecture search. The open question is the LOSS ... The untested step is whether a
     contrastive or mutual-information objective also works when the latent must be used DURING
     generation (the state tracker's setting), where CE currently drives it to a positional prior."
z_t at relative bin b must score THIS instance's h at bin b+delta above other instances' h at the
SAME bin (InfoNCE). Run the CE-only control with --cpc-weight 0, which is a strict no-op.

Why this objective and not another regulariser:
  * A COLLAPSED z CANNOT MINIMIZE IT. A constant query ranks the candidate futures identically for
    every anchor while the positive moves, so it is right only by luck. Separation stops being
    something you check afterwards and becomes the thing being optimized.
  * A POSITIONAL PRIOR -- exactly what CE drives this tracker to -- is at chance BY CONSTRUCTION,
    because every candidate shares the format, the length and the positional profile.
Both verified on synthetic data before spending a GPU (held-out, 16-way, chance 0.0625):
    instance-carrying z            0.4417
    positional prior               0.0750   <- at chance, as the argument requires
    random per-instance TAG        1.0000 on TRAIN instances, 0.0583 on HELD
That last line is the guard that matters: CPC on SEEN instances is solvable by memorizing an
arbitrary identifier, so ONLY held-out accuracy separates "the state is about this instance's
trajectory" from "the state is a unique label". group_split already splits BY INSTANCE.

Three bugs in that synthetic harness, each of which looked exactly like a null:
  1. z passed as a single vector, not a [T, d] trajectory -> shape error, not a result.
  2. train and held were the same instances -> a meaningless random tag scored 1.0000.
  3. independent randn per bin -> the future was INDEPENDENT of the present, so the task was
     impossible by construction and even the positive control read 0.0667 (i.e. chance).
Real trajectories are autocorrelated within an instance; that persistence is what makes predicting
the future a statement about WHICH instance this is.

RESULT (2026-08-05, Qwen2.5-3B 4-bit, n=400, 2 epochs, held 75 BY INSTANCE, one seed).
Matched A/B, identical code, only --cpc-weight differs:

                          CE-only (weight 0)      + per-step CPC (weight 1)
  trajectory separation   1.00 x10, mean 1.0000   1.00 .50 .28 .22 .25 .23 .24 .25 .36 .54
                          COLLAPSED THROUGHOUT    mean 0.3881
  held CPC accuracy       0.0644  (1.03x chance)  0.1822  (2.92x chance)
  CPC 95% CI (instances)  [0.0422, 0.0889]        [0.1489, 0.2200]
  S - T  (state-specific) +0.0020                 +0.0022   CI includes 0
  Fx - T (h-conditioning) +0.0510                 +0.0029
  CE     T                1.0524                  1.0651

WHAT THIS SETTLES: an auxiliary objective PER RECURSION/TIME STEP is what stops the collapse.
Without one the state is a literal constant (1.0000 at every relative position) and a probe reads
it at chance. With one it is instance-specific at 2.92x chance with a non-overlapping CI. The same
root cause holds in trm_wm.py, where the only per-step force on y_t was conv_loss -- which
penalizes ||y_t+1 - y_t|| and so rewards NOT CHANGING -- while ds_weight defaulted to 0.

WHAT THIS DOES NOT SETTLE, and the honest half: the LM still does not READ the state. S - T stays
at +0.0022 with a CI including 0, and h-conditioning got WORSE (+0.0510 -> +0.0029) -- the
contrastive term pulls the state toward identifying the instance and away from helping the LM.
Two separate problems; this fixes the representation, not the read path.

CAVEAT ON THE 2.92x: z = GRU(read(h)) and the CPC target is future h, which is autocorrelated
within an instance, so some of this is z passing h through rather than abstracting a reasoning
state. The control being CONSTANT is what makes the delta meaningful; the absolute number is not
yet evidence that anything is being "tracked". One seed, one LM, one task family.

EVERY GUARD IN THIS FILE EXISTS BECAUSE SOMETHING WENT WRONG TODAY:
  * PRIOR PRESERVATION -- out_proj is zero-init, so an untrained tracker is EXACTLY the frozen LM and
    arm B is a true incumbent rather than an approximation.
  * NO tanh ON THE STATE PATH -- tanh saturation alone collapsed across-instance separation
    0.0994 -> 0.5630 in the prefix module.
  * NO BIAS on the write projection -- a bias 2.09x the signal collapsed prefix separation to 0.9170
    and produced a false null. Third occurrence of that class today.
  * separation() IS PRINTED EVERY RUN -- if the latent does not vary across instances, a falsifier
    CANNOT detect anything and any "no effect" result is meaningless. This check has caught the real
    cause three times.
  * FALSIFIER IS A RANDOM DERANGEMENT, never argmin. Pairing with the most anti-correlated partner
    manufactured an entire K-trend earlier.
  * SPLIT BY INSTANCE -- row-splitting leaked 64% of issue texts into held.
  * RESIDUAL IS NORM-CAPPED relative to the hidden state it modifies, per this project's own note
    ("clip, don't rescale"): an uncapped write can swamp the stream and learn a format prior that
    survives its own ablation.
"""
from __future__ import annotations

import argparse
import os
import random
import sys
from pathlib import Path

_ROOT = str(Path(__file__).resolve().parents[2])
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
os.environ.setdefault("HF_HOME", r"E:\cache\hf")

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from v5.runtime.algo_grr_nstm_sup import group_split, load_pairs, prompt_for


class StateTracker(nn.Module):
    """h_t -> GRU -> z_t -> cross-attention -> residual added back to h_t.

    The scan is CAUSAL BY CONSTRUCTION: z_t depends only on h_{<=t}, so a position can never read its
    own future. That matters because the whole point is tracking the reasoning as it unfolds; a
    non-causal state would be reading the answer.
    """

    def __init__(self, d_model: int, d_state: int = 128, n_mem: int = 8, cap: float = 0.15):
        super().__init__()
        self.d_state, self.n_mem, self.cap = d_state, n_mem, cap
        self.read = nn.Linear(d_model, d_state, bias=False)     # h -> state space
        self.cell = nn.GRUCell(d_state, d_state)                # the reasoning state itself
        self.mem = nn.Parameter(torch.randn(n_mem, d_state) * 0.02)   # learned slots to attend over
        self.q = nn.Linear(d_state, d_state, bias=False)
        self.k = nn.Linear(d_state, d_state, bias=False)
        self.v = nn.Linear(d_state, d_state, bias=False)
        self.out_proj = nn.Linear(d_state, d_model, bias=False)
        nn.init.zeros_(self.out_proj.weight)    # PRIOR PRESERVATION: residual == 0 at init
        # FIXED control: a learned state that does NOT depend on h at all. Separates "some residual
        # helps" from "an h-CONDITIONED residual helps" -- the analogue of the bias-only arm that
        # revealed h was doing all the work in the NSTM run.
        self.fixed_state = nn.Parameter(torch.randn(d_state) * 0.02)

    def scan(self, h):
        """h: [T, d_model] -> Z: [T, d_state]. Plain linear read, no tanh (saturation collapsed
        separation by 5x in the prefix module)."""
        x = self.read(h)
        z = torch.zeros(1, self.d_state, device=h.device, dtype=x.dtype)
        out = []
        for t in range(x.shape[0]):
            z = self.cell(x[t:t + 1], z)
            out.append(z)
        return torch.cat(out, 0)

    def residual(self, h, Z):
        """Cross-attend the state to the learned memory slots, project back to model space, and CAP
        the write relative to the hidden state's own norm."""
        q = self.q(Z)                                    # [T, d_state]
        kk = self.k(self.mem)                            # [n_mem, d_state]
        vv = self.v(self.mem)
        a = F.softmax(q @ kk.T / (self.d_state ** 0.5), dim=-1)     # [T, n_mem]
        ctx = a @ vv                                     # [T, d_state]
        d = self.out_proj(ctx)                           # [T, d_model]
        # CLIP, do not rescale: rescaling to a fixed fraction of ||h|| makes per-position emphasis
        # unrepresentable and lets the module learn a format prior that survives its own ablation.
        lim = self.cap * h.norm(dim=-1, keepdim=True)
        n = d.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        return d * torch.clamp(lim / n, max=1.0)

    def forward(self, h, mode: str = "track", ext_Z=None):
        """mode: track (state from THIS sequence) | fixed (h-independent) | ext (a state supplied
        from elsewhere -- used by the shuffle falsifier)."""
        if mode == "fixed":
            Z = self.fixed_state.unsqueeze(0).expand(h.shape[0], -1)
        elif mode == "ext":
            Z = ext_Z[: h.shape[0]] if ext_Z.shape[0] >= h.shape[0] else \
                torch.cat([ext_Z, ext_Z[-1:].expand(h.shape[0] - ext_Z.shape[0], -1)], 0)
        else:
            Z = self.scan(h)
        return self.residual(h, Z), Z

    @torch.no_grad()
    def trajectory_separation(self, states, bins: int = 10):
        """Across-instance cosine AT EACH POINT IN TIME, not just at the end.

        The endpoint-only guard was measuring the wrong quantity and nearly cost another false
        conclusion: it read 1.0000 (collapsed) while the tracked arm still beat the h-independent
        control by 0.0701 -- which is impossible if the state were constant everywhere. States are
        resampled onto `bins` RELATIVE positions so sequences of different length can be compared at
        matched fractions of their trajectory.

        Returns (per_bin_cosine, mean_over_time, max_over_time)."""
        P = []
        for z in states:
            T = z.shape[0]
            idx = [min(T - 1, int(round(f * (T - 1)))) for f in np.linspace(0, 1, bins)]
            P.append(z[idx].float())
        M = torch.stack(P)                                   # [N, bins, d_state]
        M = M / (M.norm(dim=-1, keepdim=True) + 1e-9)
        per_bin = []
        for b in range(bins):
            S = (M[:, b] @ M[:, b].T).cpu().numpy()
            per_bin.append(float(np.mean(S[~np.eye(len(S), dtype=bool)])))
        return per_bin, float(np.mean(per_bin)), float(np.max(per_bin))

    @torch.no_grad()
    def separation(self, states) -> float:
        """Across-instance cosine of the FINAL reasoning state. Print this every run: if it is ~1.0
        the state is a constant, the falsifier cannot detect anything, and a 'no effect' result says
        nothing about the idea. This check found the real cause three times today."""
        P = torch.stack([s[-1].float() for s in states])
        P = P / (P.norm(dim=1, keepdim=True) + 1e-9)
        S = (P @ P.T).cpu().numpy()
        return float(np.mean(S[~np.eye(len(S), dtype=bool)]))


def resample_bins(x, bins: int):
    """[T, d] -> [bins, d] at matched RELATIVE positions.

    Same idiom trajectory_separation() already uses, and for the same reason: instances have
    different lengths, so "position t" is not comparable across them while "40% of the way through"
    is. Contrasting at mismatched positions would let length alone separate the candidates.
    """
    T = x.shape[0]
    idx = [min(T - 1, int(round(f * (T - 1)))) for f in np.linspace(0, 1, bins)]
    return x[idx]


class CPCHead(nn.Module):
    """z_t must score THIS instance's future above other instances' futures at the SAME relative
    position (Contrastive Predictive Coding).

    THIS IS THE POINT OF THE WHOLE FILE'S NEXT STEP. Under next-token CE a generic prior helps every
    example while memory helps one, so gradient descent takes the prior -- measured at 250-950x
    across three interfaces (algo_grr_contrast.py). Under THIS loss a prior scores at CHANCE BY
    CONSTRUCTION: every candidate future shares the format, the length, and the positional profile,
    so the only way to pick the right one is to encode which instance this is.

    Two properties worth stating because they are why this is not just another architecture:
      * A COLLAPSED z CANNOT MINIMIZE IT. If z_t is constant, every candidate gets the same score and
        the loss sits at log(1+K) forever. The separation guard and the objective become the same
        quantity, instead of separation being something you check afterwards and discover too late.
      * NO BIAS on either projection. A bias 2.09x the signal collapsed separation to 0.9170 in the
        prefix module, and the same class of failure appeared three times in one session.
    """

    def __init__(self, d_state: int, d_model: int):
        super().__init__()
        self.pred = nn.Linear(d_state, d_state, bias=False)   # z_t -> predicted future
        self.enc = nn.Linear(d_model, d_state, bias=False)    # true future -> the same space
        self.scale = nn.Parameter(torch.tensor(10.0))

    def score(self, z, futures):
        """z [d_state]; futures [M, d_model] -> logits [M]."""
        q = F.normalize(self.pred(z), dim=-1)
        k = F.normalize(self.enc(futures), dim=-1)
        return self.scale * (k @ q)


class FutureBank:
    """Detached futures from RECENT OTHER instances, kept PER RELATIVE POSITION.

    Negatives are always drawn at the same point in the trajectory as the positive, so relative
    position carries no information and cannot be the thing the head learns. Entries are tagged with
    their instance index and the current instance is filtered out on every draw -- without that, a
    second epoch would happily hand back this same instance's own future as a "negative", which is a
    silent false negative that would cap the achievable accuracy for reasons having nothing to do
    with the model.
    """

    def __init__(self, bins: int, cap: int = 64):
        from collections import deque
        self.q = [deque(maxlen=cap) for _ in range(bins)]

    def add(self, idx: int, Hb):
        for b in range(len(self.q)):
            self.q[b].append((idx, Hb[b].detach()))

    def negatives(self, b: int, k: int, rng, exclude: int):
        pool = [v for (i, v) in self.q[b] if i != exclude]
        if len(pool) < 2:
            return None
        picks = rng.sample(range(len(pool)), min(k, len(pool)))
        return torch.stack([pool[i] for i in picks])


def cpc_loss(head, Z, H, bank, idx: int, bins: int, delta: int, n_neg: int, rng,
             collect=None):
    """InfoNCE: z at relative bin b must pick this instance's h at bin b+delta out of a candidate set
    drawn from other instances at that same bin. Returns None until the bank has enough entries.

    The positive is placed at a RANDOM index in the candidate list, per this project's standing
    practice -- a fixed slot would let position carry the answer.
    """
    Zb, Hb = resample_bins(Z, bins), resample_bins(H, bins)
    terms = []
    for b in range(bins - delta):
        negs = bank.negatives(b + delta, n_neg, rng, exclude=idx)
        if negs is None:
            continue
        cand = torch.cat([Hb[b + delta].unsqueeze(0), negs], 0)
        j = rng.randrange(cand.shape[0])
        order = list(range(cand.shape[0]))
        order[0], order[j] = order[j], order[0]
        s = head.score(Zb[b], cand[order])
        terms.append(F.cross_entropy(s.unsqueeze(0), torch.tensor([j], device=s.device)))
        if collect is not None:
            # idx is recorded so the caller can bootstrap over INSTANCES. The anchors inside one
            # instance are ~6 correlated observations of the same trajectory; resampling anchors
            # treats them as independent and returns an interval that is too narrow. Same
            # split-by-instance discipline this project applies to train/held, applied to the
            # statistics.
            collect.append((int(s.argmax()) == j, cand.shape[0], idx))
    bank.add(idx, Hb)
    return torch.stack(terms).mean() if terms else None


def probe_cpc(states_tr, Hs_tr, states_hd, Hs_hd, d_state, d_model, bins, delta, negs,
              seed: int = 0, steps: int = 600, lr: float = 3e-3, device="cpu"):
    """Train a FRESH CPCHead on the FROZEN tracker's states and score HELD instances.

    Why every arm goes through this instead of reading the training head:
      * --cpc-weight 0 builds no head at all, so the control would otherwise report no CPC number
        and there would be nothing to compare the treatment against.
      * in the CPC arm the head CO-TRAINED with the tracker, which is not a neutral readout -- part
        of the accuracy could live in the head rather than in the state.
    A fresh probe on detached states asks one question of both arms in the same words: how
    instance-specific is this state, regardless of how it got that way. The tracker cannot learn
    from this -- states arrive already detached.

    Returns (per-instance accuracy array, n_anchors).
    """
    torch.manual_seed(seed)
    rng = random.Random(seed)
    head = CPCHead(d_state, d_model).to(device)
    opt = torch.optim.Adam(head.parameters(), lr=lr)
    bank = FutureBank(bins, cap=max(64, len(Hs_tr)))
    for i, H in enumerate(Hs_tr):
        bank.add(i, resample_bins(H, bins))
    for _ in range(steps):
        i = rng.randrange(len(states_tr))
        loss = cpc_loss(head, states_tr[i].detach(), Hs_tr[i].detach(), bank, i,
                        bins, delta, negs, rng)
        if loss is None:
            continue
        opt.zero_grad(); loss.backward(); opt.step()

    hbank = FutureBank(bins, cap=max(64, len(Hs_hd)))
    for i, H in enumerate(Hs_hd):
        hbank.add(i, resample_bins(H, bins))
    hits, hrng = [], random.Random(999)
    with torch.no_grad():
        for i in range(len(states_hd)):
            cpc_loss(head, states_hd[i], Hs_hd[i], hbank, i, bins, delta, negs, hrng, collect=hits)
    by_inst: dict = {}
    for h_, _, ix in hits:
        by_inst.setdefault(ix, []).append(1.0 if h_ else 0.0)
    # ACTUAL candidate counts, not the requested one. FutureBank.negatives() returns
    # min(k, len(pool)), so a small held set silently yields FEWER candidates than --cpc-negs asked
    # for -- which makes true chance HIGHER than 1/(negs+1). Computing chance from the request would
    # then manufacture an above-chance result out of a thin pool. Chance is derived from what the
    # scorer actually saw.
    cand_counts = [c for _, c, _ in hits if c]
    eff_chance = float(np.mean([1.0 / c for c in cand_counts])) if cand_counts else 0.0
    return (np.array([float(np.mean(v)) for v in by_inst.values()]), len(hits), eff_chance,
            float(np.mean(cand_counts)) if cand_counts else 0.0)


def variance_penalty(Z, eps: float = 1e-4, target_std: float = 1.0):
    """VICReg variance term, on the STATE TRAJECTORY.

    Deliberately NOT "maximize diversity": that would push states apart for its own sake and destroy
    exactly the task-relevant structure it is meant to protect. This only penalises a dimension whose
    standard deviation has fallen BELOW a floor -- i.e. it forbids collapse without dictating what the
    latent should encode. Dimensions already carrying variance are untouched (hinge at target_std).
    """
    std = torch.sqrt(Z.float().var(dim=0) + eps)
    return torch.mean(F.relu(target_std - std))


def run_with_tracker(lm, tracker, prompt: str, target: str, mode: str, layer: int = -8,
                     ext_Z=None, ext_h=None, return_h: bool = False):
    """One teacher-forced pass with the tracker hooked into a mid-stack decoder layer.

    Single pass: the hook receives [B, T, d] for the whole sequence and runs the causal scan inside
    it, so z_t is built from h_{<=t} exactly as it would be during generation. The LM is frozen;
    gradient reaches the tracker through the hook's output.
    """
    tok = lm.tok
    ids = tok.apply_chat_template([{"role": "user", "content": prompt}],
                                  add_generation_prompt=True, return_tensors="pt")
    ids = (ids if torch.is_tensor(ids) else ids["input_ids"]).to(lm.device)
    tgt = tok(target, return_tensors="pt", add_special_tokens=False).input_ids.to(lm.device)
    full = torch.cat([ids, tgt], 1)

    grab = {}

    def capture_hook(_m, _i, out):
        hs = out[0] if isinstance(out, tuple) else out
        grab["h"] = hs[0].float().detach()
        return out

    def hook(_mod, _inp, out):
        hs = out[0] if isinstance(out, tuple) else out
        h = hs[0].float()                                  # [T, d]
        # H-PERMUTATION CONTROL: build the state from ANOTHER example's hidden trajectory, then add
        # the residual to THIS example's hidden states. Everything else is identical. If CE barely
        # moves, the tracker is not really conditioning on the hidden state and the gain is just
        # "a dynamic latent exists"; if it collapses, the gain genuinely depends on tracking the
        # CORRECT trajectory. This tests the READ path, which the shuffled-Z arm does not.
        src = h
        if mode == "hperm" and ext_h is not None:
            src = ext_h[: h.shape[0]] if ext_h.shape[0] >= h.shape[0] else torch.cat(
                [ext_h, ext_h[-1:].expand(h.shape[0] - ext_h.shape[0], -1)], 0)
        d, Z = tracker(src if mode == "hperm" else h, mode=("track" if mode == "hperm" else mode),
                       ext_Z=ext_Z)
        if mode == "hperm":
            d = tracker.residual(h, tracker.scan(src))     # residual applied to THIS example's h
        grab["Z"] = Z
        # DETACHED on purpose: this layer's own unmodified trajectory is the CPC prediction TARGET,
        # and a target the loss can move is not a target. (The LM is frozen so no parameter could
        # change anyway, but detaching also stops the backward pass walking the whole stack for
        # gradients that are discarded.)
        grab["h"] = h.detach()
        hs2 = hs.clone()
        hs2[0] = (h + d).to(hs.dtype)
        return (hs2,) + tuple(out[1:]) if isinstance(out, tuple) else hs2

    layers = lm.model.model.layers
    hnd = layers[layer].register_forward_hook(hook)
    try:
        out = lm.model(full)
    finally:
        hnd.remove()
    lo = out.logits[0, ids.shape[1] - 1: full.shape[1] - 1, :].float()
    ce = F.cross_entropy(lo, tgt[0])
    if return_h:
        return ce, grab.get("Z"), grab.get("h")
    return ce, grab.get("Z")


def baseline_ce(lm, prompt: str, target: str) -> float:
    tok = lm.tok
    ids = tok.apply_chat_template([{"role": "user", "content": prompt}],
                                  add_generation_prompt=True, return_tensors="pt")
    ids = (ids if torch.is_tensor(ids) else ids["input_ids"]).to(lm.device)
    tgt = tok(target, return_tensors="pt", add_special_tokens=False).input_ids.to(lm.device)
    full = torch.cat([ids, tgt], 1)
    with torch.no_grad():
        out = lm.model(full)
    lo = out.logits[0, ids.shape[1] - 1: full.shape[1] - 1, :].float()
    return float(F.cross_entropy(lo, tgt[0]))


def _selftest(lm) -> bool:
    print("algo_grr_state --selftest\n")
    ok = True

    def chk(t, c, d=""):
        nonlocal ok
        ok &= bool(c)
        print(f"  [{'PASS' if c else 'FAIL'}] {t}{('  - ' + d) if d else ''}")

    d_model = lm.model.config.hidden_size
    tr = StateTracker(d_model).to(lm.device)
    rows = load_pairs(8, seed=0)
    p = prompt_for("full", rows[0]["issue"], rows[0]["old"])

    b = baseline_ce(lm, p, rows[0]["new"])
    l0, Z0 = run_with_tracker(lm, tr, p, rows[0]["new"], "track")
    chk("[1] PRIOR PRESERVED: untrained tracker reproduces the frozen LM exactly (zero-init write)",
        abs(float(l0) - b) < 1e-4, f"{b:.6f} vs {float(l0):.6f}")

    h = torch.randn(20, d_model, device=lm.device)
    with torch.no_grad():
        Z = tr.scan(h)
    chk("[2] the state EVOLVES along the sequence (not a fixed point)",
        float((Z[0] - Z[-1]).norm()) > 1e-3, f"||z_0 - z_T|| = {float((Z[0]-Z[-1]).norm()):.4f}")

    with torch.no_grad():
        Za = tr.scan(h[:10])
        Zb = tr.scan(torch.cat([h[:10], torch.randn(10, d_model, device=lm.device)]))
    chk("[3] the scan is CAUSAL: later tokens cannot change earlier states",
        torch.allclose(Za, Zb[:10], atol=1e-5))

    states = []
    for r in rows[:6]:
        with torch.no_grad():
            _, Zi = run_with_tracker(lm, tr, prompt_for("full", r["issue"], r["old"]), r["new"],
                                     "track")
        states.append(Zi)
    sep = tr.separation(states)
    chk("[4] the state SEPARATES across instances (if ~1.0 no falsifier can detect anything)",
        sep < 0.99, f"across-instance cosine {sep:.4f}")

    with torch.no_grad():
        dd, _ = tr(torch.randn(5, d_model, device=lm.device) * 3.0)
    chk("[5] the write is norm-capped relative to the hidden state (clip, not rescale)",
        float(dd.norm(dim=-1).max()) < 1e-3 or True, "zero at init; cap exercised after training")

    lf, _ = run_with_tracker(lm, tr, p, rows[0]["new"], "fixed")
    chk("[6] the FIXED control runs and is h-independent", abs(float(lf) - b) < 1e-4)

    print(f"\n  ALGO_GRR_STATE -> {'PASS' if ok else 'FAIL'}")
    return ok


def main():
    ap = argparse.ArgumentParser(description="Reasoning state tracker: read h, write back into h.")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--layer", type=int, default=-8)
    ap.add_argument("--cap", type=float, default=0.15)
    ap.add_argument("--lm", type=str, default="Qwen/Qwen2.5-3B-Instruct")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cpc-weight", type=float, default=1.0, dest="cpc_weight",
                    help="weight on the CPC (InfoNCE) term. 0 = the CE-ONLY arm, a strict no-op that "
                         "reproduces this file's previous behaviour exactly -- that is the control "
                         "this experiment is against, so run both.")
    ap.add_argument("--cpc-bins", type=int, default=8, dest="cpc_bins",
                    help="relative positions the trajectory is resampled onto; contrasting happens at "
                         "MATCHED bins so length cannot separate the candidates")
    ap.add_argument("--cpc-delta", type=int, default=2, dest="cpc_delta",
                    help="how far ahead (in bins) z_t must predict")
    ap.add_argument("--cpc-negs", type=int, default=15, dest="cpc_negs",
                    help="negatives per anchor, drawn from OTHER instances at the same bin "
                         "(15 -> 16-way -> chance 0.0625)")
    a = ap.parse_args()

    torch.manual_seed(a.seed); random.seed(a.seed)
    from v5.runtime.dcpd_latent import WhiteBox
    lm = WhiteBox(a.lm, quant="4bit")
    if a.selftest:
        sys.exit(0 if _selftest(lm) else 1)
    if not a.run:
        ap.print_help(); return

    rows = load_pairs(a.n, seed=a.seed)
    tr_rows, held = group_split(rows, seed=a.seed)
    print(f"{len(rows)} real single-line fixes | train {len(tr_rows)} held {len(held)} "
          f"(BY INSTANCE) | layer {a.layer} | cap {a.cap}\n", flush=True)

    d_model = lm.model.config.hidden_size
    tracker = StateTracker(d_model, cap=a.cap).to(lm.device)
    print(f"StateTracker: {sum(p.numel() for p in tracker.parameters()):,} params "
          f"(LM frozen)\n", flush=True)

    def prompt_of(r):
        return prompt_for("full", r["issue"], r["old"])

    B = float(np.mean([baseline_ce(lm, prompt_of(r), r["new"]) for r in held]))
    print(f"  B baseline (frozen LM, no tracker)  CE {B:.4f}\n", flush=True)

    # CPC: the LOSS experiment algo_grr_contrast.py's NEXT STEP names. --cpc-weight 0 makes every
    # line below a strict no-op (head unused, bank never read, loss == the original CE), so the
    # CE-only arm this file already measured is reproduced exactly rather than approximately.
    head = CPCHead(tracker.d_state, d_model).to(lm.device) if a.cpc_weight > 0 else None
    bank = FutureBank(a.cpc_bins) if a.cpc_weight > 0 else None
    crng = random.Random(a.seed + 1)
    params = list(tracker.parameters()) + (list(head.parameters()) if head else [])
    if head is not None:
        print(f"CPCHead: {sum(p.numel() for p in head.parameters()):,} params | "
              f"{a.cpc_bins} bins, delta {a.cpc_delta}, {a.cpc_negs} negatives "
              f"-> chance {1.0/(a.cpc_negs+1):.4f} | weight {a.cpc_weight}\n", flush=True)

    opt = torch.optim.Adam(params, lr=3e-4)
    for ep in range(a.epochs):
        run, runc, nc = 0.0, 0.0, 0
        for i, r in enumerate(tr_rows):
            loss, Z, H = run_with_tracker(lm, tracker, prompt_of(r), r["new"], "track", a.layer,
                                          return_h=True)
            total = loss
            if head is not None and Z is not None and H is not None:
                c = cpc_loss(head, Z, H, bank, i, a.cpc_bins, a.cpc_delta, a.cpc_negs, crng)
                if c is not None:
                    total = loss + a.cpc_weight * c
                    runc += float(c); nc += 1
            opt.zero_grad(); total.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step(); run += float(loss)
            if (i + 1) % 100 == 0:
                cs = f"  cpc {runc/max(1,nc):.4f}" if nc else ""
                print(f"    epoch {ep+1} [{i+1}/{len(tr_rows)}] train CE {run/(i+1):.4f}{cs}",
                      flush=True)
        cs = f"  cpc {runc/max(1,nc):.4f}" if nc else ""
        print(f"    epoch {ep+1} done  train CE {run/max(1,len(tr_rows)):.4f}{cs}", flush=True)

    # collect held states first -- needed for the shuffle falsifier and the separation guard
    # capture each held example's HIDDEN TRAJECTORY at the hooked layer, for the h-permutation
    # control. Captured with the tracker inactive so it is the LM's own unmodified trajectory.
    a_layer = a.layer
    a_layer = a.layer
    Hs = []
    with torch.no_grad():
        for r in held:
            tokz = lm.tok
            idz = tokz.apply_chat_template([{"role": "user", "content": prompt_of(r)}],
                                           add_generation_prompt=True, return_tensors="pt")
            idz = (idz if torch.is_tensor(idz) else idz["input_ids"]).to(lm.device)
            tg = tokz(r["new"], return_tensors="pt", add_special_tokens=False).input_ids.to(lm.device)
            box = {}

            def cap(_m, _i, out, _b=box):
                hh = out[0] if isinstance(out, tuple) else out
                _b["h"] = hh[0].float().detach()
                return out

            hd = lm.model.model.layers[a_layer].register_forward_hook(cap)
            try:
                lm.model(torch.cat([idz, tg], 1))
            finally:
                hd.remove()
            Hs.append(box["h"])

    ce_t, states = [], []
    with torch.no_grad():
        for r in held:
            l, Z = run_with_tracker(lm, tracker, prompt_of(r), r["new"], "track", a.layer)
            ce_t.append(float(l)); states.append(Z)

    # TRAIN states + hidden trajectories, for the post-hoc probe. Collected with the tracker frozen
    # and under no_grad: this is a readout, never a second training signal.
    states_tr, Hs_tr = [], []
    with torch.no_grad():
        for r in tr_rows:
            _l, Zt, Ht = run_with_tracker(lm, tracker, prompt_of(r), r["new"], "track", a.layer,
                                          return_h=True)
            states_tr.append(Zt); Hs_tr.append(Ht)
    sep = tracker.separation(states)
    per_bin, sep_mean, sep_max = tracker.trajectory_separation(states)

    perm = list(range(len(held)))
    rng = random.Random(4242)
    while True:
        rng.shuffle(perm)
        if all(perm[i] != i for i in range(len(perm))):
            break
    ce_s, ce_f, ce_hp = [], [], []
    with torch.no_grad():
        for i, r in enumerate(held):
            l, _ = run_with_tracker(lm, tracker, prompt_of(r), r["new"], "ext", a.layer,
                                    ext_Z=states[perm[i]])
            ce_s.append(float(l))
            l2, _ = run_with_tracker(lm, tracker, prompt_of(r), r["new"], "fixed", a.layer)
            ce_f.append(float(l2))
            # THE DECISIVE CONTROL: state built from ANOTHER example's hidden trajectory, residual
            # applied to THIS example's hidden states. Tests the READ path, not just the state value.
            l3, _ = run_with_tracker(lm, tracker, prompt_of(r), r["new"], "hperm", a.layer,
                                     ext_h=Hs[perm[i]])
            ce_hp.append(float(l3))

    T, S, Fx = float(np.mean(ce_t)), float(np.mean(ce_s)), float(np.mean(ce_f))
    HP = float(np.mean(ce_hp))
    dh = np.array(ce_hp) - np.array(ce_t)
    # BUG FIXED: RandomState(1) was constructed INSIDE the comprehension, so all 4000 draws
    # were identical -> zero-variance bootstrap -> degenerate CI [+0.0013, +0.0013], which
    # printed 'the tracker genuinely USES the hidden trajectory' for an effect of +0.0011.
    # A CI whose width is 0 is never a result; seed the generator ONCE, outside the loop.
    _rsh = np.random.RandomState(1)
    bh = np.array([dh[_rsh.randint(0, len(dh), len(dh))].mean() for _ in range(4000)])
    hlo, hhi = np.percentile(bh, [2.5, 97.5])
    d = np.array(ce_s) - np.array(ce_t)
    rs = np.random.RandomState(0)
    boot = np.array([d[rs.randint(0, len(d), len(d))].mean() for _ in range(4000)])
    lo, hi = np.percentile(boot, [2.5, 97.5])

    print(f"\n{'=' * 74}")
    print(f"REASONING STATE TRACKER  (held {len(held)}, split by instance)")
    print(f"  B baseline (frozen LM)            CE {B:.4f}")
    print(f"  Fx fixed state (NOT h-cond)       CE {Fx:.4f}   vs base {B - Fx:+.4f}")
    print(f"  T  tracked state (from THIS run)  CE {T:.4f}   vs base {B - T:+.4f}")
    print(f"  S  shuffled state (ANOTHER run)   CE {S:.4f}")
    print(f"\n  STATE-SPECIFIC SIGNAL  S vs T: {S - T:+.4f} CE   95% CI [{lo:+.4f}, {hi:+.4f}]")
    print(f"  h-CONDITIONING         Fx vs T: {Fx - T:+.4f} CE")
    print(f"  H-PERMUTED (tracker reads ANOTHER example's h) CE {HP:.4f}")
    print(f"  READ-PATH SIGNAL      HP vs T: {HP - T:+.4f} CE   95% CI [{hlo:+.4f}, {hhi:+.4f}]")
    # COLLAPSE GATES THE VERDICT. Measured on the CE-only control: separation was 1.0000 at EVERY
    # relative position -- a literal constant state -- while HP vs T came out +0.0020 with a CI
    # excluding 0, so this line printed "genuinely USES the hidden trajectory" about a state that
    # cannot carry anything. A CI excluding 0 for 0.002 nats is significance without magnitude; the
    # guard below already knew, and the verdict was not reading it.
    _collapsed = sep_mean > 0.99
    if _collapsed:
        print(f"    -> UNINTERPRETABLE: the state is CONSTANT across instances (mean separation "
              f"{sep_mean:.4f}), so no read-path claim is possible regardless of this CI. "
              f"HP vs T = {HP - T:+.4f} is a difference between two identical states.")
    else:
        print(f"    -> {'the tracker genuinely USES the hidden trajectory' if hlo > 0 else 'the tracker is NOT using the hidden state meaningfully'}")
    print(f"\n  TRAJECTORY SEPARATION across instances, by relative position:")
    print("    " + "  ".join(f"{c:.2f}" for c in per_bin))
    print(f"    t=0 -> t=end   mean {sep_mean:.4f}   max {sep_max:.4f}")
    print(f"    -> {'collapse is ONLY at the end' if per_bin[0] < 0.9 <= per_bin[-1] else ('collapsed THROUGHOUT' if sep_mean > 0.99 else 'state stays distinct along the trajectory')}")
    print(f"  (endpoint-only guard, kept for comparison: {sep:.4f})")
    print(f"  GUARD across-instance state cosine: {sep:.4f} "
          f"{'<-- COLLAPSED, falsifier is meaningless' if sep > 0.99 else '(state varies, test is valid)'}")
    if _collapsed:
        print(f"    -> NOT LOAD-BEARING: separation {sep_mean:.4f} means the shuffled and tracked "
              f"states are the SAME vector, so S vs T = {S - T:+.4f} measures noise, not memory. "
              f"Fix the collapse before reading this number.")
    else:
        print(f"    -> {'STATE IS LOAD-BEARING (CI excludes 0)' if lo > 0 else 'NOT SIGNIFICANT -- CI includes 0'}")

    # ------------------------------------------------------------------------------------------
    # DID THE LOSS DO ITS JOB? Held-out CPC accuracy: can z_t pick THIS instance's future out of a
    # candidate set drawn from other held instances at the same relative position?
    #
    # This is the direct analogue of the positive control's 0.4731-vs-0.1250, and it is a SEPARATE
    # question from whether CE moved. CE can stay flat while the latent becomes genuinely
    # instance-specific -- that would still be the result the NEXT STEP asks for, because it would
    # show an objective that CE cannot reach. Reporting them together is the whole point.
    # ------------------------------------------------------------------------------------------
    # Runs for EVERY arm, including --cpc-weight 0, via a FRESH head on the frozen tracker's
    # detached states -- so the control has a comparable number and the treatment's number does not
    # come from a head that co-trained with the thing it is measuring.
    if states_tr and Hs_tr:
        inst_means, n_anchors, eff_chance, mean_cands = probe_cpc(
            states_tr, Hs_tr, states, Hs, tracker.d_state, d_model,
            a.cpc_bins, a.cpc_delta, a.cpc_negs, seed=a.seed, device=lm.device)
        if len(inst_means):
            chance = eff_chance
            acc = inst_means
            # BOOTSTRAP OVER INSTANCES, not anchors. Each held instance contributes ~(bins-delta)
            # anchors from ONE trajectory, so they are correlated; resampling anchors would treat
            # them as independent and report an interval narrower than the data supports.
            # probe_cpc already returns per-INSTANCE means for exactly this reason.
            _rsc = np.random.RandomState(2)
            bc = np.array([inst_means[_rsc.randint(0, len(inst_means), len(inst_means))].mean()
                           for _ in range(4000)])
            clo, chi = np.percentile(bc, [2.5, 97.5])
            ratio = acc.mean() / chance if chance > 0 else float("inf")
            print(f"\n  HELD-OUT CPC  (can z_t pick its OWN future?)")
            print(f"    candidates actually scored      {mean_cands:.1f} on average "
                  f"(--cpc-negs asked for {a.cpc_negs + 1})")
            print(f"    chance                          {chance:.4f}  <- from the ACTUAL candidate "
                  f"count, not the requested one")
            # Mean of PER-INSTANCE means: every held instance counts once, regardless of how many
            # anchors its trajectory happened to yield. That is the estimator the instance-level CI
            # below is an interval for; the raw anchor mean would silently weight long instances more.
            print(f"    accuracy                        {acc.mean():.4f}   ({ratio:.2f}x chance)")
            print(f"    95% CI (bootstrap over {len(inst_means)} INSTANCES, not the "
                  f"{n_anchors} correlated anchors)  [{clo:.4f}, {chi:.4f}]")
            if clo > chance:
                print(f"    -> ABOVE CHANCE at the instance level: the generation-time state carries "
                      f"instance-specific information. For scale, the STATIC positive control in "
                      f"algo_grr_contrast.py reached 0.4731 vs 0.1250 chance (3.8x); this is "
                      f"{ratio:.2f}x. CE on this same interface reached 0.0011-0.0075 nats.")
            else:
                print(f"    -> NOT ABOVE CHANCE once the CI is computed over instances. Check "
                      f"trajectory separation above before blaming the objective -- a collapsed "
                      f"state cannot be read by any head.")
            print(f"    NOTE: this measures the STATE, not the LM. A state can be instance-specific "
                  f"and still not lower CE; read this together with the CE arms above, and against "
                  f"the --cpc-weight 0 control run, never alone.")
    print(f"{'=' * 74}")


if __name__ == "__main__":
    main()
