"""trm_wm.py — the TRM as a REAL reasoner COUPLED to the frozen LM (design (b)), with a WORKING MEMORY.

=====================================================================================
2026-08-05 — THE 13-15/16 HEADLINE IS FALSIFIED. Read this before using anything below.
=====================================================================================
Qwen2.5-3B-Instruct 4-bit, 40 epochs, n-train 48, n-held 16, local RTX 4050. ONE run
prints both verdicts, because the old one is computed against a control that cannot fail:

    Best held-out 7/16 (ablated 0)  => PROVEN          <- the old logic
    WM 7/16   DERANGED 8/16   ABLATED 0/16 => FALSIFIED <- the derangement arm

    CE  WM 0.2820   DERANGED 0.2828   ABLATED 7.6788
    instance-specific  der - WM   +0.0009  95% CI [-0.0056, +0.0065]   SPANS ZERO
    format/mode effect abl - der  +7.3960  95% CI [+6.8445, +7.9882]
    modal / specific ratio        8378x

DERANGED >= WM at 9 of 9 checkpoints. `ablated` sets _slots=None so _mk_hook returns
early (it is the BASE LM), and build_prompt is called without inner/outer so the prompt
never carries atom names -- 0/16 is true BY CONSTRUCTION.

The mechanism is visible in the generations, not inferred. Ablated does not emit wrong
code, it emits PROSE ("def task(n): The function `task(n)` calculates the digit sum
of..."), so 0/16 measured OUTPUT FORMAT. Format is task-independent, which is exactly
why deranging costs nothing. And every WM failure paraphrases the description already in
the prompt (num_divisors -> count_divisors, reverse_digits -> digit_reverse): the atom
names come from the prompt text, never from the slots.

CONFOUNDS (this run does NOT show the channel could not work, only that it does not
here): slot_cos sat at 0.9856-1.0000 all run, so the slot-collapse bug is present in
this tree and a fix restoring slot variance invalidates the premise; the gate climbed to
+1.43, outside the 0.1-0.25 band documented at the gate_max clamp below; Qwen2.5-3B, not
the Qwen3-4B the headline used; one seed.

2026-08-05 (later) -- THE PER-STEP CONTRASTIVE FIX DOES **NOT** TRANSFER HERE. Matched A/B,
40 epochs, Qwen2.5-3B 4-bit, n-train 48, held 16, identical code, only --contrast-weight differs:

                        contrast 0          contrast 0.5
  per_step slot cos     1.000 x4            1.000 x4        <- still fully collapsed
  slot_cos              0.996172            0.999999        <- WORSE
  WM / DERANGED         5/16 / 4/16         6/16 / 6/16
  CE der - WM           +0.0032             -0.0004
  modal/specific ratio  2278x               inf
  ct (train)            n/a                 3.019 -> 2.803

ct sat at its CHANCE FLOOR the whole run: ln(16) = 2.7726. The loss is behaving exactly as
designed -- a collapsed y_t gives every anchor an identical query, so it pins at log(1+K) -- it
is correctly REPORTING collapse without ESCAPING it. Contrast this with algo_grr_state.py, where
the same objective took separation 1.0000 -> 0.3881 and a probe 1.03x -> 2.92x chance.

WHY IT DIFFERS, measured not guessed (rule 10: check the write path before judging the idea):
  gradient from ct reaches 22/22 TRM parameters -- the channel is NOT dead
  y_t   across-task cosine AT INIT   0.8889     (state tracker's z: 0.7643)
  slots across-task cosine AT INIT   0.9576
  ||proj_y.bias|| / ||W@y||          1.280
proj_y maps y_t to the slots the LM attends, and its BIAS is a constant added to every slot of
every task -- the adapter shortcut expressed as a parameter. Zeroing it moves slots 0.9579 ->
0.8949 at init, i.e. it recovers exactly the gap down to y_t and no further. So the bias is A
constant, not THE cause: y_t is already at 0.889 before training and the contrastive term starts
inside a nearly-collapsed basin where lm_loss outweighs it. --no-proj-bias added for the next A/B.

The honest reading: "add a per-step auxiliary objective" is necessary (algo_grr_state proves it
moves the number) but NOT sufficient. Where the latent is already near-collapsed at init and the
write projection adds a large constant, the objective gets gradient and still loses.

THIS IS NOT NEW INFORMATION. algo_grr_contrast.py already lists "latent MEMORY of the
instance under CE -- 8 attempts, 3 interfaces, all ~0" under DO NOT REBUILD WITHOUT NEW
EVIDENCE. This run is attempt 9 on interface 4, under CE, and it agrees. Read that file's
NEXT STEP before adding a tenth: the open question is the LOSS, not the architecture.

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
import math
import os
import sys
from pathlib import Path

_ROOT = str(Path(__file__).resolve().parents[2])
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import numpy as np
import torch
import torch.nn as nn

from embedder import encode_batch, EMBED_DIM


# ================================================================================================
# gated cross-attention adapter — the LM's hidden states attend to the working-memory slots
# ================================================================================================
class GatedCrossAttn(nn.Module):
    """h (LM hidden [B,S,d]) attends to slots [K,d]; output = h + tanh(g)*proj(attn). g init 0 -> identity."""

    def __init__(self, d: int, n_heads: int = 4, delta_scale: float = 0.3,
                 delta_mode: str = "rescale", gate_init: float = 0.0):
        super().__init__()
        assert d % n_heads == 0
        self.h, self.dh = n_heads, d // n_heads
        self.q = nn.Linear(d, d)
        self.k = nn.Linear(d, d)
        self.v = nn.Linear(d, d)
        self.o = nn.Linear(d, d)
        self.g = nn.Parameter(torch.full((1,), gate_init))
        # CAP the injection at delta_scale*||h|| (was 100% of ||h|| -- a sledgehammer that, combined with an
        # unregularized gate free to swing to tanh~0.97, could overwrite rather than blend with the residual
        # stream, encouraging memorization over a generalizable nudge).
        self.delta_scale = delta_scale
        # "rescale" is the original behaviour and stays the default so every existing caller is
        # bit-identical; "clip" is opt-in. See forward() for why the difference matters.
        self.delta_mode = delta_mode
        for lin in (self.v, self.o):
            nn.init.eye_(lin.weight)
            nn.init.zeros_(lin.bias)

    def forward(self, h: torch.Tensor, slots: torch.Tensor,
                slots_v: torch.Tensor | None = None,
                slot_mask: torch.Tensor | None = None) -> torch.Tensor:
        """slots supplies the KEYS (what each slot is addressable by); slots_v, when given, supplies the
        VALUES (what gets copied out). Splitting them matters because the positional band that makes a
        slot addressable is pure noise once it has been addressed: v and o are eye-initialised, so
        attend->v->o->residual is a direct token-embedding copy, and Qwen ties its embeddings, so an
        injected input-embedding row lifts that token's own logit. Feeding tokemb+band as the value
        corrupts exactly the digit the copy exists to deliver. None keeps the single-stream behaviour."""
        B, S, d = h.shape
        if slots.dim() == 2:
            slots = slots.unsqueeze(0).expand(B, -1, -1)
        if slots_v is None:
            slots_v = slots
        elif slots_v.dim() == 2:
            slots_v = slots_v.unsqueeze(0).expand(B, -1, -1)
        Bk, K, _ = slots.shape
        q = self.q(h).view(B, S, self.h, self.dh).transpose(1, 2)
        k = self.k(slots).view(Bk, K, self.h, self.dh).permute(0, 2, 1, 3)
        v = self.v(slots_v).view(Bk, K, self.h, self.dh).permute(0, 2, 1, 3)
        logits = (q @ k.transpose(-1, -2)) / (self.dh ** 0.5)          # [B, heads, S, K]
        none_fired = None
        if slot_mask is not None:
            # SPIKING SLOT GATE: slot_mask [B,S,K] is 1 where that slot CROSSED THRESHOLD at that
            # position. Masked slots are removed from the softmax, so injection becomes sparse,
            # discrete and per-position instead of a dense softmax over all K at every token.
            m = slot_mask.bool().unsqueeze(1)                          # [B,1,S,K] -> broadcast heads
            none_fired = (~m).all(dim=-1, keepdim=True)                # [B,1,S,1]
            # A position where NOTHING fired must inject nothing. Masking every slot would make the
            # softmax all -inf -> NaN, so the row is left unmasked here and its delta is zeroed
            # below: "no slot crossed threshold" means no injection, not a corrupted residual.
            logits = logits.masked_fill(~m & ~none_fired, float("-inf"))
        att = torch.softmax(logits, dim=-1)
        ctx = (att @ v).transpose(1, 2).reshape(B, S, d)
        delta = self.o(ctx)
        if none_fired is not None:
            delta = delta * (~none_fired[:, 0]).to(delta.dtype).reshape(B, S, 1)
        cap = h.norm(dim=-1, keepdim=True) * self.delta_scale
        dn = delta.norm(dim=-1, keepdim=True) + 1e-6
        if self.delta_mode == "clip":
            # CLIP to the cap instead of RESCALING to it. Rescaling forces every position to receive an
            # injection of exactly delta_scale*||h||, discarding the adapter's own magnitude -- so the
            # channel pushes exactly as hard on `step` and `computed`, where the LM's prior is already
            # right, as on the digit it has to override. That makes per-position emphasis unrepresentable,
            # which is fatal for copying specifically: measured, the narration takes the ADDRESS from the
            # channel (routing 0.963 vs 0.209 chance) and the VALUE from its prior, emitting
            # `step 3 computed 1000 / 2 = 500` for a target of `step 3 computed 108 - 27 = 81` -- right
            # slot, right format, arithmetically self-consistent, invented. Clipping keeps the same bound
            # and lets the adapter stay quiet where it has nothing to add and spend the whole budget where
            # it does.
            delta = delta * torch.clamp(cap / dn, max=1.0)
        else:
            delta = delta / dn * cap
        return h + torch.tanh(self.g) * delta


# ================================================================================================
# ================================================================================================
# AlgorithmicCell — explicit search, binding, and branching for a true reasoning engine
# ================================================================================================



# ================================================================================================
# SpikingSlotGate — LIF dynamics over the K working-memory slots, gating what reaches the LM
# ================================================================================================
class SlotDIM(nn.Module):
    """Differential Impact Mechanism slot memory (user's dim_vs_attention_test.ipynb).

    Replaces the TRM's slot production. The TRM evolves slots by repeatedly applying a shared
    f_mlp(LayerNorm(.)), which MEASURES as a contraction: two slots seeded from genuinely different
    atoms (cos 0.0951) converge to 0.8072 after ONE step and 0.9157 by step 8. Residual connections
    do not fix it (0.7798). That is why every write-side intervention failed -- removing all biases,
    conv_weight 0, the per-step contrastive term, atom-seeded slots -- the forward pass destroyed
    instance information faster than any objective could install it.

    DIM asks a different question. Attention asks "how similar is A to B" via QK^T; DIM asks "how
    violently does this input perturb the system", via the closed-form derivative of the interaction
    state. Slots are then an ACCUMULATOR of input-derived features rather than a recurrently
    transformed latent -- an accumulator is not a contraction -- and a softmax over slots makes them
    COMPETE for each write, which is the thing the TRM had nothing playing the role of.

    On the notebook's char-LM this beat causal attention at equal depth (final CE 0.372 vs 0.573,
    min 0.306 vs 0.491) at O(N*k) instead of O(N^2).

    REPRODUCIBILITY: the result below was measured with NO FFN and depth 1, which was the only
    form SlotDIM had at the time. The FFN is now ON by default, so reproducing that exact number
    requires --slotdim-no-ffn.

    STACKING CONTRACTS WITHOUT THE FFN -- measured at init, across-instance slot cosine:
        depth 1/2/3, ffn OFF:  0.1246 -> 0.4542 -> 0.8983   <- the TRM's failure mode, again
        depth 1/2/3, ffn ON :  0.0539 -> 0.0236 -> 0.0175   <- no contraction
    So the FFN is not boilerplate; it is what makes depth safe. Its cost at init is that the k
    slots move toward each other (0.941 -> 0.995 at depth 2). Neither setting gives both
    properties at init -- and init probes have mispredicted trained behaviour repeatedly here,
    so this is a reason to RUN the arms, not to pick one from the table.

    FIRST REAL RESULT (Qwen2.5-3B 4-bit, 12 epochs, n-train 48, held 16, --slotdim 8 with
    --trm-no-bias --no-proj-bias --contrast-weight 0.5, NO FFN, depth 1):
                                    every previous arm        SlotDIM
        instance-specific (nats)    +0.0009 .. +0.0032        +0.0803
        modal/specific ratio        2278x .. 8378x            91.3x
        slot_cos                    0.999999 .. 1.000000      0.893153
        ct (train)                  pinned at ln(16)=2.7726   3.076 -> 0.730
        WM / DERANGED               5/4, 6/6, 7/8             7/16 / 6/16
    The effect is 25-90x larger and the format-to-content ratio 25-90x smaller, and it is the
    first signal on this module ABOVE the 0.0075 CE null band (by >10x). ct falling through its
    chance floor is the direct evidence that the contraction was the blocker: the same objective
    could not move at all while the TRM recursion produced the slots.

    NOT YET A WIN, stated as loudly: the CI is [-0.0224, +0.1901] and INCLUDES 0, and 91.3x is
    still above the 50x instance-independent-adapter bar. At held=16 the interval half-width is
    ~0.11 against an effect of 0.08, so this run cannot resolve its own headline. The binding
    constraint is N, not the mechanism -- the CE readout is forward-passes only, so held=48-64
    is cheap and is the next thing to run.

    KNOWN OMISSIONS from the notebook block, dropped to isolate the mechanism and NOT tested:
    the FFN (y = y + ffn(y)), the causal cumulative scan, the broadcast read + LayerNorm
    residual, and block stacking. Without the FFN the slots are strictly linear combinations of
    input features gated by sensitivity, so they can SELECT impactful atoms but not COMPOSE
    them; with a single application this is derivative-based pooling, not iterative routing.
    slot_cos plateauing at ~0.95 during training rather than continuing down is what that
    limitation looks like.

    HONEST NOTE ON THE REGIME. Probed at INIT on random vectors, this shows a tension in our setting:
    with few write events instances separate but the k slots receive near-identical content, and with
    many writes the slots differentiate while instances converge (the accumulator regresses to its
    mean). Those probes are NOT being treated as decisive, because init-time separation already
    mispredicted trained behaviour once this session badly enough to require retracting 644343c. The
    notebook's number is a TRAINED end-to-end result; this gets trained and judged by the derangement
    arm and the CE ratio, not by a probe.
    """

    def __init__(self, d: int, k: int = 8, depth: int = 1, ffn: bool = True):
        super().__init__()
        self.k, self.depth = k, max(1, depth)
        self.M0 = nn.Parameter(torch.randn(k, d) * 0.02)
        self.W_v = nn.ModuleList([nn.Linear(d, d, bias=False) for _ in range(self.depth)])
        self.W_read = nn.Linear(d, d, bias=False)
        # FFN + LayerNorm, restored from the notebook block. Dropping them was an untested
        # omission, and it is the load-bearing one: without a non-linearity INSIDE the slot path
        # the slots are strictly linear combinations of input features gated by the sensitivity
        # scores, so they can SELECT impactful atoms but never COMPOSE them into a higher-order
        # relation. slot_cos plateauing at ~0.95 mid-training instead of continuing down is what
        # that ceiling looks like from outside.
        self.ln = nn.ModuleList([nn.LayerNorm(d) for _ in range(self.depth)]) if ffn else None
        self.ffn = nn.ModuleList([
            nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))
            for _ in range(self.depth)]) if ffn else None

    def _write(self, x: torch.Tensor, i: int) -> torch.Tensor:
        """One DIM write pass: x [T,d] -> slot memory [k,d]."""
        M0 = self.M0.unsqueeze(0)                             # (1,k,d)
        Z = x.unsqueeze(0) @ M0.transpose(-2, -1)             # (1,T,k) interaction state
        sig = torch.sigmoid(Z)
        obs = sig * (1.0 + Z * (1.0 - sig))                   # closed-form SiLU derivative
        A = torch.softmax(obs, dim=-1)                        # slots COMPETE for each write
        v = nn.functional.silu(self.W_v[i](x)).unsqueeze(0)   # (1,T,d) candidate writes
        P = (A.unsqueeze(-1) * v.unsqueeze(2)).sum(1)         # (1,k,d) accumulate
        cnt = A.sum(1).clamp(min=1e-6).unsqueeze(-1)
        return (M0 + P / cnt).squeeze(0)                      # (k,d)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x [T, d] write events (task + retrieved atoms) -> slot memory [k, d].

        depth > 1 re-runs the write with the SLOT TABLE as the new write events, so slots refine
        against each other instead of only pooling the inputs once. A single pass is
        derivative-based pooling; iterative routing is what lets a slot resolve a relation that
        spans two atoms rather than just naming the most impactful one.
        """
        if x.dim() == 1:
            x = x.unsqueeze(0)
        cur = x
        for i in range(self.depth):
            M = self._write(cur, i)
            if self.ffn is not None:
                M = self.ln[i](M)
                M = M + self.ffn[i](M)                        # residual FFN, per the notebook block
            cur = M                                           # next pass writes FROM the slots
        return self.W_read(cur)                               # (k,d)


class SlotDIMv2(nn.Module):
    """Slot-DIM v2 (user's upgraded dim_vs_attention_test.ipynb), adapted to emit a SLOT TABLE.

    Three additions over v1, each aimed at a failure this file has measured:

    1. MULTI-HEAD SLOT SETS -- h groups of k/h slots with their own M0 and W_v. v1's slots were
       measured collapsing toward each other (within-instance cosine 0.94 at depth 1, 0.995 with
       FFN at depth 2): every slot competed for the same projection space, so an entity and a
       relation had to fight for orthogonal room. Separate heads give feature specialisation
       without attention's O(N^2).
    2. SELECTIVE INPUT-DEPENDENT DECAY -- lam = sigmoid(W_f(x)) per channel, so the memory policy
       depends on WHAT the input is, not how long ago it arrived. A static EMA/cumsum decays by
       distance and would drop a premise introduced first, which is fatal when the write events are
       a task followed by its retrieved atoms. Computed by exact parallel scan
       M_t = K_t*(M0 + cumsum(b/K)), K_t = prod(keep), in fp64 so 1/K cannot underflow.
    3. DIM-FAITHFUL SENSITIVITY GATE -- gate = tanh(obs) multiplying the write. The softmax over
       slots is only RELATIVE ("if I must write this somewhere, where?"); it will force a write for
       a token that perturbs nothing. tanh(sigma'(Z)) is an ABSOLUTE magnitude gate: no measured
       consequence in the observer, no write. This is the piece that makes the mechanism actually
       DIM rather than linear attention with extra steps.

    Heads keep their own slot sets and are CONCATENATED (not summed as in the notebook, which sums
    per-token reads) because the LM's GatedCrossAttn consumes the slot table itself.
    """

    def __init__(self, d: int, k: int = 8, n_heads: int = 2, decay_bias: float = 2.2,
                 ffn: bool = True, depth: int = 1):
        super().__init__()
        assert k % n_heads == 0, f"k={k} must be divisible by n_heads={n_heads}"
        self.h, self.ks, self.depth = n_heads, k // n_heads, max(1, depth)
        self.M0 = nn.Parameter(torch.randn(self.h, self.ks, d) * 0.02)
        self.W_v = nn.ModuleList([nn.Linear(d, d, bias=False) for _ in range(self.h * self.depth)])
        self.W_f = nn.ModuleList([nn.Linear(d, d) for _ in range(self.h * self.depth)])
        for W in self.W_f:                       # init keep ~ sigmoid(2.2) ~ 0.9, long-ish memory
            W.weight.data.mul_(0.1)
            W.bias.data.fill_(decay_bias)
        self.W_read = nn.Linear(d, d, bias=False)
        self.ln = nn.ModuleList([nn.LayerNorm(d) for _ in range(self.depth)]) if ffn else None
        self.ffn = nn.ModuleList([
            nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))
            for _ in range(self.depth)]) if ffn else None

    def _pass(self, x: torch.Tensor, di: int) -> torch.Tensor:
        """x [T, d] write events -> slot table [k, d] for one depth level."""
        outs = []
        for hh in range(self.h):
            i = di * self.h + hh
            M0 = self.M0[hh]                                   # (ks,d)
            Z = x @ M0.T                                       # (T,ks) interaction
            sig = torch.sigmoid(Z)
            obs = sig * (1.0 + Z * (1.0 - sig))                # closed-form observer
            A = torch.softmax(obs, dim=-1)                     # RELATIVE: slots compete
            gate = torch.tanh(obs)                             # ABSOLUTE: no impact, no write
            lam = torch.sigmoid(self.W_f[i](x))                # (T,d) selective decay
            v = nn.functional.silu(self.W_v[i](x))             # (T,d) candidate writes
            w = A.unsqueeze(-1) * gate.unsqueeze(-1) * v.unsqueeze(1)   # (T,ks,d)
            # LOG-SPACE EXACT SCAN -- no 1/K division anywhere.
            # The first port of this used the notebook's fp64 closed form, M = K*(M0 + cumsum(b/K)).
            # That form's gradient runs through d(1/K)/dkeep ~ -1/K^2, which explodes: at T=128 with
            # keep~0.9, K ~ 1e-6 and 1/K ~ 1e6, and the user hit NaN loss from AddmmBackward0 in W_f
            # on exactly this path. fp64 only postpones it. The recurrence form
            # M_t = keep*M_{t-1} + b_t has bounded gradients (products of keep <= 1); this log-space
            # rewrite keeps the closed form's parallelism while inheriting that boundedness, because
            # exp(logb - logK) never materialises 1/K as a value the backward pass differentiates
            # through. keep is additionally clamped to [0.7, 1.0] to bound logK over the window.
            keep = lam.unsqueeze(1).clamp(min=0.7, max=1.0)    # (T,1,d)
            b = (1.0 - keep) * w                               # (T,ks,d)
            logK = torch.cumsum(torch.log(keep.clamp(min=1e-3)), dim=0)
            logb = torch.log(b.abs().clamp(min=1e-12))
            S = torch.cumsum(torch.sign(b) * torch.exp(logb - logK), dim=0)
            M_seq = torch.exp(logK) * (M0.unsqueeze(0) + S)
            outs.append(M_seq[-1])                             # FINAL slot state (ks,d)
        return torch.cat(outs, 0)                              # (k,d)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 1:
            x = x.unsqueeze(0)
        cur = x
        for di in range(self.depth):
            M = self._pass(cur, di)
            if self.ffn is not None:
                M = self.ln[di](M)
                M = M + self.ffn[di](M)
            cur = M
        return self.W_read(cur)


class SlotContrast(nn.Module):
    """A PER-RECURSION-STEP contrastive objective on the working-memory latents.

    WHY THIS EXISTS. The only direct per-step force on y_t was conv_loss, which penalizes
    ||y_{t+1} - y_t|| and therefore literally rewards NOT CHANGING, while deep supervision
    (--ds-weight) defaults to 0 and was OFF in every run recorded here. A real 40-epoch run on
    Qwen2.5-3B shows `ds 0.0000` at every epoch with `conv` active throughout, and ended at
    across-task slot cosine 0.9880-1.0000 with a deranged slot table scoring 8/16 against the real
    one's 7/16. The collapse was not a failure to learn: a constant slot table is the OPTIMUM of the
    objective that was actually being minimized.

    WHY CONTRASTIVE RATHER THAN THE EXISTING DS. Deep supervision here is MSE onto an oracle
    INTERMEDIATE RESULT, so it only exists for synthetic composition -- math-cot and swe-action skip
    it per-example, which is most of what this file is meant to scale to. A contrastive term needs
    no oracle: the negatives are other tasks in the run. And it cannot be satisfied by collapse. If
    y_t is the same vector for every task, the query is identical for every anchor while the
    positive moves, so the loss sits at log(1+K) no matter how long it trains. Across-task cosine
    stops being a thing you measure afterwards and becomes the thing being optimized.

    HONEST NOTE ON DIFFICULTY. The slots are refined from task_emb, so a near-pass-through of the
    task embedding would already score well here. That is fine and is the point: the bar is that the
    working memory RETAIN the task's identity, which right now it does not do at all. Whether the
    retained identity is USEFUL is a separate question, and the derangement + CE arms in run_real
    are what answer it -- this term must never be read as evidence on its own.

    No bias on either projection: a bias 2.09x the signal collapsed separation to 0.9170 in the
    prefix module, three times in one session.
    """

    def __init__(self, d_trm: int, d_emb: int = EMBED_DIM, d: int = 256):
        super().__init__()
        self.q = nn.Linear(d_trm, d, bias=False)
        self.k = nn.Linear(d_emb, d, bias=False)
        self.scale = nn.Parameter(torch.tensor(10.0))

    def logits(self, y_t: torch.Tensor, cands: torch.Tensor) -> torch.Tensor:
        """y_t [d_trm]; cands [M, d_emb] -> [M]."""
        q = nn.functional.normalize(self.q(y_t), dim=-1)
        k = nn.functional.normalize(self.k(cands), dim=-1)
        return self.scale * (k @ q)


def slot_contrast_loss(head, states_tensor, tgt_embs, neg_bank, rng, n_neg: int = 15,
                       exclude_rows=None):
    """InfoNCE at EVERY recursion step. states_tensor [B, T, d_trm]; tgt_embs [B, d_emb] this
    batch's targets; neg_bank [M, d_emb] target embeddings from OTHER tasks.

    The positive sits at a RANDOM index so slot position carries no information, and the effective
    candidate count is returned so a caller can report chance from what was actually scored rather
    than from what was requested.

    exclude_rows[b] is example b's OWN row in neg_bank, and it must be excluded. neg_bank holds
    every TRAIN target, so without this an example can draw its own target as a "negative" -- the
    positive then appears twice in the candidate list under two different indices, one of which is
    scored as wrong. That is unlearnable by construction and silently caps the achievable accuracy
    for a reason having nothing to do with the model. Same self-as-negative bug already fixed in
    FutureBank in algo_grr_state.py.
    """
    if neg_bank is None or neg_bank.shape[0] < 2:
        return None, 0
    B, T = states_tensor.shape[0], states_tensor.shape[1]
    terms, n_cand_seen = [], []
    for b in range(B):
        pool = list(range(neg_bank.shape[0]))
        if exclude_rows is not None and exclude_rows[b] is not None:
            own = exclude_rows[b]
            pool = [r for r in pool if r != own]
        if len(pool) < 1:
            continue
        k = min(n_neg, len(pool))
        pick = torch.tensor(rng.sample(pool, k), device=neg_bank.device)
        negs = neg_bank[pick]
        cands = torch.cat([tgt_embs[b].unsqueeze(0), negs], 0)
        j = rng.randrange(cands.shape[0])
        order = list(range(cands.shape[0]))
        order[0], order[j] = order[j], order[0]
        cands = cands[order]
        tgt = torch.tensor([j], device=states_tensor.device)
        for t in range(T):
            lg = head.logits(states_tensor[b, t], cands)
            terms.append(nn.functional.cross_entropy(lg.unsqueeze(0), tgt))
        n_cand_seen.append(cands.shape[0])
    if not terms:
        return None, 0
    return torch.stack(terms).mean(), int(np.mean(n_cand_seen))


def per_step_slot_cosine(slot_list) -> list:
    """Across-task cosine at EACH recursion step, not pooled over them.

    The existing slot_cos flattens [T, d_lm] into one vector per task, so a table that is distinct
    at step 0 and constant by step T reads as a single middling number. Collapse is a per-step
    phenomenon -- conv_loss is weighted QUADRATICALLY toward later steps, so that is exactly where
    it should bite -- and this is the diagnostic that shows it.
    """
    if not slot_list:
        return []
    S = torch.stack([s.float() for s in slot_list])          # [N, T, d_lm]
    if S.dim() != 3 or S.shape[0] < 2:
        return []
    out = []
    for t in range(S.shape[1]):
        M = S[:, t]
        M = M / M.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        C = M @ M.T
        off = C[~torch.eye(len(M), dtype=torch.bool, device=C.device)]
        out.append(float(off.mean()))
    return out


class SpikingSlotGate(nn.Module):
    """Decides, PER TOKEN POSITION, which working-memory slots are allowed to inject.

    Today every position attends a dense softmax over all K slots through one scalar gate. That is
    the regime the soft-prompt experiment measured collapsing: scaled up, it emitted wrong-gadget
    bodies -- ROUTING COLLAPSE, slots losing their distinct identities. The three LIF mechanisms map
    onto that failure directly:

      lateral inhibition   slot i firing suppresses slots CORRELATED with it (U = slot-slot cosine,
                           off-diagonal). Inhibition between similar units is literally
                           decorrelation, so it is an anti-collapse prior rather than a penalty
                           bolted on after the fact.
      threshold            a slot injects only where its evidence crosses theta, so addressing is
                           DISCRETE. The soft-prompt result's own conclusion was that a working
                           latent memory has to be discrete addressing; a spike is exactly that.
      homeostasis          a slot that wins at every position raises its own bar, so no single slot
                           can dominate the whole sequence -- the collapse mode in one variable.

    Drive is cos(h_position, slot), so which slots light up is CONTENT-DEPENDENT and varies along
    the sequence: the token emitting a number can attend a different slot than the token emitting
    the connective.

    CLIP-COMPATIBLE. The output is a binary mask, so the injected delta is a sum over SELECTED
    slots with no renormalisation. This repo measured that rescaling an injection to a fixed
    fraction of ||h|| makes per-position emphasis unrepresentable and lets the adapter learn a
    format prior that survives its own ablation; a mask changes which slots are read, never the
    magnitude law.

    PRIOR PRESERVATION: `enabled=False` returns None and GatedCrossAttn takes its original path,
    bit-for-bit. With w_inh=0 and theta0 below the minimum cosine every slot fires, which is the
    dense softmax again. Any gain has to be shown on top of that.
    """

    def __init__(self, T: int = 3, tau: float = 0.8, alpha: float = 0.6, beta: float = 0.02,
                 theta0: float = 0.0, w_inh: float = 0.5, enabled: bool = False,
                 spike_mode: str = "hard", surrogate_gamma: float = 5.0):
        super().__init__()
        self.T, self.tau, self.beta = T, tau, beta
        self.enabled, self.spike_mode, self.surrogate_gamma = enabled, spike_mode, surrogate_gamma
        # Learnable: the threshold, how hard slots inhibit each other, how fast they habituate.
        self.theta0 = nn.Parameter(torch.tensor(float(theta0)))
        self.w_inh = nn.Parameter(torch.tensor(float(w_inh)))
        self.alpha = nn.Parameter(torch.tensor(float(alpha)))

    def _spike(self, v, theta):
        z = v - theta
        if self.spike_mode == "hard":
            return (z > 0).float()
        soft = torch.sigmoid(self.surrogate_gamma * z)                  # straight-through
        return (z > 0).float() + (soft - soft.detach())

    def forward(self, h: torch.Tensor, slots: torch.Tensor) -> torch.Tensor | None:
        """h [B,S,d], slots [K,d] or [B,K,d] -> mask [B,S,K] of 1.0 where the slot fired."""
        if not self.enabled:
            return None
        if slots.dim() == 3:
            slots = slots[0]
        hn = h / (h.norm(dim=-1, keepdim=True) + 1e-6)
        sn = slots / (slots.norm(dim=-1, keepdim=True) + 1e-6)
        drive = hn @ sn.t()                                             # [B,S,K] cosine, per position
        B, S, K = drive.shape
        # Lateral matrix from slot-slot similarity: similar slots are RIVALS for the same content.
        U = (sn @ sn.t()).clamp(min=0)
        U = U - torch.diag_embed(torch.diagonal(U))                     # no self-inhibition
        v = torch.zeros_like(drive)
        theta = self.theta0.expand(B, S, K).clone()
        y = torch.zeros_like(drive)
        for _ in range(self.T):
            v = self.tau * v + drive - self.w_inh * (y @ U.t())
            y = self._spike(v, theta)
            v = v * (1.0 - y)                                           # reset on fire
            theta = theta + self.alpha * y - self.beta
        # THE MASK IS THE SETTLED STATE, not the union over steps. Accumulating "ever fired" makes
        # inhibition a no-op by construction: y_prev is zero on the first step, so every slot above
        # threshold fires before anything can suppress it, and a union then locks that dense set in
        # permanently. Measured: with the union, two near-identical slots co-fired at every
        # inhibition strength. Returning the last step lets the competition actually resolve.
        return y


# ================================================================================================
# WMReasoner — the working memory + its recursive refinement + the coupling hooks + deep supervision
# ================================================================================================
class WMReasoner(nn.Module):
    """Working memory slots produced by TRMReasoner (proper two-latent Tiny Recursive Model),
    projected to LM space, then read by the LM via gated cross-attention adapters.

    DEEP SUPERVISION: intermediate y_t values from each TRM cycle are regressed against
    oracle-computed intermediate targets (native_text_embedding of true intermediate results).
    Loss is MSE in d_lm space — NOT CE against atom pools (TRM is not a ranker)."""
    def __init__(self, d_lm: int, couple_layers, trm, n_heads: int = 4, M: int = 4, top_trm=None,
                 gate_init: float = 0.0, delta_scale: float = 0.3,
                 delta_mode: str = "rescale"):
        super().__init__()
        self.T = trm.T
        self.M = M
        self.trm = trm                                         # BOTTOM/fast TRM (two-latent), reaches the LM

        # Project TRM's y_t [T, d] → [T, d_lm] for LM adapters
        self.proj_y = nn.Linear(trm.d, d_lm)

        # Gated cross-attention adapters (unchanged)
        self.adapters = nn.ModuleList([
            GatedCrossAttn(d_lm, n_heads, gate_init=gate_init,
                           delta_scale=delta_scale, delta_mode=delta_mode)
            for _ in couple_layers])
        self.couple_layers = list(couple_layers)
        # SPIKING SLOT GATE (opt-in; disabled => the adapters take their original dense-softmax
        # path bit-for-bit, so an existing checkpoint behaves identically).
        self.slot_gate = SpikingSlotGate(enabled=False)
        self._slots = None
        # Optional separate VALUE stream (see GatedCrossAttn.forward). None => keys are also the values.
        self._slots_v = None
        # False => the original free-latent behaviour, bit-identical. See _refine_atom_slots.
        self.atom_slots = False
        self.slotdim = None          # set by run_real when --slotdim is passed

        # Deep supervision: map y_t [d] → d_lm for MSE against native_text_embedding targets
        self.ds_proj = nn.Linear(trm.d, d_lm)

        # Self-critique (unchanged)
        # LayerNorm BEFORE the critic's tanh -- without it the critic is a literal constant predictor.
        # Measured on a real trained checkpoint (artifacts/wm_baseline_notop.pt): raw_states (the projected
        # y_t the critic is fed) have a pooled-vector norm of ~312,000, giving pre-tanh activations of
        # ~145,410 against a tanh that saturates by |x|~3. Result: 100% of units saturated, an IDENTICAL
        # +/-1 pattern for every task (pairwise cosine between different tasks = 1.000000), and
        # critique() returning exactly 1.000000 regardless of input. That fully explains why the critic
        # "never beat base rate" in any run here -- its accuracy came out at exactly 1 - base_rate (0.38 vs
        # 0.62, 0.22 vs 0.78, 0.48 vs 0.52), the signature of always predicting one class. It was not a
        # weak signal; the input was constant and the saturated tanh passed ~zero gradient, so no amount of
        # training or class balancing could have helped.
        self.critic_norm = nn.LayerNorm(d_lm)
        self.critic_pool = nn.Linear(d_lm, d_lm)
        # Projects concat(task_emb, generated_text_emb) into the critic's space -- see critique()'s ctx
        # argument. This is what turns the critic from "inspect my own hidden trajectory" into "compare
        # what I was asked for against what I actually produced," which is the only version that has a
        # chance of generalizing to domains where no verifier exists.
        self.critic_ctx = nn.Linear(EMBED_DIM * 2, d_lm)
        self.critic = nn.Sequential(nn.Linear(d_lm, d_lm // 2), nn.GELU(), nn.Linear(d_lm // 2, 1))

        # EVENT SIGNALS for the BOTTOM trm. The hierarchical design put these on the top trm only
        # (top_event_emb, created inside the `if top_trm is not None` branch below), so in MERGED mode --
        # where there IS no top trm -- generate_with_reground computed `pending_event` on every chunk and
        # then dropped it on the floor: the event triggers fired, and nothing in the network could receive
        # them. This is the bottom-level receiver, so events remain real input in both architectures.
        # Zero-init => a strict no-op until trained, same safe-by-construction convention as
        # top_to_bottom_proj and GatedCrossAttn's gate.
        #   0 = no event, 1 = trigger_patterns fired, 2 = instability fired, 3 = both
        self.event_emb = nn.Embedding(4, trm.d_in)
        nn.init.zeros_(self.event_emb.weight)

        # HIERARCHICAL (optional): a second, slower-timescale TRM (top_trm, same TRMReasoner class, its
        # own T -- the real TRM paper's own recipe runs many recursion steps for hard tasks, e.g. ~24; this
        # is exactly why TRMReasoner already takes T as a free parameter, not hardcoded). top_trm "manipulates"
        # the bottom trm (which is the one that reaches the LM) by injecting its own deeply-reasoned output
        # additively into the bottom trm's task input every time the bottom trm runs -- see
        # hierarchical_refine(). top_to_bottom_proj is ZERO-INIT (weight AND bias) so that a freshly-added,
        # untrained top_trm is a strict no-op at first: hierarchical_refine(...) == refine(...) bit-for-bit
        # until top_to_bottom_proj actually learns something -- an EXISTING trained checkpoint's behavior is
        # preserved exactly if you attach a fresh top_trm to it, matching the same safe-by-init-zero
        # convention GatedCrossAttn's gate already uses.
        self.top_trm = top_trm
        if top_trm is not None:
            self.top_to_bottom_proj = nn.Linear(top_trm.d, trm.d_in)
            nn.init.zeros_(self.top_to_bottom_proj.weight)
            nn.init.zeros_(self.top_to_bottom_proj.bias)
            # EVENT SIGNALS the top trm actually RECEIVES, rather than events merely acting as a clock that
            # decides WHEN it recomputes. A learned vector per event cause is added to the top trm's context
            # input, so it can react differently to "routine cadence tick" vs "the bottom trm was still
            # churning" vs "a real reasoning boundary appeared in the text". Zero-init => a strict no-op
            # until trained, matching top_to_bottom_proj's and GatedCrossAttn's own safe-by-init convention.
            #   0 = cadence (routine), 1 = trigger_patterns fired, 2 = instability fired, 3 = both fired
            self.top_event_emb = nn.Embedding(4, top_trm.d_in)
            nn.init.zeros_(self.top_event_emb.weight)

    def critique(self, raw_states: list[torch.Tensor], ctx: torch.Tensor | None = None) -> torch.Tensor:
        """raw_states: [T per-step projected y_t values in d_lm space] from refine(track_deltas=True).

        ctx (optional): [2*EMBED_DIM] = concat(task_emb, generated_text_emb), both MiniLM-space. This is
        the "sense of mistake" input. Without it the critic only ever saw the TRM's internal trajectory --
        never the task it was solving, and never WHAT IT ACTUALLY PRODUCED. Judging your own answer without
        looking at the answer is close to hopeless, and the trajectory is additionally squeezed toward a
        fixed point by the conv_loss regularizer during training, so it carries little task-specific
        variance by design. Passing ctx lets the critic compare intent against output, which is the signal
        a verifier-free mistake detector actually needs."""
        state = torch.stack(raw_states, dim=0)  # [T, d_lm]
        state = self.critic_norm(state)               # keeps tanh out of saturation -- see __init__
        pooled = torch.tanh(self.critic_pool(state)).mean(0, keepdim=True)   # [1, d_lm]
        if ctx is not None:
            pooled = pooled + torch.tanh(self.critic_ctx(ctx.unsqueeze(0).to(pooled.dtype)))
        return torch.sigmoid(self.critic(pooled)).squeeze()

    def critic_loss(self, raw_states_batch: list[list[torch.Tensor]], labels: list,
                    ctxs: list | None = None) -> torch.Tensor:
        if ctxs is None:
            preds = torch.stack([self.critique(s) for s in raw_states_batch])
        else:
            preds = torch.stack([self.critique(s, c) for s, c in zip(raw_states_batch, ctxs)])
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

    def _slots_from(self, y_t: torch.Tensor) -> torch.Tensor:
        """Project per-cycle y_t [T, trm.d] into the LM's space, applying ACT halting when it is on.

        Halting used to be DECORATIVE. TRMReasoner computed halt_weights, refine() stored them on
        self._halt_weights, and nothing ever read them: slots were proj_y(y_t) over all T steps, so an
        `--adaptive-t` run injected every step at full strength no matter what the halt head predicted. The
        only consumer of adaptivity at all was ponder_loss (via _n_steps), i.e. the model paid a cost for
        steps it was never allowed to skip.

        Weights are normalized by their MAX, not used raw. halt_weights sum to 1 across T, so scaling
        directly would shrink every slot by ~1/T and gut the injection strength -- that would confound
        "the model chose to halt" with "the adapter got quieter". Dividing by the max leaves the step the
        model actually committed to at full strength and attenuates the ones it wanted to skip, which is
        soft truncation, shape-stable, and differentiable."""
        slots = self.proj_y(y_t)                               # [T, d_lm]
        if self.trm.adaptive and self._halt_weights is not None:
            w = self._halt_weights.to(slots.dtype)
            slots = slots * (w / w.max().clamp(min=1e-6)).unsqueeze(-1)
        return slots

    def _refine_slotdim(self, task_emb: torch.Tensor, atom_embs: torch.Tensor,
                        track_deltas: bool = False):
        """Slots produced by the DIM accumulator instead of the TRM's recursion.

        Write events are the task embedding plus each retrieved atom, projected into the TRM's space
        with the SAME projections the TRM uses, so the only thing that differs from the other paths is
        HOW the slot table is formed. There is no recursion here by design -- the recursion is the
        measured contraction (0.0951 -> 0.8072 in one step) -- so `states` is the slot state repeated
        T times, which makes conv_loss exactly 0 (there is no step-to-step change to penalise) and
        leaves the per-step contrastive term operating on the real slot content.
        """
        writes = [self.trm.task_proj(task_emb)]
        for k in range(int(atom_embs.shape[0])):
            writes.append(self.trm.atom_proj(atom_embs[k]))
        X = torch.stack(writes)                                # [1+K, d]
        M = self.slotdim(X)                                    # [k_slots, d]
        slots = self.proj_y(M)                                 # [k_slots, d_lm]
        self._slots = slots
        self._halt_weights, self._n_steps = None, None
        pooled = M.mean(0)
        states = [pooled for _ in range(self.T)]
        if track_deltas:
            return slots, states, [0.0], [s_.detach() for s_ in slots]
        return slots, states

    def _refine_atom_slots(self, task_emb: torch.Tensor, atom_embs: torch.Tensor,
                           track_deltas: bool = False):
        """K SLOTS, ONE PER RETRIEVED ATOM, refined over T recursion steps.

        This is the design this file's header has described since it was written -- "K slots the TRM
        refines over T recursion steps, INITIALIZED from the top-K retrieved atoms -> grounded in real
        content, not a free latent (this is what killed soft-prompt)" -- and which was never built.

        What ran instead, verified by measurement:
          * the slot count was T (recursion steps), NEVER K: 1 atom and 16 atoms both produced 4 slots
          * z0 and y0 are nn.Parameter zeros, so EVERY instance started from the same latent, i.e. the
            free latent the header warns against
          * the cycle-0 attention query is exactly z0 = 0, so softmax over the atoms was exactly
            uniform ([0.3333, 0.3333, 0.3333]) and PERMUTING the retrieved atoms changed nothing --
            retrieval rank carried no information at all, and the atoms entered only as a mean
        That is enough to explain every null measured on this module: the slots could not encode WHICH
        atoms were retrieved, so a deranged slot table scored the same as the real one (WM 5/16 vs
        deranged 4/16, ratio 2278x-8378x), and no auxiliary objective could extract information the
        architecture never routed in. Removing every bias (0.885 -> 0.364 at init) and switching off
        conv_loss both left slot_cos at ~0.9997, which is what a structural cause looks like.

        Each slot is seeded with its OWN atom and refined against the shared task/context, so the slot
        count tracks K, rank is preserved (slot k is atom k), and the starting point is instance-
        specific by construction rather than by hoping an objective creates separation.
        """
        K = int(atom_embs.shape[0])
        per_slot, finals = [], []
        for k in range(K):
            y0k = self.trm.atom_proj(atom_embs[k])          # SEED slot k from atom k
            if self.trm.adaptive:
                y_t, halt_w, n_steps = self.trm(task_emb, atom_embs, y_init=y0k)
                self._halt_weights, self._n_steps = halt_w, n_steps
            else:
                y_t = self.trm(task_emb, atom_embs, y_init=y0k)
                self._halt_weights, self._n_steps = None, None
            per_slot.append(y_t)                            # [T, d]
            finals.append(y_t[-1])                          # refined value of slot k
        Y = torch.stack(finals)                             # [K, d]
        slots = self.proj_y(Y)                              # [K, d_lm] -- COUNT NOW TRACKS K
        self._slots = slots
        P = torch.stack(per_slot)                           # [K, T, d]
        # states stay a length-T list of [d] so conv_loss / ds_loss / the contrastive term are
        # unchanged; each entry is the slot table's state at recursion step t, pooled over slots.
        states = [P[:, t].mean(0) for t in range(P.shape[1])]
        if track_deltas:
            deltas = ([float((P[:, t + 1] - P[:, t]).norm()) for t in range(P.shape[1] - 1)]
                      if P.shape[1] > 1 else [0.0])
            return slots, states, deltas, [s_.detach() for s_ in slots]
        return slots, states

    def refine(self, task_emb: torch.Tensor, atom_embs: torch.Tensor, native: bool = False,
              track_deltas: bool = False):
        """task_emb [d_in], atom_embs [N, d_in] (both in TRM's d_in space, typically MiniLM 384-d).
        Runs the proper TRM (two-latent, cross-attn) to produce per-cycle y_t solution embeddings,
        projects to d_lm for GatedCrossAttn adapters.

        native=True is ACCEPTED for backward compat but is a NO-OP and must not be read as "pass me
        LM-native embeddings": task_proj/atom_proj are both nn.Linear(d_in=EMBED_DIM, d), so d_lm-space
        input (e.g. native_text_embedding's 2560-d vectors on Qwen3-4B) is an immediate shape error.
        membrane.py's WM path really did pass those, and it crashed on the first call -- nothing had ever
        constructed a Membrane(wb=..., wm=...) so it was never executed. Inputs here are MiniLM."""
        if getattr(self, "slotdim", None) is not None:
            return self._refine_slotdim(task_emb, atom_embs, track_deltas=track_deltas)
        if getattr(self, "atom_slots", False):
            return self._refine_atom_slots(task_emb, atom_embs, track_deltas=track_deltas)
        if self.trm.adaptive:
            y_t, halt_w, n_steps = self.trm(task_emb, atom_embs)  # y_t [T, d], halt_w [T]
            self._halt_weights = halt_w
            self._n_steps = n_steps
        else:
            y_t = self.trm(task_emb, atom_embs)                    # [T, trm.d]
            self._halt_weights = None
            self._n_steps = None
        slots = self._slots_from(y_t)                          # [T, d_lm] — working memory
        self._slots = slots
        states = [y_t[i] for i in range(self.T)]                # per-step y_t for DS

        if track_deltas:
            deltas = [(y_t[i + 1] - y_t[i]).norm().item() for i in range(self.T - 1)] if self.T > 1 else [0.0]
            raw_states = [s.detach() for s in slots]            # projected y_t in d_lm space for critic
            return slots, states, deltas, raw_states
        return slots, states

    def recurrent_refine(self, task_emb: torch.Tensor, atom_embs: torch.Tensor,
                         memory: torch.Tensor | None = None, resume_state: tuple | None = None,
                         track_deltas: bool = False, event_signal: int = 0):
        """MERGED architecture: the bottom TRM does the top TRM's job itself, and there is no top TRM.

        The top level's only real responsibilities were (a) carrying latent state across chunks,
        (b) attending its own accumulated memory rather than the graph, (c) absorbing evicted KV spans and
        (d) receiving event signals. None of those need a SECOND network -- the bottom trm already has the
        same two-latent z/y machinery, and TRMReasoner already accepts z_init/y_init/return_state (added
        for the top, reused here).

        The decisive argument for merging is not simplicity, it is the removal of a bottleneck that
        provably broke things. In the hierarchical version, memory could only reach the LM through
        top_to_bottom_proj -- a ZERO-INITIALIZED linear. While that projection sat near zero the bottom trm
        received the unchanged task embedding, produced identical slots, and re-grounding was a
        mathematical no-op; that is why `reground` came back byte-identical to `held WM` in every run this
        codebase ever recorded. Here memory is concatenated straight into the cross-attention context the
        bottom trm already reads, so it reaches the LM through the same trained path the graph atoms do,
        with no zero-init gate in between.

        Supporting evidence that the second network was not paying for itself: a controlled 3-arm A/B at
        identical settings gave bottom-only 15/16, hierarchical-on-graph-atoms 14/16, and
        top-as-memory 15/16 -- the extra level never won.

        memory: [M, d_in] accumulated context (progress embeddings, evicted spans) -- concatenated with
        atom_embs so the trm attends graph knowledge and its own history in one place.
        resume_state: (z, y) from this method's previous call, giving real cross-call recurrence.
        event_signal: the CAUSE of this call (0 none / 1 pattern / 2 instability / 3 both), added to the
        task input through the zero-init event_emb -- the merged path's replacement for the top trm's
        top_event_emb. Without it the event triggers in generate_with_reground were computed every chunk
        and then discarded in merged mode, so the "event-based" half of the design was inert there.
        Returns the same shapes refine() does, plus the new (z, y) state appended."""
        ctx = atom_embs
        if memory is not None and memory.numel():
            mem = memory if memory.dim() == 2 else memory.unsqueeze(0)
            ctx = torch.cat([atom_embs, mem.to(atom_embs.dtype)], dim=0)
        if event_signal:
            task_emb = task_emb + self.event_emb(
                torch.tensor(int(event_signal), device=task_emb.device))
        z_init, y_init = resume_state if resume_state is not None else (None, None)
        # ADAPTIVE-SAFE unpack. TRMReasoner.forward returns (ys, state) normally but (ys, state,
        # halt_weights, n_steps) when adaptive=True -- this used to unpack unconditionally into two names,
        # so `--merged --adaptive-t` died on "too many values to unpack" before producing a single token.
        # refine() had the adaptive branch; this method never got it.
        out = self.trm(task_emb, ctx, z_init=z_init, y_init=y_init, return_state=True)
        if self.trm.adaptive:
            y_t, new_state, halt_w, n_steps = out
            self._halt_weights, self._n_steps = halt_w, n_steps
        else:
            y_t, new_state = out
            self._halt_weights, self._n_steps = None, None
        slots = self._slots_from(y_t)
        self._slots = slots
        states = [y_t[i] for i in range(self.T)]
        if track_deltas:
            deltas = ([(y_t[i + 1] - y_t[i]).norm().item() for i in range(self.T - 1)]
                      if self.T > 1 else [0.0])
            raw_states = [s.detach() for s in slots]
            return slots, states, deltas, raw_states, new_state
        return slots, states, new_state

    def hierarchical_refine(self, task_emb: torch.Tensor, atom_embs: torch.Tensor,
                            top_context_emb: torch.Tensor | None = None,
                            top_state: torch.Tensor | None = None, recompute_top: bool = True,
                            track_deltas: bool = False, top_resume_state: tuple | None = None,
                            top_memory: torch.Tensor | None = None, top_no_graph: bool = False,
                            event_signal: int = 0):
        """Top TRM (slow timescale, its own T -- can be much larger than the bottom trm's, e.g. 24, matching
        the real TRM paper's own recipe for hard tasks) manipulates the bottom trm (fast timescale, the one
        that actually reaches the LM via GatedCrossAttn) by injecting its own deeply-reasoned output
        additively into the bottom trm's task input every time the bottom trm runs.

        Two-way information flow, as asked for: top_context_emb defaults to the SAME task_emb the bottom
        trm sees when nothing else is passed, but the caller (generate_with_reground) feeds it the embedding
        of the partial generation-so-far on each call -- so the top TRM's own reasoning is grounded in what
        the LM has actually written, not just the original static task description. That is the bottom-to-
        top flow; the top-to-bottom flow is top_signal added into the bottom trm's task input below.

        recompute_top=False lets the caller reuse a previously-computed top_state instead of re-running the
        top trm's full (possibly large-T, expensive) recursion on every single bottom tick -- this is the
        actual cadence mechanism ("top runs slower than bottom"), not real OS threads: two forward passes
        sharing one CUDA context under Python's GIL don't run concurrently in any meaningful sense, so a
        literal thread wouldn't buy real wall-clock parallelism here. Running top less often (every K bottom
        ticks) is what actually gives the fast/slow timescale split, and it's simple and correct.

        top_resume_state: the top TRM's REAL cross-call recurrent memory -- its raw (z, y) latents from the
        end of the PREVIOUS recompute, fed back in as this call's z_init/y_init (see TRMReasoner.forward).
        Without this, every recompute started over from the fixed learned z0/y0 -- recomputed periodically,
        never actually remembering anything about earlier recomputes. That's genuinely different from the
        top TRM's OWN T inner think/act cycles (always recurrent, within one call); this is recurrence
        ACROSS separate calls, the thing that makes top a real evolving memory instead of a periodic
        stateless recompute. None (default) = starts top_trm fresh from z0/y0, matching the original
        behavior for any caller that doesn't thread this through.

        If self.top_trm is None, behaves EXACTLY like refine() (no top_state/resume state ever produced) --
        this method is purely additive, never required."""
        if self.top_trm is None:
            out = self.refine(task_emb, atom_embs, track_deltas=track_deltas)
            return (*out, None, None)

        if recompute_top or top_state is None:
            top_ctx = task_emb if top_context_emb is None else top_context_emb
            # feed the event CAUSE in as real input, not just as the reason we're running now
            if event_signal:
                top_ctx = top_ctx + self.top_event_emb(
                    torch.tensor(int(event_signal), device=top_ctx.device))
            z_init, y_init = top_resume_state if top_resume_state is not None else (None, None)
            # top_no_graph: the top trm is RECURRENT MEMORY, not a second retriever. It cross-attends over
            # its OWN accumulated history (top_memory -- the sequence of progress contexts it has seen this
            # generation), NOT over the graph atoms the bottom trm uses. Without this the two levels receive
            # IDENTICAL input (same K atoms) and the "hierarchy" is only a cadence split -- largely the same
            # computation run twice, which is what a real 40-epoch Qwen3-4B A/B measured (14/16 vs ~13/16
            # baseline, i.e. noise). The division of labour: top = evolving memory over its own experience,
            # bottom = controller/communicator that holds the graph atoms and actually reaches the LM.
            top_context = atom_embs
            if top_no_graph:
                top_context = top_memory if (top_memory is not None and top_memory.numel())                     else top_ctx.unsqueeze(0)
            top_y, (new_z, new_y) = self.top_trm(top_ctx, top_context, z_init=z_init, y_init=y_init,
                                                  return_state=True)
            top_state = top_y[-1]                                # final cycle's answer, the "meta-plan"
            top_resume_state = (new_z, new_y)                    # carried to the NEXT recompute
        top_signal = self.top_to_bottom_proj(top_state)          # zero-init -> no-op until trained
        effective_task_emb = task_emb + top_signal
        out = self.refine(effective_task_emb, atom_embs, track_deltas=track_deltas)
        return (*out, top_state, top_resume_state)

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

    def set_slots_direct(self, slots: torch.Tensor, slots_v: torch.Tensor | None = None):
        self._slots = slots.unsqueeze(0) if slots.dim() == 1 else slots
        # Always assigned, never left over: a stale value stream from a previous ask would be read as
        # this ask's content and would be invisible in every metric.
        self._slots_v = None if slots_v is None else (
            slots_v.unsqueeze(0) if slots_v.dim() == 1 else slots_v)

    def clear(self):
        self._slots = None
        self._slots_v = None

    def save(self, path: str):
        """Persist the trained adapter + TRMReasoner (+ top_trm, if this is a hierarchical WMReasoner)."""
        blob = {
            "state_dict": self.state_dict(),
            "d_lm": self.proj_y.out_features,
            "couple_layers": self.couple_layers,
            "T": self.T,
            "trm_d": self.trm.d,
            "trm_d_in": self.trm.d_in,
            # Recorded so a reloader can rebuild the SAME TRMReasoner. Without it, an --adaptive-t
            # checkpoint reloads as a non-adaptive model and its halt_head weights are dropped in silence
            # by load_state_dict(strict=False) -- the checkpoint would quietly lose its halting policy.
            "trm_adaptive": bool(getattr(self.trm, "adaptive", False)),
            "n_heads": self.adapters[0].h if len(self.adapters) else 4,
        }
        if self.top_trm is not None:
            blob["top_trm_d"] = self.top_trm.d
            blob["top_trm_d_in"] = self.top_trm.d_in
            blob["top_trm_T"] = self.top_trm.T
        torch.save(blob, path)

    @classmethod
    def load(cls, path: str, trm, map_location=None, top_trm=None) -> "WMReasoner":
        """Reconstruct a WMReasoner from a save()'d checkpoint.
        Requires an already-constructed TRMReasoner instance (passed as `trm`). Pass top_trm to attach a
        hierarchical top-level TRM -- if the checkpoint was saved WITHOUT one, top_trm here is treated as a
        freshly-added (untrained, zero-init-projection) addition, safe by construction (see __init__'s
        docstring on top_to_bottom_proj); if the checkpoint WAS saved with one, its state_dict entries load
        via strict=False below."""
        blob = torch.load(path, map_location=map_location, weights_only=False)
        R = cls(blob["d_lm"], blob["couple_layers"], trm,
                n_heads=blob["n_heads"], M=blob.get("M", 4), top_trm=top_trm)
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
            sv = getattr(self, "_slots_v", None)
            gate = getattr(self, "slot_gate", None)
            # The gate reads the SAME hidden states the adapter is about to modify, so the decision
            # is made on this layer's own content at this position -- not on a sequence-level or
            # task-level summary.
            mask = gate(h.float(), self._slots.float()) if gate is not None else None
            if mask is not None:
                self._last_mask = mask.detach()
            h2 = self.adapters[idx](h.float(), self._slots.float(),
                                    None if sv is None else sv.float(),
                                    slot_mask=mask).to(h.dtype)
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
# TinyGNN + GraphSlotEncoder — active subgraph → fixed K slots for LM injection
# ================================================================================================
class TinyGNN(nn.Module):
    """One message-passing layer over typed edges. ~30K params."""

    def __init__(self, d_in: int = 384, d_hidden: int = 64, n_rel: int = 12):
        super().__init__()
        self.rel_emb = nn.Embedding(n_rel, 16)
        self.W_msg = nn.Linear(d_in + 16, d_hidden)
        self.W_self = nn.Linear(d_in, d_hidden)
        self.W_out = nn.Linear(d_hidden, d_in)
        self.norm = nn.LayerNorm(d_in)

    def forward(self, x, edge_index, edge_type, edge_strength):
        N = x.shape[0]
        if edge_index is None or edge_index.numel() == 0:
            return x
        src, dst = edge_index[0], edge_index[1]
        type_emb = self.rel_emb(edge_type)
        msg_in = torch.cat([x[src], type_emb], dim=-1)
        msg = self.W_msg(msg_in) * edge_strength.unsqueeze(-1)
        aggr = torch.zeros(N, msg.shape[-1], device=x.device, dtype=msg.dtype)
        aggr = aggr.index_add_(0, dst, msg)
        h = self.W_self(x) + aggr
        h = torch.relu(h)
        out = self.W_out(h)
        return self.norm(x + out)


class GraphSlotEncoder(nn.Module):
    """Encodes an active session subgraph into K fixed LM-space slots.
    Each slot is an attention-pool over graph nodes (interpretable, not opaque)."""

    def __init__(self, d_node: int = 384, d_lm: int = 1536, K: int = 8):
        super().__init__()
        self.K = K
        self.d_node = d_node
        self.slot_queries = nn.Parameter(torch.randn(K, d_node) * 0.02)
        self.gnn = TinyGNN(d_in=d_node, d_hidden=128)
        self.proj = nn.Linear(d_node, d_lm)
        self.slot_gate = nn.Parameter(torch.zeros(K))

    def forward(self, node_embs, edge_index, edge_type, edge_strength,
                recall_weights: torch.Tensor | None = None):
        h = self.gnn(node_embs, edge_index, edge_type, edge_strength)
        if recall_weights is not None:
            h = h * recall_weights.unsqueeze(-1)
        scores = torch.matmul(self.slot_queries, h.t()) / math.sqrt(self.d_node)
        attn = torch.softmax(scores, dim=-1)
        slots = torch.matmul(attn, h)
        slots = self.proj(slots)
        return slots * torch.sigmoid(self.slot_gate).unsqueeze(-1)


def _train_wm_graph_slots(wb, slot_encoder, train_examples, dev, epochs=40, lr=3e-4):
    """Train GraphSlotEncoder + GatedCrossAttn adapters jointly on graph slots.

    train_examples: list of (node_embs [N,d_node], edge_index, edge_type, edge_strength,
                             recall_weights [N], target_text, prompt_text).
    The encoder is NOT frozen: slot_queries/GNN/gate are learned together with the
    adapters (design doc 3.1 — each slot learns what to attend to in the graph).
    """

    import torch.nn.functional as _F
    torch.manual_seed(0)
    if len(train_examples) < 2:
        return None, None

    layers = [len(wb.layers) - 3, len(wb.layers) - 2, len(wb.layers) - 1]
    adapters = [GatedCrossAttn(wb.d_model).to(dev) for _ in layers]
    params = [p for a in adapters for p in a.parameters() if p.requires_grad]
    params += [p for p in slot_encoder.parameters() if p.requires_grad]
    opt = torch.optim.Adam(params, lr=lr)

    _slot_container = [None]

    def _make_hook(adapter):
        def _hook(mod, inp, out):
            h = out[0] if isinstance(out, tuple) else out
            if _slot_container[0] is None:
                return None
            h2 = adapter(h.float(), _slot_container[0].float()).to(h.dtype)
            if isinstance(out, tuple):
                return (h2,) + tuple(out[1:])
            return h2
        return _hook

    handles = [wb.layers[L].register_forward_hook(_make_hook(a))
               for L, a in zip(layers, adapters)]

    for ep in range(epochs):
        total = 0.0
        for node_embs, edge_index, edge_type, edge_strength, recall_weights, target, prompt in train_examples:
            slots = slot_encoder(node_embs, edge_index, edge_type, edge_strength, recall_weights)
            _slot_container[0] = slots
            tid = wb.tok(target, return_tensors="pt").input_ids.to(dev)[0]
            if tid.numel() == 0 or tid[0] == wb.tok.eos_token_id:
                continue
            pids = wb.tok(prompt, return_tensors="pt").input_ids.to(dev)
            plen = pids.shape[1]
            inp = torch.cat([pids, tid[:-1].unsqueeze(0)], dim=1)
            lg = wb.model(inp).logits
            loss = _F.cross_entropy(lg[:, plen - 1:].reshape(-1, lg.shape[-1]), tid)
            opt.zero_grad(); loss.backward(); opt.step()
            total += float(loss.detach())
        if ep % 10 == 0:
            print(f"      [graph-slot] ep {ep} loss={total / len(train_examples):.4f}")

    for h in handles:
        h.remove()
    final = total / len(train_examples)
    print(f"      [graph-slot] final loss={final:.4f}  ({len(train_examples)} examples, {epochs} epochs)  "
          f"gates=[{', '.join(f'{float(a.g.detach()):.3f}' for a in adapters)}]")
    return adapters, layers


def _ds_targets_for_task(wb, code_expr: str, atoms_needed, T: int, cache: dict) -> "torch.Tensor | None":
    """Deep-supervision targets: [T, d_lm], the PARTIAL RESULT the recursion should hold at each step.

    Targets are the PROJECTED INTERMEDIATE EXPRESSION, never the retrieved atom nodes. Supervising against
    atom-description embeddings is a real mistake this project already made: it teaches the trm to echo
    what retrieval just handed it, which is information it is already given, instead of teaching it to
    COMPOSE those atoms into a result. The composition is the entire job -- for
    `outer(inner(n))` the trm must learn that after the inner step it holds `inner(n)`, a value that
    appears nowhere in its inputs.

    Schedule over the T recursion steps: the first half target the inner partial expression `inner(n)`,
    the second half target the full composition `outer(inner(n))` -- a real curriculum through the
    recursion rather than one flat target repeated T times.

    Returns None when no oracle intermediate can be derived (math-cot, swe-action), in which case the
    caller skips DS for that example instead of inventing a target.
    """
    if not code_expr or len(atoms_needed) < 2:
        return None
    inner = atoms_needed[0]
    inner_expr = f"{inner}(n)"
    if inner_expr not in code_expr:                 # not the 2-atom composition shape -> no oracle
        return None
    key = (inner_expr, code_expr, T)
    if key in cache:
        return cache[key]
    half = max(1, T // 2)
    texts = [inner_expr] * half + [code_expr] * (T - half)
    tgt = native_text_embedding_batch(wb, texts)    # [T, d_lm] -- projected results, not atom nodes
    cache[key] = tgt
    return tgt


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
# Hierarchical working memory: periodic re-grounding during generation. The EXISTING WMReasoner.refine()
# runs ONCE per task, before generate() starts -- GatedCrossAttn then attends to that SAME fixed content
# for the entire generation. That's a persistent hint, not memory that evolves with what's actually been
# written. generate_with_reground re-invokes refine()/hierarchical_refine() every `chunk_tokens`, re-
# grounding in the real partial generation so far each time -- the working memory now tracks what's
# actually been produced, not just the original task description.
# ================================================================================================
def evict_cache(cache, keep_last: int, keep_first: int = 0) -> None:
    """Sliding-window eviction with optional ATTENTION SINKS: keep the first `keep_first` tokens AND the
    last `keep_last` tokens of each layer's KV, dropping the middle, in place.

    Real primitive on transformers' DynamicCache (confirmed on 5.9.0): each layer stores raw
    `.keys`/`.values` tensors ([B, heads, seq, head_dim]) directly -- `.crop()` keeps the FIRST n tokens
    (built for generation rollback, wrong direction for a sliding window), so this slices explicitly
    instead. `get_seq_length()` is derived live from tensor shape, so `model.generate()`'s internal
    cache-length bookkeeping picks up the shrunk cache automatically on the next call -- no separate
    counter to keep in sync.

    keep_first > 0 exists because pure sliding-window eviction (keep_first=0) is the known-broken variant,
    and this was found by real measurement, not assumed: the OLDEST tokens are the PROMPT -- the task
    description itself. On this codebase's synthetic composition domain, a real measurement showed prompts
    of 23-25 tokens against completions of only 8-9, so any window small enough to actually evict was
    necessarily deleting the question, not stale reasoning. The same holds in long CoT (a sliding window
    always drops the original problem statement first). This is exactly the failure StreamingLLM's
    attention sinks address -- keeping a handful of initial tokens recovers most of the quality that naive
    windowing destroys. keep_first=0 (default) preserves the original pure-window behavior byte-for-byte.

    NOTE for callers: with keep_first > 0 the surviving tokens are NO LONGER a contiguous suffix -- their
    true absolute positions are [0..keep_first-1] + [much later..], a gap. Any caller passing explicit
    position_ids (which generate_with_reground must, for RoPE correctness) has to track real per-token
    positions rather than a single scalar offset. generate_with_reground does exactly that."""
    for layer in cache.layers:
        seq = layer.keys.shape[-2]
        if seq <= keep_first + keep_last:
            continue
        if keep_first > 0:
            layer.keys = torch.cat([layer.keys[..., :keep_first, :], layer.keys[..., -keep_last:, :]], dim=-2)
            layer.values = torch.cat([layer.values[..., :keep_first, :], layer.values[..., -keep_last:, :]], dim=-2)
        else:
            layer.keys = layer.keys[..., -keep_last:, :]
            layer.values = layer.values[..., -keep_last:, :]


def generate_with_reground(wb, R, pids, task_emb, atom_embs, chunk_tokens: int = 16,
                           max_new_tokens: int = 128, top_every: int = 4,
                           use_kv_cache: bool = False, evict_window: int | None = None,
                           trigger_patterns: list | None = None,
                           instability_trigger: float | None = None,
                           sink_tokens: int = 0, reground_bottom: bool = False,
                           top_no_graph: bool = False, top_memory_max: int = 16,
                           evict_to_memory: bool = False, merged: bool = False,
                           session=None):
    """Generate, re-grounding WMReasoner's slots every chunk_tokens instead of once up front.

    If R.top_trm is set, also runs the slow/top-level TRM every `top_every` CHUNKS (not every chunk) --
    the actual cadence-based fast/slow split (see hierarchical_refine's docstring for why this is a cadence,
    not real OS threads: two forward passes sharing one CUDA context under the GIL don't run concurrently
    in any meaningful sense; running top less often is what gives the real timescale separation). Between
    top updates, the bottom trm reuses the last computed top_state (recompute_top=False) -- cheap, and the
    slow/fast split is exactly "top updates less often than bottom," matching the actual ask.

    use_kv_cache=False (default): UNCHANGED behavior -- every chunk calls wb.model.generate() fresh on the
    whole running sequence (no past_key_values), i.e. it recomputes the entire growing prefix from scratch
    each time. Kept as the zero-risk default; every existing caller is unaffected.

    use_kv_cache=True: threads a real KV cache between chunks (PrefixSession's proven pattern, see
    prefix_session.py) instead of recomputing the prefix every chunk -- for greedy decoding this must
    produce byte-identical output to the use_kv_cache=False path (regression-tested offline, not assumed).

    evict_window (requires use_kv_cache=True): position-compensated sliding-window eviction. Two REAL bugs
    were found and fixed here via direct testing on Qwen3-4B-Instruct-2507 (not assumed correct):
      1. RoPE bakes rotation into cached keys at the position they were computed, permanently -- naive
         eviction (slice the cache, let generate() derive position_ids from the new shorter cache length)
         gave new queries rotation for positions 0..window while surviving keys kept rotation for their
         true original (larger) positions -- confirmed via real garbled output ("is is is... task task
         task... 1111111...2222222222222"). Fixed by tracking each surviving/new token's TRUE absolute
         position ourselves (`true_pos_offset`) and passing it explicitly as `position_ids=` to generate()
         -- confirmed (by reading transformers' generation/utils.py) that an explicit position_ids is
         forwarded untouched on the first step and correctly incremented from its own last value on
         subsequent steps, not re-derived from cache length.
      2. Separately, found via direct instrumentation (cache length jumped 48->103 instead of the expected
         48->56): when `cur_ids.shape[-1]` exactly equals `past.get_seq_length()`, generate()'s internal
         "how many tokens are new beyond the cache" count comes out to 0, and `arr[:, -0:]` is a Python
         slicing quirk (`-0 == 0`) meaning "the whole array," not "nothing" -- so it silently re-prefills
         the entire already-cached cur_ids, duplicating every surviving token's KV entry. Fixed by evicting
         the cache to `evict_window - 1` while keeping cur_ids at `evict_window` tokens, restoring the
         same "cur_ids is one token ahead of the cache" invariant that was already incidentally present in
         every normal (non-eviction) chunk -- which is why only eviction chunks hit this.
    Verified on real Qwen3-4B: no more garbled/repeated-digit degeneration; cache length stays bounded
    (measured, not assumed). A separate, milder concern remains: the model can fall into ordinary
    greedy-decoding content repetition (e.g. re-stating "we check if a number is prime" several times) on
    long evicted generations -- ordinary LLM greedy-decoding behavior, not the garbled-token failure mode
    above, and not yet distinguished from "eviction lost something the model needed" vs "greedy decoding
    would have looped here anyway." The `run_real` held-out harness (held/ablated/reground) is the next
    real test for that distinction, not yet wired for evict_window.

    trigger_patterns: optional list of substrings (e.g. ["\n", ". ", "Therefore", "Step"]) checked against
    each chunk's newly-generated text. If any appear, top recomputes on the VERY NEXT chunk regardless of
    `top_every`'s cadence -- an event (a real reasoning/sentence boundary just happened) can trigger a
    recompute early; the cadence still fires as the fallback, so top is never starved if no trigger ever
    appears. None (default) = pure cadence, unchanged behavior for every existing caller.

    instability_trigger: a real, learned alternative to trigger_patterns' hand-picked strings -- the BOTTOM
    trm triggers the top trm itself, off its OWN computed state, not off surface text. Each chunk's bottom
    refine() already tracks per-cycle deltas (track_deltas=True, the same mechanism the held-out eval loop
    already uses to print `instab`); trajectory_instability(deltas) is the late/early ratio of
    ||y_{t+1}-y_t||, i.e. "is the bottom trm still churning on this chunk."

    RELATIVE, not absolute -- this is a fix for a real bug found in a real run, not a preference.
    trajectory_instability's own docstring says ">1 = still churning", so an absolute threshold of 1.0 was
    recommended first; a real 20-epoch run on Qwen3-4B then showed measured instability of 0.062 at epoch 0
    decaying to 0.001 by epoch 10 -- THREE orders of magnitude below 1.0, so the trigger never fired once
    and the whole run silently tested nothing. Worse, no fixed value can work: the signal itself decays
    ~30x as training converges, so any constant either fires on every chunk early or never fires late.
    This value is therefore a MULTIPLIER against the running mean of previous chunks' instability within
    the current generation (e.g. 1.5 = "50% above this generation's own recent average"). Scale-free, so it
    survives the decay. The first chunk has no baseline yet and never fires on this path (the cadence
    covers it). None (default) = off, unchanged behavior. Domain-agnostic, unlike trigger_patterns.

    reground_bottom: feed the generated-so-far embedding into the BOTTOM trm's task input (added to the
    original task_emb), not just the top trm's context. This fixes a real, confirmed no-op, and it is the
    reason to use this function at all. Without it, `task_emb` stays the ORIGINAL static embedding on every
    chunk and the ONLY path for mid-generation information to reach the LM is
    `top_to_bottom_proj(top_state)` -- which is deliberately zero-initialized. While that projection is at
    or near zero, the bottom trm receives exactly the static task embedding every chunk, produces identical
    slots, and the generation is BYTE-IDENTICAL to the one-shot static path. Confirmed across every real run
    in this codebase's history: `reground` matched `held WM` exactly at every checkpoint (2/2, 10/10, 7/7,
    12/12, 14/14 on one 40-epoch Qwen3-4B run, and the same in all earlier ones) -- not a coincidence, a
    mathematical consequence. The approved design called for re-running refine() on an UPDATED embedding;
    routing the update only to the top trm was the deviation. False (default) preserves the old byte-
    identical behavior for regression safety; run_real's reground arms pass True so the harness measures
    something real.

    sink_tokens (requires evict_window): keep this many tokens from the very START of the sequence, in
    addition to the recent window -- StreamingLLM-style ATTENTION SINKS. Found necessary by real
    measurement, not assumed: a sliding window drops the OLDEST tokens, which are the PROMPT -- the task
    description itself. Measured on this codebase's synthetic composition domain, prompts run 23-25 tokens
    against completions of only 8-9, so any window small enough to actually evict was necessarily deleting
    the question rather than stale reasoning (the same is true of long CoT -- a pure window always drops the
    original problem first). 0 (default) = pure sliding window, the original behavior, which is the
    known-broken variant this parameter exists to fix. Try sink_tokens ~= the real prompt length to keep
    the task statement resident while still bounding total cache growth.

    Returns the full generated text (decoded, all chunks concatenated) -- same string shape callers already
    get from a plain wb.model.generate() + decode."""
    if not use_kv_cache:
        running_ids = pids
        prompt_len = pids.shape[-1]
        top_state = None
        top_resume_state = None
        chunk_idx = 0
        generated_so_far = ""
        event_fired = False
        pending_event = 0
        bottom_state = None
        instab_history: list[float] = []
        top_memory_list: list[torch.Tensor] = []
        while (running_ids.shape[-1] - prompt_len) < max_new_tokens:
            remaining = max_new_tokens - (running_ids.shape[-1] - prompt_len)
            n_new = min(chunk_tokens, remaining)
            recompute_top = (R.top_trm is not None) and ((chunk_idx % top_every == 0) or event_fired)
            top_ctx = None
            if recompute_top and generated_so_far:
                # bottom-to-top flow: top TRM reasons over what's ACTUALLY been generated so far, not just
                # the static original task description -- grounds the slow/meta level in real, current
                # progress. MiniLM space (encode_batch), NOT native_text_embedding -- TRMReasoner.task_proj/
                # atom_proj both expect d_in (MiniLM 384) space, matching task_emb/atom_embs;
                # native_text_embedding is d_lm (LM hidden) space, only used for the actual LM-injection
                # path (GatedCrossAttn), a different tensor entirely. Mixing these raised a real
                # shape-mismatch caught by the offline test.
                top_ctx = torch.as_tensor(encode_batch([generated_so_far])[0], dtype=torch.float32, device=wb.device)
            want_deltas = instability_trigger is not None
            bottom_task_emb = task_emb
            if reground_bottom and generated_so_far:
                prog = torch.as_tensor(encode_batch([generated_so_far])[0], dtype=torch.float32,
                                       device=wb.device)
                bottom_task_emb = task_emb + prog
            if top_no_graph and recompute_top:
                mem_add = top_ctx if top_ctx is not None else task_emb
                top_memory_list.append(mem_add)
                if len(top_memory_list) > top_memory_max:
                    top_memory_list = top_memory_list[-top_memory_max:]
            top_mem = torch.stack(top_memory_list) if top_memory_list else None
            if merged:
                # MERGED: no top trm at all -- the bottom trm carries its own state and attends its own
                # memory alongside the graph atoms, with no zero-init projection in between.
                # SLIDING, not first-N. This used to append only `if len(top_memory_list) <
                # top_memory_max`, so once the bank filled it froze permanently: on any generation longer
                # than top_memory_max chunks the trm kept attending the FIRST 16 progress snapshots and
                # never saw anything recent -- the exact opposite of memory that tracks what has been
                # written. Every sibling memory path here already slid with [-top_memory_max:].
                if generated_so_far:
                    top_memory_list.append(torch.as_tensor(
                        encode_batch([generated_so_far])[0], dtype=torch.float32, device=wb.device))
                    top_memory_list = top_memory_list[-top_memory_max:]
                    top_mem = torch.stack(top_memory_list)
                out_m = R.recurrent_refine(bottom_task_emb, atom_embs, memory=top_mem,
                                           resume_state=bottom_state, track_deltas=want_deltas,
                                           event_signal=pending_event)
                bottom_state = out_m[-1]
                refine_out = (*out_m[:-1], None, None)
            else:
                refine_out = R.hierarchical_refine(
                    bottom_task_emb, atom_embs, top_context_emb=top_ctx, top_state=top_state,
                    top_resume_state=top_resume_state, recompute_top=recompute_top,
                    track_deltas=want_deltas, top_memory=top_mem, top_no_graph=top_no_graph,
                    event_signal=pending_event)
            if want_deltas:
                slots, _states, deltas, _raw, new_top_state, new_top_resume_state = refine_out
            else:
                slots, _states, new_top_state, new_top_resume_state = refine_out
                deltas = None
            top_state = new_top_state
            top_resume_state = new_top_resume_state
            R.set_slots_direct(slots)
            with torch.no_grad():
                out = wb.model.generate(running_ids, max_new_tokens=n_new, do_sample=False,
                                        pad_token_id=wb.tok.eos_token_id)
            running_ids = out
            prev_len = len(generated_so_far)
            generated_so_far = wb.tok.decode(running_ids[0][prompt_len:], skip_special_tokens=True)
            new_text = generated_so_far[prev_len:]
            pattern_fired = bool(trigger_patterns) and any(p in new_text for p in trigger_patterns)
            instability_fired = False
            if deltas is not None and instability_trigger is not None:
                instab_now = R.trajectory_instability(deltas)
                if instab_history:
                    baseline = sum(instab_history) / len(instab_history)
                    instability_fired = instab_now > baseline * instability_trigger
                instab_history.append(instab_now)
            event_fired = pattern_fired or instability_fired
            pending_event = (1 if pattern_fired else 0) | (2 if instability_fired else 0)
            chunk_idx += 1
            if running_ids[0, -1].item() == wb.tok.eos_token_id:
                break
        R.clear()
        return generated_so_far

    # use_kv_cache=True path: thread a real cache between chunks (PrefixSession's proven pattern) instead
    # of recomputing the whole prefix every chunk. generated_so_far is decoded from an independently
    # accumulated list of every real generated token id (all_new_ids), NOT by slicing cur_ids against a
    # fixed prompt_len and NOT by decoding+concatenating each chunk's tokens in isolation -- two real bugs
    # ruled out this way: (1) evict_window trims the FRONT of cur_ids, so a fixed prompt_len offset would
    # silently corrupt the decode once that happens; (2) decoding a chunk's raw token slice by itself can
    # split a multi-token Unicode character across the chunk boundary (confirmed on real Qwen output: a
    # '√' got mangled into a replacement char this way) -- decoding the FULL accumulated id list each
    # time, exactly like the use_kv_cache=False path already does, avoids ever decoding a partial fragment.
    #
    # true_positions: the TRUE absolute position of EVERY token currently in cur_ids, tracked explicitly as
    # a tensor rather than a single scalar offset. Eviction physically removes cached tokens -- it does NOT
    # re-rotate the survivors' cached keys (RoPE bakes rotation in at compute time, permanently). The fix is
    # not touching the cache at all: pass explicit `position_ids` to generate() reflecting each surviving/new
    # token's TRUE original position (confirmed by reading transformers' actual generation/utils.py:
    # prepare_inputs_for_generation forwards an explicit position_ids through untouched, and
    # _update_model_kwargs_for_generation increments every subsequent step from `position_ids[..., -1] + 1`,
    # NOT from the cache's current length) -- RoPE's attention score depends only on the RELATIVE offset
    # between a query and key's true positions, so as long as both are labeled correctly there is no
    # gap/mismatch, even though the cache itself is shorter than the true token count.
    #
    # A tensor, not the scalar offset this used to be, specifically because sink_tokens > 0 makes the
    # survivors NON-CONTIGUOUS ([0..sink-1] then a gap then the recent window). A scalar offset literally
    # cannot express that; sliced in lockstep with cur_ids, this tensor can.
    true_positions = torch.arange(pids.shape[-1], device=wb.device)
    cur_ids = pids
    past = None
    top_state = None
    top_resume_state = None
    chunk_idx = 0
    all_new_ids: list[int] = []
    generated_so_far = ""
    event_fired = False
    pending_event = 0
    bottom_state = None
    instab_history: list[float] = []
    top_memory_list: list[torch.Tensor] = []
    while len(all_new_ids) < max_new_tokens:
        n_new = min(chunk_tokens, max_new_tokens - len(all_new_ids))
        recompute_top = (R.top_trm is not None) and ((chunk_idx % top_every == 0) or event_fired)
        top_ctx = None
        if recompute_top and generated_so_far:
            top_ctx = torch.as_tensor(encode_batch([generated_so_far])[0], dtype=torch.float32, device=wb.device)
        want_deltas = instability_trigger is not None
        bottom_task_emb = task_emb
        if reground_bottom and generated_so_far:
            prog = torch.as_tensor(encode_batch([generated_so_far])[0], dtype=torch.float32,
                                   device=wb.device)
            bottom_task_emb = task_emb + prog
        if top_no_graph and recompute_top:
            mem_add = top_ctx if top_ctx is not None else task_emb
            top_memory_list.append(mem_add)
            if len(top_memory_list) > top_memory_max:
                top_memory_list = top_memory_list[-top_memory_max:]
        top_mem = torch.stack(top_memory_list) if top_memory_list else None
        if merged:
            # SLIDING window, same fix as the no-cache path above: the old `len(...) < top_memory_max`
            # guard froze the memory bank at the first N chunks instead of tracking recent progress.
            if generated_so_far:
                top_memory_list.append(torch.as_tensor(
                    encode_batch([generated_so_far])[0], dtype=torch.float32, device=wb.device))
                top_memory_list = top_memory_list[-top_memory_max:]
                top_mem = torch.stack(top_memory_list)
            out_m = R.recurrent_refine(bottom_task_emb, atom_embs, memory=top_mem,
                                       resume_state=bottom_state, track_deltas=want_deltas,
                                       event_signal=pending_event)
            bottom_state = out_m[-1]
            refine_out = (*out_m[:-1], None, None)
        else:
            refine_out = R.hierarchical_refine(
                bottom_task_emb, atom_embs, top_context_emb=top_ctx, top_state=top_state,
                top_resume_state=top_resume_state, recompute_top=recompute_top,
                track_deltas=want_deltas, top_memory=top_mem, top_no_graph=top_no_graph,
                event_signal=pending_event)
        if want_deltas:
            slots, _states, deltas, _raw, new_top_state, new_top_resume_state = refine_out
        else:
            slots, _states, new_top_state, new_top_resume_state = refine_out
            deltas = None
        top_state = new_top_state
        top_resume_state = new_top_resume_state
        R.set_slots_direct(slots)
        with torch.no_grad():
            attn = torch.ones_like(cur_ids)
            position_ids = true_positions.unsqueeze(0)
            out = wb.model.generate(input_ids=cur_ids, attention_mask=attn, past_key_values=past,
                                    position_ids=position_ids,
                                    max_new_tokens=n_new, do_sample=False, pad_token_id=wb.tok.eos_token_id,
                                    use_cache=True, return_dict_in_generate=True)
        seq = out.sequences
        n_added = seq.shape[-1] - cur_ids.shape[-1]
        new_ids_this_chunk = seq[0, cur_ids.shape[-1]:].tolist()
        all_new_ids.extend(new_ids_this_chunk)
        generated_so_far = wb.tok.decode(all_new_ids, skip_special_tokens=True)
        new_text = wb.tok.decode(new_ids_this_chunk, skip_special_tokens=True)
        pattern_fired = bool(trigger_patterns) and any(p in new_text for p in trigger_patterns)
        instability_fired = False
        if deltas is not None and instability_trigger is not None:
            instab_now = R.trajectory_instability(deltas)
            if instab_history:
                baseline = sum(instab_history) / len(instab_history)
                instability_fired = instab_now > baseline * instability_trigger
            instab_history.append(instab_now)
        event_fired = pattern_fired or instability_fired
        pending_event = (1 if pattern_fired else 0) | (2 if instability_fired else 0)
        cur_ids = seq
        # Newly generated tokens continue from the last TRUE position, matching how generate() itself
        # incremented them internally for this chunk (position_ids[..., -1] + 1, per transformers' source).
        if n_added > 0:
            nxt = true_positions[-1].item() + 1
            true_positions = torch.cat([
                true_positions,
                torch.arange(nxt, nxt + n_added, device=wb.device),
            ])
        past = getattr(out, "past_key_values", None)
        if evict_window is not None and past is not None and past.get_seq_length() > evict_window:
            # Evict the CACHE to evict_window-1, but keep cur_ids at evict_window (one token longer than
            # the cache). Real bug found by direct instrumentation, not guessed: when cur_ids.shape[-1]
            # exactly equals past.get_seq_length(), generate()'s internal "how many tokens are new beyond
            # the cache" calculation comes out to 0, and `arr[:, -0:]` is a Python slicing quirk that means
            # "the whole array" (-0 == 0), not "nothing" -- so it silently RE-PREFILLS the entire (already
            # cached) cur_ids on the next call, duplicating every surviving token's KV entry (confirmed:
            # cache jumped 48->103 instead of the expected 48->56). Keeping cur_ids one token ahead of the
            # cache (the same invariant that was already present, incidentally, in every non-eviction
            # chunk -- which is WHY only eviction chunks broke) makes the "how many new" count correctly
            # come out to 1, not 0.
            #
            # cur_ids/true_positions are sliced in EXACT lockstep with the cache (same keep_first sinks +
            # same recent window), so every surviving token keeps its real absolute position label even
            # though the kept set is non-contiguous when sinks are on.
            keep_last_cache = evict_window - 1 - sink_tokens
            # EVICTION FEEDS THE TOP TRM'S MEMORY instead of the dropped span being lost outright. The
            # tokens about to leave the KV cache are decoded, embedded once, and appended to the top trm's
            # own memory bank -- so on a long-horizon task the top level accumulates a compressed record of
            # everything the cache had to discard, and can still surface it to the bottom trm (and thus the
            # LM) via top_to_bottom_proj. Without this, eviction is pure forgetting: the sinks keep the
            # prompt and the window keeps recent tokens, but the entire middle of a long generation is gone.
            #
            # This also sidesteps a real truncation bug in the other memory path: top_ctx is
            # encode_batch([generated_so_far]) and encode_batch truncates at max_length=128 tokens, so on
            # exactly the long generations this is meant for, that context is the FIRST ~128 tokens
            # re-embedded over and over while later content is silently dropped. An evicted span is bounded
            # and small, so embedding it is faithful, and successive spans tile the real history.
            if evict_to_memory or session is not None:
                n_now = cur_ids.shape[-1]
                drop_end = n_now - (evict_window - sink_tokens)
                dropped = cur_ids[0, sink_tokens:drop_end] if sink_tokens > 0 else cur_ids[0, :n_now - evict_window]
                if dropped.numel() > 0:
                    dropped_text = wb.tok.decode(dropped.tolist(), skip_special_tokens=True).strip()
                    if dropped_text:
                        # SESSION GRAPH, when one is supplied, because the flat bank below is still a form
                        # of forgetting: it is capped at top_memory_max, so span N silently deletes span
                        # N-16, and its only retrieval key is recency -- an early span is unreachable no
                        # matter how relevant it later becomes. The graph keeps every span, addressable by
                        # meaning, and keeps the VERBATIM text rather than a 384-d pooled vector, so a
                        # recalled digit is copied instead of reconstructed. See membrane_session.py.
                        if session is not None:
                            session.write(dropped_text)
                        if evict_to_memory:
                            top_memory_list.append(torch.as_tensor(
                                encode_batch([dropped_text])[0], dtype=torch.float32, device=wb.device))
                            if len(top_memory_list) > top_memory_max:
                                top_memory_list = top_memory_list[-top_memory_max:]
            evict_cache(past, keep_last_cache, keep_first=sink_tokens)
            if sink_tokens > 0:
                cur_ids = torch.cat([cur_ids[:, :sink_tokens], cur_ids[:, -(evict_window - sink_tokens):]], dim=-1)
                true_positions = torch.cat([
                    true_positions[:sink_tokens],
                    true_positions[-(evict_window - sink_tokens):],
                ])
            else:
                cur_ids = cur_ids[:, -evict_window:]
                true_positions = true_positions[-evict_window:]
        chunk_idx += 1
        if seq[0, -1].item() == wb.tok.eos_token_id:
            break
    R.clear()
    return generated_so_far


def explain_what_happened(wb, g, session, query: str, k: int = 3) -> dict:
    """Answer a real follow-up question grounded in memory -- never a silent/unverified guess. Two tiers,
    tried in order:
      1. SHORT-TERM: session.update(query) re-seeds SessionFocus's spreading activation (membrane_session.py)
         from the current query over the persistent graph; if that activates anything, ground the answer
         in those nodes (this is "what's live in working/session memory right now").
      2. LONG-TERM fallback: if the query doesn't activate anything session-relevant (a genuinely new topic,
         or session is None), fall back to plain g.cosine_rank over the FULL persistent graph.
    Either way, the frozen LM is told to answer ONLY from the retrieved facts (same grounding-prompt
    pattern proven in membrane.py's demo_teach_explain) -- grounded, not hallucinated. Returns which tier
    and which real nodes actually answered it, so this is checkable, not a black box.
    """
    focus = session.update(query) if session is not None else set()
    if focus:
        names = list(focus)[:k]
        tier = "short-term (session focus)"
    else:
        names = g.cosine_rank(query, k=k)
        tier = "long-term (full graph)"
    # Real bug found and fixed here (not assumed): a graph node's description can be up to 4000 chars
    # (_grow_from_cot's own real-OpenThoughts cap) -- k=3 of those, uncapped, overflowed distilgpt2's
    # 1024-token position-embedding table (confirmed: real CUDA "srcIndex < srcSelectDimSize" assertion,
    # generate_plain's own tok() call has no max_length/truncation either). Cap each fact so the fact block
    # stays a small, bounded prompt regardless of how long a real banked node's description is.
    MAX_FACT_CHARS = 300
    facts = [g.get(n).description[:MAX_FACT_CHARS] for n in names if g.get(n)]
    if not facts:
        return {"tier": "none", "nodes": [], "answer": "(nothing relevant found in memory)"}
    fact_block = "\n".join(f"- {f}" for f in facts)
    prompt = (f"Use ONLY the following facts from memory to answer. Be concise and do not add anything "
              f"not supported by these facts.\nFacts:\n{fact_block}\nQuestion: {query}\nAnswer:")
    answer = wb.generate_plain(prompt, max_new=80).strip()
    return {"tier": tier, "nodes": names, "answer": answer}


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
    return tr, te, te_abl, R.adapters[0].g.detach().item(), last


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
    their implementations are incorrect, so using them in composition would always fail verify().

    Also excludes atoms that aren't a genuine int -> int function: _compose_tasks_from_graph below does an
    unconditional all-pairs cross product, composing EVERY pair as outer(inner(n)) -- so an atom's output
    must be a valid int input for whatever OTHER atom it gets composed with, not just able to accept one
    itself. _dynamic_oracle's eval() (unlike membrane.verify(), which is exception-safe) is NOT wrapped in
    try/except, so either failure mode crashes the whole run. Two real failure modes caught by this, not
    just one:
      (1) doesn't run on an int at all (e.g. _grow_skills_from_corpus's nucleotide_freq(dna): dna.upper())
      (2) runs fine alone but returns a NON-int (e.g. celsius_to_fahrenheit(c): c*9.0/5.0+32.0 -- returns a
          float; composing reverse_digits(celsius_to_fahrenheit(n)) then does int('4.73') and crashes --
          the exact crash this line was added to fix; a single fn(3)-no-exception check missed it entirely
          since celsius_to_fahrenheit raises nothing on its own, it just hands the NEXT atom a bad type).
    Real, cheap execution check across a few sample ints (not metadata/type-hints) -- protects against a bad
    atom from ANY source, not just one growth path, including whatever's already in a persisted graph file."""
    from v5.runtime.membrane import _closure
    descs, codes = {}, {}
    for name, a in g.atoms.items():
        if not a.code or a.kind == "trap":
            continue
        try:
            ns: dict = {}
            exec(compile(_closure(g, [name]), "<int-domain-check>", "exec"), ns)
            fn = ns[name]
            if not all(isinstance(fn(x), int) for x in (2, 3, 5, 7)):   # bool counts (is_prime etc.) -- IS an int subtype
                continue
        except Exception:
            continue
        descs[name] = a.description
        codes[name] = a.code
    return descs, codes


def _grow_from_cot(g, n: int, domains: str = "math,code,science,puzzle", keywords: str = "",
                   min_reasoning_chars: int = 200, docs: list | None = None) -> dict:
    """Real graph growth from open data: stream N real OpenThoughts-114k CoT traces (v5.graph_grower.
    fetch_cot -- HF-streamed, no full-dataset download) and bank each through membrane's OWN learn_any --
    the same write-time graph editor demo()/interactive_trace() already use (dedup via cosine >=0.90,
    self-organizing 'related' edges below that). Plain text with no code/oracle -> concept nodes (Tier C:
    trusted-source text, no independent recompute) -- separate from the code atoms composition trains on
    below; this step's job is only to make the LONG-TERM graph itself grow from real external data, honestly
    (some fraction will dedup-merge into existing nodes rather than add new ones -- reported, not hidden).

    RESIDUAL CRASH RISK when called from --run, stated plainly (found while validating the KV-eviction A/B,
    a real pre-existing bug, not introduced this session): HF `datasets` streaming's first real fetch in a
    process segfaults if torch was already imported/active earlier in that same process. Materializing docs
    before constructing TRMRetriever (below) is NOT sufficient by itself here, because `trm_wm.py` imports
    torch at MODULE level -- torch is already loaded the instant this file is imported, before `run_real`'s
    body (let alone this function) ever executes. Confirmed directly: --grow-cot via `--run` still
    segfaults on this machine even with this reordering. The only real fix is fetching in a genuinely
    separate process. Pass pre-fetched `docs` (e.g. via v5.graph_grower.fetch_cot.stream_openthoughts or its
    saved jsonl, produced by a torch-free process) to skip the internal live-stream entirely and avoid the
    risk. Without `docs`, this falls back to live-streaming -- fine when called before torch is touched
    anywhere in the process, NOT safe from inside a real --run invocation on this environment."""
    ot_domains = [d.strip() for d in domains.split(",") if d.strip()]
    kw = [k.strip() for k in keywords.split(",") if k.strip()] or None
    if docs is None:
        from v5.graph_grower.fetch_cot import stream_openthoughts
        docs = list(stream_openthoughts(ot_domains=ot_domains, keywords=kw, limit=n,
                                        min_reasoning_chars=min_reasoning_chars))
    from v5.runtime.membrane import learn_any, TRMRetriever
    retr = TRMRetriever(g)
    added = merged = seen = 0
    for doc in docs:
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


def _grow_from_swe_traces(g, n: int, config: str = "openhands", split: str = "minimax_m25") -> dict:
    """Real graph growth from nvidia/Open-SWE-Traces, mirroring _grow_from_cot's growth logic (stream real
    docs -> bank each through membrane's own learn_any, same dedup/self-organize rules) -- gives a real
    graph concept nodes to retrieve against for _hindsight_examples_from_swe_traces below, self-contained
    (doesn't require an existing grown graph).

    Docs are materialized into a list BEFORE importing membrane/constructing TRMRetriever -- a real,
    confirmed environment fragility, not a style choice: on this machine, HF `datasets` streaming's first
    real fetch segfaults if torch/membrane (TRMRetriever) was already imported/constructed earlier in the
    same process (confirmed by direct reproduction: crashes torch-first, works datasets-first, same crash
    either way otherwise). Same class of native-library conflict as the sentence_transformers segfault
    embedder.py already documents -- not something introduced here, just a second real instance of it."""
    from v5.graph_grower.fetch_swe_traces import stream_swe_traces
    docs = list(stream_swe_traces(config=config, split=split, resolved_only=True, limit=n))
    from v5.runtime.membrane import learn_any, TRMRetriever
    retr = TRMRetriever(g)
    added = merged = seen = 0
    for doc in docs:
        seen += 1
        res = learn_any(g, retr, doc["text"][:4000])
        if res["status"] == "banked-fact":
            added += 1
        elif res["status"] == "merged-fact":
            merged += 1
    return {"seen": seen, "added": added, "merged": merged}


def _grow_swe_step_concepts(g, n_trajectories: int = 60, min_step_chars: int = 30,
                            config: str = "openhands", split: str = "minimax_m25",
                            use_worlds: bool = True, world_join: float = 0.55,
                            world_max: int = 256) -> dict:
    """Real graph growth for _hindsight_examples_from_swe_traces specifically -- REPLACES
    _grow_from_swe_traces for that purpose, do not use the whole-trajectory version for hindsight labeling.

    Real bug found and fixed here (not assumed, confirmed via direct inspection of a real grown graph):
    _grow_from_swe_traces banks each ENTIRE trajectory (problem + all its steps, flattened, up to 4000
    chars) as ONE concept node. That makes the "graph" a pile of ~150 essentially-unrelated OTHER GitHub
    issues' full text -- confirmed the actual failure mode directly: generic step text like "let's check
    the file structure" was cosine-matching against random unrelated issues ('IAM: mock_iam() is keeping
    state...', 'Error in data transfer due to 1006...') purely because SOME whole-issue blob has to be the
    argmax, not because of real topical relevance. The hindsight-labeling premise ("will concept X be
    needed later") only means anything if X is a genuine, reusable fact/action that COULD legitimately
    recur (e.g. "how to view a file's structure", "how to run pytest") -- not another repo's entire
    unrelated issue description.

    This banks each STEP's real reasoning text as its OWN concept node instead -- real dedup (cosine>=0.90
    merge, already in add_or_merge) naturally consolidates recurring generic actions ("viewing file
    structure" showing up across many different trajectories) into shared, genuinely comparable nodes,
    instead of one node per trajectory.

    Performance, not just correctness: a real trajectory set (240 trajectories, ~55 steps each) means
    ~13,000 candidate step texts. Calling learn_any/add_or_merge one text at a time -- as the first version
    of this function did -- means ~13,000 separate encode_batch([text]) forward passes (batch size 1, the
    slow way) PLUS a full graph-matrix rebuild after every single insert, with zero progress visibility in
    between. Fixed: exact-string dedup first (real trajectories repeat the same short actions verbatim
    often -- "Let me look at the file structure." recurs across many different repos), then ONE batched
    encode_batch() call over whatever's left, then insert with the embedding already attached (add_or_merge
    only computes encode_batch itself when atom.emb is None) -- skips learn_any's per-call classification
    overhead too, safe here since every step text is already known to be a plain-text concept, no code/
    oracle involved. Heartbeat print every 500 unique texts processed so this is never silently opaque."""
    from v5.graph_grower.fetch_swe_traces import stream_swe_trajectories
    trajectories = list(stream_swe_trajectories(config=config, split=split, resolved_only=True,
                                                limit=n_trajectories))
    from v5.runtime.membrane import Atom
    raw_texts = []
    for traj in trajectories:
        for s in traj["steps"]:
            text = (s.get("reasoning") or "").strip()
            if len(text) >= min_step_chars:
                raw_texts.append(text)
    seen = len(raw_texts)
    unique_texts = list(dict.fromkeys(raw_texts))   # exact-dup removal, order-preserving
    print(f"    _grow_swe_step_concepts: {seen} step texts, {len(unique_texts)} exact-unique -- "
          f"batch-embedding + inserting...", flush=True)
    # encode_batch has NO internal chunking (confirmed by reading embedder.py) -- it runs the WHOLE list
    # through MiniLM as one forward pass. Passing all ~13k texts at once tried to allocate >10GB of RAM and
    # OOM'd, a real bug in this function's first version, not a guess. Chunk into reasonable sub-batches.
    EMBED_CHUNK = 128
    embs = []
    for c0 in range(0, len(unique_texts), EMBED_CHUNK):
        embs.append(encode_batch(unique_texts[c0:c0 + EMBED_CHUNK]))
        if (c0 // EMBED_CHUNK) % 20 == 0:
            print(f"      ...embedded {min(c0 + EMBED_CHUNK, len(unique_texts))}/{len(unique_texts)}",
                  flush=True)
    embs = np.concatenate(embs, axis=0) if embs else np.zeros((0, EMBED_DIM), dtype=np.float32)
    # NESTED WORLDS: route each step into a cluster instead of comparing it against the whole graph.
    # Measured on 6,349 real step texts: 12.9s -> 2.7s to build, 79,174 -> 36,112 edges, 788 worlds of
    # avg size 7.3, and retrieval essentially unchanged while scanning 7.3x fewer candidates per query.
    # The complexity itself changes (O(N^2) -> ~O(N*sqrt(N))), which is what actually made a real 200-
    # trajectory prep finish instead of being killed at 20 minutes.
    if use_worlds:
        g.enable_worlds(join_threshold=world_join, max_world=world_max)

    added = merged = 0
    text2node: dict[str, str] = {}
    for i, (text, emb) in enumerate(zip(unique_texts, embs)):
        _atom = Atom(name=f"swe_step_{i}", code="", description=text, kind="concept", emb=emb)
        nm, action = (g.add_or_merge_world(_atom, link_lo=0.70) if use_worlds
                      else g.add_or_merge(_atom))
        text2node[text] = nm           # add_or_merge may MERGE, so keep the surviving node's real name
        if action == "added":
            added += 1
        else:
            merged += 1
        if (i + 1) % 500 == 0:
            print(f"      ...{i + 1}/{len(unique_texts)} processed, graph now {len(g)} nodes", flush=True)

    # REAL, TYPED EDGES -- 'follows' means "this step actually came after that one in a real resolved
    # trajectory". That is a recorded fact from the data, not a similarity guess, and it is the only kind
    # of edge here that carries genuine meaning: the similarity edges add_or_merge creates say nothing
    # more than "these two texts look alike", and a direct measurement found 806,604 of them changed
    # retrieval by +0.0 points (at link_lo=0.50 on this homogeneous corpus they form a near-clique and
    # spreading activation reached 89% of the graph, so the boost was uniform and reordered nothing).
    # Self-loops are skipped: dedup legitimately merges a repeated action onto itself across steps.
    follows = bridges = 0
    for traj in trajectories:
        chain = [text2node[t] for s in traj["steps"]
                 if (t := (s.get("reasoning") or "").strip()) and t in text2node]
        for a, b in zip(chain, chain[1:]):
            if a != b:
                before = len(g.edges)
                g.link(a, b, "follows")
                if len(g.edges) > before:
                    follows += 1
                    # A 'follows' edge whose endpoints sit in DIFFERENT worlds is a long-range bridge --
                    # the thing that makes this a genuine small-world graph rather than a pile of isolated
                    # clusters. Worlds alone measured 100% intra-world edges, i.e. disconnected islands;
                    # a small-world topology needs a few long links to collapse the diameter. These
                    # bridges are recorded facts (a real trajectory moved from one kind of step to
                    # another), not similarity guesses, so they carry real meaning as well as structure.
                    if use_worlds and g.world_of.get(a) != g.world_of.get(b):
                        bridges += 1
    if use_worlds:
        st = g.world_stats()
        print(f"      +{follows} 'follows' edges ({bridges} of them BRIDGE different worlds -- real "
              f"long-range links, recorded succession not similarity)")
        print(f"      worlds: {st['worlds']} clusters, avg size {st['avg_size']:.1f}, max {st['max_size']}, "
              f"{st['singletons']} singletons; intra-world edge fraction {100*st['intra_frac']:.1f}%")
    else:
        print(f"      +{follows} 'follows' edges from real trajectory order (recorded succession)")
    return {"seen": seen, "added": added, "merged": merged, "follows": follows, "bridges": bridges}


def _hindsight_examples_from_swe_traces(g, n_trajectories: int = 30, lookahead_k: int = 10,
                                        min_relevance: float = 0.35,
                                        config: str = "openhands", split: str = "minimax_m25") -> list:
    """Real, verifiable hindsight-supervised examples for FutureNeedScorer: speculative memory needs a real
    target for "will this be needed later," and the real, non-guessed signal is recovered AFTER THE FACT
    from real completed trajectories -- at step T, was some candidate atom/concept the thing a LATER step
    (T, T+lookahead_k] actually turned out to be about? That's real, recoverable ground truth, not a guess,
    same anti-poison shape as record_success/record_failure elsewhere in this codebase.

    Requires g to already have real, FINE-GRAINED concept nodes to retrieve against -- use
    _grow_swe_step_concepts (step-level), NOT _grow_from_swe_traces (whole-trajectory-blob nodes; confirmed
    a real problem for this exact purpose, see _grow_swe_step_concepts's docstring) -- with an empty graph
    there is nothing to predict future need FOR, so trajectories are skipped rather than silently returning
    meaningless labels.

    Returns a list of (task_emb, progress_emb, candidate_emb, label) tuples, all torch.float32 CPU tensors
    in MiniLM (EMBED_DIM) space -- label=1 if `candidate` is the cosine-nearest graph node to some step
    strictly after T within the lookahead window AND that match clears min_relevance, else 0.

    min_relevance matters, confirmed by a real offline test not assumed: bare cosine_rank(k=1) always
    returns SOME node, even for filler text with no real connection to anything in the graph ("let's look
    at the file structure" spuriously matched an unrelated concept purely on embedding-space noise) --
    without a floor, that noise gets treated as a real "this step used concept X" fact. 0.35 matches the
    RELEVANCE threshold already used for the same purpose in membrane.py's interactive_trace.

    Trajectories are materialized into a list BEFORE any encode_batch/g.matrix() call in this function --
    same real, confirmed environment fragility as _grow_from_swe_traces (HF datasets streaming's first
    real fetch in a process can segfault if interleaved with torch calls; g itself already having real
    embeddings, per the precondition above, means torch is already active by the time this runs)."""
    from v5.graph_grower.fetch_swe_traces import stream_swe_trajectories
    examples = []
    if len(g) == 0:
        return examples
    trajectories = list(stream_swe_trajectories(config=config, split=split, resolved_only=True,
                                                limit=n_trajectories))

    def _nearest_or_none(text: str):
        if not text:
            return None
        M, order = g.matrix()
        if not order:
            return None
        q = encode_batch([text])[0]
        sims = M @ q
        j = int(sims.argmax())
        return order[j] if float(sims[j]) >= min_relevance else None

    for traj in trajectories:
        steps = traj["steps"]
        step_texts = [(s.get("reasoning") or s.get("tool") or "").strip() for s in steps]
        if not any(step_texts):
            continue
        nearest_per_step = [_nearest_or_none(t) for t in step_texts]
        candidate_names = sorted({n for n in nearest_per_step if n})
        if not candidate_names:
            continue
        task_emb = torch.as_tensor(encode_batch([traj["problem_text"]])[0], dtype=torch.float32)
        candidate_embs = {n: torch.as_tensor(g.get(n).emb, dtype=torch.float32) for n in candidate_names}
        for t in range(len(steps)):
            progress_text = " ".join(x for x in step_texts[:t + 1] if x) or traj["problem_text"]
            progress_emb = torch.as_tensor(encode_batch([progress_text])[0], dtype=torch.float32)
            future_used = set(n for n in nearest_per_step[t + 1:t + 1 + lookahead_k] if n)
            for name in candidate_names:
                label = 1 if name in future_used else 0
                examples.append((task_emb, progress_emb, candidate_embs[name], label))
    return examples


class FutureNeedScorer(nn.Module):
    """Predicts P(candidate atom will be needed within a future lookahead window), given the current task
    + progress-so-far -- the speculative/proactive complement to generate_with_reground's existing
    backward-looking re-grounding (which only ever looks at generated_so_far, never ahead). Modeled
    directly on WMReasoner's own critic (critique/critic_loss, this same file) -- concat -> small MLP ->
    sigmoid, real supervised BCE against real hindsight labels (_hindsight_examples_from_swe_traces), same
    "train -> report real held-out accuracy vs base rate -> only trust if it beats base rate" discipline
    already used for the critic elsewhere in run_real.

    REAL GATE RESULT (not yet beating base rate -- reported honestly, not hidden): first real run showed
    0.78 held-out accuracy vs 0.65 base rate (looked like a real signal), but that split was at the EXAMPLE
    level -- different steps of the SAME trajectory (same task_emb, overlapping candidate pool) landed in
    both train and held-out, letting the model memorize per-trajectory patterns rather than generalize.
    Re-run with a proper TRAJECTORY-level split (32 train / 8 held-out trajectories, held-out trajectories
    never contributing a single training example): 0.56 accuracy vs 0.63 base rate -- WORSE than guessing
    the majority class, on 40 real Open-SWE-Traces trajectories (13k+ real hindsight-labeled examples).
    Honest read: this is a small sample (8 held-out trajectories is a small effective N even though it
    yields ~2400 individual examples, since examples from one trajectory are highly correlated) -- not
    proof the idea can't work, but no real signal found yet at this scale. Scaling to more real
    trajectories (207K available, only 100 fetched so far) is the natural next real test before concluding
    either way. Not wired into generate_with_reground -- gated behind this test passing, per the plan."""
    def __init__(self, d_in: int = EMBED_DIM, d_hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in * 3, d_hidden), nn.GELU(),
            nn.Linear(d_hidden, d_hidden // 2), nn.GELU(),
            nn.Linear(d_hidden // 2, 1),
        )

    def forward(self, task_emb: torch.Tensor, progress_emb: torch.Tensor,
               candidate_emb: torch.Tensor) -> torch.Tensor:
        x = torch.cat([task_emb, progress_emb, candidate_emb], dim=-1)
        return torch.sigmoid(self.net(x)).squeeze(-1)

    def loss(self, examples: list) -> torch.Tensor:
        task = torch.stack([e[0] for e in examples])
        prog = torch.stack([e[1] for e in examples])
        cand = torch.stack([e[2] for e in examples])
        y = torch.tensor([float(e[3]) for e in examples])
        preds = self.forward(task, prog, cand)
        return nn.functional.binary_cross_entropy(preds, y)


# ================================================================================================
# Real long-horizon task domain: OpenThoughts math CoT, verified against the dataset's own boxed final
# answer (never the model's own guess -- same anti-poison principle as every other verifier in this
# codebase). See _math_cot_tasks_from_graph below for the task-pool builder and honest small-N caveats.
# ================================================================================================
def _extract_boxed_answer(text: str) -> str | None:
    """Extract the content of the LAST \\boxed{...} in text, brace-depth aware (LaTeX content routinely
    nests braces, e.g. \\boxed{\\dfrac{5}{2}} -- a naive non-greedy regex truncates at the first inner '}',
    confirmed against real cached data before this was written, not assumed).

    Returns None if no \\boxed{ is found, OR if the last two \\boxed{} occurrences are separated only by
    trivial punctuation (comma/whitespace/'and') -- a REAL pattern found in real cached data (a problem
    whose true final answer was the 5-value set \\boxed{2}, \\boxed{3}, \\boxed{5}, \\boxed{7}, \\boxed{13}
    -- naively taking the last occurrence would have silently kept just '13' as if it were the whole
    answer). Filtered honestly rather than mis-extracted."""
    marker = "\\boxed{"
    spans = []
    i = 0
    while True:
        idx = text.find(marker, i)
        if idx == -1:
            break
        start = idx + len(marker)
        depth = 1
        j = start
        while j < len(text) and depth > 0:
            if text[j] == "{":
                depth += 1
            elif text[j] == "}":
                depth -= 1
            j += 1
        if depth == 0:
            spans.append((idx, j, text[start:j - 1]))
        i = idx + len(marker)
    if not spans:
        return None
    if len(spans) >= 2:
        import re as _re
        prev_end = spans[-2][1]
        last_start = spans[-1][0]
        between = text[prev_end:last_start]
        if _re.fullmatch(r"[,\s]*(and)?[,\s]*", between):
            return None
    return spans[-1][2]


def _parse_numeric(s: str) -> float | None:
    """Parse a boxed answer as a real number, or None if it isn't cleanly numeric -- proof statements,
    multiple-choice letters ('E'), geometric descriptions ('the midpoints form a hexagon'), and equations
    (survived a real check: 13/18 of real cached math CoT rows are exactly this non-numeric kind) all
    correctly return None here. This IS the filter (applied once, at task-pool-build time), not a coercion
    -- a wrong parse would poison the gold target, which nothing in this codebase's verifiers ever does."""
    import re as _re
    t = s.strip().replace("\\!", "").replace(",", "").replace("$", "").strip()
    m = _re.fullmatch(r"\\d?frac\{(-?\d+(?:\.\d+)?)\}\{(-?\d+(?:\.\d+)?)\}", t)
    if m:
        num, den = float(m.group(1)), float(m.group(2))
        return num / den if den != 0 else None
    try:
        return float(t)
    except ValueError:
        return None


def _answers_match(generated: str, gold: float, tol: float = 1e-4) -> bool:
    """Extract the boxed answer from a real LM generation, parse numeric, compare to the dataset's own
    gold answer with RELATIVE tolerance (real answers range from small integers to the thousands -- a fixed
    absolute tolerance would be too strict for large values and too loose for small ones)."""
    boxed = _extract_boxed_answer(generated)
    if boxed is None:
        return False
    val = _parse_numeric(boxed)
    if val is None:
        return False
    return abs(val - gold) <= tol * max(1.0, abs(gold))


def _fmt_gold(gold: float) -> str:
    """Render a gold float as the teacher-forcing target text -- \\boxed{72} not \\boxed{72.0} for integer
    answers, matching how a real solution actually writes it (checked against real cached examples)."""
    return str(int(gold)) if float(gold).is_integer() else str(gold)


def _math_cot_tasks_from_graph(g, n_train: int = 24, n_held: int = 8, n_raw: int = 150,
                               domains: tuple = ("math",), min_reasoning_chars: int = 200,
                               k_related: int = 3, seed: int = 0, raw_rows: list | None = None):
    """Real long-horizon task pool: stream OpenThoughts-114k rows directly (bypassing fetch_cot.py's
    row_to_doc, which concatenates problem+reasoning+solution into one blob for the graph-growth use case --
    here we need the fields SEPARATE: the short problem statement for task_emb, and deepseek_solution alone
    for boxed-answer extraction), keep only rows with a genuinely NUMERIC final boxed answer (see
    _extract_boxed_answer/_parse_numeric -- symbolic/proof/multiple-choice/multi-value answers are filtered
    out, not force-fit into a numeric comparator), and ground each in K related CONCEPT nodes already banked
    in the graph (via g.cosine_rank, real per-problem retrieval, reusing the concept nodes _grow_from_cot
    already banks -- no new retrieval infra) since a free-text problem has no discrete inner/outer atom pair
    the way synthetic composition does.

    Returns (train_tasks, held_tasks, related_desc_pool, stats):
      - train_tasks/held_tasks: (text, atoms_needed, code, code_expr) 4-tuples, SAME shape
        _compose_tasks_from_graph produces. atoms_needed here = the K related concept nodes' DESCRIPTION
        TEXT (not their real graph names, which are meaningless hashes like 'fact_59378' -- embedding a
        hash string via encode_batch would carry no semantic signal; the description IS the meaningful
        content, so it doubles as both the grounding text and its own lookup key here). code = the teacher-
        forcing target string (e.g. '\\boxed{72}'); code_expr = the raw gold float.
      - related_desc_pool: deduped list of every concept description actually used -- plays the same role
        atom_names plays for composition tasks (the caller sets atom_names = this for math-cot mode).
      - stats: real seen/kept/skipped counts, reported honestly -- only a FRACTION of real CoT problems
        have a clean numeric final answer (measured directly: 4/18 on the locally cached sample; most are
        proofs/inequalities/multiple-choice). Do not silently end up with a tiny pool and call it a held-out
        set without saying so -- this is exactly the discipline every other growth function in this file
        already follows (_grow_from_cot, _grow_skills_from_corpus).

    raw_rows: pass pre-fetched rows (e.g. via v5.graph_grower.fetch_cot.stream_raw_rows or its saved jsonl,
    `python -m v5.graph_grower.fetch_cot --raw --out <path> --limit N`, from a separate torch-free process)
    to skip the internal live load_dataset() call. Real reason this matters, not just an option: HF
    `datasets` streaming's first real fetch in a process segfaults if torch was already active earlier in
    it, and by the time this function runs from --run (task_domain=math-cot), the LM has already been
    loaded -- confirmed directly, live-streaming here crashes the same way _grow_from_cot's did."""
    import random as _random

    dom_set = {d.lower() for d in domains}
    if raw_rows is not None:
        ds = raw_rows
    else:
        from v5.graph_grower.fetch_cot import DATASET, CONFIG
        from datasets import load_dataset
        ds = load_dataset(DATASET, CONFIG, split="train", streaming=True)
    seen = kept = skipped_domain = skipped_short = skipped_no_numeric_answer = 0
    pool: list[tuple[str, float]] = []
    for row in ds:
        if seen >= n_raw:
            break
        ot_domain = str(row.get("domain") or "").strip().lower()
        if ot_domain not in dom_set:
            skipped_domain += 1
            continue
        problem = str(row.get("problem") or "").strip()
        reasoning = str(row.get("deepseek_reasoning") or "").strip()
        solution = str(row.get("deepseek_solution") or "").strip()
        seen += 1
        if len(reasoning) < min_reasoning_chars or not problem or not solution:
            skipped_short += 1
            continue
        boxed = _extract_boxed_answer(solution)
        gold = _parse_numeric(boxed) if boxed is not None else None
        if gold is None:
            skipped_no_numeric_answer += 1
            continue
        pool.append((problem, gold))
        kept += 1

    rng = _random.Random(seed)
    rng.shuffle(pool)
    n_train_actual = min(n_train, max(0, len(pool) - 1)) if len(pool) > 1 else 0
    n_held_actual = min(n_held, len(pool) - n_train_actual)

    related_desc_pool: list[str] = []
    seen_desc: set[str] = set()

    def _related_descs(problem_text: str) -> list[str]:
        ranked = g.cosine_rank(problem_text, k=k_related * 4)
        out = []
        for name in ranked:
            a = g.get(name)
            if a and a.kind == "concept" and a.description:
                out.append(a.description)
                if a.description not in seen_desc:
                    seen_desc.add(a.description)
                    related_desc_pool.append(a.description)
                if len(out) >= k_related:
                    break
        return out

    def _mk(problem_text: str, gold: float):
        related = _related_descs(problem_text)
        if not related:
            return None  # no concept-node context available -- skip rather than ground in nothing
        return (problem_text, related, f"\\boxed{{{_fmt_gold(gold)}}}", gold)

    train_tasks = [t for p, gd in pool[:n_train_actual] if (t := _mk(p, gd)) is not None]
    held_tasks = [t for p, gd in pool[n_train_actual:n_train_actual + n_held_actual] if (t := _mk(p, gd)) is not None]
    stats = dict(seen=seen, kept=kept, skipped_domain=skipped_domain, skipped_short=skipped_short,
                 skipped_no_numeric_answer=skipped_no_numeric_answer,
                 n_train=len(train_tasks), n_held=len(held_tasks))
    return train_tasks, held_tasks, related_desc_pool, stats


def _swe_action_tasks_from_graph(g, n_train: int = 24, n_held: int = 8, n_trajectories: int = 60,
                                 k_related: int = 3, min_prior_steps: int = 3, max_ctx_steps: int = 12,
                                 seed: int = 0, trajectories: list | None = None,
                                 max_issue_chars: int = 1500, max_step_chars: int = 200,
                                 target_args_chars: int = 0, max_per_traj: int = 2):
    """Real long-horizon task pool: NEXT-ACTION PREDICTION on nvidia/Open-SWE-Traces.

    Given a real GitHub issue plus the agent's real prior steps 1..T, predict step T+1's tool call. This is
    the domain the hierarchical/top-TRM work actually needs and composition could never provide: real
    trajectories average ~55 steps, so generations are long enough that KV eviction genuinely fires, and
    which knowledge matters changes across the trajectory (explore the repo -> locate the defect -> edit ->
    test), which is the entire premise of a slow memory level.

    Returns (train_tasks, held_tasks, related_desc_pool, stats) -- the SAME 4-value shape
    _math_cot_tasks_from_graph returns, and each task is the SAME 4-tuple every builder here produces:
    (text, atoms_needed, code, code_expr).
      - text        = issue + compressed prior steps (the prompt content)
      - atoms_needed= K related step-concept DESCRIPTIONS from the graph (descriptions, not names --
                      exactly math-cot's convention, and required because run_real does
                      atom_names.index(a) and the caller sets atom_names = related_desc_pool)
      - code        = teacher-forcing target, the real next tool call rendered as text
      - code_expr   = the gold tool NAME alone (what verify() compares against)

    trajectories: pass pre-fetched rows to skip the internal live stream. Not optional in practice when
    called from run_real -- HF `datasets` streaming's first fetch segfaults once torch is active, a
    confirmed environment bug this codebase already works around for --grow-cot and --task-domain math-cot.

    SPLIT DISCIPLINE: train/held are split by TRAJECTORY, never by step. Two steps of the same trajectory
    share the issue text and most of their context, so a step-level split would leak badly -- the same
    mistake already made and caught once in this codebase's FutureNeedScorer work (0.78 accuracy collapsed
    to 0.56 once the split was done properly)."""
    import random as _random
    if trajectories is None:
        from v5.graph_grower.fetch_swe_traces import stream_swe_trajectories
        trajectories = list(stream_swe_trajectories(resolved_only=True, limit=n_trajectories))

    rng = _random.Random(seed)
    trajs = list(trajectories)
    rng.shuffle(trajs)
    n_train_traj = max(1, int(0.8 * len(trajs))) if len(trajs) > 1 else 1
    split_trajs = {"train": trajs[:n_train_traj], "held": trajs[n_train_traj:]}

    related_desc_pool: list[str] = []
    seen_desc: set[str] = set()

    def _related_descs(query_text: str) -> list[str]:
        # Route through nested worlds when the graph has them: same ranking quality at a fraction of the
        # scan (measured: 7.3x fewer candidates touched for equal recall, 52x fewer at top_w=3). Falls
        # back to the flat scan automatically on graphs built without worlds.
        ranked = (g.cosine_rank_world(query_text, k=k_related * 4, top_w=8)
                  if getattr(g, "_worlds_on", False) else g.cosine_rank(query_text, k=k_related * 4))
        out = []
        for name in ranked:
            a = g.get(name)
            if a and a.description:
                out.append(a.description)
                if a.description not in seen_desc:
                    seen_desc.add(a.description)
                    related_desc_pool.append(a.description)
                if len(out) >= k_related:
                    break
        return out

    stats = dict(trajectories=len(trajs), skipped_short=0, skipped_no_tool=0, skipped_no_related=0)

    def _tasks_from(traj_list, want):
        # ROUND-ROBIN across trajectories, capped at max_per_traj each. The first version was depth-first:
        # it drained one trajectory (~52 usable steps out of ~55) before touching the next, then stopped as
        # soon as it had enough. Measured consequence on a real 120-trajectory set -- all 48 TRAIN tasks
        # came from 2 GitHub issues and all 16 HELD tasks from a SINGLE issue, leaving 117 of 120
        # trajectories entirely unused.
        #
        # That is not a diversity nicety, it silently broke the benchmark. With one held trajectory the
        # held tool distribution is whatever that one issue happened to do, and it came out OPPOSITE to
        # train: train majority str_replace_editor 56%, held majority execute_bash 75%. A model that
        # correctly learns the training majority therefore scores 4/16 on held BY CONSTRUCTION -- which is
        # exactly the observed behaviour (WM emitted str_replace_editor nearly always and landed on 6/16,
        # below the 10/16 a constant execute_bash would score). The apparent "model is worse than a
        # constant predictor" result was this sampling bug, not the model.
        per_traj = max(1, max_per_traj)
        out = []
        for depth in range(0, per_traj):
            if len(out) >= want:
                break
            for traj in traj_list:
                if len(out) >= want:
                    break
                _tasks_from_one(traj, out, want, depth)
        return out

    def _tasks_from_one(traj, out, want, depth):
        for _ in (0,):
            steps = traj.get("steps") or []
            if len(steps) < min_prior_steps + 1:
                if depth == 0:
                    stats["skipped_short"] += 1
                continue
            # CAP THE ISSUE TEXT. Leaving this uncapped was a real bug: step reasoning was truncated to
            # 200 chars but the issue body was not, and real problem_text runs to 18,841 chars (p50 1,928
            # / p90 4,454 / p99 6,491 over 120 real trajectories). Combined with 12 steps of context that
            # produced ~5,300-token prompts, and since attention memory in the training backward pass is
            # QUADRATIC in sequence length, a real run reached ~60GB of VRAM on a 4-bit 4B model whose
            # weights are only ~2.5GB. The head of an issue carries the actual problem statement; the tail
            # is typically stack traces, version tables and reproduction logs.
            issue = (traj.get("problem_text") or "").strip()[:max_issue_chars]
            # `depth`-th usable step of THIS trajectory only -- the round-robin caller advances depth, so
            # every trajectory contributes its 1st step before any contributes its 2nd. Spread evenly
            # rather than draining one issue at a time.
            usable = [t for t in range(min_prior_steps, len(steps))
                      if (steps[t].get("tool") or "").strip()]
            if depth >= len(usable):
                continue
            for t in usable[depth:depth + 1]:
                if len(out) >= want:
                    break
                nxt = steps[t]
                gold_tool = (nxt.get("tool") or "").strip()
                prior = steps[max(0, t - max_ctx_steps):t]
                ctx = "\n".join(
                    f"Step {i + 1}: {(sp.get('reasoning') or '').strip()[:max_step_chars]}"
                    + (f" [action: {sp.get('tool')}]" if sp.get("tool") else "")
                    for i, sp in enumerate(prior))
                text = f"{issue}\n\nProgress so far:\n{ctx}"
                related = _related_descs(text)
                if not related:
                    stats["skipped_no_related"] += 1
                    continue
                # TARGET MUST MATCH WHAT IS SCORED. The first version teacher-forced the full
                # `tool(args)` string, but verify() only checks the TOOL NAME, and the two are wildly
                # mismatched in size: measured over 3,396 real steps the full target runs p50 43 / p90 377
                # / max 3,949 tokens while the tool name is always 3. So nearly all the loss landed on
                # argument strings -- instance-specific absolute paths and shell commands that are not
                # learnable in principle -- and none of it on the 3 tokens being graded. That is visible
                # in real output: the model emitted plausible ARGUMENTS ('"cd /workspace/old_api && ls
                # -la ...') and never named a tool, while the ablated baseline emitted
                # '[action: execute_bash]' and passed. It also explains WM getting WORSE as lm loss
                # improved -- it was optimizing the part that is not measured.
                # target_args_chars > 0 restores a truncated-args target for anyone who wants it.
                args = (nxt.get("args") or "").strip()
                if target_args_chars > 0:
                    target = f"{gold_tool}({args[:target_args_chars]})" if args else f"{gold_tool}()"
                else:
                    target = gold_tool
                out.append((text, related, target, gold_tool))
            if len(out) >= want:
                break
        return out

    train_tasks = _tasks_from(split_trajs["train"], n_train)
    held_tasks = _tasks_from(split_trajs["held"], n_held)
    stats.update(n_train=len(train_tasks), n_held=len(held_tasks),
                 n_train_traj=len(split_trajs["train"]), n_held_traj=len(split_trajs["held"]))
    # Gold-tool distribution: exact-match accuracy against a dominant tool is a base-rate artifact, so the
    # majority share is reported alongside the score rather than left for the reader to assume.
    # Real prompt-size stats. The uncapped-issue bug above was invisible until a run hit ~60GB of VRAM;
    # printing the actual character budget makes any regression here immediately obvious instead.
    _all = train_tasks + held_tasks
    if _all:
        _lens = sorted(len(t[0]) for t in _all)
        stats["train_trajectories_used"] = len({t[0][:120] for t in train_tasks})
        stats["held_trajectories_used"] = len({t[0][:120] for t in held_tasks})
        stats["prompt_chars_p50"] = _lens[len(_lens) // 2]
        stats["prompt_chars_max"] = _lens[-1]
        stats["approx_tokens_max"] = _lens[-1] // 4
    from collections import Counter as _Counter
    gold_counts = _Counter(t[3] for t in held_tasks)
    stats["held_gold_tools"] = dict(gold_counts)
    stats["held_majority_rate"] = (max(gold_counts.values()) / len(held_tasks)) if held_tasks else 0.0
    return train_tasks, held_tasks, related_desc_pool, stats


def _critic_ctx(wb, task_emb: torch.Tensor, generated: str) -> torch.Tensor:
    """concat(task_emb, generated_text_emb) -- the critic's "what was I asked for vs what did I produce"
    input. Both MiniLM-space so they are directly comparable; task_emb already is, and the generation is
    embedded here. Detached: the critic is an AMORTIZER trained post-hoc on real verified labels, and must
    never push gradients back into the reasoner it is judging (that would let the reasoner learn to fool
    its own judge -- the reward-hacking failure this codebase has deliberately avoided all along)."""
    gen = (generated or "").strip()[:1000]
    gen_emb = torch.as_tensor(encode_batch([gen if gen else "(empty)"])[0],
                              dtype=torch.float32, device=task_emb.device)
    return torch.cat([task_emb.detach().float(), gen_emb]).detach()


def _roc_auc(scores: list, labels: list) -> float:
    """Threshold-free ranking quality: P(random positive scored above random negative). Reported alongside
    accuracy because accuracy alone was actively misleading here -- with a skewed test set a constant
    predictor lands exactly on the base rate (or its complement), which is precisely how a saturated,
    literally-constant critic was mistaken for "no signal found" across many runs. AUC is 0.5 for ANY
    constant predictor regardless of class balance, so it cannot be faked that way."""
    pairs = sorted(zip(scores, labels))
    pos = sum(1 for _, l in pairs if l)
    neg = len(pairs) - pos
    if pos == 0 or neg == 0:
        return float("nan")
    # rank-sum (Mann-Whitney U), average ranks for ties
    ranks, i = {}, 0
    vals = [s for s, _ in pairs]
    while i < len(vals):
        j = i
        while j + 1 < len(vals) and vals[j + 1] == vals[i]:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[k] = avg
        i = j + 1
    rank_sum_pos = sum(ranks[idx] for idx, (_, l) in enumerate(pairs) if l)
    return (rank_sum_pos - pos * (pos + 1) / 2.0) / (pos * neg)


class _PassiveGrowth:
    """Graph growth DURING training, driven only by real verified outcomes.

    Two things were missing before this. (1) `record_success`/`record_failure` only ever adjusted
    confidence and edge strength -- no new node was ever created mid-run; `learn_any` ran once at setup and
    the graph was then a frozen embedding source for the whole run. (2) On the description-based domains
    (math-cot, swe-action) even those updates SILENTLY NO-OPPED: `atoms_needed` there holds concept
    DESCRIPTIONS, while `record_success` looks nodes up by NAME, so `g.get(description)` found nothing and
    the call did nothing at all. That was documented in the eval loop as "needs a real name<->description
    mapping, not built yet" -- this class is that mapping.

    Anti-poison rule is unchanged and is the whole point: a node is banked ONLY from an outcome the
    verifier actually confirmed. Failures become explicit trap nodes (record_failure's existing behaviour),
    giving the graph a signed memory of what did not work.

    Trap budget: an earlier real graph in this codebase had to be repaired after failure bookkeeping
    crowded out useful structure (scripts/graph_connectivity_report.py still warns above trap_frac 0.3), so
    traps are capped per epoch rather than allowed to grow without bound."""

    def __init__(self, g, enabled: bool, trap_budget_per_epoch: int = 8):
        self.g = g
        self.enabled = enabled
        self.trap_budget = trap_budget_per_epoch
        self.traps_this_epoch = 0
        self.banked = 0
        self.trapped = 0
        self.trap_skipped = 0
        self._retr = None
        # description -> real node name, so record_success/record_failure work on the text domains too
        self.desc2name = {a.description: n for n, a in g.atoms.items() if a.description}

    def _retriever(self):
        if self._retr is None:
            from v5.runtime.membrane import TRMRetriever
            self._retr = TRMRetriever(self.g)
        return self._retr

    def epoch_reset(self):
        self.traps_this_epoch = 0

    def resolve(self, atoms_needed):
        """Map whatever the domain put in atoms_needed (real names OR descriptions) to real node names,
        dropping anything that resolves to nothing rather than passing a phantom name downstream."""
        out = []
        for a in atoms_needed:
            if a in self.g.atoms:
                out.append(a)
            elif a in self.desc2name:
                out.append(self.desc2name[a])
        return out

    def update(self, atoms_needed, text: str, ok: bool, learn_text: str | None = None):
        from v5.runtime.membrane_edits import record_success, record_failure
        names = self.resolve(atoms_needed)
        if ok:
            if names:
                record_success(self.g, names, text)
            if self.enabled and learn_text and len(learn_text.strip()) >= 30:
                from v5.runtime.membrane import learn_any
                res = learn_any(self.g, self._retriever(), learn_text.strip()[:2000])
                if res.get("status") in ("banked-fact", "banked-skill"):
                    self.banked += 1
                    nm = res.get("node")
                    if nm and (a := self.g.get(nm)) is not None and a.description:
                        self.desc2name.setdefault(a.description, nm)
        else:
            if self.traps_this_epoch < self.trap_budget:
                record_failure(self.g, text)
                self.traps_this_epoch += 1
                self.trapped += 1
            else:
                self.trap_skipped += 1

    def summary(self) -> str:
        census = self.g.census()
        n = max(1, len(self.g))
        trap_frac = census.get("trap", 0) / n
        return (f"passive growth: banked {self.banked} verified nodes, {self.trapped} traps "
                f"({self.trap_skipped} suppressed by budget) -- graph now {len(self.g)} nodes, "
                f"trap_frac {trap_frac:.2f}"
                + ("  [WARN >0.30: failure bookkeeping is crowding out real structure]"
                   if trap_frac > 0.30 else ""))


def _swe_action_match(generated: str, gold_tool: str) -> bool:
    """Did the model name the right next tool? Exact tool-name match against the real trajectory's recorded
    action -- a real, recorded fact, never a model judgement (same anti-poison rule as every other verifier
    here). Tolerant of the model's surrounding prose/formatting: the gold tool name must appear as a
    whole-word token in the generation, and no OTHER known tool may appear before it (so 'I should not use
    execute_bash, instead str_replace_editor' does not silently count as execute_bash)."""
    import re as _re
    if not gold_tool:
        return False
    text = generated.strip()
    hits = [(m.start(), m.group(0)) for m in _re.finditer(r"[A-Za-z_][A-Za-z0-9_]*", text)]
    for _pos, tok in hits:
        if tok == gold_tool:
            return True
        if tok in _SWE_KNOWN_TOOLS and tok != gold_tool:
            return False          # named a different real tool first -> wrong action
    return False


# Real tool names observed in nvidia/Open-SWE-Traces (openhands config). Used only to detect "the model
# named a DIFFERENT tool first"; an unknown identifier never counts as a competing tool.
_SWE_KNOWN_TOOLS = {
    "str_replace_editor", "execute_bash", "execute_ipython_cell", "browser", "finish",
    "think", "web_read", "str_replace", "create", "edit_file",
}


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
    return _run_task, ns


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
        # Return code expression to be used by oracle
        return (text, [inner, outer], code, f"{outer}({inner}(n))")

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


_OPAQUE_TOKENS = ["qx7", "vk2", "zn9", "hb4", "tm6", "pj1", "wd8", "rc5", "gf3", "ly0"]


def _opaquify(descs: dict, codes: dict):
    """Rename every atom to a NON-SEMANTIC token, keeping its English description intact.

    THIS IS THE FIX FOR THE EXPERIMENT'S CENTRAL FLAW. With the shipped names the task text is
    generated from phrases describing exactly the operation the atom is named after -- "the square
    of the nth Fibonacci number" -> square(fibonacci(n)) -- so the LM can write the answer straight
    from the prompt and never consult working memory. Observed in a real run:

        target : def task(n): return square(fibonacci(n))
        WM     : def task(n): return fibonacci(n)**2   <- PASSED, using neither composed atom

    verify() is behavioural, so that counts as success. When the task is solvable from the prompt,
    WM and DERANGED CANNOT differ and the derangement arm has nothing to detect -- every
    "instance-specific" number measured on this task, including the +0.1350 reported earlier, was
    measuring prompt-format compliance rather than memory.

    Renaming breaks that path. The description still says WHAT to compute, so the task stays
    well-posed; only working memory can say WHICH identifier implements it, and a deranged slot
    table supplies provably wrong names.
    """
    old = list(descs.keys())
    mapping = {o: f"op_{_OPAQUE_TOKENS[i % len(_OPAQUE_TOKENS)]}" for i, o in enumerate(old)}
    nd = {mapping[o]: descs[o] for o in old}
    nc = {mapping[o]: codes[o].replace(f"def {o}(", f"def {mapping[o]}(", 1) for o in old}
    return nd, nc, mapping


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
    # Real bug, confirmed by direct execution not assumed: this used to be a single tuple assignment
    # `n_train, n_held = min(n_train, len(pairs)-4), min(n_held, len(pairs)-n_train)` -- Python evaluates
    # the WHOLE right-hand side using the ORIGINAL n_train before either name is rebound, so whenever the
    # requested n_train exceeded the pool, `len(pairs) - n_train` went NEGATIVE (confirmed:
    # n_train=100 request -> len(pairs)-100 = 64-100 = -36 -> min(16,-36) = -36 -> pairs[60:60-36] =
    # pairs[60:24], an empty slice -- 0 held-out tasks, silently, for every epoch, no matter how long
    # training ran). Fixed by computing the capped n_train FIRST, then using THAT (not the stale original)
    # to cap n_held.
    n_train = min(n_train, len(pairs) - 4)
    n_held = min(n_held, len(pairs) - n_train)

    def _mk(outer, inner):
        text = outer_template[outer].format(inner=inner_phrase[inner])
        code = f"def task(n): return {outer}({inner}(n))"
        return (text, [inner, outer], code, f"{outer}({inner}(n))")

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
            batch_size: int = 1, task_domain: str = "synthetic", math_cot_n_raw: int = 150,
            top_trm_t: int = 0, reground_chunk_tokens: int = 16, reground_top_every: int = 4,
            max_new_tokens: int = 0, use_kv_cache: bool = False, evict_window: int | None = None,
            grow_cot_docs_path: str | None = None, math_cot_docs_path: str | None = None,
            trigger_patterns: list | None = None, instability_trigger: float | None = None,
            sink_tokens: int = 0, cotrain_samples: int = -1,
            top_no_graph: bool = False, top_memory_max: int = 16,
            evict_to_memory: bool = False, swe_docs_path: str | None = None,
            passive_growth: bool = False, gate_init: float = 0.8,
            swe_max_issue_chars: int = 1500, swe_max_ctx_steps: int = 12,
            gate_max: float = 0.0, merged: bool = False, swe_target_args_chars: int = 0,
            ds_weight: float = 0.0, conv_weight: float = 0.05, gate_reg_weight: float = 0.05,
            adaptive_t: bool = False, ponder_weight: float = 0.0,
            slot_gate: bool = False, slot_theta0: float = 0.0,
            slot_w_inh: float = 0.5, slot_t: int = 3, derange: bool = True,
            n_derange: int = 5, contrast_weight: float = 0.0, contrast_negs: int = 15,
            no_proj_bias: bool = False, trm_no_bias: bool = False,
            atom_slots: bool = False, slotdim_k: int = 0,
            slotdim_depth: int = 1, slotdim_no_ffn: bool = False,
            slotdim_heads: int = 0, couple_frac: list | None = None,
            delta_scale: float = 0.3, delta_mode: str = "rescale",
            eval_every_arg: int = 0, opaque_atoms: bool = False,
            code_prompt: bool = False):
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
    (_atoms_from_graph filters on real .code), unlike grow_cot's concept nodes.

    task_domain='synthetic' (default): UNCHANGED, byte-identical behavior. task_domain='math-cot' (requires
    graph_path; run --grow-cot at least once first so concept nodes exist to ground in): real, long-horizon
    OpenThoughts math CoT problems instead of synthetic composition -- see _math_cot_tasks_from_graph.
    Verified against the dataset's own boxed final answer (_answers_match), never the model's own guess --
    same anti-poison principle, different verifier shape (no execution oracle exists for free-text math).
    Honest small-N caveat: only a fraction of real problems have a clean numeric final answer -- real counts
    are printed, not hidden."""
    from v5.runtime.dcpd_latent import WhiteBox
    from v5.runtime.algo_trm import _build as _build_trm
    from v5.runtime.membrane_edits import record_success, record_failure
    import random
    print(f"run_real: WMReasoner + TRMReasoner V3 coupled to {lm_name} ({quant}) — real composition tasks\n")

    _, _, TRMReasoner = _build_trm()
    wb = WhiteBox(lm_name, quant=quant)
    d_lm = wb.d_model
    # COUPLING DEPTH. The default injects at only the LAST TWO layers, leaving the model almost no
    # depth to integrate slot CONTENT -- enough to shift output FORMAT, not enough to change which
    # atom gets named. That is the shape every arm has measured: ~+7.3 nats of format effect against
    # a near-zero content effect. couple_frac moves the injection earlier so this is testable.
    if couple_frac:
        couple = sorted({max(0, min(wb.n_layers - 1, int(round(f * (wb.n_layers - 1)))))
                         for f in couple_frac})
        print(f"  COUPLING at layers {couple} of {wb.n_layers} (fractions {couple_frac}) instead of "
              f"the default last-two.", flush=True)
    else:
        couple = [wb.n_layers - 2, wb.n_layers - 1]
    if delta_mode != "rescale":
        print(f"  DELTA MODE '{delta_mode}': rescale forces EVERY position to receive exactly "
              f"delta_scale*||h||, making per-position emphasis unrepresentable and letting the "
              f"adapter learn a format prior that survives its own ablation. This project's standing "
              f"rule is clip, not rescale.", flush=True)
    print(f"  LM: {lm_name}  d={d_lm}  layers={wb.n_layers}  gate layers={couple}  device={wb.device}")

    trm = TRMReasoner(d_in=EMBED_DIM, d=256, T=4, n_heads=4, adaptive=adaptive_t)
    top_trm = TRMReasoner(d_in=EMBED_DIM, d=256, T=top_trm_t, n_heads=4) if top_trm_t > 0 else None
    R = WMReasoner(d_lm, couple_layers=couple, trm=trm, n_heads=4, top_trm=top_trm,
                   delta_scale=delta_scale, delta_mode=delta_mode).to(wb.device)
    # SPIKING SLOT GATE (opt-in). Off => the adapters take their original dense-softmax path
    # bit-for-bit, so this flag cannot perturb any previously measured run.
    if slot_gate:
        R.slot_gate = SpikingSlotGate(enabled=True, theta0=slot_theta0, w_inh=slot_w_inh,
                                      T=slot_t).to(wb.device)
        print(f"  SPIKING SLOT GATE ON: theta0={slot_theta0} w_inh={slot_w_inh} T={slot_t} "
              f"(per-position, over K={R.M} slots)", flush=True)
    if top_trm is not None:
        print(f"  hierarchical: top_trm T={top_trm_t} (bottom T=4), reground every {reground_chunk_tokens} "
              f"tokens, top refreshed every {reground_top_every} chunks. Training now calls "
              f"hierarchical_refine (not plain refine) so top_to_bottom_proj/top_trm get a real gradient "
              f"from the same verified-target lm_loss as everything else -- zero-init means it starts as "
              f"a no-op, but it CAN move now. 'held WM' below is the one-shot hierarchical injection; "
              f"'reground' additionally re-grounds it mid-generation -- the gap between them isolates the "
              f"value of periodic re-grounding specifically, not just of having a top_trm at all.")
    for p in wb.model.parameters():
        p.requires_grad_(False)
    handles = R.couple(wb)

    # max_new_tokens=0 -> domain-aware default. 128 was hardcoded everywhere before this, fine for
    # synthetic's ~13-token compose-two-atoms completions but far too small for real math-cot CoT
    # (OpenThoughts reasoning traces routinely run 300-1000+ tokens before reaching \boxed{...}) -- at 128
    # the model gets cut off mid-reasoning and never emits \boxed{}, so verify() reads as wrong regardless
    # of whether the reasoning was on track. 512 is a real budget for that, not a guess: still cheaper than
    # letting it run unbounded, generous enough that most real single-numeric-answer CoT problems can
    # actually reach a boxed conclusion.
    # Domain-aware generation budget. swe-action's target is a single tool call, so it needs far less
    # than math-cot's long CoT; keeping it small also keeps the many-chunk reground path cheap.
    _dom_default = {"math-cot": 512, "swe-action": 64}.get(task_domain, 128)
    eff_max_new_tokens = max_new_tokens if max_new_tokens > 0 else _dom_default
    print(f"  max_new_tokens={eff_max_new_tokens}"
          f"{' (auto, domain-aware)' if max_new_tokens == 0 else ' (explicit)'}")

    if graph_path:
        from pathlib import Path as _Path
        from v5.runtime.membrane import AtomGraph, seed_graph
        g = AtomGraph.load(graph_path) if _Path(graph_path).exists() else seed_graph()
        if grow_cot > 0:
            n0 = len(g)
            # grow_cot_docs_path: real, safe path around a real crash -- HF `datasets` streaming's first
            # real fetch in a process segfaults if torch was already active earlier in it (confirmed; see
            # _grow_from_cot's docstring), and by the time this line runs, torch has been loaded since
            # trm_wm.py's own module import -- there is no in-process ordering fix for --run specifically.
            # Pre-fetch with `python -m v5.graph_grower.fetch_cot --out <path> --limit N` (a separate,
            # torch-free process) first, then pass that file here instead of live-streaming.
            pre_docs = None
            if grow_cot_docs_path:
                import json as _json
                with open(grow_cot_docs_path, encoding="utf-8") as _f:
                    pre_docs = [_json.loads(line) for line in _f if line.strip()]
                print(f"  grow-cot: using {len(pre_docs)} pre-fetched docs from {grow_cot_docs_path} "
                      f"(avoids the real datasets/torch ordering crash -- see _grow_from_cot's docstring)")
            stats = _grow_from_cot(g, grow_cot, domains=grow_domains, keywords=grow_keywords, docs=pre_docs)
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
        if task_domain in ("math-cot", "swe-action"):
            descs, codes = {}, {}
            atom_names = []  # set below from related_desc_pool -- concept-node descriptions, not code atoms
        else:
            descs, codes = _atoms_from_graph(g)
            atom_names = list(descs.keys())
            print(f"  graph: {len(atom_names)} REAL atoms from {graph_path if _Path(graph_path).exists() else '(fresh seed_graph)'} "
                  f"(NATIVE LM-embedding-table injection)")
    else:
        if task_domain in ("math-cot", "swe-action"):
            raise ValueError(f"--task-domain {task_domain} requires --graph-path (needs real concept nodes "
                             f"to ground in -- run --grow-cot (math-cot) or --swe-docs-path (swe-action) "
                             f"at least once first)")
        descs, codes = _seed_atoms()
        if opaque_atoms:
            descs, codes, _amap = _opaquify(descs, codes)
            print(f"  OPAQUE ATOMS ON: names renamed to non-semantic tokens "
                  f"({', '.join(list(descs)[:3])}, ...). The description still says WHAT to compute; "
                  f"only working memory can say WHICH name implements it, so the prompt no longer "
                  f"contains a complete solution path.", flush=True)
        atom_names = list(descs.keys())
        # NATIVE-SPACE injection (probe-C-validated): embed each atom's description via the LM's OWN embedding
        # table, not MiniLM + an untrained proj_atom bridge -- that bridge is exactly what probe B showed
        # collapses on held-out (train fits, held-out ~0). This was very likely why composition scored 0/4 even
        # after deep supervision was fixed: the atoms fed to refine() were never in a space the LM could read.
        print(f"  graph: {len(atom_names)} atoms (NATIVE LM-embedding-table injection, MiniLM dropped for this path)")

    if task_domain in ("math-cot", "swe-action"):
        # skip the native-embedding-table precompute below -- dead weight for this domain: WMReasoner.
        # refine() takes MiniLM-space embeddings post-V3-rewrite, this dict is never consulted downstream
        # (confirmed by reading the training/eval loops -- they recompute MiniLM embeddings on the fly).
        atom_embs = {}
    else:
        # BATCHED native-text embedding: one tokenizer + embedding-table pass instead of N separate calls.
        # For large graphs (100+ atoms) this is ~10x faster; identical output (same LM embedding table).
        atom_names_list = list(atom_names)
        descs_list = [descs[n] for n in atom_names_list]
        atom_emb_tensor = native_text_embedding_batch(wb, descs_list)
        atom_embs = {n: atom_emb_tensor[i] for i, n in enumerate(atom_names_list)}

    if task_domain == "swe-action":
        pre_trajs = None
        if swe_docs_path:
            import json as _json
            with open(swe_docs_path, encoding="utf-8") as _f:
                pre_trajs = [_json.loads(line) for line in _f if line.strip()]
            print(f"  swe-action: using {len(pre_trajs)} pre-fetched trajectories from {swe_docs_path} "
                  f"(avoids the real datasets/torch ordering crash)")
        train_tasks, held_tasks, related_desc_pool, sw_stats = _swe_action_tasks_from_graph(
            g, n_train=n_train, n_held=n_held, trajectories=pre_trajs,
            max_issue_chars=swe_max_issue_chars, max_ctx_steps=swe_max_ctx_steps,
            target_args_chars=swe_target_args_chars)
        atom_names = related_desc_pool
        print(f"  swe-action: real Open-SWE-Traces next-action prediction -> {sw_stats['n_train']} train "
              f"({sw_stats['n_train_traj']} trajectories), {sw_stats['n_held']} held-out "
              f"({sw_stats['n_held_traj']} trajectories; split by TRAJECTORY not step, so issue text "
              f"cannot leak across the split). skipped: short={sw_stats['skipped_short']} "
              f"no_tool={sw_stats['skipped_no_tool']} no_related={sw_stats['skipped_no_related']}")
        print(f"  swe-action: drawn from {sw_stats.get('train_trajectories_used', 0)} distinct TRAIN "
              f"trajectories and {sw_stats.get('held_trajectories_used', 0)} distinct HELD trajectories. "
              f"A depth-first sampler once drained one issue at a time, giving 2 train / 1 held -- with a "
              f"single held trajectory the held tool distribution is one issue's habits and can invert the "
              f"train majority, which made a correct model score below a constant. Watch these numbers.")
        print(f"  swe-action: prompt size p50 {sw_stats.get('prompt_chars_p50', 0)} chars, "
              f"max {sw_stats.get('prompt_chars_max', 0)} (~{sw_stats.get('approx_tokens_max', 0)} tokens). "
              f"Training attention memory is QUADRATIC in this -- an uncapped issue body once drove a real "
              f"run to ~60GB VRAM, so --swe-max-issue-chars caps it.")
        print(f"  swe-action: held-out gold tools {sw_stats['held_gold_tools']} -- MAJORITY-CLASS RATE "
              f"{sw_stats['held_majority_rate']:.2f}. An exact-match score at or below this is a base-rate "
              f"artifact, NOT evidence the model predicts actions.")
    elif task_domain == "math-cot":
        pre_raw_rows = None
        if math_cot_docs_path:
            import json as _json
            with open(math_cot_docs_path, encoding="utf-8") as _f:
                pre_raw_rows = [_json.loads(line) for line in _f if line.strip()]
            print(f"  math-cot: using {len(pre_raw_rows)} pre-fetched raw rows from {math_cot_docs_path} "
                  f"(avoids the real datasets/torch ordering crash)")
        train_tasks, held_tasks, related_desc_pool, mc_stats = _math_cot_tasks_from_graph(
            g, n_train=n_train, n_held=n_held, n_raw=math_cot_n_raw, raw_rows=pre_raw_rows)
        atom_names = related_desc_pool
        print(f"  math-cot: real OpenThoughts math problems, numeric-boxed-answer only -> "
              f"{mc_stats['n_train']} train, {mc_stats['n_held']} held-out (seen={mc_stats['seen']} "
              f"skipped_domain={mc_stats['skipped_domain']} skipped_short={mc_stats['skipped_short']} "
              f"skipped_non_numeric={mc_stats['skipped_no_numeric_answer']} -- most real CoT problems are "
              f"proofs/inequalities/multiple-choice, not clean numeric answers, filtered honestly not "
              f"force-fit)")
    elif graph_path:
        train_tasks, held_tasks = _compose_tasks_from_graph(g, atom_names, n_train=n_train, n_held=n_held)
    else:
        train_tasks, held_tasks = _compose_tasks_real(n_train=n_train, n_held=n_held)
    all_tasks = train_tasks + held_tasks
    split = len(train_tasks)
    if task_domain == "swe-action":
        print(f"  tasks: {split} train, {len(all_tasks) - split} held-out (real SWE next-action, "
              f"trajectory-level split)\n")
    elif task_domain == "math-cot":
        print(f"  tasks: {split} train, {len(all_tasks) - split} held-out (real math CoT, no train/held overlap)\n")
    else:
        if (len(all_tasks) - split) < n_held:
            print(f"  !! n_held CLAMPED: asked for {n_held}, got {len(all_tasks) - split}. The generator "
                  f"makes a FIXED pool of 2-atom pairs and n_train takes from it first, so raising "
                  f"--n-held alone cannot widen the held set -- LOWER --n-train to free pairs. A "
                  f"silently clamped held set makes a confidence interval look better-supported than "
                  f"the data allows.", flush=True)
        print(f"  tasks: {split} train, {len(all_tasks) - split} held-out (2-atom composition, auto-generated, "
              f"no train/held PAIR overlap)\n")

    # PER-STEP CONTRASTIVE OBJECTIVE. contrast_weight 0 (default) leaves every line below inert, so
    # existing runs reproduce exactly. See SlotContrast's docstring for why this term exists: the
    # only per-step force on y_t was conv_loss, which rewards NOT changing, and ds_weight defaults
    # to 0, so collapse was the optimum rather than a training failure.
    # PROJ_Y BIAS. proj_y maps y_t -> the slots the LM actually attends, and its bias is a CONSTANT
    # added to every slot of every task -- an instance-independent component built into the
    # architecture, which is the adapter shortcut expressed as a parameter. Measured at init on this
    # exact module: ||bias|| / ||W@y|| = 1.280, y_t across-task cosine 0.8889 -> slots 0.9576. The
    # state tracker's write projection is bias=False on purpose ("a bias 2.09x the signal collapsed
    # prefix separation to 0.9170"), and that is the module where the same contrastive objective
    # DID work. Zeroing + freezing rather than changing the layer shape keeps every existing
    # checkpoint loadable.
    # TRM BIASES. **NOT the root cause** -- see the RETRACTION below. A real artifact, but it only
    # moves where training STARTS; it does not stop the collapse.
    #
    # RETRACTION (2026-08-05, commit 644343c overclaimed). That commit called this THE ROOT CAUSE on
    # the strength of an INITIALIZATION measurement, and then the run refuted it:
    #     contrast 0.5 + no-proj-bias + trm-no-bias  ->  slot_cos 1.000000,
    #     y_t[1.000 1.000 1.000 1.000], instance-specific +0.0017 CI [-0.0005,+0.0042], ratio 4226x
    # y_t is already ~1.000 at the EPOCH-0 checkpoint despite starting at 0.364, i.e. the collapse
    # happens within a single epoch of TRAINING. Init separation was a proxy for trained separation
    # and the two are different quantities -- the same proxy-for-condition error that also produced
    # "process is dead" from a log that had stopped growing, and "GPU is free" from a completion
    # marker that prints before the process exits. Three times in one session, same shape.
    #
    # What remains TRUE and worth keeping: the bias/signal ratios below are real and large, and they
    # do move initialization 0.885 -> 0.364 across 3 seeds. What is FALSE: that fixing them fixes
    # the collapse. The driver is in the TRAINING objective, not the initialization -- and note that
    # conv_weight (0.05, ALWAYS ON) was active in every arm run so far, including this one.
    # Measured on this exact module across 3 seeds:
    #     shipped                y_t across-instance cosine 0.8911 0.8964 0.8680  (mean 0.8851)
    #     input projections only 0.8307 0.8548 0.8383                             (mean 0.8413)
    #     ALL biases zeroed      0.3270 0.4044 0.3619                             (mean 0.3644)
    # Every nn.Linear here uses PyTorch's default bias init, which is sized for inputs with unit
    # VARIANCE PER DIMENSION. This model runs on L2-normalized MiniLM embeddings (each of 384 dims
    # ~1/sqrt(384) ~ 0.05) and small internal activations, so at every stage the bias is comparable
    # to or LARGER than the instance-varying signal:
    #     task_proj / atom_proj   ||b||/||Wx|| = 0.993 / 0.991  -> cos 0.001 becomes 0.499
    #     proj_y(y0) at cycle 0   ||bias|| 0.5644 vs ||x|| 0.4683 -> pre-activation cos 0.578
    #     y_head                  0.60 -> 0.836
    # Each bias is a per-instance CONSTANT, i.e. the adapter shortcut compiled into the architecture
    # at seven separate places. For calibration: algo_grr_state.py's z starts at 0.7643 and its
    # contrastive run ESCAPED collapse; a bias-free TRM starts at 0.36, well inside that regime.
    # Zeroed and frozen after construction so algo_trm.py is untouched and every other consumer and
    # checkpoint is unaffected.
    # ATOM-INITIALIZED SLOTS: the design in this file's header, finally built. See
    # WMReasoner._refine_atom_slots for the measurements that show what ran instead.
    if slotdim_k > 0:
        if slotdim_heads > 0:
            R.slotdim = SlotDIMv2(R.trm.d, k=slotdim_k, n_heads=slotdim_heads,
                                  ffn=not slotdim_no_ffn, depth=slotdim_depth).to(wb.device)
        else:
            R.slotdim = SlotDIM(R.trm.d, k=slotdim_k, depth=slotdim_depth,
                                ffn=not slotdim_no_ffn).to(wb.device)
        print(f"  SLOTDIM {'v2' if slotdim_heads > 0 else 'v1'} ON: {slotdim_k} banks, "
              f"{slotdim_heads if slotdim_heads > 0 else 1} head(s), depth {slotdim_depth}, "
              f"FFN {'OFF (ablated)' if slotdim_no_ffn else 'ON'}"
              f"{'  [multi-head slot sets + selective decay + ABSOLUTE tanh(obs) gate]' if slotdim_heads > 0 else ''}"
              f" -- replacing the TRM recursion, which measures as a contraction "
              f"(two distinct seeds -> cos 0.8072 after ONE step).", flush=True)
        print(f"     slot producer class: {type(R.slotdim).__name__}", flush=True)

    if atom_slots:
        R.atom_slots = True
        print(f"  ATOM-INITIALIZED SLOTS ON: the slot table is now K vectors, one SEEDED FROM EACH "
              f"retrieved atom and refined over {R.T} steps -- not T values read off a shared "
              f"zero-init free latent. Slot count now tracks K; retrieval rank is preserved.",
              flush=True)

    if trm_no_bias:
        _nz = 0
        with torch.no_grad():
            for _mod in R.trm.modules():
                if isinstance(_mod, nn.Linear) and _mod.bias is not None:
                    _mod.bias.zero_(); _mod.bias.requires_grad_(False); _nz += 1
            for _p in (getattr(R.trm.cross_attn, "in_proj_bias", None),
                       getattr(R.trm.cross_attn.out_proj, "bias", None)):
                if _p is not None:
                    _p.zero_(); _p.requires_grad_(False); _nz += 1
        print(f"  TRM biases ZEROED and FROZEN ({_nz} tensors): measured to take y_t across-instance "
              f"cosine 0.885 -> 0.364 at init, across 3 seeds.", flush=True)

    if no_proj_bias:
        with torch.no_grad():
            R.proj_y.bias.zero_()
        R.proj_y.bias.requires_grad_(False)
        print(f"  proj_y bias ZEROED and FROZEN: the slots the LM sees no longer carry a constant "
              f"offset (measured 1.28x the instance-varying signal at init).", flush=True)

    contrast_head = (SlotContrast(R.trm.d).to(wb.device) if contrast_weight > 0 else None)
    contrast_rng = random.Random(0)
    tgt_emb_of: dict = {}
    if contrast_head is not None:
        _tg_texts = {t[0]: t[2] for t in (train_tasks + held_tasks)}     # task text -> target code
        _keys = list(_tg_texts)
        _vecs = encode_batch([_tg_texts[k] for k in _keys])
        for _k, _v in zip(_keys, _vecs):
            tgt_emb_of[_k] = torch.as_tensor(_v, dtype=torch.float32, device=wb.device)
        # Negatives come from TRAIN targets only -- drawing them from held would leak the held set
        # into the training signal, which is the same instance-split discipline this file applies
        # everywhere else.
        _train_texts = [t[0] for t in train_tasks]
        neg_bank = torch.stack([tgt_emb_of[t] for t in _train_texts])
        neg_row_of = {t: i for i, t in enumerate(_train_texts)}
        print(f"  per-step CONTRASTIVE objective ON (weight {contrast_weight}, {contrast_negs} "
              f"negatives, applied at ALL {R.T} recursion steps). Negatives drawn from "
              f"{neg_bank.shape[0]} TRAIN targets only.\n", flush=True)
    else:
        neg_bank = None
        neg_row_of = {}

    gate_params = [a.g for a in R.adapters]
    gate_ids = {id(p) for p in gate_params}
    other_params = [p for p in R.parameters() if p.requires_grad and id(p) not in gate_ids]
    _groups = [
        {"params": other_params, "weight_decay": 1e-4},
        {"params": gate_params, "weight_decay": 5e-2},
    ]
    if contrast_head is not None:
        _groups.append({"params": list(contrast_head.parameters()), "weight_decay": 1e-4})
    # NOTE: no separate group for R.slotdim. Assigning it to R registers it as a SUBMODULE, so its
    # parameters are already inside R.parameters() and therefore already in other_params above.
    # Adding them again raised "some parameters appear in more than one parameter group".
    opt = torch.optim.Adam(_groups, lr=1e-3)
    # GATE WARM-START. 0.8 (tanh~0.66) means the adapter injects strongly from step 0, through projections
    # that are still RANDOMLY initialized. That was harmless on synthetic composition, where the ablated
    # baseline is 0/16 by construction (the base LM cannot name graph atoms it was never shown), so noise
    # could only ever be upside. It is actively destructive on any domain where the base model ALREADY has
    # real ability: measured on swe-action, held WM 0/16 against ablated 3/16 -- working memory strictly
    # WORSE than no working memory, at every checkpoint, while the gate climbed 0.79 -> 0.99 and lm loss
    # fell 2.43 -> 0.76 (the adapter channel fitting the teacher-forced target without generalizing).
    # It also contradicts the zero-init discipline every other added component here follows
    # (top_to_bottom_proj, critic_ctx, top_event_emb all start as exact no-ops and must EARN their
    # contribution). Default stays 0.8 so existing results reproduce; pass a small value (e.g. 0.05) on
    # domains with a non-zero ablated baseline so the adapter has to earn its way in.
    for a in R.adapters:
        with torch.no_grad():
            a.g.fill_(gate_init)

    def build_prompt(task_text, inner_name=None, outer_name=None):
        if task_domain == "swe-action":
            return wb.tok(f"You are fixing a software issue. Given the issue and the work done so far, "
                          f"name the single next tool to call.\n\n{task_text}\n\nNext tool:",
                          return_tensors="pt").input_ids.to(wb.device)
        if task_domain == "math-cot":
            # no inner/outer atoms, no "write function task(n)" -- this is a real free-text problem
            return wb.tok(f"Solve the following problem. Show your reasoning, then give the final answer "
                         f"as \\boxed{{answer}}.\n\nProblem: {task_text}\n\nSolution:\n",
                         return_tensors="pt").input_ids.to(wb.device)
        if inner_name and outer_name:
            hint = f"# return: {outer_name}({inner_name}(n))\n"
        else:
            hint = ""
        if code_prompt:
            # The shipped prompt ENDS in "Explanation:", i.e. it asks for PROSE, while the target is
            # a bare " return {expr}" fragment. So the ablated arm scores 0 by OBEYING the
            # instruction it was given, and the adapter's entire measured gain is "ignore
            # Explanation: and emit code" -- format compliance, not memory. Ending at "return "
            # makes the continuation BE the target, so ablated finally measures capability.
            return wb.tok(f"{hint}# {task_text}\ndef task(n):\n    return ",
                          return_tensors="pt").input_ids.to(wb.device)
        return wb.tok(f"{hint}Explain your reasoning. Then write function task(n):\n# {task_text}\nExplanation:\n",
                      return_tensors="pt").input_ids.to(wb.device)

    # Precompute static task embeddings and atom embeddings for all examples
    print("  Precomputing task + atom embeddings...", flush=True)
    task_embs = {}
    for text, atoms_needed, _, _ in all_tasks:
        if text not in task_embs:
            task_embs[text] = torch.as_tensor(encode_batch([text])[0], dtype=torch.float32, device=wb.device)

    prompt_ids = {text: build_prompt(text) for text, _, _, _ in all_tasks}
    # All 10 atom oracle functions (used for verification)
    def _fib(n):
        a, b = 0, 1
        for _ in range(n):
            a, b = b, a + b
        return a

    if task_domain in ("math-cot", "swe-action"):
        # no execution oracle for free text -- verify() is rebound below (math-cot -> _answers_match,
        # swe-action -> _swe_action_match against the trajectory's real recorded next tool).
        _run_task, _oracle_ns = None, {}
    elif graph_path:
        # DYNAMIC oracle, sourced from the graph's OWN atom code (Phase 3) -- scales to whatever atoms
        # actually exist, instead of the fixed 10-lambda dict below.
        _run_task, _oracle_ns = _dynamic_oracle(g, atom_names)
    else:
        _oracle_ns = {
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

        def _run_task(n, code_line):
            """Execute the composition code_line (e.g. 'digit_sum(fibonacci(n))') at n.

            CRITICAL FIX: eval's namespace never included 'n' itself -- every composition expression
            references n directly, so this raised NameError on EVERY call, silently caught by verify()'s
            except-> False. This meant verify() could never return True for ANY input, correct or not,
            since _run_task was written -- the true root cause under the 0/4 and 0/16 results, deeper than
            the decoding-loop issue."""
            return eval(code_line, {"__builtins__": __builtins__}, {**_oracle_ns, "n": n})

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

    def verify(code_str, tests):
        """Execute the generated code and check against provided test cases.

        BUG FOUND while investigating "held WM stuck at 0" (real, decisive, not a guess): this used to
        exec() the raw generated text directly, requiring it to already be a complete `def task(n): ...`
        statement. But the training target (return_body, above) only ever teaches the model to produce a
        bare ` return {expr}` fragment -- no `def`, no wrapper -- because build_prompt's prompt no longer
        ends in `...def task(n):` (it ends in `Explanation:\n`) the way the OLD, proven prompt did. Verified
        directly: exec'ing the LITERAL, VERBATIM training target (' return num_divisors(square(n))') raised
        IndentationError, 100% of the time, regardless of composition correctness -- verify() could
        structurally never return True no matter how well the reasoner trained. _extract_first_return
        (below) was clearly written to bridge exactly this gap and already handles messy raw output
        (explanation prose before it, decode-loops after it) -- it was just never wired back in after the
        V3 rewrite. Reconnected: extract the expression, reconstruct a real `def task(n): return {expr}`,
        THEN exec that."""
        expr = _extract_first_return(code_str)
        if expr is None:
            return False
        try:
            # SECOND bug, compounding the first: an empty ns means num_divisors/square/etc. (the atoms the
            # composition actually calls) are undefined -- task(n) would raise NameError the instant it's
            # called, for EVERY composition, correct or not. _oracle_ns (built once above, from the graph's
            # own atom code via _dynamic_oracle/_closure, or the hardcoded fn_map) already has every atom
            # callable -- seed the exec namespace from it instead of starting empty.
            ns = dict(_oracle_ns)
            exec(f"def task(n):\n    return {expr}\n", ns)
            fn = ns.get("task")
            if not callable(fn): return False
            for inp, expected in tests:
                if fn(inp) != expected: return False
            return True
        except Exception: return False

    if task_domain == "math-cot":
        # No execution oracle exists for free-text math -- verify against the dataset's own boxed final
        # answer instead (_answers_match, same anti-poison principle: gold is the dataset's stated answer,
        # extracted once at task-build time, never the model's own guess).
        def verify(code_str, tests):
            return _answers_match(code_str, tests[0][1]) if tests else False

    if task_domain == "swe-action":
        # Gold is the tool the real trajectory ACTUALLY called next -- a recorded historical fact read out
        # of the dataset at task-build time, never a model judgement. Same anti-poison rule as every other
        # verifier in this file; see _swe_action_match for why a bare substring test is not enough.
        def verify(code_str, tests):
            return _swe_action_match(code_str, tests[0][1]) if tests else False

    _MAX_SAFE_MAGNITUDE = 100_000

    def make_tests(code_expression, atoms_needed):
        """A composed outer(inner(n)) can blow up combinatorially even though every atom involved is
        individually fast and correctly int-typed: factorial(13) = 6,227,020,800, and num_divisors' naive
        trial division (sum(1 for i in range(1,abs(n)+1) if n%i==0), one of the ORIGINAL 10 seed atoms) is
        O(n) -- num_divisors(factorial(13)) alone measured ~9 minutes on this machine. Not a hang, just a
        combinatorially expensive eval with zero visible progress. Cheap pre-check: evaluate the INNER
        atom alone first (already verified fast/int-typed by _atoms_from_graph) and skip this n if its
        result is too large for a single-pass counting atom to handle quickly -- avoids ever running the
        catastrophic outer(inner(n)) eval. Direct (single-atom) tasks have no inner stage, so no risk."""
        tests = []
        for n in [2, 3, 5, 7, 10, 13]:
            if len(atoms_needed) == 2:
                try:
                    inner_val = _oracle_ns[atoms_needed[0]](n)
                except Exception:
                    continue
                if isinstance(inner_val, int) and abs(inner_val) > _MAX_SAFE_MAGNITUDE:
                    continue
            tests.append((n, _run_task(n, code_expression)))
        return tests

    if task_domain in ("math-cot", "swe-action"):
        # code_expr IS the gold already (math-cot: the gold float; swe-action: the gold tool name) -- no
        # oracle to re-derive it from and no n-sweep, just one real task with one real recorded answer.
        def make_tests(code_expression, atoms_needed):
            return [(None, code_expression)]

    # text -> oracle expression, so deep supervision can build the intermediate-result targets. The
    # 7-tuple in train_ex intentionally does not carry code_expr, and `tests` holds numeric (input,
    # expected) pairs rather than the expression, so the mapping is kept here rather than reconstructed.
    code_expr_of = {t[0]: t[3] for t in (train_tasks + held_tasks)}
    _ds_cache: dict = {}

    train_ex = [(task_embs[text], [atom_names.index(a) for a in atoms_needed], atoms_needed,
                 prompt_ids[text], text, code, make_tests(code_expr, atoms_needed))
                for text, atoms_needed, code, code_expr in train_tasks]
    held_ex = [(task_embs[text], [atom_names.index(a) for a in atoms_needed], atoms_needed,
                prompt_ids[text], text, code, make_tests(code_expr, atoms_needed))
               for text, atoms_needed, code, code_expr in held_tasks]

    # ============================================================================================
    # DERANGEMENT FALSIFIER -- the control this experiment has never had.
    #
    # `ablated` sets _slots=None, so _mk_hook returns early and the adapter is bypassed entirely -- it is
    # the BASE LM, not a gate-zeroed adapter (the older "gate ablated" wording in this file is loose).
    # It therefore answers exactly one question: does the adapter do ANYTHING.
    # It cannot separate "the LM reads THESE slots" from "the presence of any slots unlocks a mode in
    # which atom names get emitted at all" -- and on the synthetic domain the ablated baseline is 0/16
    # BY CONSTRUCTION (build_prompt is called without inner/outer, so hint="" and the base LM has never
    # been shown the atom names). A control that cannot fail is not evidence, which is why the whole
    # 13-15/16 headline is undecided. Corroborating: the across-task slot cosine added later in this
    # same loop caught a checkpoint scoring 15/16 sitting at slot_cos 1.000000, and on swe-action --
    # the one domain with a non-zero ablated baseline -- held WM measured 0/16 against ablated 3/16.
    #
    # The deranged arm hands held task i the working memory refined for held task perm(i). Same prompt,
    # same tests, same gate, same greedy decode; ONLY the content of working memory is wrong.
    #     WM - deranged      = instance-specific signal  (does it matter WHICH atoms are in memory?)
    #     deranged - ablated = format/mode effect        (does the mere PRESENCE of slots matter?)
    #
    # Random derangement, never argmin: pairing each item with its most anti-correlated partner
    # manufactured an entire K-trend in algo_grr_prefix (algo_grr_contrast.py, bug #4). held_ex is never
    # shuffled, so one fixed permutation stays consistent across every eval checkpoint.
    # ============================================================================================
    derange_perms: list = []
    if derange and len(held_ex) >= 2:
        _drng = random.Random(4242)
        _nh = len(held_ex)
        for _k in range(max(1, n_derange)):
            _p = list(range(_nh))
            for _ in range(1000):
                _drng.shuffle(_p)
                if all(_p[i] != i for i in range(_nh)):
                    break
            else:
                # Cyclic shift by a NON-ZERO offset mod _nh. A plain (i+1+_k) would wrap to offset 0
                # at _k == _nh-1 and silently reintroduce fixed points -- the one thing this arm may
                # never have.
                _off = 1 + (_k % max(1, _nh - 1))
                _p = [(i + _off) % _nh for i in range(_nh)]
            derange_perms.append(_p)
        print(f"  derangement falsifier ON: {_nh} held tasks, {len(derange_perms)} independent "
              f"derangements (no fixed points). GENERATION uses derangement 0 (1 extra generate() per "
              f"held task per checkpoint); the CE readout uses all {len(derange_perms)} (forward passes "
              f"only). Changes no existing arm.")

    print(f"  Training the adapter + WMReasoner ({epochs} epochs, {len(train_ex)} pairs; "
          f"real-close loop: refine -> LM -> verify)...")
    if batch_size > 1:
        print(f"  Batched training: batch_size={batch_size} (pads variable-length sequences, "
              f"~{batch_size}x fewer optimizer steps)")
    best_held = 0.0
    # eval_every was epochs//8, so an 8-epoch run evaluated EVERY epoch. At held=32 that is 96
    # generate() calls per checkpoint and the eval, not the training, dominates wall clock
    # (~25 min/epoch measured). The CE readout is the quotable metric and is forward-only; the
    # generation pass counts are the expensive AND noisy one. Make the cadence explicit.
    eval_every = eval_every_arg if eval_every_arg > 0 else max(1, epochs // 8)
    last_dump = None
    # Pre-bound so the final report below is readable even if no eval checkpoint ever ran (epochs=0).
    held_ok, ablated_ok, deranged_ok, der_dump, slot_cos_val = 0, 0, 0, [], None
    ce_wm, ce_abl, ce_der = [], [], []
    pad_id = wb.tok.pad_token_id or 0
    # SELF-CRITIQUE data collection (co-evolutionary arms-race idea, stage 1): every eval checkpoint runs
    # real generate()+verify() on held_ex (free byproduct) AND, now, ALSO on train_ex (a small deliberate
    # extra generation cost, see below) -- ~4x more real (trajectory, real-verify-label) pairs than
    # held_ex alone, addressing the critic's small-sample-size problem directly. Still PASSIVE: this data
    # trains the critic only, after the main loop -- does not touch the reasoner's own loss (that's stage 2,
    # gated on stage 1's critic actually beating base rate first, to avoid training the reasoner against an
    # unreliable judge -- Goodhart's law / reward-hacking risk, not built yet).
    critic_examples: list = []
    # Real graph writes on real verified outcomes. Also supplies the description<->name mapping the text
    # domains need for record_success/record_failure to do anything at all (previously a silent no-op).
    growth = _PassiveGrowth(g, enabled=passive_growth) if graph_path else None
    if growth is not None and passive_growth:
        print(f"  passive growth ON: verified generations are banked as REAL graph nodes mid-training "
              f"(failures -> trap nodes, capped at {growth.trap_budget}/epoch to avoid the trap-flooding "
              f"that required a graph repair here before).")
    # Train the recurrent regime whenever it is the one we will evaluate in. See the rollout below.
    recurrent_train = merged
    if recurrent_train:
        print(f"  merged: training is a 2-chunk RECURRENT rollout (resume_state + memory + progress-"
              f"shifted task embedding), matching what generate_with_reground does at eval. Previously "
              f"recurrent_refine was never called under a gradient at all -- the weights were fitted "
              f"one-shot and deployed recurrent. Costs ~2 LM forwards per example.")
    heartbeat_every = max(1, (len(train_ex) // batch_size) // 5)   # ~5 pings per epoch, regardless of size
    for ep in range(epochs):
        if growth is not None:
            growth.epoch_reset()
        R.train()
        random.shuffle(train_ex)
        tot_lm, n = 0.0, 0
        tot_gate_reg, tot_conv, tot_ds, tot_ponder = 0.0, 0.0, 0.0, 0.0
        tot_ct = 0.0
        for b0 in range(0, len(train_ex), batch_size):
            if ep == 0 and n % heartbeat_every == 0:
                # epoch 0 alone can run for many minutes on a real 4B model with batch_size=1 (100 forward+
                # backward passes, unbatched) with ZERO prints anywhere below -- looked hung on a real cloud
                # run (20 min, no output). This is the only signal that anything is happening before the
                # first per-epoch summary line even exists.
                print(f"    [ep 0 heartbeat] training example {n}/{len(train_ex)}...", flush=True)
            batch = train_ex[b0:b0 + batch_size]
            all_states, pids_list, tids_list, slots_list = [], [], [], []
            # Second-chunk lists, used only by the recurrent rollout below. Kept separate (rather than
            # appended to the chunk-1 lists) because an example whose target is a single token has no
            # second half and simply does not contribute a chunk 2 -- the two batches can differ in size.
            pids2_list, tids2_list, slots2_list = [], [], []
            ds_tgt_list = []
            n_steps_list = []
            for task_emb, gold_idxs, atoms_needed, pids, text, target_code, tests in batch:
                mini_atom_embs = torch.stack([
                    torch.as_tensor(encode_batch([atom_names[idx]])[0], dtype=torch.float32, device=wb.device)
                    for idx in gold_idxs
                ])
                pids = build_prompt(text)
                return_body = target_code.split(": ", 1)[1] if ": " in target_code else target_code
                tids = wb.tok(" " + return_body, return_tensors="pt").input_ids.to(wb.device)
                eos = torch.tensor([[wb.tok.eos_token_id]], device=wb.device)
                tids = torch.cat([tids, eos], dim=-1)

                if recurrent_train:
                    # RECURRENT (merged) TRAINING: a two-chunk teacher-forced rollout, so the weights are
                    # trained in the SAME regime eval runs them in.
                    #
                    # This closes a real train/test mismatch. recurrent_refine's cross-call state
                    # (resume_state) and its memory-concatenated cross-attention context existed only
                    # inside generate_with_reground, which is called under torch.no_grad() at eval time.
                    # No training call site ever touched it, so every parameter it uses was fitted under
                    # one-shot refine() -- fresh z0/y0, graph atoms only, static task embedding -- and then
                    # deployed on chunk 2+ with a resumed latent, a memory bank appended to its context and
                    # a progress-shifted task embedding it had never seen once.
                    #
                    # Chunk 1: fresh state, predict the first half of the target from the prompt.
                    # Chunk 2: resume chunk 1's (z, y), attend the first half as memory, shift the task
                    #          embedding by it (mirroring reground_bottom=True), predict the second half
                    #          with the first half in context.
                    # Cost is ~2 LM forwards per example; only --merged runs pay it.
                    h = tids.shape[-1] // 2
                    s1, st1, state1 = R.recurrent_refine(task_emb, mini_atom_embs)
                    if R._n_steps is not None:
                        n_steps_list.append(R._n_steps)
                    if h >= 1 and tids.shape[-1] - h >= 1:
                        tids_a, tids_b = tids[:, :h], tids[:, h:]
                        prog_text = wb.tok.decode(tids_a[0], skip_special_tokens=True)
                        prog = torch.as_tensor(encode_batch([prog_text])[0], dtype=torch.float32,
                                               device=wb.device)
                        s2, st2, _state2 = R.recurrent_refine(
                            task_emb + prog, mini_atom_embs, memory=prog.unsqueeze(0),
                            resume_state=state1)
                        if R._n_steps is not None:
                            n_steps_list.append(R._n_steps)
                        slots_list.append(s1); pids_list.append(pids); tids_list.append(tids_a)
                        slots2_list.append(s2)
                        pids2_list.append(torch.cat([pids, tids_a], dim=-1)); tids2_list.append(tids_b)
                        states = st2                       # chunk 2 holds the full-composition target
                    else:                                  # 1-token target: no second half to roll out
                        slots_list.append(s1); pids_list.append(pids); tids_list.append(tids)
                        states = st1
                else:
                    # hierarchical_refine (not plain refine): when R.top_trm is None this is byte-identical to
                    # refine() (confirmed by the zero-init no-op test); when top_trm is set, this is the ONLY
                    # call site in the whole training loop that exercises top_trm/top_to_bottom_proj, so it's
                    # also the only place they can ever receive a gradient. Previously only generate_with_reground
                    # (eval-time, no .backward() anywhere near it) touched hierarchical_refine -- top_to_bottom_proj
                    # was mathematically guaranteed to sit at zero-init forever, regardless of epoch count (a real,
                    # confirmed-by-reading-the-code gap, not a "needs more epochs" issue).
                    slots, states, _top_state, _top_resume = R.hierarchical_refine(task_emb, mini_atom_embs)
                    if R._n_steps is not None:
                        n_steps_list.append(R._n_steps)
                    # DEFER injection: with batch_size>1, calling set_slots_direct here would be overwritten by
                    # every subsequent example, so only the LAST example's slots would survive to the single
                    # batched forward pass below -- GatedCrossAttn then broadcasts that one example's slots to
                    # the WHOLE batch (its slots.dim()==2 branch), silently corrupting every other example's
                    # gradient (real/wrong content, right target). Stack per-example slots into [B,T,d_lm] and
                    # inject ONCE, after the loop, so each batch row attends to its OWN slots.
                    slots_list.append(slots)
                    pids_list.append(pids); tids_list.append(tids)
                all_states.append(states)
                if ds_weight > 0:
                    ds_tgt_list.append(_ds_targets_for_task(
                        wb, code_expr_of.get(text), atoms_needed, R.T, _ds_cache))

            def _chunk_loss(p_list, t_list, s_list):
                """One batched (or single) teacher-forced pass with per-example slots injected."""
                R.set_slots_direct(torch.stack(s_list, dim=0))   # [B, T, d_lm] -- per-example, always
                if len(p_list) > 1:
                    input_ids, labels, attn_mask = _pad_and_batch(p_list, t_list, pad_id, wb.device)
                    return wb.model(input_ids=input_ids, attention_mask=attn_mask, labels=labels).loss
                return wb.model(input_ids=torch.cat([p_list[0], t_list[0]], dim=-1),
                                labels=torch.cat([torch.full_like(p_list[0], -100), t_list[0]],
                                                 dim=-1)).loss

            lm_loss = _chunk_loss(pids_list, tids_list, slots_list)
            if pids2_list:
                lm_loss = lm_loss + _chunk_loss(pids2_list, tids2_list, slots2_list)

            # Gate regularization: penalize |tanh(g)| — prevents adapter overwriting hidden states
            gate_reg = sum(torch.tanh(a.g) ** 2 for a in R.adapters) / len(R.adapters)

            # Convergence bonus: penalize late-step changes in y_t trajectory
            # Quadratic weight: early steps free, late steps must converge toward fixed point
            states_tensor = torch.stack([torch.stack(s) for s in all_states], dim=0)  # [B, T, d]
            T = states_tensor.shape[1]
            step_diffs = states_tensor[:, 1:] - states_tensor[:, :-1]  # [B, T-1, d]
            w = torch.linspace(0.0, 1.0, T - 1, device=states_tensor.device) ** 2  # quadratic, [0, 1]
            conv_loss = (w.unsqueeze(0).unsqueeze(-1) * step_diffs.norm(dim=-1, keepdim=True)).mean()

            # DEEP SUPERVISION -- the only term that gives the trm a DIRECT, task-specific gradient on
            # its own latents. Without it the loss touching y_t was conv_loss alone, which penalizes
            # ||y_t+1 - y_t|| and therefore literally rewards NOT CHANGING; measured consequence on a real
            # trained checkpoint: slots had across-task cosine 1.000000 (min 0.999999) while their task_emb
            # INPUTS sat at 0.4510, i.e. the working memory was a constant and a slot-swap between
            # different tasks changed the generation on only 1 of 16 held tasks. ds_loss_batch and ds_proj
            # already existed; nothing ever called them from this loop, and ds_loss_batch returns exactly
            # 0.0 when targets is None, so even a stray call was a silent no-op.
            ds_loss = torch.tensor(0.0, device=wb.device)
            if ds_weight > 0 and ds_tgt_list and all(t is not None for t in ds_tgt_list):
                ds_loss = R.ds_loss_batch(all_states, targets=torch.stack(ds_tgt_list).to(wb.device))

            # PER-STEP CONTRASTIVE. Deliberately applied to states_tensor, the SAME [B, T, d] tensor
            # conv_loss penalizes change on, so the two terms act on exactly the same quantity and
            # their tug-of-war is visible in the printed per-epoch numbers rather than inferred.
            ct_loss = torch.tensor(0.0, device=wb.device)
            if contrast_head is not None and neg_bank is not None:
                _tb = [tgt_emb_of.get(bx[4]) for bx in batch]
                if all(v is not None for v in _tb):
                    # each example's OWN row in neg_bank, so it is never drawn as its own negative
                    _rows = [neg_row_of.get(bx[4]) for bx in batch]
                    _cl, _ncand = slot_contrast_loss(
                        contrast_head, states_tensor, torch.stack(_tb), neg_bank,
                        contrast_rng, n_neg=contrast_negs, exclude_rows=_rows)
                    if _cl is not None:
                        ct_loss = _cl
            ponder_loss = torch.tensor(0.0, device=wb.device)
            if ponder_weight > 0 and n_steps_list:
                ponder_loss = torch.stack(n_steps_list).mean()
            loss = (lm_loss + gate_reg_weight * gate_reg + conv_weight * conv_loss
                    + ds_weight * ds_loss + ponder_weight * ponder_loss
                    + contrast_weight * ct_loss)
            opt.zero_grad()
            loss.backward()
            opt.step()
            # HARD CEILING on the gate. The warm-start alone is not enough: the gate is a LEARNED
            # parameter and lm_loss actively pushes it UP, because injecting harder fits the teacher-forced
            # target better even when it destroys held-out behaviour. The 0.05 * gate_reg penalty loses
            # that tug-of-war. Measured on a real 40-epoch swe-action run started at gate_init 0.05:
            #     ep 0  gate +0.09 -> held WM 3/16 (== ablated 3/16, the no-op baseline)
            #     ep 5  gate +0.24 -> held WM 4/16 (BEATS ablated)
            #     ep 10 gate +0.35 -> held WM 0/16 (collapsed)
            #     ep 39 gate +0.80 -> held WM 0/16
            # So the usable band is roughly gate 0.1-0.25, and training walks straight out of it. Clamping
            # keeps the adapter inside the regime where it actually helps instead of letting the training
            # objective optimize held-out performance away. 0 disables the clamp.
            if gate_max > 0:
                with torch.no_grad():
                    for a in R.adapters:
                        a.g.clamp_(-gate_max, gate_max)
            tot_lm += float(lm_loss.detach())
            tot_gate_reg += float(gate_reg.detach())
            tot_conv += float(conv_loss.detach())
            tot_ds += float(ds_loss.detach()) if torch.is_tensor(ds_loss) else 0.0
            tot_ponder += float(ponder_loss.detach()) if torch.is_tensor(ponder_loss) else 0.0
            tot_ct += float(ct_loss.detach()) if torch.is_tensor(ct_loss) else 0.0
            n += 1

        ponder_str = f"  ponder {tot_ponder/max(n,1):.2f}" if ponder_weight > 0 else ""
        # ct is printed next to conv on purpose: they act on the SAME [B,T,d] tensor with opposite
        # intent (conv penalizes change, ct requires the step to identify its own task), so their
        # tug-of-war should be readable per epoch instead of reconstructed afterwards.
        ct_str = f"  ct {tot_ct/max(n,1):.4f}" if contrast_weight > 0 else ""
        print(f"  ep {ep:>3}  lm {tot_lm/max(n,1):.3f}  gate_reg {tot_gate_reg/max(n,1):.4f}  "
              f"conv {tot_conv/max(n,1):.4f}  ds {tot_ds/max(n,1):.4f}{ct_str}  "
              f"gate {R.adapters[0].g.detach().item():+.2f}{ponder_str}", flush=True)
        if ep % eval_every == 0 or ep == epochs - 1:
            R.eval()
            held_ok, ablated_ok, reground_ok_count, reground_evicted_ok_count = 0, 0, 0, 0
            deranged_ok, der_dump = 0, []          # falsifier arm; stays 0/[] when --no-derange
            held_yt = []                           # pre-projection latents, see below
            ce_wm, ce_abl, ce_der = [], [], []     # per-example CE, kept per-example so it can bootstrap
            held_slots = []
            dump = []
            if ep == 0:
                print(f"\n    [ep 0 heartbeat] running held-out eval ({len(held_ex)} tasks x 2 generate() calls)...",
                      flush=True)
            for task_emb, gold_idxs, atoms_needed, pids, text, target_code, tests in held_ex:
                # Use MiniLM atoms for TRM
                held_mini_embs = torch.stack([
                    torch.as_tensor(encode_batch([atom_names[idx]])[0], dtype=torch.float32, device=wb.device)
                    for idx in gold_idxs
                ])
                # hierarchical_refine here too (was plain refine()) -- "held WM" is now the one-shot
                # hierarchical injection (identical to before when top_trm is None), so its gap against
                # "reground" below isolates the value of PERIODIC re-grounding specifically.
                slots, wm_states, wm_deltas, wm_raw, _top_state, _top_resume = R.hierarchical_refine(
                    task_emb, held_mini_embs, track_deltas=True)
                held_slots.append(slots.detach())
                # y_t BEFORE proj_y. slot_cos measures proj_y(y_t), so a collapsed slot table cannot
                # distinguish "the TRM emits a constant" from "the TRM varies and the projection
                # compresses it". Those need opposite fixes, and only this pair separates them.
                held_yt.append(torch.stack([q.detach() for q in wm_states]))
                if R._n_steps is not None:
                    held_slots_nsteps = getattr(R, '_eval_nsteps', [])
                    held_slots_nsteps.append(float(R._n_steps.detach()))
                    R._eval_nsteps = held_slots_nsteps
                with torch.no_grad():
                    R.set_slots_direct(slots)
                    out = wb.model.generate(pids, max_new_tokens=eff_max_new_tokens,
                                            do_sample=False, pad_token_id=wb.tok.eos_token_id)
                    code = wb.tok.decode(out[0][pids.shape[-1]:], skip_special_tokens=True).strip()
                wm_ok = verify(code, tests)
                held_ok += int(wm_ok)
                critic_examples.append(([s.detach() for s in wm_raw], wm_ok,
                                        _critic_ctx(wb, task_emb, code)))
                instability = R.trajectory_instability(wm_deltas)
                # REAL graph editing on the REAL verified outcome -- previously this was computed and
                # thrown away every epoch (confirmed by grep: no record_success/record_failure/learn_any
                # anywhere in this function's training/eval loop, only at grow_cot/grow_skills setup and
                # the final g.save()). The graph sat static as a read-only embedding source for the whole
                # run instead of being the long-term memory it's supposed to be. _PassiveGrowth now also
                # supplies the description<->name mapping this used to lack, so the text domains
                # (math-cot, swe-action) get real graph updates instead of the silent no-op that the
                # previous version had to skip around; with --passive-growth it additionally BANKS the
                # verified generation as a new node.
                if graph_path and growth is not None:
                    growth.update(atoms_needed, text, wm_ok, learn_text=code if wm_ok else None)
                R.clear()
                with torch.no_grad():
                    out = wb.model.generate(pids, max_new_tokens=eff_max_new_tokens,
                                            do_sample=False, pad_token_id=wb.tok.eos_token_id)
                    code_abl = wb.tok.decode(out[0][pids.shape[-1]:], skip_special_tokens=True).strip()
                abl_ok = verify(code_abl, tests)
                ablated_ok += int(abl_ok)
                dump.append((text, target_code, code, wm_ok, code_abl, abl_ok, instability))

                # RE-GROUNDING A/B: opt-in via --top-trm-t (hierarchical) OR --merged, zero cost/behavior
                # change to any run that asks for neither. Reports a SECOND real generation (periodic
                # re-grounding, see generate_with_reground) side-by-side with the static-slots result
                # above, never replacing it -- a comparison, not a swap.
                #
                # `or merged` is a REAL fix, not a widened condition. generate_with_reground is the only
                # caller of recurrent_refine, and it was reachable only when R.top_trm was not None -- so
                # --merged, whose entire purpose is to DELETE the top trm, silently did nothing unless you
                # also passed --top-trm-t to construct the network it removes. Every "merged" result in
                # this codebase was therefore produced by a run that also built a top trm and then bypassed
                # it. Merged now runs on its own.
                if R.top_trm is not None or merged:
                    # This IS the clean baseline -- no eviction here even if --evict-window was passed, so
                    # "reground" always means the same thing across runs. use_kv_cache alone (no eviction)
                    # is validated byte-identical to the no-cache path on real Qwen3-4B, so this call's
                    # RESULT is unaffected by use_kv_cache; only its compute cost changes.
                    reground_text = generate_with_reground(
                        wb, R, pids, task_emb, held_mini_embs,
                        chunk_tokens=reground_chunk_tokens, max_new_tokens=eff_max_new_tokens,
                        top_every=reground_top_every, use_kv_cache=use_kv_cache, evict_window=None,
                        trigger_patterns=trigger_patterns, instability_trigger=instability_trigger,
                        reground_bottom=True, top_no_graph=top_no_graph, top_memory_max=top_memory_max,
                        merged=merged)
                    reground_ok = verify(reground_text, tests)
                    reground_ok_count += int(reground_ok)
                    dump[-1] = dump[-1] + (reground_text, reground_ok)

                    # SEPARATE, real side-by-side comparison -- only when --evict-window was actually
                    # requested. Same task/slots/verify() as reground above; the ONLY difference is
                    # evict_window, so any pass-rate gap between this and reground isolates the real cost
                    # (or lack of one) of eviction specifically, not a confound with use_kv_cache itself.
                    if evict_window is not None:
                        reground_evicted_text = generate_with_reground(
                            wb, R, pids, task_emb, held_mini_embs,
                            chunk_tokens=reground_chunk_tokens, max_new_tokens=eff_max_new_tokens,
                            top_every=reground_top_every, use_kv_cache=True, evict_window=evict_window,
                            trigger_patterns=trigger_patterns, instability_trigger=instability_trigger,
                            sink_tokens=sink_tokens, reground_bottom=True,
                            top_no_graph=top_no_graph, top_memory_max=top_memory_max,
                            evict_to_memory=evict_to_memory, merged=merged)
                        reground_evicted_ok = verify(reground_evicted_text, tests)
                        reground_evicted_ok_count += int(reground_evicted_ok)
                        dump[-1] = dump[-1] + (reground_evicted_text, reground_evicted_ok)

            # DERANGED ARM. Runs as its OWN pass after the loop above, deliberately:
            #   * that loop stays byte-untouched, so held/ablated/reground cannot move;
            #   * held_slots is already fully populated, so perm(i) can index a slot set computed by the
            #     real refiner -- no second hierarchical_refine() call, no chance of a different code path
            #     producing the deranged slots than produced the real ones;
            #   * generation here is greedy (do_sample=False), so it consumes no RNG and cannot perturb
            #     anything downstream.
            # NOT appended to critic_examples and NOT passed to growth.update(): these generations are
            # off-task by construction, so banking them as graph nodes or labelling critic trajectories
            # with them would poison the long-term memory and the critic with deliberate nonsense.
            # Results are kept in their own list rather than appended to `dump`, whose tuples are unpacked
            # POSITIONALLY downstream (d[7]/d[8] reground, d[9]/d[10] evicted).
            if derange_perms and len(held_slots) == len(held_ex):
                for i, hx in enumerate(held_ex):
                    pids_i, tests_i = hx[3], hx[6]
                    with torch.no_grad():
                        R.set_slots_direct(held_slots[derange_perms[0][i]])
                        out = wb.model.generate(pids_i, max_new_tokens=eff_max_new_tokens,
                                                do_sample=False, pad_token_id=wb.tok.eos_token_id)
                        code_der = wb.tok.decode(out[0][pids_i.shape[-1]:],
                                                 skip_special_tokens=True).strip()
                    R.clear()
                    der_ok = verify(code_der, tests_i)
                    deranged_ok += int(der_ok)
                    der_dump.append((code_der, der_ok))

                # ============================================================================
                # CE READOUT -- the metric that has actually resolved things in this project.
                #
                # A 16-item pass rate cannot see the effect sizes at stake here. The whole
                # adapter-shortcut result is stated in nats (instance-specific 0.0011-0.0075 under
                # CE vs 0.3474 under contrastive supervision), and a binomial on N=16 has a
                # confidence interval wider than that entire range. Teacher-forced CE on the SAME
                # three arms is continuous, needs one forward pass instead of a generation, and
                # lands on the same scale as the finding it is testing.
                #
                #   CE(deranged) - CE(WM)      = instance-specific signal  (nats)
                #   CE(ablated)  - CE(deranged) = format/mode effect       (nats)
                #
                # Generation stays at ONE derangement (it is the expensive arm); CE averages over
                # all of them per example, so the deranged estimate is not a single draw.
                # ============================================================================
                def _held_ce(p_i, t_i, slots_or_none) -> float:
                    """Teacher-forced CE of this task's target under a given working memory."""
                    if slots_or_none is None:
                        R.clear()                     # ablated == adapter bypassed, see _mk_hook
                    else:
                        R.set_slots_direct(slots_or_none)
                    with torch.no_grad():
                        o = wb.model(
                            input_ids=torch.cat([p_i, t_i], dim=-1),
                            labels=torch.cat([torch.full_like(p_i, -100), t_i], dim=-1))
                    R.clear()
                    return float(o.loss)

                _eos = torch.tensor([[wb.tok.eos_token_id]], device=wb.device)
                for i, hx in enumerate(held_ex):
                    p_i, tgt_i = hx[3], hx[5]
                    # Target tokenized EXACTLY as the training loop builds tids, so the CE reported
                    # here is the same quantity the model was trained on -- a different tokenization
                    # would make these numbers incomparable to lm_loss without saying so.
                    body = tgt_i.split(": ", 1)[1] if ": " in tgt_i else tgt_i
                    t_i = wb.tok(" " + body, return_tensors="pt").input_ids.to(wb.device)
                    t_i = torch.cat([t_i, _eos], dim=-1)
                    ce_wm.append(_held_ce(p_i, t_i, held_slots[i]))
                    ce_abl.append(_held_ce(p_i, t_i, None))
                    ce_der.append(float(np.mean([
                        _held_ce(p_i, t_i, held_slots[pm[i]]) for pm in derange_perms])))

            # CO-TRAINING data (stage 1)
            # cotrain_samples caps how many train tasks get a real generate() here. This loop was measured
            # as ~43% of total eval-checkpoint cost (a full generate() over EVERY train task, at EVERY
            # checkpoint) and its ONLY consumer is the tier-4 critic -- which has not once beaten its base
            # rate in any real run recorded in this codebase. -1 (default) = all, unchanged behavior;
            # 0 = skip entirely (the critic still gets the held_ex trajectories collected above);
            # N = a random sample of N, re-drawn each checkpoint so it still sees varied tasks over a run.
            cotrain_ex = train_ex
            if cotrain_samples == 0:
                cotrain_ex = []
            elif cotrain_samples > 0 and cotrain_samples < len(train_ex):
                cotrain_ex = random.sample(train_ex, cotrain_samples)
            if ep == 0:
                print(f"    [ep 0 heartbeat] running co-training generate() over {len(cotrain_ex)} train tasks"
                      f"{' (skipped: --cotrain-samples 0)' if not cotrain_ex else ''}...", flush=True)
            for task_emb, gold_idx, atoms_needed, pids, text, target_code, tests in cotrain_ex:
                K_atom_embs = torch.stack([
                    torch.as_tensor(encode_batch([atom_names[idx]])[0], dtype=torch.float32, device=wb.device)
                    for idx in gold_idx
                ])
                slots, tr_states, tr_deltas, tr_raw, _top_state, _top_resume = R.hierarchical_refine(
                    task_emb, K_atom_embs, track_deltas=True)
                with torch.no_grad():
                    R.set_slots_direct(slots)
                    out = wb.model.generate(pids, max_new_tokens=eff_max_new_tokens,
                                            do_sample=False, pad_token_id=wb.tok.eos_token_id)
                    tr_code = wb.tok.decode(out[0][pids.shape[-1]:], skip_special_tokens=True).strip()
                tr_ok = verify(tr_code, tests)
                critic_examples.append(([s.detach() for s in tr_raw], tr_ok,
                                        _critic_ctx(wb, task_emb, tr_code)))   # PRE-norm + REAL label + ctx
                if graph_path and growth is not None:
                    growth.update(atoms_needed, text, tr_ok, learn_text=tr_code if tr_ok else None)
                R.clear()
            R.train()

            best_held = max(best_held, held_ok)
            last_dump = dump
            inst_pass = [d[6] for d in dump if d[3]]
            inst_fail = [d[6] for d in dump if not d[3]]
            inst_str = (f"  instab(pass/fail) {sum(inst_pass)/len(inst_pass):.3f}/"
                       f"{sum(inst_fail)/len(inst_fail):.3f}" if inst_pass and inst_fail else "")
            # ACROSS-TASK SLOT COSINE -- the single number that exposes a constant working memory. If this
            # reads ~1.000 the trm is emitting the same vector for every task and any held-out gain is a
            # constant format/mode effect, not retrieved memory. It went unmeasured for this codebase's
            # whole history; a checkpoint scoring 15/16 turned out to sit at 1.000000.
            slot_cos_str, slot_cos_val = "", None
            if held_slots:
                _S = torch.stack([s_.flatten().float() for s_ in held_slots])
                _S = _S / _S.norm(dim=-1, keepdim=True).clamp_min(1e-12)
                _C = _S @ _S.T
                _off = _C[~torch.eye(len(_S), dtype=torch.bool, device=_C.device)]
                slot_cos_val = float(_off.mean())
                slot_cos_str = f"  slot_cos {slot_cos_val:.4f}"
                # PER-STEP, not pooled. conv_loss is weighted QUADRATICALLY toward later steps, so
                # if it is driving the collapse the damage should grow along the recursion -- a
                # single pooled number cannot show that, and "which step collapses" is exactly the
                # question a per-step objective is meant to answer.
                _ps = per_step_slot_cosine(held_slots)
                if _ps:
                    slot_cos_str += "  per_step[" + " ".join(f"{c:.3f}" for c in _ps) + "]"
                _py = per_step_slot_cosine(held_yt)
                if _py:
                    slot_cos_str += "  y_t[" + " ".join(f"{c:.3f}" for c in _py) + "]"

            # APPARATUS CROSS-CHECK, free, and the reason the two numbers are printed together.
            # Greedy decode + identical slots => bitwise identical output. So the deranged arm and
            # slot_cos are two views of one fact and they MUST agree:
            #   slot_cos ~ 1 (collapsed slots) yet deranged output differs  -> the difference is NOT
            #       coming from the slot path; the arm is not measuring what it claims. Hard flag.
            #   slot_cos well below 1 yet EVERY deranged output is identical -> slots vary but the LM is
            #       invariant to them, i.e. the channel is open and unused. Different finding, same value.
            # This is the class of bug (measurement apparatus, not model) that faked three results on
            # 08-04/08-05; it is cheaper to assert it than to re-derive it from a surprising number.
            deranged_str, der_note = "", ""
            if der_dump:
                deranged_str = f"  deranged {deranged_ok}/{len(held_ex)}"
                n_diff = sum(1 for i, (c_, _o) in enumerate(der_dump) if c_ != dump[i][2])
                # The hard flag tests ACTUAL PER-EXAMPLE TENSOR IDENTITY, not a cosine threshold.
                # The first version keyed it on slot_cos > 0.9999 and produced FALSE ALARMS on the real
                # 3B run: a mean cosine of 0.99995 prints as "1.0000" and clears that bar while the
                # slots still genuinely differ, and greedy decode flips readily on a near-tie. Only
                # "these two slot tensors are allclose YET the greedy text differs" is actually
                # impossible, so that is what gets flagged. Cosine stays the COLLAPSE diagnostic below,
                # which is all it was ever able to support.
                n_ident_diff = 0
                if len(held_slots) == len(held_ex):
                    for i, (c_, _o) in enumerate(der_dump):
                        same_slots = torch.allclose(held_slots[derange_perms[0][i]].float(),
                                                    held_slots[i].float(), atol=1e-6, rtol=1e-5)
                        if same_slots and c_ != dump[i][2]:
                            n_ident_diff += 1
                if n_ident_diff:
                    der_note = (f"  [!! APPARATUS: {n_ident_diff}/{len(der_dump)} examples have "
                                f"ALLCLOSE slots yet DIFFERENT greedy text -- impossible via the slot "
                                f"path, so something else is moving the output]")
                elif slot_cos_val is not None and slot_cos_val > 0.99:
                    der_note = (f"  [slots ~COLLAPSED (slot_cos {slot_cos_val:.4f}): deranged and real "
                                f"memory are near-identical inputs, so this arm cannot be informative]")
                elif slot_cos_val is not None and n_diff == 0:
                    der_note = "  [slots vary across tasks, LM output invariant to them: channel unused]"
            nsteps_str = ""
            _enst = getattr(R, '_eval_nsteps', [])
            if _enst:
                nsteps_str = f"  mean_steps {sum(_enst)/len(_enst):.2f}"
                R._eval_nsteps = []
            reground_str = (f"  reground {reground_ok_count}/{len(held_ex)}"
                            if (R.top_trm is not None or merged) else "")
            evicted_str = (f"  reground_evicted {reground_evicted_ok_count}/{len(held_ex)}"
                          if evict_window is not None else "")
            print(f"  held WM {held_ok}/{len(held_ex)}  ablated {ablated_ok}/{len(held_ex)}"
                  f"{deranged_str}{reground_str}"
                  f"{evicted_str}{slot_cos_str}{nsteps_str}  {inst_str}{der_note}", flush=True)
            if len(held_ex) <= 8:
                for d in dump:
                    t, tc, wm_code, wm_ok, abl_code, abl_ok, instab = d[:7]
                    print(f"       target: {tc}")
                    print(f"       WM:     {wm_code[:80]}  {'PASS' if wm_ok else 'FAIL'}  instab {instab:.3f}")
                    print(f"       ablt:   {abl_code[:80]}  {'PASS' if abl_ok else 'FAIL'}")
                    if len(d) > 7:
                        rg_text, rg_ok = d[7], d[8]
                        print(f"       rgnd:   {rg_text[:80]}  {'PASS' if rg_ok else 'FAIL'}")
                    if len(d) > 9:
                        rge_text, rge_ok = d[9], d[10]
                        print(f"       rgnd_evicted: {rge_text[:80]}  {'PASS' if rge_ok else 'FAIL'}")

    print(f"\n  [dump] final epoch, held-out generations (WM vs ablated) vs the verified target:")
    code_prefix = "" if task_domain in ("math-cot", "swe-action") else "def task(n): "
    for row in (last_dump or []):
        text, target_code, code, wm_ok, code_abl, abl_ok, instability = row[:7]
        print(f"     task: {text}")
        print(f"       target : {target_code}")
        print(f"       WM     : {code_prefix}{code[:90]}{'  <- PASS' if wm_ok else ''}  instab={instability:.3f}")
        print(f"       ablated: {code_prefix}{code_abl[:90]}{'  <- PASS' if abl_ok else ''}")
        if len(row) > 7:
            rg_text, rg_ok = row[7], row[8]
            print(f"       reground: {code_prefix}{rg_text[:90]}{'  <- PASS' if rg_ok else ''}")
        if len(row) > 9:
            rge_text, rge_ok = row[9], row[10]
            print(f"       reground_evicted: {code_prefix}{rge_text[:90]}{'  <- PASS' if rge_ok else ''}")

    print(f"\n  Best held-out: {best_held}/{len(held_ex)}  (gate ablated = {ablated_ok} baseline)")
    verdict = "PROVEN" if best_held > ablated_ok else "PARTIAL"
    print(f"  => {verdict}: working memory {'improves' if best_held > ablated_ok else 'does not improve'} held-out composition on {lm_name}")

    # ============================================================================================
    # THE FALSIFIER REPORT. Deliberately separate from the line above, which compares a running MAX
    # (best_held, over all checkpoints) against a LAST-checkpoint ablated count -- fine as a headline,
    # useless as a control. Everything below is from ONE checkpoint (the final one), so the three arms
    # are apples to apples.
    # ============================================================================================
    if der_dump:
        N = len(held_ex)
        specific = held_ok - deranged_ok          # does it matter WHICH atoms are in memory
        modal = deranged_ok - ablated_ok          # does the mere PRESENCE of slots matter
        print(f"\n  {'=' * 72}")
        print(f"  DERANGEMENT FALSIFIER   (final checkpoint, held {N}, one fixed derangement)")
        print(f"    WM        (own slots)        {held_ok}/{N}")
        print(f"    DERANGED  (another task's)   {deranged_ok}/{N}")
        print(f"    ABLATED   (no slots at all)  {ablated_ok}/{N}")
        print(f"    instance-specific  WM - deranged      = {specific:+d}/{N}")
        print(f"    format/mode effect deranged - ablated = {modal:+d}/{N}")
        if slot_cos_val is not None:
            print(f"    across-task slot cosine               = {slot_cos_val:.6f}")

        # ---- CE readout, bootstrapped. This is the quotable number; the pass counts above are not. ----
        if ce_wm and len(ce_wm) == len(ce_der) == len(ce_abl):
            a_wm, a_der, a_abl = np.array(ce_wm), np.array(ce_der), np.array(ce_abl)
            d_spec = a_der - a_wm            # >0 => the RIGHT memory lowers CE vs a wrong one
            d_modal = a_abl - a_der          # >0 => having ANY memory lowers CE vs none
            # PAIRED bootstrap over EXAMPLES. RandomState is constructed ONCE, outside the
            # comprehension: building it inside made all 4000 resamples identical and printed a
            # zero-width CI for an effect of +0.0011 (algo_grr_contrast.py, bug #6).
            rs = np.random.RandomState(0)
            idx = [rs.randint(0, len(a_wm), len(a_wm)) for _ in range(4000)]
            bs = np.array([d_spec[j].mean() for j in idx])
            bm = np.array([d_modal[j].mean() for j in idx])
            s_lo, s_hi = np.percentile(bs, [2.5, 97.5])
            m_lo, m_hi = np.percentile(bm, [2.5, 97.5])
            print(f"\n    teacher-forced CE, nats (lower = better), {len(derange_perms)} derangements")
            print(f"      CE  WM                              {a_wm.mean():.4f}")
            print(f"      CE  DERANGED                        {a_der.mean():.4f}")
            print(f"      CE  ABLATED                         {a_abl.mean():.4f}")
            print(f"      instance-specific  der - WM         {d_spec.mean():+.4f}  "
                  f"95% CI [{s_lo:+.4f}, {s_hi:+.4f}]")
            print(f"      format/mode effect  abl - der       {d_modal.mean():+.4f}  "
                  f"95% CI [{m_lo:+.4f}, {m_hi:+.4f}]")
            # MAGNITUDE, not just significance. A paired bootstrap over enough examples will exclude 0
            # for an arbitrarily tiny bias, so "CI excludes 0" is NOT a result on its own -- the first
            # version of this block printed "CONFIRMS a real instance-specific channel" for +0.0043
            # nats, a number sitting inside the 0.0011-0.0075 band this project already documents AS
            # the adapter-shortcut null. The two things that decide it are the SCALE (against that
            # band and against the 0.3474 contrastive positive control) and the RATIO against the
            # format/mode effect, which is what "instance-independent adapter" means quantitatively.
            NULL_BAND_HI = 0.0075     # algo_grr_contrast.py: CE-trained instance-specific ceiling
            ratio = float(d_modal.mean() / d_spec.mean()) if d_spec.mean() > 1e-9 else float("inf")
            print(f"      modal / specific ratio              {ratio:>8.1f}x   "
                  f"[>50x = instance-independent adapter]")
            if s_hi < 0:
                ce_msg = ("SIGN ERROR: the WRONG memory scores BETTER than the right one. Not a weak "
                          "effect -- check the perm indexing and slot/example alignment before "
                          "interpreting anything else in this run.")
            elif s_lo <= 0:
                ce_msg = (f"NO instance-specific signal (CI includes 0), with {d_modal.mean():+.4f} nats "
                          f"of format/mode effect present. Adapter-shortcut shape: the adapter helps, "
                          f"the CONTENT of memory does not.")
            elif d_spec.mean() <= NULL_BAND_HI:
                ce_msg = (f"ADAPTER SHORTCUT: the CI excludes 0, but {d_spec.mean():+.4f} nats is INSIDE "
                          f"the documented CE null band (<={NULL_BAND_HI}). Significance without "
                          f"magnitude is not a channel. Same signature as the three interfaces in "
                          f"algo_grr_contrast.py (250-950x).")
            elif ratio > 50:
                # Say WHICH bar failed. The first version printed "inside the null band and/or the
                # format effect is Nx larger" for an effect 18x ABOVE that band -- the opposite of the
                # truth, on exactly the run where the effect finally cleared it.
                ce_msg = (f"MIXED / FORMAT-DOMINATED: {d_spec.mean():+.4f} nats CLEARS the "
                          f"{NULL_BAND_HI} null band by {d_spec.mean()/NULL_BAND_HI:.0f}x and the CI "
                          f"excludes 0 -- a real instance-specific channel -- but the format effect is "
                          f"still {ratio:.0f}x larger, above the 50x instance-independent-adapter bar. "
                          f"Content matters here; it is not yet what the adapter mostly does.")
            else:
                ce_msg = (f"REAL instance-specific channel: {d_spec.mean():+.4f} nats, CI excludes 0, "
                          f"above the {NULL_BAND_HI} null band, and only {ratio:.0f}x smaller than the "
                          f"format effect. First result in this codebase to clear all three bars. "
                          f"Reference points: 0.0011-0.0075 CE null, 0.3474 contrastive control.")
            print(f"      -> {ce_msg}")
        # VACUOUS-RUN GUARD. Caught by the distilgpt2 plumbing run, which scored 0/0/0 and printed
        # "FALSIFIED" off specific <= 0 -- but 0-0=0 is an ABSENCE of measurement, not a falsification.
        # A falsifier that "fires" when nothing was solved would hand back exactly the kind of confident
        # null this project has already been burned by three times.
        if held_ok == 0:
            msg = ("VACUOUS: the WM arm solved NOTHING, so all three arms are 0 and no comparison "
                   "between them carries information. This says nothing about the channel -- fix the "
                   "run (model, epochs, task difficulty) before reading any verdict here.")
        elif specific > 0 and specific >= modal:
            msg = ("REAL: most of the gain needs the RIGHT working memory, not merely some working "
                   "memory. This is the first arm in this codebase that could have falsified the "
                   "13-15/16 headline and did not.")
        elif specific > 0:
            msg = (f"MIXED: the content matters ({specific:+d}) but the larger share is a format/mode "
                   f"effect ({modal:+d}) that ANY slots unlock. Report both; the headline overstates.")
        else:
            msg = ("FALSIFIED: another task's working memory scores at or above the real one. The gain "
                   "is a format/mode effect, not retrieved memory -- the same shape as the adapter "
                   "shortcut in algo_grr_contrast.py. Do not build on this channel under CE.")
        print(f"    -> {msg}")
        print(f"  {'=' * 72}")
        print(f"  CAVEAT, stated up front: N={N} and ONE derangement. A {N}-item pass rate has a wide "
              f"binomial interval; treat a gap of 1-2 as noise. Multiple permutations and a bootstrap "
              f"are required before this number is quotable.")
    elif derange:
        print(f"\n  [derangement falsifier did not run: needs >=2 held tasks and a completed eval "
              f"checkpoint; held={len(held_ex)}]")

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

        # lr 1e-3, not 1e-2: Adam moves weights by ~lr per step regardless of gradient magnitude, so the
        # old 1e-2 x 200 steps inflated critic weights even while gradients were ~0 under saturation --
        # which is why loading an old checkpoint still shows ~91% saturation after the LayerNorm fix.
        c_opt = torch.optim.Adam(list(R.critic.parameters()) + list(R.critic_pool.parameters())
                                 + list(R.critic_norm.parameters()) + list(R.critic_ctx.parameters()),
                                 lr=1e-3)
        for _ in range(200):
            random.shuffle(c_train_balanced)
            c_opt.zero_grad()
            loss = R.critic_loss([s for s, _, _ in c_train_balanced],
                                 [y for _, y, _ in c_train_balanced],
                                 ctxs=[c for _, _, c in c_train_balanced])
            loss.backward(); c_opt.step()
        with torch.no_grad():
            scores = [float(R.critique(s, c)) for s, _, c in c_test]
            labels = [bool(y) for _, y, _ in c_test]
            preds = [sc >= 0.5 for sc in scores]
            acc = sum(int(p == y) for p, y in zip(preds, labels)) / max(1, len(c_test))
            base_rate = sum(labels) / max(1, len(labels))
            auc = _roc_auc(scores, labels)
            n_distinct = len(set(round(s, 6) for s in scores))
        base_acc = max(base_rate, 1 - base_rate)
        print(f"  critic held-out accuracy: {acc:.2f}  (base rate / always-predict-majority: "
              f"{base_acc:.2f})  n_test={len(c_test)}")
        # AUC is the honest headline: it is 0.5 for ANY constant predictor regardless of class balance,
        # so unlike accuracy it cannot be faked by collapsing to one class. A saturated critic that always
        # emitted 1.0 previously scored exactly 1-base_rate on accuracy and was repeatedly misread as
        # "no signal in the data" rather than "the model is constant".
        print(f"  critic held-out AUC: {auc:.2f}  (0.50 = no ranking ability / constant predictor; "
              f"distinct scores {n_distinct}/{len(scores)})")
        if not pos or not neg:
            # Degenerate DATA, not a broken critic: with one class only, BCE's optimum IS a constant, and
            # AUC is undefined. Distinguishing this from a genuinely stuck critic matters -- conflating
            # them is how a saturated constant predictor got read as "no signal" for many runs.
            print(f"  => UNINFORMATIVE: the critic's training set had only one class "
                  f"({len(pos)} pass / {len(neg)} fail), so a constant output is the correct solution and "
                  f"AUC is undefined. This says nothing about whether a real sense of mistake is learnable "
                  f"-- it needs runs where the model both succeeds and fails.")
        elif n_distinct <= 1:
            print(f"  => CRITIC IS CONSTANT despite seeing both classes -- it is not judging anything. "
                  f"Ignore the accuracy number entirely; this is a mechanism failure, not a statement "
                  f"about the data.")
        elif auc == auc and auc > 0.5:      # auc==auc filters NaN (single-class test set)
            print(f"  => critic ranks correct above incorrect (AUC {auc:.2f} > 0.50) -- a real, if partial, "
                  f"sense of mistake. Accuracy vs base rate: {'beats' if acc > base_acc else 'does not beat'}.")
        else:
            print(f"  => no ranking signal found (AUC {auc:.2f}) -- report honest.")
    else:
        print(f"\n  (only {len(critic_examples)} labeled trajectories collected -- too few to train/report the "
              f"critic meaningfully; needs more epochs or a bigger held-out set)")

    if save_path:
        R.save(save_path)
        print(f"\n  saved trained WMReasoner to {save_path} ({sum(p.numel() for p in R.parameters())} params). "
              f"Use it live with:  python -m v5.runtime.membrane --interactive --lm {lm_name} "
              f"--wm-path {save_path}   (membrane.attach_wm rebuilds the TRM from this checkpoint's own "
              f"recorded shape and registers the LM hooks). This line used to name a Membrane(wm_path=...) "
              f"argument that did not exist, and no code path anywhere ever put a trained adapter on a live "
              f"graph.")

    # EXPLAIN: the model must be able to say what it did, grounded in real memory -- not silently. Two real
    # probes, not a design claim: (1) SHORT-TERM -- ask about a task just solved this session; SessionFocus's
    # spreading activation should light up around it. (2) LONG-TERM -- ask about a graph node this session's
    # short-term probe never activated, forcing the plain-cosine long-term fallback in explain_what_happened.
    if graph_path and last_dump:
        from v5.runtime.membrane_session import SessionFocus
        session = SessionFocus(g)
        print(f"\n  [explain] can the model say what it did, grounded in memory (short-term session focus, "
              f"or long-term over the persistent graph) instead of solving silently?")
        recent_text = last_dump[0][0]
        probe1 = explain_what_happened(wb, g, session, f"What did you just do to solve this task: {recent_text}")
        print(f"    short-term probe (about a task just solved this session):")
        print(f"      tier={probe1['tier']}  grounded on={probe1['nodes']}")
        print(f"      answer: {probe1['answer'][:200]!r}")
        outside_focus = [n for n in g.names() if n not in session.focus]
        if outside_focus:
            probe_desc = g.get(outside_focus[0]).description
            probe2 = explain_what_happened(wb, g, session, f"What do you know about: {probe_desc}")
            print(f"    long-term probe (about something this session's focus never touched):")
            print(f"      tier={probe2['tier']}  grounded on={probe2['nodes']}")
            print(f"      answer: {probe2['answer'][:200]!r}")

    if growth is not None:
        print(f"\n  {growth.summary()}")

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
    ap.add_argument("--slot-gate", action="store_true", dest="slot_gate",
                    help="spiking slot gate: LIF over the K WM slots decides per POSITION which "
                         "slots may inject (lateral inhibition = anti routing-collapse)")
    ap.add_argument("--slot-theta0", type=float, default=0.0, dest="slot_theta0")
    ap.add_argument("--slot-w-inh", type=float, default=0.5, dest="slot_w_inh")
    ap.add_argument("--slot-t", type=int, default=3, dest="slot_t")
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
    ap.add_argument("--grow-cot-docs-path", type=str, default="",
                    help="--grow-cot: read pre-fetched docs from this jsonl instead of live-streaming "
                         "OpenThoughts inside this process. Real, confirmed reason this matters, not just "
                         "an option: HF `datasets` streaming's first real fetch in a process segfaults if "
                         "torch was already active earlier in it, and torch is already loaded here (trm_wm.py "
                         "imports it at module level) by the time --grow-cot would otherwise run -- live "
                         "streaming from inside --run WILL likely crash on an environment with this "
                         "conflict. Produce the file first with a separate, torch-free process: "
                         "`python -m v5.graph_grower.fetch_cot --out <path> --limit N`.")
    ap.add_argument("--math-cot-docs-path", type=str, default="",
                    help="--task-domain math-cot: same real crash, different function -- "
                         "_math_cot_tasks_from_graph does its OWN separate live load_dataset() call, hit "
                         "by the exact same torch-already-loaded segfault risk as --grow-cot (confirmed: "
                         "a real --run with --task-domain math-cot crashed here even with --grow-cot-docs-"
                         "path already set, since this is a second, independent live-streaming call). "
                         "Pre-fetch RAW rows (not row_to_doc docs) with a separate torch-free process: "
                         "`python -m v5.graph_grower.fetch_cot --raw --out <path> --limit N`.")
    ap.add_argument("--trigger-patterns", type=str, default="",
                    help="--top-trm-t>0: comma-sep substrings (e.g. '\\n,Therefore,Step'). If any appear in "
                         "a chunk's newly generated text during reground/reground_evicted, top recomputes "
                         "on the very next chunk regardless of --reground-top-every's cadence -- an event "
                         "(a real reasoning/sentence boundary) triggers recompute early; the cadence still "
                         "fires as a fallback so top is never starved if no trigger appears. Empty "
                         "(default) = pure cadence, unchanged behavior.")
    ap.add_argument("--instability-trigger", type=float, default=0.0,
                    help="--top-trm-t>0: a real, learned alternative to --trigger-patterns -- the BOTTOM "
                         "trm triggers top itself, off its own computed state (trajectory_instability, the "
                         "same late/early y_t convergence ratio already printed as `instab` in every "
                         "held-out eval), not off hand-picked strings. RELATIVE MULTIPLIER against the "
                         "running mean of earlier chunks in the same generation -- e.g. 1.5 means '50%% "
                         "above this generation's own recent average'. NOT an absolute threshold: a real "
                         "20-epoch Qwen3-4B run measured instability at 0.062 (epoch 0) decaying to 0.001 "
                         "(epoch 10), so the absolute 1.0 originally suggested here never fired once and "
                         "silently tested nothing -- and since the signal itself decays ~30x during "
                         "training, no fixed value can work. 0.0 (default) = off, unchanged behavior.")
    ap.add_argument("--sink-tokens", type=int, default=0,
                    help="requires --evict-window: keep this many tokens from the very START of the "
                         "sequence alongside the recent window -- StreamingLLM-style attention sinks. "
                         "STRONGLY RECOMMENDED whenever --evict-window is used: measured on real "
                         "Qwen3-4B, pure sliding window (0, the default) scored 7/15 on a "
                         "list-the-primes-with-a-prefix task and collapsed into a 29x repeated-token "
                         "loop, while sinks covering the prompt scored 14/15 coherently against a "
                         "15/15 no-eviction ceiling -- because a pure window's oldest tokens ARE the "
                         "prompt (the task description). Set it to roughly the real prompt length.")
    ap.add_argument("--top-no-graph", action="store_true",
                    help="--top-trm-t>0: make the top trm REAL RECURRENT MEMORY instead of a second "
                         "retriever -- it cross-attends over its OWN accumulated history of progress "
                         "contexts, NOT the graph atoms the bottom trm uses. Without this both levels get "
                         "IDENTICAL input (the same K atoms) so the 'hierarchy' is only a cadence split, "
                         "which a real 40-epoch Qwen3-4B A/B measured as noise (14/16 vs ~13/16 baseline). "
                         "Division of labour: top = evolving memory over its own experience, bottom = "
                         "controller/communicator holding the graph atoms and reaching the LM.")
    ap.add_argument("--top-memory-max", type=int, default=16,
                    help="--top-no-graph: cap on how many past progress contexts the top trm keeps in its "
                         "own memory bank (rolling window, keeps the most recent).")
    ap.add_argument("--swe-max-ctx-steps", type=int, default=12,
                    help="--task-domain swe-action: how many prior trajectory steps go into the prompt. "
                         "At the default 12 (x200 chars) the step context is the LARGEST part of the "
                         "prompt, bigger than a capped issue body. Total prompt length is what drives "
                         "memory -- swe-action prompts run ~1,250 tokens against synthetic's ~25, a 50x "
                         "increase, and training attention activations scale with seq^2 (~7GB of attention "
                         "scores alone across 36 layers). Lower this FIRST if a run OOMs.")
    ap.add_argument("--swe-max-issue-chars", type=int, default=1500,
                    help="--task-domain swe-action: cap the GitHub issue body fed into the prompt. Real "
                         "problem_text reaches 18,841 chars (p90 4,454), and leaving it uncapped produced "
                         "~5,300-token prompts; since training attention memory is quadratic in sequence "
                         "length, a real run hit ~60GB VRAM on a 4-bit 4B model. The head of an issue holds "
                         "the actual problem statement; the tail is usually traces and version tables.")
    ap.add_argument("--swe-target-args-chars", type=int, default=0,
                    help="--task-domain swe-action: chars of tool ARGUMENTS to include in the "
                         "teacher-forcing target. 0 (default) = tool name only, which is what verify() "
                         "actually scores. Including full args was an objective/metric mismatch: measured "
                         "over 3,396 real steps the full tool(args) target runs p50 43 / p90 377 / max "
                         "3,949 tokens while the graded tool name is always 3, so nearly all the loss fell "
                         "on instance-specific paths and shell commands that cannot generalize. Set >0 "
                         "only if you also change verify() to grade arguments.")
    ap.add_argument("--merged", action="store_true",
                    help="MERGE the top trm's job into the bottom trm and drop the second network. The "
                         "bottom trm carries its own latent state across chunks and attends its own "
                         "accumulated memory concatenated with the graph atoms. Main reason is not "
                         "simplicity: in the hierarchical design memory could only reach the LM through "
                         "the ZERO-INIT top_to_bottom_proj, and a direct measurement shows memory changes "
                         "the slots by exactly 0.00e+00 there versus 1.25e-02 merged -- the two-level "
                         "version literally cannot deliver memory to the LM at init, the same failure that "
                         "made reground a no-op. Also 30%% fewer parameters, and a 3-arm A/B never showed "
                         "the extra level winning (bottom-only 15/16, hierarchical 14/16, top-memory 15/16).")
    ap.add_argument("--ds-weight", type=float, default=0.0,
                    help="weight on DEEP SUPERVISION -- MSE between ds_proj(y_t) and the projected "
                         "INTERMEDIATE RESULT the recursion should hold at step t (for outer(inner(n)): "
                         "inner(n) early, the full composition late). Targets are partial RESULTS, never "
                         "the retrieved atom-node embeddings -- supervising on atoms teaches the trm to "
                         "echo what retrieval already gave it instead of composing. Without this the only "
                         "loss touching y_t is conv_loss, which rewards NOT changing: a real trained "
                         "checkpoint had across-task slot cosine 1.000000 while its inputs sat at 0.4510, "
                         "and swapping a different task's slots changed generation on 1 of 16 tasks. "
                         "0 (default) = off, reproducing prior runs. Try 0.1. Synthetic composition only; "
                         "domains with no oracle intermediate skip it per-example.")
    ap.add_argument("--conv-weight", type=float, default=0.05,
                    help="weight on the convergence regularizer (penalizes ||y_t+1 - y_t||). Was hardcoded "
                         "0.05. It pulls directly AGAINST task-conditioning -- a strongly contractive "
                         "recursion has one fixed point reached from any input -- so lower it (or 0) when "
                         "testing whether the trm can be made task-specific.")
    ap.add_argument("--gate-reg-weight", type=float, default=0.05,
                    help="weight on the gate magnitude penalty (was hardcoded 0.05).")
    ap.add_argument("--adaptive-t", action="store_true",
                    help="ACT (Adaptive Computation Time, Graves 2016) for the TRM. Adds a learned halt "
                         "head that predicts per-step halting probability; the model decides how many "
                         "recursion steps each task needs instead of always using all T. Easy tasks (1-atom "
                         "lookup) should learn to halt early, hard tasks (nested composition) use more. "
                         "Always runs T_max steps internally (shapes stay fixed), but halt_weights give a "
                         "differentiable soft allocation. Pair with --ponder-weight to penalize unnecessary "
                         "computation. Off by default = fixed T, reproducing prior runs.")
    ap.add_argument("--ponder-weight", type=float, default=0.0,
                    help="weight on the pondering penalty (mean steps used across the batch). Only active "
                         "when --adaptive-t is set. Encourages the model to halt early when it can. "
                         "Too high -> always halts at step 1 (underthinking). Too low -> always uses all T "
                         "(no benefit from adaptive). Try 0.01. 0 (default) = no penalty.")
    ap.add_argument("--gate-max", type=float, default=0.0,
                    help="hard ceiling on |gate| after every optimizer step. The gate is a LEARNED "
                         "parameter and lm_loss pushes it UP (harder injection fits the teacher-forced "
                         "target even while destroying held-out behaviour); the 0.05*gate_reg penalty "
                         "loses that fight. Measured on a real swe-action run starting at --gate-init "
                         "0.05: ep0 gate 0.09 -> WM 3/16 (== ablated), ep5 gate 0.24 -> WM 4/16 (beats "
                         "ablated), ep10 gate 0.35 -> WM 0/16, ep39 gate 0.80 -> WM 0/16. The usable band "
                         "is roughly 0.1-0.25 and training walks out of it. Try 0.25. 0 (default) = "
                         "no clamp, unchanged behaviour.")
    ap.add_argument("--gate-init", type=float, default=0.8,
                    help="warm-start value for the GatedCrossAttn gate (effective strength is tanh of "
                         "this). 0.8 (default) injects hard from step 0 through still-random projections; "
                         "that is fine only where the ablated baseline is 0 by construction (synthetic "
                         "composition). On a domain where the base LM ALREADY has ability it destroys it -- "
                         "measured on swe-action: held WM 0/16 vs ablated 3/16, i.e. working memory strictly "
                         "WORSE than none. Use a small value (e.g. 0.05) there so the adapter must earn its "
                         "contribution, matching the zero-init discipline of every other component here.")
    ap.add_argument("--passive-growth", action="store_true",
                    help="requires --graph-path: the graph GROWS during training, not just at setup. Every "
                         "verified-correct generation is banked as a real node via learn_any and every "
                         "failure becomes a trap node (capped per epoch -- unbounded failure bookkeeping "
                         "previously crowded out real structure and needed a repair pass). Only "
                         "verifier-confirmed outcomes are ever written, so the anti-poison rule holds. "
                         "Off by default. NOTE this flag also repairs a silent no-op: on math-cot and "
                         "swe-action, atoms_needed holds concept DESCRIPTIONS while record_success looks up "
                         "by NAME, so those graph updates previously did nothing at all -- the "
                         "description->name mapping now runs whenever --graph-path is set, flag or not.")
    ap.add_argument("--swe-docs-path", type=str, default="",
                    help="--task-domain swe-action: pre-fetched Open-SWE-Traces trajectories (jsonl, one "
                         "object per line with problem_text + steps). REQUIRED in practice -- HF datasets "
                         "streaming segfaults once torch is loaded, the same confirmed conflict that forces "
                         "--grow-cot-docs-path and --math-cot-docs-path. Produce it from a separate "
                         "torch-free process using v5.graph_grower.fetch_swe_traces.stream_swe_trajectories.")
    ap.add_argument("--evict-to-memory", action="store_true",
                    help="requires --evict-window (and pairs with --top-no-graph): tokens leaving the KV "
                         "cache are decoded, embedded, and appended to the TOP trm's memory bank instead "
                         "of being lost. Makes eviction compress-into-long-term-memory rather than pure "
                         "forgetting, so a long-horizon run keeps a usable record of the middle of the "
                         "generation (sinks hold the prompt, the window holds recent tokens, and without "
                         "this everything between them is simply gone). Also avoids a real truncation "
                         "issue in the other memory path: encode_batch truncates at 128 tokens, so "
                         "embedding the whole generated-so-far repeatedly only ever captures its first "
                         "~128 tokens; an evicted span is small enough to embed faithfully.")
    ap.add_argument("--cotrain-samples", type=int, default=-1,
                    help="cap how many train tasks get a real generate() in the co-training data pass at "
                         "each eval checkpoint. Measured at ~43%% of total eval cost (a full generate() over "
                         "EVERY train task, EVERY checkpoint), feeding only the tier-4 critic -- which has "
                         "never beaten its base rate in any real run here. -1 (default) = all, unchanged; "
                         "0 = skip entirely (critic still gets the held-out trajectories); N = random "
                         "sample of N per checkpoint. Use 0 for the fastest real held-out A/B.")
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
    ap.add_argument("--task-domain", type=str, default="synthetic",
                    choices=["synthetic", "math-cot", "swe-action"],
                    help="--run: 'synthetic' (default, unchanged) = 2-atom math composition. 'math-cot' = "
                         "real OpenThoughts math CoT problems, verified against the dataset's own boxed "
                         "final answer (requires --graph-path; run --grow-cot at least once first so "
                         "concept nodes exist to ground in). Honest small-N caveat: only a fraction of real "
                         "problems have a clean numeric final answer.")
    ap.add_argument("--math-cot-n-raw", type=int, default=150,
                    help="--task-domain math-cot: how many raw OpenThoughts rows to stream before filtering "
                         "to numeric-boxed-answer ones (yield rate measured ~22%% on a real sample -- request "
                         "more raw rows than you want kept)")
    ap.add_argument("--top-trm-t", type=int, default=0,
                    help="--run: attach a hierarchical top-level TRM with this many recursion steps (e.g. "
                         "8-24, matching the real TRM paper's own recipe for hard tasks) manipulating the "
                         "bottom TRM (T=4, the one that reaches the LM) via a zero-init additive projection "
                         "-- a strict no-op until it trains. 0 (default) = no top TRM, byte-identical to "
                         "before this flag existed. When >0, held-out eval reports an extra (reground) "
                         "column alongside the existing (static)/(ablated) ones -- an A/B, not a swap.")
    ap.add_argument("--reground-chunk-tokens", type=int, default=16,
                    help="--top-trm-t>0: bottom TRM re-grounds (re-runs refine() on the partial generation "
                         "so far) every this many generated tokens, instead of once before generation starts.")
    ap.add_argument("--reground-top-every", type=int, default=4,
                    help="--top-trm-t>0: top TRM recomputes every this many bottom re-ground ticks (the "
                         "actual fast/slow cadence split -- top runs less often than bottom, not on a "
                         "separate OS thread, which wouldn't give real concurrency here anyway).")
    ap.add_argument("--max-new-tokens", type=int, default=0,
                    help="generation budget for all held/ablated/reground/co-training generate() calls. "
                         "0 (default) = domain-aware: 128 for synthetic (real completions are ~13 tokens), "
                         "512 for math-cot (real OpenThoughts reasoning traces routinely run 300-1000+ "
                         "tokens before \\boxed{...} -- 128 would cut them off before the model ever reaches "
                         "the boxed answer, regardless of whether the reasoning was on track).")
    ap.add_argument("--use-kv-cache", action="store_true",
                    help="--top-trm-t>0 only: thread a real KV cache between reground chunks instead of "
                         "recomputing the whole prefix from scratch every chunk. Validated byte-identical "
                         "to the default (no-cache) path on real Qwen3-4B under greedy decoding -- pure "
                         "compute win, no output-changing risk. Off by default (matches every prior run's "
                         "behavior exactly; opt in explicitly).")
    ap.add_argument("--evict-window", type=int, default=0,
                    help="sliding-window KV eviction, keeps VRAM roughly constant as generation grows "
                         "instead of O(total tokens) -- ~144 KiB/token for Qwen3-4B (36 layers, 8 KV heads, "
                         "head_dim 128, fp16 cache), so a real long SWE-trace-length generation (~8k "
                         "tokens) would otherwise cost ~1.1GB of cache alone; capped at evict_window tokens "
                         "instead. 0 (default) = off. Position-compensated fix validated on real Qwen3-4B "
                         "(no more garbled-token degeneration). Adds a real reground_evicted A/B column "
                         "alongside held/ablated/reground in the eval output -- same task/slots/verify(), "
                         "only evict_window differs, so the pass-rate gap (if any) isolates eviction's real "
                         "cost, not a confound with use_kv_cache itself.")
    ap.add_argument("--no-derange", action="store_false", dest="derange",
                    help="--run: SKIP the derangement falsifier (on by default). The falsifier generates "
                         "each held task a third time using ANOTHER held task's working memory, which is "
                         "the only arm that separates 'the LM reads THESE slots' from 'any slots unlock a "
                         "mode where atom names appear'. `ablated` cannot do that: on the synthetic domain "
                         "its baseline is 0/16 by construction. Costs one extra greedy generate() per held "
                         "task per eval checkpoint and changes no existing arm, so pass this only when you "
                         "are deliberately trading the control away for wall clock.")
    ap.add_argument("--slotdim", type=int, default=0, dest="slotdim_k",
                    help="use a Differential Impact Mechanism accumulator with this many competing "
                         "slot banks INSTEAD of the TRM recursion (0 = off). The recursion measures "
                         "as a contraction: two slots seeded from different atoms go cos 0.0951 -> "
                         "0.8072 in ONE step, which is why every write-side fix failed. DIM asks how "
                         "violently an input perturbs the system rather than how similar two tokens "
                         "are, and accumulates rather than re-transforming. Try 8.")
    ap.add_argument("--opaque-atoms", action="store_true", dest="opaque_atoms",
                    help="rename atoms to non-semantic tokens (op_qx7, ...) while keeping their "
                         "English descriptions. Without this the task text describes exactly the "
                         "operation each atom is named after, so the LM writes the answer from the "
                         "PROMPT and never consults memory -- observed: target square(fibonacci(n)), "
                         "generated fibonacci(n)**2, PASSED. When the prompt holds a full solution "
                         "path, WM and DERANGED cannot differ and the falsifier detects nothing.")
    ap.add_argument("--code-prompt", action="store_true", dest="code_prompt",
                    help="end the prompt at 'def task(n): return ' instead of 'Explanation:'. The "
                         "shipped prompt asks for PROSE while the target is a bare return fragment, "
                         "so ablated scores 0 by obeying its instruction and the adapter's whole gain "
                         "is 'emit code instead of prose'.")
    ap.add_argument("--eval-every", type=int, default=0, dest="eval_every_arg",
                    help="epochs between eval checkpoints (0 = the old epochs//8). Eval, not "
                         "training, dominates wall clock: at held=32 each checkpoint is ~96 generate() "
                         "calls. Raise this when you care about the FINAL numbers rather than the "
                         "per-epoch trace.")
    ap.add_argument("--slotdim-heads", type=int, default=0, dest="slotdim_heads",
                    help="use SlotDIM **v2** with this many slot HEADS (0 = v1). v2 adds multi-head "
                         "slot sets, input-dependent selective decay, and a DIM-faithful ABSOLUTE "
                         "sensitivity gate tanh(sigma'(Z)) -- the softmax alone is only relative and "
                         "will force a write for a token that perturbs nothing. Measured at init: "
                         "within-instance slot cosine 0.9411 (v1) -> 0.0102 (v2, 2 heads), i.e. the "
                         "slots stop being copies of each other.")
    ap.add_argument("--couple-frac", type=str, default="", dest="couple_frac",
                    help="comma-separated FRACTIONS of LM depth to inject at, e.g. 0.5,0.75,0.95. "
                         "Default (empty) = the last two layers, which leaves almost no depth to "
                         "integrate slot content and is a prime suspect for the persistent "
                         "format-over-content result.")
    ap.add_argument("--delta-mode", type=str, default="rescale", choices=["rescale", "clip"],
                    dest="delta_mode",
                    help="rescale (default) forces EVERY position to receive exactly "
                         "delta_scale*||h||; clip bounds the same budget but lets the adapter stay "
                         "quiet where it has nothing to add. This project's standing rule is clip.")
    ap.add_argument("--delta-scale", type=float, default=0.3, dest="delta_scale")
    ap.add_argument("--slotdim-depth", type=int, default=1, dest="slotdim_depth",
                    help="stack this many DIM write passes; pass i+1 writes FROM pass i's slot table. "
                         "depth 1 is derivative-based POOLING (grabs the most impactful atoms); "
                         "stacking is what lets slots refine against each other and resolve a relation "
                         "spanning two atoms. Try 2.")
    ap.add_argument("--slotdim-no-ffn", action="store_true", dest="slotdim_no_ffn",
                    help="ABLATE the per-pass LayerNorm+FFN inside SlotDIM (default: present). Without "
                         "it the slots are strictly linear combinations of input features gated by the "
                         "sensitivity scores -- able to SELECT impactful atoms, never to COMPOSE them.")
    ap.add_argument("--atom-slots", action="store_true", dest="atom_slots",
                    help="build the working memory as K slots, each SEEDED FROM ONE RETRIEVED ATOM and "
                         "refined over T steps -- the design this file's header has always described "
                         "and which was never implemented. What ran instead: slot count was T not K (1 "
                         "atom and 16 atoms both gave 4 slots), z0/y0 are shared zero Parameters so "
                         "every instance started from the SAME latent, and the cycle-0 attention query "
                         "is exactly 0 so the atoms entered as an exactly uniform mean with retrieval "
                         "rank discarded. Off by default; existing runs stay bit-identical.")
    ap.add_argument("--trm-no-bias", action="store_true", dest="trm_no_bias",
                    help="zero and freeze EVERY bias inside the TRM. This is the measured ROOT CAUSE "
                         "of the slot collapse: PyTorch's default nn.Linear bias init is sized for "
                         "unit-variance-per-dimension inputs, but this model runs on L2-normalized "
                         "MiniLM embeddings, so each bias is a per-instance CONSTANT comparable to or "
                         "larger than the signal (task_proj/atom_proj ||b||/||Wx|| = 0.99). Measured "
                         "across 3 seeds: y_t across-instance cosine 0.885 -> 0.364. Pair with "
                         "--contrast-weight, which cannot escape a collapse it starts inside.")
    ap.add_argument("--no-proj-bias", action="store_true", dest="no_proj_bias",
                    help="zero and freeze proj_y.bias. proj_y maps y_t to the slots the LM attends, "
                         "and its bias is a CONSTANT added to every slot of every task -- an "
                         "instance-independent component built into the architecture. Measured at "
                         "init: ||bias||/||W@y|| = 1.280, taking y_t cosine 0.8889 -> slots 0.9576. "
                         "The one module where the same per-step contrastive objective DID work "
                         "(algo_grr_state.py) has bias=False on its write projection for exactly "
                         "this reason. Pair with --contrast-weight.")
    ap.add_argument("--contrast-weight", type=float, default=0.0, dest="contrast_weight",
                    help="weight on the PER-RECURSION-STEP contrastive objective (SlotContrast). "
                         "0 (default) = off, reproducing prior runs exactly. THIS IS THE TERM THE "
                         "SLOTS WERE MISSING: the only per-step force on y_t was conv_loss, which "
                         "penalizes ||y_t+1 - y_t|| and so rewards NOT CHANGING, while --ds-weight "
                         "defaults to 0 and was off in every recorded run -- a real 40-epoch 3B run "
                         "printed `ds 0.0000` every epoch and ended at slot_cos 0.9880-1.0000 with "
                         "DERANGED slots scoring 8/16 against the real slots' 7/16. Collapse was the "
                         "OPTIMUM of the objective being minimized, not a training failure. Unlike "
                         "--ds-weight this needs no oracle intermediate, so it also applies to "
                         "math-cot and swe-action. Try 0.5 and read slot_cos + the derangement arm.")
    ap.add_argument("--contrast-negs", type=int, default=15, dest="contrast_negs",
                    help="negatives per anchor for --contrast-weight, drawn from TRAIN targets only")
    ap.add_argument("--n-derange", type=int, default=5,
                    help="--run: how many independent derangements the CE readout averages over "
                         "(default 5). GENERATION always uses derangement 0 only, so this costs "
                         "forward passes, not generations. More derangements make the deranged arm an "
                         "average rather than a single draw; a 16-item pass rate cannot resolve the "
                         "effect sizes this experiment is about, which is why CE is the quotable number.")
    a = ap.parse_args()
    if a.probe:
        probe_real(a.lm, a.quant, a.words, a.steps)
    elif a.run:
        run_real(a.lm, a.quant, a.epochs, a.n_train, a.n_held, a.graph_path, a.save_path,
                 a.grow_cot, a.grow_domains, a.grow_keywords, a.grow_skills, a.grow_skills_domains,
                 batch_size=a.batch_size, task_domain=a.task_domain, math_cot_n_raw=a.math_cot_n_raw,
                 top_trm_t=a.top_trm_t, reground_chunk_tokens=a.reground_chunk_tokens,
                 reground_top_every=a.reground_top_every, max_new_tokens=a.max_new_tokens,
                 use_kv_cache=a.use_kv_cache, evict_window=(a.evict_window or None),
                 grow_cot_docs_path=(a.grow_cot_docs_path or None),
                 math_cot_docs_path=(a.math_cot_docs_path or None),
                 trigger_patterns=([p for p in a.trigger_patterns.split(",") if p] or None),
                 instability_trigger=(a.instability_trigger or None),
                 sink_tokens=a.sink_tokens, cotrain_samples=a.cotrain_samples,
                 top_no_graph=a.top_no_graph, top_memory_max=a.top_memory_max,
                 evict_to_memory=a.evict_to_memory, swe_docs_path=(a.swe_docs_path or None),
                 passive_growth=a.passive_growth, gate_init=a.gate_init,
                 swe_max_issue_chars=a.swe_max_issue_chars,
                 swe_max_ctx_steps=a.swe_max_ctx_steps, gate_max=a.gate_max, merged=a.merged,
                 swe_target_args_chars=a.swe_target_args_chars,
                 ds_weight=a.ds_weight, conv_weight=a.conv_weight, gate_reg_weight=a.gate_reg_weight,
                 adaptive_t=a.adaptive_t, ponder_weight=a.ponder_weight,
                 slot_gate=a.slot_gate, slot_theta0=a.slot_theta0,
                 slot_w_inh=a.slot_w_inh, slot_t=a.slot_t, derange=a.derange,
                 n_derange=a.n_derange, contrast_weight=a.contrast_weight,
                 contrast_negs=a.contrast_negs, no_proj_bias=a.no_proj_bias,
                 trm_no_bias=a.trm_no_bias, atom_slots=a.atom_slots, slotdim_k=a.slotdim_k, slotdim_depth=a.slotdim_depth,
                 slotdim_no_ffn=a.slotdim_no_ffn, slotdim_heads=a.slotdim_heads,
                 couple_frac=[float(f) for f in a.couple_frac.split(",") if f.strip()],
                 delta_scale=a.delta_scale, delta_mode=a.delta_mode,
                 eval_every_arg=a.eval_every_arg, opaque_atoms=a.opaque_atoms,
                 code_prompt=a.code_prompt)
    else:
        selftest()


class TRMPointerGenerator(nn.Module):
    """TRM that generates text by mixing vocabulary generation with input-token copying.

    The TRM's own cross-attention weights serve as the pointer distribution —
    no separate pointer head. At each step the TRM refines z/y over n cycles
    against [input + generated] context, then y produces p_gen (copy vs generate)
    and vocabulary logits. The mixed output is p_gen * P_vocab + (1-p_gen) * P_copy
    where P_copy = attention over input tokens.

    For the session copy-number task: numbers get copied (p_gen→0, pointer selects
    the right input token), other text gets generated (p_gen→1)."""

    def __init__(self, d_lm: int, d_in: int = 384, d: int = 256, n: int = 6,
                 vocab_size: int = 151936):
        from v5.runtime.algo_trm import _build as _build_trm
        _, _, TRMReasoner = _build_trm()
        super().__init__()
        self.trm = TRMReasoner(d_in=d_in, d=d, T=n)
        self.p_gen = nn.Linear(d, 1)
        self.vocab_proj = nn.Linear(d, d_lm)
        self.input_proj = nn.Linear(d_lm, d_in)
        self.n = n
        self.vocab_size = vocab_size

    def forward(self, input_embs, lm_embed, generated_embs=None, task_emb=None):
        """Forward pass for one token position.

        input_embs: [N, d_in] per-token input embeddings
        lm_embed: [vocab_size, d_lm] LM embedding table (frozen)
        generated_embs: [G, d_in] or None — already-generated tokens
        task_emb: [d_in] or None — optional task embedding (zero if None)

        Returns: (pg [1], vocab_logits [vocab_size], w [N+G])"""
        R = [input_embs]
        if generated_embs is not None and generated_embs.numel():
            R.append(generated_embs)
        R = torch.cat(R)
        R_proj = self.trm.atom_proj(R)
        z, y = self.trm.z0, self.trm.y0
        if task_emb is None:
            task_emb = torch.zeros(self.trm.d_in, device=R.device, dtype=R.dtype)
        x_proj = self.trm.task_proj(task_emb)
        for _ in range(self.n):
            z, y, y_out, w = self.trm.step(x_proj, z, y, R_proj, return_attn=True)
        pg = torch.sigmoid(self.p_gen(y))
        vocab_logits = self.vocab_proj(y_out) @ lm_embed.T
        return pg, vocab_logits, w

    def mixed_dist(self, pg, vocab_logits, w, input_ids, n_input):
        """Build mixed output: p_gen * P_vocab + (1-p_gen) * P_copy.

        pg: [1]; vocab_logits: [V]; w: [N+G]; input_ids: [N]; n_input: int.
        Returns [V] probability over vocabulary."""
        device = w.device
        vocab_dist = torch.softmax(vocab_logits, dim=-1)
        input_w = w[:n_input]
        input_w = input_w / (input_w.sum() + 1e-8)
        copy_dist = torch.zeros(self.vocab_size, device=device, dtype=w.dtype)
        copy_dist.scatter_add_(0, input_ids, input_w)
        return pg * vocab_dist + (1 - pg) * copy_dist

    def generate(self, input_embs, input_ids, lm_embed, max_steps=48,
                 eos_token_id=151643, task_emb=None):
        """Greedy decode. Returns generated token IDs."""
        device = input_embs.device
        generated = []
        generated_embs = None
        n_input = input_embs.shape[0]
        if task_emb is None:
            task_emb = torch.zeros(self.trm.d_in, device=device, dtype=torch.float32)
        for _ in range(max_steps):
            pg, vocab_logits, w = self.forward(input_embs, lm_embed, generated_embs, task_emb)
            dist = self.mixed_dist(pg, vocab_logits, w, input_ids, n_input)
            tok = int(dist.argmax())
            if tok == eos_token_id:
                break
            generated.append(tok)
            tok_emb = self.input_proj(lm_embed[tok]).unsqueeze(0)
            generated_embs = tok_emb if generated_embs is None else \
                torch.cat([generated_embs, tok_emb])
        return generated


def _train_trm_pointer_generator(model, train_examples, lm_embed, dev,
                                 epochs=60, lr=3e-4):
    """Train TRMPointerGenerator on (input_ids, target_ids) pairs.

    train_examples: list of (input_ids [N], target_ids [L]).
    Uses teacher forcing with CE on the mixed output distribution."""
    import torch.nn.functional as _F
    torch.manual_seed(0)
    if len(train_examples) < 2:
        return 0.0
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()
    for ep in range(epochs):
        total = 0.0; n_tok = 0
        for input_ids, target_ids in train_examples:
            input_ids = input_ids.to(dev)
            target_ids = target_ids.to(dev)
            n_in = input_ids.shape[0]
            # Pre-compute detached embeddings: input + all target positions
            input_embs = model.input_proj(lm_embed[input_ids]).detach()
            target_embs = model.input_proj(lm_embed[target_ids]).detach()
            for pos in range(target_ids.shape[0]):
                gen = target_embs[:pos]  # [pos, d_in], already detached
                pg, vlog, w = model.forward(input_embs, lm_embed, gen)
                dist = model.mixed_dist(pg, vlog, w, input_ids, n_in)
                loss = _F.cross_entropy(dist.unsqueeze(0), target_ids[pos].unsqueeze(0))
                opt.zero_grad(); loss.backward(); opt.step()
                total += float(loss.detach()); n_tok += 1
        if ep % 10 == 0:
            print(f"      [pg-train] ep {ep} loss={total / max(1, n_tok):.4f}")
    model.eval()
    avg = total / max(1, n_tok)
    print(f"      [pg-train] final loss={avg:.4f}  ({len(train_examples)} examples, {epochs} epochs)")
    return avg


if __name__ == "__main__":
    main()
