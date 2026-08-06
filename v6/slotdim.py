"""SlotDIM v3 (write) + CrossDIM (read). The whole point of v6 is that BOTH sides are DIM.

DIM asks a kinetic question -- "how violently does this input perturb the system?" -- via the
closed-form derivative of an interaction state, rather than attention's relational "how similar is
A to B" via QK^T.

WHY THE READ SIDE MATTERS HERE, measured not assumed. v5's coupling was standard cross-attention
whose delta_mode defaulted to "rescale", forcing EVERY position to receive exactly
delta_scale*||h||. On the corrected task (opaque atom names, code-shaped prompt, so the answer is
not derivable from the prompt) that adapter was NET HARMFUL at both scales:

        Qwen2.5-0.5B    WM 0/32   DERANGED 1/32   ABLATED 4/32
        Qwen2.5-1.5B    WM 3/32   DERANGED 3/32   ABLATED 5/32

The model did BETTER with no working memory. It bought ~5.5 nats of TEACHER-FORCED CE and paid for
it in generation: WM hallucinated truncated names (op_qd, op_pj -- none real) and looped, while
ablated wrote coherent, wrong Python. An unconditional per-position write is exactly what produces
that. CrossDIM's `gate = tanh(obs)` is an ABSOLUTE term: a position that does not perturb slot
space receives NO injection. That is the sparse, content-conditional write cross-attention cannot
express, and it is the one change here that targets a measured failure rather than a hypothesis.

SECOND DERIVATIVE / INTEGRATION (v3). sigma' detects CHANGE; sigma'' detects how fast sensitivity
itself changes (curvature); the running integral accumulates evidence. Pairing them maps onto
"salient event" vs "persistent premise". All three are closed-form for SiLU, so they stay analytic
and cheap -- no autograd, no extra passes.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def silu_d1(z: torch.Tensor) -> torch.Tensor:
    """d/dz silu(z) = s(1 + z(1-s)),  s = sigmoid(z). Verified against autograd to <1e-6."""
    s = torch.sigmoid(z)
    return s * (1.0 + z * (1.0 - s))


def silu_d2(z: torch.Tensor) -> torch.Tensor:
    """d2/dz2 silu(z) = s(1-s)(2 + z(1-2s)).  The CURVATURE of the interaction.

    sigma' answers "does this input move the system"; sigma'' answers "is that sensitivity itself
    changing here" -- i.e. is this a boundary/inflection rather than steady drift. Cheap: reuses s.
    """
    s = torch.sigmoid(z)
    return s * (1.0 - s) * (2.0 + z * (1.0 - 2.0 * s))


def scan_ema(keep: torch.Tensor, b: torch.Tensor, M0: torch.Tensor) -> torch.Tensor:
    """Log-space exact EMA scan: M_t = keep_t * M_{t-1} + b_t, computed in parallel.

    NO 1/K DIVISION. The closed form M = K*(M0 + cumsum(b/K)) has gradient through d(1/K)/dkeep ~
    -1/K^2, which explodes (K ~ 1e-6 at T=128 with keep~0.9) and produced NaN loss from
    AddmmBackward0 in W_f. fp64 only postpones it. Log-space keeps the parallelism and inherits the
    recurrence's boundedness, because 1/K is never materialised as a value the backward pass
    differentiates through.

    keep, b: [T, k, d] (keep may be [T, 1, d] and broadcasts). Returns [T, k, d].
    """
    logK = torch.cumsum(torch.log(keep.clamp(min=1e-3)), dim=0)
    logb = torch.log(b.abs().clamp(min=1e-12))
    S = torch.cumsum(torch.sign(b) * torch.exp(logb - logK), dim=0)
    return torch.exp(logK) * (M0.unsqueeze(0) + S)


class SlotDIMv3(nn.Module):
    """Multi-head slot memory written by a DIM observer. The THINKER'S STATE, not the LM interface.

    Per head: interaction Z = x @ M0^T, observer obs = sigma'(Z), competition A = softmax(obs),
    absolute gate = tanh(obs), input-dependent decay lam = sigmoid(W_f(x)), write
    w = A * gate * silu(W_v(x)), memory by exact EMA scan.

    v3 adds, both optional and both zero-cost when off:
      curvature  -- sigma''(Z) as a SECOND, independently-gated write channel. Detects inflection
                    rather than drift, so a premise that flips the reasoning writes hard while
                    steady context does not.
      integral   -- the running mean of the observer, a low-pass complement to the derivative's
                    high-pass. "Persistent premise" vs "salient event".
    """

    def __init__(self, d: int, k: int = 8, n_heads: int = 2, decay_bias: float = 1.0,
                 ffn: bool = True, curvature: bool = True, integral: bool = True):
        super().__init__()
        assert k % n_heads == 0, f"k={k} must divide by n_heads={n_heads}"
        self.h, self.ks = n_heads, k // n_heads
        self.curvature, self.integral = curvature, integral
        self.M0 = nn.Parameter(torch.randn(self.h, self.ks, d) * 0.02)
        self.W_v = nn.ModuleList([nn.Linear(d, d, bias=False) for _ in range(self.h)])
        self.W_f = nn.ModuleList([nn.Linear(d, d) for _ in range(self.h)])
        # DECAY BIAS 1.0, NOT the notebook's 2.2. keep = sigmoid(decay_bias), and with T write
        # events M ~ keep*M0 + (1-keep)*writes. The notebook runs T=128 TOKENS, where keep=0.90 is
        # sensible. This thinker writes 1-5 EVENTS, where keep=0.90 means the task contributes ~10%
        # and the SHARED M0 dominates -- measured across-task slot cosine at T=1/3/8:
        #     decay_bias 2.2 (keep .90):  0.9551  0.6706  0.1889   <- identical slots, dead falsifier
        #     decay_bias 1.0 (keep .73):  0.6834  0.1391  0.0083
        #     decay_bias 0.0 (keep .50):  0.6228  0.1112  0.0094
        # Lowering decay alone is not enough at T=1 (0.62); the seed must also carry more than one
        # event, which is why run.py prefetches retrieved nodes into the seed.
        for W in self.W_f:                      # keep ~ sigmoid(1.0) ~ 0.73 at init
            W.weight.data.mul_(0.1)
            W.bias.data.fill_(decay_bias)
        # curvature and integral channels are ZERO-INIT so an untrained v3 is exactly v2. Every
        # added component in this project must start as a strict no-op and earn its contribution.
        self.w_curv = nn.Parameter(torch.zeros(self.h)) if curvature else None
        self.w_integ = nn.Parameter(torch.zeros(self.h)) if integral else None
        self.ln = nn.LayerNorm(d) if ffn else None
        self.ffn = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d)) if ffn else None

    def forward(self, x: torch.Tensor, return_obs: bool = False):
        """x [T, d] write events -> slot memory [k, d] (heads concatenated)."""
        if x.dim() == 1:
            x = x.unsqueeze(0)
        outs, obs_all = [], []
        for i in range(self.h):
            M0 = self.M0[i]                                   # (ks, d)
            Z = x @ M0.T                                      # (T, ks)
            obs = silu_d1(Z)
            A = torch.softmax(obs, dim=-1)                    # RELATIVE: slots compete
            gate = torch.tanh(obs)                            # ABSOLUTE: no perturbation, no write
            if self.curvature:
                gate = gate + self.w_curv[i] * torch.tanh(silu_d2(Z))
            if self.integral:
                run = torch.cumsum(obs, dim=0) / torch.arange(
                    1, x.shape[0] + 1, device=x.device, dtype=obs.dtype).unsqueeze(-1)
                gate = gate + self.w_integ[i] * torch.tanh(run)
            lam = torch.sigmoid(self.W_f[i](x))
            keep = lam.unsqueeze(1).clamp(min=0.7, max=1.0)   # bounds logK over the window
            v = F.silu(self.W_v[i](x))
            w = A.unsqueeze(-1) * gate.unsqueeze(-1) * v.unsqueeze(1)
            M_seq = scan_ema(keep, (1.0 - keep) * w, M0)
            outs.append(M_seq[-1])
            obs_all.append(obs)
        M = torch.cat(outs, 0)                                # (k, d)
        if self.ffn is not None:
            M = self.ln(M)
            M = M + self.ffn(M)
        return (M, torch.cat(obs_all, dim=-1)) if return_obs else M


class CrossDIM(nn.Module):
    """The LM READS the slots by the same kinetic question it wrote them with.

    delta = (softmax_k(obs) * tanh(obs)) @ M,  obs = sigma'(h @ M^T)

    RATIONALE, third and correct version. The history matters because two of the three were wrong.

      v1 (overclaim, unmeasured): "softmax sums to 1 so attention injects at every position".
      v2 (over-retraction, WRONG METRIC): I measured the aligned/random RATIO, got CrossDIM 1.62x
         vs attention 2.53x, and concluded attention was more content-conditional.
      v3 (measured, correct): RATIO IS THE WRONG QUANTITY. What damages a residual stream is the
         NOISE FLOOR on positions that have nothing to do with memory, because h + delta does not
         care about contrast. And there the difference is structural, not empirical:

            at Z = 0 (a position with NO interaction with memory)
                attention   |output| = 0.339956   -- injects the mean slot row anyway
                CrossDIM    |output| = 0.000000   -- exactly nothing

            min |output| over 4000 random interaction states
                attention   0.3518   (never goes below ~0.35)
                CrossDIM    0.0645

         softmax(Z) @ M is a CONVEX COMBINATION of the slot rows, so its norm is bounded below by
         the min-norm point of conv(M). Attention CANNOT output zero. The absolute gate has no such
         lower bound. Post-clip, on the same inputs, the noise floor on non-interacting positions
         was 0.0093*||h|| for attention vs 0.0026*||h|| for CrossDIM -- 3.6x.

         Attention's higher ratio is bought with a floor it cannot lower: "found it, inject
         everything". CrossDIM injects only the difference the interaction actually makes, and
         nothing when it makes none. For pushing an external graph / slot memory into a frozen LM,
         protecting the residual stream is the requirement; a high ratio on top of a high floor is
         not.

    This does NOT retire the v5 diagnosis: delta_mode="rescale" forced every position to exactly
    delta_scale*||h||, which is a worse version of the same disease and was the default. The
    honest baseline for any CrossDIM claim remains attention WITH CLIP, which has never been run.

    OLD NOTE, kept so the reasoning is auditable:
    The original claim here was: "softmax sums to 1, so attention necessarily injects SOMETHING at
    every position, whereas tanh(obs) is an ABSOLUTE gate that lets non-interacting positions pass
    through untouched." THAT CLAIM IS FALSE. Measured at init, magnitude of the pre-projection read
    for positions ALIGNED to a slot vs random positions:

        CrossDIM  (softmax * tanh(obs - theta))   0.1506 vs 0.0930   ratio 1.62x
        attention (softmax only)                  0.9969 vs 0.3947   ratio 2.53x

    Plain attention is MORE content-conditional, not less: softmax(Z) @ M gives a near-unit vector
    when Z is peaked and a much smaller one when Z is flat, because near-orthogonal slot rows
    cancel under averaging. Attention already carries a magnitude signal through its concentration.

    What v5 actually did wrong was almost certainly delta_mode="rescale", which OVERWRITES that
    signal by forcing every position to exactly delta_scale*||h||. That is a genuinely
    unconditional write, and it was the default and never tested against clip.

    So CrossDIM is NOT justified yet. It is kept because it is the DIM-symmetric read and worth an
    arm, but the honest baseline to beat is plain cross-attention with CLIP, not with rescale, and
    that arm has never been run. Do not describe CrossDIM as fixing the unconditional write until
    an A/B says so.

    theta centres the gate at sigma'(0) = 0.5 so a Z=0 position scores 0 rather than tanh(0.5)~0.46.
    That is a real improvement over the first version but it does not rescue the claim above.

    out_gate is zero-init (tanh(0)=0) so an untrained CrossDIM is EXACTLY the frozen LM -- the
    ablated arm is then a true incumbent rather than an approximation of one.
    The residual is CLIPPED, never rescaled: rescaling forces every position to receive exactly
    cap*||h||, which makes per-position emphasis unrepresentable and lets the module learn a format
    prior that survives its own ablation.
    """

    def __init__(self, d_lm: int, d_slot: int, cap: float = 0.15, gate_init: float = 0.05):
        super().__init__()
        self.cap = cap
        self.to_slot = nn.Linear(d_lm, d_slot, bias=False)    # no bias: bias/signal ~1 collapsed
        self.from_slot = nn.Linear(d_slot, d_lm, bias=False)  # separation 3x in v5
        # DO NOT zero-init this. The gate g is already zero, so tanh(g)=0 makes the module an exact
        # identity on its own. Zeroing from_slot TOO creates a double-zero deadlock: delta==0 kills
        # the gradient on g, and tanh(g)==0 kills the gradient on from_slot, so EVERY parameter --
        # reader, thinker, seed projection -- receives exactly 0.000e+00 and nothing ever trains.
        # Measured: 21/24 params with grad, 0 nonzero, max|grad| 0.0. That is a dead channel, and it
        # would have produced a silent null indistinguishable from "the idea does not work".
        # v5's GatedCrossAttn had this right: eye-init the value/output path, zero ONLY the gate.
        nn.init.normal_(self.from_slot.weight, std=0.02)
        # gate_init is SMALL BUT NON-ZERO on purpose. At exactly 0, tanh(g)=0 blocks the gradient
        # to delta and therefore to the slot table M, so the THINKER receives 0.000e+00 on every
        # parameter until g drifts off zero -- measured: reader 1/4 params live, thinker 0/16,
        # seed_proj 0/1. That is a warm-up chain, not a deadlock, but it wastes most of a short run.
        # This does NOT weaken the ablated control: that arm passes M=None, the hook returns early,
        # and the LM is untouched regardless of gate_init. 0.05 sits below the 0.1-0.25 band v5
        # documented as usable, so the adapter still has to earn its way up.
        self.g = nn.Parameter(torch.full((1,), gate_init))
        # THRESHOLD, and it is load-bearing. sigma'(0) = 0.5, NOT 0 -- so a position with no
        # interaction at all (Z=0) would still score tanh(0.5) ~ 0.46 and receive an injection.
        # Measured: read_sparsity was 0.000, i.e. the "absolute gate" injected at EVERY position,
        # which is precisely the unconditional write it exists to avoid. Centring at sigma'(0)
        # makes Z=0 -> gate 0, so non-interacting positions pass through untouched at init and the
        # module can still learn to open wider. Learnable: the right threshold is empirical.
        self.theta = nn.Parameter(torch.full((1,), float(silu_d1(torch.zeros(1)))))

    def forward(self, h: torch.Tensor, M: torch.Tensor) -> torch.Tensor:
        """h [B, S, d_lm] LM hidden states; M [k, d_slot] slot memory -> h + gated delta."""
        q = self.to_slot(h)                                   # (B, S, d_slot)
        Z = q @ M.T                                           # (B, S, k) interaction
        obs = silu_d1(Z)
        A = torch.softmax(obs, dim=-1)                        # which slot
        gate = torch.tanh(obs - self.theta)                   # whether to read AT ALL
        delta = self.from_slot((A * gate) @ M)                # (B, S, d_lm)
        cap = h.norm(dim=-1, keepdim=True) * self.cap
        dn = delta.norm(dim=-1, keepdim=True) + 1e-6
        delta = delta * torch.clamp(cap / dn, max=1.0)        # CLIP, never rescale
        return h + torch.tanh(self.g) * delta

    @torch.no_grad()
    def read_sparsity(self, h: torch.Tensor, M: torch.Tensor) -> float:
        """Fraction of positions where EVERY slot is gated off.

        NOTE THIS IS A WEAK METRIC and it misled once already: it takes a max over slots, so it only
        fires when all k slots are simultaneously inactive, and it reads 0.000 even when the gate is
        doing real per-position work. For "is the read content-conditional", compare the magnitude
        of the pre-projection read between positions that interact with a slot and positions that do
        not -- and compare it against plain attention, which also varies (2.53x at init).
        """
        Z = self.to_slot(h) @ M.T
        g = torch.tanh(silu_d1(Z) - self.theta).abs().max(dim=-1).values
        return float((g < 0.05).float().mean())
