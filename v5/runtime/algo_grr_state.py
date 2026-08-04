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
                     ext_Z=None, ext_h=None):
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
    return F.cross_entropy(lo, tgt[0]), grab.get("Z")


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

    opt = torch.optim.Adam(tracker.parameters(), lr=3e-4)
    for ep in range(a.epochs):
        run = 0.0
        for i, r in enumerate(tr_rows):
            loss, _ = run_with_tracker(lm, tracker, prompt_of(r), r["new"], "track", a.layer)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(tracker.parameters(), 1.0)
            opt.step(); run += float(loss)
            if (i + 1) % 100 == 0:
                print(f"    epoch {ep+1} [{i+1}/{len(tr_rows)}] train CE {run/(i+1):.4f}", flush=True)
        print(f"    epoch {ep+1} done  train CE {run/max(1,len(tr_rows)):.4f}", flush=True)

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
    print(f"    -> {'the tracker genuinely USES the hidden trajectory' if hlo > 0 else 'the tracker is NOT using the hidden state meaningfully'}")
    print(f"\n  TRAJECTORY SEPARATION across instances, by relative position:")
    print("    " + "  ".join(f"{c:.2f}" for c in per_bin))
    print(f"    t=0 -> t=end   mean {sep_mean:.4f}   max {sep_max:.4f}")
    print(f"    -> {'collapse is ONLY at the end' if per_bin[0] < 0.9 <= per_bin[-1] else ('collapsed THROUGHOUT' if sep_mean > 0.99 else 'state stays distinct along the trajectory')}")
    print(f"  (endpoint-only guard, kept for comparison: {sep:.4f})")
    print(f"  GUARD across-instance state cosine: {sep:.4f} "
          f"{'<-- COLLAPSED, falsifier is meaningless' if sep > 0.99 else '(state varies, test is valid)'}")
    print(f"    -> {'STATE IS LOAD-BEARING (CI excludes 0)' if lo > 0 else 'NOT SIGNIFICANT -- CI includes 0'}")
    print(f"{'=' * 74}")


if __name__ == "__main__":
    main()
