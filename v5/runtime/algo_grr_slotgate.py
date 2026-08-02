"""algo_grr_slotgate — selftests + the real A/B for the SPIKING SLOT GATE (trm_wm.SpikingSlotGate).

This is the TRM<->LM coupling, not retrieval. The gate decides, per token position, WHICH working-
memory slots are allowed to inject into the LM's residual stream, using LIF dynamics over the K
slots: lateral inhibition between correlated slots (decorrelation = the anti-collapse prior),
a threshold (discrete addressing), and homeostasis (no slot may dominate the whole sequence).

The failure it targets is the measured one: scaled up, the soft-prompt latent memory suffered
ROUTING COLLAPSE -- it emitted wrong-gadget bodies. Dense softmax over all K slots at every position
has nothing in it that keeps slots distinct.

FALSIFIERS ARE COUPLING METRICS, NOT RANKING METRICS: held-out composition accuracy (where WM
injection is known to matter -- 13-15/16 with WM vs 0/16 ablated) against the same run with the gate
off. Nothing here scores "was the right node retrieved".

    selftest : python -m v5.runtime.algo_grr_slotgate --selftest
    ab       : python -m v5.runtime.algo_grr_slotgate --ab --lm Qwen/Qwen2.5-0.5B-Instruct
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

_ROOT = str(Path(__file__).resolve().parents[2])
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
os.environ.setdefault("HF_HOME", r"E:\cache\hf")

import torch                                                              # noqa: E402

from v5.runtime.trm_wm import GatedCrossAttn, SpikingSlotGate             # noqa: E402


def _selftest() -> bool:
    print("algo_grr_slotgate --selftest: spiking slot gate over the K WM slots\n")
    ok = True

    def chk(t, c, d=""):
        nonlocal ok
        ok &= bool(c)
        print(f"  [{'PASS' if c else 'FAIL'}] {t}{('  - ' + d) if d else ''}")

    torch.manual_seed(0)
    B, S, K, d = 1, 6, 4, 32
    h = torch.randn(B, S, d)
    slots = torch.randn(K, d)

    # [1] PRIOR PRESERVATION — disabled gate must leave GatedCrossAttn bit-identical.
    adapter = GatedCrossAttn(d, n_heads=4, gate_init=0.5)
    off = SpikingSlotGate(enabled=False)
    base = adapter(h, slots)
    via = adapter(h, slots, None, slot_mask=off(h, slots))
    chk("[1] gate disabled -> adapter output bit-identical",
        off(h, slots) is None and torch.equal(base, via))

    # [2] ALL SLOTS FIRING == the dense softmax. theta below the minimum cosine, no inhibition.
    allfire = SpikingSlotGate(enabled=True, theta0=-2.0, w_inh=0.0, alpha=0.0, beta=0.0, T=1)
    m = allfire(h, slots)
    dense = adapter(h, slots, None, slot_mask=m)
    chk("[2] every slot firing reproduces the dense-softmax path",
        m is not None and float(m.min()) == 1.0 and torch.allclose(base, dense, atol=1e-6),
        f"mask all-ones={bool(float(m.min()) == 1.0)}")

    # [3] THRESHOLD really gates: raising theta must reduce how many slots inject -- and to a
    #     PARTIAL set, not zero. Random unit vectors in d=32 have cosines ~N(0, 0.18), so a
    #     threshold of 0.35 silences everything and tests [4]/[9] would then pass trivially by
    #     injecting nothing at all.
    hi = SpikingSlotGate(enabled=True, theta0=0.0, w_inh=0.0, alpha=0.0, beta=0.0, T=1)
    m_hi = hi(h, slots)
    chk("[3] a higher threshold fires FEWER but not zero slots",
        0 < float(m_hi.sum()) < float(m.sum()),
        f"{float(m_hi.sum())} vs {float(m.sum())} of {S*K}")

    # [4] SPARSE INJECTION CHANGES THE OUTPUT — otherwise the gate is decorative.
    sparse = adapter(h, slots, None, slot_mask=m_hi)
    chk("[4] masked injection differs from the dense path",
        not torch.allclose(base, sparse, atol=1e-5),
        f"max|delta| {float((base - sparse).abs().max()):.4f}")

    # [5] NO NaN when a position fires nothing, and that position must receive NO injection.
    #     (h is the adapter's input, so an untouched position comes out exactly as it went in.)
    kill = SpikingSlotGate(enabled=True, theta0=50.0, w_inh=0.0, alpha=0.0, beta=0.0, T=1)
    m0 = kill(h, slots)
    out0 = adapter(h, slots, None, slot_mask=m0)
    chk("[5] all-slots-below-threshold -> no NaN and zero injection",
        float(m0.sum()) == 0.0 and torch.isfinite(out0).all() and torch.allclose(out0, h, atol=1e-6))

    # [6] LATERAL INHIBITION DECORRELATES — with two nearly IDENTICAL slots, inhibition must stop
    #     both firing together at a position. This is the anti-collapse property, stated as a test.
    twin = torch.randn(2, d)
    twin = torch.stack([twin[0], twin[0] + 0.01 * torch.randn(d), twin[1]])   # slots 0,1 near-identical
    hq = twin[0].unsqueeze(0).unsqueeze(0)                                    # a position matching them
    free = SpikingSlotGate(enabled=True, theta0=0.2, w_inh=0.0, alpha=0.0, beta=0.0, T=2)
    inh = SpikingSlotGate(enabled=True, theta0=0.2, w_inh=2.0, alpha=0.0, beta=0.0, T=2)
    mf, mi = free(hq, twin), inh(hq, twin)
    chk("[6] lateral inhibition prevents near-identical slots co-firing (decorrelation)",
        float(mf[0, 0, :2].sum()) == 2.0 and float(mi[0, 0, :2].sum()) < 2.0,
        f"free={mf[0,0].tolist()} inhibited={mi[0,0].tolist()}")

    # [7] HOMEOSTASIS — a slot that keeps winning raises its own bar. With habituation the settled
    #     state must be sparser than with alpha=0 under the SAME drive; otherwise one slot can own
    #     every position, which is the collapse mode.
    hh = twin[0].unsqueeze(0).unsqueeze(0)
    no_hab = SpikingSlotGate(enabled=True, theta0=0.0, w_inh=0.0, alpha=0.0, beta=0.0, T=3)
    hab = SpikingSlotGate(enabled=True, theta0=0.0, w_inh=0.0, alpha=1.5, beta=0.0, T=3)
    chk("[7] homeostasis bounds a dominant slot (theta climbs on repeated firing)",
        float(hab(hh, twin).sum()) < float(no_hab(hh, twin).sum()),
        f"habituated {float(hab(hh, twin).sum())} vs free {float(no_hab(hh, twin).sum())}")

    # [8] PER-POSITION, CONTENT-DEPENDENT: different positions must select different slots, else
    #     this is just a global on/off and the "per-position" claim is empty.
    hp = torch.stack([slots[0], slots[2], slots[1]]).unsqueeze(0)             # each matches one slot
    mp = SpikingSlotGate(enabled=True, theta0=0.5, w_inh=0.0, alpha=0.0, beta=0.0, T=1)(hp, slots)
    rows = {tuple(mp[0, i].tolist()) for i in range(mp.shape[1])}
    chk("[8] the mask varies ACROSS positions (per-position routing)",
        len(rows) > 1, f"{len(rows)} distinct masks over {mp.shape[1]} positions")

    # [9] CLIP-COMPATIBLE: the gate changes WHICH slots are read, never the magnitude law. The
    #     delta must still obey the adapter's own cap.
    cap_adapter = GatedCrossAttn(d, n_heads=4, gate_init=2.0, delta_scale=0.3, delta_mode="clip")
    o_masked = cap_adapter(h, slots, None, slot_mask=m_hi)
    dn = (o_masked - h).norm(dim=-1)
    cap = h.norm(dim=-1) * 0.3 * float(torch.tanh(torch.tensor(2.0))) + 1e-4
    chk("[9] masked injection still respects the delta cap (no renormalisation)",
        bool((dn <= cap).all()), f"max ratio {float((dn / cap).max()):.3f}")

    # [10] TRAINABLE: theta / inhibition / habituation must receive gradient under the surrogate.
    sg = SpikingSlotGate(enabled=True, theta0=0.1, w_inh=0.5, alpha=0.5, spike_mode="surrogate", T=2)
    sg(h, slots).sum().backward()
    chk("[10] surrogate gradient reaches theta0, w_inh and alpha",
        all(p.grad is not None and float(p.grad.abs()) > 0
            for p in (sg.theta0, sg.w_inh, sg.alpha)),
        f"d/dtheta0={float(sg.theta0.grad):+.3f} d/dw_inh={float(sg.w_inh.grad):+.3f} "
        f"d/dalpha={float(sg.alpha.grad):+.3f}")

    print(f"\n  ALGO_GRR_SLOTGATE SELFTEST -> {'PASS' if ok else 'FAIL'}")
    return ok


def _ab(lm: str, epochs: int, n_train: int, n_held: int, theta0: float, w_inh: float) -> bool:
    """The real coupling A/B: identical training run, gate off vs gate on, scored on HELD-OUT
    composition -- the task where WM injection is known to be load-bearing (with-WM 13-15/16 vs
    ablated 0/16). Not a retrieval metric."""
    from v5.runtime import trm_wm as W
    rows = []
    for tag, on in (("gate OFF (dense softmax over all K)", False),
                    (f"gate ON (theta0={theta0}, w_inh={w_inh})", True)):
        print(f"\n=== {tag} ===", flush=True)
        orig_init = W.SpikingSlotGate.__init__

        def patched(self, *a, **kw):
            kw.setdefault("enabled", on)
            kw["enabled"] = on
            kw.setdefault("theta0", theta0)
            kw.setdefault("w_inh", w_inh)
            orig_init(self, *a, **kw)
        W.SpikingSlotGate.__init__ = patched
        try:
            out = W.run_real(lm, epochs=epochs, n_train=n_train, n_held=n_held)
        finally:
            W.SpikingSlotGate.__init__ = orig_init
        rows.append((tag, out))
        print(f"  -> {out}", flush=True)
    print("\n  SUMMARY (held-out composition; ablated arm is the WM-off control)")
    for tag, out in rows:
        print(f"    {tag:<44} {out}")
    return True


def main() -> None:
    ap = argparse.ArgumentParser(description="Spiking slot gate: TRM<->LM coupling.")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--ab", action="store_true")
    ap.add_argument("--lm", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--n-train", type=int, default=48, dest="n_train")
    ap.add_argument("--n-held", type=int, default=16, dest="n_held")
    ap.add_argument("--theta0", type=float, default=0.0)
    ap.add_argument("--w-inh", type=float, default=0.5, dest="w_inh")
    a = ap.parse_args()
    if a.selftest:
        sys.exit(0 if _selftest() else 1)
    if a.ab:
        sys.exit(0 if _ab(a.lm, a.epochs, a.n_train, a.n_held, a.theta0, a.w_inh) else 1)
    ap.print_help()


if __name__ == "__main__":
    main()
