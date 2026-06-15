"""Projector-from-TEXT test: can a TRAINED map turn node TEXT into the operator vectors,
preserving the effect — or does it COLLAPSE TO GENERIC like every prior injection mechanism?

The floor test (optest_invalidate) proved the operator algebra works when the steering vectors
come from clean in-context grounding shifts. The full system needs a PROJECTOR: node text ->
embedding (offline) -> a vector in the LM's hidden space. This is exactly where generic-collapse
lived before. The decisive comparison:

  CEILING   : apply the REAL in-context shift v*            (what we know works)
  PROJECTOR : apply P(embed(node_text))  (trained)          (the full pipeline)
  GENERIC   : apply mean(train v*)        (node-blind)       (the collapse baseline)

PASS = on HELD-OUT facts, PROJECTOR reproduces the operator effect (assert raises the right answer,
INVALIDATE cancels) AND beats GENERIC (proves it carries node identity from text, didn't collapse).
FAIL = projector ~= generic -> the trained map can't carry node-specific content -> the wall wins.

Local (cached 1.5B):  python -m v5.optest_projector --model Qwen/Qwen2.5-1.5B --layer 14 --alpha 4
"""
from __future__ import annotations

import argparse
import os
import torch
import torch.nn as nn

from v5.lm_loader import load_frozen_lm

# country -> capital (enough to train a projector + hold out)
CAPITALS = {
    "France": "Paris", "Japan": "Tokyo", "Italy": "Rome", "Spain": "Madrid", "Germany": "Berlin",
    "Russia": "Moscow", "China": "Beijing", "Egypt": "Cairo", "Canada": "Ottawa", "Brazil": "Brasilia",
    "India": "Delhi", "Greece": "Athens", "Turkey": "Ankara", "Poland": "Warsaw", "Austria": "Vienna",
    "Norway": "Oslo", "Sweden": "Stockholm", "Finland": "Helsinki", "Portugal": "Lisbon", "Ireland": "Dublin",
    "Mexico": "Mexico", "Peru": "Lima", "Chile": "Santiago", "Cuba": "Havana", "Kenya": "Nairobi",
    "Iran": "Tehran", "Iraq": "Baghdad", "Israel": "Jerusalem", "Thailand": "Bangkok", "Vietnam": "Hanoi",
    "Hungary": "Budapest", "Belgium": "Brussels", "Denmark": "Copenhagen", "Switzerland": "Bern",
    "Netherlands": "Amsterdam", "Argentina": "Buenos", "Colombia": "Bogota", "Morocco": "Rabat",
    "Nigeria": "Abuja", "Ukraine": "Kyiv", "Romania": "Bucharest", "Cambodia": "Phnom",
    "Mongolia": "Ulaanbaatar", "Nepal": "Kathmandu", "Pakistan": "Islamabad", "Indonesia": "Jakarta",
    "Philippines": "Manila", "Malaysia": "Kuala", "Lebanon": "Beirut", "Jordan": "Amman",
}


def _layers(model):
    m = model
    for a in ("model", "layers"):
        m = getattr(m, a)
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B")
    ap.add_argument("--layer", type=int, default=14)
    ap.add_argument("--alpha", type=float, default=4.0)
    ap.add_argument("--epochs", type=int, default=400)
    ap.add_argument("--holdout", type=int, default=12)
    a = ap.parse_args()
    trust = os.environ.get("V5_LM_TRUST_REMOTE_CODE", "0").lower() in ("1", "true", "yes")
    from transformers import AutoTokenizer
    model = load_frozen_lm(a.model); model.eval()
    tok = AutoTokenizer.from_pretrained(a.model, trust_remote_code=trust)
    dev = next(model.parameters()).device
    layers = _layers(model); L = a.layer
    emb_in = model.get_input_embeddings()
    print(f"loaded | {len(layers)} layers | layer {L} | dev {dev}", flush=True)

    cap = {"h": None}; steer = {"v": None}

    def hook(mod, inp, out):
        is_tup = isinstance(out, tuple); h = out[0] if is_tup else out
        cap["h"] = h.detach()
        if steer["v"] is not None:
            h = h + steer["v"].to(h.dtype)
            return ((h,) + tuple(out[1:])) if is_tup else h
        return out
    layers[L].register_forward_hook(hook)

    @torch.no_grad()
    def hlast(text):
        steer["v"] = None; cap["h"] = None
        ids = tok(text, return_tensors="pt").input_ids.to(dev); model(ids)
        return cap["h"][0, -1].float()

    @torch.no_grad()
    def hmean(text):
        steer["v"] = None; cap["h"] = None
        ids = tok(text, return_tensors="pt").input_ids.to(dev); model(ids)
        return cap["h"][0].mean(0).float()           # mean layer-L hidden (rich semantics)

    @torch.no_grad()
    def belief(prompt, true_c, false_c, v):
        steer["v"] = v
        ids = tok(prompt, return_tensors="pt").input_ids.to(dev)
        lg = model(ids).logits[0, -1].float(); steer["v"] = None
        ti = tok(" " + true_c, add_special_tokens=False).input_ids[0]
        fi = tok(" " + false_c, add_special_tokens=False).input_ids[0]
        return float(lg[fi] - lg[ti])

    # build per-fact targets (in-context shift v*) + node-text embeddings
    caps = list(CAPITALS.items())
    data = []
    print("building targets...", flush=True)
    for i, (country, capital) in enumerate(caps):
        false_c = caps[(i + 7) % len(caps)][1]       # a different capital
        if false_c == capital:
            continue
        prompt = f"The capital of {country} is the city of"
        assert_ctx = f"Fact: The capital of {country} is {false_c}.\n{prompt}"
        inval_ctx = f"Fact: It is FALSE that the capital of {country} is {false_c}; it is {capital}.\n{prompt}"
        base = hlast(prompt)
        va = hlast(assert_ctx) - base
        vi = hlast(inval_ctx) - base
        ea = hlast(f"The capital of {country} is {false_c}.")
        ei = hlast(f"It is FALSE that the capital of {country} is {false_c}; it is {capital}.")
        data.append(dict(prompt=prompt, true=capital, false=false_c, va=va, vi=vi, ea=ea, ei=ei))
    print(f"facts: {len(data)}", flush=True)

    tr, te = data[a.holdout:], data[:a.holdout]
    d = data[0]["va"].shape[0]; ein = data[0]["ea"].shape[0]
    P = nn.Sequential(nn.Dropout(0.2), nn.Linear(ein, 256), nn.GELU(),
                      nn.Dropout(0.2), nn.Linear(256, d)).to(dev)
    opt = torch.optim.AdamW(P.parameters(), lr=1e-3, weight_decay=1e-2)   # regularize -> less overfit
    X = torch.stack([x["ea"] for x in tr] + [x["ei"] for x in tr]).to(dev)
    Y = torch.stack([x["va"] for x in tr] + [x["vi"] for x in tr]).to(dev)
    Xh = torch.stack([x["ea"] for x in te] + [x["ei"] for x in te]).to(dev)
    Yh = torch.stack([x["va"] for x in te] + [x["vi"] for x in te]).to(dev)
    for ep in range(a.epochs):
        P.train(); opt.zero_grad(); loss = ((P(X) - Y) ** 2).mean(); loss.backward(); opt.step()
    P.eval()
    with torch.no_grad():
        print(f"projector MSE  train {((P(X)-Y)**2).mean():.3f}  heldout {((P(Xh)-Yh)**2).mean():.3f}", flush=True)

    gen_a = torch.stack([x["va"] for x in tr]).mean(0)   # generic baseline (node-blind)
    gen_i = torch.stack([x["vi"] for x in tr]).mean(0)

    # held-out: ceiling vs projector vs generic — assert fires? invalidate cancels?
    def evalset(name, get_a, get_i):
        af = ic = 0; sc = si = 0.0
        for x in te:
            va = get_a(x) * a.alpha; vi = get_i(x) * a.alpha
            c = belief(x["prompt"], x["true"], x["false"], None)
            asr = belief(x["prompt"], x["true"], x["false"], va)
            edg = belief(x["prompt"], x["true"], x["false"], va - vi)
            af += int(asr > c); ic += int(edg < asr); sc += (asr - c); si += (asr - edg)
        n = len(te)
        print(f"  {name:10} assert-fires {af}/{n} (mean +{sc/n:.2f}) | invalidate-cancels {ic}/{n} (mean +{si/n:.2f})")
        return af, ic, sc / n
    with torch.no_grad():
        print("\n=== held-out projector-from-text test ===")
        ce = evalset("CEILING", lambda x: x["va"], lambda x: x["vi"])
        pr = evalset("PROJECTOR", lambda x: P(x["ea"]), lambda x: P(x["ei"]))
        ge = evalset("GENERIC", lambda x: gen_a, lambda x: gen_i)
    n = len(te)
    ok = pr[0] >= 0.7 * n and pr[1] >= 0.6 * n and pr[2] > ge[2] + 0.3
    print(f"\n  PROJECTOR vs GENERIC assert-strength: {pr[2]:.2f} vs {ge[2]:.2f}")
    print(f"  RESULT: {'PASS — projector carries node identity from TEXT (assert fires, invalidate cancels, beats generic)' if ok else 'FAIL — projector ~= generic (collapsed; cannot carry node-specific content from text)'}")


if __name__ == "__main__":
    main()
