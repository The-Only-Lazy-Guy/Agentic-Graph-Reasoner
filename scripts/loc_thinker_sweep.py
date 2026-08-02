"""Thinker capacity sweep on the 8-channel features (n=1725, cached).

Justified by measurement, not appetite: the 468-param thinker LOST at n=300 (held-R 0.1515) and WON
at n=1725 (0.3503), so capacity was data-limited rather than wrong. This asks how far that goes and
whether channel INTERACTIONS (attention) beat independent per-channel gains.

Every variant is a RESIDUAL ON THE INCUMBENT: output layers are zero-init and the weight is
w_global * exp(delta), so exp(0)=1 reproduces the fused scorer exactly and the incumbent is the floor
by construction. Every variant is early-stopped on a slice of train held back for it -- four learned
components in this project degraded their own baseline by training to convergence on ~200 instances.
"""
import os, sys, json, pickle, random, collections
os.environ.setdefault("HF_HOME", r"E:\cache\hf")
sys.path.insert(0, r"E:\PROJECT\graph_v5")
import numpy as np
import torch
import torch.nn as nn
from v5.runtime.membrane import channel_stats, N_STAT

A = r"E:\PROJECT\graph_v5\artifacts"
F = pickle.load(open(rf"{A}\loc_feats.pkl", "rb"))
rows = json.load(open(rf"{A}\swebench_loc_big.json", encoding="utf-8"))
N_CH = 8
HELD = ("pytest-dev/pytest", "sphinx-doc/sphinx")
hr = [r for r in rows if r["repo"] in HELD and r["instance_id"] in F]
rest = [r for r in rows if r["repo"] not in HELD and r["instance_id"] in F]
random.Random(0).shuffle(rest)
hi, tr = rest[:300], rest[300:]
print(f"  train {len(tr)} | held-I {len(hi)} | held-R {len(hr)}")

_w = torch.zeros(N_CH, requires_grad=True)
_o = torch.optim.Adam([_w], lr=0.05)
_TR = [(torch.tensor(F[r["instance_id"]][0]), F[r["instance_id"]][1]) for r in tr]
for _ in range(120):
    _o.zero_grad()
    torch.stack([nn.functional.cross_entropy((_w @ M).unsqueeze(0), torch.tensor([g]))
                 for M, g in _TR]).mean().backward()
    _o.step()
WG = _w.detach()
print(f"  fused global weights {np.round(WG.numpy(), 3)}")

# per-instance stats cached once -- recomputing them per epoch dominated the loop
STAT = {k: torch.tensor(channel_stats(v[0]), dtype=torch.float32) for k, v in F.items()}


class Tiny(nn.Module):
    """One confidence gain + one persistent-trace gain per channel. 16 params at 8 channels."""
    def __init__(self):
        super().__init__()
        self.a = nn.Parameter(torch.zeros(N_CH)); self.g = nn.Parameter(torch.zeros(N_CH))
    def forward(self, st, th):
        return WG * torch.exp((self.a * st.view(N_CH, N_STAT)[:, 1] + self.g * th).clamp(-3, 3))


class MLP(nn.Module):
    """Channels are independent inputs to a shared hidden layer."""
    def __init__(self, h):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(N_CH * N_STAT + N_CH, h), nn.Tanh(), nn.Linear(h, N_CH))
        nn.init.zeros_(self.net[2].weight); nn.init.zeros_(self.net[2].bias)
    def forward(self, st, th):
        return WG * torch.exp(self.net(torch.cat([st, th])).clamp(-3, 3))


class Attn(nn.Module):
    """Each CHANNEL is a token (its own distribution stats + its trace); one self-attention layer
    lets the decision depend on how channels compare -- 'sym is sharp AND content is flat' is a
    relation between channels that an MLP over a flat concatenation has to memorise positionally."""
    def __init__(self, d=32, heads=2):
        super().__init__()
        self.inp = nn.Linear(N_STAT + 1, d)
        self.att = nn.MultiheadAttention(d, heads, batch_first=True)
        self.norm = nn.LayerNorm(d)
        self.out = nn.Linear(d, 1)
        nn.init.zeros_(self.out.weight); nn.init.zeros_(self.out.bias)
    def forward(self, st, th):
        tok = torch.cat([st.view(N_CH, N_STAT), th.view(N_CH, 1)], dim=1)
        h = self.inp(tok).unsqueeze(0)
        a, _ = self.att(h, h, h)
        d = self.out(self.norm(h + a)).squeeze(0).squeeze(-1)
        return WG * torch.exp(d.clamp(-3, 3))


def run(m, split, opt=None, beta=0.15):
    th, hit = {}, 0
    for r in sorted(split, key=lambda x: x["repo"]):
        M, gi, repo = F[r["instance_id"]][0], F[r["instance_id"]][1], F[r["instance_id"]][2]
        t = th.get(repo, torch.zeros(N_CH))
        w = m(STAT[r["instance_id"]], t)
        lg = w @ torch.tensor(M)
        if opt is not None:
            loss = nn.functional.cross_entropy(lg.unsqueeze(0), torch.tensor([gi]))
            opt.zero_grad(); loss.backward(); opt.step()
        th[repo] = (1 - beta) * t + beta * w.detach()
        hit += int(lg.argmax()) == gi
    return hit / max(1, len(split))


print(f"\n  variant            params    train    held-I    held-R")
print(f"    {'fused (no thinker)':<18} {0:>6}   "
      f"{run(lambda s, t: WG, tr):.4f}   {run(lambda s, t: WG, hi):.4f}   {run(lambda s, t: WG, hr):.4f}")
for name, mk in (("Tiny", Tiny), ("MLP-16", lambda: MLP(16)), ("MLP-64", lambda: MLP(64)),
                 ("Attn-32", Attn)):
    torch.manual_seed(0)
    m = mk()
    opt = torch.optim.Adam(m.parameters(), lr=3e-3)
    vs = max(60, len(tr) // 5)
    fit, val = tr[vs:], tr[:vs]
    best, bs, bad = -1, None, 0
    for ep in range(40):
        run(m, fit, opt)
        v = run(m, val)
        if v > best:
            best, bs, bad = v, {k: x.clone() for k, x in m.state_dict().items()}, 0
        else:
            bad += 1
        if bad >= 8:
            break
    if bs:
        m.load_state_dict(bs)
    m.eval()
    with torch.no_grad():
        print(f"    {name:<18} {sum(p.numel() for p in m.parameters()):>6}   "
              f"{run(m, tr):.4f}   {run(m, hi):.4f}   {run(m, hr):.4f}")
