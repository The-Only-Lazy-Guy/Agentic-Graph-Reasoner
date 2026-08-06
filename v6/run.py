"""v6 harness. The falsifier is built in from the start, not bolted on afterwards.

Arms, and why each exists:
  ablated    frozen LM, no injection. A TRUE incumbent: CrossDIM is zero-init so untrained == the
             LM exactly. This is the arm v5 got wrong -- its prompt pinned it at 0, which made a
             format effect read as a memory win for months.
  wm         the thinker's slots injected.
  deranged   another task's slots, same prompt, same tests. The only arm that separates "the LM
             reads THESE slots" from "any slots unlock a mode". With opaque atom names a deranged
             table supplies provably wrong identifiers, so this finally has teeth.

Read-path arms: --read {crossdim, attn-clip, attn-rescale}. attn-rescale is v5's default and is
included so the regression stays visible rather than argued; attn-clip is the honest baseline
CrossDIM has to beat, and it has never been run.

Reported together, always: pass counts AND teacher-forced CE. A 32-item pass rate cannot resolve
effects the size this project deals with (the whole adapter-shortcut result is stated in nats),
and CE is forward-only so it costs nothing. Bootstrap is over INSTANCES, never over anchors.
"""
from __future__ import annotations

import argparse
import math
import os
import random
import sys
from pathlib import Path

_ROOT = str(Path(__file__).resolve().parents[1])
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
os.environ.setdefault("HF_HOME", r"E:\cache\hf")

import numpy as np
import torch
import torch.nn as nn

from v6.graph import build_graph, content_embeddings, node_embeddings
from v6.slotdim import CrossDIM
from v6.tasks import make_tasks, make_tests, prompt_for, verify
from v6.thinker import Thinker


class AttnRead(nn.Module):
    """Plain cross-attention read, for the baseline arms. mode='clip' vs 'rescale'.

    rescale forces EVERY position to exactly cap*||h||, which destroys per-position emphasis and is
    what v5 shipped as its default. clip bounds the same budget but lets quiet positions stay quiet.
    """

    def __init__(self, d_lm: int, d_slot: int, cap: float = 0.15, mode: str = "clip"):
        super().__init__()
        self.cap, self.mode = cap, mode
        self.q = nn.Linear(d_lm, d_slot, bias=False)
        self.o = nn.Linear(d_slot, d_lm, bias=False)
        # NOT zero-init -- see CrossDIM. tanh(g)=0 already makes this an exact identity; zeroing o
        # as well deadlocks both gradients and the arm trains nothing.
        nn.init.normal_(self.o.weight, std=0.02)
        self.g = nn.Parameter(torch.full((1,), 0.05))    # see CrossDIM: 0 blocks grad to the thinker

    def forward(self, h, M):
        att = torch.softmax(self.q(h) @ M.T, dim=-1)
        delta = self.o(att @ M)
        cap = h.norm(dim=-1, keepdim=True) * self.cap
        dn = delta.norm(dim=-1, keepdim=True) + 1e-6
        delta = delta * torch.clamp(cap / dn, max=1.0) if self.mode == "clip" else delta / dn * cap
        return h + torch.tanh(self.g) * delta


def build_reader(kind: str, d_lm: int, d_slot: int):
    if kind == "crossdim":
        return CrossDIM(d_lm, d_slot)
    return AttnRead(d_lm, d_slot, mode=("rescale" if kind == "attn-rescale" else "clip"))


def main():
    ap = argparse.ArgumentParser(description="v6: thinker slots -> LM, with the falsifier built in")
    ap.add_argument("--lm", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--read", default="crossdim",
                    choices=["crossdim", "attn-clip", "attn-rescale"])
    ap.add_argument("--n-train", type=int, default=200)
    ap.add_argument("--n-held", type=int, default=200)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--k-slots", type=int, default=8)
    ap.add_argument("--heads", type=int, default=2)
    ap.add_argument("--steps", type=int, default=2, help="thinker retrieval steps")
    ap.add_argument("--retrieve", default="always", choices=["always", "gated"])
    ap.add_argument("--prefetch", type=int, default=4,
                    help="cosine-retrieved nodes seeded into the slots before the learned gate runs")
    ap.add_argument("--max-new", type=int, default=24)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    torch.manual_seed(a.seed); random.seed(a.seed)
    from v5.runtime.dcpd_latent import WhiteBox
    wb = WhiteBox(a.lm, quant="4bit")
    d_lm = wb.d_model
    for p in wb.model.parameters():
        p.requires_grad_(False)

    tr, hd, descs, codes = make_tasks(a.n_train, a.n_held, seed=a.seed)
    g = build_graph(descs)
    names = list(descs)
    nemb = torch.as_tensor(node_embeddings(g, names), dtype=torch.float32, device=wb.device)
    print(f"  LM {a.lm}  d={d_lm}  layers={wb.n_layers}  read={a.read}")
    print(f"  {len(descs)} atoms | {len(tr)} train / {len(hd)} held pairs (disjoint, pool shared)")
    print(f"  graph: {len(g)} nodes, worlds={len(getattr(g, 'worlds', {}) or {})}")

    # SLOTS LIVE IN d_lm. Retrieval is MiniLM (descriptions, 0.3886 separation); what gets written
    # into a slot is the node's NAME through the LM's own embedding table. That removes the
    # cross-model bridge entirely -- the slot already speaks the LM's language, so CrossDIM has
    # nothing to translate and CE has no bridge to collapse.
    d_slot = d_lm
    cemb = content_embeddings(wb, names).to(wb.device)          # [N, d_lm] native name rows
    thinker = Thinker(d_slot, k=a.k_slots, n_heads=a.heads, n_steps=a.steps).to(wb.device)
    reader = build_reader(a.read, d_lm, d_slot).to(wb.device)
    seed_proj = nn.Linear(384, d_slot, bias=False).to(wb.device)   # task text (MiniLM) -> slot space
    params = list(thinker.parameters()) + list(reader.parameters()) + list(seed_proj.parameters())
    print(f"  thinker {sum(p.numel() for p in thinker.parameters()):,} params | "
          f"reader {sum(p.numel() for p in reader.parameters()):,}\n", flush=True)

    from embedder import encode_batch
    layer = wb.n_layers - 2
    slots_now: dict = {"M": None}

    def hook(_m, _i, out):
        if slots_now["M"] is None:
            return None
        hs = out[0] if isinstance(out, tuple) else out
        h2 = hs.clone()
        h2[0] = reader(hs[0].float().unsqueeze(0), slots_now["M"]).squeeze(0).to(hs.dtype)
        return (h2,) + tuple(out[1:]) if isinstance(out, tuple) else h2

    handle = wb.model.model.layers[layer].register_forward_hook(hook)

    nemb_n = nemb / nemb.norm(dim=-1, keepdim=True).clamp_min(1e-6)

    def slots_for(text):
        """Seed = task embedding PLUS a cheap cosine PREFETCH of graph nodes.

        One write event is not enough: measured across-task slot cosine at T=1 is 0.62 even with
        decay lowered, because M0 is shared. Worse, the learned gate's first query is built from
        that collapsed state, so every task would retrieve the SAME nodes and the deranged arm
        would be dead by construction. The first hop is therefore plain cosine (no learned gate
        needed to look up what the task literally describes); the learned gate does the multi-hop
        refinement on top of an already task-specific state.
        """
        s = torch.as_tensor(encode_batch([text])[0], dtype=torch.float32, device=wb.device)
        pre = torch.topk(nemb_n @ (s / s.norm().clamp_min(1e-6)), k=a.prefetch).indices
        # write events: the task (projected) + the NATIVE name rows of the prefetched nodes
        seed = torch.cat([seed_proj(s).unsqueeze(0), cemb[pre]], dim=0)
        M, _ = thinker(seed, nemb, content=cemb, mode=a.retrieve)
        return M

    def ids_for(text, target=None):
        p = wb.tok(prompt_for(text), return_tensors="pt").input_ids.to(wb.device)
        if target is None:
            return p, None
        t = wb.tok(target, return_tensors="pt", add_special_tokens=False).input_ids.to(wb.device)
        return p, t

    opt = torch.optim.Adam(params, lr=1e-3)
    for ep in range(a.epochs):
        random.shuffle(tr); tot = 0.0
        for i, (text, atoms, expr) in enumerate(tr):
            p, t = ids_for(text, expr)
            slots_now["M"] = slots_for(text)
            out = wb.model(input_ids=torch.cat([p, t], -1),
                           labels=torch.cat([torch.full_like(p, -100), t], -1))
            slots_now["M"] = None
            opt.zero_grad(); out.loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step(); tot += float(out.loss)
            if (i + 1) % 50 == 0:
                print(f"    ep {ep} [{i+1}/{len(tr)}] CE {tot/(i+1):.4f}", flush=True)
        print(f"  ep {ep} done  train CE {tot/max(1,len(tr)):.4f}", flush=True)

    # ---- eval: three arms, pass counts AND CE, bootstrap over instances -----------------------
    perm = list(range(len(hd)))
    r = random.Random(4242)
    while True:
        r.shuffle(perm)
        if all(perm[i] != i for i in range(len(perm))):
            break

    thinker.eval(); reader.eval()
    with torch.no_grad():
        cache = [slots_for(t[0]) for t in hd]
    ce = {"wm": [], "der": [], "abl": []}
    ok = {"wm": 0, "der": 0, "abl": 0}
    with torch.no_grad():
        for i, (text, atoms, expr) in enumerate(hd):
            tests = make_tests(expr, codes, seed=i)
            if not tests:
                continue
            p, t = ids_for(text, expr)
            for arm, M in (("wm", cache[i]), ("der", cache[perm[i]]), ("abl", None)):
                slots_now["M"] = M
                o = wb.model(input_ids=torch.cat([p, t], -1),
                             labels=torch.cat([torch.full_like(p, -100), t], -1))
                ce[arm].append(float(o.loss))
                gen = wb.model.generate(p, max_new_tokens=a.max_new, do_sample=False,
                                        pad_token_id=wb.tok.eos_token_id)
                txt = wb.tok.decode(gen[0][p.shape[-1]:], skip_special_tokens=True)
                ok[arm] += int(verify(txt, atoms, tests, codes))
                slots_now["M"] = None
    handle.remove()

    n = len(ce["wm"])
    w, d, b = (np.array(ce[k]) for k in ("wm", "der", "abl"))
    spec, modal = d - w, b - d
    rs = np.random.RandomState(0)
    idx = [rs.randint(0, n, n) for _ in range(4000)]
    slo, shi = np.percentile([spec[j].mean() for j in idx], [2.5, 97.5])
    print(f"\n{'='*78}\nv6  read={a.read}  held={n}")
    print(f"  WM {ok['wm']}/{n}   DERANGED {ok['der']}/{n}   ABLATED {ok['abl']}/{n}")
    print(f"  CE  WM {w.mean():.4f}   DERANGED {d.mean():.4f}   ABLATED {b.mean():.4f}")
    print(f"  instance-specific  der-WM  {spec.mean():+.4f}  95% CI [{slo:+.4f}, {shi:+.4f}]")
    print(f"  format/mode        abl-der {modal.mean():+.4f}")
    if spec.mean() > 1e-9:
        ratio = modal.mean() / spec.mean()
        print(f"  modal/specific ratio {ratio:.1f}x")
    else:
        # printing "inf" for a NEGATIVE effect reads as "infinitely format-dominated" when it
        # actually means the deranged slots were BETTER -- a different and more alarming fact.
        ratio = float("inf")
        print(f"  modal/specific ratio  n/a  (instance-specific effect is {spec.mean():+.4f}, "
              f"i.e. deranged slots did as well or better)")
    if ok["wm"] == 0 and ok["abl"] == 0:
        print(f"  -> VACUOUS: WM and ABLATED both 0, no comparison carries information. Fix the run "
              f"(model, epochs, task difficulty) before reading anything here.")
    elif ok["wm"] <= ok["abl"]:
        print(f"  -> ADAPTER NET HARMFUL: the model does better with NO memory. Only visible because "
              f"ablated can score above 0 here.")
    elif slo > 0 and spec.mean() > 0.0075 and ratio <= 50:
        print(f"  -> REAL instance-specific channel: clears the null band, CI excludes 0, format "
              f"does not dominate.")
    else:
        print(f"  -> not established: need CI>0, effect above the 0.0075 null band, and ratio<=50.")
    print("=" * 78)


if __name__ == "__main__":
    main()
