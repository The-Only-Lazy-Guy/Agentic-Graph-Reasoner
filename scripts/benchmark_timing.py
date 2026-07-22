"""Fine-grained TRM timing to identify bottleneck."""

import time
import sys
from pathlib import Path

_ROOT = str(Path(__file__).resolve().parents[1])
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import torch
import numpy as np
from embedder import encode_batch, EMBED_DIM
from v5.runtime.membrane import AtomGraph, Atom, TRMRetriever, seed_graph, build_examples


def run():
    torch.manual_seed(0)
    g = seed_graph()
    # Use pre-computed embeddings for synthetic atoms to avoid per-atom encode_batch
    from embedder import encode_batch as _eb
    syn_n = 2000
    syn_descs = ["synth atom %d" % i for i in range(syn_n)]
    syn_embs = _eb(syn_descs)
    for i in range(syn_n):
        g.add(Atom(name="s_%d" % i, code="def x(n):\n    return n",
                   description=syn_descs[i], kind="concept", provenance="test",
                   emb=syn_embs[i]))
    print("Graph: %d atoms" % len(g))
    train_ex = build_examples("train")

    r = TRMRetriever(g, trm_top_k=256)
    print("TRM device: %s" % r.device)
    M, order = g.matrix()
    pos = {n: i for i, n in enumerate(order)}
    A0 = torch.from_numpy(M).to(r.device)
    data = [(r._embed_task(t), pos[g]) for t, g in train_ex if g in pos]
    xnp, gi = data[0]
    x = torch.from_numpy(xnp).to(r.device)
    r.trm.train()

    # Warmup
    _ = r.trm(x, A0)
    if r.device == "cuda":
        torch.cuda.synchronize()
    _ = r.trm(x, A0[:256])
    if r.device == "cuda":
        torch.cuda.synchronize()

    # Time full TRM forward only
    if r.device == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(20):
        _ = r.trm(x, A0)
    if r.device == "cuda":
        torch.cuda.synchronize()
    t1 = time.perf_counter()
    ft = (t1 - t0) / 20 * 1000
    print("TRM full forward:        %.3fms" % ft)

    # Time pre-filter TRM forward only
    if r.device == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(20):
        _ = r.trm(x, A0[:256])
    if r.device == "cuda":
        torch.cuda.synchronize()
    t1 = time.perf_counter()
    pt = (t1 - t0) / 20 * 1000
    print("TRM sub forward (256):   %.3fms  (%.1fx faster)" % (pt, ft / max(pt, 1e-9)))

    # Time full training step (forward + backward)
    if r.device == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(15):
        logits = r.trm(x, A0).unsqueeze(0)
        loss = torch.nn.functional.cross_entropy(logits,
                    torch.tensor([gi], device=r.device))
        loss.backward()
    if r.device == "cuda":
        torch.cuda.synchronize()
    t1 = time.perf_counter()
    ftb = (t1 - t0) / 15 * 1000
    print("TRM full fwd+bwd:        %.3fms" % ftb)

    # Time pre-filtered training step
    if r.device == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(15):
        A_sub, _, gold_sub = r._prefilter(x, A0, order, gold_pos=gi)
        logits = r.trm(x, A_sub).unsqueeze(0)
        loss = torch.nn.functional.cross_entropy(logits,
                    torch.tensor([gold_sub], device=r.device))
        loss.backward()
    if r.device == "cuda":
        torch.cuda.synchronize()
    t1 = time.perf_counter()
    ptb = (t1 - t0) / 15 * 1000
    print("TRM pf fwd+bwd (256):    %.3fms  (%.1fx faster)" % (ptb, ftb / max(ptb, 1e-9)))


if __name__ == "__main__":
    run()
