"""Train the localization retriever — the graph's actual job (find the buggy symbols).

The eval uses ORACLE support (gold-patch-derived). Naive cosine(issue, signature) over the whole
repo fails (0 applyable). This trains a retriever so the gold-touched symbols rank in the top-K
among ALL repo symbols, given only the issue.

Design (deliberate, from past lesson — separate q/sym projections destroy the shared-embedding
alignment): QUERY = frozen Qwen embedding, unchanged. SYMBOL = frozen + a ZERO-INIT residual
delta (MLP). At init delta=0 -> identical to naive cosine (can't start worse); training learns a
small symbol-side correction. InfoNCE: issue close to its gold symbols, far from the rest of the
repo's symbols (hard negatives, same pool as eval).

Train on lite-REPO instances that are NOT in lite-test (full --split test minus lite) -> domain-
matched + leakage-free. Eval recall@K on lite-test.

  python -m v5.graph_grower.train_retriever \
    --traces data/swe/fulltest_traces.jsonl --exclude-lite \
    --eval-traces data/swe/grounded_traces.jsonl \
    --out artifacts/stage_cache/retriever_delta.pt --n-train 300 --n-eval 50
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn

from v5.training.providers import RealEmbedder
from v5.graph_grower.swe_load import load_instances, checkout_repo
from v5.runtime.code_retrieve import _pool_files, SymDelta
from v5.graph_grower.code_extract import extract_paths


def _unit(a):
    a = np.asarray(a, dtype=np.float32)
    return a / (np.linalg.norm(a, axis=-1, keepdims=True) + 1e-9)


def _load_traces_full(paths):
    """{iid: {issue, support_meta:[(file,lo,hi)]}} — support from gold (for marking positives)."""
    out = {}
    for p in paths:
        for line in Path(p).read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            r = json.loads(line); v = r.get("v2_grounding") or {}
            iid = r.get("instance_id") or v.get("instance_id")
            if iid and v.get("support_ids"):
                out[iid] = {"issue": v.get("task", ""), "support_ids": v["support_ids"]}
    return out


def build_examples(ids, traces, insts, meta, repo_root, embedder, max_files, max_syms):
    """Per instance: q_vec + pool symbol vecs + gold mask (whole-package pool = eval-hard negatives)."""
    examples = []
    for k, iid in enumerate(ids):
        if k % 20 == 0:
            print(f"    build {k}/{len(ids)} (kept {len(examples)})", flush=True)
        t, inst = traces[iid], insts[iid]
        gold = {(meta[s]["file"], meta[s]["lineno"][0], meta[s]["lineno"][-1])
                for s in t["support_ids"] if s in meta}
        if not gold:
            continue
        dest = Path(repo_root) / inst["repo"].replace("/", "__")
        ok, _ = checkout_repo(inst["repo"], inst["base_commit"], dest, timeout=1800)
        if not ok:
            continue
        nodes, _ = extract_paths(str(dest), _pool_files(str(dest), max_files), repo="")
        syms = [n for n in nodes if n.get("node_type") == "symbol"
                and (n.get("text") or "").strip() and (n.get("metadata") or {}).get("lineno")][:max_syms]
        if len(syms) < 5:
            continue
        # mark positives: pool symbol overlaps a gold span in the same file
        mask = []
        for n in syms:
            m = n["metadata"]; lo, hi = m["lineno"][0], m["lineno"][-1]
            pos = any(m["file"] == gf and not (hi < glo or lo > ghi) for gf, glo, ghi in gold)
            mask.append(pos)
        if not any(mask):                    # gold function not in the extracted pool -> unusable
            continue
        qv = _unit(embedder.embed_nodes({"q": t["issue"][:1500]})["q"])
        sv = embedder.embed_nodes({n["node_id"]: n["text"] for n in syms})
        pool = _unit(np.stack([sv[n["node_id"]] for n in syms]))
        examples.append({"q": qv, "pool": pool, "mask": np.array(mask, dtype=bool)})
    return examples


def _recall(examples, model, k, device):
    if not examples:
        return 0.0
    hits = 0
    with torch.no_grad():
        for e in examples:
            q = torch.tensor(e["q"], device=device)
            s = model(torch.tensor(e["pool"], device=device))
            scores = (s @ q.T).squeeze(-1)
            topk = torch.topk(scores, min(k, scores.shape[0])).indices.cpu().numpy()
            hits += 1 if e["mask"][topk].any() else 0
    return hits / len(examples)


def run(traces_p, eval_traces_p, nodes_p, dataset, split, n_train, n_eval, exclude_lite,
        out, epochs, lr, max_files, max_syms, k_eval, device_str=None):
    device = torch.device(device_str or ("cuda" if torch.cuda.is_available() else "cpu"))
    from v5.runtime.sr_withcode import load_symbol_meta
    embedder = RealEmbedder(device)
    meta = load_symbol_meta(nodes_p)
    insts = {t["instance_id"]: t for t in load_instances(dataset, split, limit=0)}

    traces = _load_traces_full(traces_p)
    pool = [i for i in traces if i in insts]
    if exclude_lite:
        lite_ids = {t["instance_id"] for t in load_instances("lite", "test", limit=0)}
        pool = [i for i in pool if i not in lite_ids]
    print(f"train pool {len(pool)} | building {min(n_train, len(pool))} examples...", flush=True)
    train_ex = build_examples(pool[:n_train], traces, insts, meta, "data/swe_repos",
                              embedder, max_files, max_syms)

    ev_traces = _load_traces_full(eval_traces_p)
    ev_insts = {t["instance_id"]: t for t in load_instances("lite", "test", limit=0)}
    ev_ids = [i for i in ev_traces if i in ev_insts][:n_eval]
    ev_meta = load_symbol_meta(["artifacts/graph_growth/swe_code_candidates.jsonl"])
    eval_ex = build_examples(ev_ids, ev_traces, ev_insts, ev_meta, "data/swe_repos",
                             embedder, max_files, max_syms)
    print(f"examples: {len(train_ex)} train / {len(eval_ex)} eval", flush=True)
    if not train_ex:
        print("no training examples"); return

    dim = train_ex[0]["pool"].shape[1]
    model = SymDelta(dim).to(device)
    print(f"naive-cosine recall@{k_eval} (baseline, delta=0): {_recall(eval_ex, model, k_eval, device):.3f}", flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    for ep in range(epochs):
        model.train(); tot = n = 0
        for e in train_ex:
            q = torch.tensor(e["q"], device=device)                  # [1, D]
            s = model(torch.tensor(e["pool"], device=device))        # [N, D]
            scores = (s @ q.T).squeeze(-1) / 0.05                     # temp
            pos = torch.tensor(e["mask"], device=device)
            if not pos.any():
                continue
            # InfoNCE: -log sum exp(pos) / sum exp(all)
            logp = torch.log_softmax(scores, dim=0)
            loss = -torch.logsumexp(logp[pos], dim=0)
            opt.zero_grad(); loss.backward(); opt.step()
            tot += float(loss); n += 1
        rec = _recall(eval_ex, model, k_eval, device)
        print(f"  [retriever] epoch {ep+1}  loss {tot/max(1,n):.4f}  recall@{k_eval} {rec:.3f}", flush=True)

    Path(out).parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out)
    print(f"saved retriever delta -> {out}")


def main(argv=None):
    ap = argparse.ArgumentParser(description="Train the localization retriever (zero-init symbol delta).")
    ap.add_argument("--traces", nargs="+", default=["data/swe/fulltest_traces.jsonl"])
    ap.add_argument("--eval-traces", nargs="+", default=["data/swe/grounded_traces.jsonl"])
    ap.add_argument("--nodes", nargs="+", default=["artifacts/graph_growth/fulltest_code_candidates.jsonl"])
    ap.add_argument("--dataset", default="full")
    ap.add_argument("--split", default="test")
    ap.add_argument("--exclude-lite", action="store_true")
    ap.add_argument("--n-train", type=int, default=300)
    ap.add_argument("--n-eval", type=int, default=50)
    ap.add_argument("--out", default="artifacts/stage_cache/retriever_delta.pt")
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--max-files", type=int, default=400)
    ap.add_argument("--max-syms", type=int, default=1200)
    ap.add_argument("--k-eval", type=int, default=8)
    ap.add_argument("--device", default=None)
    a = ap.parse_args(argv)
    run(a.traces, a.eval_traces, a.nodes, a.dataset, a.split, a.n_train, a.n_eval, a.exclude_lite,
        a.out, a.epochs, a.lr, a.max_files, a.max_syms, a.k_eval, device_str=a.device)


if __name__ == "__main__":
    raise SystemExit(main())
