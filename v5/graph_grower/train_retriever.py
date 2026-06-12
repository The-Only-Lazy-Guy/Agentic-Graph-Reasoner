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
from v5.runtime.code_retrieve import _rank_files, SymDelta
from v5.graph_grower.code_extract import extract_paths
from v5.graph_grower.swe_grounded import parse_patch_hunks


def _unit(a):
    a = np.asarray(a, dtype=np.float32)
    return a / (np.linalg.norm(a, axis=-1, keepdims=True) + 1e-9)


def _gold_spans(patch):
    """Gold patch -> {file: [(lo, hi)]} changed-line ranges (the positives, straight from the diff)."""
    return {f: [(s, s + max(l, 1) - 1) for s, l in spans] for f, spans in parse_patch_hunks(patch).items()}


def build_examples(insts_list, repo_root, embedder, max_files, max_syms):
    """Per instance: q_vec + whole-package symbol vecs + gold mask. Gold = symbols overlapping the
    patch's changed lines (derived from the diff, NOT a prebuilt candidate graph)."""
    examples = []
    for k, inst in enumerate(insts_list):
        if k % 20 == 0:
            print(f"    build {k}/{len(insts_list)} (kept {len(examples)})", flush=True)
        gold = _gold_spans(inst.get("patch", "") or "")
        if not gold:
            continue
        dest = Path(repo_root) / inst["repo"].replace("/", "__")
        ok, _ = checkout_repo(inst["repo"], inst["base_commit"], dest, timeout=1800)
        if not ok:
            continue
        nodes, _ = extract_paths(str(dest), _rank_files(str(dest), inst.get("problem_statement", ""), max_files), repo="")
        syms = [n for n in nodes if n.get("node_type") == "symbol"
                and (n.get("text") or "").strip() and (n.get("metadata") or {}).get("lineno")][:max_syms]
        if len(syms) < 5:
            continue
        mask = []
        for n in syms:
            m = n["metadata"]; lo, hi = m["lineno"][0], m["lineno"][-1]
            pos = any(m["file"] == gf and not (hi < glo or lo > ghi)
                      for gf, spans in gold.items() for glo, ghi in spans)
            mask.append(pos)
        if not any(mask):                    # gold function not in the extracted pool -> unusable
            continue
        qv = _unit(embedder.embed_nodes({"q": (inst.get("problem_statement") or "")[:1500]})["q"])
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


def run(dataset, split, n_train, n_eval, exclude_lite, repo_root,
        out, epochs, lr, max_files, max_syms, k_eval, device_str=None):
    device = torch.device(device_str or ("cuda" if torch.cuda.is_available() else "cpu"))
    embedder = RealEmbedder(device)

    train_insts = load_instances(dataset, split, limit=0)
    if exclude_lite:
        lite_ids = {t["instance_id"] for t in load_instances("lite", "test", limit=0)}
        train_insts = [t for t in train_insts if t["instance_id"] not in lite_ids]
    print(f"train pool {len(train_insts)} | building <= {n_train} examples...", flush=True)
    train_ex = build_examples(train_insts[:n_train], repo_root, embedder, max_files, max_syms)

    eval_insts = load_instances("lite", "test", limit=0)[:n_eval]
    print(f"building <= {n_eval} eval (lite) examples...", flush=True)
    eval_ex = build_examples(eval_insts, repo_root, embedder, max_files, max_syms)
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
    ap.add_argument("--dataset", default="full", help="train source (full --split test = lite repos)")
    ap.add_argument("--split", default="test")
    ap.add_argument("--exclude-lite", action="store_true", help="drop lite-test instances (leakage-free)")
    ap.add_argument("--n-train", type=int, default=300)
    ap.add_argument("--n-eval", type=int, default=50)
    ap.add_argument("--repo-root", default="data/swe_repos")
    ap.add_argument("--out", default="artifacts/stage_cache/retriever_delta.pt")
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--max-files", type=int, default=40, help="STAGE-1 file-filter: top-K files by issue keywords")
    ap.add_argument("--max-syms", type=int, default=1200)
    ap.add_argument("--k-eval", type=int, default=8)
    ap.add_argument("--device", default=None)
    a = ap.parse_args(argv)
    run(a.dataset, a.split, a.n_train, a.n_eval, a.exclude_lite, a.repo_root,
        a.out, a.epochs, a.lr, a.max_files, a.max_syms, a.k_eval, device_str=a.device)


if __name__ == "__main__":
    raise SystemExit(main())
