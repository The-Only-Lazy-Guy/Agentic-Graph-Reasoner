"""LGGN TRAVERSE — the reasoner: goal -> operator trajectory (HRM-templated planner).

Trains WITHOUT the 4B: a FROZEN goal-embedder (RealEmbedder) + a tiny GRU decoder over the DISCOVERED
operator vocab. Fast + local; decoupled from the 4B leaf-capacity wall. Metric = held trajectory-
prediction accuracy (does the reasoner GENERALIZE to unseen goals?) vs a majority-class baseline (does
the goal actually inform the operator, or is it just frequency?). See LGGN_DESIGN.md §4c.

  python -m v5.runtime.traverse --selftest
  python -m v5.runtime.traverse --train --dataset lite --ops data/swe/discovered_ops.jsonl
"""
from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path


# ── corpus: (goal text, operator-name trajectory) ─────────────────────────────
def build_corpus_swe(dataset: str, split: str, ops: list[dict], with_code: bool = True) -> list[tuple[str, list[str]]]:
    """goal = issue (+ the CODE AT THE FIX SITE when with_code; the operator is determined by the code
    being transformed, not the issue text alone). trajectory = chunk(gold) over discovered ops."""
    from v5.graph_grower.swe_load import load_instances
    from v5.runtime.operator_discovery import chunk, extract_hunks
    out = []
    for i in load_instances(name=dataset, split=split, limit=0):
        patch = i.get("patch", "") or ""
        traj = [t for t in chunk(patch, ops) if t != "novel"]
        if not traj:
            continue
        goal = i.get("problem_statement") or i.get("issue") or i["instance_id"]
        if with_code:                              # the original code at the fix site (removed/context lines)
            code = "\n".join(l for (_f, rem, _a) in extract_hunks(patch) for l in rem)[:1200]
            goal = f"{goal[:1200]}\n\nCODE AT FIX SITE:\n{code}"
        if COARSE[0]:
            traj = [coarse_bucket(t) for t in traj]
        out.append((goal, traj))
    return out


COARSE = [False]   # module flag toggled by --coarse


def coarse_bucket(name: str) -> str:
    """Collapse a fine structural-signature op -> a coarse SEMANTIC bucket (~9 classes). Tests whether
    operator GRANULARITY is why goal->operator doesn't generalize (surface signatures hard to predict;
    semantic buckets maybe learnable)."""
    n = name.lower()
    if "import" in n:
        return "IMPORT"
    if "except" in n or "try" in n or "raise" in n:
        return "EXCEPT"
    if "none" in n:
        return "GUARD"
    if "return" in n:
        return "RETURN"
    if "for" in n or "while" in n:
        return "LOOP"
    if "if" in n or "else" in n or "cmp" in n:
        return "COND"
    if "assign" in n:
        return "ASSIGN"
    if "call" in n or "attr" in n:
        return "CALL"
    return "EDIT"


def load_ops(path: str) -> list[dict]:
    return [json.loads(l) for l in Path(path).read_text(encoding="utf-8").splitlines() if l.strip()]


# ── planner ───────────────────────────────────────────────────────────────────
def _make_planner(n_ops: int, d_goal: int, d: int = 256):
    import torch.nn as nn
    import torch

    class TraversePlanner(nn.Module):
        """g (frozen goal embedding) conditions a GRU that autoregresses the operator sequence.
        ids: [0..n_ops-1] = operators, n_ops = BOS (decoder input start), n_ops = STOP target class."""
        def __init__(self):
            super().__init__()
            self.n_ops = n_ops
            self.bos = n_ops
            self.op_emb = nn.Embedding(n_ops + 1, d)       # ops + BOS
            self.g_proj = nn.Linear(d_goal, d)
            self.gru = nn.GRU(d, d, batch_first=True)
            self.head = nn.Linear(d, n_ops + 1)            # ops + STOP(=n_ops)

        def forward(self, g, inp_ids):                     # g:[B,d_goal], inp_ids:[B,T]
            h0 = torch.tanh(self.g_proj(g)).unsqueeze(0)   # [1,B,d]
            out, _ = self.gru(self.op_emb(inp_ids), h0)    # [B,T,d]
            return self.head(out)                          # [B,T,n_ops+1]

    return TraversePlanner()


# ── train + eval ────────────────────────────────────────────────────────────
def train_traverse(corpus, op_names, embed_fn, epochs=60, lr=2e-3, d_goal=768, seed=0, log=print):
    """corpus: [(goal, [op_names])]. embed_fn: list[str]->[N,d_goal] tensor. Returns (model, metrics)."""
    import torch
    rng = random.Random(seed)
    torch.manual_seed(seed)
    name2id = {n: i for i, n in enumerate(op_names)}
    n_ops = len(op_names)
    STOP = n_ops
    data = [(g, [name2id[o] for o in tr if o in name2id]) for g, tr in corpus]
    data = [(g, t) for g, t in data if t]
    rng.shuffle(data)
    n_held = max(8, len(data) // 5)
    held, train = data[:n_held], data[n_held:]
    log(f"[traverse] corpus {len(data)} (train {len(train)} / held {len(held)}) | {n_ops} operators")

    G = embed_fn([g for g, _ in train])                    # [Ntr, d_goal] (frozen)
    Gh = embed_fn([g for g, _ in held])
    model = _make_planner(n_ops, d_goal)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    ce = torch.nn.CrossEntropyLoss()

    for ep in range(epochs):
        model.train(); order = list(range(len(train))); rng.shuffle(order); tot = 0.0
        for j in order:
            _, traj = train[j]
            g = G[j:j + 1]
            inp = torch.tensor([[model.bos] + traj])       # [1, 1+k]
            tgt = torch.tensor(traj + [STOP])              # [1+k]
            logits = model(g, inp)[0]                       # [1+k, n_ops+1]
            loss = ce(logits, tgt)
            opt.zero_grad(); loss.backward(); opt.step(); tot += float(loss)
        if ep == 0 or (ep + 1) % 20 == 0:
            log(f"  [ep {ep+1:3}] loss {tot/max(1,len(train)):.3f}")

    # ── eval: greedy decode held vs gold; + majority-class baseline ──
    @torch.no_grad()
    def decode(g, max_len=5):
        model.eval(); ids = [model.bos]; out = []
        for _ in range(max_len):
            logits = model(g, torch.tensor([ids]))[0, -1]
            nxt = int(logits.argmax())
            if nxt == STOP:
                break
            out.append(nxt); ids.append(nxt)
        return out
    maj = Counter(t[0] for _, t in train).most_common(1)[0][0]   # majority FIRST operator
    first_hit = exact = maj_hit = 0; f1s = []
    for k, (_, gold) in enumerate(held):
        pred = decode(Gh[k:k + 1])
        first_hit += int(bool(pred) and pred[0] == gold[0])
        maj_hit += int(gold[0] == maj)
        exact += int(pred == gold)
        ps, gs = set(pred), set(gold)
        inter = len(ps & gs)
        p = inter / max(1, len(ps)); r = inter / max(1, len(gs))
        f1s.append(0 if (p + r) == 0 else 2 * p * r / (p + r))
    n = len(held)
    m = {"first_op_acc": first_hit / n, "majority_baseline": maj_hit / n,
         "exact_traj": exact / n, "op_f1": sum(f1s) / n, "n_held": n}
    return model, m


# ── embedders ─────────────────────────────────────────────────────────────────
def real_embed_fn():
    import torch
    from v5.training.providers import RealEmbedder
    emb = RealEmbedder(torch.device("cpu"))

    def f(goals):
        v = emb.embed_nodes({str(i): g[:1500] for i, g in enumerate(goals)})
        return torch.tensor([list(map(float, v[str(i)])) for i in range(len(goals))])
    return f, 768


# ── selftest: learnable goal->operator mapping (no LM) ───────────────────────
def _selftest() -> bool:
    print("traverse --selftest: planner learns goal->operator trajectory (synthetic, no LM)\n")
    import torch
    # 3 goal families -> 3 distinct trajectories; held-out goals of each family must map correctly
    fam = {"crash null deref segfault": ["AddNoneGuard"],
           "missing import name error": ["AddImport", "Mod_edit"],
           "wrong value returned": ["UseRealValue"]}
    ops = sorted({o for tr in fam.values() for o in tr})
    base = {f: torch.randn(768) for f in fam}                 # each family = a fixed region + noise
    corpus = []
    for f, tr in fam.items():
        for _ in range(20):
            corpus.append((f, tr))
    vecs = {}

    def embed_fn(goals):
        return torch.stack([base[g] + 0.05 * torch.randn(768) for g in goals])
    # train; held is a random 1/5 split of the 60 (same families) -> tests generalization to noisy held
    _, m = train_traverse(corpus, ops, embed_fn, epochs=80, seed=1, log=lambda *a: None)
    print(f"  first_op_acc={m['first_op_acc']:.2f}  majority_baseline={m['majority_baseline']:.2f}  "
          f"exact_traj={m['exact_traj']:.2f}  op_f1={m['op_f1']:.2f}  (held {m['n_held']})")
    assert m["first_op_acc"] >= 0.9, "planner must learn goal->operator (>=0.9 first-op acc)"
    assert m["first_op_acc"] > m["majority_baseline"] + 0.2, "planner must beat majority-class (uses the goal)"
    print("\n  TRAVERSE SELFTEST -> PASS")
    return True


def main():
    ap = argparse.ArgumentParser(description="LGGN traverse — train the goal->operator planner.")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--train", action="store_true")
    ap.add_argument("--dataset", default="lite")
    ap.add_argument("--split", default="test")
    ap.add_argument("--ops", default="data/swe/discovered_ops.jsonl")
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--coarse", action="store_true", help="collapse fine ops -> ~9 semantic buckets (granularity test)")
    a = ap.parse_args()
    if a.selftest:
        raise SystemExit(0 if _selftest() else 1)
    if a.train:
        ops = load_ops(a.ops)
        COARSE[0] = a.coarse
        op_names = sorted({coarse_bucket(o["name"]) for o in ops}) if a.coarse else [o["name"] for o in ops]
        print(f"[traverse] operator vocab: {len(op_names)} {'COARSE buckets' if a.coarse else 'fine ops'}")
        corpus = build_corpus_swe(a.dataset, a.split, ops)
        embed_fn, d = real_embed_fn()
        _, m = train_traverse(corpus, op_names, embed_fn, epochs=a.epochs, d_goal=d)
        print("\n=== TRAVERSE held trajectory-prediction ===")
        for k, v in m.items():
            print(f"  {k}: {v:.3f}" if isinstance(v, float) else f"  {k}: {v}")
        print(f"\n  VERDICT: planner first-op {m['first_op_acc']:.0%} vs majority {m['majority_baseline']:.0%} "
              f"-> {'LEARNS goal->operator (reasoner real)' if m['first_op_acc'] > m['majority_baseline'] + 0.1 else 'no lift over frequency (goal not informing)'}")
        return
    ap.print_help()


if __name__ == "__main__":
    main()
