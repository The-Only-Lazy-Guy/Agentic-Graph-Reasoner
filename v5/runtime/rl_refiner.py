"""RL-train the graph-side REFINER policy (which operator's pattern to hand the frozen LLM), reward =
how well the frozen 4B's realization matches the gold fix. The LLM is FROZEN — ONLY the small policy
net trains (graph-side, on-thesis). REINFORCE with a moving-average baseline.

The mean-reward curve is the answer: rising -> the graph steers the frozen 4B (train it) ; flat -> the
4B is unsteerable by operator choice (graph-side selection can't help; the wall is the frozen executor).

  V5_LM_TRUST_REMOTE_CODE=1 V5_LM_QUANT=4bit python -m v5.runtime.rl_refiner --model Qwen/Qwen3.5-4B --n 20 --epochs 3
  python -m v5.runtime.rl_refiner --selftest
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


def _recall(patch, gold_added):
    from v5.runtime.solution_ladder import _fidelity
    emit = "\n".join(l[1:] for l in (patch or "").splitlines() if l.startswith("+") and not l.startswith("+++"))
    return _fidelity(emit, gold_added)[0]


def train(insts, ops, gen_fn, embed_fn, realize_fn, epochs, lr=5e-3, seed=0, log=print):
    """Policy: goal_emb -> softmax over operators. REINFORCE, reward = recall of the realized patch."""
    import numpy as np, torch, torch.nn as nn
    from v5.runtime.operator_discovery import extract_hunks
    torch.manual_seed(seed); rng = random.Random(seed)
    d = len(embed_fn(["x"])[0])
    policy = nn.Sequential(nn.Linear(d, 256), nn.ReLU(), nn.Linear(256, len(ops)))
    opt = torch.optim.Adam(policy.parameters(), lr=lr)
    # precompute goal embeddings + gold-added per instance
    prepared = []
    for t in insts:
        hs = extract_hunks(t.get("patch", "") or "")
        if not hs:
            continue
        gold = "\n".join(l for _f, _r, a in hs for l in a)
        g = torch.tensor(np.asarray(embed_fn([t["_goal"]])[0]), dtype=torch.float32)
        prepared.append((t, g, gold))
    log(f"[rl] {len(prepared)} instances, {len(ops)} operators, policy over goal-emb")
    baseline = 0.0
    curve = []
    for ep in range(epochs):
        rng.shuffle(prepared); rewards = []
        for t, g, gold in prepared:
            logits = policy(g)
            dist = torch.distributions.Categorical(logits=logits)
            a = dist.sample()
            patch = realize_fn(t, ops[int(a)], "")
            r = _recall(patch, gold)
            rewards.append(r)
            loss = -dist.log_prob(a) * (r - baseline)          # REINFORCE with baseline
            opt.zero_grad(); loss.backward(); opt.step()
            baseline = 0.9 * baseline + 0.1 * r
        m = float(np.mean(rewards)) if rewards else 0.0
        curve.append(m)
        log(f"  epoch {ep+1}/{epochs}: mean reward (recall) = {m:.3f}")
    return curve


def _report(curve):
    if not curve:
        print("  no data"); return
    print(f"\n=== RL REFINER (graph-side policy, LLM frozen) ===")
    print(f"  reward curve: {[round(c,3) for c in curve]}")
    lift = curve[-1] - curve[0]
    print(f"  lift (last - first) = {lift:+.3f} -> "
          f"{'POLICY LEARNS -> graph steers the frozen 4B (train it more)' if lift > 0.03 else 'FLAT -> operator choice does not steer the 4B (executor is the wall)'}")


def _selftest() -> bool:
    print("rl_refiner --selftest: policy must learn to pick the rewarding operator (mock realize)\n")
    import numpy as np
    ops = [{"name": "A"}, {"name": "B"}, {"name": "C"}]
    insts = [{"instance_id": f"t{i}", "_goal": "g", "patch":
              "diff --git a/f b/f\n--- a/f\n+++ b/f\n@@ -1 +1 @@\n-x\n+RIGHT\n"} for i in range(30)]

    def embed_fn(ts):
        return [np.ones(8, dtype="float32") for _ in ts]

    def realize(task, op, fb):                                 # op B -> the gold; others wrong
        return "diff\n+RIGHT" if op["name"] == "B" else "diff\n+WRONG"
    curve = train(insts, ops, None, embed_fn, realize, epochs=8, lr=0.02, log=lambda *a: None)
    print(f"  reward curve: {[round(c,2) for c in curve]}")
    assert curve[-1] > 0.8, "policy must learn to pick B (reward >> random 0.33)"
    print("\n  RL REFINER SELFTEST -> PASS (policy learned the rewarding operator)")
    return True


def main():
    ap = argparse.ArgumentParser(description="RL-train the graph-side refiner policy (LLM frozen).")
    ap.add_argument("--model", default="Qwen/Qwen3.5-4B")
    ap.add_argument("--dataset", default="lite")
    ap.add_argument("--split", default="test")
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--ops", default="data/swe/discovered_ops_emb.jsonl")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        raise SystemExit(0 if _selftest() else 1)
    import torch
    from v5.graph_grower.swe_load import load_instances, checkout_repo
    from v5.runtime.solution_ladder import real_gen_fn
    from v5.runtime.lggn_loop import make_realize, _src_window
    from v5.training.providers import RealEmbedder
    _e = RealEmbedder(torch.device("cpu"))

    def embed_fn(ts):
        dd = _e.embed_nodes({str(j): t[:1000] for j, t in enumerate(ts)})
        return [dd[str(j)] for j in range(len(ts))]
    ops = [json.loads(l) for l in Path(a.ops).read_text(encoding="utf-8").splitlines() if l.strip()]
    insts = list(load_instances(name=a.dataset, split=a.split, limit=a.n))
    cache = Path("artifacts/lggn_loop_repos"); cache.mkdir(parents=True, exist_ok=True)
    prepared = []
    for t in insts:                                            # checkout + build the goal once
        dest = cache / t["instance_id"]
        try:
            if not dest.exists():
                checkout_repo(t["repo"], t["base_commit"], dest)
        except Exception as e:                                 # noqa: BLE001
            print(f"  [skip {t['instance_id']}] {e}"); continue
        t["_dest"] = str(dest)
        t["_goal"] = t.get("problem_statement", "")[:900] + "\nCODE:\n" + _src_window(str(dest), t["patch"])[0][:600]
        prepared.append(t)
    realize_fn = make_realize(real_gen_fn(a.model))            # loads the 4B ONCE
    print(f"[rl] model={a.model}, {len(prepared)} instances, {a.epochs} epochs (LLM frozen; only policy trains)")
    _report(train(prepared, ops, None, embed_fn, realize_fn, a.epochs))


if __name__ == "__main__":
    main()
