"""POMDP policy ablation (the architecture ASSEMBLED): does the full state
    goal + prev-operator + OBSERVATION
beat each piece alone at predicting the NEXT operator? Tests the one untested claim -- do observations
carry signal (T.9, the POMDP loop) -- and whether goal+topology+obs > any single component.

5-fold GROUPED CV (by session, so a session's goal can't leak across folds). Next-op prediction over
the grown operator library. See LGGN_DESIGN.md T.5/T.9.

  python -m v5.runtime.pomdp_policy --traj data/traj/fable_heavy.jsonl
  python -m v5.runtime.pomdp_policy --selftest
"""
from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path


def load_transitions(path: str, collapse: bool = False) -> list[dict]:
    """Trajectory steps -> transitions {sess, goal, prev, obs, next}. obs = the observation AFTER prev
    (the result that informs choosing next) -> the POMDP signal."""
    rows = []
    for l in Path(path).read_text(encoding="utf-8").splitlines():
        if not l.strip():
            continue
        d = json.loads(l); traj = d["trajectory"]
        for i in range(1, len(traj)):
            prev, nxt = traj[i - 1], traj[i]
            if not prev.get("op") or not nxt.get("op"):
                continue
            if collapse and prev["op"] == nxt["op"]:
                continue
            rows.append({"sess": d.get("session_id", ""), "goal": d["goal"],
                         "prev": prev["op"], "obs": prev.get("obs") or "(none)", "next": nxt["op"]})
    return rows


def pomdp_ablation(rows, embed_fn, op_names, folds=5):
    """Returns {feature_set: (mean_acc, std)} over grouped CV + majority baseline + n."""
    import numpy as np
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GroupKFold
    nid = {n: i for i, n in enumerate(op_names)}
    rows = [r for r in rows if r["prev"] in nid and r["next"] in nid]
    if len(rows) < 2 * folds:
        return {}, 0.0, len(rows)
    y = np.array([nid[r["next"]] for r in rows])
    groups = [r["sess"] for r in rows]
    n = len(op_names)
    G = np.asarray(embed_fn([r["goal"] for r in rows]))               # session goal
    O = np.asarray(embed_fn([r["obs"] for r in rows]))                # observation after prev
    P = np.zeros((len(rows), n))                                       # prev-op one-hot (topology)
    for k, r in enumerate(rows):
        P[k, nid[r["prev"]]] = 1.0
    feats = {"goal-only": G, "topology(prev-op)": P,
             "goal+prev": np.hstack([G, P]), "goal+prev+OBS": np.hstack([G, P, O])}
    gkf = GroupKFold(n_splits=min(folds, len(set(groups))))
    out = {}
    for name, X in feats.items():
        accs = []
        for tr, te in gkf.split(X, y, groups):
            if len(set(y[tr])) < 2:
                continue
            accs.append(LogisticRegression(max_iter=2000).fit(X[tr], y[tr]).score(X[te], y[te]))
        out[name] = (float(np.mean(accs)), float(np.std(accs))) if accs else (0.0, 0.0)
    maj = collections.Counter(y).most_common(1)[0][1] / len(y)
    return out, maj, len(rows)


def _report(out, maj, n, tag):
    if not out:
        print(f"  [{tag}] too few transitions ({n})"); return
    print(f"\n=== POMDP POLICY ABLATION [{tag}] (n={n}, majority={maj:.0%}) ===")
    for name in ("goal-only", "topology(prev-op)", "goal+prev", "goal+prev+OBS"):
        a, s = out[name]
        print(f"  next-op acc | {name:20} {a:.0%} +-{s:.0%}")
    obs_gain = out["goal+prev+OBS"][0] - out["goal+prev"][0]
    full_best = out["goal+prev+OBS"][0] >= max(out["goal-only"][0], out["topology(prev-op)"][0], out["goal+prev"][0])
    print(f"  OBS gain (goal+prev+OBS - goal+prev) = {obs_gain:+.0%}")
    print(f"  VERDICT: observations {'HELP' if obs_gain > 0.03 else 'do NOT help'}; "
          f"full POMDP state {'is best (loop justified)' if full_best else 'not best'}")


def real_embed_fn():
    import torch
    from v5.training.providers import RealEmbedder
    emb = RealEmbedder(torch.device("cpu"))

    def f(texts, bs=48):
        out = []
        for i in range(0, len(texts), bs):
            ch = texts[i:i + bs]
            d = emb.embed_nodes({str(j): (t or "(none)")[:900] for j, t in enumerate(ch)})
            out.extend(d[str(j)] for j in range(len(ch)))
        return out
    return f


def _selftest() -> bool:
    print("pomdp_policy --selftest: obs determines next-op -> OBS must help (no real model)\n")
    import numpy as np
    rng = np.random.RandomState(0)
    ops = ["FIX", "NEXT", "A", "B"]
    base = {"failed": rng.randn(16), "ok": rng.randn(16), "g": rng.randn(16)}
    rows = []
    for s in range(60):                                            # next-op determined by OBS, not goal/prev
        for _ in range(8):
            obs = "failed" if rng.rand() < 0.5 else "ok"
            rows.append({"sess": f"s{s}", "goal": "g", "prev": rng.choice(["A", "B"]),
                         "obs": obs, "next": "FIX" if obs == "failed" else "NEXT"})

    def fake_embed(texts):
        return [base.get(t.split()[0] if t else "g", base["g"]) + 0.05 * rng.randn(16) for t in
                [(x if x in base else ("failed" if "failed" in x else ("ok" if "ok" in x else "g"))) for x in texts]]
    out, maj, n = pomdp_ablation(rows, fake_embed, ops, folds=5)
    for k, (a, s) in out.items():
        print(f"  {k:20} {a:.0%}")
    assert out["goal+prev+OBS"][0] > out["goal+prev"][0] + 0.2, "OBS must help when it determines next-op"
    print("\n  POMDP POLICY SELFTEST -> PASS (observations carry the signal)")
    return True


def main():
    ap = argparse.ArgumentParser(description="POMDP policy ablation (goal+prev+obs -> next op).")
    ap.add_argument("--traj", help="mined trajectories jsonl")
    ap.add_argument("--ops", default="data/traj/grown_ops.jsonl")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        raise SystemExit(0 if _selftest() else 1)
    if not a.traj:
        ap.error("--traj required, or --selftest")
    op_names = [json.loads(l)["name"] for l in Path(a.ops).read_text(encoding="utf-8").splitlines() if l.strip()]
    embed_fn = real_embed_fn()
    for collapse, tag in [(False, "raw (incl self-loops)"), (True, "op-changes only")]:
        rows = load_transitions(a.traj, collapse=collapse)
        _report(*pomdp_ablation(rows, embed_fn, op_names), tag=tag)


if __name__ == "__main__":
    main()
