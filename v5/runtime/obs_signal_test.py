"""Does the test-failure OBSERVATION carry operator signal beyond the issue? (T.9, fork A — on the
failure-rich SWE data from capture_failures.py.) Predict the gold operator from:
  issue-only   vs   issue + test-failure   vs   failure-only
5-fold CV. If issue+failure >> issue -> the observation helps (T.9 premise holds on debug data).
No Docker -- embeds + LogReg here.

  python -m v5.runtime.obs_signal_test --failures data/traj/swe_failures.jsonl
  python -m v5.runtime.obs_signal_test --selftest
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def obs_signal(rows, embed_fn, ops, folds=5):
    """rows: [{issue, test_failure, gold_op_id}]. Returns {feature: (acc,std)} + majority + n."""
    import numpy as np
    import collections
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold
    y = np.array([r["gold_op_id"] for r in rows])
    keep = np.array([c for c in range(len(rows)) if list(y).count(y[c]) >= 2])   # classes need >=2 for CV
    rows = [rows[c] for c in keep]; y = y[keep]
    if len(rows) < 2 * folds:
        return {}, 0.0, len(rows)
    I = np.asarray(embed_fn([r["issue"] for r in rows]))
    F = np.asarray(embed_fn([r["test_failure"] for r in rows]))
    feats = {"issue-only": I, "issue+failure": np.hstack([I, F]), "failure-only": F}
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=0)
    out = {}
    for name, X in feats.items():
        accs = [LogisticRegression(max_iter=2000).fit(X[tr], y[tr]).score(X[te], y[te])
                for tr, te in skf.split(X, y) if len(set(y[tr])) > 1]
        out[name] = (float(np.mean(accs)), float(np.std(accs))) if accs else (0.0, 0.0)
    maj = collections.Counter(y).most_common(1)[0][1] / len(y)
    return out, maj, len(rows)


def _report(out, maj, n):
    if not out:
        print(f"  too few usable instances (n={n})"); return
    print(f"\n=== OBS-SIGNAL TEST (T.9, n={n}, majority={maj:.0%}) ===")
    for k in ("issue-only", "issue+failure", "failure-only"):
        a, s = out[k]
        print(f"  gold-op acc | {k:16} {a:.0%} +-{s:.0%}")
    gain = out["issue+failure"][0] - out["issue-only"][0]
    print(f"  OBS gain (issue+failure - issue) = {gain:+.0%}")
    print(f"  VERDICT: test-failure observation {'CARRIES operator signal (T.9 premise holds on debug data)' if gain > 0.05 else 'adds no operator signal'}")


def main():
    ap = argparse.ArgumentParser(description="Does the test-failure help predict the gold operator? (T.9)")
    ap.add_argument("--failures", default="data/traj/swe_failures.jsonl")
    ap.add_argument("--ops", default="data/swe/discovered_ops_emb.jsonl")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        raise SystemExit(0 if _selftest() else 1)
    import torch
    from v5.training.providers import RealEmbedder
    from v5.runtime.operator_discovery import chunk_embedding
    emb = RealEmbedder(torch.device("cpu"))

    def embed_fn(texts, bs=32):
        out = []
        for i in range(0, len(texts), bs):
            ch = texts[i:i + bs]
            d = emb.embed_nodes({str(j): (t or "(none)")[:1200] for j, t in enumerate(ch)})
            out.extend(d[str(j)] for j in range(len(ch)))
        return out
    ops = [json.loads(l) for l in Path(a.ops).read_text(encoding="utf-8").splitlines() if l.strip()]
    names = {o["name"]: k for k, o in enumerate(ops)}
    raw = [json.loads(l) for l in Path(a.failures).read_text(encoding="utf-8").splitlines() if l.strip()]
    rows = []
    for r in raw:
        if not r.get("has_failure") or not r.get("gold_patch"):
            continue
        op = chunk_embedding(r["gold_patch"], ops, embed_fn)        # gold operator (nearest centroid)
        if op:
            rows.append({"issue": r["issue"], "test_failure": r["test_failure"], "gold_op_id": names[op[0]]})
    print(f"[obs-signal] {len(rows)} usable failure-rich instances")
    _report(*obs_signal(rows, embed_fn, ops))


def _selftest() -> bool:
    print("obs_signal_test --selftest: failure determines op -> issue+failure must beat issue\n")
    import numpy as np
    rng = np.random.RandomState(0)
    base = {f"f{k}": rng.randn(16) for k in range(4)}

    def fake_embed(texts):
        out = []
        for t in texts:
            v = np.zeros(16)
            for k in range(4):
                if f"f{k}" in t:
                    v = base[f"f{k}"]
            out.append(v + 0.05 * rng.randn(16))
        return out
    rows = []                                                       # op determined by which failure token
    for _ in range(120):
        k = rng.randint(0, 4)
        rows.append({"issue": "generic issue", "test_failure": f"failure f{k}", "gold_op_id": k})
    out, maj, n = obs_signal(rows, fake_embed, [{"name": str(i)} for i in range(4)])
    for kk, (acc, s) in out.items():
        print(f"  {kk:16} {acc:.0%}")
    assert out["issue+failure"][0] > out["issue-only"][0] + 0.3, "failure must add signal when it determines op"
    print("\n  OBS-SIGNAL SELFTEST -> PASS")
    return True


if __name__ == "__main__":
    main()
