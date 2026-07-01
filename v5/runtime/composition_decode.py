"""Composition-decode: the latent traversal reaches the fix by COMPOSING operators (validated,
latent_compose.py). Can the frozen LLM cash it in? Train the composer (goal -> operator weights,
graph-side, LLM frozen); deliver the TOP-K composed operators' patterns to the 4B; measure realization
recall vs a single operator vs issue-only.

  issue-only : no operators
  top-1      : the single highest-weight operator's pattern (the RL-selection regime)
  top-K      : the K composed operators' patterns, weighted (the latent-traversal regime)

  V5_LM_TRUST_REMOTE_CODE=1 V5_LM_QUANT=4bit python -m v5.runtime.composition_decode --model Qwen/Qwen3.5-4B --n 40
  python -m v5.runtime.composition_decode --selftest
"""
from __future__ import annotations

import argparse


def _prompt(issue, code, patterns):
    pat = ("\n\nKnown repair patterns from memory (compose / adapt these to the code above):\n"
           + "\n--- pattern ---\n".join(p[:300] for p in patterns)) if patterns else ""
    return (f"Fix this bug.\n\nIssue:\n{issue[:700]}\n\nBuggy code:\n{code[:900]}{pat}\n\n"
            "Output ONLY a search/replace block; SEARCH must be exact code from above:\n"
            "<<<<<<< SEARCH\n<exact buggy code>\n=======\n<fixed code>\n>>>>>>> REPLACE")


def _load(dataset, split, n):
    import numpy as np, torch
    from v5.graph_grower.swe_load import load_instances
    from v5.runtime.operator_discovery import extract_hunks, _fix_text
    from v5.training.providers import RealEmbedder
    emb = RealEmbedder(torch.device("cpu"))

    def E(texts, bs=48):
        out = []
        for i in range(0, len(texts), bs):
            ch = texts[i:i + bs]; d = emb.embed_nodes({str(j): t[:1000] for j, t in enumerate(ch)})
            out.extend(d[str(j)] for j in range(len(ch)))
        return np.asarray(out, dtype=float)
    rows = []
    for i in load_instances(name=dataset, split=split, limit=n):
        hs = extract_hunks(i.get("patch", "") or "")
        if not hs:
            continue
        added = "\n".join(l for _f, _r, a in hs for l in a)
        if not added.strip():
            continue
        rows.append({"issue": (i.get("problem_statement") or "")[:700],
                     "code": "\n".join(l for _f, r, _a in hs for l in r)[:900],
                     "added": added, "fix_text": _fix_text(hs)})
    G = E([r["issue"] + "\nCODE:\n" + r["code"] for r in rows]); Fe = E([r["fix_text"] for r in rows])
    for r, g, f in zip(rows, G, Fe):
        r["g"], r["fe"] = g, f
    return rows


def _train_composer(train, n_ops, epochs=300):
    """Graph-side composer: goal_emb -> softmax weights over operators (basis = KMeans of TRAIN fixes).
    Returns (composer, centroids, rep_patterns). LLM untouched."""
    import numpy as np, torch, torch.nn as nn
    from sklearn.cluster import KMeans
    Ftr = np.stack([r["fe"] for r in train]); Gtr = np.stack([r["g"] for r in train])
    km = KMeans(n_ops, n_init=10, random_state=0).fit(Ftr)
    centroids = km.cluster_centers_
    reps = []                                                        # representative TRAIN fix text per operator
    for c in range(n_ops):
        mem = [i for i in range(len(train)) if km.labels_[i] == c]
        rep = min(mem, key=lambda i: float(np.linalg.norm(Ftr[i] - centroids[c]))) if mem else 0
        reps.append(train[rep]["fix_text"])
    d = Gtr.shape[1]
    C = torch.tensor(centroids, dtype=torch.float32)
    composer = nn.Sequential(nn.Linear(d, 256), nn.ReLU(), nn.Linear(256, n_ops))
    opt = torch.optim.Adam(composer.parameters(), lr=1e-3, weight_decay=1e-4)
    Gt = torch.tensor(Gtr, dtype=torch.float32); Ft = torch.tensor(Ftr, dtype=torch.float32)
    Fn = torch.nn.functional.normalize
    for _ in range(epochs):
        composer.train(); opt.zero_grad()
        pred = torch.softmax(composer(Gt), -1) @ C
        loss = (1 - (Fn(pred) * Fn(Ft)).sum(1)).mean(); loss.backward(); opt.step()
    return composer, centroids, reps


def run(rows, gen_fn, n_ops=24, K=3, epochs=300, log=print):
    import numpy as np, torch
    from v5.runtime.solution_ladder import _emitted_replace, _fidelity
    rng = np.random.RandomState(0); idx = rng.permutation(len(rows))
    nh = len(idx) // 3
    test = [rows[i] for i in idx[:nh]]; train = [rows[i] for i in idx[nh:]]
    n_ops = min(n_ops, max(2, len(train) // 2))                      # basis can't exceed train size
    composer, _cen, reps = _train_composer(train, n_ops, epochs)
    log(f"  composer trained on {len(train)}; testing on {len(test)} (K={K})")
    arms = {"issue_only": [], "top1": [], "topK": []}
    with torch.no_grad():
        for t in test:
            w = torch.softmax(composer(torch.tensor(t["g"], dtype=torch.float32)), -1).numpy()
            order = list(np.argsort(-w))
            pats1 = [reps[order[0]]]
            patsK = [reps[order[j]] for j in range(K)]
            for arm, patterns in (("issue_only", []), ("top1", pats1), ("topK", patsK)):
                rec, _ = _fidelity(_emitted_replace(gen_fn(_prompt(t["issue"], t["code"], patterns))), t["added"])
                arms[arm].append(rec)
    return {a: float(np.mean(v)) if v else 0.0 for a, v in arms.items()}, len(test)


def _report(out, n):
    print(f"\n=== COMPOSITION-DECODE (n={n}) — can the frozen LLM cash in the composition? ===")
    for a in ("issue_only", "top1", "topK"):
        print(f"  {a:11}: realization recall {out[a]:.3f}")
    lift = out["topK"] - out["top1"]
    print(f"\n  compose(top-K) - select(top-1) = {lift:+.3f} -> "
          f"{'COMPOSITION STEERS the LLM (decode works)' if lift > 0.03 else 'composition no better than one op at DECODE (reasoning validated, decode is the LLM wall)'}")
    print(f"  best - issue-only = {max(out['top1'], out['topK']) - out['issue_only']:+.3f}")


def _selftest() -> bool:
    print("composition_decode --selftest: mock LLM copies a delivered pattern -> topK must beat issue-only\n")
    import numpy as np
    rows = []
    for i in range(60):
        rows.append({"issue": "bug", "code": "x = 1", "added": f"x = {i%4}", "fix_text": f"x = {i%4}",
                     "g": np.eye(4)[i % 4].astype("float32"), "fe": np.eye(4)[i % 4].astype("float32")})

    def gen(prompt):                                                 # copies the pattern if one is delivered
        if "pattern" in prompt:
            import re
            m = re.search(r"x = (\d)", prompt.split("memory")[-1])
            return f"<<<<<<< SEARCH\nx = 1\n=======\nx = {m.group(1) if m else 9}\n>>>>>>> REPLACE"
        return "<<<<<<< SEARCH\nx = 1\n=======\nx = 9\n>>>>>>> REPLACE"
    out, n = run(rows, gen, n_ops=4, K=1, epochs=150, log=lambda *a: None)
    print(f"  issue={out['issue_only']:.2f} top1={out['top1']:.2f} topK={out['topK']:.2f}")
    assert out["top1"] > out["issue_only"] + 0.2, "delivering the right pattern must beat issue-only"
    print("\n  COMPOSITION-DECODE SELFTEST -> PASS")
    return True


def main():
    ap = argparse.ArgumentParser(description="Can the frozen LLM decode the operator composition into the fix?")
    ap.add_argument("--model", default="Qwen/Qwen3.5-4B")
    ap.add_argument("--dataset", default="lite")
    ap.add_argument("--split", default="test")
    ap.add_argument("--n", type=int, default=0)
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        raise SystemExit(0 if _selftest() else 1)
    from v5.runtime.solution_ladder import real_gen_fn
    rows = _load(a.dataset, a.split, a.n)
    print(f"[compose-decode] {len(rows)} instances, model={a.model}")
    _report(*run(rows, real_gen_fn(a.model), K=a.k))


if __name__ == "__main__":
    main()
