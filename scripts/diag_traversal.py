"""Diagnostic: inspect TraversalRanker per-hop behavior on a small compose-like store.
Specifically checks whether the `sims[i] > 0` filter in _top_k discards the 2nd source.
"""
import shutil
import numpy as np
from v5.memory.store import make_mpnet_embedder
from v5.memory.episodic import ImplStore
from v5.runtime.memory_refiner import load_ranker
from v5.runtime.traversal_ranker import TraversalRanker

embed = make_mpnet_embedder()
net, feat_proj, ops, K_r = load_ranker("artifacts/traversal_ranker")

ROOT = "data/diag_store"
shutil.rmtree(ROOT, ignore_errors=True)
st = ImplStore(ROOT, embed_fn=embed)
st.add(ctx_text="fee calculation per item using RATE", old="",
       new="def fee(items): return sum(i.price * RATE for i in items)",
       trace="compute fee from items", file_path="fees.py", task_id="src_fees", kind="create")
st.add(ctx_text="config holds TAX and RATE constants", old="",
       new="TAX = 0.1\nRATE = 0.05",
       trace="config constants", file_path="config.py", task_id="src_config", kind="create")
st.add(ctx_text="checkout totals cart then adds fee", old="",
       new="def checkout(cart): return cart.total + fee(cart.items)",
       trace="checkout combines total and fee", file_path="checkout.py",
       task_id="src_checkout", kind="create")
st.add(ctx_text="clamp helper utility", old="",
       new="def clamp(x): return max(0, min(1, x))",
       trace="clamp utility", file_path="utils.py", task_id="src_utils", kind="create")

spec = "Implement checkout(cart) that returns the total using this project's fee logic."
tr = TraversalRanker(st, concepts=None, embed_fn=embed, refiner_net=net,
                      feat_proj=feat_proj, ops=ops, gap_detector=None, K_steps=K_r)

# Monkeypatch _top_k to log full sim distribution per hop
orig_topk = tr._top_k
def logged_topk(h, cand, ctx):
    ids = [iid for iid, _ in cand]
    sims = np.dot(h, ctx.T) / (np.linalg.norm(h) * np.linalg.norm(ctx, axis=1) + 1e-9)
    order = np.argsort(-sims)
    top2 = [(ids[i], round(float(sims[i]), 3)) for i in order[:2]]
    kept = [(ids[i], round(float(sims[i]), 3)) for i in order[:2] if sims[i] > 0]
    print(f"    _top_k: top2={top2}  kept(>0)={kept}")
    return orig_topk(h, cand, ctx)
tr._top_k = logged_topk

print(f"[diag] {len(st)} impls; spec='{spec[:50]}...'")
res = tr.retrieve(goal=spec, span=spec, file_path="checkout.py")
print("\n[diag] retrieve() records:", [r.get("file_path") for r in res.records])
print("[diag] hop_records:", [[r.get("file_path") for r in hr] for hr in res.hop_records])
print("[diag] all_records:", len(res.records), "hop_count:", res.hops)
