"""Test retrieval returns >=2 records with fixed _top_k (no >0 filter)."""
import shutil, tempfile, numpy as np
from v5.memory.store import make_fake_embedder
from v5.memory.episodic import ImplStore
from v5.runtime.traversal_ranker import TraversalRanker

embed = make_fake_embedder()
d = tempfile.mkdtemp()
st = ImplStore(d, embed_fn=embed)

st.add(ctx_text="Create tax.py TAX_RATE=0.1", old="",
       new="TAX_RATE = 0.1\ndef taxed(amount): return round(amount*(1+TAX_RATE),2)",
       trace="tax spec", file_path="tax.py", task_id="t0", kind="create")
st.add(ctx_text="Create fees.py FEE_RATE=0.04", old="",
       new="FEE_RATE = 0.04\ndef service_fee(amount): return round(amount*FEE_RATE,2)",
       trace="fees spec", file_path="fees.py", task_id="t1", kind="create")
st.add(ctx_text="Create catalog.py make_sku", old="",
       new="def make_sku(n): return f'SKU-{n:03d}'\ndef is_low(stock,name): return stock.get(name,0)<5",
       trace="cat spec", file_path="catalog.py", task_id="t2", kind="create")
print("store entries:", len(st))

tr = TraversalRanker(st, None, embed, refiner_net=None, feat_proj=None, ops=None,
                     gap_detector=None, K_steps=1, pool_k=16, k_impl=2, max_hops=3)
spec = "Create checkout.py final_price(p) using tax and fees."
res = tr.retrieve(goal=spec, span=spec, file_path="checkout.py")
print("n_records:", len(res.records), "  files:", [r.get("file_path") for r in res.records])
print("hops:", res.hops)
print("hop_records:", [[r.get("file_path") for r in hr] for hr in res.hop_records])

# Test with all-negative sims (worst case for old _top_k)
ctx = tr._embed(spec)
# Build a query vector that points OPPOSITE to all store entries
neg_q = -ctx / (np.linalg.norm(ctx)+1e-9)
res2 = tr.retrieve(goal=spec, span=spec, file_path="checkout.py", initial_h=neg_q)
print("\nNEGATIVE query (all sims <0):")
print("n_records:", len(res2.records), "  files:", [r.get("file_path") for r in res2.records])
assert len(res2.records) >= 2, f"NEGATIVE query should return >=2 records with fix, got {len(res2.records)}"
print("PASS: all-negative query returns >=2 records")

# Test with zero query
zero_h = np.zeros(768, dtype=np.float32)
res3 = tr.retrieve(goal=spec, span=spec, file_path="checkout.py", initial_h=zero_h)
print("\nZERO query:")
print("n_records:", len(res3.records), "  files:", [r.get("file_path") for r in res3.records])
assert len(res3.records) >= 2, f"ZERO query should return >=2 records, got {len(res3.records)}"
print("PASS: zero query returns >=2 records")

# Now test with gap detector that always stops
from v5.runtime.gap_detector import GapDetector
import torch
gd = GapDetector(d_hidden=2, d_in=768)
gd.eval()
tr2 = TraversalRanker(st, None, embed, refiner_net=None, feat_proj=None, ops=None,
                      gap_detector=gd, K_steps=1, pool_k=16, k_impl=2, max_hops=3)
res4 = tr2.retrieve(goal=spec, span=spec, file_path="checkout.py")
# gap detector with random weights may stop early, but hop 0 ALWAYS runs
print("\nWith untrained gap detector:")
print("n_records:", len(res4.records), "  files:", [r.get("file_path") for r in res4.records])
print("hops:", res4.hops)

shutil.rmtree(d, ignore_errors=True)
print("\nAll tests done.")
