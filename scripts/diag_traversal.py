"""Faithful reproduction of a compose chain's memory at the retrieval point of the
compose session (s3). Store has 3 impls: tax.py, fees.py, catalog.py (the distractor).
The compose spec withholds both rates. We run the REAL ranker's retrieve and print
how many records come back + their file_paths.
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

tax, fee, low = 0.10, 0.04, 5
# Real memory.write stores ctx_text = goal[:400] (the SESSION SPEC), NOT the code.
tax_spec = "Create tax.py. Constant TAX_RATE = 0.1. taxed(amount) returns amount*(1+TAX_RATE) rounded to 2 decimals."
fee_spec = "Create fees.py. Constant FEE_RATE = 0.04. service_fee(amount) returns amount*FEE_RATE rounded to 2 decimals."
cat_spec = "Create catalog.py. make_sku(n) returns an id string. is_low(stock, name) is True when the stock is strictly below 5."
st.add(ctx_text=tax_spec[:400], old="",
       new="TAX_RATE = 0.1\n\ndef taxed(amount):\n    return round(amount * (1 + TAX_RATE), 2)\n",
       trace=tax_spec[:400], file_path="tax.py", task_id="src_tax", kind="create")
st.add(ctx_text=fee_spec[:400], old="",
       new="FEE_RATE = 0.04\n\ndef service_fee(amount):\n    return round(amount * FEE_RATE, 2)\n",
       trace=fee_spec[:400], file_path="fees.py", task_id="src_fees", kind="create")
st.add(ctx_text=cat_spec[:400], old="",
       new="def make_sku(n):\n    return f'SKU-{n:03d}'\n\ndef is_low(stock, name):\n    return stock.get(name, 0) < 5\n",
       trace=cat_spec[:400], file_path="catalog.py", task_id="src_catalog", kind="create")

spec = ("Create checkout.py. final_price(p) returns p plus this project's tax charged on p "
        "plus this project's service fee charged on p -- that is, the base price with BOTH "
        "established rates applied to it and summed (base + tax-on-base + fee-on-base), rounded "
        "to 2 decimals. Do NOT restate the rates; use the two values this project already "
        "established. Write it self-contained in checkout.py.")

import torch
from v5.runtime.gap_detector import GapDetector
gap = GapDetector(d_hidden=256, d_in=768)
gap.load_state_dict(torch.load("artifacts/traversal_ranker/gap.pt", weights_only=True, map_location="cpu"))
gap.eval()

tr = TraversalRanker(st, concepts=None, embed_fn=embed, refiner_net=net,
                      feat_proj=feat_proj, ops=ops, gap_detector=gap, K_steps=K_r)
res = tr.retrieve(goal=spec, span=spec, file_path="checkout.py")
print("[diag] GAP-DETECTOR ON")
print("[diag] records:", [(r.get("file_path"), r.get("task_id")) for r in res.records])
print("[diag] n_records:", len(res.records), " hops:", res.hops)
print("[diag] hop_records:", [[r.get("file_path") for r in hr] for hr in res.hop_records])

# Sanity: even a GARBAGE (random) initial_h must return >=2 records with the fix
# (the >0 filter is gone). If a run returns 1 record, it is running PRE-FIX code.
rng_h = np.random.RandomState(0).randn(768).astype(np.float32)
res2 = tr.retrieve(goal=spec, span=spec, file_path="checkout.py", initial_h=rng_h)
print("[diag] RANDOM initial_h -> n_records:", len(res2.records),
      "files:", [r.get("file_path") for r in res2.records])
