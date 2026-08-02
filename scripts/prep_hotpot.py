"""Decode HotpotQA distractor into a small json. MUST run as its own process: pyarrow (like HF
datasets) segfaults when torch is already loaded in this environment, and membrane imports torch at
module scope. Same reason --grow-cot-docs-path and --math-cot-docs-path exist."""
import os, json, collections, sys
os.environ.setdefault("HF_HOME", r"E:\cache\hf")
from huggingface_hub import hf_hub_download
import pyarrow.parquet as pq

want = int(sys.argv[1]) if len(sys.argv) > 1 else 800
f = hf_hub_download("hotpotqa/hotpot_qa", "distractor/validation-00000-of-00001.parquet",
                    repo_type="dataset")
out = []
for b in pq.ParquetFile(f).iter_batches(
        batch_size=64,
        columns=["id", "question", "answer", "type", "supporting_facts", "context"]):
    for r in b.to_pylist():
        ctx, sf = r["context"], r["supporting_facts"]
        out.append({"id": r["id"], "q": r["question"], "a": r["answer"], "type": r["type"],
                    "paras": [[t, " ".join(s)[:1200]] for t, s in zip(ctx["title"], ctx["sentences"])],
                    "gold": sorted(set(sf["title"]))})
        if len(out) >= want:
            break
    if len(out) >= want:
        break
json.dump(out, open("artifacts/hotpot_multihop.json", "w", encoding="utf-8"))
print(f"wrote {len(out)} -> artifacts/hotpot_multihop.json  "
      f"types={dict(collections.Counter(r['type'] for r in out))}")
