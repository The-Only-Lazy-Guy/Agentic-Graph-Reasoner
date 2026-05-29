from __future__ import annotations

"""
build_holdout_embeddings.py

For the manual holdout samples, the candidate span texts are new and not
present in the existing pred_v1 caches. This script computes MiniLM
embeddings for every span text in the holdout jsonl and writes a merged
cand cache that contains both the existing pred_v1 keys (so old shapes
still work) plus the new holdout-specific embeddings.

Memory texts are unchanged: every memory id in the holdout is from the
existing physics1.json graph and is already in the pred_v1 mem cache.
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np

from train_pred_v1 import text_cache_key


def collect_holdout_cand_texts(holdout_path: str) -> List[str]:
    texts: List[str] = []
    seen: set[str] = set()
    with open(holdout_path, "r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            for span in row.get("spans", []) or []:
                t = str(span.get("text", "") or "").strip()
                if t and t not in seen:
                    seen.add(t)
                    texts.append(t)
    return texts


def merge_caches(
    existing_npz_path: str,
    new_texts: List[str],
    out_path: str,
    *,
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
) -> None:
    existing = np.load(existing_npz_path, allow_pickle=False)
    existing_keys = [str(k) for k in existing["keys"].tolist()]
    existing_texts = [str(t) for t in existing["texts"].tolist()]
    existing_embeddings = existing["embeddings"]
    dim = existing_embeddings.shape[1]
    existing_key_set = set(existing_keys)

    missing_texts = [t for t in new_texts if text_cache_key(t) not in existing_key_set]
    if not missing_texts:
        print(json.dumps({"out": out_path, "skipped": "no new texts to embed"}))
        return

    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(model_name)
    new_embeddings = model.encode(
        missing_texts,
        batch_size=32,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    ).astype("float32")
    if new_embeddings.shape[1] != dim:
        raise RuntimeError(
            f"Embedding dim mismatch: existing={dim}, new={new_embeddings.shape[1]}"
        )

    new_keys = [text_cache_key(t) for t in missing_texts]
    merged_keys = existing_keys + new_keys
    merged_texts = existing_texts + missing_texts
    merged_embeddings = np.vstack([existing_embeddings, new_embeddings])

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        keys=np.array(merged_keys),
        texts=np.array(merged_texts),
        embeddings=merged_embeddings,
        model=np.array([model_name]),
    )
    print(json.dumps({
        "out": out_path,
        "added_texts": len(missing_texts),
        "total_texts": len(merged_keys),
        "dim": dim,
    }, indent=2))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--holdout-jsonl", default="artifacts/holdout/proposer_holdout.jsonl")
    ap.add_argument(
        "--existing-cand-cache",
        default="artifacts/spec_emb_cache/pred_v1_fix8_minilm_cand.npz",
    )
    ap.add_argument(
        "--out-cand-cache",
        default="artifacts/spec_emb_cache/holdout_minilm_cand.npz",
    )
    args = ap.parse_args()

    cand_texts = collect_holdout_cand_texts(args.holdout_jsonl)
    merge_caches(args.existing_cand_cache, cand_texts, args.out_cand_cache)


if __name__ == "__main__":
    main()
