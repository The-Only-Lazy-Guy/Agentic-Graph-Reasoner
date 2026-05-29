from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping

import numpy as np

from graph_core import MemoryGraph
from train_pred_v1 import text_cache_key


def read_jsonl(path: str | Path) -> Iterable[Dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8-sig") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def collect_texts(paths: List[str], text_source: str) -> List[str]:
    seen: set[str] = set()
    texts: List[str] = []
    graph_cache: Dict[str, MemoryGraph] = {}
    for path in paths:
        for row in read_jsonl(path):
            if text_source in {"spec", "both"}:
                goal: Mapping[str, Any] = row.get("goal", {}) or {}
                for spec in goal.get("session_nodes", []) or []:
                    text = str(spec.get("span_text", "")).strip()
                    if text and text not in seen:
                        seen.add(text)
                        texts.append(text)
            if text_source in {"cand", "both"}:
                for span in row.get("spans", []) or []:
                    text = str(span.get("text", "")).strip()
                    if text and text not in seen:
                        seen.add(text)
                        texts.append(text)
            if text_source in {"mem", "both"}:
                graph_path = str(row.get("graph_path", ""))
                if not graph_path:
                    continue
                if graph_path not in graph_cache:
                    graph_cache[graph_path] = MemoryGraph.load_json(graph_path)
                graph = graph_cache[graph_path]
                goal: Mapping[str, Any] = row.get("goal", {}) or {}
                memory_ids: List[str] = []
                for mem in row.get("initial_memory_node_ids", []) or []:
                    mem = str(mem)
                    if mem and mem in graph.nodes and mem not in memory_ids:
                        memory_ids.append(mem)
                for att in goal.get("memory_attachments", []) or []:
                    mem = str(att.get("memory_id", ""))
                    if mem and mem in graph.nodes and mem not in memory_ids:
                        memory_ids.append(mem)
                for cov in goal.get("covered_mappings", []) or []:
                    mem = str(cov.get("memory_id", ""))
                    if mem and mem in graph.nodes and mem not in memory_ids:
                        memory_ids.append(mem)
                for mem in memory_ids:
                    text = str(graph.nodes[mem].text).strip()
                    if text and text not in seen:
                        seen.add(text)
                        texts.append(text)
    return texts


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl", action="append", required=True, help="Input pred JSONL. Can be provided multiple times.")
    ap.add_argument("--out", required=True, help="Output .npz cache path")
    ap.add_argument("--text-source", choices=["spec", "cand", "mem", "both"], default="spec")
    ap.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise SystemExit(
            "sentence-transformers is not installed. Install it before running this cache precompute step."
        ) from exc

    texts = collect_texts(args.jsonl, args.text_source)
    model = SentenceTransformer(args.model, device=args.device)
    embeddings = model.encode(
        texts,
        batch_size=args.batch_size,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    ).astype("float32")

    keys = np.array([text_cache_key(text) for text in texts])
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        keys=keys,
        texts=np.array(texts),
        embeddings=embeddings,
        model=np.array([args.model]),
    )
    print(json.dumps({
        "out": str(out),
        "model": args.model,
        "text_source": args.text_source,
        "n_texts": len(texts),
        "dim": int(embeddings.shape[1]) if len(texts) else 0,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
