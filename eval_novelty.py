"""
Run Answerer-v2 against the novelty eval suite, compute graph-grounded
metrics, and write per-question + aggregate results.

Usage:
  python eval_novelty.py --controller mock --eval-jsonl data/novelty_eval.jsonl --out artifacts/novelty_mock.json
  python eval_novelty.py --controller local --eval-jsonl data/novelty_eval.jsonl --out artifacts/novelty_local.json --max-rows 5
  python eval_novelty.py --controller local --filter contradict_   # run a subset by id prefix
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List

os.environ.setdefault("HF_HOME", os.path.join(os.getcwd(), "cache"))
sys.path.insert(0, ".")

from graph_core import MemoryGraph
from answerer_v2 import (
    answer_query_v2_with_session, MockController,
    LlamaServerConfig, LlamaServerController, DEFAULT_LLAMA_SERVER_URL,
)
from anchor_retrieval import retrieve_anchors_v2, anchor_quality
from novelty_metrics import evaluate_run, aggregate


def load_rows(path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-jsonl", default="data/novelty_eval.jsonl")
    ap.add_argument("--controller", default="mock", choices=["mock", "server"])
    ap.add_argument("--out", default="artifacts/novelty_results.json")
    ap.add_argument("--max-rows", type=int, default=0, help="0 = all")
    ap.add_argument("--max-steps", type=int, default=12,
                    help="0 = adaptive (auto-sized after PLAN by plan length); otherwise honored as the hard cap")
    ap.add_argument("--filter", default="", help="Only run rows whose id starts with this prefix")
    ap.add_argument("--ids", default="", help="Comma-separated explicit ids to run")
    ap.add_argument("--server-url", default=DEFAULT_LLAMA_SERVER_URL,
                    help=f"llama-server base url (default {DEFAULT_LLAMA_SERVER_URL})")
    ap.add_argument("--anchor-strategy", default="topk", choices=["legacy", "topk", "mmr"],
                    help="Anchor retrieval strategy (default: topk, empirical winner)")
    ap.add_argument("--role-aware", action="store_true",
                    help="Enable Stage 3 focus/role-aware candidate ranking in the briefing")
    args = ap.parse_args()

    rows = load_rows(args.eval_jsonl)
    if args.ids:
        wanted = {s.strip() for s in args.ids.split(",") if s.strip()}
        rows = [r for r in rows if r.get("id") in wanted]
    elif args.filter:
        rows = [r for r in rows if r.get("id", "").startswith(args.filter)]
    if args.max_rows and args.max_rows < len(rows):
        rows = rows[: args.max_rows]
    print(f"Running {len(rows)} rows with controller={args.controller}")

    if args.controller == "server":
        controller = LlamaServerController(LlamaServerConfig(
            base_url=args.server_url, temperature=0.3, max_tokens=512,
        ))
    else:
        controller = MockController()

    results: List[Dict[str, Any]] = []
    errors: List[Dict[str, Any]] = []
    for i, row in enumerate(rows):
        rid = row.get("id", f"row_{i}")
        print(f"\n[{i+1}/{len(rows)}] {rid}  ({row.get('category')})")
        t0 = time.time()
        try:
            graph = MemoryGraph.load_json(row["graph"])
            graph_basename = os.path.basename(row["graph"]).replace(".json", "")
            # Diagnostic: anchor_quality, computed BEFORE the LLM run.
            # Tells us "did retrieval bring the right evidence?" without
            # confounding it with downstream model behavior.
            if args.anchor_strategy == "legacy":
                from answerer_v1 import retrieve_anchors as _retrieve_legacy
                anchors_used = _retrieve_legacy(row["question"], graph, k=8)
            else:
                anchors_used = retrieve_anchors_v2(
                    row["question"], graph, k=8,
                    strategy=args.anchor_strategy, graph_basename=graph_basename,
                )
            aq = anchor_quality(anchors_used, row.get("required_evidence", []) or [])
            session, packet, trace = answer_query_v2_with_session(
                question=row["question"],
                graph=graph,
                controller=controller,
                max_steps=args.max_steps,
                k_anchors=8,
                anchor_strategy=args.anchor_strategy,
                graph_basename=graph_basename,
                role_aware=args.role_aware,
            )
            metrics = evaluate_run(session, trace, packet.answer, row)
            metrics["elapsed_sec"] = round(time.time() - t0, 1)
            metrics["question"] = row["question"]
            metrics["answer"] = packet.answer
            metrics["steps_taken"] = packet.steps_taken
            metrics["anchor_strategy"] = args.anchor_strategy
            metrics["anchor_quality"] = round(aq, 3)
            metrics["anchors_used"] = anchors_used
            results.append(metrics)
            flag = "PASS" if metrics["novelty_pass"] else "FAIL"
            print(
                f"  {flag}  anchor_q={metrics['anchor_quality']:.2f} "
                f"dep={metrics['graph_dependency']:.2f} "
                f"depth={metrics['max_support_depth']} "
                f"kw={metrics['keyword_coverage']:.2f} "
                f"usage={metrics['usage_coverage']:.2f}({metrics['usage_weak']}/{metrics['usage_pairs']} weak) "
                f"plan_cov={metrics['plan_coverage']:.2f} "
                f"({metrics['elapsed_sec']:.0f}s)"
            )
            if metrics["missing_required_evidence"]:
                print(f"  missing evidence: {metrics['missing_required_evidence']}")
            if metrics["missed_keywords"]:
                print(f"  missed keywords: {metrics['missed_keywords']}")
        except Exception as ex:
            print(f"  ERROR: {ex}")
            traceback.print_exc()
            errors.append({"id": rid, "error": str(ex)})

    summary = aggregate(results)
    out_obj = {
        "controller": args.controller,
        "n_rows": len(results),
        "n_errors": len(errors),
        "summary": summary,
        "errors": errors,
        "results": results,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out_obj, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for k, v in summary.items():
        print(f"  {k}: {v}")
    print(f"\nSaved -> {args.out}")


if __name__ == "__main__":
    main()
