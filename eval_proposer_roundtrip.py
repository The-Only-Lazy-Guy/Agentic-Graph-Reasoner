from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

import torch
from torch.utils.data import DataLoader

from eval_pred_v1 import (
    covered_row_failure_breakdown,
    long_decompose_edge_debug,
    null_span_analysis,
    null_span_by_task,
    per_task_analysis,
    per_task_cover_breakdown,
)
from pred_model import (
    PredAlignNet,
    infer_cand_emb_dim_from_state,
    infer_edge_rel_pair_feat_dim_from_state,
    infer_mem_emb_dim_from_state,
    infer_spec_emb_dim_from_state,
)
from synthesize_node_text import apply_template_synthesis
from train_pred_v1 import PredDataset, collate, compute_metrics, read_jsonl


def reconstruct_pred_row_from_slots(
    row: Mapping[str, Any],
    slots: Sequence[Mapping[str, Any]],
    *,
    apply_synthesis: bool = False,
    graph_cache: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    oracle_goal = dict((row.get("_oracle_goal", {}) or {}))
    used_slots = [slot for slot in slots if bool(slot.get("use"))]
    if apply_synthesis:
        used_slots = apply_template_synthesis(row, used_slots, graph_cache=graph_cache)
    session_nodes = [
        {
            "name": str(slot.get("session_name", f"s{i}")),
            "span_text": str(slot.get("span_text", "")),
            "node_type": str(slot.get("node_type", "unknown")),
        }
        for i, slot in enumerate(used_slots)
    ]
    span_oracle = [
        {
            "session_name": str(slot.get("session_name", f"s{i}")),
            "spec_text": str(slot.get("span_text", "")),
            "node_type": str(slot.get("node_type", "unknown")),
            "best_span_id": slot.get("span_id"),
            "best_score": float(slot.get("oracle_best_score", 0.0)),
            "span_scores": [],
        }
        for i, slot in enumerate(used_slots)
    ]

    oracle_goal["session_nodes"] = session_nodes
    return {
        "id": row.get("id", ""),
        "task_type": row.get("task_type", ""),
        "graph_path": row.get("graph_path", ""),
        "signal": row.get("signal", ""),
        "initial_memory_node_ids": list(row.get("initial_memory_node_ids", []) or []),
        "spans": list(row.get("spans", []) or []),
        "goal": oracle_goal,
        "span_oracle": span_oracle,
        "is_pseudo_goal": False,
        "meta": {
            "num_nodes": len(session_nodes),
            "num_edges": len(oracle_goal.get("session_edges", []) or []),
            "num_attachments": len(oracle_goal.get("memory_attachments", []) or []),
            "num_covered": len(oracle_goal.get("covered_mappings", []) or []),
        },
    }


def reconstruct_pred_row_from_proposer(row: Mapping[str, Any]) -> Dict[str, Any]:
    return reconstruct_pred_row_from_slots(
        row,
        [slot for slot in row.get("target_slots", []) or [] if bool(slot.get("use"))],
    )


def reconstruct_rows(rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    return [reconstruct_pred_row_from_proposer(row) for row in rows]


def reconstruct_rows_with_oracle_synthesis(rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    graph_cache: Dict[str, Any] = {}
    return [
        reconstruct_pred_row_from_slots(
            row,
            [slot for slot in row.get("target_slots", []) or [] if bool(slot.get("use"))],
            apply_synthesis=True,
            graph_cache=graph_cache,
        )
        for row in rows
    ]


def compare_against_source(
    proposer_rows: Sequence[Mapping[str, Any]],
    source_rows: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    source_by_id = {str(row.get("id", "")): row for row in source_rows}
    exact_session_node_order = 0
    exact_session_nodes = 0
    exact_goals = 0
    exact_named_session_nodes = 0
    exact_non_session_goal = 0
    missing_source = 0
    mismatch_examples: List[Dict[str, Any]] = []

    for row in proposer_rows:
        row_id = str(row.get("id", ""))
        src = source_by_id.get(row_id)
        if src is None:
            missing_source += 1
            continue
        recon = reconstruct_pred_row_from_proposer(row)
        recon_goal = recon["goal"]
        src_goal = (src.get("goal", {}) or {})
        recon_nodes = recon_goal.get("session_nodes", []) or []
        src_nodes = src_goal.get("session_nodes", []) or []
        if [node.get("name") for node in recon_nodes] == [node.get("name") for node in src_nodes]:
            exact_session_node_order += 1
        if recon_nodes == src_nodes:
            exact_session_nodes += 1
        recon_by_name = {
            str(node.get("name", "")): {
                "span_text": str(node.get("span_text", "")),
                "node_type": str(node.get("node_type", "unknown")),
            }
            for node in recon_nodes
        }
        src_by_name = {
            str(node.get("name", "")): {
                "span_text": str(node.get("span_text", "")),
                "node_type": str(node.get("node_type", "unknown")),
            }
            for node in src_nodes
        }
        if recon_by_name == src_by_name:
            exact_named_session_nodes += 1
        recon_non_session = dict(recon_goal)
        src_non_session = dict(src_goal)
        recon_non_session.pop("session_nodes", None)
        src_non_session.pop("session_nodes", None)
        if recon_non_session == src_non_session:
            exact_non_session_goal += 1
        if recon_goal == src_goal:
            exact_goals += 1
        elif len(mismatch_examples) < 5:
            mismatch_examples.append(
                {
                    "id": row_id,
                    "reconstructed_goal": recon_goal,
                    "source_goal": src_goal,
                }
            )

    denom = max(len(proposer_rows) - missing_source, 1)
    return {
        "total_rows": len(proposer_rows),
        "missing_source_rows": missing_source,
        "exact_session_node_order": exact_session_node_order,
        "exact_session_node_order_rate": exact_session_node_order / denom,
        "exact_session_nodes": exact_session_nodes,
        "exact_session_nodes_rate": exact_session_nodes / denom,
        "exact_named_session_nodes": exact_named_session_nodes,
        "exact_named_session_nodes_rate": exact_named_session_nodes / denom,
        "exact_non_session_goal": exact_non_session_goal,
        "exact_non_session_goal_rate": exact_non_session_goal / denom,
        "exact_goals": exact_goals,
        "exact_goals_rate": exact_goals / denom,
        "mismatch_examples": mismatch_examples,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Round-trip proposer rows back through the PRED-v2 aligner evaluator")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--proposer-jsonl", required=True)
    ap.add_argument("--source-pred-jsonl", default=None, help="Optional original pred_v1 split for exact round-trip comparison")
    ap.add_argument("--spec-emb-cache", default=None)
    ap.add_argument("--cand-emb-cache", default=None)
    ap.add_argument("--mem-emb-cache", default=None)
    ap.add_argument("--apply-template-synthesis", action="store_true")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--out-json", default=None)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    ckpt = torch.load(args.checkpoint, map_location=device)
    cfg = dict(ckpt.get("args", {}))
    if args.spec_emb_cache:
        cfg["spec_emb_cache"] = args.spec_emb_cache
    if args.cand_emb_cache:
        cfg["cand_emb_cache"] = args.cand_emb_cache
    if args.mem_emb_cache:
        cfg["mem_emb_cache"] = args.mem_emb_cache

    spec_emb_dim = infer_spec_emb_dim_from_state(ckpt["model"])
    cand_emb_dim = infer_cand_emb_dim_from_state(ckpt["model"])
    mem_emb_dim = infer_mem_emb_dim_from_state(ckpt["model"])
    cfg["spec_emb_dim"] = spec_emb_dim
    cfg["cand_emb_dim"] = cand_emb_dim
    cfg["mem_emb_dim"] = mem_emb_dim

    proposer_rows = read_jsonl(args.proposer_jsonl)
    reconstructed_rows = (
        reconstruct_rows_with_oracle_synthesis(proposer_rows)
        if args.apply_template_synthesis
        else reconstruct_rows(proposer_rows)
    )
    ds = PredDataset(
        reconstructed_rows,
        hash_dim=int(cfg.get("hash_dim", 512)),
        spec_emb_cache=args.spec_emb_cache,
        spec_emb_dim_override=spec_emb_dim,
        cand_emb_cache=args.cand_emb_cache,
        cand_emb_dim_override=cand_emb_dim,
        mem_emb_cache=args.mem_emb_cache,
        mem_emb_dim_override=mem_emb_dim,
    )
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate)

    model = PredAlignNet(
        hash_dim=int(cfg.get("hash_dim", 512)),
        hidden_dim=int(cfg.get("hidden_dim", 256)),
        edge_rel_pair_feat_dim=infer_edge_rel_pair_feat_dim_from_state(
            ckpt["model"], int(cfg.get("hidden_dim", 256))
        ),
        spec_emb_dim=spec_emb_dim,
        cand_emb_dim=cand_emb_dim,
        mem_emb_dim=mem_emb_dim,
    ).to(device)
    model.load_state_dict(ckpt["model"])

    global_metrics = compute_metrics(model, loader, device)
    per_task = per_task_analysis(model, reconstructed_rows, cfg, device)
    null_global = null_span_analysis(model, reconstructed_rows, cfg, device)
    null_by_task = null_span_by_task(model, reconstructed_rows, cfg, device)
    cover_by_task = per_task_cover_breakdown(model, reconstructed_rows, cfg, device)
    covered_breakdown = covered_row_failure_breakdown(model, reconstructed_rows, cfg, device)
    long_debug = long_decompose_edge_debug(model, reconstructed_rows, cfg, device)

    source_comparison = None
    if args.source_pred_jsonl:
        source_rows = read_jsonl(args.source_pred_jsonl)
        source_comparison = compare_against_source(proposer_rows, source_rows)

    report = {
        "checkpoint": args.checkpoint,
        "proposer_jsonl": args.proposer_jsonl,
        "source_pred_jsonl": args.source_pred_jsonl,
        "template_synthesis_applied": bool(args.apply_template_synthesis),
        "global": global_metrics,
        "per_task": per_task,
        "null_span_global": null_global,
        "null_span_by_task": null_by_task,
        "cover_by_task": cover_by_task,
        "covered_row_failure_breakdown": covered_breakdown,
        "long_decompose_edge_debug": long_debug,
        "source_comparison": source_comparison,
    }

    print("=== proposer round-trip metrics ===")
    print(json.dumps(global_metrics, indent=2, ensure_ascii=False))
    if source_comparison is not None:
        print("\n=== source comparison ===")
        print(json.dumps(source_comparison, indent=2, ensure_ascii=False))

    if args.out_json:
        Path(args.out_json).write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\nreport written to {args.out_json}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
