from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Mapping, Tuple

import torch
from torch.utils.data import DataLoader

from pred_model import (
    MEM_LINK_KIND_TO_ID,
    REL_WITH_NONE,
    SPEC_TYPE_TO_ID,
    SPEC_TYPE_VOCAB,
    PredAlignNet,
    infer_cand_emb_dim_from_state,
    infer_edge_rel_pair_feat_dim_from_state,
    infer_mem_emb_dim_from_state,
    infer_spec_emb_dim_from_state,
)
from train_pred_v1 import (
    IGNORE,
    PredDataset,
    collate,
    compute_metrics,
    decode_edge_predictions,
    decode_mem_kind_predictions,
    decode_span_predictions,
    read_jsonl,
    to_device,
)


# ---------------------------------------------------------------------------
# Per-task breakdown
# ---------------------------------------------------------------------------

def compute_metrics_subset(
    model: PredAlignNet,
    rows: List[Mapping[str, Any]],
    cfg: Dict[str, Any],
    device: torch.device,
) -> Dict[str, Any]:
    ds = PredDataset(
        rows,
        hash_dim=int(cfg.get("hash_dim", 512)),
        spec_emb_cache=cfg.get("spec_emb_cache"),
        spec_emb_dim_override=int(cfg.get("spec_emb_dim", 0)),
        cand_emb_cache=cfg.get("cand_emb_cache"),
        cand_emb_dim_override=int(cfg.get("cand_emb_dim", 0)),
        mem_emb_cache=cfg.get("mem_emb_cache"),
        mem_emb_dim_override=int(cfg.get("mem_emb_dim", 0)),
    )
    loader = DataLoader(ds, batch_size=32, shuffle=False, collate_fn=collate)
    return compute_metrics(model, loader, device)


def per_task_analysis(
    model: PredAlignNet,
    rows: List[Mapping[str, Any]],
    cfg: Dict[str, Any],
    device: torch.device,
) -> Dict[str, Any]:
    groups: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
    for r in rows:
        groups[str(r.get("task_type", "unknown"))].append(r)

    out: Dict[str, Any] = {}
    for tt in sorted(groups):
        m = compute_metrics_subset(model, groups[tt], cfg, device)
        out[tt] = {"n": len(groups[tt]), **m}
    return out


# ---------------------------------------------------------------------------
# Null-span / bridge analysis
# ---------------------------------------------------------------------------

def null_span_analysis(
    model: PredAlignNet,
    rows: List[Mapping[str, Any]],
    cfg: Dict[str, Any],
    device: torch.device,
) -> Dict[str, Any]:
    """
    Analyse model behaviour on specs whose oracle best_span_id is None.

    In the padded batch y_span=max_c means "none class" (see collate).
    The none logit is appended last, so none_index = span_logits.size(-1)-1 = max_c.
    """
    ds = PredDataset(
        rows,
        hash_dim=int(cfg.get("hash_dim", 512)),
        spec_emb_cache=cfg.get("spec_emb_cache"),
        spec_emb_dim_override=int(cfg.get("spec_emb_dim", 0)),
        cand_emb_cache=cfg.get("cand_emb_cache"),
        cand_emb_dim_override=int(cfg.get("cand_emb_dim", 0)),
        mem_emb_cache=cfg.get("mem_emb_cache"),
        mem_emb_dim_override=int(cfg.get("mem_emb_dim", 0)),
    )
    loader = DataLoader(ds, batch_size=32, shuffle=False, collate_fn=collate)

    # Global null/nonnull counters
    null_total = null_pred_correct = 0
    nonnull_total = nonnull_false_none = 0

    # Per-node-type counters  {type_name: [total_null, correct_none, total_nonnull, false_none]}
    type_counters: Dict[str, List[int]] = {t: [0, 0, 0, 0] for t in SPEC_TYPE_VOCAB}

    model.eval()
    with torch.no_grad():
        for batch in loader:
            batch = to_device(batch, device)
            out = model(batch)
            span_pred = decode_span_predictions(
                out["span_logits"],
                batch.spec_mask,
                batch.cand_mask,
            )
            none_index = out["span_logits"].size(-1) - 1

            spec_valid = batch.spec_mask
            is_null = (batch.y_span == none_index) & spec_valid
            is_nonnull = (batch.y_span != none_index) & spec_valid
            pred_none = span_pred == none_index

            null_total += int(is_null.sum().item())
            null_pred_correct += int((pred_none & is_null).sum().item())
            nonnull_total += int(is_nonnull.sum().item())
            nonnull_false_none += int((pred_none & is_nonnull).sum().item())

            # Per-type breakdown
            B, S = spec_valid.shape
            for b in range(B):
                for s in range(S):
                    if not spec_valid[b, s]:
                        continue
                    tid = int(batch.spec_type_ids[b, s].item())
                    tname = SPEC_TYPE_VOCAB[tid] if tid < len(SPEC_TYPE_VOCAB) else "unknown"
                    c = type_counters[tname]
                    if is_null[b, s]:
                        c[0] += 1
                        if pred_none[b, s]:
                            c[1] += 1
                    else:
                        c[2] += 1
                        if pred_none[b, s]:
                            c[3] += 1

    per_type: Dict[str, Any] = {}
    for tname, (null_n, null_ok, nonnull_n, false_none_n) in type_counters.items():
        if null_n + nonnull_n == 0:
            continue
        per_type[tname] = {
            "null_total": null_n,
            "null_pred_none_rate": round(null_ok / max(null_n, 1), 4),
            "nonnull_total": nonnull_n,
            "nonnull_false_none_rate": round(false_none_n / max(nonnull_n, 1), 4),
        }

    return {
        "null_span_total": null_total,
        "null_span_pred_none_rate": round(null_pred_correct / max(null_total, 1), 4),
        "nonnull_total": nonnull_total,
        "nonnull_false_none_rate": round(nonnull_false_none / max(nonnull_total, 1), 4),
        "per_node_type": per_type,
    }


# ---------------------------------------------------------------------------
# Per-task null-span table
# ---------------------------------------------------------------------------

def null_span_by_task(
    model: PredAlignNet,
    rows: List[Mapping[str, Any]],
    cfg: Dict[str, Any],
    device: torch.device,
) -> Dict[str, Any]:
    groups: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
    for r in rows:
        groups[str(r.get("task_type", "unknown"))].append(r)

    out: Dict[str, Any] = {}
    for tt in sorted(groups):
        a = null_span_analysis(model, groups[tt], cfg, device)
        if a["null_span_total"] > 0 or a["nonnull_total"] > 0:
            out[tt] = a
    return out


# ---------------------------------------------------------------------------
# Per-task cover precision / recall breakdown (locates FP source)
# ---------------------------------------------------------------------------

def per_task_cover_breakdown(
    model: PredAlignNet,
    rows: List[Mapping[str, Any]],
    cfg: Dict[str, Any],
    device: torch.device,
) -> Dict[str, Any]:
    """
    Break down cover TP/FP/FN by task type.

    This answers whether false positive covers come from non-covered task rows
    (task type != covered_long_signal) or from within covered rows where the
    model assigns cover to the wrong memory candidate.
    """
    groups: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
    for r in rows:
        groups[str(r.get("task_type", "unknown"))].append(r)

    out: Dict[str, Any] = {}
    for tt in sorted(groups):
        ds = PredDataset(
            groups[tt],
            hash_dim=int(cfg.get("hash_dim", 512)),
            spec_emb_cache=cfg.get("spec_emb_cache"),
            spec_emb_dim_override=int(cfg.get("spec_emb_dim", 0)),
            cand_emb_cache=cfg.get("cand_emb_cache"),
            cand_emb_dim_override=int(cfg.get("cand_emb_dim", 0)),
            mem_emb_cache=cfg.get("mem_emb_cache"),
            mem_emb_dim_override=int(cfg.get("mem_emb_dim", 0)),
        )
        loader = DataLoader(ds, batch_size=32, shuffle=False, collate_fn=collate)
        tp = fp = fn = 0
        model.eval()
        with torch.no_grad():
            for batch in loader:
                batch = to_device(batch, device)
                logits = model(batch)
                mem_kind_pred = decode_mem_kind_predictions(logits["mem_kind_logits"], batch.mem_mask)
                mem_mask = batch.mem_mask[:, None, :].expand_as(batch.y_mem_kind)
                gold_cover = (batch.y_mem_kind == MEM_LINK_KIND_TO_ID["cover"]) & mem_mask
                pred_cover = (mem_kind_pred == MEM_LINK_KIND_TO_ID["cover"]) & mem_mask
                tp += int((pred_cover & gold_cover).sum().item())
                fp += int((pred_cover & ~gold_cover).sum().item())
                fn += int((~pred_cover & gold_cover).sum().item())
        prec = tp / max(tp + fp, 1)
        rec = tp / max(tp + fn, 1)
        f1 = 0.0 if prec + rec == 0 else 2 * prec * rec / (prec + rec)
        out[tt] = {
            "n": len(groups[tt]),
            "cover_tp": tp, "cover_fp": fp, "cover_fn": fn,
            "cover_precision": round(prec, 4),
            "cover_recall": round(rec, 4),
            "cover_f1": round(f1, 4),
        }
    return out


# ---------------------------------------------------------------------------
# Covered-row failure breakdown
# ---------------------------------------------------------------------------

def covered_row_failure_breakdown(
    model: PredAlignNet,
    rows: List[Mapping[str, Any]],
    cfg: Dict[str, Any],
    device: torch.device,
) -> Dict[str, Any]:
    """
    For covered_long_signal rows only, report what fraction of rows pass each
    component check, and which single component is responsible for isolated failures.

    This answers: is span accuracy or cover accuracy the tighter bottleneck for
    covered row completion, or are they roughly equal?
    """
    covered = [r for r in rows if str(r.get("task_type", "")) == "covered_long_signal"]
    if not covered:
        return {}

    ds = PredDataset(
        covered,
        hash_dim=int(cfg.get("hash_dim", 512)),
        spec_emb_cache=cfg.get("spec_emb_cache"),
        spec_emb_dim_override=int(cfg.get("spec_emb_dim", 0)),
        cand_emb_cache=cfg.get("cand_emb_cache"),
        cand_emb_dim_override=int(cfg.get("cand_emb_dim", 0)),
        mem_emb_cache=cfg.get("mem_emb_cache"),
        mem_emb_dim_override=int(cfg.get("mem_emb_dim", 0)),
    )
    loader = DataLoader(ds, batch_size=32, shuffle=False, collate_fn=collate)

    span_ok_n = commit_ok_n = edge_ok_n = rel_ok_n = cover_ok_n = mem_rel_ok_n = 0
    span_and_cover_ok_n = all_ok_n = total_n = 0
    isolated: Dict[str, int] = {"span": 0, "edge": 0, "rel": 0, "cover": 0, "mem_rel": 0, "multi": 0}

    model.eval()
    with torch.no_grad():
        for batch in loader:
            batch = to_device(batch, device)
            out = model(batch)

            span_pred = decode_span_predictions(out["span_logits"], batch.spec_mask, batch.cand_mask)
            commit_pred = out["commit_logits"].argmax(dim=-1)
            edge_exist_pred = decode_edge_predictions(out["edge_exist_logits"], batch.edge_mask)
            edge_rel_pred = out["edge_rel_logits"].argmax(dim=-1)
            mem_kind_pred = decode_mem_kind_predictions(out["mem_kind_logits"], batch.mem_mask)
            mem_rel_pred = out["mem_rel_logits"].argmax(dim=-1)

            spec_valid = batch.spec_mask
            span_match = (span_pred == batch.y_span) & spec_valid
            gold_edge = (batch.y_edge_exist > 0.0) & batch.edge_mask
            mem_mask = batch.mem_mask[:, None, :].expand_as(batch.y_mem_kind)
            gold_cover = (batch.y_mem_kind == MEM_LINK_KIND_TO_ID["cover"]) & mem_mask
            pred_cover = (mem_kind_pred == MEM_LINK_KIND_TO_ID["cover"]) & mem_mask
            gold_attach = (batch.y_mem_kind == MEM_LINK_KIND_TO_ID["attach"]) & mem_mask

            B = batch.signal_bow.size(0)
            for b in range(B):
                total_n += 1

                s_ok = bool(span_match[b, spec_valid[b]].all().item()) if spec_valid[b].any() else True
                c_ok = bool((commit_pred[b] == batch.y_commit[b]).item())
                e_ok = (
                    bool(torch.equal(edge_exist_pred[b][batch.edge_mask[b]], gold_edge[b][batch.edge_mask[b]]))
                    if batch.edge_mask[b].any() else True
                )
                r_ok = (
                    bool(torch.equal(edge_rel_pred[b][gold_edge[b]], batch.y_edge_rel[b][gold_edge[b]]))
                    if gold_edge[b].any() else True
                )
                cov_ok = (
                    bool(torch.equal(pred_cover[b][mem_mask[b]], gold_cover[b][mem_mask[b]]))
                    if mem_mask[b].any() else True
                )
                mr_ok = (
                    bool(torch.equal(mem_rel_pred[b][gold_attach[b]], batch.y_mem_rel[b][gold_attach[b]]))
                    if gold_attach[b].any() else True
                )

                if s_ok:   span_ok_n   += 1
                if c_ok:   commit_ok_n += 1
                if e_ok:   edge_ok_n   += 1
                if r_ok:   rel_ok_n    += 1
                if cov_ok: cover_ok_n  += 1
                if mr_ok:  mem_rel_ok_n += 1
                if s_ok and cov_ok: span_and_cover_ok_n += 1

                row_ok = s_ok and c_ok and e_ok and r_ok and cov_ok and mr_ok
                if row_ok:
                    all_ok_n += 1
                else:
                    fails = (
                        ([] if s_ok  else ["span"])
                        + ([] if e_ok  else ["edge"])
                        + ([] if r_ok  else ["rel"])
                        + ([] if cov_ok else ["cover"])
                        + ([] if mr_ok  else ["mem_rel"])
                    )
                    if len(fails) == 1:
                        isolated[fails[0]] += 1
                    else:
                        isolated["multi"] += 1

    n = max(total_n, 1)
    return {
        "covered_rows": total_n,
        "row_complete_rate": round(all_ok_n / n, 4),
        "component_row_pass_rate": {
            "span":         round(span_ok_n    / n, 4),
            "commit":       round(commit_ok_n  / n, 4),
            "edge":         round(edge_ok_n    / n, 4),
            "rel":          round(rel_ok_n     / n, 4),
            "cover":        round(cover_ok_n   / n, 4),
            "mem_rel":      round(mem_rel_ok_n / n, 4),
            "span_and_cover": round(span_and_cover_ok_n / n, 4),
        },
        "isolated_failure_counts": isolated,
    }


# ---------------------------------------------------------------------------
# Long-decompose edge debug
# ---------------------------------------------------------------------------

def long_decompose_edge_debug(
    model: PredAlignNet,
    rows: List[Mapping[str, Any]],
    cfg: Dict[str, Any],
    device: torch.device,
    *,
    sample_limit: int = 15,
) -> Dict[str, Any]:
    """
    Diagnose false-positive edge predictions on long_decompose rows.

    Classification is done in two passes:
    1. positional pair match, ignoring relation
    2. reverse-direction positional match

    FP classes:
    - relation_error: (i, j) exists in gold but predicted relation differs
    - direction_error: (j, i) exists in gold but (i, j) does not
    - spurious_pair: neither direction exists in gold
    - threshold_artifact is a score bucket annotation on top of the above
    """
    target_rows = [r for r in rows if str(r.get("task_type", "")) == "long_decompose"]
    if not target_rows:
        return {}

    ds = PredDataset(
        target_rows,
        hash_dim=int(cfg.get("hash_dim", 512)),
        spec_emb_cache=cfg.get("spec_emb_cache"),
        spec_emb_dim_override=int(cfg.get("spec_emb_dim", 0)),
        cand_emb_cache=cfg.get("cand_emb_cache"),
        cand_emb_dim_override=int(cfg.get("cand_emb_dim", 0)),
        mem_emb_cache=cfg.get("mem_emb_cache"),
        mem_emb_dim_override=int(cfg.get("mem_emb_dim", 0)),
    )
    loader = DataLoader(ds, batch_size=1, shuffle=False, collate_fn=collate)

    fp_type_counts: Dict[str, int] = {
        "relation_error": 0,
        "direction_error": 0,
        "spurious_pair": 0,
    }
    relation_confusion: Dict[str, Dict[str, int]] = {}
    fp_score_buckets: Dict[str, int] = {
        "0.50-0.65": 0,
        "0.65-0.80": 0,
        "0.80+": 0,
    }
    row_debug: List[Dict[str, Any]] = []

    model.eval()
    with torch.no_grad():
        for row, batch in zip(target_rows, loader):
            batch = to_device(batch, device)
            out = model(batch)
            edge_scores = torch.sigmoid(out["edge_exist_logits"])[0]
            edge_exist_pred = decode_edge_predictions(out["edge_exist_logits"], batch.edge_mask)[0]
            edge_rel_pred = out["edge_rel_logits"][0].argmax(dim=-1)

            gold_edge = (batch.y_edge_exist[0] > 0.5) & batch.edge_mask[0]
            gold_rel = batch.y_edge_rel[0]

            gold_edges: List[Dict[str, Any]] = []
            gold_pos: Dict[Tuple[int, int], str] = {}
            for i in range(gold_edge.size(0)):
                for j in range(gold_edge.size(1)):
                    if bool(gold_edge[i, j].item()):
                        rel_id = int(gold_rel[i, j].item())
                        gold_edges.append({
                            "i": i,
                            "j": j,
                            "rel_id": rel_id,
                            "rel": REL_WITH_NONE[rel_id] if 0 <= rel_id < len(REL_WITH_NONE) else "unknown",
                        })
                        gold_pos[(i, j)] = rel_id

            pred_edges: List[Dict[str, Any]] = []
            row_fp_count = 0
            for i in range(edge_exist_pred.size(0)):
                for j in range(edge_exist_pred.size(1)):
                    if not bool(edge_exist_pred[i, j].item()):
                        continue
                    score = float(edge_scores[i, j].item())
                    pred_rel_id = int(edge_rel_pred[i, j].item())
                    pos_in_gold = (i, j) in gold_pos
                    rev_in_gold = (j, i) in gold_pos
                    gold_rel_id = gold_pos.get((i, j))
                    fp_type = None
                    if pos_in_gold and gold_rel_id == pred_rel_id:
                        pred_edges.append({
                            "i": i,
                            "j": j,
                            "pred_rel_id": pred_rel_id,
                            "pred_rel": REL_WITH_NONE[pred_rel_id] if 0 <= pred_rel_id < len(REL_WITH_NONE) else "unknown",
                            "score": round(score, 4),
                            "match": "exact",
                        })
                        continue
                    if pos_in_gold:
                        fp_type = "relation_error"
                        gold_rel_name = REL_WITH_NONE[gold_rel_id] if gold_rel_id is not None and 0 <= gold_rel_id < len(REL_WITH_NONE) else "unknown"
                        pred_rel_name = REL_WITH_NONE[pred_rel_id] if 0 <= pred_rel_id < len(REL_WITH_NONE) else "unknown"
                        relation_confusion.setdefault(gold_rel_name, {})
                        relation_confusion[gold_rel_name][pred_rel_name] = relation_confusion[gold_rel_name].get(pred_rel_name, 0) + 1
                    elif rev_in_gold:
                        fp_type = "direction_error"
                    else:
                        fp_type = "spurious_pair"

                    fp_type_counts[fp_type] += 1
                    if score < 0.65:
                        fp_score_buckets["0.50-0.65"] += 1
                    elif score < 0.80:
                        fp_score_buckets["0.65-0.80"] += 1
                    else:
                        fp_score_buckets["0.80+"] += 1
                    row_fp_count += 1
                    pred_edges.append({
                        "i": i,
                        "j": j,
                        "pred_rel_id": pred_rel_id,
                        "pred_rel": REL_WITH_NONE[pred_rel_id] if 0 <= pred_rel_id < len(REL_WITH_NONE) else "unknown",
                        "score": round(score, 4),
                        "pos_in_gold": pos_in_gold,
                        "rev_in_gold": rev_in_gold,
                        "gold_rel_id": gold_rel_id,
                        "gold_rel": REL_WITH_NONE[gold_rel_id] if gold_rel_id is not None and 0 <= gold_rel_id < len(REL_WITH_NONE) else None,
                        "fp_type": fp_type,
                    })

            if len(row_debug) < sample_limit:
                row_debug.append({
                    "row_id": row.get("id", ""),
                    "gold_edges": gold_edges,
                    "pred_edges": pred_edges,
                    "fp_count": row_fp_count,
                })

    total_fp = sum(fp_type_counts.values())
    return {
        "rows": len(target_rows),
        "total_fp": total_fp,
        "fp_type_counts": fp_type_counts,
        "fp_type_rates": {k: round(v / max(total_fp, 1), 4) for k, v in fp_type_counts.items()},
        "relation_confusion": relation_confusion,
        "fp_score_buckets": fp_score_buckets,
        "sample_rows": row_debug,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--jsonl", default="artifacts/pred_v1_20260511/pred_val.jsonl")
    ap.add_argument("--spec-emb-cache", default=None)
    ap.add_argument("--cand-emb-cache", default=None)
    ap.add_argument("--mem-emb-cache", default=None)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--out-json", default=None, help="Write full report to this path")
    ap.add_argument("--cpu", action="store_true")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    ckpt = torch.load(args.checkpoint, map_location=device)
    cfg = ckpt.get("args", {})
    if args.spec_emb_cache:
        cfg = dict(cfg)
        cfg["spec_emb_cache"] = args.spec_emb_cache
    if args.cand_emb_cache:
        cfg = dict(cfg)
        cfg["cand_emb_cache"] = args.cand_emb_cache
    if args.mem_emb_cache:
        cfg = dict(cfg)
        cfg["mem_emb_cache"] = args.mem_emb_cache

    spec_emb_dim = infer_spec_emb_dim_from_state(ckpt["model"])
    cand_emb_dim = infer_cand_emb_dim_from_state(ckpt["model"])
    mem_emb_dim = infer_mem_emb_dim_from_state(ckpt["model"])
    cfg = dict(cfg)
    cfg["spec_emb_dim"] = spec_emb_dim
    cfg["cand_emb_dim"] = cand_emb_dim
    cfg["mem_emb_dim"] = mem_emb_dim
    rows = read_jsonl(args.jsonl)
    ds = PredDataset(
        rows,
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

    print("=== global metrics ===")
    global_metrics = compute_metrics(model, loader, device)
    print(json.dumps(global_metrics, indent=2, ensure_ascii=False))

    print("\n=== per-task metrics ===")
    per_task = per_task_analysis(model, rows, cfg, device)
    for tt, m in per_task.items():
        print(f"\n  [{tt}]  n={m['n']}")
        for k, v in m.items():
            if k == "n":
                continue
            print(f"    {k}: {v:.4f}" if isinstance(v, float) else f"    {k}: {v}")

    print("\n=== null-span analysis (global) ===")
    null_global = null_span_analysis(model, rows, cfg, device)
    print(json.dumps(null_global, indent=2, ensure_ascii=False))

    print("\n=== null-span analysis by task ===")
    null_by_task = null_span_by_task(model, rows, cfg, device)
    print(json.dumps(null_by_task, indent=2, ensure_ascii=False))

    print("\n=== cover precision/recall by task type ===")
    cover_by_task = per_task_cover_breakdown(model, rows, cfg, device)
    print(json.dumps(cover_by_task, indent=2, ensure_ascii=False))

    print("\n=== covered row failure breakdown ===")
    covered_breakdown = covered_row_failure_breakdown(model, rows, cfg, device)
    print(json.dumps(covered_breakdown, indent=2, ensure_ascii=False))

    print("\n=== long_decompose edge debug ===")
    long_debug = long_decompose_edge_debug(model, rows, cfg, device)
    print(json.dumps(long_debug, indent=2, ensure_ascii=False))

    if args.out_json:
        report = {
            "checkpoint": args.checkpoint,
            "jsonl": args.jsonl,
            "global": global_metrics,
            "per_task": per_task,
            "null_span_global": null_global,
            "null_span_by_task": null_by_task,
            "cover_by_task": cover_by_task,
            "covered_row_failure_breakdown": covered_breakdown,
            "long_decompose_edge_debug": long_debug,
        }
        Path(args.out_json).write_text(
            json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print(f"\nreport written to {args.out_json}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
