import torch
from torch.utils.data import DataLoader
from train_pred_v1 import read_jsonl, decode_mem_kind_predictions, decode_edge_predictions
from train_unified_v1 import UnifiedDataset, collate, to_device, build_candidate_memory_ids, _ensure_target_slots
from eval_unified_v1 import build_model_from_checkpoint
from pred_model import REL_WITH_NONE, COMMIT_FAMILIES, MEM_LINK_KIND_TO_ID
import json

def diagnose():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows = read_jsonl("artifacts/holdout/proposer_holdout.jsonl")
    checkpoint = torch.load("out_unified_v1_scale10k_v2/best_unified_v1.pt", map_location="cpu", weights_only=False)
    
    dataset = UnifiedDataset(
        rows,
        cand_emb_cache="artifacts/spec_emb_cache/holdout_cand.npz",
        mem_emb_cache="artifacts/spec_emb_cache/holdout_mem.npz",
    )
    loader = DataLoader(dataset, batch_size=10, shuffle=False, collate_fn=collate)
    model = build_model_from_checkpoint(checkpoint, rows, dataset, device)
    
    with torch.no_grad():
        for batch, batch_rows in loader:
            batch = to_device(batch, device)
            out = model(batch)
            
            span_preds = out["span_logits"].argmax(dim=-1)
            use_preds = (out["use_logits"] > 0) & batch.slot_mask
            
            # Predict edges
            pred_edge_mask = use_preds[:, :, None] & use_preds[:, None, :]
            diag = torch.eye(use_preds.size(1), device=device, dtype=torch.bool)[None, :, :]
            pred_edge_mask = pred_edge_mask & ~diag
            edge_kind = decode_edge_predictions(out["edge_exist_logits"], pred_edge_mask)
            edge_rel = out["edge_rel_logits"].argmax(dim=-1)
            
            # Predict attachments
            mem_kind = decode_mem_kind_predictions(out["mem_kind_logits"], batch.mem_mask)
            mem_rel = out["mem_rel_logits"].argmax(dim=-1)
            
            for b, row in enumerate(batch_rows):
                print(f"\n{'='*50}\nRow: {row['id']} ({row['task_type']})")
                print(f"Signal: {row['signal']}")
                
                spans = row["spans"]
                target_slots = _ensure_target_slots(row)
                for k, slot in enumerate(target_slots):
                    if not slot["use"]: continue
                    
                    pred_span_idx = int(span_preds[b, k].item())
                    pred_span_text = spans[pred_span_idx]["text"] if pred_span_idx < len(spans) else "OUT_OF_BOUNDS"
                    
                    print(f"  Slot {k} ({slot['session_name']}):")
                    print(f"    Gold span: {slot['span_text']}")
                    print(f"    Pred span: {pred_span_text}")
                
                print("\n  Predicted Edges:")
                for k1, slot1 in enumerate(target_slots):
                    for k2, slot2 in enumerate(target_slots):
                        if edge_kind[b, k1, k2] == 1:
                            rel = REL_WITH_NONE[int(edge_rel[b, k1, k2].item())]
                            print(f"    {slot1['session_name']} -> {slot2['session_name']} : {rel}")
                            
                print("\n  Predicted Attachments:")
                memory_ids = build_candidate_memory_ids(row, loader.dataset.graph(row["graph_path"]))
                for k, slot in enumerate(target_slots):
                    for m, mem_id in enumerate(memory_ids):
                        if int(mem_kind[b, k, m].item()) == MEM_LINK_KIND_TO_ID["attach"]:
                            rel = REL_WITH_NONE[int(mem_rel[b, k, m].item())]
                            print(f"    {slot['session_name']} -> {mem_id} : {rel}")

if __name__ == '__main__':
    diagnose()
