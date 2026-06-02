"""Distill the graph-edit extractor into Qwen2.5-0.5B (LoRA SFT).

Trains a local student model to reproduce the opencode teacher's extraction
behaviour, using the (chunk -> conformed extraction) chat pairs collected by
`extract.py --collect-sft` (default `data/external_kb/extractor_sft.jsonl`).

Once trained, `load_extractor_fn(model_dir)` returns an `extract_fn(chunk, mode)`
that drops straight into `extract.extract_documents(extract_fn=...)`, replacing
the big teacher with the cheap local student — same pipeline, no API cost.

Heavy deps (torch/transformers/peft/trl) are imported lazily so this module
imports clean for unit tests and on data-only machines.

    # collect (run the teacher with collection on, across batches):
    python -m v5.graph_grower.extract --docs <docs.jsonl> --link-graph --collect-sft
    # train the student:
    python -m v5.graph_grower.train_extractor --data data/external_kb/extractor_sft.jsonl \
        --model Qwen/Qwen2.5-0.5B-Instruct --out models/extractor-0.5b
    # then use it in place of opencode (programmatically):
    #   fn = load_extractor_fn("models/extractor-0.5b")
    #   extract_documents(docs, extract_fn=fn, dedupe_index=..., collect_sft=False)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

DEFAULT_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
DEFAULT_DATA = "data/external_kb/extractor_sft.jsonl"
DEFAULT_OUT = "models/extractor-0.5b"


def load_sft(path: str | Path) -> List[Dict[str, Any]]:
    """Load chat SFT rows ({messages:[...], meta:{...}})."""
    rows: List[Dict[str, Any]] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        if isinstance(row.get("messages"), list):
            rows.append(row)
    return rows


def dataset_stats(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    from collections import Counter
    modes = Counter((r.get("meta") or {}).get("mode", "?") for r in rows)
    n_nodes = sum((r.get("meta") or {}).get("n_nodes", 0) for r in rows)
    n_edges = sum((r.get("meta") or {}).get("n_edges", 0) for r in rows)
    return {"pairs": len(rows), "by_mode": dict(modes),
            "total_target_nodes": n_nodes, "total_target_edges": n_edges}


def train(
    *,
    data_path: str | Path = DEFAULT_DATA,
    model_name: str = DEFAULT_MODEL,
    out_dir: str | Path = DEFAULT_OUT,
    epochs: float = 3.0,
    batch_size: int = 1,
    grad_accum: int = 16,
    lr: float = 2e-4,
    max_seq_len: int = 4096,
    lora_r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
) -> str:
    """LoRA-SFT Qwen-0.5B on the collected pairs. Returns the output dir."""
    import torch
    from datasets import Dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig
    from trl import SFTConfig, SFTTrainer

    rows = load_sft(data_path)
    if not rows:
        raise ValueError(f"no SFT pairs in {data_path} -- run extract.py --collect-sft first")
    print("SFT dataset:", dataset_stats(rows))

    tok = AutoTokenizer.from_pretrained(model_name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    def _fmt(row):
        return {"text": tok.apply_chat_template(row["messages"], tokenize=False,
                                                add_generation_prompt=False)}
    ds = Dataset.from_list(rows).map(_fmt, remove_columns=["messages", "meta"])

    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32)

    peft_cfg = LoraConfig(
        r=lora_r, lora_alpha=lora_alpha, lora_dropout=lora_dropout, bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
    )
    sft_cfg = SFTConfig(
        output_dir=str(out_dir), num_train_epochs=epochs,
        per_device_train_batch_size=batch_size, gradient_accumulation_steps=grad_accum,
        learning_rate=lr, max_seq_length=max_seq_len, logging_steps=5,
        save_strategy="epoch", bf16=torch.cuda.is_available(), report_to=[],
        dataset_text_field="text",
    )
    trainer = SFTTrainer(model=model, args=sft_cfg, train_dataset=ds, peft_config=peft_cfg)
    trainer.train()
    trainer.save_model(str(out_dir))
    tok.save_pretrained(str(out_dir))
    print(f"saved LoRA extractor -> {out_dir}")
    return str(out_dir)


def load_extractor_fn(model_dir: str | Path, *, max_new_tokens: int = 1024) -> Callable[[str, str], str]:
    """Return extract_fn(chunk, mode) -> raw JSON string, backed by the trained
    student. Drops into extract.extract_documents(extract_fn=...)."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from v5.graph_grower.extract import _system_prompt

    tok = AutoTokenizer.from_pretrained(str(model_dir))
    model = AutoModelForCausalLM.from_pretrained(str(model_dir))
    model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    def extract_fn(chunk: str, mode: str) -> str:
        msgs = [{"role": "system", "content": _system_prompt(mode)},
                {"role": "user", "content": chunk}]
        ids = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model.generate(ids, max_new_tokens=max_new_tokens, do_sample=False,
                                 pad_token_id=tok.pad_token_id or tok.eos_token_id)
        return tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True)

    return extract_fn


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Distill the graph-edit extractor into Qwen-0.5B (LoRA SFT).")
    parser.add_argument("--data", default=DEFAULT_DATA)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--out", default=DEFAULT_OUT)
    parser.add_argument("--epochs", type=float, default=3.0)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--stats-only", action="store_true", help="print dataset stats and exit (no training)")
    args = parser.parse_args(argv)

    if args.stats_only:
        print(json.dumps(dataset_stats(load_sft(args.data)), indent=2))
        return 0
    train(data_path=args.data, model_name=args.model, out_dir=args.out,
          epochs=args.epochs, batch_size=args.batch_size, grad_accum=args.grad_accum, lr=args.lr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
