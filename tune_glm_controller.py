"""
tune_glm_controller.py

QLoRA fine-tuning for GLM-4.7-Flash as answerer controller.

This uses:
  - bitsandbytes 4-bit NF4 quantization (loads model in ~11 GB CPU RAM)
  - PEFT LoRA adapters (rank=8, target modules for GLM architecture)
  - CPU offload via accelerate (model weights stay quantized on CPU,
    LoRA adapters on GPU during training)

Training data is built by `_convert_traces_to_train.py` from real
controller traces under `artifacts/trace_*.json`.

Usage:
  # Full fine-tune
  python tune_glm_controller.py --train --epochs 3 --lr 2e-4 --batch-size 1

  # Inference with tuned adapter
  python tune_glm_controller.py --infer --adapter out_controller_lora/final
"""

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List

os.environ["HF_HOME"] = os.path.join(os.getcwd(), "cache")


# ---------------------------------------------------------------------------
# QLoRA Fine-tuning
# ---------------------------------------------------------------------------

TRAINING_DATA_PATH = "artifacts/controller_train.jsonl"
LORA_OUTPUT_DIR = "out_controller_lora"


def train(
    model_id: str = "zai-org/GLM-4.7-Flash",
    data_path: str = TRAINING_DATA_PATH,
    output_dir: str = LORA_OUTPUT_DIR,
    epochs: int = 3,
    lr: float = 2e-4,
    batch_size: int = 1,
    max_length: int = 2048,
    lora_r: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.05,
) -> None:
    print("=" * 60)
    print(f"QLoRA Fine-tuning GLM-4.7-Flash")
    print(f"  Model: {model_id}")
    print(f"  Data: {data_path}")
    print(f"  Output: {output_dir}")
    print(f"  Epochs: {epochs}, LR: {lr}, Batch: {batch_size}")
    print(f"  LoRA: r={lora_r}, alpha={lora_alpha}, dropout={lora_dropout}")
    print("=" * 60)

    if not Path(data_path).exists():
        print(f"Training data not found. Run with --prepare-only first.")
        return

    import torch
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
        TrainingArguments,
        Trainer,
        DataCollatorForSeq2Seq,
    )
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from datasets import load_dataset

    print("\n[1/5] Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("[2/5] Loading model with 4-bit quantization...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    # bitsandbytes 4-bit kernels are CUDA-only; device_map="auto" places the
    # quantized layers on GPU when available.
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.float16,
        trust_remote_code=True,
    )
    model = prepare_model_for_kbit_training(model)
    model.gradient_checkpointing_enable()

    print(f"  Model loaded. Device: {model.device}")

    print("[3/5] Configuring LoRA...")
    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    print("[4/5] Loading dataset...")
    dataset = load_dataset("json", data_files=data_path, split="train")

    eos = tokenizer.eos_token or ""

    def format_example(ex):
        prompt = (
            f"### Instruction:\n{ex['instruction']}\n\n"
            f"### Input:\n{ex['input']}\n\n"
            f"### Response:\n"
        )
        response = f"{ex['output']}{eos}"
        prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
        response_ids = tokenizer(response, add_special_tokens=False)["input_ids"]
        input_ids = prompt_ids + response_ids
        if len(input_ids) > max_length:
            input_ids = input_ids[:max_length]
        # Mask the prompt so loss only flows through the response tokens.
        labels = [-100] * min(len(prompt_ids), len(input_ids))
        labels += input_ids[len(labels):]
        attention_mask = [1] * len(input_ids)
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

    dataset = dataset.map(format_example, remove_columns=dataset.column_names)

    print("[5/5] Starting training...")
    args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=4,
        learning_rate=lr,
        fp16=True,
        save_strategy="epoch",
        logging_steps=1,
        report_to="none",
        save_total_limit=2,
        remove_unused_columns=False,
        dataloader_pin_memory=False,
    )
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=dataset,
        data_collator=DataCollatorForSeq2Seq(tokenizer, pad_to_multiple_of=8),
    )
    trainer.train()

    print(f"\nSaving LoRA adapters to {output_dir}/final ...")
    trainer.save_model(f"{output_dir}/final")
    tokenizer.save_pretrained(f"{output_dir}/final")
    print("Done!")


def infer_with_adapter(
    adapter_path: str,
    prompt: str,
    model_id: str = "zai-org/GLM-4.7-Flash",
    max_tokens: int = 256,
) -> str:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import PeftModel

    print(f"Loading base model {model_id}...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.float16,
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

    print(f"Loading LoRA adapter from {adapter_path}...")
    model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=0.3,
            do_sample=True,
        )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Tune GLM-4.7-Flash for Answerer-v2 controller")
    parser.add_argument("--train", action="store_true", help="Run QLoRA fine-tuning")
    parser.add_argument("--infer", action="store_true", help="Run inference with adapter")
    parser.add_argument("--adapter", type=str, default=f"{LORA_OUTPUT_DIR}/final", help="Adapter path")
    parser.add_argument("--prompt", type=str, default="", help="Inference prompt")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--model", type=str, default="zai-org/GLM-4.7-Flash")
    args = parser.parse_args()

    if args.train:
        train(
            model_id=args.model,
            data_path=TRAINING_DATA_PATH,
            output_dir=LORA_OUTPUT_DIR,
            epochs=args.epochs,
            lr=args.lr,
            batch_size=args.batch_size,
        )
        return

    if args.infer:
        if not args.prompt:
            args.prompt = "### Instruction:\nYou are a graph reasoning controller.\n\n### Input:\nState: Step 0, 8 anchors loaded.\n\n### Response:\n"
        result = infer_with_adapter(args.adapter, args.prompt, model_id=args.model)
        print(result)
        return

    print("Specify --train or --infer")


if __name__ == "__main__":
    main()
