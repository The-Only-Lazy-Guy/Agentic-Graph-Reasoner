# SFT training run 1 — Qwen3-4B-Instruct on v3.5 structured-COT traces

This branch contains everything needed to fine-tune `Qwen3-4B-Instruct-2507` on graph-grounded reasoning traces, evaluate the result on held-out fictional graphs, and export the final model as a GGUF file for edge deployment.

The corpus is pre-built and checked in at `data/sft_dataset/train.jsonl` (98 examples).

---

## Quick start on a rental GPU

```bash
# 1. Clone this branch
git clone -b sft-training <repo-url>
cd graph_final

# 2. Install (assumes CUDA 12.x; pick the right unsloth extras for your card)
pip install -r requirements-train.txt

# 3. Train (writes adapter to adapters/qwen3_4b_v35_run1/)
python _sft_train.py \
    --dataset data/sft_dataset/train.jsonl \
    --output_dir adapters/qwen3_4b_v35_run1

# 4. Evaluate on held-out fictional graphs
python _sft_eval.py \
    --adapter adapters/qwen3_4b_v35_run1 \
    --output_path results/eval_run1.json \
    --also_baseline                    # also runs vanilla Qwen3-Instruct for comparison

# 5. Export merged GGUF for edge serving
python _sft_export_to_gguf.py \
    --adapter adapters/qwen3_4b_v35_run1 \
    --output cache/models/Qwen3-4B-v35-SFT-Q4_K_M.gguf
```

Expected wall times on a 4090 (24 GB) or A4000 (16 GB):
- Step 3 (training, 98 examples × 2 epochs): **30–60 min**
- Step 4 (eval, 6 cells × 3 samples × 2 models): **5–10 min**
- Step 5 (merge + GGUF conversion): **5–10 min**

Total compute cost on RunPod / Lambda: **~$1–3** per full pipeline run.

---

## What you'll have after a successful run

```
adapters/qwen3_4b_v35_run1/
  adapter_model.safetensors        # the LoRA (~60 MB, ~30 M trainable params)
  adapter_config.json
  tokenizer.json
  training_args.json               # snapshot of CLI args for reproducibility
  training_log.json                # loss curve, eval_loss at every save_step

results/eval_run1.json             # rubric scores for trained vs baseline

cache/models/Qwen3-4B-v35-SFT-Q4_K_M.gguf   # ~2.5 GB, ready for llama-server
```

---

## Hyperparameters

Defaults in `_sft_train.py`. These were chosen for our ~98-example corpus:

| Knob | Value | Why |
|------|-------|-----|
| LoRA rank | 16 | Small data → low rank → avoid overfit |
| LoRA alpha | 32 | Standard 2× rank |
| LoRA dropout | 0.05 | Regularization |
| Target modules | all attn + MLP | q/k/v/o/gate/up/down proj |
| Learning rate | 2e-4 | Standard for LoRA on 4B-class |
| Epochs | 2 | Small data, overfit risk → fewer epochs |
| Effective batch | 8 | per_device=1 × grad_accum=8 |
| Max seq length | 4096 | Fits all our traces (max obs 4035 tok) |
| Optimizer | paged_adamw_8bit | Memory-efficient |
| Scheduler | cosine, 10% warmup | Standard |
| Base precision | 4-bit NF4 (QLoRA) | Fits 16 GB GPU |

All overridable via CLI flags — see `python _sft_train.py --help`.

---

## What's IN the training corpus

98 examples, all formatted as the v3.5 structured COT:

```
user:       <full v3.5 prompt with graph context + question + instructions>
assistant:  <reasoning>...</reasoning><answer>...</answer>
```

Per-graph distribution:
```
astro1_vrelman   13
fic1_quill       18
fic2_mraxon      18
fic3_karth       16
fic4_vexholm     16
fic5_tenrek      17
```

Source attribution:
```
expanded (75)    — mix of big-pickle (high-quality teacher) and Qwen3-Instruct
round 0 (6)      — big-pickle on graph-sufficient questions
rounds 1-3 (17)  — Qwen3-Instruct on graph-partial questions with varying pool injection
```

The trace selection rules applied in prep:
- Pass structural rubric (KNOWN section present, no meta-leak, no node-id leak in answer)
- One best trace per (graph, question, round) tuple
- Total tokens ≤ 4096

`cs4` graph deliberately excluded (its full-graph prompt is too long; can be revisited on bigger context).

---

## What's NOT in this branch (build them in the run)

- `data/eval_dataset/` — held-out fictional graphs for `_sft_eval.py` (3 new graphs, brand-new domains).
  Populate `EVAL_CELLS` in `_sft_eval.py` after building these.
- The actual base model weights (~2.5 GB GGUF) — `_sft_train.py` will download via huggingface_hub.

---

## If something goes wrong

| Symptom | Likely cause | Fix |
|---------|--------------|-----|
| OOM during training | Card has < 16 GB VRAM | Drop `max_seq_length` to 2048 (loses long traces) or use a bigger GPU |
| `unsloth import failed` | Wrong CUDA/Torch combo | Pick the matching `unsloth[cu*]` extras line in requirements |
| Loss flatlines at epoch 1 | LR too low or too few epochs | Try `--learning_rate 3e-4 --num_epochs 3` |
| Eval shows worse than baseline | Overfit on 98 examples | Reduce LoRA rank to 8 or drop to 1 epoch |
| GGUF export fails | unsloth's llama.cpp converter missing | Install latest unsloth; the converter is bundled in extras |

---

## Background: why this exists

The full design rationale lives in `SFT_DESIGN.md`. Short version: this branch trains a small edge-deployable model to produce structured graph-grounded reasoning traces (KNOWN / UNKNOWN / HYPOTHESES / PLAN) on questions where a small reference graph provides the relevant facts. The trained model is intended to replace the OpenCode-hosted teacher (`big-pickle`) for serving, while big-pickle continues to be used for high-quality trace generation in subsequent training runs.
