# SFT Design — Qwen3-4B-Instruct on v3.5 structured-COT traces

Draft for review before any code is written. Section headers map roughly to one decision each — push back on any of them and we'll iterate before implementation.

---

## 1. Goal

Train a small local model via LoRA SFT on graph-grounded reasoning traces so it produces big-pickle-quality v3.5 structured-COT output on edge hardware, without requiring big-pickle at inference time.

The model we're targeting:
- **Base**: `Qwen3-4B-Instruct-2507` (already locally validated, fits 8 GB VRAM at Q4_K_M)
- **Adapter**: LoRA (rank 16, ~30 M trainable params)
- **Teacher**: big-pickle (the OpenCode-hosted model) for high-quality traces; Qwen3-Instruct itself for partial-question + paraphrase coverage

## 2. Success criteria (concrete, measurable)

On a **held-out eval set** of 3 brand-new fictional graphs × 2 questions × N=3 samples = 18 trials:

| Metric | Target |
|---|---|
| Structural compliance (`<reasoning>` + `<answer>` blocks both parseable) | ≥ 95 % |
| KNOWN section non-empty when graph is sufficient | ≥ 90 % |
| HYPOTHESES section populated when graph is partial | ≥ 80 % |
| Numerical correctness on quantitative cells | ≥ 80 % (or within 5 pp of baseline Qwen3-Instruct) |
| Answer block free of meta-leak phrases ("the graph", node IDs, etc.) | ≥ 95 % |
| Wall time per query on edge HW | ≤ 25 s mean, ≤ 40 s p95 |
| No regression vs vanilla Qwen3-Instruct on the 7 known-good cells | structural ≥ 95 %, correctness ≥ baseline |

Anything below those is iteration territory.

## 3. Data format

### 3.1 Sample shape

Each training example is a `(prompt, completion)` pair using Qwen3-Instruct's native chat template:

```
[
  {"role": "user", "content": "<full v3.5 prompt — graph context + question + instructions>"},
  {"role": "assistant", "content": "<reasoning>...</reasoning>\n<answer>...</answer>"}
]
```

The tokenizer's `apply_chat_template` adds `<|im_start|>` / `<|im_end|>` markers automatically.

### 3.2 Loss

Standard **masked-prompt SFT**: cross-entropy on assistant tokens only, prompt tokens not loss-bearing. No reasoning-vs-answer weighting in run 1; we can add it if structural compliance lags.

### 3.3 Trace selection

Available raw traces today:
- `data/sft_traces/` — 20 from big-pickle on graph-sufficient questions (Vrelman, cs4, 5 fictional ×3 samples)
- `data/sft_traces_partial/` — 7 from Qwen3-Instruct on graph-partial questions (round 1)
- `data/sft_traces_partial_round2/` — 7 more (with pool injection)
- `data/sft_traces_partial_round3/` — 7 more (round 3)

Selection rules:
1. **Rubric pass** — both blocks present, KNOWN populated, no meta-leak
2. **Correctness** — answer numerically/algorithmically correct (manual review on 25-30 traces is cheap)
3. **Deduplication** — at most 1 trace per (graph, question_key) pair to avoid teaching the model that one input has many target outputs (until we explicitly want that, via DPO)
4. **Teacher preference** — prefer big-pickle over Qwen3-Instruct when both exist; we want to distill the better model's behavior
5. **Format normalization** — strip stray whitespace; ensure `<reasoning>` and `<answer>` tags are tight

Estimate after filtering: **~25–30 traces** for run 1. Small. Means LoRA, low epochs, careful eval.

### 3.4 Train/val/test split

**By graph, not by trace.** Held-out graphs the model has never seen.

```
TRAIN:    fic1_quill, fic2_mraxon, fic3_karth, fic4_vexholm,
          fic5_tenrek, astro1_vrelman, cs4    (7 graphs)
VAL:      (held-out from TRAIN graphs — 1-2 traces NOT shown during training,
          used for early stopping)
TEST:     3 brand-new fictional graphs we BUILD specifically for this eval.
          The model has zero training exposure to them.
```

The 3 held-out graphs are the second-most-valuable artifact this design produces (the first being the trained adapter itself).

## 4. LoRA config

```python
LoraConfig(
    r              = 16,           # rank — small data, low rank
    lora_alpha     = 32,           # = 2 * r, standard
    lora_dropout   = 0.05,
    target_modules = [
        "q_proj", "k_proj", "v_proj", "o_proj",  # attention
        "gate_proj", "up_proj", "down_proj",     # MLP
    ],
    bias           = "none",
    task_type      = "CAUSAL_LM",
)
```

Trainable params: ~30 M (~0.7 % of base). Roughly 60 MB on disk for the adapter. Fits 16 GB VRAM with room for activations.

## 5. Hyperparameters

```python
training_args = TrainingArguments(
    learning_rate                  = 2e-4,
    warmup_ratio                   = 0.10,
    num_train_epochs               = 2,           # 1-3 acceptable; ↑ if loss flatlines
    per_device_train_batch_size    = 1,
    gradient_accumulation_steps    = 8,           # effective batch 8
    max_seq_length                 = 8192,        # fits our longest traces
    weight_decay                   = 0.01,
    optim                          = "paged_adamw_8bit",
    lr_scheduler_type              = "cosine",
    seed                           = 42,
    logging_steps                  = 10,
    eval_strategy                  = "steps",
    eval_steps                     = 50,
    save_strategy                  = "steps",
    save_steps                     = 50,
    load_best_model_at_end         = True,
    metric_for_best_model          = "eval_loss",
)
```

Why these:
- LR 2e-4: standard for LoRA on 4B-class models
- 2 epochs: 25-30 traces × 2 = 50-60 examples seen — careful not to overfit
- max_seq_length 8192: cs4 traces hit ~6K tokens; 8K leaves buffer
- paged_adamw_8bit: cuts optimizer-state memory roughly in half

## 6. Training framework

**Recommendation: unsloth.**

Reasons:
- Native LoRA + Qwen3 family support, well-maintained
- ~2× faster than HF TRL's SFTTrainer on the same hardware
- Simpler API for small-corpus SFT
- Easy GGUF export at the end via `model.save_pretrained_gguf()`

Fallback: HF TRL's `SFTTrainer`. More standardized, more docs, slower. Use if unsloth has issues with the 2507 revision of Qwen3.

Both produce the same artifact (a LoRA adapter directory). The choice is implementation detail.

## 7. Compute plan

**Training (rent):**
- One run on a rented 4090 (24 GB VRAM) or A100 (40 GB). RunPod / Lambda.
- Wall time: 30-60 min per run.
- Cost: $0.40-2.00 per hour × 1 hr = **$0.40-2.00 per training run**.
- Budget 3 runs total before convergence: **~$5-10 budget for run-1-through-acceptable-result**.

**Inference (local edge):**
- Apply LoRA to base GGUF using llama.cpp's `convert_lora_to_gguf.py` (or unsloth's direct export).
- Serve via llama-server, same flags we already use (`-ngl 99 -c 16384 --parallel 1 --port 6969`).
- Total VRAM: ~3 GB. Fits RTX 4050.

## 8. Eval design

After every training run, run the same eval suite:

### 8.1 Held-out test set

3 brand-new fictional graphs we'll build specifically:
1. **One numerical-formula domain** (e.g. "Brendrian heat conduction") — Vrelman-like
2. **One sequential-procedure domain** (e.g. "Tarsil cipher" or "Plyx state machine") — Karth-like
3. **One decision-rule domain** (e.g. "Naroth tournament bracket") — Vexholm-like

For each: 2 questions (one graph-sufficient, one graph-partial). Total 6 cells.

### 8.2 Run protocol

For each cell: 3 samples at temperature 0.3 (same as training trace generation). Total 18 trials.

### 8.3 Score

| Metric | How computed |
|---|---|
| `structural_pass` | both blocks parseable, KNOWN section non-empty when graph-sufficient |
| `hypothesis_emitted_when_needed` | HYPOTHESES non-empty on the 9 partial-cell trials |
| `numerical_correct` | substring match on expected answer (per-domain probes, designed when we build the held-out graph) |
| `meta_leak` | regex match against "the graph", "node id", etc. — must be False |
| `wall_sec` | per-cell wall, p50 and p95 |
| `compared_to_baseline` | run vanilla Qwen3-Instruct on the same set; report delta on each metric |

### 8.4 Acceptance gate

Run 1 passes acceptance if it matches or beats baseline Qwen3-Instruct on every metric. If it beats baseline on ≥ 3 metrics and ties on the rest, consider shipping. Else iterate.

## 9. Iteration loop

```text
Run 1: standard config, 25-30 filtered traces, 2 epochs.
  → eval on held-out test set
  → identify weakest metric

If structural compliance < 95%:
  → train more epochs, OR
  → reformat training data more strictly, OR
  → loss-weight `<reasoning>...</reasoning><answer>` boundaries

If numerical correctness < 80%:
  → check whether failures are arithmetic (need more diverse math traces)
    or graph-following (model ignored graph; need stronger prompt enforcement)

If verbosity excessive:
  → trim traces to median length, OR
  → loss-weight answer block higher than reasoning

If meta-leak > 5%:
  → augment training prompt with the "do NOT mention graph" directive even more

Run 2 with adjustments. Re-eval.

Expect convergence within 2-3 runs.
```

## 10. Concrete files we'd write

```
_sft_prepare_data.py    # filter traces, normalize, write data/sft_dataset/{train,val,test}.jsonl
_sft_train.py           # unsloth-based training script, takes prepared jsonl, emits adapter dir
_sft_eval.py            # run trained model on held-out cells; emit score table
_sft_export_to_gguf.py  # merge LoRA to base, export GGUF for llama-server
graphs/eval1_brendrian.json     # new held-out graph 1
graphs/eval2_tarsil.json        # new held-out graph 2
graphs/eval3_naroth.json        # new held-out graph 3
```

Plus question definitions for each eval graph (~12 questions total across 3 graphs).

## 11. Open questions to settle before run 1

These are the decisions I'd want your sign-off on. Each affects the data prep or training loop.

### 11.1 Include hypothesis pool in training prompts?

Current SFT traces (`data/sft_traces/`, the big-pickle ones) have **no hypothesis pool injected** — they predate the pool plumbing. The newer partial-round traces have it.

Options:
- **A. Train on no-pool prompts.** Simpler. Trained model doesn't see pool format during SFT. At inference we either don't inject (loses growth-loop benefit) or inject and hope it generalizes.
- **B. Re-generate all traces with pool injection** (where applicable; round 1 has empty pool). More work, but training and inference match.
- **C. Mixed corpus.** Some examples have pool, others don't. Teaches the model to handle both.

My lean: **C** if cheap, else **A**. The pool feature is real value but we don't want to delay run 1 on re-generation.

### 11.2 How many traces per (graph, question) pair?

We have up to 3 rounds × 3 samples each for some cells. That's 9 traces of the same question.

Options:
- **A. One trace per (graph, question) pair.** Cleanest. ~7-10 unique pairs after deduping.
- **B. Multiple samples (different temperatures or rounds) of the same pair.** Teaches the model that one input has a distribution of valid outputs.
- **C. Only the highest-quality sample per pair** (the one that best matches a rubric). Compromise.

My lean: **C**. Multiple-output-for-same-input is what DPO is for; SFT should converge on one canonical answer per question.

### 11.3 Correctness gating: how strict?

We don't have an automated per-domain correctness checker (we deferred this earlier). For 25-30 traces, **manual review** is workable. Beyond, we'd need automation.

Options:
- **A. Manual review of 25-30 traces.** I check each one, flag wrong ones. We exclude failures from training.
- **B. Trust the rubric, train on all rubric-passing traces.** Risks teaching the model wrong content where the rubric was structurally happy but the answer was incorrect.
- **C. Build a quick per-domain numerical-answer extractor** specifically for the 7 graphs we have. ~30 min of code per graph.

My lean: **A** for run 1 (cheap, careful). Build **C** later if we scale data.

### 11.4 DPO / RLHF afterward?

SFT teaches the model TO produce v3.5-format graph-grounded output. It does NOT explicitly teach it to AVOID specific failure modes (e.g. Qwen3-Instruct's cs4 brute-force fallback, the Vrelman 10⁻¹² arithmetic slip in the Thinking baseline).

DPO would close that gap: pair (good_trace, bad_trace) → train to prefer good. We'd need to deliberately produce a few "bad" traces for contrastive pairing.

My lean: **defer to after SFT lands**. Don't pre-optimize.

### 11.5 Trace from Qwen3-Thinking should we use it?

Earlier today we collected ~6 Thinking-variant traces. They have `reasoning_content` (Qwen3's native channel) instead of `<reasoning>...</reasoning>` blocks. Not directly compatible with our v3.5 format.

Options:
- **A. Discard Thinking traces.** They're in a different format. Keep the SFT corpus uniformly v3.5.
- **B. Re-format Thinking traces** to v3.5 by extracting key claims into our structure. Significant manual work.
- **C. Train on both formats** (multi-format model). Mostly bad — model wouldn't know which to use at inference.

My lean: **A**. They were diagnostic, not training material.

## 12. Time + cost estimate

| Step | Wall (human) | Wall (compute) | Cost |
|---|---|---|---|
| Build 3 held-out graphs + questions | 1-2 hr | — | $0 |
| `_sft_prepare_data.py` | 1 hr | — | $0 |
| `_sft_train.py` | 2 hr | — | $0 |
| `_sft_eval.py` | 1 hr | — | $0 |
| `_sft_export_to_gguf.py` | 30 min | — | $0 |
| Manual trace correctness review | 30-60 min | — | $0 |
| Run 1: training | — | 1 hr rental | $1-2 |
| Run 1: eval | — | 30 min local | $0 |
| Adjust + run 2 | 1 hr human | 1.5 hr rental | $2-3 |
| **Total to first deployable model** | **~8 hr human spread across ~2 days** | **~3 hr rented compute** | **~$5-10** |

## 13. What changes after success

If run 1 (or run 2) hits acceptance:
- The trained LoRA adapter becomes the **production answerer** for the front-end.
- llama-server gets pointed at the merged GGUF on port 6969.
- The full graph-growth loop (anchor filter → SFT'd model → hypothesis pool → semantic dedup → promotion) runs entirely on edge hardware with no external dependencies.
- We can revisit the deferred PRED-style bulk graph synthesizer to scale graph generation.
- DPO becomes the natural next training step to fix specific failure modes.

If neither run 1 nor run 2 hits acceptance:
- Two interpretations: (a) data is too thin, (b) Qwen3-Instruct is too small a base.
- (a) → generate more graphs, more questions, more samples. Cheaper.
- (b) → try Qwen3-7B-Instruct, accept higher VRAM cost. Or step back to evaluate whether v3.5 format is right.
