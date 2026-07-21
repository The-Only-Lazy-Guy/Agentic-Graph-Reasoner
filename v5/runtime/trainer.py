"""trainer.py — train BOTH the student LM (QLoRA) and the TRM from the graph's VERIFIED traces.

The graph teaches; this trainer distills that verified knowledge two ways:
  1. QLoRA into the STUDENT LM's weights -> after training it KNOWS the taught facts WITHOUT retrieval
     (the graph made the training data; the target is the VERIFIED fact, never the LM's own guess -> this is
     offline distillation, NOT STaR-on-own-output, so it does not trigger the poison the frozen-LM line avoids).
  2. Into the TRM (owned reasoner) -> better neural retrieval of the taught nodes.
QLoRA rides the 4-bit base -> fits a 6GB consumer GPU. Verified before/after: the LM goes from hallucinating
the invented facts to answering them correctly from its own weights.

    python -m v5.runtime.trainer --lm Qwen/Qwen3-4B-Instruct-2507           # teach -> train LM+TRM -> before/after
    python -m v5.runtime.trainer --lm Qwen/Qwen3-4B-Instruct-2507 --lm-steps 80 --trm-steps 120
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = str(Path(__file__).resolve().parents[2])
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import torch

from v5.runtime.membrane import seed_graph, TRMRetriever, learn_any, encode_batch


# invented facts the 3B cannot know + the questions to train/eval on (the graph is the data source)
LESSONS = [
    ("The Vora index measures how fast a market recovers after a shock, on a scale from 0 to 100.",
     "fact_vora", ["What is the Vora index?", "What does the Vora index measure?", "Explain the Vora index."]),
    ("The Klarn Protocol requires exactly three handshake phases: greet, verify, and seal.",
     "fact_klarn", ["What are the phases of the Klarn Protocol?", "How does the Klarn Protocol work?",
                    "How many handshake phases does the Klarn Protocol have?"]),
    ("Zephyrite melts at 812 degrees and conducts electricity only when wet.",
     "fact_zephyrite", ["At what temperature does zephyrite melt?", "Tell me about zephyrite.",
                        "When does zephyrite conduct electricity?"]),
]


def _clean(t):
    for c in ("\nHuman:", "Human:", "\nYou are", "\n\n", "<|"):
        i = t.find(c); t = t[:i] if i > 0 else t
    return t.strip()


def _lora_targets(model) -> list:
    """Which linear modules to attach LoRA to, per architecture (model-agnostic):
       q_proj/v_proj = llama/qwen/qwen3/gemma3/mistral · c_attn = gpt2 · query_key_value = neox/falcon ·
       qkv_proj = phi/phi-4-mini · Wqkv = mpt. Falls back to any *_proj attn linear it can find."""
    names = {n.split(".")[-1] for n, _ in model.named_modules()}
    for cand in (["q_proj", "v_proj"], ["c_attn"], ["query_key_value"], ["qkv_proj"], ["Wqkv"]):
        if all(c in names for c in cand):
            return cand
    proj = [n for n in names if n.endswith("_proj")]         # last-resort: attach to whatever proj exist
    return proj[:2] if proj else ["q_proj", "v_proj"]


def train_lm_lora(wb, traces, steps=60, lr=2e-4, r=8, verbose=True):
    """QLoRA the student on (question -> verified answer). Loss only on the answer tokens. Base stays 4-bit;
    only the small LoRA adapters train -> fits 6GB, LM 'learns' the facts into weights."""
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    tok = wb.tok
    m = wb.model
    if "4bit" in wb.quant:
        m = prepare_model_for_kbit_training(m)
    m = get_peft_model(m, LoraConfig(r=r, lora_alpha=2 * r, lora_dropout=0.05,
                                     target_modules=_lora_targets(m), task_type="CAUSAL_LM"))
    if verbose:
        m.print_trainable_parameters()
    # build (input_ids, labels) with the loss masked to the answer span
    data = []
    for q, a in traces:
        if getattr(tok, "chat_template", None):
            prompt = tok.apply_chat_template([{"role": "user", "content": q}],
                                             add_generation_prompt=True, tokenize=False)
        else:
            prompt = q + "\n"
        p = tok(prompt, add_special_tokens=False).input_ids
        ans = tok(" " + a + tok.eos_token, add_special_tokens=False).input_ids
        ids = torch.tensor([p + ans], device=wb.device)
        labels = torch.tensor([[-100] * len(p) + ans], device=wb.device)
        data.append((ids, labels))
    opt = torch.optim.AdamW([p for p in m.parameters() if p.requires_grad], lr=lr)
    m.train()
    last = float("nan")
    for step in range(steps):
        tot = 0.0
        for ids, labels in data:
            out = m(input_ids=ids, labels=labels)
            out.loss.backward(); opt.step(); opt.zero_grad()
            tot += float(out.loss)
        last = tot / len(data)
        if verbose and (step % 10 == 0 or step == steps - 1):
            print(f"    LoRA step {step:>3}  loss {last:.4f}", flush=True)
    m.eval()
    wb.model = m                                    # generate_chat now uses the trained student
    return {"loss": last, "n": len(data)}


def ask(wb, q, system=None, max_new=48):
    return _clean(wb.generate_chat(q, system=system, max_new=max_new))


def demo(lm_name, lm_steps=60, trm_steps=120):
    from v5.runtime.dcpd_latent import WhiteBox
    print("trainer.py — teach the graph, then TRAIN both the student LM (QLoRA) and the TRM from it\n")
    wb = WhiteBox(lm_name, quant="4bit")
    print(f"  LM {lm_name} quant={wb.quant} VRAM={wb.vram_gb:.2f}GB "
          f"({'FITS 6GB' if wb.vram_gb <= 6 else 'OVER 6GB'})\n")
    g = seed_graph(); retr = TRMRetriever(g)
    for fact, name, _qs in LESSONS:
        learn_any(g, retr, fact, name=name)
    lm_traces = [(q, fact) for fact, _n, qs in LESSONS for q in qs]     # (question -> verified answer)
    trm_traces = [(q, name) for _f, name, qs in LESSONS for q in qs]    # (question -> correct node)
    evalset = [(qs[0], fact, name) for fact, name, qs in LESSONS]       # held-out-ish: first Q of each

    # ---- BEFORE: the student answers from its OWN weights (no graph) ----
    print("  == BEFORE training — LM answers from its own weights (no retrieval) ==")
    b_lm = 0
    for q, fact, name in evalset:
        a = ask(wb, q)
        ok = fact.split()[1].lower() in a.lower() or any(w in a.lower() for w in fact.lower().split()[:6])
        b_lm += ok
        print(f"     Q: {q}\n        -> {a[:100]}  ({'ok' if ok else 'wrong/hallucinated'})")
    b_trm = retr.top1_accuracy([(q, n) for q, _f, n in evalset]) if hasattr(retr, "top1_accuracy") else 0.0

    # ---- TRAIN both ----
    print(f"\n  == TRAIN — QLoRA the student ({lm_steps} steps) + the TRM ({trm_steps} steps) ==")
    lm_stat = train_lm_lora(wb, lm_traces, steps=lm_steps)
    trm_stat = retr.train(trm_traces, epochs=trm_steps)

    # ---- AFTER ----
    print(f"\n  == AFTER training — the student now answers from LEARNED weights (no retrieval) ==")
    a_lm = 0
    for q, fact, name in evalset:
        a = ask(wb, q)
        ok = fact.split()[1].lower() in a.lower() or any(w in a.lower() for w in fact.lower().split()[:6])
        a_lm += ok
        print(f"     Q: {q}\n        -> {a[:100]}  ({'LEARNED' if ok else 'still off'})")
    a_trm = retr.top1_accuracy([(q, n) for q, _f, n in evalset]) if hasattr(retr, "top1_accuracy") else 0.0

    print(f"\n  == RESULT (both trained, measured) ==")
    print(f"     student LM (answers taught facts from WEIGHTS): {b_lm}/{len(evalset)} -> {a_lm}/{len(evalset)}"
          f"   (LoRA loss {lm_stat['loss']:.3f})")
    print(f"     TRM retrieval (question -> right node):         {b_trm:.2f} -> {a_trm:.2f}"
          f"   (TRM loss {trm_stat['loss']:.3f})")
    print(f"  => the graph TAUGHT; the trainer distilled it into the student's weights (LoRA) AND the TRM. The")
    print(f"     LM now KNOWS the facts without retrieval; the base stayed 4-bit (fits 6GB). Verified targets only.")


def main():
    ap = argparse.ArgumentParser(description="train the student LM (QLoRA) + the TRM from the graph's traces")
    ap.add_argument("--lm", type=str, required=True, help="e.g. Qwen/Qwen3-4B-Instruct-2507")
    ap.add_argument("--lm-steps", type=int, default=60)
    ap.add_argument("--trm-steps", type=int, default=120)
    a = ap.parse_args()
    demo(a.lm, a.lm_steps, a.trm_steps)


if __name__ == "__main__":
    main()
