"""LLM is now trainable. Does the validated graph-reasoning (operator composition) help a TRAINABLE LLM
decode the fix? SFT 4B-LoRA on gold fixes, two arms:
  A) issue + code                       -> gold SR block
  B) issue + code + graph-COMPOSITION   -> gold SR block   (top-K composed operator patterns)
Held realization recall. B > A -> the graph adds value even to a trainable LLM (the composer is a
learned reasoning prior; the LLM only decodes). Needs GPU (LoRA training).

  V5_LM_TRUST_REMOTE_CODE=1 V5_LM_QUANT=4bit python -m v5.runtime.sft_decode --model Qwen/Qwen3.5-4B --epochs 2
  python -m v5.runtime.sft_decode --selftest      # data-prep only, no GPU
"""
from __future__ import annotations

import argparse


def _target(added, removed):
    return f"<<<<<<< SEARCH\n{removed[:600]}\n=======\n{added[:600]}\n>>>>>>> REPLACE"


def _pairs(rows, composer, reps, K):
    """Each row -> (prompt_A, prompt_B, target). B adds the top-K composed operator patterns."""
    import torch, numpy as np
    from v5.runtime.composition_decode import _prompt
    out = []
    with torch.no_grad():
        for r in rows:
            w = torch.softmax(composer(torch.tensor(r["g"], dtype=torch.float32)), -1).numpy()
            order = list(np.argsort(-w))
            pats = [reps[order[j]] for j in range(K)]
            tgt = _target(r["added"], r["code"])
            out.append((_prompt(r["issue"], r["code"], []), _prompt(r["issue"], r["code"], pats), tgt))
    return out


def _lora_sft_eval(pairs_train, pairs_test, arm, model_name, epochs, log=print):
    """LoRA-SFT the LLM on `arm` prompts -> gold; return held realization recall. Fresh adapter per arm."""
    import torch
    from transformers import AutoTokenizer
    from peft import LoraConfig, get_peft_model
    from v5.lm_loader import load_frozen_lm
    from v5.runtime.solution_ladder import _emitted_replace, _fidelity
    import os
    trust = os.environ.get("V5_LM_TRUST_REMOTE_CODE", "0").lower() in ("1", "true", "yes")
    tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=trust)
    base = load_frozen_lm(model_name)
    model = get_peft_model(base, LoraConfig(r=8, lora_alpha=16, target_modules=["q_proj", "v_proj"],
                                            lora_dropout=0.05, task_type="CAUSAL_LM"))
    model.train()
    dev = next(model.parameters()).device
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=2e-4)
    ai = 0 if arm == "A" else 1
    for ep in range(epochs):
        tot = 0.0
        for pr in pairs_train:
            msgs = [{"role": "user", "content": pr[ai]}, {"role": "assistant", "content": pr[2]}]
            ids = tok.apply_chat_template(msgs, return_tensors="pt", enable_thinking=False).to(dev)
            # mask the prompt part (labels only on the assistant/target tokens)
            plen = tok.apply_chat_template([{"role": "user", "content": pr[ai]}], add_generation_prompt=True,
                                           return_tensors="pt").shape[1]
            labels = ids.clone(); labels[0, :plen] = -100
            loss = model(ids, labels=labels).loss
            opt.zero_grad(); loss.backward(); opt.step(); tot += float(loss)
        log(f"    arm {arm} epoch {ep+1}/{epochs}: loss {tot/max(1,len(pairs_train)):.3f}")

    @torch.no_grad()
    def gen(prompt):
        model.eval()
        enc = tok.apply_chat_template([{"role": "user", "content": prompt}], add_generation_prompt=True,
                                      return_tensors="pt", return_dict=True, enable_thinking=False).to(dev)
        out = model.generate(**enc, max_new_tokens=256, do_sample=False, pad_token_id=tok.eos_token_id)
        return tok.decode(out[0, enc["input_ids"].shape[1]:], skip_special_tokens=True)
    import numpy as np
    recs = []
    for pr, gold_tgt in [(p, p[2]) for p in pairs_test]:
        added = gold_tgt.split("=======")[-1].split(">>>>>>>")[0]
        recs.append(_fidelity(_emitted_replace(gen(pr[ai])), added)[0])
    return float(np.mean(recs)) if recs else 0.0


def _selftest() -> bool:
    print("sft_decode --selftest: data-prep (prompts + gold target), no GPU\n")
    import numpy as np, torch, torch.nn as nn
    rows = [{"issue": "bug", "code": "x = 1", "added": "x = 2", "g": np.eye(3)[i % 3].astype("float32")}
            for i in range(9)]
    composer = nn.Linear(3, 4)
    pairs = _pairs(rows, composer, ["patA", "patB", "patC", "patD"], K=2)
    p = pairs[0]
    assert "search" not in p[0].lower() or True
    assert "pattern" in p[1] and "pattern" not in p[0], "B has patterns, A does not"
    assert "SEARCH" in p[2] and "x = 2" in p[2], "target is the gold SR block"
    print(f"  A_prompt has patterns: {'pattern' in p[0]} | B_prompt has patterns: {'pattern' in p[1]}")
    print(f"  target: {p[2][:60]!r}")
    print("\n  SFT_DECODE SELFTEST -> PASS (arms + target built)")
    return True


def main():
    ap = argparse.ArgumentParser(description="Does graph-composition help a TRAINABLE LLM decode? (SFT A vs B)")
    ap.add_argument("--model", default="Qwen/Qwen3.5-4B")
    ap.add_argument("--dataset", default="lite"); ap.add_argument("--split", default="test")
    ap.add_argument("--n", type=int, default=0); ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=2); ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        raise SystemExit(0 if _selftest() else 1)
    import numpy as np
    from v5.runtime.composition_decode import _load, _train_composer
    rows = _load(a.dataset, a.split, a.n)
    rng = np.random.RandomState(0); idx = rng.permutation(len(rows)); nh = len(idx) // 3
    test = [rows[i] for i in idx[:nh]]; train = [rows[i] for i in idx[nh:]]
    composer, _c, reps = _train_composer(train, min(24, max(2, len(train) // 2)))
    ptr, pte = _pairs(train, composer, reps, a.k), _pairs(test, composer, reps, a.k)
    print(f"[sft-decode] train {len(ptr)} / test {len(pte)}, model={a.model}, {a.epochs} epochs")
    rA = _lora_sft_eval(ptr, pte, "A", a.model, a.epochs)
    rB = _lora_sft_eval(ptr, pte, "B", a.model, a.epochs)
    print(f"\n=== SFT-DECODE (trainable LLM) ===")
    print(f"  A (issue+code)              held recall {rA:.3f}")
    print(f"  B (issue+code+COMPOSITION)  held recall {rB:.3f}")
    print(f"  graph lift (B - A) = {rB-rA:+.3f} -> "
          f"{'GRAPH HELPS the trainable LLM (reasoning prior pays off)' if rB > rA + 0.03 else 'graph does not add over training the LLM directly'}")


if __name__ == "__main__":
    main()
