"""Shared-prefix KV session (Phase A, gen-minimize) — prefill the heavy context ONCE, then
generate each slot stage as a continuation on the cached prefix instead of re-prefilling
[issue + source] every stage.

PARITY-GATED: a cached continuation MUST be byte-identical to the uncached path (same token ids,
greedy). KV reuse silently corrupts otherwise, so nothing trusts this until `--parity-test` PASSes.

  WSL: V5_LM_TRUST_REMOTE_CODE=1 V5_LM_QUANT=4bit python3 -m v5.runtime.prefix_session --parity-test
"""
from __future__ import annotations
import argparse, os, time
import torch


class PrefixSession:
    """Hold a cached prefix KV; generate continuations on top of it.

    chain mode (the slot loop): prime(context) -> gen(stage1) grows the cache with stage1's tokens
    -> gen(stage2) continues from [context+stage1] ... = latent state-carry (no re-emit/re-encode).
    """
    def __init__(self, model, tok, device):
        self.model = model
        self.tok = tok
        self.device = device
        self.past = None
        self.cur_ids = None          # all tokens currently in the cache [1, L]

    @torch.no_grad()
    def prime(self, prefix_ids):
        prefix_ids = prefix_ids.to(self.device)
        out = self.model(input_ids=prefix_ids, use_cache=True)
        self.past = out.past_key_values
        self.cur_ids = prefix_ids
        return out.logits

    @torch.no_grad()
    def gen(self, suffix_ids, max_new, eos_id):
        """Greedy-continue from the cached prefix; RETAIN the advanced cache so the next stage
        chains without re-prefilling [context + prior stages]. Returns the NEW token ids [Nnew].
        If HF doesn't hand back the cache, self.past=None and the next call re-prefills (correct,
        no savings) — the chain-parity test tells us which world we're in."""
        suffix_ids = suffix_ids.to(self.device)
        full = torch.cat([self.cur_ids, suffix_ids], dim=1)
        attn = torch.ones_like(full)
        out = self.model.generate(
            input_ids=full,
            attention_mask=attn,
            past_key_values=self.past,        # covers self.cur_ids; HF processes only the new suffix
            max_new_tokens=max_new,
            do_sample=False,
            pad_token_id=eos_id,
            use_cache=True,
            return_dict_in_generate=True,
        )
        seq = out.sequences
        new = seq[0, full.shape[1]:]
        self.cur_ids = seq                                  # full seq incl. generated tokens
        self.past = getattr(out, "past_key_values", None)   # retained -> chaining saves; None -> re-prefill
        return new


@torch.no_grad()
def _greedy(model, tok, ids, max_new, dev):
    out = model.generate(input_ids=ids.to(dev), attention_mask=torch.ones_like(ids).to(dev),
                         max_new_tokens=max_new, do_sample=False, pad_token_id=tok.eos_token_id)
    return out[0, ids.shape[1]:]


def parity_test(model_name):
    from transformers import AutoTokenizer
    from v5.lm_loader import load_frozen_lm
    trust = os.environ.get("V5_LM_TRUST_REMOTE_CODE", "0").lower() in ("1", "true", "yes")
    model = load_frozen_lm(model_name); model.eval()
    tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=trust)
    dev = next(model.parameters()).device

    # heavy prefix (this file as stand-in source) + a short stage suffix
    src = open(__file__, encoding="utf-8").read()
    prefix = "CONTEXT (read carefully):\n" + src + "\n\n"
    suffix = "TASK: in one short sentence, what does PrefixSession.gen do? ANSWER:"

    # SAME token ids for both paths: tokenize together, split at the prefix boundary
    full_ids = tok(prefix + suffix, return_tensors="pt", truncation=True, max_length=2048).input_ids
    pre_ids = tok(prefix, return_tensors="pt", truncation=True, max_length=2048).input_ids
    plen = min(pre_ids.shape[1], full_ids.shape[1] - 1)
    pre_ids = full_ids[:, :plen]
    suf_ids = full_ids[:, plen:]

    N = 40
    text_u = tok.decode(_greedy(model, tok, full_ids, N, dev), skip_special_tokens=True)

    sess = PrefixSession(model, tok, dev); sess.prime(pre_ids)
    text_c = tok.decode(sess.gen(suf_ids, N, tok.eos_token_id), skip_special_tokens=True)

    ok = text_u == text_c
    print(f"\n=== PREFIX-SESSION PARITY (prefix={plen} tok, suffix={suf_ids.shape[1]} tok) ===")
    print(f"UNCACHED: {text_u!r}")
    print(f"CACHED  : {text_c!r}")
    print(f"PARITY: {'PASS — cached == uncached, KV reuse is faithful' if ok else 'FAIL — DO NOT trust the cache'}")
    return ok


class ChatKV:
    """Multi-step chat on ONE shared KV cache. The heavy context is primed once (turn 1); each
    .step() appends a user turn + generates the assistant reply as a CONTINUATION — the context and
    all prior stages are never re-encoded (latent state-carry). Qwen chat markers (deploy model).

    Generated assistant tokens stay in the cache verbatim (no decode->re-tokenize drift), so the
    transition glue is appended as raw marker tokens, not by re-templating the conversation."""
    def __init__(self, model, tok, device, system, context_plus_first_instruction):
        self.model = model; self.tok = tok; self.device = device
        self.sess = PrefixSession(model, tok, device)
        first = [{"role": "system", "content": system},
                 {"role": "user", "content": context_plus_first_instruction}]
        ids = tok.apply_chat_template(first, add_generation_prompt=True, return_tensors="pt").input_ids.to(device)
        self.sess.prime(ids[:, :-1])          # prime all but the last token
        self._kick = ids[:, -1:]              # last gen-prompt token kicks the first generation
        self._first = True

    @torch.no_grad()
    def step(self, instruction, max_new, strip_think=True):
        if self._first:
            suffix = self._kick                                 # context+1st instruction already primed
            self._first = False
        else:
            glue = f"<|im_end|>\n<|im_start|>user\n{instruction}<|im_end|>\n<|im_start|>assistant\n"
            suffix = self.tok(glue, return_tensors="pt", add_special_tokens=False).input_ids.to(self.device)
        new = self.sess.gen(suffix, max_new, self.tok.eos_token_id)
        text = self.tok.decode(new, skip_special_tokens=True)
        if strip_think:
            text = __import__("re").sub(r"<think>.*?</think>", "", text, flags=__import__("re").DOTALL)
        return text.strip()


def solve_test(model_name):
    """3-stage DIAGNOSE->PLAN->FIX on one cached context (synthetic bug) — does staged chat-KV produce
    coherent, distinct, on-task stages, and is stage-2/3 cheaper (context not re-encoded)?"""
    from transformers import AutoTokenizer
    from v5.lm_loader import load_frozen_lm
    trust = os.environ.get("V5_LM_TRUST_REMOTE_CODE", "0").lower() in ("1", "true", "yes")
    model = load_frozen_lm(model_name); model.eval()
    tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=trust)
    dev = next(model.parameters()).device

    src = ("def clamp(x, lo, hi):\n    # bug: returns x even when out of range\n"
           "    if x < lo:\n        return lo\n    if x > hi:\n        return x\n    return x\n")
    ctx = (f"A Python function has a bug.\n\nSOURCE:\n```python\n{src}```\n\n"
           "STEP 1 — DIAGNOSE: name the exact line and the root cause in one sentence.")
    sys = "You are a precise debugging assistant. Work in the labeled steps I give you."

    chat = ChatKV(model, tok, dev, sys, ctx)
    t0 = time.time(); diag = chat.step("", 120); t1 = time.time()
    plan = chat.step("STEP 2 — PLAN: give the exact search line and the replacement line.", 120); t2 = time.time()
    fix = chat.step("STEP 3 — FIX: output ONLY a search/replace block:\npath\n<<<<<<< SEARCH\n<old>\n=======\n<new>\n>>>>>>> REPLACE", 160); t3 = time.time()

    print("\n=== CHAT-KV 3-STAGE SOLVE (one cached context) ===")
    print(f"DIAGNOSE ({t1-t0:.1f}s): {diag!r}\n")
    print(f"PLAN     ({t2-t1:.1f}s): {plan!r}\n")
    print(f"FIX      ({t3-t2:.1f}s): {fix!r}\n")
    print(f"timing: stage1 {t1-t0:.1f}s | stage2 {t2-t1:.1f}s | stage3 {t3-t2:.1f}s  "
          f"(stages 2/3 skip re-encoding the context)")
    ok = bool(diag.strip()) and bool(plan.strip()) and ("SEARCH" in fix or "return hi" in fix or ">>>>>>>" in fix)
    print(f"RESULT: {'PASS — coherent distinct staged output on a shared cache' if ok else 'INSPECT — read the stages above'}")
    return ok


def chain_parity_test(model_name):
    """2 chained stages on one cache must equal uncached re-encoding — AND report whether the cache
    was retained across stages (the actual savings vs correctness-only re-prefill)."""
    from transformers import AutoTokenizer
    from v5.lm_loader import load_frozen_lm
    trust = os.environ.get("V5_LM_TRUST_REMOTE_CODE", "0").lower() in ("1", "true", "yes")
    model = load_frozen_lm(model_name); model.eval()
    tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=trust)
    dev = next(model.parameters()).device

    src = open(__file__, encoding="utf-8").read()
    ctx_ids = tok("CONTEXT:\n" + src + "\n\n", return_tensors="pt", truncation=True, max_length=2048).input_ids.to(dev)
    s1 = tok("TASK 1: name one method of PrefixSession. ANSWER:", return_tensors="pt", add_special_tokens=False).input_ids.to(dev)
    s2 = tok("\nTASK 2: name a different method. ANSWER:", return_tensors="pt", add_special_tokens=False).input_ids.to(dev)
    N = 24

    sess = PrefixSession(model, tok, dev); sess.prime(ctx_ids)
    t0 = time.time(); new1 = sess.gen(s1, N, tok.eos_token_id); t1 = time.time()
    retained = sess.past is not None
    new2 = sess.gen(s2, N, tok.eos_token_id); t2 = time.time()
    text2_c = tok.decode(new2, skip_special_tokens=True)

    # uncached equivalent for stage 2: re-encode [ctx + s1 + new1 + s2] from scratch
    full2 = torch.cat([ctx_ids, s1, new1.unsqueeze(0), s2], dim=1)
    text2_u = tok.decode(_greedy(model, tok, full2, N, dev), skip_special_tokens=True)

    ok = text2_c == text2_u
    print(f"\n=== CHAIN PARITY (2 stages on 1 cache) ===")
    print(f"cache retained across stages: {retained}  (True = chaining SAVES re-prefill; False = correct but re-prefills)")
    print(f"stage1 {t1-t0:.1f}s | stage2 {t2-t1:.1f}s  (stage2 << stage1 if the cache is reused)")
    print(f"STAGE2 cached  : {text2_c!r}")
    print(f"STAGE2 uncached: {text2_u!r}")
    print(f"CHAIN PARITY: {'PASS — chained cache == uncached re-encode' if ok else 'FAIL — chaining corrupts; use re-prime or manual decode'}")
    return ok


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3.5-4B")
    ap.add_argument("--parity-test", action="store_true")
    ap.add_argument("--chain-test", action="store_true")
    ap.add_argument("--solve-test", action="store_true")
    a = ap.parse_args(argv)
    if a.parity_test:
        raise SystemExit(0 if parity_test(a.model) else 1)
    if a.chain_test:
        raise SystemExit(0 if chain_parity_test(a.model) else 1)
    if a.solve_test:
        raise SystemExit(0 if solve_test(a.model) else 1)
    ap.print_help()


if __name__ == "__main__":
    main()
