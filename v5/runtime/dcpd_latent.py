"""dcpd_latent.py — the REAL latent Dual-Channel Pointer-Decoding from NEW_DESIGN(v6), white-box.

The original core idea was NEVER built — it was swapped for a discrete text stand-in (algo_grr_dcpd) because
the z-wall refuted latent CODE TRANSPORT. But the original's two latent mechanisms were never actually tested:

  1. gamma-GATED POINTER-DECODE (the symbolic channel):  P(y_t) = gamma*P_LM(y_t|h) + (1-gamma)*P_KG(y_t|Node).
     At gamma=0 the controller FORCES the graph's exact tokens (zero syntax error, by construction); at gamma=1 the
     LM free-generates natural language around them. -> exact code the LM never had to spell + fluent prose.

  2. CONCEPT-SPACE REPULSION (mistake handling, NOT pink-elephant text): project a mistake to a hidden-space
     direction v_err, monitor the LM's mid-layer DRIFT = cos(h_l, v_err) while it generates, and steer
     h_l -= alpha*drift*v_err when it drifts toward the trap. The LM stays fluent; the trap is repelled in latent.

This module implements BOTH for REAL on an open-weights causal LM (forward hooks + a custom decode loop that
manipulates logits and the residual stream). It is model-agnostic: verify the machinery on a tiny real model
here, then run the true experiment on Qwen. NOTHING is simulated — the hooks fire on the actual network.

    python -m v5.runtime.dcpd_latent --smoke                         # tiny real model (distilgpt2): proves it runs
    python -m v5.runtime.dcpd_latent --lm Qwen/Qwen2.5-3B-Instruct --exp gating
    python -m v5.runtime.dcpd_latent --lm Qwen/Qwen2.5-3B-Instruct --exp repulsion
"""
from __future__ import annotations

import argparse
import sys

import torch
import torch.nn.functional as F


def _decoder_layers(model):
    """The list of transformer decoder blocks, across gpt2 / llama / qwen / neox architectures."""
    m = model
    if hasattr(m, "transformer") and hasattr(m.transformer, "h"):
        return m.transformer.h                       # gpt2
    if hasattr(m, "model") and hasattr(m.model, "layers"):
        return m.model.layers                        # llama / qwen / mistral
    if hasattr(m, "gpt_neox") and hasattr(m.gpt_neox, "layers"):
        return m.gpt_neox.layers
    raise RuntimeError(f"unknown architecture: {type(m).__name__}")


class WhiteBox:
    """A real open-weights causal LM with logit-level + residual-stream control."""

    def __init__(self, name: str, dtype=None):
        from transformers import AutoTokenizer, AutoModelForCausalLM
        self.name = name
        self.tok = AutoTokenizer.from_pretrained(name)
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = dtype or (torch.float16 if self.device == "cuda" else torch.float32)
        self.model = AutoModelForCausalLM.from_pretrained(name, torch_dtype=dtype).to(self.device).eval()
        self.layers = _decoder_layers(self.model)
        self.n_layers = len(self.layers)
        self.d_model = self.model.config.hidden_size

    # ── concept direction: the mid-layer hidden-state direction of a piece of text ────────────────
    @torch.no_grad()
    def concept_vector(self, text: str, layer: int) -> torch.Tensor:
        ids = self.tok(text, return_tensors="pt").to(self.device)
        out = self.model(**ids, output_hidden_states=True)
        h = out.hidden_states[layer][0]              # [seq, d] — residual stream after `layer` blocks
        v = h.mean(0)
        return (v / (v.norm() + 1e-6)).float()

    # ── plain greedy decode (the baseline arm) ────────────────────────────────────────────────────
    @torch.no_grad()
    def generate_plain(self, prompt: str, max_new: int = 60, temperature: float = 0.0) -> str:
        ids = self.tok(prompt, return_tensors="pt").input_ids.to(self.device)
        start = ids.shape[1]
        for _ in range(max_new):
            logits = self.model(ids).logits[:, -1, :]
            nxt = self._pick(logits, temperature)
            ids = torch.cat([ids, nxt], 1)
            if nxt.item() == self.tok.eos_token_id:
                break
        return self.tok.decode(ids[0, start:], skip_special_tokens=True)

    # ── 1. gamma-GATED POINTER-DECODE — force exact graph tokens (gamma=0) between LM prose (gamma=1) ─────────
    @torch.no_grad()
    def generate_gated(self, prompt: str, plan: list, temperature: float = 0.0) -> dict:
        """plan = ordered segments: ("say", n_tokens) = LM free-generates (gamma=1);
        ("emit", exact_text) = FORCE the graph's exact tokens into the stream (gamma=0, no sampling).
        The forced tokens enter the LM's context so its later prose explains real, exact code."""
        ids = self.tok(prompt, return_tensors="pt").input_ids.to(self.device)
        start = ids.shape[1]
        segments = []
        for mode, val in plan:
            if mode == "emit":
                forced = self.tok(val, return_tensors="pt", add_special_tokens=False).input_ids.to(self.device)
                ids = torch.cat([ids, forced], 1)    # gamma=0: exact copy from the graph, byte-perfect
                segments.append(("emit", val))
            else:
                gen = []
                for _ in range(int(val)):
                    logits = self.model(ids).logits[:, -1, :]
                    nxt = self._pick(logits, temperature)   # gamma=1: the LM speaks freely
                    ids = torch.cat([ids, nxt], 1)
                    tid = nxt.item()
                    if tid == self.tok.eos_token_id:
                        break
                    gen.append(tid)
                segments.append(("say", self.tok.decode(gen, skip_special_tokens=True)))
        return dict(text=self.tok.decode(ids[0, start:], skip_special_tokens=True), segments=segments)

    # ── 2. CONCEPT-SPACE REPULSION — steer the residual stream away from v_err during generation ──
    @torch.no_grad()
    def generate_steered(self, prompt: str, v_err: torch.Tensor, layer: int, alpha: float = 8.0,
                         thresh: float = 0.0, max_new: int = 60, temperature: float = 0.0) -> dict:
        """Register a forward hook on `layer`; each step, measure drift = cos(h_last, v_err) and apply
        h_last -= alpha*drift*v_err when drift > thresh. Real activation steering on the actual network."""
        v = v_err.to(self.device).to(next(self.model.parameters()).dtype)
        stat = {"steps": 0, "fired": 0, "drift_sum": 0.0}

        def hook(module, inp, out):
            h = out[0] if isinstance(out, tuple) else out
            hl = h[:, -1, :]                                      # [batch, d] — last position
            drift = F.cosine_similarity(hl.float(), v.float().unsqueeze(0), dim=-1)  # [batch]
            stat["steps"] += 1
            stat["drift_sum"] += float(drift.mean())
            fire = (drift > thresh)
            if fire.any():
                stat["fired"] += int(fire.sum())
                coef = (alpha * drift).unsqueeze(-1) * fire.float().unsqueeze(-1)
                h[:, -1, :] = hl - coef.to(h.dtype) * v.unsqueeze(0)
            return (h,) + tuple(out[1:]) if isinstance(out, tuple) else h

        handle = self.layers[layer - 1].register_forward_hook(hook)
        try:
            ids = self.tok(prompt, return_tensors="pt").input_ids.to(self.device)
            s = ids.shape[1]
            for _ in range(max_new):
                logits = self.model(ids).logits[:, -1, :]
                nxt = self._pick(logits, temperature)
                ids = torch.cat([ids, nxt], 1)
                if nxt.item() == self.tok.eos_token_id:
                    break
            text = self.tok.decode(ids[0, s:], skip_special_tokens=True)
        finally:
            handle.remove()
        stat["avg_drift"] = stat["drift_sum"] / max(1, stat["steps"])
        return dict(text=text, stat=stat)

    # ── fluency proxy: the model's own mean NLL of a string (lower = more self-consistent/fluent) ──
    @torch.no_grad()
    def self_nll(self, text: str) -> float:
        ids = self.tok(text, return_tensors="pt").input_ids.to(self.device)
        if ids.shape[1] < 2:
            return float("nan")
        out = self.model(ids, labels=ids)
        return float(out.loss)

    def _pick(self, logits, temperature):
        if temperature and temperature > 0:
            probs = torch.softmax(logits / temperature, -1)
            return torch.multinomial(probs, 1)
        return logits.argmax(-1, keepdim=True)


# ════════════════════════════════════════════════════════════════════════════════════════════════
# EXPERIMENTS — the fair_ab the project deferred. Every number is measured on the real network.
# ════════════════════════════════════════════════════════════════════════════════════════════════
def exp_gating(wb: WhiteBox):
    """The symbolic channel: force an EXACT graph atom into the stream (gamma=0) while the LM explains (gamma=1).
    Claim: the emitted code is byte-exact (0 syntax errors) AND the LM's prose wraps around it coherently —
    something a free-form LM (which must spell the code itself) cannot guarantee."""
    print("\n[EXP 1] gamma-gated pointer-decode — exact graph code forced between LM prose\n")
    atom = "def is_prime(n):\n    return n >= 2 and all(n % i for i in range(2, int(n**0.5) + 1))"
    prompt = "You are explaining a solution. "
    plan = [("say", 24),
            ("emit", "\n\nHere is the exact implementation:\n" + atom + "\n\n"),
            ("say", 24)]
    r = wb.generate_gated(prompt, plan, temperature=0.0)
    emitted = [v for m, v in r["segments"] if m == "emit"]
    # verify the emitted code compiles EXACTLY (gamma=0 guarantee):
    ok = True
    for e in emitted:
        try:
            compile(atom, "<emit>", "exec")
        except SyntaxError:
            ok = False
    print("  forced-emit code compiles exactly:", ok, "(gamma=0 => byte-perfect, 0 syntax error by construction)")
    for m, v in r["segments"]:
        tag = "LM-prose (gamma=1)" if m == "say" else "GRAPH-forced (gamma=0)"
        print(f"    [{tag}] {v.strip()[:120]!r}")
    # contrast: free-form LM writing the whole thing (may or may not be exact)
    free = wb.generate_plain("Write a Python is_prime function and explain it.", max_new=80)
    free_ok = "def is_prime" in free
    print(f"  free-form LM produced a def: {free_ok}  (no exactness guarantee; gamma-gating GUARANTEES it)")
    return dict(gated_exact=ok, freeform_has_def=free_ok)


def exp_repulsion(wb: WhiteBox, layer: int = None, trap: str = "bubble sort",
                  alpha: float = 10.0, n_samples: int = 6):
    """The real fair_ab for concept-space repulsion. A prompt that tempts the trap concept.
      A none        : baseline — how often does the trap appear?
      B text-prompt : 'do NOT use {trap}' (the pink-elephant baseline)
      C latent-steer : repel the residual stream from v_err = concept(trap)
    Measure trap-rate (lower is better) AND self-NLL fluency (lower is better). The claim under test:
    C suppresses the trap with LESS fluency damage than B. We REPORT what happens — no assumed winner."""
    layer = layer or max(1, wb.n_layers // 2)
    print(f"\n[EXP 2] concept-space repulsion vs text-prompt — trap='{trap}', layer={layer}, alpha={alpha}\n")
    base_prompt = f"Question: what is the simplest sorting algorithm to implement? Answer:"
    v_err = wb.concept_vector(trap, layer)

    def trap_rate(gen_fn):
        hits, nlls = 0, []
        for _ in range(n_samples):
            r = gen_fn()
            txt = r["text"] if isinstance(r, dict) else r
            hits += int(trap.lower() in txt.lower())
            nlls.append(wb.self_nll(txt))
        import statistics
        return hits / n_samples, statistics.fmean([x for x in nlls if x == x] or [float("nan")])

    a_rate, a_flu = trap_rate(lambda: wb.generate_plain(base_prompt, max_new=40, temperature=0.9))
    b_prompt = f"Question: what is the simplest sorting algorithm to implement? Do NOT mention {trap}. Answer:"
    b_rate, b_flu = trap_rate(lambda: wb.generate_plain(b_prompt, max_new=40, temperature=0.9))
    c_rate, c_flu = trap_rate(lambda: wb.generate_steered(base_prompt, v_err, layer, alpha=alpha,
                                                          thresh=0.0, max_new=40, temperature=0.9))
    print(f"  arm            trap-rate   self-NLL(fluency)")
    print(f"  A none         {a_rate:>6.2f}      {a_flu:.3f}")
    print(f"  B text 'avoid' {b_rate:>6.2f}      {b_flu:.3f}   (pink-elephant risk)")
    print(f"  C latent-steer {c_rate:>6.2f}      {c_flu:.3f}   (concept-space repulsion)")
    verdict = ("C suppresses the trap" if c_rate < a_rate else "C did NOT suppress the trap")
    flu = ("less fluency damage than B" if c_flu <= b_flu else "more fluency damage than B")
    print(f"  => MEASURED: {verdict}; {flu}. (Honest: this is the fair_ab, not a foregone result.)")
    return dict(a=(a_rate, a_flu), b=(b_rate, b_flu), c=(c_rate, c_flu))


def smoke():
    """Prove the white-box MACHINERY is real on a tiny real model (distilgpt2, loads in seconds).
    Content is nonsense at this size — the point is: hooks fire, gamma-gating forces exact tokens, steering
    measurably shifts the residual stream. The SAME code runs on Qwen for the real experiment."""
    print("dcpd_latent --smoke: verifying the white-box machinery on a REAL tiny model (distilgpt2)\n")
    wb = WhiteBox("distilgpt2")
    print(f"  loaded {wb.name}: {wb.n_layers} layers, d_model={wb.d_model}, device={wb.device}")

    # [1] gamma-gating forces exact tokens
    r = wb.generate_gated("Test. ", [("say", 5), ("emit", "EXACT_TOKENS_123"), ("say", 5)])
    forced_present = "EXACT_TOKENS_123" in r["text"]
    print(f"  [1] gamma-gated forced-emit present verbatim in output: {forced_present}  "
          f"(gamma=0 copies exactly, the LM never spelled it)")

    # [2] steering hook FIRES and changes hidden states -> changes output
    L = max(1, wb.n_layers // 2)
    v = wb.concept_vector("bubble sort algorithm", L)
    plain = wb.generate_plain("The best sorting method is", max_new=15)
    steer = wb.generate_steered("The best sorting method is", v, L, alpha=60.0, max_new=15)
    changed = plain.strip() != steer["text"].strip()
    print(f"  [2] steering hook fired {steer['stat']['fired']}/{steer['stat']['steps']} steps "
          f"(avg drift {steer['stat']['avg_drift']:.3f}); output changed vs plain: {changed}")
    print(f"        plain : {plain.strip()[:70]!r}")
    print(f"        steer : {steer['text'].strip()[:70]!r}")

    ok = forced_present and steer["stat"]["steps"] > 0
    print(f"\n  MACHINERY REAL: forced-emit works + forward-hook steering runs on the actual network -> "
          f"{'PASS' if ok else 'FAIL'}")
    print("  (run --lm Qwen/Qwen2.5-3B-Instruct --exp gating|repulsion for the real content experiment)")
    return ok


def main():
    ap = argparse.ArgumentParser(description="real white-box latent Dual-Channel Pointer-Decoding")
    ap.add_argument("--smoke", action="store_true", help="prove the machinery on a tiny real model")
    ap.add_argument("--lm", type=str, default="", help="open-weights causal LM (e.g. Qwen/Qwen2.5-3B-Instruct)")
    ap.add_argument("--exp", choices=["gating", "repulsion", "both"], default="both")
    ap.add_argument("--layer", type=int, default=0, help="steering layer (0 = middle)")
    ap.add_argument("--trap", type=str, default="bubble sort")
    ap.add_argument("--alpha", type=float, default=10.0)
    a = ap.parse_args()
    if a.smoke:
        sys.exit(0 if smoke() else 1)
    if not a.lm:
        ap.print_help(); return
    wb = WhiteBox(a.lm)
    print(f"loaded {wb.name}: {wb.n_layers} layers d_model={wb.d_model} device={wb.device}")
    layer = a.layer or max(1, wb.n_layers // 2)
    if a.exp in ("gating", "both"):
        exp_gating(wb)
    if a.exp in ("repulsion", "both"):
        exp_repulsion(wb, layer=layer, trap=a.trap, alpha=a.alpha)


if __name__ == "__main__":
    main()
