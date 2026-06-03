"""Shared frozen base-LM loader with optional 4-bit/8-bit quantization.

ONE place to decide base-LM precision so every entrypoint loads it identically.
This matters for the V5 plan: the adapter conditions on the base LM's layer-8/20
hidden states, so TRAIN and DEPLOY must use the same precision. The 6GB-inference
target is a 4-bit (nf4) Qwen3-4B; if h_init is precomputed in fp32 but deployed in
4-bit, the adapter sees a different distribution than it trained on.

Precision is env-driven so you flip it globally without editing call sites:

    V5_LM_QUANT = none (default) | 4bit | 8bit
    V5_LM_DTYPE = auto (default: bf16 on cuda, fp32 on cpu) | bf16 | fp16 | fp32

Dev (0.5B / 1.5B): leave V5_LM_QUANT unset -> bf16/fp32, no bitsandbytes needed.
Deploy (Qwen3-4B, 6GB): V5_LM_QUANT=4bit  (needs bitsandbytes; rented GPU / Linux).

bitsandbytes is imported lazily ONLY when a quant mode is requested, so the
Windows dev box never needs it.
"""
from __future__ import annotations

import os
from typing import Optional

import torch

_QUANT_MODES = {"none", "4bit", "8bit"}


def resolve_dtype(device: torch.device) -> torch.dtype:
    """Pick a compute dtype from V5_LM_DTYPE (default: bf16 on cuda, fp32 on cpu)."""
    pref = os.environ.get("V5_LM_DTYPE", "auto").lower()
    if pref in ("fp32", "float32"):
        return torch.float32
    if pref in ("bf16", "bfloat16"):
        return torch.bfloat16
    if pref in ("fp16", "float16"):
        return torch.float16
    if device.type == "cuda":
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    return torch.float32   # bf16/fp16 on CPU is slow / poorly supported


def resolve_quant(quant: Optional[str]) -> str:
    q = (quant if quant is not None else os.environ.get("V5_LM_QUANT", "none")).lower()
    if q in ("", "no", "off", "false"):
        q = "none"
    if q not in _QUANT_MODES:
        raise ValueError(f"V5_LM_QUANT/quant must be one of {_QUANT_MODES}, got {q!r}")
    return q


def load_frozen_lm(model_name: str, *, device: Optional[torch.device] = None,
                   quant: Optional[str] = None):
    """Load a frozen AutoModelForCausalLM with optional 4-bit/8-bit quantization.

    quant: None -> env V5_LM_QUANT (default "none"). Returns the model in eval mode
    with all params frozen (requires_grad False). Caller loads the tokenizer.
    """
    from transformers import AutoModelForCausalLM
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    q = resolve_quant(quant)
    dtype = resolve_dtype(device)
    # new custom architectures (e.g. Qwen3.5 hybrid) ship a remote modeling file;
    # enable via env until transformers integrates them natively.
    trust = os.environ.get("V5_LM_TRUST_REMOTE_CODE", "0").lower() in ("1", "true", "yes")
    extra = {"trust_remote_code": True} if trust else {}

    if q in ("4bit", "8bit"):
        from transformers import BitsAndBytesConfig   # lazy: only when quantizing
        if q == "4bit":
            qcfg = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=dtype)
        else:
            qcfg = BitsAndBytesConfig(load_in_8bit=True)
        # bitsandbytes places the weights itself via device_map; do NOT call .to().
        # NOTE: torch.device("cuda") has index None -> {"": None} crashes the loader;
        # resolve to a concrete cuda:N (or cpu).
        if device.type == "cuda":
            dev = f"cuda:{device.index if device.index is not None else 0}"
        else:
            dev = "cpu"
        model = AutoModelForCausalLM.from_pretrained(
            model_name, quantization_config=qcfg, device_map={"": dev}, **extra)
    else:
        # transformers>=5 renamed torch_dtype -> dtype
        try:
            model = AutoModelForCausalLM.from_pretrained(model_name, dtype=dtype, **extra)
        except TypeError:
            model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype, **extra)
        model = model.to(device)

    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model
