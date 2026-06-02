from __future__ import annotations

import pytest
import torch

from v5.lm_loader import resolve_dtype, resolve_quant


def test_resolve_quant_default_none(monkeypatch):
    monkeypatch.delenv("V5_LM_QUANT", raising=False)
    assert resolve_quant(None) == "none"


def test_resolve_quant_env_and_explicit(monkeypatch):
    monkeypatch.setenv("V5_LM_QUANT", "4bit")
    assert resolve_quant(None) == "4bit"        # from env
    assert resolve_quant("8bit") == "8bit"      # explicit arg wins
    assert resolve_quant("off") == "none"       # aliases -> none


def test_resolve_quant_rejects_bad():
    with pytest.raises(ValueError):
        resolve_quant("3bit")


def test_resolve_dtype_cpu_defaults_fp32(monkeypatch):
    monkeypatch.delenv("V5_LM_DTYPE", raising=False)
    assert resolve_dtype(torch.device("cpu")) == torch.float32


def test_resolve_dtype_env_override(monkeypatch):
    monkeypatch.setenv("V5_LM_DTYPE", "bf16")
    assert resolve_dtype(torch.device("cpu")) == torch.bfloat16
    monkeypatch.setenv("V5_LM_DTYPE", "fp16")
    assert resolve_dtype(torch.device("cpu")) == torch.float16
