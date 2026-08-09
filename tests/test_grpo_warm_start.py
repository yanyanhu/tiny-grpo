"""CPU-only tests for GRPO initialization from an SFT LoRA adapter."""

import dataclasses
from types import SimpleNamespace

import pytest

import train_grpo
from tiny_grpo.config import smoke_config
from tiny_grpo.hardware import CUDA_4GB
from tiny_grpo.model_profiles import QWEN3_0_6B


class _Model:
    def to(self, device):
        self.device = device
        return self


def _config(path):
    return dataclasses.replace(
        smoke_config(CUDA_4GB, model_profile=QWEN3_0_6B),
        initial_adapter_path=str(path),
        initial_adapter_source="matched-distilled-sft",
    )


def test_warm_start_loads_matching_adapter_as_trainable(monkeypatch, tmp_path):
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    base = _Model()
    wrapped = _Model()
    calls = {}
    monkeypatch.setattr(train_grpo.AutoModelForCausalLM, "from_pretrained", lambda *a, **k: base)
    monkeypatch.setattr(
        train_grpo.PeftConfig, "from_pretrained",
        lambda path: SimpleNamespace(base_model_name_or_path="Qwen/Qwen3-0.6B"),
    )

    def load(model, path, **kwargs):
        calls.update(model=model, path=path, **kwargs)
        return wrapped

    monkeypatch.setattr(train_grpo.PeftModel, "from_pretrained", load)
    result = train_grpo.load_training_model(_config(adapter), "cpu")
    assert result is wrapped
    assert calls["model"] is base
    assert calls["is_trainable"] is True


def test_warm_start_rejects_base_model_mismatch(monkeypatch, tmp_path):
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    monkeypatch.setattr(train_grpo.AutoModelForCausalLM, "from_pretrained", lambda *a, **k: _Model())
    monkeypatch.setattr(
        train_grpo.PeftConfig, "from_pretrained",
        lambda path: SimpleNamespace(base_model_name_or_path="different/model"),
    )
    with pytest.raises(ValueError, match="base model mismatch"):
        train_grpo.load_training_model(_config(adapter), "cpu")


def test_warm_start_rejects_missing_directory(monkeypatch, tmp_path):
    monkeypatch.setattr(train_grpo.AutoModelForCausalLM, "from_pretrained", lambda *a, **k: _Model())
    with pytest.raises(FileNotFoundError, match="not a directory"):
        train_grpo.load_training_model(_config(tmp_path / "missing"), "cpu")
