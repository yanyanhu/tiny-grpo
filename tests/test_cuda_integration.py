"""CUDA integration test: real model + LoRA + precision compatibility.

Mirrors tests/test_mps_integration.py for the cuda_4gb profile. Loads the real
model — not a pure unit test. Skipped (not failed) when no CUDA device is
present, so it's inert on this Mac but runs for real on the RTX workstation.
"""

import pytest
import torch
from peft import get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

from tiny_grpo.config import LoraConfig, TrainingConfig
from tiny_grpo.hardware import resolve_dtype
from tiny_grpo.lora import to_peft_lora_config

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available on this machine")

MODEL_ID = TrainingConfig.__dataclass_fields__["model_id"].default


@pytest.fixture(scope="module")
def tokenizer():
    return AutoTokenizer.from_pretrained(MODEL_ID)


@pytest.mark.parametrize("precision", ["fp32", "bf16", "fp16"])
def test_model_loads_wraps_with_lora_and_generates_on_cuda(precision, tokenizer):
    dtype = resolve_dtype(precision)
    model_kwargs = {"device_map": None}
    if dtype is not None:
        model_kwargs["dtype"] = dtype

    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, **model_kwargs)
    if dtype is not None:
        assert {p.dtype for p in model.parameters()} == {dtype}

    peft_model = get_peft_model(model, to_peft_lora_config(LoraConfig()))
    peft_model.to("cuda")

    inputs = tokenizer("What is 2 + 2?", return_tensors="pt").to("cuda")
    with torch.no_grad():
        output = peft_model.generate(**inputs, max_new_tokens=5, do_sample=False)

    assert output.shape[1] > inputs["input_ids"].shape[1]
    assert torch.isfinite(output.float()).all()
