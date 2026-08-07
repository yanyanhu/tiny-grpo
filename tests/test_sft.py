"""CPU-only tests for SFT profiles, targets, and trainer configuration."""

import dataclasses

import pytest

from tiny_grpo.hardware import CUDA_4GB, MPS_16GB
from tiny_grpo.model_profiles import QWEN3_0_6B
from tiny_grpo.sft_config import (
    SFTConfigError,
    sft_debug_config,
    sft_smoke_config,
    sft_stronger_config,
)
from tiny_grpo.sft_data import SFTTargetError, audit_sft_lengths, build_sft_target, to_sft_example
from train_sft import build_trainer_config


@pytest.mark.parametrize("factory", [sft_smoke_config, sft_debug_config, sft_stronger_config])
@pytest.mark.parametrize("hardware", [MPS_16GB, CUDA_4GB])
def test_sft_profiles_validate_on_both_hardware(factory, hardware):
    config = factory(hardware)
    assert config.hardware_profile_name == hardware.name
    assert config.completion_only_loss is True
    assert config.checkpoint_retention <= 2


def test_cuda_sft_requires_gradient_checkpointing():
    with pytest.raises(SFTConfigError):
        dataclasses.replace(sft_smoke_config(CUDA_4GB), gradient_checkpointing=False)


def test_qwen_sft_profile_preserves_non_thinking_template_kwargs():
    config = sft_smoke_config(CUDA_4GB, model_profile=QWEN3_0_6B)
    assert config.run_name == "sft_smoke_qwen3_0_6b"
    assert config.chat_template_kwargs == {"enable_thinking": False}
    mapped = to_sft_example(
        {"question": "What is 1 + 1?", "answer": "Add.\n#### 2"},
        chat_template_kwargs=config.chat_template_kwargs,
    )
    assert mapped["chat_template_kwargs"] == {"enable_thinking": False}


def test_sft_rejects_device_that_does_not_match_hardware_profile():
    with pytest.raises(SFTConfigError, match="device.*does not match"):
        dataclasses.replace(sft_smoke_config(CUDA_4GB), device="mps")


def test_sft_allows_explicit_precision_override():
    config = dataclasses.replace(sft_smoke_config(CUDA_4GB), precision="fp16")
    assert config.precision == "fp16"


@pytest.mark.parametrize("field", ["logging_steps", "checkpoint_steps", "eval_steps"])
def test_sft_rejects_nonpositive_cadence(field):
    with pytest.raises(SFTConfigError, match=field):
        dataclasses.replace(sft_smoke_config(MPS_16GB), **{field: 0})


def test_stronger_profile_is_two_effective_epochs_on_each_hardware():
    cuda = sft_stronger_config(CUDA_4GB)
    mps = sft_stronger_config(MPS_16GB)
    assert cuda.dataset.train_size == mps.dataset.train_size == 1024
    assert cuda.max_steps == 256  # 1024 / (batch 1 * accumulation 8) * 2
    assert mps.max_steps == 512  # 1024 / (batch 1 * accumulation 4) * 2
    assert cuda.checkpoint_steps == cuda.eval_steps == 64
    assert mps.checkpoint_steps == mps.eval_steps == 128


def test_target_removes_calculator_markup_and_rewrites_final_answer():
    target = build_sft_target("First compute 6 * 7 = <<6*7=42>>42.\n#### 42.0")
    assert target == "First compute 6 * 7 = 42.\n<answer>42</answer>"
    assert target.endswith("<answer>42</answer>")
    assert target.count("<answer>") == 1
    assert "####" not in target and "<<" not in target


def test_invalid_target_fails_loudly():
    with pytest.raises(SFTTargetError):
        build_sft_target("There is no native answer marker here.")


def test_sft_example_keeps_gold_out_of_final_user_message():
    example = {"question": "What is 40 + 2?", "answer": "Add them.\n#### 42"}
    mapped = to_sft_example(example)
    assert mapped["prompt"][-1] == {"role": "user", "content": example["question"]}
    assert mapped["completion"][0]["content"].endswith("<answer>42</answer>")


def test_trainer_config_enables_completion_only_loss(tmp_path):
    args = build_trainer_config(sft_smoke_config(MPS_16GB), tmp_path)
    assert args.completion_only_loss is True
    assert args.assistant_only_loss is False
    assert args.packing is False
    assert args.save_total_limit == 2


class _MappingTokenizer:
    def apply_chat_template(self, conversation, tokenize):
        assert tokenize is True
        return {"input_ids": [10, 11, 12, 13], "attention_mask": [1, 1, 1, 1]}


def test_length_audit_counts_mapping_input_ids_not_mapping_keys():
    dataset = [{"prompt": [{"role": "user", "content": "q"}], "completion": [{"role": "assistant", "content": "a"}]}]
    stats = audit_sft_lengths(dataset, _MappingTokenizer(), max_sequence_length=4)
    assert stats["min_tokens"] == 4
    assert stats["max_tokens"] == 4


def test_length_audit_rejects_silent_truncation():
    dataset = [{"prompt": [{"role": "user", "content": "q"}], "completion": [{"role": "assistant", "content": "a"}]}]
    with pytest.raises(SFTTargetError, match="refusing silent truncation"):
        audit_sft_lengths(dataset, _MappingTokenizer(), max_sequence_length=3)
