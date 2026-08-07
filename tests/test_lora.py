"""CPU-only unit tests for tiny_grpo.lora. No model loaded."""

from peft import TaskType

from tiny_grpo.config import LoraConfig
from tiny_grpo.lora import missing_lora_target_modules, to_peft_lora_config


def test_maps_fields_correctly():
    lora = LoraConfig(r=8, alpha=16, dropout=0.05, target_modules=("q_proj", "k_proj", "v_proj", "o_proj"))

    peft_config = to_peft_lora_config(lora)

    assert peft_config.r == 8
    assert peft_config.lora_alpha == 16
    assert peft_config.lora_dropout == 0.05
    assert set(peft_config.target_modules) == {"q_proj", "k_proj", "v_proj", "o_proj"}


def test_sets_causal_lm_task_type():
    peft_config = to_peft_lora_config(LoraConfig())
    assert peft_config.task_type == TaskType.CAUSAL_LM


def test_custom_values_roundtrip():
    lora = LoraConfig(r=16, alpha=32, dropout=0.1, target_modules=("q_proj",))
    peft_config = to_peft_lora_config(lora)
    assert peft_config.r == 16
    assert peft_config.lora_alpha == 32
    assert peft_config.lora_dropout == 0.1
    assert list(peft_config.target_modules) == ["q_proj"]


def test_missing_lora_targets_uses_module_name_suffixes():
    class FakeModel:
        def named_modules(self):
            return iter([("layers.0.q_proj", object()), ("layers.0.v_proj", object())])

    assert missing_lora_target_modules(FakeModel(), ("q_proj", "k_proj", "v_proj")) == ["k_proj"]
