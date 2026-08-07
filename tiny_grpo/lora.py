"""Conversion from tiny_grpo's own LoraConfig to peft's.

A single pure function — `peft` is imported lazily inside it so that
tiny_grpo.config stays free of a peft dependency (same pattern as
tiny_grpo/rewards.py staying free of trl), and this stays testable without
loading a model.
"""

from tiny_grpo.config import LoraConfig


def missing_lora_target_modules(model, target_modules: tuple[str, ...]) -> list[str]:
    """Return configured target suffixes absent from the loaded model."""
    module_suffixes = {name.rsplit(".", 1)[-1] for name, _ in model.named_modules()}
    return sorted(set(target_modules) - module_suffixes)


def to_peft_lora_config(lora: LoraConfig):
    from peft import LoraConfig as PeftLoraConfig
    from peft import TaskType

    return PeftLoraConfig(
        r=lora.r,
        lora_alpha=lora.alpha,
        lora_dropout=lora.dropout,
        target_modules=list(lora.target_modules),
        task_type=TaskType.CAUSAL_LM,
    )
