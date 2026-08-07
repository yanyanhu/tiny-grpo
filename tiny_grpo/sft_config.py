"""Typed SFT run profiles composed with the existing hardware profiles."""

import dataclasses
import math

from tiny_grpo.config import DatasetConfig, LoraConfig, MAX_CHECKPOINTS_TO_KEEP, ResumeConfig
from tiny_grpo.hardware import CUDA_4GB, HARDWARE_PROFILES, MPS_16GB, HardwareProfile, Precision
from tiny_grpo.model_profiles import DEFAULT_MODEL_PROFILE_NAME, MODEL_PROFILES, SMOLLM2_135M, ModelProfile


class SFTConfigError(ValueError):
    """Raised when an SFT configuration is internally inconsistent."""


@dataclasses.dataclass(frozen=True)
class SFTHardwareDefaults:
    per_device_train_batch_size: int
    gradient_accumulation_steps: int
    max_sequence_length: int


SFT_HARDWARE_DEFAULTS = {
    CUDA_4GB.name: SFTHardwareDefaults(
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        max_sequence_length=1024,
    ),
    MPS_16GB.name: SFTHardwareDefaults(
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        max_sequence_length=1024,
    ),
}


@dataclasses.dataclass(frozen=True)
class SFTTrainingConfig:
    run_name: str
    hardware_profile_name: str
    seed: int = 42
    model_profile_name: str = DEFAULT_MODEL_PROFILE_NAME
    model_id: str = SMOLLM2_135M.model_id
    chat_template_kwargs: dict = dataclasses.field(default_factory=dict)
    precision: Precision = "fp32"
    device: str = "mps"
    env_setup: dict = dataclasses.field(default_factory=dict)
    lora: LoraConfig = dataclasses.field(default_factory=LoraConfig)
    dataset: DatasetConfig = dataclasses.field(default_factory=DatasetConfig)
    max_sequence_length: int = 1024
    max_completion_length: int = 128
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = 0
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 1
    gradient_checkpointing: bool = False
    learning_rate: float = 2e-4
    max_steps: int = 3
    logging_steps: int = 1
    checkpoint_steps: int = 2
    checkpoint_retention: int = MAX_CHECKPOINTS_TO_KEEP
    eval_steps: int = 2
    output_dir: str = "outputs"
    completion_only_loss: bool = True
    resume: ResumeConfig = dataclasses.field(default_factory=ResumeConfig)

    def __post_init__(self) -> None:
        validate_sft_config(self)


def validate_sft_config(config: SFTTrainingConfig) -> None:
    if config.hardware_profile_name not in HARDWARE_PROFILES:
        raise SFTConfigError(f"unknown hardware profile {config.hardware_profile_name!r}")
    if config.hardware_profile_name not in SFT_HARDWARE_DEFAULTS:
        raise SFTConfigError(f"no SFT defaults for hardware profile {config.hardware_profile_name!r}")
    if config.model_profile_name not in MODEL_PROFILES:
        raise SFTConfigError(f"unknown model profile {config.model_profile_name!r}")
    model_profile = MODEL_PROFILES[config.model_profile_name]
    if config.model_id != model_profile.model_id:
        raise SFTConfigError("model_id does not match the selected model profile")
    if config.chat_template_kwargs != model_profile.chat_template_kwargs:
        raise SFTConfigError("chat_template_kwargs do not match the selected model profile")
    hardware = HARDWARE_PROFILES[config.hardware_profile_name]
    if config.device != hardware.device:
        raise SFTConfigError(
            f"device {config.device!r} does not match hardware profile "
            f"{config.hardware_profile_name!r} (expected {hardware.device!r})"
        )
    if config.precision not in ("fp32", "bf16", "fp16"):
        raise SFTConfigError(f"unsupported precision {config.precision!r}")
    if config.max_sequence_length < 1:
        raise SFTConfigError("max_sequence_length must be >= 1")
    if config.max_completion_length < 1:
        raise SFTConfigError("max_completion_length must be >= 1")
    if config.per_device_train_batch_size < 1:
        raise SFTConfigError("per_device_train_batch_size must be >= 1")
    if config.gradient_accumulation_steps < 1:
        raise SFTConfigError("gradient_accumulation_steps must be >= 1")
    if config.learning_rate <= 0:
        raise SFTConfigError("learning_rate must be > 0")
    if config.max_steps < 1:
        raise SFTConfigError("max_steps must be >= 1")
    if config.logging_steps < 1:
        raise SFTConfigError("logging_steps must be >= 1")
    if config.checkpoint_steps < 1:
        raise SFTConfigError("checkpoint_steps must be >= 1")
    if config.eval_steps < 1:
        raise SFTConfigError("eval_steps must be >= 1")
    if not config.completion_only_loss:
        raise SFTConfigError("SFT requires completion_only_loss=True so prompt tokens are masked")
    if config.checkpoint_retention < 1 or config.checkpoint_retention > MAX_CHECKPOINTS_TO_KEEP:
        raise SFTConfigError(f"checkpoint_retention must be between 1 and {MAX_CHECKPOINTS_TO_KEEP}")
    if config.hardware_profile_name == CUDA_4GB.name and not config.gradient_checkpointing:
        raise SFTConfigError("cuda_4gb SFT requires gradient_checkpointing=True")


def _compose_sft(
    run_name: str,
    dataset: DatasetConfig,
    run_shape: dict,
    hardware: HardwareProfile,
    model_profile: ModelProfile = SMOLLM2_135M,
    **overrides,
) -> SFTTrainingConfig:
    defaults = SFT_HARDWARE_DEFAULTS[hardware.name]
    fields = dict(
        run_name=model_profile.run_name(run_name),
        hardware_profile_name=hardware.name,
        model_profile_name=model_profile.name,
        model_id=model_profile.model_id,
        chat_template_kwargs=model_profile.chat_template_kwargs,
        lora=LoraConfig(target_modules=model_profile.lora_target_modules),
        precision=hardware.precision,
        device=hardware.device,
        env_setup=dict(hardware.env_setup),
        dataset=dataset,
        max_sequence_length=defaults.max_sequence_length,
        max_completion_length=min(hardware.max_completion_length, 128),
        per_device_train_batch_size=defaults.per_device_train_batch_size,
        gradient_accumulation_steps=defaults.gradient_accumulation_steps,
        gradient_checkpointing=hardware.gradient_checkpointing,
        **run_shape,
    )
    fields.update(overrides)
    return SFTTrainingConfig(**fields)


def sft_smoke_config(hardware: HardwareProfile, model_profile: ModelProfile = SMOLLM2_135M, **overrides) -> SFTTrainingConfig:
    return _compose_sft(
        run_name="sft_smoke",
        dataset=DatasetConfig(split_seed=42, train_size=64, val_size=16, test_size=32),
        run_shape=dict(max_steps=3, checkpoint_steps=2, eval_steps=2, logging_steps=1),
        hardware=hardware,
        model_profile=model_profile,
        **overrides,
    )


def sft_debug_config(hardware: HardwareProfile, model_profile: ModelProfile = SMOLLM2_135M, **overrides) -> SFTTrainingConfig:
    # With batch=1 and accumulation=8 on cuda_4gb, 32 optimizer steps consume
    # approximately one epoch over the 256-example debug subset.
    return _compose_sft(
        run_name="sft_debug",
        dataset=DatasetConfig(split_seed=42, train_size=256, val_size=32, test_size=64),
        run_shape=dict(max_steps=32, checkpoint_steps=8, eval_steps=8, logging_steps=1),
        hardware=hardware,
        model_profile=model_profile,
        **overrides,
    )


def sft_stronger_config(hardware: HardwareProfile, model_profile: ModelProfile = SMOLLM2_135M, **overrides) -> SFTTrainingConfig:
    """Two effective epochs over the full 1,024-example reserved train set.

    Compute optimizer steps from the hardware-specific effective batch so the
    amount of supervised exposure is the same on CUDA and MPS.
    """
    dataset = DatasetConfig(split_seed=42, train_size=1024, val_size=64, test_size=128)
    defaults = SFT_HARDWARE_DEFAULTS[hardware.name]
    updates_per_epoch = math.ceil(
        dataset.train_size
        / (defaults.per_device_train_batch_size * defaults.gradient_accumulation_steps)
    )
    return _compose_sft(
        run_name="sft_stronger",
        dataset=dataset,
        run_shape=dict(
            max_steps=2 * updates_per_epoch,
            checkpoint_steps=max(1, updates_per_epoch // 2),
            eval_steps=max(1, updates_per_epoch // 2),
            logging_steps=1,
        ),
        hardware=hardware,
        model_profile=model_profile,
        **overrides,
    )
