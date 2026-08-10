"""CPU-only unit tests for tiny_grpo.config. No model/dataset access."""

import dataclasses

import pytest

from tiny_grpo.config import (
    MAX_CHECKPOINTS_TO_KEEP,
    ConfigError,
    DatasetConfig,
    TrainingConfig,
    apply_config_override,
    debug_config,
    longer_config,
    smoke_config,
)
from tiny_grpo.hardware import CUDA_4GB, MPS_16GB
from tiny_grpo.model_profiles import QWEN3_0_6B

PROFILE_FACTORIES = [smoke_config, debug_config, longer_config]
HARDWARE_PROFILES = [MPS_16GB, CUDA_4GB]


def test_max_checkpoints_constant_is_two():
    # Guards against silently loosening the hard cap the spec requires.
    assert MAX_CHECKPOINTS_TO_KEEP == 2


@pytest.mark.parametrize("factory", PROFILE_FACTORIES)
@pytest.mark.parametrize("hardware", HARDWARE_PROFILES)
def test_profiles_construct_and_validate_on_both_hardware_profiles(factory, hardware):
    config = factory(hardware)
    assert isinstance(config, TrainingConfig)
    assert config.checkpoint_retention <= MAX_CHECKPOINTS_TO_KEEP
    assert config.hardware_profile_name == hardware.name


@pytest.mark.parametrize("factory", PROFILE_FACTORIES)
@pytest.mark.parametrize("hardware", HARDWARE_PROFILES)
def test_profiles_keep_batch_size_multiple_of_num_generations(factory, hardware):
    config = factory(hardware)
    assert config.per_device_train_batch_size % config.num_generations == 0


def test_profile_picks_up_hardware_specific_values():
    mps_config = smoke_config(MPS_16GB)
    cuda_config = smoke_config(CUDA_4GB)

    assert mps_config.precision == "fp32"
    assert mps_config.gradient_checkpointing is False
    assert cuda_config.precision == "bf16"
    assert cuda_config.gradient_checkpointing is True
    # 0.04, not 0.0: under LoRA, TRL never loads a second full model for the KL
    # reference regardless of beta, so the memory-cost rationale for
    # defaulting to 0 here no longer applies (see tiny_grpo/hardware.py).
    assert cuda_config.beta == 0.04


def test_qwen_model_profile_is_explicit_and_suffixes_run_name():
    config = smoke_config(CUDA_4GB, model_profile=QWEN3_0_6B)
    assert config.run_name == "smoke_qwen3_0_6b"
    assert config.model_id == "Qwen/Qwen3-0.6B"
    assert config.chat_template_kwargs == {"enable_thinking": False}


def test_initial_adapter_path_and_source_are_paired():
    with pytest.raises(ConfigError, match="must be set together"):
        dataclasses.replace(smoke_config(CUDA_4GB), initial_adapter_path="adapter")


def test_training_manifest_path_and_source_are_paired():
    with pytest.raises(ConfigError, match="training_manifest_path"):
        dataclasses.replace(smoke_config(CUDA_4GB), training_manifest_path="ids.json")


def test_initial_adapter_cannot_be_combined_with_checkpoint_resume():
    base = smoke_config(CUDA_4GB)
    with pytest.raises(ConfigError, match="cannot be combined"):
        dataclasses.replace(
            base,
            initial_adapter_path="adapter",
            initial_adapter_source="sft-run",
            resume=dataclasses.replace(base.resume, mode="latest"),
        )


def test_model_id_cannot_drift_from_selected_profile():
    with pytest.raises(ConfigError, match="model_id does not match"):
        dataclasses.replace(smoke_config(CUDA_4GB, model_profile=QWEN3_0_6B), model_id="wrong")


def test_smoke_caps_max_completion_length_below_hardware_default_on_mps():
    # mps_16gb's own default is 256 (SPEC_MACOS_MPS.md), but smoke deliberately
    # overrides it down for speed — training-quality profiles (debug/longer)
    # keep the hardware profile's actual value.
    assert MPS_16GB.max_completion_length == 256
    assert smoke_config(MPS_16GB).max_completion_length == 128
    assert debug_config(MPS_16GB).max_completion_length == 256
    assert longer_config(MPS_16GB).max_completion_length == 256


def test_smoke_leaves_max_completion_length_alone_when_hardware_default_is_smaller():
    # cuda_4gb's own default (128) is already <= smoke's cap, so no change.
    assert CUDA_4GB.max_completion_length == 128
    assert smoke_config(CUDA_4GB).max_completion_length == 128


def test_smoke_max_completion_length_still_explicitly_overridable():
    config = smoke_config(MPS_16GB, max_completion_length=64)
    assert config.max_completion_length == 64


def test_profile_overrides():
    config = smoke_config(MPS_16GB, output_dir="custom_outputs", max_steps=3)
    assert config.output_dir == "custom_outputs"
    assert config.max_steps == 3
    # Unrelated fields keep their profile defaults.
    assert config.num_generations == 4


def test_sampling_defaults_are_explicit_and_profile_independent():
    for hardware in HARDWARE_PROFILES:
        config = smoke_config(hardware)
        assert config.temperature == 1.0
        assert config.top_p == 1.0
        assert config.top_k == 0


def test_invalid_precision_raises():
    with pytest.raises(ConfigError):
        TrainingConfig(run_name="x", hardware_profile_name=MPS_16GB.name, precision="int8")


def test_unknown_hardware_profile_name_raises():
    with pytest.raises(ConfigError):
        TrainingConfig(run_name="x", hardware_profile_name="rocm_8gb")


def test_negative_beta_raises():
    with pytest.raises(ConfigError):
        TrainingConfig(run_name="x", hardware_profile_name=MPS_16GB.name, beta=-0.1)


def test_beta_zero_is_valid_no_reference_model_mode():
    config = TrainingConfig(run_name="x", hardware_profile_name=MPS_16GB.name, beta=0.0)
    assert config.beta == 0.0


def test_batch_size_not_multiple_of_num_generations_raises():
    with pytest.raises(ConfigError):
        TrainingConfig(
            run_name="x", hardware_profile_name=MPS_16GB.name, num_generations=3, per_device_train_batch_size=4
        )


def test_num_generations_below_two_raises():
    # trl's own GRPOConfig hard-requires >= 2 — a single generation leaves
    # nothing to compare within a group, so no advantage can be computed.
    with pytest.raises(ConfigError):
        TrainingConfig(run_name="x", hardware_profile_name=MPS_16GB.name, num_generations=1)


def test_num_generations_two_is_valid():
    config = TrainingConfig(
        run_name="x", hardware_profile_name=MPS_16GB.name, num_generations=2, per_device_train_batch_size=2
    )
    assert config.num_generations == 2


def test_batch_divisibility_rejects_configs_that_would_pass_a_naive_grad_accum_check():
    # A config where per_device_train_batch_size * gradient_accumulation_steps
    # IS a multiple of num_generations (4), but per_device_train_batch_size
    # alone is NOT (2 % 4 != 0). trl's real train-side constraint is on the
    # product (would pass), but eval always reuses per_device_train_batch_size
    # directly with no grad-accumulation multiplier, so this must still be
    # rejected — locks in the reasoning in tiny_grpo/config.py's validate().
    with pytest.raises(ConfigError):
        TrainingConfig(
            run_name="x",
            hardware_profile_name=MPS_16GB.name,
            num_generations=4,
            per_device_train_batch_size=2,
            gradient_accumulation_steps=2,
        )


def test_cuda_profile_without_gradient_checkpointing_raises():
    # SPEC_CUDA_4GB.md: gradient checkpointing is a required default on cuda_4gb,
    # not optional — this is the "obviously inconsistent" combo the spec calls out.
    with pytest.raises(ConfigError):
        TrainingConfig(run_name="x", hardware_profile_name=CUDA_4GB.name, gradient_checkpointing=False)


def test_cuda_profile_with_gradient_checkpointing_is_valid():
    config = TrainingConfig(run_name="x", hardware_profile_name=CUDA_4GB.name, gradient_checkpointing=True)
    assert config.gradient_checkpointing is True


def test_mps_profile_without_gradient_checkpointing_is_valid():
    # The cuda_4gb-specific requirement must not leak onto mps_16gb.
    config = TrainingConfig(run_name="x", hardware_profile_name=MPS_16GB.name, gradient_checkpointing=False)
    assert config.gradient_checkpointing is False


def test_checkpoint_retention_above_cap_raises():
    with pytest.raises(ConfigError):
        TrainingConfig(
            run_name="x", hardware_profile_name=MPS_16GB.name, checkpoint_retention=MAX_CHECKPOINTS_TO_KEEP + 1
        )


def test_checkpoint_retention_at_cap_is_valid():
    config = TrainingConfig(
        run_name="x", hardware_profile_name=MPS_16GB.name, checkpoint_retention=MAX_CHECKPOINTS_TO_KEEP
    )
    assert config.checkpoint_retention == MAX_CHECKPOINTS_TO_KEEP


def test_checkpoint_retention_below_one_raises():
    with pytest.raises(ConfigError):
        TrainingConfig(run_name="x", hardware_profile_name=MPS_16GB.name, checkpoint_retention=0)


def test_zero_max_steps_raises():
    with pytest.raises(ConfigError):
        TrainingConfig(run_name="x", hardware_profile_name=MPS_16GB.name, max_steps=0)


def test_nonpositive_learning_rate_raises():
    with pytest.raises(ConfigError):
        TrainingConfig(run_name="x", hardware_profile_name=MPS_16GB.name, learning_rate=0.0)


def test_learning_rate_scheduler_is_explicit_and_linear_by_default():
    assert smoke_config(MPS_16GB).lr_scheduler_type == "linear"


def test_constant_learning_rate_scheduler_is_supported():
    config = debug_config(MPS_16GB, learning_rate=5e-6, lr_scheduler_type="constant")
    assert config.learning_rate == 5e-6
    assert config.lr_scheduler_type == "constant"


def test_nested_dataset_override_preserves_other_debug_settings():
    config = debug_config(CUDA_4GB)

    updated = apply_config_override(config, "dataset.train_size", 512)

    assert updated.dataset.train_size == 512
    assert config.dataset.train_size == 256
    assert updated.max_steps == config.max_steps == 50
    assert updated.lr_scheduler_type == config.lr_scheduler_type == "linear"


def test_override_rejects_path_deeper_than_one_nested_field():
    config = debug_config(CUDA_4GB)

    with pytest.raises(ValueError, match="at most one dot"):
        apply_config_override(config, "dataset.extra.train_size", 512)


def test_unknown_learning_rate_scheduler_raises():
    with pytest.raises(ConfigError, match="unsupported lr_scheduler_type"):
        TrainingConfig(run_name="x", hardware_profile_name=MPS_16GB.name, lr_scheduler_type="cosine")


@pytest.mark.parametrize(
    "overrides",
    [
        {"temperature": 0.0},
        {"top_p": 0.0},
        {"top_p": 1.1},
        {"top_k": -1},
    ],
)
def test_invalid_sampling_settings_raise(overrides):
    with pytest.raises(ConfigError):
        TrainingConfig(run_name="x", hardware_profile_name=MPS_16GB.name, **overrides)


@pytest.mark.parametrize("field_name", ["train_size", "val_size", "test_size"])
def test_zero_dataset_size_raises(field_name):
    with pytest.raises(ConfigError):
        TrainingConfig(
            run_name="x", hardware_profile_name=MPS_16GB.name, dataset=DatasetConfig(**{field_name: 0})
        )


def test_config_is_frozen():
    config = smoke_config(MPS_16GB)
    with pytest.raises(dataclasses.FrozenInstanceError):
        config.max_steps = 999


def test_longer_profile_never_auto_selected_by_default():
    # smoke_config/debug_config/longer_config are plain functions the caller must
    # invoke explicitly with a hardware profile — there's no default/auto
    # entrypoint that picks "longer" or a default device.
    assert smoke_config(MPS_16GB).run_name == "smoke"
    assert debug_config(MPS_16GB).run_name == "debug"
    assert longer_config(MPS_16GB).run_name == "longer"
