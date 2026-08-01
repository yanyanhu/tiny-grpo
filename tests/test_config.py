"""CPU-only unit tests for tiny_grpo.config. No model/dataset access."""

import dataclasses

import pytest

from tiny_grpo.config import (
    MAX_CHECKPOINTS_TO_KEEP,
    ConfigError,
    DatasetConfig,
    TrainingConfig,
    debug_config,
    longer_config,
    smoke_config,
)
from tiny_grpo.hardware import CUDA_4GB, MPS_16GB

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
