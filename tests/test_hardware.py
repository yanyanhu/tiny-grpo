"""CPU-only unit tests for tiny_grpo.hardware. No real GPU/MPS device required
— device availability is injected as plain data."""

import pytest

from tiny_grpo.hardware import (
    CUDA_4GB,
    MPS_16GB,
    DeviceUnavailableError,
    PrecisionUnsupportedError,
    UnknownHardwareProfileError,
    detect_available_devices,
    resolve_device,
    resolve_dtype,
    resolve_hardware_profile,
    verify_precision_supported,
)


def test_resolve_known_profiles():
    assert resolve_hardware_profile("mps_16gb") is MPS_16GB
    assert resolve_hardware_profile("cuda_4gb") is CUDA_4GB


def test_resolve_unknown_profile_raises():
    with pytest.raises(UnknownHardwareProfileError):
        resolve_hardware_profile("rocm_8gb")


def test_resolve_device_when_available():
    assert resolve_device(MPS_16GB, available_devices={"mps": True, "cuda": False}) == "mps"
    assert resolve_device(CUDA_4GB, available_devices={"mps": False, "cuda": True}) == "cuda"


def test_resolve_device_when_unavailable_raises():
    with pytest.raises(DeviceUnavailableError):
        resolve_device(MPS_16GB, available_devices={"mps": False, "cuda": True})
    with pytest.raises(DeviceUnavailableError):
        resolve_device(CUDA_4GB, available_devices={"mps": True, "cuda": False})


def test_resolve_device_missing_key_treated_as_unavailable():
    with pytest.raises(DeviceUnavailableError):
        resolve_device(CUDA_4GB, available_devices={})


def test_profiles_carry_distinct_precision_and_env_setup():
    assert MPS_16GB.precision == "fp32"
    assert CUDA_4GB.precision == "bf16"
    assert "PYTORCH_ENABLE_MPS_FALLBACK" in MPS_16GB.env_setup
    assert "PYTORCH_CUDA_ALLOC_CONF" in CUDA_4GB.env_setup


def test_cuda_profile_defaults_gradient_checkpointing_on():
    # SPEC_CUDA_4GB.md: required default on this profile, not optional.
    assert CUDA_4GB.gradient_checkpointing is True


def test_mps_profile_defaults_gradient_checkpointing_off():
    assert MPS_16GB.gradient_checkpointing is False


def test_batch_size_is_multiple_of_preferred_num_generations_on_both_profiles():
    # trl requires per-device batch size to be an exact multiple of
    # num_generations for single-process training; preferred num_generations
    # is 4 on both profiles (see PROJECT_SPEC.md).
    preferred_num_generations = 4
    assert MPS_16GB.per_device_train_batch_size % preferred_num_generations == 0
    assert CUDA_4GB.per_device_train_batch_size % preferred_num_generations == 0


def test_resolve_dtype_fp32_means_loader_default():
    assert resolve_dtype("fp32") is None


def test_resolve_dtype_bf16_and_fp16():
    import torch

    assert resolve_dtype("bf16") is torch.bfloat16
    assert resolve_dtype("fp16") is torch.float16


def test_verify_precision_cuda_bf16_supported_is_noop():
    verify_precision_supported("cuda", "bf16", bf16_supported=True)  # must not raise


def test_verify_precision_cuda_bf16_unsupported_raises():
    with pytest.raises(PrecisionUnsupportedError):
        verify_precision_supported("cuda", "bf16", bf16_supported=False)


@pytest.mark.parametrize(
    "device,precision",
    [
        ("cuda", "fp16"),
        ("cuda", "fp32"),
        ("mps", "bf16"),
        ("mps", "fp16"),
        ("mps", "fp32"),
    ],
)
def test_verify_precision_only_checks_cuda_bf16(device, precision):
    # Every other combination is a no-op even if bf16_supported=False is passed
    # in, since it's irrelevant to non-(cuda, bf16) combinations.
    verify_precision_supported(device, precision, bf16_supported=False)  # must not raise


def test_detect_available_devices_shape():
    # Calls the real torch backends — doesn't require any device to actually
    # be available, just that the query returns a well-shaped dict.
    result = detect_available_devices()
    assert set(result) == {"mps", "cuda"}
    assert isinstance(result["mps"], bool)
    assert isinstance(result["cuda"], bool)
