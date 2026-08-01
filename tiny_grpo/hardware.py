"""Hardware profile registry + device resolution.

Resolution logic takes device availability as data (a plain dict), not by
calling torch directly — so it's fully unit-testable without a real GPU/MPS
device. `detect_available_devices()` is the one place that actually queries
torch, used only by the production path (`resolve_device` with no override).
"""

import dataclasses
from typing import Literal

Precision = Literal["fp32", "bf16", "fp16"]
Device = Literal["mps", "cuda"]


class UnknownHardwareProfileError(ValueError):
    """Raised when a hardware profile name isn't in the registry."""


class DeviceUnavailableError(RuntimeError):
    """Raised when a profile's required device isn't available."""


@dataclasses.dataclass(frozen=True)
class HardwareProfile:
    name: str
    device: Device
    precision: Precision
    per_device_train_batch_size: int
    gradient_accumulation_steps: int
    gradient_checkpointing: bool
    beta: float
    max_prompt_length: int
    max_completion_length: int
    # Environment variables to set before training starts (e.g. MPS fallback,
    # CUDA allocator config). Applied by the caller, not by this module, so
    # resolving a profile never has a side effect.
    env_setup: dict = dataclasses.field(default_factory=dict)


# per_device_train_batch_size=4 on both profiles (not the 2 / 1 "starting
# points" the profile docs suggest) because TRL requires per-device batch size
# to be an exact multiple of num_generations for single-process training,
# and num_generations=4 is the shared preferred default (see PROJECT_SPEC.md).
# Effective batch size is instead tuned via gradient_accumulation_steps.
MPS_16GB = HardwareProfile(
    name="mps_16gb",
    device="mps",
    precision="fp32",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=1,
    gradient_checkpointing=False,
    beta=0.04,
    max_prompt_length=256,
    # SPEC_MACOS_MPS.md's suggested default — gives the model room to reason,
    # which matters for actual training quality (debug/longer runs). An
    # empirical smoke run showed this is NOT just a linear cost increase from
    # 128: completions rarely hit an early EOS at 256, so eval runtime went
    # from ~120s to ~645s (~5x) for a 2x length increase. smoke_config()
    # deliberately overrides this down to 128 (see tiny_grpo/config.py) since
    # smoke exists to verify the loop runs quickly, not to give the model full
    # reasoning room — debug/longer keep this hardware-profile default.
    max_completion_length=256,
    env_setup={"PYTORCH_ENABLE_MPS_FALLBACK": "1"},
)

CUDA_4GB = HardwareProfile(
    name="cuda_4gb",
    device="cuda",
    precision="bf16",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=8,
    gradient_checkpointing=True,
    beta=0.0,
    max_prompt_length=128,
    max_completion_length=128,
    env_setup={"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"},
)

HARDWARE_PROFILES = {
    "mps_16gb": MPS_16GB,
    "cuda_4gb": CUDA_4GB,
}


def resolve_hardware_profile(name: str) -> HardwareProfile:
    try:
        return HARDWARE_PROFILES[name]
    except KeyError:
        raise UnknownHardwareProfileError(
            f"unknown hardware profile {name!r}; expected one of {sorted(HARDWARE_PROFILES)}"
        ) from None


def detect_available_devices() -> dict:
    """Query torch for real device availability. Production path only —
    tests should pass an explicit `available_devices` dict instead."""
    import torch

    return {
        "mps": torch.backends.mps.is_available(),
        "cuda": torch.cuda.is_available(),
    }


def resolve_device(profile: HardwareProfile, available_devices: dict | None = None) -> str:
    """Confirm `profile`'s required device is available and return it.

    `available_devices` maps device name -> bool. Pass it explicitly in tests
    to check resolution logic without needing real GPU/MPS hardware; omit it
    in production to query torch directly.
    """
    if available_devices is None:
        available_devices = detect_available_devices()

    if not available_devices.get(profile.device, False):
        raise DeviceUnavailableError(
            f"hardware profile {profile.name!r} requires device {profile.device!r}, "
            f"which is not available (available_devices={available_devices})"
        )
    return profile.device
