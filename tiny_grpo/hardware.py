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


class PrecisionUnsupportedError(RuntimeError):
    """Raised when the active device doesn't actually support the configured precision."""


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
    # SPEC_CUDA_4GB.md originally defaulted this to 0.0 to avoid loading a
    # second full model copy for the KL reference. With LoRA (see
    # tiny_grpo/lora.py), TRL never loads a second full model regardless of
    # beta — it disables/clones a tiny adapter instead (confirmed against
    # trl/trainer/grpo_trainer.py) — so that rationale no longer applies.
    # Aligned with MPS_16GB's value now that both profiles carry the same
    # (negligible) reference-computation memory cost under LoRA.
    beta=0.04,
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


def resolve_dtype(precision: Precision):
    """Map a precision string to the torch dtype to load model weights in.

    None means "let the loader use its own default" (float32, confirmed
    against the installed trl's GRPOTrainer docstring). GRPOConfig(bf16=True)/
    (fp16=True) only control mixed-precision autocast during training, not the
    stored weight dtype — this is what actually makes bf16/fp16 profiles load
    bf16/fp16 weights.
    """
    import torch

    return {"fp32": None, "bf16": torch.bfloat16, "fp16": torch.float16}[precision]


def verify_precision_supported(device: Device, precision: Precision, *, bf16_supported: bool | None = None) -> None:
    """Fail loudly if `precision` isn't actually supported on `device` — never
    silently substitute a different precision (same principle as this
    project's OOM-handling stance: hiding what actually happened breaks
    reproducibility).

    Only checks combinations with a reliable hardware-capability query:
    - cuda + bf16: `torch.cuda.is_bf16_supported(including_emulation=False)` —
      real Ampere+ tensor-core support, not degraded software emulation.
    - fp16 on CUDA is supported on effectively every CUDA GPU; not checked.
    - mps has no equivalent static capability query in PyTorch, so precision
      compatibility there is verified empirically instead, via
      tests/test_mps_integration.py — not gated here.
    Every other (device, precision) combination is a no-op.

    `bf16_supported` overrides the real `torch.cuda.is_bf16_supported()` query
    — pass it explicitly in tests to check this logic without needing a real
    CUDA device; omit it in production to query torch directly.
    """
    if device != "cuda" or precision != "bf16":
        return

    if bf16_supported is None:
        import torch

        bf16_supported = torch.cuda.is_available() and torch.cuda.is_bf16_supported(including_emulation=False)

    if not bf16_supported:
        raise PrecisionUnsupportedError(
            "cuda_4gb defaults to bf16, but this GPU does not report real bf16 tensor-core "
            "support (torch.cuda.is_bf16_supported(including_emulation=False) is False; "
            "requires Ampere or newer, compute capability >= 8.0). Override precision "
            "explicitly (e.g. to fp16) rather than assuming the default works here."
        )
