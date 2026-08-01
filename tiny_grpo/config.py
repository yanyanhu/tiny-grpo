"""Typed training configuration: fields + validation + smoke/debug/longer profiles.

A concrete run is a *run profile* (smoke/debug/longer — training length, cadence)
composed with a *hardware profile* (device/precision/batch size/gradient
checkpointing/beta — see tiny_grpo/hardware.py). Run-profile factories below
require an explicit HardwareProfile; there is no implicit default device.

Pure dataclasses and validation logic only — no model/dataset access — so
profiles and validation rules are unit testable on their own.
"""

import dataclasses
from typing import Literal

from tiny_grpo.hardware import CUDA_4GB, HARDWARE_PROFILES, HardwareProfile, Precision

# Hard cap, not a default to raise casually: checkpoints store optimizer/scheduler
# state on top of the adapter, so they're disk-heavy even at this model scale, and
# neither supported hardware profile should be assumed to have much disk headroom.
MAX_CHECKPOINTS_TO_KEEP = 2


class ConfigError(ValueError):
    """Raised when a TrainingConfig fails validation."""


@dataclasses.dataclass(frozen=True)
class LoraConfig:
    r: int = 8
    alpha: int = 16
    dropout: float = 0.05
    target_modules: tuple[str, ...] = ("q_proj", "k_proj", "v_proj", "o_proj")


@dataclasses.dataclass(frozen=True)
class DatasetConfig:
    split_seed: int = 42
    train_size: int = 64
    val_size: int = 16
    test_size: int = 32


@dataclasses.dataclass(frozen=True)
class ResumeConfig:
    # "latest": resume from the most recent checkpoint in output_dir if one exists.
    # "none": always start fresh, even if checkpoints are present.
    # Any other string: an explicit checkpoint path to resume from.
    mode: Literal["latest", "none"] | str = "latest"


@dataclasses.dataclass(frozen=True)
class TrainingConfig:
    run_name: str
    hardware_profile_name: str
    seed: int = 42
    model_id: str = "HuggingFaceTB/SmolLM2-135M-Instruct"
    precision: Precision = "fp32"
    device: str = "mps"
    # Environment variables applied before training (MPS fallback / CUDA alloc
    # config), copied from the hardware profile at construction time.
    env_setup: dict = dataclasses.field(default_factory=dict)
    lora: LoraConfig = dataclasses.field(default_factory=LoraConfig)
    # KL coefficient. 0.0 is an explicit memory-saving mode: GRPOTrainer skips
    # loading a separate reference model, at the cost of no KL constraint
    # against drift from the base policy. Hardware-profile-specific default.
    beta: float = 0.04
    dataset: DatasetConfig = dataclasses.field(default_factory=DatasetConfig)
    num_generations: int = 4
    max_prompt_length: int = 256
    max_completion_length: int = 128
    per_device_train_batch_size: int = 4
    gradient_accumulation_steps: int = 1
    gradient_checkpointing: bool = False
    learning_rate: float = 1e-5
    max_steps: int = 10
    logging_steps: int = 1
    checkpoint_steps: int = 5
    checkpoint_retention: int = MAX_CHECKPOINTS_TO_KEEP
    eval_steps: int = 5
    output_dir: str = "outputs"
    resume: ResumeConfig = dataclasses.field(default_factory=ResumeConfig)

    def __post_init__(self) -> None:
        validate(self)


def validate(config: TrainingConfig) -> None:
    if config.hardware_profile_name not in HARDWARE_PROFILES:
        raise ConfigError(
            f"unknown hardware_profile_name {config.hardware_profile_name!r}; "
            f"expected one of {sorted(HARDWARE_PROFILES)}"
        )

    if config.precision not in ("fp32", "bf16", "fp16"):
        raise ConfigError(f"unsupported precision: {config.precision!r}")

    if config.beta < 0:
        raise ConfigError(f"beta must be >= 0, got {config.beta}")

    # trl's GRPOConfig hard-requires >= 2 (raises ValueError otherwise): with a
    # single generation there's nothing to compare within a group, so no
    # advantage can be computed. Enforced here too so this fails at config
    # construction, not deep inside GRPOTrainer.
    if config.num_generations < 2:
        raise ConfigError(f"num_generations must be >= 2 (GRPO needs a group to compare), got {config.num_generations}")

    # Note: per_device_train_batch_size counts completions (rows, post prompt-
    # repeat-expansion), not unique prompts — trl's own
    # generation_batch_size = per_device_train_batch_size * num_processes * steps_per_generation
    # has no separate "* num_generations" factor, which only holds together
    # dimensionally if per_device_train_batch_size is already in the same
    # (completions) units as generation_batch_size. Confirmed directly against
    # the RepeatSampler diagram in trl/trainer/grpo_trainer.py's
    # _get_train_sampler: per_device_train_batch_size=3 there means 3
    # completion-rows per device, not 3 prompts (with num_processes > 1, a
    # single prompt's repeats can even be split across devices — gathered back
    # together for reward normalization; irrelevant here since this project
    # never uses >1 process).
    #
    # trl's real constraint on the *train* side is on its own derived
    # generation_batch_size = per_device_train_batch_size * gradient_accumulation_steps
    # (this project never sets generation_batch_size/steps_per_generation
    # explicitly, and never uses >1 process, so this is exactly how trl
    # computes it) — that product must be a multiple of num_generations.
    # Separately, trl's *eval* side requires per_device_eval_batch_size %
    # num_generations == 0, with no grad-accumulation multiplier. Since this
    # project always sets per_device_eval_batch_size equal to
    # per_device_train_batch_size (see train_grpo.py's build_grpo_config —
    # there's no independent eval batch size field), checking
    # per_device_train_batch_size alone is the real necessary condition (for
    # eval) and is automatically sufficient for train too: if
    # per_device_train_batch_size already divides evenly, any multiple of it
    # via gradient_accumulation_steps does too. Do not "simplify" this to just
    # checking generation_batch_size — that would accept configs whose eval
    # batch size doesn't divide evenly (e.g. batch=2, grad_accum=2,
    # num_generations=4: generation_batch_size=4 passes, but eval reuses
    # per_device_train_batch_size=2, which does not).
    if config.per_device_train_batch_size % config.num_generations != 0:
        raise ConfigError(
            f"per_device_train_batch_size ({config.per_device_train_batch_size}) must be a "
            f"multiple of num_generations ({config.num_generations}) so every device holds "
            "complete prompt groups (required for both training and eval batches, since "
            "eval reuses this same value)"
        )

    # SPEC_CUDA_4GB.md: gradient checkpointing is a required default on this
    # profile (not optional) — 4GB dedicated VRAM generally can't fit
    # generation + backward pass without it. Catches the "obviously
    # inconsistent" run+hardware combination the spec calls out.
    if config.hardware_profile_name == CUDA_4GB.name and not config.gradient_checkpointing:
        raise ConfigError(
            f"hardware profile {CUDA_4GB.name!r} requires gradient_checkpointing=True "
            "(4GB VRAM budget); got False"
        )

    if config.checkpoint_retention < 1:
        raise ConfigError(f"checkpoint_retention must be >= 1, got {config.checkpoint_retention}")

    if config.checkpoint_retention > MAX_CHECKPOINTS_TO_KEEP:
        raise ConfigError(
            f"checkpoint_retention ({config.checkpoint_retention}) exceeds the hard cap of "
            f"{MAX_CHECKPOINTS_TO_KEEP}. Raise MAX_CHECKPOINTS_TO_KEEP explicitly, with a "
            "documented reason, if more retained checkpoints are truly needed."
        )

    if config.max_steps < 1:
        raise ConfigError(f"max_steps must be >= 1, got {config.max_steps}")

    if config.learning_rate <= 0:
        raise ConfigError(f"learning_rate must be > 0, got {config.learning_rate}")

    for field_name in ("train_size", "val_size", "test_size"):
        size = getattr(config.dataset, field_name)
        if size < 1:
            raise ConfigError(f"dataset.{field_name} must be >= 1, got {size}")


def _compose(run_name: str, dataset: DatasetConfig, run_shape: dict, hardware: HardwareProfile,
             profile_defaults: dict | None = None, **overrides) -> TrainingConfig:
    """`run_shape` carries the run-profile-specific fields (max_steps, checkpoint_steps,
    eval_steps, logging_steps) as a dict, not individual keyword params, so that
    `**overrides` (arbitrary caller keyword overrides) can never collide with them.

    Precedence, lowest to highest: hardware profile defaults -> `profile_defaults`
    (a run profile's own deliberate deviation, e.g. smoke's shorter completion
    length) -> `**overrides` (explicit caller overrides, always win).
    """
    fields = dict(
        run_name=run_name,
        hardware_profile_name=hardware.name,
        precision=hardware.precision,
        device=hardware.device,
        env_setup=dict(hardware.env_setup),
        beta=hardware.beta,
        dataset=dataset,
        num_generations=4,  # preferred on both profiles; override explicitly to fall back to 2
        max_prompt_length=hardware.max_prompt_length,
        max_completion_length=hardware.max_completion_length,
        per_device_train_batch_size=hardware.per_device_train_batch_size,
        gradient_accumulation_steps=hardware.gradient_accumulation_steps,
        gradient_checkpointing=hardware.gradient_checkpointing,
        **run_shape,
    )
    fields.update(profile_defaults or {})
    fields.update(overrides)
    return TrainingConfig(**fields)


def smoke_config(hardware: HardwareProfile, **overrides) -> TrainingConfig:
    """Fast sanity-check profile: verifies the training loop runs end to end.

    Deliberately caps max_completion_length at 128 regardless of the hardware
    profile's own default — smoke exists to verify the loop runs quickly, not
    to give the model full reasoning room. An empirical run showed the
    mps_16gb profile's 256 default increases eval runtime ~5x (not just ~2x),
    since completions rarely hit an early EOS at that length. debug/longer
    keep the hardware profile's actual value, since training quality matters
    there. Pass max_completion_length= explicitly to override this back up.
    """
    return _compose(
        run_name="smoke",
        dataset=DatasetConfig(split_seed=42, train_size=64, val_size=16, test_size=32),
        run_shape=dict(max_steps=10, checkpoint_steps=5, eval_steps=5, logging_steps=1),
        hardware=hardware,
        profile_defaults=dict(max_completion_length=min(hardware.max_completion_length, 128)),
        **overrides,
    )


def debug_config(hardware: HardwareProfile, **overrides) -> TrainingConfig:
    """Slightly larger profile for iterating on reward/config changes."""
    return _compose(
        run_name="debug",
        dataset=DatasetConfig(split_seed=42, train_size=256, val_size=32, test_size=64),
        run_shape=dict(max_steps=50, checkpoint_steps=10, eval_steps=10, logging_steps=1),
        hardware=hardware,
        **overrides,
    )


def longer_config(hardware: HardwareProfile, **overrides) -> TrainingConfig:
    """Larger profile for an explicit, user-requested experiment.

    Never launched automatically — a multi-hour run must be started deliberately,
    not as routine verification. Keeps num_generations at the shared preferred
    default (4) rather than pre-emptively raising it; increase only after a
    smoke run demonstrates headroom (see hardware profile docs).
    """
    return _compose(
        run_name="longer",
        dataset=DatasetConfig(split_seed=42, train_size=1024, val_size=64, test_size=128),
        run_shape=dict(max_steps=200, checkpoint_steps=25, eval_steps=25, logging_steps=5),
        hardware=hardware,
        **overrides,
    )
