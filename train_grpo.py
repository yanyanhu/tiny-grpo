"""GRPO training entrypoint: wires config, dataset splits, and reward logic together.

Usage:
    uv run python train_grpo.py --profile {smoke,debug,longer} --hardware {mps_16gb,cuda_4gb}
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

from datasets import load_dataset
from transformers import TrainerCallback
from trl import GRPOConfig, GRPOTrainer

from tiny_grpo.config import TrainingConfig, debug_config, longer_config, smoke_config
from tiny_grpo.hardware import HARDWARE_PROFILES, resolve_device, resolve_hardware_profile
from tiny_grpo.monitoring import device_memory_mb, process_memory_mb
from tiny_grpo.rewards import accuracy_reward, format_reward, to_prompt
from tiny_grpo.run_context import make_run_dir, save_run_metadata, update_run_status
from tiny_grpo.splits import build_split_metadata, select_split

RUN_PROFILES = {"smoke": smoke_config, "debug": debug_config, "longer": longer_config}


class JsonlLoggerCallback(TrainerCallback):
    """Appends every trainer.log() call, plus memory stats, to a JSONL file.

    This is the detailed source of truth. Nothing should parse console output
    instead of this file.
    """

    def __init__(self, path, device: str):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.device = device

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None:
            return
        record = {"step": state.global_step, **logs, "process_memory_mb": process_memory_mb()}
        device_mem = device_memory_mb(self.device)
        if device_mem is not None:
            record[f"{self.device}_memory_mb"] = device_mem
        with self.path.open("a") as f:
            f.write(json.dumps(record) + "\n")


class ConsoleProgressCallback(TrainerCallback):
    """Throttled, single-line, human-readable progress update at logging_steps.

    A derived view for live monitoring only — not a logging path, and nothing
    downstream should parse it. Falls back to plain newline-terminated lines
    when stdout isn't a tty (piped output, gtimeout, CI).
    """

    def __init__(self, device: str):
        self._start_time = None
        self._is_tty = sys.stdout.isatty()
        self.device = device

    def on_train_begin(self, args, state, control, **kwargs):
        self._start_time = time.monotonic()

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None or self._start_time is None:
            return
        # Eval logs carry "eval_"-prefixed keys; keep this line about live
        # training progress and let eval metrics live in the JSONL only.
        if any(key.startswith("eval_") for key in logs):
            return

        elapsed = time.monotonic() - self._start_time
        step = state.global_step
        max_steps = state.max_steps or 0
        eta = (elapsed / step) * (max_steps - step) if 0 < step < max_steps else 0.0

        loss = logs.get("loss", float("nan"))
        reward = logs.get("reward", float("nan"))
        accuracy = logs.get("rewards/accuracy_reward/mean", float("nan"))
        fmt = logs.get("rewards/format_reward/mean", float("nan"))
        mem_mb = process_memory_mb()
        device_mem = device_memory_mb(self.device)
        device_suffix = f" {self.device} {device_mem['allocated_mb']:.0f}MB" if device_mem else ""

        line = (
            f"step {step}/{max_steps} | elapsed {elapsed:.0f}s | eta {eta:.0f}s | "
            f"loss {loss:.4f} | reward {reward:.3f} (acc {accuracy:.3f} fmt {fmt:.3f}) | "
            f"mem {mem_mb:.0f}MB{device_suffix}"
        )

        if self._is_tty:
            sys.stdout.write("\r" + line.ljust(120))
            sys.stdout.flush()
        else:
            print(line, flush=True)


def build_datasets(config: TrainingConfig):
    train_pool = load_dataset("openai/gsm8k", "main", split="train")
    test_pool = load_dataset("openai/gsm8k", "main", split="test")

    split_metadata = build_split_metadata(
        train_pool_size=len(train_pool),
        test_pool_size=len(test_pool),
        train_size=config.dataset.train_size,
        val_size=config.dataset.val_size,
        test_size=config.dataset.test_size,
        seed=config.dataset.split_seed,
    )

    train_dataset = select_split(train_pool, split_metadata.train_indices).map(
        to_prompt, remove_columns=train_pool.column_names
    )
    val_dataset = select_split(train_pool, split_metadata.val_indices).map(
        to_prompt, remove_columns=train_pool.column_names
    )
    return train_dataset, val_dataset, split_metadata


def build_grpo_config(config: TrainingConfig, run_dir: Path) -> GRPOConfig:
    return GRPOConfig(
        output_dir=str(run_dir),
        seed=config.seed,
        per_device_train_batch_size=config.per_device_train_batch_size,
        num_generations=config.num_generations,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        gradient_checkpointing=config.gradient_checkpointing,
        max_steps=config.max_steps,
        # max_prompt_length has no equivalent field in the installed trl version
        # (1.9.2) — GRPOConfig dropped/never had it. Left out rather than passed
        # silently; config.max_prompt_length stays recorded for documentation and
        # for whenever a trl version that supports it is adopted.
        max_completion_length=config.max_completion_length,
        beta=config.beta,
        learning_rate=config.learning_rate,
        logging_steps=config.logging_steps,
        disable_tqdm=True,  # ConsoleProgressCallback is the single progress line instead.
        report_to=["tensorboard"],
        save_strategy="steps",
        save_steps=config.checkpoint_steps,
        save_total_limit=config.checkpoint_retention,
        eval_strategy="steps",
        eval_steps=config.eval_steps,
        # per_device_eval_batch_size must be a multiple of num_generations (trl requirement).
        per_device_eval_batch_size=config.per_device_train_batch_size,
        log_completions=True,
        num_completions_to_print=4,
        bf16=config.precision == "bf16",
        fp16=config.precision == "fp16",
        # device_map="auto" (trl's default when loading a model by name) hangs on MPS;
        # force a plain single-device load instead.
        model_init_kwargs={"device_map": None},
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=sorted(RUN_PROFILES), default="smoke")
    parser.add_argument("--hardware", choices=sorted(HARDWARE_PROFILES), required=True)
    parser.add_argument(
        "--verification-run",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Tag this run as verification-only (auto-deletion-eligible via tiny_grpo.cleanup). "
        "Defaults to True for --profile smoke, False otherwise.",
    )
    args = parser.parse_args()

    hardware = resolve_hardware_profile(args.hardware)
    device = resolve_device(hardware)  # raises DeviceUnavailableError if this machine lacks it
    verification_run = args.verification_run if args.verification_run is not None else (args.profile == "smoke")

    config = RUN_PROFILES[args.profile](hardware)

    for env_name, env_value in config.env_setup.items():
        os.environ[env_name] = env_value

    run_dir = make_run_dir(config.output_dir, config.run_name)
    print(f"Run directory: {run_dir}  (hardware={hardware.name}, device={device}, verification_run={verification_run})")

    train_dataset, val_dataset, split_metadata = build_datasets(config)
    save_run_metadata(run_dir, config, split_metadata, verification_run=verification_run)

    grpo_config = build_grpo_config(config, run_dir)

    trainer = GRPOTrainer(
        model=config.model_id,
        reward_funcs=[accuracy_reward, format_reward],
        args=grpo_config,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        callbacks=[
            JsonlLoggerCallback(run_dir / "metrics.jsonl", device=device),
            ConsoleProgressCallback(device=device),
        ],
    )

    # Every run gets a fresh, uniquely-named directory (see make_run_dir), so
    # there is never a checkpoint to resume from yet. Resume support (explicit
    # checkpoint path, fresh-vs-resumed reporting, cross-profile mismatch check)
    # is a later stage's scope.
    print("Starting fresh run (resume support lands in a later stage).")
    try:
        trainer.train(resume_from_checkpoint=None)
    except BaseException:
        update_run_status(run_dir, "failed")
        raise
    else:
        update_run_status(run_dir, "completed")


if __name__ == "__main__":
    main()
