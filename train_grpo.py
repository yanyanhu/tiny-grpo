"""GRPO training entrypoint: wires config, dataset splits, and reward logic together.

Usage:
    uv run python train_grpo.py --profile {smoke,debug,longer} \
        --hardware {mps_16gb,cuda_4gb} --model-profile {smollm2_135m,qwen3_0_6b}
"""

import argparse
import ast
import dataclasses
import json
import os
from pathlib import Path

from datasets import load_dataset
from peft import PeftConfig, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import GRPOConfig, GRPOTrainer

from tiny_grpo.config import (
    ResumeConfig,
    TrainingConfig,
    apply_config_override,
    debug_config,
    longer_config,
    smoke_config,
)
from tiny_grpo.evaluate import evaluate_model
from tiny_grpo.hardware import (
    HARDWARE_PROFILES,
    resolve_device,
    resolve_dtype,
    resolve_hardware_profile,
    verify_precision_supported,
)
from tiny_grpo.lora import to_peft_lora_config
from tiny_grpo.model_profiles import MODEL_PROFILES, resolve_model_profile
from tiny_grpo.resume import resolve_resume_target
from tiny_grpo.rewards import accuracy_reward, format_reward, to_prompt
from tiny_grpo.run_context import make_run_dir, save_run_metadata, update_run_status
from tiny_grpo.splits import SplitMetadata, assert_disjoint, build_split_metadata, select_split
from tiny_grpo.trainer_callbacks import ConsoleProgressCallback, JsonlLoggerCallback

RUN_PROFILES = {"smoke": smoke_config, "debug": debug_config, "longer": longer_config}


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
    if config.training_manifest_path is not None:
        manifest = json.loads(Path(config.training_manifest_path).read_text())
        prompt_ids = list(manifest["selected_prompt_ids"])
        if len(prompt_ids) != len(set(prompt_ids)):
            raise ValueError("training manifest contains duplicate prompt IDs")
        reserved = build_split_metadata(
            len(train_pool), len(test_pool), 1024, config.dataset.val_size,
            config.dataset.test_size, config.dataset.split_seed,
        )
        if not set(prompt_ids).issubset(reserved.train_indices):
            raise ValueError("training manifest prompt IDs are outside the reserved training split")
        assert_disjoint(prompt_ids, split_metadata.val_indices)
        split_metadata = SplitMetadata(
            seed=split_metadata.seed, train_indices=prompt_ids,
            val_indices=split_metadata.val_indices, test_indices=split_metadata.test_indices,
        )

    # load_from_cache_file=False: datasets.map() fingerprints by more than just
    # source code in ways that don't reliably invalidate on every to_prompt
    # edit (confirmed empirically — a prompt-wording change silently kept
    # serving stale cached prompts). to_prompt is pure and these datasets are
    # tiny, so re-mapping is effectively free — always recompute instead.
    train_dataset = select_split(train_pool, split_metadata.train_indices).map(
        to_prompt, remove_columns=train_pool.column_names, load_from_cache_file=False
    )
    val_dataset = select_split(train_pool, split_metadata.val_indices).map(
        to_prompt, remove_columns=train_pool.column_names, load_from_cache_file=False
    )
    return train_dataset, val_dataset, split_metadata


def model_init_kwargs(config: TrainingConfig) -> dict:
    kwargs = {
        # device_map="auto" (the default when loading a model by name) hangs on MPS;
        # force a plain single-device load instead.
        "device_map": None,
    }
    dtype = resolve_dtype(config.precision)
    if dtype is not None:
        kwargs["dtype"] = dtype
    return kwargs


def load_training_model(config: TrainingConfig, device: str):
    """Load the base model, optionally continuing one existing LoRA adapter."""
    model = AutoModelForCausalLM.from_pretrained(config.model_id, **model_init_kwargs(config))
    if config.initial_adapter_path is not None:
        adapter_path = Path(config.initial_adapter_path)
        if not adapter_path.is_dir():
            raise FileNotFoundError(f"initial adapter path is not a directory: {adapter_path}")
        adapter_config = PeftConfig.from_pretrained(str(adapter_path))
        if adapter_config.base_model_name_or_path != config.model_id:
            raise ValueError(
                "initial adapter base model mismatch: "
                f"expected {config.model_id!r}, got {adapter_config.base_model_name_or_path!r}"
            )
        model = PeftModel.from_pretrained(
            model, str(adapter_path), is_trainable=True, config=adapter_config
        )
    return model.to(device)


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
        temperature=config.temperature,
        top_p=config.top_p,
        top_k=config.top_k,
        chat_template_kwargs=config.chat_template_kwargs,
        beta=config.beta,
        learning_rate=config.learning_rate,
        lr_scheduler_type=config.lr_scheduler_type,
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
    )


def _write_eval_result(run_dir: Path, name: str, result: dict) -> None:
    (run_dir / f"eval_{name}.json").write_text(json.dumps(result, indent=2))


def _read_eval_result(run_dir: Path, name: str) -> dict | None:
    path = run_dir / f"eval_{name}.json"
    return json.loads(path.read_text()) if path.exists() else None


def _print_eval_comparison(baseline: dict, post_training: dict) -> None:
    print("\n=== Baseline vs. post-training evaluation ===")
    fields = [
        ("accuracy", "{:.3f}"),
        ("format_rate", "{:.3f}"),
        ("parse_failure_rate", "{:.3f}"),
        ("mean_reward", "{:.3f}"),
        ("mean_completion_length", "{:.1f}"),
        ("runtime_seconds", "{:.1f}"),
        ("process_memory_mb", "{:.0f}"),
    ]
    for key, fmt in fields:
        print(f"  {key:24s} baseline={fmt.format(baseline[key]):>10s}  post_training={fmt.format(post_training[key]):>10s}")
    print(f"  num_examples: {baseline['num_examples']}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=sorted(RUN_PROFILES), default="smoke")
    parser.add_argument("--hardware", choices=sorted(HARDWARE_PROFILES), required=True)
    parser.add_argument("--model-profile", choices=sorted(MODEL_PROFILES), default="smollm2_135m")
    parser.add_argument(
        "--verification-run",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Tag this run as verification-only (auto-deletion-eligible via tiny_grpo.cleanup). "
        "Defaults to True for --profile smoke, False otherwise.",
    )
    parser.add_argument(
        "--resume",
        default="none",
        help='"none" (default, always fresh), "latest" (resume the most recent incomplete run of the '
        "same --profile/--hardware, if one exists), or an explicit checkpoint/run-directory path.",
    )
    parser.add_argument(
        "--allow-cross-profile-resume",
        action="store_true",
        help="Allow resuming a checkpoint that was run under a different --hardware profile "
        "(otherwise this fails loudly rather than silently attempting it).",
    )
    parser.add_argument("--initial-adapter-path", type=Path)
    parser.add_argument("--initial-adapter-source")
    parser.add_argument("--training-manifest", type=Path)
    parser.add_argument("--training-manifest-source")
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="FIELD=VALUE",
        help="Override a top-level resolved TrainingConfig field, e.g. "
        "--set max_completion_length=128, or a nested dataclass field, e.g. "
        "--set dataset.train_size=512. Repeatable. VALUE is parsed as a Python "
        "literal (int/float/bool/str) via ast.literal_eval, falling back to the raw string.",
    )
    args = parser.parse_args()

    hardware = resolve_hardware_profile(args.hardware)
    device = resolve_device(hardware)  # raises DeviceUnavailableError if this machine lacks it
    verification_run = args.verification_run if args.verification_run is not None else (args.profile == "smoke")

    model_profile = resolve_model_profile(args.model_profile)
    config = RUN_PROFILES[args.profile](hardware, model_profile=model_profile)
    config = dataclasses.replace(config, resume=ResumeConfig(mode=args.resume))
    for item in args.set:
        field, _, raw_value = item.partition("=")
        try:
            value = ast.literal_eval(raw_value)
        except (ValueError, SyntaxError):
            value = raw_value
        config = apply_config_override(config, field, value)
    if (args.initial_adapter_path is None) != (args.initial_adapter_source is None):
        parser.error("--initial-adapter-path and --initial-adapter-source must be provided together")
    if args.initial_adapter_path is not None:
        config = dataclasses.replace(
            config,
            initial_adapter_path=str(args.initial_adapter_path.resolve()),
            initial_adapter_source=args.initial_adapter_source,
        )
    if (args.training_manifest is None) != (args.training_manifest_source is None):
        parser.error("--training-manifest and --training-manifest-source must be provided together")
    if args.training_manifest is not None:
        config = dataclasses.replace(
            config,
            training_manifest_path=str(args.training_manifest.resolve()),
            training_manifest_source=args.training_manifest_source,
        )
    verify_precision_supported(device, config.precision)  # fails loudly, never silently substitutes

    for env_name, env_value in config.env_setup.items():
        os.environ[env_name] = env_value

    resume_target = resolve_resume_target(
        config.resume.mode,
        config.output_dir,
        config.run_name,
        hardware.name,
        allow_cross_profile=args.allow_cross_profile_resume,
    )
    if resume_target is not None:
        run_dir = resume_target.run_dir
        checkpoint_path = resume_target.checkpoint_path
        print(
            f"Resuming from checkpoint: {checkpoint_path} (run dir: {run_dir}, "
            f"originally run on hardware profile: {resume_target.origin_hardware_profile!r})"
        )
    else:
        run_dir = make_run_dir(config.output_dir, config.run_name)
        checkpoint_path = None
        print(f"Starting fresh run: {run_dir}")
    print(f"(hardware={hardware.name}, device={device}, verification_run={verification_run})")

    train_dataset, val_dataset, split_metadata = build_datasets(config)
    save_run_metadata(run_dir, config, split_metadata, verification_run=verification_run)

    grpo_config = build_grpo_config(config, run_dir)

    print(f"Loading model {config.model_id!r} (precision={config.precision})...")
    model = load_training_model(config, device)
    tokenizer = AutoTokenizer.from_pretrained(config.model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Same generation settings for baseline and post-training eval as training
    # itself uses (spec requirement) — read straight off the resolved
    # GRPOConfig rather than duplicating separate constants that could drift.
    eval_kwargs = dict(
        max_new_tokens=config.max_completion_length,
        temperature=grpo_config.temperature,
        top_p=grpo_config.top_p,
        top_k=grpo_config.top_k,
        seed=config.seed,
        chat_template_kwargs=config.chat_template_kwargs,
    )

    try:
        # Baseline eval only makes sense once, at true step 0 — on a resumed
        # run the model is already partially trained, so re-running it would
        # mislabel a partially-trained checkpoint as "baseline". Reuse the
        # original run's eval_baseline.json instead.
        if checkpoint_path is None:
            print("Running baseline evaluation (pre-training)...")
            baseline_eval = evaluate_model(model, tokenizer, val_dataset, device, **eval_kwargs)
            _write_eval_result(run_dir, "baseline", baseline_eval)
        else:
            baseline_eval = _read_eval_result(run_dir, "baseline")

        trainer = GRPOTrainer(
            model=model,
            reward_funcs=[accuracy_reward, format_reward],
            args=grpo_config,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            peft_config=(
                None if config.initial_adapter_path is not None
                else to_peft_lora_config(config.lora)
            ),
            callbacks=[
                JsonlLoggerCallback(run_dir / "metrics.jsonl", device=device),
                ConsoleProgressCallback(device=device),
            ],
        )

        trainer.train(resume_from_checkpoint=checkpoint_path)

        # The final adapter, kept independent of the 2-checkpoint rolling cap
        # (save_total_limit only governs checkpoint-N dirs) — this is the
        # selected result, not training-state history.
        trainer.save_model(str(run_dir / "final_adapter"))

        print("Running post-training evaluation...")
        post_training_eval = evaluate_model(trainer.model, tokenizer, val_dataset, device, **eval_kwargs)
        _write_eval_result(run_dir, "post_training", post_training_eval)
    except BaseException:
        update_run_status(run_dir, "failed")
        raise
    else:
        update_run_status(run_dir, "completed")

    if baseline_eval is not None:
        _print_eval_comparison(baseline_eval, post_training_eval)
    else:
        print("\n(No baseline evaluation available to compare against — eval_baseline.json was missing.)")


if __name__ == "__main__":
    main()
