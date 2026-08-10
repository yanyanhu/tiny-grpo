"""SFT warm-start entrypoint for the tiny GRPO project.

Usage: uv run python train_sft.py --profile {smoke,debug,stronger} \
    --hardware {mps_16gb,cuda_4gb} --model-profile {smollm2_135m,qwen3_0_6b}
"""

import argparse
import ast
import dataclasses
import json
import os
from pathlib import Path

from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer

from tiny_grpo.config import ResumeConfig
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
from tiny_grpo.rewards import to_prompt
from tiny_grpo.run_context import make_run_dir, save_run_metadata, update_run_status
from tiny_grpo.sft_config import (
    SFTTrainingConfig,
    sft_debug_config,
    sft_smoke_config,
    sft_stronger_config,
)
from tiny_grpo.sft_data import audit_sft_lengths, to_sft_example
from tiny_grpo.splits import SplitMetadata, assert_disjoint, build_split_metadata, select_split
from tiny_grpo.trainer_callbacks import ConsoleProgressCallback, JsonlLoggerCallback

RUN_PROFILES = {
    "smoke": sft_smoke_config,
    "debug": sft_debug_config,
    "stronger": sft_stronger_config,
}


def build_datasets(config: SFTTrainingConfig):
    train_pool = load_dataset("openai/gsm8k", "main", split="train")
    test_pool = load_dataset("openai/gsm8k", "main", split="test")
    metadata = build_split_metadata(
        len(train_pool),
        len(test_pool),
        config.dataset.train_size,
        config.dataset.val_size,
        config.dataset.test_size,
        config.dataset.split_seed,
    )
    train_rows = select_split(train_pool, metadata.train_indices)
    val_rows = select_split(train_pool, metadata.val_indices)
    map_kwargs = {"chat_template_kwargs": config.chat_template_kwargs}
    if config.training_data_path is None:
        train = train_rows.map(
            to_sft_example,
            fn_kwargs=map_kwargs,
            remove_columns=train_pool.column_names,
            load_from_cache_file=False,
        )
    else:
        train = load_dataset("json", data_files=config.training_data_path, split="train")
        required = {"prompt_id", "prompt", "completion", "chat_template_kwargs"}
        if not required.issubset(train.column_names):
            raise ValueError(f"external SFT data is missing columns {sorted(required - set(train.column_names))}")
        prompt_ids = list(train["prompt_id"])
        if len(prompt_ids) != len(set(prompt_ids)):
            raise ValueError("external SFT data contains duplicate prompt IDs")
        if any(value != config.chat_template_kwargs for value in train["chat_template_kwargs"]):
            raise ValueError("external SFT chat-template mode does not match the model profile")
        reserved = build_split_metadata(
            len(train_pool), len(test_pool), 1024, config.dataset.val_size,
            config.dataset.test_size, config.dataset.split_seed,
        )
        if not set(prompt_ids).issubset(reserved.train_indices):
            raise ValueError("external SFT prompt IDs are outside the reserved training split")
        assert_disjoint(prompt_ids, metadata.val_indices)
        metadata = SplitMetadata(
            seed=metadata.seed,
            train_indices=prompt_ids,
            val_indices=metadata.val_indices,
            test_indices=metadata.test_indices,
        )
    validation = val_rows.map(
        to_sft_example,
        fn_kwargs=map_kwargs,
        remove_columns=train_pool.column_names,
        load_from_cache_file=False,
    )
    generation_validation = val_rows.map(
        to_prompt, remove_columns=train_pool.column_names, load_from_cache_file=False
    )
    return train, validation, generation_validation, metadata


def model_init_kwargs(config: SFTTrainingConfig) -> dict:
    kwargs = {"device_map": None}
    dtype = resolve_dtype(config.precision)
    if dtype is not None:
        kwargs["dtype"] = dtype
    return kwargs


def build_trainer_config(config: SFTTrainingConfig, run_dir: Path) -> SFTConfig:
    return SFTConfig(
        output_dir=str(run_dir),
        seed=config.seed,
        data_seed=config.seed,
        per_device_train_batch_size=config.per_device_train_batch_size,
        per_device_eval_batch_size=config.per_device_train_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        gradient_checkpointing=config.gradient_checkpointing,
        learning_rate=config.learning_rate,
        max_steps=config.max_steps,
        max_length=config.max_sequence_length,
        completion_only_loss=config.completion_only_loss,
        assistant_only_loss=False,
        packing=False,
        logging_steps=config.logging_steps,
        disable_tqdm=True,
        report_to=["tensorboard"],
        save_strategy="steps",
        save_steps=config.checkpoint_steps,
        save_total_limit=config.checkpoint_retention,
        eval_strategy="steps",
        eval_steps=config.eval_steps,
        optim="adamw_torch",
        bf16=config.precision == "bf16",
        fp16=config.precision == "fp16",
    )


def _write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2) + "\n")


def _print_comparison(baseline: dict, post: dict) -> None:
    print("\n=== SFT baseline vs. post-training generation evaluation ===")
    for key in ("accuracy", "format_rate", "parse_failure_rate", "mean_reward"):
        print(f"  {key:24s} baseline={baseline[key]:.3f}  post_training={post[key]:.3f}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=sorted(RUN_PROFILES), default="smoke")
    parser.add_argument("--hardware", choices=sorted(HARDWARE_PROFILES), required=True)
    parser.add_argument("--model-profile", choices=sorted(MODEL_PROFILES), default="smollm2_135m")
    parser.add_argument("--verification-run", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--resume", default="none")
    parser.add_argument("--allow-cross-profile-resume", action="store_true")
    parser.add_argument("--train-data-path", type=Path)
    parser.add_argument(
        "--training-data-source",
        choices=("matched_gold_short", "matched_teacher_distilled", "expanded_teacher_distilled"),
    )
    parser.add_argument("--set", action="append", default=[], metavar="FIELD=VALUE")
    args = parser.parse_args()

    hardware = resolve_hardware_profile(args.hardware)
    device = resolve_device(hardware)
    verification = args.verification_run if args.verification_run is not None else args.profile == "smoke"
    model_profile = resolve_model_profile(args.model_profile)
    config = dataclasses.replace(
        RUN_PROFILES[args.profile](hardware, model_profile=model_profile),
        resume=ResumeConfig(mode=args.resume),
    )
    for item in args.set:
        field, separator, raw = item.partition("=")
        if not separator:
            parser.error(f"invalid --set {item!r}; expected FIELD=VALUE")
        try:
            value = ast.literal_eval(raw)
        except (ValueError, SyntaxError):
            value = raw
        config = dataclasses.replace(config, **{field: value})
    if (args.train_data_path is None) != (args.training_data_source is None):
        parser.error("--train-data-path and --training-data-source must be provided together")
    if args.train_data_path is not None:
        config = dataclasses.replace(
            config,
            training_data_path=str(args.train_data_path.resolve()),
            training_data_source=args.training_data_source,
        )
    verify_precision_supported(device, config.precision)
    for name, value in config.env_setup.items():
        os.environ[name] = value

    resume = resolve_resume_target(
        config.resume.mode,
        config.output_dir,
        config.run_name,
        hardware.name,
        allow_cross_profile=args.allow_cross_profile_resume,
    )
    run_dir = resume.run_dir if resume else make_run_dir(config.output_dir, config.run_name)
    checkpoint = resume.checkpoint_path if resume else None
    print(f"{'Resuming' if resume else 'Starting'} SFT run: {run_dir}")
    print(f"(hardware={hardware.name}, device={device}, verification_run={verification})")

    train, validation, generation_validation, metadata = build_datasets(config)
    save_run_metadata(run_dir, config, metadata, verification_run=verification)
    trainer_config = build_trainer_config(config, run_dir)

    print(f"Loading model {config.model_id!r} (precision={config.precision})...")
    model = AutoModelForCausalLM.from_pretrained(config.model_id, **model_init_kwargs(config)).to(device)
    tokenizer = AutoTokenizer.from_pretrained(config.model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    length_stats = {
        "train": audit_sft_lengths(
            train, tokenizer, config.max_sequence_length, config.max_completion_length
        ),
        "validation": audit_sft_lengths(
            validation, tokenizer, config.max_sequence_length, config.max_completion_length
        ),
    }
    _write_json(run_dir / "sft_data_stats.json", length_stats)
    eval_kwargs = dict(
        max_new_tokens=config.max_completion_length,
        temperature=config.temperature,
        top_p=config.top_p,
        top_k=config.top_k,
        seed=config.seed,
        chat_template_kwargs=config.chat_template_kwargs,
    )

    try:
        if checkpoint is None:
            print("Running baseline generation evaluation...")
            baseline = evaluate_model(model, tokenizer, generation_validation, device, **eval_kwargs)
            _write_json(run_dir / "eval_baseline.json", baseline)
        else:
            baseline_path = run_dir / "eval_baseline.json"
            baseline = json.loads(baseline_path.read_text()) if baseline_path.exists() else None
        trainer = SFTTrainer(
            model=model,
            args=trainer_config,
            train_dataset=train,
            eval_dataset=validation,
            processing_class=tokenizer,
            peft_config=to_peft_lora_config(config.lora),
            callbacks=[
                JsonlLoggerCallback(run_dir / "metrics.jsonl", device),
                ConsoleProgressCallback(device),
            ],
        )
        trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model(str(run_dir / "final_adapter"))
        print("Running post-training generation evaluation...")
        post = evaluate_model(trainer.model, tokenizer, generation_validation, device, **eval_kwargs)
        _write_json(run_dir / "eval_post_training.json", post)
    except BaseException:
        update_run_status(run_dir, "failed")
        raise
    else:
        update_run_status(run_dir, "completed")

    if baseline is not None:
        _print_comparison(baseline, post)


if __name__ == "__main__":
    main()
