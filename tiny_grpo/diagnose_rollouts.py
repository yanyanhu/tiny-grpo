"""Generation-only diagnostic for exact-reward GRPO viability.

No optimizer or backward pass is created. The command samples grouped
completions from a fixed, versioned manifest and measures whether exact reward
has enough within-group variation to bootstrap GRPO.
"""

import argparse
import dataclasses
import json
import os
import subprocess
import time
from pathlib import Path

from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from tiny_grpo.config import debug_config, longer_config, smoke_config
from tiny_grpo.diagnostics import aggregate_rollout_groups, build_prompt_record, score_rollout_completion
from tiny_grpo.hardware import (
    HARDWARE_PROFILES,
    resolve_device,
    resolve_dtype,
    resolve_hardware_profile,
    verify_precision_supported,
)
from tiny_grpo.monitoring import device_memory_mb, process_memory_mb
from tiny_grpo.model_profiles import chat_template_kwargs
from tiny_grpo.rewards import to_prompt
from tiny_grpo.run_context import RunTags, collect_environment_info, make_run_dir, save_run_tags, update_run_status
from tiny_grpo.splits import load_diagnostic_manifest, select_split

RUN_PROFILES = {"smoke": smoke_config, "debug": debug_config, "longer": longer_config}
DEFAULT_MANIFEST = Path(__file__).resolve().parents[1] / "data" / "diagnostic_manifest_v1.json"
CHAT_TEMPLATE_MODES = ("default", "thinking", "non-thinking")


def _git_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip()


def _completion_ids_and_termination(sequence, prompt_length: int, eos_token_ids: set[int]):
    completion_ids = sequence[prompt_length:]
    for index, token_id in enumerate(completion_ids.tolist()):
        if token_id in eos_token_ids:
            return completion_ids[: index + 1], True
    return completion_ids, False


def _write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2) + "\n")


def _print_summary(summary: dict) -> None:
    pass_at_k = f"pass_at_{summary['num_generations']}"
    print("\n=== Rollout viability diagnostic ===")
    print(f"Prompts: {summary['num_prompts']}")
    print(f"Completions per prompt: {summary['num_generations']}")
    print(f"Total completions: {summary['total_completions']}")
    print(f"First-sample pass@1: {summary['pass_at_1']:.3%}")
    print(f"Sample exact accuracy: {summary['sample_exact_accuracy']:.3%}")
    print(f"{pass_at_k}: {summary[pass_at_k]:.3%}")
    print(f"Valid format: {summary['valid_format_rate']:.3%}")
    print(f"Exact among valid: {summary['exact_accuracy_given_valid']:.3%}")
    print(f"Mixed exact-reward groups: {summary['mixed_exact_reward_group_rate']:.3%}")
    print(f"Zero exact-reward-std groups: {summary['zero_exact_reward_std_group_rate']:.3%}")
    print(f"Zero total-reward-std groups: {summary['zero_total_reward_std_group_rate']:.3%}")
    print(f"Truncation rate: {summary['truncation_rate']:.3%}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=sorted(RUN_PROFILES), default="debug")
    parser.add_argument("--hardware", choices=sorted(HARDWARE_PROFILES), required=True)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--num-prompts", type=int, default=16)
    parser.add_argument("--num-generations", type=int, default=4)
    parser.add_argument("--max-completion-length", type=int, default=None)
    parser.add_argument("--sampling-seed", type=int, default=42)
    parser.add_argument(
        "--model-id",
        default=None,
        help="Override the profile's base model for generation-only capability comparison.",
    )
    parser.add_argument(
        "--chat-template-mode",
        choices=CHAT_TEMPLATE_MODES,
        default="default",
        help="Control models such as Qwen3 that expose thinking through their chat template.",
    )
    parser.add_argument("--adapter-path", type=Path, default=None)
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument(
        "--command-timeout-seconds",
        type=int,
        required=True,
        help="Outer watchdog timeout, recorded for experiment reproducibility.",
    )
    args = parser.parse_args()

    if args.num_prompts < 1:
        parser.error("--num-prompts must be >= 1")
    if args.num_generations < 1:
        parser.error("--num-generations must be >= 1")
    if args.command_timeout_seconds < 1:
        parser.error("--command-timeout-seconds must be >= 1")

    hardware = resolve_hardware_profile(args.hardware)
    for env_name, env_value in hardware.env_setup.items():
        os.environ[env_name] = env_value
    device = resolve_device(hardware)
    config = RUN_PROFILES[args.profile](hardware)
    verify_precision_supported(device, config.precision)
    model_id = args.model_id or config.model_id
    template_kwargs = chat_template_kwargs(args.chat_template_mode)

    max_completion_length = args.max_completion_length or config.max_completion_length
    if max_completion_length < 1:
        parser.error("--max-completion-length must be >= 1")

    manifest = load_diagnostic_manifest(args.manifest)
    if args.num_prompts > len(manifest.diagnostic_indices):
        parser.error(
            f"--num-prompts={args.num_prompts} exceeds manifest size {len(manifest.diagnostic_indices)}"
        )
    selected_indices = manifest.diagnostic_indices[: args.num_prompts]

    run_dir = make_run_dir(args.output_dir, f"diagnostic_{args.profile}")
    save_run_tags(
        run_dir,
        RunTags(
            run_profile=f"diagnostic_{args.profile}",
            hardware_profile=hardware.name,
            verification_run=True,
        ),
    )

    diagnostic_config = {
        "kind": "rollout_diagnostic",
        "git_commit": _git_commit(),
        "run_profile": args.profile,
        "hardware_profile": hardware.name,
        "device": device,
        "precision": config.precision,
        "model_id": model_id,
        "chat_template_mode": args.chat_template_mode,
        "adapter_path": str(args.adapter_path) if args.adapter_path else None,
        "manifest_path": str(args.manifest),
        "manifest_version": manifest.version,
        "selected_prompt_ids": selected_indices,
        "num_prompts": args.num_prompts,
        "num_generations": args.num_generations,
        "max_completion_length": max_completion_length,
        "temperature": config.temperature,
        "top_p": config.top_p,
        "top_k": config.top_k,
        "sampling_seed": args.sampling_seed,
        "per_prompt_seed_formula": "sampling_seed + prompt_id",
        "command_timeout_seconds": args.command_timeout_seconds,
        "reward_values": {"accuracy": 1.0, "format": 0.2},
    }
    _write_json(run_dir / "config.json", diagnostic_config)
    _write_json(run_dir / "environment.json", collect_environment_info())
    _write_json(
        run_dir / "diagnostic_manifest.json",
        {**manifest.to_dict(), "selected_prompt_ids": selected_indices},
    )

    print(f"Starting rollout diagnostic: {run_dir}")
    print(
        f"(hardware={hardware.name}, device={device}, prompts={args.num_prompts}, "
        f"generations={args.num_generations}, max_completion_length={max_completion_length})"
    )

    try:
        pool = load_dataset(
            manifest.source_dataset,
            manifest.source_config,
            split=manifest.source_split,
        )
        if max(selected_indices) >= len(pool):
            raise ValueError(
                f"manifest prompt ID {max(selected_indices)} is outside loaded pool size {len(pool)}"
            )
        dataset = select_split(pool, selected_indices).map(
            to_prompt,
            remove_columns=pool.column_names,
            load_from_cache_file=False,
        )

        dtype = resolve_dtype(config.precision)
        model_kwargs = {"device_map": None}
        if dtype is not None:
            model_kwargs["dtype"] = dtype
        model = AutoModelForCausalLM.from_pretrained(model_id, **model_kwargs)
        if args.adapter_path is not None:
            from peft import PeftModel

            model = PeftModel.from_pretrained(model, str(args.adapter_path))
        model.to(device)
        model.eval()

        tokenizer = AutoTokenizer.from_pretrained(model_id)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        eos_token_ids = {tokenizer.eos_token_id}
        if getattr(model.generation_config, "eos_token_id", None) is not None:
            configured_eos = model.generation_config.eos_token_id
            eos_token_ids.update(configured_eos if isinstance(configured_eos, list) else [configured_eos])
        eos_token_ids.discard(None)

        import torch

        groups = []
        started = time.monotonic()
        records_path = run_dir / "prompt_results.jsonl"
        with records_path.open("w") as records_file:
            for position, (prompt_id, example) in enumerate(zip(selected_indices, dataset), start=1):
                prompt_text = tokenizer.apply_chat_template(
                    example["prompt"],
                    tokenize=False,
                    add_generation_prompt=True,
                    **template_kwargs,
                )
                inputs = tokenizer(prompt_text, return_tensors="pt").to(device)
                prompt_seed = args.sampling_seed + prompt_id
                torch.manual_seed(prompt_seed)
                if device == "cuda":
                    torch.cuda.manual_seed_all(prompt_seed)

                with torch.inference_mode():
                    output_ids = model.generate(
                        **inputs,
                        max_new_tokens=max_completion_length,
                        do_sample=True,
                        temperature=config.temperature,
                        top_p=config.top_p,
                        top_k=config.top_k,
                        num_return_sequences=args.num_generations,
                        pad_token_id=tokenizer.pad_token_id,
                    )

                scored = []
                prompt_length = inputs["input_ids"].shape[1]
                for sequence in output_ids:
                    completion_ids, terminated = _completion_ids_and_termination(
                        sequence, prompt_length, eos_token_ids
                    )
                    completion_text = tokenizer.decode(completion_ids, skip_special_tokens=True)
                    scored.append(
                        score_rollout_completion(
                            completion_text,
                            example["answer"],
                            completion_token_count=completion_ids.shape[0],
                            max_completion_length=max_completion_length,
                            terminated=terminated,
                        )
                    )

                group = build_prompt_record(prompt_id, prompt_text, example["answer"], scored)
                group["sampling_seed"] = prompt_seed
                groups.append(group)
                records_file.write(json.dumps(group) + "\n")
                records_file.flush()  # preserve completed groups if an external timeout interrupts the run
                print(f"diagnostic prompt {position}/{args.num_prompts} id={prompt_id}", flush=True)

        summary = aggregate_rollout_groups(groups)
        summary["runtime_seconds"] = time.monotonic() - started
        summary["process_memory_mb"] = process_memory_mb()
        memory = device_memory_mb(device)
        if memory is not None:
            summary[f"{device}_memory_mb"] = memory
        _write_json(run_dir / "summary.json", summary)
    except BaseException:
        update_run_status(run_dir, "failed")
        raise
    else:
        update_run_status(run_dir, "completed")

    _print_summary(summary)
    print(f"Structured results: {run_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
