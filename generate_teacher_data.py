"""Generate verified thinking traces from training-only GSM8K prompts."""

import argparse
import json
import os
import random
import subprocess
import time
from pathlib import Path

from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from tiny_grpo.diagnostics import completion_ids_and_termination, score_rollout_completion
from tiny_grpo.hardware import (
    HARDWARE_PROFILES,
    resolve_device,
    resolve_dtype,
    resolve_hardware_profile,
    verify_precision_supported,
)
from tiny_grpo.monitoring import device_memory_mb, process_memory_mb
from tiny_grpo.model_profiles import MODEL_PROFILES, chat_template_kwargs, resolve_model_profile
from tiny_grpo.rewards import to_prompt
from tiny_grpo.run_context import RunTags, collect_environment_info, make_run_dir, save_run_tags, update_run_status
from tiny_grpo.splits import (
    assert_disjoint,
    build_split_metadata,
    load_diagnostic_manifest,
    save_split_metadata,
    select_split,
)
from tiny_grpo.teacher_data import (
    annotate_compression_candidate,
    annotate_teacher_candidate,
    build_compression_messages,
    build_student_target,
    select_shortest_accepted,
    select_shortest_verified_teacher,
    summarize_teacher_records,
)

DEFAULT_MANIFEST = Path(__file__).resolve().parent / "data" / "diagnostic_manifest_v1.json"


def _write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2) + "\n")


def _git_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True, timeout=5
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hardware", choices=sorted(HARDWARE_PROFILES), required=True)
    parser.add_argument("--model-profile", choices=sorted(MODEL_PROFILES), default="qwen3_0_6b")
    parser.add_argument("--num-prompts", type=int, default=8)
    parser.add_argument("--num-generations", type=int, default=2)
    parser.add_argument("--num-compressions", type=int, default=0)
    parser.add_argument("--teacher-max-completion-length", type=int, default=1024)
    parser.add_argument("--student-max-target-tokens", type=int, default=128)
    parser.add_argument("--train-pool-size", type=int, default=256)
    parser.add_argument("--sampling-seed", type=int, default=42)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--verification-run", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--command-timeout-seconds", type=int, required=True)
    args = parser.parse_args()

    for name in ("num_prompts", "num_generations", "teacher_max_completion_length",
                 "student_max_target_tokens", "train_pool_size", "command_timeout_seconds"):
        if getattr(args, name) < 1:
            parser.error(f"--{name.replace('_', '-')} must be >= 1")
    if args.num_compressions < 0:
        parser.error("--num-compressions must be >= 0")
    if args.num_prompts > args.train_pool_size:
        parser.error("--num-prompts cannot exceed --train-pool-size")

    hardware = resolve_hardware_profile(args.hardware)
    model_profile = resolve_model_profile(args.model_profile)
    if model_profile.name != "qwen3_0_6b":
        parser.error("thinking-teacher generation currently requires --model-profile qwen3_0_6b")
    for name, value in hardware.env_setup.items():
        os.environ[name] = value
    device = resolve_device(hardware)
    verify_precision_supported(device, hardware.precision)

    run_dir = make_run_dir(args.output_dir, "teacher_generation")
    save_run_tags(
        run_dir,
        RunTags("teacher_generation", hardware.name, args.verification_run),
    )
    config = {
        "kind": "thinking_teacher_generation",
        "git_commit": _git_commit(),
        "hardware_profile": hardware.name,
        "device": device,
        "precision": hardware.precision,
        "model_profile": model_profile.name,
        "model_id": model_profile.model_id,
        "teacher_chat_template_mode": "thinking",
        "student_chat_template_mode": "non-thinking",
        "num_prompts": args.num_prompts,
        "num_generations": args.num_generations,
        "num_compressions": args.num_compressions,
        "teacher_max_completion_length": args.teacher_max_completion_length,
        "student_max_target_tokens": args.student_max_target_tokens,
        "train_pool_size": args.train_pool_size,
        "sampling_seed": args.sampling_seed,
        "per_prompt_seed_formula": "sampling_seed + prompt_id",
        "command_timeout_seconds": args.command_timeout_seconds,
    }
    _write_json(run_dir / "config.json", config)
    _write_json(run_dir / "environment.json", collect_environment_info())
    print(f"Starting thinking-teacher generation: {run_dir}")

    try:
        train_pool = load_dataset("openai/gsm8k", "main", split="train")
        test_pool = load_dataset("openai/gsm8k", "main", split="test")
        split = build_split_metadata(
            len(train_pool), len(test_pool), args.train_pool_size, 32, 64, args.sampling_seed
        )
        manifest = load_diagnostic_manifest(args.manifest)
        assert_disjoint(split.train_indices, split.val_indices)
        assert_disjoint(split.train_indices, manifest.diagnostic_indices)
        selected_ids = list(split.train_indices)
        random.Random(args.sampling_seed).shuffle(selected_ids)
        selected_ids = selected_ids[: args.num_prompts]
        save_split_metadata(run_dir / "split_metadata.json", split)
        _write_json(run_dir / "teacher_manifest.json", {"selected_prompt_ids": selected_ids})

        raw_dataset = select_split(train_pool, selected_ids)
        dataset = raw_dataset.map(
            to_prompt, remove_columns=train_pool.column_names, load_from_cache_file=False
        )
        dtype = resolve_dtype(hardware.precision)
        model_kwargs = {"device_map": None}
        if dtype is not None:
            model_kwargs["dtype"] = dtype
        model = AutoModelForCausalLM.from_pretrained(model_profile.model_id, **model_kwargs).to(device)
        model.eval()
        tokenizer = AutoTokenizer.from_pretrained(model_profile.model_id)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        eos_ids = {tokenizer.eos_token_id}
        configured_eos = getattr(model.generation_config, "eos_token_id", None)
        if configured_eos is not None:
            eos_ids.update(configured_eos if isinstance(configured_eos, list) else [configured_eos])
        eos_ids.discard(None)

        import torch

        records = []
        started = time.monotonic()
        with (run_dir / "teacher_records.jsonl").open("w") as records_file, \
             (run_dir / "accepted_sft.jsonl").open("w") as accepted_file:
            for position, (prompt_id, raw_example, example) in enumerate(
                zip(selected_ids, raw_dataset, dataset), start=1
            ):
                prompt_text = tokenizer.apply_chat_template(
                    example["prompt"], tokenize=False, add_generation_prompt=True,
                    **chat_template_kwargs("thinking"),
                )
                inputs = tokenizer(prompt_text, return_tensors="pt").to(device)
                prompt_seed = args.sampling_seed + prompt_id
                torch.manual_seed(prompt_seed)
                if device == "cuda":
                    torch.cuda.manual_seed_all(prompt_seed)
                with torch.inference_mode():
                    output_ids = model.generate(
                        **inputs,
                        max_new_tokens=args.teacher_max_completion_length,
                        do_sample=True,
                        temperature=1.0,
                        top_p=1.0,
                        top_k=0,
                        num_return_sequences=args.num_generations,
                        pad_token_id=tokenizer.pad_token_id,
                    )

                candidates = []
                prompt_length = inputs["input_ids"].shape[1]
                for sequence in output_ids:
                    completion_ids, terminated = completion_ids_and_termination(sequence, prompt_length, eos_ids)
                    completion = tokenizer.decode(completion_ids, skip_special_tokens=True)
                    scored = score_rollout_completion(
                        completion,
                        example["answer"],
                        completion_token_count=completion_ids.shape[0],
                        max_completion_length=args.teacher_max_completion_length,
                        terminated=terminated,
                    )
                    target = build_student_target(completion, example["answer"])
                    target_tokens = len(tokenizer(target, add_special_tokens=False)["input_ids"]) if target else None
                    candidates.append(
                        annotate_teacher_candidate(
                            scored, example["answer"], target_tokens, args.student_max_target_tokens
                        )
                    )
                selected = select_shortest_accepted(candidates)
                selected_index = candidates.index(selected) if selected is not None else None
                verified_teacher = select_shortest_verified_teacher(candidates)
                compression_candidates = []
                if verified_teacher is not None and args.num_compressions > 0:
                    compression_messages = build_compression_messages(
                        raw_example["question"],
                        verified_teacher["student_target"],
                        args.student_max_target_tokens,
                    )
                    compression_text = tokenizer.apply_chat_template(
                        compression_messages,
                        tokenize=False,
                        add_generation_prompt=True,
                        **chat_template_kwargs("non-thinking"),
                    )
                    compression_inputs = tokenizer(compression_text, return_tensors="pt").to(device)
                    compression_seed = prompt_seed + 1_000_000
                    torch.manual_seed(compression_seed)
                    if device == "cuda":
                        torch.cuda.manual_seed_all(compression_seed)
                    with torch.inference_mode():
                        compressed_ids = model.generate(
                            **compression_inputs,
                            max_new_tokens=args.student_max_target_tokens,
                            do_sample=True,
                            temperature=1.0,
                            top_p=1.0,
                            top_k=0,
                            num_return_sequences=args.num_compressions,
                            pad_token_id=tokenizer.pad_token_id,
                        )
                    compression_prompt_length = compression_inputs["input_ids"].shape[1]
                    for sequence in compressed_ids:
                        completion_ids, terminated = completion_ids_and_termination(
                            sequence, compression_prompt_length, eos_ids
                        )
                        completion = tokenizer.decode(completion_ids, skip_special_tokens=True)
                        scored = score_rollout_completion(
                            completion,
                            example["answer"],
                            completion_token_count=completion_ids.shape[0],
                            max_completion_length=args.student_max_target_tokens,
                            terminated=terminated,
                        )
                        compression_candidates.append(
                            annotate_compression_candidate(
                                scored, example["answer"], args.student_max_target_tokens
                            )
                        )
                selected_compression = select_shortest_accepted(compression_candidates)
                selected_compression_index = (
                    compression_candidates.index(selected_compression)
                    if selected_compression is not None else None
                )
                final_selected = selected_compression or selected
                record = {
                    "prompt_id": prompt_id,
                    "gold_answer": example["answer"],
                    "sampling_seed": prompt_seed,
                    "selected_candidate_index": selected_index,
                    "candidates": candidates,
                    "selected_compression_index": selected_compression_index,
                    "compression_candidates": compression_candidates,
                }
                records.append(record)
                records_file.write(json.dumps(record) + "\n")
                records_file.flush()
                if final_selected is not None:
                    accepted_file.write(json.dumps({
                        "prompt_id": prompt_id,
                        "prompt": example["prompt"],
                        "completion": [{"role": "assistant", "content": final_selected["student_target"]}],
                        "source": (
                            "thinking_teacher_compressed"
                            if selected_compression is not None else "thinking_teacher_uncompressed"
                        ),
                    }) + "\n")
                    accepted_file.flush()
                print(f"teacher prompt {position}/{args.num_prompts} id={prompt_id}", flush=True)

        summary = summarize_teacher_records(records)
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

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
