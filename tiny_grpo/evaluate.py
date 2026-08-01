"""Baseline / post-training validation evaluation.

Split into a pure aggregation part (no model/tokenizer, fully unit-testable)
and a model-touching generation part, per the project's usual pattern of
keeping pure logic separate from anything that needs a real model.
"""

import time

from tiny_grpo.monitoring import device_memory_mb, process_memory_mb
from tiny_grpo.rewards import accuracy_reward, extract_predicted_answer, format_reward

ACCURACY_REWARD_VALUE = 1.0
FORMAT_REWARD_VALUE = 0.2


def _score_example(prompt_text: str, gold: str, completion_text: str, completion_token_count: int) -> dict:
    """Score one (prompt, gold, completion) triple using the *same* reward
    functions training uses — not a reimplementation, so scoring can never
    silently drift from what training actually optimizes.
    """
    completion = [{"role": "assistant", "content": completion_text}]
    accuracy = accuracy_reward(completions=[completion], answer=[gold])[0]
    fmt = format_reward(completions=[completion])[0]
    return {
        "prompt": prompt_text,
        "gold_answer": gold,
        "completion": completion_text,
        "extracted_answer": extract_predicted_answer(completion_text),
        "accuracy_reward": accuracy,
        "format_reward": fmt,
        "total_reward": accuracy + fmt,
        "completion_token_count": completion_token_count,
    }


def aggregate_eval_records(records: list[dict]) -> dict:
    """Pure aggregation over per-example score records — no model/tokenizer
    access, so this is testable with plain synthetic dicts.
    """
    n = len(records)
    if n == 0:
        raise ValueError("aggregate_eval_records requires at least one record")

    accuracy = sum(r["accuracy_reward"] for r in records) / n / ACCURACY_REWARD_VALUE
    format_rate = sum(r["format_reward"] for r in records) / n / FORMAT_REWARD_VALUE
    parse_failure_rate = sum(1 for r in records if r["extracted_answer"] is None) / n
    mean_reward = sum(r["total_reward"] for r in records) / n
    mean_completion_length = sum(r["completion_token_count"] for r in records) / n

    return {
        "num_examples": n,
        "accuracy": accuracy,
        "format_rate": format_rate,
        "parse_failure_rate": parse_failure_rate,
        "mean_reward": mean_reward,
        "mean_completion_length": mean_completion_length,
    }


def evaluate_model(
    model,
    tokenizer,
    dataset,
    device: str,
    *,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int,
    seed: int,
    num_samples_to_keep: int = 4,
) -> dict:
    """Generate + score every example in `dataset`, one at a time (small
    validation sets; avoids batch-padding complexity). Uses the exact same
    generation settings passed in — callers should pass the already-resolved
    GRPOConfig's temperature/top_p/top_k so eval never drifts from training.
    """
    import torch

    torch.manual_seed(seed)

    was_training = model.training
    model.eval()

    records = []
    start = time.monotonic()
    try:
        for example in dataset:
            prompt_text = tokenizer.apply_chat_template(
                example["prompt"], tokenize=False, add_generation_prompt=True
            )
            inputs = tokenizer(prompt_text, return_tensors="pt").to(device)
            with torch.no_grad():
                output_ids = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    pad_token_id=tokenizer.pad_token_id,
                )
            completion_ids = output_ids[0][inputs["input_ids"].shape[1] :]
            completion_text = tokenizer.decode(completion_ids, skip_special_tokens=True)
            records.append(
                _score_example(prompt_text, example["answer"], completion_text, completion_ids.shape[0])
            )
    finally:
        if was_training:
            model.train()

    runtime_seconds = time.monotonic() - start

    result = aggregate_eval_records(records)
    result["runtime_seconds"] = runtime_seconds
    result["process_memory_mb"] = process_memory_mb()
    device_mem = device_memory_mb(device)
    if device_mem is not None:
        result[f"{device}_memory_mb"] = device_mem
    result["samples"] = records[:num_samples_to_keep]
    return result
