"""Pure scoring and aggregation for generation-only GRPO viability checks.

This module deliberately has no model, dataset, or device imports. The runtime
command supplies completion text/token counts; everything here is CPU-only and
unit-testable.
"""

import math
import statistics

from tiny_grpo.rewards import accuracy_reward, extract_predicted_answer, format_reward


def completion_ids_and_termination(sequence, prompt_length: int, eos_token_ids: set[int]):
    """Return generated IDs through the first EOS and whether EOS was seen."""
    completion_ids = sequence[prompt_length:]
    for index, token_id in enumerate(completion_ids.tolist()):
        if token_id in eos_token_ids:
            return completion_ids[: index + 1], True
    return completion_ids, False


def wilson_interval(successes: int, total: int, z: float = 1.959963984540054) -> dict:
    """Return a two-sided 95% Wilson score interval for a binomial rate."""
    if total < 1:
        raise ValueError(f"total must be >= 1, got {total}")
    if not 0 <= successes <= total:
        raise ValueError(f"successes must be between 0 and total, got {successes}/{total}")

    proportion = successes / total
    denominator = 1 + z * z / total
    centre = (proportion + z * z / (2 * total)) / denominator
    margin = z * math.sqrt(proportion * (1 - proportion) / total + z * z / (4 * total * total)) / denominator
    low = 0.0 if successes == 0 else max(0.0, centre - margin)
    high = 1.0 if successes == total else min(1.0, centre + margin)
    return {"low": low, "high": high}


def classify_completion(
    completion_text: str,
    *,
    exact_reward: float,
    completion_token_count: int,
    max_completion_length: int,
    terminated: bool,
) -> str:
    """Classify why a completion did or did not receive exact reward."""
    if exact_reward > 0:
        return "exact_correct"
    if extract_predicted_answer(completion_text) is not None:
        return "valid_incorrect"
    if not terminated and completion_token_count >= max_completion_length:
        return "truncated_invalid"
    if "<answer" in completion_text or "</answer" in completion_text:
        return "malformed_answer_tag"
    return "missing_answer_tag"


def score_rollout_completion(
    completion_text: str,
    gold_answer: str,
    *,
    completion_token_count: int,
    max_completion_length: int,
    terminated: bool = True,
) -> dict:
    """Score and classify one generated completion with production rewards."""
    completion = [{"role": "assistant", "content": completion_text}]
    exact = accuracy_reward(completions=[completion], answer=[gold_answer])[0]
    fmt = format_reward(completions=[completion])[0]
    return {
        "completion": completion_text,
        "extracted_answer": extract_predicted_answer(completion_text),
        "exact_reward": exact,
        "format_reward": fmt,
        "total_reward": exact + fmt,
        "completion_token_count": completion_token_count,
        "terminated": terminated,
        "truncated": not terminated and completion_token_count >= max_completion_length,
        "failure_category": classify_completion(
            completion_text,
            exact_reward=exact,
            completion_token_count=completion_token_count,
            max_completion_length=max_completion_length,
            terminated=terminated,
        ),
    }


def build_prompt_record(prompt_id: int, prompt_text: str, gold_answer: str, completions: list[dict]) -> dict:
    """Attach group-level statistics to one prompt's scored completions."""
    if not completions:
        raise ValueError("a prompt record requires at least one completion")
    exact_rewards = [record["exact_reward"] for record in completions]
    total_rewards = [record["total_reward"] for record in completions]
    return {
        "prompt_id": prompt_id,
        "prompt": prompt_text,
        "gold_answer": gold_answer,
        "completions": completions,
        "group_mean_reward": statistics.fmean(total_rewards),
        "group_reward_std": statistics.pstdev(total_rewards),
        "group_exact_reward_std": statistics.pstdev(exact_rewards),
        "mixed_exact_rewards": min(exact_rewards) != max(exact_rewards),
        "any_exact_correct": any(reward > 0 for reward in exact_rewards),
    }


def aggregate_rollout_groups(groups: list[dict]) -> dict:
    """Aggregate pass@k, sparsity, formatting, truncation, and group metrics."""
    if not groups:
        raise ValueError("aggregate_rollout_groups requires at least one prompt group")

    completion_counts = {len(group["completions"]) for group in groups}
    if len(completion_counts) != 1 or 0 in completion_counts:
        raise ValueError("every prompt group must contain the same positive number of completions")

    num_generations = completion_counts.pop()
    completions = [completion for group in groups for completion in group["completions"]]
    num_prompts = len(groups)
    total_completions = len(completions)

    first_sample_correct = sum(group["completions"][0]["exact_reward"] > 0 for group in groups)
    any_correct = sum(group["any_exact_correct"] for group in groups)
    exact_count = sum(completion["exact_reward"] > 0 for completion in completions)
    valid_count = sum(completion["extracted_answer"] is not None for completion in completions)
    mixed_count = sum(group["mixed_exact_rewards"] for group in groups)
    zero_exact_std_count = sum(group["group_exact_reward_std"] == 0 for group in groups)
    zero_total_std_count = sum(group["group_reward_std"] == 0 for group in groups)
    all_wrong_count = sum(not group["any_exact_correct"] for group in groups)
    same_total_count = zero_total_std_count
    truncated_count = sum(completion["truncated"] for completion in completions)

    failure_counts = {}
    for completion in completions:
        category = completion["failure_category"]
        failure_counts[category] = failure_counts.get(category, 0) + 1

    pass_at_k_key = f"pass_at_{num_generations}"
    rates = {
        "pass_at_1": first_sample_correct / num_prompts,
        pass_at_k_key: any_correct / num_prompts,
        "sample_exact_accuracy": exact_count / total_completions,
        "valid_format_rate": valid_count / total_completions,
        "parse_failure_rate": (total_completions - valid_count) / total_completions,
        "exact_accuracy_given_valid": exact_count / valid_count if valid_count else 0.0,
        "mixed_exact_reward_group_rate": mixed_count / num_prompts,
        "zero_exact_reward_std_group_rate": zero_exact_std_count / num_prompts,
        "zero_total_reward_std_group_rate": zero_total_std_count / num_prompts,
        "all_completions_wrong_group_rate": all_wrong_count / num_prompts,
        "same_total_reward_group_rate": same_total_count / num_prompts,
        "truncation_rate": truncated_count / total_completions,
    }

    counts = {
        "first_sample_correct": first_sample_correct,
        "prompts_with_any_correct": any_correct,
        "exact_completions": exact_count,
        "valid_completions": valid_count,
        "mixed_exact_reward_groups": mixed_count,
        "zero_exact_reward_std_groups": zero_exact_std_count,
        "zero_total_reward_std_groups": zero_total_std_count,
        "all_completions_wrong_groups": all_wrong_count,
        "same_total_reward_groups": same_total_count,
        "truncated_completions": truncated_count,
        "failure_categories": failure_counts,
    }

    confidence_intervals = {
        "pass_at_1": wilson_interval(first_sample_correct, num_prompts),
        pass_at_k_key: wilson_interval(any_correct, num_prompts),
        "sample_exact_accuracy": wilson_interval(exact_count, total_completions),
        "mixed_exact_reward_group_rate": wilson_interval(mixed_count, num_prompts),
    }

    return {
        "num_prompts": num_prompts,
        "num_generations": num_generations,
        "total_completions": total_completions,
        **rates,
        "average_exact_reward": statistics.fmean(c["exact_reward"] for c in completions),
        "average_format_reward": statistics.fmean(c["format_reward"] for c in completions),
        "mean_total_reward": statistics.fmean(c["total_reward"] for c in completions),
        "average_group_reward_std": statistics.fmean(group["group_reward_std"] for group in groups),
        "average_completion_length": statistics.fmean(c["completion_token_count"] for c in completions),
        "maximum_completion_length": max(c["completion_token_count"] for c in completions),
        "counts": counts,
        "confidence_intervals_95": confidence_intervals,
    }
