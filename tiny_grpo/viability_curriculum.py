"""Pure rollout-viability classification and deterministic curriculum selection."""

import random


def classify_viability_group(group: dict, verified_teacher_ids: set[int]) -> dict:
    completions = group["completions"]
    exact_count = sum(item["exact_reward"] > 0 for item in completions)
    truncation_count = sum(item["truncated"] for item in completions)
    valid_format_count = sum(item["extracted_answer"] is not None for item in completions)
    terminated_count = sum(item["terminated"] for item in completions)
    lengths = [item["completion_token_count"] for item in completions]
    if exact_count == len(completions):
        category = "stable"
    elif 0 < exact_count < len(completions):
        category = "high_signal"
    elif truncation_count >= 3 or valid_format_count < 2:
        category = "low_value"
    elif (exact_count == 0 and terminated_count >= 3 and valid_format_count >= 3
          and group["prompt_id"] in verified_teacher_ids):
        category = "frontier"
    else:
        category = "unclassified"
    return {
        "prompt_id": group["prompt_id"],
        "category": category,
        "exact_count": exact_count,
        "truncation_count": truncation_count,
        "valid_format_count": valid_format_count,
        "terminated_count": terminated_count,
        "completion_lengths": lengths,
    }


def select_curriculum(records: list[dict], size: int, seed: int) -> tuple[list[int], dict]:
    """Select 60/30/10 high-signal/frontier/stable-or-unfiltered prompts."""
    if size < 1 or size > len(records):
        raise ValueError("curriculum size must be between 1 and the record count")
    rng = random.Random(seed)
    by_category = {}
    for record in records:
        by_category.setdefault(record["category"], []).append(record["prompt_id"])
    for values in by_category.values():
        rng.shuffle(values)
    targets = {
        "high_signal": round(size * 0.60),
        "frontier": round(size * 0.30),
    }
    targets["stable_or_unfiltered"] = size - sum(targets.values())
    selected = []
    selected_counts = {}
    for category in ("high_signal", "frontier"):
        take = min(targets[category], len(by_category.get(category, [])))
        selected.extend(by_category.get(category, [])[:take])
        selected_counts[category] = take
    remainder_pool = []
    for category in ("stable", "unclassified", "low_value"):
        remainder_pool.extend(by_category.get(category, []))
    rng.shuffle(remainder_pool)
    take = min(targets["stable_or_unfiltered"], len(remainder_pool))
    selected.extend(remainder_pool[:take])
    selected_counts["stable_or_unfiltered"] = take
    if len(selected) < size:
        already = set(selected)
        fallback = [record["prompt_id"] for record in records if record["prompt_id"] not in already]
        rng.shuffle(fallback)
        selected.extend(fallback[: size - len(selected)])
    category_by_id = {record["prompt_id"]: record["category"] for record in records}
    final_counts = {}
    for prompt_id in selected:
        category = category_by_id[prompt_id]
        final_counts[category] = final_counts.get(category, 0) + 1
    rng.shuffle(selected)
    return selected, {
        "requested_size": size,
        "selection_seed": seed,
        "target_counts": targets,
        "selected_primary_counts": selected_counts,
        "fallback_count": size - sum(selected_counts.values()),
        "final_selected_counts": dict(sorted(final_counts.items())),
        "available_counts": {key: len(value) for key, value in sorted(by_category.items())},
    }
