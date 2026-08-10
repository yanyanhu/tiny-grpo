"""CPU-only tests for rollout-relative curriculum construction."""

from tiny_grpo.viability_curriculum import classify_viability_group, select_curriculum


def _group(prompt_id, exact, valid, truncated):
    completions = []
    for index in range(4):
        completions.append({
            "exact_reward": float(index < exact),
            "extracted_answer": "1" if index < valid else None,
            "truncated": index < truncated,
            "terminated": index >= truncated,
            "completion_token_count": 128 if index < truncated else 20,
        })
    return {"prompt_id": prompt_id, "completions": completions}


def test_viability_categories_follow_predeclared_priority():
    assert classify_viability_group(_group(1, 2, 4, 0), set())["category"] == "high_signal"
    assert classify_viability_group(_group(2, 4, 4, 0), set())["category"] == "stable"
    assert classify_viability_group(_group(3, 0, 4, 0), {3})["category"] == "frontier"
    assert classify_viability_group(_group(4, 0, 1, 3), {4})["category"] == "low_value"
    assert classify_viability_group(_group(5, 0, 4, 0), set())["category"] == "unclassified"


def test_curriculum_selection_is_deterministic_and_records_fallback():
    records = []
    for category, count in (("high_signal", 6), ("frontier", 3), ("stable", 1)):
        start = len(records)
        records.extend({"prompt_id": start + i, "category": category} for i in range(count))
    first, summary = select_curriculum(records, 10, 31415)
    second, _ = select_curriculum(records, 10, 31415)
    assert first == second
    assert len(first) == len(set(first)) == 10
    assert summary["fallback_count"] == 0
    assert summary["final_selected_counts"] == {
        "frontier": 3, "high_signal": 6, "stable": 1
    }
