"""CPU-only tests for thinking-teacher candidate processing."""

from generate_teacher_data import DEFAULT_MANIFEST
from tiny_grpo.teacher_data import (
    annotate_compression_candidate,
    annotate_teacher_candidate,
    build_compression_messages,
    build_student_target,
    extract_thinking_reasoning,
    select_shortest_accepted,
    select_shortest_verified_teacher,
    summarize_teacher_records,
)


def test_teacher_command_default_manifest_exists():
    assert DEFAULT_MANIFEST.is_file()


def _scored(*, exact=1.0, truncated=False, completion=None, tokens=20):
    return {
        "completion": completion or "<think>Compute 6 * 7 = 42.</think>\n<answer>42</answer>",
        "exact_reward": exact,
        "truncated": truncated,
        "completion_token_count": tokens,
    }


def test_extracts_reasoning_and_builds_non_thinking_target():
    completion = "<think>First add 40 + 2 = 42.</think>\n\n<answer>42</answer>"
    assert extract_thinking_reasoning(completion) == "First add 40 + 2 = 42."
    assert build_student_target(completion, "42.0") == "First add 40 + 2 = 42.\n<answer>42</answer>"


def test_candidate_rejection_reasons_are_explicit():
    assert annotate_teacher_candidate(_scored(exact=0), "42", 10, 128)["decision_reason"] == "incorrect"
    assert annotate_teacher_candidate(_scored(truncated=True), "42", 10, 128)["decision_reason"] == "truncated"
    assert annotate_teacher_candidate(_scored(), "42", 129, 128)["decision_reason"] == "target_too_long"
    assert annotate_teacher_candidate(_scored(), "42", 20, 128)["accepted"] is True


def test_shortest_accepted_candidate_is_selected():
    longer = annotate_teacher_candidate(_scored(tokens=30), "42", 25, 128)
    shorter = annotate_teacher_candidate(_scored(tokens=40), "42", 15, 128)
    rejected = annotate_teacher_candidate(_scored(exact=0), "42", 5, 128)
    assert select_shortest_accepted([longer, rejected, shorter]) is shorter


def test_shortest_verified_teacher_can_be_over_student_limit():
    longer = annotate_teacher_candidate(_scored(), "42", 300, 128)
    shorter = annotate_teacher_candidate(_scored(), "42", 230, 128)
    assert select_shortest_verified_teacher([longer, shorter]) is shorter


def test_compression_prompt_and_candidate_keep_safe_short_format():
    messages = build_compression_messages("What is 6 * 7?", "Multiply.\n<answer>42</answer>", 128)
    assert "128 tokens" in messages[0]["content"]
    scored = _scored(completion="Multiply 6 * 7 = 42.\n<answer>42</answer>", tokens=18)
    candidate = annotate_compression_candidate(scored, "42", 128)
    assert candidate["accepted"] is True
    assert candidate["student_target_token_count"] == 18


def test_compression_rejects_thinking_tags_and_text_after_answer():
    thinking = _scored(completion="<think>x</think>\n<answer>42</answer>", tokens=10)
    trailing = _scored(completion="Work.\n<answer>42</answer> trailing", tokens=10)
    assert annotate_compression_candidate(thinking, "42", 128)["decision_reason"] == "contains_thinking_tags"
    assert annotate_compression_candidate(trailing, "42", 128)["decision_reason"] == "unsafe_target_format"


def test_compression_rejects_answer_only_or_non_reasoning_prefix():
    answer_only = _scored(completion="<answer>42</answer>", tokens=8)
    label_only = _scored(completion="Final answer: <answer>42</answer>", tokens=12)
    assert annotate_compression_candidate(answer_only, "42", 128)["decision_reason"] == "insufficient_reasoning"
    assert annotate_compression_candidate(label_only, "42", 128)["decision_reason"] == "insufficient_reasoning"


def test_summary_separates_verified_coverage_from_short_target_yield():
    too_long = annotate_teacher_candidate(_scored(), "42", 200, 128)
    accepted = annotate_teacher_candidate(_scored(), "42", 20, 128)
    records = [
        {"selected_candidate_index": None, "candidates": [too_long]},
        {"selected_candidate_index": 0, "candidates": [accepted]},
    ]
    summary = summarize_teacher_records(records)
    assert summary["verified_teacher_prompts"] == 2
    assert summary["accepted_student_target_prompts"] == 1
    assert summary["accepted_compressed_target_prompts"] == 0
    assert summary["usable_student_target_prompts"] == 1
    assert summary["verified_teacher_coverage"] == 1.0
    assert summary["accepted_student_target_yield"] == 0.5
    assert summary["usable_student_target_yield"] == 0.5
