"""Tests for matched SFT comparison dataset construction."""

import pytest

from tiny_grpo.sft_comparison import build_matched_sft_rows


class _Tokenizer:
    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": text.split()}


def _teacher(prompt_id, target):
    return {
        "prompt_id": prompt_id,
        "completion": [{"role": "assistant", "content": target}],
    }


def test_comparison_uses_identical_prompts_and_only_changes_targets():
    pool = [{"question": "What is 2 + 3?", "answer": "Add 2 and 3.\n#### 5"}]
    teacher = [_teacher(0, "Compute 2 + 3 = 5.\n<answer>5</answer>")]
    gold, distilled, summary = build_matched_sft_rows(
        teacher, pool, _Tokenizer(), 128, {"enable_thinking": False}
    )
    assert len(gold) == len(distilled) == 1
    assert gold[0]["prompt_id"] == distilled[0]["prompt_id"] == 0
    assert gold[0]["prompt"] == distilled[0]["prompt"]
    assert gold[0]["completion"] != distilled[0]["completion"]
    assert summary["matched_prompt_ids"] == [0]


def test_comparison_drops_pair_if_either_target_exceeds_cap():
    pool = [{"question": "Q?", "answer": "one two three four\n#### 5"}]
    teacher = [_teacher(0, "short reasoning\n<answer>5</answer>")]
    gold, distilled, summary = build_matched_sft_rows(
        teacher, pool, _Tokenizer(), 3, {"enable_thinking": False}
    )
    assert gold == distilled == []
    assert summary["dropped_reasons"] == {"gold_target_too_long": 1}


def test_comparison_rejects_duplicate_teacher_prompt_ids():
    pool = [{"question": "Q?", "answer": "Reason.\n#### 5"}]
    teacher = [_teacher(0, "Work 2 + 3 = 5.\n<answer>5</answer>")] * 2
    with pytest.raises(ValueError, match="duplicate"):
        build_matched_sft_rows(teacher, pool, _Tokenizer(), 128, {})
