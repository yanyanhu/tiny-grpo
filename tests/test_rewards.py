"""CPU-only unit tests for tiny_grpo.rewards. No model or dataset access."""

import pytest

from tiny_grpo.rewards import (
    ACCURACY_REWARD_VALUE,
    FORMAT_REWARD_VALUE,
    accuracy_reward,
    extract_gold_answer,
    extract_predicted_answer,
    format_reward,
    normalize_numeric_answer,
    to_prompt,
)


class TestNormalizeNumericAnswer:
    @pytest.mark.parametrize("raw", ["42", "42.0", "042", "42.000", "0042.000"])
    def test_equivalent_integer_spellings(self, raw):
        assert normalize_numeric_answer(raw) == "42"

    @pytest.mark.parametrize("raw", ["0", "0.0", "-0", "-0.000"])
    def test_signed_zero(self, raw):
        assert normalize_numeric_answer(raw) == "0"

    def test_decimal_commas_and_negative_values(self):
        assert normalize_numeric_answer("1,234.500") == "1234.5"
        assert normalize_numeric_answer("-002.500") == "-2.5"

    @pytest.mark.parametrize("raw", ["NaN", "inf", "1e3", "1/2", "about 4", ""])
    def test_rejects_non_bare_finite_decimal(self, raw):
        assert normalize_numeric_answer(raw) is None


def _completion(text: str):
    return [{"role": "assistant", "content": text}]


class TestExtractGoldAnswer:
    def test_basic(self):
        assert extract_gold_answer("She has 3 apples.\n#### 3") == "3"

    def test_strips_thousands_comma(self):
        assert extract_gold_answer("Total cost.\n#### 1,234") == "1234"

    def test_negative(self):
        assert extract_gold_answer("Net change.\n#### -5") == "-5"

    def test_decimal(self):
        assert extract_gold_answer("Average.\n#### 3.5") == "3.5"

    def test_missing_marker_returns_none(self):
        assert extract_gold_answer("No marker here at all.") is None


class TestExtractPredictedAnswer:
    def test_basic(self):
        assert extract_predicted_answer("Step 1...\n<answer>42</answer>") == "42"

    def test_surrounding_text_ignored(self):
        text = "Reasoning here.\n<answer>7</answer>\nDone."
        assert extract_predicted_answer(text) == "7"

    def test_strips_thousands_comma(self):
        assert extract_predicted_answer("<answer>1,234</answer>") == "1234"

    def test_negative_and_decimal(self):
        assert extract_predicted_answer("<answer>-2.5</answer>") == "-2.5"

    def test_missing_tag_returns_none(self):
        assert extract_predicted_answer("The answer is 42.") is None

    def test_unclosed_tag_returns_none(self):
        assert extract_predicted_answer("<answer>42") is None

    def test_empty_tag_returns_none(self):
        assert extract_predicted_answer("<answer></answer>") is None

    def test_non_numeric_content_returns_none(self):
        assert extract_predicted_answer("<answer>about 42</answer>") is None

    def test_first_of_multiple_tags_wins(self):
        text = "<answer>1</answer> ... <answer>2</answer>"
        assert extract_predicted_answer(text) == "1"


class TestAccuracyReward:
    def test_correct_match(self):
        completions = [_completion("<answer>42</answer>")]
        rewards = accuracy_reward(completions=completions, answer=["42"])
        assert rewards == [ACCURACY_REWARD_VALUE]

    def test_incorrect_match(self):
        completions = [_completion("<answer>41</answer>")]
        rewards = accuracy_reward(completions=completions, answer=["42"])
        assert rewards == [0.0]

    @pytest.mark.parametrize("prediction", ["42.0", "042", "42.000"])
    def test_numeric_representation_differences_still_match(self, prediction):
        completions = [_completion(f"<answer>{prediction}</answer>")]
        rewards = accuracy_reward(completions=completions, answer=["42"])
        assert rewards == [ACCURACY_REWARD_VALUE]

    def test_missing_tag_is_zero(self):
        completions = [_completion("no tag here")]
        rewards = accuracy_reward(completions=completions, answer=["42"])
        assert rewards == [0.0]

    def test_batch(self):
        completions = [
            _completion("<answer>42</answer>"),
            _completion("<answer>0</answer>"),
        ]
        rewards = accuracy_reward(completions=completions, answer=["42", "42"])
        assert rewards == [ACCURACY_REWARD_VALUE, 0.0]

    def test_works_without_log_extra_hook(self):
        # No log_extra kwarg supplied (as in these unit tests) must not raise.
        completions = [_completion("<answer>42</answer>")]
        rewards = accuracy_reward(completions=completions, answer=["42"])
        assert rewards == [ACCURACY_REWARD_VALUE]

    def test_calls_log_extra_with_extracted_and_gold_answers(self):
        logged = {}

        def fake_log_extra(column, values):
            logged[column] = values

        completions = [
            _completion("<answer>42</answer>"),
            _completion("no tag here"),
        ]
        accuracy_reward(completions=completions, answer=["42", "7"], log_extra=fake_log_extra)

        assert logged["extracted_answer"] == ["42", ""]
        assert logged["gold_answer"] == ["42", "7"]


class TestFormatReward:
    def test_valid_tag(self):
        rewards = format_reward(completions=[_completion("<answer>42</answer>")])
        assert rewards == [FORMAT_REWARD_VALUE]

    def test_missing_tag(self):
        rewards = format_reward(completions=[_completion("no tag here")])
        assert rewards == [0.0]

    def test_malformed_tag_content(self):
        rewards = format_reward(completions=[_completion("<answer>forty-two</answer>")])
        assert rewards == [0.0]

    def test_unclosed_tag(self):
        rewards = format_reward(completions=[_completion("<answer>42")])
        assert rewards == [0.0]


class TestToPrompt:
    def test_shape_and_gold_extraction(self):
        example = {"question": "What is 2+2?", "answer": "Add them.\n#### 4"}
        result = to_prompt(example)
        assert result["answer"] == "4"
        assert result["prompt"][0]["role"] == "system"
        # Two few-shot examples (user+assistant pairs) come before the real question.
        roles = [m["role"] for m in result["prompt"]]
        assert roles == ["system", "user", "assistant", "user", "assistant", "user"]
        assert "<answer>" in result["prompt"][2]["content"]
        assert "<answer>" in result["prompt"][4]["content"]
        assert result["prompt"][-1] == {"role": "user", "content": "What is 2+2?"}

    def test_never_leaks_gold_answer_into_prompt(self):
        example = {"question": "What is seven times six?", "answer": "Multiply.\n#### 42"}
        result = to_prompt(example)
        prompt_text = " ".join(m["content"] for m in result["prompt"])
        assert "42" not in prompt_text
        assert "####" not in prompt_text


@pytest.mark.parametrize(
    "completion_text",
    [
        "",
        "<answer>",
        "</answer>",
        "<answer><answer>1</answer>",
        "<answer>1.2.3</answer>",
        "<answer>  </answer>",
    ],
)
def test_malformed_completions_yield_none(completion_text):
    assert extract_predicted_answer(completion_text) is None
