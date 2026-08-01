"""CPU-only unit tests for tiny_grpo.evaluate's pure parts. No model access."""

import pytest

from tiny_grpo.evaluate import _score_example, aggregate_eval_records


class TestScoreExample:
    def test_correct_answer(self):
        record = _score_example("prompt", "42", "<answer>42</answer>", completion_token_count=5)
        assert record["extracted_answer"] == "42"
        assert record["accuracy_reward"] == 1.0
        assert record["format_reward"] == 0.2
        assert record["total_reward"] == 1.2
        assert record["completion_token_count"] == 5

    def test_wrong_answer(self):
        record = _score_example("prompt", "42", "<answer>41</answer>", completion_token_count=5)
        assert record["accuracy_reward"] == 0.0
        assert record["format_reward"] == 0.2
        assert record["total_reward"] == 0.2

    def test_missing_tag(self):
        record = _score_example("prompt", "42", "no tag here", completion_token_count=5)
        assert record["extracted_answer"] is None
        assert record["accuracy_reward"] == 0.0
        assert record["format_reward"] == 0.0
        assert record["total_reward"] == 0.0

    def test_carries_prompt_and_gold_through(self):
        record = _score_example("What is 2+2?", "4", "<answer>4</answer>", completion_token_count=3)
        assert record["prompt"] == "What is 2+2?"
        assert record["gold_answer"] == "4"
        assert record["completion"] == "<answer>4</answer>"


def _record(accuracy, fmt, extracted, token_count=10):
    return {
        "accuracy_reward": accuracy,
        "format_reward": fmt,
        "total_reward": accuracy + fmt,
        "extracted_answer": extracted,
        "completion_token_count": token_count,
    }


class TestAggregateEvalRecords:
    def test_all_correct(self):
        records = [_record(1.0, 0.2, "42") for _ in range(4)]
        result = aggregate_eval_records(records)
        assert result["num_examples"] == 4
        assert result["accuracy"] == 1.0
        assert result["format_rate"] == 1.0
        assert result["parse_failure_rate"] == 0.0
        assert result["mean_reward"] == pytest.approx(1.2)

    def test_all_wrong_but_well_formatted(self):
        records = [_record(0.0, 0.2, "41") for _ in range(4)]
        result = aggregate_eval_records(records)
        assert result["accuracy"] == 0.0
        assert result["format_rate"] == 1.0
        assert result["parse_failure_rate"] == 0.0

    def test_all_parse_failures(self):
        records = [_record(0.0, 0.0, None) for _ in range(4)]
        result = aggregate_eval_records(records)
        assert result["accuracy"] == 0.0
        assert result["format_rate"] == 0.0
        assert result["parse_failure_rate"] == 1.0

    def test_mixed(self):
        records = [
            _record(1.0, 0.2, "42"),  # correct
            _record(0.0, 0.2, "41"),  # wrong but formatted
            _record(0.0, 0.0, None),  # parse failure
            _record(0.0, 0.0, None),  # parse failure
        ]
        result = aggregate_eval_records(records)
        assert result["num_examples"] == 4
        assert result["accuracy"] == pytest.approx(0.25)
        assert result["format_rate"] == pytest.approx(0.5)
        assert result["parse_failure_rate"] == pytest.approx(0.5)

    def test_mean_completion_length(self):
        records = [_record(1.0, 0.2, "42", token_count=t) for t in (10, 20, 30)]
        result = aggregate_eval_records(records)
        assert result["mean_completion_length"] == pytest.approx(20.0)

    def test_empty_records_raises(self):
        with pytest.raises(ValueError):
            aggregate_eval_records([])
