"""CPU-only tests for rollout diagnostic scoring and aggregation."""

import pytest

from tiny_grpo.diagnose_rollouts import chat_template_kwargs
from tiny_grpo.diagnostics import (
    aggregate_rollout_groups,
    build_prompt_record,
    classify_completion,
    score_rollout_completion,
    wilson_interval,
)


def test_chat_template_mode_kwargs_are_explicit():
    assert chat_template_kwargs("default") == {}
    assert chat_template_kwargs("thinking") == {"enable_thinking": True}
    assert chat_template_kwargs("non-thinking") == {"enable_thinking": False}


def test_chat_template_mode_rejects_unknown_value():
    with pytest.raises(ValueError, match="unknown chat template mode"):
        chat_template_kwargs("sometimes")


def _score(text: str, gold: str = "42", tokens: int = 5, cap: int = 128) -> dict:
    return score_rollout_completion(
        text,
        gold,
        completion_token_count=tokens,
        max_completion_length=cap,
        terminated=tokens < cap,
    )


class TestFailureClassification:
    def test_exact_correct(self):
        assert _score("<answer>42.0</answer>")["failure_category"] == "exact_correct"

    def test_valid_incorrect(self):
        assert _score("<answer>41</answer>")["failure_category"] == "valid_incorrect"

    def test_missing_tag(self):
        assert _score("The answer is 42.")["failure_category"] == "missing_answer_tag"

    def test_malformed_tag(self):
        assert _score("<answer>forty two</answer>")["failure_category"] == "malformed_answer_tag"

    def test_length_cap_takes_precedence_for_invalid_output(self):
        assert _score("unfinished reasoning", tokens=128)["failure_category"] == "truncated_invalid"

    def test_direct_classifier_matches_scored_path(self):
        assert (
            classify_completion(
                "<answer>41</answer>",
                exact_reward=0.0,
                completion_token_count=5,
                max_completion_length=128,
                terminated=True,
            )
            == "valid_incorrect"
        )


class TestWilsonInterval:
    def test_zero_successes_still_has_nonzero_upper_bound(self):
        interval = wilson_interval(0, 200)
        assert interval["low"] == 0.0
        assert 0.0 < interval["high"] < 0.03

    def test_all_successes_still_has_lower_uncertainty(self):
        interval = wilson_interval(20, 20)
        assert 0.8 < interval["low"] < 1.0
        assert interval["high"] == 1.0

    @pytest.mark.parametrize("successes,total", [(-1, 10), (11, 10), (0, 0)])
    def test_invalid_counts_raise(self, successes, total):
        with pytest.raises(ValueError):
            wilson_interval(successes, total)


class TestAggregateRolloutGroups:
    def test_pass_rates_and_group_sparsity(self):
        groups = [
            build_prompt_record(
                10,
                "p1",
                "42",
                [_score("<answer>42</answer>"), _score("<answer>41</answer>")],
            ),
            build_prompt_record(
                20,
                "p2",
                "42",
                [_score("no tag"), _score("<answer>42.0</answer>")],
            ),
            build_prompt_record(
                30,
                "p3",
                "42",
                [_score("<answer>0</answer>"), _score("no tag")],
            ),
        ]

        result = aggregate_rollout_groups(groups)

        assert result["num_prompts"] == 3
        assert result["num_generations"] == 2
        assert result["total_completions"] == 6
        assert result["pass_at_1"] == pytest.approx(1 / 3)
        assert result["pass_at_2"] == pytest.approx(2 / 3)
        assert result["sample_exact_accuracy"] == pytest.approx(2 / 6)
        assert result["mixed_exact_reward_group_rate"] == pytest.approx(2 / 3)
        assert result["zero_exact_reward_std_group_rate"] == pytest.approx(1 / 3)
        assert result["all_completions_wrong_group_rate"] == pytest.approx(1 / 3)
        assert result["counts"]["exact_completions"] == 2
        assert result["counts"]["failure_categories"]["exact_correct"] == 2

    def test_total_reward_variance_can_exist_without_exact_variance(self):
        group = build_prompt_record(
            1,
            "p",
            "42",
            [_score("<answer>41</answer>"), _score("no tag")],
        )
        result = aggregate_rollout_groups([group])
        assert result["zero_exact_reward_std_group_rate"] == 1.0
        assert result["zero_total_reward_std_group_rate"] == 0.0

    def test_truncation_and_lengths(self):
        group = build_prompt_record(
            1,
            "p",
            "42",
            [_score("unfinished", tokens=128), _score("<answer>0</answer>", tokens=10)],
        )
        result = aggregate_rollout_groups([group])
        assert result["truncation_rate"] == 0.5
        assert result["counts"]["truncated_completions"] == 1
        assert result["average_completion_length"] == 69
        assert result["maximum_completion_length"] == 128

    def test_empty_or_ragged_groups_raise(self):
        with pytest.raises(ValueError):
            aggregate_rollout_groups([])
        group_a = build_prompt_record(1, "p", "1", [_score("<answer>1</answer>", gold="1")])
        group_b = build_prompt_record(
            2,
            "p",
            "1",
            [_score("<answer>1</answer>", gold="1"), _score("<answer>0</answer>", gold="1")],
        )
        with pytest.raises(ValueError):
            aggregate_rollout_groups([group_a, group_b])
