"""Prompt formatting, answer extraction, and reward functions for the GSM8K GRPO trial.

Pure functions only — no model loading, no dataset access, no trainer state — so
they can be unit tested without downloading anything.
"""

import re

SYSTEM_PROMPT = (
    "You are a careful math tutor. Solve the problem step by step. "
    "Your response MUST end with exactly one line in this exact format, "
    "with nothing written after it:\n"
    "<answer>NUMBER</answer>\n"
    "Do not write \"Final answer\", a restated sentence, or anything else after "
    "solving the problem — the very last line of your response must be the "
    "<answer> tag and nothing more."
)

# Two worked few-shot examples, unrelated to GSM8K (no leakage risk), shown as
# actual prior turns rather than described in text. A single example wasn't
# enough for SmolLM2-135M to override its own habit of concluding with
# "Final answer:" instead of the tag (confirmed empirically — see CLAUDE.md)
# — two repetitions of the exact pattern reinforce it further.
_FEW_SHOT_QUESTION_1 = "A store has 3 boxes with 4 apples in each box. How many apples are there in total?"
_FEW_SHOT_ANSWER_1 = (
    "There are 3 boxes with 4 apples each, so the total is 3 * 4 = 12 apples.\n"
    "<answer>12</answer>"
)
_FEW_SHOT_QUESTION_2 = "Tom has 10 marbles and gives away 4 of them. How many marbles does he have left?"
_FEW_SHOT_ANSWER_2 = (
    "Tom starts with 10 marbles and gives away 4, so he has 10 - 4 = 6 marbles left.\n"
    "<answer>6</answer>"
)

# GSM8K's ground-truth `answer` field always ends with a line like "#### 42" —
# this format is fixed by the dataset and is unrelated to how we ask the model
# to format its own completions.
_GOLD_NUMBER_RE = re.compile(r"####\s*(-?[\d,]+(?:\.\d+)?)")

# The model is instructed (via SYSTEM_PROMPT) to wrap its final answer in
# <answer>...</answer> tags. re.search finds the first tag pair; a second/stray
# pair later in the completion is ignored rather than treated as an error.
_ANSWER_TAG_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)
_NUMBER_ONLY_RE = re.compile(r"^-?\d+(?:\.\d+)?$")

FORMAT_REWARD_VALUE = 0.2
ACCURACY_REWARD_VALUE = 1.0


def _normalize_captured_number(raw: str) -> str | None:
    cleaned = raw.strip().replace(",", "")
    if not _NUMBER_ONLY_RE.fullmatch(cleaned):
        return None
    return cleaned


def extract_gold_answer(answer_text: str) -> str | None:
    """Parse the ground-truth numeric answer out of GSM8K's native `#### <n>` line."""
    match = _GOLD_NUMBER_RE.search(answer_text)
    if not match:
        return None
    return match.group(1).replace(",", "").strip()


def extract_predicted_answer(completion_text: str) -> str | None:
    """Parse the model's predicted numeric answer out of `<answer>...</answer>`.

    Returns None if the tag is missing, unclosed, or its content isn't a bare
    number (e.g. "<answer>about 42</answer>" is treated as a format failure,
    not silently coerced to 42).
    """
    match = _ANSWER_TAG_RE.search(completion_text)
    if not match:
        return None
    return _normalize_captured_number(match.group(1))


def to_prompt(example: dict) -> dict:
    """Map a raw GSM8K row into the {prompt, answer} shape GRPOTrainer expects."""
    return {
        "prompt": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": _FEW_SHOT_QUESTION_1},
            {"role": "assistant", "content": _FEW_SHOT_ANSWER_1},
            {"role": "user", "content": _FEW_SHOT_QUESTION_2},
            {"role": "assistant", "content": _FEW_SHOT_ANSWER_2},
            {"role": "user", "content": example["question"]},
        ],
        "answer": extract_gold_answer(example["answer"]),
    }


def _completion_text(completion) -> str:
    if isinstance(completion, str):
        return completion
    return completion[0]["content"]


def accuracy_reward(completions, answer, **kwargs) -> list[float]:
    """1.0 if the predicted answer exactly matches gold, else 0.0.

    If the trainer supplied a `log_extra` hook (trl's GRPOTrainer does, to attach
    extra columns to its completions log), attach the extracted/gold answers so
    the spec's required per-sample fields show up in the sample logs, not just
    the aggregate reward. Absent in unit tests, where it's a no-op.
    """
    rewards = []
    predicted_answers = []
    for completion, gold in zip(completions, answer):
        predicted = extract_predicted_answer(_completion_text(completion))
        predicted_answers.append(predicted if predicted is not None else "")
        rewards.append(ACCURACY_REWARD_VALUE if predicted is not None and predicted == gold else 0.0)

    log_extra = kwargs.get("log_extra")
    if log_extra is not None:
        log_extra("extracted_answer", predicted_answers)
        log_extra("gold_answer", list(answer))

    return rewards


def format_reward(completions, **kwargs) -> list[float]:
    """0.2 if the completion has a valid numeric <answer> tag, else 0.0."""
    rewards = []
    for completion in completions:
        predicted = extract_predicted_answer(_completion_text(completion))
        rewards.append(FORMAT_REWARD_VALUE if predicted is not None else 0.0)
    return rewards
