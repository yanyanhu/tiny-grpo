"""Deterministic GSM8K target construction for completion-only SFT."""

import re
import statistics
from collections.abc import Mapping

from tiny_grpo.rewards import extract_gold_answer, to_prompt

_CALCULATOR_ANNOTATION_RE = re.compile(r"<<[^<>]*>>")
_FINAL_ANSWER_LINE_RE = re.compile(r"(?m)^####\s*[^\n]*\s*$")


class SFTTargetError(ValueError):
    """Raised when a GSM8K row cannot produce a safe supervised target."""


def build_sft_target(answer_text: str) -> str:
    gold = extract_gold_answer(answer_text)
    if gold is None:
        raise SFTTargetError("GSM8K answer is missing a valid '#### NUMBER' marker")
    reasoning, replacements = _FINAL_ANSWER_LINE_RE.subn("", answer_text, count=1)
    if replacements != 1:
        raise SFTTargetError("GSM8K answer is missing its final answer line")
    reasoning = _CALCULATOR_ANNOTATION_RE.sub("", reasoning).strip()
    if not reasoning:
        raise SFTTargetError("GSM8K answer has no reasoning before its final answer")
    return f"{reasoning}\n<answer>{gold}</answer>"


def to_sft_example(example: dict, chat_template_kwargs: dict | None = None) -> dict:
    prompt = to_prompt(example)["prompt"]
    result = {
        "prompt": prompt,
        "completion": [{"role": "assistant", "content": build_sft_target(example["answer"])}],
    }
    if chat_template_kwargs:
        result["chat_template_kwargs"] = dict(chat_template_kwargs)
    return result


def _token_ids(encoded):
    token_ids = encoded["input_ids"] if isinstance(encoded, Mapping) else encoded
    if token_ids and isinstance(token_ids[0], list):
        if len(token_ids) != 1:
            raise SFTTargetError("length audit expected one conversation at a time")
        token_ids = token_ids[0]
    return token_ids


def _length_summary(lengths: list[int]) -> dict:
    ordered = sorted(lengths)

    def percentile(fraction: float) -> int:
        index = max(0, min(len(ordered) - 1, int((len(ordered) * fraction) + 0.999999) - 1))
        return ordered[index]

    return {
        "min": ordered[0],
        "mean": statistics.fmean(ordered),
        "p50": percentile(0.50),
        "p95": percentile(0.95),
        "max": ordered[-1],
    }


def audit_sft_lengths(dataset, tokenizer, max_sequence_length: int,
                      max_completion_length: int | None = None) -> dict:
    total_lengths = []
    prompt_lengths = []
    assistant_lengths = []
    for example in dataset:
        template_kwargs = example.get("chat_template_kwargs", {})
        conversation = example["prompt"] + example["completion"]
        total_ids = _token_ids(tokenizer.apply_chat_template(
            conversation,
            tokenize=True,
            **template_kwargs,
        ))
        prompt_ids = _token_ids(tokenizer.apply_chat_template(
            example["prompt"],
            tokenize=True,
            add_generation_prompt=True,
            **template_kwargs,
        ))
        total_lengths.append(len(total_ids))
        prompt_lengths.append(len(prompt_ids))
        assistant_lengths.append(max(0, len(total_ids) - len(prompt_ids)))
    too_long = sum(length > max_sequence_length for length in total_lengths)
    assistant_over_limit = (
        sum(length > max_completion_length for length in assistant_lengths)
        if max_completion_length is not None else None
    )
    result = {
        "num_examples": len(total_lengths),
        "min_tokens": min(total_lengths),
        "max_tokens": max(total_lengths),
        "mean_tokens": statistics.fmean(total_lengths),
        "max_sequence_length": max_sequence_length,
        "num_over_limit": too_long,
        "prompt_tokens": _length_summary(prompt_lengths),
        "assistant_tokens": _length_summary(assistant_lengths),
        "total_tokens": _length_summary(total_lengths),
    }
    if max_completion_length is not None:
        result["max_completion_length"] = max_completion_length
        result["num_assistant_over_completion_limit"] = assistant_over_limit
    if too_long:
        raise SFTTargetError(
            f"{too_long} SFT examples exceed max_sequence_length={max_sequence_length}; "
            "refusing silent truncation of supervised answers"
        )
    return result
