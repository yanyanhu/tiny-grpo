"""Deterministic GSM8K target construction for completion-only SFT."""

import re
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


def to_sft_example(example: dict) -> dict:
    prompt = to_prompt(example)["prompt"]
    return {
        "prompt": prompt,
        "completion": [{"role": "assistant", "content": build_sft_target(example["answer"])}],
    }


def audit_sft_lengths(dataset, tokenizer, max_sequence_length: int) -> dict:
    lengths = []
    for example in dataset:
        conversation = example["prompt"] + example["completion"]
        encoded = tokenizer.apply_chat_template(conversation, tokenize=True)
        token_ids = encoded["input_ids"] if isinstance(encoded, Mapping) else encoded
        if token_ids and isinstance(token_ids[0], list):
            if len(token_ids) != 1:
                raise SFTTargetError("length audit expected one conversation at a time")
            token_ids = token_ids[0]
        lengths.append(len(token_ids))
    too_long = sum(length > max_sequence_length for length in lengths)
    result = {
        "num_examples": len(lengths),
        "min_tokens": min(lengths),
        "max_tokens": max(lengths),
        "mean_tokens": sum(lengths) / len(lengths),
        "max_sequence_length": max_sequence_length,
        "num_over_limit": too_long,
    }
    if too_long:
        raise SFTTargetError(
            f"{too_long} SFT examples exceed max_sequence_length={max_sequence_length}; "
            "refusing silent truncation of supervised answers"
        )
    return result
