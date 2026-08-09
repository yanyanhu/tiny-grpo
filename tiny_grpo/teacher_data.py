"""Pure processing for verified thinking-teacher SFT candidates."""

import re

from tiny_grpo.rewards import extract_predicted_answer, normalize_numeric_answer

_THINKING_BLOCK_RE = re.compile(r"<think>\s*(.*?)\s*</think>", re.DOTALL)
_NUMBER_RE = re.compile(r"[-+]?\d+(?:[.,]\d+)*")


def extract_thinking_reasoning(completion: str) -> str | None:
    """Extract a non-empty reasoning block without retaining thinking tags."""
    match = _THINKING_BLOCK_RE.search(completion)
    if match is None:
        return None
    reasoning = match.group(1).strip()
    return reasoning or None


def build_student_target(completion: str, gold_answer: str) -> str | None:
    """Build the uncompressed short-policy target from a thinking completion."""
    reasoning = extract_thinking_reasoning(completion)
    canonical_gold = normalize_numeric_answer(gold_answer)
    if reasoning is None or canonical_gold is None:
        return None
    return f"{reasoning}\n<answer>{canonical_gold}</answer>"


def annotate_teacher_candidate(scored: dict, gold_answer: str, student_target_token_count: int | None,
                               max_student_target_tokens: int) -> dict:
    """Attach target provenance and one explicit acceptance/rejection reason."""
    target = build_student_target(scored["completion"], gold_answer)
    if scored["exact_reward"] <= 0:
        reason = "incorrect"
    elif scored["truncated"]:
        reason = "truncated"
    elif target is None:
        reason = "missing_thinking_reasoning"
    elif student_target_token_count is None:
        reason = "target_not_tokenized"
    elif student_target_token_count > max_student_target_tokens:
        reason = "target_too_long"
    else:
        reason = "accepted"
    return {
        **scored,
        "student_target": target,
        "student_target_token_count": student_target_token_count,
        "accepted": reason == "accepted",
        "decision_reason": reason,
    }


def select_shortest_accepted(candidates: list[dict]) -> dict | None:
    accepted = [candidate for candidate in candidates if candidate["accepted"]]
    if not accepted:
        return None
    return min(
        accepted,
        key=lambda candidate: (
            candidate["student_target_token_count"],
            candidate["completion_token_count"],
        ),
    )


def select_shortest_verified_teacher(candidates: list[dict]) -> dict | None:
    verified = [
        candidate for candidate in candidates
        if candidate["exact_reward"] > 0 and not candidate["truncated"] and candidate["student_target"]
    ]
    if not verified:
        return None
    return min(verified, key=lambda candidate: candidate["student_target_token_count"])


def build_compression_messages(question: str, verified_teacher_target: str,
                               max_student_target_tokens: int) -> list[dict]:
    return [
        {
            "role": "system",
            "content": (
                "Compress the verified math solution into concise, complete reasoning. "
                f"The entire response must fit within {max_student_target_tokens} tokens. "
                "Preserve the necessary arithmetic, do not use <think> tags, and end with "
                "exactly one <answer>NUMBER</answer> line with nothing after it. Include "
                "concise reasoning before the answer; an answer-only response will be rejected."
            ),
        },
        {
            "role": "user",
            "content": f"Problem:\n{question}\n\nVerified solution:\n{verified_teacher_target}",
        },
    ]


def annotate_compression_candidate(scored: dict, gold_answer: str,
                                   max_student_target_tokens: int) -> dict:
    completion = scored["completion"].strip()
    safe_ending = completion.endswith("</answer>") and completion.count("<answer>") == 1
    reasoning = completion.rsplit("<answer>", 1)[0].strip()
    useful_reasoning = len(reasoning.split()) >= 6 and len(_NUMBER_RE.findall(reasoning)) >= 2
    if scored["exact_reward"] <= 0:
        reason = "incorrect"
    elif scored["truncated"]:
        reason = "truncated"
    elif "<think>" in completion or "</think>" in completion:
        reason = "contains_thinking_tags"
    elif not safe_ending or extract_predicted_answer(completion) != normalize_numeric_answer(gold_answer):
        reason = "unsafe_target_format"
    elif not useful_reasoning:
        reason = "insufficient_reasoning"
    elif scored["completion_token_count"] > max_student_target_tokens:
        reason = "target_too_long"
    else:
        reason = "accepted"
    return {
        **scored,
        "student_target": completion,
        "student_target_token_count": scored["completion_token_count"],
        "accepted": reason == "accepted",
        "decision_reason": reason,
    }


def summarize_teacher_records(records: list[dict]) -> dict:
    candidates = [candidate for record in records for candidate in record["candidates"]]
    compressed = [candidate for record in records for candidate in record.get("compression_candidates", [])]
    reasons = {}
    for candidate in candidates:
        reason = candidate["decision_reason"]
        reasons[reason] = reasons.get(reason, 0) + 1
    verified_prompts = sum(any(c["exact_reward"] > 0 and not c["truncated"] for c in r["candidates"]) for r in records)
    accepted_prompts = sum(record["selected_candidate_index"] is not None for record in records)
    compressed_reasons = {}
    for candidate in compressed:
        reason = candidate["decision_reason"]
        compressed_reasons[reason] = compressed_reasons.get(reason, 0) + 1
    compressed_prompts = sum(record.get("selected_compression_index") is not None for record in records)
    usable_prompts = sum(
        record["selected_candidate_index"] is not None
        or record.get("selected_compression_index") is not None
        for record in records
    )
    return {
        "num_prompts": len(records),
        "num_candidates": len(candidates),
        "verified_teacher_prompts": verified_prompts,
        "accepted_student_target_prompts": accepted_prompts,
        "accepted_compressed_target_prompts": compressed_prompts,
        "usable_student_target_prompts": usable_prompts,
        "verified_teacher_coverage": verified_prompts / len(records) if records else 0.0,
        "accepted_student_target_yield": accepted_prompts / len(records) if records else 0.0,
        "usable_student_target_yield": usable_prompts / len(records) if records else 0.0,
        "candidate_decision_reasons": reasons,
        "compression_decision_reasons": compressed_reasons,
    }
