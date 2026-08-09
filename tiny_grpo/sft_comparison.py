"""Build prompt-matched gold-short and teacher-distilled SFT rows."""

from tiny_grpo.rewards import to_prompt
from tiny_grpo.sft_data import SFTTargetError, build_sft_target


def target_token_count(tokenizer, target: str) -> int:
    return len(tokenizer(target, add_special_tokens=False)["input_ids"])


def build_matched_sft_rows(teacher_rows: list[dict], train_pool, tokenizer,
                           max_target_tokens: int, chat_template_kwargs: dict) -> tuple[list, list, dict]:
    """Intersect accepted teacher rows with gold targets fitting the same cap."""
    seen = set()
    gold_rows = []
    distilled_rows = []
    dropped = {}
    target_lengths = {"gold_short": [], "teacher_distilled": []}
    for teacher_row in teacher_rows:
        prompt_id = teacher_row["prompt_id"]
        if prompt_id in seen:
            raise ValueError(f"duplicate teacher prompt_id {prompt_id}")
        seen.add(prompt_id)
        raw = train_pool[prompt_id]
        try:
            gold_target = build_sft_target(raw["answer"])
        except SFTTargetError:
            dropped["invalid_gold_target"] = dropped.get("invalid_gold_target", 0) + 1
            continue
        teacher_target = teacher_row["completion"][0]["content"]
        gold_tokens = target_token_count(tokenizer, gold_target)
        teacher_tokens = target_token_count(tokenizer, teacher_target)
        if gold_tokens > max_target_tokens:
            dropped["gold_target_too_long"] = dropped.get("gold_target_too_long", 0) + 1
            continue
        if teacher_tokens > max_target_tokens:
            dropped["teacher_target_too_long"] = dropped.get("teacher_target_too_long", 0) + 1
            continue
        prompt = to_prompt(raw)["prompt"]
        common = {"prompt_id": prompt_id, "prompt": prompt,
                  "chat_template_kwargs": dict(chat_template_kwargs)}
        gold_rows.append({**common, "completion": [{"role": "assistant", "content": gold_target}]})
        distilled_rows.append({
            **common,
            "completion": [{"role": "assistant", "content": teacher_target}],
        })
        target_lengths["gold_short"].append(gold_tokens)
        target_lengths["teacher_distilled"].append(teacher_tokens)
    summary = {
        "teacher_input_examples": len(teacher_rows),
        "matched_examples": len(gold_rows),
        "matched_prompt_ids": [row["prompt_id"] for row in gold_rows],
        "dropped_reasons": dropped,
        "target_token_lengths": target_lengths,
    }
    return gold_rows, distilled_rows, summary
