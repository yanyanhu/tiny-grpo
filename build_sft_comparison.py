"""Create prompt-matched gold-short and thinking-distilled SFT JSONL files."""

import argparse
import datetime
import json
from pathlib import Path

from datasets import load_dataset
from transformers import AutoTokenizer

from tiny_grpo.model_profiles import chat_template_kwargs, resolve_model_profile
from tiny_grpo.sft_comparison import build_matched_sft_rows
from tiny_grpo.splits import assert_disjoint, build_split_metadata, load_diagnostic_manifest

DEFAULT_MANIFEST = Path(__file__).resolve().parent / "data" / "diagnostic_manifest_v1.json"


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--teacher-run", type=Path, required=True)
    parser.add_argument("--model-profile", default="qwen3_0_6b")
    parser.add_argument("--max-target-tokens", type=int, default=128)
    parser.add_argument("--train-pool-size", type=int, default=256)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    args = parser.parse_args()
    if args.max_target_tokens < 1 or args.train_pool_size < 1:
        parser.error("token cap and train pool size must be >= 1")

    accepted_path = args.teacher_run / "accepted_sft.jsonl"
    if not accepted_path.is_file():
        parser.error(f"missing accepted teacher data: {accepted_path}")
    teacher_rows = _read_jsonl(accepted_path)
    model_profile = resolve_model_profile(args.model_profile)
    if model_profile.name != "qwen3_0_6b":
        parser.error("the matched comparison currently requires qwen3_0_6b")
    train_pool = load_dataset("openai/gsm8k", "main", split="train")
    test_pool = load_dataset("openai/gsm8k", "main", split="test")
    split = build_split_metadata(
        len(train_pool), len(test_pool), args.train_pool_size, 32, 64, args.split_seed
    )
    accepted_ids = [row["prompt_id"] for row in teacher_rows]
    if not set(accepted_ids).issubset(split.train_indices):
        parser.error("teacher data contains prompt IDs outside the configured training split")
    manifest = load_diagnostic_manifest(args.manifest)
    assert_disjoint(accepted_ids, split.val_indices)
    assert_disjoint(accepted_ids, manifest.diagnostic_indices)

    tokenizer = AutoTokenizer.from_pretrained(model_profile.model_id)
    gold, distilled, summary = build_matched_sft_rows(
        teacher_rows,
        train_pool,
        tokenizer,
        args.max_target_tokens,
        chat_template_kwargs("non-thinking"),
    )
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output = args.output_dir / f"sft_comparison_data_{stamp}"
    output.mkdir(parents=True, exist_ok=False)
    _write_jsonl(output / "gold_short.jsonl", gold)
    _write_jsonl(output / "teacher_distilled.jsonl", distilled)
    provenance = {
        "teacher_run": str(args.teacher_run),
        "model_profile": model_profile.name,
        "student_mode": "non-thinking",
        "max_target_tokens": args.max_target_tokens,
        "train_pool_size": args.train_pool_size,
        "split_seed": args.split_seed,
        "canonical_manifest": str(args.manifest),
        **summary,
    }
    (output / "comparison_manifest.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(f"Created matched SFT comparison data: {output}")
    print(json.dumps({k: v for k, v in provenance.items() if k != "target_token_lengths"}, indent=2))


if __name__ == "__main__":
    main()
