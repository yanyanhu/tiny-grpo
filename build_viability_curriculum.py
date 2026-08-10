"""Classify training rollouts and persist a deterministic GRPO curriculum."""

import argparse
import json
import random
from pathlib import Path

from tiny_grpo.viability_curriculum import classify_viability_group, select_curriculum


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rollout-run", type=Path, required=True)
    parser.add_argument("--verified-teacher-data", type=Path, required=True)
    parser.add_argument("--curriculum-size", type=int, default=256)
    parser.add_argument("--selection-seed", type=int, default=31415)
    args = parser.parse_args()
    groups = _read_jsonl(args.rollout_run / "prompt_results.jsonl")
    teacher_rows = _read_jsonl(args.verified_teacher_data)
    verified_ids = {row["prompt_id"] for row in teacher_rows}
    records = [classify_viability_group(group, verified_ids) for group in groups]
    selected_ids, summary = select_curriculum(records, args.curriculum_size, args.selection_seed)
    with (args.rollout_run / "viability_records.jsonl").open("w") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")
    manifest = {
        "kind": "grpo_viability_curriculum",
        "rollout_run": str(args.rollout_run),
        "verified_teacher_data": str(args.verified_teacher_data),
        "criteria": {
            "high_signal": "1-3 exact of 4",
            "frontier": "0 exact, >=3 terminated, >=3 valid, verified teacher target exists",
            "low_value": ">=3 truncated or <2 valid",
            "stable": "4 exact of 4",
        },
        "selected_prompt_ids": selected_ids,
        **summary,
    }
    (args.rollout_run / "curriculum_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    control_ids = [record["prompt_id"] for record in records]
    random.Random(args.selection_seed).shuffle(control_ids)
    control_manifest = {
        "kind": "ordinary_deterministic_control",
        "source_pool": str(args.rollout_run),
        "selection_seed": args.selection_seed,
        "selected_prompt_ids": control_ids[:args.curriculum_size],
    }
    (args.rollout_run / "ordinary_control_manifest.json").write_text(
        json.dumps(control_manifest, indent=2) + "\n"
    )
    print(json.dumps({k: v for k, v in manifest.items() if k != "selected_prompt_ids"}, indent=2))


if __name__ == "__main__":
    main()
