# grpo-test

Small, understandable GRPO training project: fine-tune
`HuggingFaceTB/SmolLM2-135M-Instruct` on GSM8K with TRL's `GRPOTrainer`, targeting
an Apple M2 MacBook Pro (MPS backend, 16GB unified memory). See `CLAUDE.md` and
`docs/PROJECT_SPEC.md` for the full design and current milestone.

## Running

```sh
uv run python train_grpo.py --profile smoke   # default; also: debug, longer
```

or, to run in the background with a watchdog that kills the process if it hangs or
runs too long:

```sh
./run_with_watchdog.sh
```

`longer` is a bigger, slower profile — start it deliberately, never as routine
verification.

## Output layout

Every invocation creates a fresh, uniquely-named directory under `outputs/`
(e.g. `outputs/smoke_20260801_145837/`) — a run is never silently overwritten.
Each run directory contains:

- `config.json` — the fully resolved `TrainingConfig` used for that run.
- `environment.json` — pinned package versions (`torch`, `transformers`, `trl`,
  `accelerate`, `datasets`, `peft`) and Python/platform info.
- `split_metadata.json` — the exact train/validation/test indices and seed
  selected from GSM8K for that run (see `tiny_grpo/splits.py`).
- `metrics.jsonl` — every `trainer.log()` call (train and eval), one JSON
  object per line, including process and MPS memory at that point.
- `completions/completions_*.parquet` — sampled (prompt, completion, reward
  breakdown, extracted answer, gold answer, advantage) rows per logging step.
- `checkpoint-*/` — model/optimizer/scheduler state; capped at the 2 most
  recent per run (`checkpoint_retention` in config, hard-capped at 2).
- `tensorboard/` — `tensorboard --logdir outputs/<run>/tensorboard` for charts.

While training runs, a single throttled console line shows step/elapsed/ETA/
loss/reward/memory; it's a human-readable view only — `metrics.jsonl` is the
source of truth.

## Disk usage and cleanup

At the current stage (full fine-tuning, fp32, no LoRA yet), each run's 2
retained checkpoints total **~3GB** (`model.safetensors` + Adam's optimizer
state for all 135M params). This shrinks substantially once LoRA lands, since
only adapter weights get checkpointed — but until then, repeated smoke/debug
runs during development can silently eat disk space.

Nothing deletes old run directories automatically. Prune them explicitly:

```sh
uv run python -m tiny_grpo.cleanup --keep 3            # keep the 3 most recent runs
uv run python -m tiny_grpo.cleanup --keep 3 --dry-run   # preview what would be removed
```

It only ever removes whole run directories (oldest first) and always prints
what it removes — never silent, never automatic.

## Reward functions (`tiny_grpo/rewards.py`)

The model is prompted to end its response with `<answer>final numeric
answer</answer>`. Two reward functions:

- `accuracy_reward` — 1.0 if the extracted `<answer>` value exactly matches the
  gold answer (parsed from GSM8K's native `#### <n>` format), else 0.0.
- `format_reward` — 0.2 if the completion has a valid numeric `<answer>` tag at
  all, regardless of correctness, else 0.0.

Both are pure functions, unit tested without loading a model (`tests/test_rewards.py`).
On this tiny model, `accuracy_reward` frequently stays at 0 for a whole run —
expected at this scale, not a bug (see `CLAUDE.md`).

## Dataset splits (`tiny_grpo/splits.py`)

Deterministic, seeded train/validation split from GSM8K's **train** split, plus
a separate test split held out from GSM8K's **test** split (reserved for a
one-time final evaluation once a configuration is chosen — not used during
iterative training/eval). Indices are persisted per-run in `split_metadata.json`.

## Configuration (`tiny_grpo/config.py`)

Typed `TrainingConfig` with three profiles:

- `smoke_config()` — fast sanity check (64/16/32 train/val/test, 10 steps).
- `debug_config()` — larger iteration profile (256/32/64, 50 steps).
- `longer_config()` — explicit-experiment profile (1024/64/128, 200 steps);
  never launched automatically.

## Testing

```sh
uv run pytest -q tests/
```

CPU-only, no model or dataset download — covers reward/answer extraction,
split determinism and overlap, config validation, run-directory/metadata
persistence, memory reporting, and cleanup pruning.
