# grpo-test

Small, understandable GRPO training project: fine-tune
`HuggingFaceTB/SmolLM2-135M-Instruct` on GSM8K with TRL's `GRPOTrainer` + LoRA,
supporting two hardware profiles — an M2 MacBook Pro (`mps_16gb`) and an RTX
3050 laptop (`cuda_4gb`). See `CLAUDE.md` and `docs/PROJECT_SPEC.md` (plus
`docs/SPEC_MACOS_MPS.md` / `docs/SPEC_CUDA_4GB.md` for the concrete per-profile
settings) for the full design and current milestone.

## Running

```sh
uv run python train_grpo.py --profile smoke --hardware mps_16gb
```

- `--profile`: `smoke` (default, fast sanity check) / `debug` / `longer`.
  `longer` is bigger and slower — start it deliberately, never as routine
  verification.
- `--hardware`: required, `mps_16gb` or `cuda_4gb`.

### SFT warm-start

Stage 2 provides a separate LoRA SFT path before further exact-reward GRPO:

```sh
uv run python train_sft.py --profile smoke --hardware cuda_4gb
```

SFT profiles are `smoke` (3-step end-to-end verification), `debug` (one epoch
over 256 examples), and `stronger` (two effective epochs over the reserved
1,024-example train set). All use deterministic GSM8K splits,
completion-only loss so prompt tokens are masked, a hard two-checkpoint cap,
baseline/post-training generation evaluation, and the same metadata/resume
conventions as GRPO. SFT targets preserve gold reasoning, remove GSM8K's
calculator annotations, and end with exactly one canonical
`<answer>NUMBER</answer>` line. The preflight length audit refuses to train if
an answer would be silently truncated.

The smoke profile verifies mechanics and memory only; it is too short to
support an accuracy claim. Start `--profile debug` deliberately after
reviewing smoke artifacts.

To run in the background with a watchdog that kills the process if it hangs
or runs too long:

```sh
./run_with_watchdog.sh --profile smoke --hardware mps_16gb
```

### Resuming an interrupted run

```sh
uv run python train_grpo.py --profile smoke --hardware mps_16gb --resume latest
```

Resume is **opt-in only** — `--resume` defaults to `none`, so a plain re-run
is always fresh. `--resume latest` continues the most recent incomplete run
matching the same `--profile`/`--hardware`, if one exists; `--resume <path>`
targets an explicit checkpoint or run directory. Resuming a checkpoint saved
under a *different* hardware profile fails loudly unless you also pass
`--allow-cross-profile-resume` (see `tiny_grpo/resume.py`).

## Output layout

Every invocation creates a fresh, uniquely-named directory under `outputs/`
(e.g. `outputs/smoke_20260801_211021/`) — a run is never silently overwritten,
except when explicitly resumed (which reuses the original run's directory).
Each run directory contains:

- `config.json` — the fully resolved config for that run (run profile ×
  hardware profile, LoRA settings, everything else).
- `environment.json` — pinned package versions (`torch`, `transformers`,
  `trl`, `accelerate`, `datasets`, `peft`) and Python/platform info.
- `split_metadata.json` — the exact train/validation/test indices and seed
  selected from GSM8K for that run (see `tiny_grpo/splits.py`).
- `run_tags.json` — run profile, hardware profile, verification-run flag, and
  status (`running`/`completed`/`failed`) — what `tiny_grpo/cleanup.py` and
  `tiny_grpo/resume.py` key off, not directory naming.
- `metrics.jsonl` — every `trainer.log()` call (train and eval), one JSON
  object per line, including process memory and device-appropriate memory
  (`mps_memory_mb` or `cuda_memory_mb`, whichever is active).
- `completions/completions_*.parquet` — sampled (prompt, completion, reward
  breakdown, extracted answer, gold answer, advantage) rows per logging step.
  GRPO only; SFT does not produce rollout-completion logs.
- `eval_baseline.json` / `eval_post_training.json` — accuracy, format rate,
  parse-failure rate, mean reward, mean completion length, runtime, memory,
  and sample completions, before and after training, against the same
  validation set/generation settings/answer extraction (see
  `tiny_grpo/evaluate.py`). A resumed run doesn't recompute the baseline —
  it's read back from the original run's file.
- `checkpoint-*/` — LoRA adapter + optimizer/scheduler state; capped at the 2
  most recent per run (`checkpoint_retention` in config, hard-capped at 2).
- `final_adapter/` — the final trained LoRA adapter, kept independent of the
  2-checkpoint rolling cap.
- `sft_data_stats.json` — SFT-only token-length audit for train and validation,
  including the count over the configured sequence limit (required to be zero).
- `tensorboard/` — `tensorboard --logdir outputs/<run>/tensorboard` for charts.

While training runs, a single throttled console line shows step/elapsed/ETA/
loss/reward/memory; it's a human-readable view only — `metrics.jsonl` is the
source of truth.

## Disk usage and cleanup

Training goes through LoRA adapters, not full fine-tuning, so checkpoints are
small — a full smoke run's 2 retained checkpoints + final adapter typically
total **~30MB**, not gigabytes.

Nothing deletes old run directories automatically. Manage them explicitly:

```sh
uv run python -m tiny_grpo.cleanup list                        # see tagged runs, age, size
uv run python -m tiny_grpo.cleanup prune --keep 3               # keep the 3 most recent verification runs
uv run python -m tiny_grpo.cleanup prune --keep 3 --dry-run     # preview without deleting
uv run python -m tiny_grpo.cleanup prune --older-than-days 7    # age-based pruning
```

Only runs tagged `verification_run=True` (smoke runs, by default) are ever
eligible for deletion — `debug`/`longer` runs and untagged/foreign
directories are never touched, and the most recent *failed* run of a given
profile is protected until a later success supersedes it. Always prints what
it finds/removes — never silent.

## Hardware profiles (`tiny_grpo/hardware.py`)

| | `mps_16gb` | `cuda_4gb` |
|---|---|---|
| Precision | fp32 | bf16 |
| Gradient checkpointing | off | on (required) |
| `beta` (KL coefficient) | 0.04 | 0.04 |

Both profiles prefer `num_generations=4`. `beta` is the same on both profiles
because LoRA means TRL never loads a second full reference model regardless
of `beta` — it disables/clones a tiny adapter instead, so the old
memory-cost argument for `cuda_4gb` defaulting to `beta=0` no longer applies.

`verify_precision_supported` checks `torch.cuda.is_bf16_supported()` at
startup on `cuda_4gb` and **fails loudly** (never silently substitutes a
different precision) if the GPU doesn't actually support real bf16.

## LoRA (`tiny_grpo/lora.py`)

All training goes through LoRA adapters (`peft`), not full fine-tuning —
checkpoints and the final adapter are adapter-only.

## Reward functions (`tiny_grpo/rewards.py`)

The model is prompted to end its response with `<answer>final numeric
answer</answer>`. Two reward functions:

- `accuracy_reward` — 1.0 if the extracted `<answer>` value exactly matches the
  gold answer (parsed from GSM8K's native `#### <n>` format), else 0.0. Valid
  decimal spellings are canonicalized exactly, so representation-only
  differences such as `42` versus `42.0` do not create false negatives.
- `format_reward` — 0.2 if the completion has a valid numeric `<answer>` tag at
  all, regardless of correctness, else 0.0.

Both are pure functions, unit tested without loading a model
(`tests/test_rewards.py`). On this tiny model, `accuracy_reward` frequently
stays at 0 for a whole smoke run — expected at this scale, not a bug (see
`CLAUDE.md`).

## Dataset splits (`tiny_grpo/splits.py`)

Deterministic, seeded train/validation split from GSM8K's **train** split, plus
a separate test split held out from GSM8K's **test** split (reserved for a
one-time final evaluation once a configuration is chosen — not used during
iterative training/eval). Indices are persisted per-run in `split_metadata.json`.

## Rollout viability diagnostic (`tiny_grpo/diagnose_rollouts.py`)

Before adding more GRPO steps, measure whether grouped base-model samples
contain any exact-reward variation:

```sh
timeout --signal=TERM --kill-after=30s 1200s \
  uv run python -m tiny_grpo.diagnose_rollouts \
  --profile debug --hardware cuda_4gb \
  --num-prompts 16 --num-generations 4 \
  --command-timeout-seconds 1200
```

Use `--model-id <hub-id>` for a generation-only model capability comparison.
Models such as Qwen3 that expose reasoning behavior through their chat template
can be controlled explicitly with `--chat-template-mode thinking` or
`--chat-template-mode non-thinking`; the selected mode is persisted in the
diagnostic config. Omit both flags to preserve the project model and its native
chat-template behavior.

The command performs generation only (no optimizer/backward pass), using the
versioned `data/diagnostic_manifest_v1.json`. It saves every prompt group and
reports pass@1, pass@k, sample exact accuracy, formatting/failure categories,
truncation, within-group reward variance, confidence intervals, runtime, and
memory in a unique `outputs/diagnostic_*` directory.

## Evaluation (`tiny_grpo/evaluate.py`)

Baseline (pre-training) and post-training validation against the same
held-out set, reusing training's explicit generation settings (temperature/
top_p/top_k from the resolved typed config) and exact answer-extraction
logic (`tiny_grpo.rewards`) — never a separate reimplementation that could
silently drift. Prints a concise before/after comparison and persists both
passes to the run directory.

## Configuration (`tiny_grpo/config.py`)

A `TrainingConfig` is a **run profile** (`smoke_config()` / `debug_config()` /
`longer_config()` — training length, logging/checkpoint/eval cadence) composed
with a **hardware profile** (`tiny_grpo.hardware.MPS_16GB` /
`CUDA_4GB` — device, precision, batch size, gradient checkpointing, `beta`).
Neither hardcodes the other's settings.

## Testing

```sh
uv run pytest -q tests/
```

Mostly CPU-only, model-free unit tests — reward/answer extraction, split
determinism and overlap, config validation, hardware-profile resolution
(mocked device availability), cleanup/resume selection logic (synthetic
tagged directories) — plus two real integration tests
(`tests/test_mps_integration.py`, `tests/test_cuda_integration.py`) that load
the actual model on whichever device is present and skip (not fail) when
that device isn't available on the machine running them.
