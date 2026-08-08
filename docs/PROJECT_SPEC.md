# Tiny GRPO Project Spec

> This is the shared, hardware-agnostic design and milestone spec for the
> project — reward math, dataset handling, evaluation, testing, checkpointing,
> logging, and the current milestone. It applies the same way regardless of
> which machine you're running on.
>
> Concrete hardware settings (device, precision, batch size, gradient
> checkpointing, KL/reference-model defaults, memory reporting APIs) live in:
> - `docs/SPEC_MACOS_MPS.md` — MacBook Pro M2, 16 GB unified memory
> - `docs/SPEC_CUDA_4GB.md` — RTX 3050 laptop GPU, 4 GB dedicated VRAM
>
> Read the profile doc matching your machine alongside this one before
> running anything — this file will tell you *what* to configure; the
> profile doc tells you *what value* is safe to configure it to.
>
> `CLAUDE.md` at the repo root is the always-loaded summary; this file is not
> auto-loaded every turn. Read it explicitly at the start of a work session,
> and update "Current Milestone" / "Completion Criteria" as work progresses
> so it stays accurate rather than aspirational.

## Project Goal

Build a small, understandable, and reproducible GRPO training project using
Hugging Face TRL, that runs correctly on either of two supported hardware
profiles (see `docs/SPEC_MACOS_MPS.md` and `docs/SPEC_CUDA_4GB.md`).

Keep the project simple enough to understand and debug. Do not turn it into a
general-purpose RL framework.

## Core Technology Choices

Use the following unless a documented compatibility issue requires a change.
Items marked **(profile-specific)** have concrete values defined per hardware
profile, not here — this section only fixes what's constant across profiles.

- Model: `HuggingFaceTB/SmolLM2-135M-Instruct` (current default — a
  model-capacity ceiling has since been diagnosed for `accuracy_reward` at
  this scale; see "Accuracy-Reward Constraints" below and
  `docs/ACCURACY_IMPROVEMENT_PLAN.md` before assuming this stays
  fixed)
- Dataset: `openai/gsm8k`
- Trainer: Hugging Face TRL `GRPOTrainer`
- Adaptation: LoRA
- Precision **(profile-specific)** — see profile docs. Do not assume the same
  precision strategy is appropriate on both profiles; MPS and CUDA have
  different maturity levels for fp16/bf16.
- Device **(profile-specific)** — resolved from the active hardware profile,
  never hardcoded to `"mps"` or `"cuda"` in training logic.
- Generations per prompt: **prefer 4**, on both profiles, as the default target. Fall back to 2 only when a hardware profile's memory budget doesn't support 4 (verify with a smoke run rather than assuming) — 2 is the practical floor for GRPO's group-relative advantage, not a starting point to default to for convenience.

  **Known limitation:** GRPO's advantage estimate is the within-group
  comparison of rewards. At `num_generations=2` the advantage collapses to an
  essentially binary "which of the two completions was better" signal — high
  variance, and noticeably noisier training curves than with 4+ generations.
  This is why 4 is the preferred default rather than 2: the group-relative
  baseline is a much better estimate with 4 samples. Only drop to 2 when the
  active hardware profile's memory budget genuinely can't support 4 (see
  profile docs), and document that the run used the reduced group size when
  reporting/comparing results — it's a real methodological difference, not a
  cosmetic config change.

- Reference model / KL penalty (`beta`) **(profile-specific default)** —
  `GRPOTrainer` by default loads a separate reference model to compute the KL
  penalty, which adds meaningful memory pressure. Expose `beta` as a
  first-class config field with `beta=0` supported as an explicit
  no-reference-model mode. Which default is safe depends heavily on available
  memory — see profile docs; do not assume the same default works on both.
- Rollout engine: standard Transformers generation
- vLLM: disabled on both profiles (reasons differ by device — see profile docs)
- Distributed training: disabled

Do not introduce DeepSpeed, Docker, multi-GPU training, or complex
orchestration at this stage, on either profile.

**Model-scale expectation:** SmolLM2-135M-Instruct is a very small model.
Expect a high early rate of malformed or missing `<answer>` tags and low
exact-answer accuracy, especially before training and in early steps, on
either profile. This is expected tiny-model behavior. Do not spend debugging
time trying to "fix" this unless format/parse failure rates fail to improve
at all over the course of training.

For the specific diagnosis and experiment history behind low exact-answer
accuracy *after* format compliance was fixed, see "Accuracy-Reward
Constraints" below and `docs/ACCURACY_IMPROVEMENT_PLAN.md` — that is a
separate, since-diagnosed issue from the format/parse problem this paragraph
describes.

## Training Objective

For each GSM8K prompt:

```
Prompt
    ↓
Generate multiple completions
    ↓
Score each completion
    ↓
Compare rewards within the group
    ↓
Update LoRA parameters using GRPO
```

Use two initial reward functions:

**Exact-answer reward**
- 1.0 when the numeric answer inside `<answer>...</answer>` matches the GSM8K ground truth.
- 0.0 otherwise.

**Format reward**
- 0.2 when the completion contains a valid numeric value inside `<answer>...</answer>`.
- 0.0 otherwise.

Keep reward logic independently testable, and identical across both hardware profiles.

## Accuracy-Reward Constraints

Two corrections to the reward/prompt logic, made after diagnosing why
`accuracy_reward` stayed near zero, are load-bearing and must not be
reverted:
- Numeric answer comparison must use exact `Decimal`-based canonicalization
  (`tiny_grpo/rewards.py::normalize_numeric_answer`), not string/float
  comparison — `"42"`, `"42.0"`, `"042"`, and `"42.000"` must all normalize
  equal. Reverting to naive string comparison reintroduces false-negative
  accuracy rewards.
- The system prompt and few-shot examples must keep the stronger,
  instruction-following phrasing that fixed an earlier format-compliance
  bottleneck (the model wasn't emitting a parseable `<answer>` tag at all).
  That was an instruction-following limitation, not a token-length one — do
  not try to re-fix it by raising `max_completion_length` instead.

Beyond those two fixes, whether `SmolLM2-135M-Instruct` can support
meaningful `accuracy_reward` learning at all is a diagnosed, evidence-gated
open question, not something to assume away. The full diagnostic
methodology, raw numbers, and confidence intervals — rollout-variance
diagnostics, the SFT warm-start experiments (all three strengths, a
documented negative result), and the small-model capability bakeoff — live
in `docs/ACCURACY_IMPROVEMENT_PLAN.md`. See the "Current Milestone"
exception below for what that evidence currently sanctions as a next step.

## Development Principles

- Work incrementally.
- Keep every milestone runnable.
- Make one meaningful change at a time.
- Prefer explicit code over premature abstraction.
- Preserve the last working version before major refactoring.
- Create only files that are currently needed.
- Keep configuration separate from training logic.
- Keep reward functions separate from trainer orchestration.
- Treat successful execution and model-quality improvement as separate outcomes.
- Do not claim success unless the relevant command was actually executed successfully.
- When an API is version-sensitive, inspect the installed package or current official documentation — do not rely on memorized API shapes, since `GRPOTrainer` argument names and reward-function signatures have changed across TRL versions.
- Pin dependency versions (`trl`, `transformers`, `peft`, `torch`, `accelerate`) in a lockfile/requirements file, not just log them at runtime.
- Never overwrite an existing run directory silently.
- Do not change dataset splits, reward weights, or evaluation settings between comparable runs without documenting the change.
- Treat **both** MPS and CUDA compatibility as first-class constraints. Do not assume CUDA-only or MPS-only APIs, memory utilities, or performance tricks are available — gate any device-specific call behind an explicit device check, and provide a fallback (at minimum, process-level memory via a library like `psutil`) for whichever device isn't active.
- Do not assume a single hardware profile when writing training/eval/logging code — both `mps_16gb` and `cuda_4gb` must be exercised by the test/verification plan over time, even if only one machine is available in a given session.

## Dataset Requirements

Use three separate datasets:

**Training set** — used for GRPO updates.

**Validation set** — used during development for:
- comparing runs;
- evaluating checkpoints;
- inspecting progress;
- tuning configuration.

**Test set** — used only after the final configuration or checkpoint has been selected.

Use:
- training and validation subsets from the official GSM8K train split;
- final testing from the official GSM8K test split.

Requirements:
- use a fixed split seed;
- ensure training and validation do not overlap;
- keep the test split separate;
- persist selected example IDs or indices;
- save split sizes and seed;
- reuse the same validation and test subsets across comparable runs, **regardless of which hardware profile ran them** — splits must not vary by device;
- never include ground-truth answers in prompts.

## Configuration Requirements

Move important settings out of the training code. Config should be composed
from two layers so hardware and run-length settings don't have to be
duplicated combinatorially:

1. A **run profile** (smoke / debug / longer) — controls training length,
   logging/checkpoint/eval cadence, and similar run-shape settings.
2. A **hardware profile** (`mps_16gb` / `cuda_4gb`) — controls device,
   precision, batch size, gradient accumulation, gradient checkpointing, and
   `beta`/reference-model defaults, as defined in the corresponding SPEC doc.

A concrete run is the combination of the two (e.g. "smoke run on `cuda_4gb`").
Do not hardcode a hardware profile's values into a run profile's config, or
vice versa.

At minimum, configuration must cover:

- run name and seed;
- model name and precision;
- device selection (resolved from hardware profile, not hardcoded);
- MPS fallback setting (profile-specific — only meaningful on `mps_16gb`);
- LoRA settings;
- KL coefficient (`beta`), including a `beta=0` no-reference-model mode;
- dataset sizes and split seed;
- number of generations per prompt;
- maximum prompt length;
- maximum completion length;
- batch size;
- gradient accumulation;
- gradient checkpointing toggle (default value is profile-specific — see SPEC docs);
- learning rate;
- learning-rate scheduler type;
- maximum steps;
- logging interval;
- checkpoint interval;
- checkpoint retention limit (default and hard cap: 2 checkpoints on disk, on both profiles);
- evaluation interval;
- output directory;
- resume settings;
- which hardware profile a run used (recorded, not just applied, so logs are self-describing).

**GRPO batch-divisibility constraint (validate at config construction, not at
trainer construction):**
- `per_device_train_batch_size` counts *completions* (rows, after each prompt
  is repeated `num_generations` times), not unique prompts — a reasonable
  assumption to have but not what trl implements. Confirmed two ways against
  the installed trl: (1) `generation_batch_size = per_device_train_batch_size
  * num_processes * steps_per_generation` has no separate `* num_generations`
  factor, which only holds dimensionally if `per_device_train_batch_size` is
  already in completions units; (2) the `RepeatSampler` diagram in
  `trl/trainer/grpo_trainer.py`'s `_get_train_sampler` — measured by column
  position, not eyeballed — shows `per_device_train_batch_size=3` as 3
  completion-rows per device (with `num_processes > 1`, a single prompt's
  repeats can even be split across devices and gathered back together for
  reward normalization; irrelevant here since this project never uses more
  than one process).
- `num_generations` must be >= 2 — GRPO needs at least a group of 2 completions
  per prompt to compute an advantage; trl's `GRPOConfig` hard-rejects `< 2`.
- `per_device_train_batch_size` must be an exact multiple of `num_generations`,
  so every device holds complete prompt groups. This is stricter than trl's
  raw training-side requirement (which only requires
  `per_device_train_batch_size * gradient_accumulation_steps` to be a multiple
  of `num_generations`) — but this project always sets the eval batch size
  equal to `per_device_train_batch_size` with no independent field, and trl's
  eval-side constraint applies with no gradient-accumulation multiplier. So
  checking `per_device_train_batch_size` alone is the real necessary condition
  (for eval) and remains sufficient for training. Do not loosen this check to
  the raw training-side formula without also giving eval its own,
  independently-validated batch size.
- Validate both of these when the config is constructed, so a bad combination
  fails immediately with a clear message — not deep inside `GRPOTrainer`
  construction or, worse, only once eval fires mid-run.

Support separate run profiles for smoke / debug / longer, each usable with
either hardware profile.

Do not hard-code long-run settings, or either hardware profile's settings,
into the training logic itself.

## Logging Requirements

Every run must create a unique output directory.

Record at minimum:
- resolved configuration, **including which hardware profile was active**;
- pinned package and environment versions;
- model and dataset information;
- split metadata;
- training metrics;
- reward metrics;
- selected generated samples;
- checkpoint locations;
- final adapter location;
- elapsed time;
- process memory;
- device memory statistics (CUDA stats on `cuda_4gb`, MPS stats on `mps_16gb` — see profile docs for exact APIs) when available;
- final run status.

Log selected completions with:
- prompt or prompt ID;
- ground truth;
- generated completion;
- extracted answer;
- accuracy reward;
- format reward;
- total reward.

Use file-based structured logging such as JSON Lines where practical.

Avoid excessive console output — the JSONL logs are the source of truth for detail, not the terminal.

**Console progress output is a separate, required exception to "avoid excessive console output."** Emit a concise, single-line progress update to stdout/tty at the same cadence as `logging_interval`, so training can be monitored live without tailing log files. Each update should be one line (or a small fixed number of lines that overwrite in place, e.g. via `\r`), containing at minimum:
- current step / total steps;
- elapsed time (and, if easy to compute cheaply, estimated time remaining);
- current loss;
- mean reward for the current batch (and ideally accuracy reward and format reward separately);
- current process memory (and device memory if available — CUDA or MPS depending on active profile).

Requirements for this output:
- It must be throttled to `logging_interval` — do not print per-sample or per-token; that's what "excessive" refers to and remains disallowed.
- It must not replace or duplicate the full JSONL record — it's a lightweight derived view, not an alternate logging destination. Do not let console formatting logic leak into what gets persisted to disk.
- It must degrade gracefully if run in a non-interactive context (e.g. piped to a file, CI, or `gtimeout`-wrapped invocation) — don't assume a real tty; fall back to plain newline-terminated lines rather than relying on carriage-return overwrite tricks in that case.
- It is for human monitoring only — nothing downstream (tests, resume logic, evaluation) should parse or depend on console output.
- It must not assume a particular device's memory API is available — resolve which stats to fetch (or omit) based on the active hardware profile.

Do not require CUDA-specific metrics on the `mps_16gb` profile, and do not require MPS-specific metrics on the `cuda_4gb` profile. Use device-appropriate memory reporting per profile, and record process-level memory as a universal fallback on both.

## Evaluation Requirements

Evaluation must be independent from training.

Implement:

**Baseline validation** — evaluate the base model before training.

**Post-training validation** — evaluate the trained adapter using the same:
- validation examples;
- prompt template;
- generation settings;
- answer extraction;
- metrics.

Baseline and post-training evaluation must be comparable **within a hardware
profile** (same device, same run). Cross-profile comparisons (e.g. an
`mps_16gb` run vs. a `cuda_4gb` run) are informative but not a substitute for
same-profile before/after comparison, since precision and generation settings
differ by profile.

**Final test evaluation** — run only after selecting the final configuration or checkpoint.

Record at minimum:
- exact-answer accuracy;
- valid-format rate;
- parse-failure rate;
- average reward;
- average completion length;
- sample outputs;
- runtime;
- process memory;
- device memory statistics when available (per active profile).

Print a concise base-versus-trained comparison.

Do not repeatedly tune against the test set.

## Checkpoint and Resume Requirements

Support:
- configurable checkpoint intervals;
- checkpoint retention limit, **capped at a maximum of 2 checkpoints on disk at any time, on both hardware profiles**. When a new checkpoint is saved and the cap is exceeded, delete the oldest checkpoint(s) first (excluding the final saved adapter, which is retained separately). Do not make this configurable to a higher value without an explicit documented reason — checkpoints store optimizer and scheduler state in addition to the adapter, so they are disk-heavy relative to model size, and disk headroom should not be assumed to be large on either machine;
- explicit resume from a checkpoint;
- resume from the latest valid checkpoint;
- final LoRA adapter saving (kept independently of the 2-checkpoint rolling cap, since it represents the selected result rather than training-state history);
- clear reporting of fresh start versus resumed run, **including which hardware profile the original and resuming run used** (flag it clearly if they differ — resuming a `cuda_4gb` run's checkpoint under `mps_16gb`, or vice versa, is not guaranteed to work and must not be silently assumed safe).

Verify resume with a short test:
1. start a small run;
2. save a checkpoint;
3. stop after the checkpoint;
4. resume;
5. verify the global step continues;
6. verify the final adapter loads.

Do not assume that saving only the LoRA adapter is sufficient to resume optimizer and scheduler state.

## Run Directory Cleanup

Smoke runs and other verification-only runs (anything run purely to check
that a code path works — e.g. "does resume work", "does generation complete"
— not to produce a result worth keeping) create a full run directory just
like any other run: logs, samples, checkpoints. Left unmanaged these
accumulate fast, since verification runs happen far more often than real
experiment runs. This is a separate concern from the 2-checkpoint cap above,
which governs checkpoints *within* one run directory, not the number of run
directories themselves.

Requirements:

- Tag every run directory, in its saved metadata, with its run profile
  (`smoke` / `debug` / `longer`) and whether it was purely a verification run.
  Cleanup logic (manual or automatic) must key off this metadata, not off
  directory-naming conventions alone.
- Only `smoke`-profile and explicitly-flagged verification run directories
  are ever eligible for automatic deletion. `debug` and `longer` runs are
  **never** auto-deleted — they may contain results worth keeping, and
  pruning them automatically risks silently discarding real work.
- Apply a retention policy to smoke/verification run directories: keep only
  the most recent N (default: 3) on disk; delete older ones automatically as
  new smoke/verification runs complete — the same pattern as the 2-checkpoint cap, one level up.
- A **failed** smoke/verification run must not be auto-deleted before someone
  has had a chance to inspect it. Keep at least the single most recent failed
  run of a given kind regardless of the retention count, until a successful
  run of the same kind supersedes it or it's cleared manually.
- Automatic cleanup must report what it deleted (directory names, and
  ideally freed disk space) — never delete silently. Same principle as
  "never overwrite a run directory silently": deletion is fine, silent
  deletion is not.
- Also provide an explicit manual cleanup command/script, separate from the
  automatic policy, that:
  - lists run directories with their tagged profile, age, and size;
  - supports a dry-run mode that previews what would be deleted without deleting anything;
  - lets the user prune smoke/verification runs beyond the retention count or older than a given age, on demand.

## Testing Requirements

Add CPU-only unit tests for:
- GSM8K ground-truth extraction;
- generated-answer extraction;
- numeric normalisation;
- accuracy reward;
- format reward;
- malformed outputs;
- deterministic split creation;
- split overlap detection;
- configuration validation, **including hardware-profile field validation** (e.g. rejecting an unrecognized profile name, and catching a run profile + hardware profile combination that's internally inconsistent, such as `beta` and reference-model settings that would obviously not fit in 4 GB);
- device/precision resolution logic (given a requested hardware profile and mocked "available" devices, confirm the correct device/precision/gradient-checkpointing defaults are selected) — this must not load a model or require an actual GPU/MPS device to run;
- run-directory cleanup/retention logic (given a mocked set of tagged run directories with varying profiles, ages, and pass/fail status, confirm only smoke/verification runs beyond the retention count are selected for deletion, that the most recent failed run is protected, and that `debug`/`longer` runs are never selected) — this must operate on mocked directory metadata, not real runs or a real model.

Unit tests must not download or load the model.

Keep MPS and CUDA integration tests separate and minimal, and keep them
distinguishable (e.g. skip CUDA integration tests when no CUDA device is
present, and skip MPS integration tests when no MPS device is present, rather
than failing).

Before running training, verify (per the active profile):
- the relevant device availability check is `True` (`torch.backends.mps.is_available()` for `mps_16gb`, `torch.cuda.is_available()` for `cuda_4gb`);
- the model can be moved to that device;
- one short generation completes;
- no code path specific to the *other* device is required to reach this point.

## Watchdog and Timeout Requirements

Every command run by the coding agent for verification must have a finite timeout.

This includes:
- environment checks;
- imports;
- unit tests;
- model loading;
- dataset loading;
- generation;
- training;
- evaluation;
- checkpoint-resume tests.

Suggested limits (starting points — adjust per profile if observed step
throughput differs meaningfully; see profile docs):
- environment and import checks: 30–120 seconds;
- unit tests: 1–5 minutes;
- model or dataset loading: 5–10 minutes;
- smoke training: 10–20 minutes;
- small evaluation: 10–20 minutes;
- resume verification: 15–30 minutes.

Use macOS `timeout` when GNU coreutils is installed:

```
brew install coreutils
```

Then use `gtimeout`:

```
gtimeout --signal=TERM --kill-after=30s 300s pytest -q

gtimeout --signal=TERM --kill-after=60s 1200s \
  python -m tiny_grpo.train --config configs/smoke_mps.yaml
```

On a non-macOS `cuda_4gb` machine, use the platform's native `timeout`
utility (GNU coreutils `timeout` on Linux) with the same signal/kill-after
pattern; the exact binary name is a profile detail, the timeout discipline is not.

When no such utility is available, use an equivalent watchdog implemented in
Python or the shell. Do not run verification commands without a finite timeout.

The watchdog must:
- send SIGTERM when the main timeout expires;
- allow a short cleanup period;
- send SIGKILL if needed;
- return a non-zero exit status;
- preserve the final visible logs;
- make it clear that the command timed out.

Do not silently remove or greatly extend a timeout merely to make a command pass.

A multi-hour run must never be launched as routine code verification. It must be started explicitly as an experiment.

## Current Milestone

Implement the following:

1. Refactor data preparation and reward logic into testable modules.
2. Add CPU-only unit tests.
3. Add typed configuration with smoke, debug, and longer-run profiles, composed with a hardware profile (`mps_16gb` / `cuda_4gb`), including `beta`, `max_prompt_length`, and gradient-checkpointing toggle.
4. Add a device/precision resolution utility driven by hardware profile, unit-tested without requiring an actual GPU/MPS device.
5. Add deterministic training, validation, and test split handling.
6. Persist split metadata.
7. Add unique run directories.
8. Save resolved configuration (including active hardware profile), pinned dependency versions, and environment metadata.
9. Add structured training and sample logging, plus throttled console progress output.
10. Add baseline validation evaluation.
11. Add post-training validation evaluation.
12. Add checkpoint saving with the 2-checkpoint retention cap.
13. Add checkpoint resume, including a cross-profile mismatch check.
14. Verify resume with a short timeout-protected run.
15. Add run-directory tagging (profile + verification flag) and a retention/cleanup mechanism for smoke/verification run directories, plus a manual dry-run-capable cleanup command.
16. Keep the workflow stable on at least one of the two profiles per session, with both profiles supported by the code (no hardcoded single-device assumption).

Do not yet:
- launch a multi-hour run;
- add external experiment tracking;
- add distributed infrastructure;
- add medical-safety data or rewards.

**"Add a larger model" is no longer a blanket "never."** The
accuracy-improvement diagnosis (see "Accuracy-Reward Constraints" above and
`docs/ACCURACY_IMPROVEMENT_PLAN.md`) found a model-capacity ceiling, and the
capability bakeoff gives evidence-based grounds to evaluate a swap. It
remains explicit, not casual. The Qwen3 gate was completed on 2026-08-07:
- do not switch the default model in place — add any candidate (e.g.
  `Qwen3-0.6B-Instruct`) as an explicit, config-selectable model profile;
- `qwen3_0_6b` is now an explicit profile; SmolLM2 remains the default;
- a real timeout-protected LoRA+GRPO step on `cuda_4gb` completed with four
  generations, `beta=0.04`, gradient checkpointing, and 2,809 MiB maximum CUDA
  memory allocated, so training-time feasibility is established for the smoke
  configuration;
- verify its LoRA target module names before assuming they match
  `q_proj`/`k_proj`/`v_proj`/`o_proj`;
- those four targets were verified against the loaded Qwen3 model before the
  smoke runs;
- document the swap (or the decision not to make it) the same way every
  other deviation in this project is documented, rather than by drift.

The base-Qwen3 direct-GRPO gate and its immediate one-variable follow-ups are
complete. Canonical 200-prompt diagnostics found only modest, statistically
inconclusive changes after 50 steps with either the original linear schedule
or constant learning rate. Expanding the same 50-step linear run from 256 to
512 unique training examples also did not improve pass@4 (38.5%, versus 38.0%
base and 39.5% for the 256-example linear run). Do not extend the same
configuration or claim model-quality improvement from these runs. Full
commands, paired comparisons, and retained output paths are recorded in
`docs/ACCURACY_IMPROVEMENT_PLAN.md`. The successful Qwen3 SFT and GRPO smoke
runs remain memory/integration evidence; they do not independently establish
quality improvement.

## Completion Criteria

The milestone is complete only when:

- unit tests pass;
- device/precision resolution is unit-tested for both hardware profiles without requiring the actual hardware to be present;
- dataset splits are deterministic and non-overlapping;
- split metadata is saved;
- baseline validation metrics are produced;
- training metrics are logged;
- a concise per-`logging_interval` progress line is emitted to the console during a run (verify by observing stdout during a smoke or debug run, not just by inspecting code);
- selected completions are saved;
- checkpoints are created;
- checkpoint retention never exceeds 2 on disk at any point during a run (verify by inspecting the checkpoint directory during/after a multi-checkpoint smoke or debug run);
- smoke/verification run directories are tagged with profile and verification status, and stay within the retention policy after several smoke runs (verify by running multiple smoke/verification commands and inspecting the run-directory listing — including that a deliberately-failed run isn't pruned before inspection);
- a run resumes correctly;
- post-training validation metrics are produced;
- base-versus-trained comparison is generated;
- final adapter is saved and loadable;
- process and device memory metrics are recorded, using the correct API for the active hardware profile;
- all verification commands use explicit timeouts;
- the workflow runs successfully on **the currently active hardware profile** as configured (`mps_16gb` and/or `cuda_4gb`), per the corresponding profile spec — this doesn't require both machines in the same session, but the code must not hardcode assumptions that would break the other profile.

At the end, report:

- files created or modified;
- exact commands executed;
- timeout used for each command;
- whether each command succeeded, failed, or timed out;
- **which hardware profile was used**;
- dataset split sizes and seed;
- baseline metrics;
- post-training metrics;
- checkpoint-resume result;
- process and device memory usage where available;
- unresolved issues;
- recommended next step.

Never claim that training, evaluation, or resume succeeded unless the relevant timeout-protected command completed successfully.
