# CLAUDE.md — Tiny GRPO Project

Small, understandable GRPO training project using Hugging Face TRL. Not a
general-purpose RL framework — keep it simple and debuggable.

This project supports **two hardware profiles**:

| Profile | Machine | Device | VRAM/memory |
|---|---|---|---|
| `mps_16gb` | MacBook Pro M2 | MPS | 16 GB unified |
| `cuda_4gb` | RTX 3050 laptop GPU | CUDA | 4 GB dedicated |

Nothing in the training/eval/reward code should hardcode an assumption about
either device or memory budget. Device selection, precision, batch size,
gradient checkpointing, and KL/reference-model settings must all come from
config, resolved per hardware profile — not from `if mac: ... else: ...`
branches scattered through the code.

Full docs:
- **`docs/PROJECT_SPEC.md`** — shared design: reward functions, dataset
  splits, evaluation, testing, checkpointing, current milestone. Hardware-
  agnostic. Read this before starting substantive work.
- **`docs/SPEC_MACOS_MPS.md`** — concrete settings/config for the M2 16GB profile.
- **`docs/SPEC_CUDA_4GB.md`** — concrete settings/config for the RTX 3050 4GB profile.
- **`docs/ACCURACY_IMPROVEMENT_PLAN.md`** — full methodology, raw numbers, and
  experiment history behind the `accuracy_reward` diagnosis; see
  `docs/PROJECT_SPEC.md`'s "Accuracy-Reward Constraints" section for the summary
  and what's sanctioned as a next step.

Read the profile doc that matches the machine you're actually running on
before touching config defaults — the two profiles are not interchangeable
(what's a safe default on 16 GB unified memory can OOM instantly on 4 GB
dedicated VRAM).

## Stack (same across both profiles)

- Model: `HuggingFaceTB/SmolLM2-135M-Instruct` (current default — a
  model-capacity ceiling has since been diagnosed for `accuracy_reward` at
  this scale; see "Known, expected (not bugs)" below and
  `docs/PROJECT_SPEC.md`'s "Accuracy-Reward Constraints" before assuming this
  stays fixed)
- Dataset: `openai/gsm8k`
- Trainer: TRL `GRPOTrainer` + LoRA
- Rollout: standard Transformers generation (vLLM disabled on both profiles —
  see profile docs for why, since the reason differs by device)
- `trl`/`transformers`/`peft`/`torch` versions pinned in the lockfile — check
  installed versions before assuming an API shape; `GRPOTrainer`'s args have
  changed across TRL releases.

Precision, device, batch size, gradient checkpointing, and `beta` (KL
coefficient / reference-model usage) are **profile-specific** — see the two
SPEC docs. Do not assume MPS defaults are safe on CUDA or vice versa.

## Project layout conventions

- Config lives separately from training logic (typed config, smoke/debug/longer
  profiles, composed with a hardware profile — see Configuration Requirements
  in `docs/PROJECT_SPEC.md`).
- Reward functions live separately from trainer orchestration and are
  unit-testable without loading a model.
- Every run gets its own output directory — never overwrite an existing run silently.
- Two logging destinations, not one: JSONL files are the detailed source of
  truth; a throttled, single-line console progress update (step, loss, reward,
  elapsed time, memory) at `logging_interval` is for live human monitoring
  only. Nothing should parse console output — it's a derived view, not a
  logging path.
- Memory reporting is device-abstracted: use CUDA stats
  (`torch.cuda.memory_allocated`, `max_memory_allocated`) on the `cuda_4gb`
  profile, MPS stats (`torch.mps.current_allocated_memory`, guarded with
  `hasattr` since the API has shifted across torch versions) on the
  `mps_16gb` profile, and process-level memory (e.g. via `psutil`) always, on
  both. Never call a CUDA-only or MPS-only API without checking which device
  is active first.

## Non-negotiable rules

- **Timeouts on every verification command**, no exceptions (env checks, imports,
  unit tests, model/dataset loading, generation, training, eval, resume tests).
  Use `gtimeout --signal=TERM --kill-after=<n>s <total>s <cmd>` or an equivalent
  watchdog. Never silently drop or balloon a timeout to make a command pass.
- Never launch a multi-hour run as routine verification — only as an explicit,
  user-requested experiment.
- Don't claim training/eval/resume succeeded unless the actual timeout-protected
  command completed successfully.
- Don't change dataset splits, reward weights, or eval settings between comparable
  runs without documenting it.
- Saving the LoRA adapter alone is **not** sufficient to resume optimizer/scheduler
  state — resume support needs more than the adapter.
- **Checkpoint retention is capped at 2**, on both profiles. Keep at most the 2
  most recent checkpoints on disk at any time; delete older ones automatically
  as new ones are saved. The final adapter is retained separately, outside this
  rolling cap.
- **Smoke/verification run directories accumulate fast and need their own cap**
  (default: keep the most recent 3). Only `smoke`/verification runs are ever
  eligible for automatic deletion — never `debug` or `longer` runs, and never
  the most recent *failed* run of a kind before it's been inspected. Any
  deletion, automatic or manual, must be reported, not silent. See "Run
  Directory Cleanup" in `docs/PROJECT_SPEC.md`.
- No CUDA-only or MPS-only code path may be required to run — both must stay
  supported, gated by config/device resolution, not by editing code per machine.
- **Verify hardware precision support, don't silently substitute.** On
  `cuda_4gb`, `bf16` is checked against `torch.cuda.is_bf16_supported()` at
  startup and fails loudly if unsupported — it never silently falls back to
  fp16/fp32. Same principle as OOM handling: hiding what actually happened
  breaks reproducibility. See `docs/SPEC_CUDA_4GB.md` / `SPEC_MACOS_MPS.md`.

## Known, expected (not bugs)

- Preferred `num_generations` is **4**, on both profiles — better GRPO
  advantage-estimation quality than 2. Drop to `num_generations=2` only as a
  documented fallback when a hardware profile's memory budget can't support 4
  (verify with a smoke run, don't assume). At 2, the advantage signal is
  essentially binary and noticeably noisier — expected, not a defect, but not
  the preferred operating point either.
- SmolLM2-135M is tiny — expect frequent malformed/missing `<answer>` tags and
  low accuracy early on, especially pre-training.
- On `cuda_4gb`, memory margins are tight by design (4 GB dedicated VRAM). OOM
  is a real possibility if any single setting is increased without adjusting
  another (see `docs/SPEC_CUDA_4GB.md`). This is a profile characteristic to
  design around, not a bug to "fix" by quietly upsizing the GPU assumption.
- `accuracy_reward` staying near zero on `SmolLM2-135M-Instruct` is now a
  diagnosed model-capacity limitation, not a bug in reward/training logic.
  Rollout diagnostics on the fixed 200-prompt canonical manifest
  (`data/diagnostic_manifest_v1.json`) show only ~3.5% of groups have any
  reward variance to learn from; SFT warm-start at three escalating
  strengths didn't fix it (and regressed pass@4 to 0% at the strongest
  setting); a capability bakeoff shows `Qwen3-0.6B-Instruct` (non-thinking
  mode) dramatically outperforms SmolLM2 on the identical manifest. See
  `docs/PROJECT_SPEC.md`'s "Accuracy-Reward Constraints" and
  `docs/ACCURACY_IMPROVEMENT_PLAN.md` for full evidence. Evaluating a
  larger-model swap is now a sanctioned, evidence-gated next step (see
  `docs/PROJECT_SPEC.md`'s "Current Milestone" exception) — not the blanket
  "no larger model" this project started with.

## See also

- `docs/PROJECT_SPEC.md` — full shared spec, reward definitions,
  dataset/eval/checkpoint requirements, current milestone, completion criteria.
- `docs/SPEC_MACOS_MPS.md` — MacBook Pro M2 16GB hardware profile.
- `docs/SPEC_CUDA_4GB.md` — RTX 3050 4GB hardware profile.
