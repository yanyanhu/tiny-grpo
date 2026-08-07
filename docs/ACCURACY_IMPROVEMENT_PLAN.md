# Tiny GRPO Accuracy Improvement Plan

## Goal

Improve the usefulness of `accuracy_reward` for GRPO training of:

- Model: `HuggingFaceTB/SmolLM2-135M-Instruct`
- Dataset: `openai/gsm8k`
- Trainer: TRL `GRPOTrainer`
- Adaptation: LoRA
- Primary runtime: `cuda_4gb` — RTX 3050 Laptop GPU under WSL2
- Secondary runtime: `mps_16gb` — MacBook Pro M2

The existing pipeline is already complete and working. Do not redesign the project architecture or replace the current training, evaluation, checkpointing, configuration, logging, or hardware-profile systems.

The current problem is that exact-answer reward is almost always zero. With `num_generations=4`, nearly all completion groups appear to contain four incorrect answers, causing zero within-group variance for `accuracy_reward` and therefore little or no useful GRPO signal from that reward.

The work should establish whether the base model has enough latent GSM8K capability for exact-reward GRPO, add a supervised warm-start if required, and only then consider conservative reward shaping.

---

## Guiding Principles

- Diagnose before changing training.
- Do not assume that more steps or a higher learning rate will solve sparse reward.
- Keep exact-answer correctness as the primary target.
- Do not replace exact correctness with a proximity metric.
- Introduce only one major intervention at a time.
- Preserve deterministic train, validation, and test splits.
- Reuse the same validation prompts and sampling settings across comparisons.
- Do not tune against the test set.
- Keep the implementation compatible with both hardware profiles.
- Do not introduce a larger model, vLLM, distributed training, external tracking, or medical data.
- Every verification command must use the existing watchdog/timeout mechanism.
- Do not claim an experiment succeeded unless the actual command completed successfully.
- Never silently change reward weights, prompt templates, generation settings, or evaluation subsets between comparable runs.

---

# Phase 0 — Audit Scoring and Freeze the Comparison Set

## Objective

Rule out measurement artifacts before diagnosing a model-capability problem,
and establish one validation/diagnostic manifest that is independent of run
profile sizes.

## 0.1 Exact Numeric Equivalence

Canonicalize valid numeric answers before exact comparison using exact decimal
arithmetic, not binary floating-point tolerance. Numerically equivalent forms
such as `42`, `42.0`, `042`, `0.50`, and `0.500` must compare equal. Preserve
strict parsing: malformed text and non-finite values remain invalid.

Add CPU-only tests for equivalent decimal spellings, signed zero, commas,
negative values, and genuinely unequal values.

## 0.2 Failure Classification

Classify every diagnostic completion as one of:

- exact correct;
- valid numeric answer but incorrect;
- missing answer tag;
- malformed answer tag/content;
- invalid answer at the completion-length cap (likely truncation).

Report these categories separately. A completion cut off before its final tag
is not evidence of the same failure mode as a naturally terminated, incorrect
answer.

## 0.3 Canonical Diagnostic Manifest

Create and persist one versioned 200-example diagnostic manifest from the
official GSM8K train split. It must be disjoint from the largest currently
planned training prefix (1,024 examples), and must not depend on whether a run
uses the smoke, debug, or longer profile. All Base, SFT, and SFT+GRPO rollout
comparisons reuse these exact prompt IDs.

The 16-prompt smoke diagnostic is the first 16 IDs of this manifest, not a
separately sampled subset.

---

# Phase 1 — Establish a Proper GRPO Viability Baseline

## Objective

Measure whether the current model produces enough correct samples and mixed-reward completion groups for exact-answer GRPO to learn.

This phase must not perform training.

## 1.1 Add a Generation-Only Diagnostic Command

Add a command or module such as:

```bash
python -m tiny_grpo.diagnose_rollouts \
  --profile debug \
  --hardware cuda_4gb
```

It should:

- load the base model without the trained adapter;
- use the same prompt construction used by GRPO training;
- use the same answer extraction and reward functions;
- use the same intended sampling settings;
- generate `num_generations=4` completions per prompt;
- operate on a deterministic validation subset;
- perform no backward pass and no optimizer update;
- write structured results to a unique run directory.

Do not duplicate prompt, dataset, parsing, or reward logic. Reuse the existing project modules.

## 1.2 Diagnostic Dataset Size

Support a configurable number of prompts.

Recommended sequence:

```text
Smoke diagnostic:  8–16 prompts
Main diagnostic:   100–200 prompts
```

Use the canonical versioned diagnostic manifest established in Phase 0. Do not
derive diagnostic prompts from profile-dependent `train_size`/`val_size`
settings and do not create a new random subset per run.

## 1.3 Required Metrics

Record:

- number of unique prompts;
- number of completions per prompt;
- total completion count;
- first-sample pass@1 exact-answer accuracy;
- sample exact accuracy (correct completions divided by all completions);
- pass@4: fraction of prompts with at least one correct completion;
- valid-format rate;
- parse-failure rate;
- exact accuracy conditional on valid format;
- average exact reward;
- average format reward;
- mean total reward;
- average within-group reward standard deviation;
- fraction of groups with zero total-reward standard deviation;
- fraction of groups with zero exact-reward standard deviation;
- fraction of groups containing both correct and incorrect completions;
- fraction of groups where all completions are wrong;
- fraction of groups where all completions receive the same total reward;
- average and maximum completion length;
- truncation rate;
- runtime;
- process memory;
- device-memory statistics.
- raw numerator/denominator counts and a binomial confidence interval for
  pass@1, pass@k, and mixed exact-reward group rate.

If the installed TRL version exposes an equivalent metric such as `frac_reward_zero_std`, use the same terminology where practical, but compute the diagnostic independently so it is available before training.

## 1.4 Save Per-Prompt Results

For every prompt, save:

- prompt ID;
- prompt text or stable prompt reference;
- gold answer;
- all generated completions;
- extracted answer from each completion;
- exact reward;
- format reward;
- total reward;
- group mean reward;
- group reward standard deviation;
- whether the group contains mixed exact rewards;
- whether any completion is correct.

Use JSONL or another existing structured log format.

## 1.5 Diagnostic Report

Produce a concise human-readable report such as:

```text
Prompts: 200
Completions per prompt: 4
Total completions: 800

Pass@1 exact: 1.0%
Pass@4 exact: 3.5%
Valid format: 51.0%
Exact among valid format: 2.0%

Mixed exact-reward groups: 1.5%
Zero exact-reward-std groups: 98.5%
Zero total-reward-std groups: 62.0%
```

`pass@1` means correctness of the first generated completion for each prompt.
`sample exact accuracy` means correctness across all generated completions.
`pass@k` means the fraction of prompts with at least one exact completion among
the `k` generated samples. Seed each prompt deterministically so results do not
change merely because a larger prefix of the manifest is evaluated.

## 1.6 Decision Gate

Use the following as practical guidance, not hard-coded scientific thresholds:

```text
Mixed exact-reward groups below 1%
    Exact-only GRPO is not meaningfully bootstrapped.

Mixed exact-reward groups around 1–5%
    Signal exists but is weak and likely unstable.

Mixed exact-reward groups above 5%
    Exact-reward GRPO has a more credible starting signal.
```

Do not automatically start GRPO based on these thresholds. Report the result and continue to Phase 2 unless the exact signal is already clearly healthy.

If a result lies near a decision threshold, repeat the main diagnostic with
two or three predeclared sampling seeds. Treat thresholds as guidance and
report counts/confidence intervals rather than over-interpreting a single
event in a 100–200 prompt sample.

## Phase 0/1 execution record — 2026-08-02

Implemented exact decimal canonicalization, failure classification, the
versioned 200-prompt manifest (`data/diagnostic_manifest_v1.json`), pure group
metrics with Wilson intervals, and the generation-only diagnostic command.

Base-model results on `cuda_4gb`, sampling seed 42, four generations per
prompt:

| Diagnostic | Prompts | Pass@1 | Pass@4 | Sample exact | Mixed exact groups | Zero exact std | Valid format | Truncation |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Smoke, 128 tokens | 16 | 0.0% | 6.25% | 1.56% | 6.25% | 93.75% | 40.63% | 46.88% |
| Main, 128 tokens | 200 | 0.5% | 3.5% | 1.0% | 3.5% | 96.5% | 42.25% | 42.13% |
| Same smoke prompts, 256 tokens | 16 | 0.0% | 6.25% | 1.56% | 6.25% | 93.75% | 51.56% | 26.56% |

For the 200-prompt diagnostic, pass@4/mixed exact groups were 7/200 with
95% Wilson interval 1.71–7.05%; exact completions were 8/800. Runtime was
734.0 seconds. Peak process RSS was 1551.7 MiB; CUDA peak allocated was 320.2
MiB and reserved memory was 350 MiB.

The 256-token comparison reduced truncation and improved valid formatting but
did not add an exact answer on the fixed 16-prompt comparison. The capability
signal is therefore real but weak, with 96.5% of main-diagnostic groups
providing no exact-reward variance. Phase 1 gates to the supervised LoRA
warm-start in Phase 2; more GRPO-only steps are not the recommended next
experiment.

Run directories:

- `outputs/diagnostic_debug_20260802_155623` (16 prompts, 128 tokens)
- `outputs/diagnostic_debug_20260802_155848` (200 prompts, 128 tokens)
- `outputs/diagnostic_debug_20260802_161149` (16 prompts, 256 tokens)

---

# Phase 2 — Add a Supervised LoRA Warm-Start

## Objective

Teach the model basic GSM8K solution behavior and the required answer format before asking GRPO to optimize exact correctness.

The SFT stage should be small, explicit, and integrated into the existing project rather than becoming a separate general-purpose training framework.

## 2.1 Add an SFT Training Entry Point

Use:

```bash
uv run python train_sft.py \
  --profile debug \
  --hardware cuda_4gb
```

The SFT path should:

- use the same base model;
- use LoRA;
- use the existing deterministic training split;
- use the current prompt format;
- train on GSM8K gold reasoning plus the final required answer tag;
- save an adapter compatible with the GRPO stage;
- use the existing run-directory and metadata conventions;
- support checkpointing and resume where practical;
- remain compatible with CUDA and MPS profiles.

Use TRL `SFTTrainer` or the existing Transformers training stack, whichever introduces the least new complexity and best matches the pinned dependencies.

Inspect the installed package versions before assuming the current `SFTTrainer` API.

## 2.2 SFT Target Format

The supervised assistant target should end with exactly:

```text
<answer>N</answer>
```

Do not append text after the closing tag.

The target should preserve the GSM8K reasoning when available, followed by the required final format.

Example structure:

```text
We need to calculate ...
...
Therefore the answer is:

<answer>42</answer>
```

Do not include the gold answer in the user prompt.

## 2.3 Loss Masking

Where supported, apply supervised loss only to assistant response tokens.

Prompt and system-message tokens should not contribute to the SFT loss.

If the installed trainer stack cannot reliably implement assistant-only loss, document the limitation rather than silently claiming it is active.

## 2.4 Initial SFT Profiles

Add conservative SFT configurations.

### SFT smoke

Purpose:

- validate data formatting;
- validate LoRA training;
- validate adapter save/load;
- validate the RTX 3050 memory path.

Suggested characteristics:

- very small training subset;
- a few optimizer steps;
- batch size appropriate for `cuda_4gb`;
- short timeout;
- no claim of quality improvement.

### SFT debug

Purpose:

- produce the first potentially useful warm-start adapter.

Suggested starting point:

- a few hundred to approximately 1,000 training examples;
- one epoch or a capped number of steps;
- conservative sequence length;
- LoRA only;
- BF16 only if the existing CUDA capability and smoke tests support it;
- otherwise FP16;
- gradient checkpointing according to the hardware profile.

Do not launch a multi-hour SFT run as routine verification.

## 2.5 SFT Evaluation

Evaluate three model states using exactly the same validation prompts and generation settings:

```text
Base model
Existing GRPO-only adapter, if retained for comparison
SFT adapter
```

Record:

- pass@1;
- pass@4;
- valid-format rate;
- parse-failure rate;
- mixed exact-reward group rate;
- zero exact-reward-standard-deviation group rate;
- average completion length;
- sample outputs.

The primary SFT gate is not merely higher average reward. It is whether the model now generates correct completions often enough to create mixed GRPO groups.

## 2.6 SFT Completion Criteria

The SFT stage is ready to hand off to GRPO when:

- the adapter loads successfully;
- validation format compliance is stable;
- pass@1 or pass@4 improves over the base model;
- mixed exact-reward groups occur repeatedly rather than as one isolated lucky hit;
- the rollout diagnostic can run from the SFT adapter;
- all measurements use the same validation manifest and sampling settings.

## 2.7 Implementation and Smoke Evidence (2026-08-02)

Stage 2 mechanics are implemented in `train_sft.py`, `tiny_grpo/sft_config.py`,
and `tiny_grpo/sft_data.py`. The implementation uses the installed TRL
`SFTTrainer` prompt/completion path with `completion_only_loss=True`. The
SmolLM2 chat template does not expose assistant-token generation masks, so
`assistant_only_loss` is deliberately disabled rather than claimed active;
completion-only masking supervises the assistant completion and masks the
prompt.

Implemented behavior:

- `sft_smoke` and `sft_debug` profiles composed with both hardware profiles;
- deterministic existing GSM8K splits and current prompt format;
- gold reasoning targets ending in one canonical answer tag;
- calculator annotation removal and no gold answer in the user prompt;
- preflight refusal if any supervised sequence exceeds `max_sequence_length`;
- LoRA-only training, JSONL/memory logging, baseline/post evaluation;
- rolling checkpoint retention capped at two, resume support, and a separately
  retained final adapter.

Verification results:

- final full tests: 216 passed, 7 hardware-dependent skips in the restricted process;
- real RTX 3050 CUDA integration: 3 passed;
- corrected focused SFT tests: 19 passed;
- successful run: `outputs/sft_smoke_20260802_190238`;
- config: 64 train, 16 validation, batch size 1, accumulation 8, BF16,
  gradient checkpointing, sequence length 1024, learning rate `2e-4`, 3 steps;
- audited train length: 315–543 tokens; validation: 336–475; zero over limit;
- logged CUDA memory: about 284 MiB allocated after steps, with a 527 MiB
  maximum-allocation watermark and 564 MiB reserved;
- checkpoints retained: `checkpoint-2` and `checkpoint-3`; final adapter saved;
- independent fresh-process base-model plus adapter reload and held-out
  generation succeeded.

The 16-example smoke generation comparison changed valid-format rate from
18.8% to 50.0%, but exact accuracy remained 0% before and after. This is not
evidence of a quality improvement: three optimizer steps and 16 examples are
only a mechanical smoke test. Phase 2 has therefore passed its implementation
and memory-safety gate, but not its quality/handoff criteria. The next planned
experiment is the explicit `sft_debug` run, followed by the fixed-manifest
rollout diagnostic from its adapter and a Base-versus-SFT comparison.

## 2.8 SFT Debug Evidence (2026-08-02)

The first deliberate SFT debug experiment completed successfully:

- run: `outputs/sft_debug_20260802_201146`;
- 256 training examples, 32 validation examples, one effective epoch;
- batch size 1, gradient accumulation 8, BF16, gradient checkpointing;
- sequence length 1024, learning rate `2e-4`, 32 optimizer steps;
- training runtime: 91.8 seconds;
- final aggregate training loss: 1.019;
- validation loss by checkpoint: 1.017, 0.992, 0.979, 0.976;
- all audited sequences fit: train 307–558 tokens and validation 300–531;
- CUDA maximum-allocation watermark: about 530 MiB; maximum reserved: 568 MiB;
- retained checkpoints: `checkpoint-24` and `checkpoint-32`;
- final adapter saved successfully.

On the run's 32-example, one-sample generation evaluation, exact accuracy
changed from 0/32 to 1/32 (3.1%), format rate from 40.6% to 46.9%, and parse
failure rate from 59.4% to 53.1%. This isolated exact hit was then checked with
the canonical first 16 prompts × 4 samples in
`outputs/diagnostic_debug_20260802_204641`.

Fixed-manifest comparison at 128 completion tokens:

| Metric | Base | SFT debug |
|---|---:|---:|
| pass@1 | 0.0% | 0.0% |
| pass@4 | 6.25% | 6.25% |
| sample exact accuracy | 1.56% | 1.56% |
| mixed exact-reward groups | 6.25% | 6.25% |
| zero exact-reward-std groups | 93.75% | 93.75% |
| valid format | 40.63% | 34.38% |
| truncation | 46.88% | 48.44% |

Therefore this 256-example, one-epoch adapter passes the implementation and
memory gates but does not pass the Phase 2 quality/handoff gate. The one exact
validation hit is not corroborated by improved fixed-manifest pass@4 or mixed
groups. Do not start Phase 3 GRPO from this adapter as the recommended next
experiment. Increase SFT exposure conservatively (prefer a larger deterministic
subset and/or more than one epoch), run smoke-level memory verification for the
chosen profile, and repeat the identical fixed-manifest diagnostic before the
GRPO handoff decision.

## 2.9 Stronger SFT Evidence (2026-08-02)

A distinct `sft_stronger` profile was added so this experiment is reproducible
without ad-hoc overrides. It computes optimizer steps from each hardware
profile's effective batch, giving both CUDA and MPS two effective epochs over
the same reserved training set. The CUDA experiment used:

- run: `outputs/sft_stronger_20260802_205646`;
- 1,024 training examples and 64 validation examples;
- two effective epochs, 256 optimizer steps;
- batch size 1, gradient accumulation 8, BF16, gradient checkpointing;
- sequence length 1,024 and learning rate `2e-4` with linear decay;
- evaluation/checkpoint cadence every 64 steps and two-checkpoint retention;
- training runtime: 662.9 seconds;
- aggregate training loss: 0.942;
- validation loss: 0.915, 0.893, 0.887, 0.882;
- audited train length 292–663 tokens and validation 306–559, zero over limit;
- CUDA maximum-allocation watermark about 534 MiB and maximum reserved 626 MiB;
- retained checkpoints: `checkpoint-192` and `checkpoint-256`;
- final adapter saved successfully.

The 64-prompt single-sample generation comparison did not improve exact
accuracy: Base scored 1/64 (1.6%) and the stronger adapter 0/64. Format rate
improved from 45.3% to 56.3%, parse failures fell from 54.7% to 43.8%, and mean
reward changed from 0.106 to 0.113.

The canonical 16-prompt × 4-sample diagnostic is stored at
`outputs/diagnostic_debug_20260802_211819`:

| Metric | Base | SFT debug | SFT stronger |
|---|---:|---:|---:|
| pass@1 | 0.0% | 0.0% | 0.0% |
| pass@4 | 6.25% | 6.25% | 0.0% |
| sample exact accuracy | 1.56% | 1.56% | 0.0% |
| mixed exact-reward groups | 6.25% | 6.25% | 0.0% |
| zero exact-reward-std groups | 93.75% | 93.75% | 100.0% |
| valid format | 40.63% | 34.38% | 59.38% |
| truncation | 46.88% | 48.44% | 37.50% |

The stronger SFT profile clearly improves format compliance and reduces
truncation, and its supervised validation loss improves consistently. It does
not improve exact arithmetic capability on the fixed diagnostic, however: all
38 valid-format completions were incorrect and no GRPO group had exact-reward
variance. Phase 2's quality/handoff gate therefore still fails, and Phase 3
exact-reward GRPO should not start from this adapter. Further undirected SFT
scaling is not justified by this result alone; the next plan should inspect
task difficulty/target behavior and consider the easy-task curriculum gate in
Phase 5 before spending another longer training run.

## 2.10 Small-Model Capability Bakeoff (2026-08-02)

Stage 2 evidence indicated a model-capacity bottleneck, so the rollout
diagnostic gained explicit `--model-id` and `--chat-template-mode` controls.
This keeps candidate selection generation-only and records Qwen-style thinking
behavior rather than silently accepting a tokenizer default. All candidates
used the same manifest prefix, prompts, four samples, per-prompt seeds,
128-token cap, and sampling parameters.

The 16-prompt candidate gate produced:

| Model | Template mode | pass@4 | Sample exact | Mixed groups | Valid format | Truncation | CUDA max allocated |
|---|---|---:|---:|---:|---:|---:|---:|
| SmolLM2-135M-Instruct | default | 6.25% | 1.56% | 6.25% | 40.63% | 46.88% | 316 MiB |
| Qwen2.5-0.5B-Instruct | default | 6.25% | 1.56% | 6.25% | 6.25% | 73.44% | 1,018 MiB |
| Qwen3-0.6B | non-thinking | 43.75% | 10.94% | 43.75% | 81.25% | 10.94% | 1,358 MiB |

Qwen2.5 did not pass the small gate under the common experiment settings, so
it was not expanded. Qwen3-0.6B passed decisively and was evaluated on the
full canonical 200-prompt manifest in
`outputs/diagnostic_debug_20260802_213816`.

| Metric | SmolLM2-135M Base | Qwen3-0.6B non-thinking |
|---|---:|---:|
| first-sample pass@1 | 0.5% (1/200) | 15.5% (31/200) |
| pass@4 | 3.5% (7/200) | 38.0% (76/200) |
| sample exact accuracy | 1.0% (8/800) | 16.63% (133/800) |
| mixed exact-reward groups | 3.5% (7/200) | 36.0% (72/200) |
| zero exact-reward-std groups | 96.5% | 64.0% |
| valid format | 42.25% | 87.38% |
| truncation | 42.13% | 6.88% |
| runtime | 734.0 s | 648.8 s |
| CUDA maximum allocated | 320 MiB | 1,382 MiB |
| CUDA reserved | 350 MiB | 1,424 MiB |

Qwen3 confidence intervals were pass@1 11.14–21.16%, pass@4 31.56–44.89%,
sample exact accuracy 14.21–19.36%, and mixed groups 29.67–42.86%. These are
well separated from the corresponding SmolLM2 intervals. Qwen3 therefore
passes the base-model capability gate and provides repeated exact-reward
variation suitable for GRPO. Generation memory also leaves substantial room
within 4 GiB, but training feasibility must still be measured independently.

Recommended next action: add Qwen3-0.6B as an explicit model profile with
non-thinking chat-template behavior, verify its LoRA target modules and
completion-only SFT formatting, and run only an SFT/GRPO memory smoke before
choosing whether a warm-start is needed. Do not infer backward-pass memory from
the successful generation diagnostic.

## 2.11 Qwen3 Training-Memory Gate (2026-08-07)

The recommended integration and both training-memory smokes were completed on
the RTX 3050 4GB profile:

- Added the explicit `qwen3_0_6b` model profile for `Qwen/Qwen3-0.6B` with
  non-thinking chat-template behavior. SmolLM2 remains the default.
- Centralized model ID, template kwargs, and LoRA target modules in the typed
  profile. Non-default profiles receive model-specific run names so resume
  cannot silently cross model families.
- Verified on the actually loaded Qwen3 model that `q_proj`, `k_proj`,
  `v_proj`, and `o_proj` all exist.
- A three-step SFT smoke completed in
  `outputs/sft_smoke_qwen3_0_6b_20260807_203020`. It peaked at 2,189 MiB CUDA
  allocated, saved checkpoints and a final adapter, and had no sequence-length
  audit failures. Its 16-example evaluation changed accuracy from 18.8% to
  31.2%, but that tiny smoke evaluation is not a model-quality comparison.
- A one-step direct GRPO smoke completed in
  `outputs/smoke_qwen3_0_6b_20260807_204037` with four generations,
  `beta=0.04`, and gradient checkpointing. It peaked at 2,809 MiB CUDA
  allocated, logged a non-zero exact-reward standard deviation, and saved
  checkpoint 1 plus the final adapter. The 16-example evaluation changed
  accuracy from 18.8% to 25.0% and format rate from 87.5% to 93.8%; these are
  execution-smoke observations, not an improvement claim.

The GRPO training-time gate therefore passes with about 1.25 GiB between the
measured allocated watermark and the nominal 4 GiB device capacity. PyTorch's
expandable-segments reserved-memory counter exceeded physical capacity during
the run, so allocated memory is the interpretable feasibility watermark; the
run itself completed without OOM.

Decision: use base Qwen3 non-thinking for the next controlled direct-GRPO
experiment. The full 200-prompt diagnostic already showed 36% mixed
exact-reward groups, so SFT is not required to bootstrap reward variance. Keep
the Qwen3 SFT path available for a later controlled comparison, but do not make
it the default starting point based on smoke metrics.

## 2.12 Qwen3 Direct-GRPO Debug Run (2026-08-07)

The first controlled base-Qwen3 debug run completed successfully in
`outputs/debug_qwen3_0_6b_20260807_205205` under a 7,200-second hard timeout.
It used the unchanged `cuda_4gb` debug configuration: 50 steps, 256 training
examples, 32 validation examples, four generations, `beta=0.04`, BF16, and
gradient checkpointing. It started from the base model with a fresh LoRA
adapter, not from SFT.

Execution evidence:

- trainer runtime: 3,334.5 seconds;
- maximum CUDA memory allocated: 3,108 MiB;
- final status: completed without OOM or timeout;
- retained checkpoints: 40 and 50, respecting the two-checkpoint cap;
- final adapter and optimizer-bearing checkpoints were saved;
- intermediate validation exact-reward means at steps 10/20/30/40/50 were
  32.8%, 35.9%, 36.7%, 28.9%, and 29.7%, respectively;
- intermediate zero-reward-variance group fractions were 46.9%, 43.8%,
  37.5%, 31.2%, and 31.2%.

The separate fixed 32-example, single-generation comparison was:

| Metric | Base | Post-GRPO |
|---|---:|---:|
| Exact accuracy | 28.1% (9/32) | 31.2% (10/32) |
| Valid format | 90.6% | 93.8% |
| Parse failure | 9.4% | 6.2% |
| Mean reward | 0.4625 | 0.5000 |

This is a successful and stable training run, but the one-example accuracy
difference is not sufficient evidence of model-quality improvement. The
intermediate trainer evaluations also fluctuate rather than improve
monotonically. The next quality gate is the unchanged canonical 200-prompt,
four-generation rollout diagnostic loaded from this run's final adapter. Use
that comparison to measure pass@1, pass@4, sample exact accuracy, and mixed
exact-reward groups before choosing a longer GRPO run or another intervention.

## 2.13 Canonical Diagnostic after Direct GRPO (2026-08-07)

The prescribed final-adapter diagnostic completed in
`outputs/diagnostic_debug_20260807_215822` under the same 1,800-second timeout
as the base-Qwen3 diagnostic. A programmatic configuration comparison found no
mismatch in manifest version or prompt IDs, prompt count, generation count,
completion cap, temperature, top-p, top-k, sampling seed/formula, model ID, or
non-thinking template mode. The only intended difference was loading the
final adapter from the 50-step debug run.

| Metric | Base Qwen3 | After 50-step GRPO | Change |
|---|---:|---:|---:|
| First-sample pass@1 | 15.5% (31/200) | 17.0% (34/200) | +1.5 pp |
| pass@4 | 38.0% (76/200) | 39.5% (79/200) | +1.5 pp |
| Sample exact accuracy | 16.63% (133/800) | 18.63% (149/800) | +2.0 pp |
| Mixed exact-reward groups | 36.0% (72/200) | 37.5% (75/200) | +1.5 pp |
| Zero exact-reward-std groups | 64.0% | 62.5% | -1.5 pp |
| Valid format | 87.38% | 87.75% | +0.37 pp |
| Truncation | 6.88% | 7.75% | +0.87 pp |
| Mean total reward | 0.3410 | 0.3618 | +0.0208 |

Adapter runtime was 1,001.3 seconds; CUDA maximum allocated memory was 1,390.6
MiB. The run completed without OOM or timeout and saved all 200 prompt records
and 800 completions.

The direction is mildly positive, but the coverage metrics do not establish a
clear improvement. Paired transitions were 7 adapter-only versus 4 base-only
correct prompts for pass@1, and 12 versus 9 for pass@4 (two-sided exact
McNemar p-values 0.55 and 0.66). At the prompt-cluster level, the adapter
produced more correct samples on 29 prompts, fewer on 18, and tied on 153
(two-sided sign p=0.14). Treating all 800 completions as independent would
overstate evidence because four samples share each prompt.

Decision: record the 50-step run as a stable, modest directional result, not a
demonstrated quality improvement. Do not jump directly to the 200-step longer
profile on this evidence alone. A sensible next planning step is a bounded
replication or a predeclared intermediate-duration run with a non-decaying or
slower-decaying learning-rate schedule, followed by the same canonical paired
diagnostic. Change only one training variable and retain the base and 50-step
results as fixed references.

---

# Phase 3 — Continue the SFT Adapter with Exact-Reward GRPO

## Objective

Test whether GRPO can improve an SFT-warmed model using the original exact and format rewards.

## 3.1 Adapter Initialisation

Add a clear configuration field such as:

```yaml
initial_adapter_path: runs/.../final_adapter
```

The GRPO training path must support:

- base model plus no adapter;
- base model plus an SFT LoRA adapter;
- explicit logging of the adapter source;
- validation that the adapter matches the base model;
- clear failure if the path is missing or incompatible.

Do not merge the adapter into the base model unless required by the pinned TRL/PEFT stack.

Document whether GRPO continues updating the same LoRA parameters or adds a new adapter. Prefer continuing the same adapter if the toolchain supports it cleanly.

## 3.2 Initial GRPO Configuration

Start with:

```text
num_generations = 4
accuracy reward = 1.0
format reward = existing small weight
beta = profile default (currently 0.04 on cuda_4gb)
```

Do not initially add numeric-closeness shaping.

For the current `cuda_4gb` profile the default is `beta=0.04`, not zero. Use
`0.04` initially. A `beta=0` run is an explicit, separately documented
override rather than the profile default.

Use a short GRPO debug run before any longer run.

## 3.3 Required Training Logs

During GRPO, log:

- total mean reward;
- accuracy reward mean;
- format reward mean;
- reward standard deviation;
- fraction of zero-standard-deviation groups;
- mixed exact-reward group fraction where available;
- policy loss;
- gradient norm;
- learning rate;
- completion length;
- valid-format rate where practical;
- CUDA memory allocated, reserved, and peak allocated;
- elapsed time and step time.

Continue the existing concise console progress output.

The reward metrics and validation accuracy are more important than policy loss alone.

## 3.4 Evaluation

Compare:

```text
Base
SFT
SFT + GRPO
```

using identical validation prompts and generation settings.

Do not compare only against the earlier GRPO-only run.

## 3.5 Success Criteria

GRPO is useful if at least one of the following occurs without material regression elsewhere:

- validation pass@1 improves over SFT;
- validation pass@4 improves over SFT;
- mixed exact-reward group frequency improves;
- exact reward increases while format rate remains stable;
- independently sampled qualitative outputs show better arithmetic rather than merely more aggressive answer guessing.

A lower policy loss is not sufficient evidence.

---

# Phase 4 — Add Conservative Reward Shaping Only If Needed

## Trigger

Implement this phase only if:

- SFT clearly improves formatting and plausibility;
- exact reward remains too sparse for stable GRPO;
- mixed exact-reward groups remain very rare;
- exact-only GRPO does not improve validation accuracy.

## 4.1 Add a Bounded Numeric-Proximity Reward

Add a separately testable reward function.

Requirements:

- exact correctness remains the dominant reward;
- all wrong answers receive substantially less reward than an exact answer;
- proximity reward is bounded;
- proximity reward handles negative values and zero safely;
- malformed or missing answers receive zero proximity reward;
- no NaN or infinity can be produced;
- reward scale is configurable;
- default weight is small.

Suggested conceptual form:

```python
distance = abs(log1p(abs(prediction)) - log1p(abs(gold)))
closeness = exp(-distance / temperature)
```

The final contribution should be capped to approximately:

```text
0.05–0.10 maximum
```

Do not use unbounded inverse distance.

Do not treat proximity as a correctness metric.

## 4.2 Reward Hierarchy

Preserve a clear ordering:

```text
Exact correct answer:
    dominant reward

Correct output format:
    small auxiliary reward

Numeric proximity:
    smaller bootstrapping reward
```

An incorrect answer, regardless of proximity, must not receive a reward close to an exact answer.

## 4.3 Unit Tests

Add tests for:

- exact match;
- near positive values;
- distant positive values;
- negative prediction and positive gold;
- zero gold;
- zero prediction;
- malformed text;
- missing tag;
- very large values;
- commas and decimal normalisation;
- monotonicity: closer answers receive at least as much proximity reward;
- bounded maximum;
- exact answer still receives the dominant total reward.

## 4.4 Controlled Comparison

Run:

```text
SFT + exact GRPO
versus
SFT + shaped GRPO
```

Keep constant:

- adapter starting point;
- dataset;
- validation prompts;
- seed;
- generation settings;
- training steps;
- learning rate;
- LoRA configuration;
- hardware profile.

Report whether shaping improves validation exact accuracy, not merely average shaped reward.

---

# Phase 5 — Easier-Task Curriculum If the Model Still Cannot Learn

## Trigger

Use this only if the SFT warm-start still produces negligible pass@4 and almost no mixed exact-reward groups.

## 5.1 Add a Deterministic Easy-GSM8K Training View

Create a training-only filtered subset using transparent rules such as:

- shorter gold solution;
- fewer arithmetic operations;
- fewer numeric values in the question;
- no fractions or percentages in the first stage;
- shorter target sequence.

Do not use validation or test performance to select examples.

Persist:

- filtering rules;
- selected IDs;
- counts;
- seed;
- subset version.

## 5.2 Curriculum Stages

A possible progression:

```text
Stage 1: easiest arithmetic word problems
Stage 2: broader easy-GSM8K subset
Stage 3: normal GSM8K training subset
Stage 4: exact-reward GRPO
```

Evaluate after every stage using the unchanged full validation subset.

Do not claim improvement based only on performance against the easier training subset.

---

# Optional Diagnostic — Generation Count

Increasing `num_generations` is not the first intervention on the 4GB GPU.

However, support a generation-only diagnostic using:

```text
num_generations = 8
```

with:

- no backward pass;
- no optimizer;
- a very small prompt batch;
- a conservative completion cap;
- timeout protection.

Compare pass@4 and pass@8.

Interpretation:

```text
pass@8 ≈ pass@4 ≈ 0
    More rollout samples are unlikely to solve the capability problem.

pass@8 materially exceeds pass@4
    More sampling may help after SFT, but training-memory feasibility must
    still be tested separately.
```

Do not infer GRPO training feasibility from generation-only memory use.

---

# Configuration Additions

Add only the fields needed to support this work.

Potential fields:

```yaml
diagnostic:
  num_prompts:
  num_generations:
  save_all_completions:

sft:
  enabled:
  train_subset_size:
  max_steps:
  num_train_epochs:
  learning_rate:
  max_sequence_length:
  assistant_only_loss:
  initial_adapter_path:

grpo:
  initial_adapter_path:
  num_generations:
  beta:

rewards:
  accuracy_weight:
  format_weight:
  proximity_enabled:
  proximity_weight:
  proximity_temperature:
```

Do not create a new configuration system. Extend the existing typed configuration and run-profile composition.

Validate incompatible settings clearly.

---

# Testing Requirements

Add CPU-only tests for:

- rollout-group metric calculations;
- pass@k calculations;
- mixed-reward group detection;
- zero-standard-deviation group calculation;
- SFT target formatting;
- no text after `</answer>`;
- SFT dataset construction;
- adapter-path configuration validation;
- proximity reward if Phase 4 is implemented.

Integration tests should remain small and hardware-specific.

CUDA tests should skip when CUDA is unavailable.

MPS tests should skip when MPS is unavailable.

No CPU unit test may download or load the model.

---

# Experiment Tracking Requirements

Every diagnostic or training run must record:

- git commit;
- hardware profile;
- run profile;
- base model;
- initial adapter path and source run;
- resolved configuration;
- package versions;
- dataset manifest;
- prompt-template version;
- reward-function version and weights;
- sampling parameters;
- random seed;
- completion count;
- timeout used;
- final status;
- relevant accuracy, reward, and zero-variance metrics.

Comparable runs must use the same validation manifest.

Do not overwrite previous results.

---

# Recommended Execution Order

Execute in this order:

1. Run existing unit tests.
2. Run the existing CUDA integration tests.
3. Audit and test exact numeric canonicalization and failure categories.
4. Persist the canonical 200-example diagnostic manifest.
5. Implement rollout diagnostics and pure metric tests.
6. Run the 16-prompt diagnostic.
7. Run the 200-prompt baseline diagnostic at completion length 128.
8. If truncation is material, compare a small fixed subset at length 256
   (generation-only; do not infer GRPO memory feasibility from it).
9. Review pass@4, mixed exact-reward group rate, raw counts, and confidence intervals.
10. Implement SFT data preparation and unit tests.
11. Implement the SFT training entry point.
12. Run SFT smoke.
13. Run a short SFT debug experiment.
14. Run the same rollout diagnostic from the SFT adapter.
15. Compare Base versus SFT.
16. Continue the SFT adapter with a short exact-reward GRPO run using the profile-default beta.
17. Compare Base versus SFT versus SFT+GRPO.
18. Implement conservative proximity shaping only if exact GRPO remains starved.
19. Consider an easy-task curriculum only if SFT still cannot bootstrap correct samples.

Do not jump directly to Phase 4 or Phase 5 without recording the earlier diagnostic results.

---

# Deliverables

At the end of the work, report:

- files created or modified;
- tests added;
- exact commands executed;
- timeout used for each command;
- command outcome;
- CUDA memory use;
- diagnostic dataset size;
- base pass@1 and pass@4;
- base mixed exact-reward group rate;
- base zero-reward-standard-deviation rate;
- SFT configuration;
- SFT pass@1 and pass@4;
- SFT mixed exact-reward group rate;
- GRPO configuration;
- SFT+GRPO pass@1 and pass@4;
- whether reward shaping was implemented;
- unresolved issues;
- recommended next experiment.

Do not report model improvement unless the fixed validation evaluation actually demonstrates it.
