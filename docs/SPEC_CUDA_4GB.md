# Hardware Profile: `cuda_4gb` — RTX 3050 Laptop GPU, 4 GB VRAM

> Concrete settings for this hardware profile. For everything else (reward
> math, dataset handling, evaluation, testing, checkpointing, milestone), see
> `docs/PROJECT_SPEC.md`. This doc only covers what's specific to running on
> this machine.
>
> **Read this before changing any default.** 4 GB of *dedicated* VRAM is
> genuinely tight for this workload, even at 135M parameters — the base model,
> generation activations for multiple completions per prompt, and LoRA
> optimizer state all compete for the same small budget (the KL reference
> computation itself is cheap under LoRA — see "Reference Model / KL
> Coefficient" below). Treat this profile as memory-constrained-first:
> validate any config change with a short smoke run before a longer one, and
> don't assume settings that are safe on `mps_16gb` are safe here.

## Target Environment

- Laptop with NVIDIA GeForce RTX 3050 (mobile, Ampere architecture)
- 4 GB **dedicated** VRAM (not shared with system RAM the way Apple unified
  memory is — but 4 GB is a hard, small ceiling)
- Linux or Windows with a CUDA-enabled PyTorch install
- PyTorch **CUDA** backend

## Device & Environment Setup

- Device: `cuda`
- No MPS fallback env var is relevant here. Instead, set the CUDA allocator
  to reduce fragmentation-driven OOMs on a small VRAM budget:

  ```
  export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
  ```

- vLLM: disabled on this profile too, **by explicit project decision**, not
  because it's technically impossible — vLLM does support CUDA. Keeping it
  disabled is a "keep it simple" choice consistent with the rest of the
  project's scope. If vLLM is reconsidered later for this profile, do so as a
  deliberate scope change, not as an ad hoc addition.

## Precision

Unlike the MPS profile, do not start with fp32. CUDA's fp16/bf16 support is
mature, and fp32 training at this VRAM budget is unlikely to leave enough
headroom for anything beyond a trivial batch size.

1. **bf16 mixed precision** — preferred starting point. RTX 3050 (Ampere) has
   bf16 tensor core support.
2. **fp16 mixed precision** — use if bf16 isn't behaving well for some reason;
   CUDA fp16 training is mature and well-trodden, unlike MPS fp16.
3. fp32 is a fallback only for debugging a suspected precision-related bug,
   not a normal operating mode on this profile — expect to need a much smaller
   batch size if you drop to it.

**Verified at startup, not assumed:** before training starts, the code checks
`torch.cuda.is_bf16_supported(including_emulation=False)` — real Ampere+
tensor-core support, not degraded software emulation — whenever this profile's
resolved precision is `bf16`. If unsupported, it **fails loudly with a clear
error** rather than silently substituting fp16 or fp32. This matches the
project's "no silent fallback" stance (see Out-of-Memory Handling below):
if this specific GPU can't do real bf16, that's something to see and decide
about explicitly, not something the code should quietly work around. fp16 has
no equivalent hardware-capability query on CUDA (it's supported on effectively
every CUDA GPU), so it isn't gated the same way.

## Reference Model / KL Coefficient (`beta`)

**Default to a small non-zero `beta` (0.04, matching `mps_16gb`) on this
profile.** This project always trains through LoRA (see `docs/PROJECT_SPEC.md`
— "Adaptation: LoRA"), and TRL never loads a second full model copy for the KL
reference when the model is a PEFT model: it either disables the adapter to
recover the base model's log probs, or clones a tiny extra adapter for
`beta != 0` (confirmed against the installed `trl` source). The "second full
model roughly doubles memory footprint" concern that previously justified
`beta=0` here doesn't apply once LoRA is in use — the reference computation's
memory cost is negligible on either hardware profile.

`beta=0` (no reference at all) is still fully supported as an explicit,
deliberate mode if you want to skip the KL term entirely — just no longer the
*default* on this profile. If memory pressure appears for other reasons:
- treat further precision/batch/length changes as the fix, not `beta`;
- validate with a smoke run and check CUDA memory stats before trusting any
  change won't OOM mid-run.

## Gradient Checkpointing

**Default ON for this profile** (not optional, unlike `mps_16gb`). At this
VRAM budget, gradient checkpointing is generally necessary, not a nice-to-have,
to fit generation + backward pass in 4 GB.

## Recommended Defaults

Starting points for the smoke / debug / longer run profiles on this hardware.
`num_generations=4` is worth attempting first despite the tight budget, since
group size matters a lot for GRPO's advantage quality — validate it with a
smoke run rather than pre-emptively dropping to 2. The rest of these are
deliberately conservative; validate any increase with a smoke run before
scaling up, and change one setting at a time so an OOM is attributable.

| Setting | Default | Notes |
|---|---|---|
| `num_generations` | 4, attempted first | Validate with a smoke run before trusting it. **Fall back to 2 only if 4 causes OOM or clear memory pressure** — 2 is the practical floor for GRPO, not the default starting point, even on this constrained profile. If you do fall back to 2, document it (see `docs/PROJECT_SPEC.md`) since it's a real methodological difference, not just a config tweak |
| `max_prompt_length` | 128 | Shorter cap than `mps_16gb`; controls memory during the prompt-processing pass |
| `max_completion_length` | 64–128 | Shorter than `mps_16gb`'s default; completion-length is a direct driver of generation-time activation memory |
| `batch_size` | 1 | Start here; this is not a placeholder to "fix" later, it may simply be right-sized for the VRAM budget |
| `gradient_accumulation_steps` | 8–16 | Compensates for `batch_size=1` to reach a reasonable effective batch size |
| `gradient_checkpointing` | on (required default) | See above |
| `beta` | 0.04 (default) | See above — no longer 0 by default now that LoRA makes the reference-computation memory cost negligible |

## Out-of-Memory Handling

- Catch CUDA OOM errors explicitly and fail with a clear, actionable message
  that includes the current CUDA memory stats at the point of failure.
- Do **not** silently auto-reduce batch size, generations, or sequence length
  and retry — that breaks reproducibility and hides what actually happened.
  Surface the OOM, log the memory state, and stop.
- If OOM recurs after adjusting one setting, adjust only one further setting
  at a time, in this order of impact: `max_completion_length` reduction →
  `num_generations` **4 → 2** (document this fallback explicitly if used —
  see `docs/PROJECT_SPEC.md`) → confirm `gradient_checkpointing` is actually
  on → confirm `beta=0` → precision.

## Memory Reporting

- Process-level memory: always record (e.g. via `psutil`), same as the other profile.
- CUDA-specific stats (available and precise on this profile):
  - `torch.cuda.memory_allocated()`
  - `torch.cuda.max_memory_allocated()`
  - `torch.cuda.memory_reserved()`
- Log both allocated and reserved figures — reserved-but-unallocated memory
  (fragmentation) is a real failure mode at 4 GB and worth surfacing
  separately from live allocation.
- Never call `torch.mps.*` anywhere in code that might run on this profile.

## Known Considerations

- 4 GB is tight even for a 135M-parameter model once generation activations,
  optimizer state, and (if enabled) a reference model are accounted for.
  Increasing *any one* of `num_generations`, `max_completion_length`, or
  `beta` without compensating elsewhere carries real OOM risk — this is a
  profile characteristic, not something to engineer away.
- Smaller batch size with higher gradient accumulation means more optimizer
  steps for the same effective batch size, which can make wall-clock time
  per "logical step" longer than the `mps_16gb` profile even if raw GPU
  compute is faster. Don't assume the suggested watchdog timeout minutes from
  `docs/PROJECT_SPEC.md` transfer directly — adjust them based on observed
  step throughput on this machine.
- Because VRAM is dedicated (not shared with the OS the way Apple unified
  memory is), a clean CUDA memory read is more reliable/precise here than the
  MPS equivalent — take advantage of that for tighter OOM-avoidance logic
  rather than treating it as equally fuzzy to the Mac profile's numbers.
