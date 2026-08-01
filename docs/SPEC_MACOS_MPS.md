# Hardware Profile: `mps_16gb` — MacBook Pro M2, 16 GB Unified Memory

> Concrete settings for this hardware profile. For everything else (reward
> math, dataset handling, evaluation, testing, checkpointing, milestone), see
> `docs/PROJECT_SPEC.md`. This doc only covers what's specific to running on
> this machine.

## Target Environment

- MacBook Pro with Apple M2
- 16 GB unified memory (shared between OS, other apps, and this workload —
  don't assume all 16 GB is available; leave headroom)
- macOS
- PyTorch **MPS** backend

## Device & Environment Setup

- Device: `mps`
- Set when unsupported MPS operations are hit:

  ```
  export PYTORCH_ENABLE_MPS_FALLBACK=1
  ```

- vLLM: disabled — **hard constraint**, not a preference. vLLM has no MPS
  backend, so there is no version of this profile where enabling it is an option.

## Precision

Order of preference, moving to the next only after verifying stability:

1. **fp32** — start here. Maximum MPS operator compatibility.
2. **bf16 autocast** — try this before fp16. Recent PyTorch/MPS builds
   generally support bf16 better than fp16.
3. **fp16** — last resort. MPS fp16 training has historically had operator
   gaps and stability issues; only use after specifically verifying it holds
   up on this machine, not by default.

## Reference Model / KL Coefficient (`beta`)

At this model scale (135M) and 16 GB unified memory, loading a separate
reference model for the KL penalty is generally affordable. Default to a
non-zero `beta` (standard TRL default is a reasonable starting point) with a
real reference model loaded.

Fall back to `beta=0` (no reference model) if:
- memory pressure appears during a smoke run (check process + MPS memory stats);
- you're increasing `num_generations` or batch size beyond the defaults below
  and need the headroom.

## Recommended Defaults

These are starting points for the smoke / debug / longer run profiles on this
hardware profile. Treat them as adjustable, but validate any increase with a
smoke run before a longer one.

| Setting | Default | Notes |
|---|---|---|
| `num_generations` | 4 | Preferred default at this memory budget (16 GB unified) — meaningfully better advantage-estimation quality than 2. Try 8 if a smoke run shows headroom; fall back to 2 only if 4 causes memory pressure, and note it as a deviation if so |
| `max_prompt_length` | 256 | GSM8K prompts are short; generous cap |
| `max_completion_length` | 256 | Room for reasoning + final answer tag |
| `batch_size` | 2–4 | Start at 2, increase only after a clean smoke run |
| `gradient_accumulation_steps` | 4–8 | Tune with batch size for a reasonable effective batch |
| `gradient_checkpointing` | off (default) | Optional toggle; likely unnecessary at this model scale, available if memory pressure appears |
| `beta` | non-zero (standard default) | Fall back to 0 under memory pressure — see above |

## Memory Reporting

- Process-level memory: always record (e.g. via `psutil`), as the universal fallback.
- MPS-specific stats, when available on the installed PyTorch version:
  - `torch.mps.current_allocated_memory()`
  - `torch.mps.driver_allocated_memory()`
- Guard these calls with `hasattr(torch.mps, ...)` checks — the MPS memory API
  surface has changed across PyTorch versions, and there's no CUDA-style
  `max_memory_allocated` guarantee on MPS.
- Never call `torch.cuda.*` anywhere in code that might run on this profile.

## Known Considerations

- Unified memory is shared with the OS and other running apps. Close
  memory-heavy applications before a longer run, and don't assume the full
  16 GB is available headroom.
- MPS has operator gaps relative to CUDA; `PYTORCH_ENABLE_MPS_FALLBACK=1`
  covers most of these by falling back to CPU for unsupported ops, at a
  performance cost — expect and accept this rather than trying to route
  around it.
- fp16 on MPS has a track record of subtle numerical/stability issues in
  training (not just inference). Don't reach for it as a default; bf16 first.
