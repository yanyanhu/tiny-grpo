# First run on the RTX 3050 / WSL workstation — checklist

> This is a practical, sequential checklist for the *first* time running this
> project on the `cuda_4gb` profile for real. `docs/SPEC_CUDA_4GB.md` covers
> the design/settings ("what value is safe"); this doc covers "what to
> actually do, in what order." It was created when the `cuda_4gb` path had
> only been unit/mock-tested; the completed first-run record is appended below.

## 0. Environment (WSL-specific)

* [x] `nvidia-smi` — run **inside WSL**, not just on the Windows side. WSL2's
  CUDA support works through a driver-interop layer: the NVIDIA driver
  installs on the Windows host, not inside WSL, and WSL2 talks to it via a
  `libcuda.so` shim. If this doesn't work, it's a Windows-driver/WSL-CUDA
  support problem one layer below Python — fix this before touching
  `uv`/torch at all.
* [x] Clone into the **native WSL filesystem**, not `/mnt/c/...`:
  `sh
      cd ~
      git clone git@github.com:yanyanhu/tiny-grpo.git
      `
  WSL2 has notably slow I/O crossing the Windows/Linux filesystem
  boundary — matters for dataset loading and checkpoint writes.

## 1. Sanity-check torch/CUDA before anything else

```sh
cd tiny-grpo
uv sync
uv run python -c "import torch; print(torch.cuda.is_available(), torch.cuda.is_bf16_supported(including_emulation=False))"
```

Both should print `True`. If either is `False`, stop here — the CUDA
toolkit/driver/torch-wheel combination doesn't match, and no amount of
retrying training will fix it. This is a 5-second check versus discovering it
mid-run.

## 2. Run the full test suite before any real training

```sh
uv run pytest -q tests/
```

For the first time in this project, `tests/test_cuda_integration.py` will
actually **run** instead of skip — it loads the real model, wraps it with
LoRA, and generates at all three precisions (fp32/bf16/fp16) on real CUDA.
Cheap, and it validates the model+LoRA+precision path before committing to a
training run. `tests/test_mps_integration.py` will now skip (no MPS here) —
expected, not a problem.

## 3. Re-read `docs/SPEC_CUDA_4GB.md` on this machine

Matches `CLAUDE.md`'s own stated practice: read the profile doc matching the
machine you're on before touching config defaults. Refresh the recommended
defaults and the OOM fallback order (below) while it's fresh.

## 4. First real run — expect it might OOM

At the start of this checklist, this exact configuration
(`per_device_train_batch_size=4`, `num_generations=4`,
`gradient_checkpointing=True`, `beta=0.04`, bf16, LoRA) had **never been run
against real 4GB VRAM** — only unit/mock-tested.

```sh
STALL_LIMIT=600 HARD_TIMEOUT=3600 ./run_with_watchdog.sh --profile smoke --hardware cuda_4gb
```

Generous timeout because step throughput on this machine is unmeasured —
`SPEC_CUDA_4GB.md` itself warns not to assume the MPS profile's timing
transfers.

**If it OOMs**, the documented fallback order (one change at a time, so
whatever fixes it is attributable):

1. reduce `max_completion_length` further
2. drop `num_generations` 4 → 2 (document this — it's a real methodological
   difference, not a cosmetic tweak, per `docs/PROJECT_SPEC.md`)
3. confirm `gradient_checkpointing` is actually on
4. confirm/try `beta=0` (no reference computation at all — note this is no
   longer the *default* here since the LoRA-driven fix, but it's still a
   valid fallback)
5. precision (last resort)

Use `--set field=value` to try an override quickly without touching code,
e.g. `--set max_completion_length=64`.

## 5. If smoke succeeds: redo the resume verification on `cuda_4gb` too

The full save → interrupt → resume → verify sequence (see
`docs/PROJECT_SPEC.md`'s Checkpoint and Resume Requirements) was only ever
verified end-to-end on `mps_16gb`. Repeating it here:

1. Start a smoke run, let it save a checkpoint, kill it (`SIGTERM`, then
   `SIGKILL` after a grace period — see `run_with_watchdog.sh`'s pattern).
2. Confirm the checkpoint exists and `run_tags.json` status reflects it
   (`"running"` if killed externally, `"failed"` if it errored on its own).
3. Resume: `./run_with_watchdog.sh --profile smoke --hardware cuda_4gb --resume latest`.
4. Confirm it reused the *same* run directory, training continued from the
   correct global step (not step 0), and it completed.
5. Load `final_adapter/` fresh (`AutoModelForCausalLM` + `PeftModel.from_pretrained`)
   and confirm it actually generates — not just that the file exists.

Closing this out on `cuda_4gb` completes the project's cross-platform
verification goal (`CLAUDE.md`: "both `mps_16gb` and `cuda_4gb` must be
exercised by the test/verification plan over time").

## Verification record — 2026-08-02

All first-run checklist stages completed on the target RTX 3050 Laptop GPU
under WSL. Every verification command used a finite timeout or the project
watchdog.

### Environment and tests

- `nvidia-smi` detected the RTX 3050 with 4096 MiB VRAM and no competing GPU
  process at the start of verification.
- PyTorch `2.13.0+cu130` reported CUDA available and native bf16 support:
  `torch.cuda.is_available() == True` and
  `torch.cuda.is_bf16_supported(including_emulation=False) == True`.
- The full suite passed on real CUDA: **159 passed, 4 skipped**. The four
  skips were the expected MPS-only integration cases; the CUDA model + LoRA
  generation cases ran at fp32, bf16, and fp16.

### First smoke run

The preferred configuration completed without OOM and required no fallback:

- bf16, `num_generations=4`, `per_device_train_batch_size=4`;
- `gradient_accumulation_steps=8`, gradient checkpointing enabled;
- `beta=0.04`, `max_completion_length=128`;
- 10 training steps, checkpoints at steps 5 and 10.

Run directory: `outputs/smoke_20260802_095947`.

| Metric | Baseline | Post-training |
|---|---:|---:|
| Exact-answer accuracy | 0.000 | 0.000 |
| Valid-format rate | 0.188 | 0.437 |
| Parse-failure rate | 0.812 | 0.562 |
| Mean reward | 0.038 | 0.087 |
| Mean completion length | 111.8 | 98.7 |
| Evaluation runtime | 60.3 s | 87.3 s |

Peak recorded training memory was 2084 MiB process RSS, 738.7 MiB CUDA
allocated, and 1228 MiB CUDA reserved. The GPU-wide `nvidia-smi` reading was
approximately 1337 MiB during the run. Training itself completed in 346
seconds.

The zero exact-answer accuracy is consistent with the documented tiny-model
expectation for a 10-step smoke run; format compliance improved materially.

### Checkpoint and resume

The successful interrupt/resume verification used
`outputs/smoke_20260802_101323`:

1. A timeout-protected run was interrupted after `checkpoint-5` was saved.
2. `run_tags.json` remained `"running"`, as expected for an external signal,
   and `checkpoint-5/trainer_state.json` recorded `global_step: 5`.
3. `--resume latest` reused the same run directory and checkpoint.
4. The first resumed progress record was step 6/10, proving it did not restart
   from step 0.
5. The resumed run completed with checkpoints 5 and 10 (the retention cap of
   two), saved `final_adapter/`, and completed post-training evaluation.
6. A fresh `AutoModelForCausalLM` + `PeftModel.from_pretrained()` load of the
   final adapter generated successfully on CUDA. That check recorded 268.2
   MiB allocated and 268.7 MiB peak CUDA memory.

During this verification, `run_with_watchdog.sh` was corrected to send
SIGTERM first, allow a configurable grace period before SIGKILL, and propagate
non-zero child/watchdog exit status. Shell syntax validation and the full test
suite passed after the change.
