"""Shared trainer logging callbacks for GRPO and SFT."""

import json
import sys
import time
from pathlib import Path

from transformers import TrainerCallback

from tiny_grpo.monitoring import device_memory_mb, process_memory_mb


class JsonlLoggerCallback(TrainerCallback):
    def __init__(self, path, device: str):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.device = device

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None:
            return
        record = {"step": state.global_step, **logs, "process_memory_mb": process_memory_mb()}
        device_mem = device_memory_mb(self.device)
        if device_mem is not None:
            record[f"{self.device}_memory_mb"] = device_mem
        with self.path.open("a") as handle:
            handle.write(json.dumps(record) + "\n")


class ConsoleProgressCallback(TrainerCallback):
    def __init__(self, device: str):
        self._start_time = None
        self._is_tty = sys.stdout.isatty()
        self.device = device

    def on_train_begin(self, args, state, control, **kwargs):
        self._start_time = time.monotonic()

    def on_log(self, args, state, control, logs=None, **kwargs):
        if (
            logs is None
            or self._start_time is None
            or any(key.startswith("eval_") for key in logs)
            or ("loss" not in logs and "reward" not in logs)
        ):
            return
        elapsed = time.monotonic() - self._start_time
        step = state.global_step
        max_steps = state.max_steps or 0
        eta = (elapsed / step) * (max_steps - step) if 0 < step < max_steps else 0.0
        parts = [
            f"step {step}/{max_steps}",
            f"elapsed {elapsed:.0f}s",
            f"eta {eta:.0f}s",
            f"loss {logs.get('loss', float('nan')):.4f}",
        ]
        if "reward" in logs:
            parts.append(
                "reward "
                f"{logs['reward']:.3f} (acc {logs.get('rewards/accuracy_reward/mean', float('nan')):.3f} "
                f"fmt {logs.get('rewards/format_reward/mean', float('nan')):.3f})"
            )
        device_mem = device_memory_mb(self.device)
        memory = f"mem {process_memory_mb():.0f}MB"
        if device_mem:
            memory += f" {self.device} {device_mem['allocated_mb']:.0f}MB"
        parts.append(memory)
        line = " | ".join(parts)
        if self._is_tty:
            sys.stdout.write("\r" + line.ljust(120))
            sys.stdout.flush()
        else:
            print(line, flush=True)
