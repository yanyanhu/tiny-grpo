"""Resolve what (if anything) a run should resume from.

Resume is opt-in only — "none" (the default) never touches this module's
resolution logic at all; train_grpo.py always creates a fresh run directory
in that case. "latest" and explicit-path modes are the deliberate exception to
"every run gets a fresh directory": when a resume target resolves, the caller
reuses *that* target's run directory instead of creating a new one, so
training continues writing into the same place it was interrupted in.
"""

import dataclasses
import re
from pathlib import Path

from transformers.trainer_utils import get_last_checkpoint

from tiny_grpo.cleanup import RunEntry, list_run_entries
from tiny_grpo.run_context import load_run_tags

_CHECKPOINT_DIR_RE = re.compile(r"^checkpoint-\d+$")


class CrossProfileResumeError(RuntimeError):
    """Raised when a resume target was saved under a different hardware
    profile than the current invocation, unless explicitly allowed."""


class ResumeTargetNotFoundError(ValueError):
    """Raised when an explicit resume path has no checkpoint to resume from."""


@dataclasses.dataclass(frozen=True)
class ResumeTarget:
    run_dir: Path
    checkpoint_path: str
    origin_hardware_profile: str


def find_latest_resumable_run(entries: list[RunEntry], run_profile: str, hardware_profile: str) -> RunEntry | None:
    """Pure selection over already-listed run entries (oldest first, matching
    tiny_grpo.cleanup.list_run_entries's contract). A run is resumable if it's
    tagged, matches run_profile + hardware_profile, and hasn't already
    completed successfully (nothing left to continue in that case).
    """
    candidates = [
        e
        for e in entries
        if e.tags is not None
        and e.tags.run_profile == run_profile
        and e.tags.hardware_profile == hardware_profile
        and e.tags.status in ("running", "failed")
    ]
    return candidates[-1] if candidates else None


def _resolve_explicit_path(path_str: str) -> tuple[Path, str]:
    path = Path(path_str)
    if _CHECKPOINT_DIR_RE.match(path.name):
        return path.parent, str(path)

    checkpoint = get_last_checkpoint(str(path))
    if checkpoint is None:
        raise ResumeTargetNotFoundError(f"no checkpoint found under {path_str!r} to resume from")
    return path, checkpoint


def resolve_resume_target(
    mode: str,
    output_root: str | Path,
    run_profile: str,
    hardware_profile: str,
    *,
    allow_cross_profile: bool = False,
) -> ResumeTarget | None:
    """Resolve `mode` ("none" | "latest" | an explicit checkpoint/run-dir path)
    into a ResumeTarget, or None if there's nothing to resume (only possible
    for "none", or "latest" when no resumable run exists — explicit paths
    always resolve or raise).
    """
    if mode == "none":
        return None

    if mode == "latest":
        entries = list_run_entries(output_root)
        candidate = find_latest_resumable_run(entries, run_profile, hardware_profile)
        if candidate is None:
            return None
        checkpoint = get_last_checkpoint(str(candidate.path))
        if checkpoint is None:
            return None
        run_dir, checkpoint_path = candidate.path, checkpoint
    else:
        run_dir, checkpoint_path = _resolve_explicit_path(mode)

    origin_hardware_profile = load_run_tags(run_dir).hardware_profile
    if origin_hardware_profile != hardware_profile and not allow_cross_profile:
        raise CrossProfileResumeError(
            f"resume target {checkpoint_path!r} was run under hardware profile "
            f"{origin_hardware_profile!r}, but this invocation is {hardware_profile!r}. "
            "Resuming across hardware profiles is not guaranteed to work — pass "
            "allow_cross_profile=True (--allow-cross-profile-resume) if this is deliberate."
        )

    return ResumeTarget(run_dir=run_dir, checkpoint_path=checkpoint_path, origin_hardware_profile=origin_hardware_profile)
