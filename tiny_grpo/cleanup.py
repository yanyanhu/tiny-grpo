"""Prune smoke/verification run directories, keyed off tagged metadata
(run_tags.json) — never directory naming or mtime alone.

- Only runs explicitly tagged `verification_run=True` (smoke runs, by default —
  see train_grpo.py) are ever eligible for automatic deletion. `debug`/`longer`
  runs, and any untagged/foreign directory under output_root, are never
  touched.
- The most recent *failed* run of each run profile is protected until a later
  *completed* run of the same profile supersedes it, or it's cleared manually.
- Never silent: both `list` and `prune` print what they find/remove.

Not automatic — nothing in train_grpo.py calls this. Invoke explicitly:

    uv run python -m tiny_grpo.cleanup list
    uv run python -m tiny_grpo.cleanup prune --keep 3
    uv run python -m tiny_grpo.cleanup prune --keep 3 --dry-run
    uv run python -m tiny_grpo.cleanup prune --older-than-days 7
"""

import argparse
import dataclasses
import shutil
import time
from pathlib import Path

from tiny_grpo.run_context import RunTags, has_run_tags, load_run_tags


@dataclasses.dataclass(frozen=True)
class RunEntry:
    path: Path
    tags: RunTags | None  # None for untagged/foreign directories — never auto-eligible
    mtime: float
    size_bytes: int


def _dir_size_bytes(path: Path) -> int:
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())


def list_run_entries(output_root: str | Path) -> list[RunEntry]:
    """Immediate subdirectories of output_root, oldest first (by mtime), each
    with its tags (if tagged) and total disk size."""
    root = Path(output_root)
    if not root.exists():
        return []
    entries = []
    for path in root.iterdir():
        if not path.is_dir():
            continue
        tags = load_run_tags(path) if has_run_tags(path) else None
        entries.append(RunEntry(path=path, tags=tags, mtime=path.stat().st_mtime, size_bytes=_dir_size_bytes(path)))
    return sorted(entries, key=lambda e: e.mtime)


def _is_deletion_eligible(entry: RunEntry) -> bool:
    return entry.tags is not None and entry.tags.verification_run


def _protected_failed_paths(entries: list[RunEntry]) -> set:
    """The most recent failed run of each run_profile, kept until a later
    completed run of the same profile supersedes it. `entries` must be
    oldest-first so later iterations correctly overwrite earlier ones."""
    protected = {}
    for entry in entries:
        profile = entry.tags.run_profile
        if entry.tags.status == "failed":
            protected[profile] = entry.path
        elif entry.tags.status == "completed":
            protected.pop(profile, None)
    return set(protected.values())


def select_prunable(
    output_root: str | Path,
    keep: int = 3,
    older_than_days: float | None = None,
) -> list[RunEntry]:
    """Select verification-run directories eligible for deletion: beyond the
    `keep` most recent verification runs, or older than `older_than_days` (if
    given) — either condition selects for deletion. `debug`/`longer` runs and
    untagged directories are never selected. The most recent failed run of
    each run profile is protected regardless of the above.
    """
    if keep < 0:
        raise ValueError(f"keep must be >= 0, got {keep}")

    entries = list_run_entries(output_root)
    eligible = [e for e in entries if _is_deletion_eligible(e)]
    protected = _protected_failed_paths(eligible)
    candidates = [e for e in eligible if e.path not in protected]

    by_count = candidates[: max(0, len(candidates) - keep)]

    selected = {e.path: e for e in by_count}
    if older_than_days is not None:
        cutoff = time.time() - older_than_days * 86400
        for e in candidates:
            if e.mtime < cutoff:
                selected[e.path] = e

    return sorted(selected.values(), key=lambda e: e.mtime)


def prune(
    output_root: str | Path,
    keep: int = 3,
    older_than_days: float | None = None,
    dry_run: bool = False,
) -> list[RunEntry]:
    to_remove = select_prunable(output_root, keep=keep, older_than_days=older_than_days)
    if not dry_run:
        for entry in to_remove:
            shutil.rmtree(entry.path)
    return to_remove


def _format_size(size_bytes: int) -> str:
    size = float(size_bytes)
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024:
            return f"{size:.1f}{unit}"
        size /= 1024
    return f"{size:.1f}TB"


def _tag_summary(tags: RunTags | None) -> str:
    if tags is None:
        return "untagged (never auto-deletion-eligible)"
    return (
        f"profile={tags.run_profile} hardware={tags.hardware_profile} "
        f"verification={tags.verification_run} status={tags.status}"
    )


def _print_listing(entries: list[RunEntry]) -> None:
    if not entries:
        print("No run directories found.")
        return
    for entry in entries:
        age_days = (time.time() - entry.mtime) / 86400
        print(f"{entry.path}  age={age_days:.1f}d  size={_format_size(entry.size_bytes)}  {_tag_summary(entry.tags)}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["list", "prune"])
    parser.add_argument("--output-root", default="outputs")
    parser.add_argument("--keep", type=int, default=3, help="Verification runs to keep (default: 3)")
    parser.add_argument(
        "--older-than-days", type=float, default=None, help="Also prune verification runs older than N days"
    )
    parser.add_argument("--dry-run", action="store_true", help="Preview prune without deleting anything")
    args = parser.parse_args()

    if args.command == "list":
        _print_listing(list_run_entries(args.output_root))
        return

    removed = prune(args.output_root, keep=args.keep, older_than_days=args.older_than_days, dry_run=args.dry_run)
    if not removed:
        print(f"Nothing to prune under {args.output_root!r} (keep={args.keep}, older_than_days={args.older_than_days}).")
        return

    verb = "Would remove" if args.dry_run else "Removed"
    total_bytes = 0
    for entry in removed:
        print(f"{verb}: {entry.path}  ({_format_size(entry.size_bytes)}, {_tag_summary(entry.tags)})")
        total_bytes += entry.size_bytes
    print(f"{verb} {len(removed)} run director{'y' if len(removed) == 1 else 'ies'}, freeing {_format_size(total_bytes)}.")


if __name__ == "__main__":
    main()
