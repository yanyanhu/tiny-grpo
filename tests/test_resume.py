"""CPU-only unit tests for tiny_grpo.resume. No model access — real tmp_path
directories with synthetic checkpoint-N subdirs and run_tags.json files, same
style as tests/test_cleanup.py.
"""

import os
import time

import pytest

from tiny_grpo.cleanup import list_run_entries
from tiny_grpo.resume import (
    CrossProfileResumeError,
    ResumeTargetNotFoundError,
    find_latest_resumable_run,
    resolve_resume_target,
)
from tiny_grpo.run_context import RunTags, save_run_tags


def _make_run_dir(root, name, mtime_offset_seconds, run_profile, hardware_profile="mps_16gb",
                   status="completed", checkpoints=(), tagged=True):
    run_dir = root / name
    run_dir.mkdir()
    for step in checkpoints:
        (run_dir / f"checkpoint-{step}").mkdir()

    if tagged:
        save_run_tags(
            run_dir,
            RunTags(run_profile=run_profile, hardware_profile=hardware_profile, verification_run=True, status=status),
        )

    now = time.time()
    os.utime(run_dir, (now + mtime_offset_seconds, now + mtime_offset_seconds))
    return run_dir


class TestFindLatestResumableRun:
    def test_matches_profile_and_hardware(self, tmp_path):
        smoke_mps = _make_run_dir(tmp_path, "a", -100, "smoke", "mps_16gb", status="failed", checkpoints=[5])
        _make_run_dir(tmp_path, "b", -50, "debug", "mps_16gb", status="failed", checkpoints=[5])
        _make_run_dir(tmp_path, "c", -50, "smoke", "cuda_4gb", status="failed", checkpoints=[5])

        entries = list_run_entries(tmp_path)
        result = find_latest_resumable_run(entries, "smoke", "mps_16gb")

        assert result.path == smoke_mps

    def test_excludes_completed(self, tmp_path):
        _make_run_dir(tmp_path, "a", -100, "smoke", "mps_16gb", status="completed", checkpoints=[5])
        entries = list_run_entries(tmp_path)
        assert find_latest_resumable_run(entries, "smoke", "mps_16gb") is None

    def test_includes_running_and_failed(self, tmp_path):
        _make_run_dir(tmp_path, "a", -100, "smoke", "mps_16gb", status="running")
        _make_run_dir(tmp_path, "b", -50, "smoke", "mps_16gb", status="failed")
        entries = list_run_entries(tmp_path)
        assert find_latest_resumable_run(entries, "smoke", "mps_16gb") is not None

    def test_excludes_untagged(self, tmp_path):
        _make_run_dir(tmp_path, "a", -100, "smoke", "mps_16gb", checkpoints=[5], tagged=False)
        entries = list_run_entries(tmp_path)
        assert find_latest_resumable_run(entries, "smoke", "mps_16gb") is None

    def test_picks_most_recent(self, tmp_path):
        _make_run_dir(tmp_path, "old", -200, "smoke", "mps_16gb", status="failed", checkpoints=[5])
        newest = _make_run_dir(tmp_path, "new", -100, "smoke", "mps_16gb", status="failed", checkpoints=[5])

        entries = list_run_entries(tmp_path)
        result = find_latest_resumable_run(entries, "smoke", "mps_16gb")

        assert result.path == newest

    def test_no_match_returns_none(self):
        assert find_latest_resumable_run([], "smoke", "mps_16gb") is None


class TestResolveResumeTarget:
    def test_none_mode_always_none(self, tmp_path):
        _make_run_dir(tmp_path, "a", -100, "smoke", "mps_16gb", status="failed", checkpoints=[5])
        assert resolve_resume_target("none", tmp_path, "smoke", "mps_16gb") is None

    def test_latest_mode_resolves_most_recent_checkpoint(self, tmp_path):
        run_dir = _make_run_dir(tmp_path, "a", -100, "smoke", "mps_16gb", status="failed", checkpoints=[5, 10])

        target = resolve_resume_target("latest", tmp_path, "smoke", "mps_16gb")

        assert target.run_dir == run_dir
        assert target.checkpoint_path == str(run_dir / "checkpoint-10")
        assert target.origin_hardware_profile == "mps_16gb"

    def test_latest_mode_no_match_returns_none(self, tmp_path):
        assert resolve_resume_target("latest", tmp_path, "smoke", "mps_16gb") is None

    def test_explicit_checkpoint_path(self, tmp_path):
        run_dir = _make_run_dir(tmp_path, "a", -100, "smoke", "mps_16gb", status="failed", checkpoints=[5, 10])

        target = resolve_resume_target(str(run_dir / "checkpoint-5"), tmp_path, "smoke", "mps_16gb")

        assert target.run_dir == run_dir
        assert target.checkpoint_path == str(run_dir / "checkpoint-5")

    def test_explicit_run_dir_path_resolves_its_latest_checkpoint(self, tmp_path):
        run_dir = _make_run_dir(tmp_path, "a", -100, "smoke", "mps_16gb", status="failed", checkpoints=[5, 10])

        target = resolve_resume_target(str(run_dir), tmp_path, "smoke", "mps_16gb")

        assert target.checkpoint_path == str(run_dir / "checkpoint-10")

    def test_explicit_path_without_checkpoint_raises(self, tmp_path):
        run_dir = _make_run_dir(tmp_path, "a", -100, "smoke", "mps_16gb", status="failed", checkpoints=[])
        with pytest.raises(ResumeTargetNotFoundError):
            resolve_resume_target(str(run_dir), tmp_path, "smoke", "mps_16gb")

    def test_cross_profile_mismatch_raises(self, tmp_path):
        run_dir = _make_run_dir(tmp_path, "a", -100, "smoke", "cuda_4gb", status="failed", checkpoints=[5])
        with pytest.raises(CrossProfileResumeError):
            resolve_resume_target(str(run_dir), tmp_path, "smoke", "mps_16gb")

    def test_cross_profile_mismatch_allowed_when_flagged(self, tmp_path):
        run_dir = _make_run_dir(tmp_path, "a", -100, "smoke", "cuda_4gb", status="failed", checkpoints=[5])

        target = resolve_resume_target(str(run_dir), tmp_path, "smoke", "mps_16gb", allow_cross_profile=True)

        assert target.origin_hardware_profile == "cuda_4gb"

    def test_latest_mode_never_needs_cross_profile_check(self, tmp_path):
        # "latest" only ever matches entries already filtered to the current
        # hardware_profile, so it should never raise regardless of what other
        # (non-matching) profiles' runs exist alongside it.
        _make_run_dir(tmp_path, "other", -200, "smoke", "cuda_4gb", status="failed", checkpoints=[5])
        run_dir = _make_run_dir(tmp_path, "mine", -100, "smoke", "mps_16gb", status="failed", checkpoints=[5])

        target = resolve_resume_target("latest", tmp_path, "smoke", "mps_16gb")

        assert target.run_dir == run_dir
