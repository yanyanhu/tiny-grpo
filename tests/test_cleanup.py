"""CPU-only unit tests for tiny_grpo.cleanup. No model/dataset access.

Uses mocked/synthetic tagged run directories — never real runs or a model.
"""

import os
import time

import pytest

from tiny_grpo.cleanup import list_run_entries, prune, select_prunable
from tiny_grpo.run_context import RunTags, save_run_tags


def _make_run_dir(root, name, mtime_offset_seconds, run_profile=None, hardware_profile="mps_16gb",
                   verification_run=None, status="completed", tagged=True):
    run_dir = root / name
    run_dir.mkdir()
    (run_dir / "payload.bin").write_bytes(b"x" * 1024)

    if tagged:
        save_run_tags(
            run_dir,
            RunTags(
                run_profile=run_profile or name,
                hardware_profile=hardware_profile,
                verification_run=verification_run if verification_run is not None else True,
                status=status,
            ),
        )

    now = time.time()
    os.utime(run_dir, (now + mtime_offset_seconds, now + mtime_offset_seconds))
    return run_dir


def test_list_run_entries_on_missing_root_returns_empty(tmp_path):
    assert list_run_entries(tmp_path / "missing") == []


def test_list_run_entries_reports_untagged_directory(tmp_path):
    run_dir = _make_run_dir(tmp_path, "legacy_run", -100, tagged=False)
    entries = list_run_entries(tmp_path)
    assert len(entries) == 1
    assert entries[0].path == run_dir
    assert entries[0].tags is None
    assert entries[0].size_bytes >= 1024


def test_list_run_entries_sorted_oldest_first(tmp_path):
    newest = _make_run_dir(tmp_path, "smoke_c", -100, run_profile="smoke")
    oldest = _make_run_dir(tmp_path, "smoke_a", -300, run_profile="smoke")
    middle = _make_run_dir(tmp_path, "smoke_b", -200, run_profile="smoke")
    assert [e.path for e in list_run_entries(tmp_path)] == [oldest, middle, newest]


def test_untagged_directory_never_selected_for_pruning(tmp_path):
    _make_run_dir(tmp_path, "legacy_run", -10_000, tagged=False)
    assert select_prunable(tmp_path, keep=0) == []


def test_debug_and_longer_runs_never_auto_selected_even_when_oldest(tmp_path):
    _make_run_dir(tmp_path, "debug_old", -10_000, run_profile="debug", verification_run=False)
    _make_run_dir(tmp_path, "longer_old", -9_000, run_profile="longer", verification_run=False)
    selected = select_prunable(tmp_path, keep=0)
    assert selected == []


def test_smoke_runs_beyond_keep_count_are_selected(tmp_path):
    oldest = _make_run_dir(tmp_path, "smoke_a", -300, run_profile="smoke")
    middle = _make_run_dir(tmp_path, "smoke_b", -200, run_profile="smoke")
    newest = _make_run_dir(tmp_path, "smoke_c", -100, run_profile="smoke")

    selected = select_prunable(tmp_path, keep=1)

    assert [e.path for e in selected] == [oldest, middle]
    assert newest not in [e.path for e in selected]


def test_most_recent_failed_run_is_protected_even_beyond_keep_count(tmp_path):
    failed = _make_run_dir(tmp_path, "smoke_failed", -300, run_profile="smoke", status="failed")
    # Neither later run is a *completed* run of the same profile, so the
    # failure is never superseded (see the supersession test below for that case).
    _make_run_dir(tmp_path, "smoke_b", -200, run_profile="smoke", status="running")
    _make_run_dir(tmp_path, "smoke_c", -100, run_profile="smoke", status="running")

    selected = select_prunable(tmp_path, keep=0)

    assert failed not in [e.path for e in selected]


def test_failed_run_superseded_by_later_completed_run_becomes_prunable(tmp_path):
    failed = _make_run_dir(tmp_path, "smoke_failed", -300, run_profile="smoke", status="failed")
    _make_run_dir(tmp_path, "smoke_success", -200, run_profile="smoke", status="completed")

    selected = select_prunable(tmp_path, keep=0)

    assert failed in [e.path for e in selected]


def test_debug_run_failure_never_protected_since_never_eligible(tmp_path):
    # Protection only matters for already-eligible (verification) runs; a
    # failed debug run is untouched regardless, same as any debug run.
    debug_failed = _make_run_dir(tmp_path, "debug_failed", -10_000, run_profile="debug",
                                  verification_run=False, status="failed")
    assert select_prunable(tmp_path, keep=0) == []
    assert debug_failed.exists()


def test_older_than_days_selects_regardless_of_keep_count(tmp_path):
    old = _make_run_dir(tmp_path, "smoke_old", -10 * 86400, run_profile="smoke")
    recent = _make_run_dir(tmp_path, "smoke_recent", -10, run_profile="smoke")

    selected = select_prunable(tmp_path, keep=5, older_than_days=1)

    assert [e.path for e in selected] == [old]
    assert recent not in [e.path for e in selected]


def test_prune_dry_run_does_not_delete(tmp_path):
    oldest = _make_run_dir(tmp_path, "smoke_a", -300, run_profile="smoke")
    _make_run_dir(tmp_path, "smoke_b", -100, run_profile="smoke")

    removed = prune(tmp_path, keep=1, dry_run=True)

    assert [e.path for e in removed] == [oldest]
    assert oldest.exists()


def test_prune_actually_deletes(tmp_path):
    oldest = _make_run_dir(tmp_path, "smoke_a", -300, run_profile="smoke")
    newest = _make_run_dir(tmp_path, "smoke_b", -100, run_profile="smoke")

    removed = prune(tmp_path, keep=1)

    assert [e.path for e in removed] == [oldest]
    assert not oldest.exists()
    assert newest.exists()


def test_select_prunable_negative_keep_raises(tmp_path):
    with pytest.raises(ValueError):
        select_prunable(tmp_path, keep=-1)


def test_select_prunable_on_empty_root_is_noop(tmp_path):
    assert select_prunable(tmp_path, keep=3) == []
