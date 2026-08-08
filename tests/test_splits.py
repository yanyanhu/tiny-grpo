"""CPU-only unit tests for tiny_grpo.splits. No dataset download."""

import json
from pathlib import Path

import pytest

from tiny_grpo.splits import (
    DEFAULT_RESERVED_TRAINING_SIZE,
    DiagnosticManifest,
    SplitMetadata,
    SplitOverlapError,
    SplitSizeError,
    assert_disjoint,
    build_diagnostic_manifest,
    build_split_metadata,
    load_diagnostic_manifest,
    load_split_metadata,
    save_diagnostic_manifest,
    save_split_metadata,
)

SMALL_RESERVED_TRAINING_SIZE = 40


def _small_split(**overrides):
    fields = dict(
        train_pool_size=100,
        test_pool_size=50,
        train_size=20,
        val_size=10,
        test_size=8,
        seed=42,
        reserved_training_size=SMALL_RESERVED_TRAINING_SIZE,
    )
    fields.update(overrides)
    return build_split_metadata(**fields)


def test_determinism_same_seed_same_split():
    a = _small_split()
    b = _small_split()
    assert a.train_indices == b.train_indices
    assert a.val_indices == b.val_indices
    assert a.test_indices == b.test_indices


def test_different_seed_gives_different_split():
    a = _small_split(seed=1)
    b = _small_split(seed=2)
    assert a.train_indices != b.train_indices


def test_train_and_val_are_disjoint():
    meta = _small_split()
    assert_disjoint(meta.train_indices, meta.val_indices)  # must not raise


def test_validation_is_invariant_when_training_size_changes():
    smaller = _small_split(train_size=20)
    larger = _small_split(train_size=35)

    assert smaller.val_indices == larger.val_indices
    assert set(smaller.train_indices) < set(larger.train_indices)
    assert_disjoint(larger.train_indices, larger.val_indices)


def test_training_size_cannot_exceed_reserved_region():
    with pytest.raises(SplitSizeError, match="exceeds reserved_training_size"):
        _small_split(train_size=41)


def test_overlap_detection_raises():
    with pytest.raises(SplitOverlapError):
        assert_disjoint([1, 2, 3], [3, 4, 5])


def test_sizes_match_request():
    meta = _small_split()
    assert meta.train_size == 20
    assert meta.val_size == 10
    assert meta.test_size == 8


def test_indices_within_pool_bounds():
    meta = _small_split()
    assert all(0 <= i < 100 for i in meta.train_indices + meta.val_indices)
    assert all(0 <= i < 50 for i in meta.test_indices)


def test_oversized_request_raises():
    with pytest.raises(SplitSizeError):
        build_split_metadata(
            train_pool_size=10,
            test_pool_size=50,
            train_size=8,
            val_size=8,
            test_size=1,
            seed=42,
            reserved_training_size=8,
        )


def test_persistence_roundtrip(tmp_path):
    meta = _small_split()
    path = tmp_path / "split_metadata.json"
    save_split_metadata(path, meta)
    loaded = load_split_metadata(path)
    assert loaded == meta


def test_persisted_file_is_readable_json(tmp_path):
    meta = _small_split()
    path = tmp_path / "split_metadata.json"
    save_split_metadata(path, meta)
    raw = json.loads(path.read_text())
    assert raw["seed"] == 42
    assert len(raw["train_indices"]) == 20
    assert len(raw["val_indices"]) == 10
    assert len(raw["test_indices"]) == 8


def test_metadata_equality_and_from_dict():
    meta = _small_split()
    rebuilt = SplitMetadata.from_dict(meta.to_dict())
    assert rebuilt == meta


class TestDiagnosticManifest:
    def test_is_deterministic_and_disjoint_from_reserved_training_prefix(self):
        first = build_diagnostic_manifest(2000, reserved_training_size=1024, diagnostic_size=200, seed=42)
        second = build_diagnostic_manifest(2000, reserved_training_size=1024, diagnostic_size=200, seed=42)
        training_meta = build_split_metadata(
            train_pool_size=2000,
            test_pool_size=100,
            train_size=1024,
            val_size=200,
            test_size=10,
            seed=42,
            reserved_training_size=1024,
        )

        assert first == second
        assert set(first.diagnostic_indices) == set(training_meta.val_indices)
        assert len(first.diagnostic_indices) == 200
        assert_disjoint(training_meta.train_indices, first.diagnostic_indices)

    def test_smaller_profiles_use_stable_prefixes_of_canonical_validation(self):
        manifest = build_diagnostic_manifest(2000, reserved_training_size=1024, diagnostic_size=200, seed=42)
        smoke = build_split_metadata(
            2000, 100, train_size=64, val_size=16, test_size=10, seed=42,
            reserved_training_size=1024,
        )
        debug = build_split_metadata(
            2000, 100, train_size=256, val_size=32, test_size=10, seed=42,
            reserved_training_size=1024,
        )

        assert smoke.val_indices == sorted(manifest.diagnostic_indices[:16])
        assert debug.val_indices == sorted(manifest.diagnostic_indices[:32])
        assert smoke.val_indices == build_split_metadata(
            2000, 100, train_size=512, val_size=16, test_size=10, seed=42,
            reserved_training_size=1024,
        ).val_indices

    def test_default_reservation_matches_diagnostic_manifest_default(self):
        assert DEFAULT_RESERVED_TRAINING_SIZE == build_diagnostic_manifest(2000).reserved_training_size

    def test_persistence_roundtrip(self, tmp_path):
        manifest = build_diagnostic_manifest(2000, reserved_training_size=1024, diagnostic_size=200, seed=42)
        path = tmp_path / "diagnostic_manifest.json"
        save_diagnostic_manifest(path, manifest)
        assert load_diagnostic_manifest(path) == manifest

    def test_from_dict_copies_indices(self):
        raw = {
            "version": 1,
            "source_dataset": "openai/gsm8k",
            "source_config": "main",
            "source_split": "train",
            "seed": 42,
            "reserved_training_size": 1024,
            "diagnostic_indices": [1, 2, 3],
        }
        manifest = DiagnosticManifest.from_dict(raw)
        raw["diagnostic_indices"].append(4)
        assert manifest.diagnostic_indices == [1, 2, 3]

    def test_versioned_repository_manifest_matches_builder(self):
        path = Path(__file__).parents[1] / "data" / "diagnostic_manifest_v1.json"
        persisted = load_diagnostic_manifest(path)
        expected = build_diagnostic_manifest(7473)
        assert persisted == expected
