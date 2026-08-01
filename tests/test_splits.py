"""CPU-only unit tests for tiny_grpo.splits. No dataset download."""

import json

import pytest

from tiny_grpo.splits import (
    SplitMetadata,
    SplitOverlapError,
    SplitSizeError,
    assert_disjoint,
    build_split_metadata,
    load_split_metadata,
    save_split_metadata,
)


def test_determinism_same_seed_same_split():
    a = build_split_metadata(train_pool_size=100, test_pool_size=50, train_size=20, val_size=10, test_size=8, seed=42)
    b = build_split_metadata(train_pool_size=100, test_pool_size=50, train_size=20, val_size=10, test_size=8, seed=42)
    assert a.train_indices == b.train_indices
    assert a.val_indices == b.val_indices
    assert a.test_indices == b.test_indices


def test_different_seed_gives_different_split():
    a = build_split_metadata(train_pool_size=100, test_pool_size=50, train_size=20, val_size=10, test_size=8, seed=1)
    b = build_split_metadata(train_pool_size=100, test_pool_size=50, train_size=20, val_size=10, test_size=8, seed=2)
    assert a.train_indices != b.train_indices


def test_train_and_val_are_disjoint():
    meta = build_split_metadata(train_pool_size=100, test_pool_size=50, train_size=20, val_size=10, test_size=8, seed=42)
    assert_disjoint(meta.train_indices, meta.val_indices)  # must not raise


def test_overlap_detection_raises():
    with pytest.raises(SplitOverlapError):
        assert_disjoint([1, 2, 3], [3, 4, 5])


def test_sizes_match_request():
    meta = build_split_metadata(train_pool_size=100, test_pool_size=50, train_size=20, val_size=10, test_size=8, seed=42)
    assert meta.train_size == 20
    assert meta.val_size == 10
    assert meta.test_size == 8


def test_indices_within_pool_bounds():
    meta = build_split_metadata(train_pool_size=100, test_pool_size=50, train_size=20, val_size=10, test_size=8, seed=42)
    assert all(0 <= i < 100 for i in meta.train_indices + meta.val_indices)
    assert all(0 <= i < 50 for i in meta.test_indices)


def test_oversized_request_raises():
    with pytest.raises(SplitSizeError):
        build_split_metadata(train_pool_size=10, test_pool_size=50, train_size=8, val_size=8, test_size=1, seed=42)


def test_persistence_roundtrip(tmp_path):
    meta = build_split_metadata(train_pool_size=100, test_pool_size=50, train_size=20, val_size=10, test_size=8, seed=42)
    path = tmp_path / "split_metadata.json"
    save_split_metadata(path, meta)
    loaded = load_split_metadata(path)
    assert loaded == meta


def test_persisted_file_is_readable_json(tmp_path):
    meta = build_split_metadata(train_pool_size=100, test_pool_size=50, train_size=20, val_size=10, test_size=8, seed=42)
    path = tmp_path / "split_metadata.json"
    save_split_metadata(path, meta)
    raw = json.loads(path.read_text())
    assert raw["seed"] == 42
    assert len(raw["train_indices"]) == 20
    assert len(raw["val_indices"]) == 10
    assert len(raw["test_indices"]) == 8


def test_metadata_equality_and_from_dict():
    meta = build_split_metadata(train_pool_size=100, test_pool_size=50, train_size=20, val_size=10, test_size=8, seed=42)
    rebuilt = SplitMetadata.from_dict(meta.to_dict())
    assert rebuilt == meta
