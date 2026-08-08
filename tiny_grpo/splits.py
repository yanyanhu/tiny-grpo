"""Deterministic train/validation/test split selection for GSM8K.

Operates on plain index lists, not `datasets.Dataset` objects, so split
determinism/overlap/persistence can be unit tested without downloading
anything. `select_split` is the thin, untested-by-unit-test glue that applies
the resulting indices to an actually-loaded HF dataset.
"""

import dataclasses
import json
import random
from pathlib import Path

DEFAULT_RESERVED_TRAINING_SIZE = 1024


class SplitOverlapError(ValueError):
    """Raised when train and validation indices are not disjoint."""


class SplitSizeError(ValueError):
    """Raised when the requested split sizes don't fit the available pool."""


@dataclasses.dataclass(frozen=True)
class DiagnosticManifest:
    """Versioned, profile-independent prompt IDs for rollout comparisons."""

    version: int
    source_dataset: str
    source_config: str
    source_split: str
    seed: int
    reserved_training_size: int
    diagnostic_indices: list[int]

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "DiagnosticManifest":
        return cls(
            version=data["version"],
            source_dataset=data["source_dataset"],
            source_config=data["source_config"],
            source_split=data["source_split"],
            seed=data["seed"],
            reserved_training_size=data["reserved_training_size"],
            diagnostic_indices=list(data["diagnostic_indices"]),
        )


@dataclasses.dataclass(frozen=True)
class SplitMetadata:
    seed: int
    train_indices: list[int]
    val_indices: list[int]
    test_indices: list[int]

    @property
    def train_size(self) -> int:
        return len(self.train_indices)

    @property
    def val_size(self) -> int:
        return len(self.val_indices)

    @property
    def test_size(self) -> int:
        return len(self.test_indices)

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "SplitMetadata":
        return cls(
            seed=data["seed"],
            train_indices=list(data["train_indices"]),
            val_indices=list(data["val_indices"]),
            test_indices=list(data["test_indices"]),
        )


def _sample_disjoint_subsets(pool_size: int, sizes: list[int], seed: int) -> list[list[int]]:
    if sum(sizes) > pool_size:
        raise SplitSizeError(
            f"requested sizes {sizes} (total {sum(sizes)}) exceed pool size {pool_size}"
        )
    rng = random.Random(seed)
    shuffled = list(range(pool_size))
    rng.shuffle(shuffled)
    subsets = []
    cursor = 0
    for size in sizes:
        subsets.append(sorted(shuffled[cursor : cursor + size]))
        cursor += size
    return subsets


def assert_disjoint(a: list[int], b: list[int]) -> None:
    overlap = set(a) & set(b)
    if overlap:
        raise SplitOverlapError(f"indices overlap between splits: {sorted(overlap)}")


def build_split_metadata(
    train_pool_size: int,
    test_pool_size: int,
    train_size: int,
    val_size: int,
    test_size: int,
    seed: int,
    reserved_training_size: int = DEFAULT_RESERVED_TRAINING_SIZE,
) -> SplitMetadata:
    """Select stable, disjoint train/validation/test indices.

    Training is a prefix of a fixed-size reserved region. Validation begins
    after that region, so changing ``train_size`` does not silently change the
    validation examples. The default boundary matches the largest sanctioned
    training profile and the canonical diagnostic manifest.
    """
    if train_size > reserved_training_size:
        raise SplitSizeError(
            f"train_size {train_size} exceeds reserved_training_size "
            f"{reserved_training_size}; raise the shared reservation explicitly"
        )
    if reserved_training_size + val_size > train_pool_size:
        raise SplitSizeError(
            f"reserved training size and validation size "
            f"{[reserved_training_size, val_size]} exceed pool size {train_pool_size}"
        )

    reserved_training, val_indices = _sample_disjoint_subsets(
        train_pool_size, [reserved_training_size, val_size], seed
    )
    # Sample the smaller training prefix with the same seed rather than slicing
    # ``reserved_training``: the helper sorts returned IDs for stable persisted
    # metadata, so slicing that list would select numerically-small IDs instead
    # of the first IDs in the seeded shuffle.
    (train_indices,) = _sample_disjoint_subsets(train_pool_size, [train_size], seed)
    if not set(train_indices).issubset(reserved_training):
        raise AssertionError("training prefix escaped the reserved training region")
    assert_disjoint(train_indices, val_indices)
    (test_indices,) = _sample_disjoint_subsets(test_pool_size, [test_size], seed)
    return SplitMetadata(
        seed=seed,
        train_indices=train_indices,
        val_indices=val_indices,
        test_indices=test_indices,
    )


def build_diagnostic_manifest(
    train_pool_size: int,
    *,
    reserved_training_size: int = 1024,
    diagnostic_size: int = 200,
    seed: int = 42,
) -> DiagnosticManifest:
    """Build the canonical diagnostic IDs after a reserved training prefix.

    This deliberately does not accept a run profile. The first
    ``reserved_training_size`` shuffled IDs are held out for current/future
    training profiles; the following IDs form one stable diagnostic set.
    """
    if reserved_training_size + diagnostic_size > train_pool_size:
        raise SplitSizeError(
            f"requested reserved/diagnostic sizes {[reserved_training_size, diagnostic_size]} "
            f"exceed pool size {train_pool_size}"
        )
    rng = random.Random(seed)
    shuffled = list(range(train_pool_size))
    rng.shuffle(shuffled)
    reserved = shuffled[:reserved_training_size]
    # Preserve randomized order so a first-N smoke diagnostic remains a
    # representative deterministic prefix rather than the numerically lowest
    # dataset IDs. The full set still matches the longer profile's val set.
    diagnostic = shuffled[reserved_training_size : reserved_training_size + diagnostic_size]
    assert_disjoint(reserved, diagnostic)
    return DiagnosticManifest(
        version=1,
        source_dataset="openai/gsm8k",
        source_config="main",
        source_split="train",
        seed=seed,
        reserved_training_size=reserved_training_size,
        diagnostic_indices=diagnostic,
    )


def save_split_metadata(path: str | Path, metadata: SplitMetadata) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metadata.to_dict(), indent=2))


def load_split_metadata(path: str | Path) -> SplitMetadata:
    return SplitMetadata.from_dict(json.loads(Path(path).read_text()))


def save_diagnostic_manifest(path: str | Path, manifest: DiagnosticManifest) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest.to_dict(), indent=2) + "\n")


def load_diagnostic_manifest(path: str | Path) -> DiagnosticManifest:
    return DiagnosticManifest.from_dict(json.loads(Path(path).read_text()))


def select_split(dataset, indices: list[int]):
    """Apply previously-selected indices to an actually-loaded HF dataset."""
    return dataset.select(indices)
