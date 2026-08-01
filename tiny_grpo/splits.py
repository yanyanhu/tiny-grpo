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


class SplitOverlapError(ValueError):
    """Raised when train and validation indices are not disjoint."""


class SplitSizeError(ValueError):
    """Raised when the requested split sizes don't fit the available pool."""


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
) -> SplitMetadata:
    """Select disjoint train/val indices from the GSM8K *train* pool, and
    independent test indices from the GSM8K *test* pool, using a fixed seed.
    """
    train_indices, val_indices = _sample_disjoint_subsets(train_pool_size, [train_size, val_size], seed)
    assert_disjoint(train_indices, val_indices)
    (test_indices,) = _sample_disjoint_subsets(test_pool_size, [test_size], seed)
    return SplitMetadata(
        seed=seed,
        train_indices=train_indices,
        val_indices=val_indices,
        test_indices=test_indices,
    )


def save_split_metadata(path: str | Path, metadata: SplitMetadata) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metadata.to_dict(), indent=2))


def load_split_metadata(path: str | Path) -> SplitMetadata:
    return SplitMetadata.from_dict(json.loads(Path(path).read_text()))


def select_split(dataset, indices: list[int]):
    """Apply previously-selected indices to an actually-loaded HF dataset."""
    return dataset.select(indices)
