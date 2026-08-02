"""Per-run output directory + resolved config/environment/split/tag persistence.

`RunTags` is the metadata cleanup.py keys off — run profile, hardware profile,
whether the run is purely for verification, and its final status. Deliberately
a separate small file (not buried in config.json) so cleanup logic has one
narrow, stable thing to read.
"""

import dataclasses
import datetime
import importlib.metadata
import json
import platform
import sys
from pathlib import Path
from typing import Protocol

from tiny_grpo.splits import SplitMetadata, save_split_metadata

PACKAGES_TO_PIN = ["torch", "transformers", "trl", "accelerate", "datasets", "peft"]

RunStatus = str  # "running" | "completed" | "failed"


class RunConfig(Protocol):
    """Metadata fields shared by the GRPO and SFT config dataclasses."""

    run_name: str
    hardware_profile_name: str


def make_run_dir(output_root: str | Path, run_name: str, now: datetime.datetime | None = None) -> Path:
    """Create a fresh, uniquely-named run directory under `output_root`.

    Never overwrites an existing run: if the timestamp-based name collides
    (e.g. two runs started in the same second), a numeric suffix is added
    until an unused path is found.
    """
    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=True)
    timestamp = (now or datetime.datetime.now()).strftime("%Y%m%d_%H%M%S")

    candidate = root / f"{run_name}_{timestamp}"
    suffix = 0
    while candidate.exists():
        suffix += 1
        candidate = root / f"{run_name}_{timestamp}_{suffix}"

    candidate.mkdir(parents=True)
    return candidate


def collect_environment_info() -> dict:
    versions = {}
    for package in PACKAGES_TO_PIN:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return {
        "python_version": sys.version,
        "platform": platform.platform(),
        "packages": versions,
    }


@dataclasses.dataclass(frozen=True)
class RunTags:
    run_profile: str
    hardware_profile: str
    verification_run: bool
    status: RunStatus = "running"

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "RunTags":
        return cls(
            run_profile=data["run_profile"],
            hardware_profile=data["hardware_profile"],
            verification_run=data["verification_run"],
            status=data.get("status", "running"),
        )


def _run_tags_path(run_dir: str | Path) -> Path:
    return Path(run_dir) / "run_tags.json"


def save_run_tags(run_dir: str | Path, tags: RunTags) -> None:
    _run_tags_path(run_dir).write_text(json.dumps(tags.to_dict(), indent=2))


def load_run_tags(run_dir: str | Path) -> RunTags:
    return RunTags.from_dict(json.loads(_run_tags_path(run_dir).read_text()))


def has_run_tags(run_dir: str | Path) -> bool:
    return _run_tags_path(run_dir).exists()


def update_run_status(run_dir: str | Path, status: RunStatus) -> None:
    tags = load_run_tags(run_dir)
    save_run_tags(run_dir, dataclasses.replace(tags, status=status))


def save_run_metadata(
    run_dir: str | Path,
    config: RunConfig,
    split_metadata: SplitMetadata,
    *,
    verification_run: bool,
) -> None:
    """Write config.json, environment.json, split_metadata.json, and the
    initial run_tags.json (status="running") into `run_dir`.
    """
    run_dir = Path(run_dir)
    (run_dir / "config.json").write_text(json.dumps(dataclasses.asdict(config), indent=2))
    (run_dir / "environment.json").write_text(json.dumps(collect_environment_info(), indent=2))
    save_split_metadata(run_dir / "split_metadata.json", split_metadata)
    save_run_tags(
        run_dir,
        RunTags(
            run_profile=config.run_name,
            hardware_profile=config.hardware_profile_name,
            verification_run=verification_run,
            status="running",
        ),
    )
