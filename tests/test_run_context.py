"""CPU-only unit tests for tiny_grpo.run_context. No model/dataset access."""

import datetime
import json

import pytest

from tiny_grpo.config import smoke_config
from tiny_grpo.hardware import MPS_16GB
from tiny_grpo.run_context import (
    RunTags,
    collect_environment_info,
    has_run_tags,
    load_run_tags,
    make_run_dir,
    save_run_metadata,
    save_run_tags,
    update_run_status,
)
from tiny_grpo.splits import build_split_metadata


def test_make_run_dir_creates_unique_named_directory(tmp_path):
    run_dir = make_run_dir(tmp_path, "smoke", now=datetime.datetime(2026, 1, 2, 3, 4, 5))
    assert run_dir.exists()
    assert run_dir.name == "smoke_20260102_030405"


def test_make_run_dir_never_overwrites_existing_run(tmp_path):
    fixed_time = datetime.datetime(2026, 1, 2, 3, 4, 5)
    first = make_run_dir(tmp_path, "smoke", now=fixed_time)
    (first / "marker.txt").write_text("first run")

    second = make_run_dir(tmp_path, "smoke", now=fixed_time)

    assert second != first
    assert (first / "marker.txt").read_text() == "first run"


def test_collect_environment_info_shape():
    info = collect_environment_info()
    assert "python_version" in info
    assert "platform" in info
    assert "torch" in info["packages"]
    assert "trl" in info["packages"]


def test_save_run_metadata_writes_all_files(tmp_path):
    config = smoke_config(MPS_16GB, output_dir=str(tmp_path))
    split_metadata = build_split_metadata(
        train_pool_size=100,
        test_pool_size=50,
        train_size=20,
        val_size=10,
        test_size=8,
        seed=42,
        reserved_training_size=40,
    )

    save_run_metadata(tmp_path, config, split_metadata, verification_run=True)

    config_data = json.loads((tmp_path / "config.json").read_text())
    assert config_data["run_name"] == "smoke"
    assert config_data["hardware_profile_name"] == "mps_16gb"
    assert config_data["dataset"]["train_size"] == 64

    env_data = json.loads((tmp_path / "environment.json").read_text())
    assert "packages" in env_data

    split_data = json.loads((tmp_path / "split_metadata.json").read_text())
    assert split_data["seed"] == 42
    assert len(split_data["train_indices"]) == 20

    tags = load_run_tags(tmp_path)
    assert tags == RunTags(
        run_profile="smoke", hardware_profile="mps_16gb", verification_run=True, status="running"
    )


def test_has_run_tags(tmp_path):
    assert has_run_tags(tmp_path) is False
    save_run_tags(tmp_path, RunTags(run_profile="debug", hardware_profile="cuda_4gb", verification_run=False))
    assert has_run_tags(tmp_path) is True


def test_update_run_status_roundtrip(tmp_path):
    save_run_tags(
        tmp_path, RunTags(run_profile="smoke", hardware_profile="mps_16gb", verification_run=True, status="running")
    )

    update_run_status(tmp_path, "completed")

    assert load_run_tags(tmp_path).status == "completed"


def test_run_tags_default_status_is_running():
    tags = RunTags(run_profile="smoke", hardware_profile="mps_16gb", verification_run=True)
    assert tags.status == "running"


def test_load_run_tags_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_run_tags(tmp_path)
