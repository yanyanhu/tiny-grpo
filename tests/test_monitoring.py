"""CPU-only unit tests for tiny_grpo.monitoring. No model/dataset access.

Runs on whatever device is actually present (or absent) on this machine —
does not require CUDA or MPS to be available; the "unavailable" path is
exercised naturally on a machine that lacks the device.
"""

from tiny_grpo.monitoring import cuda_memory_mb, device_memory_mb, mps_memory_mb, process_memory_mb


def test_process_memory_mb_is_positive():
    assert process_memory_mb() > 0


def test_mps_memory_mb_returns_expected_shape_or_none():
    result = mps_memory_mb()
    assert result is None or {"allocated_mb", "driver_allocated_mb"} <= result.keys()


def test_cuda_memory_mb_returns_expected_shape_or_none():
    result = cuda_memory_mb()
    assert result is None or {"allocated_mb", "max_allocated_mb", "reserved_mb"} <= result.keys()


def test_device_memory_mb_dispatches_to_mps():
    assert device_memory_mb("mps") == mps_memory_mb()


def test_device_memory_mb_dispatches_to_cuda():
    assert device_memory_mb("cuda") == cuda_memory_mb()


def test_device_memory_mb_unknown_device_returns_none():
    assert device_memory_mb("rocm") is None
