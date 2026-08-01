"""Process and device memory reporting.

No CUDA-only or MPS-only assumption is baked into a single call path — callers
must go through `device_memory_mb(device)`, dispatching on the active hardware
profile's device, so mps_16gb code never calls torch.cuda.* and cuda_4gb code
never calls torch.mps.*.
"""

import resource


def process_memory_mb() -> float:
    """Peak resident set size of this process, in MB. Universal fallback,
    recorded regardless of which hardware profile is active.

    macOS reports `ru_maxrss` in bytes (Linux reports it in KB) — branch on
    platform since cuda_4gb targets Linux/Windows while mps_16gb is macOS-only.
    """
    import platform

    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return usage / (1024 * 1024) if platform.system() == "Darwin" else usage / 1024


def mps_memory_mb() -> dict | None:
    """Current/driver MPS memory usage in MB, or None if unavailable.

    Only call this when the active device is "mps" — never on cuda_4gb.
    """
    try:
        import torch

        if not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
            return None
        if not (hasattr(torch.mps, "current_allocated_memory") and hasattr(torch.mps, "driver_allocated_memory")):
            return None
        return {
            "allocated_mb": torch.mps.current_allocated_memory() / (1024 * 1024),
            "driver_allocated_mb": torch.mps.driver_allocated_memory() / (1024 * 1024),
        }
    except ImportError:
        return None


def cuda_memory_mb() -> dict | None:
    """Current/max/reserved CUDA memory usage in MB, or None if unavailable.

    Only call this when the active device is "cuda" — never on mps_16gb.
    Reserved-but-unallocated memory is logged separately from live allocation,
    since fragmentation is a real failure mode at 4GB.
    """
    try:
        import torch

        if not torch.cuda.is_available():
            return None
        return {
            "allocated_mb": torch.cuda.memory_allocated() / (1024 * 1024),
            "max_allocated_mb": torch.cuda.max_memory_allocated() / (1024 * 1024),
            "reserved_mb": torch.cuda.memory_reserved() / (1024 * 1024),
        }
    except ImportError:
        return None


def device_memory_mb(device: str) -> dict | None:
    """Dispatch to the memory reporter matching `device` ("mps" or "cuda").

    This is the only function training/logging code should call for device
    memory — it guarantees the wrong device's API is never invoked.
    """
    if device == "mps":
        return mps_memory_mb()
    if device == "cuda":
        return cuda_memory_mb()
    return None
