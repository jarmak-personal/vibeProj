"""GPU type detection for automatic precision selection.

Queries the NVIDIA driver to determine fp64:fp32 throughput ratio.
Consumer GPUs (RTX series): 1:64 — use double-single fp32 arithmetic.
Datacenter GPUs (A100, H100): 1:2 — use native fp64.
"""

from __future__ import annotations

from functools import lru_cache


@lru_cache(maxsize=1)
def get_fp64_ratio() -> float:
    """Return the fp64:fp32 throughput ratio for the current GPU.

    Returns 1.0 if no GPU is available (CPU mode — always use fp64).
    Returns ratio >= 0.25 for datacenter GPUs (use native fp64).
    Returns ratio < 0.25 for consumer GPUs (use compensated fp32).
    """
    try:
        import cupy as cp

        dev = cp.cuda.Device(0)
        ratio_raw = dev.attributes.get("SingleToDoublePrecisionPerfRatio", 0)
        if ratio_raw > 0:
            return 1.0 / float(ratio_raw)
        return 1.0 / 32.0  # conservative fallback
    except (ImportError, RuntimeError, OSError):
        return 1.0  # CPU mode


@lru_cache(maxsize=1)
def favors_native_fp64() -> bool:
    """True if fp64 is fast enough to use directly (datacenter GPU or CPU)."""
    return get_fp64_ratio() >= 0.25


def select_compute_precision() -> str:
    """Select compute precision based on GPU type.

    Always returns "fp64" — projection math is dominated by transcendental
    functions (sin, cos, atan2, asinh) which use the SFU. The fp64:fp32
    ratio for SFU ops is ~4x (not 64x like ALU ops), so the theoretical
    64x fp32 throughput advantage doesn't materialize for projections.

    On the RTX 4090 (1:64 ALU ratio), fp64 TM still runs at 0.49ms/1M points
    = 183x faster than CPU. The ds path exists for experimentation but provides
    no speedup for trig-heavy projection kernels.
    """
    return "fp64"
