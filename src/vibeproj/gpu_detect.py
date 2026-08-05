"""GPU type detection for automatic precision selection.

Queries the NVIDIA driver to determine fp64:fp32 throughput ratio.
Projection arithmetic remains fp64. The central registry can select bounded
transcendental implementations for qualified Helmert, UTM, and projection
domains on validated Ada consumer GPUs. Datacenter and unknown GPUs retain
native fp64 until independently qualified.
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

    Always returns ``"fp64"`` for arithmetic and I/O. Individual fused
    kernels can still select any registry-qualified bounded transcendental
    strategy for the current device and exact operation domain without
    exposing a lower-precision public mode.
    """
    return "fp64"


def _supports_fixed_int64_trig(major: int, minor: int, fp32_to_fp64_ratio: int) -> bool:
    """Compatibility predicate delegated to the central strategy resolver."""
    from vibeproj.transcendentals import (
        HELMERT_FIXED_Q62,
        DeviceCapability,
        TranscendentalOperation,
        resolve_transcendental_strategy,
    )

    device = DeviceCapability(
        backend="cuda",
        compute_capability=(major, minor),
        fp32_to_fp64_ratio=fp32_to_fp64_ratio,
    )
    decision = resolve_transcendental_strategy(
        TranscendentalOperation.HELMERT, "auto", device=device
    )
    return decision.implementation_id == HELMERT_FIXED_Q62


def select_helmert_trig_mode() -> str:
    """Return the legacy name for the centrally resolved Helmert strategy."""
    from vibeproj.transcendentals import (
        HELMERT_FIXED_Q62,
        TranscendentalOperation,
        resolve_transcendental_strategy,
    )

    decision = resolve_transcendental_strategy(TranscendentalOperation.HELMERT)
    return "int64" if decision.implementation_id == HELMERT_FIXED_Q62 else "fp64"


def select_tmerc_forward_mode() -> str:
    """Return the legacy name for the centrally resolved forward-UTM strategy."""
    from vibeproj.transcendentals import (
        TMERC_FIXED_Q62,
        TranscendentalOperation,
        resolve_transcendental_strategy,
    )

    decision = resolve_transcendental_strategy(TranscendentalOperation.TMERC_FORWARD, domain="utm")
    return "int64" if decision.implementation_id == TMERC_FIXED_Q62 else "fp64"
