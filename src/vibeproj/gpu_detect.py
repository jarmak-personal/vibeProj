"""GPU type detection for automatic precision selection.

Queries the NVIDIA driver to determine fp64:fp32 throughput ratio.
Projection arithmetic remains fp64. Validated Ada consumer GPUs use bounded
INT64 trig inside Helmert and guarded forward-UTM transcendentals; datacenter
and unknown GPUs use native paired fp64 sine/cosine.
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
    kernels can still select validated bounded transcendental strategies for
    the current device without exposing a lower-precision public mode.
    """
    return "fp64"


def _supports_fixed_int64_trig(major: int, minor: int, fp32_to_fp64_ratio: int) -> bool:
    """Whether a GPU is validated for automatic fixed-point trig.

    The first production rollout is deliberately limited to Ada sm_89 consumer
    GPUs, where the implementation has been accuracy- and performance-tested.
    Explicit internal dispatch can still exercise the kernel elsewhere.
    """
    return (major, minor) == (8, 9) and fp32_to_fp64_ratio >= 16


@lru_cache(maxsize=None)
def _select_helmert_trig_mode_for_device(device_id: int) -> str:
    try:
        import cupy as cp

        properties = cp.cuda.runtime.getDeviceProperties(device_id)
        device = cp.cuda.Device(device_id)
        ratio = int(device.attributes.get("SingleToDoublePrecisionPerfRatio", 0))
        if _supports_fixed_int64_trig(
            int(properties.get("major", 0)), int(properties.get("minor", 0)), ratio
        ):
            return "int64"
    except (ImportError, RuntimeError, OSError):
        pass
    return "fp64"


def select_helmert_trig_mode() -> str:
    """Select the Helmert CUDA trig implementation for the current GPU.

    Returns ``"int64"`` only for validated Ada consumer GPUs. Unknown and
    datacenter GPUs conservatively use paired native fp64 sine/cosine.
    """
    try:
        import cupy as cp

        device_id = int(cp.cuda.runtime.getDevice())
    except (ImportError, RuntimeError, OSError):
        return "fp64"
    return _select_helmert_trig_mode_for_device(device_id)


def select_tmerc_forward_mode() -> str:
    """Select the guarded forward-UTM transcendental implementation.

    The forward-TM and Helmert fixed-point paths share the same validated GPU
    gate. H100, unknown, and future architectures remain on native fp64 until
    they have independent performance and accuracy measurements.
    """
    return select_helmert_trig_mode()
