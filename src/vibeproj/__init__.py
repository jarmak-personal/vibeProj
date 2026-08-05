"""vibeProj — GPU-accelerated coordinate projection library."""

__version__ = "1.0.5"

from vibeproj.crs import CRSInput
from vibeproj.exceptions import (
    CoordinateValidationError,
    CRSResolutionError,
    UnsupportedProjectionError,
    VibeProjectionError,
)
from vibeproj.transformer import Transformer
from vibeproj.transcendentals import (
    AccuracyContract,
    DeviceCapability,
    StrategyDecision,
    StrategyExplanation,
    StrategyImplementation,
    TranscendentalOperation,
    TranscendentalPolicy,
    list_transcendental_strategies,
)


def list_projections() -> dict[str, dict]:
    """Return supported projections and their metadata.

    Returns
    -------
    dict[str, dict]
        Keys are internal projection names. Each value has:
        - "methods": list of pyproj method names that map to this projection
        - "fused": True if a GPU-accelerated fused kernel is available
    """
    from vibeproj.crs import _METHOD_MAP
    from vibeproj.fused_kernels import _SUPPORTED
    from vibeproj.projections import PROJECTION_REGISTRY

    inverse_map: dict[str, list[str]] = {}
    for method, name in _METHOD_MAP.items():
        inverse_map.setdefault(name, []).append(method)

    result = {}
    for name in sorted(PROJECTION_REGISTRY):
        result[name] = {
            "methods": sorted(inverse_map.get(name, [])),
            "fused": (name, "forward") in _SUPPORTED,
        }
    return result


def warm_up(
    projections: list[str] | None = None,
    *,
    precision: str = "auto",
    transcendentals: TranscendentalPolicy = "auto",
) -> None:
    """Pre-compile fused NVRTC kernels to eliminate first-call latency.

    Parameters
    ----------
    projections : list of str, optional
        Projection names to compile (e.g. ["tmerc", "webmerc"]).
        If None, compiles all supported projections.
    precision : {"auto", "fp64", "fp32", "ds"}
        Compute precision: ``"auto"``, ``"fp64"``, ``"fp32"``, or ``"ds"``.
    transcendentals : {"auto", "native", "accelerated"}
        Hardware-aware transcendental implementation policy, independent of
        compute precision. Warm-up has no concrete workload size, so ``"auto"``
        selects an otherwise-qualified implementation without applying runtime
        crossover thresholds.

    Notes
    -----
    This compiles fused projection kernels. Per-Transformer correction scratch,
    pinned staging buffers, device buffers, and persistent chunk streams remain
    lazily allocated on first use or growth.

    Examples
    --------
    >>> import vibeproj
    >>> vibeproj.warm_up(["tmerc", "webmerc"])  # selective
    >>> vibeproj.warm_up()                       # all projections
    """
    from vibeproj.transcendentals import (
        NATIVE_LIBDEVICE,
        TranscendentalOperation,
        detect_device_capability,
        normalize_compute_precision,
        normalize_transcendental_policy,
        resolve_transcendental_strategy,
    )

    precision = normalize_compute_precision(precision)
    transcendentals = normalize_transcendental_policy(transcendentals)
    from vibeproj.fused_kernels import compile_kernels

    transcendental_impl = NATIVE_LIBDEVICE
    if projections is None or "tmerc" in projections:
        transcendental_impl = resolve_transcendental_strategy(
            TranscendentalOperation.TMERC_FORWARD,
            transcendentals,
            device=detect_device_capability(),
            domain="utm",
            precision=precision,
        ).implementation_id
    compile_kernels(
        projections,
        precision=precision,
        transcendental_impl=transcendental_impl,
    )


__all__ = [
    "CRSInput",
    "AccuracyContract",
    "DeviceCapability",
    "StrategyDecision",
    "StrategyExplanation",
    "StrategyImplementation",
    "Transformer",
    "TranscendentalOperation",
    "TranscendentalPolicy",
    "list_projections",
    "list_transcendental_strategies",
    "warm_up",
    "VibeProjectionError",
    "UnsupportedProjectionError",
    "CRSResolutionError",
    "CoordinateValidationError",
    "__version__",
]
