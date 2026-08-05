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
    lazily allocated on first use or growth. Module warm-up has no CRS domain;
    qualified ``tmerc`` warm-up therefore includes both generic-TM native and
    forward-UTM accelerated variants.

    Examples
    --------
    >>> import vibeproj
    >>> vibeproj.warm_up(["tmerc", "webmerc"])  # selective
    >>> vibeproj.warm_up()                       # all projections
    """
    from vibeproj.transcendentals import (
        TranscendentalOperation,
        detect_device_capability,
        normalize_compute_precision,
        normalize_transcendental_policy,
        resolve_transcendental_strategy,
    )

    precision = normalize_compute_precision(precision)
    transcendentals = normalize_transcendental_policy(transcendentals)
    from vibeproj.fused_kernels import _SUPPORTED, compile_kernels

    targets = sorted(
        _SUPPORTED
        if projections is None
        else {
            (projection, direction)
            for projection in projections
            for direction in ("forward", "inverse")
            if (projection, direction) in _SUPPORTED
        }
    )
    device = detect_device_capability()
    projection_variants: set[tuple[str, str, str]] = set()
    for projection, direction in targets:
        if (projection, direction) == ("tmerc", "forward"):
            operation = TranscendentalOperation.TMERC_FORWARD
            domains = ("global", "utm")
        else:
            operation = TranscendentalOperation.PROJECTION
            domains = (f"{projection}.{direction}",)
        for domain in domains:
            implementation_id = resolve_transcendental_strategy(
                operation,
                transcendentals,
                device=device,
                domain=domain,
                precision=precision,
            ).implementation_id
            projection_variants.add((projection, direction, implementation_id))
    if projection_variants:
        compile_kernels(
            precision=precision,
            projection_variants=tuple(sorted(projection_variants)),
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
